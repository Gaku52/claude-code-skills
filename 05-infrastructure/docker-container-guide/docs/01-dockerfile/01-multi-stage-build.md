# マルチステージビルド

> ビルダーパターンを活用してイメージサイズを大幅に削減し、セキュリティと効率を両立させる実践ガイド。Node.js、Go、Rust の言語別例を含む。

---

## この章で学ぶこと

1. **マルチステージビルドの仕組み**を理解し、ビルド環境と実行環境を分離できる
2. **言語別の最適なビルダーパターン**を実装し、最小サイズのイメージを構築できる
3. **キャッシュ戦略と中間ステージの活用**で、ビルド速度とイメージ品質を最適化できる
4. **CI/CD パイプラインとの統合**でテスト・リント・セキュリティスキャンをビルドに組み込める

---

## 1. マルチステージビルドとは

### 1.1 問題: シングルステージビルドの課題

```
+------------------------------------------------------+
|         シングルステージビルド（従来型）                  |
|                                                      |
|  FROM node:20                                        |
|  +------------------------------------------------+ |
|  |  Node.js ランタイム         ~300 MB             | |
|  |  npm / yarn                 ~50 MB              | |
|  |  ビルドツール (gcc等)        ~200 MB             | |
|  |  node_modules (dev含む)     ~400 MB             | |
|  |  ソースコード                ~10 MB              | |
|  |  ビルド成果物               ~5 MB               | |
|  +------------------------------------------------+ |
|  合計: ~965 MB  <- ビルドツールが実行時に不要           |
|                                                      |
|         マルチステージビルド                            |
|                                                      |
|  Stage 1: ビルド             Stage 2: 実行            |
|  +--------------------+     +--------------------+  |
|  | Node.js + npm      |     | Node.js (Alpine)   |  |
|  | ビルドツール        |     | 本番 node_modules  |  |
|  | 全 node_modules    | --> | ビルド成果物        |  |
|  | ソースコード        |COPY | (必要なものだけ)    |  |
|  +--------------------+     +--------------------+  |
|  ~965 MB (破棄)              ~150 MB (最終イメージ)  |
+------------------------------------------------------+
```

シングルステージビルドでは、ビルドに必要なコンパイラ、リンカ、開発用ライブラリ、テストフレームワークがすべて最終イメージに含まれてしまう。これにより以下の問題が発生する:

- **イメージサイズの肥大化**: 不要なツールが数百MBを占有する
- **セキュリティリスクの増大**: 攻撃対象面（アタックサーフェス）が広がる。ビルドツールに脆弱性があれば本番環境にも影響する
- **ダウンロード時間の増加**: デプロイ時のイメージプル時間が長くなる
- **ストレージコストの増大**: レジストリの保存容量とデータ転送量が増える

マルチステージビルドはこれらの問題を、1つの Dockerfile 内で複数のビルドステージを定義し、最終ステージに必要なファイルだけをコピーすることで解決する。

### 1.2 基本構文

```dockerfile
# ステージ 1: ビルド
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# ステージ 2: 実行（最終イメージ）
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
EXPOSE 3000
CMD ["node", "dist/server.js"]
```

```bash
# ビルド（最終ステージのみがイメージに含まれる）
docker build -t my-app:v1.0.0 .

# 特定のステージまでビルド
docker build --target builder -t my-app-builder .

# ビルド進捗の詳細表示
DOCKER_BUILDKIT=1 docker build --progress=plain -t my-app:v1.0.0 .
```

### 1.3 COPY --from の仕組み

```
+------------------------------------------------------+
|          COPY --from の動作原理                        |
|                                                      |
|  COPY --from=builder /app/dist ./dist                |
|                |          |          |               |
|                |          |          +-- 現在のステージの|
|                |          |              コピー先       |
|                |          +-- ソースステージのコピー元   |
|                +-- ステージ名（AS で指定した名前）       |
|                                                      |
|  他の指定方法:                                        |
|  COPY --from=0 ...    # ステージ番号（0始まり）        |
|  COPY --from=nginx:alpine ...  # 外部イメージ         |
+------------------------------------------------------+
```

---

## 2. 言語別マルチステージビルド

### 2.1 Node.js (Express + TypeScript)

```dockerfile
# === ステージ 1: 依存関係インストール ===
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# === ステージ 2: ビルド ===
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build
# dist/ ディレクトリにコンパイル済みJSが生成される

# === ステージ 3: 本番用依存関係 ===
FROM node:20-alpine AS prod-deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production && npm cache clean --force

# === ステージ 4: 実行 ===
FROM node:20-alpine
RUN addgroup -S app && adduser -S app -G app

# PID 1 問題の解決
RUN apk add --no-cache dumb-init

WORKDIR /app

COPY --from=prod-deps --chown=app:app /app/node_modules ./node_modules
COPY --from=builder --chown=app:app /app/dist ./dist
COPY --chown=app:app package.json ./

USER app
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]
```

#### Node.js における依存関係分離の詳細

```
+------------------------------------------------------+
|     Node.js の4ステージ構成の理由                      |
|                                                      |
|  deps (全依存関係)                                    |
|  └── devDependencies を含む（TypeScript コンパイラ等）  |
|      ↓                                               |
|  builder (ビルド)                                     |
|  └── deps の node_modules を使って TypeScript を       |
|      JavaScript にコンパイル                           |
|      ↓                                               |
|  prod-deps (本番依存関係)                              |
|  └── devDependencies を除外した node_modules を作成    |
|      ↓                                               |
|  runner (実行)                                        |
|  └── prod-deps の node_modules + builder の dist のみ |
|                                                      |
|  なぜ deps と prod-deps を分けるか:                    |
|  npm ci --only=production を builder でやると          |
|  ソースコード変更のたびに再実行されてしまう。              |
|  別ステージにすることでキャッシュが効く。                  |
+------------------------------------------------------+
```

### 2.2 Go

```dockerfile
# === ステージ 1: ビルド ===
FROM golang:1.22-alpine AS builder

# セキュリティ: 証明書と非rootユーザーを事前準備
RUN apk add --no-cache ca-certificates tzdata
RUN adduser -D -g '' appuser

WORKDIR /app

# 依存関係を先にダウンロード（キャッシュ効率化）
COPY go.mod go.sum ./
RUN go mod download && go mod verify

# ソースコードをコピーしてビルド
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-w -s" -o /server ./cmd/server

# === ステージ 2: 実行（scratch = 空のベースイメージ） ===
FROM scratch

# ビルドステージから必要なファイルのみコピー
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /server /server

USER appuser
EXPOSE 8080
ENTRYPOINT ["/server"]
```

```bash
# ビルドとサイズ確認
docker build -t go-app .
docker images go-app
# REPOSITORY   TAG       IMAGE ID       CREATED          SIZE
# go-app       latest    abc123         10 seconds ago   12.3MB
# <- Go バイナリ + 証明書のみ。OS すらない。
```

#### Go のクロスコンパイル対応

```dockerfile
# マルチプラットフォーム対応の Go ビルド
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS builder

ARG TARGETOS
ARG TARGETARCH

RUN apk add --no-cache ca-certificates tzdata git

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# クロスコンパイル（ビルドマシンのアーキテクチャに依存しない）
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build \
        -ldflags="-w -s -X main.version=$(git describe --tags 2>/dev/null || echo dev)" \
        -o /server \
        ./cmd/server

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

```bash
# マルチプラットフォームビルド
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t ghcr.io/myorg/go-app:v1.0.0 \
    --push .
```

### 2.3 Rust

```dockerfile
# === ステージ 1: 依存関係ビルド（キャッシュ用） ===
FROM rust:1.75-alpine AS chef
RUN apk add --no-cache musl-dev
RUN cargo install cargo-chef
WORKDIR /app

# === ステージ 2: レシピ生成 ===
FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# === ステージ 3: 依存関係ビルド ===
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
# 依存関係のみビルド（ソースコード変更時にキャッシュが効く）
RUN cargo chef cook --release --recipe-path recipe.json

# アプリケーションビルド
COPY . .
RUN cargo build --release

# === ステージ 4: 実行 ===
FROM alpine:3.19
RUN apk add --no-cache ca-certificates
RUN addgroup -S app && adduser -S app -G app

COPY --from=builder /app/target/release/myapp /usr/local/bin/

USER app
EXPOSE 8080
CMD ["myapp"]
```

#### Rust の静的リンクで scratch を使う

```dockerfile
FROM rust:1.75-alpine AS builder
RUN apk add --no-cache musl-dev

WORKDIR /app

# ターゲットの追加
RUN rustup target add x86_64-unknown-linux-musl

COPY Cargo.toml Cargo.lock ./
# ダミービルドで依存関係のみコンパイル
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release --target x86_64-unknown-linux-musl
RUN rm -rf src

# 実際のソースでビルド
COPY src ./src
RUN touch src/main.rs
RUN RUSTFLAGS="-C target-feature=+crt-static" \
    cargo build --release --target x86_64-unknown-linux-musl

FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/myapp /myapp
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

EXPOSE 8080
ENTRYPOINT ["/myapp"]
```

### 2.4 Next.js (スタンドアロン出力)

```dockerfile
# === ステージ 1: 依存関係 ===
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# === ステージ 2: ビルド ===
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next.js のスタンドアロン出力を有効化
# next.config.js に output: 'standalone' が必要
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# === ステージ 3: 実行 ===
FROM node:20-alpine
WORKDIR /app

RUN addgroup -S app && adduser -S app -G app

# スタンドアロン出力のみコピー（node_modules の最小サブセット含む）
COPY --from=builder --chown=app:app /app/.next/standalone ./
COPY --from=builder --chown=app:app /app/.next/static ./.next/static
COPY --from=builder --chown=app:app /app/public ./public

USER app
EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
ENV NEXT_TELEMETRY_DISABLED=1
CMD ["node", "server.js"]
```

#### next.config.js の設定

```javascript
// next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',  // スタンドアロン出力を有効化
  // 必要に応じて追加設定
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: '**.example.com' },
    ],
  },
}

module.exports = nextConfig
```

### 2.5 Java (Spring Boot)

```dockerfile
# === ステージ 1: ビルド ===
FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /app

COPY gradlew build.gradle.kts settings.gradle.kts ./
COPY gradle ./gradle
RUN ./gradlew dependencies --no-daemon

COPY src ./src
RUN ./gradlew bootJar --no-daemon

# レイヤードJAR展開（Spring Boot 3.x）
RUN java -Djarmode=layertools -jar build/libs/*.jar extract --destination extracted

# === ステージ 2: 実行 ===
FROM eclipse-temurin:21-jre-alpine
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app

# レイヤー順にコピー（変更頻度: 低 -> 高）
COPY --from=builder /app/extracted/dependencies/ ./
COPY --from=builder /app/extracted/spring-boot-loader/ ./
COPY --from=builder /app/extracted/snapshot-dependencies/ ./
COPY --from=builder /app/extracted/application/ ./

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

ENTRYPOINT ["java", "org.springframework.boot.loader.launch.JarLauncher"]
```

#### Spring Boot のレイヤード JAR

```
+------------------------------------------------------+
|     Spring Boot レイヤード JAR の構造                   |
|                                                      |
|  dependencies/                変更頻度: 最低           |
|  └── BOOT-INF/lib/*.jar     (サードパーティ依存)       |
|                                                      |
|  spring-boot-loader/          変更頻度: 低             |
|  └── org/springframework/    (Boot ローダー)          |
|                                                      |
|  snapshot-dependencies/       変更頻度: 中             |
|  └── BOOT-INF/lib/*-SNAPSHOT.jar                     |
|                                                      |
|  application/                 変更頻度: 高             |
|  └── BOOT-INF/classes/       (アプリケーションコード)   |
|      META-INF/                                       |
|                                                      |
|  Docker のレイヤーキャッシュにより、依存関係が            |
|  変わらなければ再ダウンロード不要で高速ビルド              |
+------------------------------------------------------+
```

### 2.6 Python (FastAPI / Django)

```dockerfile
# === ステージ 1: ビルド ===
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ビルド依存関係
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python パッケージのインストール（prefix で分離）
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# === ステージ 2: 実行 ===
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ランタイム依存関係のみ（gcc は不要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# ビルドステージからインストール済みパッケージをコピー
COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "app.main:app"]
```

### 2.7 PHP (Laravel)

```dockerfile
# === ステージ 1: Composer 依存関係 ===
FROM composer:2 AS vendor
WORKDIR /app
COPY composer.json composer.lock ./
RUN composer install \
    --no-dev \
    --no-interaction \
    --no-scripts \
    --ignore-platform-reqs \
    --prefer-dist

# === ステージ 2: フロントエンドビルド ===
FROM node:20-alpine AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY resources ./resources
COPY vite.config.js ./
RUN npm run build

# === ステージ 3: 実行 ===
FROM php:8.3-fpm-alpine

# PHP 拡張
RUN docker-php-ext-install pdo pdo_mysql opcache

# Composer の vendor ディレクトリをコピー
COPY --from=vendor /app/vendor ./vendor

# フロントエンドビルド成果物をコピー
COPY --from=frontend /app/public/build ./public/build

# アプリケーションコード
COPY . .

# PHP-FPM 設定
COPY docker/php/php.ini /usr/local/etc/php/php.ini

RUN chown -R www-data:www-data /app/storage /app/bootstrap/cache

USER www-data
EXPOSE 9000
CMD ["php-fpm"]
```

---

## 3. イメージサイズの比較

### 比較表 1: シングル vs マルチステージ

| アプリ | シングルステージ | マルチステージ | 削減率 |
|---|---|---|---|
| Node.js (Express) | ~950 MB | ~150 MB | 84% |
| Go (Web API) | ~800 MB | ~12 MB | 98% |
| Rust (Web API) | ~1.5 GB | ~15 MB | 99% |
| Java (Spring Boot) | ~600 MB | ~200 MB | 67% |
| Next.js (SSR) | ~1.2 GB | ~120 MB | 90% |
| Python (FastAPI) | ~900 MB | ~180 MB | 80% |
| PHP (Laravel) | ~700 MB | ~250 MB | 64% |

### 比較表 2: ベースイメージ別サイズ

| ベースイメージ | サイズ | 用途 | パッケージマネージャ |
|---|---|---|---|
| `ubuntu:22.04` | ~77 MB | 汎用開発 | apt |
| `debian:bookworm-slim` | ~74 MB | 汎用（slim版） | apt |
| `alpine:3.19` | ~7 MB | 最小Linux | apk |
| `node:20` | ~1.1 GB | Node.js 開発 | apt |
| `node:20-slim` | ~200 MB | Node.js 本番 | apt |
| `node:20-alpine` | ~130 MB | Node.js 最小 | apk |
| `gcr.io/distroless/nodejs20` | ~120 MB | Node.js 最小(Distroless) | なし |
| `python:3.12` | ~1.0 GB | Python 開発 | apt |
| `python:3.12-slim` | ~130 MB | Python 本番 | apt |
| `scratch` | 0 B | 静的バイナリ専用 | なし |

### 比較表 3: ベースイメージの特性比較

| 特性 | scratch | distroless | alpine | slim | full |
|---|---|---|---|---|---|
| シェル | なし | なし* | ash | bash | bash |
| パッケージMgr | なし | なし | apk | apt | apt |
| libc | なし | glibc | musl | glibc | glibc |
| デバッグ | 不可 | :debug タグ | 可能 | 可能 | 可能 |
| 攻撃面 | 最小 | 極小 | 小 | 中 | 大 |
| サイズ | 0 MB | 20-120 MB | 7 MB | 70-130 MB | 300+ MB |

\* distroless の `:debug` バリアントには busybox シェルが含まれる

---

## 4. 高度なテクニック

### 4.1 外部イメージからのコピー

```dockerfile
# 他のイメージからファイルをコピー
FROM alpine:3.19
COPY --from=nginx:alpine /etc/nginx/nginx.conf /etc/nginx/
COPY --from=busybox:uclibc /bin/wget /usr/local/bin/

# 特定のバイナリツールだけを持ってくる
FROM alpine:3.19
COPY --from=docker:24-cli /usr/local/bin/docker /usr/local/bin/
COPY --from=docker/compose:v2.24.0 /usr/local/bin/docker-compose /usr/local/bin/
```

### 4.2 テストステージの組み込み

```dockerfile
# === ビルドステージ ===
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o /server ./cmd/server

# === テストステージ ===
FROM builder AS tester
RUN go test -v ./...
RUN go vet ./...

# === リントステージ ===
FROM builder AS linter
RUN go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
RUN golangci-lint run

# === 実行ステージ ===
FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

```bash
# テストのみ実行
docker build --target tester .

# リントのみ実行
docker build --target linter .

# 全ステージを実行（テスト -> リント -> ビルド -> 最終イメージ）
docker build .

# テストが通らないと最終ステージもビルドされない
# （CI/CD で活用）
```

#### Node.js でのテスト統合

```dockerfile
# === 依存関係 ===
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# === リント ===
FROM deps AS linter
COPY . .
RUN npm run lint
RUN npm run type-check

# === テスト ===
FROM deps AS tester
COPY . .
RUN npm run test -- --coverage

# === ビルド ===
FROM deps AS builder
COPY . .
RUN npm run build

# === 本番 ===
FROM node:20-alpine AS production
WORKDIR /app
RUN addgroup -S app && adduser -S app -G app

COPY --from=builder --chown=app:app /app/dist ./dist
COPY package.json package-lock.json ./
RUN npm ci --only=production && npm cache clean --force

USER app
CMD ["node", "dist/server.js"]
```

```bash
# CI/CD での段階的実行
docker build --target linter -t lint-check .     # リントのみ
docker build --target tester -t test-check .     # テストのみ
docker build --target production -t my-app .      # 本番ビルド
```

### 4.3 BuildKit のマウントキャッシュ

```dockerfile
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./

# パッケージキャッシュをマウント（ビルド間で再利用）
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o /server ./cmd/server

FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

```dockerfile
# Node.js の npm キャッシュ
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci
COPY . .
RUN npm run build
```

```dockerfile
# Python の pip キャッシュ
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /install /usr/local
COPY . .
CMD ["python", "app.py"]
```

```dockerfile
# Rust の cargo キャッシュ
FROM rust:1.75-alpine AS builder
RUN apk add --no-cache musl-dev
WORKDIR /app
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    cp target/release/myapp /usr/local/bin/

FROM alpine:3.19
COPY --from=builder /usr/local/bin/myapp /usr/local/bin/
CMD ["myapp"]
```

### 4.4 シークレットのマウント

```dockerfile
# ビルド時のシークレット（イメージに残らない）
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./

# プライベートレジストリの認証情報をマウント
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc \
    npm ci

COPY . .
CMD ["node", "server.js"]
```

```bash
# シークレットを指定してビルド
docker build --secret id=npmrc,src=.npmrc -t my-app .

# 複数のシークレット
docker build \
    --secret id=npmrc,src=.npmrc \
    --secret id=aws,src=$HOME/.aws/credentials \
    -t my-app .
```

### 4.5 SSH マウント

```dockerfile
# SSH 鍵を使ったプライベートリポジトリのクローン
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache git openssh-client

# SSH known_hosts の設定
RUN mkdir -p -m 0700 ~/.ssh && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=ssh go mod download

COPY . .
RUN go build -o /server .

FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

```bash
# SSH エージェントを使ってビルド
docker build --ssh default -t my-app .

# 特定の SSH 鍵を指定
docker build --ssh default=$HOME/.ssh/id_rsa -t my-app .
```

---

## 5. ステージ構成パターン

```
+------------------------------------------------------+
|          マルチステージ構成パターン                      |
|                                                      |
|  パターン 1: シンプル（2ステージ）                      |
|  [builder] --COPY--> [runner]                        |
|                                                      |
|  パターン 2: 依存関係分離（3ステージ）                   |
|  [deps] --COPY--> [builder] --COPY--> [runner]       |
|                                                      |
|  パターン 3: テスト統合（4ステージ）                     |
|  [deps] --> [builder] --> [tester]                   |
|                 |                                    |
|                 +--COPY--> [runner]                   |
|                                                      |
|  パターン 4: 開発/本番分岐                              |
|  [base] --> [dev]  (ホットリロード、デバッグツール)      |
|         --> [builder] --> [prod] (最小構成)            |
|                                                      |
|  パターン 5: 並列ビルド                                |
|  [api-builder]   --+                                 |
|  [worker-builder] -+--COPY--> [runner]               |
|  [frontend]      --+                                 |
+------------------------------------------------------+
```

### 開発/本番分岐の例

```dockerfile
# === 共通ベース ===
FROM node:20-alpine AS base
WORKDIR /app
COPY package.json package-lock.json ./

# === 開発環境 ===
FROM base AS development
RUN npm install  # devDependencies も含む
COPY . .
# 開発ツール
RUN apk add --no-cache git curl
EXPOSE 3000
CMD ["npm", "run", "dev"]

# === ビルド ===
FROM base AS builder
RUN npm ci
COPY . .
RUN npm run build
RUN npm run test

# === 本番環境 ===
FROM node:20-alpine AS production
WORKDIR /app
RUN addgroup -S app && adduser -S app -G app
RUN apk add --no-cache dumb-init

COPY --from=builder --chown=app:app /app/dist ./dist
COPY package.json package-lock.json ./
RUN npm ci --only=production && npm cache clean --force

USER app
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]
```

```bash
# 開発環境でビルド
docker build --target development -t my-app:dev .

# 本番環境でビルド
docker build --target production -t my-app:prod .

# docker-compose.yml での使い分け
```

```yaml
# docker-compose.yml (開発用)
services:
  app:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/node_modules
    ports:
      - "3000:3000"
    command: npm run dev
```

```yaml
# docker-compose.prod.yml (本番用)
services:
  app:
    build:
      context: .
      target: production
    ports:
      - "3000:3000"
    restart: unless-stopped
```

### 並列ビルドパターン

```dockerfile
# BuildKit は依存関係のないステージを自動的に並列ビルドする

# === API ビルド ===
FROM golang:1.22-alpine AS api-builder
WORKDIR /app/api
COPY api/ .
RUN go build -o /api-server .

# === ワーカービルド（API と並列で実行される） ===
FROM golang:1.22-alpine AS worker-builder
WORKDIR /app/worker
COPY worker/ .
RUN go build -o /worker .

# === フロントエンドビルド（上記と並列で実行される） ===
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# === 最終イメージ ===
FROM alpine:3.19
RUN apk add --no-cache ca-certificates

COPY --from=api-builder /api-server /usr/local/bin/
COPY --from=worker-builder /worker /usr/local/bin/
COPY --from=frontend-builder /app/frontend/dist /var/www/html/

EXPOSE 8080
CMD ["api-server"]
```

---

## 6. CI/CD との統合

### 6.1 GitHub Actions でのマルチステージ活用

```yaml
# .github/workflows/docker.yml
name: Docker Build and Test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          target: linter
          cache-from: type=gha
          cache-to: type=gha,mode=max

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/build-push-action@v5
        with:
          context: .
          target: tester
          cache-from: type=gha
          cache-to: type=gha,mode=max

  build:
    needs: [lint, test]
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
          context: .
          target: production
          push: ${{ github.event_name != 'pull_request' }}
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 6.2 キャッシュ戦略

```
+------------------------------------------------------+
|          CI/CD でのキャッシュ戦略                       |
|                                                      |
|  1. GitHub Actions Cache (GHA)                       |
|     cache-from: type=gha                             |
|     cache-to: type=gha,mode=max                      |
|     -> GitHub の Cache API を利用                     |
|     -> 同一ブランチ + デフォルトブランチのキャッシュ共有  |
|                                                      |
|  2. レジストリキャッシュ                               |
|     cache-from: type=registry,ref=img:cache           |
|     cache-to: type=registry,ref=img:cache,mode=max    |
|     -> レジストリにキャッシュレイヤーを保存              |
|     -> 異なる CI ランナー間でキャッシュ共有              |
|                                                      |
|  3. ローカルキャッシュ                                 |
|     cache-from: type=local,src=/tmp/.buildx-cache     |
|     cache-to: type=local,dest=/tmp/.buildx-cache-new  |
|     -> 自前の CI サーバーで利用                        |
|                                                      |
|  4. インラインキャッシュ                               |
|     --build-arg BUILDKIT_INLINE_CACHE=1               |
|     -> イメージ自体にキャッシュメタデータを埋め込む      |
|     -> 最もシンプルだがキャッシュ効率は最低             |
+------------------------------------------------------+
```

---

## 7. アンチパターン

### アンチパターン 1: ビルドステージの成果物を丸ごとコピー

```dockerfile
# NG: ビルドステージの全ファイルをコピー
FROM node:20-alpine AS builder
WORKDIR /app
COPY . .
RUN npm ci && npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app .  # <- 全てコピー（ソース、devDependencies含む）
CMD ["node", "dist/server.js"]
# -> マルチステージの意味がない

# OK: 必要なファイルだけをコピー
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
CMD ["node", "dist/server.js"]
```

### アンチパターン 2: Go で scratch を使うのに証明書を忘れる

```dockerfile
# NG: HTTPS通信ができない
FROM golang:1.22 AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
# -> 外部APIへのHTTPS通信で証明書エラー

# OK: CA証明書をコピー
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

### アンチパターン 3: ビルドステージでキャッシュ効率を無視

```dockerfile
# NG: 毎回全依存関係を再インストール
FROM node:20-alpine AS builder
WORKDIR /app
COPY . .                    # ソースコード変更 → 全キャッシュ無効
RUN npm ci                  # 毎回再実行
RUN npm run build

# OK: 依存関係ファイルを先にコピー
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./  # 依存関係変更時のみキャッシュ無効
RUN npm ci                               # キャッシュが効く
COPY . .                                 # ソースコードのみ再コピー
RUN npm run build
```

### アンチパターン 4: scratch でデバッグ不能なイメージ

```dockerfile
# 問題: scratch にはシェルがないためデバッグ困難
FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
# -> docker exec でシェルに入れない
# -> ファイルシステムの確認ができない

# 解決策 1: デバッグ用タグを用意
FROM alpine:3.19 AS debug
COPY --from=builder /server /server
ENTRYPOINT ["/server"]

FROM scratch AS production
COPY --from=builder /server /server
ENTRYPOINT ["/server"]

# 解決策 2: distroless の debug バリアント
FROM gcr.io/distroless/static-debian12:debug
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
# -> busybox シェルでデバッグ可能
```

---

## 8. FAQ

### Q1: マルチステージビルドはビルド時間が長くなりますか？

**A:** ステージ数が増えるため初回ビルドは若干長くなるが、キャッシュが効く2回目以降はむしろ高速になることが多い。依存関係のインストールとソースコードのコピーを分離することで、ソースコード変更時に依存関係の再インストールをスキップできる。BuildKit の `--mount=type=cache` を使えばさらにキャッシュ効率が向上する。また、BuildKit は依存関係のないステージを自動的に並列でビルドするため、ステージを適切に分割することでビルド時間を短縮できる。

### Q2: scratch と distroless はどう違いますか？

**A:** `scratch` は完全に空のベースイメージで、シェルもファイルシステムユーティリティもない。静的にリンクされたバイナリ（Go, Rust）向け。`distroless`（Google提供）は最小限のランタイム（glibc, CA証明書等）を含み、動的リンクが必要な言語（Node.js, Java, Python）で使える。デバッグ用に `:debug` タグで busybox シェルが入ったバリアントも提供されている。

### Q3: CI/CD でのキャッシュ戦略はどうすべきですか？

**A:** 以下の方法がある:
- **GitHub Actions Cache**: `type=gha` で GitHub の Cache API を利用。設定が最もシンプル。
- **レジストリキャッシュ**: `type=registry` でレジストリにキャッシュレイヤーを保存。異なる CI ランナー間で共有可能。
- **BuildKit インラインキャッシュ**: `BUILDKIT_INLINE_CACHE=1` でキャッシュメタデータをイメージに埋め込む。追加インフラ不要だがキャッシュ効率は低い。
- **ローカルキャッシュ**: `type=local` で CI サーバーのローカルディスクにキャッシュ。自前のCI環境向け。

### Q4: マルチステージで中間ステージのイメージは削除されますか？

**A:** ビルド完了後、中間ステージのレイヤーはビルドキャッシュとして保持されるが、最終イメージには含まれない。`docker system prune` や `docker builder prune` でビルドキャッシュを手動で削除できる。`--target` で特定のステージを指定してビルドした場合は、そのステージが最終イメージとなる。

### Q5: ステージ間でファイルを共有する方法は COPY --from だけですか？

**A:** `COPY --from` が主要な方法だが、BuildKit のマウントオプションも利用できる:
- `RUN --mount=type=bind,from=builder,source=/app/dist,target=/tmp/dist ...` で一時的にマウントして参照（COPY とは異なりレイヤーを生成しない）
- ボリュームマウント（docker compose で開発時）

---

## 9. まとめ

| 項目 | ポイント |
|---|---|
| 基本概念 | ビルド環境と実行環境を分離し、最終イメージを最小化 |
| COPY --from | ビルドステージから必要なファイルだけを実行ステージにコピー |
| Go / Rust | scratch ベースで 10-15MB のイメージが可能 |
| Node.js | Alpine + スタンドアロン出力で 100-150MB に削減 |
| Java | JRE + レイヤードJAR で 200MB 程度に削減 |
| Python | slim + prefix インストールでビルドツール分離 |
| キャッシュ | 依存関係とソースコードを分離してキャッシュ効率を最大化 |
| テスト統合 | テストステージを挟んでビルドパイプラインに組み込む |
| 開発/本番 | --target で開発・テスト・本番を切り替え |
| CI/CD | BuildKit キャッシュ（gha, registry, local）で高速化 |

---

## 次に読むべきガイド

- [02-optimization.md](./02-optimization.md) -- Dockerfile の最適化とセキュリティ
- [03-language-specific.md](./03-language-specific.md) -- 言語別 Dockerfile テンプレート集
- [../02-compose/00-compose-basics.md](../02-compose/00-compose-basics.md) -- Docker Compose の基礎

---

## 参考文献

1. **Docker Documentation - Multi-stage builds** https://docs.docker.com/build/building/multi-stage/ -- マルチステージビルドの公式ガイド。
2. **Google - Distroless Container Images** https://github.com/GoogleContainerTools/distroless -- Distroless イメージの公式リポジトリ。対応言語と使い方の説明。
3. **BuildKit - Dockerfile frontend** https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md -- `--mount=type=cache`, `--mount=type=secret` 等の高度な機能のリファレンス。
4. **cargo-chef** https://github.com/LukeMathWalker/cargo-chef -- Rust プロジェクトの Docker ビルドキャッシュを最適化するツール。
5. **Next.js - Docker Deployment** https://nextjs.org/docs/app/building-your-application/deploying#docker-image -- Next.js 公式の Docker デプロイガイド。
6. **Spring Boot - Container Images** https://docs.spring.io/spring-boot/docs/current/reference/html/container-images.html -- Spring Boot のコンテナイメージ最適化ガイド。
7. **Docker Build Cache** https://docs.docker.com/build/cache/ -- ビルドキャッシュの仕組みと最適化手法の公式ガイド。
8. **Python Speed - Multi-stage Docker builds** https://pythonspeed.com/articles/multi-stage-docker-python/ -- Python におけるマルチステージビルドの実践パターン。
