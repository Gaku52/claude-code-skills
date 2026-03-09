# 言語別 Dockerfile

> Node.js、Python、Go、Rust、Java それぞれの最適な Dockerfile パターンを、開発環境と本番環境の両方で示す実践リファレンス。

---

## この章で学ぶこと

1. **各言語固有のビルド特性**を理解し、言語に最適化された Dockerfile を書ける
2. **開発環境と本番環境で異なるステージ**を設計し、用途に応じたイメージを構築できる
3. **各言語のベストプラクティス**（依存関係管理、キャッシュ、セキュリティ）を適用できる
4. **パッケージマネージャ別のキャッシュ戦略**を理解し、ビルド時間を最小化できる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Dockerfile 最適化](./02-optimization.md) の内容を理解していること

---

## 1. Node.js

### 1.1 Express / Fastify (API サーバー)

```dockerfile
# syntax=docker/dockerfile:1

# === 依存関係インストール ===
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production

# === ビルド（TypeScript） ===
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json tsconfig.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci
COPY src/ ./src/
RUN npm run build

# === 本番 ===
FROM node:20-alpine
RUN addgroup -S app && adduser -S app -G app

# セキュリティ: 不要なツールを削除
RUN apk add --no-cache dumb-init

WORKDIR /app

COPY --from=deps --chown=app:app /app/node_modules ./node_modules
COPY --from=builder --chown=app:app /app/dist ./dist
COPY --chown=app:app package.json ./

USER app
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# dumb-init: PID 1 問題の解決（シグナル転送）
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/server.js"]
```

### 1.2 Next.js (SSR)

```dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# 環境変数をビルド時に注入
ARG NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_TELEMETRY_DISABLED=1

RUN npm run build

FROM node:20-alpine
WORKDIR /app
RUN addgroup -S app && adduser -S app -G app

# Next.js standalone 出力
COPY --from=builder --chown=app:app /app/.next/standalone ./
COPY --from=builder --chown=app:app /app/.next/static ./.next/static
COPY --from=builder --chown=app:app /app/public ./public

USER app
EXPOSE 3000
ENV PORT=3000 HOSTNAME="0.0.0.0"
CMD ["node", "server.js"]
```

### 1.3 NestJS

```dockerfile
# syntax=docker/dockerfile:1

FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci

FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS prod-deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production

FROM node:20-alpine
RUN addgroup -S app && adduser -S app -G app
RUN apk add --no-cache dumb-init

WORKDIR /app

COPY --from=prod-deps --chown=app:app /app/node_modules ./node_modules
COPY --from=builder --chown=app:app /app/dist ./dist
COPY --chown=app:app package.json ./

USER app
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/main.js"]
```

### 1.4 Node.js 開発環境

```dockerfile
# === 開発環境用 Dockerfile ===
FROM node:20-alpine AS development

# 開発ツールのインストール
RUN apk add --no-cache git curl

WORKDIR /app

# 依存関係のインストール（devDependencies を含む）
COPY package.json package-lock.json ./
RUN npm install

# ソースコードはバインドマウントで共有するため COPY 不要
# docker-compose.yml で volumes: [".:/app"] を設定

EXPOSE 3000

# ホットリロード対応
CMD ["npm", "run", "dev"]
```

### 1.5 Node.js 固有のポイント

```
+------------------------------------------------------+
|          Node.js Dockerfile のポイント                  |
|                                                      |
|  1. npm ci vs npm install                            |
|     npm ci: lockfile に完全一致、CI向け、高速          |
|     npm install: lockfile を更新する可能性あり         |
|                                                      |
|  2. PID 1 問題                                        |
|     node プロセスが PID 1 で動くとシグナル処理が不正確   |
|     -> dumb-init または tini を使う                    |
|     -> または --init フラグ: docker run --init ...    |
|                                                      |
|  3. NODE_ENV                                          |
|     production: devDependencies をスキップ             |
|     npm ci --only=production と併用                   |
|                                                      |
|  4. .npmrc の扱い                                     |
|     プライベートレジストリ認証は --mount=type=secret    |
|                                                      |
|  5. Alpine の互換性問題                                |
|     ネイティブバイナリ (sharp, bcrypt等) は             |
|     Alpine (musl) で問題が出る場合がある               |
|     -> npm rebuild で解決する場合あり                  |
|     -> 解決しない場合は node:20-slim を使用            |
+------------------------------------------------------+
```

#### npm vs yarn vs pnpm の比較

| 項目 | npm | yarn (v3+) | pnpm |
|---|---|---|---|
| ロックファイル | package-lock.json | yarn.lock | pnpm-lock.yaml |
| CI インストール | `npm ci` | `yarn install --immutable` | `pnpm install --frozen-lockfile` |
| キャッシュパス | `/root/.npm` | `/root/.yarn/cache` | `/root/.local/share/pnpm/store` |
| ワークスペース | npm workspaces | yarn workspaces | pnpm workspaces |
| ディスク効率 | 通常 | PnP で改善 | ハードリンクで最良 |

#### pnpm を使った Dockerfile

```dockerfile
FROM node:20-alpine AS builder
RUN corepack enable

WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

COPY . .
RUN pnpm run build

FROM node:20-alpine AS prod-deps
RUN corepack enable
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile --prod

FROM node:20-alpine
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app
COPY --from=prod-deps --chown=app:app /app/node_modules ./node_modules
COPY --from=builder --chown=app:app /app/dist ./dist
USER app
CMD ["node", "dist/server.js"]
```

---

## 2. Python

### 2.1 Flask / FastAPI

```dockerfile
# syntax=docker/dockerfile:1

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ビルド依存関係（ネイティブ拡張コンパイル用）
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# === 本番 ===
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ランタイム依存関係のみ（gcc は不要）
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

### 2.2 FastAPI + uvicorn

```dockerfile
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /install /usr/local

RUN useradd --create-home appuser
COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# uvicorn で非同期サーバーを起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2.3 Poetry を使う場合

```dockerfile
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install poetry==1.7.1
RUN poetry config virtualenvs.create false

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-dev --no-interaction --no-ansi

# === 本番 ===
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

RUN useradd --create-home appuser
COPY --chown=appuser:appuser . .
USER appuser

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
```

### 2.4 uv を使う場合（高速パッケージマネージャ）

```dockerfile
FROM python:3.12-slim AS builder

# uv のインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 依存関係のインストール
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# アプリケーションコード
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim
WORKDIR /app

RUN useradd --create-home appuser

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app .

ENV PATH="/app/.venv/bin:$PATH"

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.5 Django

```dockerfile
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=config.settings.production

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

# 静的ファイルの収集
RUN python manage.py collectstatic --noinput

RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "config.wsgi:application"]
```

### 2.6 Python 固有のポイント

```
+------------------------------------------------------+
|          Python Dockerfile のポイント                   |
|                                                      |
|  重要な環境変数                                        |
|  +------------------------------------------------+ |
|  | PYTHONDONTWRITEBYTECODE=1 | .pyc 生成を抑制     | |
|  | PYTHONUNBUFFERED=1        | バッファリングなし    | |
|  | PIP_NO_CACHE_DIR=1        | pip キャッシュ無効   | |
|  | PIP_DISABLE_PIP_VERSION_CHECK=1 | 更新チェック省略| |
|  +------------------------------------------------+ |
|                                                      |
|  パッケージマネージャの選択                             |
|  +------------------------------------------------+ |
|  | pip        | 標準、最もシンプル                   | |
|  | poetry     | pyproject.toml、依存関係管理が優秀   | |
|  | uv         | Rust製、非常に高速（pip の10-100倍）  | |
|  | pipenv     | Pipfile、仮想環境統合               | |
|  | pdm        | PEP 582準拠、モダン                  | |
|  +------------------------------------------------+ |
|                                                      |
|  マルチステージのポイント                               |
|  - pip install --prefix=/install で分離              |
|  - ビルドステージでのみ gcc をインストール              |
|  - ランタイムステージでは共有ライブラリのみ (.so)       |
+------------------------------------------------------+
```

---

## 3. Go

### 3.1 Web API サーバー

```dockerfile
# syntax=docker/dockerfile:1

FROM golang:1.22-alpine AS builder

# CA証明書とタイムゾーンデータを取得
RUN apk add --no-cache ca-certificates tzdata

# 非rootユーザーの事前作成
RUN adduser -D -g '' appuser

WORKDIR /app

# 依存関係を先にダウンロード
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download && go mod verify

# ソースコードのコピーとビルド
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build \
        -ldflags="-w -s -X main.version=$(git describe --tags 2>/dev/null || echo dev)" \
        -o /server \
        ./cmd/server

# === 本番（scratch）===
FROM scratch

# 必要なファイルのみコピー
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /server /server

USER appuser
EXPOSE 8080
ENTRYPOINT ["/server"]
```

### 3.2 CGO が必要な場合

```dockerfile
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache gcc musl-dev

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# CGO_ENABLED=1 (デフォルト) で静的リンク
RUN go build -ldflags="-w -s -linkmode external -extldflags '-static'" \
    -o /server ./cmd/server

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
RUN adduser -D appuser
COPY --from=builder /server /server
USER appuser
EXPOSE 8080
ENTRYPOINT ["/server"]
```

### 3.3 マルチプラットフォーム対応

```dockerfile
FROM --platform=$BUILDPLATFORM golang:1.22-alpine AS builder

ARG TARGETOS
ARG TARGETARCH

RUN apk add --no-cache ca-certificates tzdata git
RUN adduser -D -g '' appuser

WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download

COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -ldflags="-w -s" -o /server ./cmd/server

FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /server /server
USER appuser
EXPOSE 8080
ENTRYPOINT ["/server"]
```

### 3.4 Go 固有のポイント

```
+------------------------------------------------------+
|              Go Dockerfile のポイント                   |
|                                                      |
|  ビルドフラグ                                         |
|  +------------------------------------------------+ |
|  | CGO_ENABLED=0  | Cライブラリ依存なし             | |
|  |                | -> scratch が使える             | |
|  | -ldflags="-w"  | DWARF デバッグ情報を除去        | |
|  | -ldflags="-s"  | シンボルテーブルを除去           | |
|  | GOOS=linux     | Linux 向けバイナリ              | |
|  | GOARCH=amd64   | x86_64 アーキテクチャ           | |
|  +------------------------------------------------+ |
|                                                      |
|  ベースイメージ選択                                    |
|  +------------------------------------------------+ |
|  | scratch        | 最小 (バイナリのみ)             | |
|  | alpine         | シェルが使える (デバッグ用)      | |
|  | distroless     | 中間 (glibc あり)              | |
|  +------------------------------------------------+ |
|                                                      |
|  scratch で必要な追加ファイル                           |
|  +------------------------------------------------+ |
|  | /etc/ssl/certs/ | HTTPS 通信用 CA 証明書         | |
|  | /usr/share/zoneinfo | タイムゾーン情報            | |
|  | /etc/passwd    | non-root ユーザー情報           | |
|  | /tmp           | 一時ファイル用 (必要な場合)     | |
|  +------------------------------------------------+ |
+------------------------------------------------------+
```

---

## 4. Rust

### 4.1 Actix-web / Axum

```dockerfile
# syntax=docker/dockerfile:1

# === 依存関係プランニング ===
FROM rust:1.75-alpine AS chef
RUN apk add --no-cache musl-dev
RUN cargo install cargo-chef --locked
WORKDIR /app

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# === 依存関係ビルド（キャッシュ用） ===
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json

# 依存関係のみビルド（ソースコード変更時にキャッシュが効く）
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    cargo chef cook --release --recipe-path recipe.json

# アプリケーションビルド
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release && \
    cp target/release/myapp /usr/local/bin/

# === 本番 ===
FROM alpine:3.19
RUN apk add --no-cache ca-certificates
RUN addgroup -S app && adduser -S app -G app

COPY --from=builder /usr/local/bin/myapp /usr/local/bin/

USER app
EXPOSE 8080
CMD ["myapp"]
```

### 4.2 静的リンク（musl）で scratch を使う

```dockerfile
FROM rust:1.75-alpine AS builder
RUN apk add --no-cache musl-dev

WORKDIR /app
COPY . .

# musl ターゲットで静的リンク
RUN rustup target add x86_64-unknown-linux-musl
RUN RUSTFLAGS="-C target-feature=+crt-static" \
    cargo build --release --target x86_64-unknown-linux-musl

FROM scratch
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/myapp /myapp
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

EXPOSE 8080
ENTRYPOINT ["/myapp"]
```

### 4.3 sccache を使ったコンパイルキャッシュ

```dockerfile
FROM rust:1.75-alpine AS builder
RUN apk add --no-cache musl-dev

# sccache のインストール
RUN cargo install sccache --locked
ENV RUSTC_WRAPPER=sccache
ENV SCCACHE_DIR=/sccache

WORKDIR /app
COPY . .

RUN --mount=type=cache,target=/sccache \
    --mount=type=cache,target=/usr/local/cargo/registry \
    cargo build --release && \
    cp target/release/myapp /usr/local/bin/

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=builder /usr/local/bin/myapp /usr/local/bin/
EXPOSE 8080
CMD ["myapp"]
```

### 4.4 Rust 固有のポイント

```
+------------------------------------------------------+
|           Rust Dockerfile のポイント                    |
|                                                      |
|  課題: Rust のビルドは非常に遅い                        |
|  (フルビルドで数分〜数十分)                             |
|                                                      |
|  解決策:                                              |
|  1. cargo-chef で依存関係のみを先にビルド              |
|     -> ソースコード変更時に依存関係キャッシュが効く      |
|                                                      |
|  2. BuildKit マウントキャッシュ                        |
|     -> cargo registry と target のキャッシュ           |
|                                                      |
|  3. sccache (共有コンパイルキャッシュ)                  |
|     -> CI環境で複数ビルド間のキャッシュ共有             |
|                                                      |
|  4. ダミー main.rs パターン                            |
|     -> cargo-chef なしでも依存関係キャッシュ可能        |
|                                                      |
|  静的リンク: musl libc でコンパイル                     |
|  -> scratch ベースで 5-15MB のイメージが可能           |
+------------------------------------------------------+
```

---

## 5. Java

### 5.1 Spring Boot (Gradle)

```dockerfile
# syntax=docker/dockerfile:1

# === ビルド ===
FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /app

# Gradle Wrapper と設定ファイル
COPY gradlew build.gradle.kts settings.gradle.kts ./
COPY gradle ./gradle

# 依存関係のダウンロード（キャッシュ用）
RUN --mount=type=cache,target=/root/.gradle \
    ./gradlew dependencies --no-daemon

# ソースコードのコピーとビルド
COPY src ./src
RUN --mount=type=cache,target=/root/.gradle \
    ./gradlew bootJar --no-daemon

# JAR のレイヤー展開
RUN java -Djarmode=layertools \
    -jar build/libs/*.jar extract --destination extracted

# === 本番 ===
FROM eclipse-temurin:21-jre-alpine

RUN addgroup -S app && adduser -S app -G app
WORKDIR /app

# Spring Boot レイヤー（変更頻度: 低 -> 高）
COPY --from=builder --chown=app:app /app/extracted/dependencies/ ./
COPY --from=builder --chown=app:app /app/extracted/spring-boot-loader/ ./
COPY --from=builder --chown=app:app /app/extracted/snapshot-dependencies/ ./
COPY --from=builder --chown=app:app /app/extracted/application/ ./

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

# JVM チューニング
ENV JAVA_OPTS="-XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0"
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS org.springframework.boot.loader.launch.JarLauncher"]
```

### 5.2 Maven の場合

```dockerfile
FROM eclipse-temurin:21-jdk-alpine AS builder
WORKDIR /app

COPY pom.xml ./
COPY .mvn ./.mvn
COPY mvnw ./
RUN --mount=type=cache,target=/root/.m2 \
    ./mvnw dependency:go-offline -B

COPY src ./src
RUN --mount=type=cache,target=/root/.m2 \
    ./mvnw package -DskipTests -B

FROM eclipse-temurin:21-jre-alpine
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app
COPY --from=builder --chown=app:app /app/target/*.jar app.jar
USER app
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
```

### 5.3 GraalVM Native Image

```dockerfile
# === ステージ 1: ネイティブイメージビルド ===
FROM ghcr.io/graalvm/native-image-community:21 AS builder
WORKDIR /app

COPY gradlew build.gradle.kts settings.gradle.kts ./
COPY gradle ./gradle
COPY src ./src

# ネイティブイメージのビルド（時間がかかる）
RUN ./gradlew nativeCompile --no-daemon

# === ステージ 2: 実行 ===
FROM debian:bookworm-slim
RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /app/build/native/nativeCompile/myapp /usr/local/bin/

USER app
EXPOSE 8080
ENTRYPOINT ["myapp"]
```

### 5.4 Java 固有のポイント

```
+------------------------------------------------------+
|            Java Dockerfile のポイント                   |
|                                                      |
|  JDK vs JRE                                          |
|  +------------------------------------------------+ |
|  | ビルド: eclipse-temurin:21-jdk-alpine           | |
|  | 実行:   eclipse-temurin:21-jre-alpine           | |
|  |   JRE はコンパイラを含まない -> 軽量              | |
|  +------------------------------------------------+ |
|                                                      |
|  JVM コンテナサポート (Java 10+)                       |
|  +------------------------------------------------+ |
|  | -XX:+UseContainerSupport  | コンテナ認識         | |
|  | -XX:MaxRAMPercentage=75.0 | メモリの75%使用      | |
|  | -XX:InitialRAMPercentage  | 初期ヒープ           | |
|  +------------------------------------------------+ |
|                                                      |
|  起動時間の改善                                        |
|  +------------------------------------------------+ |
|  | Spring Boot レイヤードJAR | レイヤーキャッシュ     | |
|  | CDS (Class Data Sharing)  | クラスロード高速化   | |
|  | GraalVM Native Image      | ネイティブコンパイル  | |
|  | Spring AOT                | ビルド時最適化       | |
|  +------------------------------------------------+ |
|                                                      |
|  GraalVM Native Image の特徴                          |
|  +------------------------------------------------+ |
|  | 起動時間: 数十ミリ秒（JVM: 数秒〜数十秒）         | |
|  | メモリ使用量: 大幅に削減                          | |
|  | ビルド時間: 非常に長い（数分〜数十分）              | |
|  | リフレクション: 設定が必要                        | |
|  | 対応ライブラリ: 一部制約あり                      | |
|  +------------------------------------------------+ |
+------------------------------------------------------+
```

---

## 6. Ruby

### 6.1 Ruby on Rails

```dockerfile
FROM ruby:3.3-slim AS builder

ENV RAILS_ENV=production \
    BUNDLE_WITHOUT=development:test \
    BUNDLE_DEPLOYMENT=1

WORKDIR /app

# システム依存関係
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

# Gem のインストール
COPY Gemfile Gemfile.lock ./
RUN --mount=type=cache,target=/usr/local/bundle/cache \
    bundle install --jobs 4 --retry 3

# アプリケーションコード
COPY . .

# アセットプリコンパイル
RUN SECRET_KEY_BASE=dummy bundle exec rails assets:precompile

# === 本番 ===
FROM ruby:3.3-slim

ENV RAILS_ENV=production \
    RAILS_LOG_TO_STDOUT=true \
    RAILS_SERVE_STATIC_FILES=true \
    BUNDLE_WITHOUT=development:test \
    BUNDLE_DEPLOYMENT=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpq5 \
        curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bundle /usr/local/bundle
COPY --from=builder /app .

RUN useradd --create-home --shell /bin/bash rails && \
    chown -R rails:rails /app/log /app/tmp /app/storage
USER rails

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["bundle", "exec", "puma", "-C", "config/puma.rb"]
```

---

## 7. PHP

### 7.1 Laravel

```dockerfile
FROM composer:2 AS vendor
WORKDIR /app
COPY composer.json composer.lock ./
RUN composer install --no-dev --no-interaction --no-scripts --prefer-dist

FROM node:20-alpine AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY resources ./resources
COPY vite.config.js ./
RUN npm run build

FROM php:8.3-fpm-alpine

# PHP 拡張のインストール
RUN docker-php-ext-install pdo pdo_mysql opcache bcmath

# Composer からの vendor
COPY --from=vendor /app/vendor ./vendor

# フロントエンドアセット
COPY --from=frontend /app/public/build ./public/build

# アプリケーションコード
COPY . .

# パーミッション設定
RUN chown -R www-data:www-data storage bootstrap/cache

# PHP 設定
COPY docker/php/php.ini /usr/local/etc/php/php.ini
COPY docker/php/opcache.ini /usr/local/etc/php/conf.d/opcache.ini

USER www-data
EXPOSE 9000
CMD ["php-fpm"]
```

---

## 8. 比較表

### 比較表 1: 言語別 Dockerfile 特性

| 言語 | ベースイメージ (本番) | 典型的サイズ | ビルド時間 | 特殊考慮事項 |
|---|---|---|---|---|
| Node.js | node:20-alpine | 100-200MB | 中 | PID 1 問題, standalone出力 |
| Python | python:3.12-slim | 100-300MB | 中 | venv不要, ネイティブ拡張 |
| Go | scratch | 5-20MB | 速い | 静的バイナリ, CA証明書 |
| Rust | scratch / alpine | 5-20MB | 遅い | cargo-chef, 長いコンパイル |
| Java | temurin:21-jre-alpine | 150-300MB | 中〜遅い | JVM チューニング, レイヤードJAR |
| Ruby | ruby:3.3-slim | 200-400MB | 中 | native gem, アセットコンパイル |
| PHP | php:8.3-fpm-alpine | 100-250MB | 速い | 拡張インストール, composer |

### 比較表 2: 依存関係管理のキャッシュ戦略

| 言語 | ロックファイル | キャッシュ対象 | マウントキャッシュパス |
|---|---|---|---|
| Node.js (npm) | package-lock.json | node_modules | `/root/.npm` |
| Node.js (pnpm) | pnpm-lock.yaml | pnpm store | `/root/.local/share/pnpm/store` |
| Node.js (yarn) | yarn.lock | yarn cache | `/root/.yarn/cache` |
| Python (pip) | requirements.txt | site-packages | `/root/.cache/pip` |
| Python (Poetry) | poetry.lock | site-packages | `/root/.cache/pypoetry` |
| Python (uv) | uv.lock | uv cache | `/root/.cache/uv` |
| Go | go.sum | module cache | `/go/pkg/mod` |
| Rust | Cargo.lock | registry + target | `/usr/local/cargo/registry` |
| Java (Gradle) | gradle.lockfile | .gradle | `/root/.gradle` |
| Java (Maven) | pom.xml | .m2 | `/root/.m2` |
| Ruby | Gemfile.lock | bundle | `/usr/local/bundle/cache` |
| PHP | composer.lock | vendor | `/tmp/composer-cache` |

### 比較表 3: .dockerignore テンプレート（言語別）

| 言語 | 除外すべきファイル |
|---|---|
| Node.js | `node_modules`, `dist`, `.next`, `coverage`, `.env` |
| Python | `__pycache__`, `*.pyc`, `.venv`, `*.egg-info`, `.mypy_cache` |
| Go | `vendor/` (go mod 使用時), `*.test`, `coverage.out` |
| Rust | `target/`, `*.pdb` |
| Java | `build/`, `target/`, `.gradle/`, `*.class`, `*.jar` |
| Ruby | `vendor/bundle`, `node_modules`, `tmp/`, `log/` |
| PHP | `vendor/`, `node_modules/`, `storage/logs/` |

---

## 9. アンチパターン

### アンチパターン 1: 開発用依存関係を本番イメージに含める

```dockerfile
# NG: devDependencies が含まれる
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm install  # <- devDependencies も入る
CMD ["node", "server.js"]
# -> eslint, jest, typescript 等がイメージに含まれる

# OK: 本番用依存関係のみ
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production
COPY dist/ ./dist/
CMD ["node", "dist/server.js"]
```

### アンチパターン 2: 全言語で同じベースイメージを使う

```dockerfile
# NG: Go なのに ubuntu を使う
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y golang
COPY . /app
RUN cd /app && go build -o /server
CMD ["/server"]
# -> 不要な OS パッケージで 800MB+

# OK: scratch で最小イメージ
FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /server .

FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
# -> 12MB 程度
```

### アンチパターン 3: 依存関係のロックファイルを使わない

```dockerfile
# NG: バージョンが固定されない
FROM python:3.12-slim
COPY requirements.txt .
# requirements.txt に numpy>=1.0 のような範囲指定
RUN pip install -r requirements.txt
# -> ビルドのたびに異なるバージョンがインストールされる可能性

# OK: 厳密なバージョン固定
FROM python:3.12-slim
COPY requirements.txt .
# requirements.txt に numpy==1.26.4 のような完全固定
# または pip-compile で生成
RUN pip install --no-cache-dir -r requirements.txt
```

### アンチパターン 4: ビルドツールを本番イメージに残す

```dockerfile
# NG: gcc がイメージに残る
FROM python:3.12-slim
RUN apt-get update && apt-get install -y gcc libpq-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app"]
# -> gcc (~200MB) が不要なのにイメージに含まれる

# OK: マルチステージでビルドツールを分離
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
COPY . .
CMD ["gunicorn", "app:app"]
```

### アンチパターン 5: HEALTHCHECK を省略する

```dockerfile
# NG: ヘルスチェックなし
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm ci --only=production
CMD ["node", "server.js"]
# -> コンテナは起動しているがアプリがクラッシュしていても検知できない

# OK: 言語ごとの適切なヘルスチェック
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm ci --only=production && apk add --no-cache curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1
CMD ["node", "server.js"]
```

**問題点**: HEALTHCHECK がないと、プロセスは生きているがアプリケーションがデッドロックやメモリリークで応答不能になった場合に検知できない。特に Java や Node.js のようなランタイムでは、プロセスが終了せずに応答不能になるケースが多い。言語ごとに適切なヘルスチェックコマンドを設定すべきである。

### アンチパターン 6: マルチプラットフォーム非対応の構成

```dockerfile
# NG: amd64 固有のバイナリをハードコード
FROM node:20-alpine
RUN wget https://example.com/tool-linux-amd64 -O /usr/local/bin/tool
# -> ARM (Apple Silicon M1/M2/M3, Graviton) で動かない

# OK: プラットフォームに応じたバイナリを選択
FROM node:20-alpine
ARG TARGETARCH
RUN wget "https://example.com/tool-linux-${TARGETARCH}" -O /usr/local/bin/tool && \
    chmod +x /usr/local/bin/tool
```

**問題点**: Apple Silicon Mac や AWS Graviton の普及により、ARM64 対応は必須となりつつある。ベースイメージ自体はマルチプラットフォーム対応でも、追加でダウンロードするバイナリやネイティブ拡張がアーキテクチャ固有の場合がある。`TARGETARCH` ビルド引数を活用してプラットフォームに依存しない Dockerfile を書くべきである。


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
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

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 10. FAQ

### Q1: Node.js でバイナリを含む npm パッケージ（sharp 等）を使う場合はどうしますか？

**A:** Alpine の場合、ネイティブバイナリの互換性問題が起きることがある。対策は以下の通り:
- `npm ci --only=production` を Alpine 上で実行する（ネイティブバイナリが Alpine 向けにビルドされる）
- `sharp` の場合は `--platform=linuxmusl` を指定する
- どうしても解決しない場合は `node:20-slim`（Debian ベース）に切り替える
- `npm rebuild` を実行してネイティブモジュールを再コンパイルする

### Q2: Python で仮想環境（venv）はコンテナ内でも必要ですか？

**A:** 基本的に不要である。コンテナ自体が隔離環境なので、venv による二重の隔離は通常不要。ただし、マルチステージビルドで `pip install --prefix=/install` を使って依存関係を特定のディレクトリにインストールし、実行ステージにコピーするパターンが推奨される。Poetry を使う場合は `virtualenvs.create false` で venv を無効化する。uv を使う場合は `.venv` ディレクトリごとコピーするパターンが標準的。

### Q3: Java の GraalVM Native Image はコンテナで有効ですか？

**A:** 非常に有効である。通常の JVM モードでは起動に数秒〜数十秒かかるが、Native Image では数十ミリ秒で起動し、メモリ使用量も大幅に削減される。ただし、ビルド時間が非常に長い（数分〜数十分）、リフレクションの設定が必要、一部ライブラリが非対応という制約がある。サーバーレスやスケールアウトが頻繁な環境で特に効果的。

### Q4: Go の scratch イメージでデバッグするにはどうすればよいですか？

**A:** scratch にはシェルがないため、以下の方法でデバッグする:
- **alpine ベースのデバッグイメージ**: `FROM alpine:3.19` をベースとした別のステージを用意し、`--target debug` でビルドする
- **distroless debug バリアント**: `FROM gcr.io/distroless/static-debian12:debug` を使えば busybox シェルが利用可能
- **kubectl debug** (Kubernetes): エフェメラルコンテナでデバッグ用コンテナをアタッチする
- **docker cp**: コンテナからファイルをホストにコピーして確認する

### Q5: 開発環境と本番環境で同じ Dockerfile を使うべきですか？

**A:** マルチステージビルドの `--target` を活用して同じ Dockerfile 内で開発環境と本番環境を分岐させるのが推奨される。開発環境ステージには devDependencies やデバッグツール、ホットリロード設定を含め、本番環境ステージでは最小構成にする。`docker-compose.yml` で `build.target` を指定することで、同一 Dockerfile から異なるイメージをビルドできる。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 11. まとめ

| 項目 | ポイント |
|---|---|
| Node.js | npm ci + Alpine + standalone出力。PID 1 問題に注意。pnpm/yarn も対応可 |
| Python | slim ベース + マルチステージでビルドツール分離。uv で高速化 |
| Go | scratch ベースで最小イメージ。CA証明書を忘れずにコピー |
| Rust | cargo-chef + マウントキャッシュで長いビルド時間を緩和 |
| Java | JRE + レイヤードJAR。JVM コンテナサポートのチューニング |
| Ruby | slim + bundle cache。native gem のビルド依存を分離 |
| PHP | fpm-alpine + composer + node (フロントエンド)。拡張管理が重要 |
| 共通 | non-root ユーザー、HEALTHCHECK、.dockerignore、マルチステージ |

---

## 次に読むべきガイド

- [../02-compose/00-compose-basics.md](../02-compose/00-compose-basics.md) -- Docker Compose の基礎
- [../02-compose/02-development-workflow.md](../02-compose/02-development-workflow.md) -- Compose 開発ワークフロー
- [02-optimization.md](./02-optimization.md) -- Dockerfile の最適化

---

## 参考文献

1. **Node.js Docker Best Practices** https://github.com/nodejs/docker-node/blob/main/docs/BestPractices.md -- Node.js 公式の Docker ベストプラクティス。
2. **Python Speed - Docker packaging guide** https://pythonspeed.com/docker/ -- Python コンテナの最適化に関する包括的なガイド。
3. **Google - Distroless Images** https://github.com/GoogleContainerTools/distroless -- 各言語の Distroless イメージの説明と使い方。
4. **cargo-chef documentation** https://github.com/LukeMathWalker/cargo-chef -- Rust Docker ビルドキャッシュ最適化ツール。
5. **GraalVM Native Image** https://www.graalvm.org/latest/reference-manual/native-image/ -- GraalVM ネイティブイメージの公式ドキュメント。
6. **uv - Python package manager** https://docs.astral.sh/uv/ -- Rust 製の高速 Python パッケージマネージャ。
