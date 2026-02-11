# 言語別 Dockerfile

> Node.js、Python、Go、Rust、Java それぞれの最適な Dockerfile パターンを、開発環境と本番環境の両方で示す実践リファレンス。

---

## この章で学ぶこと

1. **各言語固有のビルド特性**を理解し、言語に最適化された Dockerfile を書ける
2. **開発環境と本番環境で異なるステージ**を設計し、用途に応じたイメージを構築できる
3. **各言語のベストプラクティス**（依存関係管理、キャッシュ、セキュリティ）を適用できる

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

### 1.3 Node.js 固有のポイント

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
+------------------------------------------------------+
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
    apt-get install -y --no-install-recommends libpq5 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash appuser
COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

### 2.2 Poetry を使う場合

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

### 2.3 Python 固有のポイント

```dockerfile
# 重要な環境変数
ENV PYTHONDONTWRITEBYTECODE=1  # .pyc ファイルを作らない
ENV PYTHONUNBUFFERED=1         # stdout/stderr をバッファリングしない
ENV PIP_NO_CACHE_DIR=1         # pip キャッシュを無効化
ENV PIP_DISABLE_PIP_VERSION_CHECK=1  # バージョンチェックをスキップ
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

### 3.3 Go 固有のポイント

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

### 4.3 Rust 固有のポイント

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

### 5.3 Java 固有のポイント

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
|  +------------------------------------------------+ |
+------------------------------------------------------+
```

---

## 6. 比較表

### 比較表 1: 言語別 Dockerfile 特性

| 言語 | ベースイメージ (本番) | 典型的サイズ | ビルド時間 | 特殊考慮事項 |
|---|---|---|---|---|
| Node.js | node:20-alpine | 100-200MB | 中 | PID 1 問題, standalone出力 |
| Python | python:3.12-slim | 100-300MB | 中 | venv不要, ネイティブ拡張 |
| Go | scratch | 5-20MB | 速い | 静的バイナリ, CA証明書 |
| Rust | scratch / alpine | 5-20MB | 遅い | cargo-chef, 長いコンパイル |
| Java | temurin:21-jre-alpine | 150-300MB | 中〜遅い | JVM チューニング, レイヤードJAR |

### 比較表 2: 依存関係管理のキャッシュ戦略

| 言語 | ロックファイル | キャッシュ対象 | マウントキャッシュパス |
|---|---|---|---|
| Node.js | package-lock.json | node_modules | `/root/.npm` |
| Python (pip) | requirements.txt | site-packages | `/root/.cache/pip` |
| Python (Poetry) | poetry.lock | site-packages | `/root/.cache/pypoetry` |
| Go | go.sum | module cache | `/go/pkg/mod` |
| Rust | Cargo.lock | registry + target | `/usr/local/cargo/registry` |
| Java (Gradle) | gradle.lockfile | .gradle | `/root/.gradle` |
| Java (Maven) | pom.xml | .m2 | `/root/.m2` |

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: Node.js でバイナリを含む npm パッケージ（sharp 等）を使う場合はどうしますか？

**A:** Alpine の場合、ネイティブバイナリの互換性問題が起きることがある。対策は以下の通り:
- `npm ci --only=production` を Alpine 上で実行する（ネイティブバイナリが Alpine 向けにビルドされる）
- `sharp` の場合は `--platform=linuxmusl` を指定する
- どうしても解決しない場合は `node:20-slim`（Debian ベース）に切り替える

### Q2: Python で仮想環境（venv）はコンテナ内でも必要ですか？

**A:** 基本的に不要である。コンテナ自体が隔離環境なので、venv による二重の隔離は通常不要。ただし、マルチステージビルドで `pip install --prefix=/install` を使って依存関係を特定のディレクトリにインストールし、実行ステージにコピーするパターンが推奨される。Poetry を使う場合は `virtualenvs.create false` で venv を無効化する。

### Q3: Java の GraalVM Native Image はコンテナで有効ですか？

**A:** 非常に有効である。通常の JVM モードでは起動に数秒〜数十秒かかるが、Native Image では数十ミリ秒で起動し、メモリ使用量も大幅に削減される。ただし、ビルド時間が非常に長い（数分〜数十分）、リフレクションの設定が必要、一部ライブラリが非対応という制約がある。サーバーレスやスケールアウトが頻繁な環境で特に効果的。

---

## 9. まとめ

| 項目 | ポイント |
|---|---|
| Node.js | npm ci + Alpine + standalone出力。PID 1 問題に注意 |
| Python | slim ベース + マルチステージでビルドツール分離。PYTHONUNBUFFERED=1 |
| Go | scratch ベースで最小イメージ。CA証明書を忘れずにコピー |
| Rust | cargo-chef + マウントキャッシュで長いビルド時間を緩和 |
| Java | JRE + レイヤードJAR。JVM コンテナサポートのチューニング |
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
