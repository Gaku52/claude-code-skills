# マルチステージビルド

> ビルダーパターンを活用してイメージサイズを大幅に削減し、セキュリティと効率を両立させる実践ガイド。Node.js、Go、Rust の言語別例を含む。

---

## この章で学ぶこと

1. **マルチステージビルドの仕組み**を理解し、ビルド環境と実行環境を分離できる
2. **言語別の最適なビルダーパターン**を実装し、最小サイズのイメージを構築できる
3. **キャッシュ戦略と中間ステージの活用**で、ビルド速度とイメージ品質を最適化できる

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
WORKDIR /app

COPY --from=prod-deps --chown=app:app /app/node_modules ./node_modules
COPY --from=builder --chown=app:app /app/dist ./dist
COPY --chown=app:app package.json ./

USER app
EXPOSE 3000
CMD ["node", "dist/server.js"]
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
CMD ["node", "server.js"]
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
ENTRYPOINT ["java", "org.springframework.boot.loader.launch.JarLauncher"]
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
| `scratch` | 0 B | 静的バイナリ専用 | なし |

---

## 4. 高度なテクニック

### 4.1 外部イメージからのコピー

```dockerfile
# 他のイメージからファイルをコピー
FROM alpine:3.19
COPY --from=nginx:alpine /etc/nginx/nginx.conf /etc/nginx/
COPY --from=busybox:uclibc /bin/wget /usr/local/bin/
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

# === 実行ステージ ===
FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]
```

```bash
# テストのみ実行
docker build --target tester .

# テストが通らないと最終ステージもビルドされない
# （CI/CD で活用）
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
CMD ["npm", "run", "dev"]

# === ビルド ===
FROM base AS builder
RUN npm ci
COPY . .
RUN npm run build

# === 本番環境 ===
FROM node:20-alpine AS production
WORKDIR /app
RUN addgroup -S app && adduser -S app -G app

COPY --from=builder --chown=app:app /app/dist ./dist
COPY --from=builder --chown=app:app /app/node_modules ./node_modules
COPY --chown=app:app package.json ./

USER app
CMD ["node", "dist/server.js"]
```

```bash
# 開発環境でビルド
docker build --target development -t my-app:dev .

# 本番環境でビルド
docker build --target production -t my-app:prod .
```

---

## 6. アンチパターン

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

---

## 7. FAQ

### Q1: マルチステージビルドはビルド時間が長くなりますか？

**A:** ステージ数が増えるため初回ビルドは若干長くなるが、キャッシュが効く2回目以降はむしろ高速になることが多い。依存関係のインストールとソースコードのコピーを分離することで、ソースコード変更時に依存関係の再インストールをスキップできる。BuildKit の `--mount=type=cache` を使えばさらにキャッシュ効率が向上する。

### Q2: scratch と distroless はどう違いますか？

**A:** `scratch` は完全に空のベースイメージで、シェルもファイルシステムユーティリティもない。静的にリンクされたバイナリ（Go, Rust）向け。`distroless`（Google提供）は最小限のランタイム（glibc, CA証明書等）を含み、動的リンクが必要な言語（Node.js, Java, Python）で使える。デバッグ用に `:debug` タグで busybox シェルが入ったバリアントも提供されている。

### Q3: CI/CD でのキャッシュ戦略はどうすべきですか？

**A:** 以下の方法がある:
- **レジストリキャッシュ**: `docker build --cache-from` で前回のイメージをキャッシュとして利用
- **BuildKit インラインキャッシュ**: `--build-arg BUILDKIT_INLINE_CACHE=1` でキャッシュメタデータをイメージに埋め込む
- **GitHub Actions Cache**: `docker/build-push-action` の `cache-from/cache-to` で GitHub のキャッシュを利用
- **BuildKit ローカルキャッシュ**: `--cache-to type=local` でローカルディレクトリにキャッシュを永続化

---

## 8. まとめ

| 項目 | ポイント |
|---|---|
| 基本概念 | ビルド環境と実行環境を分離し、最終イメージを最小化 |
| COPY --from | ビルドステージから必要なファイルだけを実行ステージにコピー |
| Go / Rust | scratch ベースで 10-15MB のイメージが可能 |
| Node.js | Alpine + スタンドアロン出力で 100-150MB に削減 |
| Java | JRE + レイヤードJAR で 200MB 程度に削減 |
| キャッシュ | 依存関係とソースコードを分離してキャッシュ効率を最大化 |
| テスト統合 | テストステージを挟んでビルドパイプラインに組み込む |

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
