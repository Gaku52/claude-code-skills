# Dockerfile 最適化

> レイヤーキャッシュの活用、.dockerignore の設計、セキュリティスキャン、ベストプラクティスを網羅し、本番品質のコンテナイメージを構築する。

---

## この章で学ぶこと

1. **レイヤーキャッシュの仕組み**を深く理解し、ビルド時間を最小化する戦略を実装できる
2. **セキュリティスキャンとハードニング**を実施し、脆弱性の少ないイメージを構築できる
3. **Dockerfile のベストプラクティス**を体系的に適用し、保守性・効率性の高いイメージを作成できる
4. **マルチプラットフォームビルド**の設計と実行を理解し、AMD64/ARM64 両対応のイメージを配布できる
5. **CI/CD パイプライン**でのビルド最適化手法を理解し、実践に活かせる

---

## 1. レイヤーキャッシュ戦略

### 1.1 キャッシュの動作原理

```
+------------------------------------------------------+
|              キャッシュ判定フロー                        |
|                                                      |
|  各命令に対して:                                       |
|                                                      |
|  1. FROM: ベースイメージが同じか？                      |
|     -> 異なれば全レイヤー再ビルド                       |
|                                                      |
|  2. RUN: コマンド文字列が同じか？                       |
|     -> 文字列が1文字でも異なれば再ビルド                 |
|     -> コマンドの実行結果は比較しない                    |
|                                                      |
|  3. COPY/ADD: ファイルのチェックサムが同じか？           |
|     -> ファイル内容が変わればキャッシュ無効              |
|     -> タイムスタンプは無視（内容のみ比較）              |
|                                                      |
|  重要: あるレイヤーのキャッシュが無効になると             |
|        それ以降の全レイヤーが再ビルドされる              |
|                                                      |
|  [キャッシュヒット] -> [キャッシュヒット] -> [ミス!]     |
|  -> [再ビルド] -> [再ビルド] -> [再ビルド]              |
+------------------------------------------------------+
```

Docker のビルドキャッシュは各レイヤー（Dockerfile の各命令）単位で判定される。ビルドエンジンは上から順にレイヤーを処理し、各レイヤーのキャッシュが有効かどうかを判定する。FROM 命令ではベースイメージのダイジェストが一致するかを確認し、RUN 命令ではコマンド文字列の完全一致を確認する。COPY や ADD ではコピー対象ファイルのメタデータ（サイズ、パーミッション、内容のハッシュ）を比較する。

キャッシュの最も重要な特性は「カスケード無効化」である。あるレイヤーでキャッシュが無効になると、そのレイヤー以降のすべてのレイヤーが再ビルドされる。これは、各レイヤーが前のレイヤーの結果に依存しているためである。この性質を理解することが、キャッシュ最適化の基盤となる。

### 1.2 最適なレイヤー順序

```dockerfile
# === 最適化された Dockerfile ===

# 1. ベースイメージ（変更頻度: 最低）
FROM node:20-alpine

WORKDIR /app

# 2. システム依存関係（変更頻度: 低）
RUN apk add --no-cache curl

# 3. 言語依存関係の定義ファイル（変更頻度: 中低）
COPY package.json package-lock.json ./

# 4. 依存関係のインストール（変更頻度: 中低）
RUN npm ci --only=production

# 5. 設定ファイル（変更頻度: 中）
COPY tsconfig.json ./

# 6. ソースコード（変更頻度: 最高）
COPY src/ ./src/

# 7. ビルド
RUN npm run build

CMD ["node", "dist/server.js"]
```

レイヤー順序の最適化原則は「変更頻度の低いものを上に、高いものを下に」配置することである。ソースコードは最も頻繁に変更されるため、Dockerfile の最下部に配置する。依存関係の定義ファイル（package.json 等）は比較的安定しているため、ソースコードよりも上に配置する。

### 1.3 BuildKit マウントキャッシュ

```dockerfile
# syntax=docker/dockerfile:1

FROM node:20-alpine
WORKDIR /app

COPY package.json package-lock.json ./

# npm キャッシュディレクトリをマウント
# ビルド間で再利用される（レイヤーには含まれない）
RUN --mount=type=cache,target=/root/.npm \
    npm ci --only=production

COPY . .
RUN npm run build

CMD ["node", "dist/server.js"]
```

```dockerfile
# Python の pip キャッシュ
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```dockerfile
# Go のモジュール + ビルドキャッシュ
FROM golang:1.22-alpine
WORKDIR /app
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod \
    go mod download
COPY . .
RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -o /server .
```

BuildKit のマウントキャッシュは、レイヤーキャッシュとは異なる仕組みである。`--mount=type=cache` で指定されたディレクトリは、ビルド間で永続化されるがイメージには含まれない。これにより、パッケージマネージャーのキャッシュを効率的に再利用できる。

### 1.4 マウントキャッシュの詳細オプション

```dockerfile
# キャッシュ ID を指定（同じ ID のキャッシュを共有）
RUN --mount=type=cache,id=npm-cache,target=/root/.npm \
    npm ci

# キャッシュのシェアリングモード
# shared: 複数のビルドが同時にアクセス可能（デフォルト）
# private: 1つのビルドのみアクセス可能
# locked: 同時アクセスを排他制御
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm ci

# 読み取り専用マウント
RUN --mount=type=cache,target=/root/.npm,readonly \
    npm ls

# キャッシュの初期値をディレクトリから設定
RUN --mount=type=cache,target=/root/.npm,from=base-deps \
    npm ci
```

### 1.5 キャッシュ無効化の回避テクニック

```dockerfile
# NG: 日時を含むコマンドはキャッシュが常に無効
RUN echo "Build date: $(date)" > /app/build-info.txt

# OK: ARG で制御（同じ値ならキャッシュ有効）
ARG BUILD_DATE=unknown
RUN echo "Build date: $BUILD_DATE" > /app/build-info.txt

# NG: apt-get update を単独で実行
RUN apt-get update
RUN apt-get install -y curl  # update のキャッシュが古くなる

# OK: update と install を1つの RUN にまとめる
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# テクニック: git リビジョンをビルド引数に
ARG GIT_REVISION
LABEL git.revision=$GIT_REVISION
# build 時: docker build --build-arg GIT_REVISION=$(git rev-parse HEAD) .
```

### 1.6 条件付きキャッシュ破棄

```dockerfile
# 特定の条件でのみキャッシュを無効化する
# 例: 依存関係ファイルが変更された場合のみ再インストール

FROM node:20-alpine
WORKDIR /app

# package.json のみ先にコピー（変更がなければキャッシュが効く）
COPY package.json package-lock.json ./

# チェックサムで変更を検出
RUN --mount=type=cache,target=/root/.npm \
    npm ci

# tsconfig.json が変わっても依存関係の再インストールは不要
COPY tsconfig.json ./
COPY src/ ./src/
RUN npm run build
```

---

## 2. .dockerignore 設計

### 2.1 包括的な .dockerignore

```bash
# ==========================================
# .dockerignore
# ==========================================

# --- バージョン管理 ---
.git
.gitignore
.gitattributes

# --- 依存関係（コンテナ内で再インストール） ---
node_modules
vendor
.venv
__pycache__
*.pyc

# --- ビルド成果物（コンテナ内で再ビルド） ---
dist
build
out
target
*.o
*.a

# --- IDE / エディタ ---
.vscode
.idea
*.swp
*.swo
*~

# --- Docker 関連 ---
Dockerfile*
docker-compose*.yml
.dockerignore

# --- ドキュメント ---
README.md
LICENSE
CHANGELOG.md
docs/

# --- テスト ---
coverage
.nyc_output
*.test.js
*.spec.js
__tests__
tests

# --- 環境変数・シークレット ---
.env
.env.*
!.env.example
*.pem
*.key
credentials.json

# --- OS ファイル ---
.DS_Store
Thumbs.db

# --- CI/CD ---
.github
.gitlab-ci.yml
Jenkinsfile
```

### 2.2 .dockerignore の効果

```
+------------------------------------------------------+
|          .dockerignore 適用前後の比較                   |
|                                                      |
|  適用前:                                              |
|  $ docker build . 2>&1 | grep "Sending"             |
|  Sending build context to Docker daemon  500MB       |
|                                                      |
|  内訳:                                               |
|  +-- .git/          200 MB  ← 不要                  |
|  +-- node_modules/  280 MB  ← コンテナ内で再インストール|
|  +-- src/            10 MB  ← 必要                  |
|  +-- その他           10 MB                          |
|                                                      |
|  適用後:                                              |
|  $ docker build . 2>&1 | grep "Sending"             |
|  Sending build context to Docker daemon  15MB        |
|                                                      |
|  効果: 97% 削減、ビルド時間も大幅短縮                   |
+------------------------------------------------------+
```

### 2.3 言語別 .dockerignore テンプレート

```bash
# === Node.js プロジェクト用 ===
node_modules
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnpm-debug.log*
dist
build
.next
.nuxt
coverage
.nyc_output
*.test.js
*.spec.js
*.test.ts
*.spec.ts
__tests__
jest.config.*
.eslintrc*
.prettierrc*
tsconfig.tsbuildinfo
```

```bash
# === Python プロジェクト用 ===
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.egg-info
.eggs
*.egg
dist
build
.venv
venv
env
.mypy_cache
.pytest_cache
.tox
htmlcov
.coverage
*.cover
```

```bash
# === Go プロジェクト用 ===
vendor/
*.test
*.out
*.exe
*.dll
*.so
*.dylib
coverage.txt
profile.out
```

```bash
# === Java プロジェクト用 ===
target/
build/
*.class
*.jar
*.war
*.ear
.gradle
.mvn/wrapper/maven-wrapper.jar
*.iml
.idea
out/
```

### 2.4 .dockerignore のデバッグ

```bash
# ビルドコンテキストに含まれるファイルを確認する方法

# 1. コンテキストサイズを確認
docker build --no-cache -t test . 2>&1 | head -5

# 2. BuildKit でコンテキスト転送量を確認
DOCKER_BUILDKIT=1 docker build --progress=plain -t test . 2>&1 | grep "transferring"

# 3. .dockerignore の効果をテスト（空の Dockerfile で）
echo "FROM scratch" > Dockerfile.test
docker build -f Dockerfile.test . 2>&1 | grep "Sending"
rm Dockerfile.test

# 4. rsync --dry-run で除外ファイルを確認
rsync -avz --dry-run --exclude-from=.dockerignore . /dev/null
```

---

## 3. イメージサイズ最適化

### 3.1 ベースイメージの選択

```dockerfile
# サイズ比較用ビルド
# === ubuntu ベース (~77MB) ===
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y nodejs npm

# === slim ベース (~74MB) ===
FROM node:20-slim

# === alpine ベース (~7MB) ===
FROM node:20-alpine

# === distroless (~120MB Node.js含む) ===
FROM gcr.io/distroless/nodejs20-debian12
```

```bash
# サイズの確認
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

### 3.2 ベースイメージ詳細比較

| ベースイメージ | サイズ | C ライブラリ | パッケージマネージャ | シェル | セキュリティ | 用途 |
|---|---|---|---|---|---|---|
| ubuntu:22.04 | ~77MB | glibc | apt | bash | 低 | 汎用開発 |
| debian:bookworm-slim | ~74MB | glibc | apt | bash | 中 | 汎用サーバー |
| alpine:3.19 | ~7MB | musl | apk | ash | 高 | 軽量コンテナ |
| distroless | ~数MB | glibc | なし | なし | 最高 | 本番実行のみ |
| scratch | 0MB | なし | なし | なし | 最高 | 静的バイナリ |
| chainguard/static | ~数MB | なし | なし | なし | 最高 | Distroless 代替 |
| wolfi-base | ~12MB | glibc | apk | ash | 高 | Chainguard 推奨 |

### 3.3 パッケージのクリーンアップ

```dockerfile
# Debian/Ubuntu: キャッシュの削除
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Alpine: --no-cache で キャッシュを残さない
RUN apk add --no-cache curl ca-certificates

# pip: キャッシュを無効化
RUN pip install --no-cache-dir -r requirements.txt

# npm: キャッシュをクリア
RUN npm ci --only=production && npm cache clean --force

# 不要なファイルの削除
RUN rm -rf /tmp/* /var/tmp/* /usr/share/doc /usr/share/man
```

### 3.4 レイヤー数の最適化

```dockerfile
# NG: レイヤーが多い
FROM alpine:3.19
RUN apk add --no-cache curl
RUN apk add --no-cache git
RUN apk add --no-cache bash
RUN mkdir /app
RUN chmod 755 /app
# -> 5 レイヤー

# OK: まとめる
FROM alpine:3.19
RUN apk add --no-cache curl git bash && \
    mkdir /app && \
    chmod 755 /app
# -> 1 レイヤー
```

### 3.5 マルチステージビルドによるサイズ削減

```dockerfile
# === マルチステージビルドの典型的パターン ===

# ステージ 1: 依存関係インストール
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ステージ 2: ビルド
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ステージ 3: 本番用（最小イメージ）
FROM node:20-alpine AS production
WORKDIR /app

# 本番依存関係のみ
COPY --from=deps /app/node_modules ./node_modules
# ビルド成果物のみ
COPY --from=builder /app/dist ./dist
COPY package.json ./

RUN addgroup -S app && adduser -S app -G app
USER app

CMD ["node", "dist/server.js"]

# 結果:
# deps ステージ:    devDependencies 含む (~500MB)
# builder ステージ: ソースコード + ビルドツール含む (~600MB)
# 最終イメージ:     本番依存 + dist のみ (~150MB)
```

### 3.6 UPX によるバイナリ圧縮

```dockerfile
# Go バイナリを UPX で圧縮する例
FROM golang:1.22-alpine AS builder
RUN apk add --no-cache upx

WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -ldflags="-w -s" -o /server .

# UPX で圧縮（50-70% のサイズ削減）
RUN upx --best --lzma /server

FROM scratch
COPY --from=builder /server /server
ENTRYPOINT ["/server"]

# 圧縮前: 15MB → 圧縮後: 5MB（起動時間は微増）
```

### 3.7 不要ファイルの特定と削除

```bash
# イメージ内の大きなファイルを確認
docker run --rm myapp:latest find / -type f -size +1M -exec ls -lh {} \; 2>/dev/null

# レイヤーごとのサイズを確認
docker history myapp:latest --format "table {{.ID}}\t{{.CreatedBy}}\t{{.Size}}"

# dive ツールでレイヤーを視覚的に分析
docker run --rm -it \
    -v /var/run/docker.sock:/var/run/docker.sock \
    wagoodman/dive:latest myapp:latest
```

---

## 4. セキュリティハードニング

### 4.1 non-root ユーザー

```dockerfile
# Alpine の場合
FROM node:20-alpine
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app
COPY --chown=app:app . .
RUN npm ci --only=production
USER app
CMD ["node", "server.js"]

# Debian の場合
FROM node:20-slim
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app
WORKDIR /app
COPY --chown=app:app . .
USER app
```

### 4.2 読み取り専用ファイルシステム

```bash
# 読み取り専用で実行
docker run --read-only \
    --tmpfs /tmp:rw,size=100m \
    --tmpfs /var/run:rw \
    my-app

# docker-compose.yml での設定
# services:
#   app:
#     read_only: true
#     tmpfs:
#       - /tmp
#       - /var/run
```

### 4.3 脆弱性スキャンの組み込み

```
+------------------------------------------------------+
|         CI/CD パイプラインでのスキャンフロー             |
|                                                      |
|  [コード変更] --> [ビルド] --> [スキャン] --> [プッシュ] |
|                                  |                   |
|                            +-----+-----+             |
|                            |           |             |
|                         [Pass]      [Fail]           |
|                            |           |             |
|                         [Push]    [ブロック]          |
|                                   [通知]             |
+------------------------------------------------------+
```

```bash
# Trivy でスキャン
trivy image --severity HIGH,CRITICAL my-app:v1.0.0

# 脆弱性があればビルドを失敗させる
trivy image --exit-code 1 --severity CRITICAL my-app:v1.0.0

# Docker Scout
docker scout cves my-app:v1.0.0
docker scout recommendations my-app:v1.0.0

# Dockerfile 自体のリント
docker run --rm -i hadolint/hadolint < Dockerfile
```

### 4.4 シークレット管理

```dockerfile
# NG: 環境変数にシークレットを埋め込む（イメージに残る）
ENV DATABASE_URL=postgres://user:password@host/db
# -> docker history で見える

# NG: ARG でシークレットを渡す（ビルドキャッシュに残る可能性）
ARG SECRET_KEY
RUN echo $SECRET_KEY > /app/.secret

# OK: BuildKit シークレットマウント（イメージに残らない）
RUN --mount=type=secret,id=db_url \
    cat /run/secrets/db_url > /dev/null && \
    ./setup-database.sh

# OK: 実行時に環境変数で渡す
# docker run -e DATABASE_URL=postgres://... my-app
```

```bash
# シークレットを使ったビルド
docker build \
    --secret id=db_url,src=./db_url.txt \
    --secret id=api_key,src=./api_key.txt \
    -t my-app .
```

### 4.5 コンテナの権限制限

```dockerfile
# セキュリティ強化された Dockerfile
FROM node:20-alpine

# 不要な setuid/setgid ビットを削除
RUN find / -perm /6000 -type f -exec chmod a-s {} \; 2>/dev/null || true

# non-root ユーザー作成
RUN addgroup -S app && adduser -S app -G app

WORKDIR /app
COPY --chown=app:app . .
RUN npm ci --only=production

USER app

# ヘルスチェック（non-root でも動作するコマンド）
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

CMD ["node", "server.js"]
```

```bash
# 実行時のセキュリティオプション
docker run \
    --read-only \
    --cap-drop ALL \
    --cap-add NET_BIND_SERVICE \
    --security-opt no-new-privileges:true \
    --tmpfs /tmp:rw,noexec,nosuid,size=100m \
    --pids-limit 100 \
    --memory 512m \
    --cpus 1.0 \
    my-app:latest
```

### 4.6 SBOM（Software Bill of Materials）の生成

```bash
# Docker BuildKit による SBOM 生成
docker buildx build --sbom=true -t my-app:v1.0.0 .

# Syft で SBOM 生成
syft my-app:v1.0.0 -o spdx-json > sbom.json

# SBOM から脆弱性チェック
grype sbom:sbom.json

# Trivy で SBOM を生成
trivy image --format spdx-json --output sbom.json my-app:v1.0.0
```

### 4.7 イメージ署名と検証

```bash
# cosign でイメージに署名
cosign sign --key cosign.key myregistry/my-app:v1.0.0

# 署名の検証
cosign verify --key cosign.pub myregistry/my-app:v1.0.0

# Keyless 署名（Sigstore/Fulcio）
cosign sign myregistry/my-app:v1.0.0
# → OIDCプロバイダーで認証

# Docker Content Trust
export DOCKER_CONTENT_TRUST=1
docker push myregistry/my-app:v1.0.0  # 自動的に署名
docker pull myregistry/my-app:v1.0.0  # 署名を検証
```

---

## 5. ビルドパフォーマンス

### 5.1 BuildKit の活用

```bash
# BuildKit を有効化（Docker 23.0+ ではデフォルト）
export DOCKER_BUILDKIT=1

# 並列ビルドの確認
docker build --progress=plain -t my-app .

# ビルドキャッシュのエクスポート/インポート
docker build \
    --cache-from type=registry,ref=myregistry/my-app:cache \
    --cache-to type=registry,ref=myregistry/my-app:cache,mode=max \
    -t my-app .

# ローカルキャッシュ
docker build \
    --cache-from type=local,src=/tmp/docker-cache \
    --cache-to type=local,dest=/tmp/docker-cache \
    -t my-app .
```

### 5.2 マルチプラットフォームビルド

```bash
# buildx ビルダーの作成
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# マルチプラットフォームビルド
docker buildx build \
    --platform linux/amd64,linux/arm64 \
    -t myregistry/my-app:v1.0.0 \
    --push .
```

### 5.3 並列ステージビルド

```dockerfile
# syntax=docker/dockerfile:1

# 並列実行可能なステージ
FROM node:20-alpine AS frontend-deps
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

FROM python:3.12-slim AS backend-deps
WORKDIR /backend
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY --from=frontend-deps /frontend/node_modules ./node_modules
COPY frontend/ .
RUN npm run build

# 最終ステージで統合
FROM python:3.12-slim AS production
WORKDIR /app
COPY --from=backend-deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=frontend-build /frontend/dist ./static
COPY backend/ .
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]

# frontend-deps と backend-deps は並列でビルドされる（BuildKit）
```

### 5.4 CI でのビルドキャッシュ戦略

```yaml
# GitHub Actions でのキャッシュ設定
name: Build
on: push

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

```yaml
# GitLab CI でのキャッシュ設定
build:
  stage: build
  image: docker:24
  services:
    - docker:24-dind
  variables:
    DOCKER_BUILDKIT: "1"
  script:
    - docker build
      --cache-from type=registry,ref=$CI_REGISTRY_IMAGE:cache
      --cache-to type=registry,ref=$CI_REGISTRY_IMAGE:cache,mode=max
      -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
      --push .
```

### 5.5 ビルド時間の計測と分析

```bash
# ビルド時間を詳細表示
DOCKER_BUILDKIT=1 docker build --progress=plain -t my-app . 2>&1 | tee build.log

# 各ステージの時間を抽出
grep -E "^#[0-9]+ (DONE|CACHED)" build.log

# BuildKit のステータスを確認
docker buildx du
docker buildx prune  # 不要なキャッシュを削除

# ビルドキャッシュの使用量確認
docker system df
docker builder prune --all --force  # 全キャッシュ削除
```

---

## 6. Dockerfile リント

### 6.1 Hadolint

```bash
# Hadolint の実行
docker run --rm -i hadolint/hadolint < Dockerfile

# 出力例:
# DL3008 warning: Pin versions in apt get install
# DL3009 info: Delete the apt-get lists after installing
# DL3018 warning: Pin versions in apk add
# DL4006 warning: Set the SHELL option -o pipefail
# SC2086 info: Double quote to prevent globbing

# 特定のルールを無視
docker run --rm -i hadolint/hadolint \
    --ignore DL3008 --ignore DL3018 < Dockerfile

# .hadolint.yaml で設定
# ignored:
#   - DL3008
# trustedRegistries:
#   - docker.io
#   - ghcr.io
```

### 6.2 Hadolint の CI 統合

```yaml
# GitHub Actions での Hadolint
name: Lint Dockerfile
on: pull_request

jobs:
  hadolint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          failure-threshold: warning
```

```yaml
# .hadolint.yaml の詳細設定
ignored:
  - DL3008  # apt パッケージのバージョン未固定
  - DL3018  # apk パッケージのバージョン未固定

trustedRegistries:
  - docker.io
  - ghcr.io
  - gcr.io

override:
  error:
    - DL3001  # 不正なコマンド
    - DL3002  # root ユーザー
  warning:
    - DL3006  # FROM タグなし
  info:
    - DL3009  # apt lists 未削除
  style:
    - DL3015  # apt --no-install-recommends 未使用
```

### 比較表 1: Hadolint 主要ルール

| ルールID | 重要度 | 内容 | 対処法 |
|---|---|---|---|
| DL3006 | warning | FROM でタグ指定なし | `FROM image:tag` を使用 |
| DL3008 | warning | apt パッケージのバージョン未固定 | `apt-get install pkg=version` |
| DL3009 | info | apt-get lists 未削除 | `rm -rf /var/lib/apt/lists/*` |
| DL3018 | warning | apk パッケージのバージョン未固定 | `apk add pkg=version` |
| DL3025 | warning | CMD がシェル形式 | exec 形式 `CMD ["cmd"]` |
| DL4006 | warning | pipefail 未設定 | `SHELL ["/bin/bash", "-o", "pipefail", "-c"]` |
| DL3002 | warning | USER が root のまま | `USER nonroot` を追加 |
| DL3003 | error | sudo の使用 | non-root ユーザーに切り替え前に必要な操作を実行 |
| DL3007 | warning | FROM で latest タグ使用 | 明示的なバージョンタグを指定 |
| DL3013 | warning | pip --no-cache-dir 未使用 | `pip install --no-cache-dir` |
| DL3015 | info | apt --no-install-recommends 未使用 | `--no-install-recommends` を追加 |
| DL3020 | error | ADD の代わりに COPY を使用 | URL や tar 展開以外は COPY を使う |
| DL3028 | warning | gem --no-document 未使用 | `gem install --no-document` |

### 比較表 2: セキュリティスキャンツール比較

| ツール | 種類 | 対象 | CI統合 | 特徴 |
|---|---|---|---|---|
| Hadolint | リンター | Dockerfile | GitHub Actions, GitLab CI | Dockerfile の書き方をチェック |
| Trivy | スキャナー | イメージ, FS, リポ | 全主要CI | OSS, 高速, 包括的 |
| Docker Scout | スキャナー | イメージ | Docker Desktop | Docker 統合, SBOM |
| Snyk | スキャナー | イメージ, コード | 全主要CI | 修正提案が充実 |
| Grype | スキャナー | イメージ, FS | GitHub Actions | Anchore 製, 高速 |
| Dockle | リンター | イメージ | GitHub Actions | CIS Benchmark 準拠 |
| cosign | 署名 | イメージ | GitHub Actions | Sigstore エコシステム |
| syft | SBOM | イメージ, FS | GitHub Actions | SBOM 生成ツール |

---

## 7. ベストプラクティスチェックリスト

```
+------------------------------------------------------+
|         Dockerfile ベストプラクティス                   |
|                                                      |
|  基本                                                |
|  [x] FROM でバージョンタグを固定                       |
|  [x] .dockerignore を設定                            |
|  [x] マルチステージビルドを使用                        |
|  [x] 変更頻度の低い命令を上に配置                      |
|                                                      |
|  セキュリティ                                         |
|  [x] non-root ユーザーで実行                          |
|  [x] 最小ベースイメージを使用 (alpine/distroless)      |
|  [x] 脆弱性スキャンを CI に組み込み                    |
|  [x] シークレットをイメージに含めない                   |
|  [x] HEALTHCHECK を定義                              |
|  [x] setuid/setgid ビットを削除                       |
|  [x] --cap-drop ALL で実行                           |
|                                                      |
|  効率                                                |
|  [x] RUN 命令をまとめてレイヤー数を削減                |
|  [x] パッケージキャッシュを削除                        |
|  [x] --no-install-recommends / --no-cache を使用     |
|  [x] BuildKit マウントキャッシュを活用                 |
|  [x] 並列ステージビルドを設計                          |
|                                                      |
|  保守性                                               |
|  [x] LABEL でメタデータを付与                          |
|  [x] CMD/ENTRYPOINT は exec 形式                     |
|  [x] Hadolint でリントを実施                          |
|  [x] EXPOSE でポートをドキュメント                     |
|  [x] 環境変数にデフォルト値を設定                      |
+------------------------------------------------------+
```

### 7.1 LABEL のベストプラクティス

```dockerfile
# OCI 標準ラベル
LABEL org.opencontainers.image.title="My Application" \
      org.opencontainers.image.description="Production-ready API server" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="team@example.com" \
      org.opencontainers.image.source="https://github.com/example/my-app" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.created="2024-01-15T10:30:00Z" \
      org.opencontainers.image.revision="abc123"

# ビルド情報を動的に設定
ARG BUILD_DATE
ARG GIT_REVISION
ARG VERSION
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.revision=$GIT_REVISION \
      org.opencontainers.image.version=$VERSION
```

### 7.2 HEALTHCHECK の設計パターン

```dockerfile
# HTTP エンドポイントへのヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# TCP ポートの確認
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
    CMD nc -z localhost 8080 || exit 1

# カスタムスクリプト
COPY healthcheck.sh /usr/local/bin/
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["healthcheck.sh"]

# gRPC サービスのヘルスチェック
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD ["grpc_health_probe", "-addr=:50051"]
```

### 7.3 ENTRYPOINT と CMD の使い分け

```dockerfile
# パターン 1: CMD のみ（最もシンプル）
CMD ["node", "server.js"]
# -> docker run myapp (デフォルト実行)
# -> docker run myapp node repl (コマンド上書き)

# パターン 2: ENTRYPOINT + CMD（推奨）
ENTRYPOINT ["node"]
CMD ["server.js"]
# -> docker run myapp (node server.js を実行)
# -> docker run myapp repl (node repl を実行)

# パターン 3: entrypoint スクリプト
COPY docker-entrypoint.sh /usr/local/bin/
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "server.js"]
```

```bash
#!/bin/sh
# docker-entrypoint.sh

set -e

# 環境変数に基づく初期化処理
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    npx prisma migrate deploy
fi

# シグナル転送のために exec を使用
exec "$@"
```

---

## 8. アンチパターン

### アンチパターン 1: apt-get update と install を別レイヤーにする

```dockerfile
# NG: update と install が別レイヤー
RUN apt-get update
RUN apt-get install -y curl
# -> update のキャッシュが残り、古いパッケージリストで install される可能性

# OK: 同じ RUN にまとめる
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
```

### アンチパターン 2: ビルドツールを最終イメージに残す

```dockerfile
# NG: ビルドツールが残る
FROM python:3.12-slim
RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    pip install numpy pandas
# -> gcc, python3-dev が最終イメージに残る（数百MB）

# OK: マルチステージでビルドツールを分離
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y gcc python3-dev
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim
COPY --from=builder /install /usr/local
COPY . /app
CMD ["python", "/app/main.py"]
```

### アンチパターン 3: COPY . . を複数回実行

```dockerfile
# NG: 同じファイルを何度もコピー
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm ci
COPY . .  # <- 無意味な2回目のコピー（キャッシュも壊す）
RUN npm run build

# OK: 必要なファイルを段階的にコピー
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
```

### アンチパターン 4: ADD を COPY の代わりに使う

```dockerfile
# NG: ADD を不必要に使う
ADD ./src /app/src          # COPY で十分
ADD https://example.com/file.txt /app/  # レイヤーキャッシュが効かない

# OK: COPY を使い、URL は RUN で取得
COPY ./src /app/src
RUN curl -L -o /app/file.txt https://example.com/file.txt

# ADD が適切な場面: tar アーカイブの自動展開
ADD archive.tar.gz /app/    # 自動的に展開される
```

### アンチパターン 5: ENV で変更頻度の高い値を設定

```dockerfile
# NG: バージョン情報を ENV で設定（キャッシュが壊れる）
FROM node:20-alpine
ENV APP_VERSION=1.0.0      # 毎リリースで変更 → 以降全レイヤー再ビルド
WORKDIR /app
COPY package.json .
RUN npm ci
COPY . .

# OK: ENV は最下部に配置、または LABEL を使用
FROM node:20-alpine
WORKDIR /app
COPY package.json .
RUN npm ci
COPY . .
ARG APP_VERSION=unknown
LABEL version=$APP_VERSION
ENV APP_VERSION=$APP_VERSION
```

### アンチパターン 6: 大きなコンテキストを無視しない

```dockerfile
# NG: .dockerignore なしで node_modules を含めてしまう
FROM node:20-alpine
WORKDIR /app
COPY . .           # node_modules (300MB+) もコピーされる
RUN npm ci         # 再インストールするので完全に無駄

# OK: .dockerignore で除外 + 段階的コピー
# .dockerignore に node_modules を追加
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
```

---

## 9. 高度な最適化テクニック

### 9.1 Heredoc 構文（BuildKit）

```dockerfile
# syntax=docker/dockerfile:1

FROM debian:bookworm-slim

# Heredoc で複数行スクリプトを記述
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends curl ca-certificates
rm -rf /var/lib/apt/lists/*
EOF

# ファイル生成にも使える
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 80;
    server_name localhost;
    location / {
        proxy_pass http://app:3000;
    }
}
EOF
```

### 9.2 条件付きビルド

```dockerfile
# syntax=docker/dockerfile:1

FROM node:20-alpine AS base
WORKDIR /app

# 環境に応じて異なるビルドを実行
ARG NODE_ENV=production

FROM base AS development
RUN npm install
CMD ["npm", "run", "dev"]

FROM base AS production
RUN npm ci --only=production
CMD ["node", "dist/server.js"]

# ビルド時にターゲットを指定
# docker build --target development -t my-app:dev .
# docker build --target production -t my-app:prod .
```

### 9.3 外部イメージからのファイルコピー

```dockerfile
FROM node:20-alpine

# 外部イメージから直接ファイルをコピー
COPY --from=busybox:latest /bin/wget /usr/local/bin/wget
COPY --from=ghcr.io/grpc-ecosystem/grpc-health-probe:v0.4.25 \
    /ko-app/grpc-health-probe /usr/local/bin/grpc_health_probe

# 別のイメージからバイナリを取得するパターン
COPY --from=minio/mc:latest /usr/bin/mc /usr/local/bin/mc
```

### 9.4 ビルド引数による柔軟な Dockerfile

```dockerfile
# syntax=docker/dockerfile:1

# ベースイメージを引数で切り替え
ARG BASE_IMAGE=node:20-alpine
FROM ${BASE_IMAGE}

ARG NODE_ENV=production
ARG PORT=3000
ARG LOG_LEVEL=info

WORKDIR /app

# 条件に応じたインストール
COPY package.json package-lock.json ./
RUN if [ "$NODE_ENV" = "development" ]; then \
        npm install; \
    else \
        npm ci --only=production; \
    fi

COPY . .

ENV NODE_ENV=$NODE_ENV \
    PORT=$PORT \
    LOG_LEVEL=$LOG_LEVEL

EXPOSE $PORT
CMD ["node", "server.js"]
```

---

## 10. イメージの継続的最適化

### 10.1 定期的なベースイメージ更新

```bash
# ベースイメージの更新確認
docker pull node:20-alpine
docker images --digests node:20-alpine

# Dependabot / Renovate Bot で自動化
# renovate.json の例:
# {
#   "docker": {
#     "fileMatch": ["Dockerfile$"],
#     "pinDigests": true
#   }
# }
```

```dockerfile
# ダイジェスト固定でベースイメージを指定（最高の再現性）
FROM node:20-alpine@sha256:abc123def456...
```

### 10.2 イメージサイズの監視

```bash
# イメージサイズの推移を記録
docker images myapp --format "{{.Tag}}\t{{.Size}}" | sort -V

# CI でサイズチェック
MAX_SIZE_MB=200
SIZE=$(docker image inspect myapp:latest --format '{{.Size}}')
SIZE_MB=$((SIZE / 1024 / 1024))
if [ $SIZE_MB -gt $MAX_SIZE_MB ]; then
    echo "ERROR: Image size ($SIZE_MB MB) exceeds limit ($MAX_SIZE_MB MB)"
    exit 1
fi
```

### 10.3 レイヤー分析ツール

```bash
# dive でレイヤーを分析
dive myapp:latest

# docker history で各レイヤーのサイズを確認
docker history --no-trunc --format "table {{.Size}}\t{{.CreatedBy}}" myapp:latest

# buildctl で詳細なビルド情報を取得
docker buildx build --progress=plain --metadata-file build-metadata.json -t myapp .
```

---

## 11. FAQ

### Q1: Alpine と Debian slim のどちらを選ぶべきですか？

**A:** Alpine（musl libc）はサイズが非常に小さい（~7MB）が、glibc ベースのバイナリとの互換性問題が起きることがある。特に Python のネイティブ拡張（numpy 等）や Node.js のネイティブモジュールでビルドに時間がかかったり失敗することがある。互換性問題がなければ Alpine、問題がある場合は Debian slim を選ぶ。Go や Rust のように静的リンクするバイナリには Alpine が最適。

### Q2: レイヤーキャッシュがCIで効かないのですが？

**A:** CI 環境は通常ステートレスなため、ビルドごとにキャッシュが失われる。対策として以下がある:
- **レジストリキャッシュ**: `--cache-from type=registry` で前回のイメージをキャッシュとして利用
- **GitHub Actions Cache**: `docker/build-push-action` の cache 機能を利用
- **BuildKit のリモートキャッシュ**: `--cache-to` / `--cache-from` でキャッシュを永続化
これらを設定することで CI でも 50-80% 程度のキャッシュヒット率を達成できる。

### Q3: HEALTHCHECK はどのように設定すべきですか？

**A:** アプリケーションの `/health` エンドポイントに対してチェックを行うのが一般的。設定のポイントは:
- **interval**: 30秒程度（頻繁すぎるとオーバーヘッド）
- **timeout**: 5秒（レスポンスが返らない場合のタイムアウト）
- **retries**: 3回（一時的な障害を許容）
- **start-period**: アプリの起動時間（Java なら 60 秒等）
curl が使えない場合は wget や専用のヘルスチェックバイナリを使う。

### Q4: distroless イメージのデバッグはどうすればよいですか？

**A:** distroless にはシェルがないためデバッグが困難である。以下のアプローチがある:
- **debug バリアント**: `gcr.io/distroless/base:debug` には busybox シェルが含まれる
- **ephemeral コンテナ**: `kubectl debug` で一時的なデバッグコンテナをアタッチする（Kubernetes）
- **docker exec の代替**: `docker cp` でファイルをコピーして確認する
- **マルチステージの活用**: 開発用ステージでは Alpine を使い、本番のみ distroless にする

### Q5: Docker イメージのダイジェスト固定は必要ですか？

**A:** セキュリティとレプロダクタビリティの観点では推奨される。`node:20-alpine` のようなタグは上書き可能で、同じタグで異なるイメージが配布される可能性がある。`node:20-alpine@sha256:...` のようにダイジェストを固定すると、完全に同一のイメージが保証される。ただし、セキュリティパッチの自動適用が阻害されるため、Renovate / Dependabot による自動更新と組み合わせるのが実務的なベストプラクティスである。

### Q6: マルチステージビルドのステージ数に制限はありますか？

**A:** Dockerfile の仕様上、ステージ数に上限はない。ただし実務的には 3-5 ステージが一般的である（依存関係、ビルド、テスト、本番）。ステージが多すぎると Dockerfile の可読性が下がるため、複雑な場合は別の Dockerfile に分割するか、ビルドスクリプトで管理することを検討する。

### Q7: BuildKit のシークレットマウントと環境変数の使い分けは？

**A:** ビルド時にのみ必要なシークレット（プライベートレジストリの認証トークンなど）は `--mount=type=secret` で渡すべきである。これはイメージのレイヤーに残らないため安全である。実行時に必要なシークレット（DB パスワードなど）は `docker run -e` や Docker Secrets、Kubernetes Secrets で実行時に注入する。`ARG` や `ENV` でシークレットを渡すと `docker history` で確認できてしまうため、絶対に使用しない。

---

## 12. まとめ

| 項目 | ポイント |
|---|---|
| キャッシュ戦略 | 変更頻度の低い命令を上に、高い命令を下に配置 |
| .dockerignore | node_modules, .git, .env 等を除外してコンテキストを最小化 |
| ベースイメージ | Alpine/slim/distroless を用途に応じて選択 |
| セキュリティ | non-root, 脆弱性スキャン, シークレットマウント, SBOM |
| リント | Hadolint でベストプラクティス違反を自動検出 |
| BuildKit | マウントキャッシュ、シークレット、並列ビルドを活用 |
| CI/CD | レジストリキャッシュで CI のビルド時間を短縮 |
| マルチプラットフォーム | buildx で AMD64/ARM64 対応イメージを構築 |
| 署名と検証 | cosign/Docker Content Trust でイメージの信頼性を保証 |
| 継続的最適化 | イメージサイズの監視、ベースイメージの自動更新 |

---

## 次に読むべきガイド

- [03-language-specific.md](./03-language-specific.md) -- 言語別 Dockerfile テンプレート集
- [../02-compose/00-compose-basics.md](../02-compose/00-compose-basics.md) -- Docker Compose の基礎
- [../02-compose/02-development-workflow.md](../02-compose/02-development-workflow.md) -- Compose 開発ワークフロー

---

## 参考文献

1. **Docker Documentation - Build best practices** https://docs.docker.com/build/building/best-practices/ -- Docker 公式のビルドベストプラクティス。
2. **Hadolint** https://github.com/hadolint/hadolint -- Dockerfile リンターの公式リポジトリ。全ルールの説明と設定方法。
3. **Aqua Security - Trivy** https://aquasecurity.github.io/trivy/ -- 脆弱性スキャナーの公式ドキュメント。CI 統合の設定例も充実。
4. **Sysdig - Dockerfile Best Practices** https://sysdig.com/blog/dockerfile-best-practices/ -- セキュリティ観点からの Dockerfile ベストプラクティス。
5. **Docker BuildKit** https://docs.docker.com/build/buildkit/ -- BuildKit の機能と設定の公式ドキュメント。
6. **Sigstore - cosign** https://docs.sigstore.dev/cosign/overview/ -- コンテナイメージの署名と検証ツール。
7. **dive** https://github.com/wagoodman/dive -- Docker イメージのレイヤー分析ツール。
8. **Chainguard Images** https://www.chainguard.dev/chainguard-images -- セキュリティに特化した最小コンテナイメージ。
