# Dockerfile 最適化

> レイヤーキャッシュの活用、.dockerignore の設計、セキュリティスキャン、ベストプラクティスを網羅し、本番品質のコンテナイメージを構築する。

---

## この章で学ぶこと

1. **レイヤーキャッシュの仕組み**を深く理解し、ビルド時間を最小化する戦略を実装できる
2. **セキュリティスキャンとハードニング**を実施し、脆弱性の少ないイメージを構築できる
3. **Dockerfile のベストプラクティス**を体系的に適用し、保守性・効率性の高いイメージを作成できる

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

### 3.2 パッケージのクリーンアップ

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

### 3.3 レイヤー数の最適化

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

### 比較表 2: セキュリティスキャンツール比較

| ツール | 種類 | 対象 | CI統合 | 特徴 |
|---|---|---|---|---|
| Hadolint | リンター | Dockerfile | GitHub Actions, GitLab CI | Dockerfile の書き方をチェック |
| Trivy | スキャナー | イメージ, FS, リポ | 全主要CI | OSS, 高速, 包括的 |
| Docker Scout | スキャナー | イメージ | Docker Desktop | Docker 統合, SBOM |
| Snyk | スキャナー | イメージ, コード | 全主要CI | 修正提案が充実 |
| Grype | スキャナー | イメージ, FS | GitHub Actions | Anchore 製, 高速 |
| Dockle | リンター | イメージ | GitHub Actions | CIS Benchmark 準拠 |

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
|                                                      |
|  効率                                                |
|  [x] RUN 命令をまとめてレイヤー数を削減                |
|  [x] パッケージキャッシュを削除                        |
|  [x] --no-install-recommends / --no-cache を使用     |
|  [x] BuildKit マウントキャッシュを活用                 |
|                                                      |
|  保守性                                               |
|  [x] LABEL でメタデータを付与                          |
|  [x] CMD/ENTRYPOINT は exec 形式                     |
|  [x] Hadolint でリントを実施                          |
|  [x] EXPOSE でポートをドキュメント                     |
+------------------------------------------------------+
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

---

## 9. FAQ

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

---

## 10. まとめ

| 項目 | ポイント |
|---|---|
| キャッシュ戦略 | 変更頻度の低い命令を上に、高い命令を下に配置 |
| .dockerignore | node_modules, .git, .env 等を除外してコンテキストを最小化 |
| ベースイメージ | Alpine/slim/distroless を用途に応じて選択 |
| セキュリティ | non-root, 脆弱性スキャン, シークレットマウント |
| リント | Hadolint でベストプラクティス違反を自動検出 |
| BuildKit | マウントキャッシュ、シークレット、並列ビルドを活用 |
| CI/CD | レジストリキャッシュで CI のビルド時間を短縮 |

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
