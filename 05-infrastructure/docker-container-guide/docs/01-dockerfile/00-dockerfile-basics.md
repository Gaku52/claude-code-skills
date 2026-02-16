# Dockerfile 基礎

> Dockerfile の基本命令（FROM, RUN, COPY, CMD, ENTRYPOINT）、レイヤー構造、ビルドコンテキストを理解し、再現性のあるコンテナイメージを構築する。

---

## この章で学ぶこと

1. **Dockerfile の主要命令**を理解し、目的に応じた命令を選択できる
2. **レイヤー構造とビルドキャッシュ**の仕組みを把握し、効率的なビルドができる
3. **ビルドコンテキストの最適化**により、高速かつ安全なイメージビルドを実践できる
4. **BuildKit の拡張機能**を活用し、シークレット管理やキャッシュマウント等の高度なビルドを実装できる

---

## 1. Dockerfile とは

Dockerfile はコンテナイメージを構築するための命令書である。テキストファイルとして管理でき、バージョン管理・レビュー・自動化が可能になる。Dockerfile を使うことで「Infrastructure as Code」の原則に従い、再現性のあるイメージ構築が実現する。

### 1.1 基本的な Dockerfile

```dockerfile
# ベースイメージの指定
FROM node:20-alpine

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係ファイルのコピー
COPY package.json package-lock.json ./

# 依存関係のインストール
RUN npm ci --only=production

# アプリケーションコードのコピー
COPY . .

# ポートの公開（ドキュメント用）
EXPOSE 3000

# コンテナ起動時のコマンド
CMD ["node", "server.js"]
```

### 1.2 ビルドと実行

```bash
# イメージのビルド
docker build -t my-app:v1.0.0 .

# -t: タグ名を指定
# . : ビルドコンテキスト（現在のディレクトリ）

# ビルドしたイメージで実行
docker run -d -p 3000:3000 my-app:v1.0.0

# ビルド時にビルド引数を渡す
docker build --build-arg NODE_ENV=production -t my-app:v1.0.0 .

# Dockerfile のパスを明示的に指定
docker build -f docker/Dockerfile.production -t my-app:prod .

# ビルドの進捗を詳細に表示（BuildKit）
DOCKER_BUILDKIT=1 docker build --progress=plain -t my-app:v1.0.0 .

# ビルドキャッシュを使わずにフルビルド
docker build --no-cache -t my-app:v1.0.0 .

# 特定のステージまでビルド（マルチステージ用）
docker build --target builder -t my-app-builder .
```

### 1.3 Dockerfile の命名規則とディレクトリ構成

```
プロジェクト/
├── Dockerfile              # デフォルトの Dockerfile
├── Dockerfile.dev          # 開発用
├── Dockerfile.test         # テスト用
├── docker/
│   ├── Dockerfile.api      # API サーバー用
│   ├── Dockerfile.worker   # ワーカー用
│   └── Dockerfile.nginx    # リバースプロキシ用
├── .dockerignore           # ビルドコンテキストの除外設定
├── docker-compose.yml      # Compose 設定
└── src/
    └── ...
```

---

## 2. 主要命令

### 2.1 FROM - ベースイメージ

```dockerfile
# 公式イメージを使用
FROM ubuntu:22.04

# Alpine ベース（軽量）
FROM node:20-alpine

# Distroless（最小構成）
FROM gcr.io/distroless/static-debian12

# scratch（空のベース、Goバイナリ等に）
FROM scratch

# 特定のダイジェストで固定（完全な再現性）
FROM node:20-alpine@sha256:abc123def456...

# ビルドステージに名前を付ける（マルチステージ用）
FROM node:20-alpine AS builder

# ビルド引数でベースイメージを動的に指定
ARG BASE_IMAGE=node:20-alpine
FROM ${BASE_IMAGE}
```

#### ベースイメージ選択ガイド

```
+------------------------------------------------------+
|           ベースイメージの選択基準                      |
|                                                      |
|  用途に応じた選択:                                     |
|                                                      |
|  [最小 / 静的バイナリ]                                |
|  scratch       -> Go, Rust の静的バイナリ専用          |
|                   シェルなし、パッケージマネージャなし   |
|                                                      |
|  [最小 / ランタイム必要]                               |
|  distroless    -> Node.js, Java, Python のランタイム   |
|                   シェルなし (debug タグにはあり)       |
|                                                      |
|  [軽量 / 汎用]                                        |
|  alpine        -> 7MB。apk パッケージマネージャ       |
|                   musl libc (glibc 依存に注意)        |
|                                                      |
|  [標準 / 互換性重視]                                   |
|  debian-slim   -> 74MB。apt パッケージマネージャ       |
|                   glibc。互換性問題が少ない            |
|                                                      |
|  [フル / 開発向け]                                     |
|  ubuntu/debian -> 77MB+。開発ツール豊富               |
|                   本番環境には過大                      |
+------------------------------------------------------+
```

### 2.2 RUN - コマンド実行

```dockerfile
# シェル形式（/bin/sh -c で実行）
RUN apt-get update && apt-get install -y curl

# exec 形式（シェルを介さない）
RUN ["apt-get", "update"]

# 複数コマンドを1つのRUNにまとめる（レイヤー削減）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git && \
    rm -rf /var/lib/apt/lists/*

# Alpine の場合
RUN apk add --no-cache curl git

# BuildKit: キャッシュマウント（パッケージキャッシュの再利用）
RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y --no-install-recommends curl

# BuildKit: シークレットマウント（認証情報をレイヤーに残さない）
RUN --mount=type=secret,id=npmrc,target=/root/.npmrc \
    npm ci

# BuildKit: バインドマウント（一時的なファイル参照）
RUN --mount=type=bind,source=scripts/setup.sh,target=/tmp/setup.sh \
    bash /tmp/setup.sh

# heredoc 構文（BuildKit、Docker 1.5+）
RUN <<EOF
apt-get update
apt-get install -y curl git
rm -rf /var/lib/apt/lists/*
EOF
```

#### パッケージインストールのベストプラクティス

```dockerfile
# Debian / Ubuntu の場合
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # ネットワークツール
        curl \
        wget \
        ca-certificates \
        # ビルドツール（必要な場合のみ）
        gcc \
        make \
        # ランタイム依存
        libpq5 && \
    # キャッシュの削除（レイヤーサイズ削減）
    rm -rf /var/lib/apt/lists/* && \
    # APT キャッシュの削除
    apt-get clean

# Alpine の場合
RUN apk add --no-cache \
        curl \
        wget \
        ca-certificates \
        # ビルド依存は --virtual でグループ化
    && apk add --no-cache --virtual .build-deps \
        gcc \
        musl-dev \
        python3-dev \
    # ビルド後にまとめて削除
    && pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps
```

### 2.3 COPY と ADD

```dockerfile
# 基本的なファイルコピー（推奨）
COPY package.json /app/
COPY src/ /app/src/

# ワイルドカード
COPY *.json /app/

# --chown でオーナーを指定
COPY --chown=node:node . /app/

# --chmod でパーミッションを指定 (BuildKit)
COPY --chmod=755 entrypoint.sh /usr/local/bin/

# --link でレイヤーを独立化（並列ビルド高速化、BuildKit）
COPY --link package.json /app/

# ADD - URL からダウンロード（非推奨、curl + RUN を使うべき）
ADD https://example.com/file.tar.gz /tmp/

# ADD - tar の自動展開
ADD archive.tar.gz /app/
# -> /app/ に展開される

# 基本的に COPY を使い、tar 展開が必要な場合のみ ADD を使う
```

#### COPY と ADD の使い分け判断フロー

```
+------------------------------------------------------+
|          COPY / ADD の使い分け                         |
|                                                      |
|  ファイルコピー？                                      |
|  └── はい → COPY を使用                               |
|                                                      |
|  tar アーカイブの自動展開が必要？                       |
|  └── はい → ADD を使用                                |
|                                                      |
|  URL からファイルをダウンロード？                       |
|  └── はい → RUN curl/wget + COPY を使用               |
|        (ADD のURL機能は非推奨)                        |
|                                                      |
|  理由: COPY は動作が予測可能で明示的。                  |
|        ADD は tar 展開等の暗黙的な動作があり            |
|        意図しない結果を招くことがある。                  |
+------------------------------------------------------+
```

### 2.4 WORKDIR - 作業ディレクトリ

```dockerfile
# 作業ディレクトリの設定
WORKDIR /app

# 存在しない場合は自動的に作成される
WORKDIR /app/src/components

# 相対パスも可能（前の WORKDIR からの相対）
WORKDIR /app
WORKDIR src     # -> /app/src
WORKDIR tests   # -> /app/src/tests

# 環境変数を使用
ENV APP_HOME=/opt/myapp
WORKDIR $APP_HOME
```

### 2.5 CMD と ENTRYPOINT

```
+------------------------------------------------------+
|          CMD と ENTRYPOINT の関係                      |
|                                                      |
|  ENTRYPOINT = 実行するコマンド（固定部分）              |
|  CMD        = デフォルト引数（上書き可能）              |
|                                                      |
|  例: ENTRYPOINT ["python"] + CMD ["app.py"]          |
|                                                      |
|  docker run my-app                                   |
|  -> python app.py                                    |
|                                                      |
|  docker run my-app test.py                           |
|  -> python test.py  (CMD が上書きされる)              |
|                                                      |
|  docker run --entrypoint sh my-app                   |
|  -> sh  (ENTRYPOINT が上書きされる)                   |
+------------------------------------------------------+
```

```dockerfile
# CMD のみ（最も一般的）
CMD ["node", "server.js"]

# docker run my-app           -> node server.js
# docker run my-app bash      -> bash (CMD が上書きされる)

# ENTRYPOINT + CMD（推奨パターン）
ENTRYPOINT ["python"]
CMD ["app.py"]

# docker run my-app           -> python app.py
# docker run my-app test.py   -> python test.py

# ENTRYPOINT のみ（引数必須）
ENTRYPOINT ["curl"]

# docker run my-app https://example.com -> curl https://example.com

# シェル形式（非推奨 - シグナルが正しく伝わらない）
CMD node server.js
# -> /bin/sh -c "node server.js" として実行される
```

#### ENTRYPOINT スクリプトパターン

```dockerfile
# entrypoint スクリプトを使用するパターン
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["server"]
```

```bash
#!/bin/sh
# docker-entrypoint.sh
set -e

# 初期化処理
echo "Starting application..."
echo "Environment: ${APP_ENV:-development}"

# データベースマイグレーション（必要に応じて）
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    python manage.py migrate
fi

# 引数に応じた処理分岐
case "$1" in
    server)
        echo "Starting web server..."
        exec gunicorn --bind 0.0.0.0:8000 app:app
        ;;
    worker)
        echo "Starting background worker..."
        exec celery -A tasks worker
        ;;
    shell)
        exec /bin/sh
        ;;
    *)
        # 引数をそのまま実行
        exec "$@"
        ;;
esac
```

```bash
# 使用例
docker run my-app                    # -> gunicorn 起動（デフォルト: server）
docker run my-app worker             # -> celery worker 起動
docker run my-app shell              # -> シェル起動
docker run my-app python script.py   # -> python script.py 実行
```

### 2.6 ENV - 環境変数

```dockerfile
# ENV - 環境変数（ビルド時 + 実行時に有効）
ENV NODE_ENV=production
ENV APP_PORT=3000

# 複数の環境変数を1行で（レガシー構文）
ENV NODE_ENV=production APP_PORT=3000

# 後の命令で環境変数を参照
ENV APP_HOME=/app
WORKDIR $APP_HOME
COPY . $APP_HOME

# 環境変数の上書き（docker run 時）
# docker run -e NODE_ENV=development my-app
```

### 2.7 ARG - ビルド時変数

```dockerfile
# ARG - ビルド時変数（実行時には残らない）
ARG NODE_VERSION=20
FROM node:${NODE_VERSION}-alpine

# FROM 後に再度宣言が必要（FROM でスコープがリセットされる）
ARG BUILD_DATE
ARG VCS_REF

LABEL build-date=${BUILD_DATE}
LABEL vcs-ref=${VCS_REF}

# docker build \
#   --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
#   --build-arg VCS_REF=$(git rev-parse --short HEAD) \
#   .

# ARG のデフォルト値と上書き
ARG APP_ENV=production
RUN echo "Building for ${APP_ENV}"
# docker build --build-arg APP_ENV=staging .
```

#### ENV と ARG の違い

```
+------------------------------------------------------+
|          ENV vs ARG の比較                             |
|                                                      |
|  ARG:                                                 |
|  - ビルド時のみ有効                                    |
|  - docker build --build-arg で上書き可能               |
|  - FROM の前でも宣言可能（FROM のイメージ指定に使える）  |
|  - FROM 後に再宣言が必要                               |
|  - 最終イメージのメタデータに残らない                    |
|                                                      |
|  ENV:                                                 |
|  - ビルド時 + 実行時に有効                              |
|  - docker run -e で上書き可能                          |
|  - 最終イメージのメタデータに含まれる                    |
|  - docker inspect で確認可能                           |
|                                                      |
|  セキュリティ注意:                                     |
|  - ARG の値は docker history で確認できる場合がある     |
|  - パスワード等は ARG/ENV ではなく                      |
|    --mount=type=secret を使うべき                      |
+------------------------------------------------------+
```

### 2.8 EXPOSE - ポート公開

```dockerfile
# EXPOSE - ポートのドキュメント（実際のポート公開は -p で行う）
EXPOSE 3000
EXPOSE 8080/tcp
EXPOSE 53/udp

# 複数ポートの公開
EXPOSE 80 443
```

### 2.9 LABEL - メタデータ

```dockerfile
# LABEL - メタデータ
LABEL maintainer="team@example.com"
LABEL version="1.0.0"
LABEL description="My application"

# OCI Image Spec 準拠のラベル
LABEL org.opencontainers.image.title="My App"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="team@example.com"
LABEL org.opencontainers.image.source="https://github.com/org/repo"
LABEL org.opencontainers.image.licenses="MIT"

# 複数ラベルを1つの LABEL 命令で
LABEL \
    org.opencontainers.image.title="My App" \
    org.opencontainers.image.version="1.0.0" \
    org.opencontainers.image.authors="team@example.com"
```

### 2.10 USER - 実行ユーザーの切り替え

```dockerfile
# non-root ユーザーの作成（Alpine）
RUN addgroup -S appgroup && adduser -S appuser -G appgroup

# non-root ユーザーの作成（Debian/Ubuntu）
RUN groupadd -r appgroup && useradd --no-log-init -r -g appgroup appuser

# ユーザーの切り替え
USER appuser

# UID/GID での指定
USER 1001:1001

# node イメージの場合（node ユーザーが事前定義されている）
USER node
```

### 2.11 HEALTHCHECK - ヘルスチェック

```dockerfile
# HTTP エンドポイントでのヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:3000/health || exit 1

# wget を使用（Alpine の場合、curl がない場合）
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

# カスタムヘルスチェックスクリプト
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD /usr/local/bin/healthcheck.sh || exit 1

# ヘルスチェックを無効化（ベースイメージで設定されている場合）
HEALTHCHECK NONE
```

```
+------------------------------------------------------+
|          HEALTHCHECK パラメータ                        |
|                                                      |
|  --interval=30s   : チェック間隔（デフォルト: 30s）    |
|  --timeout=5s     : タイムアウト（デフォルト: 30s）    |
|  --retries=3      : 失敗判定までの回数（デフォルト: 3）|
|  --start-period=0 : 初期化猶予期間（デフォルト: 0s）   |
|                                                      |
|  戻り値:                                              |
|  0 = healthy（正常）                                   |
|  1 = unhealthy（異常）                                 |
|                                                      |
|  docker ps での表示:                                   |
|  STATUS: Up 5 minutes (healthy)                      |
|  STATUS: Up 5 minutes (unhealthy)                    |
+------------------------------------------------------+
```

### 2.12 VOLUME - ボリュームマウントポイント

```dockerfile
# VOLUME - ボリュームマウントポイント
VOLUME ["/data", "/logs"]

# 単一ボリューム
VOLUME /var/lib/postgresql/data

# 注意: VOLUME 命令で宣言されたパスは、それ以降の
# Dockerfile 命令でファイルを変更しても反映されない場合がある
```

### 2.13 SHELL - デフォルトシェルの変更

```dockerfile
# SHELL - デフォルトシェルの変更
SHELL ["/bin/bash", "-c"]

# PowerShell（Windows コンテナ）
SHELL ["powershell", "-Command"]
RUN Get-ChildItem

# bash 特有の機能を使う場合
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN curl -fsSL https://example.com/script.sh | bash
```

### 2.14 STOPSIGNAL - 停止シグナル

```dockerfile
# デフォルトは SIGTERM
STOPSIGNAL SIGTERM

# SIGQUIT を使う場合（nginx のグレースフルシャットダウン）
STOPSIGNAL SIGQUIT

# 数値でも指定可能
STOPSIGNAL 9  # SIGKILL
```

---

## 3. レイヤー構造

### 3.1 レイヤーの生成

```
+------------------------------------------------------+
|              Dockerfile -> レイヤー                    |
|                                                      |
|  FROM node:20-alpine     -> ベースイメージレイヤー     |
|  WORKDIR /app            -> メタデータのみ(レイヤーなし)|
|  COPY package.json .     -> 新規レイヤー (Layer A)    |
|  RUN npm ci              -> 新規レイヤー (Layer B)    |
|  COPY . .                -> 新規レイヤー (Layer C)    |
|  EXPOSE 3000             -> メタデータのみ(レイヤーなし)|
|  CMD ["node","server.js"]-> メタデータのみ(レイヤーなし)|
|                                                      |
|  レイヤーを生成する命令: FROM, RUN, COPY, ADD          |
|  メタデータのみの命令: WORKDIR, EXPOSE, ENV,           |
|                        CMD, ENTRYPOINT, LABEL 等     |
+------------------------------------------------------+
```

### 3.2 ビルドキャッシュ

```
+------------------------------------------------------+
|              ビルドキャッシュの仕組み                    |
|                                                      |
|  1回目のビルド:                                       |
|  FROM node:20-alpine    [実行] ----+                 |
|  COPY package.json .    [実行] ----|-- キャッシュ保存  |
|  RUN npm ci             [実行] ----+                 |
|  COPY . .               [実行] ----+                 |
|                                                      |
|  2回目のビルド (ソースコードのみ変更):                   |
|  FROM node:20-alpine    [キャッシュ利用]               |
|  COPY package.json .    [キャッシュ利用] <- 変更なし    |
|  RUN npm ci             [キャッシュ利用] <- 変更なし    |
|  COPY . .               [再実行] <- ここから再ビルド   |
|                                                      |
|  重要: キャッシュが無効になると、それ以降の全レイヤーが   |
|        再ビルドされる（キャッシュの連鎖破壊）            |
+------------------------------------------------------+
```

#### キャッシュ無効化の条件

```
+------------------------------------------------------+
|          キャッシュが無効化される条件                    |
|                                                      |
|  1. FROM: ベースイメージが変更された場合                |
|                                                      |
|  2. RUN: コマンド文字列が変更された場合                 |
|     - "RUN apt-get install -y curl"                  |
|       -> curl のバージョンが変わっても                  |
|          コマンド文字列が同じならキャッシュ有効           |
|     - キャッシュを無効化するには --no-cache を使う       |
|                                                      |
|  3. COPY/ADD: ファイルの内容（チェックサム）が変更       |
|     - ファイルの中身のハッシュで判定                     |
|     - タイムスタンプやパーミッションは無視               |
|                                                      |
|  4. ARG: ビルド引数の値が変更された場合                 |
|     - ARG を使う命令以降のキャッシュが無効化             |
|                                                      |
|  5. 親レイヤー: 上位レイヤーのキャッシュが無効になると    |
|     それ以降の全レイヤーのキャッシュも無効               |
+------------------------------------------------------+
```

```bash
# キャッシュを使ったビルド（デフォルト）
docker build -t my-app .

# キャッシュを無視して完全リビルド
docker build --no-cache -t my-app .

# 特定のステージまでビルド
docker build --target builder -t my-app-builder .

# BuildKit でキャッシュの詳細を表示
DOCKER_BUILDKIT=1 docker build --progress=plain -t my-app .

# 外部キャッシュソースの利用
docker build --cache-from my-app:latest -t my-app:v2.0.0 .

# BuildKit インラインキャッシュ（レジストリにキャッシュ情報を埋め込む）
docker build --build-arg BUILDKIT_INLINE_CACHE=1 -t my-app:latest .
docker push my-app:latest
# 次回ビルド時にキャッシュとして利用
docker build --cache-from my-app:latest -t my-app:v2.0.0 .
```

---

## 4. ビルドコンテキスト

### 4.1 ビルドコンテキストとは

```
+------------------------------------------------------+
|              ビルドコンテキスト                         |
|                                                      |
|  docker build -t my-app .                            |
|                          ^                           |
|                          |                           |
|                    ビルドコンテキスト                   |
|                    (この例ではカレントディレクトリ)       |
|                                                      |
|  プロジェクト/                                        |
|  +-- src/                                            |
|  |   +-- app.js          <- COPY で使える            |
|  +-- package.json        <- COPY で使える            |
|  +-- Dockerfile                                      |
|  +-- .dockerignore       <- 除外ルール               |
|  +-- node_modules/       <- 除外すべき               |
|  +-- .git/               <- 除外すべき               |
|  +-- .env                <- 除外すべき(機密情報)      |
|                                                      |
|  ビルドコンテキスト全体が Docker デーモンに送信される    |
|  -> 不要ファイルは .dockerignore で除外する            |
+------------------------------------------------------+
```

### 4.2 .dockerignore

```bash
# .dockerignore ファイルの例

# バージョン管理
.git
.gitignore
.gitattributes

# 依存関係（コンテナ内で再インストールするため）
node_modules
vendor
__pycache__
*.pyc

# ビルド成果物
dist
build
coverage
.next

# 環境設定・機密情報
.env
.env.*
*.pem
*.key
credentials.json

# Docker 関連（二重コピーを避ける）
Dockerfile
Dockerfile.*
docker-compose*.yml
.dockerignore

# IDE / エディタ
.vscode
.idea
*.swp
*.swo
*~

# ドキュメント
README.md
LICENSE
CHANGELOG.md
docs/

# テスト
tests/
test/
__tests__
*.test.js
*.spec.js
.nyc_output
jest.config.js

# CI/CD
.github
.gitlab-ci.yml
.circleci
Makefile

# OS 生成ファイル
.DS_Store
Thumbs.db
```

```bash
# ビルドコンテキストのサイズ確認
docker build -t my-app . 2>&1 | grep "Sending build context"
# Sending build context to Docker daemon  2.048kB  <- 小さいほど良い

# .dockerignore なしの場合
# Sending build context to Docker daemon  500MB  <- node_modules 等が含まれている

# .dockerignore のテスト（何が送信されるか確認）
# BuildKit の場合、不要なファイルはそもそも送信されない
```

### 4.3 リモートビルドコンテキスト

```bash
# Git リポジトリを直接ビルド
docker build https://github.com/user/repo.git#main

# 特定のディレクトリを指定
docker build https://github.com/user/repo.git#main:docker

# tar アーカイブからビルド
docker build - < archive.tar.gz

# stdin からの Dockerfile でビルド（コンテキストなし）
echo "FROM alpine" | docker build -t test -

# stdin からの Dockerfile + ローカルコンテキスト
docker build -f - . <<EOF
FROM alpine
COPY . /app
EOF
```

---

## 5. 実践例

### 5.1 Express.js アプリケーション

```dockerfile
FROM node:20-alpine

# セキュリティ: non-root ユーザー
RUN addgroup -S app && adduser -S app -G app

# tini（PID 1 問題の解決）
RUN apk add --no-cache tini

WORKDIR /app

# 依存関係を先にコピー（キャッシュ効率化）
COPY package.json package-lock.json ./
RUN npm ci --only=production && npm cache clean --force

# アプリケーションコードをコピー
COPY --chown=app:app . .

USER app

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["node", "server.js"]
```

### 5.2 Python Flask アプリケーション

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# システム依存関係
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Python 依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# non-root ユーザー
RUN useradd --create-home appuser
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

### 5.3 Go アプリケーション

```dockerfile
FROM golang:1.22-alpine AS builder

RUN apk add --no-cache ca-certificates tzdata
RUN adduser -D -g '' appuser

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /server ./cmd/server

# 最小の実行環境
FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /server /server
USER appuser
EXPOSE 8080
ENTRYPOINT ["/server"]
```

### 5.4 静的 Web サイト (nginx)

```dockerfile
FROM nginx:alpine

# カスタム設定
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 静的ファイル
COPY dist/ /usr/share/nginx/html/

# セキュリティヘッダー用の追加設定
COPY security-headers.conf /etc/nginx/conf.d/security-headers.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1
```

### 5.5 マルチコマンド用 entrypoint スクリプト

```dockerfile
FROM postgres:16-alpine

# 初期化スクリプトのコピー
COPY init-scripts/ /docker-entrypoint-initdb.d/

# カスタム entrypoint
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["postgres"]
```

```bash
#!/bin/sh
# entrypoint.sh
set -e

# 環境に応じた前処理
echo "Starting with environment: $APP_ENV"

# 元のエントリポイントに処理を委譲
exec docker-entrypoint.sh "$@"
```

### 5.6 Ruby on Rails アプリケーション

```dockerfile
FROM ruby:3.3-slim

ENV RAILS_ENV=production \
    RAILS_LOG_TO_STDOUT=true \
    BUNDLE_WITHOUT=development:test

WORKDIR /app

# システム依存関係
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        nodejs \
        yarn && \
    rm -rf /var/lib/apt/lists/*

# Gem 依存関係
COPY Gemfile Gemfile.lock ./
RUN bundle install --jobs 4 --retry 3

# アセットプリコンパイル
COPY . .
RUN bundle exec rails assets:precompile

# non-root ユーザー
RUN useradd --create-home --shell /bin/bash rails
USER rails

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

CMD ["bundle", "exec", "puma", "-C", "config/puma.rb"]
```

### 5.7 Rust アプリケーション

```dockerfile
FROM rust:1.75-alpine AS builder

RUN apk add --no-cache musl-dev

WORKDIR /app
COPY Cargo.toml Cargo.lock ./

# ダミーの main.rs で依存関係のみビルド（キャッシュ用）
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release
RUN rm -rf src

# 実際のソースコードでビルド
COPY src ./src
RUN touch src/main.rs && cargo build --release

FROM scratch
COPY --from=builder /app/target/release/myapp /myapp
EXPOSE 8080
ENTRYPOINT ["/myapp"]
```

---

## 6. BuildKit の高度な機能

### 6.1 syntax ディレクティブ

```dockerfile
# syntax=docker/dockerfile:1
# 最新の Dockerfile パーサーを使用

FROM node:20-alpine
WORKDIR /app
COPY . .
CMD ["node", "server.js"]
```

### 6.2 heredoc 構文

```dockerfile
# syntax=docker/dockerfile:1

# 複数行のスクリプトを読みやすく記述
RUN <<EOF
apt-get update
apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates
rm -rf /var/lib/apt/lists/*
EOF

# ファイル生成
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 80;
    location / {
        proxy_pass http://app:3000;
    }
}
EOF

# 複数ファイル同時生成
COPY <<nginx.conf /etc/nginx/nginx.conf
user nginx;
worker_processes auto;
nginx.conf

COPY <<app.conf /etc/nginx/conf.d/app.conf
server {
    listen 80;
    root /usr/share/nginx/html;
}
app.conf
```

### 6.3 マウントオプション

```dockerfile
# キャッシュマウント（ビルド間でキャッシュを再利用）
RUN --mount=type=cache,target=/root/.npm \
    npm ci

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

RUN --mount=type=cache,target=/go/pkg/mod \
    --mount=type=cache,target=/root/.cache/go-build \
    go build -o /app .

# シークレットマウント（イメージに残らない）
RUN --mount=type=secret,id=aws_credentials,target=/root/.aws/credentials \
    aws s3 cp s3://bucket/file /app/

# SSH マウント（SSH キーを使ったプライベートリポジトリアクセス）
RUN --mount=type=ssh \
    git clone git@github.com:private/repo.git

# バインドマウント（ビルドコンテキスト外のファイルを参照）
RUN --mount=type=bind,from=builder,source=/app/dist,target=/dist \
    cp -r /dist /usr/share/nginx/html/
```

---

## 7. 比較表

### 比較表 1: CMD vs ENTRYPOINT

| 項目 | CMD | ENTRYPOINT |
|---|---|---|
| 目的 | デフォルトコマンド/引数 | 固定のメインコマンド |
| docker run で上書き | コマンド引数で上書き可能 | `--entrypoint` でのみ上書き |
| 形式 | exec形式 `["cmd","arg"]` / shell形式 `cmd arg` | exec形式推奨 |
| 組み合わせ | ENTRYPOINT のデフォルト引数として機能 | CMD と組み合わせ可能 |
| 典型的な用途 | アプリケーション起動コマンド | CLI ツール、ラッパースクリプト |
| 例 | `CMD ["npm", "start"]` | `ENTRYPOINT ["python"]` |

### 比較表 2: COPY vs ADD

| 項目 | COPY | ADD |
|---|---|---|
| ローカルファイルコピー | 可能 | 可能 |
| URL からダウンロード | 不可 | 可能（非推奨） |
| tar の自動展開 | しない | する |
| 予測可能性 | 高い（単純なコピー） | 低い（自動展開等の副作用） |
| 推奨度 | 基本的にこちらを使う | tar 展開が必要な場合のみ |
| --chown | 対応 | 対応 |
| --chmod (BuildKit) | 対応 | 対応 |
| --link (BuildKit) | 対応 | 対応 |

### 比較表 3: 全命令一覧

| 命令 | レイヤー | 用途 | スコープ |
|---|---|---|---|
| FROM | 生成 | ベースイメージ指定 | ステージ区切り |
| RUN | 生成 | コマンド実行 | ビルド時 |
| COPY | 生成 | ファイルコピー | ビルド時 |
| ADD | 生成 | ファイルコピー + 展開 | ビルド時 |
| CMD | なし | デフォルトコマンド | 実行時 |
| ENTRYPOINT | なし | エントリポイント | 実行時 |
| ENV | なし | 環境変数 | ビルド時 + 実行時 |
| ARG | なし | ビルド引数 | ビルド時のみ |
| EXPOSE | なし | ポート宣言 | ドキュメント |
| WORKDIR | なし | 作業ディレクトリ | ビルド時 + 実行時 |
| USER | なし | 実行ユーザー | ビルド時 + 実行時 |
| VOLUME | なし | ボリュームポイント | 実行時 |
| LABEL | なし | メタデータ | イメージメタデータ |
| HEALTHCHECK | なし | ヘルスチェック | 実行時 |
| SHELL | なし | デフォルトシェル | ビルド時 |
| STOPSIGNAL | なし | 停止シグナル | 実行時 |
| ONBUILD | なし | 派生イメージ用トリガー | 派生ビルド時 |

---

## 8. アンチパターン

### アンチパターン 1: 変更頻度を考慮しないレイヤー順序

```dockerfile
# NG: ソースコードを先にコピー
FROM node:20-alpine
WORKDIR /app
COPY . .                        # <- ソース変更で全レイヤーが再ビルド
RUN npm ci --only=production    # <- 毎回 npm install が走る
CMD ["node", "server.js"]

# OK: 変更頻度の低いものから先にコピー
FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./  # <- 変更少ない
RUN npm ci --only=production            # <- キャッシュが効く
COPY . .                                # <- ソース変更はここだけ再ビルド
CMD ["node", "server.js"]
```

### アンチパターン 2: RUN を分割しすぎる

```dockerfile
# NG: 各コマンドを別の RUN にする
FROM ubuntu:22.04
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
RUN apt-get install -y vim
RUN rm -rf /var/lib/apt/lists/*
# -> 5つのレイヤーが作成される
# -> apt-get update のキャッシュが古くなる可能性

# OK: 1つの RUN にまとめる
FROM ubuntu:22.04
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        vim && \
    rm -rf /var/lib/apt/lists/*
# -> 1つのレイヤーで完結
# -> update と install が同じレイヤーで実行される
```

### アンチパターン 3: root ユーザーでアプリを実行

```dockerfile
# NG: root のまま実行
FROM node:20-alpine
WORKDIR /app
COPY . .
RUN npm ci
CMD ["node", "server.js"]
# -> コンテナ内で root 権限で動作（セキュリティリスク）

# OK: non-root ユーザーを作成して切り替え
FROM node:20-alpine
RUN addgroup -S app && adduser -S app -G app
WORKDIR /app
COPY --chown=app:app . .
RUN npm ci
USER app
CMD ["node", "server.js"]
```

### アンチパターン 4: .dockerignore を使わない

```dockerfile
# NG: .dockerignore なし
FROM node:20-alpine
WORKDIR /app
COPY . .    # <- node_modules, .git, .env, テスト等が全て含まれる
            # -> ビルドコンテキストが巨大になる
            # -> 機密情報がイメージに含まれる

# OK: .dockerignore で不要ファイルを除外
# .dockerignore:
# node_modules
# .git
# .env
# tests/
# coverage/
```

### アンチパターン 5: ENV でシークレットを設定する

```dockerfile
# NG: 環境変数にパスワードを直接記載
FROM node:20-alpine
ENV DATABASE_PASSWORD=supersecret123
# -> docker inspect で確認可能
# -> イメージの全レイヤーに残る

# OK: 実行時に環境変数を渡す
FROM node:20-alpine
# docker run -e DATABASE_PASSWORD=supersecret123 my-app
# または docker-compose.yml の env_file で管理
```

---

## 9. FAQ

### Q1: exec 形式とシェル形式はどちらを使うべきですか？

**A:** CMD と ENTRYPOINT では **exec 形式** `["cmd", "arg"]` を推奨する。シェル形式 `cmd arg` は `/bin/sh -c` を介して実行されるため、シグナル（SIGTERM 等）がアプリケーションに直接届かず、グレースフルシャットダウンに失敗する場合がある。RUN 命令ではシェル形式でも問題ない（ビルド時のみ実行されるため）。ただし、パイプ等のシェル機能を使う場合は `SHELL` 命令と組み合わせて `pipefail` オプションを設定すべきである。

### Q2: EXPOSE は必須ですか？

**A:** EXPOSE はドキュメント用であり、実際のポート公開は `docker run -p` で行う。EXPOSE がなくても `-p` でポートマッピングは可能だが、EXPOSE を書くことで「このコンテナはどのポートを使うか」を明示でき、ツール（Docker Compose, Kubernetes 等）がメタデータとして利用する。記載を推奨する。

### Q3: WORKDIR の代わりに `RUN cd /app` を使えますか？

**A:** 使うべきではない。`RUN cd /app` は新しいシェルで実行されるため、次の RUN 命令では元のディレクトリに戻ってしまう。`WORKDIR /app` はそれ以降の全命令（RUN, CMD, ENTRYPOINT, COPY, ADD）に影響するメタデータとして機能し、ディレクトリが存在しない場合は自動作成される。

### Q4: BuildKit はどのように有効化しますか？

**A:** Docker Desktop では BuildKit がデフォルトで有効になっている。Linux の Docker Engine では、環境変数 `DOCKER_BUILDKIT=1` を設定するか、`/etc/docker/daemon.json` に `{"features": {"buildkit": true}}` を追加する。Docker Engine 23.0 以降ではデフォルトで BuildKit が使用される。BuildKit が有効かどうかは `docker build` の出力形式で判別できる（BuildKit はステップ番号ではなくステージ名で表示される）。

### Q5: Dockerfile のサイズに制限はありますか？

**A:** Dockerfile 自体のサイズに厳密な制限はないが、レイヤー数の上限が 127 レイヤー（OverlayFS の場合）である。1つの Dockerfile で大量の RUN / COPY 命令を使うとこの制限に達する可能性がある。マルチステージビルドでは各ステージのレイヤーが独立してカウントされるため、最終ステージのレイヤー数のみが問題になる。

### Q6: ビルドが遅い場合のデバッグ方法は？

**A:** 以下の方法で原因を特定する:
1. `--progress=plain` でビルドログを詳細表示し、どのステップが遅いかを特定する
2. `docker system df` でビルドキャッシュの状態を確認する
3. ビルドコンテキストのサイズを確認し、`.dockerignore` を最適化する
4. レイヤーの順序を見直し、変更頻度の高いものを後に配置する
5. BuildKit の `--mount=type=cache` でパッケージマネージャのキャッシュを再利用する
6. マルチステージビルドで不要なビルドツールを最終イメージから除外する

---

## 10. まとめ

| 項目 | ポイント |
|---|---|
| FROM | ベースイメージ。Alpine や Distroless で軽量化 |
| RUN | 命令はまとめて1つの RUN に。キャッシュのクリーンアップも忘れずに |
| COPY | ファイルコピーは COPY を使う。ADD は tar 展開時のみ |
| CMD / ENTRYPOINT | exec 形式を使う。用途に応じて組み合わせ |
| ENV / ARG | ENV は実行時も有効、ARG はビルド時のみ。シークレットには使わない |
| レイヤー | 変更頻度の低いものから先に配置（キャッシュ効率化） |
| ビルドコンテキスト | .dockerignore で不要ファイルを除外 |
| セキュリティ | non-root ユーザーで実行。シークレットは --mount=type=secret |
| HEALTHCHECK | アプリケーションの正常性を定期的に確認 |
| BuildKit | キャッシュマウント、シークレットマウント、heredoc 等の拡張機能を活用 |

---

## 次に読むべきガイド

- [01-multi-stage-build.md](./01-multi-stage-build.md) -- マルチステージビルドによるイメージサイズ削減
- [02-optimization.md](./02-optimization.md) -- Dockerfile の最適化とベストプラクティス
- [03-language-specific.md](./03-language-specific.md) -- 言語別 Dockerfile テンプレート

---

## 参考文献

1. **Docker Documentation - Dockerfile reference** https://docs.docker.com/reference/dockerfile/ -- Dockerfile の全命令の公式リファレンス。
2. **Docker Documentation - Best practices for Dockerfile** https://docs.docker.com/develop/develop-images/dockerfile_best-practices/ -- Docker 公式のベストプラクティスガイド。
3. **BuildKit - Advanced Dockerfile features** https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md -- BuildKit 固有の拡張機能（--mount, --security 等）のリファレンス。
4. **Dockerfile heredocs** https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/ -- Dockerfile での heredoc 構文の紹介。
5. **Docker BuildKit** https://docs.docker.com/build/buildkit/ -- BuildKit の公式ドキュメント。並列ビルド、キャッシュ制御、シークレット管理。
