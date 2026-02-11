# Dockerfile 基礎

> Dockerfile の基本命令（FROM, RUN, COPY, CMD, ENTRYPOINT）、レイヤー構造、ビルドコンテキストを理解し、再現性のあるコンテナイメージを構築する。

---

## この章で学ぶこと

1. **Dockerfile の主要命令**を理解し、目的に応じた命令を選択できる
2. **レイヤー構造とビルドキャッシュ**の仕組みを把握し、効率的なビルドができる
3. **ビルドコンテキストの最適化**により、高速かつ安全なイメージビルドを実践できる

---

## 1. Dockerfile とは

Dockerfile はコンテナイメージを構築するための命令書である。テキストファイルとして管理でき、バージョン管理・レビュー・自動化が可能になる。

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

# ADD - URL からダウンロード（非推奨、curl + RUN を使うべき）
ADD https://example.com/file.tar.gz /tmp/

# ADD - tar の自動展開
ADD archive.tar.gz /app/
# -> /app/ に展開される

# 基本的に COPY を使い、tar 展開が必要な場合のみ ADD を使う
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

### 2.6 その他の命令

```dockerfile
# ENV - 環境変数
ENV NODE_ENV=production
ENV APP_PORT=3000

# ARG - ビルド時変数（実行時には残らない）
ARG NODE_VERSION=20
FROM node:${NODE_VERSION}-alpine

ARG BUILD_DATE
LABEL build-date=${BUILD_DATE}
# docker build --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) .

# EXPOSE - ポートのドキュメント（実際のポート公開は -p で行う）
EXPOSE 3000
EXPOSE 8080/tcp
EXPOSE 53/udp

# LABEL - メタデータ
LABEL maintainer="team@example.com"
LABEL version="1.0.0"
LABEL description="My application"

# USER - 実行ユーザーの切り替え
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
USER appuser

# HEALTHCHECK - ヘルスチェック
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# VOLUME - ボリュームマウントポイント
VOLUME ["/data", "/logs"]

# SHELL - デフォルトシェルの変更
SHELL ["/bin/bash", "-c"]
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

```bash
# キャッシュを使ったビルド（デフォルト）
docker build -t my-app .

# キャッシュを無視して完全リビルド
docker build --no-cache -t my-app .

# 特定のステージまでビルド
docker build --target builder -t my-app-builder .

# BuildKit でキャッシュの詳細を表示
DOCKER_BUILDKIT=1 docker build --progress=plain -t my-app .
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
node_modules
npm-debug.log*
.git
.gitignore
.env
.env.*
Dockerfile
docker-compose*.yml
.dockerignore
README.md
LICENSE
.vscode
.idea
coverage
dist
*.md
```

```bash
# ビルドコンテキストのサイズ確認
docker build -t my-app . 2>&1 | grep "Sending build context"
# Sending build context to Docker daemon  2.048kB  <- 小さいほど良い

# .dockerignore なしの場合
# Sending build context to Docker daemon  500MB  <- node_modules 等が含まれている
```

---

## 5. 実践例

### 5.1 Express.js アプリケーション

```dockerfile
FROM node:20-alpine

# セキュリティ: non-root ユーザー
RUN addgroup -S app && adduser -S app -G app

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
    apt-get install -y --no-install-recommends gcc && \
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

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### 5.3 Go アプリケーション

```dockerfile
FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o /server ./cmd/server

# 最小の実行環境
FROM scratch
COPY --from=builder /server /server
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

---

## 6. 比較表

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

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: exec 形式とシェル形式はどちらを使うべきですか？

**A:** CMD と ENTRYPOINT では **exec 形式** `["cmd", "arg"]` を推奨する。シェル形式 `cmd arg` は `/bin/sh -c` を介して実行されるため、シグナル（SIGTERM 等）がアプリケーションに直接届かず、グレースフルシャットダウンに失敗する場合がある。RUN 命令ではシェル形式でも問題ない（ビルド時のみ実行されるため）。

### Q2: EXPOSE は必須ですか？

**A:** EXPOSE はドキュメント用であり、実際のポート公開は `docker run -p` で行う。EXPOSE がなくても `-p` でポートマッピングは可能だが、EXPOSE を書くことで「このコンテナはどのポートを使うか」を明示でき、ツール（Docker Compose, Kubernetes 等）がメタデータとして利用する。記載を推奨する。

### Q3: WORKDIR の代わりに `RUN cd /app` を使えますか？

**A:** 使うべきではない。`RUN cd /app` は新しいシェルで実行されるため、次の RUN 命令では元のディレクトリに戻ってしまう。`WORKDIR /app` はそれ以降の全命令（RUN, CMD, ENTRYPOINT, COPY, ADD）に影響するメタデータとして機能し、ディレクトリが存在しない場合は自動作成される。

---

## 9. まとめ

| 項目 | ポイント |
|---|---|
| FROM | ベースイメージ。Alpine や Distroless で軽量化 |
| RUN | 命令はまとめて1つの RUN に。キャッシュのクリーンアップも忘れずに |
| COPY | ファイルコピーは COPY を使う。ADD は tar 展開時のみ |
| CMD / ENTRYPOINT | exec 形式を使う。用途に応じて組み合わせ |
| レイヤー | 変更頻度の低いものから先に配置（キャッシュ効率化） |
| ビルドコンテキスト | .dockerignore で不要ファイルを除外 |
| セキュリティ | non-root ユーザーで実行 |

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
