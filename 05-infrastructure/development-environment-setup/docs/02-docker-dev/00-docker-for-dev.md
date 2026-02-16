# 開発用 Docker

> Docker Desktop、Dev Containers、docker compose を活用し、再現性の高い開発環境を構築するための実践ガイド。

## この章で学ぶこと

1. Docker Desktop のインストール・リソース設定・パフォーマンスチューニング
2. docker compose による開発環境の構築と管理
3. ホットリロード・ボリュームマウント・ネットワーク設計の実践テクニック
4. マルチステージビルドによる開発/本番イメージの分離
5. Docker のセキュリティベストプラクティスと運用ノウハウ
6. トラブルシューティングと CI/CD パイプラインとの連携

---

## 1. Docker Desktop のセットアップ

### 1.1 インストール

```bash
# macOS (Homebrew)
brew install --cask docker

# macOS (OrbStack -- 推奨代替)
brew install --cask orbstack

# Windows (WSL2 バックエンド推奨)
winget install Docker.DockerDesktop

# Linux (Docker Engine -- 公式スクリプト)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 再ログイン後に有効化

# Linux (Ubuntu -- apt)
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 1.2 リソース設定

```
Docker Desktop 推奨リソース設定:

┌─────────────────────────────────────┐
│ Settings → Resources                │
│                                      │
│  CPU:     4+ コア (ホストの半分)     │
│  Memory:  4-8 GB                     │
│  Swap:    1 GB                       │
│  Disk:    64+ GB                     │
│                                      │
│ Settings → General                   │
│  ✅ Use virtualization framework     │
│  ✅ VirtioFS (macOS - 高速)          │
│  ✅ Use Rosetta for x86/amd64       │
│     emulation (Apple Silicon)        │
│                                      │
│ Settings → Docker Engine             │
│  {                                   │
│    "builder": {                      │
│      "gc": {                         │
│        "enabled": true,              │
│        "defaultKeepStorage": "20GB"  │
│      }                               │
│    },                                │
│    "features": {                     │
│      "buildkit": true                │
│    },                                │
│    "log-driver": "json-file",        │
│    "log-opts": {                     │
│      "max-size": "10m",              │
│      "max-file": "3"                 │
│    }                                 │
│  }                                   │
└─────────────────────────────────────┘

プロジェクト規模別の推奨:
┌──────────────┬────────┬────────┬──────┐
│ 規模          │ CPU    │ Memory │ Disk │
├──────────────┼────────┼────────┼──────┤
│ 小規模 (1-2)  │ 2 コア │ 2 GB   │ 32GB │
│ 中規模 (3-5)  │ 4 コア │ 4 GB   │ 64GB │
│ 大規模 (5+)   │ 6 コア │ 8 GB   │ 128GB│
│ ML/AI 開発    │ 8 コア │ 16 GB  │ 256GB│
└──────────────┴────────┴────────┴──────┘
```

### 1.3 代替ツール

| ツール | OS | 特徴 | 料金 |
|--------|-----|------|------|
| Docker Desktop | 全OS | 公式・GUI付き | 個人無料/企業有料 |
| OrbStack | macOS | 軽量・高速・低メモリ | 個人無料 |
| Rancher Desktop | 全OS | OSS・containerd対応 | 無料 |
| Podman Desktop | 全OS | rootless・デーモンレス | 無料 |
| Colima | macOS/Linux | CLI専用・軽量 | 無料 |
| Lima | macOS | VM ベース・柔軟 | 無料 |

```
代替ツールの選択フローチャート:

  Q1: OS は何か？
  │
  ├── macOS
  │   │
  │   └── Q2: GUI は必要？
  │       ├── Yes → OrbStack (推奨) / Docker Desktop
  │       └── No  → Colima
  │
  ├── Windows
  │   │
  │   └── Q2: WSL2 を使える？
  │       ├── Yes → Docker Desktop / Rancher Desktop
  │       └── No  → Docker Desktop (Hyper-V)
  │
  └── Linux
      │
      └── Q2: rootless が必要？
          ├── Yes → Podman
          └── No  → Docker Engine (公式)

  OrbStack vs Docker Desktop (macOS):
  ┌────────────────────┬───────────┬──────────────┐
  │ 項目                │ OrbStack  │ Docker Desktop│
  ├────────────────────┼───────────┼──────────────┤
  │ メモリ使用量        │ ~200 MB   │ ~1-2 GB      │
  │ 起動時間            │ ~2 秒     │ ~30 秒       │
  │ ファイルI/O速度     │ 高速      │ 普通         │
  │ ライセンス問題      │ なし      │ 大企業は有料  │
  │ Docker CLI 互換     │ 100%      │ 100%         │
  │ K8s サポート        │ あり      │ あり          │
  │ GUI                │ あり      │ あり          │
  └────────────────────┴───────────┴──────────────┘
```

### 1.4 Docker CLI の基本確認

```bash
# バージョン確認
docker version
docker compose version

# システム情報
docker system info

# ディスク使用量
docker system df
docker system df -v     # 詳細表示

# ヘルスチェック
docker run --rm hello-world
```

---

## 2. 開発用 Dockerfile

### 2.1 マルチステージビルド

```dockerfile
# ─── ステージ 1: 依存インストール ───
FROM node:20-slim AS deps
WORKDIR /app

# パッケージマネージャーの有効化
RUN corepack enable

# 依存定義ファイルのみコピー (キャッシュ活用)
COPY package.json pnpm-lock.yaml .npmrc ./
RUN pnpm install --frozen-lockfile

# ─── ステージ 2: 開発環境 ───
FROM node:20-slim AS dev
WORKDIR /app

RUN corepack enable

# 開発に必要なツール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    jq \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deps /app/node_modules ./node_modules
COPY . .

EXPOSE 3000
CMD ["pnpm", "dev"]

# ─── ステージ 3: ビルド ───
FROM node:20-slim AS build
WORKDIR /app

RUN corepack enable

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# ビルド時環境変数
ARG NODE_ENV=production
ENV NODE_ENV=${NODE_ENV}

RUN pnpm build

# 本番用の依存のみ再インストール
RUN pnpm install --prod --frozen-lockfile

# ─── ステージ 4: 本番 ───
FROM node:20-slim AS production
WORKDIR /app

# セキュリティ: 非 root ユーザー
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 appuser

# 本番に必要なファイルのみコピー
COPY --from=build --chown=appuser:nodejs /app/dist ./dist
COPY --from=build --chown=appuser:nodejs /app/node_modules ./node_modules
COPY --from=build --chown=appuser:nodejs /app/package.json ./package.json

# セキュリティ: 読み取り専用ファイルシステム対応
USER appuser

EXPOSE 3000

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:3000/api/health || exit 1

CMD ["node", "dist/index.js"]
```

### 2.2 Python プロジェクト用 Dockerfile

```dockerfile
# ─── ステージ 1: ビルダー ───
FROM python:3.12-slim AS builder
WORKDIR /app

# uv を使った高速インストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 依存定義のみコピー
COPY pyproject.toml uv.lock ./

# 仮想環境を作成して依存インストール
RUN uv sync --frozen --no-dev

# ─── ステージ 2: 開発環境 ───
FROM python:3.12-slim AS dev
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 開発ツール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ─── ステージ 3: 本番 ───
FROM python:3.12-slim AS production
WORKDIR /app

RUN adduser --system --uid 1001 appuser

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2.3 ステージの構造

```
マルチステージビルドのフロー:

  deps ──→ dev (開発時)
    │
    └──→ build ──→ production (本番時)

  開発時:
    docker compose up     → dev ステージを使用
                           (ソースをマウント + ホットリロード)

  本番時:
    docker build --target production → 最小イメージ
                                       (node_modules + dist のみ)

  イメージサイズ比較:
  ┌────────────────────┬───────────┬──────────────────┐
  │ ステージ            │ サイズ     │ 含まれるもの      │
  ├────────────────────┼───────────┼──────────────────┤
  │ dev (全依存 + src)  │ ~800 MB   │ devDeps + ツール  │
  │ build (全依存+成果) │ ~700 MB   │ ビルド成果物      │
  │ production (最小)   │ ~150 MB   │ prodDeps + dist  │
  │ distroless         │ ~80 MB    │ ランタイムのみ     │
  └────────────────────┴───────────┴──────────────────┘
```

### 2.4 .dockerignore

```bash
# .dockerignore
node_modules
.next
dist
build
coverage
.turbo
.env.local
.env.*.local
*.log
.git
.vscode
.idea
*.md
!README.md
Dockerfile*
docker-compose*.yml
compose*.yaml
.dockerignore
```

### 2.5 Dockerfile のベストプラクティス

```
Dockerfile 最適化チェックリスト:

□ マルチステージビルドを使用している
□ .dockerignore でビルドコンテキストを最小化
□ COPY は変更頻度の低いファイルから順にコピー
  (package.json → lockfile → ソースコード)
□ RUN は && で連結して層を削減
□ apt-get は install 後に rm -rf /var/lib/apt/lists/*
□ 非 root ユーザーで実行 (USER)
□ HEALTHCHECK を設定
□ 不要な環境変数を本番イメージに含めない
□ slim / alpine ベースイメージを使用
□ LABEL でメタデータを付与
□ COPY --chown でファイル権限を設定
□ ARG で環境固有の値を注入可能に
```

---

## 3. docker compose で開発環境構築

### 3.1 基本構成

```yaml
# compose.yaml (docker compose v2 形式)
name: my-project

services:
  # ─── アプリケーション ───
  app:
    build:
      context: .
      dockerfile: Dockerfile
      target: dev
      args:
        NODE_VERSION: "20"
    ports:
      - "3000:3000"
    volumes:
      - .:/app                           # ソースコードマウント
      - /app/node_modules                # node_modules はコンテナ内を使用
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/mydb
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=debug
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    develop:
      watch:                             # docker compose watch 用
        - action: sync
          path: ./src
          target: /app/src
        - action: rebuild
          path: package.json
        - action: rebuild
          path: pnpm-lock.yaml
    restart: unless-stopped

  # ─── データベース ───
  db:
    image: postgres:16-alpine
    container_name: ${COMPOSE_PROJECT_NAME:-myproject}-db
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: mydb
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
      TZ: Asia/Tokyo
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s
    restart: unless-stopped

  # ─── キャッシュ ───
  redis:
    image: redis:7-alpine
    container_name: ${COMPOSE_PROJECT_NAME:-myproject}-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  # ─── メールテスト ───
  mailpit:
    image: axllent/mailpit:latest
    container_name: ${COMPOSE_PROJECT_NAME:-myproject}-mail
    ports:
      - "1025:1025"    # SMTP
      - "8025:8025"    # Web UI
    environment:
      MP_SMTP_AUTH_ACCEPT_ANY: 1
      MP_SMTP_AUTH_ALLOW_INSECURE: 1
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 3.2 docker compose の override パターン

```yaml
# compose.yaml (ベース -- CI/本番共通)
services:
  app:
    build:
      context: .
      target: production
    environment:
      - NODE_ENV=production
```

```yaml
# compose.override.yaml (ローカル開発用 -- 自動読み込み)
services:
  app:
    build:
      target: dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
      - LOG_LEVEL=debug
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

```yaml
# compose.ci.yaml (CI 用)
services:
  app:
    build:
      target: build
    environment:
      - NODE_ENV=test
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/testdb

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: testdb
    tmpfs:
      - /var/lib/postgresql/data  # CI ではメモリ上で高速化
```

```bash
# 使い方
docker compose up                                    # base + override (自動)
docker compose -f compose.yaml -f compose.ci.yaml up # base + CI
docker compose -f compose.yaml up                    # base のみ
```

### 3.3 よく使うコマンド

```bash
# ─── 起動 ───
docker compose up -d                    # バックグラウンド起動
docker compose up --build               # リビルドして起動
docker compose up --build --force-recreate  # 強制再作成
docker compose watch                    # ファイル変更を監視 (v2.22+)
docker compose up -d --wait             # ヘルスチェック完了まで待機

# ─── ログ ───
docker compose logs -f app              # app のログを追跡
docker compose logs --tail 100 db       # db の最新100行
docker compose logs --since 5m          # 直近5分のログ
docker compose logs -f --no-log-prefix  # プレフィックスなし

# ─── 操作 ───
docker compose exec app sh              # コンテナ内でシェル
docker compose exec app bash            # bash が使える場合
docker compose exec db psql -U postgres # DB に接続
docker compose exec redis redis-cli     # Redis CLI
docker compose run --rm app pnpm test   # 一時コンテナでテスト
docker compose run --rm app pnpm prisma migrate dev  # マイグレーション

# ─── スケーリング ───
docker compose up -d --scale worker=3   # worker を3インスタンスに

# ─── 状態確認 ───
docker compose ps                       # サービス一覧
docker compose ps -a                    # 停止中も含む
docker compose top                      # プロセス一覧
docker compose stats                    # リソース使用状況

# ─── 停止・削除 ───
docker compose down                     # 停止
docker compose down -v                  # ボリュームも削除
docker compose down --rmi all           # イメージも削除
docker compose down --remove-orphans    # 孤立コンテナも削除

# ─── クリーンアップ ───
docker system prune -af                 # 不要な全リソース削除
docker volume prune                     # 未使用ボリューム削除
docker builder prune -af                # ビルドキャッシュ削除
docker image prune -af                  # 不要イメージ削除
```

### 3.4 Makefile による操作簡略化

```makefile
# Makefile
.PHONY: up down restart build logs shell db-cli redis-cli seed clean

# ─── 起動・停止 ───
up:
	docker compose up -d --wait

down:
	docker compose down

restart:
	docker compose restart

build:
	docker compose up -d --build --wait

# ─── ログ ───
logs:
	docker compose logs -f

logs-app:
	docker compose logs -f app

# ─── シェル ───
shell:
	docker compose exec app sh

db-cli:
	docker compose exec db psql -U postgres -d mydb

redis-cli:
	docker compose exec redis redis-cli

# ─── データベース ───
migrate:
	docker compose exec app pnpm prisma migrate dev

seed:
	docker compose exec app pnpm prisma db seed

db-reset:
	docker compose exec app pnpm prisma migrate reset --force

# ─── テスト ───
test:
	docker compose run --rm app pnpm test

test-watch:
	docker compose run --rm app pnpm test:watch

# ─── クリーンアップ ───
clean:
	docker compose down -v --rmi all --remove-orphans
	docker system prune -af

# ─── CI ───
ci:
	docker compose -f compose.yaml -f compose.ci.yaml up -d --wait
	docker compose exec app pnpm test
	docker compose -f compose.yaml -f compose.ci.yaml down -v
```

---

## 4. ボリュームマウントとパフォーマンス

### 4.1 macOS のパフォーマンス問題と解決策

```
macOS でのファイルシステムパフォーマンス:

  ┌─────────────────────────────────────┐
  │  macOS ホスト                        │
  │  ┌───────────────────────────────┐  │
  │  │  ソースコード (/Users/...)     │  │
  │  └───────────┬───────────────────┘  │
  │              │                       │
  │         VirtioFS / gRPC FUSE         │
  │         (ファイル共有レイヤー)        │
  │              │                       │
  │  ┌───────────┴───────────────────┐  │
  │  │  Linux VM (Docker Engine)      │  │
  │  │  ┌─────────────────────────┐  │  │
  │  │  │  コンテナ                │  │  │
  │  │  │  /app (マウントポイント)  │  │  │
  │  │  └─────────────────────────┘  │  │
  │  └───────────────────────────────┘  │
  └─────────────────────────────────────┘

  パフォーマンス (npm install の比較):
  ┌──────────────────────┬──────────┬──────────┐
  │ 方式                  │ 速度     │ 推奨度    │
  ├──────────────────────┼──────────┼──────────┤
  │ ネイティブ (ホスト)    │ 1x (基準)│ -        │
  │ VirtioFS             │ 1.5-2x  │ ★★★★    │
  │ gRPC FUSE            │ 3-5x    │ ★★      │
  │ 名前付きボリューム     │ 1.1x    │ ★★★★★  │
  │ OrbStack             │ 1.2x    │ ★★★★★  │
  │ 匿名ボリューム        │ 1.1x    │ ★★★     │
  └──────────────────────┴──────────┴──────────┘

  ※ 名前付きボリュームが最速だが、ホストから直接アクセスできない
  ※ VirtioFS は Docker Desktop v4.6+ でデフォルト
```

### 4.2 パフォーマンス最適化

```yaml
# compose.yaml のベストプラクティス
services:
  app:
    volumes:
      # ソースコードはバインドマウント (ホットリロード用)
      - .:/app

      # node_modules は名前付きボリューム (高速)
      - node_modules:/app/node_modules

      # ビルドキャッシュも名前付きボリューム
      - next_cache:/app/.next
      - turbo_cache:/app/.turbo

      # 一時ファイルは tmpfs (メモリ上)
      - type: tmpfs
        target: /app/tmp

volumes:
  node_modules:
  next_cache:
  turbo_cache:
```

```
ボリューム戦略の使い分け:

  ┌──────────────────────┬──────────────────────────────┐
  │ データの種類          │ 推奨マウント方式               │
  ├──────────────────────┼──────────────────────────────┤
  │ ソースコード          │ バインドマウント (ホスト→コンテナ) │
  │ node_modules         │ 名前付きボリューム              │
  │ DB データ            │ 名前付きボリューム              │
  │ ビルドキャッシュ       │ 名前付きボリューム              │
  │ ログ (一時)          │ tmpfs                        │
  │ テスト成果物          │ バインドマウント (結果取得用)     │
  │ 設定ファイル          │ バインドマウント (読み取り専用)   │
  └──────────────────────┴──────────────────────────────┘
```

### 4.3 docker compose watch

```yaml
# compose.yaml (docker compose watch 設定)
services:
  app:
    build:
      context: .
      target: dev
    develop:
      watch:
        # ソースコード変更 → コンテナにコピー (高速)
        - action: sync
          path: ./src
          target: /app/src
          ignore:
            - "**/*.test.ts"

        # 設定ファイル変更 → コンテナにコピー
        - action: sync
          path: ./public
          target: /app/public

        # 依存変更 → コンテナ再ビルド
        - action: rebuild
          path: package.json

        - action: rebuild
          path: pnpm-lock.yaml

        # Dockerfile 変更 → コンテナ再ビルド
        - action: rebuild
          path: Dockerfile
```

```bash
# docker compose watch の実行
docker compose watch

# バックグラウンドで実行
docker compose watch &

# ログも表示
docker compose watch --no-up  # 既に起動済みの場合
```

```
docker compose watch の動作:

  ホスト側でファイル変更を検知
       │
       ▼
  ┌──────────────────────┐
  │  action: sync        │ → ファイルをコンテナにコピー
  │  (src/ の変更)        │   ビルド不要、即反映
  │                      │   ホットリロードが効く
  ├──────────────────────┤
  │  action: rebuild     │ → コンテナを再ビルド
  │  (package.json 変更)  │   新しい依存を反映
  │                      │   数十秒かかる
  ├──────────────────────┤
  │  action: sync+restart│ → ファイルコピー後に再起動
  │  (設定ファイル変更)    │   プロセスの再読み込みが必要
  └──────────────────────┘

  vs バインドマウント:
  ┌────────────────────────┬──────────────────────┐
  │ バインドマウント        │ docker compose watch  │
  ├────────────────────────┼──────────────────────┤
  │ リアルタイム反映        │ イベント駆動          │
  │ I/O オーバーヘッド大    │ コピー時のみ          │
  │ macOS で遅い          │ OS 依存しない          │
  │ node_modules も共有    │ sync 対象のみ          │
  │ 設定不要              │ develop.watch 設定必要  │
  └────────────────────────┴──────────────────────┘
```

---

## 5. 環境変数管理

### 5.1 .env ファイルの構成

```bash
# .env (docker compose が自動読込 -- チーム共有)
COMPOSE_PROJECT_NAME=my-project
NODE_ENV=development

# データベース
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=mydb
DATABASE_URL=postgresql://postgres:postgres@db:5432/mydb

# Redis
REDIS_URL=redis://redis:6379

# .env.local (個人設定 -- .gitignore に追加)
GITHUB_TOKEN=ghp_xxxxx
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# .env.example (テンプレート -- リポジトリにコミット)
# コピーして .env.local を作成: cp .env.example .env.local
GITHUB_TOKEN=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
```

```yaml
# compose.yaml
services:
  app:
    env_file:
      - path: .env
        required: true
      - path: .env.local
        required: false   # 存在しなくてもエラーにしない
    environment:
      # env_file の値を上書き
      - LOG_LEVEL=debug
      - ENABLE_FEATURE_X=true
```

### 5.2 シークレット管理

```yaml
# compose.yaml (Docker Secrets を使った管理)
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt   # ローカルファイル
    # または
    # environment: DB_PASSWORD        # 環境変数から
```

```
シークレット管理のベストプラクティス:

  開発環境:
  ├── .env (共有設定) → リポジトリにコミット
  ├── .env.local (個人のシークレット) → .gitignore
  └── .env.example (テンプレート) → リポジトリにコミット

  CI 環境:
  ├── GitHub Actions Secrets → 環境変数として注入
  └── GitHub Actions Variables → 非機密設定

  本番環境:
  ├── AWS Secrets Manager / SSM Parameter Store
  ├── HashiCorp Vault
  └── Docker Secrets (Swarm 使用時)

  絶対にやってはいけないこと:
  ❌ .env.local をリポジトリにコミット
  ❌ Dockerfile に ENV でシークレットを埋め込む
  ❌ docker-compose.yml にパスワードを直書き (開発用以外)
  ❌ ビルドイメージにシークレットを含める
```

---

## 6. ネットワーク設計

### 6.1 サービス間通信

```
docker compose のネットワーク:

  ┌─────────────── my-project_default ──────────────┐
  │            (Docker 内部ネットワーク)                │
  │                                                    │
  │  ┌─────────┐   ┌──────┐   ┌───────┐              │
  │  │   app   │──→│  db  │   │ redis │              │
  │  │ :3000   │   │:5432 │   │ :6379 │              │
  │  └────┬────┘   └──────┘   └───────┘              │
  │       │                                            │
  │  サービス名で通信:                                  │
  │  db:5432 (NOT localhost:5432)                      │
  │  redis:6379 (NOT localhost:6379)                   │
  └────┬──────────────────────────────────────────────┘
       │
       │ ports: "3000:3000"
       ▼
  ┌──────────┐
  │  ホスト    │
  │ localhost │
  │  :3000    │
  └──────────┘

  ※ ホストからは localhost:3000 でアクセス
  ※ コンテナ間はサービス名で通信
  ※ DNS 解決は Docker の内部 DNS が自動処理
```

### 6.2 カスタムネットワーク

```yaml
# compose.yaml (複数ネットワーク)
services:
  app:
    networks:
      - frontend
      - backend

  web:
    image: nginx:alpine
    networks:
      - frontend
    ports:
      - "80:80"

  db:
    networks:
      - backend
    # ↑ frontend からはアクセス不可 (セキュリティ)

  redis:
    networks:
      - backend

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # 外部からのアクセスを遮断
```

### 6.3 複数プロジェクト間の通信

```yaml
# project-a/compose.yaml
services:
  api:
    networks:
      - shared

networks:
  shared:
    name: shared-network
    external: true

# project-b/compose.yaml
services:
  web:
    networks:
      - shared
    environment:
      - API_URL=http://api:3000

networks:
  shared:
    name: shared-network
    external: true
```

```bash
# 共有ネットワークの作成
docker network create shared-network

# 両プロジェクトを起動
cd project-a && docker compose up -d
cd project-b && docker compose up -d

# project-b の web から project-a の api にアクセス可能
```

---

## 7. Docker ビルドの最適化

### 7.1 BuildKit の活用

```dockerfile
# syntax=docker/dockerfile:1

# BuildKit のキャッシュマウント
FROM node:20-slim AS deps
WORKDIR /app
RUN corepack enable

COPY package.json pnpm-lock.yaml ./

# パッケージマネージャーのキャッシュをマウント
RUN --mount=type=cache,id=pnpm,target=/root/.local/share/pnpm/store \
    pnpm install --frozen-lockfile

# シークレットのマウント (イメージに残らない)
RUN --mount=type=secret,id=npm_token \
    NPM_TOKEN=$(cat /run/secrets/npm_token) pnpm install
```

```bash
# BuildKit を有効にしてビルド
DOCKER_BUILDKIT=1 docker build .

# シークレットを渡してビルド
docker build --secret id=npm_token,src=./.npmrc .

# マルチプラットフォームビルド
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:latest .
```

### 7.2 ビルドキャッシュの最適化

```
Docker レイヤーキャッシュの仕組み:

  Dockerfile の各命令 (FROM, COPY, RUN 等) はレイヤーを生成
  レイヤーは前回のビルド結果をキャッシュ
  変更があったレイヤー以降は全て再実行

  最適化の原則:
  1. 変更頻度の低いものを上に
  2. 変更頻度の高いものを下に

  ❌ 悪い例:
  COPY . .                    # ← ソース変更で毎回
  RUN pnpm install            # ← 依存は変わってないのに再実行

  ✅ 良い例:
  COPY package.json pnpm-lock.yaml ./  # ← 依存定義のみ
  RUN pnpm install                      # ← 依存が変わった時のみ
  COPY . .                              # ← ソース変更は最後
```

---

## 8. セキュリティベストプラクティス

### 8.1 イメージのセキュリティ

```dockerfile
# ─── ベストプラクティス ───

# 1. 特定バージョンを指定 (latest は使わない)
FROM node:20.12.0-slim   # ✅ 固定バージョン
# FROM node:latest       # ❌ バージョン不定

# 2. 非 root ユーザーで実行
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 appuser
USER appuser

# 3. 読み取り専用ファイルシステム
# compose.yaml で: read_only: true

# 4. セキュリティスキャン
# docker scout quickview myapp:latest
# docker scout cves myapp:latest

# 5. 不要なパッケージを含めない
# slim / distroless イメージを使用
```

### 8.2 compose.yaml のセキュリティ設定

```yaml
services:
  app:
    # セキュリティオプション
    security_opt:
      - no-new-privileges:true    # 権限昇格を防止
    read_only: true               # ファイルシステムを読み取り専用
    tmpfs:
      - /tmp                      # 書き込み可能な一時領域
      - /app/tmp
    cap_drop:
      - ALL                       # 全ケーパビリティを除去
    cap_add:
      - NET_BIND_SERVICE          # 必要なものだけ追加
```

---

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

```
問題: コンテナが起動しない / すぐに終了する

解決:
  1. ログを確認: docker compose logs app
  2. インタラクティブに起動: docker compose run --rm app sh
  3. CMD を確認: docker inspect myapp:latest | jq '.[0].Config.Cmd'
  4. ヘルスチェック状態: docker compose ps (STATUS 列)

---

問題: ポートが既に使用されている (port already in use)

解決:
  1. 使用中のポートを確認: lsof -i :3000
  2. compose.yaml でポートを変更: "3001:3000"
  3. 既存コンテナを停止: docker compose down
  4. ホスト側のプロセスを停止

---

問題: node_modules がホストとコンテナで競合する

解決:
  1. 名前付きボリュームで分離:
     volumes:
       - node_modules:/app/node_modules
  2. ホストでも pnpm install を実行 (IDE 補完用)
  3. Dev Container を使う (推奨)

---

問題: ファイル変更がコンテナに反映されない

解決:
  1. ボリュームマウントを確認: docker compose config
  2. .dockerignore を確認 (マウント対象外?)
  3. docker compose watch を使う
  4. inotify の制限を確認 (Linux):
     echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf

---

問題: コンテナのディスク容量が不足

解決:
  1. 不要リソースの削除: docker system prune -af
  2. ビルドキャッシュの削除: docker builder prune -af
  3. Docker Desktop のディスクサイズを拡張
  4. 定期的な自動クリーンアップスクリプトを設定

---

問題: macOS でのビルド / I/O が遅い

解決:
  1. VirtioFS を有効化 (Docker Desktop Settings)
  2. OrbStack に切り替え
  3. node_modules を名前付きボリュームに
  4. .dockerignore を適切に設定
  5. docker compose watch を使う (バインドマウントの代替)
```

### 9.2 デバッグコマンド

```bash
# ─── コンテナの状態確認 ───
docker compose ps -a
docker compose logs --tail 50 app
docker inspect <container_id>

# ─── ネットワーク確認 ───
docker network ls
docker network inspect my-project_default
docker compose exec app ping db    # サービス間の疎通確認

# ─── リソース確認 ───
docker stats                        # リアルタイムリソース使用量
docker compose top                  # コンテナ内プロセス
docker system df -v                 # ディスク使用量詳細

# ─── イメージの中身を確認 ───
docker run --rm -it myapp:latest sh
docker history myapp:latest         # レイヤー履歴
docker inspect myapp:latest | jq '.[0].Config'  # 設定確認

# ─── ビルドのデバッグ ───
docker build --progress=plain .     # ビルドログの詳細表示
docker build --no-cache .           # キャッシュ無しでビルド
```

---

## 10. CI/CD との連携

### 10.1 GitHub Actions での Docker ビルド

```yaml
# .github/workflows/docker.yml
name: Docker Build
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Docker Buildx のセットアップ
      - uses: docker/setup-buildx-action@v3

      # Docker レイヤーキャッシュ
      - uses: actions/cache@v4
        with:
          path: /tmp/.buildx-cache
          key: ${{ runner.os }}-buildx-${{ github.sha }}
          restore-keys: |
            ${{ runner.os }}-buildx-

      # ビルド
      - uses: docker/build-push-action@v5
        with:
          context: .
          target: production
          push: false
          cache-from: type=local,src=/tmp/.buildx-cache
          cache-to: type=local,dest=/tmp/.buildx-cache-new,mode=max

      # キャッシュの更新
      - run: |
          rm -rf /tmp/.buildx-cache
          mv /tmp/.buildx-cache-new /tmp/.buildx-cache
```

### 10.2 docker compose を使ったテスト

```yaml
# .github/workflows/test.yml
name: Test with Docker Compose
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start services
        run: docker compose -f compose.yaml -f compose.ci.yaml up -d --wait

      - name: Run tests
        run: docker compose exec -T app pnpm test

      - name: Run lint
        run: docker compose exec -T app pnpm lint

      - name: Cleanup
        if: always()
        run: docker compose -f compose.yaml -f compose.ci.yaml down -v
```

---

## 11. アンチパターン

### 11.1 開発用と本番用で同じ Dockerfile を使う

```
❌ アンチパターン: 1つの Dockerfile を全環境で共用

FROM node:20
WORKDIR /app
COPY . .
RUN npm install       # devDependencies も入る
CMD ["npm", "start"]  # 開発ツールも含んだ巨大イメージ

問題:
  - 本番イメージが不必要に大きい
  - 開発ツール (eslint等) が本番に含まれる
  - セキュリティリスク増大
  - 攻撃対象面 (attack surface) が拡大

✅ 正しいアプローチ:
  - マルチステージビルドで分離
  - dev ステージ: 全依存 + ホットリロード
  - production ステージ: 最小依存 + ビルド成果物のみ
  - --target フラグで使い分け
```

### 11.2 ボリュームデータのバックアップを取らない

```
❌ アンチパターン: docker compose down -v で開発データ全消失

問題:
  - テストデータの再作成に時間がかかる
  - シードデータが失われる
  - 「あの不具合が再現できない」

✅ 正しいアプローチ:
  - シードスクリプトを用意 (init.sql, seed.ts)
  - docker-entrypoint-initdb.d/ に初期化SQLを配置
  - Makefile に seed コマンドを定義
  - 定期的な docker compose down は -v なしで
  - 重要なデータは pg_dump でバックアップスクリプトを用意
```

### 11.3 latest タグを使う

```
❌ アンチパターン: ベースイメージに latest を使用

FROM node:latest
FROM postgres:latest

問題:
  - ビルドごとに異なるバージョンが使われる可能性
  - 「先週まで動いていたのに」問題
  - 再現性がない

✅ 正しいアプローチ:
  FROM node:20.12.0-slim
  FROM postgres:16.2-alpine
  - メジャー.マイナー.パッチまで固定
  - Renovate / Dependabot で自動更新 PR を作成
```

### 11.4 root ユーザーで実行する

```
❌ アンチパターン: 本番コンテナを root で実行

FROM node:20
WORKDIR /app
COPY . .
# USER 指定なし → root で実行

問題:
  - コンテナ脱出時にホスト root 権限を取得される
  - ファイル書き込みの権限問題
  - セキュリティ監査で指摘される

✅ 正しいアプローチ:
  RUN adduser --system --uid 1001 appuser
  USER appuser
  - 開発時は root でも可 (Dev Container 等)
  - 本番は必ず非 root
```

---

## 12. FAQ

### Q1: Docker Desktop と OrbStack、どちらを使うべき？

**A:** macOS ユーザーには OrbStack を推奨。Docker Desktop と比較してメモリ使用量が半分以下 (約200MB vs 1-2GB)、起動が数秒で完了し、ファイルシステムのパフォーマンスも優れている。Docker CLI と完全互換なので移行コストはゼロ。Docker Desktop のライセンス問題（従業員250人以上 or 年間売上1000万ドル以上の企業は有料）を避けられるメリットもある。

### Q2: `docker compose up` と `docker compose watch` の違いは？

**A:** `docker compose up` はコンテナ起動のみ。ボリュームマウントでファイル変更は反映されるが、`package.json` の変更などは手動リビルドが必要。`docker compose watch` は `compose.yaml` の `develop.watch` セクションに基づき、ファイル変更を検知して sync（コピー）や rebuild（再ビルド）を自動実行する。macOS でバインドマウントのパフォーマンスが問題になる場合は watch の sync 方式が効果的。

### Q3: コンテナ内の node_modules とホストの IDE の補完が合わない場合は？

**A:** 名前付きボリュームで node_modules を分離している場合、ホストには node_modules が存在しないため IDE の補完が効かない。対策は以下の通り。
1. ホストでも `pnpm install` を実行（二重管理になるが最も簡単）
2. Dev Containers を使ってコンテナ内で VS Code を動かす（推奨）
3. ボリュームマウント自体を使わず、`docker compose watch` の sync を使う
4. `.vscode/settings.json` で TypeScript SDK のパスをコンテナ内のものに設定

### Q4: Apple Silicon (M1/M2/M3) で amd64 イメージを使う場合の注意点は？

**A:** Rosetta 2 エミュレーションにより動作するが、パフォーマンスが低下する。Docker Desktop の Settings > General > "Use Rosetta for x86_64/amd64 emulation on Apple Silicon" を有効にすると qemu より高速。ただし、可能な限り arm64 対応のイメージ (`:alpine`, `:slim` の多くは multi-arch) を使用すべき。`--platform linux/arm64` を明示するとネイティブ速度で動作する。

### Q5: docker compose の環境変数の優先順位は？

**A:** 以下の優先順位で適用される（上が高い）。
1. `docker compose run -e KEY=VALUE` (コマンドライン)
2. `environment:` セクション (compose.yaml)
3. `env_file:` で指定したファイル
4. Dockerfile の `ENV`
5. シェルの環境変数

`environment:` が `env_file:` より優先されるため、`env_file` でデフォルト値を設定し、`environment` で上書きするパターンが有効。

---

## 13. まとめ

| 構成要素 | 推奨 | 備考 |
|---------|------|------|
| ランタイム | Docker Desktop / OrbStack | macOS は OrbStack 推奨 |
| Dockerfile | マルチステージ | dev / production 分離 |
| ベースイメージ | slim / alpine | バージョン固定必須 |
| Compose 形式 | compose.yaml (v2) | docker-compose.yml は旧形式 |
| ファイル共有 | VirtioFS + 名前付きボリューム | node_modules は分離 |
| ファイル同期 | docker compose watch | バインドマウントの代替 |
| 環境変数 | .env + .env.local | .env.local は gitignore |
| ヘルスチェック | 必須 | depends_on の condition |
| セキュリティ | 非 root + no-new-privileges | 本番は必須 |
| クリーンアップ | docker system prune | 定期実行推奨 |
| 操作簡略化 | Makefile | チーム共通のインターフェース |
| CI 連携 | BuildKit + レイヤーキャッシュ | ビルド時間短縮 |

---

## 次に読むべきガイド

- [01-devcontainer.md](./01-devcontainer.md) -- Dev Container で VS Code をコンテナ内で動かす
- [02-local-services.md](./02-local-services.md) -- DB・キャッシュ等のローカルサービス構築
- [../03-team-setup/01-onboarding-automation.md](../03-team-setup/01-onboarding-automation.md) -- Docker を使ったオンボーディング自動化

---

## 参考文献

1. **Docker Compose Documentation** -- https://docs.docker.com/compose/ -- docker compose の公式ドキュメント。
2. **Docker Development Best Practices** -- https://docs.docker.com/develop/dev-best-practices/ -- 公式のベストプラクティスガイド。
3. **OrbStack** -- https://orbstack.dev/ -- macOS 向け高速 Docker 代替。
4. **Dockerfile Best Practices** -- https://docs.docker.com/build/building/best-practices/ -- マルチステージビルド等の公式ガイド。
5. **Docker Compose Watch** -- https://docs.docker.com/compose/file-watch/ -- ファイル監視・同期機能の公式ドキュメント。
6. **Docker Security** -- https://docs.docker.com/engine/security/ -- Docker セキュリティのベストプラクティス。
7. **BuildKit** -- https://docs.docker.com/build/buildkit/ -- 高速ビルドエンジンの公式ガイド。
