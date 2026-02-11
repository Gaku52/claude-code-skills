# 開発用 Docker

> Docker Desktop、Dev Containers、docker compose を活用し、再現性の高い開発環境を構築するための実践ガイド。

## この章で学ぶこと

1. Docker Desktop のインストール・リソース設定・パフォーマンスチューニング
2. docker compose による開発環境の構築と管理
3. ホットリロード・ボリュームマウント・ネットワーク設計の実践テクニック

---

## 1. Docker Desktop のセットアップ

### 1.1 インストール

```bash
# macOS
brew install --cask docker

# Windows (WSL2 バックエンド推奨)
winget install Docker.DockerDesktop

# Linux (Docker Engine)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# 再ログイン後に有効化
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
│                                      │
│ Settings → Docker Engine             │
│  {                                   │
│    "builder": {                      │
│      "gc": {                         │
│        "enabled": true,              │
│        "defaultKeepStorage": "20GB"  │
│      }                               │
│    }                                 │
│  }                                   │
└─────────────────────────────────────┘
```

### 1.3 代替ツール

| ツール | OS | 特徴 | 料金 |
|--------|-----|------|------|
| Docker Desktop | 全OS | 公式・GUI付き | 個人無料/企業有料 |
| OrbStack | macOS | 軽量・高速 | 個人無料 |
| Rancher Desktop | 全OS | OSS・containerd対応 | 無料 |
| Podman Desktop | 全OS | rootless・デーモンレス | 無料 |
| Colima | macOS/Linux | CLI専用・軽量 | 無料 |

---

## 2. 開発用 Dockerfile

### 2.1 マルチステージビルド

```dockerfile
# ─── ステージ 1: 依存インストール ───
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile

# ─── ステージ 2: 開発環境 ───
FROM node:20-slim AS dev
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
CMD ["pnpm", "dev"]

# ─── ステージ 3: ビルド ───
FROM node:20-slim AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm build

# ─── ステージ 4: 本番 ───
FROM node:20-slim AS production
WORKDIR /app
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 appuser
COPY --from=build --chown=appuser:nodejs /app/dist ./dist
COPY --from=build --chown=appuser:nodejs /app/node_modules ./node_modules
USER appuser
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

### 2.2 ステージの構造

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
  ┌────────────────────┬───────────┐
  │ ステージ            │ サイズ     │
  ├────────────────────┼───────────┤
  │ dev (全依存 + src)  │ ~800 MB   │
  │ production (最小)   │ ~150 MB   │
  └────────────────────┴───────────┘
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
      target: dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app                           # ソースコードマウント
      - /app/node_modules                # node_modules はコンテナ内を使用
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/mydb
      - REDIS_URL=redis://redis:6379
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

  # ─── データベース ───
  db:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  # ─── キャッシュ ───
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 3.2 よく使うコマンド

```bash
# ─── 起動 ───
docker compose up -d                    # バックグラウンド起動
docker compose up --build               # リビルドして起動
docker compose watch                    # ファイル変更を監視

# ─── ログ ───
docker compose logs -f app              # app のログを追跡
docker compose logs --tail 100 db       # db の最新100行

# ─── 操作 ───
docker compose exec app sh              # コンテナ内でシェル
docker compose exec db psql -U postgres # DB に接続
docker compose run --rm app pnpm test   # 一時コンテナでテスト

# ─── 停止・削除 ───
docker compose down                     # 停止
docker compose down -v                  # ボリュームも削除
docker compose down --rmi all           # イメージも削除

# ─── クリーンアップ ───
docker system prune -af                 # 不要な全リソース削除
docker volume prune                     # 未使用ボリューム削除
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
  ┌──────────────────┬──────────┐
  │ 方式              │ 速度     │
  ├──────────────────┼──────────┤
  │ ネイティブ        │ 1x (基準)│
  │ VirtioFS          │ 1.5-2x  │
  │ gRPC FUSE         │ 3-5x    │
  │ 名前付きボリューム │ 1.1x    │
  └──────────────────┴──────────┘
```

### 4.2 パフォーマンス最適化

```yaml
# compose.yaml のベストプラクティス
services:
  app:
    volumes:
      # ソースコードはマウント
      - .:/app

      # node_modules は名前付きボリューム (高速)
      - node_modules:/app/node_modules

      # ビルド出力も名前付きボリューム
      - next_cache:/app/.next

volumes:
  node_modules:
  next_cache:
```

---

## 5. 環境変数管理

### 5.1 .env ファイルの構成

```bash
# .env (docker compose が自動読込)
COMPOSE_PROJECT_NAME=my-project
NODE_ENV=development

# .env.local (個人設定 — .gitignore に追加)
GITHUB_TOKEN=ghp_xxxxx
AWS_ACCESS_KEY_ID=AKIA...

# 使い分け:
# compose.yaml: env_file で明示的に指定
# .env: チーム共有のデフォルト値
# .env.local: 個人の秘密情報
```

```yaml
# compose.yaml
services:
  app:
    env_file:
      - .env
      - .env.local    # 存在しなくてもエラーにしない場合
    environment:
      # env_file の値を上書き
      - LOG_LEVEL=debug
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
```

---

## 7. アンチパターン

### 7.1 開発用と本番用で同じ Dockerfile を使う

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

✅ 正しいアプローチ:
  - マルチステージビルドで分離
  - dev ステージ: 全依存 + ホットリロード
  - production ステージ: 最小依存 + ビルド成果物のみ
  - --target フラグで使い分け
```

### 7.2 ボリュームデータのバックアップを取らない

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
```

---

## 8. FAQ

### Q1: Docker Desktop と OrbStack、どちらを使うべき？

**A:** macOS ユーザーには OrbStack を推奨。Docker Desktop と比較してメモリ使用量が半分以下、起動が数秒で完了し、ファイルシステムのパフォーマンスも優れている。Docker CLI と完全互換なので移行コストはゼロ。Docker Desktop のライセンス問題（大企業は有料）を避けられるメリットもある。

### Q2: `docker compose up` と `docker compose watch` の違いは？

**A:** `docker compose up` はコンテナ起動のみ。ボリュームマウントでファイル変更は反映されるが、`package.json` の変更などは手動リビルドが必要。`docker compose watch` は `compose.yaml` の `develop.watch` セクションに基づき、ファイル変更を検知して sync（コピー）や rebuild（再ビルド）を自動実行する。

### Q3: コンテナ内の node_modules とホストの IDE の補完が合わない場合は？

**A:** 名前付きボリュームで node_modules を分離している場合、ホストには node_modules が存在しないため IDE の補完が効かない。対策は以下の通り。
1. ホストでも `pnpm install` を実行（二重管理になるが最も簡単）
2. Dev Containers を使ってコンテナ内で VS Code を動かす（推奨）
3. ボリュームマウント自体を使わず、`docker compose watch` の sync を使う

---

## 9. まとめ

| 構成要素 | 推奨 | 備考 |
|---------|------|------|
| ランタイム | Docker Desktop / OrbStack | macOS は OrbStack 推奨 |
| Dockerfile | マルチステージ | dev / production 分離 |
| Compose 形式 | compose.yaml (v2) | docker-compose.yml は旧形式 |
| ファイル共有 | VirtioFS + 名前付きボリューム | node_modules は分離 |
| 環境変数 | .env + .env.local | .env.local は gitignore |
| ヘルスチェック | 必須 | depends_on の condition |
| クリーンアップ | docker system prune | 定期実行推奨 |

---

## 次に読むべきガイド

- [01-devcontainer.md](./01-devcontainer.md) — Dev Container で VS Code をコンテナ内で動かす
- [02-local-services.md](./02-local-services.md) — DB・キャッシュ等のローカルサービス構築
- [../03-team-setup/01-onboarding-automation.md](../03-team-setup/01-onboarding-automation.md) — Docker を使ったオンボーディング自動化

---

## 参考文献

1. **Docker Compose Documentation** — https://docs.docker.com/compose/ — docker compose の公式ドキュメント。
2. **Docker Development Best Practices** — https://docs.docker.com/develop/dev-best-practices/ — 公式のベストプラクティスガイド。
3. **OrbStack** — https://orbstack.dev/ — macOS 向け高速 Docker 代替。
4. **Dockerfile Best Practices** — https://docs.docker.com/build/building/best-practices/ — マルチステージビルド等の公式ガイド。
