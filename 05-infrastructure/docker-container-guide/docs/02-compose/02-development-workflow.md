# Docker Compose 開発ワークフロー (Development Workflow)

> Docker Compose を活用した日常の開発ワークフローを最適化し、ホットリロード、デバッガ接続、CI 統合を実現する実践的なパターンを学ぶ。

## この章で学ぶこと

1. **ホットリロードとファイル同期の最適化** -- コンテナ内でのコード変更即時反映を実現し、快適な開発体験を構築する
2. **デバッグ環境の構築** -- VS Code / JetBrains からコンテナ内のプロセスにデバッガを接続する方法を習得する
3. **CI/CD パイプラインへの統合** -- Docker Compose をテスト・ビルドの CI/CD に組み込み、環境の一貫性を確保する

---

## 1. 開発ワークフローの全体像

```
+------------------------------------------------------------------+
|              Docker Compose 開発ワークフロー                        |
+------------------------------------------------------------------+
|                                                                  |
|  [ローカル開発]                                                   |
|  1. git clone && make setup                                      |
|  2. docker compose up -d (DB, Redis 等)                          |
|  3. エディタでコード編集                                          |
|     → バインドマウントでコンテナに即時反映                         |
|     → ホットリロードで自動再読み込み                               |
|  4. デバッガ接続 (ブレークポイント)                                |
|  5. docker compose logs -f で確認                                |
|                                                                  |
|  [テスト]                                                        |
|  1. docker compose --profile test run --rm test-runner           |
|  2. テスト用 DB を自動作成 → テスト → 破棄                        |
|                                                                  |
|  [CI/CD]                                                         |
|  1. docker compose -f docker-compose.ci.yml up -d               |
|  2. テスト実行 → カバレッジ → ビルド                              |
|  3. docker compose down -v (クリーンアップ)                       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. ホットリロードの設定

### 2.1 Node.js (Next.js / Vite) のホットリロード

```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
      - "24678:24678"   # Vite HMR WebSocket ポート
    volumes:
      # ソースコードをバインドマウント
      - .:/app
      # node_modules はボリュームで分離 (パフォーマンス)
      - node_modules:/app/node_modules
    environment:
      # Vite: コンテナ外からの HMR 接続を許可
      VITE_HMR_HOST: localhost
      VITE_HMR_PORT: 24678
      # Next.js: ファイル監視に polling を使用
      WATCHPACK_POLLING: "true"
      # Chokidar: polling fallback
      CHOKIDAR_USEPOLLING: "true"
    command: npm run dev

volumes:
  node_modules:
```

### 2.2 Dockerfile.dev (開発用)

```dockerfile
# Dockerfile.dev
FROM node:20-alpine

WORKDIR /app

# 依存関係のみ先にコピー (キャッシュ活用)
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile

# ソースコードはバインドマウントされるため COPY 不要
# COPY . . ← 開発用では不要

EXPOSE 3000

CMD ["pnpm", "dev"]
```

### 2.3 Docker Compose Watch (V2.22+)

```yaml
# docker-compose.yml (watch 機能)
services:
  app:
    build: .
    ports:
      - "3000:3000"
    develop:
      watch:
        # ソースファイル変更 → コンテナ内に同期
        - action: sync
          path: ./src
          target: /app/src
          ignore:
            - node_modules/
            - "**/*.test.ts"

        # package.json 変更 → rebuild
        - action: rebuild
          path: ./package.json

        # 設定ファイル変更 → コンテナ再起動
        - action: sync+restart
          path: ./config
          target: /app/config
```

```bash
# watch モードで起動
docker compose watch

# 通常起動 + watch
docker compose up -d && docker compose watch
```

### 2.4 ファイル同期方式の比較

| 方式 | 速度 | 設定難易度 | 双方向 | 適用場面 |
|------|------|-----------|--------|---------|
| バインドマウント | macOS: 遅い / Linux: 速い | 低 | あり | 一般的な開発 |
| Volume + sync | 中 | 中 | なし | macOS でのパフォーマンス改善 |
| Compose Watch | 速い | 低 | なし | Compose V2.22+ 推奨 |
| Mutagen | 非常に速い | 中 | 設定可 | macOS の大規模プロジェクト |
| Docker Desktop VirtioFS | 速い | 不要 | あり | Docker Desktop 利用時 |

---

## 3. デバッグ環境

### 3.1 Node.js デバッグ

```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
      - "9229:9229"     # Node.js デバッガポート
    volumes:
      - .:/app
      - node_modules:/app/node_modules
    command: >
      node --inspect=0.0.0.0:9229 node_modules/.bin/next dev
    # または
    # command: node --inspect=0.0.0.0:9229 src/index.ts
```

### 3.2 VS Code launch.json

```jsonc
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach to Node",
      "type": "node",
      "request": "attach",
      "port": 9229,
      "address": "localhost",
      "localRoot": "${workspaceFolder}",
      "remoteRoot": "/app",
      "restart": true,
      "skipFiles": ["<node_internals>/**"]
    },
    {
      "name": "Docker: Debug Tests",
      "type": "node",
      "request": "attach",
      "port": 9230,
      "address": "localhost",
      "localRoot": "${workspaceFolder}",
      "remoteRoot": "/app",
      "restart": true
    }
  ]
}
```

### 3.3 Python デバッグ (debugpy)

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
      - "5678:5678"     # debugpy ポート
    volumes:
      - .:/app
    command: >
      python -m debugpy --listen 0.0.0.0:5678 --wait-for-client
      -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

```jsonc
// .vscode/launch.json (Python)
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach to Python",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "/app"
        }
      ]
    }
  ]
}
```

---

## 4. テスト環境

### 4.1 テスト用 Compose 設定

```yaml
# docker-compose.yml
services:
  app:
    build: .
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_dev
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  # テストランナー (プロファイル)
  test:
    profiles: ["test"]
    build:
      context: .
      target: test
    environment:
      NODE_ENV: test
      DATABASE_URL: postgresql://postgres:postgres@db-test:5432/myapp_test
    depends_on:
      db-test:
        condition: service_healthy
    command: npm run test:ci

  # テスト専用 DB
  db-test:
    profiles: ["test"]
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_test
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    # tmpfs でテスト高速化 (永続化不要)
    tmpfs:
      - /var/lib/postgresql/data
```

### 4.2 テスト実行コマンド

```bash
# テスト実行
docker compose --profile test run --rm test

# テスト後のクリーンアップ
docker compose --profile test down -v

# E2E テスト (ブラウザ付き)
docker compose --profile e2e run --rm e2e-tests

# カバレッジレポート出力
docker compose --profile test run --rm \
  -v ./coverage:/app/coverage \
  test npm run test:coverage
```

---

## 5. CI/CD 統合

### 5.1 GitHub Actions での Compose 利用

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      # Docker Compose は GitHub Actions に標準搭載
      - name: Start services
        run: docker compose -f docker-compose.ci.yml up -d

      - name: Wait for DB
        run: |
          until docker compose -f docker-compose.ci.yml exec -T db \
            pg_isready -U postgres; do
            echo "Waiting for DB..."
            sleep 2
          done

      - name: Run migrations
        run: docker compose -f docker-compose.ci.yml exec -T app \
          npx prisma migrate deploy

      - name: Run tests
        run: docker compose -f docker-compose.ci.yml exec -T app \
          npm run test:ci

      - name: Run lint
        run: docker compose -f docker-compose.ci.yml exec -T app \
          npm run lint

      - name: Collect coverage
        run: docker compose -f docker-compose.ci.yml exec -T app \
          npm run test:coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage/lcov.info

      - name: Cleanup
        if: always()
        run: docker compose -f docker-compose.ci.yml down -v
```

### 5.2 CI 用 Compose ファイル

```yaml
# docker-compose.ci.yml
services:
  app:
    build:
      context: .
      target: test       # テストステージを使用
    environment:
      NODE_ENV: test
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp_test
      REDIS_URL: redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - coverage:/app/coverage

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_test
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 3s
      timeout: 3s
      retries: 10
    tmpfs:
      - /var/lib/postgresql/data   # CI ではメモリ上で高速化

  redis:
    image: redis:7-alpine
    tmpfs:
      - /data

volumes:
  coverage:
```

---

## 6. パフォーマンス最適化

### 6.1 macOS でのパフォーマンス改善

```
+------------------------------------------------------------------+
|          macOS でのファイル I/O パフォーマンス比較                   |
+------------------------------------------------------------------+
|                                                                  |
|  方式                    | npm install 時間 | HMR 反映速度       |
|--------------------------|-----------------|-------------------|
|  バインドマウント (grpcfuse)| 120秒          | 2-5秒             |
|  バインドマウント (VirtioFS)| 60秒           | 0.5-2秒           |
|  名前付き Volume          | 15秒            | N/A (同期なし)     |
|  Compose Watch           | 15秒            | 0.5-1秒           |
|  Mutagen                 | 15秒            | 0.3-0.5秒         |
|                                                                  |
|  推奨: VirtioFS + node_modules は Volume 分離                    |
|                                                                  |
+------------------------------------------------------------------+
```

### 6.2 ビルドキャッシュの活用

```dockerfile
# Dockerfile (マルチステージ + キャッシュ)
FROM node:20-alpine AS base
WORKDIR /app

# 依存関係レイヤー (変更頻度: 低)
FROM base AS deps
COPY package.json pnpm-lock.yaml ./
RUN --mount=type=cache,target=/root/.local/share/pnpm/store \
    corepack enable && pnpm install --frozen-lockfile

# 開発ステージ
FROM base AS development
COPY --from=deps /app/node_modules ./node_modules
COPY . .
CMD ["pnpm", "dev"]

# ビルドステージ
FROM base AS build
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN pnpm build

# テストステージ
FROM base AS test
COPY --from=deps /app/node_modules ./node_modules
COPY . .
CMD ["pnpm", "test"]

# 本番ステージ
FROM base AS production
COPY --from=build /app/dist ./dist
COPY --from=deps /app/node_modules ./node_modules
CMD ["node", "dist/index.js"]
```

---

## 7. 便利なスクリプトとタスク

### 7.1 Makefile の開発タスク

```makefile
# Makefile (Docker Compose 関連)
.PHONY: dev up down logs shell db-shell test clean

dev: up ## 開発サーバー起動 (Docker + ローカル)
	npm run dev

up: ## Docker サービス起動
	docker compose up -d
	@docker compose ps

down: ## Docker サービス停止
	docker compose down

logs: ## ログ表示
	docker compose logs -f --tail=100

shell: ## app コンテナに入る
	docker compose exec app sh

db-shell: ## DB に接続
	docker compose exec db psql -U postgres -d myapp_dev

db-dump: ## DB ダンプ
	docker compose exec db pg_dump -U postgres myapp_dev > backup.sql

db-restore: ## DB リストア
	cat backup.sql | docker compose exec -T db psql -U postgres myapp_dev

test: ## テスト (Docker 上)
	docker compose --profile test run --rm test

clean: ## 全削除 (ボリューム含む)
	docker compose down -v --remove-orphans
	docker system prune -f
```

---

## アンチパターン

### アンチパターン 1: 開発用コンテナに本番 Dockerfile をそのまま使用

```dockerfile
# NG: 本番用 Dockerfile をそのまま開発に使用
FROM node:20-alpine
WORKDIR /app
COPY . .               # 全ファイルコピー → バインドマウントと競合
RUN npm ci --production # devDependencies がない
CMD ["node", "dist/index.js"]  # ビルド済み前提

# OK: マルチステージで開発ステージを用意
FROM node:20-alpine AS development
WORKDIR /app
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install  # devDependencies も含む
# COPY は省略 → バインドマウントでソースを注入
CMD ["pnpm", "dev"]
```

**問題点**: 本番用 Dockerfile は最小限のファイルコピーと production 依存のみを含む。開発時に必要な devDependencies (テストフレームワーク、Lint ツール等) がなく、バインドマウントとの COPY 競合でファイル同期も壊れる。

### アンチパターン 2: CI で docker compose up のまま放置

```yaml
# NG: クリーンアップを忘れる
steps:
  - run: docker compose up -d
  - run: npm test
  # docker compose down を忘れている → 次回実行時にポート競合

# OK: always ステップでクリーンアップを保証
steps:
  - run: docker compose -f docker-compose.ci.yml up -d
  - run: npm test
  - name: Cleanup
    if: always()  # テスト失敗時も必ず実行
    run: docker compose -f docker-compose.ci.yml down -v --remove-orphans
```

**問題点**: CI 環境でコンテナを停止し忘れると、次の CI 実行時にポート競合やボリュームのゴミが残り、テストが不安定になる。`if: always()` で必ずクリーンアップを実行する。

---

## FAQ

### Q1: バインドマウントと Compose Watch のどちらを使うべきですか？

**A**: Linux ではバインドマウントが最も高速で設定も単純なため、そのまま使えばよい。macOS / Windows ではバインドマウントの I/O が遅いため、Compose Watch (V2.22+) を推奨する。Watch は変更を検知してコンテナ内に同期する方式で、ファイルシステムのオーバーヘッドを回避できる。ただし双方向同期ではないため、コンテナ内で生成されるファイル（ビルド成果物等）はホスト側に反映されない点に注意。

### Q2: コンテナ内でデバッガを使うとブレークポイントの行番号がずれるのですが？

**A**: ソースマップとパスマッピングの設定が原因であることが多い。VS Code の `launch.json` で `localRoot` (ホスト側パス) と `remoteRoot` (コンテナ内パス) が正しく対応していることを確認する。TypeScript の場合は `tsconfig.json` で `"sourceMap": true` を設定し、トランスパイル後のファイルではなくソースファイルにブレークポイントを設定する。

### Q3: CI で Compose のビルドが毎回遅いのですが、キャッシュを効かせる方法はありますか？

**A**: (1) GitHub Actions の `actions/cache` で Docker レイヤーキャッシュを保存する。(2) `docker compose build --build-arg BUILDKIT_INLINE_CACHE=1` でインラインキャッシュを有効化し、前回のイメージを `cache_from` に指定する。(3) Dockerfile で `RUN --mount=type=cache` を使い、npm/pip のキャッシュをビルド間で共有する。(4) GitHub Actions の場合は `setup-buildx-action` + `build-push-action` で GHCR にキャッシュを保存するのが最も効果的。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ホットリロード | バインドマウント + Volume 分離(node_modules)が基本 |
| Compose Watch | V2.22+ の公式同期機能。macOS/Windows で推奨 |
| デバッグ | `--inspect=0.0.0.0:9229` + VS Code Attach で実現 |
| テスト | profiles + tmpfs で高速なテスト専用 DB を構築 |
| CI 統合 | 専用 Compose ファイル + `if: always()` クリーンアップ |
| パフォーマンス | VirtioFS + Volume 分離 + BuildKit キャッシュで最適化 |
| マルチステージ | development / test / production ステージを分離 |
| Makefile | 日常タスクを make コマンドに集約 |

## 次に読むべきガイド

- [Compose 応用](./01-compose-advanced.md) -- プロファイル、healthcheck、環境変数の高度な設定
- [Docker Compose 基礎](./00-compose-basics.md) -- Compose の基本構文
- [コンテナセキュリティ](../06-security/00-container-security.md) -- 開発環境でも意識すべきセキュリティ

## 参考文献

1. **Docker Compose Watch** -- https://docs.docker.com/compose/file-watch/ -- Compose Watch 機能の公式ドキュメント
2. **VS Code Remote Debugging** -- https://code.visualstudio.com/docs/nodejs/nodejs-debugging -- VS Code からのリモートデバッグ設定
3. **Docker Build Cache** -- https://docs.docker.com/build/cache/ -- BuildKit のキャッシュ機構と最適化
4. **Docker Compose in CI** -- https://docs.docker.com/compose/ci-cd/ -- CI/CD 環境での Compose の使い方
