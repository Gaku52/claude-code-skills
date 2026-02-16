# Docker Compose 開発ワークフロー (Development Workflow)

> Docker Compose を活用した日常の開発ワークフローを最適化し、ホットリロード、デバッガ接続、CI 統合を実現する実践的なパターンを学ぶ。

## この章で学ぶこと

1. **ホットリロードとファイル同期の最適化** -- コンテナ内でのコード変更即時反映を実現し、快適な開発体験を構築する
2. **デバッグ環境の構築** -- VS Code / JetBrains からコンテナ内のプロセスにデバッガを接続する方法を習得する
3. **CI/CD パイプラインへの統合** -- Docker Compose をテスト・ビルドの CI/CD に組み込み、環境の一貫性を確保する
4. **E2E テスト環境の構築** -- Playwright / Cypress を使ったブラウザテストをコンテナ上で実行する
5. **開発効率を上げるスクリプトとツール** -- Makefile、シェルスクリプト、pre-commit フックで日常タスクを自動化する

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
|  [E2E テスト]                                                    |
|  1. docker compose --profile e2e up -d                          |
|  2. Playwright / Cypress でブラウザテスト実行                     |
|  3. スクリーンショット・動画の収集                                 |
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

### 2.3 Python (FastAPI / Django) のホットリロード

```yaml
# docker-compose.yml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - .:/app
    environment:
      PYTHONDONTWRITEBYTECODE: "1"
      PYTHONUNBUFFERED: "1"
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/src
```

```dockerfile
# Dockerfile.dev (Python)
FROM python:3.12-slim

WORKDIR /app

# システム依存パッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のインストール
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# ソースコードはバインドマウントで注入
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

### 2.4 Go のホットリロード (Air)

```yaml
# docker-compose.yml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8080:8080"
    volumes:
      - .:/app
    command: air -c .air.toml
```

```toml
# .air.toml
root = "."
tmp_dir = "tmp"

[build]
cmd = "go build -o ./tmp/main ./cmd/server"
bin = "./tmp/main"
full_bin = "./tmp/main"
include_ext = ["go", "tpl", "tmpl", "html"]
exclude_dir = ["assets", "tmp", "vendor", "node_modules"]
delay = 1000
```

```dockerfile
# Dockerfile.dev (Go)
FROM golang:1.22-alpine

WORKDIR /app

# Air (ホットリロードツール) のインストール
RUN go install github.com/air-verse/air@latest

# 依存関係のダウンロード
COPY go.mod go.sum ./
RUN go mod download

EXPOSE 8080

CMD ["air", "-c", ".air.toml"]
```

### 2.5 Ruby on Rails のホットリロード

```yaml
# docker-compose.yml
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - .:/app
      - bundle_cache:/usr/local/bundle
    environment:
      RAILS_ENV: development
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp_dev
    depends_on:
      db:
        condition: service_healthy
    command: bin/rails server -b 0.0.0.0

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  bundle_cache:
  pgdata:
```

### 2.6 Docker Compose Watch (V2.22+)

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

# 特定サービスのみ watch
docker compose watch app
```

### 2.7 Docker Compose Watch の詳細設定例

```yaml
# docker-compose.yml - フルスタックアプリの Watch 設定
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    develop:
      watch:
        # TypeScript ソースの同期
        - action: sync
          path: ./frontend/src
          target: /app/src
          ignore:
            - "**/*.test.tsx"
            - "**/*.spec.tsx"
            - "**/__tests__/"
            - "**/__mocks__/"

        # 静的アセットの同期
        - action: sync
          path: ./frontend/public
          target: /app/public

        # package.json / lockfile 変更 → 再ビルド
        - action: rebuild
          path: ./frontend/package.json
        - action: rebuild
          path: ./frontend/pnpm-lock.yaml

        # Vite 設定変更 → 再起動
        - action: sync+restart
          path: ./frontend/vite.config.ts
          target: /app/vite.config.ts

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    develop:
      watch:
        - action: sync
          path: ./backend/src
          target: /app/src

        - action: sync+restart
          path: ./backend/config
          target: /app/config

        - action: rebuild
          path: ./backend/package.json
```

### 2.8 ファイル同期方式の比較

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

### 3.4 Go デバッグ (Delve)

```yaml
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.debug
    ports:
      - "8080:8080"
      - "2345:2345"     # Delve デバッガポート
    volumes:
      - .:/app
    security_opt:
      - "seccomp:unconfined"   # Delve が ptrace を使用するために必要
    command: >
      dlv debug ./cmd/server --headless --listen=:2345
      --api-version=2 --accept-multiclient --continue
```

```dockerfile
# Dockerfile.debug (Go)
FROM golang:1.22

WORKDIR /app

# Delve デバッガのインストール
RUN go install github.com/go-delve/delve/cmd/dlv@latest

COPY go.mod go.sum ./
RUN go mod download

COPY . .

EXPOSE 8080 2345

CMD ["dlv", "debug", "./cmd/server", "--headless", "--listen=:2345", "--api-version=2", "--accept-multiclient", "--continue"]
```

```jsonc
// .vscode/launch.json (Go)
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Docker: Attach to Go (Delve)",
      "type": "go",
      "request": "attach",
      "mode": "remote",
      "remotePath": "/app",
      "port": 2345,
      "host": "127.0.0.1",
      "showLog": true
    }
  ]
}
```

### 3.5 JetBrains IDE でのリモートデバッグ

JetBrains IDE (IntelliJ IDEA, GoLand, PyCharm, WebStorm) でのリモートデバッグ設定は以下の手順で行う。

```
1. Run → Edit Configurations → + (追加)
2. "Remote JVM Debug" (Java) / "Go Remote" (Go) / "Python Debug Server" (Python) を選択
3. 設定:
   - Host: localhost
   - Port: <デバッガポート> (例: 9229, 5678, 2345)
   - Path mappings: ローカルパス ↔ コンテナ内パス
4. docker compose up -d でコンテナ起動
5. Run → Debug でアタッチ
```

### 3.6 デバッグ時のトラブルシューティング

```bash
# デバッガポートが開いているか確認
docker compose exec app sh -c "netstat -tlnp | grep 9229"

# デバッグモードでプロセスが起動しているか確認
docker compose exec app ps aux | grep inspect

# ネットワーク接続テスト（ホスト側から）
nc -zv localhost 9229

# コンテナのログでデバッグ情報を確認
docker compose logs -f app | grep -i "debugger"
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

# 特定のテストファイルのみ実行
docker compose --profile test run --rm \
  test npm test -- --testPathPattern="auth"

# ウォッチモードでテスト実行（開発中）
docker compose --profile test run --rm \
  test npm test -- --watch
```

### 4.3 E2E テスト環境 (Playwright)

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "3000:3000"
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_test
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    tmpfs:
      - /var/lib/postgresql/data

  # Playwright E2E テスト
  e2e:
    profiles: ["e2e"]
    image: mcr.microsoft.com/playwright:v1.42.0-jammy
    working_dir: /app
    volumes:
      - .:/app
      - node_modules:/app/node_modules
      - ./test-results:/app/test-results
      - ./playwright-report:/app/playwright-report
    environment:
      BASE_URL: http://app:3000
      CI: "true"
    depends_on:
      app:
        condition: service_healthy
    command: npx playwright test --reporter=html
    networks:
      - default

volumes:
  node_modules:
```

### 4.4 E2E テスト環境 (Cypress)

```yaml
services:
  # Cypress E2E テスト
  cypress:
    profiles: ["e2e"]
    image: cypress/included:13.6.0
    working_dir: /e2e
    volumes:
      - ./cypress:/e2e/cypress
      - ./cypress.config.ts:/e2e/cypress.config.ts
      - ./cypress/screenshots:/e2e/cypress/screenshots
      - ./cypress/videos:/e2e/cypress/videos
    environment:
      CYPRESS_baseUrl: http://app:3000
      CYPRESS_RECORD_KEY: ${CYPRESS_RECORD_KEY:-}
    depends_on:
      app:
        condition: service_healthy
    command: cypress run --browser chrome
```

### 4.5 テスト用データベースの並列実行

テストの並列実行時にデータベース競合を防ぐためのパターン。

```yaml
services:
  # テスト用 DB プール（各ワーカーに専用 DB を割り当て）
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
    tmpfs:
      - /var/lib/postgresql/data
    volumes:
      - ./scripts/create-test-databases.sh:/docker-entrypoint-initdb.d/create-test-databases.sh
```

```bash
#!/bin/bash
# scripts/create-test-databases.sh
# テストワーカー用のデータベースを事前作成

for i in $(seq 1 4); do
  psql -U postgres -c "CREATE DATABASE myapp_test_${i};"
done
echo "Test databases created: myapp_test_1 through myapp_test_4"
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

### 5.3 GitLab CI での Compose 利用

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_HOST: tcp://docker:2376
  DOCKER_TLS_CERTDIR: "/certs"
  COMPOSE_PROJECT_NAME: "ci-${CI_PIPELINE_ID}"

test:
  stage: test
  image: docker:24.0
  services:
    - docker:24.0-dind
  before_script:
    - apk add --no-cache docker-compose
  script:
    - docker compose -f docker-compose.ci.yml up -d
    - docker compose -f docker-compose.ci.yml exec -T app npm run test:ci
    - docker compose -f docker-compose.ci.yml exec -T app npm run lint
  after_script:
    - docker compose -f docker-compose.ci.yml down -v
  artifacts:
    reports:
      junit: test-results/junit.xml
    paths:
      - coverage/
    expire_in: 7 days
```

### 5.4 CI でのビルドキャッシュ戦略

```yaml
# .github/workflows/ci.yml (キャッシュ最適化版)
name: CI with Cache

on:
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      # Docker レイヤーキャッシュ
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Cache Docker layers
        uses: actions/cache@v4
        with:
          path: /tmp/.buildx-cache
          key: ${{ runner.os }}-buildx-${{ hashFiles('**/Dockerfile', '**/package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-buildx-

      # キャッシュを活用してビルド
      - name: Build test image
        run: |
          docker buildx build \
            --cache-from type=local,src=/tmp/.buildx-cache \
            --cache-to type=local,dest=/tmp/.buildx-cache-new,mode=max \
            --target test \
            --load \
            -t myapp:test \
            .

      - name: Start services
        run: docker compose -f docker-compose.ci.yml up -d

      - name: Run tests
        run: docker compose -f docker-compose.ci.yml exec -T app npm run test:ci

      # キャッシュのローテーション（サイズ肥大防止）
      - name: Rotate cache
        run: |
          rm -rf /tmp/.buildx-cache
          mv /tmp/.buildx-cache-new /tmp/.buildx-cache

      - name: Cleanup
        if: always()
        run: docker compose -f docker-compose.ci.yml down -v
```

### 5.5 CI での E2E テスト実行

```yaml
# .github/workflows/e2e.yml
name: E2E Tests

on:
  pull_request:
    branches: [main]

jobs:
  e2e:
    runs-on: ubuntu-latest
    timeout-minutes: 30

    steps:
      - uses: actions/checkout@v4

      - name: Start application
        run: docker compose -f docker-compose.ci.yml up -d

      - name: Wait for application to be ready
        run: |
          timeout 60 bash -c 'until curl -sf http://localhost:3000/health; do sleep 2; done'

      - name: Run E2E tests
        run: |
          docker compose --profile e2e run --rm \
            -e CI=true \
            -e BASE_URL=http://app:3000 \
            e2e

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 30

      - name: Upload screenshots
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: test-screenshots
          path: test-results/
          retention-days: 7

      - name: Cleanup
        if: always()
        run: docker compose -f docker-compose.ci.yml --profile e2e down -v
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

### 6.2 Docker Desktop の設定最適化

```json
// Docker Desktop settings.json
{
  "filesharingDirectories": [
    "/Users/<username>/projects"
  ],
  "memoryMiB": 8192,
  "cpus": 4,
  "diskSizeMiB": 65536,
  "swapMiB": 1024,
  "useVirtualizationFrameworkVirtioFS": true,
  "useVirtualizationFrameworkRosetta": true
}
```

### 6.3 node_modules のボリューム分離パターン

```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      # ソースコードはバインドマウント
      - .:/app
      # node_modules は名前付きボリュームで分離
      # → macOS のファイルI/Oボトルネックを回避
      - node_modules:/app/node_modules
      # .next キャッシュも分離（Next.js の場合）
      - next_cache:/app/.next
    # ボリュームの初期化（コンテナ起動時に依存関係をインストール）
    entrypoint: >
      sh -c "
        if [ ! -d /app/node_modules/.package-lock.json ]; then
          echo 'Installing dependencies...'
          pnpm install --frozen-lockfile
        fi
        exec pnpm dev
      "

volumes:
  node_modules:
  next_cache:
```

### 6.4 ビルドキャッシュの活用

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

### 6.5 ビルドコンテキストの最適化

```dockerignore
# .dockerignore
node_modules
.next
dist
build
coverage
test-results
playwright-report

# Git
.git
.gitignore

# IDE
.vscode
.idea

# Docker
docker-compose*.yml
Dockerfile*
.dockerignore

# ドキュメント
*.md
LICENSE

# テスト
__tests__
*.test.ts
*.spec.ts
cypress
```

---

## 7. 便利なスクリプトとタスク

### 7.1 Makefile の開発タスク

```makefile
# Makefile (Docker Compose 関連)
.PHONY: dev up down logs shell db-shell test clean setup help

# デフォルトターゲット
.DEFAULT_GOAL := help

help: ## ヘルプを表示
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## 初期セットアップ（.env コピー、依存関係インストール）
	@test -f .env || cp .env.example .env
	docker compose build
	docker compose up -d db redis
	@echo "Waiting for DB..."
	@sleep 5
	docker compose run --rm app npx prisma migrate deploy
	docker compose run --rm app npx prisma db seed
	@echo "Setup complete! Run 'make dev' to start."

dev: up ## 開発サーバー起動 (Docker + ローカル)
	npm run dev

up: ## Docker サービス起動
	docker compose up -d
	@docker compose ps

down: ## Docker サービス停止
	docker compose down

logs: ## ログ表示
	docker compose logs -f --tail=100

logs-app: ## アプリログのみ表示
	docker compose logs -f --tail=100 app

shell: ## app コンテナに入る
	docker compose exec app sh

db-shell: ## DB に接続
	docker compose exec db psql -U postgres -d myapp_dev

db-dump: ## DB ダンプ
	docker compose exec db pg_dump -U postgres myapp_dev > backup.sql

db-restore: ## DB リストア
	cat backup.sql | docker compose exec -T db psql -U postgres myapp_dev

db-reset: ## DB リセット（マイグレーション再実行）
	docker compose exec app npx prisma migrate reset --force

test: ## テスト (Docker 上)
	docker compose --profile test run --rm test

test-watch: ## テスト ウォッチモード
	docker compose --profile test run --rm test npm test -- --watch

test-e2e: ## E2E テスト
	docker compose --profile e2e run --rm e2e

lint: ## Lint 実行
	docker compose exec app npm run lint

format: ## コードフォーマット
	docker compose exec app npm run format

typecheck: ## 型チェック
	docker compose exec app npm run typecheck

clean: ## 全削除 (ボリューム含む)
	docker compose down -v --remove-orphans
	docker system prune -f

rebuild: ## イメージ再ビルドして起動
	docker compose build --no-cache
	docker compose up -d

update-deps: ## 依存関係を更新
	docker compose exec app pnpm update
	docker compose exec app pnpm install --frozen-lockfile
```

### 7.2 シェルスクリプトによるセットアップ自動化

```bash
#!/bin/bash
# scripts/setup.sh - プロジェクト初期セットアップ

set -euo pipefail

echo "=== Project Setup ==="

# 1. 環境変数ファイルのコピー
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "  Please edit .env with your settings"
fi

# 2. Docker Compose ビルド
echo "Building Docker images..."
docker compose build

# 3. サービス起動
echo "Starting services..."
docker compose up -d db redis

# 4. DB が起動するまで待機
echo "Waiting for database..."
until docker compose exec -T db pg_isready -U postgres 2>/dev/null; do
    printf "."
    sleep 1
done
echo " Ready!"

# 5. マイグレーション実行
echo "Running migrations..."
docker compose run --rm app npx prisma migrate deploy

# 6. シードデータ投入
echo "Seeding database..."
docker compose run --rm app npx prisma db seed

# 7. 全サービス起動
echo "Starting all services..."
docker compose up -d

echo ""
echo "=== Setup Complete ==="
echo "  App: http://localhost:3000"
echo "  DB:  localhost:5432"
echo ""
echo "Run 'make dev' or 'docker compose up -d' to start."
```

### 7.3 pre-commit フック

```bash
#!/bin/bash
# .git/hooks/pre-commit
# コミット前にコンテナ内で lint + typecheck を実行

echo "Running pre-commit checks..."

# lint チェック
if ! docker compose exec -T app npm run lint --quiet 2>/dev/null; then
    echo "Lint check failed. Please fix the errors and try again."
    exit 1
fi

# 型チェック
if ! docker compose exec -T app npm run typecheck 2>/dev/null; then
    echo "Type check failed. Please fix the errors and try again."
    exit 1
fi

echo "Pre-commit checks passed."
```

### 7.4 VS Code タスク設定

```jsonc
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Docker: Up",
      "type": "shell",
      "command": "docker compose up -d",
      "group": "build",
      "presentation": {
        "reveal": "silent"
      }
    },
    {
      "label": "Docker: Down",
      "type": "shell",
      "command": "docker compose down",
      "group": "build"
    },
    {
      "label": "Docker: Logs",
      "type": "shell",
      "command": "docker compose logs -f --tail=100",
      "group": "build",
      "isBackground": true
    },
    {
      "label": "Docker: Test",
      "type": "shell",
      "command": "docker compose --profile test run --rm test",
      "group": "test"
    },
    {
      "label": "Docker: Shell",
      "type": "shell",
      "command": "docker compose exec app sh",
      "group": "none"
    },
    {
      "label": "Docker: DB Reset",
      "type": "shell",
      "command": "docker compose exec app npx prisma migrate reset --force",
      "group": "none",
      "problemMatcher": []
    }
  ]
}
```

### 7.5 devcontainer.json 設定

VS Code の Dev Containers 拡張を使うことで、コンテナ内で直接開発できる。

```jsonc
// .devcontainer/devcontainer.json
{
  "name": "My App Dev Container",
  "dockerComposeFile": ["../docker-compose.yml", "docker-compose.devcontainer.yml"],
  "service": "app",
  "workspaceFolder": "/app",

  "features": {
    "ghcr.io/devcontainers/features/git:1": {},
    "ghcr.io/devcontainers/features/github-cli:1": {}
  },

  "customizations": {
    "vscode": {
      "extensions": [
        "dbaeumer.vscode-eslint",
        "esbenp.prettier-vscode",
        "Prisma.prisma",
        "ms-vscode.vscode-typescript-next"
      ],
      "settings": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "esbenp.prettier-vscode",
        "terminal.integrated.defaultProfile.linux": "zsh"
      }
    }
  },

  "forwardPorts": [3000, 5432, 6379],

  "postCreateCommand": "pnpm install && npx prisma generate",

  "remoteUser": "node"
}
```

```yaml
# .devcontainer/docker-compose.devcontainer.yml
services:
  app:
    build:
      context: ..
      dockerfile: Dockerfile.dev
    volumes:
      - ..:/app:cached
      - node_modules:/app/node_modules
    command: sleep infinity   # Dev Container はシェルを使うため
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp_dev
      REDIS_URL: redis://redis:6379

volumes:
  node_modules:
```

---

## 8. 開発環境の完全な構成例

### 8.1 フルスタック Web アプリケーション

```yaml
# docker-compose.yml - 完全な開発環境
services:
  # --- フロントエンド ---
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - frontend_node_modules:/app/node_modules
    environment:
      VITE_API_URL: http://localhost:8080
      VITE_HMR_HOST: localhost
    command: pnpm dev --host

  # --- バックエンド API ---
  api:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    ports:
      - "8080:8080"
      - "9229:9229"    # デバッガポート
    volumes:
      - ./backend:/app
      - backend_node_modules:/app/node_modules
    environment:
      NODE_ENV: development
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp_dev
      REDIS_URL: redis://redis:6379
      SMTP_HOST: mailhog
      SMTP_PORT: 1025
      S3_ENDPOINT: http://minio:9000
      S3_ACCESS_KEY: minioadmin
      S3_SECRET_KEY: minioadmin
      S3_BUCKET: uploads
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    command: >
      node --inspect=0.0.0.0:9229 node_modules/.bin/tsx watch src/index.ts

  # --- データベース ---
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
    ports:
      - "5432:5432"    # 開発時は外部からアクセス可能にする
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

  # --- Redis ---
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # --- オブジェクトストレージ (S3 互換) ---
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"    # Console
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    command: server /data --console-address ":9001"

  # --- メールキャッチャー ---
  mailhog:
    image: mailhog/mailhog:latest
    profiles: ["debug"]
    ports:
      - "1025:1025"    # SMTP
      - "8025:8025"    # Web UI

  # --- DB 管理ツール ---
  pgadmin:
    image: dpage/pgadmin4:latest
    profiles: ["debug"]
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"

  # --- Redis 管理ツール ---
  redis-commander:
    image: rediscommander/redis-commander:latest
    profiles: ["debug"]
    environment:
      REDIS_HOSTS: local:redis:6379
    ports:
      - "8081:8081"

volumes:
  frontend_node_modules:
  backend_node_modules:
  pgdata:
  redis_data:
  minio_data:
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

### アンチパターン 3: デバッガポートを本番に残す

```yaml
# NG: デバッガポートが本番で公開されたまま
services:
  app:
    ports:
      - "3000:3000"
      - "9229:9229"    # デバッガポート → 本番では絶対に不可
    command: node --inspect=0.0.0.0:9229 dist/index.js

# OK: デバッガポートは開発 override のみ
# docker-compose.yml (ベース)
services:
  app:
    ports:
      - "3000:3000"
    command: node dist/index.js

# docker-compose.override.yml (開発)
services:
  app:
    ports:
      - "9229:9229"
    command: node --inspect=0.0.0.0:9229 node_modules/.bin/tsx watch src/index.ts
```

**問題点**: `--inspect` オプションを有効にしたまま本番にデプロイすると、任意のコードが実行可能なデバッグインターフェースが外部に公開される。これは最も深刻なセキュリティ脆弱性の一つである。

### アンチパターン 4: バインドマウントで node_modules を共有

```yaml
# NG: ホストの node_modules がコンテナ内を上書き
services:
  app:
    volumes:
      - .:/app        # node_modules も含まれてしまう
    # → ホスト(macOS)のバイナリがLinuxコンテナで動かない

# OK: node_modules はボリュームで分離
services:
  app:
    volumes:
      - .:/app
      - node_modules:/app/node_modules   # コンテナ専用
volumes:
  node_modules:
```

**問題点**: ホスト（macOS/Windows）でインストールされたネイティブバイナリ（`bcrypt`, `sharp` 等）はLinuxコンテナ内では動作しない。`node_modules` をボリュームで分離することで、コンテナ内で正しいプラットフォーム用のバイナリが使用される。

---

## FAQ

### Q1: バインドマウントと Compose Watch のどちらを使うべきですか？

**A**: Linux ではバインドマウントが最も高速で設定も単純なため、そのまま使えばよい。macOS / Windows ではバインドマウントの I/O が遅いため、Compose Watch (V2.22+) を推奨する。Watch は変更を検知してコンテナ内に同期する方式で、ファイルシステムのオーバーヘッドを回避できる。ただし双方向同期ではないため、コンテナ内で生成されるファイル（ビルド成果物等）はホスト側に反映されない点に注意。

### Q2: コンテナ内でデバッガを使うとブレークポイントの行番号がずれるのですが？

**A**: ソースマップとパスマッピングの設定が原因であることが多い。VS Code の `launch.json` で `localRoot` (ホスト側パス) と `remoteRoot` (コンテナ内パス) が正しく対応していることを確認する。TypeScript の場合は `tsconfig.json` で `"sourceMap": true` を設定し、トランスパイル後のファイルではなくソースファイルにブレークポイントを設定する。

### Q3: CI で Compose のビルドが毎回遅いのですが、キャッシュを効かせる方法はありますか？

**A**: (1) GitHub Actions の `actions/cache` で Docker レイヤーキャッシュを保存する。(2) `docker compose build --build-arg BUILDKIT_INLINE_CACHE=1` でインラインキャッシュを有効化し、前回のイメージを `cache_from` に指定する。(3) Dockerfile で `RUN --mount=type=cache` を使い、npm/pip のキャッシュをビルド間で共有する。(4) GitHub Actions の場合は `setup-buildx-action` + `build-push-action` で GHCR にキャッシュを保存するのが最も効果的。

### Q4: 開発環境と本番環境で Dockerfile を分けるべきですか？

**A**: 分けるべきではない。マルチステージビルドを使い、1つの Dockerfile 内で `development`、`test`、`production` のステージを定義する。`docker compose` 側で `build.target` を指定してステージを切り替える。これにより、Dockerfile のメンテナンスコストが下がり、環境間の差異を最小限に抑えられる。

### Q5: Docker Desktop の VirtioFS と gRPC FUSE の違いは？

**A**: VirtioFS は macOS の Virtualization.framework を使った高速なファイル共有方式で、gRPC FUSE（旧方式）と比較して 2〜5 倍のパフォーマンス向上が期待できる。Docker Desktop 4.15+ でデフォルトで有効。Settings → General → "Use VirtioFS" で確認・設定できる。大規模プロジェクトでは VirtioFS + node_modules のボリューム分離が最も効果的な組み合わせである。

### Q6: Dev Containers と通常の Docker Compose 開発のどちらが良いですか？

**A**: チーム全員が VS Code を使うなら Dev Containers が統一された開発体験を提供できる。エディタが混在するチームではバインドマウント方式が柔軟。Dev Containers のメリットは、エディタの拡張機能やターミナルがコンテナ内で動作するため、ホスト環境に依存しない完全に同一の開発環境が実現できること。デメリットは VS Code 必須であること、コンテナ再構築時の待ち時間が発生すること。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ホットリロード | バインドマウント + Volume 分離(node_modules)が基本 |
| Compose Watch | V2.22+ の公式同期機能。macOS/Windows で推奨 |
| デバッグ | `--inspect=0.0.0.0:9229` + VS Code Attach で実現 |
| テスト | profiles + tmpfs で高速なテスト専用 DB を構築 |
| E2E テスト | Playwright/Cypress をコンテナ化して安定実行 |
| CI 統合 | 専用 Compose ファイル + `if: always()` クリーンアップ |
| パフォーマンス | VirtioFS + Volume 分離 + BuildKit キャッシュで最適化 |
| マルチステージ | development / test / production ステージを分離 |
| Makefile | 日常タスクを make コマンドに集約 |
| Dev Containers | VS Code + コンテナで完全統一された開発環境 |

## 次に読むべきガイド

- [Compose 応用](./01-compose-advanced.md) -- プロファイル、healthcheck、環境変数の高度な設定
- [Docker Compose 基礎](./00-compose-basics.md) -- Compose の基本構文
- [コンテナセキュリティ](../06-security/00-container-security.md) -- 開発環境でも意識すべきセキュリティ

## 参考文献

1. **Docker Compose Watch** -- https://docs.docker.com/compose/file-watch/ -- Compose Watch 機能の公式ドキュメント
2. **VS Code Remote Debugging** -- https://code.visualstudio.com/docs/nodejs/nodejs-debugging -- VS Code からのリモートデバッグ設定
3. **Docker Build Cache** -- https://docs.docker.com/build/cache/ -- BuildKit のキャッシュ機構と最適化
4. **Docker Compose in CI** -- https://docs.docker.com/compose/ci-cd/ -- CI/CD 環境での Compose の使い方
5. **Dev Containers** -- https://containers.dev/ -- Development Containers の公式仕様
6. **Playwright Docker** -- https://playwright.dev/docs/docker -- Playwright のコンテナ実行ガイド
7. **Docker Desktop VirtioFS** -- https://docs.docker.com/desktop/settings/mac/ -- VirtioFS の設定と最適化
