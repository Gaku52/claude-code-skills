# Docker Compose 応用 (Compose Advanced)

> プロファイル、depends_on の高度な制御、healthcheck、環境変数の管理パターンなど、Docker Compose の応用機能を活用してプロダクション品質の構成を構築する。

## この章で学ぶこと

1. **プロファイルによるサービスの選択的起動** -- 開発・テスト・監視など、用途に応じたサービスのグルーピングと選択的起動を実装する
2. **depends_on と healthcheck の高度な制御** -- サービス間の依存関係を精密に管理し、確実な起動順序を保証する
3. **環境変数と設定の管理パターン** -- 複数環境での設定切り替え、シークレット管理、ファイルのオーバーライドを実践する
4. **YAML アンカーと Extension Fields の活用** -- 設定の DRY 化と保守性の向上を実現する
5. **リソース制限・ロギング・セキュリティ設定** -- プロダクション品質の Compose 構成を構築する

---

## 1. プロファイル (Profiles)

### 1.1 プロファイルの概要

Docker Compose のプロファイル機能は、サービスを論理的にグルーピングし、必要に応じて選択的に起動する仕組みである。開発ツール、テストランナー、監視スタック、デバッグ用ツールなど、常時稼働が不要なサービスを管理するのに最適である。

プロファイルが指定されていないサービスは「デフォルト」として常に起動される。プロファイルが指定されたサービスは、明示的にそのプロファイルを有効化しない限り起動されない。

```
+------------------------------------------------------------------+
|              プロファイルによるサービスのグルーピング                  |
+------------------------------------------------------------------+
|                                                                  |
|  [デフォルト] (プロファイルなし → 常に起動)                         |
|    app, db, redis                                                |
|                                                                  |
|  [debug プロファイル] (--profile debug で起動)                    |
|    pgadmin, redis-commander                                      |
|                                                                  |
|  [monitoring プロファイル] (--profile monitoring で起動)           |
|    prometheus, grafana, alertmanager                              |
|                                                                  |
|  [test プロファイル] (--profile test で起動)                      |
|    test-runner, db-test, test-mail                                |
|                                                                  |
|  [seed プロファイル] (--profile seed で起動)                      |
|    db-seeder, sample-data-loader                                 |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.2 プロファイルの設定

```yaml
# docker-compose.yml
services:
  # プロファイルなし → 常に起動
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
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine

  # debug プロファイル
  pgadmin:
    image: dpage/pgadmin4:latest
    profiles: ["debug"]
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"

  redis-commander:
    image: rediscommander/redis-commander:latest
    profiles: ["debug"]
    environment:
      REDIS_HOSTS: local:redis:6379
    ports:
      - "8081:8081"

  # monitoring プロファイル
  prometheus:
    image: prom/prometheus:latest
    profiles: ["monitoring"]
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    profiles: ["monitoring"]
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana

  alertmanager:
    image: prom/alertmanager:latest
    profiles: ["monitoring"]
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"

  # test プロファイル
  test-runner:
    build:
      context: .
      target: test
    profiles: ["test"]
    depends_on:
      db:
        condition: service_healthy
    command: npm test

  # seed プロファイル (初期データ投入)
  db-seeder:
    build:
      context: .
      dockerfile: Dockerfile.seed
    profiles: ["seed"]
    depends_on:
      db:
        condition: service_healthy
    command: npx prisma db seed
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp

volumes:
  grafana_data:
```

### 1.3 プロファイルの起動コマンド

```bash
# デフォルトサービスのみ起動
docker compose up -d

# デバッグツールを追加起動
docker compose --profile debug up -d

# 複数プロファイル同時
docker compose --profile debug --profile monitoring up -d

# テスト実行
docker compose --profile test run --rm test-runner

# 環境変数で指定
COMPOSE_PROFILES=debug,monitoring docker compose up -d

# プロファイル指定のサービスのみ停止
docker compose --profile debug stop

# 特定プロファイルのサービス一覧を確認
docker compose --profile test ps

# 全プロファイルを含む全サービスの状態確認
docker compose --profile "*" ps
```

### 1.4 プロファイルの実践的な活用パターン

#### パターン A: 開発/ステージング/本番の切り替え

```yaml
services:
  app:
    build: .
    ports:
      - "3000:3000"

  # 開発専用のメールキャッチャー
  mailhog:
    image: mailhog/mailhog:latest
    profiles: ["dev"]
    ports:
      - "1025:1025"
      - "8025:8025"    # Web UI

  # ステージング用の負荷テストツール
  k6:
    image: grafana/k6:latest
    profiles: ["staging"]
    volumes:
      - ./tests/load:/scripts
    command: run /scripts/load-test.js

  # 本番用のログ収集
  fluentd:
    image: fluent/fluentd:v1.16
    profiles: ["production"]
    volumes:
      - ./fluentd/conf:/fluentd/etc
    ports:
      - "24224:24224"
```

#### パターン B: データベースマイグレーション管理

```yaml
services:
  db:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  # マイグレーション実行
  migrate:
    build: .
    profiles: ["migrate"]
    depends_on:
      db:
        condition: service_healthy
    command: npx prisma migrate deploy
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp

  # マイグレーション生成（開発時のみ）
  migrate-dev:
    build: .
    profiles: ["migrate-dev"]
    depends_on:
      db:
        condition: service_healthy
    command: npx prisma migrate dev
    volumes:
      - ./prisma:/app/prisma
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp

  # DB スキーマのリセット（危険操作）
  db-reset:
    build: .
    profiles: ["db-reset"]
    depends_on:
      db:
        condition: service_healthy
    command: npx prisma migrate reset --force
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp
```

---

## 2. depends_on と healthcheck

### 2.1 depends_on の 3 つの条件

Docker Compose では、サービス間の依存関係を 3 つの条件で制御できる。これにより、単純な起動順序の制御から、ヘルスチェックの通過やワンショットタスクの完了待ちまで、柔軟な制御が可能になる。

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy    # ヘルスチェックが通るまで待機
      redis:
        condition: service_started    # コンテナが起動したら OK
      migration:
        condition: service_completed_successfully  # 正常終了まで待機
        restart: true                 # 再起動時も待機
```

各条件の詳細な動作は以下の通りである。

| 条件 | 動作 | 典型的な用途 |
|------|------|-------------|
| `service_started` | コンテナのプロセスが起動したら即座に次へ進む | 起動が速いサービス（Redis 等） |
| `service_healthy` | healthcheck が passing になるまで待機する | DB、Elasticsearch 等の初期化に時間がかかるサービス |
| `service_completed_successfully` | コンテナが終了コード 0 で完了するまで待機する | マイグレーション、シード、初期化スクリプト |

### 2.2 healthcheck の詳細設定

各種データストア・サービスに対する healthcheck の実装例を示す。ヘルスチェックは、サービスが「起動した」だけでなく「リクエストを受け付けられる状態になった」ことを確認するために不可欠である。

```yaml
services:
  # PostgreSQL
  db:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d myapp"]
      interval: 5s        # チェック間隔
      timeout: 5s          # タイムアウト
      retries: 5           # 失敗許容回数
      start_period: 30s    # 起動猶予時間 (この間の失敗はカウントしない)

  # MySQL
  mysql:
    image: mysql:8.0
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-p$$MYSQL_ROOT_PASSWORD"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 30s

  # MariaDB
  mariadb:
    image: mariadb:11
    healthcheck:
      test: ["CMD", "healthcheck.sh", "--connect", "--innodb_initialized"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 30s

  # Redis
  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # Redis (パスワード付き)
  redis-auth:
    image: redis:7-alpine
    command: redis-server --requirepass mypassword
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "mypassword", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # MongoDB
  mongodb:
    image: mongo:7
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s

  # HTTP サービス
  api:
    build: .
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s

  # HTTP サービス (wget を使う場合 - curl がないイメージ向け)
  api-alpine:
    build: .
    healthcheck:
      test: ["CMD-SHELL", "wget --no-verbose --tries=1 --spider http://localhost:3000/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3
      start_period: 15s

  # Elasticsearch
  elasticsearch:
    image: elasticsearch:8.12.0
    healthcheck:
      test: ["CMD-SHELL", "curl -s http://localhost:9200/_cluster/health | grep -q '\"status\":\"green\"\\|\"status\":\"yellow\"'"]
      interval: 10s
      timeout: 10s
      retries: 10
      start_period: 60s

  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3-management-alpine
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
      interval: 10s
      timeout: 10s
      retries: 5
      start_period: 30s

  # Kafka (KRaft mode)
  kafka:
    image: bitnami/kafka:3.7
    healthcheck:
      test: ["CMD-SHELL", "kafka-broker-api-versions.sh --bootstrap-server localhost:9092 > /dev/null 2>&1"]
      interval: 10s
      timeout: 10s
      retries: 10
      start_period: 60s

  # MinIO (S3互換ストレージ)
  minio:
    image: minio/minio:latest
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 15s
```

### 2.3 依存関係の可視化

```
+------------------------------------------------------------------+
|              サービス依存関係グラフ                                  |
+------------------------------------------------------------------+
|                                                                  |
|  migration ──(completed)──> db ──(healthy)──+                    |
|                              ^               |                   |
|                              |               v                   |
|  seed ──(completed)──────────+             app ──> redis         |
|                                              |    (started)      |
|                                              v                   |
|                                           worker ──> redis       |
|                                           (started)              |
|                                                                  |
+------------------------------------------------------------------+
```

### 2.4 複雑な依存関係チェーンの実装

実際のアプリケーションでは、DB 起動 → マイグレーション → シードデータ投入 → アプリ起動という一連の流れが必要になる。以下はその完全な実装例である。

```yaml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 30s
    volumes:
      - pgdata:/var/lib/postgresql/data

  # ステップ 1: マイグレーション実行
  migration:
    build:
      context: .
      target: migration
    depends_on:
      db:
        condition: service_healthy
    command: npx prisma migrate deploy
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp

  # ステップ 2: シードデータ投入 (マイグレーション完了後)
  seed:
    build:
      context: .
      target: seed
    depends_on:
      migration:
        condition: service_completed_successfully
    command: npx prisma db seed
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp

  # ステップ 3: アプリ起動 (シード完了後)
  app:
    build:
      context: .
      target: production
    depends_on:
      seed:
        condition: service_completed_successfully
      redis:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp
      REDIS_URL: redis://redis:6379

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # ワーカープロセス (アプリと同じ依存関係)
  worker:
    build:
      context: .
      target: production
    depends_on:
      seed:
        condition: service_completed_successfully
      redis:
        condition: service_healthy
    command: node dist/worker.js
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp
      REDIS_URL: redis://redis:6379

volumes:
  pgdata:
```

### 2.5 ヘルスチェックのカスタムスクリプト

複雑なヘルスチェックが必要な場合は、専用のスクリプトを用意してコンテナにコピーする。

```bash
#!/bin/bash
# healthcheck.sh - 複合的なヘルスチェック

# 1. HTTP エンドポイントの確認
if ! curl -sf http://localhost:3000/health > /dev/null 2>&1; then
    echo "HTTP health check failed"
    exit 1
fi

# 2. DB 接続の確認
if ! node -e "
  const { PrismaClient } = require('@prisma/client');
  const prisma = new PrismaClient();
  prisma.\$queryRaw\`SELECT 1\`.then(() => process.exit(0)).catch(() => process.exit(1));
" 2>/dev/null; then
    echo "Database connection check failed"
    exit 1
fi

# 3. Redis 接続の確認
if ! node -e "
  const Redis = require('ioredis');
  const redis = new Redis(process.env.REDIS_URL);
  redis.ping().then(() => process.exit(0)).catch(() => process.exit(1));
" 2>/dev/null; then
    echo "Redis connection check failed"
    exit 1
fi

echo "All health checks passed"
exit 0
```

```dockerfile
# Dockerfile
FROM node:20-alpine AS production
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
COPY healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh
HEALTHCHECK --interval=15s --timeout=10s --retries=3 --start-period=30s \
    CMD /usr/local/bin/healthcheck.sh
```

---

## 3. 環境変数の管理

### 3.1 環境変数の優先順位

Docker Compose では、環境変数の値が複数のソースから供給される場合、明確な優先順位が定められている。

```
+------------------------------------------------------------------+
|           環境変数の優先順位 (上が最優先)                            |
+------------------------------------------------------------------+
|                                                                  |
|  1. docker compose run -e VAR=value  (CLI 直接指定)              |
|  2. environment: セクション                                      |
|  3. --env-file で指定したファイル                                  |
|  4. env_file: セクション                                         |
|  5. Dockerfile の ENV                                            |
|  6. シェルの環境変数 (.env ファイル経由)                           |
|                                                                  |
+------------------------------------------------------------------+
```

### 3.2 .env ファイルの使い分け

```bash
# .env (Compose 変数の展開用。docker compose が自動読み込み)
COMPOSE_PROJECT_NAME=myapp
POSTGRES_VERSION=16
NODE_VERSION=20
APP_PORT=3000

# .env.development (アプリ用。env_file で明示的に読み込む)
NODE_ENV=development
DATABASE_URL=postgresql://postgres:postgres@db:5432/myapp_dev
REDIS_URL=redis://redis:6379
LOG_LEVEL=debug
CORS_ORIGIN=http://localhost:3000
SESSION_SECRET=dev-secret-key-not-for-production
SMTP_HOST=mailhog
SMTP_PORT=1025

# .env.staging (ステージング用)
NODE_ENV=staging
DATABASE_URL=postgresql://staging_user:staging_pass@db:5432/myapp_staging
REDIS_URL=redis://redis:6379
LOG_LEVEL=info
CORS_ORIGIN=https://staging.example.com

# .env.production (本番用)
NODE_ENV=production
DATABASE_URL=postgresql://user:password@db-prod:5432/myapp
LOG_LEVEL=warn
CORS_ORIGIN=https://www.example.com
```

```yaml
# docker-compose.yml
services:
  app:
    image: node:${NODE_VERSION}-alpine  # .env の変数を使用
    env_file:
      - .env.development                # アプリ用環境変数
    environment:
      # env_file の値を上書き
      LOG_LEVEL: ${LOG_LEVEL:-info}     # デフォルト値付き

  db:
    image: postgres:${POSTGRES_VERSION}-alpine
```

### 3.3 環境変数の展開構文

```yaml
services:
  app:
    environment:
      # 基本形
      DB_HOST: ${DB_HOST}

      # デフォルト値 (未設定 or 空文字の場合)
      DB_PORT: ${DB_PORT:-5432}

      # デフォルト値 (未定義の場合のみ)
      DB_NAME: ${DB_NAME-myapp}

      # 未設定時にエラー
      DB_PASSWORD: ${DB_PASSWORD:?Database password must be set}

      # 設定済みの場合に代替値を使用
      DB_SSL: ${DB_HOST:+true}

      # ネストした変数展開（Compose V2.24+）
      FULL_DB_URL: "postgresql://${DB_USER:-postgres}:${DB_PASSWORD}@${DB_HOST:-db}:${DB_PORT:-5432}/${DB_NAME:-myapp}"
```

### 3.4 シークレット管理

Docker Compose のシークレット機能は、パスワードや API キーなどの機密情報を環境変数に直接書かずに管理する方法を提供する。

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password

  app:
    build: .
    secrets:
      - db_password
      - api_key
      - jwt_secret
    environment:
      # アプリケーション側でシークレットファイルを読む
      DB_PASSWORD_FILE: /run/secrets/db_password
      API_KEY_FILE: /run/secrets/api_key
      JWT_SECRET_FILE: /run/secrets/jwt_secret

secrets:
  db_password:
    file: ./secrets/db_password.txt     # ファイルから読み込み
  api_key:
    environment: API_KEY                 # 環境変数から (Compose V2.22+)
  jwt_secret:
    file: ./secrets/jwt_secret.txt
```

アプリケーション側でシークレットファイルを読む実装例（Node.js）:

```javascript
// config/secrets.js
const fs = require('fs');
const path = require('path');

function readSecret(name) {
  const filePath = process.env[`${name}_FILE`];
  if (filePath && fs.existsSync(filePath)) {
    return fs.readFileSync(filePath, 'utf8').trim();
  }
  // フォールバック: 環境変数から直接取得
  return process.env[name];
}

module.exports = {
  dbPassword: readSecret('DB_PASSWORD'),
  apiKey: readSecret('API_KEY'),
  jwtSecret: readSecret('JWT_SECRET'),
};
```

### 3.5 .env ファイルの .gitignore 設定

```gitignore
# .gitignore
.env
.env.local
.env.*.local
.env.production
.env.staging
secrets/

# テンプレートはコミットする
!.env.example
!.env.development.example
```

```bash
# .env.example (テンプレートとしてコミット)
COMPOSE_PROJECT_NAME=myapp
POSTGRES_VERSION=16
NODE_VERSION=20
DB_PASSWORD=<SET_YOUR_PASSWORD>
API_KEY=<SET_YOUR_API_KEY>
```

---

## 4. 複数 Compose ファイルのマージ

### 4.1 オーバーライドパターン

Docker Compose は複数の設定ファイルをマージして一つの構成を作成できる。これにより、ベース設定と環境固有の設定を分離し、DRY な構成管理を実現できる。

```yaml
# docker-compose.yml (ベース設定)
services:
  app:
    build: .
    environment:
      NODE_ENV: production

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  pgdata:
```

```yaml
# docker-compose.override.yml (開発用オーバーライド。自動マージ)
services:
  app:
    build:
      target: development
    environment:
      NODE_ENV: development
      DEBUG: "true"
    volumes:
      - .:/app
      - node_modules:/app/node_modules
    ports:
      - "3000:3000"
      - "9229:9229"   # デバッガポート

  db:
    ports:
      - "5432:5432"   # 開発時のみ外部公開
    environment:
      POSTGRES_PASSWORD: postgres

volumes:
  node_modules:
```

```yaml
# docker-compose.prod.yml (本番用)
services:
  app:
    restart: always
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 128M
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  db:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G

  redis:
    restart: always
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
```

```yaml
# docker-compose.ci.yml (CI 専用)
services:
  app:
    build:
      target: test
    environment:
      NODE_ENV: test
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp_test

  db:
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_test
    tmpfs:
      - /var/lib/postgresql/data    # CI ではメモリ上で高速化
```

### 4.2 マージのコマンド

```bash
# 開発 (compose.yml + compose.override.yml を自動マージ)
docker compose up -d

# 本番 (override を除外し、prod を適用)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# CI (override を除外し、ci を適用)
docker compose -f docker-compose.yml -f docker-compose.ci.yml up -d

# 設定のマージ結果を確認
docker compose -f docker-compose.yml -f docker-compose.prod.yml config

# 特定のサービスのみマージ結果を確認
docker compose -f docker-compose.yml -f docker-compose.prod.yml config --services

# マージ結果をファイルに出力
docker compose -f docker-compose.yml -f docker-compose.prod.yml config > docker-compose.resolved.yml
```

### 4.3 マージの規則詳細

| 設定項目 | マージ動作 |
|----------|-----------|
| `image`, `command`, `entrypoint` | 後のファイルで上書き |
| `environment` | マージ（キー単位で上書き） |
| `volumes` | マージ（追加される） |
| `ports` | マージ（追加される） |
| `networks` | マージ（追加される） |
| `labels` | マージ（キー単位で上書き） |
| `deploy` | ディープマージ |
| `build.args` | マージ（キー単位で上書き） |
| `healthcheck` | 後のファイルで完全上書き |

### 4.4 COMPOSE_FILE 環境変数による自動選択

```bash
# .env ファイルで読み込むファイルを指定
# 開発環境
COMPOSE_FILE=docker-compose.yml:docker-compose.override.yml

# ステージング環境
COMPOSE_FILE=docker-compose.yml:docker-compose.staging.yml

# 本番環境
COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml

# 区切り文字はデフォルトで「:」(Linux/macOS) または「;」(Windows)
```

---

## 5. リソース制限とロギング

### 5.1 リソース制限

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '0.5'        # CPU 0.5 コア
          memory: 256M        # メモリ 256MB
          pids: 100           # プロセス数上限
        reservations:
          cpus: '0.25'       # 最低保証 CPU
          memory: 128M        # 最低保証メモリ

    # OOM 時の動作
    oom_kill_disable: false
    oom_score_adj: 100         # OOM スコア調整 (-1000 to 1000)

    # ファイルディスクリプタ制限
    ulimits:
      nofile:
        soft: 65536
        hard: 65536
      nproc:
        soft: 4096
        hard: 4096

    # SHM サイズ制限 (共有メモリ)
    shm_size: '256m'

    # ストップシグナルとタイムアウト
    stop_signal: SIGTERM
    stop_grace_period: 30s
```

### 5.2 各サービスの推奨リソース設定

```yaml
services:
  # Node.js アプリ
  app:
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 128M

  # PostgreSQL
  db:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 256M
    shm_size: '256m'    # PostgreSQL は共有メモリを多用

  # Redis
  redis:
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 256M
        reservations:
          cpus: '0.1'
          memory: 64M

  # Elasticsearch
  elasticsearch:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    environment:
      ES_JAVA_OPTS: "-Xms512m -Xmx1g"   # JVM ヒープもメモリ制限に合わせる
```

### 5.3 ロギング設定

```yaml
services:
  app:
    logging:
      driver: json-file
      options:
        max-size: "10m"      # ログファイル最大サイズ
        max-file: "3"        # ローテーション数
        compress: "true"     # 圧縮
        labels: "service"
        tag: "{{.Name}}/{{.ID}}"  # ログタグのカスタマイズ

  # 全サービス共通のログ設定 (YAML アンカー)
  db:
    logging: &default-logging
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  redis:
    logging: *default-logging  # アンカーを参照
```

### 5.4 外部ロギングドライバーの設定

```yaml
services:
  # Fluentd ドライバー
  app:
    logging:
      driver: fluentd
      options:
        fluentd-address: localhost:24224
        tag: myapp.{{.Name}}
        fluentd-async: "true"
        fluentd-retry-wait: "1s"
        fluentd-max-retries: "10"

  # syslog ドライバー
  api:
    logging:
      driver: syslog
      options:
        syslog-address: "tcp://logserver:514"
        syslog-facility: "daemon"
        tag: "{{.Name}}"

  # ログを無効化 (出力が多すぎるサービス)
  noisy-service:
    logging:
      driver: none
```

---

## 6. YAML アンカーとエイリアス

### 6.1 基本的なアンカーとエイリアス

```yaml
# 共通設定をアンカーで定義
x-common-env: &common-env
  TZ: Asia/Tokyo
  LANG: ja_JP.UTF-8

x-healthcheck-defaults: &healthcheck-defaults
  interval: 10s
  timeout: 5s
  retries: 3
  start_period: 30s

x-logging: &default-logging
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"

services:
  app:
    environment:
      <<: *common-env          # マージ (アンカー展開)
      NODE_ENV: production
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD-SHELL", "curl -f http://localhost:3000/health"]
    logging: *default-logging

  worker:
    environment:
      <<: *common-env
      WORKER_TYPE: background
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD-SHELL", "curl -f http://localhost:3001/health"]
    logging: *default-logging
```

### 6.2 Extension Fields (x- プレフィックス) の高度な活用

Extension Fields は Compose が解釈しないカスタムフィールドで、アンカーの定義場所として使用する。サービス定義全体を共通化する場合に特に効果的である。

```yaml
# サービスのテンプレート
x-app-base: &app-base
  build:
    context: .
    dockerfile: Dockerfile
  restart: always
  networks:
    - app-net
  logging: &default-logging
    driver: json-file
    options:
      max-size: "10m"
      max-file: "3"
  deploy:
    resources:
      limits:
        cpus: '1.0'
        memory: 512M
      reservations:
        cpus: '0.25'
        memory: 128M
  environment: &common-env
    TZ: Asia/Tokyo
    LANG: ja_JP.UTF-8
    NODE_ENV: production
    DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db:5432/myapp
    REDIS_URL: redis://redis:6379

x-db-healthcheck: &db-healthcheck
  test: ["CMD-SHELL", "pg_isready -U postgres"]
  interval: 5s
  timeout: 5s
  retries: 5
  start_period: 30s

services:
  # テンプレートを継承してカスタマイズ
  web:
    <<: *app-base
    ports:
      - "3000:3000"
    command: node dist/web.js
    environment:
      <<: *common-env
      SERVER_TYPE: web
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/health"]
      interval: 10s
      timeout: 5s
      retries: 3

  api:
    <<: *app-base
    ports:
      - "8080:8080"
    command: node dist/api.js
    environment:
      <<: *common-env
      SERVER_TYPE: api

  worker:
    <<: *app-base
    command: node dist/worker.js
    environment:
      <<: *common-env
      SERVER_TYPE: worker
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G  # ワーカーはメモリを多く使う

  scheduler:
    <<: *app-base
    command: node dist/scheduler.js
    environment:
      <<: *common-env
      SERVER_TYPE: scheduler

  db:
    image: postgres:16-alpine
    restart: always
    healthcheck:
      <<: *db-healthcheck
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - app-net
    logging: *default-logging

networks:
  app-net:

volumes:
  pgdata:
```

### 6.3 条件分岐的な設定（アンカーとオーバーライドの組み合わせ）

```yaml
# docker-compose.yml
x-app-volumes: &app-volumes
  volumes:
    - app-data:/data

services:
  app:
    <<: *app-volumes
    image: myapp:latest
```

```yaml
# docker-compose.override.yml (開発環境で上書き)
services:
  app:
    volumes:
      - .:/app
      - app-data:/data    # 元の Volume も維持
```

---

## 7. ネットワーク分離の高度な設定

### 7.1 マルチネットワーク構成

```yaml
services:
  # フロントエンド (public + app-tier のみ)
  nginx:
    image: nginx:alpine
    networks:
      - public
      - app-tier
    ports:
      - "80:80"
      - "443:443"

  # アプリケーション (app-tier + data-tier)
  app:
    build: .
    networks:
      - app-tier
      - data-tier
      - cache-tier

  # データベース (data-tier のみ / 外部アクセス不可)
  db:
    image: postgres:16-alpine
    networks:
      - data-tier

  # Redis (cache-tier のみ)
  redis:
    image: redis:7-alpine
    networks:
      - cache-tier

networks:
  public:
    driver: bridge
  app-tier:
    driver: bridge
  data-tier:
    driver: bridge
    internal: true     # 外部アクセスを完全遮断
  cache-tier:
    driver: bridge
    internal: true
```

### 7.2 IP アドレスの固定

```yaml
services:
  app:
    networks:
      app-net:
        ipv4_address: 172.28.0.10

  db:
    networks:
      app-net:
        ipv4_address: 172.28.0.20

networks:
  app-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/24
          gateway: 172.28.0.1
```

---

## 8. 高度な設定比較

| 機能 | 基本設定 | 応用設定 |
|------|---------|---------|
| サービス起動 | `depends_on: [db]` | `depends_on: { db: { condition: service_healthy } }` |
| 環境変数 | `environment: {KEY: val}` | `env_file` + `secrets` + 優先順位管理 |
| ネットワーク | デフォルト | `internal: true` + 複数ネットワーク分離 |
| ログ | デフォルト (無制限) | `json-file` + `max-size` + `max-file` |
| リソース | 無制限 | `deploy.resources.limits` で CPU/メモリ制限 |
| プロファイル | 全サービス起動 | `profiles` で用途別グルーピング |
| 設定管理 | 単一ファイル | `override.yml` + `prod.yml` でレイヤー化 |
| ヘルスチェック | なし | サービスごとの専用チェック + カスタムスクリプト |
| シークレット | 環境変数に直接記載 | `secrets` + `*_FILE` パターン |
| YAML 再利用 | コピー&ペースト | `x-` Extension Fields + アンカー |

---

## 9. Compose の便利なコマンド集

### 9.1 日常操作

```bash
# サービスの状態確認
docker compose ps
docker compose ps -a    # 停止中のコンテナも表示

# ログの確認
docker compose logs -f              # 全サービスのログをフォロー
docker compose logs -f app worker   # 特定サービスのみ
docker compose logs --tail=50 app   # 最新50行
docker compose logs --since=1h      # 直近1時間のログ

# サービスの再起動
docker compose restart app          # app のみ再起動
docker compose up -d --force-recreate app   # 強制再作成

# 設定の確認
docker compose config               # マージ結果を表示
docker compose config --services    # サービス一覧
docker compose config --volumes     # ボリューム一覧

# イメージのビルド
docker compose build                # 全サービスビルド
docker compose build --no-cache     # キャッシュなしでビルド
docker compose build --parallel     # 並列ビルド
docker compose build app worker     # 特定サービスのみ

# コンテナ内でコマンド実行
docker compose exec app sh                  # シェルに入る
docker compose exec -T app npm run migrate  # TTY なし（スクリプト向け）
docker compose run --rm app npm test         # ワンショット実行
```

### 9.2 クリーンアップ

```bash
# サービス停止
docker compose stop                 # 停止のみ
docker compose down                 # 停止 + コンテナ削除
docker compose down -v              # 停止 + コンテナ + ボリューム削除
docker compose down --remove-orphans # 孤立コンテナも削除
docker compose down --rmi local     # ローカルイメージも削除
docker compose down -v --rmi all    # 全て削除

# 特定サービスのみ停止
docker compose stop app
docker compose rm -f app
```

---

## アンチパターン

### アンチパターン 1: ヘルスチェックなしの depends_on

```yaml
# NG: condition 未指定 → コンテナ起動 = サービス利用可能 と誤解
services:
  app:
    depends_on:
      - db         # DB コンテナが起動した瞬間に app も起動する
                   # → DB がまだ接続受付前でアプリがクラッシュ

# OK: healthcheck + condition で確実に待機
services:
  app:
    depends_on:
      db:
        condition: service_healthy
  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
```

**問題点**: PostgreSQL コンテナが起動してから実際に接続を受け付けるまでに数秒〜十数秒かかる。ヘルスチェックなしでは、アプリが接続エラーでクラッシュし、手動で再起動が必要になる。

### アンチパターン 2: ログのローテーション未設定

```yaml
# NG: ログ設定なし → ディスクが枯渇する
services:
  app:
    image: myapp:latest
    # logging 未設定 → json-file ドライバ、サイズ無制限

# OK: ログサイズとローテーションを設定
services:
  app:
    image: myapp:latest
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

**問題点**: Docker のデフォルトログドライバ (json-file) はサイズ無制限でログを蓄積する。長時間稼働するサービスではログファイルがディスクを圧迫し、最終的にホストマシンのディスクが枯渇してシステム全体が停止する。

### アンチパターン 3: シークレットを環境変数に直接記載

```yaml
# NG: パスワードをファイルに直接記載
services:
  db:
    environment:
      POSTGRES_PASSWORD: my_super_secret_password  # Git にコミットされる

# OK: .env ファイルまたは secrets を使用
services:
  db:
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}  # .env から取得
    # または secrets を使用
    secrets:
      - db_password
```

**問題点**: `docker-compose.yml` にハードコードされたパスワードは Git リポジトリにコミットされ、漏洩リスクが極めて高い。`.env` ファイルを `.gitignore` に追加するか、Docker の `secrets` 機能を使用する。

### アンチパターン 4: リソース制限なしの本番運用

```yaml
# NG: リソース制限なし → 1つのサービスがホストのリソースを食い尽くす
services:
  app:
    image: myapp:latest
    # deploy.resources 未設定

# OK: 適切なリソース制限を設定
services:
  app:
    image: myapp:latest
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 128M
```

**問題点**: メモリリークやCPU 暴走が発生した場合、制限がないとホスト全体のリソースが枯渇し、他のサービスや SSH 接続すらも影響を受ける。特に本番環境では必ず制限を設定する。

---

## FAQ

### Q1: YAML アンカーと Extension Fields (x- プレフィックス) の違いは何ですか？

**A**: YAML アンカー (`&` / `*`) は YAML 標準の参照機構で、同じ値を複数箇所で再利用する。Extension Fields (`x-` プレフィックス) は Compose 仕様の機能で、Compose が無視するカスタムフィールドを定義できる。両者を組み合わせ、`x-common: &common` でトップレベルに共通設定を定義し、各サービスで `<<: *common` で展開するのが一般的なパターン。

### Q2: プロファイルと複数 Compose ファイルのどちらを使うべきですか？

**A**: 同じ `docker-compose.yml` 内で用途別にサービスをグルーピングしたい場合はプロファイルが適切（例: debug ツール、監視ツール）。環境全体の設定を切り替えたい場合（開発 vs 本番、ポート公開の有無、リソース制限など）は複数ファイルのオーバーライドが適切。両方を併用することもできる。

### Q3: Compose で使える Interpolation (変数展開) の構文は？

**A**: `${VARIABLE}` が基本形。デフォルト値は `${VARIABLE:-default}` (未設定時) と `${VARIABLE-default}` (未定義時のみ)。エラーにする場合は `${VARIABLE:?error message}`。これらは `.env` ファイルまたはシェルの環境変数から値を取得する。Compose ファイル内の `environment:` セクションの値は展開されるが、コンテナ内での展開とは異なる点に注意。

### Q4: depends_on の restart: true オプションは何ですか？

**A**: Compose V2.20+ で追加された機能で、依存先のサービスが再起動された場合に、依存元のサービスも自動的に再起動させる。例えば、DB コンテナが再起動された場合に、自動的にアプリコンテナも再起動させたい場合に使用する。

```yaml
services:
  app:
    depends_on:
      db:
        condition: service_healthy
        restart: true    # db が再起動されたら app も再起動
```

### Q5: healthcheck の start_period はどう設定すべきですか？

**A**: `start_period` はサービスの初期化に必要な時間を見積もって設定する。この期間中のヘルスチェック失敗はリトライ回数にカウントされない。PostgreSQL なら 30 秒、Elasticsearch なら 60 秒程度が目安。テスト環境で `docker compose up` してから実際にサービスが応答するまでの時間を計測し、その値の 1.5〜2 倍を設定するのが安全である。

### Q6: Compose V2 と V1 (docker-compose コマンド) の違いは？

**A**: Compose V2 は `docker compose`（ハイフンなし）で呼び出す Go 言語で実装されたプラグインである。V1 は `docker-compose`（ハイフンあり）で呼び出す Python 実装で、2023 年に EOL となった。V2 では `version:` フィールドが不要になり、`profiles`、`watch`、`include` など多くの新機能が追加されている。新規プロジェクトでは必ず V2 を使用すべきである。

---

## まとめ

| 項目 | 要点 |
|------|------|
| プロファイル | `profiles:` でサービスをグルーピングし、`--profile` で選択起動 |
| depends_on | `condition: service_healthy` で確実な起動順序を保証 |
| healthcheck | DB/Redis/HTTP それぞれに適切なチェックコマンドを設定 |
| 環境変数 | `.env` (Compose変数) + `env_file` (アプリ変数) + `secrets` の使い分け |
| ファイルマージ | `override.yml` (開発) + `prod.yml` (本番) でレイヤー化 |
| YAML アンカー | `x-` Extension Fields + アンカーで設定の DRY 化 |
| リソース制限 | `deploy.resources.limits` で CPU/メモリを制限 |
| ログ管理 | `max-size` + `max-file` でディスク枯渇を防止 |
| ネットワーク分離 | `internal: true` + 複数ネットワークでセキュリティ強化 |
| シークレット管理 | `secrets` + `*_FILE` パターンで安全に機密情報を管理 |

## 次に読むべきガイド

- [Compose 開発ワークフロー](./02-development-workflow.md) -- ホットリロード、デバッグ、CI 統合
- [Docker Compose 基礎](./00-compose-basics.md) -- 基本構文の復習
- [コンテナセキュリティ](../06-security/00-container-security.md) -- セキュリティのベストプラクティス

## 参考文献

1. **Compose Specification - Profiles** -- https://docs.docker.com/compose/profiles/ -- プロファイル機能の公式ドキュメント
2. **Compose Specification - Healthcheck** -- https://docs.docker.com/compose/compose-file/05-services/#healthcheck -- ヘルスチェック設定の詳細
3. **Environment variables in Compose** -- https://docs.docker.com/compose/environment-variables/ -- 環境変数の優先順位と管理方法
4. **Compose file merge** -- https://docs.docker.com/compose/multiple-compose-files/ -- 複数ファイルのマージ規則
5. **Compose Specification - Extension Fields** -- https://docs.docker.com/compose/compose-file/11-extension/ -- Extension Fields の仕様
6. **Compose Specification - Secrets** -- https://docs.docker.com/compose/use-secrets/ -- シークレット管理の公式ドキュメント
7. **Docker Compose V2 Release Notes** -- https://docs.docker.com/compose/release-notes/ -- V2 の新機能と変更点
