# ローカルサービスの Docker 化

> PostgreSQL、MySQL、Redis、MailHog、MinIO などの開発用サービスを Docker で統一管理し、チーム全員が同一のローカル環境で開発できるインフラ構成を学ぶ。

## この章で学ぶこと

1. **データベース (PostgreSQL / MySQL) の Docker 化** -- 初期スキーマ・シードデータの自動投入を含む、開発用 DB コンテナの構築パターンを習得する
2. **キャッシュ・メール・ストレージの Docker 化** -- Redis、MailHog、MinIO を組み合わせた開発インフラの構成を学ぶ
3. **Docker Compose による統合管理とデータ永続化** -- 複数サービスの依存関係管理、Volume 設計、ヘルスチェックを実践する

---

## 1. 全体アーキテクチャ

```
+------------------------------------------------------------------+
|              ローカル開発サービス構成図                              |
+------------------------------------------------------------------+
|                                                                  |
|  [アプリケーション]                                                |
|       |         |          |          |           |              |
|       v         v          v          v           v              |
|  +--------+ +--------+ +-------+ +--------+ +---------+         |
|  |Postgres| | MySQL  | | Redis | |MailHog | | MinIO   |         |
|  |  :5432 | |  :3306 | | :6379 | | :1025  | |  :9000  |         |
|  |        | |        | |       | | :8025  | |  :9001  |         |
|  +--------+ +--------+ +-------+ +--------+ +---------+         |
|       |         |          |          |           |              |
|       v         v          v          v           v              |
|  [pgdata]  [mysqldata] (ephemeral) (ephemeral) [miniodata]      |
|  Named Vol  Named Vol                          Named Vol        |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.1 サービス選択の判断基準

ローカル開発でどのサービスを Docker 化するかは、プロジェクトの要件に応じて判断する。以下のフローチャートを参考にする。

```
+------------------------------------------------------------------+
|        ローカルサービス選択フロー                                    |
+------------------------------------------------------------------+
|                                                                  |
|  RDB が必要か？                                                   |
|    |                                                             |
|    +--[Yes]--> JSON/配列型が多い？ → PostgreSQL                   |
|    |           シンプルなCRUD？ → MySQL でも可                    |
|    |           既存が MySQL？ → MySQL を継続                      |
|    |                                                             |
|    +--[No]---> NoSQL が必要か？                                   |
|                  |                                               |
|                  +--[Yes]--> MongoDB / DynamoDB Local             |
|                  +--[No]---> SQLite で十分な場合もある             |
|                                                                  |
|  キャッシュが必要か？ → Redis                                      |
|  セッションストアが必要か？ → Redis                                |
|  メール送信テストが必要か？ → Mailpit (MailHog 後継)               |
|  ファイルアップロードが必要か？ → MinIO (S3 互換)                   |
|  全文検索が必要か？ → Meilisearch or Elasticsearch                 |
|  メッセージキューが必要か？ → RabbitMQ or Redis Streams            |
|  認証テストが必要か？ → Keycloak or mock-oauth2-server             |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.2 ポート管理戦略

複数プロジェクトや複数のデータベースを同時に使う場合、ポート競合を避ける戦略が必要である。

| 戦略 | 説明 | 向いているケース |
|------|------|----------------|
| プロジェクトごとにオフセット | Project A: 5432, Project B: 5433 | 2-3 プロジェクト同時開発 |
| `.env` でポートを変数化 | `${DB_PORT:-5432}` | チームで柔軟に運用 |
| Docker ネットワーク分離 | ポート公開せず、コンテナ間通信のみ | Dev Container 環境 |
| profiles で排他管理 | `docker compose --profile projectA up` | 多数のプロジェクト |

```yaml
# .env でポートを変数化する例
# .env
POSTGRES_PORT=5432
REDIS_PORT=6379
MAIL_SMTP_PORT=1025
MAIL_UI_PORT=8025
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
```

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    ports:
      - "${POSTGRES_PORT:-5432}:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "${REDIS_PORT:-6379}:6379"
```

---

## 2. PostgreSQL の Docker 化

### 2.1 基本設定

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    container_name: myapp-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_development
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
      TZ: Asia/Tokyo
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/postgres/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s

volumes:
  pgdata:
    driver: local
```

### 2.2 初期化スクリプト

`/docker-entrypoint-initdb.d/` に配置したファイルは、コンテナ初回起動時にアルファベット順で実行される。`.sql`, `.sql.gz`, `.sh` ファイルがサポートされている。

```sql
-- docker/postgres/init/01-create-databases.sql
-- 開発用・テスト用の DB を自動作成

CREATE DATABASE myapp_test;

-- テスト用ユーザー
CREATE USER test_user WITH PASSWORD 'test_password';
GRANT ALL PRIVILEGES ON DATABASE myapp_test TO test_user;

-- 拡張機能のインストール
\c myapp_development
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "citext";
CREATE EXTENSION IF NOT EXISTS "hstore";

\c myapp_test
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
```

```bash
#!/bin/bash
# docker/postgres/init/02-seed-data.sh
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname myapp_development <<-EOSQL
  INSERT INTO users (name, email) VALUES
    ('開発太郎', 'dev@example.com'),
    ('テスト花子', 'test@example.com')
  ON CONFLICT DO NOTHING;
EOSQL
```

```sql
-- docker/postgres/init/03-create-schemas.sql
-- マルチテナント用スキーマ分離パターン

\c myapp_development

-- テナントごとのスキーマ
CREATE SCHEMA IF NOT EXISTS tenant_demo;
CREATE SCHEMA IF NOT EXISTS tenant_test;

-- 共有テーブル（public スキーマ）
CREATE TABLE IF NOT EXISTS public.tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    schema_name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO public.tenants (name, schema_name) VALUES
    ('Demo Company', 'tenant_demo'),
    ('Test Company', 'tenant_test')
ON CONFLICT DO NOTHING;
```

### 2.3 PostgreSQL 設定のカスタマイズ

```conf
# docker/postgres/postgresql.conf (開発用に最適化)
# パフォーマンス (開発マシン向け)
shared_buffers = 256MB
work_mem = 16MB
maintenance_work_mem = 128MB
effective_cache_size = 512MB

# WAL 設定 (開発用)
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 1GB

# ログ (デバッグしやすく)
log_statement = 'all'
log_duration = on
log_min_duration_statement = 100
log_line_prefix = '%t [%p] %u@%d '
log_lock_waits = on
log_temp_files = 0

# 開発用設定 (本番では使わない)
fsync = off
synchronous_commit = off
full_page_writes = off

# 接続設定
max_connections = 100
```

カスタム設定ファイルを適用するには、Compose ファイルで以下のように指定する。

```yaml
services:
  postgres:
    image: postgres:16-alpine
    command: >
      postgres
        -c config_file=/etc/postgresql/postgresql.conf
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./docker/postgres/init:/docker-entrypoint-initdb.d
```

### 2.4 PostgreSQL のバックアップとリストア

開発中のデータベース状態を保存・復元する方法は以下の通りである。

```bash
# バックアップ（ダンプ）
docker compose exec postgres pg_dump -U postgres -d myapp_development \
  --format=custom --file=/tmp/backup.dump

# ホストにコピー
docker compose cp postgres:/tmp/backup.dump ./backups/

# リストア
docker compose cp ./backups/backup.dump postgres:/tmp/
docker compose exec postgres pg_restore -U postgres -d myapp_development \
  --clean --if-exists /tmp/backup.dump

# テキスト形式でのダンプ（Git 管理しやすい）
docker compose exec postgres pg_dump -U postgres -d myapp_development \
  --schema-only --no-owner --no-privileges > ./docker/postgres/schema.sql

# データのみダンプ
docker compose exec postgres pg_dump -U postgres -d myapp_development \
  --data-only --inserts > ./docker/postgres/seed-data.sql
```

### 2.5 PostgreSQL の監視とデバッグ

```bash
# 実行中のクエリを確認
docker compose exec postgres psql -U postgres -d myapp_development -c \
  "SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
   FROM pg_stat_activity
   WHERE state != 'idle' AND query NOT ILIKE '%pg_stat_activity%'
   ORDER BY duration DESC;"

# テーブルサイズの確認
docker compose exec postgres psql -U postgres -d myapp_development -c \
  "SELECT schemaname, tablename,
          pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size
   FROM pg_tables
   WHERE schemaname = 'public'
   ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;"

# インデックスの使用状況
docker compose exec postgres psql -U postgres -d myapp_development -c \
  "SELECT relname, indexrelname, idx_scan, idx_tup_read, idx_tup_fetch
   FROM pg_stat_user_indexes
   ORDER BY idx_scan ASC;"

# コネクション数の確認
docker compose exec postgres psql -U postgres -c \
  "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

### 2.6 pgAdmin の追加（GUI 管理ツール）

```yaml
# docker-compose.yml に追加
services:
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: myapp-pgadmin
    restart: unless-stopped
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
      PGADMIN_LISTEN_PORT: 5050
    ports:
      - "5050:5050"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
      - ./docker/pgadmin/servers.json:/pgadmin4/servers.json:ro
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  pgadmin_data:
```

```json
// docker/pgadmin/servers.json (自動接続設定)
{
  "Servers": {
    "1": {
      "Name": "Local Development",
      "Group": "Development",
      "Host": "postgres",
      "Port": 5432,
      "MaintenanceDB": "postgres",
      "Username": "postgres",
      "SSLMode": "prefer",
      "PassFile": "/tmp/pgpassfile"
    }
  }
}
```

---

## 3. MySQL の Docker 化

### 3.1 基本設定

```yaml
# docker-compose.yml (MySQL 版)
services:
  mysql:
    image: mysql:8.0
    container_name: myapp-mysql
    restart: unless-stopped
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: myapp_development
      MYSQL_USER: developer
      MYSQL_PASSWORD: developer
      TZ: Asia/Tokyo
    ports:
      - "3306:3306"
    volumes:
      - mysqldata:/var/lib/mysql
      - ./docker/mysql/init:/docker-entrypoint-initdb.d
      - ./docker/mysql/my.cnf:/etc/mysql/conf.d/custom.cnf
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 5s
      timeout: 5s
      retries: 5
    command: >
      --default-authentication-plugin=mysql_native_password
      --character-set-server=utf8mb4
      --collation-server=utf8mb4_unicode_ci

volumes:
  mysqldata:
    driver: local
```

### 3.2 MySQL カスタム設定

```ini
# docker/mysql/my.cnf
[mysqld]
# 文字コード
character-set-server = utf8mb4
collation-server = utf8mb4_unicode_ci

# パフォーマンス (開発用)
innodb_buffer_pool_size = 256M
innodb_log_file_size = 64M
innodb_flush_log_at_trx_commit = 0
sync_binlog = 0
innodb_flush_method = O_DIRECT

# ログ
general_log = 1
general_log_file = /var/log/mysql/general.log
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 1

# タイムゾーン
default-time-zone = '+09:00'

# 接続設定
max_connections = 100
wait_timeout = 28800
interactive_timeout = 28800

[client]
default-character-set = utf8mb4
```

### 3.3 MySQL の初期化スクリプト

```sql
-- docker/mysql/init/01-create-databases.sql
CREATE DATABASE IF NOT EXISTS myapp_test
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

-- テスト用ユーザーに権限付与
GRANT ALL PRIVILEGES ON myapp_test.* TO 'developer'@'%';
FLUSH PRIVILEGES;
```

```sql
-- docker/mysql/init/02-create-tables.sql
USE myapp_development;

CREATE TABLE IF NOT EXISTS users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_email (email),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

### 3.4 MySQL のバックアップとリストア

```bash
# バックアップ
docker compose exec mysql mysqldump -u root -proot \
  --single-transaction --routines --triggers \
  myapp_development > ./backups/mysql_backup.sql

# リストア
docker compose exec -T mysql mysql -u root -proot \
  myapp_development < ./backups/mysql_backup.sql

# 特定のテーブルのみダンプ
docker compose exec mysql mysqldump -u root -proot \
  myapp_development users orders > ./backups/partial_backup.sql

# スキーマのみダンプ
docker compose exec mysql mysqldump -u root -proot \
  --no-data myapp_development > ./docker/mysql/schema.sql
```

### 3.5 MySQL 8.4 への移行

MySQL 8.4 LTS では `mysql_native_password` が非推奨となり、`caching_sha2_password` がデフォルトになっている。

```yaml
# MySQL 8.4 LTS 対応
services:
  mysql:
    image: mysql:8.4
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: myapp_development
    # mysql_native_password は非推奨
    # アプリケーションの MySQL クライアントが caching_sha2_password に対応しているか確認
    command: >
      --character-set-server=utf8mb4
      --collation-server=utf8mb4_unicode_ci
```

---

## 4. Redis の Docker 化

### 4.1 基本設定

```yaml
# docker-compose.yml に追加
services:
  redis:
    image: redis:7-alpine
    container_name: myapp-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - ./docker/redis/redis.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

### 4.2 Redis 設定

```conf
# docker/redis/redis.conf (開発用)
# メモリ制限
maxmemory 128mb
maxmemory-policy allkeys-lru

# 永続化 (開発では不要なら off)
save ""
appendonly no

# ログ
loglevel verbose

# パスワード (開発環境)
# requirepass dev_redis_password

# キーの有効期限通知（Pub/Sub で期限切れイベントを受け取る）
notify-keyspace-events Ex
```

### 4.3 Redis の用途別設定パターン

#### セッションストアとして使う場合

```conf
# docker/redis/redis-session.conf
# セッションデータは永続化が必要
save 60 1000
save 300 100
appendonly yes
appendfsync everysec

maxmemory 256mb
maxmemory-policy volatile-lru
```

#### キャッシュとして使う場合

```conf
# docker/redis/redis-cache.conf
# キャッシュは永続化不要
save ""
appendonly no

maxmemory 512mb
maxmemory-policy allkeys-lfu
```

#### ジョブキュー（BullMQ / Sidekiq）として使う場合

```conf
# docker/redis/redis-queue.conf
# ジョブデータの永続化が必要
save 60 1
appendonly yes
appendfsync everysec

maxmemory 256mb
maxmemory-policy noeviction  # キューデータは削除しない
```

### 4.4 Redis の監視とデバッグ

```bash
# Redis の情報を表示
docker compose exec redis redis-cli INFO

# メモリ使用状況
docker compose exec redis redis-cli INFO memory

# リアルタイムのコマンド監視
docker compose exec redis redis-cli MONITOR

# キーの一覧（開発環境のみ）
docker compose exec redis redis-cli KEYS "*"

# 特定のキーの内容を確認
docker compose exec redis redis-cli GET "session:abc123"
docker compose exec redis redis-cli HGETALL "user:1"

# キーの有効期限を確認
docker compose exec redis redis-cli TTL "cache:products"

# 全キーの削除（開発環境のみ）
docker compose exec redis redis-cli FLUSHALL

# スロークエリの確認
docker compose exec redis redis-cli SLOWLOG GET 10
```

### 4.5 Redis のアプリケーション接続

```typescript
// config/redis.ts
import { Redis } from 'ioredis';

function createRedisClient(): Redis {
  const redisUrl = process.env.REDIS_URL || 'redis://localhost:6379';

  const client = new Redis(redisUrl, {
    maxRetriesPerRequest: 3,
    retryStrategy(times) {
      const delay = Math.min(times * 50, 2000);
      return delay;
    },
    // 接続が切れた場合の自動再接続
    reconnectOnError(err) {
      const targetError = 'READONLY';
      if (err.message.includes(targetError)) {
        return true;
      }
      return false;
    },
  });

  client.on('error', (err) => {
    console.error('Redis connection error:', err);
  });

  client.on('connect', () => {
    console.log('Redis connected');
  });

  return client;
}

export const redis = createRedisClient();
```

### 4.6 RedisInsight（GUI 管理ツール）

```yaml
# docker-compose.yml に追加
services:
  redis-insight:
    image: redis/redisinsight:latest
    container_name: myapp-redis-insight
    restart: unless-stopped
    ports:
      - "5540:5540"
    volumes:
      - redisinsight_data:/data
    depends_on:
      redis:
        condition: service_healthy

volumes:
  redisinsight_data:
```

---

## 5. MailHog / Mailpit (メールテスト)

### 5.1 Mailpit（推奨 -- MailHog の後継）

MailHog はメンテナンスが停止しているため、後継の Mailpit を推奨する。API 互換性があり、移行は容易である。

```yaml
# docker-compose.yml に追加
services:
  mailpit:
    image: axllent/mailpit:latest
    container_name: myapp-mailpit
    restart: unless-stopped
    ports:
      - "1025:1025"    # SMTP サーバー
      - "8025:8025"    # Web UI
    environment:
      MP_SMTP_AUTH_ACCEPT_ANY: 1
      MP_SMTP_AUTH_ALLOW_INSECURE: 1
      MP_MAX_MESSAGES: 5000
      MP_DATABASE: /data/mailpit.db
      MP_SMTP_RELAY_CONFIG: ""
    volumes:
      - mailpit_data:/data

volumes:
  mailpit_data:
```

### 5.2 MailHog（レガシー）

既存プロジェクトで MailHog を使用している場合の設定は以下の通りである。

```yaml
# docker-compose.yml に追加
services:
  mailhog:
    image: mailhog/mailhog:latest
    container_name: myapp-mailhog
    restart: unless-stopped
    ports:
      - "1025:1025"    # SMTP サーバー
      - "8025:8025"    # Web UI
    environment:
      MH_STORAGE: memory
      MH_SMTP_BIND_ADDR: 0.0.0.0:1025
      MH_UI_BIND_ADDR: 0.0.0.0:8025
```

### 5.3 アプリからの接続設定

```typescript
// config/mail.ts
import nodemailer from 'nodemailer';

function createMailTransport() {
  if (process.env.NODE_ENV === 'development' || process.env.NODE_ENV === 'test') {
    // Mailpit / MailHog （SMTP 互換）
    return nodemailer.createTransport({
      host: process.env.SMTP_HOST || 'localhost',
      port: parseInt(process.env.SMTP_PORT || '1025'),
      secure: false,
      // Mailpit / MailHog は認証不要
      tls: {
        rejectUnauthorized: false,
      },
    });
  }

  // 本番環境
  return nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: parseInt(process.env.SMTP_PORT || '587'),
    secure: process.env.SMTP_SECURE === 'true',
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS,
    },
  });
}

export const mailTransport = createMailTransport();
```

```python
# config/mail.py (Python / Django)
import os

if os.getenv('ENVIRONMENT', 'development') == 'development':
    EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
    EMAIL_HOST = os.getenv('SMTP_HOST', 'localhost')
    EMAIL_PORT = int(os.getenv('SMTP_PORT', '1025'))
    EMAIL_USE_TLS = False
    EMAIL_HOST_USER = ''
    EMAIL_HOST_PASSWORD = ''
else:
    EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
    EMAIL_HOST = os.getenv('SMTP_HOST')
    EMAIL_PORT = int(os.getenv('SMTP_PORT', '587'))
    EMAIL_USE_TLS = True
    EMAIL_HOST_USER = os.getenv('SMTP_USER')
    EMAIL_HOST_PASSWORD = os.getenv('SMTP_PASS')
```

### 5.4 Mailpit API の活用（E2E テスト）

Mailpit の API を使って、E2E テストでメール送信を検証できる。

```typescript
// tests/helpers/mail.ts
interface MailpitMessage {
  ID: string;
  From: { Address: string; Name: string };
  To: { Address: string; Name: string }[];
  Subject: string;
  Snippet: string;
  Created: string;
}

interface MailpitSearchResult {
  total: number;
  messages: MailpitMessage[];
}

export async function waitForEmail(
  to: string,
  subject?: string,
  timeout = 10000
): Promise<MailpitMessage> {
  const startTime = Date.now();
  const mailpitUrl = process.env.MAILPIT_URL || 'http://localhost:8025';

  while (Date.now() - startTime < timeout) {
    const query = subject
      ? `to:${to} subject:"${subject}"`
      : `to:${to}`;

    const res = await fetch(
      `${mailpitUrl}/api/v1/search?query=${encodeURIComponent(query)}`
    );
    const data: MailpitSearchResult = await res.json();

    if (data.total > 0) {
      return data.messages[0];
    }

    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  throw new Error(`Email not received within ${timeout}ms`);
}

export async function getEmailHtml(messageId: string): Promise<string> {
  const mailpitUrl = process.env.MAILPIT_URL || 'http://localhost:8025';
  const res = await fetch(`${mailpitUrl}/api/v1/message/${messageId}`);
  const data = await res.json();
  return data.HTML;
}

export async function deleteAllEmails(): Promise<void> {
  const mailpitUrl = process.env.MAILPIT_URL || 'http://localhost:8025';
  await fetch(`${mailpitUrl}/api/v1/messages`, { method: 'DELETE' });
}
```

```typescript
// tests/e2e/password-reset.test.ts
import { test, expect } from '@playwright/test';
import { waitForEmail, getEmailHtml, deleteAllEmails } from '../helpers/mail';

test.beforeEach(async () => {
  await deleteAllEmails();
});

test('パスワードリセットメールが送信される', async ({ page }) => {
  // パスワードリセットを要求
  await page.goto('/forgot-password');
  await page.fill('[name="email"]', 'user@example.com');
  await page.click('button[type="submit"]');

  // メールが届くまで待機
  const email = await waitForEmail('user@example.com', 'パスワードリセット');

  expect(email.Subject).toContain('パスワードリセット');

  // メール本文からリセットリンクを取得
  const html = await getEmailHtml(email.ID);
  const resetLink = html.match(/href="([^"]*reset-password[^"]*)"/)?.[1];

  expect(resetLink).toBeTruthy();

  // リセットリンクにアクセス
  await page.goto(resetLink!);
  await expect(page.locator('h1')).toContainText('新しいパスワード');
});
```

---

## 6. MinIO (S3 互換ストレージ)

### 6.1 設定

```yaml
# docker-compose.yml に追加
services:
  minio:
    image: minio/minio:latest
    container_name: myapp-minio
    restart: unless-stopped
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"    # API
      - "9001:9001"    # Console
    volumes:
      - miniodata:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 5

  # 初期バケット作成
  minio-init:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 minioadmin minioadmin;
      mc mb local/uploads --ignore-existing;
      mc mb local/avatars --ignore-existing;
      mc mb local/documents --ignore-existing;
      mc anonymous set download local/avatars;
      mc ilm rule add local/uploads --expire-days 30;
      echo 'MinIO buckets initialized';
      "

volumes:
  miniodata:
    driver: local
```

### 6.2 AWS SDK での接続

```typescript
// config/storage.ts
import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

function createStorageClient(): S3Client {
  if (process.env.NODE_ENV === 'development') {
    return new S3Client({
      endpoint: process.env.S3_ENDPOINT || 'http://localhost:9000',
      region: 'us-east-1',
      credentials: {
        accessKeyId: process.env.S3_ACCESS_KEY || 'minioadmin',
        secretAccessKey: process.env.S3_SECRET_KEY || 'minioadmin',
      },
      forcePathStyle: true, // MinIO ではパススタイル必須
    });
  }

  // 本番は通常の S3
  return new S3Client({ region: process.env.AWS_REGION || 'ap-northeast-1' });
}

export const storageClient = createStorageClient();

// ファイルアップロード
export async function uploadFile(
  bucket: string,
  key: string,
  body: Buffer | ReadableStream,
  contentType: string
): Promise<string> {
  await storageClient.send(
    new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: body,
      ContentType: contentType,
    })
  );

  if (process.env.NODE_ENV === 'development') {
    return `http://localhost:9000/${bucket}/${key}`;
  }
  return `https://${bucket}.s3.amazonaws.com/${key}`;
}

// 署名付き URL の生成
export async function getPresignedUrl(
  bucket: string,
  key: string,
  expiresIn = 3600
): Promise<string> {
  const command = new GetObjectCommand({ Bucket: bucket, Key: key });
  return getSignedUrl(storageClient, command, { expiresIn });
}

// ファイル削除
export async function deleteFile(bucket: string, key: string): Promise<void> {
  await storageClient.send(
    new DeleteObjectCommand({ Bucket: bucket, Key: key })
  );
}
```

### 6.3 MinIO 管理コマンド

```bash
# MinIO Client (mc) の設定
docker compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin

# バケット一覧
docker compose exec minio mc ls local

# バケット内のオブジェクト一覧
docker compose exec minio mc ls local/uploads --recursive

# ファイルのアップロード（ホストから）
docker compose exec minio mc cp /data/test.jpg local/uploads/test.jpg

# バケットのポリシー確認
docker compose exec minio mc anonymous get local/avatars

# バケットの統計情報
docker compose exec minio mc stat local/uploads

# 全オブジェクトの削除
docker compose exec minio mc rm --recursive --force local/uploads
```

---

## 7. その他のサービス

### 7.1 Elasticsearch / OpenSearch

```yaml
services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    container_name: myapp-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - esdata:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Kibana (管理UI)
  kibana:
    image: docker.elastic.co/kibana/kibana:8.12.0
    container_name: myapp-kibana
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: '["http://elasticsearch:9200"]'
    depends_on:
      elasticsearch:
        condition: service_healthy

volumes:
  esdata:
```

### 7.2 Meilisearch（軽量全文検索）

Elasticsearch が重い場合の代替として Meilisearch が適している。

```yaml
services:
  meilisearch:
    image: getmeili/meilisearch:v1.6
    container_name: myapp-meilisearch
    restart: unless-stopped
    environment:
      MEILI_ENV: development
      MEILI_MASTER_KEY: dev_master_key
      MEILI_NO_ANALYTICS: true
    ports:
      - "7700:7700"
    volumes:
      - meilidata:/meili_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7700/health"]
      interval: 5s
      timeout: 3s
      retries: 5

volumes:
  meilidata:
```

### 7.3 RabbitMQ（メッセージキュー）

```yaml
services:
  rabbitmq:
    image: rabbitmq:3.13-management-alpine
    container_name: myapp-rabbitmq
    restart: unless-stopped
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest
    ports:
      - "5672:5672"    # AMQP
      - "15672:15672"  # Management UI
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "check_port_connectivity"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  rabbitmq_data:
```

### 7.4 MongoDB

```yaml
services:
  mongodb:
    image: mongo:7.0
    container_name: myapp-mongodb
    restart: unless-stopped
    environment:
      MONGO_INITDB_ROOT_USERNAME: root
      MONGO_INITDB_ROOT_PASSWORD: root
      MONGO_INITDB_DATABASE: myapp_development
    ports:
      - "27017:27017"
    volumes:
      - mongodata:/data/db
      - ./docker/mongo/init:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 5s
      timeout: 3s
      retries: 5

  # Mongo Express (管理UI)
  mongo-express:
    image: mongo-express:latest
    container_name: myapp-mongo-express
    restart: unless-stopped
    ports:
      - "8081:8081"
    environment:
      ME_CONFIG_MONGODB_ADMINUSERNAME: root
      ME_CONFIG_MONGODB_ADMINPASSWORD: root
      ME_CONFIG_MONGODB_URL: mongodb://root:root@mongodb:27017/
      ME_CONFIG_BASICAUTH_USERNAME: admin
      ME_CONFIG_BASICAUTH_PASSWORD: admin
    depends_on:
      mongodb:
        condition: service_healthy

volumes:
  mongodata:
```

### 7.5 Keycloak（認証サーバー）

OAuth2 / OpenID Connect のテスト用として Keycloak を使用する。

```yaml
services:
  keycloak:
    image: quay.io/keycloak/keycloak:24.0
    container_name: myapp-keycloak
    restart: unless-stopped
    environment:
      KEYCLOAK_ADMIN: admin
      KEYCLOAK_ADMIN_PASSWORD: admin
      KC_DB: postgres
      KC_DB_URL: jdbc:postgresql://postgres:5432/keycloak
      KC_DB_USERNAME: postgres
      KC_DB_PASSWORD: postgres
    ports:
      - "8080:8080"
    command: start-dev
    depends_on:
      postgres:
        condition: service_healthy
```

### 7.6 LocalStack（AWS サービスエミュレーター）

```yaml
services:
  localstack:
    image: localstack/localstack:latest
    container_name: myapp-localstack
    restart: unless-stopped
    environment:
      SERVICES: s3,sqs,sns,dynamodb,ses,lambda
      DEBUG: 1
      DEFAULT_REGION: ap-northeast-1
    ports:
      - "4566:4566"      # Gateway
      - "4510-4559:4510-4559"  # Service ports
    volumes:
      - localstack_data:/var/lib/localstack
      - /var/run/docker.sock:/var/run/docker.sock
      - ./docker/localstack/init:/etc/localstack/init/ready.d

volumes:
  localstack_data:
```

```bash
#!/bin/bash
# docker/localstack/init/init-aws.sh
# LocalStack 初期化スクリプト

# S3 バケット作成
awslocal s3 mb s3://uploads
awslocal s3 mb s3://avatars

# SQS キュー作成
awslocal sqs create-queue --queue-name email-queue
awslocal sqs create-queue --queue-name notification-queue

# DynamoDB テーブル作成
awslocal dynamodb create-table \
  --table-name Sessions \
  --attribute-definitions AttributeName=id,AttributeType=S \
  --key-schema AttributeName=id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST

echo "LocalStack initialization complete"
```

---

## 8. 統合 Docker Compose

### 8.1 完全な構成例

```yaml
# docker-compose.yml (統合版)
services:
  postgres:
    image: postgres:16-alpine
    container_name: myapp-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_development
      TZ: Asia/Tokyo
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./docker/postgres/init:/docker-entrypoint-initdb.d
      - ./docker/postgres/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    command: postgres -c config_file=/etc/postgresql/postgresql.conf

  redis:
    image: redis:7-alpine
    container_name: myapp-redis
    restart: unless-stopped
    ports:
      - "${REDIS_PORT:-6379}:6379"
    volumes:
      - ./docker/redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  mailpit:
    image: axllent/mailpit:latest
    container_name: myapp-mailpit
    restart: unless-stopped
    ports:
      - "${MAIL_SMTP_PORT:-1025}:1025"
      - "${MAIL_UI_PORT:-8025}:8025"
    environment:
      MP_SMTP_AUTH_ACCEPT_ANY: 1
      MP_SMTP_AUTH_ALLOW_INSECURE: 1
      MP_MAX_MESSAGES: 5000

  minio:
    image: minio/minio:latest
    container_name: myapp-minio
    restart: unless-stopped
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "${MINIO_API_PORT:-9000}:9000"
      - "${MINIO_CONSOLE_PORT:-9001}:9001"
    volumes:
      - miniodata:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 5

  minio-init:
    image: minio/mc:latest
    depends_on:
      minio:
        condition: service_healthy
    restart: "no"
    entrypoint: >
      /bin/sh -c "
      mc alias set local http://minio:9000 minioadmin minioadmin;
      mc mb local/uploads --ignore-existing;
      mc mb local/avatars --ignore-existing;
      mc anonymous set download local/avatars;
      echo 'MinIO buckets initialized';
      "

volumes:
  pgdata:
  miniodata:
```

### 8.2 プロファイルによるサービス分離

```yaml
# docker-compose.yml (プロファイル対応)
services:
  postgres:
    image: postgres:16-alpine
    # ... (基本設定)
    profiles: ["db", "full"]

  mysql:
    image: mysql:8.0
    # ... (基本設定)
    profiles: ["mysql", "full"]

  redis:
    image: redis:7-alpine
    # ... (基本設定)
    profiles: ["cache", "full"]

  mailpit:
    image: axllent/mailpit:latest
    # ... (基本設定)
    profiles: ["mail", "full"]

  minio:
    image: minio/minio:latest
    # ... (基本設定)
    profiles: ["storage", "full"]

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    # ... (基本設定)
    profiles: ["search", "full"]
```

```bash
# 必要なサービスだけ起動
docker compose --profile db --profile cache up -d

# 全サービス起動
docker compose --profile full up -d
```

### 8.3 Makefile による操作の簡略化

```makefile
# Makefile
.PHONY: up down restart logs status clean db-shell redis-shell psql

# サービスの起動
up:
	docker compose up -d

# サービスの停止
down:
	docker compose down

# サービスの再起動
restart:
	docker compose restart

# ログの表示
logs:
	docker compose logs -f

# 特定サービスのログ
logs-%:
	docker compose logs -f $*

# ステータス確認
status:
	docker compose ps

# データベースシェル
db-shell:
	docker compose exec postgres psql -U postgres -d myapp_development

# Redis シェル
redis-shell:
	docker compose exec redis redis-cli

# MySQL シェル
mysql-shell:
	docker compose exec mysql mysql -u root -proot myapp_development

# データベースのリセット
db-reset:
	docker compose down -v
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 5
	@echo "Database reset complete"

# 全データの削除
clean:
	docker compose down -v --remove-orphans
	docker volume prune -f

# バックアップ
backup:
	@mkdir -p backups
	docker compose exec postgres pg_dump -U postgres -d myapp_development \
		--format=custom > backups/backup_$(shell date +%Y%m%d_%H%M%S).dump
	@echo "Backup created"

# リストア (使用法: make restore FILE=backups/backup_xxx.dump)
restore:
	docker compose exec -T postgres pg_restore -U postgres -d myapp_development \
		--clean --if-exists < $(FILE)
	@echo "Restore complete"
```

---

## 9. サービス接続情報まとめ

```
+------------------------------------------------------------------+
|           ローカルサービス接続情報一覧                               |
+------------------------------------------------------------------+
| サービス     | ホスト:ポート       | UI / 管理画面               |
|-------------|--------------------|-----------------------------|
| PostgreSQL  | localhost:5432     | pgAdmin (localhost:5050)    |
| MySQL       | localhost:3306     | phpMyAdmin or DBeaver       |
| Redis       | localhost:6379     | RedisInsight (localhost:5540)|
| Mailpit     | localhost:1025(SMTP)| http://localhost:8025       |
| MinIO       | localhost:9000(API) | http://localhost:9001       |
| Elasticsearch| localhost:9200    | Kibana (localhost:5601)     |
| Meilisearch | localhost:7700     | http://localhost:7700       |
| RabbitMQ    | localhost:5672     | http://localhost:15672      |
| MongoDB     | localhost:27017    | Mongo Express (localhost:8081)|
| Keycloak    | localhost:8080     | http://localhost:8080/admin |
| LocalStack  | localhost:4566     | -                           |
+------------------------------------------------------------------+
```

### サービス選択ガイド

| 要件 | 推奨サービス | 代替 | 備考 |
|------|------------|------|------|
| RDB (汎用) | PostgreSQL 16 | MySQL 8.0 | JSON, 配列型が豊富 |
| RDB (レガシー) | MySQL 8.0 | MariaDB 11 | 既存資産との互換性 |
| キャッシュ | Redis 7 | Memcached | Pub/Sub も使える |
| セッション | Redis 7 | PostgreSQL | 永続化設定を有効に |
| メールテスト | Mailpit | MailHog (非推奨) | API でテスト検証可 |
| S3互換ストレージ | MinIO | LocalStack | 軽量で高速 |
| AWS 全般 | LocalStack | - | 複数サービスをエミュレート |
| 全文検索 (軽量) | Meilisearch | - | セットアップが簡単 |
| 全文検索 (高機能) | Elasticsearch | OpenSearch | 集約・分析機能が豊富 |
| メッセージキュー | RabbitMQ | Redis Streams | 複雑なルーティング |
| NoSQL | MongoDB 7 | DynamoDB (LocalStack) | ドキュメント DB |
| 認証テスト | Keycloak | mock-oauth2-server | OAuth2/OIDC 完全サポート |

---

## 10. ヘルスチェックとサービス起動順序

### 10.1 ヘルスチェックの重要性

`depends_on` だけではコンテナの「起動」しか保証されず、サービスが「使用可能」になるまで待機しない。`healthcheck` と `condition: service_healthy` を組み合わせることで、確実な起動順序を実現する。

```yaml
services:
  app:
    build: .
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      mailpit:
        condition: service_started  # ヘルスチェック不要なサービス

  postgres:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s  # 初期化中はチェックしない

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

### 10.2 アプリケーション側のリトライ

ヘルスチェックだけに依存せず、アプリケーション側でもリトライロジックを実装する。

```typescript
// lib/db.ts
import { PrismaClient } from '@prisma/client';

async function createPrismaClient(maxRetries = 5): Promise<PrismaClient> {
  const prisma = new PrismaClient();

  for (let i = 0; i < maxRetries; i++) {
    try {
      await prisma.$connect();
      console.log('Database connected');
      return prisma;
    } catch (error) {
      console.log(`Database connection attempt ${i + 1}/${maxRetries} failed`);
      if (i === maxRetries - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }
  }

  throw new Error('Failed to connect to database');
}

export const prisma = await createPrismaClient();
```

---

## アンチパターン

### アンチパターン 1: Volume なしでデータベースを運用

```yaml
# NG: Volume 未設定 → docker-compose down でデータ全消失
services:
  postgres:
    image: postgres:16-alpine
    # volumes が未設定

# OK: 名前付き Volume でデータを永続化
services:
  postgres:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
    driver: local
```

**問題点**: Volume を設定しないと `docker-compose down` や `docker-compose rm` でデータベースの全データが消失する。開発中のテストデータやマイグレーション状態が失われ、再構築に時間がかかる。

### アンチパターン 2: 本番と同じ認証情報をローカルで使用

```yaml
# NG: 本番の認証情報をそのまま使用
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: ${PROD_DB_PASSWORD}  # 本番パスワード

# OK: 開発専用の固定パスワード
services:
  postgres:
    environment:
      POSTGRES_PASSWORD: postgres  # 開発専用
```

**問題点**: 本番認証情報がローカル環境に残ると、`docker-compose.yml` のコミットや `.env` の誤共有で漏洩するリスクがある。開発環境では固定のシンプルなパスワードを使い、本番環境とは完全に分離する。

### アンチパターン 3: バインドマウントでデータベースのデータを管理

```yaml
# NG: バインドマウント (I/O が遅い + 権限問題)
services:
  postgres:
    volumes:
      - ./data/postgres:/var/lib/postgresql/data

# OK: 名前付き Volume (高速 + 権限問題なし)
services:
  postgres:
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

**問題点**: macOS / Windows ではバインドマウントの I/O 性能が低く、特にデータベースの書き込みパフォーマンスに大きく影響する。また、コンテナ内の UID/GID とホストの権限が一致しない問題も発生しやすい。名前付き Volume はこれらの問題を回避する。

### アンチパターン 4: ヘルスチェックなしの depends_on

```yaml
# NG: ヘルスチェックなし → DB 起動前にアプリが接続を試みる
services:
  app:
    depends_on:
      - postgres  # コンテナ起動のみ保証

# OK: ヘルスチェックで起動完了を保証
services:
  app:
    depends_on:
      postgres:
        condition: service_healthy

  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
```

**問題点**: `depends_on` はコンテナの起動順序しか保証しない。PostgreSQL のコンテナが起動しても、初期化スクリプトの実行やソケットのリッスン開始まで数秒かかる。この間にアプリケーションが接続を試みると接続エラーが発生する。

### アンチパターン 5: latest タグを本番同等のサービスに使用

```yaml
# NG: バージョンが不定
services:
  postgres:
    image: postgres:latest
  redis:
    image: redis:latest

# OK: メジャーバージョンを固定
services:
  postgres:
    image: postgres:16-alpine
  redis:
    image: redis:7-alpine
```

**問題点**: `latest` タグは `docker pull` のタイミングで異なるバージョンが取得される可能性がある。チームメンバー間でバージョンが異なると、互換性の問題やデータフォーマットの不整合が発生する。メジャーバージョンを固定し、マイナーバージョンのアップデートは意図的に行う。

---

## FAQ

### Q1: Docker for Mac/Windows でデータベースの I/O が遅いのですが、改善方法はありますか？

**A**: 名前付き Volume を使うことが最も効果的。バインドマウント (`./data:/var/lib/postgresql/data`) は macOS/Windows ではファイルシステムの変換オーバーヘッドが大きい。名前付き Volume は Docker VM 内のネイティブファイルシステムを使うため、I/O 性能が大幅に向上する。PostgreSQL の場合、`fsync=off` や `synchronous_commit=off` を開発専用設定として追加するのも有効。macOS では OrbStack の利用も検討する価値がある。

### Q2: 複数プロジェクトで同じポート (5432 等) を使いたい場合はどうしますか？

**A**: プロジェクトごとにポートをずらす (`5432`, `5433`, `5434` 等) か、Docker Compose のプロファイル機能で排他的に起動する。もう一つの方法は、すべてのプロジェクトで共通の開発用インフラを一つの `docker-compose.yml` で管理し、データベース名で分離する方式。Dev Container を使う場合は、コンテナ内から Docker ネットワーク経由でアクセスするためポート競合は発生しない。`.env` ファイルでポート番号を変数化するのも有効な手段である。

### Q3: MailHog の代わりに Mailpit を使うべきですか？

**A**: はい。MailHog はメンテナンスが停止しており、Mailpit がその後継として活発に開発されている。Mailpit は MailHog と API 互換性があり、より高速で、HTML メールのレンダリングや添付ファイルの表示も改善されている。Docker イメージは `axllent/mailpit` で、SMTP ポートは `1025`、Web UI は `8025` と同じ設定で移行可能。さらに、Mailpit は検索 API が充実しており、E2E テストでのメール検証にも優れている。

### Q4: docker compose down と docker compose stop の違いは？

**A**: `docker compose stop` はコンテナを停止するだけで、コンテナとネットワークは保持される。次回 `docker compose start` で高速に再開できる。一方、`docker compose down` はコンテナとネットワークを削除する。`-v` フラグを付けると Volume も削除される。開発中は `stop` / `start` を使い、環境をリセットしたい場合のみ `down` を使うのが効率的である。

### Q5: 初期化スクリプトが再実行されないのはなぜですか？

**A**: PostgreSQL / MySQL の初期化スクリプト (`docker-entrypoint-initdb.d/`) は、データディレクトリが空の場合にのみ実行される。Volume にデータが残っている場合はスキップされる。初期化スクリプトを再実行するには、Volume を削除して再作成する必要がある: `docker compose down -v && docker compose up -d`。

### Q6: LocalStack と MinIO の使い分けは？

**A**: MinIO は S3 互換ストレージに特化しており、軽量で高速。S3 のみが必要な場合は MinIO を推奨する。LocalStack は S3 以外にも SQS, SNS, DynamoDB, Lambda, SES など多数の AWS サービスをエミュレートする。複数の AWS サービスを使う場合は LocalStack が適している。ただし LocalStack は Pro 版（有料）でないとサポートされないサービスもある。

### Q7: テスト用データベースはどのように管理するのが良いですか？

**A**: 以下の 3 つのパターンがある。
1. **テスト用 DB を別途作成**: 初期化スクリプトで `myapp_test` DB を作成し、テスト実行前にマイグレーション
2. **テストごとにリセット**: 各テストの前にトランザクションを開始し、終了後にロールバック
3. **テスト用コンテナ**: `docker compose --profile test up -d` で専用コンテナを起動

最も一般的なのはパターン 1 + パターン 2 の組み合わせであり、テスト DB をパターン 1 で用意し、各テストの分離をパターン 2 で行う。

---

## まとめ

| 項目 | 要点 |
|------|------|
| PostgreSQL | alpine イメージ + init スクリプトで開発DB自動構築 |
| MySQL | `utf8mb4` + 開発用パフォーマンス設定を `my.cnf` で管理 |
| Redis | 用途別に設定を分離（キャッシュ / セッション / キュー） |
| Mailpit | MailHog の後継。SMTP テスト + API テスト検証に最適 |
| MinIO | S3 互換 API。`forcePathStyle: true` が必須 |
| LocalStack | 複数 AWS サービスのエミュレーション |
| Meilisearch | 軽量全文検索。Elasticsearch の代替 |
| Volume 設計 | DB データは名前付き Volume。バインドマウントは避ける |
| ヘルスチェック | 全サービスに `healthcheck` を設定し、依存順序を保証 |
| ポート管理 | `.env` で変数化、またはプロファイルで排他管理 |
| プロファイル | `--profile` で必要なサービスだけ起動 |
| Makefile | 頻繁に使うコマンドを簡略化 |
| バックアップ | `pg_dump` / `mysqldump` でスキーマとデータを管理 |
| GUI ツール | pgAdmin / RedisInsight / Kibana 等を必要に応じて追加 |

## 次に読むべきガイド

- [Dev Container](./01-devcontainer.md) -- Docker 開発環境を VS Code / Codespaces と統合
- [Docker Compose 基礎](../../docker-container-guide/docs/02-compose/00-compose-basics.md) -- Compose ファイルの構文と設計パターン
- [プロジェクト標準](../03-team-setup/00-project-standards.md) -- チーム共通の設定ファイル管理

## 参考文献

1. **Docker Hub 公式イメージ** -- https://hub.docker.com/ -- PostgreSQL, MySQL, Redis 等の公式イメージと設定オプション
2. **Mailpit 公式** -- https://mailpit.axllent.org/ -- MailHog 後継の SMTP テストツール
3. **MinIO 公式ドキュメント** -- https://min.io/docs/minio/container/index.html -- MinIO の Docker デプロイとクライアント設定
4. **Docker Compose 公式リファレンス** -- https://docs.docker.com/compose/compose-file/ -- Compose ファイル仕様の詳細
5. **LocalStack 公式** -- https://localstack.cloud/ -- AWS サービスのローカルエミュレーション
6. **Meilisearch 公式** -- https://www.meilisearch.com/ -- 軽量全文検索エンジン
7. **Redis 公式ドキュメント** -- https://redis.io/docs/ -- Redis の設定とコマンドリファレンス
