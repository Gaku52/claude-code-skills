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

### 2.3 PostgreSQL 設定のカスタマイズ

```conf
# docker/postgres/postgresql.conf (開発用に最適化)
# パフォーマンス (開発マシン向け)
shared_buffers = 256MB
work_mem = 16MB
maintenance_work_mem = 128MB
effective_cache_size = 512MB

# ログ (デバッグしやすく)
log_statement = 'all'
log_duration = on
log_min_duration_statement = 100
log_line_prefix = '%t [%p] %u@%d '

# 開発用設定 (本番では使わない)
fsync = off
synchronous_commit = off
full_page_writes = off
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

# ログ
general_log = 1
general_log_file = /var/log/mysql/general.log
slow_query_log = 1
slow_query_log_file = /var/log/mysql/slow.log
long_query_time = 1

[client]
default-character-set = utf8mb4
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
```

---

## 5. MailHog (メールテスト)

### 5.1 設定

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

### 5.2 アプリからの接続設定

```typescript
// config/mail.ts
const mailConfig = {
  development: {
    host: 'localhost',
    port: 1025,
    secure: false,
    // MailHog は認証不要
  },
  production: {
    host: process.env.SMTP_HOST,
    port: parseInt(process.env.SMTP_PORT || '587'),
    secure: true,
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS,
    },
  },
};

export default mailConfig[process.env.NODE_ENV || 'development'];
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
      mc anonymous set download local/avatars;
      echo 'MinIO buckets initialized';
      "

volumes:
  miniodata:
    driver: local
```

### 6.2 AWS SDK での接続

```typescript
// config/storage.ts
import { S3Client } from '@aws-sdk/client-s3';

function createStorageClient(): S3Client {
  if (process.env.NODE_ENV === 'development') {
    return new S3Client({
      endpoint: 'http://localhost:9000',
      region: 'us-east-1',
      credentials: {
        accessKeyId: 'minioadmin',
        secretAccessKey: 'minioadmin',
      },
      forcePathStyle: true, // MinIO ではパススタイル必須
    });
  }

  // 本番は通常の S3
  return new S3Client({ region: process.env.AWS_REGION });
}

export const storageClient = createStorageClient();
```

---

## 7. 統合 Docker Compose

### 7.1 完全な構成例

```yaml
# docker-compose.yml (統合版)
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: myapp-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp_development
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

  redis:
    image: redis:7-alpine
    container_name: myapp-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  mailhog:
    image: mailhog/mailhog:latest
    container_name: myapp-mailhog
    ports:
      - "1025:1025"
      - "8025:8025"

  minio:
    image: minio/minio:latest
    container_name: myapp-minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - miniodata:/data
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "mc", "ready", "local"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
  miniodata:
```

---

## 8. サービス接続情報まとめ

```
+------------------------------------------------------------------+
|           ローカルサービス接続情報一覧                               |
+------------------------------------------------------------------+
| サービス     | ホスト:ポート       | UI / 管理画面               |
|-------------|--------------------|-----------------------------|
| PostgreSQL  | localhost:5432     | pgAdmin or DBeaver          |
| MySQL       | localhost:3306     | phpMyAdmin or DBeaver       |
| Redis       | localhost:6379     | RedisInsight (別途)         |
| MailHog     | localhost:1025(SMTP)| http://localhost:8025       |
| MinIO       | localhost:9000(API) | http://localhost:9001       |
+------------------------------------------------------------------+
```

### サービス選択ガイド

| 要件 | 推奨サービス | 代替 |
|------|------------|------|
| RDB (汎用) | PostgreSQL 16 | MySQL 8.0 |
| キャッシュ | Redis 7 | Memcached |
| メールテスト | MailHog | Mailpit (後継) |
| S3互換ストレージ | MinIO | LocalStack |
| 全文検索 | Elasticsearch | Meilisearch |
| メッセージキュー | RabbitMQ | Redis Streams |

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

---

## FAQ

### Q1: Docker for Mac/Windows でデータベースの I/O が遅いのですが、改善方法はありますか？

**A**: 名前付き Volume を使うことが最も効果的。バインドマウント (`./data:/var/lib/postgresql/data`) は macOS/Windows ではファイルシステムの変換オーバーヘッドが大きい。名前付き Volume は Docker VM 内のネイティブファイルシステムを使うため、I/O 性能が大幅に向上する。PostgreSQL の場合、`fsync=off` や `synchronous_commit=off` を開発専用設定として追加するのも有効。

### Q2: 複数プロジェクトで同じポート (5432 等) を使いたい場合はどうしますか？

**A**: プロジェクトごとにポートをずらす (`5432`, `5433`, `5434` 等) か、Docker Compose のプロファイル機能で排他的に起動する。もう一つの方法は、すべてのプロジェクトで共通の開発用インフラを一つの `docker-compose.yml` で管理し、データベース名で分離する方式。Dev Container を使う場合は、コンテナ内から Docker ネットワーク経由でアクセスするためポート競合は発生しない。

### Q3: MailHog の代わりに Mailpit を使うべきですか？

**A**: はい。MailHog はメンテナンスが停止しており、Mailpit がその後継として活発に開発されている。Mailpit は MailHog と API 互換性があり、より高速で、HTML メールのレンダリングや添付ファイルの表示も改善されている。Docker イメージは `axllent/mailpit` で、SMTP ポートは `1025`、Web UI は `8025` と同じ設定で移行可能。

---

## まとめ

| 項目 | 要点 |
|------|------|
| PostgreSQL | alpine イメージ + init スクリプトで開発DB自動構築 |
| MySQL | `utf8mb4` + 開発用パフォーマンス設定を `my.cnf` で管理 |
| Redis | 開発では永続化不要。メモリ制限とLRUポリシーを設定 |
| MailHog/Mailpit | SMTP テスト用。Web UI でメール確認。認証不要 |
| MinIO | S3 互換 API。`forcePathStyle: true` が必須 |
| Volume 設計 | DB データは名前付き Volume。バインドマウントは避ける |
| ヘルスチェック | 全サービスに `healthcheck` を設定し、依存順序を保証 |
| ポート管理 | プロジェクトごとにポートを分離するか、Dev Container 内で解決 |

## 次に読むべきガイド

- [Dev Container](./01-devcontainer.md) -- Docker 開発環境を VS Code / Codespaces と統合
- [Docker Compose 基礎](../../docker-container-guide/docs/02-compose/00-compose-basics.md) -- Compose ファイルの構文と設計パターン
- [プロジェクト標準](../03-team-setup/00-project-standards.md) -- チーム共通の設定ファイル管理

## 参考文献

1. **Docker Hub 公式イメージ** -- https://hub.docker.com/ -- PostgreSQL, MySQL, Redis 等の公式イメージと設定オプション
2. **MailHog GitHub** -- https://github.com/mailhog/MailHog -- SMTP テストツールのドキュメント (後継: Mailpit)
3. **MinIO 公式ドキュメント** -- https://min.io/docs/minio/container/index.html -- MinIO の Docker デプロイとクライアント設定
4. **Docker Compose 公式リファレンス** -- https://docs.docker.com/compose/compose-file/ -- Compose ファイル仕様の詳細
