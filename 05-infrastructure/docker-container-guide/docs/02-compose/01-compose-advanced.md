# Docker Compose 応用 (Compose Advanced)

> プロファイル、depends_on の高度な制御、healthcheck、環境変数の管理パターンなど、Docker Compose の応用機能を活用してプロダクション品質の構成を構築する。

## この章で学ぶこと

1. **プロファイルによるサービスの選択的起動** -- 開発・テスト・監視など、用途に応じたサービスのグルーピングと選択的起動を実装する
2. **depends_on と healthcheck の高度な制御** -- サービス間の依存関係を精密に管理し、確実な起動順序を保証する
3. **環境変数と設定の管理パターン** -- 複数環境での設定切り替え、シークレット管理、ファイルのオーバーライドを実践する

---

## 1. プロファイル (Profiles)

### 1.1 プロファイルの概要

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
|    prometheus, grafana                                           |
|                                                                  |
|  [test プロファイル] (--profile test で起動)                      |
|    test-runner, db-test                                          |
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
```

---

## 2. depends_on と healthcheck

### 2.1 depends_on の 3 つの条件

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

### 2.2 healthcheck の詳細設定

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

  # Redis
  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  # HTTP サービス
  api:
    build: .
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/health || exit 1"]
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

---

## 3. 環境変数の管理

### 3.1 環境変数の優先順位

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

# .env.development (アプリ用。env_file で明示的に読み込む)
NODE_ENV=development
DATABASE_URL=postgresql://postgres:postgres@db:5432/myapp_dev
REDIS_URL=redis://redis:6379
LOG_LEVEL=debug

# .env.production (本番用)
NODE_ENV=production
DATABASE_URL=postgresql://user:password@db-prod:5432/myapp
LOG_LEVEL=warn
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

### 3.3 シークレット管理

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

secrets:
  db_password:
    file: ./secrets/db_password.txt     # ファイルから読み込み
  api_key:
    environment: API_KEY                 # 環境変数から (Compose V2.22+)
```

---

## 4. 複数 Compose ファイルのマージ

### 4.1 オーバーライドパターン

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
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

### 4.2 マージのコマンド

```bash
# 開発 (compose.yml + compose.override.yml を自動マージ)
docker compose up -d

# 本番 (override を除外し、prod を適用)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 設定のマージ結果を確認
docker compose -f docker-compose.yml -f docker-compose.prod.yml config
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
```

### 5.2 ロギング設定

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

---

## 6. YAML アンカーとエイリアス

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

---

## 7. 高度な設定比較

| 機能 | 基本設定 | 応用設定 |
|------|---------|---------|
| サービス起動 | `depends_on: [db]` | `depends_on: { db: { condition: service_healthy } }` |
| 環境変数 | `environment: {KEY: val}` | `env_file` + `secrets` + 優先順位管理 |
| ネットワーク | デフォルト | `internal: true` + 複数ネットワーク分離 |
| ログ | デフォルト (無制限) | `json-file` + `max-size` + `max-file` |
| リソース | 無制限 | `deploy.resources.limits` で CPU/メモリ制限 |
| プロファイル | 全サービス起動 | `profiles` で用途別グルーピング |
| 設定管理 | 単一ファイル | `override.yml` + `prod.yml` でレイヤー化 |

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

---

## FAQ

### Q1: YAML アンカーと Extension Fields (x- プレフィックス) の違いは何ですか？

**A**: YAML アンカー (`&` / `*`) は YAML 標準の参照機構で、同じ値を複数箇所で再利用する。Extension Fields (`x-` プレフィックス) は Compose 仕様の機能で、Compose が無視するカスタムフィールドを定義できる。両者を組み合わせ、`x-common: &common` でトップレベルに共通設定を定義し、各サービスで `<<: *common` で展開するのが一般的なパターン。

### Q2: プロファイルと複数 Compose ファイルのどちらを使うべきですか？

**A**: 同じ `docker-compose.yml` 内で用途別にサービスをグルーピングしたい場合はプロファイルが適切（例: debug ツール、監視ツール）。環境全体の設定を切り替えたい場合（開発 vs 本番、ポート公開の有無、リソース制限など）は複数ファイルのオーバーライドが適切。両方を併用することもできる。

### Q3: Compose で使える Interpolation (変数展開) の構文は？

**A**: `${VARIABLE}` が基本形。デフォルト値は `${VARIABLE:-default}` (未設定時) と `${VARIABLE-default}` (未定義時のみ)。エラーにする場合は `${VARIABLE:?error message}`。これらは `.env` ファイルまたはシェルの環境変数から値を取得する。Compose ファイル内の `environment:` セクションの値は展開されるが、コンテナ内での展開とは異なる点に注意。

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

## 次に読むべきガイド

- [Compose 開発ワークフロー](./02-development-workflow.md) -- ホットリロード、デバッグ、CI 統合
- [Docker Compose 基礎](./00-compose-basics.md) -- 基本構文の復習
- [コンテナセキュリティ](../06-security/00-container-security.md) -- セキュリティのベストプラクティス

## 参考文献

1. **Compose Specification - Profiles** -- https://docs.docker.com/compose/profiles/ -- プロファイル機能の公式ドキュメント
2. **Compose Specification - Healthcheck** -- https://docs.docker.com/compose/compose-file/05-services/#healthcheck -- ヘルスチェック設定の詳細
3. **Environment variables in Compose** -- https://docs.docker.com/compose/environment-variables/ -- 環境変数の優先順位と管理方法
4. **Compose file merge** -- https://docs.docker.com/compose/multiple-compose-files/ -- 複数ファイルのマージ規則
