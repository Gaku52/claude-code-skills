# Docker Compose 基礎 (Compose Basics)

> docker-compose.yml の構文と概念を体系的に理解し、services / volumes / networks を組み合わせたマルチコンテナアプリケーション環境を構築する基礎力を身につける。

## この章で学ぶこと

1. **docker-compose.yml の構文と基本構造** -- YAML 記法に基づく Compose ファイルの各セクション（services, volumes, networks）の役割と記法を理解する
2. **サービス定義とコンテナのライフサイクル管理** -- イメージ指定、ビルド設定、ポート公開、環境変数など、サービス定義の主要オプションを習得する
3. **ボリュームとネットワークによるデータ・通信の管理** -- コンテナ間のデータ永続化と内部通信の設計パターンを学ぶ

---

## 1. Docker Compose とは

### 1.1 単一コンテナ vs Compose

```
+------------------------------------------------------------------+
|          単一コンテナ vs Docker Compose                            |
+------------------------------------------------------------------+
|                                                                  |
|  [単一コンテナ (docker run)]                                      |
|  $ docker run -d --name web -p 3000:3000 \                       |
|      -e DATABASE_URL=... \                                       |
|      --network mynet myapp:latest                                |
|  $ docker run -d --name db -p 5432:5432 \                        |
|      -v pgdata:/var/lib/postgresql/data \                        |
|      --network mynet postgres:16                                 |
|  → コマンドが長い、管理が煩雑、再現性が低い                        |
|                                                                  |
|  [Docker Compose]                                                |
|  $ docker compose up -d                                          |
|  → 1コマンドで全サービス起動。設定は YAML ファイルで管理            |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.2 Docker Compose のアーキテクチャ

```
+------------------------------------------------------------------+
|              Docker Compose の内部アーキテクチャ                     |
+------------------------------------------------------------------+
|                                                                  |
|  docker compose up                                               |
|    |                                                             |
|    +-- compose.yml のパース & バリデーション                       |
|    |     |                                                       |
|    |     +-- YAML → 内部モデルへの変換                            |
|    |     +-- 環境変数の展開 (.env ファイル含む)                     |
|    |     +-- 複数 Compose ファイルのマージ                         |
|    |     +-- プロファイルのフィルタリング                           |
|    |                                                             |
|    +-- 依存関係グラフの構築                                       |
|    |     |                                                       |
|    |     +-- depends_on の解析                                   |
|    |     +-- 循環依存の検出                                       |
|    |     +-- 起動順序の決定                                       |
|    |                                                             |
|    +-- リソースの作成                                             |
|    |     |                                                       |
|    |     +-- ネットワーク作成 (docker network create)             |
|    |     +-- ボリューム作成 (docker volume create)                |
|    |     +-- シークレット/コンフィグの準備                          |
|    |                                                             |
|    +-- サービスの起動                                             |
|          |                                                       |
|          +-- イメージの pull または build                          |
|          +-- コンテナ作成 (docker create)                         |
|          +-- コンテナ起動 (docker start)                          |
|          +-- ヘルスチェック待機                                    |
|          +-- 依存サービスの起動                                    |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.3 プロジェクト名とコンテナ名の仕組み

Docker Compose はプロジェクト名をベースにリソースの名前を決定する。プロジェクト名は以下の優先順位で決まる:

```bash
# 1. -p / --project-name フラグ（最高優先度）
docker compose -p myproject up -d

# 2. COMPOSE_PROJECT_NAME 環境変数
export COMPOSE_PROJECT_NAME=myproject
docker compose up -d

# 3. compose.yml 内の name フィールド
# compose.yml
# name: myproject

# 4. compose.yml があるディレクトリ名（デフォルト）
# /home/user/my-app/ → プロジェクト名: my-app
```

リソースの命名規則:

```
+------------------------------------------------------------------+
|              プロジェクト名によるリソース命名                        |
+------------------------------------------------------------------+
|                                                                  |
|  プロジェクト名: myproject                                        |
|                                                                  |
|  コンテナ名:    myproject-web-1, myproject-db-1                   |
|  ネットワーク名: myproject_default                                |
|  ボリューム名:   myproject_pgdata                                 |
|                                                                  |
|  container_name で明示的に指定も可能:                              |
|  services:                                                       |
|    web:                                                          |
|      container_name: my-web-server  # 固定名                     |
|      # ※ container_name を指定すると                              |
|      #   スケール (--scale) が使えなくなる                         |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.4 Compose ファイルのバージョン

```
+------------------------------------------------------------------+
|              Compose ファイル仕様の歴史                             |
+------------------------------------------------------------------+
| バージョン        | 特徴                      | 推奨度           |
|------------------|--------------------------|-----------------|
| version: "2"     | Docker Engine 統合前      | 非推奨           |
| version: "3"     | Swarm 対応                | 非推奨           |
| version: "3.8"   | 最終明示バージョン         | 互換性用途のみ    |
| (バージョン省略)  | Compose Spec 準拠         | 推奨 (現在の標準) |
+------------------------------------------------------------------+
|                                                                  |
|  現在は version キーを省略し、Compose Specification に              |
|  準拠するのが推奨。Docker Compose V2 が自動判定する。              |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. docker-compose.yml の基本構造

### 2.1 全体構造

```yaml
# docker-compose.yml

# (version は省略推奨)

# サービス定義 (コンテナの設定)
services:
  web:
    image: node:20-alpine
    # ... 設定
  db:
    image: postgres:16-alpine
    # ... 設定

# ボリューム定義 (データ永続化)
volumes:
  pgdata:
    driver: local

# ネットワーク定義 (コンテナ間通信)
networks:
  backend:
    driver: bridge

# シークレット定義 (機密情報)
secrets:
  db_password:
    file: ./secrets/db_password.txt

# 設定定義 (設定ファイル)
configs:
  nginx_conf:
    file: ./nginx/nginx.conf
```

### 2.2 Compose ファイルの階層図

```
+------------------------------------------------------------------+
|              docker-compose.yml の構造                             |
+------------------------------------------------------------------+
|                                                                  |
|  docker-compose.yml                                              |
|    |                                                             |
|    +-- services:          ← コンテナ定義 (必須)                   |
|    |     +-- web:                                                |
|    |     |    +-- image / build                                  |
|    |     |    +-- ports                                          |
|    |     |    +-- environment                                    |
|    |     |    +-- volumes                                        |
|    |     |    +-- networks                                       |
|    |     |    +-- depends_on                                     |
|    |     |    +-- restart                                        |
|    |     +-- db:                                                 |
|    |          +-- ...                                            |
|    |                                                             |
|    +-- volumes:           ← ボリューム定義 (任意)                  |
|    |     +-- pgdata:                                             |
|    |                                                             |
|    +-- networks:          ← ネットワーク定義 (任意)                |
|    |     +-- backend:                                            |
|    |                                                             |
|    +-- secrets:           ← シークレット定義 (任意)                |
|    +-- configs:           ← 設定ファイル定義 (任意)                |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 3. services の詳細

### 3.1 イメージ指定 vs ビルド

```yaml
services:
  # パターン 1: 既存イメージを使用
  db:
    image: postgres:16-alpine

  # パターン 2: Dockerfile からビルド
  web:
    build:
      context: .               # ビルドコンテキスト
      dockerfile: Dockerfile   # Dockerfile パス (デフォルト: Dockerfile)
      args:                    # ビルド引数
        NODE_ENV: production
      target: runner           # マルチステージの対象ステージ
      cache_from:
        - myapp:latest
    image: myapp:latest        # ビルド後のタグ名

  # パターン 3: 簡易ビルド
  api:
    build: ./api               # context のみ指定 (Dockerfile は自動検出)
```

### 3.2 ポート公開

```yaml
services:
  web:
    ports:
      # ホスト:コンテナ
      - "3000:3000"            # localhost:3000 → コンテナ:3000
      - "443:443"

      # ホスト IP 指定
      - "127.0.0.1:3000:3000"  # localhost のみ (外部からアクセス不可)

      # ホストポートをランダムに割り当て
      - "3000"                 # ランダムポート → コンテナ:3000

      # プロトコル指定
      - "6379:6379/tcp"

    # コンテナ間のみ公開 (ホストからはアクセス不可)
    expose:
      - "3000"
```

### 3.3 環境変数

```yaml
services:
  web:
    environment:
      # キー=値 形式
      NODE_ENV: production
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp
      # 値なし = ホストの環境変数を引き継ぐ
      API_KEY:

    # .env ファイルから読み込み
    env_file:
      - .env
      - .env.local             # 後のファイルが優先
```

### 3.4 ボリュームマウント

```yaml
services:
  web:
    volumes:
      # 名前付きボリューム
      - node_modules:/app/node_modules

      # バインドマウント (ホストディレクトリ)
      - ./src:/app/src

      # 読み取り専用
      - ./config:/app/config:ro

      # tmpfs (メモリ上)
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 100000000  # 100MB

      # 詳細構文
      - type: bind
        source: ./data
        target: /app/data
        consistency: cached   # macOS パフォーマンス改善
```

### 3.5 再起動ポリシー

```yaml
services:
  web:
    restart: unless-stopped
    # no           : 再起動しない (デフォルト)
    # always       : 常に再起動
    # on-failure   : 異常終了時のみ再起動
    # unless-stopped: 手動停止以外は再起動
```

### 3.6 depends_on と起動順序制御

```yaml
services:
  web:
    build: .
    depends_on:
      db:
        condition: service_healthy     # ヘルスチェック通過後に起動
        restart: true                  # db 再起動時に web も再起動
      redis:
        condition: service_started     # コンテナ起動のみ確認
      migrations:
        condition: service_completed_successfully  # 正常終了後に起動

  migrations:
    build: .
    command: ["npm", "run", "migrate"]
    depends_on:
      db:
        condition: service_healthy

  db:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s

  redis:
    image: redis:7-alpine
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
```

```
+------------------------------------------------------------------+
|              depends_on の condition 一覧                          |
+------------------------------------------------------------------+
|                                                                  |
|  condition               | 説明                                  |
|  ----------------------- | ------------------------------------- |
|  service_started         | コンテナが起動したら (デフォルト)       |
|  service_healthy         | ヘルスチェックが healthy になったら     |
|  service_completed_      | コンテナが正常終了 (exit 0) したら      |
|    successfully          |                                       |
|                                                                  |
|  起動順序の例:                                                    |
|  db (healthy) → migrations (completed) → web (start)             |
|                                                                  |
|  ※ service_healthy には healthcheck の定義が必須                  |
|  ※ service_completed_successfully は                              |
|    マイグレーションやシード処理で活用                               |
|                                                                  |
+------------------------------------------------------------------+
```

### 3.7 リソース制限

```yaml
services:
  web:
    deploy:
      resources:
        limits:
          cpus: "1.0"           # CPU コア数の上限
          memory: 512M          # メモリ上限
        reservations:
          cpus: "0.25"          # 予約 CPU
          memory: 128M          # 予約メモリ

  db:
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 256M

    # OOM Killer の調整
    oom_kill_disable: false     # OOM Killer を無効にしない（推奨）
    oom_score_adj: -500         # OOM スコアの調整（低い = kill されにくい）
```

### 3.8 ログ設定

```yaml
services:
  web:
    logging:
      driver: json-file        # デフォルトのログドライバー
      options:
        max-size: "10m"        # ログファイルの最大サイズ
        max-file: "3"          # ローテーション数
        compress: "true"       # 圧縮の有効化
        tag: "{{.Name}}"       # ログタグ

  # syslog に送信
  api:
    logging:
      driver: syslog
      options:
        syslog-address: "tcp://logserver:514"
        syslog-facility: daemon
        tag: "api-service"

  # ログを無効化（大量ログを出すサービス向け）
  load-test:
    logging:
      driver: none
```

### 3.9 ヘルスチェック

```yaml
services:
  web:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s            # チェック間隔
      timeout: 10s             # タイムアウト
      retries: 3               # リトライ回数
      start_period: 40s        # 起動猶予期間（この間の失敗はカウントしない）
      start_interval: 5s       # 起動中のチェック間隔 (Compose v2.20+)

  # シェルコマンドを使ったヘルスチェック
  db:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d myapp"]
      interval: 5s
      timeout: 5s
      retries: 5
      start_period: 10s

  # ヘルスチェックを無効化（デフォルトのヘルスチェックがあるイメージ）
  custom-service:
    healthcheck:
      disable: true
```

```
+------------------------------------------------------------------+
|              ヘルスチェックのステートマシン                          |
+------------------------------------------------------------------+
|                                                                  |
|  コンテナ起動                                                     |
|    |                                                             |
|    v                                                             |
|  [starting]  ← start_period の間はここに留まる                    |
|    |                                                             |
|    +-- チェック成功 → [healthy]                                   |
|    |                    |                                        |
|    |                    +-- チェック失敗 (retries 回) → [unhealthy]|
|    |                    |                                        |
|    |                    +-- チェック成功 → [healthy] (ループ)      |
|    |                                                             |
|    +-- start_period 経過後もチェック失敗 → [unhealthy]             |
|                                                                  |
|  ※ unhealthy になっても restart ポリシーがないと再起動しない       |
|  ※ depends_on + service_healthy で他サービスの起動をブロック       |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 4. volumes (ボリューム)

### 4.1 ボリュームの種類

```
+------------------------------------------------------------------+
|              ボリュームの種類と特徴                                  |
+------------------------------------------------------------------+
|                                                                  |
|  [名前付きボリューム] (Named Volume)                               |
|  volumes:                                                        |
|    pgdata:                                                       |
|      driver: local                                               |
|  → Docker が管理。docker volume ls で確認可能                     |
|  → コンテナ間で共有可能。永続化が保証される                        |
|  → macOS/Windows でも高速 (Docker VM 内)                          |
|                                                                  |
|  [バインドマウント] (Bind Mount)                                   |
|  volumes:                                                        |
|    - ./src:/app/src                                              |
|  → ホストのディレクトリをそのままマウント                           |
|  → 開発中のソースコード共有に最適                                  |
|  → macOS/Windows では I/O が遅い場合がある                        |
|                                                                  |
|  [tmpfs]                                                         |
|  tmpfs:                                                          |
|    - /tmp                                                        |
|  → メモリ上に作成。コンテナ停止で消失                              |
|  → 一時ファイルやキャッシュに最適                                  |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.2 ボリュームの定義

```yaml
volumes:
  # シンプルな定義
  pgdata:

  # ドライバー指定
  mysql_data:
    driver: local

  # 外部ボリューム (docker volume create で事前作成)
  shared_data:
    external: true

  # ドライバーオプション
  nfs_data:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.100,rw
      device: ":/exports/data"
```

---

## 5. networks (ネットワーク)

### 5.1 ネットワークの仕組み

```
+------------------------------------------------------------------+
|              Compose ネットワークの仕組み                           |
+------------------------------------------------------------------+
|                                                                  |
|  docker compose up 実行時:                                       |
|  → デフォルトで {プロジェクト名}_default ネットワークが作成される   |
|  → 全サービスがこのネットワークに接続                              |
|  → サービス名で DNS 解決が可能                                    |
|                                                                  |
|  +--- default ネットワーク -------------------------+              |
|  |                                                 |              |
|  |  [web]                        [db]              |              |
|  |  curl http://db:5432    <---> PostgreSQL         |              |
|  |  curl http://redis:6379 <---> [redis]           |              |
|  |                                                 |              |
|  +-------------------------------------------------+              |
|                                                                  |
|  ※ サービス名 = DNS ホスト名 として自動解決される                  |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.2 ネットワークの使い分け

```yaml
services:
  web:
    networks:
      - frontend
      - backend

  api:
    networks:
      - backend

  db:
    networks:
      - backend    # web からは直接アクセスできない

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # 外部からのアクセスを遮断
```

### 5.3 ネットワーク比較

| 項目 | デフォルト | カスタム bridge | internal | host |
|------|----------|---------------|----------|------|
| 自動作成 | あり | なし | なし | N/A |
| コンテナ間通信 | 全サービス | 指定サービスのみ | 指定サービスのみ | ホストネットワーク |
| 外部アクセス | ports で公開 | ports で公開 | 不可 | ポート直接 |
| DNS 解決 | サービス名 | サービス名 | サービス名 | ホスト名 |
| セキュリティ | 低 | 中 | 高 | 低 |
| 用途 | 小規模 | 一般 | DB/内部API | パフォーマンス |

### 5.4 ネットワークの詳細設定

```yaml
networks:
  # カスタムサブネット指定
  backend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
          gateway: 172.28.0.1

  # DNS 設定
  custom_dns:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: custom0

services:
  web:
    networks:
      backend:
        ipv4_address: 172.28.0.10   # 固定 IP アドレス
        aliases:
          - web.local               # 追加の DNS エイリアス
          - frontend.local

  api:
    networks:
      backend:
        ipv4_address: 172.28.0.20
        aliases:
          - api.local

    # DNS 設定
    dns:
      - 8.8.8.8
      - 8.8.4.4
    dns_search:
      - example.com

    # /etc/hosts に追加
    extra_hosts:
      - "host.docker.internal:host-gateway"  # ホストマシンへのアクセス
      - "api.external.com:192.168.1.100"     # 外部サービスの解決
```

### 5.5 ネットワーク分離パターン

```
+------------------------------------------------------------------+
|              マイクロサービスのネットワーク分離設計                    |
+------------------------------------------------------------------+
|                                                                  |
|  [Internet]                                                      |
|      |                                                           |
|      v                                                           |
|  +--- public ネットワーク ---+                                    |
|  |  [nginx/traefik]         |                                    |
|  |    (リバースプロキシ)     |                                    |
|  +-----|-------|-------------+                                    |
|        |       |                                                 |
|        v       v                                                 |
|  +--- frontend ネットワーク ---+                                  |
|  |  [web-app]    [admin-app]  |                                  |
|  +-----|-------|-----|--------+                                   |
|        |       |     |                                           |
|        v       v     v                                           |
|  +--- api ネットワーク ------+                                    |
|  |  [api-gateway]           |                                    |
|  |    |         |           |                                    |
|  |    v         v           |                                    |
|  |  [user-svc] [order-svc]  |                                    |
|  +----|---------|------------+                                    |
|       |         |                                                |
|       v         v                                                |
|  +--- data ネットワーク (internal) ---+                           |
|  |  [postgres]  [redis]  [rabbitmq]  |                           |
|  |  ※ 外部から直接アクセス不可       |                            |
|  +-----------------------------------+                           |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 6. secrets と configs

### 6.1 シークレットの定義と利用

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:16-alpine
    secrets:
      - db_password
      - db_user
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      POSTGRES_USER_FILE: /run/secrets/db_user

  web:
    build: .
    secrets:
      - source: db_password        # シークレット名
        target: database_password   # コンテナ内のファイル名
        uid: "1000"                 # ファイルの所有者 UID
        gid: "1000"                 # ファイルの所有者 GID
        mode: 0440                  # ファイルのパーミッション

secrets:
  # ファイルベースのシークレット
  db_password:
    file: ./secrets/db_password.txt

  # 環境変数ベースのシークレット
  db_user:
    environment: POSTGRES_USER      # ホストの環境変数から取得
```

```
+------------------------------------------------------------------+
|              シークレットの仕組み                                   |
+------------------------------------------------------------------+
|                                                                  |
|  ホスト                          コンテナ                         |
|  ./secrets/db_password.txt  -->  /run/secrets/db_password        |
|                                                                  |
|  ※ シークレットは tmpfs にマウントされる                          |
|  ※ 環境変数と異なり docker inspect で値が見えない                 |
|  ※ ファイルとしてアクセスするためアプリ側の対応が必要              |
|                                                                  |
|  PostgreSQL の _FILE サフィックス対応:                             |
|    POSTGRES_PASSWORD_FILE=/run/secrets/db_password               |
|    → ファイルの内容を POSTGRES_PASSWORD として認識                 |
|                                                                  |
+------------------------------------------------------------------+
```

### 6.2 configs の定義と利用

```yaml
services:
  nginx:
    image: nginx:alpine
    configs:
      - source: nginx_conf
        target: /etc/nginx/nginx.conf    # コンテナ内のパス
        uid: "0"
        gid: "0"
        mode: 0444

  prometheus:
    image: prom/prometheus:v2.51.0
    configs:
      - source: prometheus_conf
        target: /etc/prometheus/prometheus.yml

configs:
  nginx_conf:
    file: ./nginx/nginx.conf

  prometheus_conf:
    file: ./prometheus/prometheus.yml
```

---

## 7. 基本コマンド

### 7.1 よく使うコマンド

```bash
# 起動
docker compose up -d          # バックグラウンドで全サービス起動
docker compose up web db      # 指定サービスのみ起動
docker compose up --build     # ビルドしてから起動

# 停止
docker compose stop           # サービス停止 (コンテナ保持)
docker compose down           # サービス停止 + コンテナ削除
docker compose down -v        # + ボリュームも削除
docker compose down --rmi all # + イメージも削除

# 状態確認
docker compose ps             # サービス一覧
docker compose logs           # ログ表示
docker compose logs -f web    # 特定サービスのログをフォロー
docker compose top            # プロセス一覧

# 実行
docker compose exec web bash  # 起動中コンテナでコマンド実行
docker compose run web npm test # 新しいコンテナでコマンド実行

# その他
docker compose config         # 設定の検証 & 展開結果表示
docker compose pull           # イメージを最新に更新
docker compose build          # サービスのビルドのみ
```

### 7.2 コマンドフロー図

```
+------------------------------------------------------------------+
|              docker compose コマンドフロー                          |
+------------------------------------------------------------------+
|                                                                  |
|  docker compose up -d                                            |
|    |                                                             |
|    +-- ネットワーク作成 (なければ)                                 |
|    +-- ボリューム作成 (なければ)                                   |
|    +-- イメージ pull/build (なければ)                              |
|    +-- コンテナ作成 & 起動                                        |
|    +-- ヘルスチェック待機 (設定されていれば)                        |
|                                                                  |
|  docker compose down                                             |
|    |                                                             |
|    +-- コンテナ停止                                               |
|    +-- コンテナ削除                                               |
|    +-- ネットワーク削除                                           |
|    +-- (ボリュームは保持。-v で削除)                               |
|                                                                  |
+------------------------------------------------------------------+
```

### 7.3 コマンドの実行パターン比較

```
+------------------------------------------------------------------+
|              exec vs run の違い                                    |
+------------------------------------------------------------------+
|                                                                  |
|  docker compose exec web bash                                    |
|  → 起動中のコンテナに接続                                         |
|  → コンテナの環境変数・ネットワークをそのまま使用                  |
|  → コンテナが停止すると使えない                                   |
|  → ファイル変更は永続（コンテナ再作成まで）                       |
|                                                                  |
|  docker compose run web npm test                                 |
|  → 新しいコンテナを作成して実行                                   |
|  → ポートマッピングはデフォルトで無効 (--service-ports で有効化)   |
|  → depends_on のサービスも起動される                              |
|  → 終了後にコンテナが残る (--rm で自動削除)                       |
|                                                                  |
|  使い分けガイド:                                                  |
|  +---------------------------+-----------------------------------+|
|  | ユースケース              | コマンド                          ||
|  +---------------------------+-----------------------------------+|
|  | デバッグ (シェル接続)     | exec web bash                     ||
|  | テスト実行                | run --rm web npm test             ||
|  | マイグレーション          | run --rm web npm run migrate      ||
|  | 一回限りのスクリプト      | run --rm web node script.js       ||
|  | データベースクライアント  | exec db psql -U postgres          ||
|  +---------------------------+-----------------------------------+|
|                                                                  |
+------------------------------------------------------------------+
```

---

## 8. 実践的な構成例

### 8.1 Web アプリケーション + DB + Redis

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@db:5432/myapp
      REDIS_URL: redis://redis:6379
    volumes:
      - .:/app
      - node_modules:/app/node_modules
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: myapp
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
  node_modules:
```

### 8.2 Django + PostgreSQL + Nginx

```yaml
# docker-compose.yml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx/default.conf:/etc/nginx/conf.d/default.conf:ro
      - static_volume:/app/staticfiles:ro
      - media_volume:/app/media:ro
    depends_on:
      web:
        condition: service_healthy
    restart: unless-stopped

  web:
    build:
      context: .
      target: production
    command: gunicorn config.wsgi:application --bind 0.0.0.0:8000 --workers 4
    volumes:
      - static_volume:/app/staticfiles
      - media_volume:/app/media
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db:5432/django_app
      DJANGO_SETTINGS_MODULE: config.settings.production
      SECRET_KEY: ${DJANGO_SECRET_KEY}
      ALLOWED_HOSTS: ${ALLOWED_HOSTS:-localhost}
    expose:
      - "8000"
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    environment:
      POSTGRES_DB: django_app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d django_app"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # マイグレーション（起動時に一度だけ実行）
  migrate:
    build:
      context: .
      target: production
    command: python manage.py migrate --noinput
    environment:
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@db:5432/django_app
    depends_on:
      db:
        condition: service_healthy

  # 静的ファイル収集（起動時に一度だけ実行）
  collectstatic:
    build:
      context: .
      target: production
    command: python manage.py collectstatic --noinput
    volumes:
      - static_volume:/app/staticfiles
    depends_on:
      migrate:
        condition: service_completed_successfully

volumes:
  pgdata:
  static_volume:
  media_volume:
```

### 8.3 マイクロサービス構成

```yaml
# docker-compose.yml
services:
  # API ゲートウェイ
  gateway:
    build: ./gateway
    ports:
      - "8080:8080"
    environment:
      USER_SERVICE_URL: http://user-service:3001
      ORDER_SERVICE_URL: http://order-service:3002
      PRODUCT_SERVICE_URL: http://product-service:3003
    networks:
      - frontend
      - backend
    depends_on:
      user-service:
        condition: service_healthy
      order-service:
        condition: service_healthy
      product-service:
        condition: service_healthy
    restart: unless-stopped

  # ユーザーサービス
  user-service:
    build: ./services/user
    expose:
      - "3001"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@user-db:5432/users
      REDIS_URL: redis://redis:6379/0
    networks:
      - backend
      - data
    depends_on:
      user-db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3001/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # 注文サービス
  order-service:
    build: ./services/order
    expose:
      - "3002"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@order-db:5432/orders
      RABBITMQ_URL: amqp://guest:guest@rabbitmq:5672
    networks:
      - backend
      - data
    depends_on:
      order-db:
        condition: service_healthy
      rabbitmq:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3002/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # 商品サービス
  product-service:
    build: ./services/product
    expose:
      - "3003"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@product-db:5432/products
    networks:
      - backend
      - data
    depends_on:
      product-db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3003/health"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # データベース群
  user-db:
    image: postgres:16-alpine
    volumes:
      - user_pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: users
      POSTGRES_PASSWORD: postgres
    networks:
      - data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  order-db:
    image: postgres:16-alpine
    volumes:
      - order_pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: orders
      POSTGRES_PASSWORD: postgres
    networks:
      - data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  product-db:
    image: postgres:16-alpine
    volumes:
      - product_pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_DB: products
      POSTGRES_PASSWORD: postgres
    networks:
      - data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # メッセージキュー
  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "15672:15672"       # 管理画面（開発用）
    expose:
      - "5672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - data
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "-q", "ping"]
      interval: 10s
      timeout: 10s
      retries: 5
    restart: unless-stopped

  # キャッシュ
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    restart: unless-stopped

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
  data:
    driver: bridge
    internal: true          # 外部アクセス不可

volumes:
  user_pgdata:
  order_pgdata:
  product_pgdata:
  rabbitmq_data:
  redis_data:
```

### 8.4 環境変数と .env ファイルの管理

```bash
# .env (docker compose が自動読み込み)
COMPOSE_PROJECT_NAME=myapp
DB_PASSWORD=secure_password_here
DJANGO_SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,example.com
NODE_ENV=production
```

```yaml
# docker-compose.yml での環境変数の展開
services:
  web:
    image: myapp:${APP_VERSION:-latest}     # デフォルト値付き
    environment:
      DB_HOST: ${DB_HOST:?DB_HOST is required}  # 未設定ならエラー
      DB_PORT: ${DB_PORT:-5432}                 # 未設定ならデフォルト
      NODE_ENV: ${NODE_ENV}                     # .env から読み込み
```

```
+------------------------------------------------------------------+
|              環境変数の優先順位 (高 → 低)                           |
+------------------------------------------------------------------+
|                                                                  |
|  1. docker compose run -e で渡した値                              |
|  2. シェルの環境変数 (export した値)                               |
|  3. compose.yml の environment セクション                          |
|  4. --env-file で指定したファイル                                  |
|  5. compose.yml の env_file で指定したファイル                      |
|  6. Dockerfile の ENV 命令                                        |
|                                                                  |
|  ※ .env ファイルは compose.yml 内の変数展開 (${VAR}) に使用       |
|  ※ env_file はコンテナの環境変数に直接設定                        |
|  ※ .env と env_file は別物であることに注意                        |
|                                                                  |
+------------------------------------------------------------------+
```

---

## アンチパターン

### アンチパターン 1: latest タグの使用

```yaml
# NG: バージョン未固定
services:
  db:
    image: postgres:latest     # どのバージョンが来るかわからない
  redis:
    image: redis               # タグ省略 = latest

# OK: バージョンを明示的に固定
services:
  db:
    image: postgres:16-alpine  # メジャーバージョン + バリアント
  redis:
    image: redis:7-alpine
```

**問題点**: `latest` はイメージ更新のたびに異なるバージョンが pull される可能性があり、チームメンバー間や CI/CD と環境差異が生じる。PostgreSQL のメジャーバージョンアップは破壊的変更を含むことが多く、意図しないアップグレードでデータ破損のリスクもある。

### アンチパターン 2: ホストネットワークモードの乱用

```yaml
# NG: ホストネットワークで全ポートを露出
services:
  db:
    image: postgres:16
    network_mode: host         # 全ポートがホストに直接公開

# OK: 必要なポートだけを公開
services:
  db:
    image: postgres:16
    ports:
      - "127.0.0.1:5432:5432" # localhost のみに公開
```

**問題点**: `network_mode: host` はコンテナのネットワーク分離を完全に無効化する。DB やキャッシュサーバーが外部ネットワークから直接アクセス可能になり、セキュリティリスクが増大する。

### アンチパターン 3: depends_on を condition なしで使う

```yaml
# NG: コンテナ起動のみ確認（サービス準備完了を待たない）
services:
  web:
    build: .
    depends_on:
      - db              # db コンテナが起動したら即 web を起動
  db:
    image: postgres:16-alpine
    # ヘルスチェックなし
# -> PostgreSQL が接続受付前に web が起動し、接続エラーが発生

# OK: ヘルスチェック + condition で準備完了を待つ
services:
  web:
    build: .
    depends_on:
      db:
        condition: service_healthy
  db:
    image: postgres:16-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
```

**問題点**: `depends_on` のデフォルト (`service_started`) はコンテナの起動のみを確認する。データベースが実際にクエリを受け付けられる状態になるまでには数秒かかるため、アプリケーションが起動直後に接続エラーを起こす。アプリ側のリトライロジックだけに頼るのではなく、Compose のヘルスチェック連携を活用すべきである。

### アンチパターン 4: ボリュームのバックアップを考慮しない

```yaml
# NG: バックアップ手段のない名前付きボリューム
services:
  db:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
# -> docker compose down -v でデータ完全消失

# OK: バックアップスクリプトを用意
# backup.sh
# docker compose exec db pg_dump -U postgres myapp > backup_$(date +%Y%m%d).sql

# より安全な構成: バックアップ用のサービスを追加
services:
  db:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
  backup:
    image: postgres:16-alpine
    volumes:
      - ./backups:/backups
    entrypoint: /bin/sh
    command: >
      -c "pg_dump -h db -U postgres myapp > /backups/backup_$$(date +%Y%m%d_%H%M%S).sql"
    depends_on:
      db:
        condition: service_healthy
    profiles:
      - backup              # docker compose --profile backup run backup で手動実行
volumes:
  pgdata:
```

**問題点**: 名前付きボリュームはコンテナのライフサイクルとは独立して存在するが、`docker compose down -v` や `docker volume prune` で削除される。本番データや重要なデータを扱う場合は、定期的なバックアップの仕組みを必ず用意する。

### アンチパターン 5: 環境変数でシークレットを管理する

```yaml
# NG: パスワードが compose ファイルにハードコード
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: my_secret_password  # Git にコミットされる
  web:
    build: .
    environment:
      DB_PASSWORD: my_secret_password        # docker inspect で見える

# OK: .env ファイル + secrets を活用
services:
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    secrets:
      - db_password
  web:
    build: .
    secrets:
      - db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt   # .gitignore に追加
```

**問題点**: 環境変数にシークレットをハードコードすると Git リポジトリにコミットされ、`docker inspect` でも値が見える。`.env` ファイルを使う場合は `.gitignore` に追加し、シークレットが必要な場合は Docker の secrets 機能を使うべきである。

---

## FAQ

### Q1: docker-compose と docker compose (ハイフンなし) の違いは何ですか？

**A**: `docker-compose` は Python 製の Compose V1 (スタンドアロンバイナリ)、`docker compose` は Go 製の Compose V2 (Docker CLI プラグイン)。V1 は 2023 年 6 月に EOL を迎えており、現在は V2 の `docker compose` を使うべき。機能的にはほぼ互換だが、V2 の方が高速で、`docker compose` サブコマンドとして Docker CLI に統合されている。

### Q2: depends_on を設定すればサービスの起動順序は保証されますか？

**A**: `depends_on` はコンテナの起動順序のみを制御し、サービスが「準備完了」になったことは保証しない。例えば PostgreSQL コンテナが起動してから実際に接続を受け付けるまでには数秒かかる。`depends_on` に `condition: service_healthy` を指定し、ヘルスチェックと組み合わせることで、サービスが実際に利用可能になるまで待機できる。

### Q3: 開発用と本番用で Compose ファイルを分けるべきですか？

**A**: はい。`docker-compose.yml` (共通/開発用) と `docker-compose.prod.yml` (本番用オーバーライド) に分けるのが一般的。`docker compose -f docker-compose.yml -f docker-compose.prod.yml up` のように複数ファイルを指定すると、後のファイルで前のファイルの設定を上書きできる。Compose V2 では `compose.yml` と `compose.override.yml` を自動的にマージする機能もある。

### Q4: Compose で特定のサービスだけを再ビルドして更新するには？

**A**: `docker compose up -d --build web` のようにサービス名を指定する。`--build` フラグを付けると、起動前にイメージを再ビルドする。`--no-deps` を追加すると、依存サービスの再起動を防げる: `docker compose up -d --build --no-deps web`。イメージの再ビルドだけを行いたい場合は `docker compose build web` を使う。

### Q5: macOS で Compose のバインドマウントが遅いのですが、対策はありますか？

**A**: macOS (Docker Desktop for Mac) ではバインドマウントの I/O パフォーマンスがネイティブに比べて遅い。以下の対策がある:
- **VirtioFS** を使う: Docker Desktop の設定で `VirtioFS` を有効化する（デフォルトで有効な場合が多い）。gRPC FUSE より大幅に高速
- **ボリュームの分離**: `node_modules` や `vendor` などの依存関係ディレクトリは名前付きボリュームに分離する
- **Synchronized file shares** (Docker Desktop 4.27+): ファイル同期を最適化する機能

```yaml
# macOS パフォーマンス改善例
services:
  web:
    volumes:
      - .:/app                               # ソースコード
      - node_modules:/app/node_modules       # 名前付きボリュームで分離
volumes:
  node_modules:
```

### Q6: docker compose watch とは何ですか？

**A**: Docker Compose v2.22+ で導入された機能で、ファイル変更を検知して自動的にアクションを実行する。ホットリロードやオートリビルドを Compose レベルで管理できる。

```yaml
services:
  web:
    build: .
    develop:
      watch:
        - action: sync           # ファイルをコンテナに同期
          path: ./src
          target: /app/src
        - action: rebuild         # イメージを再ビルド
          path: package.json
        - action: sync+restart    # 同期してコンテナ再起動
          path: ./config
          target: /app/config
```

`docker compose watch` で起動すると、ファイル変更に応じて `sync`（ファイル同期）、`rebuild`（イメージ再ビルド + コンテナ再作成）、`sync+restart`（ファイル同期 + コンテナ再起動）が自動的に実行される。

### Q7: 複数の Compose ファイルを使い分けるにはどうすればよいですか？

**A**: 複数の Compose ファイルをマージする方法がいくつかある:

```bash
# 1. -f フラグで明示的にファイルを指定（後のファイルで上書き）
docker compose -f compose.yml -f compose.prod.yml up -d

# 2. compose.override.yml は自動的にマージされる
# compose.yml         ← ベース設定
# compose.override.yml ← 開発用の上書き設定（自動マージ）

# 3. COMPOSE_FILE 環境変数
export COMPOSE_FILE=compose.yml:compose.prod.yml
docker compose up -d

# 4. include (Compose v2.20+)
# compose.yml
# include:
#   - path: ./monitoring/compose.yml
#   - path: ./logging/compose.yml
```

```yaml
# compose.yml (ベース)
services:
  web:
    build: .
    ports:
      - "3000:3000"

# compose.override.yml (開発用 - 自動マージ)
services:
  web:
    volumes:
      - .:/app
    environment:
      NODE_ENV: development

# compose.prod.yml (本番用 - 明示指定)
services:
  web:
    restart: always
    environment:
      NODE_ENV: production
    deploy:
      resources:
        limits:
          memory: 512M
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| Compose ファイル | version キーは省略。Compose Specification 準拠が現在の標準 |
| services | コンテナの定義。image / build / ports / environment / volumes が基本 |
| volumes | 名前付き Volume を推奨。DB データの永続化に必須 |
| networks | デフォルトでサービス名による DNS 解決。分離が必要なら明示定義 |
| secrets | 機密情報はシークレットで管理。環境変数よりも安全 |
| configs | 設定ファイルのマウント。nginx.conf や prometheus.yml 等 |
| depends_on | `condition: service_healthy` でヘルスチェック連携が重要 |
| healthcheck | サービス準備完了の検知に必須。DB は pg_isready、HTTP は curl |
| リソース制限 | deploy.resources で CPU/メモリの上限と予約を設定 |
| ログ管理 | logging で max-size/max-file を設定してディスク枯渇を防止 |
| コマンド | `up -d` / `down` / `logs -f` / `exec` が日常の基本操作 |
| イメージタグ | `latest` を避け、メジャーバージョン + バリアントを明示 |
| V1 vs V2 | `docker compose` (V2, CLI プラグイン) を使用。V1 は EOL |
| 環境変数 | .env ファイルで管理。シークレットのハードコードは避ける |
| ファイル分割 | compose.override.yml で開発/本番の設定を分離 |

## 次に読むべきガイド

- [Compose 応用](./01-compose-advanced.md) -- プロファイル、depends_on、healthcheck、環境変数の高度な使い方
- [Compose 開発ワークフロー](./02-development-workflow.md) -- ホットリロード、デバッグ、CI 統合
- [ローカルサービスの Docker 化](../../development-environment-setup/docs/02-docker-dev/02-local-services.md) -- DB / Redis / MailHog の実践的な Compose 構成

## 参考文献

1. **Docker Compose 公式リファレンス** -- https://docs.docker.com/compose/compose-file/ -- Compose ファイル仕様の完全なリファレンス
2. **Compose Specification** -- https://compose-spec.io/ -- Docker Compose の公式仕様 (GitHub)
3. **Docker 公式チュートリアル** -- https://docs.docker.com/compose/gettingstarted/ -- Compose のクイックスタートガイド
4. **Docker Compose Networking** -- https://docs.docker.com/compose/networking/ -- Compose のネットワーク設定の詳細ガイド
5. **Docker Compose Environment Variables** -- https://docs.docker.com/compose/environment-variables/ -- 環境変数の設定方法と優先順位
6. **Docker Compose Watch** -- https://docs.docker.com/compose/file-watch/ -- ファイル監視による自動同期・リビルド機能のドキュメント
7. **Awesome Docker Compose** -- https://github.com/docker/awesome-compose -- Docker 公式の Compose サンプル集
