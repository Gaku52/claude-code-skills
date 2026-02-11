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

### 1.2 Compose ファイルのバージョン

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

---

## 6. 基本コマンド

### 6.1 よく使うコマンド

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

### 6.2 コマンドフロー図

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

---

## 7. 実践的な構成例

### 7.1 Web アプリ + DB + Redis

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

---

## FAQ

### Q1: docker-compose と docker compose (ハイフンなし) の違いは何ですか？

**A**: `docker-compose` は Python 製の Compose V1 (スタンドアロンバイナリ)、`docker compose` は Go 製の Compose V2 (Docker CLI プラグイン)。V1 は 2023 年 6 月に EOL を迎えており、現在は V2 の `docker compose` を使うべき。機能的にはほぼ互換だが、V2 の方が高速で、`docker compose` サブコマンドとして Docker CLI に統合されている。

### Q2: depends_on を設定すればサービスの起動順序は保証されますか？

**A**: `depends_on` はコンテナの起動順序のみを制御し、サービスが「準備完了」になったことは保証しない。例えば PostgreSQL コンテナが起動してから実際に接続を受け付けるまでには数秒かかる。`depends_on` に `condition: service_healthy` を指定し、ヘルスチェックと組み合わせることで、サービスが実際に利用可能になるまで待機できる。

### Q3: 開発用と本番用で Compose ファイルを分けるべきですか？

**A**: はい。`docker-compose.yml` (共通/開発用) と `docker-compose.prod.yml` (本番用オーバーライド) に分けるのが一般的。`docker compose -f docker-compose.yml -f docker-compose.prod.yml up` のように複数ファイルを指定すると、後のファイルで前のファイルの設定を上書きできる。Compose V2 では `compose.yml` と `compose.override.yml` を自動的にマージする機能もある。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Compose ファイル | version キーは省略。Compose Specification 準拠が現在の標準 |
| services | コンテナの定義。image / build / ports / environment / volumes が基本 |
| volumes | 名前付き Volume を推奨。DB データの永続化に必須 |
| networks | デフォルトでサービス名による DNS 解決。分離が必要なら明示定義 |
| depends_on | `condition: service_healthy` でヘルスチェック連携が重要 |
| コマンド | `up -d` / `down` / `logs -f` / `exec` が日常の基本操作 |
| イメージタグ | `latest` を避け、メジャーバージョン + バリアントを明示 |
| V1 vs V2 | `docker compose` (V2, CLI プラグイン) を使用。V1 は EOL |

## 次に読むべきガイド

- [Compose 応用](./01-compose-advanced.md) -- プロファイル、depends_on、healthcheck、環境変数の高度な使い方
- [Compose 開発ワークフロー](./02-development-workflow.md) -- ホットリロード、デバッグ、CI 統合
- [ローカルサービスの Docker 化](../../development-environment-setup/docs/02-docker-dev/02-local-services.md) -- DB / Redis / MailHog の実践的な Compose 構成

## 参考文献

1. **Docker Compose 公式リファレンス** -- https://docs.docker.com/compose/compose-file/ -- Compose ファイル仕様の完全なリファレンス
2. **Compose Specification** -- https://compose-spec.io/ -- Docker Compose の公式仕様 (GitHub)
3. **Docker 公式チュートリアル** -- https://docs.docker.com/compose/gettingstarted/ -- Compose のクイックスタートガイド
