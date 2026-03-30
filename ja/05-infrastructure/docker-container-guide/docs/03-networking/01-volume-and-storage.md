# ボリュームとストレージ

> コンテナのライフサイクルを超えてデータを永続化するための3つのマウント方式とストレージドライバーの仕組みを理解する。

---

## この章で学ぶこと

1. **Named Volume / Bind Mount / tmpfs の違いと使い分け**を理解する
2. **ボリュームのライフサイクル管理**（作成・バックアップ・移行・削除）を習得する
3. **ストレージドライバーの仕組み**とパフォーマンス特性を把握する
4. **本番環境でのストレージ設計パターン**を学ぶ
5. **各種データベースに最適なボリューム設定**を実践する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Dockerネットワーク](./00-docker-networking.md) の内容を理解していること

---

## 1. なぜデータ永続化が必要か

Dockerコンテナはイミュータブルに設計されている。コンテナの書き込み可能レイヤー（writable layer）はコンテナ削除と同時に消失する。データベースのデータ、アップロードされたファイル、設定ファイルなど、コンテナのライフサイクルを超えて保持すべきデータにはボリュームが必要。

### コンテナのレイヤー構造

```
┌──────────────────────────────────────────────┐
│         Container (書き込み可能レイヤー)        │
│  ┌────────────────────────────────────────┐ │
│  │  Thin R/W Layer (CoW: Copy-on-Write)  │ │ ← コンテナ削除で消失
│  └────────────────────────────────────────┘ │
├──────────────────────────────────────────────┤
│         Image Layers (読み取り専用)            │
│  ┌────────────────────────────────────────┐ │
│  │  Layer 4: COPY app.js /app/           │ │
│  ├────────────────────────────────────────┤ │
│  │  Layer 3: RUN npm install             │ │
│  ├────────────────────────────────────────┤ │
│  │  Layer 2: RUN apt-get install nodejs  │ │
│  ├────────────────────────────────────────┤ │
│  │  Layer 1: Ubuntu 22.04 base           │ │
│  └────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

### データ永続化の3方式

```
┌──────────────────────────────────────────────────────────┐
│                    Docker Host                           │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │              Container                        │       │
│  │                                               │       │
│  │  /app/data ──┐  /app/config ──┐  /tmp ──┐   │       │
│  └──────────────┼────────────────┼─────────┼───┘       │
│                 │                │         │             │
│           ┌─────▼──────┐  ┌─────▼────┐  ┌─▼────────┐  │
│           │Named Volume│  │Bind Mount│  │  tmpfs   │  │
│           │            │  │          │  │ (RAM)    │  │
│           │/var/lib/   │  │ホストの   │  │メモリ上  │  │
│           │docker/     │  │任意の     │  │ディスク  │  │
│           │volumes/    │  │ディレクトリ│  │書き込み  │  │
│           └────────────┘  └──────────┘  │なし      │  │
│            Docker管理       ユーザー管理  └──────────┘  │
│                                          カーネル管理   │
└──────────────────────────────────────────────────────────┘
```

### 3方式の比較表

| 特性 | Named Volume | Bind Mount | tmpfs |
|------|-------------|------------|-------|
| 保存先 | Docker管理領域 | ホスト上の任意パス | メモリ（RAM） |
| Docker CLIで管理 | 可能 | 不可 | 不可 |
| コンテナ間共有 | 容易 | 可能 | 不可 |
| データ永続化 | コンテナ削除後も残る | コンテナ削除後も残る | コンテナ停止で消失 |
| パフォーマンス | ドライバー依存 | ネイティブ | 最高速 |
| ホストOSへの依存 | 低い | 高い（パス依存） | 低い |
| 本番推奨度 | 高い | 低い（開発向き） | 特殊用途 |
| バックアップ | Docker CLI で可能 | ホストのツールで可能 | 不可 |
| ドライバー変更 | 可能（NFS等） | 不可 | 不可 |

---

## 2. Named Volume

DockerエンジンがManageするボリューム。ホスト上の `/var/lib/docker/volumes/` 配下に保存される。

### コード例1: Named Volumeの基本操作

```bash
# ボリュームの作成
docker volume create my-data

# ボリューム一覧
docker volume ls

# ボリュームのフィルタリング
docker volume ls --filter "driver=local"
docker volume ls --filter "dangling=true"   # 未使用ボリューム

# ボリュームの詳細情報
docker volume inspect my-data
# 出力例:
# [
#     {
#         "CreatedAt": "2025-01-15T10:30:00Z",
#         "Driver": "local",
#         "Labels": {},
#         "Mountpoint": "/var/lib/docker/volumes/my-data/_data",
#         "Name": "my-data",
#         "Options": {},
#         "Scope": "local"
#     }
# ]

# ボリュームをマウントしてコンテナ起動
docker run -d \
  --name postgres-db \
  -v my-data:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=secret \
  postgres:16-alpine

# --mount 構文（推奨: より明示的）
docker run -d \
  --name postgres-db \
  --mount type=volume,source=my-data,target=/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=secret \
  postgres:16-alpine

# コンテナを削除してもボリュームは残る
docker rm -f postgres-db
docker volume ls  # my-data は健在

# 未使用ボリュームの一括削除（注意して使用）
docker volume prune

# 全ての未使用ボリュームを強制削除
docker volume prune -a -f
```

### コード例2: Docker Composeでのボリューム定義

```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: myapp
    volumes:
      - pgdata:/var/lib/postgresql/data       # Named Volume
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro  # Bind Mount (読取専用)
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data

  backup:
    image: alpine:3.19
    volumes:
      - pgdata:/source:ro          # 同一ボリュームを読取専用でマウント
      - ./backups:/backup
    command: >
      sh -c "tar czf /backup/pgdata-$$(date +%Y%m%d).tar.gz -C /source ."

volumes:
  pgdata:
    driver: local
    labels:
      com.example.description: "PostgreSQLデータ"
      com.example.environment: "production"
  redis-data:
    driver: local
```

### ボリュームのラベルとフィルタリング

```bash
# ラベル付きボリュームの作成
docker volume create \
  --label environment=production \
  --label service=postgres \
  prod-pgdata

# ラベルでフィルタリング
docker volume ls --filter "label=environment=production"
docker volume ls --filter "label=service=postgres"
```

---

## 3. Bind Mount

ホストマシン上の任意のディレクトリやファイルをコンテナにマウントする方式。開発時のソースコード同期に多用される。

### コード例3: Bind Mountの活用

```bash
# 基本的なBind Mount（-v 構文）
docker run -d \
  --name dev-server \
  -v /home/user/project/src:/app/src \
  -v /home/user/project/config.yaml:/app/config.yaml:ro \
  my-app:dev

# --mount 構文（より明示的で推奨）
docker run -d \
  --name dev-server \
  --mount type=bind,source=/home/user/project/src,target=/app/src \
  --mount type=bind,source=/home/user/project/config.yaml,target=/app/config.yaml,readonly \
  my-app:dev

# 読み書き権限の制御
# :ro  → 読み取り専用
# :rw  → 読み書き可（デフォルト）

# 存在しないホストパスを指定した場合の挙動
# -v 構文    : 自動でディレクトリが作成される（意図せぬ空ディレクトリ生成の危険）
# --mount 構文: エラーになる（安全）
```

### Bind Mountの開発ワークフロー

```
┌────────────────────────────┐
│       開発マシン            │
│                            │
│  ~/project/src/ ◄──── エディタで編集
│       │                    │
│  ┌────▼──────────────┐    │
│  │    Container       │    │
│  │  /app/src/ (bind)  │    │
│  │       │            │    │
│  │  ┌────▼────┐      │    │
│  │  │ nodemon │      │    │  ← ファイル変更を検知して
│  │  │ (watch) │      │    │    自動リロード
│  │  └─────────┘      │    │
│  └────────────────────┘    │
└────────────────────────────┘
```

### Docker Compose での Bind Mount パターン

```yaml
services:
  app:
    build: .
    volumes:
      # ソースコード（読み書き）
      - ./src:/app/src

      # 設定ファイル（読み取り専用）
      - ./config:/app/config:ro

      # 単一ファイルのマウント
      - ./nginx.conf:/etc/nginx/nginx.conf:ro

      # node_modules は Named Volume で分離
      - node_modules:/app/node_modules

      # 長文構文
      - type: bind
        source: ./data
        target: /app/data
        read_only: false

volumes:
  node_modules:
```

### Bind Mount の SELinux 対応 (RHEL/CentOS)

```bash
# SELinux が有効な環境でのBind Mount
# :z  → 共有ラベルを設定（複数コンテナで共有可能）
# :Z  → プライベートラベルを設定（単一コンテナ専用）
docker run -d \
  -v /data/app:/app:z \
  my-app:latest

# Docker Compose での指定
# volumes:
#   - ./data:/app/data:z
```

---

## 4. tmpfs マウント

メモリ上にのみ存在する一時ファイルシステム。ディスクに書き込まれないため、機密データの一時保存やパフォーマンスが重要な一時ファイルに適する。

### コード例4: tmpfsの使用

```bash
# tmpfsマウント
docker run -d \
  --name secure-app \
  --tmpfs /tmp:rw,noexec,nosuid,size=100m \
  --mount type=tmpfs,destination=/run/secrets,tmpfs-size=10m,tmpfs-mode=0700 \
  my-app

# Docker Composeでの指定
```

```yaml
# docker-compose.yml
services:
  app:
    image: my-app
    tmpfs:
      - /tmp:size=100m,mode=1777
    volumes:
      - type: tmpfs
        target: /run/secrets
        tmpfs:
          size: 10485760  # 10MB
          mode: 0700

  # テスト用DB（永続化不要 → tmpfsで高速化）
  db-test:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: test
    tmpfs:
      - /var/lib/postgresql/data:size=512m
    # → ディスクI/Oなしでテスト用DBが動作
```

### tmpfs の活用シーン

| シーン | 理由 |
|--------|------|
| テスト用DB | 永続化不要。メモリ上で高速にテスト実行 |
| セッションストア | 再起動時にリセットされても問題ない一時データ |
| 一時ファイル処理 | 画像変換やPDF生成の中間ファイル |
| シークレット保存 | ディスクに書き込まれないため安全 |
| CI/CDパイプライン | テスト実行の高速化 |

### 用途別マウント方式の選定フロー

```
データを永続化する必要がある？
    │
    ├── Yes ──► ホスト側のパスを指定する必要がある？
    │               │
    │               ├── Yes ──► Bind Mount
    │               │           (開発時のソースコード同期など)
    │               │
    │               └── No ───► Named Volume
    │                           (DB, アプリデータなど)
    │
    └── No ───► セキュリティ/パフォーマンスが重要？
                    │
                    ├── Yes ──► tmpfs
                    │           (一時ファイル, シークレット)
                    │
                    └── No ───► コンテナの書き込みレイヤー
                                (ログなど一時的なデータ)
```

---

## 5. ボリュームのバックアップと移行

### コード例5: バックアップ・リストア・移行

```bash
# === バックアップ ===
# 別コンテナからボリュームをtarで圧縮バックアップ
docker run --rm \
  -v my-data:/source:ro \
  -v $(pwd)/backups:/backup \
  alpine:3.19 \
  tar czf /backup/my-data-backup.tar.gz -C /source .

# === リストア ===
# バックアップからボリュームを復元
docker volume create my-data-restored

docker run --rm \
  -v my-data-restored:/target \
  -v $(pwd)/backups:/backup:ro \
  alpine:3.19 \
  tar xzf /backup/my-data-backup.tar.gz -C /target

# === ボリュームの移行（ホスト間） ===
# 1. 送信元でバックアップ
docker run --rm \
  -v my-data:/source:ro \
  alpine:3.19 \
  tar czf - -C /source . | ssh user@remote-host \
  "docker run --rm -i -v my-data:/target alpine:3.19 tar xzf - -C /target"

# === ボリュームのコピー ===
docker volume create my-data-copy

docker run --rm \
  -v my-data:/from:ro \
  -v my-data-copy:/to \
  alpine:3.19 \
  sh -c "cp -av /from/. /to/"
```

### コード例5b: 定期バックアップの自動化

```yaml
# docker-compose.yml - 定期バックアップ設定
services:
  postgres:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  # 定期バックアップコンテナ
  backup:
    image: postgres:16-alpine
    volumes:
      - ./backups:/backups
    environment:
      PGPASSWORD: ${DB_PASSWORD}
    # 毎日3時にバックアップ（cron代替として entrypoint スクリプト）
    entrypoint: >
      sh -c "
        while true; do
          echo \"[$(date)] Starting backup...\"
          pg_dump -h postgres -U postgres myapp | \
            gzip > /backups/myapp-$(date +%Y%m%d-%H%M%S).sql.gz
          echo \"[$(date)] Backup completed.\"
          # 7日以上前のバックアップを削除
          find /backups -name '*.sql.gz' -mtime +7 -delete
          sleep 86400
        done
      "
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  pgdata:
```

```bash
#!/bin/bash
# scripts/backup.sh - 手動バックアップスクリプト

set -euo pipefail

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$BACKUP_DIR"

echo "=== PostgreSQL Backup ==="
docker compose exec -T postgres \
  pg_dump -U postgres myapp | gzip > "${BACKUP_DIR}/postgres-${TIMESTAMP}.sql.gz"
echo "  → ${BACKUP_DIR}/postgres-${TIMESTAMP}.sql.gz"

echo "=== Volume Backup ==="
docker run --rm \
  -v myapp_pgdata:/source:ro \
  -v "$(pwd)/backups":/backup \
  alpine:3.19 \
  tar czf "/backup/pgdata-${TIMESTAMP}.tar.gz" -C /source .
echo "  → ${BACKUP_DIR}/pgdata-${TIMESTAMP}.tar.gz"

echo "=== Redis Backup ==="
docker compose exec redis redis-cli BGSAVE
sleep 2
docker compose cp redis:/data/dump.rdb "${BACKUP_DIR}/redis-${TIMESTAMP}.rdb"
echo "  → ${BACKUP_DIR}/redis-${TIMESTAMP}.rdb"

echo "=== Backup Complete ==="
ls -lh "${BACKUP_DIR}/"*"${TIMESTAMP}"*
```

### コード例6: NFSボリュームドライバー

```bash
# NFSバックエンドのボリュームを作成
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw,nfsvers=4 \
  --opt device=:/exports/data \
  nfs-data
```

```yaml
# docker-compose.yml - NFS ボリューム
volumes:
  shared-data:
    driver: local
    driver_opts:
      type: nfs
      o: "addr=192.168.1.100,rw,nfsvers=4"
      device: ":/exports/data"

  # CIFS/SMB ボリューム（Windows ファイルサーバー）
  smb-data:
    driver: local
    driver_opts:
      type: cifs
      o: "addr=192.168.1.200,username=user,password=pass,file_mode=0777,dir_mode=0777"
      device: "//192.168.1.200/shared"
```

---

## 6. ストレージドライバー

### ストレージドライバーの仕組み（Union File System）

```
┌─────────────────────────────────────────────┐
│           Container (読み書き可能レイヤー)     │
│  ┌────────────────────────────────────────┐ │
│  │  Thin R/W Layer (CoW: Copy-on-Write)  │ │
│  └────────────────────────────────────────┘ │
├─────────────────────────────────────────────┤
│           Image Layers (読み取り専用)        │
│  ┌────────────────────────────────────────┐ │
│  │  Layer 4: COPY app.js /app/           │ │
│  ├────────────────────────────────────────┤ │
│  │  Layer 3: RUN npm install             │ │
│  ├────────────────────────────────────────┤ │
│  │  Layer 2: RUN apt-get install nodejs  │ │
│  ├────────────────────────────────────────┤ │
│  │  Layer 1: Ubuntu 22.04 base           │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### Copy-on-Write (CoW) の仕組み

```
読み取り時:
  アプリが /app/config.json を読む
    → R/W レイヤーにファイルがない
    → 下位レイヤー (Layer 4) から読む
    → ファイルが見つかった → 返す

書き込み時 (Copy-on-Write):
  アプリが /app/config.json を変更する
    1. 下位レイヤーからファイルを R/W レイヤーにコピー
    2. R/W レイヤー上のコピーを変更
    3. 以降の読み取りは R/W レイヤーのコピーを返す
    ※ 元のレイヤーのファイルは変更されない
```

### ストレージドライバーの比較表

| ドライバー | バッキングFS | 特徴 | 推奨環境 |
|-----------|-------------|------|---------|
| overlay2 | xfs, ext4 | 現在のデフォルト。安定・高速 | 全環境（推奨） |
| btrfs | btrfs | スナップショット活用 | btrfs利用環境 |
| zfs | zfs | スナップショット・圧縮 | zfs利用環境 |
| devicemapper | direct-lvm | ブロックレベル操作 | RHEL/CentOS（非推奨） |
| vfs | 全FS | CoWなし（コピー）。最も遅い | テスト用途のみ |

### コード例7: ストレージドライバーの確認と設定

```bash
# 現在のストレージドライバーを確認
docker info | grep "Storage Driver"
# 出力例: Storage Driver: overlay2

# ストレージ使用状況を確認
docker system df
# 出力例:
# TYPE            TOTAL   ACTIVE  SIZE      RECLAIMABLE
# Images          15      5       3.2GB     1.8GB (56%)
# Containers      8       3       256MB     128MB (50%)
# Local Volumes   12      4       5.1GB     3.2GB (62%)
# Build Cache     45      0       890MB     890MB (100%)

# 詳細表示
docker system df -v

# 不要データの一括クリーンアップ
docker system prune -a --volumes
# WARNING: ボリュームも含めて全削除される
```

### ストレージドライバーの変更

```json
// /etc/docker/daemon.json
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true",
    "overlay2.size=20G"
  ]
}
```

```bash
# 設定変更後にDockerデーモンを再起動
sudo systemctl restart docker

# 変更の確認
docker info | grep "Storage Driver"
```

---

## 7. 各種データベースのボリューム設定

### PostgreSQL

```yaml
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: myapp
      # パフォーマンスチューニング
      POSTGRES_INITDB_ARGS: "--data-checksums"
    volumes:
      - pgdata:/var/lib/postgresql/data
      # 初期化スクリプト
      - ./initdb:/docker-entrypoint-initdb.d:ro
      # カスタム設定
      - ./postgresql.conf:/etc/postgresql/postgresql.conf:ro
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    shm_size: '256m'    # PostgreSQL は共有メモリを多用
    deploy:
      resources:
        limits:
          memory: 1G

volumes:
  pgdata:
```

### MySQL / MariaDB

```yaml
services:
  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: ${DB_ROOT_PASSWORD}
      MYSQL_DATABASE: myapp
      MYSQL_USER: app_user
      MYSQL_PASSWORD: ${DB_PASSWORD}
    volumes:
      - mysql-data:/var/lib/mysql
      - ./my.cnf:/etc/mysql/conf.d/my.cnf:ro
      - ./initdb:/docker-entrypoint-initdb.d:ro
    deploy:
      resources:
        limits:
          memory: 1G

volumes:
  mysql-data:
```

### MongoDB

```yaml
services:
  mongodb:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
    volumes:
      - mongo-data:/data/db
      - mongo-config:/data/configdb
      # 初期化スクリプト
      - ./mongo-init:/docker-entrypoint-initdb.d:ro
    command: mongod --wiredTigerCacheSizeGB 0.5

volumes:
  mongo-data:
  mongo-config:
```

### Elasticsearch

```yaml
services:
  elasticsearch:
    image: elasticsearch:8.12.0
    environment:
      - discovery.type=single-node
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
      - xpack.security.enabled=false
    volumes:
      - es-data:/usr/share/elasticsearch/data
    ulimits:
      memlock:
        soft: -1
        hard: -1
      nofile:
        soft: 65536
        hard: 65536

volumes:
  es-data:
```

---

## 8. パフォーマンス最適化

### コード例8: macOSでのBind Mountパフォーマンス改善

```yaml
# docker-compose.yml
# macOSではBind Mountが遅い問題の対策
services:
  app:
    build: .
    volumes:
      # ソースコードはバインドマウント
      - ./src:/app/src

      # node_modules はNamed Volumeで管理（Bind Mountより高速）
      - node_modules:/app/node_modules

      # ビルド成果物も Volume で分離
      - build_cache:/app/.next
      - dist_cache:/app/dist

volumes:
  node_modules:
  build_cache:
  dist_cache:
```

### パフォーマンスベンチマーク（macOS）

```
┌──────────────────────────────────────────────┐
│    macOS でのファイルI/Oパフォーマンス比較      │
├──────────────────────────────────────────────┤
│                                              │
│  操作                 │ Bind Mount │ Volume  │
│  ─────────────────────┼────────────┼─────────│
│  npm install (10000+) │ 120秒      │ 15秒    │
│  tsc コンパイル       │ 30秒       │ 5秒     │
│  Next.js ビルド       │ 90秒       │ 20秒    │
│  ファイル読み取り     │ 遅い       │ 高速    │
│  ファイル書き込み     │ 遅い       │ 高速    │
│                                              │
│  結論: 大量ファイルの操作は Volume が圧倒的   │
│        ソースコードの同期は Bind Mount が必要  │
│        → 「ソースは Bind、依存は Volume」     │
└──────────────────────────────────────────────┘
```

### ボリュームのI/Oパフォーマンスチューニング

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:16-alpine
    volumes:
      # WALログ用の高速ストレージ
      - pgdata:/var/lib/postgresql/data
      - pg-wal:/var/lib/postgresql/data/pg_wal
    # PostgreSQL のI/Oチューニング
    command: >
      postgres
        -c shared_buffers=256MB
        -c effective_cache_size=768MB
        -c wal_buffers=8MB
        -c checkpoint_completion_target=0.9
        -c random_page_cost=1.1

volumes:
  pgdata:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /ssd/postgres/data    # SSD上のディレクトリ
  pg-wal:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /nvme/postgres/wal    # NVMe上のディレクトリ
```

---

## 9. ボリュームの監視とメンテナンス

### ボリュームサイズの監視

```bash
# ボリュームごとのディスク使用量を確認
docker system df -v

# 特定ボリュームのサイズを確認
docker run --rm -v myapp_pgdata:/data alpine du -sh /data

# 全ボリュームのサイズを一覧表示
for vol in $(docker volume ls -q); do
  size=$(docker run --rm -v "${vol}":/data alpine du -sh /data 2>/dev/null | cut -f1)
  echo "${vol}: ${size}"
done
```

### 定期メンテナンススクリプト

```bash
#!/bin/bash
# scripts/volume-maintenance.sh

echo "=== Docker Volume Maintenance ==="
echo "Date: $(date)"

# 1. ディスク使用状況
echo ""
echo "--- Disk Usage ---"
docker system df

# 2. 未使用ボリューム
echo ""
echo "--- Dangling Volumes ---"
docker volume ls --filter "dangling=true"

# 3. 各ボリュームのサイズ
echo ""
echo "--- Volume Sizes ---"
for vol in $(docker volume ls -q); do
  size=$(docker run --rm -v "${vol}":/data alpine du -sh /data 2>/dev/null | cut -f1)
  echo "  ${vol}: ${size}"
done

# 4. 未使用ボリュームのクリーンアップ（確認付き）
echo ""
read -p "Remove dangling volumes? (y/N): " confirm
if [ "$confirm" = "y" ]; then
  docker volume prune -f
  echo "Dangling volumes removed."
fi
```

---

## アンチパターン

### アンチパターン1: コンテナの書き込みレイヤーへの大量書き込み

```bash
# NG: ボリュームなしでDBを運用
docker run -d postgres:16
# → コンテナ削除でデータ全喪失
# → 書き込みレイヤーはCoWオーバーヘッドで低速

# OK: Named Volumeにデータを保存
docker run -d \
  -v pgdata:/var/lib/postgresql/data \
  postgres:16
```

**なぜ問題か**: コンテナの書き込みレイヤーはCopy-on-Write方式で動作するため、大量の書き込みはパフォーマンスが劣化する。またコンテナ削除でデータが消失する。

### アンチパターン2: 本番環境でのBind Mount多用

```yaml
# NG: 本番でホストパスに依存
services:
  app:
    volumes:
      - /opt/myapp/data:/data         # ホストパスへの強い依存
      - /opt/myapp/config:/config     # 別ホストへの移行が困難

# OK: Named Volumeでポータビリティを確保
services:
  app:
    volumes:
      - app-data:/data
      - app-config:/config
volumes:
  app-data:
  app-config:
```

**なぜ問題か**: Bind Mountはホストのディレクトリ構造に依存するため、異なるホストへの移行やスケールアウトが困難になる。Named Volumeはポータブルで、ボリュームドライバーを変更するだけでNFSやクラウドストレージに切り替えられる。

### アンチパターン3: ボリュームの定期バックアップなし

```yaml
# NG: バックアップ未設定のDB
services:
  db:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data
    # バックアップの仕組みがない → ディスク障害でデータ全喪失

# OK: バックアップコンテナを併設
services:
  db:
    image: postgres:16-alpine
    volumes:
      - pgdata:/var/lib/postgresql/data

  backup:
    image: postgres:16-alpine
    volumes:
      - ./backups:/backups
    entrypoint: >
      sh -c "while true; do
        pg_dump -h db -U postgres myapp | gzip > /backups/daily-$$(date +%Y%m%d).sql.gz;
        find /backups -mtime +30 -delete;
        sleep 86400;
      done"
```

**なぜ問題か**: ボリュームのデータもディスク障害やオペレーションミスで失われる可能性がある。定期的なバックアップと復元テストは必須。

### アンチパターン4: docker volume prune の安易な実行

```bash
# NG: 確認なしで全未使用ボリュームを削除
docker volume prune -f
# → 停止中のコンテナのデータも含まれる可能性がある

# OK: まず確認してから削除
docker volume ls --filter "dangling=true"
# 出力を確認してから:
docker volume rm <特定のボリューム名>
```

**なぜ問題か**: `docker volume prune` は「どのコンテナにもマウントされていない」ボリュームを全て削除する。停止中のコンテナが使っていたボリュームも対象になるため、意図せず重要なデータを失う可能性がある。


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない
---

## FAQ

### Q1: Named Volumeのデータはどこに保存されている？

Linux環境では `/var/lib/docker/volumes/<ボリューム名>/_data/` に保存される。Docker Desktop（macOS/Windows）では仮想マシン内のパスとなるため、直接アクセスするには `docker run --rm -v <ボリューム名>:/data alpine ls /data` のようにコンテナ経由でアクセスする。

### Q2: `-v` と `--mount` のどちらを使うべき？

`--mount` を推奨する。理由は以下の通り:
- 構文が明示的で読みやすい
- 存在しないホストパスを指定するとエラーになる（`-v` は自動作成してしまう）
- tmpfsのオプション指定が豊富

```bash
# -v 構文（暗黙的な挙動あり）
docker run -v mydata:/data app

# --mount 構文（明示的で安全）
docker run --mount type=volume,source=mydata,target=/data app
```

### Q3: ボリュームの権限問題（Permission Denied）はどう解決する？

Dockerコンテナ内のプロセスがroot以外のユーザーで実行される場合、ボリューム上のファイル所有権が一致しないことがある。

```dockerfile
# Dockerfileでユーザーを明示的に設定
FROM node:20-alpine
RUN mkdir -p /app/data && chown -R node:node /app/data
USER node
VOLUME /app/data
```

```bash
# 既存ボリュームの権限を修正
docker run --rm -v mydata:/data alpine chown -R 1000:1000 /data
```

### Q4: Named Volume と外部ストレージ（S3等）を連携するには？

Docker Volume Plugin を使用する。例えば `rexray/s3fs` プラグインでS3をボリュームとしてマウントできる。ただし、ブロックストレージ（EBS等）の方がパフォーマンスが良い場合が多い。

```bash
# S3 volume driver プラグインのインストール
docker plugin install rexray/s3fs \
  S3FS_ACCESSKEY=xxx \
  S3FS_SECRETKEY=xxx

# S3バックエンドのボリュームを作成
docker volume create -d rexray/s3fs my-s3-data
```

### Q5: ボリュームの暗号化はどう実現する？

Docker 自体にはボリューム暗号化機能がない。以下の方法で対応する:
- ホストOS側でディスク暗号化 (LUKS, dm-crypt)
- クラウドプロバイダーの暗号化ストレージ (AWS EBS暗号化, GCP Persistent Disk暗号化)
- ボリュームプラグインの暗号化機能

### Q6: コンテナ間でボリュームを共有する場合の注意点は？

複数のコンテナが同一ボリュームを同時にマウントする場合、データの一貫性に注意が必要。

```yaml
# docker-compose.yml
services:
  writer:
    image: my-writer-app:latest
    volumes:
      - shared-data:/data   # 書き込みあり

  reader:
    image: my-reader-app:latest
    volumes:
      - shared-data:/data:ro   # 読み取り専用

  processor:
    image: my-processor:latest
    volumes:
      - shared-data:/data:ro   # 読み取り専用

volumes:
  shared-data:
```

注意事項:
- **ファイルロック**: 複数コンテナが同一ファイルに書き込む場合、アプリケーションレベルでロック機構を実装する
- **読み取り専用**: 読み取りだけのコンテナは `:ro` で明示的にマウントする
- **データベース**: データベースボリュームは原則として1コンテナからのみアクセスする。レプリケーションが必要なら、データベースのネイティブ機能（PostgreSQLストリーミングレプリケーション等）を使う
- **NFS**: 複数ホスト間でファイル共有する場合は NFS ボリュームを使用する

### Q7: ボリュームのクリーンアップ戦略は？

未使用ボリュームが蓄積するとディスクを圧迫する。安全なクリーンアップ手順を確立しておく。

```bash
# 未使用ボリュームの確認（削除はしない）
docker volume ls -f dangling=true

# 未使用ボリュームの削除
docker volume prune

# 全未使用リソース（イメージ、コンテナ、ネットワーク、ボリューム）の削除
docker system prune --volumes

# ラベルベースのクリーンアップ（安全性向上）
docker volume ls --filter "label=environment=development" -q | xargs docker volume rm
```

```bash
#!/bin/bash
# cleanup-volumes.sh - 安全なボリュームクリーンアップスクリプト

set -euo pipefail

echo "=== 現在のボリューム使用状況 ==="
docker system df -v | head -20

echo ""
echo "=== 未使用ボリューム一覧 ==="
DANGLING=$(docker volume ls -f dangling=true -q)

if [ -z "$DANGLING" ]; then
    echo "未使用ボリュームはありません。"
    exit 0
fi

echo "$DANGLING"
echo ""
echo "合計: $(echo "$DANGLING" | wc -l) 個"
echo ""

# 保護対象ボリュームの確認（名前にprod/productionが含まれるものは除外）
SAFE_TO_DELETE=$(echo "$DANGLING" | grep -v -E "(prod|production|backup)" || true)

if [ -z "$SAFE_TO_DELETE" ]; then
    echo "安全に削除可能なボリュームはありません。"
    exit 0
fi

echo "以下のボリュームを削除します:"
echo "$SAFE_TO_DELETE"
echo ""
read -p "実行しますか？ (y/N): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo "$SAFE_TO_DELETE" | xargs docker volume rm
    echo "削除完了。"
else
    echo "キャンセルしました。"
fi
```

### Q8: Docker Composeでボリューム名を明示的に設定するには？

デフォルトでは `<プロジェクト名>_<ボリューム名>` 形式になるが、`name` フィールドで明示的に設定できる。

```yaml
volumes:
  pgdata:
    name: my-app-pgdata   # 明示的な名前（プロジェクト名プレフィックスなし）
    driver: local
    labels:
      com.example.project: "my-app"
      com.example.type: "database"
```

### Q9: ボリュームデータの移行手順は？

あるホストから別のホストへボリュームデータを移行する方法。

```bash
#!/bin/bash
# migrate-volume.sh - ボリュームデータの移行

SOURCE_VOLUME=$1
TARGET_HOST=$2
TARGET_VOLUME=$3

# 1. ソースボリュームをtarにエクスポート
echo "[1/3] ボリュームをエクスポート中..."
docker run --rm \
  -v ${SOURCE_VOLUME}:/source:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/volume-backup.tar.gz -C /source .

# 2. tarファイルをリモートホストに転送
echo "[2/3] リモートホストに転送中..."
scp volume-backup.tar.gz ${TARGET_HOST}:/tmp/

# 3. リモートホストでボリュームにインポート
echo "[3/3] リモートホストでインポート中..."
ssh ${TARGET_HOST} << 'EOF'
docker volume create ${TARGET_VOLUME}
docker run --rm \
  -v ${TARGET_VOLUME}:/target \
  -v /tmp:/backup:ro \
  alpine sh -c "cd /target && tar xzf /backup/volume-backup.tar.gz"
rm /tmp/volume-backup.tar.gz
echo "移行完了。"
EOF

# ローカルのバックアップファイルを削除
rm volume-backup.tar.gz
echo "すべての処理が完了しました。"
```

### Q10: ボリュームのサイズ制限は設定できる？

Dockerのデフォルトlocalドライバーでは直接的なサイズ制限機能はない。以下の方法で対応できる。

1. **tmpfsの場合**: `size` オプションで制限可能

```yaml
services:
  app:
    tmpfs:
      - /tmp:size=100m
```

2. **xfs + pquota**: ホストがxfsファイルシステムを使用している場合

```bash
# xfsでプロジェクトクォータを有効化
docker daemon --storage-opt dm.basesize=20G
```

3. **ボリュームプラグイン**: 一部のプラグインはサイズ制限をサポート

4. **監視ベース**: サイズ制限の代わりにモニタリングとアラートで対応

```bash
# ボリュームサイズの定期チェックスクリプト
docker system df -v | grep "VOLUME" -A 100 | \
  awk '$NF ~ /GB/ && $NF+0 > 10 {print "WARNING: " $1 " is " $NF}'
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Named Volume | Docker管理。本番推奨。ポータブル |
| Bind Mount | ホストパス直結。開発向き。`--mount` 構文推奨 |
| tmpfs | メモリ上。機密データの一時保存に最適 |
| ストレージドライバー | overlay2がデフォルト推奨。CoW方式で動作 |
| バックアップ | tar + 別コンテナで実施。定期バックアップ必須 |
| パフォーマンス | DBは必ずNamed Volume。macOSでは依存をVolume分離 |
| 権限 | Dockerfile内でchown。非rootユーザー設定と組み合わせ |
| NFS/外部ストレージ | driver_opts で設定。マルチホスト共有に活用 |
| 監視 | `docker system df -v` で定期確認 |

---

## 次に読むべきガイド

- [リバースプロキシ](./02-reverse-proxy.md) -- Nginx/Traefikの設定とDocker連携
- [本番ベストプラクティス](../04-production/00-production-best-practices.md) -- ボリューム戦略を含む本番構成
- [Kubernetes永続ボリューム](../05-orchestration/02-kubernetes-advanced.md) -- PV/PVCによるストレージ管理

---

## 参考文献

1. Docker公式ドキュメント "Manage data in Docker" -- https://docs.docker.com/storage/
2. Docker公式ドキュメント "Use volumes" -- https://docs.docker.com/storage/volumes/
3. Docker公式ドキュメント "Storage drivers" -- https://docs.docker.com/storage/storagedriver/
4. Docker公式ドキュメント "Bind mounts" -- https://docs.docker.com/storage/bind-mounts/
5. Docker公式ドキュメント "tmpfs mounts" -- https://docs.docker.com/storage/tmpfs/
6. Nigel Poulton (2023) *Docker Deep Dive*, Chapter 13: Volumes and Persistent Data
7. Adrian Mouat (2023) *Using Docker*, Chapter 8: Managing Data with Volumes
