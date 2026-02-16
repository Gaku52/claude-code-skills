# ボリュームとストレージ

> コンテナのライフサイクルを超えてデータを永続化するための3つのマウント方式とストレージドライバーの仕組みを理解する。

---

## この章で学ぶこと

1. **Named Volume / Bind Mount / tmpfs の違いと使い分け**を理解する
2. **ボリュームのライフサイクル管理**（作成・バックアップ・移行・削除）を習得する
3. **ストレージドライバーの仕組み**とパフォーマンス特性を把握する
4. **本番環境でのストレージ設計パターン**を学ぶ
5. **各種データベースに最適なボリューム設定**を実践する

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
