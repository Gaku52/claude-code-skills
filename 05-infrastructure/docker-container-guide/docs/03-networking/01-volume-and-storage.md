# ボリュームとストレージ

> コンテナのライフサイクルを超えてデータを永続化するための3つのマウント方式とストレージドライバーの仕組みを理解する。

---

## この章で学ぶこと

1. **Named Volume / Bind Mount / tmpfs の違いと使い分け**を理解する
2. **ボリュームのライフサイクル管理**（作成・バックアップ・移行・削除）を習得する
3. **ストレージドライバーの仕組み**とパフォーマンス特性を把握する

---

## 1. なぜデータ永続化が必要か

Dockerコンテナはイミュータブルに設計されている。コンテナの書き込み可能レイヤー（writable layer）はコンテナ削除と同時に消失する。データベースのデータ、アップロードされたファイル、設定ファイルなど、コンテナのライフサイクルを超えて保持すべきデータにはボリュームが必要。

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

---

## 2. Named Volume

DockerエンジンがManageするボリューム。ホスト上の `/var/lib/docker/volumes/` 配下に保存される。

### コード例1: Named Volumeの基本操作

```bash
# ボリュームの作成
docker volume create my-data

# ボリューム一覧
docker volume ls

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

# コンテナを削除してもボリュームは残る
docker rm -f postgres-db
docker volume ls  # my-data は健在

# 未使用ボリュームの一括削除（注意して使用）
docker volume prune
```

### コード例2: Docker Composeでのボリューム定義

```yaml
# docker-compose.yml
version: "3.9"

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
  redis-data:
    driver: local
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
# docker-compose.yml
# services:
#   app:
#     image: my-app
#     tmpfs:
#       - /tmp:size=100m
#     volumes:
#       - type: tmpfs
#         target: /run/secrets
#         tmpfs:
#           size: 10485760  # 10MB
#           mode: 0700
```

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

### コード例6: NFSボリュームドライバー

```bash
# NFSバックエンドのボリュームを作成
docker volume create \
  --driver local \
  --opt type=nfs \
  --opt o=addr=192.168.1.100,rw,nfsvers=4 \
  --opt device=:/exports/data \
  nfs-data

# Docker Composeでの定義
# docker-compose.yml
# volumes:
#   shared-data:
#     driver: local
#     driver_opts:
#       type: nfs
#       o: "addr=192.168.1.100,rw,nfsvers=4"
#       device: ":/exports/data"
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

---

## 7. パフォーマンス最適化

### コード例8: macOSでのBind Mountパフォーマンス改善

```yaml
# docker-compose.yml
# macOSではBind Mountが遅い問題の対策
version: "3.9"

services:
  app:
    build: .
    volumes:
      # :cached - ホスト側の変更がコンテナに反映されるまで遅延許容
      - ./src:/app/src:cached

      # :delegated - コンテナ側の変更がホストに反映されるまで遅延許容
      - ./logs:/app/logs:delegated

      # node_modules はNamed Volumeで管理（Bind Mountより高速）
      - node_modules:/app/node_modules

volumes:
  node_modules:
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

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Named Volume | Docker管理。本番推奨。ポータブル |
| Bind Mount | ホストパス直結。開発向き。`--mount` 構文推奨 |
| tmpfs | メモリ上。機密データの一時保存に最適 |
| ストレージドライバー | overlay2がデフォルト推奨。CoW方式で動作 |
| バックアップ | tar + 別コンテナで実施。定期バックアップ必須 |
| パフォーマンス | DBは必ずNamed Volume。macOSでは:cached活用 |
| 権限 | Dockerfile内でchown。非rootユーザー設定と組み合わせ |

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
4. Nigel Poulton (2023) *Docker Deep Dive*, Chapter 13: Volumes and Persistent Data
5. Adrian Mouat (2023) *Using Docker*, Chapter 8: Managing Data with Volumes
