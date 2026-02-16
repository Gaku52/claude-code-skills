# Docker 基本操作

> イメージの取得からコンテナの起動・停止・削除、ログ確認、コンテナ内操作まで、Docker の日常的な操作を体系的に学ぶ。

---

## この章で学ぶこと

1. **イメージとコンテナの関係**を理解し、ライフサイクル全体を把握する
2. **docker run の主要オプション**を使いこなし、目的に応じたコンテナ起動ができる
3. **ログ確認・コンテナ内操作・リソース管理**の実践的なスキルを身につける
4. **ネットワーク接続とボリュームマウント**の仕組みを理解し、実務で活用できる
5. **リソース制限と監視**によりコンテナの安定運用ができる

---

## 1. イメージとコンテナの関係

### 1.1 概念モデル

```
+--------------------------------------------------+
|                  レジストリ                         |
|              (Docker Hub 等)                      |
|  +----------+  +----------+  +----------+        |
|  | nginx    |  | postgres |  | node     |        |
|  | :1.25    |  | :16      |  | :20      |        |
|  +----------+  +----------+  +----------+        |
+--------|-----------------------------------------+
         | docker pull
         v
+--------------------------------------------------+
|              ローカルイメージ                       |
|  +--------------------------------------------+  |
|  |  イメージ = 読み取り専用テンプレート            |  |
|  |  (レイヤーの積み重ね)                         |  |
|  |                                            |  |
|  |  Layer 3: アプリケーションコード              |  |
|  |  Layer 2: 依存パッケージ                     |  |
|  |  Layer 1: ベースOS (alpine等)               |  |
|  +--------------------------------------------+  |
+--------|-----------------------------------------+
         | docker run (イメージ + 書き込み可能レイヤー)
         v
+--------------------------------------------------+
|              コンテナ (実行中インスタンス)            |
|  +--------------------------------------------+  |
|  |  書き込み可能レイヤー (コンテナ固有)           |  |
|  |  --------------------------------          |  |
|  |  イメージレイヤー (読み取り専用・共有)         |  |
|  +--------------------------------------------+  |
|  1つのイメージから複数のコンテナを作成可能          |
+--------------------------------------------------+
```

### 1.2 コンテナのライフサイクル

```
                docker create
                     |
                     v
  +--------+    +---------+    docker start    +---------+
  | 不存在  |--->| Created |------------------>| Running |
  +--------+    +---------+                    +---------+
       ^             |                          |   |   |
       |             |  docker rm               |   |   |
       +-------------+                         |   |   |
       |                                       |   |   |
       |        docker stop / コンテナ終了       |   |   |
       |             +-------------------------+   |   |
       |             v                             |   |
       |        +---------+    docker restart      |   |
       |        | Stopped  |---------------------->+   |
       |        | (Exited) |                           |
       |        +---------+                           |
       |             |                                |
       |  docker rm  |       docker pause             |
       +-------------+            |                   |
                                  v                   |
                            +----------+              |
                            |  Paused  |--------------+
                            +----------+  docker unpause
```

### 1.3 コンテナの状態一覧

| 状態 | 説明 | 遷移元 | 遷移コマンド |
|---|---|---|---|
| Created | コンテナが作成されたが未起動 | - | `docker create` |
| Running | コンテナが実行中 | Created, Stopped, Paused | `docker start`, `docker restart`, `docker unpause` |
| Paused | プロセスが一時停止中 | Running | `docker pause` |
| Stopped (Exited) | メインプロセスが終了 | Running | `docker stop`, プロセス終了 |
| Removing | 削除処理中 | Created, Stopped | `docker rm` |
| Dead | 異常終了（リソース解放失敗） | Running | 異常発生時 |

### 1.4 コンテナ vs 仮想マシン

```
+-----------------------------------------------+
|  仮想マシン (VM)          |  コンテナ            |
|                          |                     |
|  +---+ +---+ +---+      |  +---+ +---+ +---+ |
|  |App| |App| |App|      |  |App| |App| |App| |
|  +---+ +---+ +---+      |  +---+ +---+ +---+ |
|  |Lib| |Lib| |Lib|      |  |Lib| |Lib| |Lib| |
|  +---+ +---+ +---+      |  +---+ +---+ +---+ |
|  |OS | |OS | |OS |      |  +-----------------+|
|  +---+ +---+ +---+      |  |   Docker Engine ||
|  +-------------------+   |  +-----------------+|
|  |   Hypervisor      |   |  |    ホスト OS     ||
|  +-------------------+   |  +-----------------+|
|  |    ホスト OS       |   |                     |
|  +-------------------+   |                     |
|                          |                     |
|  起動: 数分              |  起動: 数秒           |
|  サイズ: 数GB            |  サイズ: 数十MB       |
|  オーバーヘッド: 大       |  オーバーヘッド: 小    |
|  分離レベル: 高           |  分離レベル: 中       |
+-----------------------------------------------+
```

---

## 2. docker run の基本

### 2.1 基本構文

```bash
docker run [オプション] イメージ名[:タグ] [コマンド] [引数...]
```

### 2.2 最もシンプルな実行

```bash
# 実行して結果を表示（フォアグラウンド）
docker run --rm alpine echo "Hello, Docker!"
# Hello, Docker!

# --rm: コンテナ終了時に自動削除
# alpine: 軽量Linuxイメージ（約5MB）
# echo "Hello, Docker!": コンテナ内で実行するコマンド
```

### 2.3 インタラクティブモード

```bash
# コンテナ内でシェルを起動
docker run -it --rm alpine /bin/sh

# -i: 標準入力を開いたままにする (interactive)
# -t: 疑似TTYを割り当てる (tty)
# 組み合わせて -it でインタラクティブなシェルになる

# コンテナ内で操作
/ # ls
/ # cat /etc/os-release
/ # exit

# Ubuntu ベースでインタラクティブ操作
docker run -it --rm ubuntu:22.04 /bin/bash
root@abc123:/# apt-get update
root@abc123:/# apt-get install -y curl
root@abc123:/# curl -s https://httpbin.org/ip
root@abc123:/# exit

# Python のインタラクティブシェル
docker run -it --rm python:3.12-slim python
>>> print("Hello from Docker!")
>>> import sys; print(sys.version)
>>> exit()
```

### 2.4 バックグラウンド実行

```bash
# デタッチモードで実行
docker run -d --name my-nginx -p 8080:80 nginx:alpine

# -d: バックグラウンドで実行 (detach)
# --name my-nginx: コンテナに名前を付ける
# -p 8080:80: ホストのポート8080をコンテナのポート80にマッピング

# 動作確認
curl http://localhost:8080

# フォアグラウンドに戻す（ログをストリーミング）
docker attach my-nginx
# Ctrl+C で停止、Ctrl+P Ctrl+Q でデタッチ

# コンテナIDの確認
docker ps
# CONTAINER ID   IMAGE          COMMAND                  CREATED          STATUS          PORTS                  NAMES
# a1b2c3d4e5f6   nginx:alpine   "/docker-entrypoint.…"   10 seconds ago   Up 9 seconds    0.0.0.0:8080->80/tcp   my-nginx
```

### 2.5 環境変数の設定

```bash
# 環境変数を指定して実行
docker run -d --name my-db \
    -e POSTGRES_USER=admin \
    -e POSTGRES_PASSWORD=secret123 \
    -e POSTGRES_DB=myapp \
    -p 5432:5432 \
    postgres:16-alpine

# .env ファイルから読み込み
# .env ファイル内容:
# POSTGRES_USER=admin
# POSTGRES_PASSWORD=secret123
# POSTGRES_DB=myapp
docker run -d --name my-db \
    --env-file .env \
    -p 5432:5432 \
    postgres:16-alpine

# 環境変数の確認
docker exec my-db env | grep POSTGRES

# ホストの環境変数を引き継ぐ
export MY_VAR=hello
docker run --rm -e MY_VAR alpine env | grep MY_VAR
# MY_VAR=hello
```

### 2.6 再起動ポリシー

```bash
# 常に再起動（手動停止以外）
docker run -d --name always-up \
    --restart unless-stopped \
    nginx:alpine

# 再起動ポリシーの種類
# no:            再起動しない（デフォルト）
# on-failure:    異常終了時のみ再起動
# on-failure:5:  最大5回まで再起動
# always:        常に再起動（手動停止しても Docker 起動時に再開）
# unless-stopped: 常に再起動（手動停止時は Docker 起動時に再開しない）

# 再起動ポリシーの変更（実行中のコンテナ）
docker update --restart unless-stopped my-nginx

# 再起動回数の確認
docker inspect --format '{{.RestartCount}}' my-container
docker inspect --format '{{.State.StartedAt}}' my-container
```

### 2.7 ラベルの活用

```bash
# コンテナにラベルを付与
docker run -d --name web \
    --label env=production \
    --label team=backend \
    --label version=1.2.3 \
    nginx:alpine

# ラベルでフィルタリング
docker ps --filter "label=env=production"
docker ps --filter "label=team=backend"

# ラベルの確認
docker inspect --format '{{.Config.Labels}}' web
# map[env:production team:backend version:1.2.3]
```

---

## 3. ポートマッピング

### 3.1 ポートマッピングの仕組み

```
+-----------------------------------------------------+
|                    ホストマシン                        |
|                                                     |
|  ブラウザ ----> localhost:8080 ---+                  |
|                                  |                  |
|  +-------------------------------|---------+        |
|  |        Docker ネットワーク      |         |        |
|  |                               v         |        |
|  |  +----------+  +-----------+           |        |
|  |  | コンテナA |  | コンテナB  |           |        |
|  |  | :80      |  | :3000     |           |        |
|  |  +----------+  +-----------+           |        |
|  |   8080:80       3000:3000              |        |
|  +----------------------------------------+        |
+-----------------------------------------------------+
```

```bash
# 基本的なポートマッピング
docker run -d -p 8080:80 nginx:alpine
# ホスト:8080 -> コンテナ:80

# 複数ポートのマッピング
docker run -d -p 8080:80 -p 8443:443 nginx:alpine

# ランダムポートの割り当て
docker run -d -P nginx:alpine
docker port $(docker ps -q -l)
# 0.0.0.0:32768->80/tcp

# 特定のIPにバインド
docker run -d -p 127.0.0.1:8080:80 nginx:alpine
# localhost からのみアクセス可能

# UDPポートのマッピング
docker run -d -p 5353:53/udp dns-server

# 全インターフェースにバインド（デフォルト）
docker run -d -p 0.0.0.0:8080:80 nginx:alpine

# IPv6 でバインド
docker run -d -p "[::1]:8080:80" nginx:alpine
```

### 3.2 ポートマッピングの確認

```bash
# コンテナのポートマッピング確認
docker port my-nginx
# 80/tcp -> 0.0.0.0:8080

# 特定のポートの確認
docker port my-nginx 80
# 0.0.0.0:8080

# docker ps でポートマッピングを確認
docker ps --format "table {{.Names}}\t{{.Ports}}"
# NAMES       PORTS
# my-nginx    0.0.0.0:8080->80/tcp

# ホスト側で使用中のポートを確認
# macOS
lsof -i :8080
# Linux
ss -tlnp | grep 8080
```

### 3.3 ネットワークモード

```bash
# bridge（デフォルト）: 独立したネットワーク名前空間
docker run -d --network bridge nginx:alpine

# host: ホストのネットワークを直接使用（Linux のみ）
docker run -d --network host nginx:alpine
# ポートマッピング不要、直接 80 番ポートでアクセス

# none: ネットワーク無効
docker run -d --network none alpine sleep infinity

# ネットワークモードの比較
# bridge: 分離性あり、ポートマッピング必要、デフォルト
# host:   分離なし、ポートマッピング不要、高パフォーマンス
# none:   完全に分離、外部通信不可、セキュリティ重視
```

---

## 4. ボリュームマウント

### 4.1 マウントの種類

```
+------------------------------------------------------+
|                   マウントの種類                        |
|                                                      |
|  1. バインドマウント (Bind Mount)                     |
|  +------------------+     +-------------------+      |
|  | ホストのディレクトリ | --> | コンテナ内パス     |      |
|  | ./src             |     | /app/src          |      |
|  +------------------+     +-------------------+      |
|                                                      |
|  2. 名前付きボリューム (Named Volume)                  |
|  +------------------+     +-------------------+      |
|  | Docker管理領域    | --> | コンテナ内パス     |      |
|  | my-data          |     | /var/lib/data      |      |
|  +------------------+     +-------------------+      |
|                                                      |
|  3. tmpfs マウント                                    |
|  +------------------+     +-------------------+      |
|  | メモリ上           | --> | コンテナ内パス     |      |
|  | (揮発性)          |     | /tmp               |      |
|  +------------------+     +-------------------+      |
+------------------------------------------------------+
```

```bash
# バインドマウント（開発時に最適）
docker run -d --name dev-app \
    -v $(pwd)/src:/app/src \
    -p 3000:3000 \
    node:20-alpine

# 名前付きボリューム（データ永続化に最適）
docker volume create db-data
docker run -d --name my-db \
    -v db-data:/var/lib/postgresql/data \
    postgres:16-alpine

# 読み取り専用マウント
docker run -d --name web \
    -v $(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro \
    nginx:alpine

# tmpfs マウント（一時データ・機密データ）
docker run -d --name app \
    --tmpfs /tmp:rw,size=100m \
    my-app

# ボリュームの一覧と詳細
docker volume ls
docker volume inspect db-data
```

### 4.2 --mount オプション（推奨構文）

```bash
# -v 構文（短縮形）
docker run -d -v db-data:/var/lib/postgresql/data postgres:16-alpine

# --mount 構文（推奨、より明確）
docker run -d \
    --mount type=volume,source=db-data,target=/var/lib/postgresql/data \
    postgres:16-alpine

# バインドマウントの --mount 構文
docker run -d \
    --mount type=bind,source=$(pwd)/src,target=/app/src \
    node:20-alpine

# 読み取り専用マウント
docker run -d \
    --mount type=bind,source=$(pwd)/config.yml,target=/app/config.yml,readonly \
    my-app

# tmpfs の --mount 構文
docker run -d \
    --mount type=tmpfs,target=/tmp,tmpfs-size=100m \
    my-app
```

### 4.3 ボリュームの管理

```bash
# ボリュームの作成
docker volume create my-data

# ドライバを指定したボリューム作成
docker volume create --driver local \
    --opt type=nfs \
    --opt o=addr=192.168.1.100,rw \
    --opt device=:/export/data \
    nfs-data

# ボリュームの一覧
docker volume ls
docker volume ls --filter "name=my-"
docker volume ls --filter "dangling=true"

# ボリュームの詳細
docker volume inspect my-data
# [
#     {
#         "CreatedAt": "2024-01-15T10:00:00Z",
#         "Driver": "local",
#         "Labels": {},
#         "Mountpoint": "/var/lib/docker/volumes/my-data/_data",
#         "Name": "my-data",
#         "Options": {},
#         "Scope": "local"
#     }
# ]

# ボリュームの削除
docker volume rm my-data

# 未使用ボリュームの一括削除
docker volume prune

# ボリューム間のデータコピー
docker run --rm \
    -v source-vol:/from \
    -v dest-vol:/to \
    alpine sh -c "cp -a /from/. /to/"

# ボリュームのバックアップ
docker run --rm \
    -v my-data:/data:ro \
    -v $(pwd):/backup \
    alpine tar czf /backup/my-data-backup.tar.gz -C /data .

# バックアップからの復元
docker run --rm \
    -v my-data:/data \
    -v $(pwd):/backup:ro \
    alpine tar xzf /backup/my-data-backup.tar.gz -C /data
```

### 比較表: マウントの種類

| 種類 | データ永続性 | ホストからアクセス | パフォーマンス | 用途 |
|---|---|---|---|---|
| バインドマウント | ホストに依存 | 直接可能 | OS依存 | 開発時のソースコード共有 |
| 名前付きボリューム | Docker管理で永続 | Docker経由 | 高い | データベース、永続データ |
| 匿名ボリューム | コンテナ削除で孤立 | Docker経由 | 高い | 一時的なデータ |
| tmpfs | メモリ上（揮発性） | 不可 | 最高 | 機密情報、一時ファイル |

---

## 5. コンテナ管理

### 5.1 一覧表示

```bash
# 実行中のコンテナ一覧
docker ps

# 全コンテナ一覧（停止中も含む）
docker ps -a

# コンテナIDのみ表示
docker ps -q

# フォーマット指定
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# JSON 形式で出力
docker ps --format json

# カスタムフォーマット
docker ps --format "{{.ID}}: {{.Names}} ({{.Status}}) - {{.Image}}"

# フィルタリング
docker ps --filter "status=exited"
docker ps --filter "name=my-"
docker ps --filter "label=env=production"
docker ps --filter "ancestor=nginx:alpine"
docker ps --filter "health=healthy"

# 最後に作成されたコンテナ
docker ps -l

# コンテナの数をカウント
docker ps -q | wc -l
```

### 5.2 停止と削除

```bash
# コンテナの停止（SIGTERM -> 10秒後 SIGKILL）
docker stop my-nginx

# タイムアウトを指定して停止
docker stop -t 30 my-nginx

# 複数コンテナの一括停止
docker stop my-nginx my-db my-redis

# 強制停止（SIGKILL）
docker kill my-nginx

# 特定のシグナルを送信
docker kill --signal=SIGHUP my-nginx

# コンテナの削除
docker rm my-nginx

# 停止と削除を一度に
docker rm -f my-nginx

# 停止中の全コンテナを削除
docker container prune

# 確認なしで削除
docker container prune -f

# 特定条件のコンテナを一括削除
docker rm $(docker ps -aq --filter "status=exited")
docker rm $(docker ps -aq --filter "label=env=test")
```

### 5.3 その他の管理操作

```bash
# コンテナの再起動
docker restart my-nginx

# コンテナの一時停止と再開
docker pause my-nginx
docker unpause my-nginx

# コンテナ名の変更
docker rename my-nginx web-server

# コンテナの詳細情報
docker inspect my-nginx

# 特定の情報を抽出
docker inspect --format '{{.NetworkSettings.IPAddress}}' my-nginx
docker inspect --format '{{.State.Status}}' my-nginx
docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' my-nginx
docker inspect --format '{{.HostConfig.Memory}}' my-nginx

# コンテナのプロセス一覧
docker top my-nginx
# UID     PID     PPID    C    STIME   TTY   TIME     CMD
# root    12345   12330   0    10:00   ?     00:00:00 nginx: master process
# nobody  12346   12345   0    10:00   ?     00:00:00 nginx: worker process

# コンテナの変更差分
docker diff my-nginx
# A /var/log/nginx/access.log
# C /run
# A /run/nginx.pid

# コンテナからホストへファイルコピー
docker cp my-nginx:/etc/nginx/nginx.conf ./nginx.conf

# ホストからコンテナへファイルコピー
docker cp ./custom.conf my-nginx:/etc/nginx/conf.d/

# コンテナを一時停止してファイルコピー
docker pause my-nginx
docker cp my-nginx:/var/log/nginx/ ./logs/
docker unpause my-nginx

# コンテナの待機（終了を待つ）
docker wait my-container
# 終了コード（0, 1 等）が返る
```

---

## 6. ログ管理

### 6.1 ログの表示

```bash
# ログの表示
docker logs my-nginx

# リアルタイムでログを追跡（tail -f 相当）
docker logs -f my-nginx

# 最新N行のみ表示
docker logs --tail 100 my-nginx

# タイムスタンプ付きで表示
docker logs -t my-nginx

# 特定時刻以降のログ
docker logs --since "2024-01-15T10:00:00" my-nginx
docker logs --since 30m my-nginx  # 30分前から
docker logs --since 2h my-nginx   # 2時間前から

# 特定時刻までのログ
docker logs --until "2024-01-15T12:00:00" my-nginx

# 組み合わせ
docker logs -f --tail 50 -t my-nginx

# ログをファイルに出力
docker logs my-nginx > nginx.log 2>&1
docker logs my-nginx 2>/dev/null > stdout.log
docker logs my-nginx 2>stderr.log >/dev/null
```

### 6.2 ログドライバ

```
+-----------------------------------------------------+
|              Docker ログドライバ                       |
|                                                     |
|  コンテナ stdout/stderr                               |
|       |                                             |
|       v                                             |
|  +------------------+                               |
|  | ログドライバ       |                               |
|  +-----|------------+                               |
|        |                                            |
|  +-----+------+--------+--------+-------+          |
|  |     |      |        |        |       |          |
|  v     v      v        v        v       v          |
| json  syslog fluentd  awslogs  gcplogs local       |
| -file                                              |
|  (デフォルト)                                        |
+-----------------------------------------------------+
```

### 6.3 ログドライバの設定

```bash
# コンテナ起動時にログドライバを指定
docker run -d --name app \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=5 \
    my-app

# fluentd ログドライバ
docker run -d --name app \
    --log-driver fluentd \
    --log-opt fluentd-address=localhost:24224 \
    --log-opt tag=docker.app \
    my-app

# syslog ログドライバ
docker run -d --name app \
    --log-driver syslog \
    --log-opt syslog-address=udp://logs.example.com:514 \
    --log-opt tag=myapp \
    my-app

# AWS CloudWatch Logs
docker run -d --name app \
    --log-driver awslogs \
    --log-opt awslogs-region=ap-northeast-1 \
    --log-opt awslogs-group=my-app \
    --log-opt awslogs-stream=production \
    my-app

# ログドライバの比較
```

| ログドライバ | 用途 | ログ確認方法 | docker logs 対応 |
|---|---|---|---|
| `json-file` | デフォルト、ローカル開発 | `docker logs` | 対応 |
| `local` | ローカル（効率的） | `docker logs` | 対応 |
| `syslog` | syslog サーバー転送 | syslog | 非対応 |
| `fluentd` | Fluentd 転送 | Fluentd | 非対応 |
| `awslogs` | AWS CloudWatch | CloudWatch | 非対応 |
| `gcplogs` | GCP Cloud Logging | Cloud Logging | 非対応 |
| `journald` | systemd journal | `journalctl` | 対応 |
| `none` | ログ無効 | なし | 非対応 |

---

## 7. コンテナ内操作 (exec)

### 7.1 基本操作

```bash
# 実行中のコンテナでコマンドを実行
docker exec my-nginx ls /etc/nginx

# インタラクティブシェルで接続
docker exec -it my-nginx /bin/sh

# bash が使えるコンテナの場合
docker exec -it my-nginx /bin/bash

# 特定のユーザーで実行
docker exec -u root my-nginx whoami
docker exec -u 1000:1000 my-nginx id

# 環境変数を設定して実行
docker exec -e MY_VAR=hello my-nginx env

# 作業ディレクトリを指定
docker exec -w /etc/nginx my-nginx ls

# バックグラウンドでコマンドを実行
docker exec -d my-nginx touch /tmp/marker
```

### 7.2 実践的な exec の活用

```bash
# データベースへの接続
docker exec -it my-db psql -U admin -d myapp
docker exec -it my-mysql mysql -u root -p

# Redis CLI への接続
docker exec -it my-redis redis-cli
127.0.0.1:6379> PING
PONG

# ファイルの内容確認
docker exec my-nginx cat /etc/nginx/nginx.conf

# プロセスの確認
docker exec my-nginx ps aux

# ネットワークの確認
docker exec my-nginx ping -c 3 google.com
docker exec my-nginx nslookup my-db
docker exec my-nginx wget -qO- http://localhost:80

# ディスク使用量の確認
docker exec my-nginx df -h
docker exec my-nginx du -sh /var/log/

# 環境変数の確認
docker exec my-nginx env | sort

# パッケージのインストール（デバッグ用、非推奨）
docker exec -it my-nginx sh -c "apk add --no-cache curl && curl localhost"
```

### 7.3 exec vs run の違い

```bash
# docker exec: 既存の実行中コンテナ内でコマンド実行
docker exec my-nginx cat /etc/nginx/nginx.conf
# -> my-nginx コンテナ内でファイルを表示

# docker run: 新しいコンテナを作成して実行
docker run --rm nginx:alpine cat /etc/nginx/nginx.conf
# -> 新しいコンテナを作成し、表示後に削除
```

| 観点 | docker exec | docker run |
|---|---|---|
| コンテナ | 既存の実行中コンテナ | 新規コンテナを作成 |
| 状態 | コンテナの状態を共有 | 独立した状態 |
| ネットワーク | コンテナのネットワークを使用 | 新しいネットワーク設定 |
| ボリューム | コンテナのボリュームを使用 | 新たに指定が必要 |
| 用途 | デバッグ、管理タスク | 一時的なコマンド実行 |
| 前提条件 | コンテナが実行中であること | イメージがあること |

---

## 8. リソース監視

### 8.1 docker stats

```bash
# リアルタイムのリソース使用状況
docker stats

# 出力例:
# CONTAINER ID   NAME       CPU %   MEM USAGE / LIMIT   MEM %   NET I/O         BLOCK I/O
# a1b2c3d4e5f6   my-nginx   0.05%   5.2MiB / 7.67GiB    0.07%   1.45kB / 0B    0B / 0B

# 特定のコンテナのみ
docker stats my-nginx my-db

# ワンショット（ストリーミングなし）
docker stats --no-stream

# フォーマット指定
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# JSON 形式で出力
docker stats --no-stream --format json
```

### 8.2 リソース制限の設定

```bash
# メモリ制限
docker run -d --name limited-app \
    --memory=256m \
    --memory-swap=512m \
    nginx:alpine

# メモリ予約（ソフトリミット）
docker run -d --name app \
    --memory=512m \
    --memory-reservation=256m \
    my-app

# CPU制限
docker run -d --name cpu-limited \
    --cpus=1.5 \
    nginx:alpine

# CPU シェア（相対的な重み）
docker run -d --name high-priority \
    --cpu-shares=1024 \
    my-app
docker run -d --name low-priority \
    --cpu-shares=256 \
    my-app

# CPU ピンニング（特定のCPUに固定）
docker run -d --name pinned-app \
    --cpuset-cpus="0,1" \
    my-app

# I/O 制限
docker run -d --name io-limited \
    --device-read-bps /dev/sda:10mb \
    --device-write-bps /dev/sda:10mb \
    my-app

# PID 制限
docker run -d --name pid-limited \
    --pids-limit=100 \
    my-app

# 実行中のコンテナのリソース制限を変更
docker update --memory=512m --cpus=2.0 limited-app

# 全コンテナのリソース制限を確認
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

### 8.3 ヘルスチェック

```bash
# ヘルスチェック付きでコンテナを起動
docker run -d --name web \
    --health-cmd="wget --no-verbose --tries=1 --spider http://localhost/ || exit 1" \
    --health-interval=30s \
    --health-timeout=5s \
    --health-retries=3 \
    --health-start-period=10s \
    nginx:alpine

# ヘルスチェックの状態確認
docker inspect --format '{{.State.Health.Status}}' web
# healthy / unhealthy / starting

# ヘルスチェックのログ確認
docker inspect --format '{{json .State.Health}}' web | python3 -m json.tool

# ヘルスチェックでフィルタリング
docker ps --filter "health=healthy"
docker ps --filter "health=unhealthy"
```

---

## 9. Docker ネットワーク

### 9.1 ネットワークの基本

```bash
# ネットワーク一覧
docker network ls
# NETWORK ID     NAME      DRIVER    SCOPE
# abc123         bridge    bridge    local
# def456         host      host      local
# ghi789         none      null      local

# カスタムネットワークの作成
docker network create my-network
docker network create --driver bridge --subnet 172.20.0.0/16 my-custom-net

# ネットワークの詳細
docker network inspect my-network

# コンテナをネットワークに接続
docker network connect my-network my-nginx

# コンテナをネットワークから切断
docker network disconnect my-network my-nginx

# ネットワークの削除
docker network rm my-network

# 未使用ネットワークの一括削除
docker network prune
```

### 9.2 コンテナ間通信

```bash
# カスタムネットワークで DNS による名前解決
docker network create app-network

docker run -d --name db \
    --network app-network \
    -e POSTGRES_PASSWORD=secret \
    postgres:16-alpine

docker run -d --name app \
    --network app-network \
    -e DATABASE_URL=postgresql://postgres:secret@db:5432/postgres \
    my-app

# app コンテナから db コンテナに「db」という名前でアクセス可能
docker exec app ping -c 3 db
# PING db (172.20.0.2): 56 data bytes
# 64 bytes from 172.20.0.2: seq=0 ttl=64 time=0.085 ms

# ネットワークエイリアス
docker run -d --name db-primary \
    --network app-network \
    --network-alias database \
    postgres:16-alpine
# 「database」という名前でもアクセス可能
```

### 9.3 ネットワークドライバの比較

| ドライバ | 説明 | 用途 | DNS解決 |
|---|---|---|---|
| bridge | デフォルト、独立ネットワーク | 単一ホスト上のコンテナ通信 | カスタムネットワークのみ |
| host | ホストネットワークを共有 | パフォーマンス重視 | ホストのDNS |
| overlay | 複数ホスト間のネットワーク | Docker Swarm / 分散システム | あり |
| macvlan | 物理ネットワークに直接接続 | レガシーアプリ統合 | なし |
| none | ネットワーク無効 | セキュリティ重視の分離 | なし |

---

## 10. クリーンアップ

### 比較表 1: クリーンアップコマンド

| コマンド | 対象 | 説明 |
|---|---|---|
| `docker container prune` | 停止済みコンテナ | 停止中の全コンテナを削除 |
| `docker image prune` | 未使用イメージ | タグなし（dangling）イメージを削除 |
| `docker image prune -a` | 全未使用イメージ | コンテナが使用していない全イメージを削除 |
| `docker volume prune` | 未使用ボリューム | どのコンテナにもマウントされていないボリュームを削除 |
| `docker network prune` | 未使用ネットワーク | コンテナが接続していないネットワークを削除 |
| `docker system prune` | 上記全て | 一括クリーンアップ |
| `docker system prune -a` | 上記全て + 全未使用イメージ | 完全クリーンアップ |
| `docker builder prune` | ビルドキャッシュ | BuildKit のキャッシュを削除 |

### 比較表 2: docker run 主要オプション

| オプション | 短縮形 | 説明 | 例 |
|---|---|---|---|
| `--detach` | `-d` | バックグラウンド実行 | `-d` |
| `--interactive` | `-i` | 標準入力を開く | `-i` |
| `--tty` | `-t` | 疑似TTY割り当て | `-t` |
| `--rm` | | 終了時に自動削除 | `--rm` |
| `--name` | | コンテナ名を指定 | `--name web` |
| `--publish` | `-p` | ポートマッピング | `-p 8080:80` |
| `--volume` | `-v` | ボリュームマウント | `-v data:/app` |
| `--mount` | | マウント（推奨構文） | `--mount type=volume,...` |
| `--env` | `-e` | 環境変数設定 | `-e KEY=val` |
| `--env-file` | | envファイル読み込み | `--env-file .env` |
| `--network` | | ネットワーク指定 | `--network my-net` |
| `--memory` | `-m` | メモリ制限 | `-m 256m` |
| `--cpus` | | CPU制限 | `--cpus 1.5` |
| `--restart` | | 再起動ポリシー | `--restart unless-stopped` |
| `--platform` | | プラットフォーム指定 | `--platform linux/amd64` |
| `--label` | `-l` | ラベル付与 | `-l env=prod` |
| `--health-cmd` | | ヘルスチェック | `--health-cmd "curl ..."` |
| `--user` | `-u` | 実行ユーザー | `-u 1000:1000` |
| `--workdir` | `-w` | 作業ディレクトリ | `-w /app` |
| `--hostname` | `-h` | ホスト名設定 | `-h myhost` |
| `--add-host` | | hosts エントリ追加 | `--add-host db:10.0.0.1` |
| `--dns` | | DNS サーバー | `--dns 8.8.8.8` |
| `--cap-add` | | Linux capability 追加 | `--cap-add SYS_PTRACE` |
| `--cap-drop` | | Linux capability 削除 | `--cap-drop ALL` |
| `--read-only` | | ルートFS読み取り専用 | `--read-only` |
| `--tmpfs` | | tmpfs マウント | `--tmpfs /tmp` |
| `--init` | | PID 1 に tini を使用 | `--init` |
| `--pid` | | PID 名前空間 | `--pid host` |

```bash
# ディスク使用量の確認
docker system df

# 出力例:
# TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
# Images          15        5         4.2GB     2.8GB (66%)
# Containers      8         3         120MB     80MB (66%)
# Local Volumes   12        4         1.5GB     800MB (53%)
# Build Cache     50        0         2.1GB     2.1GB

# 詳細表示
docker system df -v

# 一括クリーンアップ（確認付き）
docker system prune

# ボリュームも含めて完全クリーンアップ
docker system prune -a --volumes

# フィルタ付きクリーンアップ
docker system prune --filter "until=24h"
docker image prune -a --filter "until=168h"  # 1週間以上前

# 定期的なクリーンアップスクリプト
# cron に登録: 0 3 * * 0 /usr/local/bin/docker-cleanup.sh
#!/bin/bash
# docker-cleanup.sh
echo "=== Docker Cleanup Start ==="
echo "Before:"
docker system df
docker container prune -f
docker image prune -a --filter "until=168h" -f
docker volume prune -f
docker network prune -f
docker builder prune -f --keep-storage=5GB
echo "After:"
docker system df
echo "=== Docker Cleanup Complete ==="
```

---

## 11. 実践的なワークフロー例

### 11.1 Web アプリケーション開発環境

```bash
# データベースの起動
docker run -d --name dev-db \
    --network dev-net \
    -e POSTGRES_USER=dev \
    -e POSTGRES_PASSWORD=devpass \
    -e POSTGRES_DB=myapp_dev \
    -v db-data:/var/lib/postgresql/data \
    -p 5432:5432 \
    postgres:16-alpine

# Redis の起動
docker run -d --name dev-redis \
    --network dev-net \
    -p 6379:6379 \
    redis:7-alpine

# アプリケーションの起動（ソースコードをマウント）
docker run -d --name dev-app \
    --network dev-net \
    -v $(pwd):/app \
    -p 3000:3000 \
    -e DATABASE_URL=postgresql://dev:devpass@dev-db:5432/myapp_dev \
    -e REDIS_URL=redis://dev-redis:6379 \
    node:20-alpine sh -c "cd /app && npm install && npm run dev"

# ログの確認
docker logs -f dev-app

# データベースに接続してデバッグ
docker exec -it dev-db psql -U dev -d myapp_dev

# 環境の停止
docker stop dev-app dev-redis dev-db

# 環境の削除（データは保持）
docker rm dev-app dev-redis dev-db

# データも含めて完全削除
docker rm -f dev-app dev-redis dev-db
docker volume rm db-data
docker network rm dev-net
```

### 11.2 マルチサービスのデバッグ

```bash
# ネットワーク作成
docker network create debug-net

# 問題のあるコンテナのネットワークデバッグ
docker run -it --rm \
    --network debug-net \
    nicolaka/netshoot \
    bash

# netshoot コンテナ内で:
# dig db    # DNS 解決確認
# ping db   # 疎通確認
# curl app:3000/health  # HTTP 確認
# tcpdump -i eth0 port 5432  # パケットキャプチャ
# nmap -sT db  # ポートスキャン
```

---

## 12. アンチパターン

### アンチパターン 1: --rm を付けずにテスト用コンテナを量産

```bash
# NG: 使い捨てコンテナが溜まる
docker run alpine echo "test1"
docker run alpine echo "test2"
docker run alpine echo "test3"
# docker ps -a で大量の Exited コンテナが表示される

# OK: テスト・一時実行には --rm を付ける
docker run --rm alpine echo "test1"
docker run --rm alpine echo "test2"
docker run --rm alpine echo "test3"
# 実行後にコンテナが自動削除される
```

### アンチパターン 2: docker exec で本番コンテナを変更する

```bash
# NG: 実行中のコンテナ内でファイルを変更
docker exec -it production-app bash
root@abc123:/# apt-get install vim
root@abc123:/# vim /app/config.json
# -> コンテナ再起動で変更が消失
# -> 変更の追跡ができない
# -> 他の環境で再現できない

# OK: Dockerfile やConfigMapで設定を管理
# 設定変更 -> イメージ再ビルド -> 再デプロイ
docker build -t my-app:v2 .
docker stop production-app
docker run -d --name production-app my-app:v2
```

### アンチパターン 3: ホストネットワークを安易に使う

```bash
# NG: 全コンテナを host ネットワークで起動
docker run -d --network host my-app
docker run -d --network host my-db
# -> ポート衝突のリスク
# -> ネットワーク分離がない
# -> セキュリティリスク

# OK: カスタムネットワークを使用
docker network create app-net
docker run -d --network app-net --name app my-app
docker run -d --network app-net --name db my-db
# -> DNS による名前解決
# -> ネットワーク分離
# -> ポート衝突なし
```

### アンチパターン 4: コンテナにデータを直接保存する

```bash
# NG: コンテナ内にデータを保存
docker run -d --name db postgres:16-alpine
# -> コンテナ削除でデータ消失
# -> コンテナ更新でデータ消失

# OK: ボリュームを使用してデータを永続化
docker run -d --name db \
    -v db-data:/var/lib/postgresql/data \
    postgres:16-alpine
# -> コンテナを削除してもデータは保持
# -> コンテナ更新時もデータは維持
```

---

## 13. FAQ

### Q1: `docker run` と `docker create` + `docker start` の違いは何ですか？

**A:** `docker run` は `docker create`（コンテナ作成）と `docker start`（コンテナ起動）を一度に実行する。分離する利点は、起動前にコンテナの設定を確認したり、ネットワーク接続を変更したりできること。実際の開発では `docker run` を使うことがほとんどであり、`create` + `start` は自動化スクリプトで使われることが多い。

### Q2: コンテナが即座に停止してしまうのはなぜですか？

**A:** コンテナはメインプロセス（PID 1）が終了すると自動的に停止する。よくある原因は以下の通り:
- フォアグラウンドプロセスがない（例: デーモンがバックグラウンドで起動しようとする）
- コマンドが即座に完了する（例: `echo` だけ実行）
- アプリケーションがエラーで終了する
`docker logs <container>` でログを確認し、原因を特定する。

### Q3: `-p 8080:80` のどちらがホスト側でどちらがコンテナ側ですか？

**A:** `ホスト:コンテナ` の順番である。`-p 8080:80` の場合、ホストの 8080 番ポートにアクセスすると、コンテナの 80 番ポートに転送される。覚え方は「外から内へ」（左がホスト=外側、右がコンテナ=内側）。ボリュームマウント `-v` も同じ順番で `ホスト:コンテナ` である。

### Q4: コンテナの IP アドレスを調べるには？

**A:** 以下のコマンドで確認できる:

```bash
# IPアドレスの取得
docker inspect --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' my-nginx

# カスタムネットワークの場合
docker inspect --format '{{.NetworkSettings.Networks.my_network.IPAddress}}' my-nginx

# 全ネットワーク情報
docker inspect --format '{{json .NetworkSettings.Networks}}' my-nginx | python3 -m json.tool
```

ただし、コンテナの IP アドレスは動的に変わるため、固定 IP に依存するのは避け、Docker ネットワークの DNS 名前解決（コンテナ名やネットワークエイリアス）を使用することを推奨する。

### Q5: docker run 時に「--init」オプションを使うべきですか？

**A:** `--init` はコンテナ内で `tini` を PID 1 として起動し、シグナルの適切な伝播とゾンビプロセスの回収を行う。アプリケーションが子プロセスを生成する場合や、シグナルハンドリングを正しく実装していない場合に有用である。Node.js や Python のアプリケーションでは `--init` を付けることを推奨する。

### Q6: bridge ネットワークでコンテナ間通信ができないのはなぜですか？

**A:** デフォルトの bridge ネットワークでは DNS による名前解決ができない。コンテナ間通信には、カスタムネットワークを作成して使用する必要がある。カスタムネットワークでは、コンテナ名による DNS 解決が自動的に有効になる。

---

## 14. まとめ

| 項目 | ポイント |
|---|---|
| イメージとコンテナ | イメージは読み取り専用テンプレート、コンテナは実行インスタンス |
| docker run | `-d`(デタッチ), `-it`(インタラクティブ), `--rm`(自動削除) |
| ポートマッピング | `-p ホスト:コンテナ` でネットワークを接続 |
| ボリューム | バインドマウント(開発用)、名前付きボリューム(永続化) |
| ネットワーク | カスタムネットワークで DNS 名前解決、コンテナ間通信 |
| ログ | `docker logs -f` でリアルタイム追跡、ログドライバで転送 |
| exec | `docker exec -it` で実行中コンテナに接続 |
| リソース制限 | `--memory`, `--cpus` で制限、`docker stats` で監視 |
| ヘルスチェック | `--health-cmd` でコンテナの健全性を監視 |
| クリーンアップ | `docker system prune` で一括削除 |

---

## 次に読むべきガイド

- [03-image-management.md](./03-image-management.md) -- イメージの管理とレジストリ
- [../01-dockerfile/00-dockerfile-basics.md](../01-dockerfile/00-dockerfile-basics.md) -- Dockerfile の基礎
- [../02-compose/00-compose-basics.md](../02-compose/00-compose-basics.md) -- Docker Compose の基礎

---

## 参考文献

1. **Docker Documentation - docker run** https://docs.docker.com/reference/cli/docker/container/run/ -- `docker run` の全オプションリファレンス。
2. **Docker Documentation - Manage data in Docker** https://docs.docker.com/storage/ -- ボリューム、バインドマウント、tmpfs の詳細な解説。
3. **Docker Documentation - Configure logging drivers** https://docs.docker.com/config/containers/logging/ -- ログドライバの設定と各ドライバの特徴。
4. **Docker Documentation - Networking overview** https://docs.docker.com/network/ -- Docker ネットワークの仕組みとドライバの解説。
5. **Docker Documentation - Resource constraints** https://docs.docker.com/config/containers/resource_constraints/ -- メモリ、CPU 等のリソース制限の詳細。
6. **Docker Documentation - Healthcheck** https://docs.docker.com/reference/dockerfile/#healthcheck -- ヘルスチェックの設定方法と活用パターン。
