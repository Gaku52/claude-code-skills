# Docker 基本操作

> イメージの取得からコンテナの起動・停止・削除、ログ確認、コンテナ内操作まで、Docker の日常的な操作を体系的に学ぶ。

---

## この章で学ぶこと

1. **イメージとコンテナの関係**を理解し、ライフサイクル全体を把握する
2. **docker run の主要オプション**を使いこなし、目的に応じたコンテナ起動ができる
3. **ログ確認・コンテナ内操作・リソース管理**の実践的なスキルを身につける

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

# フィルタリング
docker ps --filter "status=exited"
docker ps --filter "name=my-"
docker ps --filter "label=env=production"
```

### 5.2 停止と削除

```bash
# コンテナの停止（SIGTERM -> 10秒後 SIGKILL）
docker stop my-nginx

# タイムアウトを指定して停止
docker stop -t 30 my-nginx

# 強制停止（SIGKILL）
docker kill my-nginx

# コンテナの削除
docker rm my-nginx

# 停止と削除を一度に
docker rm -f my-nginx

# 停止中の全コンテナを削除
docker container prune

# 確認なしで削除
docker container prune -f
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
```

---

## 6. ログ管理

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

# 特定時刻までのログ
docker logs --until "2024-01-15T12:00:00" my-nginx

# 組み合わせ
docker logs -f --tail 50 -t my-nginx
```

### ログドライバ

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

---

## 7. コンテナ内操作 (exec)

```bash
# 実行中のコンテナでコマンドを実行
docker exec my-nginx ls /etc/nginx

# インタラクティブシェルで接続
docker exec -it my-nginx /bin/sh

# 特定のユーザーで実行
docker exec -u root my-nginx whoami

# 環境変数を設定して実行
docker exec -e MY_VAR=hello my-nginx env

# 作業ディレクトリを指定
docker exec -w /etc/nginx my-nginx ls

# バックグラウンドでコマンドを実行
docker exec -d my-nginx touch /tmp/marker
```

### exec vs run の違い

```bash
# docker exec: 既存の実行中コンテナ内でコマンド実行
docker exec my-nginx cat /etc/nginx/nginx.conf
# -> my-nginx コンテナ内でファイルを表示

# docker run: 新しいコンテナを作成して実行
docker run --rm nginx:alpine cat /etc/nginx/nginx.conf
# -> 新しいコンテナを作成し、表示後に削除
```

---

## 8. リソース監視

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
```

### リソース制限の設定

```bash
# メモリ制限
docker run -d --name limited-app \
    --memory=256m \
    --memory-swap=512m \
    nginx:alpine

# CPU制限
docker run -d --name cpu-limited \
    --cpus=1.5 \
    nginx:alpine

# 実行中のコンテナのリソース制限を変更
docker update --memory=512m --cpus=2.0 limited-app
```

---

## 9. クリーンアップ

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
| `--env` | `-e` | 環境変数設定 | `-e KEY=val` |
| `--env-file` | | envファイル読み込み | `--env-file .env` |
| `--network` | | ネットワーク指定 | `--network my-net` |
| `--memory` | `-m` | メモリ制限 | `-m 256m` |
| `--cpus` | | CPU制限 | `--cpus 1.5` |
| `--restart` | | 再起動ポリシー | `--restart unless-stopped` |
| `--platform` | | プラットフォーム指定 | `--platform linux/amd64` |

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
```

---

## 10. アンチパターン

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

---

## 11. FAQ

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

---

## 12. まとめ

| 項目 | ポイント |
|---|---|
| イメージとコンテナ | イメージは読み取り専用テンプレート、コンテナは実行インスタンス |
| docker run | `-d`(デタッチ), `-it`(インタラクティブ), `--rm`(自動削除) |
| ポートマッピング | `-p ホスト:コンテナ` でネットワークを接続 |
| ボリューム | バインドマウント(開発用)、名前付きボリューム(永続化) |
| ログ | `docker logs -f` でリアルタイム追跡 |
| exec | `docker exec -it` で実行中コンテナに接続 |
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
