# コンテナ技術概要

> 仮想マシンとコンテナの違いを理解し、Docker とコンテナエコシステムの全体像を把握するための入門ガイド。

---

## この章で学ぶこと

1. **仮想化とコンテナ化の本質的な違い**を理解し、それぞれの適用領域を判断できる
2. **Docker の歴史と OCI 標準**を知り、コンテナエコシステムの全体像を把握する
3. **コンテナのユースケース**を理解し、自分のプロジェクトへの適用を検討できる

---

## 1. 仮想化とコンテナ化

### 1.1 従来の仮想化（ハイパーバイザ型）

仮想マシン（VM）は、ハイパーバイザ上にゲスト OS を丸ごと起動する技術である。各 VM は独立したカーネルを持ち、完全なアイソレーションを提供する。

```
+---------------------------------------------+
|              ホスト OS                        |
+---------------------------------------------+
|            ハイパーバイザ                      |
+----------+----------+----------+------------+
|  VM 1    |  VM 2    |  VM 3    |            |
| +------+ | +------+ | +------+ |            |
| |アプリ | | |アプリ | | |アプリ | |            |
| +------+ | +------+ | +------+ |            |
| | Bins | | | Bins | | | Bins | |            |
| +------+ | +------+ | +------+ |            |
| |ゲストOS| | |ゲストOS| | |ゲストOS| |            |
| +------+ | +------+ | +------+ |            |
+----------+----------+----------+------------+
```

### 1.2 コンテナ化

コンテナはホスト OS のカーネルを共有し、プロセスレベルでアイソレーションを実現する。ゲスト OS が不要なため、起動が高速でリソース効率が高い。

```
+---------------------------------------------+
|              ホスト OS カーネル                 |
+---------------------------------------------+
|          コンテナランタイム (Docker)            |
+----------+----------+----------+------------+
| コンテナ1 | コンテナ2 | コンテナ3 |            |
| +------+ | +------+ | +------+ |            |
| |アプリ | | |アプリ | | |アプリ | |            |
| +------+ | +------+ | +------+ |            |
| | Bins | | | Bins | | | Bins | |            |
| +------+ | +------+ | +------+ |            |
+----------+----------+----------+------------+
  ゲスト OS なし ─ カーネルを共有
```

### 1.3 Linux カーネル技術

コンテナの基盤となる Linux カーネル技術は以下の 2 つである。

```
+---------------------------------------------------+
|               Linux カーネル                        |
|                                                   |
|  +-------------------+  +----------------------+  |
|  |   namespaces      |  |      cgroups         |  |
|  |                   |  |                      |  |
|  |  - pid namespace  |  |  - CPU 制限          |  |
|  |  - net namespace  |  |  - メモリ制限         |  |
|  |  - mnt namespace  |  |  - I/O 制限          |  |
|  |  - uts namespace  |  |  - プロセス数制限      |  |
|  |  - ipc namespace  |  |                      |  |
|  |  - user namespace |  |                      |  |
|  +-------------------+  +----------------------+  |
|                                                   |
|  namespaces = 見える範囲の制限（アイソレーション）    |
|  cgroups    = 使える量の制限（リソース制御）         |
+---------------------------------------------------+
```

**namespaces の種類:**

```bash
# PID namespace - プロセスIDの分離
# コンテナ内ではPID 1から始まる
docker run --rm alpine ps aux
# PID   USER     COMMAND
#   1   root     ps aux

# Network namespace - ネットワークスタックの分離
docker run --rm alpine ip addr
# コンテナ固有のネットワークインターフェースが表示される

# Mount namespace - ファイルシステムの分離
docker run --rm alpine ls /
# コンテナ独自のルートファイルシステム

# UTS namespace - ホスト名の分離
docker run --rm alpine hostname
# コンテナ固有のホスト名

# User namespace - ユーザーIDの分離
docker run --rm alpine id
# uid=0(root) - コンテナ内のrootはホストのrootではない
```

**cgroups によるリソース制限:**

```bash
# メモリを256MBに制限
docker run --memory=256m --rm alpine free -m

# CPUを1コアに制限
docker run --cpus=1.0 --rm alpine cat /proc/cpuinfo

# メモリとCPUの両方を制限
docker run --memory=512m --cpus=2.0 --rm nginx

# リソース使用状況の確認
docker stats --no-stream
```

---

## 2. 仮想マシン vs コンテナ 比較

### 比較表 1: 技術的特性

| 特性 | 仮想マシン (VM) | コンテナ |
|---|---|---|
| アイソレーション | ハードウェアレベル | プロセスレベル |
| OS | 各VMにゲストOS | ホストOSカーネル共有 |
| 起動時間 | 数分 | 数秒〜数百ミリ秒 |
| サイズ | GB単位 | MB単位 |
| パフォーマンス | ハイパーバイザのオーバーヘッド | ほぼネイティブ |
| 密度 | 1ホストに数十VM | 1ホストに数百〜数千コンテナ |
| セキュリティ | 強い分離 | カーネル共有のリスク |
| ポータビリティ | VMイメージが巨大 | コンテナイメージが軽量 |

### 比較表 2: 適用場面

| ユースケース | 推奨 | 理由 |
|---|---|---|
| マイクロサービス | コンテナ | 軽量・高速デプロイ |
| レガシーOS対応 | VM | 異なるカーネルが必要 |
| 開発環境の統一 | コンテナ | 再現性が高い |
| マルチテナント | VM | 強い分離が必要 |
| CI/CD パイプライン | コンテナ | 起動が高速 |
| デスクトップ仮想化 | VM | GUI・ドライバ対応 |
| バッチ処理 | コンテナ | スケールが容易 |
| セキュリティテスト | VM | 完全な分離 |

---

## 3. Docker の歴史

### 3.1 年表

```
2008  LXC (Linux Containers) リリース
  |
2013  Docker 0.1 リリース (dotCloud社)
  |   - LXC をラップした使いやすい CLI
  |
2014  Docker 1.0 GA
  |   - libcontainer で LXC 依存を脱却
  |
2015  OCI (Open Container Initiative) 設立
  |   - Docker, CoreOS, Google 等が参加
  |   - コンテナの標準仕様を策定
  |
2016  Docker 1.12 - Swarm Mode 統合
  |
2017  containerd を CNCF に寄贈
  |   - Moby プロジェクト開始
  |   - Kubernetes が CRI 対応
  |
2019  Docker Desktop 有料化方針
  |
2020  Kubernetes が dockershim を非推奨化
  |   - containerd / CRI-O に移行
  |
2021  Docker Desktop ライセンス変更
  |   (大企業は有料サブスクリプション)
  |
2023  Docker Scout (脆弱性スキャン)
  |   Docker Init (Dockerfile自動生成)
  |
2024  Docker Compose Watch
      Docker Build Cloud
```

### 3.2 Docker のアーキテクチャ

```bash
# Docker のバージョン確認（クライアントとサーバー）
docker version

# 出力例:
# Client:
#  Version:           24.0.7
#  API version:       1.43
#
# Server:
#  Engine:
#   Version:          24.0.7
#   containerd:       1.7.6
#   runc:             1.1.10

# Docker システム情報の確認
docker info

# Docker が使用しているストレージドライバの確認
docker info --format '{{.Driver}}'
# overlay2
```

---

## 4. OCI 標準

OCI（Open Container Initiative）は、コンテナの業界標準を定める組織である。

### 4.1 OCI の 3 つの仕様

```
+--------------------------------------------------+
|           OCI (Open Container Initiative)         |
|                                                  |
|  +-------------------------------------------+  |
|  | Runtime Specification (runtime-spec)       |  |
|  | - コンテナの実行方法を定義                    |  |
|  | - 実装例: runc, crun, youki                 |  |
|  +-------------------------------------------+  |
|                                                  |
|  +-------------------------------------------+  |
|  | Image Specification (image-spec)           |  |
|  | - コンテナイメージのフォーマットを定義          |  |
|  | - レイヤー構造、メタデータ                    |  |
|  +-------------------------------------------+  |
|                                                  |
|  +-------------------------------------------+  |
|  | Distribution Specification (dist-spec)     |  |
|  | - イメージの配布方法を定義                    |  |
|  | - レジストリ API                            |  |
|  +-------------------------------------------+  |
+--------------------------------------------------+
```

### 4.2 OCI 準拠のツール群

```bash
# runc - OCI ランタイムリファレンス実装
runc --version

# Podman - Docker互換のデーモンレスコンテナエンジン
podman run --rm alpine echo "Hello from Podman"

# Buildah - OCI イメージビルドツール
buildah from alpine

# Skopeo - コンテナイメージ操作ツール
skopeo inspect docker://alpine:latest
```

---

## 5. コンテナのユースケース

### 5.1 マイクロサービスアーキテクチャ

```bash
# 各サービスを独立したコンテナとして実行
docker run -d --name api-gateway -p 8080:8080 api-gateway:v1
docker run -d --name user-service -p 8081:8081 user-service:v1
docker run -d --name order-service -p 8082:8082 order-service:v1
docker run -d --name payment-service -p 8083:8083 payment-service:v1

# ネットワークで接続
docker network create microservices
docker network connect microservices api-gateway
docker network connect microservices user-service
docker network connect microservices order-service
docker network connect microservices payment-service
```

### 5.2 開発環境の統一

```yaml
# docker-compose.yml による開発環境定義
version: "3.9"
services:
  app:
    build: .
    volumes:
      - .:/app
    ports:
      - "3000:3000"
  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: devpass
  redis:
    image: redis:7-alpine
```

### 5.3 CI/CD パイプライン

```yaml
# GitHub Actions でのコンテナ活用例
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t myapp:test .
      - run: docker run --rm myapp:test npm test
```

---

## 6. アンチパターン

### アンチパターン 1: コンテナを VM のように使う

```bash
# NG: 1コンテナに複数サービスを詰め込む
# SSH, cron, アプリ, DB を全部1コンテナに入れる
docker run -d my-monolith-container
# -> メンテナンス困難、スケール不可

# OK: 1コンテナ1プロセスの原則
docker run -d --name app my-app
docker run -d --name db postgres:16
docker run -d --name cache redis:7
# -> 個別にスケール・更新・監視が可能
```

### アンチパターン 2: latest タグへの依存

```bash
# NG: バージョンを指定しない
docker run -d nginx:latest
# -> 再現性がない。ある日突然動かなくなる可能性

# OK: 具体的なバージョンを指定
docker run -d nginx:1.25.3-alpine
# -> いつ実行しても同じイメージが使われる
```

### アンチパターン 3: ホストネットワークの安易な使用

```bash
# NG: ホストネットワークを常用
docker run --network host my-app
# -> ポート衝突、セキュリティリスク

# OK: ブリッジネットワークでポートマッピング
docker run -p 8080:80 my-app
# -> 明示的なポート制御、分離されたネットワーク
```

---

## 7. FAQ

### Q1: コンテナは仮想マシンの上位互換ですか？

**A:** いいえ。コンテナと VM は相互補完的な技術である。コンテナはホスト OS のカーネルを共有するため、異なる OS カーネルを必要とする場面（例: Linux ホスト上で Windows アプリを動かす）には VM が必要である。また、強いセキュリティ分離が求められるマルチテナント環境では VM が適している。多くのクラウド環境では、VM の中でコンテナを動かすハイブリッド構成が採用されている。

### Q2: Docker と Podman はどう違いますか？

**A:** 最大の違いはアーキテクチャである。Docker はデーモン（dockerd）が常駐する クライアント-サーバー型 だが、Podman はデーモンレスで各コンテナが独立したプロセスとして動く。Podman は OCI 準拠で Docker CLI と高い互換性がある。rootless 実行がデフォルトでサポートされている点もセキュリティ上の利点である。ただし、Docker Compose のような統合ツールチェーンは Docker の方が成熟している。

### Q3: Windows や macOS でコンテナはネイティブに動きますか？

**A:** Linux コンテナは Linux カーネルの機能（namespaces, cgroups）に依存するため、macOS や Windows ではネイティブに動作しない。Docker Desktop は内部で軽量な Linux VM を起動し、その中でコンテナを実行している。macOS では Apple の Virtualization.framework、Windows では WSL2（Windows Subsystem for Linux 2）が使われる。Windows コンテナ（Windows Server 上）は Windows カーネルでネイティブに動作する。

### Q4: コンテナのセキュリティは VM より弱いのですか？

**A:** カーネル共有によるリスクは存在するが、適切な対策により実用上十分なセキュリティを確保できる。具体的には、rootless コンテナの使用、seccomp プロファイル、AppArmor/SELinux、read-only ファイルシステム、最小権限の原則を適用する。さらに、gVisor や Kata Containers のようなサンドボックスランタイムを使えば、VM に近いレベルの分離を実現できる。

---

## 8. まとめ

| 項目 | ポイント |
|---|---|
| コンテナとは | ホストOSカーネルを共有するプロセスレベルの仮想化 |
| 基盤技術 | Linux namespaces（分離）+ cgroups（リソース制限） |
| VM との違い | ゲストOS不要で高速・軽量、ただし分離レベルは異なる |
| Docker の位置づけ | コンテナエコシステムの事実上の標準ツールチェーン |
| OCI 標準 | runtime-spec, image-spec, distribution-spec の3仕様 |
| 主なユースケース | マイクロサービス、開発環境統一、CI/CD |
| 設計原則 | 1コンテナ1プロセス、イミュータブル、バージョン固定 |

---

## 次に読むべきガイド

- [01-docker-install.md](./01-docker-install.md) -- Docker のインストールと初期設定
- [02-docker-basics.md](./02-docker-basics.md) -- Docker の基本操作
- [../01-dockerfile/00-dockerfile-basics.md](../01-dockerfile/00-dockerfile-basics.md) -- Dockerfile の基礎

---

## 参考文献

1. **Docker Documentation - Get Started** https://docs.docker.com/get-started/ -- Docker 公式のチュートリアル。コンテナの基本概念から実践まで網羅。
2. **Open Container Initiative (OCI)** https://opencontainers.org/ -- OCI の公式サイト。runtime-spec, image-spec, distribution-spec の仕様書を公開。
3. **Linux man pages - namespaces(7)** https://man7.org/linux/man-pages/man7/namespaces.7.html -- Linux namespaces の公式ドキュメント。コンテナの基盤技術を深く理解するために参照。
4. **Kubernetes Documentation - Container Runtimes** https://kubernetes.io/docs/setup/production-environment/container-runtimes/ -- containerd, CRI-O 等のコンテナランタイムの比較と設定方法。
