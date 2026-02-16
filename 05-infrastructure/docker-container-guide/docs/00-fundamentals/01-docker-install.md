# Docker インストールガイド

> Docker Desktop と Docker Engine のインストール方法、初期設定、動作確認までを網羅する実践的セットアップガイド。

---

## この章で学ぶこと

1. **各 OS に最適な Docker のインストール方法**を選択し、確実にセットアップできる
2. **Docker Desktop と Docker Engine の違い**を理解し、用途に応じて使い分けられる
3. **インストール後の初期設定と動作確認**を完了し、開発を開始できる状態にする
4. **Docker のアーキテクチャ**を理解し、トラブル発生時に適切に対処できる
5. **ネットワークとストレージの初期設定**を最適化し、安定した開発環境を構築できる

---

## 1. Docker Desktop vs Docker Engine

### 1.1 製品比較

```
+------------------------------------------------------------+
|                Docker のプロダクトライン                      |
|                                                            |
|  +------------------------+  +-------------------------+  |
|  |   Docker Desktop       |  |   Docker Engine         |  |
|  |                        |  |                         |  |
|  |  - macOS / Windows     |  |  - Linux サーバー向け    |  |
|  |  - GUI ダッシュボード   |  |  - CLI のみ             |  |
|  |  - VM 内蔵             |  |  - ネイティブ動作        |  |
|  |  - Docker Compose 同梱 |  |  - 手動でCompose導入    |  |
|  |  - Kubernetes 同梱     |  |  - 軽量・高速            |  |
|  |  - 自動アップデート     |  |  - 手動アップデート      |  |
|  |                        |  |                         |  |
|  |  個人開発/小企業: 無料  |  |  完全無料 (OSS)         |  |
|  |  大企業: 有料           |  |                         |  |
|  +------------------------+  +-------------------------+  |
+------------------------------------------------------------+
```

### 比較表 1: Docker Desktop vs Docker Engine

| 項目 | Docker Desktop | Docker Engine |
|---|---|---|
| 対応 OS | macOS, Windows, Linux | Linux のみ |
| ライセンス | 大企業は有料（250人以上/年商$10M以上） | 無料 (Apache 2.0) |
| GUI | あり（ダッシュボード） | なし |
| Compose | 同梱 | 別途インストール（plugin） |
| Kubernetes | 同梱（ワンクリック有効化） | 別途インストール |
| VM | 内蔵（macOS/Windows） | 不要 |
| リソース消費 | 高い（VM分） | 低い |
| 用途 | ローカル開発 | 本番サーバー、CI/CD |
| Extensions | マーケットプレイスから追加可能 | なし |
| Dev Environments | サポート | なし |
| Volume Management | GUI で管理可能 | CLI のみ |

### 1.2 Docker のアーキテクチャ

```
+--------------------------------------------------------------------+
|                     Docker のアーキテクチャ                           |
|                                                                    |
|  +-------------------+     +------------------------------+       |
|  |   Docker Client   |     |   Docker Host (デーモン)       |       |
|  |                   |     |                              |       |
|  |  docker build     | --> |  +------------------------+  |       |
|  |  docker pull      | API |  |     Docker Daemon       |  |       |
|  |  docker run       | --> |  |     (dockerd)           |  |       |
|  |  docker compose   |     |  +------|-------|----------+  |       |
|  +-------------------+     |         |       |             |       |
|                            |    +----v--+ +--v--------+   |       |
|                            |    |Images | |Containers |   |       |
|                            |    +-------+ +-----------+   |       |
|                            |    |Networks| |Volumes    |   |       |
|                            |    +-------+ +-----------+   |       |
|                            +------------------------------+       |
|                                         |                         |
|                                         | docker push/pull        |
|                                         v                         |
|                            +------------------------------+       |
|                            |       Registry               |       |
|                            |   (Docker Hub, GHCR, ECR)    |       |
|                            +------------------------------+       |
+--------------------------------------------------------------------+
```

Docker は **クライアント-サーバーアーキテクチャ** を採用している。Docker Client（CLI）が Docker Daemon（dockerd）に REST API 経由でコマンドを送信し、Daemon がコンテナの作成・実行・管理を行う。この仕組みを理解しておくと、接続エラーやパーミッション問題のトラブルシューティングに役立つ。

### 1.3 コンテナランタイムの選択肢

Docker 以外にもコンテナランタイムは存在する。プロジェクトの要件に応じて適切なツールを選択する。

| ランタイム | 特徴 | 用途 |
|---|---|---|
| Docker Engine | 最も広く使われている | 開発・本番全般 |
| Podman | デーモンレス、rootless | RHEL/Fedora 環境 |
| containerd | 軽量ランタイム | Kubernetes CRI |
| CRI-O | Kubernetes 専用 | Kubernetes ノード |
| nerdctl | containerd の CLI | Docker CLI 互換 |
| Colima | macOS 用軽量 VM | Docker Desktop 代替 |
| Rancher Desktop | GUI 付き代替ツール | Docker Desktop 代替 |
| OrbStack | macOS 専用高速 VM | Docker Desktop 代替（高速） |

---

## 2. macOS へのインストール

### 2.1 Docker Desktop (推奨)

```bash
# 方法1: 公式サイトからダウンロード
# https://www.docker.com/products/docker-desktop/
# Apple Silicon (M1/M2/M3/M4) と Intel 版を選択

# 方法2: Homebrew でインストール
brew install --cask docker

# インストール後、アプリケーションから Docker.app を起動
open /Applications/Docker.app

# 初回起動時にヘルパーツールのインストール許可を求められる
# パスワードを入力して許可する
```

### 2.2 Colima (Docker Desktop の代替)

Docker Desktop のライセンスが問題になる場合や、より軽量な環境が必要な場合は Colima を利用できる。

```bash
# Colima のインストール
brew install colima docker docker-compose docker-credential-helper

# Colima の起動（デフォルト設定: 2 CPU, 2 GB メモリ）
colima start

# カスタム設定で起動
colima start --cpu 4 --memory 8 --disk 60

# Apple Silicon で x86_64 エミュレーション
colima start --arch x86_64

# Kubernetes 付きで起動
colima start --kubernetes

# 状態確認
colima status

# 停止
colima stop

# 削除
colima delete
```

### 2.3 OrbStack (高速な代替)

OrbStack は macOS 専用の Docker Desktop 代替ツールで、起動速度とリソース効率に優れている。

```bash
# OrbStack のインストール
brew install --cask orbstack

# インストール後、docker コマンドが自動的に OrbStack に接続される
docker version
# Client: OrbStack
# Server: Docker Engine via OrbStack

# Docker Desktop との切り替え
# OrbStack の設定から Docker Desktop との共存設定が可能
```

### 2.4 動作確認

```bash
# Docker デーモンが起動しているか確認
docker version

# 期待される出力:
# Client:
#  Cloud integration: v1.0.35
#  Version:           24.0.7
# Server: Docker Desktop 4.x.x
#  Engine:
#   Version:          24.0.7

# Hello World コンテナを実行
docker run --rm hello-world

# 期待される出力:
# Hello from Docker!
# This message shows that your installation appears to be working correctly.

# Docker Compose のバージョン確認
docker compose version
# Docker Compose version v2.x.x

# Docker の詳細情報を確認
docker info
# Server Version, Storage Driver, OS/Arch 等が表示される

# BuildKit が有効か確認
docker buildx version
# github.com/docker/buildx v0.x.x
```

### 2.5 Apple Silicon (ARM64) の注意点

```bash
# ARM64 イメージが存在するか確認
docker manifest inspect --verbose nginx:alpine | grep architecture
# "architecture": "arm64"

# AMD64 イメージを強制的に使う場合（互換性問題時）
docker run --platform linux/amd64 --rm nginx:alpine nginx -v

# マルチプラットフォームビルドの準備
docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap

# マルチプラットフォームビルドの実行
docker buildx build --platform linux/amd64,linux/arm64 \
    -t my-app:v1.0.0 --push .

# Rosetta 2 エミュレーションの確認
# Docker Desktop > Settings > General > Use Rosetta for x86_64/amd64 emulation
# Rosetta を有効にすると amd64 イメージの実行が高速化される

# QEMU ベースのエミュレーションを使用する場合
docker run --platform linux/amd64 --rm alpine uname -m
# x86_64

# プラットフォーム情報の確認
docker run --rm alpine uname -m
# aarch64 (ARM64 ネイティブの場合)
```

### 2.6 macOS での VirtioFS 設定

macOS では Docker がVM 上で動作するため、ファイルシステムのパフォーマンスが重要になる。

```bash
# VirtioFS の確認（Docker Desktop > Settings > General）
# "Choose file sharing implementation for your containers"
# -> VirtioFS を選択（推奨）

# VirtioFS vs gRPC FUSE vs osxfs のパフォーマンス比較
# ベンチマーク例: Node.js プロジェクトの npm install
# osxfs:    120秒
# gRPC FUSE: 80秒
# VirtioFS:  45秒

# ファイル同期の最適化設定
# docker-compose.yml で consistency オプションを指定
# services:
#   app:
#     volumes:
#       - ./src:/app/src:cached    # ホスト -> コンテナの同期は遅延OK
#       - ./logs:/app/logs:delegated  # コンテナ -> ホストの同期は遅延OK
```

---

## 3. Windows へのインストール

### 3.1 前提条件: WSL2

```
+----------------------------------------------------+
|              Windows での Docker 構成                |
|                                                    |
|  +----------------------------------------------+ |
|  |            Docker Desktop                     | |
|  |  +------+  +-----+  +----+  +-------------+ | |
|  |  | CLI  |  | GUI |  | API|  | Compose     | | |
|  |  +------+  +-----+  +----+  +-------------+ | |
|  +---------------------|------------------------+ |
|                        v                          |
|  +----------------------------------------------+ |
|  |              WSL2 (Linux カーネル)             | |
|  |  +----------------------------------------+  | |
|  |  |      Docker Engine (Linux)             |  | |
|  |  |  +----------+  +------------------+   |  | |
|  |  |  |containerd|  | イメージストレージ  |   |  | |
|  |  |  +----------+  +------------------+   |  | |
|  |  +----------------------------------------+  | |
|  +----------------------------------------------+ |
|              Windows ホスト OS                     |
+----------------------------------------------------+
```

```powershell
# PowerShell (管理者) で WSL2 を有効化
wsl --install

# 特定のディストリビューションを指定してインストール
wsl --install -d Ubuntu-22.04

# WSL2 がデフォルトか確認
wsl --set-default-version 2

# WSL のバージョン確認
wsl --list --verbose
# NAME                   STATE           VERSION
# * Ubuntu               Running         2

# 利用可能なディストリビューション一覧
wsl --list --online

# WSL2 のメモリ制限設定（推奨）
# %USERPROFILE%\.wslconfig を作成
# [wsl2]
# memory=8GB
# processors=4
# swap=2GB
# localhostForwarding=true
```

### 3.2 Docker Desktop インストール

```powershell
# 方法1: 公式サイトからダウンロード
# https://www.docker.com/products/docker-desktop/

# 方法2: winget でインストール
winget install Docker.DockerDesktop

# 方法3: Chocolatey でインストール
choco install docker-desktop

# インストール後、再起動が必要な場合がある
# Settings > General > "Use the WSL 2 based engine" にチェック
```

### 3.3 Windows 固有の設定

```powershell
# WSL2 統合の設定
# Docker Desktop > Settings > Resources > WSL integration
# 使用する WSL ディストリビューションを選択

# Windows ファイアウォールの設定
# Docker Desktop はインストール時に自動でファイアウォールルールを追加する
# 問題がある場合は手動で Docker Desktop Backend を許可

# Windows Defender の除外設定（パフォーマンス改善）
# 以下のパスを除外に追加:
# - C:\ProgramData\Docker
# - C:\Users\<username>\AppData\Local\Docker
# - WSL2 のファイルシステム（\\wsl$）

# PowerShell で除外を追加
Add-MpPreference -ExclusionPath "C:\ProgramData\Docker"
Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\Docker"
```

### 3.4 動作確認

```powershell
# PowerShell で確認
docker version
docker run --rm hello-world

# WSL2 ディストリビューション内からも使えることを確認
wsl -d Ubuntu -e docker version

# Docker Compose の確認
docker compose version

# ボリュームのパス変換確認
# Windows パスと Linux パスの変換
docker run --rm -v "C:\Users\user\project:/app" alpine ls /app
# または WSL2 内から
docker run --rm -v "/mnt/c/Users/user/project:/app" alpine ls /app

# Windows コンテナの切り替え（必要な場合）
# Docker Desktop のトレイアイコンから "Switch to Windows containers" を選択
# ※通常は Linux コンテナを使用する
```

### 3.5 WSL2 のトラブルシューティング

```powershell
# WSL2 が起動しない場合
# 1. 仮想化機能が有効か確認（BIOS/UEFI で設定）
systeminfo | findstr /i "Hyper-V"

# 2. WSL2 カーネルの更新
wsl --update

# 3. WSL2 のリセット
wsl --shutdown
wsl

# 4. Docker デーモンが起動しない場合
# Docker Desktop のログを確認
# %LOCALAPPDATA%\Docker\log\

# 5. DNS 解決の問題
# WSL2 内で /etc/resolv.conf を確認
wsl -d Ubuntu -e cat /etc/resolv.conf
# nameserver が設定されていることを確認

# 6. メモリ消費が大きい場合
# .wslconfig でメモリ上限を設定
# [wsl2]
# memory=4GB

# 設定反映
wsl --shutdown
```

---

## 4. Linux (Ubuntu/Debian) へのインストール

### 4.1 Docker Engine インストール

```bash
# 古いバージョンの削除
sudo apt-get remove docker docker-engine docker.io containerd runc

# 必要なパッケージのインストール
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Docker 公式 GPG キーの追加
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# リポジトリの追加
echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker Engine のインストール
sudo apt-get update
sudo apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
```

### 4.2 Linux (RHEL/Fedora) へのインストール

```bash
# 古いバージョンの削除
sudo dnf remove docker docker-client docker-client-latest \
    docker-common docker-latest docker-latest-logrotate \
    docker-logrotate docker-engine

# リポジトリの追加
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo \
    https://download.docker.com/linux/fedora/docker-ce.repo

# Docker Engine のインストール
sudo dnf install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# サービスの起動と有効化
sudo systemctl start docker
sudo systemctl enable docker
```

### 4.3 Linux (Arch Linux) へのインストール

```bash
# Docker のインストール
sudo pacman -S docker docker-compose docker-buildx

# サービスの起動と有効化
sudo systemctl start docker
sudo systemctl enable docker

# ユーザーを docker グループに追加
sudo usermod -aG docker $USER
newgrp docker
```

### 4.4 Linux (Alpine) へのインストール

```bash
# Docker のインストール
sudo apk add docker docker-compose docker-cli-buildx

# サービスの起動と有効化
sudo rc-update add docker boot
sudo service docker start

# ユーザーを docker グループに追加
sudo addgroup $USER docker
```

### 4.5 インストール後の設定

```bash
# docker グループにユーザーを追加（sudo なしで実行可能にする）
sudo usermod -aG docker $USER

# グループ変更を反映（再ログインまたは以下を実行）
newgrp docker

# 確認
docker run --rm hello-world
# sudo なしで実行できれば成功

# Docker サービスのステータス確認
sudo systemctl status docker

# Docker サービスの自動起動設定
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
```

### 4.6 特定バージョンのインストール

```bash
# 利用可能なバージョンの一覧
apt-cache madison docker-ce
# docker-ce | 5:24.0.7-1~ubuntu.22.04~jammy | ...
# docker-ce | 5:24.0.6-1~ubuntu.22.04~jammy | ...

# 特定バージョンのインストール
VERSION_STRING=5:24.0.7-1~ubuntu.22.04~jammy
sudo apt-get install -y \
    docker-ce=$VERSION_STRING \
    docker-ce-cli=$VERSION_STRING \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# バージョン固定（自動アップデートを防止）
sudo apt-mark hold docker-ce docker-ce-cli

# 固定を解除
sudo apt-mark unhold docker-ce docker-ce-cli
```

### 4.7 便利スクリプト（非推奨だが便利）

```bash
# Docker 公式の convenience script（テスト・開発環境向け）
# 本番環境では上記の手動インストールを推奨
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# DRY RUN（実際にはインストールしない）
DRY_RUN=1 sh ./get-docker.sh

# 注意: convenience script は以下の理由で本番非推奨
# - 既存の Docker 設定を上書きする可能性
# - セキュリティレビューなしで root 権限で実行
# - バージョンの細かい制御ができない
```

---

## 5. 初期設定

### 5.1 Docker デーモン設定

```bash
# /etc/docker/daemon.json を作成・編集
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-address-pools": [
    {
      "base": "172.17.0.0/16",
      "size": 24
    }
  ],
  "dns": ["8.8.8.8", "8.8.4.4"],
  "features": {
    "buildkit": true
  }
}
EOF

# 設定の反映
sudo systemctl restart docker
```

### 5.2 daemon.json の詳細設定

```bash
# 本番環境向けの詳細設定例
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "50m",
    "max-file": "5",
    "compress": "true"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-address-pools": [
    {
      "base": "172.17.0.0/12",
      "size": 24
    }
  ],
  "dns": ["8.8.8.8", "8.8.4.4"],
  "dns-search": ["example.com"],
  "bip": "172.17.0.1/16",
  "fixed-cidr": "172.17.0.0/24",
  "features": {
    "buildkit": true
  },
  "live-restore": true,
  "userland-proxy": false,
  "no-new-privileges": true,
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 32768
    },
    "nproc": {
      "Name": "nproc",
      "Hard": 4096,
      "Soft": 2048
    }
  },
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5,
  "max-download-attempts": 5,
  "shutdown-timeout": 15,
  "debug": false,
  "tls": false,
  "insecure-registries": [],
  "registry-mirrors": []
}
EOF

# 設定の検証（デーモンを再起動する前に）
sudo dockerd --validate --config-file /etc/docker/daemon.json

# 設定の反映
sudo systemctl daemon-reload
sudo systemctl restart docker

# 設定が反映されたか確認
docker info
```

### 5.3 daemon.json 設定項目リファレンス

| 設定項目 | 説明 | 推奨値 |
|---|---|---|
| `log-driver` | ログドライバ | `json-file` (デフォルト) |
| `log-opts.max-size` | ログファイルの最大サイズ | `10m` - `50m` |
| `log-opts.max-file` | ログファイルのローテーション数 | `3` - `5` |
| `storage-driver` | ストレージドライバ | `overlay2` |
| `live-restore` | デーモン停止時にコンテナを維持 | `true` (本番) |
| `userland-proxy` | ユーザーランドプロキシ | `false` (パフォーマンス) |
| `no-new-privileges` | 特権エスカレーション防止 | `true` (セキュリティ) |
| `default-ulimits` | コンテナのデフォルト ulimit | プロジェクトに依存 |
| `max-concurrent-downloads` | 同時ダウンロード数 | `10` |
| `insecure-registries` | 非 HTTPS レジストリ | 本番では空 |
| `registry-mirrors` | レジストリミラー | Rate Limit 対策に設定 |
| `debug` | デバッグモード | `false` (本番) |

### 5.4 Docker Desktop の設定 (GUI)

```
+------------------------------------------------------------+
|  Docker Desktop Settings                                   |
|                                                            |
|  General                                                   |
|  [x] Start Docker Desktop when you sign in                |
|  [x] Use the WSL 2 based engine (Windows)                 |
|  [x] Use Virtualization framework (macOS)                 |
|  [x] VirtioFS (macOS, 推奨)                               |
|                                                            |
|  Resources                                                 |
|  +------------------------------------------------------+ |
|  |  CPUs:    [====------]  4 / 10                       | |
|  |  Memory:  [======----]  8 GB / 16 GB                 | |
|  |  Swap:    [==--------]  1 GB                         | |
|  |  Disk:    [========--]  64 GB                        | |
|  +------------------------------------------------------+ |
|                                                            |
|  Docker Engine (daemon.json を直接編集可能)                 |
|  Kubernetes                                                |
|  [x] Enable Kubernetes                                    |
|                                                            |
|  Software Updates                                          |
|  [x] Automatically check for updates                      |
|  [ ] Always download updates                              |
+------------------------------------------------------------+
```

### 5.5 プロキシ環境での設定

企業ネットワーク等でプロキシを使用する場合の設定。

```bash
# Docker デーモンのプロキシ設定
sudo mkdir -p /etc/systemd/system/docker.service.d/
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<'EOF'
[Service]
Environment="HTTP_PROXY=http://proxy.example.com:8080"
Environment="HTTPS_PROXY=http://proxy.example.com:8080"
Environment="NO_PROXY=localhost,127.0.0.1,docker-registry.example.com,.corp"
EOF

# 設定の反映
sudo systemctl daemon-reload
sudo systemctl restart docker

# 設定の確認
sudo systemctl show --property=Environment docker

# Docker クライアント側のプロキシ設定
mkdir -p ~/.docker
cat > ~/.docker/config.json <<'EOF'
{
  "proxies": {
    "default": {
      "httpProxy": "http://proxy.example.com:8080",
      "httpsProxy": "http://proxy.example.com:8080",
      "noProxy": "localhost,127.0.0.1,.corp"
    }
  }
}
EOF

# ビルド時のプロキシ設定
docker build \
    --build-arg HTTP_PROXY=http://proxy.example.com:8080 \
    --build-arg HTTPS_PROXY=http://proxy.example.com:8080 \
    --build-arg NO_PROXY=localhost,127.0.0.1 \
    -t my-app .
```

### 5.6 Docker のストレージ設定

```bash
# Docker のデータディレクトリを変更する場合
# デフォルト: /var/lib/docker
# 大容量ストレージに変更したい場合等

# 方法1: daemon.json で指定
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "data-root": "/mnt/docker-data"
}
EOF

# 方法2: 既存データを移行
sudo systemctl stop docker
sudo rsync -aP /var/lib/docker/ /mnt/docker-data/
sudo mv /var/lib/docker /var/lib/docker.bak
sudo ln -s /mnt/docker-data /var/lib/docker
sudo systemctl start docker

# ストレージドライバの確認
docker info | grep "Storage Driver"
# Storage Driver: overlay2

# ディスク使用量の確認
docker system df
# TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
# Images          15        5         4.2GB     2.8GB (66%)
# Containers      8         3         120MB     80MB (66%)
# Local Volumes   12        4         1.5GB     800MB (53%)
# Build Cache     50        0         2.1GB     2.1GB
```

### 比較表 2: Linux ディストリビューション別インストール方法

| ディストリビューション | パッケージマネージャ | リポジトリ設定 | 備考 |
|---|---|---|---|
| Ubuntu 22.04/24.04 | apt | docker.list | 最も安定 |
| Debian 12 (Bookworm) | apt | docker.list | Ubuntu と同様 |
| Fedora 38/39/40 | dnf | docker-ce.repo | SELinux 注意 |
| RHEL 9 / Rocky 9 | dnf | docker-ce.repo | Podman がデフォルト |
| Arch Linux | pacman | 公式リポジトリ | `pacman -S docker` |
| Alpine | apk | community リポジトリ | `apk add docker` |
| openSUSE | zypper | Docker 公式リポ | `zypper install docker` |
| Amazon Linux 2023 | dnf | extras リポジトリ | `dnf install docker` |

---

## 6. 動作確認チェックリスト

### 6.1 基本チェック

```bash
# 1. Docker バージョン確認
docker version
# Client と Server 両方のバージョンが表示されること

# 2. Docker 情報確認
docker info
# Server Version, Storage Driver, OS/Arch が正しいこと

# 3. Hello World 実行
docker run --rm hello-world
# "Hello from Docker!" メッセージが表示されること

# 4. コンテナのライフサイクル確認
docker run -d --name test-nginx -p 8080:80 nginx:alpine
curl http://localhost:8080  # nginx のデフォルトページ
docker stop test-nginx
docker rm test-nginx

# 5. ボリュームの動作確認
docker volume create test-vol
docker run --rm -v test-vol:/data alpine sh -c "echo 'test' > /data/test.txt"
docker run --rm -v test-vol:/data alpine cat /data/test.txt
# "test" と表示されること
docker volume rm test-vol

# 6. Docker Compose の確認
docker compose version
# Docker Compose version v2.x.x

# 7. BuildKit の確認
docker buildx version
# github.com/docker/buildx v0.x.x
```

### 6.2 詳細チェック

```bash
# 8. ネットワーク確認
docker network ls
# NETWORK ID     NAME      DRIVER    SCOPE
# abc123         bridge    bridge    local
# def456         host      host      local
# ghi789         none      null      local

# 9. DNS 解決確認
docker run --rm alpine nslookup google.com
# Name:      google.com
# Address 1: xxx.xxx.xxx.xxx

# 10. イメージのプル確認
docker pull alpine:latest
docker pull nginx:alpine

# 11. コンテナ間通信確認
docker network create test-net
docker run -d --name server --network test-net nginx:alpine
docker run --rm --network test-net alpine \
    wget -qO- http://server:80
docker stop server && docker rm server
docker network rm test-net

# 12. リソース制限の動作確認
docker run --rm --memory=128m --cpus=0.5 alpine \
    sh -c "cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 'cgroup v1'"

# 13. Docker Compose の統合テスト
cat > /tmp/docker-compose-test.yml <<'EOF'
services:
  web:
    image: nginx:alpine
    ports:
      - "8888:80"
  app:
    image: alpine
    command: sleep 30
    depends_on:
      - web
EOF
docker compose -f /tmp/docker-compose-test.yml up -d
docker compose -f /tmp/docker-compose-test.yml ps
docker compose -f /tmp/docker-compose-test.yml down
rm /tmp/docker-compose-test.yml
```

### 6.3 パフォーマンスベンチマーク

```bash
# ディスク I/O テスト
docker run --rm alpine sh -c "
    dd if=/dev/zero of=/tmp/testfile bs=1M count=100 conv=fsync 2>&1 | tail -1
    rm /tmp/testfile
"

# ネットワークスループットテスト
docker run --rm alpine sh -c "
    apk add --no-cache curl > /dev/null 2>&1
    curl -o /dev/null -s -w '%{speed_download}' https://speed.cloudflare.com/__down?bytes=10000000
    echo ' bytes/sec'
"

# ビルド速度テスト
time docker build --no-cache -t test-build - <<'EOF'
FROM alpine:latest
RUN apk add --no-cache curl wget git
RUN echo "Build test complete"
EOF
docker rmi test-build
```

---

## 7. セキュリティ設定

### 7.1 Docker ソケットの保護

```bash
# Docker ソケットのパーミッション確認
ls -la /var/run/docker.sock
# srw-rw---- 1 root docker 0 ... /var/run/docker.sock

# Docker グループのメンバー確認
getent group docker
# docker:x:999:user1,user2

# 注意: docker グループのメンバーは実質的に root 権限を持つ
# 信頼できるユーザーのみ追加すること

# TCP ソケットでのリモートアクセス（TLS 必須）
# 本番環境では TLS 証明書を使用すること

# CA 証明書の生成
openssl genrsa -aes256 -out ca-key.pem 4096
openssl req -new -x509 -days 365 -key ca-key.pem -sha256 -out ca.pem

# サーバー証明書の生成
openssl genrsa -out server-key.pem 4096
openssl req -subj "/CN=docker-host" -sha256 -new -key server-key.pem -out server.csr
echo subjectAltName = DNS:docker-host,IP:192.168.1.100 > extfile.cnf
openssl x509 -req -days 365 -sha256 -in server.csr -CA ca.pem \
    -CAkey ca-key.pem -CAcreateserial -out server-cert.pem -extfile extfile.cnf

# daemon.json に TLS 設定を追加
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "tls": true,
  "tlscacert": "/etc/docker/certs/ca.pem",
  "tlscert": "/etc/docker/certs/server-cert.pem",
  "tlskey": "/etc/docker/certs/server-key.pem",
  "hosts": ["unix:///var/run/docker.sock", "tcp://0.0.0.0:2376"]
}
EOF
```

### 7.2 Rootless Docker

```bash
# Rootless Docker のインストール（root 権限不要で Docker を実行）
# Ubuntu/Debian の場合
sudo apt-get install -y uidmap dbus-user-session

# Rootless Docker のセットアップ
dockerd-rootless-setuptool.sh install

# 環境変数の設定
export PATH=/usr/bin:$PATH
export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock

# ~/.bashrc に追加
echo 'export DOCKER_HOST=unix:///run/user/$(id -u)/docker.sock' >> ~/.bashrc

# Rootless Docker の起動
systemctl --user start docker
systemctl --user enable docker

# 確認
docker info | grep "rootless"
# rootless: true
```

### 7.3 セキュリティベストプラクティス

```bash
# Docker Bench for Security の実行
docker run --rm --net host --pid host \
    --cap-add audit_control \
    -v /var/lib:/var/lib:ro \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    -v /usr/lib/systemd:/usr/lib/systemd:ro \
    -v /etc:/etc:ro \
    docker/docker-bench-security

# Content Trust の有効化（署名付きイメージのみ許可）
export DOCKER_CONTENT_TRUST=1

# AppArmor プロファイルの確認（Ubuntu）
docker info | grep "Security Options"
# Security Options: apparmor, seccomp, cgroupns

# seccomp プロファイルの確認
docker info --format '{{ .SecurityOptions }}'
```

---

## 8. アンチパターン

### アンチパターン 1: 公式リポジトリを使わずに OS 標準パッケージでインストール

```bash
# NG: OS のデフォルトリポジトリの Docker は古い場合が多い
sudo apt install docker.io
# -> バージョンが古く、BuildKit や Compose V2 が使えない場合がある

# OK: Docker 公式リポジトリからインストール
# (上記セクション4の手順に従う)
sudo apt-get install docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
# -> 最新の機能とセキュリティパッチが適用される
```

### アンチパターン 2: root で Docker を直接実行

```bash
# NG: 常に sudo で Docker を使う
sudo docker run ...
sudo docker build ...
# -> 作成されるファイルが root 所有になり権限問題が発生

# OK: docker グループにユーザーを追加
sudo usermod -aG docker $USER
newgrp docker
docker run ...
# -> ユーザー権限で実行。ただしdockerグループはroot相当の
#    権限を持つことに注意（信頼できるユーザーのみ追加）
```

### アンチパターン 3: Docker Desktop のリソースをデフォルトのまま使う

```bash
# NG: デフォルト設定のまま開発
# -> メモリ不足でビルドが遅い、コンテナが OOM で停止

# OK: プロジェクトに応じてリソースを調整
# Docker Desktop > Settings > Resources
# - 開発用: CPU 4コア / Memory 8GB
# - ビルド重視: CPU 6コア / Memory 12GB
# - 本番テスト: CPU 8コア / Memory 16GB
```

### アンチパターン 4: ログ設定をしない

```bash
# NG: ログ制限なしでコンテナを長期稼働
docker run -d --name app my-app
# -> ログファイルが無限に肥大化し、ディスクを圧迫

# OK: daemon.json でデフォルトのログ制限を設定
# "log-opts": { "max-size": "10m", "max-file": "3" }
# または個別コンテナで指定
docker run -d --name app \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    my-app
```

### アンチパターン 5: Docker ソケットをコンテナにマウントする

```bash
# NG: Docker ソケットを無制限にマウント
docker run -v /var/run/docker.sock:/var/run/docker.sock my-tool
# -> コンテナがホストの全 Docker リソースにアクセス可能
# -> コンテナ脱出の攻撃ベクトルになる

# OK: 必要な場合は読み取り専用 + 最小権限のユーザー
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    --user $(id -u):$(getent group docker | cut -d: -f3) \
    my-tool
# または Docker API プロキシ（Tecnativa/docker-socket-proxy）を使用
```

---

## 8. トラブルシューティング

### 8.1 よくあるエラーと解決策

```bash
# エラー: "Cannot connect to the Docker daemon"
# 原因: Docker デーモンが起動していない
# 解決:
sudo systemctl start docker
# Docker Desktop の場合はアプリケーションを起動

# エラー: "Got permission denied while trying to connect to the Docker daemon socket"
# 原因: ユーザーが docker グループに属していない
# 解決:
sudo usermod -aG docker $USER
# ログアウト → ログイン（または newgrp docker）

# エラー: "no space left on device"
# 原因: Docker のディスク領域が枯渇
# 解決:
docker system prune -a --volumes
docker system df  # 使用量確認

# エラー: "port is already allocated"
# 原因: ポートが他のプロセスに使用されている
# 解決:
# Linux
sudo lsof -i :8080
sudo ss -tlnp | grep 8080
# macOS
lsof -i :8080

# エラー: "image platform does not match"
# 原因: アーキテクチャの不一致（ARM vs x86）
# 解決:
docker run --platform linux/amd64 my-image
# または適切なプラットフォームのイメージを使用

# エラー: "OCI runtime create failed"
# 原因: コンテナランタイムの問題
# 解決:
sudo systemctl restart docker
# または containerd を再起動
sudo systemctl restart containerd
```

### 8.2 ログの確認方法

```bash
# Docker デーモンのログ確認
# systemd の場合
sudo journalctl -u docker.service -f
sudo journalctl -u docker.service --since "1 hour ago"

# Docker Desktop のログ
# macOS: ~/Library/Containers/com.docker.docker/Data/log/
# Windows: %LOCALAPPDATA%\Docker\log\

# コンテナのログ確認
docker logs <container-name>
docker logs -f --tail 100 <container-name>

# Docker events のモニタリング
docker events
docker events --since "30m"
docker events --filter 'type=container' --filter 'event=die'
```

### 8.3 Docker のリセット

```bash
# 全コンテナの停止と削除
docker stop $(docker ps -aq) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

# 全イメージの削除
docker rmi $(docker images -aq) 2>/dev/null

# 全ボリュームの削除
docker volume prune -f

# 全ネットワークの削除
docker network prune -f

# ビルドキャッシュの削除
docker builder prune -af

# 完全リセット（全リソース削除）
docker system prune -af --volumes

# Docker Desktop の完全リセット
# Docker Desktop > Troubleshoot > Reset to factory defaults

# Linux での Docker Engine の完全アンインストールと再インストール
sudo systemctl stop docker
sudo apt-get purge docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd
sudo rm -f /etc/docker/daemon.json
# 再インストール手順を実行
```

---

## 9. FAQ

### Q1: Docker Desktop の有料ライセンスはどの範囲に適用されますか？

**A:** Docker Desktop は、従業員 250 人以上 かつ 年間売上 $10M 以上の企業で商用利用する場合に有料サブスクリプションが必要である（2024 年時点）。個人開発者、オープンソースプロジェクト、小規模企業、教育目的での利用は無料である。Docker Engine（CLI のみ）は完全にオープンソースであり、企業規模に関わらず無料で利用できる。大企業で Docker Desktop を使う場合は、Docker Business プラン（ユーザーあたり月額 $24）を検討する。代替として Colima、Rancher Desktop、OrbStack 等のオープンソースツールを使う方法もある。

### Q2: macOS で Docker が遅いのですが改善方法はありますか？

**A:** macOS では Docker は VM 内で動作するため、ネイティブ Linux に比べて I/O が遅くなる。改善策として以下がある:
- **VirtioFS** を有効化する（Docker Desktop > Settings > General > VirtioFS）
- **不要なバインドマウントを減らす**（node_modules 等は名前付きボリュームに）
- **リソース割り当てを増やす**（CPU / Memory）
- **.dockerignore** で不要ファイルをビルドコンテキストから除外する
- **Rosetta 2 エミュレーション** を有効化する（Apple Silicon で amd64 イメージ使用時）
- **OrbStack** に切り替える（Docker Desktop より高速な場合が多い）
- **開発用の docker-compose.yml** で `cached` / `delegated` マウントオプションを使う

### Q3: WSL2 と Hyper-V バックエンドはどちらがよいですか？

**A:** WSL2 バックエンドが推奨される。WSL2 は Hyper-V に比べてメモリ消費が少なく、起動が高速で、Linux ファイルシステムとの互換性が高い。また、WSL2 ディストリビューション内から直接 Docker CLI を使えるため、Linux ネイティブに近い開発体験が得られる。Hyper-V バックエンドは Windows Home Edition では使用できない点も考慮すべきである。

### Q4: Docker のバージョンアップはどうすればよいですか？

**A:** Docker Desktop の場合は GUI から自動アップデートが可能である。Docker Engine の場合は以下の手順で行う:

```bash
# 現在のバージョン確認
docker version

# パッケージの更新
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# 更新後の確認
docker version
```

アップデート前にコンテナの停止とバックアップを推奨する。`live-restore: true` を設定していれば、デーモンの再起動時にコンテナが維持される。

### Q5: Docker と Podman はどちらを使うべきですか？

**A:** Docker は最も広く使われており、エコシステムが豊富で、ドキュメントも充実している。Podman は RHEL/Fedora 環境でデフォルトツールとして提供されており、デーモンレス・rootless で動作するためセキュリティ面で優位性がある。Docker CLI との互換性も高いため、`alias docker=podman` で移行可能な場合が多い。企業ポリシーや OS 環境に応じて選択する。

### Q6: CI/CD 環境での Docker のインストール方法は？

**A:** CI/CD 環境では Docker-in-Docker (DinD) または Docker ソケットのマウントが一般的である:

```yaml
# GitHub Actions での Docker セットアップ
# docker はプリインストールされているため追加設定不要

# GitLab CI での DinD
services:
  - docker:dind
variables:
  DOCKER_HOST: tcp://docker:2375

# Jenkins での Docker
# Jenkins エージェントに Docker をインストール
# または Docker Pipeline プラグインを使用
```

---

## 10. まとめ

| 項目 | ポイント |
|---|---|
| macOS | Docker Desktop を Homebrew または公式サイトからインストール。代替: Colima, OrbStack |
| Windows | WSL2 を有効化し、Docker Desktop をインストール。.wslconfig でリソース制限 |
| Linux | Docker 公式リポジトリから Docker Engine をインストール。OS 標準パッケージは避ける |
| 初期設定 | daemon.json でログ、ストレージ、DNS を設定。本番では live-restore を有効化 |
| ユーザー権限 | docker グループにユーザーを追加。本番では Rootless Docker を検討 |
| 動作確認 | `docker run --rm hello-world` で検証。ネットワークとボリュームも確認 |
| リソース | プロジェクトに応じて CPU/Memory を調整。VirtioFS で I/O パフォーマンス改善 |
| セキュリティ | Docker Bench で監査。TLS 設定、Content Trust、seccomp を活用 |
| プロキシ | 企業環境ではデーモンとクライアント両方にプロキシ設定が必要 |

---

## 次に読むべきガイド

- [02-docker-basics.md](./02-docker-basics.md) -- Docker の基本操作（run, stop, rm, logs, exec）
- [03-image-management.md](./03-image-management.md) -- イメージの管理とレジストリ
- [../01-dockerfile/00-dockerfile-basics.md](../01-dockerfile/00-dockerfile-basics.md) -- Dockerfile の基礎

---

## 参考文献

1. **Docker Documentation - Install Docker Engine** https://docs.docker.com/engine/install/ -- 各 OS 向けの公式インストール手順。最新の手順は常にここを参照。
2. **Docker Desktop Release Notes** https://docs.docker.com/desktop/release-notes/ -- Docker Desktop の変更履歴。新機能やバグ修正の確認に利用。
3. **Microsoft - WSL2 Documentation** https://learn.microsoft.com/en-us/windows/wsl/ -- WSL2 の公式ドキュメント。Windows での Docker 利用に不可欠。
4. **Docker Documentation - Post-installation steps for Linux** https://docs.docker.com/engine/install/linux-postinstall/ -- Linux インストール後の推奨設定。
5. **Docker Documentation - Docker daemon configuration** https://docs.docker.com/reference/cli/dockerd/#daemon-configuration-file -- daemon.json の全設定項目リファレンス。
6. **Docker Security Best Practices** https://docs.docker.com/develop/security-best-practices/ -- Docker のセキュリティベストプラクティスガイド。
7. **Colima - Container runtimes on macOS** https://github.com/abiosoft/colima -- macOS 向けの Docker Desktop 代替ツール。
8. **OrbStack - Fast, light, simple Docker** https://orbstack.dev/ -- macOS 専用の高速 Docker 環境。
