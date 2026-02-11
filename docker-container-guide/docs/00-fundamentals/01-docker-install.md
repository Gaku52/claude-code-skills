# Docker インストールガイド

> Docker Desktop と Docker Engine のインストール方法、初期設定、動作確認までを網羅する実践的セットアップガイド。

---

## この章で学ぶこと

1. **各 OS に最適な Docker のインストール方法**を選択し、確実にセットアップできる
2. **Docker Desktop と Docker Engine の違い**を理解し、用途に応じて使い分けられる
3. **インストール後の初期設定と動作確認**を完了し、開発を開始できる状態にする

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

---

## 2. macOS へのインストール

### 2.1 Docker Desktop (推奨)

```bash
# 方法1: 公式サイトからダウンロード
# https://www.docker.com/products/docker-desktop/
# Apple Silicon (M1/M2/M3) と Intel 版を選択

# 方法2: Homebrew でインストール
brew install --cask docker

# インストール後、アプリケーションから Docker.app を起動
open /Applications/Docker.app
```

### 2.2 動作確認

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
```

### 2.3 Apple Silicon (ARM64) の注意点

```bash
# ARM64 イメージが存在するか確認
docker manifest inspect --verbose nginx:alpine | grep architecture
# "architecture": "arm64"

# AMD64 イメージを強制的に使う場合（互換性問題時）
docker run --platform linux/amd64 --rm nginx:alpine nginx -v

# マルチプラットフォームビルドの準備
docker buildx create --name mybuilder --use
docker buildx inspect --bootstrap
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

# WSL2 がデフォルトか確認
wsl --set-default-version 2

# WSL のバージョン確認
wsl --list --verbose
# NAME                   STATE           VERSION
# * Ubuntu               Running         2
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

### 3.3 動作確認

```powershell
# PowerShell で確認
docker version
docker run --rm hello-world

# WSL2 ディストリビューション内からも使えることを確認
wsl -d Ubuntu -e docker version
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

### 4.3 インストール後の設定

```bash
# docker グループにユーザーを追加（sudo なしで実行可能にする）
sudo usermod -aG docker $USER

# グループ変更を反映（再ログインまたは以下を実行）
newgrp docker

# 確認
docker run --rm hello-world
# sudo なしで実行できれば成功
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

### 5.2 Docker Desktop の設定 (GUI)

```
+------------------------------------------------------------+
|  Docker Desktop Settings                                   |
|                                                            |
|  General                                                   |
|  [x] Start Docker Desktop when you sign in                |
|  [x] Use the WSL 2 based engine (Windows)                 |
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
+------------------------------------------------------------+
```

### 比較表 2: Linux ディストリビューション別インストール方法

| ディストリビューション | パッケージマネージャ | リポジトリ設定 | 備考 |
|---|---|---|---|
| Ubuntu 22.04/24.04 | apt | docker.list | 最も安定 |
| Debian 12 (Bookworm) | apt | docker.list | Ubuntu と同様 |
| Fedora 38/39 | dnf | docker-ce.repo | SELinux 注意 |
| RHEL 9 / Rocky 9 | dnf | docker-ce.repo | Podman がデフォルト |
| Arch Linux | pacman | 公式リポジトリ | `pacman -S docker` |
| Alpine | apk | community リポジトリ | `apk add docker` |

---

## 6. 動作確認チェックリスト

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
DOCKER_BUILDKIT=1 docker build --help | head -5
```

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: Docker Desktop の有料ライセンスはどの範囲に適用されますか？

**A:** Docker Desktop は、従業員 250 人以上 かつ 年間売上 $10M 以上の企業で商用利用する場合に有料サブスクリプションが必要である（2024 年時点）。個人開発者、オープンソースプロジェクト、小規模企業、教育目的での利用は無料である。Docker Engine（CLI のみ）は完全にオープンソースであり、企業規模に関わらず無料で利用できる。

### Q2: macOS で Docker が遅いのですが改善方法はありますか？

**A:** macOS では Docker は VM 内で動作するため、ネイティブ Linux に比べて I/O が遅くなる。改善策として以下がある:
- **VirtioFS** を有効化する（Docker Desktop > Settings > General > VirtioFS）
- **不要なバインドマウントを減らす**（node_modules 等は名前付きボリュームに）
- **リソース割り当てを増やす**（CPU / Memory）
- **.dockerignore** で不要ファイルをビルドコンテキストから除外する

### Q3: WSL2 と Hyper-V バックエンドはどちらがよいですか？

**A:** WSL2 バックエンドが推奨される。WSL2 は Hyper-V に比べてメモリ消費が少なく、起動が高速で、Linux ファイルシステムとの互換性が高い。また、WSL2 ディストリビューション内から直接 Docker CLI を使えるため、Linux ネイティブに近い開発体験が得られる。Hyper-V バックエンドは Windows Home Edition では使用できない点も考慮すべきである。

---

## 9. まとめ

| 項目 | ポイント |
|---|---|
| macOS | Docker Desktop を Homebrew または公式サイトからインストール |
| Windows | WSL2 を有効化し、Docker Desktop をインストール |
| Linux | Docker 公式リポジトリから Docker Engine をインストール |
| 初期設定 | daemon.json でログ、ストレージ、DNS を設定 |
| ユーザー権限 | docker グループにユーザーを追加 |
| 動作確認 | `docker run --rm hello-world` で検証 |
| リソース | プロジェクトに応じて CPU/Memory を調整 |

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
