# コンテナ技術

> コンテナは「アプリケーションとその依存関係をパッケージ化し、どこでも同じように動作させる」技術である。
> OS カーネルの隔離機構を活用し、仮想マシンより遥かに軽量かつ高速に起動できる。
> 本章ではコンテナの原理から実践的な運用、セキュリティ、オーケストレーションまでを体系的に解説する。

## この章で学ぶこと

- [ ] コンテナの技術的仕組み（Namespace, cgroups, Union FS）を理解する
- [ ] OCI 標準仕様とコンテナランタイムの階層構造を把握する
- [ ] Docker / Podman によるイメージビルドと運用の基本を習得する
- [ ] マルチステージビルド、セキュリティ強化の手法を身につける
- [ ] コンテナネットワーキングとストレージの設計原則を理解する
- [ ] Kubernetes を中心としたオーケストレーションの概念を学ぶ
- [ ] コンテナ運用におけるアンチパターンと対策を認識する
- [ ] CI/CD パイプラインにおけるコンテナ活用を理解する

---

## 1. コンテナ技術の歴史と背景

### 1.1 仮想化からコンテナへの進化

コンテナ技術は突然生まれたものではなく、数十年に及ぶ OS レベル仮想化の歴史の上に成り立っている。

```
コンテナ技術の年表:

1979  chroot          Unix V7 で登場。ルートディレクトリの変更
2000  FreeBSD Jails   chroot を拡張した本格的な隔離環境
2001  Linux VServer   Linux 上のサーバー仮想化パッチ
2004  Solaris Zones   Solaris のコンテナ技術
2006  Process Containers → cgroups として Linux カーネルにマージ
2008  LXC (Linux Containers)  Namespace + cgroups を統合
2013  Docker 0.1      LXC ベースで登場。ユーザーフレンドリーな CLI
2014  Kubernetes      Google が社内 Borg の知見を元にOSS化
2015  OCI 設立        Open Container Initiative。標準仕様の策定
2015  runc 1.0        OCI 準拠の低レベルランタイム
2017  containerd 1.0  CNCF プロジェクトとして独立
2018  Podman 1.0      デーモンレス・ルートレスコンテナ
2020  K8s が Docker shim 非推奨化（containerd/CRI-O 推奨）
2022  WebAssembly コンテナ（Spin, wasmCloud）の台頭
2024  Kata Containers 3.0  マイクロ VM による強固な隔離
```

### 1.2 仮想マシンとコンテナの比較

```
┌─────────────────────────────────────────────────────────┐
│              仮想マシン (VM)                              │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │  App A   │ │  App B   │ │  App C   │               │
│  ├──────────┤ ├──────────┤ ├──────────┤               │
│  │Guest OS  │ │Guest OS  │ │Guest OS  │  ← 各VMにOS   │
│  │(Ubuntu)  │ │(CentOS)  │ │(Alpine)  │    数GB単位   │
│  └──────────┘ └──────────┘ └──────────┘               │
│  ┌─────────────────────────────────────┐               │
│  │       Hypervisor (KVM / Xen)        │  ← HW仮想化  │
│  ├─────────────────────────────────────┤               │
│  │       Host OS (Linux)               │               │
│  ├─────────────────────────────────────┤               │
│  │       Hardware                      │               │
│  └─────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              コンテナ                                    │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │  App A   │ │  App B   │ │  App C   │               │
│  ├──────────┤ ├──────────┤ ├──────────┤               │
│  │  Bins/   │ │  Bins/   │ │  Bins/   │  ← 必要な     │
│  │  Libs    │ │  Libs    │ │  Libs    │    ライブラリ  │
│  └──────────┘ └──────────┘ └──────────┘    のみ (MB)   │
│  ┌─────────────────────────────────────┐               │
│  │   Container Runtime (containerd)    │  ← カーネル   │
│  ├─────────────────────────────────────┤    共有       │
│  │       Host OS (Linux Kernel)        │               │
│  ├─────────────────────────────────────┤               │
│  │       Hardware                      │               │
│  └─────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

**比較表: 仮想マシン vs コンテナ**

| 特性 | 仮想マシン (VM) | コンテナ |
|------|----------------|---------|
| 隔離レベル | ハードウェアレベル（強固） | プロセスレベル（軽量） |
| 起動時間 | 数十秒〜数分 | ミリ秒〜数秒 |
| イメージサイズ | 数 GB〜数十 GB | 数 MB〜数百 MB |
| リソース効率 | 低（各VMにゲストOS） | 高（カーネル共有） |
| 密度 | 1ホストに数十VM | 1ホストに数百〜数千コンテナ |
| カーネル | 独自カーネル | ホストカーネル共有 |
| セキュリティ | 強固な隔離 | カーネル共有のリスク |
| ポータビリティ | 限定的 | 高い（OCI 標準） |
| ライブマイグレーション | 成熟した技術 | 発展途上（CRIU） |
| ユースケース | 異種OS混在、強隔離が必要 | マイクロサービス、CI/CD |

---

## 2. コンテナの仕組み — Linux カーネル機能

コンテナは本質的に「Linux カーネルが提供する隔離機構の組み合わせ」である。魔法のような新技術ではなく、既存のカーネル機能を巧みに組み合わせたものだ。

### 2.1 Namespace（名前空間）

Namespace はカーネルリソースを隔離する仕組みである。各 Namespace は特定のリソースの「見え方」をプロセスごとに分離する。

```
Linux Namespace の種類と役割:

┌───────────────┬──────────────────────────────────────────────┐
│ Namespace     │ 隔離対象                                      │
├───────────────┼──────────────────────────────────────────────┤
│ PID           │ プロセス ID 空間                               │
│               │ コンテナ内の PID 1 = コンテナの init プロセス    │
│               │ ホストからは別の PID で見える                    │
├───────────────┼──────────────────────────────────────────────┤
│ NET (Network) │ ネットワークスタック（インターフェース,           │
│               │ ルーティングテーブル, iptables, ソケット）       │
│               │ 各コンテナが独自の eth0 を持つ                   │
├───────────────┼──────────────────────────────────────────────┤
│ MNT (Mount)   │ ファイルシステムのマウントポイント               │
│               │ コンテナ固有のファイルシステムツリー              │
├───────────────┼──────────────────────────────────────────────┤
│ UTS           │ ホスト名とドメイン名                            │
│               │ 各コンテナが独自のホスト名を持つ                 │
├───────────────┼──────────────────────────────────────────────┤
│ IPC           │ プロセス間通信（セマフォ、メッセージキュー、      │
│               │ 共有メモリ）                                    │
├───────────────┼──────────────────────────────────────────────┤
│ User          │ UID/GID のマッピング                            │
│               │ コンテナ内 root = ホストの非特権ユーザー          │
├───────────────┼──────────────────────────────────────────────┤
│ Cgroup        │ cgroup のルートディレクトリビュー                │
│               │ コンテナが自身の cgroup ツリーのみ参照            │
├───────────────┼──────────────────────────────────────────────┤
│ Time          │ システムクロック（Linux 5.6 以降）               │
│               │ コンテナ固有の時刻設定                           │
└───────────────┴──────────────────────────────────────────────┘
```

**コード例 1: Namespace の確認と手動コンテナ作成**

```bash
#!/bin/bash
# === Namespace を使った手動コンテナ作成 ===

# 現在のプロセスの Namespace を確認
ls -la /proc/$$/ns/
# lrwxrwxrwx 1 root root 0 ... cgroup -> cgroup:[4026531835]
# lrwxrwxrwx 1 root root 0 ... ipc -> ipc:[4026531839]
# lrwxrwxrwx 1 root root 0 ... mnt -> mnt:[4026531840]
# lrwxrwxrwx 1 root root 0 ... net -> net:[4026531992]
# lrwxrwxrwx 1 root root 0 ... pid -> pid:[4026531836]
# lrwxrwxrwx 1 root root 0 ... user -> user:[4026531837]
# lrwxrwxrwx 1 root root 0 ... uts -> uts:[4026531838]

# unshare で新しい Namespace を作成して bash を起動
# PID, UTS, Mount Namespace を分離
sudo unshare --pid --uts --mount --fork /bin/bash

# 新しい Namespace 内での操作
hostname container-demo        # UTS Namespace: 独自のホスト名
mount -t proc proc /proc       # MNT Namespace: proc を再マウント
ps aux                          # PID Namespace: PID 1 から始まる

# 別ターミナルからホスト側で確認
# ホストからは通常の PID で見える
ps aux | grep "unshare"

# nsenter で既存の Namespace に入る
# (PID はコンテナプロセスのホスト側 PID)
sudo nsenter --target <PID> --pid --uts --mount
```

### 2.2 cgroups（Control Groups）

cgroups はプロセスグループに対するリソース制限・優先度制御・監視を行う。

```
cgroups v2 の階層構造:

/sys/fs/cgroup/
├── cgroup.controllers        # 利用可能なコントローラ一覧
├── cgroup.subtree_control    # サブツリーで有効化するコントローラ
├── system.slice/
│   ├── docker-<container-id>.scope/
│   │   ├── cpu.max           # CPU 制限 (quota period)
│   │   ├── cpu.weight        # CPU 重み (1-10000)
│   │   ├── memory.max        # メモリ上限 (bytes)
│   │   ├── memory.current    # 現在のメモリ使用量
│   │   ├── memory.swap.max   # Swap 上限
│   │   ├── io.max            # ブロック I/O 制限
│   │   ├── pids.max          # 最大プロセス数
│   │   └── cgroup.procs      # 所属プロセスの PID 一覧
│   └── docker-<another-id>.scope/
│       └── ...
└── user.slice/
    └── ...

リソース制御の仕組み:

  CPU 制限:
  cpu.max = "200000 100000"
  → 100ms の期間 (period) 中 200ms 分の CPU 時間
  → 実質 2 コア分の CPU を使用可能

  メモリ制限:
  memory.max = 536870912    (512 MB)
  memory.swap.max = 0       (Swap 無効)
  → 超過時: OOM Killer がコンテナ内プロセスを kill

  PID 制限:
  pids.max = 512
  → fork bomb 対策
```

**コード例 2: cgroups を使ったリソース制限の確認**

```bash
#!/bin/bash
# === cgroups v2 によるリソース制限の観察 ===

# Docker コンテナを制限付きで起動
docker run -d \
  --name cgroup-demo \
  --cpus="1.5" \
  --memory="256m" \
  --memory-swap="256m" \
  --pids-limit=100 \
  nginx:alpine

# コンテナの cgroup パスを確認
CONTAINER_ID=$(docker inspect --format '{{.Id}}' cgroup-demo)
CGROUP_PATH="/sys/fs/cgroup/system.slice/docker-${CONTAINER_ID}.scope"

# CPU 制限の確認
cat ${CGROUP_PATH}/cpu.max
# 出力例: 150000 100000
# → 100ms 期間中 150ms 分 = 1.5 コア

# メモリ制限の確認
cat ${CGROUP_PATH}/memory.max
# 出力例: 268435456 (256 MB)

# 現在のメモリ使用量
cat ${CGROUP_PATH}/memory.current

# PID 制限の確認
cat ${CGROUP_PATH}/pids.max
# 出力例: 100

# リアルタイムでリソース使用量を監視
docker stats cgroup-demo --no-stream
# CONTAINER ID  NAME         CPU %  MEM USAGE / LIMIT  MEM %  NET I/O  ...
# abc123def456  cgroup-demo  0.02%  3.5MiB / 256MiB    1.37%  ...

# ストレステストでメモリ制限を検証
docker run --rm --memory="64m" --memory-swap="64m" \
  alpine:latest sh -c "
    # メモリを大量に確保してみる
    dd if=/dev/zero of=/dev/null bs=1M count=128
  "
# → OOM Killed される

# クリーンアップ
docker rm -f cgroup-demo
```

### 2.3 Union FS（OverlayFS）

Union FS はコンテナイメージの効率的なレイヤー構造を実現するファイルシステムである。

```
OverlayFS の動作原理:

  ファイル読み取り (Read):
  ┌─────────────────────────┐
  │ Upper Layer (Container)  │  1. まず upperdir を確認
  │ (Read-Write)             │     ファイルがあれば返す
  └────────────┬────────────┘
               │ なければ下へ
  ┌────────────▼────────────┐
  │ Lower Layer 3 (App)      │  2. lower の最上位から順に検索
  │ (Read-Only)              │
  └────────────┬────────────┘
               │ なければ下へ
  ┌────────────▼────────────┐
  │ Lower Layer 2 (Runtime)  │  3. 見つかった時点で返す
  │ (Read-Only)              │
  └────────────┬────────────┘
               │ なければ下へ
  ┌────────────▼────────────┐
  │ Lower Layer 1 (Base OS)  │  4. 最下層まで検索
  │ (Read-Only)              │
  └─────────────────────────┘

  ファイル書き込み (Write):
  Copy-on-Write (CoW) 戦略
  ┌─────────────────────────┐
  │ Upper Layer              │  書き込み先は常に upper
  │  /etc/nginx/nginx.conf ←──── 変更時: lower からコピーして
  │  (modified copy)         │         upper で書き換え
  └─────────────────────────┘
  ┌─────────────────────────┐
  │ Lower Layer              │  元のファイルは不変
  │  /etc/nginx/nginx.conf   │  (他のコンテナと共有可能)
  │  (original, untouched)   │
  └─────────────────────────┘

  ファイル削除:
  Whiteout ファイルで「削除済み」を示す
  upper に .wh.<filename> を作成
  → lower のファイルは実際には消えないが見えなくなる
```

### 2.4 seccomp と Capabilities

```
セキュリティ機構の階層:

  ┌─────────────────────────────────────────────────┐
  │           アプリケーション                        │
  ├─────────────────────────────────────────────────┤
  │ AppArmor / SELinux   MAC (強制アクセス制御)       │
  ├─────────────────────────────────────────────────┤
  │ seccomp-bpf          システムコール フィルタ       │
  │                      約300+ syscall から必要な     │
  │                      ものだけを許可                │
  ├─────────────────────────────────────────────────┤
  │ Capabilities         root 権限の細分化             │
  │                      CAP_NET_BIND_SERVICE: 特権    │
  │                      ポートのバインド              │
  │                      CAP_SYS_ADMIN: マウント等     │
  ├─────────────────────────────────────────────────┤
  │ Namespace            リソースの可視性を隔離         │
  ├─────────────────────────────────────────────────┤
  │ cgroups              リソース使用量を制限           │
  ├─────────────────────────────────────────────────┤
  │ Linux Kernel                                     │
  └─────────────────────────────────────────────────┘

Docker デフォルトで許可される Capabilities (一部):
  CAP_CHOWN            ファイルの所有者変更
  CAP_DAC_OVERRIDE     ファイルアクセス権を無視
  CAP_FSETID           set-user-ID ビットの維持
  CAP_FOWNER           ファイル所有者関連の権限
  CAP_NET_RAW          RAW ソケットの使用
  CAP_NET_BIND_SERVICE 特権ポート (< 1024) のバインド
  CAP_SYS_CHROOT       chroot の使用
  CAP_SETUID           プロセスの UID 変更
  CAP_SETGID           プロセスの GID 変更

Docker デフォルトで拒否される Capabilities (一部):
  CAP_SYS_ADMIN        多数の管理操作 (mount 等)
  CAP_SYS_PTRACE       プロセスのトレース
  CAP_SYS_MODULE       カーネルモジュールのロード
  CAP_NET_ADMIN        ネットワーク設定の変更
  CAP_SYS_RAWIO        I/O ポートへの直接アクセス
  CAP_SYS_BOOT         システムの再起動
```

---

## 3. OCI 標準仕様とコンテナランタイム

### 3.1 OCI（Open Container Initiative）

OCI は 2015 年に Docker 社と CoreOS 社を中心に Linux Foundation 傘下で設立された。コンテナの相互運用性を保証する 3 つの標準仕様を定義している。

```
OCI 標準仕様の構成:

1. Runtime Specification (runtime-spec)
   コンテナの実行方法を定義
   ├── config.json    コンテナの設定
   │   ├── ociVersion     OCI バージョン
   │   ├── process         実行するプロセス情報
   │   │   ├── args        コマンドライン引数
   │   │   ├── env         環境変数
   │   │   ├── cwd         作業ディレクトリ
   │   │   └── user        実行ユーザー
   │   ├── root            ルートファイルシステム
   │   ├── mounts          マウントポイント
   │   ├── linux           Linux 固有設定
   │   │   ├── namespaces  使用する Namespace
   │   │   ├── resources   cgroups リソース制限
   │   │   └── seccomp     seccomp プロファイル
   │   └── hooks           ライフサイクルフック
   └── rootfs/        ルートファイルシステム

2. Image Specification (image-spec)
   コンテナイメージのフォーマットを定義
   ├── Image Index      マルチアーキテクチャ対応の一覧
   ├── Image Manifest   レイヤーと設定の参照情報
   ├── Image Config     実行時設定 (CMD, ENV, EXPOSE 等)
   └── Filesystem Layers  tar+gzip 形式のレイヤー群

3. Distribution Specification (distribution-spec)
   コンテナイメージの配布方法を定義
   ├── Push     レジストリへのイメージ送信
   ├── Pull     レジストリからのイメージ取得
   ├── Content Discovery  タグ一覧等のメタデータ取得
   └── Content Management  イメージの削除等
```

### 3.2 コンテナランタイムの階層

```
コンテナランタイムのアーキテクチャ:

  ユーザー操作
      │
      ▼
  ┌──────────────────────────────────┐
  │ CLI / API                         │
  │ docker, nerdctl, podman, crictl   │
  └──────────────┬───────────────────┘
                 │
      ▼          ▼
  ┌──────────────────────────────────┐
  │ 高レベルランタイム (CRI 実装)      │  デーモンプロセス
  │                                   │  イメージ管理
  │  containerd         CRI-O         │  スナップショット
  │  (Docker/K8s両対応) (K8s専用)      │  ネットワーク管理
  └──────────────┬───────────────────┘
                 │ OCI Runtime Spec
                 ▼
  ┌──────────────────────────────────┐
  │ 低レベルランタイム (OCI Runtime)    │  Namespace 作成
  │                                   │  cgroups 設定
  │  runc           crun              │  プロセス起動
  │  (Go, 標準)     (C, 高速)          │
  │                                   │
  │  gVisor (runsc)  Kata (kata-rt)   │  サンドボックス型
  │  (ユーザー空間    (マイクロVM       │  強固な隔離
  │   カーネル)       ベース)           │
  └──────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │ Linux Kernel                      │
  │ Namespaces + cgroups + seccomp    │
  └──────────────────────────────────┘
```

**比較表: 低レベルコンテナランタイムの比較**

| ランタイム | 言語 | 隔離方式 | 起動速度 | セキュリティ | メモリオーバーヘッド | ユースケース |
|-----------|------|---------|---------|-------------|-------------------|------------|
| runc | Go | Namespace+cgroups | 高速 | 標準 | 最小 | 汎用（デフォルト） |
| crun | C | Namespace+cgroups | 最速 | 標準 | 最小 | パフォーマンス重視 |
| gVisor (runsc) | Go | ユーザー空間カーネル | やや遅い | 高い | 中程度 (数十MB) | マルチテナント |
| Kata Containers | Go/Rust | マイクロVM | 遅い | 最も高い | 大きい (数十MB) | 高セキュリティ |
| youki | Rust | Namespace+cgroups | 高速 | 標準 | 最小 | Rust エコシステム |
| WasmEdge | C++ | Wasm サンドボックス | 最速級 | 高い | 最小 | エッジ/サーバーレス |

---

## 4. Docker によるコンテナ実践

### 4.1 Docker アーキテクチャ

```
Docker のアーキテクチャ全体像:

  ┌─────────────────────────────────────────────────────────┐
  │ Client (docker CLI)                                      │
  │                                                         │
  │  docker build    docker run    docker pull    docker ps  │
  └────────────────────────┬────────────────────────────────┘
                           │ REST API (Unix Socket / TCP)
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Docker Daemon (dockerd)                                  │
  │                                                         │
  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐   │
  │  │ Image Mgmt  │  │ Network     │  │ Volume Mgmt   │   │
  │  │ Build/Pull  │  │ bridge/host │  │ Bind/Named    │   │
  │  │ Push/Tag    │  │ overlay/mac │  │ tmpfs         │   │
  │  └─────────────┘  └─────────────┘  └───────────────┘   │
  └────────────────────────┬────────────────────────────────┘
                           │ gRPC
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ containerd                                               │
  │                                                         │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
  │  │Snapshots │ │Content   │ │Tasks     │ │Events    │  │
  │  │(OverlayFS)│ │Store    │ │(Process) │ │Stream    │  │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
  └────────────────────────┬────────────────────────────────┘
                           │ OCI Runtime Spec
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ runc                                                     │
  │ → Namespace 作成 → cgroups 設定 → seccomp 適用           │
  │ → pivot_root → プロセス exec                              │
  └─────────────────────────────────────────────────────────┘
```

### 4.2 Dockerfile の体系的理解

**コード例 3: 本番品質の Dockerfile（マルチステージビルド）**

```dockerfile
# === ステージ 1: 依存関係のインストール ===
FROM node:20-slim AS deps
WORKDIR /app

# package.json と lock ファイルのみ先にコピー
# → 依存関係が変わらなければキャッシュが効く
COPY package.json package-lock.json ./
RUN npm ci --production && npm cache clean --force

# === ステージ 2: ビルド ===
FROM node:20-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
# ビルド成果物: /app/dist/

# === ステージ 3: 本番イメージ ===
FROM gcr.io/distroless/nodejs20-debian12 AS production

# メタデータラベル (OCI Image Spec 準拠)
LABEL org.opencontainers.image.title="my-api-server"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Production API Server"
LABEL org.opencontainers.image.source="https://github.com/example/my-api"

WORKDIR /app

# 必要なファイルのみコピー（ビルドツールは含まない）
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

# 環境変数
ENV NODE_ENV=production
ENV PORT=3000

# ポート宣言（ドキュメント目的）
EXPOSE 3000

# distroless イメージのため USER 設定は不要
# (デフォルトで非 root ユーザーで実行)

# ヘルスチェック
# distroless ではシェルがないため、
# K8s の livenessProbe/readinessProbe を使用推奨

# 起動コマンド
CMD ["dist/server.js"]
```

```
マルチステージビルドのイメージサイズ比較:

  ┌────────────────────────────────────────────────────┐
  │ シングルステージ (node:20)                          │
  │ ████████████████████████████████████  1.1 GB       │
  │ [Node.js + npm + build tools + src + node_modules] │
  ├────────────────────────────────────────────────────┤
  │ シングルステージ (node:20-slim)                     │
  │ ██████████████████████  650 MB                     │
  │ [Node.js + src + node_modules]                     │
  ├────────────────────────────────────────────────────┤
  │ マルチステージ (node:20-slim → distroless)          │
  │ ████████  180 MB                                   │
  │ [Node.js runtime + dist + prod node_modules]       │
  ├────────────────────────────────────────────────────┤
  │ マルチステージ (node:20-slim → alpine)              │
  │ ███████  150 MB                                    │
  │ [Node.js (musl) + dist + prod node_modules]        │
  └────────────────────────────────────────────────────┘

  削減率: 最大 85% のサイズ削減
  セキュリティ: 攻撃対象面 (Attack Surface) も大幅に縮小
```

### 4.3 Docker Compose による複数コンテナ管理

**コード例 4: 本番レベルの docker-compose.yml**

```yaml
# docker-compose.yml
# API + DB + Cache + Reverse Proxy の構成例

version: "3.9"

services:
  # --- Reverse Proxy ---
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      api:
        condition: service_healthy
    networks:
      - frontend
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 128M

  # --- API Server ---
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://app:${DB_PASSWORD}@db:5432/myapp
      - REDIS_URL=redis://cache:6379
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    networks:
      - frontend
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 128M

  # --- Database ---
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - db-data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1G

  # --- Cache ---
  cache:
    image: redis:7-alpine
    command: redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru
    volumes:
      - cache-data:/data
    networks:
      - backend
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256M

volumes:
  db-data:
    driver: local
  cache-data:
    driver: local

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true   # 外部アクセス不可（DB/Cache を保護）
```

### 4.4 .dockerignore のベストプラクティス

```
# .dockerignore
# ビルドコンテキストから除外するファイル

# バージョン管理
.git
.gitignore

# 依存関係（コンテナ内で再インストール）
node_modules
vendor/
__pycache__
*.pyc

# ビルド成果物
dist
build
*.o
*.a

# 環境設定・機密情報
.env
.env.*
*.pem
*.key
credentials.json

# IDE / エディタ
.vscode
.idea
*.swp
*.swo
*~

# テスト・ドキュメント
tests/
test/
docs/
*.md
LICENSE

# Docker 関連
Dockerfile*
docker-compose*
.dockerignore

# OS ファイル
.DS_Store
Thumbs.db
```

---

## 5. コンテナネットワーキング

### 5.1 Docker ネットワークドライバ

コンテナのネットワーキングは Linux の仮想ネットワーク機能（veth ペア、ブリッジ、iptables、VXLAN 等）を基盤とする。

```
Docker ネットワークの種類と通信経路:

  1. bridge (デフォルト)
  ┌──────────────────────────────────────────────────┐
  │ Host                                              │
  │                                                   │
  │  ┌──────────┐    ┌──────────┐                    │
  │  │Container │    │Container │                    │
  │  │  A       │    │  B       │                    │
  │  │ eth0     │    │ eth0     │                    │
  │  └──┬───────┘    └──┬───────┘                    │
  │     │ veth pair     │ veth pair                   │
  │  ┌──▼───────────────▼───────┐                    │
  │  │   docker0 (bridge)       │ 172.17.0.1         │
  │  │   172.17.0.0/16          │                    │
  │  └──────────┬───────────────┘                    │
  │             │ NAT (iptables)                      │
  │  ┌──────────▼───────────────┐                    │
  │  │   eth0 (host NIC)        │ 192.168.1.100      │
  │  └──────────────────────────┘                    │
  └──────────────────────────────────────────────────┘

  2. host
  コンテナがホストのネットワークスタックを直接使用
  → ポートマッピング不要、最高のネットワーク性能
  → ポート競合のリスク、隔離なし

  3. overlay (Swarm / K8s)
  複数ホスト間でコンテナネットワークを構築
  → VXLAN トンネリングでL2接続を実現
  → クラスタ環境でのサービス間通信

  4. macvlan
  コンテナに固有の MAC アドレスを割り当て
  → 物理ネットワークに直接接続
  → レガシーシステムとの統合に有用

  5. none
  ネットワークなし（完全隔離）
  → バッチ処理やセキュリティ目的
```

**Docker ネットワークドライバの比較表**

| ドライバ | 性能 | 隔離 | マルチホスト | 主な用途 |
|---------|------|------|------------|---------|
| bridge | 中 | あり | 不可 | 開発環境、単一ホストの本番 |
| host | 高 | なし | 不可 | 性能重視のアプリケーション |
| overlay | 中〜低 | あり | 可能 | Swarm/K8s クラスタ |
| macvlan | 高 | あり | 不可 | レガシー統合、直接L2接続 |
| ipvlan | 高 | あり | 不可 | MAC アドレス制限のある環境 |
| none | - | 完全 | - | セキュリティ隔離、バッチ処理 |

### 5.2 コンテナ間通信のパターン

```
サービスディスカバリと通信パターン:

  パターン1: Docker Compose の DNS ベース
  ┌─────────────────────────────────────────┐
  │ User-Defined Bridge Network              │
  │                                          │
  │  api ──── "redis://cache:6379" ───► cache│
  │   │                                      │
  │   └───── "postgresql://db:5432" ──► db   │
  │                                          │
  │  Docker の組み込み DNS (127.0.0.11)       │
  │  サービス名 → コンテナ IP を自動解決       │
  └─────────────────────────────────────────┘

  パターン2: K8s の Service ベース
  ┌─────────────────────────────────────────┐
  │ Kubernetes Cluster                       │
  │                                          │
  │  Pod A ─── "http://api-svc:3000" ──►    │
  │            ClusterIP Service             │
  │                │ kube-proxy (iptables)   │
  │            ┌───▼────┐  ┌────────┐       │
  │            │ Pod B-1│  │ Pod B-2│       │
  │            │ (api)  │  │ (api)  │       │
  │            └────────┘  └────────┘       │
  │                                          │
  │  CoreDNS: <svc>.<ns>.svc.cluster.local   │
  └─────────────────────────────────────────┘
```

---

## 6. コンテナストレージとデータ管理

### 6.1 ストレージの種類

```
Docker のストレージオプション:

  1. Volumes (推奨)
  ┌──────────────────────────────────────┐
  │ Container                            │
  │ /app/data ──────┐                    │
  └─────────────────┼────────────────────┘
                    │ mount
  ┌─────────────────▼────────────────────┐
  │ /var/lib/docker/volumes/mydata/_data  │
  │ Docker が管理するストレージ            │
  │ ├── バックアップが容易                 │
  │ ├── Linux / Mac / Windows 対応        │
  │ └── Volume ドライバで拡張可能          │
  └──────────────────────────────────────┘

  2. Bind Mounts
  ┌──────────────────────────────────────┐
  │ Container                            │
  │ /app/src ───────┐                    │
  └─────────────────┼────────────────────┘
                    │ mount
  ┌─────────────────▼────────────────────┐
  │ /home/user/project/src               │
  │ ホストの任意のパスをマウント           │
  │ ├── 開発時のライブリロードに便利       │
  │ ├── ホストのディレクトリ構造に依存     │
  │ └── セキュリティリスク（ホスト露出）    │
  └──────────────────────────────────────┘

  3. tmpfs Mounts
  ┌──────────────────────────────────────┐
  │ Container                            │
  │ /app/tmp ───────┐                    │
  └─────────────────┼────────────────────┘
                    │
  ┌─────────────────▼────────────────────┐
  │ メモリ上のファイルシステム             │
  │ ├── ディスクに書き込まない            │
  │ ├── 高速だがコンテナ停止で消失         │
  │ └── 一時ファイルやシークレットに適切    │
  └──────────────────────────────────────┘
```

### 6.2 データ永続化のベストプラクティス

```bash
#!/bin/bash
# === ストレージ管理のコマンド例 ===

# Named Volume の作成と使用
docker volume create app-data
docker run -d \
  --name db \
  -v app-data:/var/lib/postgresql/data \
  postgres:16

# Volume の詳細情報
docker volume inspect app-data
# 出力例:
# [{
#   "CreatedAt": "2024-01-15T10:30:00Z",
#   "Driver": "local",
#   "Mountpoint": "/var/lib/docker/volumes/app-data/_data",
#   "Name": "app-data",
#   "Scope": "local"
# }]

# Volume のバックアップ
docker run --rm \
  -v app-data:/source:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/app-data-backup.tar.gz -C /source .

# Volume のリストア
docker run --rm \
  -v app-data:/target \
  -v $(pwd):/backup:ro \
  alpine tar xzf /backup/app-data-backup.tar.gz -C /target

# 未使用 Volume の一括削除
docker volume prune -f

# Bind Mount (開発環境向け)
docker run -d \
  --name dev-server \
  -v $(pwd)/src:/app/src:cached \
  -v /app/node_modules \
  node:20-slim npm run dev
# :cached → macOS で書き込みパフォーマンス改善
# /app/node_modules → 匿名 Volume でホストのを上書きしない

# tmpfs (機密データ向け)
docker run -d \
  --name secure-app \
  --tmpfs /app/secrets:rw,noexec,nosuid,size=64m \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  my-app:latest

# Read-Only ルートファイルシステム + tmpfs
docker run -d \
  --name readonly-app \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid \
  --tmpfs /var/run:rw,noexec,nosuid \
  nginx:alpine
```

---

## 7. コンテナセキュリティ

### 7.1 脅威モデルとセキュリティレイヤー

```
コンテナセキュリティの多層防御:

  攻撃面 (Attack Surface)
  ┌─────────────────────────────────────────────────────┐
  │ Layer 7: アプリケーション脆弱性                       │
  │   SQLi, XSS, RCE → WAF, 入力検証, 脆弱性スキャン     │
  ├─────────────────────────────────────────────────────┤
  │ Layer 6: 依存関係の脆弱性                             │
  │   CVE 付きライブラリ → Trivy/Snyk スキャン, SCA       │
  ├─────────────────────────────────────────────────────┤
  │ Layer 5: コンテナイメージ                              │
  │   不要なツール/シェル → distroless, minimal base       │
  │   root 実行 → USER 命令で非 root 化                   │
  ├─────────────────────────────────────────────────────┤
  │ Layer 4: コンテナランタイム                            │
  │   特権コンテナ → --privileged 禁止                    │
  │   過剰な Capabilities → drop ALL + 必要分のみ add     │
  │   syscall 悪用 → seccomp プロファイル                  │
  ├─────────────────────────────────────────────────────┤
  │ Layer 3: ホスト OS                                    │
  │   カーネル脆弱性 → パッチ適用, gVisor/Kata で隔離      │
  │   Docker ソケット露出 → ソケットのマウント禁止          │
  ├─────────────────────────────────────────────────────┤
  │ Layer 2: ネットワーク                                  │
  │   横方向移動 → NetworkPolicy, internal ネットワーク    │
  │   平文通信 → mTLS (サービスメッシュ)                   │
  ├─────────────────────────────────────────────────────┤
  │ Layer 1: オーケストレーション                          │
  │   RBAC 設定ミス → 最小権限の原則                      │
  │   Secret 平文 → Vault, Sealed Secrets                │
  └─────────────────────────────────────────────────────┘
```

### 7.2 セキュリティ強化の実践

**コード例 5: セキュリティ強化された Docker 実行**

```bash
#!/bin/bash
# === セキュリティ強化コンテナの実行例 ===

# ---- 基本的なセキュリティ強化 ----

# 1. 非 root ユーザーで実行
docker run -d \
  --name secure-nginx \
  --user 1000:1000 \
  nginx:alpine

# 2. Capabilities を最小化
docker run -d \
  --name minimal-caps \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --cap-add=CHOWN \
  --cap-add=SETUID \
  --cap-add=SETGID \
  nginx:alpine

# 3. Read-Only ファイルシステム
docker run -d \
  --name readonly-web \
  --read-only \
  --tmpfs /var/cache/nginx:rw,noexec,nosuid \
  --tmpfs /var/run:rw,noexec,nosuid \
  --tmpfs /tmp:rw,noexec,nosuid \
  nginx:alpine

# 4. seccomp プロファイルの適用
docker run -d \
  --name seccomp-app \
  --security-opt seccomp=./custom-seccomp.json \
  my-app:latest

# 5. AppArmor プロファイルの適用
docker run -d \
  --name apparmor-app \
  --security-opt apparmor=docker-custom \
  my-app:latest

# ---- 総合的なセキュリティ強化 ----

docker run -d \
  --name hardened-app \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges:true \
  --security-opt seccomp=./seccomp-profile.json \
  --user 1000:1000 \
  --memory=256m \
  --memory-swap=256m \
  --cpus="0.5" \
  --pids-limit=64 \
  --network=app-net \
  --restart=unless-stopped \
  --health-cmd="wget --spider -q http://localhost:8080/health" \
  --health-interval=30s \
  --health-timeout=5s \
  --health-retries=3 \
  my-app:latest

# ---- イメージスキャン ----

# Trivy によるイメージスキャン
trivy image --severity HIGH,CRITICAL my-app:latest

# 出力例:
# my-app:latest (alpine 3.19.0)
# ================================
# Total: 2 (HIGH: 1, CRITICAL: 1)
#
# ┌───────────────┬──────────────┬──────────┬────────┐
# │   Library     │ Vulnerability│ Severity │ Status │
# ├───────────────┼──────────────┼──────────┼────────┤
# │ openssl       │ CVE-2024-XXX │ CRITICAL │ fixed  │
# │ curl          │ CVE-2024-YYY │ HIGH     │ fixed  │
# └───────────────┴──────────────┴──────────┴────────┘

# Trivy で Dockerfile 自体をスキャン（設定ミス検出）
trivy config ./Dockerfile

# Docker Scout による脆弱性分析
docker scout cves my-app:latest
docker scout recommendations my-app:latest
```

### 7.3 Podman: デーモンレス・ルートレスコンテナ

```
Docker vs Podman のアーキテクチャ比較:

  Docker:
  ┌─────────┐     ┌──────────────────────┐
  │ docker   │────►│ dockerd (root デーモン)│
  │ CLI      │     │     │                │
  └─────────┘     │  containerd          │
                  │     │                │
                  │   runc               │
                  └──────────────────────┘
  → デーモンが SPOF (Single Point of Failure)
  → root 権限で動作するリスク
  → デーモン再起動で全コンテナに影響

  Podman:
  ┌─────────┐
  │ podman   │──── fork/exec ────► runc
  │ CLI      │     (デーモンなし)
  └─────────┘
  → デーモンレス: 各コマンドが直接 runc を呼び出し
  → ルートレス: 一般ユーザーで実行可能
  → systemd 統合: コンテナを systemd ユニットとして管理
  → 互換性: docker CLI とほぼ同じ使い方

Podman の特徴的な機能:
  # Pod (K8s の Pod と同概念)
  podman pod create --name my-pod -p 8080:80
  podman run --pod my-pod nginx:alpine
  podman run --pod my-pod php:fpm

  # systemd ユニット生成
  podman generate systemd --new --name my-container \
    > ~/.config/systemd/user/my-container.service
  systemctl --user enable --now my-container

  # K8s YAML の生成/適用
  podman generate kube my-pod > pod.yaml
  podman play kube pod.yaml
```

---

## 8. Kubernetes によるコンテナオーケストレーション

### 8.1 Kubernetes アーキテクチャ

```
Kubernetes クラスタの詳細アーキテクチャ:

  ┌─────────────────────────────────────────────────────────────┐
  │                    Control Plane                             │
  │                                                             │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
  │  │ kube-apiserver│  │kube-scheduler│  │  kube-controller │  │
  │  │              │  │              │  │  -manager        │  │
  │  │ REST API     │  │ Pod の配置    │  │ ReplicaSet       │  │
  │  │ 認証/認可    │  │ ノード選択    │  │ Deployment       │  │
  │  │ Admission    │  │ リソース考慮  │  │ Node/Job/...     │  │
  │  └──────┬───────┘  └──────────────┘  └──────────────────┘  │
  │         │                                                   │
  │  ┌──────▼───────┐  ┌──────────────┐                        │
  │  │   etcd       │  │ cloud-ctrl-  │                        │
  │  │ 分散 KV Store│  │ manager      │                        │
  │  │ クラスタ状態 │  │ (CSP連携)    │                        │
  │  │ の唯一の     │  └──────────────┘                        │
  │  │ 情報源       │                                          │
  │  └──────────────┘                                          │
  └─────────────────────────────────────────────────────────────┘
                           │
                  ─────────┼──────────
                           │
  ┌─────────────────────────────────────────────────────────────┐
  │                    Worker Node                              │
  │                                                             │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
  │  │  kubelet     │  │ kube-proxy   │  │ Container        │  │
  │  │              │  │              │  │ Runtime          │  │
  │  │ Pod の管理    │  │ Service の    │  │ (containerd/     │  │
  │  │ ヘルスチェック│  │ ネットワーク  │  │  CRI-O)          │  │
  │  │ CRI 経由で   │  │ ルール管理    │  │                  │  │
  │  │ ランタイム制御│  │ (iptables/   │  │ OCI Runtime      │  │
  │  │              │  │  IPVS)       │  │ (runc)           │  │
  │  └──────────────┘  └──────────────┘  └──────────────────┘  │
  │                                                             │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │ Pod                                                  │   │
  │  │ ┌────────────┐ ┌────────────┐ ┌──────────────────┐  │   │
  │  │ │ Container  │ │ Container  │ │ Pause Container  │  │   │
  │  │ │ (app)      │ │ (sidecar)  │ │ (network ns 保持)│  │   │
  │  │ └────────────┘ └────────────┘ └──────────────────┘  │   │
  │  │          共有: Network NS, IPC NS, Volume            │   │
  │  └──────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────┘
```

### 8.2 Kubernetes の主要リソース

```
K8s リソースの階層と関係:

  Deployment
  ├── strategy: RollingUpdate / Recreate
  ├── replicas: 3
  └── ReplicaSet (自動管理)
      ├── Pod-1
      │   ├── Container (app)
      │   ├── Container (sidecar)
      │   └── Volume
      ├── Pod-2
      │   └── ...
      └── Pod-3
          └── ...

  Service (Pod への安定したアクセス)
  ├── ClusterIP    クラスタ内部のみ (デフォルト)
  ├── NodePort     各ノードのポートで公開 (30000-32767)
  ├── LoadBalancer  外部 LB を自動プロビジョニング
  └── ExternalName  外部 DNS 名のエイリアス

  Ingress (HTTP/HTTPS ルーティング)
  ├── host: api.example.com → api-service:3000
  ├── host: web.example.com → web-service:80
  └── TLS 終端

  ConfigMap / Secret (設定とシークレットの外部化)
  ├── 環境変数として注入
  ├── ファイルとしてマウント
  └── コマンドライン引数として使用

  PersistentVolume (PV) / PersistentVolumeClaim (PVC)
  ├── StorageClass で動的プロビジョニング
  ├── AccessMode: ReadWriteOnce / ReadOnlyMany / ReadWriteMany
  └── Reclaim Policy: Retain / Delete

  HorizontalPodAutoscaler (HPA)
  ├── CPU/メモリ使用率に基づくスケール
  ├── カスタムメトリクス (Prometheus 連携)
  └── min/max レプリカ数の制約
```

### 8.3 Kubernetes マニフェスト例

**コード例 6: 本番レベルの Kubernetes Deployment**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
  labels:
    app: api-server
    version: v1.2.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # ローリング更新中の最大追加 Pod 数
      maxUnavailable: 0     # 更新中に利用不可にならない
  template:
    metadata:
      labels:
        app: api-server
        version: v1.2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: api-server
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: api
          image: registry.example.com/api-server:v1.2.0
          ports:
            - containerPort: 3000
              protocol: TCP
          env:
            - name: NODE_ENV
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: database-url
          resources:
            requests:
              cpu: "250m"       # 0.25 コア
              memory: "128Mi"
            limits:
              cpu: "1000m"      # 1 コア
              memory: "512Mi"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 3000
            initialDelaySeconds: 15
            periodSeconds: 20
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health/live
              port: 3000
            failureThreshold: 30
            periodSeconds: 2
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: config
              mountPath: /app/config
              readOnly: true
      volumes:
        - name: tmp
          emptyDir:
            medium: Memory
            sizeLimit: 64Mi
        - name: config
          configMap:
            name: api-config
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: api-server
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: api-server
  namespace: production
spec:
  type: ClusterIP
  selector:
    app: api-server
  ports:
    - port: 80
      targetPort: 3000
      protocol: TCP
---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### 8.4 軽量 Kubernetes の選択肢

```
Kubernetes ディストリビューションの比較:

  ┌─────────────────────────────────────────────────────────────┐
  │               フル K8s (kubeadm / kops)                     │
  │  ████████████████████████████████████████  リソース: 大      │
  │  Control Plane: 3+ nodes (HA)                               │
  │  メモリ: 2GB+ per node                                      │
  │  用途: 大規模本番環境                                        │
  ├─────────────────────────────────────────────────────────────┤
  │               K3s (Rancher)                                 │
  │  ████████████████████  リソース: 中                          │
  │  シングルバイナリ (~70MB)                                    │
  │  メモリ: 512MB で動作                                       │
  │  SQLite / etcd 選択可                                       │
  │  用途: エッジ, IoT, 小〜中規模, CI/CD                        │
  ├─────────────────────────────────────────────────────────────┤
  │               k0s (Mirantis)                                │
  │  ██████████████████  リソース: 中                            │
  │  ゼロ依存のシングルバイナリ                                  │
  │  メモリ: 300MB で動作                                       │
  │  用途: エッジ, エアギャップ環境                              │
  ├─────────────────────────────────────────────────────────────┤
  │               minikube / kind / k3d                         │
  │  ████████████  リソース: 小                                  │
  │  ローカル開発専用                                           │
  │  minikube: VM ベース, kind: Docker-in-Docker                │
  │  k3d: K3s in Docker (最速)                                  │
  │  用途: 開発, テスト, 学習                                    │
  └─────────────────────────────────────────────────────────────┘
```
