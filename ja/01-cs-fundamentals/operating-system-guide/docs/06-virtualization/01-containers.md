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


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [仮想マシンの基礎](./00-vm-basics.md) の内容を理解していること

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

---

## 9. CI/CD パイプラインにおけるコンテナ活用

### 9.1 コンテナベースの CI/CD アーキテクチャ

```
コンテナを活用した CI/CD パイプライン:

  Developer
      │
      │ git push
      ▼
  ┌────────────────────────────────────────────────────────────┐
  │ CI Pipeline (GitHub Actions / GitLab CI / Jenkins)         │
  │                                                            │
  │  Stage 1: Build                                            │
  │  ┌──────────────────────────────────────────┐              │
  │  │ docker build --target builder -t app:ci  │              │
  │  │ → ソースのビルドとテスト実行              │              │
  │  └──────────────────────────────────────────┘              │
  │           │                                                │
  │           ▼                                                │
  │  Stage 2: Test                                             │
  │  ┌──────────────────────────────────────────┐              │
  │  │ docker compose -f docker-compose.test.yml│              │
  │  │ → 統合テスト（DB, Redis 等と結合）        │              │
  │  └──────────────────────────────────────────┘              │
  │           │                                                │
  │           ▼                                                │
  │  Stage 3: Scan                                             │
  │  ┌──────────────────────────────────────────┐              │
  │  │ trivy image app:ci                       │              │
  │  │ → 脆弱性スキャン (CRITICAL で失敗)        │              │
  │  └──────────────────────────────────────────┘              │
  │           │                                                │
  │           ▼                                                │
  │  Stage 4: Push                                             │
  │  ┌──────────────────────────────────────────┐              │
  │  │ docker push registry/app:v1.2.0          │              │
  │  │ docker push registry/app:latest          │              │
  │  └──────────────────────────────────────────┘              │
  └────────────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────────┐
  │ CD Pipeline                                                │
  │                                                            │
  │  ┌────────────────┐    ┌─────────────────────────────┐    │
  │  │ GitOps (ArgoCD) │───►│ Kubernetes Cluster           │    │
  │  │ マニフェスト同期 │    │ Rolling Update               │    │
  │  └────────────────┘    │ → v1.1.0 → v1.2.0           │    │
  │                        └─────────────────────────────┘    │
  └────────────────────────────────────────────────────────────┘
```

### 9.2 GitHub Actions によるコンテナ CI/CD

**コード例 7: GitHub Actions ワークフロー**

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build test image
        uses: docker/build-push-action@v5
        with:
          context: .
          target: builder
          load: true
          tags: app:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run unit tests
        run: docker run --rm app:test npm test

      - name: Run integration tests
        run: |
          docker compose -f docker-compose.test.yml up -d
          docker compose -f docker-compose.test.yml run --rm test
          docker compose -f docker-compose.test.yml down -v

  security-scan:
    needs: build-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build production image
        uses: docker/build-push-action@v5
        with:
          context: .
          load: true
          tags: app:scan

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:scan
          format: sarif
          output: trivy-results.sarif
          severity: CRITICAL,HIGH

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

  publish:
    needs: [build-and-test, security-scan]
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          platforms: linux/amd64,linux/arm64
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 10. アンチパターンと対策

### 10.1 アンチパターン 1: 「Fat Container」（肥大化コンテナ）

```
問題:
  1つのコンテナに複数のプロセスを詰め込む
  「仮想マシンのようにコンテナを使う」

  NG 例:
  ┌──────────────────────────────────────┐
  │ Fat Container                        │
  │                                      │
  │  ┌────────┐ ┌────────┐ ┌────────┐   │
  │  │ nginx  │ │ Node.js│ │ cron   │   │
  │  └────────┘ └────────┘ └────────┘   │
  │  ┌────────┐ ┌────────┐              │
  │  │ Redis  │ │ sshd   │              │
  │  └────────┘ └────────┘              │
  │                                      │
  │  supervisord で全プロセスを管理       │
  │  → イメージサイズ 2GB+               │
  │  → ログ管理が複雑                    │
  │  → 個別スケーリング不可              │
  │  → 障害分離ができない                │
  └──────────────────────────────────────┘

  OK 例:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ nginx    │ │ Node.js  │ │ Redis    │
  │ container│ │ container│ │ container│
  │ 25MB     │ │ 180MB    │ │ 30MB     │
  └──────────┘ └──────────┘ └──────────┘
  各サービスが独立
  → 個別スケーリング可能
  → 障害分離が明確
  → イメージの再利用性が高い
  → ログは stdout/stderr へ

対策:
  - 1コンテナ = 1プロセスの原則
  - サイドカーパターンで補助プロセスを分離
  - Docker Compose / K8s で複数コンテナを連携
  - 例外: 初期化スクリプト、シグナルハンドラは許容
```

### 10.2 アンチパターン 2: 「Latest タグ依存」

```
問題:
  本番環境で :latest タグを使用する

  NG 例:
  # Dockerfile
  FROM node:latest          # どのバージョンか不明
  ...

  # K8s Deployment
  image: my-app:latest      # ロールバック不可能

  なぜ危険か:
  ┌──────────────────────────────────────────────────────────┐
  │ Day 1: docker pull node:latest → Node.js 20.10.0        │
  │ Day 2: docker pull node:latest → Node.js 20.11.0 (自動) │
  │ Day 3: docker pull node:latest → Node.js 21.0.0 (破壊!) │
  │                                                          │
  │ → ビルドの再現性がない                                   │
  │ → 本番と開発で異なるバージョンが動く                      │
  │ → 障害時のロールバックが困難                              │
  │ → K8s の imagePullPolicy: Always で毎回 pull             │
  └──────────────────────────────────────────────────────────┘

  OK 例:
  # Dockerfile
  FROM node:20.11.0-slim     # 完全なバージョン指定
  ...

  # さらに良い: ダイジェスト指定
  FROM node:20.11.0-slim@sha256:abc123...

  # K8s Deployment
  image: registry.example.com/my-app:v1.2.0  # セマンティックバージョン

対策:
  - ベースイメージは必ずバージョンを固定
  - 本番デプロイは必ずタグ（v1.2.0）またはダイジェストを使用
  - CI/CD で自動的にバージョンタグを付与
  - Dependabot / Renovate でベースイメージの更新を管理
  - イメージの署名と検証（cosign / Notary）
```

### 10.3 アンチパターン 3: 「Docker ソケットマウント」

```
問題:
  Docker ソケットをコンテナにマウントする

  NG 例:
  docker run -v /var/run/docker.sock:/var/run/docker.sock my-tool

  なぜ危険か:
  → コンテナから Docker デーモンを完全制御可能
  → ホストの任意のファイルにアクセス可能（特権コンテナ作成）
  → 事実上のホスト root 権限と同等

  docker run -v /var/run/docker.sock:/var/run/docker.sock \
    alpine sh -c "
      # ホストの / をマウントしたコンテナを作成
      docker run -v /:/host alpine cat /host/etc/shadow
    "
  → コンテナエスケープの典型的な手口

対策:
  - Docker ソケットのマウントは原則禁止
  - CI/CD では Docker-in-Docker (DinD) や Kaniko を使用
  - K8s では PodSecurityPolicy / PodSecurityStandard で制限
  - 必要な場合は Docker Socket Proxy (Tecnativa) で API を制限
```

---

## 11. 演習問題

### 演習 1: 基礎レベル — Dockerfile の最適化

```
課題:
  以下の非効率な Dockerfile を最適化せよ。

  === 最適化前 ===
  FROM ubuntu:latest
  RUN apt-get update
  RUN apt-get install -y nodejs npm python3 gcc make
  COPY . /app
  WORKDIR /app
  RUN npm install
  RUN npm run build
  EXPOSE 3000
  CMD ["node", "dist/server.js"]

  最適化の観点:
  1. ベースイメージの選択（サイズ削減）
  2. レイヤーキャッシュの活用（ビルド高速化）
  3. マルチステージビルド（最終イメージの軽量化）
  4. セキュリティ（非 root 実行、不要ツール排除）
  5. .dockerignore の作成

  === 模範解答 ===
  # ステージ 1: ビルド
  FROM node:20-slim AS builder
  WORKDIR /app
  COPY package.json package-lock.json ./
  RUN npm ci
  COPY . .
  RUN npm run build

  # ステージ 2: 本番
  FROM node:20-slim AS production
  RUN groupadd -r appuser && useradd -r -g appuser appuser
  WORKDIR /app
  COPY --from=builder /app/package.json /app/package-lock.json ./
  RUN npm ci --production && npm cache clean --force
  COPY --from=builder /app/dist ./dist
  USER appuser
  EXPOSE 3000
  HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget --spider -q http://localhost:3000/health || exit 1
  CMD ["node", "dist/server.js"]

  改善ポイント:
  - ubuntu:latest → node:20-slim（サイズ削減、不要パッケージ排除）
  - RUN 命令の統合はあえてしない（キャッシュの粒度を保持）
  - package.json を先にコピー（依存関係キャッシュ活用）
  - マルチステージで gcc/make 等のビルドツールを排除
  - USER 命令で非 root 実行
  - HEALTHCHECK の追加
```

### 演習 2: 中級レベル — Docker Compose によるマイクロサービス構築

```
課題:
  以下の要件を満たす docker-compose.yml を作成せよ。

  要件:
  - フロントエンド: React アプリ (Nginx で配信)
  - バックエンド: Node.js API (3 レプリカ)
  - データベース: PostgreSQL (データ永続化)
  - キャッシュ: Redis
  - ネットワーク: フロント用とバック用を分離
  - セキュリティ: DB/Redis は外部アクセス不可
  - ヘルスチェック: 全サービスに設定

  ヒント:
  - networks の internal オプション
  - depends_on の condition
  - deploy.resources でリソース制限
  - volumes の named volume

  評価基準:
  □ サービス間の依存関係が正しい
  □ ネットワーク分離が適切
  □ データが永続化されている
  □ ヘルスチェックが全サービスに設定されている
  □ リソース制限が設定されている
  □ 環境変数で機密情報を外部化している
```

### 演習 3: 上級レベル — Kubernetes デプロイメント設計

```
課題:
  以下のアプリケーションを Kubernetes にデプロイするマニフェストを設計せよ。

  アプリケーション構成:
  - Web API: 3 レプリカ、CPU/メモリの自動スケーリング
  - ワーカー: 2 レプリカ、キュー処理
  - PostgreSQL: StatefulSet、永続ボリューム
  - Redis: Sentinel 構成

  設計要件:
  1. セキュリティ:
     - Pod Security Standards: restricted
     - NetworkPolicy でサービス間通信を制限
     - Secret は外部シークレットストアから取得
     - 全コンテナ non-root、read-only rootfs

  2. 可用性:
     - Pod Disruption Budget (PDB)
     - Pod Topology Spread Constraints
     - Rolling Update (maxUnavailable: 0)
     - Liveness / Readiness / Startup Probe

  3. 可観測性:
     - Prometheus メトリクスエンドポイント
     - 構造化ログ (JSON)
     - 分散トレーシング (OpenTelemetry)

  4. リソース管理:
     - Resource Requests / Limits
     - LimitRange / ResourceQuota
     - HPA (CPU 70%, メモリ 80% で Scale-up)
     - VPA (推奨値の自動調整)

  評価基準:
  □ YAML マニフェストが正しい構文である
  □ セキュリティ要件を全て満たしている
  □ 可用性要件を全て満たしている
  □ Zero-downtime deployment が実現できる
  □ コスト効率的なリソース設定である
  □ 障害シナリオへの対策が考慮されている
```

---

## 12. コンテナの将来と新潮流

### 12.1 WebAssembly (Wasm) コンテナ

```
Wasm コンテナの位置づけ:

  隔離レベルとオーバーヘッドの関係:

  強  │ VM (KVM/Xen)
  い  │   ● 数百MB, 数秒起動
  隔  │
  離  │ Kata Containers
      │   ● 数十MB, 1秒以内
      │
      │ gVisor
      │   ● 数十MB, 100ms
      │
      │ 従来のコンテナ (runc)
      │   ● 数MB, 50ms
      │
      │ Wasm コンテナ
  弱  │   ● 数KB〜MB, 1ms 以下
  い  └──────────────────────────────►
       小さい    オーバーヘッド    大きい

  Wasm の利点:
  - 起動時間: コールドスタート 1ms 以下
  - メモリ: 数 KB〜数 MB
  - セキュリティ: サンドボックスが言語レベルで保証
  - ポータビリティ: CPU アーキテクチャ非依存
  - 多言語: Rust, Go, C/C++, Python, JS, ...

  制約:
  - ファイルシステムアクセスが制限的 (WASI)
  - ネットワーク機能が発展途上
  - エコシステムがまだ成熟していない
  - 全てのワークロードに適するわけではない
```

### 12.2 eBPF によるコンテナ可観測性

```
eBPF (extended Berkeley Packet Filter):

  カーネル内でサンドボックス化されたプログラムを実行
  → コンテナの可観測性とセキュリティを革新

  従来の方法:
  App → syscall → Kernel → (後から) ログ分析
                              ↑ オーバーヘッド大

  eBPF:
  App → syscall → Kernel ← eBPF プログラム (in-kernel)
                              ↑ リアルタイム、低オーバーヘッド

  代表的なツール:
  ┌─────────────────────────────────────────────────┐
  │ Cilium        K8s ネットワーキング + セキュリティ  │
  │               kube-proxy 代替、NetworkPolicy     │
  │               L3/L4/L7 の可視化                   │
  ├─────────────────────────────────────────────────┤
  │ Tetragon      ランタイムセキュリティ               │
  │               プロセス実行、ファイルアクセス監視     │
  │               ネットワーク接続の追跡               │
  ├─────────────────────────────────────────────────┤
  │ Pixie         アプリケーション可観測性             │
  │               コード変更なしで HTTP/gRPC/SQL 追跡 │
  │               サービスマップの自動生成              │
  ├─────────────────────────────────────────────────┤
  │ Falco         ランタイム脅威検知                   │
  │               不審なシステムコール検出              │
  │               コンテナエスケープの検知              │
  └─────────────────────────────────────────────────┘
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 13. まとめ

| 概念 | ポイント |
|------|---------|
| Namespace | PID, NET, MNT, UTS, IPC, User, Cgroup, Time の 8 種類でリソースの可視性を隔離 |
| cgroups | CPU, メモリ, I/O, PID 数のリソース制限と監視。v2 で統合的な管理 |
| Union FS | OverlayFS による CoW レイヤー構造。読み取り専用レイヤーの共有でディスク節約 |
| seccomp | システムコールフィルタリング。約 300 の syscall から必要なもののみ許可 |
| OCI | Runtime Spec, Image Spec, Distribution Spec の 3 仕様でコンテナを標準化 |
| Docker | イメージビルド + 実行の事実上の標準。containerd + runc の上に構築 |
| Podman | デーモンレス・ルートレスの代替。Docker CLI 互換 |
| Kubernetes | コンテナオーケストレーション。Pod, Service, Deployment が基本概念 |
| セキュリティ | 多層防御: 非 root, Capabilities 最小化, seccomp, read-only rootfs |
| CI/CD | コンテナベースのパイプラインで再現性のあるビルド・テスト・デプロイ |
| Wasm | 次世代のコンテナ代替候補。超軽量・高速起動だがエコシステムは発展途上 |

---

## 14. FAQ（よくある質問）

### Q1: Docker と Podman はどちらを使うべきか？

**A**: 用途によって異なる。開発環境では Docker Desktop の利便性が高く、エコシステムも充実している。しかし、セキュリティが重視される本番環境では Podman の「デーモンレス・ルートレス」アーキテクチャが有利である。また、Docker Desktop は商用利用でライセンス料が発生する場合がある（従業員 250 人以上または年間収益 1000 万ドル以上の企業）。Podman は完全に無償かつオープンソースである。多くの組織では開発に Docker、本番に containerd（K8s 経由）という構成を採用している。

### Q2: コンテナは仮想マシンを完全に置き換えるのか？

**A**: 置き換えない。両者は異なる用途に最適化されており、共存が続く。コンテナはマイクロサービス、CI/CD、ステートレスなワークロードに適している。一方、VM は異種 OS の混在（Linux と Windows の共存）、強固な隔離が必要なマルチテナント環境、レガシーアプリケーションの移行、カーネルモジュールのテストなどに引き続き必要である。Kata Containers のように、VM の隔離強度とコンテナの運用性を組み合わせるハイブリッドアプローチも存在する。

### Q3: Kubernetes は小規模プロジェクトにも必要か？

**A**: 多くの場合不要である。Kubernetes は学習コストと運用コストが高く、小規模プロジェクトではオーバーエンジニアリングになりやすい。代替手段として以下がある。

- **Docker Compose**: 単一サーバーで十分な場合の最適解。設定がシンプルで学習コストが低い
- **マネージドサービス**: AWS ECS/Fargate, Google Cloud Run, Azure Container Apps。K8s の複雑さなしにコンテナを実行
- **K3s**: どうしても K8s の機能が必要な場合の軽量代替。512MB のメモリで動作

K8s が適するのは、複数チームが数十以上のサービスを運用し、自動スケーリング、ゼロダウンタイムデプロイ、サービスメッシュ等の高度な機能が必要な場合である。

### Q4: Alpine ベースイメージは常に最適か？

**A**: 必ずしもそうではない。Alpine は musl libc を使用しており、glibc を前提としたアプリケーションで互換性問題が発生することがある。特に Python のネイティブ拡張や Node.js のネイティブアドオンで問題が報告されている。サイズが最小であることのメリットと、デバッグの困難さ（シェルの制約等）を天秤にかける必要がある。代替として Debian slim 系（`node:20-slim`, `python:3.12-slim`）が良いバランスを提供する。distroless イメージはさらに小さく、シェルすら含まないため攻撃対象面を最小化できるが、デバッグ時にエフェメラルコンテナ等の対策が必要になる。

### Q5: コンテナ内でデータベースを動かすべきか？

**A**: 開発・テスト環境では積極的に推奨する。本番環境では慎重な検討が必要である。コンテナ DB の課題として、データ永続化の設計（Volume 管理）、パフォーマンス（OverlayFS のオーバーヘッド）、バックアップ/リストアの運用、HA 構成の複雑さがある。多くの組織ではマネージド DB サービス（RDS, Cloud SQL 等）を選択する。ただし、Kubernetes の StatefulSet と Operator パターン（CloudNativePG, Crunchy Postgres Operator 等）の成熟により、本番でのコンテナ DB 運用も現実的になりつつある。

---

## 15. 用語集

| 用語 | 説明 |
|------|------|
| OCI | Open Container Initiative。コンテナの標準仕様を策定する団体 |
| CRI | Container Runtime Interface。K8s とコンテナランタイム間のインターフェース |
| CNI | Container Network Interface。コンテナネットワーキングのプラグインインターフェース |
| CSI | Container Storage Interface。コンテナストレージのプラグインインターフェース |
| CoW | Copy-on-Write。書き込み時にのみデータをコピーする戦略 |
| DinD | Docker-in-Docker。Docker コンテナ内で Docker を実行する手法 |
| distroless | Google が提供するアプリケーション実行に最低限必要なファイルのみを含むベースイメージ |
| etcd | K8s のクラスタ状態を保存する分散 Key-Value ストア |
| HPA | Horizontal Pod Autoscaler。メトリクスに基づく Pod の水平スケーリング |
| Init Container | Pod のメインコンテナの前に実行される初期化用コンテナ |
| Sidecar | メインコンテナを補助する同一 Pod 内のコンテナ（ログ収集、プロキシ等） |
| StatefulSet | ステートフルなアプリケーション（DB 等）のための K8s リソース |
| Wasm | WebAssembly。ブラウザ外でも動作するポータブルなバイナリフォーマット |
| WASI | WebAssembly System Interface。Wasm のシステムインターフェース仕様 |

---

## 次に読むべきガイド


---

## 参考文献

1. Lukša, M. "Kubernetes in Action." 2nd Ed, Manning, 2022.
2. Kane, S. et al. "Docker: Up & Running." 3rd Ed, O'Reilly, 2023.
3. Rice, L. "Container Security: Fundamental Technology Concepts that Protect Containerized Applications." O'Reilly, 2020.
4. Hausenblas, M. & Cindy Sridharan. "Cloud Native Infrastructure." O'Reilly, 2017.
5. Burns, B. et al. "Kubernetes: Up and Running." 3rd Ed, O'Reilly, 2022.
6. Linux man pages: namespaces(7), cgroups(7), capabilities(7), seccomp(2).
7. Open Container Initiative Specifications. https://opencontainers.org/
8. CNCF Cloud Native Landscape. https://landscape.cncf.io/
9. NIST SP 800-190 "Application Container Security Guide." 2017.
10. CIS Docker Benchmark. Center for Internet Security, 2023.
