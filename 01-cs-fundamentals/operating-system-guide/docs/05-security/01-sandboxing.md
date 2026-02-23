# サンドボックスと隔離

> サンドボックスは「信頼できないコードを安全な砂場に閉じ込めて、システムへの影響を最小化する」技術。

## この章で学ぶこと

- [ ] サンドボックスの概念と設計原則を理解する
- [ ] 主要な隔離技術を比較できる
- [ ] Linux Namespaces の各種別を詳細に理解する
- [ ] cgroups v2 によるリソース制御を実践できる
- [ ] コンテナの隔離メカニズムを知る
- [ ] gVisor, Firecracker 等の高度な隔離技術を理解する
- [ ] ブラウザ・モバイルOS・デスクトップOSのサンドボックスを比較できる
- [ ] seccomp-bpf のフィルタを設計できる

---

## 1. サンドボックスの基本概念

### 1.1 サンドボックスとは

```
サンドボックス（Sandbox）:
  信頼境界を設けて、プログラムの実行環境を制限する技術

  設計原則:
  ┌──────────────────────────────────────────────────┐
  │ 1. 最小権限の原則:                                │
  │    必要最小限のリソースのみアクセスを許可          │
  │                                                    │
  │ 2. 隔離:                                          │
  │    サンドボックス内のプロセスは外部に影響しない    │
  │                                                    │
  │ 3. 仲介（Mediation）:                             │
  │    すべてのリソースアクセスをチェックポイントで    │
  │    検査する                                        │
  │                                                    │
  │ 4. 防御の深層（Defense in Depth）:                 │
  │    複数の隔離レイヤーを重ねて防御する              │
  │    → 1つの層が突破されても他の層が防ぐ            │
  └──────────────────────────────────────────────────┘

  サンドボックスの分類:
  ┌──────────────────────────────────────────────────┐
  │ OS レベル:                                        │
  │   → Namespaces, cgroups, chroot, jail             │
  │   → コンテナ（Docker, Podman）                    │
  │   → VM（KVM, Xen, Hyper-V）                      │
  │                                                    │
  │ アプリケーションレベル:                            │
  │   → ブラウザサンドボックス（Chromium）             │
  │   → Java SecurityManager（非推奨）                │
  │   → .NET Code Access Security                     │
  │   → WebAssembly（Wasm）サンドボックス              │
  │                                                    │
  │ 言語レベル:                                        │
  │   → Rust の所有権システム                          │
  │   → Deno のパーミッションシステム                  │
  │   → Wasm の線形メモリモデル                        │
  │                                                    │
  │ ハードウェアレベル:                                │
  │   → Intel SGX / TDX                               │
  │   → ARM TrustZone / CCA                           │
  │   → AMD SEV-SNP                                   │
  └──────────────────────────────────────────────────┘
```

### 1.2 隔離レベルの比較

```
隔離レベルの比較:

  弱い隔離 ←────────────────────→ 強い隔離
  プロセス   chroot  namespace   コンテナ   gVisor   VM     TEE
  分離       分離    + cgroup    (Docker)  (sandbox) (KVM)  (SGX)

  各レベルの詳細:
  ┌──────────────────────────────────────────────────────────────┐
  │ レベル      │ 隔離対象           │ 攻撃面     │ 性能影響   │
  ├─────────────┼────────────────────┼────────────┼────────────┤
  │ プロセス    │ メモリ空間のみ     │ 非常に広い │ なし       │
  │ chroot      │ + ファイルシステム │ 広い       │ ほぼなし   │
  │ Namespace   │ + PID,Net,IPC等    │ 中程度     │ 最小       │
  │ コンテナ    │ + seccomp,cap      │ 中程度     │ 最小       │
  │ gVisor      │ + システムコール   │ 狭い       │ 10-30%     │
  │ microVM     │ + 仮想化           │ 狭い       │ 5-15%      │
  │ VM          │ ハードウェアレベル │ 非常に狭い │ 5-10%      │
  │ TEE         │ + 暗号化メモリ     │ 最小       │ 10-30%     │
  └─────────────┴────────────────────┴────────────┴────────────┘

  脱獄（Escape）の難易度:
  ┌──────────────────────────────────────────────────┐
  │ chroot: 比較的容易                                │
  │   → chroot 内で root 権限があれば脱獄可能        │
  │   → mknod, mount, ptrace 等を使った攻撃          │
  │   → 本格的なセキュリティ境界としては不十分       │
  │                                                    │
  │ コンテナ: 中程度の難易度                          │
  │   → カーネルの脆弱性を利用した脱獄事例あり       │
  │   → CVE-2019-5736 (runc 脆弱性)                  │
  │   → CVE-2020-15257 (containerd 脆弱性)           │
  │   → 適切な設定で大幅にリスク軽減可能             │
  │                                                    │
  │ VM: 非常に困難                                    │
  │   → ハイパーバイザの脆弱性が必要                 │
  │   → VENOM (CVE-2015-3456) 等の事例はあるが稀     │
  │   → 攻撃には高度な技術が必要                     │
  │                                                    │
  │ TEE: 極めて困難                                   │
  │   → ハードウェアレベルの保護                     │
  │   → サイドチャネル攻撃で一部情報漏洩の事例あり   │
  │   → Spectre/Meltdown 系の攻撃に注意             │
  └──────────────────────────────────────────────────┘
```

---

## 2. chroot と FreeBSD Jail

### 2.1 chroot

```
chroot（Change Root）:
  ファイルシステムのルートを変更
  → プロセスから見えるファイルを制限
  → 最も古い隔離技術（1979年, Unix V7）

  chroot の仕組み:
  ┌──────────────────────────────────────────────────┐
  │ 通常のプロセス:                                   │
  │   / (真のルート)                                  │
  │   ├── etc/                                        │
  │   ├── usr/                                        │
  │   ├── home/                                       │
  │   └── var/                                        │
  │                                                    │
  │ chroot されたプロセス:                             │
  │   /srv/jail/ ← これが / に見える                 │
  │   ├── etc/   (jail内の設定)                       │
  │   ├── usr/   (最小限のバイナリ)                   │
  │   └── tmp/                                        │
  │   → /srv/jail/ の外は見えない                    │
  └──────────────────────────────────────────────────┘

  chroot の限界:
  - root 権限があれば脱獄可能
  - ネットワーク、プロセス、IPC は隔離されない
  - /proc, /sys はマウントされていれば見える
  - セキュリティ機能ではなく、環境分離ツール
```

```bash
# chroot 環境の構築

# 1. ディレクトリ構造の作成
sudo mkdir -p /srv/jail/{bin,lib,lib64,etc,usr/lib,dev,proc}

# 2. 必要なバイナリのコピー
sudo cp /bin/bash /srv/jail/bin/
sudo cp /bin/ls /srv/jail/bin/
sudo cp /bin/cat /srv/jail/bin/

# 3. 依存ライブラリのコピー
# ldd でライブラリを確認してコピー
ldd /bin/bash
# linux-vdso.so.1 (0x00007fff...)
# libtinfo.so.6 => /lib/x86_64-linux-gnu/libtinfo.so.6
# libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
# /lib64/ld-linux-x86-64.so.2

sudo cp /lib/x86_64-linux-gnu/libtinfo.so.6 /srv/jail/lib/
sudo cp /lib/x86_64-linux-gnu/libc.so.6 /srv/jail/lib/
sudo cp /lib64/ld-linux-x86-64.so.2 /srv/jail/lib64/

# 4. chroot に入る
sudo chroot /srv/jail /bin/bash
# → / は /srv/jail を指す
ls /        # bin lib lib64 etc usr dev proc
# → /srv/jail の外にはアクセスできない

# 5. 脱獄の例（root権限がある場合）
# ※ 教育目的のみ。実際のシステムでは試さないこと
mkdir /tmp/escape
chroot /tmp/escape
cd ../../../..   # 真のルートに到達
chroot .         # 真のルートに chroot
# → chroot はセキュリティ境界としては不十分

# chroot の実務的な使い方:
# - ビルド環境の隔離（debootstrap + chroot）
# - BIND DNS サーバーの隔離
# - パッケージのビルド環境
# - 壊れたシステムの修復（rescue mode）
```

### 2.2 FreeBSD Jail

```
FreeBSD Jail:
  chroot の強化版。プロセス、ネットワーク、ユーザーも隔離
  → 2000年に導入。コンテナの先駆け
  → Linux の Namespace + cgroups に相当

  Jail の特徴:
  ┌──────────────────────────────────────────────────┐
  │ ファイルシステムの隔離:                           │
  │   → chroot と同様だが脱獄が困難                  │
  │                                                    │
  │ プロセスの隔離:                                    │
  │   → Jail 内のプロセスは Jail 外を見えない        │
  │   → Jail 内の root でも制限される                │
  │                                                    │
  │ ネットワークの隔離:                               │
  │   → Jail ごとに IP アドレスを割り当て             │
  │   → VNET（仮想ネットワークスタック）対応          │
  │                                                    │
  │ ユーザーの隔離:                                    │
  │   → Jail 内の root は Jail 外にアクセスできない  │
  │   → securelevel で権限をさらに制限               │
  └──────────────────────────────────────────────────┘

  Jail の設定例（/etc/jail.conf）:
  webserver {
      host.hostname = "web.jail.local";
      ip4.addr = "10.0.0.2";
      path = "/jails/webserver";
      exec.start = "/bin/sh /etc/rc";
      exec.stop = "/bin/sh /etc/rc.shutdown";
      mount.devfs;
      allow.raw_sockets;
  }

  管理コマンド:
  # jail -c webserver        # Jail の起動
  # jls                      # Jail の一覧
  # jexec webserver /bin/sh  # Jail 内でコマンド実行
```

---

## 3. Linux Namespaces

### 3.1 Namespace の種類と詳細

```
Linux Namespaces:
  OSリソースをプロセスごとに分離
  → コンテナ技術の基盤

  Namespace の種類:
  ┌──────────────┬──────────────────────────┬──────────┐
  │ Namespace    │ 分離対象                  │ カーネル │
  ├──────────────┼──────────────────────────┼──────────┤
  │ PID          │ プロセスID空間            │ 2.6.24   │
  │ Network      │ ネットワークスタック      │ 2.6.29   │
  │ Mount        │ マウントポイント          │ 2.4.19   │
  │ UTS          │ ホスト名とドメイン名      │ 2.6.19   │
  │ IPC          │ プロセス間通信            │ 2.6.19   │
  │ User         │ UID/GID マッピング        │ 3.8      │
  │ Cgroup       │ cgroupの可視性            │ 4.6      │
  │ Time         │ システム時刻（CLOCK_*）   │ 5.6      │
  └──────────────┴──────────────────────────┴──────────┘
```

### 3.2 PID Namespace

```
PID Namespace:
  プロセスID空間を隔離
  → Namespace 内では PID 1 から始まる
  → 外部の PID は見えない

  PID Namespace の階層:
  ┌──────────────────────────────────────────────────┐
  │ Host PID Namespace:                               │
  │   PID 1 (systemd/init)                           │
  │   PID 100 (sshd)                                  │
  │   PID 200 (container runtime)                     │
  │     │                                              │
  │     └── Child PID Namespace:                      │
  │           PID 1 (container init) ← Host PID 201  │
  │           PID 2 (app process)    ← Host PID 202  │
  │           PID 3 (worker)         ← Host PID 203  │
  │                                                    │
  │ → 親Namespace からは子の PID が見える            │
  │ → 子Namespace からは親の PID は見えない          │
  │ → PID 1 が終了すると、Namespace内の全プロセスが  │
  │   SIGKILL を受ける                                │
  └──────────────────────────────────────────────────┘

  PID Namespace の特殊な挙動:
  - PID 1 はシグナルハンドラを登録していないシグナルを無視
  - 孤児プロセスは PID 1 に reparent される
  - /proc のマウントで正確なプロセス情報を表示
```

```bash
# PID Namespace の作成と確認

# 新しい PID Namespace でシェルを起動
sudo unshare --pid --fork --mount-proc bash
echo $$      # PID 1
ps aux       # Namespace 内のプロセスのみ表示
# USER  PID %CPU %MEM    VSZ   RSS TTY  STAT START   TIME COMMAND
# root    1  0.0  0.0   8532  5240 pts/0 S   12:00   0:00 bash
# root    2  0.0  0.0  10068  3456 pts/0 R+  12:00   0:00 ps aux

# 別のターミナルから確認
ps aux | grep bash  # ホストでは別の PID で見える

# --fork が必要な理由:
# unshare 自体は新しい Namespace に入るが、
# PID Namespace は子プロセスから有効になるため
# --fork で新しいプロセスを作成する必要がある
```

### 3.3 Network Namespace

```
Network Namespace:
  ネットワークスタック全体を隔離
  → インターフェース、ルーティングテーブル、ファイアウォール、ソケット
  → コンテナのネットワーク隔離の基盤

  Network Namespace の構成:
  ┌──────────────────────────────────────────────────┐
  │ Host Namespace:                                   │
  │   eth0: 192.168.1.100/24                         │
  │   veth-host ──┐                                   │
  │               │ veth pair（仮想イーサネット）     │
  │   Container Namespace:                            │
  │   veth-cont ──┘                                   │
  │   eth0: 172.17.0.2/16                            │
  │   lo: 127.0.0.1                                   │
  │   → 独立したネットワークスタック                 │
  └──────────────────────────────────────────────────┘

  Docker のネットワークモデル:
  ┌──────────────────────────────────────────────────┐
  │ ホスト                                            │
  │ ┌──────────────────────────────────┐              │
  │ │ docker0 ブリッジ (172.17.0.1)   │              │
  │ │  ┌─────────┐  ┌─────────┐       │              │
  │ │  │ veth1   │  │ veth2   │       │              │
  │ │  └────┬────┘  └────┬────┘       │              │
  │ └───────┼────────────┼────────────┘              │
  │         │            │                            │
  │    Container1    Container2                       │
  │    eth0:         eth0:                            │
  │    172.17.0.2    172.17.0.3                       │
  │                                                    │
  │ → NAT でホストの eth0 経由で外部通信             │
  │ → iptables の MASQUERADE ルール                  │
  └──────────────────────────────────────────────────┘
```

```bash
# Network Namespace の操作

# Namespace の作成
sudo ip netns add test-ns

# Namespace の一覧
ip netns list

# Namespace 内でコマンド実行
sudo ip netns exec test-ns ip addr
# → lo インターフェースのみ（DOWN状態）

# veth ペアの作成と設定
sudo ip link add veth-host type veth peer name veth-ns
sudo ip link set veth-ns netns test-ns

# ホスト側の設定
sudo ip addr add 10.0.0.1/24 dev veth-host
sudo ip link set veth-host up

# Namespace 側の設定
sudo ip netns exec test-ns ip addr add 10.0.0.2/24 dev veth-ns
sudo ip netns exec test-ns ip link set veth-ns up
sudo ip netns exec test-ns ip link set lo up

# 疎通確認
ping 10.0.0.2                              # ホスト → Namespace
sudo ip netns exec test-ns ping 10.0.0.1   # Namespace → ホスト

# Namespace からインターネット接続（NAT設定）
sudo ip netns exec test-ns ip route add default via 10.0.0.1
sudo iptables -t nat -A POSTROUTING -s 10.0.0.0/24 -j MASQUERADE
sudo sysctl -w net.ipv4.ip_forward=1

# Namespace 内で独立したファイアウォール
sudo ip netns exec test-ns iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo ip netns exec test-ns iptables -A INPUT -j DROP

# Namespace の削除
sudo ip netns delete test-ns
```

### 3.4 Mount Namespace

```
Mount Namespace:
  マウントポイントを隔離
  → Namespace ごとに異なるファイルシステムビューを持つ
  → コンテナのファイルシステム隔離の基盤

  Mount Namespace の動作:
  ┌──────────────────────────────────────────────────┐
  │ Host Mount Namespace:                             │
  │   /           (ext4)                              │
  │   /home       (ext4)                              │
  │   /tmp        (tmpfs)                             │
  │                                                    │
  │ Container Mount Namespace:                        │
  │   /           (overlay2 - コンテナイメージ)      │
  │   /etc/hosts  (bind mount - Docker管理)          │
  │   /proc       (procfs)                            │
  │   /sys        (sysfs - 読み取り専用)             │
  │   /dev        (devtmpfs - 制限付き)              │
  │   /app/data   (volume mount - 永続データ)        │
  │                                                    │
  │ → ホストのファイルシステムとは完全に独立         │
  │ → 必要なファイルのみ bind mount で共有           │
  └──────────────────────────────────────────────────┘

  Propagation タイプ（マウント伝播）:
  ┌───────────┬──────────────────────────────────────┐
  │ shared    │ マウントイベントが双方向に伝播       │
  │ slave     │ 親 → 子 のみ伝播                    │
  │ private   │ 伝播しない（デフォルト）             │
  │ unbindable│ private + bind mount 不可            │
  └───────────┴──────────────────────────────────────┘
```

```bash
# Mount Namespace の操作

# 新しい Mount Namespace を作成
sudo unshare --mount bash

# Namespace 内で tmpfs をマウント
mount -t tmpfs tmpfs /mnt
echo "isolated" > /mnt/test.txt
cat /mnt/test.txt  # "isolated"

# 別のターミナル（ホスト側）で確認
ls /mnt/           # 何もない → 隔離されている

# プライベートマウントの設定
mount --make-private /

# Namespace 内で /proc を再マウント（PID Namespace と組み合わせ）
sudo unshare --pid --fork --mount bash
mount -t proc proc /proc
ps aux  # Namespace 内のプロセスのみ
```

### 3.5 User Namespace

```
User Namespace:
  UID/GID のマッピングを隔離
  → Namespace 内で root（UID 0）でも外部では非特権ユーザー
  → ルートレスコンテナの基盤技術

  User Namespace のマッピング:
  ┌──────────────────────────────────────────────────┐
  │ Host:                                             │
  │   alice (UID 1000)                                │
  │                                                    │
  │ User Namespace 内:                                │
  │   root (UID 0) ← 実際には alice (UID 1000)      │
  │   → Namespace 内では root として動作             │
  │   → ホストでは alice の権限のみ                  │
  │   → Namespace 内でファイルを作成すると           │
  │     ホストでは alice 所有になる                   │
  └──────────────────────────────────────────────────┘

  /proc/PID/uid_map の内容:
  # Namespace内UID  ホストUID  マッピング数
  0                 1000       1
  → Namespace 内の UID 0 = ホストの UID 1000

  User Namespace で可能になること:
  - 非特権ユーザーによる Namespace 作成
  - Namespace 内での mount 操作
  - Namespace 内での chroot
  - Rootless コンテナの実現
  - → セキュリティ上の利点が大きい
```

```bash
# User Namespace の操作

# User Namespace の作成（非特権ユーザーでも可能）
unshare --user --map-root-user bash
id
# uid=0(root) gid=0(root) groups=0(root)
# → Namespace 内では root に見える

whoami  # root（Namespace内）

# UID マッピングの確認
cat /proc/self/uid_map
# 0  1000  1
# → Namespace内の UID 0 = ホストの UID 1000

# ファイル作成テスト
touch /tmp/test-user-ns
ls -la /tmp/test-user-ns
# → ホストでは alice (UID 1000) の所有

# Rootless コンテナ（Podman）
podman run --rm -it alpine sh
# → root 権限なしでコンテナを実行
# → User Namespace で内部は root、外部は一般ユーザー
```

### 3.6 UTS, IPC, Cgroup, Time Namespace

```
UTS Namespace:
  ホスト名とドメイン名を隔離
  → コンテナごとに異なるホスト名を設定可能
  → UTS = Unix Time Sharing

IPC Namespace:
  System V IPC と POSIX メッセージキューを隔離
  → 共有メモリ、セマフォ、メッセージキュー
  → Namespace 間でのデータ漏洩を防止

Cgroup Namespace:
  cgroup の可視性を隔離
  → コンテナ内からは自身の cgroup ツリーのみ見える
  → ホストの cgroup 構造が隠蔽される

Time Namespace (Linux 5.6+):
  CLOCK_MONOTONIC と CLOCK_BOOTTIME を隔離
  → コンテナのライブマイグレーション時に有用
  → ホストの起動時間とコンテナの起動時間を独立にできる
```

```bash
# 各 Namespace の確認

# 現在の Namespace 確認
ls -la /proc/self/ns/
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 cgroup -> 'cgroup:[4026531835]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 ipc -> 'ipc:[4026531839]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 mnt -> 'mnt:[4026531840]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 net -> 'net:[4026531992]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 pid -> 'pid:[4026531836]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 user -> 'user:[4026531837]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 uts -> 'uts:[4026531838]'

# UTS Namespace（ホスト名の隔離）
sudo unshare --uts bash
hostname container-1
hostname    # container-1
# 別ターミナルで: hostname → ホスト名は変わらない

# IPC Namespace
sudo unshare --ipc bash
ipcs        # IPC リソースは空の状態から開始
# → ホストの共有メモリやセマフォは見えない

# 複数の Namespace を同時に作成
sudo unshare --pid --fork --mount-proc \
  --net --uts --ipc --user --map-root-user bash
# → 完全に隔離された環境（≒ コンテナ）
```

---

## 4. cgroups（Control Groups）

### 4.1 cgroups v1 vs v2

```
cgroups（Control Groups）:
  プロセスのリソース使用量を制限・監視・隔離する仕組み
  → カーネル 2.6.24 で導入（v1）
  → カーネル 4.5 で v2 が導入
  → コンテナのリソース制限の基盤

  cgroups v1 vs v2:
  ┌─────────────┬──────────────────┬──────────────────┐
  │ 項目        │ v1               │ v2               │
  ├─────────────┼──────────────────┼──────────────────┤
  │ 階層構造    │ コントローラごと │ 統一された単一   │
  │             │ に独立した階層   │ 階層             │
  │ 管理        │ 複雑             │ シンプル         │
  │ 圧力監視    │ なし             │ PSI 対応        │
  │ メモリ管理  │ 不正確な場合あり │ 正確             │
  │ I/O制御     │ blkio            │ io（改善版）     │
  │ ステータス  │ レガシー         │ 推奨             │
  └─────────────┴──────────────────┴──────────────────┘

  cgroups v2 の階層構造:
  /sys/fs/cgroup/                   ← ルート cgroup
  ├── cgroup.controllers            ← 利用可能なコントローラ
  ├── cgroup.subtree_control        ← サブツリーで有効なコントローラ
  ├── system.slice/                 ← systemd サービス
  │   ├── nginx.service/
  │   │   ├── cgroup.procs          ← プロセスIDリスト
  │   │   ├── memory.max            ← メモリ上限
  │   │   ├── memory.current        ← 現在のメモリ使用量
  │   │   ├── cpu.max               ← CPU制限
  │   │   └── io.max                ← I/O制限
  │   └── postgresql.service/
  └── user.slice/                   ← ユーザーセッション
      └── user-1000.slice/
          └── session-1.scope/

  主要なコントローラ（v2）:
  ┌──────────┬──────────────────────────────────────┐
  │ cpu      │ CPU 時間の制限と重み付け             │
  │ cpuset   │ CPU コアとメモリノードの割り当て     │
  │ memory   │ メモリ使用量の制限と監視             │
  │ io       │ ブロック I/O の制限                  │
  │ pids     │ プロセス数の制限                     │
  │ rdma     │ RDMA リソースの制限                  │
  │ hugetlb  │ Huge Pages の制限                    │
  │ misc     │ その他のリソース（DRM等）            │
  └──────────┴──────────────────────────────────────┘
```

### 4.2 cgroups v2 の実践

```bash
# cgroups v2 の確認
mount | grep cgroup2
# cgroup2 on /sys/fs/cgroup type cgroup2 (rw,nosuid,nodev,noexec,relatime)

# 利用可能なコントローラの確認
cat /sys/fs/cgroup/cgroup.controllers
# cpuset cpu io memory hugetlb pids rdma misc

# ========================================
# メモリ制限の設定
# ========================================

# cgroup の作成
sudo mkdir /sys/fs/cgroup/myapp

# サブツリーのコントローラを有効化
echo "+memory +cpu +io +pids" | \
  sudo tee /sys/fs/cgroup/cgroup.subtree_control

# メモリ制限の設定
echo 256M | sudo tee /sys/fs/cgroup/myapp/memory.max
echo 200M | sudo tee /sys/fs/cgroup/myapp/memory.high
# memory.max: ハード制限（超過するとOOM Killer発動）
# memory.high: ソフト制限（超過するとスロットリング）

# スワップの制限
echo 0 | sudo tee /sys/fs/cgroup/myapp/memory.swap.max

# プロセスの追加
echo $$ | sudo tee /sys/fs/cgroup/myapp/cgroup.procs

# メモリ使用状況の確認
cat /sys/fs/cgroup/myapp/memory.current    # 現在の使用量
cat /sys/fs/cgroup/myapp/memory.stat       # 詳細な統計
cat /sys/fs/cgroup/myapp/memory.events     # OOM等のイベント

# ========================================
# CPU 制限の設定
# ========================================

# CPU 時間の制限（50%）
echo "50000 100000" | sudo tee /sys/fs/cgroup/myapp/cpu.max
# 100000μsの期間中、50000μsのCPU時間を使用可能 = 50%

# CPU の重み（相対的な優先度）
echo 100 | sudo tee /sys/fs/cgroup/myapp/cpu.weight
# デフォルト: 100, 範囲: 1-10000
# weight=200 のグループは weight=100 の2倍のCPU時間を得る

# CPU コアの割り当て
echo "0-1" | sudo tee /sys/fs/cgroup/myapp/cpuset.cpus
# CPU 0 と CPU 1 のみ使用可能

# ========================================
# I/O 制限の設定
# ========================================

# デバイスの確認
lsblk
# sda  8:0

# I/O 帯域幅の制限
echo "8:0 rbps=10485760 wbps=5242880" | \
  sudo tee /sys/fs/cgroup/myapp/io.max
# sda の読み取り: 10MB/s, 書き込み: 5MB/s

# I/O の重み
echo "8:0 200" | sudo tee /sys/fs/cgroup/myapp/io.weight
# デフォルト: 100, 範囲: 1-10000

# ========================================
# PID 数の制限
# ========================================

# プロセス数の制限（fork bomb 対策）
echo 100 | sudo tee /sys/fs/cgroup/myapp/pids.max
# → 100プロセスまで

# 現在のプロセス数
cat /sys/fs/cgroup/myapp/pids.current

# ========================================
# PSI（Pressure Stall Information）の監視
# ========================================

# リソース圧力の確認
cat /sys/fs/cgroup/myapp/memory.pressure
# some avg10=0.00 avg60=0.00 avg300=0.00 total=0
# full avg10=0.00 avg60=0.00 avg300=0.00 total=0
# → some: 一部のタスクが待機中, full: すべてのタスクが待機中

cat /sys/fs/cgroup/myapp/cpu.pressure
cat /sys/fs/cgroup/myapp/io.pressure

# PSI の通知を設定（メモリ圧力が5秒中500ms超えたら通知）
echo "some 500000 5000000" > /sys/fs/cgroup/myapp/memory.pressure
# → epoll/poll で監視可能

# cgroup の削除
echo $$ | sudo tee /sys/fs/cgroup/cgroup.procs  # プロセスを移動
sudo rmdir /sys/fs/cgroup/myapp
```

### 4.3 systemd と cgroups

```
systemd は cgroups v2 を使用してサービスのリソースを管理:

  サービスファイルでのリソース制限:
  ┌──────────────────────────────────────────────────┐
  │ /etc/systemd/system/myapp.service                │
  │                                                    │
  │ [Service]                                         │
  │ # メモリ制限                                      │
  │ MemoryMax=512M          # ハード制限              │
  │ MemoryHigh=400M         # ソフト制限              │
  │ MemorySwapMax=0         # スワップ禁止            │
  │                                                    │
  │ # CPU 制限                                        │
  │ CPUQuota=200%           # 2コア分のCPU時間        │
  │ CPUWeight=50            # 低い優先度              │
  │ AllowedCPUs=0-3         # 使用可能なCPUコア       │
  │                                                    │
  │ # I/O 制限                                        │
  │ IOWeight=100                                      │
  │ IOReadBandwidthMax=/dev/sda 50M                   │
  │ IOWriteBandwidthMax=/dev/sda 20M                  │
  │                                                    │
  │ # プロセス数制限                                  │
  │ TasksMax=64                                       │
  │                                                    │
  │ # セキュリティ強化                                │
  │ ProtectSystem=strict    # / を読み取り専用に      │
  │ ProtectHome=true        # /home を隠す            │
  │ PrivateTmp=true         # 独立した /tmp           │
  │ NoNewPrivileges=true    # 権限昇格禁止            │
  │ PrivateDevices=true     # デバイスアクセス制限    │
  └──────────────────────────────────────────────────┘
```

```bash
# systemd でのリソース監視
systemd-cgtop                    # cgroup のリソース使用量一覧
systemctl status myapp.service   # サービスの状態と cgroup 情報
systemctl show myapp.service --property=MemoryMax
systemctl show myapp.service --property=CPUQuota

# 実行中のサービスのリソース制限を変更
sudo systemctl set-property myapp.service MemoryMax=1G
sudo systemctl set-property myapp.service CPUQuota=150%

# slice の作成（関連サービスのグループ化）
# /etc/systemd/system/myapp.slice
# [Slice]
# MemoryMax=2G
# CPUQuota=400%
#
# → サービスファイルで Slice=myapp.slice を指定
```

---

## 5. 仮想マシン vs コンテナ

### 5.1 アーキテクチャ比較

```
┌──────────────────────────────────────────┐
│ 仮想マシン                                │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │App A │ │App B │ │App C │              │
│ │Guest │ │Guest │ │Guest │              │
│ │ OS   │ │ OS   │ │ OS   │              │
│ └──┬───┘ └──┬───┘ └──┬───┘              │
│ ┌──┴────────┴────────┴───┐              │
│ │ Hypervisor (KVM/Xen)    │              │
│ └──────────────────────────┘              │
│ ┌──────────────────────────┐              │
│ │ Host OS + Hardware       │              │
│ └──────────────────────────┘              │
│ → 完全な隔離、異なるOS可能                │
│ → オーバーヘッド大、起動に秒〜分          │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ コンテナ                                  │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │App A │ │App B │ │App C │              │
│ │Libs  │ │Libs  │ │Libs  │              │
│ └──┬───┘ └──┬───┘ └──┬───┘              │
│ ┌──┴────────┴────────┴───┐              │
│ │ Container Runtime       │              │
│ │ (Docker/containerd)     │              │
│ └──────────────────────────┘              │
│ ┌──────────────────────────┐              │
│ │ Host OS (共有カーネル)    │              │
│ └──────────────────────────┘              │
│ → カーネル共有、軽量                      │
│ → 起動がミリ秒、メモリ効率良              │
└──────────────────────────────────────────┘

詳細比較:
┌────────────┬──────────────────┬──────────────────┐
│ 項目       │ VM               │ コンテナ         │
├────────────┼──────────────────┼──────────────────┤
│ 隔離       │ 強い(HW分離)     │ 中程度(OS分離)   │
│ 起動       │ 秒〜分           │ ミリ秒〜秒       │
│ サイズ     │ GB               │ MB               │
│ 密度       │ 数十/ホスト      │ 数百〜数千/ホスト│
│ OS         │ 異なるOS可       │ ホストOS共有     │
│ カーネル   │ 独立             │ 共有             │
│ 性能       │ ほぼネイティブ   │ ネイティブ       │
│ セキュリティ│ 高い            │ 中程度           │
│ ライブ     │ 可能             │ 困難             │
│ マイグレーション│              │                  │
│ 用途       │ マルチテナント   │ マイクロサービス │
│ 例         │ EC2, GCE         │ ECS, GKE, EKS   │
└────────────┴──────────────────┴──────────────────┘
```

### 5.2 中間技術: gVisor, Firecracker, Kata Containers

```
gVisor（Google）:
  ユーザー空間カーネル
  → アプリのシステムコールをユーザー空間で処理
  → ホストカーネルへの攻撃面を大幅に削減

  gVisor のアーキテクチャ:
  ┌──────────────────────────────────────────────────┐
  │ アプリケーション                                  │
  │       ↓ システムコール                           │
  │ Sentry（ユーザー空間カーネル）                   │
  │   → ~200のシステムコールをユーザー空間で実装     │
  │   → メモリ管理、ファイルシステム、ネットワーク   │
  │       ↓ 限定的なシステムコール                   │
  │ Gofer（ファイルシステムプロキシ）                 │
  │       ↓                                           │
  │ Host Kernel（seccomp で制限された状態）           │
  │                                                    │
  │ 特徴:                                             │
  │ - Go 言語で実装（メモリ安全）                    │
  │ - OCI 互換（Docker, K8s で使用可能）              │
  │ - ptrace または KVM をプラットフォームとして使用  │
  │ - 性能オーバーヘッド: 10-30%（ワークロード依存）│
  │ - Google Cloud Run で使用                         │
  └──────────────────────────────────────────────────┘

Firecracker（Amazon）:
  マイクロVM（超軽量VM）
  → AWS Lambda, Fargate の基盤

  Firecracker のアーキテクチャ:
  ┌──────────────────────────────────────────────────┐
  │ 特徴:                                             │
  │ - Rust で実装（メモリ安全）                      │
  │ - 起動時間: 125ms 以下                           │
  │ - メモリ: 5MB 以下のオーバーヘッド               │
  │ - 最小限のデバイスモデル（virtio のみ）          │
  │ - KVM ベースの完全な VM 隔離                     │
  │                                                    │
  │ 通常の VM との違い:                               │
  │ - BIOS/UEFI なし → 直接カーネルブート            │
  │ - USB, PCI, グラフィックスなし                    │
  │ - virtio-net, virtio-block のみ                   │
  │ - → 攻撃面が非常に小さい                        │
  │                                                    │
  │ 用途:                                             │
  │ - サーバーレスコンピューティング                  │
  │ - マルチテナント環境                              │
  │ - 1ホストに数千のマイクロVMを実行可能             │
  └──────────────────────────────────────────────────┘

Kata Containers:
  軽量VMの中でコンテナを実行
  → VM の隔離 + コンテナの互換性

  Kata Containers のアーキテクチャ:
  ┌──────────────────────────────────────────────────┐
  │ Kubernetes / Docker                               │
  │       ↓ CRI / OCI                                │
  │ Kata Runtime                                      │
  │       ↓                                           │
  │ ┌─────────────────────┐                           │
  │ │ 軽量 VM (QEMU/CLH)  │                           │
  │ │ ┌─────────────────┐ │                           │
  │ │ │ Guest Kernel    │ │                           │
  │ │ │ + kata-agent    │ │                           │
  │ │ │ + コンテナ      │ │                           │
  │ │ └─────────────────┘ │                           │
  │ └─────────────────────┘                           │
  │                                                    │
  │ → Pod ごとに VM を作成                           │
  │ → コンテナの OCI 互換性を維持                    │
  │ → Kubernetes からは通常のコンテナに見える        │
  │ → Cloud Hypervisor (CLH) でさらに軽量化          │
  └──────────────────────────────────────────────────┘

隔離技術の比較:
┌──────────────┬──────────┬──────────┬─────────────┐
│ 技術         │ 起動時間 │ メモリ   │ セキュリティ│
├──────────────┼──────────┼──────────┼─────────────┤
│ Docker       │ ~100ms   │ ~10MB    │ 中          │
│ gVisor       │ ~150ms   │ ~30MB    │ 中〜高      │
│ Firecracker  │ ~125ms   │ ~5MB     │ 高          │
│ Kata         │ ~500ms   │ ~30MB    │ 高          │
│ 通常のVM     │ ~5s      │ ~500MB   │ 非常に高    │
└──────────────┴──────────┴──────────┴─────────────┘
```

---

## 6. アプリケーションサンドボックス

### 6.1 ブラウザのサンドボックス

```
Chromium のサンドボックスアーキテクチャ:
  世界で最も広く使われているサンドボックスの1つ

  マルチプロセスアーキテクチャ:
  ┌──────────────────────────────────────────────────┐
  │ Browser Process（特権プロセス）                   │
  │ → UI、ネットワーク、ファイルアクセスを担当       │
  │ → 唯一の高権限プロセス                           │
  │                                                    │
  │ Renderer Process（サンドボックス化）              │
  │ → HTML/CSS/JSのレンダリング                      │
  │ → サイトごとに独立プロセス（Site Isolation）      │
  │ → seccomp-bpf でシステムコール制限              │
  │ → Namespace でファイルシステム分離               │
  │ → ネットワークアクセス不可（IPC経由で依頼）     │
  │                                                    │
  │ GPU Process                                       │
  │ → グラフィックス処理を担当                       │
  │ → 中程度のサンドボックス                         │
  │                                                    │
  │ Plugin Process                                    │
  │ → 拡張機能の実行                                 │
  │ → 独自のサンドボックス                           │
  │                                                    │
  │ Network Service                                   │
  │ → ネットワーク通信を担当                         │
  │ → サンドボックス化済み                           │
  └──────────────────────────────────────────────────┘

  Chromium のサンドボックス層:
  ┌──────────────────────────────────────────────────┐
  │ Layer 1: Linux Namespace                          │
  │   → PID Namespace: 他プロセスを見えなくする      │
  │   → Network Namespace: 直接通信を禁止            │
  │   → User Namespace: 非特権ユーザーとして実行     │
  │                                                    │
  │ Layer 2: seccomp-bpf                              │
  │   → 使用可能なシステムコールを最小限に制限       │
  │   → open, exec, socket 等を禁止                  │
  │   → 必要なI/OはIPCで Browser Process に依頼     │
  │                                                    │
  │ Layer 3: ファイルシステム制限                     │
  │   → chroot + pivot_root                           │
  │   → /proc の最小限マウント                       │
  │                                                    │
  │ Layer 4: プロセスレベル                           │
  │   → No New Privileges (PR_SET_NO_NEW_PRIVS)      │
  │   → Capabilities のドロップ                      │
  │                                                    │
  │ → 4層の防御で脆弱性の影響を最小化               │
  └──────────────────────────────────────────────────┘

  Site Isolation（Spectre対策）:
  → 異なるサイトは異なるプロセスで実行
  → サイト間のメモリ空間が完全に分離
  → Spectre/Meltdown によるクロスサイト攻撃を防止
  → Chrome 67 以降でデフォルト有効
```

### 6.2 モバイルOSのサンドボックス

```
iOS のサンドボックス:
  ┌──────────────────────────────────────────────────┐
  │ App Sandbox:                                      │
  │ → 各アプリが独立したコンテナで実行               │
  │ → アプリのホームディレクトリ外へのアクセス不可   │
  │                                                    │
  │ ディレクトリ構造:                                 │
  │ /var/mobile/Containers/                           │
  │   Bundle/Application/UUID/        ← アプリバイナリ│
  │   Data/Application/UUID/          ← アプリデータ │
  │     ├── Documents/                ← ユーザーデータ│
  │     ├── Library/                  ← 設定、キャッシュ│
  │     └── tmp/                      ← 一時ファイル │
  │                                                    │
  │ セキュリティメカニズム:                           │
  │ 1. コード署名: すべてのアプリに Apple の署名必要 │
  │ 2. Entitlements: 機能ごとの権限宣言               │
  │ 3. Sandbox profiles: TrustedBSD MACベース        │
  │ 4. ASLR: アドレス空間のランダム化               │
  │ 5. PAC（Pointer Authentication）: ポインタの署名 │
  │ 6. MTE（Memory Tagging, A17+）: メモリタグ       │
  │                                                    │
  │ アプリ間通信:                                     │
  │ → URL Scheme, App Extensions, Shared Keychain    │
  │ → 明示的な許可が必要                             │
  └──────────────────────────────────────────────────┘

Android のサンドボックス:
  ┌──────────────────────────────────────────────────┐
  │ アプリの隔離:                                     │
  │ → 各アプリに固有の Linux UID を割り当て          │
  │ → SELinux のタイプエンフォースメント              │
  │ → seccomp-bpf でシステムコール制限              │
  │                                                    │
  │ セキュリティ層:                                   │
  │ 1. Linux UID 分離: アプリ間のプロセス隔離        │
  │ 2. SELinux: 強制アクセス制御                     │
  │ 3. seccomp: システムコール制限                   │
  │ 4. パーミッションモデル: API アクセスの制御      │
  │ 5. Verified Boot: ブート完全性の検証              │
  │ 6. dm-verity: システムパーティションの検証        │
  │                                                    │
  │ Android のパーミッション:                         │
  │ - Normal: 自動許可（インターネットアクセス等）    │
  │ - Dangerous: ユーザーの明示的許可が必要          │
  │   → カメラ、位置情報、連絡先、ストレージ等     │
  │ - Signature: 同じ署名のアプリのみ                │
  │ - Special: システム設定での手動許可              │
  │                                                    │
  │ Android 10+ のストレージサンドボックス:           │
  │ → Scoped Storage                                  │
  │ → アプリは自身のディレクトリのみ自由にアクセス   │
  │ → 他のファイルは MediaStore API 経由              │
  │ → Storage Access Framework で明示的選択          │
  └──────────────────────────────────────────────────┘
```

### 6.3 デスクトップOSのサンドボックス

```
macOS のサンドボックス:
  ┌──────────────────────────────────────────────────┐
  │ App Sandbox（App Store アプリは必須）:            │
  │ → TrustedBSD MAC フレームワークベース            │
  │ → Entitlements で権限を宣言                      │
  │                                                    │
  │ 主要な Entitlements:                              │
  │ - com.apple.security.app-sandbox: サンドボックス化│
  │ - com.apple.security.files.user-selected.read-only│
  │ - com.apple.security.network.client               │
  │ - com.apple.security.network.server               │
  │ - com.apple.security.device.camera                │
  │ - com.apple.security.device.microphone            │
  │                                                    │
  │ Gatekeeper:                                       │
  │ → 署名されていないアプリの実行を防止             │
  │ → Notarization（Apple による事前スキャン）       │
  │                                                    │
  │ SIP（System Integrity Protection）:               │
  │ → /System, /usr, /bin 等のシステムファイル保護   │
  │ → root でも変更不可                              │
  └──────────────────────────────────────────────────┘

Windows のサンドボックス:
  ┌──────────────────────────────────────────────────┐
  │ Windows Sandbox:                                   │
  │ → 使い捨ての軽量 VM（Windows 10 Pro以上）        │
  │ → ホストのカーネルを共有（軽量）                 │
  │ → 閉じるとすべてのデータが消去                   │
  │ → 疑わしいファイルの検証に有用                   │
  │                                                    │
  │ WDAC（Windows Defender Application Control）:     │
  │ → アプリケーションのホワイトリスト               │
  │ → コード署名ベースの制御                         │
  │                                                    │
  │ AppContainer:                                     │
  │ → UWP アプリのサンドボックス                     │
  │ → ファイルシステム、レジストリ、ネットワーク制限│
  │ → Capabilities で権限を宣言                      │
  │                                                    │
  │ Hyper-V ベースの保護:                             │
  │ → VBS（Virtualization-Based Security）           │
  │ → HVCI（Hypervisor-protected Code Integrity）    │
  │ → Credential Guard: 認証情報を隔離VM内で保護    │
  └──────────────────────────────────────────────────┘

Linux のサンドボックス:
  ┌──────────────────────────────────────────────────┐
  │ Flatpak:                                          │
  │ → デスクトップアプリのサンドボックス化            │
  │ → bubblewrap（setuid不要のサンドボックス）使用   │
  │ → Portals: ファイルピッカー等のAPI                │
  │ → 制限: --filesystem, --device, --socket          │
  │                                                    │
  │ Snap:                                             │
  │ → Ubuntu のアプリケーション隔離                  │
  │ → AppArmor プロファイルベース                    │
  │ → Snap Store からインストール                    │
  │                                                    │
  │ Firejail:                                         │
  │ → 既存アプリのサンドボックス化ツール             │
  │ → Namespace + seccomp + Capabilities              │
  │ → 設定例: firejail --seccomp firefox             │
  │                                                    │
  │ bubblewrap (bwrap):                               │
  │ → 低レベルのサンドボックスツール                 │
  │ → Flatpak, GNOME の基盤                          │
  │ → User Namespace ベース（setuid不要）            │
  └──────────────────────────────────────────────────┘
```

### 6.4 WebAssembly サンドボックス

```
WebAssembly（Wasm）のサンドボックス:
  ┌──────────────────────────────────────────────────┐
  │ ブラウザ内の Wasm:                                │
  │ → 線形メモリモデル（ホストメモリから隔離）       │
  │ → 型安全（バッファオーバーフロー耐性）          │
  │ → ファイルシステム、ネットワーク直接アクセス不可│
  │ → JavaScript API 経由でのみ外部と通信           │
  │                                                    │
  │ WASI（WebAssembly System Interface）:             │
  │ → Capability-based セキュリティ                  │
  │ → 明示的に渡されたファイルディスクリプタのみ     │
  │   アクセス可能                                    │
  │ → 「コンテナの次」の隔離技術として期待          │
  │                                                    │
  │ Wasm の利点:                                      │
  │ - 起動時間: マイクロ秒レベル                     │
  │ - サイズ: キロバイト〜メガバイト                  │
  │ - ポータビリティ: 任意のプラットフォームで実行   │
  │ - セキュリティ: デフォルトで最小権限             │
  │                                                    │
  │ Wasm ランタイム:                                  │
  │ - Wasmtime (Bytecode Alliance)                    │
  │ - Wasmer                                          │
  │ - WasmEdge (CNCF)                                 │
  │ - wazero (Go ネイティブ)                          │
  │                                                    │
  │ Solomon Hykes (Docker創設者):                     │
  │ "If WASM+WASI existed in 2008, we wouldn't       │
  │  have needed to create Docker."                   │
  └──────────────────────────────────────────────────┘
```

---

## 7. seccomp-bpf の詳細

### 7.1 seccomp-bpf フィルタの設計

```
seccomp-bpf:
  BPF（Berkeley Packet Filter）プログラムで
  プロセスのシステムコールをフィルタリング

  フィルタの動作:
  ┌──────────────────────────────────────────────────┐
  │ アプリケーション                                  │
  │       ↓ システムコール                           │
  │ seccomp-bpf フィルタ                              │
  │   → ALLOW: システムコールを許可                  │
  │   → KILL: プロセスを SIGKILL                     │
  │   → TRAP: SIGSYS シグナルを送信                  │
  │   → ERRNO: エラー番号を返す                      │
  │   → TRACE: ptrace に通知                         │
  │   → LOG: ログのみ（許可）                        │
  │   → USER_NOTIF: ユーザー空間に通知              │
  │       ↓                                           │
  │ カーネル                                          │
  └──────────────────────────────────────────────────┘
```

```c
/* seccomp-bpf フィルタの C 実装例 */
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stddef.h>

/* BPF フィルタ: write, exit_group, sigreturn のみ許可 */
static struct sock_filter filter[] = {
    /* アーキテクチャの確認 */
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
             offsetof(struct seccomp_data, arch)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
             AUDIT_ARCH_X86_64, 1, 0),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),

    /* システムコール番号を取得 */
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
             offsetof(struct seccomp_data, nr)),

    /* 許可するシステムコール */
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rt_sigreturn, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

    /* それ以外はすべて拒否 */
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
};

int main() {
    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof(filter[0]),
        .filter = filter,
    };

    /* NO_NEW_PRIVS を設定（SUID を無効化） */
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    /* seccomp フィルタを適用 */
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);

    /* 以降、write, exit_group, sigreturn のみ実行可能 */
    write(1, "Hello, sandboxed world!\n", 24);

    return 0;
}
```

### 7.2 libseccomp による簡易設定

```c
/* libseccomp を使った設定（より簡単） */
#include <seccomp.h>
#include <unistd.h>

int main() {
    /* デフォルトアクション: KILL */
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL_PROCESS);

    /* 許可するシステムコール */
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(rt_sigreturn), 0);

    /* 引数の条件付き許可 */
    /* write は fd=1 (stdout) と fd=2 (stderr) のみ */
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
                     SCMP_A0(SCMP_CMP_EQ, STDOUT_FILENO));
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
                     SCMP_A0(SCMP_CMP_EQ, STDERR_FILENO));

    /* フィルタをロード */
    seccomp_load(ctx);
    seccomp_release(ctx);

    write(1, "Sandboxed!\n", 11);
    return 0;
}
```

---

## 8. Confidential Computing（機密計算）

```
Confidential Computing:
  データを「使用中」も暗号化で保護する技術
  → 従来: 保存時の暗号化（at rest）+ 転送時の暗号化（in transit）
  → 追加: 使用中の暗号化（in use）

  TEE（Trusted Execution Environment）:
  ┌──────────────────────────────────────────────────┐
  │ Intel SGX（Software Guard Extensions）:           │
  │ → Enclave: 暗号化されたメモリ領域               │
  │ → CPU のみが復号可能                             │
  │ → OS/ハイパーバイザも読み取り不可               │
  │ → Remote Attestation で真正性を検証              │
  │ → 用途: 鍵管理、機械学習モデルの保護            │
  │                                                    │
  │ Intel TDX（Trust Domain Extensions）:             │
  │ → VM 全体を暗号化（VM単位のTEE）                │
  │ → SGX より大きなワークロードに対応              │
  │ → Azure の Confidential VM で採用               │
  │                                                    │
  │ AMD SEV-SNP（Secure Encrypted Virtualization）:   │
  │ → VM のメモリを AES で暗号化                    │
  │ → SNP: Secure Nested Paging（完全性保護追加）   │
  │ → AWS, Google Cloud で採用                      │
  │                                                    │
  │ ARM CCA（Confidential Compute Architecture）:     │
  │ → ARMv9 で導入                                   │
  │ → Realm: 隔離された実行環境                     │
  │ → モバイル/エッジデバイスの機密計算              │
  └──────────────────────────────────────────────────┘

  活用事例:
  - マルチパーティ計算: 複数組織のデータを保護しながら共同分析
  - 医療データ分析: 患者データを復号せずに機械学習
  - ブロックチェーン: スマートコントラクトの秘匿実行
  - 金融: 顧客データの安全な処理
```

---

## 実践演習

### 演習1: [基礎] -- Namespace の確認

```bash
# 現在のNamespace確認
ls -la /proc/self/ns/

# 新しいNamespaceでコマンド実行（要root）
sudo unshare --pid --fork --mount-proc bash
ps aux  # PID 1 から始まる別世界

# Namespace のID比較
readlink /proc/self/ns/pid
# → ホストとNamespace内で異なるID
```

### 演習2: [基礎] -- Network Namespace

```bash
# Network Namespace の作成と通信
sudo ip netns add ns1
sudo ip netns add ns2

# veth ペアで ns1 と ns2 を接続
sudo ip link add veth1 type veth peer name veth2
sudo ip link set veth1 netns ns1
sudo ip link set veth2 netns ns2

sudo ip netns exec ns1 ip addr add 10.0.0.1/24 dev veth1
sudo ip netns exec ns1 ip link set veth1 up
sudo ip netns exec ns1 ip link set lo up

sudo ip netns exec ns2 ip addr add 10.0.0.2/24 dev veth2
sudo ip netns exec ns2 ip link set veth2 up
sudo ip netns exec ns2 ip link set lo up

# 疎通確認
sudo ip netns exec ns1 ping -c 3 10.0.0.2

# クリーンアップ
sudo ip netns delete ns1
sudo ip netns delete ns2
```

### 演習3: [応用] -- cgroup でリソース制限

```bash
# cgroup v2 でメモリ制限（Linux）
sudo mkdir /sys/fs/cgroup/test

# コントローラの有効化
echo "+memory +pids" | sudo tee /sys/fs/cgroup/cgroup.subtree_control

# メモリ制限
echo 50M | sudo tee /sys/fs/cgroup/test/memory.max
echo 30M | sudo tee /sys/fs/cgroup/test/memory.high

# PID 数制限
echo 10 | sudo tee /sys/fs/cgroup/test/pids.max

# プロセスを追加
echo $$ | sudo tee /sys/fs/cgroup/test/cgroup.procs

# メモリ使用量の確認
cat /sys/fs/cgroup/test/memory.current
cat /sys/fs/cgroup/test/memory.stat

# ストレステスト
python3 -c "
data = []
try:
    while True:
        data.append('x' * 1024 * 1024)  # 1MB ずつ確保
except MemoryError:
    print(f'OOM at {len(data)} MB')
"

# クリーンアップ
echo $$ | sudo tee /sys/fs/cgroup/cgroup.procs
sudo rmdir /sys/fs/cgroup/test
```

### 演習4: [応用] -- 完全隔離環境の構築

```bash
# Namespace + cgroup で簡易コンテナを作成

# 1. ルートファイルシステムの準備
sudo debootstrap --variant=minbase focal /srv/container

# 2. 完全隔離で起動
sudo unshare --pid --fork --mount-proc \
  --net --uts --ipc --user --map-root-user \
  --mount \
  chroot /srv/container /bin/bash

# 3. 隔離環境内で確認
hostname isolated-container
ps aux           # PID 1 のみ
ip addr          # lo のみ
whoami           # root（実際は非特権ユーザー）
cat /etc/os-release  # Ubuntu Focal
```

### 演習5: [実務] -- Docker コンテナのセキュリティ強化

```bash
# 最小権限のDockerコンテナ
docker run --rm -it \
  --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --security-opt no-new-privileges:true \
  --security-opt seccomp=default.json \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid \
  --user 1000:1000 \
  --pids-limit 64 \
  --memory 256m \
  --memory-swap 256m \
  --cpus 0.5 \
  nginx:alpine

# gVisor を使用（より強い隔離）
docker run --runtime=runsc --rm -it alpine sh

# rootless Docker
dockerd-rootless-setuptool.sh install
docker context use rootless
docker run --rm hello-world
```

---

## まとめ

| 技術 | 隔離レベル | 用途 |
|------|----------|------|
| chroot | 弱 | 簡易的なFS隔離、ビルド環境 |
| FreeBSD Jail | 中 | サーバー隔離（FreeBSD） |
| Namespace | 中 | コンテナの基盤 |
| cgroup | リソース制限 | コンテナ、マルチテナント |
| seccomp-bpf | システムコール制限 | コンテナ、ブラウザ |
| コンテナ | 中 | マイクロサービス、CI/CD |
| gVisor | 中〜強 | サーバーレス、マルチテナント |
| Firecracker | 強 | サーバーレス（Lambda/Fargate） |
| Kata | 強 | セキュリティ重視のコンテナ |
| VM | 強 | マルチテナント、異なるOS |
| TEE | 最強 | 機密計算、金融、医療 |
| Wasm | 中〜強 | エッジ、プラグイン、ブラウザ |

---

## 次に読むべきガイド
→ [[../06-virtualization/00-vm-basics.md]] -- 仮想マシンの基礎

---

## 参考文献
1. Lieberman, H. "Container Security." O'Reilly, 2020.
2. Provos, N. "Preventing Privilege Escalation." USENIX Security, 2003.
3. Rice, L. "Container Security: Fundamental Technology Concepts that Protect Containerized Applications." O'Reilly, 2020.
4. Google. "gVisor: Container Runtime Sandbox." gVisor Documentation, 2024.
5. Amazon. "Firecracker: Secure and Fast microVMs for Serverless Computing." Firecracker Documentation, 2024.
6. Chromium. "Sandbox Design." Chromium Security Documentation, 2024.
7. Kerrisk, M. "Namespaces in Operation." LWN.net, 2013-2014.
8. Rosen, R. "Linux Kernel Networking: Implementation and Theory." Apress, 2014.
9. Confidential Computing Consortium. "A Technical Analysis of Confidential Computing." 2023.
10. Bytecode Alliance. "WebAssembly System Interface (WASI)." 2024.
