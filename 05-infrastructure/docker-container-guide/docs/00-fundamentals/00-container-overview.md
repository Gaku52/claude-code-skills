# コンテナ技術概要

> 仮想マシンとコンテナの違いを理解し、Docker とコンテナエコシステムの全体像を把握するための入門ガイド。Linux カーネル技術の基盤から OCI 標準、実務でのユースケース、代替ツールの比較まで体系的に解説する。

---

## この章で学ぶこと

1. **仮想化とコンテナ化の本質的な違い**を理解し、それぞれの適用領域を判断できる
2. **Linux カーネル技術（namespaces / cgroups / UnionFS）**の仕組みを深く理解する
3. **Docker の歴史と OCI 標準**を知り、コンテナエコシステムの全体像を把握する
4. **コンテナランタイムの階層構造**（高レベル・低レベル）を理解する
5. **コンテナのユースケース**を理解し、自分のプロジェクトへの適用を検討できる
6. **Docker 以外の選択肢**（Podman, nerdctl, Buildah 等）を比較検討できる

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

**Type 1（ベアメタル）ハイパーバイザ:**

ハードウェア上に直接動作するハイパーバイザで、パフォーマンスが高い。

```
+-------------------------------------------------+
|  ハードウェア (CPU, メモリ, ストレージ, NIC)       |
+-------------------------------------------------+
|  Type 1 ハイパーバイザ                            |
|  例: VMware ESXi, Microsoft Hyper-V,            |
|      Xen, KVM (Linux カーネル統合)               |
+-------------------------------------------------+
|  VM 1        |  VM 2        |  VM 3            |
|  Ubuntu 22   |  Windows 11  |  RHEL 9          |
+-------------------------------------------------+
```

**Type 2（ホスト型）ハイパーバイザ:**

ホスト OS 上のアプリケーションとして動作するハイパーバイザ。

```
+-------------------------------------------------+
|  ハードウェア                                     |
+-------------------------------------------------+
|  ホスト OS (macOS, Windows, Linux)              |
+-------------------------------------------------+
|  Type 2 ハイパーバイザ                            |
|  例: VirtualBox, VMware Workstation/Fusion,     |
|      Parallels Desktop, QEMU                    |
+-------------------------------------------------+
|  VM 1        |  VM 2        |  VM 3            |
+-------------------------------------------------+
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
  ゲスト OS なし - カーネルを共有
```

**コンテナ化の本質的なメリット:**

```
+----------------------------------------------------------+
|  イミュータビリティ (不変性)                                 |
|  ├─ イメージは一度ビルドしたら変更しない                      |
|  ├─ 設定変更 = 新しいイメージのビルド                        |
|  └─ デプロイ = 古いコンテナ破棄 + 新しいコンテナ起動           |
|                                                          |
|  ポータビリティ (可搬性)                                    |
|  ├─ 開発環境と本番環境で同一イメージを使用                    |
|  ├─ "手元では動くのに本番で動かない" 問題の解消                |
|  └─ クラウドベンダーに依存しない                             |
|                                                          |
|  効率性                                                   |
|  ├─ 起動時間: 数百ミリ秒 (VM は数分)                        |
|  ├─ メモリ: アプリ分のみ (VM は OS 分も必要)                 |
|  ├─ ディスク: レイヤー共有で重複排除                          |
|  └─ 密度: 1ホストに数百コンテナ (VM は数十)                  |
+----------------------------------------------------------+
```

### 1.3 ハイブリッド構成（VM + コンテナ）

実際のクラウド環境では、VM の上でコンテナを動かすハイブリッド構成が一般的である。

```
+-----------------------------------------------------+
|              クラウドプロバイダ (AWS / GCP / Azure)     |
+-----------------------------------------------------+
|              物理サーバー群                             |
+-----------------------------------------------------+
|              ハイパーバイザ (KVM 等)                    |
+-----------------------------------------------------+
|  VM (EC2)   |  VM (EC2)   |  VM (EC2)               |
|  +---------+|  +---------+|  +---------+            |
|  |コンテナ群||  |コンテナ群||  |コンテナ群|            |
|  | K8s Node||  | K8s Node||  | K8s Node|            |
|  +---------+|  +---------+|  +---------+            |
+-----------------------------------------------------+
  VM = テナント分離 / コンテナ = アプリ分離
```

```bash
# AWS EKS の場合の構成例
# EC2 インスタンス (VM) が Kubernetes ワーカーノードとして動作
# その上で Pod (コンテナ) が実行される

# ワーカーノード (VM) の確認
kubectl get nodes -o wide
# NAME                         STATUS   ROLES    AGE   VERSION
# ip-10-0-1-100.ec2.internal   Ready    <none>   5d    v1.28.3
# ip-10-0-2-200.ec2.internal   Ready    <none>   5d    v1.28.3

# Pod (コンテナ) の確認
kubectl get pods -o wide
# NAME                    READY   STATUS    NODE
# web-7d8f9c-abc12        1/1     Running   ip-10-0-1-100.ec2.internal
# api-5b6c7d-def34        1/1     Running   ip-10-0-2-200.ec2.internal
```

---

## 2. Linux カーネル技術の詳細

### 2.1 namespaces - リソースの可視性制御

コンテナの基盤となる Linux カーネル技術で、プロセスから見えるリソースの範囲を制限する。

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
|  |  - cgroup ns      |  |                      |  |
|  |  - time ns        |  |                      |  |
|  +-------------------+  +----------------------+  |
|                                                   |
|  namespaces = 見える範囲の制限（アイソレーション）    |
|  cgroups    = 使える量の制限（リソース制御）         |
+---------------------------------------------------+
```

**全 8 種類の namespaces 詳細:**

```bash
# ===================================
# 1. PID namespace - プロセスIDの分離
# ===================================
# コンテナ内ではPID 1から始まる独立したプロセスツリー
docker run --rm alpine ps aux
# PID   USER     COMMAND
#   1   root     ps aux

# ホスト側から見るとコンテナプロセスは別のPIDを持つ
docker run -d --name test-pid alpine sleep 3600
docker inspect --format '{{.State.Pid}}' test-pid
# 例: 45678 (ホスト側のPID)

# コンテナ内から見たPID
docker exec test-pid ps aux
# PID   USER     COMMAND
#   1   root     sleep 3600

# /proc ファイルシステムでnamespaceを確認
docker exec test-pid ls -la /proc/1/ns/pid
# lrwxrwxrwx 1 root root 0 /proc/1/ns/pid -> 'pid:[4026532456]'

docker rm -f test-pid

# ===================================
# 2. Network namespace - ネットワークスタックの分離
# ===================================
# 各コンテナは独自のネットワークインターフェース、IPアドレス、
# ルーティングテーブル、iptablesルールを持つ
docker run --rm alpine ip addr
# 1: lo: <LOOPBACK,UP,LOWER_UP>
#     inet 127.0.0.1/8 scope host lo
# 2: eth0@if123: <BROADCAST,MULTICAST,UP,LOWER_UP>
#     inet 172.17.0.2/16 brd 172.17.255.255

# ネットワーク名前空間の一覧（ホスト側）
sudo ip netns list

# コンテナのネットワーク設定を詳細確認
docker run --rm alpine sh -c "ip route && echo '---' && ip addr && echo '---' && cat /etc/resolv.conf"

# ===================================
# 3. Mount namespace - ファイルシステムの分離
# ===================================
# コンテナは独自のマウントポイントを持つ
docker run --rm alpine ls /
# bin    etc    lib    mnt    proc   run    srv    tmp    var
# dev    home   media  opt    root   sbin   sys    usr

# ホストのファイルシステムとは完全に分離されている
docker run --rm alpine cat /etc/os-release
# NAME="Alpine Linux"

# マウントポイントの確認
docker run --rm alpine mount
# overlay on / type overlay (...)
# proc on /proc type proc (...)
# tmpfs on /dev type tmpfs (...)

# ===================================
# 4. UTS namespace - ホスト名の分離
# ===================================
# 各コンテナは独自のホスト名を持つ
docker run --rm --hostname my-container alpine hostname
# my-container

docker run --rm alpine hostname
# ランダムな12文字のコンテナID

# ===================================
# 5. IPC namespace - プロセス間通信の分離
# ===================================
# 共有メモリ、セマフォ、メッセージキューの分離
docker run --rm alpine ipcs
# ------ Message Queues --------
# ------ Shared Memory Segments --------
# ------ Semaphore Arrays --------

# ===================================
# 6. User namespace - ユーザーIDの分離
# ===================================
# コンテナ内のroot(UID 0)をホストの非特権ユーザーにマッピング
docker run --rm alpine id
# uid=0(root) gid=0(root) groups=0(root)

# rootless モードでの実行
# コンテナ内のroot -> ホストの非特権ユーザー
docker run --rm --userns=host alpine cat /proc/self/uid_map
#          0       1000          1

# ===================================
# 7. Cgroup namespace (Linux 4.6+)
# ===================================
# cgroupファイルシステムの仮想化
docker run --rm alpine cat /proc/self/cgroup
# 0::/

# ===================================
# 8. Time namespace (Linux 5.6+)
# ===================================
# CLOCK_MONOTONIC と CLOCK_BOOTTIME の仮想化
# コンテナごとに異なるブート時間を設定可能
```

### 2.2 cgroups（Control Groups）- リソースの使用量制御

cgroups はプロセスグループのリソース使用量を制限・監視する機能である。

```bash
# ===================================
# cgroups v1 と v2 の確認
# ===================================
# cgroups バージョンの確認
stat -fc %T /sys/fs/cgroup/
# cgroup2fs -> cgroups v2
# tmpfs     -> cgroups v1

# Docker が使用している cgroup driver の確認
docker info | grep -i cgroup
# Cgroup Driver: systemd
# Cgroup Version: 2

# ===================================
# メモリ制限
# ===================================
# メモリを256MBに制限
docker run --memory=256m --rm alpine free -m

# メモリ + スワップの制限
docker run --memory=256m --memory-swap=512m --rm alpine free -m

# メモリ予約（ソフトリミット）
docker run --memory=512m --memory-reservation=256m --rm nginx

# OOM (Out Of Memory) の挙動制御
docker run --memory=64m --oom-kill-disable --rm stress-ng --vm 1 --vm-bytes 128m
# OOM Killer を無効化（危険: ホスト全体に影響する可能性）

# ===================================
# CPU 制限
# ===================================
# CPUを1コアに制限
docker run --cpus=1.0 --rm alpine cat /proc/cpuinfo

# CPU シェア（相対的な重み付け）
docker run --cpu-shares=1024 --rm nginx   # デフォルト
docker run --cpu-shares=512 --rm nginx    # 半分の重み

# 特定のCPUコアに固定（CPU ピニング）
docker run --cpuset-cpus="0,1" --rm nginx  # CPU 0 と 1 のみ使用
docker run --cpuset-cpus="0-3" --rm nginx  # CPU 0〜3 を使用

# CPU クォータ（周期ベースの制限）
docker run --cpu-period=100000 --cpu-quota=50000 --rm nginx
# 100ms あたり 50ms の CPU 時間 = 0.5 CPU

# ===================================
# I/O 制限
# ===================================
# ブロックデバイスの読み書き速度制限
docker run --device-read-bps=/dev/sda:1mb --rm alpine dd if=/dev/zero of=/tmp/test bs=1M count=10
docker run --device-write-bps=/dev/sda:1mb --rm alpine dd if=/dev/zero of=/tmp/test bs=1M count=10

# I/O ウェイト（相対的な重み付け）
docker run --blkio-weight=500 --rm nginx  # デフォルト: 500, 範囲: 10-1000

# ===================================
# プロセス数制限 (pids cgroup)
# ===================================
docker run --pids-limit=100 --rm alpine sh -c "ulimit -u"
# Fork Bomb 対策として重要

# ===================================
# リソース使用状況の確認
# ===================================
docker stats --no-stream
# CONTAINER ID   NAME     CPU %   MEM USAGE / LIMIT   MEM %   NET I/O       BLOCK I/O    PIDS
# abc123def456   web      0.50%   45.2MiB / 256MiB    17.66%  1.2kB / 0B    8.19kB / 0B  5

# 特定のコンテナの詳細なリソース情報
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.PIDs}}"

# cgroup ファイルシステムから直接情報を取得
docker run -d --name cgroup-test --memory=256m nginx
CONTAINER_ID=$(docker inspect --format '{{.Id}}' cgroup-test)
# cgroups v2 の場合
cat /sys/fs/cgroup/system.slice/docker-${CONTAINER_ID}.scope/memory.max
# 268435456 (256MB)
docker rm -f cgroup-test
```

### 2.3 UnionFS / OverlayFS - レイヤー化ファイルシステム

コンテナイメージのレイヤー構造を支える技術である。

```
+--------------------------------------------------+
|  コンテナレイヤー構造                               |
|                                                  |
|  +--------------------------------------------+ |
|  | コンテナ層 (Read-Write)                      | |
|  | 実行時の変更がここに書き込まれる                 | |
|  +--------------------------------------------+ |
|  | レイヤー 4: COPY . /app (アプリコード)         | |
|  +--------------------------------------------+ |
|  | レイヤー 3: RUN npm install (依存パッケージ)   | |
|  +--------------------------------------------+ |
|  | レイヤー 2: RUN apt-get install (OS パッケージ)| |
|  +--------------------------------------------+ |
|  | レイヤー 1: ベースイメージ (ubuntu:22.04)      | |
|  +--------------------------------------------+ |
|  ↑ すべてのレイヤーは Read-Only                   |
|  ↑ Copy-on-Write でファイル変更を上位層に書込み    |
+--------------------------------------------------+
```

```bash
# イメージのレイヤー情報を確認
docker history nginx:1.25-alpine
# IMAGE          CREATED       CREATED BY                                      SIZE
# 1234abcd5678   2 weeks ago   CMD ["nginx" "-g" "daemon off;"]                0B
# <missing>      2 weeks ago   EXPOSE map[80/tcp:{}]                           0B
# <missing>      2 weeks ago   STOPSIGNAL SIGQUIT                              0B
# <missing>      2 weeks ago   RUN /bin/sh -c set -x && addgroup ...           5.14MB
# <missing>      2 weeks ago   /bin/sh -c #(nop) ADD file:... in /            7.38MB

# レイヤーの詳細を JSON で確認
docker inspect nginx:1.25-alpine | python3 -m json.tool | head -50

# overlay2 ストレージドライバの情報
docker info --format '{{.Driver}}'
# overlay2

# イメージの保存先を確認
docker info --format '{{.DockerRootDir}}'
# /var/lib/docker

# 特定コンテナの OverlayFS マウント情報
docker run -d --name overlay-test nginx
docker inspect overlay-test --format '{{.GraphDriver.Data.MergedDir}}'
# /var/lib/docker/overlay2/<hash>/merged

docker inspect overlay-test --format '{{.GraphDriver.Data.UpperDir}}'
# /var/lib/docker/overlay2/<hash>/diff  (Read-Write 層)

docker inspect overlay-test --format '{{.GraphDriver.Data.LowerDir}}'
# /var/lib/docker/overlay2/<hash1>/diff:/var/lib/docker/overlay2/<hash2>/diff  (Read-Only 層)

docker rm -f overlay-test

# レイヤーの共有を確認（同じベースイメージを使うコンテナはレイヤーを共有）
docker pull nginx:1.25-alpine
docker pull nginx:1.25  # alpine と non-alpine で共通レイヤーがあれば共有される

# ディスク使用量の確認
docker system df
# TYPE            TOTAL     ACTIVE    SIZE      RECLAIMABLE
# Images          15        5         3.258GB   2.1GB (64%)
# Containers      5         3         120MB     45MB (37%)
# Local Volumes   8         3         500MB     200MB (40%)
# Build Cache     20        0         1.5GB     1.5GB

docker system df -v  # 詳細表示
```

### 2.4 seccomp - システムコールのフィルタリング

```bash
# Docker のデフォルト seccomp プロファイルを確認
# 約300以上のシステムコールのうち、約50が制限される
docker run --rm alpine cat /proc/self/status | grep Seccomp
# Seccomp:  2
# Seccomp_filters:  1

# seccomp プロファイルなしで実行（危険 - デバッグ目的のみ）
docker run --rm --security-opt seccomp=unconfined alpine cat /proc/self/status | grep Seccomp
# Seccomp:  0

# カスタム seccomp プロファイルの例
cat > /tmp/my-seccomp.json << 'SECCOMP_EOF'
{
  "defaultAction": "SCMP_ACT_ALLOW",
  "syscalls": [
    {
      "names": ["chmod", "fchmod", "fchmodat"],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 1
    },
    {
      "names": ["ptrace"],
      "action": "SCMP_ACT_ERRNO",
      "errnoRet": 1
    }
  ]
}
SECCOMP_EOF

docker run --rm --security-opt seccomp=/tmp/my-seccomp.json alpine chmod 777 /tmp
# chmod: /tmp: Operation not permitted
```

### 2.5 capabilities - 権限の細分化

```bash
# Linux capabilities の一覧（コンテナに関連するもの）
# Docker はデフォルトで限定的な capabilities のみ付与する

# デフォルトで付与される capabilities
docker run --rm alpine sh -c 'cat /proc/self/status | grep Cap'
# CapPrm:  00000000a80425fb
# CapEff:  00000000a80425fb

# capsh で人間が読める形式に変換
docker run --rm alpine sh -c 'apk add -q libcap && capsh --decode=00000000a80425fb'
# 0x00000000a80425fb=cap_chown,cap_dac_override,cap_fowner,...

# 全ての capabilities を削除
docker run --rm --cap-drop=ALL alpine id
# uid=0(root) gid=0(root)  # root だが実質的に権限なし

# 必要最小限の capabilities のみ追加
docker run --rm --cap-drop=ALL --cap-add=NET_BIND_SERVICE alpine sh -c 'id'

# 危険な capabilities の例
# --cap-add=SYS_ADMIN  # ほぼ root 相当（避けるべき）
# --cap-add=NET_ADMIN  # ネットワーク設定変更可能
# --cap-add=SYS_PTRACE # 他プロセスのデバッグ可能
```

---

## 3. 仮想マシン vs コンテナ 徹底比較

### 比較表 1: 技術的特性

| 特性 | 仮想マシン (VM) | コンテナ |
|---|---|---|
| アイソレーション | ハードウェアレベル | プロセスレベル |
| OS | 各VMにゲストOS | ホストOSカーネル共有 |
| 起動時間 | 数分 | 数秒〜数百ミリ秒 |
| サイズ | GB単位 (数GB〜数十GB) | MB単位 (数MB〜数百MB) |
| パフォーマンス | ハイパーバイザのオーバーヘッド (5-10%) | ほぼネイティブ (<1%) |
| 密度 | 1ホストに数十VM | 1ホストに数百〜数千コンテナ |
| セキュリティ | 強い分離（カーネルレベル） | カーネル共有のリスク |
| ポータビリティ | VMイメージが巨大 (数GB) | コンテナイメージが軽量 (数十MB) |
| ライブマイグレーション | サポートあり | 標準ではサポートなし |
| ネステッド実行 | VM in VM（パフォーマンス低下） | コンテナ in コンテナ（DinD/DooD） |
| スナップショット | VM単位のスナップショット | イメージレイヤーによるバージョン管理 |
| ネットワーク | 仮想NIC、独立スタック | veth ペア、ブリッジネットワーク |

### 比較表 2: 適用場面

| ユースケース | 推奨 | 理由 |
|---|---|---|
| マイクロサービス | コンテナ | 軽量・高速デプロイ・個別スケール |
| レガシーOS対応 | VM | 異なるカーネルが必要 |
| 開発環境の統一 | コンテナ | 再現性が高く、docker-compose で一発構築 |
| マルチテナント (SaaS) | VM | 強い分離が必要、カーネルの脆弱性リスク回避 |
| CI/CD パイプライン | コンテナ | 起動が高速、使い捨て可能 |
| デスクトップ仮想化 (VDI) | VM | GUI・ドライバ・周辺機器対応 |
| バッチ処理・ジョブ | コンテナ | スケールが容易、完了後に自動破棄 |
| セキュリティテスト | VM | 完全な分離、マルウェア解析に安全 |
| GPU ワークロード | 両方 | VM: GPU パススルー / コンテナ: NVIDIA Container Runtime |
| データベース | 両方 | 開発: コンテナ / 本番: VM or ベアメタル（I/O性能重視） |
| エッジコンピューティング | コンテナ | リソース制約のある環境に適合 |
| レガシーアプリ移行 | VM → コンテナ | 段階的移行、Lift & Shift → リファクタリング |

### 比較表 3: 運用コスト

| 観点 | 仮想マシン | コンテナ |
|---|---|---|
| 初期学習コスト | 低い（従来のサーバー運用に近い） | 中〜高（新しい概念の理解が必要） |
| 構築時間 | 数十分〜数時間 | 数秒〜数分 |
| ライセンス費用 | ゲストOS分のライセンスが必要 | OS共有のためライセンス不要 |
| ハードウェア効率 | 低い（OS分のオーバーヘッド） | 高い（アプリのみ） |
| パッチ適用 | 各VMのOSにパッチ | ベースイメージ更新 + リビルド |
| バックアップ | VMスナップショット（巨大） | イメージ + ボリュームバックアップ（軽量） |
| 障害復旧 | スナップショットからリストア | イメージから即座に再作成 |
| 監視・ログ | VM単位の従来型監視 | コンテナ対応の監視ツールが必要 |

### 実測値での比較

```bash
# ===================================
# 起動時間の比較実験
# ===================================

# コンテナの起動時間を計測
time docker run --rm alpine echo "Hello"
# real    0m0.432s  # 約0.4秒

time docker run --rm nginx sh -c "echo started"
# real    0m0.512s  # 約0.5秒

# 100コンテナの同時起動
time for i in $(seq 1 100); do
  docker run -d --rm --name "bench-${i}" alpine sleep 30
done
# real    0m12.345s  # 約12秒で100コンテナ起動

# クリーンアップ
docker stop $(docker ps -q --filter "name=bench-") 2>/dev/null

# ===================================
# イメージサイズの比較
# ===================================
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | sort -k3 -h
# REPOSITORY   TAG              SIZE
# alpine       3.19             7.38MB
# busybox      1.36             4.26MB
# debian       12-slim          74.8MB
# ubuntu       22.04            77.9MB
# node         20-alpine        127MB
# python       3.12-slim        130MB
# golang       1.22-alpine      256MB
# node         20               1.1GB
# ubuntu       22.04 (VM OVA)   約2.5GB (参考: VM イメージ)

# ===================================
# メモリ使用量の比較
# ===================================
# コンテナ: アプリ分のメモリのみ
docker run -d --name mem-test nginx
docker stats --no-stream mem-test
# CONTAINER   MEM USAGE / LIMIT     MEM %
# mem-test    3.5MiB / 16GiB        0.02%

# VM: OS + アプリのメモリが必要
# 最小構成の Ubuntu VM でも 512MB〜1GB のメモリが必要
```

---

## 4. Docker の歴史とエコシステム

### 4.1 年表

```
2000  FreeBSD Jail (コンテナ的分離の先駆け)
  |
2004  Solaris Zones / Containers
  |
2006  Google Process Containers → cgroups としてLinuxカーネルに統合
  |
2008  LXC (Linux Containers) リリース
  |   - namespaces + cgroups を組み合わせた初のコンテナ技術
  |
2013  Docker 0.1 リリース (dotCloud社)
  |   - LXC をラップした使いやすい CLI
  |   - Dockerfile によるイメージ定義
  |   - Docker Hub によるイメージ共有
  |
2014  Docker 1.0 GA
  |   - libcontainer で LXC 依存を脱却
  |   - Docker Machine, Docker Swarm 発表
  |   - Google が Kubernetes をオープンソース化
  |
2015  OCI (Open Container Initiative) 設立
  |   - Docker, CoreOS, Google, Microsoft 等が参加
  |   - コンテナの標準仕様を策定
  |   - Docker 1.8: Content Trust (イメージ署名)
  |
2016  Docker 1.12 - Swarm Mode 統合
  |   - CRI (Container Runtime Interface) 策定
  |
2017  containerd を CNCF に寄贈
  |   - Moby プロジェクト開始 (Docker のオープンソース部分)
  |   - Kubernetes が CRI 対応
  |   - LinuxKit 発表
  |
2018  Docker Enterprise Edition 強化
  |   - BuildKit がデフォルトビルダーに
  |
2019  Docker Desktop 有料化方針
  |   - Mirantis が Docker Enterprise を買収
  |
2020  Kubernetes が dockershim を非推奨化
  |   - containerd / CRI-O に移行
  |   - rootless Docker が安定版に
  |
2021  Docker Desktop ライセンス変更
  |   - 大企業 (従業員250名以上 or 年商$10M以上) は有料サブスクリプション
  |   - Docker Compose V2 (Go で書き直し)
  |
2022  Docker Desktop for Linux リリース
  |   - Docker Extensions マーケットプレイス
  |   - WebAssembly (Wasm) ランタイムサポート
  |
2023  Docker Scout (脆弱性スキャン)
  |   Docker Init (Dockerfile自動生成)
  |   Docker Debug (コンテナデバッグツール)
  |
2024  Docker Compose Watch (ファイル変更検知)
  |   Docker Build Cloud (リモートビルド)
  |   Docker Model Runner (AI モデル実行)
  |
2025  Docker AI Agent (開発支援 AI)
      Docker MCP Catalog & Toolkit
```

### 4.2 Docker のアーキテクチャ詳細

```
+-------------------------------------------------------------+
|                  Docker クライアント                           |
|  docker CLI / Docker Desktop / Docker Compose                |
+-------------------------------------------------------------+
        | REST API (unix:///var/run/docker.sock)
        v
+-------------------------------------------------------------+
|                  Docker デーモン (dockerd)                     |
|  ├─ イメージ管理                                              |
|  ├─ ネットワーク管理                                           |
|  ├─ ボリューム管理                                             |
|  └─ ビルド管理 (BuildKit)                                     |
+-------------------------------------------------------------+
        | gRPC
        v
+-------------------------------------------------------------+
|                  containerd                                   |
|  ├─ コンテナライフサイクル管理                                  |
|  ├─ イメージの pull/push                                      |
|  ├─ スナップショット管理                                       |
|  └─ タスク実行                                                |
+-------------------------------------------------------------+
        | OCI Runtime Spec
        v
+-------------------------------------------------------------+
|                  containerd-shim                              |
|  ├─ コンテナプロセスの親プロセス                                |
|  ├─ デーモン再起動時もコンテナを維持                             |
|  └─ exit status の管理                                        |
+-------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------+
|                  runc (OCI Runtime)                           |
|  ├─ namespaces の作成                                         |
|  ├─ cgroups の設定                                            |
|  ├─ seccomp プロファイル適用                                   |
|  └─ コンテナプロセスの起動                                     |
+-------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------+
|                  Linux カーネル                                |
|  namespaces / cgroups / OverlayFS / netfilter / seccomp      |
+-------------------------------------------------------------+
```

```bash
# Docker のバージョン確認（クライアントとサーバー）
docker version
# Client:
#  Version:           24.0.7
#  API version:       1.43
#  Go version:        go1.21.3
#  Built:             Thu Oct 26 09:07:41 2023
#  OS/Arch:           linux/amd64
#
# Server:
#  Engine:
#   Version:          24.0.7
#   API version:      1.43 (minimum version 1.12)
#   Go version:       go1.21.3
#   Built:            Thu Oct 26 09:07:41 2023
#   OS/Arch:          linux/amd64
#   containerd:       1.7.6
#   runc:             1.1.10
#   docker-init:      0.19.0

# Docker システム情報の確認
docker info
# Containers: 5
#  Running: 3
#  Paused: 0
#  Stopped: 2
# Images: 25
# Server Version: 24.0.7
# Storage Driver: overlay2
# Logging Driver: json-file
# Cgroup Driver: systemd
# Cgroup Version: 2
# Kernel Version: 6.5.0-14-generic
# Operating System: Ubuntu 22.04.3 LTS

# Docker デーモンのプロセス構成を確認
ps aux | grep -E "(dockerd|containerd|shim)"
# root  1234  dockerd --group docker
# root  1235  containerd
# root  5678  containerd-shim-runc-v2 -namespace moby -id abc123

# Docker ソケットの確認
ls -la /var/run/docker.sock
# srw-rw---- 1 root docker 0 /var/run/docker.sock

# Docker API に直接アクセス
curl --unix-socket /var/run/docker.sock http://localhost/v1.43/info 2>/dev/null | python3 -m json.tool | head -20
```

---

## 5. OCI 標準

OCI（Open Container Initiative）は、コンテナの業界標準を定める組織である。2015 年に Linux Foundation 傘下のプロジェクトとして設立され、Docker, Google, CoreOS, Microsoft, Red Hat, IBM 等が参加している。

### 5.1 OCI の 3 つの仕様

```
+--------------------------------------------------+
|           OCI (Open Container Initiative)         |
|                                                  |
|  +-------------------------------------------+  |
|  | Runtime Specification (runtime-spec)       |  |
|  | - コンテナの実行方法を定義                    |  |
|  | - config.json によるコンテナ設定              |  |
|  | - ライフサイクル: create → start → stop     |  |
|  | - 実装例: runc, crun, youki, gVisor,       |  |
|  |           Kata Containers                   |  |
|  +-------------------------------------------+  |
|                                                  |
|  +-------------------------------------------+  |
|  | Image Specification (image-spec)           |  |
|  | - コンテナイメージのフォーマットを定義          |  |
|  | - レイヤー構造（tar + gzip）                 |  |
|  | - マニフェスト、設定、レイヤーの3要素           |  |
|  | - マルチプラットフォーム対応                   |  |
|  +-------------------------------------------+  |
|                                                  |
|  +-------------------------------------------+  |
|  | Distribution Specification (dist-spec)     |  |
|  | - イメージの配布方法を定義                    |  |
|  | - レジストリ API (HTTP ベース)              |  |
|  | - pull / push / discover の標準化           |  |
|  +-------------------------------------------+  |
+--------------------------------------------------+
```

### 5.2 OCI Runtime Spec の詳細

```bash
# runc でOCIバンドルを手動で作成・実行する例
# (Docker の内部動作を理解するためのデモ)

# 1. ルートファイルシステムを準備
mkdir -p /tmp/oci-demo/rootfs
docker export $(docker create alpine) | tar -C /tmp/oci-demo/rootfs -xf -

# 2. OCI 設定ファイル (config.json) を生成
cd /tmp/oci-demo
runc spec
# config.json が生成される

# 3. config.json の内容（主要部分）
cat config.json
# {
#   "ociVersion": "1.0.2",
#   "process": {
#     "terminal": true,
#     "user": { "uid": 0, "gid": 0 },
#     "args": ["sh"],
#     "env": ["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"],
#     "cwd": "/"
#   },
#   "root": {
#     "path": "rootfs",
#     "readonly": true
#   },
#   "linux": {
#     "namespaces": [
#       { "type": "pid" },
#       { "type": "network" },
#       { "type": "ipc" },
#       { "type": "uts" },
#       { "type": "mount" }
#     ],
#     "resources": {
#       "memory": { "limit": 268435456 }
#     }
#   }
# }

# 4. runc でコンテナを実行
sudo runc run my-container
# -> alpine の sh が起動する
```

### 5.3 OCI Image Spec の詳細

```bash
# イメージのマニフェストを確認
docker manifest inspect nginx:1.25-alpine
# {
#   "schemaVersion": 2,
#   "mediaType": "application/vnd.oci.image.index.v1+json",
#   "manifests": [
#     {
#       "mediaType": "application/vnd.oci.image.manifest.v1+json",
#       "digest": "sha256:abc123...",
#       "size": 1234,
#       "platform": {
#         "architecture": "amd64",
#         "os": "linux"
#       }
#     },
#     {
#       "mediaType": "application/vnd.oci.image.manifest.v1+json",
#       "digest": "sha256:def456...",
#       "size": 1234,
#       "platform": {
#         "architecture": "arm64",
#         "os": "linux"
#       }
#     }
#   ]
# }

# skopeo でイメージの詳細情報を取得
skopeo inspect docker://nginx:1.25-alpine
# {
#   "Name": "docker.io/library/nginx",
#   "Tag": "1.25-alpine",
#   "Digest": "sha256:...",
#   "RepoTags": ["1.25-alpine", "1.25.3-alpine", ...],
#   "Created": "2024-01-15T...",
#   "DockerVersion": "24.0.7",
#   "Labels": {},
#   "Architecture": "amd64",
#   "Os": "linux",
#   "Layers": [
#     "sha256:abc123...",
#     "sha256:def456...",
#     "sha256:ghi789..."
#   ]
# }

# イメージを OCI フォーマットで保存
docker save nginx:1.25-alpine -o nginx-alpine.tar
mkdir -p /tmp/oci-image && tar -xf nginx-alpine.tar -C /tmp/oci-image
ls /tmp/oci-image/
# blobs/  index.json  manifest.json  oci-layout
```

### 5.4 OCI 準拠のツール群

```bash
# ===================================
# runc - OCI ランタイムリファレンス実装
# ===================================
runc --version
# runc version 1.1.10
# spec: 1.0.2-dev

# ===================================
# crun - C 言語実装の高速 OCI ランタイム
# ===================================
crun --version
# crun version 1.8.7
# runc より起動が高速 (約2倍)

# ===================================
# Podman - Docker互換のデーモンレスコンテナエンジン
# ===================================
podman run --rm alpine echo "Hello from Podman"

# Docker CLI との互換性
alias docker=podman  # これだけで多くのコマンドが動く

# Podman の特徴: デーモンレス + rootless がデフォルト
podman info | grep -A5 "host"
# rootless: true

# Pod (Kubernetes 互換のグルーピング)
podman pod create --name my-pod -p 8080:80
podman run -d --pod my-pod nginx
podman run -d --pod my-pod redis

# ===================================
# Buildah - OCI イメージビルドツール
# ===================================
# Dockerfile なしでイメージを構築
container=$(buildah from alpine)
buildah run $container apk add --no-cache nginx
buildah config --port 80 $container
buildah config --cmd "nginx -g 'daemon off;'" $container
buildah commit $container my-nginx:latest

# Dockerfile からビルド (docker build 互換)
buildah bud -t my-app:v1 .

# ===================================
# Skopeo - コンテナイメージ操作ツール
# ===================================
# レジストリ間でイメージをコピー（ローカルに pull 不要）
skopeo copy docker://nginx:1.25 docker://myregistry.example.com/nginx:1.25

# イメージの詳細情報を取得（pull 不要）
skopeo inspect docker://alpine:latest

# レジストリのタグ一覧
skopeo list-tags docker://nginx

# イメージの削除
skopeo delete docker://myregistry.example.com/old-image:v1

# ===================================
# nerdctl - containerd ネイティブ CLI
# ===================================
# Docker CLI 互換の containerd クライアント
nerdctl run --rm alpine echo "Hello from nerdctl"
nerdctl build -t my-app:v1 .
nerdctl compose up -d

# Docker にない機能
nerdctl image encrypt --recipient=jwe:public.pem my-app:v1  # イメージ暗号化
nerdctl run --cosign-key=cosign.pub verified-image:v1        # 署名検証
```

---

## 6. コンテナランタイムの階層構造

### 6.1 高レベルランタイムと低レベルランタイム

```
+----------------------------------------------------------+
|  コンテナランタイムの階層                                    |
|                                                          |
|  +----------------------------------------------------+ |
|  | コンテナエンジン (ユーザー向けインターフェース)           | |
|  | Docker Engine / Podman / nerdctl                    | |
|  +----------------------------------------------------+ |
|        |                                                 |
|        v                                                 |
|  +----------------------------------------------------+ |
|  | 高レベルランタイム (コンテナライフサイクル管理)          | |
|  | containerd / CRI-O                                  | |
|  | - イメージの管理 (pull / push / store)               | |
|  | - コンテナのライフサイクル管理                         | |
|  | - ネットワーク / ストレージの抽象化                    | |
|  +----------------------------------------------------+ |
|        |                                                 |
|        v                                                 |
|  +----------------------------------------------------+ |
|  | 低レベルランタイム (OCI Runtime)                      | |
|  | runc / crun / youki / gVisor (runsc) / Kata         | |
|  | - namespaces の作成                                  | |
|  | - cgroups の設定                                     | |
|  | - コンテナプロセスの起動                              | |
|  +----------------------------------------------------+ |
|        |                                                 |
|        v                                                 |
|  +----------------------------------------------------+ |
|  | Linux カーネル                                       | |
|  | namespaces / cgroups / OverlayFS / seccomp          | |
|  +----------------------------------------------------+ |
+----------------------------------------------------------+
```

### 6.2 Kubernetes と Docker の関係

```
+----------------------------------------------------------+
|  Kubernetes のコンテナランタイムの変遷                       |
|                                                          |
|  〜2020: dockershim (非推奨化)                             |
|  +-----------+    +-----------+    +------+    +------+  |
|  | kubelet   | -> | dockershim| -> |dockerd| -> |runc  |  |
|  +-----------+    +-----------+    +------+    +------+  |
|  ※ Docker 経由で containerd を呼ぶ冗長な構成               |
|                                                          |
|  2020〜: CRI 対応ランタイムに直接接続                       |
|                                                          |
|  パターン A: containerd                                   |
|  +-----------+    +-----------+    +------+              |
|  | kubelet   | -> |containerd | -> |runc  |              |
|  +-----------+    +-----------+    +------+              |
|  ※ Docker の一部だった containerd に直接接続               |
|                                                          |
|  パターン B: CRI-O                                        |
|  +-----------+    +-----------+    +------+              |
|  | kubelet   | -> |  CRI-O   | -> |runc  |              |
|  +-----------+    +-----------+    +------+              |
|  ※ Kubernetes 専用の軽量ランタイム                         |
+----------------------------------------------------------+
```

```bash
# Kubernetes のコンテナランタイムを確認
kubectl get nodes -o wide
# NAME       STATUS   ROLES    VERSION   CONTAINER-RUNTIME
# node-1     Ready    <none>   v1.28.3   containerd://1.7.6
# node-2     Ready    <none>   v1.28.3   containerd://1.7.6

# containerd の設定確認
cat /etc/containerd/config.toml
# [plugins."io.containerd.grpc.v1.cri"]
#   [plugins."io.containerd.grpc.v1.cri".containerd]
#     default_runtime_name = "runc"
#     [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc]
#       runtime_type = "io.containerd.runc.v2"

# crictl (CRI デバッグツール) でコンテナ情報を取得
crictl ps
# CONTAINER   IMAGE    CREATED    STATE    NAME          POD ID
# abc123...   nginx    10m ago    Running  web-server    def456...

crictl images
# IMAGE                     TAG       SIZE
# docker.io/library/nginx   1.25      41.2MB
```

### 6.3 サンドボックスランタイム

従来のコンテナよりも強いセキュリティ分離を提供するランタイム。

```
+----------------------------------------------------------+
|  サンドボックスランタイムの比較                               |
|                                                          |
|  +-------------------+  +----------------------------+   |
|  | gVisor (runsc)    |  | Kata Containers            |   |
|  |                   |  |                            |   |
|  | ユーザースペースで  |  | 軽量VMの中で                |   |
|  | カーネルを再実装    |  | コンテナを実行              |   |
|  |                   |  |                            |   |
|  | +-------------+   |  | +------------------------+ |   |
|  | | アプリ       |   |  | | 軽量 VM               | |   |
|  | +-------------+   |  | | +------------------+   | |   |
|  | | gVisor      |   |  | | | アプリ            |   | |   |
|  | | Sentry      |   |  | | +------------------+   | |   |
|  | | (カーネル    |   |  | | | ゲストカーネル     |   | |   |
|  | |  エミュ)    |   |  | | +------------------+   | |   |
|  | +-------------+   |  | +------------------------+ |   |
|  | | ホストカーネル|   |  | | QEMU / Firecracker   | |   |
|  | +-------------+   |  | +------------------------+ |   |
|  +-------------------+  +----------------------------+   |
|                                                          |
|  gVisor: システムコールを          Kata: VMレベルの分離を    |
|  フィルタ・再実装して安全性向上      コンテナの使い勝手で提供   |
+----------------------------------------------------------+
```

```bash
# gVisor のインストールと使用
# Docker に gVisor ランタイムを登録
cat > /etc/docker/daemon.json << 'EOF'
{
  "runtimes": {
    "runsc": {
      "path": "/usr/local/bin/runsc"
    }
  }
}
EOF
sudo systemctl restart docker

# gVisor でコンテナを実行
docker run --runtime=runsc --rm alpine uname -a
# Linux ... 4.4.0 ... gVisor

# Kata Containers でコンテナを実行
docker run --runtime=kata --rm alpine uname -a
# Linux ... 5.15.0 ... (軽量VMのカーネル)

# ランタイムごとのパフォーマンス比較
# runc:    起動 ~300ms, メモリ +0MB (ベースライン)
# gVisor:  起動 ~500ms, メモリ +50MB (Sentry プロセス分)
# Kata:    起動 ~1.5s,  メモリ +128MB (軽量VM分)
```

---

## 7. コンテナのユースケース

### 7.1 マイクロサービスアーキテクチャ

```bash
# 各サービスを独立したコンテナとして実行
docker network create microservices

# API Gateway
docker run -d --name api-gateway \
  --network microservices \
  -p 8080:8080 \
  -e UPSTREAM_SERVICES="user-service:8081,order-service:8082,payment-service:8083" \
  api-gateway:v1

# ユーザーサービス
docker run -d --name user-service \
  --network microservices \
  -e DB_HOST=user-db \
  -e DB_PORT=5432 \
  user-service:v1

# 注文サービス
docker run -d --name order-service \
  --network microservices \
  -e DB_HOST=order-db \
  -e KAFKA_BROKERS=kafka:9092 \
  order-service:v1

# 決済サービス
docker run -d --name payment-service \
  --network microservices \
  -e STRIPE_API_KEY_FILE=/run/secrets/stripe_key \
  payment-service:v1

# データベース群
docker run -d --name user-db \
  --network microservices \
  -v user-db-data:/var/lib/postgresql/data \
  postgres:16-alpine

docker run -d --name order-db \
  --network microservices \
  -v order-db-data:/var/lib/postgresql/data \
  postgres:16-alpine

# メッセージキュー
docker run -d --name kafka \
  --network microservices \
  -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092 \
  confluentinc/cp-kafka:7.5
```

```yaml
# docker-compose.yml による マイクロサービス構成
services:
  api-gateway:
    image: api-gateway:v1
    ports:
      - "8080:8080"
    depends_on:
      - user-service
      - order-service
      - payment-service
    environment:
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
    networks:
      - frontend
      - backend

  user-service:
    image: user-service:v1
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
    depends_on:
      user-db:
        condition: service_healthy
    environment:
      - DB_HOST=user-db
    networks:
      - backend

  user-db:
    image: postgres:16-alpine
    volumes:
      - user-db-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - backend

  jaeger:
    image: jaegertracing/all-in-one:1.52
    ports:
      - "16686:16686"  # UI
      - "4317:4317"    # OTLP gRPC
    networks:
      - backend

networks:
  frontend:
  backend:

volumes:
  user-db-data:
```

### 7.2 開発環境の統一

```yaml
# docker-compose.yml による開発環境定義
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/node_modules  # node_modules はコンテナ内のものを使用
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://devuser:devpass@db:5432/devdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: devuser
      POSTGRES_PASSWORD: devpass
      POSTGRES_DB: devdb
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U devuser -d devdb"]
      interval: 5s
      timeout: 3s
      retries: 10

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru

  mailhog:
    image: mailhog/mailhog:v1.0.1
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"  # Console
    volumes:
      - minio-data:/data

volumes:
  postgres-data:
  minio-data:
```

### 7.3 CI/CD パイプライン

```yaml
# GitHub Actions でのコンテナ活用例
name: CI/CD Pipeline
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U test"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Build test image
        run: docker build --target test -t myapp:test .

      - name: Run unit tests
        run: |
          docker run --rm \
            --network host \
            -e DATABASE_URL=postgresql://test:test@localhost:5432/testdb \
            -e REDIS_URL=redis://localhost:6379 \
            myapp:test npm test

      - name: Run integration tests
        run: |
          docker run --rm \
            --network host \
            -e DATABASE_URL=postgresql://test:test@localhost:5432/testdb \
            myapp:test npm run test:integration

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Kubernetes にデプロイ
          kubectl set image deployment/myapp \
            myapp=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          kubectl rollout status deployment/myapp --timeout=300s
```

### 7.4 データサイエンス・機械学習

```yaml
# ML 開発環境の docker-compose.yml
services:
  jupyter:
    image: jupyter/scipy-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - GRANT_SUDO=yes
    user: root

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.9.0
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@mlflow-db:5432/mlflow
      --default-artifact-root s3://mlflow-artifacts/
      --host 0.0.0.0
    depends_on:
      - mlflow-db

  mlflow-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - mlflow-db-data:/var/lib/postgresql/data

  gpu-training:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0,1

volumes:
  mlflow-db-data:
```

```dockerfile
# Dockerfile.gpu - GPU 対応の ML 環境
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    transformers datasets accelerate \
    mlflow scikit-learn pandas numpy

WORKDIR /app
COPY . .

CMD ["python3", "train.py"]
```

### 7.5 ローカルツール・サービスの実行

```bash
# 一時的なデータベースの起動（開発・テスト用）
docker run --rm -d \
  --name temp-postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=mypassword \
  postgres:16-alpine

# 一時的な Redis の起動
docker run --rm -d \
  --name temp-redis \
  -p 6379:6379 \
  redis:7-alpine

# 静的サイトのプレビュー
docker run --rm -p 8080:80 \
  -v $(pwd)/dist:/usr/share/nginx/html:ro \
  nginx:alpine

# データベースのマイグレーション実行
docker run --rm \
  --network host \
  -v $(pwd)/migrations:/migrations \
  migrate/migrate \
  -path=/migrations -database "postgresql://user:pass@localhost:5432/mydb?sslmode=disable" up

# セキュリティスキャン
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image nginx:1.25

# コードフォーマット（言語に依存しない実行環境）
docker run --rm -v $(pwd):/work -w /work \
  golangci/golangci-lint:latest golangci-lint run

# PlantUML による図の生成
docker run --rm -v $(pwd):/data \
  plantuml/plantuml:latest /data/diagram.puml
```

---

## 8. アンチパターンとベストプラクティス

### アンチパターン 1: コンテナを VM のように使う

```bash
# NG: 1コンテナに複数サービスを詰め込む
# SSH, cron, アプリ, DB を全部1コンテナに入れる
docker run -d my-monolith-container
# -> メンテナンス困難、スケール不可、ログ管理が複雑

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
# -> CI で通ったバージョンと本番で異なるバージョンが動く危険

# OK: 具体的なバージョンを指定
docker run -d nginx:1.25.3-alpine
# -> いつ実行しても同じイメージが使われる

# さらに厳密: ダイジェストで固定
docker run -d nginx@sha256:abc123def456...
# -> タグが上書きされても影響を受けない
```

### アンチパターン 3: ホストネットワークの安易な使用

```bash
# NG: ホストネットワークを常用
docker run --network host my-app
# -> ポート衝突、セキュリティリスク、ポータビリティ低下

# OK: ブリッジネットワークでポートマッピング
docker run -p 8080:80 my-app
# -> 明示的なポート制御、分離されたネットワーク
```

### アンチパターン 4: コンテナ内にデータを永続化

```bash
# NG: コンテナ内にデータを保存
docker run -d my-app
# コンテナ再作成でデータが消失する

# OK: ボリュームを使用
docker run -d -v app-data:/data my-app
# コンテナを破棄してもデータは永続化される

# OK: バインドマウントを使用（開発環境向け）
docker run -d -v $(pwd)/data:/data my-app
```

### アンチパターン 5: root ユーザーでの実行

```bash
# NG: root で実行（デフォルト）
docker run -d my-app
# コンテナエスケープ時にホストの root 権限を取得されるリスク

# OK: 非特権ユーザーで実行
# Dockerfile で指定:
# RUN addgroup -S app && adduser -S app -G app
# USER app

# または実行時に指定:
docker run -d --user 1000:1000 my-app

# rootless Docker を使用:
dockerd-rootless.sh
```

### アンチパターン 6: 機密情報の埋め込み

```bash
# NG: 環境変数でシークレットを渡す
docker run -e API_KEY=sk-12345secret my-app
# -> docker inspect で丸見え、ログに漏洩する可能性

# NG: Dockerfile にハードコード
# ENV API_KEY=sk-12345secret
# -> イメージに永続的に保存される

# OK: Docker Secrets を使用（Swarm mode）
echo "sk-12345secret" | docker secret create api_key -
docker service create --secret api_key my-app
# コンテナ内で /run/secrets/api_key として読める

# OK: 外部シークレット管理ツールとの連携
docker run -d \
  -e VAULT_ADDR=https://vault.example.com \
  -e VAULT_TOKEN_FILE=/run/secrets/vault_token \
  my-app
```

---

## 9. コンテナ技術の将来展望

### 9.1 WebAssembly (Wasm) コンテナ

```bash
# Docker + Wasm の実行例
# containerd の Wasm shim を使用
docker run --runtime=io.containerd.wasmedge.v1 \
  --platform wasi/wasm \
  ghcr.io/example/wasm-app:latest

# Wasm コンテナの特徴:
# - 起動時間: ~1ms (通常のコンテナ: ~300ms)
# - サイズ: ~KB単位 (通常のコンテナ: ~MB単位)
# - セキュリティ: サンドボックスモデルが標準
# - ポータビリティ: CPU アーキテクチャに依存しない
```

### 9.2 Confidential Containers

```
+----------------------------------------------------------+
|  Confidential Containers (CoCo)                          |
|                                                          |
|  ハードウェアベースの機密コンピューティング                   |
|  TEE (Trusted Execution Environment) 内でコンテナを実行    |
|                                                          |
|  +----------------------------------------------------+ |
|  | TEE (Intel SGX / AMD SEV / ARM CCA)                | |
|  | +------------------------------------------------+ | |
|  | | 暗号化されたメモリ空間                              | | |
|  | | +--------------------------------------------+  | | |
|  | | | コンテナ (暗号化イメージから起動)               |  | | |
|  | | | データは TEE 内でのみ復号                     |  | | |
|  | | +--------------------------------------------+  | | |
|  | +------------------------------------------------+ | |
|  +----------------------------------------------------+ |
|  ホスト OS やクラウドプロバイダもデータにアクセスできない     |
+----------------------------------------------------------+
```

### 9.3 eBPF によるコンテナ監視

```bash
# eBPF ベースのコンテナセキュリティツール
# Falco - ランタイムセキュリティ検知
docker run --rm -i -t \
  --privileged \
  -v /var/run/docker.sock:/host/var/run/docker.sock \
  -v /proc:/host/proc:ro \
  falcosecurity/falco:latest

# Cilium - eBPF ベースのコンテナネットワーク
# NetworkPolicy を eBPF で実装（iptables より高速）
# L7 レベルのトラフィック可視化
```

---

## 10. FAQ

### Q1: コンテナは仮想マシンの上位互換ですか？

**A:** いいえ。コンテナと VM は相互補完的な技術である。コンテナはホスト OS のカーネルを共有するため、異なる OS カーネルを必要とする場面（例: Linux ホスト上で Windows アプリを動かす）には VM が必要である。また、強いセキュリティ分離が求められるマルチテナント環境では VM が適している。多くのクラウド環境では、VM の中でコンテナを動かすハイブリッド構成が採用されている。

### Q2: Docker と Podman はどう違いますか？

**A:** 最大の違いはアーキテクチャである。Docker はデーモン（dockerd）が常駐するクライアント-サーバー型だが、Podman はデーモンレスで各コンテナが独立したプロセスとして動く。Podman は OCI 準拠で Docker CLI と高い互換性がある（`alias docker=podman` で多くのコマンドが動く）。rootless 実行がデフォルトでサポートされている点もセキュリティ上の利点である。Podman には Pod という Kubernetes 互換のグルーピング機能もある。ただし、Docker Compose のような統合ツールチェーンは Docker の方が成熟しており、企業でのサポート体制も Docker の方が整っている。

### Q3: Windows や macOS でコンテナはネイティブに動きますか？

**A:** Linux コンテナは Linux カーネルの機能（namespaces, cgroups）に依存するため、macOS や Windows ではネイティブに動作しない。Docker Desktop は内部で軽量な Linux VM を起動し、その中でコンテナを実行している。macOS では Apple の Virtualization.framework（Intel Mac では HyperKit）、Windows では WSL2（Windows Subsystem for Linux 2）が使われる。Windows コンテナ（Windows Server 上）は Windows カーネルでネイティブに動作するが、Linux コンテナとの互換性はない。

### Q4: コンテナのセキュリティは VM より弱いのですか？

**A:** カーネル共有によるリスクは存在するが、適切な対策により実用上十分なセキュリティを確保できる。具体的には、rootless コンテナの使用、seccomp プロファイル、AppArmor/SELinux、read-only ファイルシステム、最小権限の原則（capabilities の制限）を適用する。さらに、gVisor や Kata Containers のようなサンドボックスランタイムを使えば、VM に近いレベルの分離を実現できる。マルチテナント環境では VM + コンテナのハイブリッド構成が推奨される。

### Q5: Docker Desktop のライセンスはどうなっていますか？

**A:** 2021 年 8 月の変更により、従業員 250 名以上または年間売上 $10M 以上の企業は有料サブスクリプション（Pro / Team / Business）が必要になった。個人利用、小規模企業、教育機関、オープンソースプロジェクトは引き続き無料で利用可能。代替手段として、Linux では Docker Engine（無料）を直接使用でき、macOS では Colima や Lima + nerdctl、Windows では WSL2 + Docker Engine を使う方法がある。

### Q6: containerd と CRI-O の違いは何ですか？

**A:** containerd は Docker から分離された汎用的なコンテナランタイムで、Docker Engine や nerdctl のバックエンドとしても使われる。CRI-O は Kubernetes 専用に設計された軽量ランタイムで、CRI（Container Runtime Interface）に特化している。containerd の方が汎用性が高く利用実績も多い。CRI-O は Kubernetes 以外では使われないが、そのぶん軽量でアタックサーフェスが小さい。どちらも OCI 準拠で runc を低レベルランタイムとして使用する。

### Q7: コンテナ内で systemd は使えますか？

**A:** 技術的には可能だが、推奨されない。コンテナは「1コンテナ1プロセス」の原則で設計されており、init システムを使うのはアンチパターンとされる。ただし、systemd に依存するレガシーアプリケーションの移行期には必要になることがある。その場合は `--privileged` フラグや `/sys/fs/cgroup` のマウントが必要になる。代替として `tini` や `dumb-init` などの軽量 init プロセスを PID 1 として使うことが推奨される（Docker は `--init` フラグで `tini` を自動的に使用できる）。

### Q8: Docker の代替ツールとして何がありますか？

**A:** 主な代替ツールは以下の通り。

| ツール | 用途 | 特徴 |
|---|---|---|
| Podman | コンテナ実行 | デーモンレス、rootless、Docker CLI 互換 |
| nerdctl | コンテナ実行 | containerd ネイティブ、Docker CLI 互換 |
| Buildah | イメージビルド | Dockerfile 不要でもビルド可能 |
| Skopeo | イメージ操作 | pull 不要でレジストリ間コピー |
| Kaniko | CI/CD ビルド | Docker デーモン不要でビルド |
| Lima | macOS 上の Linux VM | Docker Desktop の代替 |
| Colima | macOS 上の Docker | Lima ベースの簡易 Docker 環境 |
| Finch | コンテナ実行 | AWS 提供、Lima + nerdctl ベース |

---

## 11. まとめ

| 項目 | ポイント |
|---|---|
| コンテナとは | ホストOSカーネルを共有するプロセスレベルの仮想化 |
| 基盤技術 | Linux namespaces（分離）+ cgroups（リソース制限）+ UnionFS（レイヤーFS） |
| VM との違い | ゲストOS不要で高速・軽量、ただし分離レベルは異なる |
| Docker の位置づけ | コンテナエコシステムの事実上の標準ツールチェーン |
| アーキテクチャ | Docker CLI → dockerd → containerd → runc → カーネル |
| OCI 標準 | runtime-spec, image-spec, distribution-spec の3仕様 |
| ランタイム階層 | 高レベル (containerd/CRI-O) + 低レベル (runc/crun) |
| サンドボックス | gVisor（カーネル再実装）/ Kata（軽量VM） |
| 主なユースケース | マイクロサービス、開発環境統一、CI/CD、ML/AI |
| 設計原則 | 1コンテナ1プロセス、イミュータブル、バージョン固定 |
| セキュリティ | rootless、seccomp、capabilities制限、最小ベースイメージ |
| 将来展望 | WebAssembly コンテナ、Confidential Containers、eBPF |

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
4. **Linux man pages - cgroups(7)** https://man7.org/linux/man-pages/man7/cgroups.7.html -- cgroups の公式ドキュメント。リソース制限の仕組みを詳細に解説。
5. **Kubernetes Documentation - Container Runtimes** https://kubernetes.io/docs/setup/production-environment/container-runtimes/ -- containerd, CRI-O 等のコンテナランタイムの比較と設定方法。
6. **containerd Documentation** https://containerd.io/docs/ -- containerd の公式ドキュメント。Docker の内部で使われるコンテナランタイムの詳細。
7. **Podman Documentation** https://podman.io/docs -- Podman の公式ドキュメント。Docker 代替ツールの使い方。
8. **gVisor Documentation** https://gvisor.dev/docs/ -- gVisor の公式ドキュメント。サンドボックスランタイムによるセキュリティ強化。
9. **NIST SP 800-190 Application Container Security Guide** -- コンテナセキュリティのベストプラクティスガイド。
