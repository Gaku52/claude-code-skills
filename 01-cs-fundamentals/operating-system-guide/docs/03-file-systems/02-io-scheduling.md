# I/Oスケジューリング

> I/Oスケジューラはディスクへのリクエストを並べ替え、ヘッドの移動距離を最小化してスループットを向上させる。

## この章で学ぶこと

- [ ] I/Oスケジューリングの必要性を理解する
- [ ] 主要なスケジューラの違いを知る
- [ ] SSD時代のI/O最適化を理解する
- [ ] Linuxブロックレイヤーの構造を把握する
- [ ] io_uring を含む非同期I/O技術を理解する
- [ ] 実務でのI/Oチューニング手法を習得する

---

## 1. なぜI/Oスケジューリングが必要か

### 1.1 HDDのアクセス特性

```
HDD のアクセス時間の構成:

  全体のアクセス時間 = シーク時間 + 回転待ち + 転送時間

  シーク時間 (Seek Time):
    ヘッドの移動にかかる時間
    → 3〜10ms（平均）、最悪 15ms以上
    → 隣接トラック: 0.5ms〜1ms
    → 全ストローク: 10ms〜20ms
    ← I/O性能の最大のボトルネック

  回転待ち (Rotational Latency):
    目的のセクタがヘッド下に来るまでの待ち時間
    → 回転速度に依存
    → 7200 RPM: 平均 4.17ms（1回転 = 8.33ms の半分）
    → 15000 RPM: 平均 2.0ms
    → 5400 RPM: 平均 5.56ms

  転送時間 (Transfer Time):
    データの実際の読み書き
    → 通常 < 1ms（数十〜数百MB/s）
    → 全体に占める割合は小さい

  具体的な計算例:
  7200RPM HDD で 4KB のランダム読み取り:
    シーク: 8ms（平均）
    回転待ち: 4.17ms（平均）
    転送: 0.01ms（4KB / 200MB/s）
    合計: ≈ 12.18ms

  → 1秒間に約82回のランダムI/O = 82 IOPS
  → シーケンシャル読み取りなら 200MB/s 以上

  比較: SSD のランダム読み取り:
    レイテンシ: 0.05〜0.1ms
    → 10,000〜100,000+ IOPS
    → HDDの100〜1000倍以上
```

### 1.2 スケジューリングの効果

```
リクエストの順序最適化でシーク時間を大幅に削減:

  ディスク上のトラック位置:
  0     50    100   150   200   250   300
  |─────|─────|─────|─────|─────|─────|

  リクエストキュー（到着順）:
  位置: 98, 183, 37, 122, 14, 124, 65, 67

  現在のヘッド位置: 53

  ■ FCFS（先着順処理、何もしない場合）:
  53 → 98 → 183 → 37 → 122 → 14 → 124 → 65 → 67
  移動: 45 + 85 + 146 + 85 + 108 + 110 + 59 + 2 = 640

  53──→98──────→183
            ←──37──→122
       ←──14──→124
           ←──65→67

  ■ SSTF（最短シーク時間優先）:
  53 → 65 → 67 → 37 → 14 → 98 → 122 → 124 → 183
  移動: 12 + 2 + 30 + 23 + 84 + 24 + 2 + 59 = 236
  → FCFS比 63% 削減！

  問題: 飢餓（starvation）
  → ヘッドの現在位置から遠いリクエストは永遠に処理されない可能性

  ■ SCAN（エレベータアルゴリズム）:
  ヘッドを一方向に移動しながら処理、端に達したら反転
  53 → 37 → 14 → [0] → 65 → 67 → 98 → 122 → 124 → 183
  移動: 16 + 23 + 14 + 65 + 2 + 31 + 24 + 2 + 59 = 236
  → 飢餓を防止

  ■ C-SCAN（サーキュラーSCAN）:
  一方向のみ処理、端に達したら反対の端にジャンプ
  → より均等な待ち時間

  ■ LOOK / C-LOOK:
  SCAN/C-SCANの改良版
  → ディスクの端まで行かず、最後のリクエスト位置で反転
  → 無駄な移動を削減

  各アルゴリズムの比較:
  ┌──────────┬──────────┬──────────┬──────────┐
  │ アルゴリズム│ 移動距離 │ 飢餓     │ 待ち時間 │
  ├──────────┼──────────┼──────────┼──────────┤
  │ FCFS     │ 最大     │ なし     │ 不均等   │
  │ SSTF     │ 小       │ あり     │ 不均等   │
  │ SCAN     │ 中       │ なし     │ やや均等 │
  │ C-SCAN   │ 中       │ なし     │ 均等     │
  │ LOOK     │ 小       │ なし     │ やや均等 │
  │ C-LOOK   │ 小       │ なし     │ 均等     │
  └──────────┴──────────┴──────────┴──────────┘
```

### 1.3 I/Oスケジューリングの位置づけ

```
Linux I/Oスタックにおけるスケジューラの位置:

  アプリケーション
     │ read() / write()
     ↓
  VFS (Virtual File System)
     │
     ↓
  ファイルシステム (ext4, XFS, Btrfs)
     │ ブロックI/Oリクエスト生成
     ↓
  ページキャッシュ
     │ キャッシュヒット → ここで完了
     │ キャッシュミス ↓
     ↓
  ┌─────────────────────────────────────┐
  │ ブロックレイヤー                     │
  │ ┌─────────────────────────────────┐ │
  │ │ I/Oスケジューラ                  │ │
  │ │ → リクエストの並べ替え・マージ    │ │
  │ └─────────────────────────────────┘ │
  │ ┌─────────────────────────────────┐ │
  │ │ マルチキューブロックレイヤー      │ │
  │ │ (blk-mq)                        │ │
  │ └─────────────────────────────────┘ │
  └─────────────────────────────────────┘
     │
     ↓
  デバイスドライバ
     │
     ↓
  ハードウェア（HDD / SSD / NVMe）
```

---

## 2. Linuxの I/Oスケジューラ

### 2.1 レガシーシングルキュースケジューラ（カーネル4.x以前）

```
旧世代のI/Oスケジューラ（参考）:

  1. noop:
     スケジューリングなし（FIFO）
     → マージのみ実行、並べ替えなし
     → SSD、仮想環境向け

  2. deadline:
     各リクエストにデッドライン（期限）を設定
     → 読み取り: 500ms、書き込み: 5000ms
     → デッドライン超過のリクエストを優先
     → 飢餓防止と応答性のバランス

  3. CFQ (Completely Fair Queuing):
     プロセスごとにキューを作成し公平にディスパッチ
     → Linux 2.6.18〜4.x のデフォルト
     → デスクトップ向け
     → シングルキューのためSSD性能を活かせない

  シングルキューの問題点:
  ┌──────────────────────────────────────────┐
  │                                          │
  │   CPU0 ─┐                               │
  │   CPU1 ─┤── 単一のリクエストキュー ──→ デバイス│
  │   CPU2 ─┤     ↑ ロック競合              │
  │   CPU3 ─┘                               │
  │                                          │
  │ → マルチコア環境でスケーラビリティが低い  │
  │ → NVMe等の高速デバイスでボトルネックに    │
  └──────────────────────────────────────────┘
```

### 2.2 マルチキューブロックレイヤー（blk-mq）

```
blk-mq (Multi-Queue Block Layer, Linux 3.13+):

  設計思想:
  → マルチコアCPU + 高速ストレージ（NVMe）に対応
  → CPUコアごとにソフトウェアキューを配置
  → ハードウェアキューへの効率的なマッピング
  → ロック競合の大幅削減

  ┌──────────────────────────────────────────────────┐
  │ blk-mq の構造                                    │
  │                                                   │
  │  CPU0 → [SW Queue 0]─┐                           │
  │  CPU1 → [SW Queue 1]─┤── [HW Queue 0] → デバイス│
  │  CPU2 → [SW Queue 2]─┤── [HW Queue 1] → デバイス│
  │  CPU3 → [SW Queue 3]─┘── [HW Queue N] → デバイス│
  │                                                   │
  │  SW Queue: CPUローカル（ロック不要）               │
  │  HW Queue: デバイスのハードウェアキューに対応       │
  │  → NVMe: 最大64K個のHWキュー                      │
  │  → SATA: 通常1個のHWキュー                        │
  └──────────────────────────────────────────────────┘

  blk-mq のメリット:
  - CPUコアごとの独立したキューでロック競合なし
  - NVMe の並列性を完全に活用
  - NUMA ノードに配慮した配置
  - 低レイテンシ（ポーリングモード対応）

  Linux 5.0 以降:
  → 旧シングルキュースケジューラは完全削除
  → 全デバイスが blk-mq ベースに移行
```

### 2.3 現在のLinux I/Oスケジューラ

```
現在のLinux I/Oスケジューラ（blk-mq ベース）:

  1. mq-deadline（マルチキューデッドライン）:
  ┌──────────────────────────────────────────────┐
  │ 概要:                                         │
  │ - 旧 deadline スケジューラの blk-mq 版        │
  │ - リクエストにデッドラインを設定               │
  │ - 読み取り優先（500ms）、書き込み（5000ms）   │
  │                                               │
  │ 動作:                                         │
  │ 1. リクエストを2つのキューで管理               │
  │    - ソートキュー: セクタ番号順（SCAN風）     │
  │    - デッドラインキュー: 期限順               │
  │ 2. 通常はソートキューから処理（シーク最適化）  │
  │ 3. デッドライン超過のリクエストがあれば優先    │
  │ 4. 読み取りを書き込みより優先（対話性向上）   │
  │                                               │
  │ パラメータ:                                    │
  │ /sys/block/sda/queue/iosched/                  │
  │   read_expire:     500  (ms, 読み取り期限)    │
  │   write_expire:    5000 (ms, 書き込み期限)    │
  │   writes_starved:  2    (読み取り優先度)       │
  │   fifo_batch:      16   (バッチサイズ)        │
  │   front_merges:    1    (フロントマージ有効)   │
  │                                               │
  │ 用途: HDD 全般、DB サーバー、仮想化ホスト      │
  │ 特に適する: レイテンシ保証が重要な場合          │
  └──────────────────────────────────────────────┘

  2. BFQ（Budget Fair Queueing）:
  ┌──────────────────────────────────────────────┐
  │ 概要:                                         │
  │ - CFQ の後継（blk-mq ベース）                 │
  │ - プロセスごとに I/O 「予算」を割り当て       │
  │ - 帯域幅とレイテンシの公平な分配               │
  │                                               │
  │ 動作:                                         │
  │ 1. 各プロセスにキューを割り当て               │
  │ 2. 「予算」（処理可能なセクタ数）を設定       │
  │ 3. 予算消費 → 次のプロセスに切り替え          │
  │ 4. 対話的プロセスを自動検出して優先           │
  │ 5. 重み付けによる優先度制御                   │
  │                                               │
  │ パラメータ:                                    │
  │ /sys/block/sda/queue/iosched/                  │
  │   slice_idle:     8 (ms, アイドル待ち時間)    │
  │   low_latency:    1 (低レイテンシモード)      │
  │   timeout_sync:   125 (ms, 同期タイムアウト)  │
  │   max_budget:     0 (0=自動、セクタ数)        │
  │   strict_guarantees: 0                        │
  │                                               │
  │ cgroup 連携:                                   │
  │ → I/O コントローラーとの統合                   │
  │ → コンテナごとの I/O 帯域制限                  │
  │ → 比例配分（weight ベース）                    │
  │                                               │
  │ 用途: デスクトップ、マルチメディア             │
  │ 特に適する: 対話性が重要で低速ストレージの場合 │
  │                                               │
  │ 注意: CPU オーバーヘッドが高い                  │
  │ → 高速 NVMe では none の方が適切              │
  └──────────────────────────────────────────────┘

  3. none（noop）:
  ┌──────────────────────────────────────────────┐
  │ 概要:                                         │
  │ - スケジューリングなし                         │
  │ - リクエストの並べ替えを行わない               │
  │ - マージのみ実行（隣接リクエストの結合）       │
  │                                               │
  │ 動作:                                         │
  │ → リクエストをそのまま FIFO で処理            │
  │ → 隣接ブロックのリクエストはマージ             │
  │ → CPUオーバーヘッドが最小                      │
  │                                               │
  │ 用途:                                         │
  │ - SSD / NVMe（物理的なシークがない）          │
  │ - 仮想マシン（ホストOSがスケジュール済み）    │
  │ - ソフトウェアRAID（RAIDコントローラが最適化） │
  │ - ハードウェアRAID（RAIDカードが最適化）       │
  │                                               │
  │ → 高速デバイスではスケジューラのCPUオーバー    │
  │   ヘッドがボトルネックになりうる               │
  │ → none が最も高いスループットを達成            │
  └──────────────────────────────────────────────┘
```

### 2.4 スケジューラの設定と確認

```bash
# 現在のスケジューラ確認
cat /sys/block/sda/queue/scheduler
# [mq-deadline] bfq none
# → [] で囲まれているのが現在のスケジューラ

# 全ブロックデバイスのスケジューラ確認
for dev in /sys/block/*/queue/scheduler; do
  echo "$(dirname $(dirname $dev) | xargs basename): $(cat $dev)"
done
# sda: [mq-deadline] bfq none
# nvme0n1: [none] mq-deadline bfq

# スケジューラ変更（一時的）
echo "mq-deadline" | sudo tee /sys/block/sda/queue/scheduler
echo "none" | sudo tee /sys/block/nvme0n1/queue/scheduler

# スケジューラ変更（永続化、udevルール）
# /etc/udev/rules.d/60-ioschedulers.rules
# HDD向け
ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/rotational}=="1", \
  ATTR{queue/scheduler}="mq-deadline"

# SSD向け
ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/rotational}=="0", \
  ATTR{queue/scheduler}="none"

# NVMe向け
ACTION=="add|change", KERNEL=="nvme[0-9]*", \
  ATTR{queue/scheduler}="none"

# udevルールのリロード
sudo udevadm control --reload-rules
sudo udevadm trigger

# 回転デバイスかどうかの確認
cat /sys/block/sda/queue/rotational
# 1 = HDD（回転ディスク）
# 0 = SSD（非回転）

# スケジューラのパラメータ確認
ls /sys/block/sda/queue/iosched/
# → read_expire, write_expire, writes_starved, ...

# パラメータの変更
echo 300 | sudo tee /sys/block/sda/queue/iosched/read_expire
echo 3000 | sudo tee /sys/block/sda/queue/iosched/write_expire
```

### 2.5 SSD時代のI/Oスケジューリング

```
SSD/NVMe の特性とスケジューリング:

  HDD vs SSD の I/O 特性:
  ┌──────────────────┬──────────────────┬──────────────────┐
  │ 項目              │ HDD              │ SSD (NVMe)       │
  ├──────────────────┼──────────────────┼──────────────────┤
  │ ランダム読み取り │ 100-200 IOPS     │ 100K-1M+ IOPS   │
  │ シーケンシャル読み│ 100-200 MB/s     │ 3-7 GB/s        │
  │ レイテンシ       │ 5-15 ms          │ 0.02-0.1 ms     │
  │ 並列度           │ 1（物理ヘッド1つ）│ 最大4M（64K×64K）│
  │ シーク影響       │ 大きい           │ なし             │
  │ 推奨スケジューラ │ mq-deadline      │ none             │
  └──────────────────┴──────────────────┴──────────────────┘

  NVMe の並列I/O構造:
  ┌──────────────────────────────────────────┐
  │ NVMe の構造                               │
  │                                           │
  │ NVMe コントローラ                          │
  │ ├── Submission Queue 0 ─→ Completion Queue 0│
  │ ├── Submission Queue 1 ─→ Completion Queue 1│
  │ ├── Submission Queue 2 ─→ Completion Queue 2│
  │ ├── ...                                    │
  │ └── SQ 65535 ────────→ CQ 65535            │
  │                                           │
  │ 各キューに最大 65536 個のコマンド          │
  │ → 理論最大: 64K × 64K = 約40億の並列I/O   │
  │                                           │
  │ 実際の利用:                                │
  │ → CPU コアごとに 1 つのキューペア          │
  │ → 8コアCPU → 8 キューペア                 │
  │ → 各キューに 1024 エントリ程度             │
  └──────────────────────────────────────────┘

  なぜ SSD では none が最適か:
  1. 物理的なシークがない → 並べ替えの意味がない
  2. 超高並列 → キュー内の順序は性能に影響しない
  3. スケジューラの CPU オーバーヘッドが相対的に大きくなる
  4. デバイス側で独自のスケジューリング（FTL）を実行
  5. スケジューラの遅延がデバイスの低レイテンシを相殺

  ただし例外:
  - SATA SSD（キュー深度32が上限）: mq-deadline も有効な場合あり
  - デスクトップで BFQ を使う場合:
    → バックグラウンドの大量 I/O が対話性を妨げないようにする
    → 例: 巨大ファイルコピー中のアプリ起動速度
```

---

## 3. I/O最適化テクニック

### 3.1 ページキャッシュ

```
ページキャッシュ (Page Cache):
  読み込んだディスクデータをメモリにキャッシュする仕組み
  → 2回目以降のアクセスはメモリから読み取り（超高速）
  → Linux はメモリの大半をページキャッシュに使用

  動作原理:
  ┌──────────────────────────────────────────┐
  │ read() の処理フロー:                      │
  │                                           │
  │ 1. ページキャッシュを検索                  │
  │    ├── ヒット → メモリから直接返す（μs）  │
  │    └── ミス  → ディスクから読み込み（ms）  │
  │               └── キャッシュに格納         │
  │                  └── データを返す          │
  │                                           │
  │ write() の処理フロー:                     │
  │                                           │
  │ 1. ページキャッシュに書き込み              │
  │ 2. ページを「dirty」にマーク              │
  │ 3. write() はすぐにリターン               │
  │ 4. カーネルスレッド（pdflush/writeback）が │
  │    バックグラウンドでディスクに書き出し     │
  └──────────────────────────────────────────┘

  ページキャッシュの管理:
  ┌──────────────────────────────────────────┐
  │ メモリの使い方（4GB RAMの例）             │
  │                                          │
  │ ┌────────────────────────────────────┐   │
  │ │ アプリケーション使用: 1.5GB        │   │
  │ ├────────────────────────────────────┤   │
  │ │ ページキャッシュ: 2.0GB            │   │
  │ ├────────────────────────────────────┤   │
  │ │ カーネル/予約: 0.5GB              │   │
  │ └────────────────────────────────────┘   │
  │                                          │
  │ → "free" コマンドで Available が少なくても│
  │   ページキャッシュが多い場合は問題なし    │
  │ → アプリがメモリを要求すれば             │
  │   キャッシュは自動的に解放される         │
  └──────────────────────────────────────────┘
```

```bash
# ページキャッシュの状態確認
free -h
# total   used   free   shared  buff/cache  available
# 16Gi   4.5Gi  1.2Gi  256Mi   10.3Gi       11.0Gi
# → buff/cache の 10.3GB がページキャッシュ + バッファ
# → available の 11.0GB が実際に使えるメモリ

# より詳細な情報
cat /proc/meminfo | grep -E "^(MemTotal|MemFree|Buffers|Cached|Dirty|Writeback)"
# MemTotal:       16384000 kB
# MemFree:         1200000 kB
# Buffers:          256000 kB
# Cached:         10000000 kB   ← ページキャッシュ
# Dirty:             32000 kB   ← 未書き出しデータ
# Writeback:             0 kB   ← 書き出し中データ

# ページキャッシュのクリア（テスト・ベンチマーク用）
# 注意: 本番環境では実行しないこと！
sudo sync                                    # dirty ページを書き出し
echo 1 | sudo tee /proc/sys/vm/drop_caches  # ページキャッシュクリア
echo 2 | sudo tee /proc/sys/vm/drop_caches  # dentry/inodeキャッシュ
echo 3 | sudo tee /proc/sys/vm/drop_caches  # 両方クリア

# ダーティページの設定（書き出しタイミング制御）
# ダーティ比率（全メモリに対するダーティページの割合）
cat /proc/sys/vm/dirty_ratio           # デフォルト: 20
cat /proc/sys/vm/dirty_background_ratio # デフォルト: 10
cat /proc/sys/vm/dirty_expire_centisecs # デフォルト: 3000 (30秒)
cat /proc/sys/vm/dirty_writeback_centisecs # デフォルト: 500 (5秒)

# dirty_ratio: この割合を超えるとwrite()がブロック
# dirty_background_ratio: この割合を超えるとバックグラウンド書き出し開始

# DB サーバー向け設定（より頻繁に書き出し）
sudo sysctl -w vm.dirty_ratio=5
sudo sysctl -w vm.dirty_background_ratio=2
sudo sysctl -w vm.dirty_expire_centisecs=500

# 大容量メモリサーバー向け（バイト単位で指定）
sudo sysctl -w vm.dirty_bytes=268435456           # 256MB
sudo sysctl -w vm.dirty_background_bytes=67108864 # 64MB
```

### 3.2 先読み（Read-ahead）

```
先読み（Read-ahead）:
  連続読み取りを検知して先にデータを読む
  → アプリケーションが要求する前にデータを準備
  → シーケンシャル読み取りの大幅な高速化

  動作原理:
  ┌──────────────────────────────────────────┐
  │ アプリケーションの読み取りパターン:        │
  │                                          │
  │ 時刻1: read(offset=0, size=4KB)         │
  │ 時刻2: read(offset=4KB, size=4KB)       │
  │ 時刻3: read(offset=8KB, size=4KB)       │
  │   ↑ シーケンシャルパターンを検出！        │
  │                                          │
  │ カーネルの先読み:                         │
  │ → offset=12KB から 128KB を先行読み込み  │
  │ → アプリの次の read() はキャッシュヒット  │
  │                                          │
  │ 適応的先読み:                             │
  │ - 先読みサイズを動的に調整               │
  │ - 初期: 小さなサイズ                     │
  │ - シーケンシャル確認後: 段階的に拡大      │
  │ - 最大: readahead_kb の値まで            │
  └──────────────────────────────────────────┘
```

```bash
# 先読みサイズの確認
cat /sys/block/sda/queue/read_ahead_kb
# デフォルト: 128 (128KB)

# 先読みサイズの変更
echo 256 | sudo tee /sys/block/sda/queue/read_ahead_kb   # 256KB
echo 2048 | sudo tee /sys/block/sda/queue/read_ahead_kb  # 2MB

# 推奨設定:
# HDD: 256〜1024KB（シーケンシャル読み取りが多い場合）
# SSD: 128〜256KB（デフォルトで十分）
# RAID: 1024〜4096KB（ストライプサイズに合わせる）
# データベース: 小さめ（ランダムI/Oが多い）

# blockdev での設定
sudo blockdev --getra /dev/sda    # 先読みサイズ取得（セクタ単位）
sudo blockdev --setra 512 /dev/sda # 256KB（512セクタ×512B）

# アプリケーション単位の先読み制御
# posix_fadvise() システムコール
# POSIX_FADV_SEQUENTIAL: シーケンシャルアクセスを宣言
# POSIX_FADV_RANDOM:     ランダムアクセスを宣言
# POSIX_FADV_WILLNEED:   近い将来アクセスする
# POSIX_FADV_DONTNEED:   もうアクセスしない（キャッシュ解放ヒント）
```

### 3.3 io_uring（Linux 5.1+）

```
io_uring: 高性能非同期I/Oインターフェース

  従来の非同期I/O方式との比較:
  ┌──────────────┬────────────────────────────────────┐
  │ 方式          │ 特徴                                │
  ├──────────────┼────────────────────────────────────┤
  │ 同期I/O      │ read()/write() がブロック            │
  │ (blocking)   │ 単純だが並列性が低い                 │
  ├──────────────┼────────────────────────────────────┤
  │ select/poll  │ FD の準備状態を確認                  │
  │              │ FD 数に応じて O(n) のオーバーヘッド   │
  ├──────────────┼────────────────────────────────────┤
  │ epoll        │ イベント駆動型                       │
  │              │ O(1) で準備完了 FD を取得            │
  │              │ ただし read/write 自体は同期         │
  ├──────────────┼────────────────────────────────────┤
  │ aio          │ カーネル非同期I/O                    │
  │ (Linux AIO)  │ Direct I/O のみ対応                 │
  │              │ バッファードI/O非対応                 │
  │              │ API が複雑                          │
  ├──────────────┼────────────────────────────────────┤
  │ io_uring     │ 完全非同期I/O                       │
  │ (Linux 5.1+) │ システムコールオーバーヘッド最小     │
  │              │ バッファードI/O対応                   │
  │              │ ゼロコピー対応                       │
  │              │ ポーリングモード対応                 │
  │              │ → 最高性能の I/O インターフェース    │
  └──────────────┴────────────────────────────────────┘

  io_uring の仕組み:
  ┌──────────────────────────────────────────────────┐
  │                                                   │
  │  ユーザ空間          カーネル空間                   │
  │                                                   │
  │  ┌──────────────┐   ┌──────────────┐              │
  │  │ Submission   │   │              │              │
  │  │ Queue (SQ)   │──→│  I/O 処理    │              │
  │  │  リクエスト   │   │  エンジン    │              │
  │  │  を投入      │   │              │              │
  │  └──────────────┘   └──────┬───────┘              │
  │                            │                      │
  │  ┌──────────────┐          │                      │
  │  │ Completion   │←─────────┘                      │
  │  │ Queue (CQ)   │                                 │
  │  │  完了通知     │                                 │
  │  │  を受け取り   │                                 │
  │  └──────────────┘                                 │
  │                                                   │
  │  SQ と CQ はカーネルとユーザ空間で共有メモリ      │
  │  → システムコールなしでリクエスト投入・結果取得    │
  │  → io_uring_enter() は必要時のみ（ポーリング時不要）│
  └──────────────────────────────────────────────────┘

  io_uring の主な機能（Linux バージョン別）:
  5.1:  基本的な読み書き
  5.4:  ネットワーク I/O (accept, connect, recv, send)
  5.5:  splice, tee
  5.6:  固定バッファ登録、IO ポーリング最適化
  5.7:  リンクされた操作、タイムアウト
  5.10: 制限モード、パーミッション制御
  5.11: shutdown, renameat, unlinkat
  5.12: mkdirat, symlinkat
  5.15: sendmsg_zc（ゼロコピー送信）
  6.0:  send_zc, recv_zc
  6.1:  futex 操作
```

```c
// io_uring の基本的な使用例（C言語）
#include <liburing.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

int main() {
    struct io_uring ring;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    char buf[4096];
    int fd;

    // io_uring の初期化（キュー深度128）
    io_uring_queue_init(128, &ring, 0);

    // ファイルオープン
    fd = open("test.txt", O_RDONLY);

    // Submission Queue Entry を取得
    sqe = io_uring_get_sqe(&ring);

    // 読み取りリクエストを準備
    io_uring_prep_read(sqe, fd, buf, sizeof(buf), 0);

    // リクエストを投入
    io_uring_submit(&ring);

    // 完了待ち
    io_uring_wait_cqe(&ring, &cqe);

    printf("Read %d bytes\n", cqe->res);

    // 完了エントリを消費
    io_uring_cqe_seen(&ring, cqe);

    // クリーンアップ
    close(fd);
    io_uring_queue_exit(&ring);
    return 0;
}

// コンパイル: gcc -o io_uring_example io_uring_example.c -luring
```

### 3.4 Direct I/O

```
Direct I/O:
  ページキャッシュをバイパスしてディスクに直接アクセス

  通常のI/O（Buffered I/O）:
  アプリ → ページキャッシュ → ディスク
  → カーネルがキャッシュ管理
  → 大部分のアプリに最適

  Direct I/O:
  アプリ → ディスク（ページキャッシュをバイパス）
  → アプリが独自のキャッシュを管理
  → データベースが主な利用者

  ┌────────────────────────────────────────────┐
  │ Buffered I/O:                               │
  │ App → [Page Cache] → Disk                  │
  │       キャッシュで高速化                     │
  │       ダブルバッファリング（アプリ＋OS）     │
  │                                             │
  │ Direct I/O:                                 │
  │ App → → → → → → Disk                      │
  │       キャッシュなし                         │
  │       アプリが自前でキャッシュ管理           │
  └────────────────────────────────────────────┘

  使用条件:
  - open() に O_DIRECT フラグを指定
  - バッファのアドレスがブロックサイズにアライン
  - 読み書きサイズがブロックサイズの倍数
  - ファイルオフセットがブロックサイズの倍数

  主な利用者:
  - MySQL InnoDB (innodb_flush_method = O_DIRECT)
  - PostgreSQL (効果は限定的、通常は不使用)
  - Oracle Database
  - QEMU/KVM（仮想ディスクI/O）

  設定例:
  # MySQL
  [mysqld]
  innodb_flush_method = O_DIRECT

  # QEMU
  qemu-system-x86_64 -drive file=disk.img,cache=none
  # cache=none: O_DIRECT を使用
  # cache=writeback: ページキャッシュを使用
  # cache=writethrough: 同期書き込み
```

### 3.5 mmap I/O

```
mmap（メモリマップドI/O）:
  ファイルをプロセスのアドレス空間にマッピング

  ┌────────────────────────────────────────────┐
  │ read/write vs mmap:                         │
  │                                             │
  │ read():                                     │
  │ 1. システムコール発行                       │
  │ 2. カーネル空間にデータコピー               │
  │ 3. ユーザ空間にデータコピー                 │
  │ → 2回のデータコピー                         │
  │                                             │
  │ mmap():                                     │
  │ 1. ページテーブルを設定（初回のみ）         │
  │ 2. アクセス時にページフォルト発生           │
  │ 3. ページキャッシュのページを直接マッピング │
  │ → データコピーなし                          │
  └────────────────────────────────────────────┘

  mmap の利点:
  - データコピーが不要（ゼロコピー）
  - ランダムアクセスが効率的
  - 複数プロセスで共有可能（MAP_SHARED）
  - 実行ファイルのロード

  mmap の欠点:
  - ページフォルトのオーバーヘッド
  - 大きなファイルの一部だけアクセスする場合に非効率
  - エラーハンドリングが難しい（SIGBUS）
  - TLB ミスのオーバーヘッド

  使用例:
  - データベース（LMDB, SQLite mmap mode）
  - 実行ファイルのロード（ELF テキストセグメント）
  - 共有メモリ（IPC）
  - 設定ファイルの読み込み
```

### 3.6 I/O優先度とcgroup

```
I/O 優先度制御:

  ionice コマンド:
  → プロセスの I/O スケジューリングクラスと優先度を設定

  クラス:
  1 (Realtime):  最優先。starvation のリスクあり
  2 (Best-effort): デフォルト。優先度 0-7（0が最高）
  3 (Idle):       他にI/Oがない場合のみ処理

  使用例:
  # 低優先度でバックアップ実行
  ionice -c 3 rsync -a /data /backup/

  # 高優先度でデータベース実行
  ionice -c 1 -n 0 mysqld

  # 現在の I/O 優先度を確認
  ionice -p $(pgrep -f mysqld)

  # 実行中のプロセスの優先度変更
  ionice -c 2 -n 4 -p 1234

cgroup v2 によるI/O制御:
  → コンテナ/プロセスグループ単位でのI/O制限

  # I/O 帯域幅制限
  # cgroup v2 (systemd)
  systemctl set-property myservice.service IOWriteBandwidthMax="/dev/sda 50M"
  systemctl set-property myservice.service IOReadBandwidthMax="/dev/sda 100M"

  # I/O ウエイト（比例配分）
  systemctl set-property myservice.service IOWeight=100
  # 範囲: 1-10000、デフォルト: 100

  # Docker での I/O 制限
  docker run --device-write-bps /dev/sda:50mb \
             --device-read-bps /dev/sda:100mb \
             --blkio-weight 500 \
             myapp

  # Kubernetes での I/O 制限（cgroup v2 必要）
  # Guaranteed QoS クラスで制御
```

---

## 4. I/Oモニタリングとトラブルシューティング

### 4.1 I/Oモニタリングツール

```bash
# === iostat: デバイスレベルの I/O 統計 ===
iostat -x 1 5     # 拡張統計、1秒間隔、5回
# Device  r/s   w/s   rkB/s  wkB/s  rrqm/s  wrqm/s  %util  await
# sda     50.0  30.0  200.0  120.0  5.0     10.0    45.0   8.50

# 主要な指標:
# r/s, w/s:      読み取り/書き込みの IOPS
# rkB/s, wkB/s:  読み取り/書き込みのスループット
# await:         I/O の平均待ち時間（ms）
# %util:         デバイスの稼働率（100%に近いと飽和）
# avgqu-sz:      平均キュー長
# svctm:         平均サービス時間（非推奨、不正確）

# === iotop: プロセスレベルの I/O 統計 ===
sudo iotop -o     # I/O を行っているプロセスのみ表示
sudo iotop -b     # バッチモード（スクリプト用）
sudo iotop -a     # 累積表示

# === blktrace: ブロックレベルの詳細トレース ===
# トレースの開始
sudo blktrace -d /dev/sda -o trace

# トレースの解析
blkparse -i trace -d output.bin

# BPF ベースのトレース（より軽量）
sudo biosnoop-bpfcc             # I/O レイテンシの表示
sudo biotop-bpfcc               # I/O のトップ表示
sudo biolatency-bpfcc           # I/O レイテンシのヒストグラム

# === pidstat: プロセスごとの I/O 統計 ===
pidstat -d 1      # I/O 統計、1秒間隔
# PID   kB_rd/s  kB_wr/s  kB_ccwr/s  Command
# 1234  500.0    200.0    0.0        mysqld

# === vmstat: システム全体の I/O 概要 ===
vmstat 1
# bi: ブロック入力（読み取り、blocks/s）
# bo: ブロック出力（書き込み、blocks/s）
# wa: I/O 待ち時間の割合（%）
```

### 4.2 I/Oパフォーマンス問題の診断

```
I/O 性能問題の診断フロー:

  1. 症状の確認:
  ┌──────────────────────────────────────────┐
  │ $ iostat -x 1                             │
  │                                           │
  │ %util > 90%                               │
  │ → デバイスが飽和状態                       │
  │                                           │
  │ await > 50ms (HDD) / > 5ms (SSD)         │
  │ → I/O レイテンシが高い                     │
  │                                           │
  │ avgqu-sz > 10                             │
  │ → キューが長い（処理が追いつかない）       │
  └──────────────────────────────────────────┘

  2. 原因の特定:
  ┌──────────────────────────────────────────┐
  │ $ iotop -o                                │
  │ → I/O を大量に行っているプロセスを特定     │
  │                                           │
  │ $ sudo biosnoop-bpfcc                     │
  │ → 個々の I/O リクエストの詳細              │
  │                                           │
  │ $ cat /proc/<pid>/io                      │
  │ → 特定プロセスの I/O 統計                  │
  │   rchar: 読み取り要求バイト数              │
  │   wchar: 書き込み要求バイト数              │
  │   syscr: read システムコール数             │
  │   syscw: write システムコール数            │
  │   read_bytes: 実際のディスク読み取り       │
  │   write_bytes: 実際のディスク書き込み      │
  └──────────────────────────────────────────┘

  3. 対処法:
  ┌──────────────────────────────────────────┐
  │ ランダム I/O が多い場合:                   │
  │ → SSD/NVMe へのアップグレード              │
  │ → メモリ増設（ページキャッシュ拡大）       │
  │ → アプリケーションのアクセスパターン最適化 │
  │                                           │
  │ シーケンシャル I/O が遅い場合:             │
  │ → read_ahead_kb の増加                    │
  │ → ストライプサイズの最適化（RAID）         │
  │ → ブロックサイズの調整                     │
  │                                           │
  │ 書き込みが溜まっている場合:                │
  │ → dirty_ratio / dirty_bytes の調整        │
  │ → ジャーナルサイズの確認                   │
  │ → fsync の頻度の確認                       │
  │                                           │
  │ 特定プロセスが I/O を独占:                 │
  │ → ionice で優先度を下げる                  │
  │ → cgroup で帯域を制限                      │
  │ → BFQ スケジューラの使用                   │
  └──────────────────────────────────────────┘
```

### 4.3 I/Oベンチマーク

```bash
# === fio: 標準的なストレージベンチマークツール ===

# シーケンシャル読み取り
fio --name=seq_read \
    --rw=read \
    --bs=1M \
    --size=4G \
    --numjobs=1 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=64

# ランダム読み取り（4K、データベースワークロード模擬）
fio --name=rand_read_4k \
    --rw=randread \
    --bs=4k \
    --size=4G \
    --numjobs=8 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=32 \
    --group_reporting

# ランダム書き込み（4K）
fio --name=rand_write_4k \
    --rw=randwrite \
    --bs=4k \
    --size=4G \
    --numjobs=8 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=32 \
    --group_reporting

# 混合ワークロード（70% 読み取り / 30% 書き込み）
fio --name=mixed \
    --rw=randrw \
    --rwmixread=70 \
    --bs=4k \
    --size=4G \
    --numjobs=8 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=32 \
    --group_reporting

# io_uring エンジンの使用（Linux 5.1+）
fio --name=io_uring_test \
    --rw=randread \
    --bs=4k \
    --size=4G \
    --numjobs=4 \
    --runtime=60 \
    --time_based \
    --ioengine=io_uring \
    --direct=1 \
    --iodepth=128 \
    --fixedbufs=1 \
    --registerfiles=1 \
    --sqthread_poll=1

# レイテンシ分布の確認
fio --name=latency \
    --rw=randread \
    --bs=4k \
    --size=1G \
    --numjobs=1 \
    --runtime=30 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=1 \
    --lat_percentiles=1 \
    --percentile_list=50:90:95:99:99.9:99.99

# 結果の読み方:
# IOPS:     1秒あたりのI/O操作数
# BW:       帯域幅（スループット）
# lat:      レイテンシ（avg, min, max, percentiles）
# clat:     完了レイテンシ
# slat:     提出レイテンシ
```

---

## 5. 高度なI/O技術

### 5.1 ゼロコピー

```
ゼロコピー (Zero-Copy):
  データの不要なコピーを排除して効率化

  通常のファイル転送（sendfile以前）:
  ┌──────────────────────────────────────────┐
  │ read(fd, buf, size):                      │
  │ ディスク → カーネルバッファ → ユーザバッファ│
  │                                           │
  │ write(sockfd, buf, size):                 │
  │ ユーザバッファ → カーネルバッファ → NIC    │
  │                                           │
  │ → 4回のコピー、2回のコンテキストスイッチ  │
  └──────────────────────────────────────────┘

  sendfile() によるゼロコピー:
  ┌──────────────────────────────────────────┐
  │ sendfile(sockfd, fd, offset, size):       │
  │ ディスク → カーネルバッファ → NIC          │
  │                                           │
  │ → 2回のコピー（DMA）、ユーザ空間経由なし  │
  │ → 1回のシステムコール                     │
  └──────────────────────────────────────────┘

  splice() / tee():
  → パイプを使ったゼロコピーデータ転送
  → カーネル内でページ参照を転送（データコピーなし）

  io_uring のゼロコピー送信:
  → send_zc 操作で完全なゼロコピー
  → ページピンニングによりカーネルがユーザバッファを直接参照

  活用例:
  - Nginx: sendfile on; (静的ファイル配信)
  - Kafka: ゼロコピーによるメッセージ配信
  - 動画ストリーミングサーバー
```

### 5.2 I/Oポーリング

```
I/Oポーリング:
  割り込みの代わりにCPUがI/O完了をポーリング

  割り込み方式:
  CPU ─── 他の仕事 ──── ←割り込み── 処理
  → コンテキストスイッチのオーバーヘッド
  → 高IOPS時に割り込み処理が過大に

  ポーリング方式:
  CPU ─── ポーリング ── 完了検出 ── 処理
  → コンテキストスイッチなし
  → レイテンシが最小
  → CPU使用率は100%に近づく

  io_uring のポーリングモード:
  ┌──────────────────────────────────────────┐
  │ IORING_SETUP_IOPOLL:                      │
  │ → デバイスの完了をポーリング              │
  │ → NVMe で最大の効果                       │
  │ → Direct I/O 必須                         │
  │                                           │
  │ IORING_SETUP_SQPOLL:                      │
  │ → カーネルスレッドがSQをポーリング        │
  │ → アプリはシステムコールなしでI/O発行     │
  │ → 完全にユーザ空間で完結                  │
  │                                           │
  │ 両方を組み合わせ:                          │
  │ → 最高のI/O性能（ただしCPU消費大）        │
  └──────────────────────────────────────────┘

  NVMe のポーリングモード:
  # カーネルパラメータで有効化
  # /sys/block/nvme0n1/queue/io_poll
  echo 1 | sudo tee /sys/block/nvme0n1/queue/io_poll

  # ポーリング遅延の設定
  echo 0 | sudo tee /sys/block/nvme0n1/queue/io_poll_delay
  # -1: 無効、0: 即座にポーリング、>0: 遅延後ポーリング（ns）
```

### 5.3 I/Oのバリアとフラッシュ

```
書き込みバリアとデータの永続化:

  問題:
  → ディスクには書き込みキャッシュがある
  → OS がディスクに書き込んでも、キャッシュに留まる可能性
  → 電源断でキャッシュ内データが消失
  → ジャーナリングの整合性が壊れる可能性

  解決策:
  1. 書き込みバリア（Write Barrier）:
     → バリア前の書き込みがディスクに到達したことを保証
     → バリア後の書き込みはバリア前の後に実行される

  2. FUA（Force Unit Access）:
     → 特定の書き込みをディスクキャッシュをバイパスして直接書き込み
     → NVMe/SAS で対応

  3. fsync() / fdatasync():
     → ファイルのデータをディスクに永続化
     → fsync: データ + メタデータ
     → fdatasync: データ + 必要なメタデータのみ

  4. sync():
     → 全ファイルのダーティページをディスクに書き出し

  ┌──────────────────────────────────────────┐
  │ 永続化の保証レベル:                       │
  │                                          │
  │ write()        : ページキャッシュまで     │
  │                  → 電源断でデータ消失可能│
  │                                          │
  │ fdatasync()    : ディスクキャッシュまで   │
  │                  → BBU付きなら安全       │
  │                                          │
  │ fsync()        : ディスクまで（メタデータ含む）│
  │                  → 最も安全              │
  │                                          │
  │ O_SYNC         : 毎回の write() で fsync │
  │                  → 性能低下が大きい      │
  │                                          │
  │ O_DSYNC        : 毎回の write() で fdatasync│
  │                  → O_SYNC より高速       │
  └──────────────────────────────────────────┘

  バリアの制御:
  # バリア有効（デフォルト、推奨）
  mount -o barrier=1 /dev/sda1 /mnt

  # バリア無効（BBU付きRAIDコントローラの場合のみ）
  mount -o barrier=0 /dev/sda1 /mnt
  # → BBU（バッテリーバックアップユニット）がキャッシュを保護

  # ディスクの書き込みキャッシュ確認
  sudo hdparm -W /dev/sda
  # /dev/sda: write-caching = 1 (on)

  # 書き込みキャッシュの無効化（安全性重視）
  sudo hdparm -W 0 /dev/sda
```

---

## 実践演習

### 演習1: [基礎] -- I/Oスケジューラの確認と変更

```bash
# 全デバイスのスケジューラ確認
for dev in /sys/block/*/queue/scheduler; do
  echo "$(dirname $(dirname $dev) | xargs basename): $(cat $dev)"
done

# スケジューラの変更（一時的）
echo "bfq" | sudo tee /sys/block/sda/queue/scheduler
cat /sys/block/sda/queue/scheduler

# スケジューラパラメータの確認
ls /sys/block/sda/queue/iosched/
for param in /sys/block/sda/queue/iosched/*; do
  echo "$(basename $param): $(cat $param)"
done

# デバイス種別の確認
for dev in /sys/block/*/queue/rotational; do
  name=$(dirname $(dirname $dev) | xargs basename)
  type=$(cat $dev)
  echo "$name: $([ $type -eq 1 ] && echo 'HDD' || echo 'SSD')"
done
```

### 演習2: [応用] -- ページキャッシュの効果測定

```bash
# テスト用の大きなファイルを作成
dd if=/dev/urandom of=/tmp/testfile bs=1M count=512

# キャッシュクリア
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 1回目の読み取り（ディスクから）
echo "=== 1回目（ディスクから）==="
time dd if=/tmp/testfile of=/dev/null bs=1M
# → 数秒かかる

# 2回目の読み取り（ページキャッシュから）
echo "=== 2回目（キャッシュから）==="
time dd if=/tmp/testfile of=/dev/null bs=1M
# → 1秒未満

# キャッシュ状態の確認
cat /proc/meminfo | grep -E "^(Cached|Buffers)"

# 特定ファイルのキャッシュ状態確認（fincore, Linux 4.2+）
fincore /tmp/testfile
# → RES: キャッシュに載っているサイズ
# → PAGES: キャッシュされたページ数

# クリーンアップ
rm /tmp/testfile
```

### 演習3: [上級] -- I/Oパフォーマンス分析

```bash
# iostat で I/O 状況をモニタリング
# ターミナル1: I/O 負荷の発生
fio --name=load --rw=randrw --bs=4k --size=1G \
    --numjobs=4 --runtime=60 --time_based \
    --ioengine=libaio --direct=1 --iodepth=32

# ターミナル2: モニタリング
iostat -x 1 | tee /tmp/iostat.log

# 主要指標の解析
# await が高い → I/O レイテンシ問題
# %util が 100% に近い → デバイス飽和
# avgqu-sz が大きい → キューが長い
# rrqm/s, wrqm/s → I/O マージの効果

# iotop でプロセスレベルの分析
sudo iotop -o -d 1

# BPF ツールによる詳細分析（bcc-tools パッケージ）
# I/O レイテンシのヒストグラム
sudo biolatency-bpfcc 10 1
# → レイテンシの分布を確認

# 個々のI/Oリクエストの詳細
sudo biosnoop-bpfcc
# TIME     COMM     PID  DISK  T  SECTOR  BYTES  LAT(ms)

# I/O パターンの可視化
sudo bitesize-bpfcc
# → I/O サイズの分布を確認
```

### 演習4: [上級] -- スケジューラの性能比較

```bash
# 各スケジューラでのベンチマーク比較
for sched in mq-deadline bfq none; do
  echo "=== Scheduler: $sched ==="
  echo "$sched" | sudo tee /sys/block/sda/queue/scheduler

  fio --name=${sched}_test \
      --rw=randread --bs=4k --size=1G \
      --numjobs=4 --runtime=30 --time_based \
      --ioengine=libaio --direct=1 --iodepth=32 \
      --group_reporting \
      --output-format=terse \
    | awk -F';' '{print "IOPS:", $8, "BW:", $7, "lat:", $40}'

  echo ""
done
```

---

## まとめ

| スケジューラ | 特徴 | 用途 |
|------------|------|------|
| mq-deadline | デッドライン保証、飢餓防止 | HDD、DB サーバー |
| BFQ | 公平性重視、対話性優先 | デスクトップ、マルチメディア |
| none | スケジューリングなし、最小オーバーヘッド | SSD/NVMe、仮想マシン |

| I/O技術 | 特徴 | 用途 |
|---------|------|------|
| ページキャッシュ | 読み取りデータのメモリキャッシュ | 全般 |
| 先読み | シーケンシャル読み取りの先行読み込み | ログ処理、ストリーミング |
| io_uring | 高性能非同期I/O | 高IOPS アプリケーション |
| Direct I/O | ページキャッシュバイパス | データベース |
| ゼロコピー | データコピーの排除 | Web サーバー、ストリーミング |
| ポーリング | 割り込みなしI/O完了検出 | 超低レイテンシ要件 |

---

## 次に読むべきガイド
→ [[../04-io-and-devices/00-device-drivers.md]] -- デバイスドライバ

---

## 参考文献
1. Love, R. "Linux Kernel Development." 3rd Ed, Ch.14, 2010.
2. Bovet, D. & Cesati, M. "Understanding the Linux Kernel." 3rd Ed, O'Reilly, 2005.
3. Axboe, J. "Efficient IO with io_uring." Kernel.dk, 2019.
4. Arpaci-Dusseau, R. & Arpaci-Dusseau, A. "Operating Systems: Three Easy Pieces." Ch.37, 2018.
5. Gregg, B. "Systems Performance." 2nd Ed, Addison-Wesley, 2020.
6. Gregg, B. "BPF Performance Tools." Addison-Wesley, 2019.
7. Linux Block Layer Documentation. https://www.kernel.org/doc/html/latest/block/
8. io_uring Documentation. https://kernel.dk/io_uring.pdf
9. Bjørling, M. et al. "Linux Block IO: Introducing Multi-queue SSD Access on Multi-core Systems." SYSTOR, 2013.
