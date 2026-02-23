# 主要ファイルシステム実装

> ファイルシステムの選択はワークロードに依存する。万能なFSは存在しない。

## この章で学ぶこと

- [ ] 主要なファイルシステムの特徴を比較できる
- [ ] ワークロードに応じたFS選択ができる
- [ ] 各FSの内部構造とアルゴリズムを理解する
- [ ] FS固有のチューニング手法を習得する
- [ ] FS間のデータ移行を行える

---

## 1. ext4（Linux標準）

### 1.1 ext4の歴史と進化

```
ext ファミリーの系譜:

  ext (1992): Linux 最初のファイルシステム
  → 最大 2GB パーティション
  → UFS（Unix File System）の影響

  ext2 (1993): 本格的なLinuxファイルシステム
  → 最大 4TB パーティション、2GB ファイル
  → ジャーナリングなし
  → 長期間使用された安定設計
  → USB メモリ等でまだ使われる場合あり

  ext3 (2001): ext2 + ジャーナリング
  → ext2 との後方互換性
  → オンラインでの ext2 → ext3 アップグレード可能
  → 3つのジャーナリングモード導入

  ext4 (2008): ext3 の大幅拡張
  → 最大 1EB パーティション、16TB ファイル
  → エクステント、遅延アロケーション等の新機能
  → Ubuntu, Debian のデフォルト
  → 最も広く使われている Linux FS

  進化の図:
  ext (1992)
   │
   ↓
  ext2 (1993) ── ジャーナリング追加 ──→ ext3 (2001)
                                          │
                                          ↓
                   大容量対応/エクステント ──→ ext4 (2008)
```

### 1.2 ext4 の主要機能

```
ext4 (Fourth Extended Filesystem, 2008):
  ext2 → ext3(ジャーナリング追加) → ext4(大容量対応)

  主な特徴:
  ┌──────────────────────────────────────────────────┐
  │ 容量制限                                          │
  │ - 最大ボリューム: 1EB (エクサバイト)               │
  │ - 最大ファイル: 16TB                              │
  │ - 最大ファイル名: 255バイト                        │
  │ - 最大パス長: 4096バイト                           │
  │ - ディレクトリ内最大エントリ: 約1000万              │
  ├──────────────────────────────────────────────────┤
  │ エクステント (Extents)                             │
  │ - 連続ブロックを1エントリで管理（断片化削減）       │
  │ - エクステントツリー（B木構造）                     │
  │ - 最大128MB の連続領域を1エクステントで表現         │
  ├──────────────────────────────────────────────────┤
  │ 遅延割り当て (Delayed Allocation)                  │
  │ - 書き込みをページキャッシュに保留                  │
  │ - 実際のディスク書き込み時に最適な配置を決定        │
  │ - 連続ブロック確保の確率が向上                      │
  │ - 短命ファイルの不要な書き込みを回避                │
  ├──────────────────────────────────────────────────┤
  │ ジャーナリング                                     │
  │ - メタデータ+データの整合性保証                     │
  │ - 3つのモード: journal, ordered, writeback         │
  │ - JBD2 (Journaling Block Device 2) エンジン        │
  ├──────────────────────────────────────────────────┤
  │ ディレクトリインデックス                            │
  │ - HTree（ハッシュB木）で高速ルックアップ           │
  │ - ハーフMD4ハッシュ関数使用                        │
  │ - 数百万ファイルのディレクトリでも高速              │
  ├──────────────────────────────────────────────────┤
  │ マルチブロックアロケーション                        │
  │ - 複数ブロックを一度に割り当て                     │
  │ - ブロックアロケータの呼び出し回数を削減            │
  │ - 連続ブロック確保の効率向上                        │
  ├──────────────────────────────────────────────────┤
  │ プリアロケーション (fallocate)                      │
  │ - ファイルに事前にブロックを予約                    │
  │ - データベースファイル等で断片化を防止              │
  │ - POSIX fallocate() システムコール                 │
  ├──────────────────────────────────────────────────┤
  │ Flex Block Groups                                  │
  │ - 複数ブロックグループのメタデータを集約            │
  │ - メタデータの局所性向上                           │
  │ - 大容量FS での性能改善                            │
  └──────────────────────────────────────────────────┘

  利点: 安定性、互換性、ツールの充実、広範なテスト実績
  欠点: スナップショット非対応、データチェックサム非対応
  用途: デスクトップ、一般サーバー（Ubuntu デフォルト）
```

### 1.3 ext4 のディスク上データ構造

```
ext4 の inode 構造（256バイト、拡張可能）:

  従来の ext2 inode（128バイト）:
  ┌────────────────────────────────┐
  │ i_mode (2)    : ファイルタイプ+権限│
  │ i_uid (2)     : 所有者ID(下位16bit)│
  │ i_size_lo (4) : サイズ(下位32bit)  │
  │ i_atime (4)   : アクセス時刻      │
  │ i_ctime (4)   : 変更時刻          │
  │ i_mtime (4)   : 更新時刻          │
  │ i_dtime (4)   : 削除時刻          │
  │ i_gid (2)     : グループID        │
  │ i_links_count (2): リンク数       │
  │ i_blocks_lo (4): ブロック数       │
  │ i_flags (4)   : フラグ            │
  │ i_block[15] (60): データポインタ  │
  │ ...                               │
  └────────────────────────────────┘

  ext4 拡張フィールド（128バイト追加）:
  ┌────────────────────────────────┐
  │ i_extra_isize  : 拡張サイズ    │
  │ i_checksum_hi  : チェックサム  │
  │ i_ctime_extra  : ナノ秒精度   │
  │ i_mtime_extra  : ナノ秒精度   │
  │ i_atime_extra  : ナノ秒精度   │
  │ i_crtime       : 作成時刻     │
  │ i_crtime_extra : 作成時刻ns   │
  │ i_version_hi   : NFS用バージョン│
  │ i_projid       : プロジェクトID │
  └────────────────────────────────┘

  エクステントツリーのディスク上構造:
  ┌─────────────────────────────────────┐
  │ ext4_extent_header:                  │
  │   eh_magic  = 0xF30A                │
  │   eh_entries: エントリ数             │
  │   eh_max:    最大エントリ数          │
  │   eh_depth:  ツリーの深さ            │
  │   eh_generation: 世代番号            │
  │                                      │
  │ depth=0 の場合: ext4_extent[4]      │
  │   ee_block:  論理ブロック番号        │
  │   ee_len:    ブロック数              │
  │   ee_start:  物理ブロック番号        │
  │                                      │
  │ depth>0 の場合: ext4_extent_idx[4]  │
  │   ei_block:  論理ブロック番号        │
  │   ei_leaf:   子ノードのブロック      │
  └─────────────────────────────────────┘
```

### 1.4 ext4 のチューニング

```bash
# ext4 ファイルシステムの作成（カスタム設定）
# 一般的なサーバー向け
mkfs.ext4 -L "data" \
  -b 4096 \              # ブロックサイズ 4KB
  -i 16384 \             # bytes-per-inode（inode密度）
  -J size=256 \          # ジャーナルサイズ 256MB
  -O metadata_csum \     # メタデータチェックサム有効
  -E lazy_itable_init=1 \  # 遅延 inode テーブル初期化
  /dev/sda1

# メールサーバー向け（小ファイル多数）
mkfs.ext4 -L "mail" \
  -b 4096 \
  -i 4096 \              # inode を多く確保
  -J size=128 \
  /dev/sda1

# 大ファイル向け（メディアサーバー）
mkfs.ext4 -L "media" \
  -b 4096 \
  -i 1048576 \           # inode を少なく（大ファイル前提）
  -T largefile4 \        # 大ファイルプリセット
  /dev/sda1

# マウントオプションの最適化
# /etc/fstab
# 一般用途（推奨設定）
/dev/sda1 / ext4 defaults,noatime,commit=60 0 1

# データベース用途
/dev/sda1 /var/lib/mysql ext4 defaults,noatime,data=ordered,barrier=1,commit=5 0 2

# 一時ファイル/ビルド用途
/dev/sda1 /tmp ext4 defaults,noatime,data=writeback,barrier=0,commit=120 0 0

# SSD 向け
/dev/sda1 / ext4 defaults,noatime,discard,commit=60 0 1
```

```bash
# 運用中の ext4 チューニング

# ジャーナリングモードの確認・変更
cat /proc/fs/ext4/sda1/options | grep data
# data=ordered

# コミット間隔の調整（デフォルト5秒）
# 長くすると性能向上、短くすると安全性向上
sudo tune2fs -o commit=30 /dev/sda1

# リザーブドブロックの調整（デフォルト5%）
# 大容量ディスクでは減らしてもよい
sudo tune2fs -m 1 /dev/sda1       # 1% に削減
sudo tune2fs -l /dev/sda1 | grep "Reserved"

# ファイルシステムラベルの設定
sudo e2label /dev/sda1 "my-data"

# チェック間隔の設定
sudo tune2fs -c 50 /dev/sda1      # 50回マウントごとにチェック
sudo tune2fs -i 6m /dev/sda1      # 6ヶ月ごとにチェック

# メタデータチェックサムの有効化（ext4 >= 3.18）
sudo tune2fs -O metadata_csum /dev/sda1

# 断片化の確認
sudo e4defrag -c /mount/point
# Fragmentation score: 0-30=低, 30-55=中, 55-100=高
```

---

## 2. XFS

### 2.1 XFS の概要と歴史

```
XFS (SGI, 1993 → Linux):
  Silicon Graphics が IRIX 向けに開発
  → 2001年に Linux にポーティング
  → 2014年に RHEL 7 のデフォルトに採用
  → 現在最も広く使われる Linux FS の一つ

  設計思想:
  - 大規模ストレージ向け（元はスーパーコンピュータ向け）
  - 高い並列I/O性能
  - スケーラビリティ重視
  - 64bit ネイティブ設計（当初から）

  主な特徴:
  ┌──────────────────────────────────────────────────┐
  │ 容量制限                                          │
  │ - 最大ボリューム: 8EB (64bit)                     │
  │ - 最大ファイル: 8EB                               │
  │ - 最大ファイル名: 255バイト                        │
  ├──────────────────────────────────────────────────┤
  │ B+木ベースのメタデータ管理                         │
  │ - 全メタデータがB+木で管理                         │
  │ - inode割り当て、空きブロック管理、ディレクトリ    │
  │ - 効率的な検索・更新                               │
  ├──────────────────────────────────────────────────┤
  │ アロケーショングループ (AG)                         │
  │ - ファイルシステムを独立した領域に分割              │
  │ - 各AG が独立したメタデータを持つ                  │
  │ - 並列アクセスが可能（異なるAGへの同時書き込み）   │
  │ - ロック競合の削減                                 │
  ├──────────────────────────────────────────────────┤
  │ 遅延アロケーション                                 │
  │ - ext4 と同様、書き込み時にブロック割り当てを遅延  │
  │ - 連続ブロック確保の最適化                         │
  ├──────────────────────────────────────────────────┤
  │ ジャーナリング                                     │
  │ - メタデータのみジャーナリング                     │
  │ - 外部ログデバイスのサポート                       │
  │ - 非同期の遅延ロギング（delayed logging）          │
  ├──────────────────────────────────────────────────┤
  │ オンライン操作                                     │
  │ - オンラインリサイズ（拡張のみ、縮小不可）         │
  │ - オンラインデフラグ (xfs_fsr)                     │
  │ - オンラインダンプ/リストア (xfsdump/xfsrestore)  │
  └──────────────────────────────────────────────────┘
```

### 2.2 XFS の内部構造

```
XFS のディスクレイアウト:

  ┌──────────────────────────────────────────────┐
  │ ファイルシステム全体                          │
  ├──────────┬──────────┬──────────┬──────────────┤
  │   AG 0   │   AG 1   │   AG 2   │    AG 3     │
  └──────────┴──────────┴──────────┴──────────────┘

  各AG（Allocation Group）の構造:
  ┌──────────────────────────────────────────────┐
  │ AG ヘッダ領域:                                │
  │ ┌──────┬──────┬──────┬──────┬──────┐         │
  │ │ AGF  │ AGI  │ AGFL │Free  │inode │         │
  │ │      │      │      │Space │B+Tree│         │
  │ │      │      │      │B+Tree│      │         │
  │ └──────┴──────┴──────┴──────┴──────┘         │
  │                                              │
  │ データ領域:                                   │
  │ ┌────────────────────────────────────┐        │
  │ │ inode チャンク + データブロック      │        │
  │ └────────────────────────────────────┘        │
  └──────────────────────────────────────────────┘

  AGF (AG Free Space): 空きブロック管理
  AGI (AG Inode):      inode 管理
  AGFL (AG Free List): AGF/AGI のB+木ブロック管理

  B+木の活用:
  ┌────────────────────────────────────────────┐
  │ XFS における B+木の使用箇所                  │
  ├────────────────────────────────────────────┤
  │ 1. 空きブロック管理（ブロック番号順）       │
  │ 2. 空きブロック管理（サイズ順）             │
  │ 3. inode 割り当て管理                       │
  │ 4. ディレクトリエントリ                     │
  │ 5. エクステントマップ（ファイルデータ）     │
  │ 6. リバースマッピング（reflink用）          │
  │ 7. 参照カウント（reflink用）                │
  └────────────────────────────────────────────┘
```

### 2.3 XFS のチューニングと運用

```bash
# XFS ファイルシステムの作成
mkfs.xfs -L "xfs-data" \
  -b size=4096 \         # ブロックサイズ
  -d agcount=16 \        # AG数（並列性の調整）
  -l size=256m \         # ログサイズ
  /dev/sda1

# 外部ログデバイスの使用（性能向上）
mkfs.xfs -l logdev=/dev/sdb1,size=256m /dev/sda1
mount -o logdev=/dev/sdb1 /dev/sda1 /mnt

# XFS の情報確認
xfs_info /mount/point
# meta-data=/dev/sda1  isize=512  agcount=16, agsize=...
# data     =           bsize=4096 blocks=...
# naming   =version 2  bsize=4096 ascii-ci=0
# log      =internal   bsize=4096 blocks=...

# マウントオプション
# /etc/fstab
/dev/sda1 /data xfs defaults,noatime,inode64,logbufs=8 0 0

# パフォーマンス関連オプション:
# logbufs=N     : ログバッファ数（2-8、デフォルト8）
# logbsize=N    : ログバッファサイズ（32K-256K）
# nobarrier     : 書き込みバリア無効化（BBU付きRAID向け）
# allocsize=N   : 先読みアロケーションサイズ
# inode64       : 64bit inode番号（大容量FS必須）

# オンラインリサイズ（拡張のみ）
xfs_growfs /mount/point

# 注意: XFS は縮小できない
# 縮小が必要な場合: バックアップ → 再作成 → リストア

# デフラグ
xfs_fsr /mount/point          # ファイルシステム全体
xfs_fsr /path/to/file         # 特定ファイル
xfs_fsr -v /mount/point       # 詳細表示

# バックアップとリストア（XFS固有ツール）
xfsdump -l 0 -f /backup/dump.xfsdump /mount/point
xfsrestore -f /backup/dump.xfsdump /restore/point

# 修復
xfs_repair /dev/sda1           # アンマウント状態で
xfs_repair -L /dev/sda1        # ログをゼロクリア（最終手段）

# XFS のメタデータダンプ（デバッグ用）
xfs_db -r /dev/sda1
xfs_db> sb 0                   # スーパーブロック表示
xfs_db> freesp                 # 空き領域分布
```

### 2.4 XFS の reflink とコピー

```
reflink（参照リンク）:
  XFS 4.9+ で対応。ファイルのコピーを瞬時に行う仕組み

  通常のコピー:
  cp source dest → 全データブロックをコピー（時間 ∝ ファイルサイズ）

  reflink コピー:
  cp --reflink source dest → メタデータのみコピー（瞬時）
  → データブロックを共有
  → どちらかが変更されたら、変更部分だけ新しいブロックに書き込み（CoW）

  仕組み:
  ┌────────────────────────────────────────┐
  │ reflink 前:                            │
  │ source → [Block A] [Block B] [Block C]│
  │                                        │
  │ reflink 後:                            │
  │ source → [Block A] [Block B] [Block C]│
  │ dest   →↗          ↗          ↗       │
  │ （参照カウント = 2）                    │
  │                                        │
  │ dest の Block B を変更:                │
  │ source → [Block A] [Block B] [Block C]│
  │ dest   →↗         [Block B'] ↗        │
  │ Block A, C: 参照カウント = 2           │
  │ Block B:    参照カウント = 1           │
  │ Block B':   参照カウント = 1（新規）   │
  └────────────────────────────────────────┘

  使用例:
  # reflink コピーの作成
  $ cp --reflink=auto source.img dest.img   # 可能なら reflink
  $ cp --reflink=always source.img dest.img # reflink 必須（不可ならエラー）

  # 仮想マシンイメージの高速クローン
  $ cp --reflink=always base.qcow2 vm1.qcow2  # 瞬時
  $ cp --reflink=always base.qcow2 vm2.qcow2  # 瞬時

  # reflink が有効か確認
  $ xfs_info /mount/point | grep reflink
  # reflink=1 が表示されれば有効

  # mkfs 時に reflink を有効化（デフォルトで有効、XFS 5.1+）
  $ mkfs.xfs -m reflink=1 /dev/sda1
```

---

## 3. Btrfs

### 3.1 Btrfs の概要

```
Btrfs (B-tree File System, Oracle → Linux, 2009):
  「バターFS」と読む
  → ZFS の Linux 版を目指して開発
  → SUSE Linux Enterprise のデフォルト（15 SP1以降）
  → Fedora 33 以降のデフォルト（ワークステーション版）

  設計思想:
  - Copy-on-Write (CoW) ベース
  - 全データ・メタデータにチェックサム
  - 柔軟なストレージ管理
  - スナップショットとクローンの効率的なサポート
  - エンタープライズ機能を Linux ネイティブに

  主な特徴:
  ┌──────────────────────────────────────────────────┐
  │ Copy-on-Write                                     │
  │ - データを上書きせず新しい場所に書き込み           │
  │ - クラッシュ一貫性の構造的保証                     │
  │ - ジャーナル不要                                   │
  ├──────────────────────────────────────────────────┤
  │ スナップショット                                   │
  │ - サブボリュームの瞬時バックアップ                 │
  │ - 読み取り専用 / 読み書き可能                      │
  │ - 増分バックアップ（send/receive）                 │
  ├──────────────────────────────────────────────────┤
  │ 透過的圧縮                                        │
  │ - zstd（推奨）, lzo, zlib                         │
  │ - ファイル/ディレクトリ単位で設定可能              │
  │ - 読み書き時に自動で圧縮/展開                      │
  ├──────────────────────────────────────────────────┤
  │ データ・メタデータチェックサム                      │
  │ - CRC32C（デフォルト）, xxhash, sha256, blake2b   │
  │ - サイレントデータ破損（ビット腐敗）の検出         │
  │ - RAID と組み合わせて自動修復                      │
  ├──────────────────────────────────────────────────┤
  │ 内蔵RAID                                          │
  │ - RAID 0, 1, 10: 安定                             │
  │ - RAID 5, 6: write hole 問題あり（実験的）         │
  │ - プロファイルの動的変更可能                       │
  ├──────────────────────────────────────────────────┤
  │ サブボリューム                                     │
  │ - FS内の独立した名前空間                           │
  │ - パーティション分割の代替                         │
  │ - 個別にマウント可能                               │
  │ - 個別にスナップショット可能                       │
  ├──────────────────────────────────────────────────┤
  │ オンライン操作                                     │
  │ - オンラインリサイズ（拡張・縮小可能）             │
  │ - オンラインデフラグ                               │
  │ - オンラインスクラブ（データ整合性チェック）       │
  │ - オンラインバランス（データ再配置）               │
  ├──────────────────────────────────────────────────┤
  │ デデュプリケーション                               │
  │ - オフライン: duperemove ツール                    │
  │ - reflink ベースの効率的な重複排除                 │
  └──────────────────────────────────────────────────┘
```

### 3.2 Btrfs の内部構造

```
Btrfs のディスクレイアウト:

  全てがB木（B-tree）で管理される:

  ┌────────────────────────────────────────────┐
  │ Superblock (3つのコピー: 64KB, 64MB, 256GB)│
  │                                            │
  │ ┌──────────────────────────────────┐       │
  │ │ Tree of Trees (Root Tree)        │       │
  │ │ → 全B木のルートを管理             │       │
  │ └──────┬───────┬────────┬──────────┘       │
  │        ↓       ↓        ↓                  │
  │  ┌────────┐┌────────┐┌────────┐            │
  │  │FS Tree ││Extent  ││Checksum│            │
  │  │(ファイル││ Tree   ││ Tree   │            │
  │  │ +ディレ ││(ブロック││(チェック│            │
  │  │ クトリ) ││ 管理)  ││ サム)  │            │
  │  └────────┘└────────┘└────────┘            │
  │  ┌────────┐┌────────┐┌────────┐            │
  │  │Chunk   ││Device  ││UUID    │            │
  │  │ Tree   ││ Tree   ││ Tree   │            │
  │  │(論理→  ││(デバイス││(UUID   │            │
  │  │ 物理)  ││ 管理)  ││ 管理)  │            │
  │  └────────┘└────────┘└────────┘            │
  └────────────────────────────────────────────┘

  CoW による更新:
  ┌─────────────────────────────────────────┐
  │ 更新前:                                  │
  │ Root → A → [D] [E] [F]                 │
  │          → [G] [H]                      │
  │                                          │
  │ [E] を変更:                              │
  │ 1. E のコピーを E' として新しい場所に作成 │
  │ 2. A のコピーを A' として作成（E'を指す）│
  │ 3. Root を A' に更新（アトミック）       │
  │                                          │
  │ 更新後:                                  │
  │ Root' → A' → [D] [E'] [F]              │
  │            → [G] [H]                    │
  │                                          │
  │ 旧 Root, A, E は解放可能                │
  │ （スナップショットが参照中なら保持）     │
  └─────────────────────────────────────────┘
```

### 3.3 Btrfs のサブボリュームとスナップショット

```bash
# サブボリュームの管理
# サブボリューム作成
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@var
btrfs subvolume create /mnt/@snapshots

# サブボリューム一覧
btrfs subvolume list /mnt
# ID 256 gen 100 top level 5 path @home
# ID 257 gen 101 top level 5 path @var
# ID 258 gen 102 top level 5 path @snapshots

# サブボリュームの個別マウント
# /etc/fstab
/dev/sda1  /home       btrfs  subvol=@home,defaults,noatime,compress=zstd  0  0
/dev/sda1  /var        btrfs  subvol=@var,defaults,noatime                  0  0

# スナップショットの作成
# 読み取り専用スナップショット
btrfs subvolume snapshot -r /mnt/@home /mnt/@snapshots/home-$(date +%Y%m%d)

# 読み書き可能スナップショット
btrfs subvolume snapshot /mnt/@home /mnt/@snapshots/home-writable

# スナップショットからのファイル復旧
cp /mnt/@snapshots/home-20240101/user/important.txt /home/user/

# スナップショットへのロールバック（読み書きスナップショットの場合）
# 現在のサブボリュームを削除して、スナップショットをリネーム
btrfs subvolume delete /mnt/@home
btrfs subvolume snapshot /mnt/@snapshots/home-20240101 /mnt/@home

# スナップショットの削除
btrfs subvolume delete /mnt/@snapshots/home-20240101

# 増分バックアップ（send/receive）
# 初回: フルバックアップ
btrfs subvolume snapshot -r /mnt/@home /mnt/@snapshots/snap1
btrfs send /mnt/@snapshots/snap1 | btrfs receive /backup/

# 2回目以降: 増分バックアップ
btrfs subvolume snapshot -r /mnt/@home /mnt/@snapshots/snap2
btrfs send -p /mnt/@snapshots/snap1 /mnt/@snapshots/snap2 | btrfs receive /backup/
# → snap1 と snap2 の差分のみ転送

# SSH 越しの増分バックアップ
btrfs send -p /mnt/@snapshots/snap1 /mnt/@snapshots/snap2 | \
  ssh backup-server "btrfs receive /backup/"
```

### 3.4 Btrfs の圧縮と RAID

```bash
# 透過的圧縮の設定

# マウント時に圧縮を有効化
mount -o compress=zstd /dev/sda1 /mnt
mount -o compress=zstd:3 /dev/sda1 /mnt   # 圧縮レベル指定

# 圧縮アルゴリズムの比較:
# ┌─────────┬──────────┬──────────┬──────────┐
# │ アルゴリズム│ 圧縮率  │ 圧縮速度 │ 展開速度 │
# ├─────────┼──────────┼──────────┼──────────┤
# │ lzo     │ 低       │ 最速     │ 最速     │
# │ zstd:1  │ 中       │ 高速     │ 高速     │
# │ zstd:3  │ 中-高    │ 中速     │ 高速     │ ← 推奨
# │ zlib:6  │ 高       │ 低速     │ 中速     │
# │ zstd:15 │ 最高     │ 最低     │ 高速     │
# └─────────┴──────────┴──────────┴──────────┘

# 特定ディレクトリの圧縮を無効化
btrfs property set /mnt/database compression ""
chattr +m /mnt/database    # 圧縮無効フラグ

# 既存データの再圧縮
btrfs filesystem defragment -r -czstd /mnt/

# 圧縮統計の確認
btrfs filesystem df /mnt
compsize /mnt                # 圧縮率の詳細確認ツール

# RAID 構成

# RAID1（ミラーリング）
mkfs.btrfs -d raid1 -m raid1 /dev/sda1 /dev/sdb1

# RAID10（ストライプ+ミラー）
mkfs.btrfs -d raid10 -m raid10 /dev/sd{a,b,c,d}1

# RAID プロファイルの動的変更
# single → RAID1 に変換
btrfs balance start -dconvert=raid1 -mconvert=raid1 /mnt

# デバイスの追加
btrfs device add /dev/sdc1 /mnt
btrfs balance start /mnt      # データを再配置

# デバイスの削除
btrfs device delete /dev/sda1 /mnt

# デバイスの交換
btrfs replace start /dev/sda1 /dev/sdd1 /mnt

# RAID 情報の確認
btrfs filesystem show /mnt
btrfs filesystem df /mnt
btrfs filesystem usage /mnt
```

### 3.5 Btrfs の運用とメンテナンス

```bash
# スクラブ（データ整合性チェック）
# → 全データのチェックサムを検証
# → RAID の場合は自動修復
btrfs scrub start /mnt
btrfs scrub status /mnt

# 定期スクラブの設定（systemd timer）
# /etc/systemd/system/btrfs-scrub.timer
# [Timer]
# OnCalendar=monthly
# [Install]
# WantedBy=timers.target

# バランス（データ再配置）
btrfs balance start /mnt
btrfs balance status /mnt
btrfs balance cancel /mnt      # 中断

# 使用量の詳細確認
btrfs filesystem usage /mnt
btrfs filesystem df /mnt
btrfs filesystem show

# クォータの設定
btrfs quota enable /mnt
btrfs qgroup limit 50G /mnt/@home
btrfs qgroup show /mnt

# ファイルシステムの修復
# 通常修復
btrfs check /dev/sda1

# 修復実行（注意: 危険な操作）
btrfs check --repair /dev/sda1

# rescue モード
btrfs rescue super-recover /dev/sda1
btrfs rescue zero-log /dev/sda1
btrfs rescue chunk-recover /dev/sda1
```

---

## 4. ZFS

### 4.1 ZFS の概要

```
ZFS (Zettabyte File System, Sun Microsystems, 2005):
  「最後のファイルシステム」として設計
  → 2005年に OpenSolaris で公開
  → FreeBSD に移植（ネイティブサポート）
  → Linux では ZFS on Linux (OpenZFS) として利用可能
  → ライセンス問題（CDDL vs GPL）でカーネル統合不可

  設計思想:
  - ストレージスタック全体を統合管理
  - ボリュームマネージャ + ファイルシステム + RAID
  - エンタープライズグレードのデータ保護
  - 管理の簡素化（「設定して忘れる」）

  主な特徴:
  ┌──────────────────────────────────────────────────┐
  │ 128bit アドレッシング                              │
  │ - 事実上無限の容量（256兆ヨビバイト）              │
  │ - 全データにチェックサム（SHA-256/Fletcher4）       │
  │ - メタデータの3重コピー                            │
  ├──────────────────────────────────────────────────┤
  │ プール型ストレージ管理                             │
  │ - zpool: 物理デバイスのプール                      │
  │ - zfs dataset: プールから切り出す論理ボリューム    │
  │ - パーティション不要、動的なサイズ変更              │
  ├──────────────────────────────────────────────────┤
  │ RAID-Z（RAID5/6 の改良版）                         │
  │ - RAID-Z1: 1台のディスク障害に耐える               │
  │ - RAID-Z2: 2台のディスク障害に耐える               │
  │ - RAID-Z3: 3台のディスク障害に耐える               │
  │ - Write Hole 問題なし（CoW ベース）                │
  ├──────────────────────────────────────────────────┤
  │ ARC (Adaptive Replacement Cache)                   │
  │ - 高度なキャッシュアルゴリズム                     │
  │ - MRU（最近使用）+ MFU（頻繁使用）の適応的組合せ  │
  │ - L2ARC: SSD をキャッシュデバイスとして使用        │
  ├──────────────────────────────────────────────────┤
  │ ZIL (ZFS Intent Log)                               │
  │ - 同期書き込みの高速化                             │
  │ - SLOG: 別デバイス（SSD）にZILを配置              │
  ├──────────────────────────────────────────────────┤
  │ スナップショットとクローン                          │
  │ - 瞬時のスナップショット作成                       │
  │ - 読み書き可能なクローン                           │
  │ - send/receive による効率的なレプリケーション       │
  ├──────────────────────────────────────────────────┤
  │ デデュプリケーション（重複排除）                    │
  │ - ブロックレベルの重複排除                         │
  │ - DDT (Dedup Table) をメモリに保持                 │
  │ - 大量のメモリが必要（1TB あたり約5GB RAM）        │
  ├──────────────────────────────────────────────────┤
  │ 圧縮                                              │
  │ - LZ4（デフォルト、推奨）, gzip, zstd, lzjb       │
  │ - 透過的圧縮                                      │
  │ - 圧縮+デデュプリケーションの組合せ可能            │
  └──────────────────────────────────────────────────┘
```

### 4.2 ZFS の基本操作

```bash
# ZFS のインストール（Ubuntu）
sudo apt install zfsutils-linux

# プール（zpool）の作成

# シンプルなプール（ストライプ、冗長性なし）
sudo zpool create tank /dev/sdb

# ミラープール（RAID1 相当）
sudo zpool create tank mirror /dev/sdb /dev/sdc

# RAID-Z1 プール（RAID5 相当）
sudo zpool create tank raidz1 /dev/sdb /dev/sdc /dev/sdd

# RAID-Z2 プール（RAID6 相当）
sudo zpool create tank raidz2 /dev/sd{b,c,d,e}

# キャッシュとログ付きプール
sudo zpool create tank raidz1 /dev/sd{b,c,d} \
  cache /dev/sde \      # L2ARC（読み取りキャッシュ用SSD）
  log mirror /dev/sdf /dev/sdg  # SLOG（書き込みログ用SSD、ミラー）

# プール情報の確認
zpool status tank
zpool list
zpool iostat -v tank 5     # 5秒ごとのI/O統計

# データセットの作成と管理
sudo zfs create tank/home
sudo zfs create tank/data
sudo zfs create tank/backup

# プロパティの設定
sudo zfs set compression=lz4 tank         # 圧縮有効化
sudo zfs set atime=off tank               # atime 無効化
sudo zfs set recordsize=1M tank/media     # 大ファイル向け
sudo zfs set quota=100G tank/home         # クォータ設定
sudo zfs set reservation=50G tank/data    # 予約領域

# プロパティの確認
zfs get all tank/home
zfs get compression,compressratio tank
zfs list -o name,used,avail,refer,mountpoint
```

### 4.3 ZFS のスナップショットとレプリケーション

```bash
# スナップショットの作成
sudo zfs snapshot tank/home@daily-$(date +%Y%m%d)

# 再帰的スナップショット（全子データセット含む）
sudo zfs snapshot -r tank@backup-$(date +%Y%m%d)

# スナップショット一覧
zfs list -t snapshot

# スナップショットからのファイル復旧
# スナップショットは .zfs/snapshot/ からアクセス可能
ls /tank/home/.zfs/snapshot/daily-20240101/
cp /tank/home/.zfs/snapshot/daily-20240101/file.txt /tank/home/

# スナップショットへのロールバック
sudo zfs rollback tank/home@daily-20240101

# スナップショットの削除
sudo zfs destroy tank/home@daily-20240101

# 古いスナップショットの一括削除
sudo zfs destroy tank/home@%daily-202301  # 2023年1月分を削除

# レプリケーション（send/receive）
# 初回: フルバックアップ
sudo zfs send tank/home@snap1 | sudo zfs receive backup/home

# 増分バックアップ
sudo zfs send -i tank/home@snap1 tank/home@snap2 | \
  sudo zfs receive backup/home

# SSH 越しのレプリケーション
sudo zfs send -i tank/home@snap1 tank/home@snap2 | \
  ssh backup-server "sudo zfs receive backup/home"

# 暗号化付き送信（raw send）
sudo zfs send --raw -i tank/home@snap1 tank/home@snap2 | \
  ssh backup-server "sudo zfs receive backup/home"

# 自動スナップショット（zfs-auto-snapshot, sanoid/syncoid）
# sanoid.conf の例:
# [tank/home]
#   use_template = production
#   autosnap = yes
#   autoprune = yes
# [template_production]
#   hourly = 24
#   daily = 30
#   monthly = 12
#   yearly = 5
```

### 4.4 ZFS のパフォーマンスチューニング

```bash
# ARC（キャッシュ）の確認と調整
cat /proc/spl/kstat/zfs/arcstats | grep -E "^(size|c_max|hits|misses)"
# ARC サイズの上限設定（/etc/modprobe.d/zfs.conf）
# options zfs zfs_arc_max=8589934592   # 8GB

# ZIL（書き込みログ）の無効化（非推奨、テスト用）
# sync=disabled は同期書き込みを無効化
sudo zfs set sync=disabled tank/tmp

# レコードサイズの最適化
# データベース（小ブロックI/O）
sudo zfs set recordsize=8K tank/database

# 大ファイル（メディア、バックアップ）
sudo zfs set recordsize=1M tank/media

# 一般用途
sudo zfs set recordsize=128K tank/home    # デフォルト

# 圧縮の効果確認
zfs get compressratio tank
# NAME  PROPERTY       VALUE  SOURCE
# tank  compressratio  2.50x  -
# → 2.5倍の圧縮率 = 40%のディスク節約

# スクラブ（データ整合性チェック）
sudo zpool scrub tank
zpool status tank      # スクラブの進捗確認

# I/O パフォーマンスの確認
zpool iostat -v 5      # 5秒ごと

# デバイスの交換
sudo zpool replace tank /dev/sdb /dev/sdh

# デバイスの追加（ミラーの追加）
sudo zpool attach tank /dev/sdb /dev/sdc  # sdb のミラーとして sdc を追加
```

---

## 5. その他のFS

### 5.1 NTFS（Windows）

```
NTFS (New Technology File System, 1993):
  Windows NT 以降の標準ファイルシステム

  主な特徴:
  - MFT (Master File Table): 全ファイルのメタデータ管理
  - ジャーナリング: USN (Update Sequence Number) Journal
  - ACL: Windows 固有のアクセス制御リスト
  - 暗号化: EFS (Encrypting File System)
  - 圧縮: NTFS 圧縮（ファイル/ディレクトリ単位）
  - ハードリンク、シンボリックリンク（Vista以降）
  - 代替データストリーム (ADS): ファイルに複数のデータストリーム
  - クォータ: ユーザごとのディスク使用量制限
  - VSS (Volume Shadow Copy Service): スナップショット

  容量:
  - 最大ボリューム: 256TB（理論上16EB）
  - 最大ファイル: 256TB
  - クラスタサイズ: 4KB（デフォルト）

  Linux からのアクセス:
  # カーネルドライバ（読み取り専用）
  mount -t ntfs /dev/sda1 /mnt

  # NTFS-3G（読み書き可能、FUSEベース）
  mount -t ntfs-3g /dev/sda1 /mnt

  # ntfs3（Linux 5.15+、カーネル内蔵、読み書き可能）
  mount -t ntfs3 /dev/sda1 /mnt

  MFT の構造:
  ┌────────────────────────────────────────┐
  │ MFT エントリ（通常 1KB）               │
  │ ┌─────────────────────────────────────┐│
  │ │ $STANDARD_INFORMATION              ││
  │ │  → タイムスタンプ、フラグ            ││
  │ ├─────────────────────────────────────┤│
  │ │ $FILE_NAME                          ││
  │ │  → ファイル名、親ディレクトリ参照    ││
  │ ├─────────────────────────────────────┤│
  │ │ $DATA                              ││
  │ │  → ファイルデータ                    ││
  │ │  → 小さなファイルは MFT 内に格納    ││
  │ │  → 大きなファイルはランを参照        ││
  │ ├─────────────────────────────────────┤│
  │ │ $SECURITY_DESCRIPTOR               ││
  │ │  → ACL 情報                         ││
  │ └─────────────────────────────────────┘│
  └────────────────────────────────────────┘
```

### 5.2 APFS（Apple）

```
APFS (Apple File System, 2017):
  HFS+ の後継として Apple が開発
  → macOS High Sierra (10.13) 以降のデフォルト
  → iOS 10.3 以降で採用
  → watchOS, tvOS でも使用

  設計思想:
  - SSD/フラッシュストレージ最適化
  - 暗号化ファーストの設計
  - コンテナ+ボリュームモデル

  主な特徴:
  - CoW (Copy-on-Write)
  - スナップショット: Time Machine バックアップで使用
  - 暗号化: ボリューム全体 / ファイル単位（FileVault 2）
  - スペース共有: 複数ボリュームでコンテナの容量を共有
  - クローン: ファイル/ディレクトリの瞬時コピー
  - ナノ秒タイムスタンプ（HFS+ は秒精度）
  - クラッシュプロテクション
  - TRIM サポート（SSD 最適化）

  コンテナモデル:
  ┌────────────────────────────────────────┐
  │ APFS コンテナ（= 物理パーティション）   │
  │ ┌──────────┐┌──────────┐┌──────────┐  │
  │ │ Volume 1 ││ Volume 2 ││ Volume 3 │  │
  │ │ (macOS)  ││ (Data)   ││ (VM)     │  │
  │ │          ││          ││          │  │
  │ └──────────┘└──────────┘└──────────┘  │
  │   ← 空き容量をボリューム間で共有 →      │
  └────────────────────────────────────────┘

  macOS Catalina 以降のボリューム構成:
  - Macintosh HD (System): 読み取り専用のシステムボリューム
  - Macintosh HD - Data: ユーザデータ
  - Preboot, Recovery, VM: システム用

  diskutil での確認:
  $ diskutil list
  $ diskutil apfs list
  $ diskutil info /
```

### 5.3 FAT32 / exFAT

```
FAT32 (File Allocation Table, 1996):
  最も互換性が高いファイルシステム

  特徴:
  - 全OS（Windows, macOS, Linux, 組込み）で読み書き可能
  - 最大ファイル: 4GB（最大の制約）
  - 最大ボリューム: 2TB（理論上8TB）
  - ジャーナリングなし
  - ACL なし（基本的なパーミッションのみ）

  構造:
  ┌──────┬──────────┬───────────┬──────────┐
  │ Boot │ FAT 1    │ FAT 2     │ Data     │
  │ Sect │(ファイル ││(バックアップ)│ 領域     │
  │      │割当テーブル)│          │          │
  └──────┴──────────┴───────────┴──────────┘

  FAT（File Allocation Table）:
  各クラスタの次のクラスタ番号を記録するリンクリスト
  ┌──────────────────────────────┐
  │ クラスタ 2: → 3             │
  │ クラスタ 3: → 7             │
  │ クラスタ 4: → EOF（空き）    │
  │ クラスタ 5: → 0（空き）     │
  │ クラスタ 6: → EOF           │
  │ クラスタ 7: → EOF           │
  └──────────────────────────────┘
  ファイルA: 2 → 3 → 7 → EOF

  用途: SDカード、USBメモリ（小容量）、組込みシステム

exFAT (Extended FAT, 2006):
  FAT32 の後継。大ファイル対応

  特徴:
  - 最大ファイル: 16EB（事実上無制限）
  - 最大ボリューム: 128PB
  - FAT32 の 4GB 制限を解消
  - Microsoft がパテント公開（2019年）
  - Linux カーネル 5.4 以降でネイティブサポート
  - ジャーナリングなし
  - ACL なし

  用途:
  - SDカード（SDXC 規格の公式FS）
  - USBメモリ（大容量ファイル）
  - Windows/Mac 共用の外付けHDD
  - デジタルカメラのメモリカード

  作成:
  # FAT32
  $ mkfs.vfat -F 32 /dev/sdb1
  # exFAT
  $ mkfs.exfat /dev/sdb1
```

### 5.4 特殊な仮想ファイルシステム

```
tmpfs:
  RAM上のファイルシステム
  - マウントポイント: /tmp, /dev/shm, /run
  - 超高速だが再起動で消失
  - スワップにも書き出される
  - ビルドキャッシュ、一時ファイルに最適

  設定:
  # /etc/fstab
  tmpfs  /tmp      tmpfs  defaults,size=4G,noatime  0  0
  tmpfs  /dev/shm  tmpfs  defaults,size=2G          0  0

procfs:
  プロセスとカーネルの情報を公開する仮想FS
  マウントポイント: /proc

  重要なファイル:
  /proc/cpuinfo        : CPU 情報
  /proc/meminfo        : メモリ情報
  /proc/loadavg        : 負荷平均
  /proc/version        : カーネルバージョン
  /proc/<pid>/status   : プロセスの状態
  /proc/<pid>/maps     : メモリマッピング
  /proc/<pid>/fd/      : ファイルディスクリプタ
  /proc/sys/           : カーネルパラメータ（sysctl）

sysfs:
  デバイスとドライバの情報をツリー構造で公開
  マウントポイント: /sys

  構造:
  /sys/class/           : デバイスクラス
  /sys/block/           : ブロックデバイス
  /sys/devices/         : デバイスの物理階層
  /sys/bus/             : バスタイプ
  /sys/module/          : ロードされたモジュール
  /sys/fs/              : ファイルシステム固有の情報

debugfs:
  デバッグ情報の公開
  マウントポイント: /sys/kernel/debug
  → ftrace, tracing 等のデバッグツールが使用

devtmpfs:
  デバイスファイルの自動作成
  マウントポイント: /dev
  → udev と連携してデバイスノードを自動管理

cgroupfs:
  コントロールグループの管理
  → CPU、メモリ、I/O のリソース制限
  → コンテナ（Docker, systemd）の基盤技術

overlayfs:
  複数のディレクトリを重ね合わせる
  → Docker のイメージレイヤーの実装
  → lower（読み取り専用）+ upper（読み書き）= merged（統合ビュー）
```

### 5.5 ネットワークファイルシステム

```
NFS (Network File System):
  Unix/Linux 標準のネットワーク共有
  - NFSv3: ステートレス（復旧が容易）
  - NFSv4: ステートフル（ロック機能改善、ファイアウォール通過）
  - NFSv4.1: pNFS（並列NFS、分散ストレージ対応）
  - NFSv4.2: サーバーサイドコピー、ホール検出

  # サーバー設定（/etc/exports）
  /data  192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)

  # クライアントからのマウント
  mount -t nfs server:/data /mnt/nfs

SMB/CIFS (Server Message Block):
  Windows 標準のネットワーク共有
  - Samba を使って Linux サーバーで提供
  - SMB1 (CIFS): 古い、セキュリティ問題あり（使用非推奨）
  - SMB2: Windows Vista 以降
  - SMB3: 暗号化、マルチチャネル（Windows 8 / Server 2012）

  # Linux クライアントからのマウント
  mount -t cifs //server/share /mnt/smb -o user=username

GlusterFS:
  分散ファイルシステム
  → 複数サーバーを束ねて単一の大容量FSを構成
  → レプリケーション、ストライプ、分散

CephFS:
  分散ファイルシステム（Ceph ストレージの一部）
  → POSIX 互換
  → スケーラビリティが高い
  → OpenStack, Kubernetes で利用

FUSE (Filesystem in Userspace):
  ユーザ空間でファイルシステムを実装するフレームワーク
  → SSHFS: SSH 越しのファイルアクセス
  → S3FS: Amazon S3 をファイルシステムとしてマウント
  → NTFS-3G: NTFS の読み書きサポート
  → rclone mount: クラウドストレージのマウント

  # SSHFS の例
  sshfs user@server:/remote/path /mnt/sshfs

  # S3FS の例
  s3fs mybucket /mnt/s3 -o passwd_file=~/.passwd-s3fs
```

---

## 6. ファイルシステムの選択ガイド

### 6.1 用途別推奨

```
用途別ファイルシステム選択:

┌─────────────────────────┬──────────┬────────────────────────┐
│ 用途                     │ 推奨 FS  │ 理由                    │
├─────────────────────────┼──────────┼────────────────────────┤
│ デスクトップ（一般）     │ ext4     │ 安定、互換性、ツール充実 │
│ デスクトップ（Fedora）   │ Btrfs    │ スナップショットで復元   │
│ エンタープライズサーバー │ XFS      │ 高並列、大規模、RHEL標準│
│ データベースサーバー     │ XFS/ext4 │ 高I/O性能、安定性       │
│ NAS（家庭・小規模）     │ Btrfs    │ RAID内蔵、スナップ     │
│ NAS（エンタープライズ） │ ZFS      │ 最強のデータ保護        │
│ USBメモリ（共用）       │ exFAT    │ 全OS互換、大ファイル対応│
│ SDカード（32GB以下）    │ FAT32    │ 最大互換性              │
│ SDカード（64GB以上）    │ exFAT    │ SDXC 標準               │
│ CI/CDビルドディレクトリ │ tmpfs    │ RAM上で超高速           │
│ コンテナ（Docker）      │ overlay2 │ レイヤー管理に最適      │
│ 仮想マシンストレージ     │ XFS+reflink│ 高速クローン           │
│ バックアップサーバー     │ ZFS/Btrfs│ スナップショット+圧縮  │
│ メディアサーバー         │ XFS      │ 大ファイルの高速転送    │
│ 組込みシステム           │ SquashFS+│ 読み取り専用+書き込み領域│
│                         │ overlayfs│                        │
│ ブートパーティション     │ ext4/FAT32│ GRUB/UEFI 互換        │
└─────────────────────────┴──────────┴────────────────────────┘
```

### 6.2 総合比較表

```
主要FS の詳細比較:

┌──────────────┬──────────┬──────────┬──────────┬──────────┐
│ 項目          │ ext4     │ XFS      │ Btrfs    │ ZFS      │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ 最大FS       │ 1EB      │ 8EB      │ 16EB     │ 256ZiB   │
│ 最大ファイル  │ 16TB     │ 8EB      │ 16EB     │ 16EB     │
│ CoW          │ ✗        │ ✗(reflink)│ ✓       │ ✓       │
│ 圧縮         │ ✗        │ ✗        │ ✓        │ ✓       │
│ スナップショット│ ✗      │ ✗        │ ✓        │ ✓       │
│ RAID         │ ✗        │ ✗        │ ✓        │ ✓       │
│ チェックサム  │ meta のみ│ meta のみ│ data+meta│ data+meta│
│ 縮小         │ ✓        │ ✗        │ ✓        │ ✗       │
│ reflink      │ ✗        │ ✓(4.9+)  │ ✓        │ ✗       │
│ デデュプ     │ ✗        │ ✗        │ offline  │ ✓       │
│ 暗号化       │ fscrypt  │ ✗        │ ✗        │ ✓       │
│ クォータ     │ ✓        │ ✓(prj)   │ ✓(qgroup)│ ✓       │
│ 安定性       │ ◎        │ ◎        │ ○        │ ◎       │
│ 速度（seq）  │ ○        │ ◎        │ ○        │ ○       │
│ 速度（rand） │ ○        │ ◎        │ △        │ ○       │
│ メモリ使用   │ 少        │ 少        │ 中       │ 多       │
│ ツール充実度 │ ◎        │ ○        │ ○        │ ○       │
│ Linux統合    │ ◎        │ ◎        │ ◎        │ △(DKMS) │
└──────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 実践演習

### 演習1: [基礎] -- ファイルシステム情報の確認

```bash
# 現在のファイルシステムを確認
df -Th
lsblk -f
cat /etc/fstab

# マウントポイントの詳細情報
findmnt --real                # 物理FSのみ表示
findmnt -t ext4,xfs,btrfs    # 特定タイプのみ

# ファイルシステム固有の情報
# ext4
sudo dumpe2fs -h /dev/sda1
sudo tune2fs -l /dev/sda1

# XFS
xfs_info /mount/point

# Btrfs
btrfs filesystem show
btrfs filesystem df /mount/point
btrfs filesystem usage /mount/point

# ZFS
zpool status
zfs list
```

### 演習2: [応用] -- FS選択判断

```
以下の用途に最適なFSを選択し理由を述べよ:

1. 100TBのNASストレージ（定期バックアップ必須）
   → ZFS: チェックサム、RAID-Z2/Z3、スナップショット、
     send/receive による効率的なバックアップ

2. 高頻度のDBトランザクション（MySQL/PostgreSQL）
   → XFS: 高並列I/O性能、B+木メタデータ管理、
     安定したパフォーマンス、RHEL推奨

3. USBメモリ（Windows/Mac/Linuxで共用）
   → exFAT: 全OS対応、4GB超ファイル対応、
     SDXCカード規格準拠

4. CI/CDの一時ビルドディレクトリ
   → tmpfs: RAM上で最速、再起動で自動クリーン、
     ディスクI/Oボトルネック解消

5. 開発者のLinuxデスクトップ
   → Btrfs: スナップショットで設定変更前の状態保存、
     圧縮でSSD容量節約、Fedoraデフォルト

6. コンテナホストのストレージ
   → XFS + overlay2: Docker推奨構成、
     reflinkでイメージ層の効率的管理
```

### 演習3: [上級] -- ファイルシステムのベンチマーク比較

```bash
# テスト環境の準備（ループバックデバイスで各FSを作成）
for fs in ext4 xfs btrfs; do
  dd if=/dev/zero of=/tmp/test_${fs}.img bs=1M count=2048
  case $fs in
    ext4)  mkfs.ext4 -F /tmp/test_${fs}.img ;;
    xfs)   mkfs.xfs -f /tmp/test_${fs}.img ;;
    btrfs) mkfs.btrfs -f /tmp/test_${fs}.img ;;
  esac
  mkdir -p /mnt/test_${fs}
  mount -o loop /tmp/test_${fs}.img /mnt/test_${fs}
done

# fio でベンチマーク（各FSで実行）
for fs in ext4 xfs btrfs; do
  echo "=== ${fs} ==="
  fio --name=${fs}_seqwrite \
    --directory=/mnt/test_${fs} \
    --rw=write --bs=4k --size=512M \
    --numjobs=4 --runtime=30 --time_based \
    --group_reporting
done

# クリーンアップ
for fs in ext4 xfs btrfs; do
  umount /mnt/test_${fs}
  rm /tmp/test_${fs}.img
done
```

---

## まとめ

| FS | 特徴 | 用途 |
|----|------|------|
| ext4 | 安定、汎用、最も広く使われる | デスクトップ、一般サーバー |
| XFS | 大ファイル、高並列、B+木 | DB、メディア、RHEL |
| Btrfs | CoW、スナップショット、圧縮、RAID | NAS、バックアップ、Fedora/SUSE |
| ZFS | 最強データ保護、プール管理 | エンタープライズNAS、バックアップ |
| NTFS | Windows標準、ACL | Windows環境 |
| APFS | SSD最適化、暗号化 | macOS、iOS |
| exFAT | 全OS互換、大ファイル対応 | USBメモリ、SDカード |
| tmpfs | RAM上、超高速 | 一時ファイル、ビルドキャッシュ |

---

## 次に読むべきガイド
→ [[02-io-scheduling.md]] -- I/Oスケジューリング

---

## 参考文献
1. Carrier, B. "File System Forensic Analysis." Addison-Wesley, 2005.
2. McDougall, R. & Mauro, J. "Solaris Internals." 2nd Ed, Prentice Hall, 2006.
3. Rodeh, O. et al. "BTRFS: The Linux B-Tree Filesystem." ACM TOS, 2013.
4. Bonwick, J. & Moore, B. "ZFS: The Last Word in File Systems." Sun Microsystems, 2007.
5. Sweeney, A. et al. "Scalability in the XFS File System." USENIX ATC, 1996.
6. Mathur, A. et al. "The New ext4 filesystem." Ottawa Linux Symposium, 2007.
7. Lucas, M. W. "FreeBSD Mastery: ZFS." Tilted Windmill Press, 2015.
8. OpenZFS Documentation. https://openzfs.github.io/openzfs-docs/
9. Btrfs Wiki. https://btrfs.wiki.kernel.org/
10. XFS Documentation. https://xfs.wiki.kernel.org/
