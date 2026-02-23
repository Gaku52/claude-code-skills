# ファイルシステムの基礎

> ファイルシステムは「ディスク上のバイト列を、人間が理解できるファイルとディレクトリの階層構造に変換する」仕組み。

## この章で学ぶこと

- [ ] ファイルシステムの基本構造を理解する
- [ ] inodeとディレクトリの仕組みを説明できる
- [ ] ジャーナリングの必要性を知る
- [ ] VFSの仕組みと重要性を理解する
- [ ] ファイルシステムの整合性維持メカニズムを把握する
- [ ] 実務でのファイルシステム操作に習熟する

---

## 1. ファイルシステムの構造

### 1.1 物理構造から論理構造への変換

```
ディスクの物理構造 → ファイルシステムの論理構造:

  物理: セクタ(512B/4KB)の連続
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │S0│S1│S2│S3│S4│S5│S6│S7│S8│S9│...
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

  論理: ファイルとディレクトリの階層
  /
  ├── home/
  │   └── user/
  │       ├── document.txt
  │       └── photo.jpg
  ├── etc/
  │   └── config.yaml
  └── var/
      └── log/
          ├── syslog
          └── auth.log
```

ファイルシステムの根本的な役割は、ディスク上のセクタの羅列を、人間が直感的に扱えるファイルとディレクトリの階層構造に変換することである。この変換は以下の複数のレイヤーで実現される。

```
変換レイヤーの構造:

  ユーザ空間:  open("/home/user/doc.txt", O_RDONLY)
      │
      ↓
  VFS層:       パス名解決 → dentry キャッシュ参照
      │
      ↓
  FS固有層:    ext4_lookup() → inode 取得
      │
      ↓
  ブロック層:  ブロック番号 → セクタ番号に変換
      │
      ↓
  デバイス層:  I/Oリクエスト発行 → ディスクコントローラ
      │
      ↓
  物理層:      ヘッド移動 → セクタ読み取り（HDD）
              NANDフラッシュアクセス（SSD）
```

### 1.2 ブロックとセクタの関係

```
セクタ（Sector）:
  - ディスクの最小物理単位
  - 伝統的に512バイト
  - Advanced Format（AF）ドライブでは4096バイト（4Kn）
  - 512eドライブ: 物理4KB、論理512Bエミュレーション

ブロック（Block）:
  - ファイルシステムの最小論理単位
  - 通常1KB〜4KB（ext4デフォルトは4KB）
  - 1ブロック = N セクタ

  ブロックサイズの選択:
  ┌──────────┬──────────────────────────────────┐
  │ 小さい   │ 内部断片化が少ない                │
  │ ブロック │ メタデータのオーバーヘッドが大きい  │
  │ (1KB)    │ 小さなファイルが多い場合に有利     │
  ├──────────┼──────────────────────────────────┤
  │ 大きい   │ 内部断片化が多い                  │
  │ ブロック │ メタデータのオーバーヘッドが少ない  │
  │ (4KB)    │ 大きなファイルの転送効率が良い     │
  └──────────┴──────────────────────────────────┘

  例: 1バイトのファイルでも1ブロック（4KB）消費
  → 内部断片化（internal fragmentation）

  ブロックサイズの確認:
  $ tune2fs -l /dev/sda1 | grep "Block size"
  Block size:               4096

  $ stat -f / | grep "Block size"  # macOS
```

### 1.3 ext4の基本レイアウト

```
ext4のディスクレイアウト:

  ┌──────┬──────────┬──────────┬──────────┬──────┐
  │Boot  │ Block    │ inode    │ inode    │Data  │
  │Block │ Group    │ Table    │ Bitmap   │Blocks│
  │      │Descriptor│          │ + Block  │      │
  │      │          │          │ Bitmap   │      │
  └──────┴──────────┴──────────┴──────────┴──────┘

  詳細なブロックグループ構造:
  ┌─────────────────────────────────────────────────────┐
  │                    ブロックグループ 0                  │
  ├──────┬──────┬──────┬──────┬──────┬─────────────────┤
  │Super │Group │Block │inode │inode │    Data         │
  │block │Desc. │Bitmap│Bitmap│Table │    Blocks       │
  │(1blk)│(Nblk)│(1blk)│(1blk)│(Nblk)│  (残り全部)    │
  └──────┴──────┴──────┴──────┴──────┴─────────────────┘

  ブロックグループ 1, 2, ... も同様の構造
  （Superblockはバックアップのため一部のグループにも格納）
```

**Superblock（スーパーブロック）** はファイルシステム全体のメタデータを保持する最重要構造体である。

```
Superblock の主要フィールド:

  s_inodes_count       : inode の総数
  s_blocks_count       : ブロックの総数
  s_free_blocks_count  : 空きブロック数
  s_free_inodes_count  : 空き inode 数
  s_log_block_size     : ブロックサイズ（2の累乗で表現）
  s_blocks_per_group   : ブロックグループあたりのブロック数
  s_inodes_per_group   : ブロックグループあたりの inode 数
  s_magic              : ファイルシステム識別子（ext4 = 0xEF53）
  s_state              : ファイルシステムの状態（クリーン/エラー）
  s_feature_compat     : 互換性のある機能フラグ
  s_feature_incompat   : 非互換の機能フラグ
  s_feature_ro_compat  : 読み取り専用互換の機能フラグ
  s_uuid               : ファイルシステムの UUID
  s_volume_name        : ボリューム名
  s_last_mounted       : 最後にマウントされたパス

Superblock のバックアップ:
  - ブロックグループ 0, 1, 3, 5, 7, 9, 25, 27, ... に格納
  - 3, 5, 7 の累乗のグループに格納（sparse superblock）
  - メインの Superblock が破損しても復旧可能

  復旧例:
  $ sudo mke2fs -n /dev/sda1  # バックアップ位置を確認
  Superblock backups stored on blocks:
    32768, 98304, 163840, 229376, ...

  $ sudo e2fsck -b 32768 /dev/sda1  # バックアップから修復
```

### 1.4 ブロックグループデスクリプタ

```
Block Group Descriptor（BGD）の内容:

  bg_block_bitmap      : ブロックビットマップの位置
  bg_inode_bitmap      : inode ビットマップの位置
  bg_inode_table       : inode テーブルの開始位置
  bg_free_blocks_count : グループ内の空きブロック数
  bg_free_inodes_count : グループ内の空き inode 数
  bg_used_dirs_count   : グループ内のディレクトリ数

ビットマップの仕組み:
  ブロックビットマップ:
  各ビットが1ブロックに対応（使用中=1, 空き=0）

  例: 8ブロック分のビットマップ
  ┌─┬─┬─┬─┬─┬─┬─┬─┐
  │1│1│0│1│0│0│1│0│
  └─┴─┴─┴─┴─┴─┴─┴─┘
   B0 B1 B2 B3 B4 B5 B6 B7
   使 使 空 使 空 空 使 空

  1ブロック（4KB）のビットマップで管理できるブロック数:
  4096 × 8 = 32768 ブロック
  32768 × 4KB = 128MB

  → 1つのブロックグループは最大128MB
```

### 1.5 Flex Block Groups（ext4の拡張）

```
Flex Block Groups:
  複数のブロックグループのメタデータをまとめて配置
  → メタデータの局所性が向上し、シーク回数を削減

  通常のレイアウト:
  ┌────────┐┌────────┐┌────────┐┌────────┐
  │meta data││meta data││meta data││meta data│
  │ + data ││ + data ││ + data ││ + data │
  └────────┘└────────┘└────────┘└────────┘
   BG0       BG1       BG2       BG3

  Flex Block Groups:
  ┌──────────────────┐┌──────┐┌──────┐┌──────┐
  │ meta0+meta1+     ││ data ││ data ││ data │
  │ meta2+meta3      ││      ││      ││      │
  └──────────────────┘└──────┘└──────┘└──────┘
   Flex BG 0 (メタデータ集約)     データ領域

  → メタデータの読み取りが高速化
  → 大容量ファイルシステムで効果的
```

---

## 2. inodeとディレクトリ

### 2.1 inodeの構造

```
inode（index node）:
  ファイルのメタデータを格納する構造体
  ※ ファイル名は含まれない！名前はディレクトリが管理

  ┌──────────────────────────────────┐
  │ inode #12345                     │
  ├──────────────────────────────────┤
  │ ファイルタイプ: 通常ファイル     │
  │ パーミッション: rwxr-xr-x       │
  │ 所有者: uid=1000                │
  │ グループ: gid=1000              │
  │ サイズ: 4096 bytes              │
  │ タイムスタンプ:                 │
  │   atime: 最終アクセス時刻       │
  │   mtime: 最終変更時刻           │
  │   ctime: メタデータ変更時刻     │
  │   crtime: 作成時刻（ext4拡張）  │
  │ リンク数: 1                     │
  │ ブロック数: 8 (512B単位)        │
  │ フラグ: 0x80000 (extents使用)   │
  │ データブロックポインタ:          │
  │   直接: [B1][B2]...[B12]       │
  │   間接: [→ ブロック群]          │
  │   二重間接: [→→ ブロック群]     │
  │   三重間接: [→→→ ...]          │
  └──────────────────────────────────┘
```

### 2.2 データブロックポインタの仕組み

```
従来のブロックポインタ方式（ext2/ext3）:

  inode のデータポインタ:
  ┌──────────────┐
  │ 直接ポインタ  │ ×12個 → 12 × 4KB = 48KB
  │ [0]-[11]     │
  ├──────────────┤
  │ 間接ポインタ  │ ×1個 → 1024 × 4KB = 4MB
  │ [12]         │
  ├──────────────┤
  │ 二重間接      │ ×1個 → 1024² × 4KB = 4GB
  │ [13]         │
  ├──────────────┤
  │ 三重間接      │ ×1個 → 1024³ × 4KB = 4TB
  │ [14]         │
  └──────────────┘

  間接ポインタの展開:
  inode[12] → ┌──────┐
              │ B100 │
              │ B101 │
              │ B102 │
              │ ...  │ (1024エントリ)
              │ B1123│
              └──────┘

  問題点:
  - 大きなファイルでは多段参照が必要
  - 連続ブロックでも個別にポインタを保持 → メタデータが肥大化
  - 例: 1GBの連続ファイル → 262,144個のブロックポインタが必要
```

### 2.3 エクステント（ext4）

```
ext4のエクステント:
  連続するブロックを「開始ブロック + ブロック数」で表現
  → メタデータの大幅削減

  エクステント構造:
  ┌────────────────────────────────┐
  │ ee_block: 論理ブロック番号      │
  │ ee_len:   ブロック数            │
  │ ee_start: 物理ブロック番号      │
  └────────────────────────────────┘

  例: 1GB の連続ファイル
  従来: 262,144個のブロックポインタ
  エクステント: 1個のエクステント（開始 + 長さ）

  エクステントツリー:
  ┌──────────────────┐
  │ inode            │
  │ ┌──────────────┐ │
  │ │ ヘッダ       │ │
  │ │ エクステント1 │ │ → ブロック 0-99   → 物理 1000-1099
  │ │ エクステント2 │ │ → ブロック 100-299 → 物理 2000-2199
  │ │ エクステント3 │ │ → ブロック 300-599 → 物理 5000-5299
  │ │ インデックス  │ │ → 追加のツリーノード
  │ └──────────────┘ │
  └──────────────────┘

  inodeに直接格納できるエクステントは4個まで
  → それ以上はB木構造で管理（エクステントツリー）

  利点:
  - 連続アロケーションの効率的な表現
  - メタデータの大幅削減
  - 大ファイルの処理が高速
  - 断片化が少ない場合に特に効果的
```

### 2.4 ディレクトリの実装

```
ディレクトリ:
  ファイル名とinode番号の対応表

  /home/user/ のディレクトリエントリ:
  ┌──────────────┬──────────┬──────────┬──────────┐
  │ ファイル名    │ inode番号│ エントリ長│ タイプ   │
  ├──────────────┼──────────┼──────────┼──────────┤
  │ .            │ 201      │ 12       │ DIR      │
  │ ..           │ 100      │ 12       │ DIR      │
  │ document.txt │ 12345    │ 20       │ REG      │
  │ photo.jpg    │ 12346    │ 16       │ REG      │
  │ scripts/     │ 12400    │ 16       │ DIR      │
  └──────────────┴──────────┴──────────┴──────────┘

  ext4のディレクトリ実装方式:

  1. リニアディレクトリ:
     - エントリを順番に格納
     - 小さなディレクトリ向け
     - 検索: O(n)

  2. HTree（ハッシュツリー）ディレクトリ:
     - ファイル名のハッシュでB木を構成
     - 大きなディレクトリ向け（数万ファイル以上）
     - 検索: O(log n)
     - デフォルトで有効

  HTreeの構造:
  ┌─────────────────────────────┐
  │ ルートノード                 │
  │ hash < 0x4000 → ブロック5   │
  │ hash < 0x8000 → ブロック12  │
  │ hash < 0xC000 → ブロック18  │
  │ hash >= 0xC000 → ブロック25 │
  └─────┬───────┬───────┬───────┘
        ↓       ↓       ↓
    ┌──────┐┌──────┐┌──────┐
    │エントリ││エントリ││エントリ│
    │の一覧 ││の一覧 ││の一覧 │
    └──────┘└──────┘└──────┘
```

### 2.5 パス名解決（Path Resolution）

```
open("/home/user/document.txt") の処理:

  ステップ1: "/" を解決
  → ルートディレクトリの inode（通常 inode 2）を取得
  → dentryキャッシュを確認

  ステップ2: "home" を解決
  → "/" の inode からディレクトリエントリを検索
  → "home" に対応する inode 番号を取得（例: 100）
  → inode 100 のパーミッションチェック（x ビット）

  ステップ3: "user" を解決
  → inode 100 のディレクトリエントリから "user" を検索
  → inode 201 を取得
  → パーミッションチェック

  ステップ4: "document.txt" を解決
  → inode 201 のディレクトリエントリから "document.txt" を検索
  → inode 12345 を取得
  → パーミッションチェック（r ビット）

  ステップ5: ファイルオープン
  → inode 12345 をメモリにロード
  → ファイルディスクリプタを割り当て
  → file 構造体を作成して返す

  重要: 各ステップでパーミッションチェックが行われる
  → ディレクトリの "x"（実行）ビットが必要
  → "x" がないとディレクトリ内のファイルにアクセスできない

  パス名解決の最適化:
  - dentryキャッシュ: 解決済みのパス→inodeマッピングをメモリに保持
  - 否定dentry: 存在しないパスも記録（不要な検索を回避）
  - RCU（Read-Copy-Update）: ロックなしの並行アクセス
```

### 2.6 ハードリンクとシンボリックリンク

```
ハードリンク:
  同じinodeを指す別名

  $ echo "hello" > original.txt   # inode 12345
  $ ln original.txt link.txt       # inode 12345 (同一)

  ┌──────────────┐     ┌──────────────┐
  │ original.txt │────→│  inode       │
  └──────────────┘     │  #12345      │
  ┌──────────────┐     │  links: 2   │
  │ link.txt     │────→│  data: ...   │
  └──────────────┘     └──────────────┘

  特徴:
  - inode番号が同一
  - リンクカウントが増加
  - どちらからアクセスしても同じデータ
  - 片方を削除してもデータは残る（リンクカウント > 0 の間）
  - リンクカウントが0になるとデータ領域が解放

  制約:
  - ディレクトリのハードリンクは不可（ループ防止）
    → 「.」と「..」のみ例外（カーネルが管理）
  - パーティション/ファイルシステム境界を跨げない
    → inodeはFS内でのみ一意

シンボリックリンク（シムリンク）:
  パスを格納する特別なファイル

  $ ln -s /path/to/original symlink

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ symlink      │────→│ inode #99999 │     │ inode #12345 │
  └──────────────┘     │ type: LINK   │     │ type: REG    │
                       │ data:        │────→│ data: ...    │
                       │"/path/to/    │     └──────────────┘
                       │ original"    │
                       └──────────────┘

  特徴:
  - 別のinodeを持つ
  - パス文字列を格納（60バイト以下ならinode内に直接格納）
  - ディレクトリへのリンクが可能
  - パーティション境界を跨げる
  - リンク先が削除されるとダングリングリンク（壊れたリンク）

  ファストシムリンク（ext4）:
  - パスが60バイト以下の場合、inode内のデータポインタ領域に直接格納
  - ディスクブロックを追加消費しない
  - → 大部分のシムリンクがファストシムリンク

  比較表:
  ┌──────────────┬──────────────┬──────────────┐
  │              │ ハードリンク │ シムリンク   │
  ├──────────────┼──────────────┼──────────────┤
  │ inode        │ 同一         │ 別           │
  │ FS跨ぎ       │ 不可         │ 可能         │
  │ ディレクトリ │ 不可         │ 可能         │
  │ リンク先削除 │ データ存続   │ ダングリング │
  │ ディスク消費 │ なし         │ inode1個     │
  │ パス更新     │ 不要         │ 必要な場合   │
  └──────────────┴──────────────┴──────────────┘
```

### 2.7 特殊ファイル

```
Linuxにおける特殊ファイル:

  1. デバイスファイル:
     - キャラクタデバイス (c): /dev/tty, /dev/null
     - ブロックデバイス (b): /dev/sda, /dev/nvme0n1
     - メジャー番号 + マイナー番号でデバイスを識別

  2. 名前付きパイプ (FIFO):
     - mkfifo /tmp/mypipe
     - プロセス間通信に使用
     - 一方が書き込み、他方が読み取り

  3. UNIXドメインソケット:
     - ローカルプロセス間のネットワーク風通信
     - /var/run/docker.sock, /tmp/.X11-unix/X0

  4. 特殊な仮想ファイル:
     /dev/null   : 書き込みを破棄。読み取りは即EOF
     /dev/zero   : 無限のゼロバイトを生成
     /dev/random : 暗号学的に安全な乱数（エントロピー枯渇時ブロック）
     /dev/urandom: 非ブロッキング疑似乱数
     /dev/full   : 書き込みで ENOSPC エラーを返す（テスト用）

  ファイルタイプの確認:
  $ ls -la /dev/null /dev/sda /tmp/mypipe
  crw-rw-rw- 1 root root 1, 3 ... /dev/null     # c=キャラクタ
  brw-rw---- 1 root disk 8, 0 ... /dev/sda       # b=ブロック
  prw-r--r-- 1 user user 0    ... /tmp/mypipe    # p=パイプ
  srwxrwxrwx 1 root root 0    ... /var/run/docker.sock  # s=ソケット

  $ stat --format '%F' /dev/null
  character special file
```

---

## 3. ジャーナリング

### 3.1 クラッシュ一貫性問題

```
問題: 書き込み中に電源断が発生したら？

  ファイル作成の手順:
  1. inodeビットマップで空きinodeを見つけて確保
  2. inodeにメタデータを書き込み
  3. ディレクトリにエントリ追加
  4. ブロックビットマップで空きブロックを確保
  5. データブロックにデータを書き込み

  電源断が発生した場合の不整合パターン:

  ケース1: ステップ2の後に電源断
  → inodeは確保されたがディレクトリエントリがない
  → inodeは存在するがアクセス不可 = orphan inode
  → inode がリーク（使用中だが参照されない）

  ケース2: ステップ3の後に電源断
  → ディレクトリエントリはあるがデータがない
  → ファイルは見えるが中身がゴミデータ

  ケース3: ステップ4の後に電源断
  → ブロックは確保されたがデータが書かれていない
  → 前のファイルのデータが見える可能性（セキュリティリスク）

  ケース4: ファイル追記中に電源断
  → inodeのサイズ更新とデータ書き込みが不一致
  → ファイルの末尾にゴミデータ

  これらの問題を解決する手段:
  1. fsck（ファイルシステムチェック）: 起動時に全データを検査
     → 大容量ディスクでは数時間〜数十時間かかる
  2. ジャーナリング: 変更を先にログに記録
     → 起動時はジャーナルだけ確認すればよい（秒単位）
  3. CoW: データを上書きせず新しい場所に書く
     → 原理的にクラッシュ一貫性を保証
```

### 3.2 ジャーナリングの仕組み

```
ジャーナリング:
  変更を「ジャーナル（ログ）」に先に書き込む

  ┌──────────────────────────────────────────────┐
  │ 1. トランザクション開始                       │
  │    → 変更内容を記述したログレコードを作成      │
  │ 2. ジャーナルにログを書き込み                  │
  │    → 変更前のデータ + 変更後のデータ           │
  │ 3. ジャーナルをコミット                        │
  │    → コミットブロックを書き込み                │
  │ 4. 実際のデータ領域に書き込み（チェックポイント）│
  │ 5. ジャーナルのログを無効化                    │
  └──────────────────────────────────────────────┘

  ジャーナル領域の構造:
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ Desc │ Data │ Data │Commit│ Desc │ Data │Commit│
  │ Block│ Log1 │ Log2 │ Block│ Block│ Log3 │ Block│
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  ←──── トランザクション1 ────→←── トランザクション2 ──→

  電源断が発生した場合:
  ケース1: ジャーナル書き込み中に電源断
  → コミットブロックがない
  → トランザクション全体を破棄（ロールフォワードしない）
  → データ領域は変更されていないので一貫性維持

  ケース2: コミット後、実データ書き込み中に電源断
  → コミットブロックがある
  → ジャーナルのログを使ってリプレイ（再実行）
  → データ領域を正しい状態に復旧

  ケース3: 正常完了後
  → ジャーナルのログは不要になり無効化
```

### 3.3 ジャーナリングモード（ext4）

```
ext4のジャーナリングモード:

  1. journal モード:
     - データ + メタデータの両方をジャーナリング
     - 最も安全だが最も遅い
     - データを2回書き込む（ジャーナル + 実領域）
     - 用途: 極めて高い信頼性が求められる場合

  2. ordered モード（デフォルト）:
     - メタデータのみジャーナリング
     - データはメタデータのコミット前に書き込み
     - 保証: メタデータが更新される時、データは既に書かれている
     - 用途: 大多数の用途で最適なバランス

  3. writeback モード:
     - メタデータのみジャーナリング
     - データの書き込み順序は保証しない
     - 最速だがデータ消失のリスクあり
     - 用途: 一時ファイル、再生成可能なデータ

  パフォーマンス比較（相対値）:
  ┌──────────┬──────┬──────┬──────┐
  │ モード    │ 安全性│ 読み込み│ 書き込み│
  ├──────────┼──────┼──────┼──────┤
  │ journal  │ ◎   │ 100  │  60  │
  │ ordered  │ ○   │ 100  │  85  │
  │ writeback│ △   │ 100  │ 100  │
  └──────────┴──────┴──────┴──────┘

  設定方法:
  # /etc/fstab
  /dev/sda1  /  ext4  data=ordered  0  1

  # マウント時に指定
  $ sudo mount -o data=journal /dev/sda1 /mnt

  # 現在のモードを確認
  $ cat /proc/fs/ext4/sda1/options | grep data
  data=ordered

  # ジャーナルの状態確認
  $ sudo dumpe2fs /dev/sda1 | grep -i journal
  Journal inode:            8
  Journal backup:           inode blocks
  Journal features:         journal_incompat_revoke journal_64bit
  Journal size:             128M
  Journal length:           32768
  Journal sequence:         0x000c3a10
```

### 3.4 チェックポイントとジャーナルの管理

```
チェックポイント:
  ジャーナルのログを実データ領域に反映する処理

  ┌───────────┐     ┌───────────┐     ┌───────────┐
  │ アプリ     │     │ ジャーナル │     │ データ    │
  │ write()   │────→│ ログ記録  │────→│ 実反映    │
  └───────────┘     └───────────┘     └───────────┘
                    ← 高速書き込み →  ← バックグラウンド →

  ジャーナルの循環バッファ:
  ┌──────────────────────────────────────┐
  │                                      │
  │  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┐      │
  │  │T1│T2│  │  │T5│T6│T7│  │  │      │
  │  └──┴──┴──┴──┴──┴──┴──┴──┴──┘      │
  │       ↑              ↑               │
  │    チェックポイント  最新コミット      │
  │    済み位置           位置            │
  └──────────────────────────────────────┘

  T1, T2: チェックポイント完了 → 空き領域として再利用可能
  T5-T7: まだ実データに反映されていない

  ジャーナルが満杯になった場合:
  → 新しいトランザクションをブロック
  → チェックポイントを強制実行
  → ジャーナルサイズの適切な設定が重要

  ジャーナルサイズの推奨値:
  - 小規模（< 100GB）: 64MB
  - 中規模（100GB-1TB）: 128MB（デフォルト）
  - 大規模（> 1TB）: 256MB-1GB

  $ sudo tune2fs -J size=256 /dev/sda1  # ジャーナルサイズ変更
```

### 3.5 Copy-on-Write（CoW）ファイルシステム

```
CoW（Copy-on-Write）ファイルシステム:
  Btrfs, ZFS はジャーナルの代わりにCoWを使用
  → データを上書きせず、新しい場所に書き込み
  → アトミックな更新、スナップショットが高速

  CoW の仕組み:
  ┌──────────────────────────────────────────────┐
  │ 更新前:                                       │
  │   Root → Node A → [Leaf 1] [Leaf 2] [Leaf 3]│
  │                                              │
  │ Leaf 2 を更新する場合:                        │
  │ 1. Leaf 2 のコピーを新しい場所に作成          │
  │ 2. コピーにデータを書き込み                    │
  │ 3. Node A のコピーを作成（新Leaf 2を指す）     │
  │ 4. Root のコピーを作成（新Node Aを指す）       │
  │ 5. Superblock を新Root に更新（アトミック操作）│
  │                                              │
  │ 更新後:                                       │
  │   Root' → Node A' → [Leaf 1] [Leaf 2'] [Leaf 3]│
  │                                              │
  │ 旧 Root, Node A, Leaf 2 は解放可能           │
  │ （スナップショットが参照中なら保持）           │
  └──────────────────────────────────────────────┘

  CoW の利点:
  - クラッシュ一貫性が原理的に保証される
  - スナップショットが瞬時に作成可能（旧データを保持するだけ）
  - ロールバックが容易

  CoW の欠点:
  - 書き込み増幅（小さな変更でもツリー全体のパスをコピー）
  - 断片化しやすい（データが分散配置される）
  - ランダム書き込みのパフォーマンスが低下する場合がある

  ジャーナリング vs CoW:
  ┌──────────────┬─────────────────┬─────────────────┐
  │              │ ジャーナリング  │ CoW            │
  ├──────────────┼─────────────────┼─────────────────┤
  │ 整合性保証   │ ログベース      │ 構造的保証      │
  │ スナップショット│ 非対応         │ 瞬時に作成     │
  │ 書き込み増幅 │ 2倍（ログ+実） │ パスコピー分    │
  │ 断片化       │ 少ない          │ 多い            │
  │ 実装         │ ext4, XFS      │ Btrfs, ZFS     │
  └──────────────┴─────────────────┴─────────────────┘
```

---

## 4. VFS（Virtual File System）

### 4.1 VFSの概要

```
VFS: Linuxの統一ファイルシステムインターフェース

  アプリケーション
     │ open(), read(), write(), close()
     ↓
  ┌──────────────────────────────────────────┐
  │ VFS (Virtual File System)                │
  │ → 統一API                                │
  │ → dentryキャッシュ                        │
  │ → inodeキャッシュ                         │
  │ → ページキャッシュ                        │
  └──┬────┬────┬────┬────┬────┬────┬────────┘
     ↓    ↓    ↓    ↓    ↓    ↓    ↓
   ext4  XFS  Btrfs NTFS  NFS  procfs tmpfs
   (実際のファイルシステム実装)

  VFS の目的:
  - アプリケーションはファイルシステムの種類を意識する必要がない
  - 同じ open()/read()/write() で全FSにアクセス可能
  - 新しいFSの追加が容易（VFSインターフェースを実装するだけ）
  - ファイルシステム間でデータをコピーする際もシームレス
```

### 4.2 VFSの4つの主要オブジェクト

```
VFS の主要データ構造:

  1. struct super_block（スーパーブロック）:
     マウントされたファイルシステムの情報
     ┌────────────────────────────────┐
     │ s_dev:     デバイス識別子       │
     │ s_type:    ファイルシステム型    │
     │ s_op:      操作関数テーブル     │
     │ s_flags:   マウントフラグ       │
     │ s_root:    ルート dentry        │
     │ s_fs_info: FS固有のデータ       │
     └────────────────────────────────┘

  2. struct inode（inodeオブジェクト）:
     個々のファイルの情報（メモリ上の表現）
     ┌────────────────────────────────┐
     │ i_ino:     inode番号           │
     │ i_mode:    アクセス権限         │
     │ i_uid:     所有者ID            │
     │ i_gid:     グループID          │
     │ i_size:    ファイルサイズ       │
     │ i_op:      inode操作テーブル    │
     │ i_fop:     ファイル操作テーブル │
     │ i_sb:      所属するスーパーブロック│
     │ i_mapping: ページキャッシュ     │
     └────────────────────────────────┘

  3. struct dentry（ディレクトリエントリ）:
     パス名の各コンポーネントの情報
     ┌────────────────────────────────┐
     │ d_name:    名前                │
     │ d_inode:   対応する inode       │
     │ d_parent:  親 dentry           │
     │ d_subdirs: 子 dentry リスト     │
     │ d_op:      dentry操作テーブル   │
     │ d_flags:   状態フラグ          │
     └────────────────────────────────┘

     dentry キャッシュ:
     - ディスクI/Oなしでパス名を解決
     - LRUで管理、メモリ圧迫時に縮小
     - 否定dentry: 存在しないパスも記録

  4. struct file（ファイルオブジェクト）:
     プロセスが開いているファイルの状態
     ┌────────────────────────────────┐
     │ f_path:    パス情報             │
     │ f_inode:   対応する inode       │
     │ f_op:      ファイル操作テーブル │
     │ f_pos:     現在のオフセット     │
     │ f_flags:   オープンフラグ       │
     │ f_mode:    アクセスモード       │
     │ f_count:   参照カウント         │
     └────────────────────────────────┘
```

### 4.3 VFS操作テーブル

```c
// ファイル操作テーブル（file_operations）:
struct file_operations {
    loff_t (*llseek)(struct file *, loff_t, int);
    ssize_t (*read)(struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write)(struct file *, const char __user *, size_t, loff_t *);
    int (*open)(struct inode *, struct file *);
    int (*release)(struct inode *, struct file *);
    int (*fsync)(struct file *, loff_t, loff_t, int);
    int (*mmap)(struct file *, struct vm_area_struct *);
    // ...
};

// inode操作テーブル（inode_operations）:
struct inode_operations {
    struct dentry *(*lookup)(struct inode *, struct dentry *, unsigned int);
    int (*create)(struct user_namespace *, struct inode *, struct dentry *,
                  umode_t, bool);
    int (*link)(struct dentry *, struct inode *, struct dentry *);
    int (*unlink)(struct inode *, struct dentry *);
    int (*symlink)(struct inode *, struct dentry *, const char *);
    int (*mkdir)(struct user_namespace *, struct inode *, struct dentry *,
                 umode_t);
    int (*rmdir)(struct inode *, struct dentry *);
    int (*rename)(struct user_namespace *, struct inode *, struct dentry *,
                  struct inode *, struct dentry *, unsigned int);
    // ...
};

// 各ファイルシステムが独自の実装を提供:
// ext4の場合:
const struct file_operations ext4_file_operations = {
    .llseek    = ext4_llseek,
    .read_iter = ext4_file_read_iter,
    .write_iter = ext4_file_write_iter,
    .open      = ext4_file_open,
    .release   = ext4_release_file,
    .fsync     = ext4_sync_file,
    .mmap      = ext4_file_mmap,
};
```

### 4.4 マウントとアンマウント

```
マウント: ファイルシステムをディレクトリツリーに接合する操作

  $ mount /dev/sda1 /mnt

  マウント前:
  /
  ├── home/
  ├── mnt/        ← 空のディレクトリ
  └── tmp/

  マウント後:
  /
  ├── home/
  ├── mnt/        ← /dev/sda1 の内容が見える
  │   ├── data/
  │   └── config.txt
  └── tmp/

  マウント処理の内部:
  1. ファイルシステムのsuperblockを読み込み
  2. struct super_block を作成
  3. ルート inode を読み込み
  4. マウントポイントの dentry に紐付け
  5. mount構造体を vfsmount ツリーに追加

  マウントオプション:
  ┌────────────┬──────────────────────────────────────┐
  │ オプション  │ 説明                                  │
  ├────────────┼──────────────────────────────────────┤
  │ ro         │ 読み取り専用                          │
  │ rw         │ 読み書き可能                          │
  │ noatime    │ アクセス時刻を更新しない（性能向上）   │
  │ relatime   │ 条件付きでatime更新（デフォルト）     │
  │ nosuid     │ SUID/SGID ビットを無視               │
  │ noexec     │ 実行権限を無視                        │
  │ nodev      │ デバイスファイルを無視                │
  │ sync       │ 同期書き込み（性能低下）              │
  │ data=      │ ジャーナリングモード指定              │
  │ discard    │ TRIM/UNMAP コマンド発行（SSD向け）    │
  │ barrier=   │ 書き込みバリア制御                    │
  └────────────┴──────────────────────────────────────┘

  /etc/fstab の例:
  # <device>      <mount>  <type>  <options>              <dump> <fsck>
  /dev/sda1       /        ext4    defaults,noatime       0      1
  /dev/sda2       /home    ext4    defaults,nosuid        0      2
  /dev/sdb1       /data    xfs     defaults,nobarrier     0      0
  tmpfs           /tmp     tmpfs   defaults,size=4G       0      0
  UUID=xxxx-yyyy  /boot    ext4    defaults               0      2

  UUID でのマウント（推奨）:
  → デバイス名は変わる可能性がある（/dev/sda → /dev/sdb）
  → UUID はファイルシステム固有で変わらない
  $ blkid  # UUID の確認
```

### 4.5 ファイルディスクリプタ

```
ファイルディスクリプタ（FD）:
  プロセスがオープンしたファイルへの参照（整数値）

  標準的な FD:
  0: stdin  (標準入力)
  1: stdout (標準出力)
  2: stderr (標準エラー出力)
  3〜: ユーザがオープンしたファイル

  データ構造の関係:
  ┌─────────────────────────────────────────────────┐
  │ プロセスの task_struct                           │
  │ ┌───────────────────┐                           │
  │ │ files_struct       │                           │
  │ │ ┌───────────────┐ │                           │
  │ │ │ fd_array       │ │                           │
  │ │ │ [0] → file A  │─┼──→ struct file ──→ inode  │
  │ │ │ [1] → file B  │─┼──→ struct file ──→ inode  │
  │ │ │ [2] → file C  │─┼──→ struct file ──→ inode  │
  │ │ │ [3] → file D  │─┼──→ struct file ──→ inode  │
  │ │ └───────────────┘ │                           │
  │ └───────────────────┘                           │
  └─────────────────────────────────────────────────┘

  fork() 時の FD 共有:
  親プロセスの FD テーブルがコピーされる
  → 同じ struct file を共有（参照カウント+1）
  → ファイルオフセットも共有される

  FD の上限:
  $ ulimit -n              # ソフトリミット確認（通常1024）
  $ ulimit -Hn             # ハードリミット確認
  $ cat /proc/sys/fs/file-max  # システム全体の上限

  # リミット変更
  $ ulimit -n 65536        # セッション内で変更
  # /etc/security/limits.conf で恒久設定
  * soft nofile 65536
  * hard nofile 65536
```

---

## 5. ファイルシステムの整合性とメンテナンス

### 5.1 fsck（File System Check）

```
fsck: ファイルシステムの整合性チェックと修復ツール

  チェック項目:
  1. スーパーブロックの整合性
  2. ブロックビットマップとinodeビットマップの正確性
  3. inodeのリンクカウント
  4. ディレクトリ構造の整合性
  5. 孤立inode（参照されていないinode）の検出
  6. 不正なブロックポインタの検出
  7. 重複割り当てブロックの検出

  使用方法:
  # 注意: アンマウント状態またはリードオンリーで実行すること！
  $ sudo umount /dev/sda1
  $ sudo fsck /dev/sda1

  # ext4 専用
  $ sudo e2fsck -f /dev/sda1       # 強制チェック
  $ sudo e2fsck -p /dev/sda1       # 自動修復
  $ sudo e2fsck -y /dev/sda1       # 全質問にyes

  # XFS 専用
  $ sudo xfs_repair /dev/sda1

  # Btrfs
  $ sudo btrfs check /dev/sda1
  $ sudo btrfs scrub start /mnt    # オンラインチェック

  ジャーナリング FS での fsck:
  → 通常は不要（ジャーナルのリプレイで復旧）
  → ジャーナルが破損した場合のみ必要
  → 大容量ディスクでも秒単位で復旧

  非ジャーナリング FS での fsck:
  → 起動時に毎回実行する必要がある場合あり
  → 大容量ディスクでは数時間〜数十時間
  → ext2 時代の悩みの種だった
```

### 5.2 TRIMとSSDの考慮事項

```
SSD固有のファイルシステム考慮事項:

  TRIM（UNMAP）:
  → 削除されたブロックをSSDに通知
  → SSDのガベージコレクションを効率化
  → 書き込み性能の維持に重要

  仕組み:
  ファイル削除時:
  1. 従来: ファイルシステムはブロックを「空き」にマーク
           → SSDはまだ「使用中」と認識
  2. TRIM: ファイルシステムがSSDに「このブロックは不要」と通知
           → SSDがバックグラウンドで消去（次回書き込みが高速化）

  設定方法:
  # /etc/fstab に discard オプション追加（連続TRIM）
  /dev/sda1  /  ext4  defaults,discard  0  1

  # 定期的なバッチTRIM（推奨、性能への影響が少ない）
  $ sudo fstrim /                   # 手動実行
  $ sudo fstrim -v /                # 詳細表示

  # systemd タイマーで定期実行
  $ sudo systemctl enable fstrim.timer  # 週1回実行

  # TRIM 対応の確認
  $ lsblk --discard
  NAME   DISC-ALN DISC-GRAN DISC-MAX DISC-ZERO
  sda           0      512B       2G         0
  nvme0n1       0      512B       2T         0

  SSD のアライメント:
  → パーティション開始位置を物理ページサイズに合わせる
  → 最近のツール（fdisk, parted）はデフォルトで対処
  → 不適切なアライメントは性能低下の原因

  $ sudo parted /dev/sda align-check optimal 1
  1 aligned

  noatime の推奨:
  → ファイル読み取りのたびにatime更新 = 不要な書き込み
  → SSD の寿命に影響
  → noatime または relatime を推奨
```

### 5.3 ファイルシステムのデフラグメンテーション

```
断片化（Fragmentation）:

  外部断片化:
  ファイルのブロックが非連続に配置される
  → HDD: シーク時間増加 → 性能低下
  → SSD: 影響は小さいが完全にゼロではない

  断片化の例:
  ブロック配置:
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │A1│B1│A2│C1│A3│B2│C2│A4│B3│C3│
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
  ファイルA: ブロック 0, 2, 4, 7 → 断片化
  ファイルB: ブロック 1, 5, 8 → 断片化
  ファイルC: ブロック 3, 6, 9 → 断片化

  デフラグ後:
  ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
  │A1│A2│A3│A4│B1│B2│B3│C1│C2│C3│
  └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
  → 連続配置で読み取り性能向上

  各FSのデフラグツール:
  # ext4
  $ sudo e4defrag /path/to/file    # 特定ファイル
  $ sudo e4defrag /mount/point     # マウントポイント全体
  $ sudo e4defrag -c /mount/point  # 断片化状況の確認

  # XFS
  $ sudo xfs_fsr /dev/sda1         # オンラインデフラグ
  $ sudo xfs_db -r /dev/sda1       # 断片化状況の確認

  # Btrfs
  $ sudo btrfs filesystem defragment /path  # オンラインデフラグ
  $ sudo btrfs filesystem defragment -r /mount  # 再帰的

  断片化を防ぐ工夫:
  - 遅延アロケーション（ext4のdelalloc）
  - プリアロケーション（fallocate）
  - 十分な空き容量の確保（10%以上推奨）
```

---

## 6. 高度なファイルシステムの概念

### 6.1 拡張属性（Extended Attributes, xattr）

```
拡張属性:
  通常のパーミッション以外の追加メタデータを格納

  名前空間:
  - user.*    : ユーザ定義の属性
  - system.*  : システム用（ACL等）
  - security.*: セキュリティモジュール用（SELinux等）
  - trusted.* : 特権プロセス用

  操作コマンド:
  # 属性の設定
  $ setfattr -n user.description -v "Important document" file.txt

  # 属性の取得
  $ getfattr -n user.description file.txt
  # file: file.txt
  user.description="Important document"

  # 全属性の一覧
  $ getfattr -d file.txt

  # 属性の削除
  $ setfattr -x user.description file.txt

  # SELinux コンテキスト（security名前空間）
  $ ls -Z file.txt
  unconfined_u:object_r:user_home_t:s0 file.txt

  ACL（Access Control List）:
  → POSIX ACL は拡張属性として格納される
  → system.posix_acl_access, system.posix_acl_default

  # ACL の設定
  $ setfacl -m u:john:rw file.txt   # john に rw 権限
  $ setfacl -m g:dev:rx dir/        # dev グループに rx 権限
  $ getfacl file.txt                # ACL の確認
```

### 6.2 クォータ（Quota）

```
ディスククォータ:
  ユーザ/グループごとのディスク使用量制限

  クォータの種類:
  - ブロッククォータ: 使用容量の制限
  - inodeクォータ: ファイル数の制限
  - ソフトリミット: 猶予期間内は超過可能
  - ハードリミット: 絶対に超えられない上限

  設定手順:
  # 1. マウントオプションにクォータを有効化
  $ sudo mount -o remount,usrquota,grpquota /home

  # 2. クォータファイルの作成
  $ sudo quotacheck -cum /home     # ユーザクォータ
  $ sudo quotacheck -cgm /home     # グループクォータ

  # 3. クォータの有効化
  $ sudo quotaon /home

  # 4. ユーザクォータの設定
  $ sudo edquota -u username
  # ソフトリミット: 5GB, ハードリミット: 10GB

  # 5. 使用状況の確認
  $ sudo repquota -a               # 全ユーザの使用状況
  $ quota -u username               # 特定ユーザの状況

  ext4 プロジェクトクォータ（ディレクトリ単位）:
  # /etc/projects
  1:/home/project_a
  2:/home/project_b

  # /etc/projid
  project_a:1
  project_b:2

  $ sudo tune2fs -O project /dev/sda1  # プロジェクト機能有効化
  $ sudo mount -o prjquota /dev/sda1 /home
```

### 6.3 スパースファイルとホール

```
スパースファイル:
  実際にデータが書かれたブロックのみディスクを消費
  → 論理サイズ > 物理サイズ

  例: 1TBのスパースファイル作成
  $ truncate -s 1T sparse_file.img
  $ ls -lh sparse_file.img
  -rw-r--r-- 1 user user 1.0T ... sparse_file.img  # 論理1TB
  $ du -h sparse_file.img
  0       sparse_file.img                            # 物理0

  ホール（穴）の仕組み:
  ┌──────────────────────────────────────┐
  │ 論理ブロック:                        │
  │ [0] [1] [2] [3] [4] [5] [6] [7]    │
  │  ↓       ↓               ↓         │
  │  データ   データ            データ    │
  │          ↓                          │
  │ ブロック 3-5 はホール（未割り当て）  │
  │ → 読み取ると 0x00 が返る            │
  │ → ディスクブロックは消費しない      │
  └──────────────────────────────────────┘

  ホールの検出と操作:
  # SEEK_HOLE / SEEK_DATA でホールを検出
  $ python3 -c "
  import os
  fd = os.open('sparse_file.img', os.O_RDONLY)
  hole = os.lseek(fd, 0, os.SEEK_HOLE)
  print(f'First hole at: {hole}')
  os.close(fd)
  "

  # cp でスパースファイルを効率的にコピー
  $ cp --sparse=always source dest

  # tar でスパースファイルを効率的にアーカイブ
  $ tar -cSf archive.tar sparse_file.img

  用途:
  - 仮想マシンのディスクイメージ（qcow2, VMDK）
  - データベースのプリアロケーション
  - コアダンプファイル
```

### 6.4 メモリマップドファイル（mmap）

```
mmap:
  ファイルの内容を仮想メモリ空間に直接マッピング

  通常の read/write:
  アプリ → read() → カーネル → ページキャッシュ → ディスク
  データがカーネル空間 → ユーザ空間にコピーされる

  mmap:
  アプリの仮想アドレスが直接ページキャッシュを指す
  → コピーが不要 → 高速

  ┌──────────────────────────────────────┐
  │ プロセスの仮想アドレス空間           │
  │                                      │
  │ ┌────────────────┐                   │
  │ │ text segment   │                   │
  │ ├────────────────┤                   │
  │ │ data segment   │                   │
  │ ├────────────────┤                   │
  │ │ mmap 領域      │───→ ページキャッシュ → ディスク
  │ │ (ファイル内容) │                   │
  │ ├────────────────┤                   │
  │ │ heap           │                   │
  │ ├────────────────┤                   │
  │ │ stack          │                   │
  │ └────────────────┘                   │
  └──────────────────────────────────────┘

  フラグ:
  - MAP_SHARED:  変更がファイルに反映される
  - MAP_PRIVATE: CoW。変更はプロセス内のみ
  - MAP_ANONYMOUS: ファイルなし（メモリ確保用）

  用途:
  - 大きなファイルの効率的な読み書き
  - プロセス間共有メモリ
  - 実行可能ファイルのロード（テキストセグメント）
  - データベースのバッファ管理

  注意点:
  - ファイルサイズを超えた書き込みは SIGBUS
  - 32bit環境ではアドレス空間の制約
  - msync() でディスクへの明示的な同期
  - munmap() でマッピング解除
```

---

## 実践演習

### 演習1: [基礎] -- inode の確認

```bash
# inodeの確認
ls -li                        # inode番号表示
stat filename                 # 詳細なメタデータ
df -i                         # inode使用状況

# ハードリンクとシンボリックリンク
echo "hello" > original.txt
ln original.txt hardlink.txt
ln -s original.txt symlink.txt
ls -li original.txt hardlink.txt symlink.txt
# → hardlinkはinode同一、symlinkは異なるinode

# リンクカウントの確認
stat original.txt | grep Links
# Links: 2  ← ハードリンクがあるため

# シンボリックリンクの先を確認
readlink symlink.txt
readlink -f symlink.txt       # 絶対パスで表示

# ダングリングリンクの確認
rm original.txt
cat symlink.txt               # エラー（リンク先がない）
cat hardlink.txt              # 正常に読める（データはまだ存在）
```

### 演習2: [応用] -- ファイルシステムの調査

```bash
# マウントされたファイルシステムの確認
mount | column -t
df -Th                        # タイプ付きで表示
findmnt                       # ツリー表示
findmnt -t ext4               # ext4のみ表示

# ファイルシステムの詳細情報（Linux, ext4）
sudo dumpe2fs /dev/sda1 | head -30
sudo tune2fs -l /dev/sda1     # superblock情報

# ブロックグループの情報
sudo dumpe2fs /dev/sda1 | grep -A 5 "Group 0"

# ジャーナルの情報
sudo dumpe2fs /dev/sda1 | grep -i journal

# inodeの使用状況
df -i /                       # inode使用率
for dir in /*; do echo "$(find "$dir" -xdev 2>/dev/null | wc -l) $dir"; done | sort -rn | head

# ブロックサイズの確認
sudo tune2fs -l /dev/sda1 | grep "Block size"
stat -f /                     # ファイルシステム情報
```

### 演習3: [応用] -- ファイルシステムの作成とマウント

```bash
# テスト用のループバックファイルシステム作成
# 1. ファイルを作成
dd if=/dev/zero of=/tmp/testfs.img bs=1M count=100

# 2. ext4ファイルシステムを作成
mkfs.ext4 /tmp/testfs.img

# 3. マウント
sudo mkdir -p /mnt/testfs
sudo mount -o loop /tmp/testfs.img /mnt/testfs

# 4. 確認
df -Th /mnt/testfs
ls -la /mnt/testfs
sudo dumpe2fs /tmp/testfs.img | head -20

# 5. テスト書き込み
sudo touch /mnt/testfs/testfile
sudo ls -li /mnt/testfs/

# 6. アンマウント
sudo umount /mnt/testfs

# XFS ファイルシステムの作成
dd if=/dev/zero of=/tmp/testxfs.img bs=1M count=100
mkfs.xfs /tmp/testxfs.img
sudo mount -o loop /tmp/testxfs.img /mnt/testfs
xfs_info /mnt/testfs

# Btrfs ファイルシステムの作成
dd if=/dev/zero of=/tmp/testbtrfs.img bs=1M count=256
mkfs.btrfs /tmp/testbtrfs.img
sudo mount -o loop /tmp/testbtrfs.img /mnt/testfs
btrfs filesystem show /mnt/testfs
```

### 演習4: [上級] -- ファイルシステムのパフォーマンス測定

```bash
# fio を使ったI/Oベンチマーク
# シーケンシャル読み取り
fio --name=seqread --rw=read --bs=4k --size=1G \
    --numjobs=1 --runtime=30 --time_based

# ランダム読み取り
fio --name=randread --rw=randread --bs=4k --size=1G \
    --numjobs=4 --runtime=30 --time_based

# シーケンシャル書き込み
fio --name=seqwrite --rw=write --bs=4k --size=1G \
    --numjobs=1 --runtime=30 --time_based

# ランダム書き込み
fio --name=randwrite --rw=randwrite --bs=4k --size=1G \
    --numjobs=4 --runtime=30 --time_based

# dd を使った簡易ベンチマーク
# 書き込み速度
dd if=/dev/zero of=/tmp/testfile bs=1M count=1024 conv=fdatasync

# 読み取り速度（キャッシュクリア後）
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
dd if=/tmp/testfile of=/dev/null bs=1M

# ページキャッシュの効果を確認
# 1回目（ディスクから読み取り）
time cat /tmp/testfile > /dev/null
# 2回目（ページキャッシュから読み取り）
time cat /tmp/testfile > /dev/null
```

### 演習5: [上級] -- スパースファイルとエクステントの確認

```bash
# スパースファイルの作成と確認
truncate -s 10G /tmp/sparse_test
ls -lh /tmp/sparse_test       # 論理サイズ: 10G
du -h /tmp/sparse_test        # 実際のディスク使用: 0

# 一部にデータを書き込み
dd if=/dev/urandom of=/tmp/sparse_test bs=4K count=1 seek=1000
dd if=/dev/urandom of=/tmp/sparse_test bs=4K count=1 seek=2000

du -h /tmp/sparse_test        # 8K のみ使用

# エクステント情報の確認（ext4）
# filefrag: ファイルの断片化とエクステント情報を表示
filefrag -v /tmp/sparse_test

# debugfs で inode を直接確認
sudo debugfs -R "stat <$(stat -c %i /path/to/file)>" /dev/sda1

# hdparm でディスクキャッシュの確認
sudo hdparm -t /dev/sda       # バッファなし読み取り速度
sudo hdparm -T /dev/sda       # バッファキャッシュ読み取り速度
```

---

## 7. ファイルシステムのトラブルシューティング

### 7.1 よくある問題と対処法

```
問題1: "No space left on device" だがdfでは空きがある
  原因: inode枯渇
  確認: $ df -i
  対処: 小さなファイルの大量削除、またはinodeを増やしてFS再作成

問題2: ファイルを削除してもディスク容量が減らない
  原因: プロセスがファイルをオープン中
  確認: $ lsof +D /path/to/dir | grep deleted
  対処: プロセスを再起動、または /proc/<pid>/fd/ からFDを特定

問題3: ファイルシステムが読み取り専用になった
  原因: ファイルシステムエラー検出による自動保護
  確認: $ dmesg | grep -i "remount"
  対処: $ sudo fsck /dev/sda1 → 修復後再マウント

問題4: マウントできない
  原因: スーパーブロック破損
  確認: $ sudo file -s /dev/sda1
  対処: $ sudo e2fsck -b 32768 /dev/sda1  # バックアップSBで修復

問題5: パフォーマンスが著しく低下
  原因: 断片化、ジャーナル飽和、キャッシュ不足
  確認:
  $ sudo e4defrag -c /mount/point   # 断片化率
  $ vmstat 1                         # I/O待ち確認
  $ iostat -x 1                      # デバイスI/O詳細
  対処: デフラグ、ジャーナルサイズ調整、メモリ増設
```

### 7.2 データ復旧

```
削除されたファイルの復旧:

  なぜ復旧可能なのか:
  → ファイル削除 = ディレクトリエントリの削除 + inode解放
  → データブロック自体はすぐには上書きされない
  → 新しいデータが書き込まれるまで復旧の可能性あり

  復旧ツール:
  # ext4
  $ sudo extundelete /dev/sda1 --restore-all
  $ sudo ext4magic /dev/sda1 -r -d /tmp/recovered

  # 汎用
  $ sudo testdisk /dev/sda1       # パーティション復旧
  $ sudo photorec /dev/sda1       # ファイル復旧

  予防策:
  - 定期的なバックアップ（3-2-1ルール）
  - ゴミ箱の使用（trash-cli パッケージ）
  - rm の代わりに trash-put を使用
  - Btrfs/ZFS のスナップショットを定期取得
```

---

## FAQ

### Q1: なぜinodeが枯渇するのか？

小さなファイルが大量にある場合、ディスク容量に余裕があってもinode数が上限に達することがある（例: メールサーバーの大量メール、npm の node_modules）。`df -i` で確認可能。ext4ではmkfs時のオプションで初期inode数を指定できる。

```bash
# inode枯渇の確認
$ df -i /
Filesystem     Inodes  IUsed  IFree IUse% Mounted on
/dev/sda1      655360 655350     10  100% /

# inode を多く確保してFS作成
$ mkfs.ext4 -N 2000000 /dev/sda1   # 200万inodeを確保

# inode使用量の多いディレクトリを特定
$ for d in /*; do echo "$(find "$d" -xdev 2>/dev/null | wc -l) $d"; done | sort -rn | head -10
```

### Q2: ext4, XFS, Btrfsの選び方は？

- **ext4**: 最も安定。デスクトップ、一般サーバーに最適。最大16TBファイル。Ubuntu デフォルト。枯れた技術で信頼性が高い
- **XFS**: 大ファイル、高並列I/Oに強い。RHELのデフォルト。オンライン拡張可能だが縮小不可
- **Btrfs**: スナップショット、圧縮、RAID機能内蔵。SUSEのデフォルト。NAS用途に最適だがRAID5/6は未成熟

### Q3: noatime と relatime の違いは？

```
atime（アクセス時刻）の更新ポリシー:

  atime（デフォルト、旧方式）:
  → ファイル読み取りのたびにatime更新
  → 読み取りだけで書き込みI/Oが発生
  → SSDの寿命に悪影響

  noatime:
  → atimeを一切更新しない
  → 最もI/O効率が良い
  → メールの既読/未読判定がatime依存のソフトで問題

  relatime（現在のデフォルト）:
  → atime < mtime の場合のみ更新
  → または前回更新から24時間以上経過した場合に更新
  → atime依存ソフトとの互換性を維持しつつI/O削減
  → ほとんどの用途で最適
```

### Q4: ファイルシステムのUUIDとは？

```
UUID（Universally Unique Identifier）:
  ファイルシステム作成時にランダム生成される128bit識別子

  利点:
  - デバイス名（/dev/sda1）はハードウェア構成変更で変わる
  - UUIDは不変
  - /etc/fstab でUUIDを使うことで安定したマウントが可能

  確認方法:
  $ blkid
  /dev/sda1: UUID="a1b2c3d4-e5f6-7890-abcd-ef1234567890" TYPE="ext4"

  $ ls -la /dev/disk/by-uuid/

  $ lsblk -o NAME,UUID

  UUIDの再生成:
  $ sudo tune2fs -U random /dev/sda1  # ext4
  $ sudo xfs_admin -U generate /dev/sda1  # XFS
```

### Q5: ファイルの完全削除（安全消去）はどうするか？

```
通常の削除:
  → データブロックは残る（復旧可能）

安全な削除:
  # shred: データを上書き
  $ shred -vfz -n 3 sensitive_file
  # -v: 詳細表示, -f: 強制, -z: 最後にゼロで上書き, -n: 上書き回数

  注意: SSD では shred は信頼できない
  → FTL（Flash Translation Layer）がデータを別の場所に保持
  → TRIM + 暗号化消去（Secure Erase）を使用

  SSD の安全消去:
  $ sudo hdparm --security-set-pass password /dev/sda
  $ sudo hdparm --security-erase password /dev/sda

  推奨アプローチ:
  → 最初からフルディスク暗号化（LUKS, BitLocker）を使用
  → 廃棄時に暗号鍵を破棄するだけでデータは解読不可に
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ブロック/セクタ | FSの最小単位。ブロックサイズは性能と空間効率のトレードオフ |
| inode | ファイルのメタデータ。名前は含まない。エクステントで効率化 |
| ディレクトリ | 名前→inode番号の対応表。HTTreeで高速検索 |
| ジャーナリング | 不整合防止。電源断からの高速復旧。3つのモード |
| CoW | 上書きしない方式。スナップショット対応。Btrfs/ZFS |
| VFS | 統一API。4つの主要オブジェクト。異なるFSを透過的にアクセス |
| ファイルディスクリプタ | プロセスごとのファイル参照。上限設定に注意 |
| mmap | ファイルを仮想メモリにマッピング。コピー不要で高速 |
| TRIM | SSD向け。削除ブロックの通知。性能維持に重要 |
| xattr | 拡張属性。ACL、SELinuxコンテキスト等を格納 |

---

## 次に読むべきガイド
→ [[01-fs-implementations.md]] -- 主要FS実装

---

## 参考文献
1. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Ch.13-15, 2018.
2. Love, R. "Linux Kernel Development." 3rd Ed, Ch.13, 2010.
3. Bovet, D. & Cesati, M. "Understanding the Linux Kernel." 3rd Ed, O'Reilly, 2005.
4. Tanenbaum, A. S. & Bos, H. "Modern Operating Systems." 4th Ed, Pearson, 2014.
5. McKusick, M. K. et al. "The Design and Implementation of the FreeBSD Operating System." 2nd Ed, 2014.
6. Ts'o, T. "Design and Implementation of ext4." Ottawa Linux Symposium, 2009.
7. Linux Kernel Documentation. "Filesystems." https://www.kernel.org/doc/html/latest/filesystems/
8. Arpaci-Dusseau, R. & Arpaci-Dusseau, A. "Operating Systems: Three Easy Pieces." Ch.39-42, 2018.
