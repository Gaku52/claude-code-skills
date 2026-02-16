# Packfile/GC

> Gitのdelta圧縮とPackfileフォーマット、ガベージコレクション（GC）の仕組みを理解し、リポジトリサイズの最適化と性能チューニングの方法を習得する。

## この章で学ぶこと

1. **Packfileの構造とdelta圧縮** — looseオブジェクトがどのように効率的に圧縮されるか
2. **ガベージコレクション（GC）の仕組み** — 到達不可能オブジェクトの検出と削除プロセス
3. **リポジトリ最適化の実践** — 肥大化したリポジトリのサイズ削減と性能改善テクニック
4. **大規模リポジトリの運用** — partial clone、commit-graph、multi-pack-indexの活用
5. **トラブルシューティング** — 破損したpackfileの修復とデータ復旧

---

## 1. looseオブジェクトとpackfile

### 1.1 looseオブジェクト

各Git操作で生成されるオブジェクトは、最初は**looseオブジェクト**として`.git/objects/`に個別ファイルで保存される。

```bash
# looseオブジェクトの確認
$ find .git/objects -type f | grep -v pack | grep -v info | head -5
.git/objects/55/7db03de997c86a4a028e1ebd3a1ceb225be238
.git/objects/8f/94139338f9404f26296befa88755fc2598c289
.git/objects/a1/b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
```

```
┌────────────────────────────────────────────────┐
│  looseオブジェクトの格納形式                    │
│                                                │
│  ファイルパス: .git/objects/55/7db03de...       │
│                              ^^  ^^^^^^^^^^    │
│                            先頭2文字  残り38文字│
│                                                │
│  ファイル内容:                                  │
│  ┌──────────────────────────────┐              │
│  │ zlib_deflate(                │              │
│  │   "blob 12\0Hello, Git!\n"  │              │
│  │ )                           │              │
│  └──────────────────────────────┘              │
│  ← 個別にzlib圧縮されている                    │
└────────────────────────────────────────────────┘
```

### 1.2 looseオブジェクトの内部構造

looseオブジェクトのバイナリ形式は以下の通りである。

```bash
# looseオブジェクトの手動読み取り
$ python3 -c "
import zlib, sys
with open('.git/objects/55/7db03de997c86a4a028e1ebd3a1ceb225be238', 'rb') as f:
    data = zlib.decompress(f.read())
    print(repr(data))
"
# 出力: b'blob 12\x00Hello, Git!\n'
# 形式: <type> <size>\0<content>

# オブジェクトタイプ別のヘッダー
# blob:   "blob <size>\0<content>"
# tree:   "tree <size>\0<binary entries>"
# commit: "commit <size>\0<commit data>"
# tag:    "tag <size>\0<tag data>"
```

```bash
# SHA-1ハッシュの計算過程
$ echo -n "blob 12\0Hello, Git!\n" | shasum
557db03de997c86a4a028e1ebd3a1ceb225be238  -
# → ファイルパスはSHA-1から直接導出される

# Git 2.42+ ではSHA-256もサポート
$ git init --object-format=sha256
# → オブジェクトのハッシュが256ビットになる
```

### 1.3 looseオブジェクトの問題点

```bash
# リポジトリのオブジェクト数を確認
$ git count-objects -v
count: 1847          # looseオブジェクト数
size: 15280          # looseオブジェクトの合計サイズ(KB)
in-pack: 45231       # packfile内のオブジェクト数
packs: 2             # packfileの数
size-pack: 28456     # packfileの合計サイズ(KB)
prune-packable: 0    # pack済みで削除可能なlooseオブジェクト数
garbage: 0           # 不正なファイル数
size-garbage: 0
```

| 問題               | 説明                                                    |
|--------------------|---------------------------------------------------------|
| ファイル数の爆発   | inode消費、ファイルシステム性能の低下                    |
| 圧縮効率の低さ     | 各ファイルが独立してzlib圧縮、ファイル間の類似性を活用不可|
| I/O性能            | 多数の小ファイルの読み取りはシーク多発で遅い             |
| メタデータ消費     | ファイルシステムのメタデータ（inode等）が大量消費される  |
| ネットワーク転送   | 個別転送は非効率、packfileでの一括転送が必要             |

```bash
# looseオブジェクト数が多い場合の性能影響の確認
$ time git status
# looseオブジェクトが10万個以上 → 数秒かかる場合がある
# packfile化後 → ほぼ即座に返る

# ファイルシステムのinode使用状況
$ df -i .git
# looseオブジェクトはinode枯渇の原因になりうる
```

---

## 2. Packfileの構造

### 2.1 packfileの生成

```bash
# 手動でpackfileを生成
$ git repack -a -d
# -a: 全オブジェクトを1つのpackfileにまとめる
# -d: 不要になったlooseオブジェクトとold packを削除

# 自動生成のトリガー
# - git gc の実行
# - git push 時（サーバーへの転送用）
# - looseオブジェクト数が閾値を超えた場合（gc.auto, デフォルト6700）

# repackのオプション一覧
$ git repack -a -d -f --window=250 --depth=50
# -f: 既存のdeltaを破棄して再計算（より良いdeltaを見つける可能性）
# --window: delta候補を探すウィンドウサイズ
# --depth: deltaチェーンの最大深さ

# クルーシブ（cruft）packの作成（Git 2.37+）
$ git repack --cruft -d
# → 到達不可能オブジェクトを専用のcruft packにまとめる
# → GCでの到達可能性チェックが高速化される
```

### 2.2 packfileのフォーマット

```
┌─────────────────────────────────────────────────────┐
│  .git/objects/pack/                                  │
│                                                     │
│  pack-<SHA-1>.pack    ← オブジェクトデータ本体      │
│  pack-<SHA-1>.idx     ← オブジェクト検索インデックス │
│  pack-<SHA-1>.rev     ← リバースインデックス(v2.31+)│
│  pack-<SHA-1>.bitmap  ← ビットマップインデックス     │
│  pack-<SHA-1>.mtimes  ← cruft pack用タイムスタンプ  │
│                                                     │
│  ┌─── pack file 構造 ───────────────────────────┐   │
│  │ Header:                                      │   │
│  │   "PACK" (4 bytes magic)                     │   │
│  │   Version (4 bytes, = 2)                     │   │
│  │   Object Count (4 bytes)                     │   │
│  │                                              │   │
│  │ Objects:                                     │   │
│  │   [type + size + data] (zlib圧縮)            │   │
│  │   [OFS_DELTA + offset + delta]   ← delta圧縮│   │
│  │   [REF_DELTA + SHA-1 + delta]    ← delta圧縮│   │
│  │   ...                                       │   │
│  │                                              │   │
│  │ Trailer:                                     │   │
│  │   SHA-1 checksum (20 bytes)                  │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.3 packfileのオブジェクトタイプ

```
┌───────────────────────────────────────────────────────┐
│  packfile内のオブジェクトタイプ                         │
│                                                       │
│  非deltaオブジェクト:                                  │
│  ┌──────────┬───────┬──────────────────────────────┐  │
│  │ Type ID  │ 名称  │ 説明                          │  │
│  ├──────────┼───────┼──────────────────────────────┤  │
│  │ 1        │ commit│ コミットオブジェクト          │  │
│  │ 2        │ tree  │ ツリーオブジェクト            │  │
│  │ 3        │ blob  │ ブロブオブジェクト            │  │
│  │ 4        │ tag   │ アノテーテッドタグ            │  │
│  └──────────┴───────┴──────────────────────────────┘  │
│                                                       │
│  deltaオブジェクト:                                    │
│  ┌──────────┬───────────┬──────────────────────────┐  │
│  │ Type ID  │ 名称      │ 説明                      │  │
│  ├──────────┼───────────┼──────────────────────────┤  │
│  │ 6        │ OFS_DELTA │ オフセットでbaseを参照    │  │
│  │ 7        │ REF_DELTA │ SHA-1でbaseを参照         │  │
│  └──────────┴───────────┴──────────────────────────┘  │
│                                                       │
│  OFS_DELTA: packfile内のオフセットでbase objectを指す  │
│    → 同一packfile内のbaseのみ参照可能                  │
│    → アクセスが高速                                    │
│                                                       │
│  REF_DELTA: SHA-1ハッシュでbase objectを指す           │
│    → 別のpackfileのbaseも参照可能                      │
│    → thin packでネットワーク転送に使用                 │
└───────────────────────────────────────────────────────┘
```

### 2.4 delta圧縮

Packfile内では、類似するオブジェクト間の**差分（delta）**のみを保存して圧縮効率を高める。

```
┌────────────────────────────────────────────────┐
│  delta圧縮の例                                  │
│                                                │
│  blob v1: "Hello World\n" (12 bytes)           │
│  blob v2: "Hello World!\nGoodbye\n" (21 bytes) │
│                                                │
│  looseオブジェクト: 12 + 21 = 33 bytes          │
│                                                │
│  packfile (delta):                             │
│    base: blob v2 (21 bytes)                    │
│    delta: "copy 0-12, insert '!', ..." (~8 B)  │
│    合計: 21 + 8 = 29 bytes                     │
│                                                │
│  さらにzlib圧縮で実際にはもっと小さくなる       │
└────────────────────────────────────────────────┘
```

**重要**: Gitのdelta圧縮は**新しいバージョンをbaseに、古いバージョンをdeltaにする**。最新バージョンへのアクセスが最も高速になるよう設計されている。

### 2.5 delta命令セット

deltaデータは以下の2種類の命令で構成される。

```
┌────────────────────────────────────────────────────┐
│  delta命令セット                                    │
│                                                    │
│  1. コピー命令 (COPY):                             │
│     baseオブジェクトから指定範囲をコピー            │
│     形式: [1xxxxxxx] [offset bytes] [size bytes]   │
│     → base[offset..offset+size] をコピー           │
│                                                    │
│  2. 挿入命令 (INSERT):                             │
│     新しいデータを直接挿入                         │
│     形式: [0xxxxxxx] [data bytes]                  │
│     → x = データバイト数 (1-127)                   │
│                                                    │
│  例: "Hello World\n" → "Hello World!\nGoodbye\n"   │
│                                                    │
│  delta instructions:                               │
│    COPY offset=0 size=11    → "Hello World"        │
│    INSERT "!\n"             → "!\n"                │
│    INSERT "Goodbye\n"      → "Goodbye\n"          │
│                                                    │
│  delta header:                                     │
│    base_size = 12 (varint)                         │
│    result_size = 21 (varint)                       │
└────────────────────────────────────────────────────┘
```

```bash
# packfileの中身を確認
$ git verify-pack -v .git/objects/pack/pack-abc123.idx
SHA-1    type  size  size-in-pack  offset  depth  base-SHA-1
abc123   commit 234  156           12
def456   tree   89   67            168
789abc   blob   5280 1340          235
fedcba   blob   45   38            1575    1     789abc  ← deltaオブジェクト

# 各列の意味
# SHA-1:        オブジェクトのハッシュ
# type:         オブジェクトタイプ
# size:         展開後のオブジェクトサイズ
# size-in-pack: packfile内のサイズ（圧縮後）
# offset:       packfile内のバイトオフセット
# depth:        deltaチェーンの深さ
# base-SHA-1:   deltaのbaseオブジェクト
```

### 2.6 deltaチェーン

```
┌────────────────────────────────────────────────────┐
│  deltaチェーン（深さ制限: pack.depth, デフォルト50）│
│                                                    │
│  base object ←── delta 1 ←── delta 2 ←── delta 3  │
│  (完全なデータ)  (差分)      (差分)      (差分)    │
│                                                    │
│  delta 3 を読むには:                                │
│    base → delta 1適用 → delta 2適用 → delta 3適用  │
│                                                    │
│  深さが大きい → 圧縮率高い / 読み取り遅い          │
│  深さが小さい → 圧縮率低い / 読み取り速い          │
└────────────────────────────────────────────────────┘
```

```bash
# deltaチェーンの統計情報を確認
$ git verify-pack -v .git/objects/pack/pack-*.idx | \
    awk '/^[a-f0-9]/ && NF==7 {print $7}' | sort -n | tail -5
# → 最も深いdeltaチェーンの深さを確認

# チェーン深さの分布
$ git verify-pack -v .git/objects/pack/pack-*.idx | \
    awk '/^[a-f0-9]/ && NF==7 {depth[$7]++} END {for(d in depth) print d, depth[d]}' | sort -n
# 出力例:
# 1 4523
# 2 2341
# 3 1230
# ...
```

### 2.7 delta候補の選択アルゴリズム

```
┌────────────────────────────────────────────────────────┐
│  delta候補選択のアルゴリズム                            │
│                                                        │
│  Step 1: 全オブジェクトをソート                        │
│    - ファイル名（パスの最後の要素）                     │
│    - ファイルサイズ（降順）                             │
│    → 同名ファイルの異なるバージョンが隣接する          │
│                                                        │
│  Step 2: スライディングウィンドウで比較                 │
│    window = pack.window（デフォルト10）                 │
│    各オブジェクトについて:                              │
│      - ウィンドウ内の他オブジェクトとdeltaサイズを計算  │
│      - 最も小さいdeltaを生成するbaseを選択             │
│      - deltaサイズ > オブジェクトサイズならdelta化しない │
│                                                        │
│  例: window=4 の場合                                   │
│  [..., auth.js v5, auth.js v4, auth.js v3, auth.js v2] │
│                ^                                       │
│        このオブジェクトは左3つとdeltaを比較             │
│                                                        │
│  pack.window を大きくする → 圧縮率向上、処理時間増加   │
│  pack.windowMemory で使用メモリを制限可能              │
└────────────────────────────────────────────────────────┘
```

---

## 3. idxファイル（インデックス）

### 3.1 idxファイルの構造

```bash
# idxファイルの内容を確認
$ git verify-pack -v .git/objects/pack/pack-abc123.idx | head
```

```
┌─────────────────────────────────────────────┐
│  idx file v2 構造                            │
│                                             │
│  Header (8 bytes):                          │
│    Magic: "\377tOc" (4 bytes)              │
│    Version: 2 (4 bytes)                    │
│                                             │
│  Fanout Table (256 entries x 4 bytes)       │
│    fanout[0x55] = "55"で始まるSHA-1以下の   │
│                    オブジェクト累積数        │
│                                             │
│  SHA-1 Table                                │
│    全オブジェクトのSHA-1をソート済みで格納   │
│                                             │
│  CRC32 Table                                │
│    各オブジェクトのCRC32チェックサム         │
│                                             │
│  Offset Table (4 bytes per entry)           │
│    packfile内のオフセット（32bit）           │
│                                             │
│  Large Offset Table (8 bytes, 必要時のみ)   │
│    2GB超のpackfileでの64bitオフセット        │
│                                             │
│  Trailer:                                   │
│    Pack checksum (20 bytes)                 │
│    Idx checksum (20 bytes)                  │
└─────────────────────────────────────────────┘
```

### 3.2 オブジェクト検索のアルゴリズム

オブジェクト検索は**O(1)のfanoutテーブル + O(log n)の二分探索**で高速に行われる。

```
┌────────────────────────────────────────────────────────┐
│  オブジェクト検索の手順                                  │
│                                                        │
│  SHA-1 = "55 7d b0 3d e9 97 ..." を検索する場合        │
│                                                        │
│  Step 1: Fanout Table から範囲を特定 (O(1))            │
│    fanout[0x54] = 1200  (0x54以下のオブジェクト数)      │
│    fanout[0x55] = 1215  (0x55以下のオブジェクト数)      │
│    → SHA-1テーブルの index 1200〜1214 を検索           │
│                                                        │
│  Step 2: SHA-1 Table で二分探索 (O(log 15))            │
│    index 1200〜1214 の15エントリ内で二分探索            │
│    → index 1207 に一致するSHA-1を発見                  │
│                                                        │
│  Step 3: Offset Table から位置を取得 (O(1))            │
│    offset[1207] = 45678 (packfile内のバイト位置)        │
│                                                        │
│  Step 4: packfile の offset 45678 からデータを読み取り  │
│                                                        │
│  合計計算量: O(1) + O(log n) = O(log n)                │
│  実際には fanout で範囲が 1/256 に狭まるため非常に高速  │
└────────────────────────────────────────────────────────┘
```

### 3.3 リバースインデックス（.rev）

```bash
# リバースインデックスの生成（Git 2.31+）
$ git config pack.writeReverseIndex true
$ git repack -a -d

# リバースインデックスの用途:
# - packfile内のオフセットからSHA-1を逆引き
# - ネットワーク転送時のオブジェクト列挙に使用
# - .revファイルがないと、idxファイル全体をソートし直す必要がある
```

```
┌────────────────────────────────────────────────────┐
│  リバースインデックスの必要性                        │
│                                                    │
│  idx (正引き): SHA-1 → packfile offset             │
│  rev (逆引き): packfile offset → SHA-1             │
│                                                    │
│  .revがない場合:                                   │
│    idxの全エントリをoffsetでソートし直す必要あり    │
│    → メモリと時間を消費                            │
│                                                    │
│  .revがある場合:                                   │
│    直接逆引きが可能                                │
│    → reachability bitmap計算等が高速化             │
└────────────────────────────────────────────────────┘
```

---

## 4. ガベージコレクション（GC）

### 4.1 GCの基本

```bash
# GCの実行
$ git gc
# → repack + reflog expire + prune + rerere gc

# 積極的なGC（より強力な圧縮）
$ git gc --aggressive
# → window=250, depth=250 で再圧縮（時間がかかる）

# 自動GCの設定
$ git config gc.auto 6700          # looseオブジェクト数の閾値
$ git config gc.autoPackLimit 50   # packfile数の閾値

# 自動GCを明示的に無効化
$ git config gc.auto 0
# → CI/CD環境やパフォーマンスが重要な場合
```

### 4.2 GCのプロセス

```
┌─────────────────────────────────────────────────────┐
│  git gc の実行フロー                                 │
│                                                     │
│  1. git prune --expire=2weeks                       │
│     → 到達不可能なlooseオブジェクトの削除            │
│     → 2週間以内のものは保護（安全マージン）          │
│                                                     │
│  2. git reflog expire                               │
│     → 期限切れのreflogエントリを削除                │
│     → 到達可能: 90日 / 到達不可能: 30日              │
│                                                     │
│  3. git repack -d -l                                │
│     → looseオブジェクトをpackfileに統合              │
│     → 古いpackfileを新しいpackfileにマージ           │
│                                                     │
│  4. git rerere gc                                   │
│     → 期限切れのrerere記録を削除                    │
│                                                     │
│  5. git pack-refs --all --prune                     │
│     → looseなrefをpacked-refsに統合                 │
│                                                     │
│  6. git commit-graph write --reachable (2.34+)      │
│     → コミットグラフの生成・更新                    │
└─────────────────────────────────────────────────────┘
```

### 4.3 GCの各ステップの詳細

```bash
# Step 1: prune（到達不可能オブジェクトの削除）
$ git prune --dry-run           # 削除対象の確認（実行しない）
$ git prune --expire=2.weeks.ago  # デフォルトの猶予期間
$ git prune --expire=now          # 即座に削除（危険！）

# pruneの猶予期間の設定
$ git config gc.pruneExpire "2 weeks ago"
# → 2週間より新しい到達不可能オブジェクトは保護
# → 進行中の操作（rebase等）で一時的に到達不可能になったオブジェクトを保護

# Step 2: reflog expire
$ git reflog expire --all
# → 全refのreflogを期限切れ処理

$ git config gc.reflogExpire 90.days      # 到達可能エントリの保持期間
$ git config gc.reflogExpireUnreachable 30.days  # 到達不可能エントリの保持期間

# Step 3: repack
$ git repack -d -l --keep-unreachable
# -l: ローカルオブジェクトのみ（alternatesは除外）
# --keep-unreachable: 到達不可能オブジェクトもpackに含める

# Step 5: pack-refs
$ git pack-refs --all --prune
# → .git/refs/heads/*, .git/refs/tags/* の個別ファイルを
#   .git/packed-refs に統合
# → ファイル数を削減、refの読み取りを高速化
```

### 4.4 到達可能性の判定

```
┌──────────────────────────────────────────────────┐
│  到達可能性（reachability）の判定                  │
│                                                  │
│  到達可能なオブジェクト（GCで削除されない）:      │
│  - refs/heads/* から辿れるcommit/tree/blob       │
│  - refs/tags/* から辿れるオブジェクト             │
│  - refs/remotes/* から辿れるオブジェクト          │
│  - reflog から辿れるオブジェクト                  │
│  - refs/stash から辿れるオブジェクト              │
│  - FETCH_HEAD, MERGE_HEAD 等の特殊ref            │
│  - worktreeのHEADとインデックス                  │
│                                                  │
│  到達不可能（unreachable）:                       │
│  - amend前の旧commit（reflog期限切れ後）         │
│  - 削除されたブランチのcommit                    │
│  - rebase前の旧commit                            │
│  - reset --hard で捨てたcommit                   │
│  - filter-repoで書き換え前のcommit               │
│                                                  │
│  refs ──→ commit ──→ tree ──→ blob               │
│             │                                    │
│             ▼                                    │
│           parent commit ──→ ...                  │
│  （全て到達可能）                                 │
│                                                  │
│  [orphaned commit] ← どのrefからも辿れない       │
│  （到達不可能 → GC対象）                         │
└──────────────────────────────────────────────────┘
```

### 4.5 cruft packの仕組み（Git 2.37+）

```bash
# cruft packの有効化
$ git config gc.cruftPacks true
$ git gc

# cruft packの動作:
# 1. 到達可能オブジェクト → 通常のpackfileに格納
# 2. 到達不可能オブジェクト → cruft packに格納
# 3. cruft pack内の各オブジェクトにmtime（最終参照時刻）を記録
# 4. 猶予期間（gc.pruneExpire）を過ぎたオブジェクトのみ削除

# cruft packのメリット:
# - pruneの高速化（到達可能性チェックの範囲が狭まる）
# - 到達不可能オブジェクトの管理が効率的
# - 大規模リポジトリでのGC時間が大幅に短縮
```

```
┌────────────────────────────────────────────────────┐
│  cruft pack のアーキテクチャ                         │
│                                                    │
│  従来のGC:                                         │
│  ┌──────────────────────────────────────────┐     │
│  │  1つのpackfileに全オブジェクト           │     │
│  │  [到達可能] [到達可能] [不可能] [可能]   │     │
│  │  → prune時に全オブジェクトをスキャン     │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  cruft pack:                                       │
│  ┌──────────────────────┐  ┌──────────────────┐   │
│  │  main pack           │  │  cruft pack      │   │
│  │  [到達可能のみ]      │  │  [不可能のみ]    │   │
│  │                      │  │  + mtimes file   │   │
│  └──────────────────────┘  └──────────────────┘   │
│  → prune時はcruft packのmtimesだけチェック         │
│  → 到達可能性の再計算が不要                        │
└────────────────────────────────────────────────────┘
```

---

## 5. 実践: リポジトリの最適化

### 5.1 リポジトリサイズの調査

```bash
# リポジトリ全体のサイズ
$ du -sh .git
248M    .git

# オブジェクトの統計
$ git count-objects -vH
count: 234
size: 1.20 MiB
in-pack: 89432
packs: 3
size-pack: 245.80 MiB

# 最も大きいオブジェクトを特定
$ git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | sort -k3 -n -r \
  | head -10
blob abc123... 52428800 data/huge-dataset.csv
blob def456... 10485760 assets/video.mp4

# オブジェクトタイプ別の統計
$ git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype)' \
  | sort | uniq -c | sort -rn
  45231 blob
  12345 tree
   8765 commit
    234 tag
```

### 5.2 巨大ファイルの履歴からの削除

```bash
# git-filter-repo を使用（推奨）
$ pip install git-filter-repo
$ git filter-repo --path data/huge-dataset.csv --invert-paths

# 特定サイズ以上のblobを全て削除
$ git filter-repo --strip-blobs-bigger-than 10M

# パスの書き換え（ディレクトリの移動）
$ git filter-repo --path-rename old/path/:new/path/

# BFG Repo-Cleaner を使用
$ java -jar bfg.jar --strip-blobs-bigger-than 10M
$ git reflog expire --expire=now --all
$ git gc --prune=now --aggressive

# filter-repo後の後処理
$ git reflog expire --expire=now --all
$ git gc --prune=now --aggressive
# → 全ての到達不可能オブジェクトを即座に削除
# → リポジトリサイズが大幅に縮小
```

```bash
# 巨大ファイルを見つけるための追加コマンド
# packfile内の大きなオブジェクトを直接確認
$ git verify-pack -v .git/objects/pack/pack-*.idx \
  | sort -k3 -n -r | head -20

# 特定のblobがどのcommitで追加されたかを特定
$ git log --all --find-object=abc123def456
# → abc123def456 というblobを含むcommitを表示

# Git LFS への移行を検討
$ git lfs install
$ git lfs track "*.psd" "*.mp4" "*.zip"
$ git add .gitattributes
# → 以降の大きなバイナリファイルはLFSで管理
```

### 5.3 repackの最適化パラメータ

```bash
# ウィンドウサイズとdelta深さの調整
$ git repack -a -d --window=250 --depth=50

# マルチパックインデックス（Git 2.34+）
$ git multi-pack-index write
$ git multi-pack-index verify

# コミットグラフの生成（高速化）
$ git commit-graph write --reachable
# → .git/objects/info/commit-graph に生成
# → git log, git merge-base等が大幅に高速化

# コミットグラフのチェーン管理
$ git commit-graph write --reachable --split
# → 増分更新が可能（全体を再生成せずに済む）
```

| パラメータ          | デフォルト | 説明                                    |
|---------------------|------------|------------------------------------------|
| `pack.window`       | 10         | delta候補を探すウィンドウサイズ           |
| `pack.depth`        | 50         | deltaチェーンの最大深さ                  |
| `pack.threads`      | CPU数      | repack時の並列スレッド数                 |
| `pack.windowMemory`  | 0 (無制限) | ウィンドウのメモリ上限                   |
| `pack.deltaCacheSize`| 256MB     | deltaキャッシュのメモリ上限              |
| `gc.auto`           | 6700       | 自動GCのlooseオブジェクト閾値            |
| `gc.autoPackLimit`  | 50         | 自動GCのpackfile数閾値                   |
| `gc.cruftPacks`     | false      | cruft packの有効化                       |
| `gc.pruneExpire`    | 2 weeks    | 到達不可能オブジェクトの猶予期間         |

### 5.4 ビットマップインデックス

```bash
# ビットマップの生成
$ git repack -a -d --write-bitmap-index

# ビットマップの効果:
# - git clone/fetchの高速化
# - 到達可能性チェックの高速化
# - reachability bitmapで各commitの到達可能オブジェクトを記録

# ビットマップの仕組み
# - 各commitに対して、到達可能な全オブジェクトを1ビットずつ記録
# - ビット演算（AND, OR）で到達可能性を高速に計算
# - clone時に「どのオブジェクトを送るか」の判定が高速化
```

```
┌────────────────────────────────────────────────────────┐
│  ビットマップインデックスの動作                          │
│                                                        │
│  パックファイル内のオブジェクト:                        │
│  index: 0    1    2    3    4    5    6    7            │
│  obj:   C1   C2   C3   T1   T2   B1   B2   B3         │
│                                                        │
│  commit C1 のビットマップ:                              │
│  [1, 0, 0, 1, 0, 1, 1, 0]                             │
│  → C1から到達可能: C1, T1, B1, B2                      │
│                                                        │
│  commit C2 のビットマップ:                              │
│  [1, 1, 0, 1, 1, 1, 1, 1]                             │
│  → C2から到達可能: C1, C2, T1, T2, B1, B2, B3         │
│                                                        │
│  "C2にあってC1にないオブジェクト" =                     │
│  bitmap(C2) & ~bitmap(C1) = [0,1,0,0,1,0,0,1]         │
│  → C2, T2, B3                                         │
│  → fetchで送るべきオブジェクトが瞬時に判定             │
└────────────────────────────────────────────────────────┘
```

---

## 6. shallow cloneとpartial clone

### 6.1 shallow clone

```bash
# 最新のN件のコミットのみ取得
$ git clone --depth=1 https://github.com/user/repo.git
# → .git/shallow ファイルにshallow境界を記録

# 履歴を後から深化
$ git fetch --deepen=10
$ git fetch --unshallow    # 全履歴を取得

# shallowの制限事項
# - git log で全履歴を辿れない
# - git bisect が制限される
# - git merge-base が不正確になる可能性がある
# - git push --all が期待通り動かない場合がある

# shallow境界の確認
$ cat .git/shallow
abc123def456789abcdef1234567890abcdef1234
# → この行のcommit以降の履歴のみ保持
```

### 6.2 partial clone

```bash
# blobを取得せずにクローン（Git 2.22+）
$ git clone --filter=blob:none https://github.com/user/repo.git
# → checkout時に必要なblobをオンデマンドで取得

# 一定サイズ以上のblobを除外
$ git clone --filter=blob:limit=1m https://github.com/user/repo.git

# treeも除外（最も軽量だが機能制限あり）
$ git clone --filter=tree:0 https://github.com/user/repo.git

# 複合フィルター（Git 2.27+）
$ git clone --filter=combine:blob:none+tree:0 https://github.com/user/repo.git
```

```
┌────────────────────────────────────────────────┐
│  clone方式の比較                                │
│                                                │
│  full clone:     [commit][tree][blob] 全取得   │
│                  サイズ: 大 / 速度: 遅          │
│                  → 全操作がオフラインで可能     │
│                                                │
│  shallow clone:  最新Nコミットの全オブジェクト  │
│                  サイズ: 中 / 古い履歴なし      │
│                  → CI/CDの一時的な使用に最適    │
│                                                │
│  partial clone:  commit+tree取得、blobは遅延    │
│  (blobless)      サイズ: 小 / checkout時に取得  │
│                  → 開発者の日常使用に最適       │
│                                                │
│  partial clone:  commitのみ取得                 │
│  (treeless)      サイズ: 極小 / 制限多い        │
│                  → CIのビルド不要なジョブに     │
└────────────────────────────────────────────────┘
```

### 6.3 promisorリモートとオンデマンド取得

```bash
# partial cloneではリモートが "promisor" として登録される
$ git config remote.origin.promisor true
$ git config remote.origin.partialclonefilter "blob:none"

# オンデマンド取得の動作
$ git checkout feature-branch
# → 必要なblobが自動的にoriginから取得される

# 手動でのオブジェクト取得
$ git fetch origin --filter=blob:none
# → 新しいcommitとtreeのみ取得

# 全blobの事前取得（オフライン作業前に）
$ git fetch --unshallow
# → 全オブジェクトを取得してfull cloneに変換
```

---

## 7. commit-graphとmulti-pack-index

### 7.1 commit-graph

```bash
# commit-graphの生成
$ git commit-graph write --reachable
# → .git/objects/info/commit-graph に生成

# 効果:
# - git log の祖先関係の計算が高速化（O(1)でparent参照）
# - git merge-base の計算が高速化
# - generation number による効率的なDAGトラバーサル

# commit-graphの内容確認
$ git commit-graph verify
```

```
┌────────────────────────────────────────────────────────┐
│  commit-graph の構造と効果                              │
│                                                        │
│  通常のcommit参照:                                     │
│    commit object → parent SHA-1 → parent object読込    │
│    → 毎回packfileからオブジェクトを読み取る            │
│    → 大量のI/O発生                                    │
│                                                        │
│  commit-graph使用時:                                   │
│    commit-graph ファイル内に以下を格納:                 │
│    ┌─────────────────────────────────────┐             │
│    │ commit SHA-1                        │             │
│    │ tree SHA-1                          │             │
│    │ parent indices (graph内のindex)     │             │
│    │ generation number                   │             │
│    │ commit date                         │             │
│    └─────────────────────────────────────┘             │
│    → 固定サイズレコードで O(1) アクセス               │
│    → packfileの読み取りが不要                         │
│                                                        │
│  generation number の効果:                              │
│    gen(C) = 1 + max(gen(parents))                      │
│    → gen(A) < gen(B) なら AはBの子孫ではない           │
│    → 到達可能性チェックを早期に打ち切れる             │
│                                                        │
│  Linux kernel での効果（参考値）:                       │
│    git log --oneline: 2.3秒 → 0.3秒 (7.6倍高速)      │
│    git merge-base:    1.5秒 → 0.1秒 (15倍高速)        │
└────────────────────────────────────────────────────────┘
```

### 7.2 multi-pack-index（MIDX）

```bash
# multi-pack-indexの生成
$ git multi-pack-index write

# multi-pack-indexの検証
$ git multi-pack-index verify

# 効果:
# - 複数のpackfileを横断する統合インデックス
# - オブジェクト検索が1回のインデックス参照で完了
# - repackせずに検索性能を改善

# MIDXベースのrepack（Git 2.38+）
$ git multi-pack-index repack --batch-size=100M
# → 指定サイズ単位でpackfileを再構成
# → 全体をrepackするより高速
```

```
┌────────────────────────────────────────────────────┐
│  multi-pack-index のアーキテクチャ                   │
│                                                    │
│  従来:                                             │
│  pack-A.idx ───→ pack-A.pack                       │
│  pack-B.idx ───→ pack-B.pack                       │
│  pack-C.idx ───→ pack-C.pack                       │
│  オブジェクト検索: A.idx → 見つからない             │
│                   → B.idx → 見つからない            │
│                   → C.idx → 発見！                  │
│  最悪ケース: pack数 × O(log n)                      │
│                                                    │
│  multi-pack-index:                                 │
│  multi-pack-index ───→ pack-A.pack                 │
│          │──→ pack-B.pack                          │
│          └──→ pack-C.pack                          │
│  オブジェクト検索: MIDX → 直接発見！               │
│  常に O(log n)                                     │
└────────────────────────────────────────────────────┘
```

---

## 8. リポジトリの健全性チェックと修復

### 8.1 fsckによるチェック

```bash
# リポジトリの整合性チェック
$ git fsck
# → 壊れたオブジェクト、到達不可能オブジェクト、不正な参照を検出

$ git fsck --full
# → packfile内のオブジェクトも含めて完全チェック

$ git fsck --unreachable
# → 到達不可能オブジェクトの一覧を表示

$ git fsck --dangling
# → 「dangling」（どこからも参照されない）オブジェクトを表示
```

### 8.2 壊れたpackfileの修復

```bash
# packfileの検証
$ git verify-pack -v .git/objects/pack/pack-abc123.idx

# 壊れたpackfileの修復手順
# Step 1: 壊れたpackfileを特定
$ git fsck
# error: packfile .git/objects/pack/pack-abc123.pack index CRC mismatch

# Step 2: リモートからの回復（最も安全）
$ git clone --mirror origin-url backup-repo
# → リモートの正常なデータで復旧

# Step 3: packfileの再構築（リモートがない場合）
$ mv .git/objects/pack/pack-abc123.* /tmp/
$ git unpack-objects < /tmp/pack-abc123.pack
# → packをlooseオブジェクトに展開（壊れた部分はエラー）
$ git repack -a -d
# → 残ったオブジェクトを新しいpackfileに再構築
```

### 8.3 lost-foundによるオブジェクト復旧

```bash
# 到達不可能なオブジェクトの確認
$ git fsck --lost-found
# → .git/lost-found/commit/ と .git/lost-found/other/ に復旧

# 復旧したcommitの確認
$ ls .git/lost-found/commit/
$ git show abc123def456  # 復旧したcommitの内容を確認
$ git branch recovered abc123def456  # ブランチとして復元
```

---

## 9. 大規模リポジトリの運用ベストプラクティス

### 9.1 性能チューニングの推奨設定

```bash
# 大規模リポジトリ向けの推奨設定
$ git config core.preloadIndex true         # インデックスの並列読み込み
$ git config core.fsmonitor true            # ファイルシステムモニター（macOS/Windows）
$ git config core.untrackedCache true       # 未追跡ファイルのキャッシュ
$ git config feature.manyFiles true         # 多ファイルリポジトリ最適化

# packfile関連の最適化
$ git config pack.writeReverseIndex true    # リバースインデックスの有効化
$ git config gc.cruftPacks true             # cruft packの有効化
$ git config fetch.writeCommitGraph true    # fetch時にcommit-graphを更新

# maintenance（Git 2.31+）
$ git maintenance register
# → 定期的な自動メンテナンスを設定
# → prefetch, loose-objects, incremental-repack, pack-refs, commit-graph等を実行
```

### 9.2 git maintenanceの活用

```bash
# maintenanceの設定
$ git maintenance start
# → バックグラウンドで定期的にメンテナンスを実行

# 実行されるタスク一覧
$ git maintenance run --task=prefetch         # リモートの事前取得
$ git maintenance run --task=loose-objects    # looseオブジェクトのpack化
$ git maintenance run --task=incremental-repack  # 増分repack
$ git maintenance run --task=pack-refs        # ref圧縮
$ git maintenance run --task=commit-graph     # commit-graph更新
$ git maintenance run --task=gc               # GC

# maintenanceの設定ファイル
$ git config maintenance.auto false
$ git config maintenance.strategy incremental
# → incremental: 小さな操作を頻繁に実行
# → gc: 従来のgit gc方式
```

---

## 10. アンチパターン

### アンチパターン1: `git gc --aggressive`の頻繁な実行

```bash
# NG: 毎日git gc --aggressiveを実行
$ crontab -e
0 3 * * * cd /repo && git gc --aggressive
# → 数時間CPUを占有、実際の効果は初回以降ほとんどない

# OK: 通常のgit gcで十分、aggressiveは大規模な履歴整理後のみ
$ git gc                           # 通常のGC
$ git gc --aggressive              # filter-repo等の後に一度だけ
```

**理由**: `--aggressive`はwindow=250, depth=250で再圧縮を行うため非常に遅い。通常の`git gc`（自動実行含む）で十分な圧縮が得られる。`git maintenance`の使用が推奨される。

### アンチパターン2: packfileの手動削除

```bash
# NG: packfileを直接削除
$ rm .git/objects/pack/pack-abc123.*
# → オブジェクトが見つからなくなりリポジトリが破損

# OK: Gitコマンドを通じて操作
$ git repack -a -d        # 全オブジェクトを再パック
$ git prune-packed         # pack済みlooseオブジェクトの削除
```

**理由**: packfileはidxと対になっており、他のpackfileとのオブジェクト重複管理もGitが内部的に行っている。手動削除はデータ損失に直結する。

### アンチパターン3: gc.pruneExpire=nowの常用

```bash
# NG: 猶予期間なしでpruneを常時実行
$ git config gc.pruneExpire now
$ git gc
# → 進行中のrebase/mergeのオブジェクトが削除されるリスク

# OK: デフォルトの猶予期間を維持
$ git config gc.pruneExpire "2 weeks ago"
# → 2週間の猶予で安全にGC
# → filter-repo等の後のみ一時的にnowを使用
```

**理由**: `gc.pruneExpire=now`は進行中の操作（rebase、merge等）で一時的に到達不可能になったオブジェクトも即座に削除する。2週間の猶予期間は、これらの一時的な状態を保護するための安全マージンである。

### アンチパターン4: 巨大バイナリファイルを直接コミット

```bash
# NG: 大きなバイナリファイルをGitに直接コミット
$ git add assets/design-mockup-v3.psd  # 50MB
$ git commit -m "update design"
# → 履歴に50MBのblobが永久に残る
# → バージョンごとに差分が取れない（バイナリは差分圧縮の効果が低い）
# → リポジトリサイズが急速に肥大化

# OK: Git LFS を使用
$ git lfs track "*.psd" "*.mp4" "*.zip"
$ git add .gitattributes
$ git add assets/design-mockup-v3.psd
$ git commit -m "update design"
# → 実際のファイルはLFSサーバーに保存
# → Gitリポジトリにはポインタファイルのみ（数百バイト）
```

---

## 11. FAQ

### Q1. `git gc`はいつ実行すべきか？

**A1.** 基本的には**手動実行の必要はありません**。Gitは`gc.auto`の設定（デフォルト6700個のlooseオブジェクト）に基づいて自動的にGCを実行します。手動実行が有効な場面は、`filter-repo`等で大量のオブジェクトを削除した後や、リポジトリサイズを明示的に削減したい場合です。Git 2.31以降では`git maintenance`の使用が推奨されます。

### Q2. packfileが複数あると性能に影響するか？

**A2.** はい、影響します。オブジェクト検索時に各packfileのidxを順番に調べるため、packfile数が多いと遅くなります。Git 2.34以降の**multi-pack-index**を使うと、複数のpackfileを横断する統合インデックスが作成され、この問題が緩和されます。`gc.autoPackLimit`（デフォルト50）を超えるとGCが自動実行されます。

### Q3. delta圧縮のbaseオブジェクトはどのように選ばれるか？

**A3.** Gitはオブジェクトをファイル名とサイズでソートし、**類似するオブジェクト同士が近くなるように**配置します。次に`pack.window`（デフォルト10）の範囲内で各オブジェクトペアのdeltaサイズを計算し、最も小さいdeltaを生成するbaseを選択します。ファイル名が同じで古いバージョンのblobがdelta baseに選ばれやすい設計です。

### Q4. commit-graphは全てのリポジトリで有効にすべきか？

**A4.** はい、**特にデメリットはないため有効にすることを推奨します**。commit-graphはディスク容量をごく少量消費するだけで、`git log`、`git merge-base`、`git rev-list`等のコマンドを大幅に高速化します。Git 2.34以降では`git gc`実行時に自動的に生成されるようになっています。

```bash
# commit-graphの有効化
$ git config fetch.writeCommitGraph true
$ git commit-graph write --reachable
```

### Q5. partial cloneのリポジトリでオフライン作業は可能か？

**A5.** 限定的に可能です。既にcheckout済みのファイルは操作できますが、新しいブランチへの切り替えや、まだ取得していないblobへのアクセスにはネットワーク接続が必要です。オフライン作業が予想される場合は、事前に必要なオブジェクトを取得しておくか、`git fetch --unshallow`で完全なcloneに変換してください。

### Q6. リポジトリサイズの目安はどのくらいか？

**A6.** 一般的な目安は以下の通りです。

| サイズ          | 評価                                          |
|-----------------|-----------------------------------------------|
| 〜100MB         | 小規模、問題なし                               |
| 100MB〜1GB      | 中規模、partial cloneの検討                    |
| 1GB〜5GB        | 大規模、LFS + partial clone推奨                |
| 5GB以上         | 超大規模、monorepo戦略の見直しを検討           |

---

## まとめ

| 概念             | 要点                                                          |
|------------------|---------------------------------------------------------------|
| looseオブジェクト| 個別ファイルとしてzlib圧縮、ファイル数が増えると非効率        |
| packfile         | 複数オブジェクトを1ファイルに統合、delta圧縮で高効率          |
| delta圧縮       | 類似オブジェクト間の差分のみ保存、新→旧の方向でチェーン構築  |
| idxファイル      | packfile内オブジェクトの高速検索用インデックス                |
| GC               | 到達不可能オブジェクトの削除 + repack + reflog整理            |
| 到達可能性       | refs/reflogから辿れるオブジェクトはGCで保護される             |
| partial clone    | blobやtreeの遅延取得でclone時間とサイズを大幅削減             |
| commit-graph     | commitの親関係をキャッシュし、log/merge-baseを高速化          |
| multi-pack-index | 複数packfileの統合インデックスで検索を高速化                 |
| cruft pack       | 到達不可能オブジェクトの効率的な管理                         |
| git maintenance  | 定期的な自動メンテナンスでリポジトリを最適状態に維持         |

---

## 次に読むべきガイド

- [Gitオブジェクトモデル](./00-git-object-model.md) — packfileが格納するオブジェクトの基礎
- [Worktree/Submodule](../01-advanced-git/01-worktree-submodule.md) — 複数作業ディレクトリとGCの関係
- [Git Hooks](../01-advanced-git/03-hooks-automation.md) — GCを含む自動化の設計

---

## 参考文献

1. **Pro Git Book** — "Git Internals - Packfiles" https://git-scm.com/book/en/v2/Git-Internals-Packfiles
2. **Git公式ドキュメント** — `git-gc`, `git-repack`, `git-prune`, `git-verify-pack` https://git-scm.com/docs
3. **GitHub Engineering** — "Scaling monorepo maintenance" https://github.blog/2021-04-29-scaling-monorepo-maintenance/
4. **Git pack format specification** — Documentation/gitformat-pack.txt https://github.com/git/git/blob/master/Documentation/gitformat-pack.txt
5. **Derrick Stolee** — "Supercharging the Git Commit Graph" https://devblogs.microsoft.com/devops/supercharging-the-git-commit-graph/
6. **Git公式ドキュメント** — `git-multi-pack-index`, `git-commit-graph`, `git-maintenance` https://git-scm.com/docs
