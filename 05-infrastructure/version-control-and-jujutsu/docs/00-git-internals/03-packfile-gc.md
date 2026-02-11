# Packfile/GC

> Gitのdelta圧縮とPackfileフォーマット、ガベージコレクション（GC）の仕組みを理解し、リポジトリサイズの最適化と性能チューニングの方法を習得する。

## この章で学ぶこと

1. **Packfileの構造とdelta圧縮** — looseオブジェクトがどのように効率的に圧縮されるか
2. **ガベージコレクション（GC）の仕組み** — 到達不可能オブジェクトの検出と削除プロセス
3. **リポジトリ最適化の実践** — 肥大化したリポジトリのサイズ削減と性能改善テクニック

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

### 1.2 looseオブジェクトの問題点

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
```

### 2.2 packfileのフォーマット

```
┌─────────────────────────────────────────────────────┐
│  .git/objects/pack/                                  │
│                                                     │
│  pack-<SHA-1>.pack    ← オブジェクトデータ本体      │
│  pack-<SHA-1>.idx     ← オブジェクト検索インデックス │
│  pack-<SHA-1>.rev     ← リバースインデックス(v2.31+)│
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

### 2.3 delta圧縮

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

```bash
# packfileの中身を確認
$ git verify-pack -v .git/objects/pack/pack-abc123.idx
SHA-1    type  size  size-in-pack  offset  depth  base-SHA-1
abc123   commit 234  156           12
def456   tree   89   67            168
789abc   blob   5280 1340          235
fedcba   blob   45   38            1575    1     789abc  ← deltaオブジェクト
```

### 2.4 deltaチェーン

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

---

## 3. idxファイル（インデックス）

```bash
# idxファイルの内容を確認
$ git verify-pack -v .git/objects/pack/pack-abc123.idx | head
```

```
┌─────────────────────────────────────────────┐
│  idx file v2 構造                            │
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
└─────────────────────────────────────────────┘
```

オブジェクト検索は**O(1)のfanoutテーブル + O(log n)の二分探索**で高速に行われる。

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
└─────────────────────────────────────────────────────┘
```

### 4.3 到達可能性の判定

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
│                                                  │
│  到達不可能（unreachable）:                       │
│  - amend前の旧commit（reflog期限切れ後）         │
│  - 削除されたブランチのcommit                    │
│  - rebase前の旧commit                            │
│  - reset --hard で捨てたcommit                   │
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
```

### 5.2 巨大ファイルの履歴からの削除

```bash
# git-filter-repo を使用（推奨）
$ pip install git-filter-repo
$ git filter-repo --path data/huge-dataset.csv --invert-paths

# BFG Repo-Cleaner を使用
$ java -jar bfg.jar --strip-blobs-bigger-than 10M
$ git reflog expire --expire=now --all
$ git gc --prune=now --aggressive
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
```

| パラメータ          | デフォルト | 説明                                    |
|---------------------|------------|------------------------------------------|
| `pack.window`       | 10         | delta候補を探すウィンドウサイズ           |
| `pack.depth`        | 50         | deltaチェーンの最大深さ                  |
| `pack.threads`      | CPU数      | repack時の並列スレッド数                 |
| `pack.windowMemory`  | 0 (無制限) | ウィンドウのメモリ上限                   |
| `gc.auto`           | 6700       | 自動GCのlooseオブジェクト閾値            |
| `gc.autoPackLimit`  | 50         | 自動GCのpackfile数閾値                   |

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
```

```
┌────────────────────────────────────────────────┐
│  clone方式の比較                                │
│                                                │
│  full clone:     [commit][tree][blob] 全取得   │
│                  サイズ: 大 / 速度: 遅          │
│                                                │
│  shallow clone:  最新Nコミットの全オブジェクト  │
│                  サイズ: 中 / 古い履歴なし      │
│                                                │
│  partial clone:  commit+tree取得、blobは遅延    │
│  (blobless)      サイズ: 小 / checkout時に取得  │
│                                                │
│  partial clone:  commitのみ取得                 │
│  (treeless)      サイズ: 極小 / 制限多い        │
└────────────────────────────────────────────────┘
```

---

## 7. アンチパターン

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

**理由**: `--aggressive`はwindow=250, depth=250で再圧縮を行うため非常に遅い。通常の`git gc`（自動実行含む）で十分な圧縮が得られる。

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

---

## 8. FAQ

### Q1. `git gc`はいつ実行すべきか？

**A1.** 基本的には**手動実行の必要はありません**。Gitは`gc.auto`の設定（デフォルト6700個のlooseオブジェクト）に基づいて自動的にGCを実行します。手動実行が有効な場面は、`filter-repo`等で大量のオブジェクトを削除した後や、リポジトリサイズを明示的に削減したい場合です。

### Q2. packfileが複数あると性能に影響するか？

**A2.** はい、影響します。オブジェクト検索時に各packfileのidxを順番に調べるため、packfile数が多いと遅くなります。Git 2.34以降の**multi-pack-index**を使うと、複数のpackfileを横断する統合インデックスが作成され、この問題が緩和されます。`gc.autoPackLimit`（デフォルト50）を超えるとGCが自動実行されます。

### Q3. delta圧縮のbaseオブジェクトはどのように選ばれるか？

**A3.** Gitはオブジェクトをファイル名とサイズでソートし、**類似するオブジェクト同士が近くなるように**配置します。次に`pack.window`（デフォルト10）の範囲内で各オブジェクトペアのdeltaサイズを計算し、最も小さいdeltaを生成するbaseを選択します。ファイル名が同じで古いバージョンのblobがdelta baseに選ばれやすい設計です。

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
