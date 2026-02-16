# Jujutsu応用

> Jujutsuのrevset（リビジョンセット式）、テンプレート言語、Git連携の高度な設定を習得し、複雑なリポジトリ操作と効率的なワークフローを実現する。

## この章で学ぶこと

1. **revsetクエリ言語** — リビジョンの柔軟な選択・フィルタリング構文
2. **テンプレート言語** — ログ出力やコミット表示のカスタマイズ
3. **Git連携の高度な設定** — fetch, push, colocated repoの詳細な運用
4. **高度なワークフロー** — スタックドPR、absorb、split、parallelizeの実践活用
5. **Operation Log** — 操作履歴の追跡とundo/redo
6. **設定の高度なカスタマイズ** — revset-aliases、テンプレート、difftoolの詳細設定

---

## 1. revset — リビジョン選択クエリ

### 1.1 基本構文

revsetはコミット（リビジョン）の集合を表現するクエリ言語で、`jj log -r`、`jj rebase`、`jj diff`等のほぼ全てのコマンドで使用できる。

```bash
# 単一revision
$ jj log -r @                    # working copy
$ jj log -r @-                   # working copyの親
$ jj log -r @--                  # working copyの祖父
$ jj log -r @---                 # 3世代前
$ jj log -r rlvkpntz             # change IDで指定
$ jj log -r abc12345             # commit IDで指定
$ jj log -r main                 # ブックマーク名で指定
$ jj log -r main@origin          # リモートブックマーク
$ jj log -r 'v1.0.0'             # タグで指定

# 範囲指定
$ jj log -r 'main..@'            # mainから@までの間のcommit
$ jj log -r 'main..'             # mainの子孫（main自身は含まない）
$ jj log -r '..main'             # mainの祖先（main自身を含む）
$ jj log -r 'root()..main'       # ルートからmainまでの全commit

# 親・子の参照
$ jj log -r '@-'                 # @の第1の親
$ jj log -r '@+'                 # @の子（複数ある場合はエラー）
```

```
┌─────────────────────────────────────────────────────┐
│  revset 基本構文の図解                               │
│                                                     │
│     root()                                          │
│       │                                             │
│       ○  A                                         │
│       │                                             │
│       ○  B  = main                                 │
│      / \                                            │
│     ○   ○  C, D                                   │
│     │   │                                           │
│     ○   ○  E, F                                   │
│      \ /                                            │
│       ○  G = @-                                    │
│       │                                             │
│       ◆  H = @ (working copy)                     │
│                                                     │
│  @     = H                                         │
│  @-    = G                                         │
│  @--   = E, F（複数の場合あり）                    │
│  main..@ = C, D, E, F, G, H（mainの子孫で@の祖先） │
│  ..main  = root(), A, B（mainの祖先）              │
└─────────────────────────────────────────────────────┘
```

### 1.2 集合演算

```bash
# 和集合（union）
$ jj log -r 'branch-a | branch-b'

# 積集合（intersection）
$ jj log -r 'mine() & main..'

# 差集合（difference）
$ jj log -r 'all() ~ merges()'
# → 全commitからマージcommitを除いたもの

# 否定（complement）
$ jj log -r '~empty()'
# → 空でないcommit

# 括弧で優先順位を制御
$ jj log -r '(mine() | author("bob")) & (main..)'
# → 自分またはBobのcommitで、mainの子孫であるもの
```

```
┌─────────────────────────────────────────────────────┐
│  revset 集合演算の図解                               │
│                                                     │
│   A = {1, 2, 3, 4, 5}     B = {3, 4, 5, 6, 7}     │
│                                                     │
│   A | B  = {1, 2, 3, 4, 5, 6, 7}  (和集合)         │
│   A & B  = {3, 4, 5}              (積集合)          │
│   A ~ B  = {1, 2}                 (差集合)          │
│   ~A     = 全体 ~ A               (否定)            │
│                                                     │
│  例: 自分のcommitのうちmainにない変更               │
│  mine() & (main..)                                  │
│  = {自分のcommit} ∩ {mainの子孫}                    │
│                                                     │
│  優先順位（高い順）:                                 │
│  1. () — 括弧                                      │
│  2. :: — DAG範囲                                   │
│  3. ~ — 否定（単項）                               │
│  4. & — 積集合                                     │
│  5. | — 和集合                                     │
│  6. ~ — 差集合（二項）                             │
└─────────────────────────────────────────────────────┘
```

### 1.3 関数型revset

```bash
# 祖先・子孫
$ jj log -r 'ancestors(main, 5)'     # mainから5世代前まで
$ jj log -r 'ancestors(main)'        # mainの全祖先
$ jj log -r 'descendants(@)'         # @の全子孫
$ jj log -r 'parents(@)'             # @の親
$ jj log -r 'children(main)'         # mainの直接の子
$ jj log -r 'roots(visible_heads()..@)' # 範囲のルートcommit
$ jj log -r 'heads(all())'           # 全ヘッド（末端commit）
$ jj log -r 'root()'                 # ルートcommit

# フィルタリング
$ jj log -r 'author("gaku")'        # 著者でフィルタ
$ jj log -r 'author_date(after:"2024-01-01")' # 日付でフィルタ
$ jj log -r 'committer_date(before:"2024-06-01")' # コミッター日付
$ jj log -r 'description("feat:")'  # メッセージでフィルタ
$ jj log -r 'description(regex:"^(feat|fix):")'  # 正規表現
$ jj log -r 'empty()'               # 空のcommit
$ jj log -r 'merges()'              # マージcommit
$ jj log -r 'mine()'                # 自分のcommit
$ jj log -r 'conflict()'            # コンフリクトのあるcommit
$ jj log -r 'file("src/auth.js")'   # 特定ファイルを変更したcommit
$ jj log -r 'file("src/")'          # ディレクトリ内のファイル変更
$ jj log -r 'file(glob:"*.rs")'     # globパターンでファイル指定
$ jj log -r 'present(feature-x)'    # 存在する場合のみ（エラー回避）
$ jj log -r 'latest(mine(), 5)'     # 自分の最新5件

# ブランチ関連
$ jj log -r 'bookmarks()'           # ブックマーク付きcommit
$ jj log -r 'bookmarks("feature-")' # パターンマッチ
$ jj log -r 'remote_bookmarks()'    # リモートブックマーク
$ jj log -r 'remote_bookmarks(remote="origin")' # 特定リモート
$ jj log -r 'tags()'                # タグ付きcommit
$ jj log -r 'trunk()'               # trunk（main/master）
$ jj log -r 'visible_heads()'       # 可視ヘッド
```

### 1.4 DAG操作関数

```bash
# connected(): 接続されたrevisionセットを取得
$ jj log -r 'connected(bookmarks())'
# → ブックマーク間の全commitを含む

# reachable(): 到達可能なcommit
$ jj log -r 'reachable(@, all())'

# fork_point(): 分岐点
$ jj log -r 'fork_point(feature-a, feature-b)'
# → 2つのブランチの分岐点

# shortest_common_ancestors(): 最短共通祖先
$ jj log -r 'heads(::feature-a & ::feature-b)'
# → feature-aとfeature-bの最近共通祖先
```

```
┌─────────────────────────────────────────────────────┐
│  DAG操作の図解                                       │
│                                                     │
│          ○ feature-a                               │
│         /                                           │
│   ○───○───○ main                                  │
│         \                                           │
│          ○───○ feature-b                           │
│                                                     │
│  fork_point(feature-a, feature-b)                   │
│  = mainの分岐元のcommit                             │
│                                                     │
│  connected(feature-a | feature-b)                   │
│  = feature-aとfeature-b間の全commit                 │
│    （main上のcommitも含む）                          │
│                                                     │
│  ancestors(feature-a) & ancestors(feature-b)        │
│  = 共通祖先のcommit集合                             │
└─────────────────────────────────────────────────────┘
```

### 1.5 実用的なrevsetクエリ例

```bash
# PR候補の変更一覧（mainにない自分のcommit）
$ jj log -r 'mine() & (main..)'

# コンフリクト解決が必要なcommit
$ jj log -r 'conflict() & descendants(@)'

# 空でないcommitのみ表示
$ jj log -r '(main..) ~ empty()'

# 最新5件のcommit
$ jj log -r 'ancestors(@, 5)'

# 特定ファイルに関連する変更
$ jj log -r 'file("src/auth/")'

# 今週の自分のcommit
$ jj log -r 'mine() & committer_date(after:"1 week ago")'

# ブックマーク付きで空でないcommit
$ jj log -r 'bookmarks() ~ empty()'

# 特定のauthorの最近10件
$ jj log -r 'latest(author("alice"), 10)'

# マージコンフリクトがあるcommitの親を表示
$ jj log -r 'parents(conflict())'

# リモートにpushされていないcommit
$ jj log -r 'mine() ~ ::remote_bookmarks()'

# 特定のファイルを削除したcommit
$ jj log -r 'file("deleted-file.txt") & (main..)'

# 複数の条件を組み合わせた複雑なクエリ
$ jj log -r '(mine() | author("bob")) & (main..) & ~empty() & ~merges()'
# → 自分またはBobの、mainにない、空でない、マージでないcommit

# 分岐している全ブランチのhead
$ jj log -r 'heads(all()) ~ trunk()'

# trunkから分岐した各ブランチのルート
$ jj log -r 'roots(trunk()..heads(all()))'
```

### 1.6 revsetのパフォーマンス

```bash
# revsetの評価は遅延的（lazy）
# → 全commitを列挙せずに必要な部分だけ計算

# 効率的なrevset
$ jj log -r 'ancestors(@, 20)'
# → @から20世代だけ遡る（高速）

# 非効率なrevset（大規模リポジトリで遅い可能性）
$ jj log -r 'all()'
# → 全commitを列挙（巨大リポジトリでは遅い）

# 効率化のテクニック
# 1. 範囲を限定する
$ jj log -r 'trunk()..@ & file("src/")'
# vs
$ jj log -r 'file("src/")'  # 全履歴を検索

# 2. present()で存在チェック
$ jj log -r 'present(old-bookmark)'
# → 存在しない場合にエラーにならない

# 3. latest()で件数を制限
$ jj log -r 'latest(mine(), 10)'
# → 最新10件だけ取得
```

---

## 2. テンプレート言語

### 2.1 基本構文

```bash
# カスタムフォーマットでログを表示
$ jj log -T 'change_id.short() ++ " " ++ description.first_line() ++ "\n"'
rlvkpntz feat: 認証機能
qpvuntsm feat: 初期設定

# 条件分岐
$ jj log -T '
  if(conflict, "CONFLICT: ", "")
  ++ change_id.short()
  ++ " "
  ++ description.first_line()
  ++ "\n"
'

# 色付き出力
$ jj log -T '
  label("change_id", change_id.short())
  ++ " "
  ++ label(if(conflict, "conflict"), description.first_line())
  ++ "\n"
'

# separate()で区切り文字を指定
$ jj log -T '
  separate(" ",
    change_id.short(),
    if(bookmarks, bookmarks),
    description.first_line(),
  ) ++ "\n"
'
# → 空の要素は自動的にスキップされる
```

### 2.2 利用可能なプロパティ

| プロパティ           | 説明                                    | 例                         |
|----------------------|-----------------------------------------|----------------------------|
| `change_id`          | change ID                               | `change_id.short()`        |
| `commit_id`          | commit ID（SHA-1）                      | `commit_id.short(8)`       |
| `description`        | コミットメッセージ                      | `description.first_line()` |
| `author`             | 著者情報                                | `author.name()`            |
| `committer`          | コミッター情報                          | `committer.email()`        |
| `author.timestamp()` | 著者のタイムスタンプ                    | `author.timestamp()`       |
| `working_copies`     | working copyかどうか                    | `self.working_copies()`    |
| `conflict`           | コンフリクトがあるか                    | `if(conflict, "C", "")`    |
| `empty`              | 空のcommitか                            | `if(empty, "(empty)", "")` |
| `bookmarks`          | ブックマーク                            | `bookmarks`                |
| `tags`               | タグ                                    | `tags`                     |
| `branches`           | ブランチ（Git互換表示）                 | `branches`                 |
| `parents`            | 親commit                               | `parents`                  |
| `diff`               | 変更内容                                | `diff.summary()`           |
| `root`               | ルートcommitか                          | `if(root, "ROOT", "")`     |
| `current_working_copy` | 現在のworking copyか                  | `current_working_copy`     |
| `divergent`          | divergentか                             | `if(divergent, "D", "")`   |
| `hidden`             | 隠されたcommitか                        | `hidden`                   |
| `immutable`          | immutableか                             | `if(immutable, "I", "")`   |

### 2.3 メソッドチェーン

```bash
# 文字列メソッド
$ jj log -T 'change_id.short(8) ++ "\n"'   # 8文字に短縮
$ jj log -T 'change_id.shortest() ++ "\n"'  # 最短のユニークプレフィックス
$ jj log -T 'description.first_line() ++ "\n"' # 最初の行のみ
$ jj log -T 'description.lines() ++ "\n"'      # 全行（リスト）

# タイムスタンプメソッド
$ jj log -T 'author.timestamp().ago() ++ "\n"'  # 相対時間（3 hours ago）
$ jj log -T 'author.timestamp().format("%Y-%m-%d %H:%M") ++ "\n"' # フォーマット

# ID メソッド
$ jj log -T 'change_id.short() ++ "\n"'     # 短縮ID
$ jj log -T 'change_id.shortest(4) ++ "\n"' # 最低4文字のユニークID

# 条件付きメソッド
$ jj log -T '
  change_id.short()
  ++ " "
  ++ if(
    description.first_line().len() > 50,
    description.first_line().substr(0, 50) ++ "...",
    description.first_line()
  )
  ++ "\n"
'
```

### 2.4 テンプレートの高度な使い方

```bash
# diff.summary() — 変更ファイル一覧
$ jj log -r @ -T 'diff.summary() ++ "\n"'
# M src/auth.ts
# A src/types.ts
# D src/old.ts

# diff.stat() — 変更統計
$ jj log -r @ -T 'diff.stat(80) ++ "\n"'
# src/auth.ts  | 10 +++++-----
# src/types.ts |  5 +++++

# separate() — 区切り文字で結合（空要素はスキップ）
$ jj log -T '
  separate(" | ",
    change_id.short(),
    author.name(),
    if(conflict, "CONFLICT"),
    if(empty, "empty"),
    bookmarks,
    description.first_line(),
  ) ++ "\n"
'
# rlvk | gaku | feature-auth | feat: 認証機能
# qpvu | gaku | empty | main | 初期設定

# concat() — 単純な結合
$ jj log -T '
  concat(
    change_id.short(),
    " ",
    description.first_line(),
    "\n",
  )
'

# indent() — インデント付きの複数行
$ jj log -T '
  change_id.short() ++ "\n"
  ++ indent("  ", description)
  ++ "\n"
'

# label() — 色付け用ラベル
$ jj log -T '
  label("change_id prefix", change_id.shortest())
  ++ label("change_id rest", change_id.short())
  ++ " "
  ++ label(if(conflict, "conflict"), description.first_line())
  ++ "\n"
'
```

### 2.5 設定ファイルでのテンプレート定義

```toml
# ~/.jjconfig.toml

[template-aliases]
# ログ表示のカスタマイズ
'format_short_change_id(id)' = 'id.shortest(4)'
'format_timestamp(ts)' = 'ts.ago()'
'format_author(author)' = 'author.name()'

# ファイル変更のサマリー
'format_diff_summary()' = '''
  if(diff.summary(),
    "\n" ++ indent("  ", diff.summary()),
    ""
  )
'''

[templates]
# デフォルトのlog表示をカスタマイズ
log = '''
  label(if(current_working_copy, "wc"),
    separate(" ",
      format_short_change_id(change_id),
      if(bookmarks, label("bookmark", bookmarks)),
      if(tags, label("tag", tags)),
      if(conflict, label("conflict", "CONFLICT")),
      if(empty, label("empty", "(empty)")),
      if(divergent, label("divergent", "DIVERGENT")),
      if(immutable, label("immutable", "IMMUTABLE")),
      description.first_line(),
    )
  ) ++ "\n"
'''

# show コマンドのテンプレート
show = '''
  "Change ID: " ++ change_id ++ "\n"
  ++ "Commit ID: " ++ commit_id ++ "\n"
  ++ "Author:    " ++ author.name() ++ " <" ++ author.email() ++ ">\n"
  ++ "Date:      " ++ author.timestamp().format("%Y-%m-%d %H:%M:%S") ++ "\n"
  ++ if(bookmarks, "Bookmarks: " ++ bookmarks ++ "\n", "")
  ++ if(tags, "Tags:      " ++ tags ++ "\n", "")
  ++ "\n"
  ++ indent("    ", description)
  ++ "\n"
  ++ diff.stat(80)
  ++ "\n"
'''

# op log のテンプレート
op_log = '''
  separate(" ",
    self.id().short(),
    self.description().first_line(),
    self.time().start().format("%Y-%m-%d %H:%M"),
  ) ++ "\n"
'''
```

### 2.6 組み込みテンプレートスタイル

```bash
# 組み込みのテンプレートスタイル
$ jj log --template builtin_log_oneline
# → 1行表示

$ jj log --template builtin_log_compact
# → コンパクト表示（デフォルト）

$ jj log --template builtin_log_detailed
# → 詳細表示

# スタイルの確認
$ jj config list templates
# → 現在設定されているテンプレートを表示
```

---

## 3. Git連携

### 3.1 fetch / push

```bash
# リモートからfetch
$ jj git fetch
# → origin の全ブランチを取得

# 特定のリモートからfetch
$ jj git fetch --remote upstream

# 特定のブランチのみfetch
$ jj git fetch --branch main
$ jj git fetch --branch 'feature-*'  # globパターン

# リモートにpush
$ jj git push
# → ローカルブックマークに対応するリモートブランチを更新

# 特定のブックマークのみpush
$ jj git push --bookmark feature-auth

# 新しいブックマークをpush（ブランチを作成）
$ jj git push --bookmark feature-auth --allow-new
# → リモートに新しいブランチが作成される

# 変更されたブックマークのみpush
$ jj git push --change @
# → @のchange IDを含むブックマーク名を自動生成してpush

# dry-run（実際にはpushしない）
$ jj git push --dry-run
# → 何がpushされるか確認

# 全ブックマークをpush
$ jj git push --all
# → ローカルの全ブックマークをpush

# 削除されたブックマークをリモートからも削除
$ jj git push --deleted
```

```
┌─────────────────────────────────────────────────────┐
│  jj git push/fetch のフロー                          │
│                                                     │
│  jj git fetch:                                      │
│  ┌────────────┐      ┌────────────┐                │
│  │ リモート    │ ---> │ ローカル    │                │
│  │ refs/heads/ │      │ bookmarks   │                │
│  │ main        │      │ main@origin │                │
│  │ feature-x   │      │ feature-x   │                │
│  └────────────┘      │ @origin     │                │
│                      └────────────┘                │
│                                                     │
│  jj git push:                                       │
│  ┌────────────┐      ┌────────────┐                │
│  │ ローカル    │ ---> │ リモート    │                │
│  │ bookmarks   │      │ refs/heads/ │                │
│  │ feature-auth│      │ feature-auth│                │
│  └────────────┘      └────────────┘                │
│                                                     │
│  jj git push --change @:                            │
│  1. @のchange IDからブックマーク名を自動生成         │
│  2. push-rlvkpntz... というブランチをリモートに作成  │
│  3. GitHub上でPRを作成可能に                         │
└─────────────────────────────────────────────────────┘
```

### 3.2 co-located リポジトリの運用

```
┌─────────────────────────────────────────────────────┐
│  co-located リポジトリの構造                         │
│                                                     │
│  project/                                           │
│  ├── .git/          ← Git のオブジェクトストア       │
│  ├── .jj/           ← Jujutsu のメタデータ          │
│  │   ├── repo/                                      │
│  │   │   ├── store/                                 │
│  │   │   │   └── git_target  ← .git/ へのパス      │
│  │   │   ├── op_store/       ← Operation Log       │
│  │   │   └── op_heads/                              │
│  │   └── working_copy/                              │
│  └── src/                                           │
│                                                     │
│  → jj と git の両方のコマンドが使用可能             │
│  → git コマンドの結果は jj が自動的に取り込む      │
│  → jj の変更は .git/ にも反映される                 │
└─────────────────────────────────────────────────────┘
```

```bash
# co-located repoの作成
$ git clone https://github.com/user/repo.git
$ cd repo
$ jj git init --colocate
# → .git/ はそのまま、.jj/ が追加される

# または既存のgitリポジトリをco-locate
$ cd existing-git-repo
$ jj git init --colocate

# co-located repoでのgitコマンド使用
$ git status       # git コマンドも使える
$ jj git import    # git側の変更をjjに取り込む（通常は自動）
$ jj git export    # jjの変更をgit refに反映（通常は自動）

# gitでの変更後にjjに反映
$ git checkout -b new-branch
$ git commit -m "change from git"
$ jj git import    # gitの変更をjjに取り込む
$ jj log           # jjからも見える

# co-locatedの注意点
# - git stashはjjからは見えない
# - git rebase -iはjjのopログと不整合を起こす可能性
# - jj側の操作を推奨（jj rebase, jj squash等）
```

### 3.3 リモート管理

```bash
# リモートの追加
$ jj git remote add upstream https://github.com/upstream/repo.git

# リモートの一覧
$ jj git remote list
origin  https://github.com/user/repo.git
upstream https://github.com/upstream/repo.git

# リモートの削除
$ jj git remote remove upstream

# リモートの名前変更
$ jj git remote rename origin github

# リモートURLの変更
$ jj git remote set-url origin git@github.com:user/repo.git
```

### 3.4 Git互換性の詳細

```bash
# Gitブランチ ↔ Jujutsuブックマーク の対応
# Git: refs/heads/main → jj: main (ローカルブックマーク)
# Git: refs/remotes/origin/main → jj: main@origin (リモートブックマーク)
# Git: refs/tags/v1.0 → jj: v1.0 (タグ)

# ブックマークの追跡状態を確認
$ jj bookmark list --all
feature-auth: rlvkpntz abc12345
  @origin: rlvkpntz abc12345 (tracked)
main: qpvuntsm def67890
  @origin: qpvuntsm def67890 (tracked)
old-branch (deleted)
  @origin: xxxxxxxx xxxxxxxx (tracked)

# リモートブックマークの追跡開始
$ jj bookmark track feature-x@origin

# リモートブックマークの追跡解除
$ jj bookmark untrack feature-x@origin

# GIT_HEAD の管理
# jjではHEADの概念がGitと異なる
# working copyの親がGitのHEADに相当
$ jj log -r @-
# → これがGitのHEADに対応するcommit
```

```
┌─────────────────────────────────────────────────────┐
│  Git概念とJujutsu概念の対応                          │
│                                                     │
│  Git                    Jujutsu                     │
│  ────────────────────   ────────────────────        │
│  HEAD                   @（working copy）           │
│  branch                 bookmark                    │
│  refs/remotes/origin/*  *@origin                    │
│  staging area (index)   なし（自動追跡）            │
│  stash                  新しいcommitを作成          │
│  detached HEAD          通常状態（常にdetached的）   │
│  commit SHA             commit ID / change ID       │
│  なし                   change ID（不変の識別子）    │
│  なし                   Operation Log               │
│  merge commit           merge commit（同じ）        │
│  なし                   conflict materialization    │
│  なし                   divergent changes           │
│                                                     │
│  重要な違い:                                        │
│  - change IDはrebase後も不変                        │
│  - commit IDはrebase後に変わる                      │
│  - jjにはstagingの概念がない                        │
│  - 全ファイル変更が自動的にworking copyに含まれる   │
└─────────────────────────────────────────────────────┘
```

---

## 4. 高度なワークフロー

### 4.1 スタックドPR（積み上げPR）

```bash
# スタックされた変更の作成
$ jj new main
$ vim src/types.ts
$ jj describe -m "feat: 型定義の追加"
$ jj bookmark create pr/types -r @

$ jj new
$ vim src/auth.ts
$ jj describe -m "feat: 認証ロジック"
$ jj bookmark create pr/auth -r @

$ jj new
$ vim src/api.ts
$ jj describe -m "feat: APIエンドポイント"
$ jj bookmark create pr/api -r @

# 各ブックマークを個別にpush
$ jj git push --bookmark pr/types --allow-new
$ jj git push --bookmark pr/auth --allow-new
$ jj git push --bookmark pr/api --allow-new

# ベースの変更を修正（型定義を更新）
$ jj edit pr/types
$ vim src/types.ts
# → pr/auth と pr/api が自動リベース！
# → 各PRを再pushするだけ
$ jj git push --bookmark pr/types
$ jj git push --bookmark pr/auth
$ jj git push --bookmark pr/api
```

```
┌────────────────────────────────────────────────────┐
│  スタックドPRの構造                                  │
│                                                    │
│  ○  pr/api   feat: APIエンドポイント               │
│  ○  pr/auth  feat: 認証ロジック                    │
│  ○  pr/types feat: 型定義の追加                    │
│  ◆  main                                          │
│                                                    │
│  GitHub上:                                         │
│  PR #3: api   (base: pr/auth)                      │
│  PR #2: auth  (base: pr/types)                     │
│  PR #1: types (base: main)                         │
│                                                    │
│  pr/typesを修正 → pr/auth, pr/api が自動リベース   │
│  → 3つのPRを全て jj git push で更新                │
│                                                    │
│  PR #1がマージされた場合:                           │
│  $ jj git fetch                                    │
│  $ jj rebase -s pr/auth -d main                    │
│  # → PR #2のbaseをmainに変更                       │
│  $ jj git push --bookmark pr/auth                  │
│  $ jj git push --bookmark pr/api                   │
└────────────────────────────────────────────────────┘
```

### 4.2 `jj git push --change` による自動ブックマーク

```bash
# change IDからブックマークを自動生成してpush
$ jj git push --change rlvkpntz
# → "push-rlvkpntzqwop" のようなブックマークが自動作成される
# → GitHub上にブランチが作成され、PRが作れる

# ブックマーク名のプレフィックスをカスタマイズ
$ jj config set --user git.push-bookmark-prefix "gaku/push-"
$ jj git push --change rlvkpntz
# → "gaku/push-rlvkpntzqwop" ブランチが作成される

# 複数のchangeを同時にpush
$ jj git push --change aaa --change bbb --change ccc
```

### 4.3 並列開発ワークフロー

```bash
# 複数の独立した変更を並列に進める
$ jj new main -m "feat: ログイン画面"
$ jj new main -m "fix: パフォーマンス改善"
$ jj new main -m "docs: READMEの更新"

# 各作業はmainから独立して分岐
# ○ feat: ログイン画面
# │
# │ ○ fix: パフォーマンス改善
# │/
# │ ○ docs: READMEの更新
# │/
# ◆ main

# 作業を切り替え
$ jj edit rlvkpntz  # ログイン画面のcommitに切り替え

# ファイルを編集（自動的にworking copyに反映）
$ vim src/login.tsx
# → 保存するだけ、staging不要

# 別の作業に切り替え
$ jj edit qpvuntsm  # パフォーマンス改善に切り替え
$ vim src/core.ts
```

### 4.4 `jj split` — commitの分割

```bash
# 対話的にcommitを分割
$ jj split
# → エディタが開き、最初のcommitに含めるファイルを選択
# → 残りは新しいcommitに

# 特定のファイルだけ分離
$ jj split --path src/auth.ts
# → src/auth.tsの変更だけ最初のcommitに
# → 残りのファイルは新しいcommitに

# -rで対象commitを指定
$ jj split -r rlvkpntz
# → working copy以外のcommitも分割可能

# 対話的分割の詳細
$ jj split -i
# → diffの各hunks を対話的に選択
# → git add -p 相当の操作をcommitの分割で実行
```

```
┌────────────────────────────────────────────────────┐
│  jj split の動作                                    │
│                                                    │
│  Before:                                           │
│  @  commit-A: auth.ts, api.ts, types.ts を変更     │
│  ○  main                                          │
│                                                    │
│  $ jj split --path types.ts                        │
│                                                    │
│  After:                                            │
│  @  commit-B: auth.ts, api.ts を変更               │
│  ○  commit-A': types.ts のみ変更                  │
│  ○  main                                          │
│                                                    │
│  → commit-Aがtypes.tsだけのcommitに分割            │
│  → 残りの変更がcommit-Bに                          │
└────────────────────────────────────────────────────┘
```

### 4.5 `jj parallelize` — 直列commitの並列化

```bash
# 直列のcommitを並列に変換
$ jj parallelize rlvkpntz::@
# → 依存関係のないcommitを並列のブランチに変換

# Before:
# @  commit-C: docs変更
# ○  commit-B: テスト追加
# ○  commit-A: 機能追加
# ○  main

# After:
# ○  commit-C: docs変更
# │ ○  commit-B: テスト追加
# │/
# │ ○  commit-A: 機能追加
# │/
# ○  main
# → 各commitが独立してmainから分岐
# → 個別にPRを作成可能
```

---

## 5. Operation Log — 操作履歴

### 5.1 Operation Logの基本

```bash
# 操作履歴の表示
$ jj op log
@  abc123 gaku@host 2024-01-15 10:30 (1 minute ago)
│  describe commit rlvkpntzqwop
○  def456 gaku@host 2024-01-15 10:29 (2 minutes ago)
│  new empty commit
○  ghi789 gaku@host 2024-01-15 10:28 (3 minutes ago)
│  snapshot working copy
○  jkl012 gaku@host 2024-01-15 10:25 (6 minutes ago)
│  fetch from git remote(s) origin

# 操作の詳細表示
$ jj op show abc123
# → 操作で変更されたcommitの一覧

# 操作間の差分
$ jj op diff --from def456 --to abc123
# → 2つの操作間で何が変わったか
```

### 5.2 undo / restore

```bash
# 直前の操作を取り消す
$ jj undo
# → 1つ前の状態に戻る

# 特定の操作まで戻る
$ jj op restore abc123
# → abc123の操作時の状態に復元

# operation logの状態を確認してからundo
$ jj op log
# → 戻りたい操作を確認
$ jj op restore jkl012
# → jkl012の時点の状態に完全復元
```

```
┌─────────────────────────────────────────────────────┐
│  Operation Log の概念                                │
│                                                     │
│  Gitとの違い:                                       │
│  Git:  reflog は ref の変更のみ記録                  │
│  jj:   Operation Log は全操作を記録                  │
│                                                     │
│  記録される操作の例:                                 │
│  - new, commit, describe                            │
│  - rebase, squash, split                            │
│  - git fetch, git push                              │
│  - working copy snapshot                            │
│  - bookmark create, move, delete                    │
│                                                     │
│  利点:                                              │
│  - 全操作がundo可能                                 │
│  - rebaseの取り消しも一発                           │
│  - git fetchの取り消しも可能                        │
│  - 操作間の差分を確認可能                           │
│  - 並行操作のマージ（concurrent operations）        │
│                                                     │
│  op1 ─── op2 ─── op3 ─── op4 (current)             │
│                   ↑                                 │
│            jj op restore op2                        │
│            → op4が作られ、op2の状態に復元           │
│                                                     │
│  op1 ─ op2 ─ op3 ─ op4 ─ op5 (current = op2の状態) │
└─────────────────────────────────────────────────────┘
```

### 5.3 並行操作（Concurrent Operations）

```bash
# 2つのターミナルで同時に作業した場合
# Terminal 1: jj describe -m "feat: login"
# Terminal 2: jj new main

# jjは並行操作を自動的にマージ
# → Operation Logに分岐と合流が記録される

# op logで確認
$ jj op log
@    merge123 (merge of 2 operations)
├─ ○ describe commit
└─ ○ new empty commit
   ○ previous state

# 競合が発生した場合
# → jjが自動解決を試みる
# → 解決できない場合はエラーメッセージ
```

---

## 6. 設定のカスタマイズ

### 6.1 ~/.jjconfig.toml 完全ガイド

```toml
[user]
name = "Gaku"
email = "gaku@example.com"

[ui]
# エディタの設定
editor = "vim"
# diff用エディタ（jj diff --tool で使用）
diff-editor = "meld"
# マージ用エディタ
merge-editor = "meld"
# ページャ
pager = "less -FRX"
# デフォルトコマンド（引数なしでjjを実行した時）
default-command = "log"
# デフォルトのログ表示リビジョン
default-revset = 'ancestors(heads(all()), 10)'
# 色の有効化
color = "auto"  # auto, always, never
# diff の形式
diff.format = "git"  # git, color-words, summary

[git]
# push時のブックマーク名プレフィックス
push-bookmark-prefix = "gaku/push-"
# autotracking（新しいリモートブックマークを自動追跡）
auto-local-bookmark = false

# immutable なcommitの定義（rebase/edit を禁止）
[revset-aliases]
'immutable_heads()' = 'trunk() | tags()'
# カスタムrevset
'unpushed()' = 'mine() ~ ::remote_bookmarks()'
'pending_review()' = 'bookmarks() & mine() ~ empty()'
'stack()' = 'trunk()..@'
'needs_fix()' = 'conflict() & descendants(@)'
'wip()' = 'description(regex:"^wip")'
'recent()' = 'latest(mine(), 20)'

[aliases]
# よく使うコマンドのエイリアス
l = ["log", "-r", "ancestors(heads(all()), 10)"]
ll = ["log", "-r", "all()"]
d = ["diff"]
s = ["status"]
n = ["new"]
c = ["commit"]
e = ["edit"]
desc = ["describe"]
sq = ["squash"]
rb = ["rebase"]
# カスタムエイリアス
push-all = ["git", "push", "--all"]
sync = ["git", "fetch", "--all-remotes"]
wip-list = ["log", "-r", "description(regex:\"^wip\")"]

[colors]
# カスタムカラー設定
"change_id" = "magenta"
"commit_id" = "blue"
"bookmarks" = "green bold"
"tags" = "cyan"
"conflict" = "red bold"
"empty" = "dim"
"working_copy" = "green bold"
"divergent" = "yellow bold"
"immutable" = "dim"
"description placeholder" = "yellow dim"

[merge-tools]
# マージツールの設定

# VS Code
[merge-tools.code]
program = "code"
merge-args = ["--wait", "--merge", "$left", "$right", "$base", "$output"]
diff-args = ["--wait", "--diff", "$left", "$right"]

# IntelliJ IDEA
[merge-tools.idea]
program = "idea"
merge-args = ["merge", "$left", "$right", "$base", "$output"]
diff-args = ["diff", "$left", "$right"]

# vimdiff
[merge-tools.vimdiff]
program = "vim"
merge-args = ["-d", "$left", "$right", "$base", "-c", "wincmd J"]

# difftastic（構文対応diff）
[merge-tools.difft]
program = "difft"
diff-args = ["--color=always", "$left", "$right"]

[diff]
# diffのデフォルトツール
tool = "difft"  # difftasticを使用
```

### 6.2 revset-aliasesの高度な活用

```toml
[revset-aliases]
# 基本的なフィルタ
'unpushed()' = 'mine() ~ ::remote_bookmarks()'
'pending_review()' = 'bookmarks() & mine() ~ empty()'
'stack()' = 'trunk()..@'
'needs_fix()' = 'conflict() & descendants(@)'

# チーム開発用
'team_changes()' = 'trunk().. ~ empty()'
'alice_changes()' = 'author("alice") & trunk()..'
'recent_merges()' = 'merges() & ancestors(@, 50)'

# コードレビュー用
'review_ready()' = 'bookmarks() ~ empty() ~ conflict() & mine()'
'stale_branches()' = 'bookmarks() & committer_date(before:"30 days ago")'

# デバッグ用
'touches_auth()' = 'file("src/auth/") & trunk()..'
'large_changes()' = 'file("**") & trunk()..'  # ファイル変更あり
'wip_commits()' = 'description(regex:"^(wip|WIP|fixup!|squash!)")'

# immutableの拡張
'immutable_heads()' = 'trunk() | tags() | remote_bookmarks(remote="production")'
```

```bash
# revset-aliasesの使用例
$ jj log -r 'unpushed()'
# → まだpushしていない自分のcommit

$ jj log -r 'review_ready()'
# → レビュー準備完了のcommit

$ jj log -r 'stale_branches()'
# → 30日以上更新されていないブランチ

$ jj log -r 'wip_commits()'
# → WIPコミットの一覧
```

### 6.3 プロジェクトローカル設定

```bash
# プロジェクト固有の設定（.jj/repo/config.toml に保存）
$ jj config set --repo revset-aliases.'immutable_heads()' 'trunk() | tags() | bookmarks("release-")'

# プロジェクト固有のエイリアス
$ jj config set --repo aliases.deploy '["git", "push", "--bookmark", "production"]'

# 設定の確認
$ jj config list --repo
# → プロジェクトローカルの設定のみ表示

$ jj config list
# → 全設定（user + repo）を表示

# 設定のソースを確認
$ jj config list --include-defaults
# → デフォルト値も含めて表示
```

---

## 7. 高度な操作

### 7.1 `jj absorb` — 変更の自動振り分け

```bash
# working copyの変更を、変更した行の元のcommitに自動振り分け
$ jj absorb
# → 各行がどのcommitで最後に変更されたかを分析
# → 適切なcommitに変更を自動的に振り分け
# → 影響を受けたcommit以降は自動リベース

# 特定のファイルのみabsorb
$ jj absorb --paths src/auth.ts

# dry-runで確認
$ jj absorb --dry-run
# → 実際には変更せず、どこに振り分けられるかを表示
```

```
┌────────────────────────────────────────────────────┐
│  jj absorb の動作                                   │
│                                                    │
│  Before:                                           │
│  @  working copy (auth.js L10修正, api.js L25修正) │
│  ○  commit-B  feat: API (api.js L25を作成)         │
│  ○  commit-A  feat: 認証 (auth.js L10を作成)       │
│                                                    │
│  $ jj absorb                                       │
│                                                    │
│  After:                                            │
│  @  working copy (empty)                           │
│  ○  commit-B' feat: API (api.js L25修正を吸収)     │
│  ○  commit-A' feat: 認証 (auth.js L10修正を吸収)   │
│                                                    │
│  → 各行の修正が "元のcommit" に自動的に振り分け    │
│  → git absorb / hg absorb と同等の機能             │
│                                                    │
│  absorb の判定ロジック:                             │
│  1. working copyの各変更行を特定                   │
│  2. 各行が最後に変更されたcommitを特定（blame相当）│
│  3. そのcommitに変更を吸収                         │
│  4. 判定できない行はworking copyに残る             │
└────────────────────────────────────────────────────┘
```

### 7.2 `jj duplicate` — commitの複製

```bash
# commitを複製（別のchange IDで同じ内容）
$ jj duplicate rlvkpntz
# → 同じ変更内容で新しいchange IDを持つcommitが作成される

# 範囲の複製
$ jj duplicate main..feature-auth
# → main..feature-auth の全commitを複製

# 複製してから別のブランチにrebase
$ jj duplicate rlvkpntz
$ jj rebase -r <new-change-id> -d another-branch

# cherry-pick相当の操作
$ jj duplicate rlvkpntz -d main
# → rlvkpntzの変更をmainの上に複製
```

### 7.3 `jj squash` — commitの統合

```bash
# @を@-に統合
$ jj squash
# → @の変更が@-に吸収される
# → @は空になり、新しい空commitとなる

# 特定のcommitを親に統合
$ jj squash -r rlvkpntz
# → rlvkpntzの変更がその親に吸収される

# メッセージを指定して統合
$ jj squash -m "feat: 完全な認証機能"

# 特定のファイルのみ統合（部分squash）
$ jj squash --paths src/auth.ts
# → src/auth.ts の変更だけ親に統合
# → 他のファイルの変更はそのまま

# interactive squash
$ jj squash -i
# → 統合する変更を対話的に選択
```

### 7.4 `jj move` — 変更の移動

```bash
# @の変更を別のcommitに移動
$ jj move --to rlvkpntz
# → @の変更をrlvkpntzに移動

# 特定のファイルの変更だけ移動
$ jj move --to rlvkpntz --paths src/types.ts

# 2つのcommit間で変更を移動
$ jj move --from aaa --to bbb
# → aaaの変更をbbbに移動

# interactive move
$ jj move --to rlvkpntz -i
# → 移動する変更を対話的に選択
```

### 7.5 `jj fix` — 自動フォーマット

```bash
# 設定（~/.jjconfig.toml）
# [fix.tools.rustfmt]
# command = ["rustfmt", "--edition", "2021"]
# patterns = ["glob:*.rs"]
#
# [fix.tools.prettier]
# command = ["npx", "prettier", "--write"]
# patterns = ["glob:*.{js,ts,jsx,tsx}"]

# 現在のcommitのファイルをフォーマット
$ jj fix

# 範囲のcommitをフォーマット
$ jj fix -r 'trunk()..@'

# 特定のファイルのみ
$ jj fix --paths src/
```

---

## 8. コンフリクト管理

### 8.1 コンフリクトの表現

```bash
# jjではコンフリクトはcommitの一部として記録される
# → 解決せずにcommitを続行できる

# コンフリクトのあるcommitを確認
$ jj log -r 'conflict()'

# コンフリクトの内容を表示
$ jj diff -r <conflict-commit>

# コンフリクトを含むファイルの一覧
$ jj resolve --list
# → コンフリクトのあるファイルが表示される
```

```
┌─────────────────────────────────────────────────────┐
│  Jujutsuのコンフリクト管理                           │
│                                                     │
│  Gitとの違い:                                       │
│  Git: コンフリクト → 解決するまでcommit不可          │
│  jj:  コンフリクト → commitに記録、後で解決可能     │
│                                                     │
│  ○  commit-C (conflict) ← コンフリクトあり         │
│  ○  commit-B             でもcommitできる           │
│  ○  commit-A                                       │
│                                                     │
│  ファイル内のコンフリクトマーカー:                   │
│  <<<<<<< Conflict 1 of 1                            │
│  %%%%%%% Changes from base to side #1               │
│  -old line                                          │
│  +side 1 change                                     │
│  +++++++ Contents of side #2                        │
│  side 2 change                                      │
│  >>>>>>>                                            │
│                                                     │
│  解決方法:                                          │
│  1. ファイルを直接編集してマーカーを削除            │
│  2. jj resolve でマージツールを使用                  │
│  3. 保存するだけで自動的にコンフリクト解決          │
└─────────────────────────────────────────────────────┘
```

### 8.2 コンフリクト解決

```bash
# マージツールで解決
$ jj resolve
# → 設定されたmerge-editorが起動

# 特定のファイルだけ解決
$ jj resolve src/auth.ts

# 手動解決（ファイルを直接編集）
$ vim src/auth.ts
# → コンフリクトマーカーを削除
# → 保存するだけで自動的にコンフリクトが解消される
# → jj status で確認

# 片方の変更を採用
$ jj resolve --tool ':builtin'
# → 組み込みのマージツールを使用

# コンフリクトをリセット
$ jj restore --from @- --paths src/auth.ts
# → 親commitの状態に戻す
```

---

## 9. アンチパターン

### アンチパターン1: revsetを使わずに手動でcommitを一つずつ操作

```bash
# NG: 複数のcommitを個別に操作
$ jj rebase -r aaa -d main
$ jj rebase -r bbb -d main
$ jj rebase -r ccc -d main

# OK: revsetで一括操作
$ jj rebase -s aaa -d main
# → aaa以降の全子孫がmainの上にリベースされる

# または
$ jj rebase -r 'author("gaku") & (trunk()..@)' -d main
# → 条件に合うcommitを一括リベース
```

**理由**: revsetはJujutsuの最大の強みの一つ。複雑な条件でcommitを選択し、一括操作できる。

### アンチパターン2: co-located repoでgit rebaseを直接使う

```bash
# NG: co-located repoでgitのrebaseを直接使う
$ git rebase -i main
# → jjの操作ログと不整合が生じる可能性
# → jjのchange IDとの対応が崩れる

# OK: jjのrebaseを使う
$ jj rebase -s @ -d main
# → 操作ログに記録され、undoも可能
```

**理由**: co-located repoではjjとgitの両方のメタデータを整合的に保つ必要がある。git側の破壊的操作はjjのOperation Logと矛盾する。

### アンチパターン3: jj editの代わりにjj checkout + 修正 + squash

```bash
# NG: 古いcommitを修正するために複雑な手順
$ jj new rlvkpntz
$ vim src/auth.ts  # 修正
$ jj squash        # 親に統合
$ jj new @--       # 元の位置に戻る

# OK: jj editで直接修正
$ jj edit rlvkpntz
$ vim src/auth.ts  # 修正
# → 自動的にworking copyに反映
# → 子孫commitは自動リベース
```

**理由**: `jj edit`はworking copyを指定のcommitに切り替え、直接修正できる。修正後の自動リベースにより、下流のcommitも自動更新される。

### アンチパターン4: absorb を知らずに手動で変更を振り分ける

```bash
# NG: 各行の修正を手動で元のcommitに振り分け
$ jj split -i      # 対話的に分割
# → どの行がどのcommitに対応するか手動で判断
$ jj squash -r <split-commit> --to <target-commit>
# → 手動で移動

# OK: jj absorb で自動振り分け
$ jj absorb
# → 各行がどのcommitで最後に変更されたか自動判定
# → 適切なcommitに自動振り分け
```

**理由**: `jj absorb`は行単位でblame情報を分析し、各修正を適切なcommitに自動的に振り分ける。手動操作のエラーリスクを排除し、圧倒的に高速。

### アンチパターン5: Operation Logを活用せずにバックアップを取る

```bash
# NG: 手動でバックアップ
$ jj git push --bookmark backup-before-rebase
$ jj rebase -s @ -d main
# → 失敗したらbackupブランチから復元...

# OK: Operation Logでundo
$ jj rebase -s @ -d main
# → 問題があったら
$ jj undo
# → 完全に元の状態に復元
# → op logで任意の時点に戻ることも可能
```

**理由**: JujutsuのOperation Logは全操作を記録し、任意の時点に戻せる。手動バックアップは不要。

---

## 10. FAQ

### Q1. revsetの構文を忘れた場合、どこで確認できるか？

**A1.** `jj help revsets`で完全なドキュメントが表示されます。

```bash
$ jj help revsets
# → 全revset関数と演算子の説明が表示される

# 特定のrevsetをテスト（実行前に確認）
$ jj log -r 'mine() & (main..)' --no-graph -T 'change_id.short() ++ "\n"'
# → マッチするrevisionのchange IDのみ表示

# revsetの評価結果を確認
$ jj log -r 'trunk()..@' --no-graph -T 'change_id.short(8) ++ " " ++ description.first_line() ++ "\n"'
```

### Q2. テンプレート言語のデバッグ方法は？

**A2.** テンプレートのエラーメッセージは比較的わかりやすく、利用可能なプロパティとメソッドが表示されます。

```bash
# エラーメッセージの例
$ jj log -T 'invalid_property'
# Error: Failed to parse template
# Caused by: "invalid_property" is not defined
# Hint: Available keywords: ...

# 段階的に構築するのが安全
$ jj log -T 'change_id.short() ++ "\n"'    # まず基本
$ jj log -T 'change_id.short() ++ " " ++ description.first_line() ++ "\n"'

# ヘルプで利用可能なキーワードとメソッドを確認
$ jj help templates
```

### Q3. `jj git push`時にGitHubの認証エラーが出る場合の対処法は？

**A3.** Jujutsuは内部的にlibgit2を使用しているため、GitのHTTPS認証情報が必要です。

```bash
# SSH鍵を使う場合（推奨）
$ jj git remote set-url origin git@github.com:user/repo.git

# HTTPS + credential helperの場合
$ git config --global credential.helper osxkeychain  # macOS
$ git config --global credential.helper store         # Linux

# GitHub CLIのトークンを使う場合
$ gh auth setup-git
$ jj git push

# SSH鍵のパスフレーズ問題
# → ssh-agentを使用
$ eval "$(ssh-agent -s)"
$ ssh-add ~/.ssh/id_ed25519
$ jj git push
```

### Q4. divergent changeとは何か？どう対処するか？

**A4.** 同じchange IDを持つcommitが複数存在する状態です。

```bash
# divergentが発生する原因
# - jj duplicateでchange IDが重複
# - 並行作業で同じcommitを異なる方法で修正

# divergentの検出
$ jj log -r 'divergent()'

# 解決方法: 片方を破棄
$ jj abandon <不要な方のchange-id>

# 解決方法: マージ
$ jj new <divergent-1> <divergent-2>
# → 両方の変更をマージした新しいcommitを作成
$ jj squash
# → 親の片方に統合
```

### Q5. immutable_headsの設定で何ができるか？

**A5.** 保護したいcommitを定義し、誤った操作を防げます。

```toml
# ~/.jjconfig.toml
[revset-aliases]
# trunk（main/master）とタグを保護
'immutable_heads()' = 'trunk() | tags()'

# リリースブランチも保護
'immutable_heads()' = 'trunk() | tags() | bookmarks("release-")'

# リモートにpush済みのcommitを保護
'immutable_heads()' = 'trunk() | tags() | remote_bookmarks()'
```

```bash
# immutableなcommitを操作しようとするとエラー
$ jj edit main
# Error: Commit abc123 is immutable

# immutable設定を確認
$ jj log -r 'immutable()'
```

### Q6. jjのパフォーマンスを改善するには？

**A6.** 以下のポイントを確認してください。

```bash
# 1. watchmanの有効化（ファイル変更の高速検出）
$ jj config set --user core.watchman.register-snapshot-trigger true

# 2. fsmonitor の設定
$ jj config set --user core.fsmonitor "watchman"

# 3. 不要なcommitの整理
$ jj log -r 'empty() & mine()'
# → 空のcommitを確認
$ jj abandon 'empty() & mine() & ~@'
# → 不要な空commitを削除

# 4. Operation Logの整理（大量の操作履歴がある場合）
# → 自動的にガベージコレクションされるが、
#    手動で古い操作を整理することも可能

# 5. 大規模リポジトリでのrevset最適化
# → ancestors(@, N) で深さを制限
# → file() の範囲を限定
$ jj log -r 'ancestors(@, 50) & file("src/auth/")'
```

### Q7. jjでcherry-pickに相当する操作は？

**A7.** `jj duplicate`を使います。

```bash
# Gitのcherry-pick相当
# git cherry-pick abc123

# Jujutsuでは:
$ jj duplicate abc123
# → 新しいchange IDで同じ内容のcommitが作成される

# 特定のブランチの上にcherry-pick
$ jj new main
$ jj restore --from abc123
$ jj describe -m "cherry-picked: original message"

# または、duplicateしてrebase
$ jj duplicate abc123
$ jj rebase -r <new-change-id> -d main
```

### Q8. jjでgit stash相当の操作は？

**A8.** jjにはstashの概念はありませんが、新しいcommitを作成することで同等の操作ができます。

```bash
# Git: git stash
# jj: working copyの変更は常にcommitに含まれるため、
#     新しいcommitを作成するだけ

# 作業を一時退避
$ jj describe -m "wip: 作業中の変更"
$ jj new main  # mainの上で別の作業を開始

# 退避した作業に戻る
$ jj edit <wip-commit-change-id>

# 複数の「stash」を同時に保持可能
# → 各作業が独立したcommitとして存在
# → jj log で一覧表示
$ jj log -r 'description(regex:"^wip:")'
```

---

## まとめ

| 概念             | 要点                                                          |
|------------------|---------------------------------------------------------------|
| revset           | リビジョン集合のクエリ言語、集合演算と関数で柔軟に選択        |
| テンプレート     | ログ出力のカスタマイズ言語、プロパティとメソッドチェーン       |
| jj git fetch     | リモートからブックマークとcommitを取得                        |
| jj git push      | ローカルブックマークをリモートブランチに反映                  |
| co-located repo  | .git/と.jj/が共存、jjとgit両方使用可能                       |
| jj absorb        | working copyの変更を元のcommitに自動振り分け                 |
| jj split         | commitを対話的にファイルやhunk単位で分割                     |
| jj parallelize   | 直列commitを並列ブランチに変換                               |
| jj fix           | 設定されたフォーマッタを自動適用                             |
| Operation Log    | 全操作の履歴、undo/restore が常に可能                        |
| revset-aliases   | よく使うrevsetクエリに名前を付けて再利用                     |
| immutable_heads  | rebase/edit を禁止するcommitの定義                           |
| divergent        | 同じchange IDを持つ複数commitの状態                          |
| conflict         | コンフリクトをcommitに記録可能、後から解決可能               |

---

## 次に読むべきガイド

- [Git→Jujutsu移行](./03-git-to-jujutsu.md) — 操作対応表と実践的な移行手順
- [Jujutsuワークフロー](./01-jujutsu-workflow.md) — 基本ワークフローの復習
- [インタラクティブRebase](../01-advanced-git/00-interactive-rebase.md) — Git側のrebase操作との比較

---

## 参考文献

1. **Jujutsu公式ドキュメント** — "Revsets" https://martinvonz.github.io/jj/latest/revsets/
2. **Jujutsu公式ドキュメント** — "Templates" https://martinvonz.github.io/jj/latest/templates/
3. **Jujutsu公式ドキュメント** — "Git compatibility" https://martinvonz.github.io/jj/latest/git-compatibility/
4. **Jujutsu公式ドキュメント** — "Config" https://martinvonz.github.io/jj/latest/config/
5. **Jujutsu公式ドキュメント** — "Operation Log" https://martinvonz.github.io/jj/latest/operation-log/
6. **Austin Seipp** — "jujutsu: A new VCS" https://austinseipp.com/posts/2024-07-10-jj-hierarchies
7. **Steve Klabnik** — "jj init" https://steveklabnik.com/writing/jj-init/
