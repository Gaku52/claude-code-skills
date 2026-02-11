# Jujutsu応用

> Jujutsuのrevset（リビジョンセット式）、テンプレート言語、Git連携の高度な設定を習得し、複雑なリポジトリ操作と効率的なワークフローを実現する。

## この章で学ぶこと

1. **revsetクエリ言語** — リビジョンの柔軟な選択・フィルタリング構文
2. **テンプレート言語** — ログ出力やコミット表示のカスタマイズ
3. **Git連携の高度な設定** — fetch, push, colocated repoの詳細な運用

---

## 1. revset — リビジョン選択クエリ

### 1.1 基本構文

revsetはコミット（リビジョン）の集合を表現するクエリ言語で、`jj log -r`、`jj rebase`、`jj diff`等のほぼ全てのコマンドで使用できる。

```bash
# 単一revision
$ jj log -r @                    # working copy
$ jj log -r @-                   # working copyの親
$ jj log -r @--                  # working copyの祖父
$ jj log -r rlvkpntz             # change IDで指定
$ jj log -r abc12345             # commit IDで指定
$ jj log -r main                 # ブックマーク名で指定
$ jj log -r main@origin          # リモートブックマーク

# 範囲指定
$ jj log -r 'main..@'            # mainから@までの間のcommit
$ jj log -r 'main..'             # mainの子孫（main自身は含まない）
$ jj log -r '..main'             # mainの祖先（main自身を含む）
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
└─────────────────────────────────────────────────────┘
```

### 1.3 関数型revset

```bash
# 祖先・子孫
$ jj log -r 'ancestors(main, 5)'     # mainから5世代前まで
$ jj log -r 'descendants(@)'         # @の全子孫
$ jj log -r 'parents(@)'             # @の親
$ jj log -r 'children(main)'         # mainの直接の子
$ jj log -r 'roots(visible_heads()..@)' # 範囲のルートcommit

# フィルタリング
$ jj log -r 'author("gaku")'        # 著者でフィルタ
$ jj log -r 'description("feat:")'  # メッセージでフィルタ
$ jj log -r 'empty()'               # 空のcommit
$ jj log -r 'merges()'              # マージcommit
$ jj log -r 'mine()'                # 自分のcommit
$ jj log -r 'conflict()'            # コンフリクトのあるcommit
$ jj log -r 'file("src/auth.js")'   # 特定ファイルを変更したcommit

# ブランチ関連
$ jj log -r 'bookmarks()'           # ブックマーク付きcommit
$ jj log -r 'remote_bookmarks()'    # リモートブックマーク
$ jj log -r 'heads(all())'          # 全ヘッド（末端commit）
$ jj log -r 'trunk()'               # trunk（main/master）
```

### 1.4 実用的なrevsetクエリ例

```bash
# PR候補の変更一覧（mainにない自分のcommit）
$ jj log -r 'mine() & (main..)'

# コンフリクト解決が必要なcommit
$ jj log -r 'conflict() & descendants(@)'

# 空でないcommitのみ表示
$ jj log -r '(main..) ~ empty()'

# 最新5件のcommit
$ jj log -r '@---- | @--- | @-- | @- | @'
# または
$ jj log -r 'ancestors(@, 5)'

# 特定ファイルに関連する変更
$ jj log -r 'file("src/auth/")'

# 今週の自分のcommit
$ jj log -r 'mine() & committer_date(after:"1 week ago")'
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

### 2.3 設定ファイルでのテンプレート定義

```toml
# ~/.jjconfig.toml

[template-aliases]
# ログ表示のカスタマイズ
'format_short_change_id(id)' = 'id.shortest(4)'
'format_timestamp(ts)' = 'ts.ago()'

[templates]
log = '''
  label(if(current_working_copy, "wc"),
    separate(" ",
      format_short_change_id(change_id),
      bookmarks,
      tags,
      if(conflict, label("conflict", "CONFLICT")),
      if(empty, label("empty", "(empty)")),
      description.first_line(),
    )
  ) ++ "\n"
'''
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
# co-located repoでのgitコマンド使用
$ git status       # git コマンドも使える
$ jj git import    # git側の変更をjjに取り込む（通常は自動）
$ jj git export    # jjの変更をgit refに反映（通常は自動）
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
```

---

## 5. 設定のカスタマイズ

### 5.1 ~/.jjconfig.toml

```toml
[user]
name = "Gaku"
email = "gaku@example.com"

[ui]
editor = "vim"
default-command = "log"
diff-editor = "meld"
merge-editor = "meld"
pager = "less -FRX"

[git]
push-bookmark-prefix = "gaku/push-"

# immutable なcommitの定義（rebase/edit を禁止）
[revset-aliases]
'immutable_heads()' = 'trunk() | tags()'

[aliases]
# よく使うコマンドのエイリアス
l = ["log", "-r", "ancestors(heads(all()), 10)"]
d = ["diff"]
s = ["status"]
n = ["new"]
c = ["commit"]

[colors]
"change_id" = "magenta"
"commit_id" = "blue"
"bookmarks" = "green bold"
"conflict" = "red bold"
```

### 5.2 revset-aliasesの活用

```toml
[revset-aliases]
# 自分の未pushなcommit
'unpushed()' = 'mine() ~ ::remote_bookmarks()'

# レビュー待ちの変更
'pending_review()' = 'bookmarks() & mine() ~ empty()'

# mainからの差分
'stack()' = 'trunk()..@'

# コンフリクト解決が必要なもの
'needs_fix()' = 'conflict() & descendants(@)'
```

---

## 6. Jujutsu固有の強力な機能

### 6.1 `jj absorb` — 変更の自動振り分け

```bash
# working copyの変更を、変更した行の元のcommitに自動振り分け
$ jj absorb
# → 各行がどのcommitで最後に変更されたかを分析
# → 適切なcommitに変更を自動的に振り分け
# → 影響を受けたcommit以降は自動リベース
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
└────────────────────────────────────────────────────┘
```

### 6.2 `jj duplicate` — commitの複製

```bash
# commitを複製（別のchange IDで同じ内容）
$ jj duplicate rlvkpntz
# → 同じ変更内容で新しいchange IDを持つcommitが作成される

# 範囲の複製
$ jj duplicate main..feature-auth
# → main..feature-auth の全commitを複製
```

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1. revsetの構文を忘れた場合、どこで確認できるか？

**A1.** `jj help revsets`で完全なドキュメントが表示されます。

```bash
$ jj help revsets
# → 全revset関数と演算子の説明が表示される

# 特定のrevsetをテスト（実行前に確認）
$ jj log -r 'mine() & (main..)' --no-graph -T 'change_id.short() ++ "\n"'
# → マッチするrevisionのchange IDのみ表示
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
| revset-aliases   | よく使うrevsetクエリに名前を付けて再利用                     |
| immutable_heads  | rebase/edit を禁止するcommitの定義                           |

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
4. **Austin Seipp** — "jujutsu: A new VCS" https://austinseipp.com/posts/2024-07-10-jj-hierarchies
