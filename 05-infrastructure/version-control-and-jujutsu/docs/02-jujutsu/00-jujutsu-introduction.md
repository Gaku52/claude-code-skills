# Jujutsu入門

> 次世代バージョン管理システムJujutsu（jj）の設計思想とGitとの根本的な違いを理解し、基本操作をマスターする。

## この章で学ぶこと

1. **Jujutsuの設計思想** — Gitの課題を解決するために何が再設計されたか
2. **Gitとの根本的な違い** — working copy = commit、自動リベース、コンフリクトのファーストクラスサポート
3. **基本操作の習得** — jj init, jj new, jj describe, jj log, jj diff
4. **初期設定とカスタマイズ** — インストールから日常的に快適に使うための設定まで
5. **Jujutsuの内部構造** — Operation Log、ストレージバックエンド、スナップショッティングの仕組み

---

## 1. Jujutsuとは何か

Jujutsu（jj）はGoogleのMartin von Zweigbergk氏が開発したバージョン管理システムで、**Gitのオブジェクトストレージを内部的に利用しつつ、UIと操作モデルを根本から再設計**したツールである。

```
┌─────────────────────────────────────────────────────┐
│  Jujutsuのアーキテクチャ                             │
│                                                     │
│  ┌───────────────────────┐                          │
│  │  ユーザーインターフェース  │  ← jj コマンド       │
│  │  (Jujutsu独自モデル)      │                      │
│  └───────────┬───────────┘                          │
│              │                                      │
│  ┌───────────▼───────────┐                          │
│  │  操作ログ (Operation Log)│  ← 全操作の記録       │
│  │  (undo/redo サポート)    │                        │
│  └───────────┬───────────┘                          │
│              │                                      │
│  ┌───────────▼───────────┐                          │
│  │  バックエンドストレージ    │  ← Git互換           │
│  │  (Git objects/refs)      │                       │
│  └───────────────────────┘                          │
│                                                     │
│  → git clone したリポジトリでそのまま jj を使える   │
│  → jj で作った変更を git push できる                │
└─────────────────────────────────────────────────────┘
```

### 1.1 Jujutsuが解決する課題

| Gitの課題                          | Jujutsuの解決策                        |
|------------------------------------|----------------------------------------|
| ステージングエリアが複雑           | working copy自体がcommit               |
| detached HEADで変更を失う          | 全changeは自動追跡、失われない          |
| rebase時のコンフリクト地獄         | コンフリクトをcommitに記録、後で解決可  |
| ブランチ管理が煩雑                 | 匿名ブランチ（ブランチレス開発）        |
| undoが難しい（reflog頼み）          | Operation Logで全操作のundo/redo        |
| インデックスの理解が必要           | インデックス（ステージング）が不要      |
| rebase -iの複雑な操作              | squash, split, edit等の直感的な操作     |
| stashの管理が面倒                  | 全てがcommitなのでstash不要             |
| ブランチ切替時のworking tree汚染   | working copy = commitで状態が常に明確   |

### 1.2 Jujutsuの開発背景

Jujutsuは以下の背景から生まれた。

1. **Googleの内部VCS経験**: Google社内ではPiperやCitCといった独自のVCSが使われており、Gitとは異なるワークフローが培われてきた。Martin von Zweigbergk氏はMercurialの開発にも携わり、Gitの操作モデルの改善可能性を探っていた。

2. **Mercurialからの教訓**: Mercurialのchangeset概念、revset言語、evolve拡張などの優れた機能をGit互換の形で提供する。特にrevsetクエリ言語はMercurialの最も先進的な機能の一つだった。

3. **Git互換性の重要性**: 新しいVCSを普及させるには既存のGitエコシステムとの互換性が不可欠。JujutsuはGitのオブジェクトストレージをそのまま使用することで、GitHubやGitLabなどとのシームレスな連携を実現している。

4. **モダンなCLI設計**: Gitは2005年の設計思想に基づいており、コマンド体系が一貫していない部分がある（`git checkout`の多義性など）。Jujutsuは現代のCLI設計のベストプラクティスに従い、各コマンドが単一の責務を持つよう設計されている。

```
┌─────────────────────────────────────────────────────┐
│  Jujutsuの影響を受けたVCS・ツール                    │
│                                                     │
│  Mercurial (hg)                                     │
│  ├── revset クエリ言語                              │
│  ├── changeset の概念                               │
│  ├── evolve 拡張（obsmarkers）                      │
│  └── テンプレート言語                               │
│                                                     │
│  Google Piper / CitC                                │
│  ├── working copy = pending change                  │
│  ├── code review 統合                               │
│  └── 大規模モノレポの運用経験                       │
│                                                     │
│  Git                                                │
│  ├── オブジェクトストレージ (blob, tree, commit)    │
│  ├── DAG (有向非巡回グラフ) によるコミット管理      │
│  └── リモート連携 (fetch, push)                     │
│                                                     │
│  → これらの長所を統合した次世代VCS = Jujutsu        │
└─────────────────────────────────────────────────────┘
```

### 1.3 Jujutsuの設計原則

Jujutsuは以下の設計原則に基づいている。

1. **ファーストクラスのコンフリクト**: コンフリクトはエラーではなく、commitに記録できる通常の状態として扱う
2. **自動リベース**: 親commitの変更は子commitに自動的に伝播する
3. **不変のchange ID**: rebaseしてもcommitの論理的なアイデンティティが保持される
4. **操作のundo**: 全ての操作が記録され、安全に取り消し可能
5. **ステージング不要**: working copyの状態が直接commitに反映される
6. **匿名ブランチ**: ブランチ名なしでも開発できる柔軟な設計

---

## 2. インストールと初期設定

### 2.1 各プラットフォームでのインストール

```bash
# macOS (Homebrew)
$ brew install jj

# macOS (MacPorts)
$ sudo port install jujutsu

# Linux (cargo)
$ cargo install --locked jujutsu-cli

# Linux (Arch Linux)
$ pacman -S jujutsu

# Linux (Nix)
$ nix-env -iA nixpkgs.jujutsu

# Linux (Ubuntu/Debian - snap)
$ snap install jj-vcs

# Windows (Scoop)
$ scoop install jujutsu

# Windows (Chocolatey)
$ choco install jujutsu

# Windows (winget)
$ winget install martinvonz.jj

# ソースからビルド
$ git clone https://github.com/martinvonz/jj.git
$ cd jj
$ cargo build --release
$ cp target/release/jj ~/.local/bin/

# バージョン確認
$ jj version
jj 0.25.0
```

### 2.2 初期設定

```bash
# ユーザー名とメールアドレス（必須）
$ jj config set --user user.name "Gaku"
$ jj config set --user user.email "gaku@example.com"

# エディタの設定
$ jj config set --user ui.editor "vim"
# VS Code を使う場合
$ jj config set --user ui.editor "code --wait"
# Emacs を使う場合
$ jj config set --user ui.editor "emacs -nw"

# diff エディタの設定（split, squash で使用）
$ jj config set --user ui.diff-editor "meld"
# difftastic を使う場合
$ jj config set --user ui.diff.tool "difft"

# ページャーの設定
$ jj config set --user ui.pager "less -FRX"
# delta を使う場合
$ jj config set --user ui.pager "delta"

# デフォルトコマンドの設定（引数なしで jj を実行した時のコマンド）
$ jj config set --user ui.default-command "log"

# 設定の確認
$ jj config list
$ jj config list --user    # ユーザーレベルの設定のみ

# 設定ファイルのパスを確認
$ jj config path --user
# → ~/.jjconfig.toml
```

### 2.3 設定ファイルの直接編集

```toml
# ~/.jjconfig.toml の全体像

[user]
name = "Gaku"
email = "gaku@example.com"

[ui]
editor = "vim"
diff-editor = "meld"
merge-editor = "meld"
pager = "less -FRX"
default-command = "log"
# カラー出力の制御
color = "auto"  # "always", "never", "auto"
# ページャーを使うかどうか
paginate = "auto"  # "auto", "never"

[git]
# push時のブックマーク名プレフィックス
push-bookmark-prefix = "gaku/push-"
# fetch時に自動的にGitのタグをインポート
auto-local-bookmark = false

[aliases]
# よく使うコマンドのエイリアス
l = ["log", "-r", "ancestors(heads(all()), 10)"]
ll = ["log", "--no-graph"]
d = ["diff"]
ds = ["diff", "--stat"]
s = ["status"]
n = ["new"]
c = ["commit"]
desc = ["describe"]
```

### 2.4 シェル補完の設定

```bash
# Bash
$ jj util completion bash > ~/.local/share/bash-completion/completions/jj
# または
$ echo 'source <(jj util completion bash)' >> ~/.bashrc

# Zsh
$ jj util completion zsh > ~/.zfunc/_jj
# .zshrc に以下を追加:
# fpath=(~/.zfunc $fpath)
# autoload -Uz compinit && compinit

# Fish
$ jj util completion fish > ~/.config/fish/completions/jj.fish

# PowerShell
$ jj util completion powershell | Out-File -FilePath $PROFILE -Append
```

### 2.5 difftasticとの統合

difftasticは構文ベースのdiffツールで、Jujutsuとの統合が公式にサポートされている。

```bash
# difftasticのインストール
$ brew install difftastic   # macOS
$ cargo install difftastic  # cargo

# jjconfig.toml での設定
# [ui]
# diff.tool = ["difft", "--color=always", "$left", "$right"]

# コマンドラインでの一時的な使用
$ jj diff --tool difft
$ jj log -p --tool difft
```

---

## 3. 基本概念: working copy = commit

### 3.1 Gitとの最大の違い

```
┌─────────────────────────────────────────────────────┐
│  Git のメンタルモデル                                │
│                                                     │
│  Working Directory → Staging Area → Repository      │
│    (未追跡)          (git add)      (git commit)    │
│                                                     │
│  3つの領域を意識する必要がある                       │
│                                                     │
│  よくあるミス:                                      │
│  - git add を忘れてcommitが空になる                 │
│  - 部分的にaddしてcommitの内容が不完全になる        │
│  - git add -A で不要なファイルまで追加してしまう    │
│                                                     │
├─────────────────────────────────────────────────────┤
│  Jujutsu のメンタルモデル                            │
│                                                     │
│  Working Copy = 最新のCommit（常に自動記録）        │
│                                                     │
│  ファイルを編集 → 自動的にworking copy commitに反映 │
│  "jj new" → 新しい空のcommitを開始                  │
│                                                     │
│  ステージングエリアは存在しない                      │
│                                                     │
│  利点:                                              │
│  - git add を忘れる心配がない                       │
│  - 常に変更がcommitに記録されている                 │
│  - stash が不要（全てがcommit）                     │
│  - 部分的な変更の選択は jj split で後からでも可能   │
└─────────────────────────────────────────────────────┘
```

### 3.2 changeとcommitの違い

```
┌────────────────────────────────────────────────────┐
│  Jujutsuの用語                                      │
│                                                    │
│  change:  論理的な変更単位（change IDで識別）       │
│           change ID は rebase しても変わらない      │
│           8文字の英小文字で表示（例: rlvkpntz）     │
│                                                    │
│  commit:  特定時点のスナップショット                 │
│           commit ID (SHA-1) は内容変更で変わる      │
│           16進数で表示（例: abc12345）              │
│                                                    │
│  例:                                               │
│  change "rlvkpntz" の commit が abc12345 だった場合│
│  rebase すると commit は def67890 に変わるが        │
│  change ID は "rlvkpntz" のまま                    │
│                                                    │
│  → change ID で追跡すれば rebase しても参照可能    │
│                                                    │
│  重要な違い:                                       │
│  ┌──────────────┬──────────────┬──────────────┐    │
│  │              │ change ID    │ commit ID    │    │
│  ├──────────────┼──────────────┼──────────────┤    │
│  │ rebase時     │ 変わらない   │ 変わる       │    │
│  │ amend時      │ 変わらない   │ 変わる       │    │
│  │ 形式         │ 英小文字     │ 16進数       │    │
│  │ 用途         │ 追跡用       │ Git互換      │    │
│  │ ユニーク性   │ リポジトリ内 │ グローバル   │    │
│  └──────────────┴──────────────┴──────────────┘    │
└────────────────────────────────────────────────────┘
```

### 3.3 スナップショッティングの仕組み

Jujutsuの「working copy = commit」は、スナップショッティングという仕組みで実現されている。

```
┌─────────────────────────────────────────────────────┐
│  スナップショッティングの動作フロー                   │
│                                                     │
│  1. ユーザーがファイルを編集                         │
│     → この時点ではcommitに反映されていない           │
│                                                     │
│  2. jjコマンドを実行（jj status, jj log, jj diff等）│
│     → コマンド実行前にworking copyのスナップショット │
│       が自動的に取得される                           │
│     → ファイルシステムの変更がcommitに反映される     │
│                                                     │
│  3. スナップショットの最適化                          │
│     → watchman連携で変更ファイルのみをチェック       │
│     → 大規模リポジトリでも高速に動作                 │
│                                                     │
│  つまり:                                             │
│  - Ctrl+S でファイル保存するたびにcommitが           │
│    作られるわけではない                              │
│  - jjコマンド実行時に「まとめて」反映される          │
│  - パフォーマンスへの影響は最小限                    │
└─────────────────────────────────────────────────────┘
```

```bash
# スナップショッティングの確認
$ echo "Hello" > hello.txt
$ echo "World" > world.txt

# この時点ではまだcommitに反映されていない（ファイルシステム上のみ）

# jj status を実行するとスナップショットが取得される
$ jj status
Working copy changes:
A hello.txt
A world.txt

# 以降、jjコマンドの出力には常に最新のファイル状態が反映される
```

### 3.4 watchman連携による高速化

```bash
# watchmanのインストール
$ brew install watchman   # macOS
$ sudo apt install watchman  # Ubuntu

# jjの設定でwatchmanを有効化
$ jj config set --user core.fsmonitor "watchman"

# watchmanが有効な場合の動作:
# - ファイルシステムの変更を常時監視
# - スナップショット時に変更されたファイルのみチェック
# - 大規模リポジトリ（10万ファイル超）でも数百msで完了
```

---

## 4. 基本操作

### 4.1 リポジトリの初期化

```bash
# 新規リポジトリの作成
$ jj git init my-project
$ cd my-project

# 既存のGitリポジトリをjjで使う（co-located）
$ cd existing-git-repo
$ jj git init --colocate
# → .jj/ ディレクトリが作成され、既存の .git/ と共存

# リモートリポジトリのクローン
$ jj git clone https://github.com/user/repo.git
$ jj git clone git@github.com:user/repo.git  # SSH

# 特定のブランチのみクローン
$ jj git clone --branch main https://github.com/user/repo.git

# co-located でクローン（推奨）
$ jj git clone --colocate https://github.com/user/repo.git
```

```
┌─────────────────────────────────────────────────────┐
│  初期化パターンの選択指針                            │
│                                                     │
│  1. 新規プロジェクト:                                │
│     $ jj git init my-project                        │
│     → .jj/ と .git/ の両方が作成される              │
│                                                     │
│  2. 既存Gitリポジトリに追加:                         │
│     $ jj git init --colocate                        │
│     → .jj/ のみ追加、.git/ はそのまま              │
│     → git コマンドもそのまま使える                  │
│                                                     │
│  3. リモートからクローン:                            │
│     $ jj git clone --colocate URL                   │
│     → co-located で開始（git も使える）             │
│                                                     │
│  推奨: 常に co-located で始める                     │
│  → git コマンドが必要になった時に困らない           │
│  → チームメンバーが git を使っていても問題ない      │
└─────────────────────────────────────────────────────┘
```

### 4.2 変更の作成と記録

```bash
# ファイルを作成・編集（自動的にworking copy commitに反映）
$ echo "Hello" > hello.txt

# 現在の状態を確認
$ jj status
Working copy changes:
A hello.txt

# working copy commitに説明を追加
$ jj describe -m "feat: hello.txtを作成"

# 複数行のメッセージを設定
$ jj describe -m "feat: hello.txtを作成

初期のHelloメッセージファイルを追加した。
このファイルはプロジェクトの基盤となる。"

# エディタで説明を編集（-m を省略）
$ jj describe
# → 設定されたエディタが開き、コミットメッセージを編集

# 新しいcommitを開始（現在の変更を確定し、新しい空commitへ移動）
$ jj new
# → 今の変更が確定され、その上に新しい空のworking copy commitが作成される

# describeとnewを同時に行うショートカット
$ jj commit -m "feat: hello.txtを作成"
# → jj describe -m "..." && jj new と同等

# ログの確認
$ jj log
@  rlvkpntz gaku@example.com 2025-02-11 15:30:00 abc12345
│  (empty) (no description set)
○  qpvuntsm gaku@example.com 2025-02-11 15:25:00 def67890
│  feat: hello.txtを作成
○  zzzzzzzz root() 00000000
```

### 4.3 `jj log`の読み方

```
┌─────────────────────────────────────────────────────┐
│  jj log の出力フォーマット                           │
│                                                     │
│  @  rlvkpntz gaku@... 2025-02-11 abc12345          │
│  ^  ^^^^^^^^ ^^^^^^   ^^^^^^^^^^  ^^^^^^^^          │
│  │  │        │        │           │                 │
│  │  │        │        │           └── commit ID     │
│  │  │        │        └── タイムスタンプ             │
│  │  │        └── 作成者                             │
│  │  └── change ID（短縮形）                         │
│  └── @ = working copy（現在位置）                   │
│                                                     │
│  記号の意味:                                         │
│  @  = working copy（現在編集中のcommit）            │
│  ○  = 通常のcommit                                  │
│  ◆  = 不変（immutable）のcommit                     │
│  ×  = コンフリクトあり                              │
│                                                     │
│  グラフの線:                                        │
│  │  = 直線的な親子関係                              │
│  ├─┘ = ブランチの合流                               │
│  ├─┐ = ブランチの分岐                               │
└─────────────────────────────────────────────────────┘
```

```bash
# ログの表示オプション

# グラフなしで表示
$ jj log --no-graph

# 特定のrevision範囲を表示
$ jj log -r 'main..'

# パッチ（差分）付きで表示
$ jj log -p

# stat（変更ファイル統計）付きで表示
$ jj log -s

# カスタムテンプレートで表示
$ jj log -T 'change_id.short() ++ " " ++ description.first_line() ++ "\n"'

# 特定のファイルに関連するログのみ表示
$ jj log -r 'file("src/auth.js")'

# 直近N件のみ表示
$ jj log -n 10
# または revset で
$ jj log -r 'ancestors(@, 10)'
```

### 4.4 差分の確認

```bash
# working copyの変更を確認
$ jj diff

# 特定のchangeの変更を確認
$ jj diff -r qpvuntsm

# 2つのrevision間の差分
$ jj diff --from main --to @

# stat形式で概要を確認
$ jj diff --stat

# 特定のファイルのみ
$ jj diff src/auth.js

# サマリー形式（追加/変更/削除のファイル一覧）
$ jj diff --summary

# difftasticを使った差分表示
$ jj diff --tool difft

# 特定のrevisionの内容を詳細に表示
$ jj show qpvuntsm
$ jj show @-
$ jj show main
```

### 4.5 ファイルの追跡制御

```bash
# .gitignore と同じ仕組み（Jujutsuは.gitignoreを読む）
$ echo "node_modules/" >> .gitignore
$ echo "*.log" >> .gitignore
$ echo ".env" >> .gitignore

# 特定ファイルを復元（working copyの変更を取り消す）
$ jj restore --from @- src/auth.js
# @- = working copyの親commit

# 特定のrevisionからファイルを復元
$ jj restore --from main src/config.js

# 全変更を取り消す
$ jj restore

# 特定のファイルパターンのみ復元
$ jj restore "src/**/*.js"

# ファイル一覧の確認
$ jj file list
$ jj file list -r main  # 特定revisionのファイル一覧
```

### 4.6 ファイル操作のユーティリティ

```bash
# ファイルの内容を特定のrevisionから表示
$ jj file show src/auth.js
$ jj file show -r main src/auth.js

# 特定のファイルの変更履歴を追跡
$ jj log -r 'file("src/auth.js")'

# ファイルのコピー・リネームの扱い
# Jujutsuはコピー/リネームを自動検出する
$ mv src/old.js src/new.js
$ jj status
# Working copy changes:
# R {src/old.js => src/new.js}

# 複数ファイルの一括操作
$ jj restore --from @- "src/*.js" "test/*.js"
```

---

## 5. コンフリクトのファーストクラスサポート

Gitではコンフリクトが発生するとrebase/mergeが中断されるが、Jujutsuでは**コンフリクト状態がcommitに記録され、後から解決できる**。

### 5.1 コンフリクトの基本的な扱い

```bash
# コンフリクトが発生してもrebaseは完了する
$ jj rebase -d main
# Rebased 3 commits
# New conflicts in:
#   rlvkpntz abc12345

# コンフリクト状態のログ
$ jj log
@  rlvkpntz gaku@... 2025-02-11 abc12345 conflict
│  feat: 認証機能                         ^^^^^^^^
│                                         コンフリクトマーク
○  ...

# コンフリクトを確認
$ jj status
Working copy changes:
C src/auth.js    ← C = conflict

# ファイルを編集してコンフリクトを解決
$ vim src/auth.js
# → 通常のファイル編集でコンフリクトマーカーを解消
# → 自動的にworking copy commitに反映
# → コンフリクトが解消されると conflict マークが消える
```

### 5.2 コンフリクトマーカーの形式

```
┌─────────────────────────────────────────────────────┐
│  Jujutsuのコンフリクトマーカー                       │
│                                                     │
│  Gitの場合:                                         │
│  <<<<<<< HEAD                                       │
│  current changes                                    │
│  =======                                            │
│  incoming changes                                   │
│  >>>>>>> branch-name                                │
│                                                     │
│  Jujutsuの場合:                                     │
│  <<<<<<< Conflict 1 of 1                            │
│  %%%%%%% Changes from base to side #1               │
│  -old line from base                                │
│  +new line from side 1                              │
│  +++++++ Contents of side #2                        │
│  new line from side 2                               │
│  >>>>>>> Conflict 1 of 1 ends                       │
│                                                     │
│  Jujutsuのマーカーは3-way情報を含むため、           │
│  何が変更されたかがより明確にわかる                  │
└─────────────────────────────────────────────────────┘
```

```bash
# コンフリクトの確認と解決の流れ
$ jj status
# C src/auth.js

# コンフリクトファイルの中身を確認
$ cat src/auth.js
# <<<<<<< Conflict 1 of 1
# %%%%%%% Changes from base to side #1
# -const AUTH_KEY = "old-key";
# +const AUTH_KEY = "new-key-from-branch-a";
# +++++++ Contents of side #2
# const AUTH_KEY = "new-key-from-branch-b";
# >>>>>>> Conflict 1 of 1 ends

# エディタでコンフリクトマーカーを削除して解決
$ vim src/auth.js
# const AUTH_KEY = "new-key-from-branch-b"; と書き換え

# 解決されたか確認
$ jj status
# Working copy changes:
# M src/auth.js    ← C(conflict) から M(modified) に変わった
```

### 5.3 コンフリクトの遅延解決

```bash
# コンフリクトがあるcommitの上で作業を続行できる
$ jj log
@  rlvkpntz ... conflict
│  feat: 認証機能
○  ...

# コンフリクトを今は無視して、新しい作業を開始
$ jj new main
$ vim src/other-feature.js
$ jj describe -m "feat: 別の機能"

# 後からコンフリクトを解決
$ jj edit rlvkpntz
$ vim src/auth.js   # コンフリクトを解決
$ jj new            # 解決を確定

# コンフリクトのあるcommitを一覧
$ jj log -r 'conflict()'
```

```
┌────────────────────────────────────────────────────┐
│  Git vs Jujutsu: コンフリクトの扱い                 │
│                                                    │
│  Git:                                              │
│  rebase中にコンフリクト                             │
│    → rebaseが中断                                  │
│    → 即座に解決が必要                               │
│    → --abort で全てやり直し or --continue           │
│    → 複数commitでコンフリクトがあると辛い           │
│                                                    │
│  Jujutsu:                                          │
│  rebase中にコンフリクト                             │
│    → rebaseは完了する                               │
│    → コンフリクト状態がcommitに記録される           │
│    → 好きなタイミングで解決できる                   │
│    → 他の作業を先にすることも可能                   │
│    → 複数commitのコンフリクトも個別に解決可         │
│    → コンフリクト状態のcommitをpushしない限り安全   │
└────────────────────────────────────────────────────┘
```

---

## 6. Operation Log — 全操作のundo/redo

### 6.1 Operation Logの基本

```bash
# 操作ログの確認
$ jj op log
@  abc12345 gaku@... 2025-02-11 15:30:00
│  new empty commit
○  def67890 gaku@... 2025-02-11 15:25:00
│  describe commit rlvkpntz
○  789abcde gaku@... 2025-02-11 15:20:00
│  snapshot working copy
○  fedcba98 gaku@... 2025-02-11 15:15:00
│  add workspace 'default'

# 直前の操作を取り消す
$ jj undo
# → 直前のjjコマンドの効果が完全に取り消される

# 特定の操作時点に戻る
$ jj op restore def67890
```

### 6.2 Operation Logの活用パターン

```bash
# 間違ったrebaseを取り消す
$ jj rebase -d wrong-branch
# → しまった、間違ったブランチにrebaseしてしまった！
$ jj undo
# → rebase前の状態に完全に戻る

# abandonしたcommitを復活させる
$ jj abandon rlvkpntz
# → しまった、間違ったcommitをabandonしてしまった！
$ jj undo
# → commitが復活する

# 特定の操作時点のログを確認
$ jj op log --no-graph
# → 全操作の一覧が表示される

# 操作ログの差分を確認
$ jj op diff --from abc12345 --to def67890
# → 2つの操作間でどのcommitが変更されたかを表示

# 操作ログの詳細表示
$ jj op show abc12345
```

### 6.3 Operation LogとGit reflogの比較

```
┌─────────────────────────────────────────────────────┐
│  Operation Log vs Git reflog                         │
│                                                     │
│  Git reflog:                                        │
│  - HEADの移動履歴のみ記録                           │
│  - ブランチごとに個別のreflog                       │
│  - 表示: git reflog                                 │
│  - 復元: git reset --hard HEAD@{n}                  │
│  - 期限切れ: デフォルト90日で消える                 │
│                                                     │
│  Jujutsu Operation Log:                             │
│  - 全てのjj操作を記録                               │
│  - リポジトリ全体の状態を記録                       │
│  - 表示: jj op log                                  │
│  - 復元: jj op restore <op-id>                      │
│  - 期限切れ: なし（明示的にgcするまで保持）         │
│  - undo: jj undo で直前の操作を取り消し             │
│                                                     │
│  → Operation Logはreflogの上位互換                  │
│  → 全操作が記録されるため、安全に実験できる         │
└─────────────────────────────────────────────────────┘
```

---

## 7. Jujutsuの内部構造

### 7.1 ストレージバックエンド

```
┌─────────────────────────────────────────────────────┐
│  Jujutsu のストレージ構造                            │
│                                                     │
│  .jj/                                               │
│  ├── repo/                                          │
│  │   ├── store/                                     │
│  │   │   ├── git_target          ← Gitストアへのパス│
│  │   │   ├── type                ← "git" (ストア種別)│
│  │   │   └── extra/              ← Jujutsu固有データ│
│  │   │       ├── change_id_index ← change ID索引    │
│  │   │       └── ...                                │
│  │   ├── op_store/               ← Operation Log    │
│  │   │   ├── operations/         ← 各操作のデータ  │
│  │   │   └── views/              ← 各操作時点の状態│
│  │   └── op_heads/               ← 最新操作のID    │
│  └── working_copy/               ← working copyの  │
│      └── ...                        状態管理        │
│                                                     │
│  .git/  (co-located の場合)                         │
│  ├── objects/                    ← blob, tree, commit│
│  ├── refs/                       ← ブランチ、タグ   │
│  └── ...                                            │
│                                                     │
│  → Jujutsuは .git/objects/ にGitオブジェクトを保存  │
│  → .jj/ にはJujutsu固有のメタデータのみ            │
│  → そのためGitツール（gitk, git log等）も使える    │
└─────────────────────────────────────────────────────┘
```

### 7.2 change IDの生成と管理

```bash
# change IDの形式
# - 英小文字のみ（a-z）で構成
# - 内部的には128ビットのランダム値
# - 表示時に短縮形を使用（ユニークな最短プレフィックス）

# change IDの確認
$ jj log -T 'change_id ++ "\n"'
# → rlvkpntzsqkxyrmpqvlwxpsmkowkzmkqtlqnovpv
# 表示上は rlvkpntz のように短縮される

# change IDでの参照
$ jj show rlvkpntz
$ jj show rlvk       # 十分にユニークなら短いプレフィックスでもOK
$ jj diff -r rl      # さらに短くても、ユニークなら動作する
```

### 7.3 immutable commitsの仕組み

```bash
# immutableなcommit = rebase, edit, squash等の対象にならないcommit
# デフォルトでは trunk()（main/master）以前のcommitがimmutable

# immutableの定義を確認
$ jj config get revset-aliases.'immutable_heads()'
# → "trunk() | tags()"

# immutableを変更（例: main と release タグ）
# ~/.jjconfig.toml に追加:
# [revset-aliases]
# 'immutable_heads()' = 'trunk() | tags() | remote_bookmarks()'

# immutableなcommitを編集しようとするとエラー
$ jj edit main
# Error: Commit abc12345 is immutable
# Hint: Configure the set of immutable commits via `revset-aliases.immutable_heads()`

# 一時的にimmutableを無視（非推奨だが緊急時に使用）
$ jj rebase -r main --ignore-immutable -d @
```

---

## 8. 日常的な操作パターン

### 8.1 典型的な開発サイクル

```bash
# 1. 最新のmainから作業開始
$ jj git fetch
$ jj new main@origin

# 2. ファイルを編集
$ vim src/feature.js

# 3. コミットメッセージを設定
$ jj describe -m "feat: ユーザープロフィール機能を追加"

# 4. さらに編集を続ける（自動的にcommitに反映）
$ vim src/feature.test.js
$ vim src/types.ts

# 5. 区切りがついたら新しいcommitを開始
$ jj new

# 6. 次の機能の実装
$ vim src/another-feature.js
$ jj describe -m "feat: 通知機能を追加"

# 7. pushする前にブックマークを設定
$ jj bookmark create feature-profile -r @-  # プロフィール機能のcommit
$ jj bookmark create feature-notify -r @    # 通知機能のcommit

# 8. pushしてPRを作成
$ jj git push --bookmark feature-profile --allow-new
$ jj git push --bookmark feature-notify --allow-new
```

### 8.2 過去のcommitの修正

```bash
# 2つ前のcommitにタイポを見つけた場合

# 方法1: jj edit で直接編集
$ jj edit rlvkpntz           # 修正したいcommitに移動
$ vim src/feature.js          # ファイルを修正
$ jj new                     # 新しいworking copyに戻る
# → 間のcommitは自動リベース

# 方法2: working copyで修正してabsorbで振り分け
$ vim src/feature.js          # working copyで修正
$ jj absorb                  # 修正を元のcommitに自動振り分け

# 方法3: working copyで修正してsquashで統合
$ vim src/feature.js          # working copyで修正
$ jj squash --into rlvkpntz  # 特定のcommitに統合
```

### 8.3 複数の作業の同時進行

```bash
# 認証機能の開発中に緊急バグ修正が入った場合

# 現在の作業状態
$ jj log
@  xxx  feat: 認証機能の実装中
○  main ...

# 緊急バグ修正のために新しいcommitを作成（mainから分岐）
$ jj new main
$ vim src/bugfix.js
$ jj describe -m "fix: ログインエラーの修正"
$ jj bookmark create hotfix -r @
$ jj git push --bookmark hotfix --allow-new

# 元の作業に戻る
$ jj edit xxx    # 認証機能のcommitに戻る
# → stash不要！全てがcommitとして保存されている
```

---

## 9. アンチパターン

### アンチパターン1: Gitの癖でgit addしてからjj describe

```bash
# NG: Jujutsuでgit addは不要
$ vim src/auth.js
$ git add src/auth.js        # ← 不要！
$ jj describe -m "fix: auth"

# OK: ファイルを編集するだけで自動反映
$ vim src/auth.js
$ jj describe -m "fix: auth"
# → working copy commitに自動的に反映されている
```

**理由**: Jujutsuにはステージングエリアが存在しない。ファイルの変更は自動的にworking copy commitに記録される。co-locatedリポジトリでgit addを使うと、Gitのインデックスに変更が登録されるが、Jujutsu側では無意味であり、混乱の原因になる。

### アンチパターン2: jj newを忘れて前のcommitに追記し続ける

```bash
# NG: describe後にnewせず作業を続ける
$ jj describe -m "feat: ユーザー認証"
$ vim src/api.js       # ← この変更も "feat: ユーザー認証" に入ってしまう

# OK: 新しい論理的変更の前にjj newする
$ jj describe -m "feat: ユーザー認証"
$ jj new               # ← 新しいcommitを開始
$ vim src/api.js       # → 新しいcommitに記録される
$ jj describe -m "feat: APIエンドポイント"
```

**理由**: Jujutsuではworking copyが常に最新commitと同一視される。`jj new`を実行しないと、全ての変更が1つのcommitに蓄積されてしまう。ただし後から`jj split`で分割することも可能なので、致命的ではない。

### アンチパターン3: commit IDをchange IDの代わりに使い続ける

```bash
# NG: commit ID（SHA-1）で参照し続ける
$ jj edit abc12345
$ vim src/fix.js
$ jj new
# → abc12345 はもう存在しない（rebaseでSHAが変わった）

# OK: change IDで参照する
$ jj edit rlvkpntz
$ vim src/fix.js
$ jj new
# → rlvkpntz は rebase 後も変わらない
```

**理由**: commit IDはGitのSHA-1ハッシュであり、commitの内容（ツリー、親、メッセージ等）から計算される。rebaseやamendで内容が変わるとcommit IDも変わる。一方、change IDはJujutsu独自の識別子で、commitの内容に依存しないため、安定した参照が可能。

### アンチパターン4: jj undo を万能だと思い込む

```bash
# 注意: jj undoは直前の「1つの」jj操作のみ取り消す
$ jj rebase -d main
$ jj describe -m "new message"
$ jj undo
# → describe のみが取り消される（rebaseは取り消されない）

# 複数操作を戻したい場合は jj op restore を使う
$ jj op log
# → 戻りたい時点のoperation IDを確認
$ jj op restore <op-id>
```

**理由**: `jj undo`は直前の1操作のみを取り消す。複数の操作を取り消す場合は、`jj op log`で目的の操作時点を探し、`jj op restore`で復元する。

---

## 10. FAQ

### Q1. JujutsuはGitを完全に置き換えるのか？

**A1.** いいえ、JujutsuはGitの**上位レイヤー**として動作します。内部的にはGitのオブジェクトストレージを使用しており、`jj git push`/`jj git fetch`で通常のGitリモートと完全に互換性があります。チームの他のメンバーがGitを使い続けていても問題ありません。

### Q2. 既存のGitリポジトリでJujutsuを試すには？

**A2.** `jj git init --colocate`で即座に使い始められます。

```bash
$ cd existing-git-repo
$ jj git init --colocate
# → .jj/ が作成され、既存の .git/ と共存
# → jj log で既存の全コミット履歴が表示される
# → git コマンドもそのまま使える

# 気に入らなければ .jj/ を削除するだけで元に戻せる
$ rm -rf .jj
# → 完全にGitのみの状態に戻る
```

### Q3. Jujutsuのworking copy commitは毎回コミットされるのか？パフォーマンスは大丈夫か？

**A3.** Jujutsuは「スナップショッティング」という仕組みで、jjコマンドの実行時にworking copyの状態をcommitに反映します。ファイル保存のたびにcommitが作られるわけではなく、`jj status`、`jj log`、`jj diff`などのコマンド実行時にスナップショットが取得されます。watchman連携で高速化も可能です。

### Q4. Jujutsuはモノレポ（monorepo）でも使えるか？

**A4.** はい、使えます。Jujutsuはwatchman連携による高速なスナップショッティングを提供しており、大規模なモノレポでも実用的な速度で動作します。Google社内でのテストでも大規模リポジトリでの利用が検証されています。

### Q5. Jujutsuにはどのようなエディタ/IDE統合があるか？

**A5.** 2025年時点での統合状況は以下の通りです。

| エディタ/IDE     | 統合状況                                        |
|------------------|-------------------------------------------------|
| VS Code          | co-locatedでGit拡張を使用可能                   |
| IntelliJ IDEA    | co-locatedでGit統合を使用可能                   |
| Vim/Neovim       | fugitive.vimがco-locatedで動作                  |
| Emacs            | magitがco-locatedで動作                         |
| lazyjj           | Jujutsu専用のTUIツール                          |
| jj-fzf           | fzfベースのインタラクティブツール               |

co-locatedリポジトリを使用することで、既存のGit統合を持つエディタ/IDEをそのまま活用できます。

### Q6. Jujutsuのバージョンアップで互換性が壊れることはあるか？

**A6.** Jujutsuはまだ1.0に達しておらず、マイナーバージョン間でCLIの変更が入ることがあります。ただし、ストレージフォーマットの互換性は維持されており、`jj` をアップグレードしても既存のリポジトリは引き続き使用できます。重要な変更はCHANGELOGとマイグレーションガイドで通知されます。

### Q7. Jujutsuでサブモジュールは使えるか？

**A7.** Jujutsuは現在、Gitサブモジュールの部分的なサポートを提供しています。co-locatedリポジトリではGitのサブモジュールコマンドをそのまま使用できますが、Jujutsu固有のサブモジュール管理機能はまだ開発中です。

---

## まとめ

| 概念                   | 要点                                                          |
|------------------------|---------------------------------------------------------------|
| working copy = commit  | ステージングなし、編集が即座にcommitに反映                    |
| change ID              | rebaseで変わらない安定した識別子                              |
| commit ID              | Git互換のSHA-1、内容変更で変化する                           |
| jj new                 | 新しいcommitを開始、前の変更を確定                           |
| jj describe            | working copy commitにメッセージを設定                         |
| jj commit              | describe + new のショートカット                               |
| コンフリクト記録       | rebase/merge中断なし、commitにコンフリクト状態を保存         |
| Operation Log          | 全操作の記録、undo/redoが可能                                |
| co-located repo        | .git/と.jj/が共存、Git/Jujutsu両方使用可能                   |
| スナップショッティング | jjコマンド実行時にworking copyの状態を自動反映               |
| immutable commits      | trunk()やtags()のcommitはrebase/edit不可                     |

---

## 次に読むべきガイド

- [Jujutsuワークフロー](./01-jujutsu-workflow.md) — 変更セットと自動リベースの実践
- [Jujutsu応用](./02-jujutsu-advanced.md) — revset、テンプレート、Git連携
- [Git→Jujutsu移行](./03-git-to-jujutsu.md) — 操作対応表と移行手順

---

## 参考文献

1. **Jujutsu公式ドキュメント** — https://martinvonz.github.io/jj/
2. **Jujutsu GitHubリポジトリ** — https://github.com/martinvonz/jj
3. **Martin von Zweigbergk** — "Jujutsu: A Git-compatible VCS" Google Tech Talk https://www.youtube.com/watch?v=bx_LGilOuE4
4. **Steve Klabnik** — "jj init" (Jujutsu introduction blog) https://steveklabnik.com/writing/jj-init
5. **Austin Seipp** — "jujutsu: A new VCS" https://austinseipp.com/posts/2024-07-10-jj-hierarchies
6. **Chris Krycho** — "jj init: Jujutsu tips and tricks" https://v5.chriskrycho.com/essays/jj-init/
