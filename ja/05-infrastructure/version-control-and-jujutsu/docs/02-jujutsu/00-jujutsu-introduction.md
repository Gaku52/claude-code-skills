# Jujutsu入門

> 次世代バージョン管理システムJujutsu（jj）の設計思想とGitとの根本的な違いを理解し、基本操作をマスターする。

## この章で学ぶこと

1. **Jujutsuの設計思想** — Gitの課題を解決するために何が再設計されたか
2. **Gitとの根本的な違い** — working copy = commit、自動リベース、コンフリクトのファーストクラスサポート
3. **基本操作の習得** — jj init, jj new, jj describe, jj log, jj diff
4. **初期設定とカスタマイズ** — インストールから日常的に快適に使うための設定まで
5. **Jujutsuの内部構造** — Operation Log、ストレージバックエンド、スナップショッティングの仕組み


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
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


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

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
