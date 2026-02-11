# Jujutsu入門

> 次世代バージョン管理システムJujutsu（jj）の設計思想とGitとの根本的な違いを理解し、基本操作をマスターする。

## この章で学ぶこと

1. **Jujutsuの設計思想** — Gitの課題を解決するために何が再設計されたか
2. **Gitとの根本的な違い** — working copy = commit、自動リベース、コンフリクトのファーストクラスサポート
3. **基本操作の習得** — jj init, jj new, jj describe, jj log, jj diff

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

---

## 2. インストールと初期設定

```bash
# macOS (Homebrew)
$ brew install jj

# Linux (cargo)
$ cargo install --locked jujutsu-cli

# バージョン確認
$ jj version
jj 0.25.0

# 初期設定
$ jj config set --user user.name "Gaku"
$ jj config set --user user.email "gaku@example.com"

# エディタの設定
$ jj config set --user ui.editor "vim"

# 設定の確認
$ jj config list
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
├─────────────────────────────────────────────────────┤
│  Jujutsu のメンタルモデル                            │
│                                                     │
│  Working Copy = 最新のCommit（常に自動記録）        │
│                                                     │
│  ファイルを編集 → 自動的にworking copy commitに反映 │
│  "jj new" → 新しい空のcommitを開始                  │
│                                                     │
│  ステージングエリアは存在しない                      │
└─────────────────────────────────────────────────────┘
```

### 3.2 changeとcommitの違い

```
┌────────────────────────────────────────────────────┐
│  Jujutsuの用語                                      │
│                                                    │
│  change:  論理的な変更単位（change IDで識別）       │
│           change ID は rebase しても変わらない      │
│                                                    │
│  commit:  特定時点のスナップショット                 │
│           commit ID (SHA-1) は内容変更で変わる      │
│                                                    │
│  例:                                               │
│  change "abc" の commit が def123 だった場合        │
│  rebase すると commit は ghi456 に変わるが          │
│  change ID は "abc" のまま                          │
│                                                    │
│  → change ID で追跡すれば rebase しても参照可能    │
└────────────────────────────────────────────────────┘
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

# 新しいcommitを開始（現在の変更を確定し、新しい空commitへ移動）
$ jj new
# → 今の変更が確定され、その上に新しい空のworking copy commitが作成される

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
└─────────────────────────────────────────────────────┘
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
```

### 4.5 ファイルの追跡制御

```bash
# .gitignore と同じ仕組み（Jujutsuは.gitignoreを読む）
$ echo "node_modules/" >> .gitignore

# 特定ファイルを復元（working copyの変更を取り消す）
$ jj restore --from @- src/auth.js
# @- = working copyの親commit

# 全変更を取り消す
$ jj restore
```

---

## 5. コンフリクトのファーストクラスサポート

Gitではコンフリクトが発生するとrebase/mergeが中断されるが、Jujutsuでは**コンフリクト状態がcommitに記録され、後から解決できる**。

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

```
┌────────────────────────────────────────────────────┐
│  Git vs Jujutsu: コンフリクトの扱い                 │
│                                                    │
│  Git:                                              │
│  rebase中にコンフリクト                             │
│    → rebaseが中断                                  │
│    → 即座に解決が必要                               │
│    → --abort で全てやり直し or --continue           │
│                                                    │
│  Jujutsu:                                          │
│  rebase中にコンフリクト                             │
│    → rebaseは完了する                               │
│    → コンフリクト状態がcommitに記録される           │
│    → 好きなタイミングで解決できる                   │
│    → 他の作業を先にすることも可能                   │
└────────────────────────────────────────────────────┘
```

---

## 6. Operation Log — 全操作のundo/redo

```bash
# 操作ログの確認
$ jj op log
@  abc12345 gaku@... 2025-02-11 15:30:00
│  new empty commit
○  def67890 gaku@... 2025-02-11 15:25:00
│  describe commit rlvkpntz
○  789abcde gaku@... 2025-02-11 15:20:00
│  snapshot working copy

# 直前の操作を取り消す
$ jj undo
# → 直前のjjコマンドの効果が完全に取り消される

# 特定の操作時点に戻る
$ jj op restore def67890
```

---

## 7. アンチパターン

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

**理由**: Jujutsuにはステージングエリアが存在しない。ファイルの変更は自動的にworking copy commitに記録される。

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

**理由**: Jujutsuではworking copyが常に最新commitと同一視される。`jj new`を実行しないと、全ての変更が1つのcommitに蓄積されてしまう。

---

## 8. FAQ

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
```

### Q3. Jujutsuのworking copy commitは毎回コミットされるのか？パフォーマンスは大丈夫か？

**A3.** Jujutsuは「スナップショッティング」という仕組みで、jjコマンドの実行時にworking copyの状態をcommitに反映します。ファイル保存のたびにcommitが作られるわけではなく、`jj status`、`jj log`、`jj diff`などのコマンド実行時にスナップショットが取得されます。watchman連携で高速化も可能です。

---

## まとめ

| 概念                   | 要点                                                          |
|------------------------|---------------------------------------------------------------|
| working copy = commit  | ステージングなし、編集が即座にcommitに反映                    |
| change ID              | rebaseで変わらない安定した識別子                              |
| commit ID              | Git互換のSHA-1、内容変更で変化する                           |
| jj new                 | 新しいcommitを開始、前の変更を確定                           |
| jj describe            | working copy commitにメッセージを設定                         |
| コンフリクト記録       | rebase/merge中断なし、commitにコンフリクト状態を保存         |
| Operation Log          | 全操作の記録、undo/redoが可能                                |
| co-located repo        | .git/と.jj/が共存、Git/Jujutsu両方使用可能                   |

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
