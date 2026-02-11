# Git→Jujutsu移行

> 既存のGitワークフローからJujutsuへのスムーズな移行方法を解説し、操作対応表、co-located repoの運用、チームへの段階的導入戦略を提供する。

## この章で学ぶこと

1. **Git→Jujutsu操作対応表** — 日常的なGit操作に対応するJujutsuコマンドの完全マッピング
2. **co-located repoの実践運用** — GitとJujutsuを併用する環境の設定と注意点
3. **段階的移行戦略** — 個人利用からチーム導入までの移行ロードマップ

---

## 1. Git→Jujutsu操作対応表

### 1.1 基本操作

| Git コマンド                      | Jujutsu コマンド                    | 備考                              |
|-----------------------------------|-------------------------------------|-----------------------------------|
| `git init`                        | `jj git init`                       | `--colocate`で既存gitと共存       |
| `git clone URL`                   | `jj git clone URL`                  |                                   |
| `git status`                      | `jj status` / `jj st`              |                                   |
| `git diff`                        | `jj diff`                           | ステージの概念なし                |
| `git diff --staged`               | (不要)                              | ステージが存在しない              |
| `git log`                         | `jj log`                            | デフォルトでグラフ表示            |
| `git log --oneline`               | `jj log --no-graph`                |                                   |
| `git show COMMIT`                 | `jj show REVISION`                  |                                   |

### 1.2 変更の記録

| Git コマンド                      | Jujutsu コマンド                    | 備考                              |
|-----------------------------------|-------------------------------------|-----------------------------------|
| `git add FILE`                    | (不要)                              | 自動的にworking copyに反映        |
| `git add -p`                      | `jj split`                          | 後から分割する発想                |
| `git commit -m "MSG"`             | `jj commit -m "MSG"`               | describe + new のショートカット   |
| `git commit --amend`              | `jj describe -m "MSG"`             | working copyを直接編集            |
| `git reset HEAD FILE`             | `jj restore --from @- FILE`        |                                   |
| `git checkout -- FILE`            | `jj restore FILE`                   |                                   |
| `git stash`                       | (不要)                              | jj newで新commitに移動            |
| `git stash pop`                   | `jj edit CHANGE_ID`                | 元のcommitに戻る                  |

### 1.3 ブランチ操作

| Git コマンド                      | Jujutsu コマンド                    | 備考                              |
|-----------------------------------|-------------------------------------|-----------------------------------|
| `git branch NAME`                 | `jj bookmark create NAME`          |                                   |
| `git branch -d NAME`              | `jj bookmark delete NAME`          |                                   |
| `git branch -m OLD NEW`           | `jj bookmark rename OLD NEW`       |                                   |
| `git branch -a`                   | `jj bookmark list --all`           |                                   |
| `git checkout BRANCH`             | `jj new BRANCH`                    | 新commitを作成                    |
| `git checkout -b NAME`            | `jj new && jj bookmark create NAME`|                                   |
| `git switch BRANCH`               | `jj new BRANCH`                    |                                   |

### 1.4 履歴操作

| Git コマンド                      | Jujutsu コマンド                    | 備考                              |
|-----------------------------------|-------------------------------------|-----------------------------------|
| `git rebase -i`                   | `jj rebase` / `jj squash` / `jj split` | 個別の操作に分解            |
| `git rebase main`                 | `jj rebase -d main`                |                                   |
| `git cherry-pick COMMIT`          | `jj new DEST && jj restore --from SRC` | または `jj duplicate`     |
| `git merge BRANCH`                | `jj new @ BRANCH`                  | マージcommitを作成                |
| `git revert COMMIT`               | `jj backout -r COMMIT`             |                                   |
| `git reset --hard HEAD~1`         | `jj abandon @`                      |                                   |
| `git reset --soft HEAD~1`         | `jj squash --from @ --into @-`     | 変更を親に移動                    |
| `git reflog`                      | `jj op log`                         | Operation Log                     |
| `git bisect`                      | (git bisectを使う)                  | jjにはbisect未実装（co-located）  |

### 1.5 リモート操作

| Git コマンド                      | Jujutsu コマンド                    | 備考                              |
|-----------------------------------|-------------------------------------|-----------------------------------|
| `git fetch`                       | `jj git fetch`                      |                                   |
| `git pull`                        | `jj git fetch && jj rebase -d main@origin` | pull = fetch + rebase |
| `git push`                        | `jj git push`                       |                                   |
| `git push -u origin BRANCH`       | `jj git push --bookmark NAME --allow-new` |                          |
| `git remote add NAME URL`         | `jj git remote add NAME URL`       |                                   |
| `git remote -v`                   | `jj git remote list`               |                                   |

---

## 2. 概念の対応マップ

```
┌─────────────────────────────────────────────────────┐
│  Git と Jujutsu の概念マッピング                     │
│                                                     │
│  Git                    Jujutsu                     │
│  ─────────────────────  ─────────────────────       │
│  working directory   →  working copy (= commit)     │
│  staging area (index)→  (存在しない)                 │
│  commit              →  change / commit             │
│  branch              →  bookmark                    │
│  HEAD                →  @ (working copy)            │
│  reflog              →  operation log               │
│  stash               →  (不要、全てcommit)          │
│  cherry-pick         →  duplicate / restore         │
│  rebase -i           →  squash, split, rebase       │
│  tag                 →  tag (Git互換)               │
│  submodule           →  (Git submoduleを使用)       │
│  hook                →  (未実装、Git hookを使用)    │
└─────────────────────────────────────────────────────┘
```

```
┌─────────────────────────────────────────────────────┐
│  メンタルモデルの違い                                │
│                                                     │
│  Git:                                               │
│  "変更をステージしてコミットする"                    │
│                                                     │
│  file edit → git add → git commit → git push        │
│  (4ステップ)                                        │
│                                                     │
│  Jujutsu:                                           │
│  "常にコミットの上で作業している"                    │
│                                                     │
│  file edit → jj describe → jj new → jj git push    │
│  (file editは自動的にcommitに反映)                  │
│  (3ステップ、実質的にはdescribeとnewだけ)           │
└─────────────────────────────────────────────────────┘
```

---

## 3. co-located repoの設定と運用

### 3.1 既存リポジトリでのセットアップ

```bash
# Step 1: 既存のGitリポジトリでJujutsuを初期化
$ cd my-git-project
$ jj git init --colocate
Initialized repo in "."

# Step 2: 状態の確認
$ jj log
@  rlvkpntz gaku@example.com 2025-02-11 abc12345
│  (empty) (no description set)
○  qpvuntsm gaku@example.com 2025-02-10 def67890
│  feat: latest commit
◆  zzzzzzzz root() 00000000

# Step 3: .gitignoreにjjのファイルを追加（通常は不要、.jjは自動でignore）
```

### 3.2 ディレクトリ構造

```
my-project/
├── .git/                    ← Git のデータ
│   ├── objects/
│   ├── refs/
│   └── ...
├── .jj/                     ← Jujutsu のメタデータ
│   ├── repo/
│   │   ├── store/
│   │   │   └── git_target   ← "../../../.git" へのパス
│   │   ├── op_store/        ← Operation Log
│   │   └── op_heads/
│   └── working_copy/
├── .gitignore
├── src/
└── ...
```

### 3.3 共存時の注意点

```bash
# gitコマンドを使った後にjjの状態を同期
$ git fetch origin
$ jj git import        # git→jj のref同期（通常は自動）

# jjコマンドを使った後にgitの状態を同期
$ jj git export        # jj→git のref同期（通常は自動）

# 同期の確認
$ jj git import && jj git export
```

| 操作                     | 自動同期 | 手動同期が必要な場合                  |
|--------------------------|----------|---------------------------------------|
| `jj`コマンド実行後       | Yes      | -                                     |
| `git fetch`実行後        | Yes      | jjコマンドを次に実行した時に自動import|
| `git commit`実行後       | Yes      | jjコマンドを次に実行した時に自動import|
| `git rebase`実行後       | 注意     | `jj git import`が安全                 |
| `git reset --hard`実行後 | 注意     | `jj git import`推奨                   |

---

## 4. 段階的移行戦略

### 4.1 Phase 1: 個人での試用（1-2週間）

```bash
# 既存プロジェクトにco-locatedで導入
$ cd my-project
$ jj git init --colocate

# 日常作業をjjで行ってみる
$ jj new main
$ vim src/feature.js
$ jj describe -m "feat: 新機能"
$ jj new
$ jj git push --bookmark feature --allow-new
```

```
┌────────────────────────────────────────────────────┐
│  Phase 1 の目標                                     │
│                                                    │
│  - jj log, jj status, jj diff を使いこなす         │
│  - jj new, jj describe のリズムを掴む              │
│  - jj git push/fetch の動作を理解                  │
│  - 困ったら git に戻れることを確認                  │
│                                                    │
│  この段階では Git コマンドに戻ることを許容          │
└────────────────────────────────────────────────────┘
```

### 4.2 Phase 2: 高度な機能の活用（2-4週間）

```bash
# jj edit による過去commit の直接編集
$ jj edit <change-id>
$ vim src/auth.js
$ jj new

# jj squash / jj split によるcommit整理
$ jj squash
$ jj split src/auth.js src/api.js

# revset クエリの活用
$ jj log -r 'mine() & (main..)'

# jj absorb の活用
$ jj absorb
```

### 4.3 Phase 3: チームへの紹介（4-8週間）

```bash
# チーム共有の設定ファイル
# .jj/repo/config.toml (リポジトリレベル設定)
[revset-aliases]
'immutable_heads()' = 'trunk() | tags()'
```

```
┌────────────────────────────────────────────────────┐
│  チーム導入のポイント                               │
│                                                    │
│  1. co-located repo なので既存の Git ユーザーに     │
│     影響を与えない                                 │
│                                                    │
│  2. jj ユーザーと git ユーザーが同じリポジトリで    │
│     並行して作業できる                             │
│                                                    │
│  3. リモート(GitHub/GitLab)は Git 互換なので       │
│     サーバー側の変更は不要                          │
│                                                    │
│  4. PR/MR のワークフローはそのまま維持             │
│                                                    │
│  5. CI/CD パイプラインの変更も不要                  │
└────────────────────────────────────────────────────┘
```

---

## 5. 実践: よくあるGitワークフローのJujutsu化

### 5.1 Feature Branch ワークフロー

```bash
# Git:
$ git checkout -b feature/auth main
$ vim src/auth.js
$ git add . && git commit -m "feat: 認証"
$ git push -u origin feature/auth
# PRを作成

# Jujutsu:
$ jj new main
$ vim src/auth.js
$ jj describe -m "feat: 認証"
$ jj bookmark create feature/auth -r @
$ jj git push --bookmark feature/auth --allow-new
# PRを作成
```

### 5.2 mainへの追従（rebase）

```bash
# Git:
$ git fetch origin
$ git checkout feature/auth
$ git rebase origin/main
# コンフリクト解決...
$ git push --force-with-lease

# Jujutsu:
$ jj git fetch
$ jj rebase -d main@origin
# コンフリクトがあればcommitに記録（後で解決可）
$ jj git push --bookmark feature/auth
```

### 5.3 コミットの修正

```bash
# Git:
$ git commit --amend -m "fix: corrected message"
# または
$ git rebase -i HEAD~3

# Jujutsu:
$ jj describe -m "fix: corrected message"
# または過去のcommitを直接編集
$ jj edit <change-id>
$ vim src/auth.js
# → 子commitが自動リベース
```

### 5.4 stash相当の操作

```bash
# Git:
$ git stash
$ git checkout another-branch
$ ... 作業 ...
$ git checkout original-branch
$ git stash pop

# Jujutsu:
$ jj new main          # mainの上で新しい作業を開始
$ ... 作業 ...
$ jj edit <元のchange-id>  # 元の作業に戻る
# → stashは不要、全てがcommitとして保存されている
```

### 5.5 インタラクティブrebase相当

```bash
# Git: git rebase -i HEAD~4
# pick abc Fix A
# squash def Fix B (Aに統合)
# pick ghi Feature C
# drop jkl Remove D

# Jujutsu: 個別の操作に分解
$ jj squash --from def --into abc    # BをAに統合
$ jj abandon jkl                      # Dを削除
# → 自動リベースで残りが整列
```

---

## 6. 移行時のトラブルシューティング

```bash
# 問題: jj と git の状態が不整合になった
$ jj git import    # git→jj の同期を強制
$ jj git export    # jj→git の同期を強制

# 問題: jj git push で認証エラー
$ jj git remote set-url origin git@github.com:user/repo.git  # SSH化
# または
$ gh auth setup-git  # GitHub CLI のトークンを設定

# 問題: working copy が予期しない状態になった
$ jj op log         # 操作履歴を確認
$ jj undo           # 直前の操作を取り消し
$ jj op restore <op-id>  # 特定の時点に復元

# 問題: .jjを完全に削除してGitだけに戻りたい
$ rm -rf .jj
$ git checkout .    # working copyをGitの状態に復元
```

---

## 7. アンチパターン

### アンチパターン1: co-located repoでGitとJujutsuの破壊的操作を混在させる

```bash
# NG: jjで作業した後にgit reset --hardする
$ jj new main
$ vim src/feature.js
$ jj describe -m "feat: new feature"
$ git reset --hard HEAD    # ← jjの変更が消失する可能性
# → jjのOperation Logと不整合が生じる

# OK: 一貫してjjコマンドを使う
$ jj undo                  # jjの操作取り消し
$ jj op restore <op-id>   # 特定時点への復元
```

**理由**: co-located repoではjjとgitがオブジェクトストアを共有している。gitの破壊的操作はjjのメタデータ（Operation Log等）と不整合を起こす可能性がある。

### アンチパターン2: チーム全員に一度にJujutsuを強制する

```bash
# NG: "来週からJujutsu必須です"
# → 学習コストが高く、生産性が一時的に大幅低下
# → Gitに戻したい人のモチベーション低下

# OK: co-located repoで段階的に導入
# Phase 1: 興味のあるメンバーから試用
# Phase 2: 成功事例を共有
# Phase 3: 公式にサポートするが強制しない
# → GitユーザーとJujutsuユーザーが共存可能
```

**理由**: Jujutsuはco-located repoにより、Gitユーザーに影響を与えずに導入できる。強制ではなく段階的な移行が最も効果的。

---

## 8. FAQ

### Q1. Jujutsuに移行するとGitの履歴は失われるか？

**A1.** いいえ、**全く失われません**。Jujutsuは内部的にGitのオブジェクトストアを使用しており、co-located repoでは`.git/`がそのまま維持されます。全てのコミット履歴、タグ、ブランチ、reflogがそのまま保持されます。

### Q2. GitHub/GitLabのPRワークフローはそのまま使えるか？

**A2.** はい、**完全に互換性があります**。`jj git push`はGitのブランチとしてリモートにpushされるため、GitHub/GitLabはそれを通常のGitブランチとして認識します。PR/MRの作成、レビュー、マージは従来通りです。

```bash
# JujutsuでPRを作成するフロー
$ jj new main
$ vim src/feature.js
$ jj describe -m "feat: new feature"
$ jj bookmark create feature-branch -r @
$ jj git push --bookmark feature-branch --allow-new
# → GitHubでPRを作成（通常のGitと全く同じ）
```

### Q3. Jujutsuにはまだ不足している機能はあるか？

**A3.** 2025年時点で以下の機能がGitと比較して未実装または限定的です。

| 機能               | 状態                              | 代替手段                         |
|--------------------|-----------------------------------|----------------------------------|
| `bisect`           | 未実装                            | co-locatedでgit bisectを使用     |
| `blame`            | 未実装                            | co-locatedでgit blameを使用      |
| hooks              | 未実装                            | co-locatedでGit hooksを使用      |
| submodule          | 部分サポート                      | Git submoduleコマンドを併用      |
| sparse checkout    | 未実装                            | git sparse-checkoutを使用        |
| GUI ツール         | 限定的                            | lazyjj（TUIツール）              |
| IDE統合            | 一部IDE対応中                     | co-locatedでGit IDE統合を使用    |

---

## まとめ

| 概念                   | 要点                                                          |
|------------------------|---------------------------------------------------------------|
| co-located repo        | .git/と.jj/が共存、GitとJujutsu両方使用可能                  |
| jj git init --colocate | 既存Gitリポジトリに即座にJujutsuを追加                       |
| 操作対応              | git add→不要、git commit→jj commit、git branch→jj bookmark   |
| メンタルモデル         | "ステージ→コミット"から"常にcommit上で作業"に転換            |
| 自動同期               | jjコマンド実行時にgitとの同期が自動的に行われる              |
| 段階的移行             | 個人試用→高度な活用→チーム紹介の3段階                        |
| undo                   | jj undo / jj op restoreで安全に復元可能                      |

---

## 次に読むべきガイド

- [Jujutsu入門](./00-jujutsu-introduction.md) — 基本概念と設計思想の復習
- [Jujutsuワークフロー](./01-jujutsu-workflow.md) — 変更セットと自動リベースの実践
- [Jujutsu応用](./02-jujutsu-advanced.md) — revset、テンプレート、Git連携

---

## 参考文献

1. **Jujutsu公式ドキュメント** — "Git comparison" https://martinvonz.github.io/jj/latest/git-comparison/
2. **Jujutsu公式ドキュメント** — "Git compatibility" https://martinvonz.github.io/jj/latest/git-compatibility/
3. **Steve Klabnik** — "jj init" (Gitからの移行体験記) https://steveklabnik.com/writing/jj-init
4. **Jujutsu GitHub Discussions** — Migration tips and tricks https://github.com/martinvonz/jj/discussions
