# バージョン管理と Jujutsu

> Git は開発の基盤だが、その内部構造を理解する者は少ない。Git の内部オブジェクトモデル、高度な操作、そして次世代 VCS である Jujutsu（jj）まで、バージョン管理の深層を解説する。

## このSkillの対象者

- Git の内部構造を理解したいエンジニア
- 高度な Git 操作（rebase、bisect、reflog 等）を習得したい方
- 次世代 VCS（Jujutsu）に興味がある方

## 前提知識

- Git の基本操作（add、commit、push、pull、branch）
- ターミナルの基本操作

## 学習ガイド

### 00-git-internals — Git 内部構造

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-git-internals/00-object-model.md]] | blob/tree/commit/tag オブジェクト、SHA-1、DAG |
| 01 | [[docs/00-git-internals/01-refs-and-packfiles.md]] | refs、HEAD、reflog、packfile、gc、transfer protocol |
| 02 | [[docs/00-git-internals/02-merge-algorithms.md]] | 3-way merge、recursive、ort、conflict resolution |
| 03 | [[docs/00-git-internals/03-index-and-staging.md]] | index ファイル、staging area、assume-unchanged、skip-worktree |

### 01-advanced-git — Git 高度な操作

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-advanced-git/00-rebase-mastery.md]] | interactive rebase、fixup、autosquash、rebase onto |
| 01 | [[docs/01-advanced-git/01-history-rewriting.md]] | filter-branch、filter-repo、BFG、大容量ファイル削除 |
| 02 | [[docs/01-advanced-git/02-bisect-and-debug.md]] | git bisect（自動バイナリサーチ）、blame、log -S/-G |
| 03 | [[docs/01-advanced-git/03-worktree-and-submodule.md]] | worktree、submodule、subtree、sparse-checkout |
| 04 | [[docs/01-advanced-git/04-hooks-and-automation.md]] | Git Hooks 全種類、husky、lefthook、自動化レシピ |

### 02-jujutsu — Jujutsu（jj）

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-jujutsu/00-jj-introduction.md]] | Jujutsu 概要、Git との違い、インストール、基本概念 |
| 01 | [[docs/02-jujutsu/01-jj-workflow.md]] | jj のワークフロー、変更セット、ブランチレス開発 |
| 02 | [[docs/02-jujutsu/02-jj-advanced.md]] | コンフリクト管理、リビジョンセット、Git 連携、移行 |

## クイックリファレンス

```
Git 内部オブジェクト:
  blob   — ファイルの内容
  tree   — ディレクトリ構造
  commit — スナップショット + メタデータ
  tag    — 名前付き参照

Git 高度コマンド:
  git rebase -i HEAD~5        — 直近5コミットを編集
  git bisect start/bad/good   — バグ混入コミット特定
  git reflog                  — HEAD の移動履歴
  git worktree add ../feature — 別ディレクトリでブランチ作業
  git log -S "keyword"        — コード変更の検索

Jujutsu（jj）基本:
  jj init                     — リポジトリ初期化
  jj status                   — 状態確認
  jj describe -m "message"    — コミットメッセージ設定
  jj new                      — 新しい変更セット作成
  jj squash                   — 親に統合
  jj git push                 — Git リモートにプッシュ
```

## 参考文献

1. Chacon, S. & Straub, B. "Pro Git." git-scm.com/book, 2024.
2. Git. "Git Internals." git-scm.com/book/en/v2/Git-Internals, 2024.
3. Jujutsu. "Documentation." martinvonz.github.io/jj, 2024.
