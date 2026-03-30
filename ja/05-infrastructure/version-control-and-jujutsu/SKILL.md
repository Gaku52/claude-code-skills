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

### 01-advanced-git — Git 高度な操作

| # | ファイル | 内容 |
|---|---------|------|

### 02-jujutsu — Jujutsu（jj）

| # | ファイル | 内容 |
|---|---------|------|

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
