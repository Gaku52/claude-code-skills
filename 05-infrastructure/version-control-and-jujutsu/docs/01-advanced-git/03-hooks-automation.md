# Git Hooks

> Git Hooksの仕組みとライフサイクルを理解し、pre-commit、commit-msg、pre-pushなどのフックをhusky・lint-stagedと組み合わせて開発ワークフローを自動化する方法を解説する。

## この章で学ぶこと

1. **Git Hooksの種類とライフサイクル** — クライアントサイド/サーバーサイドの各フックの発火タイミングと用途
2. **husky + lint-staged による自動化** — モダンなフック管理ツールの導入と設定
3. **実践的なフック設計パターン** — CI/CDとの連携、チーム共有、パフォーマンス最適化
4. **サーバーサイドフックの詳細** — pre-receive/update/post-receiveの実装とポリシー適用
5. **代替ツール（lefthook, pre-commit framework）** — husky以外の選択肢と使い分け
6. **モノレポ対応とパフォーマンス最適化** — 大規模プロジェクトでのフック運用戦略

---

## 1. Git Hooksの基本

### 1.1 Hooksの保存場所

```bash
# デフォルトのhooksディレクトリ
$ ls .git/hooks/
applypatch-msg.sample     pre-commit.sample
commit-msg.sample         pre-merge-commit.sample
fsmonitor-watchman.sample pre-push.sample
post-update.sample        pre-rebase.sample
pre-applypatch.sample     prepare-commit-msg.sample
pre-auto-gc.sample        update.sample

# sampleを外して実行可能にするとhookが有効になる
$ cp .git/hooks/pre-commit.sample .git/hooks/pre-commit
$ chmod +x .git/hooks/pre-commit

# hooksディレクトリの場所を変更
$ git config core.hooksPath .githooks
# → プロジェクト内の .githooks/ をhooksディレクトリとして使用

# グローバルにhooksディレクトリを設定
$ git config --global core.hooksPath ~/.git-hooks
# → 全リポジトリで共通のフックを使用

# 現在のhooksPathを確認
$ git config --get core.hooksPath
# → .husky（huskyを使っている場合）
```

### 1.2 Hooksの実行環境

```bash
# フックが実行される際の環境変数
$ cat .git/hooks/pre-commit
#!/bin/sh

# 現在の作業ディレクトリ（リポジトリのルート）
echo "PWD: $PWD"

# GIT_DIR: .gitディレクトリのパス
echo "GIT_DIR: $GIT_DIR"

# GIT_WORK_TREE: ワークツリーのパス
echo "GIT_WORK_TREE: $GIT_WORK_TREE"

# GIT_INDEX_FILE: インデックスファイルのパス
echo "GIT_INDEX_FILE: $GIT_INDEX_FILE"

# GIT_AUTHOR_NAME/EMAIL: コミットの著者情報
echo "Author: $GIT_AUTHOR_NAME <$GIT_AUTHOR_EMAIL>"

# フックの実行シェルはshebangで決まる
# #!/bin/sh — POSIX sh
# #!/bin/bash — Bash
# #!/usr/bin/env python3 — Python
# #!/usr/bin/env node — Node.js
```

```
┌──────────────────────────────────────────────────────┐
│  フック実行時の環境                                    │
│                                                      │
│  作業ディレクトリ: リポジトリのルート                 │
│  PATH: 通常のPATHが使用される                        │
│  stdin: フックによって異なる（pre-pushではref情報）   │
│  引数: フックによって異なる                          │
│                                                      │
│  注意:                                               │
│  - フックはnon-interactiveシェルで実行される          │
│  - ~/.bashrc は読み込まれない                         │
│  - nvm等のシェル初期化が必要な場合は明示的に読み込む │
│                                                      │
│  #!/bin/sh                                           │
│  export NVM_DIR="$HOME/.nvm"                         │
│  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"    │
│  npx lint-staged                                     │
└──────────────────────────────────────────────────────┘
```

### 1.3 Hooksのライフサイクル

```
┌─────────────────────────────────────────────────────┐
│  git commit のフックライフサイクル                    │
│                                                     │
│  git commit 実行                                    │
│       │                                             │
│       ▼                                             │
│  [pre-commit]  ← lint, format, テスト               │
│       │ exit 0 で続行 / 非0 で中断                  │
│       ▼                                             │
│  [prepare-commit-msg]  ← メッセージテンプレート     │
│       │                                             │
│       ▼                                             │
│  エディタでメッセージ編集                            │
│       │                                             │
│       ▼                                             │
│  [commit-msg]  ← メッセージの検証                   │
│       │ exit 0 で続行 / 非0 で中断                  │
│       ▼                                             │
│  コミット作成                                       │
│       │                                             │
│       ▼                                             │
│  [post-commit]  ← 通知、ログ記録                    │
│                                                     │
├─────────────────────────────────────────────────────┤
│  git push のフックライフサイクル                     │
│                                                     │
│  git push 実行                                      │
│       │                                             │
│       ▼                                             │
│  [pre-push]  ← テスト実行、ブランチ名チェック       │
│       │ exit 0 で続行 / 非0 で中断                  │
│       ▼                                             │
│  リモートに送信                                     │
│       │                                             │
│       ▼ (サーバーサイド)                             │
│  [pre-receive]  ← ポリシーチェック                  │
│  [update]       ← ブランチごとのチェック            │
│  [post-receive] ← CI/CD起動、通知                   │
├─────────────────────────────────────────────────────┤
│  git merge のフックライフサイクル                    │
│                                                     │
│  git merge 実行                                     │
│       │                                             │
│       ▼                                             │
│  [pre-merge-commit]  ← マージ前チェック（Git 2.24+）│
│       │ exit 0 で続行 / 非0 で中断                  │
│       ▼                                             │
│  [prepare-commit-msg]  ← マージメッセージ準備       │
│       │                                             │
│       ▼                                             │
│  [commit-msg]  ← マージメッセージの検証             │
│       │                                             │
│       ▼                                             │
│  マージコミット作成                                  │
│       │                                             │
│       ▼                                             │
│  [post-merge]  ← 依存関係更新、ビルド               │
├─────────────────────────────────────────────────────┤
│  git rebase のフックライフサイクル                   │
│                                                     │
│  git rebase 実行                                    │
│       │                                             │
│       ▼                                             │
│  [pre-rebase]  ← rebase可否の判断                   │
│       │ exit 0 で続行 / 非0 で中断                  │
│       ▼                                             │
│  各コミットの適用:                                   │
│    [pre-commit] → [commit-msg] → [post-commit]     │
│       │                                             │
│       ▼                                             │
│  [post-rewrite]  ← amend/rebase完了後               │
│       引数: "amend" or "rebase"                     │
│       stdin: old-sha new-sha のペア（各行1つ）      │
└─────────────────────────────────────────────────────┘
```

### 1.4 クライアントサイドフック一覧

| フック                 | タイミング                    | 用途                            | 引数 |
|------------------------|-------------------------------|---------------------------------|------|
| `pre-commit`           | commit前                      | lint、format、テスト            | なし |
| `prepare-commit-msg`   | メッセージ編集前              | テンプレート挿入                | メッセージファイルパス, ソース, SHA |
| `commit-msg`           | メッセージ編集後              | メッセージ形式の検証            | メッセージファイルパス |
| `post-commit`          | commit後                      | 通知                            | なし |
| `pre-rebase`           | rebase前                      | rebase可否の判断                | upstream, branch |
| `post-rewrite`         | amend/rebase後                | 関連処理の実行                  | "amend" or "rebase" |
| `pre-push`             | push前                        | テスト、ブランチ保護            | remote名, remote URL |
| `post-checkout`        | checkout後                    | 依存関係の更新                  | prev HEAD, new HEAD, branch flag |
| `post-merge`           | merge後                       | 依存関係の更新                  | squash flag |
| `pre-auto-gc`          | GC前                          | GCの制御                        | なし |
| `pre-merge-commit`     | merge commit前（2.24+）       | マージ前チェック                | なし |
| `reference-transaction`| ref更新時（2.28+）            | ref変更の追跡                   | "prepared"/"committed"/"aborted" |
| `fsmonitor-watchman`   | ファイル変更監視              | パフォーマンス向上              | バージョン, last-update-token |

### 1.5 サーバーサイドフック一覧

| フック            | タイミング            | 用途                                | 引数/stdin |
|-------------------|-----------------------|-------------------------------------|------------|
| `pre-receive`     | push受信前            | 全refの一括チェック                 | stdin: old-sha new-sha refname |
| `update`          | 各ref更新前           | ブランチ単位のポリシー適用          | refname old-sha new-sha |
| `post-receive`    | push受信後            | CI/CDトリガー、チャット通知         | stdin: old-sha new-sha refname |
| `post-update`     | ref更新後             | `git update-server-info`の実行等    | 更新されたref名のリスト |

---

## 2. 手動でのフック作成

### 2.1 pre-commitフック

```bash
#!/bin/sh
# .git/hooks/pre-commit

# ステージされたファイルに対してlintを実行
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(js|ts|jsx|tsx)$')

if [ -z "$STAGED_FILES" ]; then
  exit 0
fi

echo "Running ESLint on staged files..."
npx eslint $STAGED_FILES
LINT_EXIT=$?

if [ $LINT_EXIT -ne 0 ]; then
  echo "ESLint failed. Commit aborted."
  exit 1
fi

echo "ESLint passed."
exit 0
```

### 2.2 高度なpre-commitフック（部分ステージ対応）

```bash
#!/bin/sh
# .git/hooks/pre-commit
# 部分的にステージされたファイルを正しく処理するpre-commit

set -e

# ステージされたファイルを取得
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACMR)

if [ -z "$STAGED_FILES" ]; then
  exit 0
fi

# ステージされた内容のみを一時ディレクトリにエクスポート
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# ステージされた内容をチェックアウト
git checkout-index --prefix="$TMPDIR/" -a

# 一時ディレクトリでlintを実行（ステージされた内容のみ）
JS_FILES=$(echo "$STAGED_FILES" | grep -E '\.(js|ts|jsx|tsx)$' || true)
if [ -n "$JS_FILES" ]; then
  echo "Linting staged JavaScript/TypeScript files..."
  # ファイルパスを一時ディレクトリのパスに変換
  LINT_FILES=""
  for file in $JS_FILES; do
    if [ -f "$TMPDIR/$file" ]; then
      LINT_FILES="$LINT_FILES $TMPDIR/$file"
    fi
  done

  if [ -n "$LINT_FILES" ]; then
    npx eslint --no-eslintrc --config .eslintrc.json $LINT_FILES
  fi
fi

# Pythonファイルのチェック
PY_FILES=$(echo "$STAGED_FILES" | grep -E '\.py$' || true)
if [ -n "$PY_FILES" ]; then
  echo "Linting staged Python files..."
  for file in $PY_FILES; do
    if [ -f "$TMPDIR/$file" ]; then
      python -m flake8 "$TMPDIR/$file"
    fi
  done
fi

echo "All checks passed."
exit 0
```

### 2.3 commit-msgフック

```bash
#!/bin/sh
# .git/hooks/commit-msg

# Conventional Commitsの形式を検証
COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

PATTERN="^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .{1,72}$"

if ! echo "$COMMIT_MSG" | head -1 | grep -qE "$PATTERN"; then
  echo "ERROR: コミットメッセージがConventional Commitsの形式に従っていません。"
  echo ""
  echo "形式: <type>(<scope>): <description>"
  echo "例:   feat(auth): ログイン機能を追加"
  echo ""
  echo "type: feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
  exit 1
fi

exit 0
```

### 2.4 prepare-commit-msgフック

```bash
#!/bin/sh
# .git/hooks/prepare-commit-msg
# コミットメッセージにブランチ名から情報を自動付加

COMMIT_MSG_FILE=$1
COMMIT_SOURCE=$2  # message, template, merge, squash, commit
SHA1=$3

# -m でメッセージが指定された場合はスキップ
if [ "$COMMIT_SOURCE" = "message" ]; then
  exit 0
fi

# マージコミットの場合はスキップ
if [ "$COMMIT_SOURCE" = "merge" ]; then
  exit 0
fi

# ブランチ名からチケット番号を抽出
BRANCH_NAME=$(git symbolic-ref --short HEAD 2>/dev/null)
TICKET=$(echo "$BRANCH_NAME" | grep -oE '[A-Z]+-[0-9]+' || true)

if [ -n "$TICKET" ]; then
  # メッセージの末尾にチケット番号を追加
  if ! grep -qF "$TICKET" "$COMMIT_MSG_FILE"; then
    echo "" >> "$COMMIT_MSG_FILE"
    echo "Refs: $TICKET" >> "$COMMIT_MSG_FILE"
  fi
fi

# テンプレートの挿入（sourceがない場合 = 通常のコミット）
if [ -z "$COMMIT_SOURCE" ]; then
  TEMPLATE="

# --- コミットメッセージガイド ---
# feat: 新機能の追加
# fix: バグ修正
# docs: ドキュメントの変更
# style: コードスタイルの変更（動作に影響なし）
# refactor: リファクタリング
# perf: パフォーマンス改善
# test: テストの追加・修正
# build: ビルドシステム・依存関係の変更
# ci: CI/CDの変更
# chore: その他の変更
#
# 形式: <type>(<scope>): <description>
# 例:   feat(auth): ログイン機能を追加"
  echo "$TEMPLATE" >> "$COMMIT_MSG_FILE"
fi
```

### 2.5 pre-pushフック

```bash
#!/bin/sh
# .git/hooks/pre-push

# mainブランチへの直接pushを禁止
REMOTE=$1
URL=$2

while read LOCAL_REF LOCAL_SHA REMOTE_REF REMOTE_SHA; do
  if echo "$REMOTE_REF" | grep -qE "refs/heads/(main|master)"; then
    echo "ERROR: main/masterへの直接pushは禁止されています。"
    echo "PRを作成してください。"
    exit 1
  fi
done

# push前にテストを実行
echo "Running tests before push..."
npm test
exit $?
```

### 2.6 post-checkoutフック

```bash
#!/bin/sh
# .git/hooks/post-checkout
# ブランチ切り替え時に依存関係を自動更新

PREV_HEAD=$1
NEW_HEAD=$2
BRANCH_CHECKOUT=$3  # 1 = ブランチ切替, 0 = ファイルチェックアウト

# ファイルのチェックアウトの場合はスキップ
if [ "$BRANCH_CHECKOUT" = "0" ]; then
  exit 0
fi

# 同じコミットの場合はスキップ
if [ "$PREV_HEAD" = "$NEW_HEAD" ]; then
  exit 0
fi

# package.jsonに変更があるか確認
if git diff --name-only "$PREV_HEAD" "$NEW_HEAD" | grep -q "package-lock.json"; then
  echo "package-lock.json changed. Running npm install..."
  npm install
fi

# Gemfileに変更があるか確認
if git diff --name-only "$PREV_HEAD" "$NEW_HEAD" | grep -q "Gemfile.lock"; then
  echo "Gemfile.lock changed. Running bundle install..."
  bundle install
fi

# requirements.txtに変更があるか確認
if git diff --name-only "$PREV_HEAD" "$NEW_HEAD" | grep -q "requirements.txt"; then
  echo "requirements.txt changed. Running pip install..."
  pip install -r requirements.txt
fi

# マイグレーションファイルに変更があるか確認
if git diff --name-only "$PREV_HEAD" "$NEW_HEAD" | grep -q "migrations/"; then
  echo "Migration files changed. You may need to run migrations."
  echo "  rails db:migrate     (Rails)"
  echo "  python manage.py migrate  (Django)"
fi

# .envファイルのテンプレートに変更があるか確認
if git diff --name-only "$PREV_HEAD" "$NEW_HEAD" | grep -q ".env.example"; then
  echo "⚠ .env.example が変更されています。.envを確認してください。"
fi
```

### 2.7 post-mergeフック

```bash
#!/bin/sh
# .git/hooks/post-merge
# マージ後に依存関係を自動更新

SQUASH_MERGE=$1  # 1 = squash merge

# 直近のマージで変更されたファイルを確認
CHANGED_FILES=$(git diff-tree --name-only -r ORIG_HEAD HEAD)

# Node.js依存関係
if echo "$CHANGED_FILES" | grep -q "package-lock.json\|yarn.lock\|pnpm-lock.yaml"; then
  echo "Dependencies changed. Installing..."
  if [ -f "pnpm-lock.yaml" ]; then
    pnpm install
  elif [ -f "yarn.lock" ]; then
    yarn install
  else
    npm install
  fi
fi

# データベースマイグレーション
if echo "$CHANGED_FILES" | grep -qE "db/migrate|migrations/"; then
  echo ""
  echo "================================================================"
  echo "  DATABASE MIGRATION DETECTED"
  echo "  新しいマイグレーションファイルが追加されました。"
  echo "  データベースのマイグレーションを実行してください。"
  echo "================================================================"
  echo ""
fi

# サブモジュールの更新
if echo "$CHANGED_FILES" | grep -q ".gitmodules"; then
  echo "Submodules changed. Updating..."
  git submodule update --init --recursive
fi
```

### 2.8 pre-rebaseフック

```bash
#!/bin/sh
# .git/hooks/pre-rebase
# 保護ブランチのrebaseを防止

UPSTREAM=$1
BRANCH=${2:-$(git symbolic-ref --short HEAD)}

# 保護ブランチの一覧
PROTECTED_BRANCHES="main master develop release"

for protected in $PROTECTED_BRANCHES; do
  if [ "$BRANCH" = "$protected" ]; then
    echo "ERROR: '$protected' ブランチのrebaseは禁止されています。"
    echo "代わりにmergeを使用してください。"
    exit 1
  fi
done

# リモートにpush済みのコミットをrebaseしようとしているか確認
REMOTE_BRANCH="origin/$BRANCH"
if git rev-parse --verify "$REMOTE_BRANCH" > /dev/null 2>&1; then
  LOCAL_ONLY=$(git log --oneline "$REMOTE_BRANCH..HEAD" | wc -l | tr -d ' ')
  TOTAL=$(git log --oneline "$UPSTREAM..HEAD" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$TOTAL" -gt "$LOCAL_ONLY" ]; then
    echo "WARNING: push済みのコミットがrebaseの対象に含まれています。"
    echo "force pushが必要になります。続行しますか？"
    # non-interactiveなフック内では自動的に中断
    # 対話的にする場合は exec < /dev/tty を使う
    exit 1
  fi
fi

exit 0
```

---

## 3. husky — モダンなフック管理

### 3.1 huskyの導入

```bash
# husky v9+ のインストール
$ npm install --save-dev husky

# huskyの初期化
$ npx husky init
# → .husky/ ディレクトリが作成される
# → package.json に "prepare": "husky" が追加される

# プロジェクト構造
.husky/
├── _/
│   ├── .gitignore
│   └── husky.sh
├── pre-commit       ← pre-commitフック
└── commit-msg       ← commit-msgフック
```

### 3.2 huskyフックの作成

```bash
# pre-commitフックの作成
$ echo "npx lint-staged" > .husky/pre-commit

# commit-msgフックの作成
$ echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg

# pre-pushフックの作成
$ echo "npm test" > .husky/pre-push
```

```
┌────────────────────────────────────────────────────┐
│  huskyの動作原理                                    │
│                                                    │
│  1. npm install 時に "prepare" スクリプトが実行    │
│  2. husky が core.hooksPath を .husky に設定       │
│  3. git commit 時に .husky/pre-commit が実行       │
│                                                    │
│  .husky/pre-commit の中身:                          │
│  ┌────────────────────────────────┐                │
│  │ npx lint-staged               │                │
│  └────────────────────────────────┘                │
│                                                    │
│  ※ v9以降はシンプルなシェルスクリプト              │
│  ※ チームメンバーもnpm installだけで自動設定       │
└────────────────────────────────────────────────────┘
```

### 3.3 huskyのバージョン間の違い

```
┌────────────────────────────────────────────────────────────┐
│  husky バージョン比較                                       │
│                                                            │
│  v4（旧式）                                                │
│  ┌──────────────────────────────────────┐                  │
│  │ package.json:                        │                  │
│  │ {                                    │                  │
│  │   "husky": {                         │                  │
│  │     "hooks": {                       │                  │
│  │       "pre-commit": "lint-staged"    │                  │
│  │     }                                │                  │
│  │   }                                  │                  │
│  │ }                                    │                  │
│  │ → .git/hooks/ を直接書き換え         │                  │
│  │ → node_modules内のhusky.shが仲介    │                  │
│  └──────────────────────────────────────┘                  │
│                                                            │
│  v9+（現行）                                               │
│  ┌──────────────────────────────────────┐                  │
│  │ .husky/pre-commit:                   │                  │
│  │ npx lint-staged                      │                  │
│  │                                      │                  │
│  │ → core.hooksPath = .husky を設定     │                  │
│  │ → シンプルなシェルスクリプト          │                  │
│  │ → Git native機能を活用               │                  │
│  └──────────────────────────────────────┘                  │
│                                                            │
│  移行のポイント:                                           │
│  - v4 → v9: package.jsonの"husky"セクション削除           │
│  - .husky/ ディレクトリに個別ファイルを作成               │
│  - "prepare": "husky" をscriptsに追加                     │
│  - HUSKY_GIT_PARAMS → $1 に変更                           │
└────────────────────────────────────────────────────────────┘
```

### 3.4 huskyの高度な設定

```bash
# .husky/pre-commit — 条件付き実行
#!/bin/sh

# CI環境ではスキップ
[ -n "$CI" ] && exit 0

# マージ中はスキップ
[ -f ".git/MERGE_HEAD" ] && exit 0

# rebase中はスキップ
[ -d ".git/rebase-merge" ] || [ -d ".git/rebase-apply" ] && exit 0

# lint-stagedを実行
npx lint-staged
```

```bash
# .husky/pre-push — 高度なpre-push
#!/bin/sh

REMOTE=$1
URL=$2

# ブランチ名を取得
BRANCH=$(git symbolic-ref --short HEAD)

# mainブランチへの直接pushを禁止
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo "ERROR: $BRANCH への直接pushは禁止されています。"
  echo "PRを作成してください。"
  exit 1
fi

# WIPコミットが含まれていないか確認
WIP_COMMITS=$(git log --oneline @{u}..HEAD 2>/dev/null | grep -i "wip" || true)
if [ -n "$WIP_COMMITS" ]; then
  echo "WARNING: WIPコミットが含まれています:"
  echo "$WIP_COMMITS"
  echo ""
  echo "WIPコミットを整理してからpushしてください。"
  echo "git rebase -i で整理できます。"
  exit 1
fi

# テストを実行（関連ファイルのみ）
CHANGED_FILES=$(git diff --name-only @{u}..HEAD 2>/dev/null | grep -E '\.(js|ts|jsx|tsx)$' || true)
if [ -n "$CHANGED_FILES" ]; then
  echo "Running tests for changed files..."
  npx jest --bail --findRelatedTests $CHANGED_FILES
fi

exit 0
```

---

## 4. lint-staged — ステージファイルのみを処理

### 4.1 lint-stagedの設定

```bash
# インストール
$ npm install --save-dev lint-staged
```

```json
// package.json
{
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{css,scss}": [
      "stylelint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml}": [
      "prettier --write"
    ]
  }
}
```

```bash
# または .lintstagedrc.js で設定
module.exports = {
  '*.{js,jsx,ts,tsx}': (filenames) => [
    `eslint --fix ${filenames.join(' ')}`,
    `prettier --write ${filenames.join(' ')}`,
    `jest --bail --findRelatedTests ${filenames.join(' ')}`,
  ],
};
```

### 4.2 lint-stagedの動作フロー

```
┌─────────────────────────────────────────────────────┐
│  lint-staged の実行フロー                            │
│                                                     │
│  1. ステージされたファイルの一覧を取得              │
│     git diff --cached --name-only --diff-filter=ACMR│
│                                                     │
│  2. ファイルパターンに基づいてコマンドをマッチ      │
│     *.js → eslint --fix, prettier --write           │
│     *.css → stylelint --fix                         │
│                                                     │
│  3. 変更をstash（安全のため）                       │
│     → ステージされていない変更を一時退避            │
│                                                     │
│  4. 各コマンドを順次実行                            │
│     eslint --fix src/auth.js src/utils.js           │
│     prettier --write src/auth.js src/utils.js       │
│                                                     │
│  5. --fix で修正されたファイルを再ステージ          │
│     git add src/auth.js src/utils.js                │
│                                                     │
│  6. stashを復元                                     │
│                                                     │
│  7. exit 0 で成功 / 非0 でコミット中断              │
└─────────────────────────────────────────────────────┘
```

### 4.3 lint-stagedの高度な設定パターン

```javascript
// .lintstagedrc.js — 高度な設定パターン

module.exports = {
  // パターン1: 関数形式でコマンドを動的に生成
  '*.{js,jsx,ts,tsx}': (filenames) => {
    // 100ファイル以上の場合はチャンク分割
    const chunks = [];
    const chunkSize = 50;
    for (let i = 0; i < filenames.length; i += chunkSize) {
      chunks.push(filenames.slice(i, i + chunkSize));
    }
    return chunks.flatMap(chunk => [
      `eslint --fix ${chunk.join(' ')}`,
      `prettier --write ${chunk.join(' ')}`,
    ]);
  },

  // パターン2: テストディレクトリのファイルには別の処理
  'src/**/*.{ts,tsx}': [
    'eslint --fix',
    'prettier --write',
  ],
  'tests/**/*.{ts,tsx}': [
    'eslint --fix',
    'prettier --write',
    'jest --bail --findRelatedTests',
  ],

  // パターン3: 特定のファイルを除外
  '!(*test).{js,ts}': [
    'eslint --fix',
  ],

  // パターン4: Pythonファイル
  '*.py': [
    'black',
    'isort',
    'flake8',
  ],

  // パターン5: Rustファイル
  '*.rs': [
    'rustfmt',
  ],

  // パターン6: Goファイル
  '*.go': (filenames) => {
    // Go は個別ファイルではなくパッケージ単位で処理
    const dirs = [...new Set(filenames.map(f => {
      const parts = f.split('/');
      parts.pop();
      return parts.join('/') || '.';
    }))];
    return dirs.map(dir => `cd ${dir} && go vet ./...`);
  },

  // パターン7: 画像の最適化
  '*.{png,jpg,jpeg,gif,svg}': [
    'imagemin-lint-staged',
  ],

  // パターン8: マークダウンのスペルチェック
  '*.md': [
    'markdownlint --fix',
    'cspell --no-must-find-files',
  ],
};
```

### 4.4 lint-stagedのトラブルシューティング

```bash
# lint-stagedのデバッグモード
$ npx lint-staged --debug

# よくあるエラーと解決法

# エラー1: "lint-staged prevented an empty git commit"
# → --fixで全エラーが修正され、ステージ内容と同一になった場合
# → --allow-empty フラグを追加するか、元のコードを修正

# エラー2: "Skipping because of an error from a previous task"
# → 前のコマンドが失敗している
# → eslintのエラーを修正してから再コミット

# エラー3: stash関連のエラー
# → lint-stagedのstash操作が競合
$ git stash list
# 不要なstashを削除
$ git stash drop

# lint-stagedの --no-stash オプション（v13+）
# stashを使わないモード（部分ステージを使わない場合に高速化）
$ npx lint-staged --no-stash
```

---

## 5. commitlint — コミットメッセージの検証

### 5.1 基本設定

```bash
# インストール
$ npm install --save-dev @commitlint/{cli,config-conventional}

# 設定ファイル
$ cat commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [2, 'always', [
      'feat', 'fix', 'docs', 'style', 'refactor',
      'perf', 'test', 'build', 'ci', 'chore', 'revert'
    ]],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
  },
};
```

| ルール                | 説明                                    | 例                    |
|-----------------------|-----------------------------------------|-----------------------|
| `type-enum`           | 許可するtype一覧                        | feat, fix, docs, ...  |
| `type-case`           | typeの大文字/小文字                      | lower-case            |
| `subject-max-length`  | タイトル行の最大文字数                  | 72                    |
| `body-max-line-length`| 本文の1行最大文字数                     | 100                   |
| `header-max-length`   | ヘッダー全体の最大文字数                | 100                   |

### 5.2 カスタムルールの作成

```javascript
// commitlint.config.js — 高度なカスタム設定
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // レベル: 0 = 無効, 1 = 警告, 2 = エラー
    // 適用: 'always' or 'never'

    // type関連
    'type-enum': [2, 'always', [
      'feat', 'fix', 'docs', 'style', 'refactor',
      'perf', 'test', 'build', 'ci', 'chore', 'revert',
      'wip',  // WIPを許可（pushフックで別途チェック）
    ]],
    'type-case': [2, 'always', 'lower-case'],
    'type-empty': [2, 'never'],

    // scope関連
    'scope-case': [2, 'always', 'lower-case'],
    'scope-enum': [1, 'always', [  // 警告のみ（柔軟に）
      'auth', 'api', 'ui', 'db', 'config', 'deps', 'ci',
    ]],

    // subject関連
    'subject-case': [2, 'never', ['start-case', 'pascal-case', 'upper-case']],
    'subject-empty': [2, 'never'],
    'subject-max-length': [2, 'always', 72],
    'subject-full-stop': [2, 'never', '.'],

    // body関連
    'body-leading-blank': [2, 'always'],
    'body-max-line-length': [2, 'always', 100],

    // footer関連
    'footer-leading-blank': [2, 'always'],
    'footer-max-line-length': [2, 'always', 100],

    // header関連
    'header-max-length': [2, 'always', 100],
  },

  // カスタムプラグイン
  plugins: [
    {
      rules: {
        // チケット番号の必須チェック（カスタムルール）
        'ticket-reference': (parsed) => {
          const { footer, body } = parsed;
          const hasTicket = (footer && /[A-Z]+-\d+/.test(footer)) ||
                           (body && /[A-Z]+-\d+/.test(body));
          return [
            hasTicket,
            'コミットメッセージにチケット番号（例: PROJ-123）を含めてください',
          ];
        },
      },
    },
  ],
};
```

### 5.3 commitlintのCI連携

```yaml
# .github/workflows/commitlint.yml
name: Commit Lint

on:
  pull_request:
    branches: [main, develop]

jobs:
  commitlint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 全履歴を取得

      - uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci
        env:
          HUSKY: 0

      - name: Lint commits
        run: npx commitlint --from ${{ github.event.pull_request.base.sha }} --to ${{ github.event.pull_request.head.sha }} --verbose
```

```bash
# ローカルでPRの全コミットを検証
$ npx commitlint --from origin/main --to HEAD --verbose

# 最後のコミットのみ検証
$ npx commitlint --from HEAD~1

# メッセージを直接検証
$ echo "feat: add login feature" | npx commitlint
```

---

## 6. サーバーサイドフックの詳細

### 6.1 pre-receiveフック

```bash
#!/bin/bash
# hooks/pre-receive — サーバーサイドの全体ポリシーチェック

# stdinから受信したref情報を読み取る
while read OLD_SHA NEW_SHA REFNAME; do
  echo "Checking: $REFNAME ($OLD_SHA -> $NEW_SHA)"

  # 削除の場合（new_sha = 0000...）
  ZERO="0000000000000000000000000000000000000000"
  if [ "$NEW_SHA" = "$ZERO" ]; then
    # 保護ブランチの削除を禁止
    if echo "$REFNAME" | grep -qE "refs/heads/(main|master|develop)"; then
      echo "ERROR: 保護ブランチ '$REFNAME' の削除は禁止されています。"
      exit 1
    fi
    continue
  fi

  # 新規ブランチの場合（old_sha = 0000...）
  if [ "$OLD_SHA" = "$ZERO" ]; then
    COMMITS=$(git rev-list "$NEW_SHA" --not --branches)
  else
    COMMITS=$(git rev-list "$OLD_SHA..$NEW_SHA")
  fi

  # 各コミットをチェック
  for COMMIT in $COMMITS; do
    # コミットメッセージの検証
    MSG=$(git log --format=%B -n 1 "$COMMIT")
    if ! echo "$MSG" | head -1 | grep -qE "^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"; then
      echo "ERROR: コミット $COMMIT のメッセージがConventional Commits形式ではありません。"
      echo "  メッセージ: $(echo "$MSG" | head -1)"
      exit 1
    fi

    # 大きなファイルのチェック
    MAX_SIZE=$((10 * 1024 * 1024))  # 10MB
    git diff-tree --no-commit-id -r "$COMMIT" | while read MODE_A MODE_B SHA_A SHA_B STATUS FILENAME; do
      if [ "$STATUS" = "D" ]; then
        continue  # 削除されたファイルはスキップ
      fi
      FILE_SIZE=$(git cat-file -s "$SHA_B" 2>/dev/null || echo 0)
      if [ "$FILE_SIZE" -gt "$MAX_SIZE" ]; then
        echo "ERROR: ファイル '$FILENAME' が${MAX_SIZE}バイトを超えています（${FILE_SIZE}バイト）。"
        echo "Git LFSの使用を検討してください。"
        exit 1
      fi
    done
    if [ $? -ne 0 ]; then
      exit 1
    fi

    # 秘密情報の検出
    git diff-tree --no-commit-id -r -p "$COMMIT" | grep -qE "(PRIVATE KEY|password\s*=\s*['\"]|AWS_SECRET|api_key\s*=)" && {
      echo "ERROR: コミット $COMMIT に秘密情報が含まれている可能性があります。"
      exit 1
    }
  done
done

echo "All checks passed."
exit 0
```

### 6.2 updateフック

```bash
#!/bin/bash
# hooks/update — ブランチごとのポリシーチェック

REFNAME=$1
OLD_SHA=$2
NEW_SHA=$3
ZERO="0000000000000000000000000000000000000000"

# ブランチ名を取得
BRANCH=$(echo "$REFNAME" | sed 's|refs/heads/||')

# pushしたユーザーを取得（環境依存）
USER=${GL_USERNAME:-${REMOTE_USER:-$(whoami)}}

echo "=== Update hook: $BRANCH by $USER ==="

# mainブランチの保護ポリシー
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  # force pushの検出
  if [ "$OLD_SHA" != "$ZERO" ] && [ "$NEW_SHA" != "$ZERO" ]; then
    MERGE_BASE=$(git merge-base "$OLD_SHA" "$NEW_SHA" 2>/dev/null || echo "")
    if [ "$MERGE_BASE" != "$OLD_SHA" ]; then
      echo "ERROR: $BRANCH へのforce pushは禁止されています。"
      exit 1
    fi
  fi

  # 管理者以外のpushを禁止（ACL）
  ADMINS="alice bob charlie"
  IS_ADMIN=false
  for admin in $ADMINS; do
    if [ "$USER" = "$admin" ]; then
      IS_ADMIN=true
      break
    fi
  done

  if [ "$IS_ADMIN" = "false" ]; then
    echo "ERROR: $BRANCH への直接pushは管理者のみ許可されています。"
    echo "PRを作成してください。"
    exit 1
  fi
fi

# リリースブランチの保護
if echo "$BRANCH" | grep -qE "^release/"; then
  # hotfixのみ許可（fixコミットのみ）
  COMMITS=$(git rev-list "$OLD_SHA..$NEW_SHA" 2>/dev/null)
  for COMMIT in $COMMITS; do
    MSG=$(git log --format=%s -n 1 "$COMMIT")
    if ! echo "$MSG" | grep -qE "^(fix|hotfix|revert)"; then
      echo "ERROR: リリースブランチにはfix/hotfix/revertのみ許可されています。"
      echo "  コミット: $COMMIT"
      echo "  メッセージ: $MSG"
      exit 1
    fi
  done
fi

# タグの保護
if echo "$REFNAME" | grep -q "refs/tags/"; then
  TAG=$(echo "$REFNAME" | sed 's|refs/tags/||')
  # 既存タグの変更を禁止
  if [ "$OLD_SHA" != "$ZERO" ]; then
    echo "ERROR: 既存タグ '$TAG' の変更は禁止されています。"
    exit 1
  fi
  # セマンティックバージョニングの検証
  if ! echo "$TAG" | grep -qE "^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$"; then
    echo "ERROR: タグ '$TAG' はセマンティックバージョニング形式ではありません。"
    echo "形式: v1.2.3 or v1.2.3-beta.1"
    exit 1
  fi
fi

exit 0
```

### 6.3 post-receiveフック

```bash
#!/bin/bash
# hooks/post-receive — push受信後の自動処理

while read OLD_SHA NEW_SHA REFNAME; do
  BRANCH=$(echo "$REFNAME" | sed 's|refs/heads/||')
  ZERO="0000000000000000000000000000000000000000"

  # ブランチ削除の場合はスキップ
  if [ "$NEW_SHA" = "$ZERO" ]; then
    continue
  fi

  # コミット情報を取得
  AUTHOR=$(git log --format='%an' -n 1 "$NEW_SHA")
  COMMIT_MSG=$(git log --format='%s' -n 1 "$NEW_SHA")

  if [ "$OLD_SHA" = "$ZERO" ]; then
    COMMIT_COUNT="新規ブランチ"
  else
    COMMIT_COUNT=$(git rev-list --count "$OLD_SHA..$NEW_SHA")
  fi

  # Slack通知（webhook）
  SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
  PAYLOAD=$(cat <<EOF
{
  "channel": "#git-notifications",
  "username": "Git Bot",
  "text": ":git: *${AUTHOR}* が *${BRANCH}* に ${COMMIT_COUNT} コミットをpushしました\n> ${COMMIT_MSG}",
  "icon_emoji": ":octocat:"
}
EOF
  )
  curl -s -X POST -H 'Content-type: application/json' \
    --data "$PAYLOAD" "$SLACK_WEBHOOK" > /dev/null 2>&1 &

  # mainブランチへのpush → デプロイ
  if [ "$BRANCH" = "main" ]; then
    echo "Deploying to production..."
    /opt/deploy/production.sh "$NEW_SHA" &
  fi

  # developブランチへのpush → ステージングデプロイ
  if [ "$BRANCH" = "develop" ]; then
    echo "Deploying to staging..."
    /opt/deploy/staging.sh "$NEW_SHA" &
  fi

  # タグのpush → リリース処理
  if echo "$REFNAME" | grep -q "refs/tags/"; then
    TAG=$(echo "$REFNAME" | sed 's|refs/tags/||')
    echo "Creating release for tag: $TAG"
    /opt/release/create-release.sh "$TAG" "$NEW_SHA" &
  fi
done

exit 0
```

---

## 7. 完全な設定例

### 7.1 package.json

```json
{
  "name": "my-project",
  "scripts": {
    "prepare": "husky",
    "lint": "eslint src/",
    "format": "prettier --write src/",
    "test": "jest",
    "test:staged": "jest --bail --findRelatedTests"
  },
  "devDependencies": {
    "husky": "^9.0.0",
    "lint-staged": "^15.0.0",
    "@commitlint/cli": "^19.0.0",
    "@commitlint/config-conventional": "^19.0.0",
    "eslint": "^9.0.0",
    "prettier": "^3.0.0"
  },
  "lint-staged": {
    "*.{js,jsx,ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ]
  }
}
```

### 7.2 .husky/ ディレクトリ

```bash
# .husky/pre-commit
npx lint-staged

# .husky/commit-msg
npx --no -- commitlint --edit $1

# .husky/pre-push
npm test
```

```
┌────────────────────────────────────────────────────┐
│  完全な自動化フロー                                 │
│                                                    │
│  git commit -m "feat: add login"                   │
│       │                                            │
│       ▼                                            │
│  .husky/pre-commit                                 │
│       │                                            │
│       ▼                                            │
│  lint-staged                                       │
│    ├── eslint --fix (対象: *.js, *.ts)            │
│    ├── prettier --write (対象: 全ファイル)         │
│    └── 修正ファイルを再ステージ                    │
│       │                                            │
│       ▼                                            │
│  .husky/commit-msg                                 │
│    └── commitlint ("feat: add login" を検証)       │
│       │                                            │
│       ▼                                            │
│  コミット作成                                      │
│                                                    │
│  git push                                          │
│       │                                            │
│       ▼                                            │
│  .husky/pre-push                                   │
│    └── npm test (全テスト実行)                     │
│       │                                            │
│       ▼                                            │
│  リモートに送信                                    │
└────────────────────────────────────────────────────┘
```

### 7.3 TypeScriptプロジェクトの完全設定

```json
{
  "name": "typescript-project",
  "scripts": {
    "prepare": "husky",
    "lint": "eslint 'src/**/*.{ts,tsx}'",
    "format": "prettier --write 'src/**/*.{ts,tsx,json,css}'",
    "typecheck": "tsc --noEmit",
    "test": "vitest run",
    "test:watch": "vitest",
    "build": "vite build"
  },
  "devDependencies": {
    "husky": "^9.1.0",
    "lint-staged": "^15.2.0",
    "@commitlint/cli": "^19.3.0",
    "@commitlint/config-conventional": "^19.2.0",
    "eslint": "^9.5.0",
    "@typescript-eslint/eslint-plugin": "^7.14.0",
    "@typescript-eslint/parser": "^7.14.0",
    "prettier": "^3.3.0",
    "typescript": "^5.5.0",
    "vitest": "^1.6.0"
  },
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml,css}": [
      "prettier --write"
    ]
  }
}
```

```bash
# .husky/pre-commit — TypeScript向け
#!/bin/sh

# lint-stagedを実行
npx lint-staged

# 型チェック（プロジェクト全体）
# ※ ステージファイルだけの型チェックは不可能なため全体実行
echo "Running type check..."
npx tsc --noEmit
```

### 7.4 Pythonプロジェクトの設定（pre-commit framework使用）

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: ['--profile', 'black']

  - repo: https://github.com/pycqa/flake8
    rev: 7.1.0
    hooks:
      - id: flake8
        args: ['--max-line-length', '88']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        additional_dependencies:
          - types-requests
          - types-PyYAML

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.9
    hooks:
      - id: bandit
        args: ['-c', 'pyproject.toml']

  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: python -m pytest tests/ -x --tb=short
        language: system
        types: [python]
        pass_filenames: false
        always_run: true
        stages: [push]
```

```bash
# pre-commit frameworkのインストールと使用
$ pip install pre-commit

# フックをインストール
$ pre-commit install
$ pre-commit install --hook-type commit-msg
$ pre-commit install --hook-type pre-push

# 全ファイルに対して手動実行
$ pre-commit run --all-files

# 特定のフックのみ実行
$ pre-commit run black --all-files

# フックの更新
$ pre-commit autoupdate

# キャッシュのクリア
$ pre-commit clean
```

---

## 8. 代替ツール

### 8.1 lefthook

```yaml
# lefthook.yml — Goで書かれた高速なフックマネージャ

pre-commit:
  parallel: true  # コマンドを並列実行
  commands:
    eslint:
      glob: "*.{js,jsx,ts,tsx}"
      run: npx eslint --fix {staged_files}
      stage_fixed: true  # 修正されたファイルを再ステージ

    prettier:
      glob: "*.{js,jsx,ts,tsx,json,md,css,yml}"
      run: npx prettier --write {staged_files}
      stage_fixed: true

    stylelint:
      glob: "*.{css,scss}"
      run: npx stylelint --fix {staged_files}
      stage_fixed: true

    typecheck:
      glob: "*.{ts,tsx}"
      run: npx tsc --noEmit

commit-msg:
  commands:
    commitlint:
      run: npx commitlint --edit {1}

pre-push:
  parallel: true
  commands:
    test:
      run: npm test

    branch-check:
      run: |
        BRANCH=$(git symbolic-ref --short HEAD)
        if [ "$BRANCH" = "main" ]; then
          echo "ERROR: mainへの直接pushは禁止"
          exit 1
        fi

post-checkout:
  commands:
    deps:
      run: |
        if git diff --name-only {1} {2} | grep -q "package-lock.json"; then
          npm install
        fi
```

```bash
# lefthookのインストール
$ npm install --save-dev lefthook
# または
$ brew install lefthook

# フックのインストール
$ npx lefthook install

# 特定のフックを手動実行
$ npx lefthook run pre-commit

# デバッグモード
$ LEFTHOOK_VERBOSE=1 npx lefthook run pre-commit
```

```
┌──────────────────────────────────────────────────────┐
│  husky vs lefthook 比較                               │
│                                                      │
│  husky                                               │
│  + シンプルな設定（シェルスクリプト）                 │
│  + Node.jsエコシステムとの親和性                     │
│  + 広いコミュニティ                                  │
│  - 並列実行は別途設定が必要                          │
│  - lint-stagedが必要（ステージファイル処理）          │
│                                                      │
│  lefthook                                            │
│  + Goバイナリで高速起動                              │
│  + 並列実行がビルトイン                              │
│  + ステージファイル処理がビルトイン                   │
│  + stage_fixedがビルトイン（再ステージ）             │
│  + Node.js以外のプロジェクトでも使いやすい           │
│  - コミュニティがhuskyより小さい                     │
│                                                      │
│  結論:                                               │
│  Node.jsプロジェクト → husky + lint-staged           │
│  マルチ言語/モノレポ → lefthook                      │
│  Pythonプロジェクト → pre-commit framework           │
└──────────────────────────────────────────────────────┘
```

### 8.2 ツール選択ガイド

```
┌─────────────────────────────────────────────────────┐
│  フック管理ツール選択フローチャート                   │
│                                                     │
│  プロジェクトの主要言語は？                          │
│       │                                             │
│       ├── Python → pre-commit framework             │
│       │   - .pre-commit-config.yaml                 │
│       │   - pip install pre-commit                  │
│       │                                             │
│       ├── JavaScript/TypeScript → husky             │
│       │   - npm install husky lint-staged            │
│       │   - シンプルで実績豊富                       │
│       │                                             │
│       ├── Go/Rust/マルチ言語 → lefthook             │
│       │   - Go製で高速                              │
│       │   - 言語非依存                              │
│       │                                             │
│       └── モノレポ（複数言語混在）                   │
│            ├── Turborepo → husky + turbo lint       │
│            └── その他 → lefthook                    │
│                - 並列実行が強い                      │
│                - glob: で対象を絞れる               │
│                                                     │
│  追加考慮事項:                                      │
│  - CI/CDとの統合 → GitHub Actions + lint設定        │
│  - セキュリティ重視 → pre-receive (サーバーサイド)  │
│  - 大規模チーム → core.hooksPath + 共有設定        │
└─────────────────────────────────────────────────────┘
```

---

## 9. モノレポ対応

### 9.1 モノレポでのlint-staged設定

```javascript
// .lintstagedrc.js — モノレポ（Turborepo）
const path = require('path');

module.exports = {
  '*.{js,jsx,ts,tsx}': (filenames) => {
    // ファイルをパッケージごとに分類
    const packages = {};
    filenames.forEach(filename => {
      const relative = path.relative(process.cwd(), filename);
      const parts = relative.split(path.sep);
      let pkg = 'root';
      if (parts[0] === 'packages' && parts.length > 2) {
        pkg = parts[1];
      } else if (parts[0] === 'apps' && parts.length > 2) {
        pkg = `app-${parts[1]}`;
      }
      if (!packages[pkg]) packages[pkg] = [];
      packages[pkg].push(filename);
    });

    const commands = [];
    Object.entries(packages).forEach(([pkg, files]) => {
      const fileList = files.join(' ');
      commands.push(`eslint --fix ${fileList}`);
      commands.push(`prettier --write ${fileList}`);
    });
    return commands;
  },
};
```

### 9.2 lefthookのモノレポ設定

```yaml
# lefthook.yml — モノレポ設定
pre-commit:
  parallel: true
  commands:
    # パッケージごとのlint
    lint-web:
      root: "apps/web/"
      glob: "*.{ts,tsx}"
      run: npx eslint --fix {staged_files}
      stage_fixed: true

    lint-api:
      root: "apps/api/"
      glob: "*.ts"
      run: npx eslint --fix {staged_files}
      stage_fixed: true

    lint-shared:
      root: "packages/shared/"
      glob: "*.ts"
      run: npx eslint --fix {staged_files}
      stage_fixed: true

    # 全体のフォーマット
    format:
      glob: "*.{ts,tsx,js,jsx,json,md,css,yml}"
      run: npx prettier --write {staged_files}
      stage_fixed: true

    # 型チェック（変更されたパッケージのみ）
    typecheck-web:
      root: "apps/web/"
      glob: "*.{ts,tsx}"
      run: npx tsc --noEmit

    typecheck-api:
      root: "apps/api/"
      glob: "*.ts"
      run: npx tsc --noEmit

pre-push:
  commands:
    # 影響を受けるパッケージのテストのみ実行
    test:
      run: npx turbo run test --filter='...[HEAD~1]'
```

### 9.3 モノレポのパフォーマンス最適化

```bash
# .husky/pre-commit — Turborepoを活用した高速化
#!/bin/sh

# 変更されたファイルからパッケージを特定
CHANGED_PACKAGES=$(npx turbo run lint --filter='...[HEAD]' --dry-run=json 2>/dev/null \
  | jq -r '.packages[]' 2>/dev/null || echo "")

if [ -z "$CHANGED_PACKAGES" ]; then
  echo "No packages affected. Skipping lint."
  exit 0
fi

echo "Affected packages: $CHANGED_PACKAGES"

# lint-stagedで変更ファイルのみ処理
npx lint-staged

# 型チェックは影響を受けるパッケージのみ
npx turbo run typecheck --filter='...[HEAD]' --cache-dir=.turbo
```

---

## 10. GitHub Actions との連携

### 10.1 フックとCIの役割分担

```
┌──────────────────────────────────────────────────────┐
│  フックとCI/CDの責任分担                              │
│                                                      │
│  クライアントサイドフック（即時フィードバック）        │
│  ┌──────────────────────────────────────────┐        │
│  │ pre-commit:                              │        │
│  │   - lint (ESLint, Stylelint)             │        │
│  │   - format (Prettier)                    │        │
│  │   - ← 数秒以内で完了すべき              │        │
│  │                                          │        │
│  │ commit-msg:                              │        │
│  │   - Conventional Commits検証             │        │
│  │   - ← 即座に完了                        │        │
│  │                                          │        │
│  │ pre-push:                                │        │
│  │   - ユニットテスト（高速なもの）         │        │
│  │   - ブランチ名チェック                   │        │
│  │   - WIPコミットチェック                  │        │
│  │   - ← 30秒以内で完了すべき              │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  CI/CD（包括的な品質チェック）                        │
│  ┌──────────────────────────────────────────┐        │
│  │ PR作成時:                                │        │
│  │   - 全テストスイート                     │        │
│  │   - E2Eテスト                            │        │
│  │   - セキュリティスキャン                 │        │
│  │   - コードカバレッジ                     │        │
│  │   - ビルド検証                           │        │
│  │   - パフォーマンスベンチマーク            │        │
│  │   - ← 数分かかっても許容される           │        │
│  │                                          │        │
│  │ マージ時:                                │        │
│  │   - デプロイ                             │        │
│  │   - リリースノート生成                   │        │
│  │   - Docker イメージビルド                │        │
│  └──────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

### 10.2 GitHub Actionsでのlintワークフロー

```yaml
# .github/workflows/lint.yml
name: Lint & Test

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci
        env:
          HUSKY: 0  # CI環境ではhuskyを無効化

      - name: ESLint
        run: npx eslint 'src/**/*.{ts,tsx}' --format=compact

      - name: Prettier check
        run: npx prettier --check 'src/**/*.{ts,tsx,json,css}'

      - name: TypeScript
        run: npx tsc --noEmit

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci
        env:
          HUSKY: 0

      - name: Test
        run: npx vitest run --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage/lcov.info

  commitlint:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci
        env:
          HUSKY: 0

      - name: Validate PR commits
        run: npx commitlint --from ${{ github.event.pull_request.base.sha }} --to ${{ github.event.pull_request.head.sha }} --verbose
```

---

## 11. セキュリティとフック

### 11.1 秘密情報の検出フック

```bash
#!/bin/sh
# .git/hooks/pre-commit — 秘密情報の検出

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
  exit 0
fi

# パターンマッチで秘密情報を検出
PATTERNS=(
  'PRIVATE KEY'
  'password\s*[:=]\s*["\x27][^"\x27]+'
  'AWS_SECRET_ACCESS_KEY'
  'AKIA[0-9A-Z]{16}'  # AWS Access Key ID
  'api[_-]?key\s*[:=]\s*["\x27][^"\x27]+'
  'secret\s*[:=]\s*["\x27][^"\x27]+'
  'token\s*[:=]\s*["\x27][^"\x27]+'
  'ghp_[a-zA-Z0-9]{36}'  # GitHub Personal Access Token
  'sk-[a-zA-Z0-9]{32,}'  # OpenAI API Key
)

FOUND_SECRETS=false
for file in $STAGED_FILES; do
  # バイナリファイルはスキップ
  if git diff --cached --diff-filter=ACM "$file" | grep -q "Binary files"; then
    continue
  fi

  DIFF=$(git diff --cached -p "$file")
  for pattern in "${PATTERNS[@]}"; do
    MATCHES=$(echo "$DIFF" | grep -n "^+" | grep -iE "$pattern" || true)
    if [ -n "$MATCHES" ]; then
      echo "WARNING: 秘密情報の可能性: $file"
      echo "$MATCHES"
      echo ""
      FOUND_SECRETS=true
    fi
  done
done

if [ "$FOUND_SECRETS" = true ]; then
  echo "============================================"
  echo "  秘密情報が検出されました！"
  echo "  .env に移動するか、git-secret/SOPS を使用してください。"
  echo "  誤検出の場合: git commit --no-verify"
  echo "============================================"
  exit 1
fi

exit 0
```

### 11.2 gitleaksとの統合

```bash
# gitleaksのインストール
$ brew install gitleaks

# pre-commitフックでgitleaksを使用
# .husky/pre-commit
npx lint-staged
gitleaks protect --staged --verbose
```

```yaml
# .pre-commit-config.yaml（pre-commit framework）
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.4
    hooks:
      - id: gitleaks
```

```toml
# .gitleaks.toml — gitleaksのカスタム設定
title = "Custom Gitleaks Config"

[allowlist]
  description = "許可リスト"
  paths = [
    '''\.env\.example$''',
    '''test/fixtures/''',
    '''\.md$''',
  ]

[[rules]]
  description = "Custom API Key"
  id = "custom-api-key"
  regex = '''my-api-key-[a-zA-Z0-9]{32}'''
  tags = ["api", "custom"]

[[rules]]
  description = "Slack Webhook"
  id = "slack-webhook"
  regex = '''https://hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[a-zA-Z0-9]+'''
  tags = ["slack"]
```

### 11.3 フックのセキュリティリスク

```
┌──────────────────────────────────────────────────────┐
│  Git Hooksのセキュリティ考慮事項                      │
│                                                      │
│  リスク1: 悪意のあるフックの実行                      │
│  ┌──────────────────────────────────────────┐        │
│  │ .git/hooks/ のフックはgit cloneで           │        │
│  │ コピーされない（安全設計）                │        │
│  │                                          │        │
│  │ しかし:                                  │        │
│  │ - core.hooksPath で外部ディレクトリを    │        │
│  │   指定されると任意のスクリプトが実行される│        │
│  │ - huskyの"prepare"スクリプトは            │        │
│  │   npm installで自動実行される            │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  リスク2: フックのバイパス                            │
│  ┌──────────────────────────────────────────┐        │
│  │ git commit --no-verify                   │        │
│  │ git push --no-verify                     │        │
│  │ → サーバーサイドフックでは防げるが、      │        │
│  │   クライアントサイドフックはバイパス可能  │        │
│  └──────────────────────────────────────────┘        │
│                                                      │
│  対策:                                               │
│  - 重要なチェックはCI/CDで実施（バイパス不可）       │
│  - Branch protection rulesを設定                     │
│  - CODEOWNERS でレビューを必須化                     │
│  - サーバーサイドフックで最終防衛                     │
│  - npm installの--ignore-scriptsを適切に使用         │
└──────────────────────────────────────────────────────┘
```

---

## 12. パフォーマンス最適化

### 12.1 フックの実行時間計測

```bash
#!/bin/sh
# .git/hooks/pre-commit — パフォーマンス計測付き

START=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time() * 1e9))')

# lint-stagedを実行
npx lint-staged
RESULT=$?

END=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time() * 1e9))')
ELAPSED=$(( (END - START) / 1000000 ))  # ミリ秒に変換

echo "pre-commit hook completed in ${ELAPSED}ms"

# 5秒以上かかった場合に警告
if [ "$ELAPSED" -gt 5000 ]; then
  echo "WARNING: pre-commit hookの実行に${ELAPSED}msかかりました。"
  echo "パフォーマンス最適化を検討してください。"
fi

exit $RESULT
```

### 12.2 高速化テクニック

```bash
# テクニック1: lint-stagedで対象ファイルを限定
# package.json
{
  "lint-staged": {
    "*.{js,ts}": [
      "eslint --fix --cache",  # --cacheでキャッシュを活用
      "prettier --write --cache"  # --cacheでキャッシュを活用
    ]
  }
}

# テクニック2: eslintのキャッシュを活用
# .eslintrc.json の追加設定はなし、--cacheフラグで十分

# テクニック3: テストは関連ファイルのみ
{
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix --cache",
      "vitest related --run"  # 変更ファイルに関連するテストのみ
    ]
  }
}

# テクニック4: 型チェックはpre-pushに移動
# pre-commitでは型チェックを行わず、pre-pushで実行
# → コミットの高速化
```

```
┌──────────────────────────────────────────────────────┐
│  フックのパフォーマンス目標                           │
│                                                      │
│  pre-commit:                                         │
│    目標: 3秒以内                                     │
│    ┌──────────────────────────────────────┐          │
│    │ lint-staged (ESLint + Prettier)      │ ~2s     │
│    │ ※ --cache フラグで高速化            │          │
│    │ ※ 対象はステージファイルのみ        │          │
│    └──────────────────────────────────────┘          │
│                                                      │
│  commit-msg:                                         │
│    目標: 1秒以内                                     │
│    ┌──────────────────────────────────────┐          │
│    │ commitlint                           │ <1s     │
│    │ ※ 正規表現チェックのみ              │          │
│    └──────────────────────────────────────┘          │
│                                                      │
│  pre-push:                                           │
│    目標: 30秒以内                                    │
│    ┌──────────────────────────────────────┐          │
│    │ 型チェック (tsc --noEmit)            │ ~5-10s  │
│    │ ユニットテスト (関連テストのみ)      │ ~10-20s │
│    │ ブランチチェック                     │ <1s     │
│    └──────────────────────────────────────┘          │
│                                                      │
│  CI/CD:                                              │
│    制限なし（数分許容）                               │
│    ┌──────────────────────────────────────┐          │
│    │ 全テスト、E2E、セキュリティスキャン   │ ~5-10m  │
│    └──────────────────────────────────────┘          │
└──────────────────────────────────────────────────────┘
```

### 12.3 並列実行による高速化

```yaml
# lefthook.yml — 並列実行設定
pre-commit:
  parallel: true  # 全コマンドを並列実行
  commands:
    eslint:
      glob: "*.{js,jsx,ts,tsx}"
      run: npx eslint --fix --cache {staged_files}
      stage_fixed: true

    prettier:
      glob: "*.{js,jsx,ts,tsx,json,css,md,yml}"
      run: npx prettier --write --cache {staged_files}
      stage_fixed: true

    stylelint:
      glob: "*.{css,scss}"
      run: npx stylelint --fix --cache {staged_files}
      stage_fixed: true

  # scripts セクション（シェルスクリプトで複雑な処理）
  scripts:
    "check-secrets.sh":
      runner: bash
```

```javascript
// .lintstagedrc.js — 並列実行（lint-staged v15+）
module.exports = {
  // lint-staged v15 では同一glob内のコマンドは順次実行
  // 異なるglobのコマンドは並列実行
  '*.{ts,tsx}': [
    'eslint --fix --cache',
    'prettier --write --cache',
  ],
  // ↓ 上記と並列実行される
  '*.css': [
    'stylelint --fix --cache',
  ],
  '*.md': [
    'markdownlint --fix',
  ],
};
```

---

## 13. トラブルシューティング

### 13.1 よくある問題と解決法

```bash
# 問題1: huskyのフックが実行されない
# 原因: core.hooksPathが設定されていない
$ git config --get core.hooksPath
# 空の場合、huskyを再インストール
$ npx husky install
# または
$ rm -rf node_modules && npm install

# 問題2: 実行権限がない
$ ls -la .husky/pre-commit
# -rw-r--r-- の場合
$ chmod +x .husky/pre-commit

# 問題3: Node.jsのパスが見つからない
# GUIのGitクライアント（SourceTree等）で発生しやすい
# → フックの先頭にパスを追加
#!/bin/sh
export PATH="/usr/local/bin:/opt/homebrew/bin:$PATH"
npx lint-staged

# 問題4: Windows + WSLでの問題
# WSL内のnpxがWindowsのGitから呼ばれない
# → Windowsネイティブのnpmを使用するか、
#    Git for Windowsを使用

# 問題5: pnpm/yarnでhuskyが動作しない
# pnpmの場合
$ pnpm exec husky init
# .npmrc に以下を追加
# enable-pre-post-scripts=true

# yarnの場合（yarn 2+/berry）
$ yarn dlx husky init
# package.json:
# "packageManager": "yarn@4.0.0"

# 問題6: Gitのバージョンが古い
# core.hooksPathは Git 2.9+ で利用可能
$ git --version
# 2.9未満の場合はアップグレードが必要
```

### 13.2 デバッグ手法

```bash
# フックのデバッグ — 手動実行
$ sh -x .husky/pre-commit
# → -x でコマンドのトレースを表示

# lint-stagedのデバッグ
$ npx lint-staged --debug 2>&1 | tee lint-staged-debug.log

# huskyのデバッグ
$ HUSKY_DEBUG=1 git commit -m "test"

# Git自体のフックデバッグ
$ GIT_TRACE=1 git commit -m "test"
# → Gitの内部処理（フック呼び出し含む）をトレース

# フックの出力をログに保存
# .husky/pre-commit
#!/bin/sh
exec > /tmp/pre-commit.log 2>&1
set -x
npx lint-staged

# 特定のフックだけ一時的に無効化
$ chmod -x .husky/pre-commit
# 再有効化
$ chmod +x .husky/pre-commit
```

### 13.3 エラーメッセージの改善

```bash
#!/bin/sh
# .husky/pre-commit — ユーザーフレンドリーなエラー表示

# カラー定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "${GREEN}Running pre-commit checks...${NC}"

# lint-stagedを実行
if ! npx lint-staged 2>&1; then
  echo ""
  echo "${RED}========================================${NC}"
  echo "${RED}  pre-commit check FAILED${NC}"
  echo "${RED}========================================${NC}"
  echo ""
  echo "${YELLOW}対処方法:${NC}"
  echo "  1. エラーメッセージを確認して修正"
  echo "  2. 修正後: git add <file> && git commit"
  echo ""
  echo "${YELLOW}緊急時（推奨しません）:${NC}"
  echo "  git commit --no-verify"
  echo ""
  exit 1
fi

echo "${GREEN}All checks passed!${NC}"
exit 0
```

---

## 14. カスタムmerge/diffドライバー

### 14.1 カスタムmergeドライバー

```bash
# .gitattributes — カスタムドライバーの適用
*.lock merge=ours
package-lock.json merge=npm-merge-driver
*.pbxproj merge=union
CHANGELOG.md merge=union
```

```bash
# カスタムmergeドライバーの登録
$ git config merge.npm-merge-driver.driver "npx npm-merge-driver merge %A %O %B %P"
$ git config merge.npm-merge-driver.name "npm merge driver for package-lock.json"

# union mergeドライバー（両方の変更を保持）
$ git config merge.union.driver "git merge-file --union %A %O %B"
$ git config merge.union.name "union merge driver"
```

### 14.2 カスタムdiffドライバー

```bash
# .gitattributes — カスタムdiffドライバー
*.png diff=exif
*.jpg diff=exif
*.pdf diff=pdf
*.xlsx diff=xlsx

# 画像のdiffドライバー
$ git config diff.exif.textconv exiftool
# → git diff で画像のEXIF情報を比較

# PDFのdiffドライバー
$ git config diff.pdf.textconv "pdftotext -layout"
# → git diff でPDFのテキスト内容を比較

# Excelのdiffドライバー
$ git config diff.xlsx.textconv "python3 -c 'import openpyxl,sys; wb=openpyxl.load_workbook(sys.argv[1]); [print(f\"{ws.title}: {[[c.value for c in r] for r in ws.iter_rows()]}\") for ws in wb]'"
```

### 14.3 clean/smudgeフィルター

```bash
# .gitattributes
*.config filter=config-vars
secrets.yml filter=vault

# clean/smudgeフィルターの設定
# clean: ワークツリー → リポジトリ（commit時）
# smudge: リポジトリ → ワークツリー（checkout時）

# 環境変数をプレースホルダーに置換
$ git config filter.config-vars.clean 'sed "s|${DATABASE_URL}|__DATABASE_URL__|g"'
$ git config filter.config-vars.smudge 'sed "s|__DATABASE_URL__|${DATABASE_URL}|g"'

# git-cryptによる暗号化
$ git config filter.vault.clean 'git-crypt clean'
$ git config filter.vault.smudge 'git-crypt smudge'
$ git config diff.vault.textconv 'git-crypt diff'
```

```
┌──────────────────────────────────────────────────────┐
│  clean/smudge フィルターの動作                        │
│                                                      │
│  git add (clean)                                     │
│  ┌────────────┐     ┌────────────┐     ┌──────────┐ │
│  │ ワークツリー│ --> │  clean     │ --> │ インデックス│ │
│  │ 平文       │     │ フィルター │     │ 暗号化    │ │
│  └────────────┘     └────────────┘     └──────────┘ │
│                                                      │
│  git checkout (smudge)                               │
│  ┌──────────┐     ┌────────────┐     ┌────────────┐ │
│  │ リポジトリ│ --> │  smudge    │ --> │ ワークツリー│ │
│  │ 暗号化   │     │ フィルター │     │ 平文       │ │
│  └──────────┘     └────────────┘     └────────────┘ │
│                                                      │
│  使用例:                                             │
│  - 秘密情報の暗号化/復号                             │
│  - 環境変数のプレースホルダー置換                    │
│  - ファイルサイズの圧縮/展開                         │
│  - コードの自動フォーマット                          │
└──────────────────────────────────────────────────────┘
```

---

## 15. アンチパターン

### アンチパターン1: pre-commitフックで全ファイルにlintを実行

```bash
# NG: 全ファイルを対象にlint
#!/bin/sh
npx eslint src/
# → ステージしていないファイルのエラーでもコミットがブロックされる
# → 大規模プロジェクトでは実行時間が長すぎる

# OK: ステージされたファイルのみを対象にする（lint-staged）
#!/bin/sh
npx lint-staged
# → 変更されたファイルのみを高速に処理
```

**理由**: 全ファイルを対象にすると、自分が変更していないファイルのエラーでコミットがブロックされる。lint-stagedはステージされたファイルのみを処理するため、開発体験を損なわない。

### アンチパターン2: フックのバイパスを常態化させる

```bash
# NG: --no-verify を日常的に使用
$ git commit --no-verify -m "wip: とりあえずコミット"
$ git push --no-verify
# → lintエラーやテスト失敗がリモートに混入

# OK: フックが煩わしいなら、フック自体を改善する
# - 実行時間を短縮（lint-stagedで対象を限定）
# - テストを必要最小限に（関連テストのみ実行）
# - 誤検出を排除（ルールの精査）
```

**理由**: `--no-verify`の使用はフックの存在意義を否定する。フックが頻繁にバイパスされる場合、フックの設計に問題がある。

### アンチパターン3: 重いテストをpre-commitに配置

```bash
# NG: pre-commitで全テストを実行
#!/bin/sh
npm test  # 全テストスイート → 数分かかる
npx tsc --noEmit  # 型チェック → プロジェクト全体で数十秒

# OK: 段階的にチェックを配置
# pre-commit: lint + format（数秒）
# pre-push: 関連テスト + 型チェック（30秒以内）
# CI: 全テスト + E2E + セキュリティ（制限なし）
```

**理由**: コミットのたびに数分待たされると、開発者はフックをバイパスするようになる。フックは素早いフィードバックを提供すべきであり、重い処理はCI/CDに委ねる。

### アンチパターン4: フックの中でインタラクティブな入力を求める

```bash
# NG: フック内で確認プロンプト
#!/bin/sh
read -p "本当にコミットしますか？ (y/n): " answer
if [ "$answer" != "y" ]; then
  exit 1
fi
# → GUIクライアントやCI環境で動作しない
# → パイプライン処理でハングする

# OK: チェックは自動的に、結果を表示するだけ
#!/bin/sh
ISSUES=$(npx eslint --format compact src/)
if [ -n "$ISSUES" ]; then
  echo "$ISSUES"
  exit 1
fi
```

**理由**: フックはnon-interactiveな環境で実行されることが多い。対話的な入力に依存すると、GUIクライアントやCI環境で問題が発生する。

### アンチパターン5: サーバーサイドフックの未設定

```bash
# NG: クライアントサイドフックのみに依存
# → --no-verify でバイパス可能
# → フックをインストールしていないメンバーからの不正なpush

# OK: サーバーサイドフック（またはCI）で最終防衛
# GitHub: Branch protection rules
#   - Require status checks to pass before merging
#   - Require pull request reviews
#   - Require linear history
# GitLab: Push rules
#   - Commit message pattern
#   - Branch name pattern
#   - File size limit
```

**理由**: クライアントサイドフックは開発体験の向上が目的であり、強制力がない。品質の最終防衛線はサーバーサイドフックまたはCI/CDで実施すべき。

### アンチパターン6: フックのエラーメッセージが不親切

```bash
# NG: エラーの原因が分からない
#!/bin/sh
npx eslint $FILES > /dev/null 2>&1
exit $?
# → 出力を完全に抑制、何が問題か分からない

# OK: 何が問題で、どう修正すべきか明示
#!/bin/sh
echo "Running ESLint..."
RESULT=$(npx eslint --format stylish $FILES 2>&1)
if [ $? -ne 0 ]; then
  echo "$RESULT"
  echo ""
  echo "修正方法:"
  echo "  npx eslint --fix <file>  # 自動修正"
  echo "  npx eslint <file>        # エラーの確認"
  exit 1
fi
```

---

## 16. FAQ

### Q1. huskyのフックが実行されない場合の対処法は？

**A1.** 以下の順番で確認してください。

```bash
# 1. huskyが正しくインストールされているか
$ cat .git/config | grep hooksPath
# → core.hooksPath = .husky と設定されているべき

# 2. フックファイルに実行権限があるか
$ ls -la .husky/pre-commit
# → -rwxr-xr-x であるべき
$ chmod +x .husky/pre-commit

# 3. prepare スクリプトが設定されているか
$ cat package.json | grep prepare
# → "prepare": "husky" であるべき

# 4. node_modulesを再インストール
$ rm -rf node_modules && npm install

# 5. Gitのバージョンを確認（2.9+必須）
$ git --version

# 6. core.hooksPathを手動設定
$ git config core.hooksPath .husky
```

### Q2. CI/CD環境ではhuskyのフックを無効にしたい場合は？

**A2.** husky v9では環境変数`HUSKY=0`で無効化できます。

```bash
# CI/CDのパイプラインで
$ HUSKY=0 npm install
# または
$ npm install --ignore-scripts
```

```yaml
# GitHub Actionsの例
- name: Install dependencies
  run: npm ci
  env:
    HUSKY: 0
```

### Q3. サーバーサイドフックはどのように管理するのか？

**A3.** サーバーサイドフック（pre-receive, update, post-receive）はGitホスティングサービスの機能で管理するのが一般的です。

- **GitHub**: Branch protection rules, GitHub Actions
- **GitLab**: Server hooks（管理者のみ）, CI/CD pipelines, Push rules
- **Bitbucket**: Repository hooks, Pipelines

自前のGitサーバーの場合は、bareリポジトリの`.git/hooks/`に直接配置します。

```bash
# 自前Gitサーバーでのフック配置
$ ssh git@server
$ cd /opt/git/myproject.git/hooks/
$ cat > pre-receive << 'EOF'
#!/bin/bash
# ポリシーチェック
while read old new ref; do
  # ... チェックロジック
done
EOF
$ chmod +x pre-receive
```

### Q4. フックをチーム全体で共有するにはどうすればよいか？

**A4.** 以下の方法があります。

```bash
# 方法1: husky（推奨 — Node.jsプロジェクト）
# npm install で自動的にフックが設定される
$ npm install --save-dev husky
$ npx husky init

# 方法2: core.hooksPath（言語非依存）
# プロジェクト内にフックディレクトリを作成
$ mkdir .githooks
$ cp .git/hooks/pre-commit .githooks/
$ git config core.hooksPath .githooks
# → チームメンバーは以下のコマンドで有効化
$ git config core.hooksPath .githooks

# 方法3: Makefileでセットアップ
# Makefile
setup:
	git config core.hooksPath .githooks
	chmod +x .githooks/*

# 方法4: setup.shスクリプト
#!/bin/sh
git config core.hooksPath .githooks
chmod +x .githooks/*
echo "Git hooks configured."
```

### Q5. 特定のファイルだけフックの対象から除外するには？

**A5.** lint-stagedの設定で除外パターンを指定します。

```json
{
  "lint-staged": {
    "!(generated|vendor)/**/*.{js,ts}": [
      "eslint --fix",
      "prettier --write"
    ]
  }
}
```

```javascript
// .lintstagedrc.js — より柔軟な除外
module.exports = {
  '*.{js,ts}': (filenames) => {
    const filtered = filenames.filter(f =>
      !f.includes('generated/') &&
      !f.includes('vendor/') &&
      !f.includes('.min.') &&
      !f.endsWith('.d.ts')
    );
    if (filtered.length === 0) return [];
    return [
      `eslint --fix ${filtered.join(' ')}`,
      `prettier --write ${filtered.join(' ')}`,
    ];
  },
};
```

### Q6. Git GUIクライアント（SourceTree, VS Code等）でフックが動作しない場合は？

**A6.** PATHの問題が最も一般的な原因です。

```bash
# GUIクライアントはシェルの初期化ファイルを読み込まないことがある
# → nvm, rbenv等のパスが通らない

# 解決策1: フックの先頭でPATHを設定
#!/bin/sh
export PATH="/usr/local/bin:/opt/homebrew/bin:$HOME/.nvm/versions/node/v20.0.0/bin:$PATH"
npx lint-staged

# 解決策2: ~/.huskyrc でPATHを設定
# ~/.huskyrc（husky v4の場合）
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

# 解決策3: VS Codeの設定
# settings.json
{
  "git.path": "/usr/local/bin/git",
  "terminal.integrated.env.osx": {
    "PATH": "/usr/local/bin:${env:PATH}"
  }
}
```

### Q7. pre-commitフックが部分ステージ（git add -p）と正しく動作しない場合は？

**A7.** lint-staged v13+では自動的に部分ステージを正しく処理します。

```bash
# lint-staged は以下の手順で部分ステージを処理:
# 1. ステージされていない変更をstash
# 2. ステージされた内容のみに対してlint実行
# 3. --fix による変更を再ステージ
# 4. stashを復元

# 手動フックの場合は明示的にstashを使用:
#!/bin/sh
# ステージされていない変更を退避
git stash -q --keep-index --include-untracked

# lintを実行
npx eslint $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(js|ts)$')
RESULT=$?

# 退避した変更を復元
git stash pop -q

exit $RESULT
```

### Q8. フックの実行順序をカスタマイズできるか？

**A8.** 同一種類のフック内では、シェルスクリプトの記述順に実行されます。複数のフックファイルを使いたい場合はディスパッチャーを使用します。

```bash
#!/bin/sh
# .husky/pre-commit — ディスパッチャーパターン

# フックを順次実行（1つでも失敗したら中断）
HOOKS_DIR=".husky/pre-commit.d"

if [ -d "$HOOKS_DIR" ]; then
  for hook in "$HOOKS_DIR"/*; do
    if [ -x "$hook" ]; then
      echo "Running $(basename "$hook")..."
      "$hook"
      RESULT=$?
      if [ $RESULT -ne 0 ]; then
        echo "$(basename "$hook") failed with exit code $RESULT"
        exit $RESULT
      fi
    fi
  done
fi
```

```
# ディレクトリ構造
.husky/
├── pre-commit          ← ディスパッチャー
├── pre-commit.d/
│   ├── 01-lint-staged  ← lint
│   ├── 02-typecheck    ← 型チェック
│   └── 03-secrets      ← 秘密情報チェック
├── commit-msg
└── pre-push
```

---

## まとめ

| 概念             | 要点                                                          |
|------------------|---------------------------------------------------------------|
| Git Hooks        | 特定のGit操作時に自動実行されるスクリプト                    |
| pre-commit       | コミット前に実行、lint/formatの自動化に最適                  |
| commit-msg       | コミットメッセージの形式を検証                                |
| pre-push         | push前にテストを実行、main保護                               |
| husky            | Git Hooksの管理ツール、npm installで自動セットアップ         |
| lint-staged      | ステージファイルのみを対象にlint/formatを実行                |
| commitlint       | Conventional Commits形式のメッセージを検証                    |
| lefthook         | Go製の高速フックマネージャ、並列実行がビルトイン             |
| pre-commit fw    | Python製のフック管理フレームワーク、多言語対応               |
| サーバーサイド   | pre-receive/update/post-receiveでポリシーを強制              |
| core.hooksPath   | フックディレクトリの場所をカスタマイズ                        |
| clean/smudge     | ファイルの変換フィルター（暗号化、プレースホルダー等）       |
| gitleaks         | 秘密情報の検出ツール、フックと統合可能                        |

---

## 次に読むべきガイド

- [インタラクティブRebase](./00-interactive-rebase.md) — コミット整理とフックの連携
- [bisect/blame](./02-bisect-blame.md) — フックで検出できなかったバグの追跡
- [Jujutsuワークフロー](../02-jujutsu/01-jujutsu-workflow.md) — Jujutsuでのフック相当機能

---

## 参考文献

1. **Pro Git Book** — "Customizing Git - Git Hooks" https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks
2. **husky公式ドキュメント** — https://typicode.github.io/husky/
3. **lint-staged公式ドキュメント** — https://github.com/lint-staged/lint-staged
4. **commitlint公式ドキュメント** — https://commitlint.js.org/
5. **lefthook公式ドキュメント** — https://github.com/evilmartians/lefthook
6. **pre-commit framework** — https://pre-commit.com/
7. **gitleaks** — https://github.com/gitleaks/gitleaks
8. **Git公式ドキュメント: githooks** — https://git-scm.com/docs/githooks
