# Git Hooks

> Git Hooksの仕組みとライフサイクルを理解し、pre-commit、commit-msg、pre-pushなどのフックをhusky・lint-stagedと組み合わせて開発ワークフローを自動化する方法を解説する。

## この章で学ぶこと

1. **Git Hooksの種類とライフサイクル** — クライアントサイド/サーバーサイドの各フックの発火タイミングと用途
2. **husky + lint-staged による自動化** — モダンなフック管理ツールの導入と設定
3. **実践的なフック設計パターン** — CI/CDとの連携、チーム共有、パフォーマンス最適化

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
```

### 1.2 Hooksのライフサイクル

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
└─────────────────────────────────────────────────────┘
```

### 1.3 クライアントサイドフック一覧

| フック                 | タイミング                    | 用途                            |
|------------------------|-------------------------------|---------------------------------|
| `pre-commit`           | commit前                      | lint、format、テスト            |
| `prepare-commit-msg`   | メッセージ編集前              | テンプレート挿入                |
| `commit-msg`           | メッセージ編集後              | メッセージ形式の検証            |
| `post-commit`          | commit後                      | 通知                            |
| `pre-rebase`           | rebase前                      | rebase可否の判断                |
| `post-rewrite`         | amend/rebase後                | 関連処理の実行                  |
| `pre-push`             | push前                        | テスト、ブランチ保護            |
| `post-checkout`        | checkout後                    | 依存関係の更新                  |
| `post-merge`           | merge後                       | 依存関係の更新                  |
| `pre-auto-gc`          | GC前                          | GCの制御                        |

### 1.4 サーバーサイドフック一覧

| フック            | タイミング            | 用途                                |
|-------------------|-----------------------|-------------------------------------|
| `pre-receive`     | push受信前            | 全refの一括チェック                 |
| `update`          | 各ref更新前           | ブランチ単位のポリシー適用          |
| `post-receive`    | push受信後            | CI/CDトリガー、チャット通知         |
| `post-update`     | ref更新後             | `git update-server-info`の実行等    |

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

### 2.2 commit-msgフック

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

### 2.3 pre-pushフック

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

---

## 5. commitlint — コミットメッセージの検証

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

---

## 6. 完全な設定例

### 6.1 package.json

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

### 6.2 .husky/ ディレクトリ

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

---

## 7. アンチパターン

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

---

## 8. FAQ

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
- **GitLab**: Server hooks（管理者のみ）, CI/CD pipelines
- **Bitbucket**: Repository hooks, Pipelines

自前のGitサーバーの場合は、bareリポジトリの`.git/hooks/`に直接配置します。

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
| core.hooksPath   | フックディレクトリの場所をカスタマイズ                        |

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
