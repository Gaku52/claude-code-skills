# GitHub Actions 基礎

> GitHub に統合された CI/CD プラットフォームで、ワークフロー・ジョブ・ステップの階層構造とYAML構文を理解する

## この章で学ぶこと

1. ワークフロー、ジョブ、ステップの関係と実行モデルを理解する
2. トリガー(イベント)の種類と使い分けを習得する
3. 基本的なワークフローYAMLの読み書きができるようになる

---

## 1. GitHub Actions の構造

### 1.1 階層構造

```
┌──────────────────────────────────────────────────┐
│ Workflow (.github/workflows/*.yml)                │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ Job 1 (runs-on: ubuntu-latest)               │ │
│  │  ┌──────────────────────────────────────────┐│ │
│  │  │ Step 1: actions/checkout@v4              ││ │
│  │  ├──────────────────────────────────────────┤│ │
│  │  │ Step 2: actions/setup-node@v4            ││ │
│  │  ├──────────────────────────────────────────┤│ │
│  │  │ Step 3: run: npm ci                      ││ │
│  │  ├──────────────────────────────────────────┤│ │
│  │  │ Step 4: run: npm test                    ││ │
│  │  └──────────────────────────────────────────┘│ │
│  └──────────────────────────────────────────────┘ │
│                                                    │
│  ┌──────────────────────────────────────────────┐ │
│  │ Job 2 (needs: job1, runs-on: ubuntu-latest)  │ │
│  │  ┌──────────────────────────────────────────┐│ │
│  │  │ Step 1: run: npm run build               ││ │
│  │  └──────────────────────────────────────────┘│ │
│  └──────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────┘

重要な関係:
- Workflow: 1つの YAML ファイル = 1つのワークフロー
- Job: 1つのランナー(仮想マシン)で実行される単位
- Step: Job 内で順次実行される個々のタスク
- Jobs はデフォルトで並列実行、needs で依存関係を定義
```

### 1.2 実行環境

```
ランナー (Runner):
  GitHub が提供するホステッドランナー:
  ┌─────────────────────────────┐
  │ ubuntu-latest (Ubuntu 22.04) │ ← 最も一般的
  │ ubuntu-24.04                 │
  │ windows-latest               │
  │ macos-latest                 │
  │ macos-14 (Apple Silicon)     │
  └─────────────────────────────┘

  セルフホステッドランナー:
  ┌─────────────────────────────┐
  │ runs-on: self-hosted        │ ← 自前のマシン
  │ runs-on: [self-hosted, gpu] │ ← ラベルで選択
  └─────────────────────────────┘
```

---

## 2. 基本構文

### 2.1 最小限のワークフロー

```yaml
# .github/workflows/ci.yml
name: CI                           # ワークフロー名

on:                                # トリガー
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:                              # ジョブ定義
  test:                            # ジョブ ID
    runs-on: ubuntu-latest         # 実行環境
    steps:                         # ステップ
      - uses: actions/checkout@v4  # アクション呼び出し
      - run: echo "Hello, World!" # シェルコマンド実行
```

### 2.2 完全な CI ワークフロー

```yaml
name: CI Pipeline

on:
  push:
    branches: [main]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [main]

# ワークフローレベルの権限設定
permissions:
  contents: read
  pull-requests: write

# 同じブランチの並行実行を制御
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  NODE_VERSION: '20'

jobs:
  lint-and-typecheck:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: ESLint
        run: npm run lint

      - name: TypeScript
        run: npm run type-check

  test:
    name: Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - run: npm ci
      - run: npm test -- --coverage

      - name: Upload coverage
        if: github.event_name == 'pull_request'
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage/

  build:
    name: Build
    needs: [lint-and-typecheck, test]   # 両ジョブ成功後に実行
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'

      - run: npm ci
      - run: npm run build

      - uses: actions/upload-artifact@v4
        with:
          name: build-output
          path: dist/
          retention-days: 7
```

---

## 3. トリガー (Events)

### 3.1 主要トリガー一覧

```yaml
on:
  # Git イベント
  push:
    branches: [main, 'release/**']
    tags: ['v*']
    paths: ['src/**', 'package.json']
    paths-ignore: ['**.md']

  pull_request:
    types: [opened, synchronize, reopened]
    branches: [main]

  # スケジュール (cron)
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜 9:00 UTC

  # 手動実行
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deploy environment'
        required: true
        type: choice
        options: [dev, staging, prod]
      dry_run:
        description: 'Dry run mode'
        type: boolean
        default: false

  # 他のワークフローから呼び出し
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'

  # リリースイベント
  release:
    types: [published]

  # Issue / PR コメント
  issue_comment:
    types: [created]
```

### 3.2 トリガー比較表

| トリガー | 用途 | コンテキスト | 注意点 |
|---|---|---|---|
| push | メインブランチCI、デプロイ | github.sha = push先コミット | paths フィルタ活用 |
| pull_request | PRのCI、レビュー支援 | github.sha = マージコミット | Fork PRは権限制限 |
| schedule | 定期バッチ、依存更新 | デフォルトブランチ | 遅延あり(保証なし) |
| workflow_dispatch | 手動デプロイ、操作 | inputs で引数受取 | GitHub UIから実行 |
| workflow_call | 再利用ワークフロー | 呼び出し元の文脈 | inputs/secrets を定義 |
| release | リリース自動化 | tag 情報を取得可能 | published を推奨 |

---

## 4. 式と関数

### 4.1 コンテキストと式

```yaml
jobs:
  example:
    runs-on: ubuntu-latest
    steps:
      # 式の基本構文: ${{ expression }}
      - run: echo "Branch: ${{ github.ref_name }}"
      - run: echo "Actor: ${{ github.actor }}"
      - run: echo "SHA: ${{ github.sha }}"
      - run: echo "Event: ${{ github.event_name }}"

      # 条件分岐
      - name: Deploy (main のみ)
        if: github.ref == 'refs/heads/main'
        run: ./deploy.sh

      # 前のステップの結果で条件分岐
      - name: On failure
        if: failure()
        run: echo "前のステップが失敗しました"

      # 常に実行 (クリーンアップ等)
      - name: Cleanup
        if: always()
        run: rm -rf tmp/

      # 式の中の関数
      - name: Contains check
        if: contains(github.event.head_commit.message, '[skip ci]')
        run: echo "CI skipped"

      - name: String comparison
        if: startsWith(github.ref, 'refs/tags/v')
        run: echo "Tag push detected"
```

### 4.2 ジョブ間のデータ受け渡し

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.value }}
    steps:
      - id: version
        run: echo "value=$(cat package.json | jq -r .version)" >> "$GITHUB_OUTPUT"

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying version ${{ needs.build.outputs.version }}"
```

---

## 5. 基本的なアクション

### 5.1 よく使うアクション

```yaml
steps:
  # リポジトリのチェックアウト
  - uses: actions/checkout@v4
    with:
      fetch-depth: 0  # 全履歴 (タグ取得等に必要)

  # Node.js セットアップ
  - uses: actions/setup-node@v4
    with:
      node-version: '20'
      cache: 'npm'
      registry-url: 'https://npm.pkg.github.com'

  # Python セットアップ
  - uses: actions/setup-python@v5
    with:
      python-version: '3.12'
      cache: 'pip'

  # Go セットアップ
  - uses: actions/setup-go@v5
    with:
      go-version: '1.22'

  # キャッシュ
  - uses: actions/cache@v4
    with:
      path: ~/.npm
      key: ${{ runner.os }}-npm-${{ hashFiles('package-lock.json') }}
      restore-keys: |
        ${{ runner.os }}-npm-

  # アーティファクトのアップロード
  - uses: actions/upload-artifact@v4
    with:
      name: my-artifact
      path: dist/
      retention-days: 7

  # アーティファクトのダウンロード
  - uses: actions/download-artifact@v4
    with:
      name: my-artifact
      path: dist/
```

---

## 6. 実行フロー図

```
ワークフロー実行の流れ:

  イベント発生 (push, PR, etc.)
       │
       ↓
  ┌──────────────────┐
  │ トリガー条件評価   │ ← branches, paths フィルタ
  │ マッチするか？     │
  └────────┬─────────┘
           │ Yes
           ↓
  ┌──────────────────┐
  │ ジョブグラフ構築   │ ← needs による依存関係解決
  └────────┬─────────┘
           │
     ┌─────┼─────┐
     ↓     ↓     ↓
  ┌─────┐┌─────┐┌─────┐
  │Job A ││Job B ││Job C │  ← 並列実行 (依存なし)
  └──┬──┘└──┬──┘└─────┘
     │      │
     ↓      ↓
  ┌──────────┐
  │  Job D    │  ← needs: [A, B]
  └──────────┘
       │
       ↓
  ┌──────────┐
  │  完了     │
  └──────────┘

各ジョブ内のステップは順次実行:
  Step 1 → Step 2 → Step 3 → ... → Step N
  (1つ失敗するとジョブは停止、if: always() を除く)
```

---

## 7. アンチパターン

### アンチパターン1: ワークフローの肥大化

```yaml
# 悪い例: 1つのワークフローに全てを詰め込む
name: Everything
on: [push]
jobs:
  everything:
    runs-on: ubuntu-latest
    steps:
      - run: npm run lint        # 10s
      - run: npm test            # 60s
      - run: npm run build       # 30s
      - run: npm run e2e         # 300s
      - run: docker build .      # 120s
      - run: ./deploy.sh         # 60s
      # 合計 ~10分、1つ失敗すると全てやり直し

# 改善: ジョブを分割して並列実行
name: CI
on: [push]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: npm run lint
  test:
    runs-on: ubuntu-latest
    steps:
      - run: npm test
  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - run: npm run build
```

### アンチパターン2: ハードコードされたバージョン

```yaml
# 悪い例: バージョンが散在
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-node@v4
    with:
      node-version: '20'
  - run: npm ci
  - run: npm test

# 別のワークフローでも同じ定義を繰り返す...

# 改善: 環境変数やReusable Workflowで一元管理
env:
  NODE_VERSION: '20'

jobs:
  ci:
    uses: ./.github/workflows/reusable-ci.yml
    with:
      node-version: '20'
```

---

## 8. FAQ

### Q1: GitHub Actions の無料枠はどのくらいか？

パブリックリポジトリは無制限無料。プライベートリポジトリは GitHub Free で月 2,000 分、Team で月 3,000 分、Enterprise で月 50,000 分。macOS と Windows のランナーは Linux の 2倍/10倍の消費レートが適用される。

### Q2: `actions/checkout@v4` は何をしているのか？

ランナーの仮想マシンにリポジトリのコードをクローンする。ステップの最初に必ず実行する必要がある。`fetch-depth: 0` で全履歴を取得(デフォルトは shallow clone で depth=1)。PR の場合はマージコミットをチェックアウトする。

### Q3: ジョブ間でファイルを共有するには？

ジョブは別々のランナー(VM)で実行されるため、ファイルシステムは共有されない。`actions/upload-artifact` と `actions/download-artifact` を使ってアーティファクト経由で受け渡す。小さな値は `outputs` で受け渡す。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 階層構造 | Workflow > Job > Step |
| ジョブ実行 | デフォルト並列、needs で依存関係定義 |
| ステップ実行 | 順次実行、uses(アクション)と run(コマンド) |
| トリガー | push, pull_request, schedule, workflow_dispatch 等 |
| 式 | ${{ expression }} でコンテキスト参照・条件分岐 |
| 権限 | permissions で最小権限を設定 |

---

## 次に読むべきガイド

- [Actions 応用](./01-actions-advanced.md) -- マトリクス、キャッシュ、シークレット
- [再利用ワークフロー](./02-reusable-workflows.md) -- DRY なワークフロー設計
- [CI レシピ集](./03-ci-recipes.md) -- 言語別の実践的 CI 設定

---

## 参考文献

1. GitHub. "GitHub Actions Documentation." https://docs.github.com/en/actions
2. GitHub. "Workflow syntax for GitHub Actions." https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions
3. GitHub. "Events that trigger workflows." https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows
