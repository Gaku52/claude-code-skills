# 再利用ワークフロー

> Composite Actions と Reusable Workflows を使って DRY 原則に基づいた保守性の高いCI/CDパイプラインを設計する

## この章で学ぶこと

1. Composite Actions と Reusable Workflows の違いと使い分けを理解する
2. 再利用可能なワークフローの設計・実装・公開方法を習得する
3. 組織全体で共有するCI/CDライブラリの構築パターンを把握する

---

## 1. 再利用の2つのアプローチ

### 1.1 全体像

```
再利用の階層:

┌──────────────────────────────────────────────┐
│ Reusable Workflow (workflow_call)              │
│ ワークフロー全体を再利用                        │
│ ┌──────────────────────────────────────────┐  │
│ │ Job A                                     │  │
│ │ ┌──────────────────────────────────────┐ │  │
│ │ │ Step 1: Composite Action             │ │  │
│ │ │ (複数ステップをまとめた再利用単位)      │ │  │
│ │ ├──────────────────────────────────────┤ │  │
│ │ │ Step 2: 通常のアクション              │ │  │
│ │ ├──────────────────────────────────────┤ │  │
│ │ │ Step 3: run コマンド                 │ │  │
│ │ └──────────────────────────────────────┘ │  │
│ └──────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘

Composite Action:
  - ステップレベルの再利用
  - 1つのジョブ内の複数ステップをまとめる
  - action.yml で定義

Reusable Workflow:
  - ワークフローレベルの再利用
  - ジョブ全体を含むワークフローを呼び出す
  - workflow_call トリガーで定義
```

---

## 2. Composite Actions

### 2.1 基本構造

```yaml
# .github/actions/setup-and-build/action.yml
name: 'Setup and Build'
description: 'Node.js のセットアップ、依存インストール、ビルドを一括実行'

inputs:
  node-version:
    description: 'Node.js バージョン'
    required: false
    default: '20'
  working-directory:
    description: '作業ディレクトリ'
    required: false
    default: '.'

outputs:
  build-path:
    description: 'ビルド出力パス'
    value: ${{ steps.build.outputs.path }}

runs:
  using: 'composite'
  steps:
    - uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}
        cache: 'npm'
        cache-dependency-path: ${{ inputs.working-directory }}/package-lock.json

    - name: Install dependencies
      shell: bash
      working-directory: ${{ inputs.working-directory }}
      run: npm ci

    - name: Build
      id: build
      shell: bash
      working-directory: ${{ inputs.working-directory }}
      run: |
        npm run build
        echo "path=${{ inputs.working-directory }}/dist" >> "$GITHUB_OUTPUT"
```

### 2.2 Composite Action の使用

```yaml
# .github/workflows/ci.yml
name: CI
on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # ローカルの Composite Action を使用
      - uses: ./.github/actions/setup-and-build
        id: build
        with:
          node-version: '20'

      - run: echo "Build output at ${{ steps.build.outputs.build-path }}"
```

### 2.3 実践的な Composite Action: テスト実行

```yaml
# .github/actions/run-tests/action.yml
name: 'Run Tests'
description: 'テスト実行とカバレッジレポート生成'

inputs:
  test-command:
    description: 'テストコマンド'
    default: 'npm test -- --coverage'
  coverage-threshold:
    description: 'カバレッジ閾値(%)'
    default: '80'

outputs:
  coverage-percent:
    description: 'カバレッジ率'
    value: ${{ steps.coverage.outputs.percent }}

runs:
  using: 'composite'
  steps:
    - name: Run tests
      shell: bash
      run: ${{ inputs.test-command }}

    - name: Check coverage threshold
      id: coverage
      shell: bash
      run: |
        COVERAGE=$(jq '.total.lines.pct' coverage/coverage-summary.json)
        echo "percent=$COVERAGE" >> "$GITHUB_OUTPUT"
        if (( $(echo "$COVERAGE < ${{ inputs.coverage-threshold }}" | bc -l) )); then
          echo "::error::Coverage ${COVERAGE}% is below threshold ${{ inputs.coverage-threshold }}%"
          exit 1
        fi
        echo "Coverage: ${COVERAGE}% (threshold: ${{ inputs.coverage-threshold }}%)"

    - name: Upload coverage
      uses: actions/upload-artifact@v4
      with:
        name: coverage-report
        path: coverage/
        retention-days: 7
```

---

## 3. Reusable Workflows

### 3.1 定義

```yaml
# .github/workflows/reusable-ci.yml
name: Reusable CI

on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
      working-directory:
        type: string
        default: '.'
      run-e2e:
        type: boolean
        default: false
    secrets:
      NPM_TOKEN:
        required: false
    outputs:
      build-version:
        description: 'ビルドバージョン'
        value: ${{ jobs.build.outputs.version }}

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'npm'
      - run: npm ci
        working-directory: ${{ inputs.working-directory }}
      - run: npm run lint
        working-directory: ${{ inputs.working-directory }}

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'npm'
      - run: npm ci
        working-directory: ${{ inputs.working-directory }}
      - run: npm test
        working-directory: ${{ inputs.working-directory }}

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.value }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'npm'
      - run: npm ci
        working-directory: ${{ inputs.working-directory }}
      - run: npm run build
        working-directory: ${{ inputs.working-directory }}
      - id: version
        run: echo "value=$(jq -r .version package.json)" >> "$GITHUB_OUTPUT"
        working-directory: ${{ inputs.working-directory }}

  e2e:
    if: inputs.run-e2e
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npx playwright install --with-deps
      - run: npm run test:e2e
```

### 3.2 呼び出し

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  ci:
    uses: ./.github/workflows/reusable-ci.yml
    with:
      node-version: '20'
      run-e2e: ${{ github.event_name == 'push' }}
    secrets:
      NPM_TOKEN: ${{ secrets.NPM_TOKEN }}

  deploy:
    needs: ci
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploy version ${{ needs.ci.outputs.build-version }}"
```

### 3.3 他リポジトリの Reusable Workflow を呼び出し

```yaml
# 組織共通のワークフローを呼び出す
name: CI
on: [push]

jobs:
  ci:
    uses: my-org/shared-workflows/.github/workflows/node-ci.yml@v1
    with:
      node-version: '20'
    secrets: inherit  # 呼び出し元のシークレットを全て継承
```

---

## 4. Composite Actions vs Reusable Workflows

### 4.1 比較表

| 項目 | Composite Action | Reusable Workflow |
|---|---|---|
| 再利用の単位 | ステップ群 | ジョブ群(ワークフロー全体) |
| 定義場所 | action.yml | .github/workflows/*.yml |
| ランナー指定 | 呼び出し元が決定 | 内部で runs-on を指定 |
| シークレット | 呼び出し元の文脈で利用可能 | secrets で明示的に受け渡し |
| ネスト | 可能(Action内でAction呼出) | 最大4階層 |
| マーケットプレイス | 公開可能 | 公開可能(リポジトリ参照) |
| 適用場面 | 共通のセットアップ手順 | 標準化されたCI/CDフロー |
| 柔軟性 | 高(ステップレベル) | 中(ジョブ単位) |

### 4.2 使い分けガイド

```
判断フローチャート:

  再利用したいものは？
       │
  ┌────┴────┐
  │         │
  ステップ   ジョブ全体
  (手順)    (フロー)
  │         │
  ↓         ↓
  Composite  Reusable
  Action     Workflow

  さらに:
  - ランナーを呼び出し側で決めたい → Composite Action
  - 環境 (environment) を使いたい → Reusable Workflow
  - マーケットプレイスに公開したい → Composite Action
  - 組織の標準CIフローを強制したい → Reusable Workflow
```

---

## 5. Action の公開

### 5.1 ディレクトリ構成

```
my-action/
├── action.yml          # アクション定義
├── src/
│   └── index.ts        # JavaScript Action の場合
├── dist/
│   └── index.js        # ビルド済みファイル
├── package.json
├── tsconfig.json
├── LICENSE
└── README.md
```

### 5.2 JavaScript Action の例

```yaml
# action.yml
name: 'PR Size Label'
description: '変更行数に基づいてPRにサイズラベルを付与する'
author: 'your-name'

inputs:
  github-token:
    description: 'GitHub Token'
    required: true
  xs-threshold:
    description: 'XSの閾値'
    default: '10'
  s-threshold:
    description: 'Sの閾値'
    default: '50'

runs:
  using: 'node20'
  main: 'dist/index.js'

branding:
  icon: 'tag'
  color: 'blue'
```

```typescript
// src/index.ts
import * as core from '@actions/core';
import * as github from '@actions/github';

async function run(): Promise<void> {
  const token = core.getInput('github-token', { required: true });
  const xsThreshold = parseInt(core.getInput('xs-threshold'));
  const sThreshold = parseInt(core.getInput('s-threshold'));

  const octokit = github.getOctokit(token);
  const { context } = github;

  if (!context.payload.pull_request) {
    core.info('Not a PR event, skipping.');
    return;
  }

  const { data: pr } = await octokit.rest.pulls.get({
    ...context.repo,
    pull_number: context.payload.pull_request.number,
  });

  const totalChanges = pr.additions + pr.deletions;
  let label = 'size/XL';
  if (totalChanges < xsThreshold) label = 'size/XS';
  else if (totalChanges < sThreshold) label = 'size/S';

  await octokit.rest.issues.addLabels({
    ...context.repo,
    issue_number: pr.number,
    labels: [label],
  });

  core.setOutput('label', label);
}

run().catch(core.setFailed);
```

---

## 6. 組織共通ワークフローのパターン

```
組織共通リポジトリ構成:

  my-org/shared-workflows/
  ├── .github/
  │   └── workflows/
  │       ├── node-ci.yml        # Node.js CI
  │       ├── python-ci.yml      # Python CI
  │       ├── docker-build.yml   # Docker ビルド
  │       └── deploy.yml         # デプロイ
  ├── actions/
  │   ├── setup-node/
  │   │   └── action.yml
  │   ├── security-scan/
  │   │   └── action.yml
  │   └── notify-slack/
  │       └── action.yml
  └── README.md

  各プロジェクトリポジトリ:
  my-org/my-app/.github/workflows/ci.yml
    → uses: my-org/shared-workflows/.github/workflows/node-ci.yml@v1
```

---

## 7. アンチパターン

### アンチパターン1: 過度な抽象化

```yaml
# 悪い例: 全てを Reusable Workflow にして理解困難に
jobs:
  setup:
    uses: ./.github/workflows/reusable-setup.yml
  lint:
    uses: ./.github/workflows/reusable-lint.yml
  test:
    uses: ./.github/workflows/reusable-test.yml
  build:
    uses: ./.github/workflows/reusable-build.yml
  # 5つのファイルを見ないと全体像がわからない

# 改善: 適切な粒度で抽象化
# - 組織で共通化すべき部分のみ Reusable に
# - プロジェクト固有のロジックはインラインで
```

### アンチパターン2: バージョン固定なしの参照

```yaml
# 悪い例: ブランチ参照 → 予期しない変更で壊れる
jobs:
  ci:
    uses: my-org/shared-workflows/.github/workflows/ci.yml@main

# 改善: セマンティックバージョニングで固定
jobs:
  ci:
    uses: my-org/shared-workflows/.github/workflows/ci.yml@v2
    # またはコミットSHAで固定
    # uses: my-org/shared-workflows/.github/workflows/ci.yml@abc1234
```

---

## 8. FAQ

### Q1: Reusable Workflow のネストは何階層まで可能か？

最大4階層まで。ただし、深いネストは可読性を大きく損なうため、2階層以内を推奨する。それ以上の共通化が必要な場合は Composite Action に切り出して、Reusable Workflow のステップ内で使う構成が良い。

### Q2: Reusable Workflow でマトリクスは使えるか？

呼び出し元でマトリクスを使って同じ Reusable Workflow を異なるパラメータで呼び出すことが可能。Reusable Workflow 内部でもマトリクスを使える。ただし、呼び出し元のマトリクスと内部のマトリクスを組み合わせると実行ジョブ数が爆発するため注意。

### Q3: secrets: inherit は安全か？

`secrets: inherit` は呼び出し元の全シークレットを渡す。便利だが、Reusable Workflow が信頼できるリポジトリにある場合のみ使うべき。外部リポジトリの Reusable Workflow には明示的に必要なシークレットだけを渡す方が安全。

---

## まとめ

| 項目 | 要点 |
|---|---|
| Composite Action | ステップ群をまとめて再利用、action.yml で定義 |
| Reusable Workflow | ジョブ群を再利用、workflow_call で定義 |
| 使い分け | セットアップ手順 → Composite、CIフロー → Reusable |
| 公開 | マーケットプレイス(Action)、リポジトリ参照(Workflow) |
| バージョニング | セマンティックバージョンかSHAで固定必須 |
| 組織パターン | shared-workflows リポジトリに集約 |

---

## 次に読むべきガイド

- [CI レシピ集](./03-ci-recipes.md) -- 再利用ワークフローを活用した実践例
- [Actions セキュリティ](./04-security-actions.md) -- 公開アクションのセキュリティ
- [GitHub Actions 基礎](./00-actions-basics.md) -- 基本構文の復習

---

## 参考文献

1. GitHub. "Reusing workflows." https://docs.github.com/en/actions/using-workflows/reusing-workflows
2. GitHub. "Creating a composite action." https://docs.github.com/en/actions/creating-actions/creating-a-composite-action
3. GitHub. "Publishing actions in GitHub Marketplace." https://docs.github.com/en/actions/creating-actions/publishing-actions-in-github-marketplace
