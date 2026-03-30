# GitHub Actions 基礎

> GitHub に統合された CI/CD プラットフォームで、ワークフロー・ジョブ・ステップの階層構造とYAML構文を理解する

## この章で学ぶこと

1. ワークフロー、ジョブ、ステップの関係と実行モデルを理解する
2. トリガー(イベント)の種類と使い分けを習得する
3. 基本的なワークフローYAMLの読み書きができるようになる
4. 式・関数・コンテキストを使った動的な制御ができる
5. 基本的なアクションの使い方とベストプラクティスを理解する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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
  ┌──────────────────────────────────┐
  │ ubuntu-latest (Ubuntu 22.04)      │ ← 最も一般的
  │ ubuntu-24.04                      │
  │ windows-latest                    │
  │ macos-latest                      │
  │ macos-14 (Apple Silicon)          │
  └──────────────────────────────────┘

  Larger Runner (有料):
  ┌──────────────────────────────────┐
  │ ubuntu-latest-4-cores            │
  │ ubuntu-latest-8-cores            │
  │ ubuntu-latest-16-cores           │
  │ windows-latest-8-cores           │
  │ macos-latest-xlarge (M1)         │
  └──────────────────────────────────┘

  セルフホステッドランナー:
  ┌──────────────────────────────────┐
  │ runs-on: self-hosted             │ ← 自前のマシン
  │ runs-on: [self-hosted, gpu]      │ ← ラベルで選択
  │ runs-on: [self-hosted, linux, x64]│ ← 複数ラベル
  └──────────────────────────────────┘
```

### 1.3 ランナーの仕様

| ランナー | CPU | メモリ | ストレージ | 消費レート |
|---|---|---|---|---|
| ubuntu-latest | 2 vCPU | 7 GB | 14 GB SSD | 1x |
| windows-latest | 2 vCPU | 7 GB | 14 GB SSD | 2x |
| macos-latest | 3 vCPU | 14 GB | 14 GB SSD | 10x |
| macos-14 (M1) | 3 vCPU | 7 GB | 14 GB SSD | 10x |
| ubuntu-latest-4-cores | 4 vCPU | 16 GB | 150 GB SSD | 有料 |
| ubuntu-latest-16-cores | 16 vCPU | 64 GB | 150 GB SSD | 有料 |

### 1.4 プリインストールソフトウェア

```
ubuntu-latest にプリインストールされているもの:
  ├── 言語: Node.js, Python, Go, Java, Ruby, .NET, Rust
  ├── パッケージマネージャ: npm, pip, Maven, Gradle
  ├── コンテナ: Docker, Docker Compose
  ├── CLI: AWS CLI, Azure CLI, gcloud, gh (GitHub CLI)
  ├── ビルドツール: Make, CMake, gcc, g++
  ├── データベース: PostgreSQL, MySQL (サービスとして利用可能)
  └── その他: Git, curl, wget, jq, zip, unzip

確認方法:
  https://github.com/actions/runner-images
```

---

## 2. 基本構文

### 2.1 最小限のワークフロー

```yaml
# .github/workflows/ci.yml
name: CI                           # ワークフロー名
                                   # (GitHub UI の Actions タブに表示)

on:                                # トリガー
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:                              # ジョブ定義
  test:                            # ジョブ ID (英数字とハイフン)
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

### 2.3 ワークフロー YAML の構造解説

```yaml
# YAML の基本構造
name: string          # ワークフロー名 (必須ではないが推奨)

on:                   # トリガー定義 (必須)
  event: ...

permissions:          # GITHUB_TOKEN の権限 (推奨: 最小権限)
  contents: read
  pull-requests: write

concurrency:          # 並行実行制御 (推奨: PR の重複実行を防ぐ)
  group: string
  cancel-in-progress: boolean

env:                  # ワークフローレベルの環境変数
  KEY: value

defaults:             # ジョブ/ステップのデフォルト設定
  run:
    shell: bash
    working-directory: ./app

jobs:                 # ジョブ定義 (必須)
  job-id:
    name: string              # 表示名
    runs-on: string           # ランナー (必須)
    needs: [job-id, ...]      # 依存ジョブ
    if: expression            # 実行条件
    timeout-minutes: number   # タイムアウト (デフォルト: 360分)
    continue-on-error: bool   # 失敗しても次のジョブに進む
    strategy:                 # マトリクス等
      matrix: ...
    env:                      # ジョブレベルの環境変数
      KEY: value
    outputs:                  # ジョブの出力
      key: value
    services:                 # サービスコンテナ
      name:
        image: ...
    steps:                    # ステップ (必須)
      - uses: action@version  # アクション
        with:                 # アクションの入力
          key: value
      - run: command          # シェルコマンド
        env:                  # ステップレベルの環境変数
          KEY: value
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

  pull_request_target:
    types: [opened, synchronize]
    # Fork からの PR でも secrets にアクセス可能 (注意: セキュリティリスク)

  # スケジュール (cron)
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜 9:00 UTC
    - cron: '0 0 1 * *'  # 毎月1日 0:00 UTC

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
      version:
        description: 'Version to deploy'
        type: string
        required: false

  # 他のワークフローから呼び出し
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
    secrets:
      npm-token:
        required: true

  # リリースイベント
  release:
    types: [published]

  # Issue / PR コメント
  issue_comment:
    types: [created]

  # デプロイステータス
  deployment_status:

  # ラベル操作
  label:
    types: [created, edited, deleted]

  # ワークフロー完了
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches: [main]
```

### 3.2 トリガー比較表

| トリガー | 用途 | コンテキスト | 注意点 |
|---|---|---|---|
| push | メインブランチCI、デプロイ | github.sha = push先コミット | paths フィルタ活用 |
| pull_request | PRのCI、レビュー支援 | github.sha = マージコミット | Fork PRは権限制限 |
| pull_request_target | Fork PRのCI | base ブランチの文脈 | セキュリティ注意 |
| schedule | 定期バッチ、依存更新 | デフォルトブランチ | 遅延あり(保証なし) |
| workflow_dispatch | 手動デプロイ、操作 | inputs で引数受取 | GitHub UIから実行 |
| workflow_call | 再利用ワークフロー | 呼び出し元の文脈 | inputs/secrets を定義 |
| release | リリース自動化 | tag 情報を取得可能 | published を推奨 |
| workflow_run | 後続処理、通知 | 前のワークフローの結果 | ブランチフィルタ必須 |

### 3.3 フィルタリングの詳細

```yaml
# push イベントのフィルタリング
on:
  push:
    # ブランチフィルタ (glob パターン対応)
    branches:
      - main
      - 'release/**'        # release/1.0, release/2.0 等
      - '!release/**-beta'  # release/1.0-beta を除外
    branches-ignore:
      - 'feature/**'

    # タグフィルタ
    tags:
      - 'v*'                # v1.0.0, v2.0.0 等
    tags-ignore:
      - 'v*-rc*'            # v1.0.0-rc1 を除外

    # パスフィルタ (変更されたファイルで絞り込み)
    paths:
      - 'src/**'
      - 'package.json'
      - '!src/**/*.test.ts'  # テストファイルの変更は除外
    paths-ignore:
      - '**.md'
      - 'docs/**'
      - '.github/**'
```

### 3.4 workflow_dispatch の実践例

```yaml
# 手動デプロイワークフロー
name: Manual Deploy
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - dev
          - staging
          - production
      version:
        description: 'Image tag to deploy (e.g., v1.2.3 or abc1234)'
        required: true
        type: string
      skip_tests:
        description: 'Skip smoke tests after deploy'
        required: false
        type: boolean
        default: false

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment }}
    steps:
      - uses: actions/checkout@v4

      - name: Deploy
        run: |
          echo "Deploying ${{ github.event.inputs.version }} to ${{ github.event.inputs.environment }}"
          ./scripts/deploy.sh \
            --env ${{ github.event.inputs.environment }} \
            --version ${{ github.event.inputs.version }}

      - name: Smoke test
        if: github.event.inputs.skip_tests != 'true'
        run: ./scripts/smoke-test.sh ${{ github.event.inputs.environment }}
```

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
      - run: echo "Repository: ${{ github.repository }}"
      - run: echo "Run ID: ${{ github.run_id }}"
      - run: echo "Run Number: ${{ github.run_number }}"

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

      # 複合条件
      - name: Deploy on main push only
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        run: ./deploy.sh

      # PR からの実行かどうか
      - name: PR only
        if: github.event_name == 'pull_request'
        run: echo "Running on PR #${{ github.event.pull_request.number }}"
```

### 4.2 主要コンテキスト一覧

| コンテキスト | 説明 | 例 |
|---|---|---|
| github | イベント情報 | github.sha, github.ref, github.actor |
| env | 環境変数 | env.NODE_VERSION |
| vars | リポジトリ/組織の変数 | vars.DEPLOY_URL |
| secrets | シークレット | secrets.API_KEY |
| job | 現在のジョブ情報 | job.status |
| steps | ステップの出力 | steps.step-id.outputs.key |
| matrix | マトリクスの値 | matrix.node-version |
| needs | 依存ジョブの出力 | needs.build.outputs.version |
| runner | ランナー情報 | runner.os, runner.arch |
| strategy | マトリクス戦略 | strategy.fail-fast |
| inputs | workflow_dispatch の入力 | inputs.environment |

### 4.3 ジョブ間のデータ受け渡し

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      version: ${{ steps.version.outputs.value }}
      should-deploy: ${{ steps.check.outputs.deploy }}
    steps:
      - uses: actions/checkout@v4

      - id: version
        run: echo "value=$(cat package.json | jq -r .version)" >> "$GITHUB_OUTPUT"

      - id: check
        run: |
          if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
            echo "deploy=true" >> "$GITHUB_OUTPUT"
          else
            echo "deploy=false" >> "$GITHUB_OUTPUT"
          fi

  deploy:
    needs: build
    if: needs.build.outputs.should-deploy == 'true'
    runs-on: ubuntu-latest
    steps:
      - run: echo "Deploying version ${{ needs.build.outputs.version }}"
```

### 4.4 主要関数一覧

```yaml
# 文字列関数
contains('Hello World', 'World')     # true
startsWith('Hello', 'He')            # true
endsWith('Hello', 'lo')              # true
format('Hello {0}', 'World')         # 'Hello World'
join(github.event.commits.*.message, ', ')  # コミットメッセージを結合

# 比較関数
success()                             # 前のステップが成功
failure()                             # 前のステップが失敗
always()                              # 常に実行
cancelled()                           # キャンセルされた

# JSON 関数
toJSON(github.event)                  # JSON に変換
fromJSON('{"key": "value"}')          # JSON をパース

# ハッシュ関数
hashFiles('**/package-lock.json')     # ファイルの SHA-256 ハッシュ
hashFiles('**/*.go', 'go.sum')        # 複数ファイルのハッシュ
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
      ref: ${{ github.head_ref }}  # PR のソースブランチ
      token: ${{ secrets.PAT }}    # プライベートサブモジュール用

  # Node.js セットアップ
  - uses: actions/setup-node@v4
    with:
      node-version: '20'
      node-version-file: '.node-version'  # ファイルからバージョン取得
      cache: 'npm'
      registry-url: 'https://npm.pkg.github.com'

  # Python セットアップ
  - uses: actions/setup-python@v5
    with:
      python-version: '3.12'
      cache: 'pip'
      cache-dependency-path: 'requirements*.txt'

  # Go セットアップ
  - uses: actions/setup-go@v5
    with:
      go-version: '1.22'
      go-version-file: 'go.mod'  # go.mod からバージョン取得
      cache: true

  # Java セットアップ
  - uses: actions/setup-java@v4
    with:
      distribution: 'temurin'
      java-version: '21'
      cache: 'maven'

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
      path: |
        dist/
        !dist/**/*.map
      retention-days: 7
      if-no-files-found: error

  # アーティファクトのダウンロード
  - uses: actions/download-artifact@v4
    with:
      name: my-artifact
      path: dist/
```

### 5.2 GitHub Script アクション

```yaml
# actions/github-script: JavaScript で GitHub API を操作
steps:
  - uses: actions/github-script@v7
    with:
      script: |
        // PR にコメントを追加
        await github.rest.issues.createComment({
          owner: context.repo.owner,
          repo: context.repo.repo,
          issue_number: context.issue.number,
          body: '## CI Results\n\nAll checks passed!'
        });

  - uses: actions/github-script@v7
    id: get-pr
    with:
      result-encoding: string
      script: |
        // PR のラベルを取得
        const { data: labels } = await github.rest.issues.listLabelsOnIssue({
          owner: context.repo.owner,
          repo: context.repo.repo,
          issue_number: context.issue.number,
        });
        return labels.map(l => l.name).join(',');

  - run: echo "Labels: ${{ steps.get-pr.outputs.result }}"
```

### 5.3 Docker アクション

```yaml
# Docker イメージのビルド&プッシュ
steps:
  - uses: actions/checkout@v4

  # Docker Buildx のセットアップ
  - uses: docker/setup-buildx-action@v3

  # コンテナレジストリへのログイン
  - uses: docker/login-action@v3
    with:
      registry: ghcr.io
      username: ${{ github.actor }}
      password: ${{ secrets.GITHUB_TOKEN }}

  # メタデータの生成 (タグ、ラベル)
  - uses: docker/metadata-action@v5
    id: meta
    with:
      images: ghcr.io/${{ github.repository }}
      tags: |
        type=sha,prefix=
        type=semver,pattern={{version}}
        type=semver,pattern={{major}}.{{minor}}

  # ビルド&プッシュ
  - uses: docker/build-push-action@v5
    with:
      context: .
      push: ${{ github.event_name != 'pull_request' }}
      tags: ${{ steps.meta.outputs.tags }}
      labels: ${{ steps.meta.outputs.labels }}
      cache-from: type=gha
      cache-to: type=gha,mode=max
      platforms: linux/amd64,linux/arm64
```

---

## 6. Services (サービスコンテナ)

### 6.1 データベースサービス

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      elasticsearch:
        image: elasticsearch:8.11.0
        ports:
          - 9200:9200
        env:
          discovery.type: single-node
          xpack.security.enabled: "false"
        options: >-
          --health-cmd "curl -f http://localhost:9200/_cluster/health"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 10

    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test:integration
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379
          ELASTICSEARCH_URL: http://localhost:9200
```

---

## 7. 実行フロー図

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
  │ concurrency 評価  │ ← 同じグループの実行をキャンセル？
  └────────┬─────────┘
           │
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

## 8. 権限管理 (Permissions)

### 8.1 最小権限の原則

```yaml
# ワークフローレベルで権限を制限 (推奨)
permissions:
  contents: read        # リポジトリの読み取り
  pull-requests: write  # PR へのコメント
  issues: write         # Issue の操作

# ジョブレベルで上書き
jobs:
  deploy:
    permissions:
      id-token: write   # OIDC 認証用
      contents: read
    steps: ...

  comment:
    permissions:
      pull-requests: write
    steps: ...
```

### 8.2 利用可能な権限一覧

| 権限 | 用途 | 典型的なケース |
|---|---|---|
| contents | リポジトリのコード | checkout, リリース作成 |
| pull-requests | PR の操作 | コメント、レビュー |
| issues | Issue の操作 | Issue 作成、ラベル付け |
| actions | Actions の操作 | ワークフロー操作 |
| checks | チェック実行 | ステータスチェック |
| deployments | デプロイ管理 | デプロイステータス |
| id-token | OIDC トークン | AWS/GCP の OIDC 認証 |
| packages | パッケージ管理 | GHCR へのプッシュ |
| security-events | セキュリティ | SARIF のアップロード |
| statuses | コミットステータス | ステータス更新 |

---

## 9. Concurrency (並行実行制御)

```yaml
# PR ごとに最新の実行のみを保持 (古い実行をキャンセル)
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

# 例: 同じ PR に対して push が連続した場合
# Push 1 → CI 開始
# Push 2 → Push 1 の CI をキャンセル → Push 2 の CI 開始
# Push 3 → Push 2 の CI をキャンセル → Push 3 の CI 開始

# デプロイは並行実行を禁止 (キャンセルはしない)
concurrency:
  group: deploy-${{ github.event.inputs.environment }}
  cancel-in-progress: false
# → 前のデプロイ完了を待ってから次のデプロイを開始
```

---

## 10. 環境変数と GITHUB_OUTPUT

### 10.1 環境変数のスコープ

```yaml
# ワークフローレベル
env:
  NODE_VERSION: '20'
  CI: true

jobs:
  build:
    # ジョブレベル
    env:
      BUILD_MODE: production
    steps:
      # ステップレベル
      - run: npm run build
        env:
          VITE_API_URL: https://api.example.com

      # 動的に環境変数を設定 (後続ステップで使用可能)
      - run: echo "VERSION=1.2.3" >> "$GITHUB_ENV"
      - run: echo "Version is $VERSION"  # 1.2.3
```

### 10.2 GITHUB_OUTPUT (ステップ間のデータ受け渡し)

```yaml
steps:
  # 出力の設定
  - id: extract
    run: |
      echo "version=$(cat package.json | jq -r .version)" >> "$GITHUB_OUTPUT"
      echo "commit_count=$(git rev-list --count HEAD)" >> "$GITHUB_OUTPUT"

      # 複数行の出力
      echo "changelog<<EOF" >> "$GITHUB_OUTPUT"
      git log --oneline -5 >> "$GITHUB_OUTPUT"
      echo "EOF" >> "$GITHUB_OUTPUT"

  # 出力の参照
  - run: |
      echo "Version: ${{ steps.extract.outputs.version }}"
      echo "Commits: ${{ steps.extract.outputs.commit_count }}"
      echo "Changelog:"
      echo "${{ steps.extract.outputs.changelog }}"
```

---

## 11. アンチパターン

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
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run lint
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test
  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
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

### アンチパターン3: secrets の不適切な利用

```yaml
# 悪い例: Fork PR で secrets を使用
on:
  pull_request_target:
    types: [opened, synchronize]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          # Fork の信頼できないコードをチェックアウト
      - run: npm ci && npm test
        env:
          SECRET_KEY: ${{ secrets.SECRET_KEY }}
          # → Fork PR からシークレットが漏洩するリスク!

# 改善: pull_request_target では信頼できないコードを実行しない
# または、ラベルベースの承認フローを導入
```

### アンチパターン4: キャッシュの未使用

```yaml
# 悪い例: 毎回依存関係をインストール
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-node@v4
    with:
      node-version: '20'
      # cache: 'npm'  ← これがない!
  - run: npm ci  # 毎回 90 秒

# 改善: キャッシュを活用
steps:
  - uses: actions/checkout@v4
  - uses: actions/setup-node@v4
    with:
      node-version: '20'
      cache: 'npm'  # ← キャッシュ有効化
  - run: npm ci  # キャッシュヒット時は 3 秒
```


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

## 12. FAQ

### Q1: GitHub Actions の無料枠はどのくらいか？

パブリックリポジトリは無制限無料。プライベートリポジトリは GitHub Free で月 2,000 分、Team で月 3,000 分、Enterprise で月 50,000 分。macOS ランナーは Linux の 10倍、Windows ランナーは 2倍の消費レートが適用される。ストレージは 500 MB (Free) から 50 GB (Enterprise) まで。

### Q2: `actions/checkout@v4` は何をしているのか？

ランナーの仮想マシンにリポジトリのコードをクローンする。ステップの最初に必ず実行する必要がある。`fetch-depth: 0` で全履歴を取得(デフォルトは shallow clone で depth=1)。PR の場合はマージコミットをチェックアウトする。サブモジュールの取得は `submodules: true` で有効化する。

### Q3: ジョブ間でファイルを共有するには？

ジョブは別々のランナー(VM)で実行されるため、ファイルシステムは共有されない。`actions/upload-artifact` と `actions/download-artifact` を使ってアーティファクト経由で受け渡す。小さな値は `outputs` で受け渡す。大量のファイルの場合はアーティファクトのアップロード・ダウンロードに時間がかかるため、可能であれば1つのジョブ内で処理を完結させることを検討する。

### Q4: サードパーティアクションは安全か？

サードパーティアクションはリポジトリのコードとシークレットにアクセスできるため、信頼性の確認が重要。対策として、(1) SHA でピン留め(タグではなく)、(2) GitHub が公式に管理するアクション(actions/*)を優先、(3) 人気度・メンテナンス状況を確認、(4) ソースコードをレビュー。Dependabot でアクションのバージョンを自動更新することも推奨。

### Q5: ローカルでワークフローをテストするには？

`act` (https://github.com/nektos/act) を使うとローカルで GitHub Actions を実行できる。`act -j test` でジョブを指定して実行可能。ただし、services や一部のアクションはサポートされていない。完全なテストにはプライベートリポジトリでのテスト実行が確実。

### Q6: `pull_request` と `pull_request_target` の違いは？

`pull_request` はPRのヘッドブランチのコンテキストで実行され、フォークからのPRではシークレットにアクセスできない(安全)。`pull_request_target` はベースブランチ(main等)のコンテキストで実行され、シークレットにアクセスできる。フォークPRで `pull_request_target` + `actions/checkout@v4 ref: head.sha` を組み合わせると、信頼されないコードがシークレットにアクセスできてしまうため危険。フォークPRからのコードを実行する場合は必ず `pull_request` を使用する。

### Q7: ワークフロー内でジョブの結果に基づいて通知するには？

`if: always()` を使えば先行ジョブが失敗しても後続ジョブを実行できる。`needs.xxx.result` で先行ジョブの結果(`success`、`failure`、`cancelled`、`skipped`)を参照できるため、これを使って条件分岐し、Slack通知やメール送信を行う。例: `if: always() && needs.test.result == 'failure'`。

### Q8: 同じワークフロー内で別のワークフローをトリガーできるか？

デフォルトの `GITHUB_TOKEN` を使ったイベント(push、tag作成等)は、再帰的なワークフロー実行を防ぐために別のワークフローをトリガーしない。これを回避するには、(1) Personal Access Token (PAT)を使用する、(2) GitHub App のインストールトークンを使用する、(3) `workflow_dispatch` でAPI経由で明示的にトリガーする。セキュリティの観点からは GitHub App トークンの使用が推奨される。

### Q9: ランナーのディスク容量が不足する場合の対処法は？

GitHub-hosted ランナーのディスク容量は約14GB(usable)。不足する場合は、(1) 不要なプリインストールソフトウェアを削除(`sudo rm -rf /usr/share/dotnet /opt/ghc`)、(2) Docker イメージのプルーニング、(3) ビルド中間成果物の削除。根本的な解決としては、Self-hosted Runner でディスク容量の大きいマシンを用意するか、Larger Runner (GitHub が提供する高スペックランナー)を使用する。

### Q10: `continue-on-error` と `if: failure()` の使い分けは？

`continue-on-error: true` はそのステップが失敗してもジョブ全体を成功とみなす。非必須のチェック(例: 実験的なリント、オプショナルなテスト)に使用する。`if: failure()` は前のステップが失敗した場合にのみ実行する条件で、エラーレポートの収集やクリーンアップ処理に使用する。`if: always()` は成功・失敗に関わらず常に実行される。

---

## 13. ワークフローコマンド

### 13.1 ログ出力の制御

```yaml
steps:
  - name: Workflow commands demo
    run: |
      # グループ化 (折りたたみ可能なログセクション)
      echo "::group::Dependencies Installation"
      npm ci
      echo "::endgroup::"

      # 警告メッセージ
      echo "::warning file=src/app.js,line=10,col=5::Deprecated API usage"

      # エラーメッセージ
      echo "::error file=src/app.js,line=20::Missing required import"

      # デバッグメッセージ (ACTIONS_STEP_DEBUG=true の場合のみ表示)
      echo "::debug::Current working directory: $(pwd)"

      # 通知メッセージ
      echo "::notice::Build completed successfully"

      # 値のマスク (ログから隠す)
      DYNAMIC_SECRET=$(some-command)
      echo "::add-mask::$DYNAMIC_SECRET"
      echo "Using secret: $DYNAMIC_SECRET"  # *** と表示される
```

### 13.2 Job Summary

```yaml
steps:
  - name: Generate job summary
    run: |
      # Job Summary にMarkdownで情報を追加
      echo "## Build Report" >> $GITHUB_STEP_SUMMARY
      echo "" >> $GITHUB_STEP_SUMMARY
      echo "| Item | Status |" >> $GITHUB_STEP_SUMMARY
      echo "|------|--------|" >> $GITHUB_STEP_SUMMARY
      echo "| Lint | :white_check_mark: Pass |" >> $GITHUB_STEP_SUMMARY
      echo "| Test | :white_check_mark: Pass |" >> $GITHUB_STEP_SUMMARY
      echo "| Build | :white_check_mark: Pass |" >> $GITHUB_STEP_SUMMARY
      echo "" >> $GITHUB_STEP_SUMMARY
      echo "**Duration**: 3m 42s" >> $GITHUB_STEP_SUMMARY
      echo "" >> $GITHUB_STEP_SUMMARY

      # コードブロック
      echo '```' >> $GITHUB_STEP_SUMMARY
      echo "Node: $(node --version)" >> $GITHUB_STEP_SUMMARY
      echo "npm: $(npm --version)" >> $GITHUB_STEP_SUMMARY
      echo '```' >> $GITHUB_STEP_SUMMARY

  - name: Coverage summary
    run: |
      # テストカバレッジの要約
      COVERAGE=$(npx jest --coverage --coverageReporters=text-summary 2>/dev/null | tail -5)
      echo "### Test Coverage" >> $GITHUB_STEP_SUMMARY
      echo '```' >> $GITHUB_STEP_SUMMARY
      echo "$COVERAGE" >> $GITHUB_STEP_SUMMARY
      echo '```' >> $GITHUB_STEP_SUMMARY
```

---

## 14. デフォルトシェルとシェル設定

```yaml
# ワークフローレベルでデフォルトシェルを設定
defaults:
  run:
    shell: bash
    working-directory: ./app

jobs:
  build:
    runs-on: ubuntu-latest
    # ジョブレベルで上書き可能
    defaults:
      run:
        working-directory: ./app/frontend
    steps:
      - uses: actions/checkout@v4

      # デフォルト: bash + ./app/frontend
      - run: npm ci

      # ステップレベルで上書き
      - run: pip install -r requirements.txt
        shell: bash
        working-directory: ./app/backend

      # Windows ランナーでの PowerShell
      # - run: Get-ChildItem
      #   shell: pwsh

      # Python スクリプトとして実行
      - run: |
          import json
          with open('package.json') as f:
              data = json.load(f)
              print(f"Version: {data['version']}")
        shell: python
```

```
利用可能なシェル:

┌─────────┬──────────────────────────────────────────────┐
│ shell   │ 説明                                          │
├─────────┼──────────────────────────────────────────────┤
│ bash    │ デフォルト (Linux/macOS)。set -eo pipefail    │
│ sh      │ POSIX互換シェル                                │
│ pwsh    │ PowerShell Core (全OS)                        │
│ python  │ Python スクリプトとして実行                      │
│ cmd     │ Windows cmd.exe                               │
│ powershell │ Windows PowerShell (レガシー)               │
└─────────┴──────────────────────────────────────────────┘

bash のデフォルト動作:
  set -e       → エラーで即座に停止
  set -o pipefail → パイプ中のエラーも検出
  ※ set -u (未定義変数エラー) は含まれないため、
    厳密にしたい場合は明示的に設定する
```

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

| 項目 | 要点 |
|---|---|
| 階層構造 | Workflow > Job > Step |
| ジョブ実行 | デフォルト並列、needs で依存関係定義 |
| ステップ実行 | 順次実行、uses(アクション)と run(コマンド) |
| トリガー | push, pull_request, schedule, workflow_dispatch 等 |
| 式 | ${{ expression }} でコンテキスト参照・条件分岐 |
| 権限 | permissions で最小権限を設定 |
| 並行制御 | concurrency で重複実行を管理 |
| データ受渡 | GITHUB_OUTPUT(ステップ間)、outputs(ジョブ間)、artifact(ファイル) |
| ランナー | ubuntu-latest が最も一般的、セルフホストも可能 |

---

## 次に読むべきガイド

- [Actions 応用](./01-actions-advanced.md) -- マトリクス、キャッシュ、シークレット
- [再利用ワークフロー](./02-reusable-workflows.md) -- DRY なワークフロー設計
- [CI レシピ集](./03-ci-recipes.md) -- 言語別の実践的 CI 設定
- [Actions セキュリティ](./04-security-actions.md) -- OIDC、依存ピン留め

---

## 参考文献

1. GitHub. "GitHub Actions Documentation." https://docs.github.com/en/actions
2. GitHub. "Workflow syntax for GitHub Actions." https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions
3. GitHub. "Events that trigger workflows." https://docs.github.com/en/actions/using-workflows/events-that-trigger-workflows
4. GitHub. "Security hardening for GitHub Actions." https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions
5. GitHub. "Using GitHub-hosted runners." https://docs.github.com/en/actions/using-github-hosted-runners
6. nektos. "act - Run your GitHub Actions locally." https://github.com/nektos/act
