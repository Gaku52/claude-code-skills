# 再利用ワークフロー

> Composite Actions と Reusable Workflows を使って DRY 原則に基づいた保守性の高いCI/CDパイプラインを設計する

## この章で学ぶこと

1. Composite Actions と Reusable Workflows の違いと使い分けを理解する
2. 再利用可能なワークフローの設計・実装・公開方法を習得する
3. 組織全体で共有するCI/CDライブラリの構築パターンを把握する
4. バージョニング戦略とメンテナンス体制を確立する
5. テスト駆動でアクションとワークフローの品質を担保する手法を学ぶ

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

### 1.2 再利用の設計原則

再利用可能なコンポーネントを設計する際には、以下の原則を念頭に置く。

```
設計原則:

1. 単一責任の原則 (SRP)
   - 1つのアクション/ワークフローは1つの明確な責任を持つ
   - 「セットアップ」「テスト」「デプロイ」を1つにまとめない

2. 入力の明確化
   - 必須パラメータと任意パラメータを明確に区別する
   - デフォルト値を適切に設定し、設定なしでも動作する状態を目指す

3. 出力の一貫性
   - 呼び出し元が利用する情報を明確に outputs で公開する
   - エラー時のメッセージフォーマットを統一する

4. バージョニング
   - セマンティックバージョニングを採用する
   - 破壊的変更はメジャーバージョンを上げる

5. ドキュメント
   - README.md に使用方法と全入力/出力の説明を記載する
   - CHANGELOG.md で変更履歴を管理する
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

### 2.4 実践的な Composite Action: Docker セットアップ

```yaml
# .github/actions/docker-setup/action.yml
name: 'Docker Build Setup'
description: 'Docker Buildx のセットアップとレジストリログインを一括実行'

inputs:
  registry:
    description: 'コンテナレジストリ URL'
    required: false
    default: 'ghcr.io'
  username:
    description: 'レジストリのユーザー名'
    required: true
  password:
    description: 'レジストリのパスワードまたはトークン'
    required: true
  platforms:
    description: 'ビルド対象プラットフォーム'
    required: false
    default: 'linux/amd64,linux/arm64'

outputs:
  builder-name:
    description: 'Buildx ビルダー名'
    value: ${{ steps.buildx.outputs.name }}

runs:
  using: 'composite'
  steps:
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
      with:
        platforms: ${{ inputs.platforms }}

    - name: Set up Docker Buildx
      id: buildx
      uses: docker/setup-buildx-action@v3
      with:
        install: true

    - name: Login to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ inputs.registry }}
        username: ${{ inputs.username }}
        password: ${{ inputs.password }}

    - name: Verify login
      shell: bash
      run: |
        echo "Logged in to ${{ inputs.registry }} as ${{ inputs.username }}"
        echo "Builder: ${{ steps.buildx.outputs.name }}"
        echo "Platforms: ${{ inputs.platforms }}"
```

### 2.5 実践的な Composite Action: Slack 通知

```yaml
# .github/actions/notify-slack/action.yml
name: 'Notify Slack'
description: 'ワークフロー結果を Slack に通知する'

inputs:
  webhook-url:
    description: 'Slack Incoming Webhook URL'
    required: true
  status:
    description: 'ジョブのステータス (success, failure, cancelled)'
    required: true
  channel:
    description: '通知先チャンネル'
    required: false
    default: '#deployments'
  mention:
    description: '失敗時にメンションするグループ'
    required: false
    default: ''
  custom-message:
    description: 'カスタムメッセージ（省略時は自動生成）'
    required: false
    default: ''

runs:
  using: 'composite'
  steps:
    - name: Determine emoji and color
      id: style
      shell: bash
      run: |
        case "${{ inputs.status }}" in
          success)
            echo "emoji=:white_check_mark:" >> "$GITHUB_OUTPUT"
            echo "color=#36a64f" >> "$GITHUB_OUTPUT"
            echo "text=成功" >> "$GITHUB_OUTPUT"
            ;;
          failure)
            echo "emoji=:x:" >> "$GITHUB_OUTPUT"
            echo "color=#dc3545" >> "$GITHUB_OUTPUT"
            echo "text=失敗" >> "$GITHUB_OUTPUT"
            ;;
          cancelled)
            echo "emoji=:warning:" >> "$GITHUB_OUTPUT"
            echo "color=#ffc107" >> "$GITHUB_OUTPUT"
            echo "text=キャンセル" >> "$GITHUB_OUTPUT"
            ;;
        esac

    - name: Build message
      id: message
      shell: bash
      run: |
        if [ -n "${{ inputs.custom-message }}" ]; then
          MSG="${{ inputs.custom-message }}"
        else
          MSG="${{ steps.style.outputs.emoji }} *${{ github.workflow }}* が ${{ steps.style.outputs.text }} しました"
        fi

        MENTION=""
        if [ "${{ inputs.status }}" = "failure" ] && [ -n "${{ inputs.mention }}" ]; then
          MENTION="\n<!subteam^${{ inputs.mention }}> 対応をお願いします"
        fi

        echo "body=${MSG}${MENTION}" >> "$GITHUB_OUTPUT"

    - name: Send Slack notification
      shell: bash
      env:
        WEBHOOK_URL: ${{ inputs.webhook-url }}
      run: |
        curl -s -X POST "$WEBHOOK_URL" \
          -H 'Content-Type: application/json' \
          -d '{
            "channel": "${{ inputs.channel }}",
            "attachments": [{
              "color": "${{ steps.style.outputs.color }}",
              "text": "${{ steps.message.outputs.body }}",
              "fields": [
                {"title": "リポジトリ", "value": "<${{ github.server_url }}/${{ github.repository }}|${{ github.repository }}>", "short": true},
                {"title": "ブランチ", "value": "`${{ github.ref_name }}`", "short": true},
                {"title": "コミット", "value": "<${{ github.server_url }}/${{ github.repository }}/commit/${{ github.sha }}|${{ github.sha }}>", "short": true},
                {"title": "実行者", "value": "${{ github.actor }}", "short": true}
              ],
              "footer": "GitHub Actions",
              "ts": "'$(date +%s)'"
            }]
          }'
```

### 2.6 実践的な Composite Action: PR コメント

```yaml
# .github/actions/pr-comment/action.yml
name: 'PR Comment'
description: 'PR にコメントを投稿（既存コメントがあれば更新）'

inputs:
  github-token:
    description: 'GitHub Token'
    required: true
  body:
    description: 'コメント本文（Markdown 対応）'
    required: true
  comment-tag:
    description: 'コメント識別タグ（更新時のマッチングに使用）'
    required: false
    default: 'github-actions-bot'

runs:
  using: 'composite'
  steps:
    - name: Find existing comment
      id: find
      uses: peter-evans/find-comment@v3
      with:
        issue-number: ${{ github.event.pull_request.number }}
        body-includes: "<!-- ${{ inputs.comment-tag }} -->"

    - name: Create or update comment
      uses: peter-evans/create-or-update-comment@v4
      with:
        token: ${{ inputs.github-token }}
        comment-id: ${{ steps.find.outputs.comment-id }}
        issue-number: ${{ github.event.pull_request.number }}
        body: |
          <!-- ${{ inputs.comment-tag }} -->
          ${{ inputs.body }}
        edit-mode: replace
```

### 2.7 Composite Action のデバッグテクニック

```yaml
# デバッグ用の環境変数を活用
runs:
  using: 'composite'
  steps:
    - name: Debug info
      if: runner.debug == '1'
      shell: bash
      run: |
        echo "::group::Input values"
        echo "node-version: ${{ inputs.node-version }}"
        echo "working-directory: ${{ inputs.working-directory }}"
        echo "::endgroup::"

        echo "::group::Environment"
        env | sort
        echo "::endgroup::"

    - name: Main step
      shell: bash
      run: |
        # ACTIONS_STEP_DEBUG=true の場合にのみ詳細ログ出力
        if [ "$RUNNER_DEBUG" = "1" ]; then
          set -x
        fi
        npm ci
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

### 3.4 実践的な Reusable Workflow: Docker ビルド＆プッシュ

```yaml
# .github/workflows/reusable-docker.yml
name: Reusable Docker Build

on:
  workflow_call:
    inputs:
      image-name:
        type: string
        required: true
        description: 'Docker イメージ名（例: ghcr.io/myorg/myapp）'
      dockerfile:
        type: string
        default: './Dockerfile'
        description: 'Dockerfile のパス'
      context:
        type: string
        default: '.'
        description: 'Docker ビルドコンテキスト'
      platforms:
        type: string
        default: 'linux/amd64,linux/arm64'
        description: 'ビルド対象プラットフォーム'
      push:
        type: boolean
        default: true
        description: 'レジストリにプッシュするか'
      build-args:
        type: string
        default: ''
        description: 'ビルド引数（改行区切り）'
    secrets:
      REGISTRY_TOKEN:
        required: false
        description: 'レジストリ認証トークン'
    outputs:
      image-digest:
        description: 'プッシュされたイメージのダイジェスト'
        value: ${{ jobs.build.outputs.digest }}
      image-tags:
        description: '生成されたタグ一覧'
        value: ${{ jobs.build.outputs.tags }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    outputs:
      digest: ${{ steps.build-push.outputs.digest }}
      tags: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-qemu-action@v3
        with:
          platforms: ${{ inputs.platforms }}

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        if: inputs.push
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.REGISTRY_TOKEN || secrets.GITHUB_TOKEN }}

      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ inputs.image-name }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha,prefix=

      - uses: docker/build-push-action@v5
        id: build-push
        with:
          context: ${{ inputs.context }}
          file: ${{ inputs.dockerfile }}
          push: ${{ inputs.push }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          platforms: ${{ inputs.platforms }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: ${{ inputs.build-args }}

      - name: Output summary
        run: |
          echo "## Docker Build Summary" >> "$GITHUB_STEP_SUMMARY"
          echo "| Property | Value |" >> "$GITHUB_STEP_SUMMARY"
          echo "|----------|-------|" >> "$GITHUB_STEP_SUMMARY"
          echo "| Image | ${{ inputs.image-name }} |" >> "$GITHUB_STEP_SUMMARY"
          echo "| Digest | ${{ steps.build-push.outputs.digest }} |" >> "$GITHUB_STEP_SUMMARY"
          echo "| Tags | ${{ steps.meta.outputs.tags }} |" >> "$GITHUB_STEP_SUMMARY"
```

### 3.5 実践的な Reusable Workflow: デプロイ

```yaml
# .github/workflows/reusable-deploy.yml
name: Reusable Deploy

on:
  workflow_call:
    inputs:
      environment:
        type: string
        required: true
        description: 'デプロイ先環境 (staging, production)'
      version:
        type: string
        required: true
        description: 'デプロイするバージョン'
      dry-run:
        type: boolean
        default: false
        description: 'ドライラン実行'
      rollback-version:
        type: string
        default: ''
        description: 'ロールバック先バージョン（空の場合は通常デプロイ）'
    secrets:
      AWS_ROLE_ARN:
        required: true
      SLACK_WEBHOOK_URL:
        required: false
    outputs:
      deploy-url:
        description: 'デプロイ先 URL'
        value: ${{ jobs.deploy.outputs.url }}
      deploy-status:
        description: 'デプロイ結果 (success / failure)'
        value: ${{ jobs.deploy.outputs.status }}

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    permissions:
      id-token: write
      contents: read
    outputs:
      url: ${{ steps.deploy.outputs.url }}
      status: ${{ steps.result.outputs.status }}
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: Pre-deploy validation
        run: |
          echo "Environment: ${{ inputs.environment }}"
          echo "Version: ${{ inputs.version }}"
          echo "Dry-run: ${{ inputs.dry-run }}"

          # ヘルスチェックエンドポイントの事前確認
          if [ "${{ inputs.environment }}" = "production" ]; then
            echo "::notice::本番環境へのデプロイです。承認が必要です。"
          fi

      - name: Deploy
        id: deploy
        run: |
          if [ "${{ inputs.dry-run }}" = "true" ]; then
            echo "::notice::ドライラン実行中。実際のデプロイは行いません。"
            echo "url=https://dry-run.example.com" >> "$GITHUB_OUTPUT"
          else
            # 実際のデプロイコマンド
            aws ecs update-service \
              --cluster my-cluster-${{ inputs.environment }} \
              --service my-service \
              --task-definition my-task:${{ inputs.version }} \
              --force-new-deployment

            echo "url=https://${{ inputs.environment }}.example.com" >> "$GITHUB_OUTPUT"
          fi

      - name: Wait for deployment
        if: inputs.dry-run == false
        run: |
          aws ecs wait services-stable \
            --cluster my-cluster-${{ inputs.environment }} \
            --services my-service
          echo "デプロイが安定しました"

      - name: Health check
        if: inputs.dry-run == false
        run: |
          for i in $(seq 1 5); do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
              https://${{ inputs.environment }}.example.com/health)
            if [ "$STATUS" = "200" ]; then
              echo "ヘルスチェック成功"
              exit 0
            fi
            echo "ヘルスチェック試行 $i/5: ステータス $STATUS"
            sleep 10
          done
          echo "::error::ヘルスチェックが5回連続で失敗しました"
          exit 1

      - name: Set result
        id: result
        if: always()
        run: |
          if [ "${{ job.status }}" = "success" ]; then
            echo "status=success" >> "$GITHUB_OUTPUT"
          else
            echo "status=failure" >> "$GITHUB_OUTPUT"
          fi

  notify:
    needs: deploy
    if: always() && inputs.dry-run == false
    runs-on: ubuntu-latest
    steps:
      - name: Notify Slack
        if: secrets.SLACK_WEBHOOK_URL != ''
        run: |
          STATUS="${{ needs.deploy.outputs.status }}"
          COLOR=$([ "$STATUS" = "success" ] && echo "#36a64f" || echo "#dc3545")
          EMOJI=$([ "$STATUS" = "success" ] && echo ":rocket:" || echo ":fire:")

          curl -s -X POST "${{ secrets.SLACK_WEBHOOK_URL }}" \
            -H 'Content-Type: application/json' \
            -d "{
              \"attachments\": [{
                \"color\": \"$COLOR\",
                \"text\": \"$EMOJI デプロイ $STATUS: ${{ inputs.environment }} v${{ inputs.version }}\",
                \"footer\": \"${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}\"
              }]
            }"
```

### 3.6 Reusable Workflow のマトリクス活用

```yaml
# 呼び出し元でマトリクスを使って同一ワークフローを複数パラメータで呼び出す
name: Multi-environment Deploy
on:
  workflow_dispatch:
    inputs:
      version:
        description: 'デプロイするバージョン'
        required: true

jobs:
  deploy-staging:
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: staging
      version: ${{ github.event.inputs.version }}
    secrets:
      AWS_ROLE_ARN: ${{ secrets.STAGING_AWS_ROLE_ARN }}
      SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}

  deploy-production:
    needs: deploy-staging
    if: needs.deploy-staging.outputs.deploy-status == 'success'
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: production
      version: ${{ github.event.inputs.version }}
    secrets:
      AWS_ROLE_ARN: ${{ secrets.PRODUCTION_AWS_ROLE_ARN }}
      SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### 3.7 Reusable Workflow と Environment 保護ルール

```yaml
# Reusable Workflow 内で environment を使用することで
# デプロイ前に承認フローを挟むことができる

# GitHub リポジトリ設定:
# Settings → Environments → production
#   - Required reviewers: team-lead, devops-lead
#   - Wait timer: 5 minutes
#   - Deployment branches: main のみ

# Reusable Workflow 側
on:
  workflow_call:
    inputs:
      environment:
        type: string
        required: true

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment }}
    # ↑ environment を指定すると、GitHub が保護ルールを自動適用
    # production の場合、承認者がApproveするまでジョブは開始されない
    steps:
      - uses: actions/checkout@v4
      - run: echo "Deploying to ${{ inputs.environment }}"
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
| 条件分岐 | steps の if で制御 | jobs の if で制御 |
| サービスコンテナ | 利用不可 | 利用可能 |
| 環境変数の継承 | 呼び出し元の env を継承 | 明示的に inputs で渡す |
| environment | 利用不可 | 利用可能（承認フロー対応） |
| concurrency | 利用不可 | 利用可能 |
| strategy/matrix | 利用不可 | 利用可能 |

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
  - サービスコンテナ(DB等)が必要 → Reusable Workflow
  - 複数ジョブの依存関係を含む → Reusable Workflow
  - 既存ワークフローの一部を共通化 → Composite Action
```

### 4.3 組み合わせパターン

Composite Action と Reusable Workflow は排他的ではなく、組み合わせて使うのが最も効果的である。

```yaml
# Reusable Workflow 内で Composite Action を使う
# .github/workflows/reusable-fullstack-ci.yml
name: Reusable Full-Stack CI

on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
      python-version:
        type: string
        default: '3.12'

jobs:
  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # Composite Action でフロントエンドのセットアップを共通化
      - uses: ./.github/actions/setup-and-build
        with:
          node-version: ${{ inputs.node-version }}
          working-directory: ./frontend
      # Composite Action でテスト実行を共通化
      - uses: ./.github/actions/run-tests
        with:
          test-command: 'cd frontend && npm test -- --coverage'

  backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
      - run: |
          cd backend
          pip install -r requirements.txt
          pytest --cov
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
├── __tests__/
│   └── index.test.ts   # テスト
├── package.json
├── tsconfig.json
├── LICENSE
├── README.md
└── CHANGELOG.md
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
  m-threshold:
    description: 'Mの閾値'
    default: '200'
  l-threshold:
    description: 'Lの閾値'
    default: '500'

outputs:
  label:
    description: '付与されたラベル名'
  total-changes:
    description: '変更行数の合計'

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

interface SizeConfig {
  label: string;
  threshold: number;
}

async function run(): Promise<void> {
  try {
    const token = core.getInput('github-token', { required: true });
    const xsThreshold = parseInt(core.getInput('xs-threshold'));
    const sThreshold = parseInt(core.getInput('s-threshold'));
    const mThreshold = parseInt(core.getInput('m-threshold'));
    const lThreshold = parseInt(core.getInput('l-threshold'));

    const octokit = github.getOctokit(token);
    const { context } = github;

    if (!context.payload.pull_request) {
      core.info('Not a PR event, skipping.');
      return;
    }

    const prNumber = context.payload.pull_request.number;

    const { data: pr } = await octokit.rest.pulls.get({
      ...context.repo,
      pull_number: prNumber,
    });

    const totalChanges = pr.additions + pr.deletions;
    core.setOutput('total-changes', totalChanges.toString());

    // サイズ判定
    const sizes: SizeConfig[] = [
      { label: 'size/XS', threshold: xsThreshold },
      { label: 'size/S', threshold: sThreshold },
      { label: 'size/M', threshold: mThreshold },
      { label: 'size/L', threshold: lThreshold },
    ];

    let label = 'size/XL';
    for (const size of sizes) {
      if (totalChanges < size.threshold) {
        label = size.label;
        break;
      }
    }

    // 既存のサイズラベルを削除
    const existingLabels = pr.labels
      .filter((l) => l.name?.startsWith('size/'))
      .map((l) => l.name!);

    for (const existingLabel of existingLabels) {
      if (existingLabel !== label) {
        await octokit.rest.issues.removeLabel({
          ...context.repo,
          issue_number: prNumber,
          name: existingLabel,
        });
      }
    }

    // 新しいラベルを追加
    if (!existingLabels.includes(label)) {
      await octokit.rest.issues.addLabels({
        ...context.repo,
        issue_number: prNumber,
        labels: [label],
      });
    }

    core.setOutput('label', label);
    core.info(
      `PR #${prNumber}: ${totalChanges} changes → ${label}`
    );
  } catch (error) {
    if (error instanceof Error) {
      core.setFailed(error.message);
    }
  }
}

run();
```

### 5.3 Action のテスト

```typescript
// __tests__/index.test.ts
import * as core from '@actions/core';
import * as github from '@actions/github';

// モックの設定
jest.mock('@actions/core');
jest.mock('@actions/github');

describe('PR Size Label Action', () => {
  const mockGetInput = core.getInput as jest.MockedFunction<
    typeof core.getInput
  >;
  const mockSetOutput = core.setOutput as jest.MockedFunction<
    typeof core.setOutput
  >;

  beforeEach(() => {
    jest.clearAllMocks();
    mockGetInput.mockImplementation((name: string) => {
      const inputs: Record<string, string> = {
        'github-token': 'fake-token',
        'xs-threshold': '10',
        's-threshold': '50',
        'm-threshold': '200',
        'l-threshold': '500',
      };
      return inputs[name] ?? '';
    });
  });

  it('should label XS for small changes', async () => {
    // PR のモック: 5行追加、2行削除 = 合計7行
    (github.getOctokit as jest.Mock).mockReturnValue({
      rest: {
        pulls: {
          get: jest.fn().mockResolvedValue({
            data: { additions: 5, deletions: 2, labels: [] },
          }),
        },
        issues: {
          addLabels: jest.fn().mockResolvedValue({}),
          removeLabel: jest.fn().mockResolvedValue({}),
        },
      },
    });

    // テスト実行
    // ...
    expect(mockSetOutput).toHaveBeenCalledWith('label', 'size/XS');
  });
});
```

### 5.4 Action のリリースワークフロー

```yaml
# .github/workflows/release-action.yml
name: Release Action

on:
  push:
    tags: ['v*']

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'

      - run: npm ci
      - run: npm run build
      - run: npm test

      # dist/ をコミットに含める
      - name: Update dist
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add dist/ -f
          git diff --staged --quiet || git commit -m "chore: update dist for ${{ github.ref_name }}"

      # メジャーバージョンタグの更新 (v1 → v1.2.3 を指す)
      - name: Update major version tag
        run: |
          MAJOR_VERSION=$(echo "${{ github.ref_name }}" | grep -oP 'v\d+')
          git tag -f "$MAJOR_VERSION"
          git push origin "$MAJOR_VERSION" --force
          git push origin "${{ github.ref_name }}"

      - name: Create Release
        uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
```

---

## 6. 組織共通ワークフローのパターン

### 6.1 リポジトリ構成

```
組織共通リポジトリ構成:

  my-org/shared-workflows/
  ├── .github/
  │   └── workflows/
  │       ├── node-ci.yml        # Node.js CI
  │       ├── python-ci.yml      # Python CI
  │       ├── docker-build.yml   # Docker ビルド
  │       ├── deploy-ecs.yml     # ECS デプロイ
  │       ├── deploy-lambda.yml  # Lambda デプロイ
  │       └── release.yml        # リリース管理
  ├── actions/
  │   ├── setup-node/
  │   │   └── action.yml
  │   ├── setup-python/
  │   │   └── action.yml
  │   ├── security-scan/
  │   │   └── action.yml
  │   ├── notify-slack/
  │   │   └── action.yml
  │   └── pr-comment/
  │       └── action.yml
  ├── docs/
  │   ├── MIGRATION.md          # バージョンアップガイド
  │   └── USAGE.md              # 使用方法
  ├── CHANGELOG.md
  └── README.md

  各プロジェクトリポジトリ:
  my-org/my-app/.github/workflows/ci.yml
    → uses: my-org/shared-workflows/.github/workflows/node-ci.yml@v1
```

### 6.2 共通ワークフローの段階的導入

```yaml
# Phase 1: 基本的な CI をまず共通化
# my-org/my-app/.github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  ci:
    uses: my-org/shared-workflows/.github/workflows/node-ci.yml@v1
    with:
      node-version: '20'
    secrets: inherit

---
# Phase 2: Docker ビルドも共通化
# my-org/my-app/.github/workflows/docker.yml
name: Docker
on:
  push:
    branches: [main]

jobs:
  build:
    uses: my-org/shared-workflows/.github/workflows/docker-build.yml@v1
    with:
      image-name: ghcr.io/my-org/my-app
    secrets: inherit

---
# Phase 3: デプロイまで共通化
# my-org/my-app/.github/workflows/deploy.yml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  ci:
    uses: my-org/shared-workflows/.github/workflows/node-ci.yml@v1
    secrets: inherit

  deploy-staging:
    needs: ci
    uses: my-org/shared-workflows/.github/workflows/deploy-ecs.yml@v1
    with:
      environment: staging
      version: ${{ needs.ci.outputs.build-version }}
    secrets: inherit

  deploy-production:
    needs: deploy-staging
    uses: my-org/shared-workflows/.github/workflows/deploy-ecs.yml@v1
    with:
      environment: production
      version: ${{ needs.ci.outputs.build-version }}
    secrets: inherit
```

### 6.3 共通ワークフローのバージョニング戦略

```
バージョニング方針:

  リリースタグの管理:
    v1.0.0 — 初回リリース
    v1.1.0 — 後方互換の機能追加
    v1.1.1 — バグ修正
    v2.0.0 — 破壊的変更

  メジャーバージョンタグ:
    v1 → v1.3.2 を指す（最新の v1.x.x）
    v2 → v2.1.0 を指す（最新の v2.x.x）

  利用者側の参照方法:
    安定重視: my-org/shared-workflows/.github/workflows/ci.yml@v1
    固定重視: my-org/shared-workflows/.github/workflows/ci.yml@v1.3.2
    最高固定: my-org/shared-workflows/.github/workflows/ci.yml@abc1234def

  破壊的変更時の移行手順:
    1. v2 ブランチで新バージョンを開発
    2. MIGRATION.md に移行手順を記載
    3. v2.0.0 をリリース
    4. 全チームに通知、移行期限を設定
    5. v1 のメンテナンス期間（3ヶ月）を設ける
    6. v1 を非推奨化し、最終的に削除
```

### 6.4 Required Workflows（組織全体で強制）

```
GitHub Organization の Required Workflows 機能を使うと、
組織内の全リポジトリ（または指定リポジトリ）に対して
特定のワークフローの実行を強制できる。

設定方法:
  Organization Settings → Actions → Required workflows

使用例:
  - セキュリティスキャンの強制
  - ライセンスチェックの強制
  - コーディング規約チェックの強制

注意点:
  - Required Workflow は PR のステータスチェックとして表示される
  - 失敗するとマージがブロックされる
  - 各リポジトリの maintainer はスキップ不可
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

### アンチパターン3: 入力パラメータの肥大化

```yaml
# 悪い例: 入力が多すぎて使いにくい
on:
  workflow_call:
    inputs:
      node-version: { type: string }
      npm-token: { type: string }
      lint-command: { type: string }
      test-command: { type: string }
      build-command: { type: string }
      e2e-command: { type: string }
      coverage-threshold: { type: string }
      docker-registry: { type: string }
      docker-image-name: { type: string }
      deploy-target: { type: string }
      slack-channel: { type: string }
      # ... 20個以上のパラメータ

# 改善: 責任を分割して複数の Reusable Workflow に
# reusable-ci.yml     → CI (lint, test, build) に集中
# reusable-docker.yml → Docker ビルドに集中
# reusable-deploy.yml → デプロイに集中
# 各ワークフローのパラメータは5個以下を目安にする
```

### アンチパターン4: テストなしの共通アクション公開

```yaml
# 悪い例: テストなしで共通アクションを公開
# → 全プロジェクトのCIが一斉に壊れるリスク

# 改善: アクション自体のCIを整備
# .github/workflows/test-action.yml
name: Test Action
on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # ユニットテスト
      - run: npm ci && npm test

      # 統合テスト: 実際にアクションを実行
      - uses: ./  # 自分自身をテスト
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
```

### アンチパターン5: 秘密情報のログ出力

```yaml
# 悪い例: デバッグ目的でシークレットをログに出力
runs:
  using: 'composite'
  steps:
    - shell: bash
      run: |
        echo "Token: ${{ inputs.github-token }}"  # シークレットがログに表示される！
        curl -H "Authorization: Bearer ${{ inputs.github-token }}" ...

# 改善: シークレットは環境変数経由で渡す
runs:
  using: 'composite'
  steps:
    - shell: bash
      env:
        GH_TOKEN: ${{ inputs.github-token }}
      run: |
        # GH_TOKEN はマスクされ、ログに表示されない
        curl -H "Authorization: Bearer $GH_TOKEN" ...
```

---

## 8. 高度なパターン

### 8.1 動的マトリクスと Reusable Workflow の組み合わせ

```yaml
# .github/workflows/dynamic-matrix.yml
name: Dynamic Matrix CI
on: [push]

jobs:
  determine-matrix:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
    steps:
      - uses: actions/checkout@v4
      - id: set-matrix
        run: |
          # 変更されたパッケージを検出してマトリクスを動的に生成
          CHANGED_PACKAGES=$(git diff --name-only HEAD~1 | \
            grep -oP 'packages/\K[^/]+' | sort -u | jq -R -s -c 'split("\n")[:-1]')
          echo "matrix={\"package\":$CHANGED_PACKAGES}" >> "$GITHUB_OUTPUT"

  ci:
    needs: determine-matrix
    if: needs.determine-matrix.outputs.matrix != '{"package":[]}'
    strategy:
      matrix: ${{ fromJSON(needs.determine-matrix.outputs.matrix) }}
    uses: ./.github/workflows/reusable-ci.yml
    with:
      working-directory: packages/${{ matrix.package }}
    secrets: inherit
```

### 8.2 Composite Action のチェーン

```yaml
# 複数の Composite Action を連携させるパターン
# .github/workflows/full-pipeline.yml
name: Full Pipeline
on: [push]

jobs:
  pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Step 1: セットアップとビルド
      - uses: ./.github/actions/setup-and-build
        id: build
        with:
          node-version: '20'

      # Step 2: テスト実行
      - uses: ./.github/actions/run-tests
        id: test
        with:
          coverage-threshold: '80'

      # Step 3: セキュリティスキャン
      - uses: ./.github/actions/security-scan
        id: security

      # Step 4: PRコメントで結果を報告
      - uses: ./.github/actions/pr-comment
        if: github.event_name == 'pull_request'
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          body: |
            ## CI Results
            | Check | Result |
            |-------|--------|
            | Build | ${{ steps.build.outputs.build-path && '✅' || '❌' }} |
            | Coverage | ${{ steps.test.outputs.coverage-percent }}% |
            | Security | ${{ steps.security.outputs.vulnerabilities == '0' && '✅' || '⚠️' }} |
```

### 8.3 ワークフロー間のアーティファクト共有

```yaml
# Reusable Workflow 間でアーティファクトを共有する
# .github/workflows/build-and-deploy.yml
name: Build and Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    uses: ./.github/workflows/reusable-ci.yml
    with:
      node-version: '20'
    secrets: inherit

  # ビルド結果をアーティファクト経由で次のジョブに渡す
  package:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build-output
          path: ./dist

      - name: Package
        run: |
          tar -czf app.tar.gz dist/
          echo "Packaged successfully"

      - uses: actions/upload-artifact@v4
        with:
          name: deployment-package
          path: app.tar.gz
          retention-days: 1

  deploy:
    needs: package
    uses: ./.github/workflows/reusable-deploy.yml
    with:
      environment: production
      version: ${{ needs.build.outputs.build-version }}
    secrets: inherit
```

### 8.4 条件付き Reusable Workflow 呼び出し

```yaml
# パスフィルターと組み合わせて必要なワークフローだけ実行
name: Smart CI
on:
  pull_request:

jobs:
  changes:
    runs-on: ubuntu-latest
    outputs:
      frontend: ${{ steps.filter.outputs.frontend }}
      backend: ${{ steps.filter.outputs.backend }}
      infra: ${{ steps.filter.outputs.infra }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            frontend:
              - 'frontend/**'
              - 'package.json'
            backend:
              - 'backend/**'
              - 'requirements.txt'
            infra:
              - 'terraform/**'
              - 'Dockerfile'

  frontend-ci:
    needs: changes
    if: needs.changes.outputs.frontend == 'true'
    uses: ./.github/workflows/reusable-node-ci.yml
    with:
      working-directory: frontend
    secrets: inherit

  backend-ci:
    needs: changes
    if: needs.changes.outputs.backend == 'true'
    uses: ./.github/workflows/reusable-python-ci.yml
    with:
      working-directory: backend
    secrets: inherit

  infra-check:
    needs: changes
    if: needs.changes.outputs.infra == 'true'
    uses: ./.github/workflows/reusable-terraform-plan.yml
    secrets: inherit
```

---

## 9. FAQ

### Q1: Reusable Workflow のネストは何階層まで可能か？

最大4階層まで。ただし、深いネストは可読性を大きく損なうため、2階層以内を推奨する。それ以上の共通化が必要な場合は Composite Action に切り出して、Reusable Workflow のステップ内で使う構成が良い。

### Q2: Reusable Workflow でマトリクスは使えるか？

呼び出し元でマトリクスを使って同じ Reusable Workflow を異なるパラメータで呼び出すことが可能。Reusable Workflow 内部でもマトリクスを使える。ただし、呼び出し元のマトリクスと内部のマトリクスを組み合わせると実行ジョブ数が爆発するため注意。

### Q3: secrets: inherit は安全か？

`secrets: inherit` は呼び出し元の全シークレットを渡す。便利だが、Reusable Workflow が信頼できるリポジトリにある場合のみ使うべき。外部リポジトリの Reusable Workflow には明示的に必要なシークレットだけを渡す方が安全。

### Q4: Composite Action で shell を省略するとどうなるか？

Composite Action の `run` ステップでは `shell` の指定が必須である。省略するとエラーになる。これは通常のワークフローの `run` ステップ（デフォルトが bash）と異なる点なので注意が必要。一般的には `shell: bash` を指定する。Windows ランナーを考慮する場合は `shell: pwsh` も検討する。

### Q5: Reusable Workflow の inputs に配列やオブジェクトは渡せるか？

直接の配列やオブジェクト型はサポートされていない。`type: string` として JSON 文字列を渡し、ワークフロー内で `fromJSON()` を使って変換するパターンが一般的。

```yaml
# 呼び出し元
jobs:
  ci:
    uses: ./.github/workflows/reusable-ci.yml
    with:
      environments: '["staging", "production"]'

# Reusable Workflow 内
jobs:
  deploy:
    strategy:
      matrix:
        env: ${{ fromJSON(inputs.environments) }}
```

### Q6: Composite Action の中で別の Composite Action を呼べるか？

はい、呼べる。Composite Action のステップ内で `uses:` を使って別の Action を参照できる。ただし、ネストが深くなるとデバッグが困難になるため、2階層以内に抑えることを推奨する。

### Q7: Reusable Workflow を workflow_dispatch と workflow_call の両方で使えるか？

はい、1つのワークフローファイルに両方のトリガーを定義できる。これにより、他のワークフローから呼び出すことも、手動で直接実行することも可能になる。

```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'デプロイ先'
        type: choice
        options:
          - staging
          - production
  workflow_call:
    inputs:
      environment:
        type: string
        required: true
```

### Q8: Reusable Workflow の実行ログはどこで確認できるか？

呼び出し元のワークフロー実行ログ内に、Reusable Workflow のジョブがネストされた形で表示される。各ジョブをクリックすると詳細なステップログが確認できる。`uses:` の横にリンクが表示されるため、Reusable Workflow のソースコードにもジャンプ可能。

---

## まとめ

| 項目 | 要点 |
|---|---|
| Composite Action | ステップ群をまとめて再利用、action.yml で定義 |
| Reusable Workflow | ジョブ群を再利用、workflow_call で定義 |
| 使い分け | セットアップ手順 → Composite、CIフロー → Reusable |
| 組み合わせ | Reusable Workflow 内で Composite Action を使うのが最も効果的 |
| 公開 | マーケットプレイス(Action)、リポジトリ参照(Workflow) |
| バージョニング | セマンティックバージョンかSHAで固定必須 |
| 組織パターン | shared-workflows リポジトリに集約 |
| テスト | アクション自体のCI を整備し、破壊を防止 |
| パラメータ設計 | 1つのワークフローの inputs は5個以下を目安 |
| 段階的導入 | CI → Docker → Deploy の順で共通化を進める |

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
4. GitHub. "Required workflows." https://docs.github.com/en/actions/using-workflows/required-workflows
5. GitHub. "Sharing workflows with your organization." https://docs.github.com/en/actions/using-workflows/sharing-workflows-secrets-and-runners-with-your-organization
