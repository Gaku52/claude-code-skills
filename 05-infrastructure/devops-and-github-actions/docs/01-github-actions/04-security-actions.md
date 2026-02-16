# GitHub Actions セキュリティ

> OIDC によるシークレットレス認証、権限最小化、依存ピン留め、サプライチェーン保護で安全なCI/CDを実現する

## この章で学ぶこと

1. OIDC (OpenID Connect) を使ったクラウドプロバイダーとのシークレットレス認証を実装できる
2. 権限最小化の原則とサードパーティアクションのリスク管理を理解する
3. ソフトウェアサプライチェーン保護のベストプラクティスを習得する
4. CI 環境のハードニングとセキュリティ監査の方法を実践できる
5. セキュリティインシデント発生時の対応手順を把握する

---

## 1. OIDC によるシークレットレス認証

### 1.1 従来方式 vs OIDC

```
従来方式 (長期認証情報):
  ┌──────────┐   AWS_ACCESS_KEY_ID    ┌──────┐
  │ GitHub   │ ──────────────────── → │ AWS  │
  │ Actions  │   AWS_SECRET_KEY       │      │
  │          │   (Secretsに保存)      │      │
  └──────────┘                        └──────┘
  問題: 長期キーの漏洩リスク、ローテーション負荷

OIDC 方式 (短期トークン):
  ┌──────────┐  1. JWT発行   ┌──────────┐
  │ GitHub   │ ───────────→ │ GitHub   │
  │ Actions  │              │ OIDC     │
  │          │ ←─────────── │ Provider │
  │          │  2. JWT受取   └──────────┘
  │          │
  │          │  3. JWT提示   ┌──────────┐
  │          │ ───────────→ │ AWS STS  │
  │          │              │          │
  │          │ ←─────────── │          │
  │          │  4. 一時認証  └──────────┘
  └──────────┘   情報受取
                (15分〜1時間で失効)
```

### 1.2 OIDC トークンの構造

```json
// GitHub OIDC トークンのペイロード例
{
  "jti": "example-id",
  "sub": "repo:myorg/myrepo:ref:refs/heads/main",
  "aud": "sts.amazonaws.com",
  "ref": "refs/heads/main",
  "sha": "abc123def456",
  "repository": "myorg/myrepo",
  "repository_owner": "myorg",
  "actor": "username",
  "workflow": "deploy",
  "event_name": "push",
  "ref_type": "branch",
  "job_workflow_ref": "myorg/myrepo/.github/workflows/deploy.yml@refs/heads/main",
  "runner_environment": "github-hosted",
  "iss": "https://token.actions.githubusercontent.com",
  "nbf": 1700000000,
  "exp": 1700003600,
  "iat": 1700000000
}
```

このトークンの `sub` (Subject) クレームが重要で、IAM ロールの信頼ポリシーでどのリポジトリ・ブランチからのアクセスを許可するかを制御する。

### 1.3 AWS OIDC 設定

```yaml
# OIDC を使った AWS 認証
name: Deploy to AWS
on:
  push:
    branches: [main]

permissions:
  id-token: write   # OIDC トークン発行に必須
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-role
          aws-region: ap-northeast-1
          # シークレット不要! OIDC で一時認証情報を取得

      - run: aws s3 ls  # 一時認証情報で AWS API を利用
```

```hcl
# AWS 側の IAM ロール設定 (Terraform)
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_actions" {
  name = "github-actions-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Federated = aws_iam_openid_connect_provider.github.arn
      }
      Action = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = {
          "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
        }
        StringLike = {
          # リポジトリとブランチを制限
          "token.actions.githubusercontent.com:sub" = "repo:myorg/myrepo:ref:refs/heads/main"
        }
      }
    }]
  })
}

# 必要最小限の権限をポリシーで付与
resource "aws_iam_role_policy" "github_actions_deploy" {
  name = "github-actions-deploy"
  role = aws_iam_role.github_actions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::my-deploy-bucket",
          "arn:aws:s3:::my-deploy-bucket/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudfront:CreateInvalidation"
        ]
        Resource = [
          "arn:aws:cloudfront::123456789012:distribution/E1234567890"
        ]
      }
    ]
  })
}
```

### 1.4 GCP OIDC 設定

```yaml
# GCP Workload Identity Federation
- uses: google-github-actions/auth@v2
  with:
    workload_identity_provider: 'projects/123456/locations/global/workloadIdentityPools/github/providers/github-actions'
    service_account: 'github-actions@my-project.iam.gserviceaccount.com'
```

```hcl
# GCP 側の設定 (Terraform)
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Actions"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }

  attribute_condition = "assertion.repository == 'myorg/myrepo'"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_binding" "github_actions" {
  service_account_id = google_service_account.github_actions.name
  role               = "roles/iam.workloadIdentityUser"

  members = [
    "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/myorg/myrepo"
  ]
}
```

### 1.5 Azure OIDC 設定

```yaml
# Azure OIDC 認証
- uses: azure/login@v2
  with:
    client-id: ${{ secrets.AZURE_CLIENT_ID }}
    tenant-id: ${{ secrets.AZURE_TENANT_ID }}
    subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}
    # client-secret は不要! Federated Credential で認証
```

```bash
# Azure CLI でフェデレーテッド認証情報を設定
az ad app federated-credential create \
  --id <application-object-id> \
  --parameters '{
    "name": "github-actions-main",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:myorg/myrepo:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

### 1.6 OIDC のトラブルシューティング

```yaml
# OIDC トークンのデバッグ
- name: Debug OIDC token
  run: |
    # トークンを取得
    TOKEN=$(curl -s -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
      "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=sts.amazonaws.com" | jq -r '.value')

    # ペイロードを確認（トークン自体はログに出さない）
    echo "$TOKEN" | cut -d '.' -f 2 | base64 -d 2>/dev/null | jq .

    # sub クレームの確認
    SUB=$(echo "$TOKEN" | cut -d '.' -f 2 | base64 -d 2>/dev/null | jq -r '.sub')
    echo "Subject claim: $SUB"
```

```
よくあるエラーと対処法:

1. "Not authorized to perform sts:AssumeRoleWithWebIdentity"
   原因: IAM ロールの信頼ポリシーの sub 条件が一致しない
   対処: OIDC トークンの sub クレームと IAM ポリシーの条件を照合する

2. "id-token: write permission is required"
   原因: permissions に id-token: write がない
   対処: ワークフローまたはジョブレベルで permissions を設定

3. "The audience is not valid"
   原因: OIDC プロバイダーの client_id_list が不一致
   対処: AWS は "sts.amazonaws.com"、GCP は設定した audience を確認

4. "Token is expired"
   原因: OIDC トークンの有効期限切れ
   対処: トークン取得直後にクラウド認証を行う
```

---

## 2. 権限最小化

### 2.1 permissions の設定

```yaml
# ワークフローレベルで全権限を無効化し、ジョブレベルで必要最小限を付与
permissions: {}  # デフォルト全無効

jobs:
  test:
    runs-on: ubuntu-latest
    permissions:
      contents: read  # チェックアウトに必要
    steps:
      - uses: actions/checkout@v4
      - run: npm test

  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write     # OIDC
      packages: write     # コンテナレジストリ
    steps:
      - uses: actions/checkout@v4
      # ...
```

### 2.2 権限一覧と用途

```
permissions マトリクス:

  権限名            read         write         用途
  ─────────────────────────────────────────────────────
  contents          checkout     commit/push
  pull-requests     PR情報読取    コメント投稿
  issues            Issue読取     Issue操作
  packages          パッケージ読取 パッケージ公開
  id-token          -            OIDC トークン
  actions           実行状態読取  キャッシュ操作
  security-events   -            CodeQL結果投稿
  deployments       状態読取      デプロイ状態更新
  statuses          状態読取      コミットステータス
  checks            チェック読取   チェック作成/更新
  attestations      -            アテステーション作成
  pages             -            Pages デプロイ
```

### 2.3 GITHUB_TOKEN の権限制御

```yaml
# リポジトリ全体のデフォルト設定
# Settings → Actions → General → Workflow permissions
# → "Read repository contents and packages permissions" を選択

# ワークフローごとのオーバーライド
name: Minimal Permissions Example
on: [push]

# ワークフローレベルで全無効化
permissions: {}

jobs:
  # ジョブごとに必要な権限のみ付与
  lint-and-test:
    permissions:
      contents: read
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm test

  comment-pr:
    if: github.event_name == 'pull_request'
    permissions:
      contents: read
      pull-requests: write
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: peter-evans/create-or-update-comment@v4
        with:
          issue-number: ${{ github.event.pull_request.number }}
          body: "CI passed!"

  publish-package:
    permissions:
      contents: read
      packages: write
      id-token: write
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # ...
```

### 2.4 GitHub App トークンの活用

```yaml
# GITHUB_TOKEN の代わりに GitHub App トークンを使用
# → より細かい権限制御と、他リポジトリへのアクセスが可能

name: Cross-repo operations
on: [push]

jobs:
  update-other-repo:
    runs-on: ubuntu-latest
    steps:
      - name: Generate GitHub App token
        id: app-token
        uses: actions/create-github-app-token@v1
        with:
          app-id: ${{ vars.APP_ID }}
          private-key: ${{ secrets.APP_PRIVATE_KEY }}
          owner: ${{ github.repository_owner }}
          repositories: "other-repo"

      - uses: actions/checkout@v4
        with:
          repository: myorg/other-repo
          token: ${{ steps.app-token.outputs.token }}

      - run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          echo "Updated at $(date)" >> updates.log
          git add . && git commit -m "Auto update" && git push
```

---

## 3. 依存ピン留め

### 3.1 コミットSHAによるアクション固定

```yaml
# 悪い例: タグ参照 → タグが上書きされるリスク
- uses: actions/checkout@v4          # v4 タグが悪意あるコードに書き換え可能

# 良い例: コミットSHAで完全固定
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1
  # SHA は改ざん不可能

# Dependabot でSHA参照も自動更新可能
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    # SHA 参照の場合もバージョンコメントがあれば自動更新される
```

### 3.2 アクション許可リスト

```yaml
# 組織レベルで許可するアクションを制限
# GitHub Organization Settings → Actions → General

# allowed-actions.txt (ドキュメントとして管理)
# 公式アクション:
#   actions/checkout
#   actions/setup-node
#   actions/cache
#   actions/upload-artifact
#   actions/download-artifact
#   actions/create-github-app-token
#
# 信頼済みサードパーティ:
#   docker/build-push-action
#   docker/login-action
#   docker/metadata-action
#   docker/setup-buildx-action
#   aws-actions/configure-aws-credentials
#   google-github-actions/auth
#   azure/login
#   softprops/action-gh-release
#   peter-evans/create-or-update-comment
```

### 3.3 Dependabot 設定の詳細

```yaml
# .github/dependabot.yml
version: 2
updates:
  # GitHub Actions の依存関係
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "Asia/Tokyo"
    # セキュリティアップデートは即座に
    open-pull-requests-limit: 10
    reviewers:
      - "devops-team"
    labels:
      - "dependencies"
      - "github-actions"

  # npm の依存関係
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      # マイナー/パッチ更新をグループ化
      minor-and-patch:
        patterns:
          - "*"
        update-types:
          - "minor"
          - "patch"
    ignore:
      # メジャーバージョンアップは手動対応
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]

  # Docker イメージの依存関係
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
```

### 3.4 Renovate による高度な依存管理

```json
// renovate.json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": [
    "config:recommended",
    ":pinAllExceptPeerDependencies",
    "group:allNonMajor"
  ],
  "github-actions": {
    "enabled": true,
    "pinDigests": true
  },
  "packageRules": [
    {
      "description": "Auto-merge non-major updates",
      "matchUpdateTypes": ["minor", "patch", "pin", "digest"],
      "automerge": true,
      "automergeType": "pr",
      "platformAutomerge": true
    },
    {
      "description": "Group GitHub Actions updates",
      "matchManagers": ["github-actions"],
      "groupName": "GitHub Actions",
      "schedule": ["before 9am on Monday"]
    }
  ]
}
```

---

## 4. サプライチェーン保護

### 4.1 脅威モデル

```
ソフトウェアサプライチェーンの脅威:

  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ ソース   │ →  │ ビルド    │ →  │ パッケージ │ →  │ デプロイ  │
  │ コード   │    │ システム  │    │ レジストリ │    │ 環境     │
  └────┬────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
       │              │               │               │
  ┌────┴────┐    ┌────┴─────┐    ┌────┴─────┐    ┌────┴─────┐
  │脅威:     │    │脅威:      │    │脅威:      │    │脅威:      │
  │依存汚染  │    │ビルド改竄  │    │パッケージ  │    │設定ミス   │
  │コード注入 │    │CI侵害    │    │すり替え    │    │権限過剰   │
  └─────────┘    └──────────┘    └──────────┘    └──────────┘

  対策:
  1. 依存関係のロック・監査
  2. ビルドの再現性・署名
  3. イメージの署名・検証
  4. 最小権限・ネットワーク制限
```

### 4.2 SLSA フレームワーク

```
SLSA (Supply-chain Levels for Software Artifacts) レベル:

Level 0: 保護なし
Level 1: ビルドプロセスの文書化、来歴情報の生成
Level 2: ホスティングされたビルドサービスの使用、来歴情報の署名
Level 3: 隔離されたビルド環境、改ざん防止された来歴情報
Level 4: 2者レビュー、再現可能なビルド（将来）
```

```yaml
# SLSA Level 3 準拠の例
name: Build with Provenance
on:
  push:
    tags: ['v*']

permissions:
  contents: read
  id-token: write
  packages: write
  attestations: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11

      - uses: docker/build-push-action@v5
        id: build
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}

      # ビルドのアテステーション(証明書)を生成
      - uses: actions/attest-build-provenance@v1
        with:
          subject-name: ghcr.io/${{ github.repository }}
          subject-digest: ${{ steps.build.outputs.digest }}
```

### 4.3 SBOM (Software Bill of Materials) の生成

```yaml
# SBOM を生成してアテステーションとして添付
name: SBOM Generation
on:
  push:
    tags: ['v*']

permissions:
  contents: read
  packages: write
  id-token: write
  attestations: write

jobs:
  sbom:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t myapp:latest .

      # Syft で SBOM を生成
      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          image: myapp:latest
          format: spdx-json
          output-file: sbom.spdx.json

      - name: Upload SBOM
        uses: actions/upload-artifact@v4
        with:
          name: sbom
          path: sbom.spdx.json

      # SBOM をアテステーションとして添付
      - uses: actions/attest-sbom@v1
        with:
          subject-name: ghcr.io/${{ github.repository }}
          subject-digest: ${{ steps.build.outputs.digest }}
          sbom-path: sbom.spdx.json
```

### 4.4 CodeQL セキュリティスキャン

```yaml
name: CodeQL Analysis
on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 6 * * 1'  # 週次スキャン

permissions:
  security-events: write
  contents: read

jobs:
  analyze:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        language: ['javascript', 'python']
    steps:
      - uses: actions/checkout@v4

      - uses: github/codeql-action/init@v3
        with:
          languages: ${{ matrix.language }}
          # カスタムクエリパックの使用
          queries: +security-and-quality

      - uses: github/codeql-action/analyze@v3
        with:
          category: "/language:${{ matrix.language }}"
```

### 4.5 コンテナイメージの署名と検証

```yaml
# cosign を使ったイメージ署名
name: Sign Container Image
on:
  push:
    tags: ['v*']

permissions:
  contents: read
  packages: write
  id-token: write

jobs:
  sign:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v5
        id: build
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.ref_name }}

      - name: Install cosign
        uses: sigstore/cosign-installer@v3

      - name: Sign image with cosign (keyless)
        run: |
          cosign sign --yes \
            ghcr.io/${{ github.repository }}@${{ steps.build.outputs.digest }}
        env:
          COSIGN_EXPERIMENTAL: 1

      # 検証コマンド（デプロイ時に実行）
      # cosign verify ghcr.io/myorg/myapp@sha256:... \
      #   --certificate-identity-regexp='https://github.com/myorg/myrepo/' \
      #   --certificate-oidc-issuer='https://token.actions.githubusercontent.com'
```

---

## 5. CI 環境のハードニング

### 5.1 ネットワーク制限

```yaml
# セルフホステッドランナーのネットワーク制限
jobs:
  build:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4

      # ネットワークアクセスを必要最小限に
      - name: Build (ネットワーク不要)
        run: |
          # キャッシュ済み依存関係でオフラインビルド
          npm ci --prefer-offline
          npm run build
```

### 5.2 Immutable ランナー

```yaml
# エフェメラル(使い捨て)ランナーの使用
# 各ジョブ実行後にランナーをクリーンアップ
jobs:
  build:
    runs-on: ubuntu-latest  # GitHub ホステッド = 毎回クリーンなVM
    # セルフホステッドの場合:
    # runs-on: [self-hosted, ephemeral]
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm test
      # ジョブ終了後、VMは破棄される → 残留データのリスクなし
```

### 5.3 StepSecurity Harden Runner

```yaml
# StepSecurity の Harden Runner でランナーの活動を監視・制限
name: Hardened Build
on: [push]

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: step-security/harden-runner@v2
        with:
          egress-policy: audit
          # 本番運用時は block に切り替え
          # egress-policy: block
          allowed-endpoints: >
            github.com:443
            api.github.com:443
            registry.npmjs.org:443
            objects.githubusercontent.com:443

      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test
```

### 5.4 セキュリティ監査ワークフロー

```yaml
# 定期的なセキュリティ監査
name: Security Audit
on:
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜 9:00 UTC
  workflow_dispatch:

permissions:
  contents: read
  security-events: write
  issues: write

jobs:
  dependency-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: npm audit
        run: |
          npm ci
          npm audit --json > audit-report.json || true

          # Critical/High の脆弱性があれば Issue を作成
          CRITICAL=$(jq '.metadata.vulnerabilities.critical' audit-report.json)
          HIGH=$(jq '.metadata.vulnerabilities.high' audit-report.json)

          if [ "$CRITICAL" -gt 0 ] || [ "$HIGH" -gt 0 ]; then
            echo "::warning::Critical: $CRITICAL, High: $HIGH vulnerabilities found"
          fi

      - name: Upload audit report
        uses: actions/upload-artifact@v4
        with:
          name: security-audit
          path: audit-report.json

  actions-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Audit GitHub Actions versions
        run: |
          echo "## GitHub Actions バージョン監査" > actions-audit.md
          echo "" >> actions-audit.md
          echo "| ファイル | アクション | 参照方法 | 状態 |" >> actions-audit.md
          echo "|---------|---------|---------|------|" >> actions-audit.md

          # SHA 参照でないアクションを検出
          for file in .github/workflows/*.yml; do
            grep -n 'uses:' "$file" | while read -r line; do
              if echo "$line" | grep -qP '@[a-f0-9]{40}'; then
                echo "| $file | $(echo $line | grep -oP 'uses: \K[^@]+') | SHA | OK |" >> actions-audit.md
              elif echo "$line" | grep -qP '@v\d+'; then
                echo "| $file | $(echo $line | grep -oP 'uses: \K[^@]+') | Tag | 要改善 |" >> actions-audit.md
              fi
            done
          done

          cat actions-audit.md

      - name: Upload audit report
        uses: actions/upload-artifact@v4
        with:
          name: actions-audit
          path: actions-audit.md
```

---

## 6. セキュリティインシデント対応

### 6.1 シークレット漏洩時の対応手順

```
シークレット漏洩時の即時対応:

1. 漏洩したシークレットの即時無効化
   - AWS: IAM コンソールでアクセスキーを無効化
   - GitHub: Settings → Secrets → 対象シークレットを更新
   - npm: トークンを revoke

2. 影響範囲の調査
   - CloudTrail / 監査ログで不正アクセスを確認
   - Git ログで漏洩箇所と期間を特定
   - 漏洩したキーでアクセス可能だったリソースを列挙

3. 新しいシークレットの発行と設定
   - OIDC への移行を検討（長期キーの場合）
   - 新しいシークレットを生成して GitHub Secrets に設定
   - 全ワークフローの動作確認

4. 再発防止策
   - git-secrets / gitleaks の pre-commit フック導入
   - Secret scanning alerts の有効化
   - シークレットの定期ローテーション設定
```

### 6.2 アクションの侵害検知

```yaml
# サードパーティアクションの改ざん検知
name: Action Integrity Check
on:
  pull_request:
    paths:
      - '.github/workflows/**'

jobs:
  check-actions:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check for non-pinned actions
        run: |
          ISSUES=0
          for file in .github/workflows/*.yml; do
            while IFS= read -r line; do
              if echo "$line" | grep -qP 'uses:\s+\S+@(?!([a-f0-9]{40}|[a-f0-9]{7}))'; then
                echo "::warning file=$file::Non-SHA pinned action: $line"
                ISSUES=$((ISSUES + 1))
              fi
            done < <(grep 'uses:' "$file" | grep -v '#' | grep -v './')
          done

          if [ "$ISSUES" -gt 0 ]; then
            echo "::error::Found $ISSUES non-SHA-pinned actions. Pin all actions to SHA."
            exit 1
          fi
```

### 6.3 Secret Scanning の設定

```yaml
# リポジトリ設定で Secret Scanning を有効化
# Settings → Code security and analysis → Secret scanning → Enable

# カスタムパターンの追加例
# Settings → Code security → Secret scanning → Custom patterns

# .github/secret_scanning.yml
# シークレットスキャンの除外設定
paths-ignore:
  - "tests/fixtures/**"
  - "docs/examples/**"
```

### 6.4 gitleaks による pre-commit チェック

```yaml
# CI での gitleaks チェック
name: Secret Detection
on:
  pull_request:

jobs:
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

```toml
# .gitleaks.toml — gitleaks 設定
title = "Custom Gitleaks Configuration"

# カスタムルール
[[rules]]
description = "Internal API Key"
regex = '''INTERNAL_API_KEY_[A-Za-z0-9]{32}'''
tags = ["key", "internal"]

# 除外パス
[allowlist]
paths = [
  '''tests/fixtures/''',
  '''\.github/workflows/''',
]

# 除外パターン
regexes = [
  '''EXAMPLE_KEY_[A-Za-z0-9]+''',
]
```

---

## 7. 比較表

### 7.1 認証方式比較

| 方式 | セキュリティ | 運用負荷 | 対応クラウド | 推奨度 |
|---|---|---|---|---|
| 長期アクセスキー | 低(漏洩リスク大) | 高(ローテーション) | 全て | 非推奨 |
| OIDC | 高(短期トークン) | 低(自動) | AWS/GCP/Azure | 強く推奨 |
| GitHub App | 高(スコープ制限) | 中 | GitHub API | API操作時推奨 |
| GITHUB_TOKEN | 中(自動生成) | なし | GitHub API | デフォルト |

### 7.2 アクション参照方式比較

| 方式 | 例 | セキュリティ | 運用性 |
|---|---|---|---|
| ブランチ参照 | `@main` | 最低(常に変化) | 高(常に最新) |
| タグ参照 | `@v4` | 低(上書き可能) | 高 |
| マイナータグ | `@v4.1.1` | 中 | 中 |
| コミットSHA | `@b4ffde...` | 最高(不変) | 低(手動更新) |
| SHA + Dependabot | SHA + 自動PR | 最高 | 高 |

### 7.3 セキュリティツール比較

| ツール | 対象 | 方式 | CI 統合 | コスト |
|---|---|---|---|---|
| CodeQL | ソースコード | SAST | actions/codeql | 無料(公開リポ) |
| Trivy | コンテナ/IaC | SCA/SAST | aquasecurity/trivy-action | 無料 |
| Grype | コンテナ | SCA | anchore/scan-action | 無料 |
| gitleaks | Git 履歴 | シークレット検出 | gitleaks/gitleaks-action | 無料 |
| Snyk | 依存関係 | SCA | snyk/actions | Freemium |
| Harden Runner | CI 環境 | ランタイム保護 | step-security/harden-runner | Freemium |

---

## 8. アンチパターン

### アンチパターン1: Fork PR での secrets 露出

```yaml
# 悪い例: Fork PR で secrets が利用可能
on:
  pull_request_target:  # ← 危険! Fork PR でも secrets が利用可能
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}  # Fork のコードを実行
      - run: echo "${{ secrets.DEPLOY_KEY }}"  # 漏洩!

# 改善: pull_request イベントを使い、Fork PR では secrets を使わない
on:
  pull_request:  # Fork PR では secrets は空になる(安全)
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm test  # secrets 不要の処理のみ
```

### アンチパターン2: スクリプトインジェクション

```yaml
# 悪い例: ユーザー入力を直接実行
steps:
  - run: echo "PR title: ${{ github.event.pull_request.title }}"
    # PRタイトルに "; rm -rf /" を含めるとコマンド注入

# 改善: 環境変数経由で渡す
steps:
  - run: echo "PR title: $PR_TITLE"
    env:
      PR_TITLE: ${{ github.event.pull_request.title }}
    # 環境変数として渡せばシェルインジェクションを防止
```

### アンチパターン3: 過剰な permissions

```yaml
# 悪い例: 全権限を付与
permissions: write-all
# → 不要な権限まで付与され、侵害時の影響範囲が最大化

# 改善: 必要最小限の権限のみ
permissions:
  contents: read
  packages: write
```

### アンチパターン4: シークレットのログ出力

```yaml
# 悪い例: デバッグ時にシークレットを出力
- run: |
    echo "Debug: ${{ secrets.API_KEY }}"
    curl -v -H "Authorization: Bearer ${{ secrets.API_KEY }}" $URL
    # -v オプションでヘッダーが表示される → シークレット露出

# 改善: 環境変数を使い、デバッグ出力を制限
- run: |
    curl -s -o response.json -w "%{http_code}" \
      -H "Authorization: Bearer $API_KEY" "$URL"
    echo "HTTP Status: $(cat response.json | jq -r '.status')"
  env:
    API_KEY: ${{ secrets.API_KEY }}
    URL: ${{ vars.API_URL }}
```

### アンチパターン5: Dependabot の無効化

```
悪い例:
  「Dependabot の PR が多すぎて邪魔だから無効化した」
  → セキュリティパッチの適用が遅れ、脆弱性が放置される

改善:
  1. グループ化設定で PR 数を削減
  2. セキュリティアップデートは必ず有効に
  3. 自動マージ設定で patch/minor は自動適用
  4. 週次でまとめて確認するルーティンを設定
```

---

## 9. FAQ

### Q1: OIDC と長期アクセスキー、どちらを使うべきか？

可能な限り OIDC を使うべきである。長期アクセスキーは漏洩リスクがあり、ローテーションの運用負荷もある。AWS、GCP、Azure は全て GitHub Actions の OIDC に対応している。既存のキーは段階的に OIDC に移行し、移行後は Secrets から削除する。

### Q2: Dependabot とRenovate のどちらを使うべきか？

GitHub エコシステムに統合されている Dependabot が手軽。Renovate はより柔軟な設定(自動マージ、グループ化、スケジュール)が可能で、大規模プロジェクトや Monorepo に向く。github-actions のエコシステム更新は Dependabot で十分。

### Q3: GITHUB_TOKEN の権限はどう決めるか？

デフォルトでは read-all。ワークフローレベルで `permissions: {}` として全無効化し、ジョブごとに必要な権限のみを明示的に付与する。`contents: read` は checkout に、`pull-requests: write` は PR コメントに、`id-token: write` は OIDC に必要。

### Q4: pull_request_target はいつ使うべきか？

Fork PR からのコード実行が不要で、かつ PR のメタデータ（ラベル付け、コメント投稿など）のみを操作する場合に使用する。Fork PR のコードを checkout して実行するのは絶対に避けるべき。コードの検証が必要な場合は `pull_request` イベントを使い、secrets 不要の処理のみ実行する。

### Q5: セルフホステッドランナーと GitHub ホステッドランナーのセキュリティ差は？

GitHub ホステッドランナーは毎回クリーンな VM で実行されるため、前回のジョブの残留データがない。セルフホステッドランナーは永続的なため、環境変数やファイルシステムにデータが残留するリスクがある。セルフホステッドを使う場合は、エフェメラル（使い捨て）モードの設定を強く推奨する。

### Q6: コンテナイメージの署名は必須か？

パブリックに公開するイメージや本番環境にデプロイするイメージでは強く推奨する。cosign の Keyless Signing（Sigstore/Fulcio による OIDC ベースの署名）を使えば、鍵管理が不要でシンプルに実装できる。デプロイパイプラインで署名の検証を行えば、改ざんされたイメージの実行を防止できる。

### Q7: Secret Scanning は有料プランでないと使えないか？

パブリックリポジトリでは無料で利用可能。プライベートリポジトリでは GitHub Advanced Security (GHAS) ライセンスが必要。GHAS が使えない場合は、gitleaks を CI に組み込むことで同等の機能を実現できる。

---

## 10. セキュリティチェックリストとコンプライアンス

### 10.1 CI/CD セキュリティ成熟度チェックリスト

```
Level 1: 基本的なセキュリティ
  [  ] permissions を全ワークフローで明示的に設定している
  [  ] サードパーティアクションをコミット SHA でピン留めしている
  [  ] GITHUB_TOKEN のデフォルト権限を read-only に設定している
  [  ] シークレットをログに出力するステップがない

Level 2: 認証と依存管理
  [  ] OIDC でクラウド認証を行っている（長期キー未使用）
  [  ] Dependabot/Renovate で依存関係を自動更新している
  [  ] npm audit / pip-audit / govulncheck を CI に組み込んでいる
  [  ] gitleaks または Secret Scanning が有効になっている

Level 3: サプライチェーン保護
  [  ] SBOM を生成してアーティファクトに含めている
  [  ] コンテナイメージに cosign で署名している
  [  ] CodeQL または Semgrep でコードスキャンを実施している
  [  ] SLSA Provenance を生成している

Level 4: 高度なハードニング
  [  ] Harden Runner でネットワーク制限を適用している
  [  ] エフェメラルランナーを使用している
  [  ] branch protection で CI 必須 + レビュー必須を設定している
  [  ] デプロイ環境に approval ゲートを設定している
  [  ] セキュリティインシデント対応手順が文書化されている
```

### 10.2 コンプライアンス対応ワークフロー

```yaml
# .github/workflows/compliance-check.yml
name: Compliance Check

on:
  push:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'  # 毎週月曜日

jobs:
  license-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check OSS Licenses
        run: |
          npx license-checker --production --onlyAllow \
            'MIT;BSD-2-Clause;BSD-3-Clause;Apache-2.0;ISC;0BSD;CC0-1.0;Unlicense' \
            --excludePackages 'some-internal-pkg@1.0.0'

      - name: Generate License Report
        run: |
          npx license-checker --production --csv > licenses.csv

      - name: Upload License Report
        uses: actions/upload-artifact@v4
        with:
          name: license-report
          path: licenses.csv

  vulnerability-audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci

      - name: Full Vulnerability Audit
        run: |
          npm audit --audit-level=critical --json > audit-report.json || true
          CRITICAL=$(jq '.metadata.vulnerabilities.critical' audit-report.json)
          HIGH=$(jq '.metadata.vulnerabilities.high' audit-report.json)
          echo "Critical: $CRITICAL, High: $HIGH"
          if [ "$CRITICAL" -gt 0 ]; then
            echo "Critical vulnerabilities found!"
            exit 1
          fi

      - name: Upload Audit Report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: audit-report
          path: audit-report.json
```

### 10.3 Workflow の権限監査スクリプト

```bash
#!/bin/bash
# scripts/audit-workflow-permissions.sh
# ワークフローファイルの権限設定を監査する

echo "=== GitHub Actions Workflow Permission Audit ==="
echo ""

WORKFLOWS_DIR=".github/workflows"
ISSUES_FOUND=0

for workflow in "$WORKFLOWS_DIR"/*.yml "$WORKFLOWS_DIR"/*.yaml; do
  [ -f "$workflow" ] || continue
  echo "Checking: $workflow"

  # トップレベル permissions の確認
  if ! grep -q "^permissions:" "$workflow"; then
    echo "  WARNING: No top-level permissions block"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
  fi

  # write-all の検出
  if grep -q "permissions: write-all" "$workflow"; then
    echo "  CRITICAL: write-all permissions detected"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
  fi

  # ピン留めされていないアクションの検出
  if grep -E "uses: [a-zA-Z].*@v[0-9]" "$workflow" | grep -v "@[0-9a-f]\{40\}" > /dev/null; then
    echo "  WARNING: Actions not pinned to commit SHA"
    grep -n -E "uses: [a-zA-Z].*@v[0-9]" "$workflow" | while read -r line; do
      echo "    $line"
    done
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
  fi

  echo ""
done

echo "=== Audit Complete ==="
echo "Issues found: $ISSUES_FOUND"
exit $ISSUES_FOUND
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| OIDC | 短期トークンでシークレットレス認証、強く推奨 |
| 権限最小化 | permissions: {} で全無効化 → ジョブごとに最小付与 |
| 依存ピン留め | コミットSHA + Dependabot が最良 |
| サプライチェーン | SLSA、CodeQL、SBOM、アテステーション |
| イメージ署名 | cosign の Keyless Signing で改ざん防止 |
| スクリプトインジェクション | ユーザー入力は環境変数経由で渡す |
| Fork PR | pull_request_target は極力避ける |
| CI ハードニング | Harden Runner、エフェメラルランナー |
| シークレット管理 | Secret Scanning + gitleaks の二重防御 |
| インシデント対応 | 漏洩時の即時無効化 → 影響調査 → 再発防止 |
| コンプライアンス | ライセンス監査・脆弱性監査を定期実行 |

---

## 次に読むべきガイド

- [GitHub Actions 基礎](./00-actions-basics.md) -- 基本に立ち返る
- [デプロイ戦略](../02-deployment/00-deployment-strategies.md) -- 安全なデプロイ手法
- [リリース管理](../02-deployment/03-release-management.md) -- 署名付きリリース

---

## 参考文献

1. GitHub. "Security hardening for GitHub Actions." https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions
2. GitHub. "About security hardening with OpenID Connect." https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
3. SLSA. "Supply-chain Levels for Software Artifacts." https://slsa.dev/
4. StepSecurity. "Harden Runner." https://github.com/step-security/harden-runner
5. Sigstore. "Cosign - Container Signing." https://docs.sigstore.dev/cosign/overview/
6. GitHub. "Secret scanning." https://docs.github.com/en/code-security/secret-scanning
