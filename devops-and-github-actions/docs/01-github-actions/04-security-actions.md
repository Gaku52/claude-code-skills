# GitHub Actions セキュリティ

> OIDC によるシークレットレス認証、権限最小化、依存ピン留め、サプライチェーン保護で安全なCI/CDを実現する

## この章で学ぶこと

1. OIDC (OpenID Connect) を使ったクラウドプロバイダーとのシークレットレス認証を実装できる
2. 権限最小化の原則とサードパーティアクションのリスク管理を理解する
3. ソフトウェアサプライチェーン保護のベストプラクティスを習得する

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

### 1.2 AWS OIDC 設定

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
```

### 1.3 GCP OIDC 設定

```yaml
- uses: google-github-actions/auth@v2
  with:
    workload_identity_provider: 'projects/123456/locations/global/workloadIdentityPools/github/providers/github-actions'
    service_account: 'github-actions@my-project.iam.gserviceaccount.com'
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
#
# 信頼済みサードパーティ:
#   docker/build-push-action
#   aws-actions/configure-aws-credentials
#   google-github-actions/auth
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

```yaml
# SLSA (Supply-chain Levels for Software Artifacts) Level 3 準拠の例
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

### 4.3 CodeQL セキュリティスキャン

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

      - uses: github/codeql-action/analyze@v3
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

---

## 6. 比較表

### 6.1 認証方式比較

| 方式 | セキュリティ | 運用負荷 | 対応クラウド | 推奨度 |
|---|---|---|---|---|
| 長期アクセスキー | 低(漏洩リスク大) | 高(ローテーション) | 全て | 非推奨 |
| OIDC | 高(短期トークン) | 低(自動) | AWS/GCP/Azure | 強く推奨 |
| GitHub App | 高(スコープ制限) | 中 | GitHub API | API操作時推奨 |
| GITHUB_TOKEN | 中(自動生成) | なし | GitHub API | デフォルト |

### 6.2 アクション参照方式比較

| 方式 | 例 | セキュリティ | 運用性 |
|---|---|---|---|
| ブランチ参照 | `@main` | 最低(常に変化) | 高(常に最新) |
| タグ参照 | `@v4` | 低(上書き可能) | 高 |
| メジャータグ | `@v4.1.1` | 中 | 中 |
| コミットSHA | `@b4ffde...` | 最高(不変) | 低(手動更新) |
| SHA + Dependabot | SHA + 自動PR | 最高 | 高 |

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: OIDC と長期アクセスキー、どちらを使うべきか？

可能な限り OIDC を使うべきである。長期アクセスキーは漏洩リスクがあり、ローテーションの運用負荷もある。AWS、GCP、Azure は全て GitHub Actions の OIDC に対応している。既存のキーは段階的に OIDC に移行し、移行後は Secrets から削除する。

### Q2: Dependabot とRenovate のどちらを使うべきか？

GitHub エコシステムに統合されている Dependabot が手軽。Renovate はより柔軟な設定(自動マージ、グループ化、スケジュール)が可能で、大規模プロジェクトや Monorepo に向く。github-actions のエコシステム更新は Dependabot で十分。

### Q3: GITHUB_TOKEN の権限はどう決めるか？

デフォルトでは read-all。ワークフローレベルで `permissions: {}` として全無効化し、ジョブごとに必要な権限のみを明示的に付与する。`contents: read` は checkout に、`pull-requests: write` は PR コメントに、`id-token: write` は OIDC に必要。

---

## まとめ

| 項目 | 要点 |
|---|---|
| OIDC | 短期トークンでシークレットレス認証、強く推奨 |
| 権限最小化 | permissions: {} で全無効化 → ジョブごとに最小付与 |
| 依存ピン留め | コミットSHA + Dependabot が最良 |
| サプライチェーン | SLSA、CodeQL、アテステーション |
| スクリプトインジェクション | ユーザー入力は環境変数経由で渡す |
| Fork PR | pull_request_target は極力避ける |

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
