# AWS CodePipeline

> AWS のフルマネージド CI/CD サービスを理解し、CodeCommit・CodeBuild・CodeDeploy・GitHub を統合した自動化パイプラインを構築する

## この章で学ぶこと

1. **CodePipeline の基本概念** — ステージ、アクション、アーティファクトの構造
2. **CodeBuild と CodeDeploy の統合** — ビルド・テスト・デプロイの自動化
3. **GitHub 統合とベストプラクティス** — GitHub Actions との連携、承認ゲート、ロールバック

---

## 1. CodePipeline とは

CodePipeline は、コードの変更を検知してビルド、テスト、デプロイを自動実行する継続的デリバリーサービスである。各ステージを連結し、ソフトウェアリリースプロセスを完全に自動化する。

### 図解 1: CodePipeline の全体フロー

```
┌─────────────────────────────────────────────────────────────┐
│                    CodePipeline                             │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Source    │  │ Build    │  │ Test     │  │ Deploy   │   │
│  │          │→│          │→│          │→│          │   │
│  │ GitHub/  │  │ CodeBuild│  │ CodeBuild│  │ CodeDeploy│  │
│  │ CodeCommit│ │          │  │          │  │ /ECS/    │   │
│  │          │  │          │  │          │  │ Lambda   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│       │              │              │              │        │
│       ▼              ▼              ▼              ▼        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              S3 Artifact Store                       │  │
│  │  source.zip → build.zip → test-results → deploy.zip │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  オプションステージ:                                        │
│  ┌──────────┐  ┌──────────┐                                │
│  │ Approval │  │ Deploy   │                                │
│  │ (手動承認)│→│ (本番)   │                                │
│  └──────────┘  └──────────┘                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. パイプラインの構築

### コード例 1: AWS CLI でパイプラインを作成

```bash
# CodeCommit リポジトリの作成
aws codecommit create-repository \
  --repository-name my-app \
  --repository-description "Application repository"

# パイプライン定義 JSON を使って作成
aws codepipeline create-pipeline --cli-input-json file://pipeline.json
```

```json
{
  "pipeline": {
    "name": "my-app-pipeline",
    "roleArn": "arn:aws:iam::123456789012:role/CodePipelineRole",
    "artifactStore": {
      "type": "S3",
      "location": "my-pipeline-artifacts-bucket"
    },
    "stages": [
      {
        "name": "Source",
        "actions": [{
          "name": "GitHub-Source",
          "actionTypeId": {
            "category": "Source",
            "owner": "AWS",
            "provider": "CodeStarSourceConnection",
            "version": "1"
          },
          "configuration": {
            "ConnectionArn": "arn:aws:codestar-connections:ap-northeast-1:123456789012:connection/xxx",
            "FullRepositoryId": "my-org/my-app",
            "BranchName": "main",
            "DetectChanges": "true"
          },
          "outputArtifacts": [{"name": "SourceOutput"}]
        }]
      },
      {
        "name": "Build",
        "actions": [{
          "name": "CodeBuild",
          "actionTypeId": {
            "category": "Build",
            "owner": "AWS",
            "provider": "CodeBuild",
            "version": "1"
          },
          "configuration": {
            "ProjectName": "my-app-build"
          },
          "inputArtifacts": [{"name": "SourceOutput"}],
          "outputArtifacts": [{"name": "BuildOutput"}]
        }]
      },
      {
        "name": "Deploy-Staging",
        "actions": [{
          "name": "Deploy-ECS",
          "actionTypeId": {
            "category": "Deploy",
            "owner": "AWS",
            "provider": "ECS",
            "version": "1"
          },
          "configuration": {
            "ClusterName": "my-cluster-staging",
            "ServiceName": "my-service",
            "FileName": "imagedefinitions.json"
          },
          "inputArtifacts": [{"name": "BuildOutput"}]
        }]
      },
      {
        "name": "Approval",
        "actions": [{
          "name": "ManualApproval",
          "actionTypeId": {
            "category": "Approval",
            "owner": "AWS",
            "provider": "Manual",
            "version": "1"
          },
          "configuration": {
            "NotificationArn": "arn:aws:sns:ap-northeast-1:123456789012:deploy-approval",
            "CustomData": "ステージング環境を確認してから承認してください"
          }
        }]
      },
      {
        "name": "Deploy-Production",
        "actions": [{
          "name": "Deploy-ECS",
          "actionTypeId": {
            "category": "Deploy",
            "owner": "AWS",
            "provider": "ECS",
            "version": "1"
          },
          "configuration": {
            "ClusterName": "my-cluster-prod",
            "ServiceName": "my-service",
            "FileName": "imagedefinitions.json"
          },
          "inputArtifacts": [{"name": "BuildOutput"}]
        }]
      }
    ]
  }
}
```

---

## 3. CodeBuild

### コード例 2: buildspec.yml の作成

```yaml
# buildspec.yml
version: 0.2

env:
  variables:
    APP_NAME: my-app
  parameter-store:
    DB_PASSWORD: /myapp/prod/db-password
  secrets-manager:
    API_KEY: myapp/api-key:API_KEY

phases:
  install:
    runtime-versions:
      python: 3.12
      nodejs: 20
    commands:
      - echo "Installing dependencies..."
      - pip install -r requirements.txt
      - npm ci

  pre_build:
    commands:
      - echo "Running linting..."
      - flake8 src/ --max-line-length 120
      - echo "Logging in to ECR..."
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION |
          docker login --username AWS --password-stdin $ECR_REPO_URI

  build:
    commands:
      - echo "Running tests..."
      - pytest tests/ -v --junitxml=reports/junit.xml --cov=src --cov-report=xml:reports/coverage.xml
      - echo "Building Docker image..."
      - docker build -t $ECR_REPO_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION .
      - docker tag $ECR_REPO_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION $ECR_REPO_URI:latest

  post_build:
    commands:
      - echo "Pushing Docker image..."
      - docker push $ECR_REPO_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION
      - docker push $ECR_REPO_URI:latest
      - echo "Creating imagedefinitions.json..."
      - printf '[{"name":"app","imageUri":"%s"}]' $ECR_REPO_URI:$CODEBUILD_RESOLVED_SOURCE_VERSION > imagedefinitions.json

reports:
  pytest-reports:
    files:
      - "reports/junit.xml"
    file-format: JUNITXML
  coverage-reports:
    files:
      - "reports/coverage.xml"
    file-format: COBERTURAXML

artifacts:
  files:
    - imagedefinitions.json
    - appspec.yml
    - taskdef.json

cache:
  paths:
    - "/root/.cache/pip/**/*"
    - "node_modules/**/*"
```

### コード例 3: CodeBuild プロジェクトの作成

```bash
aws codebuild create-project \
  --name my-app-build \
  --source type=CODEPIPELINE \
  --environment '{
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
    "computeType": "BUILD_GENERAL1_MEDIUM",
    "privilegedMode": true,
    "environmentVariables": [
      {"name": "ECR_REPO_URI", "value": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app"},
      {"name": "AWS_DEFAULT_REGION", "value": "ap-northeast-1"}
    ]
  }' \
  --service-role arn:aws:iam::123456789012:role/CodeBuildRole \
  --artifacts type=CODEPIPELINE
```

---

## 4. CodeDeploy

### 図解 2: CodeDeploy のデプロイ戦略

```
1. In-Place (インプレース):
   ┌────────┐  ┌────────┐  ┌────────┐
   │ EC2-1  │  │ EC2-2  │  │ EC2-3  │
   │  v1    │  │  v1    │  │  v1    │
   └───┬────┘  └────────┘  └────────┘
       │ デプロイ
       ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ EC2-1  │  │ EC2-2  │  │ EC2-3  │
   │  v2    │  │  v1    │  │  v1    │
   └────────┘  └───┬────┘  └────────┘
                   │ デプロイ
                   ▼ ... 順次実行

2. Blue/Green (ECS):
   ┌─────────────────────────────────┐
   │            ALB                  │
   │     ┌──────┴──────┐            │
   │     ▼             ▼            │
   │  ┌──────┐    ┌──────┐         │
   │  │Blue  │    │Green │         │
   │  │(v1)  │    │(v2)  │         │
   │  │100%  │    │ 0%   │         │
   │  └──────┘    └──────┘         │
   └─────────────────────────────────┘
        │ トラフィック移行 (Linear/Canary/AllAtOnce)
        ▼
   ┌─────────────────────────────────┐
   │            ALB                  │
   │     ┌──────┴──────┐            │
   │     ▼             ▼            │
   │  ┌──────┐    ┌──────┐         │
   │  │Blue  │    │Green │         │
   │  │(v1)  │    │(v2)  │         │
   │  │ 0%   │    │100%  │         │
   │  └──────┘    └──────┘         │
   └─────────────────────────────────┘

3. Lambda (カナリア/リニア):
   v1: 100% ──→ v1: 90% / v2: 10% ──→ v1: 0% / v2: 100%
   (Canary10Percent5Minutes / Linear10PercentEvery1Minute)
```

### コード例 4: appspec.yml (ECS Blue/Green)

```yaml
# appspec.yml (ECS)
version: 0.0
Resources:
  - TargetService:
      Type: AWS::ECS::Service
      Properties:
        TaskDefinition: <TASK_DEFINITION>
        LoadBalancerInfo:
          ContainerName: "app"
          ContainerPort: 8080
        PlatformVersion: "LATEST"
        CapacityProviderStrategy:
          - Base: 1
            CapacityProvider: "FARGATE"
            Weight: 1

Hooks:
  - BeforeInstall: "LambdaFunctionToValidateBeforeInstall"
  - AfterInstall: "LambdaFunctionToValidateAfterInstall"
  - AfterAllowTestTraffic: "LambdaFunctionToRunIntegrationTests"
  - BeforeAllowTraffic: "LambdaFunctionToValidateBeforeTraffic"
  - AfterAllowTraffic: "LambdaFunctionToRunSmokeTests"
```

---

## 5. Terraform での完全なパイプライン構築

### コード例 5: Terraform でパイプラインを構築

```hcl
# CodeStar Connection (GitHub 連携)
resource "aws_codestarconnections_connection" "github" {
  name          = "github-connection"
  provider_type = "GitHub"
}

# CodeBuild プロジェクト
resource "aws_codebuild_project" "app" {
  name         = "my-app-build"
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "CODEPIPELINE"
  }

  environment {
    compute_type                = "BUILD_GENERAL1_MEDIUM"
    image                       = "aws/codebuild/amazonlinux2-x86_64-standard:5.0"
    type                        = "LINUX_CONTAINER"
    privileged_mode             = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = aws_ecr_repository.app.repository_url
    }
  }

  source {
    type      = "CODEPIPELINE"
    buildspec = "buildspec.yml"
  }

  cache {
    type  = "S3"
    location = "${aws_s3_bucket.cache.bucket}/build-cache"
  }
}

# CodePipeline
resource "aws_codepipeline" "app" {
  name     = "my-app-pipeline"
  role_arn = aws_iam_role.codepipeline.arn

  artifact_store {
    location = aws_s3_bucket.artifacts.bucket
    type     = "S3"
  }

  # ソースステージ
  stage {
    name = "Source"
    action {
      name             = "GitHub"
      category         = "Source"
      owner            = "AWS"
      provider         = "CodeStarSourceConnection"
      version          = "1"
      output_artifacts = ["source_output"]

      configuration = {
        ConnectionArn    = aws_codestarconnections_connection.github.arn
        FullRepositoryId = "my-org/my-app"
        BranchName       = "main"
        DetectChanges    = "true"
      }
    }
  }

  # ビルドステージ
  stage {
    name = "Build"
    action {
      name             = "Build"
      category         = "Build"
      owner            = "AWS"
      provider         = "CodeBuild"
      version          = "1"
      input_artifacts  = ["source_output"]
      output_artifacts = ["build_output"]

      configuration = {
        ProjectName = aws_codebuild_project.app.name
      }
    }
  }

  # ステージングデプロイ
  stage {
    name = "Deploy-Staging"
    action {
      name            = "Deploy"
      category        = "Deploy"
      owner           = "AWS"
      provider        = "ECS"
      version         = "1"
      input_artifacts = ["build_output"]

      configuration = {
        ClusterName = "staging-cluster"
        ServiceName = "my-service"
        FileName    = "imagedefinitions.json"
      }
    }
  }

  # 手動承認
  stage {
    name = "Approval"
    action {
      name     = "Approve"
      category = "Approval"
      owner    = "AWS"
      provider = "Manual"
      version  = "1"

      configuration = {
        NotificationArn = aws_sns_topic.approval.arn
        CustomData      = "ステージング確認後、本番デプロイを承認してください"
      }
    }
  }

  # 本番デプロイ
  stage {
    name = "Deploy-Production"
    action {
      name            = "Deploy"
      category        = "Deploy"
      owner           = "AWS"
      provider        = "ECS"
      version         = "1"
      input_artifacts = ["build_output"]

      configuration = {
        ClusterName = "production-cluster"
        ServiceName = "my-service"
        FileName    = "imagedefinitions.json"
      }
    }
  }
}
```

---

## 6. 比較表

### 比較表 1: AWS CI/CD サービス比較

| 項目 | CodePipeline | CodeBuild | CodeDeploy | GitHub Actions |
|------|-------------|-----------|------------|---------------|
| **カテゴリ** | オーケストレーション | ビルド/テスト | デプロイ | CI/CD 統合 |
| **実行環境** | AWS マネージド | AWS マネージド | エージェント | GitHub ホスト |
| **課金** | パイプライン/月 | ビルド時間/分 | デプロイ/回 | 分単位 |
| **GitHub 統合** | CodeStar Connection | Webhook | なし | ネイティブ |
| **ECS デプロイ** | ECS アクション | なし | Blue/Green | aws-actions |
| **Lambda デプロイ** | Lambda アクション | SAM deploy | カナリア | SAM CLI |
| **承認ゲート** | あり | なし | なし | Environment Protection |

### 比較表 2: デプロイ戦略比較

| 戦略 | ダウンタイム | ロールバック | リスク | コスト |
|------|-------------|-------------|--------|--------|
| **All-at-once** | あり | 手動再デプロイ | 高 | 最低 |
| **Rolling** | 最小 | 手動 | 中 | 低 |
| **Blue/Green** | なし | 即座 (トラフィック切替) | 低 | 高 (2 倍のリソース) |
| **Canary** | なし | 自動 (メトリクス判定) | 最低 | 中 |
| **Linear** | なし | 自動 | 低 | 中 |

---

## 7. 図解 3: GitHub Actions との連携パターン

```
パターン 1: GitHub Actions → AWS デプロイ
  ┌─────────────┐     ┌───────────────┐     ┌──────────┐
  │ GitHub      │     │ GitHub Actions│     │ AWS      │
  │ push/PR     │ ──→ │ Build & Test  │ ──→ │ ECS/     │
  │             │     │ Docker Build  │     │ Lambda   │
  └─────────────┘     └───────────────┘     └──────────┘
  ※ OIDC で IAM ロール引き受け (アクセスキー不要)

パターン 2: GitHub → CodePipeline
  ┌─────────────┐     ┌───────────────┐     ┌──────────┐
  │ GitHub      │     │ CodePipeline  │     │ AWS      │
  │ push        │ ──→ │ CodeBuild     │ ──→ │ CodeDeploy│
  │             │     │ (buildspec)   │     │ Blue/Green│
  └─────────────┘     └───────────────┘     └──────────┘
  ※ CodeStar Connection で連携

パターン 3: ハイブリッド
  ┌─────────────┐     ┌───────────────┐
  │ GitHub      │     │ GitHub Actions│
  │ push/PR     │ ──→ │ Lint & Test   │ ← CI (GitHub)
  │             │     └───────┬───────┘
  │             │             │ merge to main
  │             │             ▼
  │             │     ┌───────────────┐
  │             │     │ CodePipeline  │ ← CD (AWS)
  │             │     │ Build → Deploy│
  │             │     └───────────────┘
  └─────────────┘
```

### コード例 6: GitHub Actions で AWS にデプロイ

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS ECS

on:
  push:
    branches: [main]

permissions:
  id-token: write   # OIDC
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: ap-northeast-1

      - name: Login to ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and Push Docker Image
        env:
          ECR_REGISTRY: ${{ steps.ecr-login.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/my-app:$IMAGE_TAG .
          docker push $ECR_REGISTRY/my-app:$IMAGE_TAG

      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: task-definition.json
          service: my-service
          cluster: production-cluster
          wait-for-service-stability: true
```

---

## 8. アンチパターン

### アンチパターン 1: 本番直接デプロイ（承認ゲートなし）

```
[悪い例]
  Source → Build → Deploy (本番)
  → テストなし、承認なし
  → バグが即座に本番に到達

[良い例]
  Source → Build → Test → Deploy(Staging) → 手動承認 → Deploy(Production)
  → ステージング環境で動作確認
  → 手動承認で人間のチェックを挟む
  → 問題があればパイプラインを停止
```

### アンチパターン 2: ビルドのシークレットをハードコード

```
[悪い例]
  # buildspec.yml
  phases:
    build:
      commands:
        - export DB_PASSWORD="MySecret123"  # ハードコード！
        - export API_KEY="sk-abc123"         # バージョン管理される！

[良い例]
  # buildspec.yml
  env:
    parameter-store:
      DB_PASSWORD: /myapp/prod/db-password
    secrets-manager:
      API_KEY: myapp/api-key:API_KEY

  # または CodeBuild の環境変数で Parameter Store を参照
  aws codebuild update-project \
    --name my-build \
    --environment '{
      "environmentVariables": [{
        "name": "DB_PASSWORD",
        "value": "/myapp/prod/db-password",
        "type": "PARAMETER_STORE"
      }]
    }'
```

---

## 9. FAQ

### Q1: CodePipeline と GitHub Actions のどちらを使うべきですか？

**A:** チームが GitHub を中心に開発しているなら GitHub Actions で CI/CD を完結させるのが効率的。AWS サービス（ECS Blue/Green、CodeDeploy Canary）の高度なデプロイ機能を使いたい場合は CodePipeline を選択するか、GitHub Actions の CI + CodePipeline の CD のハイブリッド構成を推奨する。OIDC 連携を使えば GitHub Actions から安全に AWS にデプロイできる。

### Q2: パイプラインが失敗した場合のロールバック方法は？

**A:** ECS Blue/Green デプロイの場合、CodeDeploy がヘルスチェック失敗を検知すると自動ロールバックする。手動ロールバックは `aws deploy stop-deployment` で実行する。Lambda のカナリアデプロイも CloudWatch アラームに基づく自動ロールバックが可能。EC2 の In-Place デプロイではロールバックが難しいため、Blue/Green を推奨する。

### Q3: ビルド時間を短縮するにはどうすればよいですか？

**A:** (1) CodeBuild のキャッシュ（S3 または ローカルキャッシュ）で依存関係のダウンロードを省略、(2) マルチステージ Docker ビルドでレイヤーキャッシュを活用、(3) CodeBuild のコンピュートタイプを上げる（MEDIUM → LARGE）、(4) テストの並列実行、(5) 変更されたパッケージのみをビルドするモノレポ戦略を導入する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| パイプライン構成 | Source → Build → Test → Staging → Approval → Production |
| ソース統合 | GitHub は CodeStar Connection で連携。OIDC 認証推奨 |
| ビルド | CodeBuild の buildspec.yml で定義。キャッシュで高速化 |
| デプロイ戦略 | 本番は Blue/Green またはカナリア。ロールバック自動化 |
| シークレット | Parameter Store / Secrets Manager から参照。ハードコード禁止 |
| 承認ゲート | 本番デプロイ前に手動承認ステージを設置 |
| 監視 | CloudWatch Events でパイプライン失敗を通知 |

---

## 次に読むべきガイド

- [00-iam-deep-dive.md](../08-security/00-iam-deep-dive.md) — パイプラインの IAM ロール設計
- [01-secrets-management.md](../08-security/01-secrets-management.md) — ビルド時のシークレット管理
- [00-cost-optimization.md](../09-cost/00-cost-optimization.md) — CI/CD コストの最適化

---

## 参考文献

1. **AWS 公式ドキュメント** — AWS CodePipeline ユーザーガイド
   https://docs.aws.amazon.com/codepipeline/latest/userguide/
2. **AWS CodeBuild ユーザーガイド** — buildspec リファレンスとベストプラクティス
   https://docs.aws.amazon.com/codebuild/latest/userguide/
3. **GitHub Actions — AWS デプロイ** — OIDC と aws-actions の公式ガイド
   https://docs.github.com/en/actions/deployment/deploying-to-aws
