# AWS CodePipeline

> AWS のフルマネージド CI/CD サービスを理解し、CodeCommit・CodeBuild・CodeDeploy・GitHub を統合した自動化パイプラインを構築する

## この章で学ぶこと

1. **CodePipeline の基本概念** — ステージ、アクション、アーティファクトの構造
2. **CodeBuild と CodeDeploy の統合** — ビルド・テスト・デプロイの自動化
3. **GitHub 統合とベストプラクティス** — GitHub Actions との連携、承認ゲート、ロールバック
4. **CDK による CodePipeline 構築** — Infrastructure as Code でパイプラインを管理
5. **パイプラインの監視とトラブルシューティング** — CloudWatch、EventBridge による障害対応

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

### 1.1 CodePipeline の主要コンポーネント

| コンポーネント | 説明 | 例 |
|--------------|------|-----|
| **Pipeline** | ステージの集合体。ソースからデプロイまでの一連のフロー | my-app-pipeline |
| **Stage** | パイプライン内の論理的なフェーズ | Source, Build, Test, Deploy |
| **Action** | ステージ内で実行される個別のタスク | CodeBuild アクション、ECS デプロイアクション |
| **Artifact** | ステージ間で受け渡されるファイル（S3 に保存） | SourceOutput, BuildOutput |
| **Transition** | ステージ間の接続。有効/無効を切り替え可能 | Build → Test の遷移 |

### 1.2 CodePipeline V2 の新機能

CodePipeline V2 では以下の機能が追加された:

- **トリガーフィルタ**: ブランチ名、ファイルパス、タグでパイプライン実行を制御
- **変数の拡張**: ステージ間で変数を引き渡し
- **実行モード**: QUEUED（キュー）、SUPERSEDED（最新優先）、PARALLEL（並列）
- **Git タグトリガー**: タグの作成・更新でパイプラインを起動

```json
{
  "pipeline": {
    "pipelineType": "V2",
    "executionMode": "QUEUED",
    "triggers": [
      {
        "providerType": "CodeStarSourceConnection",
        "gitConfiguration": {
          "sourceActionName": "GitHub-Source",
          "push": [
            {
              "branches": {
                "includes": ["main", "release/*"]
              },
              "filePaths": {
                "includes": ["src/**", "package.json"],
                "excludes": ["docs/**", "*.md"]
              }
            }
          ]
        }
      }
    ]
  }
}
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

# パイプラインの状態確認
aws codepipeline get-pipeline-state --name my-app-pipeline

# パイプラインの手動実行
aws codepipeline start-pipeline-execution --name my-app-pipeline

# パイプラインの実行履歴
aws codepipeline list-pipeline-executions --pipeline-name my-app-pipeline

# 特定ステージのリトライ
aws codepipeline retry-stage-execution \
  --pipeline-name my-app-pipeline \
  --stage-name Build \
  --pipeline-execution-id "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" \
  --retry-mode FAILED_ACTIONS
```

### コード例 1.5: パイプライン定義 JSON

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

### 3.1 CodeBuild の環境とコンピュートタイプ

| コンピュートタイプ | vCPU | メモリ | 月額目安（ビルド分） | 推奨用途 |
|-------------------|------|--------|---------------------|---------|
| BUILD_GENERAL1_SMALL | 2 | 3 GB | $0.005/分 | 軽量ビルド、Lint |
| BUILD_GENERAL1_MEDIUM | 4 | 7 GB | $0.010/分 | 一般的なビルド |
| BUILD_GENERAL1_LARGE | 8 | 15 GB | $0.020/分 | Docker ビルド |
| BUILD_GENERAL1_XLARGE | 36 | 70 GB | $0.040/分 | 大規模ビルド |
| BUILD_GENERAL1_2XLARGE | 72 | 145 GB | $0.080/分 | モノレポ |
| BUILD_LAMBDA_1GB | 2 | 1 GB | $0.00375/分 | Lambda 環境 |
| BUILD_LAMBDA_10GB | 2 | 10 GB | $0.01875/分 | Lambda 大 |

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
  exported-variables:
    - BUILD_ID
    - IMAGE_TAG

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
      - echo "Running security scan..."
      - bandit -r src/ -f json -o reports/bandit.json || true
      - echo "Logging in to ECR..."
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION |
          docker login --username AWS --password-stdin $ECR_REPO_URI
      - export IMAGE_TAG=$CODEBUILD_RESOLVED_SOURCE_VERSION
      - export BUILD_ID=$CODEBUILD_BUILD_NUMBER

  build:
    commands:
      - echo "Running tests..."
      - pytest tests/ -v --junitxml=reports/junit.xml --cov=src --cov-report=xml:reports/coverage.xml
      - echo "Building Docker image..."
      - docker build -t $ECR_REPO_URI:$IMAGE_TAG .
      - docker tag $ECR_REPO_URI:$IMAGE_TAG $ECR_REPO_URI:latest

  post_build:
    commands:
      - echo "Pushing Docker image..."
      - docker push $ECR_REPO_URI:$IMAGE_TAG
      - docker push $ECR_REPO_URI:latest
      - echo "Creating imagedefinitions.json..."
      - printf '[{"name":"app","imageUri":"%s"}]' $ECR_REPO_URI:$IMAGE_TAG > imagedefinitions.json
      - echo "Build completed on $(date)"

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
  secondary-artifacts:
    test-reports:
      files:
        - "reports/**/*"
      base-directory: .

cache:
  paths:
    - "/root/.cache/pip/**/*"
    - "node_modules/**/*"
    - "/root/.docker/**/*"
```

### 3.2 マルチステージ buildspec（テストとビルドの分離）

```yaml
# buildspec-test.yml（テスト専用）
version: 0.2

phases:
  install:
    runtime-versions:
      python: 3.12
    commands:
      - pip install -r requirements.txt -r requirements-dev.txt

  build:
    commands:
      # ユニットテスト
      - pytest tests/unit/ -v --junitxml=reports/unit-test.xml
      # 統合テスト
      - pytest tests/integration/ -v --junitxml=reports/integration-test.xml
      # カバレッジ
      - pytest tests/ --cov=src --cov-report=xml:reports/coverage.xml --cov-fail-under=80
      # セキュリティスキャン
      - safety check --json --output reports/safety.json || true
      - bandit -r src/ -f json -o reports/bandit.json || true

reports:
  unit-tests:
    files:
      - "reports/unit-test.xml"
    file-format: JUNITXML
  integration-tests:
    files:
      - "reports/integration-test.xml"
    file-format: JUNITXML
  coverage:
    files:
      - "reports/coverage.xml"
    file-format: COBERTURAXML
```

```yaml
# buildspec-build.yml（ビルド専用）
version: 0.2

phases:
  pre_build:
    commands:
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION |
          docker login --username AWS --password-stdin $ECR_REPO_URI
      - export IMAGE_TAG=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | head -c 8)

  build:
    commands:
      - docker build \
          --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
          --build-arg VCS_REF=$CODEBUILD_RESOLVED_SOURCE_VERSION \
          --cache-from $ECR_REPO_URI:latest \
          -t $ECR_REPO_URI:$IMAGE_TAG \
          -t $ECR_REPO_URI:latest \
          .

  post_build:
    commands:
      - docker push $ECR_REPO_URI:$IMAGE_TAG
      - docker push $ECR_REPO_URI:latest
      - printf '[{"name":"app","imageUri":"%s"}]' $ECR_REPO_URI:$IMAGE_TAG > imagedefinitions.json

artifacts:
  files:
    - imagedefinitions.json
    - appspec.yml
    - taskdef.json
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

### 3.3 CodeBuild のキャッシュ戦略

```bash
# S3 キャッシュ（ビルド間で共有）
aws codebuild update-project \
  --name my-app-build \
  --cache '{
    "type": "S3",
    "location": "my-codebuild-cache/my-app"
  }'

# ローカルキャッシュ（同じビルドホスト内で共有）
aws codebuild update-project \
  --name my-app-build \
  --cache '{
    "type": "LOCAL",
    "modes": [
      "LOCAL_DOCKER_LAYER_CACHE",
      "LOCAL_SOURCE_CACHE",
      "LOCAL_CUSTOM_CACHE"
    ]
  }'
```

### 3.4 CodeBuild のバッチビルド

```yaml
# buildspec-batch.yml
version: 0.2

batch:
  fast-fail: true
  build-graph:
    - identifier: lint
      buildspec: buildspec-lint.yml
    - identifier: unit_test
      buildspec: buildspec-unit-test.yml
    - identifier: integration_test
      buildspec: buildspec-integration-test.yml
      depend-on:
        - lint
    - identifier: build
      buildspec: buildspec-build.yml
      depend-on:
        - unit_test
        - integration_test
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

### 4.1 ECS Blue/Green デプロイの設定

```bash
# CodeDeploy アプリケーション作成
aws deploy create-application \
  --application-name my-ecs-app \
  --compute-platform ECS

# デプロイグループ作成（Canary デプロイ）
aws deploy create-deployment-group \
  --application-name my-ecs-app \
  --deployment-group-name my-ecs-dg \
  --service-role-arn arn:aws:iam::123456789012:role/CodeDeployECSRole \
  --deployment-config-name CodeDeployDefault.ECSCanary10Percent5Minutes \
  --ecs-services '[{
    "serviceName": "my-service",
    "clusterName": "my-cluster"
  }]' \
  --load-balancer-info '{
    "targetGroupPairInfoList": [{
      "targetGroups": [
        {"name": "my-tg-blue"},
        {"name": "my-tg-green"}
      ],
      "prodTrafficRoute": {
        "listenerArns": [
          "arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:listener/app/my-alb/xxx/yyy"
        ]
      },
      "testTrafficRoute": {
        "listenerArns": [
          "arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:listener/app/my-alb/xxx/zzz"
        ]
      }
    }]
  }' \
  --blue-green-deployment-configuration '{
    "terminateBlueInstancesOnDeploymentSuccess": {
      "action": "TERMINATE",
      "terminationWaitTimeInMinutes": 60
    },
    "deploymentReadyOption": {
      "actionOnTimeout": "CONTINUE_DEPLOYMENT",
      "waitTimeInMinutes": 0
    }
  }' \
  --auto-rollback-configuration '{
    "enabled": true,
    "events": ["DEPLOYMENT_FAILURE", "DEPLOYMENT_STOP_ON_ALARM"]
  }' \
  --alarm-configuration '{
    "enabled": true,
    "alarms": [
      {"name": "my-ecs-5xx-alarm"},
      {"name": "my-ecs-latency-alarm"}
    ]
  }'
```

### 4.2 デプロイ設定の比較

| デプロイ設定名 | 戦略 | トラフィック移行 | 用途 |
|---------------|------|----------------|------|
| CodeDeployDefault.ECSAllAtOnce | 一括 | 即時 100% | 開発環境 |
| CodeDeployDefault.ECSLinear10PercentEvery1Minute | リニア | 1分ごとに10% | テスト環境 |
| CodeDeployDefault.ECSLinear10PercentEvery3Minutes | リニア | 3分ごとに10% | ステージング |
| CodeDeployDefault.ECSCanary10Percent5Minutes | カナリア | 10%→5分待機→90% | 本番環境推奨 |
| CodeDeployDefault.ECSCanary10Percent15Minutes | カナリア | 10%→15分待機→90% | 本番環境高安全 |

### 4.3 Lambda デプロイのライフサイクルフック

```python
# hooks/validate_after_install.py
"""CodeDeploy のライフサイクルフック: デプロイ後の検証"""
import boto3
import json
import urllib3

codedeploy = boto3.client("codedeploy")
http = urllib3.PoolManager()


def handler(event, context):
    deployment_id = event["DeploymentId"]
    lifecycle_event_hook_execution_id = event["LifecycleEventHookExecutionId"]

    try:
        # ヘルスチェック
        response = http.request("GET", "http://localhost:8080/health", timeout=10)

        if response.status == 200:
            status = "Succeeded"
            print(f"Health check passed: {response.data.decode()}")
        else:
            status = "Failed"
            print(f"Health check failed: status={response.status}")

    except Exception as e:
        status = "Failed"
        print(f"Health check error: {str(e)}")

    # 結果を CodeDeploy に報告
    codedeploy.put_lifecycle_event_hook_execution_status(
        deploymentId=deployment_id,
        lifecycleEventHookExecutionId=lifecycle_event_hook_execution_id,
        status=status,
    )

    return {"statusCode": 200, "body": json.dumps({"status": status})}
```

---

## 5. CDK による CodePipeline 構築

### コード例 5: CDK でパイプラインを構築

```typescript
import * as cdk from 'aws-cdk-lib';
import * as codepipeline from 'aws-cdk-lib/aws-codepipeline';
import * as codepipeline_actions from 'aws-cdk-lib/aws-codepipeline-actions';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as sns_subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as events_targets from 'aws-cdk-lib/aws-events-targets';
import { Construct } from 'constructs';

interface PipelineStackProps extends cdk.StackProps {
  ecrRepository: ecr.Repository;
  ecsClusterStaging: ecs.Cluster;
  ecsClusterProd: ecs.Cluster;
  ecsServiceStaging: ecs.FargateService;
  ecsServiceProd: ecs.FargateService;
  notificationEmail: string;
}

export class PipelineStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: PipelineStackProps) {
    super(scope, id, props);

    // SNS トピック（通知用）
    const approvalTopic = new sns.Topic(this, 'ApprovalTopic', {
      displayName: 'Deploy Approval Notifications',
    });
    approvalTopic.addSubscription(
      new sns_subscriptions.EmailSubscription(props.notificationEmail)
    );

    // パイプライン失敗通知用
    const failureTopic = new sns.Topic(this, 'FailureTopic', {
      displayName: 'Pipeline Failure Notifications',
    });
    failureTopic.addSubscription(
      new sns_subscriptions.EmailSubscription(props.notificationEmail)
    );

    // アーティファクト
    const sourceOutput = new codepipeline.Artifact('SourceOutput');
    const buildOutput = new codepipeline.Artifact('BuildOutput');
    const testOutput = new codepipeline.Artifact('TestOutput');

    // CodeBuild プロジェクト（テスト）
    const testProject = new codebuild.PipelineProject(this, 'TestProject', {
      projectName: 'my-app-test',
      buildSpec: codebuild.BuildSpec.fromSourceFilename('buildspec-test.yml'),
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.MEDIUM,
      },
      timeout: cdk.Duration.minutes(15),
    });

    // CodeBuild プロジェクト（ビルド）
    const buildProject = new codebuild.PipelineProject(this, 'BuildProject', {
      projectName: 'my-app-build',
      buildSpec: codebuild.BuildSpec.fromSourceFilename('buildspec-build.yml'),
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.MEDIUM,
        privileged: true, // Docker ビルドに必要
        environmentVariables: {
          ECR_REPO_URI: {
            value: props.ecrRepository.repositoryUri,
          },
          AWS_DEFAULT_REGION: {
            value: this.region,
          },
        },
      },
      cache: codebuild.Cache.local(
        codebuild.LocalCacheMode.DOCKER_LAYER,
        codebuild.LocalCacheMode.CUSTOM,
      ),
      timeout: cdk.Duration.minutes(30),
    });

    // ECR への push 権限
    props.ecrRepository.grantPullPush(buildProject);

    // パイプライン
    const pipeline = new codepipeline.Pipeline(this, 'Pipeline', {
      pipelineName: 'my-app-pipeline',
      crossAccountKeys: false,
      restartExecutionOnUpdate: true,
    });

    // ソースステージ
    pipeline.addStage({
      stageName: 'Source',
      actions: [
        new codepipeline_actions.CodeStarConnectionsSourceAction({
          actionName: 'GitHub-Source',
          connectionArn: 'arn:aws:codestar-connections:ap-northeast-1:123456789012:connection/xxx',
          owner: 'my-org',
          repo: 'my-app',
          branch: 'main',
          output: sourceOutput,
          triggerOnPush: true,
        }),
      ],
    });

    // テストステージ
    pipeline.addStage({
      stageName: 'Test',
      actions: [
        new codepipeline_actions.CodeBuildAction({
          actionName: 'UnitTest',
          project: testProject,
          input: sourceOutput,
          outputs: [testOutput],
        }),
      ],
    });

    // ビルドステージ
    pipeline.addStage({
      stageName: 'Build',
      actions: [
        new codepipeline_actions.CodeBuildAction({
          actionName: 'DockerBuild',
          project: buildProject,
          input: sourceOutput,
          outputs: [buildOutput],
        }),
      ],
    });

    // ステージングデプロイ
    pipeline.addStage({
      stageName: 'Deploy-Staging',
      actions: [
        new codepipeline_actions.EcsDeployAction({
          actionName: 'Deploy-ECS-Staging',
          service: props.ecsServiceStaging,
          input: buildOutput,
        }),
      ],
    });

    // 手動承認
    pipeline.addStage({
      stageName: 'Approval',
      actions: [
        new codepipeline_actions.ManualApprovalAction({
          actionName: 'Approve-Production',
          notificationTopic: approvalTopic,
          additionalInformation: 'ステージング環境での動作確認が完了したら承認してください',
          externalEntityUrl: 'https://staging.example.com',
        }),
      ],
    });

    // 本番デプロイ
    pipeline.addStage({
      stageName: 'Deploy-Production',
      actions: [
        new codepipeline_actions.EcsDeployAction({
          actionName: 'Deploy-ECS-Production',
          service: props.ecsServiceProd,
          input: buildOutput,
        }),
      ],
    });

    // パイプライン失敗時の通知
    pipeline.onStateChange('PipelineStateChange', {
      target: new events_targets.SnsTopic(failureTopic),
      eventPattern: {
        detail: {
          state: ['FAILED'],
        },
      },
    });
  }
}
```

---

## 6. Terraform での完全なパイプライン構築

### コード例 6: Terraform でパイプラインを構築

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

  logs_config {
    cloudwatch_logs {
      group_name  = "/codebuild/my-app-build"
      stream_name = ""
    }
    s3_logs {
      status   = "ENABLED"
      location = "${aws_s3_bucket.logs.bucket}/codebuild-logs"
    }
  }
}

# CodePipeline
resource "aws_codepipeline" "app" {
  name     = "my-app-pipeline"
  role_arn = aws_iam_role.codepipeline.arn

  artifact_store {
    location = aws_s3_bucket.artifacts.bucket
    type     = "S3"

    encryption_key {
      id   = aws_kms_key.pipeline.arn
      type = "KMS"
    }
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

# パイプライン失敗時の EventBridge ルール
resource "aws_cloudwatch_event_rule" "pipeline_failure" {
  name        = "pipeline-failure-notification"
  description = "パイプライン失敗時の通知"

  event_pattern = jsonencode({
    source      = ["aws.codepipeline"]
    detail-type = ["CodePipeline Pipeline Execution State Change"]
    detail = {
      pipeline = [aws_codepipeline.app.name]
      state    = ["FAILED"]
    }
  })
}

resource "aws_cloudwatch_event_target" "pipeline_failure_sns" {
  rule      = aws_cloudwatch_event_rule.pipeline_failure.name
  target_id = "SendToSNS"
  arn       = aws_sns_topic.failure.arn
}

# IAM ロール: CodePipeline
resource "aws_iam_role" "codepipeline" {
  name = "CodePipelineRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "codepipeline.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "codepipeline" {
  name = "CodePipelinePolicy"
  role = aws_iam_role.codepipeline.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:GetBucketVersioning"
        ]
        Resource = [
          aws_s3_bucket.artifacts.arn,
          "${aws_s3_bucket.artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "codebuild:BatchGetBuilds",
          "codebuild:StartBuild"
        ]
        Resource = aws_codebuild_project.app.arn
      },
      {
        Effect   = "Allow"
        Action   = ["codestar-connections:UseConnection"]
        Resource = aws_codestarconnections_connection.github.arn
      },
      {
        Effect = "Allow"
        Action = [
          "ecs:DescribeServices",
          "ecs:DescribeTaskDefinition",
          "ecs:DescribeTasks",
          "ecs:ListTasks",
          "ecs:RegisterTaskDefinition",
          "ecs:UpdateService"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = "iam:PassRole"
        Resource = "*"
        Condition = {
          StringEqualsIfExists = {
            "iam:PassedToService" = [
              "ecs-tasks.amazonaws.com"
            ]
          }
        }
      },
      {
        Effect   = "Allow"
        Action   = "sns:Publish"
        Resource = aws_sns_topic.approval.arn
      }
    ]
  })
}
```

---

## 7. 比較表

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
| **パイプライン as Code** | JSON/CDK | buildspec.yml | appspec.yml | YAML workflow |
| **セルフホストランナー** | なし | なし | EC2 エージェント | あり |
| **並列実行** | ステージ内アクション | バッチビルド | なし | matrix |

### 比較表 2: デプロイ戦略比較

| 戦略 | ダウンタイム | ロールバック | リスク | コスト |
|------|-------------|-------------|--------|--------|
| **All-at-once** | あり | 手動再デプロイ | 高 | 最低 |
| **Rolling** | 最小 | 手動 | 中 | 低 |
| **Blue/Green** | なし | 即座 (トラフィック切替) | 低 | 高 (2 倍のリソース) |
| **Canary** | なし | 自動 (メトリクス判定) | 最低 | 中 |
| **Linear** | なし | 自動 | 低 | 中 |

### 比較表 3: CodePipeline V1 vs V2

| 機能 | V1 | V2 |
|------|-----|-----|
| トリガーフィルタ | なし | ブランチ、ファイルパス、タグ |
| 実行モード | SUPERSEDED のみ | QUEUED, SUPERSEDED, PARALLEL |
| 変数 | 制限あり | ステージ間変数引き渡し |
| 料金 | $1/パイプライン/月 | $1/パイプライン/月 + アクション料 |
| Git タグトリガー | なし | あり |

---

## 8. 図解 3: GitHub Actions との連携パターン

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

### コード例 7: GitHub Actions で AWS にデプロイ

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS ECS

on:
  push:
    branches: [main]

permissions:
  id-token: write   # OIDC
  contents: read

env:
  AWS_REGION: ap-northeast-1
  ECR_REPOSITORY: my-app
  ECS_CLUSTER: production-cluster
  ECS_SERVICE: my-service

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: "pip"

      - name: Install Dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt

      - name: Run Tests
        run: pytest tests/ -v --junitxml=reports/junit.xml --cov=src --cov-report=xml

      - name: Upload Test Results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results
          path: reports/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsRole
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and Push Docker Image
        env:
          ECR_REGISTRY: ${{ steps.ecr-login.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG

      - name: Download Task Definition
        run: |
          aws ecs describe-task-definition \
            --task-definition my-task \
            --query taskDefinition > task-definition.json

      - name: Update Task Definition
        id: task-def
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition: task-definition.json
          container-name: app
          image: ${{ steps.ecr-login.outputs.registry }}/${{ env.ECR_REPOSITORY }}:${{ github.sha }}

      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition: ${{ steps.task-def.outputs.task-definition }}
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
          wait-for-minutes: 10

      - name: Notify on Failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Deploy FAILED: ${{ github.repository }}@${{ github.sha }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

### コード例 8: OIDC プロバイダの設定（CloudFormation）

```yaml
# github-oidc.yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: GitHub Actions OIDC Provider and IAM Role

Resources:
  GitHubOIDCProvider:
    Type: AWS::IAM::OIDCProvider
    Properties:
      Url: https://token.actions.githubusercontent.com
      ClientIdList:
        - sts.amazonaws.com
      ThumbprintList:
        - 6938fd4d98bab03faadb97b34396831e3780aea1

  GitHubActionsRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: GitHubActionsRole
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Federated: !GetAtt GitHubOIDCProvider.Arn
            Action: sts:AssumeRoleWithWebIdentity
            Condition:
              StringEquals:
                "token.actions.githubusercontent.com:aud": sts.amazonaws.com
              StringLike:
                "token.actions.githubusercontent.com:sub":
                  - "repo:my-org/my-app:ref:refs/heads/main"
                  - "repo:my-org/my-app:environment:production"
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser
      Policies:
        - PolicyName: ECSDeployPolicy
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - ecs:DescribeServices
                  - ecs:DescribeTaskDefinition
                  - ecs:DescribeTasks
                  - ecs:ListTasks
                  - ecs:RegisterTaskDefinition
                  - ecs:UpdateService
                Resource: "*"
              - Effect: Allow
                Action: iam:PassRole
                Resource: "*"
                Condition:
                  StringEqualsIfExists:
                    "iam:PassedToService": ecs-tasks.amazonaws.com
```

---

## 9. パイプラインの監視とトラブルシューティング

### 9.1 EventBridge によるパイプラインイベント監視

```bash
# パイプラインの状態変更を監視するルール
aws events put-rule \
  --name "pipeline-state-change" \
  --event-pattern '{
    "source": ["aws.codepipeline"],
    "detail-type": ["CodePipeline Pipeline Execution State Change"],
    "detail": {
      "pipeline": ["my-app-pipeline"],
      "state": ["FAILED", "SUCCEEDED"]
    }
  }'

# Lambda をターゲットとして Slack 通知
aws events put-targets \
  --rule "pipeline-state-change" \
  --targets '[{
    "Id": "SlackNotification",
    "Arn": "arn:aws:lambda:ap-northeast-1:123456789012:function:slack-notify"
  }]'
```

### 9.2 Slack 通知 Lambda

```python
# slack_notify.py
import json
import os
import urllib3

SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
http = urllib3.PoolManager()


def handler(event, context):
    detail = event["detail"]
    pipeline = detail["pipeline"]
    state = detail["state"]
    execution_id = detail["execution-id"]

    color_map = {
        "SUCCEEDED": "#36a64f",
        "FAILED": "#ff0000",
        "STARTED": "#439FE0",
    }

    emoji_map = {
        "SUCCEEDED": ":white_check_mark:",
        "FAILED": ":x:",
        "STARTED": ":rocket:",
    }

    message = {
        "attachments": [
            {
                "color": color_map.get(state, "#808080"),
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"{emoji_map.get(state, '')} *Pipeline {state}*\n"
                                f"*Pipeline:* {pipeline}\n"
                                f"*Execution ID:* `{execution_id}`\n"
                                f"*Region:* {event['region']}"
                            ),
                        },
                    }
                ],
            }
        ]
    }

    response = http.request(
        "POST",
        SLACK_WEBHOOK_URL,
        body=json.dumps(message).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    return {"statusCode": response.status}
```

### 9.3 CloudWatch ダッシュボード

```bash
# パイプラインのメトリクスダッシュボード作成
aws cloudwatch put-dashboard \
  --dashboard-name "CICD-Dashboard" \
  --dashboard-body '{
    "widgets": [
      {
        "type": "metric",
        "properties": {
          "title": "Pipeline Execution Duration",
          "metrics": [
            ["AWS/CodePipeline", "PipelineExecutionTime",
             "PipelineName", "my-app-pipeline",
             {"stat": "Average", "period": 86400}]
          ],
          "view": "timeSeries",
          "period": 86400
        }
      },
      {
        "type": "metric",
        "properties": {
          "title": "Pipeline Success Rate",
          "metrics": [
            ["AWS/CodePipeline", "PipelineExecutionSucceeded",
             "PipelineName", "my-app-pipeline",
             {"stat": "Sum", "period": 86400}],
            ["AWS/CodePipeline", "PipelineExecutionFailed",
             "PipelineName", "my-app-pipeline",
             {"stat": "Sum", "period": 86400}]
          ],
          "view": "timeSeries"
        }
      },
      {
        "type": "metric",
        "properties": {
          "title": "CodeBuild Duration",
          "metrics": [
            ["AWS/CodeBuild", "Duration",
             "ProjectName", "my-app-build",
             {"stat": "Average"}]
          ],
          "view": "timeSeries"
        }
      }
    ]
  }'
```

### 9.4 よくあるトラブルシューティング

| 問題 | 原因 | 解決策 |
|------|------|--------|
| Source ステージ失敗 | CodeStar Connection が保留状態 | AWS コンソールで接続を手動承認 |
| Build ステージタイムアウト | Docker ビルドが遅い | キャッシュ有効化、コンピュートタイプ変更 |
| Deploy ステージ失敗 | タスク定義のイメージ不一致 | imagedefinitions.json の形式確認 |
| 権限エラー | IAM ロール不足 | CodePipeline/CodeBuild ロールにポリシー追加 |
| アーティファクト S3 エラー | バケットポリシー | KMS キーポリシーと S3 バケットポリシー確認 |
| ECS サービス不安定 | ヘルスチェック失敗 | ターゲットグループのヘルスチェック設定確認 |

---

## 10. アンチパターン

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

### アンチパターン 3: パイプラインの IAM ロールに広すぎる権限

```
[悪い例]
  {
    "Effect": "Allow",
    "Action": "*",
    "Resource": "*"
  }
  → 全サービスへの全操作が可能
  → セキュリティリスクが極めて高い

[良い例]
  パイプラインのロール:
  - codepipeline:* → 自パイプラインのみ
  - s3:GetObject/PutObject → アーティファクトバケットのみ
  - codebuild:StartBuild → 指定プロジェクトのみ

  CodeBuild のロール:
  - ecr:GetAuthorizationToken, ecr:BatchGetImage → 指定リポジトリのみ
  - logs:CreateLogGroup, PutLogEvents → 指定ロググループのみ
  - s3:GetObject → キャッシュバケットのみ
```

### アンチパターン 4: テストなしの自動デプロイ

```
[悪い例]
  Source → Build → Deploy
  → コンパイルは通るがテストなし
  → ランタイムエラーが本番で発覚

[良い例]
  Source → Lint → Unit Test → Integration Test → Build → Deploy
  → リンターで静的解析
  → ユニットテストでロジック検証
  → 統合テストで外部サービス連携検証
  → テストカバレッジ閾値でゲート
```

---

## 11. FAQ

### Q1: CodePipeline と GitHub Actions のどちらを使うべきですか？

**A:** チームが GitHub を中心に開発しているなら GitHub Actions で CI/CD を完結させるのが効率的。AWS サービス（ECS Blue/Green、CodeDeploy Canary）の高度なデプロイ機能を使いたい場合は CodePipeline を選択するか、GitHub Actions の CI + CodePipeline の CD のハイブリッド構成を推奨する。OIDC 連携を使えば GitHub Actions から安全に AWS にデプロイできる。

### Q2: パイプラインが失敗した場合のロールバック方法は？

**A:** ECS Blue/Green デプロイの場合、CodeDeploy がヘルスチェック失敗を検知すると自動ロールバックする。手動ロールバックは `aws deploy stop-deployment` で実行する。Lambda のカナリアデプロイも CloudWatch アラームに基づく自動ロールバックが可能。EC2 の In-Place デプロイではロールバックが難しいため、Blue/Green を推奨する。

### Q3: ビルド時間を短縮するにはどうすればよいですか？

**A:** (1) CodeBuild のキャッシュ（S3 または ローカルキャッシュ）で依存関係のダウンロードを省略、(2) マルチステージ Docker ビルドでレイヤーキャッシュを活用、(3) CodeBuild のコンピュートタイプを上げる（MEDIUM → LARGE）、(4) テストの並列実行、(5) 変更されたパッケージのみをビルドするモノレポ戦略を導入する。

### Q4: CodePipeline V2 に移行すべきですか？

**A:** ブランチフィルタやファイルパスフィルタが必要な場合、またはパイプライン実行のキューイングが必要な場合は V2 への移行を推奨する。V2 ではトリガーの柔軟性が大幅に向上しており、不要なパイプライン実行を削減できる。既存の V1 パイプラインは引き続きサポートされるため、新規作成時に V2 を選択するのが最も自然な移行パスとなる。

### Q5: モノレポでのパイプライン設計はどうすべきですか？

**A:** CodePipeline V2 のファイルパスフィルタを使い、変更されたディレクトリに応じて異なるパイプラインをトリガーする設計が推奨される。または、CodeBuild のバッチビルドで変更検知ロジックを実装し、影響を受けるサービスのみをビルド・デプロイする。GitHub Actions の場合は `paths` フィルタとマトリクスビルドの組み合わせが効果的である。

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
| 監視 | EventBridge + CloudWatch でパイプライン失敗を通知 |
| CDK 統合 | CDK Pipelines でセルフミューテーションパイプラインを構築 |
| IAM | 最小権限の原則。パイプライン/CodeBuild 各ロールに必要最小限の権限 |

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
4. **AWS CDK Pipelines** — CDK Pipelines のデベロッパーガイド
   https://docs.aws.amazon.com/cdk/v2/guide/cdk_pipeline.html
5. **AWS CodeDeploy** — Blue/Green デプロイの設定ガイド
   https://docs.aws.amazon.com/codedeploy/latest/userguide/
