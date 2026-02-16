# Amazon ECS 基礎

> Amazon Elastic Container Service (ECS) の基本概念であるタスク定義、サービス、Fargate と EC2 起動タイプの違い、ALB 統合、ログ設定までを体系的に学ぶ。

---

## この章で学ぶこと

1. **ECS のアーキテクチャとコア概念** -- クラスター、タスク定義、サービス、タスクの関係を理解する
2. **Fargate と EC2 起動タイプの使い分け** -- 各起動タイプの特徴、コスト、制約を比較し適切に選択する
3. **ALB 統合とログ設定** -- ロードバランサーによるトラフィック分散と CloudWatch Logs へのログ出力を構成する
4. **デプロイ戦略** -- ローリングアップデート、Blue/Green デプロイ、サーキットブレーカーの設定を習得する
5. **ECS Exec とデバッグ** -- 実行中のコンテナへのインタラクティブアクセスとトラブルシューティング手法を学ぶ
6. **CloudFormation / CDK による ECS 構成のコード化** -- インフラをコードで管理する実践的なテンプレートを作成する

---

## 1. ECS のアーキテクチャ

### 1.1 コア概念の関係

```
ECS の階層構造:

+------------------------------------------+
|  ECS クラスター                            |
|  +--------------------------------------+ |
|  |  サービス A (Desired Count: 3)        | |
|  |  +----------+ +----------+ +--------+| |
|  |  | タスク 1  | | タスク 2  | | タスク 3|| |
|  |  | +------+ | | +------+ | | +------+| |
|  |  | |コンテナ| | | |コンテナ| | | |コンテナ|| |
|  |  | |  A   | | | |  A   | | | |  A   || |
|  |  | +------+ | | +------+ | | +------+| |
|  |  | |コンテナ| | | |コンテナ| | | |コンテナ|| |
|  |  | |  B   | | | |  B   | | | |  B   || |
|  |  | +------+ | | +------+ | | +------+| |
|  |  +----------+ +----------+ +--------+| |
|  +--------------------------------------+ |
|  +--------------------------------------+ |
|  |  サービス B (Desired Count: 2)        | |
|  |  +----------+ +----------+            | |
|  |  | タスク 1  | | タスク 2  |            | |
|  |  +----------+ +----------+            | |
|  +--------------------------------------+ |
+------------------------------------------+
```

| 概念 | 説明 |
|------|------|
| クラスター | タスクとサービスの論理グループ |
| タスク定義 | コンテナの設計図(イメージ、CPU、メモリ、ポート等) |
| タスク | タスク定義のインスタンス(実行中のコンテナ群) |
| サービス | タスクの希望数を維持するスケジューラ |
| コンテナ定義 | タスク定義内の個別コンテナ設定 |

### 1.2 ECS のデータプレーン

```
コントロールプレーン (AWS管理)
+----------------------------+
|  ECS API / Scheduler       |
+----------------------------+
         |           |
         v           v
   +-----------+ +-----------+
   | Fargate   | | EC2       |
   | (サーバー  | | (自己管理  |
   |  レス)    | |  インスタンス)|
   +-----------+ +-----------+
   | MicroVM   | | EC2       |
   | 自動管理  | | ECS Agent |
   | パッチ不要 | | AMI管理   |
   +-----------+ +-----------+
```

### 1.3 ECS の主要コンポーネント詳細

```
ECS エコシステムの全体像:

+-------------------------------------------------------------+
|  Amazon ECS                                                 |
|                                                             |
|  +-------------------+  +------------------+                |
|  | ECR               |  | Service Connect  |                |
|  | (コンテナレジストリ) |  | (サービス間通信)  |                |
|  +-------------------+  +------------------+                |
|                                                             |
|  +-------------------+  +------------------+                |
|  | Task Definition   |  | Capacity Provider|                |
|  | (タスク設計図)      |  | (キャパシティ管理) |                |
|  +-------------------+  +------------------+                |
|                                                             |
|  +-------------------+  +------------------+                |
|  | Service           |  | Cluster          |                |
|  | (サービス管理)      |  | (クラスター)      |                |
|  +-------------------+  +------------------+                |
|                                                             |
|  データプレーン:                                               |
|  +-------------------+  +------------------+                |
|  | AWS Fargate       |  | EC2 Instances    |                |
|  | (サーバーレス)      |  | (セルフマネージド)  |                |
|  +-------------------+  +------------------+                |
|                                                             |
|  統合サービス:                                                 |
|  +--------+ +--------+ +--------+ +--------+ +--------+    |
|  | ALB    | | NLB    | | CloudMap| | X-Ray  | | CW Logs|    |
|  +--------+ +--------+ +--------+ +--------+ +--------+    |
+-------------------------------------------------------------+
```

---

## 2. タスク定義

### 2.1 基本的なタスク定義

```json
{
  "family": "my-web-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "web",
      "image": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "NODE_ENV", "value": "production"},
        {"name": "PORT", "value": "8080"}
      ],
      "secrets": [
        {
          "name": "DB_PASSWORD",
          "valueFrom": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:db-password"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/my-web-app",
          "awslogs-region": "ap-northeast-1",
          "awslogs-stream-prefix": "web"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

### 2.2 Fargate の CPU/メモリ組み合わせ

| CPU (vCPU) | メモリ (GB) |
|-----------|------------|
| 0.25 | 0.5, 1, 2 |
| 0.5 | 1, 2, 3, 4 |
| 1 | 2, 3, 4, 5, 6, 7, 8 |
| 2 | 4 - 16 (1GB刻み) |
| 4 | 8 - 30 (1GB刻み) |
| 8 | 16 - 60 (4GB刻み) |
| 16 | 32 - 120 (8GB刻み) |

### 2.3 マルチコンテナタスク定義 (サイドカーパターン)

```json
{
  "family": "web-with-sidecar",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "web",
      "image": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:latest",
      "essential": true,
      "portMappings": [{"containerPort": 8080}],
      "dependsOn": [
        {"containerName": "envoy", "condition": "HEALTHY"}
      ]
    },
    {
      "name": "envoy",
      "image": "public.ecr.aws/appmesh/aws-appmesh-envoy:v1.27",
      "essential": true,
      "portMappings": [{"containerPort": 9901}],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -s http://localhost:9901/server_info | grep state"],
        "interval": 10,
        "timeout": 5,
        "retries": 3
      },
      "memory": 256
    },
    {
      "name": "xray-daemon",
      "image": "public.ecr.aws/xray/aws-xray-daemon:latest",
      "essential": false,
      "portMappings": [{"containerPort": 2000, "protocol": "udp"}],
      "memory": 128
    }
  ]
}
```

### 2.4 タスク定義の実行ロールとタスクロール

```
ロールの違い:

実行ロール (Execution Role):
  ECS エージェントが使用するロール
  +--------------------------------------+
  | 用途:                                 |
  |   - ECR からのイメージプル             |
  |   - CloudWatch Logs へのログ書き込み   |
  |   - Secrets Manager からのシークレット取得|
  |   - SSM Parameter Store からの値取得  |
  +--------------------------------------+

タスクロール (Task Role):
  コンテナ内のアプリケーションが使用するロール
  +--------------------------------------+
  | 用途:                                 |
  |   - DynamoDB へのアクセス              |
  |   - S3 バケットへのアクセス             |
  |   - SQS キューへのメッセージ送信       |
  |   - その他 AWS サービスへのアクセス     |
  +--------------------------------------+
```

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAccess",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:CreateLogGroup"
      ],
      "Resource": "arn:aws:logs:ap-northeast-1:123456789012:log-group:/ecs/*"
    },
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:prod/*"
    },
    {
      "Sid": "SSMParameterAccess",
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameters",
        "ssm:GetParameter"
      ],
      "Resource": "arn:aws:ssm:ap-northeast-1:123456789012:parameter/prod/*"
    }
  ]
}
```

### 2.5 AWS CLI によるタスク定義の管理

```bash
# タスク定義の登録
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json

# タスク定義の一覧表示
aws ecs list-task-definitions --family-prefix my-web-app

# 特定のタスク定義の詳細表示
aws ecs describe-task-definition \
  --task-definition my-web-app:3

# タスク定義のリビジョン比較
aws ecs describe-task-definition --task-definition my-web-app:2 \
  --query 'taskDefinition.containerDefinitions[0].image'
aws ecs describe-task-definition --task-definition my-web-app:3 \
  --query 'taskDefinition.containerDefinitions[0].image'

# 古いタスク定義の登録解除
aws ecs deregister-task-definition \
  --task-definition my-web-app:1

# スタンドアロンタスクの実行 (バッチ処理等)
aws ecs run-task \
  --cluster my-cluster \
  --task-definition my-batch-job:1 \
  --launch-type FARGATE \
  --count 1 \
  --network-configuration '{
    "awsvpcConfiguration": {
      "subnets": ["subnet-11111111", "subnet-22222222"],
      "securityGroups": ["sg-12345678"],
      "assignPublicIp": "DISABLED"
    }
  }' \
  --overrides '{
    "containerOverrides": [
      {
        "name": "batch-processor",
        "environment": [
          {"name": "BATCH_ID", "value": "batch-20240115-001"}
        ]
      }
    ]
  }'
```

---

## 3. Fargate vs EC2 起動タイプ

### 3.1 比較表

| 特性 | Fargate | EC2 |
|------|---------|-----|
| インフラ管理 | 不要 | EC2 インスタンスの管理が必要 |
| パッチ適用 | AWS が自動管理 | ユーザーが AMI 更新 |
| スケーリング | タスク単位で自動 | ASG + タスク配置 |
| GPU サポート | なし | あり |
| ネットワークモード | awsvpc のみ | awsvpc, bridge, host, none |
| 最大 CPU/メモリ | 16 vCPU / 120 GB | インスタンスタイプに依存 |
| スポット利用 | Fargate Spot | EC2 Spot |
| 起動速度 | 30-60秒 | 即座(インスタンス起動済みなら) |
| 料金モデル | vCPU + メモリ秒課金 | EC2 インスタンス料金 |
| EBS ボリューム | 20-200 GB エフェメラルストレージ | インスタンスに依存 |
| 特権コンテナ | 不可 | 可能 |
| Docker-in-Docker | 不可 | 可能 |

### 3.2 コスト比較の目安

```
月間コスト試算 (東京リージョン, 24時間稼働, 1 vCPU / 2GB メモリ):

Fargate:
  vCPU: $0.05056/時 x 24時間 x 30日 = $36.40
  メモリ: $0.00553/GB/時 x 2GB x 24時間 x 30日 = $7.96
  合計: 約 $44.36/月/タスク

Fargate Spot:
  約 $44.36 x 0.3 = 約 $13.31/月/タスク (最大70%割引)

EC2 (t3.small オンデマンド):
  $0.0272/時 x 24時間 x 30日 = $19.58/月
  ※ 1タスクのみの場合。複数タスクを同一インスタンスに配置可能

EC2 (t3.small リザーブド 1年):
  約 $12.48/月

結論:
  - 少数タスク: EC2 の方が安い
  - 運用コスト含む: Fargate の方がTCOは有利な場合が多い
  - バースト対応: Fargate Spot が最もコスト効率が良い
```

### 3.3 Capacity Provider による柔軟なインフラ管理

```
Capacity Provider の仕組み:

+------------------------------------------+
|  ECS クラスター                            |
|                                          |
|  Capacity Provider Strategy:             |
|  +--------------------------------------+|
|  | FARGATE       : weight=1, base=2     ||
|  | FARGATE_SPOT  : weight=3, base=0     ||
|  +--------------------------------------+|
|                                          |
|  結果:                                    |
|  タスク数5の場合:                          |
|    FARGATE: 2 (base) + 0 = 2タスク        |
|    FARGATE_SPOT: 0 (base) + 3 = 3タスク   |
|                                          |
|  タスク数10の場合:                         |
|    FARGATE: 2 (base) + 2 = 4タスク        |
|    FARGATE_SPOT: 0 (base) + 6 = 6タスク   |
+------------------------------------------+
```

```bash
# Capacity Provider の設定
aws ecs put-cluster-capacity-providers \
  --cluster my-cluster \
  --capacity-providers FARGATE FARGATE_SPOT \
  --default-capacity-provider-strategy \
    capacityProvider=FARGATE,weight=1,base=2 \
    capacityProvider=FARGATE_SPOT,weight=3,base=0

# EC2 Capacity Provider の作成
aws ecs create-capacity-provider \
  --name my-ec2-capacity-provider \
  --auto-scaling-group-provider '{
    "autoScalingGroupArn": "arn:aws:autoscaling:ap-northeast-1:123456789012:autoScalingGroup:...",
    "managedScaling": {
      "status": "ENABLED",
      "targetCapacity": 80,
      "minimumScalingStepSize": 1,
      "maximumScalingStepSize": 10,
      "instanceWarmupPeriod": 300
    },
    "managedTerminationProtection": "ENABLED"
  }'
```

---

## 4. サービスの作成と管理

### 4.1 ECS サービスの作成

```bash
# クラスターの作成
aws ecs create-cluster --cluster-name my-cluster

# サービスの作成
aws ecs create-service \
  --cluster my-cluster \
  --service-name my-web-service \
  --task-definition my-web-app:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --platform-version LATEST \
  --network-configuration '{
    "awsvpcConfiguration": {
      "subnets": ["subnet-11111111", "subnet-22222222"],
      "securityGroups": ["sg-12345678"],
      "assignPublicIp": "DISABLED"
    }
  }' \
  --load-balancers '[
    {
      "targetGroupArn": "arn:aws:elasticloadbalancing:...:targetgroup/my-tg/...",
      "containerName": "web",
      "containerPort": 8080
    }
  ]' \
  --deployment-configuration '{
    "maximumPercent": 200,
    "minimumHealthyPercent": 100,
    "deploymentCircuitBreaker": {
      "enable": true,
      "rollback": true
    }
  }'

# サービスの更新 (新しいタスク定義でデプロイ)
aws ecs update-service \
  --cluster my-cluster \
  --service my-web-service \
  --task-definition my-web-app:2 \
  --force-new-deployment

# サービスの状態確認
aws ecs describe-services \
  --cluster my-cluster \
  --services my-web-service \
  --query 'services[0].{
    Status: status,
    DesiredCount: desiredCount,
    RunningCount: runningCount,
    PendingCount: pendingCount,
    TaskDefinition: taskDefinition,
    Deployments: deployments[*].{
      Status: status,
      DesiredCount: desiredCount,
      RunningCount: runningCount,
      TaskDefinition: taskDefinition
    }
  }'
```

### 4.2 Auto Scaling の設定

```bash
# Application Auto Scaling ターゲットの登録
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id "service/my-cluster/my-web-service" \
  --scalable-dimension "ecs:service:DesiredCount" \
  --min-capacity 2 \
  --max-capacity 20

# ターゲット追跡スケーリングポリシー (CPU)
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id "service/my-cluster/my-web-service" \
  --scalable-dimension "ecs:service:DesiredCount" \
  --policy-name "cpu-target-tracking" \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'

# ターゲット追跡スケーリングポリシー (メモリ)
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id "service/my-cluster/my-web-service" \
  --scalable-dimension "ecs:service:DesiredCount" \
  --policy-name "memory-target-tracking" \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 75.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageMemoryUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'

# ALB リクエスト数ベースのスケーリング
aws application-autoscaling put-scaling-policy \
  --service-namespace ecs \
  --resource-id "service/my-cluster/my-web-service" \
  --scalable-dimension "ecs:service:DesiredCount" \
  --policy-name "alb-request-count-tracking" \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 1000.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ALBRequestCountPerTarget",
      "ResourceLabel": "app/my-alb/1234567890/targetgroup/my-tg/1234567890"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'

# スケジュールベースのスケーリング
aws application-autoscaling put-scheduled-action \
  --service-namespace ecs \
  --resource-id "service/my-cluster/my-web-service" \
  --scalable-dimension "ecs:service:DesiredCount" \
  --scheduled-action-name "scale-up-morning" \
  --schedule "cron(0 8 * * ? *)" \
  --scalable-target-action "MinCapacity=5,MaxCapacity=30"

aws application-autoscaling put-scheduled-action \
  --service-namespace ecs \
  --resource-id "service/my-cluster/my-web-service" \
  --scalable-dimension "ecs:service:DesiredCount" \
  --scheduled-action-name "scale-down-night" \
  --schedule "cron(0 22 * * ? *)" \
  --scalable-target-action "MinCapacity=2,MaxCapacity=10"
```

---

## 5. デプロイ戦略

### 5.1 ローリングアップデート

```
ローリングアップデートの流れ (minimumHealthyPercent=100, maximumPercent=200):

時刻 T0: 初期状態
  [v1] [v1] [v1]  (3タスク稼働中)

時刻 T1: 新タスク起動
  [v1] [v1] [v1] [v2] [v2] [v2]  (6タスクまで許容)

時刻 T2: 新タスクがヘルシー
  [v1] [v1] [v1] [v2:healthy] [v2:healthy] [v2:healthy]

時刻 T3: 旧タスク停止
  [v2] [v2] [v2]  (3タスクに戻る)

デプロイ所要時間: 数分～10分程度
ダウンタイム: なし
```

### 5.2 Blue/Green デプロイ (CodeDeploy)

```
Blue/Green デプロイの流れ:

Phase 1: Blue (現行) が稼働中
  ALB --> Target Group 1 (Blue)
          [v1] [v1] [v1]

Phase 2: Green (新版) を起動
  ALB --> Target Group 1 (Blue)
          [v1] [v1] [v1]
          Target Group 2 (Green)  <-- テスト用リスナーで検証
          [v2] [v2] [v2]

Phase 3: トラフィック切り替え
  ALB --> Target Group 2 (Green)
          [v2] [v2] [v2]
          Target Group 1 (Blue)  <-- 一定時間保持 (ロールバック用)
          [v1] [v1] [v1]

Phase 4: Blue を削除
  ALB --> Target Group 2 (Green)
          [v2] [v2] [v2]
```

```bash
# Blue/Green デプロイ用の CodeDeploy アプリケーション作成
aws deploy create-application \
  --application-name my-ecs-app \
  --compute-platform ECS

# デプロイグループの作成
aws deploy create-deployment-group \
  --application-name my-ecs-app \
  --deployment-group-name my-ecs-dg \
  --service-role-arn arn:aws:iam::123456789012:role/CodeDeployServiceRole \
  --deployment-config-name CodeDeployDefault.ECSLinear10PercentEvery1Minutes \
  --ecs-services '{
    "clusterName": "my-cluster",
    "serviceName": "my-web-service"
  }' \
  --load-balancer-info '{
    "targetGroupPairInfoList": [
      {
        "targetGroups": [
          {"name": "my-tg-blue"},
          {"name": "my-tg-green"}
        ],
        "prodTrafficRoute": {
          "listenerArns": ["arn:aws:elasticloadbalancing:...:listener/..."]
        },
        "testTrafficRoute": {
          "listenerArns": ["arn:aws:elasticloadbalancing:...:listener/test/..."]
        }
      }
    ]
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
  }'
```

### 5.3 デプロイ設定オプション

| デプロイ設定 | 説明 |
|-------------|------|
| CodeDeployDefault.ECSAllAtOnce | 即座に全トラフィックを切り替え |
| CodeDeployDefault.ECSLinear10PercentEvery1Minutes | 毎分10%ずつ移行 |
| CodeDeployDefault.ECSLinear10PercentEvery3Minutes | 3分毎に10%ずつ移行 |
| CodeDeployDefault.ECSCanary10Percent5Minutes | 最初に10%、5分後に残り全て |
| CodeDeployDefault.ECSCanary10Percent15Minutes | 最初に10%、15分後に残り全て |

### 5.4 デプロイサーキットブレーカー

```
サーキットブレーカーの動作:

デプロイ開始
    |
    v
新タスク起動 → 起動失敗 → リトライ → 起動失敗 → リトライ → 起動失敗
    |
    v
閾値超過 (連続失敗)
    |
    v
+----------------------------+
| サーキットブレーカー発動     |
| - デプロイを停止            |
| - rollback=true なら        |
|   前バージョンに自動ロールバック|
+----------------------------+

設定:
  deploymentCircuitBreaker:
    enable: true
    rollback: true  ← 自動ロールバックを有効化
```

---

## 6. ALB 統合

### 6.1 ALB + ECS の構成

```
インターネット
    |
    v
+--------------------+
| Application Load   |
| Balancer (ALB)     |
+--------------------+
    |
    +------ リスナー (80/443)
    |         |
    |    +----+----+
    |    |ルール    |
    |    +---------+
    |    /path1 --> ターゲットグループ A
    |    /path2 --> ターゲットグループ B
    |    default -> ターゲットグループ A
    |
    v                          v
+----------+  +----------+  +----------+
| タスク 1  |  | タスク 2  |  | タスク 3  |
| :8080    |  | :8080    |  | :8080    |
| (動的Port)|  | (動的Port)|  | (動的Port)|
+----------+  +----------+  +----------+

awsvpc モードでは各タスクが独自のENIを持つ
→ 動的ポートマッピングは不要、containerPort に直接ルーティング
```

### 6.2 ALB ヘルスチェックの設定

```bash
# ターゲットグループの作成
aws elbv2 create-target-group \
  --name my-app-tg \
  --protocol HTTP \
  --port 8080 \
  --vpc-id vpc-12345678 \
  --target-type ip \
  --health-check-protocol HTTP \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --health-check-timeout-seconds 5

# ALB の作成
aws elbv2 create-load-balancer \
  --name my-app-alb \
  --subnets subnet-public-1 subnet-public-2 \
  --security-groups sg-alb-12345678 \
  --scheme internet-facing \
  --type application

# リスナーの作成 (HTTPS)
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-app-alb/... \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:...:certificate/... \
  --ssl-policy ELBSecurityPolicy-TLS13-1-2-2021-06 \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...:targetgroup/my-app-tg/...

# HTTP → HTTPS リダイレクトリスナー
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:...:loadbalancer/app/my-app-alb/... \
  --protocol HTTP \
  --port 80 \
  --default-actions '[{
    "Type": "redirect",
    "RedirectConfig": {
      "Protocol": "HTTPS",
      "Port": "443",
      "StatusCode": "HTTP_301"
    }
  }]'
```

### 6.3 パスベースルーティング

```bash
# パスベースルーティングルールの追加
# /api/* → API サービス
aws elbv2 create-rule \
  --listener-arn arn:aws:elasticloadbalancing:...:listener/... \
  --priority 10 \
  --conditions '[{
    "Field": "path-pattern",
    "Values": ["/api/*"]
  }]' \
  --actions '[{
    "Type": "forward",
    "TargetGroupArn": "arn:aws:elasticloadbalancing:...:targetgroup/api-tg/..."
  }]'

# ホストヘッダベースルーティング
# api.example.com → API サービス
aws elbv2 create-rule \
  --listener-arn arn:aws:elasticloadbalancing:...:listener/... \
  --priority 20 \
  --conditions '[{
    "Field": "host-header",
    "Values": ["api.example.com"]
  }]' \
  --actions '[{
    "Type": "forward",
    "TargetGroupArn": "arn:aws:elasticloadbalancing:...:targetgroup/api-tg/..."
  }]'
```

---

## 7. ログ設定

### 7.1 CloudWatch Logs への出力

```
ECS タスク ログフロー:

コンテナ stdout/stderr
    |
    v
+-------------------+
| awslogs ドライバー |
+-------------------+
    |
    v
+-------------------+     +-------------------+
| CloudWatch Logs   | --> | Logs Insights     |
| /ecs/my-app       |     | でクエリ分析       |
+-------------------+     +-------------------+
    |
    v (サブスクリプション)
+-------------------+
| Lambda / Kinesis  |
| / OpenSearch      |
+-------------------+
```

```
# CloudWatch Logs Insights クエリ例

# エラーログの検索
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 50

# レスポンスタイムの分析
fields @timestamp, @message
| parse @message "response_time=* ms" as response_time
| stats avg(response_time), max(response_time), p99(response_time) by bin(5m)

# HTTP ステータスコード別集計
fields @timestamp, @message
| parse @message "status=*" as status_code
| stats count(*) by status_code
| sort count desc

# 特定のリクエストIDを追跡
fields @timestamp, @message, @logStream
| filter @message like /req-abc123/
| sort @timestamp asc

# メモリ使用量の追跡
fields @timestamp, @message
| parse @message "memory_used=* MB" as memoryUsed
| stats avg(memoryUsed), max(memoryUsed), min(memoryUsed) by bin(5m)
```

### 7.2 FireLens (Fluent Bit) によるログルーティング

```json
{
  "family": "app-with-firelens",
  "containerDefinitions": [
    {
      "name": "log-router",
      "image": "public.ecr.aws/aws-observability/aws-for-fluent-bit:latest",
      "essential": true,
      "firelensConfiguration": {
        "type": "fluentbit",
        "options": {
          "config-file-type": "file",
          "config-file-value": "/fluent-bit/configs/parse-json.conf"
        }
      },
      "memory": 128
    },
    {
      "name": "web",
      "image": "my-app:latest",
      "essential": true,
      "logConfiguration": {
        "logDriver": "awsfirelens",
        "options": {
          "Name": "cloudwatch_logs",
          "region": "ap-northeast-1",
          "log_group_name": "/ecs/my-app",
          "log_stream_prefix": "web-",
          "auto_create_group": "true"
        }
      }
    }
  ]
}
```

### 7.3 複数の送信先へのログルーティング

```json
{
  "name": "web",
  "logConfiguration": {
    "logDriver": "awsfirelens",
    "options": {
      "Name": "kinesis_firehose",
      "region": "ap-northeast-1",
      "delivery_stream": "my-log-stream",
      "retry_limit": "2"
    }
  }
}
```

```ini
# Fluent Bit カスタム設定 (extra.conf)
# CloudWatch Logs と S3 の両方にログを送信

[OUTPUT]
    Name cloudwatch_logs
    Match *
    region ap-northeast-1
    log_group_name /ecs/my-app
    log_stream_prefix firelens-
    auto_create_group true

[OUTPUT]
    Name s3
    Match *
    region ap-northeast-1
    bucket my-log-archive-bucket
    s3_key_format /logs/$TAG/%Y/%m/%d/%H/
    total_file_size 50M
    upload_timeout 60s
    use_put_object On
```

---

## 8. ECS Exec (コンテナへのアクセス)

### 8.1 ECS Exec の有効化

```bash
# サービスで ECS Exec を有効化
aws ecs update-service \
  --cluster my-cluster \
  --service my-web-service \
  --enable-execute-command

# タスクロールに必要な権限を追加
# {
#   "Version": "2012-10-17",
#   "Statement": [
#     {
#       "Effect": "Allow",
#       "Action": [
#         "ssmmessages:CreateControlChannel",
#         "ssmmessages:CreateDataChannel",
#         "ssmmessages:OpenControlChannel",
#         "ssmmessages:OpenDataChannel"
#       ],
#       "Resource": "*"
#     }
#   ]
# }

# コンテナへのアクセス
aws ecs execute-command \
  --cluster my-cluster \
  --task arn:aws:ecs:ap-northeast-1:123456789012:task/my-cluster/abc123 \
  --container web \
  --interactive \
  --command "/bin/sh"

# ECS Exec の状態確認
aws ecs describe-tasks \
  --cluster my-cluster \
  --tasks arn:aws:ecs:...:task/my-cluster/abc123 \
  --query 'tasks[0].containers[*].{
    Name: name,
    ManagedAgents: managedAgents[*].{
      Name: name,
      Status: lastStatus
    }
  }'
```

### 8.2 トラブルシューティング

```bash
# タスクの一覧と状態確認
aws ecs list-tasks \
  --cluster my-cluster \
  --service-name my-web-service \
  --desired-status RUNNING

# タスクの詳細確認
aws ecs describe-tasks \
  --cluster my-cluster \
  --tasks arn:aws:ecs:...:task/... \
  --query 'tasks[0].{
    LastStatus: lastStatus,
    StoppedReason: stoppedReason,
    StopCode: stopCode,
    Containers: containers[*].{
      Name: name,
      LastStatus: lastStatus,
      ExitCode: exitCode,
      Reason: reason,
      HealthStatus: healthStatus
    }
  }'

# 停止したタスクの理由を確認
aws ecs list-tasks \
  --cluster my-cluster \
  --desired-status STOPPED \
  --started-by "ecs-svc/..." | \
  xargs -I {} aws ecs describe-tasks \
    --cluster my-cluster \
    --tasks {} \
    --query 'tasks[*].{TaskArn: taskArn, StoppedReason: stoppedReason}'

# サービスイベントの確認
aws ecs describe-services \
  --cluster my-cluster \
  --services my-web-service \
  --query 'services[0].events[:10]'
```

---

## 9. ECS Service Connect

### 9.1 Service Connect の概要

```
Service Connect の仕組み:

従来 (Cloud Map + App Mesh):
  サービスA --> DNS ルックアップ --> Cloud Map --> サービスB
  + App Mesh Envoy プロキシ (複雑な設定が必要)

Service Connect:
  サービスA --> Service Connect プロキシ --> サービスB
  (ECS が自動的にプロキシを管理)

+---------------------+          +---------------------+
| サービスA            |          | サービスB            |
| +-------+ +-------+ |          | +-------+ +-------+ |
| | App   | | SC    | | -------> | | SC    | | App   | |
| |       | |Proxy  | |          | |Proxy  | |       | |
| +-------+ +-------+ |          | +-------+ +-------+ |
+---------------------+          +---------------------+

メリット:
  - サービス間通信の自動検出
  - ロードバランシング
  - ヘルスチェック
  - リトライ
  - メトリクス収集
  - TLS 暗号化
```

```bash
# Service Connect 用の名前空間作成
aws servicediscovery create-http-namespace \
  --name my-apps \
  --description "Service Connect namespace"

# サービスの作成 (Service Connect 有効)
aws ecs create-service \
  --cluster my-cluster \
  --service-name backend-api \
  --task-definition backend-api:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration '{...}' \
  --service-connect-configuration '{
    "enabled": true,
    "namespace": "my-apps",
    "services": [
      {
        "portName": "http",
        "discoveryName": "backend-api",
        "clientAliases": [
          {
            "port": 8080,
            "dnsName": "backend-api"
          }
        ]
      }
    ]
  }'
```

---

## 10. CloudFormation テンプレート

### 10.1 ECS Fargate 完全構成テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'ECS Fargate 完全構成テンプレート'

Parameters:
  EnvironmentName:
    Type: String
    Default: prod
    AllowedValues: [dev, stg, prod]

  ImageTag:
    Type: String
    Default: latest
    Description: コンテナイメージのタグ

  DesiredCount:
    Type: Number
    Default: 3
    Description: 希望タスク数

Resources:
  # ECS クラスター
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub '${EnvironmentName}-cluster'
      ClusterSettings:
        - Name: containerInsights
          Value: enabled
      CapacityProviders:
        - FARGATE
        - FARGATE_SPOT
      DefaultCapacityProviderStrategy:
        - CapacityProvider: FARGATE
          Weight: 1
          Base: 2
        - CapacityProvider: FARGATE_SPOT
          Weight: 3
          Base: 0

  # ロググループ
  LogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/ecs/${EnvironmentName}-web-app'
      RetentionInDays: 30

  # タスク定義
  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: !Sub '${EnvironmentName}-web-app'
      NetworkMode: awsvpc
      RequiresCompatibilities: [FARGATE]
      Cpu: '512'
      Memory: '1024'
      ExecutionRoleArn: !GetAtt TaskExecutionRole.Arn
      TaskRoleArn: !GetAtt TaskRole.Arn
      ContainerDefinitions:
        - Name: web
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/my-app:${ImageTag}'
          Essential: true
          PortMappings:
            - ContainerPort: 8080
              Protocol: tcp
          Environment:
            - Name: NODE_ENV
              Value: production
            - Name: PORT
              Value: '8080'
          Secrets:
            - Name: DB_PASSWORD
              ValueFrom: !Sub 'arn:aws:secretsmanager:${AWS::Region}:${AWS::AccountId}:secret:${EnvironmentName}/db-password'
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref LogGroup
              awslogs-region: !Ref AWS::Region
              awslogs-stream-prefix: web
          HealthCheck:
            Command:
              - CMD-SHELL
              - curl -f http://localhost:8080/health || exit 1
            Interval: 30
            Timeout: 5
            Retries: 3
            StartPeriod: 60

  # ECS サービス
  Service:
    Type: AWS::ECS::Service
    DependsOn: ALBListener
    Properties:
      ServiceName: !Sub '${EnvironmentName}-web-service'
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: !Ref DesiredCount
      LaunchType: FARGATE
      PlatformVersion: LATEST
      EnableExecuteCommand: true
      NetworkConfiguration:
        AwsvpcConfiguration:
          Subnets:
            - !ImportValue PrivateSubnet1Id
            - !ImportValue PrivateSubnet2Id
          SecurityGroups:
            - !Ref ECSSecurityGroup
          AssignPublicIp: DISABLED
      LoadBalancers:
        - TargetGroupArn: !Ref TargetGroup
          ContainerName: web
          ContainerPort: 8080
      DeploymentConfiguration:
        MaximumPercent: 200
        MinimumHealthyPercent: 100
        DeploymentCircuitBreaker:
          Enable: true
          Rollback: true

  # Auto Scaling
  ScalingTarget:
    Type: AWS::ApplicationAutoScaling::ScalableTarget
    Properties:
      ServiceNamespace: ecs
      ResourceId: !Sub 'service/${ECSCluster}/${Service.Name}'
      ScalableDimension: ecs:service:DesiredCount
      MinCapacity: 2
      MaxCapacity: 20

  CPUScalingPolicy:
    Type: AWS::ApplicationAutoScaling::ScalingPolicy
    Properties:
      PolicyName: cpu-target-tracking
      PolicyType: TargetTrackingScaling
      ScalingTargetId: !Ref ScalingTarget
      TargetTrackingScalingPolicyConfiguration:
        TargetValue: 70.0
        PredefinedMetricSpecification:
          PredefinedMetricType: ECSServiceAverageCPUUtilization
        ScaleInCooldown: 300
        ScaleOutCooldown: 60

  # CloudWatch アラーム
  HighCPUAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${EnvironmentName}-ecs-high-cpu'
      MetricName: CPUUtilization
      Namespace: AWS/ECS
      Statistic: Average
      Period: 300
      EvaluationPeriods: 2
      Threshold: 85
      ComparisonOperator: GreaterThanThreshold
      Dimensions:
        - Name: ClusterName
          Value: !Ref ECSCluster
        - Name: ServiceName
          Value: !GetAtt Service.Name
      AlarmActions:
        - !ImportValue AlertTopicArn

Outputs:
  ClusterName:
    Value: !Ref ECSCluster
    Export:
      Name: !Sub '${EnvironmentName}-ClusterName'

  ServiceName:
    Value: !GetAtt Service.Name
    Export:
      Name: !Sub '${EnvironmentName}-ServiceName'

  ALBDNSName:
    Value: !GetAtt ALB.DNSName
    Description: ALB の DNS 名
```

---

## 11. アンチパターン

### 11.1 latest タグへの依存

```
[悪い例]
タスク定義で image: "my-app:latest" を使用

問題:
  - どのバージョンがデプロイされたか追跡不能
  - ロールバックが困難
  - 同一タスク定義で異なるバージョンが動作する可能性

[良い例]
image: "my-app:v1.2.3" または
image: "my-app:abc123def" (Git SHA)

CI/CD パイプラインで:
  1. イメージをビルド、一意のタグでプッシュ
  2. タスク定義を新しいイメージタグで更新
  3. サービスを更新
```

### 11.2 タスクロールに過剰な権限を付与

**問題点**: `AdministratorAccess` や広範なワイルドカードをタスクロールに付与すると、コンテナが侵害された場合に大きな被害が生じる。

**改善**: タスクが必要とする最小限のリソースとアクションのみを許可する。実行ロール(ECR プル、ログ書込み)とタスクロール(アプリケーションが使うAWSリソース)を明確に分離する。

### 11.3 ヘルスチェックの未設定

```
[悪い例]
ALB のヘルスチェックを / (トップページ) に設定
→ アプリケーションが部分的に機能不全でもヘルシー判定

[良い例]
専用のヘルスチェックエンドポイント /health を実装:
  - DB 接続を確認
  - 外部サービスの疎通を確認
  - メモリ使用量を確認
  - 依存サービスの状態を確認
```

```python
# Flask アプリケーションのヘルスチェック実装例
from flask import Flask, jsonify
import psycopg2
import redis
import os

app = Flask(__name__)

@app.route("/health")
def health_check():
    checks = {}
    overall_healthy = True

    # DB 接続チェック
    try:
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
        conn.close()
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"
        overall_healthy = False

    # Redis 接続チェック
    try:
        r = redis.Redis.from_url(os.environ["REDIS_URL"])
        r.ping()
        checks["cache"] = "healthy"
    except Exception as e:
        checks["cache"] = f"unhealthy: {str(e)}"
        overall_healthy = False

    status_code = 200 if overall_healthy else 503
    return jsonify({
        "status": "healthy" if overall_healthy else "unhealthy",
        "checks": checks
    }), status_code
```

### 11.4 Graceful Shutdown の未実装

```
[悪い例]
SIGTERM シグナルを無視して即座にプロセスが終了
→ 処理中のリクエストが失敗

[良い例]
SIGTERM を受け取ったら:
  1. 新規リクエストの受付を停止
  2. 処理中のリクエストを完了まで待機
  3. DB 接続をクリーンに切断
  4. プロセスを正常終了

ECS の stopTimeout: 120 (デフォルト30秒)
→ SIGTERM 送信後、120秒待ってから SIGKILL
```

```python
# Python の Graceful Shutdown 実装例
import signal
import sys
import time
from http.server import HTTPServer

class GracefulServer:
    def __init__(self, server):
        self.server = server
        self.is_shutting_down = False

        signal.signal(signal.SIGTERM, self.handle_sigterm)
        signal.signal(signal.SIGINT, self.handle_sigterm)

    def handle_sigterm(self, signum, frame):
        print("SIGTERM received, starting graceful shutdown...")
        self.is_shutting_down = True
        # 処理中のリクエスト完了を待つ
        self.server.shutdown()
        # クリーンアップ
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        # DB 接続の切断
        # キャッシュの保存
        # 一時ファイルの削除
        print("Cleanup completed")

    def serve_forever(self):
        self.server.serve_forever()
```

---

## 12. FAQ

### Q1. Fargate と EC2 のどちらを選ぶべきですか？

まず Fargate を検討するのが推奨される。インフラ管理が不要で、セキュリティパッチも自動適用される。GPU が必要、特殊なカーネル設定が必要、EC2 リザーブドインスタンスで大幅なコスト削減が見込める場合に EC2 起動タイプを選択する。

### Q2. ECS サービスのローリングアップデート中にダウンタイムは発生しますか？

`minimumHealthyPercent: 100`、`maximumPercent: 200` に設定すれば、新しいタスクが正常に起動してからデプロイメントが進むため、ダウンタイムは発生しない。ALB のヘルスチェックと連携させることで、トラフィックは常に正常なタスクにのみルーティングされる。

### Q3. ECS タスク内のコンテナ間通信はどうなりますか？

awsvpc ネットワークモードでは、同一タスク内のコンテナは `localhost` で通信できる。例えば Web コンテナから同一タスク内の Redis サイドカーには `localhost:6379` でアクセス可能である。

### Q4. ECS でコンテナのデバッグはどうすればよいですか？

ECS Exec を有効化することで、`aws ecs execute-command` コマンドで実行中のコンテナにシェルアクセスできる。タスクロールに SSM の権限を追加し、サービスで `enableExecuteCommand` を true に設定する必要がある。

### Q5. Fargate のエフェメラルストレージを拡張できますか？

Fargate Platform Version 1.4.0 以降では、タスク定義の `ephemeralStorage` パラメータで 20GB から 200GB まで拡張可能である。デフォルトは 20GB で、追加分にはストレージ料金が発生する。

### Q6. ECS と EKS のどちらを選ぶべきですか？

Kubernetes の経験がない、または AWS 中心のアーキテクチャであれば ECS が適している。ECS は AWS サービスとの統合が深く、学習コストが低い。Kubernetes の経験があり、マルチクラウド/ハイブリッドクラウド戦略がある場合は EKS が適している。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| クラスター | タスクとサービスの論理グループ |
| タスク定義 | コンテナの設計図。CPU/メモリ/イメージ/ポートを定義 |
| サービス | タスクの希望数を維持するスケジューラ |
| Fargate | サーバーレス。インフラ管理不要 |
| EC2 | 自己管理。GPU 対応、コスト最適化が可能 |
| ALB 統合 | awsvpc + ターゲットグループで直接ルーティング |
| ログ | awslogs ドライバーまたは FireLens で CloudWatch Logs へ |
| デプロイ | ローリングアップデート / Blue/Green / サーキットブレーカー |
| ECS Exec | 実行中のコンテナへの対話型アクセス |
| Service Connect | サービス間通信の自動化 |
| Capacity Provider | Fargate/Fargate Spot の割合を柔軟に制御 |

---

## 次に読むべきガイド

- [ECR](./01-ecr.md) -- コンテナイメージの管理
- [EKS 概要](./02-eks-overview.md) -- Kubernetes ベースのオーケストレーション
- [CodePipeline](../07-devops/02-codepipeline.md) -- ECS への CI/CD パイプライン

---

## 参考文献

1. AWS 公式ドキュメント「Amazon ECS デベロッパーガイド」 https://docs.aws.amazon.com/ecs/latest/developerguide/
2. AWS 公式ドキュメント「Amazon ECS ベストプラクティスガイド」 https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/
3. Nathan Peck「Amazon ECS ベストプラクティス」 https://ecsworkshop.com/
4. AWS Containers Blog https://aws.amazon.com/blogs/containers/
5. AWS 公式「ECS Service Connect」 https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-connect.html
