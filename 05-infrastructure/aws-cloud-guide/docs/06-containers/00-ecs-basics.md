# Amazon ECS 基礎

> Amazon Elastic Container Service (ECS) の基本概念であるタスク定義、サービス、Fargate と EC2 起動タイプの違い、ALB 統合、ログ設定までを体系的に学ぶ。

---

## この章で学ぶこと

1. **ECS のアーキテクチャとコア概念** -- クラスター、タスク定義、サービス、タスクの関係を理解する
2. **Fargate と EC2 起動タイプの使い分け** -- 各起動タイプの特徴、コスト、制約を比較し適切に選択する
3. **ALB 統合とログ設定** -- ロードバランサーによるトラフィック分散と CloudWatch Logs へのログ出力を構成する

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

# ターゲット追跡スケーリングポリシー
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
```

---

## 5. ALB 統合

### 5.1 ALB + ECS の構成

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

### 5.2 ALB ヘルスチェックの設定

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
```

---

## 6. ログ設定

### 6.1 CloudWatch Logs への出力

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

```python
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
```

### 6.2 FireLens (Fluent Bit) によるログルーティング

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

---

## 7. アンチパターン

### 7.1 latest タグへの依存

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

### 7.2 タスクロールに過剰な権限を付与

**問題点**: `AdministratorAccess` や広範なワイルドカードをタスクロールに付与すると、コンテナが侵害された場合に大きな被害が生じる。

**改善**: タスクが必要とする最小限のリソースとアクションのみを許可する。実行ロール(ECR プル、ログ書込み)とタスクロール(アプリケーションが使うAWSリソース)を明確に分離する。

---

## 8. FAQ

### Q1. Fargate と EC2 のどちらを選ぶべきですか？

まず Fargate を検討するのが推奨される。インフラ管理が不要で、セキュリティパッチも自動適用される。GPU が必要、特殊なカーネル設定が必要、EC2 リザーブドインスタンスで大幅なコスト削減が見込める場合に EC2 起動タイプを選択する。

### Q2. ECS サービスのローリングアップデート中にダウンタイムは発生しますか？

`minimumHealthyPercent: 100`、`maximumPercent: 200` に設定すれば、新しいタスクが正常に起動してからデプロイメントが進むため、ダウンタイムは発生しない。ALB のヘルスチェックと連携させることで、トラフィックは常に正常なタスクにのみルーティングされる。

### Q3. ECS タスク内のコンテナ間通信はどうなりますか？

awsvpc ネットワークモードでは、同一タスク内のコンテナは `localhost` で通信できる。例えば Web コンテナから同一タスク内の Redis サイドカーには `localhost:6379` でアクセス可能である。

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
