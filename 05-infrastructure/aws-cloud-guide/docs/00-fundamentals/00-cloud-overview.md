# クラウドコンピューティング概要

> インターネット経由でコンピューティングリソースをオンデマンドで利用する仕組みを体系的に理解する

## この章で学ぶこと

1. クラウドコンピューティングの定義と5つの本質的特性を説明できる
2. IaaS / PaaS / SaaS / FaaS / CaaS の責任分界モデルを区別し、適切なサービスを選択できる
3. AWS / GCP / Azure の主要サービスを比較し、プロジェクトに最適なクラウドを判断できる
4. クラウド導入の段階的アプローチと移行戦略を理解できる
5. クラウドネイティブアーキテクチャの基本原則を実践できる

---

## 1. クラウドコンピューティングとは

### 1.1 NIST による定義

米国国立標準技術研究所 (NIST SP 800-145) はクラウドコンピューティングを次のように定義している。

> 共有プールの構成可能なコンピューティングリソース（ネットワーク、サーバー、ストレージ、アプリケーション、サービス）に対して、最小限の管理労力またはサービスプロバイダとのやり取りで迅速にプロビジョニングおよびリリースできる、便利なオンデマンドのネットワークアクセスを可能にするモデル

この定義の要点を分解すると以下のようになる。

```
NIST定義の構成要素:

1. 共有プール (Resource Pooling)
   → 複数のテナントが物理リソースを共有
   → 仮想化技術がこれを実現

2. 構成可能 (Configurable)
   → APIやコンソールからリソースの仕様を柔軟に変更可能
   → CPU、メモリ、ストレージ等を動的に調整

3. オンデマンド (On-Demand)
   → 事前のキャパシティプランニングが不要
   → 必要な時に必要な分だけ確保

4. ネットワークアクセス (Broad Network Access)
   → インターネットまたはVPN経由で場所を問わずアクセス
   → 標準プロトコル（HTTP/HTTPS、SSH等）を使用

5. 迅速なプロビジョニング (Rapid Provisioning)
   → 数分でリソースが利用可能
   → Infrastructure as Code による自動化
```

### 1.2 5つの本質的特性

```
+--------------------------------------------------------------+
|            クラウドの5つの本質的特性 (NIST SP 800-145)           |
+--------------------------------------------------------------+
| 1. オンデマンド・セルフサービス                                   |
|    - 人手を介さずリソースを即時確保                                |
|    - Webコンソール・API・CLIで操作                               |
|    - 承認プロセスなしで開発者が直接プロビジョニング                  |
|                                                              |
| 2. 幅広いネットワークアクセス                                    |
|    - 標準プロトコルでどこからでもアクセス                          |
|    - PC、スマートフォン、タブレット等の多様なデバイス対応            |
|    - 低レイテンシのグローバルネットワーク                          |
|                                                              |
| 3. リソースプーリング                                           |
|    - マルチテナントで物理リソースを共有                            |
|    - ユーザーは物理的な場所を意識しない                            |
|    - 仮想化による論理的なリソース分離                              |
|                                                              |
| 4. 迅速な弾力性 (Rapid Elasticity)                              |
|    - 需要に応じて自動スケール                                    |
|    - スケールアウト（水平拡張）とスケールアップ（垂直拡張）          |
|    - ユーザーからは無限にリソースがあるように見える                  |
|                                                              |
| 5. 従量課金 (Measured Service)                                  |
|    - 使った分だけ支払い                                          |
|    - リソース使用量の計測・監視・報告が自動化                      |
|    - 秒単位・リクエスト単位の課金も可能                            |
+--------------------------------------------------------------+
```

### 1.3 各特性の実務での実現例

```python
# 特性1: オンデマンド・セルフサービス — boto3でEC2インスタンスを即時起動
import boto3

ec2 = boto3.resource('ec2', region_name='ap-northeast-1')

# 開発者が自分のタイミングでリソースを確保
instances = ec2.create_instances(
    ImageId='ami-0abcdef1234567890',
    MinCount=1,
    MaxCount=1,
    InstanceType='t3.micro',
    TagSpecifications=[{
        'ResourceType': 'instance',
        'Tags': [{'Key': 'Name', 'Value': 'dev-server'}]
    }]
)
print(f"インスタンス {instances[0].id} を起動しました")
```

```python
# 特性4: 迅速な弾力性 — Auto Scalingグループの設定
import boto3

autoscaling = boto3.client('autoscaling', region_name='ap-northeast-1')

# 需要に応じて1〜10台の範囲で自動スケール
autoscaling.create_auto_scaling_group(
    AutoScalingGroupName='web-asg',
    LaunchTemplate={
        'LaunchTemplateId': 'lt-0abcdef1234567890',
        'Version': '$Latest'
    },
    MinSize=1,
    MaxSize=10,
    DesiredCapacity=2,
    VPCZoneIdentifier='subnet-abc123,subnet-def456',
    TargetGroupARNs=['arn:aws:elasticloadbalancing:...'],
    Tags=[{
        'Key': 'Environment',
        'Value': 'production',
        'PropagateAtLaunch': True
    }]
)

# スケーリングポリシーの設定（CPU使用率70%を基準）
autoscaling.put_scaling_policy(
    AutoScalingGroupName='web-asg',
    PolicyName='cpu-target-tracking',
    PolicyType='TargetTrackingScaling',
    TargetTrackingConfiguration={
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'ASGAverageCPUUtilization'
        },
        'TargetValue': 70.0,
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)
```

```python
# 特性5: 従量課金 — コスト監視と予算アラート設定
import boto3

budgets = boto3.client('budgets', region_name='us-east-1')

# 月額予算を設定し、閾値超過時に通知
budgets.create_budget(
    AccountId='123456789012',
    Budget={
        'BudgetName': 'monthly-budget',
        'BudgetLimit': {
            'Amount': '1000',
            'Unit': 'USD'
        },
        'BudgetType': 'COST',
        'TimeUnit': 'MONTHLY',
        'TimePeriod': {
            'Start': '2026-01-01T00:00:00Z',
            'End': '2027-01-01T00:00:00Z'
        }
    },
    NotificationsWithSubscribers=[{
        'Notification': {
            'NotificationType': 'ACTUAL',
            'ComparisonOperator': 'GREATER_THAN',
            'Threshold': 80.0,
            'ThresholdType': 'PERCENTAGE'
        },
        'Subscribers': [{
            'SubscriptionType': 'EMAIL',
            'Address': 'admin@example.com'
        }]
    }]
)
```

### 1.4 オンプレミス vs クラウド

```
+------------------------------+      +------------------------------+
|        オンプレミス             |      |          クラウド             |
+------------------------------+      +------------------------------+
| ハードウェア購入 必要           |      | ハードウェア購入 不要          |
| 初期費用 大（CAPEX中心）        |      | 初期費用 小（OPEX中心）        |
| スケール 数週間~数ヶ月          |      | スケール 数分                 |
| 運用 自社が全責任              |      | 運用 共有責任モデル            |
| 減価償却 あり（3~5年）          |      | 減価償却 なし                 |
| キャパシティ 固定的             |      | キャパシティ 弾力的            |
| グローバル展開 困難             |      | グローバル展開 容易            |
| 障害対応 自社                  |      | 障害対応 プロバイダ+自社       |
| セキュリティ 完全自社管理        |      | セキュリティ 共有責任          |
| ライセンス 自前                |      | ライセンス 従量課金 or BYOL    |
+------------------------------+      +------------------------------+
```

### 1.5 TCO（総所有コスト）比較の実務的アプローチ

クラウド移行の投資判断では、単純な月額費用の比較ではなくTCOで評価する。

```
TCO計算の構成要素:

【オンプレミスのTCO】
┌─────────────────────────────────────────────┐
│ ハードウェア費用                               │
│   サーバー本体: ¥2,000,000 × 10台             │
│   ネットワーク機器: ¥5,000,000                │
│   ストレージ: ¥3,000,000                      │
│                                             │
│ ファシリティ費用                               │
│   データセンター利用料: ¥500,000/月            │
│   電力・空調: ¥200,000/月                     │
│                                             │
│ 人件費                                       │
│   インフラエンジニア 2名: ¥1,200,000/月        │
│   24/365オンコール手当: ¥300,000/月            │
│                                             │
│ ソフトウェアライセンス                          │
│   OS、DB、ミドルウェア: ¥2,000,000/年          │
│                                             │
│ 3年間TCO: 約 ¥120,000,000                    │
└─────────────────────────────────────────────┘

【クラウド（AWS）のTCO】
┌─────────────────────────────────────────────┐
│ コンピュート                                  │
│   EC2 (Reserved 3年): ¥300,000/月            │
│   Lambda: ¥50,000/月                         │
│                                             │
│ ストレージ・DB                                │
│   S3 + RDS: ¥150,000/月                      │
│   ElastiCache: ¥80,000/月                    │
│                                             │
│ ネットワーク                                  │
│   データ転送 + VPN: ¥100,000/月               │
│                                             │
│ 人件費（削減後）                               │
│   クラウドエンジニア 1名: ¥700,000/月          │
│                                             │
│ 3年間TCO: 約 ¥60,000,000                     │
└─────────────────────────────────────────────┘
```

```python
# TCO計算スクリプト
def calculate_tco_comparison(years: int = 3):
    """オンプレミスとクラウドのTCO比較"""

    # オンプレミスコスト
    onprem = {
        'hardware': {
            'servers': 2_000_000 * 10,
            'network': 5_000_000,
            'storage': 3_000_000,
        },
        'monthly': {
            'datacenter': 500_000,
            'power_cooling': 200_000,
            'staff': 1_200_000,
            'oncall': 300_000,
        },
        'yearly': {
            'licenses': 2_000_000,
            'maintenance': 1_500_000,  # ハードウェア保守
        }
    }

    onprem_total = (
        sum(onprem['hardware'].values()) +
        sum(onprem['monthly'].values()) * 12 * years +
        sum(onprem['yearly'].values()) * years
    )

    # クラウドコスト
    cloud = {
        'monthly': {
            'compute': 300_000,
            'serverless': 50_000,
            'storage_db': 150_000,
            'cache': 80_000,
            'network': 100_000,
            'staff': 700_000,
        },
        'yearly': {
            'support_plan': 600_000,
            'training': 300_000,
        }
    }

    cloud_total = (
        sum(cloud['monthly'].values()) * 12 * years +
        sum(cloud['yearly'].values()) * years
    )

    savings = onprem_total - cloud_total
    savings_pct = (savings / onprem_total) * 100

    print(f"=== {years}年間TCO比較 ===")
    print(f"オンプレミス: ¥{onprem_total:,.0f}")
    print(f"クラウド:     ¥{cloud_total:,.0f}")
    print(f"削減額:       ¥{savings:,.0f} ({savings_pct:.1f}%)")

    return {
        'onprem': onprem_total,
        'cloud': cloud_total,
        'savings': savings,
        'savings_pct': savings_pct
    }

result = calculate_tco_comparison(3)
# === 3年間TCO比較 ===
# オンプレミス: ¥117,400,000
# クラウド:     ¥52,680,000
# 削減額:       ¥64,720,000 (55.1%)
```

### 1.6 クラウドコンピューティングの歴史的変遷

```
クラウドコンピューティングの進化:

2002年 - AWS内部でインフラの標準化開始
2004年 - Amazon SQS (最初のAWSサービス)
2006年 - Amazon S3, EC2 リリース → IaaSの始まり
2008年 - Google App Engine (PaaSの先駆け)
2009年 - Heroku登場 → PaaS普及
2010年 - Microsoft Azure 正式リリース
2011年 - NIST SP 800-145 発行（クラウドの公式定義）
2012年 - AWS re:Invent 開始、DynamoDB リリース
2013年 - Docker登場 → コンテナ革命
2014年 - AWS Lambda → サーバーレスの始まり
        Kubernetes v1.0 → コンテナオーケストレーション
2015年 - AWS IoT, Machine Learning サービス開始
2016年 - Google Cloud Platform 本格展開
2017年 - クラウドネイティブの概念が普及 (CNCF)
2018年 - マルチクラウド戦略が主流に
2019年 - AWS Outposts → ハイブリッドクラウド強化
2020年 - リモートワーク需要でクラウド加速
2021年 - AWS Graviton3、クラウド支出が初めて$1T超え
2022年 - FinOps の普及、コスト最適化が重要トピックに
2023年 - 生成AI関連サービスの爆発的増加
2024年 - AI/MLワークロードのクラウド移行加速
2025年 - エッジ・クラウド融合、サステナブルクラウド
```

---

## 2. サービスモデル — IaaS / PaaS / SaaS / FaaS / CaaS

### 2.1 責任分界モデル

```
管理責任の範囲（上に行くほどユーザー責任）

  +--------------+---------+---------+---------+---------+---------+
  |              | IaaS    | CaaS    | PaaS    | FaaS    | SaaS    |
  +--------------+---------+---------+---------+---------+---------+
  | アプリ       | ユーザー | ユーザー | ユーザー | ユーザー | ベンダー |
  | データ       | ユーザー | ユーザー | ユーザー | ユーザー | 共有    |
  | ランタイム   | ユーザー | ユーザー | ベンダー | ベンダー | ベンダー |
  | コンテナ     | ユーザー | ベンダー | ベンダー | ベンダー | ベンダー |
  | ミドルウェア | ユーザー | ベンダー | ベンダー | ベンダー | ベンダー |
  | OS           | ユーザー | ベンダー | ベンダー | ベンダー | ベンダー |
  | 仮想化       | ベンダー | ベンダー | ベンダー | ベンダー | ベンダー |
  | サーバー     | ベンダー | ベンダー | ベンダー | ベンダー | ベンダー |
  | ストレージ   | ベンダー | ベンダー | ベンダー | ベンダー | ベンダー |
  | ネットワーク | ベンダー | ベンダー | ベンダー | ベンダー | ベンダー |
  +--------------+---------+---------+---------+---------+---------+

  CaaS = Container as a Service (ECS/EKS, GKE, AKS)
  FaaS = Function as a Service (Lambda, Cloud Functions, Azure Functions)
```

### 2.2 各モデルの代表サービスと詳細比較

| モデル | 概要 | AWS 例 | GCP 例 | Azure 例 | 主要ユースケース |
|--------|------|--------|--------|----------|----------------|
| IaaS | 仮想マシン・ネットワークを提供 | EC2, VPC | Compute Engine | Virtual Machines | フルカスタマイズが必要なワークロード |
| CaaS | コンテナ実行基盤を提供 | ECS, EKS, Fargate | GKE, Cloud Run | AKS, Container Apps | マイクロサービス、CI/CD |
| PaaS | アプリ実行基盤を提供 | Elastic Beanstalk, App Runner | App Engine | App Service | Webアプリ、API |
| FaaS | 関数実行基盤を提供 | Lambda | Cloud Functions | Azure Functions | イベント駆動処理 |
| SaaS | 完成したアプリケーション | WorkMail, Chime | Google Workspace | Microsoft 365 | エンドユーザー向けツール |

### 2.3 サービスモデル選択のフローチャート

```
サービスモデル選択の判断フロー:

Q1: アプリケーションのカスタマイズ度は？
├── 高い（OS・ミドルウェアも制御したい）
│   └── → IaaS (EC2)
│       Q2: コンテナ化されているか？
│       ├── Yes → CaaS (ECS/EKS/Fargate)
│       └── No  → IaaS のまま
│
├── 中程度（アプリコードに集中したい）
│   └── → PaaS (Elastic Beanstalk / App Runner)
│       Q3: 常時起動が必要か？
│       ├── Yes → PaaS
│       └── No  → FaaS (Lambda)
│
└── 低い（既製品で十分）
    └── → SaaS
```

### 2.4 コード例: IaaS (EC2) の起動と初期設定

```bash
# EC2インスタンスを起動する最小コマンド
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.micro \
  --key-name my-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --count 1 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=web-server-01}]'

# インスタンスの状態確認
aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=web-server-01" \
  --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress]' \
  --output table
```

```python
# Terraform (IaC) でEC2を定義する例
# main.tf
"""
resource "aws_instance" "web_server" {
  ami           = "ami-0abcdef1234567890"
  instance_type = "t3.micro"
  subnet_id     = aws_subnet.public.id

  vpc_security_group_ids = [aws_security_group.web.id]

  user_data = <<-EOF
    #!/bin/bash
    yum update -y
    yum install -y httpd
    systemctl start httpd
    systemctl enable httpd
    echo "<h1>Hello from EC2</h1>" > /var/www/html/index.html
  EOF

  tags = {
    Name        = "web-server-01"
    Environment = "production"
    ManagedBy   = "terraform"
  }

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
    encrypted   = true
  }
}
"""
```

### 2.5 コード例: CaaS (ECS Fargate) デプロイ

```json
// ECSタスク定義 (task-definition.json)
{
  "family": "web-app",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "web",
      "image": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/web-app:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/web-app",
          "awslogs-region": "ap-northeast-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "environment": [
        {"name": "NODE_ENV", "value": "production"},
        {"name": "PORT", "value": "8080"}
      ],
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

```bash
# ECSサービスをFargateで作成
aws ecs create-service \
  --cluster production \
  --service-name web-app \
  --task-definition web-app:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[subnet-abc123,subnet-def456],
    securityGroups=[sg-web123],
    assignPublicIp=DISABLED
  }" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=web,containerPort=8080"
```

### 2.6 コード例: PaaS (Elastic Beanstalk) デプロイ

```bash
# Elastic Beanstalk CLIで環境を作成
eb init my-app --platform python-3.11 --region ap-northeast-1
eb create my-env --instance-type t3.small --envvars DATABASE_URL=postgresql://...
eb deploy

# 環境変数の更新
eb setenv SECRET_KEY=my-secret-key DEBUG=false

# ログの確認
eb logs --all
```

```yaml
# .ebextensions/01-packages.config
packages:
  yum:
    postgresql-devel: []

container_commands:
  01_migrate:
    command: "python manage.py migrate"
    leader_only: true
  02_collectstatic:
    command: "python manage.py collectstatic --noinput"

option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: myapp.wsgi:application
  aws:elasticbeanstalk:environment:proxy:staticfiles:
    /static: static
  aws:autoscaling:asg:
    MinSize: 2
    MaxSize: 8
  aws:autoscaling:trigger:
    MeasureName: CPUUtilization
    Statistic: Average
    Unit: Percent
    UpperThreshold: 70
    LowerThreshold: 30
```

### 2.7 コード例: FaaS (Lambda) 関数

```python
# Lambda関数: S3にアップロードされた画像をリサイズ
import json
import boto3
from PIL import Image
import io
import os

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """S3イベントトリガーで画像をリサイズする"""

    # イベントからバケット名とキーを取得
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # 既にリサイズ済みの画像はスキップ
    if key.startswith('thumbnails/'):
        return {'statusCode': 200, 'body': 'Already processed'}

    try:
        # 元画像をダウンロード
        response = s3.get_object(Bucket=bucket, Key=key)
        image_content = response['Body'].read()

        # リサイズ処理
        image = Image.open(io.BytesIO(image_content))
        image.thumbnail((300, 300), Image.LANCZOS)

        # リサイズした画像をバッファに保存
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        buffer.seek(0)

        # サムネイルをS3にアップロード
        thumbnail_key = f"thumbnails/{os.path.basename(key)}"
        s3.put_object(
            Bucket=bucket,
            Key=thumbnail_key,
            Body=buffer,
            ContentType='image/jpeg',
            CacheControl='max-age=31536000'
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Thumbnail created',
                'source': key,
                'thumbnail': thumbnail_key,
                'original_size': f"{image.size[0]}x{image.size[1]}"
            })
        }

    except Exception as e:
        print(f"Error processing {key}: {str(e)}")
        raise
```

```yaml
# AWS SAM テンプレート (template.yaml)
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Image Resize Lambda Function

Globals:
  Function:
    Timeout: 30
    Runtime: python3.11
    MemorySize: 512
    Architectures:
      - arm64

Resources:
  ImageBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub "${AWS::StackName}-images"

  ResizeFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: resize.lambda_handler
      CodeUri: src/
      Policies:
        - S3CrudPolicy:
            BucketName: !Ref ImageBucket
      Events:
        S3Upload:
          Type: S3
          Properties:
            Bucket: !Ref ImageBucket
            Events: s3:ObjectCreated:*
            Filter:
              S3Key:
                Rules:
                  - Name: prefix
                    Value: uploads/
                  - Name: suffix
                    Value: .jpg
```

### 2.8 コード例: SaaS 連携 (AWS SES メール送信)

```python
import boto3
from botocore.exceptions import ClientError

def send_templated_email(
    sender: str,
    recipient: str,
    template_name: str,
    template_data: dict
) -> dict:
    """SESテンプレートメールを送信する"""

    client = boto3.client('ses', region_name='ap-northeast-1')

    try:
        response = client.send_templated_email(
            Source=sender,
            Destination={
                'ToAddresses': [recipient],
            },
            Template=template_name,
            TemplateData=json.dumps(template_data),
            ConfigurationSetName='tracking-config',
            Tags=[
                {'Name': 'campaign', 'Value': 'welcome'},
                {'Name': 'environment', 'Value': 'production'},
            ]
        )
        print(f"Email sent! Message ID: {response['MessageId']}")
        return response

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'MessageRejected':
            print(f"メール拒否: {e.response['Error']['Message']}")
        elif error_code == 'MailFromDomainNotVerifiedException':
            print("送信元ドメインが未検証です")
        else:
            print(f"予期しないエラー: {error_code}")
        raise

# 使用例
import json
send_templated_email(
    sender='noreply@example.com',
    recipient='user@example.com',
    template_name='WelcomeEmail',
    template_data={
        'name': '田中太郎',
        'company': 'Example Corp',
        'login_url': 'https://app.example.com/login'
    }
)
```

---

## 3. デプロイメントモデル

### 3.1 4つのデプロイメントモデル

| モデル | 説明 | ユースケース | メリット | デメリット |
|--------|------|-------------|---------|-----------|
| パブリッククラウド | プロバイダが所有・運用する共有インフラ | スタートアップ、Webサービス | 初期費用なし、即座にスケール | データ所在地の制約 |
| プライベートクラウド | 単一組織専用のクラウドインフラ | 金融機関、政府機関 | セキュリティ・コンプライアンス | 構築・運用コスト大 |
| ハイブリッドクラウド | パブリック+オンプレミスの組合せ | 段階的移行、規制対応 | 柔軟性 | 構成の複雑さ |
| マルチクラウド | 複数プロバイダの組合せ | ベンダーロックイン回避 | 最適なサービス選択 | 運用の複雑さ |

### 3.2 ハイブリッドクラウドのアーキテクチャパターン

```
ハイブリッドクラウド構成例:

  ┌─────────────────────────────┐
  │      オンプレミス             │
  │  ┌─────────┐  ┌──────────┐  │
  │  │ 基幹DB   │  │ 認証基盤  │  │
  │  │(Oracle)  │  │(AD)      │  │
  │  └────┬─────┘  └─────┬────┘  │
  │       │              │       │
  └───────┼──────────────┼───────┘
          │   VPN/         │
          │  Direct Connect│
  ┌───────┼──────────────┼───────┐
  │       ▼              ▼       │
  │  ┌─────────┐  ┌──────────┐  │
  │  │ Aurora   │  │ Cognito  │  │
  │  │ (移行先) │  │ (連携)   │  │
  │  └─────────┘  └──────────┘  │
  │                              │
  │  ┌─────────┐  ┌──────────┐  │
  │  │ ECS     │  │ S3       │  │
  │  │ (API)   │  │ (資産)   │  │
  │  └─────────┘  └──────────┘  │
  │        AWS クラウド           │
  └──────────────────────────────┘
```

```bash
# AWS Direct Connect + VPNの構成例
# Direct Connect Gateway の作成
aws directconnect create-direct-connect-gateway \
  --direct-connect-gateway-name "onprem-gateway" \
  --amazon-side-asn 64512

# VPN接続の作成（バックアップ用）
aws ec2 create-vpn-gateway \
  --type ipsec.1 \
  --amazon-side-asn 65000

aws ec2 create-customer-gateway \
  --type ipsec.1 \
  --public-ip 203.0.113.1 \
  --bgp-asn 65100

aws ec2 create-vpn-connection \
  --type ipsec.1 \
  --vpn-gateway-id vgw-abc123 \
  --customer-gateway-id cgw-def456 \
  --options '{"StaticRoutesOnly": false}'
```

### 3.3 マルチクラウド戦略の実装

```python
# マルチクラウド抽象化レイヤーの例
from abc import ABC, abstractmethod
from typing import BinaryIO

class CloudStorageProvider(ABC):
    """クラウドストレージの抽象基底クラス"""

    @abstractmethod
    def upload(self, bucket: str, key: str, data: BinaryIO) -> str:
        pass

    @abstractmethod
    def download(self, bucket: str, key: str) -> bytes:
        pass

    @abstractmethod
    def delete(self, bucket: str, key: str) -> bool:
        pass

    @abstractmethod
    def list_objects(self, bucket: str, prefix: str = '') -> list:
        pass

class AWSS3Provider(CloudStorageProvider):
    def __init__(self, region: str = 'ap-northeast-1'):
        import boto3
        self.client = boto3.client('s3', region_name=region)

    def upload(self, bucket: str, key: str, data: BinaryIO) -> str:
        self.client.upload_fileobj(data, bucket, key)
        return f"s3://{bucket}/{key}"

    def download(self, bucket: str, key: str) -> bytes:
        response = self.client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()

    def delete(self, bucket: str, key: str) -> bool:
        self.client.delete_object(Bucket=bucket, Key=key)
        return True

    def list_objects(self, bucket: str, prefix: str = '') -> list:
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]

class GCPStorageProvider(CloudStorageProvider):
    def __init__(self):
        from google.cloud import storage
        self.client = storage.Client()

    def upload(self, bucket: str, key: str, data: BinaryIO) -> str:
        bucket_obj = self.client.bucket(bucket)
        blob = bucket_obj.blob(key)
        blob.upload_from_file(data)
        return f"gs://{bucket}/{key}"

    def download(self, bucket: str, key: str) -> bytes:
        bucket_obj = self.client.bucket(bucket)
        blob = bucket_obj.blob(key)
        return blob.download_as_bytes()

    def delete(self, bucket: str, key: str) -> bool:
        bucket_obj = self.client.bucket(bucket)
        blob = bucket_obj.blob(key)
        blob.delete()
        return True

    def list_objects(self, bucket: str, prefix: str = '') -> list:
        bucket_obj = self.client.bucket(bucket)
        blobs = bucket_obj.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]

class AzureBlobProvider(CloudStorageProvider):
    def __init__(self, connection_string: str):
        from azure.storage.blob import BlobServiceClient
        self.client = BlobServiceClient.from_connection_string(connection_string)

    def upload(self, bucket: str, key: str, data: BinaryIO) -> str:
        blob_client = self.client.get_blob_client(bucket, key)
        blob_client.upload_blob(data, overwrite=True)
        return f"https://{self.client.account_name}.blob.core.windows.net/{bucket}/{key}"

    def download(self, bucket: str, key: str) -> bytes:
        blob_client = self.client.get_blob_client(bucket, key)
        return blob_client.download_blob().readall()

    def delete(self, bucket: str, key: str) -> bool:
        blob_client = self.client.get_blob_client(bucket, key)
        blob_client.delete_blob()
        return True

    def list_objects(self, bucket: str, prefix: str = '') -> list:
        container = self.client.get_container_client(bucket)
        blobs = container.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]

# ファクトリパターンでプロバイダを選択
def get_storage_provider(cloud: str, **kwargs) -> CloudStorageProvider:
    providers = {
        'aws': AWSS3Provider,
        'gcp': GCPStorageProvider,
        'azure': AzureBlobProvider,
    }
    if cloud not in providers:
        raise ValueError(f"Unsupported cloud: {cloud}")
    return providers[cloud](**kwargs)

# 使用例
provider = get_storage_provider('aws', region='ap-northeast-1')
url = provider.upload('my-bucket', 'data/report.csv', open('report.csv', 'rb'))
```

---

## 4. AWS vs GCP vs Azure — 主要サービス比較

### 4.1 コンピュート比較

| カテゴリ | AWS | GCP | Azure | 特記事項 |
|----------|-----|-----|-------|---------|
| 仮想マシン | EC2 | Compute Engine | Virtual Machines | AWS: 最多インスタンスタイプ |
| コンテナ(マネージド) | ECS / EKS | GKE | AKS | GKE: Kubernetes発祥 |
| サーバーレス | Lambda | Cloud Functions | Azure Functions | AWS: 最多トリガーソース |
| コンテナサーバーレス | Fargate | Cloud Run | Container Apps | Cloud Run: 最も簡単 |
| バッチ処理 | AWS Batch | Cloud Batch | Azure Batch | 大量並列処理向け |
| エッジコンピュート | Lambda@Edge | Cloud CDN Functions | Azure Front Door | CDN統合型 |

### 4.2 ストレージ/DB比較

| カテゴリ | AWS | GCP | Azure | 特記事項 |
|----------|-----|-----|-------|---------|
| オブジェクトストレージ | S3 | Cloud Storage | Blob Storage | S3: 業界標準API |
| ブロックストレージ | EBS | Persistent Disk | Managed Disks | VM用高性能ストレージ |
| ファイルストレージ | EFS, FSx | Filestore | Azure Files | NFS/SMB対応 |
| RDB(マネージド) | RDS / Aurora | Cloud SQL / AlloyDB | Azure SQL | Aurora: 高性能 |
| NoSQL(ドキュメント) | DynamoDB | Firestore | Cosmos DB | DynamoDB: シングルdigit ms |
| NoSQL(ワイドカラム) | Keyspaces | Bigtable | Table Storage | 大規模時系列データ向け |
| キャッシュ | ElastiCache | Memorystore | Azure Cache for Redis | Redis/Memcached互換 |
| データウェアハウス | Redshift | BigQuery | Synapse Analytics | BigQuery: サーバーレス |
| 検索 | OpenSearch | -- | Cognitive Search | Elasticsearch互換 |

### 4.3 ネットワーキング比較

| カテゴリ | AWS | GCP | Azure | 特記事項 |
|----------|-----|-----|-------|---------|
| VPC | VPC | VPC | VNet | 仮想ネットワーク |
| CDN | CloudFront | Cloud CDN | Azure CDN | グローバル配信 |
| DNS | Route 53 | Cloud DNS | Azure DNS | ドメイン管理 |
| ロードバランサー | ALB/NLB/CLB | Cloud Load Balancing | Azure LB | L4/L7対応 |
| VPN | Site-to-Site VPN | Cloud VPN | VPN Gateway | 暗号化通信 |
| 専用線 | Direct Connect | Cloud Interconnect | ExpressRoute | 低レイテンシ |
| API Gateway | API Gateway | Apigee / API Gateway | API Management | REST/WebSocket対応 |

### 4.4 AI/ML サービス比較

| カテゴリ | AWS | GCP | Azure | 特記事項 |
|----------|-----|-----|-------|---------|
| ML プラットフォーム | SageMaker | Vertex AI | Azure ML | フルマネージドML |
| 画像認識 | Rekognition | Vision AI | Computer Vision | 事前学習済みモデル |
| 自然言語処理 | Comprehend | Natural Language AI | Text Analytics | テキスト分析 |
| 音声認識 | Transcribe | Speech-to-Text | Speech Services | 文字起こし |
| 翻訳 | Translate | Translation AI | Translator | 多言語対応 |
| 生成AI | Bedrock | Vertex AI (Gemini) | Azure OpenAI | LLM統合基盤 |

### 4.5 コード例: 各クラウドの CLI でバケット作成

```bash
# AWS
aws s3 mb s3://my-bucket-2026 --region ap-northeast-1
aws s3api put-bucket-versioning \
  --bucket my-bucket-2026 \
  --versioning-configuration Status=Enabled
aws s3api put-bucket-encryption \
  --bucket my-bucket-2026 \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "aws:kms"}}]
  }'

# GCP
gsutil mb -l asia-northeast1 gs://my-bucket-2026
gsutil versioning set on gs://my-bucket-2026

# Azure
az storage account create \
  --name mystorageaccount \
  --resource-group myResourceGroup \
  --location japaneast \
  --sku Standard_LRS
az storage container create \
  --name my-bucket-2026 \
  --account-name mystorageaccount
```

### 4.6 コード例: 各クラウドの SDK — ファイルアップロード (Python)

```python
# === AWS S3 ===
import boto3

s3 = boto3.client('s3')

# アップロード（メタデータ付き）
s3.upload_file(
    'local.txt',
    'my-bucket',
    'remote.txt',
    ExtraArgs={
        'ContentType': 'text/plain',
        'ServerSideEncryption': 'aws:kms',
        'Metadata': {
            'uploaded-by': 'automation',
            'version': '1.0'
        }
    }
)

# プリサインドURLの生成（一時的な共有用）
presigned_url = s3.generate_presigned_url(
    'get_object',
    Params={'Bucket': 'my-bucket', 'Key': 'remote.txt'},
    ExpiresIn=3600  # 1時間有効
)
print(f"共有URL: {presigned_url}")

# === GCP Cloud Storage ===
from google.cloud import storage

client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('remote.txt')
blob.metadata = {'uploaded-by': 'automation', 'version': '1.0'}
blob.upload_from_filename('local.txt', content_type='text/plain')

# 署名付きURLの生成
signed_url = blob.generate_signed_url(
    version='v4',
    expiration=3600,
    method='GET'
)

# === Azure Blob Storage ===
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from datetime import datetime, timedelta

blob_service = BlobServiceClient.from_connection_string(conn_str)
blob_client = blob_service.get_blob_client('my-container', 'remote.txt')

with open('local.txt', 'rb') as data:
    blob_client.upload_blob(
        data,
        content_settings={'content_type': 'text/plain'},
        metadata={'uploaded-by': 'automation', 'version': '1.0'}
    )

# SAS URLの生成
sas_token = generate_blob_sas(
    account_name='mystorageaccount',
    container_name='my-container',
    blob_name='remote.txt',
    account_key='...',
    permission=BlobSasPermissions(read=True),
    expiry=datetime.utcnow() + timedelta(hours=1)
)
```

### 4.7 リージョンとアベイラビリティゾーン比較

```
主要クラウドのアジア太平洋リージョン:

AWS:
├── ap-northeast-1 (東京)      - 4 AZ
├── ap-northeast-2 (ソウル)     - 4 AZ
├── ap-northeast-3 (大阪)      - 3 AZ
├── ap-southeast-1 (シンガポール) - 3 AZ
├── ap-southeast-2 (シドニー)    - 3 AZ
├── ap-south-1 (ムンバイ)       - 3 AZ
└── 他多数...

GCP:
├── asia-northeast1 (東京)      - 3 Zone
├── asia-northeast2 (大阪)      - 3 Zone
├── asia-northeast3 (ソウル)     - 3 Zone
├── asia-southeast1 (シンガポール) - 3 Zone
└── asia-south1 (ムンバイ)       - 3 Zone

Azure:
├── Japan East (東京)           - 3 AZ
├── Japan West (大阪)           - 3 AZ
├── Southeast Asia (シンガポール) - 3 AZ
├── East Asia (香港)            - 3 AZ
└── Korea Central (ソウル)       - 3 AZ
```

---

## 5. クラウドネイティブアーキテクチャ

### 5.1 CNCF の定義

Cloud Native Computing Foundation (CNCF) はクラウドネイティブを次のように定義している。

```
クラウドネイティブ技術は、パブリッククラウド、プライベートクラウド、
ハイブリッドクラウドなどの近代的でダイナミックな環境において、
スケーラブルなアプリケーションを構築および実行するための能力を
組織にもたらす。

このアプローチの代表例:
- コンテナ
- サービスメッシュ
- マイクロサービス
- イミュータブルインフラストラクチャ
- 宣言型API

これらの手法により、回復力があり、管理しやすく、
可観測性の高い疎結合システムが実現される。
```

### 5.2 12 Factor App

クラウドネイティブアプリケーション設計の基本原則。

```
12 Factor App:

1.  コードベース       - バージョン管理された1つのコードベース、複数デプロイ
2.  依存関係           - 依存関係を明示的に宣言して分離する
3.  設定               - 環境変数に設定を格納する
4.  バッキングサービス  - バッキングサービスをアタッチされたリソースとして扱う
5.  ビルド・リリース・実行 - ビルドステージとランステージを厳密に分離する
6.  プロセス           - アプリをステートレスなプロセスとして実行する
7.  ポートバインディング - ポートバインディングでサービスを公開する
8.  並行性             - プロセスモデルでスケールアウトする
9.  廃棄容易性         - 高速起動とグレースフルシャットダウンで堅牢性を高める
10. 開発/本番一致      - 開発・ステージング・本番をできるだけ一致させる
11. ログ               - ログをイベントストリームとして扱う
12. 管理プロセス       - 管理タスクを1回限りのプロセスとして実行する
```

```python
# 12 Factor App 準拠のアプリケーション例

# Factor 3: 設定は環境変数から
import os

class Config:
    DATABASE_URL = os.environ['DATABASE_URL']        # 必須
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    SECRET_KEY = os.environ['SECRET_KEY']
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    PORT = int(os.environ.get('PORT', '8080'))

# Factor 4: バッキングサービスをアタッチされたリソースとして
# → DB接続先は環境変数で注入、コード変更なしで切替可能

# Factor 6: ステートレスプロセス
# → セッション情報はRedisに保持、ローカルファイルに書かない

# Factor 7: ポートバインディング
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=Config.PORT)

# Factor 9: 廃棄容易性 — グレースフルシャットダウン
import signal
import sys

def graceful_shutdown(signum, frame):
    print("Shutting down gracefully...")
    # DB接続のクローズ、進行中のリクエスト完了を待つ
    sys.exit(0)

signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)

# Factor 11: ログはstdoutに出力
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
        }
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)

handler = logging.StreamHandler(sys.stdout)  # stdoutに出力
handler.setFormatter(JSONFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(Config.LOG_LEVEL)
```

### 5.3 マイクロサービスアーキテクチャパターン

```
マイクロサービスアーキテクチャ on AWS:

                     ┌─── CloudFront (CDN)
                     │
                     ▼
                API Gateway
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ User   │  │ Order  │  │ Product│
   │ Service│  │ Service│  │ Service│
   │ (ECS)  │  │ (ECS)  │  │ (Lambda)│
   └───┬────┘  └───┬────┘  └───┬────┘
       │           │           │
       ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ Aurora │  │DynamoDB│  │ Aurora  │
   │ MySQL  │  │        │  │ (Reader)│
   └────────┘  └────────┘  └────────┘
       │           │           │
       └─────────┬─┘───────────┘
                 ▼
            EventBridge
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   ┌────────┐ ┌────┐ ┌────────┐
   │ SNS/SQS│ │ SES│ │CloudWatch│
   │ (通知) │ │(Mail)│ │(監視)  │
   └────────┘ └────┘ └────────┘
```

---

## 6. クラウド導入のメリットと課題

### 6.1 メリットの詳細

```
  メリット                              具体的な効果
  +----------------------------------+  +----------------------------------+
  | 1. 初期投資の削減 (CAPEX→OPEX)    |  | サーバー購入不要、月額課金         |
  | 2. グローバル展開の容易さ          |  | 25+リージョンに数分でデプロイ      |
  | 3. 高可用性・耐障害性             |  | マルチAZ構成で99.99% SLA          |
  | 4. 自動スケーリング               |  | ピーク時に自動拡張、閑散時に縮退    |
  | 5. マネージドサービス活用          |  | DB運用、パッチ適用をプロバイダに委託 |
  | 6. 開発スピードの向上             |  | 環境構築が数分、CI/CD統合容易      |
  | 7. イノベーションの加速           |  | AI/ML、IoT等の先端技術を即利用     |
  | 8. セキュリティの強化             |  | 暗号化、監査、コンプライアンス標準  |
  +----------------------------------+  +----------------------------------+

  課題                                  対策
  +----------------------------------+  +----------------------------------+
  | 1. ベンダーロックイン             |  | コンテナ化、IaCで抽象化           |
  | 2. データ主権・コンプライアンス    |  | リージョン選択、暗号化            |
  | 3. ネットワーク遅延               |  | エッジロケーション、CDN活用       |
  | 4. コスト管理の複雑さ             |  | FinOps実践、予算アラート          |
  | 5. セキュリティ責任の理解          |  | 共有責任モデルの教育              |
  | 6. 既存スキルのギャップ           |  | 認定資格取得、トレーニング        |
  | 7. 移行の複雑さ                  |  | 段階的移行計画、MAP活用           |
  | 8. 障害時の対応範囲の限界         |  | マルチリージョン、DR計画          |
  +----------------------------------+  +----------------------------------+
```

### 6.2 コスト最適化戦略

```python
# AWSコスト最適化ツール: Cost Explorer APIの活用
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce', region_name='us-east-1')

def get_monthly_cost_breakdown():
    """月別・サービス別コスト内訳を取得"""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    response = ce.get_cost_and_usage(
        TimePeriod={'Start': start, 'End': end},
        Granularity='MONTHLY',
        Metrics=['UnblendedCost'],
        GroupBy=[
            {'Type': 'DIMENSION', 'Key': 'SERVICE'}
        ]
    )

    for period in response['ResultsByTime']:
        print(f"\n=== {period['TimePeriod']['Start']} ===")
        groups = sorted(
            period['Groups'],
            key=lambda x: float(x['Metrics']['UnblendedCost']['Amount']),
            reverse=True
        )
        for group in groups[:10]:
            service = group['Keys'][0]
            cost = float(group['Metrics']['UnblendedCost']['Amount'])
            if cost > 0:
                print(f"  {service}: ${cost:.2f}")

def get_savings_recommendations():
    """Savings Plans / Reserved Instances の推奨を取得"""
    response = ce.get_savings_plans_purchase_recommendation(
        SavingsPlansType='COMPUTE_SP',
        TermInYears='ONE_YEAR',
        PaymentOption='NO_UPFRONT',
        LookbackPeriodInDays='SIXTY_DAYS'
    )

    recommendation = response['SavingsPlansRecommendationDetails']
    for rec in recommendation[:5]:
        print(f"推奨月額: ${float(rec['HourlyCommitmentToPurchase']) * 730:.2f}")
        print(f"推定節約額: ${float(rec['EstimatedMonthlySavingsAmount']):.2f}")
        print(f"節約率: {float(rec['EstimatedSavingsPercentage']):.1f}%")
        print("---")

get_monthly_cost_breakdown()
get_savings_recommendations()
```

```
コスト最適化の4つの柱:

1. 適正サイジング (Right Sizing)
   ┌────────────────────────────────────────────┐
   │ • CloudWatchでCPU/メモリ使用率を監視        │
   │ • Compute Optimizerで推奨サイズを確認        │
   │ • 過剰プロビジョニングを発見・修正           │
   │ • 例: t3.xlarge → t3.medium で 50%削減     │
   └────────────────────────────────────────────┘

2. 予約割引 (Reserved / Savings Plans)
   ┌────────────────────────────────────────────┐
   │ • 1年/3年の利用コミットで最大72%割引         │
   │ • Savings Plans: コンピュート全般に適用      │
   │ • Reserved Instances: 特定インスタンスに適用 │
   │ • 安定稼働のワークロードに最適               │
   └────────────────────────────────────────────┘

3. スポット活用 (Spot Instances)
   ┌────────────────────────────────────────────┐
   │ • オンデマンドの最大90%割引                  │
   │ • バッチ処理、CI/CD、開発環境に最適          │
   │ • 中断に対する耐性が必要                     │
   │ • EC2 Fleet / Spot Fleet で分散            │
   └────────────────────────────────────────────┘

4. アーキテクチャ最適化
   ┌────────────────────────────────────────────┐
   │ • サーバーレス化でアイドルコスト排除          │
   │ • S3ライフサイクルポリシーでストレージ階層化  │
   │ • CloudFront キャッシュでデータ転送削減      │
   │ • Gravitonインスタンスで価格性能比20%向上    │
   └────────────────────────────────────────────┘
```

---

## 7. クラウド移行戦略

### 7.1 AWS 6R 移行戦略

```
6つの移行戦略 (The 6 R's):

1. Rehost (リホスト) — "Lift and Shift"
   ┌──────────────────────────────────────────┐
   │ そのままクラウドに移行                      │
   │ 最小限の変更で迅速な移行が可能              │
   │ 例: オンプレEC2 → AWS EC2                  │
   │ 適用: 移行スピード重視、短期的なコスト削減   │
   └──────────────────────────────────────────┘

2. Replatform (リプラットフォーム) — "Lift, Tinker, and Shift"
   ┌──────────────────────────────────────────┐
   │ 一部をマネージドサービスに置換              │
   │ アプリコアは変更しない                      │
   │ 例: MySQL → Amazon RDS MySQL              │
   │ 適用: コスト対効果のバランス                │
   └──────────────────────────────────────────┘

3. Repurchase (リパーチェス) — "Drop and Shop"
   ┌──────────────────────────────────────────┐
   │ SaaS製品に置換                             │
   │ 例: 自社メールサーバー → WorkMail/Gmail     │
   │ 適用: コモディティ化された機能              │
   └──────────────────────────────────────────┘

4. Refactor (リファクタ) — "Re-architect"
   ┌──────────────────────────────────────────┐
   │ クラウドネイティブに再設計                   │
   │ マイクロサービス化、サーバーレス化           │
   │ 例: モノリス → Lambda + API Gateway + DynamoDB│
   │ 適用: 長期的な最適化、新機能追加が必要       │
   └──────────────────────────────────────────┘

5. Retire (リタイア)
   ┌──────────────────────────────────────────┐
   │ 不要なアプリケーションを廃止                │
   │ 移行コスト削減、ポートフォリオ整理          │
   │ 適用: 使われていないシステムの特定・廃止     │
   └──────────────────────────────────────────┘

6. Retain (リテイン)
   ┌──────────────────────────────────────────┐
   │ 現状維持（今は移行しない）                  │
   │ 技術的制約や規制要件で移行不可              │
   │ 適用: 移行の優先度が低いシステム            │
   └──────────────────────────────────────────┘
```

### 7.2 移行フェーズ

```python
# AWS Migration Hub を使った移行進捗管理
import boto3

mh = boto3.client('migration-hub', region_name='us-west-2')

# 移行タスクの作成
def create_migration_task(app_name: str, strategy: str):
    """移行タスクを作成してトラッキングする"""

    mh.notify_migration_task_state(
        ProgressUpdateStream='MyMigrationStream',
        MigrationTaskName=f'{app_name}-migration',
        Task={
            'Status': 'IN_PROGRESS',
            'StatusDetail': f'Strategy: {strategy}',
            'ProgressPercent': 0
        },
        UpdateDateTime=datetime.now(),
        NextUpdateSeconds=3600
    )

    return f'{app_name}-migration'

# 移行進捗の更新
def update_migration_progress(task_name: str, percent: int, detail: str):
    mh.notify_migration_task_state(
        ProgressUpdateStream='MyMigrationStream',
        MigrationTaskName=task_name,
        Task={
            'Status': 'IN_PROGRESS' if percent < 100 else 'COMPLETED',
            'StatusDetail': detail,
            'ProgressPercent': percent
        },
        UpdateDateTime=datetime.now()
    )

# 使用例
task = create_migration_task('legacy-crm', 'Replatform')
update_migration_progress(task, 25, 'データベース移行中')
update_migration_progress(task, 50, 'アプリケーションデプロイ中')
update_migration_progress(task, 75, 'テスト実施中')
update_migration_progress(task, 100, '移行完了・本番稼働確認済み')
```

---

## 8. アンチパターン

### アンチパターン 1: リフト&シフトで終わらせる

オンプレミスの構成をそのままクラウドに移すだけでは、クラウドのメリット（自動スケーリング、マネージドサービス）を活かせず、むしろコストが高くなるケースが多い。移行後に「クラウドネイティブ化」のフェーズを計画すべきである。

```
# 悪い例: オンプレと同じ構成をそのまま再現
EC2 (常時起動 x 10台) + 自前 MySQL on EC2 + 自前 Redis on EC2
月額コスト: 約$5,000 (管理工数別)
↓
# 良い例: マネージドサービスを活用（Phase 2 最適化）
Fargate (Auto Scaling) + Aurora Serverless v2 + ElastiCache
月額コスト: 約$2,000 (管理工数大幅削減)
```

### アンチパターン 2: 全てをひとつのクラウドアカウントで運用する

本番環境・開発環境・ステージング環境を単一アカウントで管理すると、権限分離やコスト把握が困難になる。AWS Organizations で環境ごとにアカウントを分離すべきである。

```
# 悪い例
1つのAWSアカウントに全環境を配置
→ 開発者が本番DBを誤って削除するリスク
→ コストの環境別内訳が不明確
↓
# 良い例: マルチアカウント戦略
AWS Organizations
├── Management Account (請求・ガバナンス)
│   └── AWS SSO, CloudTrail, Config
├── Security Account (セキュリティ集約)
│   └── GuardDuty, Security Hub, CloudTrail集約
├── Shared Services Account (共通基盤)
│   └── ECR, CodePipeline, Transit Gateway
├── Production Account
│   └── 本番ワークロード (最小権限)
├── Staging Account
│   └── ステージング環境
└── Development Account
    └── 開発者用 (比較的緩い権限)
```

### アンチパターン 3: セキュリティグループを全開放する

```
# 悪い例: 全ポート・全IPを許可
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --protocol -1 \
  --cidr 0.0.0.0/0
# → 全世界からの全通信を許可、重大なセキュリティリスク

# 良い例: 最小権限の原則
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx \
  --ip-permissions '[
    {"IpProtocol":"tcp","FromPort":443,"ToPort":443,"IpRanges":[{"CidrIp":"0.0.0.0/0","Description":"HTTPS"}]},
    {"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"10.0.0.0/8","Description":"Internal SSH"}]}
  ]'
```

### アンチパターン 4: ログ・モニタリングを後回しにする

```python
# 良い例: 初期段階からObservabilityを組み込む
import boto3

# CloudWatch アラームの設定
cloudwatch = boto3.client('cloudwatch', region_name='ap-northeast-1')

def setup_essential_alarms(instance_id: str):
    """EC2インスタンスの基本アラームをセットアップ"""

    alarms = [
        {
            'AlarmName': f'{instance_id}-high-cpu',
            'MetricName': 'CPUUtilization',
            'Namespace': 'AWS/EC2',
            'Statistic': 'Average',
            'Period': 300,
            'EvaluationPeriods': 3,
            'Threshold': 80.0,
            'ComparisonOperator': 'GreaterThanThreshold',
        },
        {
            'AlarmName': f'{instance_id}-status-check',
            'MetricName': 'StatusCheckFailed',
            'Namespace': 'AWS/EC2',
            'Statistic': 'Maximum',
            'Period': 60,
            'EvaluationPeriods': 2,
            'Threshold': 0,
            'ComparisonOperator': 'GreaterThanThreshold',
        },
    ]

    for alarm_config in alarms:
        cloudwatch.put_metric_alarm(
            **alarm_config,
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            AlarmActions=['arn:aws:sns:ap-northeast-1:123456789012:alerts'],
            TreatMissingData='breaching'
        )
        print(f"アラーム作成: {alarm_config['AlarmName']}")

setup_essential_alarms('i-0abcdef1234567890')
```

### アンチパターン 5: タグ戦略がない

```python
# 良い例: 体系的なタグ付け戦略
REQUIRED_TAGS = {
    'Environment': ['production', 'staging', 'development'],
    'Project': None,  # 自由入力
    'Owner': None,    # メールアドレス
    'CostCenter': None,
    'ManagedBy': ['terraform', 'cloudformation', 'manual'],
}

def validate_tags(tags: dict) -> list:
    """タグポリシーのバリデーション"""
    errors = []
    for required_key, allowed_values in REQUIRED_TAGS.items():
        if required_key not in tags:
            errors.append(f"必須タグ '{required_key}' がありません")
        elif allowed_values and tags[required_key] not in allowed_values:
            errors.append(
                f"タグ '{required_key}' の値 '{tags[required_key]}' は "
                f"許可値 {allowed_values} に含まれていません"
            )
    return errors

# AWS Config Rule でタグポリシーを強制
# required-tags ルールの設定
config_rule = {
    'ConfigRuleName': 'required-tags',
    'Source': {
        'Owner': 'AWS',
        'SourceIdentifier': 'REQUIRED_TAGS'
    },
    'InputParameters': json.dumps({
        'tag1Key': 'Environment',
        'tag2Key': 'Project',
        'tag3Key': 'Owner',
        'tag4Key': 'CostCenter'
    }),
    'Scope': {
        'ComplianceResourceTypes': [
            'AWS::EC2::Instance',
            'AWS::S3::Bucket',
            'AWS::RDS::DBInstance'
        ]
    }
}
```

---

## 9. 共有責任モデルの詳細

```
AWS共有責任モデル:

┌──────────────────────────────────────────────────────┐
│                ユーザーの責任                          │
│           "Security IN the Cloud"                    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ カスタマーデータ                               │    │
│  ├──────────────────────────────────────────────┤    │
│  │ プラットフォーム、アプリケーション、IAM管理     │    │
│  ├──────────────────────────────────────────────┤    │
│  │ OS、ネットワーク、ファイアウォール設定          │    │
│  ├──────────────────────────────────────────────┤    │
│  │ クライアント側の暗号化 / サーバー側の暗号化     │    │
│  │ ネットワークトラフィックの保護                  │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
├──────────────────────────────────────────────────────┤
│              AWSの責任                                │
│           "Security OF the Cloud"                    │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │ ソフトウェア: コンピュート、ストレージ、DB、     │    │
│  │            ネットワーキング                     │    │
│  ├──────────────────────────────────────────────┤    │
│  │ ハードウェア / AWSグローバルインフラストラクチャ │    │
│  │ リージョン、AZ、エッジロケーション               │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

```python
# 共有責任モデルに基づくセキュリティチェックリスト
security_checklist = {
    'ユーザー責任': {
        'IAM': [
            'ルートアカウントにMFAを有効化',
            '個別IAMユーザーを作成（ルート共有禁止）',
            '最小権限の原則を適用',
            'IAMロールを使用（アクセスキー最小化）',
            'パスワードポリシーの強化',
            'アクセスキーの定期ローテーション',
        ],
        'データ保護': [
            'S3バケットのパブリックアクセスブロック',
            'EBSボリュームの暗号化',
            'RDSインスタンスの暗号化',
            'SSL/TLSの強制',
            'KMSによるキー管理',
        ],
        'ネットワーク': [
            'セキュリティグループの最小権限設定',
            'NACLの適切な設定',
            'VPCフローログの有効化',
            'プライベートサブネットの活用',
            'VPCエンドポイントの使用',
        ],
        '監視': [
            'CloudTrailの有効化（全リージョン）',
            'GuardDutyの有効化',
            'AWS Configの有効化',
            'CloudWatchアラームの設定',
            'Security Hubの統合',
        ]
    },
    'AWS責任': [
        '物理的データセンターセキュリティ',
        'ハードウェアの廃棄手順',
        'ネットワークインフラの保護',
        'ハイパーバイザーのセキュリティ',
        '電力・冷却の確保',
        'コンプライアンス認証の維持',
    ]
}

def print_checklist(checklist: dict, indent: int = 0):
    prefix = "  " * indent
    for key, value in checklist.items():
        print(f"{prefix}[ ] {key}")
        if isinstance(value, dict):
            print_checklist(value, indent + 1)
        elif isinstance(value, list):
            for item in value:
                print(f"{prefix}  [ ] {item}")

print_checklist(security_checklist)
```

---

## 10. FAQ

### Q1. クラウドは本当にコスト削減になるのか？

必ずしもそうではない。常時稼働のワークロードはオンプレミスの方が安くなる場合がある。クラウドの真のメリットは「弾力性」と「俊敏性」にあり、変動するワークロードや新規プロジェクトの立ち上げでコスト優位性が高い。Reserved Instances や Savings Plans を活用すれば、固定ワークロードでも最大 72% の割引が可能。

具体的な判断基準:
- CPU使用率が平均20%以下 → クラウドの方が割高になりやすい
- ピークとオフピークの差が3倍以上 → クラウドが有利
- 新規プロジェクトで将来の需要が不確実 → クラウドが有利
- 3年以上安定稼働する基幹系 → RI/SPで対応、またはオンプレを検討

### Q2. AWS / GCP / Azure のどれを選ぶべきか？

チームのスキルセット、既存の技術スタック、必要なサービスの成熟度で判断する。一般的に AWS はサービスの幅が最も広く、GCP はデータ分析・ML に強み、Azure は Microsoft エコシステム (Active Directory, Office 365) との親和性が高い。マルチクラウド戦略も有効だが、運用複雑性が増すため慎重に検討する。

```
選択基準マトリクス:

                    AWS     GCP     Azure
サービス数          ★★★★★  ★★★    ★★★★
ML/AI              ★★★★   ★★★★★  ★★★★
コンテナ/K8s        ★★★★   ★★★★★  ★★★★
エンタープライズ連携 ★★★★   ★★★    ★★★★★
コスト透明性         ★★★    ★★★★★  ★★★
日本語サポート       ★★★★★  ★★★    ★★★★
スタートアップ向け   ★★★★   ★★★★★  ★★★
```

### Q3. クラウドのセキュリティはオンプレミスより弱いのか？

「共有責任モデル」において、クラウドプロバイダは物理インフラのセキュリティを担い、ユーザーはデータとアクセス管理を担う。主要プロバイダは SOC 2、ISO 27001、PCI DSS などの認証を取得しており、適切に設定すればオンプレミスと同等以上のセキュリティを実現できる。

### Q4. クラウド移行にどのくらいの期間がかかるのか？

規模と複雑さによるが、一般的な目安は以下の通り。

```
移行規模別の期間目安:

小規模（サーバー10台以下）
├── アセスメント: 2-4週間
├── 計画: 2-4週間
├── 移行実施: 4-8週間
└── 合計: 2-4ヶ月

中規模（サーバー10-100台）
├── アセスメント: 4-8週間
├── 計画: 4-8週間
├── 移行実施: 3-6ヶ月
├── 最適化: 2-3ヶ月
└── 合計: 6-12ヶ月

大規模（サーバー100台以上）
├── アセスメント: 2-3ヶ月
├── 計画: 2-4ヶ月
├── 移行実施: 6-18ヶ月
├── 最適化: 3-6ヶ月
└── 合計: 1-2年以上
```

### Q5. クラウドの認定資格は取得すべきか？

実務経験と合わせて取得することで大きな価値がある。

```
AWS認定資格ロードマップ:

【基礎レベル】
└── Cloud Practitioner (CLF-C02)
    学習期間: 2-4週間
    対象: 全職種、クラウド初学者

【アソシエイトレベル】
├── Solutions Architect Associate (SAA-C03)
│   学習期間: 1-2ヶ月
│   対象: インフラエンジニア、アーキテクト
│
├── Developer Associate (DVA-C02)
│   学習期間: 1-2ヶ月
│   対象: アプリケーション開発者
│
└── SysOps Administrator Associate (SOA-C02)
    学習期間: 1-2ヶ月
    対象: 運用エンジニア

【プロフェッショナルレベル】
├── Solutions Architect Professional (SAP-C02)
│   学習期間: 2-3ヶ月
│   対象: シニアアーキテクト
│
└── DevOps Engineer Professional (DOP-C02)
    学習期間: 2-3ヶ月
    対象: シニアDevOpsエンジニア

【スペシャリティ】
├── Security Specialty
├── Database Specialty
├── Advanced Networking Specialty
├── Machine Learning Specialty
└── Data Analytics Specialty
```

---

## 11. まとめ

| 項目 | ポイント |
|------|---------|
| クラウドの定義 | オンデマンドでリソースを確保・解放でき、従量課金で利用するモデル |
| サービスモデル | IaaS(インフラ) → CaaS(コンテナ) → PaaS(プラットフォーム) → FaaS(関数) → SaaS(アプリ) の順に抽象度が上がる |
| デプロイモデル | パブリック、プライベート、ハイブリッド、マルチクラウドの4種 |
| AWS の強み | サービス数最多、グローバルリージョン最多、エコシステム成熟 |
| コスト最適化 | 従量課金 + 予約割引 + スポット活用 + アーキテクチャ最適化の4層戦略 |
| セキュリティ | 共有責任モデルを理解し、ユーザー側の設定を確実に行う |
| 移行戦略 | 6R (Rehost/Replatform/Repurchase/Refactor/Retire/Retain) で分類 |
| クラウドネイティブ | 12 Factor App、マイクロサービス、IaC、コンテナ化が基本 |

---

## 次に読むべきガイド

- [01-aws-account-setup.md](./01-aws-account-setup.md) -- AWS アカウントの作成と初期設定
- [02-aws-cli-sdk.md](./02-aws-cli-sdk.md) -- CLI/SDK のセットアップと認証情報管理

---

## 参考文献

1. NIST SP 800-145 "The NIST Definition of Cloud Computing" -- https://csrc.nist.gov/publications/detail/sp/800-145/final
2. AWS Well-Architected Framework -- https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html
3. Gartner "Magic Quadrant for Cloud Infrastructure and Platform Services" -- https://www.gartner.com/en/documents/cloud-infrastructure-platform-services
4. AWS 共有責任モデル -- https://aws.amazon.com/compliance/shared-responsibility-model/
5. CNCF Cloud Native Definition -- https://github.com/cncf/toc/blob/main/DEFINITION.md
6. The Twelve-Factor App -- https://12factor.net/
7. AWS Migration Hub -- https://docs.aws.amazon.com/migrationhub/
8. AWS Cost Optimization -- https://aws.amazon.com/pricing/cost-optimization/
9. AWS Certification -- https://aws.amazon.com/certification/
