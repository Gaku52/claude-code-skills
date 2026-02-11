# AWS クラウドガイド

> AWS はクラウドコンピューティングの事実上の標準。EC2/S3/Lambda の基礎からネットワーク設計、データベース選定、サーバーレスアーキテクチャ、コンテナ運用、セキュリティ、コスト最適化まで、AWS の全てを体系的に解説する。

## このSkillの対象者

- AWS を使ったインフラ構築を学びたいエンジニア
- AWS 認定資格（SAA/SAP）の取得を目指す方
- オンプレミスからクラウドへの移行を計画する方

## 前提知識

- Linux の基本操作
- ネットワークの基礎（TCP/IP、DNS、HTTP）
- コンテナの基礎知識（Docker）

## 学習ガイド

### 00-fundamentals — AWS の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-aws-overview.md]] | AWS の全体像、リージョン/AZ、Well-Architected Framework |
| 01 | [[docs/00-fundamentals/01-iam-and-security.md]] | IAM ユーザー/ロール/ポリシー、MFA、Organizations |
| 02 | [[docs/00-fundamentals/02-aws-cli-and-sdk.md]] | AWS CLI セットアップ、プロファイル、SDK（JavaScript/Python） |
| 03 | [[docs/00-fundamentals/03-infrastructure-as-code.md]] | CloudFormation、CDK、Terraform、SAM の比較と実践 |

### 01-compute — コンピューティング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-compute/00-ec2-basics.md]] | EC2 インスタンス、AMI、セキュリティグループ、キーペア |
| 01 | [[docs/01-compute/01-ec2-advanced.md]] | Auto Scaling、スポットインスタンス、Placement Group、EBS |
| 02 | [[docs/01-compute/02-elastic-beanstalk.md]] | Elastic Beanstalk、プラットフォーム選択、デプロイ戦略 |

### 02-storage — ストレージ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-storage/00-s3-basics.md]] | S3 バケット、オブジェクト、アクセス制御、暗号化 |
| 01 | [[docs/02-storage/01-s3-advanced.md]] | ライフサイクル、レプリケーション、Transfer Acceleration、Glacier |
| 02 | [[docs/02-storage/02-efs-and-fsx.md]] | EFS、FSx、Storage Gateway の比較と用途 |

### 03-database — データベース

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-database/00-rds-and-aurora.md]] | RDS エンジン選択、Aurora、Multi-AZ、リードレプリカ |
| 01 | [[docs/03-database/01-dynamodb.md]] | DynamoDB 設計、パーティションキー、GSI/LSI、キャパシティ |
| 02 | [[docs/03-database/02-elasticache-and-others.md]] | ElastiCache（Redis/Memcached）、DocumentDB、Neptune |

### 04-networking — ネットワーキング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-networking/00-vpc-basics.md]] | VPC、サブネット、ルートテーブル、IGW、NAT Gateway |
| 01 | [[docs/04-networking/01-vpc-advanced.md]] | VPC Peering、Transit Gateway、VPN、Direct Connect |
| 02 | [[docs/04-networking/02-route53-and-cloudfront.md]] | Route 53、CloudFront、ACM、WAF |

### 05-serverless — サーバーレス

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/05-serverless/00-lambda-basics.md]] | Lambda 関数、ランタイム、デプロイ、環境変数、レイヤー |
| 01 | [[docs/05-serverless/01-lambda-advanced.md]] | コールドスタート対策、Provisioned Concurrency、Lambda@Edge |
| 02 | [[docs/05-serverless/02-api-gateway.md]] | REST API、HTTP API、WebSocket API、認証統合 |
| 03 | [[docs/05-serverless/03-serverless-patterns.md]] | SQS/SNS/EventBridge、Step Functions、サーバーレス設計パターン |

### 06-containers — コンテナサービス

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/06-containers/00-ecs-basics.md]] | ECS クラスター、タスク定義、サービス、Fargate vs EC2 |
| 01 | [[docs/06-containers/01-eks-basics.md]] | EKS セットアップ、ノードグループ、Ingress、Helm |
| 02 | [[docs/06-containers/02-ecr-and-app-runner.md]] | ECR レジストリ、App Runner、Copilot CLI |

### 07-devops — DevOps サービス

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/07-devops/00-codepipeline.md]] | CodePipeline、CodeBuild、CodeDeploy の統合 |
| 01 | [[docs/07-devops/01-cloudwatch.md]] | CloudWatch メトリクス、ログ、アラーム、ダッシュボード |

### 08-security — セキュリティサービス

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/08-security/00-security-services.md]] | GuardDuty、SecurityHub、Config、Inspector、Macie |
| 01 | [[docs/08-security/01-secrets-and-encryption.md]] | KMS、Secrets Manager、Parameter Store、ACM |

### 09-cost-management — コスト管理

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/09-cost-management/00-cost-optimization.md]] | Cost Explorer、Budgets、Reserved/Savings Plans、スポット |
| 01 | [[docs/09-cost-management/01-well-architected.md]] | Well-Architected Framework 5 つの柱、レビュー手法 |

## クイックリファレンス

```
AWS サービス選定チャート:

  コンピューティング:
    コンテナ → ECS Fargate（推奨）or EKS
    サーバーレス → Lambda + API Gateway
    VM → EC2 + Auto Scaling
    PaaS → Elastic Beanstalk

  データベース:
    RDB → Aurora（推奨）or RDS
    NoSQL → DynamoDB
    キャッシュ → ElastiCache Redis
    全文検索 → OpenSearch

  ストレージ:
    オブジェクト → S3
    ファイル → EFS
    ブロック → EBS

  ネットワーク:
    DNS → Route 53
    CDN → CloudFront
    ロードバランサー → ALB

  コスト削減:
    ✓ Savings Plans / Reserved Instances
    ✓ スポットインスタンス（耐障害性あるワークロード）
    ✓ S3 ライフサイクル（Glacier 移行）
    ✓ Lambda（低トラフィック時）
```

## 参考文献

1. AWS. "Documentation." docs.aws.amazon.com, 2024.
2. AWS. "Well-Architected Framework." aws.amazon.com/architecture, 2024.
3. AWS. "Pricing Calculator." calculator.aws, 2024.
