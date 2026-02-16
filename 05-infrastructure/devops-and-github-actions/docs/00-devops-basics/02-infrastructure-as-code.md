# Infrastructure as Code (IaC)

> インフラの構成をコードとして管理し、バージョン管理・レビュー・自動適用を可能にする手法

## この章で学ぶこと

1. IaCの基本概念と宣言的/命令的アプローチの違いを理解する
2. Terraform、CloudFormation、CDK、Pulumiの特徴と使い分けを習得する
3. IaCのベストプラクティスとGitOps連携パターンを把握する
4. IaCのテスト戦略とCI/CDパイプラインへの統合方法を理解する
5. 実運用におけるモジュール設計とマルチ環境管理を実践できる

---

## 1. IaC とは何か

### 1.1 従来のインフラ管理 vs IaC

```
従来のインフラ管理:
  管理者がGUIコンソールで手動設定
  → ドキュメントと実態が乖離
  → 再現性がない
  → 変更履歴が追えない
  → 複数人の作業で設定が衝突
  → 障害時の復旧に時間がかかる

IaC:
  コードでインフラを定義
  → Git でバージョン管理
  → PR でレビュー
  → CI/CD で自動適用
  → 環境の完全な再現が可能
  → 障害時はコードから即座に復元
```

### 1.2 IaC の利点

```
+----------------------------------------------------------+
|                    IaC の価値                               |
+----------------------------------------------------------+
|                                                            |
|  再現性          同じコードで同じ環境を何度でも構築          |
|  ┌──────────────────────────────────────────┐              |
|  │ code → dev環境 / staging環境 / prod環境   │              |
|  └──────────────────────────────────────────┘              |
|                                                            |
|  追跡可能性      Git の履歴 = インフラの変更履歴             |
|  ┌──────────────────────────────────────────┐              |
|  │ commit log = "いつ誰が何を変更したか"      │              |
|  └──────────────────────────────────────────┘              |
|                                                            |
|  レビュー可能性  PR でインフラ変更をレビュー                 |
|  ┌──────────────────────────────────────────┐              |
|  │ terraform plan の差分を PR コメントに表示  │              |
|  └──────────────────────────────────────────┘              |
|                                                            |
|  一貫性          全環境が同じコードから生成                  |
|  ┌──────────────────────────────────────────┐              |
|  │ 環境間の差分 = 変数の違いのみ              │              |
|  └──────────────────────────────────────────┘              |
|                                                            |
|  速度            新環境を数分で構築可能                      |
|  ┌──────────────────────────────────────────┐              |
|  │ terraform apply → 5分で完全な環境が完成    │              |
|  └──────────────────────────────────────────┘              |
|                                                            |
+----------------------------------------------------------+
```

### 1.3 IaC の適用範囲

IaCはクラウドインフラだけでなく、幅広い領域に適用される。

| 対象 | ツール例 | 管理対象 |
|---|---|---|
| クラウドインフラ | Terraform, CDK, Pulumi | VPC, EC2, RDS, S3 等 |
| コンテナオーケストレーション | Kubernetes マニフェスト, Helm | Pod, Service, Deployment 等 |
| 構成管理 | Ansible, Chef, Puppet | OS設定, パッケージ, ユーザー |
| ネットワーク | Terraform, Ansible | ファイアウォール, DNS, CDN |
| モニタリング | Terraform (Datadog/PagerDuty provider) | ダッシュボード, アラート |
| CI/CD | GitHub Actions YAML, GitLab CI | パイプライン, ワークフロー |
| アクセス制御 | Terraform (IAM), Vault | ポリシー, ロール, シークレット |

---

## 2. 宣言的 vs 命令的アプローチ

### 2.1 比較表

| 項目 | 宣言的 (Declarative) | 命令的 (Imperative) |
|---|---|---|
| 定義方法 | 「あるべき状態」を記述 | 「手順」を記述 |
| 冪等性 | 組み込み | 自分で担保 |
| 代表ツール | Terraform, CloudFormation | Ansible (一部), シェルスクリプト |
| 差分検知 | 自動 (plan) | 困難 |
| 学習コスト | DSL の習得が必要 | プログラミングスキルで対応可能 |
| 適用場面 | インフラ構築 | 構成管理、プロビジョニング |
| ドリフト検知 | 容易 | 困難 |
| 並行実行 | ツールが依存関係を解決 | 手動で順序を制御 |

### 2.2 宣言的の例（Terraform）

```hcl
# 「こうあるべき」を記述 → Terraform が差分を検出して適用
resource "aws_s3_bucket" "data" {
  bucket = "my-app-data-bucket"

  versioning {
    enabled = true
  }

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }

  tags = {
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

# ライフサイクルルール
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "archive-old-objects"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }
}
```

### 2.3 命令的の例（シェルスクリプト）

```bash
#!/bin/bash
# 「手順」を記述 → 冪等性は自分で担保する必要がある

# バケットが存在しなければ作成
if ! aws s3api head-bucket --bucket my-app-data-bucket 2>/dev/null; then
  aws s3api create-bucket --bucket my-app-data-bucket
fi

# バージョニングを有効化
aws s3api put-bucket-versioning \
  --bucket my-app-data-bucket \
  --versioning-configuration Status=Enabled

# 暗号化を設定
aws s3api put-bucket-encryption \
  --bucket my-app-data-bucket \
  --server-side-encryption-configuration '{
    "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
  }'

# ライフサイクルルールを設定
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-app-data-bucket \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "archive-old-objects",
      "Status": "Enabled",
      "Transitions": [{"Days": 90, "StorageClass": "GLACIER"}],
      "Expiration": {"Days": 365}
    }]
  }'
```

### 2.4 ハイブリッドアプローチ

実際のプロジェクトでは、宣言的と命令的を組み合わせることが多い。

```
典型的な組み合わせ:

1. Terraform (宣言的) + Ansible (命令的)
   Terraform: VPC, EC2 インスタンスを構築
   Ansible: EC2 内のOS設定、パッケージインストール

2. Terraform (宣言的) + User Data スクリプト (命令的)
   Terraform: インフラ構築 + User Data で初期化スクリプト実行

3. CDK (宣言的/プログラマティック) + Custom Resource (命令的)
   CDK: 標準リソース定義
   Custom Resource: CDKが対応していないリソースをLambdaで管理
```

---

## 3. 主要 IaC ツール

### 3.1 Terraform

HashiCorp が開発した、マルチクラウド対応の IaC ツール。HCL (HashiCorp Configuration Language) で記述する。最も広く使われているIaCツールであり、1,000以上のプロバイダーが利用可能。

```hcl
# Terraform の基本構成
terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "ap-northeast-1"
    dynamodb_table = "terraform-locks"
    encrypt        = true
  }
}

provider "aws" {
  region = "ap-northeast-1"

  default_tags {
    tags = {
      ManagedBy   = "terraform"
      Project     = "my-app"
      Environment = var.environment
    }
  }
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project}-${var.environment}-vpc"
  }
}

# サブネット
resource "aws_subnet" "private" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.project}-${var.environment}-private-${count.index}"
    Type = "private"
  }
}

resource "aws_subnet" "public" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index + 100)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project}-${var.environment}-public-${count.index}"
    Type = "public"
  }
}

# ECS Fargate サービス
resource "aws_ecs_service" "app" {
  name            = "my-app"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.app.id]
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  lifecycle {
    ignore_changes = [desired_count]  # オートスケーリングと競合防止
  }
}
```

```
Terraform ワークフロー:

  terraform init    → プロバイダー・モジュールのダウンロード
       ↓
  terraform plan    → 現在の状態と定義の差分を表示
       ↓
  terraform apply   → 変更を適用
       ↓
  terraform destroy → リソースを削除 (開発環境のクリーンアップ)

状態管理:
  +-----------+     +----------------+     +-----------+
  | .tf ファイル | ←→ | terraform.tfstate | ←→ | 実インフラ  |
  | (あるべき姿) |     | (現在の状態)      |     | (AWS等)    |
  +-----------+     +----------------+     +-----------+

OpenTofu (Terraformのオープンソースフォーク):
  - HashiCorp のライセンス変更(BSL)を受けて2023年に誕生
  - Terraform 1.5.x からフォーク
  - Linux Foundation 管轄
  - 既存の Terraform コードとほぼ互換
```

### 3.2 AWS CloudFormation

```yaml
# CloudFormation テンプレート
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Complete application stack with VPC, ECS, and RDS'

Parameters:
  Environment:
    Type: String
    AllowedValues: [dev, staging, prod]
  InstanceClass:
    Type: String
    Default: db.t3.medium
    AllowedValues: [db.t3.micro, db.t3.small, db.t3.medium, db.t3.large]

Conditions:
  IsProduction: !Equals [!Ref Environment, prod]

Resources:
  DataBucket:
    Type: AWS::S3::Bucket
    DeletionPolicy: Retain  # スタック削除時もバケットを保持
    Properties:
      BucketName: !Sub 'my-app-${Environment}-data'
      VersioningConfiguration:
        Status: Enabled
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256
      LifecycleConfiguration:
        Rules:
          - Id: ArchiveOldObjects
            Status: Enabled
            Transitions:
              - TransitionInDays: 90
                StorageClass: GLACIER

  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      Engine: postgres
      EngineVersion: '16.1'
      DBInstanceClass: !If [IsProduction, db.t3.large, !Ref InstanceClass]
      AllocatedStorage: !If [IsProduction, 100, 20]
      MultiAZ: !If [IsProduction, true, false]
      StorageEncrypted: true
      DeletionProtection: !If [IsProduction, true, false]
      BackupRetentionPeriod: !If [IsProduction, 30, 7]

Outputs:
  BucketArn:
    Value: !GetAtt DataBucket.Arn
    Export:
      Name: !Sub '${Environment}-data-bucket-arn'
  DatabaseEndpoint:
    Value: !GetAtt Database.Endpoint.Address
    Export:
      Name: !Sub '${Environment}-db-endpoint'
```

### 3.3 AWS CDK

```typescript
// AWS CDK (TypeScript) - プログラミング言語でインフラを定義
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as ecs_patterns from 'aws-cdk-lib/aws-ecs-patterns';

interface AppStackProps extends cdk.StackProps {
  environment: string;
  isProduction: boolean;
}

export class AppStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props: AppStackProps) {
    super(scope, id, props);

    // VPC
    const vpc = new ec2.Vpc(this, 'AppVpc', {
      maxAzs: props.isProduction ? 3 : 2,
      natGateways: props.isProduction ? 2 : 1,
      subnetConfiguration: [
        {
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
        {
          name: 'Isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
    });

    // RDS
    const database = new rds.DatabaseInstance(this, 'Database', {
      engine: rds.DatabaseInstanceEngine.postgres({
        version: rds.PostgresEngineVersion.VER_16_1,
      }),
      instanceType: props.isProduction
        ? ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.LARGE)
        : ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM),
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_ISOLATED },
      multiAz: props.isProduction,
      deletionProtection: props.isProduction,
      backupRetention: cdk.Duration.days(props.isProduction ? 30 : 7),
    });

    // ECS クラスター
    const cluster = new ecs.Cluster(this, 'AppCluster', { vpc });

    // Fargate サービス (L3 Construct: ALB + Fargate を一括定義)
    const service = new ecs_patterns.ApplicationLoadBalancedFargateService(
      this, 'AppService', {
        cluster,
        cpu: props.isProduction ? 1024 : 256,
        memoryLimitMiB: props.isProduction ? 2048 : 512,
        desiredCount: props.isProduction ? 3 : 1,
        taskImageOptions: {
          image: ecs.ContainerImage.fromRegistry('my-app:latest'),
          containerPort: 3000,
          environment: {
            NODE_ENV: props.environment,
            DB_HOST: database.instanceEndpoint.hostname,
          },
        },
        circuitBreaker: { rollback: true },
      },
    );

    // オートスケーリング
    if (props.isProduction) {
      const scaling = service.service.autoScaleTaskCount({
        minCapacity: 3,
        maxCapacity: 10,
      });
      scaling.scaleOnCpuUtilization('CpuScaling', {
        targetUtilizationPercent: 70,
      });
      scaling.scaleOnMemoryUtilization('MemoryScaling', {
        targetUtilizationPercent: 80,
      });
    }

    // DB への接続許可
    database.connections.allowFrom(
      service.service,
      ec2.Port.tcp(5432),
      'Allow ECS to access RDS',
    );
  }
}

// アプリケーションのエントリポイント
const app = new cdk.App();

new AppStack(app, 'AppDev', {
  environment: 'dev',
  isProduction: false,
  env: { account: '123456789012', region: 'ap-northeast-1' },
});

new AppStack(app, 'AppProd', {
  environment: 'prod',
  isProduction: true,
  env: { account: '987654321098', region: 'ap-northeast-1' },
});
```

### 3.4 Pulumi

```typescript
// Pulumi (TypeScript) - 汎用プログラミング言語でマルチクラウド対応
import * as pulumi from '@pulumi/pulumi';
import * as aws from '@pulumi/aws';
import * as awsx from '@pulumi/awsx';

const config = new pulumi.Config();
const env = config.require('environment');
const isProduction = env === 'prod';

// VPC (Crosswalk: 高レベル抽象化)
const vpc = new awsx.ec2.Vpc('app-vpc', {
  numberOfAvailabilityZones: isProduction ? 3 : 2,
  natGateways: isProduction ? { strategy: awsx.ec2.NatGatewayStrategy.OnePerAz } : { strategy: awsx.ec2.NatGatewayStrategy.Single },
});

// S3 バケット
const bucket = new aws.s3.Bucket('data-bucket', {
  bucket: `my-app-${env}-data`,
  versioning: { enabled: true },
  serverSideEncryptionConfiguration: {
    rule: {
      applyServerSideEncryptionByDefault: {
        sseAlgorithm: 'AES256',
      },
    },
  },
  lifecycleRules: [{
    enabled: true,
    transitions: [{
      days: 90,
      storageClass: 'GLACIER',
    }],
    expiration: { days: 365 },
  }],
});

// ECS クラスター + Fargate サービス
const cluster = new aws.ecs.Cluster('app-cluster');

const service = new awsx.ecs.FargateService('app-service', {
  cluster: cluster.arn,
  desiredCount: isProduction ? 3 : 1,
  networkConfiguration: {
    subnets: vpc.privateSubnetIds,
    securityGroups: [],
  },
  taskDefinitionArgs: {
    container: {
      name: 'app',
      image: 'my-app:latest',
      cpu: isProduction ? 1024 : 256,
      memory: isProduction ? 2048 : 512,
      portMappings: [{ containerPort: 3000 }],
      environment: [
        { name: 'NODE_ENV', value: env },
      ],
    },
  },
});

// 出力
export const bucketArn = bucket.arn;
export const bucketName = bucket.bucket;
export const vpcId = vpc.vpcId;
export const serviceUrl = pulumi.interpolate`http://${service.service.name}`;
```

### 3.5 Crossplane

Kubernetes CRD としてクラウドリソースを管理するIaCツール。

```yaml
# Crossplane: Kubernetes マニフェストでAWSリソースを管理
apiVersion: s3.aws.upbound.io/v1beta1
kind: Bucket
metadata:
  name: my-app-data
spec:
  forProvider:
    region: ap-northeast-1
  providerConfigRef:
    name: aws-provider

---
apiVersion: rds.aws.upbound.io/v1beta1
kind: Instance
metadata:
  name: my-app-db
spec:
  forProvider:
    region: ap-northeast-1
    engine: postgres
    engineVersion: "16.1"
    instanceClass: db.t3.medium
    allocatedStorage: 20
    storageEncrypted: true
  providerConfigRef:
    name: aws-provider
```

---

## 4. IaC ツール比較

| 項目 | Terraform | CloudFormation | CDK | Pulumi | Crossplane |
|---|---|---|---|---|---|
| 言語 | HCL | YAML/JSON | TypeScript等 | TypeScript/Python/Go等 | YAML (K8s CRD) |
| マルチクラウド | はい | AWS のみ | AWS のみ | はい | はい |
| 状態管理 | tfstate (S3等) | AWS管理 | CloudFormation経由 | Pulumi Cloud / S3 | Kubernetes etcd |
| 学習コスト | 中 | 中 | 低(開発者向き) | 低(開発者向き) | 中(K8s知識必要) |
| エコシステム | 最大 | AWS限定 | AWS限定 | 成長中 | 成長中 |
| ドリフト検知 | plan で検知 | drift detection | CloudFormation経由 | preview で検知 | 継続的リコンサイル |
| 推奨場面 | マルチクラウド | AWS専用 | AWS + TypeScript | マルチクラウド + 開発者 | K8s中心の組織 |
| ライセンス | BSL 1.1 | AWS サービス | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| OSS代替 | OpenTofu | - | - | - | - |

### 4.1 ツール選定ディシジョンツリー

```
IaC ツール選定:

Kubernetes 中心の組織？
├── Yes → Crossplane を検討
│         ├── クラウドリソースもK8sで管理したい → Crossplane
│         └── K8sマニフェストのみ管理 → Kustomize / Helm
└── No → マルチクラウド要件がある？
          ├── Yes → Terraform or Pulumi
          │         ├── DSLが好み / 大きなコミュニティ → Terraform
          │         └── プログラミング言語で書きたい → Pulumi
          └── No → AWS のみ？
                    ├── Yes → CDK or CloudFormation
                    │         ├── TypeScript/Python チーム → CDK
                    │         └── YAML シンプルに → CloudFormation
                    └── No → 他クラウド → Terraform / Pulumi
```

---

## 5. IaC のベストプラクティス

### 5.1 モジュール化

```hcl
# modules/ecs-service/main.tf - 再利用可能なモジュール
variable "service_name" {
  type        = string
  description = "サービス名"
}

variable "image" {
  type        = string
  description = "コンテナイメージ"
}

variable "cpu" {
  type        = number
  default     = 256
  description = "CPU ユニット (256 = 0.25 vCPU)"
}

variable "memory" {
  type        = number
  default     = 512
  description = "メモリ (MiB)"
}

variable "desired_count" {
  type        = number
  default     = 1
  description = "希望するタスク数"
}

variable "environment_variables" {
  type        = map(string)
  default     = {}
  description = "環境変数"
}

variable "cluster_id" {
  type        = string
  description = "ECS クラスター ID"
}

variable "subnet_ids" {
  type        = list(string)
  description = "サブネット ID リスト"
}

variable "security_group_ids" {
  type        = list(string)
  description = "セキュリティグループ ID リスト"
}

# タスク定義
resource "aws_ecs_task_definition" "this" {
  family                   = var.service_name
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([{
    name  = var.service_name
    image = var.image
    portMappings = [{
      containerPort = 3000
      protocol      = "tcp"
    }]
    environment = [
      for k, v in var.environment_variables : {
        name  = k
        value = v
      }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.this.name
        "awslogs-region"        = data.aws_region.current.name
        "awslogs-stream-prefix" = var.service_name
      }
    }
  }])
}

# ECS サービス
resource "aws_ecs_service" "this" {
  name            = var.service_name
  cluster         = var.cluster_id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.subnet_ids
    security_groups = var.security_group_ids
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
}

# 出力
output "service_name" {
  value = aws_ecs_service.this.name
}

output "task_definition_arn" {
  value = aws_ecs_task_definition.this.arn
}
```

```hcl
# 利用側: environments/prod/main.tf
module "api_service" {
  source       = "../../modules/ecs-service"
  service_name = "api"
  image        = "my-api:v1.2.3"
  cpu          = 512
  memory       = 1024
  desired_count = 3
  cluster_id   = module.ecs_cluster.id
  subnet_ids   = module.vpc.private_subnet_ids
  security_group_ids = [module.security.app_sg_id]
  environment_variables = {
    NODE_ENV     = "production"
    DATABASE_URL = module.database.connection_string
  }
}

module "worker_service" {
  source       = "../../modules/ecs-service"
  service_name = "worker"
  image        = "my-worker:v1.2.3"
  cpu          = 1024
  memory       = 2048
  desired_count = 2
  cluster_id   = module.ecs_cluster.id
  subnet_ids   = module.vpc.private_subnet_ids
  security_group_ids = [module.security.worker_sg_id]
  environment_variables = {
    NODE_ENV  = "production"
    QUEUE_URL = module.sqs.queue_url
  }
}
```

### 5.2 ディレクトリ構成

```
terraform/
├── modules/                   # 再利用可能モジュール
│   ├── networking/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── ecs-service/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── iam.tf
│   ├── rds/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   └── monitoring/
│       ├── main.tf
│       ├── variables.tf
│       └── dashboards.tf
├── environments/              # 環境別設定
│   ├── dev/
│   │   ├── main.tf           # モジュール呼び出し
│   │   ├── variables.tf      # 変数定義
│   │   ├── terraform.tfvars  # 環境固有の値
│   │   ├── backend.tf        # 状態ファイルの保存先
│   │   └── providers.tf      # プロバイダー設定
│   ├── staging/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   └── prod/
│       ├── main.tf
│       ├── terraform.tfvars
│       └── backend.tf
├── global/                    # 環境共通リソース
│   ├── iam/
│   │   └── main.tf
│   ├── dns/
│   │   └── main.tf
│   └── ecr/
│       └── main.tf
└── scripts/                   # ヘルパースクリプト
    ├── init.sh
    └── plan.sh
```

### 5.3 変数管理のベストプラクティス

```hcl
# variables.tf - 型・説明・バリデーション付き変数定義
variable "environment" {
  type        = string
  description = "デプロイ環境 (dev, staging, prod)"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment は dev, staging, prod のいずれかを指定してください"
  }
}

variable "instance_type" {
  type        = string
  default     = "t3.medium"
  description = "EC2 インスタンスタイプ"
  validation {
    condition     = can(regex("^t3\\.", var.instance_type))
    error_message = "t3 ファミリーのインスタンスタイプを指定してください"
  }
}

variable "alert_email" {
  type        = string
  description = "アラート通知先メールアドレス"
  sensitive   = false
  validation {
    condition     = can(regex("^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$", var.alert_email))
    error_message = "有効なメールアドレスを指定してください"
  }
}

variable "database_password" {
  type        = string
  description = "データベースパスワード"
  sensitive   = true  # plan/apply の出力でマスクされる
}
```

```hcl
# terraform.tfvars - 環境固有の値
# (Git にコミット可能、シークレットは含めない)
environment   = "prod"
instance_type = "t3.large"
alert_email   = "ops@example.com"

# シークレットは別ファイルまたは環境変数で管理
# export TF_VAR_database_password="xxx"
# または terraform.tfvars.secret (.gitignore に追加)
```

---

## 6. IaC のテスト戦略

### 6.1 テストピラミッド

```
             /\
            /  \
           /E2E \         terraform apply + 検証 + destroy
          /------\        (terratest, kitchen-terraform)
         / 統合   \       terraform plan の検証
        /テスト    \      (tfplan JSON 解析)
       /----------\
      / 静的解析   \      lint, validate, security scan
     / ユニット    \      (tflint, checkov, tfsec, OPA)
    /--------------\
```

### 6.2 静的解析

```yaml
# CI での IaC 静的解析
name: Terraform Lint & Security
on:
  pull_request:
    paths: ['terraform/**']

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: '1.7'

      # 構文チェック
      - name: Terraform fmt
        run: terraform fmt -check -recursive
        working-directory: terraform/

      - name: Terraform validate
        run: |
          cd terraform/environments/dev
          terraform init -backend=false
          terraform validate

      # ベストプラクティスチェック
      - name: TFLint
        uses: terraform-linters/setup-tflint@v4
      - run: |
          tflint --init
          tflint --recursive
        working-directory: terraform/

      # セキュリティスキャン
      - name: Checkov
        uses: bridgecrewio/checkov-action@v12
        with:
          directory: terraform/
          framework: terraform
          output_format: sarif
          output_file_path: checkov-results.sarif

      - name: tfsec
        uses: aquasecurity/tfsec-action@v1.0.3
        with:
          working_directory: terraform/

      # OPA (Open Policy Agent) によるカスタムポリシー
      - name: OPA Policy Check
        run: |
          cd terraform/environments/dev
          terraform plan -out=tfplan.binary
          terraform show -json tfplan.binary > tfplan.json
          opa eval --data policies/ --input tfplan.json "data.terraform.deny[msg]"
```

### 6.3 Plan テスト

```yaml
# terraform plan を PR にコメント
name: Terraform Plan
on:
  pull_request:
    paths: ['terraform/**']

jobs:
  plan:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    strategy:
      matrix:
        environment: [dev, staging, prod]
    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3

      - name: Terraform Init
        run: terraform init
        working-directory: terraform/environments/${{ matrix.environment }}

      - name: Terraform Plan
        id: plan
        run: terraform plan -no-color -out=tfplan
        working-directory: terraform/environments/${{ matrix.environment }}
        continue-on-error: true

      - name: Comment PR with Plan
        uses: actions/github-script@v7
        with:
          script: |
            const plan = `${{ steps.plan.outputs.stdout }}`;
            const truncated = plan.length > 60000
              ? plan.substring(0, 60000) + '\n... (truncated)'
              : plan;

            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `### Terraform Plan - ${{ matrix.environment }}
              \`\`\`
              ${truncated}
              \`\`\`
              `
            });
```

### 6.4 E2E テスト (Terratest)

```go
// test/ecs_service_test.go
package test

import (
    "testing"
    "fmt"
    "time"

    "github.com/gruntwork-io/terratest/modules/terraform"
    "github.com/gruntwork-io/terratest/modules/aws"
    "github.com/gruntwork-io/terratest/modules/http-helper"
    "github.com/stretchr/testify/assert"
)

func TestEcsService(t *testing.T) {
    t.Parallel()

    terraformOptions := terraform.WithDefaultRetryableErrors(t, &terraform.Options{
        TerraformDir: "../modules/ecs-service",
        Vars: map[string]interface{}{
            "service_name":  fmt.Sprintf("test-%d", time.Now().Unix()),
            "image":         "nginx:latest",
            "cpu":           256,
            "memory":        512,
            "desired_count": 1,
            "environment":   "test",
        },
    })

    // テスト終了時にリソースを確実に削除
    defer terraform.Destroy(t, terraformOptions)

    // リソースを作成
    terraform.InitAndApply(t, terraformOptions)

    // 出力を検証
    serviceName := terraform.Output(t, terraformOptions, "service_name")
    assert.Contains(t, serviceName, "test-")

    // ECS サービスが Running か確認
    serviceArn := terraform.Output(t, terraformOptions, "service_arn")
    assert.NotEmpty(t, serviceArn)

    // ALB エンドポイントのヘルスチェック
    albDns := terraform.Output(t, terraformOptions, "alb_dns_name")
    url := fmt.Sprintf("http://%s/health", albDns)
    http_helper.HttpGetWithRetry(t, url, nil, 200, "OK", 30, 10*time.Second)
}
```

---

## 7. CI/CD との統合

### 7.1 Terraform CI/CD パイプライン

```yaml
# 完全な Terraform CI/CD パイプライン
name: Terraform CI/CD
on:
  push:
    branches: [main]
    paths: ['terraform/**']
  pull_request:
    paths: ['terraform/**']

permissions:
  id-token: write    # OIDC 認証用
  contents: read
  pull-requests: write

jobs:
  # PR 時: lint + plan
  lint:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
      - run: terraform fmt -check -recursive
      - run: |
          cd terraform/environments/prod
          terraform init -backend=false
          terraform validate

  plan:
    if: github.event_name == 'pull_request'
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3

      - name: Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-plan
          aws-region: ap-northeast-1

      - name: Terraform Plan
        run: |
          cd terraform/environments/prod
          terraform init
          terraform plan -no-color -out=tfplan 2>&1 | tee plan.txt

      - name: Post Plan to PR
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const plan = fs.readFileSync('terraform/environments/prod/plan.txt', 'utf8');
            const body = `### Terraform Plan
            \`\`\`hcl
            ${plan.substring(0, 60000)}
            \`\`\``;
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body
            });

  # main マージ時: apply
  apply:
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: production  # 手動承認が必要
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3

      - name: Configure AWS Credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-apply
          aws-region: ap-northeast-1

      - name: Terraform Apply
        run: |
          cd terraform/environments/prod
          terraform init
          terraform apply -auto-approve
```

### 7.2 Atlantis (Terraform PR 自動化)

```yaml
# atlantis.yaml - リポジトリ設定
version: 3
automerge: false
delete_source_branch_on_merge: true
parallel_plan: true
parallel_apply: false

projects:
  - name: prod
    dir: terraform/environments/prod
    workspace: default
    autoplan:
      when_modified: ["*.tf", "../modules/**/*.tf"]
      enabled: true
    apply_requirements: [approved, mergeable]
    workflow: production

  - name: dev
    dir: terraform/environments/dev
    workspace: default
    autoplan:
      when_modified: ["*.tf", "../modules/**/*.tf"]
      enabled: true
    apply_requirements: [mergeable]
    workflow: default

workflows:
  production:
    plan:
      steps:
        - init
        - run: tflint --recursive
        - run: checkov -d .
        - plan
    apply:
      steps:
        - apply
```

---

## 8. 状態管理

### 8.1 リモートバックエンドの設定

```hcl
# S3 + DynamoDB バックエンド (推奨)
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "ap-northeast-1"
    dynamodb_table = "terraform-locks"  # ロック用
    encrypt        = true
    kms_key_id     = "alias/terraform-state-key"
  }
}
```

```hcl
# バックエンドインフラ自体の定義 (bootstrap)
resource "aws_s3_bucket" "terraform_state" {
  bucket = "my-terraform-state"

  lifecycle {
    prevent_destroy = true  # 誤削除防止
  }
}

resource "aws_s3_bucket_versioning" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state" {
  bucket = aws_s3_bucket.terraform_state.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.terraform_state.id
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state" {
  bucket                  = aws_s3_bucket.terraform_state.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "terraform_locks" {
  name         = "terraform-locks"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}
```

### 8.2 状態ファイルの分割

```
状態ファイル分割戦略:

1. 環境別分割 (最低限)
   terraform/environments/dev/   → dev.tfstate
   terraform/environments/prod/  → prod.tfstate

2. レイヤー別分割 (推奨)
   terraform/layers/networking/  → networking.tfstate
   terraform/layers/database/    → database.tfstate
   terraform/layers/application/ → application.tfstate

3. サービス別分割 (マイクロサービス)
   terraform/services/user/      → user-service.tfstate
   terraform/services/order/     → order-service.tfstate

利点:
- blast radius (影響範囲) の最小化
- plan/apply の高速化
- チーム間の並行作業が可能
- ロック競合の削減

注意:
- 分割間のデータ共有は data source / remote state で行う
```

```hcl
# レイヤー間のデータ共有
# application レイヤーから networking の出力を参照
data "terraform_remote_state" "networking" {
  backend = "s3"
  config = {
    bucket = "my-terraform-state"
    key    = "prod/networking/terraform.tfstate"
    region = "ap-northeast-1"
  }
}

resource "aws_ecs_service" "app" {
  # networking レイヤーの出力を使用
  network_configuration {
    subnets = data.terraform_remote_state.networking.outputs.private_subnet_ids
  }
}
```

---

## 9. アンチパターン

### アンチパターン1: 状態ファイルのローカル管理

```
悪い例:
  terraform.tfstate をローカルに保存し、
  Git にコミットしてしまう。

問題:
  - 機密情報(パスワード等)が Git に入る
  - 複数人での並行作業で状態が衝突
  - 状態ファイルの紛失 = インフラ管理不能

改善:
  terraform {
    backend "s3" {
      bucket         = "my-terraform-state"
      key            = "prod/terraform.tfstate"
      region         = "ap-northeast-1"
      dynamodb_table = "terraform-locks"  # ロック用
      encrypt        = true
    }
  }
  # .gitignore に *.tfstate を追加
```

### アンチパターン2: ハードコードされた値

```hcl
# 悪い例: 値をハードコード
resource "aws_instance" "web" {
  ami           = "ami-0abcdef1234567890"  # マジックナンバー
  instance_type = "t3.large"               # 環境で変わるべき
  subnet_id     = "subnet-12345"           # 環境依存
}

# 改善: 変数化 + データソース
data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_instance" "web" {
  ami           = data.aws_ami.amazon_linux.id
  instance_type = var.instance_type
  subnet_id     = var.subnet_id
}
```

### アンチパターン3: 巨大な単一状態ファイル

```
悪い例:
  1つの terraform.tfstate に全リソース(VPC, RDS, ECS, S3, IAM, ...)を管理
  → plan に5分以上かかる
  → 1つの変更が全リソースに影響するリスク
  → チーム間でロック競合が頻発

改善:
  レイヤー別 or サービス別に状態ファイルを分割
  → 各 plan は30秒以内
  → blast radius が限定される
  → チームが並行作業可能
```

### アンチパターン4: ドリフトの放置

```
問題:
  手動でAWSコンソールから変更を行い、
  コードと実態が乖離(ドリフト)する。

  コード:     instance_type = "t3.medium"
  実インフラ:  instance_type = "t3.large" (コンソールで変更)

  → terraform plan で "変更あり" と表示され続ける
  → plan の結果が信頼できなくなる
  → 次の terraform apply で意図しない変更が入る

改善:
  1. 全変更を PR 経由で行うルールを徹底
  2. AWS Config / CloudTrail でコンソール操作を検知
  3. 定期的に terraform plan を実行してドリフトを検出
  4. CI でドリフト検出を自動化
```

```yaml
# ドリフト検出の自動化
name: Drift Detection
on:
  schedule:
    - cron: '0 9 * * *'  # 毎日9時

jobs:
  detect-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/terraform-plan
          aws-region: ap-northeast-1
      - name: Check for drift
        run: |
          cd terraform/environments/prod
          terraform init
          terraform plan -detailed-exitcode -out=tfplan 2>&1 | tee plan.txt
          EXIT_CODE=$?
          if [ $EXIT_CODE -eq 2 ]; then
            echo "DRIFT DETECTED!"
            # Slack 通知
            curl -X POST "$SLACK_WEBHOOK" \
              -d "{\"text\":\"Terraform drift detected in production!\"}"
          fi
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
```

---

## 10. FAQ

### Q1: Terraform と Pulumi のどちらを選ぶべきか？

チームのスキルセットと要件による。インフラ専任チームがいる場合はTerraformのエコシステムの広さが有利。アプリケーション開発者がインフラも管理する場合は、Pulumi/CDKのプログラミング言語アプローチが学習コストを下げる。マルチクラウドが要件ならTerraformかPulumiを選ぶ。2023年のHashiCorpライセンス変更により、OSSライセンスを重視する場合はOpenTofuも選択肢に入る。

### Q2: IaC のテストはどうするか？

Terraformの場合、`terraform plan` による差分確認が基本。加えて、`terraform validate` で構文チェック、`tflint` でベストプラクティスチェック、`checkov` / `tfsec` でセキュリティスキャンを行う。統合テストには `terratest` (Go) を使い、実際にリソースを作成・検証・破棄するE2Eテストが可能。OPA (Open Policy Agent) でカスタムポリシーを定義し、組織のガバナンスルールを自動チェックすることもできる。

### Q3: 既存のインフラを IaC に移行するには？

Terraform の場合、`terraform import` コマンドで既存リソースを状態ファイルに取り込む。`terraformer` や `former2` (CloudFormation) といった逆生成ツールも活用できる。段階的に移行し、新規リソースは必ずIaCで作成するルールを設けるのが現実的。AWS の場合、AWS Application Composer や IaC Generator も利用できる。

### Q4: Terraform の状態ファイルが壊れた場合の対処法は？

まず `terraform state list` で現在の状態を確認する。S3 バックエンドの場合、バージョニングが有効なら過去の状態ファイルを復元できる。最悪の場合、`terraform import` で全リソースを再インポートする必要がある。状態ファイルのバックアップは必ずバージョニング付きS3で管理し、DynamoDBでロックをかけることが重要。

### Q5: IaCで管理すべきでないリソースは？

一時的なリソース(デバッグ用EC2など)、データベースの中身(テーブル・レコード)、アプリケーション設定(Feature Flag等)はIaCの対象外とすることが多い。また、頻繁に変更されるリソース(Auto Scalingの desired_count 等)は `lifecycle { ignore_changes }` で除外するか、別の仕組みで管理する。

### Q6: マルチアカウント環境でのIaCはどう設計するか？

AWS Organizations を使ったマルチアカウント環境では、(1) 管理アカウントでOrganization/OU/SCPを管理、(2) 共有サービスアカウントでRoute53/Transit Gatewayを管理、(3) 各環境アカウントでアプリケーションインフラを管理、という3層構成が一般的。Terraform の `provider` エイリアスや `assume_role` でクロスアカウント操作を行う。

---

## まとめ

| 項目 | 要点 |
|---|---|
| IaC の本質 | インフラをコードで定義し、バージョン管理する |
| 宣言的 vs 命令的 | 宣言的(Terraform等)が主流、冪等性が組み込み |
| 主要ツール | Terraform(マルチクラウド)、CDK(AWS+TS)、Pulumi(マルチ+言語) |
| 状態管理 | リモートバックエンド必須(S3+DynamoDB等) |
| モジュール化 | 再利用可能なモジュールで DRY 原則を実現 |
| テスト | 静的解析 + plan テスト + E2E テスト (terratest) |
| CI/CD 統合 | PR で plan、マージで apply、OIDC 認証 |
| ドリフト検出 | 定期的な plan 実行で乖離を自動検知 |
| ベストプラクティス | 変数化、環境分離、テスト、最小権限 |
| 必須スキル | plan の読み方、モジュール設計、セキュリティ |

---

## 次に読むべきガイド

- [GitOps](./03-gitops.md) -- IaCとGitを組み合わせた運用手法
- [クラウドデプロイ](../02-deployment/01-cloud-deployment.md) -- IaCで構築したインフラへのデプロイ
- [GitHub Actions基礎](../01-github-actions/00-actions-basics.md) -- IaCをCIで自動適用
- [CI/CD概念](./01-ci-cd-concepts.md) -- パイプライン設計の基礎

---

## 参考文献

1. Kief Morris. *Infrastructure as Code*, 2nd Edition. O'Reilly Media, 2020.
2. HashiCorp. "Terraform Documentation." https://developer.hashicorp.com/terraform/docs
3. AWS. "AWS CDK Developer Guide." https://docs.aws.amazon.com/cdk/v2/guide/
4. Pulumi. "Pulumi Documentation." https://www.pulumi.com/docs/
5. Yevgeniy Brikman. *Terraform: Up & Running*, 3rd Edition. O'Reilly Media, 2022.
6. OpenTofu. "OpenTofu Documentation." https://opentofu.org/docs/
7. Gruntwork. "Terratest Documentation." https://terratest.gruntwork.io/
8. Bridgecrew. "Checkov Documentation." https://www.checkov.io/
