# Infrastructure as Code (IaC)

> インフラの構成をコードとして管理し、バージョン管理・レビュー・自動適用を可能にする手法

## この章で学ぶこと

1. IaCの基本概念と宣言的/命令的アプローチの違いを理解する
2. Terraform、CloudFormation、CDK、Pulumiの特徴と使い分けを習得する
3. IaCのベストプラクティスとGitOps連携パターンを把握する

---

## 1. IaC とは何か

### 1.1 従来のインフラ管理 vs IaC

```
従来のインフラ管理:
  管理者がGUIコンソールで手動設定
  → ドキュメントと実態が乖離
  → 再現性がない
  → 変更履歴が追えない

IaC:
  コードでインフラを定義
  → Git でバージョン管理
  → PR でレビュー
  → CI/CD で自動適用
  → 環境の完全な再現が可能
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
+----------------------------------------------------------+
```

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
```

---

## 3. 主要 IaC ツール

### 3.1 Terraform

HashiCorp が開発した、マルチクラウド対応の IaC ツール。HCL (HashiCorp Configuration Language) で記述する。

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
    bucket = "my-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "ap-northeast-1"
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true

  tags = {
    Name = "main-vpc"
  }
}

# ECS Fargate サービス
resource "aws_ecs_service" "app" {
  name            = "my-app"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.app.id]
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
```

### 3.2 AWS CloudFormation

```yaml
# CloudFormation テンプレート
AWSTemplateFormatVersion: '2010-09-09'
Description: 'S3 Bucket with versioning'

Parameters:
  Environment:
    Type: String
    AllowedValues: [dev, staging, prod]

Resources:
  DataBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub 'my-app-${Environment}-data'
      VersioningConfiguration:
        Status: Enabled
      BucketEncryption:
        ServerSideEncryptionConfiguration:
          - ServerSideEncryptionByDefault:
              SSEAlgorithm: AES256

Outputs:
  BucketArn:
    Value: !GetAtt DataBucket.Arn
    Export:
      Name: !Sub '${Environment}-data-bucket-arn'
```

### 3.3 AWS CDK

```typescript
// AWS CDK (TypeScript) - プログラミング言語でインフラを定義
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

export class AppStack extends cdk.Stack {
  constructor(scope: cdk.App, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // VPC
    const vpc = new ec2.Vpc(this, 'AppVpc', {
      maxAzs: 2,
      natGateways: 1,
    });

    // ECS クラスター
    const cluster = new ecs.Cluster(this, 'AppCluster', { vpc });

    // Fargate サービス (L3 Construct: 高レベル抽象化)
    new ecs.FargateService(this, 'AppService', {
      cluster,
      taskDefinition: this.createTaskDefinition(),
      desiredCount: 3,
      circuitBreaker: { rollback: true },
    });
  }

  private createTaskDefinition(): ecs.FargateTaskDefinition {
    const taskDef = new ecs.FargateTaskDefinition(this, 'TaskDef', {
      memoryLimitMiB: 512,
      cpu: 256,
    });
    taskDef.addContainer('app', {
      image: ecs.ContainerImage.fromRegistry('my-app:latest'),
      portMappings: [{ containerPort: 3000 }],
      logging: ecs.LogDrivers.awsLogs({ streamPrefix: 'app' }),
    });
    return taskDef;
  }
}
```

### 3.4 Pulumi

```typescript
// Pulumi (TypeScript) - 汎用プログラミング言語でマルチクラウド対応
import * as pulumi from '@pulumi/pulumi';
import * as aws from '@pulumi/aws';

const config = new pulumi.Config();
const env = config.require('environment');

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
});

// 出力
export const bucketArn = bucket.arn;
export const bucketName = bucket.bucket;
```

---

## 4. IaC ツール比較

| 項目 | Terraform | CloudFormation | CDK | Pulumi |
|---|---|---|---|---|
| 言語 | HCL | YAML/JSON | TypeScript等 | TypeScript/Python/Go等 |
| マルチクラウド | はい | AWS のみ | AWS のみ | はい |
| 状態管理 | tfstate (S3等) | AWS管理 | CloudFormation経由 | Pulumi Cloud / S3 |
| 学習コスト | 中 | 中 | 低(開発者向き) | 低(開発者向き) |
| エコシステム | 最大 | AWS限定 | AWS限定 | 成長中 |
| ドリフト検知 | plan で検知 | drift detection | CloudFormation経由 | preview で検知 |
| 推奨場面 | マルチクラウド | AWS専用 | AWS + TypeScript | マルチクラウド + 開発者 |

---

## 5. IaC のベストプラクティス

### 5.1 モジュール化

```hcl
# modules/ecs-service/main.tf - 再利用可能なモジュール
variable "service_name" {
  type = string
}

variable "image" {
  type = string
}

variable "cpu" {
  type    = number
  default = 256
}

variable "memory" {
  type    = number
  default = 512
}

resource "aws_ecs_service" "this" {
  name            = var.service_name
  cluster         = var.cluster_id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
}

# 利用側
module "api_service" {
  source       = "./modules/ecs-service"
  service_name = "api"
  image        = "my-api:v1.2.3"
  cpu          = 512
  memory       = 1024
}

module "worker_service" {
  source       = "./modules/ecs-service"
  service_name = "worker"
  image        = "my-worker:v1.2.3"
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
│   └── rds/
├── environments/              # 環境別設定
│   ├── dev/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   ├── staging/
│   └── prod/
└── global/                    # 環境共通リソース
    ├── iam/
    └── dns/
```

---

## 6. アンチパターン

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

---

## 7. FAQ

### Q1: Terraform と Pulumi のどちらを選ぶべきか？

チームのスキルセットと要件による。インフラ専任チームがいる場合はTerraformのエコシステムの広さが有利。アプリケーション開発者がインフラも管理する場合は、Pulumi/CDKのプログラミング言語アプローチが学習コストを下げる。マルチクラウドが要件ならTerraformかPulumiを選ぶ。

### Q2: IaC のテストはどうするか？

Terraformの場合、`terraform plan` による差分確認が基本。加えて、`terraform validate` で構文チェック、`tflint` でベストプラクティスチェック、`checkov` / `tfsec` でセキュリティスキャンを行う。統合テストには `terratest` (Go) を使い、実際にリソースを作成・検証・破棄するE2Eテストが可能。

### Q3: 既存のインフラを IaC に移行するには？

Terraform の場合、`terraform import` コマンドで既存リソースを状態ファイルに取り込む。`terraformer` や `former2` (CloudFormation) といった逆生成ツールも活用できる。段階的に移行し、新規リソースは必ずIaCで作成するルールを設けるのが現実的。

---

## まとめ

| 項目 | 要点 |
|---|---|
| IaC の本質 | インフラをコードで定義し、バージョン管理する |
| 宣言的 vs 命令的 | 宣言的(Terraform等)が主流、冪等性が組み込み |
| 主要ツール | Terraform(マルチクラウド)、CDK(AWS+TS)、Pulumi(マルチ+言語) |
| 状態管理 | リモートバックエンド必須(S3+DynamoDB等) |
| ベストプラクティス | モジュール化、変数化、環境分離、テスト |
| 必須スキル | plan の読み方、モジュール設計、セキュリティ |

---

## 次に読むべきガイド

- [GitOps](./03-gitops.md) -- IaCとGitを組み合わせた運用手法
- [クラウドデプロイ](../02-deployment/01-cloud-deployment.md) -- IaCで構築したインフラへのデプロイ
- [GitHub Actions基礎](../01-github-actions/00-actions-basics.md) -- IaCをCIで自動適用

---

## 参考文献

1. Kief Morris. *Infrastructure as Code*, 2nd Edition. O'Reilly Media, 2020.
2. HashiCorp. "Terraform Documentation." https://developer.hashicorp.com/terraform/docs
3. AWS. "AWS CDK Developer Guide." https://docs.aws.amazon.com/cdk/v2/guide/
4. Pulumi. "Pulumi Documentation." https://www.pulumi.com/docs/
