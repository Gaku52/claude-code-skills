# クラウドセキュリティ基礎

> 責任共有モデルの正しい理解、IAM による最小権限アクセス制御、保存時・転送時の暗号化まで、クラウド環境のセキュリティ基盤を体系的に学ぶ

## この章で学ぶこと

1. **責任共有モデル** — クラウドプロバイダとユーザの責任分担の理解
2. **IAM (Identity and Access Management)** — 最小権限の原則に基づくアクセス制御
3. **データ暗号化** — 保存時 (at rest) と転送時 (in transit) の暗号化戦略

---

## 1. 責任共有モデル

### サービスモデル別の責任分担

```
+----------------------------------------------------------+
|              責任共有モデル                                 |
|----------------------------------------------------------|
|                  IaaS    PaaS    SaaS                     |
|                  (EC2)   (Lambda) (Office365)             |
|----------------------------------------------------------|
| データ          | User | User  | User                    |
| アプリケーション | User | User  | Provider                |
| ランタイム      | User | Provider| Provider               |
| ミドルウェア    | User | Provider| Provider               |
| OS             | User | Provider| Provider               |
| 仮想化         | Provider| Provider| Provider             |
| ネットワーク   | Provider| Provider| Provider              |
| 物理           | Provider| Provider| Provider              |
|----------------------------------------------------------|
| User = ユーザ責任  |  Provider = クラウド事業者責任         |
+----------------------------------------------------------+
```

### 責任共有でよくある誤解

```
+----------------------------------------------------------+
|  誤解                        | 正しい理解                  |
|----------------------------------------------------------+
|  「クラウドだから安全」        | インフラは安全、設定は自己責任 |
|  「暗号化はクラウドが自動で」  | 有効化はユーザが明示的に行う   |
|  「アクセス制御は不要」        | IAM 設定はユーザの責任        |
|  「バックアップは自動」        | 設定・テストはユーザの責任     |
|  「コンプライアンスも自動」    | 認証取得と維持はユーザの責任   |
+----------------------------------------------------------+
```

---

## 2. IAM (Identity and Access Management)

### IAM の構成要素

```
+----------------------------------------------------------+
|                    IAM の構成要素                           |
|----------------------------------------------------------|
|                                                          |
|  [アイデンティティ]                                       |
|  +-- ユーザ (人間のオペレータ)                             |
|  +-- サービスアカウント (アプリケーション)                   |
|  +-- ロール (一時的な権限の集合)                           |
|  +-- グループ (ユーザの集合)                               |
|                                                          |
|  [ポリシー]                                               |
|  +-- アイデンティティベースポリシー (誰に何を許可)           |
|  +-- リソースベースポリシー (何に誰がアクセス可能)           |
|  +-- 権限境界 (最大権限の制限)                             |
|  +-- SCP (組織全体の制限)                                 |
|                                                          |
|  [認証方式]                                               |
|  +-- パスワード + MFA                                     |
|  +-- アクセスキー (プログラムアクセス)                      |
|  +-- 一時的セキュリティ認証情報 (STS)                      |
|  +-- SSO / SAML / OIDC 連携                              |
+----------------------------------------------------------+
```

### 最小権限ポリシーの設計

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3ReadSpecificBucket",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-app-data",
                "arn:aws:s3:::my-app-data/*"
            ],
            "Condition": {
                "StringEquals": {
                    "aws:RequestedRegion": "ap-northeast-1"
                },
                "IpAddress": {
                    "aws:SourceIp": "10.0.0.0/8"
                }
            }
        },
        {
            "Sid": "DenyUnencryptedUploads",
            "Effect": "Deny",
            "Action": "s3:PutObject",
            "Resource": "arn:aws:s3:::my-app-data/*",
            "Condition": {
                "StringNotEquals": {
                    "s3:x-amz-server-side-encryption": "aws:kms"
                }
            }
        }
    ]
}
```

### IAM ロールの活用 (EC2/Lambda)

```python
import boto3

# EC2 インスタンスプロファイル経由でアクセス
# (アクセスキーのハードコーディング不要)
s3 = boto3.client('s3')  # IAM ロールの認証情報を自動取得

# Lambda 用ロールの最小権限ポリシー (Terraform)
"""
resource "aws_iam_role" "lambda_role" {
  name = "my-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["dynamodb:GetItem", "dynamodb:PutItem"]
        Resource = "arn:aws:dynamodb:*:*:table/my-table"
      },
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}
"""
```

### マルチアカウント戦略

```
+----------------------------------------------------------+
|  AWS Organizations                                       |
|                                                          |
|  +-- Management Account (請求・組織管理のみ)              |
|  |                                                       |
|  +-- Security OU                                         |
|  |   +-- Security Account (GuardDuty, Security Hub)      |
|  |   +-- Log Archive Account (CloudTrail, Config)        |
|  |                                                       |
|  +-- Workloads OU                                        |
|  |   +-- Production Account                              |
|  |   +-- Staging Account                                 |
|  |   +-- Development Account                             |
|  |                                                       |
|  +-- Sandbox OU                                          |
|      +-- Developer Sandbox Accounts                      |
|                                                          |
|  SCP (Service Control Policy) で全アカウントに制限適用     |
+----------------------------------------------------------+
```

---

## 3. データ暗号化

### 暗号化の分類

```
+----------------------------------------------------------+
|                暗号化の層                                  |
|----------------------------------------------------------|
|                                                          |
|  転送時の暗号化 (In Transit):                              |
|  +-- TLS 1.2/1.3 (HTTPS, gRPC over TLS)                 |
|  +-- VPN (IPsec, WireGuard)                              |
|  +-- SSH トンネル                                        |
|                                                          |
|  保存時の暗号化 (At Rest):                                |
|  +-- サーバサイド暗号化 (SSE)                              |
|  |   +-- SSE-S3 (S3 管理キー)                            |
|  |   +-- SSE-KMS (KMS 管理キー)                          |
|  |   +-- SSE-C (顧客提供キー)                             |
|  +-- クライアントサイド暗号化 (CSE)                        |
|  +-- ディスク暗号化 (EBS, RDS)                            |
|                                                          |
|  処理時の暗号化 (In Use):                                  |
|  +-- AWS Nitro Enclaves                                  |
|  +-- Confidential Computing                              |
+----------------------------------------------------------+
```

### S3 バケットのセキュリティ設定

```python
import boto3

s3 = boto3.client('s3')

# デフォルト暗号化の設定
s3.put_bucket_encryption(
    Bucket='my-secure-bucket',
    ServerSideEncryptionConfiguration={
        'Rules': [{
            'ApplyServerSideEncryptionByDefault': {
                'SSEAlgorithm': 'aws:kms',
                'KMSMasterKeyID': 'arn:aws:kms:ap-northeast-1:123456:key/xxx',
            },
            'BucketKeyEnabled': True,  # コスト削減
        }]
    },
)

# パブリックアクセスブロック
s3.put_public_access_block(
    Bucket='my-secure-bucket',
    PublicAccessBlockConfiguration={
        'BlockPublicAcls': True,
        'IgnorePublicAcls': True,
        'BlockPublicPolicy': True,
        'RestrictPublicBuckets': True,
    },
)

# バケットポリシー: HTTPS のみ許可
bucket_policy = {
    "Version": "2012-10-17",
    "Statement": [{
        "Sid": "DenyHTTP",
        "Effect": "Deny",
        "Principal": "*",
        "Action": "s3:*",
        "Resource": [
            "arn:aws:s3:::my-secure-bucket",
            "arn:aws:s3:::my-secure-bucket/*",
        ],
        "Condition": {
            "Bool": {"aws:SecureTransport": "false"}
        }
    }]
}
```

---

## 4. ネットワークセキュリティ (クラウド)

### VPC の設計

```
+----------------------------------------------------------+
|  VPC (10.0.0.0/16)                                       |
|                                                          |
|  +-- Public Subnet (10.0.1.0/24)                         |
|  |   +-- NAT Gateway                                    |
|  |   +-- ALB                                            |
|  |   Route: 0.0.0.0/0 → IGW                             |
|  |                                                       |
|  +-- Private Subnet (10.0.2.0/24)                        |
|  |   +-- EC2 / ECS                                      |
|  |   Route: 0.0.0.0/0 → NAT GW                          |
|  |                                                       |
|  +-- Data Subnet (10.0.3.0/24)                           |
|  |   +-- RDS / ElastiCache                               |
|  |   Route: ローカルのみ                                  |
|  |                                                       |
|  +-- VPC Endpoints (S3, DynamoDB, KMS)                   |
|      → インターネットを経由せず AWS サービスにアクセス       |
+----------------------------------------------------------+
```

---

## 5. セキュリティサービスの全体像

### クラウドセキュリティサービス比較

| カテゴリ | AWS | GCP | Azure |
|---------|-----|-----|-------|
| 脅威検知 | GuardDuty | Security Command Center | Defender for Cloud |
| 統合管理 | Security Hub | SCC Premium | Defender CSPM |
| 監査ログ | CloudTrail | Cloud Audit Logs | Activity Log |
| 設定監査 | Config | Cloud Asset Inventory | Policy |
| WAF | AWS WAF | Cloud Armor | Azure WAF |
| KMS | AWS KMS | Cloud KMS | Key Vault |
| シークレット | Secrets Manager | Secret Manager | Key Vault Secrets |

---

## 6. アンチパターン

### アンチパターン 1: ルートアカウントの日常使用

```
NG:
  → root ユーザで日常的にログイン
  → root のアクセスキーを作成してスクリプトで使用

OK:
  → root には MFA を設定し緊急時のみ使用
  → 日常運用は IAM ユーザ/ロール経由
  → root のアクセスキーは作成しない
  → 管理操作は Organizations の管理アカウントから
```

### アンチパターン 2: セキュリティグループの 0.0.0.0/0 許可

```hcl
# NG: 全ポートを全世界に公開
resource "aws_security_group_rule" "allow_all" {
  type        = "ingress"
  from_port   = 0
  to_port     = 65535
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]
}

# OK: 必要なポートと送信元のみ
resource "aws_security_group_rule" "allow_https" {
  type        = "ingress"
  from_port   = 443
  to_port     = 443
  protocol    = "tcp"
  cidr_blocks = ["0.0.0.0/0"]  # HTTPS は公開 OK
}

resource "aws_security_group_rule" "allow_ssh" {
  type        = "ingress"
  from_port   = 22
  to_port     = 22
  protocol    = "tcp"
  cidr_blocks = ["10.0.0.0/8"]  # 社内のみ
}
```

---

## 7. FAQ

### Q1. クラウドのセキュリティは自社データセンターより安全か?

インフラの物理セキュリティ、DDoS 対策、パッチ適用は大手クラウドプロバイダの方が優れている。ただし設定ミスによるデータ漏洩は依然としてユーザ責任である。S3 バケットの公開設定ミスによるデータ流出は後を絶たない。

### Q2. マルチクラウド環境のセキュリティはどう管理するか?

CSPM (Cloud Security Posture Management) ツール (Prisma Cloud, Wiz, etc.) でマルチクラウドの設定を統合監視する。IAM は各クラウドの特性を理解した上で統一的なポリシーを設計する。共通の IaC (Terraform) で全環境を管理し、セキュリティポリシーをコードで統一する。

### Q3. IAM ポリシーが複雑になりすぎた場合はどうするか?

AWS IAM Access Analyzer でポリシーの分析と未使用権限の特定を行う。許可されているが使われていない権限を削除し、必要最小限まで絞り込む。ポリシーの設計はロールベースで行い、個別ユーザへのインラインポリシーは避ける。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 責任共有モデル | インフラはプロバイダ、設定・データはユーザの責任 |
| IAM | 最小権限の原則、ロールベース、MFA 必須 |
| 暗号化 | 保存時 (KMS) + 転送時 (TLS) を常に有効化 |
| ネットワーク | VPC セグメンテーション + VPC Endpoints |
| マルチアカウント | 環境別にアカウントを分離し SCP で統制 |
| 監視 | CloudTrail + Config + GuardDuty を必ず有効化 |

---

## 次に読むべきガイド

- [AWSセキュリティ](./01-aws-security.md) — AWS 固有のセキュリティサービスの詳細
- [IaCセキュリティ](./02-infrastructure-as-code-security.md) — インフラ設定のセキュリティ自動チェック
- [鍵管理](../02-cryptography/02-key-management.md) — KMS を含む暗号鍵の管理手法

---

## 参考文献

1. **AWS Well-Architected Framework — Security Pillar** — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
2. **CIS Benchmarks for Cloud Providers** — https://www.cisecurity.org/cis-benchmarks
3. **NIST SP 800-144 — Guidelines on Security and Privacy in Public Cloud Computing** — https://csrc.nist.gov/publications/detail/sp/800-144/final
4. **CSA Cloud Controls Matrix** — https://cloudsecurityalliance.org/research/cloud-controls-matrix/
