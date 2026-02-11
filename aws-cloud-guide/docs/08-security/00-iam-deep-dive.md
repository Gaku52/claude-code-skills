# AWS IAM 詳解

> IAM のポリシー構文・STS・クロスアカウントアクセス・最小権限の原則を深く理解し、セキュアな AWS 環境を構築する

## この章で学ぶこと

1. **IAM ポリシーの構文と評価ロジック** — Effect、Action、Resource、Condition の詳細と評価順序
2. **STS とロールの活用** — AssumeRole、フェデレーション、一時的な認証情報
3. **クロスアカウントアクセスと最小権限設計** — マルチアカウント戦略と権限境界

---

## 1. IAM の基本概念

IAM (Identity and Access Management) は AWS のアクセス制御サービスで、「誰が」「何に対して」「何をできるか」を定義する。全ての AWS API 呼び出しは IAM による認証・認可を経る。

### 図解 1: IAM の構成要素

```
┌─────────────────────────────────────────────────────────┐
│                    IAM                                  │
│                                                         │
│  Identity (認証: 誰か)                                  │
│  ┌──────────────────────────────────────┐               │
│  │  User        → 人間のオペレーター    │               │
│  │  Group       → ユーザーの集合        │               │
│  │  Role        → サービス/外部アカウント│               │
│  │  Federation  → 外部 IdP (SAML/OIDC) │               │
│  └──────────────────────────────────────┘               │
│                                                         │
│  Policy (認可: 何ができるか)                             │
│  ┌──────────────────────────────────────┐               │
│  │  Identity-based  → User/Group/Role に│               │
│  │                     アタッチ         │               │
│  │  Resource-based  → リソースに直接    │               │
│  │                     (S3, SQS, etc.)  │               │
│  │  Permission      → 権限の上限を制限  │               │
│  │    Boundary                          │               │
│  │  SCP             → Organizations の  │               │
│  │                     アカウント制限   │               │
│  │  Session Policy  → 一時セッションの  │               │
│  │                     追加制限         │               │
│  └──────────────────────────────────────┘               │
│                                                         │
│  評価順序:                                              │
│  SCP → Permission Boundary → Identity Policy            │
│    → Resource Policy → Session Policy                   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. ポリシー構文の詳細

### コード例 1: IAM ポリシーの基本構文

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowS3ReadOnly",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::my-bucket",
        "arn:aws:s3:::my-bucket/*"
      ],
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "ap-northeast-1"
        },
        "IpAddress": {
          "aws:SourceIp": "203.0.113.0/24"
        },
        "Bool": {
          "aws:SecureTransport": "true"
        }
      }
    },
    {
      "Sid": "DenyDeleteActions",
      "Effect": "Deny",
      "Action": [
        "s3:DeleteBucket",
        "s3:DeleteObject"
      ],
      "Resource": "*"
    }
  ]
}
```

### ポリシー評価のフロー

```
リクエスト到着
    │
    ▼
┌─────────────────┐
│ 明示的 Deny     │──→ Deny あり ──→ 拒否 (最終)
│ のチェック      │
└────────┬────────┘
         │ Deny なし
         ▼
┌─────────────────┐
│ SCP チェック     │──→ Allow なし ──→ 暗黙的拒否
│ (Organizations) │
└────────┬────────┘
         │ Allow あり
         ▼
┌─────────────────┐
│ Permission      │──→ 範囲外 ──→ 暗黙的拒否
│ Boundary チェック│
└────────┬────────┘
         │ 範囲内
         ▼
┌─────────────────┐
│ Identity Policy │──→ Allow なし ──→ 暗黙的拒否
│ チェック        │
└────────┬────────┘
         │ Allow あり
         ▼
      許可 (最終)
```

### コード例 2: 高度な Condition の活用

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowTagBasedAccess",
      "Effect": "Allow",
      "Action": ["ec2:StartInstances", "ec2:StopInstances"],
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "StringEquals": {
          "ec2:ResourceTag/Environment": "${aws:PrincipalTag/Environment}"
        }
      }
    },
    {
      "Sid": "RestrictToWorkingHours",
      "Effect": "Deny",
      "Action": ["rds:DeleteDBInstance", "rds:DeleteDBCluster"],
      "Resource": "*",
      "Condition": {
        "DateGreaterThan": {
          "aws:CurrentTime": "2026-01-01T00:00:00Z"
        },
        "NumericGreaterThan": {
          "aws:MultiFactorAuthAge": "3600"
        }
      }
    },
    {
      "Sid": "RequireMFA",
      "Effect": "Deny",
      "NotAction": [
        "iam:CreateVirtualMFADevice",
        "iam:EnableMFADevice",
        "iam:GetUser",
        "iam:ListMFADevices",
        "iam:ListVirtualMFADevices",
        "sts:GetSessionToken"
      ],
      "Resource": "*",
      "Condition": {
        "BoolIfExists": {
          "aws:MultiFactorAuthPresent": "false"
        }
      }
    }
  ]
}
```

---

## 3. STS (Security Token Service)

### 図解 2: AssumeRole のフロー

```
1. 同一アカウント内の AssumeRole:
   ┌─────────┐  AssumeRole   ┌──────────┐
   │ EC2     │ ──────────→   │ IAM Role │
   │ (Role A)│               │ (Role B) │
   │         │ ←──────────── │          │
   │         │  一時認証情報  │          │
   └─────────┘               └──────────┘

2. クロスアカウント AssumeRole:
   Account A (111111111111)      Account B (222222222222)
   ┌─────────────┐              ┌────────────────┐
   │ IAM User/   │  AssumeRole  │ IAM Role       │
   │ Role        │ ────────→    │ (Trust Policy  │
   │             │              │  で A を許可)  │
   │             │ ←────────    │                │
   │             │ 一時認証情報 │                │
   └─────────────┘              └────────────────┘

3. フェデレーション (OIDC/SAML):
   ┌──────────┐  認証  ┌──────┐  AssumeRoleWith  ┌──────────┐
   │ ユーザー │ ────→ │ IdP  │  WebIdentity     │ IAM Role │
   │          │       │(Google│ ────────────→    │          │
   │          │       │/Okta) │                  │ AWS      │
   │          │       └──────┘                   │ リソース │
   └──────────┘                                  └──────────┘
```

### コード例 3: AssumeRole の実装

```python
import boto3

# クロスアカウントの AssumeRole
def assume_cross_account_role(
    role_arn: str,
    session_name: str,
    external_id: str = None,
    duration_seconds: int = 3600,
) -> boto3.Session:
    """別アカウントのロールを引き受けてセッションを返す"""
    sts = boto3.client("sts")

    params = {
        "RoleArn": role_arn,
        "RoleSessionName": session_name,
        "DurationSeconds": duration_seconds,
    }
    if external_id:
        params["ExternalId"] = external_id

    response = sts.assume_role(**params)
    credentials = response["Credentials"]

    return boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

# 使用例: Account B の S3 にアクセス
session_b = assume_cross_account_role(
    role_arn="arn:aws:iam::222222222222:role/CrossAccountS3Access",
    session_name="my-app-session",
    external_id="UniqueExternalId123",
)

s3 = session_b.client("s3")
objects = s3.list_objects_v2(Bucket="account-b-bucket")
```

### コード例 4: 信頼ポリシー (Trust Policy)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCrossAccountAccess",
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::111111111111:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "UniqueExternalId123"
        }
      }
    },
    {
      "Sid": "AllowGitHubOIDC",
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:my-org/my-repo:*"
        }
      }
    },
    {
      "Sid": "AllowEC2",
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

---

## 4. 最小権限の実装

### コード例 5: 最小権限ポリシーの段階的構築

```bash
# 1. IAM Access Analyzer で必要な権限を分析
aws accessanalyzer create-analyzer \
  --analyzer-name my-analyzer \
  --type ACCOUNT

# 2. CloudTrail から実際に使用されたアクションを抽出
aws accessanalyzer generate-findings-report \
  --analyzer-arn arn:aws:access-analyzer:ap-northeast-1:123456789012:analyzer/my-analyzer

# 3. IAM Access Analyzer でポリシーを生成
aws accessanalyzer generate-policy \
  --policy-generation-details '{
    "trailProperties": {
      "cloudTrailArn": "arn:aws:cloudtrail:ap-northeast-1:123456789012:trail/my-trail",
      "regions": ["ap-northeast-1"],
      "allRegions": false
    },
    "principalArn": "arn:aws:iam::123456789012:role/MyAppRole"
  }'

# 4. 未使用の権限を確認
aws iam generate-service-last-accessed-details \
  --arn arn:aws:iam::123456789012:role/MyAppRole

aws iam get-service-last-accessed-details \
  --job-id "job-id-from-above"
```

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DynamoDBMinimalAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:ap-northeast-1:123456789012:table/Users",
        "arn:aws:dynamodb:ap-northeast-1:123456789012:table/Users/index/*"
      ]
    },
    {
      "Sid": "S3SpecificBucketAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::my-app-uploads/*",
      "Condition": {
        "StringEquals": {
          "s3:x-amz-server-side-encryption": "aws:kms"
        }
      }
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:ap-northeast-1:123456789012:log-group:/aws/lambda/my-function:*"
    }
  ]
}
```

---

## 5. Permission Boundary

### 図解 3: Permission Boundary の仕組み

```
Permission Boundary の効果:

  Identity Policy (ロールに付与された権限):
  ┌──────────────────────────────────────┐
  │  S3:*                                │
  │  DynamoDB:*                          │
  │  Lambda:*                            │
  │  EC2:*          ← 広い権限           │
  │  IAM:*                               │
  └──────────────────────────────────────┘

  Permission Boundary (権限の上限):
  ┌──────────────────────────────────────┐
  │  S3:*                                │
  │  DynamoDB:*                          │
  │  Lambda:*       ← 許可範囲の上限     │
  │  CloudWatch:*                        │
  └──────────────────────────────────────┘

  有効な権限 (交差部分):
  ┌──────────────────────────────────────┐
  │  S3:*                                │
  │  DynamoDB:*     ← 両方で許可された   │
  │  Lambda:*          権限のみ有効      │
  └──────────────────────────────────────┘

  EC2:* → Boundary にないため拒否
  IAM:* → Boundary にないため拒否
```

### コード例 6: Permission Boundary の設定

```bash
# Permission Boundary ポリシーの作成
aws iam create-policy \
  --policy-name DeveloperBoundary \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "AllowedServices",
        "Effect": "Allow",
        "Action": [
          "s3:*",
          "dynamodb:*",
          "lambda:*",
          "logs:*",
          "cloudwatch:*",
          "sqs:*",
          "sns:*",
          "apigateway:*",
          "xray:*"
        ],
        "Resource": "*"
      },
      {
        "Sid": "DenyDangerousActions",
        "Effect": "Deny",
        "Action": [
          "iam:CreateUser",
          "iam:DeleteUser",
          "iam:CreateRole",
          "iam:DeleteRole",
          "organizations:*",
          "account:*"
        ],
        "Resource": "*"
      }
    ]
  }'

# ロールに Permission Boundary を設定
aws iam put-role-permissions-boundary \
  --role-name DeveloperRole \
  --permissions-boundary "arn:aws:iam::123456789012:policy/DeveloperBoundary"
```

---

## 6. 比較表

### 比較表 1: ポリシータイプ比較

| ポリシータイプ | 適用対象 | 管理者 | 用途 |
|---------------|---------|--------|------|
| **Identity-based** | User/Group/Role | アカウント管理者 | 通常のアクセス制御 |
| **Resource-based** | S3/SQS/Lambda 等 | リソース所有者 | クロスアカウント許可 |
| **Permission Boundary** | User/Role | 管理者 | 委譲の上限設定 |
| **SCP** | OU/Account | Org 管理者 | 組織レベルの制限 |
| **Session Policy** | AssumeRole 時 | 呼び出し元 | 一時的な制限 |
| **ACL** | S3/VPC | リソース所有者 | レガシー (非推奨) |

### 比較表 2: 認証方式比較

| 方式 | 安全性 | 推奨度 | 用途 |
|------|--------|--------|------|
| **IAM Role (EC2/Lambda)** | 高 | 最推奨 | AWS 内のサービス間 |
| **OIDC Federation** | 高 | 推奨 | GitHub Actions, Google 等 |
| **SAML Federation** | 高 | 推奨 | 企業 SSO (Okta, Azure AD) |
| **IAM Identity Center** | 高 | 推奨 | マルチアカウント管理 |
| **IAM User + MFA** | 中 | 条件付き | 管理者のコンソールアクセス |
| **アクセスキー** | 低 | 非推奨 | レガシー/外部システム |
| **ルートアカウント** | 最低 | 禁止 | 絶対に日常使用しない |

---

## 7. アンチパターン

### アンチパターン 1: ワイルドカード権限の付与

```
[悪い例]
  {
    "Effect": "Allow",
    "Action": "*",
    "Resource": "*"
  }
  → 全リソースに全操作が可能
  → 情報漏洩、リソース削除、コスト爆発のリスク

[良い例]
  {
    "Effect": "Allow",
    "Action": [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:Query"
    ],
    "Resource": "arn:aws:dynamodb:ap-northeast-1:123456789012:table/Users"
  }
  → 必要なアクション、必要なリソースのみ許可
  → IAM Access Analyzer で未使用権限を定期的に削除
```

### アンチパターン 2: 長期アクセスキーの使用

```
[悪い例]
  # .env ファイルにアクセスキーを保存
  AWS_ACCESS_KEY_ID=AKIA...
  AWS_SECRET_ACCESS_KEY=xxx...
  → キーの漏洩リスク
  → ローテーションの管理負荷
  → 退職者のキー無効化漏れ

[良い例]
  # EC2/ECS/Lambda → IAM ロール
  # ローカル開発 → AWS SSO (Identity Center)
  aws sso login --profile dev

  # CI/CD → OIDC フェデレーション
  # GitHub Actions の例:
  - uses: aws-actions/configure-aws-credentials@v4
    with:
      role-to-assume: arn:aws:iam::123456789012:role/GitHubRole
      aws-region: ap-northeast-1

  原則: 一時的な認証情報のみを使用
```

---

## 8. FAQ

### Q1: IAM ロールと IAM ユーザーのどちらを使うべきですか？

**A:** 原則として IAM ロールを使用する。AWS サービス（EC2, Lambda, ECS）には必ずロールをアタッチする。人間のオペレーターは IAM Identity Center (旧 SSO) でフェデレーション認証する。IAM ユーザーを作成するのは、外部システム連携で他の手段がない場合のみに限定し、MFA とアクセスキーローテーションを必須にする。

### Q2: クロスアカウントアクセスで ExternalId が必要な理由は？

**A:** ExternalId は「混乱した代理 (Confused Deputy)」攻撃を防止する。サードパーティサービスが顧客の AWS アカウントにアクセスする場合、ExternalId がないと、攻撃者が同じサードパーティサービスを使って他の顧客のロールを引き受けられる可能性がある。ExternalId は各顧客固有の値を使い、ロールの信頼ポリシーの Condition に設定する。

### Q3: 本番環境でルートアカウントを保護する方法は？

**A:** (1) ルートアカウントに強力なパスワードを設定、(2) ハードウェア MFA デバイスを有効化、(3) ルートアカウントのアクセスキーを削除、(4) ルートアカウントの使用は AWS Organizations の作成やアカウントの支払い設定変更など、IAM では実行できない操作のみに限定、(5) CloudTrail でルートアカウントの使用を監視し、アラートを設定する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 最小権限 | 必要なアクション・リソースのみ許可。IAM Access Analyzer で検証 |
| 認証方式 | IAM ロール + OIDC フェデレーション。長期キー使用禁止 |
| ポリシー評価 | 明示的 Deny > SCP > Boundary > Allow の順で評価 |
| クロスアカウント | AssumeRole + ExternalId + 条件付き信頼ポリシー |
| Permission Boundary | 権限委譲時の安全ガード。開発者に自律性を与えつつ制限 |
| 監視 | CloudTrail + IAM Access Analyzer で異常検知 |
| ルートアカウント | ハードウェア MFA + 使用禁止 + 監視 |

---

## 次に読むべきガイド

- [01-secrets-management.md](./01-secrets-management.md) — IAM と連携するシークレット管理
- [02-waf-shield.md](./02-waf-shield.md) — アプリケーション層のセキュリティ
- [01-well-architected.md](../09-cost/01-well-architected.md) — セキュリティの柱

---

## 参考文献

1. **AWS 公式ドキュメント** — IAM ユーザーガイド
   https://docs.aws.amazon.com/IAM/latest/UserGuide/
2. **AWS IAM ベストプラクティス** — セキュリティのベストプラクティス
   https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html
3. **AWS re:Invent — Become an IAM Policy Master** — IAM ポリシーの高度な設計
   https://www.youtube.com/watch?v=YQsK4MtsELU
