# AWS IAM 詳解

> IAM のポリシー構文・STS・クロスアカウントアクセス・最小権限の原則を深く理解し、セキュアな AWS 環境を構築する

## この章で学ぶこと

1. **IAM ポリシーの構文と評価ロジック** — Effect、Action、Resource、Condition の詳細と評価順序
2. **STS とロールの活用** — AssumeRole、フェデレーション、一時的な認証情報
3. **クロスアカウントアクセスと最小権限設計** — マルチアカウント戦略と権限境界
4. **IAM Identity Center と組織管理** — SSO、SCIM プロビジョニング、SCP の実践
5. **IAM の監視・監査・自動化** — Access Analyzer、CloudTrail、自動修復

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

### 1.1 IAM のグローバル性とリージョン

IAM はグローバルサービスであり、リージョンに依存しない。ただし、一部の機能にはリージョン固有の考慮が必要になる。

```
┌──────────────────────────────────────────────────────────┐
│              IAM のグローバル/リージョン特性                │
│                                                          │
│  グローバル:                                              │
│  ├── IAM User / Group / Role / Policy                    │
│  ├── IAM Identity Center (組織レベル)                     │
│  └── STS (sts.amazonaws.com)                             │
│                                                          │
│  リージョン固有:                                          │
│  ├── STS リージョナルエンドポイント (推奨)                 │
│  │   → sts.ap-northeast-1.amazonaws.com                  │
│  ├── IAM Access Analyzer (リージョンごとに作成)           │
│  └── VPC エンドポイント (リージョンごとに作成)             │
│                                                          │
│  ベストプラクティス:                                      │
│  STS はリージョナルエンドポイントを使う                    │
│  → レイテンシ削減 + グローバルエンドポイント障害の回避      │
└──────────────────────────────────────────────────────────┘
```

### 1.2 IAM User / Group / Role の使い分け

```bash
# IAM User の作成（非推奨だがレガシー対応）
aws iam create-user --user-name legacy-service-user
aws iam create-access-key --user-name legacy-service-user

# IAM Group の作成とユーザー追加
aws iam create-group --group-name Developers
aws iam add-user-to-group --group-name Developers --user-name dev-user-01
aws iam attach-group-policy \
  --group-name Developers \
  --policy-arn arn:aws:iam::123456789012:policy/DeveloperAccess

# IAM Role の作成（推奨）
aws iam create-role \
  --role-name MyAppRole \
  --assume-role-policy-document file://trust-policy.json \
  --tags Key=Environment,Value=Production Key=Team,Value=Backend

# IAM Role にポリシーをアタッチ
aws iam attach-role-policy \
  --role-name MyAppRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# インラインポリシーの追加（特定ロール固有の権限）
aws iam put-role-policy \
  --role-name MyAppRole \
  --policy-name CustomDynamoDBAccess \
  --policy-document file://dynamodb-policy.json
```

### 1.3 IAM ユーザーの棚卸しと不要リソース削除

```bash
#!/bin/bash
# IAM ユーザーの棚卸しスクリプト
# 90日以上アクセスキーを使っていないユーザーを検出

echo "=== IAM User Access Key Audit ==="
echo "Date: $(date)"
echo ""

aws iam generate-credential-report > /dev/null 2>&1
sleep 5

aws iam get-credential-report \
  --query 'Content' \
  --output text | base64 -d | \
  awk -F',' 'NR>1 {
    user=$1;
    key1_active=$9;
    key1_last_used=$11;
    key2_active=$14;
    key2_last_used=$16;
    mfa=$8;
    printf "User: %-30s MFA: %-5s Key1Active: %-5s Key1LastUsed: %-20s Key2Active: %-5s Key2LastUsed: %-20s\n",
      user, mfa, key1_active, key1_last_used, key2_active, key2_last_used
  }'

echo ""
echo "=== 90日以上未使用のアクセスキー ==="
THRESHOLD=$(date -d "90 days ago" +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-90d +%Y-%m-%dT%H:%M:%S)

for user in $(aws iam list-users --query 'Users[*].UserName' --output text); do
  for key_id in $(aws iam list-access-keys --user-name "$user" \
    --query 'AccessKeyMetadata[?Status==`Active`].AccessKeyId' --output text); do

    last_used=$(aws iam get-access-key-last-used --access-key-id "$key_id" \
      --query 'AccessKeyLastUsed.LastUsedDate' --output text)

    if [[ "$last_used" < "$THRESHOLD" ]] || [[ "$last_used" == "None" ]]; then
      echo "STALE: User=$user KeyId=$key_id LastUsed=$last_used"
    fi
  done
done
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

### 2.1 ポリシー要素の詳細解説

| 要素 | 必須 | 説明 | 例 |
|------|------|------|-----|
| **Version** | 推奨 | ポリシー言語のバージョン | `"2012-10-17"` (最新・推奨) |
| **Statement** | 必須 | 1つ以上のアクセス制御ルール | 配列形式 |
| **Sid** | 任意 | ステートメントの識別子 | `"AllowS3Read"` |
| **Effect** | 必須 | 許可か拒否か | `"Allow"` or `"Deny"` |
| **Principal** | 条件付き | 対象のエンティティ (Resource Policy で使用) | `{"AWS": "arn:aws:iam::..."}` |
| **Action** | 必須 | 許可/拒否するAPI操作 | `"s3:GetObject"` |
| **NotAction** | 任意 | 指定以外の操作 (Action の逆) | `"iam:*"` 以外を許可 |
| **Resource** | 必須 | 対象リソースのARN | `"arn:aws:s3:::bucket/*"` |
| **NotResource** | 任意 | 指定以外のリソース | 特定リソース以外に適用 |
| **Condition** | 任意 | 条件付きアクセス制御 | IP制限、MFA要求等 |

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

### 2.2 同一アカウント vs クロスアカウントの評価差異

```
┌──────────────────────────────────────────────────────────┐
│  同一アカウントの場合:                                     │
│  Identity Policy OR Resource Policy のいずれかで Allow     │
│  → アクセス許可 (和集合)                                  │
│                                                          │
│  例: S3 バケットポリシーで Allow されていれば、             │
│      Identity Policy に Allow がなくてもアクセス可能       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  クロスアカウントの場合:                                   │
│  Identity Policy AND Resource Policy の両方で Allow 必要   │
│  → 両方なければアクセス拒否 (積集合)                       │
│                                                          │
│  例: Account B の S3 バケットポリシーで Account A を許可    │
│      + Account A の Identity Policy で S3 Allow            │
│      → 両方必要                                           │
└──────────────────────────────────────────────────────────┘
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

### 2.3 Condition 演算子の一覧と実用例

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ConditionOperatorExamples",
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "s3:prefix": "home/${aws:username}/"
        },
        "StringLike": {
          "s3:prefix": ["home/*", "shared/*"]
        },
        "StringNotEquals": {
          "aws:RequestedRegion": "us-east-1"
        },
        "ArnLike": {
          "aws:SourceArn": "arn:aws:sns:*:123456789012:*"
        },
        "IpAddress": {
          "aws:SourceIp": ["203.0.113.0/24", "198.51.100.0/24"]
        },
        "NotIpAddress": {
          "aws:SourceIp": "0.0.0.0/0"
        },
        "DateLessThan": {
          "aws:CurrentTime": "2026-12-31T23:59:59Z"
        },
        "NumericLessThanEquals": {
          "s3:max-keys": "100"
        },
        "Bool": {
          "aws:SecureTransport": "true"
        },
        "Null": {
          "aws:TokenIssueTime": "false"
        },
        "ForAllValues:StringEquals": {
          "aws:TagKeys": ["Environment", "Project"]
        },
        "ForAnyValue:StringLike": {
          "aws:PrincipalOrgPaths": ["o-xxx/r-xxx/ou-xxx/*"]
        }
      }
    }
  ]
}
```

### 2.4 ポリシー変数とタグベースアクセス制御 (ABAC)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ABACFullExample",
      "Effect": "Allow",
      "Action": [
        "ec2:StartInstances",
        "ec2:StopInstances",
        "ec2:RebootInstances",
        "ec2:TerminateInstances"
      ],
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "StringEquals": {
          "ec2:ResourceTag/Project": "${aws:PrincipalTag/Project}",
          "ec2:ResourceTag/Environment": "${aws:PrincipalTag/Environment}"
        }
      }
    },
    {
      "Sid": "AllowCreateTaggedResources",
      "Effect": "Allow",
      "Action": "ec2:RunInstances",
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "StringEquals": {
          "aws:RequestTag/Project": "${aws:PrincipalTag/Project}",
          "aws:RequestTag/Environment": "${aws:PrincipalTag/Environment}"
        },
        "ForAllValues:StringEquals": {
          "aws:TagKeys": ["Project", "Environment", "Name"]
        }
      }
    },
    {
      "Sid": "DenyUntaggedResources",
      "Effect": "Deny",
      "Action": "ec2:RunInstances",
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "Null": {
          "aws:RequestTag/Project": "true",
          "aws:RequestTag/Environment": "true"
        }
      }
    },
    {
      "Sid": "AllowS3HomeDirectory",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
      "Resource": "arn:aws:s3:::company-data/home/${aws:PrincipalTag/Department}/${aws:userid}/*"
    }
  ]
}
```

### 2.5 ABAC vs RBAC の比較

```
┌──────────────────────────────────────────────────────────┐
│  RBAC (Role-Based Access Control)                         │
│                                                          │
│  ┌──────────────┐     ┌──────────────┐                   │
│  │ Project-A    │     │ Project-B    │                   │
│  │ Developer    │     │ Developer    │                   │
│  │ Role         │     │ Role         │                   │
│  └──────┬───────┘     └──────┬───────┘                   │
│         │                    │                           │
│         ▼                    ▼                           │
│  プロジェクトごとに       プロジェクトごとに              │
│  ロールを作成             ロールを作成                   │
│  → ロール数が増大         → 管理が複雑化                │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  ABAC (Attribute-Based Access Control)                    │
│                                                          │
│  ┌──────────────┐                                        │
│  │ Developer    │  Tag: Project=A                        │
│  │ Role         │  Tag: Environment=prod                 │
│  │ (共通)       │                                        │
│  └──────┬───────┘                                        │
│         │                                                │
│         ▼                                                │
│  1つのポリシーで          タグの値でアクセス範囲を         │
│  全プロジェクトに対応     動的に制御                       │
│  → ポリシー数最小化       → スケーラブル                  │
└──────────────────────────────────────────────────────────┘
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
from datetime import datetime

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

### 3.1 AssumeRole チェーン（ロールの連鎖）

```python
def assume_role_chain(role_chain: list[dict]) -> boto3.Session:
    """複数のロールを連鎖的に引き受ける

    Args:
        role_chain: [{"role_arn": "...", "session_name": "...", "external_id": "..."}]

    Returns:
        最終ロールのセッション
    """
    session = boto3.Session()  # 初期セッション（元のクレデンシャル）

    for i, role_config in enumerate(role_chain):
        sts = session.client("sts")
        params = {
            "RoleArn": role_config["role_arn"],
            "RoleSessionName": role_config.get("session_name", f"chain-step-{i}"),
            "DurationSeconds": role_config.get("duration", 3600),
        }
        if "external_id" in role_config:
            params["ExternalId"] = role_config["external_id"]

        response = sts.assume_role(**params)
        creds = response["Credentials"]

        session = boto3.Session(
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
        )
        print(f"Step {i+1}: Assumed {role_config['role_arn']}")
        print(f"  Expiration: {creds['Expiration']}")

    return session

# 使用例: Management Account → Security Account → Target Account
final_session = assume_role_chain([
    {
        "role_arn": "arn:aws:iam::111111111111:role/SecurityHubRole",
        "session_name": "security-audit",
    },
    {
        "role_arn": "arn:aws:iam::222222222222:role/AuditTargetRole",
        "session_name": "target-audit",
        "external_id": "audit-2026",
    },
])
```

### 3.2 STS のセッションタグとトランジティブタグ

```python
def assume_role_with_tags(
    role_arn: str,
    session_name: str,
    tags: dict[str, str],
    transitive_keys: list[str] = None,
) -> boto3.Session:
    """セッションタグ付きで AssumeRole"""
    sts = boto3.client("sts")

    session_tags = [
        {"Key": k, "Value": v} for k, v in tags.items()
    ]

    params = {
        "RoleArn": role_arn,
        "RoleSessionName": session_name,
        "Tags": session_tags,
    }

    if transitive_keys:
        params["TransitiveTagKeys"] = transitive_keys

    response = sts.assume_role(**params)
    creds = response["Credentials"]

    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
    )

# セッションタグでコスト配分・アクセス制御を動的に設定
session = assume_role_with_tags(
    role_arn="arn:aws:iam::123456789012:role/DeveloperRole",
    session_name="dev-session",
    tags={
        "Project": "payment-service",
        "CostCenter": "engineering-tokyo",
        "Environment": "staging",
    },
    transitive_keys=["Project", "CostCenter"],  # ロールチェーンで引き継ぐタグ
)
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

### 3.3 GitHub Actions OIDC の完全構成

```yaml
# .github/workflows/deploy.yml
name: Deploy to AWS
on:
  push:
    branches: [main]

permissions:
  id-token: write   # OIDC トークンの発行に必要
  contents: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/GitHubActionsDeployRole
          aws-region: ap-northeast-1
          role-session-name: github-actions-${{ github.run_id }}

      - name: Deploy
        run: |
          aws sts get-caller-identity
          aws s3 sync ./dist s3://my-app-bucket/
```

```bash
# OIDC プロバイダの作成
aws iam create-open-id-connect-provider \
  --url "https://token.actions.githubusercontent.com" \
  --client-id-list "sts.amazonaws.com" \
  --thumbprint-list "6938fd4d98bab03faadb97b34396831e3780aea1"

# GitHub Actions 用ロールの作成
aws iam create-role \
  --role-name GitHubActionsDeployRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
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
            "token.actions.githubusercontent.com:sub": "repo:my-org/my-repo:ref:refs/heads/main"
          }
        }
      }
    ]
  }'
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

# 5. Access Analyzer で生成されたポリシーの取得
aws accessanalyzer get-generated-policy \
  --job-id "policy-generation-job-id" \
  --include-resource-placeholders
```

### 4.1 Access Analyzer を活用した未使用権限の自動検出

```python
import boto3
import json
from datetime import datetime, timedelta

def audit_unused_permissions(days_threshold: int = 90) -> list[dict]:
    """未使用の IAM 権限を検出する"""
    iam = boto3.client("iam")
    results = []

    # 全ロールの一覧取得
    paginator = iam.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page["Roles"]:
            role_name = role["RoleName"]

            # AWS サービスロールはスキップ
            if role_name.startswith("aws-service-role/"):
                continue

            # サービス最終アクセス情報を取得
            job_id = iam.generate_service_last_accessed_details(
                Arn=role["Arn"]
            )["JobId"]

            # ジョブ完了を待機
            import time
            while True:
                result = iam.get_service_last_accessed_details(JobId=job_id)
                if result["JobStatus"] == "COMPLETED":
                    break
                time.sleep(1)

            threshold_date = datetime.now() - timedelta(days=days_threshold)

            for service in result["ServicesLastAccessed"]:
                last_accessed = service.get("LastAuthenticated")
                if last_accessed and last_accessed.replace(tzinfo=None) < threshold_date:
                    results.append({
                        "Role": role_name,
                        "Service": service["ServiceNamespace"],
                        "LastAccessed": str(last_accessed),
                        "DaysUnused": (datetime.now() - last_accessed.replace(tzinfo=None)).days,
                    })
                elif last_accessed is None and service.get("TotalAuthenticatedEntities", 0) == 0:
                    results.append({
                        "Role": role_name,
                        "Service": service["ServiceNamespace"],
                        "LastAccessed": "Never",
                        "DaysUnused": "N/A",
                    })

    return results

# 実行
unused = audit_unused_permissions(90)
for item in unused:
    print(f"Role: {item['Role']}, Service: {item['Service']}, "
          f"Last: {item['LastAccessed']}, Unused: {item['DaysUnused']} days")
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

### 5.1 Permission Boundary を使った権限委譲パターン

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowCreateRolesWithBoundary",
      "Effect": "Allow",
      "Action": [
        "iam:CreateRole",
        "iam:AttachRolePolicy",
        "iam:PutRolePolicy"
      ],
      "Resource": "arn:aws:iam::123456789012:role/app-*",
      "Condition": {
        "StringEquals": {
          "iam:PermissionsBoundary": "arn:aws:iam::123456789012:policy/DeveloperBoundary"
        }
      }
    },
    {
      "Sid": "DenyBoundaryModification",
      "Effect": "Deny",
      "Action": [
        "iam:DeleteRolePermissionsBoundary",
        "iam:PutRolePermissionsBoundary"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DenyBoundaryPolicyModification",
      "Effect": "Deny",
      "Action": [
        "iam:CreatePolicyVersion",
        "iam:DeletePolicy",
        "iam:DeletePolicyVersion",
        "iam:SetDefaultPolicyVersion"
      ],
      "Resource": "arn:aws:iam::123456789012:policy/DeveloperBoundary"
    }
  ]
}
```

---

## 6. IAM Identity Center (旧 AWS SSO)

### 6.1 Identity Center の全体像

```
┌──────────────────────────────────────────────────────────┐
│              IAM Identity Center                          │
│                                                          │
│  ┌──────────┐    SAML/SCIM    ┌──────────────────┐      │
│  │ 外部 IdP │ ←──────────── → │ Identity Center  │      │
│  │ (Okta,   │                 │ (AWS Organizations│      │
│  │  Azure AD,│                │  の管理アカウント) │      │
│  │  Google)  │                └────────┬─────────┘      │
│  └──────────┘                         │                  │
│                                       │                  │
│                    ┌──────────────────┼──────────────┐   │
│                    │                  │              │   │
│              ┌─────▼─────┐   ┌───────▼────┐  ┌─────▼──┐│
│              │ Account A │   │ Account B  │  │Account C││
│              │ (Dev)     │   │ (Staging)  │  │(Prod)   ││
│              │           │   │            │  │         ││
│              │ Permission│   │ Permission │  │Permission│
│              │ Set:      │   │ Set:       │  │Set:     ││
│              │ FullAccess│   │ ReadOnly   │  │Admin    ││
│              └───────────┘   └────────────┘  └─────────┘│
└──────────────────────────────────────────────────────────┘
```

### 6.2 Permission Set の作成

```bash
# Permission Set の作成
aws sso-admin create-permission-set \
  --instance-arn "arn:aws:sso:::instance/ssoins-xxxxxxxxxxxx" \
  --name "DeveloperAccess" \
  --description "Developer permissions for dev/staging accounts" \
  --session-duration "PT8H" \
  --relay-state "https://ap-northeast-1.console.aws.amazon.com/"

# AWS マネージドポリシーのアタッチ
aws sso-admin attach-managed-policy-to-permission-set \
  --instance-arn "arn:aws:sso:::instance/ssoins-xxxxxxxxxxxx" \
  --permission-set-arn "arn:aws:sso:::permissionSet/ssoins-xxxxxxxxxxxx/ps-xxxxxxxxxxxx" \
  --managed-policy-arn "arn:aws:iam::aws:policy/PowerUserAccess"

# カスタムインラインポリシーのアタッチ
aws sso-admin put-inline-policy-to-permission-set \
  --instance-arn "arn:aws:sso:::instance/ssoins-xxxxxxxxxxxx" \
  --permission-set-arn "arn:aws:sso:::permissionSet/ssoins-xxxxxxxxxxxx/ps-xxxxxxxxxxxx" \
  --inline-policy '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "DenyProductionResourceDeletion",
        "Effect": "Deny",
        "Action": [
          "rds:DeleteDBInstance",
          "rds:DeleteDBCluster",
          "ec2:TerminateInstances",
          "s3:DeleteBucket"
        ],
        "Resource": "*",
        "Condition": {
          "StringEquals": {
            "aws:ResourceTag/Environment": "production"
          }
        }
      }
    ]
  }'

# Permission Set をアカウントに割り当て
aws sso-admin create-account-assignment \
  --instance-arn "arn:aws:sso:::instance/ssoins-xxxxxxxxxxxx" \
  --target-id "111111111111" \
  --target-type AWS_ACCOUNT \
  --permission-set-arn "arn:aws:sso:::permissionSet/ssoins-xxxxxxxxxxxx/ps-xxxxxxxxxxxx" \
  --principal-type GROUP \
  --principal-id "group-id-from-identity-store"
```

---

## 7. Organizations と SCP (Service Control Policies)

### 7.1 SCP の設計パターン

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyRegionsOutsideAllowed",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": [
            "ap-northeast-1",
            "us-east-1"
          ]
        },
        "ArnNotLike": {
          "aws:PrincipalArn": [
            "arn:aws:iam::*:role/OrganizationAccountAccessRole"
          ]
        }
      }
    },
    {
      "Sid": "DenyRootAccountUsage",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringLike": {
          "aws:PrincipalArn": "arn:aws:iam::*:root"
        }
      }
    },
    {
      "Sid": "DenyCloudTrailModification",
      "Effect": "Deny",
      "Action": [
        "cloudtrail:StopLogging",
        "cloudtrail:DeleteTrail",
        "cloudtrail:UpdateTrail"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DenyGuardDutyDisable",
      "Effect": "Deny",
      "Action": [
        "guardduty:DisableOrganizationAdminAccount",
        "guardduty:DeleteDetector",
        "guardduty:DeleteMembers"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DenyLeavingOrganization",
      "Effect": "Deny",
      "Action": "organizations:LeaveOrganization",
      "Resource": "*"
    },
    {
      "Sid": "RequireS3Encryption",
      "Effect": "Deny",
      "Action": "s3:PutObject",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "s3:x-amz-server-side-encryption": ["aws:kms", "AES256"]
        },
        "Null": {
          "s3:x-amz-server-side-encryption": "false"
        }
      }
    }
  ]
}
```

### 7.2 Organizations の OU 構造とSCP適用例

```
Organizations Root
├── SCP: DenyRegionRestriction (全アカウントに適用)
│
├── OU: Security
│   ├── SCP: DenyAllExceptSecurityServices
│   ├── Security Tooling Account (SecurityHub, GuardDuty)
│   └── Log Archive Account (CloudTrail, Config)
│
├── OU: Infrastructure
│   ├── SCP: AllowNetworkServices
│   ├── Network Account (Transit Gateway, VPN)
│   └── Shared Services Account (Directory, DNS)
│
├── OU: Workloads
│   ├── OU: Production
│   │   ├── SCP: DenyDestructiveActions
│   │   ├── App-A Prod Account
│   │   └── App-B Prod Account
│   │
│   ├── OU: Staging
│   │   ├── App-A Staging Account
│   │   └── App-B Staging Account
│   │
│   └── OU: Development
│       ├── SCP: AllowBroadAccess (開発用に緩和)
│       ├── App-A Dev Account
│       └── App-B Dev Account
│
└── OU: Sandbox
    ├── SCP: DenyExpensiveServices + BudgetLimit
    └── Sandbox Account (個人実験用)
```

---

## 8. IAM の監視と監査

### 8.1 CloudTrail による IAM イベント監視

```python
import boto3
import json
from datetime import datetime, timedelta

def monitor_iam_events(hours: int = 24) -> list[dict]:
    """過去N時間のIAM関連イベントを監視"""
    ct = boto3.client("cloudtrail", region_name="ap-northeast-1")

    start_time = datetime.utcnow() - timedelta(hours=hours)
    end_time = datetime.utcnow()

    critical_events = [
        "CreateUser", "DeleteUser",
        "CreateRole", "DeleteRole",
        "AttachUserPolicy", "AttachRolePolicy",
        "PutUserPolicy", "PutRolePolicy",
        "CreateAccessKey",
        "UpdateAssumeRolePolicy",
        "CreateLoginProfile",
        "DeactivateMFADevice",
    ]

    results = []
    for event_name in critical_events:
        response = ct.lookup_events(
            LookupAttributes=[
                {"AttributeKey": "EventName", "AttributeValue": event_name}
            ],
            StartTime=start_time,
            EndTime=end_time,
            MaxResults=50,
        )

        for event in response.get("Events", []):
            detail = json.loads(event["CloudTrailEvent"])
            results.append({
                "EventTime": str(event["EventTime"]),
                "EventName": event["EventName"],
                "Username": event.get("Username", "N/A"),
                "SourceIP": detail.get("sourceIPAddress", "N/A"),
                "UserAgent": detail.get("userAgent", "N/A"),
                "ErrorCode": detail.get("errorCode", "None"),
                "Resources": [r.get("ARN", "") for r in event.get("Resources", [])],
            })

    return sorted(results, key=lambda x: x["EventTime"], reverse=True)
```

### 8.2 EventBridge + Lambda による自動アラート

```yaml
# CloudFormation: IAM 変更の自動検知
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  IAMChangeEventRule:
    Type: AWS::Events::Rule
    Properties:
      Name: iam-critical-change-alert
      Description: "Detect critical IAM changes"
      EventPattern:
        source:
          - "aws.iam"
        detail-type:
          - "AWS API Call via CloudTrail"
        detail:
          eventSource:
            - "iam.amazonaws.com"
          eventName:
            - "CreateUser"
            - "CreateAccessKey"
            - "AttachUserPolicy"
            - "AttachRolePolicy"
            - "PutUserPolicy"
            - "PutRolePolicy"
            - "DeleteRolePermissionsBoundary"
            - "UpdateAssumeRolePolicy"
            - "DeactivateMFADevice"
      State: ENABLED
      Targets:
        - Arn: !GetAtt AlertFunction.Arn
          Id: iam-alert-lambda
        - Arn: !Ref AlertTopic
          Id: iam-alert-sns

  AlertTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: iam-security-alerts
      Subscription:
        - Protocol: email
          Endpoint: security-team@example.com

  AlertFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: iam-change-alerter
      Runtime: python3.12
      Handler: index.handler
      Role: !GetAtt AlertFunctionRole.Arn
      Code:
        ZipFile: |
          import json
          import boto3
          import os

          def handler(event, context):
              sns = boto3.client("sns")
              detail = event["detail"]

              message = {
                  "Event": detail["eventName"],
                  "User": detail.get("userIdentity", {}).get("arn", "Unknown"),
                  "SourceIP": detail.get("sourceIPAddress", "Unknown"),
                  "Time": detail.get("eventTime", "Unknown"),
                  "Region": detail.get("awsRegion", "Unknown"),
                  "RequestParameters": detail.get("requestParameters", {}),
              }

              sns.publish(
                  TopicArn=os.environ["TOPIC_ARN"],
                  Subject=f"IAM Alert: {detail['eventName']}",
                  Message=json.dumps(message, indent=2, default=str)
              )
      Environment:
        Variables:
          TOPIC_ARN: !Ref AlertTopic
```

### 8.3 IAM Access Analyzer の外部アクセス検出

```bash
# Access Analyzer の作成（アカウントレベル）
aws accessanalyzer create-analyzer \
  --analyzer-name account-analyzer \
  --type ACCOUNT \
  --tags Environment=Production

# Access Analyzer の作成（組織レベル）
aws accessanalyzer create-analyzer \
  --analyzer-name org-analyzer \
  --type ORGANIZATION \
  --tags Environment=Production

# 検出結果の一覧取得
aws accessanalyzer list-findings \
  --analyzer-arn "arn:aws:access-analyzer:ap-northeast-1:123456789012:analyzer/account-analyzer" \
  --filter '{
    "status": {"eq": ["ACTIVE"]},
    "resourceType": {"eq": ["AWS::S3::Bucket", "AWS::IAM::Role"]}
  }'

# 検出結果の詳細
aws accessanalyzer get-finding \
  --analyzer-arn "arn:aws:access-analyzer:ap-northeast-1:123456789012:analyzer/account-analyzer" \
  --id "finding-id-xxxx"

# 未使用アクセスの検出（IAM Access Analyzer v2）
aws accessanalyzer create-analyzer \
  --analyzer-name unused-access-analyzer \
  --type ACCOUNT_UNUSED_ACCESS \
  --configuration '{
    "unusedAccess": {
      "unusedAccessAge": 90
    }
  }'
```

---

## 9. CDK による IAM の構成管理

### 9.1 CDK でのロール・ポリシー定義

```typescript
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export class IamStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Permission Boundary の定義
    const boundary = new iam.ManagedPolicy(this, 'DeveloperBoundary', {
      managedPolicyName: 'DeveloperBoundary',
      statements: [
        new iam.PolicyStatement({
          sid: 'AllowedServices',
          effect: iam.Effect.ALLOW,
          actions: [
            's3:*', 'dynamodb:*', 'lambda:*',
            'logs:*', 'cloudwatch:*', 'sqs:*',
            'sns:*', 'apigateway:*', 'xray:*',
          ],
          resources: ['*'],
        }),
        new iam.PolicyStatement({
          sid: 'DenyDangerousActions',
          effect: iam.Effect.DENY,
          actions: [
            'iam:CreateUser', 'iam:DeleteUser',
            'organizations:*', 'account:*',
          ],
          resources: ['*'],
        }),
      ],
    });

    // アプリケーションロールの定義
    const appRole = new iam.Role(this, 'AppRole', {
      roleName: 'MyAppRole',
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
        new iam.ServicePrincipal('lambda.amazonaws.com'),
      ),
      permissionsBoundary: boundary,
      maxSessionDuration: cdk.Duration.hours(4),
    });

    // 最小権限ポリシーの定義
    appRole.addToPolicy(new iam.PolicyStatement({
      sid: 'DynamoDBAccess',
      actions: [
        'dynamodb:GetItem', 'dynamodb:PutItem',
        'dynamodb:UpdateItem', 'dynamodb:Query',
      ],
      resources: [
        `arn:aws:dynamodb:${this.region}:${this.account}:table/Users`,
        `arn:aws:dynamodb:${this.region}:${this.account}:table/Users/index/*`,
      ],
    }));

    appRole.addToPolicy(new iam.PolicyStatement({
      sid: 'S3Access',
      actions: ['s3:GetObject', 's3:PutObject'],
      resources: ['arn:aws:s3:::my-app-uploads/*'],
      conditions: {
        'StringEquals': {
          's3:x-amz-server-side-encryption': 'aws:kms',
        },
      },
    }));

    // OIDC プロバイダと GitHub Actions ロール
    const githubOidc = new iam.OpenIdConnectProvider(this, 'GitHubOidc', {
      url: 'https://token.actions.githubusercontent.com',
      clientIds: ['sts.amazonaws.com'],
    });

    const githubRole = new iam.Role(this, 'GitHubDeployRole', {
      roleName: 'GitHubActionsDeployRole',
      assumedBy: new iam.OpenIdConnectPrincipal(githubOidc, {
        'StringEquals': {
          'token.actions.githubusercontent.com:aud': 'sts.amazonaws.com',
        },
        'StringLike': {
          'token.actions.githubusercontent.com:sub': 'repo:my-org/my-repo:*',
        },
      }),
      maxSessionDuration: cdk.Duration.hours(1),
    });

    // タグの付与
    cdk.Tags.of(appRole).add('Environment', 'Production');
    cdk.Tags.of(appRole).add('ManagedBy', 'CDK');
  }
}
```

---

## 10. 比較表

### 比較表 1: ポリシータイプ比較

| ポリシータイプ | 適用対象 | 管理者 | 用途 | JSON Principal |
|---------------|---------|--------|------|----------------|
| **Identity-based** | User/Group/Role | アカウント管理者 | 通常のアクセス制御 | 不要 |
| **Resource-based** | S3/SQS/Lambda 等 | リソース所有者 | クロスアカウント許可 | 必要 |
| **Permission Boundary** | User/Role | 管理者 | 委譲の上限設定 | 不要 |
| **SCP** | OU/Account | Org 管理者 | 組織レベルの制限 | 不要 |
| **Session Policy** | AssumeRole 時 | 呼び出し元 | 一時的な制限 | 不要 |
| **ACL** | S3/VPC | リソース所有者 | レガシー (非推奨) | 不要 |

### 比較表 2: 認証方式比較

| 方式 | 安全性 | 推奨度 | 用途 | キー管理 |
|------|--------|--------|------|---------|
| **IAM Role (EC2/Lambda)** | 高 | 最推奨 | AWS 内のサービス間 | 自動ローテーション |
| **OIDC Federation** | 高 | 推奨 | GitHub Actions, Google 等 | トークンベース |
| **SAML Federation** | 高 | 推奨 | 企業 SSO (Okta, Azure AD) | IdP 管理 |
| **IAM Identity Center** | 高 | 推奨 | マルチアカウント管理 | 自動管理 |
| **IAM User + MFA** | 中 | 条件付き | 管理者のコンソールアクセス | 手動ローテーション |
| **アクセスキー** | 低 | 非推奨 | レガシー/外部システム | 手動管理 |
| **ルートアカウント** | 最低 | 禁止 | 絶対に日常使用しない | ハードウェアMFA必須 |

### 比較表 3: RBAC vs ABAC 比較

| 項目 | RBAC | ABAC |
|------|------|------|
| **アクセス制御の単位** | ロール | タグ（属性） |
| **ポリシー数** | プロジェクト/チームごとに増加 | 少数のポリシーで対応 |
| **新リソースへの対応** | ポリシー更新が必要 | タグ付与で自動的に適用 |
| **スケーラビリティ** | ロール数に比例して複雑化 | タグで動的に制御 |
| **監査容易性** | ロール割り当てを確認 | タグとポリシーの組み合わせを確認 |
| **推奨場面** | 小〜中規模、明確な役割分担 | 大規模、動的なチーム構成 |
| **AWS 対応** | 従来型、全サービス対応 | ABAC 対応サービスが必要 |

---

## 11. アンチパターン

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

### アンチパターン 3: IAM ポリシーの直接アタッチ

```
[悪い例]
  IAM User に直接ポリシーをアタッチ
  → ユーザーが増えるたびに個別管理
  → 退職者のポリシー削除漏れ
  → 権限の一覧確認が困難

[良い例]
  IAM Group にポリシーをアタッチ → ユーザーをグループに所属
  または IAM Identity Center で Permission Set を管理

  # グループベースの管理
  aws iam create-group --group-name Backend-Developers
  aws iam attach-group-policy \
    --group-name Backend-Developers \
    --policy-arn arn:aws:iam::123456789012:policy/BackendDevAccess
  aws iam add-user-to-group \
    --group-name Backend-Developers \
    --user-name new-developer
```

### アンチパターン 4: MFA なしの特権アクセス

```
[悪い例]
  AdministratorAccess ポリシーを MFA なしのユーザーに付与
  → パスワード漏洩で全権限が奪取可能

[良い例]
  MFA を強制するポリシーを全ユーザーに適用:
  {
    "Sid": "DenyAllExceptMFASetup",
    "Effect": "Deny",
    "NotAction": [
      "iam:CreateVirtualMFADevice",
      "iam:EnableMFADevice",
      "iam:GetUser",
      "iam:ListMFADevices",
      "sts:GetSessionToken"
    ],
    "Resource": "*",
    "Condition": {
      "BoolIfExists": {
        "aws:MultiFactorAuthPresent": "false"
      }
    }
  }
```

### アンチパターン 5: SCP で Allow リストを使わない

```
[悪い例]
  SCP で Deny リストのみ作成
  → 新サービスが追加されるたびに Deny を追加する必要
  → 見落としが発生しやすい

[良い例]
  SCP で Allow リスト方式（ガードレール型）:
  - まず FullAWSAccess SCP でベースライン許可
  - その上で Deny ステートメントで制限
  - リージョン制限、危険操作の禁止を明示的に定義
```

---

## 12. 実践シナリオ: マルチアカウント IAM 設計

### 12.1 スタートアップの成長に合わせた IAM 設計

```
Phase 1: 単一アカウント（初期）
┌─────────────────────────────────────┐
│ Single Account                       │
│ ├── IAM User (MFA 必須)             │
│ ├── IAM Group: Admins               │
│ ├── IAM Group: Developers           │
│ └── IAM Role: Lambda/ECS            │
└─────────────────────────────────────┘

Phase 2: 2-3 アカウント（成長期）
┌──────────┐  ┌──────────┐  ┌──────────┐
│ Prod     │  │ Dev      │  │ Shared   │
│ Account  │  │ Account  │  │ Services │
│ (本番)   │  │ (開発)   │  │ (共有)   │
└──────────┘  └──────────┘  └──────────┘
  ↑ AssumeRole で分離

Phase 3: Organizations（拡大期）
┌─── Management Account ───────────────┐
│ Organizations, Billing, SSO           │
├─── Security OU ──────────────────────┤
│ SecurityHub, GuardDuty, Log Archive   │
├─── Workloads OU ─────────────────────┤
│ ├── Prod OU: App-A, App-B            │
│ ├── Staging OU: App-A, App-B         │
│ └── Dev OU: App-A, App-B             │
├─── Infrastructure OU ────────────────┤
│ Transit Gateway, Shared VPC          │
└───────────────────────────────────────┘
```

---

## 13. FAQ

### Q1: IAM ロールと IAM ユーザーのどちらを使うべきですか？

**A:** 原則として IAM ロールを使用する。AWS サービス（EC2, Lambda, ECS）には必ずロールをアタッチする。人間のオペレーターは IAM Identity Center (旧 SSO) でフェデレーション認証する。IAM ユーザーを作成するのは、外部システム連携で他の手段がない場合のみに限定し、MFA とアクセスキーローテーションを必須にする。

### Q2: クロスアカウントアクセスで ExternalId が必要な理由は？

**A:** ExternalId は「混乱した代理 (Confused Deputy)」攻撃を防止する。サードパーティサービスが顧客の AWS アカウントにアクセスする場合、ExternalId がないと、攻撃者が同じサードパーティサービスを使って他の顧客のロールを引き受けられる可能性がある。ExternalId は各顧客固有の値を使い、ロールの信頼ポリシーの Condition に設定する。

### Q3: 本番環境でルートアカウントを保護する方法は？

**A:** (1) ルートアカウントに強力なパスワードを設定、(2) ハードウェア MFA デバイスを有効化、(3) ルートアカウントのアクセスキーを削除、(4) ルートアカウントの使用は AWS Organizations の作成やアカウントの支払い設定変更など、IAM では実行できない操作のみに限定、(5) CloudTrail でルートアカウントの使用を監視し、アラートを設定する。

### Q4: Permission Boundary と SCP の違いは？

**A:** Permission Boundary は IAM エンティティ (User/Role) 単位で設定する権限の上限で、アカウント内の管理者が開発者への権限委譲に使う。SCP は Organizations の OU/Account 単位で設定するガードレールで、組織全体の統制に使う。SCP はルートユーザーにも適用されるが、Permission Boundary はルートユーザーには適用されない。

### Q5: ABAC を導入する際の注意点は？

**A:** (1) 全てのリソースに一貫したタグ付けが必須（タグ付けポリシーを SCP で強制）、(2) ABAC に対応していないサービスがあるため事前に確認、(3) タグの変更がアクセス権限に直結するため、タグ変更の権限を厳密に管理、(4) 既存の RBAC からの移行は段階的に実施し、並行運用期間を設ける。

### Q6: IAM ポリシーのサイズ制限にどう対処する？

**A:** マネージドポリシーは最大 6,144 文字（空白除く）。対処法: (1) ワイルドカードを活用（`s3:Get*` 等）、(2) リソース ARN でワイルドカードを使い一括指定、(3) 複数のマネージドポリシーに分割（最大10個/ロール）、(4) インラインポリシーを併用（別途 10,240 文字まで）、(5) Condition で動的制御し Action/Resource を減らす。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 最小権限 | 必要なアクション・リソースのみ許可。IAM Access Analyzer で検証 |
| 認証方式 | IAM ロール + OIDC フェデレーション。長期キー使用禁止 |
| ポリシー評価 | 明示的 Deny > SCP > Boundary > Allow の順で評価 |
| クロスアカウント | AssumeRole + ExternalId + 条件付き信頼ポリシー |
| Permission Boundary | 権限委譲時の安全ガード。開発者に自律性を与えつつ制限 |
| ABAC | タグベースの動的アクセス制御。大規模環境でスケーラブル |
| Identity Center | マルチアカウント環境の SSO。Permission Set で権限管理 |
| SCP | 組織レベルのガードレール。リージョン制限、危険操作の禁止 |
| 監視 | CloudTrail + IAM Access Analyzer + EventBridge で異常検知 |
| ルートアカウント | ハードウェア MFA + 使用禁止 + 監視 |
| IaC | CDK/CloudFormation で IAM をコード管理。レビュー可能に |

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
4. **AWS Organizations ユーザーガイド** — SCP の設計パターン
   https://docs.aws.amazon.com/organizations/latest/userguide/
5. **AWS IAM Identity Center ユーザーガイド** — SSO の設定と運用
   https://docs.aws.amazon.com/singlesignon/latest/userguide/
6. **AWS IAM Access Analyzer** — 外部アクセスと未使用権限の検出
   https://docs.aws.amazon.com/IAM/latest/UserGuide/what-is-access-analyzer.html
