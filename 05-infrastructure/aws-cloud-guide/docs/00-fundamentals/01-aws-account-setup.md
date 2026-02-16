# AWS アカウント設定

> AWS を安全かつ効率的に利用するための初期設定 — アカウント作成から IAM、MFA、Organizations、請求アラート、AWS Control Tower まで

## この章で学ぶこと

1. AWS アカウントを作成し、ルートユーザーのセキュリティを確保できる
2. IAM ユーザー・グループ・ポリシーを適切に設計し、最小権限の原則を適用できる
3. AWS Organizations と請求アラートを設定し、マルチアカウント運用とコスト管理を実現できる
4. IAM Identity Center (旧 SSO) を構築し、一元的なアクセス管理を導入できる
5. AWS Control Tower を活用して、ガバナンスの効いたランディングゾーンを構築できる

---

## 1. AWS アカウントの作成

### 1.1 アカウント作成フロー

```
+------------------+     +------------------+     +------------------+
| 1. サインアップ    | --> | 2. 連絡先情報     | --> | 3. 支払い情報     |
| メールアドレス     |     | 氏名/住所/電話    |     | クレジットカード   |
| パスワード設定     |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
         |                                                   |
         v                                                   v
+------------------+     +------------------+     +------------------+
| 6. 完了           | <-- | 5. サポートプラン  | <-- | 4. 本人確認       |
| コンソールログイン |     | Basic(無料)推奨   |     | SMS/音声認証      |
+------------------+     +------------------+     +------------------+
```

### 1.2 アカウント作成のベストプラクティス

```bash
# ルートユーザー用メールアドレスは専用のものを使う
# 例: aws-root@example.com（個人メールは避ける）

# アカウント作成後、最初にやるべきこと
# 1. ルートユーザーに MFA を設定
# 2. IAM 管理者ユーザーを作成
# 3. ルートユーザーのアクセスキーを作成しない（絶対に）
# 4. デフォルトリージョンを確認して東京リージョンに切り替え
# 5. 請求アラートを設定する
```

### 1.3 アカウント作成時のメールアドレス管理戦略

大規模組織では複数の AWS アカウントを運用するため、メールアドレスの管理が重要になる。

```
メールアドレス管理戦略
+----------------------------------------------------------+
|  パターン 1: メーリングリスト方式（推奨）                     |
|  aws-root-prod@example.com → チーム全員に配信             |
|  aws-root-staging@example.com → チーム全員に配信          |
|  aws-root-dev@example.com → チーム全員に配信              |
|                                                           |
|  パターン 2: Gmail エイリアス方式（小規模向け）              |
|  aws+prod@example.com                                     |
|  aws+staging@example.com                                  |
|  aws+dev@example.com                                      |
|                                                           |
|  パターン 3: 専用ドメイン方式（エンタープライズ）            |
|  root@prod.aws.example.com                                |
|  root@staging.aws.example.com                             |
|  root@dev.aws.example.com                                 |
+----------------------------------------------------------+
```

### 1.4 アカウント作成直後のセキュリティ設定スクリプト

```bash
#!/bin/bash
# AWS アカウント初期セキュリティ設定スクリプト
# 前提: IAM 管理者ユーザーの認証情報で実行

set -euo pipefail

ACCOUNT_ALIAS="my-company-prod"
ADMIN_USER="admin-user"
ADMIN_GROUP="Administrators"
REGION="ap-northeast-1"

echo "=== Step 1: アカウントエイリアスの設定 ==="
aws iam create-account-alias --account-alias "$ACCOUNT_ALIAS"
echo "アカウントエイリアス '$ACCOUNT_ALIAS' を設定しました"

echo "=== Step 2: パスワードポリシーの設定 ==="
aws iam update-account-password-policy \
  --minimum-password-length 14 \
  --require-symbols \
  --require-numbers \
  --require-uppercase-characters \
  --require-lowercase-characters \
  --allow-users-to-change-password \
  --max-password-age 90 \
  --password-reuse-prevention 12 \
  --hard-expiry
echo "パスワードポリシーを設定しました"

echo "=== Step 3: 管理者グループの作成 ==="
aws iam create-group --group-name "$ADMIN_GROUP"
aws iam attach-group-policy \
  --group-name "$ADMIN_GROUP" \
  --policy-arn arn:aws:iam::aws:policy/AdministratorAccess
echo "管理者グループ '$ADMIN_GROUP' を作成しました"

echo "=== Step 4: CloudTrail の有効化 ==="
TRAIL_BUCKET="cloudtrail-logs-$(aws sts get-caller-identity --query Account --output text)"
aws s3 mb "s3://$TRAIL_BUCKET" --region "$REGION" 2>/dev/null || true

# バケットポリシーを設定
cat > /tmp/trail-bucket-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AWSCloudTrailAclCheck",
      "Effect": "Allow",
      "Principal": {"Service": "cloudtrail.amazonaws.com"},
      "Action": "s3:GetBucketAcl",
      "Resource": "arn:aws:s3:::$TRAIL_BUCKET"
    },
    {
      "Sid": "AWSCloudTrailWrite",
      "Effect": "Allow",
      "Principal": {"Service": "cloudtrail.amazonaws.com"},
      "Action": "s3:PutObject",
      "Resource": "arn:aws:s3:::$TRAIL_BUCKET/AWSLogs/*",
      "Condition": {
        "StringEquals": {"s3:x-amz-acl": "bucket-owner-full-control"}
      }
    }
  ]
}
EOF

aws s3api put-bucket-policy \
  --bucket "$TRAIL_BUCKET" \
  --policy file:///tmp/trail-bucket-policy.json

aws cloudtrail create-trail \
  --name management-trail \
  --s3-bucket-name "$TRAIL_BUCKET" \
  --is-multi-region-trail \
  --enable-log-file-validation \
  --include-global-service-events

aws cloudtrail start-logging --name management-trail
echo "CloudTrail を有効化しました"

echo "=== Step 5: GuardDuty の有効化 ==="
aws guardduty create-detector \
  --enable \
  --finding-publishing-frequency FIFTEEN_MINUTES \
  --region "$REGION"
echo "GuardDuty を有効化しました"

echo "=== Step 6: EBS デフォルト暗号化の有効化 ==="
aws ec2 enable-ebs-encryption-by-default --region "$REGION"
echo "EBS デフォルト暗号化を有効化しました"

echo "=== Step 7: S3 パブリックアクセスブロック（アカウントレベル）==="
aws s3control put-public-access-block \
  --account-id "$(aws sts get-caller-identity --query Account --output text)" \
  --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
echo "S3 パブリックアクセスブロックを設定しました"

echo "=== 初期セキュリティ設定が完了しました ==="
```

---

## 2. ルートユーザーの保護

### 2.1 ルートユーザー vs IAM ユーザー

| 項目 | ルートユーザー | IAM ユーザー |
|------|---------------|-------------|
| 作成タイミング | アカウント作成時に自動生成 | 管理者が手動作成 |
| 権限 | 全権限（制限不可） | ポリシーで制御可能 |
| 用途 | アカウント設定のみ | 日常運用 |
| MFA | 必須 | 強く推奨 |
| アクセスキー | 作成禁止 | 必要に応じて作成 |
| SCP による制限 | 不可 | 可能 |
| 監査ログ | CloudTrail で記録 | CloudTrail で記録 |

### 2.2 ルートユーザーでしかできない操作

以下の操作はルートユーザーでのみ実行可能であり、IAM ユーザーには委任できない。

```
ルートユーザー専用タスク一覧
+-------------------------------------------------------------+
| 1. アカウント設定の変更                                        |
|    - アカウント名、メールアドレス、パスワードの変更              |
|    - 連絡先情報の変更                                         |
|                                                              |
| 2. 請求関連                                                   |
|    - 支払い方法の変更                                         |
|    - 請求情報への IAM アクセスの有効化/無効化                   |
|                                                              |
| 3. サポートプラン                                              |
|    - サポートプランの変更                                      |
|                                                              |
| 4. IAM 関連                                                   |
|    - 最初の IAM 管理者ユーザーの作成                           |
|    - アカウントの STS リージョン設定                            |
|                                                              |
| 5. サービス固有                                                |
|    - Route 53 ドメインの移管                                   |
|    - CloudFront キーペアの作成                                 |
|    - S3 バケットの MFA Delete 有効化                           |
|                                                              |
| 6. アカウントの閉鎖                                            |
|    - AWS アカウントの閉鎖（復元不可）                          |
+-------------------------------------------------------------+
```

### 2.3 MFA (多要素認証) の設定

```bash
# AWS CLI で仮想 MFA デバイスを作成
aws iam create-virtual-mfa-device \
  --virtual-mfa-device-name root-mfa \
  --outfile /tmp/QRCode.png \
  --bootstrap-method QRCodePNG

# MFA デバイスを有効化（TOTP コード2つが必要）
aws iam enable-mfa-device \
  --user-name root \
  --serial-number arn:aws:iam::123456789012:mfa/root-mfa \
  --authentication-code1 123456 \
  --authentication-code2 789012
```

### 2.4 MFA の種類比較

| MFA タイプ | セキュリティ | 利便性 | コスト | 推奨用途 |
|-----------|------------|--------|--------|---------|
| 仮想 MFA (TOTP) | 中 | 高 | 無料 | IAM ユーザー |
| FIDO2 セキュリティキー | 高 | 中 | 有料 | ルートユーザー |
| ハードウェア MFA | 最高 | 低 | 有料 | ルート/高権限 |
| パスキー | 高 | 高 | 無料 | IAM ユーザー（2024年以降） |

### 2.5 MFA 強制ポリシー

全 IAM ユーザーに MFA の使用を強制するためのポリシー例を示す。

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowViewAccountInfo",
      "Effect": "Allow",
      "Action": [
        "iam:GetAccountPasswordPolicy",
        "iam:ListVirtualMFADevices"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AllowManageOwnPasswords",
      "Effect": "Allow",
      "Action": [
        "iam:ChangePassword",
        "iam:GetUser"
      ],
      "Resource": "arn:aws:iam::*:user/${aws:username}"
    },
    {
      "Sid": "AllowManageOwnMFA",
      "Effect": "Allow",
      "Action": [
        "iam:CreateVirtualMFADevice",
        "iam:DeleteVirtualMFADevice",
        "iam:EnableMFADevice",
        "iam:ListMFADevices",
        "iam:ResyncMFADevice"
      ],
      "Resource": [
        "arn:aws:iam::*:mfa/${aws:username}",
        "arn:aws:iam::*:user/${aws:username}"
      ]
    },
    {
      "Sid": "DenyAllExceptListedIfNoMFA",
      "Effect": "Deny",
      "NotAction": [
        "iam:CreateVirtualMFADevice",
        "iam:EnableMFADevice",
        "iam:GetUser",
        "iam:ChangePassword",
        "iam:ListMFADevices",
        "iam:ListVirtualMFADevices",
        "iam:ResyncMFADevice",
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

### 2.6 ルートユーザーの緊急アクセス手順

ルートユーザーの認証情報は「金庫に保管」が原則だが、緊急時のアクセス手順を事前に文書化しておくべきである。

```
ルートユーザー緊急アクセス手順書（テンプレート）
+-------------------------------------------------------------+
| 1. 準備事項                                                   |
|    - ルートユーザーのメールアドレス: aws-root@example.com       |
|    - MFA デバイスの保管場所: 金庫 A（経理部フロア）             |
|    - バックアップ MFA の保管場所: 金庫 B（IT部門フロア）        |
|                                                              |
| 2. アクセス手順                                                |
|    a. 承認者2名以上の承認を取得（メール証跡を残す）             |
|    b. 金庫から MFA デバイスを取り出す                          |
|    c. AWS コンソールにルートユーザーでログイン                  |
|    d. 必要な操作を実施（CloudTrail で記録される）               |
|    e. 操作完了後、即座にログアウト                              |
|    f. MFA デバイスを金庫に戻す                                 |
|    g. 作業内容を記録し、チームに共有                           |
|                                                              |
| 3. 禁止事項                                                   |
|    - ルートユーザーのアクセスキーを作成しない                   |
|    - パスワードを変更しない（緊急時を除く）                    |
|    - 不要な操作を行わない                                     |
+-------------------------------------------------------------+
```

---

## 3. IAM の設計

### 3.1 IAM コンポーネント

```
AWS IAM アーキテクチャ
+------------------------------------------------------+
|  AWS Account                                          |
|                                                       |
|  +----------+    所属    +----------+                 |
|  | IAM User | --------> | IAM Group|                 |
|  +----------+           +----------+                  |
|       |                      |                        |
|       | (直接 or グループ経由)  |                      |
|       v                      v                        |
|  +-------------------------------------------+        |
|  |          IAM Policy (JSON)                |        |
|  | {                                         |        |
|  |   "Effect": "Allow",                     |        |
|  |   "Action": "s3:GetObject",              |        |
|  |   "Resource": "arn:aws:s3:::bucket/*"    |        |
|  | }                                         |        |
|  +-------------------------------------------+        |
|                                                       |
|  +----------+                                         |
|  | IAM Role | <-- EC2, Lambda などが引き受ける         |
|  +----------+                                         |
+------------------------------------------------------+
```

### 3.2 IAM ポリシーの種類

| ポリシー種類 | 説明 | 管理主体 | 用途 |
|------------|------|---------|------|
| AWS 管理ポリシー | AWS が提供する定義済みポリシー | AWS | 一般的な権限パターン |
| カスタマー管理ポリシー | ユーザーが作成するポリシー | ユーザー | 組織固有の要件 |
| インラインポリシー | エンティティに直接埋め込み | ユーザー | 1:1の権限（非推奨） |
| サービスコントロールポリシー (SCP) | Organizations で使用 | 管理者 | アカウント全体のガードレール |
| アクセス許可境界 | IAM エンティティの権限上限 | 管理者 | 権限委任の安全性確保 |
| セッションポリシー | AssumeRole 時に指定 | 呼び出し元 | 一時的な権限制限 |
| リソースベースポリシー | リソースに直接アタッチ | リソース所有者 | クロスアカウントアクセス |

### 3.3 コード例: IAM ユーザーとグループの作成

```bash
# 開発者グループを作成
aws iam create-group --group-name Developers

# IAM ポリシーをアタッチ
aws iam attach-group-policy \
  --group-name Developers \
  --policy-arn arn:aws:iam::aws:policy/PowerUserAccess

# IAM ユーザーを作成
aws iam create-user --user-name tanaka

# ユーザーをグループに追加
aws iam add-user-to-group \
  --user-name tanaka \
  --group-name Developers

# ログインプロファイル（パスワード）を作成
aws iam create-login-profile \
  --user-name tanaka \
  --password 'TempP@ssw0rd!' \
  --password-reset-required

# タグを付与（部署、チーム情報）
aws iam tag-user \
  --user-name tanaka \
  --tags \
    Key=Department,Value=Engineering \
    Key=Team,Value=Backend \
    Key=CostCenter,Value=CC-001
```

### 3.4 コード例: カスタム IAM ポリシー (JSON)

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
        "arn:aws:s3:::my-app-bucket",
        "arn:aws:s3:::my-app-bucket/*"
      ]
    },
    {
      "Sid": "DenyDeleteBucket",
      "Effect": "Deny",
      "Action": "s3:DeleteBucket",
      "Resource": "*"
    }
  ]
}
```

### 3.5 コード例: 条件付きポリシー（高度な制御）

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowEC2InTokyoRegionOnly",
      "Effect": "Allow",
      "Action": "ec2:*",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:RequestedRegion": "ap-northeast-1"
        }
      }
    },
    {
      "Sid": "DenyEC2TerminateWithoutMFA",
      "Effect": "Deny",
      "Action": "ec2:TerminateInstances",
      "Resource": "*",
      "Condition": {
        "BoolIfExists": {
          "aws:MultiFactorAuthPresent": "false"
        }
      }
    },
    {
      "Sid": "AllowActionsOnlyDuringBusinessHours",
      "Effect": "Deny",
      "Action": [
        "rds:DeleteDBInstance",
        "rds:DeleteDBCluster"
      ],
      "Resource": "*",
      "Condition": {
        "DateGreaterThan": {
          "aws:CurrentTime": "2025-01-01T18:00:00Z"
        },
        "DateLessThan": {
          "aws:CurrentTime": "2025-01-02T09:00:00Z"
        }
      }
    },
    {
      "Sid": "RestrictBySourceIP",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "NotIpAddress": {
          "aws:SourceIp": [
            "203.0.113.0/24",
            "198.51.100.0/24"
          ]
        },
        "Bool": {
          "aws:ViaAWSService": "false"
        }
      }
    }
  ]
}
```

### 3.6 コード例: IAM ロールの作成 (EC2 用)

```bash
# 信頼ポリシーを作成
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# ロールを作成
aws iam create-role \
  --role-name EC2-S3-ReadOnly \
  --assume-role-policy-document file://trust-policy.json

# ポリシーをアタッチ
aws iam attach-role-policy \
  --role-name EC2-S3-ReadOnly \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# インスタンスプロファイルを作成してロールを関連付け
aws iam create-instance-profile \
  --instance-profile-name EC2-S3-ReadOnly-Profile
aws iam add-role-to-instance-profile \
  --instance-profile-name EC2-S3-ReadOnly-Profile \
  --role-name EC2-S3-ReadOnly
```

### 3.7 コード例: クロスアカウントロールの作成

```bash
# アカウント B（対象）にロールを作成
# アカウント A（呼び出し元）からのアクセスを許可する信頼ポリシー
cat > cross-account-trust.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::111111111111:root"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "sts:ExternalId": "my-external-id-12345"
        }
      }
    }
  ]
}
EOF

aws iam create-role \
  --role-name CrossAccountReadOnly \
  --assume-role-policy-document file://cross-account-trust.json

aws iam attach-role-policy \
  --role-name CrossAccountReadOnly \
  --policy-arn arn:aws:iam::aws:policy/ReadOnlyAccess

# アカウント A（呼び出し元）からロールを引き受ける
aws sts assume-role \
  --role-arn arn:aws:iam::222222222222:role/CrossAccountReadOnly \
  --role-session-name cross-account-session \
  --external-id my-external-id-12345
```

### 3.8 アクセス許可境界（Permissions Boundary）

アクセス許可境界は、IAM エンティティが持てる最大権限を制限する仕組みである。権限委任を安全に行うために重要。

```json
{
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
        "sns:*"
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
}
```

```bash
# アクセス許可境界をユーザーに設定
aws iam put-user-permissions-boundary \
  --user-name developer-tanaka \
  --permissions-boundary arn:aws:iam::123456789012:policy/DeveloperBoundary

# アクセス許可境界をロールに設定
aws iam put-role-permissions-boundary \
  --role-name LambdaExecutionRole \
  --permissions-boundary arn:aws:iam::123456789012:policy/LambdaBoundary
```

### 3.9 最小権限の原則

```
権限設計のアプローチ
+------------------------------------------+
|                                          |
|  1. 必要最小限の権限から開始              |
|     ↓                                    |
|  2. IAM Access Analyzer で不足を検出     |
|     ↓                                    |
|  3. 必要な権限だけを追加                  |
|     ↓                                    |
|  4. 定期的に未使用権限を棚卸し            |
|     ↓                                    |
|  5. 不要な権限を削除                      |
|                                          |
|  ※ "AdministratorAccess" を安易に付与しない |
+------------------------------------------+
```

### 3.10 IAM Access Analyzer の活用

```bash
# Access Analyzer を作成（アカウントレベル）
aws accessanalyzer create-analyzer \
  --analyzer-name account-analyzer \
  --type ACCOUNT

# 分析結果（外部からアクセス可能なリソース）を確認
aws accessanalyzer list-findings \
  --analyzer-arn arn:aws:access-analyzer:ap-northeast-1:123456789012:analyzer/account-analyzer \
  --query 'findings[].{Resource:resource,ResourceType:resourceType,Status:status}' \
  --output table

# ポリシー生成（CloudTrail ログから最小権限ポリシーを生成）
aws accessanalyzer start-policy-generation \
  --policy-generation-details '{
    "principalArn": "arn:aws:iam::123456789012:role/MyAppRole",
    "cloudTrailDetails": {
      "trails": [
        {
          "cloudTrailArn": "arn:aws:cloudtrail:ap-northeast-1:123456789012:trail/management-trail",
          "regions": ["ap-northeast-1"],
          "allRegions": false
        }
      ],
      "accessRole": "arn:aws:iam::123456789012:role/AccessAnalyzerRole",
      "startTime": "2025-01-01T00:00:00Z",
      "endTime": "2025-02-01T00:00:00Z"
    }
  }'

# 未使用のアクセスキー・パスワードを検出
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

## 4. IAM Identity Center (旧 AWS SSO)

### 4.1 IAM Identity Center の概要

```
IAM Identity Center アーキテクチャ
+----------------------------------------------------------+
|  AWS Organizations (Management Account)                    |
|                                                           |
|  +----------------------------------------------------+  |
|  |  IAM Identity Center                                |  |
|  |                                                     |  |
|  |  +----------------+    +------------------------+   |  |
|  |  | ID ソース       |    | 権限セット              |  |  |
|  |  | - Identity Center|   | - AdministratorAccess  |  |  |
|  |  | - Active Dir.   |    | - PowerUserAccess      |  |  |
|  |  | - 外部 IdP      |    | - ViewOnlyAccess       |  |  |
|  |  +----------------+    | - カスタム権限セット     |  |  |
|  |                        +------------------------+   |  |
|  |  ユーザー/グループ ← 権限セット → AWSアカウント       |  |
|  +----------------------------------------------------+  |
|                                                           |
|  +-------------------+  +-------------------+             |
|  | Production Account|  | Development Account|            |
|  +-------------------+  +-------------------+             |
+----------------------------------------------------------+
```

### 4.2 IAM Identity Center のセットアップ

```bash
# IAM Identity Center のインスタンスを作成
aws sso-admin create-instance --name "my-org-sso"

# 権限セットを作成
aws sso-admin create-permission-set \
  --instance-arn arn:aws:sso:::instance/ssoins-xxxx \
  --name "DeveloperAccess" \
  --description "Developer team access" \
  --session-duration "PT8H" \
  --relay-state ""

# 管理ポリシーをアタッチ
aws sso-admin attach-managed-policy-to-permission-set \
  --instance-arn arn:aws:sso:::instance/ssoins-xxxx \
  --permission-set-arn arn:aws:sso:::permissionSet/ssoins-xxxx/ps-xxxx \
  --managed-policy-arn arn:aws:iam::aws:policy/PowerUserAccess

# カスタムインラインポリシーをアタッチ
aws sso-admin put-inline-policy-to-permission-set \
  --instance-arn arn:aws:sso:::instance/ssoins-xxxx \
  --permission-set-arn arn:aws:sso:::permissionSet/ssoins-xxxx/ps-xxxx \
  --inline-policy '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Deny",
        "Action": [
          "iam:CreateUser",
          "iam:DeleteUser",
          "organizations:*"
        ],
        "Resource": "*"
      }
    ]
  }'

# アカウントに権限セットを割り当て
aws sso-admin create-account-assignment \
  --instance-arn arn:aws:sso:::instance/ssoins-xxxx \
  --target-id 123456789012 \
  --target-type AWS_ACCOUNT \
  --permission-set-arn arn:aws:sso:::permissionSet/ssoins-xxxx/ps-xxxx \
  --principal-type GROUP \
  --principal-id "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```

### 4.3 外部 IdP との連携（SAML 2.0）

```
外部 IdP 連携フロー
+----------------------------------------------------------+
|                                                           |
|  1. ユーザーが AWS アクセスポータルにアクセス               |
|     ↓                                                     |
|  2. IAM Identity Center → 外部 IdP にリダイレクト          |
|     ↓                                                     |
|  3. 外部 IdP で認証（MFA含む）                             |
|     ↓                                                     |
|  4. SAML アサーションを IAM Identity Center に返却          |
|     ↓                                                     |
|  5. IAM Identity Center が一時認証情報を発行                |
|     ↓                                                     |
|  6. ユーザーが AWS アカウント/ロールを選択                   |
|                                                           |
|  対応 IdP:                                                 |
|  - Azure AD (Entra ID)                                    |
|  - Okta                                                   |
|  - Google Workspace                                       |
|  - OneLogin                                               |
|  - Ping Identity                                          |
+----------------------------------------------------------+
```

---

## 5. AWS Organizations

### 5.1 マルチアカウント戦略

```
AWS Organizations 構成例
+----------------------------------------------------+
| Management Account (請求統合・ガバナンス)             |
|                                                     |
| ├── OU: Security                                    |
| │   ├── Log Archive Account (CloudTrail, Config)    |
| │   └── Security Tooling Account (GuardDuty, etc.)  |
| │                                                   |
| ├── OU: Infrastructure                              |
| │   ├── Network Account (Transit Gateway, VPN)      |
| │   └── Shared Services Account (CI/CD, ECR)        |
| │                                                   |
| ├── OU: Workloads                                   |
| │   ├── Production Account                          |
| │   ├── Staging Account                             |
| │   └── Development Account                         |
| │                                                   |
| └── OU: Sandbox                                     |
|     └── Developer Sandbox Account                   |
+----------------------------------------------------+
```

### 5.2 コード例: Organizations の操作

```bash
# 組織を作成
aws organizations create-organization --feature-set ALL

# OU (組織単位) を作成
ROOT_ID=$(aws organizations list-roots --query 'Roots[0].Id' --output text)

aws organizations create-organizational-unit \
  --parent-id "$ROOT_ID" \
  --name "Security"

aws organizations create-organizational-unit \
  --parent-id "$ROOT_ID" \
  --name "Infrastructure"

aws organizations create-organizational-unit \
  --parent-id "$ROOT_ID" \
  --name "Workloads"

aws organizations create-organizational-unit \
  --parent-id "$ROOT_ID" \
  --name "Sandbox"

# 新しいメンバーアカウントを作成
aws organizations create-account \
  --email prod@example.com \
  --account-name "Production"

# SCP（サービスコントロールポリシー）をアタッチ
aws organizations attach-policy \
  --policy-id p-xxxx \
  --target-id ou-xxxx

# OU の一覧を確認
aws organizations list-organizational-units-for-parent \
  --parent-id "$ROOT_ID" \
  --query 'OrganizationalUnits[].[Id,Name]' \
  --output table

# アカウント一覧
aws organizations list-accounts \
  --query 'Accounts[].[Id,Name,Email,Status]' \
  --output table
```

### 5.3 Service Control Policy (SCP) 例

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyLeaveOrganization",
      "Effect": "Deny",
      "Action": "organizations:LeaveOrganization",
      "Resource": "*"
    },
    {
      "Sid": "RestrictRegions",
      "Effect": "Deny",
      "NotAction": [
        "iam:*",
        "sts:*",
        "organizations:*",
        "support:*",
        "budgets:*",
        "health:*",
        "ce:*"
      ],
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": [
            "ap-northeast-1",
            "us-east-1"
          ]
        }
      }
    }
  ]
}
```

### 5.4 SCP のベストプラクティス集

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyRootUserActions",
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
      "Sid": "ProtectCloudTrail",
      "Effect": "Deny",
      "Action": [
        "cloudtrail:StopLogging",
        "cloudtrail:DeleteTrail",
        "cloudtrail:UpdateTrail"
      ],
      "Resource": "*"
    },
    {
      "Sid": "ProtectGuardDuty",
      "Effect": "Deny",
      "Action": [
        "guardduty:DeleteDetector",
        "guardduty:DisassociateFromMasterAccount",
        "guardduty:UpdateDetector"
      ],
      "Resource": "*"
    },
    {
      "Sid": "DenyPublicS3",
      "Effect": "Deny",
      "Action": "s3:PutBucketPublicAccessBlock",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "s3:publicAccessBlockConfiguration/BlockPublicAcls": "true"
        }
      }
    },
    {
      "Sid": "RequireEncryptedVolumes",
      "Effect": "Deny",
      "Action": "ec2:CreateVolume",
      "Resource": "*",
      "Condition": {
        "Bool": {
          "ec2:Encrypted": "false"
        }
      }
    },
    {
      "Sid": "DenyLargeInstances",
      "Effect": "Deny",
      "Action": "ec2:RunInstances",
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "ForAnyValue:StringLike": {
          "ec2:InstanceType": [
            "*.8xlarge",
            "*.12xlarge",
            "*.16xlarge",
            "*.24xlarge",
            "*.metal",
            "p4*",
            "p5*"
          ]
        }
      }
    }
  ]
}
```

---

## 6. AWS Control Tower

### 6.1 Control Tower の概要

AWS Control Tower は、AWS のベストプラクティスに基づいたマルチアカウント環境（ランディングゾーン）を自動構築するサービスである。

```
Control Tower ランディングゾーン
+----------------------------------------------------------+
|  Management Account                                       |
|  ├── AWS Control Tower                                    |
|  ├── AWS Organizations                                    |
|  ├── AWS Service Catalog                                  |
|  └── AWS CloudFormation StackSets                         |
|                                                           |
|  OU: Security                                             |
|  ├── Log Archive Account                                  |
|  │   ├── CloudTrail ログ (全アカウント)                    |
|  │   ├── AWS Config ログ (全アカウント)                    |
|  │   └── VPC フローログ                                   |
|  └── Audit Account                                        |
|      ├── SNS 通知                                         |
|      ├── AWS Config アグリゲーター                         |
|      └── Security Hub                                     |
|                                                           |
|  OU: Sandbox (カスタム OU)                                 |
|  └── Developer Sandbox Accounts                           |
|                                                           |
|  OU: Workloads (カスタム OU)                               |
|  ├── Production Accounts                                  |
|  └── Development Accounts                                 |
+----------------------------------------------------------+
```

### 6.2 ガードレール（Controls）

```
ガードレールの分類
+----------------------------------------------------------+
|  強制（Preventive） - SCP で禁止行為をブロック               |
|  ├── CloudTrail の無効化を禁止                              |
|  ├── S3 バケットのパブリックアクセスを禁止                   |
|  ├── ルートユーザーのアクセスキー作成を禁止                  |
|  └── リージョン制限                                        |
|                                                           |
|  検出（Detective） - AWS Config Rules で違反を検出          |
|  ├── MFA が未設定のユーザーを検出                           |
|  ├── 暗号化されていない EBS ボリュームを検出                 |
|  ├── パブリック IP が付与された EC2 を検出                   |
|  └── 未使用のアクセスキーを検出                             |
|                                                           |
|  予防（Proactive） - CloudFormation フックで事前チェック     |
|  ├── EC2 に IMDSv2 が必須か検証                            |
|  ├── RDS が暗号化されているか検証                           |
|  └── Lambda が VPC 内にあるか検証                           |
+----------------------------------------------------------+
```

---

## 7. 請求アラートとコスト管理

### 7.1 コード例: 請求アラートの設定 (CloudWatch)

```bash
# 請求メトリクスを有効化（コンソールで先に有効化が必要）
# Billing > Billing Preferences > Receive Billing Alerts

# SNS トピックを作成
aws sns create-topic --name billing-alerts
aws sns subscribe \
  --topic-arn arn:aws:sns:us-east-1:123456789012:billing-alerts \
  --protocol email \
  --notification-endpoint admin@example.com

# CloudWatch 請求アラームを作成（月額 $50 超過で通知）
aws cloudwatch put-metric-alarm \
  --alarm-name "MonthlyBillingAlarm-50USD" \
  --metric-name EstimatedCharges \
  --namespace AWS/Billing \
  --statistic Maximum \
  --period 21600 \
  --threshold 50 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts \
  --dimensions Name=Currency,Value=USD \
  --region us-east-1

# 段階的なアラーム（$100, $200, $500）
for threshold in 100 200 500; do
  aws cloudwatch put-metric-alarm \
    --alarm-name "MonthlyBillingAlarm-${threshold}USD" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 21600 \
    --threshold "$threshold" \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-east-1:123456789012:billing-alerts \
    --dimensions Name=Currency,Value=USD \
    --region us-east-1
done
```

### 7.2 AWS Budgets の設定

```bash
# 月間予算を作成（$100 の予算、80% で通知）
aws budgets create-budget \
  --account-id 123456789012 \
  --budget '{
    "BudgetName": "MonthlyBudget",
    "BudgetLimit": {"Amount": "100", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }' \
  --notifications-with-subscribers '[
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 80,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [
        {
          "SubscriptionType": "EMAIL",
          "Address": "admin@example.com"
        }
      ]
    },
    {
      "Notification": {
        "NotificationType": "FORECASTED",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 100,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [
        {
          "SubscriptionType": "EMAIL",
          "Address": "admin@example.com"
        }
      ]
    }
  ]'

# サービス別の予算を作成
aws budgets create-budget \
  --account-id 123456789012 \
  --budget '{
    "BudgetName": "EC2-MonthlyBudget",
    "BudgetLimit": {"Amount": "50", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {
      "Service": ["Amazon Elastic Compute Cloud - Compute"]
    }
  }' \
  --notifications-with-subscribers '[
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 80,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [
        {
          "SubscriptionType": "EMAIL",
          "Address": "admin@example.com"
        }
      ]
    }
  ]'
```

### 7.3 AWS Cost Explorer の活用

```bash
# 過去30日のサービス別コスト
aws ce get-cost-and-usage \
  --time-period Start=$(date -v-30d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[].Groups[?Metrics.BlendedCost.Amount > `1.0`].{Service:Keys[0],Cost:Metrics.BlendedCost.Amount}' \
  --output table

# 日別のコスト推移
aws ce get-cost-and-usage \
  --time-period Start=$(date -v-7d +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --query 'ResultsByTime[].{Date:TimePeriod.Start,Cost:Total.BlendedCost.Amount}' \
  --output table

# コスト予測（今月末の予測値）
aws ce get-cost-forecast \
  --time-period Start=$(date +%Y-%m-%d),End=$(date -v+1m -v1d -v-1d +%Y-%m-%d) \
  --metric BLENDED_COST \
  --granularity MONTHLY
```

### 7.4 コスト異常検知

```bash
# コスト異常検知モニターを作成
aws ce create-anomaly-monitor \
  --anomaly-monitor '{
    "MonitorName": "ServiceMonitor",
    "MonitorType": "DIMENSIONAL",
    "MonitorDimension": "SERVICE"
  }'

# サブスクリプション（通知先）を作成
aws ce create-anomaly-subscription \
  --anomaly-subscription '{
    "SubscriptionName": "CostAnomalyAlerts",
    "MonitorArnList": ["arn:aws:ce::123456789012:anomalymonitor/xxxx"],
    "Subscribers": [
      {
        "Address": "admin@example.com",
        "Type": "EMAIL"
      }
    ],
    "Threshold": 10.0,
    "Frequency": "DAILY"
  }'
```

---

## 8. CloudFormation による IAM リソースの IaC 管理

### 8.1 IAM ユーザー・グループ・ポリシーの CloudFormation テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'IAM初期設定 - ユーザー、グループ、ポリシーの作成'

Parameters:
  EnvironmentName:
    Type: String
    Default: production
    AllowedValues: [production, staging, development]

Resources:
  # パスワードポリシー
  # CloudFormation では直接設定できないため、カスタムリソースが必要

  # 管理者グループ
  AdminGroup:
    Type: AWS::IAM::Group
    Properties:
      GroupName: !Sub '${EnvironmentName}-administrators'
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AdministratorAccess

  # 開発者グループ
  DeveloperGroup:
    Type: AWS::IAM::Group
    Properties:
      GroupName: !Sub '${EnvironmentName}-developers'
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/PowerUserAccess
      Policies:
        - PolicyName: DenyIAMAndOrganizations
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Deny
                Action:
                  - 'iam:CreateUser'
                  - 'iam:DeleteUser'
                  - 'iam:CreateRole'
                  - 'iam:DeleteRole'
                  - 'organizations:*'
                Resource: '*'

  # 読み取り専用グループ
  ReadOnlyGroup:
    Type: AWS::IAM::Group
    Properties:
      GroupName: !Sub '${EnvironmentName}-readonly'
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/ReadOnlyAccess

  # MFA 強制ポリシー
  MFAEnforcementPolicy:
    Type: AWS::IAM::ManagedPolicy
    Properties:
      ManagedPolicyName: !Sub '${EnvironmentName}-mfa-enforcement'
      Groups:
        - !Ref DeveloperGroup
        - !Ref ReadOnlyGroup
      PolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Sid: AllowManageOwnMFA
            Effect: Allow
            Action:
              - 'iam:CreateVirtualMFADevice'
              - 'iam:EnableMFADevice'
              - 'iam:ListMFADevices'
              - 'iam:ResyncMFADevice'
            Resource:
              - !Sub 'arn:aws:iam::${AWS::AccountId}:mfa/${!aws:username}'
              - !Sub 'arn:aws:iam::${AWS::AccountId}:user/${!aws:username}'
          - Sid: DenyWithoutMFA
            Effect: Deny
            NotAction:
              - 'iam:CreateVirtualMFADevice'
              - 'iam:EnableMFADevice'
              - 'iam:ListMFADevices'
              - 'iam:GetUser'
              - 'iam:ChangePassword'
              - 'sts:GetSessionToken'
            Resource: '*'
            Condition:
              BoolIfExists:
                'aws:MultiFactorAuthPresent': 'false'

  # EC2 用ロール
  EC2WebServerRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${EnvironmentName}-ec2-webserver'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy

  EC2WebServerInstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      InstanceProfileName: !Sub '${EnvironmentName}-ec2-webserver-profile'
      Roles:
        - !Ref EC2WebServerRole

Outputs:
  AdminGroupArn:
    Value: !GetAtt AdminGroup.Arn
    Export:
      Name: !Sub '${EnvironmentName}-admin-group-arn'
  DeveloperGroupArn:
    Value: !GetAtt DeveloperGroup.Arn
    Export:
      Name: !Sub '${EnvironmentName}-developer-group-arn'
  EC2WebServerRoleArn:
    Value: !GetAtt EC2WebServerRole.Arn
    Export:
      Name: !Sub '${EnvironmentName}-ec2-webserver-role-arn'
```

### 8.2 AWS CDK による IAM 設定

```typescript
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export class IamSetupStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // 開発者グループ
    const developerGroup = new iam.Group(this, 'DeveloperGroup', {
      groupName: 'developers',
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('PowerUserAccess'),
      ],
    });

    // カスタムポリシー
    const restrictedPolicy = new iam.ManagedPolicy(this, 'RestrictedPolicy', {
      managedPolicyName: 'developer-restrictions',
      statements: [
        new iam.PolicyStatement({
          effect: iam.Effect.DENY,
          actions: [
            'iam:CreateUser',
            'iam:DeleteUser',
            'organizations:*',
          ],
          resources: ['*'],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.DENY,
          actions: ['ec2:*'],
          resources: ['*'],
          conditions: {
            StringNotEquals: {
              'aws:RequestedRegion': ['ap-northeast-1', 'us-east-1'],
            },
          },
        }),
      ],
    });
    developerGroup.addManagedPolicy(restrictedPolicy);

    // EC2 用ロール
    const ec2Role = new iam.Role(this, 'EC2WebServerRole', {
      roleName: 'ec2-webserver-role',
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonS3ReadOnlyAccess'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchAgentServerPolicy'),
      ],
      maxSessionDuration: cdk.Duration.hours(4),
    });

    // Lambda 実行ロール
    const lambdaRole = new iam.Role(this, 'LambdaExecutionRole', {
      roleName: 'lambda-execution-role',
      assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          'service-role/AWSLambdaBasicExecutionRole'
        ),
      ],
      inlinePolicies: {
        dynamoAccess: new iam.PolicyDocument({
          statements: [
            new iam.PolicyStatement({
              actions: [
                'dynamodb:GetItem',
                'dynamodb:PutItem',
                'dynamodb:Query',
              ],
              resources: [
                `arn:aws:dynamodb:${this.region}:${this.account}:table/MyTable`,
              ],
            }),
          ],
        }),
      },
    });

    // 出力
    new cdk.CfnOutput(this, 'EC2RoleArn', {
      value: ec2Role.roleArn,
      exportName: 'ec2-webserver-role-arn',
    });
  }
}
```

---

## 9. 初期設定チェックリスト

| # | タスク | 優先度 | カテゴリ | 完了 |
|---|--------|--------|---------|------|
| 1 | ルートユーザーに MFA を設定 | 必須 | セキュリティ | [ ] |
| 2 | ルートユーザーのアクセスキーを削除/未作成確認 | 必須 | セキュリティ | [ ] |
| 3 | IAM 管理者ユーザーを作成 | 必須 | セキュリティ | [ ] |
| 4 | パスワードポリシーを設定 | 必須 | セキュリティ | [ ] |
| 5 | IAM グループを作成し、ポリシーをアタッチ | 高 | IAM | [ ] |
| 6 | アカウントエイリアスを設定 | 高 | アカウント | [ ] |
| 7 | CloudTrail を有効化 | 高 | 監査 | [ ] |
| 8 | 請求アラートを設定 | 高 | コスト | [ ] |
| 9 | AWS Budgets を設定 | 高 | コスト | [ ] |
| 10 | AWS Config を有効化 | 中 | コンプライアンス | [ ] |
| 11 | GuardDuty を有効化 | 中 | セキュリティ | [ ] |
| 12 | Security Hub を有効化 | 中 | セキュリティ | [ ] |
| 13 | EBS デフォルト暗号化を有効化 | 中 | セキュリティ | [ ] |
| 14 | S3 パブリックアクセスブロック（アカウントレベル） | 中 | セキュリティ | [ ] |
| 15 | Organizations で環境分離 | 中 | ガバナンス | [ ] |
| 16 | IAM Identity Center の設定 | 中 | IAM | [ ] |
| 17 | コスト異常検知を設定 | 低 | コスト | [ ] |
| 18 | VPC フローログの有効化 | 低 | 監査 | [ ] |

---

## 10. Terraform による IAM 管理

### 10.1 IAM モジュール

```hcl
# main.tf
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "ap-northeast-1"
}

# IAM グループ
resource "aws_iam_group" "developers" {
  name = "developers"
  path = "/teams/"
}

resource "aws_iam_group" "readonly" {
  name = "readonly"
  path = "/teams/"
}

# グループポリシーアタッチメント
resource "aws_iam_group_policy_attachment" "developers_power_user" {
  group      = aws_iam_group.developers.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

resource "aws_iam_group_policy_attachment" "readonly_access" {
  group      = aws_iam_group.readonly.name
  policy_arn = "arn:aws:iam::aws:policy/ReadOnlyAccess"
}

# カスタムポリシー
resource "aws_iam_policy" "developer_restrictions" {
  name        = "developer-restrictions"
  description = "開発者の権限制限"
  policy      = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DenyDangerousActions"
        Effect = "Deny"
        Action = [
          "iam:CreateUser",
          "iam:DeleteUser",
          "iam:CreateRole",
          "iam:DeleteRole",
          "organizations:*",
        ]
        Resource = "*"
      },
      {
        Sid    = "RestrictRegions"
        Effect = "Deny"
        NotAction = [
          "iam:*",
          "sts:*",
          "support:*",
        ]
        Resource = "*"
        Condition = {
          StringNotEquals = {
            "aws:RequestedRegion" = ["ap-northeast-1", "us-east-1"]
          }
        }
      }
    ]
  })
}

resource "aws_iam_group_policy_attachment" "developers_restrictions" {
  group      = aws_iam_group.developers.name
  policy_arn = aws_iam_policy.developer_restrictions.arn
}

# EC2 用ロール
resource "aws_iam_role" "ec2_webserver" {
  name = "ec2-webserver-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
  max_session_duration = 14400  # 4時間
}

resource "aws_iam_role_policy_attachment" "ec2_s3_readonly" {
  role       = aws_iam_role.ec2_webserver.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_instance_profile" "ec2_webserver" {
  name = "ec2-webserver-profile"
  role = aws_iam_role.ec2_webserver.name
}

# パスワードポリシー
resource "aws_iam_account_password_policy" "strict" {
  minimum_password_length        = 14
  require_lowercase_characters   = true
  require_numbers                = true
  require_uppercase_characters   = true
  require_symbols                = true
  allow_users_to_change_password = true
  max_password_age               = 90
  password_reuse_prevention      = 12
  hard_expiry                    = true
}

# 出力
output "ec2_webserver_role_arn" {
  value       = aws_iam_role.ec2_webserver.arn
  description = "EC2 Web Server ロールの ARN"
}
```

---

## 11. アンチパターン

### アンチパターン 1: ルートユーザーで日常操作する

ルートユーザーは権限を制限できないため、誤操作や漏洩時のリスクが甚大。アカウント設定の変更（支払い情報、アカウント閉鎖）以外は IAM ユーザーまたは IAM Identity Center (SSO) で運用すべきである。

```
# 悪い例
ルートユーザーで毎日 EC2 を操作
↓
# 良い例
IAM ユーザー (MFA有効) で操作
ルートユーザーは金庫に保管（物理 MFA 推奨）
```

### アンチパターン 2: IAM ユーザーにアクセスキーを長期間放置する

アクセスキーは漏洩リスクがあるため、90日ごとにローテーションし、不要なキーは即座に削除する。可能であれば IAM ロール（一時認証情報）を使うべきである。

```bash
# アクセスキーの最終使用日を確認
aws iam get-access-key-last-used \
  --access-key-id AKIAIOSFODNN7EXAMPLE

# 90日以上未使用のキーを一覧表示
aws iam list-access-keys --user-name tanaka
# → CreateDate を確認し、古いキーは無効化 → 削除

# アクセスキーのローテーションスクリプト
#!/bin/bash
USER_NAME="tanaka"
OLD_KEY_ID=$(aws iam list-access-keys --user-name "$USER_NAME" \
  --query 'AccessKeyMetadata[0].AccessKeyId' --output text)

# 新しいキーを作成
NEW_KEY=$(aws iam create-access-key --user-name "$USER_NAME")
echo "新しいアクセスキー: $(echo $NEW_KEY | jq -r '.AccessKey.AccessKeyId')"

# アプリケーションに新しいキーを設定した後に古いキーを削除
# aws iam delete-access-key --user-name "$USER_NAME" --access-key-id "$OLD_KEY_ID"
```

### アンチパターン 3: 全員に AdministratorAccess を付与する

開発スピードを優先して全員に管理者権限を付与するのは危険。最小権限の原則に基づき、役割に応じた権限を設計する。

```
# 悪い例
全開発者 → AdministratorAccess
→ 誤って本番 DB を削除するリスク

# 良い例
開発者 → PowerUserAccess + カスタム制限
SRE   → AdministratorAccess（限定メンバーのみ）
QA    → ReadOnlyAccess + テスト環境の操作権限
```

### アンチパターン 4: 単一アカウントで全環境を運用する

本番・ステージング・開発環境を1つのアカウントで運用すると、権限分離が困難になり、開発環境の誤操作が本番に影響するリスクがある。

```
# 悪い例
1つのアカウントに prod, staging, dev のリソースが混在
→ タグで区別（タグの付け忘れでリスク）

# 良い例
Organizations で環境ごとにアカウント分離
prod: 123456789012
staging: 234567890123
dev: 345678901234
→ SCP でガードレール、IAM Identity Center で一元管理
```

---

## 12. FAQ

### Q1. 無料枠の範囲はどこまでか？

AWS 無料枠には3種類ある。(1) 12ヶ月無料枠（EC2 t2.micro 750時間/月など）、(2) 常時無料（Lambda 100万リクエスト/月、DynamoDB 25GB など）、(3) トライアル（一部サービスの期間限定無料）。詳細は https://aws.amazon.com/free/ を確認する。

### Q2. アカウントが不正利用されたらどうする？

(1) ルートユーザーのパスワードを即座に変更、(2) 全アクセスキーを無効化、(3) 不正なリソースを停止・削除、(4) AWS サポートに連絡。事前対策として CloudTrail のログ監視と GuardDuty の有効化が重要。

```bash
# 緊急対応スクリプト
#!/bin/bash
echo "=== 不正アクセス緊急対応 ==="

# 1. 全 IAM ユーザーのアクセスキーを無効化
for user in $(aws iam list-users --query 'Users[].UserName' --output text); do
  for key in $(aws iam list-access-keys --user-name "$user" \
    --query 'AccessKeyMetadata[].AccessKeyId' --output text); do
    aws iam update-access-key --user-name "$user" --access-key-id "$key" --status Inactive
    echo "無効化: $user / $key"
  done
done

# 2. 不審な EC2 インスタンスを停止
aws ec2 describe-instances \
  --filters "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].[InstanceId,LaunchTime,InstanceType]' \
  --output table

# 3. CloudTrail で不審なアクティビティを確認
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=RunInstances \
  --start-time "$(date -v-24H -u +%Y-%m-%dT%H:%M:%SZ)" \
  --query 'Events[].[EventTime,Username,EventName]' \
  --output table
```

### Q3. IAM Identity Center (旧 SSO) と IAM ユーザーの使い分けは？

AWS Organizations を使う場合は IAM Identity Center を推奨。シングルサインオン、一元的なアクセス管理、一時認証情報の自動発行が利点。小規模・単一アカウントであれば IAM ユーザー + MFA でも十分。

### Q4. AWS アカウントを閉鎖する手順は？

アカウント閉鎖は取り消し不可能（90日以内は復元可能だが保証されない）。閉鎖前に: (1) 必要なデータを他のアカウントに移行、(2) Route 53 ドメインを移管、(3) サポートケースをクローズ、(4) ルートユーザーでコンソールから閉鎖を実行。

### Q5. 複数リージョンの利用に関する注意点は？

IAM はグローバルサービスだが、他のほとんどのサービスはリージョナル。CloudTrail はマルチリージョン対応を有効化し、SCP でリージョン制限をかけることを推奨。主な例外: IAM、Route 53、CloudFront、WAF (Global)、Organizations。

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| ルートユーザー | MFA 必須、日常利用禁止、アクセスキー作成禁止 |
| IAM 設計 | グループベースの権限管理、最小権限の原則 |
| IAM ロール | EC2/Lambda からの AWS サービスアクセスに使用 |
| アクセス許可境界 | 権限委任時の上限設定で安全性確保 |
| MFA | 全 IAM ユーザーに設定、ルートには FIDO2 推奨 |
| IAM Identity Center | マルチアカウント運用での一元的アクセス管理 |
| Organizations | 環境ごとにアカウント分離、SCP でガードレール |
| Control Tower | ベストプラクティスベースのランディングゾーン |
| コスト管理 | Budgets + CloudWatch アラーム + 異常検知で超過を早期検知 |
| IaC | CloudFormation / CDK / Terraform で IAM をコード管理 |

---

## 次に読むべきガイド

- [02-aws-cli-sdk.md](./02-aws-cli-sdk.md) — CLI/SDK のセットアップと認証情報管理
- [../01-compute/00-ec2-basics.md](../01-compute/00-ec2-basics.md) — EC2 インスタンスの基礎

---

## 参考文献

1. AWS IAM ベストプラクティス — https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html
2. AWS Organizations ユーザーガイド — https://docs.aws.amazon.com/organizations/latest/userguide/
3. AWS Security Best Practices (Whitepaper) — https://docs.aws.amazon.com/whitepapers/latest/aws-security-best-practices/
4. AWS Well-Architected Framework — Security Pillar — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
5. IAM Identity Center ユーザーガイド — https://docs.aws.amazon.com/singlesignon/latest/userguide/
6. AWS Control Tower ユーザーガイド — https://docs.aws.amazon.com/controltower/latest/userguide/
7. AWS Cost Management ユーザーガイド — https://docs.aws.amazon.com/cost-management/latest/userguide/
8. AWS CDK IAM モジュール — https://docs.aws.amazon.com/cdk/api/v2/docs/aws-cdk-lib.aws_iam-readme.html
