# AWS アカウント設定

> AWS を安全かつ効率的に利用するための初期設定 — アカウント作成から IAM、MFA、Organizations、請求アラートまで

## この章で学ぶこと

1. AWS アカウントを作成し、ルートユーザーのセキュリティを確保できる
2. IAM ユーザー・グループ・ポリシーを適切に設計し、最小権限の原則を適用できる
3. AWS Organizations と請求アラートを設定し、マルチアカウント運用とコスト管理を実現できる

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

### 2.2 MFA (多要素認証) の設定

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

### 2.3 MFA の種類比較

| MFA タイプ | セキュリティ | 利便性 | コスト | 推奨用途 |
|-----------|------------|--------|--------|---------|
| 仮想 MFA (TOTP) | 中 | 高 | 無料 | IAM ユーザー |
| FIDO2 セキュリティキー | 高 | 中 | 有料 | ルートユーザー |
| ハードウェア MFA | 最高 | 低 | 有料 | ルート/高権限 |

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

### 3.2 コード例: IAM ユーザーとグループの作成

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
```

### 3.3 コード例: カスタム IAM ポリシー (JSON)

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

### 3.4 コード例: IAM ロールの作成 (EC2 用)

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

### 3.5 最小権限の原則

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

---

## 4. AWS Organizations

### 4.1 マルチアカウント戦略

```
AWS Organizations 構成例
+----------------------------------------------------+
| Management Account (請求統合・ガバナンス)             |
|                                                     |
| ├── OU: Security                                    |
| │   ├── Log Archive Account (CloudTrail, Config)    |
| │   └── Security Tooling Account (GuardDuty, etc.)  |
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

### 4.2 コード例: Organizations の操作

```bash
# 組織を作成
aws organizations create-organization --feature-set ALL

# OU (組織単位) を作成
aws organizations create-organizational-unit \
  --parent-id r-xxxx \
  --name "Workloads"

# 新しいメンバーアカウントを作成
aws organizations create-account \
  --email prod@example.com \
  --account-name "Production"

# SCP（サービスコントロールポリシー）をアタッチ
aws organizations attach-policy \
  --policy-id p-xxxx \
  --target-id ou-xxxx
```

### 4.3 Service Control Policy (SCP) 例

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
        "organizations:*"
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

---

## 5. 請求アラートとコスト管理

### 5.1 コード例: 請求アラートの設定 (CloudWatch)

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
  --alarm-name "MonthlyBillingAlarm" \
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
```

### 5.2 AWS Budgets の設定

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
    }
  ]'
```

---

## 6. 初期設定チェックリスト

| # | タスク | 優先度 | 完了 |
|---|--------|--------|------|
| 1 | ルートユーザーに MFA を設定 | 必須 | [ ] |
| 2 | ルートユーザーのアクセスキーを削除/未作成確認 | 必須 | [ ] |
| 3 | IAM 管理者ユーザーを作成 | 必須 | [ ] |
| 4 | パスワードポリシーを設定 | 必須 | [ ] |
| 5 | IAM グループを作成し、ポリシーをアタッチ | 高 | [ ] |
| 6 | CloudTrail を有効化 | 高 | [ ] |
| 7 | 請求アラートを設定 | 高 | [ ] |
| 8 | AWS Config を有効化 | 中 | [ ] |
| 9 | GuardDuty を有効化 | 中 | [ ] |
| 10 | Organizations で環境分離 | 中 | [ ] |

---

## 7. アンチパターン

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
```

---

## 8. FAQ

### Q1. 無料枠の範囲はどこまでか？

AWS 無料枠には3種類ある。(1) 12ヶ月無料枠（EC2 t2.micro 750時間/月など）、(2) 常時無料（Lambda 100万リクエスト/月、DynamoDB 25GB など）、(3) トライアル（一部サービスの期間限定無料）。詳細は https://aws.amazon.com/free/ を確認する。

### Q2. アカウントが不正利用されたらどうする？

(1) ルートユーザーのパスワードを即座に変更、(2) 全アクセスキーを無効化、(3) 不正なリソースを停止・削除、(4) AWS サポートに連絡。事前対策として CloudTrail のログ監視と GuardDuty の有効化が重要。

### Q3. IAM Identity Center (旧 SSO) と IAM ユーザーの使い分けは？

AWS Organizations を使う場合は IAM Identity Center を推奨。シングルサインオン、一元的なアクセス管理、一時認証情報の自動発行が利点。小規模・単一アカウントであれば IAM ユーザー + MFA でも十分。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| ルートユーザー | MFA 必須、日常利用禁止、アクセスキー作成禁止 |
| IAM 設計 | グループベースの権限管理、最小権限の原則 |
| IAM ロール | EC2/Lambda からの AWS サービスアクセスに使用 |
| MFA | 全 IAM ユーザーに設定、ルートには FIDO2 推奨 |
| Organizations | 環境ごとにアカウント分離、SCP でガードレール |
| コスト管理 | Budgets + CloudWatch アラームで超過を早期検知 |

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
