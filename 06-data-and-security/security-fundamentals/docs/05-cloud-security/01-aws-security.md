# AWS セキュリティ

> GuardDuty による脅威検知、Security Hub による統合管理、CloudTrail による監査ログまで、AWS 環境を安全に運用するための実践ガイド

## この章で学ぶこと

1. **GuardDuty による脅威検知** — 機械学習ベースの異常検知と自動通知の設定
2. **Security Hub による統合管理** — セキュリティ検知結果の集約とコンプライアンスチェック
3. **CloudTrail と Config による監査** — 全 API 呼び出しの記録と構成変更の追跡

---

## 1. AWS セキュリティサービスの全体像

### サービスマップ

```
+----------------------------------------------------------+
|              AWS セキュリティサービス群                      |
|----------------------------------------------------------|
|                                                          |
|  [検知・脅威対応]                                          |
|  +-- GuardDuty: 脅威検知 (VPC Flow, DNS, CloudTrail)     |
|  +-- Inspector: EC2/ECR 脆弱性スキャン                    |
|  +-- Macie: S3 の機密データ検出                           |
|  +-- Detective: セキュリティ調査・分析                     |
|                                                          |
|  [統合管理]                                               |
|  +-- Security Hub: 検知結果の集約・コンプライアンス         |
|  +-- Config: リソース構成の記録・評価                      |
|  +-- Organizations: マルチアカウント統制                   |
|                                                          |
|  [監査・ログ]                                             |
|  +-- CloudTrail: API 監査ログ                            |
|  +-- VPC Flow Logs: ネットワークトラフィックログ           |
|  +-- CloudWatch Logs: アプリ・インフラログ                |
|                                                          |
|  [アクセス制御]                                           |
|  +-- IAM: ユーザ・ロール・ポリシー管理                    |
|  +-- IAM Access Analyzer: 外部アクセスの検出              |
|  +-- STS: 一時的認証情報の発行                            |
|                                                          |
|  [データ保護]                                             |
|  +-- KMS: 暗号鍵管理                                    |
|  +-- Secrets Manager: シークレット管理・自動ローテーション  |
|  +-- ACM: TLS 証明書管理                                 |
|                                                          |
|  [ネットワーク保護]                                       |
|  +-- WAF: Web アプリケーションファイアウォール              |
|  +-- Shield: DDoS 対策                                   |
|  +-- Network Firewall: VPC レベルのファイアウォール        |
+----------------------------------------------------------+
```

---

## 2. GuardDuty

### GuardDuty の仕組み

```
データソース                 GuardDuty                  対応
+------------------+     +------------------+     +------------------+
| CloudTrail Logs  | --> |                  | --> | EventBridge      |
| VPC Flow Logs    | --> | 機械学習エンジン  | --> | → Lambda         |
| DNS Logs         | --> | 脅威インテリジェンス| --> | → SNS 通知      |
| EKS Audit Logs   | --> | 異常検知          | --> | → 自動修復      |
| S3 Data Events   | --> |                  | --> | → Slack 通知    |
+------------------+     +------------------+     +------------------+
                           |
                           v
                    Finding (検知結果)
                    - 重大度: Low/Medium/High
                    - 分類: 200+ の検知タイプ
```

### GuardDuty の有効化と設定

```python
import boto3

guardduty = boto3.client('guardduty', region_name='ap-northeast-1')

# GuardDuty の有効化
response = guardduty.create_detector(
    Enable=True,
    DataSources={
        'S3Logs': {'Enable': True},
        'Kubernetes': {'AuditLogs': {'Enable': True}},
        'MalwareProtection': {
            'ScanEc2InstanceWithFindings': {
                'EbsVolumes': True,
            }
        },
    },
    Features=[
        {'Name': 'EKS_RUNTIME_MONITORING', 'Status': 'ENABLED'},
        {'Name': 'LAMBDA_NETWORK_LOGS', 'Status': 'ENABLED'},
    ],
)

detector_id = response['DetectorId']
```

### GuardDuty 検知結果の自動通知

```python
# EventBridge → Lambda → Slack 通知
import json
import urllib.request

def lambda_handler(event, context):
    """GuardDuty の検知結果を Slack に通知"""
    detail = event['detail']
    severity = detail['severity']
    finding_type = detail['type']
    description = detail['description']
    account_id = detail['accountId']
    region = detail['region']

    # 重大度に応じた色分け
    if severity >= 7:
        color = '#ff0000'  # 赤: High
        emoji = ':rotating_light:'
    elif severity >= 4:
        color = '#ff9900'  # オレンジ: Medium
        emoji = ':warning:'
    else:
        color = '#ffcc00'  # 黄: Low
        emoji = ':information_source:'

    slack_message = {
        'attachments': [{
            'color': color,
            'title': f'{emoji} GuardDuty Alert: {finding_type}',
            'fields': [
                {'title': 'Account', 'value': account_id, 'short': True},
                {'title': 'Region', 'value': region, 'short': True},
                {'title': 'Severity', 'value': str(severity), 'short': True},
                {'title': 'Description', 'value': description},
            ],
        }]
    }

    webhook_url = os.environ['SLACK_WEBHOOK_URL']
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(slack_message).encode(),
        headers={'Content-Type': 'application/json'},
    )
    urllib.request.urlopen(req)
```

---

## 3. Security Hub

### Security Hub の構成

```
+----------------------------------------------------------+
|                    Security Hub                           |
|----------------------------------------------------------|
|                                                          |
|  [データソース (Findings)]                                 |
|  +-- GuardDuty の検知結果                                 |
|  +-- Inspector の脆弱性                                   |
|  +-- Macie の機密データ検出                                |
|  +-- IAM Access Analyzer                                 |
|  +-- Firewall Manager                                    |
|  +-- サードパーティ (Prowler, Checkov)                     |
|                                                          |
|  [セキュリティ基準 (Standards)]                            |
|  +-- AWS Foundational Security Best Practices            |
|  +-- CIS AWS Foundations Benchmark                       |
|  +-- PCI DSS                                            |
|  +-- NIST SP 800-53                                      |
|                                                          |
|  [出力]                                                   |
|  +-- ダッシュボード (スコアカード)                          |
|  +-- EventBridge 統合 (自動修復)                          |
|  +-- カスタムアクション                                   |
+----------------------------------------------------------+
```

### Security Hub の有効化 (Terraform)

```hcl
# Security Hub の有効化
resource "aws_securityhub_account" "main" {}

# AWS Foundational Security Best Practices
resource "aws_securityhub_standards_subscription" "aws_foundational" {
  standards_arn = "arn:aws:securityhub:ap-northeast-1::standards/aws-foundational-security-best-practices/v/1.0.0"
  depends_on    = [aws_securityhub_account.main]
}

# CIS AWS Foundations Benchmark
resource "aws_securityhub_standards_subscription" "cis" {
  standards_arn = "arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark/v/1.4.0"
  depends_on    = [aws_securityhub_account.main]
}

# 自動修復用 EventBridge ルール
resource "aws_cloudwatch_event_rule" "securityhub_high" {
  name = "securityhub-high-severity"
  event_pattern = jsonencode({
    source      = ["aws.securityhub"]
    detail-type = ["Security Hub Findings - Imported"]
    detail = {
      findings = {
        Severity = {
          Label = ["CRITICAL", "HIGH"]
        }
      }
    }
  })
}
```

---

## 4. CloudTrail

### CloudTrail の設定

```hcl
# CloudTrail (全リージョン、全イベント)
resource "aws_cloudtrail" "main" {
  name                          = "organization-trail"
  s3_bucket_name                = aws_s3_bucket.cloudtrail.id
  is_organization_trail         = true
  is_multi_region_trail         = true
  enable_log_file_validation    = true  # ログの改竄検知
  kms_key_id                    = aws_kms_key.cloudtrail.arn

  # 管理イベント
  event_selector {
    read_write_type           = "All"
    include_management_events = true
  }

  # S3 データイベント
  event_selector {
    read_write_type           = "WriteOnly"
    include_management_events = false

    data_resource {
      type   = "AWS::S3::Object"
      values = ["arn:aws:s3:::sensitive-bucket/"]
    }
  }

  # CloudWatch Logs に転送
  cloud_watch_logs_group_arn = "${aws_cloudwatch_log_group.cloudtrail.arn}:*"
  cloud_watch_logs_role_arn  = aws_iam_role.cloudtrail_cloudwatch.arn
}
```

### CloudTrail ログの分析

```python
import boto3
from datetime import datetime, timedelta

# CloudTrail Insights / Athena での分析
athena = boto3.client('athena')

# 疑わしい API 呼び出しの検索 (Athena)
query = """
SELECT
    eventTime,
    eventName,
    userIdentity.arn as userArn,
    sourceIPAddress,
    errorCode
FROM cloudtrail_logs
WHERE eventTime > date_add('day', -1, now())
  AND (
    eventName IN ('DeleteTrail', 'StopLogging', 'DeleteFlowLogs')
    OR eventName LIKE 'Disable%'
    OR (eventName = 'ConsoleLogin' AND errorCode = 'Failed')
  )
ORDER BY eventTime DESC
LIMIT 100;
"""

# AWS Config Rules で継続的な構成監視
config = boto3.client('config')
config.put_config_rule(
    ConfigRule={
        'ConfigRuleName': 's3-bucket-public-read-prohibited',
        'Source': {
            'Owner': 'AWS',
            'SourceIdentifier': 'S3_BUCKET_PUBLIC_READ_PROHIBITED',
        },
    }
)
```

---

## 5. Secrets Manager

### シークレットの管理と自動ローテーション

```python
import boto3
import json

sm = boto3.client('secretsmanager')

# シークレットの作成
sm.create_secret(
    Name='myapp/database/credentials',
    SecretString=json.dumps({
        'username': 'app_user',
        'password': 'initial-password',
        'engine': 'postgres',
        'host': 'mydb.xxx.ap-northeast-1.rds.amazonaws.com',
        'port': 5432,
        'dbname': 'myapp',
    }),
    KmsKeyId='alias/myapp-secrets-key',
    Tags=[
        {'Key': 'Environment', 'Value': 'production'},
    ],
)

# シークレットの取得
response = sm.get_secret_value(SecretId='myapp/database/credentials')
credentials = json.loads(response['SecretString'])

# 自動ローテーションの設定
sm.rotate_secret(
    SecretId='myapp/database/credentials',
    RotationLambdaARN='arn:aws:lambda:ap-northeast-1:123456:function:rotate-db-secret',
    RotationRules={
        'AutomaticallyAfterDays': 30,
    },
)
```

---

## 6. アンチパターン

### アンチパターン 1: CloudTrail の無効化

```
NG:
  → CloudTrail を無効にしてコスト削減
  → 単一リージョンのみ有効
  → ログファイルの整合性検証を無効化

OK:
  → 全リージョン + 全イベントを記録
  → ログファイル検証を有効化
  → S3 バケットを KMS で暗号化し MFA Delete を有効化
  → ログアーカイブ用の別アカウントに保存
```

### アンチパターン 2: アクセスキーの長期使用

```bash
# NG: アクセスキーを作成して永久に使い続ける
aws iam create-access-key --user-name deploy-user
# → 1年以上ローテーションされないアクセスキー

# OK: IAM ロールと一時認証情報を使用
# EC2: インスタンスプロファイル
# Lambda: 実行ロール
# CI/CD: OIDC プロバイダ + AssumeRoleWithWebIdentity

# GitHub Actions の例:
# - uses: aws-actions/configure-aws-credentials@v4
#   with:
#     role-to-assume: arn:aws:iam::123456:role/github-actions-role
#     aws-region: ap-northeast-1
```

---

## 7. FAQ

### Q1. GuardDuty の検知結果が多すぎる場合はどうするか?

GuardDuty の Suppression Rules で低リスクの検知結果を抑制できる。まず全検知結果を確認して誤検知パターンを特定し、Findings Filter で自動アーカイブする。ただし、High 以上の検知結果は抑制せず必ず調査すること。

### Q2. Security Hub のスコアをどこまで上げるべきか?

100% は現実的でない場合もある。CRITICAL と HIGH の検知結果は全て対応し、全体スコアは 90% 以上を目標とする。例外が必要な場合は Suppression ルールを適用し、その理由をドキュメント化する。

### Q3. CloudTrail のコストを最適化するには?

管理イベントは基本的に全記録する (コスト影響は小さい)。データイベント (S3/Lambda) は機密バケットに限定する。CloudTrail Lake の代わりに Athena + S3 で分析することでコストを抑えられる。ログの保持期間を規制要件に合わせて設定し、不要に長期保存しない。

---

## まとめ

| 項目 | 要点 |
|------|------|
| GuardDuty | 全アカウント・全リージョンで有効化、EventBridge で自動通知 |
| Security Hub | 検知結果を集約、CIS/AWS ベストプラクティスで評価 |
| CloudTrail | 全 API 呼び出しを記録、ログ検証必須、別アカウントに保存 |
| Config | リソース構成の継続監視、ルール違反を自動検出 |
| IAM | ロールベースアクセス、一時認証情報、MFA 必須 |
| Secrets Manager | シークレットの一元管理と自動ローテーション |
| KMS | データ暗号化の鍵管理、エンベロープ暗号化 |

---

## 次に読むべきガイド

- [クラウドセキュリティ基礎](./00-cloud-security-basics.md) — 責任共有モデルと IAM の基本
- [IaCセキュリティ](./02-infrastructure-as-code-security.md) — Terraform/CloudFormation のセキュリティチェック
- [監視/ログ](../06-operations/01-monitoring-logging.md) — SIEM との統合

---

## 参考文献

1. **AWS Security Best Practices** — https://docs.aws.amazon.com/prescriptive-guidance/latest/security-best-practices/
2. **AWS GuardDuty User Guide** — https://docs.aws.amazon.com/guardduty/latest/ug/
3. **CIS Amazon Web Services Foundations Benchmark** — https://www.cisecurity.org/benchmark/amazon_web_services
4. **AWS Security Hub User Guide** — https://docs.aws.amazon.com/securityhub/latest/userguide/
