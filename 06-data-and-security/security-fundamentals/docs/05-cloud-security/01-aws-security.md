# AWS セキュリティ

> GuardDuty による脅威検知、Security Hub による統合管理、CloudTrail による監査ログまで、AWS 環境を安全に運用するための実践ガイド

## 前提知識

- AWS の基本的なサービス（EC2, S3, IAM, VPC）の理解
- [クラウドセキュリティ基礎](./00-cloud-security-basics.md) の責任共有モデルと IAM の知識
- JSON によるポリシー記述の基礎

## この章で学ぶこと

1. **GuardDuty による脅威検知** — 機械学習ベースの異常検知と自動通知の設定
2. **Security Hub による統合管理** — セキュリティ検知結果の集約とコンプライアンスチェック
3. **CloudTrail と Config による監査** — 全 API 呼び出しの記録と構成変更の追跡
4. **Secrets Manager によるシークレット管理** — 認証情報の安全な保管と自動ローテーション
5. **WAF と Shield による境界防御** — Web アプリケーションの保護と DDoS 対策
6. **IAM Access Analyzer による権限分析** — 過剰な権限と外部アクセスの検出

---

## 1. AWS セキュリティサービスの全体像

### サービスマップ

```
+------------------------------------------------------------------+
|              AWS セキュリティサービス群                              |
|------------------------------------------------------------------|
|                                                                  |
|  [検知・脅威対応]                                                 |
|  +-- GuardDuty: 脅威検知 (VPC Flow, DNS, CloudTrail, EKS)      |
|  +-- Inspector: EC2/ECR/Lambda 脆弱性スキャン                    |
|  +-- Macie: S3 の機密データ検出 (PII, クレジットカード番号)       |
|  +-- Detective: セキュリティ調査・分析 (グラフベース)             |
|                                                                  |
|  [統合管理]                                                      |
|  +-- Security Hub: 検知結果の集約・コンプライアンス               |
|  +-- Config: リソース構成の記録・評価                             |
|  +-- Organizations: マルチアカウント統制                          |
|  +-- Control Tower: ランディングゾーンの自動セットアップ          |
|                                                                  |
|  [監査・ログ]                                                    |
|  +-- CloudTrail: API 監査ログ (管理/データイベント)              |
|  +-- VPC Flow Logs: ネットワークトラフィックログ                  |
|  +-- CloudWatch Logs: アプリ・インフラログ                       |
|  +-- S3 Server Access Logging: バケットアクセスログ               |
|                                                                  |
|  [アクセス制御]                                                  |
|  +-- IAM: ユーザ・ロール・ポリシー管理                           |
|  +-- IAM Access Analyzer: 外部アクセスの検出                     |
|  +-- IAM Identity Center (SSO): 一元的な認証管理                |
|  +-- STS: 一時的認証情報の発行                                   |
|                                                                  |
|  [データ保護]                                                    |
|  +-- KMS: 暗号鍵管理 (エンベロープ暗号化)                       |
|  +-- Secrets Manager: シークレット管理・自動ローテーション        |
|  +-- ACM: TLS 証明書管理                                        |
|  +-- CloudHSM: 専用ハードウェアセキュリティモジュール             |
|                                                                  |
|  [ネットワーク保護]                                              |
|  +-- WAF: Web アプリケーションファイアウォール                    |
|  +-- Shield: DDoS 対策 (Standard/Advanced)                     |
|  +-- Network Firewall: VPC レベルのファイアウォール               |
|  +-- Firewall Manager: 組織全体のファイアウォール管理             |
|                                                                  |
+------------------------------------------------------------------+
```

### AWS セキュリティサービスの関係性

```
┌──────────── AWS セキュリティサービスの連携 ────────────┐
│                                                        │
│   CloudTrail ──┐                                       │
│   VPC Flow Logs─┤                                      │
│   DNS Logs ─────┤──→ GuardDuty ──┐                    │
│   EKS Audit ────┤                │                    │
│   S3 Events ────┘                ├──→ Security Hub    │
│                                  │     (Findings 集約) │
│   Inspector ─────────────────────┤                    │
│   Macie ─────────────────────────┤                    │
│   IAM Access Analyzer ───────────┤                    │
│   Firewall Manager ──────────────┘                    │
│                                                        │
│   Security Hub ──→ EventBridge ──→ Lambda             │
│                                    ├→ SNS (通知)      │
│                                    ├→ Step Functions  │
│                                    └→ 自動修復         │
│                                                        │
│   Config Rules ──→ 非準拠検出 ──→ SSM Automation     │
│                                    └→ 自動修復         │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### セキュリティサービスの導入優先度

| 優先度 | サービス | 理由 | コスト影響 |
|--------|---------|------|-----------|
| P0 (必須) | CloudTrail | 全 API 操作の記録。インシデント調査の基盤 | 低 (管理イベントは無料) |
| P0 (必須) | IAM | 最小権限の原則。全アクセス制御の基盤 | 無料 |
| P0 (必須) | S3 暗号化 + パブリックアクセスブロック | データ漏洩防止 | 無料〜低 |
| P1 (推奨) | GuardDuty | 継続的な脅威検知 | 中 (データ量依存) |
| P1 (推奨) | Security Hub | 検知結果の集約と基準評価 | 低 |
| P1 (推奨) | Config | リソース構成の継続監視 | 中 |
| P2 (重要) | Secrets Manager | シークレットの安全管理 | 低 (シークレット数依存) |
| P2 (重要) | WAF | Web アプリケーション保護 | 中 |
| P2 (重要) | Inspector | 脆弱性の自動検出 | 中 |
| P3 (推奨) | Macie | 機密データの検出 | 中〜高 |
| P3 (推奨) | Detective | インシデント調査の効率化 | 中 |

---

## 2. GuardDuty

### GuardDuty の仕組み

GuardDuty は AWS 環境における継続的な脅威検知サービスである。機械学習、異常検知、脅威インテリジェンスフィードを組み合わせて、悪意のあるアクティビティや不正な動作を検出する。

```
データソース                 GuardDuty                  対応
+------------------+     +------------------+     +------------------+
| CloudTrail Logs  | --> |                  | --> | EventBridge      |
| VPC Flow Logs    | --> | 機械学習エンジン  | --> | → Lambda         |
| DNS Logs         | --> | 脅威インテリジェンス| --> | → SNS 通知      |
| EKS Audit Logs   | --> | 異常検知          | --> | → 自動修復      |
| S3 Data Events   | --> |                  | --> | → Slack 通知    |
| RDS Login Events | --> |                  | --> | → SIEM 連携     |
| Lambda Network   | --> |                  | --> | → Security Hub  |
| EBS Malware Scan | --> |                  |     |                  |
+------------------+     +------------------+     +------------------+
                           |
                           v
                    Finding (検知結果)
                    - 重大度: Low (1-3.9) / Medium (4-6.9) / High (7-8.9) / Critical (9-10)
                    - 分類: 200+ の検知タイプ
                    - カテゴリ: Recon / UnauthorizedAccess / Exfiltration など
```

#### GuardDuty の検知タイプ例

| カテゴリ | 検知タイプ | 説明 |
|---------|-----------|------|
| 偵察活動 | Recon:EC2/PortProbeUnprotectedPort | EC2 インスタンスの保護されていないポートへの偵察 |
| 不正アクセス | UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration.OutsideAWS | EC2 の認証情報が AWS 外で使用 |
| 暗号通貨 | CryptoCurrency:EC2/BitcoinTool.B | EC2 で暗号通貨マイニングの通信を検出 |
| データ流出 | Exfiltration:S3/MaliciousIPCaller | 悪意のある IP から S3 へのアクセス |
| トロイの木馬 | Trojan:EC2/BlackholeTraffic | EC2 からブラックホール IP への通信 |
| バックドア | Backdoor:EC2/DenialOfService.Tcp | EC2 が DDoS 攻撃に利用されている |
| 特権昇格 | PrivilegeEscalation:IAMUser/AdministrativePermissions | 通常と異なる管理者権限の使用 |

### GuardDuty の有効化と設定

```python
import boto3
import json

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
        {'Name': 'RDS_LOGIN_EVENTS', 'Status': 'ENABLED'},
        {'Name': 'EBS_MALWARE_PROTECTION', 'Status': 'ENABLED'},
    ],
    FindingPublishingFrequency='FIFTEEN_MINUTES',
)

detector_id = response['DetectorId']
print(f"GuardDuty enabled with detector ID: {detector_id}")
```

```hcl
# Terraform による GuardDuty の設定
resource "aws_guardduty_detector" "main" {
  enable                       = true
  finding_publishing_frequency = "FIFTEEN_MINUTES"

  datasources {
    s3_logs {
      enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
    malware_protection {
      scan_ec2_instance_with_findings {
        ebs_volumes {
          enable = true
        }
      }
    }
  }

  tags = {
    Environment = "production"
    ManagedBy   = "terraform"
  }
}

# GuardDuty メンバーアカウントの招待（Organizations 経由）
resource "aws_guardduty_organization_configuration" "main" {
  detector_id                      = aws_guardduty_detector.main.id
  auto_enable_organization_members = "ALL"

  datasources {
    s3_logs {
      auto_enable = true
    }
    kubernetes {
      audit_logs {
        enable = true
      }
    }
  }
}

# Trusted IP リスト（オフィス IP を除外）
resource "aws_guardduty_ipset" "trusted" {
  detector_id = aws_guardduty_detector.main.id
  name        = "trusted-ips"
  format      = "TXT"
  location    = "s3://${aws_s3_bucket.security.id}/guardduty/trusted-ips.txt"
  activate    = true
}

# 脅威 IP リスト（カスタム脅威インテリジェンス）
resource "aws_guardduty_threatintelset" "custom" {
  detector_id = aws_guardduty_detector.main.id
  name        = "custom-threat-intel"
  format      = "TXT"
  location    = "s3://${aws_s3_bucket.security.id}/guardduty/threat-ips.txt"
  activate    = true
}
```

### GuardDuty 検知結果の自動通知

```python
# EventBridge → Lambda → Slack 通知
import json
import os
import urllib.request
from datetime import datetime

def lambda_handler(event, context):
    """GuardDuty の検知結果を Slack に通知"""
    detail = event['detail']
    severity = detail['severity']
    finding_type = detail['type']
    description = detail['description']
    account_id = detail['accountId']
    region = detail['region']
    finding_id = detail['id']
    created_at = detail.get('createdAt', '')
    updated_at = detail.get('updatedAt', '')

    # 重大度に応じた色分け
    if severity >= 7:
        color = '#ff0000'  # 赤: High/Critical
        emoji = ':rotating_light:'
        urgency = 'CRITICAL'
    elif severity >= 4:
        color = '#ff9900'  # オレンジ: Medium
        emoji = ':warning:'
        urgency = 'MEDIUM'
    else:
        color = '#ffcc00'  # 黄: Low
        emoji = ':information_source:'
        urgency = 'LOW'

    # 影響を受けたリソースの情報を抽出
    resource = detail.get('resource', {})
    resource_type = resource.get('resourceType', 'Unknown')
    resource_details = _extract_resource_details(resource)

    # GuardDuty コンソールへのリンク
    console_url = (
        f"https://{region}.console.aws.amazon.com/guardduty/home?"
        f"region={region}#/findings?fId={finding_id}"
    )

    slack_message = {
        'attachments': [{
            'color': color,
            'title': f'{emoji} GuardDuty Alert [{urgency}]: {finding_type}',
            'title_link': console_url,
            'fields': [
                {'title': 'Account', 'value': account_id, 'short': True},
                {'title': 'Region', 'value': region, 'short': True},
                {'title': 'Severity', 'value': f'{severity:.1f}', 'short': True},
                {'title': 'Resource Type', 'value': resource_type, 'short': True},
                {'title': 'Resource Details', 'value': resource_details},
                {'title': 'Description', 'value': description[:500]},
                {'title': 'First Seen', 'value': created_at, 'short': True},
                {'title': 'Last Seen', 'value': updated_at, 'short': True},
            ],
            'footer': 'AWS GuardDuty',
            'ts': int(datetime.now().timestamp()),
        }]
    }

    webhook_url = os.environ['SLACK_WEBHOOK_URL']
    req = urllib.request.Request(
        webhook_url,
        data=json.dumps(slack_message).encode(),
        headers={'Content-Type': 'application/json'},
    )
    urllib.request.urlopen(req)

    return {'statusCode': 200, 'body': 'Notification sent'}


def _extract_resource_details(resource):
    """リソースの詳細情報を抽出"""
    resource_type = resource.get('resourceType', '')

    if resource_type == 'Instance':
        instance = resource.get('instanceDetails', {})
        return (
            f"Instance ID: {instance.get('instanceId', 'N/A')}\n"
            f"Instance Type: {instance.get('instanceType', 'N/A')}\n"
            f"Private IP: {instance.get('networkInterfaces', [{}])[0].get('privateIpAddress', 'N/A')}"
        )
    elif resource_type == 'AccessKey':
        access_key = resource.get('accessKeyDetails', {})
        return (
            f"User: {access_key.get('userName', 'N/A')}\n"
            f"User Type: {access_key.get('userType', 'N/A')}\n"
            f"Principal ID: {access_key.get('principalId', 'N/A')}"
        )
    elif resource_type == 'S3Bucket':
        s3 = resource.get('s3BucketDetails', [{}])[0]
        return f"Bucket: {s3.get('name', 'N/A')}"
    else:
        return json.dumps(resource, indent=2, default=str)[:300]
```

### GuardDuty 検知結果の自動修復

```python
# GuardDuty High Severity → 自動修復 Lambda
import boto3
import json

ec2 = boto3.client('ec2')
iam = boto3.client('iam')


def lambda_handler(event, context):
    """GuardDuty の High/Critical 検知結果に対する自動修復"""
    detail = event['detail']
    finding_type = detail['type']
    severity = detail['severity']

    if severity < 7:
        print(f"Severity {severity} is below threshold, skipping auto-remediation")
        return

    # 検知タイプに応じた修復アクション
    if 'UnauthorizedAccess:EC2' in finding_type:
        _isolate_ec2_instance(detail)
    elif 'CryptoCurrency:EC2' in finding_type:
        _isolate_ec2_instance(detail)
    elif 'UnauthorizedAccess:IAMUser/InstanceCredentialExfiltration' in finding_type:
        _revoke_iam_sessions(detail)
    elif 'Exfiltration:S3' in finding_type:
        _restrict_s3_bucket(detail)

    return {'statusCode': 200, 'body': 'Remediation completed'}


def _isolate_ec2_instance(detail):
    """EC2 インスタンスをネットワーク隔離"""
    instance_id = detail['resource']['instanceDetails']['instanceId']
    vpc_id = detail['resource']['instanceDetails']['networkInterfaces'][0]['vpcId']

    # 隔離用 Security Group を作成（インバウンド・アウトバウンド全拒否）
    isolation_sg = ec2.create_security_group(
        GroupName=f'isolation-{instance_id}',
        Description=f'Isolation SG for compromised instance {instance_id}',
        VpcId=vpc_id,
    )

    # デフォルトのアウトバウンドルールを削除
    ec2.revoke_security_group_egress(
        GroupId=isolation_sg['GroupId'],
        IpPermissions=[{
            'IpProtocol': '-1',
            'IpRanges': [{'CidrIp': '0.0.0.0/0'}],
        }]
    )

    # インスタンスの SG を隔離用に変更
    ec2.modify_instance_attribute(
        InstanceId=instance_id,
        Groups=[isolation_sg['GroupId']],
    )

    print(f"Instance {instance_id} isolated with SG {isolation_sg['GroupId']}")


def _revoke_iam_sessions(detail):
    """IAM ユーザの全セッションを無効化"""
    access_key = detail['resource']['accessKeyDetails']
    user_name = access_key.get('userName', '')

    if not user_name or user_name == 'N/A':
        print("Cannot determine IAM user, skipping session revocation")
        return

    # インラインポリシーで全操作を拒否
    deny_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Deny",
            "Action": "*",
            "Resource": "*",
            "Condition": {
                "DateLessThan": {
                    "aws:TokenIssueTime": detail['updatedAt']
                }
            }
        }]
    }

    iam.put_user_policy(
        UserName=user_name,
        PolicyName='GuardDuty-DenySessions',
        PolicyDocument=json.dumps(deny_policy),
    )

    print(f"Revoked sessions for user {user_name}")


def _restrict_s3_bucket(detail):
    """S3 バケットのパブリックアクセスをブロック"""
    s3 = boto3.client('s3')
    bucket_name = detail['resource']['s3BucketDetails'][0]['name']

    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            'BlockPublicAcls': True,
            'IgnorePublicAcls': True,
            'BlockPublicPolicy': True,
            'RestrictPublicBuckets': True,
        }
    )

    print(f"Public access blocked for bucket {bucket_name}")
```

### GuardDuty のフィルタリングとサプレッション

```hcl
# GuardDuty フィルタ（特定の検知結果をフィルタリング）
resource "aws_guardduty_filter" "high_severity" {
  detector_id = aws_guardduty_detector.main.id
  name        = "high-severity-findings"
  action      = "NOOP"
  rank        = 1

  finding_criteria {
    criterion {
      field  = "severity"
      gte    = "7"
    }
    criterion {
      field   = "type"
      not_equals = ["Recon:EC2/Portscan"]  # ポートスキャンは除外
    }
  }
}

# サプレッションルール（誤検知のアーカイブ）
resource "aws_guardduty_filter" "suppress_internal_portscans" {
  detector_id = aws_guardduty_detector.main.id
  name        = "suppress-internal-portscans"
  action      = "ARCHIVE"
  rank        = 2

  finding_criteria {
    criterion {
      field  = "type"
      equals = ["Recon:EC2/Portscan"]
    }
    criterion {
      field  = "resource.instanceDetails.networkInterfaces.privateIpAddress"
      equals = ["10.0.0.0/8"]  # 内部ネットワークからのポートスキャン
    }
  }
}
```

---

## 3. Security Hub

### Security Hub の構成

Security Hub は AWS セキュリティサービスの検知結果を一元管理するサービスである。複数のセキュリティ基準に対するコンプライアンスチェックを自動的に実行し、スコアカードで可視化する。

```
+------------------------------------------------------------------+
|                    Security Hub                                    |
|------------------------------------------------------------------|
|                                                                  |
|  [データソース (Findings)]                                        |
|  +-- GuardDuty の検知結果                                        |
|  +-- Inspector の脆弱性                                          |
|  +-- Macie の機密データ検出                                       |
|  +-- IAM Access Analyzer                                        |
|  +-- Firewall Manager                                           |
|  +-- Config Rules の非準拠                                       |
|  +-- サードパーティ (Prowler, Checkov, Snyk)                     |
|                                                                  |
|  [セキュリティ基準 (Standards)]                                   |
|  +-- AWS Foundational Security Best Practices (FSBP)            |
|  +-- CIS AWS Foundations Benchmark v1.4.0 / v3.0.0             |
|  +-- PCI DSS v3.2.1                                            |
|  +-- NIST SP 800-53 Rev. 5                                      |
|                                                                  |
|  [出力]                                                          |
|  +-- ダッシュボード (スコアカード)                                 |
|  +-- EventBridge 統合 (自動修復)                                 |
|  +-- カスタムアクション                                          |
|  +-- AWS Security Lake 連携                                     |
+------------------------------------------------------------------+
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

# CIS AWS Foundations Benchmark v1.4.0
resource "aws_securityhub_standards_subscription" "cis_1_4" {
  standards_arn = "arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark/v/1.4.0"
  depends_on    = [aws_securityhub_account.main]
}

# CIS AWS Foundations Benchmark v3.0.0
resource "aws_securityhub_standards_subscription" "cis_3_0" {
  standards_arn = "arn:aws:securityhub:ap-northeast-1::standards/cis-aws-foundations-benchmark/v/3.0.0"
  depends_on    = [aws_securityhub_account.main]
}

# NIST SP 800-53 Rev. 5
resource "aws_securityhub_standards_subscription" "nist" {
  standards_arn = "arn:aws:securityhub:ap-northeast-1::standards/nist-800-53/v/5.0.0"
  depends_on    = [aws_securityhub_account.main]
}

# 特定のコントロールを無効化（正当な理由がある場合）
resource "aws_securityhub_standards_control" "disable_s3_logging" {
  standards_control_arn = "arn:aws:securityhub:ap-northeast-1:123456789012:control/aws-foundational-security-best-practices/v/1.0.0/S3.9"
  control_status        = "DISABLED"
  disabled_reason       = "S3 server access logging is replaced by CloudTrail S3 data events"
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
        Workflow = {
          Status = ["NEW"]
        }
        RecordState = ["ACTIVE"]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "securityhub_lambda" {
  rule      = aws_cloudwatch_event_rule.securityhub_high.name
  target_id = "securityhub-auto-remediate"
  arn       = aws_lambda_function.auto_remediate.arn
}

# Organizations 統合
resource "aws_securityhub_organization_configuration" "main" {
  auto_enable           = true
  auto_enable_standards = "DEFAULT"
}
```

### Security Hub カスタムアクション

```python
# Security Hub のカスタムアクション（手動トリガーの修復）
import boto3

securityhub = boto3.client('securityhub')

# カスタムアクションの作成
securityhub.create_action_target(
    Name='IsolateInstance',
    Description='Isolate a compromised EC2 instance',
    Id='IsolateInstance',
)

# Finding の更新（ワークフローステータスの変更）
securityhub.batch_update_findings(
    FindingIdentifiers=[
        {
            'Id': 'arn:aws:securityhub:ap-northeast-1:123456:finding/xxx',
            'ProductArn': 'arn:aws:securityhub:ap-northeast-1::product/aws/guardduty',
        }
    ],
    Workflow={'Status': 'RESOLVED'},
    Note={
        'Text': 'Investigated and resolved. Instance was isolated and reimaged.',
        'UpdatedBy': 'security-team',
    },
)
```

### Security Hub スコアの改善戦略

```
┌──────────── Security Hub スコア改善戦略 ─────────────┐
│                                                       │
│  Phase 1: CRITICAL の解消 (目標: 0件)                 │
│  ├── S3 パブリックアクセス                             │
│  ├── RDS パブリックアクセス                            │
│  ├── root アカウントの MFA                            │
│  └── IAM ポリシーのワイルドカード                      │
│                                                       │
│  Phase 2: HIGH の解消 (目標: スコア 90%+)             │
│  ├── 暗号化の有効化 (S3, RDS, EBS)                    │
│  ├── ログの有効化 (CloudTrail, VPC Flow Logs)         │
│  ├── バックアップの設定                                │
│  └── セキュリティグループの見直し                      │
│                                                       │
│  Phase 3: MEDIUM の対応 (目標: スコア 95%+)           │
│  ├── タグ付け規則の適用                                │
│  ├── VPC エンドポイントの設定                         │
│  ├── 不要なリソースの削除                              │
│  └── ログ保持期間の設定                                │
│                                                       │
│  Phase 4: 継続的改善                                   │
│  ├── 新規リソースの自動チェック (Config Rules)         │
│  ├── 週次でのスコアレビュー                            │
│  ├── 例外管理プロセスの運用                            │
│  └── 自動修復の拡充                                   │
│                                                       │
└───────────────────────────────────────────────────────┘
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
  include_global_service_events = true
  kms_key_id                    = aws_kms_key.cloudtrail.arn

  # 管理イベント
  event_selector {
    read_write_type           = "All"
    include_management_events = true
  }

  # S3 データイベント (機密バケット)
  event_selector {
    read_write_type           = "WriteOnly"
    include_management_events = false

    data_resource {
      type   = "AWS::S3::Object"
      values = ["arn:aws:s3:::sensitive-bucket/"]
    }
  }

  # Lambda データイベント
  event_selector {
    read_write_type           = "All"
    include_management_events = false

    data_resource {
      type   = "AWS::Lambda::Function"
      values = ["arn:aws:lambda"]
    }
  }

  # CloudWatch Logs に転送
  cloud_watch_logs_group_arn = "${aws_cloudwatch_log_group.cloudtrail.arn}:*"
  cloud_watch_logs_role_arn  = aws_iam_role.cloudtrail_cloudwatch.arn

  # Insights イベント (異常検知)
  insight_selector {
    insight_type = "ApiCallRateInsight"
  }
  insight_selector {
    insight_type = "ApiErrorRateInsight"
  }

  tags = {
    Environment = "production"
    Security    = "critical"
  }
}

# CloudTrail ログ用 S3 バケット
resource "aws_s3_bucket" "cloudtrail" {
  bucket        = "org-cloudtrail-logs-${data.aws_caller_identity.current.account_id}"
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id
  versioning_configuration {
    status    = "Enabled"
    mfa_delete = "Enabled"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cloudtrail" {
  bucket = aws_s3_bucket.cloudtrail.id

  rule {
    id     = "archive-and-expire"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }

    expiration {
      days = 2555  # 7年保持（規制要件に応じて調整）
    }
  }
}

# CloudWatch メトリクスフィルターとアラーム
resource "aws_cloudwatch_log_metric_filter" "unauthorized_api_calls" {
  name           = "UnauthorizedAPICalls"
  pattern        = "{ ($.errorCode = \"*UnauthorizedAccess*\") || ($.errorCode = \"AccessDenied*\") }"
  log_group_name = aws_cloudwatch_log_group.cloudtrail.name

  metric_transformation {
    name          = "UnauthorizedAPICalls"
    namespace     = "CloudTrailMetrics"
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_metric_alarm" "unauthorized_api_calls" {
  alarm_name          = "UnauthorizedAPICalls"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "UnauthorizedAPICalls"
  namespace           = "CloudTrailMetrics"
  period              = 300
  statistic           = "Sum"
  threshold           = 5
  alarm_description   = "Multiple unauthorized API calls detected"
  alarm_actions       = [aws_sns_topic.security_alerts.arn]
}
```

### CloudTrail ログの分析

```python
import boto3
from datetime import datetime, timedelta

# CloudTrail Insights / Athena での分析
athena = boto3.client('athena')

# Athena テーブルの作成 (初回のみ)
create_table_query = """
CREATE EXTERNAL TABLE IF NOT EXISTS cloudtrail_logs (
    eventVersion STRING,
    userIdentity STRUCT<
        type: STRING,
        principalId: STRING,
        arn: STRING,
        accountId: STRING,
        invokedBy: STRING,
        accessKeyId: STRING,
        userName: STRING,
        sessionContext: STRUCT<
            attributes: STRUCT<
                mfaAuthenticated: STRING,
                creationDate: STRING>,
            sessionIssuer: STRUCT<
                type: STRING,
                principalId: STRING,
                arn: STRING,
                accountId: STRING,
                userName: STRING>>>,
    eventTime STRING,
    eventSource STRING,
    eventName STRING,
    awsRegion STRING,
    sourceIPAddress STRING,
    userAgent STRING,
    errorCode STRING,
    errorMessage STRING,
    requestParameters STRING,
    responseElements STRING,
    additionalEventData STRING,
    requestId STRING,
    eventId STRING,
    resources ARRAY<STRUCT<
        arn: STRING,
        accountId: STRING,
        type: STRING>>,
    eventType STRING,
    recipientAccountId STRING
)
PARTITIONED BY (region STRING, year STRING, month STRING, day STRING)
ROW FORMAT SERDE 'org.apache.hive.hcatalog.data.JsonSerDe'
LOCATION 's3://org-cloudtrail-logs-123456789012/AWSLogs/123456789012/CloudTrail/'
"""

# 疑わしい API 呼び出しの検索
suspicious_activity_query = """
SELECT
    eventTime,
    eventName,
    userIdentity.arn as userArn,
    userIdentity.type as identityType,
    sourceIPAddress,
    errorCode,
    awsRegion,
    requestParameters
FROM cloudtrail_logs
WHERE year = '2024' AND month = '01'
  AND (
    -- セキュリティサービスの無効化
    eventName IN ('DeleteTrail', 'StopLogging', 'DeleteFlowLogs',
                  'DeleteDetector', 'DisableSecurityHub')
    -- 疑わしい IAM 操作
    OR eventName IN ('CreateUser', 'CreateAccessKey', 'AttachUserPolicy',
                     'PutUserPolicy', 'CreateLoginProfile')
    -- ログイン失敗
    OR (eventName = 'ConsoleLogin' AND errorCode = 'Failed')
    -- データ流出の兆候
    OR eventName IN ('GetObject', 'CopyObject') AND sourceIPAddress NOT LIKE '10.%'
  )
ORDER BY eventTime DESC
LIMIT 100;
"""

# 特定ユーザの全活動を追跡
user_activity_query = """
SELECT
    eventTime,
    eventName,
    eventSource,
    sourceIPAddress,
    userAgent,
    errorCode
FROM cloudtrail_logs
WHERE year = '2024' AND month = '01'
  AND userIdentity.arn LIKE '%suspicious-user%'
ORDER BY eventTime ASC
LIMIT 1000;
"""

# 未使用のアクセスキーの特定
unused_access_keys_query = """
SELECT
    userIdentity.accessKeyId as accessKeyId,
    userIdentity.userName as userName,
    MAX(eventTime) as lastUsed,
    COUNT(*) as apiCallCount
FROM cloudtrail_logs
WHERE year = '2024'
  AND userIdentity.type = 'IAMUser'
  AND userIdentity.accessKeyId IS NOT NULL
GROUP BY userIdentity.accessKeyId, userIdentity.userName
HAVING MAX(eventTime) < date_format(date_add('day', -90, now()), '%Y-%m-%dT%H:%i:%sZ')
ORDER BY lastUsed ASC;
"""
```

### AWS Config Rules

```hcl
# AWS Config の有効化
resource "aws_config_configuration_recorder" "main" {
  name     = "default"
  role_arn = aws_iam_role.config.arn

  recording_group {
    all_supported                 = true
    include_global_resource_types = true
  }
}

resource "aws_config_delivery_channel" "main" {
  name           = "default"
  s3_bucket_name = aws_s3_bucket.config.id
  sns_topic_arn  = aws_sns_topic.config.arn

  snapshot_delivery_properties {
    delivery_frequency = "TwentyFour_Hours"
  }
}

# マネージドルール: S3 パブリックアクセス禁止
resource "aws_config_config_rule" "s3_public_read" {
  name = "s3-bucket-public-read-prohibited"
  source {
    owner             = "AWS"
    source_identifier = "S3_BUCKET_PUBLIC_READ_PROHIBITED"
  }
  depends_on = [aws_config_configuration_recorder.main]
}

# マネージドルール: RDS 暗号化
resource "aws_config_config_rule" "rds_encrypted" {
  name = "rds-storage-encrypted"
  source {
    owner             = "AWS"
    source_identifier = "RDS_STORAGE_ENCRYPTED"
  }
}

# マネージドルール: CloudTrail 有効化
resource "aws_config_config_rule" "cloudtrail_enabled" {
  name = "cloudtrail-enabled"
  source {
    owner             = "AWS"
    source_identifier = "CLOUD_TRAIL_ENABLED"
  }
}

# マネージドルール: MFA 有効化
resource "aws_config_config_rule" "iam_root_mfa" {
  name = "root-account-mfa-enabled"
  source {
    owner             = "AWS"
    source_identifier = "ROOT_ACCOUNT_MFA_ENABLED"
  }
}

# カスタムルール: EC2 に必須タグがあるか
resource "aws_config_config_rule" "required_tags" {
  name = "required-tags"
  source {
    owner             = "AWS"
    source_identifier = "REQUIRED_TAGS"
  }
  input_parameters = jsonencode({
    tag1Key   = "Environment"
    tag2Key   = "Team"
    tag3Key   = "CostCenter"
  })
  scope {
    compliance_resource_types = [
      "AWS::EC2::Instance",
      "AWS::RDS::DBInstance",
      "AWS::S3::Bucket",
    ]
  }
}

# SSM Automation による自動修復
resource "aws_config_remediation_configuration" "s3_public_access" {
  config_rule_name = aws_config_config_rule.s3_public_read.name
  target_type      = "SSM_DOCUMENT"
  target_id        = "AWS-DisableS3BucketPublicReadWrite"

  parameter {
    name           = "S3BucketName"
    resource_value = "RESOURCE_ID"
  }

  parameter {
    name         = "AutomationAssumeRole"
    static_value = aws_iam_role.config_remediation.arn
  }

  automatic                  = true
  maximum_automatic_attempts = 3
  retry_attempt_seconds      = 60
}
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
    Description='Production database credentials for myapp',
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
        {'Key': 'Application', 'Value': 'myapp'},
    ],
)

# シークレットの取得（キャッシュ推奨）
from aws_secretsmanager_caching import SecretCache, SecretCacheConfig

cache_config = SecretCacheConfig(
    max_cache_size=100,
    exception_retry_delay_base=1,
    exception_retry_growth_factor=2,
    exception_retry_delay_max=3600,
    default_secret_version_stage='AWSCURRENT',
    secret_refresh_interval=3600,  # 1時間キャッシュ
    secret_version_stage_refresh_interval=3600,
)

cache = SecretCache(config=cache_config)

def get_db_credentials():
    """キャッシュ経由でDB認証情報を取得"""
    secret_string = cache.get_secret_string('myapp/database/credentials')
    return json.loads(secret_string)


# 自動ローテーションの設定
sm.rotate_secret(
    SecretId='myapp/database/credentials',
    RotationLambdaARN='arn:aws:lambda:ap-northeast-1:123456:function:rotate-db-secret',
    RotationRules={
        'AutomaticallyAfterDays': 30,
        'Duration': '2h',
        'ScheduleExpression': 'rate(30 days)',
    },
)
```

### ローテーション Lambda

```python
# Lambda: RDS シークレットのローテーション
import boto3
import json
import logging
import os
import psycopg2

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm = boto3.client('secretsmanager')


def lambda_handler(event, context):
    """Secrets Manager のローテーションハンドラ"""
    secret_arn = event['SecretId']
    token = event['ClientRequestToken']
    step = event['Step']

    if step == 'createSecret':
        create_secret(secret_arn, token)
    elif step == 'setSecret':
        set_secret(secret_arn, token)
    elif step == 'testSecret':
        test_secret(secret_arn, token)
    elif step == 'finishSecret':
        finish_secret(secret_arn, token)


def create_secret(secret_arn, token):
    """新しいパスワードを生成"""
    current = sm.get_secret_value(SecretId=secret_arn, VersionStage='AWSCURRENT')
    current_dict = json.loads(current['SecretString'])

    # 新しいパスワードを生成
    new_password = sm.get_random_password(
        PasswordLength=32,
        ExcludeCharacters='/@"\'\\',
        RequireEachIncludedType=True,
    )['RandomPassword']

    current_dict['password'] = new_password
    sm.put_secret_value(
        SecretId=secret_arn,
        ClientRequestToken=token,
        SecretString=json.dumps(current_dict),
        VersionStages=['AWSPENDING'],
    )


def set_secret(secret_arn, token):
    """データベースのパスワードを変更"""
    pending = sm.get_secret_value(SecretId=secret_arn, VersionStage='AWSPENDING')
    pending_dict = json.loads(pending['SecretString'])

    current = sm.get_secret_value(SecretId=secret_arn, VersionStage='AWSCURRENT')
    current_dict = json.loads(current['SecretString'])

    conn = psycopg2.connect(
        host=current_dict['host'],
        port=current_dict['port'],
        dbname=current_dict['dbname'],
        user=current_dict['username'],
        password=current_dict['password'],
    )
    conn.autocommit = True
    cursor = conn.cursor()
    cursor.execute(
        f"ALTER USER {current_dict['username']} WITH PASSWORD %s",
        (pending_dict['password'],)
    )
    conn.close()


def test_secret(secret_arn, token):
    """新しい認証情報でDB接続をテスト"""
    pending = sm.get_secret_value(SecretId=secret_arn, VersionStage='AWSPENDING')
    pending_dict = json.loads(pending['SecretString'])

    conn = psycopg2.connect(
        host=pending_dict['host'],
        port=pending_dict['port'],
        dbname=pending_dict['dbname'],
        user=pending_dict['username'],
        password=pending_dict['password'],
    )
    conn.close()
    logger.info("Successfully tested new credentials")


def finish_secret(secret_arn, token):
    """バージョンラベルを切り替え"""
    metadata = sm.describe_secret(SecretId=secret_arn)
    current_version = None

    for version_id, stages in metadata['VersionIdsToStages'].items():
        if 'AWSCURRENT' in stages:
            current_version = version_id
            break

    sm.update_secret_version_stage(
        SecretId=secret_arn,
        VersionStage='AWSCURRENT',
        MoveToVersionId=token,
        RemoveFromVersionId=current_version,
    )
```

---

## 6. WAF と Shield

### AWS WAF の設定

```hcl
# WAF Web ACL
resource "aws_wafv2_web_acl" "main" {
  name        = "main-web-acl"
  description = "Main WAF Web ACL for production"
  scope       = "REGIONAL"

  default_action {
    allow {}
  }

  # AWS マネージドルール: 一般的な攻撃パターン
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1
    override_action { none {} }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesCommonRuleSet"
      sampled_requests_enabled   = true
    }
  }

  # AWS マネージドルール: SQLi 対策
  rule {
    name     = "AWSManagedRulesSQLiRuleSet"
    priority = 2
    override_action { none {} }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesSQLiRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "AWSManagedRulesSQLiRuleSet"
      sampled_requests_enabled   = true
    }
  }

  # レートベースルール: DDoS 対策
  rule {
    name     = "RateLimit"
    priority = 3

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimit"
      sampled_requests_enabled   = true
    }
  }

  # 地理的制限
  rule {
    name     = "GeoRestriction"
    priority = 4

    action {
      block {}
    }

    statement {
      not_statement {
        statement {
          geo_match_statement {
            country_codes = ["JP", "US"]
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GeoRestriction"
      sampled_requests_enabled   = true
    }
  }

  # IP ブラックリスト
  rule {
    name     = "IPBlacklist"
    priority = 0

    action {
      block {}
    }

    statement {
      ip_set_reference_statement {
        arn = aws_wafv2_ip_set.blacklist.arn
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "IPBlacklist"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "MainWebACL"
    sampled_requests_enabled   = true
  }
}

# ALB との関連付け
resource "aws_wafv2_web_acl_association" "alb" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}

# WAF ログの有効化
resource "aws_wafv2_web_acl_logging_configuration" "main" {
  log_destination_configs = [aws_kinesis_firehose_delivery_stream.waf_logs.arn]
  resource_arn            = aws_wafv2_web_acl.main.arn

  logging_filter {
    default_behavior = "DROP"

    filter {
      behavior    = "KEEP"
      condition {
        action_condition {
          action = "BLOCK"
        }
      }
      requirement = "MEETS_ANY"
    }
  }
}
```

---

## 7. IAM Access Analyzer

```hcl
# IAM Access Analyzer の有効化
resource "aws_accessanalyzer_analyzer" "main" {
  analyzer_name = "account-analyzer"
  type          = "ACCOUNT"

  tags = {
    Environment = "production"
  }
}

# Organizations レベルのアナライザー
resource "aws_accessanalyzer_analyzer" "org" {
  analyzer_name = "organization-analyzer"
  type          = "ORGANIZATION"
}
```

```python
# IAM Access Analyzer の Finding を取得・分析
import boto3

analyzer = boto3.client('accessanalyzer')

# 外部アクセスの Finding を取得
findings = analyzer.list_findings(
    analyzerArn='arn:aws:access-analyzer:ap-northeast-1:123456:analyzer/account-analyzer',
    filter={
        'status': {'eq': ['ACTIVE']},
        'resourceType': {'eq': ['AWS::S3::Bucket']},
    },
)

for finding in findings['findings']:
    print(f"Resource: {finding['resource']}")
    print(f"  Principal: {finding['principal']}")
    print(f"  Action: {finding['action']}")
    print(f"  Condition: {finding['condition']}")
    print(f"  Status: {finding['status']}")
    print()

# 未使用アクセスの分析
unused_access = analyzer.list_findings(
    analyzerArn='arn:aws:access-analyzer:ap-northeast-1:123456:analyzer/account-analyzer',
    filter={
        'findingType': {'eq': ['UnusedIAMRole', 'UnusedIAMUserAccessKey', 'UnusedIAMUserPassword']},
    },
)
```

---

## 8. エッジケース

### エッジケース 1: クロスアカウントアクセスの見落とし

リソースベースポリシー（S3バケットポリシー、KMSキーポリシーなど）で他のアカウントにアクセスを許可している場合、IAM Access Analyzer はこれを外部アクセスとして検出する。ただし、Organizations 内のアカウントからのアクセスと外部アカウントからのアクセスの区別を正しく設定しないと、正当なクロスアカウントアクセスが過剰にアラートを生成する。

### エッジケース 2: GuardDuty の検知遅延

GuardDuty はリアルタイムではなく、データソースの更新頻度に依存する。VPC Flow Logs は約10分遅延、CloudTrail は通常15分以内に配信される。そのため、攻撃の検知から通知までに最大30分程度のラグが発生する可能性がある。

### エッジケース 3: Secrets Manager のローテーション失敗

ローテーション Lambda がタイムアウトした場合、シークレットは AWSPENDING ステータスのまま残る。この状態でアプリケーションが AWSCURRENT を参照すると、古い認証情報を使い続ける。ローテーション Lambda には適切なタイムアウト（最低5分）と、VPC 内で実行する場合は NAT Gateway 経由の Secrets Manager エンドポイントへのアクセスが必要である。

---

## 9. アンチパターン

### アンチパターン 1: CloudTrail の無効化

```
NG:
  → CloudTrail を無効にしてコスト削減
  → 単一リージョンのみ有効
  → ログファイルの整合性検証を無効化
  → ログをアプリケーションアカウントに保存

OK:
  → 全リージョン + 全イベントを記録
  → ログファイル検証を有効化
  → S3 バケットを KMS で暗号化し MFA Delete を有効化
  → ログアーカイブ用の別アカウントに保存
  → CloudWatch Logs にも転送してリアルタイム分析
```

### アンチパターン 2: アクセスキーの長期使用

```bash
# NG: アクセスキーを作成して永久に使い続ける
aws iam create-access-key --user-name deploy-user
# → 1年以上ローテーションされないアクセスキー
# → .env ファイルやスクリプトにハードコード
# → 複数人で同じアクセスキーを共有

# OK: IAM ロールと一時認証情報を使用
# EC2: インスタンスプロファイル
# Lambda: 実行ロール
# ECS: タスクロール
# CI/CD: OIDC プロバイダ + AssumeRoleWithWebIdentity

# GitHub Actions の例:
# - uses: aws-actions/configure-aws-credentials@v4
#   with:
#     role-to-assume: arn:aws:iam::123456:role/github-actions-role
#     aws-region: ap-northeast-1
```

### アンチパターン 3: セキュリティサービスの部分的な有効化

```
NG:
  → GuardDuty を一部のリージョンでのみ有効化
  → Security Hub を有効にしたが Standards を有効化していない
  → Config を有効にしたが Config Rules を設定していない
  → マネジメントアカウントのみで有効化し、メンバーアカウントは未対応

OK:
  → 全リージョン・全アカウントで有効化
  → Organizations の delegated administrator で一元管理
  → 全 Standards を有効化し、例外は正当な理由付きで管理
  → 自動修復を設定して検知結果に対応
```

---

## 10. 演習問題

### 演習 1: GuardDuty + EventBridge + Lambda による自動通知

以下を実装せよ。
1. GuardDuty を有効化（全データソース）
2. EventBridge ルールで High 以上の検知結果をフィルタ
3. Lambda で Slack に通知（重大度に応じた色分け）
4. Terraform でインフラを構築

### 演習 2: CloudTrail ログの Athena 分析

以下のクエリを作成せよ。
1. 過去24時間の全 IAM ユーザのログイン試行（成功/失敗）
2. root アカウントの全操作
3. Security Group の変更履歴
4. 特定の IP アドレスからの全 API 呼び出し

### 演習 3: Security Hub スコア改善計画

自分の AWS アカウントで以下を実施せよ。
1. Security Hub を有効化し、AWS FSBP を有効化
2. 現在のスコアを確認
3. CRITICAL の Finding を全て解消
4. HIGH の Finding の修復計画を作成

---

## 11. FAQ

### Q1. GuardDuty の検知結果が多すぎる場合はどうするか?

GuardDuty の Suppression Rules で低リスクの検知結果を自動アーカイブできる。まず全検知結果を確認して誤検知パターンを特定し、Filter で自動アーカイブする。ただし、High 以上の検知結果は抑制せず必ず調査すること。Trusted IP リストにオフィスの IP を登録することで、正常なトラフィックからの誤検知を減らせる。

### Q2. Security Hub のスコアをどこまで上げるべきか?

100% は現実的でない場合もある。CRITICAL と HIGH の検知結果は全て対応し、全体スコアは 90% 以上を目標とする。例外が必要な場合は Suppression ルールを適用し、その理由をドキュメント化する。スコアの推移を週次でモニタリングし、低下傾向がある場合は原因を調査する。

### Q3. CloudTrail のコストを最適化するには?

管理イベントは基本的に全記録する (コスト影響は小さい)。データイベント (S3/Lambda) は機密バケットに限定する。CloudTrail Lake の代わりに Athena + S3 で分析することでコストを抑えられる。ログの保持期間を規制要件に合わせて設定し、不要に長期保存しない。GLACIER/DEEP_ARCHIVE へのライフサイクルルールでストレージコストを削減する。

### Q4. Secrets Manager と Parameter Store のどちらを使うべきか?

自動ローテーションが必要な認証情報（データベースパスワード、API キー）には Secrets Manager を使用する。設定値やフラグなどの非機密パラメータには Parameter Store (Standard) が無料で適切。Parameter Store の SecureString も暗号化されるが、自動ローテーション機能はない。コスト面では Parameter Store が有利（無料枠あり）、機能面では Secrets Manager が優位。

### Q5. WAF のルール設定をどう最適化するか?

まず AWS マネージドルールセットを有効にする（CommonRuleSet, SQLiRuleSet, XSSRuleSet）。初期は Count モードで誤検知を確認し、問題ないことを確認してから Block モードに切り替える。レートベースルールで DDoS を緩和し、地理的制限で不要な国からのアクセスをブロックする。WAF ログを分析して、ブロックされたリクエストのパターンを定期的に確認する。

### Q6. マルチアカウント環境でセキュリティサービスをどう管理するか?

Organizations の delegated administrator 機能を使用して、セキュリティアカウントから全メンバーアカウントのセキュリティサービスを一元管理する。GuardDuty、Security Hub、Config、IAM Access Analyzer はいずれも delegated administrator をサポートしている。Control Tower を使用すると、新規アカウント作成時にセキュリティサービスが自動的に有効化される。

---

## まとめ

| 項目 | 要点 |
|------|------|
| GuardDuty | 全アカウント・全リージョンで有効化、EventBridge で自動通知・修復 |
| Security Hub | 検知結果を集約、CIS/AWS ベストプラクティスで評価、スコア 90%+ を目標 |
| CloudTrail | 全 API 呼び出しを記録、ログ検証必須、別アカウントに保存、Athena で分析 |
| Config | リソース構成の継続監視、ルール違反を自動検出・修復 |
| IAM | ロールベースアクセス、一時認証情報、MFA 必須、Access Analyzer で分析 |
| Secrets Manager | シークレットの一元管理と自動ローテーション、キャッシュ利用推奨 |
| KMS | データ暗号化の鍵管理、エンベロープ暗号化、キーポリシーの最小権限 |
| WAF | マネージドルール + カスタムルール、レートベース制限、Count → Block 段階的適用 |

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
5. **AWS CloudTrail User Guide** — https://docs.aws.amazon.com/awscloudtrail/latest/userguide/
6. **AWS WAF Developer Guide** — https://docs.aws.amazon.com/waf/latest/developerguide/
7. **AWS Config Developer Guide** — https://docs.aws.amazon.com/config/latest/developerguide/
8. **AWS Secrets Manager User Guide** — https://docs.aws.amazon.com/secretsmanager/latest/userguide/
9. **AWS Well-Architected Framework — Security Pillar** — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
