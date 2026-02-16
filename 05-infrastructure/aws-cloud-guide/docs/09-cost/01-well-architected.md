# Well-Architected Framework — 6 つの柱とレビュープロセス

> AWS Well-Architected Framework の 6 つの柱を理解し、自社ワークロードを体系的にレビュー・改善するための実践ガイド。

---

## この章で学ぶこと

1. **6 つの柱** それぞれの設計原則とベストプラクティス
2. **Well-Architected Tool** を使ったワークロードレビューの進め方
3. **改善の優先順位付け** と継続的なアーキテクチャ改善プロセス
4. **Trusted Advisor** との連携と自動チェック
5. **CDK / CloudFormation** による Well-Architected 準拠の自動化

---

## 1. Well-Architected Framework の全体像

### 1.1 6 つの柱

```
┌──────────────────────────────────────────────────────────┐
│            AWS Well-Architected Framework                 │
│                   6 つの柱 (Pillars)                      │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ 1. 運用上の  │  │ 2. セキュリ │  │ 3. 信頼性    │   │
│  │   優秀性     │  │   ティ      │  │ (Reliability)│   │
│  │ (Operational │  │ (Security)  │  │              │   │
│  │  Excellence) │  │             │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ 4. パフォー  │  │ 5. コスト   │  │ 6. 持続可    │   │
│  │  マンス効率  │  │   最適化    │  │   能性       │   │
│  │ (Performance │  │ (Cost       │  │(Sustainability│  │
│  │  Efficiency) │  │ Optimization│  │             ) │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### 1.2 各柱の関係性

```
                   ┌─────────────────┐
                   │   ビジネス価値   │
                   └────────┬────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │ セキュリティ│   │  信頼性   │   │ パフォー  │
     │ (基盤)     │   │ (基盤)    │   │  マンス   │
     └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
     ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
     │ 運用上の   │   │  コスト   │   │ 持続可    │
     │ 優秀性     │   │  最適化   │   │ 能性      │
     │ (横断)     │   │ (最適化)  │   │ (最適化)  │
     └───────────┘   └───────────┘   └───────────┘
```

### 1.3 Framework の適用タイミング

Well-Architected Framework はワークロードのライフサイクル全体で適用する。

```
┌─────────────────────────────────────────────────────────────────┐
│  ワークロードライフサイクルと Well-Architected                     │
│                                                                 │
│  設計フェーズ ──→ 実装フェーズ ──→ 運用フェーズ ──→ 最適化フェーズ  │
│       │               │               │                │        │
│       ▼               ▼               ▼                ▼        │
│  ┌─────────┐   ┌─────────┐   ┌─────────────┐  ┌──────────┐    │
│  │設計レビュー│  │実装レビュー│  │定期レビュー  │  │改善レビュー│   │
│  │(初回)    │   │(ローンチ前)│  │(四半期ごと) │  │(変更後)  │    │
│  └─────────┘   └─────────┘   └─────────────┘  └──────────┘    │
│                                                                 │
│  ポイント:                                                       │
│  - 設計時: アーキテクチャ方針の妥当性検証                          │
│  - ローンチ前: 本番稼働に向けた最終チェック                        │
│  - 運用中: ドリフト検出と継続改善                                  │
│  - 変更後: 大きなアーキテクチャ変更のインパクト評価                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 Well-Architected Tool の基本操作（CLI）

```bash
# ワークロード一覧の取得
aws wellarchitected list-workloads \
  --query 'WorkloadSummaries[*].{Name:WorkloadName,Id:WorkloadId,Risk:RiskCounts}'

# ワークロードの作成
aws wellarchitected create-workload \
  --workload-name "MyApp-Production" \
  --description "Production e-commerce application" \
  --environment PRODUCTION \
  --review-owner "architect@example.com" \
  --lenses wellarchitected serverless \
  --aws-regions ap-northeast-1 \
  --tags Project=myapp,Environment=production

# 特定の柱の質問一覧を取得
aws wellarchitected list-answers \
  --workload-id "abc123def456" \
  --lens-alias wellarchitected \
  --pillar-id operationalExcellence \
  --query 'AnswerSummaries[*].{Q:QuestionTitle,Risk:Risk}'

# 質問への回答を更新
aws wellarchitected update-answer \
  --workload-id "abc123def456" \
  --lens-alias wellarchitected \
  --question-id "ops-how-do-you-design-workload" \
  --selected-choices "ops_ops-how-do-you-design-workload_1" "ops_ops-how-do-you-design-workload_2" \
  --notes "IaC with CDK, CI/CD with CodePipeline"

# マイルストーンの作成
aws wellarchitected create-milestone \
  --workload-id "abc123def456" \
  --milestone-name "Q1-2025-Review"

# レビュー結果のレポート取得
aws wellarchitected get-lens-review-report \
  --workload-id "abc123def456" \
  --lens-alias wellarchitected \
  --query 'LensReviewReport.Base64String' \
  --output text | base64 --decode > wa-report.pdf
```

---

## 2. 6 つの柱の詳細

### 2.1 柱 1: 運用上の優秀性（Operational Excellence）

```yaml
# 設計原則
principles:
  - 運用をコードとして管理 (IaC)
  - 小さく可逆的な変更を頻繁に行う
  - 運用手順を頻繁に改善する
  - 障害を予測する
  - 全ての運用上の障害から学ぶ

# チェックリスト例
checklist:
  organization:
    - チームの責任範囲が明確か
    - 運用の優先順位がビジネス目標と整合しているか
  prepare:
    - ワークロードの可観測性が設計されているか
    - デプロイ戦略 (Blue/Green, Canary) が定義されているか
  operate:
    - ランブックとプレイブックが整備されているか
    - ダッシュボードとアラートが適切に設定されているか
  evolve:
    - ポストモーテム (振り返り) プロセスがあるか
    - 改善項目がバックログに追加されているか
```

#### 運用上の優秀性 — 主要 AWS サービスと実装パターン

```bash
# === IaC (Infrastructure as Code) の実装 ===

# CloudFormation スタックのドリフト検出
aws cloudformation detect-stack-drift --stack-name my-production-stack
aws cloudformation describe-stack-drift-detection-status \
  --stack-drift-detection-id "aaaabbbb-1234-5678"

# Systems Manager Automation でランブックを自動化
aws ssm start-automation-execution \
  --document-name "AWS-RestartEC2Instance" \
  --parameters '{"InstanceId":["i-0123456789abcdef0"]}'

# === 可観測性の実装 ===

# CloudWatch ダッシュボードの作成
aws cloudwatch put-dashboard \
  --dashboard-name "ProductionOverview" \
  --dashboard-body file://dashboard-definition.json

# X-Ray トレーシンググループの作成
aws xray create-group \
  --group-name "HighLatencyTraces" \
  --filter-expression 'responsetime > 5'

# CloudWatch Synthetics Canary でエンドポイント監視
aws synthetics create-canary \
  --name "api-health-check" \
  --artifact-s3-location "s3://my-canary-artifacts/" \
  --execution-role-arn "arn:aws:iam::123456789012:role/canary-role" \
  --schedule '{"Expression":"rate(5 minutes)"}' \
  --runtime-version "syn-nodejs-puppeteer-6.2" \
  --code '{"Handler":"apiCanary.handler","ZipFile":"..."}'

# === デプロイ戦略 ===

# CodeDeploy で Blue/Green デプロイ設定
aws deploy create-deployment-group \
  --application-name MyApp \
  --deployment-group-name Production \
  --deployment-config-name CodeDeployDefault.ECSLinear10PercentEvery1Minutes \
  --ecs-services '[{"ServiceName":"my-service","ClusterName":"my-cluster"}]' \
  --load-balancer-info '{"TargetGroupPairInfoList":[{"TargetGroups":[{"Name":"blue-tg"},{"Name":"green-tg"}],"ProdTrafficRoute":{"ListenerArns":["arn:aws:elasticloadbalancing:..."]}}]}'
```

```python
# 図1: 運用ダッシュボード自動生成スクリプト
import boto3
import json

def create_operations_dashboard(stack_name: str, region: str = "ap-northeast-1"):
    """CloudFormation スタックから自動的に運用ダッシュボードを生成"""
    cf_client = boto3.client("cloudformation", region_name=region)
    cw_client = boto3.client("cloudwatch", region_name=region)

    # スタックのリソースを取得
    resources = cf_client.list_stack_resources(StackName=stack_name)
    widgets = []
    y_pos = 0

    for resource in resources["StackResourceSummaries"]:
        resource_type = resource["ResourceType"]
        logical_id = resource["LogicalResourceId"]
        physical_id = resource["PhysicalResourceId"]

        if resource_type == "AWS::ECS::Service":
            # ECS サービスのCPU/メモリウィジェット
            cluster_name, service_name = physical_id.rsplit("/", 1)
            widgets.append({
                "type": "metric",
                "x": 0, "y": y_pos, "width": 12, "height": 6,
                "properties": {
                    "title": f"ECS: {logical_id}",
                    "metrics": [
                        ["AWS/ECS", "CPUUtilization", "ServiceName", service_name,
                         "ClusterName", cluster_name.split("/")[-1]],
                        [".", "MemoryUtilization", ".", ".", ".", "."],
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": region,
                },
            })
            y_pos += 6

        elif resource_type == "AWS::RDS::DBInstance":
            # RDS のコネクション数/CPU ウィジェット
            widgets.append({
                "type": "metric",
                "x": 0, "y": y_pos, "width": 12, "height": 6,
                "properties": {
                    "title": f"RDS: {logical_id}",
                    "metrics": [
                        ["AWS/RDS", "CPUUtilization",
                         "DBInstanceIdentifier", physical_id],
                        [".", "DatabaseConnections", ".", "."],
                        [".", "FreeStorageSpace", ".", "."],
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": region,
                },
            })
            y_pos += 6

        elif resource_type == "AWS::Lambda::Function":
            # Lambda のエラー率/実行時間ウィジェット
            widgets.append({
                "type": "metric",
                "x": 0, "y": y_pos, "width": 12, "height": 6,
                "properties": {
                    "title": f"Lambda: {logical_id}",
                    "metrics": [
                        ["AWS/Lambda", "Errors", "FunctionName", physical_id],
                        [".", "Duration", ".", "."],
                        [".", "ConcurrentExecutions", ".", "."],
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": region,
                },
            })
            y_pos += 6

    # ダッシュボード作成
    dashboard_body = {"widgets": widgets}
    cw_client.put_dashboard(
        DashboardName=f"{stack_name}-operations",
        DashboardBody=json.dumps(dashboard_body),
    )
    print(f"Dashboard created: {stack_name}-operations ({len(widgets)} widgets)")
    return dashboard_body
```

### 2.2 柱 2: セキュリティ（Security）

```yaml
principles:
  - 強力な ID 基盤を実装する
  - トレーサビリティを有効にする
  - 全レイヤーにセキュリティを適用する
  - セキュリティのベストプラクティスを自動化する
  - 転送中および保管中のデータを保護する
  - データに人の手を触れさせない
  - セキュリティイベントに備える

checklist:
  identity:
    - MFA が全 IAM ユーザーに強制されているか
    - ルートアカウントが日常業務で使われていないか
    - 最小権限の原則が適用されているか
  detection:
    - CloudTrail が全リージョンで有効か
    - GuardDuty が有効か
    - Security Hub が設定されているか
  protection:
    - VPC フローログが有効か
    - WAF が設定されているか
    - データが暗号化されているか (KMS)
```

#### セキュリティ — 自動監査と修復の実装

```python
# 図2: Security Hub の検出結果を集約し自動修復をトリガーするスクリプト
import boto3
from datetime import datetime, timedelta

def audit_security_hub_findings(region: str = "ap-northeast-1"):
    """Security Hub の CRITICAL/HIGH 検出結果を取得し改善アクションを生成"""
    sh_client = boto3.client("securityhub", region_name=region)

    # 過去7日間のCRITICAL/HIGH検出結果を取得
    response = sh_client.get_findings(
        Filters={
            "SeverityLabel": [
                {"Value": "CRITICAL", "Comparison": "EQUALS"},
                {"Value": "HIGH", "Comparison": "EQUALS"},
            ],
            "WorkflowStatus": [
                {"Value": "NEW", "Comparison": "EQUALS"},
            ],
            "RecordState": [
                {"Value": "ACTIVE", "Comparison": "EQUALS"},
            ],
            "CreatedAt": [
                {
                    "Start": (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z",
                    "End": datetime.utcnow().isoformat() + "Z",
                }
            ],
        },
        SortCriteria=[
            {"Field": "SeverityLabel", "SortOrder": "desc"},
        ],
        MaxResults=100,
    )

    findings_by_pillar = {
        "identity": [],
        "detection": [],
        "protection": [],
        "data_protection": [],
        "incident_response": [],
    }

    for finding in response["Findings"]:
        title = finding["Title"]
        severity = finding["Severity"]["Label"]
        resource_type = finding["Resources"][0]["Type"] if finding["Resources"] else "Unknown"
        resource_id = finding["Resources"][0]["Id"] if finding["Resources"] else "Unknown"

        item = {
            "title": title,
            "severity": severity,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "standard": finding.get("ProductFields", {}).get("StandardsControlArn", ""),
            "remediation": finding.get("Remediation", {}).get("Recommendation", {}).get("Text", ""),
        }

        # 検出結果をカテゴリ別に分類
        if "IAM" in title or "MFA" in title or "credential" in title.lower():
            findings_by_pillar["identity"].append(item)
        elif "CloudTrail" in title or "GuardDuty" in title or "logging" in title.lower():
            findings_by_pillar["detection"].append(item)
        elif "encryption" in title.lower() or "KMS" in title or "S3" in title:
            findings_by_pillar["data_protection"].append(item)
        elif "VPC" in title or "Security Group" in title or "WAF" in title:
            findings_by_pillar["protection"].append(item)
        else:
            findings_by_pillar["incident_response"].append(item)

    # サマリーレポート生成
    total = sum(len(v) for v in findings_by_pillar.values())
    print(f"=== Security Hub Audit Report ===")
    print(f"Total findings: {total}")
    for category, items in findings_by_pillar.items():
        if items:
            print(f"\n[{category.upper()}] ({len(items)} findings)")
            for item in items:
                print(f"  [{item['severity']}] {item['title']}")
                print(f"    Resource: {item['resource_id']}")
                if item["remediation"]:
                    print(f"    Fix: {item['remediation'][:100]}...")

    return findings_by_pillar
```

```bash
# Security Hub の有効化と標準の適用
aws securityhub enable-security-hub \
  --enable-default-standards

# CIS AWS Foundations Benchmark の有効化
aws securityhub batch-enable-standards \
  --standards-subscription-requests '[
    {"StandardsArn":"arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark/v/1.4.0"}
  ]'

# GuardDuty の有効化
aws guardduty create-detector --enable

# Config Rules のコンプライアンスサマリー
aws configservice get-compliance-summary-by-config-rule \
  --query 'ComplianceSummary.{Compliant:CompliantResourceCount.CappedCount,NonCompliant:NonCompliantResourceCount.CappedCount}'
```

### 2.3 柱 3: 信頼性（Reliability）

```yaml
principles:
  - 障害から自動的に復旧する
  - 復旧手順をテストする
  - 水平にスケールする
  - キャパシティの推測をやめる
  - 自動化で変更を管理する

checklist:
  foundations:
    - サービスクォータが適切に設定されているか
    - ネットワークトポロジが冗長化されているか
  workload_architecture:
    - マイクロサービス or SOA で障害分離ができているか
    - 分散システムでの障害処理が実装されているか
  change_management:
    - Auto Scaling が設定されているか
    - 変更がモニタリングされているか
  failure_management:
    - バックアップと DR 戦略が定義されているか
    - RTO/RPO が明確に定義されているか
```

#### 信頼性 — DR 戦略と実装パターン

```
┌──────────────────────────────────────────────────────────────────┐
│                   DR 戦略の比較                                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ 戦略         │ RTO      │ RPO      │ コスト │ 複雑度       │ │
│  │─────────────│──────────│──────────│────────│─────────────│ │
│  │ Backup &    │ 24h+     │ 24h      │ $      │ Low          │ │
│  │ Restore     │          │          │        │              │ │
│  │─────────────│──────────│──────────│────────│─────────────│ │
│  │ Pilot Light │ 数時間   │ 数分     │ $$     │ Medium       │ │
│  │─────────────│──────────│──────────│────────│─────────────│ │
│  │ Warm        │ 数分     │ 秒単位   │ $$$    │ Medium-High  │ │
│  │ Standby     │          │          │        │              │ │
│  │─────────────│──────────│──────────│────────│─────────────│ │
│  │ Multi-Site  │ リアル   │ ゼロに   │ $$$$   │ High         │ │
│  │ Active      │ タイム   │ 近い     │        │              │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

```bash
# === サービスクォータの管理 ===

# 現在のクォータ値を確認
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-1216C47A  # Running On-Demand Standard instances

# クォータ引き上げリクエスト
aws service-quotas request-service-quota-increase \
  --service-code ec2 \
  --quota-code L-1216C47A \
  --desired-value 200

# === バックアップ戦略 ===

# AWS Backup プラン作成
aws backup create-backup-plan --backup-plan '{
  "BackupPlanName": "DailyBackup",
  "Rules": [
    {
      "RuleName": "DailyRule",
      "TargetBackupVaultName": "Default",
      "ScheduleExpression": "cron(0 3 * * ? *)",
      "StartWindowMinutes": 60,
      "CompletionWindowMinutes": 120,
      "Lifecycle": {
        "DeleteAfterDays": 35,
        "MoveToColdStorageAfterDays": 7
      },
      "CopyActions": [
        {
          "DestinationBackupVaultArn": "arn:aws:backup:us-west-2:123456789012:backup-vault:DR-Vault",
          "Lifecycle": {"DeleteAfterDays": 90}
        }
      ]
    }
  ]
}'

# === Route 53 ヘルスチェック ===

# ヘルスチェックの作成
aws route53 create-health-check --caller-reference "$(date +%s)" \
  --health-check-config '{
    "IPAddress": "203.0.113.1",
    "Port": 443,
    "Type": "HTTPS",
    "ResourcePath": "/health",
    "FullyQualifiedDomainName": "api.example.com",
    "RequestInterval": 10,
    "FailureThreshold": 3,
    "EnableSNI": true
  }'

# フェイルオーバーレコードの作成
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "primary",
        "Failover": "PRIMARY",
        "AliasTarget": {
          "HostedZoneId": "Z35SXDOTRQ7X7K",
          "DNSName": "alb-primary.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        },
        "HealthCheckId": "abcd1234-5678-efgh"
      }
    }]
  }'
```

```python
# 図3: サービスクォータ自動監視スクリプト
import boto3

def check_service_quotas(services: list[str], threshold_pct: float = 80.0):
    """サービスクォータの使用率をチェックし閾値超過をアラート"""
    sq_client = boto3.client("service-quotas", region_name="ap-northeast-1")
    alerts = []

    for service_code in services:
        try:
            paginator = sq_client.get_paginator("list_service_quotas")
            for page in paginator.paginate(ServiceCode=service_code):
                for quota in page["Quotas"]:
                    quota_name = quota["QuotaName"]
                    quota_value = quota["Value"]

                    # 使用量の取得（利用可能な場合）
                    if quota.get("UsageMetric"):
                        metric = quota["UsageMetric"]
                        cw_client = boto3.client("cloudwatch", region_name="ap-northeast-1")
                        stats = cw_client.get_metric_statistics(
                            Namespace=metric["MetricNamespace"],
                            MetricName=metric["MetricName"],
                            Dimensions=[
                                {"Name": k, "Value": v}
                                for k, v in metric.get("MetricDimensions", {}).items()
                            ],
                            StartTime="2025-01-01T00:00:00Z",
                            EndTime="2025-01-02T00:00:00Z",
                            Period=86400,
                            Statistics=[metric.get("MetricStatisticRecommendation", "Maximum")],
                        )
                        if stats["Datapoints"]:
                            usage = stats["Datapoints"][0].get("Maximum", 0)
                            usage_pct = (usage / quota_value) * 100 if quota_value > 0 else 0

                            if usage_pct >= threshold_pct:
                                alerts.append({
                                    "service": service_code,
                                    "quota": quota_name,
                                    "limit": quota_value,
                                    "usage": usage,
                                    "usage_pct": round(usage_pct, 1),
                                })
        except Exception as e:
            print(f"Error checking {service_code}: {e}")

    # アラートレポート
    if alerts:
        print(f"⚠ {len(alerts)} quotas above {threshold_pct}% threshold:")
        for a in alerts:
            print(f"  [{a['service']}] {a['quota']}: {a['usage']}/{a['limit']} ({a['usage_pct']}%)")
    else:
        print(f"All quotas below {threshold_pct}% threshold")

    return alerts

# 使用例
# check_service_quotas(["ec2", "lambda", "rds", "elasticloadbalancing"])
```

### 2.4 柱 4: パフォーマンス効率（Performance Efficiency）

```yaml
principles:
  - 高度なテクノロジーを民主化する
  - 数分でグローバルにデプロイする
  - サーバーレスアーキテクチャを活用する
  - より頻繁に実験する
  - メカニカルシンパシー (技術への共感) を持つ

checklist:
  selection:
    - ワークロードに最適なコンピュートタイプを選択しているか
      (EC2 / Lambda / ECS / EKS / Fargate)
    - ストレージ種別は適切か (gp3 / io2 / S3 Intelligent-Tiering)
    - データベースエンジンは適切か (Aurora / DynamoDB / ElastiCache)
  review:
    - ベンチマークテストが定期的に実施されているか
    - CloudWatch メトリクスでパフォーマンスを継続監視しているか
  monitoring:
    - P50/P90/P99 レイテンシが計測されているか
    - ボトルネックの特定プロセスがあるか (X-Ray / Profiler)
  tradeoffs:
    - キャッシュ戦略が定義されているか (CloudFront / ElastiCache / DAX)
    - リードレプリカが活用されているか
```

```bash
# === パフォーマンス分析 ===

# Compute Optimizer の推奨事項を取得
aws compute-optimizer get-ec2-instance-recommendations \
  --query 'InstanceRecommendations[*].{
    Instance:InstanceArn,
    Current:CurrentInstanceType,
    Recommended:RecommendationOptions[0].InstanceType,
    Finding:Finding,
    Savings:RecommendationOptions[0].ProjectedUtilizationMetrics[?Name==`CPU`].Value|[0]
  }'

# Lambda 関数のパフォーマンス分析
aws lambda get-function-configuration \
  --function-name my-function \
  --query '{MemorySize:MemorySize,Timeout:Timeout,Architecture:Architectures[0]}'

# Lambda Power Tuning の結果を確認（Step Functions 経由で実行後）
# 最適なメモリサイズを特定: 128MB → 256MB → 512MB → 1024MB の比較

# CloudFront キャッシュヒット率の確認
aws cloudwatch get-metric-statistics \
  --namespace AWS/CloudFront \
  --metric-name CacheHitRate \
  --dimensions Name=DistributionId,Value=E1234567890 \
  --start-time "$(date -u -v-7d +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 86400 \
  --statistics Average

# RDS Performance Insights の有効化
aws rds modify-db-instance \
  --db-instance-identifier my-database \
  --enable-performance-insights \
  --performance-insights-retention-period 731
```

```python
# 図4: パフォーマンスベースライン自動取得スクリプト
import boto3
from datetime import datetime, timedelta

def get_performance_baseline(resource_type: str, resource_id: str, days: int = 30):
    """リソースの過去N日間のパフォーマンスベースラインを取得"""
    cw = boto3.client("cloudwatch", region_name="ap-northeast-1")
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)

    metrics_map = {
        "ec2": {
            "namespace": "AWS/EC2",
            "dimension": "InstanceId",
            "metrics": ["CPUUtilization", "NetworkIn", "NetworkOut",
                        "EBSReadOps", "EBSWriteOps"],
        },
        "rds": {
            "namespace": "AWS/RDS",
            "dimension": "DBInstanceIdentifier",
            "metrics": ["CPUUtilization", "DatabaseConnections",
                        "ReadLatency", "WriteLatency", "FreeableMemory"],
        },
        "lambda": {
            "namespace": "AWS/Lambda",
            "dimension": "FunctionName",
            "metrics": ["Duration", "Errors", "Throttles", "ConcurrentExecutions"],
        },
        "alb": {
            "namespace": "AWS/ApplicationELB",
            "dimension": "LoadBalancer",
            "metrics": ["TargetResponseTime", "HTTPCode_Target_5XX_Count",
                        "RequestCount", "ActiveConnectionCount"],
        },
    }

    config = metrics_map.get(resource_type)
    if not config:
        raise ValueError(f"Unsupported resource type: {resource_type}")

    baseline = {}
    for metric_name in config["metrics"]:
        response = cw.get_metric_statistics(
            Namespace=config["namespace"],
            MetricName=metric_name,
            Dimensions=[{"Name": config["dimension"], "Value": resource_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,  # 日次
            Statistics=["Average", "Maximum", "p99"],
            ExtendedStatistics=["p50", "p90", "p99"],
        )

        datapoints = sorted(response.get("Datapoints", []), key=lambda x: x["Timestamp"])
        if datapoints:
            averages = [dp["Average"] for dp in datapoints if "Average" in dp]
            maximums = [dp["Maximum"] for dp in datapoints if "Maximum" in dp]
            baseline[metric_name] = {
                "avg": round(sum(averages) / len(averages), 2) if averages else 0,
                "max": round(max(maximums), 2) if maximums else 0,
                "trend": "increasing" if len(averages) > 1 and averages[-1] > averages[0] else "stable",
            }

    print(f"=== Performance Baseline: {resource_type}/{resource_id} ({days} days) ===")
    for metric, stats in baseline.items():
        trend_indicator = "↑" if stats["trend"] == "increasing" else "→"
        print(f"  {metric}: avg={stats['avg']}, max={stats['max']} {trend_indicator}")

    return baseline
```

### 2.5 柱 5: コスト最適化（Cost Optimization）

```yaml
principles:
  - クラウド財務管理を実装する
  - 消費モデルを導入する
  - 全体的な効率を測定する
  - 差別化につながらない高負荷の作業への支出をやめる
  - 費用を分析し帰属させる

checklist:
  practice_cloud_financial_management:
    - コスト配分タグが全リソースに適用されているか
    - 予算アラートが設定されているか (AWS Budgets)
    - 月次のコストレビューミーティングが実施されているか
  expenditure_and_usage_awareness:
    - Cost Explorer で使用状況を分析しているか
    - 未使用リソース (EIP, EBS, snapshots) の棚卸しが定期的か
  cost_effective_resources:
    - Savings Plans / Reserved Instances を活用しているか
    - Spot インスタンスの活用を検討しているか
    - Graviton (ARM) インスタンスへの移行を検討しているか
  manage_demand_and_supply:
    - Auto Scaling で需要に応じたスケーリングをしているか
    - サーバーレスアーキテクチャへの移行を検討しているか
  optimize_over_time:
    - Compute Optimizer の推奨事項を定期的に確認しているか
    - S3 Intelligent-Tiering を活用しているか
```

```bash
# === コスト可視化と最適化 ===

# コスト配分タグの有効化確認
aws ce list-cost-allocation-tags \
  --status Active \
  --query 'CostAllocationTags[*].TagKey'

# 月次コストサマリー
aws ce get-cost-and-usage \
  --time-period Start=2025-01-01,End=2025-02-01 \
  --granularity MONTHLY \
  --metrics "BlendedCost" "UsageQuantity" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --query 'ResultsByTime[0].Groups | sort_by(@, &Metrics.BlendedCost.Amount) | reverse(@) | [:10]'

# 未使用リソースの検出
# 未使用 EIP
aws ec2 describe-addresses \
  --query 'Addresses[?AssociationId==null].{IP:PublicIp,AllocId:AllocationId}'

# 未アタッチ EBS ボリューム
aws ec2 describe-volumes \
  --filters Name=status,Values=available \
  --query 'Volumes[*].{Id:VolumeId,Size:Size,Type:VolumeType,Created:CreateTime}'

# 古い EBS スナップショット（90日以上前）
aws ec2 describe-snapshots --owner-ids self \
  --query "Snapshots[?StartTime<='$(date -u -v-90d +%Y-%m-%dT%H:%M:%S)'].{Id:SnapshotId,Size:VolumeSize,Date:StartTime}"

# Savings Plans カバレッジの確認
aws ce get-savings-plans-coverage \
  --time-period Start=2025-01-01,End=2025-02-01 \
  --query 'SavingsPlansCoverages[*].{Date:TimePeriod.Start,Coverage:CoveragePercentage}'

# Savings Plans 推奨事項の取得
aws ce get-savings-plans-purchase-recommendation \
  --savings-plans-type COMPUTE_SP \
  --term-in-years ONE_YEAR \
  --payment-option NO_UPFRONT \
  --lookback-period-in-days SIXTY_DAYS
```

```python
# 図5: コスト異常検出と自動アラートスクリプト
import boto3
from datetime import datetime, timedelta

def detect_cost_anomalies(threshold_pct: float = 20.0):
    """前月比でコスト異常を検出"""
    ce = boto3.client("ce", region_name="us-east-1")

    # 今月と先月のコストを取得
    today = datetime.utcnow()
    this_month_start = today.replace(day=1).strftime("%Y-%m-%d")
    last_month_start = (today.replace(day=1) - timedelta(days=1)).replace(day=1).strftime("%Y-%m-%d")
    last_month_end = today.replace(day=1).strftime("%Y-%m-%d")

    # 先月のサービス別コスト
    last_month = ce.get_cost_and_usage(
        TimePeriod={"Start": last_month_start, "End": last_month_end},
        Granularity="MONTHLY",
        Metrics=["BlendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    # 今月のサービス別コスト（日割り換算）
    this_month = ce.get_cost_and_usage(
        TimePeriod={"Start": this_month_start, "End": today.strftime("%Y-%m-%d")},
        Granularity="MONTHLY",
        Metrics=["BlendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    days_elapsed = (today - today.replace(day=1)).days or 1
    days_in_month = 30  # 概算

    anomalies = []
    last_costs = {}
    for group in last_month["ResultsByTime"][0].get("Groups", []):
        service = group["Keys"][0]
        cost = float(group["Metrics"]["BlendedCost"]["Amount"])
        last_costs[service] = cost

    for group in this_month["ResultsByTime"][0].get("Groups", []):
        service = group["Keys"][0]
        current_cost = float(group["Metrics"]["BlendedCost"]["Amount"])
        projected_cost = current_cost * (days_in_month / days_elapsed)

        last_cost = last_costs.get(service, 0)
        if last_cost > 1:  # $1以上のサービスのみ
            change_pct = ((projected_cost - last_cost) / last_cost) * 100
            if change_pct > threshold_pct:
                anomalies.append({
                    "service": service,
                    "last_month": round(last_cost, 2),
                    "projected": round(projected_cost, 2),
                    "change_pct": round(change_pct, 1),
                })

    anomalies.sort(key=lambda x: x["change_pct"], reverse=True)

    print(f"=== Cost Anomaly Report (threshold: +{threshold_pct}%) ===")
    for a in anomalies:
        print(f"  {a['service']}: ${a['last_month']} → ${a['projected']} (+{a['change_pct']}%)")

    return anomalies
```

### 2.6 柱 6: 持続可能性（Sustainability）

```yaml
principles:
  - 影響を理解する
  - 持続可能性の目標を設定する
  - 使用率を最大化する
  - より効率的な新しいハードウェアとソフトウェアを活用する
  - マネージドサービスを使用する
  - クラウドワークロードの下流への影響を最小化する

checklist:
  region_selection:
    - カーボンフリーエネルギーの比率が高いリージョンを選択しているか
    - ユーザーに近いリージョンで無駄な通信を削減しているか
  compute:
    - Graviton (ARM) プロセッサを利用しているか
    - Spot インスタンスで余剰キャパシティを活用しているか
    - Lambda でアイドル時のリソース消費をゼロにしているか
  storage:
    - S3 Intelligent-Tiering でストレージクラスを自動最適化しているか
    - 不要なデータのライフサイクルポリシーを設定しているか
  data_transfer:
    - CloudFront でエッジキャッシュを活用しているか
    - VPC エンドポイントで不要なインターネット通信を削減しているか
```

```bash
# === 持続可能性の実践 ===

# Graviton インスタンスの利用状況確認
aws ec2 describe-instances \
  --query 'Reservations[*].Instances[*].{
    Id:InstanceId,
    Type:InstanceType,
    Arch:Architecture,
    Platform:PlatformDetails
  }' --output table

# Customer Carbon Footprint Tool（コンソールでのみ利用可能）
# 代替: CloudWatch で CO2 関連メトリクスを推定

# S3 Intelligent-Tiering の設定
aws s3api put-bucket-intelligent-tiering-configuration \
  --bucket my-data-bucket \
  --id "FullTiering" \
  --intelligent-tiering-configuration '{
    "Id": "FullTiering",
    "Status": "Enabled",
    "Tierings": [
      {"AccessTier": "ARCHIVE_ACCESS", "Days": 90},
      {"AccessTier": "DEEP_ARCHIVE_ACCESS", "Days": 180}
    ]
  }'

# S3 ライフサイクルポリシーの設定
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-logs-bucket \
  --lifecycle-configuration '{
    "Rules": [
      {
        "ID": "LogRetention",
        "Status": "Enabled",
        "Filter": {"Prefix": "logs/"},
        "Transitions": [
          {"Days": 30, "StorageClass": "STANDARD_IA"},
          {"Days": 90, "StorageClass": "GLACIER"},
          {"Days": 365, "StorageClass": "DEEP_ARCHIVE"}
        ],
        "Expiration": {"Days": 730}
      }
    ]
  }'
```

### 2.7 6 つの柱のベストプラクティス要約

```python
# Well-Architected レビューの自動化スクリプト例
import boto3

def run_well_architected_review():
    """Well-Architected Tool でワークロードレビューを自動化"""
    wa_client = boto3.client("wellarchitected", region_name="ap-northeast-1")

    # ワークロード作成
    workload = wa_client.create_workload(
        WorkloadName="MyApp Production",
        Description="Main production workload",
        Environment="PRODUCTION",
        ArchitecturalDesign="https://wiki.example.com/architecture",
        ReviewOwner="architect@example.com",
        Lenses=["wellarchitected"],          # AWS Well-Architected Lens
        AwsRegions=["ap-northeast-1"],
        Tags={"Project": "myapp"},
    )

    workload_id = workload["WorkloadId"]

    # 質問一覧の取得
    answers = wa_client.list_answers(
        WorkloadId=workload_id,
        LensAlias="wellarchitected",
        PillarId="operationalExcellence",    # 柱を指定
    )

    for answer in answers["AnswerSummaries"]:
        print(f"Q: {answer['QuestionTitle']}")
        print(f"  Risk: {answer.get('Risk', 'UNANSWERED')}")
        print()

    return workload_id
```

### 2.8 Lens の活用

```bash
# 利用可能な Lens の一覧
aws wellarchitected list-lenses --query 'LensSummaries[*].{Name:LensName,Alias:LensAlias}'

# よく使う Lens:
# - wellarchitected          : AWS Well-Architected (デフォルト)
# - serverless               : Serverless Applications Lens
# - saas                     : SaaS Lens
# - foundational-technical-review : FTR Lens (APN パートナー向け)
# - machine-learning         : Machine Learning Lens

# カスタム Lens の作成
aws wellarchitected create-lens-version \
  --lens-alias "arn:aws:wellarchitected:ap-northeast-1:123456789012:lens/my-custom-lens" \
  --lens-version "2.0.0"

# ワークロードに Lens を関連付け
aws wellarchitected associate-lenses \
  --workload-id "abc123def456" \
  --lens-aliases serverless saas

# Lens の改善計画を取得
aws wellarchitected list-lens-review-improvements \
  --workload-id "abc123def456" \
  --lens-alias wellarchitected \
  --pillar-id security \
  --query 'ImprovementSummaries[*].{Question:QuestionTitle,Risk:Risk,Count:ImprovementPlans|length(@)}'
```

---

## 3. Well-Architected Tool でのレビュー

### 3.1 レビュープロセス

```
┌─────────────────────────────────────────────────────────┐
│         Well-Architected Review プロセス                  │
│                                                         │
│  Phase 1: 準備 (1-2日)                                  │
│  ┌───────────────────────────────────────┐              │
│  │ - アーキテクチャ図の準備               │              │
│  │ - ステークホルダーの特定               │              │
│  │ - 既知のリスクの整理                   │              │
│  └──────────────────┬────────────────────┘              │
│                     ▼                                   │
│  Phase 2: レビュー (2-5日)                               │
│  ┌───────────────────────────────────────┐              │
│  │ - 6つの柱ごとに質問に回答             │              │
│  │ - ベストプラクティスとの差分を特定     │              │
│  │ - リスクレベルの評価                   │              │
│  │   (High Risk / Medium Risk / No Risk) │              │
│  └──────────────────┬────────────────────┘              │
│                     ▼                                   │
│  Phase 3: 改善計画 (1-2日)                               │
│  ┌───────────────────────────────────────┐              │
│  │ - High Risk 項目の改善策を策定         │              │
│  │ - 優先順位とマイルストーンを設定       │              │
│  │ - 改善項目を Jira/Backlog に登録       │              │
│  └──────────────────┬────────────────────┘              │
│                     ▼                                   │
│  Phase 4: 実行と再レビュー (継続)                        │
│  ┌───────────────────────────────────────┐              │
│  │ - 改善を実施                           │              │
│  │ - 四半期ごとに再レビュー               │              │
│  │ - マイルストーンで進捗を記録           │              │
│  └───────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 レビュー実施の詳細手順

```bash
# Step 1: ワークロードの作成
WORKLOAD_ID=$(aws wellarchitected create-workload \
  --workload-name "E-Commerce-Production" \
  --description "Primary e-commerce platform serving ap-northeast-1" \
  --environment PRODUCTION \
  --review-owner "tech-lead@example.com" \
  --lenses wellarchitected serverless \
  --aws-regions ap-northeast-1 \
  --account-ids 123456789012 987654321098 \
  --tags Project=ecommerce,Team=platform,ReviewCycle=Q1-2025 \
  --query 'WorkloadId' --output text)

echo "Created workload: $WORKLOAD_ID"

# Step 2: 各柱の質問と回答状況を確認
for PILLAR in operationalExcellence security reliability performance costOptimization sustainability; do
  echo "=== $PILLAR ==="
  aws wellarchitected list-answers \
    --workload-id "$WORKLOAD_ID" \
    --lens-alias wellarchitected \
    --pillar-id "$PILLAR" \
    --query 'AnswerSummaries[*].{Q:QuestionTitle,Risk:Risk}' \
    --output table
done

# Step 3: 回答の更新（選択肢を選ぶ）
aws wellarchitected update-answer \
  --workload-id "$WORKLOAD_ID" \
  --lens-alias wellarchitected \
  --question-id "sec-how-do-you-manage-identities" \
  --selected-choices \
    "sec_sec-how-do-you-manage-identities_1" \
    "sec_sec-how-do-you-manage-identities_2" \
    "sec_sec-how-do-you-manage-identities_3" \
  --notes "IAM Identity Center with SAML 2.0 federation. MFA enforced for all users. Service-linked roles for AWS services."

# Step 4: マイルストーンの作成（レビュー完了時点のスナップショット）
aws wellarchitected create-milestone \
  --workload-id "$WORKLOAD_ID" \
  --milestone-name "Q1-2025-Initial-Review"

# Step 5: 改善計画の取得
aws wellarchitected list-lens-review-improvements \
  --workload-id "$WORKLOAD_ID" \
  --lens-alias wellarchitected \
  --pillar-id security \
  --query 'ImprovementSummaries[?Risk==`HIGH`].{Q:QuestionTitle,Risk:Risk}'
```

### 3.3 改善計画テンプレート

```markdown
## Well-Architected 改善計画

### High Risk Items (最優先)

| # | 柱 | 質問 | 現状のリスク | 改善アクション | 担当 | 期限 |
|---|---|------|------------|---------------|------|------|
| 1 | セキュリティ | 認証情報の管理 | ハードコード | Secrets Manager 導入 | @security | 2W |
| 2 | 信頼性 | バックアップ | 手動・不定期 | AWS Backup 自動化 | @infra | 3W |
| 3 | 運用 | モニタリング | ログ未収集 | CloudWatch + X-Ray | @sre | 4W |

### Medium Risk Items (次フェーズ)

| # | 柱 | 質問 | 改善アクション | 期限 |
|---|---|------|---------------|------|
| 4 | コスト | ライトサイジング | Compute Optimizer 適用 | Q2 |
| 5 | パフォーマンス | キャッシュ戦略 | ElastiCache 導入 | Q2 |
```

### 3.4 レビュー結果の自動レポート生成

```python
# 図6: Well-Architected レビュー結果をMarkdownレポートとして出力
import boto3
from datetime import datetime

def generate_wa_report(workload_id: str, output_file: str = "wa-report.md"):
    """Well-Architected レビュー結果をMarkdownレポートに変換"""
    wa = boto3.client("wellarchitected", region_name="ap-northeast-1")

    # ワークロード情報の取得
    workload = wa.get_workload(WorkloadId=workload_id)["Workload"]

    # Lens レビューの取得
    review = wa.get_lens_review(
        WorkloadId=workload_id,
        LensAlias="wellarchitected",
    )["LensReview"]

    report_lines = [
        f"# Well-Architected Review Report",
        f"",
        f"**Workload:** {workload['WorkloadName']}",
        f"**Date:** {datetime.utcnow().strftime('%Y-%m-%d')}",
        f"**Environment:** {workload['Environment']}",
        f"**Review Owner:** {workload.get('ReviewOwner', 'N/A')}",
        f"",
        f"## Risk Summary",
        f"",
    ]

    # リスクサマリー
    risk_counts = review.get("RiskCounts", {})
    total_questions = sum(risk_counts.values())
    report_lines.extend([
        f"| Risk Level | Count | Percentage |",
        f"|-----------|-------|-----------|",
        f"| HIGH | {risk_counts.get('HIGH', 0)} | {risk_counts.get('HIGH', 0)/total_questions*100:.0f}% |",
        f"| MEDIUM | {risk_counts.get('MEDIUM', 0)} | {risk_counts.get('MEDIUM', 0)/total_questions*100:.0f}% |",
        f"| NONE | {risk_counts.get('NONE', 0)} | {risk_counts.get('NONE', 0)/total_questions*100:.0f}% |",
        f"| UNANSWERED | {risk_counts.get('UNANSWERED', 0)} | {risk_counts.get('UNANSWERED', 0)/total_questions*100:.0f}% |",
        f"",
    ])

    # 柱ごとの詳細
    pillars = [
        ("operationalExcellence", "Operational Excellence (運用上の優秀性)"),
        ("security", "Security (セキュリティ)"),
        ("reliability", "Reliability (信頼性)"),
        ("performance", "Performance Efficiency (パフォーマンス効率)"),
        ("costOptimization", "Cost Optimization (コスト最適化)"),
        ("sustainability", "Sustainability (持続可能性)"),
    ]

    for pillar_id, pillar_name in pillars:
        report_lines.extend([f"", f"## {pillar_name}", f""])

        answers = wa.list_answers(
            WorkloadId=workload_id,
            LensAlias="wellarchitected",
            PillarId=pillar_id,
        )

        report_lines.extend([
            f"| Question | Risk | Notes |",
            f"|---------|------|-------|",
        ])

        for ans in answers["AnswerSummaries"]:
            risk = ans.get("Risk", "UNANSWERED")
            risk_emoji = {"HIGH": "[HIGH]", "MEDIUM": "[MED]", "NONE": "[OK]"}.get(risk, "[?]")
            notes = ans.get("Notes", "").replace("\n", " ")[:50]
            report_lines.append(
                f"| {ans['QuestionTitle'][:60]} | {risk_emoji} | {notes} |"
            )

    # 改善計画
    report_lines.extend([
        f"",
        f"## Improvement Plan",
        f"",
        f"### High Risk Items (Immediate Action Required)",
        f"",
    ])

    for pillar_id, pillar_name in pillars:
        improvements = wa.list_lens_review_improvements(
            WorkloadId=workload_id,
            LensAlias="wellarchitected",
            PillarId=pillar_id,
        )
        high_risk = [i for i in improvements["ImprovementSummaries"] if i.get("Risk") == "HIGH"]
        for item in high_risk:
            report_lines.append(
                f"- **[{pillar_name}]** {item['QuestionTitle']}: "
                f"{item.get('ImprovementPlanUrl', 'See AWS documentation')}"
            )

    # ファイル出力
    report_content = "\n".join(report_lines)
    with open(output_file, "w") as f:
        f.write(report_content)

    print(f"Report saved to {output_file}")
    print(f"Total risks: HIGH={risk_counts.get('HIGH', 0)}, MEDIUM={risk_counts.get('MEDIUM', 0)}")
    return report_content
```

---

## 4. Trusted Advisor との連携

### 4.1 Trusted Advisor カテゴリと Well-Architected 柱の対応

```
┌──────────────────────────────────────────────────────────────────────┐
│  Trusted Advisor カテゴリ      │  Well-Architected 柱              │
│──────────────────────────────────────────────────────────────────── │
│  Cost Optimization             │  コスト最適化                      │
│  Performance                   │  パフォーマンス効率                │
│  Security                      │  セキュリティ                      │
│  Fault Tolerance               │  信頼性                            │
│  Service Limits                │  信頼性 (Foundations)              │
│  Operational Excellence (新設) │  運用上の優秀性                    │
└──────────────────────────────────────────────────────────────────────┘
```

```bash
# Trusted Advisor チェック結果の取得（Business/Enterprise Support 必要）
aws support describe-trusted-advisor-checks \
  --language ja \
  --query 'checks[*].{Id:id,Name:name,Category:category}' \
  --output table

# 特定のチェックの結果を取得
aws support describe-trusted-advisor-check-result \
  --check-id "Qch7DwouX1" \
  --query 'result.{Status:status,Flagged:flaggedResources|length(@)}'

# 全チェックのサマリー
aws support describe-trusted-advisor-check-summaries \
  --check-ids $(aws support describe-trusted-advisor-checks \
    --language en \
    --query 'checks[*].id' --output text) \
  --query 'summaries[?status!=`ok`].{Check:checkId,Status:status,Flagged:flaggedResources.resourcesFlagged}'

# チェック結果のリフレッシュ
aws support refresh-trusted-advisor-check --check-id "Qch7DwouX1"
```

```python
# 図7: Trusted Advisor 結果を Well-Architected 形式で集約するスクリプト
import boto3

def aggregate_trusted_advisor_by_pillar():
    """Trusted Advisor の結果を Well-Architected の柱別に集約"""
    support = boto3.client("support", region_name="us-east-1")

    # カテゴリと柱のマッピング
    category_to_pillar = {
        "cost_optimizing": "costOptimization",
        "performance": "performance",
        "security": "security",
        "fault_tolerance": "reliability",
        "service_limits": "reliability",
    }

    checks = support.describe_trusted_advisor_checks(language="en")["checks"]

    pillar_results = {
        "operationalExcellence": {"ok": 0, "warning": 0, "error": 0, "items": []},
        "security": {"ok": 0, "warning": 0, "error": 0, "items": []},
        "reliability": {"ok": 0, "warning": 0, "error": 0, "items": []},
        "performance": {"ok": 0, "warning": 0, "error": 0, "items": []},
        "costOptimization": {"ok": 0, "warning": 0, "error": 0, "items": []},
    }

    for check in checks:
        pillar = category_to_pillar.get(check["category"], "operationalExcellence")

        try:
            result = support.describe_trusted_advisor_check_result(
                checkId=check["id"]
            )["result"]

            status = result["status"]
            flagged_count = len(result.get("flaggedResources", []))

            pillar_results[pillar][status] = pillar_results[pillar].get(status, 0) + 1

            if status in ("warning", "error"):
                pillar_results[pillar]["items"].append({
                    "check": check["name"],
                    "status": status,
                    "flagged": flagged_count,
                    "description": check.get("description", "")[:100],
                })
        except Exception:
            pass  # チェックがサポートプランで利用不可の場合

    # レポート出力
    print("=== Trusted Advisor → Well-Architected Mapping ===\n")
    for pillar, data in pillar_results.items():
        total = data["ok"] + data["warning"] + data["error"]
        if total > 0:
            score = data["ok"] / total * 100
            print(f"[{pillar}] Score: {score:.0f}% ({data['ok']}/{total} checks OK)")
            for item in data["items"]:
                indicator = "!!" if item["status"] == "error" else "!"
                print(f"  {indicator} {item['check']} ({item['flagged']} resources)")

    return pillar_results
```

---

## 5. CDK による Well-Architected 準拠スタック

### 5.1 CDK で Well-Architected のベストプラクティスを自動適用

```typescript
// 図8: Well-Architected 準拠の基盤スタック（CDK TypeScript）
import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as rds from "aws-cdk-lib/aws-rds";
import * as backup from "aws-cdk-lib/aws-backup";
import * as cloudwatch from "aws-cdk-lib/aws-cloudwatch";
import * as sns from "aws-cdk-lib/aws-sns";
import * as budgets from "aws-cdk-lib/aws-budgets";
import * as iam from "aws-cdk-lib/aws-iam";
import * as kms from "aws-cdk-lib/aws-kms";

export interface WellArchitectedStackProps extends cdk.StackProps {
  environment: "production" | "staging" | "development";
  monthlyBudgetUsd: number;
  alertEmail: string;
}

export class WellArchitectedFoundationStack extends cdk.Stack {
  public readonly vpc: ec2.IVpc;
  public readonly alarmTopic: sns.ITopic;

  constructor(scope: Construct, id: string, props: WellArchitectedStackProps) {
    super(scope, id, props);

    // ============================================================
    // 柱1: セキュリティ — 暗号化キーの集中管理
    // ============================================================
    const encryptionKey = new kms.Key(this, "EncryptionKey", {
      alias: `${props.environment}-master-key`,
      enableKeyRotation: true,           // 年次自動ローテーション
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      description: "Master encryption key for all data-at-rest encryption",
    });

    // ============================================================
    // 柱2: 信頼性 — マルチAZ VPC
    // ============================================================
    this.vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 3,                          // 3AZ 冗長構成
      natGateways: props.environment === "production" ? 3 : 1,
      ipAddresses: ec2.IpAddresses.cidr("10.0.0.0/16"),
      subnetConfiguration: [
        {
          name: "Public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: "Private",
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
        {
          name: "Isolated",
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
      flowLogs: {
        "VpcFlowLog": {
          destination: ec2.FlowLogDestination.toCloudWatchLogs(),
          trafficType: ec2.FlowLogTrafficType.ALL,
        },
      },
    });

    // ============================================================
    // 柱3: 運用上の優秀性 — アラーム通知
    // ============================================================
    this.alarmTopic = new sns.Topic(this, "AlarmTopic", {
      topicName: `${props.environment}-wa-alarms`,
      displayName: `[${props.environment.toUpperCase()}] Well-Architected Alarms`,
    });

    new sns.Subscription(this, "AlarmEmail", {
      topic: this.alarmTopic,
      protocol: sns.SubscriptionProtocol.EMAIL,
      endpoint: props.alertEmail,
    });

    // VPC NAT Gateway のエラー監視
    const natErrorAlarm = new cloudwatch.Alarm(this, "NatGatewayError", {
      metric: new cloudwatch.Metric({
        namespace: "AWS/NATGateway",
        metricName: "ErrorPortAllocation",
        statistic: "Sum",
        period: cdk.Duration.minutes(5),
      }),
      threshold: 0,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      evaluationPeriods: 1,
      alarmDescription: "NAT Gateway port allocation errors detected",
    });
    natErrorAlarm.addAlarmAction(
      new cdk.aws_cloudwatch_actions.SnsAction(this.alarmTopic)
    );

    // ============================================================
    // 柱4: 信頼性 — 自動バックアップ
    // ============================================================
    const backupPlan = new backup.BackupPlan(this, "BackupPlan", {
      backupPlanName: `${props.environment}-daily-backup`,
      backupPlanRules: [
        new backup.BackupPlanRule({
          ruleName: "DailyBackup",
          scheduleExpression: cdk.aws_events.Schedule.cron({
            hour: "3",
            minute: "0",
          }),
          startWindow: cdk.Duration.hours(1),
          completionWindow: cdk.Duration.hours(2),
          deleteAfter: cdk.Duration.days(35),
          moveToColdStorageAfter: cdk.Duration.days(7),
        }),
        new backup.BackupPlanRule({
          ruleName: "MonthlyBackup",
          scheduleExpression: cdk.aws_events.Schedule.cron({
            day: "1",
            hour: "3",
            minute: "0",
          }),
          deleteAfter: cdk.Duration.days(365),
          moveToColdStorageAfter: cdk.Duration.days(30),
        }),
      ],
    });

    // タグベースでバックアップ対象を選択
    backupPlan.addSelection("TaggedResources", {
      resources: [
        backup.BackupResource.fromTag("backup", "true"),
      ],
    });

    // ============================================================
    // 柱5: コスト最適化 — 予算アラート
    // ============================================================
    new budgets.CfnBudget(this, "MonthlyBudget", {
      budget: {
        budgetName: `${props.environment}-monthly-budget`,
        budgetType: "COST",
        timeUnit: "MONTHLY",
        budgetLimit: {
          amount: props.monthlyBudgetUsd,
          unit: "USD",
        },
      },
      notificationsWithSubscribers: [
        {
          notification: {
            notificationType: "ACTUAL",
            comparisonOperator: "GREATER_THAN",
            threshold: 80,
            thresholdType: "PERCENTAGE",
          },
          subscribers: [
            { subscriptionType: "EMAIL", address: props.alertEmail },
          ],
        },
        {
          notification: {
            notificationType: "FORECASTED",
            comparisonOperator: "GREATER_THAN",
            threshold: 100,
            thresholdType: "PERCENTAGE",
          },
          subscribers: [
            { subscriptionType: "EMAIL", address: props.alertEmail },
          ],
        },
      ],
    });

    // ============================================================
    // タグ付け（全柱共通のベストプラクティス）
    // ============================================================
    cdk.Tags.of(this).add("Environment", props.environment);
    cdk.Tags.of(this).add("ManagedBy", "CDK");
    cdk.Tags.of(this).add("WellArchitected", "true");
    cdk.Tags.of(this).add("backup", "true");

    // ============================================================
    // 出力
    // ============================================================
    new cdk.CfnOutput(this, "VpcId", { value: this.vpc.vpcId });
    new cdk.CfnOutput(this, "AlarmTopicArn", { value: this.alarmTopic.topicArn });
    new cdk.CfnOutput(this, "KmsKeyArn", { value: encryptionKey.keyArn });
  }
}
```

### 5.2 CloudFormation による Well-Architected 自動チェック（Config Rules）

```yaml
# 図9: Well-Architected ベストプラクティスの自動チェック（Config Rules）
AWSTemplateFormatVersion: "2010-09-09"
Description: "Well-Architected compliance checks via AWS Config"

Parameters:
  AlertEmail:
    Type: String
    Description: Email for compliance alerts

Resources:
  # === コンプライアンス通知 ===
  ComplianceTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: wa-compliance-alerts
      Subscription:
        - Protocol: email
          Endpoint: !Ref AlertEmail

  # === セキュリティの柱 ===

  # S3 バケットのパブリックアクセスブロック確認
  S3PublicAccessCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: s3-bucket-public-access-prohibited
      Source:
        Owner: AWS
        SourceIdentifier: S3_BUCKET_PUBLIC_READ_PROHIBITED

  # 暗号化されていない EBS ボリュームの検出
  EbsEncryptionCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: ebs-encryption-by-default
      Source:
        Owner: AWS
        SourceIdentifier: EC2_EBS_ENCRYPTION_BY_DEFAULT

  # MFA 削除が有効でない S3 バケット
  S3MfaDeleteCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: s3-bucket-versioning-enabled
      Source:
        Owner: AWS
        SourceIdentifier: S3_BUCKET_VERSIONING_ENABLED

  # ルートアカウントの MFA 確認
  RootMfaCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: root-account-mfa-enabled
      Source:
        Owner: AWS
        SourceIdentifier: ROOT_ACCOUNT_MFA_ENABLED

  # === 信頼性の柱 ===

  # RDS のマルチ AZ 確認
  RdsMultiAzCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: rds-multi-az-support
      Source:
        Owner: AWS
        SourceIdentifier: RDS_MULTI_AZ_SUPPORT

  # RDS の自動バックアップ確認
  RdsBackupCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: rds-automatic-backup-enabled
      Source:
        Owner: AWS
        SourceIdentifier: DB_INSTANCE_BACKUP_ENABLED
      InputParameters:
        backupRetentionMinimum: "7"

  # ELB の Cross-Zone Load Balancing 確認
  ElbCrossZoneCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: elb-cross-zone-load-balancing
      Source:
        Owner: AWS
        SourceIdentifier: ELB_CROSS_ZONE_LOAD_BALANCING_ENABLED

  # === コスト最適化の柱 ===

  # 未使用 EIP の検出
  UnusedEipCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: eip-attached
      Source:
        Owner: AWS
        SourceIdentifier: EIP_ATTACHED

  # 未使用 EBS ボリュームの検出
  UnusedEbsCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: ec2-volume-inuse-check
      Source:
        Owner: AWS
        SourceIdentifier: EC2_VOLUME_INUSE_CHECK

  # === 運用上の優秀性の柱 ===

  # CloudTrail の有効化確認
  CloudTrailEnabledCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: cloudtrail-enabled
      Source:
        Owner: AWS
        SourceIdentifier: CLOUD_TRAIL_ENABLED

  # CloudWatch ログの保持期間確認
  CloudWatchLogRetentionCheck:
    Type: AWS::Config::ConfigRule
    Properties:
      ConfigRuleName: cw-loggroup-retention-period-check
      Source:
        Owner: AWS
        SourceIdentifier: CW_LOGGROUP_RETENTION_PERIOD_CHECK
      InputParameters:
        MinRetentionTime: "90"

  # === コンプライアンス変更の自動通知 ===
  ComplianceChangeRule:
    Type: AWS::Events::Rule
    Properties:
      Name: wa-compliance-change
      EventPattern:
        source:
          - aws.config
        detail-type:
          - Config Rules Compliance Change
        detail:
          newEvaluationResult:
            complianceType:
              - NON_COMPLIANT
      Targets:
        - Arn: !Ref ComplianceTopic
          Id: ComplianceNotification

  ComplianceTopicPolicy:
    Type: AWS::SNS::TopicPolicy
    Properties:
      Topics:
        - !Ref ComplianceTopic
      PolicyDocument:
        Statement:
          - Effect: Allow
            Principal:
              Service: events.amazonaws.com
            Action: sns:Publish
            Resource: !Ref ComplianceTopic

Outputs:
  ComplianceTopicArn:
    Value: !Ref ComplianceTopic
    Description: SNS topic for compliance alerts
```

---

## 6. Well-Architected レビュー自動化パイプライン

### 6.1 CI/CD パイプラインとの統合

```python
# 図10: GitHub Actions / CodePipeline でのレビュー自動化
import boto3
import json
import sys

def ci_well_architected_gate(workload_id: str, max_high_risk: int = 0):
    """CI/CD パイプラインのゲートとして Well-Architected チェックを実行

    High Risk が閾値を超えた場合にデプロイをブロック
    """
    wa = boto3.client("wellarchitected", region_name="ap-northeast-1")

    review = wa.get_lens_review(
        WorkloadId=workload_id,
        LensAlias="wellarchitected",
    )["LensReview"]

    risk_counts = review.get("RiskCounts", {})
    high_risk = risk_counts.get("HIGH", 0)
    medium_risk = risk_counts.get("MEDIUM", 0)
    unanswered = risk_counts.get("UNANSWERED", 0)

    print(f"=== Well-Architected Gate Check ===")
    print(f"Workload: {workload_id}")
    print(f"HIGH: {high_risk}, MEDIUM: {medium_risk}, UNANSWERED: {unanswered}")

    # High Risk の詳細を取得
    if high_risk > 0:
        print(f"\n--- High Risk Details ---")
        pillars = ["operationalExcellence", "security", "reliability",
                    "performance", "costOptimization", "sustainability"]
        for pillar in pillars:
            improvements = wa.list_lens_review_improvements(
                WorkloadId=workload_id,
                LensAlias="wellarchitected",
                PillarId=pillar,
            )
            for item in improvements["ImprovementSummaries"]:
                if item.get("Risk") == "HIGH":
                    print(f"  [{pillar}] {item['QuestionTitle']}")

    # ゲート判定
    if high_risk > max_high_risk:
        print(f"\n❌ GATE FAILED: {high_risk} high risk items (max: {max_high_risk})")
        sys.exit(1)
    else:
        print(f"\n✅ GATE PASSED: {high_risk} high risk items within threshold")
        return True
```

```yaml
# GitHub Actions でのパイプライン統合例
# .github/workflows/well-architected-gate.yml
name: Well-Architected Gate

on:
  pull_request:
    branches: [main]
    paths:
      - 'infrastructure/**'
      - 'cdk/**'

jobs:
  wa-check:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-wa-check
          aws-region: ap-northeast-1

      - name: Well-Architected Gate Check
        run: |
          python3 -c "
          import boto3
          wa = boto3.client('wellarchitected')
          review = wa.get_lens_review(
              WorkloadId='${{ vars.WA_WORKLOAD_ID }}',
              LensAlias='wellarchitected',
          )['LensReview']
          risk = review.get('RiskCounts', {})
          high = risk.get('HIGH', 0)
          print(f'High Risk: {high}')
          if high > 0:
              print('::error::Well-Architected review has HIGH risk items')
              exit(1)
          "
```

---

## 7. 比較表

### 7.1 6 つの柱 概要比較

| 柱 | 焦点 | 主要 AWS サービス | KPI 例 |
|----|------|------------------|--------|
| **運用上の優秀性** | 運用の自動化と継続改善 | CloudFormation, Systems Manager, CloudWatch | デプロイ頻度, MTTR |
| **セキュリティ** | データと資産の保護 | IAM, KMS, GuardDuty, Security Hub | 未対応の検出結果数 |
| **信頼性** | 障害復旧と可用性 | Route 53, ELB, Auto Scaling, Backup | 可用性 %, RTO/RPO |
| **パフォーマンス効率** | リソースの効率的な使用 | CloudFront, ElastiCache, Lambda | レイテンシ P99 |
| **コスト最適化** | 無駄の排除と価値最大化 | Cost Explorer, Budgets, Savings Plans | 月間コスト, SP カバー率 |
| **持続可能性** | 環境への影響最小化 | Graviton, Spot, サーバーレス | CO2 排出量推定 |

### 7.2 レビュー方式比較

| 方式 | 対象 | 所要時間 | コスト | 推奨場面 |
|------|------|---------|--------|---------|
| **セルフレビュー** | 自チーム | 2-5日 | 無料 | 定期レビュー |
| **AWS SA レビュー** | SA 支援 | 1-2週間 | 無料（Enterprise Support） | 初回レビュー |
| **パートナーレビュー** | APN パートナー | 2-4週間 | 有料 | 大規模ワークロード |
| **AWS Well-Architected Tool** | ツール支援 | 1-3日 | 無料 | 全ケースで利用推奨 |

### 7.3 Lens 比較

| Lens | 対象ワークロード | 追加の柱 | 質問数（目安） | 推奨場面 |
|------|----------------|---------|-------------|---------|
| **Well-Architected (標準)** | 全般 | なし | ~58 | 全ワークロード |
| **Serverless** | Lambda, API GW, DynamoDB | なし | ~30 | サーバーレスアプリ |
| **SaaS** | マルチテナント SaaS | テナント分離 | ~40 | SaaS プロバイダー |
| **Machine Learning** | ML ワークロード | ML ライフサイクル | ~35 | ML/AI アプリ |
| **Data Analytics** | データ分析基盤 | データ品質 | ~30 | データレイク, ETL |
| **Container** | ECS, EKS | コンテナ運用 | ~25 | コンテナワークロード |
| **IoT** | IoT デバイス管理 | デバイス管理 | ~30 | IoT プラットフォーム |
| **FTR (Foundational Technical Review)** | APN パートナー | なし | ~50 | パートナー認定 |
| **カスタム Lens** | 自社基準 | 任意 | 任意 | 社内標準の強制 |

### 7.4 Well-Architected vs 他のフレームワーク

| 項目 | AWS Well-Architected | TOGAF | ITIL | ISO 27001 |
|------|---------------------|-------|------|-----------|
| **焦点** | クラウドアーキテクチャ | エンタープライズ全体 | IT サービス管理 | 情報セキュリティ |
| **スコープ** | AWS ワークロード | 組織横断 | IT 運用プロセス | セキュリティ管理 |
| **ツール** | WA Tool (無料) | 商用ツール | 商用ツール | 認証機関 |
| **更新頻度** | 随時（AWS サービス追加） | 数年ごと | 数年ごと | 定期改訂 |
| **コスト** | 無料 | ライセンス費 | トレーニング費 | 認証費用 |
| **認証/資格** | なし（自己評価） | TOGAF 認定 | ITIL 認定 | ISO 認証 |
| **クラウド対応** | AWS ネイティブ | クラウド非依存 | クラウド非依存 | クラウド非依存 |
| **補完関係** | - | 全体設計→WA で詳細化 | 運用プロセス補完 | セキュリティの柱を詳細化 |

---

## 8. アンチパターン

### 8.1 レビューを一度やって終わり

```
NG:
  リリース前に Well-Architected レビュー実施
  → "完了" として棚上げ
  → 1年後: アーキテクチャが変わり、リスクが再発

OK:
  四半期ごとの定期レビューサイクル
  ┌──────────────────────────────────┐
  │  Q1: フルレビュー                │
  │  Q2: High Risk 改善確認          │
  │  Q3: フルレビュー（再評価）      │
  │  Q4: 年間振り返り + 次年度計画   │
  └──────────────────────────────────┘
```

### 8.2 全ての柱を均等に扱う

```
NG:
  6つの柱 × 均等リソース配分
  → セキュリティのクリティカルな問題が後回しに

OK:
  リスクベースの優先順位付け
  1. セキュリティの High Risk → 即対応（1-2週間）
  2. 信頼性の High Risk → 次スプリント
  3. 運用の Medium Risk → バックログ
  4. コスト/パフォーマンス → 四半期計画
```

### 8.3 チェックリストとして形式的に実施する

```
NG:
  - 全質問に "はい" を選択して形式的に完了
  - リスクを「受容」として全てクローズ
  - レビュー結果を共有せず担当者だけが把握

OK:
  - 各質問に対してエビデンス（設定画面のスクショ、IaC コード）を添付
  - リスク受容には経営層の承認プロセスを設ける
  - レビュー結果をチーム全体に共有し、改善を透明化

  エビデンスの例:
  ┌────────────────────────────────────────────────────┐
  │ 質問: データは暗号化されていますか？                    │
  │                                                    │
  │ 回答: はい                                          │
  │                                                    │
  │ エビデンス:                                         │
  │ - RDS: storage_encrypted=true (CDK コード L.142)    │
  │ - S3: BucketEncryption SSE-KMS (CDK コード L.87)    │
  │ - EBS: encrypted=true (デフォルト設定 ON)             │
  │ - DynamoDB: SSE-KMS (CDK コード L.203)               │
  │ - Config Rule: encrypted-volumes COMPLIANT           │
  └────────────────────────────────────────────────────┘
```

### 8.4 コスト最適化を後回しにする

```
NG:
  開発完了 → 本番運用開始 → 「コストが高い」と気づく
  → 半年後にようやくコスト最適化プロジェクト開始
  → すでに数万ドルの無駄が発生

OK:
  設計段階からコスト最適化を組み込む
  ┌────────────────────────────────────────────────────┐
  │ Day 1 から実施すべきコスト最適化:                      │
  │                                                    │
  │ 1. コスト配分タグの設計と適用                         │
  │ 2. AWS Budgets の設定（80%/100% アラート）            │
  │ 3. Cost Anomaly Detection の有効化                   │
  │ 4. 開発環境の自動停止スケジュール                      │
  │ 5. S3 ライフサイクルポリシーの設定                     │
  │ 6. Compute Optimizer の有効化                        │
  └────────────────────────────────────────────────────┘
```

### 8.5 単一の柱だけに注力する

```
NG:
  セキュリティチームが主導
  → セキュリティの柱だけ詳細にレビュー
  → 信頼性やパフォーマンスのリスクを見落とし
  → 障害発生時に復旧できない

OK:
  クロスファンクショナルチームで全柱をレビュー
  ┌────────────────────────────────────────┐
  │ レビューチーム構成:                      │
  │                                        │
  │ - テックリード（運用上の優秀性）         │
  │ - セキュリティエンジニア（セキュリティ）  │
  │ - SRE（信頼性）                         │
  │ - バックエンドエンジニア（パフォーマンス） │
  │ - FinOps 担当（コスト最適化）            │
  │ - アーキテクト（持続可能性 + 全体統括）   │
  └────────────────────────────────────────┘
```

---

## 9. FAQ

### Q1. Well-Architected Review は誰が主導すべき？

**A.** ワークロードのテックリードまたはアーキテクトが主導し、開発・運用・セキュリティの各チームメンバーが参加する。AWS の Solutions Architect に初回の支援を依頼すると効率的。Enterprise Support 契約があれば無料で SA の支援を受けられる。

### Q2. 小規模なスタートアップでも Well-Architected は必要？

**A.** 規模に関わらず有用。ただし全ての質問に完璧に対応する必要はない。まずセキュリティと信頼性の High Risk 項目に集中し、ビジネスの成長に合わせて他の柱も強化していく段階的アプローチが現実的。

### Q3. Well-Architected Tool の結果は AWS に共有される？

**A.** 通常は共有されない。ただし「AWS Solutions Architect とワークロードを共有」を明示的に有効にした場合のみ、担当 SA がレビュー結果にアクセスできる。データは暗号化されアカウント所有者が管理権限を持つ。

### Q4. マルチアカウント環境でのレビューはどう進める？

**A.** Organizations の管理アカウントまたは専用のアーキテクチャアカウントで Well-Architected Tool を使い、各メンバーアカウントのワークロードを個別に登録する。共通基盤（VPC、IAM、ログ集約）は一つのワークロードとして、アプリケーションは別のワークロードとしてレビューする。AWS RAM (Resource Access Manager) を使ってレビュー結果をアカウント間で共有することも可能。

### Q5. カスタム Lens を作成するメリットは？

**A.** 自社固有のコンプライアンス要件、業界規制（PCI DSS、HIPAA 等）、社内アーキテクチャ標準をフレームワーク化できる。標準の Well-Architected Lens では対応しきれない固有の要件を体系的にレビューでき、組織全体で一貫したアーキテクチャ品質を維持できる。JSON 形式で Lens を定義し、AWS CLI でインポートする。

```bash
# カスタム Lens の作成例
aws wellarchitected import-lens \
  --json-string file://custom-lens.json \
  --tags Department=Engineering,Standard=InternalV2

# カスタム Lens の公開（組織内共有）
aws wellarchitected create-lens-share \
  --lens-alias "arn:aws:wellarchitected:ap-northeast-1:123456789012:lens/my-custom-lens" \
  --shared-with "arn:aws:organizations::123456789012:organization/o-abc123"
```

### Q6. レビューの自動化はどこまで可能？

**A.** Well-Architected Tool の API を使って、ワークロードの作成・質問への回答・マイルストーン作成・レポート取得まで完全に自動化できる。ただし、質問への回答は技術的判断を伴うため、自動回答は推奨されない。実用的なアプローチは以下の通り。

- **自動化すべき**: ワークロード作成、マイルストーン作成、レポート生成、Slack/Teams への通知
- **半自動化**: Config Rules / Security Hub の結果をもとに回答を提案
- **手動維持**: 最終的な回答判断、改善計画の策定、優先順位付け

### Q7. Well-Architected レビューと SOC 2 / ISO 27001 監査の関係は？

**A.** Well-Architected レビューは自己評価のフレームワークであり、監査や認証ではない。ただし、セキュリティの柱のベストプラクティスは SOC 2 や ISO 27001 の要件と大きく重複する。Well-Architected レビューを定期的に実施し、エビデンスを記録しておくことで、監査対応時の準備工数を大幅に削減できる。特に以下の領域が重複する。

| Well-Architected | SOC 2 | ISO 27001 |
|-----------------|-------|-----------|
| IAM / MFA | CC6.1 論理アクセス | A.9 アクセス制御 |
| 暗号化 | CC6.7 暗号化 | A.10 暗号 |
| ログ / 監査証跡 | CC7.2 モニタリング | A.12.4 ログ取得 |
| バックアップ / DR | A1.2 事業継続 | A.17 事業継続管理 |
| インシデント対応 | CC7.3 インシデント | A.16 インシデント管理 |

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| **6 つの柱** | 運用、セキュリティ、信頼性、パフォーマンス、コスト、持続可能性 |
| **レビューツール** | AWS Well-Architected Tool で質問に回答し、リスクを可視化 |
| **優先順位** | セキュリティ > 信頼性 > 運用 > パフォーマンス > コスト > 持続可能性 |
| **継続性** | 四半期ごとの定期レビュー、マイルストーンで進捗管理 |
| **Lens** | ワークロード種別に応じた専用 Lens を活用 |
| **自動化** | Config Rules + Security Hub + Trusted Advisor で継続的コンプライアンス |
| **CI/CD 統合** | パイプラインのゲートとして High Risk チェックを組み込む |
| **エビデンス** | 各質問にIaCコード・設定スクリーンショット等のエビデンスを記録 |

---

## 次に読むべきガイド

- [00-cost-optimization.md](./00-cost-optimization.md) — コスト最適化の具体的な実践
- セキュリティガイド — IAM / KMS / WAF の詳細設計
- 信頼性ガイド — マルチ AZ / DR 戦略

---

## 参考文献

1. **AWS公式ドキュメント** — "AWS Well-Architected Framework" — https://docs.aws.amazon.com/wellarchitected/latest/framework/
2. **AWS公式ドキュメント** — "AWS Well-Architected Tool User Guide" — https://docs.aws.amazon.com/wellarchitected/latest/userguide/
3. **AWS公式ホワイトペーパー** — "AWS Well-Architected Framework: Six Pillars" — https://aws.amazon.com/architecture/well-architected/
4. **AWS公式ブログ** — "Well-Architected Labs" — https://www.wellarchitectedlabs.com/
5. **AWS公式ドキュメント** — "Operational Excellence Pillar" — https://docs.aws.amazon.com/wellarchitected/latest/operational-excellence-pillar/
6. **AWS公式ドキュメント** — "Security Pillar" — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
7. **AWS公式ドキュメント** — "Reliability Pillar" — https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/
8. **AWS公式ドキュメント** — "Performance Efficiency Pillar" — https://docs.aws.amazon.com/wellarchitected/latest/performance-efficiency-pillar/
9. **AWS公式ドキュメント** — "Cost Optimization Pillar" — https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/
10. **AWS公式ドキュメント** — "Sustainability Pillar" — https://docs.aws.amazon.com/wellarchitected/latest/sustainability-pillar/
