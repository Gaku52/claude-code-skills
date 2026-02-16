# コスト最適化 — Cost Explorer / Budgets / Savings Plans

> AWS のコストを可視化・予測・削減するための実践ガイド。無駄な支出を特定し、Savings Plans やリザーブドインスタンスで最大 72% のコスト削減を実現する。

---

## この章で学ぶこと

1. **Cost Explorer** によるコスト分析とトレンド把握
2. **AWS Budgets** によるアラートと自動アクション
3. **Savings Plans / Reserved Instances** による長期コミット型割引
4. **Cost & Usage Report (CUR)** による詳細分析
5. **Compute Optimizer** によるライトサイジング
6. **Trusted Advisor** による未使用リソース検出
7. **組織的なコスト最適化文化** の醸成

---

## 1. コスト最適化の全体フレームワーク

### 1.1 4 つのステップ

```
┌──────────────────────────────────────────────────────┐
│           AWS コスト最適化フレームワーク               │
│                                                      │
│  Step 1: 可視化 (See)                                │
│  ┌──────────────────────────────────────┐            │
│  │ Cost Explorer / Cost & Usage Report  │            │
│  │ → どこにいくら使っているか把握       │            │
│  └──────────────────────┬───────────────┘            │
│                         ▼                            │
│  Step 2: 分析 (Analyze)                              │
│  ┌──────────────────────────────────────┐            │
│  │ Trusted Advisor / Compute Optimizer  │            │
│  │ → 無駄なリソースを特定               │            │
│  └──────────────────────┬───────────────┘            │
│                         ▼                            │
│  Step 3: 削減 (Optimize)                             │
│  ┌──────────────────────────────────────┐            │
│  │ Right Sizing / Savings Plans / Spot  │            │
│  │ → 適切なサイズとプラン選択           │            │
│  └──────────────────────┬───────────────┘            │
│                         ▼                            │
│  Step 4: 監視 (Monitor)                              │
│  ┌──────────────────────────────────────┐            │
│  │ Budgets / Anomaly Detection          │            │
│  │ → 継続的な監視と異常検知             │            │
│  └──────────────────────────────────────┘            │
└──────────────────────────────────────────────────────┘
```

### 1.2 コスト配分タグ戦略

```
┌─────────────────────────────────────────────┐
│          タグによるコスト配分                 │
│                                             │
│  必須タグ:                                  │
│  ┌────────────────┬───────────────────┐     │
│  │ タグキー        │ 値の例            │     │
│  ├────────────────┼───────────────────┤     │
│  │ Environment    │ prod/staging/dev  │     │
│  │ Project        │ myapp/api/data    │     │
│  │ Team           │ backend/frontend  │     │
│  │ CostCenter     │ CC-001/CC-002     │     │
│  └────────────────┴───────────────────┘     │
│                                             │
│  → Billing > Cost Allocation Tags で有効化  │
│  → 24時間後から Cost Explorer で利用可能     │
└─────────────────────────────────────────────┘
```

### 1.3 コスト最適化の成熟度モデル

組織のコスト最適化への取り組みは、以下の成熟度レベルで評価できる。

```
┌──────────────────────────────────────────────────────────────┐
│           コスト最適化 成熟度モデル                             │
│                                                              │
│  Level 1: 反応的 (Reactive)                                  │
│  ├─ 月末の請求書を見て初めてコストを認識                       │
│  ├─ タグ付けは不完全、コスト配分不能                          │
│  └─ コスト削減は ad-hoc で属人的                              │
│                                                              │
│  Level 2: 可視化 (Visible)                                   │
│  ├─ Cost Explorer を定期的に確認                              │
│  ├─ 主要プロジェクトにはタグが付与済み                        │
│  └─ 月次のコストレポートを共有                                │
│                                                              │
│  Level 3: プロアクティブ (Proactive)                          │
│  ├─ Budgets + アラートが全環境に設定済み                      │
│  ├─ Savings Plans / RI の戦略的購入                           │
│  ├─ Compute Optimizer の推奨を定期適用                        │
│  └─ コスト異常検知が自動化                                    │
│                                                              │
│  Level 4: 最適化 (Optimized)                                 │
│  ├─ FinOps チーム / Cloud COE が存在                         │
│  ├─ CUR + Athena + QuickSight で独自分析基盤                 │
│  ├─ チーム別のコスト予算と KPI が設定                         │
│  └─ アーキテクチャ選定時にコスト効率を定量評価                │
│                                                              │
│  Level 5: ビジネス価値駆動 (Value-Driven)                    │
│  ├─ ユニットエコノミクス（ユーザーあたりコスト等）を追跡      │
│  ├─ コスト効率がビジネス KPI として経営レベルで管理           │
│  └─ 自動化されたコスト最適化パイプライン                      │
└──────────────────────────────────────────────────────────────┘
```

### 1.4 FinOps の原則

FinOps（クラウド財務管理）の実践原則を以下に示す。

```yaml
# FinOps の 6 つの原則
finops_principles:
  - name: "チームの協力"
    description: "財務、技術、ビジネスチームが協力してコストを管理"
    actions:
      - "月次 FinOps レビュー会議を開催"
      - "各チームにコストオーナーを任命"
      - "コスト情報を全チームに透明化"

  - name: "ビジネス価値に基づく意思決定"
    description: "コスト削減だけでなく、ビジネス価値とのバランスを重視"
    actions:
      - "ユニットエコノミクスを定義（例: 1トランザクションあたりコスト）"
      - "コスト最適化の ROI を計算してから実行"

  - name: "クラウドの変動費モデルを活用"
    description: "オンデマンドの柔軟性を最大限活用"
    actions:
      - "開発環境はスケジュール停止で変動費化"
      - "バースト需要にはスポットインスタンスを活用"

  - name: "全員がコスト責任者"
    description: "エンジニア全員がコスト意識を持つ文化"
    actions:
      - "PR レビューでコスト影響を確認"
      - "コストダッシュボードをチーム全員に共有"
      - "コスト異常を検知したらすぐ報告するプロセス"

  - name: "タイムリーなレポート"
    description: "リアルタイムに近いコスト情報を提供"
    actions:
      - "日次のコストレポートを自動送信"
      - "異常検知アラートをリアルタイムで通知"

  - name: "集中管理と分散実行"
    description: "ガバナンスは集中、最適化は各チームが実行"
    actions:
      - "全社的なタグ付けポリシーを策定"
      - "Savings Plans は集中購入、リソース最適化は各チーム"
```

---

## 2. Cost Explorer

### 2.1 CLI でのコスト分析

```bash
# 月別サービスコスト取得（直近3ヶ月）
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-03-01 \
  --granularity MONTHLY \
  --metrics "BlendedCost" "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output json | jq '.ResultsByTime[] | {
    period: .TimePeriod,
    costs: [.Groups[] | select(.Metrics.BlendedCost.Amount | tonumber > 10) | {
      service: .Keys[0],
      cost: .Metrics.BlendedCost.Amount
    }]
  }'

# タグ別コスト分析
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=TAG,Key=Project \
  --filter '{
    "Tags": {
      "Key": "Environment",
      "Values": ["prod"],
      "MatchOptions": ["EQUALS"]
    }
  }'

# 日次でのコスト取得（直近7日間）
START_DATE=$(date -d "7 days ago" +%Y-%m-%d 2>/dev/null || date -v-7d +%Y-%m-%d)
END_DATE=$(date +%Y-%m-%d)
aws ce get-cost-and-usage \
  --time-period Start=${START_DATE},End=${END_DATE} \
  --granularity DAILY \
  --metrics "UnblendedCost" \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output json | jq '[.ResultsByTime[] | {
    date: .TimePeriod.Start,
    total: ([.Groups[].Metrics.UnblendedCost.Amount | tonumber] | add | . * 100 | round / 100),
    top_services: [.Groups | sort_by(-.Metrics.UnblendedCost.Amount | tonumber) | .[:5][] | {
      service: .Keys[0],
      cost: (.Metrics.UnblendedCost.Amount | tonumber | . * 100 | round / 100)
    }]
  }]'

# リージョン別コスト分析
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --metrics "BlendedCost" \
  --group-by Type=DIMENSION,Key=REGION \
  --output json | jq '.ResultsByTime[0].Groups | sort_by(-.Metrics.BlendedCost.Amount | tonumber) | .[] | {
    region: .Keys[0],
    cost_usd: (.Metrics.BlendedCost.Amount | tonumber | . * 100 | round / 100)
  }'

# 使用量タイプ別のコスト分析（データ転送量など）
aws ce get-cost-and-usage \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --metrics "BlendedCost" "UsageQuantity" \
  --filter '{
    "Dimensions": {
      "Key": "SERVICE",
      "Values": ["Amazon Elastic Compute Cloud - Compute"]
    }
  }' \
  --group-by Type=DIMENSION,Key=USAGE_TYPE
```

### 2.2 Python でのコスト分析自動化

```python
import boto3
from datetime import datetime, timedelta
from typing import Optional
import json


def get_monthly_cost_by_service(months: int = 3) -> list[dict]:
    """サービス別月次コストを取得"""
    client = boto3.client("ce", region_name="us-east-1")

    end = datetime.now().replace(day=1)
    start = (end - timedelta(days=months * 30)).replace(day=1)

    response = client.get_cost_and_usage(
        TimePeriod={
            "Start": start.strftime("%Y-%m-%d"),
            "End": end.strftime("%Y-%m-%d"),
        },
        Granularity="MONTHLY",
        Metrics=["BlendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    results = []
    for period in response["ResultsByTime"]:
        month = period["TimePeriod"]["Start"]
        for group in period["Groups"]:
            cost = float(group["Metrics"]["BlendedCost"]["Amount"])
            if cost > 1.0:  # $1 以上のサービスのみ
                results.append({
                    "month": month,
                    "service": group["Keys"][0],
                    "cost_usd": round(cost, 2),
                })
    return results


def get_cost_forecast(days: int = 30) -> dict:
    """コスト予測を取得"""
    client = boto3.client("ce", region_name="us-east-1")

    start = datetime.now().strftime("%Y-%m-%d")
    end = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")

    response = client.get_cost_forecast(
        TimePeriod={"Start": start, "End": end},
        Metric="BLENDED_COST",
        Granularity="MONTHLY",
    )

    return {
        "forecast_usd": float(response["Total"]["Amount"]),
        "confidence_80_low": float(response["Total"].get("Amount", 0)),
    }


def get_cost_by_tag(
    tag_key: str,
    months: int = 1,
    environment: Optional[str] = None,
) -> list[dict]:
    """タグ別のコスト分析"""
    client = boto3.client("ce", region_name="us-east-1")

    end = datetime.now().replace(day=1)
    start = (end - timedelta(days=months * 30)).replace(day=1)

    params = {
        "TimePeriod": {
            "Start": start.strftime("%Y-%m-%d"),
            "End": end.strftime("%Y-%m-%d"),
        },
        "Granularity": "MONTHLY",
        "Metrics": ["BlendedCost", "UnblendedCost"],
        "GroupBy": [{"Type": "TAG", "Key": tag_key}],
    }

    if environment:
        params["Filter"] = {
            "Tags": {
                "Key": "Environment",
                "Values": [environment],
                "MatchOptions": ["EQUALS"],
            }
        }

    response = client.get_cost_and_usage(**params)

    results = []
    for period in response["ResultsByTime"]:
        month = period["TimePeriod"]["Start"]
        for group in period["Groups"]:
            tag_value = group["Keys"][0]
            if tag_value.startswith(f"{tag_key}$"):
                tag_value = tag_value.split("$", 1)[1]
            blended = float(group["Metrics"]["BlendedCost"]["Amount"])
            unblended = float(group["Metrics"]["UnblendedCost"]["Amount"])
            if blended > 0.01:
                results.append({
                    "month": month,
                    "tag_value": tag_value or "(untagged)",
                    "blended_cost": round(blended, 2),
                    "unblended_cost": round(unblended, 2),
                })
    return results


def get_daily_cost_trend(days: int = 30) -> list[dict]:
    """日次コストトレンドを取得"""
    client = boto3.client("ce", region_name="us-east-1")

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    response = client.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
    )

    results = []
    for period in response["ResultsByTime"]:
        date = period["TimePeriod"]["Start"]
        cost = float(period["Total"]["UnblendedCost"]["Amount"])
        results.append({
            "date": date,
            "cost_usd": round(cost, 2),
        })
    return results


def get_top_cost_services(
    months: int = 1,
    top_n: int = 10,
) -> list[dict]:
    """コストの高いサービス上位 N 件を取得"""
    client = boto3.client("ce", region_name="us-east-1")

    end = datetime.now().replace(day=1)
    start = (end - timedelta(days=months * 30)).replace(day=1)

    response = client.get_cost_and_usage(
        TimePeriod={
            "Start": start.strftime("%Y-%m-%d"),
            "End": end.strftime("%Y-%m-%d"),
        },
        Granularity="MONTHLY",
        Metrics=["UnblendedCost"],
        GroupBy=[{"Type": "DIMENSION", "Key": "SERVICE"}],
    )

    services = []
    for period in response["ResultsByTime"]:
        for group in period["Groups"]:
            cost = float(group["Metrics"]["UnblendedCost"]["Amount"])
            services.append({
                "service": group["Keys"][0],
                "cost_usd": round(cost, 2),
            })

    services.sort(key=lambda x: x["cost_usd"], reverse=True)
    total = sum(s["cost_usd"] for s in services)

    # 割合を計算
    for svc in services[:top_n]:
        svc["percentage"] = round(svc["cost_usd"] / total * 100, 1) if total > 0 else 0

    return services[:top_n]


def generate_cost_report(months: int = 3) -> str:
    """コストレポートを生成して Markdown 形式で返す"""
    report_lines = ["# AWS 月次コストレポート\n"]
    report_lines.append(f"レポート生成日時: {datetime.now().isoformat()}\n")

    # サービス別コスト
    report_lines.append("## サービス別コスト\n")
    costs = get_monthly_cost_by_service(months)

    if costs:
        months_set = sorted(set(c["month"] for c in costs))
        report_lines.append("| サービス | " + " | ".join(months_set) + " |")
        report_lines.append("|" + "---|" * (len(months_set) + 1))

        services = sorted(set(c["service"] for c in costs))
        for svc in services:
            row = f"| {svc} "
            for month in months_set:
                matching = [c for c in costs if c["service"] == svc and c["month"] == month]
                if matching:
                    row += f"| ${matching[0]['cost_usd']:,.2f} "
                else:
                    row += "| - "
            row += "|"
            report_lines.append(row)

    # コスト予測
    report_lines.append("\n## コスト予測\n")
    try:
        forecast = get_cost_forecast()
        report_lines.append(f"- 今後30日の予測コスト: **${forecast['forecast_usd']:,.2f}**")
    except Exception as e:
        report_lines.append(f"- 予測取得エラー: {e}")

    # Top サービス
    report_lines.append("\n## コスト上位サービス\n")
    top_services = get_top_cost_services()
    report_lines.append("| # | サービス | コスト | 割合 |")
    report_lines.append("|---|---------|--------|------|")
    for i, svc in enumerate(top_services, 1):
        report_lines.append(
            f"| {i} | {svc['service']} | ${svc['cost_usd']:,.2f} | {svc.get('percentage', 0)}% |"
        )

    return "\n".join(report_lines)
```

### 2.3 Cost Anomaly Detection（異常検知）

```bash
# 異常検知モニターの作成
aws ce create-anomaly-monitor \
  --anomaly-monitor '{
    "MonitorName": "ServiceMonitor",
    "MonitorType": "DIMENSIONAL",
    "MonitorDimension": "SERVICE"
  }'

# 通知サブスクリプションの作成
aws ce create-anomaly-subscription \
  --anomaly-subscription '{
    "SubscriptionName": "CostAnomalyAlert",
    "MonitorArnList": ["arn:aws:ce::123456789012:anomalymonitor/monitor-id"],
    "Subscribers": [
      {
        "Address": "team@example.com",
        "Type": "EMAIL"
      }
    ],
    "Threshold": 100,
    "Frequency": "DAILY"
  }'

# タグベースの異常検知モニター（プロジェクト別）
aws ce create-anomaly-monitor \
  --anomaly-monitor '{
    "MonitorName": "ProjectCostMonitor",
    "MonitorType": "CUSTOM",
    "MonitorSpecification": {
      "Tags": {
        "Key": "Project",
        "Values": ["myapp", "api-service", "data-pipeline"],
        "MatchOptions": ["EQUALS"]
      }
    }
  }'

# 異常検知結果の取得
aws ce get-anomalies \
  --date-interval '{"StartDate": "2026-01-01", "EndDate": "2026-02-01"}' \
  --output json | jq '.Anomalies[] | {
    id: .AnomalyId,
    start: .AnomalyStartDate,
    end: .AnomalyEndDate,
    service: .DimensionValue,
    impact: .Impact.TotalImpact,
    expected: .Impact.TotalExpectedSpend,
    actual: .Impact.TotalActualSpend
  }'
```

### 2.4 Terraform での Cost Anomaly Detection 設定

```hcl
# 異常検知モニター
resource "aws_ce_anomaly_monitor" "service_monitor" {
  name              = "service-cost-anomaly-monitor"
  monitor_type      = "DIMENSIONAL"
  monitor_dimension = "SERVICE"

  tags = {
    Environment = "management"
    Purpose     = "cost-optimization"
  }
}

resource "aws_ce_anomaly_monitor" "project_monitor" {
  name         = "project-cost-anomaly-monitor"
  monitor_type = "CUSTOM"

  monitor_specification = jsonencode({
    Tags = {
      Key          = "Project"
      Values       = ["myapp", "api-service"]
      MatchOptions = ["EQUALS"]
    }
  })
}

# 異常検知サブスクリプション（Slack 通知用 SNS 連携）
resource "aws_ce_anomaly_subscription" "cost_alerts" {
  name = "cost-anomaly-alerts"

  monitor_arn_list = [
    aws_ce_anomaly_monitor.service_monitor.arn,
    aws_ce_anomaly_monitor.project_monitor.arn,
  ]

  subscriber {
    type    = "SNS"
    address = aws_sns_topic.cost_alerts.arn
  }

  # 影響額 $50 以上のみ通知
  threshold_expression {
    dimension {
      key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
      values        = ["50"]
      match_options = ["GREATER_THAN_OR_EQUAL"]
    }
  }

  frequency = "DAILY"
}

resource "aws_sns_topic" "cost_alerts" {
  name = "cost-anomaly-alerts"
}

# Slack 連携用の Lambda サブスクリプション
resource "aws_sns_topic_subscription" "slack_notification" {
  topic_arn = aws_sns_topic.cost_alerts.arn
  protocol  = "lambda"
  endpoint  = aws_lambda_function.slack_notifier.arn
}
```

### 2.5 Slack 通知用 Lambda 関数

```python
"""
Cost Anomaly Detection の結果を Slack に通知する Lambda 関数。
SNS トピック経由でトリガーされる。
"""
import json
import os
import urllib.request
from datetime import datetime


SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL", "#cost-alerts")


def lambda_handler(event, context):
    """SNS からの Cost Anomaly 通知を Slack に転送"""
    for record in event["Records"]:
        message = json.loads(record["Sns"]["Message"])

        anomaly_id = message.get("anomalyId", "N/A")
        start_date = message.get("anomalyStartDate", "N/A")
        end_date = message.get("anomalyEndDate", "N/A")
        service = message.get("dimensionValue", "N/A")

        impact = message.get("impact", {})
        total_impact = float(impact.get("totalImpact", 0))
        expected_spend = float(impact.get("totalExpectedSpend", 0))
        actual_spend = float(impact.get("totalActualSpend", 0))

        # Slack メッセージを構築
        color = "#ff0000" if total_impact > 100 else "#ffaa00"
        slack_message = {
            "channel": SLACK_CHANNEL,
            "username": "AWS Cost Anomaly Alert",
            "icon_emoji": ":money_with_wings:",
            "attachments": [
                {
                    "color": color,
                    "title": f"Cost Anomaly Detected: {service}",
                    "fields": [
                        {
                            "title": "Impact",
                            "value": f"${total_impact:,.2f}",
                            "short": True,
                        },
                        {
                            "title": "Period",
                            "value": f"{start_date} ~ {end_date}",
                            "short": True,
                        },
                        {
                            "title": "Expected Spend",
                            "value": f"${expected_spend:,.2f}",
                            "short": True,
                        },
                        {
                            "title": "Actual Spend",
                            "value": f"${actual_spend:,.2f}",
                            "short": True,
                        },
                    ],
                    "footer": f"Anomaly ID: {anomaly_id}",
                    "ts": int(datetime.now().timestamp()),
                }
            ],
        }

        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=json.dumps(slack_message).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req)

    return {"statusCode": 200, "body": "Notification sent"}
```

---

## 3. AWS Budgets

### 3.1 予算の作成と自動アクション

```bash
# 月次予算の作成（$1,000、80% と 100% でアラート）
aws budgets create-budget \
  --account-id 123456789012 \
  --budget '{
    "BudgetName": "MonthlyBudget",
    "BudgetLimit": {
      "Amount": "1000",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {},
    "CostTypes": {
      "IncludeTax": true,
      "IncludeSubscription": true,
      "UseBlended": false
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
          "Address": "team@example.com"
        }
      ]
    },
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 100,
        "ThresholdType": "PERCENTAGE"
      },
      "Subscribers": [
        {
          "SubscriptionType": "SNS",
          "Address": "arn:aws:sns:ap-northeast-1:123456789012:budget-alerts"
        }
      ]
    }
  ]'

# サービス別予算の作成（EC2 のみ）
aws budgets create-budget \
  --account-id 123456789012 \
  --budget '{
    "BudgetName": "EC2-MonthlyBudget",
    "BudgetLimit": {
      "Amount": "500",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {
      "Service": ["Amazon Elastic Compute Cloud - Compute"]
    },
    "CostTypes": {
      "IncludeTax": true,
      "IncludeSubscription": true,
      "UseBlended": false
    }
  }' \
  --notifications-with-subscribers '[
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
          "Address": "team@example.com"
        }
      ]
    }
  ]'

# 予算一覧の取得
aws budgets describe-budgets \
  --account-id 123456789012 \
  --output json | jq '.Budgets[] | {
    name: .BudgetName,
    limit: .BudgetLimit,
    actual: .CalculatedSpend.ActualSpend,
    forecast: .CalculatedSpend.ForecastedSpend,
    time_unit: .TimeUnit
  }'
```

### 3.2 Terraform での Budget 定義

```hcl
resource "aws_budgets_budget" "monthly" {
  name         = "monthly-total-budget"
  budget_type  = "COST"
  limit_amount = "1000"
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Environment$prod"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["team@example.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }
}

# 予算超過時の自動アクション（EC2 停止）
resource "aws_budgets_budget_action" "stop_ec2" {
  budget_name        = aws_budgets_budget.monthly.name
  action_type        = "RUN_SSM_DOCUMENTS"
  approval_model     = "AUTOMATIC"
  notification_type  = "ACTUAL"

  action_threshold {
    action_threshold_type  = "PERCENTAGE"
    action_threshold_value = 120
  }

  definition {
    ssm_action_definition {
      action_sub_type = "STOP_EC2_INSTANCES"
      instance_ids    = ["i-0123456789abcdef0"]
      region          = "ap-northeast-1"
    }
  }

  subscriber {
    address           = "team@example.com"
    subscription_type = "EMAIL"
  }
}

# サービス別予算の一括定義
variable "service_budgets" {
  type = map(object({
    limit_amount = string
    service_name = string
  }))
  default = {
    ec2 = {
      limit_amount = "500"
      service_name = "Amazon Elastic Compute Cloud - Compute"
    }
    rds = {
      limit_amount = "300"
      service_name = "Amazon Relational Database Service"
    }
    s3 = {
      limit_amount = "100"
      service_name = "Amazon Simple Storage Service"
    }
    lambda = {
      limit_amount = "50"
      service_name = "AWS Lambda"
    }
  }
}

resource "aws_budgets_budget" "service" {
  for_each = var.service_budgets

  name         = "${each.key}-monthly-budget"
  budget_type  = "COST"
  limit_amount = each.value.limit_amount
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = [each.value.service_name]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["team@example.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = ["team@example.com"]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }
}

# チーム別予算（タグベース）
variable "team_budgets" {
  type = map(string)
  default = {
    backend  = "2000"
    frontend = "500"
    data     = "1500"
    ml       = "3000"
  }
}

resource "aws_budgets_budget" "team" {
  for_each = var.team_budgets

  name         = "team-${each.key}-monthly-budget"
  budget_type  = "COST"
  limit_amount = each.value
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "TagKeyValue"
    values = ["user:Team$${each.key}"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["${each.key}-lead@example.com"]
  }
}
```

### 3.3 Budget Action による自動コスト制御

予算超過時に自動的にリソースを制御する仕組みを構築する。

```python
"""
Budget アクションの設定と管理を行うスクリプト。
予算超過時に開発環境の EC2 を自動停止する。
"""
import boto3
import json


def setup_budget_auto_action(
    account_id: str,
    budget_name: str,
    threshold_percentage: float = 100.0,
    target_instances: list[str] = None,
    region: str = "ap-northeast-1",
) -> dict:
    """予算超過時の自動アクションを設定"""
    budgets_client = boto3.client("budgets", region_name="us-east-1")
    iam_client = boto3.client("iam")

    # Budget Action 用の IAM ロールを作成
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "budgets.amazonaws.com"
                },
                "Action": "sts:AssumeRole",
            }
        ],
    }

    role_name = f"BudgetAction-{budget_name}-Role"

    try:
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"Role for budget action: {budget_name}",
        )
    except iam_client.exceptions.EntityAlreadyExistsException:
        role = iam_client.get_role(RoleName=role_name)

    # EC2 停止権限を付与
    ec2_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": ["ec2:StopInstances", "ec2:DescribeInstances"],
                "Resource": "*",
                "Condition": {
                    "StringEquals": {
                        "ec2:ResourceTag/Environment": "dev"
                    }
                },
            }
        ],
    }

    iam_client.put_role_policy(
        RoleName=role_name,
        PolicyName="StopDevEC2Instances",
        PolicyDocument=json.dumps(ec2_policy),
    )

    # Budget Action を作成
    action = budgets_client.create_budget_action(
        AccountId=account_id,
        BudgetName=budget_name,
        NotificationType="ACTUAL",
        ActionType="RUN_SSM_DOCUMENTS",
        ActionThreshold={
            "ActionThresholdValue": threshold_percentage,
            "ActionThresholdType": "PERCENTAGE",
        },
        Definition={
            "SsmActionDefinition": {
                "ActionSubType": "STOP_EC2_INSTANCES",
                "Region": region,
                "InstanceIds": target_instances or [],
            }
        },
        ExecutionRoleArn=role["Role"]["Arn"],
        ApprovalModel="AUTOMATIC",
        Subscribers=[
            {
                "SubscriptionType": "EMAIL",
                "Address": "admin@example.com",
            }
        ],
    )

    return {
        "action_id": action["ActionId"],
        "budget_name": budget_name,
        "threshold": threshold_percentage,
        "role_arn": role["Role"]["Arn"],
    }
```

---

## 4. Savings Plans と Reserved Instances

### 4.1 選択フロー

```
コスト削減したい
     │
     ├─ EC2 のみ？ 他のコンピュートも？
     │     │
     │     ├─ EC2 のみ
     │     │     │
     │     │     ├─ インスタンスファミリー固定可 → EC2 Instance Savings Plans
     │     │     │                                 (最大 72% 割引)
     │     │     │
     │     │     └─ リージョン・OS も固定可 → Reserved Instances
     │     │                                 (最大 72% 割引)
     │     │
     │     └─ Lambda/Fargate も含む → Compute Savings Plans
     │                                (最大 66% 割引)
     │
     └─ 短期ワークロード → Spot Instances (最大 90% 割引)
                           ※ 中断リスクあり
```

### 4.2 Savings Plans の購入推奨を確認

```bash
# Savings Plans の購入推奨を取得
aws ce get-savings-plans-purchase-recommendation \
  --savings-plans-type "COMPUTE_SP" \
  --term-in-years "ONE_YEAR" \
  --payment-option "NO_UPFRONT" \
  --lookback-period-in-days "SIXTY_DAYS" \
  --output json | jq '{
    estimated_monthly_savings: .SavingsPlansPurchaseRecommendation.SavingsPlansPurchaseRecommendationSummary.EstimatedMonthlySavingsAmount,
    hourly_commitment: .SavingsPlansPurchaseRecommendation.SavingsPlansPurchaseRecommendationSummary.HourlyCommitmentToPurchase,
    coverage_percentage: .SavingsPlansPurchaseRecommendation.SavingsPlansPurchaseRecommendationSummary.CurrentOnDemandSpend
  }'

# EC2 Instance Savings Plans の推奨も確認
aws ce get-savings-plans-purchase-recommendation \
  --savings-plans-type "EC2_INSTANCE_SP" \
  --term-in-years "ONE_YEAR" \
  --payment-option "PARTIAL_UPFRONT" \
  --lookback-period-in-days "SIXTY_DAYS"

# 現在の Savings Plans カバレッジを確認
aws ce get-savings-plans-coverage \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --output json | jq '.SavingsPlansCoverages[] | {
    period: .TimePeriod,
    coverage_percentage: .Coverage.CoveragePercentage,
    spend_covered: .Coverage.SpendCoveredBySavingsPlans,
    on_demand_cost: .Coverage.OnDemandCost
  }'

# Savings Plans の使用率を確認
aws ce get-savings-plans-utilization \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --output json | jq '.SavingsPlansUtilizationsByTime[] | {
    period: .TimePeriod,
    utilization: .Utilization.UtilizationPercentage,
    total_commitment: .Utilization.TotalCommitment,
    used_commitment: .Utilization.UsedCommitment,
    unused_commitment: .Utilization.UnusedCommitment
  }'
```

### 4.3 Savings Plans 購入戦略の策定

```python
"""
Savings Plans の購入戦略を分析するスクリプト。
過去の利用実績から最適なコミットメント額を算出する。
"""
import boto3
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class SavingsPlanRecommendation:
    """Savings Plans 購入推奨"""
    sp_type: str
    term: str
    payment_option: str
    hourly_commitment: float
    estimated_monthly_savings: float
    estimated_savings_percentage: float
    roi_months: float


def analyze_savings_plan_options() -> list[SavingsPlanRecommendation]:
    """異なる Savings Plans オプションを比較分析"""
    client = boto3.client("ce", region_name="us-east-1")

    recommendations = []

    for sp_type in ["COMPUTE_SP", "EC2_INSTANCE_SP"]:
        for term in ["ONE_YEAR", "THREE_YEARS"]:
            for payment in ["NO_UPFRONT", "PARTIAL_UPFRONT", "ALL_UPFRONT"]:
                try:
                    response = client.get_savings_plans_purchase_recommendation(
                        SavingsPlansType=sp_type,
                        TermInYears=term,
                        PaymentOption=payment,
                        LookbackPeriodInDays="SIXTY_DAYS",
                    )

                    summary = response.get(
                        "SavingsPlansPurchaseRecommendation", {}
                    ).get("SavingsPlansPurchaseRecommendationSummary", {})

                    if summary:
                        monthly_savings = float(
                            summary.get("EstimatedMonthlySavingsAmount", 0)
                        )
                        hourly_commit = float(
                            summary.get("HourlyCommitmentToPurchase", 0)
                        )
                        savings_pct = float(
                            summary.get("EstimatedSavingsPercentage", 0)
                        )

                        # ROI 計算（月単位）
                        monthly_commitment = hourly_commit * 730  # 平均月間時間
                        if payment == "ALL_UPFRONT":
                            term_months = 12 if term == "ONE_YEAR" else 36
                            upfront = monthly_commitment * term_months
                            total_savings = monthly_savings * term_months
                            roi_months = (
                                upfront / monthly_savings
                                if monthly_savings > 0
                                else float("inf")
                            )
                        else:
                            roi_months = 0  # 前払いなしは即時 ROI

                        recommendations.append(
                            SavingsPlanRecommendation(
                                sp_type=sp_type,
                                term=term,
                                payment_option=payment,
                                hourly_commitment=hourly_commit,
                                estimated_monthly_savings=monthly_savings,
                                estimated_savings_percentage=savings_pct,
                                roi_months=roi_months,
                            )
                        )
                except Exception:
                    continue

    # 月次削減額でソート
    recommendations.sort(
        key=lambda r: r.estimated_monthly_savings, reverse=True
    )
    return recommendations


def calculate_optimal_commitment(
    safety_margin: float = 0.8,
) -> dict:
    """
    過去の利用実績から最適なコミットメント額を算出。
    safety_margin: ベースラインの何%をカバーするか（デフォルト80%）
    """
    client = boto3.client("ce", region_name="us-east-1")

    # 過去90日の日次コストを取得
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    response = client.get_cost_and_usage(
        TimePeriod={"Start": start, "End": end},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={
            "Dimensions": {
                "Key": "RECORD_TYPE",
                "Values": ["Usage"],
            }
        },
    )

    daily_costs = [
        float(period["Total"]["UnblendedCost"]["Amount"])
        for period in response["ResultsByTime"]
    ]

    if not daily_costs:
        return {"error": "No cost data available"}

    daily_costs.sort()

    # パーセンタイル分析
    p10 = daily_costs[int(len(daily_costs) * 0.1)]
    p25 = daily_costs[int(len(daily_costs) * 0.25)]
    p50 = daily_costs[int(len(daily_costs) * 0.5)]
    p75 = daily_costs[int(len(daily_costs) * 0.75)]
    p90 = daily_costs[int(len(daily_costs) * 0.9)]

    # ベースライン = P25（下位25%の日次コスト）
    baseline_daily = p25
    recommended_hourly = (baseline_daily * safety_margin) / 24

    return {
        "analysis_period_days": len(daily_costs),
        "daily_cost_stats": {
            "min": round(min(daily_costs), 2),
            "p10": round(p10, 2),
            "p25_baseline": round(p25, 2),
            "p50_median": round(p50, 2),
            "p75": round(p75, 2),
            "p90": round(p90, 2),
            "max": round(max(daily_costs), 2),
            "average": round(sum(daily_costs) / len(daily_costs), 2),
        },
        "recommendation": {
            "safety_margin": safety_margin,
            "recommended_hourly_commitment": round(recommended_hourly, 2),
            "estimated_monthly_commitment": round(recommended_hourly * 730, 2),
            "coverage_percentage": round(
                (baseline_daily * safety_margin) / (sum(daily_costs) / len(daily_costs)) * 100, 1
            ),
        },
    }
```

### 4.4 Reserved Instances の管理

```bash
# RI の一覧と使用状況を確認
aws ec2 describe-reserved-instances \
  --filters Name=state,Values=active \
  --output json | jq '.ReservedInstances[] | {
    id: .ReservedInstancesId,
    type: .InstanceType,
    count: .InstanceCount,
    state: .State,
    offering: .OfferingType,
    start: .Start,
    end: .End,
    fixed_price: .FixedPrice,
    usage_price: .UsagePrice
  }'

# RI カバレッジの確認
aws ce get-reservation-coverage \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --group-by Type=DIMENSION,Key=INSTANCE_TYPE \
  --output json | jq '.CoveragesByTime[0].Groups[] | select(.Coverage.CoverageHours.CoverageHoursPercentage | tonumber > 0) | {
    instance_type: .Attributes.instanceType,
    coverage_pct: .Coverage.CoverageHours.CoverageHoursPercentage,
    on_demand_hours: .Coverage.CoverageHours.OnDemandHours,
    reserved_hours: .Coverage.CoverageHours.ReservedNormalizedUnitsPercentage
  }'

# RI 使用率の確認
aws ce get-reservation-utilization \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --granularity MONTHLY \
  --output json | jq '.UtilizationsByTime[0].Total | {
    utilization_pct: .UtilizationPercentage,
    purchased_hours: .PurchasedHours,
    total_actual_hours: .TotalActualHours,
    unused_hours: .UnusedHours,
    net_savings: .NetRISavings,
    amortized_upfront: .AmortizedUpfrontFee
  }'

# RI の購入推奨を取得
aws ce get-reservation-purchase-recommendation \
  --service "Amazon Elastic Compute Cloud - Compute" \
  --term-in-years "ONE_YEAR" \
  --payment-option "PARTIAL_UPFRONT" \
  --lookback-period-in-days "SIXTY_DAYS"
```

### 4.5 Spot Instance の活用

```python
"""
Spot Instance を活用したコスト削減の実装例。
EC2 Fleet / ECS Capacity Provider / EKS Karpenter との連携。
"""
import boto3
import json
from datetime import datetime, timedelta


def get_spot_price_history(
    instance_types: list[str],
    availability_zones: list[str] = None,
    days: int = 7,
) -> dict:
    """Spot 価格履歴を取得して分析"""
    ec2 = boto3.client("ec2", region_name="ap-northeast-1")

    params = {
        "InstanceTypes": instance_types,
        "ProductDescriptions": ["Linux/UNIX"],
        "StartTime": datetime.now() - timedelta(days=days),
        "EndTime": datetime.now(),
    }
    if availability_zones:
        params["AvailabilityZone"] = availability_zones[0]

    response = ec2.describe_spot_price_history(**params)

    # インスタンスタイプ別に集計
    price_data = {}
    for item in response["SpotPriceHistory"]:
        itype = item["InstanceType"]
        price = float(item["SpotPrice"])
        az = item["AvailabilityZone"]

        key = f"{itype}/{az}"
        if key not in price_data:
            price_data[key] = {
                "instance_type": itype,
                "az": az,
                "prices": [],
            }
        price_data[key]["prices"].append(price)

    # 統計を計算
    results = {}
    for key, data in price_data.items():
        prices = data["prices"]
        results[key] = {
            "instance_type": data["instance_type"],
            "az": data["az"],
            "avg_price": round(sum(prices) / len(prices), 4),
            "min_price": round(min(prices), 4),
            "max_price": round(max(prices), 4),
            "price_stability": round(
                1 - (max(prices) - min(prices)) / (sum(prices) / len(prices)),
                2,
            ),
            "data_points": len(prices),
        }

    return results


def create_spot_fleet_request(
    target_capacity: int = 10,
    instance_types: list[str] = None,
    subnets: list[str] = None,
    iam_fleet_role: str = "",
) -> str:
    """多様性のある Spot Fleet リクエストを作成"""
    ec2 = boto3.client("ec2", region_name="ap-northeast-1")

    if instance_types is None:
        instance_types = [
            "m5.large", "m5a.large", "m5d.large",
            "m6i.large", "m6a.large",
            "c5.large", "c5a.large", "c6i.large",
        ]

    launch_specifications = []
    for itype in instance_types:
        for subnet in (subnets or []):
            launch_specifications.append({
                "InstanceType": itype,
                "SubnetId": subnet,
                "ImageId": "ami-0123456789abcdef0",  # 最新AMI
                "KeyName": "my-key",
                "SecurityGroups": [{"GroupId": "sg-0123456789abcdef0"}],
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": [
                            {"Key": "Environment", "Value": "prod"},
                            {"Key": "LaunchType", "Value": "spot-fleet"},
                        ],
                    }
                ],
            })

    response = ec2.request_spot_fleet(
        SpotFleetRequestConfig={
            "IamFleetRole": iam_fleet_role,
            "TargetCapacity": target_capacity,
            "SpotPrice": "0.10",  # 最大支払い価格
            "AllocationStrategy": "capacityOptimized",
            "TerminateInstancesWithExpiration": True,
            "Type": "maintain",
            "LaunchSpecifications": launch_specifications,
            "OnDemandTargetCapacity": int(target_capacity * 0.2),
            "OnDemandAllocationStrategy": "lowestPrice",
            "TagSpecifications": [
                {
                    "ResourceType": "spot-fleet-request",
                    "Tags": [
                        {"Key": "Name", "Value": "production-spot-fleet"},
                    ],
                }
            ],
        }
    )

    return response["SpotFleetRequestId"]
```

---

## 5. Cost & Usage Report (CUR) と Athena 分析

### 5.1 CUR の設定

```bash
# CUR レポートの作成
aws cur put-report-definition \
  --report-definition '{
    "ReportName": "daily-cost-report",
    "TimeUnit": "DAILY",
    "Format": "Parquet",
    "Compression": "Parquet",
    "AdditionalSchemaElements": ["RESOURCES"],
    "S3Bucket": "my-cur-reports-bucket",
    "S3Prefix": "cur/",
    "S3Region": "us-east-1",
    "AdditionalArtifacts": ["ATHENA"],
    "RefreshClosedReports": true,
    "ReportVersioning": "OVERWRITE_REPORT"
  }'
```

### 5.2 Terraform での CUR + Athena 環境構築

```hcl
# S3 バケット（CUR 保存用）
resource "aws_s3_bucket" "cur_reports" {
  bucket = "my-company-cur-reports"

  tags = {
    Purpose = "cost-and-usage-reports"
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cur_lifecycle" {
  bucket = aws_s3_bucket.cur_reports.id

  rule {
    id     = "archive-old-reports"
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

resource "aws_s3_bucket_policy" "cur_policy" {
  bucket = aws_s3_bucket.cur_reports.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "billingreports.amazonaws.com" }
        Action    = ["s3:GetBucketAcl", "s3:GetBucketPolicy"]
        Resource  = aws_s3_bucket.cur_reports.arn
      },
      {
        Effect    = "Allow"
        Principal = { Service = "billingreports.amazonaws.com" }
        Action    = "s3:PutObject"
        Resource  = "${aws_s3_bucket.cur_reports.arn}/*"
      },
    ]
  })
}

# CUR レポート定義
resource "aws_cur_report_definition" "daily_report" {
  report_name                = "daily-cost-report"
  time_unit                  = "DAILY"
  format                     = "Parquet"
  compression                = "Parquet"
  additional_schema_elements = ["RESOURCES"]
  s3_bucket                  = aws_s3_bucket.cur_reports.bucket
  s3_prefix                  = "cur/"
  s3_region                  = "us-east-1"
  additional_artifacts       = ["ATHENA"]
  report_versioning          = "OVERWRITE_REPORT"
  refresh_closed_reports     = true
}

# Athena データベース
resource "aws_athena_database" "cur_db" {
  name   = "cur_database"
  bucket = aws_s3_bucket.cur_reports.bucket

  encryption_configuration {
    encryption_option = "SSE_S3"
  }
}

# Athena ワークグループ
resource "aws_athena_workgroup" "cost_analysis" {
  name = "cost-analysis"

  configuration {
    enforce_workgroup_configuration = true
    result_configuration {
      output_location = "s3://${aws_s3_bucket.cur_reports.bucket}/athena-results/"
    }
  }

  tags = {
    Purpose = "cost-analysis"
  }
}
```

### 5.3 Athena でのコスト分析クエリ

```sql
-- サービス別月次コスト（上位20サービス）
SELECT
  line_item_product_code AS service,
  DATE_FORMAT(line_item_usage_start_date, '%Y-%m') AS month,
  ROUND(SUM(line_item_unblended_cost), 2) AS cost_usd,
  ROUND(SUM(line_item_unblended_cost) / (
    SELECT SUM(line_item_unblended_cost)
    FROM cur_database.cur_report
    WHERE DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
  ) * 100, 1) AS percentage
FROM cur_database.cur_report
WHERE DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
  AND line_item_line_item_type = 'Usage'
GROUP BY line_item_product_code, DATE_FORMAT(line_item_usage_start_date, '%Y-%m')
ORDER BY cost_usd DESC
LIMIT 20;

-- タグ別コスト分析（未タグ付けリソースの特定）
SELECT
  COALESCE(resource_tags_user_project, '(untagged)') AS project,
  COALESCE(resource_tags_user_environment, '(untagged)') AS environment,
  line_item_product_code AS service,
  ROUND(SUM(line_item_unblended_cost), 2) AS cost_usd,
  COUNT(DISTINCT line_item_resource_id) AS resource_count
FROM cur_database.cur_report
WHERE DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
  AND line_item_line_item_type = 'Usage'
GROUP BY
  resource_tags_user_project,
  resource_tags_user_environment,
  line_item_product_code
HAVING SUM(line_item_unblended_cost) > 10
ORDER BY cost_usd DESC;

-- EC2 インスタンス別コスト（ライトサイジング用）
SELECT
  line_item_resource_id AS instance_id,
  product_instance_type AS instance_type,
  COALESCE(resource_tags_user_name, '(unnamed)') AS name,
  ROUND(SUM(line_item_unblended_cost), 2) AS monthly_cost,
  ROUND(SUM(line_item_usage_amount), 1) AS usage_hours,
  ROUND(SUM(line_item_usage_amount) / 730 * 100, 1) AS utilization_pct
FROM cur_database.cur_report
WHERE line_item_product_code = 'AmazonEC2'
  AND line_item_usage_type LIKE '%BoxUsage%'
  AND DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
GROUP BY line_item_resource_id, product_instance_type, resource_tags_user_name
ORDER BY monthly_cost DESC
LIMIT 50;

-- データ転送コスト分析
SELECT
  line_item_product_code AS service,
  line_item_usage_type AS usage_type,
  ROUND(SUM(line_item_unblended_cost), 2) AS cost_usd,
  ROUND(SUM(line_item_usage_amount), 2) AS usage_gb
FROM cur_database.cur_report
WHERE line_item_usage_type LIKE '%DataTransfer%'
  AND DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
GROUP BY line_item_product_code, line_item_usage_type
HAVING SUM(line_item_unblended_cost) > 1
ORDER BY cost_usd DESC;

-- Savings Plans カバレッジの詳細分析
SELECT
  DATE_FORMAT(line_item_usage_start_date, '%Y-%m-%d') AS usage_date,
  line_item_product_code AS service,
  ROUND(SUM(CASE WHEN savings_plan_savings_plan_arn != '' THEN line_item_unblended_cost ELSE 0 END), 2) AS sp_covered_cost,
  ROUND(SUM(CASE WHEN savings_plan_savings_plan_arn = '' THEN line_item_unblended_cost ELSE 0 END), 2) AS on_demand_cost,
  ROUND(SUM(line_item_unblended_cost), 2) AS total_cost
FROM cur_database.cur_report
WHERE line_item_line_item_type IN ('Usage', 'SavingsPlanCoveredUsage')
  AND DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
GROUP BY DATE_FORMAT(line_item_usage_start_date, '%Y-%m-%d'), line_item_product_code
ORDER BY usage_date, total_cost DESC;

-- 未使用 EBS ボリュームの検出
SELECT
  line_item_resource_id AS volume_id,
  product_volume_type AS volume_type,
  ROUND(SUM(line_item_usage_amount), 0) AS gb_months,
  ROUND(SUM(line_item_unblended_cost), 2) AS monthly_cost
FROM cur_database.cur_report
WHERE line_item_product_code = 'AmazonEC2'
  AND line_item_usage_type LIKE '%EBS:Volume%'
  AND DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
  AND line_item_resource_id NOT IN (
    SELECT DISTINCT line_item_resource_id
    FROM cur_database.cur_report
    WHERE line_item_usage_type LIKE '%EBS:VolumeIOUsage%'
      AND DATE_FORMAT(line_item_usage_start_date, '%Y-%m') = DATE_FORMAT(CURRENT_DATE - INTERVAL '1' MONTH, '%Y-%m')
  )
GROUP BY line_item_resource_id, product_volume_type
ORDER BY monthly_cost DESC;
```

---

## 6. Compute Optimizer によるライトサイジング

### 6.1 CLI での推奨取得

```bash
# EC2 インスタンスの最適化推奨を取得
aws compute-optimizer get-ec2-instance-recommendations \
  --output json | jq '.instanceRecommendations[] | {
    instance_id: .instanceArn | split("/") | last,
    current_type: .currentInstanceType,
    finding: .finding,
    recommendations: [.recommendationOptions[:3][] | {
      type: .instanceType,
      projected_utilization: .projectedUtilizationMetrics,
      savings_opportunity: .savingsOpportunity,
      performance_risk: .performanceRisk
    }]
  }'

# EBS ボリュームの最適化推奨
aws compute-optimizer get-ebs-volume-recommendations \
  --output json | jq '.volumeRecommendations[] | {
    volume_arn: .volumeArn,
    current_config: .currentConfiguration,
    finding: .finding,
    recommendations: [.volumeRecommendationOptions[:3][] | {
      config: .configuration,
      performance_risk: .performanceRisk,
      savings_opportunity: .savingsOpportunity
    }]
  }'

# Lambda 関数の最適化推奨
aws compute-optimizer get-lambda-function-recommendations \
  --output json | jq '.lambdaFunctionRecommendations[] | {
    function_arn: .functionArn,
    current_memory: .currentMemorySize,
    finding: .finding,
    recommendations: [.memorySizeRecommendationOptions[:3][] | {
      memory_size: .memorySize,
      projected_utilization: .projectedUtilizationMetrics,
      savings_opportunity: .savingsOpportunity
    }]
  }'

# Auto Scaling グループの推奨
aws compute-optimizer get-auto-scaling-group-recommendations \
  --output json | jq '.autoScalingGroupRecommendations[] | {
    asg_name: .autoScalingGroupName,
    current_config: .currentConfiguration,
    finding: .finding,
    recommendations: [.recommendationOptions[:3][] | {
      config: .configuration,
      projected_utilization: .projectedUtilizationMetrics
    }]
  }'
```

### 6.2 Python による自動ライトサイジングレポート

```python
"""
Compute Optimizer の推奨に基づくライトサイジングレポートを生成。
月次で実行し、コスト削減機会を一覧化する。
"""
import boto3
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class RightsizingRecommendation:
    """ライトサイジング推奨"""
    resource_id: str
    resource_type: str
    current_config: str
    recommended_config: str
    finding: str
    estimated_monthly_savings: float
    performance_risk: float
    tags: dict = field(default_factory=dict)


def get_ec2_rightsizing_recommendations() -> list[RightsizingRecommendation]:
    """EC2 のライトサイジング推奨を取得"""
    co_client = boto3.client("compute-optimizer", region_name="ap-northeast-1")
    ec2_client = boto3.client("ec2", region_name="ap-northeast-1")

    recommendations = []

    response = co_client.get_ec2_instance_recommendations()

    for rec in response.get("instanceRecommendations", []):
        instance_arn = rec["instanceArn"]
        instance_id = instance_arn.split("/")[-1]
        finding = rec["finding"]

        # OVER_PROVISIONED または UNDER_PROVISIONED のみ対象
        if finding not in ("OVER_PROVISIONED", "UNDER_PROVISIONED"):
            continue

        # タグを取得
        tags = {}
        try:
            tags_response = ec2_client.describe_tags(
                Filters=[
                    {"Name": "resource-id", "Values": [instance_id]}
                ]
            )
            tags = {
                t["Key"]: t["Value"]
                for t in tags_response.get("Tags", [])
            }
        except Exception:
            pass

        for option in rec.get("recommendationOptions", [])[:1]:
            savings = option.get("savingsOpportunity", {})
            monthly_savings = float(
                savings.get("estimatedMonthlySavings", {}).get("value", 0)
            )
            perf_risk = float(option.get("performanceRisk", 0))

            recommendations.append(
                RightsizingRecommendation(
                    resource_id=instance_id,
                    resource_type="EC2",
                    current_config=rec["currentInstanceType"],
                    recommended_config=option["instanceType"],
                    finding=finding,
                    estimated_monthly_savings=monthly_savings,
                    performance_risk=perf_risk,
                    tags=tags,
                )
            )

    return recommendations


def generate_rightsizing_report() -> str:
    """ライトサイジングレポートを Markdown で生成"""
    recs = get_ec2_rightsizing_recommendations()

    lines = [
        f"# EC2 ライトサイジングレポート",
        f"生成日時: {datetime.now().isoformat()}",
        f"推奨件数: {len(recs)}",
        "",
    ]

    # サマリー
    total_savings = sum(r.estimated_monthly_savings for r in recs)
    over_provisioned = [r for r in recs if r.finding == "OVER_PROVISIONED"]
    under_provisioned = [r for r in recs if r.finding == "UNDER_PROVISIONED"]

    lines.append("## サマリー")
    lines.append(f"- 推定月次コスト削減: **${total_savings:,.2f}**")
    lines.append(f"- Over-Provisioned: {len(over_provisioned)} 件")
    lines.append(f"- Under-Provisioned: {len(under_provisioned)} 件")
    lines.append("")

    # 詳細テーブル
    lines.append("## 詳細推奨一覧")
    lines.append("| Instance ID | Name | 現在 | 推奨 | Finding | 月次削減 | Risk |")
    lines.append("|---|---|---|---|---|---|---|")

    recs.sort(key=lambda r: r.estimated_monthly_savings, reverse=True)

    for r in recs:
        name = r.tags.get("Name", "-")
        lines.append(
            f"| {r.resource_id} | {name} | {r.current_config} | "
            f"{r.recommended_config} | {r.finding} | "
            f"${r.estimated_monthly_savings:,.2f} | {r.performance_risk} |"
        )

    return "\n".join(lines)
```

---

## 7. 比較表

### 7.1 割引プラン比較

| プラン | 最大割引 | 柔軟性 | コミット期間 | 支払い方法 |
|--------|---------|--------|-------------|-----------|
| **Compute Savings Plans** | 66% | 高（任意の EC2/Fargate/Lambda） | 1年 or 3年 | 全前払い/一部/なし |
| **EC2 Instance Savings Plans** | 72% | 中（ファミリー・リージョン固定） | 1年 or 3年 | 全前払い/一部/なし |
| **Standard RI** | 72% | 低（インスタンスタイプ・AZ 固定） | 1年 or 3年 | 全前払い/一部/なし |
| **Convertible RI** | 66% | 中（変更可能） | 1年 or 3年 | 全前払い/一部/なし |
| **Spot Instances** | 90% | なし（中断される可能性） | なし | オンデマンド |

### 7.2 コスト管理ツール比較

| ツール | 目的 | 料金 | 主要機能 |
|-------|------|------|---------|
| **Cost Explorer** | コスト分析 | 無料 | グラフ可視化、フィルタ、予測 |
| **AWS Budgets** | 予算管理 | 最初の2件無料、以降 $0.02/日 | アラート、自動アクション |
| **Cost Anomaly Detection** | 異常検知 | 無料 | ML ベースの異常検知 |
| **CUR (Cost & Usage Report)** | 詳細レポート | 無料（S3 料金のみ） | 行レベルの詳細データ |
| **Compute Optimizer** | サイジング推奨 | 無料 | ML ベースの最適化推奨 |
| **Trusted Advisor** | ベストプラクティス | Business/Enterprise サポート | 未使用リソース検出 |

### 7.3 支払いオプション比較

| 支払いオプション | 割引率 | 初期費用 | 月次費用 | 適切な場面 |
|-----------------|--------|---------|---------|-----------|
| **全前払い (All Upfront)** | 最大 | 一括支払い | なし | キャッシュに余裕、長期安定ワークロード |
| **一部前払い (Partial Upfront)** | 中程度 | 一部支払い | 残りを月払い | バランス型、最も一般的 |
| **前払いなし (No Upfront)** | 最小 | なし | 全額月払い | キャッシュフロー重視、初回の SP 購入 |

### 7.4 Savings Plans vs Reserved Instances 意思決定マトリクス

| 判断基準 | Savings Plans 推奨 | Reserved Instances 推奨 |
|---------|-------------------|----------------------|
| **利用サービス** | EC2 + Fargate + Lambda 混在 | EC2 のみ |
| **インスタンスファミリー変更** | 可能性あり | 変更しない |
| **リージョン変更** | 可能性あり | 固定 |
| **OS 変更** | 可能性あり | 固定 |
| **マーケットプレイス売却** | 不要 | 余剰分を売却したい |
| **キャパシティ予約** | 不要 | AZ 内での予約が必要 |

---

## 8. サービス別コスト最適化ベストプラクティス

### 8.1 EC2 コスト最適化

```bash
# 未使用の Elastic IP を検出
aws ec2 describe-addresses \
  --query 'Addresses[?AssociationId==`null`].[PublicIp,AllocationId]' \
  --output table

# 停止中のインスタンスに紐づく EBS ボリュームを検出
aws ec2 describe-volumes \
  --filters Name=status,Values=available \
  --query 'Volumes[*].{ID:VolumeId,Size:Size,Type:VolumeType,Created:CreateTime}' \
  --output table

# 古い AMI を検出（90日以上前）
THRESHOLD=$(date -d "90 days ago" +%Y-%m-%d 2>/dev/null || date -v-90d +%Y-%m-%d)
aws ec2 describe-images \
  --owners self \
  --query "Images[?CreationDate<'${THRESHOLD}'].[ImageId,Name,CreationDate]" \
  --output table

# 古いスナップショットを検出
aws ec2 describe-snapshots \
  --owner-ids self \
  --query "Snapshots[?StartTime<'${THRESHOLD}'].[SnapshotId,VolumeSize,StartTime,Description]" \
  --output table

# 未使用の NAT Gateway を検出（CloudWatch メトリクスで確認）
for gw in $(aws ec2 describe-nat-gateways --query 'NatGateways[*].NatGatewayId' --output text); do
  bytes=$(aws cloudwatch get-metric-statistics \
    --namespace AWS/NATGateway \
    --metric-name BytesOutToDestination \
    --dimensions Name=NatGatewayId,Value=$gw \
    --start-time "$(date -d '7 days ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-7d +%Y-%m-%dT%H:%M:%S)" \
    --end-time "$(date +%Y-%m-%dT%H:%M:%S)" \
    --period 604800 \
    --statistics Sum \
    --query 'Datapoints[0].Sum' \
    --output text 2>/dev/null)
  echo "NAT GW: $gw - Bytes out (7d): ${bytes:-0}"
done
```

### 8.2 S3 コスト最適化

```hcl
# S3 Intelligent-Tiering とライフサイクルポリシー
resource "aws_s3_bucket_intelligent_tiering_configuration" "cost_optimized" {
  bucket = aws_s3_bucket.data.id
  name   = "cost-optimized-tiering"

  tiering {
    access_tier = "ARCHIVE_ACCESS"
    days        = 90
  }

  tiering {
    access_tier = "DEEP_ARCHIVE_ACCESS"
    days        = 180
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "cost_optimized" {
  bucket = aws_s3_bucket.data.id

  # ログファイルは30日後に IA、90日後に Glacier、365日後に削除
  rule {
    id     = "log-lifecycle"
    status = "Enabled"

    filter {
      prefix = "logs/"
    }

    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    expiration {
      days = 365
    }
  }

  # マルチパートアップロードの未完了分を7日後にクリーンアップ
  rule {
    id     = "abort-multipart"
    status = "Enabled"

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  # 古いバージョンを30日後に削除
  rule {
    id     = "noncurrent-versions"
    status = "Enabled"

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}
```

### 8.3 RDS コスト最適化

```bash
# 未使用の RDS インスタンスを検出（接続数が0のもの）
for db in $(aws rds describe-db-instances --query 'DBInstances[*].DBInstanceIdentifier' --output text); do
  connections=$(aws cloudwatch get-metric-statistics \
    --namespace AWS/RDS \
    --metric-name DatabaseConnections \
    --dimensions Name=DBInstanceIdentifier,Value=$db \
    --start-time "$(date -d '7 days ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-7d +%Y-%m-%dT%H:%M:%S)" \
    --end-time "$(date +%Y-%m-%dT%H:%M:%S)" \
    --period 604800 \
    --statistics Maximum \
    --query 'Datapoints[0].Maximum' \
    --output text 2>/dev/null)
  echo "RDS: $db - Max connections (7d): ${connections:-0}"
done

# 過剰プロビジョニングされた RDS の検出（CPU 使用率10%以下）
for db in $(aws rds describe-db-instances --query 'DBInstances[*].DBInstanceIdentifier' --output text); do
  cpu=$(aws cloudwatch get-metric-statistics \
    --namespace AWS/RDS \
    --metric-name CPUUtilization \
    --dimensions Name=DBInstanceIdentifier,Value=$db \
    --start-time "$(date -d '7 days ago' +%Y-%m-%dT%H:%M:%S 2>/dev/null || date -v-7d +%Y-%m-%dT%H:%M:%S)" \
    --end-time "$(date +%Y-%m-%dT%H:%M:%S)" \
    --period 604800 \
    --statistics Average \
    --query 'Datapoints[0].Average' \
    --output text 2>/dev/null)
  if [ -n "$cpu" ] && [ "$(echo "$cpu < 10" | bc -l 2>/dev/null)" = "1" ]; then
    echo "LOW UTILIZATION - RDS: $db - Avg CPU (7d): ${cpu}%"
  fi
done
```

### 8.4 Lambda コスト最適化

```python
"""
Lambda 関数のコスト最適化分析。
メモリ設定の最適化とプロビジョンドコンカレンシーの適切な設定を推奨する。
"""
import boto3
from datetime import datetime, timedelta


def analyze_lambda_cost_optimization(
    region: str = "ap-northeast-1",
) -> list[dict]:
    """Lambda 関数のコスト最適化推奨を生成"""
    lambda_client = boto3.client("lambda", region_name=region)
    cw_client = boto3.client("cloudwatch", region_name=region)

    functions = lambda_client.list_functions()["Functions"]
    recommendations = []

    for func in functions:
        func_name = func["FunctionName"]
        memory = func["MemorySize"]

        # 過去7日のメトリクスを取得
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)

        # 実行時間の統計
        duration_stats = cw_client.get_metric_statistics(
            Namespace="AWS/Lambda",
            MetricName="Duration",
            Dimensions=[{"Name": "FunctionName", "Value": func_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=86400,  # 1日
            Statistics=["Average", "Maximum", "p90"],
        )

        # 呼び出し回数
        invocations = cw_client.get_metric_statistics(
            Namespace="AWS/Lambda",
            MetricName="Invocations",
            Dimensions=[{"Name": "FunctionName", "Value": func_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=604800,  # 1週間
            Statistics=["Sum"],
        )

        avg_duration = 0
        max_duration = 0
        total_invocations = 0

        if duration_stats["Datapoints"]:
            avg_duration = sum(
                d["Average"] for d in duration_stats["Datapoints"]
            ) / len(duration_stats["Datapoints"])
            max_duration = max(
                d["Maximum"] for d in duration_stats["Datapoints"]
            )

        if invocations["Datapoints"]:
            total_invocations = int(invocations["Datapoints"][0]["Sum"])

        # コスト計算（GB-秒あたり $0.0000166667）
        gb_seconds = (memory / 1024) * (avg_duration / 1000) * total_invocations
        estimated_weekly_cost = gb_seconds * 0.0000166667

        # 推奨メモリサイズの推定
        recommended_memory = memory
        if avg_duration > 0 and max_duration < memory * 0.5:
            recommended_memory = max(128, memory // 2)

        recommendation = {
            "function_name": func_name,
            "current_memory_mb": memory,
            "avg_duration_ms": round(avg_duration, 1),
            "max_duration_ms": round(max_duration, 1),
            "weekly_invocations": total_invocations,
            "estimated_weekly_cost": round(estimated_weekly_cost, 4),
            "recommended_memory_mb": recommended_memory,
        }

        if recommended_memory != memory:
            savings_pct = (1 - recommended_memory / memory) * 100
            recommendation["potential_savings_pct"] = round(savings_pct, 1)
            recommendation["action"] = "REDUCE_MEMORY"
        else:
            recommendation["action"] = "NO_CHANGE"

        recommendations.append(recommendation)

    return recommendations
```

---

## 9. アンチパターン

### 9.1 開発環境を 24/7 稼働させる

```
NG:
  開発用 EC2 (m5.xlarge x 3) + RDS (db.r5.large)
  → 24時間365日稼働 = 約 $500/月

OK:
  平日 9:00-21:00 のみ稼働（EventBridge Scheduler）
  → 月160時間 / 720時間 = 約 $110/月 (78% 削減)

  # EventBridge Scheduler で自動停止/起動
  aws scheduler create-schedule \
    --name "stop-dev-instances" \
    --schedule-expression "cron(0 21 ? * MON-FRI *)" \
    --target '{"Arn":"arn:aws:ssm:...:automation-definition/AWS-StopEC2Instance"}'
```

### 9.2 Savings Plans をワークロード分析なしに購入

```
NG:
  「72% 割引は魅力的」→ 3年全前払いで大量購入
  → 6ヶ月後にアーキテクチャ変更 → Savings Plans が余る

OK:
  1. Cost Explorer で過去 60-90 日の利用傾向を分析
  2. ベースライン使用量（最低使用量）を特定
  3. ベースライン分だけ Savings Plans を購入（70-80% カバー）
  4. ピーク分はオンデマンド or Spot で対応
  5. 最初は 1年・前払いなしで開始、確信が持てたら 3年に移行
```

### 9.3 タグ付けを後回しにする

```
NG:
  プロジェクト初期にタグ付けルールを定めない
  → 100+ リソースが作成された後にタグ付けを開始
  → コスト配分レポートの「未分類」が40%以上

OK:
  Day 1 からタグ付けを強制する
  1. SCP でタグなしリソースの作成を禁止
  2. AWS Config required-tags ルールで検出
  3. CI/CD パイプラインにタグ付けチェックを組み込み
  4. 定期的に Tag Editor で未タグ付けリソースを確認・是正
```

### 9.4 データ転送コストを見落とす

```
NG:
  マルチリージョン構成でリージョン間データ転送を放置
  → 毎月 $500+ のデータ転送料金

OK:
  1. VPC Endpoint を活用して S3/DynamoDB へのアクセスを最適化
  2. CloudFront を使ってオリジンからのデータ転送を削減
  3. リージョン間のデータ転送を最小限に設計
  4. CUR のデータ転送レポートを定期的に確認
```

### 9.5 一つのサイズで全てに対応しようとする

```
NG:
  全てのワークロードに m5.xlarge を使用
  → バッチ処理には過剰、API サーバーには不足

OK:
  ワークロード特性に応じたインスタンス選択
  ┌─────────────────────────────────────────────────┐
  │  Web API    → c6i.large (CPU最適化)             │
  │  バッチ処理 → m6i.large (バランス型) + Spot     │
  │  ML推論     → g5.xlarge (GPU) or inf2.xlarge    │
  │  キャッシュ  → r6i.large (メモリ最適化)          │
  │  開発環境   → t3.medium (バースト型)             │
  └─────────────────────────────────────────────────┘
```

---

## 10. 組織的なコスト管理（Organizations / マルチアカウント）

### 10.1 Organizations でのコスト管理

```hcl
# 組織内のアカウント別予算を一括定義
variable "account_budgets" {
  type = map(object({
    account_id   = string
    budget_limit = string
    email        = string
  }))
  default = {
    production = {
      account_id   = "111111111111"
      budget_limit = "5000"
      email        = "prod-team@example.com"
    }
    staging = {
      account_id   = "222222222222"
      budget_limit = "1000"
      email        = "staging-team@example.com"
    }
    development = {
      account_id   = "333333333333"
      budget_limit = "500"
      email        = "dev-team@example.com"
    }
    sandbox = {
      account_id   = "444444444444"
      budget_limit = "200"
      email        = "sandbox-admin@example.com"
    }
  }
}

resource "aws_budgets_budget" "account_budgets" {
  for_each = var.account_budgets

  name         = "${each.key}-account-budget"
  budget_type  = "COST"
  limit_amount = each.value.budget_limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "LinkedAccount"
    values = [each.value.account_id]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [each.value.email, "finops@example.com"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [each.value.email, "finops@example.com"]
    subscriber_sns_topic_arns  = [aws_sns_topic.budget_alerts.arn]
  }
}
```

### 10.2 SCP によるコスト制御

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DenyExpensiveInstances",
      "Effect": "Deny",
      "Action": "ec2:RunInstances",
      "Resource": "arn:aws:ec2:*:*:instance/*",
      "Condition": {
        "ForAnyValue:StringNotLike": {
          "ec2:InstanceType": [
            "t3.*",
            "t3a.*",
            "m5.large",
            "m5.xlarge",
            "m6i.large",
            "m6i.xlarge",
            "c5.large",
            "c5.xlarge",
            "c6i.large",
            "c6i.xlarge"
          ]
        }
      }
    },
    {
      "Sid": "DenyExpensiveRegions",
      "Effect": "Deny",
      "Action": "*",
      "Resource": "*",
      "Condition": {
        "StringNotEquals": {
          "aws:RequestedRegion": [
            "ap-northeast-1",
            "us-east-1",
            "us-west-2"
          ]
        }
      }
    },
    {
      "Sid": "RequireEnvironmentTag",
      "Effect": "Deny",
      "Action": [
        "ec2:RunInstances",
        "rds:CreateDBInstance",
        "lambda:CreateFunction"
      ],
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestTag/Environment": "true"
        }
      }
    }
  ]
}
```

---

## 11. FAQ

### Q1. Cost Explorer と CUR（Cost and Usage Report）の違いは？

**A.** Cost Explorer はコンソール上の可視化ツールで、月次・日次のサマリーに適している。CUR は行レベルの詳細データを S3 に出力し、Athena や QuickSight で独自分析ができる。数十アカウント規模の Organizations では CUR + Athena の組み合わせが必須。

### Q2. タグ付けが不十分で「未分類」コストが多い。どう改善する？

**A.** AWS Config ルール `required-tags` でタグ未設定リソースを検出し、Service Control Policy (SCP) でタグなしリソース作成を禁止する。既存リソースには Tag Editor で一括タグ付けが可能。

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "RequireTags",
      "Effect": "Deny",
      "Action": ["ec2:RunInstances"],
      "Resource": "*",
      "Condition": {
        "Null": {
          "aws:RequestTag/Environment": "true"
        }
      }
    }
  ]
}
```

### Q3. Spot Instance はどのようなワークロードに適している？

**A.** バッチ処理、CI/CD、データ分析、機械学習トレーニングなど、中断耐性のあるワークロードに最適。ECS Capacity Provider や EKS Karpenter で Spot を自動管理すれば、中断時の再スケジューリングも自動化できる。Web サーバーではオンデマンドとの混合（70% On-Demand + 30% Spot）が安全。

### Q4. Savings Plans の購入後に利用量が減った場合はどうなる？

**A.** Savings Plans は利用量に関わらず、コミットした時間あたりの金額を支払う必要がある。余剰分は無駄になるため、以下の対策を推奨する。

1. **段階的購入**: まず 70-80% のベースラインをカバーし、残りはオンデマンドで対応
2. **短期から開始**: 最初は 1年・前払いなしで購入し、利用パターンが安定してから 3年に移行
3. **定期的な見直し**: Savings Plans カバレッジと使用率を月次で確認
4. **Organizations 活用**: 組織内の全アカウントで Savings Plans を共有可能

### Q5. コスト最適化の優先順位は？

**A.** 以下の順序で取り組むと効果が大きい。

```
1. 未使用リソースの削除（即時効果、リスクなし）
   └─ 停止中の EC2、未使用の EBS/EIP/NAT GW

2. ライトサイジング（短期、低リスク）
   └─ Compute Optimizer の推奨に従い段階的に変更

3. スケジューリング（短期、低リスク）
   └─ 開発環境の自動停止/起動

4. 料金モデルの最適化（中期）
   └─ Savings Plans / Reserved Instances の購入

5. アーキテクチャの最適化（長期、高効果）
   └─ サーバーレス化、マネージドサービス活用
```

### Q6. マルチアカウント環境でのコスト管理のベストプラクティスは？

**A.** AWS Organizations を活用し、以下の構成を推奨する。

1. **管理アカウント**: 一括請求（Consolidated Billing）の有効化
2. **コスト管理アカウント**: CUR、Athena、QuickSight を集約
3. **アカウント別予算**: 各アカウントに月次予算とアラートを設定
4. **SCP**: 高額インスタンスタイプやリージョンの利用制限
5. **Savings Plans**: 管理アカウントで一括購入（全アカウントで共有）

### Q7. Graviton（ARM）インスタンスへの移行でどの程度コスト削減できるか？

**A.** Graviton インスタンスは同等の x86 インスタンスと比較して約 20% のコスト削減が見込める。加えて、パフォーマンスも最大 40% 向上するケースがある。移行の際は以下を確認する。

1. アプリケーションが ARM アーキテクチャに対応しているか
2. 依存ライブラリが ARM 向けにコンパイル可能か
3. コンテナの場合、マルチアーキテクチャイメージを構築可能か
4. CI/CD パイプラインで ARM ビルドをサポートしているか

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| **Cost Explorer** | コスト可視化の第一歩、タグ別・サービス別分析 |
| **Budgets** | 予算アラートと自動アクション（EC2 停止等） |
| **Savings Plans** | ベースライン使用量に対して 1年 Compute SP から開始 |
| **CUR + Athena** | 大規模環境での詳細コスト分析基盤 |
| **Compute Optimizer** | ML ベースのライトサイジング推奨 |
| **タグ戦略** | Environment / Project / Team の 3 タグを全リソースに必須化 |
| **FinOps 文化** | 全員がコスト責任者、月次レビューの習慣化 |
| **継続改善** | 月次コストレビュー、四半期ごとの戦略見直し |

---

## 次に読むべきガイド

- [01-well-architected.md](./01-well-architected.md) — Well-Architected Framework の 6 つの柱
- Compute Optimizer 活用ガイド — ライトサイジングの自動化
- Organizations コスト管理 — マルチアカウントでのコスト配分

---

## 参考文献

1. **AWS公式ドキュメント** — "AWS Cost Management User Guide" — https://docs.aws.amazon.com/cost-management/latest/userguide/
2. **AWS公式ドキュメント** — "Savings Plans User Guide" — https://docs.aws.amazon.com/savingsplans/latest/userguide/
3. **AWS Well-Architected Framework** — Cost Optimization Pillar — https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/
4. **AWS公式ブログ** — "AWS Cost Optimization Best Practices" — https://aws.amazon.com/blogs/aws-cloud-financial-management/
5. **AWS公式ドキュメント** — "AWS Compute Optimizer User Guide" — https://docs.aws.amazon.com/compute-optimizer/latest/ug/
6. **AWS公式ドキュメント** — "AWS Cost and Usage Reports User Guide" — https://docs.aws.amazon.com/cur/latest/userguide/
7. **FinOps Foundation** — "FinOps Framework" — https://www.finops.org/framework/
