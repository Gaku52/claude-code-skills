# コスト最適化 — Cost Explorer / Budgets / Savings Plans

> AWS のコストを可視化・予測・削減するための実践ガイド。無駄な支出を特定し、Savings Plans やリザーブドインスタンスで最大 72% のコスト削減を実現する。

---

## この章で学ぶこと

1. **Cost Explorer** によるコスト分析とトレンド把握
2. **AWS Budgets** によるアラートと自動アクション
3. **Savings Plans / Reserved Instances** による長期コミット型割引

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
```

### 2.2 Python でのコスト分析自動化

```python
import boto3
from datetime import datetime, timedelta

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
```

---

## 5. 比較表

### 5.1 割引プラン比較

| プラン | 最大割引 | 柔軟性 | コミット期間 | 支払い方法 |
|--------|---------|--------|-------------|-----------|
| **Compute Savings Plans** | 66% | 高（任意の EC2/Fargate/Lambda） | 1年 or 3年 | 全前払い/一部/なし |
| **EC2 Instance Savings Plans** | 72% | 中（ファミリー・リージョン固定） | 1年 or 3年 | 全前払い/一部/なし |
| **Standard RI** | 72% | 低（インスタンスタイプ・AZ 固定） | 1年 or 3年 | 全前払い/一部/なし |
| **Convertible RI** | 66% | 中（変更可能） | 1年 or 3年 | 全前払い/一部/なし |
| **Spot Instances** | 90% | なし（中断される可能性） | なし | オンデマンド |

### 5.2 コスト管理ツール比較

| ツール | 目的 | 料金 | 主要機能 |
|-------|------|------|---------|
| **Cost Explorer** | コスト分析 | 無料 | グラフ可視化、フィルタ、予測 |
| **AWS Budgets** | 予算管理 | 最初の2件無料、以降 $0.02/日 | アラート、自動アクション |
| **Cost Anomaly Detection** | 異常検知 | 無料 | ML ベースの異常検知 |
| **CUR (Cost & Usage Report)** | 詳細レポート | 無料（S3 料金のみ） | 行レベルの詳細データ |
| **Compute Optimizer** | サイジング推奨 | 無料 | ML ベースの最適化推奨 |
| **Trusted Advisor** | ベストプラクティス | Business/Enterprise サポート | 未使用リソース検出 |

---

## 6. アンチパターン

### 6.1 開発環境を 24/7 稼働させる

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

### 6.2 Savings Plans をワークロード分析なしに購入

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

---

## 7. FAQ

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

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| **Cost Explorer** | コスト可視化の第一歩、タグ別・サービス別分析 |
| **Budgets** | 予算アラートと自動アクション（EC2 停止等） |
| **Savings Plans** | ベースライン使用量に対して 1年 Compute SP から開始 |
| **タグ戦略** | Environment / Project / Team の 3 タグを全リソースに必須化 |
| **継続改善** | 月次コストレビュー、Compute Optimizer の推奨確認 |

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
