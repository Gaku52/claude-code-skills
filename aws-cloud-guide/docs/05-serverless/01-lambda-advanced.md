# AWS Lambda 応用

> コールドスタートの最適化、Provisioned Concurrency、Lambda Destinations、Step Functions 連携を理解し、本番運用品質のサーバーレスアプリケーションを構築する。

---

## この章で学ぶこと

1. **コールドスタートの原因と最適化手法** -- コールドスタートが発生するメカニズムを理解し、ランタイム選択やパッケージ軽量化で実戦的に対処する
2. **Provisioned Concurrency と同時実行制御** -- レイテンシ要件が厳しいワークロードに対して予め実行環境を確保する方法を習得する
3. **Lambda Destinations と Step Functions** -- 非同期処理の結果ルーティングとオーケストレーションでエラーハンドリングを設計する

---

## 1. コールドスタートの詳解

### 1.1 コールドスタートのライフサイクル

```
リクエスト到着
    |
    v
+-----------------------------+
| 実行環境はあるか？           |
+-----------------------------+
    |             |
  ある(Warm)   ない(Cold)
    |             |
    |             v
    |    +------------------------+
    |    | 1. MicroVM 確保        |  <-- AWS管理 (数百ms)
    |    | 2. ランタイム初期化     |  <-- ランタイム依存
    |    | 3. デプロイパッケージ   |  <-- サイズ依存
    |    |    ダウンロード・展開   |
    |    | 4. Init コード実行     |  <-- ユーザーコード
    |    |    (ハンドラ外)        |
    |    +------------------------+
    |             |
    v             v
+-----------------------------+
| 5. ハンドラ関数実行          |  <-- 通常の実行
+-----------------------------+
    |
    v
+-----------------------------+
| 6. 実行環境を Warm 状態で    |
|    一定時間保持 (~5-15分)    |
+-----------------------------+
```

### 1.2 ランタイム別コールドスタート時間の目安

| ランタイム | コールドスタート (128MB) | コールドスタート (1024MB) | 備考 |
|-----------|------------------------|--------------------------|------|
| Python 3.12 | 200-400 ms | 150-300 ms | 軽量、高速起動 |
| Node.js 20.x | 200-400 ms | 150-250 ms | 軽量、高速起動 |
| Go (AL2023) | 50-150 ms | 30-100 ms | コンパイル済みバイナリ |
| Java 21 | 2,000-5,000 ms | 800-2,000 ms | JVM 起動が重い |
| .NET 8 | 800-2,000 ms | 400-1,000 ms | AOT で大幅改善可能 |
| Ruby 3.3 | 300-600 ms | 200-400 ms | インタプリタ起動 |

### 1.3 コールドスタート最適化テクニック

```python
# [最適化] ハンドラ外で初期化を行い、Warm 起動時に再利用
import boto3
import os

# --- Init Phase (コールドスタート時のみ実行) ---
TABLE_NAME = os.environ["TABLE_NAME"]
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(TABLE_NAME)
# ------------------------------------------------

def lambda_handler(event, context):
    """ハンドラは Warm 起動時にも毎回実行される"""
    user_id = event["pathParameters"]["userId"]

    response = table.get_item(Key={"userId": user_id})
    return {
        "statusCode": 200,
        "body": json.dumps(response.get("Item", {}))
    }
```

```python
# [最適化] 遅延インポートで不要なモジュールの初期化を避ける
import json
import os

def lambda_handler(event, context):
    action = event.get("action")

    if action == "generate_pdf":
        # PDF 生成が必要な場合のみ重いライブラリをインポート
        from reportlab.pdfgen import canvas
        return generate_pdf(event)
    elif action == "send_email":
        import boto3
        ses = boto3.client("ses")
        return send_email(ses, event)
    else:
        return {"statusCode": 400, "body": "Unknown action"}
```

### 1.4 デプロイパッケージの軽量化

```
パッケージサイズ vs コールドスタート:

サイズ          コールドスタート影響
  1 MB  -----  最小限 (+50ms程度)
  5 MB  -----  軽微 (+100ms程度)
 10 MB  -----  顕著 (+200ms程度)
 50 MB  -----  深刻 (+500ms以上)
250 MB  -----  非常に深刻 (+1秒以上)

対策:
  - 不要な依存を除外 (dev dependencies)
  - __pycache__、テストファイルを除外
  - 軽量な代替ライブラリを利用
  - Lambda レイヤーで共通部分を分離
  - コンテナイメージ利用時は multi-stage build
```

---

## 2. Provisioned Concurrency

### 2.1 仕組みと設定

Provisioned Concurrency は、指定した数の実行環境を事前に初期化しておく機能である。コールドスタートを完全に排除し、一貫したレイテンシを実現する。

```
通常の Lambda:
リクエスト --> [コールドスタート?] --> ハンドラ実行
                    ↑
              環境がなければ発生

Provisioned Concurrency:
                    +----- 事前初期化済み環境 1
                    |
リクエスト -------> +----- 事前初期化済み環境 2  --> ハンドラ実行
                    |                               (コールドスタートなし)
                    +----- 事前初期化済み環境 3
                    |
                    +----- 事前初期化済み環境 N

※ Provisioned を超える分は通常のオンデマンドで処理
```

```bash
# Provisioned Concurrency の設定
# まずエイリアスまたはバージョンを指定
aws lambda publish-version \
  --function-name my-api-function

aws lambda put-provisioned-concurrency-config \
  --function-name my-api-function \
  --qualifier 1 \
  --provisioned-concurrent-executions 50

# 状態確認
aws lambda get-provisioned-concurrency-config \
  --function-name my-api-function \
  --qualifier 1
```

### 2.2 Application Auto Scaling との連携

```bash
# Auto Scaling ターゲットの登録
aws application-autoscaling register-scalable-target \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --min-capacity 10 \
  --max-capacity 200

# スケジュールベースのスケーリング
aws application-autoscaling put-scheduled-action \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --scheduled-action-name "morning-scale-up" \
  --schedule "cron(0 8 * * ? *)" \
  --scalable-target-action "MinCapacity=100,MaxCapacity=500"
```

### 2.3 コスト比較

| 項目 | オンデマンド | Provisioned Concurrency |
|------|------------|------------------------|
| コールドスタート | あり | なし |
| 課金開始 | リクエスト時 | 設定時から常時 |
| リクエスト料金 | $0.20/100万回 | $0.20/100万回 |
| 実行時間料金 (x86) | $0.0000166667/GB-秒 | $0.0000097222/GB-秒 (実行時) + $0.0000041667/GB-秒 (待機時) |
| 向いている用途 | 不定期/バースト | 安定トラフィック/低レイテンシ |

---

## 3. Lambda Destinations

### 3.1 非同期呼び出しの結果ルーティング

```
非同期呼び出し
    |
    v
+------------------+
| Lambda 関数実行  |
+------------------+
    |           |
  成功        失敗
    |           |
    v           v
+---------+ +---------+
| OnSuccess| | OnFailure|
| 送信先   | | 送信先   |
+---------+ +---------+
    |           |
    v           v
  SQS         SQS
  SNS         SNS
  Lambda      Lambda
  EventBridge EventBridge
```

```bash
# Destinations の設定
aws lambda put-function-event-invoke-config \
  --function-name my-async-function \
  --maximum-retry-attempts 1 \
  --maximum-event-age-in-seconds 3600 \
  --destination-config '{
    "OnSuccess": {
      "Destination": "arn:aws:sqs:ap-northeast-1:123456789012:success-queue"
    },
    "OnFailure": {
      "Destination": "arn:aws:sqs:ap-northeast-1:123456789012:failure-queue"
    }
  }'
```

### 3.2 Destinations vs DLQ

| 機能 | Lambda Destinations | Dead Letter Queue (DLQ) |
|------|-------------------|------------------------|
| 対象イベント | 成功・失敗の両方 | 失敗のみ |
| 送信先 | SQS, SNS, Lambda, EventBridge | SQS, SNS のみ |
| ペイロード | 完全な実行コンテキスト含む | 元のイベントのみ |
| 推奨度 | 新規開発では推奨 | レガシー互換 |

---

## 4. AWS Step Functions 連携

### 4.1 ステートマシンの基本構成

```
Step Functions ステートマシン:

[Start]
    |
    v
+-------------------+
| ValidateInput     |  (Lambda)
+-------------------+
    |
    v
+-------------------+
| ProcessOrder      |  (Lambda)
+-------------------+
    |        |
  成功      失敗
    |        |
    v        v
+--------+ +-------------------+
| Notify | | HandleError       |  (Lambda)
| Success| +-------------------+
+--------+      |
    |           v
    |    +-------------------+
    |    | Notify Failure    |
    |    +-------------------+
    |           |
    v           v
  [End]       [End]
```

### 4.2 ステートマシン定義 (ASL)

```json
{
  "Comment": "注文処理ワークフロー",
  "StartAt": "ValidateInput",
  "States": {
    "ValidateInput": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-northeast-1:123456789012:function:validate-input",
      "Next": "ProcessOrder",
      "Catch": [
        {
          "ErrorEquals": ["ValidationError"],
          "Next": "HandleError"
        }
      ]
    },
    "ProcessOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-northeast-1:123456789012:function:process-order",
      "TimeoutSeconds": 300,
      "Retry": [
        {
          "ErrorEquals": ["States.TaskFailed"],
          "IntervalSeconds": 5,
          "MaxAttempts": 3,
          "BackoffRate": 2.0
        }
      ],
      "Next": "NotifySuccess",
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "Next": "HandleError"
        }
      ]
    },
    "NotifySuccess": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-northeast-1:123456789012:function:notify-success",
      "End": true
    },
    "HandleError": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-northeast-1:123456789012:function:handle-error",
      "Next": "NotifyFailure"
    },
    "NotifyFailure": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-northeast-1:123456789012:function:notify-failure",
      "End": true
    }
  }
}
```

### 4.3 Standard vs Express ワークフロー

| 特性 | Standard | Express |
|------|----------|---------|
| 最大実行時間 | 1 年 | 5 分 |
| 実行開始レート | 2,000/秒 | 100,000/秒 |
| 状態遷移レート | 4,000/秒 | 無制限 |
| 実行保証 | 正確に1回 | 最低1回 (Async) / 正確に1回 (Sync) |
| 課金 | 状態遷移ごと | 実行回数 + 実行時間 |
| 向いている用途 | 長時間ワークフロー | 大量短時間処理、IoT |

### 4.4 Step Functions の Parallel 実行

```json
{
  "Type": "Parallel",
  "Branches": [
    {
      "StartAt": "SendEmail",
      "States": {
        "SendEmail": {
          "Type": "Task",
          "Resource": "arn:aws:lambda:...:send-email",
          "End": true
        }
      }
    },
    {
      "StartAt": "SendSMS",
      "States": {
        "SendSMS": {
          "Type": "Task",
          "Resource": "arn:aws:lambda:...:send-sms",
          "End": true
        }
      }
    },
    {
      "StartAt": "UpdateDB",
      "States": {
        "UpdateDB": {
          "Type": "Task",
          "Resource": "arn:aws:lambda:...:update-db",
          "End": true
        }
      }
    }
  ],
  "Next": "AggregateResults"
}
```

---

## 5. Lambda SnapStart (Java)

### 5.1 SnapStart の仕組み

```
従来の Java Lambda:
  リクエスト --> JVM起動 --> クラスロード --> DI初期化 --> ハンドラ実行
                |<---- コールドスタート (2-5秒) ---->|

SnapStart:
  [事前] バージョン公開時にスナップショット作成
         JVM起動 --> クラスロード --> DI初期化 --> スナップショット保存

  [実行時] リクエスト --> スナップショット復元 (< 200ms) --> ハンドラ実行
```

```bash
# SnapStart の有効化
aws lambda update-function-configuration \
  --function-name my-java-function \
  --snap-start '{"ApplyOn": "PublishedVersions"}'

# バージョン公開（スナップショット作成）
aws lambda publish-version \
  --function-name my-java-function
```

---

## 6. アンチパターン

### 6.1 Lambda を VPC 内に不必要に配置する

```
[悪い例]
Lambda --VPC内--> NAT Gateway --> Internet --> DynamoDB

[良い例]
Lambda --VPC外--> DynamoDB (VPCエンドポイント不要)

Lambda --VPC内--> RDS (VPC内リソースへのアクセスが必要な場合のみ)
         +-----> VPC Endpoint --> DynamoDB
```

**問題点**: VPC 配置にすると ENI アタッチ時間が追加され、コールドスタートが増加する(改善済みだが依然として若干のオーバーヘッドあり)。NAT Gateway 経由のインターネットアクセスにはコストもかかる。

**改善**: RDS や ElastiCache など VPC 内リソースへのアクセスが本当に必要な場合のみ VPC 配置にし、DynamoDB や S3 へは VPC エンドポイント経由でアクセスする。

### 6.2 Provisioned Concurrency の過剰設定

**問題点**: トラフィックパターンを分析せず、常に最大値を設定するとコストが無駄になる。

**改善**: CloudWatch メトリクスで実際の同時実行数を分析し、Application Auto Scaling でトラフィックパターンに合わせて動的に調整する。

---

## 7. FAQ

### Q1. Provisioned Concurrency と Reserved Concurrency の違いは？

Reserved Concurrency は関数の同時実行数の「上限」を設定するもので、追加コストはない。他の関数のスロットルから特定の関数を保護する。一方、Provisioned Concurrency は指定数の実行環境を「事前に初期化」しておくもので、コールドスタートをなくす代わりに追加料金が発生する。

### Q2. Step Functions のコストはどのくらいですか？

Standard ワークフローは状態遷移ごとに $0.025/1,000 遷移で課金される。1回の実行に 5 遷移あり、月間 100 万実行の場合は $125/月となる。Express ワークフローは実行回数とメモリ・実行時間で課金され、大量・短時間処理には割安になる。

### Q3. Lambda の最大同時実行数を超えるとどうなりますか？

スロットリングが発生する。同期呼び出しでは HTTP 429 エラーが返り、非同期呼び出しではイベントキューに格納され最大 6 時間リトライされる。Service Quotas からクォータ引き上げを申請するか、Reserved Concurrency で重要な関数のキャパシティを確保することで対処する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| コールドスタート | Init コードの最適化、パッケージ軽量化、メモリ増強で緩和 |
| Provisioned Concurrency | 事前に実行環境を確保しコールドスタートを排除 |
| Lambda Destinations | 非同期実行の成功・失敗を柔軟にルーティング |
| Step Functions | 複数 Lambda のオーケストレーション、エラーハンドリング、リトライ |
| SnapStart | Java のコールドスタートをスナップショット復元で大幅短縮 |
| VPC 配置 | 必要な場合のみ。VPC エンドポイントを積極的に活用 |

---

## 次に読むべきガイド

- [サーバーレスパターン](./02-serverless-patterns.md) -- 実践的なアーキテクチャパターン
- [CloudFormation](../07-devops/00-cloudformation.md) -- Lambda のインフラをコード化
- [コスト最適化](../09-cost/00-cost-optimization.md) -- Lambda のコスト管理

---

## 参考文献

1. AWS 公式ドキュメント「Lambda のパフォーマンスの最適化」 https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html
2. AWS 公式ドキュメント「AWS Step Functions デベロッパーガイド」 https://docs.aws.amazon.com/step-functions/latest/dg/
3. Yan Cui「Production-Ready Serverless」Manning Publications, 2019
4. AWS re:Invent 2023「SVS404: Optimizing Lambda performance for your serverless applications」
