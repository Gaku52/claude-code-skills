# AWS Lambda 応用

> コールドスタートの最適化、Provisioned Concurrency、Lambda Destinations、Step Functions 連携を理解し、本番運用品質のサーバーレスアプリケーションを構築する。

---

## この章で学ぶこと

1. **コールドスタートの原因と最適化手法** -- コールドスタートが発生するメカニズムを理解し、ランタイム選択やパッケージ軽量化で実戦的に対処する
2. **Provisioned Concurrency と同時実行制御** -- レイテンシ要件が厳しいワークロードに対して予め実行環境を確保する方法を習得する
3. **Lambda Destinations と Step Functions** -- 非同期処理の結果ルーティングとオーケストレーションでエラーハンドリングを設計する
4. **Lambda レイヤーとカスタムランタイム** -- 共通ライブラリの効率的な管理とカスタムランタイムの構築方法を学ぶ
5. **Lambda のモニタリングとデバッグ** -- X-Ray、CloudWatch Logs Insights、Lambda Insights を活用して本番環境の問題を迅速に特定する
6. **Lambda のセキュリティベストプラクティス** -- 最小権限の原則、VPC 設計、シークレット管理を実践する

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
| Rust (AL2023) | 30-100 ms | 20-80 ms | Go 同様にネイティブバイナリ |

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

```python
# [最適化] コネクションプールの再利用パターン
import boto3
import os
from botocore.config import Config

# Init Phase: SDK クライアントの設定を最適化
config = Config(
    retries={"max_attempts": 3, "mode": "adaptive"},
    max_pool_connections=10,
    connect_timeout=5,
    read_timeout=10,
)

# 各 AWS サービスクライアントを Init Phase で作成
dynamodb_client = boto3.client("dynamodb", config=config)
s3_client = boto3.client("s3", config=config)
sqs_client = boto3.client("sqs", config=config)

BUCKET_NAME = os.environ["BUCKET_NAME"]
QUEUE_URL = os.environ["QUEUE_URL"]

def lambda_handler(event, context):
    """全クライアントは Warm 起動時に再利用される"""
    # DynamoDB からデータ取得
    item = dynamodb_client.get_item(
        TableName="my-table",
        Key={"pk": {"S": event["id"]}}
    )

    # S3 にレポートを保存
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=f"reports/{event['id']}.json",
        Body=json.dumps(item.get("Item", {})),
        ContentType="application/json"
    )

    # SQS に通知を送信
    sqs_client.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({"status": "completed", "id": event["id"]})
    )

    return {"statusCode": 200, "body": "Processing complete"}
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

```bash
# Python での軽量パッケージ作成例
# 1. 本番依存のみインストール
pip install -r requirements.txt -t ./package --no-cache-dir

# 2. 不要ファイルの除去
cd package
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null

# 3. ZIP パッケージの作成
zip -r9 ../function.zip .
cd ..
zip -g function.zip lambda_function.py

# 4. サイズ確認
ls -lh function.zip

# 5. デプロイ
aws lambda update-function-code \
  --function-name my-function \
  --zip-file fileb://function.zip
```

```bash
# Node.js での軽量パッケージ作成例
# 1. 本番依存のみインストール
npm ci --only=production

# 2. esbuild でバンドル (tree-shaking 付き)
npx esbuild src/handler.ts \
  --bundle \
  --minify \
  --sourcemap \
  --platform=node \
  --target=node20 \
  --outfile=dist/handler.js \
  --external:@aws-sdk/*

# 3. ZIP パッケージの作成
cd dist
zip -r9 ../function.zip .

# AWS SDK v3 は Lambda ランタイムに組み込み済みのため
# --external:@aws-sdk/* で除外してサイズ削減
```

### 1.5 メモリとCPUの関係

```
Lambda のメモリとCPUの比例関係:

メモリ    CPU パワー    ネットワーク帯域
 128 MB   最小 (部分)   低
 256 MB   低           低
 512 MB   中           中
1024 MB   中～高       中
1769 MB   1 vCPU 相当  高
3538 MB   2 vCPU 相当  高
 10 GB    6 vCPU 相当  最大

ポイント:
  - 1,769 MB で 1 vCPU が完全に割り当てられる
  - CPU バウンドな処理はメモリ増強で高速化できる
  - メモリ増強によりコールドスタートも短縮される
  - コスト = 実行時間 x メモリ のため、
    メモリ倍増 → 実行時間半減なら同コストで高速に
```

```python
# メモリサイズの最適化を自動テストするスクリプト
import boto3
import json
import time
import statistics

lambda_client = boto3.client("lambda")

def benchmark_memory_sizes(function_name, payload, memory_sizes, iterations=10):
    """異なるメモリサイズでの実行時間を比較する"""
    results = {}

    for memory_size in memory_sizes:
        # メモリサイズを変更
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            MemorySize=memory_size
        )
        time.sleep(5)  # 設定反映を待つ

        durations = []
        for i in range(iterations):
            response = lambda_client.invoke(
                FunctionName=function_name,
                Payload=json.dumps(payload)
            )
            # レスポンスヘッダから実行時間を取得
            log_result = response.get("LogResult", "")
            # Duration を解析
            duration = float(response["ResponseMetadata"]["HTTPHeaders"]
                           .get("x-amz-log-result", "0"))
            durations.append(duration)

        results[memory_size] = {
            "avg_duration_ms": statistics.mean(durations),
            "p99_duration_ms": sorted(durations)[int(len(durations) * 0.99)],
            "cost_per_invocation": (memory_size / 1024) * (statistics.mean(durations) / 1000) * 0.0000166667,
        }

    return results
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

# 設定一覧の確認
aws lambda list-provisioned-concurrency-configs \
  --function-name my-api-function

# Provisioned Concurrency の削除
aws lambda delete-provisioned-concurrency-config \
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

# ターゲット追跡スケーリングポリシー
aws application-autoscaling put-scaling-policy \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --policy-name "provisioned-concurrency-target-tracking" \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 0.7,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "LambdaProvisionedConcurrencyUtilization"
    },
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'

# スケジュールベースのスケーリング
aws application-autoscaling put-scheduled-action \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --scheduled-action-name "morning-scale-up" \
  --schedule "cron(0 8 * * ? *)" \
  --scalable-target-action "MinCapacity=100,MaxCapacity=500"

# 夜間のスケールダウン
aws application-autoscaling put-scheduled-action \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --scheduled-action-name "night-scale-down" \
  --schedule "cron(0 22 * * ? *)" \
  --scalable-target-action "MinCapacity=10,MaxCapacity=50"
```

### 2.3 コスト比較

| 項目 | オンデマンド | Provisioned Concurrency |
|------|------------|------------------------|
| コールドスタート | あり | なし |
| 課金開始 | リクエスト時 | 設定時から常時 |
| リクエスト料金 | $0.20/100万回 | $0.20/100万回 |
| 実行時間料金 (x86) | $0.0000166667/GB-秒 | $0.0000097222/GB-秒 (実行時) + $0.0000041667/GB-秒 (待機時) |
| 向いている用途 | 不定期/バースト | 安定トラフィック/低レイテンシ |

```
Provisioned Concurrency のコスト試算例:

シナリオ: API バックエンド
  - メモリ: 1 GB
  - 平均実行時間: 200 ms
  - リクエスト数: 100万回/月
  - Provisioned 数: 50

オンデマンドの場合:
  リクエスト料金: 100万 x $0.20/100万 = $0.20
  実行時間料金: 100万 x 0.2秒 x 1GB x $0.0000166667 = $3.33
  合計: $3.53/月

Provisioned Concurrency の場合:
  リクエスト料金: $0.20
  実行時間料金: 100万 x 0.2秒 x 1GB x $0.0000097222 = $1.94
  待機時間料金: 50 x 30日 x 24時間 x 3600秒 x 1GB x $0.0000041667 = $540.00
  合計: $542.14/月

→ Provisioned は高額だが、コールドスタートなしの一貫したレイテンシを実現
→ Auto Scaling でトラフィックパターンに合わせて調整することでコスト最適化可能
→ 24時間常時50ではなく、ピーク時のみ高い値に設定するのが現実的
```

### 2.4 Reserved Concurrency との組み合わせ

```
同時実行制御の階層:

アカウント全体の同時実行数上限: 1,000 (デフォルト)
    |
    +-- 関数A: Reserved Concurrency = 200
    |       |
    |       +-- Provisioned: 50 (200のうち50を事前初期化)
    |       +-- オンデマンド: 残り150まで利用可能
    |
    +-- 関数B: Reserved Concurrency = 100
    |       |
    |       +-- 全てオンデマンド
    |
    +-- 他の関数: 残り700を共有 (Unreserved)

Reserved Concurrency の設定:
  - 関数の同時実行数の「上限」を設定
  - 追加コストなし
  - 他の関数からのスロットル保護
  - Provisioned と併用可能
```

```bash
# Reserved Concurrency の設定
aws lambda put-function-concurrency \
  --function-name my-api-function \
  --reserved-concurrent-executions 200

# Reserved Concurrency の確認
aws lambda get-function-concurrency \
  --function-name my-api-function

# Reserved Concurrency の削除 (アカウントプールに戻す)
aws lambda delete-function-concurrency \
  --function-name my-api-function
```

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

# 設定の確認
aws lambda get-function-event-invoke-config \
  --function-name my-async-function

# 設定の削除
aws lambda delete-function-event-invoke-config \
  --function-name my-async-function
```

### 3.2 Destinations vs DLQ

| 機能 | Lambda Destinations | Dead Letter Queue (DLQ) |
|------|-------------------|------------------------|
| 対象イベント | 成功・失敗の両方 | 失敗のみ |
| 送信先 | SQS, SNS, Lambda, EventBridge | SQS, SNS のみ |
| ペイロード | 完全な実行コンテキスト含む | 元のイベントのみ |
| 推奨度 | 新規開発では推奨 | レガシー互換 |

### 3.3 Destinations のペイロード構造

```json
{
  "version": "1.0",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "requestContext": {
    "requestId": "abc123-def456-ghi789",
    "functionArn": "arn:aws:lambda:ap-northeast-1:123456789012:function:my-function",
    "condition": "Success",
    "approximateInvokeCount": 1
  },
  "requestPayload": {
    "orderId": "ORD-001",
    "amount": 5000
  },
  "responseContext": {
    "statusCode": 200,
    "executedVersion": "$LATEST",
    "functionError": null
  },
  "responsePayload": {
    "statusCode": 200,
    "body": "{\"message\": \"Order processed successfully\"}"
  }
}
```

### 3.4 EventBridge を活用したイベント駆動パターン

```python
# Lambda Destination を EventBridge に設定し、
# 複数の後続処理をイベントルールで分岐させるパターン

import json
import boto3

eventbridge = boto3.client("events")

def order_processor(event, context):
    """注文処理Lambda - 成功時にEventBridgeへ送信"""
    order_id = event["orderId"]
    amount = event["amount"]

    # 注文処理ロジック
    result = process_order(order_id, amount)

    return {
        "statusCode": 200,
        "orderId": order_id,
        "processedAmount": amount,
        "status": "COMPLETED"
    }

# EventBridge ルールでの後続処理分岐:
# ルール1: amount > 10000 → 高額注文通知 Lambda
# ルール2: 全注文 → 注文履歴 DynamoDB 書き込み Lambda
# ルール3: status=COMPLETED → 配送手配 Step Functions
```

```yaml
# EventBridge ルール (CloudFormation)
HighValueOrderRule:
  Type: AWS::Events::Rule
  Properties:
    Name: high-value-order-notification
    EventPattern:
      source:
        - "lambda"
      detail-type:
        - "Lambda Function Invocation Result - Success"
      detail:
        requestContext:
          functionArn:
            - !GetAtt OrderProcessorFunction.Arn
        responsePayload:
          processedAmount:
            - numeric: [">=", 10000]
    Targets:
      - Arn: !GetAtt HighValueNotificationFunction.Arn
        Id: HighValueNotification
```

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

### 4.5 Map ステートによる動的並列処理

```json
{
  "Comment": "大量データの並列処理ワークフロー",
  "StartAt": "FetchItems",
  "States": {
    "FetchItems": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:fetch-items",
      "Next": "ProcessItems"
    },
    "ProcessItems": {
      "Type": "Map",
      "ItemsPath": "$.items",
      "MaxConcurrency": 10,
      "Iterator": {
        "StartAt": "ProcessSingleItem",
        "States": {
          "ProcessSingleItem": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:process-item",
            "Retry": [
              {
                "ErrorEquals": ["States.TaskFailed"],
                "IntervalSeconds": 2,
                "MaxAttempts": 3,
                "BackoffRate": 2.0
              }
            ],
            "End": true
          }
        }
      },
      "Next": "AggregateResults"
    },
    "AggregateResults": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:aggregate-results",
      "End": true
    }
  }
}
```

### 4.6 Distributed Map (大規模並列処理)

```json
{
  "Comment": "S3 の大量ファイルを分散並列処理",
  "StartAt": "DistributedProcess",
  "States": {
    "DistributedProcess": {
      "Type": "Map",
      "ItemReader": {
        "Resource": "arn:aws:states:::s3:listObjectsV2",
        "Parameters": {
          "Bucket": "my-input-bucket",
          "Prefix": "data/"
        }
      },
      "ItemProcessor": {
        "ProcessorConfig": {
          "Mode": "DISTRIBUTED",
          "ExecutionType": "EXPRESS"
        },
        "StartAt": "ProcessFile",
        "States": {
          "ProcessFile": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:...:process-file",
            "End": true
          }
        }
      },
      "MaxConcurrency": 1000,
      "Label": "DistributedProcess",
      "ResultWriter": {
        "Resource": "arn:aws:states:::s3:putObject",
        "Parameters": {
          "Bucket": "my-output-bucket",
          "Prefix": "results/"
        }
      },
      "End": true
    }
  }
}
```

### 4.7 Step Functions SDK 統合 (Optimized Integration)

```json
{
  "Comment": "AWS SDK 統合による直接サービス呼び出し",
  "StartAt": "PutItemToDynamoDB",
  "States": {
    "PutItemToDynamoDB": {
      "Type": "Task",
      "Resource": "arn:aws:states:::dynamodb:putItem",
      "Parameters": {
        "TableName": "Orders",
        "Item": {
          "orderId": {"S.$": "$.orderId"},
          "status": {"S": "PENDING"},
          "createdAt": {"S.$": "$$.State.EnteredTime"}
        }
      },
      "Next": "PublishToSNS"
    },
    "PublishToSNS": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "arn:aws:sns:ap-northeast-1:123456789012:order-notifications",
        "Message.$": "States.Format('New order: {}', $.orderId)",
        "Subject": "New Order Received"
      },
      "Next": "StartECSTask"
    },
    "StartECSTask": {
      "Type": "Task",
      "Resource": "arn:aws:states:::ecs:runTask.sync",
      "Parameters": {
        "LaunchType": "FARGATE",
        "Cluster": "arn:aws:ecs:...:cluster/my-cluster",
        "TaskDefinition": "arn:aws:ecs:...:task-definition/process-order:1",
        "Overrides": {
          "ContainerOverrides": [
            {
              "Name": "processor",
              "Environment": [
                {"Name": "ORDER_ID", "Value.$": "$.orderId"}
              ]
            }
          ]
        },
        "NetworkConfiguration": {
          "AwsvpcConfiguration": {
            "Subnets": ["subnet-111", "subnet-222"],
            "SecurityGroups": ["sg-12345678"]
          }
        }
      },
      "End": true
    }
  }
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

# SnapStart の状態確認
aws lambda get-function-configuration \
  --function-name my-java-function \
  --query 'SnapStart'
```

### 5.2 SnapStart の注意点

```
SnapStart 利用時の注意事項:

1. 一意性の問題:
   スナップショットから複数の実行環境が復元されるため、
   Init Phase で生成した乱数やUUIDが重複する可能性がある。

   [対策]
   - java.util.Random の初期化をハンドラ内で行う
   - afterRestore フックで状態をリセットする

2. ネットワーク接続の問題:
   Init Phase で確立したDB接続はスナップショット復元後に無効。

   [対策]
   - afterRestore フックでコネクションを再確立
   - コネクションプーリングライブラリの再初期化

3. 対応ランタイム:
   - Java 11 以降 (Corretto)
   - arm64 / x86_64 両対応
```

```java
// SnapStart の afterRestore フック例
import org.crac.Context;
import org.crac.Core;
import org.crac.Resource;

public class MyHandler implements RequestHandler<APIGatewayProxyRequestEvent, APIGatewayProxyResponseEvent>,
                                   Resource {

    private Connection dbConnection;

    public MyHandler() {
        // Init Phase: CRaC リソースとして登録
        Core.getGlobalContext().register(this);
        // DB接続を確立
        this.dbConnection = DriverManager.getConnection(DB_URL);
    }

    @Override
    public void afterRestore(Context<? extends Resource> context) {
        // スナップショット復元後に呼ばれる
        // DB接続を再確立
        this.dbConnection = DriverManager.getConnection(DB_URL);
        // 乱数生成器を再シード
        SecureRandom.getInstanceStrong();
    }

    @Override
    public APIGatewayProxyResponseEvent handleRequest(
            APIGatewayProxyRequestEvent event, com.amazonaws.services.lambda.runtime.Context context) {
        // ハンドラロジック
        return new APIGatewayProxyResponseEvent().withStatusCode(200);
    }
}
```

---

## 6. Lambda レイヤー

### 6.1 レイヤーの概念

```
Lambda レイヤーの仕組み:

+------------------------------------------+
| Lambda 関数                               |
| +--------------------------------------+ |
| | /var/task (関数コード)                 | |
| | +----------------------------------+ | |
| | | lambda_function.py               | | |
| | +----------------------------------+ | |
| +--------------------------------------+ |
| +--------------------------------------+ |
| | /opt (レイヤー 1 + 2 + ... + N)     | |
| | +----------------------------------+ | |
| | | /opt/python/共通ライブラリ         | | |
| | | /opt/bin/カスタムバイナリ          | | |
| | | /opt/lib/共有ライブラリ           | | |
| | +----------------------------------+ | |
| +--------------------------------------+ |
+------------------------------------------+

レイヤーのメリット:
  - 共通ライブラリの一元管理
  - デプロイパッケージの軽量化
  - 最大5レイヤーまで重ね合わせ可能
  - レイヤー単位でバージョン管理
```

### 6.2 レイヤーの作成と管理

```bash
# Python ライブラリのレイヤー作成
mkdir -p python/lib/python3.12/site-packages
pip install requests boto3-stubs[s3,dynamodb] \
  -t python/lib/python3.12/site-packages

# ZIP パッケージ作成
zip -r9 my-layer.zip python/

# レイヤーの公開
aws lambda publish-layer-version \
  --layer-name my-common-libs \
  --description "共通ライブラリ (requests, boto3-stubs)" \
  --compatible-runtimes python3.11 python3.12 \
  --compatible-architectures x86_64 arm64 \
  --zip-file fileb://my-layer.zip

# レイヤーを関数にアタッチ
aws lambda update-function-configuration \
  --function-name my-function \
  --layers \
    arn:aws:lambda:ap-northeast-1:123456789012:layer:my-common-libs:1 \
    arn:aws:lambda:ap-northeast-1:123456789012:layer:my-utilities:3

# レイヤーバージョンの一覧
aws lambda list-layer-versions \
  --layer-name my-common-libs

# レイヤーの削除
aws lambda delete-layer-version \
  --layer-name my-common-libs \
  --version-number 1
```

### 6.3 共有ユーティリティレイヤーの実装例

```python
# レイヤーに含めるユーティリティモジュール
# python/lib/python3.12/site-packages/common/response.py

import json
from typing import Any, Optional

def api_response(
    status_code: int,
    body: Any,
    headers: Optional[dict] = None
) -> dict:
    """API Gateway 用の標準レスポンスを生成する"""
    default_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "GET,POST,PUT,DELETE,OPTIONS",
    }
    if headers:
        default_headers.update(headers)

    return {
        "statusCode": status_code,
        "headers": default_headers,
        "body": json.dumps(body, ensure_ascii=False, default=str),
    }

def error_response(
    status_code: int,
    message: str,
    error_code: Optional[str] = None
) -> dict:
    """エラーレスポンスを生成する"""
    body = {"error": {"message": message}}
    if error_code:
        body["error"]["code"] = error_code
    return api_response(status_code, body)
```

```python
# レイヤーに含めるロギングモジュール
# python/lib/python3.12/site-packages/common/logger.py

import json
import logging
import os
import sys
from datetime import datetime, timezone

class StructuredLogger:
    """構造化ログを出力するロガー"""

    def __init__(self, service_name: str = None):
        self.service_name = service_name or os.environ.get("SERVICE_NAME", "unknown")
        self.logger = logging.getLogger(self.service_name)
        self.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

        # JSON フォーマッタの設定
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.handlers = [handler]

    def _format(self, level: str, message: str, **kwargs) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "service": self.service_name,
            "message": message,
            **kwargs
        }
        return json.dumps(log_entry, ensure_ascii=False, default=str)

    def info(self, message: str, **kwargs):
        self.logger.info(self._format("INFO", message, **kwargs))

    def error(self, message: str, **kwargs):
        self.logger.error(self._format("ERROR", message, **kwargs))

    def warning(self, message: str, **kwargs):
        self.logger.warning(self._format("WARNING", message, **kwargs))

    def debug(self, message: str, **kwargs):
        self.logger.debug(self._format("DEBUG", message, **kwargs))
```

---

## 7. Lambda のモニタリングとデバッグ

### 7.1 CloudWatch Logs Insights によるログ分析

```
# エラーログの検索
fields @timestamp, @message, @requestId
| filter @message like /ERROR/
| sort @timestamp desc
| limit 50

# コールドスタートの検出
filter @message like /Init Duration/
| parse @message "Init Duration: * ms" as initDuration
| stats count() as coldStarts,
        avg(initDuration) as avgInitMs,
        max(initDuration) as maxInitMs,
        pct(initDuration, 99) as p99InitMs
  by bin(1h)

# 実行時間の分析
filter @type = "REPORT"
| parse @message "Duration: * ms" as duration
| parse @message "Billed Duration: * ms" as billedDuration
| parse @message "Memory Size: * MB" as memorySize
| parse @message "Max Memory Used: * MB" as memoryUsed
| stats avg(duration) as avgDuration,
        max(duration) as maxDuration,
        pct(duration, 95) as p95Duration,
        pct(duration, 99) as p99Duration,
        avg(memoryUsed/memorySize * 100) as avgMemoryUtilization
  by bin(5m)

# タイムアウトの検出
filter @message like /Task timed out/
| parse @message "Task timed out after * seconds" as timeout
| stats count() by bin(1h)
```

### 7.2 X-Ray によるトレーシング

```bash
# Lambda 関数の X-Ray トレーシングを有効化
aws lambda update-function-configuration \
  --function-name my-function \
  --tracing-config Mode=Active

# X-Ray トレースの取得
aws xray get-trace-summaries \
  --start-time $(date -d '1 hour ago' +%s) \
  --end-time $(date +%s) \
  --filter-expression 'service("my-function")'
```

```python
# X-Ray SDK による詳細トレーシング
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# 全 AWS SDK 呼び出しを自動トレース
patch_all()

@xray_recorder.capture("process_order")
def process_order(order_data):
    """カスタムサブセグメントでビジネスロジックをトレース"""

    # アノテーションの追加 (フィルタリング用)
    subsegment = xray_recorder.current_subsegment()
    subsegment.put_annotation("order_id", order_data["orderId"])
    subsegment.put_annotation("customer_tier", order_data.get("tier", "standard"))

    # メタデータの追加 (デバッグ用)
    subsegment.put_metadata("order_details", order_data, "order")

    # ビジネスロジック
    result = validate_order(order_data)
    return result

def lambda_handler(event, context):
    return process_order(event)
```

### 7.3 Lambda Insights

```bash
# Lambda Insights の有効化 (拡張モニタリング)
aws lambda update-function-configuration \
  --function-name my-function \
  --layers \
    "arn:aws:lambda:ap-northeast-1:580247275435:layer:LambdaInsightsExtension:38"

# Lambda Insights が収集するメトリクス:
#   - cpu_total_time: CPU使用時間
#   - memory_utilization: メモリ使用率
#   - rx_bytes / tx_bytes: ネットワーク I/O
#   - init_duration: Init Phase の時間
#   - tmp_max: /tmp 使用量
```

### 7.4 カスタムメトリクスの埋め込み (EMF)

```python
# Embedded Metric Format (EMF) による
# Lambda からの高解像度カスタムメトリクス出力

import json
import time

def put_metric(namespace, metric_name, value, unit="None", dimensions=None):
    """EMF形式でCloudWatchカスタムメトリクスを出力"""
    emf_log = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": namespace,
                    "Dimensions": [list(dimensions.keys())] if dimensions else [[]],
                    "Metrics": [
                        {"Name": metric_name, "Unit": unit}
                    ]
                }
            ]
        },
        metric_name: value
    }
    if dimensions:
        emf_log.update(dimensions)

    # 標準出力に書くだけで CloudWatch メトリクスとして記録される
    print(json.dumps(emf_log))

def lambda_handler(event, context):
    start = time.time()

    # ビジネスロジック
    result = process_request(event)

    # カスタムメトリクスの出力
    elapsed = (time.time() - start) * 1000
    put_metric(
        "MyApplication",
        "ProcessingTime",
        elapsed,
        "Milliseconds",
        {"Environment": "production", "Service": "order-api"}
    )
    put_metric(
        "MyApplication",
        "OrdersProcessed",
        1,
        "Count",
        {"Environment": "production", "Service": "order-api"}
    )

    return result
```

---

## 8. Lambda のセキュリティ

### 8.1 最小権限の IAM ポリシー

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "DynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query"
      ],
      "Resource": [
        "arn:aws:dynamodb:ap-northeast-1:123456789012:table/Orders",
        "arn:aws:dynamodb:ap-northeast-1:123456789012:table/Orders/index/*"
      ]
    },
    {
      "Sid": "S3ReadAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": "arn:aws:s3:::my-config-bucket/config/*"
    },
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:my-api-key-*"
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

### 8.2 Secrets Manager / Parameter Store 統合

```python
# Secrets Manager からシークレットを取得する
# Lambda Extensions を使ったキャッシュ方式

import json
import os
import urllib3

# Lambda Extensions のキャッシュポート
SECRETS_EXTENSION_PORT = 2773
http = urllib3.PoolManager()

def get_secret(secret_name):
    """Secrets Manager Lambda Extension 経由でシークレットを取得"""
    url = (
        f"http://localhost:{SECRETS_EXTENSION_PORT}"
        f"/secretsmanager/get?secretId={secret_name}"
    )
    headers = {
        "X-Aws-Parameters-Secrets-Token": os.environ["AWS_SESSION_TOKEN"]
    }
    response = http.request("GET", url, headers=headers)
    return json.loads(response.data)["SecretString"]

# Init Phase でシークレットを取得 (キャッシュされる)
DB_CREDENTIALS = json.loads(get_secret("prod/db-credentials"))
API_KEY = get_secret("prod/external-api-key")

def lambda_handler(event, context):
    # シークレットを使用
    db_host = DB_CREDENTIALS["host"]
    db_password = DB_CREDENTIALS["password"]
    # ...
```

### 8.3 VPC Lambda のセキュリティ設計

```
VPC Lambda のネットワーク設計:

+----------------------------------------------------------+
|  VPC (10.0.0.0/16)                                       |
|                                                          |
|  Private Subnet A              Private Subnet B           |
|  +------------------------+   +------------------------+ |
|  | Lambda ENI             |   | Lambda ENI             | |
|  | (自動生成)              |   | (自動生成)              | |
|  +------------------------+   +------------------------+ |
|       |                            |                      |
|       v                            v                      |
|  +---------------------------------------------------+   |
|  | セキュリティグループ (Lambda-SG)                     |   |
|  | Outbound: 必要なポートのみ                          |   |
|  +---------------------------------------------------+   |
|       |                                                   |
|       +---> RDS (DB-SG: Lambda-SG からの 3306 許可)       |
|       |                                                   |
|       +---> ElastiCache (Cache-SG: Lambda-SG からの       |
|       |     6379 許可)                                    |
|       |                                                   |
|       +---> VPC Endpoint (DynamoDB, S3, SQS)             |
|       |     (NAT Gateway 不要)                            |
|       |                                                   |
|       +---> NAT Gateway --> IGW --> Internet              |
|             (外部API呼び出しが必要な場合のみ)              |
+----------------------------------------------------------+
```

```bash
# VPC Lambda 用のセキュリティグループ作成
aws ec2 create-security-group \
  --group-name lambda-sg \
  --description "Security group for Lambda functions" \
  --vpc-id vpc-12345678

# RDS へのアクセスを許可 (RDS の SG に追加)
aws ec2 authorize-security-group-ingress \
  --group-id sg-rds-12345678 \
  --protocol tcp \
  --port 3306 \
  --source-group sg-lambda-12345678

# VPC Endpoint の作成 (DynamoDB)
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.ap-northeast-1.dynamodb \
  --route-table-ids rtb-12345678

# VPC Endpoint の作成 (S3)
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.ap-northeast-1.s3 \
  --route-table-ids rtb-12345678

# VPC Endpoint の作成 (SQS - Interface 型)
aws ec2 create-vpc-endpoint \
  --vpc-id vpc-12345678 \
  --service-name com.amazonaws.ap-northeast-1.sqs \
  --vpc-endpoint-type Interface \
  --subnet-ids subnet-111 subnet-222 \
  --security-group-ids sg-vpce-12345678
```

---

## 9. Lambda のコンテナイメージサポート

### 9.1 コンテナイメージでのデプロイ

```dockerfile
# Lambda コンテナイメージの Dockerfile 例 (Python)
FROM public.ecr.aws/lambda/python:3.12

# 依存関係のインストール
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt --no-cache-dir

# 関数コードのコピー
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# ハンドラの指定
CMD ["lambda_function.lambda_handler"]
```

```bash
# コンテナイメージのビルドとデプロイ
# 1. ECR リポジトリの作成
aws ecr create-repository \
  --repository-name my-lambda-function \
  --image-scanning-configuration scanOnPush=true

# 2. イメージのビルド
docker build -t my-lambda-function:latest \
  --platform linux/amd64 .

# 3. ECR にプッシュ
ECR_URI=123456789012.dkr.ecr.ap-northeast-1.amazonaws.com
aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_URI}
docker tag my-lambda-function:latest ${ECR_URI}/my-lambda-function:latest
docker push ${ECR_URI}/my-lambda-function:latest

# 4. Lambda 関数の作成
aws lambda create-function \
  --function-name my-container-function \
  --package-type Image \
  --code ImageUri=${ECR_URI}/my-lambda-function:latest \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --memory-size 1024 \
  --timeout 30
```

### 9.2 マルチステージビルドによる最適化

```dockerfile
# マルチステージビルドで軽量なLambdaコンテナを作成
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --user -r requirements.txt --no-cache-dir

# 本番ステージ
FROM public.ecr.aws/lambda/python:3.12

# ビルドステージからの依存関係コピー
COPY --from=builder /root/.local/lib/python3.12/site-packages ${LAMBDA_TASK_ROOT}/
COPY app/ ${LAMBDA_TASK_ROOT}/app/
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]
```

---

## 10. Lambda 関数 URL

### 10.1 関数 URL の設定

```bash
# 関数 URL の作成 (IAM 認証なし)
aws lambda create-function-url-config \
  --function-name my-api-function \
  --auth-type NONE \
  --cors '{
    "AllowCredentials": false,
    "AllowHeaders": ["content-type", "authorization"],
    "AllowMethods": ["GET", "POST", "PUT", "DELETE"],
    "AllowOrigins": ["https://example.com"],
    "ExposeHeaders": ["x-request-id"],
    "MaxAge": 86400
  }'

# リソースベースポリシーの追加 (パブリックアクセス)
aws lambda add-permission \
  --function-name my-api-function \
  --statement-id AllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE

# 関数 URL の取得
aws lambda get-function-url-config \
  --function-name my-api-function

# IAM 認証付き関数 URL
aws lambda create-function-url-config \
  --function-name my-internal-api \
  --auth-type AWS_IAM
```

### 10.2 API Gateway vs 関数 URL

| 機能 | API Gateway | Lambda 関数 URL |
|------|------------|----------------|
| 料金 | リクエスト + データ転送 | 無料 (Lambda 料金のみ) |
| カスタムドメイン | あり | CloudFront 経由で可能 |
| 認証 | Cognito, API Key, Lambda オーソライザー | IAM or なし |
| レート制限 | あり | なし (Lambda 同時実行数のみ) |
| WAF 統合 | あり | CloudFront 経由で可能 |
| キャッシュ | あり | なし |
| リクエスト変換 | あり | なし |
| 用途 | 本格的な API | シンプルな API, Webhook |

---

## 11. アンチパターン

### 11.1 Lambda を VPC 内に不必要に配置する

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

### 11.2 Provisioned Concurrency の過剰設定

**問題点**: トラフィックパターンを分析せず、常に最大値を設定するとコストが無駄になる。

**改善**: CloudWatch メトリクスで実際の同時実行数を分析し、Application Auto Scaling でトラフィックパターンに合わせて動的に調整する。

### 11.3 Lambda 関数のモノリス化

```
[悪い例]
1つの Lambda 関数に全APIエンドポイントの処理を詰め込む:
  /users GET, POST, PUT, DELETE
  /orders GET, POST, PUT, DELETE
  /products GET, POST, PUT, DELETE
  → パッケージが肥大化、デプロイが全APIに影響

[良い例]
機能単位で関数を分離:
  user-get-function
  user-create-function
  order-process-function
  → 各関数が軽量、独立してデプロイ・スケール可能

[バランスの取れたアプローチ]
リソース単位で関数を分離:
  user-api-function (User の CRUD をまとめる)
  order-api-function (Order の CRUD をまとめる)
  → 関数数の爆発を防ぎつつ、適度に分離
```

### 11.4 同期呼び出しの連鎖

```
[悪い例]
API GW -> Lambda A -> Lambda B -> Lambda C
  各呼び出しがタイムアウトを待つ
  → レイテンシが累積、エラーハンドリングが複雑

[良い例]
API GW -> Lambda A -> SQS -> Lambda B -> SQS -> Lambda C
  非同期処理で分離
  → 各関数が独立、リトライも個別に制御

[Step Functions を使う場合]
API GW -> Step Functions
            -> Lambda A (Validate)
            -> Lambda B (Process)
            -> Lambda C (Notify)
  → オーケストレーション、エラーハンドリング、リトライが統一的に管理
```

---

## 12. CloudFormation / CDK テンプレート

### 12.1 CloudFormation テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: 'Lambda 応用構成テンプレート'

Parameters:
  EnvironmentName:
    Type: String
    Default: dev
    AllowedValues: [dev, stg, prod]

  ProvisionedConcurrency:
    Type: Number
    Default: 0
    Description: 'Provisioned Concurrency 数 (0=無効)'

Conditions:
  EnableProvisionedConcurrency: !Not [!Equals [!Ref ProvisionedConcurrency, 0]]
  IsProduction: !Equals [!Ref EnvironmentName, prod]

Globals:
  Function:
    Runtime: python3.12
    MemorySize: 1024
    Timeout: 30
    Tracing: Active
    Environment:
      Variables:
        ENVIRONMENT: !Ref EnvironmentName
        TABLE_NAME: !Ref OrdersTable
        LOG_LEVEL: !If [IsProduction, INFO, DEBUG]

Resources:
  # Lambda 関数
  OrderApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: !Sub '${EnvironmentName}-order-api'
      Handler: app.lambda_handler
      CodeUri: src/order-api/
      AutoPublishAlias: live
      ProvisionedConcurrencyConfig:
        !If
        - EnableProvisionedConcurrency
        - ProvisionedConcurrentExecutions: !Ref ProvisionedConcurrency
        - !Ref AWS::NoValue
      DeploymentPreference:
        Type: !If [IsProduction, Linear10PercentEvery1Minute, AllAtOnce]
        Alarms:
          - !Ref OrderApiErrorAlarm
      Events:
        GetOrder:
          Type: Api
          Properties:
            Path: /orders/{orderId}
            Method: get
            RestApiId: !Ref ApiGateway
        CreateOrder:
          Type: Api
          Properties:
            Path: /orders
            Method: post
            RestApiId: !Ref ApiGateway
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref OrdersTable
        - SQSSendMessagePolicy:
            QueueName: !GetAtt OrderQueue.QueueName

  # Step Functions ステートマシン
  OrderProcessingStateMachine:
    Type: AWS::StepFunctions::StateMachine
    Properties:
      StateMachineName: !Sub '${EnvironmentName}-order-processing'
      DefinitionString: !Sub |
        {
          "Comment": "注文処理ワークフロー",
          "StartAt": "ValidateOrder",
          "States": {
            "ValidateOrder": {
              "Type": "Task",
              "Resource": "${ValidateOrderFunction.Arn}",
              "Next": "ProcessPayment",
              "Catch": [{"ErrorEquals": ["ValidationError"], "Next": "FailOrder"}]
            },
            "ProcessPayment": {
              "Type": "Task",
              "Resource": "${ProcessPaymentFunction.Arn}",
              "Retry": [{"ErrorEquals": ["States.TaskFailed"], "IntervalSeconds": 5, "MaxAttempts": 3, "BackoffRate": 2.0}],
              "Next": "NotifySuccess",
              "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "FailOrder"}]
            },
            "NotifySuccess": {
              "Type": "Task",
              "Resource": "arn:aws:states:::sns:publish",
              "Parameters": {
                "TopicArn": "${OrderNotificationTopic}",
                "Message.$": "States.Format('Order {} processed successfully', $.orderId)"
              },
              "End": true
            },
            "FailOrder": {
              "Type": "Task",
              "Resource": "${FailOrderFunction.Arn}",
              "End": true
            }
          }
        }
      RoleArn: !GetAtt StepFunctionsRole.Arn

  # DynamoDB テーブル
  OrdersTable:
    Type: AWS::DynamoDB::Table
    DeletionPolicy: Retain
    Properties:
      TableName: !Sub '${EnvironmentName}-orders'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: orderId
          AttributeType: S
        - AttributeName: customerId
          AttributeType: S
      KeySchema:
        - AttributeName: orderId
          KeyType: HASH
      GlobalSecondaryIndexes:
        - IndexName: customer-index
          KeySchema:
            - AttributeName: customerId
              KeyType: HASH
          Projection:
            ProjectionType: ALL

  # CloudWatch アラーム
  OrderApiErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${EnvironmentName}-order-api-errors'
      MetricName: Errors
      Namespace: AWS/Lambda
      Statistic: Sum
      Period: 60
      EvaluationPeriods: 3
      Threshold: 5
      ComparisonOperator: GreaterThanOrEqualToThreshold
      Dimensions:
        - Name: FunctionName
          Value: !Ref OrderApiFunction
      AlarmActions:
        - !Ref AlertTopic

Outputs:
  ApiEndpoint:
    Description: API Gateway エンドポイント
    Value: !Sub 'https://${ApiGateway}.execute-api.${AWS::Region}.amazonaws.com/Prod'

  StateMachineArn:
    Description: Step Functions ステートマシン ARN
    Value: !Ref OrderProcessingStateMachine
```

---

## 13. FAQ

### Q1. Provisioned Concurrency と Reserved Concurrency の違いは？

Reserved Concurrency は関数の同時実行数の「上限」を設定するもので、追加コストはない。他の関数のスロットルから特定の関数を保護する。一方、Provisioned Concurrency は指定数の実行環境を「事前に初期化」しておくもので、コールドスタートをなくす代わりに追加料金が発生する。

### Q2. Step Functions のコストはどのくらいですか？

Standard ワークフローは状態遷移ごとに $0.025/1,000 遷移で課金される。1回の実行に 5 遷移あり、月間 100 万実行の場合は $125/月となる。Express ワークフローは実行回数とメモリ・実行時間で課金され、大量・短時間処理には割安になる。

### Q3. Lambda の最大同時実行数を超えるとどうなりますか？

スロットリングが発生する。同期呼び出しでは HTTP 429 エラーが返り、非同期呼び出しではイベントキューに格納され最大 6 時間リトライされる。Service Quotas からクォータ引き上げを申請するか、Reserved Concurrency で重要な関数のキャパシティを確保することで対処する。

### Q4. Lambda レイヤーとコンテナイメージのどちらを使うべきですか？

ZIP パッケージ + レイヤーは軽量な関数に適しており、起動速度が最も速い。コンテナイメージは 10GB までのサイズに対応し、既存の Docker ワークフローを活用できる。ML モデルや大規模な依存関係がある場合はコンテナイメージが適している。一般的な Web API やイベント処理には ZIP パッケージが推奨される。

### Q5. Lambda Extension とは何ですか？

Lambda Extension は Lambda の実行環境に統合される外部プロセスで、モニタリング、セキュリティ、ガバナンスの機能を追加できる。内部 Extension (同一プロセス) と外部 Extension (別プロセス) の 2 種類がある。代表的なものに Datadog Agent、New Relic Agent、AWS Parameters and Secrets Lambda Extension がある。

### Q6. Graviton (ARM) と x86 のどちらを選ぶべきですか？

Graviton (arm64) は x86_64 と比較して最大 34% のコスト削減 (20% の料金差 + 性能向上) が見込める。Python、Node.js、Java などのランタイムでは arm64 への移行は通常、コード変更なしで可能。ネイティブバイナリを含むレイヤーや依存関係がある場合は arm64 対応の確認が必要。新規関数では arm64 を優先的に検討すべきである。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| コールドスタート | Init コードの最適化、パッケージ軽量化、メモリ増強で緩和 |
| Provisioned Concurrency | 事前に実行環境を確保しコールドスタートを排除 |
| Lambda Destinations | 非同期実行の成功・失敗を柔軟にルーティング |
| Step Functions | 複数 Lambda のオーケストレーション、エラーハンドリング、リトライ |
| SnapStart | Java のコールドスタートをスナップショット復元で大幅短縮 |
| Lambda レイヤー | 共通ライブラリの一元管理、デプロイパッケージの軽量化 |
| モニタリング | CloudWatch Logs Insights、X-Ray、Lambda Insights で可観測性を確保 |
| セキュリティ | 最小権限 IAM、Secrets Manager 統合、VPC 設計 |
| コンテナイメージ | 大規模依存関係や既存 Docker ワークフローの活用に |
| 関数 URL | シンプルな HTTP エンドポイント (API Gateway なし) |
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
5. AWS 公式ドキュメント「Lambda レイヤー」 https://docs.aws.amazon.com/lambda/latest/dg/chapter-layers.html
6. AWS 公式ドキュメント「Lambda SnapStart」 https://docs.aws.amazon.com/lambda/latest/dg/snapstart.html
7. AWS 公式ドキュメント「Lambda 関数 URL」 https://docs.aws.amazon.com/lambda/latest/dg/lambda-urls.html
