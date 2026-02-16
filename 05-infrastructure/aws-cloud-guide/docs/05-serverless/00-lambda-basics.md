# AWS Lambda 基礎

> サーバーを一切管理せずにコードを実行できる AWS Lambda の基本概念、関数の作成方法、トリガー設定、IAM ロール、環境変数、レイヤーまでを体系的に学ぶ。

---

## この章で学ぶこと

1. **Lambda 関数の作成とデプロイ** -- ランタイム選択からコードのアップロード、テスト実行までの一連の流れを理解する
2. **トリガーと IAM ロールの設計** -- API Gateway・S3・SQS などのイベントソースと、最小権限の実行ロールを正しく構成する
3. **環境変数とレイヤーの活用** -- 設定の外部化と共通ライブラリの再利用でメンテナンス性を高める手法を身につける

---

## 1. Lambda とは何か

### 1.1 サーバーレスコンピューティングの位置づけ

```
従来型 (EC2)              コンテナ (ECS/EKS)          サーバーレス (Lambda)
+-----------------+      +-----------------+      +-----------------+
| アプリケーション |      | アプリケーション |      | アプリケーション |
+-----------------+      +-----------------+      +-----------------+
| ミドルウェア     |      | コンテナランタイム|      |                 |
+-----------------+      +-----------------+      |  AWS が全て管理  |
| OS              |      |  OS (共有)       |      |                 |
+-----------------+      +-----------------+      +-----------------+
| ハードウェア     |      | ハードウェア     |      | ハードウェア     |
+-----------------+      +-----------------+      +-----------------+
  ユーザー管理範囲:広      ユーザー管理範囲:中      ユーザー管理範囲:狭
```

Lambda はイベント駆動型のコンピューティングサービスであり、以下の特徴を持つ。

- **プロビジョニング不要** -- サーバーの起動・停止・スケーリングは AWS が自動管理
- **実行時間課金** -- リクエスト数と実行時間(1ms 単位)で課金
- **自動スケーリング** -- 同時実行数は需要に応じて自動的に増減
- **幅広い言語サポート** -- Python、Node.js、Java、Go、.NET、Ruby、カスタムランタイム

### 1.2 Lambda の実行モデル

```
イベントソース           Lambda サービス              実行環境
+------------+        +------------------+        +------------------+
|            |  呼出  |                  |  配置  |  実行環境 (MicroVM)|
| API Gateway| -----> |  Lambda Control  | -----> |  +-------------+ |
| S3         |        |  Plane           |        |  | ランタイム  | |
| SQS        |        |                  |        |  | + ユーザー  | |
| EventBridge|        +------------------+        |  |   コード    | |
+------------+               |                    |  +-------------+ |
                             |  ログ送信          +------------------+
                             v                           |
                    +------------------+                  | メトリクス
                    |  CloudWatch Logs |                  v
                    +------------------+          +------------------+
                                                  | CloudWatch       |
                                                  | Metrics          |
                                                  +------------------+
```

### 1.3 Lambda の課金モデル

Lambda の料金は「リクエスト数」と「実行時間（GB-秒）」の2軸で決定される。

| 料金要素 | 単価 (東京リージョン) | 無料枠 (月間) |
|---------|---------------------|-------------|
| リクエスト数 | $0.20 / 100 万リクエスト | 100 万リクエスト |
| 実行時間 (GB-秒) | $0.0000166667 / GB-秒 | 400,000 GB-秒 |
| Provisioned Concurrency | $0.0000041667 / GB-秒 | なし |
| Lambda@Edge リクエスト | $0.60 / 100 万リクエスト | なし |

```
コスト計算例:

関数の設定:
  メモリ: 512 MB (= 0.5 GB)
  平均実行時間: 200 ms (= 0.2 秒)
  月間リクエスト: 500 万

計算:
  GB-秒 = 0.5 GB × 0.2 秒 × 5,000,000 = 500,000 GB-秒
  無料枠差し引き = 500,000 - 400,000 = 100,000 GB-秒
  実行時間料金 = 100,000 × $0.0000166667 = $1.67
  リクエスト料金 = (5,000,000 - 1,000,000) × $0.20/1,000,000 = $0.80

  月額合計 = $1.67 + $0.80 = $2.47
```

### 1.4 Lambda のライフサイクル

```
┌──────────────────────────────────────────────────────────────┐
│  Lambda 実行環境のライフサイクル                                │
│                                                              │
│  INIT フェーズ (コールドスタート時のみ)                        │
│  ┌──────────────────────────────────────────────────┐        │
│  │ 1. Extension Init   (Lambda Extensions の初期化)  │        │
│  │ 2. Runtime Init     (ランタイムの初期化)           │        │
│  │ 3. Function Init    (ハンドラ外コードの実行)       │        │
│  │    - グローバル変数の初期化                        │        │
│  │    - SDK クライアントの生成                        │        │
│  │    - DB コネクションの確立                         │        │
│  └──────────────────────────────────────────────────┘        │
│                        │                                     │
│                        ▼                                     │
│  INVOKE フェーズ (毎回実行)                                   │
│  ┌──────────────────────────────────────────────────┐        │
│  │ 4. lambda_handler(event, context) の実行          │        │
│  │    - イベントデータの処理                          │        │
│  │    - ビジネスロジック                              │        │
│  │    - レスポンスの返却                              │        │
│  └──────────────────────────────────────────────────┘        │
│                        │                                     │
│                        ▼                                     │
│  SHUTDOWN フェーズ (環境破棄時)                                │
│  ┌──────────────────────────────────────────────────┐        │
│  │ 5. Runtime Shutdown  (ランタイムの終了処理)        │        │
│  │ 6. Extension Shutdown (Extensions の終了処理)     │        │
│  └──────────────────────────────────────────────────┘        │
│                                                              │
│  ※ 実行環境は一定時間再利用される (Warm Start)                │
│  ※ 再利用時は INVOKE フェーズのみ実行される                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Lambda 関数の作成

### 2.1 対応ランタイム一覧

| ランタイム | 識別子 | サポート状況 | 主な用途 |
|-----------|--------|-------------|---------|
| Python 3.12 | `python3.12` | GA | データ処理、API バックエンド |
| Node.js 20.x | `nodejs20.x` | GA | API バックエンド、リアルタイム処理 |
| Java 21 | `java21` | GA | エンタープライズ、バッチ処理 |
| Go (provided.al2023) | `provided.al2023` | GA | 高性能処理 |
| .NET 8 | `dotnet8` | GA | Windows 連携、エンタープライズ |
| Ruby 3.3 | `ruby3.3` | GA | スクリプト、Webhook |
| カスタムランタイム | `provided.al2023` | GA | Rust、PHP など任意の言語 |

### 2.2 Python で Hello World

```python
# lambda_function.py
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    Lambda のエントリポイント。

    Parameters:
        event (dict): トリガーから渡されるイベントデータ
        context (LambdaContext): 実行コンテキスト情報

    Returns:
        dict: API Gateway 互換のレスポンス
    """
    logger.info(f"Received event: {json.dumps(event)}")

    name = event.get("queryStringParameters", {}).get("name", "World")

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps({
            "message": f"Hello, {name}!",
            "requestId": context.aws_request_id,
            "remainingTime": context.get_remaining_time_in_millis()
        })
    }
```

### 2.3 Node.js での実装例

```javascript
// index.mjs (ES Modules)
export const handler = async (event, context) => {
  console.log("Event:", JSON.stringify(event, null, 2));

  const name = event.queryStringParameters?.name || "World";

  return {
    statusCode: 200,
    headers: {
      "Content-Type": "application/json",
      "Access-Control-Allow-Origin": "*",
    },
    body: JSON.stringify({
      message: `Hello, ${name}!`,
      requestId: context.awsRequestId,
    }),
  };
};
```

### 2.4 AWS CLI による関数作成

```bash
# 1. デプロイパッケージ作成
zip function.zip lambda_function.py

# 2. Lambda 関数の作成
aws lambda create-function \
  --function-name my-hello-function \
  --runtime python3.12 \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --handler lambda_function.lambda_handler \
  --zip-file fileb://function.zip \
  --timeout 30 \
  --memory-size 256 \
  --description "Hello World Lambda function"

# 3. テスト呼び出し
aws lambda invoke \
  --function-name my-hello-function \
  --payload '{"queryStringParameters": {"name": "AWS"}}' \
  --cli-binary-format raw-in-base64-out \
  output.json

cat output.json
```

### 2.5 Lambda のメモリとタイムアウト設定

| 設定項目 | 最小値 | 最大値 | デフォルト | 備考 |
|---------|--------|--------|-----------|------|
| メモリ | 128 MB | 10,240 MB | 128 MB | CPU は比例配分 |
| タイムアウト | 1 秒 | 900 秒 (15分) | 3 秒 | API Gateway 経由は 29 秒制限 |
| エフェメラルストレージ | 512 MB | 10,240 MB | 512 MB | /tmp 領域 |
| デプロイパッケージ | - | 50 MB (zip) / 250 MB (展開後) | - | レイヤー含む |
| コンテナイメージ | - | 10 GB | - | ECR イメージ使用時 |

```
メモリとCPUの関係:

メモリ        vCPU相当      適用シーン
128 MB   -->  ~0.08 vCPU    軽量なAPI応答
512 MB   -->  ~0.33 vCPU    一般的なAPI処理
1,024 MB -->  ~0.58 vCPU    データ変換
1,769 MB -->  1 vCPU        計算処理
3,008 MB -->  2 vCPU        画像処理
10,240 MB --> 6 vCPU        ML推論、大規模バッチ
```

### 2.6 コンテナイメージでのデプロイ

ZIP パッケージの 250 MB 制限を超える場合や、既存の Docker ワークフローがある場合はコンテナイメージを使用する。

```dockerfile
# Dockerfile
FROM public.ecr.aws/lambda/python:3.12

# 依存パッケージのインストール
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# 関数コードのコピー
COPY app.py ${LAMBDA_TASK_ROOT}

# ハンドラの指定
CMD [ "app.lambda_handler" ]
```

```bash
# 1. ECR リポジトリの作成
aws ecr create-repository \
  --repository-name my-lambda-function \
  --image-scanning-configuration scanOnPush=true

# 2. Docker イメージのビルドとプッシュ
aws ecr get-login-password --region ap-northeast-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.ap-northeast-1.amazonaws.com

docker build -t my-lambda-function .
docker tag my-lambda-function:latest \
  123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-lambda-function:latest
docker push \
  123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-lambda-function:latest

# 3. コンテナイメージから Lambda 関数を作成
aws lambda create-function \
  --function-name my-container-function \
  --package-type Image \
  --code ImageUri=123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-lambda-function:latest \
  --role arn:aws:iam::123456789012:role/lambda-execution-role \
  --timeout 60 \
  --memory-size 512
```

---

## 3. トリガーとイベントソース

### 3.1 主要なイベントソース

```
+------------------+     同期呼出      +--------+
| API Gateway      | ----------------> |        |
| ALB              |                   |        |
| Lambda URL       |                   |        |
+------------------+                   |        |
                                       | Lambda |
+------------------+     非同期呼出    |  関数  |
| S3               | ----------------> |        |
| SNS              |                   |        |
| EventBridge      |                   |        |
| IoT              |                   |        |
+------------------+                   |        |
                                       |        |
+------------------+   ポーリングベース |        |
| SQS              | ----------------> |        |
| DynamoDB Streams | (Event Source     |        |
| Kinesis          |  Mapping)         |        |
+------------------+                   +--------+
```

### 3.2 呼び出しモデルの比較

| 呼び出しモデル | イベントソース例 | リトライ動作 | エラーハンドリング |
|---------------|----------------|-------------|-----------------|
| 同期 (RequestResponse) | API Gateway, ALB | 呼び出し元が制御 | 即座にエラー応答 |
| 非同期 (Event) | S3, SNS, EventBridge | 最大2回リトライ | DLQ / Destinations |
| ポーリング (Event Source Mapping) | SQS, Kinesis, DynamoDB | ソースにより異なる | バッチ失敗時の制御 |

### 3.3 API Gateway トリガーの設定

```bash
# REST API の作成と Lambda 統合
aws apigateway create-rest-api \
  --name "HelloAPI" \
  --description "Hello World API"

# Lambda パーミッション追加
aws lambda add-permission \
  --function-name my-hello-function \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:ap-northeast-1:123456789012:abc123/*"
```

### 3.4 S3 トリガーの設定

```bash
# S3 バケットからの Lambda 呼び出しを許可
aws lambda add-permission \
  --function-name image-processor \
  --statement-id s3-invoke \
  --action lambda:InvokeFunction \
  --principal s3.amazonaws.com \
  --source-arn "arn:aws:s3:::my-upload-bucket" \
  --source-account 123456789012

# S3 バケット通知の設定
aws s3api put-bucket-notification-configuration \
  --bucket my-upload-bucket \
  --notification-configuration '{
    "LambdaFunctionConfigurations": [
      {
        "LambdaFunctionArn": "arn:aws:lambda:ap-northeast-1:123456789012:function:image-processor",
        "Events": ["s3:ObjectCreated:*"],
        "Filter": {
          "Key": {
            "FilterRules": [
              {"Name": "prefix", "Value": "uploads/"},
              {"Name": "suffix", "Value": ".jpg"}
            ]
          }
        }
      }
    ]
  }'
```

```python
# S3 トリガーの Lambda 関数
import json
import boto3
import urllib.parse

s3 = boto3.client("s3")

def lambda_handler(event, context):
    """S3 にアップロードされた画像を処理する"""
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        size = record["s3"]["object"]["size"]

        print(f"Processing: s3://{bucket}/{key} ({size} bytes)")

        # オブジェクトのメタデータ取得
        response = s3.head_object(Bucket=bucket, Key=key)
        content_type = response["ContentType"]

        # サムネイル生成 (Pillow が必要 → レイヤーで追加)
        if content_type.startswith("image/"):
            obj = s3.get_object(Bucket=bucket, Key=key)
            # ... 画像処理ロジック
            s3.put_object(
                Bucket=bucket,
                Key=f"thumbnails/{key}",
                Body=thumbnail_bytes,
                ContentType=content_type
            )

    return {"statusCode": 200, "body": "Processed"}
```

### 3.5 SQS トリガーの設定

```bash
# SQS イベントソースマッピングの作成
aws lambda create-event-source-mapping \
  --function-name order-processor \
  --event-source-arn arn:aws:sqs:ap-northeast-1:123456789012:orders-queue \
  --batch-size 10 \
  --maximum-batching-window-in-seconds 5 \
  --function-response-types ReportBatchItemFailures
```

```python
# SQS トリガーの Lambda 関数 (部分バッチ失敗レポート対応)
import json

def lambda_handler(event, context):
    """SQS メッセージをバッチ処理し、失敗したメッセージのみを報告する"""
    batch_item_failures = []

    for record in event["Records"]:
        try:
            body = json.loads(record["body"])
            order_id = body["orderId"]
            print(f"Processing order: {order_id}")

            # 注文処理ロジック
            process_order(body)

        except Exception as e:
            print(f"Error processing {record['messageId']}: {e}")
            # 失敗したメッセージ ID を記録
            batch_item_failures.append({
                "itemIdentifier": record["messageId"]
            })

    # 部分バッチ失敗レポート
    # 成功したメッセージはキューから削除され、失敗したメッセージのみリトライされる
    return {
        "batchItemFailures": batch_item_failures
    }


def process_order(order):
    """注文処理のビジネスロジック"""
    import boto3
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table("Orders")

    table.put_item(Item={
        "PK": f"ORDER#{order['orderId']}",
        "SK": "DETAIL",
        "status": "PROCESSING",
        "items": order["items"],
        "total": order["total"]
    })
```

### 3.6 EventBridge トリガーの設定

```bash
# EventBridge ルールの作成（スケジュール実行）
aws events put-rule \
  --name "daily-cleanup" \
  --schedule-expression "cron(0 3 * * ? *)" \
  --description "毎日 AM 3:00 (UTC) に実行"

# Lambda をターゲットとして追加
aws events put-targets \
  --rule "daily-cleanup" \
  --targets "Id"="1","Arn"="arn:aws:lambda:ap-northeast-1:123456789012:function:cleanup-function"

# Lambda パーミッション追加
aws lambda add-permission \
  --function-name cleanup-function \
  --statement-id eventbridge-invoke \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "arn:aws:events:ap-northeast-1:123456789012:rule/daily-cleanup"
```

```bash
# EventBridge ルール（カスタムイベントパターン）
aws events put-rule \
  --name "order-created" \
  --event-pattern '{
    "source": ["myapp.orders"],
    "detail-type": ["OrderCreated"],
    "detail": {
      "total": [{"numeric": [">=", 10000]}]
    }
  }' \
  --description "10,000円以上の注文が作成されたら通知"
```

### 3.7 DynamoDB Streams トリガーの設定

```bash
# DynamoDB Streams の有効化
aws dynamodb update-table \
  --table-name Users \
  --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES

# イベントソースマッピングの作成
STREAM_ARN=$(aws dynamodb describe-table \
  --table-name Users \
  --query 'Table.LatestStreamArn' \
  --output text)

aws lambda create-event-source-mapping \
  --function-name user-change-handler \
  --event-source-arn $STREAM_ARN \
  --starting-position LATEST \
  --batch-size 100 \
  --maximum-batching-window-in-seconds 10 \
  --bisect-batch-on-function-error \
  --maximum-retry-attempts 3 \
  --destination-config '{
    "OnFailure": {
      "Destination": "arn:aws:sqs:ap-northeast-1:123456789012:dlq-stream-failures"
    }
  }'
```

### 3.8 Lambda Function URL

API Gateway を使わずに、Lambda 関数に直接 HTTPS エンドポイントを付与する機能。

```bash
# Function URL の作成
aws lambda create-function-url-config \
  --function-name my-hello-function \
  --auth-type NONE \
  --cors '{
    "AllowOrigins": ["https://example.com"],
    "AllowMethods": ["GET", "POST"],
    "AllowHeaders": ["Content-Type", "Authorization"],
    "MaxAge": 86400
  }'

# リソースベースポリシーを追加 (AuthType=NONE の場合必須)
aws lambda add-permission \
  --function-name my-hello-function \
  --statement-id FunctionURLAllowPublicAccess \
  --action lambda:InvokeFunctionUrl \
  --principal "*" \
  --function-url-auth-type NONE

# Function URL の確認
aws lambda get-function-url-config \
  --function-name my-hello-function
# → https://abc123def456.lambda-url.ap-northeast-1.on.aws/
```

| 項目 | Lambda Function URL | API Gateway HTTP API |
|------|--------------------|--------------------|
| コスト | Lambda 料金のみ | Lambda + API Gateway 料金 |
| 認証 | IAM_AUTH or NONE | JWT, IAM, Lambda Auth |
| スロットリング | なし（Lambda 同時実行制限のみ） | ルート単位で設定可 |
| カスタムドメイン | CloudFront 経由で可能 | ネイティブサポート |
| WAF | 不可 | REST API のみ |
| 推奨 | 内部 API、Webhook、簡易エンドポイント | 本番 API |

---

## 4. IAM ロールの設計

### 4.1 実行ロールの構成要素

Lambda 関数に必要な IAM ロールは2つの部分から成る。

1. **信頼ポリシー (Trust Policy)** -- Lambda サービスがこのロールを引き受ける許可
2. **アクセス許可ポリシー (Permission Policy)** -- 関数が AWS リソースにアクセスする許可

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

### 4.2 最小権限のポリシー例

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:ap-northeast-1:123456789012:log-group:/aws/lambda/my-hello-function:*"
    },
    {
      "Sid": "DynamoDBAccess",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyTable"
    }
  ]
}
```

### 4.3 AWS CLI による IAM ロール作成

```bash
# 1. 信頼ポリシーファイルの作成
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# 2. IAM ロールの作成
aws iam create-role \
  --role-name order-processor-role \
  --assume-role-policy-document file://trust-policy.json

# 3. AWS 管理ポリシーのアタッチ (基本的な CloudWatch Logs 権限)
aws iam attach-role-policy \
  --role-name order-processor-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# 4. VPC 内で実行する場合は追加ポリシー
aws iam attach-role-policy \
  --role-name order-processor-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole

# 5. カスタムインラインポリシーの追加
aws iam put-role-policy \
  --role-name order-processor-role \
  --policy-name dynamodb-access \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
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
        "Effect": "Allow",
        "Action": ["sqs:SendMessage"],
        "Resource": "arn:aws:sqs:ap-northeast-1:123456789012:notification-queue"
      }
    ]
  }'
```

---

## 5. 環境変数

### 5.1 環境変数の設定と利用

```bash
# 環境変数の設定
aws lambda update-function-configuration \
  --function-name my-hello-function \
  --environment "Variables={
    DB_TABLE=users-table,
    LOG_LEVEL=INFO,
    REGION=ap-northeast-1,
    FEATURE_FLAG_NEW_UI=true
  }"
```

```python
# lambda_function.py -- 環境変数の読み取り
import os

TABLE_NAME = os.environ.get("DB_TABLE", "default-table")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
REGION = os.environ.get("REGION", "ap-northeast-1")
FEATURE_FLAG = os.environ.get("FEATURE_FLAG_NEW_UI", "false") == "true"

def lambda_handler(event, context):
    # TABLE_NAME を使って DynamoDB にアクセス
    import boto3
    dynamodb = boto3.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)
    # ...
```

### 5.2 環境変数の暗号化

```
環境変数の暗号化フロー:

設定時:
  平文 --> AWS KMS で暗号化 --> 暗号化された環境変数を保存

実行時:
  暗号化された環境変数 --> Lambda ランタイムが自動復号 --> 平文で利用可能

カスタムKMS利用時:
  Lambda 実行ロールに kms:Decrypt 権限が必要
```

| 暗号化方式 | 説明 | 追加設定 |
|-----------|------|---------|
| デフォルト暗号化 | AWS 管理キーで自動暗号化 | 不要 |
| カスタム KMS キー | 顧客管理キーで暗号化 | KMS キー ARN を指定 |
| ヘルパーによる暗号化 | 転送中の暗号化を追加 | Lambda コンソールで設定 |

### 5.3 Secrets Manager / Parameter Store との連携

機密情報（API キー、DB パスワード等）は環境変数ではなく Secrets Manager または Parameter Store に保存し、Lambda 実行時に取得するのがベストプラクティス。

```python
# Secrets Manager からシークレットを取得
import json
import boto3
from functools import lru_cache

secrets_client = boto3.client("secretsmanager")

@lru_cache(maxsize=1)
def get_db_credentials():
    """
    シークレットを取得し、キャッシュする。
    lru_cache により、同一実行環境内では1回のみ取得。
    """
    response = secrets_client.get_secret_value(
        SecretId="prod/myapp/db-credentials"
    )
    return json.loads(response["SecretString"])

def lambda_handler(event, context):
    creds = get_db_credentials()
    host = creds["host"]
    username = creds["username"]
    password = creds["password"]
    # DB 接続...
```

```python
# Parameter Store + Lambda Extensions (パフォーマンス最適化)
# AWS Parameters and Secrets Lambda Extension を使用
import urllib.request
import json
import os

AWS_SESSION_TOKEN = os.environ["AWS_SESSION_TOKEN"]
PARAMETERS_EXTENSION_PORT = 2773

def get_parameter(name):
    """
    Lambda Extensions 経由で Parameter Store からパラメータを取得。
    Extensions のキャッシュにより、API 呼び出し回数を削減。
    """
    url = f"http://localhost:{PARAMETERS_EXTENSION_PORT}/systemsmanager/parameters/get?name={name}&withDecryption=true"
    headers = {"X-Aws-Parameters-Secrets-Token": AWS_SESSION_TOKEN}
    req = urllib.request.Request(url, headers=headers)
    response = urllib.request.urlopen(req)
    return json.loads(response.read())["Parameter"]["Value"]

def lambda_handler(event, context):
    api_key = get_parameter("/myapp/prod/api-key")
    # ...
```

---

## 6. Lambda レイヤー

### 6.1 レイヤーの仕組み

```
Lambda 関数のファイルシステム:

/opt/                      <-- レイヤーの展開先
  ├── python/              <-- Python ライブラリ
  │   └── lib/
  │       └── python3.12/
  │           └── site-packages/
  │               ├── requests/
  │               └── boto3/
  ├── nodejs/              <-- Node.js ライブラリ
  │   └── node_modules/
  └── bin/                 <-- カスタムバイナリ

/var/task/                 <-- 関数コード
  └── lambda_function.py

/tmp/                      <-- エフェメラルストレージ (512MB-10GB)
```

### 6.2 レイヤーの作成とアタッチ

```bash
# 1. レイヤー用のディレクトリ構造を作成
mkdir -p layer/python
pip install requests -t layer/python/

# 2. ZIP パッケージ作成
cd layer && zip -r ../my-layer.zip python/

# 3. レイヤーの公開
aws lambda publish-layer-version \
  --layer-name my-common-libs \
  --description "共通ライブラリ (requests等)" \
  --zip-file fileb://my-layer.zip \
  --compatible-runtimes python3.12 python3.11

# 4. 関数にレイヤーをアタッチ
aws lambda update-function-configuration \
  --function-name my-hello-function \
  --layers arn:aws:lambda:ap-northeast-1:123456789012:layer:my-common-libs:1
```

### 6.3 レイヤーの制限事項

| 項目 | 制限 |
|------|-----|
| 関数あたりの最大レイヤー数 | 5 |
| レイヤー含む合計展開サイズ | 250 MB |
| レイヤーバージョン数 | 無制限 |
| レイヤー共有 | 同一リージョン内、クロスアカウント可 |

### 6.4 Powertools for AWS Lambda (Python) レイヤー

AWS が提供する Lambda Powertools はロギング、トレーシング、メトリクスなどの横断的関心事を簡潔に実装できるライブラリ。公開レイヤーとして利用可能。

```bash
# Powertools レイヤーの追加
aws lambda update-function-configuration \
  --function-name my-function \
  --layers arn:aws:lambda:ap-northeast-1:017000801446:layer:AWSLambdaPowertoolsPythonV2:67
```

```python
# Powertools を使ったロギング・トレーシング・メトリクス
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext

logger = Logger(service="order-service")
tracer = Tracer(service="order-service")
metrics = Metrics(namespace="MyApp", service="order-service")

@logger.inject_lambda_context(log_event=True)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext):
    order_id = event.get("orderId")

    # 構造化ログ
    logger.info("Processing order", extra={"order_id": order_id})

    # カスタムメトリクス
    metrics.add_metric(name="OrderProcessed", unit=MetricUnit.Count, value=1)

    # X-Ray サブセグメント
    with tracer.provider.in_subsegment("validate_order") as subsegment:
        subsegment.put_annotation("order_id", order_id)
        result = validate_order(order_id)

    return {"statusCode": 200, "body": "OK"}
```

---

## 7. VPC 設定と RDS Proxy

### 7.1 VPC 内での Lambda 実行

Lambda 関数を VPC 内に配置すると、RDS やElastiCache などのプライベートリソースにアクセスできる。

```
┌────────────────────────────────────────────────────────┐
│  VPC (10.0.0.0/16)                                     │
│                                                        │
│  ┌──────────────────┐    ┌──────────────────┐          │
│  │ Private Subnet A │    │ Private Subnet B │          │
│  │ (10.0.1.0/24)    │    │ (10.0.2.0/24)    │          │
│  │                  │    │                  │          │
│  │  ┌────────────┐  │    │  ┌────────────┐  │          │
│  │  │ Lambda ENI │  │    │  │ Lambda ENI │  │          │
│  │  └──────┬─────┘  │    │  └──────┬─────┘  │          │
│  │         │        │    │         │        │          │
│  │         ▼        │    │         ▼        │          │
│  │  ┌────────────┐  │    │  ┌────────────┐  │          │
│  │  │ RDS Proxy  │  │    │  │ RDS (Read  │  │          │
│  │  │ (Writer)   │  │    │  │  Replica)  │  │          │
│  │  └────────────┘  │    │  └────────────┘  │          │
│  └──────────────────┘    └──────────────────┘          │
│                                                        │
│  ┌──────────────────┐                                  │
│  │ NAT Gateway      │ ← Lambda から外部 API を呼ぶ場合 │
│  │ (Public Subnet)  │   に必要                         │
│  └──────────────────┘                                  │
└────────────────────────────────────────────────────────┘
```

```bash
# Lambda 関数を VPC に配置
aws lambda update-function-configuration \
  --function-name my-vpc-function \
  --vpc-config SubnetIds=subnet-aaa,subnet-bbb,SecurityGroupIds=sg-xxx

# VPC Lambda から外部インターネットにアクセスするには
# NAT Gateway がパブリックサブネットに必要
# (VPC Endpoint を使えば AWS サービスへのアクセスは NAT 不要)
```

### 7.2 RDS Proxy を使った接続管理

```bash
# RDS Proxy の作成
aws rds create-db-proxy \
  --db-proxy-name my-lambda-proxy \
  --engine-family MYSQL \
  --auth '[{
    "AuthScheme": "SECRETS",
    "SecretArn": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:rds-credentials",
    "IAMAuth": "REQUIRED"
  }]' \
  --role-arn arn:aws:iam::123456789012:role/rds-proxy-role \
  --vpc-subnet-ids subnet-aaa subnet-bbb \
  --vpc-security-group-ids sg-xxx

# ターゲットグループの登録
aws rds register-db-proxy-targets \
  --db-proxy-name my-lambda-proxy \
  --db-instance-identifiers my-rds-instance
```

```python
# RDS Proxy 経由の接続 (IAM 認証)
import boto3
import pymysql
import os

rds_client = boto3.client("rds")

# ハンドラ外 (グローバルスコープ) でコネクションを初期化
# → 実行環境の再利用時にコネクションを使い回す
connection = None

def get_connection():
    global connection
    if connection is None or not connection.open:
        token = rds_client.generate_db_auth_token(
            DBHostname=os.environ["PROXY_ENDPOINT"],
            Port=3306,
            DBUsername=os.environ["DB_USER"],
            Region="ap-northeast-1"
        )
        connection = pymysql.connect(
            host=os.environ["PROXY_ENDPOINT"],
            user=os.environ["DB_USER"],
            password=token,
            database=os.environ["DB_NAME"],
            connect_timeout=5,
            ssl={"ssl": True}
        )
    return connection

def lambda_handler(event, context):
    conn = get_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM users WHERE id = %s", (event["userId"],))
        result = cursor.fetchone()
    return {"statusCode": 200, "body": str(result)}
```

---

## 8. SAM / CloudFormation によるデプロイ

### 8.1 SAM テンプレート

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Lambda function with SAM

Globals:
  Function:
    Runtime: python3.12
    Timeout: 30
    MemorySize: 256
    Tracing: Active
    Environment:
      Variables:
        LOG_LEVEL: INFO
        TABLE_NAME: !Ref OrdersTable

Parameters:
  Stage:
    Type: String
    Default: prod
    AllowedValues: [dev, staging, prod]

Resources:
  # API
  HttpApi:
    Type: AWS::Serverless::HttpApi
    Properties:
      StageName: !Ref Stage
      CorsConfiguration:
        AllowOrigins:
          - "https://example.com"
        AllowMethods:
          - GET
          - POST
        AllowHeaders:
          - Authorization
          - Content-Type

  # Lambda 関数 (GET /orders)
  ListOrdersFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/list_orders.lambda_handler
      CodeUri: src/
      Description: "注文一覧を取得"
      Events:
        GetOrders:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
            Path: /orders
            Method: GET
      Policies:
        - DynamoDBReadPolicy:
            TableName: !Ref OrdersTable

  # Lambda 関数 (POST /orders)
  CreateOrderFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/create_order.lambda_handler
      CodeUri: src/
      Description: "注文を作成"
      Events:
        PostOrder:
          Type: HttpApi
          Properties:
            ApiId: !Ref HttpApi
            Path: /orders
            Method: POST
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref OrdersTable
        - SQSSendMessagePolicy:
            QueueName: !GetAtt NotificationQueue.QueueName

  # SQS キュー
  NotificationQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "${Stage}-notification-queue"
      VisibilityTimeout: 60
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt DLQ.Arn
        maxReceiveCount: 3

  DLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: !Sub "${Stage}-notification-dlq"
      MessageRetentionPeriod: 1209600  # 14日

  # SQS トリガーの Lambda
  NotificationFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/send_notification.lambda_handler
      CodeUri: src/
      Events:
        SQSEvent:
          Type: SQS
          Properties:
            Queue: !GetAtt NotificationQueue.Arn
            BatchSize: 10
            FunctionResponseTypes:
              - ReportBatchItemFailures
      Policies:
        - SESCrudPolicy:
            IdentityName: "example.com"

  # DynamoDB テーブル
  OrdersTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: !Sub "${Stage}-Orders"
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: PK
          AttributeType: S
        - AttributeName: SK
          AttributeType: S
      KeySchema:
        - AttributeName: PK
          KeyType: HASH
        - AttributeName: SK
          KeyType: RANGE

Outputs:
  ApiUrl:
    Value: !Sub "https://${HttpApi}.execute-api.${AWS::Region}.amazonaws.com/${Stage}"
  ListOrdersFunctionArn:
    Value: !GetAtt ListOrdersFunction.Arn
```

```bash
# SAM CLI によるデプロイ手順
# 1. ビルド
sam build

# 2. ローカルテスト
sam local invoke ListOrdersFunction \
  --event events/get-orders.json

# 3. ローカル API 起動
sam local start-api --port 3000

# 4. デプロイ
sam deploy \
  --stack-name my-order-api \
  --parameter-overrides Stage=prod \
  --capabilities CAPABILITY_IAM \
  --resolve-s3

# 5. ログの確認
sam logs --name ListOrdersFunction --stack-name my-order-api --tail
```

### 8.2 Terraform による Lambda デプロイ

```hcl
# main.tf

# Lambda 関数
resource "aws_lambda_function" "order_processor" {
  function_name = "${var.stage}-order-processor"
  role          = aws_iam_role.lambda_role.arn
  handler       = "handlers.order_processor.lambda_handler"
  runtime       = "python3.12"
  timeout       = 30
  memory_size   = 256

  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      TABLE_NAME = aws_dynamodb_table.orders.name
      LOG_LEVEL  = "INFO"
      STAGE      = var.stage
    }
  }

  tracing_config {
    mode = "Active"
  }

  # VPC 設定 (RDS にアクセスする場合)
  vpc_config {
    subnet_ids         = var.private_subnet_ids
    security_group_ids = [aws_security_group.lambda.id]
  }

  depends_on = [
    aws_iam_role_policy_attachment.lambda_logs,
    aws_iam_role_policy_attachment.lambda_vpc,
    aws_cloudwatch_log_group.lambda,
  ]

  tags = {
    Environment = var.stage
    Service     = "order-service"
  }
}

# デプロイパッケージの ZIP 化
data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/src"
  output_path = "${path.module}/dist/lambda.zip"
}

# CloudWatch Logs グループ (保持期間を指定)
resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.stage}-order-processor"
  retention_in_days = 30
}

# IAM ロール
resource "aws_iam_role" "lambda_role" {
  name = "${var.stage}-order-processor-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_vpc" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"
}

resource "aws_iam_role_policy" "dynamodb_access" {
  name = "dynamodb-access"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:Query"
      ]
      Resource = [
        aws_dynamodb_table.orders.arn,
        "${aws_dynamodb_table.orders.arn}/index/*"
      ]
    }]
  })
}

# SQS イベントソースマッピング
resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn                   = aws_sqs_queue.orders.arn
  function_name                      = aws_lambda_function.order_processor.arn
  batch_size                         = 10
  maximum_batching_window_in_seconds = 5

  function_response_types = ["ReportBatchItemFailures"]
}

# Lambda エイリアス (Blue/Green デプロイ用)
resource "aws_lambda_alias" "live" {
  name             = "live"
  function_name    = aws_lambda_function.order_processor.function_name
  function_version = aws_lambda_function.order_processor.version

  routing_config {
    additional_version_weights = {
      # カナリアデプロイ: 新バージョンに 10% のトラフィック
      (aws_lambda_function.order_processor.version) = 0.1
    }
  }
}
```

---

## 9. 同時実行数とスケーリング

### 9.1 同時実行数の概念

```
同時実行数の計算:
  同時実行数 = 秒間リクエスト数 × 平均実行時間(秒)

例:
  100 req/s × 0.2 秒 = 20 同時実行
  1,000 req/s × 0.5 秒 = 500 同時実行
```

### 9.2 予約済み同時実行とプロビジョニング済み同時実行

```
┌──────────────────────────────────────────────────────────┐
│  リージョン上限: 1,000 同時実行 (デフォルト)              │
│                                                          │
│  ┌─────────────────────┐ Reserved Concurrency: 200       │
│  │ 関数 A (API)        │ → 最大 200 同時実行を保証       │
│  │ Provisioned: 50     │ → うち 50 は常時ウォーム        │
│  └─────────────────────┘                                 │
│                                                          │
│  ┌─────────────────────┐ Reserved Concurrency: 100       │
│  │ 関数 B (バッチ)     │ → 最大 100 同時実行を保証       │
│  └─────────────────────┘                                 │
│                                                          │
│  ┌─────────────────────┐ Reserved Concurrency: なし      │
│  │ 関数 C (その他)     │ → 残りの 700 を他の関数と共有   │
│  └─────────────────────┘                                 │
│                                                          │
│  Unreserved = 1,000 - 200 - 100 = 700                   │
│  ※ 100 は AWS が予約 (Unreserved 最低保証)               │
└──────────────────────────────────────────────────────────┘
```

```bash
# Reserved Concurrency の設定
aws lambda put-function-concurrency \
  --function-name my-api-function \
  --reserved-concurrent-executions 200

# Provisioned Concurrency の設定 (エイリアスまたはバージョン指定)
aws lambda put-provisioned-concurrency-config \
  --function-name my-api-function \
  --qualifier prod \
  --provisioned-concurrent-executions 50

# Provisioned Concurrency の状態確認
aws lambda get-provisioned-concurrency-config \
  --function-name my-api-function \
  --qualifier prod

# Application Auto Scaling でProvisioned Concurrencyを自動調整
aws application-autoscaling register-scalable-target \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --min-capacity 10 \
  --max-capacity 100

aws application-autoscaling put-scaling-policy \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --policy-name "target-tracking" \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 0.7,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "LambdaProvisionedConcurrencyUtilization"
    }
  }'
```

---

## 10. 監視とロギング

### 10.1 CloudWatch メトリクス

| メトリクス | 説明 | 単位 |
|-----------|------|------|
| Invocations | 関数呼び出し回数 | Count |
| Duration | 実行時間 | Milliseconds |
| Errors | エラー発生回数 (ハンドラ例外) | Count |
| Throttles | スロットルされた呼び出し回数 | Count |
| ConcurrentExecutions | 同時実行数 | Count |
| IteratorAge | ストリーム系ソースの遅延 | Milliseconds |
| DeadLetterErrors | DLQ 送信失敗回数 | Count |

### 10.2 CloudWatch Alarm の設定

```bash
# エラー率アラーム (エラー率 > 5%)
aws cloudwatch put-metric-alarm \
  --alarm-name "lambda-error-rate-high" \
  --alarm-description "Lambda error rate exceeds 5%" \
  --metric-name Errors \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --dimensions Name=FunctionName,Value=my-api-function \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# スロットルアラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "lambda-throttle-alarm" \
  --metric-name Throttles \
  --namespace AWS/Lambda \
  --statistic Sum \
  --period 60 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --dimensions Name=FunctionName,Value=my-api-function \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# Duration P99 アラーム (P99 レイテンシ > 5秒)
aws cloudwatch put-metric-alarm \
  --alarm-name "lambda-duration-p99-high" \
  --metric-name Duration \
  --namespace AWS/Lambda \
  --extended-statistic p99 \
  --period 300 \
  --threshold 5000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --dimensions Name=FunctionName,Value=my-api-function \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts
```

### 10.3 構造化ロギング

```python
# 構造化ログ (JSON) の実装
import json
import logging
import os
import time

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "function_name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME"),
            "function_version": os.environ.get("AWS_LAMBDA_FUNCTION_VERSION"),
            "request_id": getattr(record, "request_id", None),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)

# ロガーの設定
logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]

def lambda_handler(event, context):
    # request_id をログに自動付与
    extra = {"request_id": context.aws_request_id}

    logger.info("Processing request", extra=extra)
    start = time.time()

    try:
        result = process(event)
        duration = (time.time() - start) * 1000
        logger.info(
            f"Request completed in {duration:.1f}ms",
            extra={**extra, "duration_ms": duration}
        )
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}", extra=extra, exc_info=True)
        raise
```

### 10.4 X-Ray トレーシング

```bash
# X-Ray トレーシングの有効化
aws lambda update-function-configuration \
  --function-name my-api-function \
  --tracing-config Mode=Active
```

```python
# X-Ray SDK による手動トレーシング
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all
import boto3

# AWS SDK の自動計装
patch_all()

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table("Orders")

def lambda_handler(event, context):
    # カスタムサブセグメント
    with xray_recorder.in_subsegment("validate_input") as subsegment:
        subsegment.put_annotation("order_id", event.get("orderId"))
        subsegment.put_metadata("event", event)
        validated = validate(event)

    with xray_recorder.in_subsegment("save_order"):
        table.put_item(Item=validated)

    return {"statusCode": 200}
```

---

## 11. アンチパターン

### 11.1 モノリシック Lambda

```
[悪い例] 1つのLambda関数に全機能を詰め込む

def lambda_handler(event, context):
    path = event["path"]
    if path == "/users":
        return handle_users(event)
    elif path == "/orders":
        return handle_orders(event)
    elif path == "/products":
        return handle_products(event)
    elif path == "/payments":
        return handle_payments(event)
    # ... 数十のルート
```

**問題点**: デプロイパッケージが巨大化し、コールドスタートが遅くなる。1つの変更で全機能に影響し、テストが困難になる。

**改善**: 機能ごとに個別の Lambda 関数を作成し、API Gateway のルーティングで振り分ける。

### 11.2 Lambda 内での同期的な待機

```python
# [悪い例] Lambda 内で長時間の同期待機
import time

def lambda_handler(event, context):
    # 外部APIを呼び、結果が出るまでポーリング
    job_id = start_external_job()
    while True:
        status = check_job_status(job_id)
        if status == "COMPLETE":
            break
        time.sleep(10)  # 10秒ごとにポーリング -- 実行時間を浪費
    return get_job_result(job_id)
```

**問題点**: 実行時間が長くなりコストが増大。タイムアウトのリスクも高まる。

**改善**: Step Functions でステートマシンを構成するか、コールバックパターンを利用する。

### 11.3 グローバルスコープでの SDK クライアント未初期化

```python
# [悪い例] ハンドラ内で毎回 SDK クライアントを生成
def lambda_handler(event, context):
    import boto3
    dynamodb = boto3.resource("dynamodb")  # 毎回初期化 → 遅い
    table = dynamodb.Table("MyTable")
    return table.get_item(Key={"PK": event["id"]})
```

```python
# [良い例] グローバルスコープで SDK クライアントを初期化
import boto3

dynamodb = boto3.resource("dynamodb")  # 実行環境再利用時はスキップされる
table = dynamodb.Table("MyTable")

def lambda_handler(event, context):
    return table.get_item(Key={"PK": event["id"]})
```

### 11.4 /tmp ストレージの未クリーンアップ

```python
# [悪い例] /tmp にファイルを溜め続ける
def lambda_handler(event, context):
    file_path = f"/tmp/{event['fileId']}.json"
    with open(file_path, "w") as f:
        json.dump(event["data"], f)
    # クリーンアップしない → 実行環境再利用時にディスクが圧迫される
```

```python
# [良い例] 処理後に /tmp をクリーンアップ
import os
import tempfile

def lambda_handler(event, context):
    # tempfile を使って自動クリーンアップ
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=".json", delete=True) as f:
        f.write(json.dumps(event["data"]).encode())
        f.flush()
        # ... f.name を使って処理
    # with ブロックを抜けると自動削除
```

### 11.5 Lambda から Lambda の直接呼び出し

```
[悪い例] Lambda が別の Lambda を同期呼び出し
  Lambda A → Lambda B → Lambda C

  問題点:
  - Lambda A は B と C の実行時間分も課金される
  - 3つの関数すべてが同時実行枠を消費
  - エラー時のリトライが複雑になる

[良い例] 非同期連携を利用
  方法 1: SQS / SNS を介した疎結合
    Lambda A → SQS → Lambda B → SNS → Lambda C

  方法 2: Step Functions でオーケストレーション
    Step Functions → Lambda A → Lambda B → Lambda C
    (各ステップの成功/失敗を管理、リトライ/分岐も容易)

  方法 3: EventBridge によるイベント駆動
    Lambda A → EventBridge → Lambda B, Lambda C (並列)
```

---

## 12. FAQ

### Q1. Lambda のコールドスタートとは何ですか？

Lambda 関数が初めて呼び出されるとき、または実行環境がリサイクルされた後に、新しい実行環境の初期化が必要になる。この初期化時間を「コールドスタート」と呼ぶ。Python/Node.js で数百ミリ秒、Java/.NET で数秒かかることがある。対策としては、Provisioned Concurrency の利用や、デプロイパッケージの軽量化が有効である。

```
コールドスタート時間の目安:

ランタイム       VPC なし        VPC あり
Python 3.12     200-500 ms     200-500 ms (Hyperplane ENI)
Node.js 20.x   200-400 ms     200-400 ms
Java 21         2-8 秒         2-8 秒
Java 21+Snap   200-500 ms     N/A (VPC 非対応)
.NET 8          1-3 秒         1-3 秒
Go              < 100 ms       < 100 ms

※ VPC Lambda の ENI 作成は 2019 年以降 Hyperplane により高速化済み
```

### Q2. Lambda 関数の同時実行数に制限はありますか？

デフォルトではリージョンあたり 1,000 同時実行がソフトリミットとして設定されている。Service Quotas から引き上げをリクエストできる。また、関数単位で `ReservedConcurrentExecutions` を設定して、特定の関数が他の関数のキャパシティを奪わないよう制御できる。

```bash
# 現在の同時実行制限を確認
aws lambda get-account-settings \
  --query '{ConcurrentExecutions: AccountLimit.ConcurrentExecutions, UnreservedConcurrentExecutions: AccountLimit.UnreservedConcurrentExecutions}'

# Service Quotas から引き上げリクエスト
aws service-quotas request-service-quota-increase \
  --service-code lambda \
  --quota-code L-B99A9384 \
  --desired-value 5000
```

### Q3. Lambda でデータベース接続をどう管理すべきですか？

RDS を利用する場合は、RDS Proxy を経由して接続プーリングを行うのが推奨される。Lambda 関数のハンドラ外(グローバルスコープ)でコネクションを初期化し、実行環境の再利用時にコネクションを使い回すパターンが基本となる。DynamoDB のような HTTP ベースのサービスであればコネクション管理の問題は発生しない。

### Q4. Lambda 関数のデバッグ方法は？

ローカルデバッグには以下の方法がある。

```bash
# 1. SAM CLI でローカル実行
sam local invoke MyFunction --event event.json --debug-port 5678

# 2. Docker コンテナでローカル実行
docker run --rm -v $(pwd)/src:/var/task \
  -e AWS_REGION=ap-northeast-1 \
  public.ecr.aws/lambda/python:3.12 \
  lambda_function.lambda_handler

# 3. pytest でユニットテスト
# tests/test_handler.py
def test_lambda_handler():
    event = {"queryStringParameters": {"name": "Test"}}
    context = MockContext()  # aws_request_id 等を持つモックオブジェクト
    response = lambda_handler(event, context)
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["message"] == "Hello, Test!"
```

### Q5. Lambda のコストを最適化するには？

(1) **AWS Lambda Power Tuning** ツールを使い、メモリとコストの最適なバランスを見つける。メモリを増やすと CPU も増えるため、実行時間が短縮されトータルコストが下がることがある。(2) **Graviton2 (arm64)** アーキテクチャを選択すると、x86 と比較して最大 34% 安価で最大 20% 高速。(3) 不要な Provisioned Concurrency を削減する。(4) ログレベルを本番では WARN 以上に設定し、CloudWatch Logs のコストを削減する。

```bash
# arm64 (Graviton2) で関数を作成
aws lambda create-function \
  --function-name my-arm-function \
  --runtime python3.12 \
  --architectures arm64 \
  --handler app.lambda_handler \
  --role arn:aws:iam::123456789012:role/lambda-role \
  --zip-file fileb://function.zip
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Lambda とは | サーバー管理不要のイベント駆動型コンピューティング |
| ランタイム | Python, Node.js, Java, Go, .NET, Ruby, カスタム |
| トリガー | 同期(API Gateway)、非同期(S3, SNS)、ポーリング(SQS, Kinesis) |
| IAM ロール | 信頼ポリシー + 最小権限のアクセス許可ポリシー |
| 環境変数 | 設定の外部化、KMS による暗号化サポート |
| レイヤー | 共通ライブラリの再利用、最大5レイヤーまでアタッチ可能 |
| VPC | RDS 等のプライベートリソースアクセスに必要、RDS Proxy 推奨 |
| 同時実行 | Reserved / Provisioned で制御、Auto Scaling で自動調整 |
| 監視 | CloudWatch Logs + Metrics + Alarms + X-Ray |
| 課金 | リクエスト数 + 実行時間(GB-秒)、arm64 で最大 34% 削減 |

---

## 次に読むべきガイド

- [Lambda 応用](./01-lambda-advanced.md) -- コールドスタート最適化、Provisioned Concurrency、Step Functions
- [サーバーレスパターン](./02-serverless-patterns.md) -- API+Lambda+DynamoDB、イベント駆動アーキテクチャ
- [IAM 詳解](../08-security/00-iam-deep-dive.md) -- Lambda 実行ロールの高度な設計

---

## 参考文献

1. AWS 公式ドキュメント「AWS Lambda デベロッパーガイド」 https://docs.aws.amazon.com/lambda/latest/dg/
2. AWS Well-Architected Framework「サーバーレスアプリケーションレンズ」 https://docs.aws.amazon.com/wellarchitected/latest/serverless-applications-lens/
3. Jeremy Daly「Serverless Architectures on AWS, 2nd Edition」Manning Publications, 2024
4. AWS ブログ「Operating Lambda: Performance optimization」 https://aws.amazon.com/blogs/compute/operating-lambda-performance-optimization-part-1/
5. AWS Lambda Powertools for Python https://docs.powertools.aws.dev/lambda/python/latest/
6. AWS SAM CLI ドキュメント https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli.html
