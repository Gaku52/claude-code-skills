# サーバーレスパターン

> API+Lambda+DynamoDB、イベント駆動、ファンアウト、CQRS などの代表的なサーバーレスアーキテクチャパターンを理解し、実践的な設計判断ができるようになる。

---

## この章で学ぶこと

1. **API バックエンドパターン** -- API Gateway + Lambda + DynamoDB の組み合わせで RESTful/GraphQL API を構築する手法
2. **イベント駆動パターン** -- SNS/SQS/EventBridge を活用した疎結合アーキテクチャの設計方法
3. **高度なパターン** -- ファンアウト、CQRS、Saga パターンなどの複雑なユースケースへの対応
4. **ストリーム処理パターン** -- Kinesis Data Streams + Lambda によるリアルタイムデータ処理
5. **スケジュール駆動パターン** -- EventBridge Scheduler + Lambda による定期実行ジョブの設計
6. **Web アプリケーションパターン** -- CloudFront + S3 + API Gateway + Lambda によるフルスタックサーバーレス

---

## 1. API バックエンドパターン

### 1.1 基本構成

```
クライアント
    |
    v
+-------------------+
| Amazon CloudFront |  (CDN, キャッシュ)
+-------------------+
    |
    v
+-------------------+
| API Gateway       |  (認証, スロットリング, リクエスト検証)
| (REST / HTTP API) |
+-------------------+
    |
    v
+-------------------+
| Lambda 関数群     |
| +---------+       |
| | GET     |       |
| | POST    |       |
| | PUT     |       |
| | DELETE  |       |
| +---------+       |
+-------------------+
    |
    v
+-------------------+
| DynamoDB          |  (NoSQL データストア)
+-------------------+
```

### 1.2 REST API + Lambda + DynamoDB の実装

```python
# handler.py -- CRUD API のエントリポイント
import json
import boto3
import os
from decimal import Decimal

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["TABLE_NAME"])

class DecimalEncoder(json.JSONEncoder):
    """DynamoDB の Decimal 型を JSON シリアライズするヘルパー"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)

def lambda_handler(event, context):
    http_method = event["httpMethod"]
    path = event.get("pathParameters") or {}

    try:
        if http_method == "GET" and "id" in path:
            return get_item(path["id"])
        elif http_method == "GET":
            return list_items(event)
        elif http_method == "POST":
            return create_item(json.loads(event["body"]))
        elif http_method == "PUT" and "id" in path:
            return update_item(path["id"], json.loads(event["body"]))
        elif http_method == "DELETE" and "id" in path:
            return delete_item(path["id"])
        else:
            return response(404, {"error": "Not found"})
    except Exception as e:
        return response(500, {"error": str(e)})

def get_item(item_id):
    result = table.get_item(Key={"id": item_id})
    if "Item" in result:
        return response(200, result["Item"])
    return response(404, {"error": "Item not found"})

def list_items(event):
    params = event.get("queryStringParameters") or {}
    limit = int(params.get("limit", "20"))
    result = table.scan(Limit=limit)
    return response(200, {"items": result["Items"], "count": result["Count"]})

def create_item(body):
    import uuid
    body["id"] = str(uuid.uuid4())
    table.put_item(Item=body)
    return response(201, body)

def update_item(item_id, body):
    body["id"] = item_id
    table.put_item(Item=body)
    return response(200, body)

def delete_item(item_id):
    table.delete_item(Key={"id": item_id})
    return response(204, None)

def response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*"
        },
        "body": json.dumps(body, cls=DecimalEncoder) if body else ""
    }
```

### 1.3 API Gateway の種類比較

| 特性 | REST API | HTTP API | WebSocket API |
|------|----------|----------|--------------|
| 料金 (100万リクエスト) | $3.50 | $1.00 | $1.00 + 接続料金 |
| レイテンシ | 高め | 低い | 低い |
| キャッシュ | あり | なし | N/A |
| リクエスト検証 | あり | なし | 部分的 |
| WAF 統合 | あり | なし | なし |
| リソースポリシー | あり | なし | なし |
| カスタムドメイン | あり | あり | あり |
| JWT オーソライザー | Lambda 経由 | ネイティブ対応 | Lambda 経由 |
| 使用量プラン | あり | なし | なし |
| API キー | あり | なし | なし |
| プライベート API | あり | なし | なし |
| 用途 | フル機能 API | 軽量 API, プロキシ | リアルタイム通信 |

### 1.4 REST API vs HTTP API の選択基準

```
REST API を選ぶケース:
  - API キーと使用量プランでのレート制限が必要
  - リクエスト/レスポンスの変換が必要
  - WAF との統合が必要
  - API キャッシュでレイテンシ削減が必要
  - VPC 内からのプライベート API が必要
  - Canary リリースデプロイメントが必要

HTTP API を選ぶケース:
  - コスト最適化が最優先 (REST API の約30%の料金)
  - JWT ネイティブ認証を使いたい
  - シンプルなプロキシ統合のみ
  - OIDC/OAuth 2.0 認可が必要
  - 低レイテンシが求められる
```

### 1.5 SAM テンプレートでの定義

```yaml
# template.yaml (AWS SAM)
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: Serverless CRUD API

Globals:
  Function:
    Runtime: python3.12
    Timeout: 30
    MemorySize: 256
    Environment:
      Variables:
        TABLE_NAME: !Ref ItemsTable
    Tracing: Active
    Layers:
      - !Ref SharedLayer

Resources:
  # Lambda レイヤー（共通ライブラリ）
  SharedLayer:
    Type: AWS::Serverless::LayerVersion
    Properties:
      LayerName: shared-utils
      ContentUri: layers/shared/
      CompatibleRuntimes:
        - python3.12

  # API 関数
  ApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handler.lambda_handler
      CodeUri: src/
      Events:
        GetItem:
          Type: Api
          Properties:
            Path: /items/{id}
            Method: get
        ListItems:
          Type: Api
          Properties:
            Path: /items
            Method: get
        CreateItem:
          Type: Api
          Properties:
            Path: /items
            Method: post
        UpdateItem:
          Type: Api
          Properties:
            Path: /items/{id}
            Method: put
        DeleteItem:
          Type: Api
          Properties:
            Path: /items/{id}
            Method: delete
      Policies:
        - DynamoDBCrudPolicy:
            TableName: !Ref ItemsTable

  ItemsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: items
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: id
          AttributeType: S
      KeySchema:
        - AttributeName: id
          KeyType: HASH
      PointInTimeRecoverySpecification:
        PointInTimeRecoveryEnabled: true
      SSESpecification:
        SSEEnabled: true
```

### 1.6 Powertools for AWS Lambda による構造化ログとメトリクス

```python
# handler_with_powertools.py -- Lambda Powertools を活用した実装
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.event_handler import APIGatewayRestResolver
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext
import boto3
import os

logger = Logger(service="items-api")
tracer = Tracer(service="items-api")
metrics = Metrics(namespace="ItemsAPI", service="items-api")
app = APIGatewayRestResolver()

dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["TABLE_NAME"])

@app.get("/items/<item_id>")
@tracer.capture_method
def get_item(item_id: str):
    logger.info("Getting item", extra={"item_id": item_id})
    result = table.get_item(Key={"id": item_id})
    if "Item" not in result:
        raise app.not_found()
    metrics.add_metric(name="ItemRetrieved", unit=MetricUnit.Count, value=1)
    return result["Item"]

@app.get("/items")
@tracer.capture_method
def list_items():
    params = app.current_event.query_string_parameters or {}
    limit = int(params.get("limit", "20"))
    logger.info("Listing items", extra={"limit": limit})
    result = table.scan(Limit=limit)
    metrics.add_metric(name="ItemsListed", unit=MetricUnit.Count, value=result["Count"])
    return {"items": result["Items"], "count": result["Count"]}

@app.post("/items")
@tracer.capture_method
def create_item():
    import uuid
    body = app.current_event.json_body
    body["id"] = str(uuid.uuid4())
    table.put_item(Item=body)
    logger.info("Item created", extra={"item_id": body["id"]})
    metrics.add_metric(name="ItemCreated", unit=MetricUnit.Count, value=1)
    return body, 201

@app.put("/items/<item_id>")
@tracer.capture_method
def update_item(item_id: str):
    body = app.current_event.json_body
    body["id"] = item_id
    table.put_item(Item=body)
    logger.info("Item updated", extra={"item_id": item_id})
    metrics.add_metric(name="ItemUpdated", unit=MetricUnit.Count, value=1)
    return body

@app.delete("/items/<item_id>")
@tracer.capture_method
def delete_item(item_id: str):
    table.delete_item(Key={"id": item_id})
    logger.info("Item deleted", extra={"item_id": item_id})
    metrics.add_metric(name="ItemDeleted", unit=MetricUnit.Count, value=1)
    return "", 204

@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    return app.resolve(event, context)
```

### 1.7 GraphQL API (AppSync + Lambda)

```yaml
# AppSync GraphQL API の SAM テンプレート
Resources:
  GraphQLApi:
    Type: AWS::AppSync::GraphQLApi
    Properties:
      Name: items-graphql-api
      AuthenticationType: AMAZON_COGNITO_USER_POOLS
      UserPoolConfig:
        UserPoolId: !Ref UserPool
        DefaultAction: ALLOW
        AwsRegion: !Ref AWS::Region
      LogConfig:
        CloudWatchLogsRoleArn: !GetAtt AppSyncLogsRole.Arn
        FieldLogLevel: ERROR
      XrayEnabled: true

  GraphQLSchema:
    Type: AWS::AppSync::GraphQLSchema
    Properties:
      ApiId: !GetAtt GraphQLApi.ApiId
      Definition: |
        type Item {
          id: ID!
          name: String!
          description: String
          price: Float
          category: String
          createdAt: AWSDateTime
          updatedAt: AWSDateTime
        }

        input CreateItemInput {
          name: String!
          description: String
          price: Float
          category: String
        }

        input UpdateItemInput {
          name: String
          description: String
          price: Float
          category: String
        }

        type ItemConnection {
          items: [Item]
          nextToken: String
        }

        type Query {
          getItem(id: ID!): Item
          listItems(limit: Int, nextToken: String): ItemConnection
          searchItems(keyword: String!, limit: Int): ItemConnection
        }

        type Mutation {
          createItem(input: CreateItemInput!): Item
          updateItem(id: ID!, input: UpdateItemInput!): Item
          deleteItem(id: ID!): Item
        }

        type Subscription {
          onCreateItem: Item
            @aws_subscribe(mutations: ["createItem"])
          onUpdateItem(id: ID): Item
            @aws_subscribe(mutations: ["updateItem"])
        }

        schema {
          query: Query
          mutation: Mutation
          subscription: Subscription
        }
```

---

## 2. イベント駆動パターン

### 2.1 イベント駆動の全体像

```
イベント駆動アーキテクチャ:

プロデューサー        イベントルーター        コンシューマー
+-----------+        +---------------+      +------------+
| 注文API    | -----> |               | ---> | 在庫更新    |
+-----------+        |               |      +------------+
                     |  EventBridge  |
+-----------+        |  / SNS       | ---> +------------+
| 決済完了   | -----> |               |      | メール送信  |
+-----------+        |               |      +------------+
                     |               |
+-----------+        |               | ---> +------------+
| 在庫変動   | -----> |               |      | 分析記録    |
+-----------+        +---------------+      +------------+

特徴:
  - 疎結合: プロデューサーはコンシューマーを知らない
  - 拡張容易: 新しいコンシューマーを追加するだけ
  - 非同期: 即座にレスポンスを返せる
```

### 2.2 S3 イベント駆動の画像処理

```python
# image_processor.py -- S3 トリガーによる画像リサイズ
import boto3
import os
from PIL import Image
import io

s3 = boto3.client("s3")
DEST_BUCKET = os.environ["DEST_BUCKET"]
SIZES = [(128, 128), (256, 256), (512, 512)]

def lambda_handler(event, context):
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = record["s3"]["object"]["key"]

        # 元画像のダウンロード
        response = s3.get_object(Bucket=bucket, Key=key)
        image = Image.open(io.BytesIO(response["Body"].read()))

        # 各サイズにリサイズしてアップロード
        for width, height in SIZES:
            resized = image.copy()
            resized.thumbnail((width, height))

            buffer = io.BytesIO()
            resized.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)

            dest_key = f"thumbnails/{width}x{height}/{key}"
            s3.put_object(
                Bucket=DEST_BUCKET,
                Key=dest_key,
                Body=buffer,
                ContentType="image/jpeg"
            )

    return {"processed": len(event["Records"])}
```

### 2.3 EventBridge によるイベントルーティング

```json
{
  "Source": "com.myapp.orders",
  "DetailType": "OrderCreated",
  "Detail": {
    "orderId": "ORD-12345",
    "customerId": "CUST-67890",
    "amount": 15000,
    "items": [
      {"productId": "PROD-001", "quantity": 2}
    ]
  }
}
```

```bash
# EventBridge ルールの作成
aws events put-rule \
  --name "high-value-orders" \
  --event-pattern '{
    "source": ["com.myapp.orders"],
    "detail-type": ["OrderCreated"],
    "detail": {
      "amount": [{"numeric": [">=", 10000]}]
    }
  }'

# ターゲットの設定（Lambda 関数）
aws events put-targets \
  --rule "high-value-orders" \
  --targets '[
    {
      "Id": "notify-vip-team",
      "Arn": "arn:aws:lambda:...:notify-vip-team"
    },
    {
      "Id": "fraud-check",
      "Arn": "arn:aws:lambda:...:fraud-check"
    }
  ]'
```

### 2.4 EventBridge Pipes による変換パイプライン

```
EventBridge Pipes の構成:

ソース         フィルタリング      エンリッチメント       ターゲット
+--------+     +----------+       +------------+        +---------+
| SQS    | --> | イベント  | --->  | Lambda     | -----> | Step    |
| DynamoDB|    | パターン  |       | (データ変換) |        | Functions|
| Kinesis |    | マッチング |       | API GW     |        | Lambda  |
+--------+     +----------+       +------------+        +---------+
```

```yaml
# EventBridge Pipes の SAM テンプレート
Resources:
  OrderProcessingPipe:
    Type: AWS::Pipes::Pipe
    Properties:
      Name: order-processing-pipe
      RoleArn: !GetAtt PipeRole.Arn
      Source: !GetAtt OrderQueue.Arn
      SourceParameters:
        SqsQueueParameters:
          BatchSize: 10
          MaximumBatchingWindowInSeconds: 30
        FilterCriteria:
          Filters:
            - Pattern: '{"body": {"orderType": ["premium"]}}'
      Enrichment: !GetAtt EnrichmentFunction.Arn
      Target: !Ref OrderProcessingStateMachine
      TargetParameters:
        StepFunctionStateMachineParameters:
          InvocationType: FIRE_AND_FORGET
```

### 2.5 DynamoDB Streams によるデータ変更キャプチャ

```python
# dynamodb_stream_handler.py -- DynamoDB Streams CDC パターン
import json
import boto3
import os
from datetime import datetime

sqs = boto3.client("sqs")
AUDIT_QUEUE_URL = os.environ["AUDIT_QUEUE_URL"]

def lambda_handler(event, context):
    """DynamoDB Streams からのイベントを処理し、監査キューに送信"""
    for record in event["Records"]:
        event_name = record["eventName"]  # INSERT, MODIFY, REMOVE
        event_id = record["eventID"]
        timestamp = record["dynamodb"]["ApproximateCreationDateTime"]

        audit_event = {
            "eventId": event_id,
            "eventType": event_name,
            "timestamp": str(timestamp),
            "tableName": record["eventSourceARN"].split("/")[1],
            "processedAt": datetime.utcnow().isoformat()
        }

        if event_name in ("INSERT", "MODIFY"):
            new_image = deserialize(record["dynamodb"]["NewImage"])
            audit_event["newImage"] = new_image

        if event_name in ("MODIFY", "REMOVE"):
            old_image = deserialize(record["dynamodb"]["OldImage"])
            audit_event["oldImage"] = old_image

        if event_name == "MODIFY":
            # 変更フィールドの検出
            changes = detect_changes(
                deserialize(record["dynamodb"]["OldImage"]),
                deserialize(record["dynamodb"]["NewImage"])
            )
            audit_event["changedFields"] = changes

        sqs.send_message(
            QueueUrl=AUDIT_QUEUE_URL,
            MessageBody=json.dumps(audit_event),
            MessageGroupId=audit_event.get("newImage", {}).get("id", "default")
        )

    return {"batchItemFailures": []}

def deserialize(image):
    """DynamoDB の型付き形式を通常の dict に変換"""
    from boto3.dynamodb.types import TypeDeserializer
    deserializer = TypeDeserializer()
    return {k: deserializer.deserialize(v) for k, v in image.items()}

def detect_changes(old_image, new_image):
    """新旧イメージの差分を検出"""
    changes = []
    all_keys = set(list(old_image.keys()) + list(new_image.keys()))
    for key in all_keys:
        old_val = old_image.get(key)
        new_val = new_image.get(key)
        if old_val != new_val:
            changes.append({
                "field": key,
                "oldValue": str(old_val),
                "newValue": str(new_val)
            })
    return changes
```

---

## 3. ファンアウトパターン

### 3.1 SNS + SQS ファンアウト

```
                          +-------+     +-------+     +---------+
                     +--> | SQS 1 | --> | Lambda| --> | 在庫更新 |
                     |    +-------+     +-------+     +---------+
                     |
+-----------+   +----+    +-------+     +-------+     +---------+
| 注文イベント| --> | SNS | --> | SQS 2 | --> | Lambda| --> | 請求処理 |
+-----------+   +----+    +-------+     +-------+     +---------+
                     |
                     |    +-------+     +-------+     +---------+
                     +--> | SQS 3 | --> | Lambda| --> | 通知送信 |
                          +-------+     +-------+     +---------+

メリット:
  - 各SQSキューが独立したバッファとして機能
  - 1つのコンシューマーが遅延しても他に影響しない
  - SQSのリトライ/DLQ機能で耐障害性向上
```

```python
# order_publisher.py -- SNS へのイベント発行
import boto3
import json
import os

sns = boto3.client("sns")
TOPIC_ARN = os.environ["ORDER_TOPIC_ARN"]

def lambda_handler(event, context):
    order = json.loads(event["body"])

    # SNS トピックにパブリッシュ
    sns.publish(
        TopicArn=TOPIC_ARN,
        Message=json.dumps(order),
        MessageAttributes={
            "orderType": {
                "DataType": "String",
                "StringValue": order.get("type", "standard")
            },
            "amount": {
                "DataType": "Number",
                "StringValue": str(order["amount"])
            }
        }
    )

    return {
        "statusCode": 202,
        "body": json.dumps({"message": "Order accepted", "orderId": order["id"]})
    }
```

### 3.2 SQS バッチ処理

```python
# batch_processor.py -- SQS バッチ処理 with 部分失敗報告
import json

def lambda_handler(event, context):
    batch_item_failures = []

    for record in event["Records"]:
        try:
            body = json.loads(record["body"])
            # SNS でラップされている場合
            if "Message" in body:
                message = json.loads(body["Message"])
            else:
                message = body

            process_order(message)
        except Exception as e:
            print(f"Error processing {record['messageId']}: {e}")
            batch_item_failures.append({
                "itemIdentifier": record["messageId"]
            })

    return {"batchItemFailures": batch_item_failures}

def process_order(order):
    # 注文処理ロジック
    print(f"Processing order: {order['id']}")
```

### 3.3 SNS フィルタリングポリシー

```json
{
  "orderType": ["premium", "vip"],
  "amount": [{"numeric": [">=", 10000]}],
  "region": [{"prefix": "asia-"}]
}
```

```yaml
# SAM テンプレートでの SNS + SQS ファンアウト定義
Resources:
  OrderTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: order-events

  # 在庫更新キュー - 全注文を受信
  InventoryQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: inventory-updates
      VisibilityTimeout: 60
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt InventoryDLQ.Arn
        maxReceiveCount: 3

  InventoryDLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: inventory-updates-dlq
      MessageRetentionPeriod: 1209600  # 14日

  InventorySubscription:
    Type: AWS::SNS::Subscription
    Properties:
      TopicArn: !Ref OrderTopic
      Protocol: sqs
      Endpoint: !GetAtt InventoryQueue.Arn
      RawMessageDelivery: true

  # VIP 通知キュー - 高額注文のみ受信
  VipNotificationQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: vip-notifications
      VisibilityTimeout: 30

  VipSubscription:
    Type: AWS::SNS::Subscription
    Properties:
      TopicArn: !Ref OrderTopic
      Protocol: sqs
      Endpoint: !GetAtt VipNotificationQueue.Arn
      RawMessageDelivery: true
      FilterPolicy:
        orderType:
          - premium
          - vip
        amount:
          - numeric:
              - ">="
              - 10000

  # SQS ポリシー（SNS からの送信許可）
  InventoryQueuePolicy:
    Type: AWS::SQS::QueuePolicy
    Properties:
      Queues:
        - !Ref InventoryQueue
      PolicyDocument:
        Statement:
          - Effect: Allow
            Principal:
              Service: sns.amazonaws.com
            Action: sqs:SendMessage
            Resource: !GetAtt InventoryQueue.Arn
            Condition:
              ArnEquals:
                aws:SourceArn: !Ref OrderTopic
```

---

## 4. CQRS パターン

### 4.1 CQRS (Command Query Responsibility Segregation)

```
CQRS アーキテクチャ:

書込み側 (Command)                    読取り側 (Query)
+------------+                        +------------+
| API GW     |                        | API GW     |
| POST/PUT   |                        | GET        |
+------------+                        +------------+
     |                                     |
     v                                     v
+------------+                        +------------+
| Lambda     |                        | Lambda     |
| (Writer)   |                        | (Reader)   |
+------------+                        +------------+
     |                                     |
     v                                     v
+------------+    DynamoDB Streams    +------------------+
| DynamoDB   | ---------------------> | DynamoDB (GSI)   |
| (書込み最適化)|    +------------+    | / ElastiCache    |
+------------+    | Lambda     |    | / OpenSearch     |
                  | (同期処理)  |    | (読取り最適化)    |
                  +------------+    +------------------+
                       |
                       v
                  +------------------+
                  | 読取りモデル更新   |
                  +------------------+

メリット:
  - 読み書きを独立してスケーリング
  - 読取りモデルをユースケースに最適化
  - 複雑なクエリを効率化
```

### 4.2 DynamoDB Streams による同期

```python
# stream_processor.py -- DynamoDB Streams から読取りモデルを更新
import boto3
import os
import json

opensearch_endpoint = os.environ["OPENSEARCH_ENDPOINT"]

def lambda_handler(event, context):
    for record in event["Records"]:
        event_name = record["eventName"]  # INSERT, MODIFY, REMOVE

        if event_name in ("INSERT", "MODIFY"):
            new_image = record["dynamodb"]["NewImage"]
            item = deserialize_dynamodb(new_image)
            index_to_opensearch(item)
        elif event_name == "REMOVE":
            old_image = record["dynamodb"]["OldImage"]
            item = deserialize_dynamodb(old_image)
            remove_from_opensearch(item["id"])

def deserialize_dynamodb(image):
    """DynamoDB の型付き形式を通常の dict に変換"""
    from boto3.dynamodb.types import TypeDeserializer
    deserializer = TypeDeserializer()
    return {k: deserializer.deserialize(v) for k, v in image.items()}

def index_to_opensearch(item):
    """OpenSearch にドキュメントをインデックス"""
    import requests
    from requests_aws4auth import AWS4Auth

    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key, credentials.secret_key,
        os.environ["AWS_REGION"], "es",
        session_token=credentials.token
    )

    url = f"{opensearch_endpoint}/items/_doc/{item['id']}"
    requests.put(url, auth=auth, json=item)
```

### 4.3 ElastiCache を使った読取りモデル

```python
# cache_updater.py -- DynamoDB Streams から ElastiCache (Redis) を更新
import boto3
import json
import os
import redis

redis_client = redis.Redis(
    host=os.environ["REDIS_ENDPOINT"],
    port=6379,
    decode_responses=True,
    ssl=True
)

def lambda_handler(event, context):
    for record in event["Records"]:
        event_name = record["eventName"]

        if event_name in ("INSERT", "MODIFY"):
            new_image = deserialize(record["dynamodb"]["NewImage"])
            item_id = new_image["id"]

            # 個別アイテムのキャッシュ更新
            redis_client.set(
                f"item:{item_id}",
                json.dumps(new_image),
                ex=3600  # 1時間TTL
            )

            # カテゴリ別ソート済みセットの更新
            if "category" in new_image and "updatedAt" in new_image:
                redis_client.zadd(
                    f"category:{new_image['category']}",
                    {item_id: float(new_image["updatedAt"])}
                )

            # 検索用の逆引きインデックス更新
            if "tags" in new_image:
                for tag in new_image["tags"]:
                    redis_client.sadd(f"tag:{tag}", item_id)

        elif event_name == "REMOVE":
            old_image = deserialize(record["dynamodb"]["OldImage"])
            item_id = old_image["id"]

            redis_client.delete(f"item:{item_id}")

            if "category" in old_image:
                redis_client.zrem(f"category:{old_image['category']}", item_id)

            if "tags" in old_image:
                for tag in old_image["tags"]:
                    redis_client.srem(f"tag:{tag}", item_id)

    return {"batchItemFailures": []}

def deserialize(image):
    from boto3.dynamodb.types import TypeDeserializer
    d = TypeDeserializer()
    return {k: d.deserialize(v) for k, v in image.items()}
```

---

## 5. Saga パターン

### 5.1 分散トランザクションの管理

```
Saga パターン (Step Functions):

[Start]
   |
   v
+------------------+     失敗
| 1. 在庫予約       | ----------+
+------------------+           |
   |  成功                     v
   v                    +------------------+
+------------------+    | 1'. 在庫予約取消  |
| 2. 決済処理       |    +------------------+
+------------------+           ^
   |  成功      | 失敗         |
   v            +------------>-+
+------------------+
| 3. 配送手配       |
+------------------+           +------------------+
   |  成功      | 失敗  +----> | 2'. 決済返金      |
   v            +------+      +------------------+
+------------------+                   |
| 4. 注文確定       |                   v
+------------------+           +------------------+
   |                           | 1'. 在庫予約取消  |
   v                           +------------------+
 [End]                                 |
                                       v
                                     [Fail]
```

```json
{
  "Comment": "Order Saga",
  "StartAt": "ReserveInventory",
  "States": {
    "ReserveInventory": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:reserve-inventory",
      "Next": "ProcessPayment",
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "Fail"}]
    },
    "ProcessPayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:process-payment",
      "Next": "ArrangeShipping",
      "Catch": [{
        "ErrorEquals": ["States.ALL"],
        "Next": "CancelReservation"
      }]
    },
    "ArrangeShipping": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:arrange-shipping",
      "Next": "ConfirmOrder",
      "Catch": [{
        "ErrorEquals": ["States.ALL"],
        "Next": "RefundPayment"
      }]
    },
    "ConfirmOrder": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:confirm-order",
      "End": true
    },
    "RefundPayment": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:refund-payment",
      "Next": "CancelReservation"
    },
    "CancelReservation": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:cancel-reservation",
      "Next": "Fail"
    },
    "Fail": {
      "Type": "Fail",
      "Error": "OrderFailed",
      "Cause": "Saga compensation completed"
    }
  }
}
```

### 5.2 Step Functions Express Workflow

```
Step Functions のワークフロータイプ:

Standard Workflow:
  - 最大実行時間: 1年
  - 実行保証: 1回のみ (Exactly-once)
  - 料金: 状態遷移ごとに課金 ($0.025/1000遷移)
  - 用途: 長時間ワークフロー、人間の承認待ち

Express Workflow:
  - 最大実行時間: 5分
  - 実行保証: 少なくとも1回 (At-least-once)
  - 料金: 実行回数 + 実行時間で課金
  - 用途: 大量の短時間処理、IoTデータ処理
  - 同期/非同期の2種類

同期 Express:
  API Gateway --> Step Functions (同期) --> レスポンス
  → リクエスト/レスポンスパターンに最適

非同期 Express:
  イベント --> Step Functions (非同期) --> 完了通知
  → バックグラウンド処理に最適
```

```yaml
# Step Functions Express Workflow の SAM テンプレート
Resources:
  OrderProcessingStateMachine:
    Type: AWS::Serverless::StateMachine
    Properties:
      DefinitionUri: statemachine/order-processing.asl.json
      Type: EXPRESS
      Tracing:
        Enabled: true
      Logging:
        Destinations:
          - CloudWatchLogsLogGroup:
              LogGroupArn: !GetAtt StateMachineLogGroup.Arn
        IncludeExecutionData: true
        Level: ALL
      Policies:
        - LambdaInvokePolicy:
            FunctionName: !Ref ReserveInventoryFunction
        - LambdaInvokePolicy:
            FunctionName: !Ref ProcessPaymentFunction
        - LambdaInvokePolicy:
            FunctionName: !Ref ArrangeShippingFunction

  StateMachineLogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: /aws/stepfunctions/order-processing
      RetentionInDays: 30
```

### 5.3 Step Functions の並列処理とエラーハンドリング

```json
{
  "Comment": "Parallel processing with error handling",
  "StartAt": "ValidateInput",
  "States": {
    "ValidateInput": {
      "Type": "Pass",
      "Next": "ParallelProcessing"
    },
    "ParallelProcessing": {
      "Type": "Parallel",
      "Branches": [
        {
          "StartAt": "ProcessImages",
          "States": {
            "ProcessImages": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:...:process-images",
              "Retry": [
                {
                  "ErrorEquals": ["ServiceException", "TooManyRequestsException"],
                  "IntervalSeconds": 2,
                  "MaxAttempts": 3,
                  "BackoffRate": 2.0
                }
              ],
              "End": true
            }
          }
        },
        {
          "StartAt": "ProcessMetadata",
          "States": {
            "ProcessMetadata": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:...:process-metadata",
              "Retry": [
                {
                  "ErrorEquals": ["States.TaskFailed"],
                  "IntervalSeconds": 1,
                  "MaxAttempts": 2,
                  "BackoffRate": 2.0
                }
              ],
              "End": true
            }
          }
        },
        {
          "StartAt": "SendNotification",
          "States": {
            "SendNotification": {
              "Type": "Task",
              "Resource": "arn:aws:lambda:...:send-notification",
              "End": true
            }
          }
        }
      ],
      "Catch": [
        {
          "ErrorEquals": ["States.ALL"],
          "Next": "HandleError",
          "ResultPath": "$.error"
        }
      ],
      "Next": "AggregateResults"
    },
    "AggregateResults": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:aggregate-results",
      "End": true
    },
    "HandleError": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:...:handle-error",
      "End": true
    }
  }
}
```

---

## 6. ストリーム処理パターン

### 6.1 Kinesis Data Streams + Lambda

```
リアルタイムストリーム処理:

データソース          ストリーム           処理              格納
+----------+       +----------+       +----------+       +----------+
| IoT      | ----> |          | ----> | Lambda   | ----> | DynamoDB |
| デバイス  |       |          |       | (リアルタイム|      | (最新状態) |
+----------+       | Kinesis  |       |  集約)    |       +----------+
                   | Data     |       +----------+
+----------+       | Streams  |                          +----------+
| Web      | ----> |          | ----> +----------+ ----> | S3       |
| クリック  |       |          |       | Firehose |       | (履歴)   |
+----------+       +----------+       +----------+       +----------+
                                                         +----------+
                                                  -----> | OpenSearch|
                                                         | (検索)   |
                                                         +----------+
```

```python
# kinesis_processor.py -- Kinesis Data Streams のレコード処理
import json
import base64
import boto3
import os
from datetime import datetime
from collections import defaultdict

dynamodb = boto3.resource("dynamodb")
metrics_table = dynamodb.Table(os.environ["METRICS_TABLE"])

def lambda_handler(event, context):
    """Kinesis ストリームからのイベントを集約処理"""
    batch_item_failures = []
    aggregated = defaultdict(lambda: {"count": 0, "total_value": 0})

    for record in event["Records"]:
        try:
            # Kinesis レコードのデコード
            payload = base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
            data = json.loads(payload)

            # 時間ウィンドウでの集約
            timestamp = datetime.fromisoformat(data["timestamp"])
            window_key = timestamp.strftime("%Y-%m-%dT%H:%M")  # 分単位
            metric_key = f"{data['metric_name']}#{window_key}"

            aggregated[metric_key]["count"] += 1
            aggregated[metric_key]["total_value"] += data.get("value", 0)
            aggregated[metric_key]["metric_name"] = data["metric_name"]
            aggregated[metric_key]["window"] = window_key

        except Exception as e:
            print(f"Error processing record {record['kinesis']['sequenceNumber']}: {e}")
            batch_item_failures.append({
                "itemIdentifier": record["kinesis"]["sequenceNumber"]
            })

    # 集約結果をDynamoDBに書き込み
    with metrics_table.batch_writer() as batch:
        for key, agg in aggregated.items():
            batch.put_item(Item={
                "pk": key,
                "metric_name": agg["metric_name"],
                "window": agg["window"],
                "count": agg["count"],
                "total_value": int(agg["total_value"]),
                "avg_value": int(agg["total_value"] / agg["count"]),
                "ttl": int(datetime.utcnow().timestamp()) + 86400 * 7  # 7日保持
            })

    return {"batchItemFailures": batch_item_failures}
```

### 6.2 Kinesis のシャード管理とスケーリング

```yaml
# Kinesis Data Streams + Lambda の SAM テンプレート
Resources:
  ClickStream:
    Type: AWS::Kinesis::Stream
    Properties:
      Name: click-stream
      ShardCount: 4
      StreamModeDetails:
        StreamMode: ON_DEMAND  # 自動スケーリング
      RetentionPeriodHours: 168  # 7日間保持
      StreamEncryption:
        EncryptionType: KMS
        KeyId: alias/aws/kinesis

  StreamProcessor:
    Type: AWS::Serverless::Function
    Properties:
      Handler: kinesis_processor.lambda_handler
      Runtime: python3.12
      MemorySize: 512
      Timeout: 300
      Events:
        KinesisEvent:
          Type: Kinesis
          Properties:
            Stream: !GetAtt ClickStream.Arn
            StartingPosition: LATEST
            BatchSize: 100
            MaximumBatchingWindowInSeconds: 30
            ParallelizationFactor: 10
            MaximumRetryAttempts: 3
            BisectBatchOnFunctionError: true
            DestinationConfig:
              OnFailure:
                Destination: !GetAtt FailedRecordsDLQ.Arn
            FunctionResponseTypes:
              - ReportBatchItemFailures
```

---

## 7. スケジュール駆動パターン

### 7.1 定期実行ジョブ

```
スケジュール駆動パターン:

+-------------------+       +----------+       +----------+
| EventBridge       | ----> | Lambda   | ----> | 処理結果  |
| Scheduler         |       |          |       |          |
+-------------------+       +----------+       +----------+

ユースケース:
  - 日次レポート生成
  - 古いデータの定期削除 (TTL 補助)
  - ヘルスチェック / 外部API監視
  - データ同期 / ETL バッチ
  - 一時ファイルのクリーンアップ
```

```python
# scheduled_report.py -- 日次レポート生成
import boto3
import json
import os
from datetime import datetime, timedelta

dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3")
ses = boto3.client("ses")

orders_table = dynamodb.Table(os.environ["ORDERS_TABLE"])
REPORT_BUCKET = os.environ["REPORT_BUCKET"]
ADMIN_EMAIL = os.environ["ADMIN_EMAIL"]

def lambda_handler(event, context):
    """毎日 AM 9:00 (JST) に前日の注文レポートを生成"""
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    # 前日の注文データを取得
    response = orders_table.query(
        IndexName="date-index",
        KeyConditionExpression="orderDate = :date",
        ExpressionAttributeValues={":date": yesterday}
    )
    orders = response["Items"]

    # レポート生成
    report = generate_report(yesterday, orders)

    # S3 にレポートを保存
    report_key = f"reports/daily/{yesterday}.json"
    s3.put_object(
        Bucket=REPORT_BUCKET,
        Key=report_key,
        Body=json.dumps(report, ensure_ascii=False, indent=2),
        ContentType="application/json"
    )

    # メール通知
    send_report_email(yesterday, report)

    return {"date": yesterday, "orderCount": len(orders)}

def generate_report(date, orders):
    total_revenue = sum(float(o.get("amount", 0)) for o in orders)
    categories = {}
    for order in orders:
        cat = order.get("category", "other")
        categories[cat] = categories.get(cat, 0) + 1

    return {
        "date": date,
        "totalOrders": len(orders),
        "totalRevenue": total_revenue,
        "averageOrderValue": total_revenue / len(orders) if orders else 0,
        "ordersByCategory": categories,
        "generatedAt": datetime.utcnow().isoformat()
    }

def send_report_email(date, report):
    ses.send_email(
        Source=ADMIN_EMAIL,
        Destination={"ToAddresses": [ADMIN_EMAIL]},
        Message={
            "Subject": {"Data": f"日次注文レポート: {date}"},
            "Body": {
                "Text": {
                    "Data": f"注文件数: {report['totalOrders']}\n"
                            f"売上合計: ¥{report['totalRevenue']:,.0f}\n"
                            f"平均注文額: ¥{report['averageOrderValue']:,.0f}"
                }
            }
        }
    )
```

```yaml
# スケジュール駆動の SAM テンプレート
Resources:
  DailyReportFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: scheduled_report.lambda_handler
      Runtime: python3.12
      MemorySize: 256
      Timeout: 300
      Events:
        DailySchedule:
          Type: ScheduleV2
          Properties:
            ScheduleExpression: cron(0 0 * * ? *)  # 毎日 AM 9:00 JST (UTC 0:00)
            ScheduleExpressionTimezone: Asia/Tokyo
            RetryPolicy:
              MaximumRetryAttempts: 2
              MaximumEventAgeInSeconds: 3600
```

---

## 8. Web アプリケーションパターン

### 8.1 フルスタックサーバーレス構成

```
フルスタックサーバーレスアーキテクチャ:

ユーザー
    |
    v
+-------------------+
| CloudFront        |  (CDN + カスタムドメイン + SSL)
+-------------------+
    |             |
    v             v
+--------+  +-----------+
| S3     |  | API GW    |  (/api/* パスパターン)
| (SPA)  |  | (HTTP API)|
+--------+  +-----------+
                  |
                  v
            +----------+
            | Lambda   |
            +----------+
                  |
    +-------------+-------------+
    v             v             v
+--------+  +----------+  +---------+
|DynamoDB|  | Cognito  |  | S3      |
|(データ) |  | (認証)   |  |(ファイル)|
+--------+  +----------+  +---------+
```

```yaml
# フルスタックサーバーレスの SAM テンプレート
Resources:
  # S3 バケット (フロントエンド)
  WebBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::StackName}-web'
      PublicAccessBlockConfiguration:
        BlockPublicAcls: true
        BlockPublicPolicy: true
        IgnorePublicAcls: true
        RestrictPublicBuckets: true

  # CloudFront OAC
  CloudFrontOAC:
    Type: AWS::CloudFront::OriginAccessControl
    Properties:
      OriginAccessControlConfig:
        Name: !Sub '${AWS::StackName}-oac'
        OriginAccessControlOriginType: s3
        SigningBehavior: always
        SigningProtocol: sigv4

  # CloudFront ディストリビューション
  Distribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Origins:
          - Id: S3Origin
            DomainName: !GetAtt WebBucket.RegionalDomainName
            OriginAccessControlId: !GetAtt CloudFrontOAC.Id
            S3OriginConfig:
              OriginAccessIdentity: ''
          - Id: ApiOrigin
            DomainName: !Sub '${HttpApi}.execute-api.${AWS::Region}.amazonaws.com'
            CustomOriginConfig:
              HTTPSPort: 443
              OriginProtocolPolicy: https-only
        DefaultCacheBehavior:
          TargetOriginId: S3Origin
          ViewerProtocolPolicy: redirect-to-https
          CachePolicyId: 658327ea-f89d-4fab-a63d-7e88639e58f6  # CachingOptimized
        CacheBehaviors:
          - PathPattern: /api/*
            TargetOriginId: ApiOrigin
            ViewerProtocolPolicy: https-only
            CachePolicyId: 4135ea2d-6df8-44a3-9df3-4b5a84be39ad  # CachingDisabled
            OriginRequestPolicyId: b689b0a8-53d0-40ab-baf2-68738e2966ac  # AllViewerExceptHostHeader
        DefaultRootObject: index.html
        CustomErrorResponses:
          - ErrorCode: 404
            ResponseCode: 200
            ResponsePagePath: /index.html  # SPA ルーティング対応
        Enabled: true
        HttpVersion: http2and3
```

---

## 9. パターン比較表

| パターン | ユースケース | 複雑さ | レイテンシ | コスト効率 |
|---------|------------|--------|----------|-----------|
| API + Lambda + DynamoDB | CRUD API | 低 | 低 | 高 |
| イベント駆動 | 非同期処理 | 中 | 中 | 高 |
| ファンアウト (SNS+SQS) | 1対多通知 | 中 | 中 | 高 |
| CQRS | 読み書き分離 | 高 | 読取り: 低 | 中 |
| Saga | 分散トランザクション | 高 | 高 | 中 |
| ストリーム処理 | リアルタイム集約 | 中 | 低〜中 | 中 |
| スケジュール駆動 | 定期バッチ | 低 | N/A | 高 |
| フルスタック | Web アプリ | 中 | 低 | 高 |

| パターン | スケーラビリティ | 結合度 | 運用難易度 |
|---------|----------------|--------|-----------|
| API + Lambda + DynamoDB | 高 | 中 | 低 |
| イベント駆動 | 高 | 低 | 中 |
| ファンアウト (SNS+SQS) | 高 | 低 | 中 |
| CQRS | 非常に高 | 低 | 高 |
| Saga | 高 | 低 | 高 |
| ストリーム処理 | 非常に高 | 低 | 中 |
| スケジュール駆動 | 高 | 低 | 低 |
| フルスタック | 高 | 中 | 中 |

---

## 10. コールドスタート対策

### 10.1 コールドスタートの仕組み

```
Lambda コールドスタートの発生フロー:

初回リクエスト (コールドスタート):
  [リクエスト] --> [環境準備: ~200ms] --> [コード読込: ~100ms] --> [初期化: ~500ms] --> [処理]
                   MicroVM作成          デプロイパッケージ展開    ランタイム初期化

後続リクエスト (ウォームスタート):
  [リクエスト] --> [処理]
                  既存の実行環境を再利用

コールドスタートの要因:
  - 新しいリクエストに空き実行環境がない
  - 一定時間アイドル後に実行環境が回収された
  - Lambda のデプロイ/設定変更後
  - VPC 内 Lambda の ENI 作成 (現在は大幅に改善)
```

### 10.2 Provisioned Concurrency

```bash
# Provisioned Concurrency の設定
aws lambda put-provisioned-concurrency-config \
  --function-name my-api-function \
  --qualifier prod \
  --provisioned-concurrent-executions 10

# Application Auto Scaling との連携
aws application-autoscaling register-scalable-target \
  --service-namespace lambda \
  --resource-id "function:my-api-function:prod" \
  --scalable-dimension "lambda:function:ProvisionedConcurrency" \
  --min-capacity 5 \
  --max-capacity 50

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

### 10.3 SnapStart (Java)

```yaml
# SnapStart 対応 Lambda (Java)
Resources:
  JavaFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: com.example.Handler::handleRequest
      Runtime: java21
      MemorySize: 1024
      SnapStart:
        ApplyOn: PublishedVersions
      AutoPublishAlias: live
```

### 10.4 コールドスタート削減のベストプラクティス

```
コールドスタート対策チェックリスト:

1. デプロイパッケージの最小化
   - 不要な依存関係を除外
   - Lambda Layers で共通ライブラリを分離
   - Tree-shaking で未使用コードを除外 (Node.js)

2. ランタイム選択
   - 高速: Python, Node.js (~100ms)
   - 中速: Go, .NET (~200ms)
   - 低速: Java (~500ms, SnapStart で改善可能)

3. SDK クライアントの初期化
   - ハンドラー外で SDK クライアントを初期化 (グローバルスコープ)
   - 接続の再利用を有効化

4. メモリ設定
   - メモリを増やすと CPU も比例して増加
   - 初期化時間が短縮される場合がある
   - AWS Lambda Power Tuning で最適値を検出

5. VPC 設定
   - 不要な場合は VPC 設定を避ける
   - VPC が必要な場合はハイパープレーンENI (大幅改善済み)
```

---

## 11. アンチパターン

### 11.1 Lambda チェーン (同期的な連鎖呼び出し)

```
[悪い例] Lambda から Lambda を直接同期呼び出し

Lambda A --> Lambda B --> Lambda C --> Lambda D
  3秒        2秒         1秒         2秒
  合計: 8秒 (全Lambda の実行時間で課金)

[良い例] Step Functions でオーケストレーション

Step Functions --> Lambda A (3秒)
              --> Lambda B (2秒)
              --> Lambda C (1秒)
              --> Lambda D (2秒)
  各Lambdaは自分の実行時間のみ課金
```

**問題点**: 前段の Lambda が後段の完了を待つ間も課金される。エラーハンドリングが複雑になり、タイムアウトのリスクが連鎖する。

**改善**: Step Functions、SQS、EventBridge を使って非同期に連携する。

### 11.2 DynamoDB のスキャンに依存した API

```python
# [悪い例] 全件スキャンで検索
def search_items(keyword):
    result = table.scan(
        FilterExpression=Attr("name").contains(keyword)
    )
    return result["Items"]  # テーブル全体を読み取る

# [良い例] GSI を活用した効率的なクエリ
def search_items(category, date_from):
    result = table.query(
        IndexName="category-date-index",
        KeyConditionExpression=Key("category").eq(category) & Key("created_at").gte(date_from)
    )
    return result["Items"]  # 必要な範囲のみ読み取る
```

### 11.3 Lambda 関数内でのハードコーディング

```python
# [悪い例] 接続先やシークレットをハードコーディング
import boto3

def lambda_handler(event, context):
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table("my-production-table")  # テーブル名がハードコード
    api_key = "sk-abc123def456"  # シークレットがコード内

# [良い例] 環境変数と Secrets Manager を活用
import boto3
import os
import json

# ハンドラー外で初期化（コールドスタート時のみ実行）
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(os.environ["TABLE_NAME"])
secrets_client = boto3.client("secretsmanager")

_cached_secret = None

def get_api_key():
    global _cached_secret
    if _cached_secret is None:
        response = secrets_client.get_secret_value(
            SecretId=os.environ["API_KEY_SECRET_ARN"]
        )
        _cached_secret = json.loads(response["SecretString"])["api_key"]
    return _cached_secret

def lambda_handler(event, context):
    api_key = get_api_key()
    # 処理...
```

### 11.4 Lambda のタイムアウトと SQS の可視性タイムアウトの不整合

```
[悪い例]
Lambda タイムアウト: 300秒
SQS 可視性タイムアウト: 30秒

→ Lambda が 30秒以上かかると、SQS がメッセージを再度配信
→ 同じメッセージが複数のLambdaで同時処理される

[良い例]
Lambda タイムアウト: 300秒
SQS 可視性タイムアウト: 360秒 (Lambda タイムアウト + マージン)

→ Lambda が処理中の間、他のコンシューマーはメッセージを受信しない
```

### 11.5 モノリシック Lambda 関数

```
[悪い例] 単一の巨大Lambda関数
Lambda Function (1つ):
  - ユーザー管理
  - 注文処理
  - 在庫管理
  - レポート生成
  → デプロイが遅い、メモリが無駄、権限が過剰

[良い例] 機能別に分割
Lambda: user-management    → IAM: DynamoDB Users テーブルのみ
Lambda: order-processing   → IAM: DynamoDB Orders テーブル + SQS
Lambda: inventory-manager  → IAM: DynamoDB Inventory テーブル
Lambda: report-generator   → IAM: S3 + DynamoDB ReadOnly
  → 最小権限、独立デプロイ、適切なリソース配分
```

---

## 12. 監視とオブザーバビリティ

### 12.1 サーバーレスの監視戦略

```
サーバーレス監視の4つの柱:

1. メトリクス (CloudWatch Metrics)
   - Lambda: Invocations, Duration, Errors, Throttles, ConcurrentExecutions
   - API GW: Count, Latency, 4XXError, 5XXError
   - DynamoDB: ConsumedReadCapacityUnits, ThrottledRequests
   - SQS: ApproximateNumberOfMessagesVisible, ApproximateAgeOfOldestMessage

2. ログ (CloudWatch Logs + Logs Insights)
   - 構造化ログ (JSON) で出力
   - 相関ID でリクエストを追跡
   - Logs Insights でクエリ分析

3. トレース (AWS X-Ray)
   - サービス間の呼び出しを可視化
   - ボトルネックの特定
   - エラーの発生箇所の特定

4. アラーム (CloudWatch Alarms + SNS)
   - エラー率の閾値超過
   - レイテンシの異常増加
   - DLQ にメッセージ滞留
```

```yaml
# 監視アラームの CloudFormation テンプレート
Resources:
  LambdaErrorAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${AWS::StackName}-lambda-errors'
      AlarmDescription: Lambda エラー率が5%を超過
      Namespace: AWS/Lambda
      MetricName: Errors
      Dimensions:
        - Name: FunctionName
          Value: !Ref ApiFunction
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 2
      Threshold: 5
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Ref AlertTopic

  ApiLatencyAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${AWS::StackName}-api-latency'
      AlarmDescription: API レイテンシ P99 が 3秒を超過
      Namespace: AWS/ApiGateway
      MetricName: Latency
      Dimensions:
        - Name: ApiName
          Value: !Ref HttpApi
      ExtendedStatistic: p99
      Period: 300
      EvaluationPeriods: 3
      Threshold: 3000
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Ref AlertTopic

  DLQAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${AWS::StackName}-dlq-messages'
      AlarmDescription: DLQ にメッセージが滞留
      Namespace: AWS/SQS
      MetricName: ApproximateNumberOfMessagesVisible
      Dimensions:
        - Name: QueueName
          Value: !GetAtt DeadLetterQueue.QueueName
      Statistic: Sum
      Period: 60
      EvaluationPeriods: 1
      Threshold: 1
      ComparisonOperator: GreaterThanOrEqualToThreshold
      AlarmActions:
        - !Ref AlertTopic

  AlertTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub '${AWS::StackName}-alerts'
      Subscription:
        - Endpoint: ops-team@example.com
          Protocol: email
```

---

## 13. FAQ

### Q1. サーバーレスアーキテクチャの適用に向かないケースは？

長時間実行(15分超)、高頻度・定常トラフィック(EC2/ECS の方がコスト有利)、GPU が必要な ML 推論、WebSocket の長時間接続(API Gateway WebSocket の制限内なら可能)、レイテンシに極めて敏感なリアルタイム処理(コールドスタートが許容できない場合)が該当する。

### Q2. イベント駆動とリクエスト・レスポンスの使い分けは？

ユーザーが即座に結果を必要とする操作(認証、データ取得)はリクエスト・レスポンスが適している。結果が遅延しても問題ない操作(メール送信、レポート生成、データ同期)はイベント駆動が適している。多くの場合、1つのシステム内で両方のパターンを組み合わせる。

### Q3. DynamoDB と RDS のどちらを選ぶべきですか？

DynamoDB はキーバリュー/ドキュメント型のアクセスパターンに強く、サーバーレスとの親和性が高い。RDS はリレーショナルデータモデルが必要な場合、複雑な JOIN やトランザクションが頻繁な場合に適している。Lambda + RDS の場合は RDS Proxy によるコネクション管理が必須となる。

### Q4. Lambda のメモリサイズはどう決めるべきですか？

AWS Lambda Power Tuning ツール（https://github.com/alexcasalboni/aws-lambda-power-tuning）を使って、コストとパフォーマンスの最適バランスを見つけるのが推奨される。一般的に、CPU バウンドの処理はメモリを増やすと処理時間が短縮され、トータルコストが下がる場合がある。I/O バウンドの処理では、メモリを増やしても効果が限定的になる。

### Q5. サーバーレスでのテスト戦略はどうあるべきですか？

ユニットテストはビジネスロジックを Lambda ハンドラーから分離してテストする。統合テストは LocalStack や SAM CLI の `sam local invoke` を使ってローカルでテストする。E2E テストはステージング環境にデプロイして実施する。コントラクトテストでイベントスキーマの互換性を検証する。

### Q6. Lambda 関数のデプロイパッケージサイズ制限は？

直接アップロード: 50MB (zip 圧縮後)、S3 経由: 250MB (解凍後)、コンテナイメージ: 10GB。デプロイパッケージが大きい場合は Lambda Layers で共通ライブラリを分離するか、コンテナイメージとしてデプロイする。

---

## まとめ

| パターン | 構成要素 | 主な用途 |
|---------|---------|---------|
| API バックエンド | API GW + Lambda + DynamoDB | RESTful API |
| GraphQL | AppSync + Lambda + DynamoDB | GraphQL API |
| イベント駆動 | EventBridge + Lambda | 非同期処理、マイクロサービス連携 |
| ファンアウト | SNS + SQS + Lambda | 1対多の並列処理 |
| CQRS | DynamoDB Streams + Lambda | 読み書き分離、検索最適化 |
| Saga | Step Functions + Lambda | 分散トランザクション |
| ストリーム処理 | Kinesis + Lambda | リアルタイムデータ集約 |
| スケジュール駆動 | EventBridge Scheduler + Lambda | 定期バッチ処理 |
| フルスタック | CloudFront + S3 + API GW + Lambda | Web アプリケーション |

---

## 次に読むべきガイド

- [ECS 基礎](../06-containers/00-ecs-basics.md) -- コンテナによる代替アーキテクチャ
- [CloudFormation](../07-devops/00-cloudformation.md) -- サーバーレスインフラのコード化
- [IAM 詳解](../08-security/00-iam-deep-dive.md) -- サーバーレスのセキュリティ設計

---

## 参考文献

1. AWS 公式「Serverless Application Lens - AWS Well-Architected Framework」 https://docs.aws.amazon.com/wellarchitected/latest/serverless-applications-lens/
2. Alex DeBrie「The DynamoDB Book」DynamoDB Book, 2020
3. AWS Samples「Serverless Patterns Collection」 https://serverlessland.com/patterns
4. Gregor Hohpe, Bobby Woolf「Enterprise Integration Patterns」Addison-Wesley, 2003
5. AWS 公式「Lambda Powertools for Python」 https://docs.powertools.aws.dev/lambda/python/latest/
6. AWS 公式「Step Functions デベロッパーガイド」 https://docs.aws.amazon.com/step-functions/latest/dg/
