# サーバーレスパターン

> API+Lambda+DynamoDB、イベント駆動、ファンアウト、CQRS などの代表的なサーバーレスアーキテクチャパターンを理解し、実践的な設計判断ができるようになる。

---

## この章で学ぶこと

1. **API バックエンドパターン** -- API Gateway + Lambda + DynamoDB の組み合わせで RESTful/GraphQL API を構築する手法
2. **イベント駆動パターン** -- SNS/SQS/EventBridge を活用した疎結合アーキテクチャの設計方法
3. **高度なパターン** -- ファンアウト、CQRS、Saga パターンなどの複雑なユースケースへの対応

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
| 用途 | フル機能 API | 軽量 API, プロキシ | リアルタイム通信 |

### 1.4 SAM テンプレートでの定義

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

Resources:
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

---

## 6. パターン比較表

| パターン | ユースケース | 複雑さ | レイテンシ | コスト効率 |
|---------|------------|--------|----------|-----------|
| API + Lambda + DynamoDB | CRUD API | 低 | 低 | 高 |
| イベント駆動 | 非同期処理 | 中 | 中 | 高 |
| ファンアウト (SNS+SQS) | 1対多通知 | 中 | 中 | 高 |
| CQRS | 読み書き分離 | 高 | 読取り: 低 | 中 |
| Saga | 分散トランザクション | 高 | 高 | 中 |

| パターン | スケーラビリティ | 結合度 | 運用難易度 |
|---------|----------------|--------|-----------|
| API + Lambda + DynamoDB | 高 | 中 | 低 |
| イベント駆動 | 高 | 低 | 中 |
| ファンアウト (SNS+SQS) | 高 | 低 | 中 |
| CQRS | 非常に高 | 低 | 高 |
| Saga | 高 | 低 | 高 |

---

## 7. アンチパターン

### 7.1 Lambda チェーン (同期的な連鎖呼び出し)

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

### 7.2 DynamoDB のスキャンに依存した API

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

---

## 8. FAQ

### Q1. サーバーレスアーキテクチャの適用に向かないケースは？

長時間実行(15分超)、高頻度・定常トラフィック(EC2/ECS の方がコスト有利)、GPU が必要な ML 推論、WebSocket の長時間接続(API Gateway WebSocket の制限内なら可能)、レイテンシに極めて敏感なリアルタイム処理(コールドスタートが許容できない場合)が該当する。

### Q2. イベント駆動とリクエスト・レスポンスの使い分けは？

ユーザーが即座に結果を必要とする操作(認証、データ取得)はリクエスト・レスポンスが適している。結果が遅延しても問題ない操作(メール送信、レポート生成、データ同期)はイベント駆動が適している。多くの場合、1つのシステム内で両方のパターンを組み合わせる。

### Q3. DynamoDB と RDS のどちらを選ぶべきですか？

DynamoDB はキーバリュー/ドキュメント型のアクセスパターンに強く、サーバーレスとの親和性が高い。RDS はリレーショナルデータモデルが必要な場合、複雑な JOIN やトランザクションが頻繁な場合に適している。Lambda + RDS の場合は RDS Proxy によるコネクション管理が必須となる。

---

## まとめ

| パターン | 構成要素 | 主な用途 |
|---------|---------|---------|
| API バックエンド | API GW + Lambda + DynamoDB | RESTful API |
| イベント駆動 | EventBridge + Lambda | 非同期処理、マイクロサービス連携 |
| ファンアウト | SNS + SQS + Lambda | 1対多の並列処理 |
| CQRS | DynamoDB Streams + Lambda | 読み書き分離、検索最適化 |
| Saga | Step Functions + Lambda | 分散トランザクション |

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
