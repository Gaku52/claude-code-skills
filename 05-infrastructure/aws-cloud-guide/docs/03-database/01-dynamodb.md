# Amazon DynamoDB

> AWS のフルマネージド NoSQL データベースを理解し、テーブル設計・GSI/LSI・キャパシティモード・DynamoDB Streams・バックアップ・グローバルテーブルを実践的に習得する

## この章で学ぶこと

1. **DynamoDB のデータモデル** — パーティションキー、ソートキー、アイテム構造の設計原則
2. **セカンダリインデックス** — GSI（グローバル）と LSI（ローカル）の使い分けとクエリパターン
3. **キャパシティモードと運用** — オンデマンド vs プロビジョンド、DAX キャッシュ、TTL 管理
4. **DynamoDB Streams とイベント駆動** — CDC（変更データキャプチャ）と Lambda 連携
5. **バックアップとリストア** — PITR（ポイントインタイムリカバリ）とオンデマンドバックアップ
6. **グローバルテーブル** — マルチリージョンレプリケーションの設計と運用

---

## 1. DynamoDB の基本アーキテクチャ

```
+----------------------------------------------------------------+
|                    DynamoDB テーブル                             |
|  +----------------------------------------------------------+  |
|  | Partition A (Hash: user#001)                              |  |
|  |  +------+----------+--------+--------+--------+          |  |
|  |  | PK   | SK       | name   | email  | amount |          |  |
|  |  +------+----------+--------+--------+--------+          |  |
|  |  |user  |PROFILE   | Taro   | t@e.co | -      |          |  |
|  |  |#001  |ORDER#001 | -      | -      | 1200   |          |  |
|  |  |      |ORDER#002 | -      | -      | 3400   |          |  |
|  |  +------+----------+--------+--------+--------+          |  |
|  +----------------------------------------------------------+  |
|  | Partition B (Hash: user#002)                              |  |
|  |  +------+----------+--------+--------+--------+          |  |
|  |  |user  |PROFILE   | Hanako | h@e.co | -      |          |  |
|  |  |#002  |ORDER#001 | -      | -      | 5600   |          |  |
|  |  +------+----------+--------+--------+--------+          |  |
|  +----------------------------------------------------------+  |
+----------------------------------------------------------------+
```

### パーティショニングの仕組み

DynamoDB は内部的にデータを 10GB 単位のパーティションに分割して格納する。各パーティションは 3 つの AZ（アベイラビリティゾーン）にレプリケートされ、高可用性を実現している。

```
パーティショニングの内部構造:

                    DynamoDB テーブル
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
    Partition 0     Partition 1     Partition 2
    (Hash 0-33%)   (Hash 34-66%)  (Hash 67-100%)
      │  │  │        │  │  │        │  │  │
      ▼  ▼  ▼        ▼  ▼  ▼        ▼  ▼  ▼
     AZ-a AZ-c AZ-d  AZ-a AZ-c AZ-d  AZ-a AZ-c AZ-d
     (3つのレプリカで冗長化)

各パーティションの制限:
  - 最大 10GB のデータ
  - 最大 3,000 RCU / 1,000 WCU（プロビジョンドモード時）
  - パーティション分割は自動で発生（透過的）
```

### コード例 1: テーブル作成（AWS CLI）

```bash
# シングルテーブル設計のテーブルを作成
aws dynamodb create-table \
  --table-name MyApp \
  --attribute-definitions \
    AttributeName=PK,AttributeType=S \
    AttributeName=SK,AttributeType=S \
    AttributeName=GSI1PK,AttributeType=S \
    AttributeName=GSI1SK,AttributeType=S \
  --key-schema \
    AttributeName=PK,KeyType=HASH \
    AttributeName=SK,KeyType=RANGE \
  --global-secondary-indexes \
    '[{
      "IndexName": "GSI1",
      "KeySchema": [
        {"AttributeName":"GSI1PK","KeyType":"HASH"},
        {"AttributeName":"GSI1SK","KeyType":"RANGE"}
      ],
      "Projection": {"ProjectionType":"ALL"}
    }]' \
  --billing-mode PAY_PER_REQUEST \
  --tags Key=Environment,Value=production

# テーブルの状態確認
aws dynamodb describe-table \
  --table-name MyApp \
  --query 'Table.{Status:TableStatus,ItemCount:ItemCount,Size:TableSizeBytes}'

# テーブル一覧の取得
aws dynamodb list-tables --output table
```

### コード例 2: Terraform 定義

```hcl
resource "aws_dynamodb_table" "main" {
  name         = "MyApp"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "PK"
  range_key    = "SK"

  attribute {
    name = "PK"
    type = "S"
  }
  attribute {
    name = "SK"
    type = "S"
  }
  attribute {
    name = "GSI1PK"
    type = "S"
  }
  attribute {
    name = "GSI1SK"
    type = "S"
  }

  global_secondary_index {
    name            = "GSI1"
    hash_key        = "GSI1PK"
    range_key       = "GSI1SK"
    projection_type = "ALL"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  ttl {
    attribute_name = "ExpiresAt"
    enabled        = true
  }

  tags = {
    Environment = "production"
  }
}
```

### コード例 2b: CloudFormation 定義

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: DynamoDB Single Table Design

Parameters:
  Environment:
    Type: String
    Default: production
    AllowedValues: [production, staging, development]

Resources:
  MyAppTable:
    Type: AWS::DynamoDB::Table
    DeletionPolicy: Retain
    UpdateReplacePolicy: Retain
    Properties:
      TableName: !Sub '${Environment}-MyApp'
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: PK
          AttributeType: S
        - AttributeName: SK
          AttributeType: S
        - AttributeName: GSI1PK
          AttributeType: S
        - AttributeName: GSI1SK
          AttributeType: S
        - AttributeName: GSI2PK
          AttributeType: S
        - AttributeName: GSI2SK
          AttributeType: S
      KeySchema:
        - AttributeName: PK
          KeyType: HASH
        - AttributeName: SK
          KeyType: RANGE
      GlobalSecondaryIndexes:
        - IndexName: GSI1
          KeySchema:
            - AttributeName: GSI1PK
              KeyType: HASH
            - AttributeName: GSI1SK
              KeyType: RANGE
          Projection:
            ProjectionType: ALL
        - IndexName: GSI2
          KeySchema:
            - AttributeName: GSI2PK
              KeyType: HASH
            - AttributeName: GSI2SK
              KeyType: RANGE
          Projection:
            ProjectionType: KEYS_ONLY
      PointInTimeRecoverySpecification:
        PointInTimeRecoveryEnabled: true
      SSESpecification:
        SSEEnabled: true
        SSEType: KMS
      TimeToLiveSpecification:
        AttributeName: ExpiresAt
        Enabled: true
      StreamSpecification:
        StreamViewType: NEW_AND_OLD_IMAGES
      Tags:
        - Key: Environment
          Value: !Ref Environment

Outputs:
  TableName:
    Value: !Ref MyAppTable
  TableArn:
    Value: !GetAtt MyAppTable.Arn
  StreamArn:
    Value: !GetAtt MyAppTable.StreamArn
```

### コード例 2c: AWS CDK（TypeScript）定義

```typescript
import * as cdk from 'aws-cdk-lib';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import { Construct } from 'constructs';

export class DynamoDBStack extends cdk.Stack {
  public readonly table: dynamodb.Table;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    this.table = new dynamodb.Table(this, 'MyAppTable', {
      tableName: 'MyApp',
      partitionKey: { name: 'PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'SK', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      pointInTimeRecovery: true,
      encryption: dynamodb.TableEncryption.AWS_MANAGED,
      stream: dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      timeToLiveAttribute: 'ExpiresAt',
    });

    // GSI1: メール検索、ステータス検索
    this.table.addGlobalSecondaryIndex({
      indexName: 'GSI1',
      partitionKey: { name: 'GSI1PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'GSI1SK', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.ALL,
    });

    // GSI2: 日付ベースの検索
    this.table.addGlobalSecondaryIndex({
      indexName: 'GSI2',
      partitionKey: { name: 'GSI2PK', type: dynamodb.AttributeType.STRING },
      sortKey: { name: 'GSI2SK', type: dynamodb.AttributeType.STRING },
      projectionType: dynamodb.ProjectionType.KEYS_ONLY,
    });

    // CloudFormation 出力
    new cdk.CfnOutput(this, 'TableName', { value: this.table.tableName });
    new cdk.CfnOutput(this, 'TableArn', { value: this.table.tableArn });
  }
}
```

---

## 2. シングルテーブル設計

### アクセスパターンからのテーブル設計

```
アクセスパターン          PK           SK            GSI1PK        GSI1SK
---------------------------------------------------------------------------
ユーザー取得          USER#<id>    PROFILE       EMAIL#<e>     USER#<id>
注文一覧(ユーザー別)  USER#<id>    ORDER#<id>    ORDER#<id>    <date>
注文検索(ステータス別) USER#<id>    ORDER#<id>    STATUS#<s>    <date>
商品取得              PROD#<id>    METADATA      CAT#<cat>     PRICE#<p>
```

### 設計プロセスの詳細

シングルテーブル設計は以下のステップで行う。

```
Step 1: エンティティの洗い出し
  - ユーザー (User)
  - 注文 (Order)
  - 商品 (Product)
  - カテゴリ (Category)

Step 2: アクセスパターンの列挙
  AP-1: ユーザー ID でプロフィール取得
  AP-2: ユーザー ID で全注文取得
  AP-3: メールアドレスでユーザー検索
  AP-4: ステータス + 日付範囲で注文検索
  AP-5: カテゴリ + 価格範囲で商品検索
  AP-6: 注文 ID で注文詳細取得
  AP-7: ユーザーのプロフィールと最新N件の注文を一括取得

Step 3: PK/SK の設計
  - PK はエンティティタイプ + ID のプレフィックスパターン
  - SK はアイテムタイプまたはソート用の値

Step 4: GSI の設計
  - 1つの GSI で複数のアクセスパターンに対応（オーバーロード）
  - 必要最小限の GSI 数に抑える

Step 5: スパースインデックスの活用
  - GSI のキー属性が存在しないアイテムはインデックスに含まれない
  - これを利用してフィルタリング済みのインデックスを構成
```

### コード例 3: CRUD 操作（Python / boto3）

```python
import boto3
from datetime import datetime, timezone
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError

dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
table = dynamodb.Table('MyApp')

# === Create: ユーザー作成 ===
def create_user(user_id: str, name: str, email: str):
    table.put_item(
        Item={
            'PK': f'USER#{user_id}',
            'SK': 'PROFILE',
            'GSI1PK': f'EMAIL#{email}',
            'GSI1SK': f'USER#{user_id}',
            'name': name,
            'email': email,
            'created_at': datetime.now(timezone.utc).isoformat(),
        },
        ConditionExpression='attribute_not_exists(PK)',  # 重複防止
    )

# === Read: ユーザーと全注文を一括取得 ===
def get_user_with_orders(user_id: str):
    response = table.query(
        KeyConditionExpression=Key('PK').eq(f'USER#{user_id}')
    )
    items = response['Items']
    profile = next((i for i in items if i['SK'] == 'PROFILE'), None)
    orders = [i for i in items if i['SK'].startswith('ORDER#')]
    return {'profile': profile, 'orders': orders}

# === Read: ページネーション対応のクエリ ===
def get_user_orders_paginated(user_id: str, page_size: int = 20, last_key: dict = None):
    params = {
        'KeyConditionExpression': (
            Key('PK').eq(f'USER#{user_id}') &
            Key('SK').begins_with('ORDER#')
        ),
        'Limit': page_size,
        'ScanIndexForward': False,  # 新しい順
    }
    if last_key:
        params['ExclusiveStartKey'] = last_key

    response = table.query(**params)
    return {
        'orders': response['Items'],
        'last_key': response.get('LastEvaluatedKey'),
        'has_more': 'LastEvaluatedKey' in response,
    }

# === Update: ユーザー名の更新 ===
def update_user_name(user_id: str, new_name: str):
    table.update_item(
        Key={'PK': f'USER#{user_id}', 'SK': 'PROFILE'},
        UpdateExpression='SET #n = :name, updated_at = :now',
        ExpressionAttributeNames={'#n': 'name'},
        ExpressionAttributeValues={
            ':name': new_name,
            ':now': datetime.now(timezone.utc).isoformat(),
        },
        ConditionExpression='attribute_exists(PK)',
    )

# === Update: アトミックカウンター ===
def increment_view_count(product_id: str):
    response = table.update_item(
        Key={'PK': f'PROD#{product_id}', 'SK': 'METADATA'},
        UpdateExpression='SET view_count = if_not_exists(view_count, :zero) + :inc',
        ExpressionAttributeValues={
            ':zero': 0,
            ':inc': 1,
        },
        ReturnValues='UPDATED_NEW',
    )
    return response['Attributes']['view_count']

# === Delete: 注文の削除 ===
def delete_order(user_id: str, order_id: str):
    table.delete_item(
        Key={'PK': f'USER#{user_id}', 'SK': f'ORDER#{order_id}'},
    )

# === Query: GSI を使ったメールアドレス検索 ===
def find_user_by_email(email: str):
    response = table.query(
        IndexName='GSI1',
        KeyConditionExpression=Key('GSI1PK').eq(f'EMAIL#{email}'),
    )
    return response['Items']

# === BatchWrite: 一括書き込み ===
def batch_create_products(products: list):
    with table.batch_writer() as batch:
        for product in products:
            batch.put_item(Item={
                'PK': f'PROD#{product["id"]}',
                'SK': 'METADATA',
                'GSI1PK': f'CAT#{product["category"]}',
                'GSI1SK': f'PRICE#{str(product["price"]).zfill(10)}',
                'name': product['name'],
                'price': product['price'],
                'category': product['category'],
                'created_at': datetime.now(timezone.utc).isoformat(),
            })

# === BatchGet: 一括読み取り ===
def batch_get_users(user_ids: list):
    response = dynamodb.batch_get_item(
        RequestItems={
            'MyApp': {
                'Keys': [
                    {'PK': {'S': f'USER#{uid}'}, 'SK': {'S': 'PROFILE'}}
                    for uid in user_ids
                ],
            }
        }
    )
    return response['Responses']['MyApp']
```

### コード例 4: トランザクション操作

```python
def create_order_with_stock_update(user_id, order_id, product_id, qty, total):
    """注文作成 + 在庫減少をアトミックに実行"""
    client = boto3.client('dynamodb', region_name='ap-northeast-1')

    try:
        client.transact_write_items(
            TransactItems=[
                {
                    'Put': {
                        'TableName': 'MyApp',
                        'Item': {
                            'PK': {'S': f'USER#{user_id}'},
                            'SK': {'S': f'ORDER#{order_id}'},
                            'GSI1PK': {'S': f'STATUS#PENDING'},
                            'GSI1SK': {'S': datetime.now(timezone.utc).isoformat()},
                            'product_id': {'S': product_id},
                            'quantity': {'N': str(qty)},
                            'total': {'N': str(total)},
                        },
                        'ConditionExpression': 'attribute_not_exists(PK)',
                    }
                },
                {
                    'Update': {
                        'TableName': 'MyApp',
                        'Key': {
                            'PK': {'S': f'PROD#{product_id}'},
                            'SK': {'S': 'METADATA'},
                        },
                        'UpdateExpression': 'SET stock = stock - :qty',
                        'ConditionExpression': 'stock >= :qty',
                        'ExpressionAttributeValues': {
                            ':qty': {'N': str(qty)},
                        },
                    }
                },
            ]
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'TransactionCanceledException':
            reasons = e.response.get('CancellationReasons', [])
            for i, reason in enumerate(reasons):
                if reason['Code'] != 'None':
                    print(f"Transaction item {i} failed: {reason['Code']} - {reason.get('Message', '')}")
            raise
        raise


def transfer_between_accounts(from_id, to_id, amount):
    """口座間送金をトランザクションで実行"""
    client = boto3.client('dynamodb', region_name='ap-northeast-1')

    client.transact_write_items(
        TransactItems=[
            {
                'Update': {
                    'TableName': 'MyApp',
                    'Key': {
                        'PK': {'S': f'ACCOUNT#{from_id}'},
                        'SK': {'S': 'BALANCE'},
                    },
                    'UpdateExpression': 'SET balance = balance - :amount',
                    'ConditionExpression': 'balance >= :amount',
                    'ExpressionAttributeValues': {
                        ':amount': {'N': str(amount)},
                    },
                }
            },
            {
                'Update': {
                    'TableName': 'MyApp',
                    'Key': {
                        'PK': {'S': f'ACCOUNT#{to_id}'},
                        'SK': {'S': 'BALANCE'},
                    },
                    'UpdateExpression': 'SET balance = balance + :amount',
                    'ExpressionAttributeValues': {
                        ':amount': {'N': str(amount)},
                    },
                }
            },
            {
                'Put': {
                    'TableName': 'MyApp',
                    'Item': {
                        'PK': {'S': f'TX#{datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")}'},
                        'SK': {'S': f'FROM#{from_id}#TO#{to_id}'},
                        'amount': {'N': str(amount)},
                        'timestamp': {'S': datetime.now(timezone.utc).isoformat()},
                        'status': {'S': 'COMPLETED'},
                    },
                }
            },
        ]
    )
```

---

## 3. GSI と LSI の違い

```
テーブル (PK=UserID, SK=OrderDate)
|
+--- LSI (PK=UserID, SK=OrderAmount)
|      -> 同一パーティション内の別ソート
|      -> テーブル作成時のみ定義可能
|      -> 10GB/パーティション制限を共有
|
+--- GSI (PK=ProductID, SK=OrderDate)
       -> 完全に別のパーティション構成
       -> いつでも追加/削除可能
       -> 独自のキャパシティ設定
```

### GSI vs LSI 比較表

| 特性 | GSI（グローバル） | LSI（ローカル） |
|---|---|---|
| **パーティションキー** | テーブルと異なるキー可 | テーブルと同じ PK |
| **ソートキー** | 任意の属性 | テーブルと異なる SK |
| **作成タイミング** | いつでも追加/削除可 | テーブル作成時のみ |
| **最大数** | 20個 | 5個 |
| **一貫性** | 結果整合性のみ | 強い整合性も可 |
| **キャパシティ** | 独自の RCU/WCU | テーブルと共有 |
| **サイズ制限** | なし | パーティションあたり 10GB |
| **プロジェクション** | ALL / KEYS_ONLY / INCLUDE | ALL / KEYS_ONLY / INCLUDE |
| **空のキー値** | スパースインデックス対応 | スパースインデックス対応 |

### コード例 5: GSI オーバーロードパターン

```python
# 1つの GSI で複数のクエリに対応するオーバーロード
items = [
    # メールで検索
    {'PK': 'USER#001', 'SK': 'PROFILE',
     'GSI1PK': 'EMAIL#taro@example.com', 'GSI1SK': 'USER#001'},
    # ステータス+日付で検索
    {'PK': 'USER#001', 'SK': 'ORDER#001',
     'GSI1PK': 'STATUS#SHIPPED', 'GSI1SK': '2026-02-11T10:00:00Z'},
    # カテゴリ+価格で検索
    {'PK': 'PROD#001', 'SK': 'METADATA',
     'GSI1PK': 'CAT#electronics', 'GSI1SK': 'PRICE#000029900'},
]

# 配送済み注文を日付降順で取得
response = table.query(
    IndexName='GSI1',
    KeyConditionExpression=(
        Key('GSI1PK').eq('STATUS#SHIPPED') &
        Key('GSI1SK').between('2026-01-01', '2026-02-28')
    ),
    ScanIndexForward=False,
)
```

### コード例 5b: スパースインデックスの活用

```python
# スパースインデックス: GSI のキー属性を持つアイテムだけがインデックスに含まれる
# 例: 「フィーチャード商品」だけを GSI に登録

# フィーチャード商品（GSI2 のキーが存在するのでインデックスに含まれる）
table.put_item(Item={
    'PK': 'PROD#001',
    'SK': 'METADATA',
    'name': '高級ヘッドフォン',
    'price': 29900,
    'GSI2PK': 'FEATURED',           # この属性があるのでGSI2に含まれる
    'GSI2SK': 'PRICE#000029900',
})

# 通常商品（GSI2 のキーが存在しないのでインデックスに含まれない）
table.put_item(Item={
    'PK': 'PROD#002',
    'SK': 'METADATA',
    'name': '普通のイヤホン',
    'price': 3000,
    # GSI2PK/GSI2SK なし → スパースインデックスにより GSI2 に含まれない
})

# フィーチャード商品だけを価格順に取得（インデックスのスキャンが効率的）
response = table.query(
    IndexName='GSI2',
    KeyConditionExpression=Key('GSI2PK').eq('FEATURED'),
    ScanIndexForward=True,  # 価格昇順
)
```

---

## 4. キャパシティモード

### オンデマンド vs プロビジョンド比較表

| 観点 | オンデマンド | プロビジョンド |
|---|---|---|
| **課金方式** | リクエスト単位 | 予約容量 |
| **コスト（低負荷時）** | 安い | 高い（最低限の WCU/RCU） |
| **コスト（高負荷時）** | 高い（約5-7倍） | 安い |
| **スパイク対応** | 自動対応 | Auto Scaling 遅延あり |
| **予測可能性** | 低い | 高い |
| **推奨シーン** | 新規サービス、不定期アクセス | 安定トラフィック |
| **モード切替** | 24時間に1回切替可能 | 24時間に1回切替可能 |

```
コスト推移のイメージ
===========================

コスト
  ^
  |     オンデマンド
  |    /
  |   /
  |  /  . . . . . . . . プロビジョンド + AutoScaling
  | / .
  |/.     損益分岐点: 安定利用が25%以上ならプロビジョンド有利
  +-------------------------> リクエスト量
```

### コード例 6: Auto Scaling の設定

```bash
# プロビジョンドモードに切り替え
aws dynamodb update-table \
  --table-name MyApp \
  --billing-mode PROVISIONED \
  --provisioned-throughput ReadCapacityUnits=100,WriteCapacityUnits=50

# Auto Scaling ターゲットの登録（読み取り）
aws application-autoscaling register-scalable-target \
  --service-namespace dynamodb \
  --resource-id "table/MyApp" \
  --scalable-dimension "dynamodb:table:ReadCapacityUnits" \
  --min-capacity 5 \
  --max-capacity 500

# Auto Scaling ポリシーの設定（読み取り）
aws application-autoscaling put-scaling-policy \
  --service-namespace dynamodb \
  --resource-id "table/MyApp" \
  --scalable-dimension "dynamodb:table:ReadCapacityUnits" \
  --policy-name "MyApp-ReadAutoScaling" \
  --policy-type "TargetTrackingScaling" \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "DynamoDBReadCapacityUtilization"
    },
    "ScaleInCooldown": 60,
    "ScaleOutCooldown": 60
  }'

# Auto Scaling ターゲットの登録（書き込み）
aws application-autoscaling register-scalable-target \
  --service-namespace dynamodb \
  --resource-id "table/MyApp" \
  --scalable-dimension "dynamodb:table:WriteCapacityUnits" \
  --min-capacity 5 \
  --max-capacity 200

# Auto Scaling ポリシーの設定（書き込み）
aws application-autoscaling put-scaling-policy \
  --service-namespace dynamodb \
  --resource-id "table/MyApp" \
  --scalable-dimension "dynamodb:table:WriteCapacityUnits" \
  --policy-name "MyApp-WriteAutoScaling" \
  --policy-type "TargetTrackingScaling" \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "DynamoDBWriteCapacityUtilization"
    },
    "ScaleInCooldown": 60,
    "ScaleOutCooldown": 60
  }'
```

### コード例 6b: RCU/WCU の計算

```
RCU（読み取りキャパシティユニット）の計算:
===========================================

1 RCU = 1 回の強い整合性読み取り（最大 4KB）
      = 2 回の結果整合性読み取り（最大 4KB）
      = 0.5 回のトランザクション読み取り（最大 4KB）

例: 8KB のアイテムを毎秒 100 回、結果整合性で読み取る場合
  アイテムサイズ: 8KB → ceil(8/4) = 2 RCU/回
  結果整合性: 2 RCU / 2 = 1 RCU/回
  合計: 1 × 100 = 100 RCU

WCU（書き込みキャパシティユニット）の計算:
===========================================

1 WCU = 1 回の書き込み（最大 1KB）
      = 0.5 回のトランザクション書き込み（最大 1KB）

例: 3KB のアイテムを毎秒 50 回書き込む場合
  アイテムサイズ: 3KB → ceil(3/1) = 3 WCU/回
  合計: 3 × 50 = 150 WCU
```

---

## 5. DynamoDB Streams

### コード例 7: DynamoDB Streams + Lambda

```python
# Lambda ハンドラ: DynamoDB Streams からの変更イベント処理
import json
import boto3
from datetime import datetime, timezone

sns_client = boto3.client('sns')
sqs_client = boto3.client('sqs')

def handler(event, context):
    for record in event['Records']:
        event_name = record['eventName']  # INSERT, MODIFY, REMOVE

        if event_name == 'INSERT':
            new_image = record['dynamodb']['NewImage']
            pk = new_image['PK']['S']
            if pk.startswith('ORDER#'):
                send_order_notification(new_image)

        elif event_name == 'MODIFY':
            old_image = record['dynamodb']['OldImage']
            new_image = record['dynamodb']['NewImage']
            old_status = old_image.get('status', {}).get('S')
            new_status = new_image.get('status', {}).get('S')
            if old_status != new_status:
                handle_status_change(old_status, new_status, new_image)

        elif event_name == 'REMOVE':
            old_image = record['dynamodb']['OldImage']
            if record.get('userIdentity', {}).get('type') == 'Service':
                # TTL による自動削除
                handle_ttl_expiry(old_image)

    return {'statusCode': 200}


def send_order_notification(new_image):
    """新規注文の通知を SNS に送信"""
    order_id = new_image.get('SK', {}).get('S', '')
    user_id = new_image.get('PK', {}).get('S', '')
    total = new_image.get('total', {}).get('N', '0')

    sns_client.publish(
        TopicArn='arn:aws:sns:ap-northeast-1:123456789012:order-notifications',
        Subject=f'新規注文: {order_id}',
        Message=json.dumps({
            'order_id': order_id,
            'user_id': user_id,
            'total': total,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }),
    )


def handle_status_change(old_status, new_status, new_image):
    """注文ステータス変更時の処理"""
    if new_status == 'SHIPPED':
        # 配送通知を送信
        send_shipping_notification(new_image)
    elif new_status == 'DELIVERED':
        # 配達完了処理
        process_delivery_confirmation(new_image)
    elif new_status == 'CANCELLED':
        # キャンセル処理（在庫戻し等）
        process_cancellation(new_image)
```

### Stream の設定オプション

| StreamViewType | 含まれるデータ | ユースケース |
|---|---|---|
| `KEYS_ONLY` | キー属性のみ | 変更の検知のみ |
| `NEW_IMAGE` | 変更後のアイテム全体 | 集計・インデックス更新 |
| `OLD_IMAGE` | 変更前のアイテム全体 | 監査ログ |
| `NEW_AND_OLD_IMAGES` | 変更前後のアイテム全体 | 差分検出・監査ログ |

```bash
# DynamoDB Streams の有効化
aws dynamodb update-table \
  --table-name MyApp \
  --stream-specification StreamEnabled=true,StreamViewType=NEW_AND_OLD_IMAGES

# Stream ARN の取得
aws dynamodb describe-table \
  --table-name MyApp \
  --query 'Table.LatestStreamArn' \
  --output text

# Lambda の Event Source Mapping を作成
aws lambda create-event-source-mapping \
  --function-name process-dynamodb-stream \
  --event-source-arn arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyApp/stream/2026-01-01T00:00:00.000 \
  --batch-size 100 \
  --maximum-batching-window-in-seconds 5 \
  --starting-position LATEST \
  --maximum-retry-attempts 3 \
  --bisect-batch-on-function-error \
  --destination-config '{
    "OnFailure": {
      "Destination": "arn:aws:sqs:ap-northeast-1:123456789012:dynamodb-stream-dlq"
    }
  }'
```

---

## 6. DAX（DynamoDB Accelerator）

DAX は DynamoDB 専用のインメモリキャッシュで、マイクロ秒単位のレイテンシを実現する。

```
DAX アーキテクチャ:

  App --> DAX Cluster --> DynamoDB
          (< 0.1ms)      (< 10ms)

  DAX クラスター:
  +------------------+
  | Primary Node     |  ← 書き込み処理
  +------------------+
  | Read Replica 1   |  ← 読み取り分散
  +------------------+
  | Read Replica 2   |  ← 読み取り分散
  +------------------+

  キャッシュ:
  - Item Cache: GetItem/PutItem の結果をキャッシュ（デフォルト 5分）
  - Query Cache: Query/Scan の結果をキャッシュ（デフォルト 5分）
  - Write-Through: 書き込み時にキャッシュも更新
```

### コード例 8: DAX クライアント（Python）

```python
import amazondax
import boto3

# DAX クライアントの作成（DynamoDB SDK と互換性あり）
dax_client = amazondax.AmazonDaxClient(
    endpoints=['dax-cluster.abcdef.dax-clusters.ap-northeast-1.amazonaws.com:8111'],
    region_name='ap-northeast-1',
)
dax_resource = boto3.resource('dynamodb', region_name='ap-northeast-1')
# DAX 経由のテーブル操作
dax_table = dax_resource.Table('MyApp')

# 通常の DynamoDB SDK と同じインターフェースで利用可能
response = dax_table.get_item(
    Key={'PK': 'USER#001', 'SK': 'PROFILE'}
)
user = response.get('Item')

# DAX を使うか DynamoDB 直接かを切り替え可能にする設計
import os

def get_table():
    if os.environ.get('USE_DAX', 'false') == 'true':
        return dax_table
    else:
        return boto3.resource('dynamodb').Table('MyApp')
```

### DAX の制限事項

| 項目 | 制限 |
|---|---|
| 対応オペレーション | GetItem, Query, Scan, PutItem, UpdateItem, DeleteItem, BatchGetItem, BatchWriteItem |
| 非対応オペレーション | TransactWriteItems, TransactGetItems, CreateTable, UpdateTable |
| ネットワーク | VPC 内からのみアクセス可能 |
| 暗号化 | 転送中の暗号化対応、保管時は非対応（DynamoDB 側で暗号化） |
| 整合性 | 結果整合性のみ（強い整合性の読み取りは DAX をバイパス） |

---

## 7. バックアップとリストア

### オンデマンドバックアップ

```bash
# オンデマンドバックアップの作成
aws dynamodb create-backup \
  --table-name MyApp \
  --backup-name "MyApp-backup-$(date +%Y%m%d)"

# バックアップ一覧の取得
aws dynamodb list-backups \
  --table-name MyApp \
  --time-range-lower-bound 2026-01-01T00:00:00Z

# バックアップからのリストア（別テーブル名で復元）
aws dynamodb restore-table-from-backup \
  --target-table-name MyApp-restored \
  --backup-arn arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyApp/backup/01234567890123-abcdefgh
```

### PITR（ポイントインタイムリカバリ）

```bash
# PITR の有効化
aws dynamodb update-continuous-backups \
  --table-name MyApp \
  --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true

# PITR の状態確認
aws dynamodb describe-continuous-backups \
  --table-name MyApp

# 特定時点へのリストア（過去35日以内の任意の時点）
aws dynamodb restore-table-to-point-in-time \
  --source-table-name MyApp \
  --target-table-name MyApp-pitr-restore \
  --restore-date-time "2026-02-15T10:30:00Z"
```

---

## 8. グローバルテーブル

マルチリージョンでの Active-Active レプリケーションを提供する。

```
グローバルテーブルのアーキテクチャ:

  ap-northeast-1 (東京)          us-east-1 (バージニア)
  +------------------+          +------------------+
  | DynamoDB Table   | <------> | DynamoDB Table   |
  | (レプリカ)       |  双方向   | (レプリカ)       |
  +------------------+  レプリ  +------------------+
         ↑                ケー           ↑
         |                ション         |
  App (東京リージョン)           App (米国リージョン)

  特徴:
  - 1秒以内のレプリケーション遅延（通常）
  - 各リージョンで読み書き可能（Active-Active）
  - コンフリクト解決: Last Writer Wins（タイムスタンプベース）
  - 全リージョンで同一のテーブル構成が必要
```

### コード例 9: グローバルテーブルの設定

```bash
# 前提: ソーステーブルが ap-northeast-1 に存在

# レプリカの追加（us-east-1 にレプリケート）
aws dynamodb update-table \
  --table-name MyApp \
  --replica-updates '[{
    "Create": {
      "RegionName": "us-east-1"
    }
  }]' \
  --region ap-northeast-1

# レプリカの追加（eu-west-1 にもレプリケート）
aws dynamodb update-table \
  --table-name MyApp \
  --replica-updates '[{
    "Create": {
      "RegionName": "eu-west-1"
    }
  }]' \
  --region ap-northeast-1

# レプリケーション状態の確認
aws dynamodb describe-table \
  --table-name MyApp \
  --query 'Table.Replicas' \
  --output table

# レプリカの削除
aws dynamodb update-table \
  --table-name MyApp \
  --replica-updates '[{
    "Delete": {
      "RegionName": "eu-west-1"
    }
  }]' \
  --region ap-northeast-1
```

---

## 9. TTL（Time to Live）

### TTL の仕組みと活用

```
TTL の動作フロー:
=================

1. アイテムに ExpiresAt 属性（Unix エポック秒）を設定
2. DynamoDB が定期的にスキャン（通常 48 時間以内に削除）
3. 削除されたアイテムは Streams に REMOVE イベントとして記録
4. Streams の userIdentity.type が "Service" なら TTL 削除

活用パターン:
- セッションデータの自動クリーンアップ
- 一時的なトークン/OTP の管理
- ログ/監査データの自動アーカイブ
- キャッシュデータの自動失効
```

### コード例 10: TTL の実装

```python
import time
from datetime import datetime, timezone, timedelta

# セッションデータ（30分後に自動削除）
def create_session(session_id: str, user_id: str):
    expires_at = int(time.time()) + 1800  # 30分後
    table.put_item(Item={
        'PK': f'SESSION#{session_id}',
        'SK': 'DATA',
        'user_id': user_id,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'ExpiresAt': expires_at,
    })

# OTP（5分後に自動削除）
def create_otp(user_id: str, otp_code: str):
    expires_at = int(time.time()) + 300  # 5分後
    table.put_item(Item={
        'PK': f'OTP#{user_id}',
        'SK': f'CODE#{otp_code}',
        'ExpiresAt': expires_at,
    })

# 監査ログ（90日後に自動削除）
def write_audit_log(action: str, user_id: str, details: dict):
    expires_at = int(time.time()) + (90 * 24 * 3600)  # 90日後
    table.put_item(Item={
        'PK': f'AUDIT#{datetime.now(timezone.utc).strftime("%Y-%m-%d")}',
        'SK': f'{datetime.now(timezone.utc).isoformat()}#{user_id}',
        'action': action,
        'user_id': user_id,
        'details': details,
        'ExpiresAt': expires_at,
    })
```

---

## 10. DynamoDB Export to S3

大量データの分析には、DynamoDB Export to S3 を使用して Athena で分析する。

```bash
# S3 へのエクスポート（フルエクスポート）
aws dynamodb export-table-to-point-in-time \
  --table-arn arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyApp \
  --s3-bucket my-dynamodb-exports \
  --s3-prefix exports/myapp/ \
  --export-format DYNAMODB_JSON \
  --export-time "2026-02-15T00:00:00Z"

# インクリメンタルエクスポート
aws dynamodb export-table-to-point-in-time \
  --table-arn arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyApp \
  --s3-bucket my-dynamodb-exports \
  --s3-prefix exports/incremental/ \
  --export-format DYNAMODB_JSON \
  --export-type INCREMENTAL_EXPORT \
  --incremental-export-specification '{
    "ExportFromTime": "2026-02-14T00:00:00Z",
    "ExportToTime": "2026-02-15T00:00:00Z",
    "ExportViewType": "NEW_AND_OLD_IMAGES"
  }'

# エクスポート状態の確認
aws dynamodb describe-export \
  --export-arn arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyApp/export/01234567890123-abcdefgh
```

---

## 11. CloudWatch 監視

### 主要メトリクス一覧

| メトリクス | 説明 | アラーム閾値の目安 |
|---|---|---|
| ConsumedReadCapacityUnits | 消費された RCU | プロビジョンドの 80% |
| ConsumedWriteCapacityUnits | 消費された WCU | プロビジョンドの 80% |
| ThrottledRequests | スロットルされたリクエスト数 | 0 より大きい場合 |
| SystemErrors | サーバー側エラー数 | 0 より大きい場合 |
| UserErrors | クライアント側エラー数 | 急増時 |
| SuccessfulRequestLatency | リクエストレイテンシ | p99 が 20ms 超過 |
| ReplicationLatency | グローバルテーブルのレプリケーション遅延 | 1000ms 超過 |

### コード例 11: CloudWatch アラームの設定

```bash
# スロットリングアラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "DynamoDB-MyApp-Throttle" \
  --alarm-description "DynamoDB throttling detected" \
  --metric-name ThrottledRequests \
  --namespace AWS/DynamoDB \
  --statistic Sum \
  --period 60 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 1 \
  --dimensions Name=TableName,Value=MyApp \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts \
  --treat-missing-data notBreaching

# レイテンシアラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "DynamoDB-MyApp-Latency" \
  --alarm-description "DynamoDB high latency detected" \
  --metric-name SuccessfulRequestLatency \
  --namespace AWS/DynamoDB \
  --statistic p99 \
  --period 300 \
  --threshold 20 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --dimensions Name=TableName,Value=MyApp Name=Operation,Value=Query \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts
```

---

## アンチパターン

### 1. ホットパーティション

**問題**: 特定のパーティションキー（例: `STATUS#ACTIVE`）にアクセスが集中すると、スロットリングが発生する。DynamoDB はパーティション単位でスループットを分配するため、偏りは致命的。

```
[NG] ホットパーティション
PK = "STATUS#ACTIVE" --> 全アクティブユーザーがここに集中
  --> スロットリング発生

[OK] Write Sharding
PK = "STATUS#ACTIVE#3"  (0-9のサフィックスをランダム付与)
  --> 10パーティションに分散
  --> 読み取り時は10回のクエリを並列実行して統合
```

```python
import random

# Write Sharding の実装
SHARD_COUNT = 10

def write_with_sharding(status: str, item: dict):
    shard = random.randint(0, SHARD_COUNT - 1)
    item['GSI1PK'] = f'STATUS#{status}#{shard}'
    table.put_item(Item=item)

def query_with_sharding(status: str) -> list:
    """全シャードを並列クエリして結果を統合"""
    import concurrent.futures

    def query_shard(shard: int):
        response = table.query(
            IndexName='GSI1',
            KeyConditionExpression=Key('GSI1PK').eq(f'STATUS#{status}#{shard}'),
        )
        return response['Items']

    with concurrent.futures.ThreadPoolExecutor(max_workers=SHARD_COUNT) as executor:
        futures = [executor.submit(query_shard, i) for i in range(SHARD_COUNT)]
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())

    return results
```

### 2. Scan 操作の多用

**問題**: `Scan` はテーブル全体を読み取るため、コストが高く遅い。フィルタ式は読み取り後に適用されるため、RCU は節約できない。

**対策**: アクセスパターンを事前に洗い出し、GSI を設計してすべてのクエリを `Query` で実行できるようにする。どうしても `Scan` が必要な場合は、並列スキャン（`TotalSegments`）と `Limit` パラメータを組み合わせる。

```python
# [NG] フィルタ付き Scan（RCU が無駄に消費される）
response = table.scan(
    FilterExpression=Attr('status').eq('ACTIVE'),
)

# [OK] 並列 Scan（やむを得ない場合）
import concurrent.futures

def parallel_scan(total_segments: int = 4):
    def scan_segment(segment: int):
        items = []
        params = {
            'Segment': segment,
            'TotalSegments': total_segments,
            'FilterExpression': Attr('status').eq('ACTIVE'),
        }
        while True:
            response = table.scan(**params)
            items.extend(response['Items'])
            if 'LastEvaluatedKey' not in response:
                break
            params['ExclusiveStartKey'] = response['LastEvaluatedKey']
        return items

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_segments) as executor:
        futures = [executor.submit(scan_segment, i) for i in range(total_segments)]
        all_items = []
        for future in concurrent.futures.as_completed(futures):
            all_items.extend(future.result())

    return all_items
```

### 3. 大きすぎるアイテム

**問題**: DynamoDB のアイテムサイズ上限は 400KB。大きな JSON やバイナリデータを無理に格納しようとするとエラーになる。

**対策**: 大きなデータは S3 に格納し、DynamoDB にはメタデータと S3 キーを保存する。

```python
# [OK] 大きなデータを S3 に保存するパターン
import boto3
import json

s3 = boto3.client('s3')

def save_large_document(doc_id: str, content: str, metadata: dict):
    # 大きなコンテンツは S3 に保存
    s3.put_object(
        Bucket='my-documents-bucket',
        Key=f'documents/{doc_id}.json',
        Body=json.dumps({'content': content}),
        ContentType='application/json',
    )

    # メタデータを DynamoDB に保存
    table.put_item(Item={
        'PK': f'DOC#{doc_id}',
        'SK': 'METADATA',
        'title': metadata['title'],
        'author': metadata['author'],
        's3_key': f'documents/{doc_id}.json',
        's3_bucket': 'my-documents-bucket',
        'size_bytes': len(content),
        'created_at': datetime.now(timezone.utc).isoformat(),
    })
```

---

## FAQ

### Q1: シングルテーブル設計は常に正しいですか？

**A**: 必ずしもそうではありません。シングルテーブル設計はクエリ効率を最大化しますが、以下の場合は複数テーブルが適切です:
- エンティティ間のアクセスパターンが完全に独立している
- チームごとに異なるテーブルの権限管理が必要
- テーブルごとに異なるキャパシティ/バックアップ設定が必要
マイクロサービスではサービスごとに別テーブルが自然です。

### Q2: DynamoDB で集計クエリ（COUNT、SUM）を実行するには？

**A**: DynamoDB にはネイティブの集計機能がありません。以下の方法で対応します:
1. **集計用アイテムの維持**: 書き込み時にカウンタアイテムをアトミックに更新（`ADD` 操作）
2. **DynamoDB Streams + Lambda**: 変更をストリームで受け取り、集計テーブルに反映
3. **DynamoDB Export + S3 + Athena**: 定期エクスポートして Athena で分析

### Q3: RDS から DynamoDB に移行すべきタイミングは？

**A**: 以下の条件が揃う場合に検討してください:
- アクセスパターンが明確で、複雑な JOIN が不要
- ミリ秒単位のレイテンシが必要
- スケールが数万 RPS 以上に拡張する
- スキーマが頻繁に変更される
逆に、複雑なクエリ・トランザクション・レポーティングが中心なら RDS が適切です。

### Q4: DynamoDB のコストを最適化するには？

**A**: 以下のアプローチを組み合わせます:
1. **キャパシティモードの選択**: 安定トラフィックならプロビジョンド + Auto Scaling + リザーブドキャパシティで最大75%削減
2. **TTL の活用**: 不要データを自動削除してストレージコストを削減
3. **GSI の最適化**: 不要な GSI を削除し、プロジェクションタイプを `KEYS_ONLY` や `INCLUDE` に絞る
4. **アイテムサイズの最適化**: 属性名を短縮（例: `created_at` → `ca`）してストレージと RCU/WCU を節約
5. **結果整合性の活用**: 強い整合性が不要なら結果整合性で RCU を半減

### Q5: DynamoDB のセキュリティベストプラクティスは？

**A**:
1. **暗号化**: SSE（サーバーサイド暗号化）は必ず有効化。KMS キーは CMK（カスタマーマネージドキー）を推奨
2. **IAM ポリシー**: テーブルレベルだけでなく、Leading Key Condition で行レベルのアクセス制御を実装
3. **VPC エンドポイント**: パブリック IP を経由せず VPC エンドポイント経由でアクセス
4. **CloudTrail**: データプレーンの操作もログに記録
5. **バックアップ**: PITR を有効化し、オンデマンドバックアップも定期的に取得

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowAccessToOwnItems",
      "Effect": "Allow",
      "Action": [
        "dynamodb:GetItem",
        "dynamodb:PutItem",
        "dynamodb:UpdateItem",
        "dynamodb:DeleteItem",
        "dynamodb:Query"
      ],
      "Resource": "arn:aws:dynamodb:ap-northeast-1:123456789012:table/MyApp",
      "Condition": {
        "ForAllValues:StringEquals": {
          "dynamodb:LeadingKeys": ["USER#${aws:PrincipalTag/userId}"]
        }
      }
    }
  ]
}
```

### Q6: DynamoDB のテスト方法は？

**A**: ローカル開発には DynamoDB Local を使用します。

```bash
# DynamoDB Local の起動（Docker）
docker run -d -p 8000:8000 amazon/dynamodb-local

# ローカルでテーブル作成
aws dynamodb create-table \
  --table-name MyApp \
  --attribute-definitions AttributeName=PK,AttributeType=S AttributeName=SK,AttributeType=S \
  --key-schema AttributeName=PK,KeyType=HASH AttributeName=SK,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --endpoint-url http://localhost:8000
```

```python
# pytest でのテスト例
import pytest
import boto3

@pytest.fixture
def dynamodb_table():
    dynamodb = boto3.resource('dynamodb', endpoint_url='http://localhost:8000', region_name='ap-northeast-1')
    table = dynamodb.create_table(
        TableName='TestTable',
        KeySchema=[
            {'AttributeName': 'PK', 'KeyType': 'HASH'},
            {'AttributeName': 'SK', 'KeyType': 'RANGE'},
        ],
        AttributeDefinitions=[
            {'AttributeName': 'PK', 'AttributeType': 'S'},
            {'AttributeName': 'SK', 'AttributeType': 'S'},
        ],
        BillingMode='PAY_PER_REQUEST',
    )
    table.meta.client.get_waiter('table_exists').wait(TableName='TestTable')
    yield table
    table.delete()

def test_create_user(dynamodb_table):
    dynamodb_table.put_item(Item={
        'PK': 'USER#001',
        'SK': 'PROFILE',
        'name': 'Test User',
        'email': 'test@example.com',
    })
    response = dynamodb_table.get_item(Key={'PK': 'USER#001', 'SK': 'PROFILE'})
    assert response['Item']['name'] == 'Test User'
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| データモデル | PK + SK の複合キーでエンティティを表現。シングルテーブル設計が基本 |
| GSI | 異なるアクセスパターンに対応。オーバーロードで1つの GSI を多目的に活用 |
| LSI | 同一 PK 内の別ソート。テーブル作成時のみ定義可能 |
| キャパシティ | 新規/不定期はオンデマンド、安定利用はプロビジョンド + Auto Scaling |
| トランザクション | TransactWriteItems で最大100アイテムの ACID トランザクション |
| Streams | 変更データキャプチャ。Lambda と連携してイベント駆動処理 |
| TTL | 自動削除でコスト最適化。Streams と組み合わせてアーカイブ |
| DAX | マイクロ秒レイテンシのインメモリキャッシュ。VPC 内限定 |
| グローバルテーブル | マルチリージョン Active-Active。Last Writer Wins でコンフリクト解決 |
| バックアップ | PITR（35日間）+ オンデマンドバックアップ |
| セキュリティ | SSE + VPC エンドポイント + IAM 行レベル制御 + CloudTrail |

## 次に読むべきガイド

- [ElastiCache](./02-elasticache.md) — DynamoDB の前段キャッシュとして DAX と比較
- [RDS 基礎](./00-rds-basics.md) — リレーショナルデータベースとの使い分け
- [VPC 基礎](../04-networking/00-vpc-basics.md) — DynamoDB VPC エンドポイントの設定

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon DynamoDB 開発者ガイド](https://docs.aws.amazon.com/ja_jp/amazondynamodb/latest/developerguide/) — API リファレンスとベストプラクティス
2. **Alex DeBrie**: [The DynamoDB Book](https://www.dynamodbbook.com/) — シングルテーブル設計の決定版ガイド
3. **AWS re:Invent**: [Advanced Design Patterns for DynamoDB (DAT403)](https://www.youtube.com/watch?v=6yqfmXiZTlM) — Rick Houlihan による高度な設計パターン
4. **AWS ブログ**: [Best Practices for Designing and Architecting with DynamoDB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html) — 設計のベストプラクティス
5. **AWS Pricing**: [DynamoDB 料金](https://aws.amazon.com/dynamodb/pricing/) — オンデマンド/プロビジョンドの料金比較
