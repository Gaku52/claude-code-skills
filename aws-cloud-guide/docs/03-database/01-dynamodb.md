# Amazon DynamoDB

> AWS のフルマネージド NoSQL データベースを理解し、テーブル設計・GSI/LSI・キャパシティモードを実践的に習得する

## この章で学ぶこと

1. **DynamoDB のデータモデル** — パーティションキー、ソートキー、アイテム構造の設計原則
2. **セカンダリインデックス** — GSI（グローバル）と LSI（ローカル）の使い分けとクエリパターン
3. **キャパシティモードと運用** — オンデマンド vs プロビジョンド、DAX キャッシュ、TTL 管理

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

### コード例 3: CRUD 操作（Python / boto3）

```python
import boto3
from datetime import datetime, timezone
from boto3.dynamodb.conditions import Key, Attr

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
```

### コード例 4: トランザクション操作

```python
def create_order_with_stock_update(user_id, order_id, product_id, qty, total):
    """注文作成 + 在庫減少をアトミックに実行"""
    client = boto3.client('dynamodb', region_name='ap-northeast-1')

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

### コード例 6: DynamoDB Streams + Lambda

```python
# Lambda ハンドラ: DynamoDB Streams からの変更イベント処理
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

### 2. Scan 操作の多用

**問題**: `Scan` はテーブル全体を読み取るため、コストが高く遅い。フィルタ式は読み取り後に適用されるため、RCU は節約できない。

**対策**: アクセスパターンを事前に洗い出し、GSI を設計してすべてのクエリを `Query` で実行できるようにする。どうしても `Scan` が必要な場合は、並列スキャン（`TotalSegments`）と `Limit` パラメータを組み合わせる。

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

## 次に読むべきガイド

- [ElastiCache](./02-elasticache.md) — DynamoDB の前段キャッシュとして DAX と比較
- [RDS 基礎](./00-rds-basics.md) — リレーショナルデータベースとの使い分け
- [VPC 基礎](../04-networking/00-vpc-basics.md) — DynamoDB VPC エンドポイントの設定

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon DynamoDB 開発者ガイド](https://docs.aws.amazon.com/ja_jp/amazondynamodb/latest/developerguide/) — API リファレンスとベストプラクティス
2. **Alex DeBrie**: [The DynamoDB Book](https://www.dynamodbbook.com/) — シングルテーブル設計の決定版ガイド
3. **AWS re:Invent**: [Advanced Design Patterns for DynamoDB (DAT403)](https://www.youtube.com/watch?v=6yqfmXiZTlM) — Rick Houlihan による高度な設計パターン
