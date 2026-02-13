# NoSQL 比較

> MongoDB、Redis、DynamoDB の特性を比較し、RDB との使い分けとポリグロット永続化の設計戦略を実践的に習得する。本章ではNoSQLの理論的背景から各データベースの内部アーキテクチャ、データモデリングパターン、パフォーマンス特性までを掘り下げ、プロダクション環境でのハイブリッドアーキテクチャ設計を可能にする知識を提供する。

## 前提知識

- [01-schema-design.md](../02-design/01-schema-design.md) — RDBスキーマ設計の理解
- [02-joins.md](../00-basics/02-joins.md) — JOINの概念理解
- 分散システムの基礎概念（レプリケーション、パーティショニング）

## この章で学ぶこと

1. **NoSQL の分類と特性** — ドキュメント型、KVS、ワイドカラム、グラフの違い
2. **主要 NoSQL の比較** — MongoDB、Redis、DynamoDB の設計哲学と適用場面
3. **CAP定理と整合性モデル** — 分散システムにおけるトレードオフの理解
4. **ポリグロット永続化** — RDB と NoSQL を組み合わせたハイブリッドアーキテクチャ
5. **データモデリングの違い** — 正規化 vs 非正規化の設計判断
6. **移行判断フレームワーク** — RDBからNoSQLへの移行判断基準

---

## 1. NoSQL の分類

### NoSQLの歴史的背景

NoSQL（Not Only SQL）は2009年頃から普及した概念で、RDBの制約を克服するために生まれた。主な動機は以下の通り。

```
NoSQL 誕生の背景
==================

2000年代後半の課題:
  1. Web 2.0 のスケール要件
     - 数十億ユーザーのデータ
     - ペタバイト級のストレージ
     - ミリ秒以下のレイテンシ

  2. RDB のスケーリング限界
     - 垂直スケーリング（スケールアップ）にはハードウェア限界
     - 水平スケーリング（シャーディング）は複雑で JOIN に制約

  3. スキーマの柔軟性要求
     - アジャイル開発でのスキーマ変更頻度増加
     - 多様なデータ形式（JSON, 画像, 時系列等）

  4. 可用性要件の高まり
     - 24/7 稼働の要求
     - 地理分散レプリケーション

主要なイノベーション:
  2006: Google Bigtable 論文
  2007: Amazon Dynamo 論文
  2009: MongoDB 公開
  2010: Redis 1.0 リリース
  2012: DynamoDB サービス開始
```

### NoSQLデータベースの分類

```
NoSQL データベースの分類
==========================

+-------------------+  +-------------------+
| ドキュメント型     |  | Key-Value 型       |
| MongoDB, CouchDB  |  | Redis, Memcached   |
| Firestore         |  | Valkey, KeyDB      |
| --> JSON/BSON 文書 |  | --> 高速 KV         |
| --> 柔軟スキーマ   |  | --> キャッシュ       |
| --> リッチクエリ   |  | --> データ構造       |
+-------------------+  +-------------------+

+-------------------+  +-------------------+
| ワイドカラム型     |  | グラフ型           |
| DynamoDB,Cassandra |  | Neo4j, Neptune     |
| HBase, ScyllaDB   |  | ArangoDB, JanusGraph|
| --> 大規模分散     |  | --> 関係性探索       |
| --> 高書き込み     |  | --> SNS/推薦        |
| --> 設計が鍵       |  | --> パス検索        |
+-------------------+  +-------------------+

+-------------------+  +-------------------+
| 時系列型           |  | 検索エンジン       |
| TimescaleDB,       |  | Elasticsearch      |
| InfluxDB, QuestDB  |  | OpenSearch         |
| --> IoT/メトリクス |  | --> 全文検索        |
| --> 時間ベース集約 |  | --> ログ分析        |
+-------------------+  +-------------------+
```

### コード例 1: 各 NoSQL のデータモデル

```javascript
// === MongoDB（ドキュメント型）===
// 柔軟なスキーマで入れ子構造を自然に表現
db.users.insertOne({
  _id: ObjectId("..."),
  name: "Taro",
  email: "taro@example.com",
  address: {
    city: "Tokyo",
    zip: "100-0001",
    prefecture: "東京都"
  },
  orders: [
    { product: "Widget", amount: 1200, date: ISODate("2026-02-01") },
    { product: "Gadget", amount: 3400, date: ISODate("2026-02-10") }
  ],
  tags: ["premium", "early-adopter"],
  metadata: {
    loginCount: 42,
    lastLoginAt: ISODate("2026-02-13"),
    preferences: {
      theme: "dark",
      language: "ja",
      notifications: { email: true, push: false }
    }
  }
});

// ドキュメントの検索（リッチなクエリ言語）
db.users.find({
  "address.city": "Tokyo",
  "orders.amount": { $gt: 2000 },
  tags: { $in: ["premium"] }
}).sort({ "metadata.lastLoginAt": -1 });
```

```python
# === Redis（Key-Value 型）===
import redis
r = redis.Redis()

# シンプルな KV
r.set("user:1:name", "Taro")
r.get("user:1:name")  # → b"Taro"

# Hash（オブジェクト風）
r.hset("user:1", mapping={
    "name": "Taro",
    "email": "taro@example.com",
    "login_count": 42
})
r.hgetall("user:1")  # → {b"name": b"Taro", ...}

# Sorted Set でランキング
r.zadd("leaderboard", {"user:1": 1500, "user:2": 2300, "user:3": 800})
r.zrevrange("leaderboard", 0, 9, withscores=True)  # Top 10

# List でメッセージキュー
r.lpush("queue:emails", '{"to": "taro@example.com", "subject": "Welcome"}')
r.brpop("queue:emails", timeout=30)  # ブロッキング取得

# Set で集合演算
r.sadd("user:1:interests", "python", "sql", "redis")
r.sadd("user:2:interests", "python", "mongodb", "go")
r.sinter("user:1:interests", "user:2:interests")  # → {b"python"}

# Stream でイベントストリーミング（Redis 5.0+）
r.xadd("events:orders", {"action": "created", "order_id": "123"})
r.xread({"events:orders": "0"}, count=10)

# HyperLogLog でカーディナリティ推定
r.pfadd("daily_visitors:2026-02-13", "user:1", "user:2", "user:3")
r.pfcount("daily_visitors:2026-02-13")  # → 3（推定値）
```

```python
# === DynamoDB（ワイドカラム型）===
import boto3
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('MyApp')

# Single Table Design: 1テーブルに複数エンティティを格納
# PK（パーティションキー）+ SK（ソートキー）の複合キー

# ユーザープロフィール
table.put_item(Item={
    'PK': 'USER#001',
    'SK': 'PROFILE',
    'name': 'Taro',
    'email': 'taro@example.com',
    'created_at': '2026-01-01T00:00:00Z',
    'GSI1PK': 'USER',  # GSI用
    'GSI1SK': 'taro@example.com'
})

# ユーザーの注文
table.put_item(Item={
    'PK': 'USER#001',
    'SK': 'ORDER#2026-02-01#001',
    'product': 'Widget',
    'amount': Decimal('1200'),
    'status': 'shipped'
})

# クエリ: ユーザー001の全データ（プロフィール + 注文）
response = table.query(
    KeyConditionExpression='PK = :pk',
    ExpressionAttributeValues={':pk': 'USER#001'}
)

# クエリ: ユーザー001の2026年2月の注文のみ
response = table.query(
    KeyConditionExpression='PK = :pk AND begins_with(SK, :sk_prefix)',
    ExpressionAttributeValues={
        ':pk': 'USER#001',
        ':sk_prefix': 'ORDER#2026-02'
    }
)
```

```cypher
// === Neo4j（グラフ型）===
// Cypher クエリ言語

// ノードとリレーションの作成
CREATE (taro:User {name: "Taro", email: "taro@example.com"})
CREATE (hanako:User {name: "Hanako", email: "hanako@example.com"})
CREATE (python:Skill {name: "Python"})
CREATE (sql:Skill {name: "SQL"})
CREATE (taro)-[:KNOWS]->(hanako)
CREATE (taro)-[:HAS_SKILL {level: "expert"}]->(python)
CREATE (taro)-[:HAS_SKILL {level: "intermediate"}]->(sql)
CREATE (hanako)-[:HAS_SKILL {level: "expert"}]->(sql)

// 友達の友達（2ホップ探索）
MATCH (u:User {name: "Taro"})-[:KNOWS*2]->(fof:User)
WHERE fof <> u
RETURN DISTINCT fof.name

// 共通スキルを持つユーザーの推薦
MATCH (u:User {name: "Taro"})-[:HAS_SKILL]->(s:Skill)<-[:HAS_SKILL]-(other:User)
WHERE other <> u
RETURN other.name, collect(s.name) AS shared_skills, count(s) AS skill_count
ORDER BY skill_count DESC
```

---

## 2. 主要 NoSQL 比較

### 総合比較表

| 特性 | MongoDB | Redis | DynamoDB | Cassandra | Neo4j |
|---|---|---|---|---|---|
| **カテゴリ** | ドキュメント | KVS + データ構造 | ワイドカラム | ワイドカラム | グラフ |
| **データモデル** | JSON (BSON) | 文字列 + 高度なデータ構造 | アイテム (属性の集合) | 行 (カラムファミリー) | ノード + エッジ |
| **スキーマ** | 柔軟（スキーマレス） | なし | 柔軟（キーのみ固定） | 柔軟（カラム可変） | 柔軟 |
| **クエリ** | MQL (リッチ) | コマンドベース | Query/Scan (制限的) | CQL (SQL風) | Cypher (グラフ) |
| **トランザクション** | マルチドキュメント (4.0+) | MULTI/EXEC | TransactWriteItems | LWT (制限的) | ACID |
| **一貫性** | 設定可能 | 結果整合（Cluster） | 設定可能 | 設定可能 | 強一貫性 |
| **スケーリング** | シャーディング | Cluster | 自動（フルマネージド） | リングトポロジ | フェデレーション |
| **レイテンシ** | 1-10ms | < 1ms | 1-10ms | 1-10ms | 1-20ms |
| **永続化** | ディスク (WiredTiger) | メモリ + オプションで永続化 | ディスク（SSD） | ディスク (SSTable) | ディスク |
| **運用** | セルフ or Atlas | セルフ or ElastiCache | フルマネージド | セルフ or Astra | セルフ or AuraDB |
| **コスト特性** | ストレージベース | メモリベース（高額） | リクエストベース | ストレージベース | ライセンスベース |
| **最大データサイズ** | 実質無制限 | メモリ制約 | 400KB/item | 実質無制限 | 実質無制限 |

### 用途別推奨比較表

| ユースケース | 推奨 | 理由 |
|---|---|---|
| **Web アプリの主 DB** | PostgreSQL or MongoDB | 柔軟なクエリが必要 |
| **セッション管理** | Redis | 低レイテンシ、TTL サポート |
| **キャッシュ** | Redis | サブミリ秒応答 |
| **リアルタイムランキング** | Redis (Sorted Set) | O(log N) のスコア操作 |
| **IoT/時系列データ** | DynamoDB or TimescaleDB | 高書き込みスループット |
| **全文検索** | Elasticsearch | 転置インデックス |
| **ソーシャルグラフ** | Neo4j / Neptune | グラフ走査が高速 |
| **E コマースカタログ** | MongoDB | 商品ごとに異なる属性 |
| **サーバーレス API** | DynamoDB | フルマネージド、オートスケール |
| **メッセージブローカー** | Redis Streams / Kafka | 高スループット、低レイテンシ |
| **設定管理** | Redis / etcd | 高速読み取り、Pub/Sub |
| **コンテンツ管理** | MongoDB | 柔軟なスキーマ、リッチクエリ |
| **地理空間検索** | MongoDB / PostgreSQL+PostGIS | GeoJSON対応、空間インデックス |
| **推薦エンジン** | Neo4j + Redis | グラフ走査 + キャッシュ |
| **ログ/監査証跡** | Elasticsearch / DynamoDB | 高書き込み、検索対応 |

### 内部アーキテクチャの比較

```
MongoDB のアーキテクチャ
=========================

  クライアント → mongos (ルーター)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Shard 1      Shard 2      Shard 3
    ┌─────┐     ┌─────┐     ┌─────┐
    │ P   │     │ P   │     │ P   │
    │ S S │     │ S S │     │ S S │
    └─────┘     └─────┘     └─────┘
    (レプリカセット)

  P = Primary（書き込み）
  S = Secondary（読み取り/フェイルオーバー）

  ストレージエンジン: WiredTiger
  - B-Tree インデックス
  - ドキュメントレベルロック（同時実行性高い）
  - スナップショット分離（MVCC）
  - 圧縮（snappy, zlib, zstd）


Redis のアーキテクチャ
=======================

  クライアント → Redis Cluster
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
    Slot 0-5460  Slot 5461-10922 Slot 10923-16383
    ┌─────┐     ┌─────┐        ┌─────┐
    │ M   │     │ M   │        │ M   │
    │  R  │     │  R  │        │  R  │
    └─────┘     └─────┘        └─────┘

  M = Master
  R = Replica
  16384個のハッシュスロットにキーを分散

  メモリ管理:
  - 全データをメモリ上に保持
  - 永続化: RDB（スナップショット）/ AOF（ログ追記）
  - メモリ上限: maxmemory 設定
  - 退避ポリシー: allkeys-lru, volatile-ttl 等


DynamoDB のアーキテクチャ
=========================

  クライアント → DynamoDB エンドポイント
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   Partition 1  Partition 2  Partition 3
   (10GB max)   (10GB max)   (10GB max)
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Leader  │  │ Leader  │  │ Leader  │
   │ Replica │  │ Replica │  │ Replica │
   │ Replica │  │ Replica │  │ Replica │
   └─────────┘  └─────────┘  └─────────┘

  パーティションキーのハッシュ値で分散
  3つのAZに自動レプリケーション
  容量・スループットに応じて自動分割
```

---

## 3. CAP 定理と整合性モデル

### CAP 定理の正確な理解

```
CAP 定理
==========

        Consistency (一貫性)
           /\
          /  \
         /    \
   CA   / CP   \
  ------+------+------
  RDBMS | MongoDB*  |
  (単一) | HBase    |
        |          |
   AP   +----------+   CP
  Cassandra        |
  DynamoDB*     Redis Cluster
  CouchDB

* MongoDB, DynamoDB は設定により CA/CP/AP を選択可能

重要な誤解の訂正:
  CAP定理は「3つから2つを選ぶ」という単純な話ではない
  正確には:
  「ネットワーク分断（P）が発生した時に、
   一貫性（C）と可用性（A）のどちらを犠牲にするか」

  通常運用時（分断なし）:
  → 3つとも概ね達成可能
  → レイテンシ vs 一貫性のトレードオフ（PACELC定理）

PACELC 定理（CAPの拡張）:
  P（分断時）→ A or C を選択
  E（通常時）→ L（レイテンシ）or C（一貫性）を選択

  例:
  DynamoDB:  PA/EL（分断時は可用性、通常時は低レイテンシ優先）
  MongoDB:   PC/EC（一貫性優先だが結果整合も可能）
  Cassandra: PA/EL（可用性と低レイテンシ優先）
```

### 整合性レベルの比較

| 整合性レベル | 説明 | 例 | トレードオフ |
|---|---|---|---|
| 強一貫性（Strong） | 書き込み後すぐに最新値が読める | RDB、DynamoDB(ConsistentRead) | レイテンシ大、スループット低 |
| 線形化可能性（Linearizable） | 全操作がリアルタイム順序で見える | Spanner、CockroachDB | 最も厳密、最も遅い |
| 因果一貫性（Causal） | 因果関係のある操作の順序を保証 | MongoDB(causal sessions) | 強と結果の中間 |
| 結果整合性（Eventual） | 最終的にはすべてのレプリカが一致 | Cassandra(ONE)、S3 | 最も高速だが古いデータの可能性 |
| セッション一貫性 | 同一セッション内での一貫性を保証 | DynamoDB(default)、MongoDB | ユーザー体験に影響少ない |

### コード例 2: 一貫性レベルの設定

```javascript
// MongoDB: 読み取り一貫性の設定
// 強い整合性（プライマリから読み取り）
db.orders.find({ userId: "001" }).readConcern("majority");

// 結果整合性（セカンダリから読み取り、高速）
db.orders.find({ userId: "001" }).readPref("secondaryPreferred");

// 因果一貫性（セッション内）
const session = db.getMongo().startSession({ causalConsistency: true });
const orders = session.getDatabase("mydb").orders;
orders.insertOne({ userId: "001", product: "Widget" });
// 同じセッション内では必ず上記の挿入結果が見える
orders.find({ userId: "001" });
session.endSession();

// 書き込み確認レベル
db.orders.insertOne(
  { userId: "001", product: "Widget" },
  { writeConcern: { w: "majority", j: true, wtimeout: 5000 } }
);
// w: "majority" → 過半数のレプリカに書き込み完了を確認
// j: true → ジャーナルへの書き込みを確認
// wtimeout: 5000 → 5秒以内に完了しなければエラー
```

```python
# DynamoDB: 読み取り一貫性の設定
import boto3
table = boto3.resource('dynamodb').Table('MyApp')

# 強い整合性（RCU 2倍消費）
response = table.get_item(
    Key={'PK': 'USER#001', 'SK': 'PROFILE'},
    ConsistentRead=True
)

# 結果整合性（デフォルト、RCU 半分）
response = table.get_item(
    Key={'PK': 'USER#001', 'SK': 'PROFILE'},
    ConsistentRead=False
)

# DynamoDB トランザクション（ACID保証）
client = boto3.client('dynamodb')
client.transact_write_items(
    TransactItems=[
        {
            'Put': {
                'TableName': 'MyApp',
                'Item': {
                    'PK': {'S': 'ORDER#123'},
                    'SK': {'S': 'DETAIL'},
                    'status': {'S': 'confirmed'},
                    'amount': {'N': '5000'}
                },
                'ConditionExpression': 'attribute_not_exists(PK)'
            }
        },
        {
            'Update': {
                'TableName': 'MyApp',
                'Key': {
                    'PK': {'S': 'USER#001'},
                    'SK': {'S': 'PROFILE'}
                },
                'UpdateExpression': 'SET order_count = order_count + :inc',
                'ExpressionAttributeValues': {':inc': {'N': '1'}}
            }
        }
    ]
)
```

```python
# Cassandra: 一貫性レベルの設定
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement

cluster = Cluster(['node1', 'node2', 'node3'])
session = cluster.connect('mykeyspace')

# 強い整合性（QUORUM: 過半数のノードから応答）
statement = SimpleStatement(
    "SELECT * FROM orders WHERE user_id = %s",
    consistency_level=ConsistencyLevel.QUORUM
)
rows = session.execute(statement, ['user001'])

# 結果整合性（ONE: 1ノードから応答、最も高速）
statement = SimpleStatement(
    "SELECT * FROM orders WHERE user_id = %s",
    consistency_level=ConsistencyLevel.ONE
)

# ALL: 全ノードから応答（最も遅いが最も一貫性が高い）
statement = SimpleStatement(
    "SELECT * FROM orders WHERE user_id = %s",
    consistency_level=ConsistencyLevel.ALL
)
```

---

## 4. ポリグロット永続化

### コード例 3: ハイブリッドアーキテクチャ

```
ポリグロット永続化の設計例
============================

                   +------------------+
                   | Web Application  |
                   +--------+---------+
                            |
                   +--------+---------+
                   | API Gateway /    |
                   | Service Mesh     |
                   +--------+---------+
                            |
    +-------+-------+-------+-------+-------+
    |       |       |       |       |       |
    v       v       v       v       v       v
+------+ +------+ +-----+ +------+ +------+ +------+
|Postgre| |Mongo | |Redis| |Elastic| |DynamoDB| |Neo4j|
|SQL    | |DB    | |     | |Search | |       | |     |
+------+ +------+ +-----+ +------+ +------+ +------+
 ユーザー  商品     キャッシュ 検索     IoT     推薦
 注文     カタログ  セッション 全文検索  ログ    グラフ
 決済     レビュー  ランキング ログ分析  メトリクス 関係性
 在庫     CMS      Pub/Sub  ダッシュボード       SNS

データフロー:
  PostgreSQL ──(CDC)──> Elasticsearch (検索インデックス)
  PostgreSQL ──(CDC)──> Redis (キャッシュウォーミング)
  DynamoDB ──(Streams)──> Lambda ──> OpenSearch
  MongoDB ──(Change Streams)──> Kafka ──> 各サービス
```

```python
# ポリグロット永続化のサービス例
class OrderService:
    def __init__(self):
        self.pg = PostgresClient()       # トランザクション処理
        self.redis = RedisClient()       # キャッシュ
        self.mongo = MongoClient()       # 注文履歴（非正規化）
        self.es = ElasticsearchClient()  # 検索

    async def create_order(self, order_data):
        # 1. PostgreSQL でトランザクション処理（Source of Truth）
        async with self.pg.transaction() as tx:
            order = await tx.execute("""
                INSERT INTO orders (user_id, total, status)
                VALUES ($1, $2, 'pending')
                RETURNING id, created_at
            """, order_data['user_id'], order_data['total'])

            # 在庫の減算（同一トランザクション）
            for item in order_data['items']:
                await tx.execute("""
                    UPDATE products SET stock = stock - $1
                    WHERE id = $2 AND stock >= $1
                """, item['quantity'], item['product_id'])

        # 2. MongoDB に非正規化データを保存（高速読み取り用）
        await self.mongo.orders.insert_one({
            'order_id': order['id'],
            'user': order_data['user'],  # ユーザー情報を埋め込み
            'items': order_data['items'],  # 商品情報を埋め込み
            'total': order_data['total'],
            'status': 'pending',
            'created_at': order['created_at'],
        })

        # 3. Redis キャッシュを無効化
        await self.redis.delete(f"user:{order_data['user_id']}:orders")
        await self.redis.delete(f"user:{order_data['user_id']}:order_count")

        # 4. Elasticsearch にインデックス（非同期でも可）
        await self.es.index('orders', {
            'order_id': order['id'],
            'user_name': order_data['user']['name'],
            'items': [i['name'] for i in order_data['items']],
            'total': order_data['total'],
            'status': 'pending',
            'created_at': order['created_at'],
        })

        return order

    async def get_order_history(self, user_id, page=1, per_page=20):
        # キャッシュチェック
        cache_key = f"user:{user_id}:orders:page:{page}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # MongoDB から読み取り（非正規化データ、JOIN不要で高速）
        orders = await self.mongo.orders.find(
            {'user.id': user_id}
        ).sort('created_at', -1).skip((page - 1) * per_page).limit(per_page).to_list()

        # キャッシュに保存（5分間有効）
        await self.redis.setex(cache_key, 300, json.dumps(orders))

        return orders

    async def search_orders(self, query, filters=None):
        # Elasticsearch で全文検索
        body = {
            'query': {
                'bool': {
                    'must': [
                        {'multi_match': {
                            'query': query,
                            'fields': ['items', 'user_name']
                        }}
                    ]
                }
            }
        }
        if filters:
            body['query']['bool']['filter'] = filters

        return await self.es.search(index='orders', body=body)
```

### データ同期パターン

```
データ同期のパターン比較
==========================

1. Dual Write（二重書き込み）
   アプリ → DB1 に書き込み
        → DB2 に書き込み
   [問題] DB2への書き込みが失敗すると不整合
   [対策] 最終整合性を許容 + リトライ + 補正ジョブ

2. CDC（Change Data Capture）
   アプリ → DB1 に書き込み
   DB1 → CDC → DB2 に同期
   [利点] DB1が Source of Truth、DB2は派生
   [実装] Debezium, DynamoDB Streams, MongoDB Change Streams

3. Event Sourcing
   アプリ → イベントストア → 各DBに反映
   [利点] 完全な監査証跡、任意の時点に復元可能
   [欠点] 実装が複雑、最終整合性

4. CQRS（Command Query Responsibility Segregation）
   Write → PostgreSQL（正規化）
   Read  → MongoDB/Redis（非正規化、最適化済み）
   [利点] 読み書きを独立にスケーリング
   [欠点] 同期の遅延、複雑性

推奨パターン:
  小規模: Dual Write + 補正ジョブ
  中規模: CDC（Debezium）
  大規模: Event Sourcing + CQRS
```

---

## 5. RDB から NoSQL への移行判断

### 移行判断フレームワーク

```
RDB → NoSQL の移行判断チェックリスト
======================================

NoSQL を検討すべきシグナル:
  [?] JOIN が5テーブル以上で性能問題が発生
  [?] テーブルごとにカラム数や構造が大きく異なる
  [?] 書き込みスループットが垂直スケールの限界に到達
  [?] 地理分散（マルチリージョン）が必要
  [?] スキーマ変更が頻繁で運用負荷が高い
  [?] 読み取りパターンが限定的（PKアクセスが主）

RDB を維持すべきシグナル:
  [?] 複雑なアドホッククエリが必要（分析/BI）
  [?] 強い一貫性が必須（金融、在庫管理）
  [?] マルチテーブルACIDトランザクションが必須
  [?] データの関係性が複雑（多対多が多数）
  [?] スキーマによるデータ品質保証が重要
  [?] 既存のSQL知識を活用したい

判断フロー:
  1. まず PostgreSQL で対応できないか検討
  2. JSONB カラムで柔軟性を追加
  3. 読み取り最適化が必要 → Redis キャッシュ追加
  4. 特定ワークロードのみ NoSQL に切り出し
  5. フルリプレースは最終手段
```

### コード例 4: MongoDB のアグリゲーション

```javascript
// MongoDB のパイプライン集計（RDBのGROUP BY + JOIN相当）
db.orders.aggregate([
  // Stage 1: フィルタ（WHERE相当）
  { $match: {
    status: "shipped",
    createdAt: { $gte: ISODate("2026-01-01") }
  }},

  // Stage 2: 配列の展開（UNNEST相当）
  { $unwind: "$items" },

  // Stage 3: グループ集計（GROUP BY相当）
  { $group: {
      _id: "$items.category",
      totalRevenue: { $sum: "$items.price" },
      orderCount: { $sum: 1 },
      avgPrice: { $avg: "$items.price" },
      maxPrice: { $max: "$items.price" },
      uniqueProducts: { $addToSet: "$items.productId" }
  }},

  // Stage 4: 計算フィールドの追加
  { $addFields: {
    uniqueProductCount: { $size: "$uniqueProducts" },
    avgOrderValue: { $divide: ["$totalRevenue", "$orderCount"] }
  }},

  // Stage 5: ソート（ORDER BY相当）
  { $sort: { totalRevenue: -1 } },

  // Stage 6: リミット
  { $limit: 10 },

  // Stage 7: 結果の整形
  { $project: {
    category: "$_id",
    totalRevenue: { $round: ["$totalRevenue", 2] },
    orderCount: 1,
    avgPrice: { $round: ["$avgPrice", 2] },
    uniqueProductCount: 1,
    _id: 0
  }}
]);

// $lookup: JOIN相当（ただしNoSQLでは非推奨パターン）
db.orders.aggregate([
  { $lookup: {
    from: "users",
    localField: "userId",
    foreignField: "_id",
    as: "user"
  }},
  { $unwind: "$user" },
  { $project: {
    orderId: "$_id",
    userName: "$user.name",
    total: 1,
    status: 1
  }}
]);
// ※ $lookupはシャーディング環境でパフォーマンスが劣化する
// → 非正規化（embedding）で回避すべき
```

### コード例 5: RDB vs NoSQL のデータモデリング比較

```sql
-- RDB: 正規化モデル（4テーブル + JOIN）
-- 第3正規形: データの重複なし、更新異常なし
SELECT
    u.name,
    o.id AS order_id,
    o.order_date,
    p.name AS product_name,
    p.category,
    oi.quantity,
    oi.unit_price
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE u.id = 12345;
-- 4テーブル結合 → 複雑だが、データの整合性が保証される
```

```javascript
// MongoDB: 非正規化モデル（1ドキュメントで完結）
// データの重複はあるが、読み取りが高速
db.orders.findOne({
  userId: 12345
}, {
  "user.name": 1,
  "items.productName": 1,
  "items.category": 1,
  "items.quantity": 1,
  "items.unitPrice": 1,
  "orderDate": 1
});

// ドキュメント構造の例:
{
  _id: ObjectId("..."),
  userId: 12345,
  user: {               // ユーザー情報を埋め込み
    name: "田中太郎",
    email: "tanaka@example.com"
  },
  orderDate: ISODate("2026-02-01"),
  status: "shipped",
  items: [              // 商品情報を埋め込み
    {
      productId: 100,
      productName: "ノートPC",
      category: "家電",
      quantity: 1,
      unitPrice: 120000
    },
    {
      productId: 201,
      productName: "マウス",
      category: "周辺機器",
      quantity: 2,
      unitPrice: 3000
    }
  ],
  total: 126000,
  shippingAddress: {
    zip: "100-0001",
    city: "東京都千代田区"
  }
}
// 1回のクエリで全データ取得（JOIN不要）
// ただし、ユーザー名変更時に全注文ドキュメントを更新必要
```

### データモデリングのトレードオフ

```
RDB vs NoSQL のトレードオフ
==============================

RDB（正規化）:
  [+] データ整合性が高い（単一の真実の源）
  [+] 複雑なクエリ/集計が得意
  [+] スキーマでデータ品質を保証
  [+] ACIDトランザクション
  [+] アドホックなクエリが容易
  [-] JOIN が多いと性能劣化
  [-] スケールアウトが困難
  [-] スキーマ変更に手間がかかる

NoSQL（非正規化）:
  [+] 読み取りが高速（JOIN 不要）
  [+] スケールアウトが容易
  [+] スキーマの柔軟性
  [+] 地理分散レプリケーション
  [-] データ重複（更新が複雑）
  [-] 複雑なクエリが苦手
  [-] トランザクションに制約
  [-] データ整合性の保証が弱い

ハイブリッドアプローチ:
  PostgreSQL + JSONB:
  [+] リレーショナル + ドキュメントの良いとこ取り
  [+] 同一DBで両方のパターンを使い分け
  [+] ACIDトランザクション内でJSONBを操作可能
  [-] 大規模スケールアウトはNoSQLに劣る
```

### コード例 6: PostgreSQL JSONB vs MongoDB

```sql
-- PostgreSQL JSONB: RDBの枠内でドキュメント的な柔軟性を実現

-- テーブル定義（固定カラム + JSONB）
CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price    DECIMAL(10, 2) NOT NULL,
    attrs    JSONB DEFAULT '{}'  -- 可変属性
);

-- カテゴリごとに異なる属性を格納
INSERT INTO products (name, category, price, attrs) VALUES
('ノートPC', '家電', 120000, '{
    "brand": "Dell",
    "cpu": "Core i7",
    "ram_gb": 16,
    "storage": {"type": "SSD", "size_gb": 512},
    "ports": ["USB-C", "HDMI", "USB-A"]
}'),
('Tシャツ', 'アパレル', 3000, '{
    "brand": "Uniqlo",
    "size": "M",
    "color": "black",
    "material": "cotton"
}');

-- JSONB のクエリ（GINインデックスで高速）
CREATE INDEX idx_products_attrs ON products USING GIN (attrs);

-- 特定の属性値で検索
SELECT name, price, attrs->>'brand' AS brand
FROM products
WHERE attrs->>'cpu' = 'Core i7';

-- ネストした属性の検索
SELECT name, attrs->'storage'->>'type' AS storage_type
FROM products
WHERE (attrs->'storage'->>'size_gb')::int >= 256;

-- 配列内の値を検索
SELECT name FROM products
WHERE attrs->'ports' ? 'USB-C';

-- JSONB の集約
SELECT
    attrs->>'brand' AS brand,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price
FROM products
GROUP BY attrs->>'brand';
```

---

## 6. DynamoDB シングルテーブルデザイン

### コード例 7: DynamoDB のシングルテーブルデザイン

```
DynamoDB シングルテーブルデザイン
==================================

アクセスパターン:
  1. ユーザー情報の取得
  2. ユーザーの注文一覧
  3. 注文の詳細
  4. メールアドレスでユーザー検索

テーブル設計:
  PK             | SK                    | 属性
  ===============|=======================|============
  USER#001       | PROFILE               | name, email
  USER#001       | ORDER#2026-02-01#001  | total, status
  USER#001       | ORDER#2026-02-10#002  | total, status
  ORDER#001      | ITEM#001              | product, qty
  ORDER#001      | ITEM#002              | product, qty
  -----------------------------------------------
  GSI1PK         | GSI1SK                |
  USER           | taro@example.com      | (ユーザー検索用)
  ORDER          | 2026-02-01            | (日付順検索用)
```

```python
# DynamoDB シングルテーブルデザインの実装
import boto3
from datetime import datetime
from decimal import Decimal

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('MyApp')

class UserRepository:
    def create_user(self, user_id, name, email):
        """ユーザー作成"""
        table.put_item(Item={
            'PK': f'USER#{user_id}',
            'SK': 'PROFILE',
            'name': name,
            'email': email,
            'created_at': datetime.utcnow().isoformat(),
            'GSI1PK': 'USER',
            'GSI1SK': email  # メールで検索可能
        })

    def get_user(self, user_id):
        """ユーザー情報取得"""
        response = table.get_item(
            Key={'PK': f'USER#{user_id}', 'SK': 'PROFILE'}
        )
        return response.get('Item')

    def get_user_with_orders(self, user_id):
        """ユーザー情報 + 注文一覧（1クエリ）"""
        response = table.query(
            KeyConditionExpression='PK = :pk',
            ExpressionAttributeValues={':pk': f'USER#{user_id}'}
        )
        items = response['Items']
        user = next((i for i in items if i['SK'] == 'PROFILE'), None)
        orders = [i for i in items if i['SK'].startswith('ORDER#')]
        return {'user': user, 'orders': orders}

    def find_by_email(self, email):
        """メールアドレスでユーザー検索（GSI使用）"""
        response = table.query(
            IndexName='GSI1',
            KeyConditionExpression='GSI1PK = :pk AND GSI1SK = :email',
            ExpressionAttributeValues={
                ':pk': 'USER',
                ':email': email
            }
        )
        return response['Items']

class OrderRepository:
    def create_order(self, user_id, order_id, items, total):
        """注文作成（トランザクション使用）"""
        client = boto3.client('dynamodb')
        transact_items = [
            # 注文をユーザーのPKの下に作成
            {
                'Put': {
                    'TableName': 'MyApp',
                    'Item': {
                        'PK': {'S': f'USER#{user_id}'},
                        'SK': {'S': f'ORDER#{datetime.utcnow().strftime("%Y-%m-%d")}#{order_id}'},
                        'order_id': {'S': order_id},
                        'total': {'N': str(total)},
                        'status': {'S': 'pending'},
                        'GSI1PK': {'S': 'ORDER'},
                        'GSI1SK': {'S': datetime.utcnow().strftime("%Y-%m-%d")}
                    }
                }
            }
        ]
        # 各商品アイテムも追加
        for i, item in enumerate(items):
            transact_items.append({
                'Put': {
                    'TableName': 'MyApp',
                    'Item': {
                        'PK': {'S': f'ORDER#{order_id}'},
                        'SK': {'S': f'ITEM#{i:03d}'},
                        'product': {'S': item['name']},
                        'quantity': {'N': str(item['quantity'])},
                        'price': {'N': str(item['price'])}
                    }
                }
            })

        client.transact_write_items(TransactItems=transact_items)
```

---

## 7. Redis の高度なデータ構造

### コード例 8: Redis のデータ構造活用パターン

```python
import redis
import json
import time

r = redis.Redis()

# === セッション管理 ===
class SessionStore:
    def create_session(self, session_id, user_data, ttl=3600):
        """セッション作成（1時間有効）"""
        r.setex(
            f"session:{session_id}",
            ttl,
            json.dumps(user_data)
        )

    def get_session(self, session_id):
        """セッション取得"""
        data = r.get(f"session:{session_id}")
        if data:
            r.expire(f"session:{session_id}", 3600)  # TTLリフレッシュ
            return json.loads(data)
        return None

    def delete_session(self, session_id):
        """セッション削除"""
        r.delete(f"session:{session_id}")


# === レート制限 ===
class RateLimiter:
    def is_allowed(self, user_id, max_requests=100, window_seconds=60):
        """固定ウィンドウレート制限"""
        key = f"rate:{user_id}:{int(time.time()) // window_seconds}"
        current = r.incr(key)
        if current == 1:
            r.expire(key, window_seconds)
        return current <= max_requests

    def is_allowed_sliding(self, user_id, max_requests=100, window_seconds=60):
        """スライディングウィンドウレート制限"""
        key = f"rate_sliding:{user_id}"
        now = time.time()
        pipe = r.pipeline()
        pipe.zremrangebyscore(key, 0, now - window_seconds)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window_seconds)
        results = pipe.execute()
        return results[2] <= max_requests


# === 分散ロック ===
class DistributedLock:
    def acquire(self, lock_name, ttl=10):
        """ロック取得（Redlock簡易版）"""
        lock_key = f"lock:{lock_name}"
        token = str(time.time())
        acquired = r.set(lock_key, token, nx=True, ex=ttl)
        return token if acquired else None

    def release(self, lock_name, token):
        """ロック解放（Luaスクリプトでアトミックに）"""
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        r.eval(script, 1, f"lock:{lock_name}", token)


# === リアルタイムランキング ===
class Leaderboard:
    def update_score(self, board_name, user_id, score):
        """スコア更新"""
        r.zadd(f"leaderboard:{board_name}", {user_id: score})

    def get_rank(self, board_name, user_id):
        """順位取得（0-indexed）"""
        rank = r.zrevrank(f"leaderboard:{board_name}", user_id)
        return rank + 1 if rank is not None else None

    def get_top_n(self, board_name, n=10):
        """上位N位を取得"""
        return r.zrevrange(
            f"leaderboard:{board_name}", 0, n - 1,
            withscores=True
        )

    def get_around_me(self, board_name, user_id, range_size=5):
        """自分の前後N位を取得"""
        rank = r.zrevrank(f"leaderboard:{board_name}", user_id)
        if rank is None:
            return []
        start = max(0, rank - range_size)
        end = rank + range_size
        return r.zrevrange(
            f"leaderboard:{board_name}", start, end,
            withscores=True
        )
```

---

## エッジケース

### エッジケース1: ホットパーティション問題

```
DynamoDB ホットパーティション
==============================

問題:
  PK = "USER#popular_user" に大量のアクセスが集中
  → 1パーティションのスループット上限（3000 RCU / 1000 WCU）に到達
  → スロットリング発生

対策1: Write Sharding
  PK = "USER#popular_user#" + (hash % 10)
  → 10パーティションに分散
  → 読み取り時に10回クエリ + マージ

対策2: DAX（DynamoDB Accelerator）キャッシュ
  → 読み取りをインメモリキャッシュに吸収

対策3: アクセスパターンの見直し
  → カウンターは Redis に移行
  → 高頻度更新データは別テーブルに分離
```

### エッジケース2: MongoDB のドキュメントサイズ上限

```javascript
// MongoDB のドキュメントサイズ上限: 16MB
// 配列の無制限な成長はアンチパターン

// [NG] コメントを全て埋め込み → ドキュメント肥大化
db.posts.updateOne(
  { _id: postId },
  { $push: { comments: newComment } }
);
// → 人気投稿は数万コメントでドキュメントサイズ上限に到達

// [OK] バケットパターン（サブドキュメント分割）
db.post_comments.insertOne({
  postId: postId,
  bucket: Math.floor(commentCount / 100),  // 100件ごとにバケット
  comments: [newComment],
  count: 1
});

// [OK] 参照パターン（別コレクション）
db.comments.insertOne({
  postId: postId,
  userId: userId,
  text: "Great post!",
  createdAt: new Date()
});
// → 無制限に成長可能
```

### エッジケース3: Redis のメモリ枯渇

```python
# Redis メモリ管理のベストプラクティス

# 1. TTL の設定（全キーに推奨）
r.setex("cache:user:1", 300, user_json)  # 5分で期限切れ

# 2. メモリポリシーの設定
# redis.conf: maxmemory 4gb
# redis.conf: maxmemory-policy allkeys-lru

# 3. メモリ使用量の監視
info = r.info('memory')
print(f"使用メモリ: {info['used_memory_human']}")
print(f"ピークメモリ: {info['used_memory_peak_human']}")
print(f"フラグメンテーション比率: {info['mem_fragmentation_ratio']}")

# 4. 大きなキーの検出
# redis-cli --bigkeys
# redis-cli MEMORY USAGE key_name
```

---

## セキュリティに関する注意事項

### NoSQL インジェクション

```javascript
// MongoDB NoSQL インジェクション
// [NG] ユーザー入力を直接クエリに使用
const user = await db.users.findOne({
  username: req.body.username,
  password: req.body.password  // {"$gt": ""} が渡されると全ユーザーにマッチ
});

// [OK] 入力の型チェック + サニタイズ
const username = String(req.body.username);
const password = String(req.body.password);
const user = await db.users.findOne({
  username: username,
  password: password  // 文字列型が保証される
});

// [推奨] パスワードはハッシュ化
const user = await db.users.findOne({ username: username });
if (user && await bcrypt.compare(password, user.passwordHash)) {
  // 認証成功
}
```

### Redis セキュリティ

```
Redis セキュリティチェックリスト
==================================

1. 認証の設定
   requirepass strong_password_here
   # Redis 6.0+: ACL（アクセス制御リスト）
   user app_user on >password ~cache:* +get +set +del

2. ネットワーク制限
   bind 127.0.0.1  # ローカルのみ
   protected-mode yes

3. 危険なコマンドの無効化
   rename-command FLUSHALL ""
   rename-command FLUSHDB ""
   rename-command CONFIG ""
   rename-command DEBUG ""

4. TLS の有効化
   tls-port 6380
   tls-cert-file /path/to/cert.pem
   tls-key-file /path/to/key.pem

5. インターネット直接公開の禁止
   → VPC内またはプライベートネットワーク内に配置
```

---

## アンチパターン

### 1. NoSQL をリレーショナルに使う

**問題**: MongoDB で正規化設計を行い、複数コレクション間で `$lookup`（JOIN 相当）を多用する。NoSQL のメリットが活かせず、RDB より遅くなる。

**対策**: NoSQL ではアクセスパターンに最適化した非正規化モデルを設計する。1回のクエリで必要なデータがすべて取得できるようにドキュメントを構成する。

### 2. すべてを1つの DB で解決しようとする

**問題**: PostgreSQL で全文検索、キャッシュ、リアルタイム処理をすべて担わせると、各機能が中途半端になり、運用複雑性が増す。

**対策**: 各ワークロードに最適な DB を選択するポリグロット永続化を検討する。ただし、DB の数が増えると運用コストも増えるため、マイクロサービス境界に合わせて適切に分割する。

### 3. NoSQL でトランザクションを無視する

**問題**: NoSQL にはトランザクションがないと思い込み、データ整合性を考慮しない設計を行う。結果として不整合データが蓄積する。

**対策**: MongoDB 4.0+のマルチドキュメントトランザクション、DynamoDB のTransactWriteItemsなど、各NoSQLのトランザクション機能を理解して適切に使用する。トランザクションが不十分な場合は、冪等性 + リトライ + 補正ジョブで整合性を保つ。

### 4. DynamoDB でスキャンを多用する

**問題**: DynamoDBのScanは全データを読み取るため、テーブルが大きくなると性能とコストが劣化する。

**対策**: アクセスパターンを事前に定義し、パーティションキー + ソートキー + GSI で全てのクエリを効率的なQueryで実行できるようにテーブルを設計する。

---

## 演習問題

### 演習1（基礎）: データベース選定

以下の要件に対して、最適なデータベースとその理由を述べよ。

1. ECサイトの商品カタログ（カテゴリごとに異なる属性、柔軟な検索）
2. リアルタイムチャットアプリのメッセージ保存（高書き込み、時系列）
3. ソーシャルメディアの友達推薦機能（6次の隔たり探索）

<details>
<summary>解答例</summary>

1. **MongoDB** または **PostgreSQL + JSONB**: 商品カテゴリごとに異なる属性（ノートPCにはCPU/RAM、衣類にはサイズ/色）を柔軟に表現でき、リッチなクエリで検索可能。検索要件が高度なら Elasticsearch を追加。

2. **DynamoDB** または **Cassandra**: パーティションキー=チャットルームID、ソートキー=タイムスタンプで効率的に時系列データを管理。高書き込みスループットとスケーラビリティが強み。直近メッセージの高速取得にはRedisキャッシュを併用。

3. **Neo4j**: グラフDBはノード間の関係性探索に特化。「友達の友達の友達」のような多段階のパス探索が、RDBのJOIN連鎖より桁違いに高速。

</details>

### 演習2（応用）: ポリグロット永続化設計

ニュースサイトのアーキテクチャを設計せよ。要件:
- 記事の作成・編集（年間10万記事）
- 全文検索（タイトル + 本文）
- ユーザーの閲覧履歴（日次1億PV）
- リアルタイム人気ランキング
- コメント機能

<details>
<summary>解答例</summary>

```
データベース設計:
  PostgreSQL: 記事、ユーザー、コメント（Source of Truth）
  Elasticsearch: 全文検索インデックス（記事タイトル + 本文）
  Redis: 人気ランキング（Sorted Set）、セッション、記事キャッシュ
  DynamoDB: 閲覧履歴（高書き込みスループット）

データフロー:
  記事作成 → PostgreSQL → CDC → Elasticsearch（検索用）
                              → Redis（キャッシュ）
  閲覧 → DynamoDB（履歴記録）→ Redis ZINCRBY（ランキング更新）
  検索 → Elasticsearch
  ランキング → Redis ZREVRANGE
  コメント → PostgreSQL（ACIDトランザクション）
```

</details>

### 演習3（発展）: 移行計画

PostgreSQL で運用中のECサイト（1000万ユーザー、1億注文）の商品カタログ部分をMongoDBに移行する計画を立案せよ。以下を含めること。
- データ移行手順
- 移行期間中のデータ同期方式
- ロールバック計画
- リスクと対策

<details>
<summary>解答例</summary>

```
移行計画:
1. Phase 1（準備）: 2週間
   - MongoDBクラスタ構築
   - データモデル設計（非正規化）
   - 移行スクリプト開発・テスト

2. Phase 2（並行運用）: 4週間
   - 初期データ移行（pg_dump → 変換 → mongoimport）
   - Dual Write: アプリがPostgreSQL + MongoDBの両方に書き込み
   - 読み取りはPostgreSQLのまま
   - データ整合性チェックジョブ実行

3. Phase 3（切替）: 1週間
   - 読み取りをMongoDBに切替（フィーチャーフラグ）
   - 段階的にトラフィック移行（10% → 50% → 100%）
   - パフォーマンス監視

4. Phase 4（完了）: 2週間
   - PostgreSQLからの商品カタログテーブル削除
   - Dual Write停止

ロールバック計画:
   - Phase 2-3: フィーチャーフラグでPostgreSQLに即座に切り戻し
   - Phase 4後: MongoDBからPostgreSQLへの逆移行（最悪ケース）

リスク:
   - データ不整合: 定期的な整合性チェックで検出
   - パフォーマンス劣化: カナリアデプロイで段階的に検証
   - スキルギャップ: チームへのMongoDB研修
```

</details>

---

## FAQ

### Q1: MongoDB と PostgreSQL の JSONB はどう使い分けますか？

**A**:
- **PostgreSQL JSONB**: リレーショナルデータが主体で、一部の属性が柔軟。トランザクション・JOIN が必要。既に PostgreSQL を使用中。テーブルの80%以上がリレーショナルなデータの場合。
- **MongoDB**: ドキュメントが主体。スキーマが頻繁に変更。水平スケーリングが必要。深い入れ子構造が多い。アクセスパターンがドキュメント単位に集中する場合。

### Q2: Redis を主データベースとして使えますか？

**A**: 技術的には可能ですが推奨しません。Redis はインメモリ DB のため、データ量がメモリに制約され、コストが高くなります。AOF/RDB で永続化できますが、再起動時のデータ読み込みに時間がかかります。Redis 7.0+ の Redis Functions やRedis Stackで機能は充実していますが、キャッシュ・セッション・リアルタイム処理の補助として使い、主データは RDB や DynamoDB に保存する設計が一般的です。

### Q3: DynamoDB で複雑な検索が必要になった場合どうしますか？

**A**: DynamoDB Streams で変更を Elasticsearch/OpenSearch に同期し、全文検索や複雑な集計はそちらで実行します。あるいは DynamoDB Export to S3 + Athena でアドホック分析を行います。DynamoDB 自体で複雑なクエリを実行しようとするのはアンチパターンです。

### Q4: NoSQL を選んで後悔するケースは？

**A**: 以下のケースで後悔が報告されています:
- **要件が複雑になった**: 当初はシンプルなKVアクセスだったが、複雑な集計やJOINが必要になった → RDBに移行
- **トランザクションが必要になった**: 決済や在庫管理でACIDが必要になった → RDBに部分移行
- **運用コストの増大**: 複数DBの運用・監視・バックアップのコストが想定以上

### Q5: NoSQLのスキーマレスは本当にスキーマがないのか？

**A**: 「スキーマレス」は正確ではなく、「スキーマオンリード（読み取り時スキーマ）」が正しい表現です。データベース側はスキーマを強制しませんが、アプリケーション側では暗黙のスキーマが存在します。MongoDBのSchema Validationなどでサーバー側でのスキーマ検証も可能です。スキーマの管理はアプリケーション側の責任となるため、十分な設計が必要です。

---

## トラブルシューティング

### 問題1: MongoDB のスロークエリ

```javascript
// スロークエリの調査
db.setProfilingLevel(1, { slowms: 100 });  // 100ms以上のクエリを記録
db.system.profile.find().sort({ ts: -1 }).limit(5);

// 実行計画の確認
db.orders.find({ userId: "001" }).explain("executionStats");
// COLLSCAN → インデックスが使われていない

// インデックスの作成
db.orders.createIndex({ userId: 1, createdAt: -1 });
```

### 問題2: Redis の接続数枯渇

```python
# 接続プーリングの設定
pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=50,  # 接続プールサイズ
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True
)
r = redis.Redis(connection_pool=pool)
```

### 問題3: DynamoDB のスロットリング

```python
# 指数バックオフ付きリトライ
import time
from botocore.exceptions import ClientError

def put_with_retry(item, max_retries=5):
    for attempt in range(max_retries):
        try:
            table.put_item(Item=item)
            return
        except ClientError as e:
            if e.response['Error']['Code'] == 'ProvisionedThroughputExceededException':
                wait_time = (2 ** attempt) * 0.1  # 0.1, 0.2, 0.4, 0.8, 1.6秒
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| NoSQL の分類 | ドキュメント、KVS、ワイドカラム、グラフの4種類 + 時系列、検索 |
| MongoDB | 柔軟なスキーマのドキュメント DB。リッチなクエリ対応 |
| Redis | サブミリ秒 KVS。キャッシュ・セッション・リアルタイム処理 |
| DynamoDB | フルマネージドのワイドカラム DB。サーバーレスと相性良好 |
| CAP 定理 | 分断時の一貫性 vs 可用性のトレードオフ。PACELC定理も理解すべき |
| ポリグロット | ワークロードに応じて最適な DB を組み合わせる |
| 移行判断 | JOIN 不要 + スケール必要 + 柔軟スキーマ なら NoSQL を検討 |
| PostgreSQL JSONB | RDB内でドキュメント的柔軟性を実現する中間解 |
| セキュリティ | NoSQLインジェクション対策、認証・暗号化の設定 |

## 次に読むべきガイド

- [インデックス](../01-advanced/03-indexing.md) — RDB のインデックス最適化
- [マイグレーション](../02-design/02-migration.md) — スキーマ変更の安全な手法
- [スキーマ設計](../02-design/01-schema-design.md) — RDBスキーマ設計の基礎

## 参考文献

1. **Martin Kleppmann**: [Designing Data-Intensive Applications](https://dataintensive.net/) — データシステム設計の名著
2. **MongoDB 公式**: [MongoDB Manual](https://www.mongodb.com/docs/manual/) — MongoDB の包括的ドキュメント
3. **AWS 公式**: [DynamoDB Best Practices](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/best-practices.html) — DynamoDB設計ガイド
4. **Redis 公式**: [Redis Documentation](https://redis.io/docs/) — Redisデータ構造とコマンドリファレンス
5. **Brewer, Eric**: "CAP Twelve Years Later" (2012) — CAP定理の正確な解釈
6. **Rick Houlihan**: [DynamoDB Single Table Design](https://www.alexdebrie.com/posts/dynamodb-single-table/) — シングルテーブルデザインの解説
