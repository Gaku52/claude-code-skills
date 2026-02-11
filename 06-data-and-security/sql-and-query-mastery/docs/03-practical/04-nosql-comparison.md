# NoSQL 比較

> MongoDB、Redis、DynamoDB の特性を比較し、RDB との使い分けとポリグロット永続化の設計戦略を実践的に習得する

## この章で学ぶこと

1. **NoSQL の分類と特性** — ドキュメント型、KVS、ワイドカラム、グラフの違い
2. **主要 NoSQL の比較** — MongoDB、Redis、DynamoDB の設計哲学と適用場面
3. **ポリグロット永続化** — RDB と NoSQL を組み合わせたハイブリッドアーキテクチャ

---

## 1. NoSQL の分類

```
NoSQL データベースの分類
==========================

+------------------+  +------------------+
| ドキュメント型   |  | Key-Value 型     |
| MongoDB, CouchDB |  | Redis, Memcached |
| --> JSON 文書    |  | --> 高速 KV     |
| --> 柔軟スキーマ |  | --> キャッシュ   |
+------------------+  +------------------+

+------------------+  +------------------+
| ワイドカラム型   |  | グラフ型         |
| DynamoDB, Cassandra | Neo4j, Neptune  |
| --> 大規模分散   |  | --> 関係性探索   |
| --> 高書き込み   |  | --> SNS/推薦     |
+------------------+  +------------------+
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
    zip: "100-0001"
  },
  orders: [
    { product: "Widget", amount: 1200, date: ISODate("2026-02-01") },
    { product: "Gadget", amount: 3400, date: ISODate("2026-02-10") }
  ],
  tags: ["premium", "early-adopter"]
});
```

```python
# === Redis（Key-Value 型）===
import redis
r = redis.Redis()

# シンプルな KV
r.set("user:1:name", "Taro")
r.hset("user:1", mapping={"name": "Taro", "email": "taro@example.com"})

# Sorted Set でランキング
r.zadd("leaderboard", {"user:1": 1500, "user:2": 2300})
```

```python
# === DynamoDB（ワイドカラム型）===
import boto3
table = boto3.resource('dynamodb').Table('MyApp')

table.put_item(Item={
    'PK': 'USER#001',
    'SK': 'PROFILE',
    'name': 'Taro',
    'email': 'taro@example.com',
})
```

---

## 2. 主要 NoSQL 比較

### 総合比較表

| 特性 | MongoDB | Redis | DynamoDB |
|---|---|---|---|
| **カテゴリ** | ドキュメント | KVS + データ構造 | ワイドカラム |
| **データモデル** | JSON (BSON) | 文字列 + 高度なデータ構造 | アイテム (属性の集合) |
| **スキーマ** | 柔軟（スキーマレス） | なし | 柔軟（キーのみ固定） |
| **クエリ** | MQL (リッチ) | コマンドベース | Query/Scan (制限的) |
| **一貫性** | 設定可能 | 結果整合（Cluster） | 設定可能 |
| **スケーリング** | シャーディング | Cluster | 自動（フルマネージド） |
| **レイテンシ** | 1-10ms | < 1ms | 1-10ms |
| **永続化** | ディスク | メモリ + オプションで永続化 | ディスク（SSD） |
| **運用** | セルフ or Atlas | セルフ or ElastiCache | フルマネージド |
| **コスト特性** | ストレージベース | メモリベース（高額） | リクエストベース |

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

---

## 3. CAP 定理と整合性モデル

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

注意: CAP 定理はネットワーク分断時の2択
  - CP: 分断時に一貫性を優先（一部リクエスト拒否）
  - AP: 分断時に可用性を優先（古いデータを返す可能性）
  通常運用では3つとも概ね達成可能
```

### コード例 2: 一貫性レベルの設定

```javascript
// MongoDB: 読み取り一貫性の設定
// 強い整合性（プライマリから読み取り）
db.orders.find({ userId: "001" }).readConcern("majority");

// 結果整合性（セカンダリから読み取り、高速）
db.orders.find({ userId: "001" }).readPref("secondaryPreferred");
```

```python
# DynamoDB: 読み取り一貫性の設定
# 強い整合性
table.get_item(
    Key={'PK': 'USER#001', 'SK': 'PROFILE'},
    ConsistentRead=True
)

# 結果整合性（デフォルト、RCU 半分）
table.get_item(
    Key={'PK': 'USER#001', 'SK': 'PROFILE'},
    ConsistentRead=False
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
    +----+----+----+----+----+
    |         |         |    |
    v         v         v    v
+------+  +------+  +----+ +--------+
|Postgre| |MongoDB| |Redis| |Elastic |
|SQL    | |      | |     | |Search  |
+------+  +------+  +----+ +--------+
 ユーザー  商品       キャッシュ  検索
 注文     カタログ   セッション  全文検索
 決済     レビュー   ランキング  ログ分析
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
        # 1. PostgreSQL でトランザクション処理
        order = await self.pg.execute("""
            INSERT INTO orders (user_id, total, status)
            VALUES ($1, $2, 'pending')
            RETURNING id
        """, order_data['user_id'], order_data['total'])

        # 2. MongoDB に非正規化データを保存（高速読み取り用）
        await self.mongo.orders.insert_one({
            'order_id': order['id'],
            'user': order_data['user'],  # ユーザー情報を埋め込み
            'items': order_data['items'],  # 商品情報を埋め込み
            'total': order_data['total'],
            'status': 'pending',
        })

        # 3. Redis キャッシュを無効化
        await self.redis.delete(f"user:{order_data['user_id']}:orders")

        # 4. Elasticsearch にインデックス
        await self.es.index('orders', {
            'order_id': order['id'],
            'user_name': order_data['user']['name'],
            'items': [i['name'] for i in order_data['items']],
            'total': order_data['total'],
            'created_at': datetime.utcnow(),
        })

        return order
```

---

## 5. RDB から NoSQL への移行判断

### コード例 4: MongoDB のアグリゲーション

```javascript
// MongoDB のパイプライン集計
db.orders.aggregate([
  { $match: { status: "shipped", createdAt: { $gte: ISODate("2026-01-01") } } },
  { $unwind: "$items" },
  { $group: {
      _id: "$items.category",
      totalRevenue: { $sum: "$items.price" },
      orderCount: { $sum: 1 },
      avgPrice: { $avg: "$items.price" }
  }},
  { $sort: { totalRevenue: -1 } },
  { $limit: 10 }
]);
```

### コード例 5: RDB vs NoSQL のデータモデリング比較

```sql
-- RDB: 正規化モデル（3テーブル + JOIN）
SELECT u.name, o.id AS order_id, p.name AS product_name, oi.quantity
FROM users u
JOIN orders o ON u.id = o.user_id
JOIN order_items oi ON o.id = oi.order_id
JOIN products p ON oi.product_id = p.id
WHERE u.id = 12345;
```

```javascript
// MongoDB: 非正規化モデル（1ドキュメントで完結）
db.orders.findOne({
  userId: 12345
}, {
  "user.name": 1,
  "items.productName": 1,
  "items.quantity": 1
});

// 1回のクエリで全データ取得（JOIN 不要）
```

```
RDB vs NoSQL のトレードオフ
==============================

RDB（正規化）:
  [+] データ整合性が高い
  [+] 複雑なクエリ/集計が得意
  [+] スキーマでデータ品質を保証
  [-] JOIN が多いと性能劣化
  [-] スケールアウトが困難

NoSQL（非正規化）:
  [+] 読み取りが高速（JOIN 不要）
  [+] スケールアウトが容易
  [+] スキーマの柔軟性
  [-] データ重複（更新が複雑）
  [-] 複雑なクエリが苦手
  [-] トランザクションに制約
```

---

## アンチパターン

### 1. NoSQL をリレーショナルに使う

**問題**: MongoDB で正規化設計を行い、複数コレクション間で `$lookup`（JOIN 相当）を多用する。NoSQL のメリットが活かせず、RDB より遅くなる。

**対策**: NoSQL ではアクセスパターンに最適化した非正規化モデルを設計する。1回のクエリで必要なデータがすべて取得できるようにドキュメントを構成する。

### 2. すべてを1つの DB で解決しようとする

**問題**: PostgreSQL で全文検索、キャッシュ、リアルタイム処理をすべて担わせると、各機能が中途半端になり、運用複雑性が増す。

**対策**: 各ワークロードに最適な DB を選択するポリグロット永続化を検討する。ただし、DB の数が増えると運用コストも増えるため、マイクロサービス境界に合わせて適切に分割する。

---

## FAQ

### Q1: MongoDB と PostgreSQL の JSONB はどう使い分けますか？

**A**:
- **PostgreSQL JSONB**: リレーショナルデータが主体で、一部の属性が柔軟。トランザクション・JOIN が必要。既に PostgreSQL を使用中
- **MongoDB**: ドキュメントが主体。スキーマが頻繁に変更。水平スケーリングが必要。深い入れ子構造が多い
データの80%以上がリレーショナルなら PostgreSQL + JSONB、80%以上がドキュメントなら MongoDB が適切です。

### Q2: Redis を主データベースとして使えますか？

**A**: 技術的には可能ですが推奨しません。Redis はインメモリ DB のため、データ量がメモリに制約され、コストが高くなります。AOF/RDB で永続化できますが、再起動時のデータ読み込みに時間がかかります。キャッシュ・セッション・リアルタイム処理の補助として使い、主データは RDB や DynamoDB に保存する設計が一般的です。

### Q3: DynamoDB で複雑な検索が必要になった場合どうしますか？

**A**: DynamoDB Streams で変更を Elasticsearch/OpenSearch に同期し、全文検索や複雑な集計はそちらで実行します。あるいは DynamoDB Export to S3 + Athena でアドホック分析を行います。DynamoDB 自体で複雑なクエリを実行しようとするのはアンチパターンです。

---

## まとめ

| 項目 | 要点 |
|---|---|
| NoSQL の分類 | ドキュメント、KVS、ワイドカラム、グラフの4種類 |
| MongoDB | 柔軟なスキーマのドキュメント DB。リッチなクエリ対応 |
| Redis | サブミリ秒 KVS。キャッシュ・セッション・リアルタイム処理 |
| DynamoDB | フルマネージドのワイドカラム DB。サーバーレスと相性良好 |
| CAP 定理 | 分断時の一貫性 vs 可用性のトレードオフ |
| ポリグロット | ワークロードに応じて最適な DB を組み合わせる |
| 移行判断 | JOIN 不要 + スケール必要 + 柔軟スキーマ なら NoSQL を検討 |

## 次に読むべきガイド

- [インデックス](../01-advanced/03-indexing.md) — RDB のインデックス最適化
- [マイグレーション](../02-design/02-migration.md) — スキーマ変更の安全な手法

## 参考文献

1. **Martin Kleppmann**: [Designing Data-Intensive Applications](https://dataintensive.net/) — データシステム設計の名著
2. **MongoDB 公式**: [MongoDB Manual](https://www.mongodb.com/docs/manual/) — MongoDB の包括的ドキュメント
3. **AWS 公式**: [Choosing the Right Database](https://aws.amazon.com/products/databases/) — AWS データベースサービスの選定ガイド
