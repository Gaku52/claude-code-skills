# Amazon ElastiCache

> AWS のフルマネージドインメモリキャッシュサービスを理解し、Redis/Memcached の選択・キャッシュ戦略・運用パターンを実践的に習得する

## この章で学ぶこと

1. **ElastiCache の基本概念** — Redis と Memcached の特性比較と選定基準
2. **キャッシュ戦略** — Cache-Aside、Write-Through、Write-Behind のパターン選択
3. **運用と最適化** — クラスター設計、フェイルオーバー、メモリ管理、監視

---

## 1. ElastiCache アーキテクチャ

```
+------------------------------------------------------------------+
|  典型的なキャッシュ構成                                           |
|                                                                  |
|  Client --> ALB --> App Server --+--> ElastiCache (Redis)        |
|                                  |   (< 1ms 応答)               |
|                                  |      | Cache Miss時           |
|                                  +----> RDS / DynamoDB           |
|                                      (5-20ms 応答)              |
|                                                                  |
|  レイテンシ比較:                                                  |
|    ElastiCache : < 1ms                                           |
|    RDS         : 5-20ms                                          |
|    DynamoDB    : 5-10ms                                          |
+------------------------------------------------------------------+
```

### コード例 1: Redis クラスターの作成（AWS CLI）

```bash
# Redis クラスター（レプリケーショングループ）の作成
aws elasticache create-replication-group \
  --replication-group-id my-redis-cluster \
  --replication-group-description "Production Redis Cache" \
  --engine redis \
  --engine-version 7.1 \
  --node-type cache.r7g.large \
  --num-node-groups 3 \
  --replicas-per-node-group 2 \
  --cache-subnet-group-name my-cache-subnet \
  --security-group-ids sg-0abc123 \
  --at-rest-encryption-enabled \
  --transit-encryption-enabled \
  --automatic-failover-enabled \
  --multi-az-enabled \
  --snapshot-retention-limit 7 \
  --snapshot-window "03:00-05:00" \
  --tags Key=Environment,Value=production
```

---

## 2. Redis vs Memcached

### 機能比較表

| 機能 | Redis | Memcached |
|---|---|---|
| **データ構造** | String, List, Set, Hash, Sorted Set, Stream 等 | String のみ |
| **永続化** | RDB + AOF | なし（揮発性） |
| **レプリケーション** | 対応（自動フェイルオーバー） | なし |
| **クラスタリング** | Redis Cluster（シャーディング） | 分散ハッシュ（クライアント側） |
| **Pub/Sub** | 対応 | なし |
| **Lua スクリプト** | 対応 | なし |
| **マルチスレッド** | I/O マルチスレッド（7.0+） | マルチスレッド |
| **最大メモリ** | クラスター合計 ~500GB | ノードあたり数百GB |
| **TLS** | 対応 | 対応（1.6.12+） |

### 選定フローチャート

```
Redis vs Memcached 選定フロー
=============================

データ構造が必要? (List, Set, Hash等)
   |         |
  Yes        No
   |         |
   v         v
 Redis    永続化が必要?
            |         |
           Yes        No
            |         |
            v         v
          Redis    レプリカ/フェイルオーバーが必要?
                     |         |
                    Yes        No
                     |         |
                     v         v
                   Redis    Memcached
                            (シンプルな KV キャッシュ)
```

---

## 3. キャッシュ戦略パターン

### パターン比較表

| パターン | 読み取り | 書き込み | 一貫性 | 適用場面 |
|---|---|---|---|---|
| **Cache-Aside** | App がキャッシュ確認 -> Miss時にDB読み取り -> キャッシュ書き込み | DB に直接書き込み | 結果整合 | 汎用、最も一般的 |
| **Read-Through** | キャッシュが自動でDB読み取り | DB に直接書き込み | 結果整合 | ライブラリがサポート時 |
| **Write-Through** | キャッシュから読み取り | キャッシュ -> DB の同期書き込み | 強い整合 | 読み取り頻度が高い |
| **Write-Behind** | キャッシュから読み取り | キャッシュ -> DB の非同期書き込み | 結果整合 | 書き込み頻度が高い |

### コード例 2: Cache-Aside パターン（Python）

```python
import redis
import json

r = redis.Redis(
    host='my-redis-cluster.xxxx.apne1.cache.amazonaws.com',
    port=6379, ssl=True, decode_responses=True,
)

class CacheAside:
    """Cache-Aside パターンの実装"""

    def __init__(self, redis_client, default_ttl=300):
        self.redis = redis_client
        self.default_ttl = default_ttl

    def get_or_set(self, key: str, fetch_fn, ttl: int = None):
        """キャッシュがあれば返し、なければ fetch_fn で取得してキャッシュ"""
        cached = self.redis.get(key)
        if cached is not None:
            return json.loads(cached)

        data = fetch_fn()
        if data is not None:
            self.redis.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(data, default=str),
            )
        return data

    def invalidate(self, key: str):
        """キャッシュの無効化"""
        self.redis.delete(key)

    def invalidate_pattern(self, pattern: str):
        """パターンに一致するキャッシュの一括無効化"""
        cursor = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                self.redis.delete(*keys)
            if cursor == 0:
                break

# 使用例
cache = CacheAside(r, default_ttl=600)

def get_user(user_id):
    return cache.get_or_set(
        f"user:{user_id}",
        lambda: db.query("SELECT * FROM users WHERE id = %s", user_id),
        ttl=3600,
    )

def update_user(user_id, data):
    db.execute("UPDATE users SET name = %s WHERE id = %s", data['name'], user_id)
    cache.invalidate(f"user:{user_id}")
```

### コード例 3: Write-Through パターン

```python
class WriteThrough:
    """Write-Through パターン: キャッシュとDBを同期的に書き込み"""

    def __init__(self, redis_client, db_client, default_ttl=3600):
        self.redis = redis_client
        self.db = db_client
        self.default_ttl = default_ttl

    def read(self, key: str, query: str, params: tuple):
        cached = self.redis.get(key)
        if cached:
            return json.loads(cached)
        data = self.db.query(query, params)
        if data:
            self.redis.setex(key, self.default_ttl, json.dumps(data, default=str))
        return data

    def write(self, key: str, db_query: str, params: tuple, data: dict):
        self.db.execute(db_query, params)
        self.redis.setex(key, self.default_ttl, json.dumps(data, default=str))
```

---

## 4. Redis データ構造の活用

### コード例 4: 実践的なユースケース

```python
import redis
import time

r = redis.Redis(host='my-redis.cache.amazonaws.com', port=6379, ssl=True)

# === セッション管理 ===
def create_session(session_id: str, user_data: dict, ttl: int = 1800):
    r.hset(f"session:{session_id}", mapping=user_data)
    r.expire(f"session:{session_id}", ttl)

def get_session(session_id: str):
    data = r.hgetall(f"session:{session_id}")
    if data:
        r.expire(f"session:{session_id}", 1800)  # スライディング期限
    return data

# === リアルタイムランキング ===
def add_score(leaderboard: str, user_id: str, score: float):
    r.zadd(f"lb:{leaderboard}", {user_id: score})

def get_top_n(leaderboard: str, n: int = 10):
    return r.zrevrange(f"lb:{leaderboard}", 0, n - 1, withscores=True)

def get_user_rank(leaderboard: str, user_id: str):
    rank = r.zrevrank(f"lb:{leaderboard}", user_id)
    return rank + 1 if rank is not None else None

# === レートリミッター（スライディングウィンドウ） ===
def is_rate_limited(user_id: str, max_requests: int = 100, window: int = 60):
    key = f"rate:{user_id}:{int(time.time()) // window}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, window)
    return current > max_requests

# === 分散ロック ===
def acquire_lock(lock_name: str, ttl: int = 10):
    token = str(time.time())
    acquired = r.set(f"lock:{lock_name}", token, nx=True, ex=ttl)
    return token if acquired else None

def release_lock(lock_name: str, token: str):
    script = """
    if redis.call("GET", KEYS[1]) == ARGV[1] then
        return redis.call("DEL", KEYS[1])
    else
        return 0
    end
    """
    r.eval(script, 1, f"lock:{lock_name}", token)
```

### コード例 5: Terraform による ElastiCache 定義

```hcl
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "app-redis-prod"
  description          = "Production Redis Cluster"

  engine               = "redis"
  engine_version       = "7.1"
  node_type            = "cache.r7g.large"

  num_node_groups         = 3
  replicas_per_node_group = 2

  subnet_group_name  = aws_elasticache_subnet_group.redis.name
  security_group_ids = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  automatic_failover_enabled = true
  multi_az_enabled           = true

  snapshot_retention_limit = 7
  snapshot_window          = "03:00-05:00"
  maintenance_window       = "Mon:05:00-Mon:06:00"

  parameter_group_name = aws_elasticache_parameter_group.redis71.name

  tags = { Environment = "production" }
}

resource "aws_elasticache_parameter_group" "redis71" {
  family = "redis7"
  name   = "app-redis71-params"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }
}
```

---

## 5. メモリ管理とエビクションポリシー

```
エビクションポリシー選択ガイド
===============================

allkeys-lru    --> 全キーから LRU で削除（最も一般的）
volatile-lru   --> TTL 付きキーから LRU で削除
allkeys-lfu    --> 全キーから LFU で削除（使用頻度ベース）
volatile-lfu   --> TTL 付きキーから LFU で削除
volatile-ttl   --> TTL が近いキーから削除
allkeys-random --> 全キーからランダム削除
noeviction     --> 削除せずエラー返却（データ損失不可の場合）

推奨:
  キャッシュ用途     --> allkeys-lru or allkeys-lfu
  セッション用途     --> volatile-lru
  永続データ混在     --> volatile-lru
  データ損失不可     --> noeviction (メモリ監視必須)
```

---

## アンチパターン

### 1. キャッシュスタンピード（Thundering Herd）

**問題**: 人気キーの TTL が切れた瞬間に、大量のリクエストが同時にキャッシュミスとなり、全てがデータベースに殺到する。

```python
# [NG] 単純な Cache-Aside
def get_popular_item(item_id):
    cached = redis.get(f"item:{item_id}")
    if cached is None:
        # 100リクエストが同時にここに到達 --> DB 過負荷
        data = db.query("SELECT * FROM items WHERE id = %s", item_id)
        redis.setex(f"item:{item_id}", 300, json.dumps(data))
        return data
    return json.loads(cached)

# [OK] ロック + 確率的早期再計算
import random

def get_popular_item_safe(item_id):
    key = f"item:{item_id}"
    cached = redis.get(key)
    if cached is not None:
        ttl = redis.ttl(key)
        if ttl < 30 and random.random() < 0.1:
            _refresh_cache(key, item_id)  # 確率的早期更新
        return json.loads(cached)

    lock_key = f"lock:{key}"
    if redis.set(lock_key, "1", nx=True, ex=5):
        try:
            data = db.query("SELECT * FROM items WHERE id = %s", item_id)
            redis.setex(key, 300, json.dumps(data))
            return data
        finally:
            redis.delete(lock_key)
    else:
        time.sleep(0.1)
        return get_popular_item_safe(item_id)
```

### 2. 巨大なキーの格納

**問題**: 1つのキーに数 MB のデータを格納すると、読み書き時にブロッキングが発生し、クラスター全体の性能が劣化する。Redis はシングルスレッドでコマンドを処理するため影響が大きい。

**対策**: 大きなデータは分割して格納する。リストは `LRANGE` でページネーション、ハッシュは `HSCAN` で部分取得。1キーのサイズは 100KB 以下を目安とする。

---

## FAQ

### Q1: ElastiCache と DynamoDB DAX はどう使い分けますか？

**A**:
- **ElastiCache**: 汎用キャッシュ。RDS・DynamoDB・API レスポンスなど何でもキャッシュ可能。セッション管理、ランキング等の独自データ構造にも対応
- **DAX**: DynamoDB 専用キャッシュ。アプリケーションコードの変更最小限で DynamoDB の読み取りを高速化

DynamoDB のみのキャッシュなら DAX、複数データソースや高度なデータ構造が必要なら ElastiCache を選択します。

### Q2: Redis のメモリが枯渇した場合どうなりますか？

**A**: エビクションポリシーに依存します。`allkeys-lru` なら古いキーが自動削除され、`noeviction` なら書き込みエラーが返ります。CloudWatch の `DatabaseMemoryUsagePercentage` を監視し、75% 超過でアラートを設定してください。

### Q3: Redis クラスターモードの有効/無効はどう判断しますか？

**A**: データ量が単一ノードのメモリに収まるなら無効（シンプル）。データが大きい、または書き込みスループットをスケールしたい場合は有効（シャーディング）。クラスターモード有効時はマルチキー操作に制約（同一スロット内のみ）があるため、アクセスパターンとの整合性を確認してください。

---

## まとめ

| 項目 | 要点 |
|---|---|
| サービス概要 | フルマネージドインメモリキャッシュ。Redis / Memcached を選択可能 |
| エンジン選択 | 迷ったら Redis。データ構造・永続化・レプリケーション全対応 |
| キャッシュ戦略 | Cache-Aside が基本。書き込み頻度が高い場合は Write-Behind |
| 高可用性 | マルチ AZ + 自動フェイルオーバーで可用性を確保 |
| メモリ管理 | allkeys-lru が標準。75% 以上でスケーリング検討 |
| 監視 | CacheHitRate、CPUUtilization、DatabaseMemoryUsage が主要メトリクス |

## 次に読むべきガイド

- [RDS 基礎](./00-rds-basics.md) — キャッシュ対象のリレーショナルデータベース
- [DynamoDB](./01-dynamodb.md) — NoSQL との組み合わせパターン
- [VPC 基礎](../04-networking/00-vpc-basics.md) — ElastiCache のネットワーク配置

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon ElastiCache for Redis ユーザーガイド](https://docs.aws.amazon.com/ja_jp/AmazonElastiCache/latest/red-ug/) — 設定・運用の詳細リファレンス
2. **Redis 公式**: [Redis Documentation](https://redis.io/docs/) — データ構造・コマンドリファレンス
3. **AWS アーキテクチャブログ**: [Caching Best Practices](https://aws.amazon.com/caching/best-practices/) — AWS でのキャッシュ戦略ガイド
