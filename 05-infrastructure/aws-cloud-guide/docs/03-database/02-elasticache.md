# Amazon ElastiCache

> AWS のフルマネージドインメモリキャッシュサービスを理解し、Redis/Memcached の選択・キャッシュ戦略・クラスター設計・運用パターン・障害対応を実践的に習得する

## この章で学ぶこと

1. **ElastiCache の基本概念** — Redis と Memcached の特性比較と選定基準
2. **キャッシュ戦略** — Cache-Aside、Write-Through、Write-Behind のパターン選択
3. **運用と最適化** — クラスター設計、フェイルオーバー、メモリ管理、監視
4. **高可用性設計** — マルチ AZ、レプリケーション、バックアップ/リストア
5. **セキュリティ** — 暗号化、認証、ネットワーク設計のベストプラクティス

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

### キャッシュの効果測定

```
キャッシュヒット率とレイテンシの関係:
======================================

ヒット率    平均レイテンシ      DB負荷
0%          20ms (全てDB)      100%
50%         ~10ms              50%
80%         ~4ms               20%
90%         ~2ms               10%
95%         ~1.5ms             5%
99%         ~1.2ms             1%

損益分岐点:
  ElastiCache cache.r7g.large (3ノード) ≈ $700/月
  RDS db.r6g.xlarge ≈ $500/月
  → キャッシュで RDS インスタンスサイズを1段階下げられれば元が取れる
  → 読み取り比率が高い（80%+）ワークロードで特に効果的
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

# Memcached クラスターの作成
aws elasticache create-cache-cluster \
  --cache-cluster-id my-memcached-cluster \
  --engine memcached \
  --engine-version 1.6.22 \
  --cache-node-type cache.r7g.large \
  --num-cache-nodes 3 \
  --cache-subnet-group-name my-cache-subnet \
  --security-group-ids sg-0abc123 \
  --az-mode cross-az \
  --tags Key=Environment,Value=production

# クラスターの状態確認
aws elasticache describe-replication-groups \
  --replication-group-id my-redis-cluster \
  --query 'ReplicationGroups[0].{Status:Status,Nodes:NodeGroups[*].NodeGroupMembers[*].{Id:CacheClusterId,AZ:PreferredAvailabilityZone,Role:CurrentRole}}'

# エンドポイントの取得
aws elasticache describe-replication-groups \
  --replication-group-id my-redis-cluster \
  --query 'ReplicationGroups[0].{Primary:NodeGroups[0].PrimaryEndpoint,Reader:NodeGroups[0].ReaderEndpoint}'
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
| **Streams** | 対応（ログ構造データ） | なし |
| **Geospatial** | 対応（位置情報クエリ） | なし |
| **JSON サポート** | RedisJSON モジュール対応 | なし |

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

### ノードタイプの選定ガイド

| 用途 | 推奨ノードタイプ | メモリ | ネットワーク | 月額概算（東京） |
|---|---|---|---|---|
| 開発/テスト | cache.t4g.micro | 0.5 GB | 最大 5 Gbps | ~$15 |
| 小規模本番 | cache.r7g.large | 13.07 GB | 最大 12.5 Gbps | ~$230 |
| 中規模本番 | cache.r7g.xlarge | 26.32 GB | 最大 12.5 Gbps | ~$460 |
| 大規模本番 | cache.r7g.2xlarge | 52.82 GB | 最大 12.5 Gbps | ~$920 |
| 超大規模 | cache.r7g.4xlarge | 105.81 GB | 最大 12.5 Gbps | ~$1,840 |

---

## 3. キャッシュ戦略パターン

### パターン比較表

| パターン | 読み取り | 書き込み | 一貫性 | 適用場面 |
|---|---|---|---|---|
| **Cache-Aside** | App がキャッシュ確認 -> Miss時にDB読み取り -> キャッシュ書き込み | DB に直接書き込み | 結果整合 | 汎用、最も一般的 |
| **Read-Through** | キャッシュが自動でDB読み取り | DB に直接書き込み | 結果整合 | ライブラリがサポート時 |
| **Write-Through** | キャッシュから読み取り | キャッシュ -> DB の同期書き込み | 強い整合 | 読み取り頻度が高い |
| **Write-Behind** | キャッシュから読み取り | キャッシュ -> DB の非同期書き込み | 結果整合 | 書き込み頻度が高い |

```
キャッシュ戦略の詳細フロー:

1. Cache-Aside (Lazy Loading):
   Read:
     App --> Redis.GET(key)
       |-- HIT  --> return data
       |-- MISS --> DB.SELECT --> Redis.SET(key, data, TTL) --> return data

   Write:
     App --> DB.UPDATE --> Redis.DEL(key)  ← キャッシュ無効化

2. Write-Through:
   Write:
     App --> Redis.SET(key, data) --> DB.UPDATE  ← 同期的
   Read:
     App --> Redis.GET(key)
       |-- HIT  --> return data
       |-- MISS --> DB.SELECT --> Redis.SET(key, data) --> return data

3. Write-Behind (Write-Back):
   Write:
     App --> Redis.SET(key, data) --> return success
                |
                +---> [非同期] DB.UPDATE  ← バッチ処理/遅延書き込み
```

### コード例 2: Cache-Aside パターン（Python）

```python
import redis
import json
import hashlib
import logging
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

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
            logger.debug(f"Cache HIT: {key}")
            return json.loads(cached)

        logger.debug(f"Cache MISS: {key}")
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
        logger.debug(f"Cache INVALIDATED: {key}")

    def invalidate_pattern(self, pattern: str):
        """パターンに一致するキャッシュの一括無効化"""
        cursor = 0
        deleted = 0
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            if keys:
                self.redis.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break
        logger.info(f"Cache INVALIDATED {deleted} keys matching: {pattern}")

    def cached(self, prefix: str, ttl: int = None):
        """デコレータとして使えるキャッシュ"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 引数からキャッシュキーを生成
                key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
                cache_key = f"{prefix}:{hashlib.md5(key_data.encode()).hexdigest()}"
                return self.get_or_set(cache_key, lambda: func(*args, **kwargs), ttl)
            return wrapper
        return decorator

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

# デコレータパターン
@cache.cached("product", ttl=1800)
def get_product(product_id: str):
    return db.query("SELECT * FROM products WHERE id = %s", product_id)
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
        # DB に先に書き込み（失敗時にキャッシュだけ更新されることを防ぐ）
        self.db.execute(db_query, params)
        self.redis.setex(key, self.default_ttl, json.dumps(data, default=str))


class WriteBehind:
    """Write-Behind パターン: キャッシュに即時書き込み、DBに非同期書き込み"""

    def __init__(self, redis_client, default_ttl=3600):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.write_queue_key = "write_behind:queue"

    def write(self, key: str, data: dict):
        """キャッシュに即時書き込み + キューに追加"""
        pipe = self.redis.pipeline()
        pipe.setex(key, self.default_ttl, json.dumps(data, default=str))
        pipe.rpush(self.write_queue_key, json.dumps({
            'key': key,
            'data': data,
            'timestamp': time.time(),
        }))
        pipe.execute()

    def process_queue(self, batch_size: int = 100):
        """キューからバッチで取り出してDBに書き込み"""
        items = []
        for _ in range(batch_size):
            item = self.redis.lpop(self.write_queue_key)
            if item is None:
                break
            items.append(json.loads(item))

        if items:
            # バッチでDBに書き込み
            db.batch_upsert(items)
            logger.info(f"Write-Behind: processed {len(items)} items")
```

---

## 4. Redis データ構造の活用

### コード例 4: 実践的なユースケース

```python
import redis
import time
import json
from datetime import datetime, timezone

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

def destroy_session(session_id: str):
    r.delete(f"session:{session_id}")

# === リアルタイムランキング ===
def add_score(leaderboard: str, user_id: str, score: float):
    r.zadd(f"lb:{leaderboard}", {user_id: score})

def get_top_n(leaderboard: str, n: int = 10):
    return r.zrevrange(f"lb:{leaderboard}", 0, n - 1, withscores=True)

def get_user_rank(leaderboard: str, user_id: str):
    rank = r.zrevrank(f"lb:{leaderboard}", user_id)
    return rank + 1 if rank is not None else None

def get_around_user(leaderboard: str, user_id: str, n: int = 5):
    """ユーザー前後のランキングを取得"""
    rank = r.zrevrank(f"lb:{leaderboard}", user_id)
    if rank is None:
        return None
    start = max(0, rank - n)
    end = rank + n
    return r.zrevrange(f"lb:{leaderboard}", start, end, withscores=True)

# === レートリミッター（スライディングウィンドウ） ===
def is_rate_limited(user_id: str, max_requests: int = 100, window: int = 60):
    key = f"rate:{user_id}:{int(time.time()) // window}"
    current = r.incr(key)
    if current == 1:
        r.expire(key, window)
    return current > max_requests

# === 高精度レートリミッター（スライディングログ） ===
def is_rate_limited_precise(user_id: str, max_requests: int = 100, window: int = 60):
    """タイムスタンプベースの高精度レートリミッター"""
    key = f"rate:log:{user_id}"
    now = time.time()
    window_start = now - window

    pipe = r.pipeline()
    pipe.zremrangebyscore(key, 0, window_start)  # 古いエントリを削除
    pipe.zadd(key, {f"{now}": now})  # 現在のリクエストを追加
    pipe.zcard(key)  # ウィンドウ内のリクエスト数を取得
    pipe.expire(key, window)
    results = pipe.execute()

    return results[2] > max_requests

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

# === Pub/Sub メッセージング ===
def publish_event(channel: str, event_type: str, data: dict):
    message = json.dumps({
        'type': event_type,
        'data': data,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })
    r.publish(channel, message)

def subscribe_events(channel: str, callback):
    """イベントの購読（ブロッキング）"""
    pubsub = r.pubsub()
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message['type'] == 'message':
            event = json.loads(message['data'])
            callback(event)

# === Redis Streams（イベントログ） ===
def add_to_stream(stream: str, data: dict, maxlen: int = 10000):
    """Redis Streams にイベントを追加"""
    r.xadd(
        f"stream:{stream}",
        data,
        maxlen=maxlen,
        approximate=True,
    )

def read_stream(stream: str, last_id: str = '0', count: int = 100):
    """Redis Streams からイベントを読み取り"""
    return r.xread(
        {f"stream:{stream}": last_id},
        count=count,
        block=5000,  # 5秒間ブロック
    )

# === カウンター（HyperLogLog） ===
def add_unique_visitor(page: str, visitor_id: str):
    """ユニーク訪問者をカウント（メモリ効率的）"""
    r.pfadd(f"uv:{page}:{datetime.now().strftime('%Y-%m-%d')}", visitor_id)

def get_unique_visitors(page: str, date: str = None):
    """ユニーク訪問者数を取得（誤差 0.81%）"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    return r.pfcount(f"uv:{page}:{date}")

# === Geospatial（位置情報） ===
def add_location(key: str, name: str, longitude: float, latitude: float):
    """位置情報を追加"""
    r.geoadd(f"geo:{key}", (longitude, latitude, name))

def find_nearby(key: str, longitude: float, latitude: float, radius_km: float):
    """近隣の位置を検索"""
    return r.geosearch(
        f"geo:{key}",
        longitude=longitude,
        latitude=latitude,
        radius=radius_km,
        unit='km',
        sort='ASC',
        withcoord=True,
        withdist=True,
    )
```

### コード例 5: Terraform による ElastiCache 定義

```hcl
# サブネットグループ
resource "aws_elasticache_subnet_group" "redis" {
  name       = "app-redis-subnet"
  subnet_ids = var.private_subnet_ids

  tags = { Environment = var.environment }
}

# セキュリティグループ
resource "aws_security_group" "redis" {
  name        = "app-redis-sg"
  description = "Security group for Redis cluster"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.app_security_group_id]
    description     = "Allow Redis access from app servers"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "app-redis-sg" }
}

# Redis クラスター
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "app-redis-${var.environment}"
  description          = "${var.environment} Redis Cluster"

  engine               = "redis"
  engine_version       = "7.1"
  node_type            = var.redis_node_type

  num_node_groups         = var.num_shards
  replicas_per_node_group = var.replicas_per_shard

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

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "slow-log"
  }

  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_engine_log.name
    destination_type = "cloudwatch-logs"
    log_format       = "json"
    log_type         = "engine-log"
  }

  tags = {
    Environment = var.environment
    Service     = "cache"
  }
}

resource "aws_elasticache_parameter_group" "redis71" {
  family = "redis7"
  name   = "app-redis71-params-${var.environment}"

  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }

  parameter {
    name  = "notify-keyspace-events"
    value = "Ex"  # 期限切れイベントを通知
  }

  parameter {
    name  = "timeout"
    value = "300"  # アイドル接続のタイムアウト（秒）
  }

  parameter {
    name  = "tcp-keepalive"
    value = "60"
  }
}

# CloudWatch ロググループ
resource "aws_cloudwatch_log_group" "redis_slow_log" {
  name              = "/elasticache/${var.environment}/redis/slow-log"
  retention_in_days = 30
}

resource "aws_cloudwatch_log_group" "redis_engine_log" {
  name              = "/elasticache/${var.environment}/redis/engine-log"
  retention_in_days = 30
}

# 出力
output "redis_primary_endpoint" {
  value = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "redis_reader_endpoint" {
  value = aws_elasticache_replication_group.redis.reader_endpoint_address
}

output "redis_configuration_endpoint" {
  value = aws_elasticache_replication_group.redis.configuration_endpoint_address
}
```

### コード例 5b: CloudFormation 定義

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: ElastiCache Redis Cluster

Parameters:
  Environment:
    Type: String
    Default: production
  NodeType:
    Type: String
    Default: cache.r7g.large
  NumShards:
    Type: Number
    Default: 3
  ReplicasPerShard:
    Type: Number
    Default: 2

Resources:
  RedisSubnetGroup:
    Type: AWS::ElastiCache::SubnetGroup
    Properties:
      Description: Subnet group for Redis
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2
        - !Ref PrivateSubnet3

  RedisParameterGroup:
    Type: AWS::ElastiCache::ParameterGroup
    Properties:
      CacheParameterGroupFamily: redis7
      Description: Custom Redis 7 parameters
      Properties:
        maxmemory-policy: allkeys-lru
        notify-keyspace-events: Ex
        timeout: '300'

  RedisCluster:
    Type: AWS::ElastiCache::ReplicationGroup
    Properties:
      ReplicationGroupDescription: !Sub '${Environment} Redis Cluster'
      Engine: redis
      EngineVersion: '7.1'
      CacheNodeType: !Ref NodeType
      NumNodeGroups: !Ref NumShards
      ReplicasPerNodeGroup: !Ref ReplicasPerShard
      CacheSubnetGroupName: !Ref RedisSubnetGroup
      CacheParameterGroupName: !Ref RedisParameterGroup
      SecurityGroupIds:
        - !Ref RedisSecurityGroup
      AtRestEncryptionEnabled: true
      TransitEncryptionEnabled: true
      AutomaticFailoverEnabled: true
      MultiAZEnabled: true
      SnapshotRetentionLimit: 7
      SnapshotWindow: '03:00-05:00'
      PreferredMaintenanceWindow: 'Mon:05:00-Mon:06:00'
      Tags:
        - Key: Environment
          Value: !Ref Environment

Outputs:
  PrimaryEndpoint:
    Value: !GetAtt RedisCluster.PrimaryEndPoint.Address
  ReaderEndpoint:
    Value: !GetAtt RedisCluster.ReaderEndPoint.Address
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

### メモリ使用量の分析

```bash
# Redis のメモリ情報を取得
redis-cli -h my-redis-cluster.xxxx.apne1.cache.amazonaws.com \
  --tls -p 6379 INFO memory

# メモリ使用量の詳細（Redis 4.0+）
redis-cli -h my-redis-cluster.xxxx.apne1.cache.amazonaws.com \
  --tls -p 6379 MEMORY DOCTOR

# 特定キーのメモリ使用量
redis-cli -h my-redis-cluster.xxxx.apne1.cache.amazonaws.com \
  --tls -p 6379 MEMORY USAGE "user:12345"

# 大きなキーの検出
redis-cli -h my-redis-cluster.xxxx.apne1.cache.amazonaws.com \
  --tls -p 6379 --bigkeys

# スロークエリログの確認
redis-cli -h my-redis-cluster.xxxx.apne1.cache.amazonaws.com \
  --tls -p 6379 SLOWLOG GET 10
```

### メモリ最適化のベストプラクティス

```
メモリ最適化チェックリスト:
==============================

1. データ構造の選択
   - 小さなハッシュ（<128フィールド）は ziplist で圧縮保存
   - 小さなリスト（<128要素）は ziplist で圧縮保存
   - 小さなセット（<128要素）は intset/ziplist で圧縮保存

2. キーの命名
   - 短いキー名を使用（user:123 vs user_profile_data:123）
   - 一貫したプレフィックス（SCAN でのパターン検索に有効）

3. TTL の設定
   - 全キャッシュキーに TTL を設定
   - ビジネスロジックに応じた適切な TTL
   - ランダムな TTL オフセットでスタンピードを防止

4. データの圧縮
   - 大きな JSON は gzip/lz4 で圧縮して保存
   - MessagePack 等のバイナリフォーマットの利用

5. 不要データの削除
   - UNLINK（非同期削除）を使用
   - SCAN + DEL でバッチ削除
```

```python
import gzip
import json

class CompressedCache:
    """圧縮キャッシュ: 大きなデータを圧縮して保存"""

    def __init__(self, redis_client, compression_threshold=1024):
        self.redis = redis_client
        self.threshold = compression_threshold

    def set(self, key: str, data: Any, ttl: int = 3600):
        serialized = json.dumps(data, default=str).encode('utf-8')
        if len(serialized) > self.threshold:
            compressed = gzip.compress(serialized)
            self.redis.setex(f"gz:{key}", ttl, compressed)
        else:
            self.redis.setex(key, ttl, serialized)

    def get(self, key: str) -> Optional[Any]:
        # 圧縮版を先にチェック
        data = self.redis.get(f"gz:{key}")
        if data is not None:
            return json.loads(gzip.decompress(data))

        data = self.redis.get(key)
        if data is not None:
            return json.loads(data)

        return None
```

---

## 6. 高可用性設計

### クラスターモードの比較

```
クラスターモード無効 (Disabled):
================================
  +------------------+
  | Primary          |
  | (Read/Write)     |
  +------------------+
         |
    +----+----+
    |         |
  +-----+  +-----+
  | R1  |  | R2  |  ← Read Replica
  +-----+  +-----+

  特徴:
  - 単一シャード
  - 最大5レプリカ
  - 最大メモリ: ノードのメモリ
  - Multi-AZ フェイルオーバー対応


クラスターモード有効 (Enabled):
================================
  Shard 1              Shard 2              Shard 3
  +--------+          +--------+          +--------+
  |Primary |          |Primary |          |Primary |
  |(0-5460)|          |(5461-  |          |(10923- |
  +--------+          |10922)  |          |16383)  |
     |  |             +--------+          +--------+
  +--+ +--+              |  |                |  |
  |R1|  |R2|          +--+ +--+           +--+ +--+
  +--+  +--+          |R1|  |R2|          |R1|  |R2|
                      +--+  +--+          +--+  +--+

  特徴:
  - 最大500シャード
  - シャードあたり最大5レプリカ
  - ハッシュスロットベースの分散（16384スロット）
  - オンラインリシャーディング対応
  - 最大メモリ: ノード数 × ノードメモリ
```

### フェイルオーバーの動作

```
フェイルオーバーのフロー:
==========================

1. Primary ノード障害検知
   ElastiCache → ヘルスチェック失敗（数秒）
                ↓
2. フェイルオーバー開始
   ElastiCache → Read Replica を Primary に昇格
                ↓
3. DNS 更新
   Primary Endpoint → 新 Primary の IP に更新
                ↓
4. 新 Primary が書き込み受付開始
   ダウンタイム: 通常 30秒～数分

対策:
  - アプリケーション側でリトライロジックを実装
  - 接続プールのリフレッシュ機構
  - CloudWatch アラームで通知
```

### コード例 6: 接続プール管理

```python
import redis
from redis.sentinel import Sentinel
from redis.retry import Retry
from redis.backoff import ExponentialBackoff
import logging

logger = logging.getLogger(__name__)

def create_redis_client(
    host: str,
    port: int = 6379,
    ssl: bool = True,
    max_connections: int = 50,
    socket_timeout: float = 5.0,
    retry_on_timeout: bool = True,
) -> redis.Redis:
    """本番環境向け Redis クライアントの作成"""

    # リトライ設定
    retry = Retry(ExponentialBackoff(), retries=3)

    pool = redis.ConnectionPool(
        host=host,
        port=port,
        ssl=ssl,
        max_connections=max_connections,
        socket_timeout=socket_timeout,
        socket_connect_timeout=5.0,
        socket_keepalive=True,
        health_check_interval=30,
        retry_on_timeout=retry_on_timeout,
        retry=retry,
        decode_responses=True,
    )

    client = redis.Redis(connection_pool=pool)

    # 接続テスト
    try:
        client.ping()
        logger.info(f"Redis connection established: {host}:{port}")
    except redis.ConnectionError as e:
        logger.error(f"Redis connection failed: {e}")
        raise

    return client


def create_cluster_client(
    host: str,
    port: int = 6379,
    ssl: bool = True,
) -> redis.RedisCluster:
    """クラスターモード有効時のクライアント"""

    return redis.RedisCluster(
        host=host,
        port=port,
        ssl=ssl,
        decode_responses=True,
        skip_full_coverage_check=True,
        socket_timeout=5.0,
        retry_on_timeout=True,
    )
```

---

## 7. CloudWatch 監視

### 主要メトリクス一覧

| メトリクス | 説明 | アラーム閾値 |
|---|---|---|
| CacheHitRate | キャッシュヒット率 | < 80% |
| CPUUtilization | CPU 使用率 | > 70% |
| EngineCPUUtilization | Redis エンジン CPU | > 90% |
| DatabaseMemoryUsagePercentage | メモリ使用率 | > 75% |
| CurrConnections | 現在の接続数 | > 最大の 80% |
| Evictions | エビクション数 | > 0（監視） |
| ReplicationLag | レプリケーション遅延 | > 1 秒 |
| SwapUsage | スワップ使用量 | > 0（要調査） |
| NetworkBandwidthInAllowanceExceeded | ネットワーク帯域超過 | > 0 |

### コード例 7: CloudWatch アラーム設定

```bash
# メモリ使用率アラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "Redis-MemoryUsage-High" \
  --alarm-description "Redis memory usage exceeds 75%" \
  --metric-name DatabaseMemoryUsagePercentage \
  --namespace AWS/ElastiCache \
  --statistic Average \
  --period 300 \
  --threshold 75 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --dimensions \
    Name=CacheClusterId,Value=my-redis-cluster-001 \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# キャッシュヒット率アラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "Redis-CacheHitRate-Low" \
  --alarm-description "Cache hit rate below 80%" \
  --metric-name CacheHitRate \
  --namespace AWS/ElastiCache \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator LessThanThreshold \
  --evaluation-periods 3 \
  --dimensions \
    Name=CacheClusterId,Value=my-redis-cluster-001 \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# CPU 使用率アラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "Redis-CPU-High" \
  --alarm-description "Redis engine CPU exceeds 90%" \
  --metric-name EngineCPUUtilization \
  --namespace AWS/ElastiCache \
  --statistic Average \
  --period 60 \
  --threshold 90 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 5 \
  --dimensions \
    Name=CacheClusterId,Value=my-redis-cluster-001 \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# エビクションアラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "Redis-Evictions" \
  --alarm-description "Redis evictions detected" \
  --metric-name Evictions \
  --namespace AWS/ElastiCache \
  --statistic Sum \
  --period 300 \
  --threshold 100 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --dimensions \
    Name=CacheClusterId,Value=my-redis-cluster-001 \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# レプリケーション遅延アラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "Redis-ReplicationLag" \
  --alarm-description "Redis replication lag exceeds 1 second" \
  --metric-name ReplicationLag \
  --namespace AWS/ElastiCache \
  --statistic Maximum \
  --period 60 \
  --threshold 1 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 3 \
  --dimensions \
    Name=CacheClusterId,Value=my-redis-cluster-002 \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts
```

---

## 8. バックアップとリストア

```bash
# 手動スナップショットの作成
aws elasticache create-snapshot \
  --replication-group-id my-redis-cluster \
  --snapshot-name "manual-backup-$(date +%Y%m%d)"

# スナップショットの一覧
aws elasticache describe-snapshots \
  --replication-group-id my-redis-cluster \
  --query 'Snapshots[*].{Name:SnapshotName,Status:SnapshotStatus,Time:NodeSnapshots[0].SnapshotCreateTime}'

# スナップショットからのリストア（新しいクラスターとして）
aws elasticache create-replication-group \
  --replication-group-id my-redis-restored \
  --replication-group-description "Restored from snapshot" \
  --snapshot-name "manual-backup-20260215" \
  --engine redis \
  --engine-version 7.1 \
  --node-type cache.r7g.large \
  --num-node-groups 3 \
  --replicas-per-node-group 2 \
  --cache-subnet-group-name my-cache-subnet \
  --security-group-ids sg-0abc123

# S3 へのスナップショットエクスポート
aws elasticache copy-snapshot \
  --source-snapshot-name "manual-backup-20260215" \
  --target-snapshot-name "s3-export-20260215" \
  --target-bucket my-redis-backups

# スナップショットの削除
aws elasticache delete-snapshot \
  --snapshot-name "manual-backup-20260215"
```

---

## 9. セキュリティ設計

### 認証と暗号化

```
ElastiCache セキュリティレイヤー:
=================================

1. ネットワーク分離
   - VPC 内に配置（パブリックアクセス不可）
   - プライベートサブネットに配置
   - セキュリティグループでアクセス元を制限

2. 暗号化
   - 転送中の暗号化 (TLS)
   - 保管時の暗号化 (KMS)

3. 認証
   - Redis AUTH（パスワード認証）
   - RBAC（ロールベースアクセス制御、Redis 7.0+）
   - IAM 認証（ElastiCache Serverless）

4. 監査
   - CloudTrail（API 操作の記録）
   - Slow Log（スロークエリの記録）
   - Engine Log（エンジンイベントの記録）
```

### コード例 8: AUTH 認証付き接続

```python
import redis

# AUTH パスワード認証
r = redis.Redis(
    host='my-redis-cluster.xxxx.apne1.cache.amazonaws.com',
    port=6379,
    password='MySecurePassword123!',
    ssl=True,
    ssl_cert_reqs='required',
    ssl_ca_certs='/etc/ssl/certs/ca-certificates.crt',
    decode_responses=True,
)

# RBAC 認証（Redis 7.0+）
r_rbac = redis.Redis(
    host='my-redis-cluster.xxxx.apne1.cache.amazonaws.com',
    port=6379,
    username='app-user',
    password='AppUserPassword123!',
    ssl=True,
    decode_responses=True,
)
```

```bash
# RBAC ユーザーの作成
aws elasticache create-user \
  --user-id app-readonly \
  --user-name app-readonly \
  --engine redis \
  --passwords "ReadOnlyPassword123!" \
  --access-string "on ~app:* +get +mget +hget +hgetall -@write"

# ユーザーグループの作成
aws elasticache create-user-group \
  --user-group-id app-users \
  --engine redis \
  --user-ids default app-readonly

# ユーザーグループをクラスターに割り当て
aws elasticache modify-replication-group \
  --replication-group-id my-redis-cluster \
  --user-group-ids-to-add app-users
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

# [OK] TTL にジッターを追加
import random

def set_with_jitter(key: str, data: Any, base_ttl: int = 300):
    """TTL にランダムなジッターを追加してスタンピードを防止"""
    jitter = random.randint(0, int(base_ttl * 0.1))  # 10% のジッター
    redis.setex(key, base_ttl + jitter, json.dumps(data, default=str))
```

### 2. 巨大なキーの格納

**問題**: 1つのキーに数 MB のデータを格納すると、読み書き時にブロッキングが発生し、クラスター全体の性能が劣化する。Redis はシングルスレッドでコマンドを処理するため影響が大きい。

**対策**: 大きなデータは分割して格納する。リストは `LRANGE` でページネーション、ハッシュは `HSCAN` で部分取得。1キーのサイズは 100KB 以下を目安とする。

### 3. KEYS コマンドの使用

**問題**: `KEYS *` はブロッキング操作で、Redis が応答不能になる。

```python
# [NG] KEYS コマンド（本番環境で絶対に使用禁止）
keys = redis.keys("user:*")  # テーブルスキャンと同じ

# [OK] SCAN コマンド（ノンブロッキング）
cursor = 0
keys = []
while True:
    cursor, batch = redis.scan(cursor, match="user:*", count=100)
    keys.extend(batch)
    if cursor == 0:
        break
```

### 4. 接続管理の不備

**問題**: Lambda など短命なプロセスで毎回新しい接続を作成すると、接続数が爆発する。

```python
# [NG] 関数呼び出しごとに接続作成
def handler(event, context):
    r = redis.Redis(host='...', port=6379)  # 毎回新しい接続
    data = r.get("key")
    r.close()  # 明示的にクローズしても、同時実行時に接続数が膨れる

# [OK] グローバルスコープで接続を再利用
r = redis.Redis(
    host='...',
    port=6379,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True,
)

def handler(event, context):
    data = r.get("key")  # 既存の接続を再利用
```

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

### Q4: ElastiCache のスケーリング方法は？

**A**: 以下の方法があります:
1. **スケールアップ**: ノードタイプを変更（ダウンタイムあり）
2. **スケールアウト（読み取り）**: レプリカノードを追加（最大5）
3. **シャード追加（クラスターモード）**: オンラインリシャーディングでシャードを追加
4. **ElastiCache Serverless**: 自動スケーリング対応のサーバーレスオプション

```bash
# レプリカの追加
aws elasticache increase-replica-count \
  --replication-group-id my-redis-cluster \
  --new-replica-count 3 \
  --apply-immediately

# ノードタイプの変更（スケールアップ）
aws elasticache modify-replication-group \
  --replication-group-id my-redis-cluster \
  --cache-node-type cache.r7g.xlarge \
  --apply-immediately

# シャードの追加（クラスターモード有効時）
aws elasticache modify-replication-group-shard-configuration \
  --replication-group-id my-redis-cluster \
  --node-group-count 5 \
  --apply-immediately
```

### Q5: ElastiCache Serverless とは？

**A**: 2023年に発表された新しいオプションで、キャパシティの自動スケーリングとパッチ適用を自動管理します。ECPU（ElastiCache Processing Unit）とデータストレージ量に基づく従量課金で、小規模から大規模まで柔軟に対応できます。ただし、従来のノードベースと比較するとコスト単価は高くなるため、安定した高負荷ワークロードでは従来型が有利です。

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
| セキュリティ | VPC 内配置 + TLS + AUTH/RBAC + 保管時暗号化 |
| バックアップ | 自動スナップショット（最大35日）+ 手動スナップショット |
| スケーリング | レプリカ追加（読み取り）、リシャーディング（書き込み）、ノードタイプ変更 |

## 次に読むべきガイド

- [RDS 基礎](./00-rds-basics.md) — キャッシュ対象のリレーショナルデータベース
- [DynamoDB](./01-dynamodb.md) — NoSQL との組み合わせパターン
- [VPC 基礎](../04-networking/00-vpc-basics.md) — ElastiCache のネットワーク配置

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon ElastiCache for Redis ユーザーガイド](https://docs.aws.amazon.com/ja_jp/AmazonElastiCache/latest/red-ug/) — 設定・運用の詳細リファレンス
2. **Redis 公式**: [Redis Documentation](https://redis.io/docs/) — データ構造・コマンドリファレンス
3. **AWS アーキテクチャブログ**: [Caching Best Practices](https://aws.amazon.com/caching/best-practices/) — AWS でのキャッシュ戦略ガイド
4. **AWS Well-Architected**: [Performance Efficiency Pillar](https://docs.aws.amazon.com/wellarchitected/latest/performance-efficiency-pillar/) — キャッシュ設計のベストプラクティス
5. **Redis University**: [Redis University](https://university.redis.com/) — 無料のオンライン学習コース
