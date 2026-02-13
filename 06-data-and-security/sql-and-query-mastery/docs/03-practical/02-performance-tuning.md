# パフォーマンスチューニング — 接続プール / キャッシュ / クエリ最適化

> データベースのレスポンスタイムを劇的に改善する実践テクニック。接続プール、キャッシュ戦略、スロークエリ分析を体系的に学ぶ。

## 前提知識

- SQL の基本（SELECT, JOIN, WHERE, GROUP BY）
- インデックスの基本概念
- ネットワーク通信の基礎（TCP, TLS）
- [00-postgresql-features.md](./00-postgresql-features.md) — PostgreSQL 固有機能

---

## この章で学ぶこと

1. **接続プール** の設計と適切なサイズ計算
2. **キャッシュ戦略** — アプリケーションキャッシュからクエリキャッシュまで
3. **クエリ最適化** — EXPLAIN ANALYZE の読み方とインデックス設計
4. **クエリオプティマイザ** の内部動作と統計情報
5. **パーティショニング** によるスキャン最適化
6. **バルク処理** の最適化パターン

---

## 1. 接続プールの設計

### 1.1 なぜ接続プールが必要か

```
┌──────────────────────────────────────────────────────┐
│  接続プールなし（NG）                                 │
│                                                      │
│  リクエスト1 ─┐                                      │
│  リクエスト2 ─┼─→ 毎回 TCP + TLS + 認証 → DB        │
│  リクエスト3 ─┘   (50-200ms のオーバーヘッド)        │
│                                                      │
│  問題:                                               │
│  - 接続確立に 50-200ms                               │
│  - DB の max_connections を超えると接続拒否           │
│  - メモリ消費が接続数に比例して増大                   │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  接続プールあり（OK）                                 │
│                                                      │
│  リクエスト1 ─┐     ┌──────────┐                     │
│  リクエスト2 ─┼─→   │ Pool     │ ── conn1 ──→ DB    │
│  リクエスト3 ─┘     │ (再利用) │ ── conn2 ──→ DB    │
│                     └──────────┘ ── conn3 ──→ DB    │
│                                                      │
│  利点:                                               │
│  - 接続確立は初回のみ（0-1ms で再利用）              │
│  - 接続数を制限してDB負荷をコントロール              │
│  - アイドル接続の自動回収                             │
└──────────────────────────────────────────────────────┘
```

### 1.2 接続のライフサイクルとコスト

接続確立の内部プロセスを理解すると、プールの価値が明確になる。

```
┌─────────── DB接続確立の内部プロセス ───────────────┐
│                                                     │
│  Client                          Server             │
│    │                               │                │
│    │── TCP SYN ──────────────────→ │  (1) TCP       │
│    │←──── SYN-ACK ────────────── │      ~1ms       │
│    │── ACK ──────────────────────→ │                │
│    │                               │                │
│    │── ClientHello ──────────────→ │  (2) TLS       │
│    │←── ServerHello + Cert ────── │      ~5-50ms   │
│    │── Key Exchange ─────────────→ │  (1-2 RTT)    │
│    │←── Finished ────────────── │                │
│    │                               │                │
│    │── StartupMessage ───────────→ │  (3) PG Auth   │
│    │←── AuthenticationOk ──────── │      ~5-20ms   │
│    │←── ParameterStatus × N ──── │                │
│    │←── BackendKeyData ────────── │                │
│    │←── ReadyForQuery ─────────── │                │
│    │                               │                │
│    │                  合計: 50-200ms (初回)         │
│    │                  プール再利用: < 0.1ms        │
└─────────────────────────────────────────────────────┘
```

### 1.3 接続プールサイズの計算

```
最適プールサイズの目安:

  pool_size = (CPU cores * 2) + effective_spindle_count

  例: 4コアCPU + SSD(1スピンドル相当)
      pool_size = (4 * 2) + 1 = 9

  ただし、実測ベースで調整が必要:
  ┌─────────────────────────────────────────────┐
  │  接続数   │  レイテンシ  │  スループット     │
  │     5     │    15ms     │    333 req/s      │
  │    10     │    12ms     │    833 req/s  ← 最適│
  │    20     │    14ms     │    714 req/s      │
  │    50     │    25ms     │    400 req/s      │
  └─────────────────────────────────────────────┘
  ※ 増やしすぎるとコンテキストスイッチで悪化
```

### 1.4 接続プールサイズ計算の詳細モデル

```
┌──────── 接続プールサイズの決定要因 ────────────┐
│                                                │
│  1. DBサーバーの処理能力                        │
│     max_connections = (CPU * 2) + spindle       │
│                                                │
│  2. アプリケーション層の構成                    │
│     合計接続数 = pool_size × アプリインスタンス数│
│     → 合計接続数 < max_connections              │
│                                                │
│  3. クエリの特性                                │
│     短いクエリ(< 10ms): 少ないプールで十分      │
│     長いクエリ(> 100ms): より大きなプールが必要  │
│                                                │
│  4. リトルの法則                                │
│     必要接続数 = 到着率 × 平均処理時間          │
│     例: 1000 req/s × 10ms = 10 接続            │
│                                                │
│  計算例:                                        │
│  - DB: 8コアCPU + SSD → max = 17              │
│  - アプリ: 3インスタンス                        │
│  - pool_size/instance = 17 / 3 ≈ 5            │
│  - max_overflow = 2 (バースト対応)             │
└────────────────────────────────────────────────┘
```

### 1.5 各言語での接続プール設定

```python
# Python — SQLAlchemy
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://user:pass@localhost:5432/mydb",
    pool_size=10,           # 常時維持する接続数
    max_overflow=5,         # pool_size 超過時の追加接続数
    pool_timeout=30,        # 接続取得のタイムアウト(秒)
    pool_recycle=1800,      # 接続の再作成間隔(秒) ← コネクションリーク対策
    pool_pre_ping=True,     # 使用前に接続の生存確認
)
```

```typescript
// Node.js — pg (node-postgres)
import { Pool } from 'pg';

const pool = new Pool({
  host: 'localhost',
  port: 5432,
  database: 'mydb',
  user: 'user',
  password: 'pass',
  max: 10,                    // 最大接続数
  idleTimeoutMillis: 30000,   // アイドル接続のタイムアウト
  connectionTimeoutMillis: 5000, // 接続取得のタイムアウト
});

// 使用例
const result = await pool.query('SELECT * FROM users WHERE id = $1', [userId]);
```

```go
// Go — database/sql
import (
    "database/sql"
    _ "github.com/lib/pq"
    "time"
)

db, err := sql.Open("postgres", "postgres://user:pass@localhost:5432/mydb?sslmode=disable")
if err != nil {
    log.Fatal(err)
}

db.SetMaxOpenConns(10)                  // 最大接続数
db.SetMaxIdleConns(5)                   // 最大アイドル接続数
db.SetConnMaxLifetime(30 * time.Minute) // 接続の最大生存時間
db.SetConnMaxIdleTime(5 * time.Minute)  // アイドル接続の最大生存時間
```

```java
// Java — HikariCP (Spring Boot デフォルト)
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;

HikariConfig config = new HikariConfig();
config.setJdbcUrl("jdbc:postgresql://localhost:5432/mydb");
config.setUsername("user");
config.setPassword("pass");
config.setMaximumPoolSize(10);          // 最大プールサイズ
config.setMinimumIdle(5);               // 最小アイドル接続数
config.setIdleTimeout(300000);          // アイドルタイムアウト(ms)
config.setConnectionTimeout(30000);     // 接続取得タイムアウト(ms)
config.setMaxLifetime(1800000);         // 接続の最大生存時間(ms)
config.setLeakDetectionThreshold(60000); // リーク検出閾値(ms)

HikariDataSource ds = new HikariDataSource(config);
```

### 1.6 外部接続プール — pgBouncer

大規模環境では、アプリケーション側のプールに加えて DB 側にも pgBouncer を配置する。

```
┌──────── pgBouncer アーキテクチャ ─────────┐
│                                            │
│  App1 (pool=5) ──┐                         │
│  App2 (pool=5) ──┼─→ pgBouncer ──→ DB     │
│  App3 (pool=5) ──┘   (100→20 に集約)      │
│                                            │
│  モード:                                   │
│  - session:     セッション単位（最も安全）  │
│  - transaction: トランザクション単位（推奨）│
│  - statement:   ステートメント単位（制限多）│
└────────────────────────────────────────────┘
```

```ini
; pgbouncer.ini の設定例
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

; プールモード（transaction が一般的）
pool_mode = transaction

; プールサイズ
default_pool_size = 20
min_pool_size = 5
max_client_conn = 200        ; クライアント最大接続数
max_db_connections = 20      ; DB への最大接続数

; タイムアウト
server_idle_timeout = 600
client_idle_timeout = 0      ; 0 = 無制限
query_timeout = 0
```

```sql
-- pgBouncer の統計確認
-- pgBouncer 管理コンソールに接続
-- psql -p 6432 -U admin pgbouncer

SHOW POOLS;
-- database | user | cl_active | cl_waiting | sv_active | sv_idle | ...

SHOW STATS;
-- database | total_xact_count | total_query_count | avg_xact_time | ...

SHOW CLIENTS;
-- type | user | database | state | addr | port | ...
```

### 1.7 接続プールの監視

```sql
-- PostgreSQL: 現在の接続状態を確認
SELECT
    state,
    COUNT(*) AS connections,
    ROUND(100.0 * COUNT(*) / (SELECT setting::INT FROM pg_settings
        WHERE name = 'max_connections'), 1) AS pct_of_max
FROM pg_stat_activity
GROUP BY state
ORDER BY connections DESC;

-- アイドル接続の詳細（長時間アイドルは問題の兆候）
SELECT
    pid,
    usename,
    application_name,
    client_addr,
    state,
    NOW() - state_change AS idle_duration,
    query
FROM pg_stat_activity
WHERE state = 'idle'
  AND NOW() - state_change > INTERVAL '5 minutes'
ORDER BY idle_duration DESC;

-- 接続待ちの検出
SELECT
    COUNT(*) FILTER (WHERE state = 'active') AS active,
    COUNT(*) FILTER (WHERE state = 'idle') AS idle,
    COUNT(*) FILTER (WHERE state = 'idle in transaction') AS idle_in_tx,
    COUNT(*) FILTER (WHERE wait_event IS NOT NULL) AS waiting,
    (SELECT setting::INT FROM pg_settings WHERE name = 'max_connections') AS max_conn
FROM pg_stat_activity
WHERE backend_type = 'client backend';
```

---

## 2. キャッシュ戦略

### 2.1 キャッシュレイヤーの全体像

```
┌───────────────────────────────────────────────────────┐
│                キャッシュレイヤー                       │
│                                                       │
│  Layer 1: アプリケーション内キャッシュ (L1)            │
│  ┌───────────────────────────────────────┐            │
│  │ インメモリ (HashMap, LRU Cache)       │            │
│  │ TTL: 数秒〜数分 | レイテンシ: < 1ms  │            │
│  │ 容量: 小 (プロセスメモリに制限)       │            │
│  │ 整合性: プロセス間で不整合の可能性    │            │
│  └──────────────────┬────────────────────┘            │
│                     │ Miss                            │
│                     ▼                                 │
│  Layer 2: 分散キャッシュ (L2)                          │
│  ┌───────────────────────────────────────┐            │
│  │ Redis / Memcached                     │            │
│  │ TTL: 数分〜数時間 | レイテンシ: 1-5ms │            │
│  │ 容量: 大 (専用サーバー)               │            │
│  │ 整合性: 全プロセスで共有              │            │
│  └──────────────────┬────────────────────┘            │
│                     │ Miss                            │
│                     ▼                                 │
│  Layer 3: データベースキャッシュ                       │
│  ┌───────────────────────────────────────┐            │
│  │ shared_buffers / Buffer Pool          │            │
│  │ レイテンシ: < 1ms (メモリ内)          │            │
│  │ 自動管理 (LRU/Clock Sweep)            │            │
│  └──────────────────┬────────────────────┘            │
│                     │ Miss                            │
│                     ▼                                 │
│  Layer 4: ディスク                                    │
│  ┌───────────────────────────────────────┐            │
│  │ OS Page Cache → SSD/HDD              │            │
│  │ レイテンシ: 0.1-10ms                  │            │
│  └───────────────────────────────────────┘            │
└───────────────────────────────────────────────────────┘
```

### 2.2 Cache-Aside パターン（最も一般的）

```python
import redis
import json
from sqlalchemy.orm import Session

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def get_user(db: Session, user_id: str) -> dict:
    """Cache-Aside パターンでユーザーを取得"""
    cache_key = f"user:{user_id}"

    # 1. キャッシュを確認
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)  # Cache Hit

    # 2. キャッシュミス → DB から取得
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return None

    user_dict = {"id": str(user.id), "name": user.name, "email": user.email}

    # 3. キャッシュに保存 (TTL: 5分)
    r.setex(cache_key, 300, json.dumps(user_dict))

    return user_dict

def update_user(db: Session, user_id: str, name: str) -> dict:
    """更新時はキャッシュを削除（Write-Invalidate）"""
    user = db.query(User).filter(User.id == user_id).first()
    user.name = name
    db.commit()

    # キャッシュ削除（次回読み取り時に再キャッシュ）
    r.delete(f"user:{user_id}")

    return {"id": str(user.id), "name": user.name}
```

### 2.3 Write-Through パターン

```python
def update_user_write_through(db: Session, user_id: str, name: str) -> dict:
    """Write-Through: DB とキャッシュを同時更新"""
    user = db.query(User).filter(User.id == user_id).first()
    user.name = name
    db.commit()

    # キャッシュも同時に更新（削除ではなく上書き）
    user_dict = {"id": str(user.id), "name": user.name, "email": user.email}
    r.setex(f"user:{user_id}", 300, json.dumps(user_dict))

    return user_dict
```

### 2.4 Write-Behind（Write-Back）パターン

```python
import asyncio
from collections import defaultdict

class WriteBehindCache:
    """Write-Behind: キャッシュに書き込み、非同期でDBに反映"""

    def __init__(self, redis_client, db_session_factory, flush_interval=5):
        self.redis = redis_client
        self.db_factory = db_session_factory
        self.flush_interval = flush_interval
        self.pending_writes = defaultdict(dict)

    async def set(self, key: str, value: dict):
        """キャッシュに即座に書き込み"""
        self.redis.setex(key, 600, json.dumps(value))
        self.pending_writes[key] = value

    async def flush_to_db(self):
        """定期的にDBに書き込み（バッチ処理）"""
        while True:
            await asyncio.sleep(self.flush_interval)
            if not self.pending_writes:
                continue

            writes = dict(self.pending_writes)
            self.pending_writes.clear()

            db = self.db_factory()
            try:
                for key, value in writes.items():
                    # バルクUPSERTでDB負荷を軽減
                    db.execute(
                        """INSERT INTO users (id, name, email)
                           VALUES (:id, :name, :email)
                           ON CONFLICT (id) DO UPDATE
                           SET name = EXCLUDED.name, email = EXCLUDED.email""",
                        value
                    )
                db.commit()
            except Exception as e:
                db.rollback()
                # 失敗した書き込みを戻す
                self.pending_writes.update(writes)
                raise
            finally:
                db.close()
```

### 2.5 キャッシュ無効化パターン

```
┌────────────────────────────────────────────────────────┐
│          キャッシュ無効化 3 パターン                     │
│                                                        │
│  1. TTL ベース (Time-To-Live)                          │
│     SET key value EX 300                               │
│     → 5分後に自動削除                                  │
│     → 最もシンプル、多少の古いデータを許容              │
│                                                        │
│  2. Write-Invalidate (書き込み時削除)                   │
│     UPDATE → DEL cache_key                             │
│     → 更新時にキャッシュ削除、次回読み取りで再構築     │
│     → 整合性が高い                                     │
│                                                        │
│  3. Write-Through (書き込み時更新)                      │
│     UPDATE → SET cache_key new_value                   │
│     → 更新時にキャッシュも同時更新                     │
│     → 読み取り頻度が高い場合に有効                     │
│                                                        │
│  4. Event-Driven (イベント駆動)                        │
│     UPDATE → publish event → subscriber DEL cache      │
│     → DBのCDC(Change Data Capture)やPub/Subで通知     │
│     → マイクロサービス環境に適合                       │
└────────────────────────────────────────────────────────┘
```

### 2.6 Redis キャッシュの高度なパターン

```python
# ===========================
# Lua スクリプトによるアトミック操作
# ===========================
RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, window)
end

if current > limit then
    return 0  -- 拒否
else
    return 1  -- 許可
end
"""

def check_rate_limit(user_id: str, limit: int = 100, window: int = 60) -> bool:
    """スライディングウィンドウによるレート制限"""
    key = f"rate:{user_id}:{int(time.time()) // window}"
    result = r.eval(RATE_LIMIT_SCRIPT, 1, key, limit, window)
    return bool(result)


# ===========================
# Redis Pipeline でバッチ取得
# ===========================
def get_users_batch(user_ids: list[str]) -> list[dict]:
    """パイプラインで複数キーを一括取得（N+1問題の回避）"""
    cache_keys = [f"user:{uid}" for uid in user_ids]

    # 1. パイプラインで一括取得
    pipe = r.pipeline(transaction=False)
    for key in cache_keys:
        pipe.get(key)
    results = pipe.execute()

    users = []
    missing_ids = []

    for uid, cached in zip(user_ids, results):
        if cached:
            users.append(json.loads(cached))
        else:
            missing_ids.append(uid)

    # 2. キャッシュミス分をDBから取得
    if missing_ids:
        db_users = db.query(User).filter(User.id.in_(missing_ids)).all()
        pipe = r.pipeline(transaction=False)
        for user in db_users:
            user_dict = {"id": str(user.id), "name": user.name, "email": user.email}
            users.append(user_dict)
            pipe.setex(f"user:{user.id}", 300, json.dumps(user_dict))
        pipe.execute()

    return users


# ===========================
# キャッシュウォーミング
# ===========================
def warm_cache_on_deploy():
    """デプロイ時にホットデータをプリロード"""
    # アクセス頻度の高いデータをロード
    hot_users = db.query(User).order_by(User.last_login_at.desc()).limit(1000).all()
    pipe = r.pipeline(transaction=False)
    for user in hot_users:
        user_dict = {"id": str(user.id), "name": user.name, "email": user.email}
        pipe.setex(f"user:{user.id}", 600, json.dumps(user_dict))
    pipe.execute()
    print(f"Warmed cache with {len(hot_users)} users")
```

---

## 3. クエリ最適化

### 3.1 EXPLAIN ANALYZE の読み方

```sql
-- スロークエリの例
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2025-01-01'
GROUP BY u.name
ORDER BY order_count DESC
LIMIT 10;

-- 出力例:
-- Limit  (cost=1234.56..1234.58 rows=10 width=40)
--        (actual time=45.123..45.130 rows=10 loops=1)
--   -> Sort  (cost=1234.56..1237.89 rows=1000 width=40)
--            (actual time=45.120..45.125 rows=10 loops=1)
--     Sort Key: (count(o.id)) DESC
--     Sort Method: top-N heapsort  Memory: 26kB
--       -> HashAggregate  (cost=1200.00..1210.00 rows=1000 width=40)
--                         (actual time=44.000..44.500 rows=1000 loops=1)
--           -> Hash Left Join  (cost=100.00..900.00 rows=50000 width=36)
--                              (actual time=5.000..35.000 rows=50000 loops=1)
--               Hash Cond: (o.user_id = u.id)
--               -> Seq Scan on orders o  ← 全件スキャン! 改善ポイント
--                  (actual time=0.010..15.000 rows=100000 loops=1)
--               -> Hash  (cost=80.00..80.00 rows=1000 width=20)
--                  -> Index Scan using idx_users_created_at on users u
--                     Filter: (created_at > '2025-01-01')
--                     (actual time=0.020..2.000 rows=1000 loops=1)
-- Buffers: shared hit=5000 read=200
-- Planning Time: 0.500 ms
-- Execution Time: 45.200 ms
```

```
EXPLAIN ANALYZE の読み方:

  cost=開始コスト..合計コスト
  actual time=開始時間..合計時間 (ms)
  rows=推定行数 vs actual rows=実際の行数

  注目ポイント:
  1. Seq Scan → Index Scan に変更できないか
  2. 推定 rows と actual rows の乖離 → ANALYZE でテーブル統計を更新
  3. Buffers: read が多い → インデックス不足
  4. loops が多い → Nested Loop Join が非効率
```

### 3.2 実行計画ノードの詳解

```
┌───────── EXPLAIN ノード階層 ──────────────────────┐
│                                                    │
│  スキャンノード（葉ノード）:                       │
│  ┌──────────────────────────────────────────┐     │
│  │ Seq Scan        : 全行スキャン            │     │
│  │ Index Scan      : インデックス→テーブル   │     │
│  │ Index Only Scan : インデックスのみ（最速） │     │
│  │ Bitmap Index Scan + Bitmap Heap Scan      │     │
│  │                  : ビットマップ結合スキャン │     │
│  │ TID Scan        : 行ID直接アクセス        │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  結合ノード:                                       │
│  ┌──────────────────────────────────────────┐     │
│  │ Nested Loop   : 外側×内側（小テーブル向け）│    │
│  │ Hash Join     : ハッシュテーブル構築後結合 │     │
│  │ Merge Join    : ソート済みデータの結合     │     │
│  └──────────────────────────────────────────┘     │
│                                                    │
│  集約ノード:                                       │
│  ┌──────────────────────────────────────────┐     │
│  │ HashAggregate : ハッシュで GROUP BY       │     │
│  │ GroupAggregate: ソート済みで GROUP BY      │     │
│  │ Sort          : メモリ or ディスクソート   │     │
│  │ Limit         : 行数制限                  │     │
│  │ Materialize   : 結果のメモリ保持          │     │
│  └──────────────────────────────────────────┘     │
└────────────────────────────────────────────────────┘
```

### 3.3 JOIN アルゴリズムの選択基準

```
┌──────── JOIN アルゴリズム選択 ─────────────┐
│                                            │
│  Nested Loop Join                          │
│  ┌────────────────────────────────────┐   │
│  │ 外側: 小テーブル (数行〜数百行)    │   │
│  │ 内側: インデックスあり             │   │
│  │ コスト: O(N × M/index)            │   │
│  │ 最適: 一方が小さい + インデックス  │   │
│  └────────────────────────────────────┘   │
│                                            │
│  Hash Join                                 │
│  ┌────────────────────────────────────┐   │
│  │ ビルド: 小テーブルのハッシュを構築  │   │
│  │ プローブ: 大テーブルをスキャン      │   │
│  │ コスト: O(N + M)                   │   │
│  │ メモリ: work_mem に依存            │   │
│  │ 最適: 等価結合 + 十分なメモリ      │   │
│  └────────────────────────────────────┘   │
│                                            │
│  Merge Join                                │
│  ┌────────────────────────────────────┐   │
│  │ 両テーブルをソート後にマージ        │   │
│  │ コスト: O(N log N + M log M)       │   │
│  │ 最適: 既にソート済み（インデックス） │   │
│  │ 追加: 範囲結合にも使える           │   │
│  └────────────────────────────────────┘   │
│                                            │
│  選択フロー:                               │
│  Q: 等価結合？                             │
│  ├─ No → Nested Loop or Merge Join        │
│  └─ Yes                                   │
│     Q: 一方が非常に小さい？               │
│     ├─ Yes → Nested Loop                  │
│     └─ No                                 │
│        Q: 両方ソート済み？                 │
│        ├─ Yes → Merge Join                │
│        └─ No → Hash Join                  │
└────────────────────────────────────────────┘
```

### 3.4 インデックス設計

```sql
-- 1. 単一カラムインデックス
CREATE INDEX idx_users_email ON users (email);

-- 2. 複合インデックス（カーディナリティの高いカラムを先に）
CREATE INDEX idx_orders_user_status ON orders (user_id, status);

-- 3. 部分インデックス（条件を絞ってサイズ削減）
CREATE INDEX idx_orders_active ON orders (user_id)
WHERE status = 'active';

-- 4. カバリングインデックス（INCLUDE でインデックスオンリースキャン）
CREATE INDEX idx_users_email_covering ON users (email)
INCLUDE (name, created_at);

-- 5. 式インデックス
CREATE INDEX idx_users_lower_email ON users (LOWER(email));

-- 6. GINインデックス（全文検索、JSONB、配列）
CREATE INDEX idx_products_tags ON products USING GIN (tags);

-- 7. GiSTインデックス（地理空間、範囲型）
CREATE INDEX idx_events_period ON events USING GIST (
    tstzrange(start_at, end_at)
);

-- 8. BRINインデックス（大規模テーブルのタイムスタンプ）
CREATE INDEX idx_logs_created ON logs USING BRIN (created_at)
WITH (pages_per_range = 32);
-- → B-Tree の1/100以下のサイズで、時系列データに有効
```

### 3.5 インデックス選択のフロー

```
┌──────── インデックス種類の選択フロー ─────────┐
│                                                │
│  Q: 検索パターンは？                           │
│  │                                             │
│  ├─ 等価検索 (=) → B-Tree                     │
│  │                                             │
│  ├─ 範囲検索 (<, >, BETWEEN)                   │
│  │  Q: テーブルサイズは？                       │
│  │  ├─ 小〜中 → B-Tree                        │
│  │  └─ 大（数億行）+ 時系列 → BRIN            │
│  │                                             │
│  ├─ LIKE 'prefix%' → B-Tree (前方一致のみ)    │
│  │                                             │
│  ├─ 全文検索 / 配列 / JSONB → GIN            │
│  │                                             │
│  ├─ 地理空間 / 範囲型 → GiST                  │
│  │                                             │
│  └─ 最近傍検索 → SP-GiST or pgvector         │
│                                                │
│  追加の最適化:                                  │
│  - 特定条件のみ → 部分インデックス (WHERE)     │
│  - SELECT に含むカラムも → INCLUDE (カバリング)│
│  - 関数適用 → 式インデックス                   │
└────────────────────────────────────────────────┘
```

### 3.6 スロークエリの検出と対策

```sql
-- PostgreSQL: スロークエリログの有効化
ALTER SYSTEM SET log_min_duration_statement = 100;  -- 100ms以上をログ
ALTER SYSTEM SET log_statement = 'none';             -- 通常クエリはログしない
SELECT pg_reload_conf();

-- 実行中の遅いクエリを確認
SELECT pid, now() - pg_stat_activity.query_start AS duration,
       query, state
FROM pg_stat_activity
WHERE (now() - pg_stat_activity.query_start) > interval '5 seconds'
  AND state != 'idle'
ORDER BY duration DESC;

-- pg_stat_statements による統計分析（要拡張インストール）
-- CREATE EXTENSION pg_stat_statements;
SELECT
    queryid,
    LEFT(query, 80) AS query_preview,
    calls,
    ROUND(total_exec_time::NUMERIC, 2) AS total_time_ms,
    ROUND(mean_exec_time::NUMERIC, 2) AS avg_time_ms,
    ROUND(stddev_exec_time::NUMERIC, 2) AS stddev_ms,
    rows,
    shared_blks_hit + shared_blks_read AS total_blocks,
    CASE WHEN shared_blks_hit + shared_blks_read > 0
         THEN ROUND(100.0 * shared_blks_hit /
              (shared_blks_hit + shared_blks_read), 1)
         ELSE 100
    END AS cache_hit_pct
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- インデックス使用率の確認
SELECT schemaname, tablename, indexname,
       idx_scan as times_used,
       pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;
-- idx_scan = 0 のインデックスは不要な可能性

-- テーブル統計の更新
ANALYZE users;
ANALYZE orders;
```

### 3.7 統計情報とクエリプランナー

```sql
-- テーブルの統計情報を確認
SELECT
    attname,
    n_distinct,          -- ユニーク値の推定数（負値 = 行数に対する割合）
    most_common_vals,    -- 最頻値
    most_common_freqs,   -- 最頻値の出現頻度
    histogram_bounds     -- ヒストグラムの区切り値
FROM pg_stats
WHERE tablename = 'orders' AND attname = 'status';

-- 拡張統計（相関のあるカラム群）
-- PostgreSQL 10+
CREATE STATISTICS stat_orders_user_status (dependencies, ndistinct, mcv)
ON user_id, status FROM orders;
ANALYZE orders;

-- 統計情報の精度を上げる（デフォルト=100、最大=10000）
ALTER TABLE orders ALTER COLUMN status SET STATISTICS 500;
ANALYZE orders;
```

### 3.8 クエリ書き換えの実践パターン

```sql
-- =============================================
-- パターン1: N+1 問題の解決
-- =============================================
-- NG: アプリケーション側のループ（N+1クエリ）
-- for user in users:
--     orders = SELECT * FROM orders WHERE user_id = user.id

-- OK: JOINで一括取得
SELECT u.*, o.id AS order_id, o.total_amount
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2025-01-01';

-- OK: LATERAL JOIN で「各ユーザーの最新3件」
SELECT u.name, recent_orders.*
FROM users u
CROSS JOIN LATERAL (
    SELECT o.id, o.total_amount, o.created_at
    FROM orders o
    WHERE o.user_id = u.id
    ORDER BY o.created_at DESC
    LIMIT 3
) recent_orders
WHERE u.created_at > '2025-01-01';

-- =============================================
-- パターン2: カーソルベースページネーション
-- =============================================
-- NG: OFFSET が大きくなるほど遅くなる
SELECT * FROM orders ORDER BY created_at DESC LIMIT 20 OFFSET 100000;
-- → 100,020行を読んで100,000行を捨てる

-- OK: カーソルベースページネーション
SELECT * FROM orders
WHERE created_at < '2026-01-15T10:30:00Z'  -- 前ページ最後のタイムスタンプ
ORDER BY created_at DESC
LIMIT 20;
-- → インデックスを使って20行だけ読む

-- OK: カーソル + ID（同一タイムスタンプ対応）
SELECT * FROM orders
WHERE (created_at, id) < ('2026-01-15T10:30:00Z', 12345)
ORDER BY created_at DESC, id DESC
LIMIT 20;

-- =============================================
-- パターン3: EXISTS vs IN vs JOIN
-- =============================================
-- 外側が小さく内側が大きい場合: EXISTS が有利
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.id AND o.status = 'active'
);

-- 内側が小さい場合: IN が有利
SELECT * FROM orders
WHERE user_id IN (SELECT id FROM users WHERE role = 'admin');

-- =============================================
-- パターン4: DISTINCT を最適化
-- =============================================
-- NG: 全行ソート後に重複排除
SELECT DISTINCT category FROM products;

-- OK: Loose Index Scan（PostgreSQL 14+ で自動適用される場合あり）
-- 手動で再現する場合:
WITH RECURSIVE categories AS (
    (SELECT category FROM products ORDER BY category LIMIT 1)
    UNION ALL
    SELECT (SELECT p.category FROM products p
            WHERE p.category > c.category
            ORDER BY p.category LIMIT 1)
    FROM categories c
    WHERE c.category IS NOT NULL
)
SELECT category FROM categories WHERE category IS NOT NULL;

-- =============================================
-- パターン5: バッチ UPDATE
-- =============================================
-- NG: 一括で大量更新（ロック時間が長い）
UPDATE orders SET status = 'archived'
WHERE created_at < '2024-01-01';
-- → 100万行ロック、他のトランザクションがブロックされる

-- OK: バッチに分割
DO $$
DECLARE
    batch_size INT := 10000;
    updated INT;
BEGIN
    LOOP
        UPDATE orders SET status = 'archived'
        WHERE id IN (
            SELECT id FROM orders
            WHERE created_at < '2024-01-01'
              AND status != 'archived'
            LIMIT batch_size
            FOR UPDATE SKIP LOCKED  -- ロック競合を回避
        );
        GET DIAGNOSTICS updated = ROW_COUNT;
        EXIT WHEN updated = 0;
        COMMIT;
        PERFORM pg_sleep(0.1);  -- 他のトランザクションに実行機会を与える
    END LOOP;
END $$;
```

---

## 4. データベースキャッシュの内部動作

### 4.1 PostgreSQL の shared_buffers

```
┌──────── PostgreSQL メモリアーキテクチャ ────────┐
│                                                  │
│  shared_buffers (共有バッファ)                    │
│  ┌────────────────────────────────────────┐     │
│  │ デフォルト: 128MB                       │     │
│  │ 推奨: 物理メモリの 25%                  │     │
│  │ 最大効果: 8-16GB 程度で頭打ち           │     │
│  │                                        │     │
│  │ Clock Sweep アルゴリズムで管理          │     │
│  │ → LRU に似ているが、使用カウンタで判定 │     │
│  └────────────────────────────────────────┘     │
│                                                  │
│  work_mem (ワーカーメモリ)                        │
│  ┌────────────────────────────────────────┐     │
│  │ デフォルト: 4MB                         │     │
│  │ 用途: ソート、ハッシュ結合、集約        │     │
│  │ 注意: 接続数 × クエリ数分確保される     │     │
│  │       work_mem=64MB × 100接続 = 6.4GB  │     │
│  └────────────────────────────────────────┘     │
│                                                  │
│  maintenance_work_mem                             │
│  ┌────────────────────────────────────────┐     │
│  │ デフォルト: 64MB                        │     │
│  │ 用途: VACUUM, CREATE INDEX, ALTER TABLE │     │
│  │ 推奨: 512MB - 1GB                      │     │
│  └────────────────────────────────────────┘     │
│                                                  │
│  effective_cache_size                             │
│  ┌────────────────────────────────────────┐     │
│  │ プランナーへのヒント（実際のメモリ割当なし）│  │
│  │ 推奨: 物理メモリの 50-75%               │     │
│  │ → Index Scan の選択に影響する           │     │
│  └────────────────────────────────────────┘     │
└──────────────────────────────────────────────────┘
```

### 4.2 バッファキャッシュの監視

```sql
-- バッファキャッシュのヒット率
SELECT
    datname,
    blks_hit,
    blks_read,
    CASE WHEN blks_hit + blks_read > 0
         THEN ROUND(100.0 * blks_hit / (blks_hit + blks_read), 2)
         ELSE 100
    END AS cache_hit_ratio
FROM pg_stat_database
WHERE datname = current_database();
-- 目標: 99% 以上

-- テーブル別のキャッシュヒット率
SELECT
    schemaname, relname,
    heap_blks_hit,
    heap_blks_read,
    CASE WHEN heap_blks_hit + heap_blks_read > 0
         THEN ROUND(100.0 * heap_blks_hit / (heap_blks_hit + heap_blks_read), 2)
         ELSE 100
    END AS cache_hit_ratio
FROM pg_statio_user_tables
ORDER BY heap_blks_read DESC
LIMIT 20;

-- pg_buffercache 拡張で詳細確認
-- CREATE EXTENSION pg_buffercache;
SELECT
    c.relname,
    COUNT(*) AS buffers,
    pg_size_pretty(COUNT(*) * 8192) AS cached_size,
    ROUND(100.0 * COUNT(*) /
        (SELECT setting::INT FROM pg_settings WHERE name = 'shared_buffers'), 1)
        AS pct_of_shared_buffers
FROM pg_buffercache b
    JOIN pg_class c ON b.relfilenode = pg_relation_filenode(c.oid)
WHERE b.reldatabase = (SELECT oid FROM pg_database WHERE datname = current_database())
GROUP BY c.relname
ORDER BY buffers DESC
LIMIT 20;
```

---

## 5. パーティショニングによる最適化

### 5.1 パーティション戦略

```sql
-- =============================================
-- RANGE パーティション（最も一般的: 日付ベース）
-- =============================================
CREATE TABLE events (
    id          BIGSERIAL,
    event_type  VARCHAR(50),
    payload     JSONB,
    created_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 月別パーティション作成
CREATE TABLE events_2024_01 PARTITION OF events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE events_2024_02 PARTITION OF events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- パーティションの自動作成（pg_partman 拡張）
-- CREATE EXTENSION pg_partman;
-- SELECT partman.create_parent(
--     p_parent_table := 'public.events',
--     p_control := 'created_at',
--     p_type := 'range',
--     p_interval := '1 month',
--     p_premake := 3  -- 3ヶ月先まで事前作成
-- );

-- =============================================
-- LIST パーティション（カテゴリベース）
-- =============================================
CREATE TABLE orders (
    id          BIGSERIAL,
    region      VARCHAR(20) NOT NULL,
    total       DECIMAL(12,2),
    created_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, region)
) PARTITION BY LIST (region);

CREATE TABLE orders_japan PARTITION OF orders
    FOR VALUES IN ('tokyo', 'osaka', 'nagoya');
CREATE TABLE orders_us PARTITION OF orders
    FOR VALUES IN ('new_york', 'san_francisco', 'chicago');
CREATE TABLE orders_eu PARTITION OF orders
    FOR VALUES IN ('london', 'paris', 'berlin');

-- =============================================
-- HASH パーティション（均等分散）
-- =============================================
CREATE TABLE sessions (
    id          UUID NOT NULL,
    user_id     BIGINT NOT NULL,
    data        JSONB,
    expires_at  TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id)
) PARTITION BY HASH (id);

CREATE TABLE sessions_0 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE sessions_1 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE sessions_2 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE sessions_3 PARTITION OF sessions
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);
```

### 5.2 Partition Pruning の確認

```sql
-- Partition Pruning の動作確認
EXPLAIN (ANALYZE)
SELECT * FROM events
WHERE created_at BETWEEN '2024-03-01' AND '2024-03-31';

-- 出力例:
-- Append (actual rows=50000)
--   Subplans Removed: 11        ← 12パーティション中11個が除外された
--   -> Seq Scan on events_2024_03 (actual rows=50000)
--        Filter: (created_at >= '2024-03-01' AND created_at <= '2024-03-31')
```

---

## 6. バルク処理の最適化

```sql
-- =============================================
-- COPY による高速バルクロード（INSERT より 10-100倍高速）
-- =============================================
COPY users (name, email, created_at)
FROM '/tmp/users.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- プログラムからの COPY
-- Python (psycopg2)
-- with cursor.copy("COPY users (name, email) FROM STDIN") as copy:
--     for row in data:
--         copy.write_row(row)

-- =============================================
-- バルクINSERT の最適化
-- =============================================
-- NG: 1行ずつ INSERT
INSERT INTO users (name, email) VALUES ('A', 'a@x.com');
INSERT INTO users (name, email) VALUES ('B', 'b@x.com');
-- ...10000回

-- OK: バルク INSERT (1文で複数行)
INSERT INTO users (name, email) VALUES
    ('A', 'a@x.com'),
    ('B', 'b@x.com'),
    ('C', 'c@x.com');
-- → ネットワークラウンドトリップを削減

-- =============================================
-- 大量ロード時の最適化テクニック
-- =============================================
-- 1. インデックスを一時的に削除
DROP INDEX idx_users_email;

-- 2. 制約チェックを遅延
SET session_replication_role = replica;  -- トリガー無効化

-- 3. バルクロード実行
COPY users FROM '/tmp/bulk_users.csv' WITH (FORMAT csv);

-- 4. インデックスを再作成（CONCURRENTLY で無停止）
CREATE INDEX CONCURRENTLY idx_users_email ON users (email);

-- 5. 制約チェックを復元
SET session_replication_role = DEFAULT;

-- 6. 統計情報を更新
ANALYZE users;

-- =============================================
-- UPSERT（INSERT ON CONFLICT）
-- =============================================
INSERT INTO products (sku, name, price, updated_at)
VALUES ('SKU001', 'Widget', 19.99, NOW()),
       ('SKU002', 'Gadget', 29.99, NOW()),
       ('SKU003', 'Doohickey', 39.99, NOW())
ON CONFLICT (sku)
DO UPDATE SET
    name = EXCLUDED.name,
    price = EXCLUDED.price,
    updated_at = EXCLUDED.updated_at
WHERE products.price != EXCLUDED.price;  -- 実際に変更がある場合のみ更新
```

---

## 7. 比較表

### 7.1 キャッシュ戦略比較

| 戦略 | 整合性 | 書き込み負荷 | 読み取り性能 | 実装複雑度 | ユースケース |
|------|--------|------------|------------|-----------|------------|
| **Cache-Aside** | 中（TTL依存） | 低 | 高（Hit時） | 低 | 一般的な読み取りキャッシュ |
| **Write-Through** | 高 | 高（二重書き込み） | 高 | 中 | 読み取り頻度 >> 書き込み頻度 |
| **Write-Behind** | 中（非同期） | 低（バッチ化） | 高 | 高 | 書き込み頻度が高い |
| **Read-Through** | 中 | 低 | 高 | 中 | ORM 統合キャッシュ |

### 7.2 接続プールライブラリ比較

| ライブラリ | 言語 | プーリング | 接続集約 | 推奨設定 |
|-----------|------|---------|---------|---------|
| **HikariCP** | Java | アプリ側 | なし | max=10, min=5 |
| **pgBouncer** | 外部 | DB側 | あり | transaction mode |
| **PgCat** | 外部 | DB側 | あり | Rust製、シャーディング対応 |
| **SQLAlchemy Pool** | Python | アプリ側 | なし | pool_size=10, max_overflow=5 |
| **node-pg Pool** | Node.js | アプリ側 | なし | max=10 |
| **sqlx::Pool** | Rust | アプリ側 | なし | max_connections=10 |
| **database/sql** | Go | アプリ側 | なし | MaxOpenConns=10 |

### 7.3 インデックス種類比較

| インデックス | 用途 | サイズ | 検索速度 | 更新コスト |
|------------|------|--------|---------|-----------|
| **B-Tree** | 等価・範囲 | 中 | O(log n) | O(log n) |
| **Hash** | 等価のみ | 小 | O(1) | O(1) |
| **GIN** | 全文検索・JSONB・配列 | 大 | O(1)〜 | 高（遅延更新） |
| **GiST** | 地理空間・範囲型 | 中 | O(log n) | O(log n) |
| **SP-GiST** | 階層的データ | 小〜中 | O(log n) | O(log n) |
| **BRIN** | 物理ソート済み大テーブル | 極小 | O(1) | O(1) |

### 7.4 パーティション戦略比較

| 方式 | 用途 | Pruning | メンテナンス | 注意点 |
|------|------|---------|-----------|--------|
| **RANGE** | 時系列データ | WHERE句で期間指定 | 古いパーティションの DROP | 最も一般的 |
| **LIST** | カテゴリ分類 | WHERE句で値指定 | 新カテゴリ追加時にパーティション作成 | 値の網羅が必要 |
| **HASH** | 均等分散 | WHERE句で等価条件 | パーティション数の変更が困難 | 範囲検索は全パーティション走査 |

---

## 8. アンチパターン

### 8.1 OFFSET ベースのページネーション

```sql
-- NG: OFFSET が大きくなるほど遅くなる
SELECT * FROM orders ORDER BY created_at DESC LIMIT 20 OFFSET 100000;
-- → 100,020行を読んで100,000行を捨てる

-- OK: カーソルベースページネーション
SELECT * FROM orders
WHERE created_at < '2026-01-15T10:30:00Z'  -- 前ページ最後のタイムスタンプ
ORDER BY created_at DESC
LIMIT 20;
-- → インデックスを使って20行だけ読む
```

### 8.2 キャッシュの雪崩（Cache Stampede）

```python
# NG: 全てのキャッシュが同時に期限切れ → DB に大量リクエスト
def bad_cache_set(key, value):
    r.setex(key, 3600, value)  # 全て TTL=1時間

# OK: TTL にジッター（ランダム変動）を追加
import random

def good_cache_set(key, value, base_ttl=3600):
    jitter = random.randint(0, 600)  # 0-10分のランダム
    r.setex(key, base_ttl + jitter, value)

# OK: ロックで同時再構築を防止（Mutex パターン）
def get_with_lock(key, rebuild_fn, ttl=3600):
    value = r.get(key)
    if value:
        return json.loads(value)

    lock_key = f"lock:{key}"
    if r.set(lock_key, "1", nx=True, ex=10):  # 10秒ロック
        try:
            value = rebuild_fn()
            r.setex(key, ttl + random.randint(0, 600), json.dumps(value))
            return value
        finally:
            r.delete(lock_key)
    else:
        # 他のプロセスが構築中 → 短時間待って再試行
        import time
        time.sleep(0.1)
        return get_with_lock(key, rebuild_fn, ttl)

# OK: 論理的期限切れ（Probabilistic Early Expiration）
import math

def get_with_early_expiry(key, rebuild_fn, ttl=3600, beta=1.0):
    """値がまだ有効でも、期限切れが近づくと確率的に再構築"""
    data = r.get(key)
    if data:
        cached = json.loads(data)
        remaining_ttl = r.ttl(key)
        # 残りTTLが短いほど再構築の確率が高くなる
        if remaining_ttl > 0:
            delta = ttl - remaining_ttl
            threshold = delta * beta * math.log(random.random())
            if threshold < remaining_ttl:
                return cached["value"]

    # 再構築
    value = rebuild_fn()
    r.setex(key, ttl, json.dumps({"value": value}))
    return value
```

### 8.3 SELECT * の乱用

```sql
-- NG: 不要なカラムも全て取得
SELECT * FROM users WHERE id = 1;
-- → LOBカラムや大きなJSONBも読み込む
-- → Index Only Scan が使えない

-- OK: 必要なカラムのみ
SELECT id, name, email FROM users WHERE id = 1;
-- → カバリングインデックスがあれば Index Only Scan
```

### 8.4 インデックスの過剰作成

```sql
-- NG: 全カラムにインデックス
CREATE INDEX idx_users_name ON users (name);
CREATE INDEX idx_users_email ON users (email);
CREATE INDEX idx_users_age ON users (age);
CREATE INDEX idx_users_city ON users (city);
CREATE INDEX idx_users_name_email ON users (name, email);
CREATE INDEX idx_users_email_name ON users (email, name);
-- → INSERT/UPDATE が遅くなる、ストレージ浪費

-- OK: 実際のクエリパターンに基づいて必要最小限のインデックスを作成
-- 1. 最も頻繁なWHERE句のカラム
-- 2. JOIN条件のカラム（FK）
-- 3. ORDER BY のカラム
-- 4. pg_stat_user_indexes でidx_scan=0のインデックスは定期的に削除
```

---

## 9. エッジケース

### エッジケース1: 接続プールの枯渇

```python
# 問題: 長時間トランザクションがプールを占有
async def process_batch_bad(items):
    """NG: 1つのコネクションで長時間処理"""
    async with pool.acquire() as conn:
        for item in items:  # 10,000件を1接続で逐次処理
            await conn.execute("INSERT INTO results VALUES ($1)", item)
            await external_api_call(item)  # 100ms/call = 合計1000秒

# 解決: バッチを分割し、接続を早期に返却
async def process_batch_good(items):
    """OK: 小バッチで接続を使い回し"""
    for batch in chunks(items, 100):
        async with pool.acquire() as conn:
            async with conn.transaction():
                for item in batch:
                    await conn.execute("INSERT INTO results VALUES ($1)", item)
        # ここで接続がプールに返却される
        await asyncio.gather(*[external_api_call(item) for item in batch])
```

### エッジケース2: キャッシュとトランザクションの不整合

```python
# 問題: DB更新とキャッシュ削除の間にクラッシュ
def update_user_bad(db, user_id, name):
    user = db.query(User).filter(User.id == user_id).first()
    user.name = name
    db.commit()          # ← DBは更新された
    # ↓ ここでクラッシュするとキャッシュが古いまま
    r.delete(f"user:{user_id}")

# 解決: Outbox パターン（DB トランザクションに含める）
def update_user_good(db, user_id, name):
    user = db.query(User).filter(User.id == user_id).first()
    user.name = name
    # キャッシュ無効化イベントもDBに記録
    db.execute(
        "INSERT INTO outbox (event_type, payload) VALUES (:type, :payload)",
        {"type": "cache_invalidate", "payload": json.dumps({"key": f"user:{user_id}"})}
    )
    db.commit()
    # バックグラウンドワーカーがoutboxを処理してキャッシュを削除
```

### エッジケース3: HOT UPDATE の活用

```sql
-- PostgreSQL の HOT (Heap-Only Tuple) UPDATE
-- 条件: 更新カラムがインデックスに含まれていない
-- 効果: インデックスの更新をスキップし、テーブルブロック内で完結

-- HOT UPDATE が効くケース
CREATE TABLE counters (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(50),
    count INTEGER DEFAULT 0
);
CREATE INDEX idx_counters_name ON counters (name);

-- name は変更しない → HOT UPDATE が発動
UPDATE counters SET count = count + 1 WHERE id = 1;

-- HOT UPDATE の状況を確認
SELECT
    relname,
    n_tup_upd,
    n_tup_hot_upd,
    CASE WHEN n_tup_upd > 0
         THEN ROUND(100.0 * n_tup_hot_upd / n_tup_upd, 1)
         ELSE 0
    END AS hot_update_pct
FROM pg_stat_user_tables
WHERE relname = 'counters';
-- hot_update_pct が高いほど効率的
```

---

## 10. 演習問題

### 演習1: 基礎 — EXPLAIN ANALYZE の分析

以下の EXPLAIN ANALYZE の出力から問題点を特定し、改善策を提案せよ。

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT o.id, o.total_amount, u.name
FROM orders o
JOIN users u ON u.id = o.user_id
WHERE o.status = 'pending'
  AND o.created_at > '2025-06-01'
ORDER BY o.created_at DESC
LIMIT 50;

-- Sort  (cost=15000.00..15000.13 rows=50 width=52)
--       (actual time=450.123..450.130 rows=50 loops=1)
--   Sort Key: o.created_at DESC
--   Sort Method: top-N heapsort  Memory: 32kB
--   ->  Hash Join  (cost=200.00..14500.00 rows=5000 width=52)
--                  (actual time=3.000..448.000 rows=5000 loops=1)
--         Hash Cond: (o.user_id = u.id)
--         ->  Seq Scan on orders o  (actual time=0.030..440.000 rows=5000 loops=1)
--               Filter: (status = 'pending' AND created_at > '2025-06-01')
--               Rows Removed by Filter: 995000
--               Buffers: shared read=25000
--         ->  Hash  (cost=150.00..150.00 rows=10000 width=20)
--               ->  Seq Scan on users u  (actual time=0.010..2.000 rows=10000 loops=1)
--               Buffers: shared hit=100
-- Buffers: shared hit=100 read=25000
-- Execution Time: 450.200 ms
```

**解答例**:

問題点:
1. orders テーブルで Seq Scan が発生（100万行中99.5万行がフィルタで除外）
2. Buffers: shared read=25000（キャッシュヒットなし、全てディスク読み取り）
3. 推定rows=5000 と actual rows=5000 は一致しているので統計は正確

改善策:
```sql
-- 複合インデックス（部分インデックス）
CREATE INDEX idx_orders_pending_created
ON orders (created_at DESC)
WHERE status = 'pending';
-- → Seq Scan → Index Scan に改善
-- → Rows Removed が 0 に近づく
-- → 期待される改善: 450ms → 5ms 以下

-- さらに最適化: カバリングインデックス
CREATE INDEX idx_orders_pending_covering
ON orders (created_at DESC)
INCLUDE (total_amount, user_id)
WHERE status = 'pending';
-- → Index Only Scan が可能に
```

### 演習2: 応用 — キャッシュ戦略の設計

以下の要件に対して、適切なキャッシュ戦略を設計せよ。

**要件**: EC サイトの商品詳細ページ
- 商品情報は1日数回程度更新される
- 閲覧数: 1商品あたり 1000 回/日
- 在庫数はリアルタイム性が必要
- 商品画像 URL は変更されない

**解答例**:

```python
# 商品情報: Cache-Aside + TTL 30分 + Write-Invalidate
def get_product(product_id: str) -> dict:
    key = f"product:{product_id}"
    cached = r.get(key)
    if cached:
        return json.loads(cached)

    product = db.query(Product).filter(Product.id == product_id).first()
    data = product.to_dict()
    r.setex(key, 1800, json.dumps(data))  # TTL 30分
    return data

# 在庫数: TTL 短め（10秒）+ 更新時即時反映
def get_stock(product_id: str) -> int:
    key = f"stock:{product_id}"
    cached = r.get(key)
    if cached:
        return int(cached)

    stock = db.query(Inventory.quantity).filter(
        Inventory.product_id == product_id
    ).scalar()
    r.setex(key, 10, str(stock))  # TTL 10秒（ほぼリアルタイム）
    return stock

def purchase_product(product_id: str, quantity: int):
    # DB更新
    db.execute(
        "UPDATE inventory SET quantity = quantity - :qty WHERE product_id = :pid",
        {"qty": quantity, "pid": product_id}
    )
    db.commit()
    # 在庫キャッシュは即座に削除（リアルタイム性）
    r.delete(f"stock:{product_id}")

# 画像URL: 長期キャッシュ（変更されないため）
# → CDN + Cache-Control: max-age=31536000 で対応
```

### 演習3: 発展 — 接続プール設計

以下の環境で最適な接続プール構成を設計せよ。

**環境**:
- DB: PostgreSQL, 16コアCPU, 64GB RAM, max_connections=200
- アプリ: Kubernetes 上に 10 Pod（オートスケール 5-20）
- 平均クエリ時間: 15ms
- ピーク時リクエスト: 5000 req/s
- 長時間クエリ（レポート）: 月に数回、最大30秒

**解答例**:

```
1. DB側の接続上限
   max_connections = 200
   superuser_reserved = 5 (管理用)
   使用可能 = 195

2. pgBouncer を導入（接続集約）
   max_db_connections = 50  (DB への実際の接続)
   max_client_conn = 300    (アプリからの接続を受け付ける)
   pool_mode = transaction  (トランザクション単位で接続を共有)

3. アプリ側プール（各Pod）
   pool_size = 10
   max_overflow = 5
   合計: 10Pod × 15接続 = 150 (pgBouncer のmax_client_conn以下)

4. リトルの法則で検証
   必要接続数 = 5000 req/s × 0.015s = 75 接続
   pgBouncer の max_db_connections = 50
   → transaction mode なら 50 接続で 75 並列を処理可能
   （1リクエスト中のDB利用時間 < クエリ時間の 50%）

5. 長時間レポート対策
   - 別の接続プール（pgBouncer の別セクション）を用意
   - session mode で接続を専有
   - max_db_connections = 3（レポート用は少数）
   - リードレプリカに接続
```

---

## 11. FAQ

### Q1. 接続プールのサイズはどう決める？

**A.** 「CPU コア数 * 2 + ディスクスピンドル数」が初期値の目安（HikariCP 推奨）。ただし実測が最重要。負荷テストで接続数を変えながらスループットとレイテンシを計測し、スループットが最大かつレイテンシが安定するポイントを見つける。多くの場合 10-20 で十分。

### Q2. Redis キャッシュの TTL はどう設定する？

**A.** データの更新頻度と許容できる古さで決める。
- 頻繁に更新（秒単位）: TTL 10-30 秒
- 日次更新: TTL 1-6 時間
- ほぼ不変（マスタデータ）: TTL 24 時間 + Write-Invalidate

重要なのは TTL だけに頼らず、データ更新時のキャッシュ無効化も併用すること。

### Q3. EXPLAIN ANALYZE で「推定行数」と「実際の行数」が大きく乖離する場合は？

**A.** テーブルの統計情報が古い可能性が高い。`ANALYZE テーブル名` で統計を更新する。自動 VACUUM/ANALYZE の設定が不十分な場合、`autovacuum_analyze_threshold` と `autovacuum_analyze_scale_factor` を調整する。相関のある複数カラムの場合、`CREATE STATISTICS` で拡張統計を作成することで改善できる。

### Q4. shared_buffers はどのくらいに設定すべきか？

**A.** 一般的にはサーバーの物理メモリの 25% が推奨。ただし、残りの 75% のうち大部分は OS のページキャッシュとして機能するため、合計で 75% 程度のデータがメモリ上にキャッシュされる。shared_buffers を 8-16GB 以上に増やしても効果は限定的で、むしろ OS のページキャッシュが減るデメリットが生じる場合がある。

### Q5. pgBouncer の pool_mode はどれを選ぶべきか？

**A.** ほとんどの場合 `transaction` モードが最適。`session` モードはPrepared Statement や LISTEN/NOTIFY が必要な場合のみ。`statement` モードはトランザクションが使えないため、特殊な用途（接続集約の最大化）に限る。注意点として、`transaction` モードではセッション変数（SET 文）やPrepared Statementが使えない制限がある。

---

## 12. トラブルシューティング

| 症状 | 原因 | 対策 |
|------|------|------|
| 接続タイムアウトが頻発 | プール枯渇 or max_connections 超過 | プールサイズ見直し、pgBouncer 導入 |
| idle in transaction が多い | トランザクションの閉じ忘れ | idle_in_transaction_session_timeout 設定 |
| キャッシュヒット率が低い | shared_buffers 不足 or ワーキングセット超過 | shared_buffers 増加、不要データのアーカイブ |
| クエリが突然遅くなった | 統計情報の陳腐化 or テーブル膨張 | ANALYZE + VACUUM FULL |
| EXPLAIN の推定行数が大幅に乖離 | 統計情報が古い or 相関カラム | ANALYZE + CREATE STATISTICS |
| インデックスが使われない | データ量が少ない or 型不一致 | クエリの型キャスト確認、SET enable_seqscan=off で検証 |
| Seq Scan が止まらない | random_page_cost が高すぎる(SSD環境) | random_page_cost = 1.1 に設定(SSD向け) |
| VACUUM が追いつかない | 大量UPDATE + autovacuum 設定不足 | autovacuum_vacuum_cost_delay = 2ms に短縮 |
| OOM で落ちる | work_mem × 接続数がメモリ超過 | work_mem を下げる or 接続数を制限 |
| レプリカ遅延 | 重い書き込み + レプリカスペック不足 | wal_level = logical、レプリカスペック増強 |

---

## 13. パフォーマンス最適化チェックリスト

```python
# パフォーマンス改善の優先順位（効果が大きい順）

optimization_checklist = [
    {
        "priority": 1,
        "category": "インデックス",
        "actions": [
            "WHERE句のカラムにインデックスがあるか確認",
            "JOIN条件のカラムにインデックスがあるか確認",
            "ORDER BY のカラムにインデックスがあるか確認",
            "複合インデックスのカラム順序を確認",
            "不要なインデックスの削除（idx_scan=0）",
        ],
    },
    {
        "priority": 2,
        "category": "クエリ書き換え",
        "actions": [
            "SELECT * を必要なカラムのみに変更",
            "サブクエリを JOIN に書き換え",
            "DISTINCT を GROUP BY に書き換え",
            "OFFSET ページネーションをカーソルベースに変更",
            "N+1 クエリを JOIN またはバッチ取得に変更",
        ],
    },
    {
        "priority": 3,
        "category": "テーブル設計",
        "actions": [
            "正規化の見直し（読み取り重視なら非正規化）",
            "パーティショニングの検討（大テーブル）",
            "マテリアライズドビューの利用（集約クエリ）",
            "BRIN インデックスの検討（時系列データ）",
        ],
    },
    {
        "priority": 4,
        "category": "キャッシュ",
        "actions": [
            "Cache-Aside + TTL + Write-Invalidate の導入",
            "キャッシュヒット率の監視",
            "Cache Stampede 対策（TTL ジッター + ロック）",
            "ホットデータのプリウォーミング",
        ],
    },
    {
        "priority": 5,
        "category": "インフラ",
        "actions": [
            "接続プールの最適化（実測ベース）",
            "pgBouncer 導入（多数アプリインスタンス）",
            "リードレプリカの導入（読み取り分散）",
            "SSD 移行と random_page_cost の調整",
        ],
    },
]
```

---

## 14. セキュリティ考慮事項

1. **接続プールの認証情報管理**: 接続文字列にパスワードを直接書かず、環境変数やシークレットマネージャ（AWS Secrets Manager, HashiCorp Vault 等）を使用する。

2. **Redis キャッシュのセキュリティ**: Redis はデフォルトで認証なし。本番環境では `requirepass` を設定し、TLS を有効化する。キャッシュに個人情報を格納する場合はデータの暗号化も検討する。

3. **SQL インジェクション対策**: パフォーマンス最適化のために動的SQLを構築する場合も、必ずパラメータバインドを使用する。

```python
# NG: 文字列結合
query = f"SELECT * FROM users WHERE email = '{email}'"

# OK: パラメータバインド
query = "SELECT * FROM users WHERE email = :email"
db.execute(query, {"email": email})
```

4. **pg_stat_statements のアクセス制御**: クエリ統計にはビジネスロジックが含まれる場合がある。一般ユーザーからの参照を制限する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| **接続プール** | CPUコア数 * 2 が初期値、実測で調整、pool_pre_ping で安定化 |
| **外部プール** | pgBouncer (transaction mode) で接続集約、大規模環境に必須 |
| **キャッシュ** | Cache-Aside + TTL + Write-Invalidate の組み合わせ |
| **キャッシュ雪崩対策** | TTL ジッター + ロックで同時再構築を防止 |
| **クエリ最適化** | EXPLAIN ANALYZE → インデックス追加 → クエリ書き換え |
| **統計情報** | 定期的な ANALYZE + CREATE STATISTICS（相関カラム） |
| **ページネーション** | OFFSET → カーソルベースに変更で大幅改善 |
| **バルク処理** | COPY > マルチINSERT > 単一INSERT |
| **パーティション** | 大テーブルは RANGE パーティションで Pruning を活用 |
| **監視** | pg_stat_statements + キャッシュヒット率 + 接続数の常時監視 |

---

## 次に読むべきガイド

- [03-orm-comparison.md](./03-orm-comparison.md) — ORM 比較と選定基準
- [00-postgresql-features.md](./00-postgresql-features.md) — PostgreSQL 固有の高度な機能
- [03-data-modeling.md](../02-design/03-data-modeling.md) — 分析クエリのためのデータモデリング
- [01-schema-design.md](../02-design/01-schema-design.md) — テーブル設計とインデックス戦略

---

## 参考文献

1. **PostgreSQL 公式ドキュメント** — "Performance Tips" — https://www.postgresql.org/docs/current/performance-tips.html
2. **HikariCP Wiki** — "About Pool Sizing" — https://github.com/brettwooldridge/HikariCP/wiki/About-Pool-Sizing
3. **Redis 公式ドキュメント** — "Caching Patterns" — https://redis.io/docs/manual/patterns/
4. **Use The Index, Luke** — SQL インデックス設計の包括的ガイド — https://use-the-index-luke.com/
5. **pgBouncer 公式ドキュメント** — https://www.pgbouncer.org/
6. **Citus Data** — "Connection Management in PostgreSQL" — https://www.citusdata.com/blog/
7. **Percona** — "PostgreSQL Performance Tuning" — https://www.percona.com/blog/
