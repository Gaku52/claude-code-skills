# パフォーマンスチューニング — 接続プール / キャッシュ / クエリ最適化

> データベースのレスポンスタイムを劇的に改善する実践テクニック。接続プール、キャッシュ戦略、スロークエリ分析を体系的に学ぶ。

---

## この章で学ぶこと

1. **接続プール** の設計と適切なサイズ計算
2. **キャッシュ戦略** — アプリケーションキャッシュからクエリキャッシュまで
3. **クエリ最適化** — EXPLAIN ANALYZE の読み方とインデックス設計

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

### 1.2 接続プールサイズの計算

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

### 1.3 各言語での接続プール設定

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
│  └──────────────────┬────────────────────┘            │
│                     │ Miss                            │
│                     ▼                                 │
│  Layer 2: 分散キャッシュ (L2)                          │
│  ┌───────────────────────────────────────┐            │
│  │ Redis / Memcached                     │            │
│  │ TTL: 数分〜数時間 | レイテンシ: 1-5ms │            │
│  └──────────────────┬────────────────────┘            │
│                     │ Miss                            │
│                     ▼                                 │
│  Layer 3: データベース                                 │
│  ┌───────────────────────────────────────┐            │
│  │ PostgreSQL / MySQL                    │            │
│  │ レイテンシ: 5-100ms                   │            │
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

### 2.3 キャッシュ無効化パターン

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
└────────────────────────────────────────────────────────┘
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

### 3.2 インデックス設計

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
```

### 3.3 スロークエリの検出と対策

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

### 3.4 クエリ最適化チェックリスト

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
        ],
    },
    {
        "priority": 3,
        "category": "テーブル設計",
        "actions": [
            "正規化の見直し（読み取り重視なら非正規化）",
            "パーティショニングの検討",
            "マテリアライズドビューの利用",
        ],
    },
    {
        "priority": 4,
        "category": "インフラ",
        "actions": [
            "接続プールの最適化",
            "リードレプリカの導入",
            "キャッシュ層の追加",
        ],
    },
]
```

---

## 4. 比較表

### 4.1 キャッシュ戦略比較

| 戦略 | 整合性 | 書き込み負荷 | 読み取り性能 | ユースケース |
|------|--------|------------|------------|------------|
| **Cache-Aside** | 中（TTL依存） | 低 | 高（Hit時） | 一般的な読み取りキャッシュ |
| **Write-Through** | 高 | 高（二重書き込み） | 高 | 読み取り頻度 >> 書き込み頻度 |
| **Write-Behind** | 中（非同期） | 低（バッチ化） | 高 | 書き込み頻度が高い |
| **Read-Through** | 中 | 低 | 高 | ORM 統合キャッシュ |

### 4.2 接続プールライブラリ比較

| ライブラリ | 言語 | 特徴 | 推奨設定 |
|-----------|------|------|---------|
| **HikariCP** | Java | 最速、Spring Boot デフォルト | max=10, min=5 |
| **pgBouncer** | 外部 | DB 側のプール、接続集約 | transaction mode |
| **SQLAlchemy Pool** | Python | ORM 統合、柔軟な設定 | pool_size=10, max_overflow=5 |
| **node-pg Pool** | Node.js | シンプル、Promise 対応 | max=10 |
| **sqlx::Pool** | Rust | 非同期ネイティブ | max_connections=10 |
| **database/sql** | Go | 標準ライブラリ | MaxOpenConns=10 |

---

## 5. アンチパターン

### 5.1 OFFSET ベースのページネーション

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

### 5.2 キャッシュの雪崩（Cache Stampede）

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
```

---

## 6. FAQ

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

---

## 7. まとめ

| 項目 | ポイント |
|------|---------|
| **接続プール** | CPUコア数 * 2 が初期値、実測で調整、pool_pre_ping で安定化 |
| **キャッシュ** | Cache-Aside + TTL + Write-Invalidate の組み合わせ |
| **キャッシュ雪崩対策** | TTL ジッター + ロックで同時再構築を防止 |
| **クエリ最適化** | EXPLAIN ANALYZE → インデックス追加 → クエリ書き換え |
| **ページネーション** | OFFSET → カーソルベースに変更で大幅改善 |

---

## 次に読むべきガイド

- [03-orm-comparison.md](./03-orm-comparison.md) — ORM 比較と選定基準
- インデックス設計ガイド — B-Tree、GIN、GiST の使い分け
- Redis 運用ガイド — キャッシュ設計とメモリ管理

---

## 参考文献

1. **PostgreSQL 公式ドキュメント** — "Performance Tips" — https://www.postgresql.org/docs/current/performance-tips.html
2. **HikariCP Wiki** — "About Pool Sizing" — https://github.com/brettwooldridge/HikariCP/wiki/About-Pool-Sizing
3. **Redis 公式ドキュメント** — "Caching Patterns" — https://redis.io/docs/manual/patterns/
4. **Use The Index, Luke** — SQL インデックス設計の包括的ガイド — https://use-the-index-luke.com/
