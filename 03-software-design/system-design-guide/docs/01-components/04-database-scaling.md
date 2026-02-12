# DBスケーリング

> データベースの負荷増大に対するスケーリング戦略として、レプリケーション・シャーディング・パーティショニングの設計原則と実装手法を、具体的なコード例とトレードオフ分析を通じて解説する

## この章で学ぶこと

1. **垂直スケーリング vs 水平スケーリング** --- スケールアップの限界と水平分散の設計判断
2. **レプリケーション戦略** --- マスター/レプリカ構成、読み書き分離、レプリケーションラグへの対処
3. **シャーディングとパーティショニング** --- シャードキーの選定、データ分散アルゴリズム、ホットスポット回避
4. **段階的スケーリング** --- 負荷レベルに応じた最適なスケーリング手法の選定
5. **運用と監視** --- レプリケーションラグ監視、リシャーディング、バックアップ戦略

## 前提知識

| トピック | 必要レベル | 参照先 |
|---------|-----------|--------|
| SQL 基礎 | 中級 | [プログラミング基礎](../../02-programming/) |
| トランザクション (ACID) | 基礎 | [CAP 定理](../00-fundamentals/03-cap-theorem.md) |
| インデックス | 基礎 | データベース基礎 |
| キャッシング | 基礎 | [キャッシング](./01-caching.md) |

---

## 0. WHY --- なぜ DB スケーリングが必要か

### 0.1 単一 DB の限界

```
単一 PostgreSQL サーバーの現実的な限界:

  ┌───────────────────────────────────────────┐
  │  ハードウェア上限                          │
  │  ├─ CPU: 128コアまで                      │
  │  ├─ RAM: 2TB まで                         │
  │  ├─ ストレージ IOPS: ~100,000             │
  │  └─ ネットワーク: 25 Gbps                 │
  │                                           │
  │  性能限界 (チューニング済み):               │
  │  ├─ 書き込み: ~50,000 TPS                 │
  │  ├─ 読み取り: ~200,000 QPS                │
  │  ├─ テーブルサイズ: ~10億行で性能劣化      │
  │  └─ 同時接続: ~5,000                      │
  │                                           │
  │  単一障害点 (SPOF):                        │
  │  ├─ サーバー故障 → 全サービス停止          │
  │  └─ RTO: 数分〜数時間 (バックアップ復旧)   │
  └───────────────────────────────────────────┘

  月間 1億リクエスト超で性能問題が顕在化する目安
```

### 0.2 スケーリングの定量的効果

| 指標 | 単一 DB | Read Replica x3 | シャーディング x4 |
|------|--------|-----------------|----------------|
| 読み取り QPS | 200K | 800K | 800K |
| 書き込み TPS | 50K | 50K (変わらず) | 200K |
| データ容量 | ~10TB | ~10TB (同じ) | ~40TB |
| 可用性 | 99.9% | 99.99% | 99.99% |
| フェイルオーバー時間 | 数分 | 数秒 (自動) | 数秒 (シャード単位) |
| 運用複雑度 | 低 | 中 | 高 |

---

## 1. スケーリング戦略の全体像

### 1.1 スケールアップ vs スケールアウト

```
【スケールアップ (垂直)】
  +--------+          +============+
  | DB     |   --->   || DB        ||
  | 4CPU   |          || 64CPU     ||
  | 16GB   |          || 512GB     ||
  +--------+          +============+
  限界: ハードウェア上限、単一障害点、コスト非線形増大
  利点: アプリケーション変更不要

【スケールアウト (水平)】
  +--------+          +------+ +------+ +------+ +------+
  | DB     |   --->   | DB 1 | | DB 2 | | DB 3 | | DB 4 |
  | 全データ|          | A-F  | | G-L  | | M-R  | | S-Z  |
  +--------+          +------+ +------+ +------+ +------+
  メリット: 理論上無限にスケール、耐障害性
  デメリット: アプリケーション層の変更が必要
```

### 1.2 段階的スケーリングロードマップ

```
フェーズ 0: 最適化 (0円〜)
  App --> [Primary DB + インデックス最適化 + クエリチューニング]
  月間 ~100万リクエスト
  ↓ 「まだ単一 DB で頑張れるか？」

フェーズ 1: コネクションプーリング
  App --> [PgBouncer] --> [Primary DB]
  月間 ~500万リクエスト
  ↓ 「読み取りが書き込みの 10倍以上か？」

フェーズ 2: 読み書き分離 (Read Replica)
  App --> [Primary] (Write)
      --> [Replica 1] (Read)
      --> [Replica 2] (Read)
  月間 ~3000万リクエスト
  ↓ 「キャッシュで読み取り負荷を吸収できるか？」

フェーズ 3: キャッシュ層追加
  App --> [Redis Cache] --> [Primary / Replicas]
  月間 ~1億リクエスト
  ↓ 「テーブルが大きすぎないか？」

フェーズ 4: パーティショニング
  App --> [Primary (パーティションテーブル)]
  月間 ~3億リクエスト
  ↓ 「書き込みが単一 Primary の限界を超えたか？」

フェーズ 5: シャーディング
  App --> [Router] --> [Shard 0] [Shard 1] [Shard 2] ...
  月間 ~10億リクエスト以上
```

### 1.3 スケーリング判断のフローチャート

```
DB 性能問題が発生
  │
  ├─ クエリが遅い？
  │   └─ EXPLAIN ANALYZE → インデックス追加 / クエリ書き換え
  │
  ├─ コネクション数が上限？
  │   └─ PgBouncer / ProxySQL 導入
  │
  ├─ 読み取り負荷が高い？
  │   ├─ キャッシュで対応可能？ → Redis 導入
  │   └─ キャッシュ不適 → Read Replica 追加
  │
  ├─ テーブルが巨大 (数億行)？
  │   └─ パーティショニング (単一DB内で分割)
  │
  ├─ 書き込み負荷が高い？
  │   ├─ バッチ/バルク化で対応可能？ → アプリ最適化
  │   └─ 対応不可 → シャーディング
  │
  └─ ストレージ容量不足？
      └─ パーティショニング + 古いデータのアーカイブ
```

---

## 2. レプリケーション

### 2.1 マスター/レプリカ構成

```
                      +------------------+
                      |   Primary (RW)   |
                      |   (書き込み専用)   |
                      +--------+---------+
                               |
                   WAL / Binlog ストリーム
                     +---------|--------+
                     |         |        |
               +-----v--+ +---v----+ +-v-------+
               |Replica 1| |Replica2| |Replica 3|
               | (Read)  | | (Read) | |  (Read) |
               +---------+ +--------+ +---------+
               | 同期/非同期レプリケーションの選択
               |
               | 同期: データ損失ゼロ、レイテンシ増加
               | 非同期: 高スループット、ラグ発生
               | 半同期: 1台は同期、他は非同期 (推奨)
```

### 2.2 レプリケーション方式の比較

| 方式 | データ損失 | 書き込みレイテンシ | スループット | ユースケース |
|------|----------|------------------|------------|------------|
| 同期レプリケーション | なし | 高 (2-10ms追加) | 低 | 金融、決済 |
| 非同期レプリケーション | あり (ラグ分) | なし | 高 | 一般的なWebアプリ |
| 半同期レプリケーション | 最小 | 中 (1-5ms追加) | 中〜高 | 推奨デフォルト |
| マルチマスター | 競合リスク | 変動 | 高 | グローバル分散 |

### 2.3 読み書き分離の実装

```python
# SQLAlchemy での読み書き分離 (Python)
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import random
import time

class ReadWriteSplitter:
    """読み書き分離を管理するクラス"""

    def __init__(
        self,
        write_dsn: str,
        read_dsns: list[str],
        replication_lag_threshold: float = 5.0,
    ):
        # 書き込み用 Primary
        self.write_engine = create_engine(
            write_dsn,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,  # コネクション健全性チェック
            pool_recycle=1800,   # 30分でコネクション再接続
        )

        # 読み取り用 Replicas
        self.read_engines = [
            create_engine(
                dsn,
                pool_size=30,
                max_overflow=15,
                pool_pre_ping=True,
            )
            for dsn in read_dsns
        ]

        self.replication_lag_threshold = replication_lag_threshold
        self._recent_writes: dict[str, float] = {}  # {entity_key: timestamp}

    @contextmanager
    def write_session(self) -> Session:
        """書き込み用セッション（常に Primary）"""
        session = sessionmaker(bind=self.write_engine)()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def read_session(self, entity_key: str = None) -> Session:
        """
        読み取り用セッション
        - 直近の書き込みがある場合は Primary から読む (Read-your-writes)
        - それ以外は Replica からランダムに選択
        """
        use_primary = False

        if entity_key:
            last_write = self._recent_writes.get(entity_key, 0)
            use_primary = (time.time() - last_write) < self.replication_lag_threshold

        if use_primary:
            engine = self.write_engine
        else:
            engine = random.choice(self.read_engines)

        session = sessionmaker(bind=engine)()
        try:
            yield session
        finally:
            session.close()

    def record_write(self, entity_key: str):
        """書き込みを記録（Read-your-writes 用）"""
        self._recent_writes[entity_key] = time.time()

        # 古い記録を定期的にクリーンアップ
        cutoff = time.time() - self.replication_lag_threshold * 2
        self._recent_writes = {
            k: v for k, v in self._recent_writes.items() if v > cutoff
        }

# 使用例
db = ReadWriteSplitter(
    write_dsn='postgresql://user:pass@primary:5432/myapp',
    read_dsns=[
        'postgresql://user:pass@replica-1:5432/myapp',
        'postgresql://user:pass@replica-2:5432/myapp',
        'postgresql://user:pass@replica-3:5432/myapp',
    ],
    replication_lag_threshold=5.0,
)

# 書き込み
def create_order(user_id: str, items: list) -> dict:
    with db.write_session() as session:
        order = Order(user_id=user_id, items=items, status="pending")
        session.add(order)
        session.flush()  # ID を取得
        db.record_write(f"user:{user_id}")
        return {"order_id": order.id}

# 読み取り (書き込み直後は Primary から読む)
def get_user_orders(user_id: str) -> list:
    with db.read_session(entity_key=f"user:{user_id}") as session:
        return session.query(Order).filter_by(user_id=user_id).all()
```

### 2.4 レプリケーションラグの監視

```python
"""レプリケーションラグの監視と対策"""
from dataclasses import dataclass
from datetime import datetime
import psycopg2

@dataclass
class ReplicationStatus:
    replica_host: str
    lag_bytes: int
    lag_seconds: float
    state: str  # streaming, catchup, startup
    is_healthy: bool

class ReplicationMonitor:
    """PostgreSQL レプリケーションラグ監視"""

    LAG_WARNING_SECONDS = 5.0
    LAG_CRITICAL_SECONDS = 30.0

    def __init__(self, primary_dsn: str, replica_dsns: list[str]):
        self.primary_dsn = primary_dsn
        self.replica_dsns = replica_dsns

    def check_primary_status(self) -> list[dict]:
        """Primary からレプリカの状態を確認"""
        conn = psycopg2.connect(self.primary_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT
                        client_addr,
                        state,
                        sent_lsn,
                        write_lsn,
                        flush_lsn,
                        replay_lsn,
                        pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes,
                        reply_time
                    FROM pg_stat_replication
                    ORDER BY client_addr;
                """)
                return [
                    {
                        'host': row[0],
                        'state': row[1],
                        'lag_bytes': row[6],
                        'reply_time': row[7],
                    }
                    for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def check_replica_lag(self, replica_dsn: str) -> ReplicationStatus:
        """Replica 側からラグを確認"""
        conn = psycopg2.connect(replica_dsn)
        try:
            with conn.cursor() as cur:
                # レプリケーションラグを秒数で取得
                cur.execute("""
                    SELECT
                        CASE
                            WHEN pg_last_wal_receive_lsn() = pg_last_wal_replay_lsn()
                            THEN 0
                            ELSE EXTRACT(EPOCH FROM now() - pg_last_xact_replay_timestamp())
                        END AS lag_seconds;
                """)
                lag_seconds = cur.fetchone()[0] or 0

                # レプリケーションの状態
                cur.execute("SELECT pg_is_in_recovery();")
                is_replica = cur.fetchone()[0]

                return ReplicationStatus(
                    replica_host=replica_dsn.split('@')[1].split(':')[0],
                    lag_bytes=0,
                    lag_seconds=lag_seconds,
                    state='streaming' if is_replica else 'primary',
                    is_healthy=lag_seconds < self.LAG_WARNING_SECONDS,
                )
        finally:
            conn.close()

    def get_all_replica_status(self) -> list[ReplicationStatus]:
        """全レプリカのステータスを取得"""
        statuses = []
        for dsn in self.replica_dsns:
            try:
                status = self.check_replica_lag(dsn)
                statuses.append(status)
            except Exception as e:
                statuses.append(ReplicationStatus(
                    replica_host=dsn.split('@')[1].split(':')[0],
                    lag_bytes=0,
                    lag_seconds=-1,
                    state='unreachable',
                    is_healthy=False,
                ))
        return statuses

    def should_use_primary_for_reads(
        self, statuses: list[ReplicationStatus]
    ) -> bool:
        """全レプリカがクリティカルラグの場合、Primary からの読み取りに切り替え"""
        healthy = [s for s in statuses if s.is_healthy]
        return len(healthy) == 0

# Prometheus メトリクス公開用
# pg_replication_lag_seconds{replica="replica-1"} 0.5
# pg_replication_lag_seconds{replica="replica-2"} 1.2
# pg_replication_lag_seconds{replica="replica-3"} 0.8
```

### 2.5 コネクションプーリング

```python
"""PgBouncer / コネクションプーリングの設計"""
from dataclasses import dataclass

@dataclass
class ConnectionPoolConfig:
    """コネクションプール設計の指針"""

    # PgBouncer 設定例
    pgbouncer_config = """
    [databases]
    myapp_write = host=primary port=5432 dbname=myapp
    myapp_read = host=replica-1 port=5432 dbname=myapp
                 host=replica-2 port=5432 dbname=myapp

    [pgbouncer]
    listen_addr = 0.0.0.0
    listen_port = 6432

    # プーリングモード:
    # session: セッション単位 (PREPARE 対応)
    # transaction: トランザクション単位 (推奨、最もコネクション効率が良い)
    # statement: ステートメント単位 (非推奨)
    pool_mode = transaction

    # コネクション数の設計:
    # アプリ側: pool_size * app_instances = 最大クライアント接続数
    # DB側: max_connections = PgBouncer の server_pool_size
    max_client_conn = 1000        # アプリからの最大接続
    default_pool_size = 50        # DB への同時接続数
    reserve_pool_size = 10        # バースト用予備
    reserve_pool_timeout = 3      # 予備プール使用前の待機秒数

    # ヘルスチェック
    server_check_query = select 1
    server_check_delay = 10
    server_connect_timeout = 5
    server_login_retry = 3

    # タイムアウト
    query_timeout = 30            # クエリタイムアウト
    client_idle_timeout = 600     # アイドルクライアント切断
    """

    # コネクション数の計算式
    @staticmethod
    def calculate_pool_size(
        app_instances: int,
        connections_per_instance: int,
        db_max_connections: int = 200,
    ) -> dict:
        """最適なプールサイズを計算"""
        total_app_connections = app_instances * connections_per_instance

        # PgBouncer の server_pool_size は DB の max_connections の 50-70%
        server_pool = int(db_max_connections * 0.6)

        # 多重化率: アプリ接続 / DB 接続
        multiplexing_ratio = total_app_connections / server_pool

        return {
            'app_instances': app_instances,
            'connections_per_instance': connections_per_instance,
            'total_app_connections': total_app_connections,
            'pgbouncer_server_pool': server_pool,
            'db_max_connections': db_max_connections,
            'multiplexing_ratio': f"{multiplexing_ratio:.1f}x",
            'recommendation': (
                'OK' if multiplexing_ratio < 20
                else 'WARNING: 多重化率が高すぎる。DB接続数の増加を検討'
            ),
        }

# 例: 10 アプリインスタンス x 20 接続/インスタンス
result = ConnectionPoolConfig.calculate_pool_size(
    app_instances=10,
    connections_per_instance=20,
    db_max_connections=200,
)
# 結果:
# total_app_connections: 200
# pgbouncer_server_pool: 120
# multiplexing_ratio: 1.7x (良好)
```

---

## 3. シャーディング

### 3.1 シャーディング戦略の比較

| 戦略 | 説明 | メリット | デメリット | ユースケース |
|------|------|---------|-----------|------------|
| レンジベース | キーの範囲で分割 (A-F, G-L...) | シンプル、範囲クエリが容易 | データ偏り、ホットスポット | 時系列データ (月次パーティション) |
| ハッシュベース | hash(key) % N で分割 | 均等分散 | 範囲クエリ不可、リシャーディング困難 | ユーザーデータ |
| コンシステントハッシュ | ハッシュリング上で分割 | ノード追加時の移動データ最小 | 実装が複雑、負荷偏り可能 | 大規模分散システム |
| ディレクトリベース | ルックアップテーブルで管理 | 柔軟なマッピング | テーブルがSPOF、追加レイテンシ | マルチテナント |
| ジオベース | 地理的に分割 | 低レイテンシ | データ偏り | グローバルサービス |

### 3.2 コンシステントハッシュの仕組み

```
          コンシステントハッシュリング

              0 (= 2^32)
              |
       Shard C ●-------- ● Shard A
             /              \
           /     Key X →      \
          |     (Shard A に配置)  |
          |                      |
           \                  /
            \   ● Shard B   /
              +----●------+
                 Key Y →
              (Shard B に配置)

  仮想ノード: 各シャードを 100〜200 の仮想ノードに分割
  → 物理ノード数が少なくてもハッシュ空間上で均等に分散

  ノード追加時: Shard D を追加
  → Shard A から一部のキーのみ Shard D に移動
  → 他のシャードは影響を受けない
  → 移動データ量: 全体の 1/N (N = シャード数)
```

### 3.3 シャーディングの実装

```python
# ハッシュベースシャーディング (Python)
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from sqlalchemy import create_engine, text
from contextlib import contextmanager

@dataclass
class ShardConfig:
    shard_id: int
    dsn: str
    is_primary: bool = True
    replicas: list[str] = field(default_factory=list)

class ShardRouter:
    """シャードルーターの実装"""

    def __init__(self, shard_configs: list[ShardConfig]):
        self.shards: dict[int, ShardConfig] = {}
        self.engines: dict[int, any] = {}
        self.replica_engines: dict[int, list] = {}

        for config in shard_configs:
            self.shards[config.shard_id] = config
            self.engines[config.shard_id] = create_engine(
                config.dsn,
                pool_size=20,
                pool_pre_ping=True,
            )
            self.replica_engines[config.shard_id] = [
                create_engine(dsn, pool_size=30, pool_pre_ping=True)
                for dsn in config.replicas
            ]

        self.shard_count = len(shard_configs)

    def get_shard_id(self, shard_key: str) -> int:
        """シャードキーからシャード ID を決定"""
        hash_value = int(hashlib.sha256(
            str(shard_key).encode()
        ).hexdigest(), 16)
        return hash_value % self.shard_count

    @contextmanager
    def write_connection(self, shard_key: str):
        """書き込み用コネクション (Primary)"""
        shard_id = self.get_shard_id(shard_key)
        engine = self.engines[shard_id]
        conn = engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def read_connection(self, shard_key: str):
        """読み取り用コネクション (Replica 優先)"""
        shard_id = self.get_shard_id(shard_key)
        replicas = self.replica_engines[shard_id]

        if replicas:
            engine = random.choice(replicas)
        else:
            engine = self.engines[shard_id]

        conn = engine.connect()
        try:
            yield conn
        finally:
            conn.close()

    def execute_on_shard(
        self, shard_key: str, query: str, params: dict = None
    ) -> list:
        """特定シャードでクエリ実行"""
        with self.write_connection(shard_key) as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()

    def scatter_gather(
        self, query: str, params: dict = None,
        sort_key: str = None, limit: int = None,
    ) -> list:
        """
        全シャードにクエリを実行して結果を集約 (Scatter-Gather)
        注意: N個のシャードに並列クエリ → レイテンシは最遅シャードに依存
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []

        def query_shard(shard_id: int):
            engine = self.engines[shard_id]
            with engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return result.fetchall()

        # 並列実行
        with ThreadPoolExecutor(max_workers=self.shard_count) as executor:
            futures = {
                executor.submit(query_shard, sid): sid
                for sid in self.engines.keys()
            }
            for future in as_completed(futures):
                shard_id = futures[future]
                try:
                    rows = future.result()
                    results.extend(rows)
                except Exception as e:
                    print(f"Shard {shard_id} query failed: {e}")

        # ソートとリミット
        if sort_key:
            results.sort(key=lambda r: r[sort_key])
        if limit:
            results = results[:limit]

        return results

# 使用例
router = ShardRouter([
    ShardConfig(
        shard_id=0,
        dsn='postgresql://user:pass@shard0-primary:5432/myapp',
        replicas=['postgresql://user:pass@shard0-replica:5432/myapp'],
    ),
    ShardConfig(
        shard_id=1,
        dsn='postgresql://user:pass@shard1-primary:5432/myapp',
        replicas=['postgresql://user:pass@shard1-replica:5432/myapp'],
    ),
    ShardConfig(
        shard_id=2,
        dsn='postgresql://user:pass@shard2-primary:5432/myapp',
        replicas=['postgresql://user:pass@shard2-replica:5432/myapp'],
    ),
])

# ユーザー ID をシャードキーとして使用
user_id = 'user-12345'
shard_id = router.get_shard_id(user_id)  # → 例: 1
print(f"User {user_id} → Shard {shard_id}")
```

### 3.4 シャードキー設計の原則

```python
"""シャードキー設計の判断フレームワーク"""
from dataclasses import dataclass

@dataclass
class ShardKeyEvaluation:
    """シャードキーの評価基準"""

    key_name: str
    cardinality: str       # "高" / "中" / "低"
    distribution: str      # "均等" / "偏り" / "ホットスポット"
    query_isolation: str   # "高" / "中" / "低"
    join_locality: str     # "高" / "中" / "低"
    verdict: str           # "推奨" / "条件付き" / "非推奨"

SHARD_KEY_EVALUATIONS = [
    ShardKeyEvaluation(
        key_name="user_id",
        cardinality="高",
        distribution="均等 (UUIDの場合)",
        query_isolation="高 (ユーザー単位のクエリが多い場合)",
        join_locality="高 (ユーザーのデータが同一シャード)",
        verdict="推奨: 最も一般的で安全なシャードキー",
    ),
    ShardKeyEvaluation(
        key_name="tenant_id",
        cardinality="中",
        distribution="偏りやすい (大テナントの存在)",
        query_isolation="高 (テナント単位の完全分離)",
        join_locality="高",
        verdict="推奨: マルチテナント SaaS の標準",
    ),
    ShardKeyEvaluation(
        key_name="created_at (日時)",
        cardinality="高",
        distribution="ホットスポット (最新データに集中)",
        query_isolation="高 (時間範囲クエリ)",
        join_locality="低",
        verdict="非推奨: 書き込みが最新シャードに集中",
    ),
    ShardKeyEvaluation(
        key_name="auto_increment_id",
        cardinality="高",
        distribution="ホットスポット (最新IDに集中)",
        query_isolation="低",
        join_locality="低",
        verdict="非推奨: 書き込みが最新シャードに集中",
    ),
    ShardKeyEvaluation(
        key_name="country_code",
        cardinality="低 (~200)",
        distribution="偏り (US/JP/CN に集中)",
        query_isolation="高",
        join_locality="高",
        verdict="条件付き: 地理分散が目的なら有効",
    ),
    ShardKeyEvaluation(
        key_name="compound (user_id + order_date)",
        cardinality="高",
        distribution="均等",
        query_isolation="高 (ユーザー+時間範囲クエリ)",
        join_locality="中",
        verdict="推奨: 複合キーで均等分散と範囲クエリを両立",
    ),
]

# シャードキー選定の原則:
# 1. カーディナリティが高い (ユニーク値が多い)
# 2. アクセスパターンに一致 (最頻クエリの WHERE 句に含まれる)
# 3. 均等に分散される (ホットスポットを避ける)
# 4. 関連データが同一シャードに配置される (JOIN の局所性)
# 5. 将来のデータ増加でも偏りが生じない
```

### 3.5 PostgreSQL テーブルパーティショニング

```sql
-- PostgreSQL: 範囲パーティショニング
CREATE TABLE orders (
    id          BIGSERIAL,
    user_id     BIGINT NOT NULL,
    amount      DECIMAL(10, 2) NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 月別パーティション (自動作成スクリプトで管理)
CREATE TABLE orders_2026_01 PARTITION OF orders
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE orders_2026_02 PARTITION OF orders
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE orders_2026_03 PARTITION OF orders
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

-- パーティションごとにインデックス (各パーティションに自動的に作成)
CREATE INDEX idx_orders_user_id ON orders (user_id);
CREATE INDEX idx_orders_status ON orders (status);

-- クエリ: PostgreSQL が自動的に該当パーティションのみスキャン
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE created_at >= '2026-02-01' AND created_at < '2026-03-01'
  AND user_id = 12345;
-- → orders_2026_02 のみスキャン（パーティションプルーニング）

-- ハッシュパーティショニング (均等分散)
CREATE TABLE user_events (
    id         BIGSERIAL,
    user_id    BIGINT NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload    JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (id, user_id)
) PARTITION BY HASH (user_id);

CREATE TABLE user_events_0 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE user_events_1 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE user_events_2 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE user_events_3 PARTITION OF user_events
    FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- パーティション管理の自動化
CREATE OR REPLACE FUNCTION create_monthly_partition()
RETURNS void AS $$
DECLARE
    partition_date DATE;
    partition_name TEXT;
    start_date TEXT;
    end_date TEXT;
BEGIN
    -- 3ヶ月先まで事前作成
    FOR i IN 0..3 LOOP
        partition_date := date_trunc('month', NOW()) + (i || ' months')::interval;
        partition_name := 'orders_' || to_char(partition_date, 'YYYY_MM');
        start_date := to_char(partition_date, 'YYYY-MM-DD');
        end_date := to_char(partition_date + '1 month'::interval, 'YYYY-MM-DD');

        -- パーティションが存在しなければ作成
        IF NOT EXISTS (
            SELECT 1 FROM pg_class WHERE relname = partition_name
        ) THEN
            EXECUTE format(
                'CREATE TABLE %I PARTITION OF orders FOR VALUES FROM (%L) TO (%L)',
                partition_name, start_date, end_date
            );
            RAISE NOTICE 'Created partition: %', partition_name;
        END IF;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- 古いパーティションのアーカイブ
CREATE OR REPLACE FUNCTION archive_old_partitions(retention_months INT)
RETURNS void AS $$
DECLARE
    partition_name TEXT;
    cutoff_date DATE;
BEGIN
    cutoff_date := date_trunc('month', NOW()) - (retention_months || ' months')::interval;

    FOR partition_name IN
        SELECT tablename FROM pg_tables
        WHERE tablename LIKE 'orders_____'
        AND tablename < 'orders_' || to_char(cutoff_date, 'YYYY_MM')
    LOOP
        -- 1. S3 にエクスポート (pg_dump)
        -- 2. パーティションをデタッチ
        EXECUTE format('ALTER TABLE orders DETACH PARTITION %I', partition_name);
        -- 3. テーブルを削除
        EXECUTE format('DROP TABLE %I', partition_name);
        RAISE NOTICE 'Archived and dropped: %', partition_name;
    END LOOP;
END;
$$ LANGUAGE plpgsql;
```

---

## 4. リシャーディング

### 4.1 リシャーディングの課題と戦略

```
リシャーディング: シャード数の変更 (例: 4 → 8 シャード)

  方法 1: ダウンタイムあり (最もシンプル)
  ┌─────────────────────────────────────────────┐
  │ 1. アプリケーションを停止                     │
  │ 2. 全データを新しいシャード構成にコピー        │
  │ 3. ルーティングを更新                         │
  │ 4. アプリケーションを再起動                   │
  │                                              │
  │ 所要時間: データ量に依存 (数時間〜数日)        │
  │ リスク: 低 (オフラインで安全に実行)            │
  └─────────────────────────────────────────────┘

  方法 2: オンラインリシャーディング (ゼロダウンタイム)
  ┌─────────────────────────────────────────────┐
  │ 1. 新シャードを追加                           │
  │ 2. ダブルライト: 旧+新シャードに同時書き込み  │
  │ 3. バックグラウンドで旧→新にデータ移行        │
  │ 4. 整合性検証                                 │
  │ 5. 読み取りを新シャードに切り替え              │
  │ 6. 旧シャードの書き込みを停止                 │
  │                                              │
  │ 所要時間: 数日〜数週間                        │
  │ リスク: 高 (データ整合性の維持が困難)          │
  └─────────────────────────────────────────────┘

  方法 3: Vitess / ProxySQL によるシャーディング管理
  → DB ミドルウェアがリシャーディングを自動化
```

### 4.2 オンラインリシャーディングの実装

```python
"""オンラインリシャーディング: ダブルライト方式"""
import hashlib
from enum import Enum

class MigrationPhase(Enum):
    DUAL_WRITE = "dual_write"      # 旧+新に同時書き込み
    BACKFILL = "backfill"          # 旧→新にデータ移行
    VERIFY = "verify"              # 整合性検証
    CUTOVER = "cutover"            # 読み取りを新に切り替え
    CLEANUP = "cleanup"            # 旧データの削除

class OnlineResharder:
    """ゼロダウンタイム リシャーディング"""

    def __init__(
        self,
        old_router: 'ShardRouter',
        new_router: 'ShardRouter',
    ):
        self.old_router = old_router
        self.new_router = new_router
        self.phase = MigrationPhase.DUAL_WRITE
        self._migrated_keys: set = set()

    def write(self, shard_key: str, query: str, params: dict):
        """フェーズに応じた書き込みルーティング"""
        if self.phase == MigrationPhase.DUAL_WRITE:
            # 旧と新の両方に書き込み
            self.old_router.execute_on_shard(shard_key, query, params)
            self.new_router.execute_on_shard(shard_key, query, params)
        elif self.phase in (MigrationPhase.CUTOVER, MigrationPhase.CLEANUP):
            # 新シャードのみに書き込み
            self.new_router.execute_on_shard(shard_key, query, params)
        else:
            self.old_router.execute_on_shard(shard_key, query, params)

    def read(self, shard_key: str, query: str, params: dict):
        """フェーズに応じた読み取りルーティング"""
        if self.phase in (MigrationPhase.CUTOVER, MigrationPhase.CLEANUP):
            return self.new_router.execute_on_shard(shard_key, query, params)
        else:
            return self.old_router.execute_on_shard(shard_key, query, params)

    def backfill_batch(self, batch_keys: list[str]):
        """バッチ単位でデータを移行"""
        for key in batch_keys:
            if key not in self._migrated_keys:
                # 旧シャードからデータ読み取り
                data = self.old_router.execute_on_shard(
                    key, "SELECT * FROM users WHERE id = :id", {"id": key}
                )
                # 新シャードに書き込み
                for row in data:
                    self.new_router.execute_on_shard(
                        key,
                        "INSERT INTO users (...) VALUES (...) ON CONFLICT DO NOTHING",
                        dict(row)
                    )
                self._migrated_keys.add(key)

    def verify_consistency(self, sample_keys: list[str]) -> dict:
        """旧と新のデータ整合性を検証"""
        mismatches = []
        for key in sample_keys:
            old_data = self.old_router.execute_on_shard(
                key, "SELECT * FROM users WHERE id = :id", {"id": key}
            )
            new_data = self.new_router.execute_on_shard(
                key, "SELECT * FROM users WHERE id = :id", {"id": key}
            )
            if old_data != new_data:
                mismatches.append(key)

        return {
            'total_verified': len(sample_keys),
            'mismatches': len(mismatches),
            'consistency_rate': (len(sample_keys) - len(mismatches)) / len(sample_keys),
            'mismatch_keys': mismatches[:10],  # 最初の10件
        }
```

---

## 5. スケーリング手法の比較

### 比較表 1: 手法の特性比較

| 特性 | レプリケーション | パーティショニング | シャーディング |
|------|:-------------:|:---------------:|:------------:|
| 目的 | 読み取りスケール + 可用性 | 単一DB内のデータ管理 | 書き込みスケール |
| データ分散 | 全データを複製 | 単一DB内でテーブル分割 | 異なるDBサーバーに分散 |
| 書き込みスケール | 不可（単一Primary） | 限定的 | 可能 |
| 読み取りスケール | 可能（Replica追加） | 可能（プルーニング） | 可能 |
| 実装の複雑さ | 低 | 低〜中 | 高 |
| クロスデータ結合 | 容易 | 容易 | 困難 |
| 適用タイミング | 読み取り負荷 > 書き込み負荷 | 大テーブルの管理 | 書き込み負荷がDB上限超 |

### 比較表 2: 判断ポイント

| 判断ポイント | 推奨手法 | 理由 |
|-------------|---------|------|
| 読み取り負荷が高い | Read Replica | 書き込みは1台で十分 |
| テーブルが巨大 (数億行) | パーティショニング | 単一DB内で管理可能 |
| 書き込みが秒間数万超 | シャーディング | Primary分散が必須 |
| グローバル分散 | マルチリージョンレプリケーション | 地理的レイテンシ最適化 |
| マルチテナント | テナント単位シャーディング | テナント間の完全分離 |
| 時系列データの蓄積 | パーティショニング + アーカイブ | 古いデータの効率的管理 |

### 比較表 3: マネージドサービス比較

| 特性 | Amazon RDS | Amazon Aurora | Cloud Spanner | CockroachDB | Vitess |
|------|-----------|-------------|--------------|-------------|--------|
| タイプ | マネージドRDB | クラウドネイティブRDB | NewSQL | NewSQL | シャーディングミドルウェア |
| 自動スケール | 手動 (インスタンス変更) | ストレージ自動 | 自動 (ノード追加) | 自動 | 半自動 |
| レプリケーション | 非同期/半同期 | 同期 (6-way) | 同期 (Paxos) | 同期 (Raft) | 非同期 |
| シャーディング | 手動実装 | 不要 (Limitless) | 自動 | 自動 | 自動 |
| 読み取りスケール | Replica追加 | 15 Replica | ノード追加 | ノード追加 | Replica追加 |
| 書き込みスケール | なし (単一Primary) | 制限あり | リニア | リニア | シャード追加 |
| グローバル分散 | マルチリージョンRead | Global Database | マルチリージョン | マルチリージョン | 手動 |
| 互換性 | MySQL / PostgreSQL | MySQL / PostgreSQL 互換 | 独自 (SQL準拠) | PostgreSQL 互換 | MySQL |
| コスト | $$ | $$$ | $$$$ | $$$ | $ (OSS) |

---

## 6. アンチパターン

### アンチパターン 1: 早すぎるシャーディング

```python
# NG: ユーザー数1万人でシャーディング導入
class PrematureSharding:
    """早すぎるシャーディングの例"""
    def __init__(self):
        # 1万人のユーザーに4シャード
        self.shards = [
            create_engine(f'postgresql://shard{i}:5432/myapp')
            for i in range(4)
        ]

    def get_user(self, user_id: str):
        shard_id = hash(user_id) % 4
        # 問題: JOIN が必要なクエリがクロスシャードに
        # SELECT u.*, o.* FROM users u JOIN orders o ON u.id = o.user_id
        # → 2つのシャードにアクセスする必要がある
        pass

# 問題点:
# - 運用コスト 4倍 (バックアップ、監視、アップグレード)
# - クロスシャードクエリの複雑性
# - 単一DBで十分な規模 (1万ユーザー)
# - リシャーディングが困難

# OK: 段階的なスケーリング
class GradualScaling:
    """段階的にスケーリングする正しいアプローチ"""
    def __init__(self, current_load: str):
        if current_load == "low":
            # Step 1: インデックス最適化、クエリチューニング
            self.optimize_queries()
        elif current_load == "medium_read":
            # Step 2: Read Replica による読み取り分散
            self.add_read_replicas()
        elif current_load == "medium_write":
            # Step 3: キャッシュ層 (Redis) の追加
            self.add_cache_layer()
        elif current_load == "high":
            # Step 4: テーブルパーティショニング
            self.add_partitioning()
        elif current_load == "extreme":
            # Step 5: シャーディング（本当に必要になったら）
            self.implement_sharding()
```

### アンチパターン 2: 不適切なシャードキーの選定

```python
# NG: 作成日時をシャードキーに使用
class BadShardKey:
    """ホットスポットが発生するシャードキー"""
    def get_shard(self, created_at: datetime) -> int:
        # 月別シャーディング
        month = created_at.month
        return month % self.shard_count

    # 問題:
    # - 最新月のシャードに全ての書き込みが集中
    # - 古いシャードはほとんどアクセスなし
    # - リソースの無駄遣い

# NG: 自動インクリメント ID をシャードキーに使用
class BadAutoIncrementShardKey:
    def get_shard(self, auto_id: int) -> int:
        return auto_id % self.shard_count
    # 問題: 新規書き込みが常に同じシャードに集中

# OK: ユーザー ID / テナント ID をシャードキーに使用
class GoodShardKey:
    """均等分散されるシャードキー"""
    def get_shard(self, user_id: str) -> int:
        # SHA-256 ハッシュで均等分散
        hash_val = int(hashlib.sha256(user_id.encode()).hexdigest(), 16)
        return hash_val % self.shard_count

    # メリット:
    # - アクセスが均等に分散
    # - 同一ユーザーのデータが同じシャード (JOIN 可能)
    # - UUIDならカーディナリティが高い
```

### アンチパターン 3: レプリケーションラグの無視

```python
# NG: 書き込み直後に Replica から読む
class BadReadAfterWrite:
    """書き込み直後のレプリカ読み取り"""
    def update_and_read(self, user_id: str, new_name: str):
        # Primary に書き込み
        with self.write_session() as session:
            user = session.query(User).get(user_id)
            user.name = new_name
            session.commit()

        # 即座に Replica から読む → 古い名前が返る可能性！
        with self.read_session() as session:
            user = session.query(User).get(user_id)
            return user.name  # "古い名前" が返る可能性

# OK: Read-your-writes consistency を実装
class GoodReadAfterWrite:
    """書き込み直後は Primary から読む"""
    def __init__(self):
        self._recent_writes: dict[str, float] = {}

    def update_and_read(self, user_id: str, new_name: str):
        # Primary に書き込み
        with self.write_session() as session:
            user = session.query(User).get(user_id)
            user.name = new_name
            session.commit()

        # 書き込みを記録
        self._recent_writes[f"user:{user_id}"] = time.time()

        # 読み取り時に直近の書き込みをチェック
        last_write = self._recent_writes.get(f"user:{user_id}", 0)
        use_primary = (time.time() - last_write) < 5.0  # 5秒以内

        if use_primary:
            with self.write_session() as session:
                user = session.query(User).get(user_id)
                return user.name  # 確実に最新値
        else:
            with self.read_session() as session:
                user = session.query(User).get(user_id)
                return user.name
```

---

## 7. 練習問題

### 演習 1 (基礎): 読み書き分離の設計

以下の要件を満たす読み書き分離を設計せよ。

```
要件:
- Primary 1台、Replica 3台
- 書き込み: Primary のみ
- 読み取り: 通常は Replica、書き込み直後は Primary
- Replica の健全性チェック (レプリケーションラグ 5秒以上は除外)
- コネクションプーリング (アプリ20接続 → DB 50接続)

課題:
1. ReadWriteSplitter クラスを設計せよ (get_write_session, get_read_session)
2. レプリケーションラグが 5秒を超えた Replica を自動除外する仕組みを実装せよ
3. 全 Replica が不健全な場合のフォールバック戦略を設計せよ
```

**期待される出力:**

```python
# ReadWriteSplitter クラスの実装:
# - write_session(): Primary への接続を返す
# - read_session(entity_key):
#     直近 5秒以内に entity_key で書き込みがあれば Primary
#     そうでなければ健全な Replica からランダム選択
# - health_check(): 各 Replica のラグを監視し、不健全なものを除外
# - fallback: 全 Replica 不健全時は Primary から読み取り (負荷制限あり)
```

### 演習 2 (応用): シャードキーの選定

以下のテーブル構造とクエリパターンから、最適なシャードキーを選定せよ。

```sql
-- テーブル構造
CREATE TABLE orders (
    id          UUID PRIMARY KEY,
    user_id     UUID NOT NULL,
    merchant_id UUID NOT NULL,
    amount      DECIMAL(10, 2),
    status      VARCHAR(20),
    country     VARCHAR(2),
    created_at  TIMESTAMP
);

-- 主要クエリパターン (頻度順)
-- 1. SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT 20;  -- 60%
-- 2. SELECT * FROM orders WHERE merchant_id = ? AND created_at >= ?;              -- 25%
-- 3. SELECT SUM(amount) FROM orders WHERE country = ? AND created_at >= ?;        -- 10%
-- 4. SELECT * FROM orders WHERE id = ?;                                           -- 5%
```

**課題:**
1. 各候補キー (user_id, merchant_id, country, id) を評価せよ
2. 最適なシャードキーとその理由を述べよ
3. 選定したキーで対応できないクエリへの対策を設計せよ

### 演習 3 (上級): リシャーディング計画

4シャードから8シャードへのオンラインリシャーディングを計画せよ。

```
前提条件:
- 現在 4 シャード (各 500GB, 合計 2TB)
- シャードキー: user_id (SHA-256 ハッシュ % N)
- 日次書き込み: 1億レコード
- ダウンタイム: 0 (ゼロダウンタイム必須)
- データ整合性: 100% 保証

設計項目:
1. リシャーディング手順 (フェーズ分け)
2. データ移行の並列度と所要時間の見積もり
3. 整合性検証の方法
4. ロールバック計画
5. モニタリング項目
```

**期待される出力:** 各フェーズの詳細手順、リスクと対策、所要時間の見積もり

---

## 8. FAQ

### Q1. シャーディングではトランザクションはどうなる？

**A.** 単一シャード内のトランザクションは通常通り ACID を保証できる。クロスシャードトランザクションには 2PC (Two-Phase Commit) や Saga パターンが必要だが、性能・複雑性のコストが高い。設計段階で「同一トランザクション内のデータは同一シャードに配置する」ことが最も重要な原則。例えばユーザーの注文データはユーザーIDでシャーディングし、ユーザーと注文が同じシャードに配置されるようにする。クロスシャードの集計は CQRS パターンで非同期に計算する方が現実的。

### Q2. レプリケーションラグはどの程度発生する？

**A.** 環境と設定に依存する。PostgreSQL のストリーミングレプリケーション (非同期) で通常 < 1秒、MySQL の半同期レプリケーションで < 100ms が目安。ただし、大量の書き込みバースト時やネットワーク遅延時には数秒〜数十秒に拡大する。監視は `pg_stat_replication` (PostgreSQL) や `SHOW SLAVE STATUS` (MySQL) で行い、閾値を超えたらアラートを発報する。書き込み直後の読み取り (Read-your-writes) が必要な場合は、一時的に Primary から読む戦略を実装する。

### Q3. パーティショニングとシャーディングの違いは？

**A.** パーティショニングは**単一データベースサーバー内**でテーブルを論理的に分割する手法。PostgreSQL の `PARTITION BY RANGE/HASH/LIST` で実現し、アプリケーション側の変更は不要。パーティションプルーニングにより特定範囲のクエリが高速化する。一方、シャーディングは**複数の独立したデータベースサーバー**にデータを分散する手法。アプリケーション層でのルーティングが必要。パーティショニングは運用が簡単だが単一サーバーの制約内 (CPU/メモリ/ストレージ)、シャーディングはスケール上限がないが運用が複雑になる。

### Q4. NewSQL (Spanner, CockroachDB) は従来のシャーディングを不要にするか？

**A.** 部分的に Yes。NewSQL はシャーディング・レプリケーション・分散トランザクションを自動化し、アプリケーション層でのシャードルーティングが不要。しかし、(1) コストが高い (Cloud Spanner は RDS の 3-5倍)、(2) レイテンシが高い (分散合意プロトコルのオーバーヘッド)、(3) エコシステムが限定的 (ORM/ツールの互換性)。単一リージョンで秒間 10万 TPS 未満なら PostgreSQL + Read Replica で十分。NewSQL は「グローバル分散 + 強整合性 + 書き込みスケール」の3つが全て必要な場合に検討する。

### Q5. コネクションプーリングは必須か？

**A.** 本番環境では必須。PostgreSQL はコネクションごとにプロセスを fork するため、500コネクション以上でメモリ消費とコンテキストスイッチが問題になる。PgBouncer (transaction mode) を導入することで、アプリケーション側の 1000 接続を DB 側の 50 接続に多重化できる。これにより (1) DB サーバーの負荷を 90% 削減、(2) コネクション確立のレイテンシを 10ms → < 1ms に短縮、(3) コネクションリークのリスクを軽減。MySQL では ProxySQL が同等の役割を果たす。

### Q6. データベースのバックアップ戦略は？

**A.** 3-2-1 ルールを基本とする: 3コピー、2種類のメディア、1つはオフサイト。具体的には (1) **連続バックアップ (PITR)** --- WAL アーカイブを S3 に継続的に保存。任意の時点に復旧可能。RDS では自動化されている。(2) **日次フルバックアップ** --- pg_dump / mysqldump で論理バックアップ。整合性検証が容易。(3) **Replica からのバックアップ** --- Primary への負荷を避けるため、Replica からバックアップを取得。復旧テストを定期的 (月次) に実施し、RTO (復旧時間目標) と RPO (復旧ポイント目標) を確認する。シャーディング環境では各シャードのバックアップタイミングの整合性にも注意が必要。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| スケーリングの段階 | インデックス最適化 → プーリング → Replica → キャッシュ → パーティション → シャーディング |
| レプリケーション | 読み取りスケールと高可用性。レプリケーションラグへの対策 (Read-your-writes) が必須 |
| コネクションプーリング | PgBouncer (transaction mode) で接続数を 10-20倍に多重化 |
| パーティショニング | 単一DB内で大テーブルを分割。プルーニングによるクエリ高速化。月次自動作成 |
| シャーディング | 書き込みスケールの最終手段。シャードキー設計が成否を分ける |
| シャードキー | アクセス均等分散 + 関連データの同一シャード配置 + 高カーディナリティが原則 |
| リシャーディング | ゼロダウンタイムにはダブルライト + バックフィル方式。Vitess 等のツール活用 |
| トランザクション | クロスシャードを避ける設計。必要なら Saga パターンまたは CQRS |
| 監視 | レプリケーションラグ、コネクション数、クエリレイテンシ p99 を継続監視 |

---

## 次に読むべきガイド

- [キャッシング](./01-caching.md) --- DB 負荷軽減のためのキャッシュ戦略
- [メッセージキュー](./02-message-queue.md) --- 非同期処理によるDB負荷軽減
- [CDN](./03-cdn.md) --- 読み取り負荷をエッジにオフロード
- [CAP 定理](../00-fundamentals/03-cap-theorem.md) --- 分散データベースの理論的基盤
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) --- CQRS とイベントソーシング

---

## 参考文献

1. **Designing Data-Intensive Applications** --- Martin Kleppmann (O'Reilly, 2017) --- レプリケーション・パーティショニングの理論的基盤
2. **Database Internals** --- Alex Petrov (O'Reilly, 2019) --- 分散データベースの内部構造と実装
3. **High Performance MySQL, 4th Edition** --- Silvia Botros & Jeremy Tinley (O'Reilly, 2021) --- MySQL スケーリングの実践
4. **PostgreSQL Documentation: Table Partitioning** --- https://www.postgresql.org/docs/current/ddl-partitioning.html
5. **Vitess Documentation** --- https://vitess.io/docs/ --- MySQL シャーディングミドルウェアの公式ドキュメント
