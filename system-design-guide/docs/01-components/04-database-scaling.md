# DBスケーリング

> データベースの負荷増大に対するスケーリング戦略として、レプリケーション・シャーディング・パーティショニングの設計原則と実装手法を、具体的なコード例とトレードオフ分析を通じて解説する

## この章で学ぶこと

1. **垂直スケーリング vs 水平スケーリング** — スケールアップの限界と水平分散の設計判断
2. **レプリケーション戦略** — マスター/レプリカ構成、読み書き分離、レプリケーションラグへの対処
3. **シャーディングとパーティショニング** — シャードキーの選定、データ分散アルゴリズム、ホットスポット回避

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
  限界: ハードウェア上限、単一障害点

【スケールアウト (水平)】
  +--------+          +------+ +------+ +------+ +------+
  | DB     |   --->   | DB 1 | | DB 2 | | DB 3 | | DB 4 |
  | 全データ|          | A-F  | | G-L  | | M-R  | | S-Z  |
  +--------+          +------+ +------+ +------+ +------+
  メリット: 理論上無限にスケール、耐障害性
```

### 1.2 段階的スケーリングロードマップ

```
フェーズ 1: 単一サーバー
  App --> [Primary DB]
  月間 ~100万リクエスト

フェーズ 2: 読み書き分離 (Read Replica)
  App --> [Primary] (Write)
      --> [Replica 1] (Read)
      --> [Replica 2] (Read)
  月間 ~1000万リクエスト

フェーズ 3: キャッシュ層追加
  App --> [Redis Cache] --> [Primary / Replicas]
  月間 ~1億リクエスト

フェーズ 4: シャーディング
  App --> [Router] --> [Shard 0] [Shard 1] [Shard 2] ...
  月間 ~10億リクエスト以上
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
```

### 2.2 読み書き分離の実装

```python
# SQLAlchemy での読み書き分離 (Python)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import random

# 書き込み用 Primary
write_engine = create_engine(
    'postgresql://user:pass@primary-db:5432/myapp',
    pool_size=20, max_overflow=10
)

# 読み取り用 Replicas
read_engines = [
    create_engine(f'postgresql://user:pass@replica-{i}:5432/myapp',
                  pool_size=30, max_overflow=15)
    for i in range(3)
]

WriteSession = sessionmaker(bind=write_engine)
ReadSessions = [sessionmaker(bind=engine) for engine in read_engines]

def get_write_session():
    """書き込み用セッション（常に Primary）"""
    return WriteSession()

def get_read_session():
    """読み取り用セッション（Replica をランダム選択）"""
    session_class = random.choice(ReadSessions)
    return session_class()

# 使用例
def create_order(order_data):
    """書き込み → Primary"""
    session = get_write_session()
    try:
        order = Order(**order_data)
        session.add(order)
        session.commit()
        return order.id
    finally:
        session.close()

def get_orders(user_id):
    """読み取り → Replica"""
    session = get_read_session()
    try:
        return session.query(Order).filter_by(user_id=user_id).all()
    finally:
        session.close()
```

### 2.3 レプリケーションラグへの対処

```python
# レプリケーションラグ対策: Read-your-writes consistency
import time

class ConsistentReader:
    """書き込み直後は Primary から読む戦略"""

    def __init__(self, write_session_factory, read_session_factory):
        self._write = write_session_factory
        self._read = read_session_factory
        self._recent_writes = {}  # {entity_key: write_timestamp}

    def write(self, entity_key, operation):
        session = self._write()
        try:
            result = operation(session)
            session.commit()
            # 書き込み時刻を記録
            self._recent_writes[entity_key] = time.time()
            return result
        finally:
            session.close()

    def read(self, entity_key, query_func):
        # 直近5秒以内に書き込みがあれば Primary から読む
        last_write = self._recent_writes.get(entity_key, 0)
        use_primary = (time.time() - last_write) < 5.0

        session = self._write() if use_primary else self._read()
        try:
            return query_func(session)
        finally:
            session.close()
```

---

## 3. シャーディング

### 3.1 シャーディング戦略の比較

| 戦略 | 説明 | メリット | デメリット |
|------|------|---------|-----------|
| レンジベース | キーの範囲で分割 (A-F, G-L...) | シンプル、範囲クエリが容易 | データ偏り、ホットスポット |
| ハッシュベース | hash(key) % N で分割 | 均等分散 | 範囲クエリ不可、リシャーディング困難 |
| コンシステントハッシュ | ハッシュリング上で分割 | ノード追加時の移動データ最小 | 実装が複雑 |
| ディレクトリベース | ルックアップテーブルで管理 | 柔軟なマッピング | テーブルがSPOF、追加レイテンシ |

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

  ノード追加時: Shard D を追加
  → Shard A から一部のキーのみ Shard D に移動
  → 他のシャードは影響を受けない
```

### 3.3 シャーディングの実装

```python
# ハッシュベースシャーディング (Python)
import hashlib
from typing import Dict, List

class ShardRouter:
    """シャードルーターの実装"""

    def __init__(self, shard_configs: List[Dict]):
        self.shards = {}
        for config in shard_configs:
            shard_id = config['id']
            self.shards[shard_id] = create_engine(config['dsn'])
        self.shard_count = len(shard_configs)

    def get_shard_id(self, shard_key: str) -> int:
        """シャードキーからシャード ID を決定"""
        hash_value = int(hashlib.md5(
            str(shard_key).encode()
        ).hexdigest(), 16)
        return hash_value % self.shard_count

    def get_connection(self, shard_key: str):
        """シャードキーに対応するDB接続を返す"""
        shard_id = self.get_shard_id(shard_key)
        return self.shards[shard_id].connect()

    def execute_on_shard(self, shard_key: str, query: str, params: dict):
        """特定シャードでクエリ実行"""
        conn = self.get_connection(shard_key)
        try:
            return conn.execute(query, params)
        finally:
            conn.close()

    def scatter_gather(self, query: str, params: dict):
        """全シャードにクエリを実行して結果を集約"""
        results = []
        for shard_id, engine in self.shards.items():
            conn = engine.connect()
            try:
                result = conn.execute(query, params)
                results.extend(result.fetchall())
            finally:
                conn.close()
        return results

# 使用例
router = ShardRouter([
    {'id': 0, 'dsn': 'postgresql://user:pass@shard0:5432/myapp'},
    {'id': 1, 'dsn': 'postgresql://user:pass@shard1:5432/myapp'},
    {'id': 2, 'dsn': 'postgresql://user:pass@shard2:5432/myapp'},
])

# ユーザー ID をシャードキーとして使用
user_id = 'user-12345'
shard_id = router.get_shard_id(user_id)  # → 例: 1
print(f"User {user_id} → Shard {shard_id}")
```

### 3.4 PostgreSQL テーブルパーティショニング

```sql
-- PostgreSQL: 範囲パーティショニング
CREATE TABLE orders (
    id          BIGSERIAL,
    user_id     BIGINT NOT NULL,
    amount      DECIMAL(10, 2) NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 月別パーティション
CREATE TABLE orders_2026_01 PARTITION OF orders
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
CREATE TABLE orders_2026_02 PARTITION OF orders
    FOR VALUES FROM ('2026-02-01') TO ('2026-03-01');
CREATE TABLE orders_2026_03 PARTITION OF orders
    FOR VALUES FROM ('2026-03-01') TO ('2026-04-01');

-- パーティションごとにインデックス
CREATE INDEX idx_orders_2026_01_user ON orders_2026_01 (user_id);
CREATE INDEX idx_orders_2026_02_user ON orders_2026_02 (user_id);

-- クエリ: PostgreSQL が自動的に該当パーティションのみスキャン
EXPLAIN SELECT * FROM orders
WHERE created_at >= '2026-02-01' AND created_at < '2026-03-01';
-- → orders_2026_02 のみスキャン（パーティションプルーニング）
```

---

## 4. スケーリング手法の比較

| 特性 | レプリケーション | パーティショニング | シャーディング |
|------|:-------------:|:---------------:|:------------:|
| 目的 | 読み取りスケール + 可用性 | 単一DB内のデータ管理 | 書き込みスケール |
| データ分散 | 全データを複製 | 単一DB内でテーブル分割 | 異なるDBサーバーに分散 |
| 書き込みスケール | 不可（単一Primary） | 限定的 | 可能 |
| 読み取りスケール | 可能（Replica追加） | 可能（プルーニング） | 可能 |
| 実装の複雑さ | 低 | 低〜中 | 高 |
| クロスデータ結合 | 容易 | 容易 | 困難 |
| 適用タイミング | 読み取り負荷 > 書き込み負荷 | 大テーブルの管理 | 書き込み負荷がDB上限超 |

| 判断ポイント | 推奨手法 | 理由 |
|-------------|---------|------|
| 読み取り負荷が高い | Read Replica | 書き込みは1台で十分 |
| テーブルが巨大 (数億行) | パーティショニング | 単一DB内で管理可能 |
| 書き込みが秒間数万超 | シャーディング | Primary分散が必須 |
| グローバル分散 | マルチリージョンレプリケーション | 地理的レイテンシ最適化 |

---

## 5. アンチパターン

### アンチパターン 1: 早すぎるシャーディング

```
BAD: ユーザー数1万人でシャーディング導入
  → 運用コスト増大、クロスシャードクエリの複雑性
  → 単一DBで十分対応可能な規模

GOOD: 段階的なスケーリング
  Step 1: インデックス最適化、クエリチューニング
  Step 2: Read Replica による読み取り分散
  Step 3: キャッシュ層 (Redis) の追加
  Step 4: テーブルパーティショニング
  Step 5: シャーディング（本当に必要になったら）
```

### アンチパターン 2: 不適切なシャードキーの選定

```
BAD: 作成日時をシャードキーに使用
  → 最新データを扱うシャードに負荷が集中（ホットスポット）
  → 古いシャードはほとんどアクセスなし

BAD: 自動インクリメント ID をシャードキーに使用
  → 新規書き込みが常に最後のシャードに集中

GOOD: ユーザー ID / テナント ID をシャードキーに使用
  → アクセスが均等に分散
  → 同一ユーザーのデータが同じシャードに集約（JOIN 可能）
```

---

## 6. FAQ

### Q1. シャーディングではトランザクションはどうなる？

**A.** 単一シャード内のトランザクションは通常通り ACID を保証できる。クロスシャードトランザクションには 2PC (Two-Phase Commit) や Saga パターンが必要だが、性能・複雑性のコストが高い。設計段階で「同一トランザクション内のデータは同一シャードに配置する」ことが最も重要な原則。これがシャードキー選定の最大の判断基準になる。

### Q2. レプリケーションラグはどの程度発生する？

**A.** 同期レプリケーションでは0だがスループットが低下する。非同期レプリケーションでは通常ミリ秒〜数秒のラグが発生する。PostgreSQL のストリーミングレプリケーションで通常 <1秒、MySQL の半同期レプリケーションで <100ms が目安。書き込み直後の読み取り（Read-your-writes）が必要な場合は、一時的に Primary から読む戦略を実装する。

### Q3. パーティショニングとシャーディングの違いは？

**A.** パーティショニングは単一データベースサーバー内でテーブルを分割する手法で、PostgreSQL や MySQL のネイティブ機能で実現する。シャーディングは複数の独立したデータベースサーバーにデータを分散する手法で、アプリケーション層でのルーティングが必要。パーティショニングは運用が簡単だが単一サーバーの制約内、シャーディングはスケール上限がないが運用が複雑になる。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| スケーリングの段階 | インデックス最適化 → Replica → キャッシュ → パーティション → シャーディング |
| レプリケーション | 読み取りスケールと高可用性。レプリケーションラグへの対策が必須 |
| パーティショニング | 単一DB内で大テーブルを分割。プルーニングによるクエリ高速化 |
| シャーディング | 書き込みスケールの最終手段。シャードキー設計が成否を分ける |
| シャードキー | アクセス均等分散 + 関連データの同一シャード配置が原則 |
| トランザクション | クロスシャードを避ける設計。必要なら Saga パターン |

---

## 次に読むべきガイド

- [メッセージキュー](./02-message-queue.md) — 非同期処理によるDB負荷軽減
- [CDN](./03-cdn.md) — 読み取り負荷をエッジにオフロード
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) — CQRS とイベントソーシング

---

## 参考文献

1. **Designing Data-Intensive Applications** — Martin Kleppmann (O'Reilly, 2017) — レプリケーション・パーティショニングの理論
2. **Database Internals** — Alex Petrov (O'Reilly, 2019) — 分散データベースの内部構造
3. **High Performance MySQL, 4th Edition** — Silvia Botros & Jeremy Tinley (O'Reilly, 2021) — MySQL スケーリングの実践
4. **PostgreSQL Documentation: Table Partitioning** — https://www.postgresql.org/docs/current/ddl-partitioning.html
