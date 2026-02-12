# キャッシュ

> 頻繁にアクセスされるデータを高速ストレージに保持し、レイテンシ削減とスループット向上を実現するキャッシュ戦略を習得する。

## この章で学ぶこと

1. キャッシュの階層（ブラウザ、CDN、アプリケーション、データベース）と各層の役割
2. キャッシュパターン（Cache-Aside、Write-Through、Write-Behind、Read-Through）の動作と使い分け
3. キャッシュの無効化戦略（TTL、イベント駆動、バージョニング、タグベース）と一貫性の維持方法
4. キャッシュスタンピード（Thundering Herd）対策の実装手法
5. Redis の実践的なデータ構造活用とクラスタ運用

---

## 前提知識

| トピック | 内容 | 参照ガイド |
|---------|------|-----------|
| ハッシュテーブル | O(1)ルックアップの仕組み | データ構造基礎 |
| HTTP ヘッダー | Cache-Control、ETag、Last-Modified | Web基礎 |
| ロードバランサー | トラフィック分散の基本 | [ロードバランサー](./00-load-balancer.md) |
| データベース基礎 | RDB/NoSQL の読み書き性能特性 | DB基礎 |
| 分散システム基礎 | 一貫性モデル、ネットワーク遅延 | [CAP定理](../00-fundamentals/03-cap-theorem.md) |

---

## なぜキャッシュを学ぶのか

キャッシュは**システム性能を劇的に改善する最もコスト効率の良い手段**である。適切なキャッシュ戦略により、データベースの負荷を90%以上削減し、レスポンスタイムを数十分の一に短縮できる。

**数値で見るキャッシュの効果:**
- メモリアクセス: ~100ns vs ディスクI/O: ~10ms → **100,000倍の速度差**
- Redis の読み取り: ~0.1ms vs MySQL の読み取り: ~5-50ms → **50-500倍の改善**
- キャッシュヒット率 95% の場合、DB負荷は元の 1/20 に削減

**実世界の事例:**
- Facebook: Memcached クラスタで毎秒数十億リクエストをキャッシュヒット（NSDI '13論文）
- Twitter: タイムラインをRedisにキャッシュし、レイテンシを数百msから数msに短縮
- Amazon: ページ表示100ms遅延で売上1%減少（キャッシュによる高速化の経済的価値）

---

## 1. キャッシュの階層

### ASCII図解1: キャッシュの多層構造

```
  Client
    │
    ▼
  ┌─────────────────┐  Hit率: 30-50%
  │ Browser Cache   │  ← Cache-Control, ETag
  └────────┬────────┘
           │ miss
  ┌────────▼────────┐  Hit率: 70-90%
  │ CDN (Edge)      │  ← 静的ファイル、APIレスポンス
  └────────┬────────┘
           │ miss
  ┌────────▼────────┐
  │ Load Balancer   │
  └────────┬────────┘
           │
  ┌────────▼────────┐  Hit率: 80-99%
  │ App Cache       │  ← Redis / Memcached
  │ (in-memory)     │
  └────────┬────────┘
           │ miss
  ┌────────▼────────┐  Hit率: 95-99%
  │ DB Query Cache  │  ← MySQL Query Cache等
  └────────┬────────┘
           │ miss
  ┌────────▼────────┐
  │ Database        │  ← ディスクI/O
  └─────────────────┘

  各層でキャッシュヒットすると、
  下位層へのアクセスを回避できる
```

### 多層キャッシュの累積効果

```
リクエスト 1000件 が到着した場合の各層通過量:

  Browser Cache (Hit 40%)  → 600件通過
  CDN (Hit 60%)            → 240件通過
  App Cache (Hit 90%)      → 24件通過
  DB Query Cache (Hit 80%) → 約5件通過

  結果: 1000件中 995件がキャッシュヒット
  DB アクセスは 0.5% (5件) のみ

  累積キャッシュヒット率の計算:
  1 - (1-0.4) × (1-0.6) × (1-0.9) × (1-0.8) = 1 - 0.0048 = 99.52%
```

---

## 2. キャッシュパターン

### ASCII図解2: 4つの主要パターン

```
■ Cache-Aside (Lazy Loading)
  App ──read──→ Cache ──hit──→ return
                  │miss
                  ▼
               Database ──→ Cache に書き込み ──→ return

■ Read-Through
  App ──read──→ Cache ──hit──→ return
                  │miss
                  │ (Cache自身がDBから取得)
                  ▼
               Database ──→ Cache に自動保存 ──→ return

■ Write-Through
  App ──write──→ Cache ──同期書き込み──→ Database
                  │
                  └──→ return (両方完了後)

■ Write-Behind (Write-Back)
  App ──write──→ Cache ──→ return (即座に応答)
                  │
                  └──非同期──→ Database (バッチで後から書き込み)
```

### コード例1: Cache-Aside パターン

```python
import redis
import json
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """キャッシュ統計情報"""
    hits: int = 0
    misses: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __repr__(self):
        return (f"CacheStats(hits={self.hits}, misses={self.misses}, "
                f"hit_rate={self.hit_rate:.1%})")


class CacheAsideRepository:
    """Cache-Aside (Lazy Loading) パターン

    動作:
    1. 読み込み時: まずキャッシュを確認、ミスならDBから取得してキャッシュ保存
    2. 書き込み時: DBを更新してからキャッシュを無効化

    メリット:
    - 最も汎用的。アプリケーション側でキャッシュ制御が可能
    - 読み込みが多いワークロードに最適
    - キャッシュ障害時もDBから直接読める（フォールバック）

    デメリット:
    - 初回アクセスは必ずキャッシュミス（Cold Start問題）
    - キャッシュとDBの一貫性は保証されない（TTLによる最終的一貫性）
    """

    def __init__(self, redis_client: redis.Redis, db_client,
                 ttl: int = 300, prefix: str = ""):
        self.cache = redis_client
        self.db = db_client
        self.ttl = ttl
        self.prefix = prefix
        self.stats = CacheStats()

    def _cache_key(self, entity: str, id: str) -> str:
        """一貫したキャッシュキー命名規則"""
        return f"{self.prefix}{entity}:{id}"

    def get_user(self, user_id: str) -> Optional[dict]:
        cache_key = self._cache_key("user", user_id)

        # Step 1: キャッシュを確認
        try:
            cached = self.cache.get(cache_key)
            if cached:
                self.stats.hits += 1
                logger.debug(f"[CACHE HIT] {cache_key}")
                return json.loads(cached)
        except redis.RedisError as e:
            self.stats.errors += 1
            logger.warning(f"[CACHE ERROR] {cache_key}: {e}")
            # キャッシュ障害時はDBフォールバック

        # Step 2: キャッシュミス → DBから取得
        self.stats.misses += 1
        logger.debug(f"[CACHE MISS] {cache_key}")
        user = self.db.find_user(user_id)
        if user is None:
            # ネガティブキャッシュ: 存在しないデータも短TTLでキャッシュ
            # → DB穿刺攻撃（Cache Penetration）防止
            try:
                self.cache.setex(cache_key, 60, json.dumps(None))
            except redis.RedisError:
                pass
            return None

        # Step 3: キャッシュに書き込み（TTL付き）
        try:
            self.cache.setex(cache_key, self.ttl, json.dumps(user))
        except redis.RedisError as e:
            logger.warning(f"[CACHE WRITE ERROR] {cache_key}: {e}")
        return user

    def update_user(self, user_id: str, data: dict):
        """更新時はDBを更新してからキャッシュを無効化"""
        cache_key = self._cache_key("user", user_id)

        # Step 1: DB を更新
        self.db.update_user(user_id, data)

        # Step 2: キャッシュを無効化（削除）
        try:
            self.cache.delete(cache_key)
            logger.info(f"[CACHE INVALIDATE] {cache_key}")
        except redis.RedisError as e:
            logger.error(f"[CACHE DELETE ERROR] {cache_key}: {e}")
            # 削除失敗時はTTLで自然に期限切れになるのを待つ

    def bulk_warmup(self, user_ids: list[str]):
        """キャッシュウォームアップ: 頻出データを事前にロード"""
        pipe = self.cache.pipeline()
        users = self.db.find_users_batch(user_ids)
        for user in users:
            cache_key = self._cache_key("user", user["id"])
            pipe.setex(cache_key, self.ttl, json.dumps(user))
        pipe.execute()
        logger.info(f"[WARMUP] {len(users)} users pre-cached")
```

### コード例2: Write-Through パターン

```python
class WriteThroughRepository:
    """Write-Through パターン: 書き込み時にキャッシュとDBを同期更新

    メリット:
    - キャッシュとDBの一貫性が高い
    - 読み込み時は常にキャッシュヒット（書き込みで最新データがキャッシュ済み）

    デメリット:
    - 書き込みレイテンシが増加（DB + キャッシュの2箇所に書く）
    - 読まれないデータもキャッシュに書かれる（メモリ浪費の可能性）
    """

    def __init__(self, redis_client: redis.Redis, db_client,
                 ttl: int = 3600):
        self.cache = redis_client
        self.db = db_client
        self.ttl = ttl

    def get_user(self, user_id: str) -> Optional[dict]:
        cache_key = f"user:{user_id}"
        cached = self.cache.get(cache_key)
        if cached:
            return json.loads(cached)

        user = self.db.find_user(user_id)
        if user:
            self.cache.setex(cache_key, self.ttl, json.dumps(user))
        return user

    def update_user(self, user_id: str, data: dict):
        """DBとキャッシュを同時に更新（同期）"""
        cache_key = f"user:{user_id}"

        # Step 1: DB を更新
        self.db.update_user(user_id, data)

        # Step 2: キャッシュも更新（削除ではなく上書き）
        updated_user = self.db.find_user(user_id)
        self.cache.setex(cache_key, self.ttl, json.dumps(updated_user))
        logger.info(f"[WRITE-THROUGH] {cache_key} updated in both DB and cache")

    def create_user(self, user_data: dict) -> str:
        """作成時もキャッシュに同期書き込み"""
        user_id = self.db.create_user(user_data)
        cache_key = f"user:{user_id}"
        user_data["id"] = user_id
        self.cache.setex(cache_key, self.ttl, json.dumps(user_data))
        logger.info(f"[WRITE-THROUGH] {cache_key} created in both DB and cache")
        return user_id
```

### コード例3: Write-Behind パターン

```python
import queue
import threading
import time

class WriteBehindRepository:
    """Write-Behind (Write-Back) パターン: 非同期でDBに書き込み

    メリット:
    - 書き込みレイテンシが最小（キャッシュへの書き込みのみで応答）
    - バッチ処理でDB書き込みを効率化

    デメリット:
    - キャッシュ障害時にデータ損失リスク（まだDBに反映されていないデータ）
    - 実装が複雑（バックグラウンドライター、リトライ、デッドレターキュー）

    ユースケース:
    - アクセスカウンター、ページビュー数
    - 位置情報の頻繁な更新
    - IoT デバイスからのテレメトリデータ
    """

    def __init__(self, redis_client: redis.Redis, db_client,
                 flush_interval: float = 5.0, batch_size: int = 100):
        self.cache = redis_client
        self.db = db_client
        self.write_queue = queue.Queue()
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self._failed_count = 0
        self._total_flushed = 0
        self._start_background_writer()

    def update_user(self, user_id: str, data: dict):
        """キャッシュに書き込み、非同期でDBに反映"""
        cache_key = f"user:{user_id}"

        # Step 1: キャッシュに即座に書き込み
        self.cache.set(cache_key, json.dumps(data))

        # Step 2: キューに追加（後でバッチ処理）
        self.write_queue.put(("update_user", user_id, data))
        logger.debug(f"[WRITE-BEHIND] {cache_key} cached, queued for DB write "
                     f"(queue size: {self.write_queue.qsize()})")

    def _start_background_writer(self):
        """バックグラウンドスレッドでキューをバッチ処理"""
        def writer():
            while True:
                batch = []
                deadline = time.time() + self.flush_interval
                while len(batch) < self.batch_size and time.time() < deadline:
                    try:
                        item = self.write_queue.get(timeout=0.5)
                        batch.append(item)
                    except queue.Empty:
                        continue

                if batch:
                    self._flush_batch(batch)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        logger.info(f"[WRITE-BEHIND] Background writer started "
                    f"(interval={self.flush_interval}s, batch={self.batch_size})")

    def _flush_batch(self, batch: list):
        """バッチでDBに書き込み"""
        logger.info(f"[FLUSH] Writing {len(batch)} items to DB")
        success = 0
        for op, user_id, data in batch:
            try:
                self.db.update_user(user_id, data)
                success += 1
            except Exception as e:
                self._failed_count += 1
                logger.error(f"[FLUSH ERROR] Failed to write {user_id}: {e}")
                # 失敗時はキューに戻す（リトライ）
                self.write_queue.put((op, user_id, data))

        self._total_flushed += success
        logger.info(f"[FLUSH] Completed: {success}/{len(batch)} "
                    f"(total: {self._total_flushed}, failed: {self._failed_count})")

    def get_stats(self) -> dict:
        return {
            "queue_size": self.write_queue.qsize(),
            "total_flushed": self._total_flushed,
            "failed_count": self._failed_count,
        }
```

---

## 3. キャッシュの無効化

### ASCII図解3: キャッシュ無効化の戦略フロー

```
  データ更新イベント
       │
       ├─── TTL方式 ────→ 自然に期限切れ（受動的）
       │                   適用: 厳密な一貫性不要
       │
       ├─── イベント駆動 ─→ DB変更 → キャッシュ削除（能動的）
       │                   適用: リアルタイム一貫性
       │
       ├─── バージョニング → version番号で有効性判定
       │                   適用: 競合回避
       │
       └─── タグベース ──→ タグに紐づく全キーを一括削除
                           適用: 関連データのグループ無効化

  推奨: TTL（ベースライン）+ イベント駆動（即時反映）の併用
```

### コード例4: キャッシュ無効化戦略

```python
import time
import hashlib
from typing import Optional

class CacheInvalidation:
    """キャッシュ無効化の各戦略"""

    def __init__(self, redis_client: redis.Redis):
        self.cache = redis_client

    # 1. TTL（Time To Live）
    def set_with_ttl(self, key: str, value: str, ttl: int = 300):
        """一定時間で自動的に期限切れ

        用途別の推奨TTL:
        - セッション:   1800秒 (30分)
        - ユーザー情報: 300-900秒 (5-15分)
        - 設定データ:   3600-86400秒 (1-24時間)
        - 静的データ:   3600-604800秒 (1時間-7日)
        - 検索結果:     60-300秒 (1-5分)
        """
        self.cache.setex(key, ttl, value)

    # 2. イベント駆動無効化
    def on_data_changed(self, entity_type: str, entity_id: str):
        """データ変更イベントで関連キャッシュを無効化"""
        patterns = [
            f"{entity_type}:{entity_id}",
            f"{entity_type}:list:*",
            f"{entity_type}:count",
        ]
        pipe = self.cache.pipeline()
        for pattern in patterns:
            for key in self.cache.scan_iter(match=pattern):
                pipe.delete(key)
        deleted = pipe.execute()
        count = sum(1 for d in deleted if d)
        logger.info(f"[INVALIDATE] {entity_type}:{entity_id} "
                    f"→ {count} related caches cleared")

    # 3. バージョニング
    def get_with_version(self, key: str, current_version: int) -> Optional[str]:
        """バージョン番号でキャッシュの有効性を判定"""
        cached = self.cache.hgetall(f"v:{key}")
        if cached and int(cached.get(b"version", 0)) >= current_version:
            return cached[b"data"]
        return None  # 古いバージョン → キャッシュミス扱い

    def set_with_version(self, key: str, value: str, version: int):
        self.cache.hset(f"v:{key}", mapping={
            "data": value,
            "version": version
        })

    # 4. キャッシュタグ（グループ無効化）
    def set_with_tags(self, key: str, value: str, tags: list[str],
                     ttl: int = 300):
        """タグを付与し、タグ単位で一括無効化"""
        pipe = self.cache.pipeline()
        pipe.setex(key, ttl, value)
        for tag in tags:
            pipe.sadd(f"tag:{tag}", key)
            pipe.expire(f"tag:{tag}", ttl + 3600)  # タグは少し長めに保持
        pipe.execute()

    def invalidate_tag(self, tag: str):
        """タグに紐づく全キャッシュを無効化

        例: "user:123" タグで、そのユーザーの
        プロフィール、注文一覧、推薦結果を一括無効化
        """
        keys = self.cache.smembers(f"tag:{tag}")
        if keys:
            pipe = self.cache.pipeline()
            for key in keys:
                pipe.delete(key)
            pipe.delete(f"tag:{tag}")
            pipe.execute()
            logger.info(f"[TAG INVALIDATE] {tag}: {len(keys)} keys cleared")

    # 5. CDC (Change Data Capture) 連携
    def setup_cdc_invalidation(self, debezium_event: dict):
        """DBの変更ログ（CDC）からキャッシュを自動無効化

        Debezium等のCDCツールと連携し、
        DB変更を検知してキャッシュを即座に無効化する
        """
        table = debezium_event["source"]["table"]
        op = debezium_event["op"]  # c=create, u=update, d=delete
        entity_id = debezium_event["after"]["id"] if op != "d" \
                    else debezium_event["before"]["id"]

        if op in ("u", "d"):
            self.on_data_changed(table, str(entity_id))
            logger.info(f"[CDC] {table}:{entity_id} op={op} → cache invalidated")
```

### コード例5: キャッシュスタンピード対策

```python
import threading
import random

class ThunderingHerdProtection:
    """キャッシュスタンピード（Thundering Herd）対策

    問題: 人気キーのTTL期限切れ → 同時に数千リクエストがDBアクセス
    → DB過負荷 → カスケード障害

    対策3つ:
    1. ロック方式: 1リクエストだけDBアクセス、他はロック待ち
    2. ソフトTTL: 期限前にバックグラウンドで先行更新
    3. 確率的早期再計算: TTL切れ前にランダムに再計算
    """

    def __init__(self, redis_client: redis.Redis):
        self.cache = redis_client

    def get_with_lock(self, key: str, ttl: int, fetch_func,
                     lock_timeout: int = 10):
        """方式1: ロックを使って1リクエストだけがDBアクセス"""
        # Step 1: キャッシュ確認
        cached = self.cache.get(key)
        if cached:
            return json.loads(cached)

        # Step 2: ロック取得を試みる
        lock_key = f"lock:{key}"
        acquired = self.cache.set(lock_key, "1", nx=True, ex=lock_timeout)

        if acquired:
            try:
                # Step 3: ロック取得成功 → DBから取得してキャッシュ更新
                value = fetch_func()
                self.cache.setex(key, ttl, json.dumps(value))
                return value
            finally:
                self.cache.delete(lock_key)
        else:
            # Step 4: ロック取得失敗 → 短時間待ってリトライ
            for _ in range(5):
                time.sleep(0.1)
                cached = self.cache.get(key)
                if cached:
                    return json.loads(cached)
            # まだなければ自分でフェッチ（フォールバック）
            return fetch_func()

    def get_with_early_expiry(self, key: str, ttl: int,
                              soft_ttl: int, fetch_func):
        """方式2: ソフトTTL（期限前にバックグラウンドで先行更新）

        例: TTL=300秒, soft_ttl=240秒
        → 240秒でソフト期限切れ → 古いデータを返しつつ裏で更新
        → 300秒までにはハードTTLで完全期限切れ
        """
        data = self.cache.hgetall(f"soft:{key}")
        if data:
            expires_at = float(data.get(b"expires_at", 0))
            if time.time() < expires_at:
                return json.loads(data[b"value"])
            # ソフト期限切れ → バックグラウンドで更新
            threading.Thread(
                target=self._refresh,
                args=(key, ttl, soft_ttl, fetch_func),
                daemon=True
            ).start()
            return json.loads(data[b"value"])  # 古いデータを返す

        # 完全な期限切れ → 同期で取得
        return self._refresh(key, ttl, soft_ttl, fetch_func)

    def _refresh(self, key, ttl, soft_ttl, fetch_func):
        value = fetch_func()
        self.cache.hset(f"soft:{key}", mapping={
            "value": json.dumps(value),
            "expires_at": str(time.time() + soft_ttl)
        })
        self.cache.expire(f"soft:{key}", ttl)
        return value

    def get_with_probabilistic_expiry(self, key: str, ttl: int,
                                       fetch_func, beta: float = 1.0):
        """方式3: 確率的早期再計算 (Probabilistic Early Recomputation)

        XFetch アルゴリズム:
        TTL残り時間が少なくなるにつれ、再計算確率が増加
        → 1リクエストだけが自然にDBアクセスし、スタンピード回避

        beta: 再計算の積極性（大きいほど早めに再計算）
        """
        cached = self.cache.get(f"pxf:{key}")
        if cached:
            data = json.loads(cached)
            expiry = data["expiry"]
            remaining = expiry - time.time()

            if remaining > 0:
                # XFetch確率判定: -beta * log(random) が残TTLを超えたら再計算
                threshold = -beta * ttl * 0.1 * \
                           (random.random() if random.random() > 0 else 0.001)
                if remaining > threshold:
                    return data["value"]
                # 確率的に再計算をトリガー
                logger.debug(f"[XFETCH] Probabilistic recompute for {key}")

        # 再計算
        start = time.time()
        value = fetch_func()
        compute_time = time.time() - start

        self.cache.setex(f"pxf:{key}", ttl, json.dumps({
            "value": value,
            "expiry": time.time() + ttl,
            "compute_time": compute_time,
        }))
        return value
```

---

## 4. Redis の実践的な使い方

### ASCII図解4: Redis のデータ構造と用途

```
  ┌──────────────────────────────────────────────────┐
  │               Redis データ構造                     │
  ├──────────┬───────────────────────────────────────┤
  │ String   │ セッション、単純キャッシュ、カウンター   │
  │ Hash     │ ユーザープロフィール、設定              │
  │ List     │ タイムライン、キュー                   │
  │ Set      │ タグ、ユニーク訪問者                   │
  │ SortedSet│ ランキング、レート制限                  │
  │ Stream   │ イベントログ、メッセージング             │
  │ HyperLogLog│ ユニークカウント（近似）             │
  │ Bitmap   │ 日次アクティブユーザー                  │
  └──────────┴───────────────────────────────────────┘
```

### コード例6: Redis データ構造の活用例

```python
import redis
import json
import time
from datetime import datetime

class RedisPatterns:
    """Redis データ構造の実践的な活用パターン"""

    def __init__(self, client: redis.Redis):
        self.r = client

    # --- 1. Sorted Set: リアルタイムランキング ---
    def update_leaderboard(self, game_id: str, user_id: str, score: int):
        """ランキングの更新（O(log N)）"""
        key = f"leaderboard:{game_id}"
        self.r.zadd(key, {user_id: score})

    def get_top_players(self, game_id: str, count: int = 10) -> list:
        """上位N位を取得（O(log N + M)）"""
        key = f"leaderboard:{game_id}"
        return self.r.zrevrange(key, 0, count - 1, withscores=True)

    def get_rank(self, game_id: str, user_id: str) -> Optional[int]:
        """特定ユーザーの順位を取得（O(log N)）"""
        key = f"leaderboard:{game_id}"
        rank = self.r.zrevrank(key, user_id)
        return rank + 1 if rank is not None else None

    # --- 2. Sorted Set: スライディングウィンドウレート制限 ---
    def check_rate_limit(self, user_id: str, max_requests: int = 100,
                         window_sec: int = 60) -> bool:
        """スライディングウィンドウによるレート制限"""
        key = f"ratelimit:{user_id}"
        now = time.time()
        pipe = self.r.pipeline()

        # 古いエントリを削除
        pipe.zremrangebyscore(key, 0, now - window_sec)
        # 現在のリクエストを追加
        pipe.zadd(key, {f"{now}:{id(now)}": now})
        # ウィンドウ内のリクエスト数を取得
        pipe.zcard(key)
        # TTL設定
        pipe.expire(key, window_sec)

        results = pipe.execute()
        request_count = results[2]

        if request_count > max_requests:
            return False  # レート制限超過
        return True

    # --- 3. HyperLogLog: ユニーク訪問者カウント ---
    def track_unique_visitor(self, page: str, visitor_id: str):
        """ユニーク訪問者を近似カウント（メモリ効率: ~12KB/キー）"""
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"uv:{page}:{today}"
        self.r.pfadd(key, visitor_id)
        self.r.expire(key, 86400 * 7)  # 7日間保持

    def get_unique_visitors(self, page: str, date: str) -> int:
        """ユニーク訪問者数を取得（誤差率 ~0.81%）"""
        key = f"uv:{page}:{date}"
        return self.r.pfcount(key)

    # --- 4. Bitmap: 日次アクティブユーザー ---
    def mark_active(self, user_id: int):
        """ユーザーをアクティブとしてマーク（1ビット/ユーザー）"""
        today = datetime.now().strftime("%Y-%m-%d")
        key = f"dau:{today}"
        self.r.setbit(key, user_id, 1)
        self.r.expire(key, 86400 * 30)  # 30日間保持

    def get_dau(self, date: str) -> int:
        """日次アクティブユーザー数を取得"""
        key = f"dau:{date}"
        return self.r.bitcount(key)

    def get_retention(self, date1: str, date2: str) -> int:
        """2日間の両方でアクティブなユーザー数（リテンション分析）"""
        result_key = f"retention:{date1}:{date2}"
        self.r.bitop("AND", result_key, f"dau:{date1}", f"dau:{date2}")
        count = self.r.bitcount(result_key)
        self.r.delete(result_key)
        return count

    # --- 5. Pub/Sub: キャッシュ無効化の分散通知 ---
    def publish_invalidation(self, entity_type: str, entity_id: str):
        """キャッシュ無効化イベントを全インスタンスに通知"""
        channel = f"cache:invalidate:{entity_type}"
        message = json.dumps({"entity_id": entity_id, "timestamp": time.time()})
        self.r.publish(channel, message)

    def subscribe_invalidation(self, entity_type: str, callback):
        """キャッシュ無効化イベントを購読"""
        pubsub = self.r.pubsub()
        pubsub.subscribe(f"cache:invalidate:{entity_type}")
        for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                callback(data)


# === デモ実行 ===
r = redis.Redis(host='localhost', port=6379, db=0)
patterns = RedisPatterns(r)

# ランキング
patterns.update_leaderboard("game1", "alice", 1500)
patterns.update_leaderboard("game1", "bob", 2100)
patterns.update_leaderboard("game1", "charlie", 1800)
top = patterns.get_top_players("game1", 3)
print(f"Top 3: {top}")
# [(b'bob', 2100.0), (b'charlie', 1800.0), (b'alice', 1500.0)]

# レート制限
allowed = patterns.check_rate_limit("user:42", max_requests=5, window_sec=10)
print(f"Rate limit allowed: {allowed}")

# ユニーク訪問者
patterns.track_unique_visitor("/home", "visitor-1")
patterns.track_unique_visitor("/home", "visitor-2")
patterns.track_unique_visitor("/home", "visitor-1")  # 重複
count = patterns.get_unique_visitors("/home", datetime.now().strftime("%Y-%m-%d"))
print(f"Unique visitors: {count}")  # 2
```

---

## 5. 比較表

### 比較表1: キャッシュパターンの比較

| パターン | 読み込み性能 | 書き込み性能 | 一貫性 | データ損失リスク | 実装複雑度 | 適するケース |
|---------|------------|------------|--------|---------------|-----------|-------------|
| Cache-Aside | 高い（ヒット時） | 中（DB直書き） | 中（TTLで許容） | 低い | 低 | 読み込み多、汎用 |
| Read-Through | 高い | 中 | 中 | 低い | 中 | キャッシュ層で抽象化 |
| Write-Through | 高い | 低い（同期2書き） | 高い | 低い | 中 | 一貫性重要 |
| Write-Behind | 高い | 最高（非同期） | 低い | 高い（障害時） | 高 | 書き込み多 |

### 比較表2: Redis vs Memcached

| 項目 | Redis | Memcached |
|------|-------|-----------|
| データ構造 | 豊富（String, Hash, List, Set, SortedSet等） | String のみ |
| 永続化 | RDB / AOF | なし |
| クラスタリング | Redis Cluster（自動シャーディング） | クライアント側で実装 |
| Pub/Sub | あり | なし |
| スクリプト | Lua スクリプト | なし |
| メモリ効率 | 中程度（構造体オーバーヘッド） | 高い（シンプル） |
| マルチスレッド | Redis 7.0+ でI/Oスレッド | マルチスレッド |
| トランザクション | MULTI/EXEC, Lua | CAS (Check And Set) |
| レイテンシ | ~0.1ms (単一ノード) | ~0.1ms |
| 最大値サイズ | 512MB | 1MB（デフォルト） |
| 適するケース | 多機能キャッシュ、セッション、ランキング | 単純な高速キャッシュ |

### 比較表3: キャッシュ無効化戦略の比較

| 戦略 | 即時性 | 実装コスト | 一貫性保証 | スケーラビリティ | 適するケース |
|------|--------|-----------|-----------|----------------|-------------|
| TTL | 低い（期限待ち） | 最低 | 最終的一貫性 | 高い | 大半のユースケース |
| イベント駆動 | 高い（即座） | 中 | ほぼリアルタイム | 中 | リアルタイム反映が必要 |
| バージョニング | 高い | 中 | 強い | 高い | 競合が多い環境 |
| タグベース | 高い | 中 | グループ単位 | 中 | 関連データの一括無効化 |
| CDC | 高い（DB変更検知） | 高 | DB主導の一貫性 | 高い | マイクロサービス、大規模 |

---

## 6. アンチパターン

### アンチパターン1: キャッシュを唯一のデータソースにする

```python
# NG: キャッシュのみにデータを保存

class CacheOnlyStore:
    """キャッシュを唯一のデータストアとして使用"""

    def __init__(self, redis_client: redis.Redis):
        self.cache = redis_client

    def save_order(self, order_id: str, data: dict):
        # DBに保存しない！
        self.cache.set(f"order:{order_id}", json.dumps(data))
        # 問題:
        # - Redis再起動でデータ消失
        # - maxmemory到達でLRU削除 → データ消失
        # - 複雑なクエリが不可能（JOIN、集計等）
        # - バックアップ・リカバリが困難

    def get_order(self, order_id: str) -> Optional[dict]:
        cached = self.cache.get(f"order:{order_id}")
        if not cached:
            return None  # データが消失している可能性
        return json.loads(cached)


# OK: キャッシュは高速レイヤー、DBがSource of Truth

class CacheWithDbStore:
    """キャッシュ + DB の正しい構成"""

    def __init__(self, redis_client: redis.Redis, db_client):
        self.cache = redis_client
        self.db = db_client

    def save_order(self, order_id: str, data: dict):
        # DB が Source of Truth
        self.db.save_order(order_id, data)
        # キャッシュは高速アクセス用の「揮発性レイヤー」
        self.cache.setex(f"order:{order_id}", 3600, json.dumps(data))

    def get_order(self, order_id: str) -> Optional[dict]:
        # キャッシュ → DB のフォールバック
        cached = self.cache.get(f"order:{order_id}")
        if cached:
            return json.loads(cached)
        # キャッシュミス → DBから取得して再キャッシュ
        order = self.db.find_order(order_id)
        if order:
            self.cache.setex(f"order:{order_id}", 3600, json.dumps(order))
        return order
```

### アンチパターン2: TTLなしの無期限キャッシュ

```python
# NG: TTLなしでキャッシュ

class NoTTLCache:
    def save(self, key: str, data: dict):
        self.cache.set(key, json.dumps(data))
        # 問題:
        # - メモリリーク（使われないデータが蓄積）
        # - データの鮮度が保証されない
        # - 手動で無効化しない限り古いデータが残り続ける


# OK: 用途に応じた適切なTTL設定

class ProperTTLCache:
    # 用途別のTTL定数
    TTL_SESSION = 1800       # 30分: セッション
    TTL_USER_PROFILE = 300   # 5分: ユーザー情報
    TTL_CONFIG = 3600        # 1時間: 設定データ
    TTL_STATIC = 86400       # 1日: 静的データ
    TTL_SEARCH = 60          # 1分: 検索結果
    TTL_NEGATIVE = 60        # 1分: ネガティブキャッシュ

    def save(self, key: str, data: dict, ttl_category: str = "default"):
        ttl_map = {
            "session": self.TTL_SESSION,
            "user": self.TTL_USER_PROFILE,
            "config": self.TTL_CONFIG,
            "static": self.TTL_STATIC,
            "search": self.TTL_SEARCH,
            "negative": self.TTL_NEGATIVE,
            "default": 300,
        }
        ttl = ttl_map.get(ttl_category, 300)
        self.cache.setex(key, ttl, json.dumps(data))

    def save_negative(self, key: str):
        """ネガティブキャッシュ: 存在しないデータの短期キャッシュ
        → Cache Penetration（キャッシュ穿刺攻撃）防止"""
        self.cache.setex(key, self.TTL_NEGATIVE, json.dumps(None))
```

### アンチパターン3: キャッシュキーの設計ミス

```python
# NG: 曖昧なキャッシュキー

class BadCacheKeys:
    def get_data(self, user_id: str, page: int):
        # 問題1: user_id だけでキーを構成 → ページネーションが効かない
        key = f"user:{user_id}"  # page情報が抜けている！

        # 問題2: クエリパラメータ全部をキーに含む
        key = f"search:{query}&page={page}&utm_source={utm}"
        # utm_source はキャッシュに関係ない → 不要なキャッシュミス

        # 問題3: キーにオブジェクト全体を含む
        key = f"result:{json.dumps(complex_filter)}"
        # キーが長すぎてRedisのメモリを浪費


# OK: 構造化されたキャッシュキー設計

class GoodCacheKeys:
    """キャッシュキーの設計規則:
    1. プレフィックス: エンティティタイプ
    2. 識別子: エンティティID
    3. サフィックス: バリエーション（ページ、言語等）
    4. 不要なパラメータは除外
    """

    def user_profile_key(self, user_id: str) -> str:
        return f"user:profile:{user_id}"

    def user_orders_key(self, user_id: str, page: int) -> str:
        return f"user:orders:{user_id}:p{page}"

    def search_key(self, query: str, page: int, filters: dict) -> str:
        # フィルタを正規化してハッシュ化
        filter_hash = hashlib.md5(
            json.dumps(filters, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"search:{query}:p{page}:f{filter_hash}"

    def product_key(self, product_id: str, locale: str = "ja") -> str:
        return f"product:{product_id}:{locale}"
```

---

## 7. 練習問題

### 演習1（基礎）: Cache-Aside パターンの実装と統計

**課題**: CacheAsideRepository を使い、100回のランダムアクセス（ユーザー数10人）を実行して、キャッシュヒット率を計測せよ。

```python
import random

# シンプルなインメモリDB（テスト用）
class MockDB:
    def __init__(self):
        self.users = {f"user-{i}": {"id": f"user-{i}", "name": f"User {i}"}
                     for i in range(10)}

    def find_user(self, user_id):
        return self.users.get(user_id)

# テスト実行
db = MockDB()
r = redis.Redis()
repo = CacheAsideRepository(r, db, ttl=60)

for _ in range(100):
    user_id = f"user-{random.randint(0, 9)}"
    repo.get_user(user_id)

print(f"Stats: {repo.stats}")
```

**期待される出力**:
```
Stats: CacheStats(hits=90, misses=10, hit_rate=90.0%)
```

### 演習2（応用）: キャッシュスタンピード対策の比較

**課題**: ロック方式とソフトTTL方式の両方で、10スレッドが同時にキャッシュミスした場合のDB呼び出し回数を比較せよ。

```python
import threading
from unittest.mock import MagicMock

db_call_count = 0
lock = threading.Lock()

def slow_db_fetch():
    global db_call_count
    with lock:
        db_call_count += 1
    time.sleep(0.1)  # DB遅延をシミュレート
    return {"data": "value"}

# ロック方式テスト
r = redis.Redis()
r.flushdb()
protection = ThunderingHerdProtection(r)
db_call_count = 0

threads = []
for _ in range(10):
    t = threading.Thread(target=protection.get_with_lock,
                        args=("test-key", 60, slow_db_fetch))
    threads.append(t)
    t.start()
for t in threads:
    t.join()

print(f"ロック方式: DB呼び出し回数 = {db_call_count}")
# 期待: 1-2回（ほとんどのスレッドがロック待ち）
```

### 演習3（発展）: 多層キャッシュシステムの設計

**課題**: ローカルキャッシュ（インプロセス）→ 分散キャッシュ（Redis）→ DB の3層キャッシュを実装し、各層のヒット率と平均レイテンシを計測せよ。

```python
import time
from collections import OrderedDict

class LRUCache:
    """インプロセスLRUキャッシュ（ローカルキャッシュ層）"""

    def __init__(self, max_size: int = 100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class MultiLayerCache:
    """3層キャッシュ: Local → Redis → DB"""

    def __init__(self, local_cache: LRUCache,
                 redis_client: redis.Redis, db_client):
        self.local = local_cache
        self.redis = redis_client
        self.db = db_client
        self.stats = {"l1_hits": 0, "l2_hits": 0, "db_hits": 0}

    def get(self, key: str):
        # Layer 1: ローカルキャッシュ (~0.001ms)
        value = self.local.get(key)
        if value is not None:
            self.stats["l1_hits"] += 1
            return value

        # Layer 2: Redis (~0.5ms)
        cached = self.redis.get(key)
        if cached:
            value = json.loads(cached)
            self.local.set(key, value)  # L1にも保存
            self.stats["l2_hits"] += 1
            return value

        # Layer 3: DB (~10ms)
        value = self.db.find(key)
        if value:
            self.redis.setex(key, 300, json.dumps(value))
            self.local.set(key, value)
            self.stats["db_hits"] += 1
        return value
```

---

## 8. FAQ

### Q1: キャッシュヒット率はどの程度を目指すべきですか？

一般的に80%以上が目標、90%以上で良好、95%以上で優秀とされる。ヒット率が低い場合は (1) TTLが短すぎる、(2) キャッシュキーの設計が細かすぎる、(3) データのアクセスパターンが分散している（ロングテール）、のいずれかを疑う。Pareto の法則（80/20ルール）により、20%のデータが80%のアクセスを占めることが多く、この上位データをキャッシュするだけで大幅な改善が見込める。ヒット率の監視は `redis-cli INFO stats` の `keyspace_hits` / `keyspace_misses` で確認できる。

### Q2: キャッシュとDBのデータが不整合になったらどうしますか？

不整合の原因は (1) 更新時のキャッシュ無効化漏れ、(2) Race Condition（同時更新）、(3) ネットワーク障害によるキャッシュ更新失敗。対策は TTL を適切に設定して自然治癒を待つ「最終手段」と、変更イベント（CDC: Change Data Capture）でキャッシュを更新する「能動的手段」の併用が効果的。不整合が致命的なデータ（残高等）はキャッシュしないか、Write-Through で同期更新する。二重書き込み問題（DB成功→キャッシュ失敗）は、更新時にキャッシュを「削除」し、次の読み込みで自然に再キャッシュする方式が最もシンプルで確実。

### Q3: Redis のメモリが不足したらどうなりますか？

Redis の `maxmemory-policy` 設定で動作が決まる。`volatile-lru`（TTL付きキーをLRUで削除）、`allkeys-lru`（全キーをLRUで削除）、`noeviction`（書き込みエラーを返す）が代表的。キャッシュ用途では `allkeys-lru` が推奨。メモリ使用量の監視と、不要なキーの定期的なクリーンアップも重要。Redis Cluster でノードを追加してメモリを拡張することも可能である。メモリ使用量は `INFO memory` の `used_memory_human` で確認する。

### Q4: Cache Penetration（キャッシュ穿刺）とCache Breakdown の違いは？

**Cache Penetration**: 存在しないデータへのリクエストが大量に来る場合。キャッシュもDBもミスし、毎回DBにアクセスする。対策はネガティブキャッシュ（null値を短TTLでキャッシュ）とブルームフィルター（存在しないキーを事前にフィルタリング）。**Cache Breakdown**: 人気キーのTTLが切れた瞬間に大量リクエストが殺到する場合。これがキャッシュスタンピード（Thundering Herd）問題。対策はロック、ソフトTTL、確率的早期再計算。

### Q5: ローカルキャッシュと分散キャッシュの使い分けは？

**ローカルキャッシュ**（Guava Cache, Caffeine等）: プロセス内メモリで超高速（~ns）。サーバー間で共有されないため、一貫性の保証は困難。設定データ、マスタデータ等の変更頻度が低いデータに適する。**分散キャッシュ**（Redis, Memcached）: ネットワーク越しでアクセス（~0.1-1ms）。全サーバーから同一データを参照でき一貫性が高い。セッション、ユーザーデータ等に適する。両方を多層キャッシュとして組み合わせるのが最も効果的。

### Q6: Redis Cluster と Redis Sentinel の違いは？

**Redis Sentinel**: マスター/レプリカ構成の高可用性ソリューション。マスター障害時に自動フェイルオーバーでレプリカを昇格する。データは分散されず、単一マスターにメモリ上限がある。**Redis Cluster**: 自動シャーディングによる水平スケーリング。データを16384スロットに分散し、複数マスターノードで分担する。単一ノードのメモリ上限を超えるデータ量を扱える。10GB以下ならSentinel、それ以上ならClusterが目安。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| キャッシュの目的 | レイテンシ削減、スループット向上、DB負荷軽減 |
| 多層キャッシュ | ブラウザ → CDN → App Cache → DB Cache |
| Cache-Aside | 最も一般的。読み込み時にキャッシュ、ミスでDB |
| Write-Through | 一貫性重視。DB+キャッシュを同期更新 |
| Write-Behind | 書き込み性能重視。非同期でDB反映（データ損失リスクあり） |
| 無効化戦略 | TTL（ベースライン）+ イベント駆動（即時反映）の併用 |
| スタンピード対策 | ロック、ソフトTTL、確率的早期再計算の3方式 |
| Redis活用 | SortedSet(ランキング)、HyperLogLog(UV)、Bitmap(DAU) |
| キーの設計 | エンティティタイプ:ID:バリエーションの構造化命名規則 |

---

## 次に読むべきガイド

- [CDN](./03-cdn.md) -- エッジキャッシュによるグローバル配信
- [メッセージキュー](./02-message-queue.md) -- 非同期処理と組み合わせたキャッシュ更新
- [DBスケーリング](./04-database-scaling.md) -- キャッシュと併用するDB最適化
- [ロードバランサー](./00-load-balancer.md) -- LBの背後のキャッシュ配置
- [信頼性](../00-fundamentals/02-reliability.md) -- キャッシュ障害時のフォールバック戦略

---

## 参考文献

1. Fitzpatrick, B. (2004). "Distributed Caching with Memcached." *Linux Journal*.
2. Redis Documentation -- https://redis.io/documentation
3. Nishtala, R. et al. (2013). "Scaling Memcache at Facebook." *NSDI '13*.
4. Vattani, A. et al. (2015). "Optimal and Efficient Approximate Algorithms for Probabilistic Early Expiration." *Proceedings of the VLDB Endowment*.
5. Carlson, J. (2013). *Redis in Action*. Manning Publications.
