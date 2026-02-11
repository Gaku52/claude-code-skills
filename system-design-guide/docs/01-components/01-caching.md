# キャッシュ

> 頻繁にアクセスされるデータを高速ストレージに保持し、レイテンシ削減とスループット向上を実現するキャッシュ戦略を習得する。

## この章で学ぶこと

1. キャッシュの階層（ブラウザ、CDN、アプリケーション、データベース）と各層の役割
2. キャッシュパターン（Cache-Aside、Write-Through、Write-Behind）の動作と使い分け
3. キャッシュの無効化戦略と一貫性の維持方法

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

---

## 2. キャッシュパターン

### ASCII図解2: 3つの主要パターン

```
■ Cache-Aside (Lazy Loading)
  App ──read──→ Cache ──hit──→ return
                  │miss
                  ▼
               Database ──→ Cache に書き込み ──→ return

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
from typing import Optional

class CacheAsideRepository:
    """Cache-Aside (Lazy Loading) パターン"""

    def __init__(self, redis_client: redis.Redis, db_client, ttl: int = 300):
        self.cache = redis_client
        self.db = db_client
        self.ttl = ttl

    def get_user(self, user_id: str) -> Optional[dict]:
        cache_key = f"user:{user_id}"

        # Step 1: キャッシュを確認
        cached = self.cache.get(cache_key)
        if cached:
            print(f"[CACHE HIT] {cache_key}")
            return json.loads(cached)

        # Step 2: キャッシュミス → DBから取得
        print(f"[CACHE MISS] {cache_key}")
        user = self.db.find_user(user_id)
        if user is None:
            return None

        # Step 3: キャッシュに書き込み（TTL付き）
        self.cache.setex(cache_key, self.ttl, json.dumps(user))
        return user

    def update_user(self, user_id: str, data: dict):
        """更新時はDBを更新してからキャッシュを無効化"""
        cache_key = f"user:{user_id}"

        # Step 1: DB を更新
        self.db.update_user(user_id, data)

        # Step 2: キャッシュを無効化（削除）
        self.cache.delete(cache_key)
        print(f"[CACHE INVALIDATE] {cache_key}")
        # 次の読み込み時にDBから再取得される
```

### コード例2: Write-Through パターン

```python
class WriteThroughRepository:
    """Write-Through パターン: 書き込み時にキャッシュとDBを同期更新"""

    def __init__(self, redis_client: redis.Redis, db_client, ttl: int = 3600):
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
        """DBとキャッシュを同時に更新"""
        cache_key = f"user:{user_id}"

        # Step 1: DB を更新
        self.db.update_user(user_id, data)

        # Step 2: キャッシュも更新（削除ではなく上書き）
        updated_user = self.db.find_user(user_id)
        self.cache.setex(cache_key, self.ttl, json.dumps(updated_user))
        print(f"[WRITE-THROUGH] {cache_key} updated in both DB and cache")
```

### コード例3: Write-Behind パターン

```python
import queue
import threading
import time

class WriteBehindRepository:
    """Write-Behind (Write-Back) パターン: 非同期でDBに書き込み"""

    def __init__(self, redis_client: redis.Redis, db_client,
                 flush_interval: float = 5.0, batch_size: int = 100):
        self.cache = redis_client
        self.db = db_client
        self.write_queue = queue.Queue()
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self._start_background_writer()

    def update_user(self, user_id: str, data: dict):
        """キャッシュに書き込み、非同期でDBに反映"""
        cache_key = f"user:{user_id}"

        # Step 1: キャッシュに即座に書き込み
        self.cache.set(cache_key, json.dumps(data))

        # Step 2: キューに追加（後でバッチ処理）
        self.write_queue.put(("update_user", user_id, data))
        print(f"[WRITE-BEHIND] {cache_key} cached, queued for DB write")

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

    def _flush_batch(self, batch: list):
        """バッチでDBに書き込み"""
        print(f"[FLUSH] Writing {len(batch)} items to DB")
        for op, user_id, data in batch:
            try:
                self.db.update_user(user_id, data)
            except Exception as e:
                print(f"[ERROR] Failed to write {user_id}: {e}")
                # 失敗時はキューに戻す（リトライ）
                self.write_queue.put((op, user_id, data))
```

---

## 3. キャッシュの無効化

### コード例4: キャッシュ無効化戦略

```python
import time
import hashlib

class CacheInvalidation:
    """キャッシュ無効化の各戦略"""

    def __init__(self, redis_client: redis.Redis):
        self.cache = redis_client

    # 1. TTL（Time To Live）
    def set_with_ttl(self, key: str, value: str, ttl: int = 300):
        """一定時間で自動的に期限切れ"""
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
        pipe.execute()
        print(f"[INVALIDATE] {entity_type}:{entity_id} related caches cleared")

    # 3. バージョニング
    def get_with_version(self, key: str, current_version: int):
        """バージョン番号でキャッシュの有効性を判定"""
        cached = self.cache.hgetall(f"v:{key}")
        if cached and int(cached.get(b"version", 0)) >= current_version:
            return cached[b"data"]
        return None  # 古いバージョン → キャッシュミス扱い

    def set_with_version(self, key: str, value: str, version: int):
        self.cache.hset(f"v:{key}", mapping={"data": value, "version": version})

    # 4. キャッシュタグ（グループ無効化）
    def set_with_tags(self, key: str, value: str, tags: list[str]):
        """タグを付与し、タグ単位で一括無効化"""
        pipe = self.cache.pipeline()
        pipe.set(key, value)
        for tag in tags:
            pipe.sadd(f"tag:{tag}", key)
        pipe.execute()

    def invalidate_tag(self, tag: str):
        """タグに紐づく全キャッシュを無効化"""
        keys = self.cache.smembers(f"tag:{tag}")
        if keys:
            pipe = self.cache.pipeline()
            for key in keys:
                pipe.delete(key)
            pipe.delete(f"tag:{tag}")
            pipe.execute()
            print(f"[TAG INVALIDATE] {tag}: {len(keys)} keys cleared")
```

### コード例5: キャッシュスタンピード対策

```python
import threading

class ThunderdingHerdProtection:
    """キャッシュスタンピード（Thundering Herd）対策"""

    def __init__(self, redis_client: redis.Redis):
        self.cache = redis_client
        self.locks = {}

    def get_with_lock(self, key: str, ttl: int, fetch_func):
        """
        ロックを使って1リクエストだけがDBアクセスし、
        他はキャッシュ更新を待つ
        """
        # Step 1: キャッシュ確認
        cached = self.cache.get(key)
        if cached:
            return json.loads(cached)

        # Step 2: ロック取得を試みる
        lock_key = f"lock:{key}"
        acquired = self.cache.set(lock_key, "1", nx=True, ex=10)

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
            time.sleep(0.1)
            cached = self.cache.get(key)
            if cached:
                return json.loads(cached)
            # まだなければ自分でフェッチ（フォールバック）
            return fetch_func()

    def get_with_early_expiry(self, key: str, ttl: int,
                              soft_ttl: int, fetch_func):
        """
        ソフトTTL: 期限前に先行更新してスタンピードを回避
        """
        data = self.cache.hgetall(f"soft:{key}")
        if data:
            expires_at = float(data.get(b"expires_at", 0))
            if time.time() < expires_at:
                return json.loads(data[b"value"])
            # ソフト期限切れ → バックグラウンドで更新
            threading.Thread(
                target=self._refresh,
                args=(key, ttl, soft_ttl, fetch_func)
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
```

---

## 4. Redis の実践的な使い方

### ASCII図解3: Redis のデータ構造と用途

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

---

## 5. 比較表

### 比較表1: キャッシュパターンの比較

| パターン | 読み込み性能 | 書き込み性能 | 一貫性 | データ損失リスク | 適するケース |
|---------|------------|------------|--------|---------------|-------------|
| Cache-Aside | 高い（ヒット時） | 中（DB直書き） | 中（TTLで許容） | 低い | 読み込み多、汎用 |
| Write-Through | 高い | 低い（同期2書き） | 高い | 低い | 一貫性重要 |
| Write-Behind | 高い | 最高（非同期） | 低い | 高い（障害時） | 書き込み多 |
| Read-Through | 高い | 中 | 中 | 低い | キャッシュ層で抽象化 |

### 比較表2: Redis vs Memcached

| 項目 | Redis | Memcached |
|------|-------|-----------|
| データ構造 | 豊富（String, Hash, List, Set等） | String のみ |
| 永続化 | RDB / AOF | なし |
| クラスタリング | Redis Cluster | クライアント側で実装 |
| Pub/Sub | あり | なし |
| スクリプト | Lua スクリプト | なし |
| メモリ効率 | 中程度（構造体オーバーヘッド） | 高い（シンプル） |
| マルチスレッド | Redis 7.0+ でI/Oスレッド | マルチスレッド |
| 適するケース | 多機能キャッシュ、セッション | 単純なキャッシュ |

---

## 6. アンチパターン

### アンチパターン1: キャッシュを唯一のデータソースにする

```
❌ ダメな例:
「パフォーマンスのためDBを廃止してRedisだけで運用」

問題:
- Redis再起動でデータ消失
- メモリ容量の制限
- 複雑なクエリが不可能
- バックアップ・リカバリが困難

✅ 正しい理解:
キャッシュは「揮発性の高速レイヤー」
データの正本（Source of Truth）は常にDBに置く
```

### アンチパターン2: TTLなしの無期限キャッシュ

```
❌ ダメな例:
cache.set("user:123", data)  # TTLなし → 永久にキャッシュ

問題:
- メモリリーク（使われないデータが蓄積）
- データの鮮度が保証されない
- キャッシュ無効化の仕組みがないと古いデータが残り続ける

✅ 正しいアプローチ:
cache.setex("user:123", 300, data)  # 5分TTL

# 用途別の目安:
# セッション:   30分
# ユーザー情報: 5〜15分
# 設定データ:   1〜24時間
# 静的データ:   1時間〜7日
```

---

## 7. FAQ

### Q1: キャッシュヒット率はどの程度を目指すべきですか？

一般的に80%以上が目標、90%以上で良好、95%以上で優秀とされる。ヒット率が低い場合は (1) TTLが短すぎる、(2) キャッシュキーの設計が細かすぎる、(3) データのアクセスパターンが分散している（ロングテール）、のいずれかを疑う。Pareto の法則（80/20ルール）により、20%のデータが80%のアクセスを占めることが多く、この上位データをキャッシュするだけで大幅な改善が見込める。

### Q2: キャッシュとDBのデータが不整合になったらどうしますか？

不整合の原因は (1) 更新時のキャッシュ無効化漏れ、(2) Race Condition（同時更新）、(3) ネットワーク障害によるキャッシュ更新失敗。対策は TTL を適切に設定して自然治癒を待つ「最終手段」と、変更イベント（CDC: Change Data Capture）でキャッシュを更新する「能動的手段」の併用が効果的。不整合が致命的なデータ（残高等）はキャッシュしないか、Write-Through で同期更新する。

### Q3: Redis のメモリが不足したらどうなりますか？

Redis の `maxmemory-policy` 設定で動作が決まる。`volatile-lru`（TTL付きキーをLRUで削除）、`allkeys-lru`（全キーをLRUで削除）、`noeviction`（書き込みエラーを返す）が代表的。キャッシュ用途では `allkeys-lru` が推奨。メモリ使用量の監視と、不要なキーの定期的なクリーンアップも重要。Redis Cluster でノードを追加してメモリを拡張することも可能である。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| キャッシュの目的 | レイテンシ削減、スループット向上、DB負荷軽減 |
| 多層キャッシュ | ブラウザ → CDN → App Cache → DB Cache |
| Cache-Aside | 最も一般的。読み込み時にキャッシュ、ミスでDB |
| Write-Through | 一貫性重視。DB+キャッシュを同期更新 |
| Write-Behind | 書き込み性能重視。非同期でDB反映 |
| 無効化戦略 | TTL、イベント駆動、バージョニング、タグベース |
| スタンピード対策 | ロック、ソフトTTL、確率的早期再計算 |

---

## 次に読むべきガイド

- [CDN](./03-cdn.md) — エッジキャッシュによるグローバル配信
- [メッセージキュー](./02-message-queue.md) — 非同期処理と組み合わせたキャッシュ更新
- [DBスケーリング](./04-database-scaling.md) — キャッシュと併用するDB最適化

---

## 参考文献

1. Fitzpatrick, B. (2004). "Distributed Caching with Memcached." *Linux Journal*.
2. Redis Documentation — https://redis.io/documentation
3. Nishtala, R. et al. (2013). "Scaling Memcache at Facebook." *NSDI '13*.
