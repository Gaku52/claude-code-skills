# レートリミッター設計

> APIやサービスへのリクエスト頻度を制御し、過負荷・不正利用・DDoS攻撃からシステムを保護するレートリミッターの設計原則とアルゴリズム、分散環境での実装手法を解説する

## この章で学ぶこと

1. **レートリミッティングの基本概念** — なぜ必要か、どこに配置するか、HTTP 429 の設計
2. **主要アルゴリズム** — Token Bucket、Leaky Bucket、Fixed Window、Sliding Window の仕組みと比較
3. **分散環境でのレートリミッター** — Redis を用いた実装、レースコンディション対策、多層防御

---

## 1. レートリミッターの全体設計

### 1.1 配置パターン

```
Client --> [CDN Edge Rate Limit] --> [API Gateway Rate Limit] --> [App Rate Limit]
                  |                          |                         |
            DDoS防御                   API Key別制限              ビジネスロジック制限
            IP別制限                   エンドポイント別制限        ユーザー別制限
            地域別制限                  プラン別制限               機能別制限

  多層防御: 外側で粗い制限、内側で細かい制限
```

### 1.2 レスポンス設計

```
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 30
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1707638400

{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "リクエスト制限を超過しました。30秒後に再試行してください。",
    "retry_after": 30
  }
}
```

### 1.3 システム構成

```
                    レートリミッター アーキテクチャ

  Client
    |
    v
  [Load Balancer]
    |
    v
  [Rate Limiter Middleware]
    |
    +---> [Redis Cluster] (カウンター/トークン管理)
    |         |
    |    [Key: "rate:user:123:api:/orders"]
    |    [Value: {count: 45, window_start: 1707638400}]
    |
    v
  [Application Server]
    |
    v
  [Backend Services]
```

---

## 2. アルゴリズム

### 2.1 Token Bucket

```
Token Bucket アルゴリズム

  +------------------+
  |  Token Bucket    |
  |                  |   ← トークンが一定レートで補充
  |  [T] [T] [T]    |      (例: 10 tokens/sec)
  |  [T] [T]        |
  |  max = 10       |   ← バケット容量 = バースト許容量
  +--------+---------+
           |
     リクエスト到着
     → トークンあり: 1トークン消費して処理
     → トークンなし: 429 拒否
```

```python
# Token Bucket 実装 (Redis + Lua Script)
import redis
import time

class TokenBucketLimiter:
    """Token Bucket レートリミッター"""

    # Lua スクリプト（アトミック操作でレースコンディション防止）
    LUA_SCRIPT = """
    local key = KEYS[1]
    local max_tokens = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])   -- tokens per second
    local now = tonumber(ARGV[3])

    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1]) or max_tokens
    local last_refill = tonumber(data[2]) or now

    -- トークン補充
    local elapsed = now - last_refill
    local new_tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

    if new_tokens >= 1 then
        -- トークン消費
        new_tokens = new_tokens - 1
        redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
        redis.call('EXPIRE', key, math.ceil(max_tokens / refill_rate) * 2)
        return {1, new_tokens}   -- 許可, 残りトークン
    else
        return {0, 0}            -- 拒否
    end
    """

    def __init__(self, redis_client, max_tokens=100, refill_rate=10):
        self._redis = redis_client
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._script = self._redis.register_script(self.LUA_SCRIPT)

    def allow_request(self, key: str) -> tuple:
        """リクエストを許可するか判定"""
        result = self._script(
            keys=[f"ratelimit:{key}"],
            args=[self._max_tokens, self._refill_rate, time.time()]
        )
        allowed = bool(result[0])
        remaining = int(result[1])
        return allowed, remaining

# 使用例
limiter = TokenBucketLimiter(
    redis.Redis(host='localhost'),
    max_tokens=100,    # バースト: 最大100リクエスト
    refill_rate=10,    # 定常: 10 req/sec
)

allowed, remaining = limiter.allow_request("user:123:/api/orders")
if not allowed:
    return Response("Rate limit exceeded", status=429)
```

### 2.2 Sliding Window Log

```python
# Sliding Window Log 実装
class SlidingWindowLogLimiter:
    """正確な時間窓でのレート制限"""

    LUA_SCRIPT = """
    local key = KEYS[1]
    local max_requests = tonumber(ARGV[1])
    local window_size = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    -- ウィンドウ外の古いエントリを削除
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window_size)

    -- 現在のカウント
    local count = redis.call('ZCARD', key)

    if count < max_requests then
        -- リクエストを記録
        redis.call('ZADD', key, now, now .. ':' .. math.random(1000000))
        redis.call('EXPIRE', key, window_size)
        return {1, max_requests - count - 1}   -- 許可
    else
        return {0, 0}                           -- 拒否
    end
    """

    def __init__(self, redis_client, max_requests=100, window_seconds=60):
        self._redis = redis_client
        self._max = max_requests
        self._window = window_seconds
        self._script = self._redis.register_script(self.LUA_SCRIPT)

    def allow_request(self, key: str) -> tuple:
        result = self._script(
            keys=[f"ratelimit:sw:{key}"],
            args=[self._max, self._window, time.time()]
        )
        return bool(result[0]), int(result[1])
```

### 2.3 Fixed Window Counter

```python
# Fixed Window Counter（最もシンプル）
class FixedWindowLimiter:
    def __init__(self, redis_client, max_requests=100, window_seconds=60):
        self._redis = redis_client
        self._max = max_requests
        self._window = window_seconds

    def allow_request(self, key: str) -> tuple:
        window_key = f"ratelimit:fw:{key}:{int(time.time()) // self._window}"
        pipe = self._redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, self._window)
        count, _ = pipe.execute()

        allowed = count <= self._max
        remaining = max(0, self._max - count)
        return allowed, remaining
```

---

## 3. Flask / FastAPI ミドルウェア統合

```python
# FastAPI ミドルウェアとしての統合
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limiter: TokenBucketLimiter):
        super().__init__(app)
        self.limiter = limiter

    async def dispatch(self, request: Request, call_next):
        # レートリミットキーの構築
        client_id = request.headers.get('X-API-Key', request.client.host)
        endpoint = request.url.path
        key = f"{client_id}:{endpoint}"

        allowed, remaining = self.limiter.allow_request(key)

        if not allowed:
            return Response(
                content='{"error":"Rate limit exceeded"}',
                status_code=429,
                headers={
                    'Retry-After': '60',
                    'X-RateLimit-Limit': '100',
                    'X-RateLimit-Remaining': '0',
                },
                media_type='application/json',
            )

        response = await call_next(request)
        response.headers['X-RateLimit-Remaining'] = str(remaining)
        return response

app.add_middleware(RateLimitMiddleware, limiter=limiter)
```

---

## 4. アルゴリズム比較表

| アルゴリズム | メモリ使用量 | 精度 | バースト許容 | 実装の複雑さ |
|------------|:----------:|:----:|:----------:|:----------:|
| Token Bucket | O(1) | 高 | 制御可能 | 中 |
| Leaky Bucket | O(1) | 高 | なし（均等化） | 中 |
| Fixed Window | O(1) | 低（境界問題） | あり（境界で2倍） | 低 |
| Sliding Window Log | O(N) | 最高 | なし | 高 |
| Sliding Window Counter | O(1) | 中〜高 | 軽微 | 中 |

| ユースケース | 推奨アルゴリズム | 理由 |
|------------|---------------|------|
| API ゲートウェイ | Token Bucket | バースト許容 + 一定レートの両立 |
| DDoS 防御 | Fixed Window | シンプルで高速 |
| 課金 API | Sliding Window Log | 正確なカウントが必要 |
| ストリーミング | Leaky Bucket | 均等な処理レート維持 |

---

## 5. アンチパターン

### アンチパターン 1: レートリミットをアプリケーションメモリで管理

```
BAD: 各サーバーのメモリでカウント管理
  Server A: user-123 = 50 requests
  Server B: user-123 = 50 requests
  → 合計100リクエストだが各サーバーは50と認識 → 制限が効かない

GOOD: Redis で一元管理
  Server A --count--> [Redis] <--count-- Server B
  → user-123 = 100 (正確な合計値)
```

### アンチパターン 2: 単一のレート制限ルールのみ

```
BAD: 全ユーザー・全エンドポイントに同一制限
  100 req/min for everyone, every endpoint

GOOD: 多層ルール
  - 無料プラン: 60 req/min
  - 有料プラン: 1000 req/min
  - /api/search: 30 req/min (高コスト)
  - /api/health: 制限なし
  - IP 単位: 1000 req/min (DDoS 防御)
```

---

## 6. FAQ

### Q1. レートリミッターの配置はどこが最適か？

**A.** 多層防御が理想。(1) CDN/エッジ層で IP 単位の粗い制限（DDoS防御）、(2) API ゲートウェイで API Key/プラン別の制限、(3) アプリケーション層で機能固有の制限。全てをゲートウェイに集約すると単一障害点になるため、クリティカルな制限は複数層で冗長化する。

### Q2. Redis がダウンしたらどうなる？

**A.** フェイルオープン（制限なしで通す）かフェイルクローズ（全拒否）かのポリシーを決めておく。一般的にはフェイルオープンを採用し、Redis 復旧まで一時的に制限を緩和する。Redis Cluster やレプリケーションで可用性を確保し、ローカルキャッシュ（Guava / Caffeine）をフォールバックに使う手法もある。

### Q3. クライアント側でのレートリミット対応はどう実装する？

**A.** (1) レスポンスヘッダー `X-RateLimit-Remaining` を監視し、残量が少なくなったらリクエスト間隔を広げる。(2) 429 レスポンスを受けたら `Retry-After` ヘッダーの秒数だけ待って再試行する。(3) Exponential Backoff + Jitter を実装し、多数のクライアントが同時にリトライする「Thundering Herd」を防ぐ。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 過負荷防止、不正利用対策、公平なリソース配分 |
| 配置 | 多層防御（Edge → Gateway → Application） |
| Token Bucket | バースト許容 + 定常レート制御。最も汎用的 |
| Sliding Window | 高精度なカウント。課金 API に最適 |
| 分散環境 | Redis + Lua スクリプトでアトミックな操作 |
| HTTP ヘッダー | X-RateLimit-*, Retry-After で制限状態を通知 |
| フォールバック | Redis 障害時のフェイルオープン/クローズ戦略 |

---

## 次に読むべきガイド

- [検索エンジン設計](./04-search-engine.md) — 検索 API のレート制限設計
- [CDN](../01-components/03-cdn.md) — エッジ層でのレート制限
- [API設計](../../clean-code-principles/docs/03-practices-advanced/03-api-design.md) — API のレスポンスヘッダー設計

---

## 参考文献

1. **System Design Interview** — Alex Xu (Byte Code LLC, 2020) — レートリミッターの設計面接ガイド
2. **Designing Data-Intensive Applications** — Martin Kleppmann (O'Reilly, 2017) — 分散システムの基礎理論
3. **Stripe Rate Limiting** — https://stripe.com/blog/rate-limiters — 実運用でのレートリミッター設計事例
4. **RFC 6585** — Additional HTTP Status Codes — 429 Too Many Requests の仕様
