# レートリミッター設計

> APIやサービスへのリクエスト頻度を制御し、過負荷・不正利用・DDoS攻撃からシステムを保護するレートリミッターの設計原則とアルゴリズム、分散環境での実装手法を解説する。Token Bucket、Sliding Window 等の主要アルゴリズムを Redis + Lua スクリプトで実装し、多層防御アーキテクチャを構築する。

---

## この章で学ぶこと

1. **レートリミッティングの基本概念** — なぜ必要か、どこに配置するか、HTTP 429 の設計、レスポンスヘッダーの標準
2. **主要アルゴリズムの仕組みと比較** — Token Bucket、Leaky Bucket、Fixed Window、Sliding Window Log、Sliding Window Counter の内部動作と使い分け
3. **分散環境でのレートリミッター実装** — Redis + Lua スクリプトによるアトミック操作、レースコンディション対策、フェイルオーバー戦略
4. **多層防御とプロダクション運用** — Edge / Gateway / Application 層の役割分担、監視、動的ルール変更、グレースフルデグラデーション

---

## 前提知識

このガイドを読む前に、以下の知識があるとスムーズに理解できます。

| トピック | 参照先 |
|---------|--------|
| システム設計の基礎概念 | [システム設計概要](../00-fundamentals/00-system-design-overview.md) |
| スケーラビリティの原則 | [スケーラビリティ](../00-fundamentals/01-scalability.md) |
| 可用性と信頼性 | [信頼性](../00-fundamentals/02-reliability.md) |
| キャッシング戦略 | [キャッシング](../01-components/01-caching.md) |
| ロードバランサーの仕組み | [ロードバランサー](../01-components/00-load-balancer.md) |
| CDN の基礎 | [CDN](../01-components/03-cdn.md) |
| API 設計のベストプラクティス | [API 設計](../../clean-code-principles/docs/03-practices-advanced/03-api-design.md) |
| Proxy パターン | [Proxy パターン](../../design-patterns-guide/docs/01-structural/03-proxy.md) |

---

## 1. レートリミッターの全体設計

### 1.1 なぜレートリミッターが必要か

```
レートリミッターが解決する問題:

1. サービス保護 (Availability)
   - 正当なトラフィック急増 (バイラルコンテンツ等) からバックエンドを保護
   - カスケード障害の防止: 1つのサービスの過負荷が他に波及するのを防ぐ
   - リソースの公平な配分

2. セキュリティ (Security)
   - DDoS 攻撃の軽減
   - ブルートフォース攻撃の防止 (ログイン試行回数の制限)
   - Web スクレイピングの抑制
   - API キーの不正利用検知

3. コスト管理 (Cost)
   - クラウドリソースの過剰消費を防止
   - サードパーティ API のコスト制御
   - 従量課金サービスの予算管理

4. ビジネスルール (Business Logic)
   - 無料/有料プランの差別化 (API 呼び出し数制限)
   - SLA の実装 (契約に基づくリクエスト数の保証)
   - フェアユースポリシーの強制

WHY: レートリミッターがないとどうなるか？
  +-----------------+------------------------------------+
  | シナリオ         | 結果                                |
  +-----------------+------------------------------------+
  | トラフィック急増  | サーバーダウン → 全ユーザーに影響     |
  | DDoS 攻撃       | サービス停止 → SLA 違反              |
  | バグのある Client | 無限ループで API 呼び出し → 高額請求  |
  | スクレイピング   | DB 負荷増大 → レスポンス劣化          |
  | プラン超過       | 無料ユーザーが有料相当の利用 → 損失    |
  +-----------------+------------------------------------+
```

### 1.2 配置パターン

```
レートリミッターの多層配置

  Client
    |
    v
  +---------------------------------------------+
  |  Layer 1: CDN / Edge (Cloudflare, AWS WAF)   |
  |  ・IP 単位の粗い制限 (DDoS 防御)              |
  |  ・地域別の制限                               |
  |  ・Bot 検知                                   |
  +---------------------------------------------+
    |
    v
  +---------------------------------------------+
  |  Layer 2: API Gateway (Kong, Envoy, Nginx)   |
  |  ・API Key / Client ID 別の制限              |
  |  ・エンドポイント別の制限                      |
  |  ・プラン別の制限 (Free: 60/min, Pro: 1000/min)|
  +---------------------------------------------+
    |
    v
  +---------------------------------------------+
  |  Layer 3: Application (ミドルウェア)           |
  |  ・ユーザー ID 別の制限                       |
  |  ・機能固有の制限 (検索: 30/min, 投稿: 10/min) |
  |  ・ビジネスルールに基づく制限                   |
  +---------------------------------------------+
    |
    v
  +---------------------------------------------+
  |  Layer 4: Database / External Service         |
  |  ・コネクションプール制限                      |
  |  ・外部 API のレート制限遵守                    |
  |  ・書き込みレートの制御                        |
  +---------------------------------------------+

  ★ 多層防御の原則: 外側で粗い制限、内側で細かい制限
  ★ 各層が独立して動作し、1層の障害が全体に影響しない
```

### 1.3 レスポンス設計

```
HTTP 429 レスポンスの標準設計:

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
    "retry_after": 30,
    "limit": 100,
    "window": "1m"
  }
}

レスポンスヘッダーの標準:
+------------------------+------------------------------------------+
| ヘッダー                | 意味                                     |
+------------------------+------------------------------------------+
| X-RateLimit-Limit      | ウィンドウ内の最大リクエスト数               |
| X-RateLimit-Remaining  | 残りリクエスト数                            |
| X-RateLimit-Reset      | 制限がリセットされる Unix タイムスタンプ       |
| Retry-After            | リトライまでの待ち時間 (秒)                  |
+------------------------+------------------------------------------+

★ 正常レスポンスにも X-RateLimit-* ヘッダーを含める
  → クライアントが事前に制限状況を把握可能
  → 429 になる前に自主的にスロットリング可能
```

### 1.4 システム全体構成図

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
    |         +--> [Redis Sentinel] (高可用性)
    |         +--> [Local Cache] (フォールバック)
    |
    v
  [Application Server]
    |
    v
  [Backend Services]

  +---> [Rules DB] (動的ルール管理)
  |         |
  |    [API Key: "sk_abc" → Plan: "pro" → Limit: 1000/min]
  |    [Endpoint: "/api/search" → Limit: 30/min]
  |
  +---> [Metrics Collector] (Prometheus)
  |         |
  |    [rate_limit_total{status="allowed|rejected"}]
  |    [rate_limit_remaining{key="user:123"}]
  |
  +---> [Alerting] (PagerDuty)
            |
       [Alert: "User X exceeded 10x normal usage"]
```

---

## 2. アルゴリズム詳解

### 2.1 Token Bucket

```
Token Bucket アルゴリズム

  概念:
  +--------------------------+
  |  Token Bucket            |
  |                          |   ← トークンが一定レートで補充
  |  [T] [T] [T] [T] [T]   |      (例: refill_rate = 10 tokens/sec)
  |  [T] [T] [T]            |
  |  max_tokens = 10        |   ← バケット容量 = バースト許容量
  +-----------+--------------+
              |
        リクエスト到着
        → トークンあり: 1トークン消費して処理
        → トークンなし: 429 拒否

  特性:
  - バースト許容: max_tokens までの瞬間的な大量リクエストを許可
  - 定常レート制御: refill_rate で長期的なレートを制御
  - メモリ効率: O(1) (トークン数と最終補充時刻のみ保持)

  パラメータの意味:
  +------------------+----------------------------------------+
  | max_tokens = 100 | 瞬間的に100リクエストまで許容             |
  | refill_rate = 10 | 定常状態で毎秒10リクエストまで許容         |
  +------------------+----------------------------------------+

  時系列での動作例:
  t=0.0s: tokens=100 (初期値)
  t=0.0s: 50 requests → tokens=50 (バースト消費)
  t=0.1s: refill +1 → tokens=51
  t=0.5s: refill +4 → tokens=55 (0.1~0.5sで4トークン補充)
  t=1.0s: refill +5 → tokens=60
  ...
  t=10.0s: tokens=100 (max に到達、それ以上は補充されない)

  WHY Token Bucket が最も広く使われるか:
  1. バーストと定常レートの両方を制御可能
  2. パラメータが直感的 (max=バースト、rate=定常)
  3. メモリ効率が良い (O(1))
  4. 実装が比較的シンプル
  → AWS API Gateway, Stripe, GitHub API で採用
```

```python
# コード例 1: Token Bucket 実装 (Redis + Lua Script)
import redis
import time
from typing import NamedTuple

class RateLimitResult(NamedTuple):
    """レート制限の判定結果"""
    allowed: bool
    remaining: int
    reset_at: float
    limit: int

class TokenBucketLimiter:
    """
    Token Bucket レートリミッター。

    WHY Lua Script を使うのか:
    1. アトミック操作: 複数の Redis コマンドを1回のラウンドトリップで実行
    2. レースコンディション防止: 複数サーバーからの同時アクセスでも正確
    3. パフォーマンス: ネットワークラウンドトリップを最小化

    WHY Redis Hash を使うのか:
    1. tokens と last_refill を1つのキーで管理 → アトミックに更新可能
    2. TTL で自動クリーンアップ → メモリリーク防止
    """

    LUA_SCRIPT = """
    local key = KEYS[1]
    local max_tokens = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])   -- tokens per second
    local now = tonumber(ARGV[3])

    -- 現在の状態を取得
    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1]) or max_tokens
    local last_refill = tonumber(data[2]) or now

    -- トークン補充 (経過時間に比例)
    local elapsed = math.max(0, now - last_refill)
    local new_tokens = math.min(max_tokens, tokens + elapsed * refill_rate)

    if new_tokens >= 1 then
        -- トークン消費: 許可
        new_tokens = new_tokens - 1
        redis.call('HMSET', key, 'tokens', new_tokens, 'last_refill', now)
        redis.call('EXPIRE', key, math.ceil(max_tokens / refill_rate) * 2)
        return {1, math.floor(new_tokens), 0}   -- {許可, 残りトークン, 待ち時間}
    else
        -- トークン不足: 拒否
        local wait_time = (1 - new_tokens) / refill_rate
        return {0, 0, math.ceil(wait_time)}      -- {拒否, 残り0, 待ち時間(秒)}
    end
    """

    def __init__(self, redis_client: redis.Redis,
                 max_tokens: int = 100,
                 refill_rate: float = 10.0):
        """
        Args:
            redis_client: Redis クライアント
            max_tokens: バケット容量 (バースト許容量)
            refill_rate: 補充レート (tokens/sec)
        """
        self._redis = redis_client
        self._max_tokens = max_tokens
        self._refill_rate = refill_rate
        self._script = self._redis.register_script(self.LUA_SCRIPT)

    def allow_request(self, key: str) -> RateLimitResult:
        """
        リクエストを許可するか判定する。

        Args:
            key: レートリミットキー (例: "user:123:/api/orders")

        Returns:
            RateLimitResult: 判定結果
        """
        now = time.time()
        result = self._script(
            keys=[f"ratelimit:tb:{key}"],
            args=[self._max_tokens, self._refill_rate, now]
        )
        allowed = bool(result[0])
        remaining = int(result[1])
        retry_after = float(result[2])

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=now + retry_after if not allowed else 0,
            limit=self._max_tokens,
        )

# 使用例
limiter = TokenBucketLimiter(
    redis.Redis(host='localhost'),
    max_tokens=100,    # バースト: 最大100リクエスト
    refill_rate=10,    # 定常: 10 req/sec
)

result = limiter.allow_request("user:123:/api/orders")
if not result.allowed:
    return Response(
        content='{"error":"Rate limit exceeded"}',
        status_code=429,
        headers={
            "Retry-After": str(int(result.reset_at - time.time())),
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": "0",
        },
    )
```

### 2.2 Sliding Window Log

```
Sliding Window Log アルゴリズム

  概念:
  各リクエストのタイムスタンプをログ (Sorted Set) に記録し、
  ウィンドウ内のリクエスト数をカウントする。

  時間軸:
  |----window (60s)----|
  |                    |
  t-60s              t (now)
  [req1][req2]...[reqN]
  ← この範囲のリクエスト数をカウント

  ウィンドウ外の古いリクエストは削除:
  [old1][old2]|[req1][req2]...[reqN]|
  ← 削除      ← カウント対象         ← now

  特性:
  - 精度: 最高 (各リクエストの正確なタイムスタンプを保持)
  - メモリ: O(N) (ウィンドウ内のリクエスト数に比例)
  - 用途: 課金 API など正確なカウントが必要な場面

  WHY Sorted Set を使うのか:
  - ZREMRANGEBYSCORE: O(log N + M) でウィンドウ外を効率削除
  - ZCARD: O(1) でカウント取得
  - Redis のアトミック操作でレースコンディション防止
```

```python
# コード例 2: Sliding Window Log 実装
class SlidingWindowLogLimiter:
    """
    Sliding Window Log レートリミッター。

    正確な時間窓でのレート制限を提供する。
    各リクエストのタイムスタンプを Redis Sorted Set に記録し、
    ウィンドウ内のリクエスト数をカウントする。

    トレードオフ:
    - 長所: 最も正確なカウント、Fixed Window の境界問題なし
    - 短所: メモリ使用量が O(N)、高トラフィックではコスト大
    """

    LUA_SCRIPT = """
    local key = KEYS[1]
    local max_requests = tonumber(ARGV[1])
    local window_size = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local request_id = ARGV[4]

    -- ウィンドウ外の古いエントリを削除
    redis.call('ZREMRANGEBYSCORE', key, 0, now - window_size)

    -- 現在のカウント
    local count = redis.call('ZCARD', key)

    if count < max_requests then
        -- リクエストを記録 (member はユニークにする)
        redis.call('ZADD', key, now, request_id)
        redis.call('EXPIRE', key, window_size + 1)
        return {1, max_requests - count - 1}   -- 許可, 残り
    else
        -- 制限超過: 最も古いリクエストの時刻を取得して待ち時間を計算
        local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
        local retry_after = 0
        if #oldest > 0 then
            retry_after = math.ceil(tonumber(oldest[2]) + window_size - now)
        end
        return {0, 0, retry_after}             -- 拒否, 残り0, 待ち時間
    end
    """

    def __init__(self, redis_client, max_requests: int = 100,
                 window_seconds: int = 60):
        self._redis = redis_client
        self._max = max_requests
        self._window = window_seconds
        self._script = self._redis.register_script(self.LUA_SCRIPT)

    def allow_request(self, key: str) -> RateLimitResult:
        now = time.time()
        request_id = f"{now}:{id(self)}:{hash(key)}"

        result = self._script(
            keys=[f"ratelimit:sw:{key}"],
            args=[self._max, self._window, now, request_id]
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        retry_after = float(result[2]) if len(result) > 2 else 0

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=now + retry_after if not allowed else 0,
            limit=self._max,
        )
```

### 2.3 Fixed Window Counter

```
Fixed Window Counter アルゴリズム

  概念:
  時間を固定ウィンドウに分割し、各ウィンドウ内のカウントを管理する。

  |--- Window 1 (10:00-10:01) ---|--- Window 2 (10:01-10:02) ---|
  |  [req][req][req]...[req]     |  [req][req]...               |
  |  count = 95                  |  count = 12                  |
  |  limit = 100                 |  limit = 100                 |

  境界問題 (WHY Fixed Window は精度が低いか):
  |--- Window 1 ---|--- Window 2 ---|
  |         [90req]|[90req]         |
  |    last 30s    | first 30s     |
  → 60秒のウィンドウで100件制限なのに、
    30秒間に180件が通ってしまう (境界をまたぐ)

  特性:
  - 精度: 低 (境界問題あり)
  - メモリ: O(1) (カウンターのみ)
  - 速度: 最速 (INCR のみ)
  - 用途: DDoS 防御など精度よりスピード重視の場面
```

```python
# コード例 3: Fixed Window Counter (最もシンプル)
class FixedWindowLimiter:
    """
    Fixed Window Counter レートリミッター。

    最もシンプルな実装。Redis の INCR + EXPIRE で実現。
    境界問題があるため、精度が必要な場面には不向き。

    WHY それでも使われるのか:
    1. 実装が最もシンプル (Lua Script 不要)
    2. メモリ効率が最良 (O(1))
    3. 処理速度が最速
    4. DDoS 防御など「大まかに制限できれば良い」ケースに最適
    """

    def __init__(self, redis_client, max_requests: int = 100,
                 window_seconds: int = 60):
        self._redis = redis_client
        self._max = max_requests
        self._window = window_seconds

    def allow_request(self, key: str) -> RateLimitResult:
        """
        Fixed Window でリクエストを判定。

        キー設計: ratelimit:fw:{key}:{window_id}
        window_id = int(now / window_seconds)
        → ウィンドウが変わると新しいキーが使われる
        → 古いキーは EXPIRE で自動削除
        """
        now = time.time()
        window_id = int(now) // self._window
        window_key = f"ratelimit:fw:{key}:{window_id}"

        pipe = self._redis.pipeline()
        pipe.incr(window_key)
        pipe.expire(window_key, self._window + 1)  # +1秒マージン
        count, _ = pipe.execute()

        allowed = count <= self._max
        remaining = max(0, self._max - count)
        reset_at = (window_id + 1) * self._window

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            limit=self._max,
        )
```

### 2.4 Sliding Window Counter (ハイブリッド)

```
Sliding Window Counter アルゴリズム

  概念:
  Fixed Window のシンプルさと Sliding Window の精度を両立する
  ハイブリッドアルゴリズム。

  前のウィンドウと現在のウィンドウのカウントを加重平均する。

  |--- Previous Window ---|--- Current Window ---|
  |  count_prev = 80      |  count_curr = 30     |
  |                       |      ^(now, 40%経過)  |
  |                       |                       |

  推定カウント = count_prev * (1 - 経過率) + count_curr
              = 80 * 0.6 + 30
              = 48 + 30 = 78

  精度: Fixed Window より大幅に改善、Sliding Window Log に近い
  メモリ: O(1) (前ウィンドウと現ウィンドウの2つのカウンターのみ)
  → 精度とメモリのバランスが最良
```

```python
# コード例 4: Sliding Window Counter 実装
class SlidingWindowCounterLimiter:
    """
    Sliding Window Counter (ハイブリッド方式)。

    Fixed Window の O(1) メモリ効率と、
    Sliding Window の精度を両立する。

    計算式:
    estimated_count = prev_window_count * overlap_ratio + current_window_count
    overlap_ratio = 1 - (current_time - current_window_start) / window_size
    """

    LUA_SCRIPT = """
    local curr_key = KEYS[1]
    local prev_key = KEYS[2]
    local max_requests = tonumber(ARGV[1])
    local window_size = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    -- 現在と前のウィンドウのカウントを取得
    local curr_count = tonumber(redis.call('GET', curr_key) or '0')
    local prev_count = tonumber(redis.call('GET', prev_key) or '0')

    -- 現在のウィンドウ内の経過率
    local window_start = math.floor(now / window_size) * window_size
    local elapsed_ratio = (now - window_start) / window_size

    -- スライディングウィンドウの推定カウント
    local estimated = prev_count * (1 - elapsed_ratio) + curr_count

    if estimated < max_requests then
        -- 許可: 現在のウィンドウのカウントをインクリメント
        redis.call('INCR', curr_key)
        redis.call('EXPIRE', curr_key, window_size * 2)
        return {1, math.floor(max_requests - estimated - 1)}
    else
        return {0, 0}
    end
    """

    def __init__(self, redis_client, max_requests: int = 100,
                 window_seconds: int = 60):
        self._redis = redis_client
        self._max = max_requests
        self._window = window_seconds
        self._script = self._redis.register_script(self.LUA_SCRIPT)

    def allow_request(self, key: str) -> RateLimitResult:
        now = time.time()
        curr_window = int(now) // self._window
        prev_window = curr_window - 1

        curr_key = f"ratelimit:swc:{key}:{curr_window}"
        prev_key = f"ratelimit:swc:{key}:{prev_window}"

        result = self._script(
            keys=[curr_key, prev_key],
            args=[self._max, self._window, now]
        )

        allowed = bool(result[0])
        remaining = int(result[1])

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=(curr_window + 1) * self._window if not allowed else 0,
            limit=self._max,
        )
```

### 2.5 Leaky Bucket

```
Leaky Bucket アルゴリズム

  概念:
  水が一定レートで漏れるバケツ。
  リクエストはバケツに入り、一定レートで処理される。

  +------------------+
  |  入力 (リクエスト) |   ← リクエストがバケツに入る
  +------------------+
  |                  |
  |  [req3]          |   ← バケツ (キュー)
  |  [req2]          |      容量 = burst_size
  |  [req1]          |
  +-------+----------+
          |
          v (一定レート)  ← leak_rate で処理 (例: 10 req/sec)
       [処理]

  バケツが満杯 → 新しいリクエストは破棄 (429)

  Token Bucket との違い:
  +-------------------+--------------------+---------------------+
  | 特性               | Token Bucket       | Leaky Bucket        |
  +-------------------+--------------------+---------------------+
  | バースト           | 許容 (max_tokens)   | 不許容 (均等化)      |
  | 出力レート         | 可変 (バースト時)    | 一定 (leak_rate)     |
  | メモリ             | O(1)               | O(N) (キュー)       |
  | 用途               | API Gateway        | ストリーミング       |
  +-------------------+--------------------+---------------------+

  WHY Leaky Bucket を選ぶケース:
  - ストリーミング処理: 均等なレートが必要
  - ネットワークトラフィック: バーストを平滑化したい
  - バッチ処理: 後続サービスの負荷を均一にしたい
```

```python
# コード例 5: Leaky Bucket 実装
class LeakyBucketLimiter:
    """
    Leaky Bucket レートリミッター。

    リクエストを均等なレートで処理する。
    バーストを平滑化し、後続サービスへの負荷を均一にする。
    """

    LUA_SCRIPT = """
    local key = KEYS[1]
    local burst_size = tonumber(ARGV[1])   -- バケツ容量
    local leak_rate = tonumber(ARGV[2])     -- 処理レート (req/sec)
    local now = tonumber(ARGV[3])

    local data = redis.call('HMGET', key, 'water_level', 'last_leak')
    local water_level = tonumber(data[1]) or 0
    local last_leak = tonumber(data[2]) or now

    -- 漏れた水の計算 (時間経過分を減少)
    local elapsed = math.max(0, now - last_leak)
    local leaked = elapsed * leak_rate
    water_level = math.max(0, water_level - leaked)

    if water_level < burst_size then
        -- バケツに空きあり: リクエストを受け入れ
        water_level = water_level + 1
        redis.call('HMSET', key, 'water_level', water_level, 'last_leak', now)
        redis.call('EXPIRE', key, math.ceil(burst_size / leak_rate) * 2)
        return {1, math.floor(burst_size - water_level)}
    else
        -- バケツ満杯: 拒否
        local wait_time = (water_level - burst_size + 1) / leak_rate
        return {0, 0, math.ceil(wait_time)}
    end
    """

    def __init__(self, redis_client, burst_size: int = 10,
                 leak_rate: float = 1.0):
        self._redis = redis_client
        self._burst_size = burst_size
        self._leak_rate = leak_rate
        self._script = self._redis.register_script(self.LUA_SCRIPT)

    def allow_request(self, key: str) -> RateLimitResult:
        now = time.time()
        result = self._script(
            keys=[f"ratelimit:lb:{key}"],
            args=[self._burst_size, self._leak_rate, now]
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        retry_after = float(result[2]) if len(result) > 2 else 0

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=now + retry_after if not allowed else 0,
            limit=self._burst_size,
        )
```

---

## 3. FastAPI / Flask ミドルウェア統合

### 3.1 FastAPI ミドルウェア

```python
# コード例 6: FastAPI ミドルウェアとしての統合
from fastapi import FastAPI, Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import json
import time

app = FastAPI()

class RateLimitConfig:
    """エンドポイント別のレート制限設定"""

    # デフォルト設定
    DEFAULT = {"max_requests": 100, "window_seconds": 60}

    # エンドポイント別の設定
    ENDPOINT_RULES = {
        "/api/v1/search": {"max_requests": 30, "window_seconds": 60},
        "/api/v1/login": {"max_requests": 5, "window_seconds": 300},
        "/api/v1/signup": {"max_requests": 3, "window_seconds": 3600},
        "/api/v1/health": None,  # None = 制限なし
    }

    # プラン別の設定
    PLAN_RULES = {
        "free": {"multiplier": 1.0},
        "starter": {"multiplier": 5.0},
        "pro": {"multiplier": 20.0},
        "enterprise": {"multiplier": 100.0},
    }

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI のレートリミッティングミドルウェア。

    キーの構築:
    1. API Key がある場合: API Key + エンドポイント
    2. 認証済みユーザー: User ID + エンドポイント
    3. 未認証: IP アドレス + エンドポイント

    レスポンスヘッダー:
    - 常に X-RateLimit-* ヘッダーを付与
    - 429 の場合は Retry-After も付与
    """

    def __init__(self, app, limiter: TokenBucketLimiter,
                 config: RateLimitConfig = None):
        super().__init__(app)
        self.limiter = limiter
        self.config = config or RateLimitConfig()

    async def dispatch(self, request: Request, call_next):
        # ヘルスチェック等の除外パス
        endpoint = request.url.path
        endpoint_rule = self.config.ENDPOINT_RULES.get(endpoint)
        if endpoint_rule is None and endpoint in self.config.ENDPOINT_RULES:
            # 明示的に None が設定されている = 制限なし
            return await call_next(request)

        # レートリミットキーの構築
        client_id = self._extract_client_id(request)
        key = f"{client_id}:{endpoint}"

        # 判定
        result = self.limiter.allow_request(key)

        if not result.allowed:
            retry_after = max(1, int(result.reset_at - time.time()))
            return Response(
                content=json.dumps({
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"リクエスト制限を超過しました。{retry_after}秒後に再試行してください。",
                        "retry_after": retry_after,
                    }
                }),
                status_code=429,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(result.reset_at)),
                },
            )

        # 正常レスポンスにもヘッダーを付与
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(result.remaining)
        return response

    def _extract_client_id(self, request: Request) -> str:
        """リクエストからクライアント識別子を抽出"""
        # 優先順位: API Key > User ID > IP
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"apikey:{api_key}"

        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            # JWT から user_id を抽出 (簡略化)
            user_id = extract_user_id_from_jwt(auth[7:])
            if user_id:
                return f"user:{user_id}"

        # フォールバック: IP アドレス
        client_ip = request.headers.get(
            "X-Forwarded-For", request.client.host
        )
        return f"ip:{client_ip}"

app.add_middleware(RateLimitMiddleware, limiter=limiter)
```

### 3.2 デコレータベースの制限

```python
# コード例 7: デコレータベースのエンドポイント別レート制限
import functools
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

def rate_limit(max_requests: int, window_seconds: int,
               key_func=None):
    """
    エンドポイント固有のレート制限デコレータ。

    使い方:
    @rate_limit(max_requests=10, window_seconds=60)
    async def search(query: str):
        ...

    key_func で制限キーをカスタマイズ:
    @rate_limit(max_requests=5, window_seconds=300,
                key_func=lambda req: req.client.host)
    async def login(credentials):
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, request: Request = None, **kwargs):
            if request is None:
                return await func(*args, **kwargs)

            # キーの構築
            if key_func:
                key = key_func(request)
            else:
                user_id = getattr(request.state, 'user_id', request.client.host)
                key = f"{user_id}:{request.url.path}"

            # 専用リミッターで判定
            limiter = SlidingWindowCounterLimiter(
                redis_client=get_redis(),
                max_requests=max_requests,
                window_seconds=window_seconds,
            )

            result = limiter.allow_request(key)
            if not result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": int(result.reset_at - time.time()),
                    },
                    headers={
                        "Retry-After": str(int(result.reset_at - time.time())),
                    },
                )

            return await func(*args, request=request, **kwargs)
        return wrapper
    return decorator

# 使用例
@router.get("/api/v1/search")
@rate_limit(max_requests=30, window_seconds=60)
async def search_products(query: str, request: Request):
    """検索 API: 1分あたり30リクエストに制限"""
    return await perform_search(query)

@router.post("/api/v1/auth/login")
@rate_limit(
    max_requests=5,
    window_seconds=300,
    key_func=lambda req: f"login:{req.client.host}",  # IP 別
)
async def login(credentials: dict, request: Request):
    """ログイン API: IP あたり5分で5回に制限 (ブルートフォース防止)"""
    return await authenticate(credentials)

@router.post("/api/v1/emails/send")
@rate_limit(max_requests=10, window_seconds=3600)
async def send_email(email_data: dict, request: Request):
    """メール送信 API: 1時間あたり10通に制限"""
    return await send_email_service(email_data)
```

---

## 4. 分散環境での考慮事項

### 4.1 レースコンディション

```
レースコンディションの問題:

  Server A                    Server B
     |                           |
  GET counter → 99              GET counter → 99
     |                           |
  99 < 100 → 許可               99 < 100 → 許可
     |                           |
  SET counter = 100             SET counter = 100
     |                           |
  → 実際は101リクエスト目が通ってしまう!

解決策 1: Lua スクリプト (推奨)
  → 全ての操作がアトミックに実行される
  → 本ガイドの全実装で採用

解決策 2: Redis WATCH + MULTI/EXEC
  → Optimistic Lock
  → 競合時はリトライが必要で複雑

解決策 3: Redis SET NX + GET (CAS操作)
  → Compare-and-Swap
  → Lua より遅いがシンプル

  ★ 結論: Lua スクリプトが最も効率的で推奨
```

### 4.2 Redis 障害時のフォールバック

```python
# コード例 8: Redis 障害時のフォールバック戦略
import time
from threading import Lock
from collections import defaultdict

class ResilientRateLimiter:
    """
    Redis 障害に耐性のあるレートリミッター。

    フォールバック戦略:
    1. Primary: Redis Cluster (高精度)
    2. Secondary: ローカルメモリ (近似値)
    3. Tertiary: フェイルオープン (制限なし)

    WHY フェイルオープンを選ぶか:
    - レートリミッターの障害でサービス停止は本末転倒
    - 短時間のレート制限なしは許容可能
    - 代わりにアラートを発報し、迅速に復旧を目指す
    """

    def __init__(self, redis_limiter: TokenBucketLimiter,
                 fail_open: bool = True):
        self.redis_limiter = redis_limiter
        self.fail_open = fail_open
        self._local_counters = defaultdict(lambda: {"count": 0, "reset": 0})
        self._lock = Lock()
        self._redis_healthy = True
        self._last_health_check = 0

    def allow_request(self, key: str) -> RateLimitResult:
        # Redis の健全性確認 (10秒ごと)
        if not self._redis_healthy:
            if time.time() - self._last_health_check > 10:
                self._check_redis_health()

        # Primary: Redis
        if self._redis_healthy:
            try:
                result = self.redis_limiter.allow_request(key)
                return result
            except Exception as e:
                self._redis_healthy = False
                self._last_health_check = time.time()
                metrics.increment("rate_limiter.redis_error")
                logger.error(f"Redis rate limiter failed: {e}")

        # Secondary: ローカルメモリ (近似)
        try:
            return self._local_rate_limit(key)
        except Exception:
            pass

        # Tertiary: フェイルオープン/クローズ
        if self.fail_open:
            metrics.increment("rate_limiter.fail_open")
            return RateLimitResult(
                allowed=True, remaining=-1, reset_at=0, limit=0
            )
        else:
            metrics.increment("rate_limiter.fail_closed")
            return RateLimitResult(
                allowed=False, remaining=0, reset_at=time.time() + 60, limit=0
            )

    def _local_rate_limit(self, key: str) -> RateLimitResult:
        """ローカルメモリによる近似的なレート制限"""
        now = time.time()
        with self._lock:
            counter = self._local_counters[key]

            # ウィンドウリセット
            if now > counter["reset"]:
                counter["count"] = 0
                counter["reset"] = now + 60  # 1分ウィンドウ

            counter["count"] += 1

            # ★ ローカルなのでサーバー台数分の誤差がある
            # 例: 4台のサーバーで100/min 制限
            #     → 各サーバーで25/min に設定
            local_limit = 25  # max_requests / num_servers
            allowed = counter["count"] <= local_limit

            return RateLimitResult(
                allowed=allowed,
                remaining=max(0, local_limit - counter["count"]),
                reset_at=counter["reset"],
                limit=local_limit,
            )

    def _check_redis_health(self):
        """Redis の健全性チェック"""
        try:
            self.redis_limiter._redis.ping()
            self._redis_healthy = True
            logger.info("Redis rate limiter recovered")
        except Exception:
            self._redis_healthy = False
            self._last_health_check = time.time()
```

### 4.3 動的ルール管理

```python
# コード例 9: 動的ルール管理サービス
import json

class RateLimitRuleService:
    """
    レート制限ルールを動的に管理するサービス。

    ルールの優先順位:
    1. ユーザー個別ルール (user:123 → 500/min)
    2. プラン別ルール (plan:pro → 1000/min)
    3. エンドポイント別ルール (/api/search → 30/min)
    4. デフォルトルール (100/min)

    WHY 動的ルールが必要か:
    - 特定ユーザーの一時的な制限緩和 (大量インポート等)
    - プラン変更の即時反映
    - インシデント対応での緊急制限
    - A/B テストでの制限値の実験
    """

    def __init__(self, redis_client, db):
        self.redis = redis_client
        self.db = db
        self._cache_ttl = 300  # 5分キャッシュ

    async def get_limit(self, client_id: str,
                         endpoint: str) -> dict:
        """適用されるレート制限ルールを取得"""

        # 1. キャッシュ確認
        cache_key = f"rate_rule:{client_id}:{endpoint}"
        cached = await self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # 2. ユーザー個別ルール
        user_rule = await self.db.fetch_one(
            "SELECT * FROM rate_limit_rules "
            "WHERE client_id = :cid AND endpoint = :ep AND is_active = TRUE",
            {"cid": client_id, "ep": endpoint},
        )
        if user_rule:
            rule = self._format_rule(user_rule)
            await self.redis.set(cache_key, json.dumps(rule), ex=self._cache_ttl)
            return rule

        # 3. プラン別ルール
        plan = await self._get_user_plan(client_id)
        plan_rule = await self.db.fetch_one(
            "SELECT * FROM rate_limit_rules "
            "WHERE client_id = :plan AND endpoint = :ep AND is_active = TRUE",
            {"plan": f"plan:{plan}", "ep": endpoint},
        )
        if plan_rule:
            rule = self._format_rule(plan_rule)
            await self.redis.set(cache_key, json.dumps(rule), ex=self._cache_ttl)
            return rule

        # 4. デフォルトルール
        default = {"max_requests": 100, "window_seconds": 60,
                    "algorithm": "token_bucket"}
        await self.redis.set(cache_key, json.dumps(default), ex=self._cache_ttl)
        return default

    async def set_temporary_override(self, client_id: str,
                                       endpoint: str,
                                       new_limit: dict,
                                       duration_hours: int = 24):
        """一時的なルールオーバーライド (インシデント対応等)"""
        await self.db.execute(
            "INSERT INTO rate_limit_overrides "
            "(client_id, endpoint, max_requests, window_seconds, expires_at) "
            "VALUES (:cid, :ep, :max, :win, NOW() + INTERVAL ':dur hours')",
            {
                "cid": client_id, "ep": endpoint,
                "max": new_limit["max_requests"],
                "win": new_limit["window_seconds"],
                "dur": duration_hours,
            },
        )

        # キャッシュを即時無効化
        cache_key = f"rate_rule:{client_id}:{endpoint}"
        await self.redis.delete(cache_key)

    def _format_rule(self, rule) -> dict:
        return {
            "max_requests": rule["max_requests"],
            "window_seconds": rule["window_seconds"],
            "algorithm": rule.get("algorithm", "token_bucket"),
        }
```

---

## 5. アルゴリズム比較表

### 5.1 技術的比較

| アルゴリズム | メモリ使用量 | 精度 | バースト許容 | 実装の複雑さ | レースコンディション |
|------------|:----------:|:----:|:----------:|:----------:|:-----------------:|
| Token Bucket | O(1) | 高 | 制御可能 | 中 | Lua で解決 |
| Leaky Bucket | O(1) | 高 | なし (均等化) | 中 | Lua で解決 |
| Fixed Window | O(1) | 低 (境界問題) | あり (境界で2倍) | 低 | Pipeline で十分 |
| Sliding Window Log | O(N) | 最高 | なし | 高 | Lua で解決 |
| Sliding Window Counter | O(1) | 中-高 | 軽微 | 中 | Lua で解決 |

### 5.2 ユースケース別推奨

| ユースケース | 推奨アルゴリズム | 理由 |
|------------|---------------|------|
| API ゲートウェイ | Token Bucket | バースト許容 + 一定レートの両立 |
| DDoS 防御 | Fixed Window | シンプルで高速、精度は二の次 |
| 課金 API | Sliding Window Log | 正確なカウントが必要 |
| ストリーミング | Leaky Bucket | 均等な処理レート維持 |
| 一般的な API | Sliding Window Counter | 精度とメモリ効率のバランス |
| ログイン (ブルートフォース防止) | Sliding Window Log | 正確なカウントで安全性確保 |

### 5.3 パフォーマンス比較

| アルゴリズム | Redis 操作数 (Lua) | レイテンシ (p99) | メモリ/キー |
|------------|:-----------------:|:---------------:|:----------:|
| Token Bucket | 3 (HMGET + HMSET + EXPIRE) | < 1ms | ~100 bytes |
| Fixed Window | 2 (INCR + EXPIRE) | < 0.5ms | ~50 bytes |
| Sliding Window Log | 3 (ZREMRANGE + ZCARD + ZADD) | < 2ms | ~100KB (1000 req) |
| Sliding Window Counter | 3 (GET + GET + INCR) | < 1ms | ~100 bytes |
| Leaky Bucket | 3 (HMGET + HMSET + EXPIRE) | < 1ms | ~100 bytes |

---

## 6. アンチパターン

### アンチパターン 1: レートリミットをアプリケーションメモリで管理

```python
# NG: 各サーバーのメモリでカウント管理
class BadInMemoryLimiter:
    def __init__(self):
        self.counters = {}  # ローカルメモリ

    def allow(self, key):
        count = self.counters.get(key, 0)
        if count >= 100:
            return False
        self.counters[key] = count + 1
        return True

# 問題:
#   Server A: user-123 = 50 requests
#   Server B: user-123 = 50 requests
#   → 合計100リクエストだが各サーバーは50と認識
#   → 制限が全く効かない (N台のサーバーで N倍の制限)

# さらに:
#   - サーバー再起動でカウンターがリセットされる
#   - メモリリークのリスク (キーの自動削除がない)


# OK: Redis で一元管理
class GoodRedisLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    def allow(self, key):
        # Redis で一元管理 → 全サーバーで共有
        result = self.redis.eval(LUA_SCRIPT, 1, key, ...)
        return bool(result[0])

# Redis クラスターの場合:
#   Server A --count--> [Redis Cluster] <--count-- Server B
#   → user-123 = 100 (正確な合計値)
```

### アンチパターン 2: 単一のレート制限ルールのみ

```python
# NG: 全ユーザー・全エンドポイントに同一制限
class BadSingleRuleLimiter:
    GLOBAL_LIMIT = 100  # req/min for everyone

    def allow(self, key):
        count = self.redis.incr(f"rate:{key}")
        return count <= self.GLOBAL_LIMIT

# 問題:
# 1. 無料ユーザーと有料ユーザーが同じ制限 → 有料の価値がない
# 2. /api/health (軽量) と /api/search (重量) が同じ制限
# 3. DDoS 防御と通常制限が区別できない


# OK: 多層・多次元のルール
class GoodMultiLayerLimiter:
    RULES = {
        # プラン別
        "plan:free":       {"default": 60, "search": 10},
        "plan:pro":        {"default": 1000, "search": 100},
        "plan:enterprise": {"default": 10000, "search": 1000},

        # エンドポイント別
        "endpoint:/api/search":  {"limit": 30, "window": 60},
        "endpoint:/api/health":  None,  # 制限なし
        "endpoint:/api/login":   {"limit": 5, "window": 300},

        # グローバル (DDoS 防御)
        "global:ip": {"limit": 1000, "window": 60},
    }

    def allow(self, client_id, endpoint, plan):
        # 1. グローバル IP 制限
        if not self._check_ip_limit(client_id):
            return False
        # 2. エンドポイント固有制限
        if not self._check_endpoint_limit(client_id, endpoint):
            return False
        # 3. プラン別制限
        if not self._check_plan_limit(client_id, plan, endpoint):
            return False
        return True
```

### アンチパターン 3: Retry-After を返さない

```python
# NG: 429 を返すだけで、いつリトライすべきか教えない
@app.get("/api/data")
async def bad_endpoint(request: Request):
    if not rate_limiter.allow(get_client_id(request)):
        return Response(
            content="Too Many Requests",
            status_code=429,
        )
        # クライアントはいつリトライすればよいか不明
        # → 即座にリトライしてさらに 429 を受ける
        # → Thundering Herd 問題


# OK: 標準ヘッダーで制限状態を通知
@app.get("/api/data")
async def good_endpoint(request: Request):
    result = rate_limiter.allow_request(get_client_id(request))

    if not result.allowed:
        retry_after = max(1, int(result.reset_at - time.time()))
        return Response(
            content=json.dumps({
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": f"{retry_after}秒後に再試行してください",
                    "retry_after": retry_after,
                }
            }),
            status_code=429,
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(result.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(result.reset_at)),
            },
        )

    response = await process_request(request)
    response.headers["X-RateLimit-Remaining"] = str(result.remaining)
    return response
```

### アンチパターン 4: クライアント側でレート制限を無視

```python
# NG: クライアント側でリトライ制御なし
class BadClient:
    async def call_api(self, url):
        for _ in range(100):  # 100回リトライ
            response = await http.get(url)
            if response.status == 200:
                return response
            # 429 でも即座にリトライ → サーバーに負荷

# OK: Exponential Backoff + Jitter + Retry-After 遵守
class GoodClient:
    async def call_api(self, url, max_retries=5):
        for attempt in range(max_retries):
            response = await http.get(url)

            if response.status == 200:
                return response

            if response.status == 429:
                # Retry-After ヘッダーを尊重
                retry_after = int(response.headers.get("Retry-After", 60))

                # Exponential Backoff + Jitter
                base_delay = min(retry_after, 2 ** attempt)
                jitter = random.uniform(0, base_delay * 0.5)
                wait_time = base_delay + jitter

                logger.warning(
                    f"Rate limited. Waiting {wait_time:.1f}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)

            elif response.status >= 500:
                await asyncio.sleep(2 ** attempt)
            else:
                raise APIError(response.status, await response.text())

        raise MaxRetriesExceeded(f"Failed after {max_retries} attempts")
```

---

## 7. 実践演習

### 演習 1（基礎）: 多層レートリミッターの設計

**課題**: IP 層とユーザー層の2段階レート制限を実装してください。

```python
# 要件:
# 1. IP 単位: 1000 req/min (DDoS 防御)
# 2. ユーザー単位: 100 req/min (通常利用)
# 3. 未認証の場合は IP 制限のみ適用
# 4. 両方の制限を通過した場合のみリクエストを許可
# 5. レスポンスヘッダーにはより厳しい方の残り回数を表示

# スケルトンコード:
class TwoLayerRateLimiter:
    def __init__(self, redis_client):
        self.ip_limiter = FixedWindowLimiter(
            redis_client, max_requests=1000, window_seconds=60
        )
        self.user_limiter = TokenBucketLimiter(
            redis_client, max_tokens=100, refill_rate=1.67
        )

    def check(self, ip: str, user_id: str = None) -> RateLimitResult:
        # TODO: IP 制限チェック
        # TODO: user_id があればユーザー制限チェック
        # TODO: より厳しい方の結果を返す
        pass
```

**期待される出力**:
```
# 認証済みユーザー (IP: 192.168.1.1, User: user-123)
IP layer:   allowed=True, remaining=955/1000
User layer: allowed=True, remaining=85/100
→ Result:   allowed=True, remaining=85 (より少ない方)

# 未認証ユーザー (IP: 10.0.0.1)
IP layer:   allowed=True, remaining=990/1000
User layer: (skip)
→ Result:   allowed=True, remaining=990

# IP 制限超過 (DDoS)
IP layer:   allowed=False, remaining=0/1000
→ Result:   allowed=False (ユーザー層チェックはスキップ)
```

---

### 演習 2（応用）: プラン別のレートリミッター

**課題**: 料金プランに応じたレート制限を動的に適用するシステムを実装してください。

```python
# 要件:
# 1. Free プラン: 60 req/min, バースト不可
# 2. Starter プラン: 300 req/min, バースト 50
# 3. Pro プラン: 1000 req/min, バースト 200
# 4. Enterprise プラン: カスタム制限
# 5. プラン変更は即時反映
# 6. API Key から自動でプランを判定

# スケルトンコード:
class PlanBasedRateLimiter:
    PLAN_CONFIGS = {
        "free":       {"algorithm": "fixed_window", "max": 60, "window": 60},
        "starter":    {"algorithm": "token_bucket", "max": 300, "refill": 5.0},
        "pro":        {"algorithm": "token_bucket", "max": 1000, "refill": 16.7},
        "enterprise": None,  # DB から取得
    }

    async def check(self, api_key: str) -> RateLimitResult:
        # TODO: API Key からプランを取得
        # TODO: プランに対応するアルゴリズムを選択
        # TODO: レート制限を適用
        pass
```

**期待される出力**:
```
API Key: sk_free_abc123
Plan: free
Algorithm: Fixed Window
→ Result: allowed=True, remaining=42/60, reset=2024-01-15T10:01:00Z

API Key: sk_pro_xyz789
Plan: pro
Algorithm: Token Bucket
→ Result: allowed=True, remaining=850/1000, refill_rate=16.7/sec
```

---

### 演習 3（発展）: 分散レートリミッターの障害耐性テスト

**課題**: Redis 障害時のフォールバック動作を含む、完全なレートリミッターシステムを設計・実装してください。

```python
# 要件:
# 1. 正常時: Redis Cluster でレート制限
# 2. Redis 障害時: ローカルメモリでフォールバック
# 3. フォールバック中のメトリクス収集
# 4. Redis 復旧の自動検知と切り戻し
# 5. 障害時のアラート発報
# 6. 障害耐性のテストシナリオ

# テストシナリオ:
class RateLimiterResiliencyTest:
    """
    テスト 1: Redis 正常動作
      → 100 req/min の制限が正確に動作すること

    テスト 2: Redis 障害発生
      → フォールバックに自動切替
      → ローカルメモリで近似的な制限が動作
      → alert が発報されること

    テスト 3: Redis 復旧
      → 10秒以内に Redis に切り戻しされること
      → ローカルカウンターがクリアされること

    テスト 4: ネットワーク分断 (Split Brain)
      → 各サーバーが独立してレート制限を維持
      → 全体としての精度は低下するが、サービスは継続
    """

    async def test_redis_failure_and_recovery(self):
        # TODO: Redis を停止
        # TODO: リクエストが通ることを確認 (フェイルオープン)
        # TODO: ローカルフォールバックの動作確認
        # TODO: Redis を復旧
        # TODO: Redis に切り戻されることを確認
        pass
```

**期待される出力**:
```
[Test 1: Normal Operation]
  Request 1-100: allowed=True (via Redis)
  Request 101:   allowed=False (rate limited)
  → PASS

[Test 2: Redis Failure]
  [10:00:00] Redis connection lost
  [10:00:00] Alert: "Rate limiter Redis failure - falling back to local"
  [10:00:00] Fallback: local memory limiter activated
  Request 1-25: allowed=True (via local, limit=25 per server)
  Request 26:   allowed=False (local rate limited)
  → PASS

[Test 3: Redis Recovery]
  [10:00:15] Redis connection restored
  [10:00:15] Log: "Rate limiter Redis recovered"
  Request next: allowed=True (via Redis, counter reset)
  → PASS
```

---

## 8. FAQ

### Q1. レートリミッターの配置はどこが最適か？

**A.** 多層防御が理想です。

1. **CDN/Edge 層** (Cloudflare, AWS WAF): IP 単位の粗い制限。DDoS 防御が主目的。クラウドベンダーの処理能力を活用し、自前のインフラに到達する前に異常トラフィックを遮断する。
2. **API Gateway 層** (Kong, Envoy, Nginx): API Key / プラン別の制限。認証済みクライアントの利用量を管理。ビジネスルールに基づく制限はここで実装。
3. **Application 層** (ミドルウェア): 機能固有の制限。検索 API は30 req/min、投稿は10 req/min のようなきめ細かい制御。ビジネスロジックに密接な制限を実装。

全てを1箇所に集約すると単一障害点になるため、クリティカルな制限は複数層で冗長化する。

### Q2. Redis がダウンしたらどうなる？

**A.** フェイルオープン (制限なしで通す) かフェイルクローズ (全拒否) かのポリシーを事前に決めておきます。

- **フェイルオープン** (推奨): 一般的なケースではこちらを採用。レートリミッターの障害でサービス全体が止まるのは本末転倒。Redis 復旧まで一時的に制限を緩和し、ローカルメモリキャッシュ (Guava / Caffeine) をフォールバックに使う。
- **フェイルクローズ**: セキュリティ重視の場面 (ログイン試行制限、決済 API) ではこちら。不正アクセスのリスクがサービス停止より大きい場合。
- **ハイブリッド**: エンドポイントごとに戦略を切り替え。`/api/login` はフェイルクローズ、`/api/products` はフェイルオープン。
- **Redis Cluster + Sentinel** で可用性を確保し、障害自体を極力発生させない。

### Q3. クライアント側でのレートリミット対応はどう実装する？

**A.** 以下の3つを組み合わせます。

1. **予防的スロットリング**: レスポンスヘッダー `X-RateLimit-Remaining` を監視し、残量が少なくなったらリクエスト間隔を広げる。制限の 80% に達した時点でスロットリングを開始するのが一般的。
2. **リアクティブリトライ**: 429 レスポンスを受けたら `Retry-After` ヘッダーの秒数だけ待って再試行する。ヘッダーがない場合は Exponential Backoff を使用。
3. **Thundering Herd 対策**: Exponential Backoff + Jitter を実装し、多数のクライアントが同時にリトライする問題を防ぐ。`jitter = random.uniform(0, base_delay)` を追加するだけで大幅に改善する。

### Q4. マイクロサービス間のレートリミットはどう設計する？

**A.** サービスメッシュ (Istio, Envoy) を活用して、サービス間通信にもレート制限を適用します。

- **サービス間トークン**: 各マイクロサービスに「呼び出しクォータ」を割り当て。Service A は Service B を 1000 req/min まで呼び出せる。
- **Circuit Breaker との連携**: レート制限 + サーキットブレーカーを組み合わせ。レート制限超過が続く場合はサーキットを開いて呼び出し自体を停止。
- **バックプレッシャー**: 下流サービスが過負荷の場合、上流に 429 を返して負荷を伝搬。Kafka のようなキューを挟んでバッファリングする方法もある。

### Q5. レートリミッターのテストはどうする？

**A.** 以下の観点でテストを行います。

1. **ユニットテスト**: 各アルゴリズムの正確性。制限内のリクエストが通ること、制限超過で拒否されること。
2. **タイミングテスト**: ウィンドウのリセット、トークンの補充が正確に動作すること。
3. **同時実行テスト**: 複数スレッド/プロセスからの同時アクセスでレースコンディションが発生しないこと。
4. **障害テスト**: Redis 障害時のフォールバック動作。復旧時の切り戻し。
5. **負荷テスト**: 10万 req/sec でのパフォーマンス。Redis の CPU/メモリ使用量の確認。

---

## 9. 高度なトピック

### 9.1 分散レートリミッターの一貫性

```
分散環境でのレート制限の課題:

問題: 複数の Redis ノードに分散された場合の整合性

  [App Server 1] --> [Redis Node A] : user-123 = 50
  [App Server 2] --> [Redis Node B] : user-123 = 50
  → 合計 100 だが、各ノードは 50 と認識

解決策:

1. 単一 Redis ノード (推奨)
   - ハッシュスロットで user-123 のキーは常に同じノードに配置
   - Redis Cluster のキーベースルーティングを活用
   - 注意: {user:123} のようにハッシュタグを使って
     関連キーを同一ノードに配置する

2. Gossip Protocol (近似)
   - 各ノードがローカルカウントを持ち、定期的に同期
   - 完全な精度は保証されないが、スケーラビリティに優れる
   - Envoy のグローバルレートリミッターで採用

3. Central Coordinator
   - 全てのリクエストを1つのコーディネーターが処理
   - 精度は最高だが、単一障害点になる
```

### 9.2 適応型レートリミッティング

```python
# コード例 10: サーバー負荷に応じた動的レート制限
class AdaptiveRateLimiter:
    """
    サーバーの負荷状況に応じてレート制限を動的に調整する。

    WHY 適応型が必要か:
    - 固定制限だと、サーバーに余裕がある時にリクエストを不要に拒否
    - 負荷が高い時には固定制限では不十分な場合がある
    - トラフィックパターンは時間帯によって大きく変動する

    戦略:
    - CPU 使用率 > 80% → 制限を厳しくする (50% に削減)
    - CPU 使用率 > 90% → 最低限のみ許可 (20% に削減)
    - CPU 使用率 < 50% → 制限を緩和 (150% に拡大)
    """

    def __init__(self, base_limiter, health_checker):
        self.base_limiter = base_limiter
        self.health_checker = health_checker
        self._adjustment_factor = 1.0

    async def update_factor(self):
        """定期的に (10秒ごと) 調整係数を更新"""
        health = await self.health_checker.get_metrics()

        cpu_usage = health["cpu_percent"]
        if cpu_usage > 90:
            self._adjustment_factor = 0.2
        elif cpu_usage > 80:
            self._adjustment_factor = 0.5
        elif cpu_usage > 60:
            self._adjustment_factor = 0.8
        elif cpu_usage < 30:
            self._adjustment_factor = 1.5
        else:
            self._adjustment_factor = 1.0

        metrics.gauge("rate_limit.adjustment_factor", self._adjustment_factor)

    def get_effective_limit(self, base_limit: int) -> int:
        """調整後の制限値を返す"""
        return max(1, int(base_limit * self._adjustment_factor))
```

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| 目的 | 過負荷防止、不正利用対策、公平なリソース配分、ビジネスルール強制 |
| 配置 | 多層防御 (Edge → Gateway → Application) |
| Token Bucket | バースト許容 + 定常レート制御。最も汎用的。AWS, Stripe で採用 |
| Sliding Window Counter | 精度と効率のバランス。一般 API に最適 |
| Sliding Window Log | 最高精度。課金 API やセキュリティ重視に |
| Fixed Window | 最もシンプル・高速。DDoS 防御に |
| Leaky Bucket | 均等レート出力。ストリーミングに |
| 分散環境 | Redis + Lua スクリプトでアトミックな操作 |
| HTTP ヘッダー | X-RateLimit-*, Retry-After で制限状態を通知 |
| フォールバック | Redis 障害時のフェイルオープン/クローズ + ローカルキャッシュ |
| 動的ルール | DB + Redis キャッシュでプラン別・エンドポイント別ルール管理 |
| 適応型 | サーバー負荷に応じて制限値を動的に調整 |

---

## 11. 設計面接での回答フレームワーク

```
レートリミッターの設計面接で聞かれるポイント:

1. 要件の明確化 (5分)
   - クライアント側 or サーバー側？ → サーバー側
   - 制限の粒度は？ → IP / API Key / User ID
   - 分散環境？ → はい、複数サーバー
   - 精度の要件は？ → 近似で OK or 正確に

2. 高レベル設計 (10分)
   - 多層配置 (Edge → Gateway → App)
   - Redis でカウンター管理
   - HTTP 429 + Retry-After

3. アルゴリズム選択 (10分)
   - Token Bucket (汎用) vs Sliding Window (高精度)
   - トレードオフの説明

4. 詳細設計 (10分)
   - Lua スクリプトでアトミック操作
   - レースコンディション対策
   - Redis 障害時のフォールバック

5. 運用 (5分)
   - 動的ルール変更
   - 監視とアラート
   - 適応型レートリミッティング
```

---

## 次に読むべきガイド

- [検索エンジン設計](./04-search-engine.md) — 検索 API のレート制限設計
- [通知システム設計](./02-notification-system.md) — 通知のレート制限
- [CDN](../01-components/03-cdn.md) — エッジ層でのレート制限
- [ロードバランサー](../01-components/00-load-balancer.md) — L4/L7 でのレート制限
- [API設計](../../clean-code-principles/docs/03-practices-advanced/03-api-design.md) — API のレスポンスヘッダー設計
- [Proxy パターン](../../design-patterns-guide/docs/01-structural/03-proxy.md) — レートリミッターを Proxy として実装
- [Strategy パターン](../../design-patterns-guide/docs/02-behavioral/01-strategy.md) — アルゴリズム切り替えの設計
- [信頼性](../00-fundamentals/02-reliability.md) — フォールバック戦略の基礎

---

## 参考文献

1. Xu, A. (2020). *System Design Interview: An Insider's Guide*. Chapter 4: Design a Rate Limiter. Byte Code LLC. https://www.systemdesigninterview.com/
2. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapter 8: The Trouble with Distributed Systems.
3. Stripe Engineering. (2017). "Scaling your API with rate limiters." *Stripe Blog*. https://stripe.com/blog/rate-limiters
4. IETF. (2012). *RFC 6585: Additional HTTP Status Codes*. Section 4: 429 Too Many Requests. https://www.rfc-editor.org/rfc/rfc6585
5. Google Cloud. (2024). "Rate limiting strategies and techniques." *Cloud Architecture Center*. https://cloud.google.com/architecture/rate-limiting-strategies-techniques
6. Redis. (2024). "Rate Limiting with Redis." *Redis Documentation*. https://redis.io/docs/latest/develop/use/patterns/rate-limiting/
7. Envoy Proxy. (2024). "Global rate limiting." *Envoy Documentation*. https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting
