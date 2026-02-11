# レート制限

> レート制限はAPIの安定性とフェアネスを守る防衛線。Token Bucket、Sliding Window、分散レート制限のアルゴリズム、実装パターン、クライアント側の対応まで、プロダクション品質のレート制限を設計する。

## この章で学ぶこと

- [ ] レート制限アルゴリズムの種類と特性を理解する
- [ ] Redis を使った分散レート制限の実装を把握する
- [ ] レート制限のレスポンス設計とクライアント対応を学ぶ

---

## 1. レート制限の目的

```
なぜレート制限が必要か:

  ① サービスの安定性:
     → 1クライアントの大量リクエストで他のクライアントに影響しない
     → サーバーリソースの保護

  ② フェアネス:
     → 全クライアントに公平なリソース配分
     → 無料/有料プランの差別化

  ③ セキュリティ:
     → ブルートフォース攻撃の防止
     → DDoS の軽減
     → スクレイピングの抑制

  ④ コスト管理:
     → 外部API呼び出しのコスト制御
     → インフラコストの予測可能性

レート制限の粒度:
  → ユーザー単位: user_id ごと
  → API Key 単位: api_key ごと
  → IP 単位: IPアドレスごと
  → エンドポイント単位: /users と /orders で別制限
  → プラン単位: Free: 100req/min, Pro: 1000req/min
```

---

## 2. アルゴリズム

```
① Fixed Window（固定ウィンドウ）:

  時間窓を固定（例: 毎分0秒〜59秒）
  窓内のリクエスト数をカウント

  |--- window 1 ---|--- window 2 ---|
  |  ■■■■■■■■■     |  ■■            |
  |  9 requests     |  2 requests    |
  limit = 10/分

  問題: ウィンドウ境界でバースト
  → 0:59に10リクエスト + 1:00に10リクエスト
  → 2秒間に20リクエスト（制限の2倍）

② Sliding Window Log（スライディングウィンドウログ）:

  各リクエストのタイムスタンプを記録
  現在時刻から1分間のリクエスト数をカウント

  正確だがメモリ消費が大きい
  → 各リクエストのタイムスタンプを保持

③ Sliding Window Counter（スライディングウィンドウカウンター）:

  前の窓と現在の窓のカウントを重み付け

  前の窓: 8リクエスト（60%経過）
  現在の窓: 3リクエスト（40%経過）
  推定: 8 × 0.6 + 3 = 7.8 → 制限内

  → 精度とメモリのバランスが良い

④ Token Bucket（トークンバケット）:

  バケットに一定速度でトークンを追加
  リクエスト1回 = トークン1個消費
  トークンがない = リクエスト拒否

  パラメータ:
  → capacity: バケットの最大トークン数（バースト許容量）
  → refill_rate: トークン補充速度

  例: capacity=10, refill_rate=2/秒
  → 最大10リクエストのバースト
  → 定常状態では2リクエスト/秒

  利点: バーストを許容しつつ長期的な制限を維持
  → AWS API Gateway, Nginx が採用

⑤ Leaky Bucket（漏れバケット）:

  リクエストをキューに入れ、一定速度で処理
  キューが満杯 = リクエスト拒否

  → 出力レートが一定（スムーズ）
  → バーストを平滑化
```

---

## 3. Redis 実装

```javascript
// Sliding Window Counter の Redis 実装

const Redis = require('ioredis');
const redis = new Redis();

async function slidingWindowRateLimit(key, limit, windowSizeMs) {
  const now = Date.now();
  const windowStart = now - windowSizeMs;

  // Lua スクリプト（アトミック操作）
  const script = `
    local key = KEYS[1]
    local now = tonumber(ARGV[1])
    local window_start = tonumber(ARGV[2])
    local limit = tonumber(ARGV[3])
    local window_ms = tonumber(ARGV[4])

    -- 期限切れのエントリを削除
    redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

    -- 現在のカウント
    local count = redis.call('ZCARD', key)

    if count < limit then
      -- リクエストを記録
      redis.call('ZADD', key, now, now .. ':' .. math.random(100000))
      redis.call('PEXPIRE', key, window_ms)
      return {1, limit - count - 1, 0}  -- allowed, remaining, retryAfter
    else
      -- 最も古いエントリの時刻
      local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
      local retry_after = oldest[2] + window_ms - now
      return {0, 0, retry_after}  -- denied, remaining, retryAfter
    end
  `;

  const [allowed, remaining, retryAfter] = await redis.eval(
    script, 1, key, now, windowStart, limit, windowSizeMs
  );

  return {
    allowed: allowed === 1,
    remaining,
    retryAfter: Math.ceil(retryAfter / 1000),
    limit,
    reset: Math.ceil((now + windowSizeMs) / 1000),
  };
}

// Token Bucket の Redis 実装
async function tokenBucketRateLimit(key, capacity, refillRate) {
  const script = `
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_rate = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])

    local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(bucket[1]) or capacity
    local last_refill = tonumber(bucket[2]) or now

    -- トークン補充
    local elapsed = (now - last_refill) / 1000
    tokens = math.min(capacity, tokens + elapsed * refill_rate)

    if tokens >= 1 then
      tokens = tokens - 1
      redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
      redis.call('PEXPIRE', key, 60000)
      return {1, math.floor(tokens)}
    else
      redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
      redis.call('PEXPIRE', key, 60000)
      local retry_after = (1 - tokens) / refill_rate
      return {0, 0, math.ceil(retry_after)}
    end
  `;

  const result = await redis.eval(script, 1, key, capacity, refillRate, Date.now());
  return { allowed: result[0] === 1, remaining: result[1], retryAfter: result[2] };
}
```

---

## 4. ミドルウェア実装

```javascript
// Express ミドルウェア
function rateLimitMiddleware(options) {
  const { limit = 100, windowMs = 60000, keyGenerator } = options;

  return async (req, res, next) => {
    const key = keyGenerator
      ? keyGenerator(req)
      : `rate_limit:${req.ip}`;

    const result = await slidingWindowRateLimit(key, limit, windowMs);

    // レスポンスヘッダーに制限情報を設定
    res.set({
      'X-RateLimit-Limit': result.limit,
      'X-RateLimit-Remaining': result.remaining,
      'X-RateLimit-Reset': result.reset,
    });

    if (!result.allowed) {
      res.set('Retry-After', result.retryAfter);
      return res.status(429).json({
        type: 'https://api.example.com/errors/rate-limit',
        title: 'Rate Limit Exceeded',
        status: 429,
        detail: `Rate limit of ${limit} requests per ${windowMs / 1000}s exceeded.`,
        retryAfter: result.retryAfter,
      });
    }

    next();
  };
}

// 使用例: エンドポイントごとに異なる制限
app.use('/api/v1/',
  rateLimitMiddleware({
    limit: 100,
    windowMs: 60000,
    keyGenerator: (req) => `rate:${req.apiKey?.id || req.ip}`,
  })
);

app.use('/api/v1/auth/login',
  rateLimitMiddleware({
    limit: 5,          // ログインは厳しく制限
    windowMs: 300000,  // 5分間
    keyGenerator: (req) => `rate:login:${req.ip}`,
  })
);
```

---

## 5. レスポンスヘッダー

```
標準的なレート制限ヘッダー:

  X-RateLimit-Limit: 100        ← ウィンドウ内の上限
  X-RateLimit-Remaining: 42     ← 残りリクエスト数
  X-RateLimit-Reset: 1640000000 ← リセット時刻（UNIX秒）
  Retry-After: 30               ← 再試行までの秒数（429時）

IETF標準（draft-ietf-httpapi-ratelimit-headers）:
  RateLimit-Limit: 100
  RateLimit-Remaining: 42
  RateLimit-Reset: 30           ← リセットまでの秒数

429 Too Many Requests レスポンス:
  HTTP/1.1 429 Too Many Requests
  Content-Type: application/json
  Retry-After: 30
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 0
  X-RateLimit-Reset: 1640000030

  {
    "type": "https://api.example.com/errors/rate-limit",
    "title": "Rate Limit Exceeded",
    "status": 429,
    "detail": "You have exceeded the rate limit of 100 requests per minute.",
    "retryAfter": 30
  }
```

---

## 6. クライアント側の対応

```javascript
// SDK内でのレート制限対応

async function requestWithRateLimit(fn, maxRetries = 3) {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (error.status === 429 && attempt < maxRetries) {
        const retryAfter = error.headers?.['retry-after']
          ? parseInt(error.headers['retry-after']) * 1000
          : Math.min(1000 * 2 ** attempt, 30000);

        console.warn(`Rate limited. Retrying in ${retryAfter}ms...`);
        await new Promise(r => setTimeout(r, retryAfter));
        continue;
      }
      throw error;
    }
  }
}

// プロアクティブなレート制限
// → レスポンスヘッダーを監視し、残り少ない場合にスロットリング
class RateLimitAwareClient {
  private remaining = Infinity;
  private resetAt = 0;

  async request(url, options) {
    // 残りが少ない場合はウェイト
    if (this.remaining <= 1 && Date.now() < this.resetAt) {
      const waitMs = this.resetAt - Date.now();
      await new Promise(r => setTimeout(r, waitMs));
    }

    const response = await fetch(url, options);

    // ヘッダーからレート制限情報を更新
    this.remaining = parseInt(response.headers.get('X-RateLimit-Remaining') || 'Infinity');
    this.resetAt = parseInt(response.headers.get('X-RateLimit-Reset') || '0') * 1000;

    return response;
  }
}
```

---

## まとめ

| アルゴリズム | 特徴 | 用途 |
|------------|------|------|
| Fixed Window | シンプル、境界バースト問題 | 簡易な制限 |
| Sliding Window | 正確、メモリ効率的 | 一般的なAPI |
| Token Bucket | バースト許容、柔軟 | AWS, Nginx |
| Leaky Bucket | 出力一定、平滑化 | キュー処理 |

---

## 次に読むべきガイド
→ [[02-input-validation.md]] — 入力バリデーション

---

## 参考文献
1. Stripe. "Rate Limiting." stripe.com/docs, 2024.
2. Cloudflare. "Rate Limiting Best Practices." blog.cloudflare.com, 2024.
3. draft-ietf-httpapi-ratelimit-headers. IETF, 2024.
