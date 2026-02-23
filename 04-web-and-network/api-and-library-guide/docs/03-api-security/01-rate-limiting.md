# レート制限

> レート制限はAPIの安定性とフェアネスを守る防衛線。Token Bucket、Sliding Window、分散レート制限のアルゴリズム、実装パターン、クライアント側の対応まで、プロダクション品質のレート制限を設計する。

## この章で学ぶこと

- [ ] レート制限アルゴリズムの種類と特性を理解する
- [ ] Redis を使った分散レート制限の実装を把握する
- [ ] レート制限のレスポンス設計とクライアント対応を学ぶ
- [ ] 多層レート制限の設計パターンを理解する
- [ ] 分散環境でのレート制限の課題と解決策を学ぶ
- [ ] レート制限のテスト手法とモニタリングを把握する

---

## 1. レート制限の目的と基本概念

### 1.1 なぜレート制限が必要か

```
レート制限が必要な4つの理由:

  ① サービスの安定性:
     → 1クライアントの大量リクエストで他のクライアントに影響しない
     → サーバーリソース（CPU、メモリ、DB接続）の保護
     → カスケード障害の防止
     → バックエンドサービスの過負荷防止

  ② フェアネス:
     → 全クライアントに公平なリソース配分
     → 無料/有料プランの差別化
     → SLA に基づくリソース保証
     → テナント間の影響の分離（ノイジーネイバー問題）

  ③ セキュリティ:
     → ブルートフォース攻撃の防止
     → DDoS の軽減
     → スクレイピングの抑制
     → 認証エンドポイントへの攻撃防御
     → API キーの不正利用検出

  ④ コスト管理:
     → 外部API呼び出しのコスト制御
     → インフラコストの予測可能性
     → データベース負荷の制御
     → ネットワーク帯域の保護
```

### 1.2 レート制限の粒度

```
レート制限を適用する単位（粒度）:

  ┌─────────────────────────────────────────────────────┐
  │               レート制限の粒度ピラミッド              │
  │                                                     │
  │                    ┌───────┐                        │
  │                    │ Global│  ← サービス全体          │
  │                   ┌┴───────┴┐                       │
  │                   │  Tenant │  ← テナント単位         │
  │                  ┌┴─────────┴┐                      │
  │                  │   User    │  ← ユーザー単位        │
  │                 ┌┴───────────┴┐                     │
  │                 │  API Key    │  ← APIキー単位       │
  │                ┌┴─────────────┴┐                    │
  │                │   IP Address  │  ← IPアドレス単位   │
  │               ┌┴───────────────┴┐                   │
  │               │   Endpoint      │  ← エンドポイント  │
  │              ┌┴─────────────────┴┐                  │
  │              │   Resource + Action│  ← リソース操作  │
  │              └───────────────────┘                  │
  └─────────────────────────────────────────────────────┘

  各粒度の使い分け:

  ① ユーザー単位 (user_id):
     → 認証済みリクエスト
     → ユーザーごとの公平性保証
     → 例: 1ユーザー 100req/分

  ② API Key 単位 (api_key):
     → サービス間通信
     → プランベースの制限
     → 例: Free=100req/分, Pro=1000req/分

  ③ IP 単位 (ip_address):
     → 未認証リクエスト
     → ブルートフォース防止
     → 例: 1IP 60req/分

  ④ エンドポイント単位:
     → /users と /orders で別制限
     → 重い処理のエンドポイントを厳しく制限
     → 例: /search=10req/分, /users=100req/分

  ⑤ プラン単位:
     → SaaS のティア制御
     → 例: Free=100req/分, Pro=1000req/分, Enterprise=10000req/分

  ⑥ 複合キー:
     → user_id + endpoint の組み合わせ
     → tenant_id + resource_type
     → 例: ユーザーAは /upload に対して 10req/時間
```

### 1.3 レート制限の設計原則

```
実務で重要な設計原則:

  ① 透明性:
     → レスポンスヘッダーで残りリクエスト数を通知
     → ドキュメントに制限値を明記
     → 制限超過時に明確なエラーメッセージ

  ② 段階的制限:
     → いきなり完全ブロックではなく段階的に制限
     → Warning → Throttle → Block の3段階
     → 異常検知による動的制限

  ③ グレースフル・デグラデーション:
     → レート制限システム自体がダウンした場合のフォールバック
     → Allow-by-default vs Deny-by-default
     → ローカルキャッシュによるフォールバック

  ④ 柔軟性:
     → プランアップグレードで即時に制限緩和
     → 一時的な制限緩和（バースト許可）
     → ホワイトリスト対応

  ⑤ モニタリング:
     → レート制限の発動頻度の監視
     → 誤検知の検出と対応
     → 容量計画へのフィードバック
```

---

## 2. レート制限アルゴリズム

### 2.1 Fixed Window（固定ウィンドウ）

```
① Fixed Window（固定ウィンドウ）:

  時間窓を固定（例: 毎分0秒〜59秒）
  窓内のリクエスト数をカウント

  |--- window 1 ---|--- window 2 ---|--- window 3 ---|
  |  ■■■■■■■■■     |  ■■            |  ■■■■■         |
  |  9 requests     |  2 requests    |  5 requests    |
  limit = 10/分

  メリット:
  → 実装がシンプル
  → メモリ効率が良い（各ウィンドウ1カウンタ）
  → 計算コストが低い

  デメリット:
  → ウィンドウ境界でバースト
  → 0:59に10リクエスト + 1:00に10リクエスト
  → 2秒間に20リクエスト（制限の2倍）

  ┌────────────┬────────────┐
  │  Window 1  │  Window 2  │
  │        ■■■■│■■■■        │
  │    10 req  │  10 req    │
  └────────────┴────────────┘
       ↑ 境界付近で20req/2秒 ↑
```

```javascript
// Fixed Window の実装
class FixedWindowRateLimiter {
  constructor(limit, windowMs) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.windows = new Map(); // key -> { count, windowStart }
  }

  isAllowed(key) {
    const now = Date.now();
    const windowStart = Math.floor(now / this.windowMs) * this.windowMs;

    const entry = this.windows.get(key);

    if (!entry || entry.windowStart !== windowStart) {
      // 新しいウィンドウ
      this.windows.set(key, { count: 1, windowStart });
      return {
        allowed: true,
        remaining: this.limit - 1,
        resetAt: windowStart + this.windowMs,
      };
    }

    if (entry.count < this.limit) {
      entry.count++;
      return {
        allowed: true,
        remaining: this.limit - entry.count,
        resetAt: windowStart + this.windowMs,
      };
    }

    return {
      allowed: false,
      remaining: 0,
      resetAt: windowStart + this.windowMs,
      retryAfter: Math.ceil((windowStart + this.windowMs - now) / 1000),
    };
  }

  // 古いウィンドウのクリーンアップ
  cleanup() {
    const now = Date.now();
    for (const [key, entry] of this.windows.entries()) {
      if (now - entry.windowStart > this.windowMs * 2) {
        this.windows.delete(key);
      }
    }
  }
}

// 使用例
const limiter = new FixedWindowRateLimiter(100, 60000); // 100req/分

function handleRequest(req) {
  const key = `rate:${req.ip}`;
  const result = limiter.isAllowed(key);

  if (!result.allowed) {
    return { status: 429, retryAfter: result.retryAfter };
  }

  return { status: 200, remaining: result.remaining };
}
```

### 2.2 Sliding Window Log（スライディングウィンドウログ）

```
② Sliding Window Log（スライディングウィンドウログ）:

  各リクエストのタイムスタンプを記録
  現在時刻から遡ったウィンドウ内のリクエスト数をカウント

  時刻の流れ →
  ─────────────────────────────────────────────
  t1  t2    t3  t4 t5   t6  t7    t8  t9  t10
  ■   ■     ■   ■  ■    ■   ■     ■   ■   ■
  |←────── 60秒のウィンドウ ───────→|
                                   ↑ 現在時刻

  古いタイムスタンプ（t1, t2）はウィンドウ外なので除外

  メリット:
  → 正確なレート制限（境界バースト問題なし）
  → どの時点でもウィンドウ内のリクエスト数が正確

  デメリット:
  → メモリ消費が大きい（各リクエストのタイムスタンプを保持）
  → ウィンドウ内のリクエスト数に比例するメモリ使用
  → クリーンアップ処理が必要
```

```javascript
// Sliding Window Log の実装
class SlidingWindowLogLimiter {
  constructor(limit, windowMs) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.logs = new Map(); // key -> timestamp[]
  }

  isAllowed(key) {
    const now = Date.now();
    const windowStart = now - this.windowMs;

    // 既存のログを取得、なければ初期化
    let timestamps = this.logs.get(key) || [];

    // ウィンドウ外のタイムスタンプを削除
    timestamps = timestamps.filter(ts => ts > windowStart);

    if (timestamps.length < this.limit) {
      timestamps.push(now);
      this.logs.set(key, timestamps);

      return {
        allowed: true,
        remaining: this.limit - timestamps.length,
        resetAt: Math.ceil((timestamps[0] + this.windowMs) / 1000),
      };
    }

    // 最も古いタイムスタンプから次にスロットが空く時間を計算
    const oldestInWindow = timestamps[0];
    const retryAfter = Math.ceil((oldestInWindow + this.windowMs - now) / 1000);

    this.logs.set(key, timestamps);

    return {
      allowed: false,
      remaining: 0,
      retryAfter,
      resetAt: Math.ceil((oldestInWindow + this.windowMs) / 1000),
    };
  }
}

// Redis を使った Sliding Window Log
async function slidingWindowLogRedis(redis, key, limit, windowMs) {
  const now = Date.now();
  const windowStart = now - windowMs;

  const pipeline = redis.pipeline();

  // 古いエントリを削除
  pipeline.zremrangebyscore(key, '-inf', windowStart);
  // 現在のカウントを取得
  pipeline.zcard(key);

  const results = await pipeline.exec();
  const count = results[1][1];

  if (count < limit) {
    // 新しいエントリを追加（ユニークなメンバー名が必要）
    const member = `${now}:${Math.random().toString(36).substr(2, 9)}`;
    await redis.zadd(key, now, member);
    await redis.pexpire(key, windowMs);

    return { allowed: true, remaining: limit - count - 1 };
  }

  // 最も古いエントリの時間を取得
  const oldest = await redis.zrange(key, 0, 0, 'WITHSCORES');
  const retryAfter = oldest.length >= 2
    ? Math.ceil((parseFloat(oldest[1]) + windowMs - now) / 1000)
    : 1;

  return { allowed: false, remaining: 0, retryAfter };
}
```

### 2.3 Sliding Window Counter（スライディングウィンドウカウンター）

```
③ Sliding Window Counter（スライディングウィンドウカウンター）:

  Fixed Window のメモリ効率 + Sliding Window の精度を両立
  前の窓と現在の窓のカウントを重み付けで推定

  |--- 前のウィンドウ ---|--- 現在のウィンドウ ---|
  |  8 requests         |  3 requests           |
  |                     |←── 40%経過 ──→|       |

  前のウィンドウ: 8リクエスト
  現在のウィンドウ: 3リクエスト
  現在のウィンドウ経過率: 40%
  前のウィンドウの残存率: 100% - 40% = 60%

  推定リクエスト数 = 前のウィンドウ × 残存率 + 現在のウィンドウ
                   = 8 × 0.6 + 3
                   = 4.8 + 3
                   = 7.8 → 制限(10)内 → 許可

  メリット:
  → 精度とメモリのバランスが良い
  → 各ウィンドウ2カウンタのみ（メモリ効率）
  → 境界バースト問題を大幅に軽減

  デメリット:
  → 推定値であり完全に正確ではない
  → 実際のリクエスト分布が均一でない場合に誤差
```

```javascript
// Sliding Window Counter の実装
class SlidingWindowCounter {
  constructor(limit, windowMs) {
    this.limit = limit;
    this.windowMs = windowMs;
    this.counters = new Map();
  }

  isAllowed(key) {
    const now = Date.now();
    const currentWindow = Math.floor(now / this.windowMs) * this.windowMs;
    const previousWindow = currentWindow - this.windowMs;

    let data = this.counters.get(key);
    if (!data) {
      data = { windows: {} };
      this.counters.set(key, data);
    }

    const currentCount = data.windows[currentWindow] || 0;
    const previousCount = data.windows[previousWindow] || 0;

    // 現在のウィンドウ内の経過率
    const elapsed = (now - currentWindow) / this.windowMs;
    // 前のウィンドウの重み
    const previousWeight = 1 - elapsed;

    // 推定リクエスト数
    const estimatedCount = previousCount * previousWeight + currentCount;

    if (estimatedCount < this.limit) {
      data.windows[currentWindow] = currentCount + 1;

      // 古いウィンドウを削除
      for (const w of Object.keys(data.windows)) {
        if (parseInt(w) < previousWindow) {
          delete data.windows[w];
        }
      }

      return {
        allowed: true,
        remaining: Math.floor(this.limit - estimatedCount - 1),
        resetAt: Math.ceil((currentWindow + this.windowMs) / 1000),
      };
    }

    return {
      allowed: false,
      remaining: 0,
      retryAfter: Math.ceil((this.windowMs - (now - currentWindow)) / 1000),
    };
  }
}
```

### 2.4 Token Bucket（トークンバケット）

```
④ Token Bucket（トークンバケット）:

  バケットに一定速度でトークンを追加
  リクエスト1回 = トークン1個消費
  トークンがない = リクエスト拒否

  パラメータ:
  → capacity: バケットの最大トークン数（バースト許容量）
  → refill_rate: トークン補充速度（トークン/秒）

  ┌──────────────────────────────────┐
  │         Token Bucket             │
  │                                  │
  │  capacity = 10                   │
  │  refill_rate = 2/秒             │
  │                                  │
  │  ┌────────────────────┐          │
  │  │ ● ● ● ● ● ○ ○ ○ ○ ○│ ←トークン │
  │  │ 5/10 tokens         │          │
  │  └────────────────────┘          │
  │       ↑ 2トークン/秒で補充       │
  │       ↓ リクエストでトークン消費  │
  │                                  │
  │  最大10リクエストのバースト       │
  │  定常状態では2リクエスト/秒       │
  └──────────────────────────────────┘

  時間経過のシミュレーション:
  t=0   : tokens=10 (満タン)
  t=0   : 5リクエスト → tokens=5
  t=1   : +2トークン → tokens=7
  t=1   : 3リクエスト → tokens=4
  t=2   : +2トークン → tokens=6
  t=5   : +6トークン → tokens=10 (上限で頭打ち)

  メリット:
  → バーストを許容しつつ長期的な制限を維持
  → パラメータが直感的
  → AWS API Gateway, Nginx, GitHub API が採用

  デメリット:
  → 2つのパラメータの調整が必要
  → 短期間のバーストがバックエンドに負荷を与える可能性
```

```javascript
// Token Bucket の実装（メモリ内）
class TokenBucket {
  constructor(capacity, refillRate) {
    this.capacity = capacity;       // 最大トークン数
    this.refillRate = refillRate;   // トークン/秒
    this.buckets = new Map();
  }

  isAllowed(key, tokensRequired = 1) {
    const now = Date.now();

    let bucket = this.buckets.get(key);
    if (!bucket) {
      bucket = { tokens: this.capacity, lastRefill: now };
      this.buckets.set(key, bucket);
    }

    // トークンを補充
    const elapsed = (now - bucket.lastRefill) / 1000;
    bucket.tokens = Math.min(
      this.capacity,
      bucket.tokens + elapsed * this.refillRate
    );
    bucket.lastRefill = now;

    if (bucket.tokens >= tokensRequired) {
      bucket.tokens -= tokensRequired;
      return {
        allowed: true,
        remaining: Math.floor(bucket.tokens),
        retryAfter: 0,
      };
    }

    // トークンが足りない場合、次にトークンが利用可能になる時間
    const deficit = tokensRequired - bucket.tokens;
    const retryAfter = Math.ceil(deficit / this.refillRate);

    return {
      allowed: false,
      remaining: 0,
      retryAfter,
    };
  }
}

// リクエストサイズに応じたトークン消費
class WeightedTokenBucket extends TokenBucket {
  constructor(capacity, refillRate, weightFn) {
    super(capacity, refillRate);
    this.weightFn = weightFn;
  }

  isAllowedForRequest(key, request) {
    const weight = this.weightFn(request);
    return this.isAllowed(key, weight);
  }
}

// 使用例: エンドポイントの重さに応じた消費
const weightedLimiter = new WeightedTokenBucket(
  100, // capacity
  10,  // refill_rate: 10 tokens/sec
  (req) => {
    // GETは1トークン、POSTは5トークン、ファイルアップロードは20トークン
    const weights = {
      'GET': 1,
      'POST': 5,
      'PUT': 5,
      'DELETE': 3,
    };
    if (req.path.includes('/upload')) return 20;
    return weights[req.method] || 1;
  }
);
```

### 2.5 Leaky Bucket（漏れバケット）

```
⑤ Leaky Bucket（漏れバケット）:

  リクエストをキューに入れ、一定速度で処理
  キューが満杯 = リクエスト拒否

  ┌──────────────────────────────────┐
  │         Leaky Bucket             │
  │                                  │
  │  ┌─────────┐                     │
  │  │ req req  │ ← リクエスト流入    │
  │  │ req req  │   （不規則）        │
  │  │ req req  │                     │
  │  │ req req  │  queue_size = 10    │
  │  └────┬─────┘                     │
  │       │                           │
  │       ▼  一定速度で流出            │
  │    ● ● ● ● ●                     │
  │    leak_rate = 2/秒               │
  │                                  │
  │  → 出力レートが一定（スムーズ）    │
  │  → バーストを平滑化               │
  └──────────────────────────────────┘

  Token Bucket との違い:
  ┌─────────────────┬─────────────────┐
  │   Token Bucket  │  Leaky Bucket   │
  ├─────────────────┼─────────────────┤
  │ バースト許容    │ バースト平滑化   │
  │ 入力側で制御    │ 出力側で制御     │
  │ トークン消費    │ キューで管理     │
  │ 即座にレスポンス │ キュー待ち発生  │
  └─────────────────┴─────────────────┘
```

```javascript
// Leaky Bucket の実装
class LeakyBucket {
  constructor(capacity, leakRate) {
    this.capacity = capacity;     // キューの最大サイズ
    this.leakRate = leakRate;     // 処理速度（リクエスト/秒）
    this.buckets = new Map();
  }

  isAllowed(key) {
    const now = Date.now();

    let bucket = this.buckets.get(key);
    if (!bucket) {
      bucket = { water: 0, lastLeak: now };
      this.buckets.set(key, bucket);
    }

    // 経過時間に応じてキューを排出
    const elapsed = (now - bucket.lastLeak) / 1000;
    bucket.water = Math.max(0, bucket.water - elapsed * this.leakRate);
    bucket.lastLeak = now;

    if (bucket.water < this.capacity) {
      bucket.water += 1;
      return {
        allowed: true,
        queuePosition: Math.ceil(bucket.water),
        estimatedWait: Math.ceil(bucket.water / this.leakRate),
      };
    }

    return {
      allowed: false,
      retryAfter: Math.ceil(1 / this.leakRate),
    };
  }
}

// Leaky Bucket をキューとして使う場合
class LeakyBucketQueue {
  constructor(capacity, processRate) {
    this.capacity = capacity;
    this.processRate = processRate; // 1秒あたりの処理数
    this.queue = [];
    this.processing = false;
  }

  async enqueue(task) {
    if (this.queue.length >= this.capacity) {
      throw new Error('Queue is full. Try again later.');
    }

    return new Promise((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this.startProcessing();
    });
  }

  startProcessing() {
    if (this.processing) return;
    this.processing = true;

    const interval = 1000 / this.processRate;
    const timer = setInterval(async () => {
      if (this.queue.length === 0) {
        clearInterval(timer);
        this.processing = false;
        return;
      }

      const { task, resolve, reject } = this.queue.shift();
      try {
        const result = await task();
        resolve(result);
      } catch (error) {
        reject(error);
      }
    }, interval);
  }
}
```

### 2.6 アルゴリズム比較

```
各アルゴリズムの総合比較:

┌──────────────────┬────────┬────────┬────────┬──────────┬──────────┐
│                  │ 精度   │ メモリ │ 実装   │ バースト │ 適用場面  │
├──────────────────┼────────┼────────┼────────┼──────────┼──────────┤
│ Fixed Window     │ 低     │ 最小   │ 最易   │ 境界問題 │ 簡易制限  │
│ Sliding Log      │ 最高   │ 大     │ 中     │ なし     │ 厳密制限  │
│ Sliding Counter  │ 高     │ 小     │ 中     │ ほぼなし │ 一般API  │
│ Token Bucket     │ 高     │ 小     │ 中     │ 許容     │ API GW   │
│ Leaky Bucket     │ 高     │ 中     │ やや難 │ 平滑化   │ キュー   │
└──────────────────┴────────┴────────┴────────┴──────────┴──────────┘

主要サービスの採用アルゴリズム:
  → AWS API Gateway: Token Bucket
  → Nginx: Leaky Bucket (limit_req)
  → GitHub API: Sliding Window
  → Stripe: Token Bucket + Sliding Window
  → Cloudflare: Sliding Window Counter
  → Google Cloud: Token Bucket
```

---

## 3. Redis を使った分散レート制限

### 3.1 Sliding Window Counter の Redis 実装

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
```

### 3.2 Token Bucket の Redis 実装

```javascript
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

### 3.3 分散レート制限の課題と解決

```
分散環境でのレート制限の課題:

  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Server 1 │  │ Server 2 │  │ Server 3 │
  │          │  │          │  │          │
  │ count=3  │  │ count=4  │  │ count=2  │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │             │             │
       └─────────────┼─────────────┘
                     │
              ┌──────┴──────┐
              │    Redis    │
              │ count = 9   │  ← 一元管理
              └─────────────┘

  課題1: Redis の単一障害点（SPOF）
  → 解決: Redis Cluster / Redis Sentinel
  → フォールバック: ローカルメモリでの制限

  課題2: ネットワーク遅延
  → 解決: ローカルバッファ + 定期的な同期
  → トレードオフ: 精度 vs レイテンシ

  課題3: Redis の一貫性
  → 解決: Lua スクリプトによるアトミック操作
  → WATCH/MULTI/EXEC は使わない（Cluster非対応）
```

```javascript
// Redis Cluster 対応のレート制限
class DistributedRateLimiter {
  constructor(redisCluster, options) {
    this.redis = redisCluster;
    this.options = options;
    this.localCache = new Map(); // ローカルフォールバック
    this.localLimiter = new SlidingWindowCounter(
      options.limit * 1.2, // ローカルは少し緩めに
      options.windowMs
    );
  }

  async isAllowed(key) {
    try {
      return await this.checkRedis(key);
    } catch (error) {
      console.warn(`Redis rate limit failed, using local fallback: ${error.message}`);
      return this.checkLocal(key);
    }
  }

  async checkRedis(key) {
    const { limit, windowMs } = this.options;
    return await slidingWindowRateLimit(
      `rate_limit:${key}`,
      limit,
      windowMs
    );
  }

  checkLocal(key) {
    return this.localLimiter.isAllowed(key);
  }
}

// Redis Sentinel を使った高可用性レート制限
const redis = new Redis({
  sentinels: [
    { host: 'sentinel-1', port: 26379 },
    { host: 'sentinel-2', port: 26379 },
    { host: 'sentinel-3', port: 26379 },
  ],
  name: 'rate-limit-master',
  retryStrategy(times) {
    return Math.min(times * 50, 2000);
  },
});
```

### 3.4 ローカルバッファ付き分散レート制限

```javascript
// ローカルバッファで Redis アクセスを最小化
class BufferedRateLimiter {
  constructor(redis, options) {
    this.redis = redis;
    this.limit = options.limit;
    this.windowMs = options.windowMs;
    this.batchSize = options.batchSize || 10;   // 10リクエスト分をローカルバッファ
    this.syncInterval = options.syncInterval || 1000; // 1秒ごとに同期

    this.localCounters = new Map();
    this.startSync();
  }

  async isAllowed(key) {
    let local = this.localCounters.get(key);
    if (!local) {
      local = { count: 0, quota: this.batchSize, synced: 0 };
      this.localCounters.set(key, local);
      // 初回は Redis からクオータを取得
      await this.fetchQuota(key, local);
    }

    if (local.count < local.quota) {
      local.count++;
      return { allowed: true, remaining: local.quota - local.count };
    }

    // ローカルクオータ使い切り → Redis に同期して追加取得
    await this.syncToRedis(key, local);
    await this.fetchQuota(key, local);

    if (local.count < local.quota) {
      local.count++;
      return { allowed: true, remaining: local.quota - local.count };
    }

    return { allowed: false, remaining: 0 };
  }

  async fetchQuota(key, local) {
    const script = `
      local key = KEYS[1]
      local limit = tonumber(ARGV[1])
      local batch = tonumber(ARGV[2])
      local window_ms = tonumber(ARGV[3])
      local now = tonumber(ARGV[4])

      local current = tonumber(redis.call('GET', key) or '0')
      local available = limit - current

      if available <= 0 then
        return 0
      end

      local grant = math.min(batch, available)
      redis.call('INCRBY', key, grant)

      if redis.call('PTTL', key) == -1 then
        redis.call('PEXPIRE', key, window_ms)
      end

      return grant
    `;

    const granted = await this.redis.eval(
      script, 1, `rate:${key}`,
      this.limit, this.batchSize, this.windowMs, Date.now()
    );

    local.quota = granted;
    local.count = 0;
  }

  async syncToRedis(key, local) {
    // 使い切れなかったクオータを戻す
    const unused = local.quota - local.count;
    if (unused > 0) {
      await this.redis.decrby(`rate:${key}`, unused);
    }
    local.count = 0;
    local.quota = 0;
  }

  startSync() {
    setInterval(async () => {
      for (const [key, local] of this.localCounters.entries()) {
        await this.syncToRedis(key, local);
        await this.fetchQuota(key, local);
      }
    }, this.syncInterval);
  }
}
```

---

## 4. ミドルウェア実装

### 4.1 Express ミドルウェア

```javascript
// Express ミドルウェア - プロダクション品質
function rateLimitMiddleware(options) {
  const {
    limit = 100,
    windowMs = 60000,
    keyGenerator,
    handler,
    skip,
    onLimitReached,
    headers = true,
    draft7Headers = false,
  } = options;

  return async (req, res, next) => {
    // スキップ条件
    if (skip && await skip(req)) {
      return next();
    }

    const key = keyGenerator
      ? keyGenerator(req)
      : `rate_limit:${req.ip}`;

    try {
      const result = await slidingWindowRateLimit(key, limit, windowMs);

      // レスポンスヘッダーに制限情報を設定
      if (headers) {
        res.set({
          'X-RateLimit-Limit': result.limit,
          'X-RateLimit-Remaining': Math.max(0, result.remaining),
          'X-RateLimit-Reset': result.reset,
        });
      }

      // IETF draft-7 ヘッダー
      if (draft7Headers) {
        res.set({
          'RateLimit-Limit': result.limit,
          'RateLimit-Remaining': Math.max(0, result.remaining),
          'RateLimit-Reset': Math.ceil((result.reset * 1000 - Date.now()) / 1000),
        });
      }

      if (!result.allowed) {
        res.set('Retry-After', result.retryAfter);

        // カスタムハンドラー
        if (onLimitReached) {
          onLimitReached(req, res, result);
        }

        if (handler) {
          return handler(req, res, next, result);
        }

        return res.status(429).json({
          type: 'https://api.example.com/errors/rate-limit',
          title: 'Rate Limit Exceeded',
          status: 429,
          detail: `Rate limit of ${limit} requests per ${windowMs / 1000}s exceeded.`,
          retryAfter: result.retryAfter,
        });
      }

      next();
    } catch (error) {
      // レート制限システムのエラー時はリクエストを許可
      console.error('Rate limit error:', error);
      next();
    }
  };
}

// 使用例: エンドポイントごとに異なる制限
const app = require('express')();

// グローバル制限
app.use('/api/v1/',
  rateLimitMiddleware({
    limit: 100,
    windowMs: 60000,
    keyGenerator: (req) => `rate:${req.apiKey?.id || req.ip}`,
  })
);

// ログインエンドポイント: 厳しく制限
app.use('/api/v1/auth/login',
  rateLimitMiddleware({
    limit: 5,
    windowMs: 300000,  // 5分間
    keyGenerator: (req) => `rate:login:${req.ip}`,
    onLimitReached: (req, res, result) => {
      // セキュリティチームに通知
      securityAlert('login_rate_limit', {
        ip: req.ip,
        userAgent: req.headers['user-agent'],
      });
    },
  })
);

// パスワードリセット: さらに厳しく
app.use('/api/v1/auth/reset-password',
  rateLimitMiddleware({
    limit: 3,
    windowMs: 3600000,  // 1時間
    keyGenerator: (req) => `rate:reset:${req.body?.email || req.ip}`,
  })
);

// ファイルアップロード: リソース消費が大きいので制限
app.use('/api/v1/upload',
  rateLimitMiddleware({
    limit: 10,
    windowMs: 3600000,  // 1時間に10ファイル
    keyGenerator: (req) => `rate:upload:${req.userId}`,
  })
);

// 検索エンドポイント: 中程度の制限
app.use('/api/v1/search',
  rateLimitMiddleware({
    limit: 30,
    windowMs: 60000,
    keyGenerator: (req) => `rate:search:${req.userId || req.ip}`,
    skip: (req) => req.user?.plan === 'enterprise', // Enterpriseはスキップ
  })
);
```

### 4.2 プランベースのレート制限

```javascript
// プランに応じた動的レート制限
class PlanBasedRateLimiter {
  constructor(redis) {
    this.redis = redis;
    this.plans = {
      free: {
        global: { limit: 100, windowMs: 60000 },
        search: { limit: 10, windowMs: 60000 },
        upload: { limit: 5, windowMs: 3600000 },
        ai: { limit: 20, windowMs: 3600000 },
      },
      pro: {
        global: { limit: 1000, windowMs: 60000 },
        search: { limit: 100, windowMs: 60000 },
        upload: { limit: 50, windowMs: 3600000 },
        ai: { limit: 200, windowMs: 3600000 },
      },
      enterprise: {
        global: { limit: 10000, windowMs: 60000 },
        search: { limit: 1000, windowMs: 60000 },
        upload: { limit: 500, windowMs: 3600000 },
        ai: { limit: 2000, windowMs: 3600000 },
      },
    };
  }

  middleware(endpoint = 'global') {
    return async (req, res, next) => {
      const plan = req.user?.plan || 'free';
      const limits = this.plans[plan]?.[endpoint] || this.plans.free.global;

      const key = `rate:${plan}:${endpoint}:${req.user?.id || req.ip}`;
      const result = await slidingWindowRateLimit(
        key, limits.limit, limits.windowMs
      );

      res.set({
        'X-RateLimit-Limit': limits.limit,
        'X-RateLimit-Remaining': Math.max(0, result.remaining),
        'X-RateLimit-Reset': result.reset,
        'X-RateLimit-Plan': plan,
      });

      if (!result.allowed) {
        res.set('Retry-After', result.retryAfter);
        return res.status(429).json({
          type: 'rate_limit_exceeded',
          message: `${plan} plan limit of ${limits.limit} requests exceeded.`,
          upgrade: plan !== 'enterprise'
            ? 'Upgrade your plan for higher limits: https://example.com/pricing'
            : undefined,
          retryAfter: result.retryAfter,
        });
      }

      next();
    };
  }

  // プランの動的更新（アップグレード時に即座に反映）
  async updatePlan(userId, newPlan) {
    // キャッシュされたレート制限情報をリセット
    const keys = await this.redis.keys(`rate:*:*:${userId}`);
    if (keys.length > 0) {
      await this.redis.del(...keys);
    }
  }
}

// 使用例
const planLimiter = new PlanBasedRateLimiter(redis);
app.use('/api/v1/', planLimiter.middleware('global'));
app.use('/api/v1/search', planLimiter.middleware('search'));
app.use('/api/v1/upload', planLimiter.middleware('upload'));
app.use('/api/v1/ai', planLimiter.middleware('ai'));
```

### 4.3 NestJS でのレート制限

```typescript
// NestJS デコレータベースのレート制限
import { SetMetadata, UseGuards, Injectable, CanActivate } from '@nestjs/common';
import { Reflector } from '@nestjs/core';

// カスタムデコレータ
export const RATE_LIMIT_KEY = 'rateLimit';

export interface RateLimitOptions {
  limit: number;
  windowMs: number;
  keyPrefix?: string;
}

export const RateLimit = (options: RateLimitOptions) =>
  SetMetadata(RATE_LIMIT_KEY, options);

// レート制限ガード
@Injectable()
export class RateLimitGuard implements CanActivate {
  constructor(
    private reflector: Reflector,
    private rateLimiter: RateLimiterService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const options = this.reflector.get<RateLimitOptions>(
      RATE_LIMIT_KEY,
      context.getHandler(),
    );

    if (!options) return true;

    const request = context.switchToHttp().getRequest();
    const response = context.switchToHttp().getResponse();

    const key = `${options.keyPrefix || 'rate'}:${request.user?.id || request.ip}`;
    const result = await this.rateLimiter.check(key, options.limit, options.windowMs);

    response.set({
      'X-RateLimit-Limit': options.limit.toString(),
      'X-RateLimit-Remaining': Math.max(0, result.remaining).toString(),
      'X-RateLimit-Reset': result.reset.toString(),
    });

    if (!result.allowed) {
      response.set('Retry-After', result.retryAfter.toString());
      throw new HttpException(
        {
          statusCode: 429,
          message: 'Rate limit exceeded',
          retryAfter: result.retryAfter,
        },
        HttpStatus.TOO_MANY_REQUESTS,
      );
    }

    return true;
  }
}

// コントローラでの使用
@Controller('users')
@UseGuards(RateLimitGuard)
export class UsersController {
  @Get()
  @RateLimit({ limit: 100, windowMs: 60000 })
  findAll() {
    return this.usersService.findAll();
  }

  @Post()
  @RateLimit({ limit: 20, windowMs: 60000, keyPrefix: 'rate:create' })
  create(@Body() dto: CreateUserDto) {
    return this.usersService.create(dto);
  }

  @Post('bulk-import')
  @RateLimit({ limit: 5, windowMs: 3600000, keyPrefix: 'rate:bulk' })
  bulkImport(@Body() dto: BulkImportDto) {
    return this.usersService.bulkImport(dto);
  }
}
```

### 4.4 Go でのレート制限ミドルウェア

```go
package ratelimit

import (
    "context"
    "fmt"
    "net/http"
    "strconv"
    "time"

    "github.com/go-redis/redis/v8"
)

// RateLimiter はレート制限の設定を保持する
type RateLimiter struct {
    redis    *redis.Client
    limit    int
    windowMs int64
}

// Result はレート制限チェックの結果
type Result struct {
    Allowed    bool
    Remaining  int
    RetryAfter int
    Reset      int64
}

// NewRateLimiter は新しいレート制限インスタンスを生成する
func NewRateLimiter(rdb *redis.Client, limit int, windowMs int64) *RateLimiter {
    return &RateLimiter{
        redis:    rdb,
        limit:    limit,
        windowMs: windowMs,
    }
}

// Check はレート制限をチェックする
func (rl *RateLimiter) Check(ctx context.Context, key string) (*Result, error) {
    now := time.Now().UnixMilli()
    windowStart := now - rl.windowMs

    script := redis.NewScript(`
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window_ms = tonumber(ARGV[4])

        redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
        local count = redis.call('ZCARD', key)

        if count < limit then
            redis.call('ZADD', key, now, now .. ':' .. math.random(100000))
            redis.call('PEXPIRE', key, window_ms)
            return {1, limit - count - 1, 0}
        else
            local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
            local retry_after = 0
            if oldest[2] then
                retry_after = oldest[2] + window_ms - now
            end
            return {0, 0, retry_after}
        end
    `)

    result, err := script.Run(ctx, rl.redis, []string{key},
        now, windowStart, rl.limit, rl.windowMs).Int64Slice()
    if err != nil {
        return nil, fmt.Errorf("rate limit check failed: %w", err)
    }

    return &Result{
        Allowed:    result[0] == 1,
        Remaining:  int(result[1]),
        RetryAfter: int(result[2] / 1000),
        Reset:      (now + rl.windowMs) / 1000,
    }, nil
}

// Middleware はHTTPミドルウェアとしてレート制限を適用する
func (rl *RateLimiter) Middleware(keyFn func(*http.Request) string) func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            key := keyFn(r)
            result, err := rl.Check(r.Context(), key)
            if err != nil {
                // レート制限システムエラー時はリクエストを許可
                next.ServeHTTP(w, r)
                return
            }

            w.Header().Set("X-RateLimit-Limit", strconv.Itoa(rl.limit))
            w.Header().Set("X-RateLimit-Remaining", strconv.Itoa(result.Remaining))
            w.Header().Set("X-RateLimit-Reset", strconv.FormatInt(result.Reset, 10))

            if !result.Allowed {
                w.Header().Set("Retry-After", strconv.Itoa(result.RetryAfter))
                w.Header().Set("Content-Type", "application/json")
                w.WriteHeader(http.StatusTooManyRequests)
                fmt.Fprintf(w, `{"error":"rate_limit_exceeded","retry_after":%d}`,
                    result.RetryAfter)
                return
            }

            next.ServeHTTP(w, r)
        })
    }
}

// 使用例
func setupRoutes() {
    rdb := redis.NewClient(&redis.Options{Addr: "localhost:6379"})
    limiter := NewRateLimiter(rdb, 100, 60000) // 100 req/min

    mux := http.NewServeMux()
    mux.Handle("/api/",
        limiter.Middleware(func(r *http.Request) string {
            return "rate:" + r.RemoteAddr
        })(apiHandler),
    )
}
```

---

## 5. レスポンスヘッダーとエラー設計

### 5.1 標準レスポンスヘッダー

```
標準的なレート制限ヘッダー（de facto 標準）:

  X-RateLimit-Limit: 100        ← ウィンドウ内の上限
  X-RateLimit-Remaining: 42     ← 残りリクエスト数
  X-RateLimit-Reset: 1640000000 ← リセット時刻（UNIX秒）
  Retry-After: 30               ← 再試行までの秒数（429時）

IETF標準（RFC 9110 / draft-ietf-httpapi-ratelimit-headers）:
  RateLimit-Limit: 100
  RateLimit-Remaining: 42
  RateLimit-Reset: 30           ← リセットまでの秒数（UNIX秒ではない）

  注意: IETF標準の RateLimit-Reset は「リセットまでの残り秒数」
  de facto 標準の X-RateLimit-Reset は「リセット時刻（UNIX epoch秒）」

複数ポリシーの表現:
  RateLimit-Limit: 100, 100;w=60, 1000;w=3600
  → 60秒間に100リクエスト AND 3600秒間に1000リクエスト
```

### 5.2 429 Too Many Requests レスポンス

```
429 Too Many Requests レスポンス例:

  HTTP/1.1 429 Too Many Requests
  Content-Type: application/problem+json
  Retry-After: 30
  X-RateLimit-Limit: 100
  X-RateLimit-Remaining: 0
  X-RateLimit-Reset: 1640000030

  {
    "type": "https://api.example.com/errors/rate-limit",
    "title": "Rate Limit Exceeded",
    "status": 429,
    "detail": "You have exceeded the rate limit of 100 requests per minute.",
    "instance": "/api/v1/users",
    "retryAfter": 30,
    "limit": {
      "requests": 100,
      "window": "60s",
      "remaining": 0,
      "reset": "2024-12-21T00:00:30Z"
    },
    "upgrade": {
      "message": "Upgrade to Pro plan for 1000 requests/minute",
      "url": "https://example.com/pricing"
    }
  }

プラン別の制限情報を含むレスポンス例:

  {
    "type": "rate_limit_exceeded",
    "status": 429,
    "message": "Free plan rate limit exceeded",
    "limits": {
      "current_plan": "free",
      "limits": {
        "global": "100/min",
        "search": "10/min",
        "upload": "5/hour"
      },
      "usage": {
        "global": { "used": 100, "limit": 100, "reset_in": 30 },
        "search": { "used": 4, "limit": 10, "reset_in": 45 }
      }
    },
    "upgrade_options": [
      { "plan": "pro", "global": "1000/min", "price": "$29/mo" },
      { "plan": "enterprise", "global": "10000/min", "price": "contact us" }
    ]
  }
```

### 5.3 レスポンスヘッダーの実装ヘルパー

```javascript
// レスポンスヘッダーの設定ヘルパー
class RateLimitHeaders {
  static set(res, result, options = {}) {
    const {
      prefix = 'X-RateLimit',
      includeIetf = false,
      includePlan = false,
      plan = null,
    } = options;

    // de facto 標準ヘッダー
    res.set({
      [`${prefix}-Limit`]: result.limit.toString(),
      [`${prefix}-Remaining`]: Math.max(0, result.remaining).toString(),
      [`${prefix}-Reset`]: result.reset.toString(),
    });

    // IETF draft ヘッダー
    if (includeIetf) {
      const resetInSeconds = Math.max(0, result.reset - Math.floor(Date.now() / 1000));
      res.set({
        'RateLimit-Limit': result.limit.toString(),
        'RateLimit-Remaining': Math.max(0, result.remaining).toString(),
        'RateLimit-Reset': resetInSeconds.toString(),
      });
    }

    // プラン情報
    if (includePlan && plan) {
      res.set(`${prefix}-Plan`, plan);
    }

    // 429 の場合
    if (!result.allowed && result.retryAfter > 0) {
      res.set('Retry-After', result.retryAfter.toString());
    }
  }

  static error429(result, options = {}) {
    const { detail, upgrade } = options;

    return {
      type: 'https://api.example.com/errors/rate-limit',
      title: 'Rate Limit Exceeded',
      status: 429,
      detail: detail || `Rate limit of ${result.limit} requests exceeded.`,
      retryAfter: result.retryAfter,
      ...(upgrade ? { upgrade } : {}),
    };
  }
}
```

---

## 6. クライアント側の対応

### 6.1 リトライ戦略

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

        // ジッターを追加（全クライアントが同時にリトライしないように）
        const jitter = Math.random() * 1000;
        const delay = retryAfter + jitter;

        console.warn(
          `Rate limited (attempt ${attempt + 1}/${maxRetries}). ` +
          `Retrying in ${Math.round(delay)}ms...`
        );
        await new Promise(r => setTimeout(r, delay));
        continue;
      }
      throw error;
    }
  }
}

// Exponential Backoff with Jitter
class RetryWithBackoff {
  constructor(options = {}) {
    this.maxRetries = options.maxRetries || 5;
    this.baseDelay = options.baseDelay || 1000;
    this.maxDelay = options.maxDelay || 60000;
    this.jitterFactor = options.jitterFactor || 0.5;
  }

  async execute(fn) {
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        return await fn();
      } catch (error) {
        if (!this.isRetryable(error) || attempt === this.maxRetries) {
          throw error;
        }

        const delay = this.calculateDelay(attempt, error);
        console.warn(`Retry attempt ${attempt + 1}/${this.maxRetries} in ${delay}ms`);
        await this.sleep(delay);
      }
    }
  }

  isRetryable(error) {
    return error.status === 429 || error.status >= 500;
  }

  calculateDelay(attempt, error) {
    // Retry-After ヘッダーがあれば優先
    if (error.headers?.['retry-after']) {
      const retryAfter = parseInt(error.headers['retry-after']);
      if (!isNaN(retryAfter)) {
        return retryAfter * 1000;
      }
    }

    // Exponential backoff with full jitter
    const exponential = this.baseDelay * Math.pow(2, attempt);
    const capped = Math.min(exponential, this.maxDelay);
    const jitter = capped * this.jitterFactor * Math.random();
    return capped + jitter;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// 使用例
const retrier = new RetryWithBackoff({ maxRetries: 5, baseDelay: 1000 });
const result = await retrier.execute(() => fetch('/api/data'));
```

### 6.2 プロアクティブなレート制限対応

```javascript
// プロアクティブなレート制限
// → レスポンスヘッダーを監視し、残り少ない場合にスロットリング
class RateLimitAwareClient {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || '';
    this.remaining = Infinity;
    this.resetAt = 0;
    this.limit = 0;

    // プロアクティブスロットリング設定
    this.throttleThreshold = options.throttleThreshold || 0.1; // 残り10%でスロットリング
    this.requestQueue = [];
    this.processing = false;
  }

  async request(url, options = {}) {
    // 残りが少ない場合はプロアクティブにウェイト
    if (this.shouldThrottle()) {
      await this.waitForReset();
    }

    const response = await fetch(this.baseUrl + url, options);

    // ヘッダーからレート制限情報を更新
    this.updateLimits(response);

    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get('Retry-After') || '1');
      await new Promise(r => setTimeout(r, retryAfter * 1000));
      return this.request(url, options); // リトライ
    }

    return response;
  }

  shouldThrottle() {
    if (this.limit === 0) return false;
    return this.remaining / this.limit <= this.throttleThreshold;
  }

  async waitForReset() {
    if (this.remaining <= 0 && Date.now() < this.resetAt) {
      const waitMs = this.resetAt - Date.now() + 100; // 100ms のバッファ
      console.warn(`Proactive throttle: waiting ${waitMs}ms for rate limit reset`);
      await new Promise(r => setTimeout(r, waitMs));
    }
  }

  updateLimits(response) {
    const limit = response.headers.get('X-RateLimit-Limit');
    const remaining = response.headers.get('X-RateLimit-Remaining');
    const reset = response.headers.get('X-RateLimit-Reset');

    if (limit) this.limit = parseInt(limit);
    if (remaining) this.remaining = parseInt(remaining);
    if (reset) this.resetAt = parseInt(reset) * 1000;
  }
}

// バッチリクエストでレート制限を最適化
class BatchRequestClient extends RateLimitAwareClient {
  constructor(options = {}) {
    super(options);
    this.batchSize = options.batchSize || 10;
    this.batchDelay = options.batchDelay || 100; // ms between batches
  }

  async batchRequest(urls, options = {}) {
    const results = [];

    for (let i = 0; i < urls.length; i += this.batchSize) {
      const batch = urls.slice(i, i + this.batchSize);

      const batchResults = await Promise.allSettled(
        batch.map(url => this.request(url, options))
      );

      results.push(...batchResults);

      // バッチ間にディレイ
      if (i + this.batchSize < urls.length) {
        // 残りリクエスト数に基づいて動的にディレイを調整
        const delay = this.calculateBatchDelay();
        await new Promise(r => setTimeout(r, delay));
      }
    }

    return results;
  }

  calculateBatchDelay() {
    if (this.remaining <= this.batchSize * 2) {
      // 残りが少ない場合は長いディレイ
      return this.batchDelay * 5;
    }
    if (this.remaining <= this.batchSize * 5) {
      return this.batchDelay * 2;
    }
    return this.batchDelay;
  }
}
```

### 6.3 Python クライアントでの対応

```python
import time
import random
import requests
from functools import wraps
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class RateLimitInfo:
    """レート制限情報"""
    limit: int = 0
    remaining: int = float('inf')
    reset_at: float = 0
    retry_after: Optional[int] = None


class RateLimitedClient:
    """レート制限対応のHTTPクライアント"""

    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url
        self.max_retries = max_retries
        self.session = requests.Session()
        self.rate_info = RateLimitInfo()

    def get(self, path: str, **kwargs) -> requests.Response:
        return self._request('GET', path, **kwargs)

    def post(self, path: str, **kwargs) -> requests.Response:
        return self._request('POST', path, **kwargs)

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"

        for attempt in range(self.max_retries + 1):
            # プロアクティブなウェイト
            self._proactive_wait()

            try:
                response = self.session.request(method, url, **kwargs)
                self._update_rate_info(response)

                if response.status_code == 429:
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt, response)
                        print(f"Rate limited. Retrying in {delay:.1f}s "
                              f"(attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(delay)
                        continue
                    raise RateLimitError(response)

                return response

            except requests.ConnectionError:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue
                raise

        raise RateLimitError("Max retries exceeded")

    def _proactive_wait(self):
        """残りリクエスト数が少ない場合にプロアクティブにウェイト"""
        if self.rate_info.remaining <= 1:
            wait_time = self.rate_info.reset_at - time.time()
            if wait_time > 0:
                print(f"Proactive wait: {wait_time:.1f}s")
                time.sleep(wait_time + 0.1)

    def _update_rate_info(self, response: requests.Response):
        """レスポンスヘッダーからレート制限情報を更新"""
        headers = response.headers
        self.rate_info = RateLimitInfo(
            limit=int(headers.get('X-RateLimit-Limit', 0)),
            remaining=int(headers.get('X-RateLimit-Remaining', float('inf'))),
            reset_at=float(headers.get('X-RateLimit-Reset', 0)),
            retry_after=int(headers['Retry-After']) if 'Retry-After' in headers else None,
        )

    def _calculate_delay(self, attempt: int, response: requests.Response) -> float:
        """リトライまでの遅延を計算"""
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            return float(retry_after) + random.uniform(0, 1)

        # Exponential backoff with jitter
        base = min(2 ** attempt, 60)
        jitter = random.uniform(0, base * 0.5)
        return base + jitter


class RateLimitError(Exception):
    """レート制限エラー"""
    def __init__(self, response_or_message):
        if isinstance(response_or_message, requests.Response):
            self.response = response_or_message
            self.retry_after = int(
                response_or_message.headers.get('Retry-After', 0)
            )
            super().__init__(
                f"Rate limit exceeded. Retry after {self.retry_after}s"
            )
        else:
            super().__init__(str(response_or_message))


# デコレータとしての使用
def rate_limited(max_retries=3, base_delay=1.0):
    """レート制限対応デコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    if attempt < max_retries:
                        delay = e.retry_after or (base_delay * 2 ** attempt)
                        jitter = random.uniform(0, delay * 0.25)
                        time.sleep(delay + jitter)
                        continue
                    raise
        return wrapper
    return decorator


# 使用例
client = RateLimitedClient("https://api.example.com", max_retries=3)

@rate_limited(max_retries=5)
def fetch_users():
    return client.get("/v1/users")

users = fetch_users()
```

---

## 7. 多層レート制限

### 7.1 多層レート制限の設計

```
多層レート制限アーキテクチャ:

  リクエスト
  ↓
  ┌──────────────────────────────────────┐
  │  Layer 1: CDN / Edge Rate Limiting   │ ← Cloudflare, AWS WAF
  │  → IP ベース: 1000 req/min           │
  │  → DDoS 防御                         │
  └────────────────┬─────────────────────┘
                   ↓
  ┌──────────────────────────────────────┐
  │  Layer 2: API Gateway Rate Limiting  │ ← Kong, AWS API GW
  │  → API Key ベース: 500 req/min       │
  │  → プランベースの制限                 │
  └────────────────┬─────────────────────┘
                   ↓
  ┌──────────────────────────────────────┐
  │  Layer 3: Application Rate Limiting  │ ← アプリ内ミドルウェア
  │  → ユーザー単位: 100 req/min         │
  │  → エンドポイント単位: 10 req/min     │
  │  → リソース操作単位: 5 req/hour       │
  └────────────────┬─────────────────────┘
                   ↓
  ┌──────────────────────────────────────┐
  │  Layer 4: Service Rate Limiting      │ ← サービス間通信
  │  → Circuit Breaker                   │
  │  → Bulkhead パターン                 │
  └──────────────────────────────────────┘

  各レイヤーの役割:
  Layer 1: 大規模攻撃の防御（雑なフィルタリング）
  Layer 2: APIキー/プランベースの制限（ビジネスロジック）
  Layer 3: きめ細かいリソース保護（アプリケーションレベル）
  Layer 4: 内部サービスの保護（サービスメッシュ）
```

### 7.2 複合レート制限の実装

```javascript
// 複数のレート制限を組み合わせる
class CompositeRateLimiter {
  constructor(limiters) {
    this.limiters = limiters; // { name, limiter, key, limit, windowMs }[]
  }

  async isAllowed(request) {
    const results = [];
    let mostRestrictive = null;

    for (const config of this.limiters) {
      const key = typeof config.key === 'function'
        ? config.key(request)
        : config.key;

      const result = await config.limiter.isAllowed(key);
      results.push({ name: config.name, ...result });

      if (!result.allowed) {
        if (!mostRestrictive || result.retryAfter > mostRestrictive.retryAfter) {
          mostRestrictive = { name: config.name, ...result };
        }
      }
    }

    return {
      allowed: results.every(r => r.allowed),
      results,
      mostRestrictive,
    };
  }
}

// 使用例: 3層のレート制限
const compositeLimiter = new CompositeRateLimiter([
  {
    name: 'global',
    limiter: new SlidingWindowCounter(1000, 60000),
    key: (req) => `global:${req.ip}`,
  },
  {
    name: 'user',
    limiter: new SlidingWindowCounter(100, 60000),
    key: (req) => `user:${req.userId}`,
  },
  {
    name: 'endpoint',
    limiter: new SlidingWindowCounter(20, 60000),
    key: (req) => `endpoint:${req.userId}:${req.path}`,
  },
]);

// ミドルウェアとして使用
app.use(async (req, res, next) => {
  const result = await compositeLimiter.isAllowed(req);

  // 最も厳しい制限の情報をヘッダーに設定
  const globalResult = result.results.find(r => r.name === 'global');
  res.set({
    'X-RateLimit-Limit': globalResult.limit || 1000,
    'X-RateLimit-Remaining': Math.max(0, globalResult.remaining || 0),
  });

  if (!result.allowed) {
    const { mostRestrictive } = result;
    res.set('Retry-After', mostRestrictive.retryAfter);
    return res.status(429).json({
      error: 'rate_limit_exceeded',
      limitType: mostRestrictive.name,
      retryAfter: mostRestrictive.retryAfter,
    });
  }

  next();
});
```

### 7.3 動的レート制限

```javascript
// サーバー負荷に応じた動的レート制限
class AdaptiveRateLimiter {
  constructor(redis, options) {
    this.redis = redis;
    this.baseLimit = options.limit;
    this.windowMs = options.windowMs;
    this.currentMultiplier = 1.0;

    // 定期的にサーバー負荷をチェック
    this.startHealthCheck(options.healthCheckInterval || 10000);
  }

  async isAllowed(key) {
    const effectiveLimit = Math.floor(this.baseLimit * this.currentMultiplier);
    return await slidingWindowRateLimit(key, effectiveLimit, this.windowMs);
  }

  startHealthCheck(interval) {
    setInterval(async () => {
      const health = await this.checkServerHealth();
      this.adjustMultiplier(health);
    }, interval);
  }

  async checkServerHealth() {
    // サーバーメトリクスを収集
    const os = require('os');
    const cpuUsage = os.loadavg()[0] / os.cpus().length;
    const memUsage = 1 - os.freemem() / os.totalmem();

    // 外部メトリクス（例: DB接続プール使用率）
    const dbPoolUsage = await this.getDbPoolUsage();

    return { cpuUsage, memUsage, dbPoolUsage };
  }

  adjustMultiplier(health) {
    const { cpuUsage, memUsage, dbPoolUsage } = health;
    const maxUsage = Math.max(cpuUsage, memUsage, dbPoolUsage);

    if (maxUsage > 0.9) {
      // 危険水準: 制限を大幅に強化
      this.currentMultiplier = 0.3;
      console.warn('CRITICAL: Rate limit reduced to 30%');
    } else if (maxUsage > 0.8) {
      // 高負荷: 制限を強化
      this.currentMultiplier = 0.5;
      console.warn('HIGH LOAD: Rate limit reduced to 50%');
    } else if (maxUsage > 0.6) {
      // やや高い: 少し制限
      this.currentMultiplier = 0.8;
    } else {
      // 正常: フル制限
      this.currentMultiplier = 1.0;
    }
  }

  async getDbPoolUsage() {
    // DB接続プールの使用率を取得（例）
    try {
      const result = await this.redis.get('metrics:db_pool_usage');
      return parseFloat(result) || 0;
    } catch {
      return 0;
    }
  }
}
```

---

## 8. Nginx / API Gateway でのレート制限

### 8.1 Nginx でのレート制限設定

```nginx
# Nginx レート制限設定

# レート制限ゾーンの定義
http {
    # IP ベースのレート制限
    # binary_remote_addr: クライアントIPのバイナリ表現
    # zone=api_limit:10m: 共有メモリゾーン（10MB ≈ 160,000 IPアドレス）
    # rate=10r/s: 1秒あたり10リクエスト
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;

    # API Key ベースのレート制限
    map $http_x_api_key $api_key {
        default $http_x_api_key;
        "" $binary_remote_addr;
    }
    limit_req_zone $api_key zone=api_key_limit:10m rate=100r/s;

    # エンドポイント × IP の複合制限
    limit_req_zone $binary_remote_addr zone=login_limit:5m rate=1r/s;
    limit_req_zone $binary_remote_addr zone=search_limit:5m rate=5r/s;

    # レート制限超過時のステータスコード
    limit_req_status 429;

    # レート制限ログレベル
    limit_req_log_level warn;

    server {
        listen 80;

        # グローバルレート制限
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            # burst=20: 最大20リクエストのバーストを許可
            # nodelay: バーストリクエストを即座に処理（キューイングしない）

            proxy_pass http://backend;
        }

        # ログインエンドポイント: 厳しい制限
        location /api/auth/login {
            limit_req zone=login_limit burst=5;
            # burst=5: 5リクエストまでキューイング
            # nodelay なし: キューに入れて順番に処理

            proxy_pass http://backend;
        }

        # 検索エンドポイント
        location /api/search {
            limit_req zone=search_limit burst=10 nodelay;
            proxy_pass http://backend;
        }

        # 429 レスポンスのカスタマイズ
        error_page 429 = @rate_limited;
        location @rate_limited {
            default_type application/json;
            return 429 '{"error":"rate_limit_exceeded","message":"Too many requests"}';
        }
    }
}
```

### 8.2 AWS API Gateway のレート制限

```
AWS API Gateway のレート制限:

  Usage Plan で制御:
  ┌────────────────────────────────────────┐
  │           Usage Plan: Free             │
  │                                        │
  │  Throttle:                             │
  │    Rate Limit: 10 req/sec              │
  │    Burst Limit: 20 requests            │
  │                                        │
  │  Quota:                                │
  │    Limit: 10,000 requests              │
  │    Period: MONTH                        │
  │    Offset: 0 (月初リセット)             │
  │                                        │
  │  API Keys: [key-001, key-002, ...]     │
  └────────────────────────────────────────┘

  ┌────────────────────────────────────────┐
  │           Usage Plan: Pro              │
  │                                        │
  │  Throttle:                             │
  │    Rate Limit: 100 req/sec             │
  │    Burst Limit: 200 requests           │
  │                                        │
  │  Quota:                                │
  │    Limit: 1,000,000 requests           │
  │    Period: MONTH                        │
  │                                        │
  │  API Keys: [key-101, key-102, ...]     │
  └────────────────────────────────────────┘
```

```yaml
# AWS SAM テンプレートでのレート制限設定
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  ApiGateway:
    Type: AWS::Serverless::Api
    Properties:
      StageName: prod
      # メソッドレベルのスロットリング
      MethodSettings:
        - HttpMethod: '*'
          ResourcePath: '/*'
          ThrottlingBurstLimit: 100
          ThrottlingRateLimit: 50
        - HttpMethod: POST
          ResourcePath: '/auth/login'
          ThrottlingBurstLimit: 10
          ThrottlingRateLimit: 5

  # Usage Plan
  FreePlan:
    Type: AWS::ApiGateway::UsagePlan
    Properties:
      UsagePlanName: free
      Throttle:
        BurstLimit: 20
        RateLimit: 10
      Quota:
        Limit: 10000
        Period: MONTH
      ApiStages:
        - ApiId: !Ref ApiGateway
          Stage: prod

  ProPlan:
    Type: AWS::ApiGateway::UsagePlan
    Properties:
      UsagePlanName: pro
      Throttle:
        BurstLimit: 200
        RateLimit: 100
      Quota:
        Limit: 1000000
        Period: MONTH
      ApiStages:
        - ApiId: !Ref ApiGateway
          Stage: prod
```

### 8.3 Kong でのレート制限プラグイン

```yaml
# Kong レート制限プラグイン設定
plugins:
  - name: rate-limiting
    config:
      # ポリシー: local, cluster, redis
      policy: redis
      redis_host: redis-host
      redis_port: 6379
      redis_database: 0
      redis_timeout: 2000

      # 制限値（複数の時間窓を同時に設定可能）
      second: 10        # 10 req/sec
      minute: 100       # 100 req/min
      hour: 5000        # 5000 req/hour
      day: 100000       # 100000 req/day

      # ヘッダー設定
      hide_client_headers: false  # X-RateLimit-* ヘッダーを返す

      # 制限超過時のレスポンス
      error_code: 429
      error_message: "Rate limit exceeded"

  # エンドポイント固有の制限
  - name: rate-limiting
    route: login-route
    config:
      policy: redis
      redis_host: redis-host
      minute: 5
      hour: 50
      error_message: "Too many login attempts"

  # コンシューマーごとの制限
  - name: rate-limiting
    consumer: free-tier
    config:
      policy: redis
      redis_host: redis-host
      minute: 60
      hour: 1000

  - name: rate-limiting
    consumer: pro-tier
    config:
      policy: redis
      redis_host: redis-host
      minute: 600
      hour: 10000
```

---

## 9. テストとモニタリング

### 9.1 レート制限のテスト

```javascript
// Jest でのレート制限テスト
const { describe, it, expect, beforeEach } = require('@jest/globals');

describe('SlidingWindowCounter', () => {
  let limiter;

  beforeEach(() => {
    limiter = new SlidingWindowCounter(10, 60000); // 10 req/min
  });

  it('should allow requests within the limit', () => {
    const key = 'test-user';
    for (let i = 0; i < 10; i++) {
      const result = limiter.isAllowed(key);
      expect(result.allowed).toBe(true);
    }
  });

  it('should deny requests exceeding the limit', () => {
    const key = 'test-user';
    // 制限まで消費
    for (let i = 0; i < 10; i++) {
      limiter.isAllowed(key);
    }
    // 11番目は拒否
    const result = limiter.isAllowed(key);
    expect(result.allowed).toBe(false);
    expect(result.remaining).toBe(0);
    expect(result.retryAfter).toBeGreaterThan(0);
  });

  it('should track remaining count correctly', () => {
    const key = 'test-user';
    for (let i = 0; i < 5; i++) {
      const result = limiter.isAllowed(key);
      expect(result.remaining).toBe(10 - i - 1);
    }
  });

  it('should isolate different keys', () => {
    for (let i = 0; i < 10; i++) {
      limiter.isAllowed('user-a');
    }
    // user-b は影響を受けない
    const result = limiter.isAllowed('user-b');
    expect(result.allowed).toBe(true);
    expect(result.remaining).toBe(9);
  });
});

// 統合テスト: Redis を使ったレート制限
describe('Redis Rate Limiting Integration', () => {
  let redis;

  beforeAll(async () => {
    redis = new Redis({ host: 'localhost', port: 6379, db: 15 });
    await redis.flushdb();
  });

  afterAll(async () => {
    await redis.flushdb();
    await redis.quit();
  });

  it('should rate limit across multiple calls', async () => {
    const key = 'test:integration:user1';
    const limit = 5;
    const windowMs = 10000;

    // 5リクエストは許可
    for (let i = 0; i < limit; i++) {
      const result = await slidingWindowRateLimit(key, limit, windowMs);
      expect(result.allowed).toBe(true);
    }

    // 6番目は拒否
    const result = await slidingWindowRateLimit(key, limit, windowMs);
    expect(result.allowed).toBe(false);
    expect(result.retryAfter).toBeGreaterThan(0);
  });

  it('should reset after window expires', async () => {
    const key = 'test:integration:user2';
    const limit = 3;
    const windowMs = 2000; // 2秒ウィンドウ

    // 制限まで消費
    for (let i = 0; i < limit; i++) {
      await slidingWindowRateLimit(key, limit, windowMs);
    }

    // ウィンドウが過ぎるまで待機
    await new Promise(r => setTimeout(r, 2100));

    // リセット後は許可
    const result = await slidingWindowRateLimit(key, limit, windowMs);
    expect(result.allowed).toBe(true);
  });
});

// 負荷テスト: レート制限の動作確認
describe('Rate Limit Load Test', () => {
  it('should handle concurrent requests correctly', async () => {
    const key = 'test:concurrent';
    const limit = 50;
    const windowMs = 60000;
    const concurrency = 100;

    const results = await Promise.all(
      Array(concurrency).fill(null).map(() =>
        slidingWindowRateLimit(key, limit, windowMs)
      )
    );

    const allowed = results.filter(r => r.allowed).length;
    const denied = results.filter(r => !r.allowed).length;

    expect(allowed).toBe(limit);
    expect(denied).toBe(concurrency - limit);
  });
});
```

### 9.2 負荷テストスクリプト

```bash
#!/bin/bash
# レート制限の負荷テスト

API_URL="http://localhost:3000/api/v1/users"
API_KEY="test-api-key"
REQUESTS=200
CONCURRENT=20

echo "=== Rate Limit Load Test ==="
echo "URL: $API_URL"
echo "Total Requests: $REQUESTS"
echo "Concurrency: $CONCURRENT"
echo ""

# Apache Bench を使用
ab -n $REQUESTS -c $CONCURRENT \
   -H "X-API-Key: $API_KEY" \
   -H "Accept: application/json" \
   "$API_URL" 2>/dev/null | grep -E "(Requests per|Time per|Failed|Non-2xx)"

echo ""
echo "=== Checking Rate Limit Headers ==="
for i in $(seq 1 5); do
  echo "--- Request $i ---"
  curl -s -o /dev/null -w "Status: %{http_code}\n" \
       -H "X-API-Key: $API_KEY" \
       -D - "$API_URL" 2>/dev/null | grep -iE "(X-RateLimit|Retry-After|HTTP)"
  echo ""
done
```

```python
# Python 負荷テスト（asyncio ベース）
import asyncio
import aiohttp
import time
from collections import Counter


async def rate_limit_load_test(
    url: str,
    total_requests: int = 200,
    concurrency: int = 20,
    api_key: str = None,
):
    """レート制限の負荷テスト"""
    results = Counter()
    headers = {}
    if api_key:
        headers['X-API-Key'] = api_key

    semaphore = asyncio.Semaphore(concurrency)
    rate_limit_info = {'last_remaining': None, 'last_reset': None}

    async def make_request(session, i):
        async with semaphore:
            try:
                async with session.get(url, headers=headers) as resp:
                    results[resp.status] += 1

                    # レート制限ヘッダーを記録
                    remaining = resp.headers.get('X-RateLimit-Remaining')
                    if remaining:
                        rate_limit_info['last_remaining'] = remaining
                    reset_at = resp.headers.get('X-RateLimit-Reset')
                    if reset_at:
                        rate_limit_info['last_reset'] = reset_at

                    if resp.status == 429:
                        retry_after = resp.headers.get('Retry-After', 'N/A')
                        results['retry_after'] = retry_after

            except Exception as e:
                results['error'] += 1

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [make_request(session, i) for i in range(total_requests)]
        await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    print(f"\n{'='*50}")
    print(f"Rate Limit Load Test Results")
    print(f"{'='*50}")
    print(f"Total Requests: {total_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Duration: {elapsed:.2f}s")
    print(f"Throughput: {total_requests/elapsed:.1f} req/s")
    print(f"\nStatus Code Distribution:")
    for status, count in sorted(results.items()):
        if isinstance(status, int):
            print(f"  {status}: {count} ({count/total_requests*100:.1f}%)")
    print(f"\nLast Rate Limit Info:")
    print(f"  Remaining: {rate_limit_info['last_remaining']}")
    print(f"  Reset At: {rate_limit_info['last_reset']}")

    if 'retry_after' in results:
        print(f"  Retry-After: {results['retry_after']}s")


# 実行
asyncio.run(rate_limit_load_test(
    url="http://localhost:3000/api/v1/users",
    total_requests=200,
    concurrency=20,
    api_key="test-key",
))
```

### 9.3 モニタリングとアラート

```javascript
// レート制限のモニタリング
class RateLimitMonitor {
  constructor(redis, metricsClient) {
    this.redis = redis;
    this.metrics = metricsClient; // Prometheus, DataDog, etc.
  }

  async recordResult(key, result, metadata = {}) {
    const { endpoint, userId, plan, ip } = metadata;

    // メトリクスを記録
    if (result.allowed) {
      this.metrics.increment('rate_limit.allowed', {
        endpoint,
        plan,
      });
    } else {
      this.metrics.increment('rate_limit.denied', {
        endpoint,
        plan,
      });

      // 拒否回数をRedisに記録（アラート用）
      const denyKey = `rate_limit:denied:${key}`;
      const denyCount = await this.redis.incr(denyKey);
      await this.redis.expire(denyKey, 300); // 5分間

      // 短時間に大量の拒否 → アラート
      if (denyCount >= 50) {
        this.alert({
          level: 'warning',
          message: `High rate limit denial rate for ${key}`,
          details: {
            denyCount,
            endpoint,
            userId,
            ip,
            plan,
          },
        });
      }
    }

    // 残りリクエスト数のゲージ
    this.metrics.gauge('rate_limit.remaining', result.remaining, {
      endpoint,
      plan,
    });
  }

  alert(alertData) {
    console.warn('RATE LIMIT ALERT:', JSON.stringify(alertData));

    // Slack, PagerDuty, etc. に通知
    // this.notifier.send(alertData);
  }
}

// Prometheus メトリクス
const promClient = require('prom-client');

const rateLimitAllowed = new promClient.Counter({
  name: 'api_rate_limit_allowed_total',
  help: 'Number of requests allowed by rate limiter',
  labelNames: ['endpoint', 'plan'],
});

const rateLimitDenied = new promClient.Counter({
  name: 'api_rate_limit_denied_total',
  help: 'Number of requests denied by rate limiter',
  labelNames: ['endpoint', 'plan'],
});

const rateLimitRemaining = new promClient.Gauge({
  name: 'api_rate_limit_remaining',
  help: 'Remaining requests in current rate limit window',
  labelNames: ['endpoint', 'plan', 'user_id'],
});

// Grafana ダッシュボード用クエリ（PromQL）
/*
  # レート制限拒否率
  rate(api_rate_limit_denied_total[5m])
  / (rate(api_rate_limit_allowed_total[5m]) + rate(api_rate_limit_denied_total[5m]))

  # エンドポイント別拒否率
  rate(api_rate_limit_denied_total{endpoint=~".*"}[5m])

  # プラン別の制限到達率
  api_rate_limit_remaining{plan="free"} == 0
*/
```

---

## 10. 実務パターンとベストプラクティス

### 10.1 ホワイトリスト/ブラックリスト

```javascript
// ホワイトリスト/ブラックリスト対応
class RateLimiterWithACL {
  constructor(redis, options) {
    this.redis = redis;
    this.limiter = new SlidingWindowCounter(options.limit, options.windowMs);
    this.whitelist = new Set(options.whitelist || []);
    this.blacklist = new Set(options.blacklist || []);
  }

  async isAllowed(key, metadata = {}) {
    // ブラックリスト: 即座に拒否
    if (this.blacklist.has(key) || this.blacklist.has(metadata.ip)) {
      return { allowed: false, reason: 'blacklisted', retryAfter: -1 };
    }

    // ホワイトリスト: レート制限をスキップ
    if (this.whitelist.has(key) || this.whitelist.has(metadata.ip)) {
      return { allowed: true, reason: 'whitelisted', remaining: Infinity };
    }

    // 通常のレート制限チェック
    return this.limiter.isAllowed(key);
  }

  // 動的にホワイトリスト/ブラックリストを更新
  async refreshACL() {
    const whitelistKeys = await this.redis.smembers('rate_limit:whitelist');
    const blacklistKeys = await this.redis.smembers('rate_limit:blacklist');

    this.whitelist = new Set(whitelistKeys);
    this.blacklist = new Set(blacklistKeys);
  }

  async addToBlacklist(key, ttlSeconds = 3600) {
    await this.redis.sadd('rate_limit:blacklist', key);
    if (ttlSeconds > 0) {
      // 一定時間後に自動解除
      setTimeout(() => {
        this.redis.srem('rate_limit:blacklist', key);
        this.blacklist.delete(key);
      }, ttlSeconds * 1000);
    }
    this.blacklist.add(key);
  }
}
```

### 10.2 グレースフル・デグラデーション

```javascript
// Redis ダウン時のフォールバック戦略
class ResilientRateLimiter {
  constructor(options) {
    this.redis = options.redis;
    this.localLimiter = new SlidingWindowCounter(
      options.limit * 1.5, // ローカルは少し緩く
      options.windowMs
    );
    this.redisLimiter = null;
    this.redisAvailable = true;
    this.healthCheckInterval = options.healthCheckInterval || 5000;

    this.startHealthCheck();
  }

  async isAllowed(key) {
    if (this.redisAvailable) {
      try {
        const result = await Promise.race([
          slidingWindowRateLimit(key, this.limit, this.windowMs),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Redis timeout')), 100)
          ),
        ]);
        return result;
      } catch (error) {
        console.warn('Redis rate limit failed, falling back to local:', error.message);
        this.redisAvailable = false;
        return this.localLimiter.isAllowed(key);
      }
    }

    return this.localLimiter.isAllowed(key);
  }

  startHealthCheck() {
    setInterval(async () => {
      try {
        await this.redis.ping();
        if (!this.redisAvailable) {
          console.info('Redis rate limit recovered');
          this.redisAvailable = true;
        }
      } catch {
        this.redisAvailable = false;
      }
    }, this.healthCheckInterval);
  }
}
```

### 10.3 レート制限のバイパス防止

```
レート制限バイパスの攻撃手法と対策:

  ① IP ローテーション:
     攻撃: 多数のIPアドレスからリクエスト
     対策:
     → IP単位だけでなく、ユーザー/API Key単位でも制限
     → 異常な振る舞いパターンの検出
     → CAPTCHA の導入

  ② ヘッダー偽装:
     攻撃: X-Forwarded-For ヘッダーの偽装
     対策:
     → 信頼できるプロキシからの X-Forwarded-For のみ採用
     → 接続元IPを優先

  ③ アカウント作成ボット:
     攻撃: 大量のアカウントを作成してレート制限を分散
     対策:
     → アカウント作成にCAPTCHA
     → 新規アカウントに厳しい制限
     → デバイスフィンガープリンティング

  ④ Slow Rate Attack:
     攻撃: ギリギリの速度でリクエストし続ける
     対策:
     → 複数の時間窓で制限（秒/分/時/日）
     → 異常パターンの検出

  ⑤ API Key の共有:
     攻撃: 複数のクライアントで同じAPI Keyを使い回し
     対策:
     → API Key ごとの同時接続数制限
     → 使用パターンの異常検知
```

```javascript
// IP偽装対策: 信頼できるIPの取得
function getClientIp(req) {
  // 信頼できるプロキシのIPリスト
  const trustedProxies = new Set([
    '10.0.0.0/8',
    '172.16.0.0/12',
    '192.168.0.0/16',
  ]);

  // X-Forwarded-For が信頼できるプロキシからの場合のみ使用
  if (req.headers['x-forwarded-for'] && isTrustedProxy(req.socket.remoteAddress, trustedProxies)) {
    const forwardedFor = req.headers['x-forwarded-for'].split(',');
    // 最も左のIPが元のクライアントIP
    return forwardedFor[0].trim();
  }

  // それ以外は接続元IPを使用
  return req.socket.remoteAddress;
}

// デバイスフィンガープリント対応のレート制限
function generateRateLimitKey(req) {
  const ip = getClientIp(req);
  const userAgent = req.headers['user-agent'] || '';
  const acceptLanguage = req.headers['accept-language'] || '';

  // フィンガープリントの生成
  const fingerprint = crypto
    .createHash('sha256')
    .update(`${ip}:${userAgent}:${acceptLanguage}`)
    .digest('hex')
    .substring(0, 16);

  return `rate:fp:${fingerprint}`;
}
```

### 10.4 コスト制御のためのレート制限

```javascript
// 外部API呼び出しのコスト制御
class CostAwareRateLimiter {
  constructor(redis, options) {
    this.redis = redis;
    this.costLimits = options.costLimits;
    // 例: { daily: 100.00, monthly: 2000.00 } (USD)
  }

  async isAllowed(key, estimatedCost) {
    const now = new Date();
    const dayKey = `cost:daily:${key}:${now.toISOString().split('T')[0]}`;
    const monthKey = `cost:monthly:${key}:${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;

    // 日次コストチェック
    const dailyCost = parseFloat(await this.redis.get(dayKey) || '0');
    if (dailyCost + estimatedCost > this.costLimits.daily) {
      return {
        allowed: false,
        reason: 'daily_cost_limit',
        currentCost: dailyCost,
        limit: this.costLimits.daily,
      };
    }

    // 月次コストチェック
    const monthlyCost = parseFloat(await this.redis.get(monthKey) || '0');
    if (monthlyCost + estimatedCost > this.costLimits.monthly) {
      return {
        allowed: false,
        reason: 'monthly_cost_limit',
        currentCost: monthlyCost,
        limit: this.costLimits.monthly,
      };
    }

    // コストを記録
    await this.redis.incrbyfloat(dayKey, estimatedCost);
    await this.redis.expire(dayKey, 86400 * 2); // 2日後に期限切れ
    await this.redis.incrbyfloat(monthKey, estimatedCost);
    await this.redis.expire(monthKey, 86400 * 35); // 35日後に期限切れ

    return {
      allowed: true,
      dailyCost: dailyCost + estimatedCost,
      monthlyCost: monthlyCost + estimatedCost,
      dailyRemaining: this.costLimits.daily - dailyCost - estimatedCost,
      monthlyRemaining: this.costLimits.monthly - monthlyCost - estimatedCost,
    };
  }
}

// 使用例: OpenAI API のコスト制御
const costLimiter = new CostAwareRateLimiter(redis, {
  costLimits: { daily: 50.00, monthly: 1000.00 },
});

app.post('/api/ai/generate', async (req, res) => {
  const estimatedTokens = estimateTokenCount(req.body.prompt);
  const estimatedCost = estimatedTokens * 0.00002; // $0.02/1K tokens

  const result = await costLimiter.isAllowed(req.userId, estimatedCost);

  if (!result.allowed) {
    return res.status(429).json({
      error: 'cost_limit_exceeded',
      reason: result.reason,
      currentCost: result.currentCost,
      limit: result.limit,
    });
  }

  // AI API呼び出し
  const aiResult = await callOpenAI(req.body.prompt);

  // 実際のコストを更新（推定と異なる場合）
  const actualCost = aiResult.usage.total_tokens * 0.00002;
  if (actualCost !== estimatedCost) {
    const diff = actualCost - estimatedCost;
    await redis.incrbyfloat(`cost:daily:${req.userId}:${today}`, diff);
    await redis.incrbyfloat(`cost:monthly:${req.userId}:${month}`, diff);
  }

  res.json(aiResult);
});
```

---

## まとめ

| アルゴリズム | 特徴 | メモリ | 精度 | 用途 |
|------------|------|--------|------|------|
| Fixed Window | シンプル、境界バースト問題 | 最小 | 低 | 簡易な制限 |
| Sliding Window Log | 正確だがメモリ消費大 | 大 | 最高 | 厳密な制限 |
| Sliding Window Counter | 精度とメモリのバランス | 小 | 高 | 一般的なAPI |
| Token Bucket | バースト許容、柔軟 | 小 | 高 | API GW, Nginx |
| Leaky Bucket | 出力一定、平滑化 | 中 | 高 | キュー処理 |

```
実務での推奨構成:

  ① 小規模API（<1000 req/min）:
     → メモリ内 Fixed Window で十分
     → 単一サーバーならRedis不要

  ② 中規模API（1000-10000 req/min）:
     → Redis + Sliding Window Counter
     → プランベースのレート制限

  ③ 大規模API（>10000 req/min）:
     → 多層レート制限（CDN + API GW + App）
     → ローカルバッファ付き分散レート制限
     → 動的レート制限（負荷適応型）

  ④ マイクロサービス:
     → サービスメッシュのレート制限（Istio, Envoy）
     → Circuit Breaker との連携
     → 各サービスのローカル制限 + グローバル制限
```

---

## 次に読むべきガイド
→ [[02-input-validation.md]] -- 入力バリデーション

---

## 参考文献
1. Stripe. "Rate Limiting." stripe.com/docs, 2024.
2. Cloudflare. "Rate Limiting Best Practices." blog.cloudflare.com, 2024.
3. draft-ietf-httpapi-ratelimit-headers. IETF, 2024.
4. Kong. "Rate Limiting Plugin." docs.konghq.com, 2024.
5. AWS. "API Gateway Throttling." docs.aws.amazon.com, 2024.
6. Google Cloud. "Rate Limiting Strategies." cloud.google.com/architecture, 2024.
7. Redis. "Rate Limiting with Redis." redis.io/glossary, 2024.
