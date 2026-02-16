# リトライ戦略

> 一時的な障害は避けられない。指数バックオフ、ジッター、サーキットブレーカーなど、信頼性の高いリトライ戦略を設計する。

## この章で学ぶこと

- [ ] リトライすべきエラーとすべきでないエラーを区別する
- [ ] 指数バックオフとジッターの仕組みを理解する
- [ ] サーキットブレーカーパターンを把握する
- [ ] リトライ戦略のテスト手法を習得する
- [ ] 分散システムにおけるリトライの設計を理解する
- [ ] バルクヘッドパターンとの組み合わせを学ぶ

---

## 1. リトライの基本

### 1.1 リトライすべきエラーの分類

```
リトライすべきエラー（一時的）:
  ✓ HTTP 429（Too Many Requests）
  ✓ HTTP 503（Service Unavailable）
  ✓ HTTP 502（Bad Gateway）
  ✓ HTTP 504（Gateway Timeout）
  ✓ ネットワークタイムアウト
  ✓ DBコネクションプール枯渇
  ✓ DNS一時障害
  ✓ TCP接続リセット（ECONNRESET）
  ✓ ソケット切断（EPIPE）
  ✓ 一時的なSSL/TLSハンドシェイク失敗
  ✓ AWS/GCP/Azure の一時的なAPIエラー

リトライすべきでないエラー（恒久的）:
  ✗ HTTP 400（Bad Request）— リクエストが間違い
  ✗ HTTP 401（Unauthorized）— 認証エラー
  ✗ HTTP 403（Forbidden）— 認可エラー
  ✗ HTTP 404（Not Found）— リソースが存在しない
  ✗ HTTP 405（Method Not Allowed）— メソッドが不正
  ✗ HTTP 409（Conflict）— 冪等でない操作の競合
  ✗ HTTP 413（Payload Too Large）— ペイロード過大
  ✗ HTTP 422（Unprocessable Entity）— バリデーションエラー
  ✗ ビジネスロジックエラー
  ✗ データ不整合エラー
```

### 1.2 リトライ判定の実装

```typescript
// リトライ可能なエラーかどうかを判定するヘルパー
class RetryPolicy {
  // HTTPステータスコードによる判定
  static isRetryableStatus(status: number): boolean {
    const retryableStatuses = new Set([
      408, // Request Timeout
      429, // Too Many Requests
      500, // Internal Server Error（場合による）
      502, // Bad Gateway
      503, // Service Unavailable
      504, // Gateway Timeout
    ]);
    return retryableStatuses.has(status);
  }

  // ネットワークエラーによる判定
  static isRetryableNetworkError(error: Error): boolean {
    const retryableCodes = new Set([
      'ECONNRESET',
      'ECONNREFUSED',
      'ENOTFOUND',
      'EPIPE',
      'ETIMEDOUT',
      'EAI_AGAIN',
      'EHOSTUNREACH',
      'ENETUNREACH',
    ]);

    const code = (error as NodeJS.ErrnoException).code;
    if (code && retryableCodes.has(code)) {
      return true;
    }

    // AbortError（タイムアウト）もリトライ可能
    if (error.name === 'AbortError') {
      return true;
    }

    return false;
  }

  // 総合判定
  static isRetryable(error: unknown): boolean {
    if (error instanceof HttpError) {
      return RetryPolicy.isRetryableStatus(error.statusCode);
    }
    if (error instanceof Error) {
      return RetryPolicy.isRetryableNetworkError(error);
    }
    return false;
  }
}
```

### 1.3 冪等性とリトライの関係

```
リトライ安全性の判断基準:

  冪等な操作（リトライ安全）:
    ✓ GET  — 読み取り（何度実行しても同じ結果）
    ✓ PUT  — 全体更新（同じ状態に上書き）
    ✓ DELETE — 削除（既に削除済みでも結果は同じ）
    ✓ HEAD — ヘッダー取得

  冪等でない操作（リトライ注意）:
    ⚠ POST — 作成（重複作成のリスク）
    ⚠ PATCH — 部分更新（相対値の場合、二重適用リスク）

  POST/PATCH をリトライ安全にする方法:
    → 冪等キー（Idempotency Key）の使用
    → クライアントがリクエストごとに一意のキーを生成
    → サーバーは同じキーのリクエストを二重処理しない
```

```typescript
// 冪等キーを使ったリトライ安全なPOST
class IdempotentClient {
  async createOrder(orderData: OrderData): Promise<Order> {
    const idempotencyKey = crypto.randomUUID();

    return retryWithBackoff(
      async () => {
        const response = await fetch('/api/orders', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Idempotency-Key': idempotencyKey, // 同じキーを使い回す
          },
          body: JSON.stringify(orderData),
        });

        if (!response.ok) {
          throw new HttpError(response.status, await response.text());
        }

        return response.json();
      },
      {
        maxRetries: 3,
        shouldRetry: (error) => {
          if (error instanceof HttpError) {
            // 409はすでに処理済みなので成功とみなしてもよい
            return RetryPolicy.isRetryableStatus(error.statusCode);
          }
          return true;
        },
      },
    );
  }
}

// サーバー側の冪等キー処理
class IdempotencyMiddleware {
  private store = new Map<string, { response: any; timestamp: number }>();

  async handle(req: Request, res: Response, next: NextFunction): Promise<void> {
    const key = req.headers['idempotency-key'] as string;
    if (!key) {
      return next();
    }

    // 既に処理済みのリクエスト
    const cached = this.store.get(key);
    if (cached) {
      res.json(cached.response);
      return;
    }

    // 処理中フラグ（他のリクエストをブロック）
    this.store.set(key, { response: null, timestamp: Date.now() });

    // 元のレスポンスを傍受
    const originalJson = res.json.bind(res);
    res.json = (body: any) => {
      this.store.set(key, { response: body, timestamp: Date.now() });
      return originalJson(body);
    };

    next();
  }
}
```

---

## 2. 指数バックオフ + ジッター

### 2.1 基本概念

```
指数バックオフ（Exponential Backoff）:
  リトライ1: 1秒後
  リトライ2: 2秒後
  リトライ3: 4秒後
  リトライ4: 8秒後
  リトライ5: 16秒後
  → wait = base × 2^(attempt - 1)
  → 上限（cap）を設けて無限に増えるのを防ぐ

ジッター（Jitter）:
  → ランダムな遅延を追加
  → 複数クライアントの同時リトライを避ける（thundering herd）

  ジッターなし: 全クライアントが同時にリトライ → サーバー再過負荷
  ジッターあり: リトライがばらける → サーバー負荷が分散

ジッターの種類:
  1. フルジッター（Full Jitter）:
     → wait = random(0, min(cap, base × 2^attempt))
     → 最も分散効果が高い

  2. 等価ジッター（Equal Jitter）:
     → temp = min(cap, base × 2^attempt)
     → wait = temp/2 + random(0, temp/2)
     → 最低待機時間を保証

  3. 相関ジッター（Decorrelated Jitter）:
     → wait = min(cap, random(base, prev_wait × 3))
     → 前回の待機時間に基づく
```

### 2.2 TypeScript 実装

```typescript
// 指数バックオフ + ジッター（フル実装）
interface RetryOptions {
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  jitterStrategy: 'full' | 'equal' | 'decorrelated';
  shouldRetry: (error: Error, attempt: number) => boolean;
  onRetry?: (error: Error, attempt: number, delayMs: number) => void;
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 30000,
  jitterStrategy: 'full',
  shouldRetry: () => true,
};

// ジッター計算
function calculateDelay(
  attempt: number,
  options: RetryOptions,
  previousDelay?: number,
): number {
  const { baseDelayMs, maxDelayMs, jitterStrategy } = options;
  const exponentialDelay = baseDelayMs * Math.pow(2, attempt);
  const cappedDelay = Math.min(exponentialDelay, maxDelayMs);

  switch (jitterStrategy) {
    case 'full':
      // フルジッター: [0, cappedDelay]
      return Math.random() * cappedDelay;

    case 'equal':
      // 等価ジッター: [cappedDelay/2, cappedDelay]
      return cappedDelay / 2 + Math.random() * (cappedDelay / 2);

    case 'decorrelated':
      // 相関ジッター: [baseDelayMs, previousDelay * 3]
      const prev = previousDelay ?? baseDelayMs;
      return Math.min(
        maxDelayMs,
        baseDelayMs + Math.random() * (prev * 3 - baseDelayMs),
      );

    default:
      return cappedDelay;
  }
}

// リトライ関数
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: Partial<RetryOptions> = {},
): Promise<T> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastError: Error;
  let previousDelay: number | undefined;

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === opts.maxRetries || !opts.shouldRetry(lastError, attempt)) {
        throw lastError;
      }

      const delay = calculateDelay(attempt, opts, previousDelay);
      previousDelay = delay;

      opts.onRetry?.(lastError, attempt + 1, delay);

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}
```

### 2.3 Python 実装

```python
import asyncio
import random
import logging
from typing import TypeVar, Callable, Awaitable, Optional
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger(__name__)


class RetryError(Exception):
    """全てのリトライが失敗した場合に送出"""
    def __init__(self, message: str, attempts: int, last_error: Exception):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


async def retry_with_backoff(
    fn: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: str = 'full',  # 'full', 'equal', 'decorrelated'
    should_retry: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    **kwargs,
) -> T:
    """指数バックオフ + ジッター付きリトライ"""
    last_error: Optional[Exception] = None
    prev_delay = base_delay

    for attempt in range(max_retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            last_error = e

            if attempt == max_retries:
                break

            if should_retry and not should_retry(e):
                raise

            # ジッター計算
            exp_delay = base_delay * (2 ** attempt)
            capped = min(exp_delay, max_delay)

            if jitter == 'full':
                delay = random.uniform(0, capped)
            elif jitter == 'equal':
                delay = capped / 2 + random.uniform(0, capped / 2)
            elif jitter == 'decorrelated':
                delay = min(max_delay, random.uniform(base_delay, prev_delay * 3))
            else:
                delay = capped

            prev_delay = delay

            if on_retry:
                on_retry(e, attempt + 1, delay)

            logger.warning(
                f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}"
            )
            await asyncio.sleep(delay)

    raise RetryError(
        f"All {max_retries} retries failed",
        attempts=max_retries,
        last_error=last_error,
    )


# デコレータ版
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    should_retry: Optional[Callable[[Exception], bool]] = None,
):
    """リトライデコレータ"""
    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_with_backoff(
                fn, *args,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                should_retry=should_retry,
                **kwargs,
            )
        return wrapper
    return decorator


# 使用例
@with_retry(max_retries=3, should_retry=lambda e: isinstance(e, (ConnectionError, TimeoutError)))
async def fetch_user_data(user_id: str) -> dict:
    """ユーザーデータを取得（リトライ付き）"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as resp:
            if resp.status >= 500:
                raise ConnectionError(f"Server error: {resp.status}")
            if resp.status == 429:
                raise ConnectionError("Rate limited")
            resp.raise_for_status()
            return await resp.json()
```

### 2.4 Go 実装

```go
package retry

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Config はリトライの設定
type Config struct {
	MaxRetries   int
	BaseDelay    time.Duration
	MaxDelay     time.Duration
	Jitter       JitterStrategy
	ShouldRetry  func(error) bool
	OnRetry      func(err error, attempt int, delay time.Duration)
}

type JitterStrategy int

const (
	FullJitter JitterStrategy = iota
	EqualJitter
	DecorrelatedJitter
)

// DefaultConfig はデフォルト設定
var DefaultConfig = Config{
	MaxRetries:  3,
	BaseDelay:   1 * time.Second,
	MaxDelay:    30 * time.Second,
	Jitter:     FullJitter,
	ShouldRetry: func(err error) bool { return true },
}

// Do はリトライ付きで関数を実行する
func Do(ctx context.Context, fn func(ctx context.Context) error, cfg Config) error {
	var lastErr error
	prevDelay := cfg.BaseDelay

	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		// コンテキストのキャンセルチェック
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		err := fn(ctx)
		if err == nil {
			return nil
		}
		lastErr = err

		if attempt == cfg.MaxRetries {
			break
		}

		if cfg.ShouldRetry != nil && !cfg.ShouldRetry(err) {
			return err
		}

		delay := calculateDelay(attempt, cfg, prevDelay)
		prevDelay = delay

		if cfg.OnRetry != nil {
			cfg.OnRetry(err, attempt+1, delay)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(delay):
		}
	}

	return fmt.Errorf("all %d retries failed: %w", cfg.MaxRetries, lastErr)
}

func calculateDelay(attempt int, cfg Config, prevDelay time.Duration) time.Duration {
	expDelay := time.Duration(float64(cfg.BaseDelay) * math.Pow(2, float64(attempt)))
	capped := expDelay
	if capped > cfg.MaxDelay {
		capped = cfg.MaxDelay
	}

	switch cfg.Jitter {
	case FullJitter:
		return time.Duration(rand.Int63n(int64(capped)))
	case EqualJitter:
		half := capped / 2
		return half + time.Duration(rand.Int63n(int64(half)))
	case DecorrelatedJitter:
		min := int64(cfg.BaseDelay)
		max := int64(prevDelay) * 3
		if max < min {
			max = min
		}
		d := time.Duration(min + rand.Int63n(max-min))
		if d > cfg.MaxDelay {
			d = cfg.MaxDelay
		}
		return d
	default:
		return capped
	}
}
```

### 2.5 使用例とベンチマーク

```typescript
// 各ジッター戦略の比較
function benchmarkJitter(): void {
  const strategies: Array<'full' | 'equal' | 'decorrelated'> = [
    'full', 'equal', 'decorrelated',
  ];

  for (const strategy of strategies) {
    const delays: number[] = [];
    for (let attempt = 0; attempt < 5; attempt++) {
      const samples: number[] = [];
      for (let i = 0; i < 1000; i++) {
        samples.push(calculateDelay(attempt, {
          ...DEFAULT_RETRY_OPTIONS,
          jitterStrategy: strategy,
        }));
      }
      const avg = samples.reduce((a, b) => a + b) / samples.length;
      const min = Math.min(...samples);
      const max = Math.max(...samples);
      delays.push(avg);
      console.log(
        `${strategy} attempt=${attempt}: avg=${avg.toFixed(0)}ms, ` +
        `min=${min.toFixed(0)}ms, max=${max.toFixed(0)}ms`
      );
    }
  }
}

// 出力例:
// full attempt=0: avg=497ms, min=2ms, max=999ms
// full attempt=1: avg=1003ms, min=3ms, max=1999ms
// full attempt=2: avg=1987ms, min=5ms, max=3998ms
// equal attempt=0: avg=752ms, min=500ms, max=999ms
// equal attempt=1: avg=1498ms, min=1000ms, max=1999ms
// decorrelated attempt=0: avg=1507ms, min=1001ms, max=2999ms
```

---

## 3. サーキットブレーカー

### 3.1 パターンの概要

```
サーキットブレーカーパターン:
  → 連続的な障害を検出して、リクエストを遮断する
  → 障害が回復するまで「即座に失敗」を返す
  → マイクロサービスの連鎖障害を防ぐ

  3つの状態:
  ┌────────┐  成功   ┌────────┐  連続失敗  ┌────────┐
  │ Closed │───────→│ Closed │──────────→│  Open  │
  │ (正常) │        │ (正常) │           │ (遮断) │
  └────────┘        └────────┘           └────┬───┘
                                              │ 一定時間経過
                                         ┌────▼─────┐
                                         │Half-Open │
                                         │ (試行)   │
                                         └────┬─────┘
                                      成功 ↙     ↘ 失敗
                                  ┌────────┐   ┌────────┐
                                  │ Closed │   │  Open  │
                                  └────────┘   └────────┘

設計上の考慮事項:
  1. 閾値の設定
     → 失敗回数ベース vs エラー率ベース
     → ウィンドウサイズ（直近N秒 or 直近Nリクエスト）

  2. リセット時間
     → Open → Half-Open までの待機時間
     → 短すぎると無意味、長すぎると回復が遅い

  3. Half-Open での挙動
     → 1リクエストだけ通す vs 限定的に通す
     → 成功率の閾値を設ける

  4. フォールバック
     → キャッシュから返す
     → デフォルト値を返す
     → 代替サービスを使う
```

### 3.2 TypeScript フル実装

```typescript
// サーキットブレーカーの状態
type CircuitState = 'closed' | 'open' | 'half-open';

// イベント
type CircuitEvent =
  | { type: 'state_change'; from: CircuitState; to: CircuitState }
  | { type: 'success'; duration: number }
  | { type: 'failure'; error: Error; duration: number }
  | { type: 'rejected' };

// 設定
interface CircuitBreakerConfig {
  failureThreshold: number;      // Open に遷移する失敗回数
  successThreshold: number;      // Half-Open → Closed に戻る成功回数
  resetTimeoutMs: number;        // Open → Half-Open までの時間
  halfOpenMaxConcurrent: number; // Half-Open 時に通すリクエスト数
  monitorWindowMs: number;       // 失敗カウントのウィンドウ
  errorRateThreshold?: number;   // エラー率閾値（0-1）
  minimumRequests?: number;      // エラー率を計算する最小リクエスト数
  onStateChange?: (from: CircuitState, to: CircuitState) => void;
  onEvent?: (event: CircuitEvent) => void;
}

class CircuitOpenError extends Error {
  constructor(message: string = 'Circuit breaker is open') {
    super(message);
    this.name = 'CircuitOpenError';
  }
}

class CircuitBreaker {
  private state: CircuitState = 'closed';
  private failureCount = 0;
  private successCount = 0;
  private lastFailureTime = 0;
  private halfOpenInFlight = 0;
  private requestLog: Array<{ timestamp: number; success: boolean }> = [];

  constructor(private readonly config: CircuitBreakerConfig) {}

  get currentState(): CircuitState {
    return this.state;
  }

  get stats(): {
    state: CircuitState;
    failureCount: number;
    successCount: number;
    errorRate: number;
  } {
    return {
      state: this.state,
      failureCount: this.failureCount,
      successCount: this.successCount,
      errorRate: this.getErrorRate(),
    };
  }

  async execute<T>(fn: () => Promise<T>, fallback?: () => Promise<T>): Promise<T> {
    if (!this.canExecute()) {
      this.emitEvent({ type: 'rejected' });

      if (fallback) {
        return fallback();
      }

      throw new CircuitOpenError(
        `Circuit breaker is ${this.state}. ` +
        `Failures: ${this.failureCount}/${this.config.failureThreshold}`
      );
    }

    if (this.state === 'half-open') {
      this.halfOpenInFlight++;
    }

    const startTime = Date.now();

    try {
      const result = await fn();
      this.onSuccess(Date.now() - startTime);
      return result;
    } catch (error) {
      this.onFailure(error as Error, Date.now() - startTime);
      throw error;
    } finally {
      if (this.state === 'half-open') {
        this.halfOpenInFlight--;
      }
    }
  }

  // 手動でリセット
  reset(): void {
    this.transitionTo('closed');
    this.failureCount = 0;
    this.successCount = 0;
    this.requestLog = [];
  }

  private canExecute(): boolean {
    switch (this.state) {
      case 'closed':
        return true;

      case 'open':
        if (Date.now() - this.lastFailureTime >= this.config.resetTimeoutMs) {
          this.transitionTo('half-open');
          return true;
        }
        return false;

      case 'half-open':
        return this.halfOpenInFlight < this.config.halfOpenMaxConcurrent;

      default:
        return false;
    }
  }

  private onSuccess(duration: number): void {
    this.recordRequest(true);
    this.emitEvent({ type: 'success', duration });

    switch (this.state) {
      case 'half-open':
        this.successCount++;
        if (this.successCount >= this.config.successThreshold) {
          this.transitionTo('closed');
          this.failureCount = 0;
          this.successCount = 0;
        }
        break;

      case 'closed':
        // 成功時は失敗カウントをリセット（設計による）
        this.failureCount = 0;
        break;
    }
  }

  private onFailure(error: Error, duration: number): void {
    this.recordRequest(false);
    this.lastFailureTime = Date.now();
    this.emitEvent({ type: 'failure', error, duration });

    switch (this.state) {
      case 'closed':
        this.failureCount++;
        if (this.shouldOpen()) {
          this.transitionTo('open');
        }
        break;

      case 'half-open':
        // Half-Open で1回でも失敗したら再度 Open
        this.transitionTo('open');
        this.successCount = 0;
        break;
    }
  }

  private shouldOpen(): boolean {
    // 失敗回数ベース
    if (this.failureCount >= this.config.failureThreshold) {
      return true;
    }

    // エラー率ベース（オプション）
    if (this.config.errorRateThreshold !== undefined) {
      const minReqs = this.config.minimumRequests ?? 10;
      const recentRequests = this.getRecentRequests();

      if (recentRequests.length >= minReqs) {
        const errorRate = this.getErrorRate();
        if (errorRate >= this.config.errorRateThreshold) {
          return true;
        }
      }
    }

    return false;
  }

  private recordRequest(success: boolean): void {
    const now = Date.now();
    this.requestLog.push({ timestamp: now, success });

    // ウィンドウ外の古いログを削除
    const windowStart = now - this.config.monitorWindowMs;
    this.requestLog = this.requestLog.filter(r => r.timestamp >= windowStart);
  }

  private getRecentRequests(): Array<{ timestamp: number; success: boolean }> {
    const windowStart = Date.now() - this.config.monitorWindowMs;
    return this.requestLog.filter(r => r.timestamp >= windowStart);
  }

  private getErrorRate(): number {
    const recent = this.getRecentRequests();
    if (recent.length === 0) return 0;
    const failures = recent.filter(r => !r.success).length;
    return failures / recent.length;
  }

  private transitionTo(newState: CircuitState): void {
    const oldState = this.state;
    if (oldState === newState) return;

    this.state = newState;
    this.config.onStateChange?.(oldState, newState);
    this.emitEvent({ type: 'state_change', from: oldState, to: newState });
  }

  private emitEvent(event: CircuitEvent): void {
    this.config.onEvent?.(event);
  }
}
```

### 3.3 Python 実装

```python
import asyncio
import time
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Awaitable, TypeVar, Optional, Generic

T = TypeVar('T')


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


class CircuitOpenError(Exception):
    """サーキットブレーカーが Open 状態のときに送出"""
    pass


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0  # seconds
    half_open_max_concurrent: int = 1
    monitor_window: float = 60.0  # seconds
    error_rate_threshold: Optional[float] = None
    minimum_requests: int = 10


@dataclass
class RequestRecord:
    timestamp: float
    success: bool


class AsyncCircuitBreaker:
    """非同期サーキットブレーカー"""

    def __init__(self, config: CircuitBreakerConfig = CircuitBreakerConfig()):
        self._config = config
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_semaphore = asyncio.Semaphore(config.half_open_max_concurrent)
        self._request_log: deque[RequestRecord] = deque()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    async def execute(
        self,
        fn: Callable[..., Awaitable[T]],
        *args,
        fallback: Optional[Callable[..., Awaitable[T]]] = None,
        **kwargs,
    ) -> T:
        async with self._lock:
            if not self._can_execute():
                if fallback:
                    return await fallback(*args, **kwargs)
                raise CircuitOpenError(
                    f"Circuit is {self._state.value}. "
                    f"Failures: {self._failure_count}/{self._config.failure_threshold}"
                )

        try:
            result = await fn(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise

    def _can_execute(self) -> bool:
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._config.reset_timeout:
                self._transition_to(CircuitState.HALF_OPEN)
                return True
            return False
        elif self._state == CircuitState.HALF_OPEN:
            return not self._half_open_semaphore.locked()
        return False

    async def _on_success(self) -> None:
        async with self._lock:
            self._record_request(True)
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._config.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
                    self._reset_counts()
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    async def _on_failure(self, error: Exception) -> None:
        async with self._lock:
            self._record_request(False)
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._should_open():
                    self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                self._success_count = 0

    def _should_open(self) -> bool:
        if self._failure_count >= self._config.failure_threshold:
            return True
        if self._config.error_rate_threshold is not None:
            recent = self._get_recent_requests()
            if len(recent) >= self._config.minimum_requests:
                error_rate = sum(1 for r in recent if not r.success) / len(recent)
                if error_rate >= self._config.error_rate_threshold:
                    return True
        return False

    def _record_request(self, success: bool) -> None:
        now = time.monotonic()
        self._request_log.append(RequestRecord(timestamp=now, success=success))
        cutoff = now - self._config.monitor_window
        while self._request_log and self._request_log[0].timestamp < cutoff:
            self._request_log.popleft()

    def _get_recent_requests(self) -> list[RequestRecord]:
        cutoff = time.monotonic() - self._config.monitor_window
        return [r for r in self._request_log if r.timestamp >= cutoff]

    def _transition_to(self, new_state: CircuitState) -> None:
        old_state = self._state
        self._state = new_state
        print(f"Circuit breaker: {old_state.value} -> {new_state.value}")

    def _reset_counts(self) -> None:
        self._failure_count = 0
        self._success_count = 0


# 使用例
async def main():
    breaker = AsyncCircuitBreaker(CircuitBreakerConfig(
        failure_threshold=3,
        reset_timeout=10.0,
    ))

    async def unreliable_api():
        import random
        if random.random() < 0.7:
            raise ConnectionError("Service unavailable")
        return {"status": "ok"}

    for i in range(20):
        try:
            result = await breaker.execute(
                unreliable_api,
                fallback=lambda: {"status": "fallback"},
            )
            print(f"Request {i}: {result}")
        except CircuitOpenError as e:
            print(f"Request {i}: REJECTED - {e}")
        except Exception as e:
            print(f"Request {i}: FAILED - {e}")
        await asyncio.sleep(1)
```

---

## 4. バルクヘッドパターン

### 4.1 概念

```
バルクヘッド（Bulkhead）パターン:
  → 船の隔壁に由来するパターン
  → リソースを区画に分離して、1つの障害が全体に波及しないようにする
  → サーキットブレーカーと組み合わせて使うことが多い

  例:
    サービスA用: 最大10接続
    サービスB用: 最大20接続
    サービスC用: 最大5接続

    → サービスAが遅延しても、サービスB・Cの接続には影響しない

  バルクヘッドの種類:
    1. スレッドプール隔離 — サービスごとに専用スレッドプールを割り当て
    2. セマフォ隔離 — 同時実行数を制限
    3. キュー隔離 — サービスごとに専用キューを割り当て
```

### 4.2 実装

```typescript
// セマフォベースのバルクヘッド
class Bulkhead {
  private currentConcurrency = 0;
  private queue: Array<{
    resolve: () => void;
    reject: (error: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }> = [];

  constructor(
    private readonly maxConcurrent: number,
    private readonly maxQueueSize: number = 100,
    private readonly queueTimeoutMs: number = 30000,
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();

    try {
      return await fn();
    } finally {
      this.release();
    }
  }

  private async acquire(): Promise<void> {
    if (this.currentConcurrency < this.maxConcurrent) {
      this.currentConcurrency++;
      return;
    }

    if (this.queue.length >= this.maxQueueSize) {
      throw new Error(
        `Bulkhead queue full (${this.maxQueueSize}). ` +
        `Current concurrency: ${this.currentConcurrency}`
      );
    }

    return new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        const index = this.queue.findIndex(item => item.resolve === resolve);
        if (index !== -1) {
          this.queue.splice(index, 1);
        }
        reject(new Error('Bulkhead queue timeout'));
      }, this.queueTimeoutMs);

      this.queue.push({ resolve, reject, timer });
    });
  }

  private release(): void {
    if (this.queue.length > 0) {
      const next = this.queue.shift()!;
      clearTimeout(next.timer);
      next.resolve();
    } else {
      this.currentConcurrency--;
    }
  }

  get stats(): { concurrent: number; queued: number } {
    return {
      concurrent: this.currentConcurrency,
      queued: this.queue.length,
    };
  }
}

// サービスごとにバルクヘッドを分離
class BulkheadRegistry {
  private bulkheads = new Map<string, Bulkhead>();

  get(
    name: string,
    maxConcurrent: number = 10,
    maxQueue: number = 50,
  ): Bulkhead {
    if (!this.bulkheads.has(name)) {
      this.bulkheads.set(name, new Bulkhead(maxConcurrent, maxQueue));
    }
    return this.bulkheads.get(name)!;
  }

  stats(): Record<string, { concurrent: number; queued: number }> {
    const result: Record<string, { concurrent: number; queued: number }> = {};
    for (const [name, bulkhead] of this.bulkheads) {
      result[name] = bulkhead.stats;
    }
    return result;
  }
}
```

---

## 5. 実践: HTTPクライアント

### 5.1 レジリエントHTTPクライアント

```typescript
// リトライ + サーキットブレーカー + バルクヘッド + タイムアウトを組み合わせ
interface ResilientClientConfig {
  timeout: number;
  maxRetries: number;
  circuitBreaker: CircuitBreakerConfig;
  bulkhead: {
    maxConcurrent: number;
    maxQueue: number;
  };
}

class ResilientHttpClient {
  private breaker: CircuitBreaker;
  private bulkhead: Bulkhead;

  constructor(private config: ResilientClientConfig) {
    this.breaker = new CircuitBreaker(config.circuitBreaker);
    this.bulkhead = new Bulkhead(
      config.bulkhead.maxConcurrent,
      config.bulkhead.maxQueue,
    );
  }

  async request(url: string, options?: RequestInit): Promise<Response> {
    // 外側: バルクヘッド（同時実行制限）
    return this.bulkhead.execute(() =>
      // 中間: サーキットブレーカー（障害遮断）
      this.breaker.execute(() =>
        // 内側: リトライ + タイムアウト
        retryWithBackoff(
          () => this.fetchWithTimeout(url, options),
          {
            maxRetries: this.config.maxRetries,
            shouldRetry: (error) => RetryPolicy.isRetryable(error),
            onRetry: (error, attempt, delay) => {
              console.log(
                `[${url}] Retry ${attempt} after ${delay.toFixed(0)}ms: ${error.message}`
              );
            },
          },
        ),
        // フォールバック
        async () => {
          console.warn(`[${url}] Circuit open, returning cached response`);
          return this.getCachedResponse(url);
        },
      ),
    );
  }

  private async fetchWithTimeout(
    url: string,
    options?: RequestInit,
  ): Promise<Response> {
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      this.config.timeout,
    );

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new HttpError(response.status, await response.text());
      }

      return response;
    } finally {
      clearTimeout(timeout);
    }
  }

  private async getCachedResponse(url: string): Promise<Response> {
    // 実装省略: キャッシュからレスポンスを返す
    throw new Error('No cached response available');
  }

  // 統計情報
  get stats() {
    return {
      circuitBreaker: this.breaker.stats,
      bulkhead: this.bulkhead.stats,
    };
  }
}

// 使用例
const client = new ResilientHttpClient({
  timeout: 5000,
  maxRetries: 3,
  circuitBreaker: {
    failureThreshold: 5,
    successThreshold: 2,
    resetTimeoutMs: 30000,
    halfOpenMaxConcurrent: 1,
    monitorWindowMs: 60000,
    onStateChange: (from, to) => {
      console.log(`Circuit breaker: ${from} -> ${to}`);
      if (to === 'open') {
        // アラート送信
        alertService.send('Circuit breaker opened', { service: 'payment' });
      }
    },
  },
  bulkhead: {
    maxConcurrent: 10,
    maxQueue: 50,
  },
});

// API呼び出し
const response = await client.request('https://api.payment.example.com/charge', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ amount: 1000, currency: 'JPY' }),
});
```

### 5.2 Retry-After ヘッダーの処理

```typescript
// Retry-After ヘッダーを考慮したリトライ
async function retryWithRetryAfter<T>(
  fn: () => Promise<Response>,
  maxRetries: number = 3,
): Promise<Response> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const response = await fn();

    if (response.status === 429 || response.status === 503) {
      if (attempt === maxRetries) {
        throw new HttpError(response.status, 'Max retries exceeded');
      }

      const retryAfter = response.headers.get('Retry-After');
      let delayMs: number;

      if (retryAfter) {
        // Retry-After は秒数または日付文字列
        const seconds = parseInt(retryAfter, 10);
        if (!isNaN(seconds)) {
          delayMs = seconds * 1000;
        } else {
          const date = new Date(retryAfter);
          delayMs = Math.max(0, date.getTime() - Date.now());
        }
      } else {
        // Retry-After がない場合は指数バックオフ
        delayMs = Math.min(1000 * Math.pow(2, attempt), 30000);
      }

      console.log(`Rate limited. Waiting ${delayMs}ms before retry.`);
      await new Promise(resolve => setTimeout(resolve, delayMs));
      continue;
    }

    return response;
  }

  throw new Error('Unexpected: loop exhausted');
}
```

---

## 6. 分散システムにおけるリトライ

### 6.1 リトライストームの防止

```
リトライストーム（Retry Storm）:
  → サービスAがサービスBを呼び、BがCを呼ぶ
  → Cが障害 → Bがリトライ × Aがリトライ
  → リトライの掛け算でCへのリクエストが爆発
  → A(3回) × B(3回) = Cに9回のリクエスト

  対策:
  1. リトライバジェット（Retry Budget）
     → 全体のリトライ率を制限（例: 直近のリクエストの10%まで）

  2. レイヤーごとのリトライ制限
     → エッジ（API Gateway）でのみリトライ
     → 内部サービス間ではリトライしない

  3. サーキットブレーカーの配置
     → 各サービス間にサーキットブレーカーを配置
     → 下流の障害を素早く遮断

  4. デッドラインの伝播
     → リクエストにデッドラインを付与
     → 残り時間が少なければリトライしない
```

```typescript
// リトライバジェット
class RetryBudget {
  private requestCount = 0;
  private retryCount = 0;
  private windowStart = Date.now();

  constructor(
    private readonly maxRetryRatio: number = 0.1, // 10%
    private readonly windowMs: number = 10000,    // 10秒
    private readonly minRetriesPerSecond: number = 10, // 最低保証
  ) {}

  canRetry(): boolean {
    this.maybeResetWindow();

    // 最低保証以下ならOK
    const windowSeconds = (Date.now() - this.windowStart) / 1000;
    const minRetries = this.minRetriesPerSecond * windowSeconds;
    if (this.retryCount < minRetries) {
      return true;
    }

    // リトライ率チェック
    if (this.requestCount === 0) return true;
    return this.retryCount / this.requestCount < this.maxRetryRatio;
  }

  recordRequest(): void {
    this.maybeResetWindow();
    this.requestCount++;
  }

  recordRetry(): void {
    this.maybeResetWindow();
    this.retryCount++;
  }

  private maybeResetWindow(): void {
    if (Date.now() - this.windowStart > this.windowMs) {
      this.requestCount = 0;
      this.retryCount = 0;
      this.windowStart = Date.now();
    }
  }
}

// デッドラインの伝播
class DeadlineContext {
  private deadline: number;

  constructor(timeoutMs: number) {
    this.deadline = Date.now() + timeoutMs;
  }

  get remaining(): number {
    return Math.max(0, this.deadline - Date.now());
  }

  get isExpired(): boolean {
    return this.remaining <= 0;
  }

  // 子リクエスト用のサブデッドラインを作成
  child(marginMs: number = 100): DeadlineContext {
    const remaining = this.remaining - marginMs;
    if (remaining <= 0) {
      throw new Error('Deadline already expired');
    }
    const ctx = new DeadlineContext(0);
    ctx.deadline = Date.now() + remaining;
    return ctx;
  }

  shouldRetry(estimatedDurationMs: number): boolean {
    return this.remaining > estimatedDurationMs;
  }
}

// 使用例: デッドライン付きリトライ
async function retryWithDeadline<T>(
  fn: () => Promise<T>,
  deadline: DeadlineContext,
  options: {
    maxRetries: number;
    baseDelayMs: number;
  },
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
    if (deadline.isExpired) {
      throw new Error('Deadline expired before retry');
    }

    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === options.maxRetries) break;

      const delay = options.baseDelayMs * Math.pow(2, attempt);

      if (!deadline.shouldRetry(delay)) {
        throw new Error(
          `Insufficient time for retry (remaining: ${deadline.remaining}ms, ` +
          `needed: ${delay}ms)`
        );
      }

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
}
```

### 6.2 gRPC のリトライ戦略

```
gRPC リトライポリシー:
  → gRPC ではリトライポリシーをサービス設定で宣言的に定義可能

  リトライ可能なステータスコード:
    UNAVAILABLE  — サービスが利用不可（一時的）
    DEADLINE_EXCEEDED — デッドライン超過
    RESOURCE_EXHAUSTED — リソース枯渇
    ABORTED — トランザクション競合

  リトライ不可:
    INVALID_ARGUMENT — 不正な引数
    NOT_FOUND — リソースが存在しない
    PERMISSION_DENIED — 権限なし
    UNAUTHENTICATED — 認証なし
```

```json
{
  "methodConfig": [{
    "name": [{ "service": "mypackage.MyService" }],
    "retryPolicy": {
      "maxAttempts": 4,
      "initialBackoff": "0.5s",
      "maxBackoff": "30s",
      "backoffMultiplier": 2,
      "retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"]
    }
  }]
}
```

---

## 7. リトライ戦略のテスト

### 7.1 ユニットテスト

```typescript
describe('retryWithBackoff', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('成功した場合はリトライしない', async () => {
    const fn = jest.fn().mockResolvedValue('success');

    const result = await retryWithBackoff(fn, { maxRetries: 3 });

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('一時的な障害後に成功する', async () => {
    const fn = jest.fn()
      .mockRejectedValueOnce(new Error('transient'))
      .mockRejectedValueOnce(new Error('transient'))
      .mockResolvedValue('success');

    const promise = retryWithBackoff(fn, {
      maxRetries: 3,
      baseDelayMs: 100,
      jitterStrategy: 'equal', // テストで予測しやすいジッター
    });

    // 1回目のリトライ待ち
    await jest.advanceTimersByTimeAsync(200);
    // 2回目のリトライ待ち
    await jest.advanceTimersByTimeAsync(400);

    const result = await promise;
    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  test('最大リトライ回数を超えたらエラーを投げる', async () => {
    const error = new Error('persistent failure');
    const fn = jest.fn().mockRejectedValue(error);

    const promise = retryWithBackoff(fn, { maxRetries: 2, baseDelayMs: 100 });

    await jest.advanceTimersByTimeAsync(100);
    await jest.advanceTimersByTimeAsync(200);

    await expect(promise).rejects.toThrow('persistent failure');
    expect(fn).toHaveBeenCalledTimes(3); // 初回 + 2回リトライ
  });

  test('shouldRetry が false を返したらリトライしない', async () => {
    const fn = jest.fn().mockRejectedValue(new HttpError(404));

    await expect(
      retryWithBackoff(fn, {
        maxRetries: 3,
        shouldRetry: (err) => !(err instanceof HttpError && err.statusCode === 404),
      })
    ).rejects.toThrow();

    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('onRetry コールバックが正しく呼ばれる', async () => {
    const onRetry = jest.fn();
    const fn = jest.fn()
      .mockRejectedValueOnce(new Error('fail1'))
      .mockResolvedValue('success');

    const promise = retryWithBackoff(fn, {
      maxRetries: 3,
      baseDelayMs: 100,
      onRetry,
    });

    await jest.advanceTimersByTimeAsync(200);
    await promise;

    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(onRetry).toHaveBeenCalledWith(
      expect.any(Error),
      1,
      expect.any(Number),
    );
  });
});

describe('CircuitBreaker', () => {
  test('閾値に達したら Open になる', async () => {
    const breaker = new CircuitBreaker({
      failureThreshold: 3,
      successThreshold: 1,
      resetTimeoutMs: 1000,
      halfOpenMaxConcurrent: 1,
      monitorWindowMs: 60000,
    });

    const failingFn = () => Promise.reject(new Error('fail'));

    // 3回失敗
    for (let i = 0; i < 3; i++) {
      await expect(breaker.execute(failingFn)).rejects.toThrow();
    }

    // 4回目は CircuitOpenError
    await expect(breaker.execute(failingFn)).rejects.toThrow(CircuitOpenError);
    expect(breaker.currentState).toBe('open');
  });

  test('リセット時間後に Half-Open になる', async () => {
    jest.useFakeTimers();

    const breaker = new CircuitBreaker({
      failureThreshold: 2,
      successThreshold: 1,
      resetTimeoutMs: 5000,
      halfOpenMaxConcurrent: 1,
      monitorWindowMs: 60000,
    });

    const failingFn = () => Promise.reject(new Error('fail'));

    // Open にする
    await expect(breaker.execute(failingFn)).rejects.toThrow();
    await expect(breaker.execute(failingFn)).rejects.toThrow();
    expect(breaker.currentState).toBe('open');

    // 5秒待つ
    jest.advanceTimersByTime(5001);

    // 次のリクエストで Half-Open になる
    const successFn = () => Promise.resolve('ok');
    const result = await breaker.execute(successFn);
    expect(result).toBe('ok');
    expect(breaker.currentState).toBe('closed');

    jest.useRealTimers();
  });
});
```

### 7.2 統合テスト

```typescript
// Nock を使った統合テスト
import nock from 'nock';

describe('ResilientHttpClient Integration', () => {
  afterEach(() => {
    nock.cleanAll();
  });

  test('一時的な503エラー後にリトライで成功する', async () => {
    nock('https://api.example.com')
      .get('/data')
      .reply(503, 'Service Unavailable')
      .get('/data')
      .reply(503, 'Service Unavailable')
      .get('/data')
      .reply(200, { result: 'success' });

    const client = new ResilientHttpClient({
      timeout: 5000,
      maxRetries: 3,
      circuitBreaker: {
        failureThreshold: 10,
        successThreshold: 1,
        resetTimeoutMs: 30000,
        halfOpenMaxConcurrent: 1,
        monitorWindowMs: 60000,
      },
      bulkhead: { maxConcurrent: 10, maxQueue: 50 },
    });

    const response = await client.request('https://api.example.com/data');
    const data = await response.json();
    expect(data.result).toBe('success');
  });

  test('Retry-After ヘッダーを尊重する', async () => {
    const start = Date.now();

    nock('https://api.example.com')
      .get('/limited')
      .reply(429, 'Too Many Requests', { 'Retry-After': '2' })
      .get('/limited')
      .reply(200, { result: 'ok' });

    const response = await retryWithRetryAfter(
      () => fetch('https://api.example.com/limited'),
      3,
    );

    const elapsed = Date.now() - start;
    expect(elapsed).toBeGreaterThanOrEqual(2000);
    expect(response.status).toBe(200);
  });
});
```

---

## 8. 主要ライブラリ・フレームワーク

### 8.1 各言語のリトライライブラリ

```
TypeScript/JavaScript:
  - p-retry: Promise ベースのリトライ
  - cockatiel: サーキットブレーカー + リトライ + バルクヘッド
  - axios-retry: Axios 用リトライプラグイン
  - got: HTTP クライアント（リトライ内蔵）

Python:
  - tenacity: 汎用リトライライブラリ
  - aiohttp-retry: aiohttp 用リトライ
  - stamina: 最新のリトライライブラリ
  - pybreaker: サーキットブレーカー

Go:
  - cenkalti/backoff: 指数バックオフ
  - sony/gobreaker: サーキットブレーカー
  - avast/retry-go: リトライライブラリ
  - hashicorp/go-retryablehttp: HTTPクライアント

Java/Kotlin:
  - resilience4j: サーキットブレーカー + リトライ + バルクヘッド
  - Spring Retry: Springフレームワーク用
  - Failsafe: リトライ + サーキットブレーカー

Rust:
  - backon: 非同期リトライ
  - reqwest-retry: reqwest用リトライミドルウェア
```

### 8.2 ライブラリ使用例

```typescript
// cockatiel の使用例
import {
  retry,
  handleAll,
  ExponentialBackoff,
  CircuitBreakerPolicy,
  ConsecutiveBreaker,
  wrap,
  bulkhead,
} from 'cockatiel';

// リトライポリシー
const retryPolicy = retry(handleAll, {
  maxAttempts: 3,
  backoff: new ExponentialBackoff({
    initialDelay: 1000,
    maxDelay: 30000,
  }),
});

// サーキットブレーカーポリシー
const circuitBreakerPolicy = new CircuitBreakerPolicy(handleAll, {
  halfOpenAfter: 30000,
  breaker: new ConsecutiveBreaker(5),
});

// バルクヘッドポリシー
const bulkheadPolicy = bulkhead(10, 50);

// ポリシーを組み合わせ
const policy = wrap(bulkheadPolicy, circuitBreakerPolicy, retryPolicy);

// 使用
const result = await policy.execute(() => fetch('https://api.example.com/data'));
```

```python
# tenacity の使用例
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    after_log,
)
import logging

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    after=after_log(logger, logging.INFO),
)
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=5) as resp:
            resp.raise_for_status()
            return await resp.json()


# カスタムリトライ条件
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(ConnectionError),
)
async def create_order(order_data: dict) -> dict:
    return await api_client.post("/orders", json=order_data)
```

```go
// go-retryablehttp の使用例
package main

import (
	"log"
	"net/http"
	"time"

	"github.com/hashicorp/go-retryablehttp"
)

func main() {
	client := retryablehttp.NewClient()
	client.RetryMax = 3
	client.RetryWaitMin = 1 * time.Second
	client.RetryWaitMax = 30 * time.Second
	client.Logger = log.Default()

	// カスタムリトライ条件
	client.CheckRetry = func(ctx context.Context, resp *http.Response, err error) (bool, error) {
		if err != nil {
			return true, nil // ネットワークエラーはリトライ
		}
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			return true, nil
		}
		return false, nil
	}

	resp, err := client.Get("https://api.example.com/data")
	if err != nil {
		log.Fatal(err)
	}
	defer resp.Body.Close()
}
```

---

## 9. 運用のベストプラクティス

### 9.1 メトリクスと監視

```
リトライに関する重要メトリクス:
  1. リトライ率（retry rate）
     → 全リクエストに対するリトライの割合
     → 高い場合は下流サービスに問題がある

  2. リトライ成功率（retry success rate）
     → リトライにより最終的に成功した割合
     → 低い場合はリトライが無駄に負荷をかけている

  3. サーキットブレーカーの状態遷移回数
     → Open になる頻度
     → Half-Open → Open に戻る頻度

  4. 平均リトライ回数
     → 成功するまでの平均リトライ回数
     → 増加傾向なら問題の悪化を示す

  5. デッドレターキューのサイズ
     → リトライが全て失敗したジョブの数
     → 増加している場合はアラート
```

```typescript
// Prometheus メトリクスの収集例
import { Counter, Histogram, Gauge } from 'prom-client';

const retryCounter = new Counter({
  name: 'http_client_retries_total',
  help: 'Total number of retries',
  labelNames: ['service', 'status', 'method'],
});

const retryDuration = new Histogram({
  name: 'http_client_retry_duration_seconds',
  help: 'Duration of retry cycles',
  labelNames: ['service', 'result'],
  buckets: [0.1, 0.5, 1, 2, 5, 10, 30],
});

const circuitBreakerState = new Gauge({
  name: 'circuit_breaker_state',
  help: 'Circuit breaker state (0=closed, 1=open, 2=half-open)',
  labelNames: ['service'],
});

// メトリクス付きリトライ
async function retryWithMetrics<T>(
  serviceName: string,
  fn: () => Promise<T>,
  options: Partial<RetryOptions>,
): Promise<T> {
  const timer = retryDuration.startTimer({ service: serviceName });

  try {
    const result = await retryWithBackoff(fn, {
      ...options,
      onRetry: (error, attempt, delay) => {
        retryCounter.inc({
          service: serviceName,
          status: error instanceof HttpError ? String(error.statusCode) : 'network',
          method: 'GET',
        });
        options.onRetry?.(error, attempt, delay);
      },
    });

    timer({ result: 'success' });
    return result;
  } catch (error) {
    timer({ result: 'failure' });
    throw error;
  }
}
```

### 9.2 設定のチューニング

```
リトライ設定のガイドライン:

  外部 API コール:
    maxRetries: 3
    baseDelay: 1000ms
    maxDelay: 30000ms
    jitter: full

  データベース接続:
    maxRetries: 5
    baseDelay: 500ms
    maxDelay: 10000ms
    jitter: equal

  メッセージキュー:
    maxRetries: 10
    baseDelay: 1000ms
    maxDelay: 60000ms
    jitter: decorrelated

  サーキットブレーカー:
    failureThreshold: 5-10
    resetTimeout: 15-60秒
    halfOpenMaxConcurrent: 1-3

  バルクヘッド:
    maxConcurrent: サービスのキャパシティに応じて
    maxQueue: maxConcurrent の 2-5 倍

  注意:
    → リトライ回数 × バックオフ時間 < リクエストタイムアウト
    → 上流のタイムアウトを考慮（デッドラインの伝播）
    → 負荷テストでチューニングを検証
```

---

## まとめ

| 戦略 | 目的 | 適用場面 |
|------|------|---------|
| 固定間隔リトライ | 単純な再試行 | 軽微な一時障害 |
| 指数バックオフ | 負荷を段階的に下げる | API制限、サーバー障害 |
| ジッター | 同時リトライを分散 | 多クライアント環境 |
| サーキットブレーカー | 連鎖障害の防止 | マイクロサービス |
| バルクヘッド | リソース隔離 | マルチサービス依存 |
| リトライバジェット | リトライストーム防止 | 分散システム |
| デッドライン伝播 | タイムアウト連鎖の制御 | 多段呼び出し |
| 冪等キー | 安全なリトライ | POST/PATCH操作 |

---

## 次に読むべきガイド
→ [[03-structured-concurrency.md]] — 構造化並行性

---

## 参考文献
1. AWS Architecture Blog. "Exponential Backoff and Jitter." 2015.
2. Nygard, M. "Release It!" Pragmatic Bookshelf, 2018.
3. Google SRE Book. "Handling Overload." O'Reilly, 2016.
4. Polly Project. "Resilience and transient-fault-handling." GitHub.
5. Netflix Tech Blog. "Making the Netflix API More Resilient." 2011.
6. Fowler, M. "CircuitBreaker." martinfowler.com, 2014.
7. gRPC Documentation. "Retry Design." grpc.io.
8. Microsoft Azure Architecture Center. "Retry Pattern." docs.microsoft.com.
9. Cockroach Labs. "Building a Resilient System with Retry Budgets." 2020.
10. Envoy Proxy. "Circuit Breaking." envoyproxy.io.
