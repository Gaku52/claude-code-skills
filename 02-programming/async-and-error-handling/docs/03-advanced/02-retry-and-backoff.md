# Retry Strategies

> Transient failures are inevitable. Design reliable retry strategies using exponential backoff, jitter, circuit breakers, and more.

## What You Will Learn in This Chapter

- [ ] Distinguish between errors that should and should not be retried
- [ ] Understand how exponential backoff and jitter work
- [ ] Grasp the circuit breaker pattern
- [ ] Master testing techniques for retry strategies
- [ ] Understand retry design in distributed systems
- [ ] Learn how to combine retry with the bulkhead pattern


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Cancellation Handling](./01-cancellation.md)

---

## 1. Retry Fundamentals

### 1.1 Classifying Errors for Retry

```
Errors that should be retried (transient):
  ✓ HTTP 429 (Too Many Requests)
  ✓ HTTP 503 (Service Unavailable)
  ✓ HTTP 502 (Bad Gateway)
  ✓ HTTP 504 (Gateway Timeout)
  ✓ Network timeouts
  ✓ DB connection pool exhaustion
  ✓ Temporary DNS failures
  ✓ TCP connection reset (ECONNRESET)
  ✓ Socket disconnection (EPIPE)
  ✓ Temporary SSL/TLS handshake failures
  ✓ Transient AWS/GCP/Azure API errors

Errors that should NOT be retried (permanent):
  ✗ HTTP 400 (Bad Request) — Invalid request
  ✗ HTTP 401 (Unauthorized) — Authentication error
  ✗ HTTP 403 (Forbidden) — Authorization error
  ✗ HTTP 404 (Not Found) — Resource does not exist
  ✗ HTTP 405 (Method Not Allowed) — Invalid method
  ✗ HTTP 409 (Conflict) — Conflict on non-idempotent operation
  ✗ HTTP 413 (Payload Too Large) — Payload too large
  ✗ HTTP 422 (Unprocessable Entity) — Validation error
  ✗ Business logic errors
  ✗ Data inconsistency errors
```

### 1.2 Implementing Retry Decision Logic

```typescript
// Helper to determine whether an error is retryable
class RetryPolicy {
  // Determine by HTTP status code
  static isRetryableStatus(status: number): boolean {
    const retryableStatuses = new Set([
      408, // Request Timeout
      429, // Too Many Requests
      500, // Internal Server Error (depends on context)
      502, // Bad Gateway
      503, // Service Unavailable
      504, // Gateway Timeout
    ]);
    return retryableStatuses.has(status);
  }

  // Determine by network error
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

    // AbortError (timeout) is also retryable
    if (error.name === 'AbortError') {
      return true;
    }

    return false;
  }

  // Combined determination
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

### 1.3 Idempotency and Its Relationship to Retry

```
Criteria for retry safety:

  Idempotent operations (safe to retry):
    ✓ GET  — Read (same result no matter how many times executed)
    ✓ PUT  — Full update (overwrites to the same state)
    ✓ DELETE — Delete (result is the same even if already deleted)
    ✓ HEAD — Header retrieval

  Non-idempotent operations (retry with caution):
    ⚠ POST — Create (risk of duplicate creation)
    ⚠ PATCH — Partial update (risk of double application with relative values)

  How to make POST/PATCH retry-safe:
    → Use an Idempotency Key
    → Client generates a unique key for each request
    → Server does not process the same key twice
```

```typescript
// Retry-safe POST using idempotency keys
class IdempotentClient {
  async createOrder(orderData: OrderData): Promise<Order> {
    const idempotencyKey = crypto.randomUUID();

    return retryWithBackoff(
      async () => {
        const response = await fetch('/api/orders', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Idempotency-Key': idempotencyKey, // Reuse the same key
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
            // 409 means already processed, so it can be treated as success
            return RetryPolicy.isRetryableStatus(error.statusCode);
          }
          return true;
        },
      },
    );
  }
}

// Server-side idempotency key handling
class IdempotencyMiddleware {
  private store = new Map<string, { response: any; timestamp: number }>();

  async handle(req: Request, res: Response, next: NextFunction): Promise<void> {
    const key = req.headers['idempotency-key'] as string;
    if (!key) {
      return next();
    }

    // Already processed request
    const cached = this.store.get(key);
    if (cached) {
      res.json(cached.response);
      return;
    }

    // In-progress flag (blocks other requests)
    this.store.set(key, { response: null, timestamp: Date.now() });

    // Intercept the original response
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

## 2. Exponential Backoff + Jitter

### 2.1 Core Concepts

```
Exponential Backoff:
  Retry 1: after 1 second
  Retry 2: after 2 seconds
  Retry 3: after 4 seconds
  Retry 4: after 8 seconds
  Retry 5: after 16 seconds
  → wait = base × 2^(attempt - 1)
  → Set a cap to prevent unbounded growth

Jitter:
  → Add random delay
  → Avoid simultaneous retries from multiple clients (thundering herd)

  Without jitter: All clients retry at the same time → Server overloaded again
  With jitter: Retries are spread out → Server load is distributed

Types of jitter:
  1. Full Jitter:
     → wait = random(0, min(cap, base × 2^attempt))
     → Highest distribution effect

  2. Equal Jitter:
     → temp = min(cap, base × 2^attempt)
     → wait = temp/2 + random(0, temp/2)
     → Guarantees a minimum wait time

  3. Decorrelated Jitter:
     → wait = min(cap, random(base, prev_wait × 3))
     → Based on the previous wait time
```

### 2.2 TypeScript Implementation

```typescript
// Exponential backoff + jitter (full implementation)
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

// Jitter calculation
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
      // Full jitter: [0, cappedDelay]
      return Math.random() * cappedDelay;

    case 'equal':
      // Equal jitter: [cappedDelay/2, cappedDelay]
      return cappedDelay / 2 + Math.random() * (cappedDelay / 2);

    case 'decorrelated':
      // Decorrelated jitter: [baseDelayMs, previousDelay * 3]
      const prev = previousDelay ?? baseDelayMs;
      return Math.min(
        maxDelayMs,
        baseDelayMs + Math.random() * (prev * 3 - baseDelayMs),
      );

    default:
      return cappedDelay;
  }
}

// Retry function
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

### 2.3 Python Implementation

```python
import asyncio
import random
import logging
from typing import TypeVar, Callable, Awaitable, Optional
from functools import wraps

T = TypeVar('T')
logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Raised when all retries have failed"""
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
    """Retry with exponential backoff + jitter"""
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

            # Jitter calculation
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


# Decorator version
def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    should_retry: Optional[Callable[[Exception], bool]] = None,
):
    """Retry decorator"""
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


# Usage example
@with_retry(max_retries=3, should_retry=lambda e: isinstance(e, (ConnectionError, TimeoutError)))
async def fetch_user_data(user_id: str) -> dict:
    """Fetch user data (with retry)"""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as resp:
            if resp.status >= 500:
                raise ConnectionError(f"Server error: {resp.status}")
            if resp.status == 429:
                raise ConnectionError("Rate limited")
            resp.raise_for_status()
            return await resp.json()
```

### 2.4 Go Implementation

```go
package retry

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"time"
)

// Config holds the retry configuration
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

// DefaultConfig provides default settings
var DefaultConfig = Config{
	MaxRetries:  3,
	BaseDelay:   1 * time.Second,
	MaxDelay:    30 * time.Second,
	Jitter:     FullJitter,
	ShouldRetry: func(err error) bool { return true },
}

// Do executes a function with retry
func Do(ctx context.Context, fn func(ctx context.Context) error, cfg Config) error {
	var lastErr error
	prevDelay := cfg.BaseDelay

	for attempt := 0; attempt <= cfg.MaxRetries; attempt++ {
		// Check for context cancellation
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

### 2.5 Usage Examples and Benchmarks

```typescript
// Comparison of each jitter strategy
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

// Example output:
// full attempt=0: avg=497ms, min=2ms, max=999ms
// full attempt=1: avg=1003ms, min=3ms, max=1999ms
// full attempt=2: avg=1987ms, min=5ms, max=3998ms
// equal attempt=0: avg=752ms, min=500ms, max=999ms
// equal attempt=1: avg=1498ms, min=1000ms, max=1999ms
// decorrelated attempt=0: avg=1507ms, min=1001ms, max=2999ms
```

---

## 3. Circuit Breaker

### 3.1 Pattern Overview

```
Circuit Breaker Pattern:
  → Detects continuous failures and blocks requests
  → Returns "fail fast" until the failure recovers
  → Prevents cascading failures in microservices

  Three states:
  ┌────────┐  Success  ┌────────┐  Consecutive  ┌────────┐
  │ Closed │─────────→│ Closed │  failures    →│  Open  │
  │(Normal)│          │(Normal)│              │(Blocked)│
  └────────┘          └────────┘              └────┬───┘
                                                   │ After a set period
                                              ┌────▼─────┐
                                              │Half-Open │
                                              │ (Trial)  │
                                              └────┬─────┘
                                           Success ↙     ↘ Failure
                                       ┌────────┐   ┌────────┐
                                       │ Closed │   │  Open  │
                                       └────────┘   └────────┘

Design considerations:
  1. Threshold settings
     → Failure count-based vs. error rate-based
     → Window size (last N seconds or last N requests)

  2. Reset time
     → Wait time from Open → Half-Open
     → Too short is meaningless, too long delays recovery

  3. Half-Open behavior
     → Allow only 1 request vs. allow a limited number
     → Set a success rate threshold

  4. Fallback
     → Return from cache
     → Return default values
     → Use an alternative service
```

### 3.2 Full TypeScript Implementation

```typescript
// Circuit breaker states
type CircuitState = 'closed' | 'open' | 'half-open';

// Events
type CircuitEvent =
  | { type: 'state_change'; from: CircuitState; to: CircuitState }
  | { type: 'success'; duration: number }
  | { type: 'failure'; error: Error; duration: number }
  | { type: 'rejected' };

// Configuration
interface CircuitBreakerConfig {
  failureThreshold: number;      // Number of failures to transition to Open
  successThreshold: number;      // Number of successes to transition Half-Open → Closed
  resetTimeoutMs: number;        // Time from Open → Half-Open
  halfOpenMaxConcurrent: number; // Number of requests allowed in Half-Open
  monitorWindowMs: number;       // Window for failure counting
  errorRateThreshold?: number;   // Error rate threshold (0-1)
  minimumRequests?: number;      // Minimum requests to calculate error rate
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

  // Manual reset
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
        // Reset failure count on success (design choice)
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
        // Any failure in Half-Open transitions back to Open
        this.transitionTo('open');
        this.successCount = 0;
        break;
    }
  }

  private shouldOpen(): boolean {
    // Failure count-based
    if (this.failureCount >= this.config.failureThreshold) {
      return true;
    }

    // Error rate-based (optional)
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

    // Remove old logs outside the window
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

### 3.3 Python Implementation

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
    """Raised when the circuit breaker is in the Open state"""
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
    """Asynchronous circuit breaker"""

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


# Usage example
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

## 4. Bulkhead Pattern

### 4.1 Concept

```
Bulkhead Pattern:
  → A pattern derived from ship compartment walls (bulkheads)
  → Isolates resources into compartments so that a failure in one does not spread to the whole
  → Often used in combination with circuit breakers

  Example:
    Service A: max 10 connections
    Service B: max 20 connections
    Service C: max 5 connections

    → Even if Service A experiences delays, connections to Services B and C are unaffected

  Types of bulkheads:
    1. Thread pool isolation — Dedicated thread pool per service
    2. Semaphore isolation — Limit concurrent execution count
    3. Queue isolation — Dedicated queue per service
```

### 4.2 Implementation

```typescript
// Semaphore-based bulkhead
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

// Isolate bulkheads per service
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

## 5. Practical Example: HTTP Client

### 5.1 Resilient HTTP Client

```typescript
// Combining retry + circuit breaker + bulkhead + timeout
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
    // Outer layer: Bulkhead (concurrency limiting)
    return this.bulkhead.execute(() =>
      // Middle layer: Circuit breaker (failure blocking)
      this.breaker.execute(() =>
        // Inner layer: Retry + timeout
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
        // Fallback
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
    // Implementation omitted: return response from cache
    throw new Error('No cached response available');
  }

  // Statistics
  get stats() {
    return {
      circuitBreaker: this.breaker.stats,
      bulkhead: this.bulkhead.stats,
    };
  }
}

// Usage example
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
        // Send alert
        alertService.send('Circuit breaker opened', { service: 'payment' });
      }
    },
  },
  bulkhead: {
    maxConcurrent: 10,
    maxQueue: 50,
  },
});

// API call
const response = await client.request('https://api.payment.example.com/charge', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ amount: 1000, currency: 'JPY' }),
});
```

### 5.2 Handling the Retry-After Header

```typescript
// Retry that respects the Retry-After header
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
        // Retry-After can be seconds or a date string
        const seconds = parseInt(retryAfter, 10);
        if (!isNaN(seconds)) {
          delayMs = seconds * 1000;
        } else {
          const date = new Date(retryAfter);
          delayMs = Math.max(0, date.getTime() - Date.now());
        }
      } else {
        // Exponential backoff when Retry-After is not present
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

## 6. Retry in Distributed Systems

### 6.1 Preventing Retry Storms

```
Retry Storm:
  → Service A calls Service B, which calls C
  → C fails → B retries × A retries
  → Retries multiply, causing an explosion of requests to C
  → A(3 retries) × B(3 retries) = 9 requests to C

  Countermeasures:
  1. Retry Budget
     → Limit the overall retry rate (e.g., up to 10% of recent requests)

  2. Per-layer retry limits
     → Retry only at the edge (API Gateway)
     → No retries between internal services

  3. Circuit breaker placement
     → Place circuit breakers between each service
     → Quickly block downstream failures

  4. Deadline propagation
     → Attach a deadline to each request
     → Do not retry if remaining time is insufficient
```

```typescript
// Retry budget
class RetryBudget {
  private requestCount = 0;
  private retryCount = 0;
  private windowStart = Date.now();

  constructor(
    private readonly maxRetryRatio: number = 0.1, // 10%
    private readonly windowMs: number = 10000,    // 10 seconds
    private readonly minRetriesPerSecond: number = 10, // Minimum guarantee
  ) {}

  canRetry(): boolean {
    this.maybeResetWindow();

    // OK if below the minimum guarantee
    const windowSeconds = (Date.now() - this.windowStart) / 1000;
    const minRetries = this.minRetriesPerSecond * windowSeconds;
    if (this.retryCount < minRetries) {
      return true;
    }

    // Check retry ratio
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

// Deadline propagation
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

  // Create a sub-deadline for child requests
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

// Usage example: Retry with deadline
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

### 6.2 gRPC Retry Strategy

```
gRPC Retry Policy:
  → gRPC allows retry policies to be declaratively defined in service configuration

  Retryable status codes:
    UNAVAILABLE  — Service unavailable (transient)
    DEADLINE_EXCEEDED — Deadline exceeded
    RESOURCE_EXHAUSTED — Resource exhaustion
    ABORTED — Transaction conflict

  Non-retryable:
    INVALID_ARGUMENT — Invalid argument
    NOT_FOUND — Resource does not exist
    PERMISSION_DENIED — No permission
    UNAUTHENTICATED — Not authenticated
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

## 7. Testing Retry Strategies

### 7.1 Unit Tests

```typescript
describe('retryWithBackoff', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('does not retry on success', async () => {
    const fn = jest.fn().mockResolvedValue('success');

    const result = await retryWithBackoff(fn, { maxRetries: 3 });

    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('succeeds after transient failures', async () => {
    const fn = jest.fn()
      .mockRejectedValueOnce(new Error('transient'))
      .mockRejectedValueOnce(new Error('transient'))
      .mockResolvedValue('success');

    const promise = retryWithBackoff(fn, {
      maxRetries: 3,
      baseDelayMs: 100,
      jitterStrategy: 'equal', // Jitter that is more predictable for testing
    });

    // Wait for first retry
    await jest.advanceTimersByTimeAsync(200);
    // Wait for second retry
    await jest.advanceTimersByTimeAsync(400);

    const result = await promise;
    expect(result).toBe('success');
    expect(fn).toHaveBeenCalledTimes(3);
  });

  test('throws error after exceeding max retries', async () => {
    const error = new Error('persistent failure');
    const fn = jest.fn().mockRejectedValue(error);

    const promise = retryWithBackoff(fn, { maxRetries: 2, baseDelayMs: 100 });

    await jest.advanceTimersByTimeAsync(100);
    await jest.advanceTimersByTimeAsync(200);

    await expect(promise).rejects.toThrow('persistent failure');
    expect(fn).toHaveBeenCalledTimes(3); // Initial + 2 retries
  });

  test('does not retry when shouldRetry returns false', async () => {
    const fn = jest.fn().mockRejectedValue(new HttpError(404));

    await expect(
      retryWithBackoff(fn, {
        maxRetries: 3,
        shouldRetry: (err) => !(err instanceof HttpError && err.statusCode === 404),
      })
    ).rejects.toThrow();

    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('onRetry callback is called correctly', async () => {
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
  test('transitions to Open when threshold is reached', async () => {
    const breaker = new CircuitBreaker({
      failureThreshold: 3,
      successThreshold: 1,
      resetTimeoutMs: 1000,
      halfOpenMaxConcurrent: 1,
      monitorWindowMs: 60000,
    });

    const failingFn = () => Promise.reject(new Error('fail'));

    // 3 failures
    for (let i = 0; i < 3; i++) {
      await expect(breaker.execute(failingFn)).rejects.toThrow();
    }

    // 4th call gets CircuitOpenError
    await expect(breaker.execute(failingFn)).rejects.toThrow(CircuitOpenError);
    expect(breaker.currentState).toBe('open');
  });

  test('transitions to Half-Open after reset time', async () => {
    jest.useFakeTimers();

    const breaker = new CircuitBreaker({
      failureThreshold: 2,
      successThreshold: 1,
      resetTimeoutMs: 5000,
      halfOpenMaxConcurrent: 1,
      monitorWindowMs: 60000,
    });

    const failingFn = () => Promise.reject(new Error('fail'));

    // Transition to Open
    await expect(breaker.execute(failingFn)).rejects.toThrow();
    await expect(breaker.execute(failingFn)).rejects.toThrow();
    expect(breaker.currentState).toBe('open');

    // Wait 5 seconds
    jest.advanceTimersByTime(5001);

    // Next request transitions to Half-Open
    const successFn = () => Promise.resolve('ok');
    const result = await breaker.execute(successFn);
    expect(result).toBe('ok');
    expect(breaker.currentState).toBe('closed');

    jest.useRealTimers();
  });
});
```

### 7.2 Integration Tests

```typescript
// Integration tests using Nock
import nock from 'nock';

describe('ResilientHttpClient Integration', () => {
  afterEach(() => {
    nock.cleanAll();
  });

  test('succeeds after transient 503 errors via retry', async () => {
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

  test('respects the Retry-After header', async () => {
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

## 8. Major Libraries and Frameworks

### 8.1 Retry Libraries by Language

```
TypeScript/JavaScript:
  - p-retry: Promise-based retry
  - cockatiel: Circuit breaker + retry + bulkhead
  - axios-retry: Retry plugin for Axios
  - got: HTTP client (retry built-in)

Python:
  - tenacity: General-purpose retry library
  - aiohttp-retry: Retry for aiohttp
  - stamina: Modern retry library
  - pybreaker: Circuit breaker

Go:
  - cenkalti/backoff: Exponential backoff
  - sony/gobreaker: Circuit breaker
  - avast/retry-go: Retry library
  - hashicorp/go-retryablehttp: HTTP client

Java/Kotlin:
  - resilience4j: Circuit breaker + retry + bulkhead
  - Spring Retry: For the Spring framework
  - Failsafe: Retry + circuit breaker

Rust:
  - backon: Async retry
  - reqwest-retry: Retry middleware for reqwest
```

### 8.2 Library Usage Examples

```typescript
// cockatiel usage example
import {
  retry,
  handleAll,
  ExponentialBackoff,
  CircuitBreakerPolicy,
  ConsecutiveBreaker,
  wrap,
  bulkhead,
} from 'cockatiel';

// Retry policy
const retryPolicy = retry(handleAll, {
  maxAttempts: 3,
  backoff: new ExponentialBackoff({
    initialDelay: 1000,
    maxDelay: 30000,
  }),
});

// Circuit breaker policy
const circuitBreakerPolicy = new CircuitBreakerPolicy(handleAll, {
  halfOpenAfter: 30000,
  breaker: new ConsecutiveBreaker(5),
});

// Bulkhead policy
const bulkheadPolicy = bulkhead(10, 50);

// Combine policies
const policy = wrap(bulkheadPolicy, circuitBreakerPolicy, retryPolicy);

// Usage
const result = await policy.execute(() => fetch('https://api.example.com/data'));
```

```python
# tenacity usage example
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


# Custom retry condition
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(ConnectionError),
)
async def create_order(order_data: dict) -> dict:
    return await api_client.post("/orders", json=order_data)
```

```go
// go-retryablehttp usage example
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

	// Custom retry condition
	client.CheckRetry = func(ctx context.Context, resp *http.Response, err error) (bool, error) {
		if err != nil {
			return true, nil // Retry on network errors
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

## 9. Operational Best Practices

### 9.1 Metrics and Monitoring

```
Important metrics for retry:
  1. Retry rate
     → Ratio of retries to total requests
     → If high, there may be issues with a downstream service

  2. Retry success rate
     → Percentage that ultimately succeeded via retry
     → If low, retries are adding unnecessary load

  3. Circuit breaker state transition count
     → Frequency of transitioning to Open
     → Frequency of transitioning from Half-Open back to Open

  4. Average retry count
     → Average number of retries before success
     → An increasing trend indicates a worsening problem

  5. Dead letter queue size
     → Number of jobs where all retries failed
     → Alert if increasing
```

```typescript
// Prometheus metrics collection example
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

// Retry with metrics
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

### 9.2 Configuration Tuning

```
Retry configuration guidelines:

  External API calls:
    maxRetries: 3
    baseDelay: 1000ms
    maxDelay: 30000ms
    jitter: full

  Database connections:
    maxRetries: 5
    baseDelay: 500ms
    maxDelay: 10000ms
    jitter: equal

  Message queues:
    maxRetries: 10
    baseDelay: 1000ms
    maxDelay: 60000ms
    jitter: decorrelated

  Circuit breaker:
    failureThreshold: 5-10
    resetTimeout: 15-60 seconds
    halfOpenMaxConcurrent: 1-3

  Bulkhead:
    maxConcurrent: Adjust according to service capacity
    maxQueue: 2-5 times maxConcurrent

  Notes:
    → Retry count × backoff time < request timeout
    → Consider upstream timeouts (deadline propagation)
    → Validate tuning with load tests
```

---


## FAQ

### Q1: What is the most important takeaway when learning this topic?

Gaining hands-on experience is the most important thing. Understanding deepens not just through theory, but by actually writing and running code.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in real-world work?

The knowledge covered in this topic is frequently used in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Strategy | Purpose | Use Case |
|----------|---------|----------|
| Fixed interval retry | Simple retry | Minor transient failures |
| Exponential backoff | Gradually reduce load | API limits, server failures |
| Jitter | Distribute simultaneous retries | Multi-client environments |
| Circuit breaker | Prevent cascading failures | Microservices |
| Bulkhead | Resource isolation | Multi-service dependencies |
| Retry budget | Prevent retry storms | Distributed systems |
| Deadline propagation | Control timeout chains | Multi-hop calls |
| Idempotency key | Safe retry | POST/PATCH operations |

---

## Recommended Next Guides

---

## References
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
