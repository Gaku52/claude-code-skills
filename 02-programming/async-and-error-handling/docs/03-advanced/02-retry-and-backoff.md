# リトライ戦略

> 一時的な障害は避けられない。指数バックオフ、ジッター、サーキットブレーカーなど、信頼性の高いリトライ戦略を設計する。

## この章で学ぶこと

- [ ] リトライすべきエラーとすべきでないエラーを区別する
- [ ] 指数バックオフとジッターの仕組みを理解する
- [ ] サーキットブレーカーパターンを把握する

---

## 1. リトライの基本

```
リトライすべきエラー（一時的）:
  ✓ HTTP 429（Too Many Requests）
  ✓ HTTP 503（Service Unavailable）
  ✓ ネットワークタイムアウト
  ✓ DBコネクションプール枯渇
  ✓ DNS一時障害

リトライすべきでないエラー（恒久的）:
  ✗ HTTP 400（Bad Request）— リクエストが間違い
  ✗ HTTP 401（Unauthorized）— 認証エラー
  ✗ HTTP 404（Not Found）— リソースが存在しない
  ✗ バリデーションエラー
```

---

## 2. 指数バックオフ + ジッター

```
指数バックオフ（Exponential Backoff）:
  リトライ1: 1秒後
  リトライ2: 2秒後
  リトライ3: 4秒後
  リトライ4: 8秒後
  → wait = base × 2^(attempt - 1)

ジッター（Jitter）:
  → ランダムな遅延を追加
  → 複数クライアントの同時リトライを避ける（thundering herd）

  ジッターなし: 全クライアントが同時にリトライ → サーバー再過負荷
  ジッターあり: リトライがばらける → サーバー負荷が分散
```

```typescript
// 指数バックオフ + ジッター
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    baseDelayMs?: number;
    maxDelayMs?: number;
    shouldRetry?: (error: Error) => boolean;
  } = {},
): Promise<T> {
  const {
    maxRetries = 3,
    baseDelayMs = 1000,
    maxDelayMs = 30000,
    shouldRetry = () => true,
  } = options;

  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === maxRetries || !shouldRetry(lastError)) {
        throw lastError;
      }

      // 指数バックオフ + フルジッター
      const exponentialDelay = baseDelayMs * Math.pow(2, attempt);
      const jitteredDelay = Math.random() * Math.min(exponentialDelay, maxDelayMs);

      console.log(`Retry ${attempt + 1}/${maxRetries} after ${Math.round(jitteredDelay)}ms`);
      await new Promise(resolve => setTimeout(resolve, jitteredDelay));
    }
  }

  throw lastError!;
}

// 使用
const data = await retryWithBackoff(
  () => fetch('/api/data').then(r => {
    if (!r.ok) throw new HttpError(r.status);
    return r.json();
  }),
  {
    maxRetries: 3,
    shouldRetry: (err) => {
      if (err instanceof HttpError) {
        return [429, 502, 503, 504].includes(err.status);
      }
      return true; // ネットワークエラーはリトライ
    },
  },
);
```

---

## 3. サーキットブレーカー

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
```

```typescript
class CircuitBreaker {
  private state: "closed" | "open" | "half-open" = "closed";
  private failureCount = 0;
  private lastFailureTime = 0;

  constructor(
    private readonly threshold: number = 5,
    private readonly resetTimeMs: number = 30000,
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === "open") {
      if (Date.now() - this.lastFailureTime > this.resetTimeMs) {
        this.state = "half-open";
      } else {
        throw new CircuitOpenError("Circuit breaker is open");
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failureCount = 0;
    this.state = "closed";
  }

  private onFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    if (this.failureCount >= this.threshold) {
      this.state = "open";
    }
  }
}
```

---

## 4. 実践: HTTPクライアント

```typescript
// リトライ + サーキットブレーカー + タイムアウトを組み合わせ
class ResilientHttpClient {
  private breaker = new CircuitBreaker(5, 30000);

  async request(url: string, options?: RequestInit): Promise<Response> {
    return this.breaker.execute(() =>
      retryWithBackoff(
        () => fetchWithTimeout(url, { ...options, timeoutMs: 5000 }),
        { maxRetries: 3, shouldRetry: isRetryableError },
      )
    );
  }
}
```

---

## まとめ

| 戦略 | 目的 | 適用場面 |
|------|------|---------|
| 固定間隔リトライ | 単純な再試行 | 軽微な一時障害 |
| 指数バックオフ | 負荷を段階的に下げる | API制限、サーバー障害 |
| ジッター | 同時リトライを分散 | 多クライアント環境 |
| サーキットブレーカー | 連鎖障害の防止 | マイクロサービス |

---

## 次に読むべきガイド
→ [[03-structured-concurrency.md]] — 構造化並行性

---

## 参考文献
1. AWS Architecture Blog. "Exponential Backoff and Jitter." 2015.
2. Nygard, M. "Release It!" Pragmatic Bookshelf, 2018.
