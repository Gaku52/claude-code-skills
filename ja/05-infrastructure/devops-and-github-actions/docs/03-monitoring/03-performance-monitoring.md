# パフォーマンス監視

> APM、RUM、Core Web Vitals を活用してバックエンドとフロントエンドの両面からパフォーマンスを計測し、ユーザー体験を継続的に改善する

## この章で学ぶこと

1. **APM (Application Performance Monitoring)** — バックエンドのレイテンシ、スループット、エラー率のリアルタイム監視
2. **RUM (Real User Monitoring)** — 実際のユーザーが体験するフロントエンドパフォーマンスの計測
3. **Core Web Vitals** — Google が定義する UX 指標（LCP、INP、CLS）の計測と改善
4. **Synthetic Monitoring (合成監視)** — 定期的なシナリオ実行による継続的なパフォーマンス計測
5. **パフォーマンス最適化** — ボトルネック特定から改善実施までの体系的アプローチ


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [アラート戦略](./02-alerting.md) の内容を理解していること

---

## 1. パフォーマンス監視の全体像

```
┌──────────────────────────────────────────────────────────┐
│             パフォーマンス監視の全体像                       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ユーザー側 (フロントエンド)       サーバー側 (バックエンド) │
│  ┌─────────────────────┐        ┌─────────────────────┐ │
│  │  RUM                │        │  APM                │ │
│  │  ┌───────────────┐  │        │  ┌───────────────┐  │ │
│  │  │ Core Web      │  │        │  │ レイテンシ     │  │ │
│  │  │ Vitals        │  │        │  │ (p50/p95/p99) │  │ │
│  │  │ - LCP         │  │  HTTP  │  │               │  │ │
│  │  │ - INP         │  │◄──────►│  │ スループット   │  │ │
│  │  │ - CLS         │  │        │  │ (req/sec)     │  │ │
│  │  └───────────────┘  │        │  │               │  │ │
│  │  ┌───────────────┐  │        │  │ エラー率      │  │ │
│  │  │ Navigation    │  │        │  │ (5xx/total)   │  │ │
│  │  │ Timing        │  │        │  └───────────────┘  │ │
│  │  │ Resource      │  │        │  ┌───────────────┐  │ │
│  │  │ Timing        │  │        │  │ DB クエリ     │  │ │
│  │  └───────────────┘  │        │  │ 外部API呼出   │  │ │
│  └─────────────────────┘        │  │ キャッシュHit率│  │ │
│                                  │  └───────────────┘  │ │
│                                  └─────────────────────┘ │
│  Synthetic Monitoring (合成監視)                          │
│  ┌──────────────────────────────────────────┐           │
│  │ 定期的にシナリオを実行してパフォーマンスを計測  │           │
│  │ (Lighthouse CI, Checkly, Datadog Synthetics)│           │
│  └──────────────────────────────────────────┘           │
└──────────────────────────────────────────────────────────┘
```

### 1.1 RED メソッドと USE メソッド

```
パフォーマンス監視の2つのフレームワーク:

RED メソッド (サービス指向 — マイクロサービスに最適):
┌──────────────────────────────────────────┐
│ R — Rate (リクエストレート)               │
│     1秒あたりのリクエスト数               │
│                                          │
│ E — Errors (エラーレート)                │
│     失敗したリクエストの割合              │
│                                          │
│ D — Duration (レイテンシ)                │
│     リクエストの処理時間 (p50/p95/p99)    │
└──────────────────────────────────────────┘

USE メソッド (リソース指向 — インフラ監視に最適):
┌──────────────────────────────────────────┐
│ U — Utilization (使用率)                 │
│     リソースがビジー状態の割合            │
│                                          │
│ S — Saturation (飽和度)                  │
│     リソースの待ちキュー長                │
│                                          │
│ E — Errors (エラー)                      │
│     エラーイベントの数                    │
└──────────────────────────────────────────┘

USE メソッドの適用例:
┌──────────┬──────────────┬──────────────┬──────────────┐
│ リソース │ Utilization  │ Saturation   │ Errors       │
├──────────┼──────────────┼──────────────┼──────────────┤
│ CPU      │ CPU 使用率   │ Run Queue    │ Machine Check│
│ メモリ   │ メモリ使用率 │ Swap 使用量  │ OOM Kill     │
│ ディスク │ ディスクI/O  │ I/O Wait     │ I/O Errors   │
│ ネットワーク│ 帯域使用率│ Drop/Overflow│ CRC Errors   │
└──────────┴──────────────┴──────────────┴──────────────┘
```

### 1.2 パフォーマンスバジェットの設計

```
パフォーマンスバジェットの階層:

  ┌─────────────────────────────────────────────┐
  │ ビジネス目標                                 │
  │ 「ページ読み込み1秒遅延で売上7%減少」        │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │ ユーザー体験目標                             │
  │ LCP ≤ 2.5s, INP ≤ 200ms, CLS ≤ 0.1        │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │ 技術バジェット                               │
  │ JS Bundle ≤ 300KB, Total ≤ 500KB           │
  │ API Latency p95 ≤ 200ms                    │
  │ 画像合計 ≤ 200KB                            │
  │ フォント ≤ 100KB                            │
  └────────────────────┬────────────────────────┘
                       │
  ┌────────────────────▼────────────────────────┐
  │ CI/CD 強制                                   │
  │ バジェット超過で PR ブロック or 警告          │
  └─────────────────────────────────────────────┘
```

---

## 2. APM — バックエンドパフォーマンス監視

### 2.1 Express APM ミドルウェア

```typescript
// apm-middleware.ts — Express 用 APM ミドルウェア
import { Request, Response, NextFunction } from 'express';
import { metrics } from '@opentelemetry/api';
import { trace, SpanStatusCode } from '@opentelemetry/api';

const meter = metrics.getMeter('http-server');
const tracer = trace.getTracer('http-server');

// ヒストグラム: レイテンシ分布の計測
const httpDuration = meter.createHistogram('http.server.duration', {
  description: 'HTTP リクエストの処理時間',
  unit: 'ms',
  advice: {
    explicitBucketBoundaries: [5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
  },
});

// カウンター: リクエスト数
const httpRequests = meter.createCounter('http.server.requests', {
  description: 'HTTP リクエストの総数',
});

// ゲージ: 同時接続数
const activeRequests = meter.createUpDownCounter('http.server.active_requests', {
  description: '処理中のリクエスト数',
});

export function apmMiddleware(req: Request, res: Response, next: NextFunction) {
  const startTime = performance.now();
  activeRequests.add(1);

  // レスポンス完了時にメトリクスを記録
  res.on('finish', () => {
    const duration = performance.now() - startTime;
    const labels = {
      method: req.method,
      route: req.route?.path ?? req.path,
      status: String(res.statusCode),
      status_class: `${Math.floor(res.statusCode / 100)}xx`,
    };

    httpDuration.record(duration, labels);
    httpRequests.add(1, labels);
    activeRequests.add(-1);
  });

  next();
}

// Slow Query 検出ミドルウェア
export function slowQueryDetector(thresholdMs: number = 1000) {
  return (req: Request, res: Response, next: NextFunction) => {
    const start = performance.now();

    res.on('finish', () => {
      const duration = performance.now() - start;
      if (duration > thresholdMs) {
        console.warn({
          event: 'slow_request',
          method: req.method,
          path: req.path,
          duration: Math.round(duration),
          threshold: thresholdMs,
          statusCode: res.statusCode,
        });
      }
    });

    next();
  };
}
```

### 2.2 データベースクエリの監視

```typescript
// db-query-monitor.ts — データベースクエリの監視
import { trace } from '@opentelemetry/api';
import { metrics } from '@opentelemetry/api';

const meter = metrics.getMeter('database');
const tracer = trace.getTracer('database');

const queryDuration = meter.createHistogram('db.query.duration', {
  description: 'データベースクエリの実行時間',
  unit: 'ms',
});

const queryCounter = meter.createCounter('db.query.count', {
  description: 'データベースクエリの実行回数',
});

const slowQueryCounter = meter.createCounter('db.query.slow', {
  description: 'スロークエリの数',
});

// N+1 クエリ検出
class QueryMonitor {
  private queryCounts = new Map<string, number>();
  private readonly threshold = 10; // 同一パターンが10回以上で警告
  private readonly slowQueryThresholdMs = 100; // 100ms以上でスロークエリ

  trackQuery(sql: string, duration: number): void {
    const pattern = this.normalizeQuery(sql);

    queryDuration.record(duration, { query_pattern: pattern });
    queryCounter.add(1, { query_pattern: pattern });

    // スロークエリの記録
    if (duration > this.slowQueryThresholdMs) {
      slowQueryCounter.add(1, { query_pattern: pattern });
      console.warn({
        event: 'slow_query',
        pattern,
        duration,
        threshold: this.slowQueryThresholdMs,
      });
    }

    // N+1 検出
    const count = (this.queryCounts.get(pattern) ?? 0) + 1;
    this.queryCounts.set(pattern, count);

    if (count === this.threshold) {
      console.warn({
        event: 'n_plus_one_detected',
        pattern,
        count,
        message: `同一パターンのクエリが${count}回実行されました (N+1の疑い)`,
      });
    }
  }

  private normalizeQuery(sql: string): string {
    return sql
      .replace(/\d+/g, '?')           // 数値をプレースホルダに
      .replace(/'[^']*'/g, "'?'")      // 文字列をプレースホルダに
      .replace(/\s+/g, ' ')           // 空白を正規化
      .trim();
  }

  reset(): void {
    this.queryCounts.clear();
  }
}

export const queryMonitor = new QueryMonitor();
```

### 2.3 外部 API 呼び出しの監視

```typescript
// external-api-monitor.ts — 外部 API 呼び出しの監視
import { trace, SpanStatusCode, context, propagation } from '@opentelemetry/api';
import { metrics } from '@opentelemetry/api';

const meter = metrics.getMeter('external-api');
const tracer = trace.getTracer('external-api');

const apiDuration = meter.createHistogram('external_api.duration', {
  description: '外部 API 呼び出しのレイテンシ',
  unit: 'ms',
});

const apiErrors = meter.createCounter('external_api.errors', {
  description: '外部 API 呼び出しのエラー数',
});

const circuitBreakerState = meter.createObservableGauge(
  'external_api.circuit_breaker.state',
  { description: 'Circuit Breaker の状態 (0=closed, 1=open, 2=half-open)' }
);

// 計測付き HTTP クライアント
async function instrumentedFetch(
  url: string,
  options: RequestInit = {},
  provider: string = 'unknown'
): Promise<Response> {
  const parsedUrl = new URL(url);
  const labels = {
    provider,
    host: parsedUrl.host,
    method: options.method ?? 'GET',
    path: parsedUrl.pathname,
  };

  return tracer.startActiveSpan(
    `HTTP ${labels.method} ${labels.host}${labels.path}`,
    async (span) => {
      const startTime = performance.now();

      // トレースコンテキストの伝播
      const headers = new Headers(options.headers);
      propagation.inject(context.active(), headers, {
        set: (carrier, key, value) => carrier.set(key, value),
      });

      try {
        const response = await fetch(url, {
          ...options,
          headers,
          signal: AbortSignal.timeout(30000), // 30秒タイムアウト
        });

        const duration = performance.now() - startTime;
        apiDuration.record(duration, {
          ...labels,
          status: String(response.status),
        });

        span.setAttributes({
          'http.status_code': response.status,
          'http.url': url,
          'http.method': labels.method,
          'external_api.duration_ms': Math.round(duration),
        });

        if (!response.ok) {
          apiErrors.add(1, { ...labels, status: String(response.status) });
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: `HTTP ${response.status}`,
          });
        }

        return response;
      } catch (error) {
        const duration = performance.now() - startTime;
        apiDuration.record(duration, { ...labels, status: 'error' });
        apiErrors.add(1, { ...labels, status: 'error' });

        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: (error as Error).message,
        });
        span.recordException(error as Error);

        throw error;
      } finally {
        span.end();
      }
    }
  );
}

// Circuit Breaker パターン
class CircuitBreaker {
  private failures = 0;
  private lastFailure = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';

  constructor(
    private readonly name: string,
    private readonly failureThreshold: number = 5,
    private readonly resetTimeoutMs: number = 30000,
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailure > this.resetTimeoutMs) {
        this.state = 'half-open';
      } else {
        throw new Error(`Circuit breaker is open for ${this.name}`);
      }
    }

    try {
      const result = await fn();

      if (this.state === 'half-open') {
        this.state = 'closed';
        this.failures = 0;
      }

      return result;
    } catch (error) {
      this.failures++;
      this.lastFailure = Date.now();

      if (this.failures >= this.failureThreshold) {
        this.state = 'open';
        console.warn({
          event: 'circuit_breaker_opened',
          name: this.name,
          failures: this.failures,
        });
      }

      throw error;
    }
  }

  getState(): number {
    switch (this.state) {
      case 'closed': return 0;
      case 'open': return 1;
      case 'half-open': return 2;
    }
  }
}
```

### 2.4 キャッシュパフォーマンスの監視

```typescript
// cache-monitor.ts — キャッシュパフォーマンスの監視
import { metrics } from '@opentelemetry/api';

const meter = metrics.getMeter('cache');

const cacheHits = meter.createCounter('cache.hits', {
  description: 'キャッシュヒット数',
});

const cacheMisses = meter.createCounter('cache.misses', {
  description: 'キャッシュミス数',
});

const cacheDuration = meter.createHistogram('cache.operation.duration', {
  description: 'キャッシュ操作のレイテンシ',
  unit: 'ms',
});

const cacheSize = meter.createObservableGauge('cache.size', {
  description: 'キャッシュのエントリ数',
});

class MonitoredCache<T> {
  private cache = new Map<string, { value: T; expiry: number }>();

  constructor(private readonly name: string) {
    // キャッシュサイズの定期報告
    cacheSize.addCallback((result) => {
      result.observe(this.cache.size, { cache: this.name });
    });
  }

  async get(key: string): Promise<T | undefined> {
    const start = performance.now();
    const entry = this.cache.get(key);
    const duration = performance.now() - start;

    if (entry && entry.expiry > Date.now()) {
      cacheHits.add(1, { cache: this.name });
      cacheDuration.record(duration, { cache: this.name, operation: 'get', result: 'hit' });
      return entry.value;
    }

    cacheMisses.add(1, { cache: this.name });
    cacheDuration.record(duration, { cache: this.name, operation: 'get', result: 'miss' });

    if (entry) {
      this.cache.delete(key); // 期限切れエントリの削除
    }

    return undefined;
  }

  async set(key: string, value: T, ttlMs: number): Promise<void> {
    const start = performance.now();
    this.cache.set(key, { value, expiry: Date.now() + ttlMs });
    const duration = performance.now() - start;

    cacheDuration.record(duration, { cache: this.name, operation: 'set' });
  }

  // キャッシュヒット率 (PromQL で計算)
  // rate(cache_hits{cache="products"}[5m])
  // / (rate(cache_hits{cache="products"}[5m]) + rate(cache_misses{cache="products"}[5m]))
}

// Grafana ダッシュボード用 PromQL クエリ集
/*
# キャッシュヒット率 (%)
sum(rate(cache_hits[5m])) by (cache)
/ (sum(rate(cache_hits[5m])) by (cache) + sum(rate(cache_misses[5m])) by (cache))
* 100

# キャッシュ操作のレイテンシ (p95)
histogram_quantile(0.95,
  sum(rate(cache_operation_duration_bucket[5m])) by (le, cache, operation)
)

# キャッシュサイズの推移
cache_size
*/
```

### 2.5 Grafana ダッシュボード用 PromQL (APM)

```promql
# --- RED メトリクス (サービス別) ---

# Rate: リクエストレート
sum(rate(http_server_requests_total[5m])) by (service)

# Errors: エラーレート (%)
sum(rate(http_server_requests_total{status_class="5xx"}[5m])) by (service)
/
sum(rate(http_server_requests_total[5m])) by (service)
* 100

# Duration: p50/p95/p99 レイテンシ
histogram_quantile(0.5,
  sum(rate(http_server_duration_bucket[5m])) by (service, le)
)

histogram_quantile(0.95,
  sum(rate(http_server_duration_bucket[5m])) by (service, le)
)

histogram_quantile(0.99,
  sum(rate(http_server_duration_bucket[5m])) by (service, le)
)

# --- エンドポイント別の詳細 ---

# 最も遅いエンドポイント Top 10
topk(10,
  histogram_quantile(0.95,
    sum(rate(http_server_duration_bucket[5m])) by (route, le)
  )
)

# 最もリクエスト数が多いエンドポイント Top 10
topk(10,
  sum(rate(http_server_requests_total[5m])) by (route)
)

# エラーが多いエンドポイント Top 10
topk(10,
  sum(rate(http_server_requests_total{status_class="5xx"}[5m])) by (route)
)

# --- DB クエリパフォーマンス ---

# スロークエリの発生率
sum(rate(db_query_slow_total[5m])) by (query_pattern)

# クエリレイテンシ p95
histogram_quantile(0.95,
  sum(rate(db_query_duration_bucket[5m])) by (le, query_pattern)
)

# N+1 クエリの検出回数
sum(increase(n_plus_one_detected_total[1h])) by (query_pattern)

# --- 外部 API パフォーマンス ---

# 外部 API 呼び出しのレイテンシ (プロバイダ別)
histogram_quantile(0.95,
  sum(rate(external_api_duration_bucket[5m])) by (le, provider)
)

# 外部 API のエラーレート
sum(rate(external_api_errors_total[5m])) by (provider)
/
sum(rate(external_api_duration_count[5m])) by (provider)
* 100

# --- キャッシュパフォーマンス ---

# キャッシュヒット率
sum(rate(cache_hits_total[5m])) by (cache)
/ (sum(rate(cache_hits_total[5m])) by (cache) + sum(rate(cache_misses_total[5m])) by (cache))
* 100

# キャッシュ操作のレイテンシ
histogram_quantile(0.95,
  sum(rate(cache_operation_duration_bucket[5m])) by (le, cache, operation)
)
```

---

## 3. RUM — フロントエンドパフォーマンス監視

### 3.1 RUM データ収集の実装

```typescript
// rum-collector.ts — Real User Monitoring の実装
interface PerformanceData {
  // Navigation Timing
  dns: number;
  tcp: number;
  tls: number;
  ttfb: number;          // Time to First Byte
  domContentLoaded: number;
  load: number;

  // Core Web Vitals
  lcp: number | null;     // Largest Contentful Paint
  inp: number | null;     // Interaction to Next Paint
  cls: number | null;     // Cumulative Layout Shift

  // コンテキスト
  url: string;
  userAgent: string;
  connectionType: string;
  timestamp: number;
}

class RUMCollector {
  private data: Partial<PerformanceData> = {};

  constructor(private readonly endpoint: string) {
    this.collectNavigationTiming();
    this.collectWebVitals();

    // ページ離脱時に送信
    window.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.send();
      }
    });
  }

  private collectNavigationTiming(): void {
    window.addEventListener('load', () => {
      setTimeout(() => {
        const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
        if (!nav) return;

        this.data.dns = nav.domainLookupEnd - nav.domainLookupStart;
        this.data.tcp = nav.connectEnd - nav.connectStart;
        this.data.tls = nav.secureConnectionStart > 0
          ? nav.connectEnd - nav.secureConnectionStart : 0;
        this.data.ttfb = nav.responseStart - nav.requestStart;
        this.data.domContentLoaded = nav.domContentLoadedEventEnd - nav.startTime;
        this.data.load = nav.loadEventEnd - nav.startTime;
      }, 0);
    });
  }

  private collectWebVitals(): void {
    // LCP (Largest Contentful Paint)
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.data.lcp = lastEntry.startTime;
    }).observe({ type: 'largest-contentful-paint', buffered: true });

    // CLS (Cumulative Layout Shift)
    let clsValue = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (!(entry as any).hadRecentInput) {
          clsValue += (entry as any).value;
        }
      }
      this.data.cls = clsValue;
    }).observe({ type: 'layout-shift', buffered: true });

    // INP (Interaction to Next Paint)
    let maxINP = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const duration = entry.duration;
        if (duration > maxINP) {
          maxINP = duration;
          this.data.inp = duration;
        }
      }
    }).observe({ type: 'event', buffered: true });
  }

  private send(): void {
    const payload: PerformanceData = {
      ...this.data as PerformanceData,
      url: window.location.href,
      userAgent: navigator.userAgent,
      connectionType: (navigator as any).connection?.effectiveType ?? 'unknown',
      timestamp: Date.now(),
    };

    // Beacon API で確実に送信 (ページ離脱時も)
    navigator.sendBeacon(
      this.endpoint,
      JSON.stringify(payload)
    );
  }
}

// 使用
new RUMCollector('/api/rum/collect');
```

### 3.2 web-vitals ライブラリの活用

```typescript
// web-vitals-reporter.ts — web-vitals ライブラリを使った計測
import { onLCP, onINP, onCLS, onFCP, onTTFB, Metric } from 'web-vitals';

interface VitalsReport {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  delta: number;
  id: string;
  navigationType: string;
  url: string;
  timestamp: number;
}

class WebVitalsReporter {
  private reports: VitalsReport[] = [];
  private readonly batchSize = 10;
  private readonly flushIntervalMs = 5000;

  constructor(private readonly endpoint: string) {
    this.startAutoFlush();
    this.registerMetrics();
  }

  private registerMetrics(): void {
    const reportCallback = (metric: Metric) => {
      const report: VitalsReport = {
        name: metric.name,
        value: metric.value,
        rating: metric.rating,
        delta: metric.delta,
        id: metric.id,
        navigationType: metric.navigationType,
        url: window.location.href,
        timestamp: Date.now(),
      };

      this.reports.push(report);

      // コンソールにも出力 (開発用)
      if (process.env.NODE_ENV === 'development') {
        const color = metric.rating === 'good'
          ? 'green'
          : metric.rating === 'needs-improvement'
            ? 'orange'
            : 'red';
        console.log(
          `%c[Web Vitals] ${metric.name}: ${metric.value.toFixed(1)} (${metric.rating})`,
          `color: ${color}; font-weight: bold;`
        );
      }

      if (this.reports.length >= this.batchSize) {
        this.flush();
      }
    };

    onLCP(reportCallback);
    onINP(reportCallback);
    onCLS(reportCallback);
    onFCP(reportCallback);
    onTTFB(reportCallback);
  }

  private flush(): void {
    if (this.reports.length === 0) return;

    const payload = [...this.reports];
    this.reports = [];

    // Beacon API で送信
    navigator.sendBeacon(
      this.endpoint,
      JSON.stringify(payload)
    );
  }

  private startAutoFlush(): void {
    setInterval(() => this.flush(), this.flushIntervalMs);

    // ページ離脱時にフラッシュ
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.flush();
      }
    });
  }
}

// 使用
new WebVitalsReporter('/api/vitals/report');
```

### 3.3 RUM データの集約 API

```typescript
// rum-api.ts — RUM データの受信と集約
import express from 'express';
import { metrics } from '@opentelemetry/api';

const app = express();
const meter = metrics.getMeter('rum');

// Web Vitals メトリクスの定義
const lcpHistogram = meter.createHistogram('web_vitals.lcp', {
  description: 'Largest Contentful Paint',
  unit: 'ms',
  advice: {
    explicitBucketBoundaries: [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000],
  },
});

const inpHistogram = meter.createHistogram('web_vitals.inp', {
  description: 'Interaction to Next Paint',
  unit: 'ms',
  advice: {
    explicitBucketBoundaries: [50, 100, 150, 200, 300, 400, 500, 750, 1000],
  },
});

const clsHistogram = meter.createHistogram('web_vitals.cls', {
  description: 'Cumulative Layout Shift',
  advice: {
    explicitBucketBoundaries: [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.5],
  },
});

const fcpHistogram = meter.createHistogram('web_vitals.fcp', {
  description: 'First Contentful Paint',
  unit: 'ms',
});

const ttfbHistogram = meter.createHistogram('web_vitals.ttfb', {
  description: 'Time to First Byte',
  unit: 'ms',
});

// RUM データ受信エンドポイント
app.post('/api/vitals/report', express.json(), (req, res) => {
  const reports = Array.isArray(req.body) ? req.body : [req.body];

  for (const report of reports) {
    const labels = {
      page: new URL(report.url).pathname,
      connection: report.connectionType ?? 'unknown',
      navigation_type: report.navigationType ?? 'navigate',
      rating: report.rating,
    };

    switch (report.name) {
      case 'LCP':
        lcpHistogram.record(report.value, labels);
        break;
      case 'INP':
        inpHistogram.record(report.value, labels);
        break;
      case 'CLS':
        clsHistogram.record(report.value, labels);
        break;
      case 'FCP':
        fcpHistogram.record(report.value, labels);
        break;
      case 'TTFB':
        ttfbHistogram.record(report.value, labels);
        break;
    }
  }

  res.status(204).end();
});

// Navigation Timing データ受信
app.post('/api/rum/collect', express.json(), (req, res) => {
  const data = req.body;
  const page = new URL(data.url).pathname;

  // TTFB
  if (data.ttfb) {
    ttfbHistogram.record(data.ttfb, { page });
  }

  // LCP
  if (data.lcp) {
    lcpHistogram.record(data.lcp, { page });
  }

  // CLS
  if (data.cls != null) {
    clsHistogram.record(data.cls, { page });
  }

  // INP
  if (data.inp) {
    inpHistogram.record(data.inp, { page });
  }

  res.status(204).end();
});
```

---

## 4. Core Web Vitals の基準と改善

### 4.1 基準値

```
Core Web Vitals の評価基準 (2024年更新):

  LCP (Largest Contentful Paint) — 読み込み速度
  ┌─────────────┬──────────────┬──────────────┐
  │  Good       │ Needs Work   │  Poor        │
  │  ≤ 2.5秒    │ ≤ 4.0秒      │ > 4.0秒      │
  │  ████████   │ ████████     │ ████████     │
  │  (緑)       │ (黄)         │ (赤)         │
  └─────────────┴──────────────┴──────────────┘

  INP (Interaction to Next Paint) — 応答性
  ┌─────────────┬──────────────┬──────────────┐
  │  Good       │ Needs Work   │  Poor        │
  │  ≤ 200ms    │ ≤ 500ms      │ > 500ms      │
  │  ████████   │ ████████     │ ████████     │
  │  (緑)       │ (黄)         │ (赤)         │
  └─────────────┴──────────────┴──────────────┘

  CLS (Cumulative Layout Shift) — 視覚的安定性
  ┌─────────────┬──────────────┬──────────────┐
  │  Good       │ Needs Work   │  Poor        │
  │  ≤ 0.1      │ ≤ 0.25       │ > 0.25       │
  │  ████████   │ ████████     │ ████████     │
  │  (緑)       │ (黄)         │ (赤)         │
  └─────────────┴──────────────┴──────────────┘
```

### 4.2 改善ガイド

```
各指標の改善チェックリスト:

LCP 改善:
┌────────────────────────────────────────────────────────┐
│ □ LCP 要素の特定 (通常は hero 画像 or 大きなテキスト)   │
│ □ サーバーレスポンスタイムの改善 (TTFB < 800ms)         │
│ □ レンダリングブロッキングリソースの排除                │
│   - CSS: critical CSS のインライン化                    │
│   - JS: defer / async 属性                             │
│ □ 画像の最適化                                         │
│   - 適切なフォーマット (WebP/AVIF)                      │
│   - srcset による適切なサイズの提供                     │
│   - fetchpriority="high" で LCP 画像を優先              │
│   - プリロード: <link rel="preload" as="image">        │
│ □ CDN の活用                                           │
│ □ SSR / SSG によるサーバーサイドレンダリング             │
└────────────────────────────────────────────────────────┘

INP 改善:
┌────────────────────────────────────────────────────────┐
│ □ 重い JavaScript 処理の分割                           │
│   - Long Task (50ms+) の特定と分割                     │
│   - requestIdleCallback / scheduler.yield()            │
│ □ メインスレッドのブロック回避                          │
│   - Web Worker への処理移譲                             │
│   - requestAnimationFrame の活用                        │
│ □ イベントハンドラの最適化                             │
│   - デバウンス / スロットル                              │
│   - パッシブイベントリスナー                            │
│ □ 不要な re-render の防止 (React)                      │
│   - React.memo, useMemo, useCallback                   │
│   - 仮想化 (react-window, react-virtuoso)              │
│ □ Third-party スクリプトの影響評価                     │
└────────────────────────────────────────────────────────┘

CLS 改善:
┌────────────────────────────────────────────────────────┐
│ □ 画像・動画に明示的なサイズ指定                       │
│   - width/height 属性 or aspect-ratio CSS              │
│ □ Web フォントのフラッシュ防止                         │
│   - font-display: swap + preload                       │
│   - size-adjust でフォールバック調整                    │
│ □ 動的コンテンツの挿入位置                             │
│   - 広告やバナーのスペースを事前確保                   │
│   - contain-intrinsic-size の活用                       │
│ □ アニメーションの transform 使用                      │
│   - width/height アニメーション → transform: scale()    │
│   - top/left アニメーション → transform: translate()    │
│ □ レイアウトシフトの原因特定                           │
│   - DevTools Performance パネル                        │
│   - Layout Shift デバッガー                             │
└────────────────────────────────────────────────────────┘
```

---

## 5. Lighthouse CI による継続的計測

### 5.1 GitHub Actions ワークフロー

```yaml
# .github/workflows/lighthouse-ci.yml
name: Lighthouse CI

on:
  pull_request:
    branches: [main]

jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci && npm run build

      - name: Start server
        run: npm run preview &
        env:
          PORT: 3000

      - name: Wait for server
        run: npx wait-on http://localhost:3000

      - name: Run Lighthouse
        uses: treosh/lighthouse-ci-action@v11
        with:
          urls: |
            http://localhost:3000/
            http://localhost:3000/products
            http://localhost:3000/checkout
          budgetPath: ./lighthouse-budget.json
          uploadArtifacts: true
          temporaryPublicStorage: true

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('.lighthouseci/manifest.json'));

            let comment = '## Lighthouse CI Results\n\n';
            comment += '| URL | Performance | Accessibility | Best Practices | SEO |\n';
            comment += '|-----|-----------|---------------|---------------|-----|\n';

            for (const result of results) {
              const summary = JSON.parse(fs.readFileSync(result.jsonPath));
              const scores = summary.categories;

              const getEmoji = (score) => score >= 0.9 ? '🟢' : score >= 0.5 ? '🟡' : '🔴';

              comment += `| ${result.url} `;
              comment += `| ${getEmoji(scores.performance.score)} ${Math.round(scores.performance.score * 100)} `;
              comment += `| ${getEmoji(scores.accessibility.score)} ${Math.round(scores.accessibility.score * 100)} `;
              comment += `| ${getEmoji(scores['best-practices'].score)} ${Math.round(scores['best-practices'].score * 100)} `;
              comment += `| ${getEmoji(scores.seo.score)} ${Math.round(scores.seo.score * 100)} |\n`;
            }

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment,
            });
```

### 5.2 パフォーマンスバジェット

```json
[
  {
    "path": "/*",
    "timings": [
      { "metric": "interactive", "budget": 3000 },
      { "metric": "first-contentful-paint", "budget": 1500 },
      { "metric": "largest-contentful-paint", "budget": 2500 },
      { "metric": "total-blocking-time", "budget": 300 }
    ],
    "resourceSizes": [
      { "resourceType": "script", "budget": 300 },
      { "resourceType": "total", "budget": 500 },
      { "resourceType": "image", "budget": 200 },
      { "resourceType": "stylesheet", "budget": 100 },
      { "resourceType": "font", "budget": 100 },
      { "resourceType": "third-party", "budget": 150 }
    ],
    "resourceCounts": [
      { "resourceType": "script", "budget": 10 },
      { "resourceType": "total", "budget": 50 },
      { "resourceType": "third-party", "budget": 5 }
    ]
  }
]
```

### 5.3 Lighthouse CI 設定ファイル

```javascript
// lighthouserc.js — Lighthouse CI の詳細設定
module.exports = {
  ci: {
    collect: {
      url: [
        'http://localhost:3000/',
        'http://localhost:3000/products',
        'http://localhost:3000/products/1',
        'http://localhost:3000/checkout',
      ],
      numberOfRuns: 3,  // 各 URL を 3 回実行して中央値を取得
      settings: {
        preset: 'desktop',  // 'desktop' or 'mobile'
        throttling: {
          // Fast 3G シミュレーション
          rttMs: 150,
          throughputKbps: 1638.4,
          cpuSlowdownMultiplier: 4,
        },
        // Chrome フラグ
        chromeFlags: '--no-sandbox --headless',
        // 特定の監査のみ実行
        onlyCategories: ['performance', 'accessibility', 'best-practices', 'seo'],
      },
    },
    assert: {
      assertions: {
        // パフォーマンスカテゴリ
        'categories:performance': ['error', { minScore: 0.8 }],
        'categories:accessibility': ['error', { minScore: 0.9 }],
        'categories:best-practices': ['warn', { minScore: 0.9 }],
        'categories:seo': ['warn', { minScore: 0.9 }],

        // Core Web Vitals
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'total-blocking-time': ['error', { maxNumericValue: 300 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],

        // その他の重要な監査
        'first-contentful-paint': ['warn', { maxNumericValue: 1500 }],
        'speed-index': ['warn', { maxNumericValue: 3000 }],
        'interactive': ['warn', { maxNumericValue: 3000 }],

        // リソースサイズ
        'resource-summary:script:size': ['error', { maxNumericValue: 307200 }],  // 300KB
        'resource-summary:total:size': ['error', { maxNumericValue: 512000 }],    // 500KB
      },
    },
    upload: {
      target: 'temporary-public-storage',
    },
  },
};
```

---

## 6. Synthetic Monitoring (合成監視)

### 6.1 Checkly による合成監視

```typescript
// checkly.config.ts — Checkly の設定
import { defineConfig } from 'checkly';

export default defineConfig({
  projectName: 'MyApp Monitoring',
  logicalId: 'myapp-monitoring',
  repoUrl: 'https://github.com/example/myapp',
  checks: {
    activated: true,
    muted: false,
    runtimeId: '2024.02',
    frequency: 5,  // 5分ごと
    locations: ['ap-northeast-1', 'us-east-1', 'eu-west-1'],
    tags: ['production'],
    checkMatch: '**/*.check.ts',
    browserChecks: {
      frequency: 10,  // 10分ごと
      testMatch: '**/*.spec.ts',
    },
  },
});
```

```typescript
// checks/api-health.check.ts — API ヘルスチェック
import { ApiCheck, AssertionBuilder } from 'checkly/constructs';

new ApiCheck('api-health-check', {
  name: 'API Health Check',
  activated: true,
  frequency: 1,  // 1分ごと
  locations: ['ap-northeast-1'],
  request: {
    method: 'GET',
    url: 'https://api.example.com/health',
    assertions: [
      AssertionBuilder.statusCode().equals(200),
      AssertionBuilder.responseTime().lessThan(500),
      AssertionBuilder.jsonBody('$.status').equals('healthy'),
    ],
  },
  alertChannels: [
    { id: 'slack-alerts' },
    { id: 'pagerduty-critical' },
  ],
});

// checks/order-flow.spec.ts — E2E シナリオテスト
import { test, expect } from '@playwright/test';

test('注文フロー E2E', async ({ page }) => {
  // 1. 商品一覧ページにアクセス
  const startTime = Date.now();
  await page.goto('https://www.example.com/products');
  expect(Date.now() - startTime).toBeLessThan(3000);

  // 2. 商品を選択
  await page.click('[data-testid="product-card"]:first-child');
  await expect(page.locator('[data-testid="product-detail"]')).toBeVisible();

  // 3. カートに追加
  await page.click('[data-testid="add-to-cart"]');
  await expect(page.locator('[data-testid="cart-count"]')).toHaveText('1');

  // 4. チェックアウトへ
  await page.click('[data-testid="checkout-button"]');
  await expect(page).toHaveURL(/\/checkout/);

  // 5. フォーム入力
  await page.fill('[data-testid="email"]', 'test@example.com');
  await page.fill('[data-testid="card-number"]', '4242424242424242');

  // 6. 注文確定 (テスト環境のみ)
  if (process.env.CHECKLY_TEST_ENVIRONMENT === 'staging') {
    await page.click('[data-testid="place-order"]');
    await expect(page.locator('[data-testid="order-confirmation"]')).toBeVisible({ timeout: 10000 });
  }
});
```

### 6.2 Datadog Synthetics

```yaml
# datadog-synthetics.tf — Terraform で Datadog Synthetics 管理
resource "datadog_synthetics_test" "api_health" {
  name      = "API Health Check"
  type      = "api"
  subtype   = "http"
  status    = "live"
  message   = "API ヘルスチェックが失敗しました @pagerduty-critical"
  tags      = ["env:production", "service:api"]

  locations = ["aws:ap-northeast-1"]

  request_definition {
    method = "GET"
    url    = "https://api.example.com/health"
  }

  request_headers = {
    Accept = "application/json"
  }

  assertion {
    type     = "statusCode"
    operator = "is"
    target   = "200"
  }

  assertion {
    type     = "responseTime"
    operator = "lessThan"
    target   = "500"
  }

  assertion {
    type     = "body"
    operator = "validatesJSONPath"
    targetjsonpath {
      jsonpath    = "$.status"
      operator    = "is"
      targetvalue = "healthy"
    }
  }

  options_list {
    tick_every = 60  # 1分ごと
    retry {
      count    = 2
      interval = 300
    }
    monitor_options {
      renotify_interval = 120
    }
  }
}

resource "datadog_synthetics_test" "browser_checkout" {
  name      = "Checkout Flow Browser Test"
  type      = "browser"
  status    = "live"
  message   = "チェックアウトフローのテストが失敗しました @slack-alerts-warning"
  tags      = ["env:production", "service:frontend"]

  locations = ["aws:ap-northeast-1"]

  request_definition {
    method = "GET"
    url    = "https://www.example.com/products"
  }

  options_list {
    tick_every = 600  # 10分ごと
  }

  browser_step {
    name = "商品をクリック"
    type = "click"
    params {
      element = ".product-card:first-child"
    }
  }

  browser_step {
    name = "カートに追加"
    type = "click"
    params {
      element = "[data-testid='add-to-cart']"
    }
  }

  browser_step {
    name = "カート数を確認"
    type = "assertElementContent"
    params {
      element = "[data-testid='cart-count']"
      value   = "1"
    }
  }
}
```

---

## 7. バンドルサイズの監視

### 7.1 webpack-bundle-analyzer + CI

```yaml
# .github/workflows/bundle-size.yml
name: Bundle Size Check

on:
  pull_request:
    branches: [main]

jobs:
  bundle-size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm

      - run: npm ci

      - name: Build and analyze
        run: npm run build -- --stats
        env:
          ANALYZE: true

      - name: Check bundle size
        uses: andresz1/size-limit-action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          skip_step: build
```

```javascript
// .size-limit.js — バンドルサイズの上限設定
module.exports = [
  {
    name: 'Full Bundle',
    path: 'dist/**/*.js',
    limit: '300 KB',
    gzip: true,
  },
  {
    name: 'Initial JS',
    path: 'dist/assets/index-*.js',
    limit: '150 KB',
    gzip: true,
  },
  {
    name: 'Vendor Bundle',
    path: 'dist/assets/vendor-*.js',
    limit: '200 KB',
    gzip: true,
  },
  {
    name: 'CSS Bundle',
    path: 'dist/assets/*.css',
    limit: '50 KB',
    gzip: true,
  },
];
```

### 7.2 Import Cost の可視化

```typescript
// scripts/analyze-imports.ts — インポートコストの分析
import { build } from 'esbuild';
import { gzipSync } from 'zlib';

interface ImportCost {
  package: string;
  size: number;
  gzipSize: number;
}

async function analyzeImport(packageName: string): Promise<ImportCost> {
  const result = await build({
    stdin: {
      contents: `export * from '${packageName}'`,
      resolveDir: process.cwd(),
    },
    bundle: true,
    write: false,
    minify: true,
    format: 'esm',
    platform: 'browser',
    external: ['react', 'react-dom'],
  });

  const code = result.outputFiles[0].contents;
  const gzipped = gzipSync(code);

  return {
    package: packageName,
    size: code.length,
    gzipSize: gzipped.length,
  };
}

// 分析対象のパッケージ
const packages = [
  'lodash',
  'lodash-es',
  'date-fns',
  'moment',
  'dayjs',
  'axios',
  '@tanstack/react-query',
  'zod',
];

async function main() {
  console.log('Package Import Cost Analysis\n');
  console.log('Package              | Raw Size  | Gzip Size');
  console.log('---------------------|-----------|----------');

  for (const pkg of packages) {
    try {
      const cost = await analyzeImport(pkg);
      const rawKB = (cost.size / 1024).toFixed(1);
      const gzipKB = (cost.gzipSize / 1024).toFixed(1);
      console.log(`${pkg.padEnd(21)}| ${rawKB.padStart(7)} KB | ${gzipKB.padStart(7)} KB`);
    } catch {
      console.log(`${pkg.padEnd(21)}| (error)    | (error)`);
    }
  }
}

main();
```

---

## 8. 比較表

| 指標 | APM (バックエンド) | RUM (フロントエンド) | Synthetic (合成) |
|------|-------------------|---------------------|-----------------|
| 計測対象 | サーバー処理 | 実ユーザー体験 | スクリプト実行 |
| データ量 | 中 | 多い | 少ない |
| リアルタイム性 | 高い | 中 (集計後) | 定期実行 |
| 環境差異 | なし | デバイス/ネットワーク依存 | 統制環境 |
| コスト | 中 | トラフィック比例 | 実行回数比例 |
| ユースケース | API遅延/DB問題 | UX劣化検知 | リグレッション検知 |

| RUM ツール比較 | web-vitals (OSS) | Datadog RUM | New Relic Browser | Sentry |
|---------------|-----------------|-------------|-------------------|--------|
| Core Web Vitals | 対応 | 対応 | 対応 | 対応 |
| セッションリプレイ | なし | あり | あり | あり |
| エラー追跡 | なし | あり | あり | 充実 |
| 料金 | 無料 | 有料 | 有料 | 無料枠あり |
| バンドルサイズ | 極小 (1.5KB) | 中 (~30KB) | 中 (~30KB) | 中 (~20KB) |

| Synthetic ツール比較 | Checkly | Datadog Synthetics | Grafana k6 | Playwright Test |
|--------------------|---------|-------------------|-----------|----------------|
| API テスト | 対応 | 対応 | 対応 | 限定的 |
| ブラウザテスト | 対応 (Playwright) | 対応 | 限定的 | 対応 |
| 負荷テスト | 限定的 | 限定的 | 充実 | なし |
| マルチリージョン | 対応 | 対応 | Cloud のみ | なし |
| CI/CD 統合 | 充実 | 対応 | 充実 | ネイティブ |
| 料金 | $30/月〜 | 含む | OSS (Cloud有料) | 無料 |

| バンドル分析ツール | size-limit | bundlesize | webpack-bundle-analyzer | source-map-explorer |
|------------------|-----------|------------|------------------------|-------------------|
| CI 統合 | GitHub Action | GitHub Action | レポート生成 | レポート生成 |
| 差分表示 | あり | あり | なし | なし |
| 可視化 | なし | なし | Treemap | Treemap |
| 設定の柔軟性 | 高い | 中 | 高い | 低い |

---

## 9. アンチパターン

### アンチパターン 1: 平均値だけを見る

```
[悪い例]
- 「平均レスポンスタイム 200ms で問題なし」
- しかし p99 は 5秒を超えている (100リクエストに1回は5秒待ち)
- 上位顧客ほどリクエスト数が多く、遅いレスポンスに当たりやすい

[良い例]
- パーセンタイルで監視:
  p50 (中央値):  通常のユーザー体験
  p95:           多くのユーザーが体験する最悪ケース
  p99:           テールレイテンシ (SLO の対象に)
  p99.9:         極端なケース (デバッグ用)

- PromQL でパーセンタイル計算:
  histogram_quantile(0.99,
    sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
  )
```

### アンチパターン 2: パフォーマンスバジェットなしの開発

```
[悪い例]
- 「リリースしてから計測すればいい」
- バンドルサイズが 2MB を超えてから気づく
- LCP が 5秒超でも誰も検知しない
- 「このライブラリ追加したらバンドルが 500KB 増えた」を PR で指摘できない

[良い例]
- CI にパフォーマンスバジェットを組み込む:
  - JS バンドル: 300KB 以下
  - 画像合計: 200KB 以下
  - LCP: 2.5秒以下
  - INP: 200ms 以下
- バジェット超過で PR をブロック (またはコメント警告)
- Lighthouse CI で継続的にスコアを追跡
```

### アンチパターン 3: 本番環境でのみ計測

```
[悪い例]
- 開発環境では DevTools を手動で見るだけ
- ステージング環境でのパフォーマンステストがない
- 本番リリース後に初めて問題に気づく
- ロールバックを繰り返す

[良い例]
- 計測の3段階:
  1. 開発時: Lighthouse DevTools + web-vitals ログ
  2. CI/CD: Lighthouse CI + バンドルサイズチェック
  3. 本番: RUM + Synthetic Monitoring + APM
- ステージング環境での負荷テスト (k6, Artillery)
- パフォーマンスリグレッションの早期検知
```

### アンチパターン 4: サードパーティスクリプトの無計画な追加

```
[悪い例]
- Google Analytics, GTM, Intercom, Hotjar, Facebook Pixel...
  を全ページに読み込み
- 各スクリプトが 50-200KB、合計で 1MB 以上
- メインスレッドをブロックして INP が悪化
- 「マーケティングチームが追加した」で管理不在

[良い例]
- サードパーティスクリプトの棚卸し (四半期ごと)
- 各スクリプトの影響を計測:
  - バンドルサイズへの影響
  - メインスレッドブロック時間
  - LCP/INP への影響
- 遅延読み込みの活用:
  - Partytown (Web Worker で実行)
  - IntersectionObserver による遅延初期化
  - requestIdleCallback での非同期読み込み
- パフォーマンスバジェットにサードパーティ枠を設定
```

---

## 10. パフォーマンステスト (負荷テスト)

### 10.1 k6 による負荷テスト

```javascript
// k6-load-test.js — k6 負荷テストスクリプト
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// カスタムメトリクス
const errorRate = new Rate('errors');
const orderLatency = new Trend('order_latency');

// テストシナリオ
export const options = {
  scenarios: {
    // 段階的な負荷増加
    ramp_up: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // 2分で50 VU まで増加
        { duration: '5m', target: 50 },   // 5分間維持
        { duration: '2m', target: 100 },  // 2分で100 VU まで増加
        { duration: '5m', target: 100 },  // 5分間維持
        { duration: '2m', target: 200 },  // 2分で200 VU まで増加
        { duration: '5m', target: 200 },  // 5分間維持
        { duration: '3m', target: 0 },    // 3分で 0 に
      ],
    },
    // スパイクテスト
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '1m', target: 10 },
        { duration: '30s', target: 500 },  // 急激なスパイク
        { duration: '1m', target: 500 },
        { duration: '30s', target: 10 },   // 急激な減少
        { duration: '1m', target: 0 },
      ],
      startTime: '25m',  // ramp_up 完了後に開始
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],  // p95 < 500ms, p99 < 1s
    http_req_failed: ['rate<0.01'],  // エラーレート < 1%
    errors: ['rate<0.05'],  // カスタムエラーレート < 5%
    order_latency: ['p(95)<2000'],  // 注文レイテンシ p95 < 2s
  },
};

export default function () {
  // 1. 商品一覧の取得
  const productsRes = http.get('https://api.example.com/products', {
    headers: { 'Accept': 'application/json' },
  });
  check(productsRes, {
    'products: status 200': (r) => r.status === 200,
    'products: response time < 500ms': (r) => r.timings.duration < 500,
  });
  errorRate.add(productsRes.status !== 200);

  sleep(1);

  // 2. 商品詳細の取得
  const productId = Math.floor(Math.random() * 100) + 1;
  const productRes = http.get(`https://api.example.com/products/${productId}`);
  check(productRes, {
    'product: status 200': (r) => r.status === 200,
  });

  sleep(0.5);

  // 3. 注文の作成 (10% のユーザーのみ)
  if (Math.random() < 0.1) {
    const orderStart = Date.now();
    const orderRes = http.post(
      'https://api.example.com/orders',
      JSON.stringify({
        productId,
        quantity: 1,
        paymentMethod: 'credit_card',
      }),
      {
        headers: { 'Content-Type': 'application/json' },
      }
    );
    orderLatency.add(Date.now() - orderStart);

    check(orderRes, {
      'order: status 201': (r) => r.status === 201,
      'order: response time < 2s': (r) => r.timings.duration < 2000,
    });
    errorRate.add(orderRes.status !== 201);
  }

  sleep(Math.random() * 3);
}

// テスト結果の出力設定
export function handleSummary(data) {
  return {
    'stdout': textSummary(data, { indent: ' ', enableColors: true }),
    'k6-results.json': JSON.stringify(data),
  };
}
```

### 10.2 k6 CI 統合

```yaml
# .github/workflows/load-test.yml
name: Load Test

on:
  schedule:
    - cron: '0 3 * * 1'  # 毎週月曜 AM3:00 (JST 12:00)
  workflow_dispatch:

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run k6 load test
        uses: grafana/k6-action@v0.3.1
        with:
          filename: tests/load/k6-load-test.js
        env:
          K6_CLOUD_TOKEN: ${{ secrets.K6_CLOUD_TOKEN }}

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: k6-results
          path: k6-results.json

      - name: Notify on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "負荷テストが失敗しました: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

---

## 11. FAQ

### Q1: APM と RUM の両方が必要ですか？

はい、両方を導入することを強く推奨します。APM はサーバー側の問題（遅いクエリ、外部 API のタイムアウト）を特定し、RUM はクライアント側の問題（遅いネットワーク、重い JS 実行）を特定します。ユーザーが「遅い」と感じる原因は両方にあるため、片方だけでは根本原因の特定が困難です。

### Q2: Core Web Vitals は SEO にどの程度影響しますか？

Google は Core Web Vitals をランキングシグナルの一つとして使用しています。ただし、コンテンツの関連性ほど重要ではありません。同程度の関連性を持つページ間で差がつく「タイブレーカー」的な役割です。とはいえ、UX の観点から CWV を改善すること自体がコンバージョン率やエンゲージメントの向上に直結するため、SEO 関係なく取り組む価値があります。

### Q3: Synthetic Monitoring（合成監視）は RUM があれば不要ですか？

不要ではありません。合成監視は「統制された環境で定期的に計測する」ため、リグレッション検知に優れています。RUM はトラフィックがないページ（新規ページ、低アクセスページ）のデータが集まりません。また、合成監視は「ベースライン」を提供し、RUM のデータと比較することで、ネットワークやデバイスの影響を分離して分析できます。

### Q4: パフォーマンスバジェットの適切な設定値は？

業界やサービスの特性によりますが、一般的な指針として、(1) LCP: 2.5秒以下（Google 推奨の "Good" 基準）、(2) INP: 200ms 以下、(3) CLS: 0.1 以下、(4) JS バンドル: 300KB 以下 (gzip)、(5) 総転送量: 500KB 以下、があります。まず現状を計測し、そこから 10-20% 改善した値をバジェットに設定するのが現実的です。

### Q5: 負荷テストはどの頻度で実施すべきですか？

定期的な負荷テストは週次または隔週で実施することを推奨します。CI/CD に組み込む場合は、本番相当のステージング環境で実施してください。大規模なリリース前、インフラ変更前、予想されるトラフィック増加（セール、キャンペーン）前には必ず追加テストを実施します。結果は前回との比較で見ることが重要です。

### Q6: フロントエンドのパフォーマンス改善で最も効果が高い施策は？

多くのケースで最も効果が高いのは「不要なリソースの削減」です。具体的には、(1) 未使用の JavaScript の削除（Tree Shaking、コード分割）、(2) 画像の最適化（WebP/AVIF、適切なサイズ）、(3) サードパーティスクリプトの削減、(4) レンダリングブロッキングリソースの排除、の順に取り組むと効果的です。計測→改善→計測のサイクルを回すことが重要です。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 項目 | 要点 |
|------|------|
| APM | バックエンドの p50/p95/p99 レイテンシ、スループット、エラー率を監視 |
| RUM | 実ユーザーの Core Web Vitals、Navigation Timing を収集 |
| Core Web Vitals | LCP ≤ 2.5s、INP ≤ 200ms、CLS ≤ 0.1 を目標に |
| Lighthouse CI | PR ごとにパフォーマンススコアを自動計測 |
| Synthetic Monitoring | 定期的な外形監視でリグレッションを検知 |
| パフォーマンスバジェット | CI でバジェット超過を検知。リグレッションを防止 |
| パーセンタイル | 平均ではなく p95/p99 を監視。テールレイテンシに注目 |
| 負荷テスト | k6 等で定期的にスケーラビリティを検証 |
| バンドルサイズ | size-limit で CI に組み込み。300KB (gzip) を目標に |

---

## 次に読むべきガイド

- [00-observability.md](./00-observability.md) — オブザーバビリティの3本柱
- [01-monitoring-tools.md](./01-monitoring-tools.md) — 監視ツールの選定と構築
- [02-alerting.md](./02-alerting.md) — アラート戦略とポストモーテム

---

## 参考文献

1. **Web Vitals** — https://web.dev/vitals/ — Google による Core Web Vitals の公式ガイド
2. **High Performance Browser Networking** — Ilya Grigorik (O'Reilly, 2013) — ブラウザネットワーキングの原理
3. **web-vitals JavaScript Library** — https://github.com/GoogleChrome/web-vitals — CWV 計測ライブラリ
4. **Lighthouse CI** — https://github.com/GoogleChrome/lighthouse-ci — CI/CD でのパフォーマンス計測ツール
5. **k6 Documentation** — https://k6.io/docs/ — 負荷テストツール k6 の公式ガイド
6. **Checkly** — https://www.checklyhq.com/ — Synthetic Monitoring プラットフォーム
7. **size-limit** — https://github.com/ai/size-limit — バンドルサイズ制限ツール
8. **The RED Method** — https://www.weave.works/blog/the-red-method-key-metrics-for-microservices-architecture/ — マイクロサービス監視の RED メソッド
