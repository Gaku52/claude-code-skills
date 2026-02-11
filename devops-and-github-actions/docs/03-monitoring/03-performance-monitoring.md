# パフォーマンス監視

> APM、RUM、Core Web Vitals を活用してバックエンドとフロントエンドの両面からパフォーマンスを計測し、ユーザー体験を継続的に改善する

## この章で学ぶこと

1. **APM (Application Performance Monitoring)** — バックエンドのレイテンシ、スループット、エラー率のリアルタイム監視
2. **RUM (Real User Monitoring)** — 実際のユーザーが体験するフロントエンドパフォーマンスの計測
3. **Core Web Vitals** — Google が定義する UX 指標（LCP、INP、CLS）の計測と改善

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

---

## 2. APM — バックエンドパフォーマンス監視

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

// N+1 クエリ検出
class QueryMonitor {
  private queryCounts = new Map<string, number>();
  private readonly threshold = 10; // 同一パターンが10回以上で警告

  trackQuery(sql: string, duration: number): void {
    const pattern = this.normalizeQuery(sql);

    queryDuration.record(duration, { query_pattern: pattern });
    queryCounter.add(1, { query_pattern: pattern });

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
```

---

## 3. RUM — フロントエンドパフォーマンス監視

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

---

## 4. Core Web Vitals の基準

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

---

## 5. Lighthouse CI による継続的計測

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
```

```json
// lighthouse-budget.json — パフォーマンスバジェット
[
  {
    "path": "/*",
    "timings": [
      { "metric": "interactive", "budget": 3000 },
      { "metric": "first-contentful-paint", "budget": 1500 },
      { "metric": "largest-contentful-paint", "budget": 2500 }
    ],
    "resourceSizes": [
      { "resourceType": "script", "budget": 300 },
      { "resourceType": "total", "budget": 500 },
      { "resourceType": "image", "budget": 200 }
    ],
    "resourceCounts": [
      { "resourceType": "script", "budget": 10 },
      { "resourceType": "total", "budget": 50 }
    ]
  }
]
```

---

## 6. 比較表

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

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: APM と RUM の両方が必要ですか？

はい、両方を導入することを強く推奨します。APM はサーバー側の問題（遅いクエリ、外部 API のタイムアウト）を特定し、RUM はクライアント側の問題（遅いネットワーク、重い JS 実行）を特定します。ユーザーが「遅い」と感じる原因は両方にあるため、片方だけでは根本原因の特定が困難です。

### Q2: Core Web Vitals は SEO にどの程度影響しますか？

Google は Core Web Vitals をランキングシグナルの一つとして使用しています。ただし、コンテンツの関連性ほど重要ではありません。同程度の関連性を持つページ間で差がつく「タイブレーカー」的な役割です。とはいえ、UX の観点から CWV を改善すること自体がコンバージョン率やエンゲージメントの向上に直結するため、SEO 関係なく取り組む価値があります。

### Q3: Synthetic Monitoring（合成監視）は RUM があれば不要ですか？

不要ではありません。合成監視は「統制された環境で定期的に計測する」ため、リグレッション検知に優れています。RUM はトラフィックがないページ（新規ページ、低アクセスページ）のデータが集まりません。また、合成監視は「ベースライン」を提供し、RUM のデータと比較することで、ネットワークやデバイスの影響を分離して分析できます。

---

## まとめ

| 項目 | 要点 |
|------|------|
| APM | バックエンドの p50/p95/p99 レイテンシ、スループット、エラー率を監視 |
| RUM | 実ユーザーの Core Web Vitals、Navigation Timing を収集 |
| Core Web Vitals | LCP ≤ 2.5s、INP ≤ 200ms、CLS ≤ 0.1 を目標に |
| Lighthouse CI | PR ごとにパフォーマンススコアを自動計測 |
| パフォーマンスバジェット | CI でバジェット超過を検知。リグレッションを防止 |
| パーセンタイル | 平均ではなく p95/p99 を監視。テールレイテンシに注目 |

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
