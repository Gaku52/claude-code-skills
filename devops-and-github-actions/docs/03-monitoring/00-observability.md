# オブザーバビリティ

> ログ、メトリクス、トレースの3本柱を理解し、OpenTelemetry を活用してシステムの内部状態を可視化・診断できる力を身につける

## この章で学ぶこと

1. **オブザーバビリティの3本柱** — ログ、メトリクス、分散トレースの役割と相互関係
2. **OpenTelemetry による計装** — ベンダー非依存の統一的なテレメトリ収集の実装方法
3. **構造化ログとコンテキスト伝播** — 分散システムでのデバッグを可能にするログ設計と Trace Context の伝播

---

## 1. オブザーバビリティの全体像

```
┌─────────────────────────────────────────────────────────┐
│            オブザーバビリティの3本柱                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────┐  │
│  │    ログ     │ │  メトリクス   │ │  分散トレース    │  │
│  │  (Logs)     │ │  (Metrics)   │ │  (Traces)       │  │
│  ├─────────────┤ ├──────────────┤ ├─────────────────┤  │
│  │ 何が起きた  │ │ 集計された   │ │ リクエストが    │  │
│  │ かの詳細な  │ │ 数値データ   │ │ どう流れたか    │  │
│  │ イベント記録│ │              │ │                 │  │
│  ├─────────────┤ ├──────────────┤ ├─────────────────┤  │
│  │ デバッグ    │ │ ダッシュボード│ │ ボトルネック    │  │
│  │ 監査証跡    │ │ アラート     │ │ 依存関係の把握  │  │
│  │ エラー追跡  │ │ SLO/SLI     │ │ レイテンシ分析  │  │
│  └─────────────┘ └──────────────┘ └─────────────────┘  │
│         │                │                │             │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│              ┌────────────────────┐                     │
│              │  OpenTelemetry     │                     │
│              │  (統一収集基盤)     │                     │
│              └────────────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 構造化ログ

```typescript
// structured-logger.ts — 構造化ログの実装
import { pino } from 'pino';

// 構造化ログの基本設定
const logger = pino({
  level: process.env.LOG_LEVEL ?? 'info',
  formatters: {
    level(label) {
      return { level: label };
    },
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  base: {
    service: 'user-service',
    version: process.env.APP_VERSION ?? 'unknown',
    environment: process.env.NODE_ENV ?? 'development',
  },
  // 本番では JSON、開発では見やすい形式
  transport: process.env.NODE_ENV !== 'production'
    ? { target: 'pino-pretty', options: { colorize: true } }
    : undefined,
});

// リクエストコンテキスト付きの子ロガー
function createRequestLogger(req: {
  id: string;
  method: string;
  url: string;
  userId?: string;
}) {
  return logger.child({
    requestId: req.id,
    method: req.method,
    url: req.url,
    userId: req.userId,
  });
}

// 使用例
const reqLogger = createRequestLogger({
  id: 'req-abc123',
  method: 'POST',
  url: '/api/orders',
  userId: 'user-456',
});

reqLogger.info({ orderId: 'order-789' }, 'Order created successfully');
// 出力 (JSON):
// {
//   "level": "info",
//   "time": "2025-03-15T10:30:00.000Z",
//   "service": "user-service",
//   "version": "1.2.3",
//   "environment": "production",
//   "requestId": "req-abc123",
//   "method": "POST",
//   "url": "/api/orders",
//   "userId": "user-456",
//   "orderId": "order-789",
//   "msg": "Order created successfully"
// }

reqLogger.error(
  { err: new Error('Payment failed'), orderId: 'order-789' },
  'Failed to process payment'
);
```

---

## 3. OpenTelemetry 計装

```typescript
// otel-setup.ts — OpenTelemetry の初期化
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { Resource } from '@opentelemetry/resources';
import {
  ATTR_SERVICE_NAME,
  ATTR_SERVICE_VERSION,
} from '@opentelemetry/semantic-conventions';

const sdk = new NodeSDK({
  resource: new Resource({
    [ATTR_SERVICE_NAME]: 'order-service',
    [ATTR_SERVICE_VERSION]: process.env.APP_VERSION ?? '0.0.0',
    'deployment.environment': process.env.NODE_ENV ?? 'development',
  }),

  // トレースのエクスポーター
  traceExporter: new OTLPTraceExporter({
    url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT ?? 'http://localhost:4318/v1/traces',
  }),

  // メトリクスのエクスポーター
  metricReader: new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({
      url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT ?? 'http://localhost:4318/v1/metrics',
    }),
    exportIntervalMillis: 60000, // 60秒ごとにエクスポート
  }),

  // 自動計装 (HTTP, Express, pg, redis 等)
  instrumentations: [
    getNodeAutoInstrumentations({
      '@opentelemetry/instrumentation-fs': { enabled: false },
    }),
  ],
});

sdk.start();
console.log('OpenTelemetry SDK initialized');

// シャットダウン処理
process.on('SIGTERM', async () => {
  await sdk.shutdown();
  console.log('OpenTelemetry SDK shut down');
  process.exit(0);
});
```

```typescript
// custom-spans.ts — カスタムスパンの作成
import { trace, SpanStatusCode, Span } from '@opentelemetry/api';
import { Meter } from '@opentelemetry/api';

const tracer = trace.getTracer('order-service');

// カスタムメトリクスの定義
import { metrics } from '@opentelemetry/api';
const meter = metrics.getMeter('order-service');

const orderCounter = meter.createCounter('orders.created', {
  description: '作成された注文の総数',
  unit: '1',
});

const orderDuration = meter.createHistogram('orders.processing_duration', {
  description: '注文処理にかかった時間',
  unit: 'ms',
});

// カスタムスパン付きのビジネスロジック
async function createOrder(input: OrderInput): Promise<Order> {
  return tracer.startActiveSpan('createOrder', async (span: Span) => {
    try {
      span.setAttribute('order.customer_id', input.customerId);
      span.setAttribute('order.item_count', input.items.length);

      // 在庫確認スパン
      const inventory = await tracer.startActiveSpan(
        'checkInventory',
        async (inventorySpan) => {
          const result = await inventoryService.check(input.items);
          inventorySpan.setAttribute('inventory.available', result.available);
          inventorySpan.end();
          return result;
        }
      );

      if (!inventory.available) {
        span.setStatus({ code: SpanStatusCode.ERROR, message: 'Out of stock' });
        throw new Error('在庫不足');
      }

      // 決済処理スパン
      const payment = await tracer.startActiveSpan(
        'processPayment',
        async (paymentSpan) => {
          paymentSpan.setAttribute('payment.method', input.paymentMethod);
          const result = await paymentService.charge(input);
          paymentSpan.setAttribute('payment.transaction_id', result.transactionId);
          paymentSpan.end();
          return result;
        }
      );

      const order = await orderRepository.save({
        ...input,
        transactionId: payment.transactionId,
        status: 'confirmed',
      });

      // メトリクス記録
      orderCounter.add(1, { status: 'success', payment_method: input.paymentMethod });
      orderDuration.record(Date.now() - startTime, { status: 'success' });

      span.setAttribute('order.id', order.id);
      span.setStatus({ code: SpanStatusCode.OK });

      return order;
    } catch (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      span.recordException(error as Error);
      orderCounter.add(1, { status: 'failure' });
      throw error;
    } finally {
      span.end();
    }
  });
}
```

---

## 4. OpenTelemetry Collector 構成

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 10s
    send_batch_size: 1024

  memory_limiter:
    check_interval: 5s
    limit_mib: 512
    spike_limit_mib: 128

  attributes:
    actions:
      - key: environment
        value: production
        action: upsert

exporters:
  otlp/jaeger:
    endpoint: jaeger:4317
    tls:
      insecure: true

  prometheus:
    endpoint: 0.0.0.0:8889

  loki:
    endpoint: http://loki:3100/loki/api/v1/push

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, attributes]
      exporters: [otlp/jaeger]
    metrics:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch]
      exporters: [loki]
```

```
OpenTelemetry データフロー:

  アプリケーション群                OTel Collector         バックエンド
  ┌──────────────┐                ┌─────────────┐
  │ Service A    │── traces ────► │             │───► Jaeger (トレース)
  │ (Node.js)    │── metrics ───► │  Receiver   │
  └──────────────┘── logs ──────► │      │      │───► Prometheus (メトリクス)
  ┌──────────────┐                │  Processor  │
  │ Service B    │── traces ────► │      │      │───► Loki (ログ)
  │ (Python)     │── metrics ───► │  Exporter   │
  └──────────────┘── logs ──────► │             │───► Grafana (可視化)
  ┌──────────────┐                └─────────────┘
  │ Service C    │── traces ────►
  │ (Go)         │── metrics ───►
  └──────────────┘
```

---

## 5. SLI/SLO の定義

```typescript
// slo-definitions.ts — SLI/SLO の定義例
interface SLI {
  name: string;
  description: string;
  query: string;        // PromQL クエリ
  unit: string;
}

interface SLO {
  name: string;
  sli: SLI;
  target: number;       // 例: 0.999 = 99.9%
  window: string;       // 例: '30d'
  burnRateAlerts: BurnRateAlert[];
}

// 可用性 SLI
const availabilitySLI: SLI = {
  name: 'availability',
  description: 'HTTP リクエストの成功率',
  query: `
    sum(rate(http_requests_total{status!~"5.."}[5m]))
    /
    sum(rate(http_requests_total[5m]))
  `,
  unit: 'ratio',
};

// レイテンシ SLI
const latencySLI: SLI = {
  name: 'latency',
  description: 'p99 レイテンシが 500ms 以内のリクエスト割合',
  query: `
    sum(rate(http_request_duration_seconds_bucket{le="0.5"}[5m]))
    /
    sum(rate(http_request_duration_seconds_count[5m]))
  `,
  unit: 'ratio',
};

// SLO 定義
const apiAvailabilitySLO: SLO = {
  name: 'API Availability',
  sli: availabilitySLI,
  target: 0.999,  // 99.9%
  window: '30d',  // 30日間のローリングウィンドウ
  burnRateAlerts: [
    { severity: 'critical', shortWindow: '5m', longWindow: '1h', factor: 14.4 },
    { severity: 'warning',  shortWindow: '30m', longWindow: '6h', factor: 6 },
  ],
};
```

---

## 6. 比較表

| 柱 | ログ | メトリクス | トレース |
|----|------|----------|---------|
| データ形式 | テキスト/JSON イベント | 数値の時系列 | スパンのツリー構造 |
| カーディナリティ | 高い | 低い | 中 |
| ストレージコスト | 高い | 低い | 中 |
| リアルタイム性 | 高い | 中 (集計間隔) | 高い |
| 用途 | デバッグ、監査 | アラート、ダッシュボード | パフォーマンス分析 |
| クエリ速度 | 遅い (全文検索) | 速い (時系列DB) | 中 |

| OTel バックエンド | Jaeger | Zipkin | Tempo | Datadog |
|-------------------|--------|--------|-------|---------|
| トレース | 対応 | 対応 | 対応 | 対応 |
| メトリクス | 非対応 | 非対応 | 非対応 | 対応 |
| ログ | 非対応 | 非対応 | 非対応 | 対応 |
| ストレージ | Elasticsearch/Cassandra | MySQL/Elasticsearch | オブジェクトストレージ | SaaS |
| 運用負荷 | 中 | 低い | 低い | 最低 (SaaS) |
| コスト | 無料 (OSS) | 無料 (OSS) | 無料 (OSS) | 有料 |

---

## 7. アンチパターン

### アンチパターン 1: 非構造化ログの乱用

```typescript
// 悪い例: 非構造化ログ
console.log('User ' + userId + ' created order ' + orderId + ' at ' + new Date());
console.log('Error: ' + err.message);

// → 検索困難、パース不能、コンテキスト不足

// 良い例: 構造化ログ
logger.info({
  event: 'order_created',
  userId,
  orderId,
  timestamp: new Date().toISOString(),
}, 'Order created successfully');

logger.error({
  event: 'order_failed',
  err,
  userId,
  orderId,
}, 'Failed to create order');

// → JSON で構造化、フィールドで検索・フィルタ可能
```

### アンチパターン 2: 計装なしの本番運用

```
[悪い例]
- ログだけで障害対応
- 「どのサービスが遅い？」→ 各サーバーに SSH してログを grep
- 「リクエスト数は？」→ access.log を wc -l
- 障害の根本原因特定に数時間〜数日

[良い例]
- OpenTelemetry で3本柱を統一的に収集
- ダッシュボードでリアルタイムに状況把握
- トレースでリクエストフローを可視化
- メトリクスで SLO 違反を自動検知
- 障害の根本原因を数分で特定
```

---

## 8. FAQ

### Q1: OpenTelemetry の自動計装と手動計装、どちらを使うべきですか？

まず自動計装（auto-instrumentation）から始めてください。HTTP リクエスト、データベースクエリ、外部 API 呼び出しなどの基本的なスパンが自動生成されます。その上で、ビジネスロジック固有の情報（注文処理、決済処理など）は手動でカスタムスパンを追加します。自動計装だけでは「何を処理しているか」が分からないため、両方を組み合わせるのが最善です。

### Q2: ログレベルの使い分けはどうすべきですか？

**ERROR**: システムが処理を続行できない障害（DB接続失敗、外部API障害）。**WARN**: 予期しない状態だが処理は継続可能（リトライ成功、フォールバック使用）。**INFO**: 正常なビジネスイベント（注文作成、ユーザー登録）。**DEBUG**: 開発時の詳細情報（関数呼び出し、変数値）。本番では INFO 以上を推奨し、障害調査時に一時的に DEBUG に下げます。

### Q3: SLO のターゲットはどう決めるべきですか？

まず現状のメトリクスを2〜4週間収集し、ベースラインを把握します。その上で「ユーザーにとって許容できるレベル」と「達成可能なレベル」のバランスで設定します。一般的な Web API なら 99.9%（月間ダウンタイム約43分）が出発点です。100% を目指すとコストが指数関数的に増加するため、エラーバジェット（許容できるエラー量）の考え方を導入してください。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ログ | 構造化 JSON 形式。requestId でリクエストを追跡可能に |
| メトリクス | 数値の時系列データ。SLI/SLO の基盤 |
| トレース | 分散システムのリクエストフローを可視化 |
| OpenTelemetry | ベンダー非依存の計装標準。自動+手動計装を併用 |
| OTel Collector | テレメトリデータの収集・加工・転送の中央ハブ |
| SLI/SLO | ユーザー視点の信頼性指標。エラーバジェットで管理 |

---

## 次に読むべきガイド

- [01-monitoring-tools.md](./01-monitoring-tools.md) — Datadog、Grafana、CloudWatch の活用
- [02-alerting.md](./02-alerting.md) — アラート設計とエスカレーション
- [03-performance-monitoring.md](./03-performance-monitoring.md) — APM、RUM、Core Web Vitals

---

## 参考文献

1. **Observability Engineering** — Charity Majors, Liz Fong-Jones, George Miranda (O'Reilly, 2022) — オブザーバビリティの実践ガイド
2. **OpenTelemetry Documentation** — https://opentelemetry.io/docs/ — OTel 公式ドキュメント
3. **Google SRE Book - Monitoring Distributed Systems** — https://sre.google/sre-book/monitoring-distributed-systems/ — Google の監視手法
4. **Site Reliability Engineering** — Betsy Beyer et al. (O'Reilly, 2016) — SRE の原典
