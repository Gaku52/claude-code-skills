# オブザーバビリティ

> ログ、メトリクス、トレースの3本柱を理解し、OpenTelemetry を活用してシステムの内部状態を可視化・診断できる力を身につける

## この章で学ぶこと

1. **オブザーバビリティの3本柱** — ログ、メトリクス、分散トレースの役割と相互関係
2. **OpenTelemetry による計装** — ベンダー非依存の統一的なテレメトリ収集の実装方法
3. **構造化ログとコンテキスト伝播** — 分散システムでのデバッグを可能にするログ設計と Trace Context の伝播
4. **SLI/SLO/SLA の設計と運用** — ユーザー視点の信頼性指標の定義とエラーバジェットの管理
5. **OpenTelemetry Collector の構成と運用** — テレメトリデータの収集・加工・転送パイプラインの設計

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

### モニタリングとオブザーバビリティの違い

オブザーバビリティは「既知の問題を検知する」モニタリングを包含しつつ、「未知の問題を診断する」能力を加えた概念である。

```
モニタリング vs オブザーバビリティ:

  モニタリング (Monitoring)
  ┌──────────────────────────────────────────┐
  │ 「何が壊れたか」を検知する               │
  │                                          │
  │ - CPU使用率が80%を超えた                  │
  │ - エラーレートが閾値を超えた               │
  │ - ディスク容量が残り10%                    │
  │                                          │
  │ 既知の障害パターンに対するアラート         │
  │ 事前に「何を監視するか」を決める必要がある │
  └──────────────────────────────────────────┘

  オブザーバビリティ (Observability)
  ┌──────────────────────────────────────────┐
  │ 「なぜ壊れたか」を診断する               │
  │                                          │
  │ - このリクエストはどのサービスを通った？    │
  │ - なぜ特定ユーザーだけ遅い？              │
  │ - 昨日と今日で何が変わった？              │
  │                                          │
  │ 未知の障害パターンにも対応可能             │
  │ 高カーディナリティデータで自由に探索       │
  │                                          │
  │ ログ + メトリクス + トレースの相関分析     │
  └──────────────────────────────────────────┘
```

### オブザーバビリティの成熟度モデル

```
成熟度レベル:

  Level 0: なし
  ┌──────────────────────────────────────┐
  │ - ログはコンソール出力のみ            │
  │ - 障害は「ユーザーからの報告」で検知   │
  │ - ssh してログファイルを grep          │
  └──────────────────────────────────────┘

  Level 1: 基本的なモニタリング
  ┌──────────────────────────────────────┐
  │ - CloudWatch/Datadog でメトリクス収集 │
  │ - 基本的なアラート (CPU, メモリ)      │
  │ - ログ集約 (CloudWatch Logs等)       │
  └──────────────────────────────────────┘

  Level 2: 構造化された可観測性
  ┌──────────────────────────────────────┐
  │ - 構造化ログ (JSON)                  │
  │ - カスタムメトリクス (ビジネスKPI)     │
  │ - 分散トレーシング                    │
  │ - ダッシュボード体系化                 │
  └──────────────────────────────────────┘

  Level 3: SLO ドリブン (目指すべき姿)
  ┌──────────────────────────────────────┐
  │ - SLI/SLO ベースのアラート            │
  │ - エラーバジェットによる意思決定       │
  │ - ログ↔メトリクス↔トレースの相関分析  │
  │ - OpenTelemetry で統一計装            │
  │ - ポストモーテム文化の定着            │
  └──────────────────────────────────────┘
```

---

## 2. 構造化ログ

### 2.1 構造化ログの基本実装

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

### 2.2 ログレベルの設計指針

```
ログレベルの使い分け:

  ┌─────────┬──────────────────────────────────────────┐
  │ FATAL   │ プロセスが継続不能。即座に終了する障害     │
  │         │ 例: DB接続不能で起動失敗                   │
  ├─────────┼──────────────────────────────────────────┤
  │ ERROR   │ 処理が失敗。人間の介入が必要               │
  │         │ 例: 決済API障害、データ不整合               │
  ├─────────┼──────────────────────────────────────────┤
  │ WARN    │ 予期しない状態だが処理は継続可能           │
  │         │ 例: リトライ成功、フォールバック使用         │
  ├─────────┼──────────────────────────────────────────┤
  │ INFO    │ 正常なビジネスイベント                     │
  │         │ 例: 注文作成、ユーザー登録、デプロイ完了    │
  ├─────────┼──────────────────────────────────────────┤
  │ DEBUG   │ 開発時の詳細情報                           │
  │         │ 例: 関数呼び出し、変数値、SQL クエリ        │
  ├─────────┼──────────────────────────────────────────┤
  │ TRACE   │ 最も詳細なデバッグ情報                     │
  │         │ 例: ループ内の各イテレーション               │
  └─────────┴──────────────────────────────────────────┘

  本番環境: INFO 以上
  ステージング: DEBUG 以上
  開発環境: TRACE 以上

  障害調査時: 一時的に DEBUG に下げて詳細ログを収集
```

### 2.3 Express/Fastify でのリクエストログ

```typescript
// request-logging-middleware.ts — リクエスト/レスポンスログ
import { Request, Response, NextFunction } from 'express';
import { pino } from 'pino';
import { v4 as uuidv4 } from 'uuid';

const logger = pino({
  level: process.env.LOG_LEVEL ?? 'info',
  base: { service: 'api-gateway' },
});

// センシティブデータのマスキング
function maskSensitiveHeaders(headers: Record<string, string>): Record<string, string> {
  const masked = { ...headers };
  const sensitiveKeys = ['authorization', 'cookie', 'x-api-key'];

  for (const key of sensitiveKeys) {
    if (masked[key]) {
      masked[key] = '***REDACTED***';
    }
  }
  return masked;
}

// リクエストログミドルウェア
export function requestLogger(req: Request, res: Response, next: NextFunction) {
  // リクエストID の生成/伝播
  const requestId = req.headers['x-request-id'] as string ?? uuidv4();
  req.headers['x-request-id'] = requestId;
  res.setHeader('x-request-id', requestId);

  const startTime = process.hrtime.bigint();

  // 子ロガーの作成
  const reqLogger = logger.child({
    requestId,
    method: req.method,
    url: req.url,
    userAgent: req.headers['user-agent'],
    ip: req.ip,
  });

  // リクエスト開始ログ
  reqLogger.info(
    { headers: maskSensitiveHeaders(req.headers as Record<string, string>) },
    'Request received'
  );

  // レスポンス完了時のログ
  res.on('finish', () => {
    const durationNs = process.hrtime.bigint() - startTime;
    const durationMs = Number(durationNs) / 1_000_000;

    const logData = {
      statusCode: res.statusCode,
      duration: Math.round(durationMs * 100) / 100,
      contentLength: res.getHeader('content-length'),
    };

    if (res.statusCode >= 500) {
      reqLogger.error(logData, 'Request failed (5xx)');
    } else if (res.statusCode >= 400) {
      reqLogger.warn(logData, 'Request failed (4xx)');
    } else {
      reqLogger.info(logData, 'Request completed');
    }
  });

  next();
}
```

### 2.4 Python (FastAPI) での構造化ログ

```python
# structured_logger.py — Python での構造化ログ
import logging
import json
import sys
import uuid
from datetime import datetime, timezone
from contextvars import ContextVar
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

# コンテキスト変数でリクエストIDを伝播
request_id_var: ContextVar[str] = ContextVar('request_id', default='')

class JSONFormatter(logging.Formatter):
    """JSON 形式のログフォーマッター"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname.lower(),
            'message': record.getMessage(),
            'service': 'order-service',
            'logger': record.name,
            'request_id': request_id_var.get(''),
        }

        # エラー情報の追加
        if record.exc_info:
            log_data['error'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
            }

        # 追加フィールド
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)

        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(name: str = 'app') -> logging.Logger:
    """構造化ログのセットアップ"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    return logger


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """リクエストログミドルウェア"""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = setup_logger('http')

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get('x-request-id', str(uuid.uuid4()))
        request_id_var.set(request_id)

        start_time = time.perf_counter()

        self.logger.info(
            'Request received',
            extra={'extra_fields': {
                'method': request.method,
                'url': str(request.url),
                'client_ip': request.client.host if request.client else None,
            }}
        )

        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers['x-request-id'] = request_id

        self.logger.info(
            'Request completed',
            extra={'extra_fields': {
                'method': request.method,
                'url': str(request.url),
                'status_code': response.status_code,
                'duration_ms': round(duration_ms, 2),
            }}
        )

        return response
```

---

## 3. OpenTelemetry 計装

### 3.1 Node.js での初期化

```typescript
// otel-setup.ts — OpenTelemetry の初期化
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { OTLPLogExporter } from '@opentelemetry/exporter-logs-otlp-http';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { BatchLogRecordProcessor } from '@opentelemetry/sdk-logs';
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
    'service.namespace': 'myapp',
    'host.name': process.env.HOSTNAME ?? 'unknown',
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

  // ログのエクスポーター
  logRecordProcessor: new BatchLogRecordProcessor(
    new OTLPLogExporter({
      url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT ?? 'http://localhost:4318/v1/logs',
    })
  ),

  // 自動計装 (HTTP, Express, pg, redis 等)
  instrumentations: [
    getNodeAutoInstrumentations({
      '@opentelemetry/instrumentation-fs': { enabled: false },
      '@opentelemetry/instrumentation-http': {
        ignoreIncomingRequestHook: (req) => {
          // ヘルスチェックはトレースしない
          return req.url === '/health' || req.url === '/readyz';
        },
      },
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

### 3.2 カスタムスパンとメトリクスの作成

```typescript
// custom-spans.ts — カスタムスパンの作成
import { trace, SpanStatusCode, Span } from '@opentelemetry/api';
import { metrics } from '@opentelemetry/api';

const tracer = trace.getTracer('order-service');

// カスタムメトリクスの定義
const meter = metrics.getMeter('order-service');

const orderCounter = meter.createCounter('orders.created', {
  description: '作成された注文の総数',
  unit: '1',
});

const orderDuration = meter.createHistogram('orders.processing_duration', {
  description: '注文処理にかかった時間',
  unit: 'ms',
});

const activeOrders = meter.createUpDownCounter('orders.active', {
  description: '処理中の注文数',
  unit: '1',
});

const orderValue = meter.createHistogram('orders.value', {
  description: '注文金額の分布',
  unit: 'JPY',
});

// カスタムスパン付きのビジネスロジック
async function createOrder(input: OrderInput): Promise<Order> {
  return tracer.startActiveSpan('createOrder', async (span: Span) => {
    const startTime = Date.now();
    activeOrders.add(1);

    try {
      span.setAttribute('order.customer_id', input.customerId);
      span.setAttribute('order.item_count', input.items.length);
      span.setAttribute('order.payment_method', input.paymentMethod);

      // 在庫確認スパン
      const inventory = await tracer.startActiveSpan(
        'checkInventory',
        async (inventorySpan) => {
          const result = await inventoryService.check(input.items);
          inventorySpan.setAttribute('inventory.available', result.available);
          inventorySpan.setAttribute('inventory.items_checked', input.items.length);
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
          paymentSpan.setAttribute('payment.amount', input.totalAmount);
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
      orderCounter.add(1, {
        status: 'success',
        payment_method: input.paymentMethod,
      });
      orderDuration.record(Date.now() - startTime, { status: 'success' });
      orderValue.record(input.totalAmount, {
        payment_method: input.paymentMethod,
      });

      span.setAttribute('order.id', order.id);
      span.setStatus({ code: SpanStatusCode.OK });

      return order;
    } catch (error) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: String(error) });
      span.recordException(error as Error);
      orderCounter.add(1, { status: 'failure' });
      orderDuration.record(Date.now() - startTime, { status: 'failure' });
      throw error;
    } finally {
      activeOrders.add(-1);
      span.end();
    }
  });
}
```

### 3.3 コンテキスト伝播 (W3C Trace Context)

```typescript
// context-propagation.ts — サービス間のコンテキスト伝播
import { context, propagation, trace } from '@opentelemetry/api';

// HTTP クライアントでのコンテキスト伝播 (送信側)
async function callExternalService(url: string, body: object): Promise<Response> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  // 現在のコンテキストからトレースヘッダーを注入
  // W3C Trace Context: traceparent, tracestate
  propagation.inject(context.active(), headers);

  // ヘッダーに以下が追加される:
  // traceparent: 00-<trace-id>-<span-id>-01
  // tracestate: vendor=value

  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
}

// HTTP サーバーでのコンテキスト抽出 (受信側)
function extractContext(req: Request): void {
  // リクエストヘッダーからトレースコンテキストを抽出
  const extractedContext = propagation.extract(context.active(), req.headers);

  // このコンテキスト内でスパンを作成すると
  // 親スパンと自動的にリンクされる
  const tracer = trace.getTracer('my-service');
  const span = tracer.startSpan('handleRequest', {}, extractedContext);

  // 処理...
  span.end();
}
```

```
W3C Trace Context によるサービス間のトレース伝播:

  Service A                    Service B                    Service C
  ┌────────────┐               ┌────────────┐               ┌────────────┐
  │ Span A     │               │ Span B     │               │ Span C     │
  │ trace: abc │── HTTP ──►    │ trace: abc │── HTTP ──►    │ trace: abc │
  │ span: 001  │  traceparent  │ span: 002  │  traceparent  │ span: 003  │
  │ parent: -  │  ヘッダー付与  │ parent: 001│  ヘッダー付与  │ parent: 002│
  └────────────┘               └────────────┘               └────────────┘

  traceparent ヘッダーの構造:
  00-<trace-id (32hex)>-<parent-span-id (16hex)>-<flags (2hex)>
  00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01

  全スパンが同じ trace-id を共有 → 1つのリクエストフローとして可視化
```

---

## 4. OpenTelemetry Collector 構成

### 4.1 基本構成

```yaml
# otel-collector-config.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

  # ホストメトリクスの自動収集
  hostmetrics:
    collection_interval: 30s
    scrapers:
      cpu:
      memory:
      disk:
      network:

  # Prometheus メトリクスのスクレイプ
  prometheus:
    config:
      scrape_configs:
        - job_name: 'app-metrics'
          scrape_interval: 15s
          static_configs:
            - targets: ['app:3000']

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

  # センシティブデータのフィルタリング
  attributes/remove-sensitive:
    actions:
      - key: http.request.header.authorization
        action: delete
      - key: http.request.header.cookie
        action: delete
      - key: db.statement
        action: hash  # SQL をハッシュ化

  # テールサンプリング (エラーと遅いリクエストを優先)
  tail_sampling:
    decision_wait: 10s
    num_traces: 100
    policies:
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      - name: slow-requests
        type: latency
        latency:
          threshold_ms: 1000
      - name: random-sample
        type: probabilistic
        probabilistic:
          sampling_percentage: 10

exporters:
  otlp/jaeger:
    endpoint: jaeger:4317
    tls:
      insecure: true

  prometheus:
    endpoint: 0.0.0.0:8889
    namespace: myapp

  loki:
    endpoint: http://loki:3100/loki/api/v1/push

  debug:
    verbosity: detailed

service:
  extensions: [health_check, pprof, zpages]
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, tail_sampling, batch, attributes]
      exporters: [otlp/jaeger]
    metrics:
      receivers: [otlp, hostmetrics, prometheus]
      processors: [memory_limiter, batch]
      exporters: [prometheus]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, attributes/remove-sensitive, batch]
      exporters: [loki]

  telemetry:
    logs:
      level: info
    metrics:
      address: 0.0.0.0:8888
```

### 4.2 Docker Compose でのオブザーバビリティスタック

```yaml
# docker-compose.observability.yml
version: "3.8"

services:
  # OpenTelemetry Collector
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.96.0
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "8889:8889"   # Prometheus exporter
    depends_on:
      - jaeger
      - loki

  # Jaeger (分散トレーシング)
  jaeger:
    image: jaegertracing/all-in-one:1.54
    ports:
      - "16686:16686"  # UI
      - "14250:14250"  # gRPC
    environment:
      - COLLECTOR_OTLP_ENABLED=true

  # Prometheus (メトリクス)
  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  # Loki (ログ集約)
  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"

  # Grafana (可視化)
  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_AUTH_ANONYMOUS_ENABLED=true
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
      - jaeger
      - loki
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

### 5.1 SLI/SLO/SLA の関係

```
SLI/SLO/SLA の階層:

  ┌─────────────────────────────────────────┐
  │  SLA (Service Level Agreement)           │
  │  ビジネス契約。違反時にペナルティ          │
  │  例: 月間可用性 99.95% を保証             │
  │      違反時はサービス料金を返金            │
  ├─────────────────────────────────────────┤
  │  SLO (Service Level Objective)           │
  │  内部目標。SLA より厳しく設定             │
  │  例: 月間可用性 99.99% を目標             │
  │      (SLA より厳しいので余裕がある)        │
  ├─────────────────────────────────────────┤
  │  SLI (Service Level Indicator)           │
  │  測定指標。SLO の達成度を計測             │
  │  例: 成功リクエスト数 / 全リクエスト数     │
  └─────────────────────────────────────────┘

  ダウンタイム許容量 (30日間):
  ┌──────────┬───────────────┬──────────────┐
  │ SLO      │ エラーバジェット │ ダウンタイム │
  ├──────────┼───────────────┼──────────────┤
  │ 99%      │ 1%            │ 7時間12分    │
  │ 99.9%    │ 0.1%          │ 43分12秒     │
  │ 99.95%   │ 0.05%         │ 21分36秒     │
  │ 99.99%   │ 0.01%         │ 4分19秒      │
  │ 99.999%  │ 0.001%        │ 26秒         │
  └──────────┴───────────────┴──────────────┘
```

### 5.2 SLI/SLO の定義例

```typescript
// slo-definitions.ts — SLI/SLO の定義例
interface SLI {
  name: string;
  description: string;
  query: string;        // PromQL クエリ
  unit: string;
}

interface BurnRateAlert {
  severity: string;
  shortWindow: string;
  longWindow: string;
  factor: number;
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

// スループット SLI
const throughputSLI: SLI = {
  name: 'throughput',
  description: '1秒あたりの処理リクエスト数',
  query: `sum(rate(http_requests_total[5m]))`,
  unit: 'requests/sec',
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
    { severity: 'ticket',   shortWindow: '6h', longWindow: '3d', factor: 1 },
  ],
};

const apiLatencySLO: SLO = {
  name: 'API Latency',
  sli: latencySLI,
  target: 0.99,  // 99%
  window: '30d',
  burnRateAlerts: [
    { severity: 'critical', shortWindow: '5m', longWindow: '1h', factor: 14.4 },
    { severity: 'warning',  shortWindow: '30m', longWindow: '6h', factor: 6 },
  ],
};
```

### 5.3 エラーバジェットの運用

```
エラーバジェットの考え方:

  SLO: 99.9% (30日間)
  エラーバジェット: 0.1% = 43.2分のダウンタイム

  月初:
  ┌──────────────────────────────────────────┐
  │ ████████████████████████████████ 100%     │
  │ エラーバジェット残: 43.2分                  │
  └──────────────────────────────────────────┘

  15日目 (インシデント発生: 20分のダウンタイム):
  ┌──────────────────────────────────────────┐
  │ █████████████████░░░░░░░░░░░░░░ 54%      │
  │ エラーバジェット残: 23.2分                  │
  └──────────────────────────────────────────┘

  20日目 (バジェット枯渇):
  ┌──────────────────────────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0%       │
  │ エラーバジェット残: 0分                     │
  └──────────────────────────────────────────┘
  → 機能開発を一時停止し、信頼性改善に注力

  エラーバジェットポリシー:
  ┌──────────────────────────────────────────┐
  │ バジェット > 50%: 通常の開発を継続         │
  │ バジェット 25-50%: リスクの高い変更を制限   │
  │ バジェット < 25%: 信頼性改善に優先投資      │
  │ バジェット = 0%: 機能開発を凍結             │
  └──────────────────────────────────────────┘
```

### 5.4 Grafana での SLO ダッシュボード

```yaml
# grafana/dashboards/slo-dashboard.json (概要)
# SLO ダッシュボードに含めるべきパネル:
#
# 1. 現在の SLI 値 (Stat パネル)
#    - 可用性: 99.95% (目標: 99.9%)
#    - レイテンシ p99: 350ms (目標: 500ms)
#
# 2. エラーバジェット残量 (Gauge パネル)
#    - 残り: 65% (28分)
#
# 3. エラーバジェットの推移 (Time series パネル)
#    - 30日間のバーンダウンチャート
#
# 4. バーンレート (Time series パネル)
#    - 1h, 6h, 3d のウィンドウでのバーンレート
#
# 5. SLI の時系列推移 (Time series パネル)
#    - 可用性とレイテンシの推移グラフ
```

```promql
# SLO ダッシュボード用 PromQL クエリ

# 1. 現在の可用性 (30日ローリング)
1 - (
  sum(increase(http_requests_total{status=~"5.."}[30d]))
  /
  sum(increase(http_requests_total[30d]))
)

# 2. エラーバジェット残量 (%)
(
  1 - (
    sum(increase(http_requests_total{status=~"5.."}[30d]))
    /
    sum(increase(http_requests_total[30d]))
  )
  - 0.999
) / 0.001 * 100

# 3. バーンレート (1時間ウィンドウ)
(
  sum(rate(http_requests_total{status=~"5.."}[1h]))
  /
  sum(rate(http_requests_total[1h]))
) / 0.001

# 4. p99 レイテンシ
histogram_quantile(0.99,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)
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
| 保持期間 | 短い (7-30日) | 長い (1-2年) | 中 (7-30日) |
| サンプリング | 通常なし | 常に全量 | 推奨 (1-10%) |

| OTel バックエンド | Jaeger | Zipkin | Tempo | Datadog |
|-------------------|--------|--------|-------|---------|
| トレース | 対応 | 対応 | 対応 | 対応 |
| メトリクス | 非対応 | 非対応 | 非対応 | 対応 |
| ログ | 非対応 | 非対応 | 非対応 | 対応 |
| ストレージ | Elasticsearch/Cassandra | MySQL/Elasticsearch | オブジェクトストレージ | SaaS |
| 運用負荷 | 中 | 低い | 低い | 最低 (SaaS) |
| コスト | 無料 (OSS) | 無料 (OSS) | 無料 (OSS) | 有料 |

| ログ集約ツール | Loki | Elasticsearch | CloudWatch Logs | Datadog Logs |
|---------------|------|--------------|-----------------|-------------|
| インデックス方式 | ラベルのみ | 全文検索 | 全文検索 | 全文検索 |
| ストレージ効率 | 高い | 低い | 中 | 中 |
| クエリ速度 | 中 | 高い | 中 | 高い |
| 運用負荷 | 低い | 高い | 最低 | 最低 |
| コスト | 低い | 高い | 従量課金 | 従量課金 |
| Grafana 連携 | ネイティブ | 対応 | 非対応 | 非対応 |

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

### アンチパターン 3: 過剰なログ出力

```
[悪い例]
- 全リクエストの全パラメータをログに記録
- DEBUG レベルのログを本番で有効化
- ログストレージコストが月10万円超
- ログが多すぎて重要な情報が埋もれる

[良い例]
- 本番は INFO 以上のみ
- サンプリングで重要なリクエストのみ詳細ログ
- ログの保持期間を適切に設定 (7-30日)
- コスト監視とログ量のアラートを設定
- 機密情報 (パスワード、トークン) のマスキング
```

### アンチパターン 4: SLO を設定しない

```
[悪い例]
- 「可用性100%を目指す」→ 非現実的でチームが疲弊
- アラートが多すぎて対応しきれない
- 障害対応と機能開発の優先度が曖昧
- 「どこまで信頼性に投資すべきか」が決まらない

[良い例]
- SLO を明確に定義 (例: 可用性 99.9%)
- エラーバジェットで開発と信頼性のバランスを取る
- バジェット消費に応じてリリース速度を調整
- 四半期ごとに SLO を見直し・調整
```

---

## 8. FAQ

### Q1: OpenTelemetry の自動計装と手動計装、どちらを使うべきですか？

まず自動計装（auto-instrumentation）から始めてください。HTTP リクエスト、データベースクエリ、外部 API 呼び出しなどの基本的なスパンが自動生成されます。その上で、ビジネスロジック固有の情報（注文処理、決済処理など）は手動でカスタムスパンを追加します。自動計装だけでは「何を処理しているか」が分からないため、両方を組み合わせるのが最善です。具体的には、自動計装で全体のリクエストフローを把握し、手動計装でビジネスに関連するアトリビュート（注文ID、顧客ID、金額など）を付加します。

### Q2: ログレベルの使い分けはどうすべきですか？

**ERROR**: システムが処理を続行できない障害（DB接続失敗、外部API障害）。**WARN**: 予期しない状態だが処理は継続可能（リトライ成功、フォールバック使用）。**INFO**: 正常なビジネスイベント（注文作成、ユーザー登録）。**DEBUG**: 開発時の詳細情報（関数呼び出し、変数値）。本番では INFO 以上を推奨し、障害調査時に一時的に DEBUG に下げます。ログレベルを動的に変更できる仕組み（環境変数や API エンドポイント）を用意しておくと便利です。

### Q3: SLO のターゲットはどう決めるべきですか？

まず現状のメトリクスを2〜4週間収集し、ベースラインを把握します。その上で「ユーザーにとって許容できるレベル」と「達成可能なレベル」のバランスで設定します。一般的な Web API なら 99.9%（月間ダウンタイム約43分）が出発点です。100% を目指すとコストが指数関数的に増加するため、エラーバジェット（許容できるエラー量）の考え方を導入してください。SLO は固定ではなく、四半期ごとに見直して調整するのが良い実践です。

### Q4: トレースのサンプリング率はどのくらいが適切ですか？

トラフィック量によりますが、一般的には以下が目安です:
- **低トラフィック** (< 100 req/s): 100% (全リクエスト)
- **中トラフィック** (100-1000 req/s): 10-50%
- **高トラフィック** (> 1000 req/s): 1-10%

ただし、エラーのあるリクエストは常に100%キャプチャする「テールサンプリング」を推奨します。OpenTelemetry Collector の `tail_sampling` プロセッサを使うと、エラー・遅延リクエストを優先的に保持しつつ、正常リクエストのサンプリング率を下げることができます。

### Q5: OpenTelemetry Collector は必須ですか？

必須ではありませんが、本番環境では強く推奨します。Collector を介すメリット:
1. **デカップリング**: アプリケーションはバックエンドを意識しない
2. **バッチ処理**: ネットワーク効率の向上
3. **加工・フィルタリング**: センシティブデータの除去、サンプリング
4. **リトライ**: バックエンド障害時のバッファリング
5. **マルチバックエンド**: 同じデータを複数の宛先に送信

開発環境では Collector なしで直接バックエンドに送信しても問題ありません。

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
| コンテキスト伝播 | W3C Trace Context で分散トレースを実現 |
| サンプリング | テールサンプリングでコストと品質のバランスを取る |
| ログレベル | 本番は INFO 以上。動的変更の仕組みを用意 |
| エラーバジェット | SLO 違反の許容量。開発速度と信頼性のバランス指標 |

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
5. **Implementing Service Level Objectives** — Alex Hidalgo (O'Reilly, 2020) — SLO の実装ガイド
6. **W3C Trace Context** — https://www.w3.org/TR/trace-context/ — 分散トレースの標準仕様
7. **OpenTelemetry Collector Documentation** — https://opentelemetry.io/docs/collector/ — Collector の公式ドキュメント
