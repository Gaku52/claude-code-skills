# ログとモニタリング

> エラーは発生する。重要なのは「素早く検知し、原因を特定し、修正する」こと。構造化ログ、エラートラッキング（Sentry）、アラート設計のベストプラクティスを解説。

## この章で学ぶこと

- [ ] 構造化ログの設計を理解する
- [ ] エラートラッキングサービスの活用を把握する
- [ ] 効果的なアラート設計を学ぶ
- [ ] 分散トレーシングの基礎を理解する
- [ ] メトリクス収集とダッシュボード設計を習得する
- [ ] ログのセキュリティとコンプライアンスを把握する

---

## 1. 構造化ログ

### 1.1 構造化ログ vs 非構造化ログ

```
非構造化ログ（従来）:
  [2025-01-15 10:30:45] ERROR: Failed to process order 12345 for user abc
  → 人間には読みやすいが、機械処理が困難
  → grep で検索するしかない
  → 集計やダッシュボードに使いにくい

構造化ログ（推奨）:
  {
    "timestamp": "2025-01-15T10:30:45.123Z",
    "level": "error",
    "message": "Failed to process order",
    "service": "order-service",
    "traceId": "abc-123",
    "orderId": "12345",
    "userId": "abc",
    "error": {
      "name": "PaymentError",
      "message": "Insufficient funds",
      "code": "PAYMENT_INSUFFICIENT_FUNDS"
    },
    "duration_ms": 1234
  }

利点:
  → JSON で検索・フィルタリング可能
  → ダッシュボードで集計可能
  → 自動アラートのトリガーに
  → ELK Stack, CloudWatch Logs Insights 等で分析可能
  → 構造化されているため型安全にできる
```

### 1.2 TypeScript: pino による構造化ロガー

```typescript
// pino: 高性能構造化ロガー
import pino from 'pino';

// ロガーの設定
const logger = pino({
  level: process.env.LOG_LEVEL ?? 'info',
  formatters: {
    level(label) { return { level: label }; },
    bindings(bindings) {
      return {
        pid: bindings.pid,
        hostname: bindings.hostname,
        service: process.env.SERVICE_NAME ?? 'unknown',
        version: process.env.APP_VERSION ?? 'unknown',
        environment: process.env.NODE_ENV ?? 'development',
      };
    },
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  // 本番環境ではJSON、開発環境ではpretty
  transport: process.env.NODE_ENV === 'development'
    ? { target: 'pino-pretty', options: { colorize: true } }
    : undefined,
  // 機密情報の除去
  redact: {
    paths: [
      'req.headers.authorization',
      'req.headers.cookie',
      'req.body.password',
      'req.body.creditCard',
      'req.body.ssn',
      '*.password',
      '*.token',
      '*.secret',
    ],
    censor: '[REDACTED]',
  },
});

// 基本的な使い方
logger.info({ orderId: '12345', userId: 'abc' }, 'Order created');

logger.error(
  {
    orderId: '12345',
    error: { name: err.name, message: err.message, code: err.code },
    duration_ms: Date.now() - startTime,
  },
  'Order processing failed',
);

// 子ロガー: コンテキスト情報を自動付与
const orderLogger = logger.child({
  module: 'order-service',
  version: '2.1.0',
});

orderLogger.info({ orderId: '12345' }, 'Processing order');
// → { module: "order-service", version: "2.1.0", orderId: "12345", ... }
```

### 1.3 リクエストスコープのロガー

```typescript
// Express ミドルウェア: リクエストごとにロガーを作成
import { randomUUID } from 'crypto';
import { AsyncLocalStorage } from 'async_hooks';

// AsyncLocalStorage でリクエストコンテキストを管理
const als = new AsyncLocalStorage<{
  traceId: string;
  logger: pino.Logger;
}>();

// ミドルウェア
function requestLoggerMiddleware(req: Request, res: Response, next: NextFunction) {
  const traceId = req.headers['x-trace-id'] as string
    ?? req.headers['x-request-id'] as string
    ?? randomUUID();

  const requestLogger = logger.child({
    traceId,
    method: req.method,
    path: req.path,
    ip: req.ip,
    userAgent: req.headers['user-agent'],
  });

  // レスポンスヘッダーにトレースIDを設定
  res.setHeader('X-Trace-Id', traceId);

  // リクエスト開始ログ
  const startTime = Date.now();
  requestLogger.info('Request started');

  // レスポンス完了時のログ
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const logData = {
      statusCode: res.statusCode,
      duration_ms: duration,
      contentLength: res.getHeader('content-length'),
    };

    if (res.statusCode >= 500) {
      requestLogger.error(logData, 'Request completed with server error');
    } else if (res.statusCode >= 400) {
      requestLogger.warn(logData, 'Request completed with client error');
    } else {
      requestLogger.info(logData, 'Request completed');
    }
  });

  // AsyncLocalStorage にロガーを保存
  als.run({ traceId, logger: requestLogger }, () => {
    next();
  });
}

// どこからでもリクエストスコープのロガーを取得
function getLogger(): pino.Logger {
  const store = als.getStore();
  return store?.logger ?? logger;
}

function getTraceId(): string {
  const store = als.getStore();
  return store?.traceId ?? 'no-trace';
}

// サービス層での使用
class OrderService {
  async createOrder(data: CreateOrderInput): Promise<Order> {
    const log = getLogger();

    log.info({ data }, 'Creating order');

    try {
      const order = await this.repo.create(data);
      log.info({ orderId: order.id }, 'Order created successfully');
      return order;
    } catch (error) {
      log.error({ error, data }, 'Failed to create order');
      throw error;
    }
  }
}
```

### 1.4 Python: structlog による構造化ロガー

```python
import structlog
import logging
import json
from datetime import datetime


# structlog の設定
def configure_logging(environment: str = "production"):
    """構造化ログの設定"""

    # 共通のプロセッサー
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if environment == "development":
        # 開発環境: 色付きの人間可読フォーマット
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        )
    else:
        # 本番環境: JSON フォーマット
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )


# 使用例
logger = structlog.get_logger()

# 基本的なログ
logger.info("order_created", order_id="12345", user_id="abc", amount=1500)

# エラーログ
try:
    process_order(order)
except Exception as e:
    logger.error(
        "order_processing_failed",
        order_id=order.id,
        error=str(e),
        error_type=type(e).__name__,
        exc_info=True,
    )

# コンテキスト変数（リクエストスコープ）
import structlog.contextvars

# FastAPI ミドルウェア
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id", str(uuid.uuid4()))

        # コンテキスト変数にバインド
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            trace_id=trace_id,
            method=request.method,
            path=str(request.url.path),
            client_ip=request.client.host if request.client else "unknown",
        )

        log = structlog.get_logger()
        log.info("request_started")

        start = time.monotonic()
        try:
            response = await call_next(request)
            duration = (time.monotonic() - start) * 1000

            log.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration, 2),
            )
            response.headers["X-Trace-Id"] = trace_id
            return response
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            log.error(
                "request_failed",
                error=str(e),
                duration_ms=round(duration, 2),
                exc_info=True,
            )
            raise
```

### 1.5 Go: slog による構造化ロガー

```go
package main

import (
	"context"
	"log/slog"
	"os"
	"time"
)

// ロガーの設定
func setupLogger(env string) *slog.Logger {
	var handler slog.Handler

	if env == "production" {
		// 本番: JSON
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
	} else {
		// 開発: テキスト
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})
	}

	return slog.New(handler)
}

// リクエストスコープのロガー
type contextKey string

const loggerKey contextKey = "logger"

func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, loggerKey, logger)
}

func LoggerFrom(ctx context.Context) *slog.Logger {
	if logger, ok := ctx.Value(loggerKey).(*slog.Logger); ok {
		return logger
	}
	return slog.Default()
}

// HTTPミドルウェア
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		traceID := r.Header.Get("X-Trace-Id")
		if traceID == "" {
			traceID = uuid.New().String()
		}

		requestLogger := slog.Default().With(
			slog.String("trace_id", traceID),
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.String("remote_addr", r.RemoteAddr),
		)

		ctx := WithLogger(r.Context(), requestLogger)

		start := time.Now()
		requestLogger.Info("request started")

		// レスポンスラッパー
		rw := &responseWriter{ResponseWriter: w, statusCode: 200}
		next.ServeHTTP(rw, r.WithContext(ctx))

		duration := time.Since(start)
		requestLogger.Info("request completed",
			slog.Int("status_code", rw.statusCode),
			slog.Duration("duration", duration),
		)
	})
}

// サービス層での使用
func (s *OrderService) CreateOrder(ctx context.Context, input CreateOrderInput) (*Order, error) {
	logger := LoggerFrom(ctx)

	logger.Info("creating order",
		slog.String("user_id", input.UserID),
		slog.Int("item_count", len(input.Items)),
	)

	order, err := s.repo.Create(ctx, input)
	if err != nil {
		logger.Error("failed to create order",
			slog.String("error", err.Error()),
			slog.String("user_id", input.UserID),
		)
		return nil, err
	}

	logger.Info("order created",
		slog.String("order_id", order.ID),
		slog.Float64("total", order.Total),
	)
	return order, nil
}
```

---

## 2. ログレベル

### 2.1 ログレベルの定義と使い分け

```
┌─────────┬──────────────────────────────────────────────────────────┐
│ レベル  │ 用途                                                    │
├─────────┼──────────────────────────────────────────────────────────┤
│ fatal   │ アプリケーション停止を伴うエラー                         │
│         │ 例: DB接続不可、必須設定の欠如                           │
│         │ → 即座にアラート、オンコール対応                         │
├─────────┼──────────────────────────────────────────────────────────┤
│ error   │ 操作の失敗。ユーザーに影響するエラー                     │
│         │ 例: API呼び出し失敗、データ保存失敗                      │
│         │ → エラートラッキング（Sentry）に送信                     │
├─────────┼──────────────────────────────────────────────────────────┤
│ warn    │ 潜在的な問題。今は動くが注意が必要                       │
│         │ 例: 非推奨APIの使用、リトライ発生、閾値接近              │
│         │ → 定期的にチェック                                      │
├─────────┼──────────────────────────────────────────────────────────┤
│ info    │ 重要なビジネスイベント                                   │
│         │ 例: 注文完了、ユーザー登録、決済成功                     │
│         │ → ビジネスメトリクスの基盤                               │
├─────────┼──────────────────────────────────────────────────────────┤
│ debug   │ 開発時のデバッグ情報                                    │
│         │ 例: 変数の値、処理の分岐点                               │
│         │ → 本番では通常無効                                      │
├─────────┼──────────────────────────────────────────────────────────┤
│ trace   │ 詳細なトレース情報                                      │
│         │ 例: 関数の入出力、SQLクエリ、HTTP通信の詳細              │
│         │ → 問題調査時のみ一時的に有効化                           │
└─────────┴──────────────────────────────────────────────────────────┘

環境ごとの推奨レベル:
  本番:       info 以上
  ステージング: debug 以上
  開発:       trace 以上

動的なログレベル変更:
  → 本番で問題調査時に一時的に debug に変更
  → 環境変数やAPI経由で変更可能にする
  → 一定時間後に自動で元に戻す
```

### 2.2 ログレベルの判断基準

```typescript
// ログレベルの使い分けガイドライン

// FATAL: アプリケーションが起動・継続できない
logger.fatal({ port: 3000, error: err }, 'Failed to bind to port');
logger.fatal({ dsn: dbConfig.dsn }, 'Database connection failed on startup');

// ERROR: 操作が失敗した（ユーザーに影響がある）
logger.error({ orderId, error: err }, 'Failed to process payment');
logger.error({ userId, error: err }, 'Failed to send password reset email');

// WARN: 問題の予兆、ただし操作は完了した
logger.warn({ queueSize: 950, maxSize: 1000 }, 'Queue approaching capacity');
logger.warn({ attempt: 2, maxRetries: 3 }, 'Retry attempt for external API');
logger.warn({ deprecatedField: 'oldField' }, 'Deprecated field used in request');

// INFO: 重要なビジネスイベント
logger.info({ orderId, amount: 5000 }, 'Order completed');
logger.info({ userId, plan: 'premium' }, 'User upgraded subscription');
logger.info({ batch: 'daily-report', count: 1500 }, 'Batch processing completed');

// DEBUG: 開発・調査用の詳細情報
logger.debug({ userId, filters }, 'Searching users with filters');
logger.debug({ query, params, duration_ms: 45 }, 'SQL query executed');

// TRACE: 非常に詳細な情報
logger.trace({ headers, body }, 'Outgoing HTTP request');
logger.trace({ response, duration_ms: 123 }, 'Incoming HTTP response');
```

---

## 3. エラートラッキング

### 3.1 Sentry の設定と使用

```typescript
// Sentry: エラートラッキングの設定
import * as Sentry from '@sentry/node';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  release: process.env.APP_VERSION,
  serverName: process.env.HOSTNAME,

  // トレーシング設定
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  // エラーフィルタリング
  beforeSend(event, hint) {
    const error = hint?.originalException;

    // 4xxエラーはSentryに送信しない
    if (error instanceof AppError && error.statusCode < 500) {
      return null;
    }

    // 特定のエラーを除外
    if (error instanceof AbortError) {
      return null;
    }

    // 機密情報の除去
    if (event.request?.headers) {
      delete event.request.headers['authorization'];
      delete event.request.headers['cookie'];
    }

    return event;
  },

  // パンくずリストのフィルタリング
  beforeBreadcrumb(breadcrumb) {
    // 機密URLをフィルタ
    if (breadcrumb.category === 'http' && breadcrumb.data?.url) {
      const url = new URL(breadcrumb.data.url);
      if (url.pathname.includes('/auth/')) {
        breadcrumb.data.url = url.origin + '/auth/[redacted]';
      }
    }
    return breadcrumb;
  },

  // 統合設定
  integrations: [
    new Sentry.Integrations.Http({ tracing: true }),
    new Sentry.Integrations.Express({ app }),
    new Sentry.Integrations.Postgres(),
  ],
});

// エラーのキャプチャ
async function processOrder(orderId: string): Promise<void> {
  try {
    await doProcessOrder(orderId);
  } catch (error) {
    Sentry.withScope(scope => {
      // タグ: フィルタリング用
      scope.setTag('feature', 'order-processing');
      scope.setTag('order_type', 'standard');

      // コンテキスト: 詳細情報
      scope.setContext('order', {
        orderId,
        userId: currentUser.id,
        amount: order.totalAmount,
        itemCount: order.items.length,
      });

      // ユーザー情報
      scope.setUser({
        id: currentUser.id,
        email: currentUser.email,
        subscription: currentUser.plan,
      });

      // フィンガープリント: エラーのグルーピング
      scope.setFingerprint([
        'order-processing',
        error instanceof HttpError ? String(error.statusCode) : 'unknown',
      ]);

      // エラーレベル
      scope.setLevel('error');

      Sentry.captureException(error);
    });

    throw error;
  }
}
```

### 3.2 パフォーマンスモニタリング

```typescript
// Sentry: パフォーマンスモニタリング
import * as Sentry from '@sentry/node';

// カスタムトランザクション
async function processPayment(paymentData: PaymentData): Promise<PaymentResult> {
  return Sentry.startSpan(
    {
      name: 'processPayment',
      op: 'payment.process',
      attributes: {
        'payment.amount': paymentData.amount,
        'payment.currency': paymentData.currency,
      },
    },
    async (span) => {
      // 子スパン: バリデーション
      const validationResult = await Sentry.startSpan(
        { name: 'validatePayment', op: 'validation' },
        async () => validatePaymentData(paymentData),
      );

      // 子スパン: 外部API呼び出し
      const chargeResult = await Sentry.startSpan(
        {
          name: 'chargePaymentProvider',
          op: 'http.client',
          attributes: { 'http.url': 'https://api.stripe.com/v1/charges' },
        },
        async () => stripe.charges.create(paymentData),
      );

      // 子スパン: DB保存
      await Sentry.startSpan(
        { name: 'savePaymentRecord', op: 'db.query' },
        async () => db.payments.create({ data: chargeResult }),
      );

      return chargeResult;
    },
  );
}
```

### 3.3 Python: Sentry 統合

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.aiohttp import AioHttpIntegration


def init_sentry(dsn: str, environment: str, release: str):
    """Sentryの初期化"""
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=0.1 if environment == "production" else 1.0,
        profiles_sample_rate=0.1,  # プロファイリング
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            AioHttpIntegration(),
        ],
        before_send=before_send_filter,
    )


def before_send_filter(event, hint):
    """送信前フィルタリング"""
    exception = hint.get("exc_info")
    if exception:
        exc_type, exc_value, _ = exception
        # 4xxエラーは送信しない
        if isinstance(exc_value, AppError) and exc_value.status_code < 500:
            return None
    return event


# エラーキャプチャ
async def process_order(order_id: str):
    try:
        await do_process_order(order_id)
    except Exception as e:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("feature", "order-processing")
            scope.set_context("order", {
                "order_id": order_id,
                "user_id": current_user.id,
            })
            scope.set_user({
                "id": current_user.id,
                "email": current_user.email,
            })
            sentry_sdk.capture_exception(e)
        raise


# カスタムスパン
async def fetch_user_data(user_id: str) -> dict:
    with sentry_sdk.start_span(op="http.client", description="fetch user data"):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{USER_SERVICE_URL}/users/{user_id}") as resp:
                return await resp.json()
```

---

## 4. 分散トレーシング

### 4.1 OpenTelemetry の基礎

```
分散トレーシングの概念:
  → マイクロサービス間のリクエストを追跡
  → 1つのユーザーリクエストが複数のサービスを経由する場合に有用
  → ボトルネックの特定、エラーの発生箇所の特定

用語:
  Trace: 1つのリクエストの全体的な流れ
  Span: Trace 内の個々の処理単位
  Context: Span 間で伝播するメタデータ
  Baggage: サービス間で伝播するカスタムデータ

  例:
  Trace: ユーザーの注文リクエスト
    ├─ Span: API Gateway (10ms)
    ├─ Span: Order Service (200ms)
    │   ├─ Span: DB Query - Create Order (50ms)
    │   └─ Span: Payment Service Call (120ms)
    │       ├─ Span: Stripe API Call (80ms)
    │       └─ Span: DB Query - Save Payment (20ms)
    └─ Span: Notification Service (30ms)
        └─ Span: SendGrid API Call (25ms)
```

```typescript
// OpenTelemetry の設定
import { NodeSDK } from '@opentelemetry/sdk-node';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { Resource } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';

const sdk = new NodeSDK({
  resource: new Resource({
    [SemanticResourceAttributes.SERVICE_NAME]: 'order-service',
    [SemanticResourceAttributes.SERVICE_VERSION]: '1.0.0',
    [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV,
  }),
  traceExporter: new OTLPTraceExporter({
    url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT + '/v1/traces',
  }),
  metricReader: new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter({
      url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT + '/v1/metrics',
    }),
    exportIntervalMillis: 60000,
  }),
  instrumentations: [
    getNodeAutoInstrumentations({
      '@opentelemetry/instrumentation-http': {
        ignoreIncomingPaths: ['/health', '/metrics'],
      },
      '@opentelemetry/instrumentation-express': {},
      '@opentelemetry/instrumentation-pg': {},
    }),
  ],
});

sdk.start();

// カスタムスパン
import { trace, SpanStatusCode } from '@opentelemetry/api';

const tracer = trace.getTracer('order-service');

async function processOrder(order: Order): Promise<ProcessResult> {
  return tracer.startActiveSpan('processOrder', async (span) => {
    span.setAttribute('order.id', order.id);
    span.setAttribute('order.amount', order.totalAmount);
    span.setAttribute('order.item_count', order.items.length);

    try {
      // バリデーション
      await tracer.startActiveSpan('validateOrder', async (validationSpan) => {
        await validateOrder(order);
        validationSpan.setStatus({ code: SpanStatusCode.OK });
        validationSpan.end();
      });

      // 決済処理
      const paymentResult = await tracer.startActiveSpan(
        'processPayment',
        async (paymentSpan) => {
          paymentSpan.setAttribute('payment.provider', 'stripe');
          const result = await paymentService.charge(order);
          paymentSpan.setAttribute('payment.id', result.paymentId);
          paymentSpan.setStatus({ code: SpanStatusCode.OK });
          paymentSpan.end();
          return result;
        },
      );

      span.setStatus({ code: SpanStatusCode.OK });
      return { orderId: order.id, paymentId: paymentResult.paymentId };
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: (error as Error).message,
      });
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  });
}
```

### 4.2 トレースコンテキストの伝播

```typescript
// サービス間のトレースコンテキスト伝播

// HTTP ヘッダーで伝播（W3C Trace Context）
// traceparent: 00-trace_id-span_id-trace_flags
// tracestate: vendor-specific-data

// サービスAからサービスBを呼ぶ場合
import { context, propagation } from '@opentelemetry/api';

async function callOrderService(orderData: OrderInput): Promise<Order> {
  const headers: Record<string, string> = {};

  // 現在のコンテキストをHTTPヘッダーに注入
  propagation.inject(context.active(), headers);

  const response = await fetch('http://order-service/api/orders', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers, // traceparent, tracestate が含まれる
    },
    body: JSON.stringify(orderData),
  });

  return response.json();
}

// サービスBでコンテキストを抽出
function extractContextMiddleware(req: Request, res: Response, next: NextFunction) {
  // HTTPヘッダーからコンテキストを抽出
  const extractedContext = propagation.extract(context.active(), req.headers);

  // 抽出したコンテキストでリクエストを処理
  context.with(extractedContext, () => {
    next();
  });
}
```

---

## 5. メトリクス

### 5.1 Prometheus メトリクス

```typescript
// Prometheus メトリクスの収集
import { Counter, Histogram, Gauge, Summary, Registry } from 'prom-client';

const register = new Registry();

// デフォルトメトリクスの収集（CPU, メモリ等）
import { collectDefaultMetrics } from 'prom-client';
collectDefaultMetrics({ register });

// カスタムメトリクス

// カウンター: 単調増加する値
const httpRequestsTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'path', 'status_code'],
  registers: [register],
});

// ヒストグラム: 値の分布
const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration in seconds',
  labelNames: ['method', 'path', 'status_code'],
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  registers: [register],
});

// ゲージ: 上下する値
const activeConnections = new Gauge({
  name: 'active_connections',
  help: 'Number of active connections',
  registers: [register],
});

const queueSize = new Gauge({
  name: 'job_queue_size',
  help: 'Number of jobs in the queue',
  labelNames: ['queue_name'],
  registers: [register],
});

// サマリー: パーセンタイル
const dbQueryDuration = new Summary({
  name: 'db_query_duration_seconds',
  help: 'Database query duration in seconds',
  labelNames: ['query_type', 'table'],
  percentiles: [0.5, 0.9, 0.95, 0.99],
  registers: [register],
});

// ミドルウェアでメトリクスを収集
function metricsMiddleware(req: Request, res: Response, next: NextFunction) {
  const start = Date.now();
  activeConnections.inc();

  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    const labels = {
      method: req.method,
      path: req.route?.path ?? req.path,
      status_code: String(res.statusCode),
    };

    httpRequestsTotal.inc(labels);
    httpRequestDuration.observe(labels, duration);
    activeConnections.dec();
  });

  next();
}

// メトリクスエンドポイント
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// ビジネスメトリクス
const ordersCreated = new Counter({
  name: 'orders_created_total',
  help: 'Total number of orders created',
  labelNames: ['status', 'payment_method'],
  registers: [register],
});

const orderAmount = new Histogram({
  name: 'order_amount_jpy',
  help: 'Order amount in JPY',
  buckets: [100, 500, 1000, 5000, 10000, 50000, 100000],
  registers: [register],
});

// 使用例
async function createOrder(data: CreateOrderInput): Promise<Order> {
  const order = await orderRepo.create(data);

  ordersCreated.inc({
    status: 'success',
    payment_method: data.paymentMethod,
  });
  orderAmount.observe(order.totalAmount);

  return order;
}
```

### 5.2 RED メソッド

```
RED メソッド（サービスのモニタリング）:
  R - Rate:     リクエスト数/秒
  E - Errors:   エラー数/秒（またはエラー率）
  D - Duration: レイテンシ（P50, P95, P99）

  → マイクロサービスの健全性を3つのメトリクスで把握

USE メソッド（リソースのモニタリング）:
  U - Utilization: 使用率（CPU, メモリ, ディスク）
  S - Saturation:  飽和度（キュー長、待ちスレッド数）
  E - Errors:      エラー数

  → インフラリソースの健全性を把握

Four Golden Signals（Google SRE）:
  1. Latency:    レスポンス時間
  2. Traffic:    リクエスト数
  3. Errors:     エラー率
  4. Saturation: リソース飽和度
```

---

## 6. アラート設計

### 6.1 アラートの原則

```
アラートの原則:
  1. アクション可能（受けたら何かできる）
     → 「エラーが発生しました」ではなく「決済処理のエラー率が5%を超えました」
     → アラートにランブック（対応手順書）へのリンクを含める

  2. 低ノイズ（誤報が少ない）
     → アラート疲れ（Alert Fatigue）を防ぐ
     → 閾値は十分に吟味する
     → 一時的なスパイクに反応しすぎない

  3. 適切な宛先（オンコール担当者）
     → 緊急度に応じたエスカレーション
     → Critical: PagerDuty → SMS/電話
     → Warning: Slack通知 → 翌営業日対応

  4. コンテキスト付き
     → ダッシュボードへのリンク
     → 関連ログへのリンク
     → ランブックへのリンク
     → 影響範囲の概要
```

### 6.2 アラートルールの設計

```
エラー率ベース:
  → 5xxエラー率 > 1%（5分間）→ Warning
  → 5xxエラー率 > 5%（5分間）→ Critical
  → 特定エンドポイントのエラー率 > 10% → Critical

レイテンシベース:
  → P95 > 2秒（5分間）→ Warning
  → P99 > 5秒（5分間）→ Critical
  → P50 > 1秒（持続的）→ Warning（性能劣化の兆候）

ビジネスメトリクス:
  → 決済成功率 < 95%（10分間）→ Critical
  → 注文数が前時間比 50% 減 → Warning
  → 新規登録数が前日比 70% 減 → Warning

インフラメトリクス:
  → CPU使用率 > 80%（15分間）→ Warning
  → CPU使用率 > 95%（5分間）→ Critical
  → メモリ使用率 > 85% → Warning
  → ディスク使用率 > 90% → Critical
  → DB接続プール使用率 > 80% → Warning

サーキットブレーカー:
  → 任意のサーキットブレーカーが Open → Warning
  → 主要サービスのサーキットブレーカーが Open → Critical
```

### 6.3 Prometheus Alertmanager の設定

```yaml
# Prometheus アラートルール
groups:
  - name: http_alerts
    rules:
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status_code=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m]))
          > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High 5xx error rate ({{ $value | humanizePercentage }})"
          description: "5xxエラー率が5%を超えています"
          runbook_url: "https://wiki.example.com/runbooks/high-error-rate"
          dashboard: "https://grafana.example.com/d/http/overview"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High P95 latency ({{ $value | humanizeDuration }})"
          description: "P95レイテンシが2秒を超えています"

      - alert: HighQueueSize
        expr: job_queue_size > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Job queue size is high ({{ $value }})"
          description: "ジョブキューのサイズが1000を超えています"

  - name: business_alerts
    rules:
      - alert: LowPaymentSuccessRate
        expr: |
          sum(rate(payments_total{status="success"}[10m]))
          / sum(rate(payments_total[10m]))
          < 0.95
        for: 10m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "Payment success rate below 95% ({{ $value | humanizePercentage }})"
          description: "決済成功率が95%を下回っています"

# Alertmanager の設定
route:
  receiver: default
  group_by: [alertname, severity]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  routes:
    - match:
        severity: critical
      receiver: pagerduty-critical
      continue: true
    - match:
        severity: warning
      receiver: slack-warnings

receivers:
  - name: default
    slack_configs:
      - channel: '#alerts'
        send_resolved: true

  - name: pagerduty-critical
    pagerduty_configs:
      - service_key: '<pagerduty-key>'
        severity: critical

  - name: slack-warnings
    slack_configs:
      - channel: '#alerts-warnings'
        send_resolved: true
        title: '{{ .CommonAnnotations.summary }}'
        text: '{{ .CommonAnnotations.description }}'
```

---

## 7. ログのセキュリティとコンプライアンス

### 7.1 機密情報の取り扱い

```typescript
// 機密情報のマスキング
class LogSanitizer {
  private static readonly SENSITIVE_FIELDS = new Set([
    'password',
    'token',
    'secret',
    'authorization',
    'cookie',
    'creditCard',
    'ssn',
    'apiKey',
    'accessToken',
    'refreshToken',
  ]);

  private static readonly PII_PATTERNS = [
    // メールアドレス
    { pattern: /[\w.-]+@[\w.-]+\.\w+/g, replacement: '[email]' },
    // クレジットカード番号
    { pattern: /\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/g, replacement: '[card]' },
    // 電話番号
    { pattern: /\b0\d{1,4}[-\s]?\d{1,4}[-\s]?\d{3,4}\b/g, replacement: '[phone]' },
  ];

  static sanitize(obj: Record<string, any>): Record<string, any> {
    const sanitized: Record<string, any> = {};

    for (const [key, value] of Object.entries(obj)) {
      if (this.SENSITIVE_FIELDS.has(key.toLowerCase())) {
        sanitized[key] = '[REDACTED]';
      } else if (typeof value === 'object' && value !== null) {
        sanitized[key] = this.sanitize(value);
      } else if (typeof value === 'string') {
        let sanitizedValue = value;
        for (const { pattern, replacement } of this.PII_PATTERNS) {
          sanitizedValue = sanitizedValue.replace(pattern, replacement);
        }
        sanitized[key] = sanitizedValue;
      } else {
        sanitized[key] = value;
      }
    }

    return sanitized;
  }
}

// pino のシリアライザーに統合
const logger = pino({
  serializers: {
    req(req) {
      return LogSanitizer.sanitize({
        method: req.method,
        url: req.url,
        headers: req.headers,
        body: req.body,
      });
    },
    err(err) {
      return {
        type: err.constructor.name,
        message: err.message,
        code: err.code,
        stack: process.env.NODE_ENV !== 'production' ? err.stack : undefined,
      };
    },
  },
});
```

### 7.2 ログの保持とローテーション

```
ログ保持ポリシー:

  ホットストレージ（高速検索、高コスト）:
    → 直近 7-30 日
    → Elasticsearch, CloudWatch Logs
    → リアルタイム検索・分析

  ウォームストレージ（中速検索、中コスト）:
    → 30-90 日
    → S3 Standard, GCS Standard
    → 必要時にクエリ

  コールドストレージ（低速、低コスト）:
    → 90日 - 数年
    → S3 Glacier, GCS Coldline
    → コンプライアンス要件に基づく保持

  コンプライアンス要件:
    → GDPR: 個人データの保持期間制限
    → PCI DSS: 監査ログ1年以上保持
    → SOX: 財務関連ログ7年保持
    → HIPAA: 医療データ関連ログ6年保持
```

---

## 8. ダッシュボード設計

### 8.1 Grafana ダッシュボードの構成

```
推奨ダッシュボード構成:

  1. サービス概要ダッシュボード
     → リクエスト数/秒（Rate）
     → エラー率（Errors）
     → P50/P95/P99 レイテンシ（Duration）
     → アクティブ接続数
     → 直近のアラート一覧

  2. エンドポイント別ダッシュボード
     → エンドポイントごとのリクエスト数
     → エンドポイントごとのエラー率
     → エンドポイントごとのレイテンシ
     → トップ10 スロークエリ

  3. インフラダッシュボード
     → CPU使用率
     → メモリ使用率
     → ディスクI/O
     → ネットワークI/O
     → コンテナ数（Kubernetes）

  4. ビジネスダッシュボード
     → 注文数/時間
     → 決済成功率
     → 新規登録数
     → アクティブユーザー数

  5. 依存サービスダッシュボード
     → 各外部API のレスポンスタイム
     → サーキットブレーカーの状態
     → DB接続プールの使用率
     → キャッシュヒット率
```

### 8.2 SLI/SLO の設計

```
SLI（Service Level Indicator）:
  → サービスの品質を測定する指標
  → 例: 可用性、レイテンシ、スループット

SLO（Service Level Objective）:
  → SLI の目標値
  → 例: 可用性 99.9%、P95 レイテンシ < 200ms

エラーバジェット:
  → SLO を超えた分のエラーが許容される「予算」
  → 99.9% SLO = 月間約43分のダウンタイム予算
  → エラーバジェット消費時はリリースを抑制

実装例:
  SLI: 正常レスポンス率 = 200-499レスポンス / 全レスポンス
  SLO: 30日間で 99.9% 以上
  エラーバジェット: 0.1% = 約43分/月

  Prometheus クエリ:
    # 30日間のSLI
    sum(rate(http_requests_total{status_code!~"5.."}[30d]))
    / sum(rate(http_requests_total[30d]))

    # 残りエラーバジェット
    1 - (
      sum(increase(http_requests_total{status_code=~"5.."}[30d]))
      / (sum(increase(http_requests_total[30d])) * 0.001)
    )
```

---

## まとめ

| 手法 | 目的 | ツール例 |
|------|------|---------|
| 構造化ログ | 検索・分析可能なログ | pino, structlog, slog |
| エラートラッキング | エラーの集約・通知 | Sentry, Datadog, Bugsnag |
| 分散トレーシング | サービス間の追跡 | OpenTelemetry, Jaeger, Zipkin |
| メトリクス | 数値データの監視 | Prometheus, Grafana, Datadog |
| アラート | 異常の即時通知 | PagerDuty, Alertmanager, Opsgenie |
| ダッシュボード | 可視化 | Grafana, Kibana, Datadog |
| ログ管理 | ログ収集・検索 | ELK Stack, Loki, CloudWatch |

---

## 次に読むべきガイド
→ [[02-testing-async.md]] — 非同期テスト

---

## 参考文献
1. Sentry Documentation. docs.sentry.io.
2. Google SRE Book. "Monitoring Distributed Systems." O'Reilly, 2016.
3. OpenTelemetry Documentation. opentelemetry.io.
4. Prometheus Documentation. prometheus.io.
5. Grafana Documentation. grafana.com/docs.
6. Beyer, B. et al. "Site Reliability Engineering." O'Reilly, 2016.
7. pino Documentation. github.com/pinojs/pino.
8. structlog Documentation. structlog.org.
9. Go slog Documentation. pkg.go.dev/log/slog.
10. W3C Trace Context. w3.org/TR/trace-context.
