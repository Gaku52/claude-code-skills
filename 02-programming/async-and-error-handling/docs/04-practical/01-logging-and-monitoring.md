# Logging and Monitoring

> Errors will happen. What matters is "detecting them quickly, identifying the cause, and fixing them." This guide covers structured logging, error tracking (Sentry), and alert design best practices.

## What You Will Learn

- [ ] Understand structured logging design
- [ ] Learn how to use error tracking services
- [ ] Learn effective alert design
- [ ] Understand the basics of distributed tracing
- [ ] Master metrics collection and dashboard design
- [ ] Understand logging security and compliance

## Prerequisites

Before reading this guide, having the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [API Error Design](./00-api-error-design.md)

---

## 1. Structured Logging

### 1.1 Structured Logs vs Unstructured Logs

```
Unstructured Logs (traditional):
  [2025-01-15 10:30:45] ERROR: Failed to process order 12345 for user abc
  -> Human-readable but difficult for machine processing
  -> Can only search with grep
  -> Difficult to use for aggregation and dashboards

Structured Logs (recommended):
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

Benefits:
  -> Searchable and filterable as JSON
  -> Aggregatable in dashboards
  -> Can trigger automatic alerts
  -> Analyzable with ELK Stack, CloudWatch Logs Insights, etc.
  -> Can be made type-safe since it is structured
```

### 1.2 TypeScript: Structured Logger with pino

```typescript
// pino: High-performance structured logger
import pino from 'pino';

// Logger configuration
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
  // JSON in production, pretty in development
  transport: process.env.NODE_ENV === 'development'
    ? { target: 'pino-pretty', options: { colorize: true } }
    : undefined,
  // Remove sensitive information
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

// Basic usage
logger.info({ orderId: '12345', userId: 'abc' }, 'Order created');

logger.error(
  {
    orderId: '12345',
    error: { name: err.name, message: err.message, code: err.code },
    duration_ms: Date.now() - startTime,
  },
  'Order processing failed',
);

// Child logger: automatically add context information
const orderLogger = logger.child({
  module: 'order-service',
  version: '2.1.0',
});

orderLogger.info({ orderId: '12345' }, 'Processing order');
// -> { module: "order-service", version: "2.1.0", orderId: "12345", ... }
```

### 1.3 Request-Scoped Logger

```typescript
// Express middleware: create a logger per request
import { randomUUID } from 'crypto';
import { AsyncLocalStorage } from 'async_hooks';

// Manage request context with AsyncLocalStorage
const als = new AsyncLocalStorage<{
  traceId: string;
  logger: pino.Logger;
}>();

// Middleware
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

  // Set trace ID in response header
  res.setHeader('X-Trace-Id', traceId);

  // Request start log
  const startTime = Date.now();
  requestLogger.info('Request started');

  // Log on response completion
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

  // Store logger in AsyncLocalStorage
  als.run({ traceId, logger: requestLogger }, () => {
    next();
  });
}

// Retrieve request-scoped logger from anywhere
function getLogger(): pino.Logger {
  const store = als.getStore();
  return store?.logger ?? logger;
}

function getTraceId(): string {
  const store = als.getStore();
  return store?.traceId ?? 'no-trace';
}

// Usage in service layer
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

### 1.4 Python: Structured Logger with structlog

```python
import structlog
import logging
import json
from datetime import datetime


# Structured logging configuration
def configure_logging(environment: str = "production"):
    """Configure structured logging"""

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if environment == "development":
        # Development: colored human-readable format
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        )
    else:
        # Production: JSON format
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        )


# Usage example
logger = structlog.get_logger()

# Basic logging
logger.info("order_created", order_id="12345", user_id="abc", amount=1500)

# Error logging
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

# Context variables (request scope)
import structlog.contextvars

# FastAPI middleware
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("x-trace-id", str(uuid.uuid4()))

        # Bind to context variables
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

### 1.5 Go: Structured Logger with slog

```go
package main

import (
	"context"
	"log/slog"
	"os"
	"time"
)

// Logger configuration
func setupLogger(env string) *slog.Logger {
	var handler slog.Handler

	if env == "production" {
		// Production: JSON
		handler = slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelInfo,
		})
	} else {
		// Development: Text
		handler = slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
			Level: slog.LevelDebug,
		})
	}

	return slog.New(handler)
}

// Request-scoped logger
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

// HTTP middleware
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

		// Response wrapper
		rw := &responseWriter{ResponseWriter: w, statusCode: 200}
		next.ServeHTTP(rw, r.WithContext(ctx))

		duration := time.Since(start)
		requestLogger.Info("request completed",
			slog.Int("status_code", rw.statusCode),
			slog.Duration("duration", duration),
		)
	})
}

// Usage in service layer
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

## 2. Log Levels

### 2.1 Log Level Definitions and Usage

```
+---------+----------------------------------------------------------+
| Level   | Usage                                                    |
+---------+----------------------------------------------------------+
| fatal   | Errors that cause application shutdown                   |
|         | Example: DB connection failure, missing required config  |
|         | -> Immediate alert, on-call response                    |
+---------+----------------------------------------------------------+
| error   | Operation failure. Errors that affect users              |
|         | Example: API call failure, data save failure             |
|         | -> Send to error tracking (Sentry)                      |
+---------+----------------------------------------------------------+
| warn    | Potential issues. Working now but needs attention        |
|         | Example: Deprecated API usage, retry occurred,           |
|         |          approaching threshold                           |
|         | -> Check periodically                                    |
+---------+----------------------------------------------------------+
| info    | Important business events                                |
|         | Example: Order completed, user registered,               |
|         |          payment successful                              |
|         | -> Foundation for business metrics                       |
+---------+----------------------------------------------------------+
| debug   | Debug information for development                        |
|         | Example: Variable values, processing branch points       |
|         | -> Usually disabled in production                        |
+---------+----------------------------------------------------------+
| trace   | Detailed trace information                               |
|         | Example: Function I/O, SQL queries,                      |
|         |          HTTP communication details                      |
|         | -> Temporarily enabled only during investigation         |
+---------+----------------------------------------------------------+

Recommended levels by environment:
  Production:  info and above
  Staging:     debug and above
  Development: trace and above

Dynamic log level changes:
  -> Temporarily change to debug when investigating issues in production
  -> Make changeable via environment variables or API
  -> Automatically revert after a set period
```

### 2.2 Log Level Decision Criteria

```typescript
// Log level usage guidelines

// FATAL: Application cannot start or continue
logger.fatal({ port: 3000, error: err }, 'Failed to bind to port');
logger.fatal({ dsn: dbConfig.dsn }, 'Database connection failed on startup');

// ERROR: Operation failed (user is affected)
logger.error({ orderId, error: err }, 'Failed to process payment');
logger.error({ userId, error: err }, 'Failed to send password reset email');

// WARN: Signs of problems, but operation completed
logger.warn({ queueSize: 950, maxSize: 1000 }, 'Queue approaching capacity');
logger.warn({ attempt: 2, maxRetries: 3 }, 'Retry attempt for external API');
logger.warn({ deprecatedField: 'oldField' }, 'Deprecated field used in request');

// INFO: Important business events
logger.info({ orderId, amount: 5000 }, 'Order completed');
logger.info({ userId, plan: 'premium' }, 'User upgraded subscription');
logger.info({ batch: 'daily-report', count: 1500 }, 'Batch processing completed');

// DEBUG: Detailed information for development and investigation
logger.debug({ userId, filters }, 'Searching users with filters');
logger.debug({ query, params, duration_ms: 45 }, 'SQL query executed');

// TRACE: Very detailed information
logger.trace({ headers, body }, 'Outgoing HTTP request');
logger.trace({ response, duration_ms: 123 }, 'Incoming HTTP response');
```

---

## 3. Error Tracking

### 3.1 Sentry Setup and Usage

```typescript
// Sentry: Error tracking configuration
import * as Sentry from '@sentry/node';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  release: process.env.APP_VERSION,
  serverName: process.env.HOSTNAME,

  // Tracing configuration
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,

  // Error filtering
  beforeSend(event, hint) {
    const error = hint?.originalException;

    // Do not send 4xx errors to Sentry
    if (error instanceof AppError && error.statusCode < 500) {
      return null;
    }

    // Exclude specific errors
    if (error instanceof AbortError) {
      return null;
    }

    // Remove sensitive information
    if (event.request?.headers) {
      delete event.request.headers['authorization'];
      delete event.request.headers['cookie'];
    }

    return event;
  },

  // Breadcrumb filtering
  beforeBreadcrumb(breadcrumb) {
    // Filter sensitive URLs
    if (breadcrumb.category === 'http' && breadcrumb.data?.url) {
      const url = new URL(breadcrumb.data.url);
      if (url.pathname.includes('/auth/')) {
        breadcrumb.data.url = url.origin + '/auth/[redacted]';
      }
    }
    return breadcrumb;
  },

  // Integration configuration
  integrations: [
    new Sentry.Integrations.Http({ tracing: true }),
    new Sentry.Integrations.Express({ app }),
    new Sentry.Integrations.Postgres(),
  ],
});

// Capturing errors
async function processOrder(orderId: string): Promise<void> {
  try {
    await doProcessOrder(orderId);
  } catch (error) {
    Sentry.withScope(scope => {
      // Tags: for filtering
      scope.setTag('feature', 'order-processing');
      scope.setTag('order_type', 'standard');

      // Context: detailed information
      scope.setContext('order', {
        orderId,
        userId: currentUser.id,
        amount: order.totalAmount,
        itemCount: order.items.length,
      });

      // User information
      scope.setUser({
        id: currentUser.id,
        email: currentUser.email,
        subscription: currentUser.plan,
      });

      // Fingerprint: error grouping
      scope.setFingerprint([
        'order-processing',
        error instanceof HttpError ? String(error.statusCode) : 'unknown',
      ]);

      // Error level
      scope.setLevel('error');

      Sentry.captureException(error);
    });

    throw error;
  }
}
```

### 3.2 Performance Monitoring

```typescript
// Sentry: Performance monitoring
import * as Sentry from '@sentry/node';

// Custom transaction
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
      // Child span: validation
      const validationResult = await Sentry.startSpan(
        { name: 'validatePayment', op: 'validation' },
        async () => validatePaymentData(paymentData),
      );

      // Child span: external API call
      const chargeResult = await Sentry.startSpan(
        {
          name: 'chargePaymentProvider',
          op: 'http.client',
          attributes: { 'http.url': 'https://api.stripe.com/v1/charges' },
        },
        async () => stripe.charges.create(paymentData),
      );

      // Child span: DB save
      await Sentry.startSpan(
        { name: 'savePaymentRecord', op: 'db.query' },
        async () => db.payments.create({ data: chargeResult }),
      );

      return chargeResult;
    },
  );
}
```

### 3.3 Python: Sentry Integration

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
from sentry_sdk.integrations.aiohttp import AioHttpIntegration


def init_sentry(dsn: str, environment: str, release: str):
    """Initialize Sentry"""
    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        traces_sample_rate=0.1 if environment == "production" else 1.0,
        profiles_sample_rate=0.1,  # Profiling
        integrations=[
            FastApiIntegration(),
            SqlalchemyIntegration(),
            AioHttpIntegration(),
        ],
        before_send=before_send_filter,
    )


def before_send_filter(event, hint):
    """Pre-send filtering"""
    exception = hint.get("exc_info")
    if exception:
        exc_type, exc_value, _ = exception
        # Do not send 4xx errors
        if isinstance(exc_value, AppError) and exc_value.status_code < 500:
            return None
    return event


# Error capture
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


# Custom span
async def fetch_user_data(user_id: str) -> dict:
    with sentry_sdk.start_span(op="http.client", description="fetch user data"):
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{USER_SERVICE_URL}/users/{user_id}") as resp:
                return await resp.json()
```

---

## 4. Distributed Tracing

### 4.1 OpenTelemetry Basics

```
Distributed Tracing Concepts:
  -> Track requests across microservices
  -> Useful when a single user request passes through multiple services
  -> Identify bottlenecks and error locations

Terminology:
  Trace: The overall flow of a single request
  Span: An individual unit of work within a Trace
  Context: Metadata propagated between Spans
  Baggage: Custom data propagated between services

  Example:
  Trace: User's order request
    +-- Span: API Gateway (10ms)
    +-- Span: Order Service (200ms)
    |   +-- Span: DB Query - Create Order (50ms)
    |   +-- Span: Payment Service Call (120ms)
    |       +-- Span: Stripe API Call (80ms)
    |       +-- Span: DB Query - Save Payment (20ms)
    +-- Span: Notification Service (30ms)
        +-- Span: SendGrid API Call (25ms)
```

```typescript
// OpenTelemetry configuration
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

// Custom spans
import { trace, SpanStatusCode } from '@opentelemetry/api';

const tracer = trace.getTracer('order-service');

async function processOrder(order: Order): Promise<ProcessResult> {
  return tracer.startActiveSpan('processOrder', async (span) => {
    span.setAttribute('order.id', order.id);
    span.setAttribute('order.amount', order.totalAmount);
    span.setAttribute('order.item_count', order.items.length);

    try {
      // Validation
      await tracer.startActiveSpan('validateOrder', async (validationSpan) => {
        await validateOrder(order);
        validationSpan.setStatus({ code: SpanStatusCode.OK });
        validationSpan.end();
      });

      // Payment processing
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

### 4.2 Trace Context Propagation

```typescript
// Trace context propagation between services

// Propagation via HTTP headers (W3C Trace Context)
// traceparent: 00-trace_id-span_id-trace_flags
// tracestate: vendor-specific-data

// When calling Service B from Service A
import { context, propagation } from '@opentelemetry/api';

async function callOrderService(orderData: OrderInput): Promise<Order> {
  const headers: Record<string, string> = {};

  // Inject current context into HTTP headers
  propagation.inject(context.active(), headers);

  const response = await fetch('http://order-service/api/orders', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...headers, // Contains traceparent, tracestate
    },
    body: JSON.stringify(orderData),
  });

  return response.json();
}

// Extract context in Service B
function extractContextMiddleware(req: Request, res: Response, next: NextFunction) {
  // Extract context from HTTP headers
  const extractedContext = propagation.extract(context.active(), req.headers);

  // Process the request with the extracted context
  context.with(extractedContext, () => {
    next();
  });
}
```

---

## 5. Metrics

### 5.1 Prometheus Metrics

```typescript
// Prometheus metrics collection
import { Counter, Histogram, Gauge, Summary, Registry } from 'prom-client';

const register = new Registry();

// Collect default metrics (CPU, memory, etc.)
import { collectDefaultMetrics } from 'prom-client';
collectDefaultMetrics({ register });

// Custom metrics

// Counter: monotonically increasing value
const httpRequestsTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'path', 'status_code'],
  registers: [register],
});

// Histogram: distribution of values
const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration in seconds',
  labelNames: ['method', 'path', 'status_code'],
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  registers: [register],
});

// Gauge: value that goes up and down
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

// Summary: percentiles
const dbQueryDuration = new Summary({
  name: 'db_query_duration_seconds',
  help: 'Database query duration in seconds',
  labelNames: ['query_type', 'table'],
  percentiles: [0.5, 0.9, 0.95, 0.99],
  registers: [register],
});

// Collect metrics with middleware
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

// Metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// Business metrics
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

// Usage example
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

### 5.2 RED Method

```
RED Method (Service Monitoring):
  R - Rate:     Requests per second
  E - Errors:   Errors per second (or error rate)
  D - Duration: Latency (P50, P95, P99)

  -> Understand service health with 3 metrics

USE Method (Resource Monitoring):
  U - Utilization: Usage rate (CPU, memory, disk)
  S - Saturation:  Saturation (queue length, waiting threads)
  E - Errors:      Error count

  -> Understand infrastructure resource health

Four Golden Signals (Google SRE):
  1. Latency:    Response time
  2. Traffic:    Request count
  3. Errors:     Error rate
  4. Saturation: Resource saturation
```

---

## 6. Alert Design

### 6.1 Alert Principles

```
Alert Principles:
  1. Actionable (you can do something when you receive it)
     -> Not "An error occurred" but "Payment processing error rate exceeded 5%"
     -> Include a link to a runbook (response procedure document) in the alert

  2. Low Noise (few false positives)
     -> Prevent Alert Fatigue
     -> Carefully consider thresholds
     -> Do not overreact to temporary spikes

  3. Appropriate Recipient (on-call personnel)
     -> Escalation based on urgency
     -> Critical: PagerDuty -> SMS/Phone
     -> Warning: Slack notification -> Handle next business day

  4. Context-Rich
     -> Link to dashboard
     -> Link to related logs
     -> Link to runbook
     -> Overview of impact scope
```

### 6.2 Alert Rule Design

```
Error Rate Based:
  -> 5xx error rate > 1% (5 minutes) -> Warning
  -> 5xx error rate > 5% (5 minutes) -> Critical
  -> Specific endpoint error rate > 10% -> Critical

Latency Based:
  -> P95 > 2 seconds (5 minutes) -> Warning
  -> P99 > 5 seconds (5 minutes) -> Critical
  -> P50 > 1 second (sustained) -> Warning (sign of performance degradation)

Business Metrics:
  -> Payment success rate < 95% (10 minutes) -> Critical
  -> Order count 50% less than previous hour -> Warning
  -> New registrations 70% less than previous day -> Warning

Infrastructure Metrics:
  -> CPU usage > 80% (15 minutes) -> Warning
  -> CPU usage > 95% (5 minutes) -> Critical
  -> Memory usage > 85% -> Warning
  -> Disk usage > 90% -> Critical
  -> DB connection pool usage > 80% -> Warning

Circuit Breaker:
  -> Any circuit breaker is Open -> Warning
  -> Major service circuit breaker is Open -> Critical
```

### 6.3 Prometheus Alertmanager Configuration

```yaml
# Prometheus alert rules
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
          description: "5xx error rate has exceeded 5%"
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
          description: "P95 latency has exceeded 2 seconds"

      - alert: HighQueueSize
        expr: job_queue_size > 1000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Job queue size is high ({{ $value }})"
          description: "Job queue size has exceeded 1000"

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
          description: "Payment success rate has dropped below 95%"

# Alertmanager configuration
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

## 7. Logging Security and Compliance

### 7.1 Handling Sensitive Information

```typescript
// Sensitive information masking
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
    // Email addresses
    { pattern: /[\w.-]+@[\w.-]+\.\w+/g, replacement: '[email]' },
    // Credit card numbers
    { pattern: /\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b/g, replacement: '[card]' },
    // Phone numbers
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

// Integrate with pino serializers
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

### 7.2 Log Retention and Rotation

```
Log Retention Policy:

  Hot Storage (fast search, high cost):
    -> Last 7-30 days
    -> Elasticsearch, CloudWatch Logs
    -> Real-time search and analysis

  Warm Storage (medium-speed search, medium cost):
    -> 30-90 days
    -> S3 Standard, GCS Standard
    -> Query when needed

  Cold Storage (slow, low cost):
    -> 90 days to several years
    -> S3 Glacier, GCS Coldline
    -> Retention based on compliance requirements

  Compliance Requirements:
    -> GDPR: Retention period limits for personal data
    -> PCI DSS: Audit logs retained for 1+ years
    -> SOX: Financial-related logs retained for 7 years
    -> HIPAA: Medical data-related logs retained for 6 years
```

---

## 8. Dashboard Design

### 8.1 Grafana Dashboard Structure

```
Recommended Dashboard Structure:

  1. Service Overview Dashboard
     -> Requests per second (Rate)
     -> Error rate (Errors)
     -> P50/P95/P99 latency (Duration)
     -> Active connections
     -> Recent alert list

  2. Endpoint-Specific Dashboard
     -> Request count per endpoint
     -> Error rate per endpoint
     -> Latency per endpoint
     -> Top 10 slow queries

  3. Infrastructure Dashboard
     -> CPU usage
     -> Memory usage
     -> Disk I/O
     -> Network I/O
     -> Container count (Kubernetes)

  4. Business Dashboard
     -> Orders per hour
     -> Payment success rate
     -> New registrations
     -> Active users

  5. Dependency Services Dashboard
     -> Response time of each external API
     -> Circuit breaker status
     -> DB connection pool usage
     -> Cache hit rate
```

### 8.2 SLI/SLO Design

```
SLI (Service Level Indicator):
  -> Metrics that measure service quality
  -> Example: Availability, latency, throughput

SLO (Service Level Objective):
  -> Target values for SLIs
  -> Example: 99.9% availability, P95 latency < 200ms

Error Budget:
  -> The "budget" of errors allowed beyond the SLO
  -> 99.9% SLO = approximately 43 minutes of downtime budget per month
  -> Restrict releases when error budget is consumed

Implementation Example:
  SLI: Successful response rate = 200-499 responses / total responses
  SLO: 99.9% or more over 30 days
  Error Budget: 0.1% = approximately 43 minutes/month

  Prometheus Queries:
    # 30-day SLI
    sum(rate(http_requests_total{status_code!~"5.."}[30d]))
    / sum(rate(http_requests_total[30d]))

    # Remaining error budget
    1 - (
      sum(increase(http_requests_total{status_code=~"5.."}[30d]))
      / (sum(increase(http_requests_total[30d])) * 0.001)
    )
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Create test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient: {slow_time:.4f}s")
    print(f"Efficient:   {fast_time:.6f}s")
    print(f"Speedup:     {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be conscious of algorithm complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify the path and format of configuration files |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Increased data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Verify user permissions, review configuration |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Steps

1. **Check error messages**: Read the stack trace and identify the location of occurrence
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Form hypotheses**: List possible causes
4. **Verify step by step**: Use log output or debuggers to verify hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas as well

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function input and output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Check the status of disk and network I/O
4. **Check concurrent connections**: Check the status of the connection pool

| Problem Type | Diagnostic Tool | Countermeasure |
|-------------|-----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Method | Purpose | Example Tools |
|--------|---------|---------------|
| Structured Logging | Searchable and analyzable logs | pino, structlog, slog |
| Error Tracking | Error aggregation and notification | Sentry, Datadog, Bugsnag |
| Distributed Tracing | Cross-service tracking | OpenTelemetry, Jaeger, Zipkin |
| Metrics | Numerical data monitoring | Prometheus, Grafana, Datadog |
| Alerting | Immediate anomaly notification | PagerDuty, Alertmanager, Opsgenie |
| Dashboards | Visualization | Grafana, Kibana, Datadog |
| Log Management | Log collection and search | ELK Stack, Loki, CloudWatch |

---

## Recommended Next Guides

---

## References
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
