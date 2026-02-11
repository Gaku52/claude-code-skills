# 監視とロギング

> API監視はサービス品質の可視化。エラー率、レイテンシ、スループットの計測、構造化ログ、分散トレーシング、アラート設計まで、プロダクションAPIの安定運用に必要な監視体制を構築する。

## この章で学ぶこと

- [ ] APIの主要なメトリクスとSLI/SLOを理解する
- [ ] 構造化ログと分散トレーシングの実装を把握する
- [ ] アラート設計とインシデント対応を学ぶ

---

## 1. 主要メトリクス

```
RED メソッド（API向け）:

  R — Rate（リクエストレート）:
     → リクエスト数/秒（RPS, QPS）
     → エンドポイント別、ステータスコード別

  E — Errors（エラー率）:
     → 5xx エラーの割合
     → 4xx エラーの割合（クライアント起因）
     → タイムアウト率

  D — Duration（レイテンシ）:
     → P50（中央値）、P95、P99
     → エンドポイント別のレイテンシ

追加メトリクス:
  → スループット: 処理データ量/秒
  → 同時接続数: アクティブな接続数
  → キューサイズ: 待ちリクエスト数
  → DB クエリ数: 1リクエストあたりのクエリ数
  → 外部API呼び出し: 依存サービスへのリクエスト

SLI（Service Level Indicator）:
  → 可用性:  成功レスポンス / 全レスポンス
  → レイテンシ: P99 < 500ms のリクエスト割合
  → エラー率: 5xx / 全レスポンス

SLO（Service Level Objective）:
  → 可用性:  99.9%（月間43分のダウンタイム許容）
  → レイテンシ: P99 < 500ms を 99% の時間達成
  → エラー率: < 0.1%

エラーバジェット:
  SLO 99.9% → エラーバジェット 0.1%
  月間リクエスト 100万 → 1000リクエストまで失敗OK
  → バジェット消費率で新機能リリースの判断
```

---

## 2. 構造化ログ

```javascript
// 構造化ログ（JSON形式）
import pino from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
});

// リクエストログ ミドルウェア
function requestLogger(req, res, next) {
  const requestId = req.headers['x-request-id'] || crypto.randomUUID();
  const startTime = performance.now();

  req.requestId = requestId;
  req.log = logger.child({ requestId });

  // レスポンス完了時にログ
  res.on('finish', () => {
    const duration = performance.now() - startTime;
    const logData = {
      method: req.method,
      path: req.originalUrl,
      statusCode: res.statusCode,
      duration: Math.round(duration),
      userAgent: req.headers['user-agent'],
      ip: req.ip,
      userId: req.user?.sub,
      contentLength: res.getHeader('content-length'),
    };

    if (res.statusCode >= 500) {
      req.log.error(logData, 'Server error');
    } else if (res.statusCode >= 400) {
      req.log.warn(logData, 'Client error');
    } else {
      req.log.info(logData, 'Request completed');
    }
  });

  next();
}

// 出力例:
// {
//   "level": "info",
//   "time": "2024-01-15T10:30:00.000Z",
//   "requestId": "req_abc123",
//   "method": "GET",
//   "path": "/api/v1/users?limit=20",
//   "statusCode": 200,
//   "duration": 45,
//   "userId": "user_123"
// }
```

---

## 3. 分散トレーシング

```
分散トレーシング:
  → マイクロサービス間のリクエストフローを追跡
  → 1つのリクエストが複数サービスを横断する際の可視化

トレースの構造:
  Trace（トレース）: リクエスト全体
  └── Span（スパン）: 各サービスでの処理単位
      ├── Span: API Gateway（5ms）
      ├── Span: User Service（15ms）
      │   ├── Span: DB Query（8ms）
      │   └── Span: Cache Check（2ms）
      └── Span: Order Service（20ms）
          └── Span: DB Query（12ms）

ヘッダー伝搬:
  traceparent: 00-trace_id-span_id-01
  tracestate: vendor=value

  W3C Trace Context（標準）:
  traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
                 |  |                                |                  |
                 ver trace-id (128bit)              span-id (64bit)   flags
```

```javascript
// OpenTelemetry セットアップ
import { NodeSDK } from '@opentelemetry/sdk-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';

const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://localhost:4318/v1/traces',
  }),
  instrumentations: [
    new HttpInstrumentation(),
    new ExpressInstrumentation(),
  ],
});

sdk.start();

// カスタムスパンの追加
import { trace } from '@opentelemetry/api';

const tracer = trace.getTracer('user-service');

async function getUser(id) {
  return tracer.startActiveSpan('getUser', async (span) => {
    span.setAttribute('user.id', id);
    try {
      const user = await db.users.findOne({ id });
      span.setAttribute('user.found', !!user);
      return user;
    } catch (error) {
      span.recordException(error);
      span.setStatus({ code: 2, message: error.message });
      throw error;
    } finally {
      span.end();
    }
  });
}
```

---

## 4. メトリクス収集

```javascript
// Prometheus メトリクス（prom-client）
import { Registry, Counter, Histogram, Gauge, collectDefaultMetrics } from 'prom-client';

const registry = new Registry();
collectDefaultMetrics({ register: registry });

// リクエストカウンター
const httpRequestTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'path', 'status_code'],
  registers: [registry],
});

// レイテンシ ヒストグラム
const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration in seconds',
  labelNames: ['method', 'path', 'status_code'],
  buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  registers: [registry],
});

// アクティブ接続数
const activeConnections = new Gauge({
  name: 'http_active_connections',
  help: 'Number of active HTTP connections',
  registers: [registry],
});

// ミドルウェア
function metricsMiddleware(req, res, next) {
  activeConnections.inc();
  const end = httpRequestDuration.startTimer();

  res.on('finish', () => {
    const labels = {
      method: req.method,
      path: req.route?.path || req.path,
      status_code: res.statusCode,
    };
    httpRequestTotal.inc(labels);
    end(labels);
    activeConnections.dec();
  });

  next();
}

// メトリクスエンドポイント
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', registry.contentType);
  res.end(await registry.metrics());
});
```

---

## 5. アラート設計

```
アラート設計の原則:

  ① SLO ベースのアラート:
     → エラーバジェットの消費率でアラート
     → 瞬間的なスパイクでは鳴らない
     → 持続的な品質低下を検出

  ② 重大度レベル:

     Critical（即座に対応）:
     → 可用性 < 99%（SLO違反）
     → P99 > 5秒
     → 5xxエラー率 > 5%
     → DB接続不可

     Warning（営業時間内に対応）:
     → エラーバジェット消費率 > 50%/日
     → P99 > 1秒
     → 5xxエラー率 > 1%
     → ディスク使用率 > 80%

     Info（記録のみ）:
     → デプロイ完了
     → 新しいAPIバージョンへの移行状況
     → レート制限ヒット率の上昇

  ③ アラート疲れを避ける:
     → アクション可能なアラートのみ
     → 同じ問題で複数アラートを出さない（グループ化）
     → 自動復旧するものはアラートを出さない
     → エスカレーションルールを定義
```

---

## 6. ダッシュボード設計

```
API ダッシュボードの構成:

  Overview パネル:
  → リクエストレート（RPS）のグラフ
  → エラー率のグラフ
  → P50 / P95 / P99 レイテンシ
  → アクティブ接続数

  エンドポイント別パネル:
  → 各エンドポイントのRPS
  → 各エンドポイントのレイテンシ
  → 最も遅いエンドポイント Top 10

  エラー分析パネル:
  → エラーコード別の分布
  → エラーレートの時系列
  → 最新のエラーログ

  依存サービスパネル:
  → 外部API呼び出しのレイテンシ
  → DB クエリのレイテンシ
  → キャッシュヒット率

ツール:
  Grafana + Prometheus: メトリクス可視化
  Jaeger / Zipkin: 分散トレーシング
  Elasticsearch + Kibana: ログ分析
  Datadog / New Relic: オールインワン
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| RED | Rate, Errors, Duration を監視 |
| SLI/SLO | 可用性 99.9%、P99 < 500ms |
| ログ | 構造化JSON + requestId |
| トレーシング | OpenTelemetry + W3C Trace Context |
| アラート | SLOベース、アクション可能なもののみ |

---

## 次に読むべきガイド
→ [[02-api-gateway.md]] — APIゲートウェイ

---

## 参考文献
1. Google. "Site Reliability Engineering." sre.google, 2024.
2. OpenTelemetry. "Documentation." opentelemetry.io, 2024.
3. Prometheus. "Monitoring and Alerting." prometheus.io, 2024.
