# APIモニタリング・ロギング

> API監視はサービス品質の可視化と安定運用の基盤である。エラー率、レイテンシ、スループットの計測から構造化ログ、分散トレーシング、Prometheus/Grafanaによるメトリクス可視化、OpenTelemetryによるオブザーバビリティ統合まで、プロダクションAPIの信頼性を支える監視体制を体系的に解説する。

## この章で学ぶこと

- [ ] APIの主要メトリクス（RED/USE）とSLI/SLO/SLAの関係を理解する
- [ ] 構造化ログの設計原則と実装パターンを習得する
- [ ] 分散トレーシングの仕組みとOpenTelemetryによる実装を把握する
- [ ] Prometheus + Grafanaによるメトリクス収集・可視化を構築できる
- [ ] アラート設計とインシデント対応のベストプラクティスを学ぶ
- [ ] ログ集約基盤（ELK/Loki）の選定と構築方法を理解する

---

## 前提知識

- APIテストの基本 → 参照: [APIテスト](./00-api-testing.md)
- HTTPステータスコードの理解 → 参照: [HTTPの基礎](../../network-fundamentals/docs/02-http/00-http-basics.md)
- ログの基本概念（構造化ログ、ログレベル）

---

## 1. オブザーバビリティの三本柱

現代のAPIモニタリングは「オブザーバビリティ（Observability）」という概念を中核に据えている。オブザーバビリティとは、システムの外部出力（ログ、メトリクス、トレース）からシステム内部の状態を推測できる能力のことである。

```
+===========================================================================+
|                     オブザーバビリティの三本柱                                |
+===========================================================================+
|                                                                           |
|   +-------------------+  +-------------------+  +-------------------+     |
|   |      Logs         |  |     Metrics       |  |     Traces        |     |
|   |   (ログ)          |  |   (メトリクス)     |  |   (トレース)      |     |
|   +-------------------+  +-------------------+  +-------------------+     |
|   | - 離散イベント     |  | - 数値データ       |  | - リクエスト追跡   |     |
|   | - テキスト/JSON    |  | - 時系列集約      |  | - サービス横断    |     |
|   | - デバッグ向き     |  | - アラート向き     |  | - 依存関係把握    |     |
|   | - 高カーディナリティ|  | - 低オーバーヘッド |  | - ボトルネック特定 |     |
|   +--------+----------+  +--------+----------+  +--------+----------+     |
|            |                       |                       |              |
|            +----------+------------+-----------+-----------+              |
|                       |                        |                         |
|              +--------v--------+     +---------v--------+                |
|              |   Correlation   |     |   Exemplars      |                |
|              | (相関付け)       |     | (代表サンプル)    |                |
|              +-----------------+     +------------------+                |
|                       |                        |                         |
|              +--------v------------------------v--------+                |
|              |        統合オブザーバビリティ基盤          |                |
|              |  (Grafana / Datadog / New Relic / Splunk) |                |
|              +------------------------------------------+                |
+===========================================================================+
```

### 1.1 各柱の役割と使い分け

| 観点 | ログ (Logs) | メトリクス (Metrics) | トレース (Traces) |
|------|-------------|---------------------|-------------------|
| データ形式 | テキスト/構造化JSON | 数値（カウンタ/ゲージ/ヒストグラム） | スパンのツリー構造 |
| 粒度 | 個別イベント単位 | 集約された統計値 | リクエスト単位のフロー |
| ストレージコスト | 高（全イベント保存） | 低（集約値のみ） | 中（サンプリング可能） |
| 主な用途 | デバッグ、監査 | アラート、容量計画 | ボトルネック特定、依存分析 |
| 代表ツール | Elasticsearch, Loki | Prometheus, InfluxDB | Jaeger, Zipkin, Tempo |
| カーディナリティ | 非常に高い | 低〜中 | 中〜高 |
| リアルタイム性 | 秒単位 | 秒〜分単位 | 秒単位 |
| 保持期間の目安 | 30〜90日 | 13ヶ月（長期トレンド） | 7〜30日 |

### 1.2 三本柱の相関付け

オブザーバビリティの真価は、三本柱を相互に関連付けることで発揮される。たとえば、メトリクスでレイテンシの異常を検出した場合、そのタイミングのトレースを確認してボトルネックとなっているサービスを特定し、該当サービスのログから根本原因を突き止めるという流れが理想的なトラブルシューティングフローとなる。

```
トラブルシューティングフロー:

  [Grafana Dashboard]            [Jaeger / Tempo]           [Elasticsearch / Loki]
  メトリクス異常検出              トレース分析                ログ詳細調査
        |                              |                          |
        v                              v                          v
  P99レイテンシが                 遅延スパンを特定            エラーの根本原因を
  閾値を超過                      (DB Query: 3.2s)           ログから特定
        |                              |                          |
        +------> trace_id で紐付け ----+----> request_id で紐付け +
        |                              |                          |
        v                              v                          v
  exemplar から                   span の属性から             スタックトレースと
  該当 trace_id を取得            サービス名・操作を特定      コンテキスト情報を確認
```

---

## 2. 主要メトリクスの設計

### 2.1 RED メソッド（リクエスト駆動型サービス向け）

REDメソッドはTom Wilkie（Grafana Labs）が提唱した、リクエスト駆動型サービスのモニタリング手法である。API サービスのモニタリングに最適な方法論として広く採用されている。

```
RED メソッド（API向け主要メトリクス）:

  R — Rate（リクエストレート）:
     定義: 単位時間あたりのリクエスト数
     指標:
       -> リクエスト数/秒（RPS, QPS）
       -> エンドポイント別の内訳
       -> ステータスコード別の内訳
       -> HTTP メソッド別の内訳
     PromQL例:
       rate(http_requests_total[5m])
       sum by (path) (rate(http_requests_total[5m]))

  E — Errors（エラー率）:
     定義: 失敗したリクエストの割合
     指標:
       -> 5xx エラーの割合（サーバー起因）
       -> 4xx エラーの割合（クライアント起因）
       -> タイムアウト率
       -> サーキットブレーカー発動率
     PromQL例:
       rate(http_requests_total{status_code=~"5.."}[5m])
       / rate(http_requests_total[5m])

  D — Duration（レイテンシ）:
     定義: リクエスト処理にかかる時間
     指標:
       -> P50（中央値）: 典型的なユーザー体験
       -> P95: 大多数のユーザー体験
       -> P99: テール・レイテンシ
       -> P99.9: 最悪ケースに近い値
     PromQL例:
       histogram_quantile(0.99,
         rate(http_request_duration_seconds_bucket[5m]))
```

### 2.2 USE メソッド（リソース向け）

Brendan Gregg が提唱した USE メソッドは、CPU、メモリ、ディスク、ネットワークなどのインフラリソースのモニタリングに適している。APIサーバーのリソース状況を把握するために RED と併用する。

| リソース | Utilization（使用率） | Saturation（飽和度） | Errors（エラー） |
|---------|---------------------|---------------------|-----------------|
| CPU | CPU使用率 (%) | ランキュー長 | マシンチェック例外 |
| メモリ | メモリ使用率 (%) | スワップ使用量 | OOM キル回数 |
| ディスクI/O | I/O使用率 (%) | I/Oキュー長 | デバイスエラー |
| ネットワーク | 帯域使用率 (%) | パケットドロップ | CRCエラー |
| ファイル記述子 | FD使用率 (%) | ソケットキュー | 接続拒否 |

### 2.3 SLI / SLO / SLA の定義と運用

SLI（Service Level Indicator）、SLO（Service Level Objective）、SLA（Service Level Agreement）は、サービスの信頼性を定量的に管理するためのフレームワークである。

```
SLI / SLO / SLA の階層:

  +-------------------------------------------------------------------+
  |  SLA (Service Level Agreement)                                     |
  |  契約上の合意: 「99.9% の可用性を保証。違反時はクレジット返金」       |
  |                                                                    |
  |  +--------------------------------------------------------------+  |
  |  |  SLO (Service Level Objective)                                |  |
  |  |  内部目標: 「99.95% の可用性を目標とする」                     |  |
  |  |  ※ SLA より厳しく設定してバッファを確保                       |  |
  |  |                                                               |  |
  |  |  +----------------------------------------------------------+ |  |
  |  |  |  SLI (Service Level Indicator)                            | |  |
  |  |  |  測定指標: 「成功レスポンス数 / 全レスポンス数」            | |  |
  |  |  +----------------------------------------------------------+ |  |
  |  +--------------------------------------------------------------+  |
  +-------------------------------------------------------------------+

  代表的な SLI:
    可用性 SLI:   成功レスポンス / 全レスポンス
    レイテンシ SLI: P99 < 閾値 のリクエスト割合
    品質 SLI:     正常データ返却数 / 全レスポンス数
    鮮度 SLI:     最新データ返却数 / 全レスポンス数

  SLO 設計の指針:
    可用性:    99.9%（月間43分のダウンタイム許容）
    レイテンシ: P99 < 500ms を 99% の時間で達成
    エラー率:  < 0.1%

  エラーバジェットの概念:
    SLO 99.9% の場合 → エラーバジェット = 0.1%
    月間リクエスト 100万件 → 1,000リクエストまで失敗許容
    消費速度による意思決定:
      -> バジェット余裕あり: 新機能リリースを推進
      -> バジェット消費中:  リリース速度を調整
      -> バジェット枯渇:   新機能停止、信頼性改善に注力
```

### 2.4 ゴールデンシグナルとの対応

Google SRE が定義するFour Golden Signals との対応関係を整理する。

| Golden Signal | RED対応 | 説明 | 具体的メトリクス |
|--------------|---------|------|----------------|
| Latency | Duration | リクエスト処理時間 | http_request_duration_seconds |
| Traffic | Rate | リクエスト量 | http_requests_total |
| Errors | Errors | エラー率 | http_errors_total |
| Saturation | (USE) | リソース飽和度 | cpu_usage, memory_usage |

---

## 3. 構造化ログの設計と実装

### 3.1 なぜ構造化ログが必要か

従来のプレーンテキストログは、人間が読むには直感的だが、機械的な解析には適さない。構造化ログ（JSON形式）を採用することで、ログ集約基盤での検索・集計・アラートが容易になる。

```
従来のプレーンテキストログ:
  2024-01-15 10:30:00 INFO [UserService] GET /api/v1/users 200 45ms uid=user_123

構造化ログ (JSON):
  {
    "timestamp": "2024-01-15T10:30:00.000Z",
    "level": "info",
    "service": "user-service",
    "requestId": "req_abc123",
    "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
    "spanId": "00f067aa0ba902b7",
    "method": "GET",
    "path": "/api/v1/users",
    "statusCode": 200,
    "duration": 45,
    "userId": "user_123",
    "userAgent": "Mozilla/5.0..."
  }

構造化ログの利点:
  -> 検索可能:  path="/api/v1/users" AND statusCode>=500
  -> 集計可能:  AVG(duration) GROUP BY path
  -> 相関可能:  traceId で分散トレースと紐付け
  -> 型安全:    数値は数値として、文字列は文字列として扱える
  -> 拡張可能:  フィールド追加が容易
```

### 3.2 ログレベルの設計指針

ログレベルの使い分けはチーム内で統一しなければならない。以下に指針を示す。

| レベル | 用途 | プロダクション出力 | 例 |
|--------|------|------------------|-----|
| FATAL | プロセス停止が必要な致命的エラー | 常に出力 | DB接続不可、設定ファイル読み込み失敗 |
| ERROR | 処理失敗だがプロセスは継続可能 | 常に出力 | API呼び出し失敗、データ不整合 |
| WARN | 潜在的な問題、注意が必要な状況 | 常に出力 | レート制限接近、非推奨APIの使用 |
| INFO | 正常な業務イベント | 常に出力 | リクエスト完了、バッチ処理完了 |
| DEBUG | デバッグ用の詳細情報 | 通常は無効 | SQL クエリ内容、キャッシュヒット/ミス |
| TRACE | 最も詳細なトレース情報 | 通常は無効 | 関数の入出力、変数値 |

### 3.3 構造化ログの実装（Node.js / pino）

```javascript
// ===== 構造化ログ基盤の実装 =====
import pino from 'pino';
import { randomUUID } from 'crypto';
import { AsyncLocalStorage } from 'async_hooks';

// AsyncLocalStorage でリクエストコンテキストを管理
const asyncLocalStorage = new AsyncLocalStorage();

// ロガーの初期化
const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  formatters: {
    level: (label) => ({ level: label }),
    bindings: (bindings) => ({
      service: process.env.SERVICE_NAME || 'api-service',
      version: process.env.APP_VERSION || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      hostname: bindings.hostname,
      pid: bindings.pid,
    }),
  },
  timestamp: pino.stdTimeFunctions.isoTime,
  // 本番環境ではシリアライズを最適化
  serializers: {
    req: pino.stdSerializers.req,
    res: pino.stdSerializers.res,
    err: pino.stdSerializers.err,
  },
  // 機密情報のレダクション
  redact: {
    paths: [
      'req.headers.authorization',
      'req.headers.cookie',
      'body.password',
      'body.creditCard',
      'body.ssn',
    ],
    censor: '[REDACTED]',
  },
});

// コンテキスト付きロガーを取得
function getLogger() {
  const store = asyncLocalStorage.getStore();
  if (store && store.logger) {
    return store.logger;
  }
  return logger;
}

// リクエストログ・ミドルウェア
function requestLogger(req, res, next) {
  const requestId = req.headers['x-request-id'] || randomUUID();
  const traceId = req.headers['x-trace-id'] || randomUUID().replace(/-/g, '');
  const startTime = performance.now();

  // リクエスト情報をセット
  req.requestId = requestId;
  res.setHeader('X-Request-Id', requestId);

  // コンテキスト付きの子ロガーを生成
  const childLogger = logger.child({
    requestId,
    traceId,
    method: req.method,
    path: req.originalUrl,
  });

  req.log = childLogger;

  // リクエスト開始ログ
  childLogger.info({
    event: 'request_started',
    userAgent: req.headers['user-agent'],
    ip: req.ip,
    contentType: req.headers['content-type'],
    contentLength: req.headers['content-length'],
  }, 'Incoming request');

  // AsyncLocalStorage にコンテキストをセット
  asyncLocalStorage.run({ logger: childLogger, requestId, traceId }, () => {
    // レスポンス完了時にログ
    res.on('finish', () => {
      const duration = performance.now() - startTime;
      const logData = {
        event: 'request_completed',
        statusCode: res.statusCode,
        duration: Math.round(duration * 100) / 100,
        contentLength: res.getHeader('content-length'),
        userId: req.user?.sub,
      };

      if (res.statusCode >= 500) {
        childLogger.error(logData, 'Server error response');
      } else if (res.statusCode >= 400) {
        childLogger.warn(logData, 'Client error response');
      } else {
        childLogger.info(logData, 'Successful response');
      }
    });

    next();
  });
}

// 出力例:
// {
//   "level": "info",
//   "time": "2024-01-15T10:30:00.000Z",
//   "service": "user-service",
//   "version": "2.1.0",
//   "environment": "production",
//   "requestId": "req_abc123",
//   "traceId": "4bf92f3577b34da6a3ce929d0e0e4736",
//   "method": "GET",
//   "path": "/api/v1/users?limit=20",
//   "event": "request_completed",
//   "statusCode": 200,
//   "duration": 45.23,
//   "userId": "user_123"
// }
```

### 3.4 構造化ログの実装（Python / structlog）

```python
# ===== Python での構造化ログ実装（structlog + FastAPI） =====
import structlog
import uuid
import time
from contextvars import ContextVar
from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# コンテキスト変数
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
trace_id_var: ContextVar[str] = ContextVar('trace_id', default='')

# structlog の設定
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        # 本番環境では JSON レンダラー
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    wrapper_class=structlog.make_filtering_bound_logger(
        structlog.get_config()["wrapper_class"]
    ),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    """リクエスト/レスポンスログのミドルウェア"""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(
            'x-request-id', str(uuid.uuid4())
        )
        trace_id = request.headers.get(
            'x-trace-id', uuid.uuid4().hex
        )

        # コンテキスト変数にセット
        request_id_var.set(request_id)
        trace_id_var.set(trace_id)

        # structlog のコンテキストにバインド
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            trace_id=trace_id,
            method=request.method,
            path=str(request.url.path),
            service="user-service",
        )

        start_time = time.perf_counter()

        log = logger.bind()
        log.info(
            "request_started",
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000

            log_method = log.info
            if response.status_code >= 500:
                log_method = log.error
            elif response.status_code >= 400:
                log_method = log.warning

            log_method(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            response.headers['X-Request-Id'] = request_id
            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            log.exception(
                "request_failed",
                duration_ms=round(duration_ms, 2),
                error=str(exc),
            )
            raise
```

### 3.5 ログのセキュリティとプライバシー

ログに機密情報を含めないことは必須要件である。以下のフィールドは必ずマスキングまたは除外する。

```
機密情報の分類とマスキング方針:

  絶対に記録しない:
    -> パスワード、APIキー、トークン
    -> クレジットカード番号
    -> 社会保障番号（マイナンバー）
    -> 暗号鍵、シークレット

  マスキングして記録:
    -> メールアドレス: u***@example.com
    -> 電話番号: ***-****-1234
    -> IPアドレス: 192.168.xxx.xxx（必要に応じて）

  そのまま記録可能:
    -> リクエストID、トレースID
    -> HTTPメソッド、パス、ステータスコード
    -> レイテンシ、タイムスタンプ
    -> ユーザーID（内部識別子）
    -> User-Agent
```

---

## 4. 分散トレーシングの設計と実装

### 4.1 分散トレーシングの基本概念

マイクロサービスアーキテクチャでは、1つのユーザーリクエストが複数のサービスを横断して処理される。分散トレーシングは、このリクエストフロー全体を追跡し可視化する技術である。

```
分散トレーシングの構造:

  Trace（トレース）: 1つのリクエストの全体像
  |
  +-- Span A: API Gateway (開始 0ms, 終了 60ms)
  |   |
  |   +-- Span B: Auth Service (開始 2ms, 終了 12ms)
  |   |   |
  |   |   +-- Span C: Redis Cache Lookup (開始 3ms, 終了 5ms)
  |   |   +-- Span D: JWT Verify (開始 5ms, 終了 11ms)
  |   |
  |   +-- Span E: User Service (開始 13ms, 終了 55ms)
  |       |
  |       +-- Span F: PostgreSQL Query (開始 15ms, 終了 35ms)
  |       +-- Span G: Response Serialization (開始 36ms, 終了 42ms)
  |       +-- Span H: Cache Write (開始 43ms, 終了 50ms)

  タイムライン表示:
  |--A (API Gateway)----------------------------------------------|
    |--B (Auth)---------|
      |-C-| |-D-------|
                        |--E (User Service)----------------------|
                          |--F (DB Query)--------|
                                                  |-G-| |--H--|

  各 Span が保持する情報:
    -> trace_id:    リクエスト全体の一意識別子 (128bit)
    -> span_id:     個別処理の一意識別子 (64bit)
    -> parent_span_id: 親スパンのID
    -> operation:   操作名 (例: "GET /api/users")
    -> start_time:  開始時刻
    -> end_time:    終了時刻
    -> status:      成功/エラー
    -> attributes:  任意の属性 (key-value)
    -> events:      スパン内のイベント (ログ的な情報)
```

### 4.2 W3C Trace Context 標準

W3C Trace Context は、分散トレースのコンテキスト伝搬を標準化した仕様である。異なるベンダーのトレーシングツール間でも一貫したトレースが可能になる。

```
W3C Trace Context ヘッダー:

  traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01
               |  |                                |                  |
               |  |                                |                  +-- flags
               |  |                                |                      01 = sampled
               |  |                                +-- span-id (64bit, 16 hex)
               |  +-- trace-id (128bit, 32 hex chars)
               +-- version (常に "00")

  tracestate: rojo=00f067aa0ba902b7,congo=t61rcWkgMzE
              ベンダー固有の追加情報を伝搬

  baggage: userId=user_123,tenantId=tenant_456
           アプリケーション固有のコンテキストを伝搬
```

### 4.3 OpenTelemetry による分散トレーシング実装

OpenTelemetry（OTel）は、CNCF がホストするオブザーバビリティフレームワークであり、ベンダー非依存のテレメトリデータ（トレース、メトリクス、ログ）収集を可能にする。

```javascript
// ===== OpenTelemetry 完全セットアップ =====
// tracing.js - アプリケーション起動前に読み込む
import { NodeSDK } from '@opentelemetry/sdk-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-http';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { HttpInstrumentation } from '@opentelemetry/instrumentation-http';
import { ExpressInstrumentation } from '@opentelemetry/instrumentation-express';
import { PgInstrumentation } from '@opentelemetry/instrumentation-pg';
import { RedisInstrumentation } from '@opentelemetry/instrumentation-redis-4';
import { Resource } from '@opentelemetry/resources';
import {
  SEMRESATTRS_SERVICE_NAME,
  SEMRESATTRS_SERVICE_VERSION,
  SEMRESATTRS_DEPLOYMENT_ENVIRONMENT,
} from '@opentelemetry/semantic-conventions';

// リソース情報の定義
const resource = new Resource({
  [SEMRESATTRS_SERVICE_NAME]: process.env.SERVICE_NAME || 'user-service',
  [SEMRESATTRS_SERVICE_VERSION]: process.env.APP_VERSION || '1.0.0',
  [SEMRESATTRS_DEPLOYMENT_ENVIRONMENT]: process.env.NODE_ENV || 'development',
});

// トレースエクスポーターの設定
const traceExporter = new OTLPTraceExporter({
  url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/traces',
  headers: {
    // 認証が必要な場合
    // 'Authorization': `Bearer ${process.env.OTEL_AUTH_TOKEN}`,
  },
});

// メトリクスエクスポーターの設定
const metricExporter = new OTLPMetricExporter({
  url: process.env.OTEL_EXPORTER_OTLP_ENDPOINT || 'http://localhost:4318/v1/metrics',
});

const metricReader = new PeriodicExportingMetricReader({
  exporter: metricExporter,
  exportIntervalMillis: 30000, // 30秒間隔でエクスポート
});

// SDK の初期化
const sdk = new NodeSDK({
  resource,
  traceExporter,
  metricReader,
  instrumentations: [
    new HttpInstrumentation({
      // ヘルスチェックなど不要なトレースを除外
      ignoreIncomingRequestHook: (req) => {
        return req.url === '/health' || req.url === '/metrics';
      },
      // レスポンスヘッダーからカスタム属性を追加
      responseHook: (span, response) => {
        span.setAttribute('http.response.content_length',
          response.headers['content-length'] || 0);
      },
    }),
    new ExpressInstrumentation(),
    new PgInstrumentation({
      enhancedDatabaseReporting: true,
    }),
    new RedisInstrumentation(),
  ],
});

sdk.start();

// グレースフルシャットダウン
process.on('SIGTERM', () => {
  sdk.shutdown()
    .then(() => console.log('Tracing terminated'))
    .catch((error) => console.error('Error shutting down tracing', error))
    .finally(() => process.exit(0));
});
```

### 4.4 カスタムスパンの作成

自動計装だけでは捕捉できないビジネスロジックの処理を、カスタムスパンとして記録する。

```javascript
// ===== カスタムスパンの作成例 =====
import { trace, SpanStatusCode, context } from '@opentelemetry/api';

const tracer = trace.getTracer('user-service', '1.0.0');

// 基本的なカスタムスパン
async function getUser(id) {
  return tracer.startActiveSpan('getUser', async (span) => {
    // 属性の設定
    span.setAttribute('user.id', id);
    span.setAttribute('db.system', 'postgresql');

    try {
      // データベースクエリ
      const user = await tracer.startActiveSpan('db.query.findUser',
        async (dbSpan) => {
          dbSpan.setAttribute('db.statement', 'SELECT * FROM users WHERE id = $1');
          dbSpan.setAttribute('db.sql.table', 'users');

          const result = await db.users.findOne({ id });

          dbSpan.setAttribute('db.rows_affected', result ? 1 : 0);
          dbSpan.end();
          return result;
        }
      );

      if (!user) {
        span.setAttribute('user.found', false);
        span.addEvent('user_not_found', { 'user.id': id });
        return null;
      }

      // キャッシュ書き込み
      await tracer.startActiveSpan('cache.write', async (cacheSpan) => {
        cacheSpan.setAttribute('cache.type', 'redis');
        cacheSpan.setAttribute('cache.key', `user:${id}`);
        await redis.set(`user:${id}`, JSON.stringify(user), 'EX', 3600);
        cacheSpan.end();
      });

      span.setAttribute('user.found', true);
      span.setStatus({ code: SpanStatusCode.OK });
      return user;

    } catch (error) {
      span.recordException(error);
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: error.message,
      });
      throw error;
    } finally {
      span.end();
    }
  });
}

// サンプリング戦略の設定
import { ParentBasedSampler, TraceIdRatioBasedSampler } from '@opentelemetry/sdk-trace-base';

const sampler = new ParentBasedSampler({
  // 親スパンがない場合は 10% サンプリング
  root: new TraceIdRatioBasedSampler(0.1),
});
```

### 4.5 サンプリング戦略

全リクエストのトレースを収集するとストレージコストが膨大になるため、適切なサンプリング戦略が必要である。

| サンプリング方式 | 説明 | 適用場面 | トレードオフ |
|----------------|------|---------|------------|
| 常時サンプリング | 全トレースを収集 | 開発環境、低トラフィック | ストレージコスト大 |
| 確率サンプリング | 一定割合（例: 10%）を収集 | 一般的な本番環境 | レアイベントを見逃す可能性 |
| レートリミット | 秒間N件のトレースを収集 | 高トラフィック環境 | トラフィック急増時にカバー率低下 |
| テールベースサンプリング | エラーや遅延リクエストを優先収集 | 大規模本番環境 | Collector側の複雑性増加 |
| ルールベース | エンドポイント別にサンプリング率を設定 | 複雑なAPI群 | 設定管理の負荷 |

---

## 5. Prometheus によるメトリクス収集

### 5.1 Prometheus のアーキテクチャ

Prometheus は CNCF が管理するオープンソースの監視システムで、Pull型のメトリクス収集、時系列データベース、強力なクエリ言語（PromQL）を特徴とする。

```
Prometheus アーキテクチャ:

  +------------------+     +------------------+     +------------------+
  |  API Server A    |     |  API Server B    |     |  API Server C    |
  |  /metrics        |     |  /metrics        |     |  /metrics        |
  +--------+---------+     +--------+---------+     +--------+---------+
           |                        |                        |
           |   Pull (HTTP GET)      |                        |
           +------------------------+------------------------+
                                    |
                          +---------v---------+
                          |    Prometheus      |
                          |  +-------------+  |
                          |  | TSDB        |  |
                          |  | (時系列DB)  |  |
                          |  +-------------+  |
                          |  | PromQL      |  |
                          |  | (クエリ)    |  |
                          |  +-------------+  |
                          |  | Alert Rules |  |
                          |  | (アラート)  |  |
                          |  +-------------+  |
                          +---------+---------+
                                    |
                     +--------------+--------------+
                     |                             |
           +---------v---------+         +---------v---------+
           |   Alertmanager    |         |     Grafana       |
           |  (通知管理)       |         |  (可視化)         |
           +---------+---------+         +-------------------+
                     |
          +----------+----------+
          |          |          |
       Slack     PagerDuty   Email
```

### 5.2 メトリクスの型と使い分け

Prometheus には4種類のメトリクス型がある。それぞれの特性と適切な使い分けを理解することが重要である。

| 型 | 説明 | 用途 | 注意点 |
|---|------|------|-------|
| Counter | 単調増加する累積値 | リクエスト数、エラー数 | リセットは再起動時のみ。rate() で変化率を見る |
| Gauge | 増減する現在値 | CPU使用率、接続数、キューサイズ | スナップショット値。直接表示可能 |
| Histogram | 値の分布を観測（バケット） | レイテンシ、レスポンスサイズ | バケット境界の設計が重要 |
| Summary | クライアント側でパーセンタイル計算 | レイテンシ（サーバー側集約不要時） | 集約不可。Histogram を推奨 |

### 5.3 API メトリクス計装の実装

```javascript
// ===== Prometheus メトリクス計装（prom-client） =====
import {
  Registry, Counter, Histogram, Gauge, Summary,
  collectDefaultMetrics
} from 'prom-client';

const registry = new Registry();

// Node.js ランタイムのデフォルトメトリクスを収集
collectDefaultMetrics({
  register: registry,
  prefix: 'api_',
  gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5],
});

// ----- カスタムメトリクスの定義 -----

// リクエストカウンター
const httpRequestTotal = new Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'path', 'status_code'],
  registers: [registry],
});

// レイテンシ・ヒストグラム
const httpRequestDuration = new Histogram({
  name: 'http_request_duration_seconds',
  help: 'HTTP request duration in seconds',
  labelNames: ['method', 'path', 'status_code'],
  // レイテンシ分布に適したバケット設計
  buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  registers: [registry],
});

// アクティブ接続数（ゲージ）
const activeConnections = new Gauge({
  name: 'http_active_connections',
  help: 'Number of active HTTP connections',
  registers: [registry],
});

// レスポンスサイズ・ヒストグラム
const httpResponseSize = new Histogram({
  name: 'http_response_size_bytes',
  help: 'HTTP response size in bytes',
  labelNames: ['method', 'path'],
  buckets: [100, 500, 1000, 5000, 10000, 50000, 100000, 500000],
  registers: [registry],
});

// DB クエリカウンター
const dbQueryTotal = new Counter({
  name: 'db_queries_total',
  help: 'Total number of database queries',
  labelNames: ['operation', 'table', 'status'],
  registers: [registry],
});

// DB クエリ・レイテンシ
const dbQueryDuration = new Histogram({
  name: 'db_query_duration_seconds',
  help: 'Database query duration in seconds',
  labelNames: ['operation', 'table'],
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
  registers: [registry],
});

// 外部API呼び出し
const externalApiDuration = new Histogram({
  name: 'external_api_duration_seconds',
  help: 'External API call duration in seconds',
  labelNames: ['service', 'method', 'status'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
  registers: [registry],
});

// キャッシュ・ヒット率
const cacheOperations = new Counter({
  name: 'cache_operations_total',
  help: 'Total cache operations',
  labelNames: ['operation', 'result'], // result: hit, miss, error
  registers: [registry],
});

// ビジネスメトリクス例
const userRegistrations = new Counter({
  name: 'user_registrations_total',
  help: 'Total user registrations',
  labelNames: ['method'], // method: email, google, github
  registers: [registry],
});

// ----- ミドルウェア -----
function metricsMiddleware(req, res, next) {
  activeConnections.inc();
  const end = httpRequestDuration.startTimer();

  res.on('finish', () => {
    // パスの正規化（パスパラメータを :param に置換）
    const normalizedPath = req.route?.path || normalizePath(req.path);
    const labels = {
      method: req.method,
      path: normalizedPath,
      status_code: res.statusCode,
    };

    httpRequestTotal.inc(labels);
    end(labels);
    activeConnections.dec();

    // レスポンスサイズの記録
    const contentLength = parseInt(res.getHeader('content-length') || '0', 10);
    if (contentLength > 0) {
      httpResponseSize.observe(
        { method: req.method, path: normalizedPath },
        contentLength
      );
    }
  });

  next();
}

// パスの正規化（高カーディナリティを防止）
function normalizePath(path) {
  return path
    .replace(/\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/g, '/:id')
    .replace(/\/\d+/g, '/:id');
}

// メトリクスエンドポイント
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', registry.contentType);
  res.end(await registry.metrics());
});
```

### 5.4 PromQL クエリの実践

PromQL は Prometheus のクエリ言語であり、時系列データの選択、集約、変換を行う。以下に頻出クエリパターンを示す。

```
PromQL 頻出クエリ集:

  ===== Rate / Throughput =====

  # 全体のリクエストレート（5分間の移動平均）
  rate(http_requests_total[5m])

  # エンドポイント別のリクエストレート
  sum by (path) (rate(http_requests_total[5m]))

  # HTTP メソッド別のリクエストレート
  sum by (method) (rate(http_requests_total[5m]))

  ===== Error Rate =====

  # 5xx エラー率
  sum(rate(http_requests_total{status_code=~"5.."}[5m]))
  /
  sum(rate(http_requests_total[5m]))

  # エンドポイント別のエラー率
  sum by (path) (rate(http_requests_total{status_code=~"5.."}[5m]))
  /
  sum by (path) (rate(http_requests_total[5m]))

  # 特定エラーコード（429: Rate Limit）の発生率
  sum(rate(http_requests_total{status_code="429"}[5m]))

  ===== Latency (Percentile) =====

  # P50（中央値）
  histogram_quantile(0.50,
    sum by (le) (rate(http_request_duration_seconds_bucket[5m])))

  # P95
  histogram_quantile(0.95,
    sum by (le) (rate(http_request_duration_seconds_bucket[5m])))

  # P99
  histogram_quantile(0.99,
    sum by (le) (rate(http_request_duration_seconds_bucket[5m])))

  # エンドポイント別の P99
  histogram_quantile(0.99,
    sum by (le, path) (rate(http_request_duration_seconds_bucket[5m])))

  ===== Saturation =====

  # アクティブ接続数
  http_active_connections

  # Node.js イベントループ遅延
  api_nodejs_eventloop_lag_seconds

  # メモリ使用率
  api_process_resident_memory_bytes
  / on() group_left()
  machine_memory_bytes

  ===== SLO 関連 =====

  # 可用性 SLI（30日間）
  1 - (
    sum(increase(http_requests_total{status_code=~"5.."}[30d]))
    /
    sum(increase(http_requests_total[30d]))
  )

  # エラーバジェット残り（SLO 99.9%）
  1 - (
    sum(increase(http_requests_total{status_code=~"5.."}[30d]))
    /
    sum(increase(http_requests_total[30d]))
  ) - 0.999

  # エラーバジェット消費率（1時間あたり）
  (
    sum(rate(http_requests_total{status_code=~"5.."}[1h]))
    /
    sum(rate(http_requests_total[1h]))
  ) / 0.001

  ===== Database =====

  # DB クエリレート
  sum by (operation) (rate(db_queries_total[5m]))

  # DB クエリの P95 レイテンシ
  histogram_quantile(0.95,
    sum by (le, table) (rate(db_query_duration_seconds_bucket[5m])))

  # 遅いクエリの割合（1秒超）
  sum(rate(db_query_duration_seconds_bucket{le="1"}[5m]))
  /
  sum(rate(db_query_duration_seconds_count[5m]))

  ===== Cache =====

  # キャッシュヒット率
  sum(rate(cache_operations_total{result="hit"}[5m]))
  /
  sum(rate(cache_operations_total{result=~"hit|miss"}[5m]))
```

### 5.5 Prometheus の設定

```yaml
# ===== prometheus.yml =====
global:
  scrape_interval: 15s        # メトリクス収集間隔
  evaluation_interval: 15s    # ルール評価間隔
  scrape_timeout: 10s         # スクレイプタイムアウト

# アラートルールファイルの指定
rule_files:
  - "rules/api_alerts.yml"
  - "rules/slo_alerts.yml"

# Alertmanager の設定
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - "alertmanager:9093"

# スクレイプ対象の設定
scrape_configs:
  # API サーバー
  - job_name: 'api-servers'
    metrics_path: '/metrics'
    scrape_interval: 10s
    static_configs:
      - targets:
          - 'api-server-1:3000'
          - 'api-server-2:3000'
          - 'api-server-3:3000'
        labels:
          cluster: 'production'
          region: 'ap-northeast-1'

    # Kubernetes Service Discovery の場合
    # kubernetes_sd_configs:
    #   - role: pod
    # relabel_configs:
    #   - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
    #     action: keep
    #     regex: true

  # Node Exporter（インフラメトリクス）
  - job_name: 'node-exporter'
    static_configs:
      - targets:
          - 'node-exporter:9100'

  # PostgreSQL Exporter
  - job_name: 'postgres'
    static_configs:
      - targets:
          - 'postgres-exporter:9187'

  # Redis Exporter
  - job_name: 'redis'
    static_configs:
      - targets:
          - 'redis-exporter:9121'
```

---

## 6. Grafana によるダッシュボード設計

### 6.1 API ダッシュボードの構成原則

効果的なダッシュボードは、問題発生時に素早く原因を特定できる構造を持つ。「Overview → 詳細 → 根本原因」の階層的な構成が推奨される。

```
API ダッシュボードの階層構造:

  Level 1: Overview Dashboard（概要）
  +================================================================+
  |  [可用性]    [リクエスト/秒]   [P99 レイテンシ]   [エラー率]     |
  |   99.95%       1,245 RPS        123ms             0.03%        |
  +================================================================+
  |                                                                 |
  |  [RPS グラフ]              [レイテンシ分布グラフ]                |
  |  ~~~~~~~~                  ~~P50~~~~                            |
  |  ~~~~~~~~~~                ~~~~P95~~~~~                         |
  |  ~~~~~~~~~~~~              ~~~~~~P99~~~~~~~                     |
  |                                                                 |
  |  [エラー率グラフ]          [アクティブ接続数グラフ]              |
  |  __/\___                   ~~~~~~~~~                            |
  |  _______/\_                ~~~~~~~~~~~                          |
  +================================================================+

  Level 2: Endpoint Dashboard（エンドポイント別）
  +================================================================+
  |  エンドポイント別 RPS / レイテンシ / エラー率 テーブル           |
  |                                                                 |
  |  Path              RPS    P50    P99    Error%   Status         |
  |  GET /api/users    320    12ms   89ms   0.01%    OK             |
  |  POST /api/orders  180    45ms   230ms  0.05%    OK             |
  |  GET /api/products 450    8ms    45ms   0.02%    OK             |
  |  POST /api/auth    290    23ms   510ms  0.15%    WARN           |
  +================================================================+

  Level 3: Service Dependencies（依存サービス）
  +================================================================+
  |  [DB クエリ P99]   [Redis レイテンシ]   [外部API レイテンシ]    |
  |                                                                 |
  |  [DB コネクションプール使用率]  [キャッシュヒット率]             |
  +================================================================+

  Level 4: Infrastructure（インフラ）
  +================================================================+
  |  [CPU使用率]  [メモリ使用率]  [ディスクI/O]  [ネットワーク帯域] |
  +================================================================+
```

### 6.2 Grafana ダッシュボードの JSON プロビジョニング

```json
{
  "dashboard": {
    "title": "API Overview Dashboard",
    "tags": ["api", "production"],
    "timezone": "browser",
    "refresh": "30s",
    "panels": [
      {
        "title": "Request Rate (RPS)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m]))",
            "legendFormat": "Total RPS"
          },
          {
            "expr": "sum by (status_code) (rate(http_requests_total[5m]))",
            "legendFormat": "{{status_code}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth",
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "title": "Latency Percentiles",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))",
            "legendFormat": "P50"
          },
          {
            "expr": "histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))",
            "legendFormat": "P95"
          },
          {
            "expr": "histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket[5m])))",
            "legendFormat": "P99"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": {
              "drawStyle": "line",
              "lineInterpolation": "smooth"
            }
          }
        }
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 0, "y": 8 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status_code=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100",
            "legendFormat": "Error Rate %"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 0.1 },
                { "color": "red", "value": 1 }
              ]
            }
          }
        }
      }
    ]
  }
}
```

---

## 7. アラート設計

### 7.1 アラート設計の原則

効果的なアラートは「対処可能（actionable）」でなければならない。受け取った人が何をすべきか明確でないアラートは、アラート疲れ（alert fatigue）を招き、本当に重要なアラートの見逃しにつながる。

```
アラート設計の5原則:

  1. アクション可能であること
     -> アラートを受けた人が何をすべか明確
     -> 自動復旧するものはアラートにしない
     -> Runbook（対応手順書）を紐付ける

  2. SLO ベースであること
     -> エラーバジェットの消費率でアラート
     -> 瞬間的なスパイクでは発報しない
     -> 持続的な品質低下を検出する

  3. 重大度が適切であること
     -> Critical: 即時対応（ページ通知）
     -> Warning: 営業時間内に対応
     -> Info: 記録のみ（通知不要）

  4. 重複を排除すること
     -> 同じ根本原因のアラートをグループ化
     -> 上位レベルのアラートが下位を包含
     -> 連鎖的なアラート発報を抑制

  5. 定期的に見直すこと
     -> 発報ゼロのアラートは削除を検討
     -> 頻繁に誤報するアラートは閾値を調整
     -> インシデント事後分析でアラートの有効性を評価
```

### 7.2 アラートルールの実装（Prometheus Alerting Rules）

```yaml
# ===== rules/api_alerts.yml =====
groups:
  - name: api_availability
    rules:
      # ----- Critical Alerts -----

      # 5xx エラー率が 5% を超過（5分間持続）
      - alert: HighErrorRate
        expr: |
          sum(rate(http_requests_total{status_code=~"5.."}[5m]))
          /
          sum(rate(http_requests_total[5m]))
          > 0.05
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "High 5xx error rate detected"
          description: >
            Error rate is {{ $value | humanizePercentage }}
            (threshold: 5%). This indicates a significant service
            degradation affecting users.
          runbook_url: "https://wiki.example.com/runbooks/high-error-rate"
          dashboard_url: "https://grafana.example.com/d/api-overview"

      # P99 レイテンシが 5秒超（5分間持続）
      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
          ) > 5
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "P99 latency exceeds 5 seconds"
          description: >
            P99 latency is {{ $value | humanizeDuration }}.
            Check database queries and external API calls.
          runbook_url: "https://wiki.example.com/runbooks/high-latency"

      # サービスダウン（メトリクス収集不可）
      - alert: ServiceDown
        expr: up{job="api-servers"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "API server is down"
          description: "Instance {{ $labels.instance }} is unreachable."

      # ----- Warning Alerts -----

      # エラーバジェット消費率が 1日あたり 2% 超
      - alert: ErrorBudgetBurnRate
        expr: |
          (
            sum(rate(http_requests_total{status_code=~"5.."}[1h]))
            /
            sum(rate(http_requests_total[1h]))
          ) > 14.4 * 0.001
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "Error budget burn rate is high"
          description: >
            Current burn rate will exhaust the monthly error budget
            in less than 2 days.

      # P99 レイテンシが 1秒超
      - alert: ElevatedLatency
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
          ) > 1
        for: 10m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "P99 latency exceeds 1 second"

      # ディスク使用率 80% 超
      - alert: DiskSpaceWarning
        expr: |
          (node_filesystem_avail_bytes{mountpoint="/"}
          / node_filesystem_size_bytes{mountpoint="/"}) < 0.2
        for: 15m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Disk space is running low"

      # DB 接続プール枯渇気味
      - alert: DBConnectionPoolExhaustion
        expr: |
          pg_stat_activity_count / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "Database connection pool is nearly exhausted"

  - name: api_slo
    rules:
      # SLO 可用性（30日ローリング）
      - record: slo:availability:ratio30d
        expr: |
          1 - (
            sum(increase(http_requests_total{status_code=~"5.."}[30d]))
            /
            sum(increase(http_requests_total[30d]))
          )

      # SLO レイテンシ（P99 < 500ms 達成率）
      - record: slo:latency:ratio30d
        expr: |
          sum(increase(http_request_duration_seconds_bucket{le="0.5"}[30d]))
          /
          sum(increase(http_request_duration_seconds_count[30d]))

      # エラーバジェット残量
      - record: slo:error_budget:remaining
        expr: |
          1 - (
            (1 - slo:availability:ratio30d) / (1 - 0.999)
          )
```

### 7.3 Alertmanager の設定

```yaml
# ===== alertmanager.yml =====
global:
  resolve_timeout: 5m
  slack_api_url: 'https://hooks.slack.com/services/xxx/yyy/zzz'

# 通知テンプレート
templates:
  - '/etc/alertmanager/templates/*.tmpl'

# ルーティング設定
route:
  # デフォルトの受信者
  receiver: 'slack-default'
  # グループ化するラベル
  group_by: ['alertname', 'team']
  # グループ化の待機時間
  group_wait: 30s
  # 同一グループの再通知間隔
  group_interval: 5m
  # 同一アラートの再通知間隔
  repeat_interval: 4h

  routes:
    # Critical アラート → PagerDuty + Slack
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      group_wait: 10s
      repeat_interval: 1h
      continue: true  # 後続ルートも評価

    - match:
        severity: critical
      receiver: 'slack-critical'

    # Warning アラート → Slack のみ
    - match:
        severity: warning
      receiver: 'slack-warning'
      repeat_interval: 8h

# 通知の抑制ルール
inhibit_rules:
  # ServiceDown が発報中は、同インスタンスの他アラートを抑制
  - source_match:
      alertname: 'ServiceDown'
    target_match_re:
      alertname: '.+'
    equal: ['instance']

# 受信者の定義
receivers:
  - name: 'slack-default'
    slack_configs:
      - channel: '#alerts-info'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.summary }}'

  - name: 'slack-critical'
    slack_configs:
      - channel: '#alerts-critical'
        title: '[CRITICAL] {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
        color: 'danger'

  - name: 'slack-warning'
    slack_configs:
      - channel: '#alerts-warning'
        title: '[WARNING] {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.summary }}'
        color: 'warning'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
        severity: 'critical'
        description: '{{ .CommonAnnotations.summary }}'
```

---

## 8. ログ集約基盤の構築

### 8.1 ログ集約アーキテクチャの比較

| 項目 | ELK Stack | Grafana Loki | Datadog Logs |
|------|-----------|-------------|--------------|
| 構成 | Elasticsearch + Logstash + Kibana | Loki + Promtail + Grafana | SaaS（マネージド） |
| インデックス方式 | 全文検索インデックス | ラベルのみインデックス | 全文検索 |
| ストレージコスト | 高（全フィールドインデックス） | 低（ログ本文は非インデックス） | 従量課金 |
| クエリ速度 | 高速（インデックス済み） | ラベル検索は高速、本文検索はやや遅い | 高速 |
| 運用負荷 | 高（Elasticsearch クラスタ管理） | 低（シンプルなアーキテクチャ） | なし（SaaS） |
| Grafana統合 | プラグインで可能 | ネイティブ統合 | プラグインで可能 |
| 適用規模 | 中〜大規模 | 小〜大規模 | 全規模 |

### 8.2 Grafana Loki によるログ集約

Loki は Grafana Labs が開発したログ集約システムで、「Prometheus のログ版」とも呼ばれる。メタデータ（ラベル）のみをインデックス化し、ログ本文はそのまま保存する設計により、低コストで大量のログを扱える。

```yaml
# ===== Loki + Promtail の Docker Compose 構成 =====
version: "3.8"

services:
  loki:
    image: grafana/loki:2.9.4
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/config.yml
      - loki-data:/loki
    command: -config.file=/etc/loki/config.yml

  promtail:
    image: grafana/promtail:2.9.4
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml

  grafana:
    image: grafana/grafana:10.3.1
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Admin

volumes:
  loki-data:
  grafana-data:
```

```yaml
# ===== promtail-config.yml =====
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: api-logs
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_label_service']
        target_label: 'service'
    pipeline_stages:
      # JSON ログのパース
      - json:
          expressions:
            level: level
            requestId: requestId
            traceId: traceId
            method: method
            path: path
            statusCode: statusCode
            duration: duration
      # ラベルとして抽出
      - labels:
          level:
          service:
          method:
          statusCode:
      # タイムスタンプの設定
      - timestamp:
          source: time
          format: RFC3339Nano
      # メトリクスの生成（ログからメトリクスを導出）
      - metrics:
          log_lines_total:
            type: Counter
            description: "Total log lines"
            source: level
            config:
              match_all: true
              action: inc
          request_duration_from_logs:
            type: Histogram
            description: "Request duration from logs"
            source: duration
            config:
              buckets: [10, 50, 100, 250, 500, 1000, 2500, 5000]
```

### 8.3 LogQL クエリの実践

LogQL は Loki のクエリ言語であり、PromQL に似た構文でログの検索と集計を行う。

```
LogQL 頻出クエリ集:

  ===== ログストリームの選択 =====

  # サービス名でフィルタ
  {service="user-service"}

  # 複数条件
  {service="user-service", level="error"}

  # 正規表現マッチ
  {service=~"user-.*|order-.*"}

  ===== ログ行のフィルタリング =====

  # テキスト検索（含む）
  {service="user-service"} |= "timeout"

  # テキスト検索（含まない）
  {service="user-service"} != "healthcheck"

  # 正規表現フィルタ
  {service="user-service"} |~ "status_code=(5[0-9]{2})"

  ===== JSON パース + フィルタ =====

  # JSON フィールドでフィルタ
  {service="user-service"}
    | json
    | statusCode >= 500

  # 特定パスのエラー
  {service="user-service"}
    | json
    | path="/api/v1/orders"
    | statusCode >= 500

  # 遅いリクエスト（500ms以上）
  {service="user-service"}
    | json
    | duration > 500

  ===== メトリクスクエリ（集計） =====

  # エラーログの発生率
  rate({service="user-service", level="error"}[5m])

  # エンドポイント別のリクエスト数
  sum by (path) (
    count_over_time(
      {service="user-service"}
        | json
        | __error__=""
      [5m]
    )
  )

  # P99 レイテンシ（ログから算出）
  quantile_over_time(0.99,
    {service="user-service"}
      | json
      | unwrap duration
    [5m]
  ) by (path)
```

---

## 9. OpenTelemetry Collector の構成

OpenTelemetry Collector は、テレメトリデータ（トレース、メトリクス、ログ）を受信、処理、エクスポートするエージェントである。アプリケーションとバックエンドの間に配置することで、ベンダー非依存のデータパイプラインを構築できる。

```
OpenTelemetry Collector アーキテクチャ:

  +----------------+   +----------------+   +----------------+
  |  API Server A  |   |  API Server B  |   |  API Server C  |
  |  (OTLP gRPC)   |   |  (OTLP HTTP)   |   |  (OTLP gRPC)   |
  +-------+--------+   +-------+--------+   +-------+--------+
          |                     |                     |
          +---------------------+---------------------+
                                |
                 +--------------v--------------+
                 |    OTel Collector           |
                 |  +-----------------------+  |
                 |  | Receivers             |  |
                 |  | - OTLP (gRPC/HTTP)    |  |
                 |  | - Prometheus           |  |
                 |  | - Jaeger               |  |
                 |  +-----------+-----------+  |
                 |              |               |
                 |  +-----------v-----------+  |
                 |  | Processors            |  |
                 |  | - Batch               |  |
                 |  | - Memory Limiter      |  |
                 |  | - Attributes          |  |
                 |  | - Filter              |  |
                 |  | - Tail Sampling       |  |
                 |  +-----------+-----------+  |
                 |              |               |
                 |  +-----------v-----------+  |
                 |  | Exporters             |  |
                 |  | - OTLP                |  |
                 |  | - Prometheus          |  |
                 |  | - Jaeger              |  |
                 |  | - Loki                |  |
                 |  +-----------------------+  |
                 +-----------------------------+
                                |
              +-----------------+------------------+
              |                 |                  |
    +---------v------+  +------v-------+  +-------v------+
    |  Prometheus    |  |   Jaeger     |  |    Loki      |
    |  (Metrics)     |  |  (Traces)    |  |   (Logs)     |
    +----------------+  +--------------+  +--------------+
```

```yaml
# ===== otel-collector-config.yml =====
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: "0.0.0.0:4317"
      http:
        endpoint: "0.0.0.0:4318"

  # Prometheus メトリクスも受信可能
  prometheus:
    config:
      scrape_configs:
        - job_name: 'otel-collector'
          scrape_interval: 10s
          static_configs:
            - targets: ['0.0.0.0:8888']

processors:
  # バッチ処理（パフォーマンス最適化）
  batch:
    timeout: 5s
    send_batch_size: 1024
    send_batch_max_size: 2048

  # メモリ制限
  memory_limiter:
    check_interval: 1s
    limit_mib: 512
    spike_limit_mib: 128

  # 属性の追加・変換
  attributes:
    actions:
      - key: environment
        value: production
        action: upsert
      - key: cluster
        value: ap-northeast-1
        action: upsert

  # 不要なデータのフィルタリング
  filter:
    error_mode: ignore
    traces:
      span:
        - 'attributes["http.target"] == "/health"'
        - 'attributes["http.target"] == "/metrics"'

  # テールベースサンプリング
  tail_sampling:
    decision_wait: 10s
    num_traces: 100000
    policies:
      # エラーは全て収集
      - name: errors
        type: status_code
        status_code:
          status_codes: [ERROR]
      # 遅いリクエストは全て収集
      - name: slow-requests
        type: latency
        latency:
          threshold_ms: 1000
      # その他は 10% サンプリング
      - name: probabilistic
        type: probabilistic
        probabilistic:
          sampling_percentage: 10

exporters:
  # トレース -> Jaeger
  otlp/jaeger:
    endpoint: "jaeger:4317"
    tls:
      insecure: true

  # メトリクス -> Prometheus
  prometheusremotewrite:
    endpoint: "http://prometheus:9090/api/v1/write"

  # ログ -> Loki
  loki:
    endpoint: "http://loki:3100/loki/api/v1/push"

  # デバッグ用ログ出力
  logging:
    loglevel: info

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [memory_limiter, batch, filter, tail_sampling, attributes]
      exporters: [otlp/jaeger, logging]
    metrics:
      receivers: [otlp, prometheus]
      processors: [memory_limiter, batch, attributes]
      exporters: [prometheusremotewrite]
    logs:
      receivers: [otlp]
      processors: [memory_limiter, batch, attributes]
      exporters: [loki]

  telemetry:
    logs:
      level: info
    metrics:
      address: ":8888"
```

---

## 10. アンチパターン

### アンチパターン1: ログの過剰出力と高カーディナリティ

**症状**: ログストレージが急速に肥大化し、検索パフォーマンスが劣化。メトリクスのラベル爆発によりPrometheusのメモリが枯渇する。

```
問題のあるログ出力:

  // 全リクエストのボディをログに含める
  logger.info({
    event: 'request',
    body: req.body,         // 大量のデータがログに流れる
    headers: req.headers,   // 機密情報を含む可能性
    query: req.query,
  });

  // デバッグログを本番で有効にしたまま
  logger.debug({ cache: entireCacheContents }); // 巨大オブジェクト

問題のあるメトリクスラベル:

  // ユーザーIDをラベルに含める（高カーディナリティ）
  httpRequestTotal.inc({
    method: req.method,
    path: req.originalUrl,  // クエリパラメータ込み -> 無限のラベル組合せ
    userId: req.user.id,    // ユーザー数分のラベル値
    requestId: req.id,      // リクエスト毎にユニーク -> 致命的
  });

  結果:
    -> Prometheus のメモリ使用量が指数関数的に増大
    -> クエリ速度が大幅に低下
    -> カーディナリティ爆発（cardinality explosion）

正しい設計:

  // ログ: 必要最小限のフィールドに絞る
  logger.info({
    event: 'request_completed',
    method: req.method,
    path: req.route?.path,  // テンプレートパス（/users/:id）
    statusCode: res.statusCode,
    duration: durationMs,
    userId: req.user?.id,   // ログには可（メトリクスには不可）
  });

  // メトリクス: 低カーディナリティのラベルのみ
  httpRequestTotal.inc({
    method: req.method,
    path: req.route?.path || normalizePath(req.path),
    status_code: res.statusCode,
  });
```

### アンチパターン2: アラート設定の不備によるアラート疲れ

**症状**: 大量の不要なアラートが発報し、オンコールエンジニアが疲弊。重要なアラートが他のノイズに埋もれて見逃される。

```
問題のあるアラート設定:

  1. 閾値が厳しすぎる:
     alert: HighLatency
     expr: histogram_quantile(0.99, ...) > 0.1   # 100ms は厳しすぎ
     for: 1m                                       # 1分で発報は早すぎ
     -> 一時的なスパイクで頻繁に発報

  2. アクション不明確:
     annotations:
       summary: "Something went wrong"   # 何が問題か不明
       # runbook_url なし               # 対処方法が不明
     -> 受け取っても何をすべきかわからない

  3. 重複アラート:
     DB接続エラー -> 以下が同時に発報:
       - DBConnectionError
       - HighErrorRate
       - SlowQueries
       - ServiceDegraded
       - SLOViolation
     -> 根本原因は1つなのに5つのアラート

  4. 自動復旧する問題へのアラート:
     一時的なネットワーク切断 -> 自動リトライで復旧
     -> 不要なアラート発報

正しいアラート設計:

  1. 適切な閾値と持続時間:
     alert: HighLatency
     expr: histogram_quantile(0.99, ...) > 1     # 1秒は合理的
     for: 10m                                      # 10分持続で確信

  2. 明確なアノテーション:
     annotations:
       summary: "P99 latency exceeds 1s for 10 minutes"
       description: "Current P99: {{ $value }}s. Check DB and cache."
       runbook_url: "https://wiki.example.com/runbooks/high-latency"
       dashboard_url: "https://grafana.example.com/d/api"

  3. 抑制ルール（Inhibition）:
     DBConnectionError が発報中 -> 派生アラートを抑制

  4. アクション可能なもののみ:
     自動復旧する問題 -> アラートではなくメトリクスで記録
```

---

## 11. エッジケース分析

### エッジケース1: メトリクス収集の時間ずれとサンプリングの落とし穴

Prometheus は Pull 型でメトリクスを収集するため、スクレイプ間隔の間に発生した短時間のスパイクを捕捉できない場合がある。また、ヒストグラムのバケット設計が不適切だと、パーセンタイル値に大きな誤差が生じる。

```
問題: ヒストグラムバケットの設計不備

  設定:
    buckets: [0.1, 1, 10]  // 3バケットのみ

  実際の分布:
    0-50ms:   80% のリクエスト
    50-100ms: 15% のリクエスト
    100-200ms: 4% のリクエスト
    200ms+:    1% のリクエスト

  P99 の計算結果:
    -> 0.1s (100ms) バケットに99%が収まるため
    -> P99 = 約100ms と報告される
    -> 実際の P99 は約180ms（大きな誤差）

  対策:
    buckets: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    -> レイテンシ分布に合った細かいバケットを設計
    -> デフォルトバケット（上記）は多くのAPI向けに適切

問題: スクレイプ間隔とレートの誤差

  スクレイプ間隔: 15秒
  実際のリクエストパターン:
    0-5s:  100 req/s（スパイク）
    5-15s: 10 req/s（通常）

  rate(http_requests_total[1m]) の結果:
    -> 約 40 req/s（平均化されてしまう）
    -> 100 req/s のスパイクは観測できない

  対策:
    -> irate() を使用して瞬間レートを計算
    -> スクレイプ間隔を短くする（5秒等。ただしリソース消費増加）
    -> アプリケーション側でもメトリクスを記録
```

### エッジケース2: 分散トレーシングにおけるコンテキスト消失

非同期処理やメッセージキューを介した通信では、トレースコンテキストが消失しやすい。特にイベント駆動アーキテクチャでは、明示的なコンテキスト伝搬が必要になる。

```
問題: メッセージキュー経由のコンテキスト消失

  [API Server] --HTTP--> [Order Service] --Kafka--> [Payment Service]
       |                       |                          |
    trace_id: abc123      trace_id: abc123           trace_id: ???
    span_id:  001         span_id:  002              (コンテキスト消失)

  原因:
    -> Kafka メッセージにトレースコンテキストを含めていない
    -> Consumer 側でコンテキストを復元していない

  対策: メッセージヘッダーにコンテキストを埋め込む

    // Producer 側
    import { propagation, context } from '@opentelemetry/api';

    async function publishToKafka(topic, message) {
      return tracer.startActiveSpan('kafka.produce', async (span) => {
        span.setAttribute('messaging.system', 'kafka');
        span.setAttribute('messaging.destination', topic);

        // 現在のコンテキストをヘッダーに注入
        const headers = {};
        propagation.inject(context.active(), headers);

        await producer.send({
          topic,
          messages: [{
            value: JSON.stringify(message),
            headers,  // トレースコンテキストを含むヘッダー
          }],
        });

        span.end();
      });
    }

    // Consumer 側
    async function consumeFromKafka(message) {
      // ヘッダーからコンテキストを復元
      const parentContext = propagation.extract(
        context.active(),
        message.headers
      );

      // 復元したコンテキストを親として新しいスパンを作成
      return context.with(parentContext, () => {
        return tracer.startActiveSpan('kafka.consume', async (span) => {
          span.setAttribute('messaging.system', 'kafka');

          // ビジネスロジック
          await processPayment(JSON.parse(message.value));

          span.end();
        });
      });
    }

問題: setTimeout / setInterval でのコンテキスト消失

  対策:
    import { context } from '@opentelemetry/api';

    // コンテキストを明示的に伝搬
    const currentContext = context.active();
    setTimeout(() => {
      context.with(currentContext, () => {
        // ここではトレースコンテキストが保持される
        doSomething();
      });
    }, 5000);
```

---

## 12. 演習問題

### 演習1: 基礎 -- 構造化ログの実装

以下の要件を満たすログミドルウェアを Express.js で実装せよ。

```
要件:
  1. JSON 形式の構造化ログを出力する
  2. 各リクエストに一意の requestId を付与する
  3. レスポンス完了時に以下の情報を記録する:
     - method, path, statusCode, duration(ms)
     - userId（認証済みの場合）
  4. ステータスコードに応じてログレベルを変える:
     - 5xx -> error
     - 4xx -> warn
     - それ以外 -> info
  5. パスワードやトークンなどの機密情報をマスキングする

ヒント:
  - pino または winston を使用する
  - res.on('finish', callback) でレスポンス完了を検知する
  - pino の redact オプションで機密情報を除外する

確認ポイント:
  -> ログが JSON 形式で出力されているか
  -> requestId が全ログに含まれているか
  -> 機密情報がマスキングされているか
  -> ログレベルが適切に設定されているか
```

### 演習2: 中級 -- Prometheus メトリクスとアラート設計

以下の要件を満たす API モニタリング基盤を構築せよ。

```
要件:
  1. prom-client で以下のメトリクスを計装する:
     - http_requests_total (Counter)
     - http_request_duration_seconds (Histogram)
     - http_active_connections (Gauge)
     - db_query_duration_seconds (Histogram)
  2. /metrics エンドポイントを公開する
  3. 以下のアラートルールを Prometheus に設定する:
     - 5xx エラー率 > 5% が 5分間持続 -> Critical
     - P99 レイテンシ > 2秒 が 10分間持続 -> Warning
     - サービスダウン（up == 0）が 1分間持続 -> Critical
  4. Grafana ダッシュボードで RPS、レイテンシ、エラー率を可視化する

構成:
  docker-compose.yml で以下を起動:
    - API サーバー（Node.js）
    - Prometheus
    - Grafana
    - Alertmanager

確認ポイント:
  -> curl http://localhost:3000/metrics でメトリクスが取得できるか
  -> Prometheus の Targets で API サーバーが UP であるか
  -> Grafana でダッシュボードが表示されるか
  -> 意図的にエラーを発生させてアラートが発報するか
```

### 演習3: 上級 -- 分散トレーシングとオブザーバビリティ統合

以下のマイクロサービス構成で、エンドツーエンドの分散トレーシングを実装せよ。

```
構成:
  [Client] -> [API Gateway] -> [User Service] -> [PostgreSQL]
                             -> [Order Service] -> [Redis]
                                                -> [Payment API (外部)]

要件:
  1. OpenTelemetry SDK で各サービスを計装する
  2. W3C Trace Context ヘッダーでコンテキストを伝搬する
  3. カスタムスパンでビジネスロジックの処理時間を記録する
  4. エラー発生時に span.recordException() でエラー情報を記録する
  5. Jaeger でトレースを可視化する
  6. Grafana でメトリクス、トレース、ログを相互リンクする

追加課題:
  - テールベースサンプリングを OTel Collector で設定する
  - エラートレースと遅延トレースを 100% 収集する
  - ログに trace_id と span_id を含めてトレースと紐付ける
  - Grafana の Explore ビューでトレースからログへジャンプする

確認ポイント:
  -> Jaeger でサービス間のトレースが一貫して表示されるか
  -> ボトルネックとなっているスパンを特定できるか
  -> エラーが発生したスパンが正しくマークされているか
  -> ログから対応するトレースへ遷移できるか
  -> サンプリングポリシーが期待通りに動作しているか
```

---

## 13. 本番環境でのベストプラクティス

### 13.1 ログローテーションと保持ポリシー

```
ログ保持ポリシーの設計指針:

  Tier 1 -- ホットストレージ（高速検索可能）:
    期間:   7-14日
    用途:   アクティブなデバッグ、インシデント対応
    ストレージ: SSD / Elasticsearch Hot Node

  Tier 2 -- ウォームストレージ（検索可能だがやや遅い）:
    期間:   30-90日
    用途:   トレンド分析、過去のインシデント調査
    ストレージ: HDD / Elasticsearch Warm Node / S3

  Tier 3 -- コールドストレージ（アーカイブ）:
    期間:   1年以上（コンプライアンス要件による）
    用途:   監査、法的要件
    ストレージ: S3 Glacier / GCS Coldline

  ログレベル別の保持期間:
    ERROR/FATAL: 90日（ホット14日 + ウォーム76日）
    WARN:        30日（ホット7日 + ウォーム23日）
    INFO:        14日（全てホット）
    DEBUG:       3日（開発環境のみ）
```

### 13.2 パフォーマンスへの影響を最小化する

```
モニタリングのオーバーヘッドを抑える原則:

  1. 非同期ログ出力:
     -> ログの書き込みをバッファリングし、非同期でフラッシュ
     -> pino はデフォルトで非同期書き込みに対応
     -> 同期書き込みはレイテンシに直接影響する

  2. メトリクスのラベル数を制限:
     -> ラベルの組合せ数（カーディナリティ）を 1,000 以下に
     -> path ラベルはテンプレート（/users/:id）を使用
     -> ユーザーIDやリクエストIDはラベルに含めない

  3. トレースのサンプリング:
     -> 本番環境では 1-10% のサンプリング
     -> エラーと遅延リクエストは 100% 収集
     -> テールベースサンプリングで重要なトレースを確保

  4. ログのフィルタリング:
     -> ヘルスチェックやメトリクスエンドポイントのログを除外
     -> 本番では DEBUG/TRACE レベルを無効化
     -> 動的ログレベル変更の仕組みを用意する

  5. バッチ処理:
     -> テレメトリデータのエクスポートをバッチ化
     -> 個別送信ではなく、まとめて送信することでオーバーヘッド削減
```

### 13.3 障害時のモニタリング継続性

```
モニタリング基盤自体の可用性確保:

  問題: 障害発生時こそモニタリングが重要だが、
        障害がモニタリング基盤に波及する場合がある

  対策:
    1. モニタリング基盤を監視対象とは別のインフラに配置
    2. Prometheus の Federation で階層化
    3. Thanos/Cortex で長期保存と高可用性を実現
    4. アラート通知経路を多重化（Slack + PagerDuty + Email）
    5. Dead Man's Switch アラート（常時発報するアラートが
       停止した場合にモニタリング基盤の障害を検出）

  Dead Man's Switch の例:
    - alert: PrometheusAlive
      expr: vector(1)
      labels:
        severity: critical
      annotations:
        summary: "Dead man's switch - Prometheus is alive"
    -> Alertmanager でこのアラートの受信を監視
    -> 一定時間受信しなければ、外部監視から通知
```

---

## 14. FAQ（よくある質問）

### Q1: Prometheus と Datadog のどちらを選ぶべきか

Prometheus はオープンソースで柔軟性が高く、ランニングコストを抑えられる反面、運用負荷が高い。Datadog は SaaS であり、運用負荷は低いが従量課金のコストが嵩む場合がある。選定の基準は以下の通りである。

| 判断軸 | Prometheus 推奨 | Datadog 推奨 |
|--------|----------------|-------------|
| チームの運用力 | Kubernetes/インフラ運用に強い | アプリ開発に集中したい |
| 予算 | インフラコストに余裕がある | 人件費 > SaaS費用の場合 |
| スケール | 中規模（月間数十億データポイント） | 大規模・多拠点 |
| カスタマイズ | 独自のメトリクス設計が必要 | 標準的な監視で十分 |
| 統合 | Grafana エコシステムを活用 | APM + ログ + インフラを一元管理 |

小〜中規模のチームではまず Prometheus + Grafana で始め、運用負荷が課題になった段階で SaaS への移行を検討するアプローチが合理的である。

### Q2: ログとメトリクスの使い分けの基準は何か

端的に言えば、「何が起きたか」を知りたい場合はログ、「どれくらい起きているか」を知りたい場合はメトリクスである。

- メトリクス: 集約的な問いに答える。「エラー率は何%か」「P99レイテンシは何msか」「リクエスト数は増加傾向か」
- ログ: 個別の問いに答える。「このリクエストはなぜ失敗したか」「ユーザーXの操作履歴は何か」「どのSQLクエリがエラーになったか」

アラートはメトリクスベースで設定し、インシデント調査時にログを参照するというフローが一般的である。メトリクスで異常を検出し、ログで根本原因を特定する。

### Q3: 分散トレーシングのサンプリング率はどの程度にすべきか

サンプリング率はトラフィック量とストレージ容量に依存する。一般的な指針は以下の通りである。

- 開発/ステージング環境: 100%（全トレース収集）
- 低トラフィック（< 100 RPS）の本番環境: 50-100%
- 中トラフィック（100-1000 RPS）: 10-50%
- 高トラフィック（> 1000 RPS）: 1-10%

ただし、テールベースサンプリングを導入すれば、エラーや高レイテンシのトレースは 100% 収集しつつ、正常なトレースのみサンプリング対象にできる。これにより、デバッグに最も価値のあるトレースを確実に保持できる。

### Q4: OpenTelemetry とベンダー固有の SDK はどちらを使うべきか

OpenTelemetry を推奨する。OpenTelemetry はベンダー非依存であるため、バックエンド（Jaeger, Zipkin, Datadog, New Relic 等）を後から変更できる。ベンダーロックインを避けることで、将来のコスト最適化や技術選択の自由度が確保される。主要な APM ベンダーも OpenTelemetry のネイティブサポートを進めており、互換性の問題は減少している。

### Q5: モニタリングツールの選定基準は何か

モニタリングツールの選定では以下の観点を総合的に評価する必要がある。

| 観点 | 評価ポイント |
|------|------------|
| 機能カバレッジ | メトリクス・ログ・トレースの統合度、ダッシュボード機能、アラート機能の充実度 |
| スケーラビリティ | 想定データ量（データポイント/秒、ログ量/日）に対応できるか |
| コスト | 初期費用、運用コスト、データ保持コスト、従量課金の透明性 |
| 運用負荷 | セルフホスト vs SaaS、必要な専門知識、保守の難易度 |
| エコシステム | 既存ツールとの統合、OpenTelemetry サポート、プラグインの充実度 |
| ベンダーロックイン | データのエクスポート可否、標準プロトコル対応 |

スタートアップや小規模チームであれば、運用負荷の低い SaaS（Datadog、New Relic）を選択し、成長後にコスト最適化として Prometheus + Grafana 等のオープンソースへの移行を検討するのが現実的である。一方、大規模なトラフィックを扱う組織では、初期からオープンソースベースの基盤を構築し、運用ノウハウを蓄積する戦略が有効である。

### Q6: アラート設計のベストプラクティスは何か

効果的なアラート設計には以下の原則を適用する。

1. **SLO ベースのアラート**: 「CPU 使用率 > 80%」ではなく「エラーバジェットの消費率 > 5%/hour」のようにユーザー影響に基づくアラートを設定する
2. **アクション可能性**: アラートを受け取った人が明確なアクションを取れる内容にする。「何を確認し、何をすべきか」を Runbook として整備する
3. **適切な閾値設定**: 過去のデータから P95/P99 を分析し、誤検知と見逃しのバランスを取る。静的閾値ではなく異常検知アルゴリズムの活用も検討する
4. **重複排除とグルーピング**: 関連する複数のアラートを 1 つの通知にまとめ、アラート疲れを防ぐ
5. **段階的エスカレーション**: Severity（Critical/Warning/Info）に応じた通知先と対応時間を明確化する
6. **サイレント期間の設定**: デプロイ中やメンテナンス中のアラートを自動的に抑制する仕組みを用意する

アラートは「ユーザーに影響が出ている」または「これから影響が出る可能性が高い」場合にのみ発報すべきである。単なる情報通知はアラートではなくダッシュボードやレポートで十分である。

### Q7: ログの保持期間と管理はどうすべきか

ログの保持期間は、法的要件、コスト、実用性のバランスで決定する。

| ログ種別 | 推奨保持期間 | 理由 |
|---------|------------|------|
| アプリケーションログ（エラー・警告） | 30〜90日（ホットストレージ）<br>1年（コールドストレージ） | トラブルシューティングに必要な期間<br>長期トレンド分析・監査用 |
| アクセスログ | 7〜30日（ホットストレージ）<br>6ヶ月〜1年（コールドストレージ） | 直近のトラフィック分析用<br>セキュリティインシデント調査用 |
| 監査ログ（金融・医療） | 7年以上（アーカイブストレージ） | 法的要件（SOX法、GDPR等）による |
| デバッグログ | 1〜7日 | 開発環境のみで有効化、本番では無効化が望ましい |

ストレージコスト削減のためのベストプラクティス:
- **段階的ストレージ移行**: Elasticsearch → S3 → Glacier のような階層化
- **サンプリング**: 正常系リクエストの 1-10% のみ記録する
- **構造化ログの圧縮**: JSON ログを gzip で圧縮（70-90% のサイズ削減）
- **古いログの集約**: 詳細ログを日次集計データに変換して保持

また、GDPR 等のプライバシー規制により、個人情報を含むログは削除リクエストに対応できる仕組みが必要である。ログに含まれる PII（個人識別情報）は暗号化またはマスキングを検討する。

---

## 15. モニタリング成熟度モデル

組織のモニタリング成熟度を段階的に向上させるためのロードマップを以下に示す。

```
モニタリング成熟度レベル:

  Level 0: 無監視
    -> ログ出力は console.log のみ
    -> 障害はユーザーからの報告で気付く
    -> 対応: まずログ基盤とヘルスチェックを導入

  Level 1: 基本監視
    -> 構造化ログを導入
    -> アップタイム監視（ping/healthcheck）
    -> 基本的なメトリクス（CPU, メモリ, ディスク）
    -> 対応: RED メトリクスの導入

  Level 2: アプリケーション監視
    -> RED メトリクス（Rate, Errors, Duration）
    -> エンドポイント別のメトリクス
    -> Grafana ダッシュボード
    -> 基本的なアラート設定
    -> 対応: SLI/SLO の定義と分散トレーシング

  Level 3: オブザーバビリティ
    -> SLI/SLO ベースのアラート
    -> 分散トレーシング（OpenTelemetry）
    -> ログ・メトリクス・トレースの相関付け
    -> エラーバジェットによるリリース管理
    -> 対応: 自動化とプロアクティブ監視

  Level 4: プロアクティブ監視
    -> 異常検知（ML ベース）
    -> 自動スケーリングとの連携
    -> カオスエンジニアリングとの統合
    -> SLO ダッシュボードによるビジネス可視化
    -> 継続的な改善サイクル
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| オブザーバビリティ | ログ・メトリクス・トレースの三本柱で構成 |
| RED メソッド | Rate, Errors, Duration でAPIを監視 |
| SLI/SLO | 可用性99.9%、P99 < 500ms、エラーバジェットで意思決定 |
| 構造化ログ | JSON形式 + requestId + traceId で相関可能に |
| 分散トレーシング | OpenTelemetry + W3C Trace Context で標準化 |
| Prometheus | Pull型メトリクス収集、PromQLで柔軟なクエリ |
| Grafana | 階層的ダッシュボード設計、メトリクス・ログ・トレースの統合 |
| アラート設計 | SLOベース、アクション可能、重複排除 |
| ログ集約 | Loki（低コスト）または ELK（全文検索）で集中管理 |
| OTel Collector | ベンダー非依存のテレメトリパイプライン |

---

## FAQ

### Q1: 小規模チームでオブザーバビリティを始めるにはどこから手を付けるべきか?
まずは構造化ログ（JSON形式）の導入と、ヘルスチェックエンドポイントの実装から始めることを推奨する。次に、REDメソッド（Rate、Errors、Duration）の3指標をPrometheus + Grafanaで可視化する。この段階でエラー率とP99レイテンシのアラートを設定すれば、障害の早期検出が可能になる。分散トレーシングはマイクロサービスが3つ以上になった時点で導入を検討するとよい。

### Q2: ログレベル（INFO/WARN/ERROR等）の使い分けの基準は何か?
ERRORはシステムが正常に動作していない状態を示し、即座に調査が必要なもの（DB接続失敗、外部API障害等）に使用する。WARNは現時点では問題ないが将来的にエラーになりうるもの（ディスク容量の逼迫、レート制限の接近等）に使用する。INFOはビジネス上重要なイベント（ユーザー登録、決済完了等）に使用する。DEBUGは開発時のトラブルシューティング情報であり、本番環境ではデフォルトで無効化する。本番環境でDEBUGログを有効にする場合は、動的ログレベル変更の仕組みを導入しておくとよい。

### Q3: SLO（サービスレベル目標）の初期設定値はどの程度にすべきか?
まずは控えめな目標から始め、データを蓄積しながら徐々に引き上げるアプローチを推奨する。初期設定の目安としては、可用性99.9%（月間ダウンタイム約43分）、P99レイテンシ < 1秒、エラー率 < 0.5%が一般的な出発点である。エラーバジェット（100% - SLO）の消費率を監視し、リリース判断に活用する。SLOは四半期ごとにレビューし、実績データに基づいて調整する。99.99%以上の可用性は運用コストが指数関数的に増大するため、ビジネス要件と照合して妥当性を判断すること。

---

## 次に読むべきガイド

→ [APIゲートウェイ](./02-api-gateway.md) — APIゲートウェイの設計とモニタリング統合
→ [レート制限](../03-api-security/01-rate-limiting.md) — レート制限とメトリクスの関連

---

## 参考文献

1. Google. "Site Reliability Engineering: How Google Runs Production Systems." O'Reilly Media, 2016. https://sre.google/sre-book/table-of-contents/
2. OpenTelemetry Authors. "OpenTelemetry Documentation." Cloud Native Computing Foundation, 2024. https://opentelemetry.io/docs/
3. Prometheus Authors. "Prometheus: Monitoring and Alerting Toolkit." Cloud Native Computing Foundation, 2024. https://prometheus.io/docs/introduction/overview/
4. Sridharan, Cindy. "Distributed Systems Observability." O'Reilly Media, 2018. https://www.oreilly.com/library/view/distributed-systems-observability/9781492033431/
5. Grafana Labs. "Grafana Loki Documentation." Grafana Labs, 2024. https://grafana.com/docs/loki/latest/
6. Beyer, Betsy et al. "The Site Reliability Workbook." O'Reilly Media, 2018. https://sre.google/workbook/table-of-contents/
