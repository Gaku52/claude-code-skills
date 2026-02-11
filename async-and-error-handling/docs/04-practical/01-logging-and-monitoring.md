# ログとモニタリング

> エラーは発生する。重要なのは「素早く検知し、原因を特定し、修正する」こと。構造化ログ、エラートラッキング（Sentry）、アラート設計のベストプラクティスを解説。

## この章で学ぶこと

- [ ] 構造化ログの設計を理解する
- [ ] エラートラッキングサービスの活用を把握する
- [ ] 効果的なアラート設計を学ぶ

---

## 1. 構造化ログ

```
非構造化ログ（従来）:
  [2025-01-15 10:30:45] ERROR: Failed to process order 12345 for user abc

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
```

```typescript
// 構造化ロガーの実装
import pino from 'pino';

const logger = pino({
  level: process.env.LOG_LEVEL ?? 'info',
  formatters: {
    level(label) { return { level: label }; },
  },
  timestamp: pino.stdTimeFunctions.isoTime,
});

// 使い方
logger.info({ orderId: "12345", userId: "abc" }, "Order created");

logger.error(
  {
    orderId: "12345",
    error: { name: err.name, message: err.message, code: err.code },
    duration_ms: Date.now() - startTime,
  },
  "Order processing failed"
);

// リクエストスコープのロガー（トレースID付き）
function createRequestLogger(req: Request) {
  return logger.child({
    traceId: req.headers['x-trace-id'] ?? crypto.randomUUID(),
    method: req.method,
    path: req.path,
    ip: req.ip,
  });
}
```

---

## 2. ログレベル

```
┌─────────┬──────────────────────────────────────────┐
│ レベル  │ 用途                                      │
├─────────┼──────────────────────────────────────────┤
│ fatal   │ アプリケーション停止を伴うエラー            │
│ error   │ 操作の失敗。ユーザーに影響するエラー        │
│ warn    │ 潜在的な問題。今は動くが注意が必要          │
│ info    │ 重要なビジネスイベント（注文完了等）         │
│ debug   │ 開発時のデバッグ情報                       │
│ trace   │ 詳細なトレース情報                         │
└─────────┴──────────────────────────────────────────┘

本番: info 以上
ステージング: debug 以上
開発: trace 以上
```

---

## 3. エラートラッキング（Sentry）

```typescript
// Sentry: エラートラッキング
import * as Sentry from '@sentry/node';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 0.1, // 10%のトランザクションをトレース
});

// エラーの送信
try {
  await processOrder(orderId);
} catch (error) {
  Sentry.withScope(scope => {
    scope.setTag("feature", "order-processing");
    scope.setContext("order", { orderId, userId });
    scope.setUser({ id: userId, email: userEmail });
    Sentry.captureException(error);
  });
  throw error;
}

// Express ミドルウェア
app.use(Sentry.Handlers.requestHandler());
app.use(Sentry.Handlers.errorHandler());

// パフォーマンスモニタリング
const transaction = Sentry.startTransaction({ name: "processOrder" });
const span = transaction.startChild({ op: "db.query" });
await db.query(...);
span.finish();
transaction.finish();
```

---

## 4. アラート設計

```
アラートの原則:
  1. アクション可能（受けたら何かできる）
  2. 低ノイズ（誤報が少ない）
  3. 適切な宛先（オンコール担当者）

エラー率ベース:
  → 5xxエラー率 > 1% → Warning
  → 5xxエラー率 > 5% → Critical

レイテンシベース:
  → P95 > 2秒 → Warning
  → P99 > 5秒 → Critical

ビジネスメトリクス:
  → 決済成功率 < 95% → Critical
  → 注文数が前時間比 50% 減 → Warning
```

---

## まとめ

| 手法 | 目的 | ツール例 |
|------|------|---------|
| 構造化ログ | 検索・分析可能なログ | pino, winston |
| エラートラッキング | エラーの集約・通知 | Sentry, Datadog |
| メトリクス | 数値データの監視 | Prometheus, Grafana |
| アラート | 異常の即時通知 | PagerDuty, Slack |

---

## 次に読むべきガイド
→ [[02-testing-async.md]] — 非同期テスト

---

## 参考文献
1. Sentry Documentation. docs.sentry.io.
2. Google SRE Book. "Monitoring Distributed Systems."
