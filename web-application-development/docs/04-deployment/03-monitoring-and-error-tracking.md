# 監視とエラートラッキング

> 本番環境の監視はサービス品質の生命線。Sentry、Web Vitals計測、ロギング、アラート設計まで、本番環境のWebアプリケーションを安定運用するための監視体制を構築する。

## この章で学ぶこと

- [ ] Sentryによるエラートラッキングの設定を理解する
- [ ] Web Vitalsのリアルユーザー計測を把握する
- [ ] ログ戦略とアラート設計を学ぶ

---

## 1. Sentry（エラートラッキング）

```typescript
// Next.js + Sentry セットアップ
// sentry.client.config.ts
import * as Sentry from '@sentry/nextjs';

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 0.1,        // パフォーマンス: 10%サンプリング
  replaysSessionSampleRate: 0.01, // リプレイ: 1%
  replaysOnErrorSampleRate: 1.0,  // エラー時リプレイ: 100%
  integrations: [
    Sentry.replayIntegration(),
    Sentry.browserTracingIntegration(),
  ],
});

// カスタムエラーの送信
try {
  await processPayment(order);
} catch (error) {
  Sentry.captureException(error, {
    tags: { feature: 'payment', orderId: order.id },
    extra: { orderTotal: order.total, userId: user.id },
  });
  throw error;
}

// ユーザーコンテキスト
Sentry.setUser({
  id: user.id,
  email: user.email,
  // ✗ パスワードや機密情報は含めない
});

// ブレッドクラム（ユーザー操作の記録）
Sentry.addBreadcrumb({
  message: 'User clicked checkout',
  category: 'ui',
  level: 'info',
  data: { cartItems: cart.items.length },
});
```

---

## 2. エラーバウンダリ

```typescript
// Next.js App Router の error.tsx
'use client';
import * as Sentry from '@sentry/nextjs';
import { useEffect } from 'react';

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    Sentry.captureException(error);
  }, [error]);

  return (
    <div className="text-center p-8">
      <h2 className="text-xl font-bold">Something went wrong</h2>
      <p className="text-gray-600 mt-2">
        We've been notified and are working on a fix.
      </p>
      <button onClick={reset} className="mt-4 px-4 py-2 bg-blue-500 text-white rounded">
        Try again
      </button>
    </div>
  );
}

// グローバルエラーハンドラー（キャッチされないエラー）
// global-error.tsx
'use client';
export default function GlobalError({ error, reset }) {
  return (
    <html>
      <body>
        <h1>Something went wrong</h1>
        <button onClick={reset}>Reload</button>
      </body>
    </html>
  );
}
```

---

## 3. Web Vitals 計測

```typescript
// Real User Monitoring（RUM）
// app/layout.tsx に組み込み

// web-vitals ライブラリ
import { onLCP, onINP, onCLS, onFCP, onTTFB } from 'web-vitals';

function reportWebVitals() {
  const sendToAnalytics = (metric: any) => {
    // 分析サービスに送信
    fetch('/api/analytics', {
      method: 'POST',
      body: JSON.stringify({
        name: metric.name,
        value: metric.value,
        rating: metric.rating, // 'good', 'needs-improvement', 'poor'
        delta: metric.delta,
        id: metric.id,
        navigationType: metric.navigationType,
      }),
      keepalive: true,
    });
  };

  onLCP(sendToAnalytics);
  onINP(sendToAnalytics);
  onCLS(sendToAnalytics);
  onFCP(sendToAnalytics);
  onTTFB(sendToAnalytics);
}

// Next.js の useReportWebVitals（App Router）
'use client';
import { useReportWebVitals } from 'next/web-vitals';

export function WebVitals() {
  useReportWebVitals((metric) => {
    console.log(metric);
    // Vercel Analytics, Sentry, Google Analytics等に送信
  });
  return null;
}
```

---

## 4. ログ戦略

```typescript
// フロントエンドロガー
type LogLevel = 'debug' | 'info' | 'warn' | 'error';

class Logger {
  private level: LogLevel;

  constructor() {
    this.level = process.env.NODE_ENV === 'production' ? 'warn' : 'debug';
  }

  private shouldLog(level: LogLevel): boolean {
    const levels: LogLevel[] = ['debug', 'info', 'warn', 'error'];
    return levels.indexOf(level) >= levels.indexOf(this.level);
  }

  info(message: string, data?: Record<string, any>) {
    if (!this.shouldLog('info')) return;
    console.info(`[INFO] ${message}`, data);
    // 本番では外部サービスに送信
    if (process.env.NODE_ENV === 'production') {
      this.sendToService('info', message, data);
    }
  }

  error(message: string, error?: Error, data?: Record<string, any>) {
    if (!this.shouldLog('error')) return;
    console.error(`[ERROR] ${message}`, error, data);
    Sentry.captureException(error ?? new Error(message), { extra: data });
  }

  private sendToService(level: string, message: string, data?: any) {
    navigator.sendBeacon('/api/logs', JSON.stringify({ level, message, data, timestamp: new Date().toISOString() }));
  }
}

export const logger = new Logger();
```

---

## 5. アラート設計

```
アラートの優先度:

  P0（即座に対応）:
  → アプリが完全にダウン
  → 決済処理の失敗率 > 5%
  → 認証システムの障害
  → 通知: PagerDuty, Slack

  P1（1時間以内）:
  → エラー率 > 1%
  → レスポンスタイム P99 > 5秒
  → 特定ページの5xxエラー
  → 通知: Slack

  P2（営業時間内）:
  → Web Vitals の劣化
  → 新しいエラーパターンの検出
  → 404エラーの急増
  → 通知: Email, Slack

Sentry のアラートルール:
  → 新しいエラー → Slack通知
  → エラー頻度 > 100/時間 → PagerDuty
  → 特定エラー（PaymentError）→ 即座に通知
  → パフォーマンス劣化（P75 > 4秒）→ Slack
```

---

## 6. モニタリングツール一覧

```
エラートラッキング:
  Sentry:      最も人気、ソースマップ対応
  Bugsnag:     モバイル強い
  LogRocket:   セッションリプレイ

パフォーマンス:
  Vercel Analytics:  Web Vitals（Vercelユーザー向け）
  SpeedCurve:        合成モニタリング + RUM
  web-vitals:        OSS、自前計測

ログ:
  Datadog:     フルスタック監視
  LogDNA:      ログ管理
  Axiom:       コスト効率的

ステータスページ:
  Statuspage:  Atlassian製
  Instatus:    軽量
  Betteruptime: 無料枠あり

推奨スタック:
  Sentry（エラー）+ Vercel Analytics（パフォーマンス）+ Datadog（ログ）
```

---

## まとめ

| 監視項目 | ツール |
|---------|--------|
| エラー | Sentry |
| パフォーマンス | web-vitals, Vercel Analytics |
| ログ | 構造化ログ + Datadog |
| アラート | Sentry Alerts + Slack |
| ステータス | Statuspage |

---

## 参考文献
1. Sentry. "Next.js SDK." docs.sentry.io, 2024.
2. web.dev. "Measure performance with web-vitals." web.dev, 2024.
3. Vercel. "Analytics." vercel.com/docs, 2024.
