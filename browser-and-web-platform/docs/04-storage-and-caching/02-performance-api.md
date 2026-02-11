# Performance API

> Performance APIはブラウザのパフォーマンス計測の基盤。Navigation Timing、Resource Timing、User Timing、PerformanceObserverを使い、Core Web Vitalsを含むパフォーマンス指標を計測・分析する。

## この章で学ぶこと

- [ ] Navigation TimingとResource Timingの使い方を理解する
- [ ] User Timingでカスタム計測を行う方法を把握する
- [ ] Core Web Vitalsの計測方法を学ぶ

---

## 1. Navigation Timing

```javascript
// ページ読み込みの各段階を計測
const entry = performance.getEntriesByType('navigation')[0];

const metrics = {
  // リダイレクト
  redirect: entry.redirectEnd - entry.redirectStart,

  // DNS
  dns: entry.domainLookupEnd - entry.domainLookupStart,

  // TCP接続
  tcp: entry.connectEnd - entry.connectStart,

  // TLS
  tls: entry.secureConnectionStart > 0
    ? entry.connectEnd - entry.secureConnectionStart : 0,

  // TTFB（Time to First Byte）
  ttfb: entry.responseStart - entry.requestStart,

  // コンテンツ転送
  download: entry.responseEnd - entry.responseStart,

  // DOM処理
  domProcessing: entry.domContentLoadedEventEnd - entry.responseEnd,

  // 全リソース読み込み
  loadComplete: entry.loadEventEnd - entry.startTime,

  // DOMContentLoaded
  domContentLoaded: entry.domContentLoadedEventEnd - entry.startTime,

  // transfer size
  transferSize: entry.transferSize, // バイト
  encodedBodySize: entry.encodedBodySize,
  decodedBodySize: entry.decodedBodySize,
};

console.table(metrics);
```

---

## 2. Resource Timing

```javascript
// 個別リソースの読み込み時間を計測
const resources = performance.getEntriesByType('resource');

resources.forEach(entry => {
  console.log({
    name: entry.name,
    type: entry.initiatorType, // script, css, img, fetch, xmlhttprequest
    duration: entry.duration.toFixed(0) + 'ms',
    size: entry.transferSize + ' bytes',
    cached: entry.transferSize === 0, // キャッシュヒット
  });
});

// 遅いリソースを検出
const slowResources = resources
  .filter(r => r.duration > 1000)
  .sort((a, b) => b.duration - a.duration);

console.log('Slow resources:', slowResources.map(r => ({
  url: r.name,
  duration: r.duration.toFixed(0) + 'ms',
})));

// リソースタイプ別の集計
const byType = {};
resources.forEach(r => {
  const type = r.initiatorType;
  if (!byType[type]) byType[type] = { count: 0, totalSize: 0, totalDuration: 0 };
  byType[type].count++;
  byType[type].totalSize += r.transferSize;
  byType[type].totalDuration += r.duration;
});
console.table(byType);
```

---

## 3. User Timing

```javascript
// カスタムパフォーマンス計測

// マーク（タイムスタンプの記録）
performance.mark('render-start');

// ... レンダリング処理 ...

performance.mark('render-end');

// 計測（2つのマーク間の時間）
performance.measure('render-time', 'render-start', 'render-end');

const measure = performance.getEntriesByName('render-time')[0];
console.log(`Render time: ${measure.duration.toFixed(2)}ms`);

// 実用例: APIコールの計測
async function fetchWithTiming(url) {
  const markName = `fetch-${url}`;
  performance.mark(`${markName}-start`);

  const response = await fetch(url);
  const data = await response.json();

  performance.mark(`${markName}-end`);
  performance.measure(markName, `${markName}-start`, `${markName}-end`);

  const entry = performance.getEntriesByName(markName)[0];
  console.log(`${url}: ${entry.duration.toFixed(0)}ms`);

  return data;
}

// React コンポーネントのレンダリング時間
function ProfiledComponent({ children }) {
  useEffect(() => {
    performance.mark('component-mount');
    return () => {
      performance.mark('component-unmount');
      performance.measure('component-lifetime', 'component-mount', 'component-unmount');
    };
  }, []);

  return children;
}
```

---

## 4. Core Web Vitals の計測

```javascript
// web-vitals ライブラリ（推奨）
import { onLCP, onINP, onCLS } from 'web-vitals';

onLCP((metric) => {
  console.log('LCP:', metric.value, 'ms');
  // 目標: < 2500ms
  sendToAnalytics({ name: 'LCP', value: metric.value });
});

onINP((metric) => {
  console.log('INP:', metric.value, 'ms');
  // 目標: < 200ms
  sendToAnalytics({ name: 'INP', value: metric.value });
});

onCLS((metric) => {
  console.log('CLS:', metric.value);
  // 目標: < 0.1
  sendToAnalytics({ name: 'CLS', value: metric.value });
});

// 手動計測（PerformanceObserver）
// LCP
new PerformanceObserver((list) => {
  const entries = list.getEntries();
  const lastEntry = entries[entries.length - 1];
  console.log('LCP:', lastEntry.startTime);
}).observe({ type: 'largest-contentful-paint', buffered: true });

// CLS
let clsValue = 0;
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.hadRecentInput) {
      clsValue += entry.value;
    }
  }
}).observe({ type: 'layout-shift', buffered: true });

// Long Tasks（INPのデバッグに有用）
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (entry.duration > 50) {
      console.warn('Long Task:', entry.duration, 'ms');
    }
  }
}).observe({ type: 'longtask' });
```

---

## 5. パフォーマンスデータの送信

```javascript
// Beacon API（ページ離脱時にも確実に送信）
function sendToAnalytics(data) {
  const body = JSON.stringify(data);

  // Beacon API（推奨）
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/analytics', body);
    return;
  }

  // フォールバック
  fetch('/analytics', {
    method: 'POST',
    body,
    keepalive: true, // ページ離脱後も送信を継続
  });
}

// ページ離脱時に全メトリクスを送信
window.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    sendToAnalytics(collectedMetrics);
  }
});

// パフォーマンスバジェット
const budgets = {
  'LCP': 2500,
  'INP': 200,
  'CLS': 0.1,
  'TTFB': 800,
  'bundle-size': 200 * 1024, // 200KB
};
```

---

## まとめ

| API | 用途 |
|-----|------|
| Navigation Timing | ページ読み込みの各段階 |
| Resource Timing | 個別リソースの読み込み |
| User Timing | カスタム計測（mark/measure） |
| PerformanceObserver | リアルタイムのイベント監視 |
| web-vitals | Core Web Vitals の計測 |

---

## 参考文献
1. web.dev. "User-centric performance metrics." Google, 2024.
2. W3C. "Performance Timeline." 2023.
3. web-vitals. "github.com/GoogleChrome/web-vitals." Google, 2024.
