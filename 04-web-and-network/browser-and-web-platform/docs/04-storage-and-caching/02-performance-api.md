# Performance API

> Performance API はブラウザのパフォーマンス計測の基盤である。Navigation Timing、Resource Timing、User Timing、PerformanceObserver を用いて Core Web Vitals を含むパフォーマンス指標を計測・分析し、Lighthouse や RUM（Real User Monitoring）と統合することで、継続的なパフォーマンス改善サイクルを構築できる。

---

## 目次

1. [Performance API の全体像](#1-performance-api-の全体像)
2. [Navigation Timing API](#2-navigation-timing-api)
3. [Resource Timing API](#3-resource-timing-api)
4. [User Timing API](#4-user-timing-api)
5. [PerformanceObserver の活用](#5-performanceobserver-の活用)
6. [Core Web Vitals の計測](#6-core-web-vitals-の計測)
7. [Lighthouse とパフォーマンス監査](#7-lighthouse-とパフォーマンス監査)
8. [パフォーマンスデータの送信と分析基盤](#8-パフォーマンスデータの送信と分析基盤)
9. [パフォーマンスバジェットと CI 統合](#9-パフォーマンスバジェットと-ci-統合)
10. [アンチパターンと回避策](#10-アンチパターンと回避策)
11. [エッジケース分析](#11-エッジケース分析)
12. [段階別演習](#12-段階別演習)
13. [FAQ](#13-faq)
14. [比較表](#14-比較表)
15. [参考文献](#15-参考文献)

---

## 前提知識

本章を最大限に活用するため、以下の前提知識を習得しておくことを推奨する。

- **Service Worker とキャッシュの理解**: Performance API で計測するパフォーマンス指標の多くは、Service Worker によるキャッシュ戦略の影響を受ける。Cache First や Stale-While-Revalidate がリソースタイミングや Core Web Vitals（特に LCP）にどのように作用するかを理解するため、[Service Worker とキャッシュ戦略](./01-service-worker-cache.md) の内容を事前に把握しておくことが望ましい。
- **レンダリングパイプラインの基礎**: Largest Contentful Paint（LCP）や Cumulative Layout Shift（CLS）などの指標は、ブラウザのレンダリングプロセスと密接に関連している。パースからレイアウト、ペイント、コンポジットに至るまでの流れを理解していると、パフォーマンス指標の背景にある仕組みが明確になる。詳細は [レンダリングパイプライン](../01-rendering/00-rendering-pipeline.md) を参照のこと。
- **Core Web Vitals の概念**: Google が定義する LCP（Largest Contentful Paint）、INP（Interaction to Next Paint）、CLS（Cumulative Layout Shift）の 3 つの指標は、ユーザー体験を定量化する標準的な指標である。これらの基本概念を理解していることで、本章の計測・改善手法がより実践的に活用できる。

これらの知識がない場合でも本章を読み進めることは可能だが、上記のガイドを先に参照することで理解が深まる。

---

## この章で学ぶこと

- [ ] Navigation Timing と Resource Timing の使い方を理解する
- [ ] User Timing でカスタム計測を行う方法を把握する
- [ ] PerformanceObserver によるリアルタイム監視の仕組みを学ぶ
- [ ] Core Web Vitals（LCP・INP・CLS）の計測と改善手法を習得する
- [ ] Lighthouse のスコアリングアルゴリズムと自動監査の活用法を理解する
- [ ] パフォーマンスバジェットの設計と CI/CD パイプラインへの組み込みを実践する
- [ ] RUM データの収集・送信・分析基盤を構築する方法を学ぶ

---

## 1. Performance API の全体像

### 1.1 アーキテクチャ概要

Performance API は W3C が策定する一連の仕様群であり、ブラウザにおけるパフォーマンス計測の標準基盤を提供する。以下の ASCII 図は、Performance API を構成する主要な仕様とその関係性を示している。

```
+-------------------------------------------------------------------+
|                    Performance Timeline                            |
|  (performance.getEntries / performance.getEntriesByType)          |
+-------------------------------------------------------------------+
        |              |              |              |
        v              v              v              v
+-------------+ +-------------+ +-------------+ +-------------+
| Navigation  | | Resource    | | User        | | Paint       |
| Timing      | | Timing      | | Timing      | | Timing      |
| (navigate,  | | (script,    | | (mark,      | | (first-paint|
|  reload,    | |  css, img,  | |  measure)   | |  first-     |
|  back_fwd)  | |  fetch ...) | |             | |  contentful)|
+-------------+ +-------------+ +-------------+ +-------------+
        |              |              |              |
        v              v              v              v
+-------------------------------------------------------------------+
|                   PerformanceObserver                              |
|  (リアルタイムのエントリ通知・buffered オプション)                  |
+-------------------------------------------------------------------+
        |
        v
+-------------------------------------------------------------------+
|  Analytics / RUM 基盤                                              |
|  (Beacon API / fetch keepalive / サードパーティ SDK)               |
+-------------------------------------------------------------------+
```

### 1.2 Performance Timeline の基本概念

Performance Timeline は全てのパフォーマンスエントリを統一的に扱う仕組みである。各エントリは `PerformanceEntry` インターフェースを継承し、共通のプロパティを持つ。

| プロパティ       | 型       | 説明                                 |
|------------------|----------|--------------------------------------|
| `name`           | string   | エントリの識別名（URL やマーク名）   |
| `entryType`      | string   | エントリの種別                       |
| `startTime`      | double   | 計測開始時刻（ミリ秒、timeOrigin 基準） |
| `duration`       | double   | 計測期間（ミリ秒）                   |

```javascript
// Performance Timeline の基本操作
// 全てのエントリを取得
const allEntries = performance.getEntries();

// 種別でフィルタリング
const navEntries = performance.getEntriesByType('navigation');
const resEntries = performance.getEntriesByType('resource');
const markEntries = performance.getEntriesByType('mark');
const measureEntries = performance.getEntriesByType('measure');

// 名前で検索
const specificEntry = performance.getEntriesByName('my-custom-mark');

// タイムオリジンの確認
console.log('Time origin:', performance.timeOrigin);
// => Unix エポックからの経過ミリ秒（高精度タイムスタンプ）

// 現在の高精度タイムスタンプ
console.log('Now:', performance.now());
// => timeOrigin からの経過ミリ秒
```

### 1.3 高精度タイムスタンプと Spectre 緩和

`performance.now()` はマイクロ秒精度のタイムスタンプを返すが、Spectre 等のサイドチャネル攻撃への対策として、ブラウザはタイムスタンプの精度を意図的に下げている。

```
┌─────────────────────────────────────────────────────┐
│  Spectre 緩和前後のタイムスタンプ精度                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  緩和前:  performance.now() => 1234.567890123       │
│                                 ^^^^^^^^^^^^^^^^    │
│                                 マイクロ秒精度      │
│                                                     │
│  緩和後:  performance.now() => 1234.500             │
│           (Cross-Origin-Isolated なし)              │
│                                 ^^^^^^^             │
│                                 100μs に丸め        │
│                                                     │
│  COOP+COEP 設定時:                                  │
│           performance.now() => 1234.567             │
│                                 ^^^^^^^^^           │
│                                 5μs 精度に回復      │
│                                                     │
│  ※ SharedArrayBuffer を使う場合も同様の設定が必要   │
└─────────────────────────────────────────────────────┘
```

Cross-Origin-Isolated 環境を有効化するには、以下の HTTP ヘッダを設定する。

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

---

## 2. Navigation Timing API

### 2.1 ページ読み込みライフサイクル

Navigation Timing API は、ブラウザがページを読み込む過程の各段階を計測する仕組みである。以下の図はページ読み込みの全段階を時系列で示す。

```
 navigationStart
       |
       v
 +-----------+    +-----------+    +----------+    +----------+
 | Redirect  |--->|   DNS     |--->|   TCP    |--->|   TLS    |
 | (0-N回)   |    |  Lookup   |    | Connect  |    | Handshake|
 +-----------+    +-----------+    +----------+    +----------+
       |                                                 |
       v                                                 v
 +-------------+    +-----------+    +-----------+    +----------+
 | Request     |--->| Response  |--->|  DOM      |--->|  Load    |
 | Send        |    | Receive   |    | Processing|    | Event    |
 | (TTFB算出)  |    | (download)|    | (parse +  |    | (onload) |
 +-------------+    +-----------+    |  scripts) |    +----------+
                                     +-----------+
                                          |
                                          v
                                  DOMContentLoaded
```

### 2.2 詳細な計測実装

```javascript
// ページ読み込みの各段階を体系的に計測する関数
function collectNavigationMetrics() {
  const entry = performance.getEntriesByType('navigation')[0];

  if (!entry) {
    console.warn('Navigation Timing エントリが取得できません');
    return null;
  }

  const metrics = {
    // ========== ネットワーク段階 ==========
    // リダイレクト処理時間（HTTP 301/302 など）
    redirect: {
      duration: entry.redirectEnd - entry.redirectStart,
      count: entry.redirectCount,
      note: 'リダイレクトが多いと初期表示が遅延する',
    },

    // DNS 解決時間
    dns: {
      duration: entry.domainLookupEnd - entry.domainLookupStart,
      note: 'DNS プリフェッチで短縮可能',
    },

    // TCP 接続確立時間
    tcp: {
      duration: entry.connectEnd - entry.connectStart,
      note: 'HTTP/2 の多重化で接続再利用が可能',
    },

    // TLS ハンドシェイク時間
    tls: {
      duration: entry.secureConnectionStart > 0
        ? entry.connectEnd - entry.secureConnectionStart
        : 0,
      isSecure: entry.secureConnectionStart > 0,
      note: 'TLS 1.3 で 1-RTT に短縮可能',
    },

    // ========== サーバー応答段階 ==========
    // TTFB（Time to First Byte）
    ttfb: {
      duration: entry.responseStart - entry.requestStart,
      threshold: 800, // ms - 推奨上限
      note: 'サーバー処理時間を含む重要指標',
    },

    // コンテンツダウンロード時間
    download: {
      duration: entry.responseEnd - entry.responseStart,
      transferSize: entry.transferSize,
      encodedBodySize: entry.encodedBodySize,
      decodedBodySize: entry.decodedBodySize,
      compressionRatio: entry.encodedBodySize > 0
        ? (1 - entry.encodedBodySize / entry.decodedBodySize).toFixed(2)
        : 'N/A',
    },

    // ========== レンダリング段階 ==========
    // DOM 処理時間
    domProcessing: {
      duration: entry.domContentLoadedEventEnd - entry.responseEnd,
      interactive: entry.domInteractive - entry.startTime,
      contentLoaded: entry.domContentLoadedEventEnd - entry.startTime,
    },

    // 全体の読み込み完了時間
    total: {
      loadComplete: entry.loadEventEnd - entry.startTime,
      domContentLoaded: entry.domContentLoadedEventEnd - entry.startTime,
    },

    // ========== メタ情報 ==========
    meta: {
      type: entry.type,          // 'navigate', 'reload', 'back_forward', 'prerender'
      protocol: entry.nextHopProtocol, // 'h2', 'h3', 'http/1.1'
      redirectCount: entry.redirectCount,
    },
  };

  return metrics;
}

// 使用例: ページ読み込み完了後に実行
window.addEventListener('load', () => {
  // loadEventEnd が記録されるのを待つ
  setTimeout(() => {
    const metrics = collectNavigationMetrics();
    if (metrics) {
      console.table({
        'リダイレクト': `${metrics.redirect.duration.toFixed(0)}ms (${metrics.redirect.count}回)`,
        'DNS 解決': `${metrics.dns.duration.toFixed(0)}ms`,
        'TCP 接続': `${metrics.tcp.duration.toFixed(0)}ms`,
        'TLS ハンドシェイク': `${metrics.tls.duration.toFixed(0)}ms`,
        'TTFB': `${metrics.ttfb.duration.toFixed(0)}ms`,
        'ダウンロード': `${metrics.download.duration.toFixed(0)}ms`,
        'DOM 処理': `${metrics.domProcessing.duration.toFixed(0)}ms`,
        '全体': `${metrics.total.loadComplete.toFixed(0)}ms`,
        'プロトコル': metrics.meta.protocol,
        'ナビゲーション種別': metrics.meta.type,
      });
    }
  }, 0);
});
```

### 2.3 Navigation Type の判別と活用

`PerformanceNavigationTiming.type` プロパティは、ページ遷移の種類を示す。これを用いてナビゲーション種別ごとの分析が可能になる。

| type 値        | 意味                             | 典型的なシナリオ                   |
|----------------|----------------------------------|------------------------------------|
| `navigate`     | 通常のナビゲーション             | リンククリック、アドレスバー入力   |
| `reload`       | ページリロード                   | F5、Ctrl+R                         |
| `back_forward` | 履歴ナビゲーション               | ブラウザの「戻る」「進む」ボタン   |
| `prerender`    | プリレンダリング                 | Speculation Rules API による投機的読み込み |

```javascript
// ナビゲーション種別ごとにメトリクスを分類して送信
function categorizeByNavigationType(metrics) {
  const entry = performance.getEntriesByType('navigation')[0];
  const navType = entry?.type || 'unknown';

  return {
    navigationType: navType,
    metrics,
    // 戻る/進む操作では bfcache の効果を確認
    bfcacheUsed: navType === 'back_forward' && metrics.total.loadComplete < 50,
    // プリレンダリング済みならほぼ即時表示
    prerendered: navType === 'prerender',
  };
}
```

---

## 3. Resource Timing API

### 3.1 リソース読み込みの詳細計測

Resource Timing API は、HTML ドキュメント以外の個別リソース（スクリプト、スタイルシート、画像、フォント、API リクエストなど）の読み込みパフォーマンスを計測する。

```javascript
// 個別リソースの読み込み時間を詳細に計測する
function analyzeResources() {
  const resources = performance.getEntriesByType('resource');

  // リソースごとの詳細分析
  const analysis = resources.map(entry => ({
    // 基本情報
    url: entry.name,
    type: entry.initiatorType,
    // script, css, img, link, fetch, xmlhttprequest, beacon, video, audio

    // タイミング詳細
    timing: {
      redirect: entry.redirectEnd - entry.redirectStart,
      dns: entry.domainLookupEnd - entry.domainLookupStart,
      tcp: entry.connectEnd - entry.connectStart,
      tls: entry.secureConnectionStart > 0
        ? entry.connectEnd - entry.secureConnectionStart : 0,
      ttfb: entry.responseStart - entry.requestStart,
      download: entry.responseEnd - entry.responseStart,
      total: entry.duration,
    },

    // サイズ情報
    size: {
      transferSize: entry.transferSize,
      encodedBodySize: entry.encodedBodySize,
      decodedBodySize: entry.decodedBodySize,
    },

    // キャッシュ判定
    cache: {
      fromCache: entry.transferSize === 0 && entry.decodedBodySize > 0,
      fromServiceWorker: entry.workerStart > 0,
      // 304 Not Modified の検出
      conditionalRequest: entry.transferSize > 0
        && entry.transferSize < entry.encodedBodySize,
    },

    // HTTP/2 Server Push の検出
    serverPush: entry.transferSize > 0
      && entry.requestStart === entry.responseStart,

    // レンダリングブロッキング判定
    renderBlocking: entry.renderBlockingStatus || 'unknown',
  }));

  return analysis;
}

// 遅いリソースの検出と報告
function detectSlowResources(thresholdMs = 1000) {
  const resources = performance.getEntriesByType('resource');

  const slowResources = resources
    .filter(r => r.duration > thresholdMs)
    .sort((a, b) => b.duration - a.duration)
    .map(r => ({
      url: new URL(r.name).pathname,  // パスのみ表示
      duration: `${r.duration.toFixed(0)}ms`,
      type: r.initiatorType,
      size: `${(r.transferSize / 1024).toFixed(1)}KB`,
      bottleneck: identifyBottleneck(r),
    }));

  return slowResources;
}

// ボトルネックの自動判定
function identifyBottleneck(entry) {
  const timing = {
    dns: entry.domainLookupEnd - entry.domainLookupStart,
    tcp: entry.connectEnd - entry.connectStart,
    ttfb: entry.responseStart - entry.requestStart,
    download: entry.responseEnd - entry.responseStart,
  };

  const max = Object.entries(timing)
    .reduce((a, b) => a[1] > b[1] ? a : b);

  return { phase: max[0], duration: `${max[1].toFixed(0)}ms` };
}
```

### 3.2 リソースタイプ別の集計ダッシュボード

```javascript
// リソースタイプ別に集計してパフォーマンスの全体像を把握する
function createResourceDashboard() {
  const resources = performance.getEntriesByType('resource');

  const dashboard = {};

  resources.forEach(r => {
    const type = r.initiatorType || 'other';

    if (!dashboard[type]) {
      dashboard[type] = {
        count: 0,
        totalSize: 0,
        totalDuration: 0,
        maxDuration: 0,
        cachedCount: 0,
        entries: [],
      };
    }

    const group = dashboard[type];
    group.count++;
    group.totalSize += r.transferSize;
    group.totalDuration += r.duration;
    group.maxDuration = Math.max(group.maxDuration, r.duration);

    if (r.transferSize === 0 && r.decodedBodySize > 0) {
      group.cachedCount++;
    }

    group.entries.push(r);
  });

  // 集計結果のフォーマット
  const summary = Object.entries(dashboard).map(([type, data]) => ({
    type,
    count: data.count,
    totalSize: `${(data.totalSize / 1024).toFixed(1)}KB`,
    avgDuration: `${(data.totalDuration / data.count).toFixed(0)}ms`,
    maxDuration: `${data.maxDuration.toFixed(0)}ms`,
    cacheHitRate: `${((data.cachedCount / data.count) * 100).toFixed(0)}%`,
  }));

  console.table(summary);
  return summary;
}
```

### 3.3 Timing-Allow-Origin とクロスオリジン制約

クロスオリジンリソースの計測には、サーバー側で `Timing-Allow-Origin` ヘッダの設定が必要である。このヘッダが設定されていない場合、多くのタイミング値がゼロに制限される。

```
クロスオリジンリソースのタイミング情報制約:

┌──────────────────────────────────────────────────────────────┐
│  Timing-Allow-Origin ヘッダなし（デフォルト）               │
│                                                              │
│  取得可能:                                                   │
│    - startTime, duration                                     │
│    - transferSize = 0 (隠蔽)                                │
│    - encodedBodySize = 0 (隠蔽)                             │
│    - decodedBodySize = 0 (隠蔽)                             │
│                                                              │
│  ゼロに制限:                                                 │
│    - redirectStart / redirectEnd                             │
│    - domainLookupStart / domainLookupEnd                     │
│    - connectStart / connectEnd                               │
│    - secureConnectionStart                                   │
│    - requestStart / responseStart                            │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  Timing-Allow-Origin: * （または特定オリジン）              │
│                                                              │
│  全てのタイミング値が取得可能                                │
│  サイズ情報も正確に報告される                                │
└──────────────────────────────────────────────────────────────┘
```

```javascript
// Timing-Allow-Origin の確認と影響の検出
function checkTimingAccess() {
  const resources = performance.getEntriesByType('resource');
  const crossOrigin = resources.filter(r => {
    try {
      return new URL(r.name).origin !== location.origin;
    } catch {
      return false;
    }
  });

  const restricted = crossOrigin.filter(r =>
    r.requestStart === 0 && r.responseStart === 0
  );

  console.log(`クロスオリジンリソース: ${crossOrigin.length}件`);
  console.log(`タイミング制限あり: ${restricted.length}件`);
  console.log('制限リソース一覧:');
  restricted.forEach(r => {
    console.log(`  - ${new URL(r.name).hostname}${new URL(r.name).pathname}`);
  });
}
```

---

## 4. User Timing API

### 4.1 mark と measure の基本

User Timing API は、開発者がアプリケーション固有のパフォーマンス計測ポイントを定義するための仕組みである。`performance.mark()` でタイムスタンプを記録し、`performance.measure()` で2点間の所要時間を算出する。

```javascript
// ==================================================
// User Timing API の基本操作と応用パターン
// ==================================================

// (1) 基本的な mark と measure
performance.mark('app-init-start');

// アプリケーション初期化処理
initializeApp();
loadConfiguration();
setupEventHandlers();

performance.mark('app-init-end');

// 2つの mark 間の時間を計測
performance.measure('app-initialization', 'app-init-start', 'app-init-end');

const initMeasure = performance.getEntriesByName('app-initialization')[0];
console.log(`アプリ初期化: ${initMeasure.duration.toFixed(2)}ms`);


// (2) mark にメタデータを付与（Performance API Level 3）
performance.mark('data-fetch-complete', {
  detail: {
    endpoint: '/api/users',
    recordCount: 150,
    cacheHit: false,
  },
});

// メタデータの取得
const fetchMark = performance.getEntriesByName('data-fetch-complete')[0];
console.log('取得件数:', fetchMark.detail.recordCount);


// (3) measure にもメタデータを付与可能
performance.measure('api-call', {
  start: 'api-call-start',
  end: 'api-call-end',
  detail: {
    url: '/api/products',
    method: 'GET',
    status: 200,
  },
});


// (4) navigationStart からの経過時間を計測
performance.measure('time-to-interactive', {
  start: 0,  // navigationStart を起点とする
  end: performance.now(),
});


// (5) エントリのクリーンアップ
performance.clearMarks('app-init-start');
performance.clearMarks('app-init-end');
performance.clearMeasures('app-initialization');

// 全エントリのクリア
performance.clearMarks();     // 全 mark を削除
performance.clearMeasures();  // 全 measure を削除
```

### 4.2 実用的な計測パターン集

```javascript
// ==================================================
// パターン1: 非同期処理の計測ラッパー
// ==================================================
async function measureAsync(name, asyncFn) {
  const markStart = `${name}-start`;
  const markEnd = `${name}-end`;

  performance.mark(markStart);

  try {
    const result = await asyncFn();

    performance.mark(markEnd);
    performance.measure(name, markStart, markEnd);

    const entry = performance.getEntriesByName(name).pop();
    console.log(`[Perf] ${name}: ${entry.duration.toFixed(1)}ms`);

    return result;
  } catch (error) {
    performance.mark(markEnd);
    performance.measure(`${name}-failed`, markStart, markEnd);

    const entry = performance.getEntriesByName(`${name}-failed`).pop();
    console.error(`[Perf] ${name} FAILED: ${entry.duration.toFixed(1)}ms`);

    throw error;
  }
}

// 使用例
const users = await measureAsync('fetch-users', () =>
  fetch('/api/users').then(r => r.json())
);


// ==================================================
// パターン2: React コンポーネントのライフサイクル計測
// ==================================================
function usePerformanceMark(componentName) {
  const markPrefix = `component-${componentName}`;

  useEffect(() => {
    performance.mark(`${markPrefix}-mount`);

    return () => {
      performance.mark(`${markPrefix}-unmount`);
      performance.measure(
        `${markPrefix}-lifetime`,
        `${markPrefix}-mount`,
        `${markPrefix}-unmount`
      );

      const entry = performance.getEntriesByName(
        `${markPrefix}-lifetime`
      ).pop();
      console.log(
        `${componentName} lifetime: ${entry.duration.toFixed(0)}ms`
      );
    };
  }, []);
}

// 使用例
function ProductList() {
  usePerformanceMark('ProductList');

  return (
    <ul>
      {products.map(p => <ProductItem key={p.id} product={p} />)}
    </ul>
  );
}


// ==================================================
// パターン3: ルーティング遷移の計測
// ==================================================
class RoutePerformanceTracker {
  constructor() {
    this.currentRoute = null;
    this.transitionCount = 0;
  }

  startTransition(fromRoute, toRoute) {
    this.transitionCount++;
    const id = `route-transition-${this.transitionCount}`;

    performance.mark(`${id}-start`, {
      detail: { from: fromRoute, to: toRoute },
    });

    this.currentRoute = { id, from: fromRoute, to: toRoute };
  }

  endTransition() {
    if (!this.currentRoute) return;

    const { id } = this.currentRoute;

    performance.mark(`${id}-end`);
    performance.measure(id, `${id}-start`, `${id}-end`);

    const entry = performance.getEntriesByName(id).pop();
    console.log(
      `Route ${this.currentRoute.from} -> ${this.currentRoute.to}: ` +
      `${entry.duration.toFixed(0)}ms`
    );

    this.currentRoute = null;
    return entry;
  }
}
```

---

## 5. PerformanceObserver の活用

### 5.1 基本的な使い方

PerformanceObserver は、パフォーマンスエントリが記録されるたびにコールバックを呼び出す Observer パターンの実装である。ポーリングではなくイベント駆動でエントリを取得できるため、効率的かつリアルタイムな計測が可能になる。

```javascript
// ==================================================
// PerformanceObserver の基本操作
// ==================================================

// (1) 特定の entryType を監視
const resourceObserver = new PerformanceObserver((list, observer) => {
  const entries = list.getEntries();

  entries.forEach(entry => {
    console.log(`[Resource] ${entry.name}: ${entry.duration.toFixed(0)}ms`);
  });
});

// observe の開始
resourceObserver.observe({
  type: 'resource',
  buffered: true,  // 過去のエントリも含める
});


// (2) 複数の entryType を同時に監視
const multiObserver = new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    switch (entry.entryType) {
      case 'resource':
        handleResourceEntry(entry);
        break;
      case 'mark':
        handleMarkEntry(entry);
        break;
      case 'measure':
        handleMeasureEntry(entry);
        break;
    }
  }
});

// entryTypes（複数形）で配列を渡す
multiObserver.observe({
  entryTypes: ['resource', 'mark', 'measure'],
});


// (3) 監視の停止
resourceObserver.disconnect();


// (4) 監視可能な entryType の一覧を取得
const supportedTypes = PerformanceObserver.supportedEntryTypes;
console.log('サポートされている entryType:', supportedTypes);
// 典型的な出力:
// ['element', 'event', 'first-input', 'largest-contentful-paint',
//  'layout-shift', 'longtask', 'mark', 'measure', 'navigation',
//  'paint', 'resource', 'visibility-state']
```

### 5.2 buffered オプションの重要性

`buffered: true` は、Observer の登録前に記録されたエントリも取得するオプションである。ページ読み込み完了後にスクリプトを実行するケース（defer や async で読み込まれるスクリプト）では、このオプションがないと初期のエントリを取りこぼす。

```javascript
// buffered オプションの効果
//
// スクリプト実行時点: T = 3000ms
// ページ読み込み開始: T = 0ms
//
// buffered: false の場合:
//   T=0 ~ T=3000 に発生したエントリは取得できない
//
// buffered: true の場合:
//   T=0 ~ T=3000 に発生したエントリも含めて取得可能

// 典型的な使い方（遅延読み込みスクリプトで安全に使用）
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    // 過去のエントリも含めて処理される
    processEntry(entry);
  }
}).observe({ type: 'largest-contentful-paint', buffered: true });
```

### 5.3 Long Tasks の検出

メインスレッドで 50ms を超えるタスクは "Long Task" と定義され、ユーザー入力への応答遅延を引き起こす。PerformanceObserver を用いてこれらを検出し、インタラクション品質の低下原因を特定できる。

```javascript
// Long Tasks の監視と分析
const longTasks = [];

new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    longTasks.push({
      startTime: entry.startTime,
      duration: entry.duration,
      // attribution で原因のスクリプトを特定
      attribution: entry.attribution?.map(attr => ({
        name: attr.name,
        containerType: attr.containerType,
        containerSrc: attr.containerSrc,
        containerId: attr.containerId,
        containerName: attr.containerName,
      })),
    });

    if (entry.duration > 100) {
      console.warn(
        `[Long Task] ${entry.duration.toFixed(0)}ms at ` +
        `T+${entry.startTime.toFixed(0)}ms`
      );
    }
  }
}).observe({ type: 'longtask', buffered: true });

// 一定間隔でサマリーを出力
setInterval(() => {
  if (longTasks.length === 0) return;

  const total = longTasks.reduce((sum, t) => sum + t.duration, 0);
  const avg = total / longTasks.length;
  const max = Math.max(...longTasks.map(t => t.duration));

  console.log(`Long Tasks サマリー:
    件数: ${longTasks.length}
    合計: ${total.toFixed(0)}ms
    平均: ${avg.toFixed(0)}ms
    最大: ${max.toFixed(0)}ms
  `);
}, 10000);
```

---

## 6. Core Web Vitals の計測

### 6.1 Core Web Vitals の概要

Core Web Vitals は Google が定義した、ユーザー体験の品質を評価する3つの主要指標である。2024年以降、INP（Interaction to Next Paint）が FID（First Input Delay）を正式に置き換えた。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Core Web Vitals (2024~)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LCP (Largest Contentful Paint)                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ページの主要コンテンツが表示されるまでの時間           │   │
│  │                                                         │   │
│  │  Good        Needs Improvement     Poor                 │   │
│  │  |<--- 2.5s --->|<--- 4.0s --->|<--- ... --->|         │   │
│  │  [  緑: 良好  ] [ 黄: 改善必要 ] [  赤: 不良 ]         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  INP (Interaction to Next Paint)                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ユーザー操作から画面更新までの応答時間                 │   │
│  │                                                         │   │
│  │  Good        Needs Improvement     Poor                 │   │
│  │  |<--- 200ms --->|<--- 500ms --->|<--- ... --->|       │   │
│  │  [  緑: 良好  ]  [ 黄: 改善必要 ] [  赤: 不良 ]       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  CLS (Cumulative Layout Shift)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  視覚的な安定性（レイアウトのずれの累積値）             │   │
│  │                                                         │   │
│  │  Good        Needs Improvement     Poor                 │   │
│  │  |<--- 0.1 --->|<--- 0.25 --->|<--- ... --->|         │   │
│  │  [ 緑: 良好 ]  [ 黄: 改善必要 ] [ 赤: 不良  ]         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 web-vitals ライブラリによる計測

Google が公式に提供する `web-vitals` ライブラリは、Core Web Vitals の計測を標準化された方法で行うための推奨ツールである。

```javascript
// ==================================================
// web-vitals ライブラリを使った包括的な計測
// ==================================================
import { onLCP, onINP, onCLS, onFCP, onTTFB } from 'web-vitals';

// (1) 基本的な使い方
onLCP((metric) => {
  console.log('LCP:', metric.value, 'ms');
  console.log('  評価:', metric.rating);       // 'good' | 'needs-improvement' | 'poor'
  console.log('  要素:', metric.entries);       // LCP の対象要素
  console.log('  delta:', metric.delta, 'ms');  // 前回からの変化量
  console.log('  ID:', metric.id);              // 一意な識別子
  console.log('  navigationType:', metric.navigationType);
  sendToAnalytics({ name: 'LCP', ...metric });
});

onINP((metric) => {
  console.log('INP:', metric.value, 'ms');
  console.log('  評価:', metric.rating);
  // INP の entries には最も遅かったインタラクションが含まれる
  const worstEntry = metric.entries[0];
  if (worstEntry) {
    console.log('  イベント種別:', worstEntry.name);
    console.log('  処理時間:', worstEntry.processingEnd - worstEntry.processingStart, 'ms');
    console.log('  入力遅延:', worstEntry.processingStart - worstEntry.startTime, 'ms');
    console.log('  描画遅延:', worstEntry.startTime + worstEntry.duration - worstEntry.processingEnd, 'ms');
  }
  sendToAnalytics({ name: 'INP', ...metric });
});

onCLS((metric) => {
  console.log('CLS:', metric.value);
  console.log('  評価:', metric.rating);
  // CLS の entries にはレイアウトシフトの詳細が含まれる
  metric.entries.forEach(entry => {
    console.log('  シフト値:', entry.value);
    console.log('  最近の入力:', entry.hadRecentInput);
    // シフトした要素の特定
    entry.sources?.forEach(source => {
      console.log('  要素:', source.node);
      console.log('  移動前:', source.previousRect);
      console.log('  移動後:', source.currentRect);
    });
  });
  sendToAnalytics({ name: 'CLS', ...metric });
});

// (2) 補助指標の計測
onFCP((metric) => {
  console.log('FCP:', metric.value, 'ms');
  sendToAnalytics({ name: 'FCP', ...metric });
});

onTTFB((metric) => {
  console.log('TTFB:', metric.value, 'ms');
  sendToAnalytics({ name: 'TTFB', ...metric });
});

// (3) 分析データの送信関数
function sendToAnalytics(data) {
  const payload = JSON.stringify({
    name: data.name,
    value: data.value,
    rating: data.rating,
    delta: data.delta,
    id: data.id,
    navigationType: data.navigationType,
    url: location.href,
    timestamp: Date.now(),
    // デバイス情報
    connection: navigator.connection?.effectiveType,
    deviceMemory: navigator.deviceMemory,
    hardwareConcurrency: navigator.hardwareConcurrency,
  });

  // Beacon API で確実に送信
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/api/web-vitals', payload);
  } else {
    fetch('/api/web-vitals', {
      method: 'POST',
      body: payload,
      keepalive: true,
    });
  }
}
```

### 6.3 手動での Core Web Vitals 計測

web-vitals ライブラリを使わず、PerformanceObserver で直接計測する方法を理解しておくことも重要である。ライブラリの内部動作を把握することで、トラブルシューティングが容易になる。

```javascript
// ==================================================
// PerformanceObserver による手動計測
// ==================================================

// --- LCP の手動計測 ---
// LCP は複数回報告される（最後の値が最終的な LCP）
let lcpValue = 0;
let lcpElement = null;

new PerformanceObserver((list) => {
  const entries = list.getEntries();
  const lastEntry = entries[entries.length - 1];

  lcpValue = lastEntry.startTime;
  lcpElement = lastEntry.element;  // LCP の対象 DOM 要素

  console.log('LCP 候補:', {
    value: lcpValue.toFixed(0) + 'ms',
    element: lcpElement?.tagName,
    url: lastEntry.url,           // 画像の場合の URL
    size: lastEntry.size,         // 要素の面積（ピクセル）
    loadTime: lastEntry.loadTime, // リソースの読み込み完了時刻
    renderTime: lastEntry.renderTime, // レンダリング時刻
  });
}).observe({ type: 'largest-contentful-paint', buffered: true });


// --- CLS の手動計測 ---
// CLS は Session Window 方式で計算する
let clsValue = 0;
let clsEntries = [];
let sessionValue = 0;
let sessionEntries = [];
let previousSessionEnd = 0;

new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    // ユーザー入力に起因するシフトは除外
    if (entry.hadRecentInput) continue;

    // Session Window のルール:
    // - 1秒以上のギャップがあれば新しいセッション
    // - セッションの長さは最大5秒
    if (
      sessionEntries.length > 0 &&
      (entry.startTime - previousSessionEnd > 1000 ||
       entry.startTime - sessionEntries[0].startTime > 5000)
    ) {
      // 新しいセッションの開始
      if (sessionValue > clsValue) {
        clsValue = sessionValue;
        clsEntries = [...sessionEntries];
      }
      sessionValue = 0;
      sessionEntries = [];
    }

    sessionValue += entry.value;
    sessionEntries.push(entry);
    previousSessionEnd = entry.startTime + entry.duration;
  }

  // 現在のセッションが最大の場合も更新
  if (sessionValue > clsValue) {
    clsValue = sessionValue;
    clsEntries = [...sessionEntries];
  }

  console.log('CLS:', clsValue.toFixed(4));
}).observe({ type: 'layout-shift', buffered: true });


// --- INP の手動計測 ---
// INP は全インタラクションの中で最も遅い応答を基準とする
// （98パーセンタイル値を使用）
const interactions = [];

new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    // interactionId を持つエントリのみがインタラクション
    if (!entry.interactionId) continue;

    const existing = interactions.find(
      i => i.interactionId === entry.interactionId
    );

    if (existing) {
      // 同一インタラクションの複数イベントは最大値を採用
      existing.duration = Math.max(existing.duration, entry.duration);
    } else {
      interactions.push({
        interactionId: entry.interactionId,
        duration: entry.duration,
        name: entry.name,
        startTime: entry.startTime,
        processingStart: entry.processingStart,
        processingEnd: entry.processingEnd,
        target: entry.target,
      });
    }
  }

  // INP の計算（98パーセンタイル）
  if (interactions.length > 0) {
    const sorted = [...interactions].sort(
      (a, b) => b.duration - a.duration
    );
    // 50件以上のインタラクションがある場合は98パーセンタイル
    const index = Math.min(
      sorted.length - 1,
      Math.floor(sorted.length / 50)
    );
    const inp = sorted[index].duration;
    console.log('INP:', inp, 'ms');
  }
}).observe({ type: 'event', buffered: true, durationThreshold: 16 });
```

### 6.4 INP の内部構造と最適化ポイント

INP は、ユーザーのインタラクション（クリック、タップ、キー入力）から次の描画更新までの全時間を計測する。この時間は3つのフェーズに分解できる。

```
INP の内訳 (Interaction to Next Paint)

ユーザー操作          描画更新
    |                    |
    v                    v
    +----+----+----+----+
    | ID | PT | PD | ?? |
    +----+----+----+----+

    ID = Input Delay（入力遅延）
         メインスレッドがビジーで、イベントハンドラの実行開始が遅れる時間
         原因: Long Task、大量の JavaScript 実行

    PT = Processing Time（処理時間）
         イベントハンドラの実行時間
         原因: 重い計算、同期的な DOM 操作

    PD = Presentation Delay（描画遅延）
         イベントハンドラ完了後、実際に画面が更新されるまでの時間
         原因: スタイル再計算、レイアウト、ペイント、コンポジット

    ┌──────────────────────────────────────────────────────────┐
    │  INP = ID + PT + PD                                     │
    │                                                         │
    │  改善の優先順位:                                        │
    │  1. Input Delay   -> yield to main thread               │
    │  2. Processing    -> 処理の分割・遅延・最適化           │
    │  3. Presentation  -> DOM 操作の最小化                    │
    └──────────────────────────────────────────────────────────┘
```

```javascript
// INP の各フェーズを分解して分析する
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.interactionId) continue;

    const inputDelay = entry.processingStart - entry.startTime;
    const processingTime = entry.processingEnd - entry.processingStart;
    const presentationDelay = entry.startTime + entry.duration
      - entry.processingEnd;

    if (entry.duration > 200) {
      console.warn(`[INP 警告] ${entry.name} on ${entry.target?.tagName}`, {
        total: `${entry.duration}ms`,
        inputDelay: `${inputDelay.toFixed(0)}ms`,
        processingTime: `${processingTime.toFixed(0)}ms`,
        presentationDelay: `${presentationDelay.toFixed(0)}ms`,
        bottleneck: inputDelay > processingTime
          ? (inputDelay > presentationDelay ? 'Input Delay' : 'Presentation')
          : (processingTime > presentationDelay ? 'Processing' : 'Presentation'),
      });
    }
  }
}).observe({ type: 'event', buffered: true, durationThreshold: 16 });
```

---

## 7. Lighthouse とパフォーマンス監査

### 7.1 Lighthouse のスコアリングモデル

Lighthouse はパフォーマンスを100点満点で評価する。スコアは複数のメトリクスの加重平均で算出され、各メトリクスは対数正規分布に基づくスコアリングカーブに当てはめられる。

| メトリクス                     | 重み  | Good 閾値   | Poor 閾値   |
|-------------------------------|-------|-------------|-------------|
| FCP (First Contentful Paint)  | 10%   | 1.8s        | 3.0s        |
| SI (Speed Index)              | 10%   | 3.4s        | 5.8s        |
| LCP (Largest Contentful Paint)| 25%   | 2.5s        | 4.0s        |
| TBT (Total Blocking Time)    | 30%   | 200ms       | 600ms       |
| CLS (Cumulative Layout Shift) | 25%   | 0.1         | 0.25        |

※ TBT は INP のラボ代替指標として使用される。INP はフィールド（実ユーザー）データでのみ計測可能であるため、Lighthouse では TBT がその代替を担う。

### 7.2 Lighthouse CI の自動化

```javascript
// lighthouserc.js - Lighthouse CI 設定ファイル
module.exports = {
  ci: {
    collect: {
      // 計測対象の URL 一覧
      url: [
        'http://localhost:3000/',
        'http://localhost:3000/products',
        'http://localhost:3000/checkout',
      ],
      // 計測回数（中央値を採用するため奇数が推奨）
      numberOfRuns: 5,
      // Chrome の起動オプション
      settings: {
        chromeFlags: '--no-sandbox --headless',
        // スロットリング設定（モバイル 4G 相当）
        throttling: {
          cpuSlowdownMultiplier: 4,
          downloadThroughputKbps: 1600,
          uploadThroughputKbps: 750,
          rttMs: 150,
        },
        // フォームファクター
        formFactor: 'mobile',
        screenEmulation: {
          mobile: true,
          width: 412,
          height: 823,
          deviceScaleFactor: 1.75,
        },
      },
    },
    assert: {
      // パフォーマンスバジェット
      assertions: {
        'categories:performance': ['error', { minScore: 0.9 }],
        'categories:accessibility': ['warn', { minScore: 0.9 }],
        'categories:best-practices': ['warn', { minScore: 0.9 }],
        // 個別メトリクスの閾値
        'first-contentful-paint': ['error', { maxNumericValue: 1800 }],
        'largest-contentful-paint': ['error', { maxNumericValue: 2500 }],
        'cumulative-layout-shift': ['error', { maxNumericValue: 0.1 }],
        'total-blocking-time': ['error', { maxNumericValue: 200 }],
        'speed-index': ['warn', { maxNumericValue: 3400 }],
        // リソースサイズの制限
        'resource-summary:script:size': [
          'error', { maxNumericValue: 300 * 1024 },  // 300KB
        ],
        'resource-summary:total:size': [
          'warn', { maxNumericValue: 1500 * 1024 },   // 1.5MB
        ],
      },
    },
    upload: {
      // Lighthouse CI Server にアップロード
      target: 'lhci',
      serverBaseUrl: 'https://lhci.example.com',
      token: process.env.LHCI_TOKEN,
    },
  },
};
```

### 7.3 GitHub Actions での Lighthouse CI 統合

```yaml
# .github/workflows/lighthouse.yml
name: Lighthouse CI

on:
  pull_request:
    branches: [main]

jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: npm ci

      - name: Build application
        run: npm run build

      - name: Start server
        run: npm run preview &
        env:
          PORT: 3000

      - name: Wait for server
        run: npx wait-on http://localhost:3000 --timeout 30000

      - name: Run Lighthouse CI
        run: |
          npm install -g @lhci/cli
          lhci autorun
        env:
          LHCI_TOKEN: ${{ secrets.LHCI_TOKEN }}
          LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}

      - name: Upload Lighthouse results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: lighthouse-results
          path: .lighthouseci/
```

### 7.4 Lighthouse のプログラマティック実行

```javascript
// Node.js で Lighthouse をプログラマティックに実行する
import lighthouse from 'lighthouse';
import * as chromeLauncher from 'chrome-launcher';

async function runLighthouse(url, options = {}) {
  // Chrome を起動
  const chrome = await chromeLauncher.launch({
    chromeFlags: ['--headless', '--no-sandbox'],
  });

  const defaultOptions = {
    logLevel: 'info',
    output: 'json',
    port: chrome.port,
    onlyCategories: ['performance'],
    // カスタムスロットリング設定
    throttling: {
      cpuSlowdownMultiplier: 4,
      downloadThroughputKbps: 1600,
      uploadThroughputKbps: 750,
      rttMs: 150,
    },
  };

  const mergedOptions = { ...defaultOptions, ...options };

  try {
    const result = await lighthouse(url, mergedOptions);

    // スコアと各メトリクスの取得
    const { lhr } = result;
    const perfScore = lhr.categories.performance.score * 100;

    const metrics = {
      score: perfScore,
      fcp: lhr.audits['first-contentful-paint'].numericValue,
      lcp: lhr.audits['largest-contentful-paint'].numericValue,
      tbt: lhr.audits['total-blocking-time'].numericValue,
      cls: lhr.audits['cumulative-layout-shift'].numericValue,
      si: lhr.audits['speed-index'].numericValue,
      tti: lhr.audits['interactive'].numericValue,
    };

    console.log(`Performance Score: ${perfScore}/100`);
    console.table(metrics);

    // 改善提案の取得
    const opportunities = Object.values(lhr.audits)
      .filter(audit => audit.details?.type === 'opportunity')
      .filter(audit => audit.details?.overallSavingsMs > 0)
      .sort((a, b) =>
        b.details.overallSavingsMs - a.details.overallSavingsMs
      )
      .map(audit => ({
        title: audit.title,
        savings: `${audit.details.overallSavingsMs.toFixed(0)}ms`,
        description: audit.description,
      }));

    console.log('改善提案:');
    opportunities.forEach((opp, i) => {
      console.log(`  ${i + 1}. ${opp.title} (${opp.savings} 削減可能)`);
    });

    return { metrics, opportunities, fullReport: lhr };
  } finally {
    await chrome.kill();
  }
}

// 使用例
const result = await runLighthouse('https://example.com');
```

---

## 8. パフォーマンスデータの送信と分析基盤

### 8.1 Beacon API と fetch keepalive

パフォーマンスデータの送信で最も重要なのは、ページ離脱時にもデータが失われないことである。Beacon API と `fetch` の `keepalive` オプションは、この要件を満たすために設計されている。

```javascript
// ==================================================
// パフォーマンスデータ送信の実装パターン
// ==================================================

class PerformanceReporter {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.buffer = [];
    this.flushInterval = 10000; // 10秒ごとにバッファをフラッシュ
    this.maxBufferSize = 50;    // バッファの最大件数

    this._setupAutoFlush();
    this._setupUnloadHandler();
  }

  // メトリクスをバッファに追加
  record(metric) {
    this.buffer.push({
      ...metric,
      timestamp: Date.now(),
      url: location.href,
      userAgent: navigator.userAgent,
      connection: navigator.connection?.effectiveType || 'unknown',
      deviceMemory: navigator.deviceMemory || 'unknown',
    });

    // バッファが上限に達したら即時フラッシュ
    if (this.buffer.length >= this.maxBufferSize) {
      this.flush();
    }
  }

  // バッファの内容を送信
  flush() {
    if (this.buffer.length === 0) return;

    const payload = JSON.stringify(this.buffer);
    this.buffer = [];

    // Beacon API を優先使用
    if (navigator.sendBeacon) {
      const blob = new Blob([payload], { type: 'application/json' });
      const success = navigator.sendBeacon(this.endpoint, blob);

      if (!success) {
        // Beacon API が失敗した場合のフォールバック
        this._fetchFallback(payload);
      }
      return;
    }

    this._fetchFallback(payload);
  }

  _fetchFallback(payload) {
    fetch(this.endpoint, {
      method: 'POST',
      body: payload,
      headers: { 'Content-Type': 'application/json' },
      keepalive: true,  // ページ離脱後も送信を継続
    }).catch(err => {
      console.warn('パフォーマンスデータの送信に失敗:', err);
    });
  }

  _setupAutoFlush() {
    setInterval(() => this.flush(), this.flushInterval);
  }

  _setupUnloadHandler() {
    // visibilitychange は unload/beforeunload より確実
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this.flush();
      }
    });
  }
}

// 使用例
const reporter = new PerformanceReporter('/api/performance');

// web-vitals と統合
import { onLCP, onINP, onCLS } from 'web-vitals';

onLCP(metric => reporter.record({ name: 'LCP', value: metric.value, rating: metric.rating }));
onINP(metric => reporter.record({ name: 'INP', value: metric.value, rating: metric.rating }));
onCLS(metric => reporter.record({ name: 'CLS', value: metric.value, rating: metric.rating }));
```

### 8.2 RUM（Real User Monitoring）の構築

RUM は、実際のユーザー環境でのパフォーマンスデータを収集・分析する仕組みである。合成モニタリング（Lighthouse 等）では再現できない、多様なデバイス・ネットワーク環境のパフォーマンスを把握できる。

```javascript
// ==================================================
// RUM データ収集の包括的な実装
// ==================================================
class RUMCollector {
  constructor(config) {
    this.config = {
      endpoint: config.endpoint,
      sampleRate: config.sampleRate || 1.0,  // 1.0 = 100%
      appVersion: config.appVersion || 'unknown',
      environment: config.environment || 'production',
    };

    // サンプリング判定
    this.shouldCollect = Math.random() < this.config.sampleRate;

    if (this.shouldCollect) {
      this._initCollectors();
    }
  }

  _initCollectors() {
    this._collectNavigationTiming();
    this._collectWebVitals();
    this._collectResourceTiming();
    this._collectErrors();
    this._collectLongTasks();
  }

  _collectNavigationTiming() {
    // ページ読み込み完了後に計測
    window.addEventListener('load', () => {
      setTimeout(() => {
        const nav = performance.getEntriesByType('navigation')[0];
        if (!nav) return;

        this._send('navigation', {
          dns: nav.domainLookupEnd - nav.domainLookupStart,
          tcp: nav.connectEnd - nav.connectStart,
          ttfb: nav.responseStart - nav.requestStart,
          download: nav.responseEnd - nav.responseStart,
          domProcessing: nav.domContentLoadedEventEnd - nav.responseEnd,
          loadComplete: nav.loadEventEnd - nav.startTime,
          transferSize: nav.transferSize,
          protocol: nav.nextHopProtocol,
          type: nav.type,
        });
      }, 0);
    });
  }

  _collectWebVitals() {
    // 動的インポートで web-vitals を読み込み
    import('web-vitals').then(({ onLCP, onINP, onCLS, onFCP, onTTFB }) => {
      onLCP(m => this._send('vital', { name: 'LCP', value: m.value, rating: m.rating }));
      onINP(m => this._send('vital', { name: 'INP', value: m.value, rating: m.rating }));
      onCLS(m => this._send('vital', { name: 'CLS', value: m.value, rating: m.rating }));
      onFCP(m => this._send('vital', { name: 'FCP', value: m.value, rating: m.rating }));
      onTTFB(m => this._send('vital', { name: 'TTFB', value: m.value, rating: m.rating }));
    });
  }

  _collectResourceTiming() {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        // 遅いリソースのみ報告（閾値: 2秒）
        if (entry.duration > 2000) {
          this._send('slow-resource', {
            url: entry.name,
            type: entry.initiatorType,
            duration: entry.duration,
            size: entry.transferSize,
            cached: entry.transferSize === 0 && entry.decodedBodySize > 0,
          });
        }
      }
    }).observe({ type: 'resource', buffered: true });
  }

  _collectErrors() {
    window.addEventListener('error', (event) => {
      this._send('error', {
        message: event.message,
        filename: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        type: 'uncaught',
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this._send('error', {
        message: event.reason?.message || String(event.reason),
        type: 'unhandled-rejection',
      });
    });
  }

  _collectLongTasks() {
    if (!PerformanceObserver.supportedEntryTypes.includes('longtask')) {
      return;
    }

    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration > 100) {
          this._send('long-task', {
            duration: entry.duration,
            startTime: entry.startTime,
          });
        }
      }
    }).observe({ type: 'longtask' });
  }

  _send(type, data) {
    const payload = {
      type,
      data,
      context: {
        url: location.href,
        referrer: document.referrer,
        appVersion: this.config.appVersion,
        environment: this.config.environment,
        timestamp: Date.now(),
        sessionId: this._getSessionId(),
        viewport: {
          width: window.innerWidth,
          height: window.innerHeight,
        },
        connection: {
          effectiveType: navigator.connection?.effectiveType,
          downlink: navigator.connection?.downlink,
          rtt: navigator.connection?.rtt,
          saveData: navigator.connection?.saveData,
        },
      },
    };

    const body = JSON.stringify(payload);

    if (navigator.sendBeacon) {
      navigator.sendBeacon(this.config.endpoint, body);
    } else {
      fetch(this.config.endpoint, {
        method: 'POST',
        body,
        keepalive: true,
      }).catch(() => {});
    }
  }

  _getSessionId() {
    let sessionId = sessionStorage.getItem('rum-session-id');
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      sessionStorage.setItem('rum-session-id', sessionId);
    }
    return sessionId;
  }
}

// 初期化
const rum = new RUMCollector({
  endpoint: '/api/rum',
  sampleRate: 0.1,  // 10% のユーザーからデータ収集
  appVersion: '2.3.1',
  environment: 'production',
});
```

---

## 9. パフォーマンスバジェットと CI 統合

### 9.1 パフォーマンスバジェットの設計

パフォーマンスバジェットとは、Web アプリケーションが達成すべきパフォーマンス目標を定量的に定義したものである。チーム全体で共有し、CI/CD パイプラインで自動検証することで、パフォーマンスの退行を防ぐ。

```
┌───────────────────────────────────────────────────────────────┐
│            パフォーマンスバジェットの設計フロー               │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  1. 競合分析                                                  │
│     ├── 競合サイトの Core Web Vitals を調査                  │
│     ├── CrUX（Chrome UX Report）データを参照                 │
│     └── 業界平均との比較                                     │
│                                                               │
│  2. 目標設定                                                  │
│     ├── Core Web Vitals の閾値                               │
│     ├── リソースサイズの上限                                  │
│     ├── リクエスト数の上限                                    │
│     └── Time to Interactive の目標値                          │
│                                                               │
│  3. 自動検証                                                  │
│     ├── bundlesize / size-limit によるバンドル監視            │
│     ├── Lighthouse CI によるスコア監視                        │
│     └── PR コメントでの差分レポート                           │
│                                                               │
│  4. 継続的改善                                                │
│     ├── 週次のパフォーマンスレビュー                          │
│     ├── バジェット超過時のアラート                            │
│     └── 改善施策の優先順位付け                                │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 9.2 size-limit によるバンドルサイズ監視

```javascript
// package.json に size-limit の設定を追加
// package.json (抜粋)
{
  "size-limit": [
    {
      "name": "メインバンドル",
      "path": "dist/assets/index-*.js",
      "limit": "150 kB",
      "gzip": true
    },
    {
      "name": "CSS",
      "path": "dist/assets/index-*.css",
      "limit": "30 kB",
      "gzip": true
    },
    {
      "name": "ベンダーバンドル",
      "path": "dist/assets/vendor-*.js",
      "limit": "200 kB",
      "gzip": true
    },
    {
      "name": "初期読み込み合計",
      "path": [
        "dist/assets/index-*.js",
        "dist/assets/index-*.css",
        "dist/assets/vendor-*.js"
      ],
      "limit": "350 kB",
      "gzip": true
    }
  ],
  "scripts": {
    "size": "size-limit",
    "size:check": "size-limit --json"
  }
}
```

### 9.3 カスタムパフォーマンスバジェットチェッカー

```javascript
// ==================================================
// パフォーマンスバジェットの定義と検証
// ==================================================

const PERFORMANCE_BUDGETS = {
  // Core Web Vitals バジェット
  vitals: {
    LCP: { good: 2500, poor: 4000, unit: 'ms' },
    INP: { good: 200, poor: 500, unit: 'ms' },
    CLS: { good: 0.1, poor: 0.25, unit: '' },
    FCP: { good: 1800, poor: 3000, unit: 'ms' },
    TTFB: { good: 800, poor: 1800, unit: 'ms' },
  },

  // リソースバジェット
  resources: {
    totalTransferSize: 1500 * 1024,   // 1.5MB
    totalRequests: 80,
    scriptSize: 300 * 1024,           // 300KB
    imageSize: 500 * 1024,            // 500KB
    fontSize: 100 * 1024,             // 100KB
    thirdPartySize: 200 * 1024,       // 200KB
  },

  // タイミングバジェット
  timing: {
    domContentLoaded: 3000,  // ms
    loadComplete: 5000,      // ms
    domInteractive: 2000,    // ms
  },
};

function checkBudgets(collectedData) {
  const violations = [];

  // Core Web Vitals のチェック
  for (const [name, budget] of Object.entries(PERFORMANCE_BUDGETS.vitals)) {
    const value = collectedData.vitals?.[name];
    if (value === undefined) continue;

    if (value > budget.poor) {
      violations.push({
        severity: 'error',
        metric: name,
        value: `${value}${budget.unit}`,
        budget: `${budget.poor}${budget.unit}`,
        message: `${name} が不良閾値（${budget.poor}${budget.unit}）を超過`,
      });
    } else if (value > budget.good) {
      violations.push({
        severity: 'warning',
        metric: name,
        value: `${value}${budget.unit}`,
        budget: `${budget.good}${budget.unit}`,
        message: `${name} が良好閾値（${budget.good}${budget.unit}）を超過`,
      });
    }
  }

  // リソースバジェットのチェック
  const resources = performance.getEntriesByType('resource');
  const totalSize = resources.reduce((sum, r) => sum + r.transferSize, 0);
  const totalRequests = resources.length;

  if (totalSize > PERFORMANCE_BUDGETS.resources.totalTransferSize) {
    violations.push({
      severity: 'error',
      metric: 'Total Transfer Size',
      value: `${(totalSize / 1024).toFixed(0)}KB`,
      budget: `${(PERFORMANCE_BUDGETS.resources.totalTransferSize / 1024).toFixed(0)}KB`,
      message: '合計転送サイズがバジェットを超過',
    });
  }

  if (totalRequests > PERFORMANCE_BUDGETS.resources.totalRequests) {
    violations.push({
      severity: 'warning',
      metric: 'Total Requests',
      value: totalRequests,
      budget: PERFORMANCE_BUDGETS.resources.totalRequests,
      message: 'リクエスト数がバジェットを超過',
    });
  }

  return violations;
}
```

---

## 10. アンチパターンと回避策

### 10.1 アンチパターン1: performance.getEntries() のポーリング

**問題**: `setInterval` で定期的に `performance.getEntries()` を呼び出してパフォーマンスデータを収集するパターン。これはメインスレッドに不必要な負荷をかけ、エントリの重複処理やタイミングの取りこぼしを引き起こす。

```javascript
// ============================================================
// アンチパターン: ポーリングによるエントリ収集
// ============================================================

// --- 悪い例 ---
// setInterval でポーリングする
setInterval(() => {
  const entries = performance.getEntries();
  entries.forEach(entry => {
    // 問題1: 毎回全エントリを取得するため、既に処理済みのエントリも再処理される
    // 問題2: getEntries() の呼び出しコストがエントリ数に比例して増大
    // 問題3: ポーリング間隔の間に発生したエントリを見逃す可能性がある
    // 問題4: バッファが一杯になると古いエントリが消えるが、検知できない
    processEntry(entry);
  });
}, 5000);


// --- 良い例 ---
// PerformanceObserver を使用する
const observer = new PerformanceObserver((list) => {
  // 新しいエントリのみがコールバックに渡される（重複なし）
  for (const entry of list.getEntries()) {
    processEntry(entry);
  }
});

observer.observe({
  type: 'resource',
  buffered: true,  // 過去のエントリも初回に取得
});

// 利点:
// - イベント駆動で効率的（ポーリング不要）
// - 新しいエントリのみが通知される（重複処理なし）
// - buffered: true で過去のエントリも取りこぼさない
// - バッファオーバーフローの検知が可能（droppedEntriesCount）
```

### 10.2 アンチパターン2: Core Web Vitals の誤った解釈

**問題**: ラボデータ（Lighthouse）とフィールドデータ（RUM/CrUX）の違いを理解せず、Lighthouse のスコアだけでパフォーマンスを判断するパターン。

```javascript
// ============================================================
// アンチパターン: Lighthouse スコアのみに依存する
// ============================================================

// --- 悪い例 ---
// Lighthouse で100点を目指して最適化し、それで完了と考える
//
// 問題1: Lighthouse は固定環境（特定のスロットリング設定）での計測であり、
//         実ユーザーの多様な環境を反映しない
// 問題2: TBT はラボ指標であり、フィールドの INP とは異なる
// 問題3: Lighthouse は初回読み込みのみ計測し、SPA のページ遷移は計測しない
// 問題4: サーバーのレスポンス時間やネットワーク品質の変動を反映しない

// --- 良い例 ---
// ラボデータとフィールドデータの両方を活用する

// 1. Lighthouse（ラボ）: 開発中の退行検知に使用
//    - CI で自動実行し、スコアの低下を検知
//    - 改善提案（Opportunities）を開発タスクに変換

// 2. CrUX（フィールド）: 実ユーザーのパフォーマンスを把握
async function fetchCrUXData(origin) {
  const response = await fetch(
    `https://chromeuxreport.googleapis.com/v1/records:queryRecord?key=${API_KEY}`,
    {
      method: 'POST',
      body: JSON.stringify({
        origin: origin,
        metrics: [
          'largest_contentful_paint',
          'interaction_to_next_paint',
          'cumulative_layout_shift',
        ],
      }),
    }
  );

  const data = await response.json();
  return data.record?.metrics;
}

// 3. 自前 RUM: 細粒度のフィールドデータを収集
//    - ページ別・ルート別のメトリクス
//    - ユーザーセグメント別の分析
//    - パフォーマンスとビジネス指標の相関分析
```

### 10.3 アンチパターン3: パフォーマンスバッファの枯渇

**問題**: `performance.setResourceTimingBufferSize()` を考慮せず、大量のリソースを読み込むアプリケーションでエントリが消失するパターン。

```javascript
// ============================================================
// アンチパターン: バッファサイズの未管理
// ============================================================

// --- 悪い例 ---
// デフォルトのバッファサイズ（通常250）で運用し、
// SPA で多数のリソースを読み込むとエントリが消失する

// --- 良い例 ---
// バッファサイズを管理する
performance.setResourceTimingBufferSize(500);

// バッファフルイベントを監視
performance.addEventListener('resourcetimingbufferfull', () => {
  // 現在のエントリを退避
  const entries = performance.getEntriesByType('resource');
  archiveResourceEntries(entries);

  // バッファをクリアして新しいエントリを受け入れる
  performance.clearResourceTimings();

  console.warn(
    `Resource Timing バッファが一杯になりました。` +
    `${entries.length}件のエントリをアーカイブしました。`
  );
});

// PerformanceObserver の droppedEntriesCount も活用
new PerformanceObserver((list, observer) => {
  const entries = list.getEntries();

  // ドロップされたエントリ数の確認
  if (observer.droppedEntriesCount && observer.droppedEntriesCount > 0) {
    console.warn(
      `${observer.droppedEntriesCount}件のエントリがドロップされました`
    );
  }

  entries.forEach(processEntry);
}).observe({ type: 'resource', buffered: true });
```

---

## 11. エッジケース分析

### 11.1 エッジケース1: bfcache（Back/Forward Cache）とパフォーマンス計測

bfcache は、ブラウザの「戻る」「進む」操作時にページの状態をメモリ上にそのまま保持する仕組みである。bfcache から復元されたページでは、通常のページ読み込みイベント（`load`、`DOMContentLoaded`）が発火しないため、パフォーマンス計測に特別な配慮が必要になる。

```javascript
// ==================================================
// bfcache 対応のパフォーマンス計測
// ==================================================

// bfcache からの復元を検知
window.addEventListener('pageshow', (event) => {
  if (event.persisted) {
    // bfcache から復元された
    console.log('bfcache から復元されました');

    // Navigation Timing は新しいエントリが記録されない
    // -> 復元時刻のみを手動で記録する
    performance.mark('bfcache-restore', {
      detail: { timestamp: event.timeStamp },
    });

    // web-vitals ライブラリは bfcache 復元後の計測を
    // 自動的に処理するが、カスタム計測は再初期化が必要
    reinitializeCustomMetrics();
  }
});

// bfcache 対応のページ離脱処理
// 注意: 'unload' イベントハンドラは bfcache を無効化する
// 代わりに 'pagehide' または 'visibilitychange' を使用する

// --- 悪い例（bfcache を阻害）---
window.addEventListener('unload', () => {
  sendFinalMetrics();  // この処理のせいで bfcache が使えなくなる
});

// --- 良い例（bfcache 互換）---
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    sendFinalMetrics();
  }
});

// bfcache の適格性を確認する（Chrome DevTools API）
// DevTools > Application > Back/forward cache で確認可能

// プログラマティックに bfcache ブロッカーを検出
function checkBfcacheBlockers() {
  const issues = [];

  // unload イベントリスナーの検出
  // （直接検出する API はないが、自分で管理する）

  // Cache-Control: no-store は bfcache を阻害する
  // -> Cache-Control: no-cache を使用する

  // WebSocket 接続中は bfcache が効かない
  // -> pagehide でクローズし、pageshow で再接続する

  return issues;
}
```

### 11.2 エッジケース2: Service Worker 経由のリソース計測

Service Worker がリソースのリクエストをインターセプトする場合、Resource Timing のタイミング値に Service Worker の処理時間が含まれる。これにより、ネットワーク時間と Service Worker 処理時間を区別して分析する必要がある。

```javascript
// ==================================================
// Service Worker 経由のリソース計測
// ==================================================

function analyzeServiceWorkerImpact() {
  const resources = performance.getEntriesByType('resource');

  resources.forEach(entry => {
    const swProcessing = entry.workerStart > 0
      ? entry.fetchStart - entry.workerStart
      : 0;

    const isFromSW = entry.workerStart > 0;
    const isFromCache = entry.transferSize === 0
      && entry.decodedBodySize > 0;

    if (isFromSW) {
      console.log(`[SW] ${new URL(entry.name).pathname}`, {
        // Service Worker の起動時間
        swStartup: entry.workerStart > 0
          ? `${(entry.fetchStart - entry.workerStart).toFixed(0)}ms`
          : 'N/A',
        // Service Worker 内の処理時間
        swProcessing: `${swProcessing.toFixed(0)}ms`,
        // ネットワークリクエスト時間（SW がネットワークにフォールバックした場合）
        networkTime: isFromCache
          ? '0ms (cached)'
          : `${(entry.responseEnd - entry.fetchStart).toFixed(0)}ms`,
        // 合計時間
        total: `${entry.duration.toFixed(0)}ms`,
        // キャッシュ状態の推定
        cacheStrategy: isFromCache ? 'cache-first' : 'network-first',
      });
    }
  });
}

// Service Worker のキャッシュ戦略別のパフォーマンス比較
function compareSwCacheStrategies() {
  const resources = performance.getEntriesByType('resource');

  const categories = {
    cacheFirst: [],    // transferSize === 0 && workerStart > 0
    networkFirst: [],  // transferSize > 0 && workerStart > 0
    noSw: [],          // workerStart === 0
  };

  resources.forEach(entry => {
    if (entry.workerStart === 0) {
      categories.noSw.push(entry);
    } else if (entry.transferSize === 0 && entry.decodedBodySize > 0) {
      categories.cacheFirst.push(entry);
    } else {
      categories.networkFirst.push(entry);
    }
  });

  const summarize = (entries) => ({
    count: entries.length,
    avgDuration: entries.length > 0
      ? `${(entries.reduce((s, e) => s + e.duration, 0) / entries.length).toFixed(0)}ms`
      : 'N/A',
    p95Duration: entries.length > 0
      ? `${entries.sort((a, b) => a.duration - b.duration)[Math.floor(entries.length * 0.95)]?.duration.toFixed(0)}ms`
      : 'N/A',
  });

  console.table({
    'Cache First (SW)': summarize(categories.cacheFirst),
    'Network First (SW)': summarize(categories.networkFirst),
    'No Service Worker': summarize(categories.noSw),
  });
}
```

### 11.3 エッジケース3: SPA（Single Page Application）でのルート遷移計測

SPA ではページ全体のリロードが発生しないため、Navigation Timing は初回読み込み時のみ有効である。クライアントサイドルーティングによるページ遷移は、User Timing を用いて手動で計測する必要がある。

```javascript
// ==================================================
// SPA ルート遷移のパフォーマンス計測
// ==================================================

class SPANavigationTracker {
  constructor() {
    this.navigations = [];
    this.currentNavigation = null;
    this._setupHistoryInterception();
  }

  _setupHistoryInterception() {
    // history.pushState / replaceState をインターセプト
    const originalPushState = history.pushState.bind(history);
    const originalReplaceState = history.replaceState.bind(history);

    history.pushState = (...args) => {
      this._onNavigationStart(args[2]);
      originalPushState(...args);
    };

    history.replaceState = (...args) => {
      originalReplaceState(...args);
    };

    // popstate（ブラウザの戻る/進む）
    window.addEventListener('popstate', () => {
      this._onNavigationStart(location.pathname);
    });
  }

  _onNavigationStart(toUrl) {
    const navId = `spa-nav-${Date.now()}`;
    const fromUrl = location.pathname;

    performance.mark(`${navId}-start`, {
      detail: { from: fromUrl, to: toUrl },
    });

    this.currentNavigation = {
      id: navId,
      from: fromUrl,
      to: toUrl,
      startTime: performance.now(),
    };

    // 次のフレーム描画を待って完了とする
    // (requestAnimationFrame 2回で描画完了を推定)
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        this._onNavigationEnd();
      });
    });
  }

  _onNavigationEnd() {
    if (!this.currentNavigation) return;

    const nav = this.currentNavigation;
    performance.mark(`${nav.id}-end`);
    performance.measure(nav.id, `${nav.id}-start`, `${nav.id}-end`);

    const entry = performance.getEntriesByName(nav.id).pop();
    const result = {
      ...nav,
      duration: entry.duration,
      timestamp: Date.now(),
    };

    this.navigations.push(result);
    this.currentNavigation = null;

    console.log(
      `[SPA Nav] ${result.from} -> ${result.to}: ` +
      `${result.duration.toFixed(0)}ms`
    );

    return result;
  }

  // 統計情報の取得
  getStats() {
    if (this.navigations.length === 0) return null;

    const durations = this.navigations.map(n => n.duration).sort((a, b) => a - b);
    return {
      count: durations.length,
      avg: `${(durations.reduce((s, d) => s + d, 0) / durations.length).toFixed(0)}ms`,
      median: `${durations[Math.floor(durations.length / 2)].toFixed(0)}ms`,
      p95: `${durations[Math.floor(durations.length * 0.95)].toFixed(0)}ms`,
      max: `${durations[durations.length - 1].toFixed(0)}ms`,
    };
  }
}

const spaTracker = new SPANavigationTracker();
```

---

## 12. 段階別演習

### 12.1 演習1: 初級 - ページ読み込みレポートの作成

**目標**: Navigation Timing API を使って、現在のページの読み込みパフォーマンスをコンソールに表形式で出力する。

**要件**:
1. DNS 解決、TCP 接続、TTFB、ダウンロード、DOM 処理の各時間を計測する
2. 結果を `console.table()` で見やすく表示する
3. TTFB が 800ms を超えている場合は警告を表示する

```javascript
// ==================================================
// 演習1: 解答例
// ==================================================

function generateLoadingReport() {
  const nav = performance.getEntriesByType('navigation')[0];
  if (!nav) {
    console.error('Navigation Timing データが取得できません');
    return;
  }

  const report = {
    'DNS 解決': { value: nav.domainLookupEnd - nav.domainLookupStart, unit: 'ms' },
    'TCP 接続': { value: nav.connectEnd - nav.connectStart, unit: 'ms' },
    'TLS ハンドシェイク': {
      value: nav.secureConnectionStart > 0
        ? nav.connectEnd - nav.secureConnectionStart : 0,
      unit: 'ms',
    },
    'TTFB': { value: nav.responseStart - nav.requestStart, unit: 'ms' },
    'ダウンロード': { value: nav.responseEnd - nav.responseStart, unit: 'ms' },
    'DOM 処理': { value: nav.domContentLoadedEventEnd - nav.responseEnd, unit: 'ms' },
    '合計読み込み時間': { value: nav.loadEventEnd - nav.startTime, unit: 'ms' },
    'プロトコル': { value: nav.nextHopProtocol, unit: '' },
    '転送サイズ': { value: (nav.transferSize / 1024).toFixed(1), unit: 'KB' },
  };

  // テーブル用にフォーマット
  const tableData = {};
  for (const [key, data] of Object.entries(report)) {
    tableData[key] = typeof data.value === 'number'
      ? `${data.value.toFixed(1)}${data.unit}`
      : `${data.value}${data.unit}`;
  }

  console.table(tableData);

  // TTFB の警告
  const ttfb = nav.responseStart - nav.requestStart;
  if (ttfb > 800) {
    console.warn(
      `TTFB が 800ms を超えています: ${ttfb.toFixed(0)}ms\n` +
      '改善案: サーバー処理の最適化、CDN の導入、キャッシュ戦略の見直し'
    );
  }

  return report;
}

// ページ読み込み完了後に実行
window.addEventListener('load', () => {
  setTimeout(generateLoadingReport, 0);
});
```

### 12.2 演習2: 中級 - リソース最適化ダッシュボード

**目標**: Resource Timing API を使って、リソースの読み込みパフォーマンスを分析するダッシュボードを構築する。

**要件**:
1. リソースタイプ別（script, css, img, fetch 等）に集計する
2. 各タイプの合計サイズ、平均読み込み時間、キャッシュヒット率を算出する
3. 1秒以上かかったリソースをワースト5として報告する
4. サードパーティリソースを検出して一覧化する

```javascript
// ==================================================
// 演習2: 解答例
// ==================================================

function buildResourceDashboard() {
  const resources = performance.getEntriesByType('resource');
  const currentOrigin = location.origin;

  // --- 1. タイプ別集計 ---
  const byType = {};
  resources.forEach(r => {
    const type = r.initiatorType || 'other';
    if (!byType[type]) {
      byType[type] = { count: 0, totalSize: 0, totalDuration: 0, cachedCount: 0 };
    }
    byType[type].count++;
    byType[type].totalSize += r.transferSize;
    byType[type].totalDuration += r.duration;
    if (r.transferSize === 0 && r.decodedBodySize > 0) {
      byType[type].cachedCount++;
    }
  });

  console.log('=== リソースタイプ別集計 ===');
  const typeTable = {};
  for (const [type, data] of Object.entries(byType)) {
    typeTable[type] = {
      件数: data.count,
      合計サイズ: `${(data.totalSize / 1024).toFixed(1)}KB`,
      平均時間: `${(data.totalDuration / data.count).toFixed(0)}ms`,
      キャッシュ率: `${((data.cachedCount / data.count) * 100).toFixed(0)}%`,
    };
  }
  console.table(typeTable);

  // --- 2. ワースト5 ---
  console.log('\n=== 読み込みが遅いリソース ワースト5 ===');
  const worst5 = resources
    .filter(r => r.duration > 1000)
    .sort((a, b) => b.duration - a.duration)
    .slice(0, 5);

  worst5.forEach((r, i) => {
    console.log(`  ${i + 1}. ${new URL(r.name).pathname}`);
    console.log(`     タイプ: ${r.initiatorType}`);
    console.log(`     時間: ${r.duration.toFixed(0)}ms`);
    console.log(`     サイズ: ${(r.transferSize / 1024).toFixed(1)}KB`);
  });

  // --- 3. サードパーティリソース ---
  console.log('\n=== サードパーティリソース ===');
  const thirdParty = resources.filter(r => {
    try { return new URL(r.name).origin !== currentOrigin; }
    catch { return false; }
  });

  const byDomain = {};
  thirdParty.forEach(r => {
    const domain = new URL(r.name).hostname;
    if (!byDomain[domain]) {
      byDomain[domain] = { count: 0, totalSize: 0, totalDuration: 0 };
    }
    byDomain[domain].count++;
    byDomain[domain].totalSize += r.transferSize;
    byDomain[domain].totalDuration += r.duration;
  });

  const domainTable = {};
  for (const [domain, data] of Object.entries(byDomain)) {
    domainTable[domain] = {
      リクエスト数: data.count,
      合計サイズ: `${(data.totalSize / 1024).toFixed(1)}KB`,
      合計時間: `${data.totalDuration.toFixed(0)}ms`,
    };
  }
  console.table(domainTable);

  return { byType: typeTable, worst5, thirdPartyDomains: domainTable };
}

// 実行
window.addEventListener('load', () => {
  setTimeout(buildResourceDashboard, 1000);
});
```

### 12.3 演習3: 上級 - 包括的パフォーマンスモニタリングシステム

**目標**: PerformanceObserver、User Timing、web-vitals を組み合わせた包括的なパフォーマンスモニタリングシステムを構築する。

**要件**:
1. Core Web Vitals（LCP、INP、CLS）をリアルタイムで計測する
2. Long Task を検出し、原因を分析する
3. カスタムメトリクス（API レスポンス時間、レンダリング時間）を計測する
4. 全データを統合してレポートを生成し、Beacon API で送信する
5. パフォーマンスバジェットとの比較結果を含める

```javascript
// ==================================================
// 演習3: 解答例（包括的モニタリングシステム）
// ==================================================

class PerformanceMonitor {
  constructor(config) {
    this.config = {
      endpoint: config.endpoint || '/api/perf',
      budgets: config.budgets || {},
      sampleRate: config.sampleRate || 1.0,
      debug: config.debug || false,
    };

    this.data = {
      vitals: {},
      longTasks: [],
      customMetrics: [],
      violations: [],
    };

    if (Math.random() > this.config.sampleRate) return;

    this._initVitals();
    this._initLongTaskMonitor();
    this._setupReporting();
  }

  // Core Web Vitals の計測
  _initVitals() {
    import('web-vitals').then(({ onLCP, onINP, onCLS, onFCP, onTTFB }) => {
      const recordVital = (metric) => {
        this.data.vitals[metric.name] = {
          value: metric.value,
          rating: metric.rating,
          delta: metric.delta,
        };

        // バジェットチェック
        const budget = this.config.budgets[metric.name];
        if (budget && metric.value > budget) {
          this.data.violations.push({
            metric: metric.name,
            value: metric.value,
            budget: budget,
            exceeded: metric.value - budget,
          });

          if (this.config.debug) {
            console.warn(
              `[Budget Violation] ${metric.name}: ` +
              `${metric.value} > ${budget} (超過: ${(metric.value - budget).toFixed(1)})`
            );
          }
        }
      };

      onLCP(recordVital);
      onINP(recordVital);
      onCLS(recordVital);
      onFCP(recordVital);
      onTTFB(recordVital);
    });
  }

  // Long Task の監視
  _initLongTaskMonitor() {
    if (!PerformanceObserver.supportedEntryTypes.includes('longtask')) return;

    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        this.data.longTasks.push({
          duration: entry.duration,
          startTime: entry.startTime,
          timestamp: Date.now(),
        });
      }
    }).observe({ type: 'longtask' });
  }

  // カスタムメトリクスの記録
  measure(name, fn) {
    const start = performance.now();
    const result = fn();

    if (result instanceof Promise) {
      return result.then(value => {
        this._recordCustomMetric(name, performance.now() - start);
        return value;
      }).catch(error => {
        this._recordCustomMetric(`${name}-error`, performance.now() - start);
        throw error;
      });
    }

    this._recordCustomMetric(name, performance.now() - start);
    return result;
  }

  _recordCustomMetric(name, duration) {
    this.data.customMetrics.push({
      name,
      duration,
      timestamp: Date.now(),
    });

    if (this.config.debug) {
      console.log(`[Custom Metric] ${name}: ${duration.toFixed(1)}ms`);
    }
  }

  // レポートの送信設定
  _setupReporting() {
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        this._sendReport();
      }
    });
  }

  _sendReport() {
    const report = {
      url: location.href,
      timestamp: Date.now(),
      vitals: this.data.vitals,
      longTasks: {
        count: this.data.longTasks.length,
        totalDuration: this.data.longTasks.reduce((s, t) => s + t.duration, 0),
        maxDuration: this.data.longTasks.length > 0
          ? Math.max(...this.data.longTasks.map(t => t.duration))
          : 0,
      },
      customMetrics: this.data.customMetrics,
      violations: this.data.violations,
      context: {
        connection: navigator.connection?.effectiveType,
        deviceMemory: navigator.deviceMemory,
        viewport: `${innerWidth}x${innerHeight}`,
      },
    };

    const body = JSON.stringify(report);

    if (navigator.sendBeacon) {
      navigator.sendBeacon(this.config.endpoint, body);
    } else {
      fetch(this.config.endpoint, {
        method: 'POST',
        body,
        keepalive: true,
      }).catch(() => {});
    }

    if (this.config.debug) {
      console.log('[Performance Report]', report);
    }
  }

  // 現在の状態を取得
  getReport() {
    return { ...this.data };
  }
}

// 初期化と使用例
const monitor = new PerformanceMonitor({
  endpoint: '/api/performance',
  sampleRate: 1.0,
  debug: true,
  budgets: {
    LCP: 2500,
    INP: 200,
    CLS: 0.1,
    FCP: 1800,
    TTFB: 800,
  },
});

// カスタムメトリクスの使用例
const data = await monitor.measure('fetch-products', () =>
  fetch('/api/products').then(r => r.json())
);
```

---

## 13. FAQ

### Q1: Lighthouse のスコアが DevTools と CLI で異なるのはなぜか？

**A**: Lighthouse のスコアは実行環境によって変動する。主な差異の原因は以下の通りである。

1. **スロットリング方式の違い**: DevTools は Simulated Throttling（シミュレーション）をデフォルトで使用するが、CLI では Applied Throttling（実際のネットワーク/CPU 制限）も選択できる。シミュレーション方式はネットワーク・CPU の制限をアルゴリズム的に推定するため、実環境とは異なる結果が出やすい。

2. **バックグラウンドプロセスの影響**: ローカル環境では他のタブやアプリケーションが CPU やメモリを消費しており、計測結果にノイズが入る。CI 環境でも同様に、並列実行されるジョブがリソースを競合する。

3. **ネットワーク条件の変動**: 同一のスロットリング設定でも、実際のネットワーク遅延やサーバー応答時間は変動する。

**推奨対策**:
- 複数回（3〜5回）実行して中央値を採用する
- CI 環境では一貫した設定を使用する
- スコアの絶対値よりも変化（差分）に注目する

### Q2: CLS が意図せず高くなる一般的な原因は何か？

**A**: CLS が高くなる典型的な原因と対策を以下に示す。

| 原因 | 詳細 | 対策 |
|------|------|------|
| 寸法未指定の画像・動画 | 読み込み完了後に要素サイズが確定し、周囲のコンテンツが押し出される | `width`/`height` 属性を必ず指定する。`aspect-ratio` CSS プロパティも有効 |
| Web フォントの FOIT/FOUT | フォント読み込み中にテキストが非表示（FOIT）またはフォールバックフォントで表示（FOUT）され、切り替え時にシフトが発生 | `font-display: swap` と `size-adjust` を使用。`<link rel="preload">` でフォントを先読み |
| 動的コンテンツの挿入 | 広告、バナー、Cookie 同意ダイアログなどが既存コンテンツの上に挿入される | 挿入スペースを事前に確保する。`min-height` を設定する |
| 遅延読み込みコンテンツ | API レスポンス待ちでスケルトン表示からコンテンツに切り替わる際にサイズが変わる | スケルトンのサイズをコンテンツと一致させる。`contain: layout` を使用 |

### Q3: INP と FID の違いは何か？なぜ FID は廃止されたのか？

**A**: FID（First Input Delay）は最初のインタラクションの入力遅延のみを計測する指標であった。一方、INP はページライフサイクル全体にわたる全てのインタラクションを対象とし、最も遅い応答（98パーセンタイル）を報告する。

FID が廃止された主な理由は以下の通りである。

1. **計測範囲の狭さ**: FID は「最初の」インタラクションのみを計測するため、ページ読み込み後のインタラクション品質を評価できなかった。SPA のような長寿命ページでは、ページ滞在中の操作感こそが重要である。

2. **入力遅延のみの計測**: FID はイベントハンドラの実行開始までの遅延（Input Delay）のみを計測し、処理時間（Processing Time）と描画遅延（Presentation Delay）を含まなかった。ハンドラ内の重い処理はFIDに反映されなかった。

3. **楽観的な評価**: 多くのサイトで FID が良好（< 100ms）であっても、ユーザー体感としては応答が遅いケースが多かった。INP は実態をより正確に反映する。

### Q4: パフォーマンス計測はモバイル端末でどの程度変わるか？

**A**: モバイル端末とデスクトップでは、パフォーマンス特性が大きく異なる。主な差異を以下に整理する。

| 要素 | デスクトップ | モバイル | 影響 |
|------|-------------|---------|------|
| CPU 性能 | 高速マルチコア | 低〜中速、熱による周波数低下 | JavaScript 実行時間が2〜5倍に増加 |
| メモリ | 8〜32GB | 2〜8GB | 大きなバンドルでメモリ不足の可能性 |
| ネットワーク | 有線 / Wi-Fi | 4G/5G（変動あり） | TTFB とダウンロード時間が不安定 |
| 画面サイズ | 大画面 | 小画面 | LCP の対象要素が変わる場合がある |
| タッチ入力 | マウス + キーボード | タッチ | INP の計測対象イベントが異なる |

Lighthouse のデフォルト設定では、CPU を4倍に遅延させ、ネットワークを 4G 相当にスロットリングすることで、モバイル環境をシミュレートしている。しかし、実際のモバイル端末にはバッテリー残量、熱管理、OS のバックグラウンド処理といった変動要因があり、シミュレーションでは再現できない。フィールドデータ（RUM）との併用が不可欠である。

### Q5: PerformanceObserver の observe で type と entryTypes の違いは何か？

**A**: `observe()` メソッドには2つの指定方法がある。

- **`type`（単数形）**: 1つの entryType を指定する。`buffered` オプションが使用可能。新しい API（`droppedEntriesCount` 等）にアクセスできる。Performance Timeline Level 2 で導入された推奨方式。

- **`entryTypes`（複数形）**: 複数の entryType を配列で指定できる。`buffered` オプションは使用不可。古い API 互換の方式。

```javascript
// 推奨: type（単数形）+ buffered
new PerformanceObserver(callback).observe({
  type: 'resource',
  buffered: true,
});

// 互換: entryTypes（複数形）
new PerformanceObserver(callback).observe({
  entryTypes: ['resource', 'mark', 'measure'],
  // buffered は使えない
});
```

複数の entryType を監視しつつ `buffered` も使いたい場合は、entryType ごとに別々の Observer を作成する。

### Q6: Core Web Vitals（LCP/FID/CLS）を改善する最も効果的な方法は？

**A**: 各指標に対する最も効果的な改善手法を以下にまとめる。

**LCP（Largest Contentful Paint）の改善**:
1. **画像の最適化**: WebP/AVIF 形式への変換、適切なサイズでの配信、`<img srcset>` による responsive images の実装
2. **クリティカルリソースの優先読み込み**: `<link rel="preload">` で LCP 要素（Hero 画像やメインコンテンツ）を先行ロード
3. **サーバー応答時間（TTFB）の短縮**: CDN の導入、サーバーサイドキャッシュ、データベースクエリの最適化
4. **レンダーブロッキングリソースの削減**: CSS のインライン化、JavaScript の defer/async 属性、未使用 CSS の削除

**INP（Interaction to Next Paint）の改善**:
1. **Long Task の分割**: `scheduler.yield()` または `setTimeout(fn, 0)` で処理を細かく分割し、メインスレッドを定期的に解放
2. **JavaScript バンドルサイズの削減**: Code splitting、Tree shaking、dynamic import による遅延読み込み
3. **イベントハンドラの最適化**: デバウンス/スロットリング、イベント委譲（Event Delegation）、passive リスナーの使用
4. **Web Worker の活用**: 重い計算処理をバックグラウンドスレッドに移譲

**CLS（Cumulative Layout Shift）の改善**:
1. **画像・動画の寸法指定**: `width`/`height` 属性または `aspect-ratio` CSS プロパティを必ず設定
2. **Web フォントの最適化**: `font-display: swap` と `size-adjust` の使用、フォントのプリロード
3. **動的コンテンツ用のスペース確保**: 広告やバナー用の `min-height` 設定、スケルトンスクリーンの寸法統一
4. **CSS `contain` プロパティの活用**: `contain: layout` でレイアウトの影響範囲を制限

### Q7: Performance API と Lighthouse の使い分けは？どちらを優先すべきか？

**A**: Performance API（RUM）と Lighthouse（ラボテスト）は相互補完的であり、両方を活用することが推奨される。

| 観点 | Performance API（RUM） | Lighthouse（ラボテスト） |
|------|----------------------|------------------------|
| データ収集源 | 実際のユーザー環境 | 開発者が制御する固定環境 |
| 結果の一貫性 | 低い（環境の多様性） | 高い（同一設定で再現可能） |
| 問題の発見 | 実環境での問題を検出 | 潜在的なボトルネックを検出 |
| デバッグ | 困難（環境再現が難しい） | 容易（トレースで詳細分析） |
| 改善提案 | なし | 具体的な改善策を提示 |
| コスト | バックエンド基盤が必要 | 無料（CI 統合も可能） |

**推奨の使い分け**:
1. **開発フェーズ**: Lighthouse を使って問題を早期発見し、改善提案に従って最適化を行う
2. **CI/CD パイプライン**: Lighthouse CI でパフォーマンスバジェットを設定し、リグレッションを自動検出
3. **本番環境**: Performance API で RUM データを収集し、実際のユーザー体験を監視
4. **問題調査**: Lighthouse のトレース機能で詳細なボトルネック分析を実施
5. **効果検証**: RUM データで改善の効果を定量的に測定

両者を組み合わせることで、「ラボでの理想」と「フィールドでの現実」のギャップを埋めることができる。

### Q8: リアルユーザーモニタリング（RUM）の導入方法と注意点は？

**A**: RUM の導入には以下のステップと注意点がある。

**導入ステップ**:

1. **計測対象の決定**:
   - Core Web Vitals（LCP/INP/CLS）は必須
   - Navigation Timing（TTFB, DOMContentLoaded, Load）
   - Resource Timing（重要リソースのみに絞る）
   - User Timing（アプリ固有の重要イベント）

2. **ライブラリの選定**:
   - Google の `web-vitals` ライブラリ（軽量・公式）
   - サードパーティ APM（New Relic, Datadog, Sentry など）
   - 自前実装（PerformanceObserver ベース）

3. **データ送信の実装**:
   - Beacon API または `fetch` with `keepalive: true` を使用
   - サンプリングレートを設定（全ユーザーの 10〜50% など）
   - バッチ送信で通信回数を削減

4. **バックエンド構築**:
   - エンドポイント実装（POST リクエストを受け取り、データベースに保存）
   - 集計処理（パーセンタイル計算、時系列データの生成）
   - ダッシュボード構築（グラフ化、アラート設定）

**注意点**:

| 項目 | 詳細 |
|------|------|
| プライバシー保護 | 個人識別情報を含めない。GDPR/CCPA 対応のため、事前に同意を取得 |
| パフォーマンスへの影響 | 計測処理自体がパフォーマンスに影響しないよう、非同期処理と軽量化を徹底 |
| サンプリング | 全ユーザーを計測する必要はない。10〜50% のサンプリングで十分な統計的有意性が得られる |
| ボットの除外 | ユーザーエージェント解析やキャプチャで、クローラーや自動化ツールを除外 |
| データ保持期間 | ストレージコストを考慮し、生データは 30〜90 日、集計データは長期保存 |
| アラート設定 | 指標が閾値を超えたらチームに通知（Slack, PagerDuty 等） |

**コード例（軽量な RUM 実装）**:

```javascript
import { onCLS, onINP, onLCP } from 'web-vitals';

function sendToAnalytics(metric) {
  // サンプリング（50%）
  if (Math.random() > 0.5) return;

  const body = JSON.stringify({
    name: metric.name,
    value: metric.value,
    rating: metric.rating,
    url: location.href,
    userAgent: navigator.userAgent,
    timestamp: Date.now(),
  });

  // Beacon API で送信
  navigator.sendBeacon('/api/metrics', body);
}

// Core Web Vitals を監視
onCLS(sendToAnalytics);
onINP(sendToAnalytics);
onLCP(sendToAnalytics);
```

---

### Q4: PerformanceObserver のコールバックがパフォーマンスに影響を与えることはありますか?

**A**: PerformanceObserver のコールバックはマイクロタスクとして実行されるため、コールバック内で重い処理（大量のログ出力、DOM操作、同期的なネットワーク送信など）を行うとメインスレッドをブロックし、計測対象のパフォーマンスに影響を与えます。コールバック内では最小限のデータ抽出とバッファリングのみを行い、データの送信は `requestIdleCallback` や `Beacon API`（`navigator.sendBeacon`）でアイドル時またはページ離脱時に実行するのが推奨です。

### Q5: 本番環境で全ユーザーのパフォーマンスデータを収集すべきですか?

**A**: 全ユーザーからデータを収集するとサーバー負荷とネットワークコストが増大するため、通常はサンプリングを行います。一般的には全ユーザーの1〜10%をサンプリングし、統計的に有意なデータ量を確保します。サンプリングレートは `Math.random() < 0.05`（5%）のように制御し、セッション単位で固定することで、同一ユーザーのページ遷移を一貫して追跡できます。重要なページ（ランディングページ、チェックアウト）ではサンプリングレートを高めに設定することも有効です。

### Q6: TTFB が遅い場合、フロントエンドで改善できることはありますか?

**A**: TTFB（Time to First Byte）はサーバー側の処理時間に大きく依存しますが、フロントエンド側でも改善余地があります。`<link rel="preconnect">` でCDNやAPIサーバーへの接続を事前に確立する、Service Worker でキャッシュヒット時に即座にレスポンスを返す（Stale-While-Revalidate 戦略）、Navigation Timing API で TTFB を継続的に計測しサーバーチームにデータを共有する、といった施策が有効です。

---

## 追加参考文献

- [Google Developers - Optimize Time to First Byte](https://web.dev/articles/optimize-ttfb) - TTFB最適化の包括的ガイド
- [W3C - Performance Timeline Level 2](https://www.w3.org/TR/performance-timeline/) - PerformanceObserver の公式仕様
- [Mozilla - PerformanceObserver](https://developer.mozilla.org/en-US/docs/Web/API/PerformanceObserver) - PerformanceObserver の実装ガイド

---

## 14. 比較表

### 14.1 パフォーマンス計測ツール比較

| 特性 | Performance API（RUM） | Lighthouse（ラボ） | WebPageTest（ラボ） | CrUX（フィールド） |
|------|----------------------|-------------------|--------------------|--------------------|
| データソース | 実ユーザー | シミュレーション | 実ネットワーク | 実ユーザー（Chrome） |
| 環境の多様性 | 高（全デバイス） | 低（固定設定） | 中（選択可能） | 高（Chrome のみ） |
| 計測タイミング | リアルタイム | オンデマンド/CI | オンデマンド | 28日間集計 |
| INP 対応 | 対応 | TBT で代替 | 対応（一部） | 対応 |
| カスタム指標 | 対応 | カスタム監査で対応 | カスタムスクリプト | 非対応 |
| コスト | 実装コスト | 無料 | 無料/有料プラン | 無料（API） |
| 結果の一貫性 | 低（環境依存） | 中（変動あり） | 中〜高 | 高（大量データ） |
| 改善提案 | なし | 詳細な提案あり | 詳細な提案あり | なし |
| SPA 対応 | 手動実装が必要 | 初回読み込みのみ | 初回読み込みのみ | 限定的 |

### 14.2 データ送信方式の比較

| 特性 | Beacon API | fetch (keepalive) | XMLHttpRequest | Image Pixel |
|------|-----------|-------------------|----------------|-------------|
| ページ離脱時の送信 | 確実 | 確実 | 不確実 | 不確実 |
| ペイロードサイズ制限 | 64KB | 64KB (keepalive時) | 制限なし | URL長さ制限 |
| レスポンスの取得 | 不可 | 可 | 可 | 不可 |
| HTTP メソッド | POST のみ | 任意 | 任意 | GET のみ |
| Content-Type 設定 | 限定的 | 自由 | 自由 | N/A |
| CORS プリフライト | 条件による | 条件による | 条件による | 不要 |
| ブラウザサポート | 広範 | 広範 | 広範 | 全ブラウザ |
| キャンセル可能 | 不可 | AbortController | abort() | N/A |
| 推奨用途 | パフォーマンスデータ・分析 | 大きなペイロード | レガシー対応 | 最小限のトラッキング |

### 14.3 Core Web Vitals 改善テクニック比較

| テクニック | 対象指標 | 効果の大きさ | 実装難易度 | 説明 |
|-----------|---------|-------------|-----------|------|
| 画像の遅延読み込み | LCP, CLS | 大 | 低 | `loading="lazy"` + 寸法指定 |
| Critical CSS インライン化 | FCP, LCP | 大 | 中 | Above-the-fold の CSS をインライン展開 |
| JavaScript の分割読み込み | TBT, INP | 大 | 中 | dynamic import + React.lazy |
| フォントの最適化 | CLS, FCP | 中 | 低 | `font-display: swap` + preload |
| Service Worker キャッシュ | LCP, TTFB | 大 | 高 | Cache-first 戦略でオフライン対応 |
| CDN の導入 | TTFB, LCP | 大 | 低 | エッジサーバーからの配信 |
| HTTP/2 Server Push | LCP | 中 | 中 | 重要リソースの先行送信（非推奨化の流れあり） |
| `scheduler.yield()` | INP | 大 | 中 | Long Task を分割してメインスレッドを解放 |
| `content-visibility: auto` | LCP, INP | 中 | 低 | ビューポート外のレンダリングを遅延 |
| Speculation Rules API | LCP, FCP | 大 | 中 | リンク先の投機的プリレンダリング |

---

## 15. 参考文献

1. W3C. "Performance Timeline Level 2." W3C Recommendation, 2024. https://www.w3.org/TR/performance-timeline/
2. W3C. "Navigation Timing Level 2." W3C Recommendation, 2023. https://www.w3.org/TR/navigation-timing-2/
3. W3C. "Resource Timing Level 2." W3C Recommendation, 2023. https://www.w3.org/TR/resource-timing-2/
4. W3C. "User Timing Level 3." W3C Working Draft, 2024. https://www.w3.org/TR/user-timing/
5. W3C. "Long Tasks API." W3C Working Draft, 2024. https://www.w3.org/TR/longtasks-1/
6. Google. "Web Vitals." web.dev, 2024. https://web.dev/articles/vitals
7. Google. "Interaction to Next Paint (INP)." web.dev, 2024. https://web.dev/articles/inp
8. Google. "Optimize Cumulative Layout Shift." web.dev, 2024. https://web.dev/articles/optimize-cls
9. Google. "Optimize Largest Contentful Paint." web.dev, 2024. https://web.dev/articles/optimize-lcp
10. Google Chrome. "web-vitals." GitHub, 2024. https://github.com/GoogleChrome/web-vitals
11. Google. "Lighthouse." GitHub, 2024. https://github.com/GoogleChrome/lighthouse
12. Google. "Chrome UX Report (CrUX)." 2024. https://developer.chrome.com/docs/crux
13. Philip Walton. "Are long JavaScript tasks delaying your Time to Interactive?" web.dev, 2023. https://web.dev/articles/long-tasks-devtools

---

## まとめ

Performance API は、ブラウザのパフォーマンス計測において不可欠な基盤技術である。本章で扱った内容を以下に整理する。

| カテゴリ | 主要 API / ツール | 用途 |
|---------|-------------------|------|
| ページ読み込み計測 | Navigation Timing | リダイレクト、DNS、TCP、TTFB 等の段階別計測 |
| リソース計測 | Resource Timing | 個別リソースの読み込みパフォーマンス分析 |
| カスタム計測 | User Timing (mark/measure) | アプリケーション固有のパフォーマンスポイント定義 |
| リアルタイム監視 | PerformanceObserver | イベント駆動でのパフォーマンスエントリ通知 |
| ユーザー体験指標 | Core Web Vitals (LCP/INP/CLS) | Google が定義する3大ユーザー体験指標 |
| 自動監査 | Lighthouse / Lighthouse CI | パフォーマンススコアリングと改善提案 |
| データ送信 | Beacon API / fetch keepalive | ページ離脱時にも確実なデータ送信 |
| フィールドデータ | RUM / CrUX | 実ユーザー環境でのパフォーマンス把握 |
| 品質保証 | パフォーマンスバジェット | CI/CD での自動検証によるパフォーマンス退行防止 |

パフォーマンス改善は一度の対応で完結するものではなく、計測・分析・改善・検証のサイクルを継続的に回すことが重要である。ラボデータ（Lighthouse）とフィールドデータ（RUM/CrUX）を組み合わせ、パフォーマンスバジェットによる自動検証を導入することで、チーム全体でパフォーマンス品質を維持・向上させる体制を構築できる。

---

## 次に読むべきガイド

Performance API による計測とパフォーマンス改善の基礎を習得した後は、以下のガイドに進むことを推奨する。

**ブラウザとWebプラットフォームをさらに深く学ぶ**:
- **[ネットワーク基礎](../../network-fundamentals/docs/)**: TCP/IP、HTTP/2、HTTP/3（QUIC）、TLS といったネットワーク層の仕組みを理解することで、TTFB の改善や CDN の効果をより深く把握できる。特に HTTP/2 のマルチプレクシングや Server Push、HTTP/3 の 0-RTT 接続などは、パフォーマンス最適化において重要な知識である。

**Web開発全体への応用**:
- **[Web アプリケーション開発](../../web-application-development/docs/)**: React や Vue などのフレームワークを用いた実践的な開発において、パフォーマンス計測と最適化をどのように統合するかを学ぶ。Code splitting、Lazy loading、Server-Side Rendering（SSR）、Static Site Generation（SSG）といった手法と Performance API の組み合わせが扱われる。
