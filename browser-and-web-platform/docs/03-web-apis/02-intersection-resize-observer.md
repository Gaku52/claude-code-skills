# Observer API

> IntersectionObserverとResizeObserverは要素の可視性とサイズ変更を効率的に監視するAPI。遅延読み込み、無限スクロール、レスポンシブコンポーネントの実装に不可欠。

## この章で学ぶこと

- [ ] IntersectionObserverの仕組みと活用パターンを理解する
- [ ] ResizeObserverの使い方を把握する
- [ ] パフォーマンス面での利点を学ぶ

---

## 1. IntersectionObserver

```javascript
// IntersectionObserver = 要素のビューポート交差を監視

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        console.log('要素がビューポートに入った');
        console.log('交差率:', entry.intersectionRatio);
      }
    });
  },
  {
    root: null,           // ビューポート（null=ブラウザウィンドウ）
    rootMargin: '0px',    // ビューポートの拡張/縮小
    threshold: [0, 0.5, 1], // 交差率のしきい値
  }
);

observer.observe(targetElement);
observer.unobserve(targetElement);
observer.disconnect();
```

```javascript
// パターン1: 画像の遅延読み込み
const lazyObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const img = entry.target;
      img.src = img.dataset.src;
      img.removeAttribute('data-src');
      lazyObserver.unobserve(img);
    }
  });
}, { rootMargin: '200px' }); // 200px手前から読み込み開始

document.querySelectorAll('img[data-src]').forEach(img => {
  lazyObserver.observe(img);
});

// パターン2: 無限スクロール
const sentinel = document.getElementById('sentinel');
const scrollObserver = new IntersectionObserver(async (entries) => {
  if (entries[0].isIntersecting) {
    await loadMoreItems();
  }
});
scrollObserver.observe(sentinel);

// パターン3: スクロール連動アニメーション
const animObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    entry.target.classList.toggle('visible', entry.isIntersecting);
  });
}, { threshold: 0.1 });

document.querySelectorAll('.animate-on-scroll').forEach(el => {
  animObserver.observe(el);
});

// パターン4: 広告のビューアビリティ計測
const adObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.intersectionRatio >= 0.5) {
      trackAdImpression(entry.target.dataset.adId);
      adObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.5 });
```

---

## 2. ResizeObserver

```javascript
// ResizeObserver = 要素のサイズ変更を監視

const resizeObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const { width, height } = entry.contentRect;
    console.log(`Size: ${width}x${height}`);

    // borderBoxSize（border込み）
    const borderBoxSize = entry.borderBoxSize[0];
    console.log(`Border box: ${borderBoxSize.inlineSize}x${borderBoxSize.blockSize}`);
  }
});

resizeObserver.observe(element);

// パターン1: コンテナクエリの代替
const containerObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const { width } = entry.contentRect;
    const el = entry.target;

    el.classList.toggle('compact', width < 400);
    el.classList.toggle('regular', width >= 400 && width < 800);
    el.classList.toggle('wide', width >= 800);
  }
});

// パターン2: チャートの自動リサイズ
const chartObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const { width, height } = entry.contentRect;
    chart.resize(width, height);
  }
});
chartObserver.observe(chartContainer);

// パターン3: テキストの自動縮小
const textObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const el = entry.target;
    let fontSize = 24;
    el.style.fontSize = fontSize + 'px';
    while (el.scrollWidth > el.clientWidth && fontSize > 12) {
      fontSize--;
      el.style.fontSize = fontSize + 'px';
    }
  }
});
```

---

## 3. scroll vs IntersectionObserver

```
従来のスクロール監視:
  window.addEventListener('scroll', () => {
    elements.forEach(el => {
      const rect = el.getBoundingClientRect(); // 強制レイアウト
      if (rect.top < window.innerHeight) {
        // 処理
      }
    });
  });

  問題:
  → scroll イベントは高頻度で発火（1秒に数十回）
  → getBoundingClientRect() が Layout を強制
  → throttle/debounce が必要

IntersectionObserver:
  const observer = new IntersectionObserver(callback);
  elements.forEach(el => observer.observe(el));

  利点:
  ✓ ブラウザネイティブの最適化（メインスレッドをブロックしない）
  ✓ Layout の強制なし
  ✓ throttle 不要（ブラウザが最適なタイミングで通知）
  ✓ 非アクティブタブで停止
```

---

## 4. PerformanceObserver

```javascript
// PerformanceObserver = パフォーマンスイベントの監視

// LCP（Largest Contentful Paint）の計測
new PerformanceObserver((list) => {
  const entries = list.getEntries();
  const lastEntry = entries[entries.length - 1];
  console.log('LCP:', lastEntry.startTime);
}).observe({ type: 'largest-contentful-paint', buffered: true });

// Long Tasks の検出（50ms以上の処理）
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log('Long Task:', entry.duration, 'ms');
  }
}).observe({ type: 'longtask' });

// リソース読み込みの計測
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log(`${entry.name}: ${entry.duration.toFixed(0)}ms`);
  }
}).observe({ type: 'resource', buffered: true });

// Layout Shift の検出
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    if (!entry.hadRecentInput) {
      console.log('CLS:', entry.value, entry.sources);
    }
  }
}).observe({ type: 'layout-shift', buffered: true });
```

---

## まとめ

| Observer | 監視対象 | 主な用途 |
|----------|---------|---------|
| IntersectionObserver | ビューポートとの交差 | 遅延読み込み、無限スクロール |
| ResizeObserver | 要素サイズの変更 | レスポンシブ、チャートリサイズ |
| MutationObserver | DOM変更 | サードパーティDOM監視 |
| PerformanceObserver | パフォーマンスイベント | Web Vitals計測 |

---

## 次に読むべきガイド
→ [[../04-storage-and-caching/00-web-storage.md]] — Webストレージ

---

## 参考文献
1. MDN Web Docs. "Intersection Observer API." Mozilla, 2024.
2. MDN Web Docs. "Resize Observer API." Mozilla, 2024.
