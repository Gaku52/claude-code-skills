# Observer API

> IntersectionObserver、ResizeObserver、MutationObserver、PerformanceObserverは、要素の可視性・サイズ変更・DOM変更・パフォーマンスイベントを効率的に監視するブラウザネイティブAPI群。従来のscrollイベントやsetIntervalによるポーリングに比べて大幅にパフォーマンスが優れており、遅延読み込み、無限スクロール、レスポンシブコンポーネント、Web Vitals計測など幅広い実務シーンで不可欠な技術である。

## この章で学ぶこと

- [ ] IntersectionObserverの仕組みと活用パターンを理解する
- [ ] ResizeObserverの使い方とコンテナクエリとの比較を把握する
- [ ] MutationObserverでDOM変更を効率的に監視する方法を学ぶ
- [ ] PerformanceObserverによるWeb Vitals計測を実装できるようになる
- [ ] 各Observerのパフォーマンス面での利点とベストプラクティスを理解する
- [ ] Reactやフレームワークでのカスタムフック化パターンを身につける

---

## 1. IntersectionObserver

### 1.1 基本概念とAPI

IntersectionObserverは、ターゲット要素がルート要素（デフォルトではビューポート）と交差する状態を非同期的に監視するAPIである。スクロールイベントとgetBoundingClientRect()を使った従来の手法と異なり、ブラウザの内部最適化により、メインスレッドへの負荷を最小限に抑えることができる。

```javascript
// IntersectionObserverの基本構造
const observer = new IntersectionObserver(
  (entries, observer) => {
    // entries: IntersectionObserverEntry[] の配列
    // observer: IntersectionObserver インスタンス自身
    entries.forEach(entry => {
      // entry のプロパティ
      console.log('target:', entry.target);           // 監視対象のDOM要素
      console.log('isIntersecting:', entry.isIntersecting); // 交差しているか
      console.log('intersectionRatio:', entry.intersectionRatio); // 交差率 (0.0-1.0)
      console.log('intersectionRect:', entry.intersectionRect); // 交差領域
      console.log('boundingClientRect:', entry.boundingClientRect); // ターゲットの矩形
      console.log('rootBounds:', entry.rootBounds);   // ルート要素の矩形
      console.log('time:', entry.time);               // 交差が記録された時刻
    });
  },
  {
    root: null,             // 監視のルート要素（null=ビューポート）
    rootMargin: '0px',      // ルート要素のマージン（CSS形式: "10px 20px 30px 40px"）
    threshold: [0, 0.5, 1], // コールバック発火の交差率しきい値
  }
);

// 要素の監視開始
const targetElement = document.getElementById('target');
observer.observe(targetElement);

// 特定の要素の監視停止
observer.unobserve(targetElement);

// 全ての監視を停止
observer.disconnect();

// 現在監視中のエントリを取得（非同期的に保留中のものも含む）
const pendingEntries = observer.takeRecords();
```

### 1.2 thresholdの詳細

```javascript
// threshold: 単一値
const observer1 = new IntersectionObserver(callback, {
  threshold: 0,    // 1pxでも交差したらコールバック
});

const observer2 = new IntersectionObserver(callback, {
  threshold: 1.0,  // 要素が完全に見えたらコールバック
});

// threshold: 配列（複数のしきい値）
const observer3 = new IntersectionObserver(callback, {
  threshold: [0, 0.25, 0.5, 0.75, 1.0],
  // 0%, 25%, 50%, 75%, 100% の交差率でそれぞれコールバック
});

// 細かい段階の監視（スクロール連動アニメーション用）
const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
// [0, 0.01, 0.02, ..., 0.99]
const smoothObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    // intersectionRatioをCSSカスタムプロパティに反映
    entry.target.style.setProperty(
      '--visibility',
      String(entry.intersectionRatio)
    );
  });
}, { threshold: thresholds });

// CSSで利用
// .fade-in {
//   opacity: var(--visibility, 0);
//   transform: translateY(calc((1 - var(--visibility)) * 20px));
//   transition: opacity 0.1s, transform 0.1s;
// }
```

### 1.3 rootMarginの活用

```javascript
// rootMargin で監視領域を拡張・縮小する
// ビューポートの200px手前で検知（プリロードに最適）
const preloadObserver = new IntersectionObserver(callback, {
  rootMargin: '200px 0px', // 上下200px、左右0px
});

// ビューポートの50%内側に入ったら検知
const innerObserver = new IntersectionObserver(callback, {
  rootMargin: '-50% 0px', // 上下を50%縮小
});

// 非対称なマージン（上方向に多く取る）
const asymmetricObserver = new IntersectionObserver(callback, {
  rootMargin: '300px 0px 0px 0px', // 上300px、右0px、下0px、左0px
});

// ★ rootMargin の値はCSSのmargin shorthand と同じ形式
// "10px"          → 全方向 10px
// "10px 20px"     → 上下10px、左右20px
// "10px 20px 30px"    → 上10px、左右20px、下30px
// "10px 20px 30px 40px" → 上10px、右20px、下30px、左40px

// パーセンテージも使用可能（ルート要素に対する割合）
const percentObserver = new IntersectionObserver(callback, {
  rootMargin: '-25%', // ルート要素を25%縮小して監視
});
```

### 1.4 カスタムルート要素

```javascript
// スクロールコンテナをルートに指定
const scrollContainer = document.getElementById('scroll-container');

const containerObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        console.log('要素がスクロールコンテナ内に表示された');
      }
    });
  },
  {
    root: scrollContainer, // ビューポートの代わりにこのコンテナを基準にする
    rootMargin: '50px',
    threshold: 0,
  }
);

// スクロールコンテナ内の全アイテムを監視
scrollContainer.querySelectorAll('.list-item').forEach(item => {
  containerObserver.observe(item);
});

// ★ 注意: rootはターゲット要素の祖先である必要がある
// ★ root: null はビューポート（暗黙のルート）を意味する
```

---

## 2. IntersectionObserver の実務パターン

### 2.1 画像の遅延読み込み（Lazy Loading）

```javascript
// バニラJSでの画像遅延読み込み
class LazyImageLoader {
  constructor(options = {}) {
    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        rootMargin: options.rootMargin || '200px 0px',
        threshold: 0,
      }
    );
  }

  handleIntersection(entries) {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;

      const element = entry.target;

      if (element.tagName === 'IMG') {
        this.loadImage(element);
      } else if (element.tagName === 'VIDEO') {
        this.loadVideo(element);
      } else {
        // 背景画像の遅延読み込み
        this.loadBackground(element);
      }

      this.observer.unobserve(element);
    });
  }

  loadImage(img) {
    // srcsetの処理
    if (img.dataset.srcset) {
      img.srcset = img.dataset.srcset;
    }
    // sizesの処理
    if (img.dataset.sizes) {
      img.sizes = img.dataset.sizes;
    }
    // src の処理
    if (img.dataset.src) {
      img.src = img.dataset.src;
    }

    img.classList.add('loaded');
    img.removeAttribute('data-src');
    img.removeAttribute('data-srcset');
    img.removeAttribute('data-sizes');
  }

  loadVideo(video) {
    // source要素のdata-srcを処理
    video.querySelectorAll('source').forEach(source => {
      if (source.dataset.src) {
        source.src = source.dataset.src;
      }
    });
    video.load();
    video.classList.add('loaded');
  }

  loadBackground(element) {
    if (element.dataset.bg) {
      element.style.backgroundImage = `url('${element.dataset.bg}')`;
      element.classList.add('loaded');
    }
  }

  observe(element) {
    this.observer.observe(element);
  }

  observeAll(selector) {
    document.querySelectorAll(selector).forEach(el => this.observe(el));
  }

  destroy() {
    this.observer.disconnect();
  }
}

// 使用例
const lazyLoader = new LazyImageLoader({ rootMargin: '300px 0px' });
lazyLoader.observeAll('[data-src], [data-bg]');

// HTML側
// <img data-src="large-image.jpg"
//      data-srcset="small.jpg 480w, medium.jpg 800w, large.jpg 1200w"
//      data-sizes="(max-width: 600px) 480px, (max-width: 1024px) 800px, 1200px"
//      src="placeholder.svg"
//      alt="Description"
//      class="lazy" />

// ★ 現在は loading="lazy" 属性が推奨（ブラウザネイティブ）
// <img src="image.jpg" loading="lazy" alt="Description" />
// ただし、細かい制御が必要な場合はIntersectionObserverを使用する
```

### 2.2 無限スクロール

```javascript
// 高機能な無限スクロール実装
class InfiniteScroll {
  constructor(options) {
    this.container = options.container;
    this.loadMore = options.loadMore;
    this.threshold = options.threshold || 1;
    this.loading = false;
    this.hasMore = true;

    // 番兵要素の作成
    this.sentinel = document.createElement('div');
    this.sentinel.className = 'infinite-scroll-sentinel';
    this.sentinel.setAttribute('aria-hidden', 'true');
    this.container.appendChild(this.sentinel);

    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        root: options.root || null,
        rootMargin: options.rootMargin || '400px 0px',
        threshold: 0,
      }
    );

    this.observer.observe(this.sentinel);
  }

  async handleIntersection(entries) {
    const entry = entries[0];

    if (!entry.isIntersecting || this.loading || !this.hasMore) return;

    this.loading = true;
    this.showLoadingIndicator();

    try {
      const result = await this.loadMore();

      if (result.items.length === 0 || !result.hasMore) {
        this.hasMore = false;
        this.observer.disconnect();
        this.showEndMessage();
      } else {
        this.appendItems(result.items);
      }
    } catch (error) {
      console.error('Failed to load more items:', error);
      this.showError(error);
    } finally {
      this.loading = false;
      this.hideLoadingIndicator();
    }
  }

  appendItems(items) {
    const fragment = document.createDocumentFragment();
    items.forEach(item => {
      const element = this.createItemElement(item);
      fragment.appendChild(element);
    });

    // 番兵要素の前に挿入
    this.container.insertBefore(fragment, this.sentinel);
  }

  createItemElement(item) {
    const div = document.createElement('div');
    div.className = 'scroll-item';
    div.innerHTML = `<h3>${item.title}</h3><p>${item.description}</p>`;
    return div;
  }

  showLoadingIndicator() {
    this.sentinel.textContent = 'Loading...';
    this.sentinel.classList.add('loading');
  }

  hideLoadingIndicator() {
    this.sentinel.textContent = '';
    this.sentinel.classList.remove('loading');
  }

  showEndMessage() {
    this.sentinel.textContent = 'All items loaded.';
    this.sentinel.classList.add('end');
  }

  showError(error) {
    this.sentinel.textContent = 'Error loading items. Click to retry.';
    this.sentinel.classList.add('error');
    this.sentinel.onclick = () => {
      this.sentinel.classList.remove('error');
      this.hasMore = true;
      this.observer.observe(this.sentinel);
    };
  }

  destroy() {
    this.observer.disconnect();
    this.sentinel.remove();
  }
}

// 使用例
let page = 0;
const infiniteScroll = new InfiniteScroll({
  container: document.getElementById('items-container'),
  loadMore: async () => {
    page++;
    const response = await fetch(`/api/items?page=${page}&limit=20`);
    const data = await response.json();
    return {
      items: data.items,
      hasMore: data.hasMore,
    };
  },
});
```

### 2.3 スクロール連動アニメーション

```javascript
// フェードインアニメーション
class ScrollAnimator {
  constructor(options = {}) {
    this.animations = new Map();

    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        rootMargin: options.rootMargin || '0px 0px -10% 0px',
        threshold: options.threshold || [0, 0.1, 0.2, 0.3, 0.4, 0.5],
      }
    );
  }

  handleIntersection(entries) {
    entries.forEach(entry => {
      const animationType = this.animations.get(entry.target);

      if (entry.isIntersecting) {
        this.applyAnimation(entry.target, animationType, entry.intersectionRatio);
      }
    });
  }

  applyAnimation(element, type, ratio) {
    switch (type) {
      case 'fade-in':
        element.style.opacity = Math.min(ratio * 2, 1);
        element.style.transform = `translateY(${(1 - Math.min(ratio * 2, 1)) * 30}px)`;
        break;

      case 'slide-left':
        element.style.opacity = Math.min(ratio * 2, 1);
        element.style.transform = `translateX(${(1 - Math.min(ratio * 2, 1)) * -50}px)`;
        break;

      case 'slide-right':
        element.style.opacity = Math.min(ratio * 2, 1);
        element.style.transform = `translateX(${(1 - Math.min(ratio * 2, 1)) * 50}px)`;
        break;

      case 'scale-up':
        const scale = 0.8 + Math.min(ratio * 2, 1) * 0.2;
        element.style.opacity = Math.min(ratio * 2, 1);
        element.style.transform = `scale(${scale})`;
        break;

      case 'reveal':
        if (ratio > 0.1) {
          element.classList.add('revealed');
          this.observer.unobserve(element);
        }
        break;
    }
  }

  register(element, animationType = 'fade-in') {
    this.animations.set(element, animationType);
    // 初期状態を設定
    element.style.opacity = '0';
    element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    this.observer.observe(element);
  }

  registerAll(selector, animationType = 'fade-in') {
    document.querySelectorAll(selector).forEach(el => {
      this.register(el, animationType);
    });
  }

  destroy() {
    this.observer.disconnect();
    this.animations.clear();
  }
}

// 使用例
const animator = new ScrollAnimator();
animator.registerAll('.section-title', 'fade-in');
animator.registerAll('.card-left', 'slide-left');
animator.registerAll('.card-right', 'slide-right');
animator.registerAll('.feature-icon', 'scale-up');

// CSS
// .revealed {
//   animation: reveal 0.8s ease forwards;
// }
// @keyframes reveal {
//   from { opacity: 0; transform: translateY(20px); }
//   to { opacity: 1; transform: translateY(0); }
// }
```

### 2.4 ビューアビリティ計測と分析

```javascript
// 広告やコンテンツのビューアビリティ計測
class ViewabilityTracker {
  constructor(options = {}) {
    this.minVisibleRatio = options.minVisibleRatio || 0.5;
    this.minVisibleTime = options.minVisibleTime || 1000; // 1秒
    this.timers = new Map();
    this.tracked = new Set();
    this.onViewable = options.onViewable || (() => {});

    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        threshold: [0, this.minVisibleRatio],
      }
    );
  }

  handleIntersection(entries) {
    entries.forEach(entry => {
      const target = entry.target;
      const id = target.dataset.trackId;

      if (this.tracked.has(id)) return;

      if (entry.intersectionRatio >= this.minVisibleRatio) {
        // 表示開始: タイマーを設定
        if (!this.timers.has(id)) {
          const timer = setTimeout(() => {
            this.tracked.add(id);
            this.timers.delete(id);
            this.onViewable({
              id,
              element: target,
              timestamp: Date.now(),
              ratio: entry.intersectionRatio,
            });
            this.observer.unobserve(target);
          }, this.minVisibleTime);

          this.timers.set(id, timer);
        }
      } else {
        // 非表示: タイマーをクリア
        const timer = this.timers.get(id);
        if (timer) {
          clearTimeout(timer);
          this.timers.delete(id);
        }
      }
    });
  }

  track(element) {
    if (!element.dataset.trackId) {
      element.dataset.trackId = `track-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    }
    this.observer.observe(element);
  }

  destroy() {
    this.observer.disconnect();
    this.timers.forEach(timer => clearTimeout(timer));
    this.timers.clear();
  }
}

// 使用例
const tracker = new ViewabilityTracker({
  minVisibleRatio: 0.5,
  minVisibleTime: 2000, // 2秒以上50%以上表示でビューアブル
  onViewable({ id, element }) {
    console.log(`Element ${id} is viewable`);
    // アナリティクスに送信
    fetch('/api/analytics/viewability', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        elementId: id,
        contentType: element.dataset.contentType,
        timestamp: new Date().toISOString(),
      }),
      keepalive: true,
    });
  },
});

document.querySelectorAll('[data-track]').forEach(el => tracker.track(el));
```

### 2.5 セクションナビゲーション（アクティブセクション検出）

```javascript
// スクロール位置に応じたナビゲーションのアクティブ状態更新
class SectionNavigator {
  constructor(options = {}) {
    this.sections = new Map();
    this.activeSection = null;
    this.onSectionChange = options.onSectionChange || (() => {});

    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        rootMargin: '-20% 0px -70% 0px', // ビューポート上部20-30%で検知
        threshold: 0,
      }
    );
  }

  handleIntersection(entries) {
    entries.forEach(entry => {
      const sectionId = entry.target.id;

      if (entry.isIntersecting) {
        if (this.activeSection !== sectionId) {
          this.activeSection = sectionId;
          this.updateNavigation(sectionId);
          this.onSectionChange(sectionId);
        }
      }
    });
  }

  updateNavigation(activeSectionId) {
    // ナビゲーションリンクのアクティブ状態を更新
    document.querySelectorAll('.nav-link').forEach(link => {
      const isActive = link.getAttribute('href') === `#${activeSectionId}`;
      link.classList.toggle('active', isActive);
      link.setAttribute('aria-current', isActive ? 'true' : 'false');
    });
  }

  register(section) {
    this.sections.set(section.id, section);
    this.observer.observe(section);
  }

  registerAll(selector) {
    document.querySelectorAll(selector).forEach(section => {
      if (section.id) {
        this.register(section);
      }
    });
  }

  destroy() {
    this.observer.disconnect();
    this.sections.clear();
  }
}

// 使用例
const sectionNav = new SectionNavigator({
  onSectionChange(sectionId) {
    // URLハッシュの更新（pushStateで履歴に追加しない）
    history.replaceState(null, '', `#${sectionId}`);
  },
});
sectionNav.registerAll('section[id]');
```

---

## 3. ResizeObserver

### 3.1 基本概念とAPI

ResizeObserverは要素のサイズ変更を効率的に監視するAPIである。ウィンドウのリサイズだけでなく、CSSアニメーション、DOM操作、フレックスボックス/グリッドのレイアウト変更など、あらゆる原因によるサイズ変更を検出できる。

```javascript
// ResizeObserverの基本構造
const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    // contentRect: コンテンツ領域のサイズ（paddingを除く）
    const { width, height, top, left } = entry.contentRect;
    console.log(`Content size: ${width}x${height}`);
    console.log(`Content position: (${left}, ${top})`);

    // contentBoxSize: コンテンツボックスのサイズ（新しいAPI）
    if (entry.contentBoxSize) {
      // 配列で返される（将来のフラグメンテーション対応）
      const contentBox = entry.contentBoxSize[0];
      console.log(`Content box: ${contentBox.inlineSize}x${contentBox.blockSize}`);
    }

    // borderBoxSize: ボーダーボックスのサイズ（padding + border含む）
    if (entry.borderBoxSize) {
      const borderBox = entry.borderBoxSize[0];
      console.log(`Border box: ${borderBox.inlineSize}x${borderBox.blockSize}`);
    }

    // devicePixelContentBoxSize: デバイスピクセル単位
    if (entry.devicePixelContentBoxSize) {
      const devicePixelBox = entry.devicePixelContentBoxSize[0];
      console.log(`Device pixel: ${devicePixelBox.inlineSize}x${devicePixelBox.blockSize}`);
    }

    console.log('Target element:', entry.target);
  }
});

// 要素の監視
observer.observe(element);

// 特定のboxモデルで監視
observer.observe(element, { box: 'border-box' });   // ボーダーボックス
observer.observe(element, { box: 'content-box' });   // コンテンツボックス（デフォルト）
observer.observe(element, { box: 'device-pixel-content-box' }); // デバイスピクセル

// 監視停止
observer.unobserve(element);
observer.disconnect();
```

### 3.2 inlineSize / blockSize について

```javascript
// ★ inlineSizeとblockSizeは論理的なサイズ
// 横書き（writing-mode: horizontal-tb）の場合:
//   inlineSize = width（横方向）
//   blockSize = height（縦方向）
//
// 縦書き（writing-mode: vertical-rl）の場合:
//   inlineSize = height（縦方向）
//   blockSize = width（横方向）

// 多言語対応のレイアウト処理
const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const { inlineSize, blockSize } = entry.contentBoxSize[0];

    // 論理的なサイズに基づいてレイアウトを調整
    // writing-modeに関係なく正しく動作する
    if (inlineSize < 400) {
      entry.target.classList.add('compact-layout');
    } else {
      entry.target.classList.remove('compact-layout');
    }
  }
});
```

### 3.3 コンテナクエリの代替

```javascript
// ResizeObserverによるコンテナクエリの実装
class ContainerQuery {
  constructor() {
    this.queries = new Map();
    this.observer = new ResizeObserver(this.handleResize.bind(this));
  }

  handleResize(entries) {
    for (const entry of entries) {
      const { inlineSize: width } = entry.contentBoxSize[0];
      const queries = this.queries.get(entry.target) || [];

      for (const query of queries) {
        const matches = this.evaluateQuery(width, query.condition);
        entry.target.classList.toggle(query.className, matches);
      }
    }
  }

  evaluateQuery(width, condition) {
    if (condition.minWidth !== undefined && width < condition.minWidth) return false;
    if (condition.maxWidth !== undefined && width > condition.maxWidth) return false;
    return true;
  }

  register(element, queries) {
    this.queries.set(element, queries);
    this.observer.observe(element);
  }

  destroy() {
    this.observer.disconnect();
    this.queries.clear();
  }
}

// 使用例
const cq = new ContainerQuery();
cq.register(document.querySelector('.card-container'), [
  { className: 'cq-small', condition: { maxWidth: 400 } },
  { className: 'cq-medium', condition: { minWidth: 401, maxWidth: 800 } },
  { className: 'cq-large', condition: { minWidth: 801 } },
]);

// ★ 現在はCSSネイティブのコンテナクエリが推奨
// @container (min-width: 400px) {
//   .card { grid-template-columns: 1fr 1fr; }
// }
// ただし、JavaScript連携が必要な場合はResizeObserverを使用する
```

### 3.4 チャートの自動リサイズ

```javascript
// D3.js / Chart.js / ECharts などのチャートライブラリ連携
class ResponsiveChart {
  constructor(container, chartLib) {
    this.container = container;
    this.chart = null;
    this.chartLib = chartLib;
    this.resizeTimeout = null;

    this.observer = new ResizeObserver((entries) => {
      // デバウンス処理（頻繁なリサイズを抑制）
      if (this.resizeTimeout) {
        cancelAnimationFrame(this.resizeTimeout);
      }

      this.resizeTimeout = requestAnimationFrame(() => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;

          if (width > 0 && height > 0) {
            this.resize(width, height);
          }
        }
      });
    });

    this.observer.observe(container);
  }

  resize(width, height) {
    if (this.chart) {
      // Chart.js の場合
      this.chart.resize(width, height);

      // ECharts の場合
      // this.chart.resize({ width, height });

      // D3.js の場合
      // d3.select(this.container).select('svg')
      //   .attr('width', width)
      //   .attr('height', height);
    }
  }

  destroy() {
    this.observer.disconnect();
    if (this.resizeTimeout) {
      cancelAnimationFrame(this.resizeTimeout);
    }
  }
}

// Canvas要素のデバイスピクセル比対応
class ResponsiveCanvas {
  constructor(container) {
    this.container = container;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    container.appendChild(this.canvas);

    this.observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        // devicePixelContentBoxSizeでピクセルパーフェクトなリサイズ
        if (entry.devicePixelContentBoxSize) {
          const { inlineSize, blockSize } = entry.devicePixelContentBoxSize[0];
          this.canvas.width = inlineSize;
          this.canvas.height = blockSize;
        } else {
          const dpr = window.devicePixelRatio || 1;
          const { width, height } = entry.contentRect;
          this.canvas.width = Math.round(width * dpr);
          this.canvas.height = Math.round(height * dpr);
        }

        this.render();
      }
    });

    this.observer.observe(container, { box: 'device-pixel-content-box' });
  }

  render() {
    const { width, height } = this.canvas;
    this.ctx.clearRect(0, 0, width, height);
    // 描画処理...
  }

  destroy() {
    this.observer.disconnect();
    this.canvas.remove();
  }
}
```

### 3.5 テキストの自動縮小（FitText）

```javascript
// テキストを要素幅に合わせて自動縮小
class AutoFitText {
  constructor(options = {}) {
    this.minFontSize = options.minFontSize || 10;
    this.maxFontSize = options.maxFontSize || 100;
    this.elements = new Map();

    this.observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        this.fitText(entry.target);
      }
    });
  }

  fitText(element) {
    const config = this.elements.get(element);
    if (!config) return;

    const containerWidth = element.parentElement.clientWidth;
    let fontSize = config.maxFontSize || this.maxFontSize;
    const minSize = config.minFontSize || this.minFontSize;

    // バイナリサーチで最適なフォントサイズを見つける
    let low = minSize;
    let high = fontSize;

    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      element.style.fontSize = `${mid}px`;

      if (element.scrollWidth <= containerWidth) {
        low = mid + 1;
        fontSize = mid;
      } else {
        high = mid - 1;
      }
    }

    element.style.fontSize = `${fontSize}px`;
  }

  observe(element, config = {}) {
    this.elements.set(element, config);
    this.observer.observe(element.parentElement);
    this.fitText(element);
  }

  unobserve(element) {
    this.elements.delete(element);
    if (element.parentElement) {
      this.observer.unobserve(element.parentElement);
    }
  }

  destroy() {
    this.observer.disconnect();
    this.elements.clear();
  }
}

// 使用例
const autoFit = new AutoFitText({ minFontSize: 12, maxFontSize: 48 });
autoFit.observe(document.querySelector('.headline'), {
  maxFontSize: 64,
});
```

### 3.6 仮想スクロールとの連携

```javascript
// ResizeObserverを使った動的高さの仮想スクロール
class VirtualList {
  constructor(container, options) {
    this.container = container;
    this.items = options.items || [];
    this.renderItem = options.renderItem;
    this.itemHeights = new Map();
    this.defaultHeight = options.estimatedItemHeight || 50;
    this.overscan = options.overscan || 5;

    // スクロールコンテナの設定
    this.viewport = document.createElement('div');
    this.viewport.style.cssText = 'overflow-y: auto; height: 100%;';
    this.spacer = document.createElement('div');
    this.content = document.createElement('div');
    this.viewport.appendChild(this.spacer);
    this.viewport.appendChild(this.content);
    container.appendChild(this.viewport);

    // アイテムの高さを計測
    this.heightObserver = new ResizeObserver((entries) => {
      let heightChanged = false;

      for (const entry of entries) {
        const index = parseInt(entry.target.dataset.virtualIndex, 10);
        const newHeight = entry.borderBoxSize[0].blockSize;

        if (this.itemHeights.get(index) !== newHeight) {
          this.itemHeights.set(index, newHeight);
          heightChanged = true;
        }
      }

      if (heightChanged) {
        this.updateSpacerHeight();
        this.render();
      }
    });

    // ビューポートのリサイズ監視
    this.viewportObserver = new ResizeObserver(() => {
      this.render();
    });
    this.viewportObserver.observe(this.viewport);

    this.viewport.addEventListener('scroll', () => this.render());
    this.render();
  }

  getItemHeight(index) {
    return this.itemHeights.get(index) || this.defaultHeight;
  }

  getItemTop(index) {
    let top = 0;
    for (let i = 0; i < index; i++) {
      top += this.getItemHeight(i);
    }
    return top;
  }

  getTotalHeight() {
    let total = 0;
    for (let i = 0; i < this.items.length; i++) {
      total += this.getItemHeight(i);
    }
    return total;
  }

  updateSpacerHeight() {
    this.spacer.style.height = `${this.getTotalHeight()}px`;
  }

  render() {
    const scrollTop = this.viewport.scrollTop;
    const viewportHeight = this.viewport.clientHeight;

    // 表示範囲のアイテムを計算
    let startIndex = 0;
    let accumulatedHeight = 0;

    while (startIndex < this.items.length) {
      accumulatedHeight += this.getItemHeight(startIndex);
      if (accumulatedHeight > scrollTop) break;
      startIndex++;
    }

    startIndex = Math.max(0, startIndex - this.overscan);

    let endIndex = startIndex;
    accumulatedHeight = this.getItemTop(endIndex);

    while (endIndex < this.items.length && accumulatedHeight < scrollTop + viewportHeight) {
      accumulatedHeight += this.getItemHeight(endIndex);
      endIndex++;
    }

    endIndex = Math.min(this.items.length - 1, endIndex + this.overscan);

    // DOMの更新
    this.content.innerHTML = '';
    const fragment = document.createDocumentFragment();

    for (let i = startIndex; i <= endIndex; i++) {
      const element = this.renderItem(this.items[i], i);
      element.dataset.virtualIndex = String(i);
      element.style.position = 'absolute';
      element.style.top = `${this.getItemTop(i)}px`;
      element.style.width = '100%';

      this.heightObserver.observe(element);
      fragment.appendChild(element);
    }

    this.content.style.position = 'relative';
    this.content.appendChild(fragment);
  }

  destroy() {
    this.heightObserver.disconnect();
    this.viewportObserver.disconnect();
    this.container.innerHTML = '';
  }
}
```

---

## 4. MutationObserver

### 4.1 基本概念とAPI

MutationObserverはDOMツリーの変更を監視するAPIである。属性の変更、子ノードの追加・削除、テキストコンテンツの変更などを検出できる。

```javascript
// MutationObserverの基本構造
const observer = new MutationObserver((mutations, observer) => {
  for (const mutation of mutations) {
    switch (mutation.type) {
      case 'childList':
        // 子ノードの追加・削除
        console.log('Added nodes:', mutation.addedNodes);
        console.log('Removed nodes:', mutation.removedNodes);
        break;

      case 'attributes':
        // 属性の変更
        console.log('Attribute changed:', mutation.attributeName);
        console.log('Old value:', mutation.oldValue);
        console.log('New value:', mutation.target.getAttribute(mutation.attributeName));
        break;

      case 'characterData':
        // テキストノードの変更
        console.log('Text changed:', mutation.target.textContent);
        console.log('Old value:', mutation.oldValue);
        break;
    }
  }
});

// 監視オプション
observer.observe(targetNode, {
  childList: true,         // 子ノードの追加・削除を監視
  attributes: true,        // 属性の変更を監視
  characterData: true,     // テキストノードの変更を監視
  subtree: true,           // 子孫ノードも含めて監視
  attributeOldValue: true, // 変更前の属性値を記録
  characterDataOldValue: true, // 変更前のテキストを記録
  attributeFilter: ['class', 'style', 'data-state'], // 監視する属性を限定
});

// 保留中の変更を取得して監視を停止
const pendingMutations = observer.takeRecords();
observer.disconnect();
```

### 4.2 DOM変更の監視パターン

```javascript
// パターン1: サードパーティスクリプトのDOM監視
// 外部スクリプトが意図しないDOM変更を行わないか監視
class DOMGuard {
  constructor(protectedElement) {
    this.element = protectedElement;
    this.originalHTML = protectedElement.innerHTML;

    this.observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        // 不正なスクリプトタグの挿入を検出
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            if (node.tagName === 'SCRIPT' || node.tagName === 'IFRAME') {
              console.warn('Suspicious element injected:', node);
              node.remove(); // 不正な要素を削除
            }
          }
        }

        // 重要な属性の変更を検出
        if (mutation.type === 'attributes') {
          if (mutation.attributeName === 'style' || mutation.attributeName === 'class') {
            console.warn(
              `Attribute "${mutation.attributeName}" changed on`,
              mutation.target
            );
          }
        }
      }
    });

    this.observer.observe(protectedElement, {
      childList: true,
      attributes: true,
      subtree: true,
      attributeFilter: ['style', 'class', 'href', 'src'],
    });
  }

  destroy() {
    this.observer.disconnect();
  }
}

// パターン2: 動的コンテンツの自動初期化
// SPAやサードパーティウィジェットで動的に追加される要素を自動検出
class AutoInitializer {
  constructor(config) {
    this.config = config; // { selector: string, init: (element) => void }[]

    this.observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            this.initElement(node);
            // 追加されたノードの子要素もチェック
            node.querySelectorAll?.('*').forEach(child => {
              this.initElement(child);
            });
          }
        }
      }
    });

    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // 既存の要素も初期化
    this.config.forEach(({ selector, init }) => {
      document.querySelectorAll(selector).forEach(init);
    });
  }

  initElement(element) {
    for (const { selector, init } of this.config) {
      if (element.matches?.(selector) && !element.dataset.initialized) {
        element.dataset.initialized = 'true';
        init(element);
      }
    }
  }

  destroy() {
    this.observer.disconnect();
  }
}

// 使用例
const autoInit = new AutoInitializer([
  {
    selector: '[data-tooltip]',
    init: (el) => new Tooltip(el, { content: el.dataset.tooltip }),
  },
  {
    selector: '[data-datepicker]',
    init: (el) => new DatePicker(el),
  },
  {
    selector: 'pre code',
    init: (el) => hljs.highlightElement(el),
  },
]);

// パターン3: フォームの変更検出
class FormChangeDetector {
  constructor(form) {
    this.form = form;
    this.isDirty = false;
    this.initialValues = this.captureValues();

    // 属性の変更を監視（value属性はプロパティなので直接監視できない）
    this.observer = new MutationObserver((mutations) => {
      this.checkDirty();
    });

    this.observer.observe(form, {
      attributes: true,
      subtree: true,
      attributeFilter: ['value', 'checked', 'selected'],
    });

    // inputイベントも監視（valueプロパティの変更はMutationObserverでは検出できない）
    form.addEventListener('input', () => this.checkDirty());
    form.addEventListener('change', () => this.checkDirty());
  }

  captureValues() {
    const values = {};
    new FormData(this.form).forEach((value, key) => {
      values[key] = value;
    });
    return values;
  }

  checkDirty() {
    const currentValues = this.captureValues();
    this.isDirty = JSON.stringify(currentValues) !== JSON.stringify(this.initialValues);

    this.form.dispatchEvent(new CustomEvent('dirtychange', {
      detail: { isDirty: this.isDirty },
    }));
  }

  reset() {
    this.initialValues = this.captureValues();
    this.isDirty = false;
  }

  destroy() {
    this.observer.disconnect();
  }
}
```

### 4.3 MutationObserverの注意点

```javascript
// ★ 注意1: コールバックは同期的なDOM変更がすべて完了してから呼ばれる
// （マイクロタスクとして実行される）
element.setAttribute('class', 'foo');
element.setAttribute('class', 'bar');
element.setAttribute('class', 'baz');
// → コールバックは1回だけ呼ばれ、3つのmutationが含まれる

// ★ 注意2: 無限ループに注意
// コールバック内でDOMを変更すると再度コールバックが呼ばれる
const observer = new MutationObserver((mutations) => {
  // 危険: 無限ループになる可能性がある
  // mutations[0].target.textContent = 'updated';

  // 安全: 一時的に監視を停止
  observer.disconnect();
  mutations[0].target.textContent = 'updated';
  observer.observe(targetNode, options);
});

// ★ 注意3: パフォーマンスへの影響
// subtree: true で広範囲を監視すると負荷が高い
// 必要最小限の範囲とフィルターで監視する

// ★ 注意4: CSSプロパティの変更はMutationObserverでは検出できない
// style属性の変更は検出できるが、CSSクラスの結果としてのスタイル変更は検出不可
// → ResizeObserverやgetComputedStyleを使用する
```

---

## 5. PerformanceObserver

### 5.1 基本概念とAPI

PerformanceObserverはブラウザのパフォーマンスエントリを非同期的に監視するAPIである。Performance Timelineの一部であり、さまざまなパフォーマンスメトリクスをリアルタイムで収集できる。

```javascript
// PerformanceObserverの基本構造
const observer = new PerformanceObserver((list, observer) => {
  const entries = list.getEntries();
  for (const entry of entries) {
    console.log(entry.name, entry.entryType, entry.startTime, entry.duration);
  }
});

// 監視するエントリタイプを指定
observer.observe({
  type: 'resource',     // 単一タイプ
  buffered: true,       // 過去のエントリも含める
});

// 複数タイプを同時に監視
observer.observe({
  entryTypes: ['navigation', 'resource', 'paint'],
  // ★ entryTypes と type は同時に使えない
  // ★ entryTypes では buffered オプションは使えない
});

// 監視停止
observer.disconnect();

// サポートされているエントリタイプの確認
const supportedTypes = PerformanceObserver.supportedEntryTypes;
console.log(supportedTypes);
// ['element', 'event', 'first-input', 'largest-contentful-paint',
//  'layout-shift', 'longtask', 'mark', 'measure', 'navigation',
//  'paint', 'resource', 'visibility-state']
```

### 5.2 Core Web Vitals の計測

```javascript
// LCP（Largest Contentful Paint）: 最大のコンテンツが描画される時間
function observeLCP(callback) {
  let lcpValue = 0;

  const observer = new PerformanceObserver((list) => {
    const entries = list.getEntries();
    const lastEntry = entries[entries.length - 1];
    lcpValue = lastEntry.startTime;
  });

  observer.observe({ type: 'largest-contentful-paint', buffered: true });

  // ユーザーインタラクション時にLCPを確定
  // （LCPはユーザーインタラクションまで更新され続ける）
  const reportLCP = () => {
    observer.disconnect();
    callback(lcpValue);
  };

  // 各種イベントで確定
  ['keydown', 'click', 'scroll'].forEach(type => {
    addEventListener(type, reportLCP, { once: true });
  });

  // ページ遷移時にも報告
  addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      reportLCP();
    }
  }, { once: true });
}

// FID（First Input Delay）: 最初のインタラクションの遅延
function observeFID(callback) {
  const observer = new PerformanceObserver((list) => {
    const entries = list.getEntries();
    const firstInput = entries[0];

    // processingStart - startTime が入力遅延
    const delay = firstInput.processingStart - firstInput.startTime;
    callback(delay);
    observer.disconnect();
  });

  observer.observe({ type: 'first-input', buffered: true });
}

// INP（Interaction to Next Paint）: インタラクションからの応答性
function observeINP(callback) {
  const interactions = new Map();
  let longestDuration = 0;

  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      // 同じインタラクションのイベントをグループ化
      const interactionId = entry.interactionId;
      if (!interactionId) continue;

      const existingDuration = interactions.get(interactionId) || 0;
      const newDuration = Math.max(existingDuration, entry.duration);
      interactions.set(interactionId, newDuration);

      if (newDuration > longestDuration) {
        longestDuration = newDuration;
      }
    }
  });

  observer.observe({ type: 'event', buffered: true, durationThreshold: 16 });

  // ページ非表示時に報告
  addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      // 98パーセンタイルを計算
      const sortedDurations = [...interactions.values()].sort((a, b) => a - b);
      const p98Index = Math.floor(sortedDurations.length * 0.98) - 1;
      const inp = sortedDurations[Math.max(p98Index, 0)] || 0;
      callback(inp);
      observer.disconnect();
    }
  }, { once: true });
}

// CLS（Cumulative Layout Shift）: 累積レイアウトシフト
function observeCLS(callback) {
  let clsValue = 0;
  let sessionValue = 0;
  let sessionEntries = [];
  let clsEntries = [];

  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      // ユーザー入力に起因するシフトは除外
      if (entry.hadRecentInput) continue;

      const firstSessionEntry = sessionEntries[0];
      const lastSessionEntry = sessionEntries[sessionEntries.length - 1];

      // セッションウィンドウの条件:
      // 1. 前のエントリから1秒以内
      // 2. セッション全体が5秒以内
      if (
        sessionEntries.length > 0 &&
        entry.startTime - lastSessionEntry.startTime < 1000 &&
        entry.startTime - firstSessionEntry.startTime < 5000
      ) {
        sessionValue += entry.value;
        sessionEntries.push(entry);
      } else {
        // 新しいセッションを開始
        sessionValue = entry.value;
        sessionEntries = [entry];
      }

      if (sessionValue > clsValue) {
        clsValue = sessionValue;
        clsEntries = [...sessionEntries];
      }
    }
  });

  observer.observe({ type: 'layout-shift', buffered: true });

  addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
      callback({
        value: clsValue,
        entries: clsEntries,
      });
      observer.disconnect();
    }
  }, { once: true });
}

// 統合的なWeb Vitals計測
class WebVitalsCollector {
  constructor(reportCallback) {
    this.report = reportCallback;
    this.metrics = {};

    observeLCP((value) => {
      this.metrics.lcp = value;
      this.report({ name: 'LCP', value, rating: this.rateLCP(value) });
    });

    observeFID((value) => {
      this.metrics.fid = value;
      this.report({ name: 'FID', value, rating: this.rateFID(value) });
    });

    observeINP((value) => {
      this.metrics.inp = value;
      this.report({ name: 'INP', value, rating: this.rateINP(value) });
    });

    observeCLS((result) => {
      this.metrics.cls = result.value;
      this.report({ name: 'CLS', value: result.value, rating: this.rateCLS(result.value) });
    });
  }

  rateLCP(value) {
    if (value <= 2500) return 'good';
    if (value <= 4000) return 'needs-improvement';
    return 'poor';
  }

  rateFID(value) {
    if (value <= 100) return 'good';
    if (value <= 300) return 'needs-improvement';
    return 'poor';
  }

  rateINP(value) {
    if (value <= 200) return 'good';
    if (value <= 500) return 'needs-improvement';
    return 'poor';
  }

  rateCLS(value) {
    if (value <= 0.1) return 'good';
    if (value <= 0.25) return 'needs-improvement';
    return 'poor';
  }
}

// 使用例
const vitals = new WebVitalsCollector((metric) => {
  console.log(`${metric.name}: ${metric.value} (${metric.rating})`);

  // アナリティクスに送信
  fetch('/api/analytics/web-vitals', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      ...metric,
      url: window.location.href,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      connectionType: navigator.connection?.effectiveType,
    }),
    keepalive: true,
  });
});
```

### 5.3 Long Tasks の監視

```javascript
// 50ms以上のタスクを検出
class LongTaskMonitor {
  constructor(options = {}) {
    this.threshold = options.threshold || 50;
    this.tasks = [];
    this.onLongTask = options.onLongTask || (() => {});

    this.observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const taskInfo = {
          duration: entry.duration,
          startTime: entry.startTime,
          name: entry.name,
          // attributionで原因を特定
          attribution: entry.attribution?.map(attr => ({
            containerType: attr.containerType,
            containerName: attr.containerName,
            containerSrc: attr.containerSrc,
          })),
        };

        this.tasks.push(taskInfo);
        this.onLongTask(taskInfo);

        if (entry.duration > 200) {
          console.warn(`Very long task detected: ${entry.duration}ms`, taskInfo);
        }
      }
    });

    this.observer.observe({ type: 'longtask', buffered: true });
  }

  getReport() {
    const totalBlockingTime = this.tasks.reduce(
      (sum, task) => sum + Math.max(0, task.duration - 50),
      0
    );

    return {
      totalTasks: this.tasks.length,
      totalBlockingTime,
      averageDuration: this.tasks.length
        ? this.tasks.reduce((sum, t) => sum + t.duration, 0) / this.tasks.length
        : 0,
      maxDuration: Math.max(0, ...this.tasks.map(t => t.duration)),
      tasks: this.tasks,
    };
  }

  destroy() {
    this.observer.disconnect();
  }
}

// 使用例
const longTaskMonitor = new LongTaskMonitor({
  onLongTask(task) {
    if (task.duration > 100) {
      console.warn(`Long task: ${task.duration.toFixed(0)}ms`);
    }
  },
});

// ページ離脱時にレポートを送信
window.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    const report = longTaskMonitor.getReport();
    navigator.sendBeacon('/api/analytics/long-tasks', JSON.stringify(report));
  }
});
```

### 5.4 リソース計測

```javascript
// リソースの読み込みパフォーマンスを監視
class ResourceMonitor {
  constructor() {
    this.observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const timing = {
          name: entry.name,
          type: entry.initiatorType, // 'fetch', 'xmlhttprequest', 'img', 'script', etc.
          transferSize: entry.transferSize,
          encodedBodySize: entry.encodedBodySize,
          decodedBodySize: entry.decodedBodySize,

          // タイミングの内訳
          dns: entry.domainLookupEnd - entry.domainLookupStart,
          tcp: entry.connectEnd - entry.connectStart,
          tls: entry.secureConnectionStart > 0
            ? entry.connectEnd - entry.secureConnectionStart : 0,
          ttfb: entry.responseStart - entry.requestStart,
          download: entry.responseEnd - entry.responseStart,
          total: entry.duration,

          // キャッシュ判定
          cached: entry.transferSize === 0 && entry.decodedBodySize > 0,
        };

        // 遅いリソースの警告
        if (timing.total > 3000) {
          console.warn(`Slow resource: ${timing.name} (${timing.total.toFixed(0)}ms)`);
        }

        // 大きなリソースの警告
        if (timing.decodedBodySize > 1024 * 1024) {
          console.warn(`Large resource: ${timing.name} (${(timing.decodedBodySize / 1024 / 1024).toFixed(1)}MB)`);
        }
      }
    });

    this.observer.observe({ type: 'resource', buffered: true });
  }

  destroy() {
    this.observer.disconnect();
  }
}
```

---

## 6. React でのObserverフック

### 6.1 useIntersectionObserver

```typescript
import { useEffect, useRef, useState, useCallback } from 'react';

interface UseIntersectionObserverOptions {
  threshold?: number | number[];
  root?: Element | null;
  rootMargin?: string;
  freezeOnceVisible?: boolean;
}

function useIntersectionObserver(
  options: UseIntersectionObserverOptions = {}
) {
  const {
    threshold = 0,
    root = null,
    rootMargin = '0px',
    freezeOnceVisible = false,
  } = options;

  const [entry, setEntry] = useState<IntersectionObserverEntry | null>(null);
  const [node, setNode] = useState<Element | null>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  const frozen = entry?.isIntersecting && freezeOnceVisible;

  // ref callback パターン（DOM要素をステートとして管理）
  const ref = useCallback((node: Element | null) => {
    setNode(node);
  }, []);

  useEffect(() => {
    if (!node || frozen) return;

    observerRef.current = new IntersectionObserver(
      ([entry]) => setEntry(entry),
      { threshold, root, rootMargin }
    );

    observerRef.current.observe(node);

    return () => {
      observerRef.current?.disconnect();
    };
  }, [node, threshold, root, rootMargin, frozen]);

  return {
    ref,
    entry,
    isIntersecting: entry?.isIntersecting ?? false,
    intersectionRatio: entry?.intersectionRatio ?? 0,
  };
}

// 使用例: 画像の遅延読み込み
function LazyImage({ src, alt, ...props }) {
  const { ref, isIntersecting } = useIntersectionObserver({
    rootMargin: '200px',
    freezeOnceVisible: true,
  });

  return (
    <div ref={ref}>
      {isIntersecting ? (
        <img src={src} alt={alt} {...props} />
      ) : (
        <div className="placeholder" style={{ aspectRatio: '16/9' }} />
      )}
    </div>
  );
}

// 使用例: スクロール連動フェードイン
function FadeInSection({ children }) {
  const { ref, isIntersecting } = useIntersectionObserver({
    threshold: 0.1,
    freezeOnceVisible: true,
  });

  return (
    <section
      ref={ref}
      className={`fade-section ${isIntersecting ? 'visible' : ''}`}
    >
      {children}
    </section>
  );
}

// 使用例: 無限スクロール
function InfiniteList({ fetchItems }) {
  const [items, setItems] = useState([]);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(0);

  const { ref, isIntersecting } = useIntersectionObserver({
    rootMargin: '300px',
  });

  useEffect(() => {
    if (!isIntersecting || !hasMore) return;

    fetchItems(page + 1).then(result => {
      setItems(prev => [...prev, ...result.items]);
      setHasMore(result.hasMore);
      setPage(prev => prev + 1);
    });
  }, [isIntersecting, hasMore, page, fetchItems]);

  return (
    <div>
      {items.map(item => (
        <ItemCard key={item.id} item={item} />
      ))}
      {hasMore && <div ref={ref} className="loading-sentinel" />}
    </div>
  );
}
```

### 6.2 useResizeObserver

```typescript
import { useEffect, useRef, useState, useCallback } from 'react';

interface Size {
  width: number;
  height: number;
  inlineSize: number;
  blockSize: number;
}

function useResizeObserver<T extends HTMLElement>(): {
  ref: (node: T | null) => void;
  size: Size | null;
} {
  const [size, setSize] = useState<Size | null>(null);
  const [node, setNode] = useState<T | null>(null);
  const observerRef = useRef<ResizeObserver | null>(null);

  const ref = useCallback((node: T | null) => {
    setNode(node);
  }, []);

  useEffect(() => {
    if (!node) return;

    observerRef.current = new ResizeObserver((entries) => {
      const entry = entries[0];

      if (entry.contentBoxSize) {
        const contentBox = entry.contentBoxSize[0];
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
          inlineSize: contentBox.inlineSize,
          blockSize: contentBox.blockSize,
        });
      } else {
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
          inlineSize: entry.contentRect.width,
          blockSize: entry.contentRect.height,
        });
      }
    });

    observerRef.current.observe(node);

    return () => {
      observerRef.current?.disconnect();
    };
  }, [node]);

  return { ref, size };
}

// 使用例: レスポンシブコンポーネント
function ResponsiveCard({ title, content }) {
  const { ref, size } = useResizeObserver<HTMLDivElement>();

  const layout = size
    ? size.width < 400 ? 'compact' : size.width < 800 ? 'regular' : 'wide'
    : 'regular';

  return (
    <div ref={ref} className={`card card--${layout}`}>
      <h2>{title}</h2>
      <p>{content}</p>
      {size && (
        <span className="debug-size">
          {Math.round(size.width)}x{Math.round(size.height)}
        </span>
      )}
    </div>
  );
}

// 使用例: チャートのリサイズ
function ResponsiveChartWrapper({ data }) {
  const { ref, size } = useResizeObserver<HTMLDivElement>();

  return (
    <div ref={ref} style={{ width: '100%', height: '400px' }}>
      {size && (
        <Chart
          data={data}
          width={size.width}
          height={size.height}
        />
      )}
    </div>
  );
}
```

### 6.3 useMutationObserver

```typescript
import { useEffect, useRef, useCallback } from 'react';

interface UseMutationObserverOptions extends MutationObserverInit {
  callback: MutationCallback;
}

function useMutationObserver<T extends HTMLElement>(
  options: UseMutationObserverOptions
) {
  const [node, setNode] = useState<T | null>(null);
  const callbackRef = useRef(options.callback);
  callbackRef.current = options.callback;

  const ref = useCallback((node: T | null) => {
    setNode(node);
  }, []);

  useEffect(() => {
    if (!node) return;

    const observer = new MutationObserver((...args) => {
      callbackRef.current(...args);
    });

    const { callback, ...observerOptions } = options;
    observer.observe(node, observerOptions);

    return () => observer.disconnect();
  }, [node, options.childList, options.attributes, options.characterData, options.subtree]);

  return ref;
}

// 使用例: DOM変更のデバッグ
function DebugContainer({ children }) {
  const ref = useMutationObserver({
    callback: (mutations) => {
      mutations.forEach(mutation => {
        console.log(`[DOM Change] ${mutation.type}`, mutation);
      });
    },
    childList: true,
    subtree: true,
    attributes: true,
  });

  return <div ref={ref}>{children}</div>;
}
```

---

## 7. scroll vs IntersectionObserver

### 7.1 パフォーマンス比較

```
従来のスクロール監視:
  window.addEventListener('scroll', () => {
    elements.forEach(el => {
      const rect = el.getBoundingClientRect(); // ★ 強制レイアウト（Forced Reflow）
      if (rect.top < window.innerHeight) {
        // 処理
      }
    });
  });

  問題点:
  → scroll イベントは高頻度で発火（1秒に60回以上）
  → getBoundingClientRect() がレイアウトを強制（Layout Thrashing）
  → throttle/debounce が必要だがタイミングが難しい
  → 非アクティブタブでも発火し続ける
  → 要素が多いほど処理が重くなる（O(n)）

IntersectionObserver:
  const observer = new IntersectionObserver(callback, options);
  elements.forEach(el => observer.observe(el));

  利点:
  ✓ ブラウザネイティブの最適化（メインスレッドをブロックしない）
  ✓ Layout Thrashing が発生しない
  ✓ throttle/debounce 不要（ブラウザが最適なタイミングで通知）
  ✓ 非アクティブタブで自動的に停止
  ✓ 要素数に依存しないパフォーマンス
  ✓ rootMarginで先読みが簡単

  制限:
  △ ピクセル単位のスクロール位置は取得できない
  △ スクロール方向の判定には別の仕組みが必要
  △ 連続的なアニメーション（パララックス）には不向き
```

### 7.2 使い分けガイドライン

```javascript
// IntersectionObserver が適している場合:
// - 要素の表示/非表示の検出
// - 遅延読み込み（画像、コンポーネント）
// - 無限スクロール
// - ビューアビリティ計測
// - スクロールスナップのセクション検出

// scroll イベント が適している場合:
// - パララックスエフェクト（連続的なスクロール位置が必要）
// - ヘッダーの縮小/展開（スクロール量に基づく）
// - スクロールプログレスバー
// - スクロール方向の検出

// scroll イベントを使う場合のベストプラクティス
let ticking = false;

function onScroll() {
  if (!ticking) {
    requestAnimationFrame(() => {
      // ここでスクロール位置に基づく処理を行う
      updateParallax(window.scrollY);
      ticking = false;
    });
    ticking = true;
  }
}

window.addEventListener('scroll', onScroll, { passive: true });
// passive: true でスクロールのブロックを防止
```

---

## 8. ベストプラクティスとパフォーマンス最適化

### 8.1 Observer の統合

```javascript
// ★ 同じ設定の Observer は共有する
// 悪い例: 要素ごとにObserverを作成
document.querySelectorAll('.lazy-image').forEach(img => {
  const observer = new IntersectionObserver(/* ... */); // 100個のObserver!
  observer.observe(img);
});

// 良い例: 1つのObserverで複数要素を監視
const observer = new IntersectionObserver(/* ... */); // 1つだけ
document.querySelectorAll('.lazy-image').forEach(img => {
  observer.observe(img); // 同じObserverに追加
});

// ★ 不要になったらunobserve/disconnectする
// メモリリーク防止のため、不要な監視は必ず停止する
observer.unobserve(element); // 個別停止
observer.disconnect();       // 全停止
```

### 8.2 コールバック内の処理を軽量に

```javascript
// ★ Observerのコールバック内で重い処理を避ける
// 悪い例
const observer = new ResizeObserver((entries) => {
  for (const entry of entries) {
    // 重い再描画処理を直接実行
    renderComplexChart(entry.contentRect.width, entry.contentRect.height);
  }
});

// 良い例: requestAnimationFrameでバッチ化
let rafId = null;

const observer = new ResizeObserver((entries) => {
  if (rafId) cancelAnimationFrame(rafId);

  rafId = requestAnimationFrame(() => {
    for (const entry of entries) {
      renderComplexChart(entry.contentRect.width, entry.contentRect.height);
    }
    rafId = null;
  });
});

// 良い例: デバウンスの併用
function debounce(fn, delay) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), delay);
  };
}

const debouncedResize = debounce((width, height) => {
  renderComplexChart(width, height);
}, 150);

const observer = new ResizeObserver((entries) => {
  const { width, height } = entries[0].contentRect;
  debouncedResize(width, height);
});
```

### 8.3 ブラウザサポートとPolyfill

```javascript
// Observer APIのブラウザサポート状況
// IntersectionObserver: Chrome 51+, Firefox 55+, Safari 12.1+, Edge 15+
// ResizeObserver: Chrome 64+, Firefox 69+, Safari 13.1+, Edge 79+
// MutationObserver: Chrome 26+, Firefox 14+, Safari 7+, Edge 12+
// PerformanceObserver: Chrome 52+, Firefox 57+, Safari 11+, Edge 79+

// フィーチャーデテクション
if ('IntersectionObserver' in window) {
  // IntersectionObserverを使用
} else {
  // フォールバック: scroll イベント + getBoundingClientRect
}

if ('ResizeObserver' in window) {
  // ResizeObserverを使用
} else {
  // フォールバック: window.onresize
}

// Polyfill の読み込み（必要な場合のみ）
// npm install intersection-observer
// npm install resize-observer-polyfill
```

---

## まとめ

| Observer | 監視対象 | 主な用途 | パフォーマンス |
|----------|---------|---------|--------------|
| IntersectionObserver | ビューポートとの交差 | 遅延読み込み、無限スクロール、ビューアビリティ | メインスレッド非ブロック |
| ResizeObserver | 要素サイズの変更 | レスポンシブ、チャートリサイズ、仮想スクロール | レイアウト強制なし |
| MutationObserver | DOM変更 | サードパーティ監視、自動初期化、変更検出 | マイクロタスクで実行 |
| PerformanceObserver | パフォーマンスイベント | Web Vitals計測、リソース監視、Long Task検出 | 非同期バッファリング |

### 選択指針

1. **要素の可視性を知りたい** → IntersectionObserver
2. **要素のサイズ変更に反応したい** → ResizeObserver
3. **DOMの変更を検出したい** → MutationObserver
4. **パフォーマンスメトリクスを収集したい** → PerformanceObserver
5. **連続的なスクロール位置が必要** → scroll イベント + requestAnimationFrame
6. **ウィンドウサイズの変更のみ** → matchMedia() またはCSSコンテナクエリ

---

## 次に読むべきガイド

- [[../04-storage-and-caching/00-web-storage.md]] -- Webストレージ（localStorage, sessionStorage, IndexedDB）
- [[../04-storage-and-caching/02-performance-api.md]] -- Performance API 詳解
- [[01-fetch-and-streams.md]] -- Fetch と Streams API

---

## 参考文献

1. MDN Web Docs. "Intersection Observer API." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API
2. MDN Web Docs. "Resize Observer API." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/Resize_Observer_API
3. MDN Web Docs. "MutationObserver." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/MutationObserver
4. MDN Web Docs. "PerformanceObserver." Mozilla, 2024. https://developer.mozilla.org/en-US/docs/Web/API/PerformanceObserver
5. Web.dev. "Lazy loading images with IntersectionObserver." Google, 2024.
6. Web.dev. "Web Vitals." Google, 2024. https://web.dev/vitals/
7. Philip Walton. "Monitoring Cumulative Layout Shift." web.dev, 2023.
8. Web Incubator CG. "Container Queries." W3C, 2024.
9. W3C. "Intersection Observer Specification." W3C, 2024. https://www.w3.org/TR/intersection-observer/
10. W3C. "Resize Observer Specification." W3C, 2024. https://www.w3.org/TR/resize-observer/
