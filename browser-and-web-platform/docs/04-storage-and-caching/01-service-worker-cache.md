# Service Worker とキャッシュ

> Service WorkerはWebアプリにオフライン対応、バックグラウンド同期、Push通知を実現する。キャッシュ戦略の選択とPWA（Progressive Web App）の構築方法を学ぶ。

## この章で学ぶこと

- [ ] Service Workerのライフサイクルを理解する
- [ ] キャッシュ戦略の種類と使い分けを把握する
- [ ] PWAの基本構成を学ぶ

---

## 1. Service Worker ライフサイクル

```
ライフサイクル:

  ① 登録（Register）
     navigator.serviceWorker.register('/sw.js')

  ② インストール（Install）
     → 初回訪問時またはSWファイルが変更された時
     → プリキャッシュ（必須リソースをキャッシュ）

  ③ 待機（Waiting）
     → 古いSWがまだ制御中の場合
     → 全タブが閉じられると次のSWがアクティブに
     → skipWaiting() で即座にアクティブ化可能

  ④ アクティベーション（Activate）
     → 古いキャッシュの削除
     → clients.claim() で既存タブの制御を開始

  ⑤ フェッチ（Fetch）
     → リクエストのインターセプト
     → キャッシュ戦略に基づいてレスポンスを返す

  ⑥ 更新
     → SWファイルが1バイトでも変わると新版をインストール
     → ナビゲーション時に最大24時間ごとにチェック
```

---

## 2. キャッシュ戦略

```javascript
// ① Cache First（キャッシュ優先）
// → 静的ファイル（CSS, JS, 画像）向け
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cached) => {
      return cached || fetch(event.request);
    })
  );
});

// ② Network First（ネットワーク優先）
// → APIレスポンス、動的コンテンツ向け
self.addEventListener('fetch', (event) => {
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        const clone = response.clone();
        caches.open('dynamic').then((cache) => cache.put(event.request, clone));
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});

// ③ Stale While Revalidate
// → 頻繁に更新されるが、古いデータでもOKなもの
self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.open('swr').then((cache) => {
      return cache.match(event.request).then((cached) => {
        const fetched = fetch(event.request).then((response) => {
          cache.put(event.request, response.clone());
          return response;
        });
        return cached || fetched;
      });
    })
  );
});

// 戦略の選択ガイド:
// ┌──────────────────────┬──────────────────────┐
// │ リソース              │ 推奨戦略              │
// ├──────────────────────┼──────────────────────┤
// │ CSS/JS（ハッシュ付き）│ Cache First          │
// │ フォント              │ Cache First          │
// │ 画像                  │ Cache First          │
// │ HTML                  │ Network First        │
// │ APIデータ             │ Network First        │
// │ アバター画像          │ Stale While Revalidate│
// │ ニュースフィード      │ Stale While Revalidate│
// └──────────────────────┴──────────────────────┘
```

---

## 3. Workbox（ライブラリ）

```javascript
// Workbox = Google製のService Workerライブラリ

import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';

// プリキャッシュ（ビルド時に生成されたマニフェスト）
precacheAndRoute(self.__WB_MANIFEST);

// 画像: Cache First + 有効期限
registerRoute(
  ({ request }) => request.destination === 'image',
  new CacheFirst({
    cacheName: 'images',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30日
      }),
    ],
  })
);

// API: Network First
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/'),
  new NetworkFirst({
    cacheName: 'api',
    networkTimeoutSeconds: 3,
  })
);

// フォント: Stale While Revalidate
registerRoute(
  ({ request }) => request.destination === 'font',
  new StaleWhileRevalidate({
    cacheName: 'fonts',
  })
);
```

---

## 4. PWA（Progressive Web App）

```
PWA の構成要素:
  ① Service Worker — オフライン対応
  ② Web App Manifest — インストール可能
  ③ HTTPS — セキュリティ要件

manifest.json:
  {
    "name": "My App",
    "short_name": "MyApp",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#ffffff",
    "theme_color": "#3b82f6",
    "icons": [
      { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
      { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
    ]
  }

HTML:
  <link rel="manifest" href="/manifest.json">
  <meta name="theme-color" content="#3b82f6">

インストールプロンプトの制御:
  let deferredPrompt;
  window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    showInstallButton();
  });

  installButton.addEventListener('click', async () => {
    deferredPrompt.prompt();
    const result = await deferredPrompt.userChoice;
    console.log('Install:', result.outcome);
  });

PWA対応のチェック:
  → Lighthouse > Progressive Web App
  → マニフェストの検証
  → Service Worker の動作確認
  → オフラインでの動作確認
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Service Worker | ページとネットワーク間のプロキシ |
| Cache First | 静的ファイル向け（高速） |
| Network First | APIデータ向け（最新優先） |
| SWR | 古くてもOK（速度+最新のバランス） |
| PWA | SW + Manifest + HTTPS |

---

## 次に読むべきガイド
→ [[02-performance-api.md]] — Performance API

---

## 参考文献
1. web.dev. "Service Workers." Google, 2024.
2. Google. "Workbox." workboxjs.org, 2024.
