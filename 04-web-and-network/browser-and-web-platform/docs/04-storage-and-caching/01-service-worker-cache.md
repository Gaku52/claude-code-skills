# Service Worker とキャッシュ戦略

> Service Worker は Web アプリにオフライン対応、バックグラウンド同期、Push 通知を実現するブラウザ API である。Cache API と組み合わせることで、ネットワーク状況に左右されないレジリエントなユーザー体験を構築できる。本章では、Service Worker のライフサイクルから始め、5 大キャッシュ戦略の詳細、Workbox による実務的な実装、PWA（Progressive Web App）の構築方法、そしてデバッグとトラブルシューティングまでを体系的に解説する。

## この章で学ぶこと

- [ ] Service Worker のライフサイクルと登録・更新の仕組みを正確に理解する
- [ ] Cache API の基本操作（open, put, match, delete）を使いこなす
- [ ] 5 つのキャッシュ戦略（Cache First, Network First, Stale-While-Revalidate, Cache Only, Network Only）の特性と使い分けを把握する
- [ ] Workbox ライブラリを用いた効率的な Service Worker 開発手法を習得する
- [ ] PWA の構成要素と installability（インストール可能性）の要件を学ぶ
- [ ] キャッシュのバージョニングと古いキャッシュの削除戦略を理解する
- [ ] オフラインフォールバックページの実装方法を身につける

---

## 1. Service Worker の基礎概念

### 1.1 Service Worker とは何か

Service Worker は、Web ページとネットワークの間に位置するプログラマブルなプロキシサーバーである。通常のスクリプトとは異なり、以下の特徴を持つ。

1. **独立したスレッドで動作する** -- メインスレッド（UI スレッド）とは別のスレッドで実行されるため、DOM に直接アクセスできない
2. **イベント駆動型である** -- 必要なときだけ起動し、不要になると停止する
3. **HTTPS 必須** -- セキュリティ上の理由から、localhost を除き HTTPS 環境でのみ動作する
4. **ステートレスである** -- 起動のたびに状態がリセットされるため、永続化には IndexedDB や Cache API を使用する

```
+-------------------------------------------------------------------+
|  ブラウザ                                                          |
|                                                                   |
|  +------------------+      +-------------------+                  |
|  |   Web ページ      |      |  Service Worker   |                  |
|  |  (メインスレッド)  | <--> |  (別スレッド)      |                  |
|  |                  |      |                   |                  |
|  | - DOM 操作       |      | - fetch イベント   |                  |
|  | - UI 描画        |      | - push イベント    |                  |
|  | - ユーザー操作    |      | - sync イベント    |                  |
|  +------------------+      +--------+----------+                  |
|                                     |                             |
|                                     v                             |
|                            +--------+----------+                  |
|                            |   Cache Storage   |                  |
|                            |  (Cache API)      |                  |
|                            +--------+----------+                  |
|                                     |                             |
+-------------------------------------|-----------------------------+
                                      |
                                      v
                             +--------+----------+
                             |   ネットワーク     |
                             |  (リモートサーバー) |
                             +-------------------+
```

### 1.2 Service Worker のスコープ

Service Worker には「スコープ」の概念がある。スコープとは、その Service Worker が制御するパスの範囲である。

```javascript
// デフォルトスコープ: Service Worker ファイルが置かれたディレクトリ
// /sw.js を登録 → スコープは /（サイト全体）
navigator.serviceWorker.register('/sw.js');

// /app/sw.js を登録 → スコープは /app/
navigator.serviceWorker.register('/app/sw.js');

// 明示的にスコープを指定する
navigator.serviceWorker.register('/sw.js', {
  scope: '/app/'
});

// 注意: スコープは SW ファイルの配置場所より上位には設定できない
// /app/sw.js を登録して scope: '/' は不可（Service-Worker-Allowed ヘッダーが必要）
```

### 1.3 Service Worker と通常のスクリプトの違い

| 特性 | 通常の JavaScript | Service Worker |
|------|-------------------|----------------|
| 実行スレッド | メインスレッド | ワーカースレッド |
| DOM アクセス | 可能 | 不可 |
| window オブジェクト | 利用可能 | 不可（self を使用） |
| ライフサイクル | ページと同期 | ページとは独立 |
| ネットワークリクエストの傍受 | 不可 | fetch イベントで可能 |
| HTTPS 要件 | なし | 必須（localhost 除く） |
| 永続性 | ページ離脱で終了 | ブラウザが管理 |
| 利用可能な API | すべて | Cache API, Fetch API, IndexedDB, postMessage 等 |
| バックグラウンド処理 | 不可 | Push, Sync イベント対応 |

---

## 2. Service Worker ライフサイクル（詳細）

Service Worker のライフサイクルは、Web 開発者が最もつまずきやすい領域の一つである。各フェーズを正確に理解することが、安定した実装の前提条件となる。

```
+-----------------------------------------------------------+
|              Service Worker ライフサイクル                   |
+-----------------------------------------------------------+
|                                                           |
|  [未登録] --(register())--> [登録中]                       |
|                               |                           |
|                               v                           |
|                          [インストール中]                   |
|                          install イベント                   |
|                               |                           |
|                    +----------+----------+                 |
|                    |                     |                 |
|                    v                     v                 |
|              [待機中]              [インストール失敗]        |
|            (waiting)                  (破棄)               |
|                    |                                       |
|      +-------------+-------------+                        |
|      |                           |                        |
|      v                           v                        |
| [古い SW が制御中]         [skipWaiting()]                  |
| 全タブ閉じるまで待機         即座にアクティブ化               |
|      |                           |                        |
|      +-------------+-------------+                        |
|                    |                                       |
|                    v                                       |
|              [アクティベーション]                            |
|              activate イベント                              |
|                    |                                       |
|                    v                                       |
|              [アクティブ/制御中]                             |
|              fetch, push, sync イベント処理                 |
|                    |                                       |
|                    v                                       |
|              [更新チェック]                                 |
|              24h ごと or register() 呼び出し時              |
|              1 バイトでも差異があれば                        |
|              新しい SW をインストール開始                     |
+-----------------------------------------------------------+
```

### 2.1 登録（Registration）

```javascript
// main.js（ページ側のスクリプト）
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/',
        // updateViaCache: 'none' を指定すると SW ファイル自体のHTTPキャッシュを無視
        updateViaCache: 'none'
      });

      console.log('SW registered:', registration.scope);

      // 更新チェックを手動でトリガーする（オプション）
      registration.update();

      // 登録状態の確認
      if (registration.installing) {
        console.log('Service Worker: インストール中');
      } else if (registration.waiting) {
        console.log('Service Worker: 待機中（更新あり）');
      } else if (registration.active) {
        console.log('Service Worker: アクティブ');
      }
    } catch (error) {
      console.error('SW registration failed:', error);
    }
  });
}
```

### 2.2 インストール（Installation）

```javascript
// sw.js
const CACHE_NAME = 'app-cache-v1';
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/styles/main.css',
  '/scripts/app.js',
  '/images/logo.svg',
  '/offline.html'
];

self.addEventListener('install', (event) => {
  console.log('[SW] Install event');

  // waitUntil() でインストール完了までブラウザに待機を指示する
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Pre-caching resources');
        return cache.addAll(PRECACHE_URLS);
      })
      .then(() => {
        // skipWaiting() を呼ぶと、待機をスキップして即座にアクティブ化する
        // 注意: 既存のタブが古い SW で制御されている状態で新しい SW が
        //       アクティブになるため、互換性に注意が必要
        return self.skipWaiting();
      })
  );
});
```

### 2.3 アクティベーション（Activation）

```javascript
self.addEventListener('activate', (event) => {
  console.log('[SW] Activate event');

  // 古いキャッシュを削除する
  const cacheWhitelist = [CACHE_NAME];

  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (!cacheWhitelist.includes(cacheName)) {
              console.log('[SW] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            }
          })
        );
      })
      .then(() => {
        // clients.claim() で、現在開いている全タブの制御を即座に開始する
        // これがないと、新しい SW は次回のナビゲーションまで制御を開始しない
        return self.clients.claim();
      })
  );
});
```

### 2.4 更新の仕組み

Service Worker の更新は以下のタイミングで自動チェックされる。

1. ユーザーがスコープ内のページへナビゲーションしたとき
2. `push` や `sync` などの機能イベントが発火したとき（前回のチェックから24時間以上経過している場合）
3. `registration.update()` を明示的に呼び出したとき

```javascript
// 更新の手動チェックと通知
async function checkForUpdates() {
  const registration = await navigator.serviceWorker.getRegistration();
  if (!registration) return;

  // 更新チェック
  await registration.update();

  // 待機中の SW があるか確認
  if (registration.waiting) {
    showUpdateNotification(registration.waiting);
  }

  // 新しい SW のインストールを監視
  registration.addEventListener('updatefound', () => {
    const newWorker = registration.installing;
    newWorker.addEventListener('statechange', () => {
      if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
        // 新しいバージョンが利用可能
        showUpdateNotification(newWorker);
      }
    });
  });
}

function showUpdateNotification(worker) {
  // UI で「更新あり」を表示し、ユーザーがクリックしたら SW に通知
  const updateBanner = document.getElementById('update-banner');
  updateBanner.style.display = 'block';
  updateBanner.querySelector('button').addEventListener('click', () => {
    worker.postMessage({ type: 'SKIP_WAITING' });
  });
}

// SW 側: skipWaiting メッセージを受け取る
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

// ページ側: controller が変わったらリロード
navigator.serviceWorker.addEventListener('controllerchange', () => {
  window.location.reload();
});
```

---

## 3. Cache API の基本操作

Cache API は Service Worker だけでなく、通常の Window コンテキストからも利用可能な非同期 API である。HTTP のリクエスト/レスポンスペアをキーバリュー形式で保存する。

### 3.1 基本メソッド

```javascript
// --- caches.open(cacheName) ---
// 指定した名前のキャッシュを開く（なければ作成する）
const cache = await caches.open('my-cache-v1');

// --- cache.add(request) ---
// リクエストを取得し、レスポンスをキャッシュに保存する
await cache.add('/styles/main.css');
// 内部的には以下と同等:
// const response = await fetch('/styles/main.css');
// await cache.put('/styles/main.css', response);

// --- cache.addAll(requests) ---
// 複数のリクエストを一括でキャッシュに追加する
// 1 つでも失敗するとすべて失敗する（アトミック操作）
await cache.addAll([
  '/',
  '/index.html',
  '/styles/main.css',
  '/scripts/app.js'
]);

// --- cache.put(request, response) ---
// リクエストとレスポンスのペアを直接キャッシュに保存する
const response = await fetch('/api/data');
await cache.put('/api/data', response.clone());
// 注意: response.clone() を使う。Response は一度しか読み取れないため

// --- cache.match(request, options) ---
// キャッシュからリクエストに一致するレスポンスを検索する
const cachedResponse = await cache.match('/styles/main.css');
if (cachedResponse) {
  console.log('Cache hit!');
}

// オプション: ignoreSearch でクエリパラメータを無視してマッチング
const result = await cache.match('/api/users', { ignoreSearch: true });
// /api/users?page=1 なども一致する

// --- cache.delete(request) ---
// 特定のキャッシュエントリを削除する
const deleted = await cache.delete('/old-resource.js');
console.log('Deleted:', deleted); // true or false

// --- cache.keys() ---
// キャッシュ内のすべてのリクエスト（キー）を取得する
const requests = await cache.keys();
requests.forEach((request) => {
  console.log('Cached:', request.url);
});

// --- caches.keys() ---
// すべてのキャッシュ名を取得する
const cacheNames = await caches.keys();
console.log('Available caches:', cacheNames);

// --- caches.delete(cacheName) ---
// キャッシュ全体を削除する
await caches.delete('old-cache-v1');

// --- caches.match(request) ---
// すべてのキャッシュを横断して検索する（最初に一致したものを返す）
const anyMatch = await caches.match('/styles/main.css');
```

### 3.2 Cache API の制約と注意点

| 項目 | 詳細 |
|------|------|
| ストレージ上限 | ブラウザ・デバイスにより異なる。Chrome ではディスク容量の最大 80% まで（オリジン単位で 60% まで） |
| 格納対象 | HTTP リクエスト/レスポンスペアのみ（任意のデータは IndexedDB を使う） |
| キーの一致 | URL ベースの完全一致（デフォルト）。Vary ヘッダーも考慮される |
| CORS レスポンス | opaque レスポンス（no-cors）もキャッシュ可能だがステータスは 0 になる |
| レスポンスの消費 | Response は一度しか body を読み取れない。複数回使う場合は clone() が必要 |
| 永続性 | 明示的に削除するか、ブラウザのストレージ圧迫時に evict される可能性がある |

---

## 4. 5 大キャッシュ戦略の詳細

キャッシュ戦略とは、Service Worker が fetch イベントを受け取った際に「キャッシュとネットワークをどのように組み合わせてレスポンスを返すか」を決定するパターンである。

### 4.1 Cache First（キャッシュ優先）

キャッシュにあればキャッシュから返し、なければネットワークに取りに行く。

```
リクエスト --> [Cache に存在？]
                |          |
               YES         NO
                |          |
                v          v
        [Cache から返す]  [Network へ]
                              |
                              v
                        [Cache に保存]
                              |
                              v
                        [レスポンス返却]
```

```javascript
// Cache First 戦略の実装
self.addEventListener('fetch', (event) => {
  // 対象: ハッシュ付きの静的アセット
  if (event.request.url.match(/\.(css|js|woff2?|png|jpg|svg)(\?.*)?$/)) {
    event.respondWith(
      caches.match(event.request)
        .then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          return fetch(event.request).then((networkResponse) => {
            // 正常なレスポンスのみキャッシュする
            if (networkResponse.ok) {
              const responseClone = networkResponse.clone();
              caches.open('static-assets-v1').then((cache) => {
                cache.put(event.request, responseClone);
              });
            }
            return networkResponse;
          });
        })
    );
  }
});
```

**適用対象**: ビルド済み CSS/JS（ファイル名にハッシュ含む）、Web フォント、ロゴ画像
**メリット**: 高速、オフライン対応、ネットワーク負荷軽減
**デメリット**: キャッシュが古い場合に更新が反映されない

### 4.2 Network First（ネットワーク優先）

まずネットワークに取りに行き、失敗した場合にキャッシュから返す。

```
リクエスト --> [Network へ]
                |        |
              成功       失敗
                |        |
                v        v
        [Cache に保存] [Cache に存在？]
                |        |        |
                v       YES       NO
        [レスポンス返却]  |        |
                        v        v
                  [Cache から]  [エラー or
                   [返す]     オフラインページ]
```

```javascript
// Network First 戦略の実装（タイムアウト付き）
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('/api/')) {
    event.respondWith(
      // タイムアウトを設けてネットワークリクエストを試行
      promiseWithTimeout(fetch(event.request), 3000)
        .then((networkResponse) => {
          // 成功: キャッシュに保存して返す
          const responseClone = networkResponse.clone();
          caches.open('api-cache-v1').then((cache) => {
            cache.put(event.request, responseClone);
          });
          return networkResponse;
        })
        .catch(async () => {
          // 失敗: キャッシュから返す
          const cachedResponse = await caches.match(event.request);
          if (cachedResponse) {
            return cachedResponse;
          }
          // キャッシュもない場合: エラーレスポンスを返す
          return new Response(
            JSON.stringify({ error: 'Offline', cached: false }),
            {
              status: 503,
              headers: { 'Content-Type': 'application/json' }
            }
          );
        })
    );
  }
});

// タイムアウト付き Promise のユーティリティ
function promiseWithTimeout(promise, ms) {
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('Timeout')), ms);
  });
  return Promise.race([promise, timeout]);
}
```

**適用対象**: API レスポンス、HTML ページ、動的コンテンツ
**メリット**: 常に最新データを優先する
**デメリット**: オフライン時やネットワーク遅延時に初回表示が遅い

### 4.3 Stale-While-Revalidate（SWR）

キャッシュから即座にレスポンスを返しつつ、バックグラウンドでネットワークから最新版を取得してキャッシュを更新する。

```
リクエスト --> [Cache に存在？]
                |          |
               YES         NO
                |          |
                v          |
        [Cache から即座に返す] |
                |          |
                v          v
        [バックグラウンドで   [Network へ]
         Network へ]            |
                |               v
                v         [Cache に保存]
        [Cache を更新]          |
        (次回リクエスト時に      v
         最新版が返る)    [レスポンス返却]
```

```javascript
// Stale-While-Revalidate 戦略の実装
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('/content/')) {
    event.respondWith(
      caches.open('content-cache-v1').then((cache) => {
        return cache.match(event.request).then((cachedResponse) => {
          // バックグラウンドでネットワークから最新版を取得
          const fetchPromise = fetch(event.request)
            .then((networkResponse) => {
              // キャッシュを更新
              cache.put(event.request, networkResponse.clone());
              return networkResponse;
            })
            .catch(() => {
              // ネットワーク失敗時は何もしない（キャッシュが既に返されている）
              console.log('[SW] Background fetch failed, using cached version');
            });

          // キャッシュがあればすぐに返す、なければネットワークを待つ
          return cachedResponse || fetchPromise;
        });
      })
    );
  }
});
```

**適用対象**: ユーザーアバター、ニュースフィード、ソーシャルメディアのタイムライン、更新頻度が中程度のコンテンツ
**メリット**: 高速な初回表示 + バックグラウンドで最新化
**デメリット**: 初回表示が1回分古い可能性がある

### 4.4 Cache Only

キャッシュからのみレスポンスを返す。ネットワークリクエストは一切行わない。

```javascript
// Cache Only 戦略の実装
self.addEventListener('fetch', (event) => {
  // プリキャッシュされたリソースに限定して使用
  if (event.request.url.includes('/static/')) {
    event.respondWith(
      caches.match(event.request).then((cachedResponse) => {
        if (cachedResponse) {
          return cachedResponse;
        }
        // キャッシュにない場合は 404 を返す
        return new Response('Resource not found in cache', {
          status: 404,
          statusText: 'Not Found'
        });
      })
    );
  }
});
```

**適用対象**: install イベントでプリキャッシュした静的リソース
**メリット**: 完全にオフライン対応、ネットワーク通信ゼロ
**デメリット**: キャッシュにない場合は失敗する

### 4.5 Network Only

ネットワークからのみレスポンスを取得する。キャッシュは使用しない。

```javascript
// Network Only 戦略の実装
self.addEventListener('fetch', (event) => {
  // キャッシュ不要な動的リクエスト
  if (event.request.url.includes('/api/auth/') ||
      event.request.method !== 'GET') {
    event.respondWith(
      fetch(event.request).catch(() => {
        return new Response(
          JSON.stringify({ error: 'Network required' }),
          {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
          }
        );
      })
    );
  }
});
```

**適用対象**: 認証 API、決済処理、リアルタイムデータ
**メリット**: 常に最新のデータを取得
**デメリット**: オフライン時に完全に動作しない

### 4.6 戦略の選択ガイド（総合比較表）

| 戦略 | 速度 | 鮮度 | オフライン対応 | 適用対象 |
|------|------|------|---------------|---------|
| Cache First | 最速 | 低い（キャッシュ依存） | 完全対応 | ハッシュ付き CSS/JS、フォント、ロゴ |
| Network First | 遅い（ネットワーク依存） | 最新 | キャッシュがあれば対応 | API データ、HTML ページ |
| Stale-While-Revalidate | 速い | 1 回遅れ | キャッシュがあれば対応 | アバター、フィード、中頻度更新コンテンツ |
| Cache Only | 最速 | 固定 | 完全対応 | プリキャッシュされた静的リソース |
| Network Only | 遅い | 最新 | 非対応 | 認証 API、決済、非冪等リクエスト |

---

## 5. 実践的な Service Worker の実装

### 5.1 統合的な fetch ハンドラー

実際のアプリケーションでは、リクエストの種類に応じて戦略を切り替える。

```javascript
// sw.js -- 統合的な Service Worker 実装
const STATIC_CACHE = 'static-v2';
const DYNAMIC_CACHE = 'dynamic-v1';
const API_CACHE = 'api-v1';

const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/offline.html',
  '/styles/main.css',
  '/scripts/app.js',
  '/images/logo.svg'
];

// ==========================================
// インストール: プリキャッシュ
// ==========================================
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

// ==========================================
// アクティベーション: 古いキャッシュの削除
// ==========================================
self.addEventListener('activate', (event) => {
  const validCaches = [STATIC_CACHE, DYNAMIC_CACHE, API_CACHE];
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys
          .filter((key) => !validCaches.includes(key))
          .map((key) => caches.delete(key))
      );
    }).then(() => self.clients.claim())
  );
});

// ==========================================
// フェッチ: リクエスト種別ごとに戦略を適用
// ==========================================
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // 同一オリジンのリクエストのみ処理する
  if (url.origin !== location.origin) {
    return;
  }

  // POST, PUT, DELETE はネットワークに直接転送
  if (request.method !== 'GET') {
    return;
  }

  // API リクエスト: Network First
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request, API_CACHE, 3000));
    return;
  }

  // HTML ナビゲーション: Network First（オフラインフォールバック付き）
  if (request.mode === 'navigate') {
    event.respondWith(
      networkFirst(request, DYNAMIC_CACHE, 3000)
        .catch(() => caches.match('/offline.html'))
    );
    return;
  }

  // 静的アセット: Cache First
  if (request.destination === 'style' ||
      request.destination === 'script' ||
      request.destination === 'font' ||
      request.destination === 'image') {
    event.respondWith(cacheFirst(request, STATIC_CACHE));
    return;
  }

  // その他: Stale-While-Revalidate
  event.respondWith(staleWhileRevalidate(request, DYNAMIC_CACHE));
});

// ==========================================
// 戦略関数
// ==========================================
async function cacheFirst(request, cacheName) {
  const cached = await caches.match(request);
  if (cached) return cached;

  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    return new Response('Resource not available', { status: 404 });
  }
}

async function networkFirst(request, cacheName, timeoutMs) {
  try {
    const response = await promiseWithTimeout(fetch(request), timeoutMs);
    if (response.ok) {
      const cache = await caches.open(cacheName);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) return cached;
    throw error;
  }
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => null);

  return cached || fetchPromise;
}

function promiseWithTimeout(promise, ms) {
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error('Timeout')), ms);
  });
  return Promise.race([promise, timeout]);
}
```

### 5.2 オフラインフォールバックページ

```html
<!-- offline.html -->
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>オフライン</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      margin: 0;
      background: #f5f5f5;
      color: #333;
    }
    .container {
      text-align: center;
      padding: 2rem;
    }
    .icon { font-size: 4rem; margin-bottom: 1rem; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    p { color: #666; margin-bottom: 1.5rem; }
    button {
      padding: 0.75rem 2rem;
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 0.5rem;
      cursor: pointer;
      font-size: 1rem;
    }
    button:hover { background: #2563eb; }
  </style>
</head>
<body>
  <div class="container">
    <div class="icon">&#128268;</div>
    <h1>接続がありません</h1>
    <p>インターネットに接続されていないようです。<br>接続を確認してから再試行してください。</p>
    <button onclick="window.location.reload()">再試行</button>
  </div>
</body>
</html>
```

---

## 6. Workbox による Service Worker 開発

Workbox は Google が開発した Service Worker のライブラリ群である。キャッシュ戦略の実装、プリキャッシュマニフェストの生成、ルーティングなどの機能を提供し、Service Worker 開発の生産性と品質を大幅に向上させる。

### 6.1 Workbox のアーキテクチャ

```
+-----------------------------------------------------------+
|  Workbox モジュール構成                                     |
+-----------------------------------------------------------+
|                                                           |
|  workbox-routing          workbox-strategies              |
|  +------------------+    +------------------+             |
|  | registerRoute()  |--->| CacheFirst       |             |
|  | NavigationRoute  |    | NetworkFirst     |             |
|  | RegExpRoute      |    | StaleWhileRevali.|             |
|  +------------------+    | NetworkOnly      |             |
|                          | CacheOnly        |             |
|                          +--------+---------+             |
|                                   |                       |
|  workbox-precaching        workbox-expiration             |
|  +------------------+    +------------------+             |
|  | precacheAndRoute()|    | ExpirationPlugin |             |
|  | __WB_MANIFEST    |    | maxEntries       |             |
|  +------------------+    | maxAgeSeconds    |             |
|                          +------------------+             |
|                                                           |
|  workbox-cacheable-response   workbox-background-sync     |
|  +------------------+         +------------------+        |
|  | CacheableResp.   |         | BackgroundSync   |        |
|  | Plugin           |         | Plugin           |        |
|  | statuses: [0,200]|         | Queue            |        |
|  +------------------+         +------------------+        |
|                                                           |
|  workbox-window (ページ側)                                 |
|  +------------------+                                     |
|  | Workbox class     |                                     |
|  | register()        |                                     |
|  | messageSkipWaiting|                                     |
|  +------------------+                                     |
+-----------------------------------------------------------+
```

### 6.2 Workbox の導入方法

Workbox は以下の 3 つの方法で導入できる。

```javascript
// 方法 1: CDN から importScripts で読み込み（プロトタイプ向け）
importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.0.0/workbox-sw.js');

// 方法 2: npm パッケージとしてインストール（推奨）
// npm install workbox-precaching workbox-routing workbox-strategies
//            workbox-expiration workbox-cacheable-response

// 方法 3: Workbox CLI でプロジェクトを生成
// npx workbox-cli wizard
```

### 6.3 Workbox を使った Service Worker 実装

```javascript
// sw.js -- Workbox ベースの Service Worker

import { precacheAndRoute, cleanupOutdatedCaches } from 'workbox-precaching';
import { registerRoute, NavigationRoute } from 'workbox-routing';
import {
  CacheFirst,
  NetworkFirst,
  StaleWhileRevalidate
} from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';
import { BackgroundSyncPlugin } from 'workbox-background-sync';

// ==========================================
// プリキャッシュ: ビルド時に生成されたマニフェスト
// ==========================================
// __WB_MANIFEST はビルドツール（workbox-webpack-plugin,
// workbox-build, @vite-plugin/pwa 等）によって自動的に
// プリキャッシュすべきファイルリストに置換される
precacheAndRoute(self.__WB_MANIFEST);

// 古いプリキャッシュの自動クリーンアップ
cleanupOutdatedCaches();

// ==========================================
// 画像: Cache First + 有効期限 + サイズ制限
// ==========================================
registerRoute(
  ({ request }) => request.destination === 'image',
  new CacheFirst({
    cacheName: 'images-cache',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200]  // opaque レスポンスも許可
      }),
      new ExpirationPlugin({
        maxEntries: 100,           // 最大 100 エントリ
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30 日
        purgeOnQuotaError: true    // ストレージ不足時に自動削除
      })
    ]
  })
);

// ==========================================
// フォント: Cache First（長期キャッシュ）
// ==========================================
registerRoute(
  ({ request }) => request.destination === 'font',
  new CacheFirst({
    cacheName: 'fonts-cache',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200]
      }),
      new ExpirationPlugin({
        maxEntries: 30,
        maxAgeSeconds: 365 * 24 * 60 * 60 // 1 年
      })
    ]
  })
);

// ==========================================
// CSS / JS: Stale-While-Revalidate
// ==========================================
registerRoute(
  ({ request }) =>
    request.destination === 'style' ||
    request.destination === 'script',
  new StaleWhileRevalidate({
    cacheName: 'static-resources',
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200]
      })
    ]
  })
);

// ==========================================
// API: Network First + タイムアウト
// ==========================================
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/'),
  new NetworkFirst({
    cacheName: 'api-cache',
    networkTimeoutSeconds: 3,
    plugins: [
      new CacheableResponsePlugin({
        statuses: [0, 200]
      }),
      new ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 5 * 60 // 5 分
      })
    ]
  })
);

// ==========================================
// HTML ナビゲーション: Network First + オフラインフォールバック
// ==========================================
const navigationHandler = new NetworkFirst({
  cacheName: 'pages-cache',
  networkTimeoutSeconds: 3,
  plugins: [
    new CacheableResponsePlugin({
      statuses: [0, 200]
    })
  ]
});

// NavigationRoute は mode: 'navigate' のリクエストのみに一致する
const navigationRoute = new NavigationRoute(navigationHandler, {
  // 除外パス: API や静的ファイルのリクエストをスキップ
  denylist: [
    /\/api\//,
    /\.(js|css|png|jpg|svg|woff2?)$/
  ]
});

registerRoute(navigationRoute);

// ==========================================
// バックグラウンド同期: オフライン時のフォーム送信
// ==========================================
const bgSyncPlugin = new BackgroundSyncPlugin('form-submissions', {
  maxRetentionTime: 24 * 60 // 24 時間（分単位）
});

registerRoute(
  ({ url }) => url.pathname === '/api/submit',
  new NetworkFirst({
    plugins: [bgSyncPlugin]
  }),
  'POST'
);
```

### 6.4 ページ側での Workbox Window の利用

```javascript
// main.js -- ページ側のスクリプト
import { Workbox } from 'workbox-window';

if ('serviceWorker' in navigator) {
  const wb = new Workbox('/sw.js');

  // 新しい SW がインストールされ、待機状態になったとき
  wb.addEventListener('waiting', (event) => {
    // ユーザーに更新を通知するUI を表示
    const shouldUpdate = confirm(
      '新しいバージョンが利用可能です。更新しますか？'
    );

    if (shouldUpdate) {
      // 待機中の SW に skipWaiting を指示
      wb.messageSkipWaiting();
    }
  });

  // controller が変わった（新しい SW がアクティブになった）
  wb.addEventListener('controlling', () => {
    // ページをリロードして新しい SW を適用
    window.location.reload();
  });

  // SW が初めてアクティブになったとき
  wb.addEventListener('activated', (event) => {
    if (!event.isUpdate) {
      // 初回インストール: キャッシュが完了した旨を通知
      console.log('Service Worker がインストールされました');
    }
  });

  wb.register();
}
```

### 6.5 ビルドツールとの統合

```javascript
// vite.config.js -- Vite + VitePWA プラグイン
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: 'prompt', // 'autoUpdate' or 'prompt'
      includeAssets: ['favicon.ico', 'apple-touch-icon.png'],
      manifest: {
        name: 'My Application',
        short_name: 'MyApp',
        description: 'A progressive web application',
        theme_color: '#3b82f6',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      },
      workbox: {
        // プリキャッシュの glob パターン
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        // ランタイムキャッシュの設定
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.example\.com\/.*/i,
            handler: 'NetworkFirst',
            options: {
              cacheName: 'api-cache',
              networkTimeoutSeconds: 3,
              expiration: {
                maxEntries: 50,
                maxAgeSeconds: 300
              },
              cacheableResponse: {
                statuses: [0, 200]
              }
            }
          },
          {
            urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/i,
            handler: 'CacheFirst',
            options: {
              cacheName: 'google-fonts-cache',
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365
              },
              cacheableResponse: {
                statuses: [0, 200]
              }
            }
          }
        ]
      }
    })
  ]
});
```

```javascript
// webpack.config.js -- Webpack + InjectManifest
const { InjectManifest } = require('workbox-webpack-plugin');

module.exports = {
  // ... webpack の設定
  plugins: [
    new InjectManifest({
      swSrc: './src/sw.js',        // ソースの SW ファイル
      swDest: 'sw.js',             // 出力先
      maximumFileSizeToCacheInBytes: 5 * 1024 * 1024, // 5MB
      include: [/\.html$/, /\.js$/, /\.css$/, /\.woff2$/],
      exclude: [/\.map$/, /manifest\.json$/]
    })
  ]
};
```

---

## 7. PWA（Progressive Web App）の構築

### 7.1 PWA の要件と構成要素

PWA は以下の 3 つの要件を満たすことで、ブラウザからインストール可能な Web アプリとなる。

```
+-----------------------------------------------------------+
|  PWA の構成要素                                             |
+-----------------------------------------------------------+
|                                                           |
|  1. Service Worker                                        |
|     - fetch イベントハンドラーを持つ                         |
|     - オフライン時にレスポンスを返せる                       |
|                                                           |
|  2. Web App Manifest (manifest.json)                      |
|     - name (または short_name)                             |
|     - icons (192x192 以上)                                |
|     - start_url                                           |
|     - display (standalone, fullscreen, minimal-ui)         |
|                                                           |
|  3. HTTPS                                                 |
|     - 全ページが HTTPS で配信されている                      |
|     - localhost は開発用に例外                               |
|                                                           |
|  +----------------------------------------------------+   |
|  | インストール可能になるための追加条件 (Chrome)          |   |
|  | - beforeinstallprompt イベントが発火する              |   |
|  | - ユーザーが 30 秒以上サイトを閲覧している             |   |
|  | - Service Worker に fetch ハンドラーがある             |   |
|  +----------------------------------------------------+   |
+-----------------------------------------------------------+
```

### 7.2 Web App Manifest の詳細

```json
{
  "name": "タスク管理アプリケーション",
  "short_name": "タスク管理",
  "description": "チームのタスクを効率的に管理するプログレッシブ Web アプリ",
  "start_url": "/?source=pwa",
  "scope": "/",
  "display": "standalone",
  "orientation": "any",
  "background_color": "#ffffff",
  "theme_color": "#3b82f6",
  "lang": "ja",
  "dir": "ltr",
  "categories": ["productivity", "utilities"],
  "icons": [
    {
      "src": "/icons/icon-72x72.png",
      "sizes": "72x72",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-96x96.png",
      "sizes": "96x96",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-128x128.png",
      "sizes": "128x128",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-144x144.png",
      "sizes": "144x144",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-152x152.png",
      "sizes": "152x152",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-192x192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-384x384.png",
      "sizes": "384x384",
      "type": "image/png"
    },
    {
      "src": "/icons/icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any"
    },
    {
      "src": "/icons/maskable-icon-512x512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "maskable"
    }
  ],
  "screenshots": [
    {
      "src": "/screenshots/desktop.png",
      "sizes": "1280x720",
      "type": "image/png",
      "form_factor": "wide",
      "label": "デスクトップ版のホーム画面"
    },
    {
      "src": "/screenshots/mobile.png",
      "sizes": "390x844",
      "type": "image/png",
      "form_factor": "narrow",
      "label": "モバイル版のホーム画面"
    }
  ],
  "shortcuts": [
    {
      "name": "新しいタスク",
      "short_name": "新規",
      "description": "新しいタスクを作成",
      "url": "/tasks/new?source=shortcut",
      "icons": [
        {
          "src": "/icons/shortcut-new.png",
          "sizes": "96x96"
        }
      ]
    },
    {
      "name": "今日のタスク",
      "short_name": "今日",
      "url": "/tasks/today?source=shortcut"
    }
  ],
  "share_target": {
    "action": "/share-target",
    "method": "POST",
    "enctype": "multipart/form-data",
    "params": {
      "title": "title",
      "text": "text",
      "url": "url",
      "files": [
        {
          "name": "media",
          "accept": ["image/*", "video/*"]
        }
      ]
    }
  },
  "protocol_handlers": [
    {
      "protocol": "web+task",
      "url": "/tasks/%s"
    }
  ]
}
```

### 7.3 display モードの比較

| display モード | ブラウザ UI | アドレスバー | ステータスバー | 用途 |
|---------------|------------|-------------|--------------|------|
| fullscreen | 非表示 | 非表示 | 非表示 | ゲーム、没入型コンテンツ |
| standalone | 非表示 | 非表示 | 表示 | 一般的なアプリ（推奨） |
| minimal-ui | 最小限 | 表示（縮小） | 表示 | ナビゲーション機能が必要なアプリ |
| browser | 完全表示 | 表示 | 表示 | 通常の Web サイト |

### 7.4 インストールプロンプトの制御

```javascript
// install-prompt.js -- インストールプロンプトの制御
class PWAInstallManager {
  constructor() {
    this.deferredPrompt = null;
    this.isInstalled = false;
    this.setupEventListeners();
  }

  setupEventListeners() {
    // beforeinstallprompt: ブラウザがインストール可能と判断したとき
    window.addEventListener('beforeinstallprompt', (event) => {
      // デフォルトのプロンプトを抑制
      event.preventDefault();
      this.deferredPrompt = event;

      // カスタムの「インストール」ボタンを表示
      this.showInstallButton();
    });

    // appinstalled: インストールが完了したとき
    window.addEventListener('appinstalled', () => {
      this.isInstalled = true;
      this.deferredPrompt = null;
      this.hideInstallButton();

      // アナリティクスにインストールを記録
      this.trackInstallation();
    });

    // display-mode の変化を監視（standalone で開かれたか）
    window.matchMedia('(display-mode: standalone)').addEventListener('change', (e) => {
      if (e.matches) {
        console.log('PWA がスタンドアロンモードで開かれました');
      }
    });
  }

  showInstallButton() {
    const btn = document.getElementById('pwa-install-btn');
    if (btn) {
      btn.style.display = 'block';
      btn.addEventListener('click', () => this.promptInstall());
    }
  }

  hideInstallButton() {
    const btn = document.getElementById('pwa-install-btn');
    if (btn) {
      btn.style.display = 'none';
    }
  }

  async promptInstall() {
    if (!this.deferredPrompt) return;

    // インストールプロンプトを表示
    this.deferredPrompt.prompt();

    // ユーザーの選択を待つ
    const { outcome } = await this.deferredPrompt.userChoice;
    console.log('Install prompt outcome:', outcome);

    if (outcome === 'accepted') {
      console.log('ユーザーがインストールを承認');
    } else {
      console.log('ユーザーがインストールを拒否');
    }

    this.deferredPrompt = null;
  }

  trackInstallation() {
    // Google Analytics 等にインストールイベントを送信
    if (typeof gtag === 'function') {
      gtag('event', 'pwa_install', {
        event_category: 'PWA',
        event_label: 'install'
      });
    }
  }

  // PWA として起動されたかを判定
  static isRunningAsPWA() {
    return (
      window.matchMedia('(display-mode: standalone)').matches ||
      window.navigator.standalone === true || // iOS Safari
      document.referrer.includes('android-app://') // TWA
    );
  }
}

// 初期化
const pwaInstaller = new PWAInstallManager();
```

### 7.5 iOS Safari での PWA 対応

iOS Safari は Web App Manifest の一部機能にしか対応しておらず、独自の meta タグが必要になる場合がある。

```html
<head>
  <!-- 標準の manifest -->
  <link rel="manifest" href="/manifest.json">

  <!-- iOS Safari 向けの設定 -->
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
  <meta name="apple-mobile-web-app-title" content="タスク管理">

  <!-- iOS 向けアイコン -->
  <link rel="apple-touch-icon" href="/icons/apple-touch-icon-180x180.png">

  <!-- iOS 向けスプラッシュスクリーン -->
  <link rel="apple-touch-startup-image"
        media="(device-width: 390px) and (device-height: 844px) and (-webkit-device-pixel-ratio: 3)"
        href="/splash/iPhone_13_portrait.png">
  <link rel="apple-touch-startup-image"
        media="(device-width: 428px) and (device-height: 926px) and (-webkit-device-pixel-ratio: 3)"
        href="/splash/iPhone_13_Pro_Max_portrait.png">

  <!-- テーマカラー -->
  <meta name="theme-color" content="#3b82f6">

  <!-- Windows 向けタイル設定 -->
  <meta name="msapplication-TileColor" content="#3b82f6">
  <meta name="msapplication-TileImage" content="/icons/mstile-144x144.png">
</head>
```

---

## 8. 高度なキャッシュパターン

### 8.1 キャッシュのバージョニング戦略

キャッシュのバージョニングは、アプリケーションの更新時に古いキャッシュを適切に管理するための重要な仕組みである。

```javascript
// バージョン管理の方式

// 方式 1: 単一バージョン番号（シンプルだが粒度が粗い）
const CACHE_VERSION = 'v3';
const CACHE_NAME = `app-cache-${CACHE_VERSION}`;

// 方式 2: リソース種別ごとにバージョンを分離
const CACHES = {
  static: 'static-v5',
  images: 'images-v2',
  api: 'api-v1',
  pages: 'pages-v3'
};

// 方式 3: ビルドハッシュを使用（ビルドツールと連携）
const BUILD_HASH = '8f4a2c1e'; // ビルド時に注入される
const CACHES_BY_HASH = {
  precache: `precache-${BUILD_HASH}`,
  runtime: 'runtime-v1'
};

// activate イベントで古いキャッシュを削除
self.addEventListener('activate', (event) => {
  const validCacheNames = Object.values(CACHES);
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          // 有効なキャッシュ名に含まれないものを削除
          if (!validCacheNames.includes(key)) {
            console.log('[SW] Removing old cache:', key);
            return caches.delete(key);
          }
        })
      );
    })
  );
});
```

### 8.2 Range Request への対応

動画や音声ファイルの再生では、Range Request（部分的なコンテンツ取得）への対応が必要になる。

```javascript
// Range Request に対応した Cache First 戦略
self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (request.destination === 'video' || request.destination === 'audio') {
    event.respondWith(handleRangeRequest(request));
  }
});

async function handleRangeRequest(request) {
  const cache = await caches.open('media-cache');
  const cachedResponse = await cache.match(request.url, { ignoreSearch: true });

  if (!cachedResponse) {
    // キャッシュにない: ネットワークから取得してキャッシュ
    try {
      const networkResponse = await fetch(request);
      // 全体をキャッシュに保存（Range なしで）
      const fullRequest = new Request(request.url);
      cache.put(fullRequest, networkResponse.clone());
      return networkResponse;
    } catch (error) {
      return new Response('Media not available offline', { status: 503 });
    }
  }

  // キャッシュにある場合: Range ヘッダーを処理
  const rangeHeader = request.headers.get('Range');
  if (!rangeHeader) {
    return cachedResponse;
  }

  const arrayBuffer = await cachedResponse.arrayBuffer();
  const bytes = /^bytes=(\d+)-(\d*)$/i.exec(rangeHeader);

  if (!bytes) {
    return new Response(arrayBuffer, {
      status: 200,
      headers: cachedResponse.headers
    });
  }

  const start = Number(bytes[1]);
  const end = bytes[2] ? Number(bytes[2]) : arrayBuffer.byteLength - 1;
  const slicedBuffer = arrayBuffer.slice(start, end + 1);

  return new Response(slicedBuffer, {
    status: 206,
    statusText: 'Partial Content',
    headers: new Headers({
      'Content-Type': cachedResponse.headers.get('Content-Type'),
      'Content-Range': `bytes ${start}-${end}/${arrayBuffer.byteLength}`,
      'Content-Length': slicedBuffer.byteLength
    })
  });
}
```

### 8.3 キャッシュサイズの管理

ストレージクォータを超えないよう、キャッシュサイズを管理する仕組みを構築する。

```javascript
// キャッシュサイズ管理ユーティリティ
class CacheManager {
  constructor(cacheName, options = {}) {
    this.cacheName = cacheName;
    this.maxEntries = options.maxEntries || 100;
    this.maxAgeMs = (options.maxAgeSeconds || 7 * 24 * 60 * 60) * 1000;
  }

  // エントリの追加（古いものを自動削除）
  async put(request, response) {
    const cache = await caches.open(this.cacheName);

    // タイムスタンプをヘッダーに記録
    const headers = new Headers(response.headers);
    headers.set('sw-cache-timestamp', Date.now().toString());

    const timestampedResponse = new Response(await response.blob(), {
      status: response.status,
      statusText: response.statusText,
      headers
    });

    await cache.put(request, timestampedResponse);

    // エントリ数の制限を適用
    await this.expireEntries();
  }

  // 期限切れ・超過エントリの削除
  async expireEntries() {
    const cache = await caches.open(this.cacheName);
    const keys = await cache.keys();

    // タイムスタンプとともにエントリを収集
    const entries = await Promise.all(
      keys.map(async (request) => {
        const response = await cache.match(request);
        const timestamp = response.headers.get('sw-cache-timestamp');
        return {
          request,
          timestamp: timestamp ? parseInt(timestamp, 10) : 0
        };
      })
    );

    // 古い順にソート
    entries.sort((a, b) => a.timestamp - b.timestamp);

    const now = Date.now();
    let deleted = 0;

    for (const entry of entries) {
      const isExpired = (now - entry.timestamp) > this.maxAgeMs;
      const isOverLimit = (entries.length - deleted) > this.maxEntries;

      if (isExpired || isOverLimit) {
        await cache.delete(entry.request);
        deleted++;
      }
    }

    if (deleted > 0) {
      console.log(`[CacheManager] Deleted ${deleted} entries from ${this.cacheName}`);
    }
  }

  // ストレージ使用量の確認
  static async getStorageEstimate() {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const { usage, quota } = await navigator.storage.estimate();
      return {
        usageMB: (usage / (1024 * 1024)).toFixed(2),
        quotaMB: (quota / (1024 * 1024)).toFixed(2),
        percentUsed: ((usage / quota) * 100).toFixed(2)
      };
    }
    return null;
  }
}
```

### 8.4 Navigation Preload

Navigation Preload は、Service Worker の起動とネットワークリクエストを並列化することで、ナビゲーション時のパフォーマンスを改善する仕組みである。

```javascript
// activate イベントで Navigation Preload を有効化
self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      if (self.registration.navigationPreload) {
        // Navigation Preload を有効化
        await self.registration.navigationPreload.enable();
        // カスタムヘッダーを設定（オプション）
        await self.registration.navigationPreload.setHeaderValue('true');
      }
    })()
  );
});

// fetch イベントで preloadResponse を活用
self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          // Navigation Preload のレスポンスを利用
          const preloadResponse = await event.preloadResponse;
          if (preloadResponse) {
            // キャッシュに保存
            const cache = await caches.open('pages-cache');
            cache.put(event.request, preloadResponse.clone());
            return preloadResponse;
          }

          // preload が使えない場合は通常のフェッチ
          const networkResponse = await fetch(event.request);
          return networkResponse;
        } catch (error) {
          // オフライン: キャッシュからフォールバック
          const cached = await caches.match(event.request);
          return cached || caches.match('/offline.html');
        }
      })()
    );
  }
});
