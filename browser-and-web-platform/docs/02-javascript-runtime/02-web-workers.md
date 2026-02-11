# Web Workers

> Web Workersはメインスレッドとは別のスレッドでJavaScriptを実行する仕組み。重い計算処理をオフロードしてUIの応答性を維持する。Worker, SharedWorker, ServiceWorkerの違いと使い分けを理解する。

## この章で学ぶこと

- [ ] Web Worker の基本的な使い方を理解する
- [ ] Worker の種類と使い分けを把握する
- [ ] Service Worker の役割を学ぶ

---

## 1. Dedicated Worker

```
Dedicated Worker = 1つのページ専用のワーカー

  メインスレッド                   Worker スレッド
  ┌──────────────┐              ┌──────────────┐
  │ UI操作       │  postMessage │ 重い計算      │
  │ DOM操作      │ ──────────→ │ データ処理    │
  │ イベント処理  │              │ 画像処理      │
  │              │ ←────────── │              │
  │              │  postMessage │              │
  └──────────────┘              └──────────────┘

  Worker でできないこと:
  ✗ DOM にアクセス
  ✗ document, window オブジェクト
  ✗ UI を直接操作

  Worker でできること:
  ✓ fetch() でネットワーク通信
  ✓ IndexedDB へのアクセス
  ✓ setTimeout / setInterval
  ✓ WebSocket
  ✓ importScripts() でスクリプト読み込み
```

```javascript
// main.js
const worker = new Worker('worker.js');

// Worker にデータを送信
worker.postMessage({ type: 'sort', data: largeArray });

// Worker からの結果を受信
worker.onmessage = (event) => {
  console.log('Sorted:', event.data);
};

worker.onerror = (error) => {
  console.error('Worker error:', error.message);
};

// 不要になったら終了
worker.terminate();

// worker.js
self.onmessage = (event) => {
  const { type, data } = event.data;

  switch (type) {
    case 'sort':
      const sorted = data.sort((a, b) => a - b);
      self.postMessage(sorted);
      break;
  }
};
```

---

## 2. Transferable Objects

```
postMessage のデフォルト:
  → データをコピーして送信（Structured Clone）
  → 大きなデータはコピーコストが高い

Transferable Objects:
  → データの所有権を移転（コピーではなく移動）
  → 移転後、送信元ではアクセス不可
  → ArrayBuffer, MessagePort, ImageBitmap 等

  // コピー（遅い）
  worker.postMessage(largeBuffer);

  // 移転（高速: ゼロコピー）
  worker.postMessage(largeBuffer, [largeBuffer]);
  // largeBuffer は使用不可になる

  性能比較（100MB ArrayBuffer）:
  コピー:  ~50ms
  移転:    ~0.1ms（500倍高速）

SharedArrayBuffer:
  → 複数のWorkerとメインスレッドで共有
  → Atomics API でスレッドセーフな操作
  → Cross-Origin Isolation が必要

  // SharedArrayBuffer の使用
  const sab = new SharedArrayBuffer(1024);
  const view = new Int32Array(sab);
  worker.postMessage(sab);  // 共有（コピーでも移転でもない）
```

---

## 3. Service Worker

```
Service Worker = ページとネットワークの間に立つプロキシ

  ブラウザ → Service Worker → ネットワーク
  ブラウザ → Service Worker → キャッシュ（オフライン対応）

ライフサイクル:
  1. 登録（register）
  2. インストール（install）→ キャッシュの準備
  3. 待機（waiting）→ 古いSWが制御中
  4. アクティベーション（activate）→ 古いキャッシュの削除
  5. フェッチ（fetch）→ リクエストのインターセプト

  // 登録
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js');
  }

  // sw.js
  const CACHE_NAME = 'v1';

  self.addEventListener('install', (event) => {
    event.waitUntil(
      caches.open(CACHE_NAME).then((cache) => {
        return cache.addAll([
          '/',
          '/style.css',
          '/app.js',
          '/offline.html',
        ]);
      })
    );
  });

  self.addEventListener('fetch', (event) => {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        return cached || fetch(event.request);
      })
    );
  });

キャッシュ戦略:
  Cache First:     キャッシュ優先 → なければネットワーク
  Network First:   ネットワーク優先 → 失敗時キャッシュ
  Stale While Revalidate: キャッシュ返却 + バックグラウンド更新
  Network Only:    常にネットワーク
  Cache Only:      常にキャッシュ
```

---

## 4. Worker の種類比較

```
┌──────────────────┬─────────────┬──────────────┬──────────────┐
│                  │ Dedicated   │ Shared       │ Service      │
├──────────────────┼─────────────┼──────────────┼──────────────┤
│ スコープ         │ 1ページ     │ 同一オリジン │ 同一オリジン │
│ 接続数           │ 1           │ 複数ページ   │ 全ページ     │
│ DOM アクセス     │ ✗           │ ✗           │ ✗           │
│ ライフサイクル   │ ページと同じ│ 全接続終了まで│ 独立(永続)  │
│ オフライン対応   │ ✗           │ ✗           │ ✓           │
│ Push通知         │ ✗           │ ✗           │ ✓           │
│ ネットワーク制御 │ ✗           │ ✗           │ ✓           │
│ 主な用途         │ 重い計算    │ 共有状態管理 │ キャッシュ   │
│                  │ データ処理  │ WebSocket共有│ PWA          │
└──────────────────┴─────────────┴──────────────┴──────────────┘
```

---

## 5. Worklet

```
Worklet = 軽量な Worker（レンダリングパイプラインに統合）

  種類:
  ① Paint Worklet（CSS Houdini）:
     → CSS の描画をカスタマイズ
     → canvas のように描画

  ② Animation Worklet:
     → Compositor Thread でアニメーション実行
     → メインスレッドに依存しない

  ③ Audio Worklet:
     → Web Audio の信号処理
     → リアルタイムオーディオ処理

  ④ Layout Worklet（実験的）:
     → カスタムレイアウトアルゴリズム

  // Paint Worklet の例
  CSS.paintWorklet.addModule('paint.js');

  // paint.js
  class CheckerPainter {
    paint(ctx, size) {
      const tileSize = 20;
      for (let y = 0; y < size.height; y += tileSize) {
        for (let x = 0; x < size.width; x += tileSize) {
          if ((x + y) % (tileSize * 2) === 0) {
            ctx.fillRect(x, y, tileSize, tileSize);
          }
        }
      }
    }
  }
  registerPaint('checker', CheckerPainter);
```

---

## まとめ

| Worker | 用途 | スコープ |
|--------|------|---------|
| Dedicated Worker | 重い計算のオフロード | 1ページ |
| Shared Worker | 複数タブ間の共有状態 | 同一オリジン |
| Service Worker | キャッシュ、オフライン、Push | 同一オリジン |
| Worklet | レンダリングパイプライン拡張 | 特定の処理 |

---

## 次に読むべきガイド
→ [[03-memory-management.md]] — メモリ管理

---

## 参考文献
1. MDN Web Docs. "Web Workers API." Mozilla, 2024.
2. web.dev. "Service Worker." Google, 2024.
