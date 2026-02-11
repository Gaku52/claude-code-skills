# ナビゲーションとローディング

> ブラウザのアドレスバーにURLを入力してからページが表示されるまでの全プロセスを追う。DNS解決、TLS接続、HTTP通信、HTMLパース、リソース読み込みの各段階を詳細に理解する。

## この章で学ぶこと

- [ ] ナビゲーション開始からレンダリングまでの全流れを理解する
- [ ] リソース読み込みの優先順位を把握する
- [ ] ページロードのパフォーマンス指標を学ぶ

---

## 1. ナビゲーションの全体像

```
URL入力 → ページ表示 の全プロセス:

  ① URL入力/クリック
     ↓
  ② URLの解析とセキュリティチェック
     ↓
  ③ Service Worker チェック（登録されていれば）
     ↓
  ④ DNS解決
     ↓
  ⑤ TCP接続（3-way handshake）
     ↓
  ⑥ TLSハンドシェイク（HTTPS の場合）
     ↓
  ⑦ HTTPリクエスト送信
     ↓
  ⑧ HTTPレスポンス受信（最初のバイト = TTFB）
     ↓
  ⑨ HTMLパース開始
     ↓
  ⑩ サブリソースの発見と読み込み（CSS, JS, 画像等）
     ↓
  ⑪ DOM構築 + CSSOM構築
     ↓
  ⑫ レンダーツリー構築
     ↓
  ⑬ レイアウト計算
     ↓
  ⑭ ペイント
     ↓
  ⑮ コンポジット（GPU合成）
     ↓
  ⑯ 画面表示
```

---

## 2. HTMLパースとリソース発見

```
HTMLパーサーの動作:

  <html>
  <head>
    <link rel="stylesheet" href="style.css">  ← レンダリングブロック
    <script src="app.js"></script>              ← パーサーブロック
  </head>
  <body>
    <img src="photo.jpg">                      ← 非ブロック
    <script src="analytics.js" defer></script>  ← 非ブロック
  </body>
  </html>

パーサーブロック:
  <script> タグに到達 → パース停止 → JS ダウンロード → JS 実行 → パース再開
  → JS が DOM を変更する可能性があるため

レンダリングブロック:
  CSS の読み込み → CSSOM が完成するまでレンダリングを保留
  → 正確なスタイル計算に必要

解決策:
  ┌────────────────────┬──────────────────────────────────┐
  │ 属性               │ 動作                             │
  ├────────────────────┼──────────────────────────────────┤
  │ <script>           │ パーサーブロック（ダウンロード+実行）│
  │ <script async>     │ ダウンロード並行、DL完了後即実行  │
  │ <script defer>     │ ダウンロード並行、DOMContentLoaded前に実行│
  │ <script type=module>│ defer相当 + ESModules            │
  └────────────────────┴──────────────────────────────────┘

  タイムライン:
  パーサー:    ─────パース─────│停止│─パース─
  <script>:                   │DL→│実行│
  <script async>: │──DL──│実行│  パーサーと並行DL
  <script defer>: │──DL──────│    │実行│  DOMContentLoaded前

Preload Scanner:
  → パーサーがブロックされている間も先読みスキャン
  → <link>, <script>, <img> を事前に発見
  → ダウンロードを開始（パース再開を待たない）
```

---

## 3. リソースの優先順位

```
Chromeのリソース読み込み優先順位:

  ┌─────────────────┬──────────┬────────────────────┐
  │ リソース         │ 優先度   │ 備考               │
  ├─────────────────┼──────────┼────────────────────┤
  │ HTML            │ Highest  │ 最優先              │
  │ CSS (head内)    │ Highest  │ レンダリングブロック │
  │ フォント(CSS参照)│ Highest  │ テキスト表示に必要  │
  │ Script (head内) │ High     │ async/deferで変化   │
  │ Script (body末)  │ Medium   │                    │
  │ 画像(viewport内)│ Medium   │ LCPに影響する場合High│
  │ 画像(viewport外)│ Low      │ lazy load対象       │
  │ Prefetch        │ Lowest   │ 将来のナビゲーション │
  └─────────────────┴──────────┴────────────────────┘

  fetchpriority 属性:
  <img src="hero.jpg" fetchpriority="high">  ← LCP画像の優先度アップ
  <img src="ad.jpg" fetchpriority="low">     ← 広告画像の優先度ダウン
  <script src="app.js" fetchpriority="high"> ← 重要なJSの優先度アップ

  リソースヒント:
  <link rel="preload" href="font.woff2" as="font" crossorigin>
  → 発見前からダウンロード開始

  <link rel="preconnect" href="https://api.example.com">
  → DNS + TCP + TLS を事前確立

  <link rel="prefetch" href="/next-page.html">
  → アイドル時に先読み（次のナビゲーション用）

  <link rel="modulepreload" href="/module.js">
  → ESModuleの先読み
```

---

## 4. ページロードのイベント

```
主要なイベントタイミング:

  0ms  ─── navigationStart
  │
  50ms ─── DNS解決完了
  │
  80ms ─── TCP接続完了
  │
  130ms ── TLS完了
  │
  150ms ── リクエスト送信
  │
  250ms ── TTFB（最初のバイト受信）
  │         → サーバー処理時間の指標
  │
  300ms ── FP（First Paint）
  │         → 最初のピクセルが表示
  │
  400ms ── FCP（First Contentful Paint）
  │         → 最初のテキスト/画像が表示
  │
  800ms ── DOMContentLoaded
  │         → DOM構築完了、defer script実行完了
  │         → jQuery の $(document).ready() はここ
  │
  1500ms ─ LCP（Largest Contentful Paint）
  │         → 最大のコンテンツが表示
  │         → Core Web Vitals 指標
  │
  2000ms ─ load
  │         → 全リソース（画像等）の読み込み完了
  │         → window.onload はここ
  │
  3000ms ─ fully interactive
             → JS実行完了、操作可能

  DOMContentLoaded vs load:
  DOMContentLoaded: HTMLパース完了（画像はまだかも）
  load: 画像、CSS、iframe 等全て完了
```

---

## 5. Navigation Timing API

```javascript
// ページ読み込みの各段階を計測
const entry = performance.getEntriesByType('navigation')[0];

console.log({
  // DNS
  dns: entry.domainLookupEnd - entry.domainLookupStart,

  // TCP接続
  tcp: entry.connectEnd - entry.connectStart,

  // TLS
  tls: entry.secureConnectionStart > 0
    ? entry.connectEnd - entry.secureConnectionStart : 0,

  // TTFB
  ttfb: entry.responseStart - entry.requestStart,

  // コンテンツ転送
  download: entry.responseEnd - entry.responseStart,

  // DOM処理
  domProcessing: entry.domContentLoadedEventEnd - entry.responseEnd,

  // 全体
  total: entry.loadEventEnd - entry.startTime,
});

// Web Vitals の計測
new PerformanceObserver((list) => {
  for (const entry of list.getEntries()) {
    console.log(`LCP: ${entry.startTime}ms`);
  }
}).observe({ type: 'largest-contentful-paint', buffered: true });
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ナビゲーション | DNS→TCP→TLS→HTTP→パース→レンダリング |
| パーサーブロック | <script>がHTMLパースを停止 |
| 解決策 | defer/async/preload/preconnect |
| 優先順位 | CSS=最高、画像=viewport依存 |
| イベント | DOMContentLoaded(DOM完了) vs load(全完了) |

---

## 次に読むべきガイド
→ [[02-parsing-html-css.md]] — HTML/CSSパーシング

---

## 参考文献
1. Mariko Kosaka. "Inside look at modern web browser (Part 2)." Google, 2018.
2. web.dev. "Optimizing resource loading." Google, 2024.
