# ナビゲーションとローディング

> ブラウザのアドレスバーにURLを入力してからページが表示されるまでの全プロセスを追う。DNS解決、TLS接続、HTTP通信、HTMLパース、リソース読み込みの各段階を詳細に理解する。

## この章で学ぶこと

- [ ] ナビゲーション開始からレンダリングまでの全流れを理解する
- [ ] リソース読み込みの優先順位を把握する
- [ ] ページロードのパフォーマンス指標を学ぶ
- [ ] Service Worker によるネットワーク制御を理解する
- [ ] Preload Scanner とリソースヒントの実務活用を身につける
- [ ] Navigation Timing API を使った計測とボトルネック特定を実践する
- [ ] HTTP/2・HTTP/3 がローディングに与える影響を把握する
- [ ] SPA と MPA のナビゲーション差異を理解する

---

## 1. ナビゲーションの全体像

### 1.1 URL入力からページ表示までの全プロセス

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

### 1.2 各フェーズの詳細な時間的分解

```
典型的なナビゲーションの時間構成（デスクトップ + 光回線）:

  フェーズ           │ 所要時間    │ 累積時間
  ─────────────────┼───────────┼──────────
  URL解析           │ <1ms       │ ~1ms
  キャッシュチェック   │ 1-5ms      │ ~5ms
  DNS解決           │ 20-120ms   │ ~50ms
  TCP 3-way HS     │ 10-50ms    │ ~80ms
  TLS 1.3 HS       │ 10-50ms    │ ~120ms
  HTTPリクエスト送信  │ 1-5ms      │ ~125ms
  サーバー処理       │ 50-500ms   │ ~300ms
  最初のバイト受信    │ 1ms        │ ~300ms (TTFB)
  HTML転送          │ 10-100ms   │ ~400ms
  HTMLパース開始     │ <1ms       │ ~400ms
  CSS/JSダウンロード │ 50-300ms   │ ~600ms
  DOM+CSSOM構築     │ 50-200ms   │ ~700ms
  レンダーツリー構築  │ 10-50ms    │ ~750ms
  レイアウト計算     │ 10-100ms   │ ~800ms
  ペイント          │ 5-50ms     │ ~850ms
  コンポジット       │ 1-10ms     │ ~860ms
  ─────────────────┼───────────┼──────────
  合計              │            │ ~860ms

モバイル（4G LTE）の場合:
  DNS解決: 50-200ms
  TCP + TLS: 100-300ms
  TTFB: 200-800ms
  合計: 1500-4000ms（デスクトップの2-5倍）
```

### 1.3 ブラウザのマルチプロセスアーキテクチャとナビゲーション

```
Chrome のプロセス間通信によるナビゲーション:

  Browser Process              Renderer Process (旧ページ)    Renderer Process (新ページ)
  ┌──────────────┐            ┌──────────────────┐          ┌──────────────────┐
  │ UI Thread    │            │                  │          │                  │
  │  ↓           │            │                  │          │                  │
  │ URL解析      │            │                  │          │                  │
  │  ↓           │            │                  │          │                  │
  │ Network      │            │                  │          │                  │
  │ Thread       │            │                  │          │                  │
  │  ↓           │            │                  │          │                  │
  │ DNS→TCP→TLS  │            │                  │          │                  │
  │  ↓           │            │                  │          │                  │
  │ HTTP送受信   │            │                  │          │                  │
  │  ↓           │            │                  │          │                  │
  │ レスポンス    │ ──unload──→│ beforeunload    │          │                  │
  │ ヘッダ確認   │            │ unload          │          │                  │
  │  ↓           │            │ (破棄)          │          │                  │
  │ Renderer     │ ─────────────────────────────→│ 初期化            │
  │ 選択/起動    │            │                  │          │  ↓               │
  │  ↓           │            │                  │          │ HTMLパース        │
  │ データ転送   │ ─────────────────────────────→│ DOM構築           │
  │              │            │                  │          │  ↓               │
  │              │            │                  │          │ レンダリング       │
  └──────────────┘            └──────────────────┘          └──────────────────┘

  ナビゲーションのプロセス間遷移:
  1. Browser Process の UI Thread がURL入力を受け取る
  2. Network Thread がネットワークリクエストを処理
  3. レスポンスのContent-Typeを確認
     - text/html → Renderer Process を起動
     - application/pdf → PDF Viewer
     - application/octet-stream → ダウンロードマネージャ
  4. 旧 Renderer Process に unload イベント送信
  5. 新 Renderer Process にデータを転送
  6. 新 Renderer Process が HTML パースとレンダリングを実行
```

### 1.4 Same-Site と Cross-Site ナビゲーション

```
Same-Site ナビゲーション:
  example.com/page1 → example.com/page2
  → 同じ Renderer Process を再利用可能
  → プロセス起動コストが不要
  → メモリ効率が良い

Cross-Site ナビゲーション:
  example.com → other-site.com
  → 新しい Renderer Process を起動
  → Site Isolation によるセキュリティ確保
  → プロセス起動に 50-150ms 追加

Back/Forward Cache (bfcache):
  → ページ全体をメモリに保持
  → 戻る/進むが瞬時（数ms）
  → ただし以下の条件で無効化:
    - unload イベントリスナーがある
    - Cache-Control: no-store
    - WebSocket や WebRTC が使用中
    - HTTP接続（HTTPS のみ対象）
```

---

## 2. DNS解決の詳細

### 2.1 DNS解決のフロー

```
DNSルックアップの階層構造:

  ブラウザ
  │
  ├── ① ブラウザDNSキャッシュ確認（数秒〜数分キャッシュ）
  │     → Chrome: chrome://net-internals/#dns
  │     → ヒット → 即座にIPアドレス取得（<1ms）
  │
  ├── ② OS DNSキャッシュ確認
  │     → /etc/hosts ファイルもここで参照
  │     → Windows: ipconfig /displaydns
  │     → macOS: dscacheutil -cachedump
  │     → ヒット → IPアドレス取得（1-5ms）
  │
  ├── ③ リゾルバ（ISP/パブリックDNS）に問い合わせ
  │     → Google DNS: 8.8.8.8
  │     → Cloudflare DNS: 1.1.1.1
  │     → リゾルバキャッシュヒット → 10-30ms
  │
  ├── ④ ルートDNSサーバー（.）
  │     → "example.com" → ".com のネームサーバーはここ"
  │     → 世界に13系統（エニーキャスト）
  │
  ├── ⑤ TLDネームサーバー（.com, .jp 等）
  │     → "example.com" → "example.com のNSサーバーはここ"
  │
  └── ⑥ 権威DNSサーバー（ドメイン管理者）
        → "example.com" → "93.184.216.34"
        → TTL付きで返答

  フル解決の場合: 100-200ms
  キャッシュヒット: <5ms
```

### 2.2 DNS over HTTPS (DoH) と DNS over TLS (DoT)

```
従来のDNS:
  ポート53、平文UDP → ISPや攻撃者がDNSクエリを覗き見/改ざん可能

DNS over HTTPS (DoH):
  → HTTPS(443)でDNSクエリを暗号化
  → ブラウザが直接サポート
  → Chrome: chrome://settings/security → セキュアDNS
  → Firefox: about:preferences#general → DNS over HTTPS

DNS over TLS (DoT):
  → TLS(853)でDNSクエリを暗号化
  → OS レベルでサポート

パフォーマンスへの影響:
  初回: DoH は TLS ハンドシェイク分遅い（+50-100ms）
  2回目以降: HTTP/2 接続再利用で同等

実装例（Chrome の DoH 設定確認）:
```

```javascript
// DNS解決時間の計測
async function measureDNSTime(hostname) {
  const start = performance.now();

  // Resource Timing API を使用
  const img = new Image();
  img.src = `https://${hostname}/favicon.ico?t=${Date.now()}`;

  return new Promise((resolve) => {
    img.onload = img.onerror = () => {
      const entries = performance.getEntriesByName(img.src);
      if (entries.length > 0) {
        const entry = entries[0];
        resolve({
          dnsTime: entry.domainLookupEnd - entry.domainLookupStart,
          connectTime: entry.connectEnd - entry.connectStart,
          totalTime: performance.now() - start,
        });
      }
    };
  });
}

// 使用例
measureDNSTime('api.example.com').then(console.log);
// { dnsTime: 23.5, connectTime: 45.2, totalTime: 312.8 }
```

### 2.3 DNS プリフェッチの実装

```html
<!-- DNSプリフェッチ: 事前にDNS解決だけ行う -->
<link rel="dns-prefetch" href="//api.example.com">
<link rel="dns-prefetch" href="//cdn.example.com">
<link rel="dns-prefetch" href="//fonts.googleapis.com">

<!-- preconnect: DNS + TCP + TLS を事前確立 -->
<link rel="preconnect" href="https://api.example.com">
<link rel="preconnect" href="https://cdn.example.com" crossorigin>

<!--
  dns-prefetch vs preconnect の使い分け:

  dns-prefetch:
    - コスト: 低（DNS解決のみ）
    - 対象: 使うかもしれない外部ドメイン
    - 上限目安: 10-15個

  preconnect:
    - コスト: 中（DNS + TCP + TLS）
    - 対象: 確実に使う外部ドメイン
    - 上限目安: 3-5個（接続維持のコストがある）
    - 10秒以内に使わないと接続が切断される
-->
```

```javascript
// 動的な DNS プリフェッチ
function prefetchDNS(hostname) {
  const link = document.createElement('link');
  link.rel = 'dns-prefetch';
  link.href = `//${hostname}`;
  document.head.appendChild(link);
}

// ユーザーがリンクにホバーした時に DNS を先に解決
document.querySelectorAll('a[href^="http"]').forEach((anchor) => {
  anchor.addEventListener(
    'mouseenter',
    () => {
      const url = new URL(anchor.href);
      if (url.hostname !== location.hostname) {
        prefetchDNS(url.hostname);
      }
    },
    { once: true }
  );
});
```

---

## 3. TCP接続とTLSハンドシェイク

### 3.1 TCP 3-way ハンドシェイク

```
TCP 3-way Handshake:

  クライアント                    サーバー
  │                              │
  │ ── SYN (seq=100) ──────────→│  ① SYN送信
  │                              │     クライアントが接続要求
  │                              │
  │←── SYN+ACK (seq=300,ack=101)│  ② SYN+ACK受信
  │                              │     サーバーが応答
  │                              │
  │ ── ACK (ack=301) ──────────→│  ③ ACK送信
  │                              │     接続確立
  │                              │
  │ ── HTTP GET / ─────────────→│  ④ データ送信可能
  │                              │

  所要時間 = RTT × 1.5
  （RTT: Round Trip Time）

  光回線(国内): RTT 5-20ms → TCP確立 7-30ms
  4G LTE:     RTT 30-80ms → TCP確立 45-120ms
  海外サーバー: RTT 100-300ms → TCP確立 150-450ms

TCP Fast Open (TFO):
  → 初回接続時にCookieを取得
  → 2回目以降は SYN に HTTP データを載せる
  → 1 RTT 削減
  → Linux, macOS でサポート
```

### 3.2 TLS 1.3 ハンドシェイク

```
TLS 1.3 ハンドシェイク（1-RTT）:

  クライアント                        サーバー
  │                                  │
  │ ── ClientHello ────────────────→│
  │    + サポートする暗号スイート      │
  │    + Key Share（鍵交換パラメータ） │
  │    + SNI（Server Name Indication）│
  │                                  │
  │←── ServerHello ─────────────────│
  │    + 選択した暗号スイート          │
  │    + Key Share                   │
  │    + 証明書                      │
  │    + 証明書検証                   │
  │    + Finished                    │
  │                                  │
  │ ── Finished ───────────────────→│
  │ ── HTTP リクエスト ──────────────→│  暗号化通信開始
  │                                  │

  TLS 1.2: 2-RTT（追加のラウンドトリップが必要）
  TLS 1.3: 1-RTT（鍵交換を最初のメッセージに含む）

  TLS 1.3 0-RTT（再接続時）:
  → 前回のセッションチケットを使用
  → ClientHello にアプリケーションデータを含める
  → ただしリプレイ攻撃のリスクあり（GETのみ推奨）

比較:
  TLS 1.2: TCP(1.5 RTT) + TLS(2 RTT) = 3.5 RTT
  TLS 1.3: TCP(1.5 RTT) + TLS(1 RTT) = 2.5 RTT
  TLS 1.3 0-RTT: TCP(1.5 RTT) + TLS(0 RTT) = 1.5 RTT
```

### 3.3 QUIC/HTTP3 による接続最適化

```
HTTP/3 (QUIC) のハンドシェイク:

  従来（HTTP/2 over TLS 1.3）:
    TCP 3-way HS:  1.5 RTT
    TLS 1.3 HS:    1 RTT
    合計:           2.5 RTT

  HTTP/3 (QUIC):
    QUIC HS（暗号化統合）: 1 RTT
    合計:                   1 RTT

  HTTP/3 0-RTT（再接続）:
    合計: 0 RTT（データを最初のパケットで送信）

  クライアント                    サーバー
  │                              │
  │ ── QUIC Initial ───────────→│  暗号化パラメータ + HTTP リクエスト
  │                              │  （0-RTT の場合）
  │                              │
  │←── QUIC Handshake ──────────│  暗号化完了 + HTTP レスポンス開始
  │                              │
  │ ── QUIC Short Header ──────→│  以降は暗号化されたデータ通信
  │                              │

QUIC の追加メリット:
  - ヘッドオブラインブロッキング解消
    → 1つのストリームのパケットロスが他に影響しない
  - 接続マイグレーション
    → Wi-Fi → 4G 切り替え時に接続を維持
  - 輻輳制御の改善
    → ストリーム単位での制御
```

---

## 4. HTTPリクエストとレスポンス

### 4.1 HTTPリクエストの構造

```http
GET /index.html HTTP/2
Host: example.com
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8
Accept-Language: ja,en-US;q=0.7,en;q=0.3
Accept-Encoding: gzip, deflate, br
Connection: keep-alive
Cookie: session=abc123; theme=dark
Cache-Control: max-age=0
Sec-Fetch-Dest: document
Sec-Fetch-Mode: navigate
Sec-Fetch-Site: none
Sec-Fetch-User: ?1
Upgrade-Insecure-Requests: 1
```

```
リクエストヘッダの役割:

  Accept-Encoding: gzip, deflate, br
  → サポートする圧縮形式を通知
  → Brotli(br) はGzipより15-25%効率的

  Sec-Fetch-* ヘッダ:
  → ブラウザが自動付与（改ざん不可）
  → サーバー側でリクエストの出所を判定可能

  Sec-Fetch-Dest: document    → ページナビゲーション
  Sec-Fetch-Dest: image       → 画像リクエスト
  Sec-Fetch-Dest: script      → スクリプトリクエスト
  Sec-Fetch-Mode: navigate    → ユーザー操作によるナビゲーション
  Sec-Fetch-Mode: cors        → CORS リクエスト
  Sec-Fetch-Site: same-origin → 同一オリジン
  Sec-Fetch-Site: cross-site  → クロスサイト
```

### 4.2 HTTPレスポンスの構造とキャッシュ

```http
HTTP/2 200 OK
Content-Type: text/html; charset=utf-8
Content-Length: 45230
Content-Encoding: br
Cache-Control: public, max-age=3600, stale-while-revalidate=86400
ETag: "abc123"
Last-Modified: Mon, 20 Jan 2026 10:00:00 GMT
X-Frame-Options: DENY
Content-Security-Policy: default-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

```
キャッシュ制御の詳細:

  ┌──────────────────────────────────────────────────────────┐
  │ Cache-Control ディレクティブ                              │
  ├──────────────────┬───────────────────────────────────────┤
  │ public           │ CDN・共有キャッシュに保存可            │
  │ private          │ ブラウザキャッシュのみ                  │
  │ no-cache         │ 毎回サーバーに検証（キャッシュは保持）   │
  │ no-store         │ 一切キャッシュしない                    │
  │ max-age=3600     │ 3600秒間新鮮とみなす                   │
  │ s-maxage=86400   │ 共有キャッシュ用の有効期限              │
  │ stale-while-     │ 期限切れでも表示しつつバックグラウンド   │
  │  revalidate=86400│  で再検証（86400秒まで）               │
  │ stale-if-error   │ エラー時に期限切れキャッシュを表示       │
  │ immutable        │ max-age内はリロードでも再検証しない     │
  │ must-revalidate  │ 期限切れ後は必ず再検証                  │
  └──────────────────┴───────────────────────────────────────┘

  キャッシュ判定フロー:

  リクエスト発生
    ↓
  キャッシュにある？ ─ No → ネットワークリクエスト
    │ Yes
    ↓
  max-age 内？ ─ Yes → キャッシュから返却（200 from cache）
    │ No
    ↓
  ETag/Last-Modified あり？ ─ No → ネットワークリクエスト
    │ Yes
    ↓
  条件付きリクエスト送信
  If-None-Match: "abc123"
  If-Modified-Since: Mon, 20 Jan 2026 10:00:00 GMT
    ↓
  サーバーレスポンス
    ├─ 304 Not Modified → キャッシュを使用
    └─ 200 OK → 新しいレスポンスで更新
```

### 4.3 圧縮とエンコーディング

```javascript
// サーバーサイド（Node.js/Express）での圧縮設定
const express = require('express');
const compression = require('compression');

const app = express();

// Brotli + Gzip 圧縮の設定
app.use(
  compression({
    // Brotli を優先
    filter: (req, res) => {
      if (req.headers['x-no-compression']) return false;
      return compression.filter(req, res);
    },
    // 1KB以上のレスポンスのみ圧縮
    threshold: 1024,
  })
);

// 静的ファイルの事前圧縮（ビルド時に .br, .gz を生成）
// Nginx 設定例
/*
  # Brotli の事前圧縮ファイルを優先
  brotli_static on;
  gzip_static on;

  # 動的圧縮（事前圧縮がない場合）
  brotli on;
  brotli_comp_level 6;
  brotli_types text/html text/css application/javascript application/json;

  gzip on;
  gzip_comp_level 6;
  gzip_types text/html text/css application/javascript application/json;
*/

// 圧縮効率の比較（typical values）
const compressionRatios = {
  'HTML (100KB)': { gzip: '25KB (75%)', brotli: '20KB (80%)' },
  'CSS (50KB)': { gzip: '12KB (76%)', brotli: '9KB (82%)' },
  'JavaScript (200KB)': { gzip: '55KB (72%)', brotli: '45KB (77%)' },
  'JSON API (30KB)': { gzip: '6KB (80%)', brotli: '5KB (83%)' },
};
```

---

## 5. HTMLパースとリソース発見

### 5.1 パーサーの動作モデル

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

### 5.2 Speculative Parsing（投機的パース）の詳細

```
Preload Scanner（投機的パーサー）の仕組み:

  メインパーサー                    Preload Scanner
  ─────────────────                ─────────────────
  <html> パース開始                │
  <head> パース                    │
  <link rel="stylesheet"> 発見     │
   → CSS ダウンロード開始          │
  <script src="app.js"> 発見       │
   → パーサーブロック！             │
   → JS ダウンロード待ち           │
   │                              │ 先行してHTMLをスキャン
   │ (停止中)                     │ <img src="hero.jpg"> 発見
   │                              │  → ダウンロード開始
   │                              │ <script src="util.js"> 発見
   │                              │  → ダウンロード開始
   │                              │ <link rel="stylesheet" href="page.css">
   │                              │  → ダウンロード開始
   │                              │
  app.js 実行完了                  │
  パース再開                       │
  hero.jpg → すでにDL済み！        │
  util.js → すでにDL済み！         │
  page.css → すでにDL済み！        │

  Preload Scanner による効果:
  → Without: 各リソースをシーケンシャルに発見・DL
  → With: ブロック中に先読みして並列DL
  → 典型的に20-50%のローディング時間短縮

  注意: Preload Scanner が見つけられないもの:
  - JavaScript で動的に追加されるリソース
  - CSS の @import で参照されるリソース
  - CSS の background-image
  - Web Font（CSS 内で @font-face で定義）
  → これらには明示的な preload が必要
```

### 5.3 async / defer / module の実務的使い分け

```html
<!-- ❌ パーサーブロック：避けるべき配置 -->
<head>
  <script src="analytics.js"></script> <!-- パースを止める -->
</head>

<!-- ✅ defer：DOM解析後に順序通り実行 -->
<head>
  <script src="vendor.js" defer></script>   <!-- 1番目に実行 -->
  <script src="app.js" defer></script>      <!-- 2番目に実行（依存関係を保持） -->
  <script src="init.js" defer></script>     <!-- 3番目に実行 -->
</head>

<!-- ✅ async：独立したスクリプト向け -->
<head>
  <script src="analytics.js" async></script>  <!-- 他に依存しない -->
  <script src="ads.js" async></script>        <!-- 他に依存しない -->
</head>

<!-- ✅ type="module"：ESModules（defer相当 + strict mode） -->
<head>
  <script type="module" src="app.mjs"></script>
</head>

<!-- ✅ 動的import：必要な時にロード -->
<script>
  // ユーザー操作時に初めてロード
  document.getElementById('editor-btn').addEventListener('click', async () => {
    const { Editor } = await import('./editor.mjs');
    const editor = new Editor('#container');
    editor.init();
  });
</script>
```

```javascript
// defer vs async の動作を実験するコード
// defer-test.js
console.log('defer script executed');
console.log('DOM ready:', document.readyState);
console.log('Body exists:', !!document.body);
// → "defer script executed"
// → "DOM ready: interactive"
// → "Body exists: true"

// async-test.js
console.log('async script executed');
console.log('DOM ready:', document.readyState);
// → "async script executed"
// → "DOM ready: loading" (DL完了タイミング次第で interactive の場合も)

// module-test.mjs
console.log('module script executed');
console.log('DOM ready:', document.readyState);
// → "module script executed"
// → "DOM ready: interactive" （defer と同じ）

// inline module は即座に defer 扱い
// <script type="module">
//   console.log('inline module');
//   // → DOMContentLoaded 前に実行される
// </script>
```

### 5.4 CSS の読み込み戦略

```html
<!-- クリティカルCSS：インライン化してFCPを高速化 -->
<head>
  <style>
    /* First Paint に必要な最小CSS（Above-the-fold） */
    body { margin: 0; font-family: system-ui; }
    .header { background: #1a1a2e; color: white; padding: 16px; }
    .hero { min-height: 60vh; display: flex; align-items: center; }
    .hero h1 { font-size: 2.5rem; }
  </style>

  <!-- 残りのCSSは非同期で読み込み -->
  <link rel="preload" href="/css/full.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/css/full.css"></noscript>
</head>
```

```javascript
// クリティカルCSSの自動抽出（Node.jsビルドスクリプト）
const critical = require('critical');

async function generateCriticalCSS() {
  const result = await critical.generate({
    // 対象ページのHTMLファイルまたはURL
    src: 'https://example.com',
    // ビューポートサイズ
    width: 1300,
    height: 900,
    // インライン化する
    inline: true,
    // 出力先
    target: {
      html: 'dist/index.html',
      css: 'dist/critical.css',
      uncritical: 'dist/rest.css',
    },
  });

  console.log('Critical CSS extracted:', result.css.length, 'bytes');
}

// CSS の @import はレンダリングを遅延させる
// ❌ 悪い例：チェーン読み込み
// style.css → @import "reset.css" → @import "variables.css"
// → シーケンシャルにダウンロードされる

// ✅ 良い例：並列読み込み
// <link rel="stylesheet" href="reset.css">
// <link rel="stylesheet" href="variables.css">
// <link rel="stylesheet" href="style.css">
```

---

## 6. リソースの優先順位

### 6.1 Chrome のリソース読み込み優先順位

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

### 6.2 fetchpriority の実務活用

```html
<!-- LCP要素の優先度を上げる -->
<img src="/hero-banner.webp"
     alt="Hero Banner"
     fetchpriority="high"
     width="1200"
     height="600">

<!-- ファーストビュー外の画像は遅延読み込み -->
<img src="/product-1.webp"
     alt="Product 1"
     loading="lazy"
     fetchpriority="auto"
     width="400"
     height="300">

<!-- カルーセルの最初の画像だけ高優先度 -->
<div class="carousel">
  <img src="/slide-1.webp" fetchpriority="high">
  <img src="/slide-2.webp" fetchpriority="low" loading="lazy">
  <img src="/slide-3.webp" fetchpriority="low" loading="lazy">
</div>

<!-- フォントの事前読み込み -->
<link rel="preload"
      href="/fonts/NotoSansJP-Regular.woff2"
      as="font"
      type="font/woff2"
      crossorigin
      fetchpriority="high">

<!-- 重要なAPIリクエストの優先度を上げる -->
<script>
  // fetchpriority を fetch API で使用
  const response = await fetch('/api/critical-data', {
    priority: 'high', // Fetch Priority API
  });

  // 低優先度のプリフェッチ
  const prefetchResponse = await fetch('/api/suggestions', {
    priority: 'low',
  });
</script>
```

### 6.3 HTTP/2 の優先順位とマルチプレキシング

```
HTTP/1.1 の制限:
  → 1つのTCP接続で1つのリクエスト/レスポンス
  → ブラウザはドメインあたり6接続まで
  → 7個目以降は待ち行列

  接続1: ─[HTML]──[CSS]──[JS1]──[img1]──
  接続2: ──────[JS2]──[img2]──[img3]──
  接続3: ──────[font1]──[img4]──[img5]──
  接続4: ──────────[img6]──[img7]──
  接続5: ──────────[img8]──[img9]──
  接続6: ──────────[img10]──[img11]──
  待ち:  ──────────────────[img12] [img13]...

HTTP/2 のマルチプレキシング:
  → 1つのTCP接続で複数のストリームを並行
  → ドメインあたり1接続で全リソース
  → 優先順位ベースのストリーム制御

  接続1: ─[HTML]─┬─[CSS]─┬─[JS1]──┬─[JS2]───
                 ├─[font]┤        ├─[img1]──
                 │       │        ├─[img2]──
                 │       │        └─[img3]──
                 │       │
  優先順位ツリー:
    HTML (weight: 256)
    ├── CSS (weight: 256, exclusive)
    ├── JS (weight: 220)
    ├── Font (weight: 256)
    └── Images (weight: 110)

HTTP/3 の改善:
  → QUIC ストリームレベルでの多重化
  → 1つのストリームの遅延が他に影響しない
  → パケットロス時の回復が高速
```

---

## 7. ページロードのイベントとライフサイクル

### 7.1 主要なイベントタイミング

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

### 7.2 Core Web Vitals の詳細

```
Core Web Vitals（2024年〜の指標）:

  ┌────────────────────────────────────────────┐
  │ LCP (Largest Contentful Paint)              │
  │ → ビューポート内の最大要素が表示された時刻    │
  │ → 良好: ≤2.5s / 要改善: ≤4.0s / 不良: >4.0s │
  │                                            │
  │ 対象要素:                                    │
  │   - <img>                                   │
  │   - <svg> 内の <image>                      │
  │   - <video> のポスター画像                    │
  │   - background-image の要素                  │
  │   - テキストノードを含むブロック要素            │
  └────────────────────────────────────────────┘

  ┌────────────────────────────────────────────┐
  │ INP (Interaction to Next Paint)             │
  │ → ユーザー操作から画面更新までの遅延          │
  │ → FID の後継指標（2024年3月〜）              │
  │ → 良好: ≤200ms / 要改善: ≤500ms / 不良: >500ms│
  │                                            │
  │ 計測対象のイベント:                           │
  │   - click / tap                             │
  │   - keydown / keyup                         │
  │   - mousedown / mouseup                     │
  │                                            │
  │ INP = 入力遅延 + 処理時間 + 表示遅延         │
  │   入力遅延: メインスレッドがビジーの間の待ち    │
  │   処理時間: イベントハンドラの実行時間         │
  │   表示遅延: レイアウト → ペイント → コンポジット│
  └────────────────────────────────────────────┘

  ┌────────────────────────────────────────────┐
  │ CLS (Cumulative Layout Shift)               │
  │ → 予期しないレイアウトのずれの累積             │
  │ → 良好: ≤0.1 / 要改善: ≤0.25 / 不良: >0.25  │
  │                                            │
  │ CLS を引き起こす原因:                         │
  │   - サイズ未指定の画像/iframe                 │
  │   - 動的に挿入されるコンテンツ                 │
  │   - Webフォントの読み込み（FOIT/FOUT）         │
  │   - DOM操作でのコンテンツ追加                  │
  └────────────────────────────────────────────┘
```

### 7.3 パフォーマンス指標の計測実装

```javascript
// Core Web Vitals を web-vitals ライブラリで計測
import { onLCP, onINP, onCLS, onFCP, onTTFB } from 'web-vitals';

function sendToAnalytics(metric) {
  const body = {
    name: metric.name,
    value: metric.value,
    rating: metric.rating, // "good" | "needs-improvement" | "poor"
    delta: metric.delta,
    id: metric.id,
    navigationType: metric.navigationType,
    // LCP の場合、対象要素の情報
    ...(metric.entries?.length && {
      element: metric.entries[metric.entries.length - 1]?.element?.tagName,
      url: metric.entries[metric.entries.length - 1]?.url,
    }),
  };

  // Beacon API で確実に送信（ページ離脱時も）
  if (navigator.sendBeacon) {
    navigator.sendBeacon('/analytics', JSON.stringify(body));
  } else {
    fetch('/analytics', {
      method: 'POST',
      body: JSON.stringify(body),
      keepalive: true,
    });
  }
}

// 各指標を計測・送信
onLCP(sendToAnalytics);
onINP(sendToAnalytics);
onCLS(sendToAnalytics);
onFCP(sendToAnalytics);
onTTFB(sendToAnalytics);

// PerformanceObserver を使った詳細計測
class PerformanceMonitor {
  constructor() {
    this.metrics = {};
    this.setupObservers();
  }

  setupObservers() {
    // LCP
    this.observe('largest-contentful-paint', (entries) => {
      const last = entries[entries.length - 1];
      this.metrics.lcp = {
        value: last.startTime,
        element: last.element?.tagName,
        size: last.size,
        url: last.url,
      };
      console.log(`LCP: ${last.startTime.toFixed(0)}ms`, last.element);
    });

    // CLS
    let clsValue = 0;
    this.observe('layout-shift', (entries) => {
      entries.forEach((entry) => {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
          this.metrics.cls = { value: clsValue };
          console.log(
            `Layout shift: ${entry.value.toFixed(4)}`,
            `Total CLS: ${clsValue.toFixed(4)}`,
            entry.sources?.map((s) => s.node?.tagName)
          );
        }
      });
    });

    // Long Tasks（INP の原因調査に有用）
    this.observe('longtask', (entries) => {
      entries.forEach((entry) => {
        console.warn(
          `Long task: ${entry.duration.toFixed(0)}ms`,
          entry.attribution?.[0]?.containerType,
          entry.attribution?.[0]?.containerName
        );
      });
    });

    // Resource Timing（個別リソースの読み込み時間）
    this.observe('resource', (entries) => {
      entries.forEach((entry) => {
        if (entry.duration > 500) {
          console.warn(`Slow resource: ${entry.name}`, {
            duration: `${entry.duration.toFixed(0)}ms`,
            size: `${(entry.transferSize / 1024).toFixed(1)}KB`,
            type: entry.initiatorType,
          });
        }
      });
    });
  }

  observe(type, callback) {
    try {
      const observer = new PerformanceObserver((list) => {
        callback(list.getEntries());
      });
      observer.observe({ type, buffered: true });
    } catch (e) {
      console.warn(`PerformanceObserver for ${type} not supported`);
    }
  }

  getReport() {
    return {
      ...this.metrics,
      navigation: this.getNavigationTiming(),
      resources: this.getResourceSummary(),
    };
  }

  getNavigationTiming() {
    const entry = performance.getEntriesByType('navigation')[0];
    if (!entry) return null;

    return {
      dns: Math.round(entry.domainLookupEnd - entry.domainLookupStart),
      tcp: Math.round(entry.connectEnd - entry.connectStart),
      tls:
        entry.secureConnectionStart > 0
          ? Math.round(entry.connectEnd - entry.secureConnectionStart)
          : 0,
      ttfb: Math.round(entry.responseStart - entry.requestStart),
      download: Math.round(entry.responseEnd - entry.responseStart),
      domProcessing: Math.round(
        entry.domContentLoadedEventEnd - entry.responseEnd
      ),
      domContentLoaded: Math.round(entry.domContentLoadedEventEnd),
      load: Math.round(entry.loadEventEnd),
      transferSize: entry.transferSize,
      encodedBodySize: entry.encodedBodySize,
      decodedBodySize: entry.decodedBodySize,
    };
  }

  getResourceSummary() {
    const resources = performance.getEntriesByType('resource');
    const summary = {};

    resources.forEach((r) => {
      const type = r.initiatorType || 'other';
      if (!summary[type]) {
        summary[type] = { count: 0, totalSize: 0, totalDuration: 0 };
      }
      summary[type].count++;
      summary[type].totalSize += r.transferSize || 0;
      summary[type].totalDuration += r.duration;
    });

    return summary;
  }
}

// 使用例
const monitor = new PerformanceMonitor();
window.addEventListener('load', () => {
  // ページ完全読み込み後にレポート取得
  setTimeout(() => {
    console.table(monitor.getReport().navigation);
    console.table(monitor.getReport().resources);
  }, 3000);
});
```

---

## 8. Navigation Timing API

### 8.1 基本的な計測

```javascript
// ページ読み込みの各段階を計測
const entry = performance.getEntriesByType('navigation')[0];

console.log({
  // DNS
  dns: entry.domainLookupEnd - entry.domainLookupStart,

  // TCP接続
  tcp: entry.connectEnd - entry.connectStart,

  // TLS
  tls:
    entry.secureConnectionStart > 0
      ? entry.connectEnd - entry.secureConnectionStart
      : 0,

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

### 8.2 Navigation Timing Level 2 の全プロパティ

```javascript
// Navigation Timing Level 2 のタイムライン
const nav = performance.getEntriesByType('navigation')[0];

/*
  タイムライン:

  startTime (0)
  │
  ├─ redirectStart ──── redirectEnd
  │   (リダイレクトがある場合)
  │
  ├─ fetchStart
  │   (リクエスト開始)
  │
  ├─ domainLookupStart ──── domainLookupEnd
  │   (DNS解決)
  │
  ├─ connectStart ──── secureConnectionStart ──── connectEnd
  │   (TCP接続)         (TLS開始)                  (TLS完了)
  │
  ├─ requestStart
  │   (リクエスト送信)
  │
  ├─ responseStart ──── responseEnd
  │   (TTFB)            (レスポンス受信完了)
  │
  ├─ domInteractive
  │   (HTMLパース完了、DOMが操作可能)
  │
  ├─ domContentLoadedEventStart ──── domContentLoadedEventEnd
  │   (DOMContentLoadedイベント)
  │
  └─ loadEventStart ──── loadEventEnd
      (loadイベント)
*/

// 実務で使える診断レポート
function generateLoadReport() {
  const nav = performance.getEntriesByType('navigation')[0];
  if (!nav) return null;

  const report = {
    // === ネットワーク層 ===
    redirect:
      nav.redirectEnd > 0
        ? `${(nav.redirectEnd - nav.redirectStart).toFixed(0)}ms (${nav.redirectCount} redirects)`
        : 'none',
    dns: `${(nav.domainLookupEnd - nav.domainLookupStart).toFixed(0)}ms`,
    tcp: `${(nav.connectEnd - nav.connectStart).toFixed(0)}ms`,
    tls:
      nav.secureConnectionStart > 0
        ? `${(nav.connectEnd - nav.secureConnectionStart).toFixed(0)}ms`
        : 'N/A',

    // === サーバー層 ===
    ttfb: `${(nav.responseStart - nav.requestStart).toFixed(0)}ms`,
    serverTime: `${(nav.responseStart - nav.connectEnd).toFixed(0)}ms`,

    // === コンテンツ転送 ===
    download: `${(nav.responseEnd - nav.responseStart).toFixed(0)}ms`,
    transferSize: `${(nav.transferSize / 1024).toFixed(1)}KB`,
    compressionRatio:
      nav.decodedBodySize > 0
        ? `${((1 - nav.encodedBodySize / nav.decodedBodySize) * 100).toFixed(0)}%`
        : 'N/A',

    // === クライアント層 ===
    domParsing: `${(nav.domInteractive - nav.responseEnd).toFixed(0)}ms`,
    domContentLoaded: `${nav.domContentLoadedEventEnd.toFixed(0)}ms`,
    load: `${nav.loadEventEnd.toFixed(0)}ms`,

    // === プロトコル情報 ===
    protocol: nav.nextHopProtocol, // "h2", "h3", "http/1.1"
    type: nav.type, // "navigate", "reload", "back_forward", "prerender"
  };

  return report;
}

// コンソールにテーブル表示
console.table(generateLoadReport());
```

### 8.3 Resource Timing API の活用

```javascript
// 全リソースの読み込み時間を分析
function analyzeResources() {
  const resources = performance.getEntriesByType('resource');

  // リソースタイプ別に分類
  const byType = {};
  resources.forEach((r) => {
    const type = r.initiatorType;
    if (!byType[type]) byType[type] = [];
    byType[type].push({
      name: r.name.split('/').pop().split('?')[0], // ファイル名のみ
      duration: Math.round(r.duration),
      size: Math.round(r.transferSize / 1024), // KB
      protocol: r.nextHopProtocol,
      cached: r.transferSize === 0 && r.decodedBodySize > 0,
    });
  });

  // 遅いリソースを特定
  const slowResources = resources
    .filter((r) => r.duration > 200)
    .sort((a, b) => b.duration - a.duration)
    .slice(0, 10)
    .map((r) => ({
      name: r.name,
      duration: `${Math.round(r.duration)}ms`,
      size: `${Math.round(r.transferSize / 1024)}KB`,
      type: r.initiatorType,
    }));

  console.log('=== Resource Summary ===');
  Object.entries(byType).forEach(([type, items]) => {
    const totalSize = items.reduce((sum, r) => sum + r.size, 0);
    const avgDuration =
      items.reduce((sum, r) => sum + r.duration, 0) / items.length;
    const cachedCount = items.filter((r) => r.cached).length;

    console.log(
      `${type}: ${items.length} files, ${totalSize}KB total, ` +
        `avg ${Math.round(avgDuration)}ms, ${cachedCount} cached`
    );
  });

  console.log('\n=== Slowest Resources ===');
  console.table(slowResources);

  return { byType, slowResources };
}

// Server Timing API の活用
// サーバーサイドで設定:
// Server-Timing: db;dur=42, cache;desc="Cache Read";dur=5, app;dur=123

const nav = performance.getEntriesByType('navigation')[0];
if (nav.serverTiming) {
  nav.serverTiming.forEach((timing) => {
    console.log(`${timing.name}: ${timing.duration}ms (${timing.description})`);
  });
  // db: 42ms ()
  // cache: 5ms (Cache Read)
  // app: 123ms ()
}
```

---

## 9. Service Worker とナビゲーション

### 9.1 Service Worker のライフサイクル

```
Service Worker のライフサイクル:

  ┌─────────────────────────────────────────────┐
  │ 1. Registration（登録）                       │
  │    navigator.serviceWorker.register('/sw.js')│
  │                                             │
  │ 2. Installation（インストール）                │
  │    → install イベント発火                     │
  │    → キャッシュの事前準備                      │
  │                                             │
  │ 3. Activation（有効化）                       │
  │    → activate イベント発火                    │
  │    → 古いキャッシュの削除                      │
  │                                             │
  │ 4. Controlling（制御中）                      │
  │    → fetch イベントでリクエストを傍受          │
  │    → ナビゲーションリクエストも制御可能         │
  └─────────────────────────────────────────────┘

  Service Worker によるナビゲーション制御:

  ブラウザ             Service Worker          ネットワーク
  │                   │                       │
  │ ── navigation ──→│                       │
  │                   │ fetch イベント発火      │
  │                   │                       │
  │                   │ キャッシュ確認          │
  │                   ├─ ヒット → レスポンス返却│
  │                   │                       │
  │                   ├─ ミス ─────────────→│ ネットワーク
  │                   │                     │ リクエスト
  │                   │←────── レスポンス ──│
  │←── レスポンス ───│                       │
  │                   │                       │
```

### 9.2 キャッシュ戦略の実装

```javascript
// sw.js - Service Worker
const CACHE_NAME = 'app-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/css/app.css',
  '/js/app.js',
  '/fonts/NotoSansJP-Regular.woff2',
  '/images/logo.svg',
];

// インストール時にキャッシュ
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting())
  );
});

// 有効化時に古いキャッシュを削除
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) =>
        Promise.all(
          names
            .filter((name) => name !== CACHE_NAME)
            .map((name) => caches.delete(name))
        )
      )
      .then(() => self.clients.claim())
  );
});

// フェッチ時のキャッシュ戦略
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // ナビゲーションリクエスト: Network First
  if (request.mode === 'navigate') {
    event.respondWith(networkFirstStrategy(request));
    return;
  }

  // 静的アセット: Cache First
  if (isStaticAsset(url)) {
    event.respondWith(cacheFirstStrategy(request));
    return;
  }

  // APIリクエスト: Stale While Revalidate
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(staleWhileRevalidateStrategy(request));
    return;
  }

  // その他: Network Only
  event.respondWith(fetch(request));
});

// Cache First: キャッシュ優先、なければネットワーク
async function cacheFirstStrategy(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
  }
  return response;
}

// Network First: ネットワーク優先、失敗したらキャッシュ
async function networkFirstStrategy(request) {
  try {
    const response = await fetch(request, { timeout: 3000 });
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch (error) {
    const cached = await caches.match(request);
    if (cached) return cached;

    // オフラインフォールバック
    return caches.match('/offline.html');
  }
}

// Stale While Revalidate: キャッシュを返しつつ裏で更新
async function staleWhileRevalidateStrategy(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);

  const fetchPromise = fetch(request)
    .then((response) => {
      if (response.ok) {
        cache.put(request, response.clone());
      }
      return response;
    })
    .catch(() => cached);

  return cached || fetchPromise;
}

function isStaticAsset(url) {
  return /\.(css|js|woff2?|png|jpg|webp|svg|ico)$/.test(url.pathname);
}
```

### 9.3 Navigation Preload

```javascript
// Navigation Preload: SW起動待ちの間にネットワークリクエストを開始
self.addEventListener('activate', (event) => {
  event.waitUntil(
    (async () => {
      // Navigation Preload を有効化
      if (self.registration.navigationPreload) {
        await self.registration.navigationPreload.enable();
      }
    })()
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      (async () => {
        try {
          // Navigation Preload のレスポンスを使用
          const preloadResponse = await event.preloadResponse;
          if (preloadResponse) {
            return preloadResponse;
          }

          // フォールバック：通常のネットワークリクエスト
          return await fetch(event.request);
        } catch (error) {
          // オフライン時はキャッシュを返す
          const cached = await caches.match(event.request);
          return cached || caches.match('/offline.html');
        }
      })()
    );
  }
});

/*
  Navigation Preload のメリット:

  Without Navigation Preload:
    SW起動(50ms) → fetch イベント → ネットワークリクエスト(200ms)
    合計: 250ms

  With Navigation Preload:
    SW起動(50ms)
    ネットワークリクエスト(200ms)  ← 並行して開始
    合計: 200ms（SW起動と並行）

  → 50-100ms の短縮効果
*/
```

---

## 10. SPA のナビゲーション

### 10.1 クライアントサイドナビゲーション

```javascript
// History API を使った SPA ナビゲーション
class SPARouter {
  constructor() {
    this.routes = new Map();
    this.currentPath = null;

    // ブラウザの戻る/進むボタン
    window.addEventListener('popstate', (event) => {
      this.navigate(location.pathname, false);
    });

    // リンクのクリックを傍受
    document.addEventListener('click', (event) => {
      const anchor = event.target.closest('a[href]');
      if (!anchor) return;

      const url = new URL(anchor.href);
      if (url.origin !== location.origin) return; // 外部リンクはスルー

      event.preventDefault();
      this.navigate(url.pathname);
    });
  }

  route(path, handler) {
    this.routes.set(path, handler);
    return this;
  }

  async navigate(path, pushState = true) {
    if (path === this.currentPath) return;

    // パフォーマンスマーク
    performance.mark('navigation-start');

    const handler = this.matchRoute(path);
    if (!handler) {
      console.warn(`No route for: ${path}`);
      return;
    }

    // 履歴に追加
    if (pushState) {
      history.pushState({ path }, '', path);
    }

    this.currentPath = path;

    // ページ遷移アニメーション
    const container = document.getElementById('app');
    container.classList.add('page-transitioning');

    try {
      const content = await handler(path);
      container.innerHTML = content;
    } finally {
      container.classList.remove('page-transitioning');
    }

    // パフォーマンス計測
    performance.mark('navigation-end');
    performance.measure('spa-navigation', 'navigation-start', 'navigation-end');

    const measure = performance.getEntriesByName('spa-navigation').pop();
    console.log(`SPA Navigation: ${measure.duration.toFixed(0)}ms`);

    // スクロール位置をリセット
    window.scrollTo(0, 0);

    // アナリティクスに送信
    this.trackPageView(path);
  }

  matchRoute(path) {
    // 完全一致
    if (this.routes.has(path)) return this.routes.get(path);

    // パラメータ付きルート
    for (const [pattern, handler] of this.routes) {
      const regex = new RegExp(
        '^' + pattern.replace(/:([^/]+)/g, '(?<$1>[^/]+)') + '$'
      );
      const match = path.match(regex);
      if (match) {
        return (p) => handler(p, match.groups);
      }
    }

    return null;
  }

  trackPageView(path) {
    // Soft Navigation API（Chrome 実験的機能）
    if (window.PerformanceObserver) {
      try {
        new PerformanceObserver((list) => {
          list.getEntries().forEach((entry) => {
            console.log('Soft navigation:', entry);
          });
        }).observe({ type: 'soft-navigation', buffered: true });
      } catch (e) {
        // Not supported
      }
    }
  }
}

// 使用例
const router = new SPARouter();
router
  .route('/', async () => {
    const data = await fetch('/api/home').then((r) => r.json());
    return renderHome(data);
  })
  .route('/products/:id', async (path, params) => {
    const data = await fetch(`/api/products/${params.id}`).then((r) =>
      r.json()
    );
    return renderProduct(data);
  });
```

### 10.2 View Transitions API

```javascript
// View Transitions API（Chrome 111+）
// SPA ナビゲーション時のスムーズなアニメーション

async function navigateWithTransition(url) {
  // View Transition 非対応ブラウザのフォールバック
  if (!document.startViewTransition) {
    await updateDOM(url);
    return;
  }

  // View Transition を開始
  const transition = document.startViewTransition(async () => {
    await updateDOM(url);
  });

  // トランジション完了を待つ
  await transition.finished;
}

async function updateDOM(url) {
  const response = await fetch(url);
  const html = await response.text();
  const parser = new DOMParser();
  const doc = parser.parseFromString(html, 'text/html');

  // メインコンテンツを置換
  document.querySelector('main').innerHTML =
    doc.querySelector('main').innerHTML;

  // タイトルを更新
  document.title = doc.title;
}
```

```css
/* View Transitions のカスタムアニメーション */

/* デフォルトのフェードイン/アウト */
::view-transition-old(root) {
  animation: fade-out 0.3s ease-out;
}

::view-transition-new(root) {
  animation: fade-in 0.3s ease-in;
}

@keyframes fade-out {
  from { opacity: 1; }
  to { opacity: 0; }
}

@keyframes fade-in {
  from { opacity: 0; }
  to { opacity: 1; }
}

/* 特定の要素にカスタムトランジション名を設定 */
.hero-image {
  view-transition-name: hero;
}

.page-title {
  view-transition-name: title;
}

/* 要素ごとのアニメーション */
::view-transition-old(hero) {
  animation: slide-out-left 0.4s ease-in;
}

::view-transition-new(hero) {
  animation: slide-in-right 0.4s ease-out;
}

@keyframes slide-out-left {
  from { transform: translateX(0); opacity: 1; }
  to { transform: translateX(-100%); opacity: 0; }
}

@keyframes slide-in-right {
  from { transform: translateX(100%); opacity: 0; }
  to { transform: translateX(0); opacity: 1; }
}
```

### 10.3 Speculation Rules API（プリレンダリング）

```html
<!-- Speculation Rules API: 次のナビゲーションを事前レンダリング -->
<script type="speculationrules">
{
  "prerender": [
    {
      "where": {
        "and": [
          { "href_matches": "/*" },
          { "not": { "href_matches": "/logout" } },
          { "not": { "href_matches": "/api/*" } },
          { "not": { "selector_matches": ".no-prerender" } }
        ]
      },
      "eagerness": "moderate"
    }
  ],
  "prefetch": [
    {
      "urls": ["/products", "/about"],
      "eagerness": "eager"
    }
  ]
}
</script>

<!--
  eagerness の種類:
  - "eager": 即座に実行
  - "moderate": ホバー時に実行（200msのインテントシグナル）
  - "conservative": クリック/タップ時に実行

  prefetch vs prerender:
  - prefetch: HTMLのみ取得（ネットワーク節約）
  - prerender: ページ全体を裏でレンダリング（瞬時表示）

  制限事項:
  - prerender は同一オリジンのみ
  - 1ページにつき prerender は最大10件
  - メモリ使用量に注意
-->
```

```javascript
// Speculation Rules を動的に追加
function addSpeculationRules(urls) {
  // 既存のルールを削除
  document
    .querySelectorAll('script[type="speculationrules"]')
    .forEach((el) => el.remove());

  const rules = {
    prerender: [
      {
        urls: urls,
        eagerness: 'moderate',
      },
    ],
  };

  const script = document.createElement('script');
  script.type = 'speculationrules';
  script.textContent = JSON.stringify(rules);
  document.head.appendChild(script);
}

// ユーザーの行動に基づいてプリレンダリング対象を決定
function predictNextNavigation() {
  // 最も確率の高いリンクを特定
  const links = Array.from(document.querySelectorAll('a[href^="/"]'));
  const visibleLinks = links.filter((link) => {
    const rect = link.getBoundingClientRect();
    return (
      rect.top >= 0 &&
      rect.top <= window.innerHeight &&
      rect.width > 0 &&
      rect.height > 0
    );
  });

  // ビューポート内のリンクをプリレンダリング候補に
  const urls = visibleLinks.slice(0, 3).map((link) => link.href);
  addSpeculationRules(urls);
}

// Intersection Observer でビューポート内リンクを監視
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        const href = entry.target.getAttribute('href');
        if (href) addSpeculationRules([href]);
      }
    });
  },
  { rootMargin: '200px' }
);

document.querySelectorAll('a[href^="/"]').forEach((link) => {
  observer.observe(link);
});
```

---

## 11. パフォーマンス最適化の実践

### 11.1 Critical Rendering Path の最適化

```html
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <!-- 1. DNS/接続の事前確立 -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://cdn.example.com" crossorigin>
  <link rel="dns-prefetch" href="//analytics.example.com">

  <!-- 2. クリティカルCSS（インライン） -->
  <style>
    /* Above-the-fold に必要な最小CSS */
    :root { --primary: #1a1a2e; --text: #333; }
    body { margin: 0; font-family: system-ui, sans-serif; color: var(--text); }
    .header { background: var(--primary); color: white; padding: 1rem; }
    .hero { min-height: 50vh; display: grid; place-items: center; }
  </style>

  <!-- 3. 重要フォントの事前読み込み -->
  <link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>

  <!-- 4. 非クリティカルCSS（非同期読み込み） -->
  <link rel="preload" href="/css/app.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/css/app.css"></noscript>

  <!-- 5. JavaScriptは defer で -->
  <script src="/js/vendor.js" defer></script>
  <script src="/js/app.js" defer></script>

  <!-- 6. 独立した解析系は async で -->
  <script src="/js/analytics.js" async></script>
</head>
<body>
  <header class="header">
    <nav>...</nav>
  </header>

  <main class="hero">
    <!-- 7. LCP候補の画像は高優先度 -->
    <img src="/images/hero.webp"
         alt="Hero Image"
         fetchpriority="high"
         width="1200"
         height="600"
         decoding="async">
  </main>

  <section class="products">
    <!-- 8. ファーストビュー外の画像はlazy -->
    <img src="/images/product-1.webp"
         alt="Product 1"
         loading="lazy"
         width="400"
         height="300"
         decoding="async">
  </section>

  <!-- 9. Speculation Rules -->
  <script type="speculationrules">
  {
    "prefetch": [
      { "where": { "href_matches": "/products/*" }, "eagerness": "moderate" }
    ]
  }
  </script>
</body>
</html>
```

### 11.2 ローディングパフォーマンスのチェックリスト

```
パフォーマンス最適化チェックリスト:

  ■ ネットワーク層
  □ HTTP/2 または HTTP/3 を使用
  □ CDN を利用（地理的に近いサーバーから配信）
  □ Brotli 圧縮を有効化
  □ preconnect で重要なドメインに事前接続
  □ dns-prefetch で外部ドメインを事前解決
  □ 不要なリダイレクトを削除

  ■ キャッシュ層
  □ 静的アセットに長いmax-age + immutable
  □ HTML に stale-while-revalidate
  □ Service Worker でオフライン対応
  □ ETag/Last-Modified で条件付きリクエスト
  □ CDN のキャッシュヒット率を監視

  ■ リソース層
  □ クリティカルCSS をインライン化
  □ 非クリティカルCSS を非同期読み込み
  □ JavaScript に defer/async を適用
  □ LCP 画像に fetchpriority="high"
  □ ファーストビュー外の画像に loading="lazy"
  □ 不要な JavaScript を削除（tree shaking）
  □ コード分割（dynamic import）

  ■ 画像/メディア層
  □ WebP/AVIF フォーマットを使用
  □ 適切なサイズの画像を配信（srcset）
  □ width/height 属性で CLS を防止
  □ 画像 CDN で自動最適化

  ■ フォント層
  □ WOFF2 フォーマットを使用
  □ font-display: swap/optional を設定
  □ preload でフォントを事前読み込み
  □ フォントサブセット化（日本語は特に重要）

  ■ JavaScript実行層
  □ Long Task を分割（50ms以下）
  □ requestIdleCallback で非重要処理を延期
  □ Web Worker でメインスレッドを解放
  □ Third-party スクリプトの影響を計測
```

### 11.3 Waterfall 分析の実践

```javascript
// Chrome DevTools の Network タブと同等の分析をコードで実装
class WaterfallAnalyzer {
  analyze() {
    const resources = performance.getEntriesByType('resource');
    const nav = performance.getEntriesByType('navigation')[0];

    // ウォーターフォールデータの生成
    const waterfall = resources.map((r) => ({
      name: this.getShortName(r.name),
      type: r.initiatorType,
      start: Math.round(r.startTime),
      end: Math.round(r.startTime + r.duration),
      duration: Math.round(r.duration),
      size: Math.round(r.transferSize / 1024),
      protocol: r.nextHopProtocol,

      // 各フェーズの内訳
      phases: {
        blocked: Math.round(r.fetchStart - r.startTime),
        dns: Math.round(r.domainLookupEnd - r.domainLookupStart),
        connect: Math.round(r.connectEnd - r.connectStart),
        tls:
          r.secureConnectionStart > 0
            ? Math.round(r.connectEnd - r.secureConnectionStart)
            : 0,
        waiting: Math.round(r.responseStart - r.requestStart),
        download: Math.round(r.responseEnd - r.responseStart),
      },
    }));

    // ボトルネックの特定
    const bottlenecks = this.findBottlenecks(waterfall);

    return { waterfall, bottlenecks };
  }

  findBottlenecks(waterfall) {
    const issues = [];

    waterfall.forEach((r) => {
      // DNS解決が遅い
      if (r.phases.dns > 50) {
        issues.push({
          resource: r.name,
          issue: `DNS resolution slow: ${r.phases.dns}ms`,
          recommendation: 'Add <link rel="dns-prefetch"> or <link rel="preconnect">',
        });
      }

      // TTFB が遅い
      if (r.phases.waiting > 200) {
        issues.push({
          resource: r.name,
          issue: `TTFB slow: ${r.phases.waiting}ms`,
          recommendation: 'Check server response time, consider CDN or caching',
        });
      }

      // ダウンロードが遅い（大きいファイル）
      if (r.phases.download > 500) {
        issues.push({
          resource: r.name,
          issue: `Download slow: ${r.phases.download}ms (${r.size}KB)`,
          recommendation: 'Enable compression, reduce file size, or use CDN',
        });
      }

      // ブロック時間が長い（HTTP/1.1の同時接続制限）
      if (r.phases.blocked > 100) {
        issues.push({
          resource: r.name,
          issue: `Blocked: ${r.phases.blocked}ms`,
          recommendation: 'Upgrade to HTTP/2, reduce number of requests',
        });
      }
    });

    return issues;
  }

  getShortName(url) {
    try {
      const u = new URL(url);
      return u.pathname.split('/').pop() || u.pathname;
    } catch {
      return url;
    }
  }

  // テキストベースのウォーターフォール表示
  printWaterfall() {
    const { waterfall, bottlenecks } = this.analyze();
    const maxEnd = Math.max(...waterfall.map((r) => r.end));
    const width = 60;

    console.log('=== Waterfall ===');
    console.log(`${'Resource'.padEnd(25)} ${'Timeline'.padEnd(width)} Duration`);

    waterfall.forEach((r) => {
      const startPos = Math.round((r.start / maxEnd) * width);
      const endPos = Math.round((r.end / maxEnd) * width);
      const barLen = Math.max(1, endPos - startPos);

      const bar =
        ' '.repeat(startPos) +
        '\u2588'.repeat(barLen) +
        ' '.repeat(width - startPos - barLen);

      console.log(
        `${r.name.substring(0, 24).padEnd(25)} ${bar} ${r.duration}ms`
      );
    });

    if (bottlenecks.length > 0) {
      console.log('\n=== Bottlenecks ===');
      bottlenecks.forEach((b) => {
        console.log(`${b.resource}: ${b.issue}`);
        console.log(`  → ${b.recommendation}`);
      });
    }
  }
}

// 使用例
window.addEventListener('load', () => {
  setTimeout(() => {
    const analyzer = new WaterfallAnalyzer();
    analyzer.printWaterfall();
  }, 1000);
});
```

---

## 12. 実務でのトラブルシューティング

### 12.1 よくあるローディング問題と対処法

```
問題1: TTFB が遅い（>600ms）
─────────────────────────────
原因:
  - サーバーの処理時間が長い（DB クエリ、API 呼び出し）
  - 地理的距離が遠い
  - SSL 証明書の検証に時間がかかる

対処法:
  - CDN を導入（エッジキャッシュ）
  - サーバーサイドキャッシュ（Redis, Memcached）
  - データベースクエリの最適化
  - HTTP/2 Server Push（または Early Hints 103）

問題2: LCP が遅い（>2.5s）
─────────────────────────
原因:
  - LCP 要素（画像/テキスト）の発見が遅い
  - CSS がレンダリングをブロック
  - Web フォントの読み込み待ち
  - JavaScript によるレンダリングブロック

対処法:
  - LCP 画像に preload + fetchpriority="high"
  - クリティカルCSS のインライン化
  - font-display: optional/swap
  - SSR/SSG でHTML内にコンテンツを含める

問題3: CLS が大きい（>0.1）
─────────────────────────
原因:
  - 画像にwidth/height 未指定
  - 動的に挿入される広告/バナー
  - Web フォントの FOUT（Flash of Unstyled Text）
  - 非同期で読み込まれるコンポーネント

対処法:
  - 全メディアに aspect-ratio または width/height 指定
  - 広告枠のプレースホルダーを確保
  - font-display: optional
  - コンテンツの挿入位置を固定（min-height）

問題4: JavaScript の読み込みが遅い
─────────────────────────────────
原因:
  - バンドルサイズが大きい（>200KB gzipped）
  - 全ページ共通で不要なコードまで読み込み
  - third-party スクリプトが多い

対処法:
  - コード分割（route-based code splitting）
  - tree shaking で未使用コード除去
  - dynamic import で遅延読み込み
  - third-party スクリプトの監査と削除
```

### 12.2 Lighthouse によるパフォーマンス監査

```javascript
// Lighthouse CLI の実行（Node.js）
const lighthouse = require('lighthouse');
const chromeLauncher = require('chrome-launcher');

async function runLighthouse(url) {
  const chrome = await chromeLauncher.launch({ chromeFlags: ['--headless'] });

  const result = await lighthouse(url, {
    port: chrome.port,
    output: 'json',
    onlyCategories: ['performance'],
    settings: {
      // モバイルシミュレーション
      formFactor: 'mobile',
      throttling: {
        rttMs: 150, // RTT
        throughputKbps: 1638.4, // 1.6 Mbps
        cpuSlowdownMultiplier: 4, // CPU 4x slowdown
      },
      screenEmulation: {
        mobile: true,
        width: 412,
        height: 823,
        deviceScaleFactor: 1.75,
      },
    },
  });

  const report = JSON.parse(result.report);
  const audits = report.audits;

  console.log('=== Performance Score ===');
  console.log(`Score: ${report.categories.performance.score * 100}`);

  console.log('\n=== Core Web Vitals ===');
  console.log(`FCP: ${audits['first-contentful-paint'].displayValue}`);
  console.log(`LCP: ${audits['largest-contentful-paint'].displayValue}`);
  console.log(`TBT: ${audits['total-blocking-time'].displayValue}`);
  console.log(`CLS: ${audits['cumulative-layout-shift'].displayValue}`);
  console.log(`SI:  ${audits['speed-index'].displayValue}`);

  console.log('\n=== Opportunities ===');
  const opportunities = Object.values(audits).filter(
    (a) => a.details?.type === 'opportunity' && a.details?.overallSavingsMs > 0
  );

  opportunities
    .sort((a, b) => b.details.overallSavingsMs - a.details.overallSavingsMs)
    .forEach((opp) => {
      console.log(
        `${opp.title}: ~${Math.round(opp.details.overallSavingsMs)}ms savings`
      );
    });

  await chrome.kill();
  return report;
}

// 使用例
runLighthouse('https://example.com');
```

### 12.3 Real User Monitoring (RUM) の実装

```javascript
// 本番環境での RUM データ収集
class RUMCollector {
  constructor(endpoint) {
    this.endpoint = endpoint;
    this.data = {
      url: location.href,
      userAgent: navigator.userAgent,
      connection: this.getConnectionInfo(),
      timestamp: Date.now(),
      metrics: {},
    };
  }

  getConnectionInfo() {
    const conn =
      navigator.connection ||
      navigator.mozConnection ||
      navigator.webkitConnection;
    if (!conn) return null;

    return {
      effectiveType: conn.effectiveType, // "4g", "3g", "2g", "slow-2g"
      downlink: conn.downlink, // Mbps
      rtt: conn.rtt, // ms
      saveData: conn.saveData, // boolean
    };
  }

  collectNavigationTiming() {
    const nav = performance.getEntriesByType('navigation')[0];
    if (!nav) return;

    this.data.metrics.navigation = {
      type: nav.type,
      protocol: nav.nextHopProtocol,
      redirectCount: nav.redirectCount,
      dns: Math.round(nav.domainLookupEnd - nav.domainLookupStart),
      tcp: Math.round(nav.connectEnd - nav.connectStart),
      ttfb: Math.round(nav.responseStart - nav.requestStart),
      download: Math.round(nav.responseEnd - nav.responseStart),
      domContentLoaded: Math.round(nav.domContentLoadedEventEnd),
      load: Math.round(nav.loadEventEnd),
      transferSize: nav.transferSize,
    };
  }

  collectWebVitals() {
    // LCP
    new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      this.data.metrics.lcp = {
        value: Math.round(lastEntry.startTime),
        element: lastEntry.element?.tagName,
        url: lastEntry.url,
      };
    }).observe({ type: 'largest-contentful-paint', buffered: true });

    // CLS
    let clsValue = 0;
    let clsEntries = [];
    new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (!entry.hadRecentInput) {
          clsValue += entry.value;
          clsEntries.push({
            value: entry.value,
            sources: entry.sources?.map((s) => ({
              node: s.node?.tagName,
              previousRect: s.previousRect,
              currentRect: s.currentRect,
            })),
          });
        }
      });
      this.data.metrics.cls = { value: clsValue, entries: clsEntries };
    }).observe({ type: 'layout-shift', buffered: true });

    // INP
    let maxINP = 0;
    new PerformanceObserver((list) => {
      list.getEntries().forEach((entry) => {
        if (entry.duration > maxINP) {
          maxINP = entry.duration;
          this.data.metrics.inp = {
            value: entry.duration,
            type: entry.name,
            target: entry.target?.tagName,
          };
        }
      });
    }).observe({ type: 'event', buffered: true, durationThreshold: 16 });
  }

  send() {
    this.collectNavigationTiming();

    // ページ離脱時に送信
    const sendData = () => {
      const blob = new Blob([JSON.stringify(this.data)], {
        type: 'application/json',
      });
      navigator.sendBeacon(this.endpoint, blob);
    };

    // visibilitychange を優先（pagehide のフォールバック）
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') {
        sendData();
      }
    });

    // iOS Safari 対応
    window.addEventListener('pagehide', sendData);
  }
}

// 使用例
const rum = new RUMCollector('/api/rum');
rum.collectWebVitals();
rum.send();
```

---

## 13. Early Hints (103) と Server Push の比較

### 13.1 HTTP 103 Early Hints

```
103 Early Hints の仕組み:

  クライアント                    サーバー
  │                              │
  │ ── GET /index.html ────────→│
  │                              │ サーバー処理開始
  │                              │ （DBクエリ等に 300ms）
  │                              │
  │←── 103 Early Hints ─────────│  ← サーバー処理中に先行返却！
  │    Link: </style.css>; rel=preload; as=style
  │    Link: </app.js>; rel=preload; as=script
  │    Link: <https://cdn.example.com>; rel=preconnect
  │                              │
  │  CSS/JS ダウンロード開始       │ サーバーまだ処理中...
  │  ↓↓↓ 並行ダウンロード ↓↓↓    │
  │                              │ サーバー処理完了
  │←── 200 OK ──────────────────│
  │    <html>...                 │
  │                              │
  │  CSS/JS → すでにダウンロード済み！

  メリット:
  → サーバーの処理待ち時間を有効活用
  → TTFB が長い場合に特に効果的
  → 100-300ms の改善が期待できる

  設定例（Nginx）:
```

```nginx
# Nginx で 103 Early Hints
location / {
    # 103 Early Hints を返す
    add_header Link "</css/app.css>; rel=preload; as=style" early;
    add_header Link "</js/app.js>; rel=preload; as=script" early;
    add_header Link "<https://fonts.googleapis.com>; rel=preconnect" early;

    # バックエンドにプロキシ
    proxy_pass http://backend;
}
```

```javascript
// Node.js (Express) での 103 Early Hints
const express = require('express');
const app = express();

app.get('/', (req, res) => {
  // 103 Early Hints を先行送信
  res.writeEarlyHints({
    link: [
      '</css/app.css>; rel=preload; as=style',
      '</js/app.js>; rel=preload; as=script',
      '<https://cdn.example.com>; rel=preconnect',
    ],
  });

  // 通常のレスポンス処理（DBクエリ等）
  const data = await fetchDataFromDB();

  res.render('index', { data });
});
```

---

## 14. 画像の最適化とローディング戦略

### 14.1 レスポンシブ画像の配信

```html
<!-- srcset + sizes による最適な画像配信 -->
<img
  src="/images/hero-800.webp"
  srcset="
    /images/hero-400.webp 400w,
    /images/hero-800.webp 800w,
    /images/hero-1200.webp 1200w,
    /images/hero-1600.webp 1600w
  "
  sizes="(max-width: 600px) 100vw,
         (max-width: 1200px) 50vw,
         800px"
  alt="Hero Image"
  width="1200"
  height="600"
  fetchpriority="high"
  decoding="async"
>

<!-- picture要素によるフォーマット分岐 -->
<picture>
  <!-- AVIF（最も効率的、対応ブラウザ限定） -->
  <source
    type="image/avif"
    srcset="/images/hero-400.avif 400w,
           /images/hero-800.avif 800w,
           /images/hero-1200.avif 1200w"
    sizes="(max-width: 600px) 100vw, 800px"
  >
  <!-- WebP（広くサポート） -->
  <source
    type="image/webp"
    srcset="/images/hero-400.webp 400w,
           /images/hero-800.webp 800w,
           /images/hero-1200.webp 1200w"
    sizes="(max-width: 600px) 100vw, 800px"
  >
  <!-- フォールバック（JPEG） -->
  <img
    src="/images/hero-800.jpg"
    alt="Hero Image"
    width="1200"
    height="600"
    loading="eager"
    decoding="async"
  >
</picture>
```

### 14.2 画像の遅延読み込みパターン

```javascript
// Native lazy loading + Intersection Observer のハイブリッド戦略
class ImageLazyLoader {
  constructor(options = {}) {
    this.rootMargin = options.rootMargin || '200px 0px';
    this.threshold = options.threshold || 0.01;
    this.loaded = new Set();

    // Native lazy loading 対応チェック
    this.supportsNativeLazy = 'loading' in HTMLImageElement.prototype;

    if (!this.supportsNativeLazy) {
      this.setupIntersectionObserver();
    }
  }

  setupIntersectionObserver() {
    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            this.loadImage(entry.target);
            this.observer.unobserve(entry.target);
          }
        });
      },
      {
        rootMargin: this.rootMargin,
        threshold: this.threshold,
      }
    );

    // data-src 属性を持つ画像を監視
    document.querySelectorAll('img[data-src]').forEach((img) => {
      this.observer.observe(img);
    });
  }

  loadImage(img) {
    const src = img.dataset.src;
    const srcset = img.dataset.srcset;

    if (src) {
      img.src = src;
      img.removeAttribute('data-src');
    }
    if (srcset) {
      img.srcset = srcset;
      img.removeAttribute('data-srcset');
    }

    img.classList.add('loaded');
    this.loaded.add(src);
  }
}

// 使用例
const lazyLoader = new ImageLazyLoader({ rootMargin: '300px 0px' });
```

---

## 15. 高度なプリロード戦略

### 15.1 リソースヒントの総合ガイド

```html
<!--
  リソースヒントの完全ガイド:

  ┌──────────────────┬───────────────────┬──────────────┬──────────┐
  │ ヒント           │ 動作               │ コスト       │ 用途     │
  ├──────────────────┼───────────────────┼──────────────┼──────────┤
  │ dns-prefetch     │ DNS解決のみ        │ 極低         │ 外部ドメイン│
  │ preconnect       │ DNS+TCP+TLS       │ 低           │ 確実に使う │
  │ preload          │ リソースをDL       │ 中           │ 現ページ  │
  │ prefetch         │ 将来のリソースをDL  │ 低(idle時)   │ 次ページ  │
  │ modulepreload    │ ESModuleをDL+parse│ 中           │ JSモジュール│
  │ prerender        │ ページ全体をレンダ  │ 高           │ 次ページ  │
  └──────────────────┴───────────────────┴──────────────┴──────────┘
-->

<!-- dns-prefetch: とにかく外部ドメインには付ける -->
<link rel="dns-prefetch" href="//analytics.google.com">
<link rel="dns-prefetch" href="//fonts.gstatic.com">
<link rel="dns-prefetch" href="//api.stripe.com">

<!-- preconnect: 確実に使う重要なオリジン（3-5個まで） -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://cdn.example.com" crossorigin>

<!-- preload: 現在のページで確実に必要なリソース -->
<link rel="preload" href="/fonts/main.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/css/critical.css" as="style">
<link rel="preload" href="/images/hero.webp" as="image" type="image/webp"
      imagesrcset="/images/hero-400.webp 400w, /images/hero-800.webp 800w"
      imagesizes="100vw">

<!-- prefetch: 次のナビゲーションで必要になるリソース -->
<link rel="prefetch" href="/js/product-page.js">
<link rel="prefetch" href="/api/popular-products" as="fetch" crossorigin>

<!-- modulepreload: ESModuleの事前読み込み -->
<link rel="modulepreload" href="/js/modules/cart.mjs">
<link rel="modulepreload" href="/js/modules/auth.mjs">
```

### 15.2 Priority Hints の実践

```javascript
// Fetch Priority API の活用
// 重要なAPIリクエストの優先度を制御

// 高優先度: ユーザーが待っているデータ
const criticalData = await fetch('/api/user-profile', {
  priority: 'high',
});

// 低優先度: バックグラウンドでの事前取得
const prefetchData = await fetch('/api/recommendations', {
  priority: 'low',
});

// 画像の優先度制御
function loadImage(src, priority = 'auto') {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.fetchPriority = priority;
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// LCP 画像を高優先度で読み込み
await loadImage('/images/hero.webp', 'high');

// デコレーション画像を低優先度で読み込み
await loadImage('/images/background-pattern.webp', 'low');
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ナビゲーション | DNS→TCP→TLS→HTTP→パース→レンダリング |
| パーサーブロック | `<script>`がHTMLパースを停止 |
| 解決策 | defer/async/preload/preconnect |
| 優先順位 | CSS=最高、画像=viewport依存 |
| イベント | DOMContentLoaded(DOM完了) vs load(全完了) |
| Core Web Vitals | LCP(≤2.5s), INP(≤200ms), CLS(≤0.1) |
| Service Worker | オフライン対応、キャッシュ戦略 |
| HTTP/2・HTTP/3 | マルチプレキシング、0-RTT接続 |
| Early Hints | サーバー処理待ち中にリソース先読み |
| Speculation Rules | 次のナビゲーションを事前レンダリング |
| RUM | 実ユーザーのパフォーマンスデータ収集 |
| 画像最適化 | WebP/AVIF、srcset、lazy loading |

---

## 次に読むべきガイド
→ [[02-parsing-html-css.md]] -- HTML/CSSパーシング

---

## 参考文献
1. Mariko Kosaka. "Inside look at modern web browser (Part 2)." Google, 2018.
2. web.dev. "Optimizing resource loading." Google, 2024.
3. web.dev. "Core Web Vitals." Google, 2024.
4. MDN Web Docs. "Navigation Timing API." Mozilla, 2024.
5. MDN Web Docs. "Resource Timing API." Mozilla, 2024.
6. IETF. "RFC 9110: HTTP Semantics." 2022.
7. IETF. "RFC 9114: HTTP/3." 2022.
8. web.dev. "Speculation Rules API." Google, 2024.
9. Chrome Developers. "Early Hints." Google, 2023.
10. W3C. "Navigation Timing Level 2." 2023.
