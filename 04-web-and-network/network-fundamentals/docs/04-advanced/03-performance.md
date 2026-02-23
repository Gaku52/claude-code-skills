# ネットワーク最適化

> Webアプリケーションのネットワークパフォーマンスを最適化する。レイテンシ削減、帯域最適化、接続管理、圧縮、プリロード等の手法を体系的に学び、高速なユーザー体験を実現する。

## この章で学ぶこと

- [ ] ネットワークパフォーマンスのボトルネックを理解する
- [ ] レイテンシ削減と帯域最適化の手法を把握する
- [ ] 接続管理とリソースの最適化戦略を学ぶ
- [ ] HTTP/2・HTTP/3 による多重化と高速接続を理解する
- [ ] キャッシュ戦略とService Workerを活用したオフライン対応を学ぶ
- [ ] Core Web Vitals の計測と改善手法を実践する
- [ ] CDN設計とエッジコンピューティングの活用を把握する

---

## 1. パフォーマンスのボトルネック

### 1.1 Webページ読み込みの時間内訳

```
Webページ読み込みの時間内訳:

  DNS解決:        ~50ms
  TCP接続:        ~30ms（1.5 RTT）
  TLSハンドシェイク: ~50ms（1-2 RTT）
  リクエスト送信:  ~5ms
  サーバー処理:    ~100ms（TTFB）
  コンテンツ転送:  ~200ms
  レンダリング:    ~300ms
  ─────────────────────────
  合計:            ~735ms

主要なボトルネック:
  ① レイテンシ（往復時間）:
     → 物理的距離に依存（光速の限界）
     → RTT × ラウンドトリップ回数

  ② 帯域幅:
     → 大きなファイルの転送時間
     → 画像、動画、JavaScriptバンドル

  ③ サーバー処理時間:
     → DB クエリ、API呼び出し、計算処理

  ④ レンダリング:
     → DOM構築、CSS計算、JavaScript実行
     → レンダリングブロッキング リソース
```

### 1.2 ネットワークウォーターフォールの詳細分析

```
典型的なWebページのウォーターフォール分析:

  リクエスト1: HTML (index.html)
  ├── DNS  │ TCP │ TLS │ TTFB │ Download
  │   50ms │30ms │50ms │100ms │ 20ms     = 250ms
  │
  ├── リクエスト2: CSS (styles.css) - レンダリングブロック
  │   ├── TTFB │ Download
  │   │   30ms │ 40ms     = 70ms (接続再利用)
  │   │
  │   ├── リクエスト3: JS (app.js) - パーサーブロック
  │   │   ├── TTFB │ Download │ Parse │ Execute
  │   │   │   30ms │ 100ms   │ 50ms  │ 200ms  = 380ms
  │   │   │
  │   │   ├── リクエスト4: API (GET /api/data) - JSから発火
  │   │   │   ├── DNS │ TCP │ TLS │ TTFB │ Download
  │   │   │   │   50ms│30ms│50ms │150ms │ 30ms   = 310ms
  │   │   │   │
  │   │   │   └── リクエスト5: 画像 (hero.webp) - APIデータ後に表示
  │   │   │       ├── TTFB │ Download
  │   │   │       │   20ms │ 80ms     = 100ms
  │
  合計クリティカルパス: 250 + 70 + 380 + 310 + 100 = 1,110ms

  最適化後のクリティカルパス:
  ├── preconnect: API の DNS+TCP+TLS を事前解決 (-130ms)
  ├── preload: CSS と JS を並列取得 (-70ms)
  ├── async/defer: JS のパーサーブロック解除 (-50ms)
  ├── SSR: API 呼び出し不要 (-310ms)
  ├── priority hints: hero画像を優先ロード
  └── 結果: 1,110ms → ~550ms（50%削減）
```

### 1.3 ブラウザの同時接続数制限

```
ブラウザの同時接続数制限:

  HTTP/1.1:
  ┌────────────────┬────────────────────┐
  │ ブラウザ        │ 同時接続数/ホスト   │
  ├────────────────┼────────────────────┤
  │ Chrome         │ 6                   │
  │ Firefox        │ 6                   │
  │ Safari         │ 6                   │
  │ Edge           │ 6                   │
  └────────────────┴────────────────────┘

  → 7番目以降のリクエストはキューで待機
  → ドメインシャーディング: img1.example.com, img2.example.com
    → HTTP/1.1 時代のワークアラウンド（非推奨）

  HTTP/2:
  → 1つの接続で多数のストリームを多重化
  → 同時接続数制限は実質的に無制限
  → ドメインシャーディングは逆効果（接続確立コスト）

  HTTP/3:
  → QUICベースで接続確立が高速
  → Head-of-Line Blocking が解消
  → パケットロス時のパフォーマンスが向上
```

### 1.4 クリティカルレンダリングパス

```
クリティカルレンダリングパス（CRP）:

  HTML → DOM Tree
    ↓
  CSS → CSSOM Tree
    ↓
  DOM + CSSOM → Render Tree
    ↓
  Layout（レイアウト計算）
    ↓
  Paint（画面描画）
    ↓
  Composite（合成）

  レンダリングブロッキングリソース:
  ① CSS（全てのCSSがパース完了するまでレンダリングしない）
     対策: Critical CSS のインライン化
           非クリティカルCSSの非同期読み込み

  ② JavaScript（script タグがDOMパースをブロック）
     対策: async / defer 属性
           script タグを body 末尾に配置
           dynamic import

  Critical CSS の例:
  <!-- head 内にインライン化 -->
  <style>
    /* ファーストビューに必要な最小限のCSS */
    body { margin: 0; font-family: sans-serif; }
    .hero { height: 100vh; display: flex; align-items: center; }
    .nav { position: fixed; top: 0; width: 100%; }
  </style>

  <!-- 残りのCSSは非同期ロード -->
  <link rel="preload" href="/styles.css" as="style"
        onload="this.onload=null;this.rel='stylesheet'">
  <noscript><link rel="stylesheet" href="/styles.css"></noscript>
```

---

## 2. レイテンシ削減

### 2.1 CDN の活用

```
CDN（Content Delivery Network）の活用:

  基本原理:
  → ユーザーに近いエッジサーバーから配信
  → RTT: 100ms → 5ms（東京のユーザーが東京エッジから取得）
  → オリジンサーバーへの負荷軽減

  CDNの階層構造:
  ┌──────────────────────────────────┐
  │ オリジンサーバー (us-east-1)       │
  └──────────────┬───────────────────┘
                 │
    ┌────────────┼────────────────┐
    │            │                │
  ┌─┴──┐    ┌──┴───┐     ┌─────┴──┐
  │東京 │    │ロンドン│     │サンパウロ│  ← エッジサーバー
  │PoP  │    │PoP   │     │PoP    │
  └──┬──┘    └──┬───┘     └───┬───┘
     │          │              │
   ユーザーA   ユーザーB      ユーザーC

  主要CDNプロバイダの比較:
  ┌──────────────┬─────────────┬──────────────────────────┐
  │ プロバイダ    │ エッジ数     │ 特徴                      │
  ├──────────────┼─────────────┼──────────────────────────┤
  │ CloudFront   │ 400+        │ AWS統合、Lambda@Edge      │
  │ Cloudflare   │ 300+        │ 無料プラン、Workers        │
  │ Fastly       │ 90+         │ VCL、リアルタイムパージ    │
  │ Akamai       │ 4,000+      │ 最大規模、エンタープライズ │
  │ Vercel Edge  │ 自動        │ Next.js最適化             │
  └──────────────┴─────────────┴──────────────────────────┘
```

### 2.2 CDN キャッシュ戦略

```nginx
# CDNキャッシュ戦略の設計

# 1. 静的アセット（変更時はファイル名が変わる: contenthash）
# /assets/app.a1b2c3.js
location ~* \.(js|css|woff2|png|jpg|webp|avif|svg)$ {
    add_header Cache-Control "public, max-age=31536000, immutable";
    # immutable: ブラウザが条件付きリクエストも送らない
    # 1年間キャッシュ（ファイル名にハッシュが含まれるため安全）
}

# 2. HTML（常に最新チェック）
location ~* \.html$ {
    add_header Cache-Control "public, no-cache";
    # no-cache: 毎回サーバーに検証（ETagで304応答）
    # → HTMLが最新のアセットURLを参照するため必要
}

# 3. API レスポンス（短時間キャッシュ + SWR）
location /api/ {
    add_header Cache-Control "public, max-age=10, stale-while-revalidate=60";
    # 10秒間はキャッシュ使用
    # 10-70秒: staleキャッシュを返しつつバックグラウンドで再検証
}

# 4. ユーザー固有データ（キャッシュ禁止）
location /api/me {
    add_header Cache-Control "private, no-store";
    # CDNにキャッシュさせない
}
```

```
CloudFront のキャッシュ動作:

  リクエストフロー:
  1. ユーザー → エッジ: GET /api/products
  2. エッジ: キャッシュにあるか？
     ├── HIT: キャッシュからレスポンス（<1ms）
     ├── MISS: オリジンにリクエスト
     │   → レスポンスをキャッシュ + ユーザーに返却
     └── STALE: 古いキャッシュを返しつつ再検証

  キャッシュヒット率の目標:
  静的アセット: > 95%
  API: > 70%（コンテンツによる）
  HTML: > 50%

  キャッシュキーの設計:
  → URL + Query String + Accept-Encoding + Accept(画像形式)
  → 不要なヘッダーやCookieをキャッシュキーから除外
  → Vary ヘッダーの適切な設定
```

### 2.3 接続の事前確立

```html
<!-- 接続の事前確立 -->

<!-- 1. dns-prefetch: DNSだけ事前解決（軽量） -->
<link rel="dns-prefetch" href="//api.example.com">
<link rel="dns-prefetch" href="//cdn.example.com">
<link rel="dns-prefetch" href="//fonts.googleapis.com">

<!-- 2. preconnect: DNS + TCP + TLS を事前確立（推奨） -->
<link rel="preconnect" href="https://api.example.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<!-- crossorigin: CORS リクエストの場合に必要 -->

<!-- 3. preload: リソースを優先ダウンロード -->
<link rel="preload" href="/fonts/inter.woff2" as="font" type="font/woff2" crossorigin>
<link rel="preload" href="/critical.css" as="style">
<link rel="preload" href="/hero.webp" as="image">

<!-- 4. prefetch: 次のページで必要なリソースを事前取得（低優先度） -->
<link rel="prefetch" href="/next-page.html">
<link rel="prefetch" href="/data/products.json">

<!-- 5. prerender: 次のページを事前レンダリング（Chrome） -->
<link rel="prerender" href="/likely-next-page">

<!-- 6. modulepreload: ESモジュールの事前ロード -->
<link rel="modulepreload" href="/modules/app.js">
```

```
リソースヒントの効果測定:

  preconnect の効果:
  ┌─────────────────────┬────────────┬────────────┐
  │ 操作                 │ preconnect │ なし       │
  ├─────────────────────┼────────────┼────────────┤
  │ DNS解決              │ 事前完了    │ 50ms       │
  │ TCP接続              │ 事前完了    │ 30ms       │
  │ TLSハンドシェイク     │ 事前完了    │ 50ms       │
  │ 最初のリクエスト      │ 即座       │ +130ms     │
  └─────────────────────┴────────────┴────────────┘

  preconnect は最大3-4個のドメインに限定する:
  → 過剰な preconnect は CPU/メモリを消費
  → 最も重要な外部ドメインのみに使用
```

### 2.4 HTTP/2 と HTTP/3

```
HTTP/2 の最適化:

  主要機能:
  ① 多重化（Multiplexing）:
     → 1つのTCP接続で複数のストリームを並行処理
     → リクエスト/レスポンスの順序に依存しない
     → HTTP/1.1 の Head-of-Line Blocking を解消

  ② ヘッダー圧縮（HPACK）:
     → 静的テーブル + 動的テーブル
     → 繰り返しのヘッダーは1バイトで参照
     → Cookie等の大きなヘッダーの圧縮に効果大

  ③ サーバープッシュ:
     → サーバーがクライアントの要求前にリソースを送信
     → HTML を返す際に CSS/JS も一緒にプッシュ
     → 注意: ブラウザキャッシュとの競合で非推奨の方向
     → 代替: 103 Early Hints

  ④ ストリーム優先度:
     → CSS > JS > 画像 の優先順位を設定
     → ブラウザが自動的に最適な優先度を設定

HTTP/3 (QUIC) の利点:

  HTTP/2 の問題（TCP Head-of-Line Blocking）:
  → TCPレベルでパケットロスが発生すると全ストリームが停止
  → HTTP/2 でも TCP の制約は回避不可能

  HTTP/3 の解決策:
  → QUIC（UDP上のトランスポートプロトコル）を使用
  → ストリーム単位での独立した制御
  → 1つのストリームのパケットロスが他に影響しない

  接続確立の高速化:
  ┌─────────────┬───────┬───────────────┐
  │ プロトコル    │ RTT   │ 0-RTT再接続    │
  ├─────────────┼───────┼───────────────┤
  │ HTTP/1.1+TLS│ 3 RTT │ N/A            │
  │ HTTP/2+TLS  │ 2 RTT │ N/A            │
  │ HTTP/3+QUIC │ 1 RTT │ 0 RTT（再接続時）│
  └─────────────┴───────┴───────────────┘

  コネクションマイグレーション:
  → Wi-Fi → モバイル回線 切り替え時も接続維持
  → Connection ID で接続を識別（IPアドレスに依存しない）
```

```nginx
# Nginx でのHTTP/2設定
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate /etc/ssl/certs/example.com.pem;
    ssl_certificate_key /etc/ssl/private/example.com.key;

    # ALPN（Application-Layer Protocol Negotiation）
    ssl_protocols TLSv1.2 TLSv1.3;

    # HTTP/2 サーバープッシュ（非推奨だが参考）
    # location / {
    #     http2_push /styles.css;
    #     http2_push /app.js;
    # }

    # 103 Early Hints（代替手段）
    location / {
        add_header Link "</styles.css>; rel=preload; as=style" always;
        add_header Link "</app.js>; rel=preload; as=script" always;
    }
}

# Nginx でのHTTP/3設定 (nginx-quic)
server {
    listen 443 quic reuseport;
    listen 443 ssl http2;

    ssl_protocols TLSv1.3;  # HTTP/3 は TLS 1.3 必須

    add_header Alt-Svc 'h3=":443"; ma=86400';
    # Alt-Svc: ブラウザに HTTP/3 対応を通知
}
```

### 2.5 接続の再利用と接続プーリング

```
接続の再利用:

  HTTP/1.1 Keep-Alive:
  → Connection: keep-alive（デフォルト）
  → TCP + TLS のハンドシェイクコストを回避
  → Keep-Alive-Timeout でアイドル時間を設定

  Nginx 設定:
  keepalive_timeout 65;        # クライアント接続のタイムアウト
  keepalive_requests 1000;     # 1接続あたりの最大リクエスト数

  upstream backend {
      server app:3000;
      keepalive 32;             # バックエンドへのKeep-Alive接続プール
      keepalive_requests 100;
      keepalive_timeout 60s;
  }

  サーバー側の接続プーリング:
  ┌──────────┐     Keep-Alive      ┌──────────┐
  │ ブラウザ  │ ←────────────────→ │   Nginx  │
  └──────────┘                      └─────┬────┘
                                          │ Connection Pool
                                    ┌─────┴────┐
                                    │  App(N)  │
                                    │  DB Pool │
                                    │  Redis   │
                                    └──────────┘

  データベース接続プーリング:
  → PgBouncer: PostgreSQL の接続プーリング
  → 接続確立: ~50ms → プールから取得: ~0.5ms
  → Transaction pooling: トランザクション単位で接続を再利用
```

### 2.6 サーバーの地理的分散

```
マルチリージョンデプロイ:

  構成例:
  ┌──────────────────────────────────────────────────┐
  │              Route 53 / Cloudflare DNS            │
  │        レイテンシベースルーティング                  │
  └──────────┬─────────────┬─────────────┬───────────┘
             │             │             │
    ┌────────┴───┐  ┌──────┴────┐  ┌────┴────────┐
    │ us-east-1  │  │ eu-west-1 │  │ ap-northeast│
    │ Virginia   │  │ Ireland   │  │ Tokyo       │
    ├────────────┤  ├───────────┤  ├─────────────┤
    │ App Server │  │App Server │  │ App Server  │
    │ Read DB    │  │ Read DB   │  │ Read DB     │
    └────────────┘  └───────────┘  └─────────────┘
             │             │             │
             └─────────────┼─────────────┘
                    ┌──────┴──────┐
                    │ Primary DB  │
                    │ (us-east-1) │
                    └─────────────┘

  レイテンシ比較（東京ユーザー）:
  ┌──────────────┬──────────┐
  │ サーバー位置  │ RTT      │
  ├──────────────┼──────────┤
  │ 東京         │ ~5ms     │
  │ シンガポール  │ ~70ms    │
  │ バージニア    │ ~170ms   │
  │ ロンドン      │ ~250ms   │
  └──────────────┴──────────┘

  データ同期戦略:
  → 最終整合性モデル（Eventually Consistent）
  → CRDTs（Conflict-free Replicated Data Types）
  → リードレプリカ + ライトリーダーパターン
```

---

## 3. 帯域最適化

### 3.1 テキスト圧縮

```
テキスト圧縮の比較:

  圧縮アルゴリズム:
  ┌──────────┬──────────┬──────────┬──────────────────┐
  │ 方式      │ 圧縮率    │ 圧縮速度  │ サポート          │
  ├──────────┼──────────┼──────────┼──────────────────┤
  │ gzip     │ 70-80%   │ 高速     │ 全ブラウザ        │
  │ Brotli   │ 80-90%   │ 中速     │ 全モダンブラウザ   │
  │ zstd     │ 75-85%   │ 最高速   │ Chrome 123+      │
  └──────────┴──────────┴──────────┴──────────────────┘

  実測値（React アプリの main.js: 500KB）:
  ┌──────────┬────────────┬──────────────┐
  │ 方式      │ 圧縮後サイズ │ 削減率        │
  ├──────────┼────────────┼──────────────┤
  │ 未圧縮    │ 500KB      │ -            │
  │ gzip     │ 145KB      │ 71%          │
  │ Brotli   │ 120KB      │ 76%          │
  │ zstd     │ 130KB      │ 74%          │
  └──────────┴────────────┴──────────────┘
```

```nginx
# Nginx でのBrotli圧縮設定
# ngx_brotli モジュールが必要

# 動的圧縮（リクエスト時に圧縮）
brotli on;
brotli_comp_level 6;    # 1-11（6が速度と圧縮率のバランス）
brotli_types
    text/plain
    text/css
    text/javascript
    application/json
    application/javascript
    application/x-javascript
    application/xml
    image/svg+xml;
brotli_min_length 1024;  # 1KB未満は圧縮しない

# 静的事前圧縮（ビルド時に .br ファイルを生成）
brotli_static on;
# → app.js.br が存在すれば、動的圧縮せずそのまま返す
# → ビルド時に brotli app.js で生成

# gzip フォールバック
gzip on;
gzip_comp_level 6;
gzip_types text/plain text/css application/json application/javascript;
gzip_min_length 1024;
gzip_static on;
```

```javascript
// Vite でのビルド時圧縮設定
// vite.config.ts
import { defineConfig } from 'vite';
import viteCompression from 'vite-plugin-compression';

export default defineConfig({
  plugins: [
    // Brotli 圧縮
    viteCompression({
      algorithm: 'brotliCompress',
      ext: '.br',
      threshold: 1024,     // 1KB以上のファイルを圧縮
    }),
    // gzip フォールバック
    viteCompression({
      algorithm: 'gzip',
      ext: '.gz',
      threshold: 1024,
    }),
  ],
  build: {
    // ソースマップはプロダクションでは無効化
    sourcemap: false,
    // チャンクサイズ警告の閾値
    chunkSizeWarningLimit: 500,
    rollupOptions: {
      output: {
        // ベンダーチャンクの分離
        manualChunks: {
          vendor: ['react', 'react-dom'],
          router: ['react-router-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
        },
      },
    },
  },
});
```

### 3.2 画像最適化

```
画像フォーマットの選択:

  フォーマット比較（同品質のベンチマーク）:
  ┌────────┬───────────┬──────────────────────┬──────────────┐
  │ 形式   │ 圧縮率    │ 用途                  │ ブラウザ対応  │
  ├────────┼───────────┼──────────────────────┼──────────────┤
  │ AVIF   │ JPEG比50%↓│ 写真（次世代、推奨）  │ Chrome/FF/Sf │
  │ WebP   │ JPEG比30%↓│ 写真（推奨）          │ 全モダン     │
  │ JPEG XL│ JPEG比35%↓│ 写真（実験的）        │ 限定的       │
  │ SVG    │ 極小      │ アイコン、ロゴ         │ 全ブラウザ   │
  │ PNG    │ 大きい    │ 透過が必要な場合のみ   │ 全ブラウザ   │
  │ JPEG   │ ベースライン│ フォールバック       │ 全ブラウザ   │
  └────────┴───────────┴──────────────────────┴──────────────┘

  実測値（写真 1920x1080px, 品質80）:
  ┌────────┬───────────┐
  │ 形式   │ ファイルサイズ│
  ├────────┼───────────┤
  │ PNG    │ 3.2MB     │
  │ JPEG   │ 280KB     │
  │ WebP   │ 195KB     │
  │ AVIF   │ 140KB     │
  └────────┴───────────┘
```

```html
<!-- レスポンシブ画像の完全実装 -->

<!-- 1. 基本的なpicture要素 -->
<picture>
  <source srcset="image.avif" type="image/avif">
  <source srcset="image.webp" type="image/webp">
  <img src="image.jpg"
       alt="商品画像"
       loading="lazy"
       decoding="async"
       width="800"
       height="600">
</picture>

<!-- 2. レスポンシブ対応（解像度とビューポート幅） -->
<picture>
  <!-- デスクトップ向け -->
  <source
    media="(min-width: 1024px)"
    srcset="hero-1600.avif 1600w, hero-1200.avif 1200w"
    sizes="100vw"
    type="image/avif">
  <source
    media="(min-width: 1024px)"
    srcset="hero-1600.webp 1600w, hero-1200.webp 1200w"
    sizes="100vw"
    type="image/webp">

  <!-- モバイル向け -->
  <source
    srcset="hero-800.avif 800w, hero-400.avif 400w"
    sizes="100vw"
    type="image/avif">
  <source
    srcset="hero-800.webp 800w, hero-400.webp 400w"
    sizes="100vw"
    type="image/webp">

  <img
    src="hero-800.jpg"
    alt="ヒーローイメージ"
    loading="eager"
    fetchpriority="high"
    width="1600"
    height="900">
</picture>

<!-- 3. Next.js Image コンポーネント（自動最適化） -->
<!-- 上記の複雑さをフレームワークが吸収 -->
```

```typescript
// Next.js Image コンポーネントの活用
import Image from 'next/image';

// 基本使用
function ProductCard({ product }: { product: Product }) {
  return (
    <div>
      <Image
        src={product.imageUrl}
        alt={product.name}
        width={400}
        height={300}
        // 自動的に WebP/AVIF 変換
        // レスポンシブ srcset 生成
        // lazy loading デフォルト
        placeholder="blur"
        blurDataURL={product.blurHash}
      />
    </div>
  );
}

// ヒーロー画像（LCP対象）
function HeroImage() {
  return (
    <Image
      src="/hero.jpg"
      alt="メインビジュアル"
      fill                   // 親要素を満たす
      sizes="100vw"
      priority               // lazy loading 無効、preload ヒント追加
      quality={85}
      className="object-cover"
    />
  );
}

// next.config.js での画像最適化設定
const nextConfig = {
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    minimumCacheTTL: 60 * 60 * 24 * 365, // 1年
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'images.example.com',
        pathname: '/uploads/**',
      },
    ],
  },
};
```

### 3.3 JavaScript の最適化

```typescript
// コード分割（Code Splitting）

// 1. ルートベースの分割（React Router）
import { lazy, Suspense } from 'react';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings = lazy(() => import('./pages/Settings'));
const Analytics = lazy(() => import(
  /* webpackChunkName: "analytics" */
  /* webpackPrefetch: true */
  './pages/Analytics'
));

function App() {
  return (
    <Suspense fallback={<PageSkeleton />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </Suspense>
  );
}

// 2. コンポーネントベースの分割
const HeavyEditor = lazy(() => import('./components/HeavyEditor'));
// → HeavyEditor は必要な時だけロード

function EditorPage() {
  const [showEditor, setShowEditor] = useState(false);
  return (
    <div>
      <button onClick={() => setShowEditor(true)}>Open Editor</button>
      {showEditor && (
        <Suspense fallback={<EditorSkeleton />}>
          <HeavyEditor />
        </Suspense>
      )}
    </div>
  );
}

// 3. 条件付きインポート
async function processImage(file: File) {
  // sharp は使用時にのみロード
  const { processWithSharp } = await import('./utils/imageProcessor');
  return processWithSharp(file);
}

// 4. Tree Shaking の効果的な活用
// 悪い（バンドル全体がインポートされる）
import _ from 'lodash';
_.debounce(fn, 300);

// 良い（必要な関数のみ）
import debounce from 'lodash/debounce';
debounce(fn, 300);

// 最良（ESModules で Tree Shaking 可能）
import { debounce } from 'lodash-es';
debounce(fn, 300);
```

```
バンドル分析と最適化:

  目標サイズ:
  ┌─────────────────────────┬────────────────────┐
  │ カテゴリ                 │ gzip後の目標サイズ   │
  ├─────────────────────────┼────────────────────┤
  │ 初期JSバンドル（合計）   │ < 200KB            │
  │ フレームワーク(React等)  │ < 45KB             │
  │ ルートごとのチャンク      │ < 100KB            │
  │ サードパーティ合計        │ < 100KB            │
  └─────────────────────────┴────────────────────┘

  分析ツール:
  → webpack-bundle-analyzer:
    npx webpack-bundle-analyzer stats.json

  → source-map-explorer:
    npx source-map-explorer build/static/js/*.js

  → Vite のビルドレポート:
    npx vite-bundle-visualizer

  大きなライブラリの代替:
  ┌──────────────┬──────────┬──────────────┬──────────┐
  │ ライブラリ    │ サイズ    │ 代替          │ サイズ    │
  ├──────────────┼──────────┼──────────────┼──────────┤
  │ moment       │ 72KB     │ day.js       │ 2KB      │
  │ lodash       │ 71KB     │ lodash-es    │ Tree可   │
  │ chart.js     │ 63KB     │ lightweight- │ 15KB     │
  │ uuid         │ 12KB     │ nanoid       │ 0.5KB    │
  │ axios        │ 13KB     │ fetch API    │ 0KB      │
  └──────────────┴──────────┴──────────────┴──────────┘
```

### 3.4 フォント最適化

```css
/* フォント最適化の実装 */

/* 1. font-display: swap でテキストを先に表示 */
@font-face {
  font-family: 'Inter';
  src: url('/fonts/inter-v13-latin-regular.woff2') format('woff2');
  font-weight: 400;
  font-style: normal;
  font-display: swap;  /* FOUT: 無スタイルテキストを先に表示 */
}

/* font-display の選択肢:
   auto:     ブラウザ任せ
   block:    3秒間非表示 → フォント表示（FOIT）
   swap:     即フォールバック → フォント切替（FOUT）- 推奨
   fallback: 100ms非表示 → フォールバック → 3秒以内に切替
   optional: 即フォールバック → 十分高速ならフォント使用
*/

/* 2. 日本語フォントのサブセット化 */
/* Noto Sans JP: フルセット = 5.7MB → サブセット = 200KB */
@font-face {
  font-family: 'Noto Sans JP';
  /* unicode-range でサブセットを定義 */
  src: url('/fonts/NotoSansJP-Regular-subset.woff2') format('woff2');
  unicode-range: U+3000-303F, U+3040-309F, U+30A0-30FF,
                 U+4E00-9FFF, U+FF00-FFEF;
  font-display: swap;
}

/* 3. システムフォントスタック（ゼロコスト） */
body {
  font-family:
    -apple-system,
    BlinkMacSystemFont,
    'Segoe UI',
    Roboto,
    'Helvetica Neue',
    Arial,
    'Noto Sans',
    'Noto Sans JP',
    sans-serif;
}
```

```html
<!-- フォントのpreload -->
<link rel="preload"
      href="/fonts/inter-v13-latin-regular.woff2"
      as="font"
      type="font/woff2"
      crossorigin>

<!-- Google Fonts の最適な読み込み -->
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap"
      rel="stylesheet">

<!-- Next.js の next/font（最適化済み） -->
<!-- ビルド時にフォントをダウンロードしてセルフホスティング -->
```

```typescript
// Next.js next/font の使用
import { Inter, Noto_Sans_JP } from 'next/font/google';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

const notoSansJP = Noto_Sans_JP({
  subsets: ['latin'],
  weight: ['400', '700'],
  display: 'swap',
  variable: '--font-noto-sans-jp',
  preload: false, // 日本語フォントは大きいのでpreloadしない
});

export default function RootLayout({ children }) {
  return (
    <html className={`${inter.variable} ${notoSansJP.variable}`}>
      <body>{children}</body>
    </html>
  );
}
```

### 3.5 動画とメディアの最適化

```html
<!-- 動画の最適化 -->

<!-- 1. 遅延読み込み + 適応的品質 -->
<video
  poster="/video-poster.webp"
  preload="none"
  playsinline
  muted
  loop>
  <!-- 低帯域ユーザー向けに解像度を段階的に提供 -->
  <source src="/video-720p.mp4" type="video/mp4"
          media="(max-width: 768px)">
  <source src="/video-1080p.mp4" type="video/mp4">
</video>

<!-- 2. Intersection Observer で遅延再生 -->
<script>
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    const video = entry.target;
    if (entry.isIntersecting) {
      video.play();
    } else {
      video.pause();
    }
  });
}, { threshold: 0.5 });

document.querySelectorAll('video[data-lazy]').forEach(v => observer.observe(v));
</script>
```

```
メディア最適化のチェックリスト:

  画像:
  □ WebP/AVIF 形式を使用
  □ レスポンシブ srcset + sizes を設定
  □ width/height を明示（CLS 防止）
  □ loading="lazy" をファーストビュー以外に設定
  □ LCP 画像に fetchpriority="high" を設定
  □ placeholder/blurHash を使用

  動画:
  □ poster 画像を設定
  □ preload="none" でデータ節約
  □ 適応的ビットレート（HLS/DASH）
  □ ファーストビュー外は Intersection Observer で制御

  SVG:
  □ SVGO で最適化（不要メタデータ除去）
  □ 小さなSVGはインライン化
  □ スプライトシートの活用（アイコン）
```

---

## 4. API最適化

### 4.1 レスポンスの最小化

```typescript
// REST API でのフィールド選択
// GET /api/users?fields=id,name,email

// サーバー側実装（Express + Prisma）
app.get('/api/users', async (req, res) => {
  const fields = req.query.fields?.split(',') || [];

  const select = fields.length > 0
    ? Object.fromEntries(fields.map(f => [f, true]))
    : undefined; // 未指定なら全フィールド

  const users = await prisma.user.findMany({
    select,
    take: 20,
  });

  res.json({ data: users });
});

// GraphQL でのフィールド選択（自動最適化）
const GET_USERS = gql`
  query GetUsers {
    users {
      id
      name
      email
      # avatar や profile は不要なので取得しない
    }
  }
`;

// tRPC でのフィールド選択
const users = await trpc.user.list.query({
  select: { id: true, name: true, email: true },
});
```

### 4.2 バッチリクエストとデータローダー

```typescript
// バッチリクエスト: 複数APIコールを1リクエストに

// クライアント側
const batchRequest = async (requests: BatchItem[]) => {
  const response = await fetch('/api/batch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ requests }),
  });
  return response.json();
};

// 使用例: 3つのAPIを1リクエストに
const results = await batchRequest([
  { method: 'GET', url: '/api/users/1' },
  { method: 'GET', url: '/api/users/1/orders' },
  { method: 'GET', url: '/api/users/1/notifications' },
]);

// サーバー側: DataLoader パターン（N+1防止）
import DataLoader from 'dataloader';

// ユーザーのバッチ取得
const userLoader = new DataLoader(async (userIds: readonly string[]) => {
  const users = await prisma.user.findMany({
    where: { id: { in: [...userIds] } },
  });
  // IDの順序を保持して返す
  const userMap = new Map(users.map(u => [u.id, u]));
  return userIds.map(id => userMap.get(id) || null);
});

// 使用（個別に呼んでもバッチ実行される）
const user1 = await userLoader.load('user-1');
const user2 = await userLoader.load('user-2');
// → 1つのSQLクエリにバッチ化: SELECT * FROM users WHERE id IN ('user-1', 'user-2')
```

### 4.3 ページネーション

```typescript
// Cursor-based ページネーション（推奨）

// メリット:
// → 一貫した結果（途中のデータ追加/削除に影響されない）
// → 大量データでもパフォーマンス安定（OFFSET不要）
// → 無限スクロールに最適

// API 設計
// GET /api/posts?cursor=abc123&limit=20

// サーバー側（Prisma）
async function getPosts(cursor?: string, limit = 20) {
  const posts = await prisma.post.findMany({
    take: limit + 1, // 1件多く取得して次ページ有無を判定
    ...(cursor && {
      cursor: { id: cursor },
      skip: 1, // cursorの要素自体はスキップ
    }),
    orderBy: { createdAt: 'desc' },
  });

  const hasMore = posts.length > limit;
  const data = hasMore ? posts.slice(0, limit) : posts;
  const nextCursor = hasMore ? data[data.length - 1].id : null;

  return {
    data,
    nextCursor,
    hasMore,
  };
}

// レスポンス例
{
  "data": [
    { "id": "post-20", "title": "..." },
    { "id": "post-19", "title": "..." }
  ],
  "nextCursor": "post-19",
  "hasMore": true
}

// クライアント側（TanStack Query useInfiniteQuery）
function useInfinitePosts() {
  return useInfiniteQuery({
    queryKey: ['posts'],
    queryFn: ({ pageParam }) =>
      fetch(`/api/posts?cursor=${pageParam || ''}&limit=20`).then(r => r.json()),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    initialPageParam: '',
  });
}
```

### 4.4 キャッシュ戦略の詳細

```
HTTPキャッシュの全体像:

  ブラウザキャッシュ → CDNキャッシュ → オリジンサーバー

  Cache-Control ヘッダーの解説:
  ┌─────────────────────────────────────────────────────────────┐
  │ Cache-Control: public, max-age=3600, stale-while-revalidate=86400 │
  │                                                              │
  │ public:        CDN/共有キャッシュに保存可                     │
  │ private:       ブラウザキャッシュのみ（ユーザー固有データ）   │
  │ max-age=3600:  3600秒（1時間）キャッシュ有効                 │
  │ s-maxage=3600: CDNのみに適用される max-age                   │
  │ no-cache:      毎回サーバーに確認（ETag/Last-Modified）      │
  │ no-store:      一切キャッシュしない                           │
  │ immutable:     max-age 内は条件付きリクエストも不要          │
  │ stale-while-revalidate=86400:                               │
  │   max-age 超過後、86400秒間は古いキャッシュを返しつつ再検証  │
  │ stale-if-error=86400:                                       │
  │   オリジンエラー時、86400秒間は古いキャッシュを返す          │
  └─────────────────────────────────────────────────────────────┘

  ETag + 条件付きリクエスト:
  1. レスポンス: ETag: "abc123"
  2. 次のリクエスト: If-None-Match: "abc123"
  3. 変更なし → 304 Not Modified（ボディなし、帯域節約）
  4. 変更あり → 200 OK（新しいデータ）

  キャッシュ無効化（Cache Busting）:
  → ファイル名にコンテンツハッシュ: app.a1b2c3d4.js
  → クエリパラメータ: styles.css?v=20240101（非推奨: CDNで効かない場合あり）
```

```typescript
// アプリケーションレベルのキャッシュ戦略

// 1. TanStack Query のキャッシュ設定
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,    // 5分間はfreshとみなす
      gcTime: 30 * 60 * 1000,      // 30分間メモリに保持
      refetchOnWindowFocus: true,   // フォーカス復帰時に再取得
      refetchOnReconnect: true,     // ネットワーク復帰時に再取得
      retry: 3,                     // 3回リトライ
      retryDelay: (attempt) =>      // エクスポネンシャルバックオフ
        Math.min(1000 * 2 ** attempt, 30000),
    },
  },
});

// 2. リソース種類ごとのキャッシュ設定
const userQuery = useQuery({
  queryKey: ['user', userId],
  queryFn: () => fetchUser(userId),
  staleTime: 10 * 60 * 1000,       // ユーザー情報: 10分
  gcTime: 60 * 60 * 1000,          // 1時間メモリ保持
});

const notificationsQuery = useQuery({
  queryKey: ['notifications'],
  queryFn: fetchNotifications,
  staleTime: 30 * 1000,             // 通知: 30秒（頻繁に更新）
  refetchInterval: 60 * 1000,       // 60秒ごとにポーリング
});

// 3. Service Worker キャッシュ（Workbox）
// Cache First: 静的アセット
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

// Stale While Revalidate: API
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/'),
  new StaleWhileRevalidate({
    cacheName: 'api-cache',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 5 * 60, // 5分
      }),
    ],
  })
);

// Network First: HTML
registerRoute(
  ({ request }) => request.mode === 'navigate',
  new NetworkFirst({
    cacheName: 'pages',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 20,
      }),
    ],
  })
);
```

### 4.5 データ圧縮とシリアライゼーション

```typescript
// JSON の効率的なシリアライゼーション

// 1. 不要なフィールドの除外
function serializeUser(user: User) {
  const { password, internalNotes, ...publicData } = user;
  return publicData;
}

// 2. superjson: Date, Map, Set などのシリアライゼーション
import superjson from 'superjson';

// サーバー
const data = { createdAt: new Date(), tags: new Set(['a', 'b']) };
const serialized = superjson.stringify(data);

// クライアント
const parsed = superjson.parse(serialized);
// parsed.createdAt は Date オブジェクト
// parsed.tags は Set オブジェクト

// 3. Protocol Buffers（高性能API向け）
// → JSON の 3-10倍高速なシリアライゼーション
// → ペイロードが 50-80% 小さい
// → gRPC, Connect で使用

// user.proto
// message User {
//   string id = 1;
//   string name = 2;
//   string email = 3;
//   google.protobuf.Timestamp created_at = 4;
// }

// 4. MessagePack（バイナリJSON）
// → JSONと同等のデータ構造をバイナリで表現
// → JSON比 20-30% 小さい
// → パース速度が 2-5倍
```

### 4.6 接続プーリングとコネクション管理

```typescript
// データベース接続プーリング

// 1. Prisma の接続プーリング
// prisma/schema.prisma
// datasource db {
//   provider = "postgresql"
//   url      = env("DATABASE_URL")
//   // ?connection_limit=10&pool_timeout=30
// }

// 2. PgBouncer の設定
// pgbouncer.ini
// [databases]
// mydb = host=localhost port=5432 dbname=mydb
//
// [pgbouncer]
// pool_mode = transaction        # トランザクション単位で接続を再利用
// max_client_conn = 1000         # 最大クライアント接続
// default_pool_size = 20         # プールサイズ
// min_pool_size = 5              # 最小プールサイズ
// reserve_pool_size = 5          # 予備プール

// 3. Redis 接続プーリング（ioredis）
import Redis from 'ioredis';

const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: 6379,
  maxRetriesPerRequest: 3,
  enableReadyCheck: true,
  lazyConnect: true,
  // Cluster mode
  // sentinels: [{ host: 'sentinel1', port: 26379 }],
});

// 4. HTTP クライアントの接続プーリング
import { Agent } from 'undici';

const agent = new Agent({
  connections: 100,         // 最大接続数
  pipelining: 10,          // パイプライニング数
  keepAliveTimeout: 60000, // Keep-Alive タイムアウト
  keepAliveMaxTimeout: 600000,
});

const response = await fetch('https://api.example.com/data', {
  dispatcher: agent,
});
```

---

## 5. Web Vitals とパフォーマンス計測

### 5.1 Core Web Vitals の詳細

```
Core Web Vitals:

  ┌──────┬──────────────────────────────┬─────────────┬────────────┐
  │ 指標 │ 測定内容                      │ Good        │ Poor       │
  ├──────┼──────────────────────────────┼─────────────┼────────────┤
  │ LCP  │ 最大コンテンツの表示          │ < 2.5s      │ > 4.0s     │
  │ INP  │ インタラクション遅延          │ < 200ms     │ > 500ms    │
  │ CLS  │ レイアウトのずれ              │ < 0.1       │ > 0.25     │
  └──────┴──────────────────────────────┴─────────────┴────────────┘

  LCP（Largest Contentful Paint）:
  → ビューポート内の最大要素が表示されるまでの時間
  → 対象: <img>, <video>, background-image, テキストブロック
  → 改善策:
     ① リソースのプリロード: <link rel="preload">
     ② 画像の最適化: WebP/AVIF, srcset
     ③ サーバーレスポンスの高速化: TTFB < 800ms
     ④ レンダリングブロックの排除: async/defer
     ⑤ fetchpriority="high" の設定

  INP（Interaction to Next Paint）:
  → ユーザーの操作から次の描画更新までの時間
  → FID の後継指標（2024年3月〜）
  → 対象: click, tap, keypress
  → 改善策:
     ① 長いタスクの分割: requestIdleCallback, setTimeout
     ② イベントハンドラの最適化
     ③ 不要な再レンダリングの防止: React.memo, useMemo
     ④ Web Worker での重い処理のオフロード
     ⑤ Concurrent Features の活用: useTransition, useDeferredValue

  CLS（Cumulative Layout Shift）:
  → ページ読み込み中のレイアウトのずれの累積スコア
  → 改善策:
     ① img/video に width/height を明示
     ② font-display: swap + サイズ一致フォールバック
     ③ 広告・埋め込みコンテンツの領域を事前確保
     ④ 動的コンテンツの挿入位置を工夫
     ⑤ CSS containment の活用
```

### 5.2 補助指標

```
ネットワーク関連の補助指標:

  TTFB（Time to First Byte）: < 800ms
  → サーバーレスポンスの速度
  → DNS + TCP + TLS + サーバー処理時間
  → 改善: CDN, キャッシュ, サーバー最適化

  FCP（First Contentful Paint）: < 1.8s
  → 最初のコンテンツ表示
  → テキスト、画像、SVG、非白色 canvas
  → 改善: Critical CSS, preload, SSR

  TTFB → FCP の差分:
  → レンダリングブロッキングの影響
  → 大きな差分 = CSSやJSがブロックしている

  TBT（Total Blocking Time）:
  → FCP と TTI の間のメインスレッドブロック時間の合計
  → 50ms超のタスクのうち50ms超過分の合計
  → INP と相関が高い

  Speed Index:
  → ページコンテンツの表示速度のスコア
  → ビジュアルの進捗を測定
  → 目標: < 3.4s
```

### 5.3 パフォーマンス計測の実装

```typescript
// Performance API を使用した計測

// 1. Navigation Timing API
function measurePageLoad() {
  const [navigation] = performance.getEntriesByType('navigation') as PerformanceNavigationTiming[];

  const metrics = {
    // DNS 解決時間
    dns: navigation.domainLookupEnd - navigation.domainLookupStart,
    // TCP 接続時間
    tcp: navigation.connectEnd - navigation.connectStart,
    // TLS ハンドシェイク時間
    tls: navigation.secureConnectionStart > 0
      ? navigation.connectEnd - navigation.secureConnectionStart
      : 0,
    // TTFB
    ttfb: navigation.responseStart - navigation.requestStart,
    // コンテンツダウンロード時間
    download: navigation.responseEnd - navigation.responseStart,
    // DOMContentLoaded
    domContentLoaded: navigation.domContentLoadedEventEnd - navigation.startTime,
    // Load イベント
    load: navigation.loadEventEnd - navigation.startTime,
    // DOM パース時間
    domParsing: navigation.domInteractive - navigation.responseEnd,
  };

  console.table(metrics);
  return metrics;
}

// 2. Resource Timing API
function measureResources() {
  const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];

  // 最も遅いリソース TOP 10
  const slowest = resources
    .sort((a, b) => b.duration - a.duration)
    .slice(0, 10)
    .map(r => ({
      name: r.name.split('/').pop(),
      duration: Math.round(r.duration),
      size: r.transferSize,
      type: r.initiatorType,
      protocol: r.nextHopProtocol,
    }));

  console.table(slowest);

  // リソースタイプ別の合計
  const byType = resources.reduce((acc, r) => {
    const type = r.initiatorType;
    if (!acc[type]) acc[type] = { count: 0, totalSize: 0, totalDuration: 0 };
    acc[type].count++;
    acc[type].totalSize += r.transferSize;
    acc[type].totalDuration += r.duration;
    return acc;
  }, {} as Record<string, { count: number; totalSize: number; totalDuration: number }>);

  console.table(byType);
}

// 3. Core Web Vitals の計測（web-vitals ライブラリ）
import { onLCP, onINP, onCLS, onFCP, onTTFB } from 'web-vitals';

function reportWebVitals() {
  onLCP((metric) => {
    console.log('LCP:', metric.value, 'ms');
    console.log('LCP Element:', metric.entries[0]?.element);
    sendToAnalytics('LCP', metric);
  });

  onINP((metric) => {
    console.log('INP:', metric.value, 'ms');
    sendToAnalytics('INP', metric);
  });

  onCLS((metric) => {
    console.log('CLS:', metric.value);
    // CLS の各シフトの詳細
    metric.entries.forEach(entry => {
      console.log('Shift:', entry.value, entry.sources);
    });
    sendToAnalytics('CLS', metric);
  });

  onFCP((metric) => {
    console.log('FCP:', metric.value, 'ms');
    sendToAnalytics('FCP', metric);
  });

  onTTFB((metric) => {
    console.log('TTFB:', metric.value, 'ms');
    sendToAnalytics('TTFB', metric);
  });
}

// 4. カスタムパフォーマンスマーク
function measureCustom() {
  // APIコールの計測
  performance.mark('api-start');

  fetch('/api/users').then(async (res) => {
    performance.mark('api-end');
    performance.measure('api-call', 'api-start', 'api-end');

    const [measure] = performance.getEntriesByName('api-call');
    console.log('API call duration:', measure.duration, 'ms');
  });
}

// 5. Long Tasks API（INP改善用）
const observer = new PerformanceObserver((list) => {
  list.getEntries().forEach((entry) => {
    if (entry.duration > 50) {
      console.warn('Long Task detected:', {
        duration: entry.duration,
        startTime: entry.startTime,
        name: entry.name,
      });
    }
  });
});
observer.observe({ type: 'longtask', buffered: true });
```

### 5.4 RUM（Real User Monitoring）

```typescript
// RUM データの収集と送信

interface PerformanceData {
  url: string;
  userAgent: string;
  connectionType: string;
  effectiveType: string;
  lcp: number;
  inp: number;
  cls: number;
  fcp: number;
  ttfb: number;
  timestamp: number;
}

function collectPerformanceData(): void {
  const connection = (navigator as any).connection;

  const data: Partial<PerformanceData> = {
    url: window.location.href,
    userAgent: navigator.userAgent,
    connectionType: connection?.type || 'unknown',
    effectiveType: connection?.effectiveType || 'unknown',
    timestamp: Date.now(),
  };

  // web-vitals で各指標を収集
  onLCP(m => { data.lcp = m.value; maybeSend(data); });
  onINP(m => { data.inp = m.value; maybeSend(data); });
  onCLS(m => { data.cls = m.value; maybeSend(data); });
  onFCP(m => { data.fcp = m.value; maybeSend(data); });
  onTTFB(m => { data.ttfb = m.value; maybeSend(data); });
}

function maybeSend(data: Partial<PerformanceData>): void {
  // 全指標が揃ったら送信
  if (data.lcp && data.inp !== undefined && data.cls !== undefined) {
    sendBeacon(data);
  }
}

function sendBeacon(data: Partial<PerformanceData>): void {
  // sendBeacon: ページ遷移時でも確実に送信
  navigator.sendBeacon('/api/analytics/performance', JSON.stringify(data));
}

// RUM データの集約と分析（サーバー側）
// → p75 / p95 / p99 のパーセンタイルで分析
// → ネットワーク種別ごとの比較
// → ページ別の比較
// → リグレッション検出
```

### 5.5 計測ツールとダッシュボード

```
パフォーマンス計測ツール:

  ラボデータ（合成テスト）:
  ┌──────────────────┬────────────────────────────────────┐
  │ ツール            │ 特徴                                │
  ├──────────────────┼────────────────────────────────────┤
  │ Lighthouse       │ Chrome DevTools統合、スコア + 提案   │
  │ WebPageTest      │ 世界各地からテスト、Waterfall分析    │
  │ PageSpeed Insights│ Lighthouse + CrUXデータ            │
  │ unlighthouse     │ サイト全体を一括監査                 │
  └──────────────────┴────────────────────────────────────┘

  フィールドデータ（実ユーザー）:
  ┌──────────────────┬────────────────────────────────────┐
  │ ツール            │ 特徴                                │
  ├──────────────────┼────────────────────────────────────┤
  │ CrUX             │ Chrome実ユーザーデータ（無料）       │
  │ Vercel Analytics │ Next.js向け、自動CWV計測            │
  │ Sentry           │ エラー + パフォーマンス計測           │
  │ Datadog RUM      │ エンタープライズ向け RUM              │
  │ SpeedCurve       │ RUM + 合成テスト + 可視化            │
  └──────────────────┴────────────────────────────────────┘

  CI/CD での自動計測:
  → Lighthouse CI: PR ごとにスコアを計測
  → Bundlesize: バンドルサイズの閾値チェック
  → web-vitals-reporter: CWV のリグレッション検出

  Lighthouse CI 設定例:
  // lighthouserc.js
  module.exports = {
    ci: {
      collect: {
        url: ['http://localhost:3000/', 'http://localhost:3000/products'],
        numberOfRuns: 3,
      },
      assert: {
        assertions: {
          'categories:performance': ['error', { minScore: 0.9 }],
          'largest-contentful-paint': ['warn', { maxNumericValue: 2500 }],
          'interactive': ['error', { maxNumericValue: 3800 }],
          'cumulative-layout-shift': ['warn', { maxNumericValue: 0.1 }],
        },
      },
      upload: {
        target: 'temporary-public-storage',
      },
    },
  };
```

---

## 6. Service Worker とオフライン対応

### 6.1 Service Worker の基本

```typescript
// Service Worker の登録
// app.ts
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/',
      });
      console.log('SW registered:', registration.scope);
    } catch (error) {
      console.error('SW registration failed:', error);
    }
  });
}

// sw.js - Service Worker
const CACHE_NAME = 'app-v1';
const STATIC_ASSETS = [
  '/',
  '/offline.html',
  '/styles.css',
  '/app.js',
  '/icons/icon-192x192.png',
];

// インストール: 静的アセットをキャッシュ
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting(); // 即座にアクティブ化
});

// アクティベート: 古いキャッシュを削除
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    })
  );
  self.clients.claim(); // 全クライアントを制御
});

// フェッチ: キャッシュ戦略の実装
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // API リクエスト: Network First
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }

  // 静的アセット: Cache First
  if (request.destination === 'image' ||
      request.destination === 'style' ||
      request.destination === 'script') {
    event.respondWith(cacheFirst(request));
    return;
  }

  // HTML: Network First with Offline Fallback
  if (request.mode === 'navigate') {
    event.respondWith(
      networkFirst(request).catch(() => caches.match('/offline.html'))
    );
    return;
  }
});

async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;
  const response = await fetch(request);
  const cache = await caches.open(CACHE_NAME);
  cache.put(request, response.clone());
  return response;
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone());
    return response;
  } catch {
    return caches.match(request);
  }
}
```

### 6.2 Workbox を使った Service Worker

```typescript
// Workbox: Google の Service Worker ライブラリ

// next.config.js (next-pwa)
const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true,
  runtimeCaching: [
    {
      urlPattern: /^https:\/\/api\.example\.com\/.*/i,
      handler: 'StaleWhileRevalidate',
      options: {
        cacheName: 'api-cache',
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 60 * 60, // 1時間
        },
      },
    },
    {
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|avif)$/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'image-cache',
        expiration: {
          maxEntries: 100,
          maxAgeSeconds: 30 * 24 * 60 * 60, // 30日
        },
      },
    },
    {
      urlPattern: /\.(?:js|css)$/i,
      handler: 'StaleWhileRevalidate',
      options: {
        cacheName: 'static-cache',
        expiration: {
          maxEntries: 50,
          maxAgeSeconds: 24 * 60 * 60, // 1日
        },
      },
    },
  ],
});

module.exports = withPWA({
  // Next.js config
});
```

---

## 7. エッジコンピューティングとEdge Functions

### 7.1 エッジコンピューティングの概要

```
エッジコンピューティング:

  従来のアーキテクチャ:
  ユーザー → CDN(静的配信のみ) → オリジンサーバー(ロジック)

  エッジコンピューティング:
  ユーザー → エッジ(ロジック実行可能) → オリジン(必要時のみ)

  利点:
  → レイテンシ大幅削減（ユーザーの近くでロジック実行）
  → オリジンサーバーの負荷軽減
  → グローバルに分散した処理
  → コールドスタートが高速（数ms）

  制約:
  → 実行時間制限（通常 50ms〜30秒）
  → メモリ制限（128MB〜）
  → Node.js API の一部が使用不可
  → ステートレス（永続化はKV Store等で）
```

```typescript
// Cloudflare Workers
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // A/Bテスト
    if (url.pathname === '/') {
      const bucket = request.headers.get('cf-connecting-ip')?.charCodeAt(0) % 2;
      const variant = bucket === 0 ? 'control' : 'experiment';
      const response = await fetch(`${url.origin}/variants/${variant}`);
      return new Response(response.body, {
        headers: {
          ...Object.fromEntries(response.headers),
          'X-Variant': variant,
        },
      });
    }

    // 地域別コンテンツ
    if (url.pathname === '/pricing') {
      const country = request.cf?.country || 'US';
      const currency = getCurrency(country);
      // KV Store からキャッシュされた料金を取得
      const pricing = await env.PRICING_KV.get(`pricing:${currency}`, 'json');
      return Response.json(pricing);
    }

    return fetch(request);
  },
};

// Vercel Edge Functions
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  // 地域ベースのリダイレクト
  const country = request.geo?.country || 'US';

  if (country === 'JP' && !request.nextUrl.pathname.startsWith('/ja')) {
    return NextResponse.redirect(new URL('/ja' + request.nextUrl.pathname, request.url));
  }

  // レート制限
  const ip = request.ip || 'unknown';
  // Edge KV でレート制限を実装

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
```

### 7.2 エッジでのデータアクセス

```typescript
// エッジデータベース

// 1. Cloudflare D1（エッジSQLite）
export default {
  async fetch(request: Request, env: Env) {
    const { results } = await env.DB
      .prepare('SELECT * FROM products WHERE category = ?')
      .bind('electronics')
      .all();

    return Response.json(results);
  },
};

// 2. Vercel KV（エッジRedis）
import { kv } from '@vercel/kv';

export async function GET(request: Request) {
  // セッション取得
  const session = await kv.get(`session:${sessionId}`);

  // レート制限
  const requests = await kv.incr(`ratelimit:${ip}`);
  if (requests === 1) {
    await kv.expire(`ratelimit:${ip}`, 60);
  }
  if (requests > 100) {
    return new Response('Too Many Requests', { status: 429 });
  }

  return Response.json({ data: session });
}

// 3. PlanetScale / Neon（エッジ対応DB）
import { neon } from '@neondatabase/serverless';

export async function GET(request: Request) {
  const sql = neon(process.env.DATABASE_URL!);
  const products = await sql`
    SELECT id, name, price FROM products
    WHERE category = 'electronics'
    ORDER BY created_at DESC
    LIMIT 20
  `;
  return Response.json(products);
}
```

---

## 8. ネットワークレジリエンス

### 8.1 リトライとバックオフ

```typescript
// エクスポネンシャルバックオフ付きリトライ
async function fetchWithRetry(
  url: string,
  options: RequestInit = {},
  maxRetries = 3,
  baseDelay = 1000,
): Promise<Response> {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        ...options,
        signal: AbortSignal.timeout(10000), // 10秒タイムアウト
      });

      // 5xx エラーはリトライ対象
      if (response.status >= 500 && attempt < maxRetries) {
        throw new Error(`Server error: ${response.status}`);
      }

      // 429 Too Many Requests: Retry-After ヘッダーを参照
      if (response.status === 429) {
        const retryAfter = parseInt(response.headers.get('Retry-After') || '60');
        await sleep(retryAfter * 1000);
        continue;
      }

      return response;
    } catch (error) {
      lastError = error as Error;

      if (attempt < maxRetries) {
        // エクスポネンシャルバックオフ + ジッター
        const delay = baseDelay * Math.pow(2, attempt);
        const jitter = delay * 0.1 * Math.random();
        console.warn(`Retry ${attempt + 1}/${maxRetries} after ${delay + jitter}ms`);
        await sleep(delay + jitter);
      }
    }
  }

  throw lastError!;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// サーキットブレーカーパターン
class CircuitBreaker {
  private failures = 0;
  private lastFailure: number = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';

  constructor(
    private threshold = 5,          // 失敗回数の閾値
    private resetTimeout = 30000,   // リセットまでの時間（ms）
  ) {}

  async execute<T>(fn: () => Promise<T>): Promise<T> {
    if (this.state === 'open') {
      if (Date.now() - this.lastFailure > this.resetTimeout) {
        this.state = 'half-open';
      } else {
        throw new Error('Circuit is open');
      }
    }

    try {
      const result = await fn();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess(): void {
    this.failures = 0;
    this.state = 'closed';
  }

  private onFailure(): void {
    this.failures++;
    this.lastFailure = Date.now();
    if (this.failures >= this.threshold) {
      this.state = 'open';
    }
  }
}

// 使用例
const apiBreaker = new CircuitBreaker(5, 30000);

async function fetchUserData(userId: string) {
  return apiBreaker.execute(() =>
    fetchWithRetry(`/api/users/${userId}`)
  );
}
```

### 8.2 ネットワーク状態の検出

```typescript
// ネットワーク状態の検出と適応

// 1. Online/Offline 検出
function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
}

// 2. Network Information API
function useNetworkQuality() {
  const [quality, setQuality] = useState<'fast' | 'slow' | 'offline'>('fast');

  useEffect(() => {
    const connection = (navigator as any).connection;
    if (!connection) return;

    const updateQuality = () => {
      if (!navigator.onLine) {
        setQuality('offline');
        return;
      }

      const effectiveType = connection.effectiveType;
      // effectiveType: 'slow-2g', '2g', '3g', '4g'
      if (effectiveType === '4g' && connection.downlink > 5) {
        setQuality('fast');
      } else {
        setQuality('slow');
      }
    };

    connection.addEventListener('change', updateQuality);
    updateQuality();

    return () => connection.removeEventListener('change', updateQuality);
  }, []);

  return quality;
}

// 3. 適応的コンテンツ配信
function ProductImage({ product }: { product: Product }) {
  const quality = useNetworkQuality();

  return (
    <Image
      src={product.imageUrl}
      alt={product.name}
      quality={quality === 'fast' ? 85 : 50}  // 低速時は品質を下げる
      placeholder={quality === 'fast' ? 'blur' : 'empty'}
      loading={quality === 'fast' ? 'eager' : 'lazy'}
    />
  );
}

// 4. Save-Data ヘッダーの検出
// ユーザーがデータセーバーを有効にしている場合
function useSaveData() {
  const connection = (navigator as any).connection;
  return connection?.saveData === true;
}

// サーバー側での Save-Data 対応
// Save-Data: on ヘッダーを検出
// → 低品質画像を返す
// → 自動再生動画を無効化
// → 不要なアセットの読み込みをスキップ
```

---

## 9. パフォーマンス最適化チェックリスト

```
ネットワークパフォーマンス最適化チェックリスト:

  接続:
  □ CDNを使用している
  □ HTTP/2 以上を有効化している
  □ dns-prefetch / preconnect を設定している（最大3-4ドメイン）
  □ Keep-Alive が有効
  □ TLS 1.3 を使用している
  □ 103 Early Hints を検討している
  □ HTTP/3 (QUIC) のサポートを検討している

  転送:
  □ Brotli/gzip 圧縮を有効化している
  □ 画像をWebP/AVIF形式にしている
  □ レスポンシブ画像（srcset + sizes）を設定している
  □ LCP画像に fetchpriority="high" を設定している
  □ JavaScriptをコード分割している
  □ 初期バンドル < 200KB（gzip後）
  □ フォントをWOFF2 + サブセット化している
  □ font-display: swap を設定している
  □ Critical CSS をインライン化している
  □ 非クリティカルCSS/JSを async/defer にしている

  キャッシュ:
  □ 静的ファイルに Cache-Control + ハッシュ付きファイル名 + immutable
  □ HTMLに no-cache + ETag
  □ APIに stale-while-revalidate
  □ CDNキャッシュヒット率を監視している（目標 > 90%）
  □ Service Worker でオフライン対応

  API:
  □ 不要なフィールドを返していない
  □ ページネーション（Cursor-based推奨）を実装している
  □ N+1問題が発生していない（DataLoader使用）
  □ バッチリクエストを検討している
  □ 接続プーリングを使用している

  レンダリング:
  □ img/video に width/height を明示（CLS防止）
  □ 動的コンテンツの領域を事前確保している
  □ 長いタスクを分割している（INP改善）
  □ React.memo / useMemo で不要な再レンダリングを防止

  監視:
  □ Core Web Vitals (LCP, INP, CLS) を計測している
  □ TTFB を監視している（目標 < 800ms）
  □ エラー率を監視している
  □ RUM データを収集・分析している
  □ CI/CD で Lighthouse スコアを自動チェックしている
  □ バンドルサイズの推移を追跡している

  レジリエンス:
  □ リトライ + エクスポネンシャルバックオフを実装している
  □ タイムアウトを設定している
  □ サーキットブレーカーを検討している
  □ ネットワーク状態（Online/Offline）に対応している
  □ Save-Data ヘッダーに対応している
```

---

## 10. 実務での最適化フロー

### 10.1 パフォーマンスバジェット

```
パフォーマンスバジェットの設定:

  指標ベースのバジェット:
  ┌──────────────────────────┬─────────────────┐
  │ 指標                      │ バジェット        │
  ├──────────────────────────┼─────────────────┤
  │ LCP                      │ < 2.5s           │
  │ INP                      │ < 200ms          │
  │ CLS                      │ < 0.1            │
  │ TTFB                     │ < 800ms          │
  │ FCP                      │ < 1.8s           │
  │ Lighthouse Performance   │ > 90             │
  └──────────────────────────┴─────────────────┘

  リソースベースのバジェット:
  ┌──────────────────────────┬─────────────────┐
  │ リソース                  │ バジェット        │
  ├──────────────────────────┼─────────────────┤
  │ 初期JS (gzip)            │ < 200KB          │
  │ 初期CSS (gzip)           │ < 50KB           │
  │ 画像（ページあたり）      │ < 1MB            │
  │ フォント                  │ < 100KB          │
  │ 合計ページサイズ          │ < 2MB            │
  │ HTTPリクエスト数          │ < 50             │
  └──────────────────────────┴─────────────────┘

  バジェット超過時の対応:
  → CI/CD で自動チェック（Lighthouse CI, bundlesize）
  → PR レビューで確認
  → 超過の場合: 別のリソースを削減 or 機能の見直し
```

### 10.2 段階的な最適化アプローチ

```
最適化の優先順位:

  Phase 1: Low Hanging Fruits（効果大 + 工数小）
  ① 圧縮の有効化（Brotli/gzip）
  ② 画像の最適化（WebP/AVIF変換）
  ③ Cache-Control の設定
  ④ preconnect / preload の設定
  ⑤ font-display: swap

  Phase 2: アーキテクチャ改善
  ⑥ コード分割（ルートベース + コンポーネントベース）
  ⑦ Tree Shaking + 大きなライブラリの置換
  ⑧ SSR / ISR の導入
  ⑨ Critical CSS のインライン化

  Phase 3: 高度な最適化
  ⑩ Service Worker + オフライン対応
  ⑪ Edge Functions / エッジコンピューティング
  ⑫ HTTP/3 の導入
  ⑬ RUM の導入と継続的モニタリング
  ⑭ A/Bテストによるパフォーマンス検証

  ROI の高い最適化:
  → 画像最適化: ページサイズの50%を占めることが多い
  → コード分割: 初期ロードの大幅削減
  → CDN + キャッシュ: TTFB の大幅改善
  → SSR: FCP / LCP の大幅改善
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| レイテンシ | CDN + preconnect + HTTP/2以上 + エッジコンピューティング |
| 帯域 | Brotli圧縮 + WebP/AVIF + コード分割 + Tree Shaking |
| キャッシュ | Cache-Control + ETag + SWR + Service Worker |
| API | フィールド選択 + Cursor-based pagination + バッチ + 接続プーリング |
| 計測 | Core Web Vitals + Lighthouse + RUM + CI/CD自動チェック |
| レジリエンス | リトライ + サーキットブレーカー + ネットワーク状態検出 |
| エッジ | Edge Functions + KV Store + エッジDB |

---

## 参考文献

1. web.dev. "Web Performance." Google, 2024.
2. Grigorik, I. "High Performance Browser Networking." O'Reilly, 2013.
3. RFC 7932. "Brotli Compressed Data Format." IETF, 2016.
4. RFC 9000. "QUIC: A UDP-Based Multiplexed and Secure Transport." IETF, 2021.
5. RFC 9114. "HTTP/3." IETF, 2022.
6. web.dev. "Core Web Vitals." Google, 2024.
7. Cloudflare. "Workers Documentation." developers.cloudflare.com, 2024.
8. Vercel. "Edge Functions." vercel.com/docs, 2024.
9. workboxjs.org. "Workbox Documentation." Google, 2024.
10. TanStack. "TanStack Query Documentation." tanstack.com, 2024.
