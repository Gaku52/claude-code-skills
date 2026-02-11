# ネットワーク最適化

> Webアプリケーションのネットワークパフォーマンスを最適化する。レイテンシ削減、帯域最適化、接続管理、圧縮、プリロード等の手法を体系的に学び、高速なユーザー体験を実現する。

## この章で学ぶこと

- [ ] ネットワークパフォーマンスのボトルネックを理解する
- [ ] レイテンシ削減と帯域最適化の手法を把握する
- [ ] 接続管理とリソースの最適化戦略を学ぶ

---

## 1. パフォーマンスのボトルネック

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

---

## 2. レイテンシ削減

```
① CDNの活用:
  → ユーザーに近いエッジサーバーから配信
  → RTT: 100ms → 5ms（東京のユーザーが東京エッジから取得）

② 接続の事前確立:
  <link rel="dns-prefetch" href="//api.example.com">
  → DNSを事前解決

  <link rel="preconnect" href="https://api.example.com">
  → DNS + TCP + TLS を事前確立

  <link rel="preload" href="/critical.css" as="style">
  → リソースを優先的にダウンロード

③ HTTP/2 or HTTP/3:
  → 多重化で接続数削減
  → ヘッダー圧縮
  → HTTP/3: 1 RTT接続確立、0-RTT再接続

④ 接続の再利用:
  Connection: keep-alive（HTTP/1.1デフォルト）
  → TCP + TLS のハンドシェイクコストを回避

⑤ サーバーの地理的分散:
  → マルチリージョンデプロイ
  → ユーザーの最寄りリージョンにルーティング
  → Route 53 レイテンシベースルーティング
```

---

## 3. 帯域最適化

```
① テキスト圧縮:
  Content-Encoding: gzip   → 70-80%削減
  Content-Encoding: br     → 80-90%削減（Brotli、推奨）

  Nginx設定:
  gzip on;
  gzip_types text/plain text/css application/json application/javascript;
  gzip_min_length 1024;

  Brotli（より高圧縮）:
  brotli on;
  brotli_types text/plain text/css application/json application/javascript;

② 画像最適化:
  フォーマット選択:
  ┌────────┬───────────┬──────────────────────┐
  │ 形式   │ 圧縮率    │ 用途                  │
  ├────────┼───────────┼──────────────────────┤
  │ WebP   │ JPEG比30%↓│ 写真（推奨）          │
  │ AVIF   │ JPEG比50%↓│ 写真（次世代、推奨）  │
  │ SVG    │ 極小      │ アイコン、ロゴ         │
  │ PNG    │ 大きい    │ 透過が必要な場合のみ   │
  └────────┴───────────┴──────────────────────┘

  レスポンシブ画像:
  <picture>
    <source srcset="image.avif" type="image/avif">
    <source srcset="image.webp" type="image/webp">
    <img src="image.jpg" loading="lazy" width="800" height="600">
  </picture>

③ JavaScriptの最適化:
  コード分割（Code Splitting）:
  → ページごとに必要なJSのみロード
  → dynamic import: import('./module.js')
  → React.lazy() + Suspense

  Tree Shaking:
  → 未使用コードの除去
  → ESModules の静的解析

  バンドルサイズの目標:
  初期読み込み: < 200KB（gzip後）
  ルートごとのチャンク: < 100KB

④ フォント最適化:
  font-display: swap;  → テキストを先に表示
  → サブセット化（日本語フォントは特に効果大）
  → WOFF2形式（最高圧縮率）
  → preload: <link rel="preload" href="/font.woff2" as="font" crossorigin>
```

---

## 4. API最適化

```
① レスポンスの最小化:
  → 必要なフィールドのみ返す
  → GET /api/users?fields=id,name,email
  → GraphQLのフィールド選択

② バッチリクエスト:
  → 複数のAPIコールを1リクエストにまとめる
  → POST /api/batch [{ method: "GET", url: "/users/1" }, ...]

③ ページネーション:
  → cursor方式で効率的にデータ取得
  → 一度に大量データを取得しない

④ キャッシュ活用:
  → Cache-Control + ETag
  → stale-while-revalidate
  → クライアント側キャッシュ（SWR, React Query）

⑤ データ圧縮:
  Accept-Encoding: gzip, br
  → JSONレスポンスの圧縮

⑥ 接続プーリング:
  → サーバー側でDB/外部API接続をプール
  → 接続確立コストの削減
```

---

## 5. Web Vitals とパフォーマンス計測

```
Core Web Vitals:
  ┌──────┬──────────────────────┬─────────┐
  │ 指標 │ 測定内容              │ 目標    │
  ├──────┼──────────────────────┼─────────┤
  │ LCP  │ 最大コンテンツの表示  │ < 2.5s  │
  │ INP  │ インタラクション遅延  │ < 200ms │
  │ CLS  │ レイアウトのずれ      │ < 0.1   │
  └──────┴──────────────────────┴─────────┘

ネットワーク関連の指標:
  TTFB（Time to First Byte）: < 800ms
  → サーバーレスポンスの速度

  FCP（First Contentful Paint）: < 1.8s
  → 最初のコンテンツ表示

計測ツール:
  Lighthouse:
    → Chrome DevTools > Lighthouse タブ
    → パフォーマンススコア + 改善提案

  WebPageTest:
    → webpagetest.org
    → 世界各地からのテスト
    → Waterfall分析

  Performance API:
    // Navigation Timing
    const timing = performance.getEntriesByType('navigation')[0];
    console.log('TTFB:', timing.responseStart - timing.requestStart);
    console.log('DOM Ready:', timing.domContentLoadedEventEnd);
    console.log('Load:', timing.loadEventEnd);

    // Resource Timing
    const resources = performance.getEntriesByType('resource');
    resources.forEach(r => {
      console.log(r.name, r.duration.toFixed(0) + 'ms');
    });
```

---

## 6. チェックリスト

```
ネットワークパフォーマンス最適化チェックリスト:

  接続:
  □ CDNを使用している
  □ HTTP/2 以上を有効化している
  □ dns-prefetch / preconnect を設定している
  □ Keep-Alive が有効

  転送:
  □ Brotli/gzip 圧縮を有効化している
  □ 画像をWebP/AVIF形式にしている
  □ JavaScriptをコード分割している
  □ 初期バンドル < 200KB（gzip後）
  □ フォントをWOFF2 + サブセット化している

  キャッシュ:
  □ 静的ファイルにCache-Control + ハッシュ付きファイル名
  □ APIにETag / stale-while-revalidate
  □ Service Worker でオフライン対応

  API:
  □ 不要なフィールドを返していない
  □ ページネーションを実装している
  □ N+1問題が発生していない

  監視:
  □ Core Web Vitals を計測している
  □ TTFB を監視している
  □ エラー率を監視している
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| レイテンシ | CDN + preconnect + HTTP/2以上 |
| 帯域 | Brotli圧縮 + WebP/AVIF + コード分割 |
| キャッシュ | Cache-Control + ETag + SWR |
| API | フィールド選択 + cursor + バッチ |
| 計測 | Core Web Vitals + Lighthouse |

---

## 参考文献
1. web.dev. "Web Performance." Google, 2024.
2. Grigorik, I. "High Performance Browser Networking." O'Reilly, 2013.
3. RFC 7932. "Brotli Compressed Data Format." IETF, 2016.
