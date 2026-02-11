# HTTPキャッシュ

> HTTPキャッシュはWebパフォーマンスの要。Cache-Control、ETag、CDNの仕組みを理解し、「何を」「どこで」「どれくらい」キャッシュするかの戦略を設計する。

## この章で学ぶこと

- [ ] HTTPキャッシュの仕組みと種類を理解する
- [ ] Cache-ControlとETagの使い方を把握する
- [ ] CDNキャッシュとの連携を学ぶ

---

## 1. キャッシュの種類

```
キャッシュの階層:

  ブラウザ              CDN/プロキシ          オリジンサーバー
  ┌──────────┐        ┌──────────┐        ┌──────────┐
  │ メモリ    │        │ エッジ    │        │ アプリ    │
  │ キャッシュ │──→    │ キャッシュ │──→    │ キャッシュ │──→ DB
  │          │  ミス   │          │  ミス   │(Redis等) │
  │ ディスク  │        │          │        │          │
  │ キャッシュ │        │          │        │          │
  └──────────┘        └──────────┘        └──────────┘
   プライベート          共有                 サーバー側

プライベートキャッシュ:
  → ブラウザ内（1ユーザー専用）
  → ユーザー固有のデータも安全にキャッシュ可能

共有キャッシュ:
  → CDN, プロキシ（複数ユーザーで共有）
  → ユーザー固有データはキャッシュしてはいけない
  → Cache-Control: private で共有キャッシュを回避
```

---

## 2. Cache-Control

```
Cache-Control ヘッダー:

  レスポンスディレクティブ:
  ┌───────────────────┬──────────────────────────────────┐
  │ ディレクティブ     │ 意味                             │
  ├───────────────────┼──────────────────────────────────┤
  │ max-age=3600      │ 3600秒間キャッシュ有効           │
  │ s-maxage=3600     │ 共有キャッシュでの有効期限        │
  │ no-cache          │ キャッシュするが毎回検証必須      │
  │ no-store          │ 一切キャッシュしない              │
  │ private           │ ブラウザのみキャッシュ可          │
  │ public            │ 共有キャッシュもOK               │
  │ must-revalidate   │ 期限切れ後は必ず検証             │
  │ immutable         │ 変更されないリソース             │
  │ stale-while-      │ 検証中に古いキャッシュを使用     │
  │  revalidate=60    │                                  │
  └───────────────────┴──────────────────────────────────┘

  よくある誤解:
  no-cache ≠ キャッシュしない
  no-cache = キャッシュするが、使用前に必ずサーバーに検証
  no-store = キャッシュしない（こちらが「キャッシュ禁止」）

実践的なパターン:

  ① 静的ファイル（CSS/JS/画像 — ハッシュ付きファイル名）:
     Cache-Control: public, max-age=31536000, immutable
     → 1年間キャッシュ、変更されないことを保証
     → ファイル名にハッシュ: app.a1b2c3.js

  ② HTML:
     Cache-Control: no-cache
     → 常に最新版を取得（キャッシュは検証後に使用）

  ③ APIレスポンス（パブリック）:
     Cache-Control: public, max-age=60, s-maxage=300
     → ブラウザ60秒、CDN300秒

  ④ APIレスポンス（ユーザー固有）:
     Cache-Control: private, max-age=0, must-revalidate
     → ブラウザのみ、毎回検証

  ⑤ 機密データ:
     Cache-Control: no-store
     → 一切キャッシュしない
```

---

## 3. 条件付きリクエスト（ETag / Last-Modified）

```
ETag（Entity Tag）:
  → リソースのバージョン識別子（ハッシュ値等）

  1回目のリクエスト:
  GET /api/users/123 HTTP/1.1

  レスポンス:
  HTTP/1.1 200 OK
  ETag: "abc123"
  Cache-Control: no-cache
  {"name": "Taro", "email": "taro@example.com"}

  2回目のリクエスト（条件付き）:
  GET /api/users/123 HTTP/1.1
  If-None-Match: "abc123"

  レスポンス（変更なし）:
  HTTP/1.1 304 Not Modified
  ETag: "abc123"
  （ボディなし → 帯域節約）

  レスポンス（変更あり）:
  HTTP/1.1 200 OK
  ETag: "def456"
  {"name": "Taro Updated", ...}

Last-Modified:
  → リソースの最終更新日時
  → If-Modified-Since ヘッダーで条件付きリクエスト
  → ETagより精度が低い（秒単位）

  推奨: ETag を優先、Last-Modified は補助的に使用

強いETag vs 弱いETag:
  強い: "abc123"
    → バイト単位で一致を保証
    → Range リクエストに使用可能

  弱い: W/"abc123"
    → 意味的に同等であることを示す
    → マイナーな変更（空白等）は無視
```

---

## 4. キャッシュ無効化

```
キャッシュ破棄（Cache Busting）:

  ① ファイル名ハッシュ（推奨）:
     app.a1b2c3d4.js
     style.e5f6g7h8.css
     → ファイル内容が変わるとハッシュが変わる
     → 古いキャッシュは自然に無効化
     → Webpack, Vite 等のビルドツールが自動生成

  ② クエリパラメータ:
     app.js?v=1.2.3
     → 簡易的だがCDNによってはキャッシュキーに含めない場合あり

  ③ CDN キャッシュパージ:
     → CloudFront: CreateInvalidation API
     → Cloudflare: Purge Cache API
     → デプロイ時に自動パージ

stale-while-revalidate パターン:
  Cache-Control: max-age=60, stale-while-revalidate=300

  0-60秒:   キャッシュをそのまま返す（高速）
  60-360秒: キャッシュを返しつつ、バックグラウンドで更新
  360秒以降: キャッシュ無効、サーバーに問い合わせ

  → ユーザーは常に高速なレスポンスを得る
  → データの鮮度と速度のバランス
```

---

## 5. CDNキャッシュ

```
CDN（Content Delivery Network）:
  → 世界中のエッジサーバーにコンテンツをキャッシュ
  → ユーザーに最も近いサーバーから配信

  オリジン → CDNエッジ → ユーザー

  東京ユーザー → 東京エッジ（キャッシュヒット → 5ms）
  大阪ユーザー → 大阪エッジ（キャッシュヒット → 3ms）
  USユーザー  → USエッジ（キャッシュヒット → 2ms）

CDNキャッシュ制御:
  s-maxage: 共有キャッシュ（CDN）での有効期限
  Surrogate-Control: CDN固有のキャッシュ制御
  CDN-Cache-Control: 標準化中のCDN向けヘッダー

  Cache-Control: public, max-age=60, s-maxage=3600
  → ブラウザ: 60秒
  → CDN: 3600秒（1時間）

CloudFront のキャッシュポリシー:
  → TTL: 最小/デフォルト/最大
  → キャッシュキー: URL + ヘッダー + クエリ + Cookie
  → Behavior ごとに設定可能

主要CDNサービス:
  CloudFront (AWS)
  Cloudflare
  Fastly
  Akamai
  Vercel Edge Network
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Cache-Control | キャッシュの制御（max-age, no-cache, no-store） |
| ETag | リソースのバージョン管理（304 Not Modified） |
| Cache Busting | ファイル名ハッシュが最も確実 |
| CDN | エッジキャッシュで世界中に高速配信 |
| SWR | stale-while-revalidate で速度と鮮度を両立 |

---

## 次に読むべきガイド
→ [[04-cors.md]] — CORS

---

## 参考文献
1. RFC 9111. "HTTP Caching." IETF, 2022.
2. RFC 9110. "HTTP Semantics — Conditional Requests." IETF, 2022.
3. web.dev. "HTTP Cache." Google, 2024.
