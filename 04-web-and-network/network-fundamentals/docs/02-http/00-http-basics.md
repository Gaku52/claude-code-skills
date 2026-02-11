# HTTP基礎

> HTTPはWebの基盤プロトコル。リクエスト/レスポンスモデル、メソッド、ステータスコード、ヘッダーの仕組みを理解し、Web開発に必須の知識を固める。

## この章で学ぶこと

- [ ] HTTPのリクエスト/レスポンス構造を理解する
- [ ] HTTPメソッドの意味と使い分けを把握する
- [ ] ステータスコードの分類と主要なコードを学ぶ

---

## 1. HTTPの基本

```
HTTP（HyperText Transfer Protocol）:
  → Web上でデータを転送するためのプロトコル
  → ステートレス（各リクエストは独立）
  → テキストベース（HTTP/1.1）→ バイナリ（HTTP/2以降）

バージョンの歴史:
  HTTP/0.9 (1991): GETのみ、HTML のみ
  HTTP/1.0 (1996): ヘッダー、POST、ステータスコード追加
  HTTP/1.1 (1997): Keep-Alive、チャンク転送、Host ヘッダー必須
  HTTP/2   (2015): バイナリ、多重化、サーバープッシュ
  HTTP/3   (2022): QUIC ベース、UDP 上で動作

リクエスト/レスポンスモデル:
  クライアント                    サーバー
  ┌──────────┐                 ┌──────────┐
  │ ブラウザ  │── リクエスト ──→│ Webサーバー│
  │          │←── レスポンス ──│          │
  └──────────┘                 └──────────┘
```

---

## 2. HTTPリクエスト

```
リクエスト構造:

  ┌─────────────────────────────────────────┐
  │ GET /api/users?page=1 HTTP/1.1          │ ← リクエストライン
  ├─────────────────────────────────────────┤
  │ Host: api.example.com                   │ ← ヘッダー
  │ Accept: application/json                │
  │ Authorization: Bearer eyJhbG...         │
  │ User-Agent: Mozilla/5.0                 │
  │ Accept-Encoding: gzip, deflate          │
  │ Connection: keep-alive                  │
  ├─────────────────────────────────────────┤
  │                                         │ ← 空行（ヘッダー終了）
  ├─────────────────────────────────────────┤
  │ （ボディ — GET の場合は通常なし）         │
  └─────────────────────────────────────────┘

POST リクエストの例:
  POST /api/users HTTP/1.1
  Host: api.example.com
  Content-Type: application/json
  Content-Length: 52

  {"name": "Taro", "email": "taro@example.com"}
```

---

## 3. HTTPメソッド

```
┌────────┬──────────────────────────┬──────┬──────┬──────┐
│ メソッド│ 用途                     │冪等性│安全性│ボディ│
├────────┼──────────────────────────┼──────┼──────┼──────┤
│ GET    │ リソースの取得            │ ✓    │ ✓    │ なし │
│ POST   │ リソースの作成            │ ✗    │ ✗    │ あり │
│ PUT    │ リソースの完全置換        │ ✓    │ ✗    │ あり │
│ PATCH  │ リソースの部分更新        │ ✗    │ ✗    │ あり │
│ DELETE │ リソースの削除            │ ✓    │ ✗    │ 任意 │
│ HEAD   │ レスポンスヘッダーのみ取得│ ✓    │ ✓    │ なし │
│ OPTIONS│ 対応メソッドの確認(CORS)  │ ✓    │ ✓    │ なし │
└────────┴──────────────────────────┴──────┴──────┴──────┘

冪等性（Idempotent）:
  → 同じリクエストを何度送っても結果が同じ
  → GET, PUT, DELETE は冪等
  → POST は冪等ではない（毎回新しいリソースが作成される）

安全性（Safe）:
  → サーバーの状態を変更しない
  → GET, HEAD, OPTIONS は安全

実務での使い分け:
  一覧取得:  GET  /api/users
  詳細取得:  GET  /api/users/123
  作成:      POST /api/users
  完全更新:  PUT  /api/users/123
  部分更新:  PATCH /api/users/123
  削除:      DELETE /api/users/123
```

---

## 4. ステータスコード

```
ステータスコードの分類:
  1xx: 情報（処理継続中）
  2xx: 成功
  3xx: リダイレクト
  4xx: クライアントエラー
  5xx: サーバーエラー

頻出ステータスコード:

  2xx 成功:
  ┌─────┬──────────────────────────────────┐
  │ 200 │ OK — 成功                        │
  │ 201 │ Created — リソース作成成功       │
  │ 204 │ No Content — 成功（ボディなし）  │
  └─────┴──────────────────────────────────┘

  3xx リダイレクト:
  ┌─────┬──────────────────────────────────┐
  │ 301 │ Moved Permanently — 恒久移転     │
  │ 302 │ Found — 一時移転（実装が曖昧）   │
  │ 304 │ Not Modified — キャッシュ有効     │
  │ 307 │ Temporary Redirect — 一時移転    │
  │ 308 │ Permanent Redirect — 恒久移転    │
  └─────┴──────────────────────────────────┘

  4xx クライアントエラー:
  ┌─────┬──────────────────────────────────┐
  │ 400 │ Bad Request — リクエスト不正     │
  │ 401 │ Unauthorized — 未認証            │
  │ 403 │ Forbidden — アクセス禁止         │
  │ 404 │ Not Found — リソースが存在しない │
  │ 405 │ Method Not Allowed               │
  │ 409 │ Conflict — 競合                  │
  │ 422 │ Unprocessable Entity — バリデーションエラー│
  │ 429 │ Too Many Requests — レート制限   │
  └─────┴──────────────────────────────────┘

  5xx サーバーエラー:
  ┌─────┬──────────────────────────────────┐
  │ 500 │ Internal Server Error — 内部エラー│
  │ 502 │ Bad Gateway — 上流サーバーエラー │
  │ 503 │ Service Unavailable — サービス停止│
  │ 504 │ Gateway Timeout — 上流タイムアウト│
  └─────┴──────────────────────────────────┘

  301 vs 302 vs 307 vs 308:
  301: GET に変換される可能性あり（恒久）
  302: GET に変換される可能性あり（一時）
  307: メソッドを維持（一時）← 推奨
  308: メソッドを維持（恒久）← 推奨
```

---

## 5. HTTPヘッダー

```
主要なリクエストヘッダー:
  Host: api.example.com          — 接続先ホスト（HTTP/1.1で必須）
  Accept: application/json       — 受け入れ可能なメディアタイプ
  Content-Type: application/json — ボディのメディアタイプ
  Authorization: Bearer xxx      — 認証情報
  Cookie: session_id=abc123      — クッキー
  User-Agent: Mozilla/5.0...     — クライアント情報
  Accept-Encoding: gzip          — 受け入れ可能な圧縮形式
  Accept-Language: ja,en          — 希望言語
  If-None-Match: "etag-value"    — 条件付きリクエスト
  Origin: https://example.com    — リクエスト元（CORS）

主要なレスポンスヘッダー:
  Content-Type: application/json — ボディのメディアタイプ
  Content-Length: 1234           — ボディのサイズ（バイト）
  Set-Cookie: session_id=abc    — クッキー設定
  Cache-Control: max-age=3600   — キャッシュ制御
  ETag: "abc123"                — リソースのバージョン
  Location: /api/users/456      — リダイレクト先/作成先URI
  Access-Control-Allow-Origin   — CORS許可オリジン
  X-Request-Id: uuid            — リクエスト追跡用（カスタム）

Content-Type の主要な値:
  application/json               — JSON
  application/x-www-form-urlencoded — フォームデータ
  multipart/form-data            — ファイルアップロード
  text/html                      — HTML
  text/plain                     — プレーンテキスト
  application/octet-stream       — バイナリデータ
```

---

## 6. HTTPレスポンス

```
レスポンス構造:

  ┌─────────────────────────────────────────┐
  │ HTTP/1.1 200 OK                         │ ← ステータスライン
  ├─────────────────────────────────────────┤
  │ Content-Type: application/json          │ ← ヘッダー
  │ Content-Length: 85                      │
  │ Cache-Control: no-cache                 │
  │ X-Request-Id: 550e8400-e29b-...        │
  ├─────────────────────────────────────────┤
  │                                         │ ← 空行
  ├─────────────────────────────────────────┤
  │ {                                       │ ← ボディ
  │   "id": "123",                          │
  │   "name": "Taro",                       │
  │   "email": "taro@example.com"           │
  │ }                                       │
  └─────────────────────────────────────────┘

curl でHTTPを確認:
  # リクエストとレスポンスの詳細表示
  $ curl -v https://api.example.com/users/123

  # レスポンスヘッダーのみ
  $ curl -I https://api.example.com/users/123

  # POSTリクエスト
  $ curl -X POST https://api.example.com/users \
    -H "Content-Type: application/json" \
    -d '{"name": "Taro", "email": "taro@example.com"}'
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| HTTP | ステートレスなリクエスト/レスポンスプロトコル |
| メソッド | GET(取得), POST(作成), PUT(更新), DELETE(削除) |
| ステータス | 2xx(成功), 3xx(リダイレクト), 4xx(クライアントエラー), 5xx(サーバーエラー) |
| ヘッダー | Content-Type, Authorization, Cache-Control 等 |

---

## 次に読むべきガイド
→ [[01-http2-and-http3.md]] — HTTP/2とHTTP/3

---

## 参考文献
1. RFC 9110. "HTTP Semantics." IETF, 2022.
2. RFC 9112. "HTTP/1.1." IETF, 2022.
3. MDN Web Docs. "HTTP." Mozilla, 2024.
