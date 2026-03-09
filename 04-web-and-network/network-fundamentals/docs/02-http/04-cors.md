# CORS（Cross-Origin Resource Sharing）

> CORSはブラウザのセキュリティ機構「同一オリジンポリシー」を安全に緩和する仕組み。プリフライトリクエスト、許可ヘッダー、Credentialsの設定を理解し、正しくCORSを構成する。

## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- [HTTP基礎](./00-http-basics.md) — リクエスト/レスポンス、ヘッダー、ステータスコードの仕組み
- [ブラウザのセキュリティモデル](../../browser-and-web-platform/docs/00-browser-engine/03-browser-security-model.md) — Same-Origin Policy、サンドボックス、セキュリティ境界
- [TLS/SSL](../03-security/00-tls-ssl.md) — HTTPS通信の暗号化と証明書の基礎

CORSは同一オリジンポリシー（Same-Origin Policy）という根本的なWebセキュリティ機構を理解していないと本質を掴めない。ブラウザがなぜクロスオリジンリクエストを制限するのか、どのような攻撃を防いでいるのかを知ることで、CORSの設計意図と正しい設定方法が明確になる。

---

## この章で学ぶこと

- [ ] 同一オリジンポリシーの起源と目的を理解する
- [ ] CORSの仕組みとブラウザの挙動を把握する
- [ ] シンプルリクエストとプリフライトの違いを明確に区別する
- [ ] サーバー側のCORS設定方法（Express, nginx, 各種フレームワーク）を学ぶ
- [ ] Credentials（Cookie/認証情報）を伴うCORSの注意点を理解する
- [ ] 開発環境と本番環境それぞれでのCORS戦略を習得する
- [ ] CORSに関するセキュリティリスクとベストプラクティスを把握する

---

## 1. 同一オリジンポリシー（Same-Origin Policy）

### 1.1 オリジンの定義

同一オリジンポリシーを理解するには、まず「オリジン」の定義を正確に把握する必要がある。オリジンは以下の3要素の組み合わせで決定される。

```
オリジン = スキーム + ホスト + ポート

  https://example.com:443/path/to/resource?query=value#fragment
  ↑        ↑            ↑    ↑                ↑         ↑
  スキーム   ホスト       ポート パス            クエリ     フラグメント
  (scheme)  (host)      (port) (path)         (query)   (fragment)

  ※ オリジンの判定に使われるのはスキーム・ホスト・ポートの3要素のみ
  ※ パス、クエリ、フラグメントはオリジンの判定に含まれない
```

### 1.2 同一オリジン判定の具体例

```
基準URL: https://www.example.com/page

比較対象                                結果      理由
───────────────────────────────────────────────────────────────
https://www.example.com/other          同一 ○    パスのみ異なる
https://www.example.com/page?q=1       同一 ○    クエリのみ異なる
https://www.example.com:443/page       同一 ○    HTTPSのデフォルトポート
http://www.example.com/page            異なる ✗  スキームが異なる
https://api.example.com/page           異なる ✗  ホスト（サブドメイン）が異なる
https://example.com/page               異なる ✗  ホスト（wwwの有無）が異なる
https://www.example.com:8443/page      異なる ✗  ポートが異なる
https://www.example.org/page           異なる ✗  ドメインが異なる

※ 重要: サブドメインが異なるだけでも別オリジンとなる
   www.example.com と api.example.com は別オリジン
```

### 1.3 同一オリジンポリシーの歴史的背景

同一オリジンポリシーは1995年にNetscape Navigator 2.02で初めて導入されたセキュリティモデルである。Webが発展するにつれ、異なるWebサイト間でのデータ窃取を防ぐ根本的なセキュリティ境界として機能してきた。

```
同一オリジンポリシーの保護モデル:

  攻撃シナリオ（SOPがない場合）:
  ┌──────────────────────────────────────────────────┐
  │ ユーザーが evil.com を閲覧中                      │
  │                                                    │
  │  evil.com のJS                                     │
  │    │                                               │
  │    │── fetch("https://bank.com/api/balance") ──→  │
  │    │   （ユーザーのCookieが自動送信される）         │
  │    │                                               │
  │    │←── { balance: 1000000 } ───────────────────  │
  │    │                                               │
  │    │── evil.com のサーバーに送信 ──→               │
  │    │   （ユーザーの残高情報が盗まれる）             │
  │                                                    │
  │  ※ SOPがあるため、このレスポンスの読み取りは       │
  │    ブラウザによってブロックされる                   │
  └──────────────────────────────────────────────────┘
```

### 1.4 SOPの適用範囲

同一オリジンポリシーは全てのリソースに一律に適用されるわけではない。歴史的な理由から、以下のように適用範囲が異なる。

```
SOPの適用対象と非適用対象:

  ┌─────────────────────────────────┐
  │     SOP が制限するもの          │
  ├─────────────────────────────────┤
  │ ・fetch / XMLHttpRequest        │
  │ ・Canvas への他オリジン画像描画 │
  │   （tainted canvas）            │
  │ ・Web Storage（localStorage等） │
  │ ・IndexedDB                     │
  │ ・Cookie（別途ルールあり）      │
  │ ・iframe の DOM アクセス        │
  └─────────────────────────────────┘

  ┌─────────────────────────────────┐
  │     SOP が制限しないもの        │
  ├─────────────────────────────────┤
  │ ・<img src="...">               │
  │ ・<script src="...">            │
  │ ・<link rel="stylesheet" ...>   │
  │ ・<video> / <audio>             │
  │ ・<form action="...">           │
  │ ・@font-face                    │
  │ ・<iframe>（表示は可、DOM不可） │
  └─────────────────────────────────┘

  ※ <script> や <img> がSOPの制限を受けないのは、
    Web初期からクロスオリジンでの利用が一般的だったため
  ※ ただし、これがJSONPやCSRF等の攻撃手法の温床にもなった
```

---

## 2. CORSの仕組み

### 2.1 CORSの全体像

CORS（Cross-Origin Resource Sharing）は、同一オリジンポリシーを安全に緩和するためのHTTPヘッダーベースの仕組みである。サーバーが「このオリジンからのアクセスを許可する」と明示的に宣言することで、ブラウザがクロスオリジンリクエストのレスポンスへのアクセスを許可する。

```
CORSの基本概念図:

  ┌───────────────┐                        ┌───────────────┐
  │   ブラウザ     │                        │   サーバー     │
  │               │                        │               │
  │ https://app   │                        │ https://api   │
  │ .example.com  │                        │ .example.com  │
  │               │                        │               │
  │  フロントエンド │                        │  バックエンド   │
  │  (React等)    │                        │  (Express等)  │
  │               │                        │               │
  │  ①リクエスト   │── HTTP Request ──→    │               │
  │   送信        │   Origin: https://     │  ②オリジン     │
  │               │   app.example.com      │    検証        │
  │               │                        │               │
  │  ④レスポンス   │←── HTTP Response ──   │  ③CORSヘッダ   │
  │   利用可否    │   Access-Control-      │    付与        │
  │   判定        │   Allow-Origin:        │               │
  │               │   https://app...       │               │
  └───────────────┘                        └───────────────┘

  重要: CORSはブラウザのセキュリティ機構
  → サーバー間通信（curl, サーバーサイドHTTPクライアント）には適用されない
  → ブラウザが「レスポンスをJSに渡すかどうか」を判断する仕組み
  → リクエスト自体はサーバーに到達する（※プリフライトを除く）
```

### 2.2 シンプルリクエスト（Simple Request）

シンプルリクエストは、プリフライトなしで直接サーバーに送信されるリクエストである。以下の全ての条件を満たす場合にシンプルリクエストとして扱われる。

```
シンプルリクエストの条件（全て満たす必要がある）:

  ┌────────────────────────────────────────────────────┐
  │ 条件1: HTTPメソッド                                │
  │   GET, HEAD, POST のいずれか                       │
  ├────────────────────────────────────────────────────┤
  │ 条件2: ヘッダー（以下のみ許可）                     │
  │   ・Accept                                         │
  │   ・Accept-Language                                │
  │   ・Content-Language                               │
  │   ・Content-Type（条件3を参照）                     │
  │   ・Range（単純な範囲指定のみ）                     │
  ├────────────────────────────────────────────────────┤
  │ 条件3: Content-Type（以下のいずれか）               │
  │   ・application/x-www-form-urlencoded              │
  │   ・multipart/form-data                            │
  │   ・text/plain                                     │
  ├────────────────────────────────────────────────────┤
  │ 条件4: ReadableStream を使用していない              │
  ├────────────────────────────────────────────────────┤
  │ 条件5: XMLHttpRequestUpload にイベントリスナーが    │
  │        設定されていない                             │
  └────────────────────────────────────────────────────┘

  シンプルリクエストのフロー:

     ブラウザ                              サーバー
     │                                     │
     │── GET /api/public/data ──────→     │
     │   Host: api.example.com             │
     │   Origin: https://app.example.com   │
     │                                     │
     │                              ┌──────┤ オリジンを検証し
     │                              │      │ CORSヘッダーを付与
     │                              └──────┤
     │                                     │
     │←── 200 OK ──────────────────       │
     │   Access-Control-Allow-Origin:      │
     │     https://app.example.com         │
     │   Content-Type: application/json    │
     │                                     │
     │   {"data": "public info"}           │
     │                                     │

  ブラウザの判定:
  ・Allow-Origin がリクエストの Origin と一致 → JSにレスポンスを渡す
  ・Allow-Origin がない or 不一致 → CORSエラー（レスポンスを破棄）
```

### 2.3 プリフライトリクエスト（Preflight Request）

シンプルリクエストの条件を満たさない場合、ブラウザは実際のリクエストを送信する前に、OPTIONSメソッドによるプリフライトリクエストを送信して、サーバーの許可を確認する。

```
プリフライトが発生する典型的なケース:

  ① HTTPメソッドが PUT / DELETE / PATCH
  ② Content-Type が application/json
  ③ カスタムヘッダーを使用（Authorization, X-Custom-Header 等）
  ④ 上記の組み合わせ

プリフライトの詳細シーケンス:

     ブラウザ                              サーバー
     │                                     │
     │  ※ fetch() でPUTリクエストを         │
     │    発行しようとする                   │
     │                                     │
     │  [Phase 1: プリフライト]             │
     │                                     │
     │── OPTIONS /api/users/123 ────→     │
     │   Host: api.example.com             │
     │   Origin: https://app.example.com   │
     │   Access-Control-Request-Method:    │
     │     PUT                             │
     │   Access-Control-Request-Headers:   │
     │     Content-Type, Authorization     │
     │                                     │
     │                              ┌──────┤
     │                              │ 許可  │
     │                              │ 判定  │
     │                              └──────┤
     │                                     │
     │←── 204 No Content ──────────       │
     │   Access-Control-Allow-Origin:      │
     │     https://app.example.com         │
     │   Access-Control-Allow-Methods:     │
     │     GET, POST, PUT, DELETE          │
     │   Access-Control-Allow-Headers:     │
     │     Content-Type, Authorization     │
     │   Access-Control-Max-Age: 86400     │
     │                                     │
     │  ※ プリフライト成功                  │
     │  ※ Max-Ageの間はキャッシュされる     │
     │                                     │
     │  [Phase 2: 実際のリクエスト]         │
     │                                     │
     │── PUT /api/users/123 ────────→     │
     │   Host: api.example.com             │
     │   Origin: https://app.example.com   │
     │   Content-Type: application/json    │
     │   Authorization: Bearer eyJhbG...   │
     │                                     │
     │   {"name": "Updated Name"}          │
     │                                     │
     │←── 200 OK ──────────────────       │
     │   Access-Control-Allow-Origin:      │
     │     https://app.example.com         │
     │   Content-Type: application/json    │
     │                                     │
     │   {"id": 123, "name": "Updated"}    │
     │                                     │

  注意: プリフライトが失敗した場合、実際のリクエストは送信されない
  → サーバー側でOPTIONSリクエストのハンドリングが必須
```

### 2.4 プリフライトキャッシュ

プリフライトリクエストは毎回のリクエストごとに送信されると、パフォーマンスに影響を与える。`Access-Control-Max-Age` ヘッダーにより、プリフライト結果をキャッシュできる。

```
プリフライトキャッシュの仕組み:

  Max-Age: 86400（24時間）の場合

  時刻 00:00  最初のリクエスト
  ├── OPTIONS /api/data ──→ （プリフライト送信）
  ├── 204 応答 ←──
  ├── PUT /api/data ──→ （実際のリクエスト）
  └── 200 応答 ←──

  時刻 01:00  2回目のリクエスト
  ├── （プリフライトはキャッシュヒット → 送信不要）
  ├── PUT /api/data ──→ （直接送信）
  └── 200 応答 ←──

  時刻 12:00  3回目のリクエスト
  ├── （まだキャッシュ有効）
  ├── DELETE /api/data ──→
  └── 200 応答 ←──

  時刻 24:01  キャッシュ期限切れ後
  ├── OPTIONS /api/data ──→ （再度プリフライト送信）
  ├── 204 応答 ←──
  ├── PUT /api/data ──→
  └── 200 応答 ←──

  ブラウザごとの Max-Age 上限:
  ┌──────────────────────┬────────────────┐
  │ ブラウザ              │ 上限値          │
  ├──────────────────────┼────────────────┤
  │ Chrome/Edge          │ 7200秒（2時間） │
  │ Firefox              │ 86400秒（24h）  │
  │ Safari               │ 604800秒（7日） │
  └──────────────────────┴────────────────┘

  ※ サーバーが Max-Age: 86400 を返しても、Chromeでは
    7200秒に切り詰められる
  ※ Max-Age を省略した場合、デフォルトは5秒
```

---

## 3. CORSヘッダー詳細

### 3.1 レスポンスヘッダー一覧

```
CORSレスポンスヘッダー完全リファレンス:

  ┌───────────────────────────────┬──────────────────────────────────┐
  │ ヘッダー                      │ 説明・用途                       │
  ├───────────────────────────────┼──────────────────────────────────┤
  │ Access-Control-Allow-Origin   │ 許可するオリジンを指定           │
  │                               │ 値: 特定オリジン or *            │
  │                               │ 例: https://app.example.com     │
  │                               │ ※ 複数オリジンは直接指定不可     │
  ├───────────────────────────────┼──────────────────────────────────┤
  │ Access-Control-Allow-Methods  │ 許可するHTTPメソッドを列挙       │
  │                               │ プリフライトレスポンスで使用     │
  │                               │ 例: GET, POST, PUT, DELETE      │
  ├───────────────────────────────┼──────────────────────────────────┤
  │ Access-Control-Allow-Headers  │ 許可するリクエストヘッダーを列挙 │
  │                               │ プリフライトレスポンスで使用     │
  │                               │ 例: Content-Type, Authorization │
  ├───────────────────────────────┼──────────────────────────────────┤
  │ Access-Control-Expose-Headers │ JSからアクセス可能なレスポンス   │
  │                               │ ヘッダーを指定                   │
  │                               │ デフォルトで公開: Content-Type,  │
  │                               │ Cache-Control, Expires 等        │
  │                               │ 例: X-Request-Id, X-Total-Count │
  ├───────────────────────────────┼──────────────────────────────────┤
  │ Access-Control-Allow-         │ Credentials（Cookie、認証情報）  │
  │ Credentials                   │ の送信を許可するか               │
  │                               │ 値: true のみ（falseは省略）     │
  │                               │ ※ Allow-Origin: * との併用不可   │
  ├───────────────────────────────┼──────────────────────────────────┤
  │ Access-Control-Max-Age        │ プリフライト結果のキャッシュ秒数 │
  │                               │ 値: 秒数（整数）                 │
  │                               │ 例: 86400（24時間）              │
  │                               │ ※ ブラウザごとに上限あり         │
  └───────────────────────────────┴──────────────────────────────────┘
```

### 3.2 リクエストヘッダー一覧

```
CORSリクエストヘッダー（ブラウザが自動付与）:

  ┌────────────────────────────────┬─────────────────────────────────┐
  │ ヘッダー                       │ 説明・用途                      │
  ├────────────────────────────────┼─────────────────────────────────┤
  │ Origin                         │ リクエスト元のオリジン          │
  │                                │ ブラウザが自動的に付与          │
  │                                │ JSから変更不可                  │
  │                                │ 例: https://app.example.com    │
  ├────────────────────────────────┼─────────────────────────────────┤
  │ Access-Control-Request-Method  │ 実際に使用するHTTPメソッド      │
  │                                │ プリフライト（OPTIONS）で使用   │
  │                                │ 例: PUT                        │
  ├────────────────────────────────┼─────────────────────────────────┤
  │ Access-Control-Request-Headers │ 実際に使用するヘッダー一覧      │
  │                                │ プリフライト（OPTIONS）で使用   │
  │                                │ 例: Content-Type, Authorization│
  └────────────────────────────────┴─────────────────────────────────┘

  ※ これらのヘッダーはブラウザが自動で設定する
  ※ JavaScript（fetch API等）からこれらを手動設定することはできない
  ※ Origin ヘッダーの偽装は通常のブラウザでは不可能
```

### 3.3 Credentials（資格情報）とCORS

Cookie、Authorization ヘッダー、TLSクライアント証明書などの資格情報を伴うクロスオリジンリクエストには特別なルールが適用される。

```
Credentialsモードの設定:

  fetch APIの場合:
  ┌───────────────────────┬──────────────────────────────────────┐
  │ credentials 値         │ 動作                                │
  ├───────────────────────┼──────────────────────────────────────┤
  │ "omit"                │ Cookieを一切送信しない               │
  │                       │ レスポンスのCookieも無視             │
  ├───────────────────────┼──────────────────────────────────────┤
  │ "same-origin"（既定） │ 同一オリジンのみCookieを送信         │
  │                       │ クロスオリジンでは送信しない         │
  ├───────────────────────┼──────────────────────────────────────┤
  │ "include"             │ クロスオリジンでもCookieを送信       │
  │                       │ サーバー側の許可が必須               │
  └───────────────────────┴──────────────────────────────────────┘

  Credentials使用時の制約:
  ┌──────────────────────────────────────────────────────┐
  │ credentials: "include" を使用する場合                 │
  │                                                      │
  │ サーバーは以下を全て満たす必要がある:                 │
  │                                                      │
  │ 1. Access-Control-Allow-Credentials: true            │
  │ 2. Access-Control-Allow-Origin: （特定のオリジン）    │
  │    ※ ワイルドカード "*" は使用不可                    │
  │ 3. Access-Control-Allow-Headers: （特定のヘッダー）   │
  │    ※ ワイルドカード "*" は使用不可                    │
  │ 4. Access-Control-Allow-Methods: （特定のメソッド）   │
  │    ※ ワイルドカード "*" は使用不可                    │
  │ 5. Access-Control-Expose-Headers: （特定のヘッダー）  │
  │    ※ ワイルドカード "*" は使用不可                    │
  └──────────────────────────────────────────────────────┘

  ※ Credentials モードでは全てのワイルドカード指定が無効になる
  ※ これはセキュリティ上の重要な制約
```

---

## 4. サーバー側の設定

### 4.1 Express.js での CORS 設定

```typescript
// ============================================================
// Express.js CORS設定 - 完全版
// ============================================================

import express from 'express';
import cors from 'cors';

const app = express();

// --------------------------------------------------
// 方法1: cors ミドルウェア（推奨）
// --------------------------------------------------

// 基本設定
app.use(cors({
  // 許可するオリジンのリスト
  origin: [
    'https://app.example.com',
    'https://admin.example.com',
    'https://staging.example.com',
  ],
  // 許可するHTTPメソッド
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  // 許可するリクエストヘッダー
  allowedHeaders: [
    'Content-Type',
    'Authorization',
    'X-Requested-With',
    'X-Request-Id',
  ],
  // JSからアクセス可能にするレスポンスヘッダー
  exposedHeaders: [
    'X-Total-Count',
    'X-Request-Id',
    'X-RateLimit-Remaining',
  ],
  // Credentials（Cookie等）を許可
  credentials: true,
  // プリフライト結果のキャッシュ時間（秒）
  maxAge: 86400,
  // OPTIONSリクエストに対して204を返す（デフォルト: 204）
  optionsSuccessStatus: 204,
}));

// --------------------------------------------------
// 方法2: 動的オリジン検証（パターンマッチング）
// --------------------------------------------------

const corsOptionsWithDynamicOrigin = cors({
  origin: (origin, callback) => {
    // origin が undefined の場合は同一オリジンリクエスト
    // （またはサーバー間通信）
    if (!origin) {
      return callback(null, true);
    }

    // ホワイトリスト方式
    const whitelist = [
      'https://app.example.com',
      'https://admin.example.com',
    ];

    if (whitelist.includes(origin)) {
      return callback(null, true);
    }

    // サブドメインのパターンマッチ
    const subdomainPattern = /^https:\/\/[\w-]+\.example\.com$/;
    if (subdomainPattern.test(origin)) {
      return callback(null, true);
    }

    // 開発環境のlocalhostを許可
    if (process.env.NODE_ENV === 'development') {
      const localhostPattern = /^http:\/\/localhost:\d+$/;
      if (localhostPattern.test(origin)) {
        return callback(null, true);
      }
    }

    // 許可されないオリジン
    callback(new Error(`Origin ${origin} is not allowed by CORS`));
  },
  credentials: true,
  maxAge: 3600,
});

app.use(corsOptionsWithDynamicOrigin);

// --------------------------------------------------
// 方法3: ルート単位でのCORS設定
// --------------------------------------------------

// 公開APIはワイルドカード許可
app.get('/api/public/*', cors({ origin: '*' }), (req, res) => {
  res.json({ data: 'public data' });
});

// プライベートAPIは特定オリジンのみ
const privateCors = cors({
  origin: 'https://app.example.com',
  credentials: true,
});

app.get('/api/private/*', privateCors, (req, res) => {
  res.json({ data: 'private data' });
});

// --------------------------------------------------
// 方法4: 手動実装（ミドルウェアを使わない場合）
// --------------------------------------------------

app.use((req, res, next) => {
  const allowedOrigins = [
    'https://app.example.com',
    'https://admin.example.com',
  ];
  const origin = req.headers.origin;

  // オリジンの検証
  if (origin && allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    // Vary ヘッダーを設定（CDNキャッシュ対策）
    res.setHeader('Vary', 'Origin');
  }

  res.setHeader(
    'Access-Control-Allow-Methods',
    'GET, POST, PUT, PATCH, DELETE, OPTIONS'
  );
  res.setHeader(
    'Access-Control-Allow-Headers',
    'Content-Type, Authorization, X-Requested-With'
  );
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Max-Age', '86400');
  res.setHeader(
    'Access-Control-Expose-Headers',
    'X-Total-Count, X-Request-Id'
  );

  // プリフライトリクエストへの応答
  if (req.method === 'OPTIONS') {
    return res.sendStatus(204);
  }

  next();
});
```

### 4.2 nginx での CORS 設定

```nginx
# ============================================================
# nginx CORS設定 - 本番環境向け完全版
# ============================================================

# オリジンのホワイトリストを map で定義
map $http_origin $cors_origin {
    default "";
    "https://app.example.com"     "https://app.example.com";
    "https://admin.example.com"   "https://admin.example.com";
    "https://staging.example.com" "https://staging.example.com";
    # 正規表現も使用可能
    ~^https://[\w-]+\.example\.com$  $http_origin;
}

server {
    listen 443 ssl;
    server_name api.example.com;

    location /api/ {
        # プリフライトリクエスト（OPTIONS）の処理
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin'
                       $cors_origin always;
            add_header 'Access-Control-Allow-Methods'
                       'GET, POST, PUT, PATCH, DELETE, OPTIONS' always;
            add_header 'Access-Control-Allow-Headers'
                       'Content-Type, Authorization, X-Requested-With' always;
            add_header 'Access-Control-Allow-Credentials'
                       'true' always;
            add_header 'Access-Control-Max-Age'
                       86400 always;
            add_header 'Content-Type'
                       'text/plain charset=UTF-8';
            add_header 'Content-Length' 0;
            return 204;
        }

        # 通常リクエストへのCORSヘッダー付与
        add_header 'Access-Control-Allow-Origin'
                   $cors_origin always;
        add_header 'Access-Control-Allow-Credentials'
                   'true' always;
        add_header 'Access-Control-Expose-Headers'
                   'X-Total-Count, X-Request-Id' always;
        add_header 'Vary' 'Origin' always;

        proxy_pass http://backend_upstream;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # 公開アセット（CORSヘッダーなし or ワイルドカード）
    location /static/ {
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Cache-Control' 'public, max-age=31536000';
        root /var/www/assets;
    }
}
```

### 4.3 fetch API によるクロスオリジンリクエスト

```typescript
// ============================================================
// fetch API CORS リクエスト例
// ============================================================

// --- 例1: シンプルリクエスト（GETでJSONを取得） ---
async function fetchPublicData(): Promise<void> {
  try {
    const response = await fetch('https://api.example.com/api/public/data', {
      method: 'GET',
      // シンプルリクエストの条件を満たすため
      // プリフライトは発生しない
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    console.log('取得成功:', data);
  } catch (error) {
    if (error instanceof TypeError && error.message === 'Failed to fetch') {
      // CORSエラーの場合、TypeError が発生する
      console.error('CORSエラーまたはネットワークエラー');
    } else {
      console.error('その他のエラー:', error);
    }
  }
}

// --- 例2: Credentials付きリクエスト（Cookie送信） ---
async function fetchWithCredentials(): Promise<void> {
  const response = await fetch('https://api.example.com/api/user/profile', {
    method: 'GET',
    credentials: 'include', // Cookie を送信
    // credentials: 'include' を指定した場合、
    // サーバーは Access-Control-Allow-Credentials: true を返す必要がある
    // かつ Allow-Origin に * は使用不可
  });

  const profile = await response.json();
  console.log('プロフィール:', profile);
}

// --- 例3: プリフライトが発生するリクエスト ---
async function updateUser(userId: number, data: object): Promise<void> {
  const response = await fetch(
    `https://api.example.com/api/users/${userId}`,
    {
      method: 'PUT',                      // → プリフライト発生（シンプルでない）
      headers: {
        'Content-Type': 'application/json', // → プリフライト発生
        'Authorization': 'Bearer eyJhbG...', // → プリフライト発生
        'X-Request-Id': crypto.randomUUID(), // → プリフライト発生
      },
      credentials: 'include',
      body: JSON.stringify(data),
    }
  );

  // レスポンスヘッダーへのアクセス
  // ※ Expose-Headers に含まれるヘッダーのみ取得可能
  const requestId = response.headers.get('X-Request-Id');
  const totalCount = response.headers.get('X-Total-Count');

  const result = await response.json();
  console.log('更新結果:', result);
}

// --- 例4: AbortController によるタイムアウト付きCORSリクエスト ---
async function fetchWithTimeout(
  url: string,
  timeoutMs: number = 5000
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: 'GET',
      credentials: 'include',
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
      },
    });
    return response;
  } finally {
    clearTimeout(timeoutId);
  }
}
```

---

## 5. よくあるCORSエラーと対処法

### 5.1 エラーパターン一覧

CORSエラーはブラウザのコンソールに表示されるが、セキュリティ上の理由から詳細なエラー情報はJavaScriptからは取得できない。以下に代表的なエラーパターンと対処法を網羅する。

```
エラーパターン比較表:

  ┌───┬──────────────────────────────┬──────────────────────┬─────────────────────────────┐
  │ # │ コンソールメッセージ（要約）  │ 原因                 │ 対処法                      │
  ├───┼──────────────────────────────┼──────────────────────┼─────────────────────────────┤
  │ 1 │ No 'Access-Control-Allow-    │ サーバーが CORS      │ サーバーに Allow-Origin      │
  │   │ Origin' header is present    │ ヘッダーを返して     │ ヘッダーを設定する           │
  │   │                              │ いない               │                             │
  ├───┼──────────────────────────────┼──────────────────────┼─────────────────────────────┤
  │ 2 │ The value of the 'Access-    │ Allow-Origin の値が  │ Origin と完全一致する値を    │
  │   │ Control-Allow-Origin' header │ リクエストの Origin  │ 返す、またはホワイトリスト   │
  │   │ must not be the wildcard '*' │ と一致しない、       │ で動的に設定                │
  │   │ when credentials mode is     │ credentials使用時に  │                             │
  │   │ 'include'                    │ * を使用している     │                             │
  ├───┼──────────────────────────────┼──────────────────────┼─────────────────────────────┤
  │ 3 │ Response to preflight        │ OPTIONS リクエスト   │ OPTIONS メソッドの           │
  │   │ request doesn't pass access  │ に対する応答が不正   │ ハンドラーを実装する         │
  │   │ control check                │                      │                             │
  ├───┼──────────────────────────────┼──────────────────────┼─────────────────────────────┤
  │ 4 │ Method PUT is not allowed by │ Allow-Methods に     │ 使用するメソッドを           │
  │   │ Access-Control-Allow-Methods │ 必要なメソッドが     │ Allow-Methods に追加         │
  │   │                              │ 含まれていない       │                             │
  ├───┼──────────────────────────────┼──────────────────────┼─────────────────────────────┤
  │ 5 │ Request header field         │ Allow-Headers に     │ 使用するヘッダーを           │
  │   │ Authorization is not allowed │ 必要なヘッダーが     │ Allow-Headers に追加         │
  │   │ by Access-Control-Allow-     │ 含まれていない       │                             │
  │   │ Headers                      │                      │                             │
  ├───┼──────────────────────────────┼──────────────────────┼─────────────────────────────┤
  │ 6 │ Redirect is not allowed for  │ プリフライトの応答   │ OPTIONS に対して直接         │
  │   │ a preflight request          │ がリダイレクト       │ 応答する（リダイレクト不可） │
  │   │                              │ (301/302) を返した   │                             │
  └───┴──────────────────────────────┴──────────────────────┴─────────────────────────────┘
```

### 5.2 デバッグ手順

CORSエラーが発生した場合の体系的なデバッグ手順を示す。

```
CORSエラーのデバッグフローチャート:

  START: CORSエラーがコンソールに表示された
    │
    ├── Step 1: ブラウザのDevToolsでNetworkタブを確認
    │   │
    │   ├── OPTIONSリクエストがあるか？
    │   │   ├── YES → プリフライトの応答を確認（Step 2a）
    │   │   └── NO  → シンプルリクエストの応答を確認（Step 2b）
    │   │
    │   Step 2a: プリフライト応答の確認
    │   ├── ステータスコードは 200 or 204 か？
    │   │   ├── NO → OPTIONSハンドラーを実装/修正
    │   │   └── YES → レスポンスヘッダーを確認
    │   │       ├── Allow-Origin は正しいか？
    │   │       ├── Allow-Methods に必要なメソッドがあるか？
    │   │       └── Allow-Headers に必要なヘッダーがあるか？
    │   │
    │   Step 2b: レスポンスヘッダーの確認
    │   ├── Allow-Origin ヘッダーが存在するか？
    │   │   ├── NO → サーバーにCORS設定を追加
    │   │   └── YES → 値がリクエストのOriginと一致するか確認
    │   │
    │   Step 3: Credentials関連の確認
    │   ├── credentials: 'include' を使用しているか？
    │   │   ├── YES → Allow-Origin が * になっていないか確認
    │   │   │         Allow-Credentials: true があるか確認
    │   │   └── NO → Step 4 へ
    │   │
    │   Step 4: curl で直接サーバーの応答を確認
    │       $ curl -v -X OPTIONS \
    │         -H "Origin: https://app.example.com" \
    │         -H "Access-Control-Request-Method: PUT" \
    │         https://api.example.com/api/data
    │
    └── END: 原因特定 → 修正 → ブラウザキャッシュクリア → 再検証
```

### 5.3 curl によるCORSデバッグコマンド

```bash
# ============================================================
# curl を使った CORS デバッグ
# ============================================================

# --- プリフライトリクエストのシミュレーション ---
curl -v -X OPTIONS \
  -H "Origin: https://app.example.com" \
  -H "Access-Control-Request-Method: PUT" \
  -H "Access-Control-Request-Headers: Content-Type, Authorization" \
  https://api.example.com/api/users

# 期待される応答ヘッダー:
# < HTTP/2 204
# < access-control-allow-origin: https://app.example.com
# < access-control-allow-methods: GET, POST, PUT, DELETE
# < access-control-allow-headers: Content-Type, Authorization
# < access-control-max-age: 86400

# --- シンプルリクエストのシミュレーション ---
curl -v \
  -H "Origin: https://app.example.com" \
  https://api.example.com/api/public/data

# --- Credentials付きリクエストのシミュレーション ---
curl -v \
  -H "Origin: https://app.example.com" \
  -H "Cookie: session=abc123" \
  https://api.example.com/api/user/profile

# 期待される応答ヘッダー:
# < access-control-allow-origin: https://app.example.com
# < access-control-allow-credentials: true
# （Allow-Origin が * だとブラウザではエラーになる）
```

---

## 6. 開発環境でのCORS対策

### 6.1 プロキシによる回避（推奨）

開発環境では、CORS自体を回避するアプローチが最もトラブルが少ない。フロントエンドの開発サーバーにプロキシを設定し、APIリクエストを中継させることで同一オリジンとなり、CORSが不要になる。

```
プロキシの仕組み:

  従来（CORS必要）:
  ┌─────────────────┐                    ┌─────────────────┐
  │ ブラウザ         │──── 直接通信 ───→│ APIサーバー      │
  │ localhost:5173  │     異なるオリジン  │ localhost:8080  │
  │                 │←── CORSエラー ──  │                 │
  └─────────────────┘                    └─────────────────┘

  プロキシ使用（CORS不要）:
  ┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
  │ ブラウザ         │──→│ Vite Dev     │──→│ APIサーバー      │
  │ localhost:5173  │    │ Server       │    │ localhost:8080  │
  │                 │    │ /api → proxy │    │                 │
  │ /api/users      │    │              │    │ /api/users      │
  │ は同一オリジン   │←──│              │←──│                 │
  └─────────────────┘    └──────────────┘    └─────────────────┘

  ブラウザから見ると /api/users は localhost:5173 への
  リクエストなので、同一オリジン → CORSは発生しない
```

```typescript
// ============================================================
// Vite のプロキシ設定（vite.config.ts）
// ============================================================

import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      // /api で始まるリクエストをバックエンドに転送
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        // リクエストパスの書き換え（必要な場合）
        // rewrite: (path) => path.replace(/^\/api/, ''),
      },

      // WebSocket のプロキシ
      '/ws': {
        target: 'ws://localhost:8080',
        ws: true,
      },

      // 複数のバックエンドへの振り分け
      '/auth': {
        target: 'http://localhost:9000',
        changeOrigin: true,
      },
    },
  },
});
```

```typescript
// ============================================================
// webpack-dev-server のプロキシ設定（webpack.config.js）
// ============================================================

module.exports = {
  devServer: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        pathRewrite: { '^/api': '/api' },
        // エラーハンドリング
        onError: (err, req, res) => {
          console.error('Proxy error:', err);
          res.writeHead(502, { 'Content-Type': 'text/plain' });
          res.end('Bad Gateway: Backend server is not responding');
        },
      },
    },
  },
};
```

```typescript
// ============================================================
// Next.js のリライト設定（next.config.js）
// ============================================================

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/api/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
```

### 6.2 環境別CORS設定パターン

```typescript
// ============================================================
// 環境別CORS設定（Express.js）
// ============================================================

import cors from 'cors';

function getCorsOptions(): cors.CorsOptions {
  const env = process.env.NODE_ENV || 'development';

  switch (env) {
    case 'production':
      return {
        origin: [
          'https://app.example.com',
          'https://admin.example.com',
        ],
        credentials: true,
        maxAge: 86400,
      };

    case 'staging':
      return {
        origin: [
          'https://staging-app.example.com',
          'https://staging-admin.example.com',
        ],
        credentials: true,
        maxAge: 3600,
      };

    case 'development':
      return {
        origin: (origin, callback) => {
          // 開発環境では localhost の任意のポートを許可
          if (!origin || /^http:\/\/localhost:\d+$/.test(origin)) {
            callback(null, true);
          } else {
            callback(new Error('Not allowed by CORS'));
          }
        },
        credentials: true,
        maxAge: 0, // キャッシュなし（デバッグ容易性のため）
      };

    case 'test':
      return {
        origin: '*', // テスト環境では全オリジン許可
        credentials: false,
      };

    default:
      return {
        origin: false, // CORS無効（安全側に倒す）
      };
  }
}

app.use(cors(getCorsOptions()));
```

---

## 7. フレームワーク別CORS設定

### 7.1 主要フレームワーク比較表

```
フレームワーク別CORS設定方法の比較:

  ┌────────────────┬──────────────────────┬──────────────────────────────┐
  │ フレームワーク  │ 設定方法              │ 特徴                         │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Express.js     │ cors ミドルウェア     │ 最も柔軟、動的オリジン対応   │
  │                │ or 手動ミドルウェア   │ npm: cors パッケージ          │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Fastify        │ @fastify/cors        │ Express cors と類似のAPI     │
  │                │ プラグイン            │ スキーマバリデーション対応   │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Hono           │ cors() ミドルウェア   │ 軽量、Edge Runtime対応       │
  │                │ (hono/cors)          │ Cloudflare Workers等で利用   │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Django         │ django-cors-headers  │ settings.py で一括設定       │
  │                │ パッケージ            │ CORS_ALLOWED_ORIGINS等       │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Flask          │ flask-cors 拡張      │ デコレータ or アプリ全体設定  │
  │                │                      │ リソース単位での設定可能     │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Spring Boot    │ @CrossOrigin         │ アノテーションベース          │
  │                │ WebMvcConfigurer     │ グローバル or コントローラ単位│
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ ASP.NET Core   │ services.AddCors()   │ ポリシーベース                │
  │                │ app.UseCors()        │ 名前付きポリシー対応         │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Go (net/http)  │ 手動実装 or          │ rs/cors パッケージが一般的    │
  │                │ rs/cors パッケージ    │ ミドルウェアパターン         │
  ├────────────────┼──────────────────────┼──────────────────────────────┤
  │ Rust (Actix)   │ actix-cors クレート   │ ビルダーパターンで設定       │
  │                │                      │ tower-http でも利用可能      │
  └────────────────┴──────────────────────┴──────────────────────────────┘
```

### 7.2 Hono（Edge Runtime向け）

```typescript
// ============================================================
// Hono CORS設定（Cloudflare Workers / Deno Deploy 等）
// ============================================================

import { Hono } from 'hono';
import { cors } from 'hono/cors';

const app = new Hono();

// グローバルCORS設定
app.use('/api/*', cors({
  origin: [
    'https://app.example.com',
    'https://admin.example.com',
  ],
  allowHeaders: [
    'Content-Type',
    'Authorization',
    'X-Request-Id',
  ],
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  exposeHeaders: ['X-Total-Count', 'X-Request-Id'],
  credentials: true,
  maxAge: 86400,
}));

// 動的オリジン検証
app.use('/api/v2/*', cors({
  origin: (origin) => {
    // サブドメインパターンマッチ
    if (/^https:\/\/[\w-]+\.example\.com$/.test(origin)) {
      return origin;
    }
    return null; // 許可しない
  },
  credentials: true,
}));

// 公開APIはワイルドカード
app.use('/api/public/*', cors({
  origin: '*',
  maxAge: 3600,
}));

app.get('/api/data', (c) => {
  return c.json({ message: 'Hello from Edge!' });
});

export default app;
```

### 7.3 Django での CORS 設定

```python
# ============================================================
# Django CORS設定（django-cors-headers）
# ============================================================

# settings.py

INSTALLED_APPS = [
    # ...
    'corsheaders',
    # ...
]

MIDDLEWARE = [
    # CORSミドルウェアは可能な限り上位に配置
    # （他のミドルウェアがレスポンスを変更する前にCORSヘッダーを付与するため）
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    # ...
]

# --- 許可するオリジン ---
CORS_ALLOWED_ORIGINS = [
    'https://app.example.com',
    'https://admin.example.com',
]

# 正規表現による許可（サブドメイン対応）
CORS_ALLOWED_ORIGIN_REGEXES = [
    r'^https://[\w-]+\.example\.com$',
]

# --- 許可するメソッド ---
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# --- 許可するヘッダー ---
CORS_ALLOW_HEADERS = [
    'accept',
    'authorization',
    'content-type',
    'x-csrftoken',
    'x-requested-with',
]

# --- Credentials ---
CORS_ALLOW_CREDENTIALS = True

# --- プリフライトキャッシュ ---
CORS_PREFLIGHT_MAX_AGE = 86400

# --- Expose Headers ---
CORS_EXPOSE_HEADERS = [
    'x-total-count',
    'x-request-id',
]
```

### 7.4 Go（net/http + rs/cors）

```go
// ============================================================
// Go CORS設定（github.com/rs/cors）
// ============================================================

package main

import (
    "log"
    "net/http"

    "github.com/rs/cors"
)

func main() {
    mux := http.NewServeMux()
    mux.HandleFunc("/api/data", handleData)
    mux.HandleFunc("/api/users", handleUsers)

    // CORS設定
    c := cors.New(cors.Options{
        AllowedOrigins: []string{
            "https://app.example.com",
            "https://admin.example.com",
        },
        AllowedMethods: []string{
            "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS",
        },
        AllowedHeaders: []string{
            "Content-Type",
            "Authorization",
            "X-Request-Id",
        },
        ExposedHeaders: []string{
            "X-Total-Count",
            "X-Request-Id",
        },
        AllowCredentials: true,
        MaxAge:           86400,
        // デバッグモード（開発時のみ有効にする）
        Debug: false,
    })

    handler := c.Handler(mux)

    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", handler))
}

func handleData(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.Write([]byte(`{"message": "Hello from Go!"}`))
}

func handleUsers(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.Header().Set("X-Total-Count", "42")
    w.Write([]byte(`{"users": []}`))
}
```

---

## 8. セキュリティ上の考慮事項

### 8.1 アンチパターン

CORS設定における代表的なアンチパターンを理解し、セキュリティリスクを回避する。

```
アンチパターン1: Origin をそのまま反射（Origin Reflection）

  ✗ 危険な実装:

  app.use((req, res, next) => {
    // リクエストの Origin をそのまま Allow-Origin に返す
    res.setHeader(
      'Access-Control-Allow-Origin',
      req.headers.origin  // ← 検証なし！
    );
    res.setHeader('Access-Control-Allow-Credentials', 'true');
    next();
  });

  問題点:
  ┌──────────────────────────────────────────────────────┐
  │ 1. 任意のオリジンからのリクエストが許可される         │
  │ 2. Credentials: true との組み合わせで致命的            │
  │    → 攻撃者のサイトからユーザーのCookieを使って      │
  │      APIにアクセスし、レスポンスを読み取れる          │
  │ 3. Allow-Origin: * と同等だが、Credentials も許可    │
  │    されるため、実質的に * より危険                    │
  └──────────────────────────────────────────────────────┘

  ✓ 正しい実装:

  const ALLOWED_ORIGINS = new Set([
    'https://app.example.com',
    'https://admin.example.com',
  ]);

  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (origin && ALLOWED_ORIGINS.has(origin)) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Access-Control-Allow-Credentials', 'true');
    }
    next();
  });
```

```
アンチパターン2: 本番環境での Access-Control-Allow-Origin: *

  ✗ 危険な使用例:

  // 本番環境の認証付きAPIで * を使用
  app.use(cors({
    origin: '*',
    // credentials: true は * と併用できないため
    // Cookie送信は不可だが、以下の問題がある
  }));

  問題点:
  ┌──────────────────────────────────────────────────────┐
  │ 1. 意図しないオリジンからデータを取得される可能性     │
  │ 2. Bearer トークン認証の場合、トークンをヘッダーで    │
  │    送信すれば任意のオリジンからアクセス可能           │
  │ 3. 内部APIが外部から叩かれるリスク                   │
  │ 4. レート制限やIP制限をバイパスされる可能性           │
  └──────────────────────────────────────────────────────┘

  ✓ * が許容されるケース:
  ┌──────────────────────────────────────────────────────┐
  │ ・完全に公開のAPI（認証なし、機密データなし）         │
  │ ・CDNで配信する静的アセット（画像、フォント等）       │
  │ ・公開データセットのAPI（政府オープンデータ等）       │
  │ ・OEmbed等の埋め込み用エンドポイント                  │
  └──────────────────────────────────────────────────────┘
```

### 8.2 Vary ヘッダーの重要性

```
CDNキャッシュとCORSの落とし穴:

  シナリオ: CDNがCORSレスポンスをキャッシュする場合

  ① https://app-a.example.com がリクエスト
     → サーバーが Allow-Origin: https://app-a.example.com を返す
     → CDNがこのレスポンスをキャッシュ

  ② https://app-b.example.com が同じURLにリクエスト
     → CDNがキャッシュを返す
     → Allow-Origin: https://app-a.example.com（app-aのまま！）
     → app-b ではCORSエラーになる

  対策: Vary: Origin ヘッダーを設定

  res.setHeader('Vary', 'Origin');
  // → CDNはOriginヘッダーの値ごとに異なるキャッシュを保持
  // → app-a用とapp-b用で別々のキャッシュが作られる

  Varyヘッダーのフロー:
  ┌──────────────────┐
  │ CDN キャッシュ    │
  │                  │
  │ Key: URL + Origin│
  │                  │
  │ /api/data        │
  │ + Origin: app-a  │──→ Allow-Origin: app-a
  │                  │
  │ /api/data        │
  │ + Origin: app-b  │──→ Allow-Origin: app-b
  │                  │
  │ /api/data        │
  │ + Origin: (none) │──→ Allow-Origin なし
  └──────────────────┘
```

### 8.3 null オリジンの危険性

```
null オリジンの注意点:

  Origin: null が送信されるケース:
  ┌──────────────────────────────────────────────────────┐
  │ ・file:// プロトコルからのリクエスト                  │
  │ ・data: URI からのリクエスト                          │
  │ ・サンドボックス化された iframe                       │
  │ ・リダイレクトチェーン中のリクエスト                  │
  │ ・ブラウザのプライバシー保護機能による匿名化          │
  └──────────────────────────────────────────────────────┘

  ✗ 危険な実装:

  // null オリジンを許可してしまう
  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
  // allowedOrigins に 'null'（文字列）が含まれている場合、
  // data: URI や sandboxed iframe からのアクセスが可能になる

  ✓ 安全な実装:

  // null オリジンは明示的に拒否
  if (origin && origin !== 'null' && allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  }
```

---

## FAQ（よくある質問）

### Q1: CORSエラーの一般的な解決方法 — エラーメッセージから原因を特定する

```
■ 代表的なCORSエラーと解決策:

エラー1: "has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header"

  原因: サーバーがAccess-Control-Allow-Originヘッダーを返していない

  解決策:
  // Express.js
  app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', 'https://app.example.com');
    next();
  });

  // nginx
  add_header Access-Control-Allow-Origin https://app.example.com;

  // 開発環境のみ（本番環境では禁止）
  res.setHeader('Access-Control-Allow-Origin', '*');

エラー2: "The value of the 'Access-Control-Allow-Origin' header must not be '*' when credentials mode is 'include'"

  原因: credentials: 'include'（Cookie送信）を使用している場合、
        ワイルドカード（*）は使用不可

  解決策:
  // 特定のオリジンを明示
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Credentials', 'true');
  }

エラー3: "has been blocked by CORS policy: Method POST is not allowed by Access-Control-Allow-Methods"

  原因: プリフライトリクエストで許可メソッドが返されていない

  解決策:
  app.options('/api/*', (req, res) => {
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.setHeader('Access-Control-Allow-Origin', req.headers.origin);
    res.status(204).send();
  });

エラー4: "Request header field authorization is not allowed by Access-Control-Allow-Headers"

  原因: カスタムヘッダー（Authorization等）が許可されていない

  解決策:
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Request-Id');

エラー5: "CORS policy: Response to preflight request doesn't pass access control check: status is 401"

  原因: プリフライトリクエスト（OPTIONS）に認証を要求している

  解決策:
  // OPTIONSリクエストは認証不要にする
  app.options('/api/*', (req, res) => {
    // 認証チェックをスキップ
    res.setHeader('Access-Control-Allow-Origin', req.headers.origin);
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.status(204).send();
  });

デバッグ手順:
  1. ブラウザのDevToolsでNetworkタブを開く
  2. プリフライト（OPTIONS）が成功しているか確認
  3. レスポンスヘッダーを確認:
     ・Access-Control-Allow-Origin が正しいか
     ・Access-Control-Allow-Methods が含まれているか
     ・Access-Control-Allow-Headers が含まれているか
  4. curlで直接確認:
     curl -I -X OPTIONS https://api.example.com/users \
       -H "Origin: https://app.example.com" \
       -H "Access-Control-Request-Method: POST"
```

### Q2: プリフライトリクエストの発生条件 — いつOPTIONSが送られるのか

```
■ シンプルリクエスト vs プリフライトリクエスト:

シンプルリクエスト（プリフライトなし）:
  以下の条件を全て満たす場合、プリフライトは発生しない

  ① メソッドが以下のいずれか:
     GET, HEAD, POST

  ② 自動設定されるヘッダー以外に、以下のヘッダーのみを使用:
     Accept
     Accept-Language
     Content-Language
     Content-Type（下記の値のみ）
       ・application/x-www-form-urlencoded
       ・multipart/form-data
       ・text/plain

  ③ リクエストにReadableStreamを使用していない

  ④ XMLHttpRequestでevent listenerを登録していない

プリフライトが発生するケース:

  ✓ カスタムヘッダーを使用（Authorization, X-Request-Id等）
  fetch('https://api.example.com/users', {
    headers: {
      'Authorization': 'Bearer token',  // ← プリフライト発生
    },
  });

  ✓ Content-Typeがapplication/json
  fetch('https://api.example.com/users', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',  // ← プリフライト発生
    },
    body: JSON.stringify({ name: 'Taro' }),
  });

  ✓ PUT, DELETE, PATCHメソッド
  fetch('https://api.example.com/users/123', {
    method: 'DELETE',  // ← プリフライト発生
  });

プリフライトの流れ:

  1. ブラウザがOPTIONSリクエストを送信:
     OPTIONS /api/users HTTP/1.1
     Origin: https://app.example.com
     Access-Control-Request-Method: POST
     Access-Control-Request-Headers: content-type, authorization

  2. サーバーが許可情報を返す:
     HTTP/1.1 204 No Content
     Access-Control-Allow-Origin: https://app.example.com
     Access-Control-Allow-Methods: GET, POST, PUT, DELETE
     Access-Control-Allow-Headers: content-type, authorization
     Access-Control-Max-Age: 86400

  3. ブラウザが実際のPOSTリクエストを送信
     POST /api/users HTTP/1.1
     Content-Type: application/json
     Authorization: Bearer token

プリフライトのキャッシュ:
  Access-Control-Max-Age: 86400（秒）
  → 24時間はプリフライトをスキップして直接リクエスト送信
  → ブラウザによって上限あり（Chromeは2時間）
```

### Q3: credentialsモードの設定 — CookieやAuthorizationヘッダーを送る方法

```
■ credentials モードの種類:

┌─────────────┬────────────────────────────────────────┐
│ モード      │ 挙動                                    │
├─────────────┼────────────────────────────────────────┤
│ omit        │ 認証情報を送信しない（デフォルト）      │
│             │ → Cookieなし、Authorizationなし         │
│             │                                        │
│ same-origin │ 同一オリジンの場合のみ送信              │
│             │ → クロスオリジンでは送信しない          │
│             │                                        │
│ include     │ 常に送信（クロスオリジンでも）          │
│             │ → CookieとAuthorizationを送信           │
│             │ → サーバー側で特別な設定が必要          │
└─────────────┴────────────────────────────────────────┘

クライアント側の設定:

  fetch('https://api.example.com/users', {
    credentials: 'include',  // ← Cookie/Authorizationを送信
    headers: {
      'Authorization': 'Bearer token',
    },
  });

サーバー側の必須設定（credentials: 'include'の場合）:

  ✓ Access-Control-Allow-Origin に具体的なオリジンを指定（*は不可）
  res.setHeader('Access-Control-Allow-Origin', 'https://app.example.com');

  ✓ Access-Control-Allow-Credentials: true を返す
  res.setHeader('Access-Control-Allow-Credentials', 'true');

  ✗ 間違った例（エラーになる）:
  res.setHeader('Access-Control-Allow-Origin', '*');  // ← 不可
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  // エラー: "The value of the 'Access-Control-Allow-Origin' header
  //         must not be '*' when credentials mode is 'include'"

動的にオリジンを返す実装:

  // Express.js
  const allowedOrigins = [
    'https://app.example.com',
    'https://admin.example.com',
  ];

  app.use((req, res, next) => {
    const origin = req.headers.origin;
    if (allowedOrigins.includes(origin)) {
      res.setHeader('Access-Control-Allow-Origin', origin);
      res.setHeader('Access-Control-Allow-Credentials', 'true');
    }
    next();
  });

Cookieの設定（クロスオリジンで送信する場合）:

  Set-Cookie: session_id=abc123;
    SameSite=None;   ← クロスサイトで送信を許可
    Secure;          ← HTTPS必須
    HttpOnly;        ← XSS対策
    Path=/;
    Domain=.example.com

  注意:
  → SameSite=None を使用する場合、Secure属性が必須
  → HTTP接続ではSameSite=Noneが機能しない
  → Chromeは2020年からSameSite=Laxがデフォルト

セキュリティ上の注意:
  → credentials: 'include' は CSRF攻撃のリスクを高める
  → CSRF対策（CSRFトークン）を必ず実装
  → 信頼できるオリジンのみを許可
  → Cookieには SameSite=Lax or Strict を推奨（可能な限り）
```

---

## まとめ

| 概念 | キーポイント |
|------|-------------|
| **同一オリジンポリシー** | スキーム + ホスト + ポート が全て一致する場合のみ同一オリジン |
| **CORS** | クロスオリジンリクエストを安全に許可する仕組み（サーバー側で制御） |
| **シンプルリクエスト** | GET/HEAD/POST + 限定ヘッダー → プリフライトなし |
| **プリフライトリクエスト** | OPTIONS → 許可確認 → 実際のリクエスト（カスタムヘッダー、PUT/DELETE等） |
| **Credentials** | credentials: 'include' + Allow-Origin（*不可）+ Allow-Credentials: true |
| **セキュリティ** | ワイルドカード（*）は最小限に、null オリジン拒否、Vary: Originで検証 |

### キーポイント

1. **CORSはサーバー側で制御**: ブラウザはクロスオリジンリクエストを自動的にブロックし、サーバーが明示的に許可した場合のみ通す。クライアント側（JavaScript）だけでCORSエラーは解決できない。

2. **プリフライトリクエスト（OPTIONS）の理解が鍵**: カスタムヘッダー（Authorization等）やapplication/jsonを使う場合、ブラウザは実際のリクエスト前にOPTIONSリクエストを送信。サーバーは認証不要で204を返し、Access-Control-Allow-*ヘッダーで許可情報を通知する必要がある。

3. **credentials: 'include'は慎重に**: Cookie/Authorizationを送る場合、Allow-Originにワイルドカード（*）は使用不可。具体的なオリジンを動的に返す実装にし、CSRF対策（CSRFトークン）を必ず組み合わせる。

---

## 次に読むべきガイド

- [TLS/SSL](../03-security/00-tls-ssl.md) - HTTPS通信の暗号化、証明書、暗号スイートの仕組みを学ぶ
- [認証方式](../03-security/01-authentication.md) - OAuth 2.0、JWT、セッション管理などCredentials付きCORSに必要な認証の基礎を学ぶ
- [ネットワーク攻撃と対策](../03-security/02-common-attacks.md) - CORS設定の不備を悪用する攻撃パターンと防御策を学ぶ

---

## 参考文献

1. Fetch Living Standard. "CORS Protocol." WHATWG, 2024.
   https://fetch.spec.whatwg.org/#http-cors-protocol
   CORSの正式仕様。Same-Origin Policy、プリフライトリクエスト、
   Credentialsモードの詳細を定義。

2. MDN Web Docs. "Cross-Origin Resource Sharing (CORS)." Mozilla, 2024.
   https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS
   CORSの実践的ガイド。エラーメッセージの意味、設定例、
   デバッグ方法を詳細に解説。

3. web.dev. "Cross-Origin Resource Sharing (CORS)." Google, 2024.
   https://web.dev/cross-origin-resource-sharing/
   Googleのベストプラクティス。セキュアなCORS設定、
   パフォーマンス最適化（プリフライトキャッシュ）。

4. OWASP. "CORS Security Cheat Sheet." OWASP, 2024.
   https://cheatsheetseries.owasp.org/cheatsheets/CORS_Security_Cheat_Sheet.html
   セキュリティ観点でのCORS設定。一般的な脆弱性、
   安全な実装パターン、攻撃シナリオ。

5. RFC 6454. "The Web Origin Concept." IETF, 2011.
   https://www.rfc-editor.org/rfc/rfc6454
   オリジンの定義、Same-Origin Policyの正式仕様。
   セキュリティ境界の概念を定義。

6. "Understanding CORS and Dealing with CORS Errors in Angular." Bitovi, 2023.
   https://www.bitovi.com/blog/understanding-cors-and-dealing-with-cors-errors-in-angular
   実装例とトラブルシューティング。Angular/React/Vue.jsでの
   CORS対応パターン。

