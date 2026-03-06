# REST API設計

> RESTはWeb APIの設計原則。リソース指向のURI設計、適切なHTTPメソッドの使用、ステータスコードの活用で、直感的で保守性の高いAPIを設計する。Roy Fieldingが2000年の博士論文で提唱したアーキテクチャスタイルであり、Webの成功を支えた根幹技術を体系化したものである。

## この章で学ぶこと

- [ ] RESTの6原則とRichardson成熟度モデルを理解する
- [ ] リソース指向のURI設計を把握し、適切なHTTPメソッドを選択できる
- [ ] 実践的なAPI設計パターン（ページネーション、フィルタリング、バージョニング）を学ぶ
- [ ] HATEOASの概念と適用場面を理解する
- [ ] OpenAPI仕様でAPIを文書化する方法を習得する
- [ ] Express/FastAPIでのREST API実装パターンを把握する

---

## 1. RESTの原則

### 1.1 RESTとは何か

REST（Representational State Transfer）はRoy Fieldingが2000年の博士論文で提唱したアーキテクチャスタイルである。HTTPプロトコルの主要な設計者の一人であるFieldingが、Webがなぜ成功したのかを分析し、その設計原則を体系化したものがRESTである。

RESTはプロトコルやフレームワークではなく、あくまで「制約の集合」として定義される。これらの制約を満たすシステムを「RESTful」と呼ぶ。

```
REST（Representational State Transfer）:
  → Roy Fieldingの2000年の博士論文で提唱
  → Web の既存技術（HTTP, URI）を活用したアーキテクチャスタイル
  → プロトコルではなく「制約の集合」

6つの制約:
  +-------------------------------------------------------+
  |              REST アーキテクチャ制約                      |
  +-------------------------------------------------------+
  |                                                       |
  |  ① クライアント・サーバー分離                            |
  |     → UIとデータ処理を分離                              |
  |     → 独立して進化可能                                  |
  |     → 関心の分離（Separation of Concerns）              |
  |                                                       |
  |  ② ステートレス                                        |
  |     → 各リクエストが完結                                |
  |     → サーバーはセッション状態を持たない                   |
  |     → スケーラビリティが向上                             |
  |     → リクエスト単位でロードバランシング可能               |
  |                                                       |
  |  ③ キャッシュ可能                                      |
  |     → レスポンスにキャッシュ可否を明示                    |
  |     → Cache-Control, ETag, Last-Modified               |
  |     → ネットワーク効率とレイテンシの改善                   |
  |                                                       |
  |  ④ 統一インターフェース                                 |
  |     → リソースの識別（URI）                             |
  |     → 表現を通じたリソース操作（JSON/XML）               |
  |     → 自己記述メッセージ（Content-Type等）               |
  |     → HATEOAS（ハイパーメディア駆動）                    |
  |                                                       |
  |  ⑤ 階層化システム                                      |
  |     → ロードバランサー、プロキシ、ゲートウェイを挟める     |
  |     → クライアントは中間層を意識しない                    |
  |     → セキュリティポリシーの集中管理が可能                 |
  |                                                       |
  |  ⑥ コードオンデマンド（任意）                            |
  |     → サーバーからクライアントに実行コードを送信           |
  |     → JavaScript等                                     |
  |     → 唯一のオプショナルな制約                           |
  +-------------------------------------------------------+
```

### 1.2 Richardson成熟度モデル

Leonard Richardsonが提唱した成熟度モデルは、APIがどの程度RESTfulであるかを4段階で評価する。

```
Richardson Maturity Model（REST成熟度モデル）:

  Level 3 ──── HATEOAS（ハイパーメディア制御）          ← 完全なREST
     ▲         レスポンスに次の操作リンクを含む
     │
  Level 2 ──── HTTPメソッド + ステータスコードの活用     ← 大半のAPIはここ
     ▲         GET/POST/PUT/DELETE + 200/201/404 等
     │
  Level 1 ──── リソースの導入
     ▲         個別のURIでリソースを識別
     │          /users/123, /orders/456
     │
  Level 0 ──── 単一エンドポイント（POX: Plain Old XML）
               全操作を1つのURIに POST
               SOAP的なアプローチ

  ┌─────────┬──────────────────────────┬───────────────────┐
  │ Level   │ 特徴                      │ 例                │
  ├─────────┼──────────────────────────┼───────────────────┤
  │ Level 0 │ 1つのエンドポイント        │ POST /api         │
  │         │ すべてPOST                │ body: {action:    │
  │         │                          │  "getUser"}       │
  ├─────────┼──────────────────────────┼───────────────────┤
  │ Level 1 │ リソースごとにURI          │ POST /api/users   │
  │         │ まだPOSTのみ              │ POST /api/orders  │
  ├─────────┼──────────────────────────┼───────────────────┤
  │ Level 2 │ HTTPメソッドを活用         │ GET /api/users    │
  │         │ ステータスコードも適切      │ POST /api/users   │
  │         │                          │ → 201 Created     │
  ├─────────┼──────────────────────────┼───────────────────┤
  │ Level 3 │ HATEOASを導入             │ レスポンスにリンク  │
  │         │ 自己発見可能なAPI          │ を含む             │
  └─────────┴──────────────────────────┴───────────────────┘

  現実的には Level 2 を達成していれば実用上十分。
  Level 3 は理想だが、クライアント側の対応コストが高い。
```

### 1.3 RESTと他のAPIスタイルの比較

```
┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ 観点     │ REST         │ GraphQL      │ gRPC         │ SOAP         │
├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ プロトコル│ HTTP         │ HTTP         │ HTTP/2       │ HTTP/SMTP等  │
│ データ形式│ JSON（主流） │ JSON         │ Protocol     │ XML          │
│          │              │              │ Buffers      │              │
│ 型定義   │ OpenAPI      │ Schema       │ .proto       │ WSDL         │
│ 過剰取得 │ 起こりうる   │ 起こらない   │ 起こらない   │ 起こりうる   │
│ 過少取得 │ 起こりうる   │ 起こらない   │ 起こらない   │ 起こりうる   │
│ リアルタイム│ WebSocket    │ Subscription │ Streaming    │ なし         │
│ 学習コスト│ 低い         │ 中程度       │ 高い         │ 高い         │
│ ツール   │ 豊富         │ 増加中       │ 限定的       │ 成熟         │
│ キャッシュ│ HTTP標準     │ 独自実装必要 │ 独自実装必要 │ 困難         │
│ 適用場面 │ 公開API      │ モバイル     │ マイクロ     │ エンタープ   │
│          │ Web全般      │ 複雑なUI     │ サービス間   │ ライズ       │
└──────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

RESTの最大の強みは「Webの標準技術をそのまま活用する」点にある。HTTPキャッシュ、CDN、プロキシ、ロードバランサーといった既存のWebインフラがそのまま機能する。公開APIや外部開発者向けのAPIでは、RESTが最も広く採用されている。

---

## 2. URI設計

### 2.1 リソース指向のURI

REST APIの核心は「リソース」という概念にある。すべてのデータをリソースとして捉え、URIで一意に識別する。

```
リソース指向のURI設計:

  ┌─────────────────────────────────────────────┐
  │           リソース階層の設計例                  │
  ├─────────────────────────────────────────────┤
  │                                             │
  │  /api/v1                                    │
  │    ├── /users                   コレクション  │
  │    │   ├── /users/{id}          個別リソース  │
  │    │   ├── /users/{id}/profile  サブリソース  │
  │    │   ├── /users/{id}/orders   関連コレクション│
  │    │   └── /users/{id}/orders/{oid}          │
  │    ├── /products                             │
  │    │   ├── /products/{id}                    │
  │    │   ├── /products/{id}/reviews            │
  │    │   └── /products/{id}/variants           │
  │    ├── /orders                               │
  │    │   ├── /orders/{id}                      │
  │    │   ├── /orders/{id}/items                │
  │    │   └── /orders/{id}/payments             │
  │    └── /categories                           │
  │        ├── /categories/{id}                  │
  │        └── /categories/{id}/products         │
  │                                             │
  └─────────────────────────────────────────────┘

  リソースの種類:
  ┌─────────────┬─────────────────────┬──────────────────┐
  │ 種類        │ 説明                 │ 例               │
  ├─────────────┼─────────────────────┼──────────────────┤
  │ コレクション │ リソースの集合        │ /users           │
  │ ドキュメント │ 個別のリソース        │ /users/123       │
  │ サブコレクション│ 親に属するコレクション │ /users/123/orders│
  │ コントローラ │ 手続き的操作（例外）   │ /users/123/ban   │
  └─────────────┴─────────────────────┴──────────────────┘
```

### 2.2 HTTPメソッドとCRUD操作のマッピング

```
HTTPメソッドとリソース操作の対応:

  ✓ 良い設計:
  GET    /api/v1/users              — ユーザー一覧取得
  GET    /api/v1/users/123          — ユーザー詳細取得
  POST   /api/v1/users              — ユーザー作成
  PUT    /api/v1/users/123          — ユーザー全体更新
  PATCH  /api/v1/users/123          — ユーザー部分更新
  DELETE /api/v1/users/123          — ユーザー削除
  GET    /api/v1/users/123/orders   — ユーザーの注文一覧
  GET    /api/v1/users/123/orders/456 — 注文詳細

  ✗ 悪い設計:
  GET    /api/getUsers              — 動詞を使わない
  POST   /api/createUser            — メソッドに役割を持たせる
  GET    /api/user/delete/123       — GETで副作用を起こさない
  GET    /api/Users                 — 大文字を使わない
  POST   /api/users/123/update      — URIに動詞を入れない

  メソッドの安全性と冪等性:
  ┌─────────┬────────┬────────┬───────────────────────┐
  │ メソッド │ 安全性 │ 冪等性 │ 用途                   │
  ├─────────┼────────┼────────┼───────────────────────┤
  │ GET     │ ○     │ ○     │ リソースの取得          │
  │ HEAD    │ ○     │ ○     │ ヘッダーのみ取得        │
  │ OPTIONS │ ○     │ ○     │ 対応メソッドの確認      │
  │ POST    │ ×     │ ×     │ リソースの作成          │
  │ PUT     │ ×     │ ○     │ リソースの全体置換      │
  │ PATCH   │ ×     │ △     │ リソースの部分更新      │
  │ DELETE  │ ×     │ ○     │ リソースの削除          │
  └─────────┴────────┴────────┴───────────────────────┘

  安全性: リクエストがサーバーの状態を変更しない
  冪等性: 同じリクエストを何度実行しても結果が同じ
  △: 実装依存（冪等に実装すべき）
```

### 2.3 命名規則

```
URI命名規則:

  → 名詞・複数形: /users, /orders, /products
  → ケバブケース: /user-profiles（スネークケースも許容）
  → 小文字のみ: /users（/Users ではない）
  → 末尾スラッシュなし: /users（/users/ ではない）
  → ファイル拡張子なし: /users（/users.json ではない）

  ネスト vs フラット:
    ネスト:  GET /users/123/orders/456
    フラット: GET /orders/456

    → ネストは2階層まで（/resource/{id}/sub-resource）
    → 3階層以上はフラットにする
    → リソースに一意のIDがあればフラットが良い

  特殊な操作（アクション）:
    → 標準CRUDに収まらない操作はコントローラリソースとして設計
    → POST /api/v1/users/123/activate  （アカウント有効化）
    → POST /api/v1/orders/456/cancel    （注文キャンセル）
    → POST /api/v1/reports/generate      （レポート生成）
    → これらは例外的にPOST + 動詞を使ってよい
```

---

## 3. クエリパラメータ

### 3.1 ページネーション

```
一覧取得のページネーション:

  ① オフセットベース:
  GET /api/users?page=2&per_page=20
  GET /api/users?offset=20&limit=20

  ② カーソルベース（推奨）:
  GET /api/users?cursor=eyJpZCI6MTIzfQ==&limit=20
  → レスポンスに次のカーソルを含む

  ③ キーセットベース:
  GET /api/users?after_id=123&limit=20
  → 特定カラムの値を基準にする

ページネーション方式の詳細比較:
  ┌────────────┬─────────────────┬─────────────────────────┬───────────┐
  │ 方式       │ メリット         │ デメリット               │ 推奨場面  │
  ├────────────┼─────────────────┼─────────────────────────┼───────────┤
  │ offset     │ 実装が簡単       │ 大量データで性能劣化     │ 管理画面  │
  │            │ ページ番号指定可 │ データ追加で重複/欠損    │ 少量データ│
  ├────────────┼─────────────────┼─────────────────────────┼───────────┤
  │ cursor     │ 一貫した結果     │ 任意ページに飛べない     │ 無限      │
  │            │ 高速で安定       │ カーソル値が不透明       │ スクロール│
  ├────────────┼─────────────────┼─────────────────────────┼───────────┤
  │ keyset     │ 最も高速         │ 実装がやや複雑           │ 大規模    │
  │            │ インデックス活用 │ ソートキーが必要         │ データセット│
  └────────────┴─────────────────┴─────────────────────────┴───────────┘
```

### 3.2 ソート・フィルタリング・検索

```
  ソート:
  GET /api/users?sort=created_at&order=desc
  GET /api/users?sort=-created_at               （-プレフィックスは降順）
  GET /api/users?sort=last_name,-created_at      （複数キー）

  フィルタリング:
  GET /api/users?status=active&role=admin
  GET /api/users?created_after=2024-01-01
  GET /api/users?age[gte]=18&age[lte]=65         （範囲指定）
  GET /api/users?status[in]=active,pending        （複数値）

  フィールド選択（Sparse Fieldsets）:
  GET /api/users?fields=id,name,email
  GET /api/users?fields[users]=id,name&fields[company]=name
  → レスポンスサイズ削減によるパフォーマンス向上

  検索:
  GET /api/users?q=taro
  GET /api/users/search?q=taro&fields=name,email

  組み合わせ例:
  GET /api/users?status=active&sort=-created_at&page=1&per_page=20&fields=id,name
```

---

## 4. レスポンス設計

### 4.1 成功レスポンス

```
一覧（GET /api/users → 200 OK）:
{
  "data": [
    {
      "id": "1",
      "type": "user",
      "attributes": {
        "name": "Taro Yamada",
        "email": "taro@example.com",
        "created_at": "2024-01-15T10:30:00Z"
      }
    },
    {
      "id": "2",
      "type": "user",
      "attributes": {
        "name": "Hanako Suzuki",
        "email": "hanako@example.com",
        "created_at": "2024-02-20T14:00:00Z"
      }
    }
  ],
  "meta": {
    "total": 150,
    "page": 1,
    "per_page": 20,
    "total_pages": 8
  },
  "links": {
    "self": "/api/v1/users?page=1",
    "next": "/api/v1/users?page=2",
    "last": "/api/v1/users?page=8"
  }
}

詳細（GET /api/users/1 → 200 OK）:
{
  "data": {
    "id": "1",
    "type": "user",
    "attributes": {
      "name": "Taro Yamada",
      "email": "taro@example.com",
      "role": "admin",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-06-01T09:15:00Z"
    },
    "relationships": {
      "orders": {
        "links": {
          "related": "/api/v1/users/1/orders"
        }
      },
      "profile": {
        "links": {
          "related": "/api/v1/users/1/profile"
        }
      }
    }
  }
}

作成成功（POST /api/users → 201 Created）:
HTTP/1.1 201 Created
Location: /api/v1/users/3
Content-Type: application/json

{
  "data": {
    "id": "3",
    "type": "user",
    "attributes": {
      "name": "Jiro Tanaka",
      "email": "jiro@example.com",
      "created_at": "2024-07-01T12:00:00Z"
    }
  }
}

削除成功（DELETE /api/users/3 → 204 No Content）:
HTTP/1.1 204 No Content
（ボディなし）
```

### 4.2 エラーレスポンス

```
RFC 7807 Problem Details 形式:

バリデーションエラー（422 Unprocessable Entity）:
{
  "type": "https://api.example.com/errors/validation",
  "title": "Validation Error",
  "status": 422,
  "detail": "The request body contains invalid fields.",
  "instance": "/api/v1/users",
  "errors": [
    {
      "field": "email",
      "code": "invalid_format",
      "message": "Invalid email format"
    },
    {
      "field": "age",
      "code": "out_of_range",
      "message": "Must be 18 or older"
    }
  ]
}

認証エラー（401 Unauthorized）:
{
  "type": "https://api.example.com/errors/unauthorized",
  "title": "Unauthorized",
  "status": 401,
  "detail": "The access token is expired or invalid."
}

権限エラー（403 Forbidden）:
{
  "type": "https://api.example.com/errors/forbidden",
  "title": "Forbidden",
  "status": 403,
  "detail": "You do not have permission to access this resource."
}

リソース未検出（404 Not Found）:
{
  "type": "https://api.example.com/errors/not-found",
  "title": "Not Found",
  "status": 404,
  "detail": "User with ID '999' was not found."
}

競合エラー（409 Conflict）:
{
  "type": "https://api.example.com/errors/conflict",
  "title": "Conflict",
  "status": 409,
  "detail": "A user with this email already exists."
}

主要ステータスコードの使い分け:
  ┌──────┬─────────────────────────┬──────────────────────────┐
  │ コード│ 名前                     │ 使用場面                  │
  ├──────┼─────────────────────────┼──────────────────────────┤
  │ 200  │ OK                      │ 取得・更新成功             │
  │ 201  │ Created                 │ 作成成功                   │
  │ 204  │ No Content              │ 削除成功（ボディなし）      │
  │ 301  │ Moved Permanently       │ リソースの恒久的移動       │
  │ 304  │ Not Modified            │ キャッシュ有効             │
  │ 400  │ Bad Request             │ リクエスト構文エラー        │
  │ 401  │ Unauthorized            │ 認証が必要                 │
  │ 403  │ Forbidden               │ 権限不足                   │
  │ 404  │ Not Found               │ リソースが存在しない        │
  │ 405  │ Method Not Allowed      │ 許可されていないメソッド    │
  │ 409  │ Conflict                │ リソースの競合             │
  │ 422  │ Unprocessable Entity    │ バリデーションエラー        │
  │ 429  │ Too Many Requests       │ レート制限超過             │
  │ 500  │ Internal Server Error   │ サーバー内部エラー         │
  │ 503  │ Service Unavailable     │ サービス一時停止           │
  └──────┴─────────────────────────┴──────────────────────────┘
```

---

## 5. バージョニング

### 5.1 バージョニング戦略

```
APIバージョニング戦略:

  ① URIバージョニング（最も一般的）:
     GET /api/v1/users
     GET /api/v2/users
     → メリット: わかりやすい、キャッシュしやすい、ルーティングが簡単
     → デメリット: URIが変わるためリンクが壊れる
     → 採用: GitHub, Twitter, Stripe

  ② ヘッダーバージョニング:
     GET /api/users
     Accept: application/vnd.example.v2+json
     → メリット: URIがクリーン、コンテントネゴシエーション
     → デメリット: テストしにくい、ブラウザで直接確認できない
     → 採用: GitHub（併用）

  ③ クエリパラメータバージョニング:
     GET /api/users?version=2
     → メリット: 実装が簡単、切り替えが容易
     → デメリット: キャッシュキーが増える、オプショナルに見える
     → 採用: Google, Amazon（一部）

  ④ カスタムヘッダー:
     GET /api/users
     X-API-Version: 2
     → メリット: Acceptヘッダーより明確
     → デメリット: 標準的ではない

  推奨: URIバージョニング（/api/v1/）が最も広く理解されている
```

### 5.2 バージョンアップの判断基準

```
破壊的変更（メジャーバージョンアップが必要）:
  → フィールドの削除
  → フィールドの型変更（string → number など）
  → 必須パラメータの追加
  → レスポンス構造の変更
  → エンドポイントの削除やパス変更
  → ステータスコードの意味変更

非破壊的変更（バージョンアップ不要）:
  → オプショナルなフィールドの追加
  → 新しいエンドポイントの追加
  → オプショナルなクエリパラメータの追加
  → エラーメッセージの文言変更
  → パフォーマンス改善

バージョン管理のベストプラクティス:
  → 旧バージョンは最低12ヶ月サポート
  → 非推奨化（Deprecation）をヘッダーで通知:
     Deprecation: true
     Sunset: Sat, 01 Jan 2026 00:00:00 GMT
     Link: </api/v2/users>; rel="successor-version"
  → 新バージョンリリース時にマイグレーションガイドを提供
  → APIの変更履歴（Changelog）を公開する
```

---

## 6. HATEOAS

### 6.1 HATEOASとは

HATEOAS（Hypermedia As The Engine Of Application State）はREST制約のうち「統一インターフェース」に含まれる概念である。APIレスポンスに、クライアントが次に取りうるアクションへのリンクを含めることで、APIを「自己発見可能」にする。

```
HATEOASの概念図:

  従来のAPI（リンクなし）:
  ┌─────────────┐                    ┌─────────────┐
  │  クライアント │───GET /users/1──→│   サーバー    │
  │             │←── { id: 1,    ───│             │
  │  URIを事前に│     name: "Taro"} │             │
  │  知っている  │                    │             │
  │  必要がある  │───GET /users/1/ ─→│             │
  │             │   orders          │             │
  └─────────────┘                    └─────────────┘

  HATEOASを適用したAPI（リンクあり）:
  ┌─────────────┐                    ┌─────────────┐
  │  クライアント │───GET /users/1──→│   サーバー    │
  │             │←── { id: 1,    ───│             │
  │  レスポンス  │     name: "Taro", │             │
  │  のリンクを  │     _links: {     │             │
  │  辿るだけ    │       orders:     │             │
  │             │       "/users/1/  │             │
  │             │        orders"}}  │             │
  └─────────────┘                    └─────────────┘
```

### 6.2 HATEOASレスポンスの例

```json
{
  "data": {
    "id": "order-456",
    "status": "pending",
    "total": 5800,
    "currency": "JPY",
    "created_at": "2024-07-01T12:00:00Z"
  },
  "_links": {
    "self": {
      "href": "/api/v1/orders/456",
      "method": "GET"
    },
    "cancel": {
      "href": "/api/v1/orders/456/cancel",
      "method": "POST",
      "title": "Cancel this order"
    },
    "payment": {
      "href": "/api/v1/orders/456/payments",
      "method": "POST",
      "title": "Submit payment"
    },
    "items": {
      "href": "/api/v1/orders/456/items",
      "method": "GET",
      "title": "View order items"
    },
    "customer": {
      "href": "/api/v1/users/123",
      "method": "GET",
      "title": "View customer details"
    }
  }
}
```

注文が「shipped」に変わると、`cancel`リンクは消え、代わりに`track`リンクが出現する。これにより、クライアントは状態に応じて利用可能な操作を動的に知ることができる。

```json
{
  "data": {
    "id": "order-456",
    "status": "shipped",
    "total": 5800,
    "tracking_number": "JP123456789"
  },
  "_links": {
    "self": {
      "href": "/api/v1/orders/456"
    },
    "track": {
      "href": "/api/v1/orders/456/tracking",
      "method": "GET",
      "title": "Track shipment"
    },
    "return": {
      "href": "/api/v1/orders/456/returns",
      "method": "POST",
      "title": "Request return"
    }
  }
}
```

---

## 7. 認証とレート制限

### 7.1 認証方式

```
主要な認証方式:

  ① Bearer Token（JWT）:
  Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
  → ステートレス、スケーラブル
  → トークンの失効管理が課題

  ② API Key:
  X-API-Key: your-api-key-here
  （またはクエリパラメータ: ?api_key=xxx）
  → シンプル、サーバー間通信向き
  → キーの漏洩リスク

  ③ OAuth 2.0:
  → 第三者アプリへの権限委譲
  → Authorization Code Flow が推奨
  → スコープで権限を細分化

  ④ Basic認証:
  Authorization: Basic base64(username:password)
  → 開発・テスト用途のみ
  → 本番ではHTTPS必須

  → 公開APIにはOAuth 2.0 + API Key の組み合わせが一般的
  → 内部APIにはJWT Bearer Tokenが効率的
```

### 7.2 レート制限

```
レスポンスヘッダーで制限情報を通知:

  X-RateLimit-Limit: 100       — 制限数（/分 等）
  X-RateLimit-Remaining: 42    — 残り回数
  X-RateLimit-Reset: 1640000000 — リセット時刻（Unix秒）

  制限超過時:
  HTTP/1.1 429 Too Many Requests
  Retry-After: 60
  Content-Type: application/json

  {
    "type": "https://api.example.com/errors/rate-limit",
    "title": "Rate Limit Exceeded",
    "status": 429,
    "detail": "You have exceeded 100 requests per minute.",
    "retry_after": 60
  }

  一般的な制限例:
  ┌────────────────────┬──────────────────┐
  │ ティア             │ レート制限         │
  ├────────────────────┼──────────────────┤
  │ 未認証             │ 20 req/分         │
  │ 認証済み（無料）    │ 100 req/分        │
  │ 認証済み（有料）    │ 1,000 req/分      │
  │ エンタープライズ    │ 10,000 req/分     │
  │ 書き込み操作       │ 読み取りの1/5     │
  └────────────────────┴──────────────────┘

  レート制限の実装アルゴリズム:
  → Token Bucket: バースト対応、最も一般的
  → Sliding Window: 精度が高い、計算コストがやや高い
  → Fixed Window: 実装が最も簡単、境界で2倍のリクエストが通る問題
```

---

## 8. 実装例

### 8.1 Express.js（Node.js）によるREST API実装

```javascript
// app.js - Express REST API 基本構成
const express = require('express');
const app = express();

app.use(express.json());

// ─── インメモリデータストア（デモ用） ───
let users = [
  { id: '1', name: 'Taro Yamada', email: 'taro@example.com', role: 'admin',
    created_at: '2024-01-15T10:30:00Z' },
  { id: '2', name: 'Hanako Suzuki', email: 'hanako@example.com', role: 'user',
    created_at: '2024-02-20T14:00:00Z' },
];
let nextId = 3;

// ─── ミドルウェア: レート制限ヘッダー ───
const rateLimitMiddleware = (req, res, next) => {
  res.set({
    'X-RateLimit-Limit': '100',
    'X-RateLimit-Remaining': '99',
    'X-RateLimit-Reset': String(Math.floor(Date.now() / 1000) + 60),
  });
  next();
};
app.use('/api', rateLimitMiddleware);

// ─── ユーザー一覧取得 ───
// GET /api/v1/users?page=1&per_page=20&sort=-created_at&status=active
app.get('/api/v1/users', (req, res) => {
  const page = parseInt(req.query.page) || 1;
  const perPage = Math.min(parseInt(req.query.per_page) || 20, 100);
  const offset = (page - 1) * perPage;

  // フィルタリング
  let filtered = [...users];
  if (req.query.role) {
    filtered = filtered.filter(u => u.role === req.query.role);
  }

  // ソート
  if (req.query.sort) {
    const desc = req.query.sort.startsWith('-');
    const field = desc ? req.query.sort.slice(1) : req.query.sort;
    filtered.sort((a, b) => {
      if (a[field] < b[field]) return desc ? 1 : -1;
      if (a[field] > b[field]) return desc ? -1 : 1;
      return 0;
    });
  }

  const total = filtered.length;
  const paged = filtered.slice(offset, offset + perPage);

  res.json({
    data: paged,
    meta: {
      total,
      page,
      per_page: perPage,
      total_pages: Math.ceil(total / perPage),
    },
    links: {
      self: `/api/v1/users?page=${page}&per_page=${perPage}`,
      ...(page > 1 && {
        prev: `/api/v1/users?page=${page - 1}&per_page=${perPage}`,
      }),
      ...(offset + perPage < total && {
        next: `/api/v1/users?page=${page + 1}&per_page=${perPage}`,
      }),
    },
  });
});

// ─── ユーザー詳細取得 ───
// GET /api/v1/users/:id
app.get('/api/v1/users/:id', (req, res) => {
  const user = users.find(u => u.id === req.params.id);
  if (!user) {
    return res.status(404).json({
      type: 'https://api.example.com/errors/not-found',
      title: 'Not Found',
      status: 404,
      detail: `User with ID '${req.params.id}' was not found.`,
    });
  }

  res.json({
    data: user,
    _links: {
      self: { href: `/api/v1/users/${user.id}` },
      orders: { href: `/api/v1/users/${user.id}/orders` },
      update: { href: `/api/v1/users/${user.id}`, method: 'PUT' },
      delete: { href: `/api/v1/users/${user.id}`, method: 'DELETE' },
    },
  });
});

// ─── ユーザー作成 ───
// POST /api/v1/users
app.post('/api/v1/users', (req, res) => {
  const { name, email, role } = req.body;

  // バリデーション
  const errors = [];
  if (!name) errors.push({ field: 'name', message: 'Name is required' });
  if (!email) errors.push({ field: 'email', message: 'Email is required' });
  if (email && users.some(u => u.email === email)) {
    errors.push({ field: 'email', message: 'Email already exists' });
  }

  if (errors.length > 0) {
    return res.status(422).json({
      type: 'https://api.example.com/errors/validation',
      title: 'Validation Error',
      status: 422,
      detail: 'The request body contains invalid fields.',
      errors,
    });
  }

  const newUser = {
    id: String(nextId++),
    name,
    email,
    role: role || 'user',
    created_at: new Date().toISOString(),
  };
  users.push(newUser);

  res.status(201)
    .location(`/api/v1/users/${newUser.id}`)
    .json({ data: newUser });
});

// ─── ユーザー更新（全体置換） ───
// PUT /api/v1/users/:id
app.put('/api/v1/users/:id', (req, res) => {
  const index = users.findIndex(u => u.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({
      type: 'https://api.example.com/errors/not-found',
      title: 'Not Found',
      status: 404,
      detail: `User with ID '${req.params.id}' was not found.`,
    });
  }

  const { name, email, role } = req.body;
  users[index] = {
    ...users[index],
    name,
    email,
    role,
    updated_at: new Date().toISOString(),
  };

  res.json({ data: users[index] });
});

// ─── ユーザー部分更新 ───
// PATCH /api/v1/users/:id
app.patch('/api/v1/users/:id', (req, res) => {
  const index = users.findIndex(u => u.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({
      type: 'https://api.example.com/errors/not-found',
      title: 'Not Found',
      status: 404,
      detail: `User with ID '${req.params.id}' was not found.`,
    });
  }

  users[index] = {
    ...users[index],
    ...req.body,
    id: users[index].id, // IDは変更不可
    updated_at: new Date().toISOString(),
  };

  res.json({ data: users[index] });
});

// ─── ユーザー削除 ───
// DELETE /api/v1/users/:id
app.delete('/api/v1/users/:id', (req, res) => {
  const index = users.findIndex(u => u.id === req.params.id);
  if (index === -1) {
    return res.status(404).json({
      type: 'https://api.example.com/errors/not-found',
      title: 'Not Found',
      status: 404,
      detail: `User with ID '${req.params.id}' was not found.`,
    });
  }

  users.splice(index, 1);
  res.status(204).send();
});

app.listen(3000, () => {
  console.log('REST API server running on port 3000');
});
```

### 8.2 FastAPI（Python）によるREST API実装

```python
# main.py - FastAPI REST API 基本構成
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from uuid import uuid4

app = FastAPI(
    title="User Management API",
    version="1.0.0",
    description="RESTful API for user management",
)

# ─── モデル定義 ───
class UserCreate(BaseModel):
    name: str
    email: EmailStr
    role: str = "user"

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    role: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    role: str
    created_at: str
    updated_at: Optional[str] = None

class PaginationMeta(BaseModel):
    total: int
    page: int
    per_page: int
    total_pages: int

class UserListResponse(BaseModel):
    data: list[UserResponse]
    meta: PaginationMeta

class ErrorDetail(BaseModel):
    field: str
    message: str

class ErrorResponse(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    errors: Optional[list[ErrorDetail]] = None

# ─── インメモリストア ───
users_db: dict[str, dict] = {}

# ─── エンドポイント ───
@app.get("/api/v1/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    role: Optional[str] = None,
    sort: Optional[str] = None,
):
    """ユーザー一覧取得: ページネーション、フィルタ、ソート対応"""
    all_users = list(users_db.values())

    # フィルタリング
    if role:
        all_users = [u for u in all_users if u["role"] == role]

    # ソート
    if sort:
        desc = sort.startswith("-")
        key = sort.lstrip("-")
        all_users.sort(key=lambda u: u.get(key, ""), reverse=desc)

    total = len(all_users)
    offset = (page - 1) * per_page
    paged = all_users[offset:offset + per_page]

    return UserListResponse(
        data=[UserResponse(**u) for u in paged],
        meta=PaginationMeta(
            total=total,
            page=page,
            per_page=per_page,
            total_pages=(total + per_page - 1) // per_page or 1,
        ),
    )

@app.get("/api/v1/users/{user_id}", response_model=dict)
async def get_user(user_id: str):
    """ユーザー詳細取得"""
    if user_id not in users_db:
        raise HTTPException(
            status_code=404,
            detail={
                "type": "https://api.example.com/errors/not-found",
                "title": "Not Found",
                "status": 404,
                "detail": f"User with ID '{user_id}' was not found.",
            },
        )
    return {
        "data": users_db[user_id],
        "_links": {
            "self": {"href": f"/api/v1/users/{user_id}"},
            "orders": {"href": f"/api/v1/users/{user_id}/orders"},
        },
    }

@app.post("/api/v1/users", status_code=201)
async def create_user(user: UserCreate, response: Response):
    """ユーザー作成"""
    # メール重複チェック
    for existing in users_db.values():
        if existing["email"] == user.email:
            raise HTTPException(
                status_code=409,
                detail={
                    "type": "https://api.example.com/errors/conflict",
                    "title": "Conflict",
                    "status": 409,
                    "detail": "A user with this email already exists.",
                },
            )

    user_id = str(uuid4())[:8]
    new_user = {
        "id": user_id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    users_db[user_id] = new_user
    response.headers["Location"] = f"/api/v1/users/{user_id}"
    return {"data": new_user}

@app.patch("/api/v1/users/{user_id}")
async def update_user(user_id: str, updates: UserUpdate):
    """ユーザー部分更新"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = updates.model_dump(exclude_unset=True)
    users_db[user_id].update(update_data)
    users_db[user_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    return {"data": users_db[user_id]}

@app.delete("/api/v1/users/{user_id}", status_code=204)
async def delete_user(user_id: str):
    """ユーザー削除"""
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[user_id]
```

### 8.3 curlによるAPI操作例

```bash
# ─── ユーザー一覧取得 ───
curl -s -X GET "http://localhost:3000/api/v1/users?page=1&per_page=10" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." | jq .

# ─── ユーザー詳細取得 ───
curl -s -X GET "http://localhost:3000/api/v1/users/1" \
  -H "Accept: application/json" | jq .

# ─── ユーザー作成 ───
curl -s -X POST "http://localhost:3000/api/v1/users" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{
    "name": "Saburo Kato",
    "email": "saburo@example.com",
    "role": "user"
  }' | jq .
# → HTTP 201 Created
# → Location: /api/v1/users/3

# ─── ユーザー部分更新 ───
curl -s -X PATCH "http://localhost:3000/api/v1/users/1" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -d '{
    "role": "moderator"
  }' | jq .
# → HTTP 200 OK

# ─── ユーザー削除 ───
curl -s -X DELETE "http://localhost:3000/api/v1/users/3" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  -v
# → HTTP 204 No Content

# ─── フィルタリング + ソート + ページネーション ───
curl -s -X GET \
  "http://localhost:3000/api/v1/users?role=admin&sort=-created_at&page=1&per_page=5" \
  -H "Accept: application/json" | jq .

# ─── レート制限ヘッダーの確認 ───
curl -s -D - "http://localhost:3000/api/v1/users" \
  -H "Accept: application/json" -o /dev/null 2>&1 | grep -i "x-ratelimit"
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 99
# X-RateLimit-Reset: 1720000060
```

### 8.4 OpenAPI（Swagger）仕様定義例

```yaml
# openapi.yaml - OpenAPI 3.0 仕様書
openapi: "3.0.3"
info:
  title: User Management API
  description: RESTful API for user CRUD operations
  version: "1.0.0"
  contact:
    name: API Support
    email: support@example.com
  license:
    name: MIT

servers:
  - url: https://api.example.com/api/v1
    description: Production
  - url: http://localhost:3000/api/v1
    description: Development

paths:
  /users:
    get:
      summary: ユーザー一覧取得
      operationId: listUsers
      tags:
        - Users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
            minimum: 1
        - name: per_page
          in: query
          schema:
            type: integer
            default: 20
            minimum: 1
            maximum: 100
        - name: role
          in: query
          schema:
            type: string
            enum: [admin, user, moderator]
        - name: sort
          in: query
          description: "ソートキー（-で降順）"
          schema:
            type: string
            example: "-created_at"
      responses:
        "200":
          description: ユーザー一覧
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/UserListResponse"
        "401":
          $ref: "#/components/responses/Unauthorized"

    post:
      summary: ユーザー作成
      operationId: createUser
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/UserCreate"
      responses:
        "201":
          description: 作成成功
          headers:
            Location:
              schema:
                type: string
              description: 作成されたリソースのURI
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/UserSingleResponse"
        "409":
          $ref: "#/components/responses/Conflict"
        "422":
          $ref: "#/components/responses/ValidationError"

  /users/{userId}:
    get:
      summary: ユーザー詳細取得
      operationId: getUser
      tags:
        - Users
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          description: ユーザー詳細
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/UserSingleResponse"
        "404":
          $ref: "#/components/responses/NotFound"

    patch:
      summary: ユーザー部分更新
      operationId: updateUser
      tags:
        - Users
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
      requestBody:
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/UserUpdate"
      responses:
        "200":
          description: 更新成功
        "404":
          $ref: "#/components/responses/NotFound"

    delete:
      summary: ユーザー削除
      operationId: deleteUser
      tags:
        - Users
      parameters:
        - name: userId
          in: path
          required: true
          schema:
            type: string
      responses:
        "204":
          description: 削除成功
        "404":
          $ref: "#/components/responses/NotFound"

components:
  schemas:
    UserCreate:
      type: object
      required: [name, email]
      properties:
        name:
          type: string
          example: "Taro Yamada"
        email:
          type: string
          format: email
          example: "taro@example.com"
        role:
          type: string
          enum: [admin, user, moderator]
          default: user

    UserUpdate:
      type: object
      properties:
        name:
          type: string
        email:
          type: string
          format: email
        role:
          type: string
          enum: [admin, user, moderator]

    User:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        email:
          type: string
        role:
          type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    UserListResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: "#/components/schemas/User"
        meta:
          type: object
          properties:
            total:
              type: integer
            page:
              type: integer
            per_page:
              type: integer
            total_pages:
              type: integer

    UserSingleResponse:
      type: object
      properties:
        data:
          $ref: "#/components/schemas/User"

    ProblemDetail:
      type: object
      properties:
        type:
          type: string
        title:
          type: string
        status:
          type: integer
        detail:
          type: string

  responses:
    Unauthorized:
      description: 認証エラー
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"

    NotFound:
      description: リソース未検出
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"

    Conflict:
      description: 競合エラー
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"

    ValidationError:
      description: バリデーションエラー
      content:
        application/json:
          schema:
            $ref: "#/components/schemas/ProblemDetail"

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - BearerAuth: []
```
