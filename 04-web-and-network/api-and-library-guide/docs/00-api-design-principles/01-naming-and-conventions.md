# 命名規則と慣例

> API の命名は開発者体験（DX）に直結する。一貫性のあるエンドポイント命名、レスポンス構造、エラー設計、日時・ID・列挙型の規約を確立し、使いやすいAPIを設計する。命名の統一はAPIの予測可能性を高め、ドキュメントを読まなくても直感的に利用できるAPIの基盤となる。

## この章で学ぶこと

- [ ] エンドポイントとフィールドの命名規則を理解する
- [ ] レスポンスのエンベロープ設計を把握する
- [ ] 一貫性のあるエラーレスポンスを学ぶ
- [ ] ヘッダー規約とメタデータの標準化を習得する
- [ ] 日時・ID・列挙型の統一規約を確立する
- [ ] 実務プロジェクトでの命名ガイドラインを策定する
- [ ] 国際化対応の命名パターンを理解する
- [ ] OpenAPI仕様での命名規則の実装方法を把握する

---

## 1. エンドポイント命名

### 1.1 基本ルール

```
基本ルール:
  ✓ 名詞・複数形:     /users, /orders, /products
  ✓ ケバブケース:      /user-profiles, /order-items
  ✓ 小文字のみ
  ✓ 末尾スラッシュなし: /users（✗ /users/）
  ✗ 動詞を使わない:   /getUsers, /createOrder
  ✗ 大文字:           /Users, /OrderItems
  ✗ アンダースコア:   /user_profiles
  ✗ ファイル拡張子:   /users.json

  リソースの階層:
  /users/{userId}
  /users/{userId}/orders
  /users/{userId}/orders/{orderId}
  → 2階層まで（3階層以上はフラットに）

  コレクション操作:
  GET    /users           — 一覧取得
  POST   /users           — 作成
  GET    /users/{id}      — 詳細取得
  PUT    /users/{id}      — 完全更新
  PATCH  /users/{id}      — 部分更新
  DELETE /users/{id}      — 削除

  アクション（RESTに収まらない操作）:
  POST   /users/{id}/activate     — ユーザーの有効化
  POST   /users/{id}/reset-password — パスワードリセット
  POST   /orders/{id}/cancel      — 注文キャンセル
  → 動詞が必要な場合はサブリソースとして表現
```

### 1.2 リソース名の選定ガイドライン

```
リソース名の選定基準:

  1. ビジネスドメインの用語を使う
  ─────────────────────────────
  ✓ /invoices     （請求書）
  ✗ /payment-documents
  ✓ /shipments    （出荷）
  ✗ /delivery-records
  ✓ /subscriptions（サブスクリプション）
  ✗ /recurring-payments

  2. 略語は避ける
  ─────────────────────────────
  ✓ /organizations
  ✗ /orgs
  ✓ /configurations
  ✗ /configs
  ✓ /applications
  ✗ /apps
  例外: 業界標準の略語は許容
  ✓ /urls, /ids, /apis

  3. 技術用語よりビジネス用語
  ─────────────────────────────
  ✓ /users/{id}/preferences
  ✗ /users/{id}/settings-records
  ✓ /notifications
  ✗ /push-message-queue-items

  4. 曖昧な名前を避ける
  ─────────────────────────────
  ✗ /data, /items, /things, /resources, /objects
  ✓ 具体的なリソース名を使う

  5. 単数形と複数形のルール
  ─────────────────────────────
  コレクション: 常に複数形
  ✓ /users, /orders, /products

  シングルトンリソース: 単数形
  ✓ /users/{id}/profile    （各ユーザーに1つ）
  ✓ /settings              （システム設定は1つ）
  ✓ /users/{id}/cart        （各ユーザーに1つのカート）
```

### 1.3 階層設計の実践パターン

```
パターン1: シンプルなCRUD
─────────────────────────
  GET    /products
  POST   /products
  GET    /products/{productId}
  PUT    /products/{productId}
  PATCH  /products/{productId}
  DELETE /products/{productId}

パターン2: 親子関係
─────────────────────────
  GET    /users/{userId}/orders              — ユーザーの注文一覧
  POST   /users/{userId}/orders              — ユーザーの注文作成
  GET    /users/{userId}/orders/{orderId}    — 特定注文の詳細

  ※ 注文を直接アクセスできるエイリアスも提供:
  GET    /orders/{orderId}                   — 注文詳細（直接アクセス）

パターン3: 深いネストの回避
─────────────────────────
  ✗ 避けるべき（3階層以上）:
  GET /users/{userId}/orders/{orderId}/items/{itemId}/reviews

  ✓ フラット化する:
  GET /order-items/{itemId}/reviews
  GET /reviews?orderId={orderId}

パターン4: 多対多関係
─────────────────────────
  ✓ 関連リソースとして表現:
  GET    /users/{userId}/roles               — ユーザーのロール一覧
  PUT    /users/{userId}/roles/{roleId}      — ロールの割り当て
  DELETE /users/{userId}/roles/{roleId}      — ロールの解除

  ✓ 別アプローチ（ジャンクションリソース）:
  GET    /role-assignments?userId={userId}
  POST   /role-assignments
  DELETE /role-assignments/{assignmentId}

パターン5: 検索とフィルタリング
─────────────────────────
  ✓ クエリパラメータで表現:
  GET /products?category=electronics&minPrice=1000&sort=-price

  ✓ 複雑な検索は専用エンドポイント:
  POST /products/search
  {
    "query": "laptop",
    "filters": {
      "category": ["electronics"],
      "priceRange": { "min": 50000, "max": 200000 }
    },
    "sort": [{ "field": "price", "order": "asc" }]
  }

パターン6: バルク操作
─────────────────────────
  ✓ コレクションに対する一括操作:
  POST   /users/bulk-create
  PATCH  /users/bulk-update
  DELETE /users/bulk-delete

  ✓ またはバッチリクエスト:
  POST /batch
  {
    "requests": [
      { "method": "POST", "url": "/users", "body": { ... } },
      { "method": "POST", "url": "/users", "body": { ... } }
    ]
  }

パターン7: 非同期操作
─────────────────────────
  ✓ 長時間かかる操作:
  POST /reports/generate
  → 202 Accepted
  {
    "data": {
      "jobId": "job_abc123",
      "status": "processing",
      "statusUrl": "/jobs/job_abc123"
    }
  }

  GET /jobs/job_abc123
  {
    "data": {
      "jobId": "job_abc123",
      "status": "completed",
      "resultUrl": "/reports/rpt_xyz789",
      "completedAt": "2024-06-01T12:30:00Z"
    }
  }
```

### 1.4 operationId の命名規則

```
operationIdの命名パターン:

  基本形: {動詞}{リソース名}（camelCase）

  CRUD操作:
  ─────────────────────────────
  GET    /users          → listUsers
  POST   /users          → createUser
  GET    /users/{id}     → getUser
  PUT    /users/{id}     → updateUser（完全更新）
  PATCH  /users/{id}     → patchUser（部分更新）
  DELETE /users/{id}     → deleteUser

  サブリソース:
  ─────────────────────────────
  GET    /users/{id}/orders     → listUserOrders
  POST   /users/{id}/orders     → createUserOrder
  GET    /users/{id}/profile    → getUserProfile
  PUT    /users/{id}/profile    → updateUserProfile

  アクション:
  ─────────────────────────────
  POST   /users/{id}/activate       → activateUser
  POST   /users/{id}/deactivate     → deactivateUser
  POST   /orders/{id}/cancel        → cancelOrder
  POST   /users/{id}/reset-password → resetUserPassword

  検索:
  ─────────────────────────────
  GET    /users?search=...     → searchUsers
  POST   /products/search      → searchProducts

  命名ルール:
  ✓ camelCaseで統一
  ✓ 動詞 + 名詞の組み合わせ
  ✓ APIクライアント生成時のメソッド名になる
  ✓ 一意であること（API全体で重複なし）
  ✗ get_user, GetUser, get-user は避ける
```

---

## 2. フィールド命名

### 2.1 ケーシング規約

```
JSON フィールド名:

  推奨: camelCase（JavaScript / フロントエンド親和性）
  {
    "userId": "123",
    "firstName": "Taro",
    "lastName": "Yamada",
    "emailAddress": "taro@example.com",
    "createdAt": "2024-01-15T10:30:00Z",
    "isActive": true,
    "phoneNumber": "+81-90-1234-5678",
    "postalCode": "100-0001"
  }

  許容: snake_case（Ruby/Python エコシステム）
  {
    "user_id": "123",
    "first_name": "Taro",
    "last_name": "Yamada",
    "email_address": "taro@example.com",
    "created_at": "2024-01-15T10:30:00Z",
    "is_active": true,
    "phone_number": "+81-90-1234-5678",
    "postal_code": "100-0001"
  }

  → プロジェクト内で統一が最重要
  → フロントエンドがJavaScript/TypeScriptならcamelCase推奨
  → バックエンドがPython/RubyならsnakeCase許容

ケーシング変換の自動化:
  → APIゲートウェイやミドルウェアで変換
  → クライアントライブラリで変換
  → 仕様書では1つのケーシングに統一
```

### 2.2 フィールド名の命名パターン

```
1. 日時フィールド
─────────────────────────────
  命名パターン: {動詞の過去分詞}At

  createdAt       — 作成日時
  updatedAt       — 更新日時
  deletedAt       — 削除日時（論理削除）
  publishedAt     — 公開日時
  expiredAt       — 有効期限
  lastLoginAt     — 最終ログイン日時
  scheduledAt     — 予定日時
  completedAt     — 完了日時
  startedAt       — 開始日時
  cancelledAt     — キャンセル日時

  フォーマット:
  → ISO 8601 形式: "2024-01-15T10:30:00Z"
  → UTC で統一
  → タイムゾーンは別フィールド（必要な場合）

  {
    "createdAt": "2024-01-15T10:30:00Z",
    "timezone": "Asia/Tokyo"
  }

  日付のみ（時刻なし）:
  → ISO 8601 日付形式: "2024-01-15"
  → フィールド名: birthDate, hireDate, dueDate
  → "On" サフィックスも許容: expiresOn

2. ID フィールド
─────────────────────────────
  推奨形式:

  UUID v4: "550e8400-e29b-41d4-a716-446655440000"
  → ランダム生成、衝突確率極めて低い
  → 分散システムで最適

  UUID v7: "01908816-2e7d-7c0e-8a1c-3b4d5e6f7a8b"
  → タイムスタンプ内蔵、ソート可能
  → UUID v4の後継として推奨

  ULID: "01ARZ3NDEKTSV4RRFFQ69G5FAV"
  → ソート可能、URLフレンドリー
  → 26文字で表現

  プレフィックス付きID: "user_2c9p8K3nMv", "ord_7x4mR9yLpq"
  → リソース種別が一目でわかる
  → Stripe, Twilio等が採用
  → ログやデバッグで便利

  ✗ 避けるべき:
  → 自動増分整数（予測可能、セキュリティリスク）
  → 連番（データ量の推測が可能）

  複数IDフィールドの命名:
  {
    "id": "user_2c9p8K3nMv",           ← 自身のID
    "organizationId": "org_5x8mN3pLq", ← 外部キー
    "createdBy": "user_7y2kR9wMn"      ← 作成者ID
  }

3. 真偽値フィールド
─────────────────────────────
  命名パターン: is/has/can/should + 形容詞/動詞

  is + 状態:
  isActive        — 有効か
  isVerified      — 検証済みか
  isPublished     — 公開済みか
  isDeleted       — 削除済みか（論理削除）
  isDefault       — デフォルトか
  isLocked        — ロック中か

  has + 所有:
  hasPassword     — パスワード設定済みか
  hasAvatar       — アバター画像があるか
  hasPremium      — プレミアムプランか
  hasChildren     — 子要素があるか

  can + 能力:
  canEdit         — 編集可能か
  canDelete       — 削除可能か
  canShare        — 共有可能か
  canExport       — エクスポート可能か

  should + 推奨:
  shouldNotify    — 通知すべきか
  shouldSync      — 同期すべきか

  ✗ 避けるべき:
  → active（isActiveを使う）
  → flag, status（具体的な名前を使う）
  → enabled/disabled（isEnabledを使う）

4. 列挙型フィールド
─────────────────────────────
  推奨: snake_case の小文字

  ステータス系:
  "status": "active"
  "status": "in_progress"
  "status": "completed"
  "status": "cancelled"

  種別系:
  "type": "credit_card"
  "type": "bank_transfer"
  "type": "digital_wallet"

  ロール系:
  "role": "admin"
  "role": "moderator"
  "role": "member"
  "role": "guest"

  優先度系:
  "priority": "critical"
  "priority": "high"
  "priority": "medium"
  "priority": "low"

  列挙値の命名ルール:
  ✓ snake_case の小文字で統一
  ✓ 新しい値を追加しても後方互換
  ✓ 意味が明確で省略しない
  ✗ 数値コード（1, 2, 3）は避ける
  ✗ 大文字（ACTIVE, IN_PROGRESS）は意見が分かれる

5. 金額フィールド
─────────────────────────────
  推奨: 最小単位の整数 + 通貨コード

  {
    "amount": 1500,          ← 1500円（整数で表現）
    "currency": "JPY",       ← ISO 4217 通貨コード
    "displayAmount": "¥1,500" ← 表示用（参考値）
  }

  {
    "amount": 2999,          ← $29.99（セント単位）
    "currency": "USD"
  }

  複数金額:
  {
    "subtotal": 10000,
    "tax": 1000,
    "shippingFee": 500,
    "discount": -200,
    "total": 11300,
    "currency": "JPY"
  }

6. 配列フィールド
─────────────────────────────
  命名パターン: 複数形名詞

  "users": [...]
  "tags": [...]
  "permissions": [...]
  "attachments": [...]
  "lineItems": [...]

  カウントフィールド:
  "userCount": 150
  "commentCount": 42
  "totalItems": 500

  ✗ 避けるべき:
  → "userList"（Listサフィックスは不要）
  → "userData"（Dataサフィックスは不要）

7. ネストオブジェクトの命名
─────────────────────────────
  {
    "user": {                        ← 単数形
      "id": "user_abc",
      "name": "田中太郎",
      "profile": {                   ← 関連オブジェクト
        "bio": "エンジニア",
        "avatarUrl": "https://..."
      },
      "address": {                   ← 住所オブジェクト
        "postalCode": "100-0001",
        "prefecture": "東京都",
        "city": "千代田区",
        "street": "丸の内1-1-1",
        "building": "東京ビル3F"
      },
      "metadata": {                  ← メタデータ
        "lastLoginIp": "192.168.1.1",
        "userAgent": "Mozilla/5.0..."
      }
    }
  }
```

### 2.3 フィールド命名のアンチパターン

```
アンチパターン集:

  1. 型名をフィールド名に含める
  ✗ "nameString", "ageNumber", "isActiveBool"
  ✓ "name", "age", "isActive"

  2. 冗長なプレフィックス
  ✗ "userName", "userEmail" （Userオブジェクト内）
  ✓ "name", "email"
  ※ ただしIDは "userId" のように明示が推奨

  3. 略語の乱用
  ✗ "desc", "qty", "amt", "addr", "msg"
  ✓ "description", "quantity", "amount", "address", "message"
  例外: "id", "url", "api" は許容

  4. 一貫性のないケーシング
  ✗ 同一APIで "createdAt" と "updated_at" が混在
  ✓ どちらかに統一

  5. 意味の重複
  ✗ "priceAmount"（priceだけで金額とわかる）
  ✗ "nameString"（nameだけで文字列とわかる）
  ✓ "price", "name"

  6. 否定形の真偽値
  ✗ "isNotActive", "isDisabled", "isInvalid"
  ✓ "isActive"（false で非アクティブ）
  ✓ "isEnabled"（false で無効）
  ✓ "isValid"（false で無効）
```

---

## 3. レスポンス設計

### 3.1 エンベロープパターン

```
エンベロープパターン:

  単一リソース:
  {
    "data": {
      "id": "user_abc123",
      "name": "田中太郎",
      "email": "tanaka@example.com",
      "role": "admin",
      "isActive": true,
      "createdAt": "2024-01-15T10:30:00Z",
      "updatedAt": "2024-06-01T12:00:00Z"
    }
  }

  コレクション:
  {
    "data": [
      { "id": "user_abc", "name": "田中太郎" },
      { "id": "user_def", "name": "山田花子" }
    ],
    "meta": {
      "total": 150,
      "page": 1,
      "perPage": 20,
      "totalPages": 8,
      "hasNextPage": true,
      "hasPrevPage": false
    },
    "links": {
      "self": "/users?page=1&per_page=20",
      "first": "/users?page=1&per_page=20",
      "last": "/users?page=8&per_page=20",
      "next": "/users?page=2&per_page=20",
      "prev": null
    }
  }

  空のレスポンス:
  204 No Content（ボディなし）
  → DELETE 成功時等

  作成成功:
  201 Created
  Location: /api/v1/users/user_xyz789
  {
    "data": {
      "id": "user_xyz789",
      "name": "佐藤次郎",
      "email": "sato@example.com",
      "createdAt": "2024-06-15T09:00:00Z"
    }
  }
```

### 3.2 null vs 省略 の設計方針

```
nullと省略の使い分け:

  基本方針:
  → null: フィールドが存在するが値がない
  → 省略: フィールドが該当しない、または未リクエスト

  例1: ユーザープロフィール
  {
    "name": "田中太郎",
    "phone": null,           ← 電話番号未設定
    "bio": null,             ← 自己紹介未設定
    // "deletedAt" は省略    ← 削除されていない場合
    "avatarUrl": null         ← アバター未設定
  }

  例2: 論理削除されたユーザー
  {
    "name": "田中太郎",
    "deletedAt": "2024-06-01T12:00:00Z",  ← 削除日時あり
    "phone": null
  }

  例3: フィールド選択（sparse fieldsets）
  GET /users/123?fields=name,email
  {
    "data": {
      "id": "user_abc",
      "name": "田中太郎",
      "email": "tanaka@example.com"
      // phone, bio等は省略（リクエストされていない）
    }
  }

  OpenAPIでの定義:
  nullableフィールド:
    phone:
      type: string
      nullable: true          ← null可能
    deletedAt:
      type: string
      format: date-time
      nullable: true          ← null可能（未削除時）

  省略可能フィールド:
  → requiredリストに含めない
  → ドキュメントで省略条件を明記
```

### 3.3 レスポンスの拡張パターン

```typescript
// パターン1: リソース展開（Expand / Include）
// GET /orders/ord_abc?expand=customer,items.product

// レスポンス:
{
  "data": {
    "id": "ord_abc",
    "status": "confirmed",
    "customer": {                    // ← 展開されたリソース
      "id": "user_123",
      "name": "田中太郎",
      "email": "tanaka@example.com"
    },
    "items": [
      {
        "id": "item_1",
        "quantity": 2,
        "product": {                 // ← ネストされた展開
          "id": "prod_xyz",
          "name": "ワイヤレスイヤホン",
          "price": 15000
        }
      }
    ],
    "totalAmount": 30000,
    "currency": "JPY"
  }
}

// 展開なしの場合（デフォルト）:
// GET /orders/ord_abc
{
  "data": {
    "id": "ord_abc",
    "status": "confirmed",
    "customerId": "user_123",        // ← IDのみ
    "items": [
      {
        "id": "item_1",
        "quantity": 2,
        "productId": "prod_xyz"      // ← IDのみ
      }
    ],
    "totalAmount": 30000,
    "currency": "JPY"
  }
}
```

```typescript
// パターン2: フィールド選択（Sparse Fieldsets）
// GET /users?fields[users]=name,email&fields[profile]=bio

{
  "data": [
    {
      "id": "user_abc",
      "name": "田中太郎",
      "email": "tanaka@example.com",
      "profile": {
        "bio": "エンジニア"
      }
    }
  ]
}
```

```typescript
// パターン3: サイドローディング
// 関連リソースを別セクションに含める（JSON:API風）
{
  "data": [
    {
      "id": "ord_1",
      "customerId": "user_abc",
      "productIds": ["prod_1", "prod_2"]
    }
  ],
  "included": {
    "users": [
      { "id": "user_abc", "name": "田中太郎" }
    ],
    "products": [
      { "id": "prod_1", "name": "商品A", "price": 1000 },
      { "id": "prod_2", "name": "商品B", "price": 2000 }
    ]
  },
  "meta": {
    "total": 50,
    "page": 1
  }
}
```

### 3.4 ステータスコードの使い分け

```
HTTPステータスコード使い分けガイド:

━━━ 2xx 成功 ━━━
  200 OK
  → GET: リソースの取得成功
  → PUT/PATCH: リソースの更新成功
  → POST: 操作の成功（リソース作成以外）

  201 Created
  → POST: リソースの作成成功
  → Locationヘッダーで作成されたリソースのURLを返す
  → レスポンスボディで作成されたリソースを返す

  202 Accepted
  → 非同期処理の受付成功
  → 処理はまだ完了していない
  → ステータス確認用URLをレスポンスに含める

  204 No Content
  → DELETE: 削除成功
  → PUT/PATCH: 更新成功（レスポンスボディ不要の場合）
  → レスポンスボディなし

━━━ 3xx リダイレクト ━━━
  301 Moved Permanently
  → リソースのURLが恒久的に変更
  → 新しいURLをLocationヘッダーで通知

  304 Not Modified
  → 条件付きリクエスト（If-None-Match/If-Modified-Since）
  → キャッシュが有効、ボディなし

━━━ 4xx クライアントエラー ━━━
  400 Bad Request
  → リクエスト形式が不正（JSONパースエラー等）
  → クエリパラメータの型不正

  401 Unauthorized
  → 認証トークンなし、または無効
  → ログインが必要

  403 Forbidden
  → 認証済みだが権限不足
  → アクセス拒否

  404 Not Found
  → リソースが存在しない
  → URLが不正

  405 Method Not Allowed
  → 対象リソースに対して許可されていないHTTPメソッド
  → Allowヘッダーで許可メソッドを通知

  409 Conflict
  → リソースの競合（重複メール等）
  → 楽観ロックの競合

  410 Gone
  → リソースが永久に削除された
  → 以前は存在していたが、現在は利用不可

  413 Content Too Large
  → リクエストボディが大きすぎる
  → ファイルアップロードのサイズ超過

  415 Unsupported Media Type
  → サポートしていないContent-Type

  422 Unprocessable Entity
  → リクエスト形式は正しいがバリデーションエラー
  → ビジネスロジックの制約違反

  429 Too Many Requests
  → レート制限超過
  → Retry-Afterヘッダーで再試行時期を通知

━━━ 5xx サーバーエラー ━━━
  500 Internal Server Error
  → サーバー内部のエラー
  → 詳細をクライアントに返さない
  → requestIdでサーバーログと紐付け

  502 Bad Gateway
  → 上流サーバーからの不正なレスポンス

  503 Service Unavailable
  → サービスが一時的に利用不可
  → Retry-Afterヘッダーで復旧見込みを通知

  504 Gateway Timeout
  → 上流サーバーからの応答タイムアウト
```

---

## 4. エラー設計

### 4.1 RFC 7807 Problem Details

```json
// RFC 7807 Problem Details 完全実装例

// バリデーションエラー (422)
{
  "type": "https://api.example.com/errors/validation",
  "title": "Validation Error",
  "status": 422,
  "detail": "The request body contains invalid fields.",
  "instance": "/api/v1/users",
  "errors": [
    {
      "field": "email",
      "code": "INVALID_FORMAT",
      "message": "有効なメールアドレスを入力してください",
      "rejectedValue": "not-an-email"
    },
    {
      "field": "age",
      "code": "OUT_OF_RANGE",
      "message": "年齢は18歳以上150歳以下で入力してください",
      "rejectedValue": 5
    },
    {
      "field": "name",
      "code": "REQUIRED",
      "message": "名前は必須です"
    }
  ],
  "requestId": "req_550e8400-e29b-41d4"
}

// 認証エラー (401)
{
  "type": "https://api.example.com/errors/unauthorized",
  "title": "Unauthorized",
  "status": 401,
  "detail": "認証トークンが無効または期限切れです。再ログインしてください。",
  "instance": "/api/v1/users/me"
}

// 権限エラー (403)
{
  "type": "https://api.example.com/errors/forbidden",
  "title": "Forbidden",
  "status": 403,
  "detail": "この操作を実行する権限がありません。管理者に連絡してください。",
  "instance": "/api/v1/admin/users",
  "requiredPermission": "admin:users:write"
}

// リソース未発見 (404)
{
  "type": "https://api.example.com/errors/not-found",
  "title": "Not Found",
  "status": 404,
  "detail": "ユーザー 'user_abc123' は存在しません。",
  "instance": "/api/v1/users/user_abc123"
}

// 競合 (409)
{
  "type": "https://api.example.com/errors/conflict",
  "title": "Conflict",
  "status": 409,
  "detail": "このメールアドレスは既に使用されています。",
  "instance": "/api/v1/users",
  "conflictingField": "email",
  "conflictingValue": "tanaka@example.com"
}

// レート制限 (429)
{
  "type": "https://api.example.com/errors/rate-limit",
  "title": "Too Many Requests",
  "status": 429,
  "detail": "レート制限を超えました。60秒後に再試行してください。",
  "retryAfter": 60,
  "limit": 100,
  "remaining": 0,
  "resetAt": "2024-06-01T12:01:00Z"
}

// サーバーエラー (500)
{
  "type": "https://api.example.com/errors/internal",
  "title": "Internal Server Error",
  "status": 500,
  "detail": "予期しないエラーが発生しました。問題が続く場合はサポートに連絡してください。",
  "requestId": "req_7890abcd-ef12-3456"
}
```

### 4.2 エラーコード体系

```
エラーコード体系設計:

  命名規則: DOMAIN_ENTITY_ACTION の形式（大文字スネークケース）

  認証・認可:
  ─────────────────────────
  AUTH_TOKEN_MISSING         — トークンなし
  AUTH_TOKEN_EXPIRED         — トークン期限切れ
  AUTH_TOKEN_INVALID         — トークン不正
  AUTH_REFRESH_TOKEN_EXPIRED — リフレッシュトークン期限切れ
  AUTH_INSUFFICIENT_SCOPE    — スコープ不足
  AUTH_ACCOUNT_LOCKED        — アカウントロック
  AUTH_ACCOUNT_SUSPENDED     — アカウント停止
  AUTH_MFA_REQUIRED          — 二要素認証が必要
  AUTH_PASSWORD_INCORRECT    — パスワード不正

  ユーザー:
  ─────────────────────────
  USER_NOT_FOUND             — ユーザー未発見
  USER_EMAIL_ALREADY_EXISTS  — メールアドレス重複
  USER_EMAIL_INVALID         — メールアドレス形式不正
  USER_NAME_TOO_LONG         — 名前が長すぎる
  USER_NAME_REQUIRED         — 名前が未入力
  USER_ROLE_INVALID          — 不正なロール
  USER_CANNOT_DELETE_SELF    — 自分自身を削除できない

  注文:
  ─────────────────────────
  ORDER_NOT_FOUND            — 注文未発見
  ORDER_ALREADY_CANCELLED    — 既にキャンセル済み
  ORDER_CANNOT_CANCEL        — キャンセル不可（出荷済み等）
  ORDER_PAYMENT_FAILED       — 決済失敗
  ORDER_INSUFFICIENT_STOCK   — 在庫不足
  ORDER_AMOUNT_EXCEEDS_LIMIT — 注文金額上限超過

  汎用:
  ─────────────────────────
  VALIDATION_ERROR           — バリデーションエラー（汎用）
  REQUIRED_FIELD             — 必須フィールド未入力
  INVALID_FORMAT             — 形式不正
  OUT_OF_RANGE               — 範囲外
  TOO_LONG                   — 文字数超過
  TOO_SHORT                  — 文字数不足
  RATE_LIMIT_EXCEEDED        — レート制限超過
  INTERNAL_ERROR             — 内部エラー
  SERVICE_UNAVAILABLE        — サービス一時停止
  RESOURCE_NOT_FOUND         — リソース未発見（汎用）
  DUPLICATE_RESOURCE         — リソース重複
  CONFLICT                   — 競合
```

### 4.3 エラーレスポンスの実装例

```typescript
// TypeScript - エラーハンドリングの実装
interface ProblemDetails {
  type: string;
  title: string;
  status: number;
  detail: string;
  instance?: string;
  requestId?: string;
  errors?: FieldError[];
  [key: string]: unknown; // 拡張プロパティ
}

interface FieldError {
  field: string;
  code: string;
  message: string;
  rejectedValue?: unknown;
}

// エラークラス定義
class ApiError extends Error {
  constructor(
    public readonly statusCode: number,
    public readonly errorType: string,
    public readonly title: string,
    public readonly detail: string,
    public readonly errors?: FieldError[],
    public readonly extensions?: Record<string, unknown>,
  ) {
    super(detail);
  }

  toProblemDetails(requestId: string, instance: string): ProblemDetails {
    return {
      type: `https://api.example.com/errors/${this.errorType}`,
      title: this.title,
      status: this.statusCode,
      detail: this.detail,
      instance,
      requestId,
      ...(this.errors && { errors: this.errors }),
      ...(this.extensions || {}),
    };
  }
}

// 具象エラークラス
class ValidationError extends ApiError {
  constructor(errors: FieldError[]) {
    super(
      422,
      'validation',
      'Validation Error',
      '入力データにエラーがあります。',
      errors,
    );
  }
}

class NotFoundError extends ApiError {
  constructor(resource: string, id: string) {
    super(
      404,
      'not-found',
      'Not Found',
      `${resource} '${id}' は存在しません。`,
    );
  }
}

class ConflictError extends ApiError {
  constructor(detail: string, field?: string, value?: unknown) {
    super(
      409,
      'conflict',
      'Conflict',
      detail,
      undefined,
      field ? { conflictingField: field, conflictingValue: value } : undefined,
    );
  }
}

class UnauthorizedError extends ApiError {
  constructor(detail: string = '認証が必要です。') {
    super(401, 'unauthorized', 'Unauthorized', detail);
  }
}

class ForbiddenError extends ApiError {
  constructor(detail: string = 'この操作を実行する権限がありません。') {
    super(403, 'forbidden', 'Forbidden', detail);
  }
}

class RateLimitError extends ApiError {
  constructor(retryAfter: number) {
    super(
      429,
      'rate-limit',
      'Too Many Requests',
      `レート制限を超えました。${retryAfter}秒後に再試行してください。`,
      undefined,
      { retryAfter },
    );
  }
}
```

```typescript
// Express.js エラーハンドリングミドルウェア
import { Request, Response, NextFunction } from 'express';
import { randomUUID } from 'crypto';

function errorHandler(
  err: Error,
  req: Request,
  res: Response,
  _next: NextFunction,
) {
  const requestId = req.headers['x-request-id'] as string || `req_${randomUUID()}`;

  if (err instanceof ApiError) {
    const problem = err.toProblemDetails(requestId, req.originalUrl);
    res
      .status(err.statusCode)
      .header('Content-Type', 'application/problem+json')
      .header('X-Request-Id', requestId)
      .json(problem);
    return;
  }

  // 予期しないエラー
  console.error(`[${requestId}] Unhandled error:`, err);

  res
    .status(500)
    .header('Content-Type', 'application/problem+json')
    .header('X-Request-Id', requestId)
    .json({
      type: 'https://api.example.com/errors/internal',
      title: 'Internal Server Error',
      status: 500,
      detail: '予期しないエラーが発生しました。',
      requestId,
      instance: req.originalUrl,
    });
}

// 使用例
app.post('/api/v1/users', async (req, res, next) => {
  try {
    const errors: FieldError[] = [];

    if (!req.body.name) {
      errors.push({
        field: 'name',
        code: 'REQUIRED_FIELD',
        message: '名前は必須です。',
      });
    }

    if (!req.body.email) {
      errors.push({
        field: 'email',
        code: 'REQUIRED_FIELD',
        message: 'メールアドレスは必須です。',
      });
    } else if (!isValidEmail(req.body.email)) {
      errors.push({
        field: 'email',
        code: 'INVALID_FORMAT',
        message: '有効なメールアドレスを入力してください。',
        rejectedValue: req.body.email,
      });
    }

    if (errors.length > 0) {
      throw new ValidationError(errors);
    }

    const existingUser = await userService.findByEmail(req.body.email);
    if (existingUser) {
      throw new ConflictError(
        'このメールアドレスは既に使用されています。',
        'email',
        req.body.email,
      );
    }

    const user = await userService.create(req.body);
    res
      .status(201)
      .header('Location', `/api/v1/users/${user.id}`)
      .json({ data: user });
  } catch (err) {
    next(err);
  }
});
```

```go
// Go - エラーハンドリングの実装
package api

import (
    "encoding/json"
    "fmt"
    "net/http"
)

// ProblemDetails はRFC 7807のエラーレスポンス
type ProblemDetails struct {
    Type      string       `json:"type"`
    Title     string       `json:"title"`
    Status    int          `json:"status"`
    Detail    string       `json:"detail"`
    Instance  string       `json:"instance,omitempty"`
    RequestID string       `json:"requestId,omitempty"`
    Errors    []FieldError `json:"errors,omitempty"`
}

type FieldError struct {
    Field         string      `json:"field"`
    Code          string      `json:"code"`
    Message       string      `json:"message"`
    RejectedValue interface{} `json:"rejectedValue,omitempty"`
}

// APIError はアプリケーション固有のエラー型
type APIError struct {
    StatusCode int
    ErrorType  string
    Title      string
    Detail     string
    Errors     []FieldError
}

func (e *APIError) Error() string {
    return fmt.Sprintf("[%d] %s: %s", e.StatusCode, e.Title, e.Detail)
}

func (e *APIError) ToProblemDetails(requestID, instance string) ProblemDetails {
    return ProblemDetails{
        Type:      fmt.Sprintf("https://api.example.com/errors/%s", e.ErrorType),
        Title:     e.Title,
        Status:    e.StatusCode,
        Detail:    e.Detail,
        Instance:  instance,
        RequestID: requestID,
        Errors:    e.Errors,
    }
}

// エラーファクトリ関数
func NewValidationError(errors []FieldError) *APIError {
    return &APIError{
        StatusCode: http.StatusUnprocessableEntity,
        ErrorType:  "validation",
        Title:      "Validation Error",
        Detail:     "入力データにエラーがあります。",
        Errors:     errors,
    }
}

func NewNotFoundError(resource, id string) *APIError {
    return &APIError{
        StatusCode: http.StatusNotFound,
        ErrorType:  "not-found",
        Title:      "Not Found",
        Detail:     fmt.Sprintf("%s '%s' は存在しません。", resource, id),
    }
}

func NewConflictError(detail string) *APIError {
    return &APIError{
        StatusCode: http.StatusConflict,
        ErrorType:  "conflict",
        Title:      "Conflict",
        Detail:     detail,
    }
}

// エラーハンドリングミドルウェア
func ErrorMiddleware(next http.Handler) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        defer func() {
            if rec := recover(); rec != nil {
                requestID := r.Header.Get("X-Request-Id")
                w.Header().Set("Content-Type", "application/problem+json")
                w.Header().Set("X-Request-Id", requestID)
                w.WriteHeader(http.StatusInternalServerError)

                problem := ProblemDetails{
                    Type:      "https://api.example.com/errors/internal",
                    Title:     "Internal Server Error",
                    Status:    500,
                    Detail:    "予期しないエラーが発生しました。",
                    RequestID: requestID,
                    Instance:  r.URL.Path,
                }
                json.NewEncoder(w).Encode(problem)
            }
        }()
        next.ServeHTTP(w, r)
    })
}
```

---

## 5. ヘッダー規約

### 5.1 標準ヘッダー

```
標準リクエストヘッダー:
  Content-Type: application/json
  Accept: application/json
  Authorization: Bearer <token>
  Accept-Language: ja, en;q=0.8
  Accept-Encoding: gzip, deflate
  If-None-Match: "etag-value"
  If-Modified-Since: Wed, 21 Oct 2015 07:28:00 GMT
  Idempotency-Key: key_abc123

標準レスポンスヘッダー:
  Content-Type: application/json; charset=utf-8
  Content-Language: ja
  Cache-Control: private, max-age=0, no-cache
  ETag: "33a64df551425fcc55e4d42a148795d9f25f89d4"
  Last-Modified: Wed, 21 Oct 2015 07:28:00 GMT
  Location: /api/v1/users/user_abc123     （201 Created時）
  Retry-After: 60                          （429 / 503時）
  Vary: Accept, Authorization, Accept-Language
```

### 5.2 カスタムヘッダー

```
カスタムヘッダー:

  リクエスト追跡:
  X-Request-Id: req_550e8400-e29b     — リクエスト追跡ID
  X-Correlation-Id: corr_abc123       — 分散トレーシングID
  X-Client-Version: 2.1.0             — クライアントバージョン
  X-Client-Platform: ios              — クライアントプラットフォーム

  レート制限:
  X-RateLimit-Limit: 100              — 制限値
  X-RateLimit-Remaining: 42           — 残り回数
  X-RateLimit-Reset: 1640000000       — リセット時刻（Unix timestamp）

  ページネーション:
  X-Total-Count: 150                  — 総件数
  X-Page-Count: 8                     — 総ページ数

  API廃止:
  Deprecation: true                   — 廃止予定
  Sunset: Sat, 01 Jun 2025 00:00:00 GMT  — 廃止日
  Link: </v2/users>; rel="successor-version"

  ※ X- プレフィックスの取り扱い:
  → RFC 6648 で非推奨になった
  → しかし実務では依然として広く使われている
  → 新規APIでは X- なしのカスタムヘッダーも検討
  → RateLimit-Limit, RateLimit-Remaining 等
```

### 5.3 冪等性キーの実装

```typescript
// 冪等性キーの実装例
import { Redis } from 'ioredis';

interface IdempotencyRecord {
  statusCode: number;
  headers: Record<string, string>;
  body: unknown;
  createdAt: string;
}

class IdempotencyMiddleware {
  private redis: Redis;
  private ttlSeconds: number;

  constructor(redis: Redis, ttlSeconds: number = 86400) { // 24時間
    this.redis = redis;
    this.ttlSeconds = ttlSeconds;
  }

  middleware() {
    return async (req: Request, res: Response, next: NextFunction) => {
      // GET/DELETE は冪等なのでスキップ
      if (['GET', 'DELETE', 'PUT'].includes(req.method)) {
        return next();
      }

      const idempotencyKey = req.headers['idempotency-key'] as string;

      // POSTリクエストには冪等性キーを推奨
      if (!idempotencyKey && req.method === 'POST') {
        console.warn('POST request without Idempotency-Key');
        return next(); // キーなしでも処理は続行
      }

      if (!idempotencyKey) {
        return next();
      }

      const cacheKey = `idempotency:${idempotencyKey}`;

      // 既存のレスポンスを確認
      const cached = await this.redis.get(cacheKey);
      if (cached) {
        const record: IdempotencyRecord = JSON.parse(cached);

        // キャッシュされたレスポンスを返す
        Object.entries(record.headers).forEach(([key, value]) => {
          res.setHeader(key, value);
        });
        res.setHeader('X-Idempotency-Replayed', 'true');
        res.status(record.statusCode).json(record.body);
        return;
      }

      // レスポンスをキャプチャ
      const originalJson = res.json.bind(res);
      res.json = (body: any) => {
        const record: IdempotencyRecord = {
          statusCode: res.statusCode,
          headers: {
            'Content-Type': 'application/json',
          },
          body,
          createdAt: new Date().toISOString(),
        };

        // 成功レスポンスのみキャッシュ
        if (res.statusCode >= 200 && res.statusCode < 300) {
          this.redis.setex(cacheKey, this.ttlSeconds, JSON.stringify(record));
        }

        return originalJson(body);
      };

      next();
    };
  }
}

// 使用例
const idempotency = new IdempotencyMiddleware(redis);
app.use('/api/v1', idempotency.middleware());
```

---

## 6. 国際化対応

### 6.1 多言語レスポンスの設計

```
国際化（i18n）対応のAPI設計:

  リクエスト:
  Accept-Language: ja, en;q=0.8, zh;q=0.5

  レスポンス:
  Content-Language: ja

  エラーメッセージの多言語化:
  {
    "type": "https://api.example.com/errors/validation",
    "title": "Validation Error",
    "status": 422,
    "detail": "入力データにエラーがあります。",
    "errors": [
      {
        "field": "email",
        "code": "INVALID_FORMAT",
        "message": "有効なメールアドレスを入力してください"
      }
    ]
  }

  多言語コンテンツの返却パターン:

  パターン1: Accept-Languageに基づく単一言語返却
  GET /products/123
  Accept-Language: ja
  →
  {
    "data": {
      "id": "prod_123",
      "name": "ワイヤレスイヤホン",
      "description": "高音質Bluetoothイヤホン"
    }
  }

  パターン2: 全言語を含むレスポンス
  GET /products/123?include_translations=true
  →
  {
    "data": {
      "id": "prod_123",
      "name": "ワイヤレスイヤホン",
      "description": "高音質Bluetoothイヤホン",
      "translations": {
        "en": {
          "name": "Wireless Earphones",
          "description": "High-quality Bluetooth earphones"
        },
        "zh": {
          "name": "无线耳机",
          "description": "高品质蓝牙耳机"
        }
      }
    }
  }

  パターン3: ロケール別フィールド
  {
    "data": {
      "id": "prod_123",
      "name_ja": "ワイヤレスイヤホン",
      "name_en": "Wireless Earphones",
      "name_zh": "无线耳机"
    }
  }
  → パターン3は拡張性が低いため、パターン1 or 2を推奨
```

### 6.2 タイムゾーン対応

```
タイムゾーン処理の規約:

  基本方針:
  1. サーバーは常にUTCで保存・返却
  2. クライアントがローカル時間に変換
  3. タイムゾーン情報が必要な場合は別フィールド

  リクエスト:
  {
    "scheduledAt": "2024-06-15T10:00:00Z",    ← UTC
    "timezone": "Asia/Tokyo"                    ← 表示用タイムゾーン
  }

  レスポンス:
  {
    "scheduledAt": "2024-06-15T01:00:00Z",     ← UTC（=JST 10:00）
    "timezone": "Asia/Tokyo",
    "localTime": "2024-06-15T10:00:00+09:00"   ← 参考値（ローカル時間）
  }

  日付のみ（時刻なし）のフィールド:
  {
    "birthDate": "1990-05-15",   ← ISO 8601日付形式
    "dueDate": "2024-12-31"
  }

  期間の表現:
  {
    "duration": "PT1H30M",       ← ISO 8601期間形式（1時間30分）
    "trialPeriod": "P30D"        ← 30日間
  }
```

---

## 7. OpenAPIでの命名規則の実装

### 7.1 スキーマ定義のベストプラクティス

```yaml
# OpenAPI 3.1 での命名規則適用例
openapi: '3.1.0'
info:
  title: Naming Convention Example API
  version: '1.0.0'

components:
  schemas:
    # スキーマ名: PascalCase
    User:
      type: object
      required: [id, name, email, role, isActive, createdAt]
      properties:
        # プロパティ名: camelCase
        id:
          type: string
          format: uuid
          readOnly: true
          description: ユーザーの一意識別子
          example: "550e8400-e29b-41d4-a716-446655440000"
        name:
          type: string
          minLength: 1
          maxLength: 100
          description: ユーザーの表示名
          example: "田中太郎"
        email:
          type: string
          format: email
          description: メールアドレス（システム内で一意）
          example: "tanaka@example.com"
        role:
          type: string
          # 列挙値: snake_case の小文字
          enum: [user, admin, moderator]
          default: user
          description: ユーザーのロール
        isActive:
          type: boolean
          default: true
          description: アカウントが有効かどうか
        # 日時: ISO 8601 + At サフィックス
        createdAt:
          type: string
          format: date-time
          readOnly: true
          description: 作成日時（UTC）
        updatedAt:
          type: string
          format: date-time
          readOnly: true
          nullable: true
          description: 最終更新日時（UTC）
        deletedAt:
          type: string
          format: date-time
          readOnly: true
          nullable: true
          description: 削除日時（論理削除、null=未削除）
        profile:
          $ref: '#/components/schemas/UserProfile'

    UserProfile:
      type: object
      properties:
        bio:
          type: string
          maxLength: 500
          nullable: true
          description: 自己紹介文
        avatarUrl:
          type: string
          format: uri
          nullable: true
          description: アバター画像のURL
        location:
          type: string
          maxLength: 100
          nullable: true
        birthDate:
          type: string
          format: date
          nullable: true
          description: 生年月日（YYYY-MM-DD）
        socialLinks:
          type: object
          nullable: true
          properties:
            twitter:
              type: string
              nullable: true
            github:
              type: string
              nullable: true

    # リクエスト/レスポンスのラッパー
    # 命名規則: {Action}{Resource}Request / {Resource}Response
    CreateUserRequest:
      type: object
      required: [name, email]
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        role:
          type: string
          enum: [user, admin, moderator]
          default: user

    UpdateUserRequest:
      type: object
      required: [name, email]
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        role:
          type: string
          enum: [user, admin, moderator]

    PatchUserRequest:
      type: object
      minProperties: 1
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        isActive:
          type: boolean

    UserResponse:
      type: object
      properties:
        data:
          $ref: '#/components/schemas/User'

    UserListResponse:
      type: object
      properties:
        data:
          type: array
          items:
            $ref: '#/components/schemas/User'
        meta:
          $ref: '#/components/schemas/PaginationMeta'
        links:
          $ref: '#/components/schemas/PaginationLinks'

    # 共通スキーマの命名
    PaginationMeta:
      type: object
      properties:
        total:
          type: integer
          description: 総件数
        page:
          type: integer
          description: 現在のページ番号
        perPage:
          type: integer
          description: 1ページあたりの件数
        totalPages:
          type: integer
          description: 総ページ数
        hasNextPage:
          type: boolean
        hasPrevPage:
          type: boolean

    PaginationLinks:
      type: object
      properties:
        self:
          type: string
          format: uri
        first:
          type: string
          format: uri
        last:
          type: string
          format: uri
        prev:
          type: string
          format: uri
          nullable: true
        next:
          type: string
          format: uri
          nullable: true

    # RFC 7807 エラー
    ProblemDetails:
      type: object
      required: [type, title, status]
      properties:
        type:
          type: string
          format: uri
        title:
          type: string
        status:
          type: integer
        detail:
          type: string
        instance:
          type: string
          format: uri
        requestId:
          type: string
        errors:
          type: array
          items:
            $ref: '#/components/schemas/FieldError'

    FieldError:
      type: object
      required: [field, code, message]
      properties:
        field:
          type: string
        code:
          type: string
        message:
          type: string
        rejectedValue: {}
```

### 7.2 Spectralでの命名規則チェック

```yaml
# .spectral.yaml - 命名規則のLintルール
extends:
  - spectral:oas

rules:
  # パス名: ケバブケース
  paths-kebab-case:
    given: "$.paths[*]~"
    then:
      function: pattern
      functionOptions:
        match: "^(/[a-z][a-z0-9-]*(/\\{[a-zA-Z]+\\})?)+$"
    severity: error
    message: "パス名はケバブケースで記述してください（例: /user-profiles）"

  # operationId: camelCase
  operation-id-camel-case:
    given: "$.paths[*][*].operationId"
    then:
      function: casing
      functionOptions:
        type: camel
    severity: error
    message: "operationIdはcamelCaseで記述してください"

  # スキーマ名: PascalCase
  schema-names-pascal-case:
    given: "$.components.schemas[*]~"
    then:
      function: casing
      functionOptions:
        type: pascal
    severity: error
    message: "スキーマ名はPascalCaseで記述してください"

  # プロパティ名: camelCase
  property-names-camel-case:
    given: "$..properties[*]~"
    then:
      function: casing
      functionOptions:
        type: camel
    severity: error
    message: "プロパティ名はcamelCaseで記述してください"

  # enum値: snake_case
  enum-values-snake-case:
    given: "$..enum[*]"
    then:
      function: pattern
      functionOptions:
        match: "^[a-z][a-z0-9_]*$"
    severity: warn
    message: "enum値はsnake_caseで記述してください"

  # 日時フィールド: Atサフィックス
  datetime-field-suffix:
    given: "$..properties[*][?(@.format=='date-time')]~"
    then:
      function: pattern
      functionOptions:
        match: "At$"
    severity: warn
    message: "日時フィールドは'At'サフィックスを使ってください（例: createdAt）"

  # 真偽値フィールド: is/has/canプレフィックス
  boolean-field-prefix:
    given: "$..properties[?(@.type=='boolean')]~"
    then:
      function: pattern
      functionOptions:
        match: "^(is|has|can|should)"
    severity: warn
    message: "真偽値フィールドはis/has/can/shouldプレフィックスを使ってください"
```

---

## 8. 業界標準APIの命名分析

### 8.1 主要APIの命名パターン比較

```
主要APIの命名パターン:

  Stripe API:
  ─────────────────────────
  エンドポイント: /v1/customers, /v1/payment_intents
  ID形式: cus_xxxxx, pi_xxxxx（プレフィックス付き）
  フィールド: snake_case
  列挙型: snake_case（"requires_payment_method"）
  日時: Unix timestamp
  特徴: プレフィックス付きIDで可読性が高い

  GitHub API:
  ─────────────────────────
  エンドポイント: /repos/{owner}/{repo}/issues
  ID形式: 数値ID
  フィールド: snake_case
  列挙型: snake_case（"pull_request"）
  日時: ISO 8601
  特徴: ハイパーメディア（HATEOAS）リンク

  Google Cloud API:
  ─────────────────────────
  エンドポイント: /v1/projects/{projectId}/datasets
  ID形式: 文字列ID
  フィールド: camelCase
  列挙型: UPPER_SNAKE_CASE（"RUNNING", "FAILED"）
  日時: ISO 8601 / protobuf Timestamp
  特徴: resource name パターン

  Twilio API:
  ─────────────────────────
  エンドポイント: /2010-04-01/Accounts/{sid}/Messages
  ID形式: SID（AC, SM等のプレフィックス + 32文字）
  フィールド: snake_case
  列挙型: snake_case
  日時: RFC 2822
  特徴: 日付ベースのバージョニング

  Shopify API:
  ─────────────────────────
  エンドポイント: /admin/api/2024-01/products.json
  ID形式: 数値ID
  フィールド: snake_case
  列挙型: snake_case
  日時: ISO 8601
  特徴: 日付ベースバージョン + .json拡張子

  共通パターン:
  ─────────────────────────
  → ほとんどのAPIがsnake_caseを採用
  → 日時はISO 8601が主流（Stripeを除く）
  → IDはUUIDまたはプレフィックス付き文字列
  → エンドポイントは名詞・複数形
  → エラーレスポンスはRFC 7807に収束
```

### 8.2 自社APIスタイルガイドの策定

```
自社APIスタイルガイド テンプレート:

1. 基本方針
   - フィールド名: camelCase
   - URL: ケバブケース、複数形名詞
   - operationId: camelCase
   - スキーマ名: PascalCase
   - 列挙値: snake_case（小文字）

2. ID規約
   - 形式: UUIDv7（ソート可能）
   - 表示: ハイフン付き（550e8400-e29b-41d4-...）
   - 外部公開APIはプレフィックス付きを検討（user_xxx）

3. 日時規約
   - 形式: ISO 8601（"2024-01-15T10:30:00Z"）
   - タイムゾーン: UTC
   - フィールド名: createdAt, updatedAt, deletedAt
   - 日付のみ: ISO 8601日付（"2024-01-15"）

4. レスポンス規約
   - エンベロープ: { "data": ... }
   - コレクション: { "data": [...], "meta": {...}, "links": {...} }
   - 空レスポンス: 204 No Content
   - 作成: 201 Created + Location ヘッダー

5. エラー規約
   - 形式: RFC 7807 Problem Details
   - Content-Type: application/problem+json
   - エラーコード: UPPER_SNAKE_CASE
   - requestId: 全レスポンスに含める

6. ヘッダー規約
   - 認証: Authorization: Bearer <token>
   - リクエスト追跡: X-Request-Id
   - レート制限: X-RateLimit-Limit, X-RateLimit-Remaining
   - 冪等性: Idempotency-Key（POST）

7. バージョニング
   - URL: /v1/users
   - メジャーバージョンのみ
```

---

## 9. 実践演習

### 演習1: 命名規則の修正

```
以下のAPI仕様の命名問題を特定し、修正してください:

修正前:
  POST /api/createUser
  GET /api/GetUserList
  PUT /api/user_profile/{user_id}
  DELETE /api/Users/{UserID}

  レスポンス:
  {
    "user_Name": "Taro",
    "Email": "taro@example.com",
    "created_date": "2024/01/15",
    "active": true,
    "type": 1,
    "user_id": 42
  }

修正後:
  POST /api/v1/users
  GET /api/v1/users
  PUT /api/v1/users/{userId}/profile
  DELETE /api/v1/users/{userId}

  レスポンス:
  {
    "data": {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Taro",
      "email": "taro@example.com",
      "createdAt": "2024-01-15T00:00:00Z",
      "isActive": true,
      "type": "standard",
    }
  }
```

### 演習2: エラーレスポンスの設計

```
以下のシナリオに対するエラーレスポンスを設計してください:

1. 未認証ユーザーがアクセス
2. メールアドレスの形式不正 + 名前が空
3. 既に存在するメールアドレスでユーザー作成
4. 存在しないユーザーIDでアクセス
5. レート制限超過
6. サーバー内部エラー

回答例は上記セクション4を参照してください。
各エラーに対して:
- 適切なステータスコード
- RFC 7807形式のレスポンスボディ
- エラーコード
- ユーザー向けメッセージ
を定義します。
```

### 演習3: スタイルガイドの策定

```
課題: 新規プロジェクトのAPIスタイルガイドを策定してください。

以下の項目を決定し、文書化する:
1. フィールドのケーシング規則（camelCase / snake_case）
2. ID の形式（UUID / ULID / プレフィックス付き）
3. 日時の形式と表現
4. 列挙型の命名規則
5. エラーレスポンスの形式
6. ページネーションの設計
7. バージョニング方針

チーム内で合意を取り、Spectralルールとして実装してください。
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| エンドポイント | 名詞・複数形、ケバブケース、2階層まで |
| フィールド | camelCase/snake_case統一、ISO 8601日時 |
| ID | UUID/ULID推奨、プレフィックス付きも有効 |
| 真偽値 | is/has/canプレフィックス |
| 列挙型 | snake_case小文字で統一 |
| レスポンス | data + meta エンベロープ |
| エラー | RFC 7807 Problem Details、エラーコード体系 |
| ヘッダー | 標準ヘッダー + カスタムヘッダーの規約 |
| 国際化 | Accept-Language、UTC統一 |
| 一貫性 | Spectralで自動チェック |

---

## 次に読むべきガイド
-> [[02-versioning-strategy.md]] -- バージョニング戦略
-> [[03-pagination-and-filtering.md]] -- ページネーションとフィルタリング

---

## 参考文献
1. RFC 7807. "Problem Details for HTTP APIs." IETF, 2016.
2. RFC 9457. "Problem Details for HTTP APIs (updated)." IETF, 2023.
3. RFC 6648. "Deprecating the X- Prefix." IETF, 2012.
4. Google. "API Design Guide." cloud.google.com, 2024.
5. Microsoft. "REST API Guidelines." github.com/microsoft/api-guidelines, 2024.
6. Stripe. "API Reference." stripe.com/docs/api, 2024.
7. GitHub. "REST API Documentation." docs.github.com, 2024.
8. Zalando. "RESTful API Guidelines." opensource.zalando.com/restful-api-guidelines, 2024.
9. JSON:API. "JSON:API Specification." jsonapi.org, 2024.
10. Stoplight. "API Style Guide." stoplight.io, 2024.
