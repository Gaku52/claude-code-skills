# 命名規則と慣例

> API の命名は開発者体験（DX）に直結する。一貫性のあるエンドポイント命名、レスポンス構造、エラー設計、日時・ID・列挙型の規約を確立し、使いやすいAPIを設計する。

## この章で学ぶこと

- [ ] エンドポイントとフィールドの命名規則を理解する
- [ ] レスポンスのエンベロープ設計を把握する
- [ ] 一貫性のあるエラーレスポンスを学ぶ

---

## 1. エンドポイント命名

```
基本ルール:
  ✓ 名詞・複数形:     /users, /orders, /products
  ✓ ケバブケース:      /user-profiles, /order-items
  ✓ 小文字のみ
  ✓ 末尾スラッシュなし: /users（✗ /users/）
  ✗ 動詞を使わない:   /getUsers, /createOrder
  ✗ 大文字:           /Users, /OrderItems

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

---

## 2. フィールド命名

```
JSON フィールド名:

  推奨: camelCase（JavaScript / フロントエンド親和性）
  {
    "userId": "123",
    "firstName": "Taro",
    "lastName": "Yamada",
    "emailAddress": "taro@example.com",
    "createdAt": "2024-01-15T10:30:00Z"
  }

  許容: snake_case（Ruby/Python エコシステム）
  {
    "user_id": "123",
    "first_name": "Taro",
    "created_at": "2024-01-15T10:30:00Z"
  }

  → プロジェクト内で統一が最重要

日時:
  → ISO 8601 形式: "2024-01-15T10:30:00Z"
  → UTC で統一
  → タイムゾーンは別フィールド（必要な場合）
  → フィールド名: createdAt, updatedAt, deletedAt

ID:
  → UUID v4: "550e8400-e29b-41d4-a716-446655440000"
  → ULID: "01ARZ3NDEKTSV4RRFFQ69G5FAV"（ソート可能）
  → 自動増分整数は避ける（予測可能でセキュリティリスク）

真偽値:
  → is, has, can プレフィックス
  → isActive, hasPermission, canEdit

列挙型:
  → スネークケース大文字: "PENDING", "IN_PROGRESS", "COMPLETED"
  → または小文字: "pending", "in_progress", "completed"
  → プロジェクト内で統一
```

---

## 3. レスポンス設計

```
エンベロープパターン:

  単一リソース:
  {
    "data": {
      "id": "123",
      "name": "Taro",
      "email": "taro@example.com"
    }
  }

  コレクション:
  {
    "data": [
      { "id": "1", "name": "Taro" },
      { "id": "2", "name": "Hanako" }
    ],
    "meta": {
      "total": 150,
      "page": 1,
      "perPage": 20,
      "totalPages": 8,
      "hasNextPage": true
    }
  }

  空のレスポンス:
  204 No Content（ボディなし）
  → DELETE 成功時等

  作成成功:
  201 Created
  Location: /api/v1/users/456
  {
    "data": { "id": "456", "name": "Jiro", ... }
  }

null vs 省略:
  → null: フィールドが存在するが値がない
  → 省略: フィールドが該当しない
  {
    "phone": null,           ← 電話番号未設定
    // "deletedAt" は省略    ← 削除されていない
  }
```

---

## 4. エラー設計

```
RFC 7807 Problem Details:

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
        "message": "Invalid email format"
      },
      {
        "field": "age",
        "code": "OUT_OF_RANGE",
        "message": "Must be between 18 and 150"
      }
    ],
    "requestId": "req_550e8400-e29b"
  }

エラーコード体系:
  DOMAIN_ENTITY_ACTION の形式
  USER_NOT_FOUND
  USER_EMAIL_ALREADY_EXISTS
  ORDER_PAYMENT_FAILED
  AUTH_TOKEN_EXPIRED
  RATE_LIMIT_EXCEEDED

ステータスコードとエラーのマッピング:
  400: バリデーションエラー、不正なリクエスト形式
  401: 認証トークンなし/無効
  403: 認証済みだが権限なし
  404: リソースが存在しない
  409: 競合（楽観ロック等）
  422: バリデーションエラー（400と使い分け）
  429: レート制限超過
  500: サーバー内部エラー（詳細をクライアントに返さない）

エラーメッセージのルール:
  ✓ ユーザー向け: 何が問題で、どう修正すべきか
  ✓ 開発者向け: requestId でログと紐付け可能
  ✗ スタックトレースをクライアントに返さない
  ✗ 内部的なエラーメッセージを露出しない
```

---

## 5. ヘッダー規約

```
標準ヘッダー:
  Content-Type: application/json
  Accept: application/json
  Authorization: Bearer <token>
  Accept-Language: ja

カスタムヘッダー:
  X-Request-Id: req_550e8400      — リクエスト追跡
  X-RateLimit-Limit: 100          — レート制限
  X-RateLimit-Remaining: 42       — 残り回数
  X-RateLimit-Reset: 1640000000   — リセット時刻

  ※ X- プレフィックスは RFC 6648 で非推奨になったが、
     実務では依然として広く使われている

冪等性キー:
  Idempotency-Key: key_abc123
  → POST リクエストの冪等性を保証
  → クライアントが一意のキーを生成
  → 同じキーのリクエストは1回のみ処理
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| エンドポイント | 名詞・複数形、ケバブケース、2階層まで |
| フィールド | camelCase/snake_case統一、ISO 8601日時 |
| レスポンス | data + meta エンベロープ |
| エラー | RFC 7807、エラーコード体系、requestId |

---

## 次に読むべきガイド
→ [[02-versioning-strategy.md]] — バージョニング戦略

---

## 参考文献
1. RFC 7807. "Problem Details for HTTP APIs." IETF, 2016.
2. Google. "API Design Guide." cloud.google.com, 2024.
