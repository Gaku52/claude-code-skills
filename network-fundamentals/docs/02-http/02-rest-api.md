# REST API設計

> RESTはWeb APIの設計原則。リソース指向のURI設計、適切なHTTPメソッドの使用、ステータスコードの活用で、直感的で保守性の高いAPIを設計する。

## この章で学ぶこと

- [ ] RESTの6原則を理解する
- [ ] リソース指向のURI設計を把握する
- [ ] 実践的なAPI設計パターンを学ぶ

---

## 1. RESTの原則

```
REST（Representational State Transfer）:
  → Roy Fieldingの2000年の博士論文で提唱
  → Web の既存技術（HTTP, URI）を活用したアーキテクチャスタイル

6つの原則:
  ① クライアント・サーバー分離
     → UIとデータ処理を分離
     → 独立して進化可能

  ② ステートレス
     → 各リクエストが完結（サーバーはセッション状態を持たない）
     → スケーラビリティが向上

  ③ キャッシュ可能
     → レスポンスにキャッシュ可否を明示
     → Cache-Control, ETag

  ④ 統一インターフェース
     → リソースの識別（URI）
     → 表現を通じたリソース操作（JSON/XML）
     → 自己記述メッセージ（Content-Type等）
     → HATEOAS（ハイパーメディア）

  ⑤ 階層化システム
     → ロードバランサー、プロキシ、ゲートウェイを挟める
     → クライアントは意識しない

  ⑥ コードオンデマンド（任意）
     → サーバーからクライアントに実行コードを送信
     → JavaScript等
```

---

## 2. URI設計

```
リソース指向のURI:

  ✓ 良い設計:
  GET    /api/v1/users              — ユーザー一覧
  GET    /api/v1/users/123          — ユーザー詳細
  POST   /api/v1/users              — ユーザー作成
  PUT    /api/v1/users/123          — ユーザー更新
  DELETE /api/v1/users/123          — ユーザー削除
  GET    /api/v1/users/123/orders   — ユーザーの注文一覧
  GET    /api/v1/users/123/orders/456 — 注文詳細

  ✗ 悪い設計:
  GET    /api/getUsers              — 動詞を使わない
  POST   /api/createUser            — メソッドに役割を持たせる
  GET    /api/user/delete/123       — GETで副作用を起こさない
  GET    /api/Users                 — 大文字を使わない

命名規則:
  → 名詞・複数形: /users, /orders, /products
  → ケバブケース: /user-profiles（スネークケースも可）
  → 小文字のみ
  → 末尾スラッシュなし: /users（/users/ ではない）
  → ファイル拡張子なし: /users（/users.json ではない）

ネスト vs フラット:
  ネスト:  GET /users/123/orders/456
  フラット: GET /orders/456

  → ネストは2階層まで
  → 3階層以上はフラットにする
  → リソースに一意のIDがあればフラットが良い
```

---

## 3. クエリパラメータ

```
一覧取得のクエリパラメータ:

  ページネーション:
  GET /api/users?page=2&per_page=20
  GET /api/users?offset=20&limit=20
  GET /api/users?cursor=eyJpZCI6MTIzfQ==  ← カーソルベース（推奨）

  ソート:
  GET /api/users?sort=created_at&order=desc
  GET /api/users?sort=-created_at  （-は降順）

  フィルタリング:
  GET /api/users?status=active&role=admin
  GET /api/users?created_after=2024-01-01

  フィールド選択:
  GET /api/users?fields=id,name,email

  検索:
  GET /api/users?q=taro

  組み合わせ:
  GET /api/users?status=active&sort=-created_at&page=1&per_page=20

ページネーション方式の比較:
  ┌────────────┬─────────┬─────────────────────────┐
  │ 方式       │ メリット │ デメリット               │
  ├────────────┼─────────┼─────────────────────────┤
  │ offset     │ 実装簡単 │ 大量データでパフォ低下   │
  │ cursor     │ 高速安定 │ 任意ページに飛べない     │
  │ keyset     │ 最高速   │ 実装が複雑              │
  └────────────┴─────────┴─────────────────────────┘
```

---

## 4. レスポンス設計

```
成功レスポンス:

  一覧（GET /api/users）:
  {
    "data": [
      { "id": "1", "name": "Taro", "email": "taro@example.com" },
      { "id": "2", "name": "Hanako", "email": "hanako@example.com" }
    ],
    "meta": {
      "total": 150,
      "page": 1,
      "per_page": 20,
      "total_pages": 8
    }
  }

  詳細（GET /api/users/1）:
  {
    "data": {
      "id": "1",
      "name": "Taro",
      "email": "taro@example.com",
      "created_at": "2024-01-15T10:30:00Z"
    }
  }

  作成成功（POST /api/users → 201 Created）:
  {
    "data": { "id": "3", "name": "Jiro", ... }
  }
  Location: /api/users/3

エラーレスポンス（RFC 7807 Problem Details）:
  {
    "type": "https://api.example.com/errors/validation",
    "title": "Validation Error",
    "status": 422,
    "detail": "The request body contains invalid fields.",
    "errors": [
      { "field": "email", "message": "Invalid email format" },
      { "field": "age", "message": "Must be 18 or older" }
    ]
  }
```

---

## 5. バージョニング

```
APIバージョニング戦略:

  ① URIバージョニング（最も一般的）:
     GET /api/v1/users
     GET /api/v2/users
     → わかりやすい、キャッシュしやすい

  ② ヘッダーバージョニング:
     GET /api/users
     Accept: application/vnd.example.v2+json
     → URIがクリーン

  ③ クエリパラメータ:
     GET /api/users?version=2
     → 簡単だが、キャッシュキーが増える

  推奨: URIバージョニング（/api/v1/）

  バージョンアップのルール:
  → 破壊的変更がある場合のみメジャーバージョンを上げる
  → フィールド追加は破壊的変更ではない（v1のまま）
  → フィールド削除、型変更、必須化は破壊的変更（v2へ）
  → 旧バージョンは最低12ヶ月サポート
```

---

## 6. 認証とレート制限

```
認証:
  Bearer Token（JWT）:
  Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

  API Key:
  X-API-Key: your-api-key-here

  → Authorization ヘッダーが標準的

レート制限:
  レスポンスヘッダーで通知:
  X-RateLimit-Limit: 100       — 制限数（/分 等）
  X-RateLimit-Remaining: 42    — 残り回数
  X-RateLimit-Reset: 1640000000 — リセット時刻（Unix秒）

  制限超過時:
  429 Too Many Requests
  Retry-After: 60

  一般的な制限:
  → 認証済み:    100リクエスト/分
  → 未認証:      20リクエスト/分
  → 書き込み(POST/PUT/DELETE): より厳しく
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| REST | リソース指向、ステートレス、HTTP活用 |
| URI設計 | 名詞・複数形、2階層まで |
| ページネーション | cursor方式が高速で安定 |
| バージョニング | URIベース（/api/v1/）推奨 |
| エラー | RFC 7807 Problem Details形式 |

---

## 次に読むべきガイド
→ [[03-caching.md]] — HTTPキャッシュ

---

## 参考文献
1. Fielding, R. "Architectural Styles and the Design of Network-based Software Architectures." 2000.
2. RFC 7807. "Problem Details for HTTP APIs." IETF, 2016.
