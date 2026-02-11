# RESTベストプラクティス

> RESTの6原則を超えた実践的なベストプラクティス。HATEOAS、冪等性の設計、コンテンツネゴシエーション、バルク操作、部分更新（PATCH）まで、プロダクションレベルのREST API設計を習得する。

## この章で学ぶこと

- [ ] HATEOASとハイパーメディアの活用を理解する
- [ ] 冪等性の設計と冪等性キーの実装を把握する
- [ ] PATCH（部分更新）とバルク操作の設計を学ぶ

---

## 1. REST の6原則（復習）

```
Roy Fielding の REST アーキテクチャスタイル（2000年）:

  ① Client-Server（クライアント・サーバー分離）:
     → UIとデータストレージの関心を分離
     → 各コンポーネントが独立に進化可能

  ② Stateless（ステートレス）:
     → 各リクエストに必要な情報を全て含める
     → サーバーにセッション状態を保持しない
     → スケーラビリティの基盤

  ③ Cacheable（キャッシュ可能）:
     → レスポンスにキャッシュ可否を明示
     → Cache-Control, ETag, Last-Modified

  ④ Uniform Interface（統一インターフェース）:
     → リソースの識別（URI）
     → 表現によるリソース操作（JSON/XML）
     → 自己記述的メッセージ
     → HATEOAS

  ⑤ Layered System（階層化システム）:
     → クライアントは直接サーバーか中間層か区別しない
     → ロードバランサー、CDN、APIゲートウェイ

  ⑥ Code on Demand（オプション）:
     → サーバーからクライアントにコードを送信可能
     → JavaScript、Applet 等
```

---

## 2. HATEOAS

```
HATEOAS = Hypermedia As The Engine Of Application State
  → レスポンスに次のアクションのリンクを含める
  → クライアントがURLをハードコードしなくてよい

例:
  GET /api/v1/orders/123

  {
    "data": {
      "id": "123",
      "status": "pending",
      "total": 5000,
      "items": [...]
    },
    "links": {
      "self": { "href": "/api/v1/orders/123" },
      "cancel": { "href": "/api/v1/orders/123/cancel", "method": "POST" },
      "pay": { "href": "/api/v1/orders/123/pay", "method": "POST" },
      "items": { "href": "/api/v1/orders/123/items" }
    }
  }

  statusが "shipped" の場合:
  {
    "data": { "id": "123", "status": "shipped", ... },
    "links": {
      "self": { "href": "/api/v1/orders/123" },
      "track": { "href": "/api/v1/orders/123/tracking" },
      "return": { "href": "/api/v1/orders/123/return", "method": "POST" }
    }
    // cancel と pay のリンクは消える（不可能なアクション）
  }

利点:
  ✓ クライアントがAPI構造を事前に知る必要がない
  ✓ 状態遷移の可視化
  ✓ APIの進化が容易（URLが変わってもリンクで追従）

現実:
  → 完全なHATEOASは実装コストが高い
  → 実務では「関連リソースへのリンク」程度が一般的
  → GitHub API が好例（各レスポンスにURLを含む）
```

---

## 3. 冪等性の設計

```
冪等性（Idempotency）:
  → 同じリクエストを何回実行しても結果が同じ

  冪等なメソッド:
    GET     — 常に冪等（副作用なし）
    PUT     — 冪等（同じリソースを同じ状態に上書き）
    DELETE  — 冪等（削除済みなら何もしない）
    HEAD    — 常に冪等
    OPTIONS — 常に冪等

  冪等でないメソッド:
    POST    — 2回実行すると2つリソースが作成される
    PATCH   — 相対的な変更の場合（例: +1）

冪等性キー（Idempotency Key）:
  → POSTリクエストに冪等性を持たせる仕組み
  → クライアントが一意のキーを生成

  リクエスト:
    POST /api/v1/payments
    Idempotency-Key: pay_abc123def456
    {
      "amount": 5000,
      "currency": "JPY"
    }

  サーバー側の処理:
    1. Idempotency-Key を受信
    2. キーでキャッシュを検索
    3. 未処理 → 処理実行 → 結果をキーと共に保存
    4. 処理済み → 保存済みの結果を返す
```

```javascript
// 冪等性キーのサーバー実装（Redis使用）
const Redis = require('ioredis');
const redis = new Redis();

async function idempotencyMiddleware(req, res, next) {
  const key = req.headers['idempotency-key'];
  if (!key) return next(); // キーなし → 通常処理

  const cacheKey = `idempotency:${req.path}:${key}`;
  const cached = await redis.get(cacheKey);

  if (cached) {
    // 処理済み → キャッシュした結果を返す
    const { statusCode, body } = JSON.parse(cached);
    return res.status(statusCode).json(body);
  }

  // レスポンスをインターセプトして保存
  const originalJson = res.json.bind(res);
  res.json = async (body) => {
    await redis.setex(cacheKey, 86400, JSON.stringify({
      statusCode: res.statusCode,
      body,
    }));
    return originalJson(body);
  };

  next();
}

// 使用
app.post('/api/v1/payments', idempotencyMiddleware, paymentHandler);
```

---

## 4. コンテンツネゴシエーション

```
Content Negotiation:
  → クライアントが希望する表現形式を指定

  リクエスト:
    Accept: application/json          ← JSON希望
    Accept: application/xml           ← XML希望
    Accept: text/csv                  ← CSV希望
    Accept: application/pdf           ← PDF希望
    Accept-Language: ja               ← 日本語希望
    Accept-Encoding: gzip, br         ← 圧縮形式

  レスポンス:
    Content-Type: application/json; charset=utf-8
    Content-Language: ja
    Content-Encoding: br

  406 Not Acceptable:
    → サーバーがクライアントの希望形式に対応していない場合

実装パターン:
  ① Accept ヘッダーベース（標準的）
  ② 拡張子ベース: /users.json, /users.xml
  ③ クエリパラメータ: /users?format=csv

推奨:
  → デフォルトはJSON
  → 管理画面向けにCSVエクスポート対応
  → Accept ヘッダーで切り替え
```

---

## 5. 部分更新（PATCH）

```
PUT vs PATCH:
  PUT:   リソースの完全置換（全フィールドを送信）
  PATCH: リソースの部分更新（変更フィールドのみ送信）

① Merge Patch（RFC 7396）:
  PATCH /api/v1/users/123
  Content-Type: application/merge-patch+json
  {
    "name": "Updated Name",
    "address": null          ← nullで削除
  }

  ルール:
  → 含まれるフィールド: 値を更新
  → null: フィールドを削除
  → 含まれないフィールド: 変更なし

  制限:
  → 配列の部分更新ができない（配列は全置換）
  → null値の設定と削除が区別できない

② JSON Patch（RFC 6902）:
  PATCH /api/v1/users/123
  Content-Type: application/json-patch+json
  [
    { "op": "replace", "path": "/name", "value": "Updated Name" },
    { "op": "add", "path": "/tags/-", "value": "vip" },
    { "op": "remove", "path": "/address" },
    { "op": "move", "from": "/oldField", "path": "/newField" },
    { "op": "test", "path": "/version", "value": 5 }
  ]

  操作:
  add     — フィールド/配列要素を追加
  remove  — フィールド/配列要素を削除
  replace — フィールドの値を置換
  move    — フィールドを移動
  copy    — フィールドをコピー
  test    — 値の検証（楽観ロック）

推奨:
  → 一般的なAPI: Merge Patch（シンプル）
  → 複雑な更新が必要: JSON Patch
  → 配列操作が多い: JSON Patch
```

---

## 6. バルク操作

```
複数リソースの一括操作:

① バッチリクエスト:
  POST /api/v1/users/batch
  {
    "operations": [
      { "method": "POST", "body": { "name": "Taro", "email": "taro@example.com" } },
      { "method": "POST", "body": { "name": "Hanako", "email": "hanako@example.com" } },
      { "method": "PATCH", "id": "123", "body": { "name": "Updated" } }
    ]
  }

  レスポンス:
  {
    "results": [
      { "status": 201, "data": { "id": "456", "name": "Taro" } },
      { "status": 201, "data": { "id": "457", "name": "Hanako" } },
      { "status": 200, "data": { "id": "123", "name": "Updated" } }
    ],
    "meta": {
      "total": 3,
      "succeeded": 3,
      "failed": 0
    }
  }

② 一括削除:
  DELETE /api/v1/users?ids=1,2,3

  または:
  POST /api/v1/users/batch-delete
  { "ids": ["1", "2", "3"] }

③ 一括更新:
  PATCH /api/v1/users/batch
  {
    "ids": ["1", "2", "3"],
    "update": { "status": "inactive" }
  }

設計ポイント:
  → 部分失敗を許容する（一部成功・一部失敗）
  → 各操作の結果を個別に返す
  → HTTPステータスは 200（全体として成功）
  → 個別の結果にステータスを含める
  → バッチサイズの上限を設ける（例: 100件）
  → トランザクション要否を明確にする
```

---

## 7. 楽観的ロック

```
楽観的ロック（Optimistic Locking）:
  → ETag / If-Match ヘッダーで同時更新を検出

  フロー:
  1. GET /api/v1/users/123
     → ETag: "v5"

  2. PUT /api/v1/users/123
     If-Match: "v5"
     { "name": "Updated" }

  3a. 成功（ETagが一致）:
     200 OK
     ETag: "v6"

  3b. 失敗（他のクライアントが先に更新）:
     409 Conflict
     {
       "type": "https://api.example.com/errors/conflict",
       "title": "Resource Conflict",
       "status": 409,
       "detail": "Resource was modified by another client.",
       "currentETag": "v7"
     }

  代替: バージョンフィールド
  {
    "id": "123",
    "name": "Taro",
    "version": 5
  }

  PUT /api/v1/users/123
  { "name": "Updated", "version": 5 }

  → version不一致なら409 Conflict
```

---

## 8. レスポンス圧縮とキャッシュ

```
圧縮:
  クライアント:
    Accept-Encoding: gzip, br

  サーバー:
    Content-Encoding: br
    → Brotli: JSONで30-40%小さい（gzip比）
    → gzip: 広くサポート
    → 1KB未満は圧縮不要（オーバーヘッド > 効果）

キャッシュ:
  不変リソース:
    Cache-Control: public, max-age=31536000, immutable
    → 画像、ビルドアセット等

  変化するリソース:
    Cache-Control: private, no-cache
    ETag: "abc123"
    → 毎回サーバーに確認（304 Not Modified）

  リスト:
    Cache-Control: private, max-age=60
    → 60秒間キャッシュ（許容可能な鮮度）

  機密データ:
    Cache-Control: no-store
    → キャッシュ禁止
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| HATEOAS | レスポンスに次のアクションリンクを含める |
| 冪等性 | Idempotency-Keyで POST を冪等に |
| PATCH | Merge Patch（シンプル）vs JSON Patch（高機能） |
| バルク操作 | 部分失敗を許容、個別結果を返す |
| 楽観ロック | ETag / If-Match で同時更新検出 |

---

## 次に読むべきガイド
→ [[01-graphql-fundamentals.md]] — GraphQL基礎

---

## 参考文献
1. Fielding, R. "Architectural Styles and the Design of Network-based Software Architectures." 2000.
2. RFC 7396. "JSON Merge Patch." IETF, 2014.
3. RFC 6902. "JavaScript Object Notation (JSON) Patch." IETF, 2013.
