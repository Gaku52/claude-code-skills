# RESTベストプラクティス

> RESTの6原則を超えた実践的なベストプラクティス。リソース設計、HTTPメソッド、ステータスコード、エラーレスポンス、HATEOAS、冪等性の設計、コンテンツネゴシエーション、バルク操作、部分更新（PATCH）まで、プロダクションレベルのREST API設計を網羅的に習得する。

## この章で学ぶこと

- [ ] リソース指向のURI設計原則を理解する
- [ ] HTTPメソッドの正しい使い分けを把握する
- [ ] ステータスコードの選択基準を習得する
- [ ] RFC 9457準拠のエラーレスポンス設計を学ぶ
- [ ] HATEOASとハイパーメディアの活用を理解する
- [ ] 冪等性の設計と冪等性キーの実装を把握する
- [ ] PATCH（部分更新）とバルク操作の設計を学ぶ
- [ ] アンチパターンを識別し回避する力を養う

## 前提知識

- HTTPメソッドとステータスコード → 参照: [HTTPの基礎](../../network-fundamentals/docs/02-http/00-http-basics.md)
- API設計の基本原則 → 参照: [API First設計](../00-api-design-principles/00-api-first-design.md)
- API命名規則 → 参照: [命名規則と慣例](../00-api-design-principles/01-naming-and-conventions.md)

---

## 1. REST の6原則（復習と深掘り）

Roy Fielding が2000年の博士論文で提唱したRESTアーキテクチャスタイルは、6つの制約から構成される。これらの制約はWebの成功要因を形式化したものであり、単なる「設計ガイドライン」ではなくアーキテクチャ上の**制約（constraint）**として定義されている。

```
Roy Fielding の REST アーキテクチャスタイル（2000年）:

  ┌─────────────────────────────────────────────────────────────────┐
  │                    REST の 6 つの制約                            │
  ├─────────────────┬───────────────────────────────────────────────┤
  │ ① Client-Server │ UIとデータの関心を分離                         │
  │                 │ → 各コンポーネントが独立に進化可能               │
  │                 │ → フロントエンドとバックエンドの独立デプロイ       │
  ├─────────────────┼───────────────────────────────────────────────┤
  │ ② Stateless     │ 各リクエストに必要な情報を全て含める             │
  │                 │ → サーバーにセッション状態を保持しない            │
  │                 │ → スケーラビリティの基盤（任意のサーバーが処理可能）│
  ├─────────────────┼───────────────────────────────────────────────┤
  │ ③ Cacheable     │ レスポンスにキャッシュ可否を明示                 │
  │                 │ → Cache-Control, ETag, Last-Modified           │
  │                 │ → ネットワーク効率とユーザー体感速度の改善        │
  ├─────────────────┼───────────────────────────────────────────────┤
  │ ④ Uniform       │ リソースの識別（URI）                           │
  │   Interface     │ 表現によるリソース操作（JSON/XML）               │
  │                 │ 自己記述的メッセージ                              │
  │                 │ HATEOAS                                        │
  ├─────────────────┼───────────────────────────────────────────────┤
  │ ⑤ Layered       │ クライアントは直接サーバーか中間層か区別しない    │
  │   System        │ → ロードバランサー、CDN、APIゲートウェイ         │
  │                 │ → セキュリティ、監視、変換の追加が透過的          │
  ├─────────────────┼───────────────────────────────────────────────┤
  │ ⑥ Code on       │ サーバーからクライアントにコードを送信可能         │
  │   Demand        │ → JavaScript、WebAssembly 等                   │
  │   (optional)    │ → 唯一のオプション制約                          │
  └─────────────────┴───────────────────────────────────────────────┘
```

### 1.1 制約間の関係

6つの制約は独立ではなく、互いに影響し合う。Stateless制約はCacheableを前提とし、Uniform InterfaceはLayered Systemを可能にする。

```
  ┌──────────┐     依存      ┌───────────┐
  │Stateless │ ─────────────→│ Cacheable │
  │          │ セッションなし  │           │ キャッシュで性能補完
  └──────────┘  だから必要    └───────────┘
       │                          │
       │ 前提                     │ 活用
       ▼                          ▼
  ┌──────────────┐          ┌────────────┐
  │   Uniform    │ ───────→ │  Layered   │
  │  Interface   │ 統一IF    │  System    │
  │              │ が透過    │            │
  └──────────────┘ 層を実現  └────────────┘
       │
       │ 構成要素
       ▼
  ┌──────────┐
  │ HATEOAS  │
  │          │ ← Uniform Interface の最重要構成要素
  └──────────┘
```

---

## 2. リソース設計の原則

REST APIの設計において最も重要なのは**リソースの識別と命名**である。リソースはシステム内のエンティティや概念を表し、URIによって一意に識別される。

### 2.1 URI設計のルール

```
┌────────────────────────────────────────────────────────────────┐
│                     URI 設計の基本原則                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. 名詞を使う（動詞は使わない）                                 │
│     良い例: GET /users                                         │
│     悪い例: GET /getUsers, POST /createUser                    │
│                                                                │
│  2. 複数形を使う                                                │
│     良い例: /users, /orders, /products                         │
│     悪い例: /user, /order, /product                            │
│                                                                │
│  3. ケバブケース（kebab-case）を使う                             │
│     良い例: /order-items, /shipping-addresses                  │
│     悪い例: /orderItems, /order_items                          │
│                                                                │
│  4. 階層関係はパスで表現                                         │
│     良い例: /users/123/orders                                  │
│     注意: 階層は2段まで推奨（3段以上は可読性低下）               │
│                                                                │
│  5. バージョニングはパスに含める                                  │
│     良い例: /api/v1/users                                      │
│     代替案: Accept: application/vnd.myapi.v1+json              │
│                                                                │
│  6. フィルタ・ソート・ページングはクエリパラメータ                 │
│     良い例: /users?status=active&sort=-created_at&page=2       │
│                                                                │
│  7. 末尾スラッシュは付けない                                     │
│     良い例: /users/123                                         │
│     悪い例: /users/123/                                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 リソースモデリングの実例

ECサイトを題材に、リソース設計の全体像を示す。

```javascript
// ECサイトのリソース設計例

// ── コレクションリソース（一覧） ──
// GET /api/v1/products              商品一覧
// GET /api/v1/categories            カテゴリ一覧
// GET /api/v1/users                 ユーザー一覧（管理者のみ）
// GET /api/v1/orders                注文一覧

// ── 個別リソース（単体） ──
// GET /api/v1/products/prod_abc123  商品詳細
// GET /api/v1/users/usr_def456      ユーザー詳細
// GET /api/v1/orders/ord_ghi789     注文詳細

// ── サブリソース（親子関係） ──
// GET /api/v1/users/usr_def456/orders          ユーザーの注文一覧
// GET /api/v1/orders/ord_ghi789/items          注文の商品一覧
// GET /api/v1/products/prod_abc123/reviews     商品のレビュー一覧

// ── アクションリソース（動詞的操作、RESTの例外） ──
// POST /api/v1/orders/ord_ghi789/cancel        注文キャンセル
// POST /api/v1/orders/ord_ghi789/refund        返金処理
// POST /api/v1/users/usr_def456/verify-email   メール認証

// ── シングルトンリソース ──
// GET  /api/v1/users/me                        現在のユーザー
// GET  /api/v1/settings                        アプリ設定
// GET  /api/v1/users/usr_def456/cart            ユーザーのカート（1つ）

// ── 検索リソース ──
// GET  /api/v1/search?q=laptop&category=electronics  全文検索
// POST /api/v1/products/search                        複雑な検索（ボディ使用）
```

### 2.3 リソースID設計

リソースの識別子の選択はセキュリティと運用の両面で重要である。

| ID方式 | 例 | メリット | デメリット |
|--------|-----|---------|-----------|
| 連番（auto-increment） | `123` | シンプル、ソート可能 | 推測可能、総数が漏洩 |
| UUID v4 | `550e8400-e29b-41d4-a716-446655440000` | 推測不可能、分散生成 | 長い、インデックス効率低 |
| UUID v7 | `018e4a8c-1234-7abc-8def-0123456789ab` | 時系列ソート可能、推測不可 | 比較的新しい規格 |
| プレフィックス付きID | `usr_abc123`, `ord_def456` | 型が一目でわかる、推測不可 | 独自実装が必要 |
| Snowflake ID | `1234567890123456789` | 時系列、分散生成、高性能 | 64bit整数の範囲 |

```javascript
// プレフィックス付きID生成の実装例
const crypto = require('crypto');

const ID_PREFIXES = {
  user: 'usr',
  order: 'ord',
  product: 'prod',
  payment: 'pay',
  review: 'rev',
};

function generateId(resourceType) {
  const prefix = ID_PREFIXES[resourceType];
  if (!prefix) {
    throw new Error(`Unknown resource type: ${resourceType}`);
  }
  // 16バイトのランダム文字列を生成
  const random = crypto.randomBytes(16).toString('base64url');
  return `${prefix}_${random}`;
}

// 使用例
console.log(generateId('user'));    // usr_Ab3dEfGhIjKlMnOpQrSt0w
console.log(generateId('order'));   // ord_Xy9ZaBcDeFgHiJkLmNoPq2
console.log(generateId('product'));// prod_Rs4TuVwXyZaBcDeFgHiJk1
```

---

## 3. HTTPメソッドの正しい使い方

### 3.1 メソッド一覧と特性

```
┌─────────┬──────────────────────────────────────────────────────────┐
│ メソッド │ 意味と使い方                                             │
├─────────┼──────────────────────────────────────────────────────────┤
│ GET     │ リソースの取得                                           │
│         │ ・副作用なし（Safe）                                      │
│         │ ・冪等（Idempotent）                                     │
│         │ ・キャッシュ可能                                          │
│         │ ・リクエストボディなし                                     │
├─────────┼──────────────────────────────────────────────────────────┤
│ POST    │ リソースの作成 / アクション実行                            │
│         │ ・非冪等（2回実行 = 2つ作成）                              │
│         │ ・キャッシュ不可                                          │
│         │ ・成功時 201 Created + Location ヘッダー                  │
├─────────┼──────────────────────────────────────────────────────────┤
│ PUT     │ リソースの完全置換                                        │
│         │ ・冪等（同じリクエストを何回送っても同じ結果）              │
│         │ ・リソースが存在しなければ作成（upsert的）                 │
│         │ ・全フィールドを送信                                      │
├─────────┼──────────────────────────────────────────────────────────┤
│ PATCH   │ リソースの部分更新                                        │
│         │ ・変更フィールドのみ送信                                   │
│         │ ・冪等性は実装依存（相対的変更は非冪等）                   │
├─────────┼──────────────────────────────────────────────────────────┤
│ DELETE  │ リソースの削除                                            │
│         │ ・冪等（削除済みなら何もしない or 404）                    │
│         │ ・成功時 204 No Content（ボディなし）                     │
├─────────┼──────────────────────────────────────────────────────────┤
│ HEAD    │ GETと同じだがボディなし（メタデータ確認用）                │
│         │ ・リソースの存在確認                                      │
│         │ ・Content-Length の確認                                   │
├─────────┼──────────────────────────────────────────────────────────┤
│ OPTIONS │ 対応メソッドの確認                                        │
│         │ ・CORS プリフライトリクエスト                              │
│         │ ・Allow ヘッダーで対応メソッド返却                        │
└─────────┴──────────────────────────────────────────────────────────┘
```

### 3.2 メソッド特性の比較表

| 特性 | GET | POST | PUT | PATCH | DELETE | HEAD | OPTIONS |
|------|-----|------|-----|-------|--------|------|---------|
| Safe（安全） | Yes | No | No | No | No | Yes | Yes |
| Idempotent（冪等） | Yes | No | Yes | Impl. | Yes | Yes | Yes |
| Cacheable | Yes | No | No | No | No | Yes | No |
| Request Body | No | Yes | Yes | Yes | Optional | No | No |
| 成功時の典型コード | 200 | 201 | 200 | 200 | 204 | 200 | 200 |

### 3.3 CRUD操作の完全な実装例

```javascript
const express = require('express');
const router = express.Router();

// ── GET: コレクション取得（一覧） ──
router.get('/api/v1/products', async (req, res) => {
  const {
    page = 1,
    limit = 20,
    sort = '-created_at',
    category,
    min_price,
    max_price,
    q, // 検索クエリ
  } = req.query;

  const filters = {};
  if (category) filters.category = category;
  if (min_price) filters.price = { $gte: Number(min_price) };
  if (max_price) filters.price = { ...filters.price, $lte: Number(max_price) };

  const [products, total] = await Promise.all([
    Product.find(filters)
      .sort(parseSortParam(sort))
      .skip((page - 1) * limit)
      .limit(Number(limit)),
    Product.countDocuments(filters),
  ]);

  res.status(200).json({
    data: products.map(serializeProduct),
    meta: {
      page: Number(page),
      limit: Number(limit),
      total,
      total_pages: Math.ceil(total / limit),
    },
    links: {
      self: `/api/v1/products?page=${page}&limit=${limit}`,
      first: `/api/v1/products?page=1&limit=${limit}`,
      last: `/api/v1/products?page=${Math.ceil(total / limit)}&limit=${limit}`,
      ...(page > 1 && {
        prev: `/api/v1/products?page=${page - 1}&limit=${limit}`,
      }),
      ...(page < Math.ceil(total / limit) && {
        next: `/api/v1/products?page=${Number(page) + 1}&limit=${limit}`,
      }),
    },
  });
});

// ── GET: 個別リソース取得 ──
router.get('/api/v1/products/:id', async (req, res) => {
  const product = await Product.findById(req.params.id);

  if (!product) {
    return res.status(404).json({
      type: 'https://api.example.com/errors/not-found',
      title: 'Product Not Found',
      status: 404,
      detail: `Product with id '${req.params.id}' does not exist.`,
      instance: `/api/v1/products/${req.params.id}`,
    });
  }

  res.status(200)
    .set('ETag', `"${product.version}"`)
    .set('Last-Modified', product.updated_at.toUTCString())
    .set('Cache-Control', 'private, max-age=60')
    .json({
      data: serializeProduct(product),
      links: {
        self: { href: `/api/v1/products/${product.id}` },
        reviews: { href: `/api/v1/products/${product.id}/reviews` },
        category: { href: `/api/v1/categories/${product.category_id}` },
      },
    });
});

// ── POST: リソース作成 ──
router.post('/api/v1/products', authenticate, authorize('admin'), async (req, res) => {
  // バリデーション
  const { error, value } = productSchema.validate(req.body);
  if (error) {
    return res.status(422).json({
      type: 'https://api.example.com/errors/validation',
      title: 'Validation Error',
      status: 422,
      detail: 'One or more fields failed validation.',
      errors: error.details.map(d => ({
        field: d.path.join('.'),
        message: d.message,
        code: 'INVALID_VALUE',
      })),
    });
  }

  const product = await Product.create({
    ...value,
    id: generateId('product'),
    created_by: req.user.id,
  });

  res.status(201)
    .set('Location', `/api/v1/products/${product.id}`)
    .json({
      data: serializeProduct(product),
      links: {
        self: { href: `/api/v1/products/${product.id}` },
      },
    });
});

// ── PUT: リソース完全置換 ──
router.put('/api/v1/products/:id', authenticate, authorize('admin'), async (req, res) => {
  const { error, value } = productSchema.validate(req.body);
  if (error) {
    return res.status(422).json({
      type: 'https://api.example.com/errors/validation',
      title: 'Validation Error',
      status: 422,
      errors: error.details.map(d => ({
        field: d.path.join('.'),
        message: d.message,
      })),
    });
  }

  // 楽観的ロック
  const ifMatch = req.headers['if-match'];
  const existing = await Product.findById(req.params.id);

  if (!existing) {
    return res.status(404).json({
      type: 'https://api.example.com/errors/not-found',
      title: 'Product Not Found',
      status: 404,
    });
  }

  if (ifMatch && ifMatch !== `"${existing.version}"`) {
    return res.status(412).json({
      type: 'https://api.example.com/errors/precondition-failed',
      title: 'Precondition Failed',
      status: 412,
      detail: 'The resource has been modified since your last request.',
      currentETag: `"${existing.version}"`,
    });
  }

  const updated = await Product.findByIdAndUpdate(
    req.params.id,
    { ...value, version: existing.version + 1 },
    { new: true }
  );

  res.status(200)
    .set('ETag', `"${updated.version}"`)
    .json({ data: serializeProduct(updated) });
});

// ── DELETE: リソース削除 ──
router.delete('/api/v1/products/:id', authenticate, authorize('admin'), async (req, res) => {
  const product = await Product.findById(req.params.id);

  if (!product) {
    return res.status(404).json({
      type: 'https://api.example.com/errors/not-found',
      title: 'Product Not Found',
      status: 404,
    });
  }

  // 論理削除（推奨）
  await Product.findByIdAndUpdate(req.params.id, {
    deleted_at: new Date(),
    deleted_by: req.user.id,
  });

  // 物理削除の場合: await Product.findByIdAndDelete(req.params.id);

  res.status(204).end();
});
```

---

## 4. ステータスコード完全ガイド

HTTPステータスコードはAPIの「語彙」である。適切なコードを返すことで、クライアントは追加情報なしにレスポンスの意味を把握できる。

### 4.1 ステータスコード一覧と使用場面

```
┌─────┬──────────────────────────┬────────────────────────────────────┐
│コード│ 名称                     │ 使用場面                           │
├─────┼──────────────────────────┼────────────────────────────────────┤
│     │ === 2xx 成功 ===         │                                    │
│ 200 │ OK                       │ GET/PUT/PATCH の成功               │
│ 201 │ Created                  │ POST によるリソース作成成功         │
│ 202 │ Accepted                 │ 非同期処理の受付完了                │
│ 204 │ No Content               │ DELETE 成功（ボディなし）          │
├─────┼──────────────────────────┼────────────────────────────────────┤
│     │ === 3xx リダイレクト ===  │                                    │
│ 301 │ Moved Permanently        │ リソースの恒久移動                  │
│ 302 │ Found                    │ 一時的リダイレクト                  │
│ 304 │ Not Modified             │ キャッシュが有効（条件付きGET）     │
│ 307 │ Temporary Redirect       │ メソッドを維持したリダイレクト       │
│ 308 │ Permanent Redirect       │ メソッドを維持した恒久リダイレクト   │
├─────┼──────────────────────────┼────────────────────────────────────┤
│     │ === 4xx クライアントエラー│                                    │
│ 400 │ Bad Request              │ リクエスト構文エラー                │
│ 401 │ Unauthorized             │ 認証が必要（未認証）                │
│ 403 │ Forbidden                │ 認可エラー（権限不足）              │
│ 404 │ Not Found                │ リソースが存在しない                │
│ 405 │ Method Not Allowed       │ 許可されていないHTTPメソッド        │
│ 406 │ Not Acceptable           │ Accept ヘッダーの形式に非対応       │
│ 409 │ Conflict                 │ リソースの競合（楽観ロック等）       │
│ 410 │ Gone                     │ リソースが恒久的に削除済み          │
│ 412 │ Precondition Failed      │ If-Match 等の前提条件不一致         │
│ 415 │ Unsupported Media Type   │ Content-Type 非対応                │
│ 422 │ Unprocessable Entity     │ バリデーションエラー                │
│ 429 │ Too Many Requests        │ レート制限超過                      │
├─────┼──────────────────────────┼────────────────────────────────────┤
│     │ === 5xx サーバーエラー === │                                    │
│ 500 │ Internal Server Error    │ サーバー内部エラー（汎用）          │
│ 502 │ Bad Gateway              │ 上流サーバーからの不正レスポンス    │
│ 503 │ Service Unavailable      │ サービス一時停止（メンテナンス等）  │
│ 504 │ Gateway Timeout          │ 上流サーバーのタイムアウト          │
└─────┴──────────────────────────┴────────────────────────────────────┘
```

### 4.2 よくある誤用と正しい選択

| 場面 | よくある誤用 | 正しいコード | 理由 |
|------|-------------|-------------|------|
| ログイン失敗 | 403 | **401** | 認証（Authentication）の失敗は401。403は認可（Authorization）の失敗 |
| バリデーションエラー | 400 | **422** | 構文は正しいが意味的に不正。400は構文エラー |
| リソースが既に存在 | 400 | **409** | 状態の競合を表すのが409 |
| レート制限 | 503 | **429** | クライアント側の問題であり4xx。503はサーバー側 |
| 非同期処理の受付 | 200 | **202** | 処理はまだ完了していないため200は不適切 |
| DELETEで既に削除済み | 404 | **204 or 404** | どちらも正当。冪等性を重視するなら204、厳密さなら404 |

### 4.3 ステータスコード選択のフローチャート

```
  リクエスト受信
       │
       ▼
  構文は正しい？ ─── No ──→ 400 Bad Request
       │
      Yes
       │
       ▼
  認証済み？ ─── No ──→ 401 Unauthorized
       │
      Yes
       │
       ▼
  権限あり？ ─── No ──→ 403 Forbidden
       │
      Yes
       │
       ▼
  リソースは存在？ ─── No ──→ 404 Not Found
       │                       (POST の場合は通過)
      Yes
       │
       ▼
  バリデーション通過？ ─── No ──→ 422 Unprocessable Entity
       │
      Yes
       │
       ▼
  競合なし？ ─── No ──→ 409 Conflict
       │
      Yes
       │
       ▼
  処理成功？ ─── No ──→ 500 Internal Server Error
       │
      Yes
       │
       ▼
  メソッドに応じたレスポンス:
    GET/PUT/PATCH → 200 OK
    POST          → 201 Created
    DELETE        → 204 No Content
    非同期        → 202 Accepted
```

---

## 5. エラーレスポンス設計（RFC 9457）

### 5.1 Problem Details for HTTP APIs

RFC 9457（旧RFC 7807）は、HTTP APIのエラーレスポンスを標準化する仕様である。これにより、異なるAPI間でエラー処理のコードを共通化できる。

```javascript
// RFC 9457 準拠のエラーレスポンス構造

// 基本構造
{
  "type": "https://api.example.com/errors/validation",     // エラー種別のURI
  "title": "Validation Error",                              // 人間可読なタイトル
  "status": 422,                                            // HTTPステータスコード
  "detail": "The 'email' field is not a valid email address.", // 具体的な説明
  "instance": "/api/v1/users"                               // エラーが発生したリクエストURI
}

// 拡張フィールド付き（バリデーションエラー）
{
  "type": "https://api.example.com/errors/validation",
  "title": "Validation Error",
  "status": 422,
  "detail": "One or more fields failed validation.",
  "instance": "/api/v1/users",
  "errors": [
    {
      "field": "email",
      "message": "Must be a valid email address.",
      "code": "INVALID_FORMAT",
      "rejected_value": "not-an-email"
    },
    {
      "field": "age",
      "message": "Must be between 0 and 150.",
      "code": "OUT_OF_RANGE",
      "rejected_value": -5
    }
  ]
}

// レート制限エラー
{
  "type": "https://api.example.com/errors/rate-limit",
  "title": "Rate Limit Exceeded",
  "status": 429,
  "detail": "You have exceeded the rate limit of 100 requests per minute.",
  "instance": "/api/v1/products",
  "retry_after": 30,
  "limit": 100,
  "remaining": 0,
  "reset": "2025-01-15T10:30:00Z"
}
```

### 5.2 エラーレスポンスの実装

```javascript
// エラーハンドリングのミドルウェア実装

class ApiError extends Error {
  constructor(type, title, status, detail, extensions = {}) {
    super(detail);
    this.type = type;
    this.title = title;
    this.status = status;
    this.detail = detail;
    this.extensions = extensions;
  }

  toJSON() {
    return {
      type: this.type,
      title: this.title,
      status: this.status,
      detail: this.detail,
      ...this.extensions,
    };
  }
}

// 定義済みエラーファクトリ
const Errors = {
  notFound: (resource, id) =>
    new ApiError(
      'https://api.example.com/errors/not-found',
      `${resource} Not Found`,
      404,
      `${resource} with id '${id}' does not exist.`
    ),

  validation: (errors) =>
    new ApiError(
      'https://api.example.com/errors/validation',
      'Validation Error',
      422,
      'One or more fields failed validation.',
      { errors }
    ),

  unauthorized: () =>
    new ApiError(
      'https://api.example.com/errors/unauthorized',
      'Unauthorized',
      401,
      'Authentication is required to access this resource.'
    ),

  forbidden: () =>
    new ApiError(
      'https://api.example.com/errors/forbidden',
      'Forbidden',
      403,
      'You do not have permission to access this resource.'
    ),

  conflict: (detail) =>
    new ApiError(
      'https://api.example.com/errors/conflict',
      'Resource Conflict',
      409,
      detail
    ),

  rateLimited: (retryAfter, limit) =>
    new ApiError(
      'https://api.example.com/errors/rate-limit',
      'Rate Limit Exceeded',
      429,
      `Rate limit of ${limit} requests exceeded.`,
      { retry_after: retryAfter, limit }
    ),
};

// Express グローバルエラーハンドラー
function errorHandler(err, req, res, next) {
  if (err instanceof ApiError) {
    return res
      .status(err.status)
      .set('Content-Type', 'application/problem+json')
      .json({
        ...err.toJSON(),
        instance: req.originalUrl,
      });
  }

  // 予期しないエラー
  console.error('Unhandled error:', err);
  res
    .status(500)
    .set('Content-Type', 'application/problem+json')
    .json({
      type: 'https://api.example.com/errors/internal',
      title: 'Internal Server Error',
      status: 500,
      detail: 'An unexpected error occurred. Please try again later.',
      instance: req.originalUrl,
    });
}

// 使用例
router.get('/api/v1/users/:id', async (req, res) => {
  const user = await User.findById(req.params.id);
  if (!user) throw Errors.notFound('User', req.params.id);
  res.json({ data: user });
});
```

---

## 6. HATEOAS（深掘り）

### 6.1 HATEOASの本質

HATEOAS（Hypermedia As The Engine Of Application State）は REST の Uniform Interface の最も重要な構成要素である。クライアントはAPIのURL構造を事前に知る必要がなく、レスポンスに含まれるリンクだけでAPIを操作できる。

```
HATEOAS = Hypermedia As The Engine Of Application State

  ┌───────────────────────────────────────────────────────┐
  │            HATEOAS の動作モデル                        │
  │                                                       │
  │  クライアント         サーバー                         │
  │      │                   │                            │
  │      │  GET /api/v1/     │                            │
  │      │ ───────────────→  │                            │
  │      │                   │  ルートリソース:            │
  │      │  ←─────────────── │  { links: {                │
  │      │                   │      users: "/api/v1/users"│
  │      │                   │      products: "..."       │
  │      │                   │    }                       │
  │      │                   │  }                         │
  │      │                   │                            │
  │      │  リンクを辿る      │                            │
  │      │  GET /api/v1/users│                            │
  │      │ ───────────────→  │                            │
  │      │                   │  ユーザー一覧:             │
  │      │  ←─────────────── │  { data: [...],            │
  │      │                   │    links: {                │
  │      │                   │      self, next, prev,     │
  │      │                   │      create: { method:POST}│
  │      │                   │    }                       │
  │      │                   │  }                         │
  │      │                   │                            │
  │      │  さらにリンクを辿る│                            │
  │      │  ...              │                            │
  └───────────────────────────────────────────────────────┘

  重要な概念:
  → クライアントは最初のURL（エントリポイント）だけ知っていればよい
  → 以降の操作は全てレスポンスのリンクから発見する
  → APIの構造変更（URL変更）がクライアントに影響しない
```

### 6.2 状態遷移とリンクの動的変化

HATEOASの核心は、リソースの状態に応じて利用可能なアクション（リンク）が変化する点にある。

```javascript
// 注文リソースの状態遷移に応じたHATEOASリンク

function buildOrderLinks(order) {
  const base = `/api/v1/orders/${order.id}`;
  const links = {
    self: { href: base, method: 'GET' },
    items: { href: `${base}/items`, method: 'GET' },
  };

  switch (order.status) {
    case 'draft':
      links.submit = { href: `${base}/submit`, method: 'POST' };
      links.update = { href: base, method: 'PUT' };
      links.delete = { href: base, method: 'DELETE' };
      break;

    case 'pending':
      links.pay = { href: `${base}/pay`, method: 'POST' };
      links.cancel = { href: `${base}/cancel`, method: 'POST' };
      break;

    case 'paid':
      links.ship = { href: `${base}/ship`, method: 'POST' };
      links.refund = { href: `${base}/refund`, method: 'POST' };
      links.invoice = { href: `${base}/invoice`, method: 'GET' };
      break;

    case 'shipped':
      links.track = { href: `${base}/tracking`, method: 'GET' };
      links.return_request = { href: `${base}/return`, method: 'POST' };
      break;

    case 'delivered':
      links.review = { href: `${base}/review`, method: 'POST' };
      links.return_request = { href: `${base}/return`, method: 'POST' };
      break;

    case 'cancelled':
    case 'refunded':
      // 終端状態: 追加アクションなし
      break;
  }

  return links;
}

// レスポンス例: status = 'pending'
// {
//   "data": {
//     "id": "ord_abc123",
//     "status": "pending",
//     "total": 5000,
//     "items": [...]
//   },
//   "links": {
//     "self":   { "href": "/api/v1/orders/ord_abc123", "method": "GET" },
//     "items":  { "href": "/api/v1/orders/ord_abc123/items", "method": "GET" },
//     "pay":    { "href": "/api/v1/orders/ord_abc123/pay", "method": "POST" },
//     "cancel": { "href": "/api/v1/orders/ord_abc123/cancel", "method": "POST" }
//   }
// }
```

### 6.3 HATEOASの現実的な採用レベル

```
  Level 0: 完全無視
    → URLをドキュメントで配布し、クライアントがハードコード
    → 最も一般的だが、APIの変更時に全クライアントの更新が必要

  Level 1: 関連リソースへのリンク
    → レスポンスに関連リソースの self リンクを含める
    → GitHub API, Stripe API がこのレベル
    → 推奨: 最低限この水準を目指す

  Level 2: アクションリンク付き
    → 利用可能なアクション（method付き）をリンクとして含める
    → PayPal API がこのレベル
    → 推奨: 状態遷移のあるリソースに採用

  Level 3: 完全なHATEOAS
    → クライアントはエントリポイントURLのみ知っていれば操作可能
    → HAL, JSON-LD, Siren 等のハイパーメディア形式を使用
    → 実装コストが高く、実務での採用例は限定的
```

---

## 7. 冪等性の設計

### 7.1 冪等性とは

冪等性（Idempotency）とは、同じ操作を何回実行しても結果が変わらない性質である。ネットワーク障害やリトライが発生するプロダクション環境で、データの一貫性を保つために不可欠な概念である。

```
冪等性（Idempotency）:
  → 同じリクエストを何回実行しても結果が同じ

  冪等なメソッド:
    GET     — 常に冪等（副作用なし）
    PUT     — 冪等（同じリソースを同じ状態に上書き）
    DELETE  — 冪等（削除済みなら何もしない / 404）
    HEAD    — 常に冪等
    OPTIONS — 常に冪等

  冪等でないメソッド:
    POST    — 2回実行すると2つリソースが作成される
    PATCH   — 相対的な変更の場合（例: { "op": "increment", "value": 1 }）
```

### 7.2 冪等性キー（Idempotency Key）

POSTリクエストに冪等性を持たせる仕組みがIdempotency Keyである。Stripe APIが採用し、業界標準となっている。

```javascript
// 冪等性キーのサーバー実装（Redis使用）
const Redis = require('ioredis');
const redis = new Redis();
const crypto = require('crypto');

// 冪等性ミドルウェア
async function idempotencyMiddleware(req, res, next) {
  const idempotencyKey = req.headers['idempotency-key'];

  // キーなしの場合は通常処理（冪等性なし）
  if (!idempotencyKey) return next();

  // キーのフォーマット検証（UUIDv4推奨）
  if (!/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(idempotencyKey)) {
    return res.status(400).json({
      type: 'https://api.example.com/errors/invalid-idempotency-key',
      title: 'Invalid Idempotency Key',
      status: 400,
      detail: 'Idempotency-Key must be a valid UUID v4.',
    });
  }

  const cacheKey = `idempotency:${req.method}:${req.path}:${idempotencyKey}`;

  // ロック取得（同じキーの同時処理を防ぐ）
  const lockKey = `${cacheKey}:lock`;
  const lockAcquired = await redis.set(lockKey, '1', 'EX', 60, 'NX');

  if (!lockAcquired) {
    return res.status(409).json({
      type: 'https://api.example.com/errors/concurrent-request',
      title: 'Concurrent Request',
      status: 409,
      detail: 'A request with this idempotency key is currently being processed.',
    });
  }

  try {
    // キャッシュ確認
    const cached = await redis.get(cacheKey);

    if (cached) {
      // 処理済み → キャッシュした結果をそのまま返す
      const { statusCode, headers, body } = JSON.parse(cached);
      Object.entries(headers).forEach(([key, value]) => res.set(key, value));
      return res.status(statusCode).json(body);
    }

    // リクエストボディのハッシュを保存（同じキーで異なるボディを検出）
    const bodyHash = crypto
      .createHash('sha256')
      .update(JSON.stringify(req.body))
      .digest('hex');

    // レスポンスをインターセプトして保存
    const originalJson = res.json.bind(res);
    res.json = async (body) => {
      const responseHeaders = {
        'content-type': res.get('content-type'),
        'location': res.get('location'),
      };

      await redis.setex(cacheKey, 86400, JSON.stringify({
        statusCode: res.statusCode,
        headers: responseHeaders,
        body,
        bodyHash,
      }));

      return originalJson(body);
    };

    next();
  } finally {
    // ロック解放
    await redis.del(lockKey);
  }
}

// 使用
app.post('/api/v1/payments', idempotencyMiddleware, paymentHandler);
app.post('/api/v1/orders', idempotencyMiddleware, orderHandler);
```

---

## 8. コンテンツネゴシエーション

### 8.1 基本概念

コンテンツネゴシエーションは、クライアントとサーバーが最適な表現形式を合意するプロセスである。

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
  (1) Accept ヘッダーベース（標準的・推奨）
  (2) 拡張子ベース: /users.json, /users.xml
  (3) クエリパラメータ: /users?format=csv

推奨:
  → デフォルトはJSON
  → 管理画面向けにCSVエクスポート対応
  → Accept ヘッダーで切り替え
```

### 8.2 コンテンツネゴシエーションの実装

```javascript
// Express でのコンテンツネゴシエーション実装

function contentNegotiation(formatters) {
  return (req, res, next) => {
    // req.accepts() は Express 組込みの Accept ヘッダー解析
    const format = req.accepts(Object.keys(formatters));

    if (!format) {
      return res.status(406).json({
        type: 'https://api.example.com/errors/not-acceptable',
        title: 'Not Acceptable',
        status: 406,
        detail: `Supported formats: ${Object.keys(formatters).join(', ')}`,
      });
    }

    // 選択されたフォーマッターを res に設定
    res.formatResponse = (data) => {
      const formatter = formatters[format];
      const { contentType, body } = formatter(data);
      res.set('Content-Type', contentType).send(body);
    };

    next();
  };
}

// フォーマッター定義
const userFormatters = {
  'application/json': (data) => ({
    contentType: 'application/json; charset=utf-8',
    body: JSON.stringify(data),
  }),
  'text/csv': (data) => ({
    contentType: 'text/csv; charset=utf-8',
    body: convertToCsv(data),
  }),
  'application/xml': (data) => ({
    contentType: 'application/xml; charset=utf-8',
    body: convertToXml(data),
  }),
};

// 使用
router.get(
  '/api/v1/users',
  contentNegotiation(userFormatters),
  async (req, res) => {
    const users = await User.find();
    res.formatResponse({ data: users });
  }
);
```

---

## 9. 部分更新（PATCH）の深掘り

PATCHメソッドはリソースの部分更新を行うHTTPメソッドである。PUTがリソースの完全置換を意味するのに対し、PATCHは変更が必要なフィールドのみを送信する。

### 9.1 PUT vs PATCH の比較

| 特性 | PUT | PATCH |
|------|-----|-------|
| 意味 | リソースの完全置換 | リソースの部分更新 |
| 送信するフィールド | 全フィールド | 変更フィールドのみ |
| 冪等性 | 常に冪等 | 実装依存 |
| 欠落フィールドの扱い | デフォルト値 or null | 変更なし |
| 帯域幅 | 大きい | 小さい |
| 用途 | 設定の完全上書き | プロフィール更新など |

### 9.2 Merge Patch（RFC 7396）

Merge Patchは最もシンプルな部分更新形式である。JSONオブジェクトをそのまま送信し、含まれるフィールドだけが更新される。

```javascript
// Merge Patch の実装

// リクエスト:
// PATCH /api/v1/users/usr_abc123
// Content-Type: application/merge-patch+json
// {
//   "name": "Updated Name",    ← 値を更新
//   "address": null,            ← フィールドを削除
//   // email は含めない         ← 変更なし
// }

function applyMergePatch(original, patch) {
  if (patch === null || typeof patch !== 'object' || Array.isArray(patch)) {
    return patch;
  }

  const result = { ...original };

  for (const [key, value] of Object.entries(patch)) {
    if (value === null) {
      // null → フィールドを削除
      delete result[key];
    } else if (typeof value === 'object' && !Array.isArray(value)) {
      // ネストされたオブジェクト → 再帰的にマージ
      result[key] = applyMergePatch(result[key] || {}, value);
    } else {
      // それ以外 → 値を設定
      result[key] = value;
    }
  }

  return result;
}

// Express ルート
router.patch('/api/v1/users/:id', authenticate, async (req, res) => {
  const user = await User.findById(req.params.id);
  if (!user) throw Errors.notFound('User', req.params.id);

  // バリデーション（部分スキーマで検証）
  const { error, value } = userPatchSchema.validate(req.body, {
    allowUnknown: false,
    stripUnknown: true,
  });
  if (error) throw Errors.validation(error.details);

  const updated = applyMergePatch(user.toObject(), value);
  const savedUser = await User.findByIdAndUpdate(req.params.id, updated, { new: true });

  res.status(200).json({ data: serializeUser(savedUser) });
});
```

#### Merge Patch の制限事項

```
Merge Patch の制限:

  1. 配列の部分更新ができない
     → 配列は常に全置換される
     → 例: tags: ["a", "b"] に "c" を追加したい場合
           tags: ["a", "b", "c"] を全て送る必要がある

  2. null値の設定と削除が区別できない
     → null = 「このフィールドを削除」と解釈される
     → null を値として設定したい場合に問題

  3. 空オブジェクトの扱い
     → {} は「変更なし」を意味する
     → 空オブジェクトを値として設定したい場合に問題

  これらの制限が問題になる場合 → JSON Patch を検討
```

### 9.3 JSON Patch（RFC 6902）

JSON Patchはより細かい操作が可能な部分更新形式であり、操作のリストとして表現される。

```javascript
// JSON Patch のリクエスト例:
// PATCH /api/v1/users/usr_abc123
// Content-Type: application/json-patch+json
// [
//   { "op": "replace", "path": "/name", "value": "Updated Name" },
//   { "op": "add", "path": "/tags/-", "value": "vip" },
//   { "op": "remove", "path": "/address" },
//   { "op": "move", "from": "/old_field", "path": "/new_field" },
//   { "op": "copy", "from": "/name", "path": "/display_name" },
//   { "op": "test", "path": "/version", "value": 5 }
// ]

// JSON Patch の操作一覧:
//   add     — フィールド追加 / 配列要素の挿入
//   remove  — フィールド削除 / 配列要素の削除
//   replace — フィールドの値を置換
//   move    — フィールドを移動（remove + add）
//   copy    — フィールドをコピー
//   test    — 値の検証（一致しなければ操作全体を中止）

// fast-json-patch ライブラリを使用した実装
const jsonPatch = require('fast-json-patch');

router.patch('/api/v1/users/:id', authenticate, async (req, res) => {
  const user = await User.findById(req.params.id);
  if (!user) throw Errors.notFound('User', req.params.id);

  const contentType = req.headers['content-type'];

  let updatedData;

  if (contentType === 'application/json-patch+json') {
    // JSON Patch
    const patchOps = req.body;

    // バリデーション
    const validationResult = jsonPatch.validate(patchOps, user.toObject());
    if (validationResult) {
      return res.status(422).json({
        type: 'https://api.example.com/errors/invalid-patch',
        title: 'Invalid JSON Patch',
        status: 422,
        detail: validationResult.message,
      });
    }

    // パッチ適用
    updatedData = jsonPatch.applyPatch(
      jsonPatch.deepClone(user.toObject()),
      patchOps
    ).newDocument;

  } else if (contentType === 'application/merge-patch+json') {
    // Merge Patch
    updatedData = applyMergePatch(user.toObject(), req.body);

  } else {
    return res.status(415).json({
      type: 'https://api.example.com/errors/unsupported-media-type',
      title: 'Unsupported Media Type',
      status: 415,
      detail: 'Use application/json-patch+json or application/merge-patch+json.',
    });
  }

  const savedUser = await User.findByIdAndUpdate(req.params.id, updatedData, { new: true });
  res.status(200).json({ data: serializeUser(savedUser) });
});
```

---

## 10. バルク操作

### 10.1 バルク操作の設計パターン

複数リソースを一括で操作する場合の設計パターンを示す。

```javascript
// バッチリクエストの完全な実装例

// ── バッチ作成 ──
// POST /api/v1/users/batch
router.post('/api/v1/users/batch', authenticate, authorize('admin'), async (req, res) => {
  const { operations } = req.body;

  // バッチサイズの制限
  if (!operations || operations.length === 0) {
    throw Errors.validation([{
      field: 'operations',
      message: 'At least one operation is required.',
      code: 'REQUIRED',
    }]);
  }

  if (operations.length > 100) {
    throw Errors.validation([{
      field: 'operations',
      message: 'Maximum 100 operations per batch.',
      code: 'MAX_EXCEEDED',
    }]);
  }

  const results = [];
  let succeeded = 0;
  let failed = 0;

  for (const op of operations) {
    try {
      // 個別バリデーション
      const { error, value } = userSchema.validate(op.body);
      if (error) {
        results.push({
          status: 422,
          error: {
            type: 'https://api.example.com/errors/validation',
            title: 'Validation Error',
            detail: error.details[0].message,
          },
        });
        failed++;
        continue;
      }

      // リソース作成
      const user = await User.create({
        ...value,
        id: generateId('user'),
        created_by: req.user.id,
      });

      results.push({
        status: 201,
        data: serializeUser(user),
      });
      succeeded++;

    } catch (err) {
      results.push({
        status: 500,
        error: {
          type: 'https://api.example.com/errors/internal',
          title: 'Internal Error',
          detail: 'Failed to process this operation.',
        },
      });
      failed++;
    }
  }

  // 全体のHTTPステータス:
  //   全成功 → 200
  //   部分失敗 → 207 Multi-Status
  //   全失敗 → 422
  const overallStatus = failed === 0 ? 200 : succeeded === 0 ? 422 : 207;

  res.status(overallStatus).json({
    results,
    meta: {
      total: operations.length,
      succeeded,
      failed,
    },
  });
});

// ── 一括削除 ──
// POST /api/v1/users/batch-delete
router.post('/api/v1/users/batch-delete', authenticate, authorize('admin'), async (req, res) => {
  const { ids } = req.body;

  if (!ids || ids.length === 0) {
    throw Errors.validation([{
      field: 'ids',
      message: 'At least one ID is required.',
    }]);
  }

  if (ids.length > 100) {
    throw Errors.validation([{
      field: 'ids',
      message: 'Maximum 100 IDs per batch delete.',
    }]);
  }

  const result = await User.updateMany(
    { id: { $in: ids } },
    { deleted_at: new Date(), deleted_by: req.user.id }
  );

  res.status(200).json({
    meta: {
      requested: ids.length,
      deleted: result.modifiedCount,
      not_found: ids.length - result.modifiedCount,
    },
  });
});

// ── 一括更新 ──
// PATCH /api/v1/users/batch
router.patch('/api/v1/users/batch', authenticate, authorize('admin'), async (req, res) => {
  const { ids, update } = req.body;

  if (!ids || ids.length === 0 || !update) {
    throw Errors.validation([{
      field: 'ids',
      message: 'IDs and update fields are required.',
    }]);
  }

  const result = await User.updateMany(
    { id: { $in: ids } },
    { $set: update }
  );

  res.status(200).json({
    meta: {
      requested: ids.length,
      updated: result.modifiedCount,
    },
  });
});
```

### 10.2 バルク操作の設計考慮事項

```
バルク操作の設計ポイント:

  1. トランザクション制御
     ┌─────────────────────────────────────────────────┐
     │  All-or-Nothing（トランザクション）              │
     │  → 1つでも失敗したら全てロールバック             │
     │  → データ一貫性が重要な場合（決済処理等）        │
     │  → パフォーマンスは低下するが安全                │
     ├─────────────────────────────────────────────────┤
     │  Partial Success（部分成功）                     │
     │  → 各操作を独立に処理し、個別に成功/失敗を返す  │
     │  → バルクインポート等に適する                    │
     │  → 207 Multi-Status で個別結果を返す            │
     └─────────────────────────────────────────────────┘

  2. サイズ制限
     → バッチサイズの上限を設定（例: 100件）
     → リクエストボディサイズの制限
     → タイムアウトの考慮

  3. 進捗通知（大量データの場合）
     → 非同期処理 + ポーリング
     → POST /api/v1/imports → 202 Accepted + Job ID
     → GET /api/v1/jobs/{job_id} → 進捗確認

  4. エラーレポート
     → 各操作のインデックスとエラー内容を返す
     → クライアントが再試行すべき操作を特定できるようにする
```

---

## 11. 楽観的ロック

### 11.1 ETag / If-Match による楽観的ロック

楽観的ロック（Optimistic Locking）は、同時更新の競合をHTTPヘッダーで検出する仕組みである。

```
楽観的ロックのフロー:

  Client A                Server               Client B
     │                       │                      │
     │  GET /users/123       │                      │
     │ ─────────────────────→│                      │
     │  200 OK               │                      │
     │  ETag: "v5"           │                      │
     │ ←─────────────────────│                      │
     │                       │  GET /users/123      │
     │                       │←─────────────────────│
     │                       │  200 OK              │
     │                       │  ETag: "v5"          │
     │                       │─────────────────────→│
     │                       │                      │
     │  PUT /users/123       │                      │
     │  If-Match: "v5"       │                      │
     │  { name: "Alice" }    │                      │
     │ ─────────────────────→│                      │
     │  200 OK               │                      │
     │  ETag: "v6"           │                      │
     │ ←─────────────────────│                      │
     │                       │                      │
     │                       │  PUT /users/123      │
     │                       │  If-Match: "v5"      │
     │                       │  { name: "Bob" }     │
     │                       │←─────────────────────│
     │                       │  412 Precondition    │
     │                       │  Failed              │
     │                       │  (v5 != v6)          │
     │                       │─────────────────────→│
     │                       │                      │
     │                       │  Client B は再取得   │
     │                       │  してリトライする     │
```

```javascript
// 楽観的ロックの完全な実装

router.put('/api/v1/users/:id', authenticate, async (req, res) => {
  const ifMatch = req.headers['if-match'];
  const user = await User.findById(req.params.id);

  if (!user) throw Errors.notFound('User', req.params.id);

  // If-Match ヘッダーが指定されている場合、ETag を検証
  if (ifMatch) {
    const currentETag = `"${user.version}"`;
    if (ifMatch !== currentETag) {
      return res.status(412).json({
        type: 'https://api.example.com/errors/precondition-failed',
        title: 'Precondition Failed',
        status: 412,
        detail: 'The resource has been modified since your last request. '
          + 'Please retrieve the latest version and retry.',
        current_etag: currentETag,
        your_etag: ifMatch,
      });
    }
  }

  // バリデーション
  const { error, value } = userSchema.validate(req.body);
  if (error) throw Errors.validation(error.details);

  // 更新（version をインクリメント）
  const updated = await User.findOneAndUpdate(
    { _id: req.params.id, version: user.version }, // version も条件に含める
    { ...value, version: user.version + 1 },
    { new: true }
  );

  // DB レベルでも競合検出（findOneAndUpdate が null を返す場合）
  if (!updated) {
    return res.status(409).json({
      type: 'https://api.example.com/errors/conflict',
      title: 'Resource Conflict',
      status: 409,
      detail: 'The resource was modified by another request during processing.',
    });
  }

  res.status(200)
    .set('ETag', `"${updated.version}"`)
    .json({ data: serializeUser(updated) });
});
```

---

## 12. レスポンス圧縮とキャッシュ戦略

### 12.1 圧縮

```
圧縮形式の比較:

  ┌──────────────────────────────────────────────────────┐
  │  形式     │  圧縮率   │  速度     │  ブラウザ対応     │
  ├───────────┼──────────┼──────────┼──────────────────┤
  │  gzip     │  良好     │  高速     │  ほぼ全て        │
  │  Brotli   │  最良     │  中速     │  主要ブラウザ     │
  │           │ (gzip比   │ (圧縮は   │  (IE除く)        │
  │           │  +20-30%) │  遅い)    │                  │
  │  zstd     │  最良     │  最速     │  限定的（新規格） │
  └──────────────────────────────────────────────────────┘

  推奨設定:
  → JSON APIレスポンス: Brotli（対応ブラウザ） or gzip（フォールバック）
  → 1KB未満のレスポンス: 圧縮不要（オーバーヘッドが効果を上回る）
  → 静的アセット: 事前圧縮（ビルド時に .br / .gz を生成）
```

### 12.2 キャッシュ戦略

```javascript
// Express でのキャッシュヘッダー設定

// 不変リソース（ビルドアセット、画像等）
app.use('/static', express.static('public', {
  maxAge: '365d',
  immutable: true,
  setHeaders: (res) => {
    res.set('Cache-Control', 'public, max-age=31536000, immutable');
  },
}));

// API レスポンスのキャッシュミドルウェア
function cacheControl(options = {}) {
  return (req, res, next) => {
    const {
      visibility = 'private',   // 'public' or 'private'
      maxAge = 0,               // 秒数
      sMaxAge,                  // CDN キャッシュ秒数
      mustRevalidate = false,
      noCache = false,
      noStore = false,
    } = options;

    const directives = [];

    if (noStore) {
      directives.push('no-store');
    } else if (noCache) {
      directives.push(`${visibility}`, 'no-cache');
    } else {
      directives.push(visibility);
      directives.push(`max-age=${maxAge}`);
      if (sMaxAge !== undefined) directives.push(`s-maxage=${sMaxAge}`);
      if (mustRevalidate) directives.push('must-revalidate');
    }

    res.set('Cache-Control', directives.join(', '));
    next();
  };
}

// 使用例
// 商品一覧: 60秒キャッシュ、CDNは300秒
router.get('/api/v1/products',
  cacheControl({ visibility: 'public', maxAge: 60, sMaxAge: 300 }),
  productListHandler
);

// ユーザー情報: キャッシュなし、毎回サーバー確認（ETagで304活用）
router.get('/api/v1/users/me',
  cacheControl({ visibility: 'private', noCache: true }),
  currentUserHandler
);

// 決済情報: キャッシュ禁止
router.get('/api/v1/payments/:id',
  cacheControl({ noStore: true }),
  paymentHandler
);

// 設定マスタ: 1時間キャッシュ
router.get('/api/v1/settings',
  cacheControl({ visibility: 'public', maxAge: 3600 }),
  settingsHandler
);
```

---

## 13. ページネーション設計

大量のリソース一覧を返すAPIにはページネーションが不可欠である。3つの主要なアプローチを比較する。

### 13.1 ページネーション方式の比較

| 方式 | パラメータ例 | メリット | デメリット |
|------|-------------|---------|-----------|
| オフセットベース | `?page=3&limit=20` | 実装が容易、任意ページにジャンプ可 | 大量データでSQLのOFFSETが遅い、挿入/削除時にずれる |
| カーソルベース | `?cursor=abc123&limit=20` | 高性能、大量データに強い、一貫性 | 任意ページジャンプ不可、ソート制約 |
| キーセットベース | `?after_id=123&limit=20` | カーソルと同等の性能、透過的 | ソート条件に制約 |

### 13.2 各方式の実装

```javascript
// ── オフセットベースのページネーション ──
router.get('/api/v1/products', async (req, res) => {
  const page = Math.max(1, parseInt(req.query.page) || 1);
  const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 20));
  const offset = (page - 1) * limit;

  const [products, total] = await Promise.all([
    db.query('SELECT * FROM products ORDER BY created_at DESC LIMIT $1 OFFSET $2', [limit, offset]),
    db.query('SELECT COUNT(*) FROM products'),
  ]);

  const totalPages = Math.ceil(total.rows[0].count / limit);

  res.json({
    data: products.rows,
    meta: { page, limit, total: parseInt(total.rows[0].count), total_pages: totalPages },
    links: {
      self: `/api/v1/products?page=${page}&limit=${limit}`,
      first: `/api/v1/products?page=1&limit=${limit}`,
      last: `/api/v1/products?page=${totalPages}&limit=${limit}`,
      ...(page > 1 && { prev: `/api/v1/products?page=${page - 1}&limit=${limit}` }),
      ...(page < totalPages && { next: `/api/v1/products?page=${page + 1}&limit=${limit}` }),
    },
  });
});

// ── カーソルベースのページネーション ──
router.get('/api/v1/products', async (req, res) => {
  const limit = Math.min(100, Math.max(1, parseInt(req.query.limit) || 20));
  const cursor = req.query.cursor; // Base64エンコードされたカーソル

  let query = 'SELECT * FROM products';
  const params = [limit + 1]; // 1つ多く取得（next の有無判定用）

  if (cursor) {
    const decoded = JSON.parse(Buffer.from(cursor, 'base64url').toString());
    query += ' WHERE (created_at, id) < ($2, $3)';
    params.push(decoded.created_at, decoded.id);
  }

  query += ' ORDER BY created_at DESC, id DESC LIMIT $1';

  const result = await db.query(query, params);
  const hasNext = result.rows.length > limit;
  const items = hasNext ? result.rows.slice(0, limit) : result.rows;

  const nextCursor = hasNext
    ? Buffer.from(JSON.stringify({
        created_at: items[items.length - 1].created_at,
        id: items[items.length - 1].id,
      })).toString('base64url')
    : null;

  res.json({
    data: items,
    meta: {
      limit,
      has_next: hasNext,
    },
    links: {
      self: `/api/v1/products?limit=${limit}${cursor ? `&cursor=${cursor}` : ''}`,
      ...(nextCursor && {
        next: `/api/v1/products?limit=${limit}&cursor=${nextCursor}`,
      }),
    },
  });
});
```

---

## 14. アンチパターン

### 14.1 アンチパターン1: 動詞ベースのURL設計

```
アンチパターン: 動詞ベースのURL

  悪い例（RPCスタイル）:
    POST /api/getUsers
    POST /api/createUser
    POST /api/deleteUser
    POST /api/updateUserEmail
    POST /api/searchProducts

  問題点:
  → 操作ごとに新しいエンドポイントが増殖
  → HTTPメソッドの意味が失われる（全てPOST）
  → キャッシュが効かない
  → 統一的なインターフェースが崩れる

  正しい設計（リソース指向）:
    GET    /api/v1/users                    ユーザー一覧
    POST   /api/v1/users                    ユーザー作成
    DELETE /api/v1/users/123                ユーザー削除
    PATCH  /api/v1/users/123                ユーザー更新
    GET    /api/v1/products?q=laptop        商品検索

  例外: 動詞が許容される場面
  → リソースに対する「アクション」を表す場合
  → POST /api/v1/orders/123/cancel
  → POST /api/v1/users/123/verify-email
  → これらは状態遷移であり、独立したリソースとして扱いにくい
```

### 14.2 アンチパターン2: レスポンスの不統一

```
アンチパターン: レスポンス構造が API ごとに異なる

  悪い例:
    GET /api/v1/users     → { "users": [...] }
    GET /api/v1/products  → { "data": [...], "count": 10 }
    GET /api/v1/orders    → [...] (配列そのまま)
    GET /api/v1/users/123 → { "id": "123", "name": "..." } (ラッパーなし)

  問題点:
  → クライアントが各エンドポイントの構造を個別に把握する必要がある
  → 共通のHTTPクライアントラッパーが書けない
  → メタデータ（ページネーション情報等）の配置場所が不統一

  正しい設計: 統一されたレスポンスエンベロープ

    // コレクション
    {
      "data": [...],
      "meta": { "page": 1, "limit": 20, "total": 150 },
      "links": { "self": "...", "next": "...", "prev": "..." }
    }

    // 個別リソース
    {
      "data": { "id": "123", "name": "..." },
      "links": { "self": "...", "related": "..." }
    }

    // エラー
    {
      "type": "https://api.example.com/errors/...",
      "title": "...",
      "status": 422,
      "detail": "..."
    }
```

### 14.3 アンチパターン3: 過度なネスト

```
アンチパターン: 深すぎるリソース階層

  悪い例:
    GET /api/v1/companies/1/departments/2/teams/3/members/4/tasks/5

  問題点:
  → URLが長くなり可読性が低下
  → 全ての親リソースのIDが必要（不必要に結合度が高い）
  → キャッシュの粒度が粗くなる
  → ルーティングの複雑化

  正しい設計: 最大2段のネスト + クエリパラメータ

    GET /api/v1/tasks/5                          タスクに直接アクセス
    GET /api/v1/tasks?team_id=3                  チームのタスク一覧
    GET /api/v1/tasks?member_id=4&status=active  メンバーのアクティブタスク

    // 親子関係が強い場合のみ1段のネスト
    GET /api/v1/orders/123/items                 注文の商品一覧
    GET /api/v1/users/456/notifications          ユーザーの通知一覧
```

---

## 15. エッジケース分析

### 15.1 エッジケース1: 論理削除と物理削除の共存

リソースの削除には「論理削除（ソフトデリート）」と「物理削除（ハードデリート）」がある。多くのプロダクション環境では論理削除が推奨されるが、GDPRなどの規制対応で物理削除が必要になる場合もある。

```javascript
// 論理削除と物理削除の共存設計

// ── 標準の DELETE: 論理削除 ──
// DELETE /api/v1/users/usr_abc123
// → deleted_at を設定、データは残存
router.delete('/api/v1/users/:id', authenticate, async (req, res) => {
  const user = await User.findById(req.params.id);
  if (!user) throw Errors.notFound('User', req.params.id);

  // 既に論理削除済みの場合
  if (user.deleted_at) {
    return res.status(410).json({
      type: 'https://api.example.com/errors/gone',
      title: 'Resource Gone',
      status: 410,
      detail: 'This resource has already been deleted.',
      deleted_at: user.deleted_at,
    });
  }

  await User.findByIdAndUpdate(req.params.id, {
    deleted_at: new Date(),
    deleted_by: req.user.id,
  });

  res.status(204).end();
});

// ── 物理削除（GDPR対応）: 別エンドポイント ──
// DELETE /api/v1/users/usr_abc123/permanent
// → データを完全に抹消
router.delete('/api/v1/users/:id/permanent',
  authenticate,
  authorize('admin'),
  requireMfa,  // MFA必須
  async (req, res) => {
    const user = await User.findById(req.params.id);
    if (!user) throw Errors.notFound('User', req.params.id);

    // 監査ログに記録
    await AuditLog.create({
      action: 'PERMANENT_DELETE',
      resource_type: 'User',
      resource_id: req.params.id,
      performed_by: req.user.id,
      reason: req.body.reason, // 削除理由（必須）
      timestamp: new Date(),
    });

    // 関連データの削除
    await Promise.all([
      Order.updateMany(
        { user_id: req.params.id },
        { $set: { user_id: null, user_name: '[deleted]' } }
      ),
      Review.deleteMany({ user_id: req.params.id }),
      User.findByIdAndDelete(req.params.id),
    ]);

    res.status(204).end();
  }
);

// ── 論理削除されたリソースの取得 ──
// GET /api/v1/users?include_deleted=true (管理者のみ)
router.get('/api/v1/users', authenticate, async (req, res) => {
  const filters = {};

  // 通常ユーザーは削除済みを見れない
  if (req.query.include_deleted === 'true' && req.user.role === 'admin') {
    // フィルタなし（削除済みも含む）
  } else {
    filters.deleted_at = null;
  }

  const users = await User.find(filters);
  res.json({ data: users.map(serializeUser) });
});

// ── 論理削除の復元 ──
// POST /api/v1/users/usr_abc123/restore
router.post('/api/v1/users/:id/restore',
  authenticate,
  authorize('admin'),
  async (req, res) => {
    const user = await User.findById(req.params.id);
    if (!user) throw Errors.notFound('User', req.params.id);

    if (!user.deleted_at) {
      return res.status(409).json({
        type: 'https://api.example.com/errors/conflict',
        title: 'Resource Not Deleted',
        status: 409,
        detail: 'This resource is not in a deleted state.',
      });
    }

    await User.findByIdAndUpdate(req.params.id, {
      $unset: { deleted_at: 1, deleted_by: 1 },
    });

    const restored = await User.findById(req.params.id);
    res.status(200).json({ data: serializeUser(restored) });
  }
);
```

### 15.2 エッジケース2: 非同期操作とポーリング

長時間かかる操作（大量データのインポート、レポート生成、画像処理など）を同期的に処理するとタイムアウトが発生する。非同期パターンで解決する。

```javascript
// 非同期操作のパターン

// ── Step 1: ジョブの開始 ──
// POST /api/v1/reports
// → 202 Accepted を即座に返し、バックグラウンドで処理
router.post('/api/v1/reports', authenticate, async (req, res) => {
  const { type, date_range, format } = req.body;

  // バリデーション
  const { error } = reportRequestSchema.validate(req.body);
  if (error) throw Errors.validation(error.details);

  // ジョブを作成
  const job = await Job.create({
    id: generateId('job'),
    type: 'report_generation',
    status: 'queued',
    params: { type, date_range, format },
    created_by: req.user.id,
    created_at: new Date(),
    progress: 0,
  });

  // キューに追加（実際の処理はワーカーが行う）
  await queue.add('generate-report', {
    jobId: job.id,
    ...req.body,
  });

  res.status(202)
    .set('Location', `/api/v1/jobs/${job.id}`)
    .json({
      data: {
        job_id: job.id,
        status: 'queued',
        message: 'Report generation has been queued.',
      },
      links: {
        status: { href: `/api/v1/jobs/${job.id}`, method: 'GET' },
        cancel: { href: `/api/v1/jobs/${job.id}/cancel`, method: 'POST' },
      },
    });
});

// ── Step 2: ジョブの進捗確認 ──
// GET /api/v1/jobs/job_abc123
router.get('/api/v1/jobs/:id', authenticate, async (req, res) => {
  const job = await Job.findById(req.params.id);
  if (!job) throw Errors.notFound('Job', req.params.id);

  const response = {
    data: {
      id: job.id,
      type: job.type,
      status: job.status,   // queued | processing | completed | failed | cancelled
      progress: job.progress, // 0-100
      created_at: job.created_at,
      updated_at: job.updated_at,
    },
    links: {
      self: { href: `/api/v1/jobs/${job.id}` },
    },
  };

  switch (job.status) {
    case 'queued':
    case 'processing':
      // ポーリング間隔を Retry-After で指示
      res.set('Retry-After', '5'); // 5秒後に再確認
      response.links.cancel = { href: `/api/v1/jobs/${job.id}/cancel`, method: 'POST' };
      break;

    case 'completed':
      response.data.result_url = job.result_url;
      response.links.result = { href: job.result_url, method: 'GET' };
      break;

    case 'failed':
      response.data.error = job.error_message;
      response.links.retry = { href: `/api/v1/reports`, method: 'POST' };
      break;
  }

  res.status(200).json(response);
});

// ── Step 3: ジョブのキャンセル ──
// POST /api/v1/jobs/job_abc123/cancel
router.post('/api/v1/jobs/:id/cancel', authenticate, async (req, res) => {
  const job = await Job.findById(req.params.id);
  if (!job) throw Errors.notFound('Job', req.params.id);

  if (['completed', 'failed', 'cancelled'].includes(job.status)) {
    return res.status(409).json({
      type: 'https://api.example.com/errors/conflict',
      title: 'Job Cannot Be Cancelled',
      status: 409,
      detail: `Job is already in '${job.status}' state.`,
    });
  }

  await Job.findByIdAndUpdate(req.params.id, {
    status: 'cancelled',
    cancelled_at: new Date(),
    cancelled_by: req.user.id,
  });

  res.status(200).json({
    data: { id: job.id, status: 'cancelled' },
  });
});
```

```
非同期操作のシーケンス図:

  Client                    API Server              Worker (Queue)
    │                          │                         │
    │  POST /api/v1/reports    │                         │
    │ ────────────────────────→│                         │
    │                          │  Job 作成 & キュー追加   │
    │                          │────────────────────────→│
    │  202 Accepted            │                         │
    │  Location: /jobs/abc     │                         │
    │ ←────────────────────────│                         │
    │                          │                         │
    │  (5秒後)                 │                         │
    │  GET /api/v1/jobs/abc    │         処理中...       │
    │ ────────────────────────→│                         │
    │  200 { status:processing │                         │
    │       progress: 45 }     │                         │
    │  Retry-After: 5          │                         │
    │ ←────────────────────────│                         │
    │                          │                         │
    │  (5秒後)                 │                         │
    │  GET /api/v1/jobs/abc    │    処理完了             │
    │ ────────────────────────→│←────────────────────────│
    │  200 { status:completed  │                         │
    │       result_url: "..." }│                         │
    │ ←────────────────────────│                         │
    │                          │                         │
    │  GET /results/abc.csv    │                         │
    │ ────────────────────────→│                         │
    │  200 (レポートファイル)   │                         │
    │ ←────────────────────────│                         │
```

---

## 16. レート制限

### 16.1 レート制限の実装

```javascript
// トークンバケットアルゴリズムによるレート制限

const Redis = require('ioredis');
const redis = new Redis();

async function rateLimitMiddleware(req, res, next) {
  const identifier = req.user?.id || req.ip; // 認証済みならユーザーID、未認証ならIP
  const key = `ratelimit:${identifier}`;

  // 設定
  const limit = req.user ? 1000 : 100;  // 認証済み: 1000/分, 未認証: 100/分
  const window = 60; // 60秒

  // Redisでカウント
  const multi = redis.multi();
  multi.incr(key);
  multi.ttl(key);
  const [[, count], [, ttl]] = await multi.exec();

  // 初回アクセス時にTTLを設定
  if (ttl === -1) {
    await redis.expire(key, window);
  }

  const remaining = Math.max(0, limit - count);
  const resetTime = new Date(Date.now() + (ttl > 0 ? ttl : window) * 1000);

  // レート制限ヘッダーを設定（RFC 6585 / Draft RateLimit Header）
  res.set({
    'RateLimit-Limit': limit.toString(),
    'RateLimit-Remaining': remaining.toString(),
    'RateLimit-Reset': Math.ceil(resetTime.getTime() / 1000).toString(),
    'RateLimit-Policy': `${limit};w=${window}`,
  });

  if (count > limit) {
    const retryAfter = ttl > 0 ? ttl : window;
    res.set('Retry-After', retryAfter.toString());

    return res.status(429).json({
      type: 'https://api.example.com/errors/rate-limit',
      title: 'Rate Limit Exceeded',
      status: 429,
      detail: `You have exceeded the limit of ${limit} requests per ${window} seconds.`,
      retry_after: retryAfter,
      limit,
      remaining: 0,
      reset: resetTime.toISOString(),
    });
  }

  next();
}

app.use('/api/', rateLimitMiddleware);
```

---

## 17. APIバージョニング

APIのバージョニングは、既存クライアントの互換性を維持しながらAPIを進化させるための仕組みである。

```
バージョニング戦略の比較:

  ┌──────────────────┬──────────────────────────────────────────┐
  │  方式            │  特徴                                     │
  ├──────────────────┼──────────────────────────────────────────┤
  │  URI パス         │  /api/v1/users, /api/v2/users            │
  │                  │  最も一般的で分かりやすい                  │
  │                  │  ルーティングが明確                        │
  │                  │  キャッシュが効きやすい                    │
  │                  │  (GitHub, Stripe, Google 等が採用)        │
  ├──────────────────┼──────────────────────────────────────────┤
  │  クエリパラメータ │  /api/users?version=2                    │
  │                  │  既存URLへの影響が小さい                   │
  │                  │  パラメータ省略時のデフォルト設定が必要     │
  │                  │  (Google API の一部が採用)                │
  ├──────────────────┼──────────────────────────────────────────┤
  │  カスタムヘッダー │  X-API-Version: 2                        │
  │                  │  URIがクリーン                             │
  │                  │  ブラウザから直接テストしにくい             │
  │                  │  (Azure API Management 等)               │
  ├──────────────────┼──────────────────────────────────────────┤
  │  Accept ヘッダー │  Accept: application/vnd.myapi.v2+json   │
  │                  │  RESTの原則に最も忠実                      │
  │                  │  実装と利用が複雑                          │
  │                  │  (GitHub API v3 が採用)                   │
  └──────────────────┴──────────────────────────────────────────┘

  推奨:
  → URI パスが最も実用的（/api/v1/...）
  → 新バージョンは破壊的変更がある場合のみ作成
  → マイナー変更は同一バージョン内で後方互換に追加
```

```javascript
// バージョンルーティングの実装例

const express = require('express');
const app = express();

// バージョンごとにルーターを分離
const v1Router = require('./routes/v1');
const v2Router = require('./routes/v2');

app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

// バージョン非指定時のリダイレクト
app.use('/api/users', (req, res) => {
  // 最新安定版にリダイレクト
  res.redirect(307, `/api/v2${req.url}`);
});

// 非推奨バージョンの警告ヘッダー
function deprecationWarning(sunsetDate) {
  return (req, res, next) => {
    res.set({
      'Deprecation': 'true',
      'Sunset': sunsetDate,
      'Link': '</api/v2>; rel="successor-version"',
    });
    next();
  };
}

// v1 全体に非推奨警告を付与
v1Router.use(deprecationWarning('2025-12-31T23:59:59Z'));
```

---

## 18. 演習問題

### 18.1 演習1: 基礎（リソース設計）

**課題**: 図書館管理システムのREST APIを設計せよ。

以下の要件を満たすURIとHTTPメソッドの組み合わせを設計すること。

```
要件:
  - 書籍の管理（CRUD）
  - 著者の管理（CRUD）
  - 書籍の貸出・返却
  - 貸出履歴の照会
  - 書籍の検索（タイトル、著者名、ISBN）
  - 会員の管理（CRUD）
  - 会員ごとの貸出中リスト
  - 予約機能

模範解答:

  # 書籍
  GET    /api/v1/books                        書籍一覧
  GET    /api/v1/books?q=REST&author=fielding  書籍検索
  GET    /api/v1/books/:id                    書籍詳細
  POST   /api/v1/books                        書籍登録
  PUT    /api/v1/books/:id                    書籍更新（全体）
  PATCH  /api/v1/books/:id                    書籍更新（部分）
  DELETE /api/v1/books/:id                    書籍削除

  # 著者
  GET    /api/v1/authors                      著者一覧
  GET    /api/v1/authors/:id                  著者詳細
  GET    /api/v1/authors/:id/books            著者の書籍一覧
  POST   /api/v1/authors                      著者登録
  PUT    /api/v1/authors/:id                  著者更新
  DELETE /api/v1/authors/:id                  著者削除

  # 貸出
  POST   /api/v1/books/:id/checkout           書籍の貸出
  POST   /api/v1/books/:id/return             書籍の返却

  # 会員
  GET    /api/v1/members                      会員一覧
  GET    /api/v1/members/:id                  会員詳細
  GET    /api/v1/members/:id/checkouts        会員の貸出中リスト
  GET    /api/v1/members/:id/history          会員の貸出履歴
  POST   /api/v1/members                      会員登録

  # 予約
  POST   /api/v1/books/:id/reservations       予約作成
  DELETE /api/v1/books/:id/reservations/:rid   予約キャンセル
  GET    /api/v1/members/:id/reservations      会員の予約一覧
```

### 18.2 演習2: 中級（エラーハンドリングとステータスコード）

**課題**: 以下の各シナリオに対して、適切なHTTPステータスコードとRFC 9457準拠のエラーレスポンスボディを記述せよ。

```
シナリオ:

  A) ユーザーが無効なJSONをリクエストボディに送信した
  B) 認証トークンが期限切れ
  C) 一般ユーザーが管理者専用エンドポイントにアクセス
  D) 存在しないユーザーIDを指定してGETリクエスト
  E) メールアドレスが既に登録済みのユーザーを作成しようとした
  F) 注文の同時更新が発生した（楽観ロック競合）
  G) レート制限を超過
  H) 外部決済APIがタイムアウトした

模範解答:

  A) 400 Bad Request
  {
    "type": "https://api.example.com/errors/malformed-request",
    "title": "Malformed Request Body",
    "status": 400,
    "detail": "The request body contains invalid JSON. Unexpected token at position 42."
  }

  B) 401 Unauthorized
  {
    "type": "https://api.example.com/errors/token-expired",
    "title": "Authentication Token Expired",
    "status": 401,
    "detail": "Your authentication token has expired. Please re-authenticate."
  }

  C) 403 Forbidden
  {
    "type": "https://api.example.com/errors/insufficient-permissions",
    "title": "Insufficient Permissions",
    "status": 403,
    "detail": "You need 'admin' role to access this resource."
  }

  D) 404 Not Found
  {
    "type": "https://api.example.com/errors/not-found",
    "title": "User Not Found",
    "status": 404,
    "detail": "User with id 'usr_xyz789' does not exist."
  }

  E) 409 Conflict
  {
    "type": "https://api.example.com/errors/duplicate",
    "title": "Resource Already Exists",
    "status": 409,
    "detail": "A user with email 'test@example.com' already exists.",
    "conflicting_field": "email"
  }

  F) 412 Precondition Failed
  {
    "type": "https://api.example.com/errors/precondition-failed",
    "title": "Precondition Failed",
    "status": 412,
    "detail": "The order has been modified by another client.",
    "current_etag": "\"v8\"",
    "your_etag": "\"v7\""
  }

  G) 429 Too Many Requests
  {
    "type": "https://api.example.com/errors/rate-limit",
    "title": "Rate Limit Exceeded",
    "status": 429,
    "detail": "Rate limit of 100 requests per minute exceeded.",
    "retry_after": 23
  }

  H) 502 Bad Gateway
  {
    "type": "https://api.example.com/errors/upstream-timeout",
    "title": "Payment Service Timeout",
    "status": 502,
    "detail": "The payment service did not respond in time. Please retry."
  }
```

### 18.3 演習3: 上級（完全なAPI設計）

**課題**: タスク管理アプリケーションの REST API を設計せよ。以下の要件を全て満たすこと。

```
要件:
  - タスクのCRUD
  - タスクの状態遷移（todo → in_progress → review → done）
  - タスクのアサイン（担当者の設定）
  - プロジェクト単位のタスク管理
  - タスクへのコメント機能
  - タスクの一括ステータス変更
  - HATEOAS対応（状態に応じたリンク）
  - 楽観的ロック対応
  - ページネーション対応
  - 適切なエラーレスポンス

設計すべき項目:
  1. 全エンドポイントの一覧（URI + Method）
  2. 代表的なレスポンスのJSON（HATEOAS付き）
  3. 状態遷移図
  4. エラーケースの列挙

模範解答（抜粋）:

  # エンドポイント一覧
  GET    /api/v1/projects                          プロジェクト一覧
  POST   /api/v1/projects                          プロジェクト作成
  GET    /api/v1/projects/:pid                     プロジェクト詳細
  GET    /api/v1/projects/:pid/tasks               プロジェクトのタスク一覧
  POST   /api/v1/projects/:pid/tasks               タスク作成
  GET    /api/v1/tasks/:id                         タスク詳細
  PUT    /api/v1/tasks/:id                         タスク更新
  PATCH  /api/v1/tasks/:id                         タスク部分更新
  DELETE /api/v1/tasks/:id                         タスク削除
  POST   /api/v1/tasks/:id/transition              状態遷移
  PATCH  /api/v1/tasks/:id/assignee                アサイン変更
  GET    /api/v1/tasks/:id/comments                コメント一覧
  POST   /api/v1/tasks/:id/comments                コメント投稿
  PATCH  /api/v1/tasks/batch                       一括ステータス変更

  # 状態遷移図
       ┌──────┐     start      ┌─────────────┐
       │ todo │ ──────────────→│ in_progress │
       └──────┘                └─────────────┘
                                 │        │
                          submit │        │ return
                                 ▼        │
                              ┌────────┐  │
                              │ review │──┘
                              └────────┘
                                 │
                          approve│
                                 ▼
                              ┌──────┐
                              │ done │
                              └──────┘

  # タスク詳細のレスポンス（status = "in_progress"）
  {
    "data": {
      "id": "task_abc123",
      "title": "REST APIガイドの執筆",
      "description": "...",
      "status": "in_progress",
      "assignee": {
        "id": "usr_def456",
        "name": "田中太郎"
      },
      "project_id": "proj_ghi789",
      "version": 3,
      "created_at": "2025-01-10T09:00:00Z",
      "updated_at": "2025-01-12T14:30:00Z"
    },
    "links": {
      "self":       { "href": "/api/v1/tasks/task_abc123", "method": "GET" },
      "update":     { "href": "/api/v1/tasks/task_abc123", "method": "PUT" },
      "submit":     { "href": "/api/v1/tasks/task_abc123/transition",
                      "method": "POST", "body": { "to": "review" } },
      "comments":   { "href": "/api/v1/tasks/task_abc123/comments", "method": "GET" },
      "project":    { "href": "/api/v1/projects/proj_ghi789", "method": "GET" },
      "assignee":   { "href": "/api/v1/users/usr_def456", "method": "GET" }
    }
  }
```

---

## 19. FAQ

### Q1: RESTful APIで適切なHTTPステータスコードの選び方は？

**A**: ステータスコードは「何が起きたか」を明確に伝えるために選択する。以下の基準で判断する。

```
成功系:
  200 OK           → GET/PATCH成功（レスポンスボディあり）
  201 Created      → POST成功（新規リソース作成）
  204 No Content   → DELETE/PUT成功（レスポンスボディなし）

クライアントエラー系:
  400 Bad Request      → リクエスト構文エラー（JSON不正など）
  401 Unauthorized     → 認証が必要
  403 Forbidden        → 認証済みだが権限不足
  404 Not Found        → リソースが存在しない
  422 Unprocessable Entity → バリデーションエラー

サーバーエラー系:
  500 Internal Server Error → サーバー内部エラー
  503 Service Unavailable   → 一時的な過負荷
```

特に、バリデーションエラーには422を使い、400はリクエスト形式そのものの問題（JSONパースエラー等）に限定することで、クライアント側のエラー処理が明確になる。

### Q2: HATEOASは実際のプロジェクトで採用すべきか？

**A**: 完全なHATEOAS（Level 3）は実装コストが高いため、以下の段階的アプローチを推奨する。

```
Level 0（最低限）:
  → URIハードコードを避け、APIドキュメントで関連リソースのパスを明示

Level 1（推奨）:
  → レスポンスに関連リソースのURIを含める
  {
    "id": "usr_123",
    "name": "Alice",
    "orders_url": "/api/v1/users/usr_123/orders"
  }

Level 2（状態遷移が重要な場合）:
  → 実行可能なアクションのみをリンクとして返す
  {
    "id": "ord_456",
    "status": "pending",
    "links": {
      "cancel": { "href": "/api/v1/orders/ord_456/cancel", "method": "POST" },
      "pay": { "href": "/api/v1/orders/ord_456/payment", "method": "POST" }
    }
  }

Level 3（完全なHATEOAS）:
  → すべての状態遷移をハイパーメディアで表現
  → 大規模な公開APIや複雑なワークフローのみで採用
```

多くのプロジェクトではLevel 1で十分な価値が得られる。Level 3は、Stripe APIのような複雑な状態管理が必要な場合に限定すべきである。

### Q3: REST APIでのエラーレスポンスのベストプラクティスは？

**A**: RFC 9457（Problem Details for HTTP APIs）に準拠した構造を使うことが現代の標準である。

```javascript
// RFC 9457準拠のエラーレスポンス
{
  "type": "https://api.example.com/errors/validation-error",
  "title": "Validation Error",
  "status": 422,
  "detail": "The email field must be a valid email address.",
  "instance": "/api/v1/users",
  "errors": [
    {
      "field": "email",
      "code": "invalid_format",
      "message": "Must be a valid email address"
    }
  ],
  "trace_id": "abc123def456"  // デバッグ用の追跡ID
}
```

必須フィールド:
- `type`: エラー種別を示すURI（ドキュメントへのリンク）
- `title`: 人間が読める短いエラータイトル
- `status`: HTTPステータスコード
- `detail`: 具体的なエラー詳細
- `instance`: エラーが発生したリクエストパス

拡張フィールド:
- `errors`: バリデーションエラーの詳細（配列）
- `trace_id`: サーバーログとの紐付け用ID

このフォーマットにより、クライアント側で一貫したエラーハンドリングが可能になる。

### Q4: PUT と PATCH のどちらを使うべきか？

**A**: 実務では PATCH（Merge Patch）を基本とし、設定系のリソースでのみ PUT を使うのが最も実用的である。

理由は以下の通り:
- PUT は全フィールドの送信が必要なため、クライアントの実装負荷が高い
- フィールドの追加時に、PUT では全クライアントの更新が必要
- PATCH（Merge Patch）は変更フィールドのみ送信するため、帯域幅の節約にもなる
- ただし、設定ファイルのように「全体を一括で管理する」リソースでは PUT が適切

```
判断基準:
  「このリソースは部分的に更新されるか？」
    → Yes → PATCH（Merge Patch）
    → No（常に全体を置換） → PUT
```

### Q5: ネストが深いリソースはどう設計するか？

**A**: ネストは最大2段までとし、3段以上が必要な場合はクエリパラメータでフィルタリングする。

```
悪い例:  GET /companies/1/departments/2/teams/3/members
良い例:  GET /members?team_id=3
         GET /teams/3/members  （1段のネストは許容）
```

リソースが独立してアクセスされる場面があるなら、トップレベルのエンドポイントを提供すべきである。例えば、チームに属する「メンバー」は `/members/:id` でも直接アクセスできるようにする。

### Q6: 認証にはどの方式を採用すべきか？

**A**: マシン間通信（M2M）ではAPIキーまたはOAuth2 Client Credentials、ユーザー操作が伴う場合はOAuth2 Authorization Code + PKCE を推奨する。

```
用途別の推奨認証方式:

  ┌──────────────────┬────────────────────────────────┐
  │  用途            │  推奨方式                       │
  ├──────────────────┼────────────────────────────────┤
  │  SPA             │  OAuth2 Authorization Code     │
  │                  │  + PKCE + Secure Cookie        │
  ├──────────────────┼────────────────────────────────┤
  │  モバイルアプリ   │  OAuth2 Authorization Code     │
  │                  │  + PKCE                        │
  ├──────────────────┼────────────────────────────────┤
  │  サーバー間通信   │  OAuth2 Client Credentials     │
  │                  │  or API Key + Secret           │
  ├──────────────────┼────────────────────────────────┤
  │  管理画面        │  OAuth2 + MFA                  │
  ├──────────────────┼────────────────────────────────┤
  │  Webhook         │  HMAC署名検証                  │
  └──────────────────┴────────────────────────────────┘

  非推奨:
  → Basic認証（パスワードが毎回送信される）
  → JWT をローカルストレージに保存（XSS脆弱性）
  → APIキーのみ（ユーザー操作に使う場合）
```

### Q7: レスポンスのフィールド名はキャメルケースかスネークケースか？

**A**: JSON APIではスネークケース（`snake_case`）が推奨される。JavaScript のプロパティ名の慣習はキャメルケースだが、APIレベルでは以下の理由でスネークケースが優勢である。

- Google JSON Style Guide がスネークケースを推奨
- Ruby, Python, Go など多くのバックエンド言語の慣習と一致
- GitHub, Stripe, Twilio 等の主要APIがスネークケースを採用
- ただし、組織内で統一されていることが最も重要

### Q8: 空のレスポンスはどう返すべきか？

**A**: 空のコレクションは200 OKで空配列を返す。リソースが見つからない場合は404を返す。DELETEの成功は204 No Contentでボディなしとする。

```javascript
// 空のコレクション: 200 + 空配列（404ではない）
// GET /api/v1/users?status=vip （VIPユーザーがいない場合）
{
  "data": [],
  "meta": { "page": 1, "limit": 20, "total": 0, "total_pages": 0 }
}

// リソースが見つからない: 404
// GET /api/v1/users/nonexistent
{
  "type": "https://api.example.com/errors/not-found",
  "title": "User Not Found",
  "status": 404,
  "detail": "User with id 'nonexistent' does not exist."
}

// DELETE成功: 204 No Content（ボディなし）
// DELETE /api/v1/users/usr_abc123
// → 204 (空レスポンス)
```

---

## 20. まとめ

| 概念 | ポイント |
|------|---------|
| リソース設計 | 名詞・複数形・ケバブケース、ネストは2段まで |
| HTTPメソッド | GET=取得, POST=作成, PUT=完全置換, PATCH=部分更新, DELETE=削除 |
| ステータスコード | 201=作成成功, 204=ボディなし成功, 422=バリデーション, 429=レート制限 |
| エラーレスポンス | RFC 9457準拠: type, title, status, detail, instance |
| HATEOAS | 状態に応じたリンクの動的変化、最低でもLevel 1を目指す |
| 冪等性 | Idempotency-Keyで POST を冪等に、Stripe方式が業界標準 |
| PATCH | Merge Patch（シンプル）vs JSON Patch（高機能） |
| バルク操作 | 部分失敗を許容、207 Multi-Statusで個別結果を返す |
| 楽観ロック | ETag / If-Match で同時更新検出、412で競合通知 |
| ページネーション | 大量データにはカーソルベース、UI向けにはオフセットベース |
| バージョニング | URIパス方式（/api/v1/...）が最も実用的 |
| レート制限 | RateLimit-* ヘッダーで残数通知、429で制限超過 |

**キーポイント**:

1. **リソース指向設計**: URIは「動詞」ではなく「名詞」で表現し、HTTPメソッドで操作を示す。`POST /users` は OK、`POST /createUser` は NG。
2. **エラーレスポンスの標準化**: RFC 9457準拠のProblem Detailsフォーマットを採用することで、クライアント側のエラーハンドリングが一貫し、デバッグ効率が向上する。
3. **冪等性の保証**: 重要な操作（決済、リソース作成）には Idempotency-Key を導入し、ネットワークリトライによる重複実行を防止する。

---

## FAQ

### Q1: PATCHリクエストでJSON Merge PatchとJSON Patchのどちらを採用すべきか?
JSON Merge Patch（RFC 7396）はシンプルで直感的であり、一般的なフィールド更新に適している。JSONオブジェクトの部分更新をそのまま送信するだけでよく、学習コストが低い。一方、JSON Patch（RFC 6902）は配列操作（要素の追加・削除・移動）やフィールド名の変更など、より複雑な操作に対応できる。多くの場合はJSON Merge Patchで十分であり、配列の部分更新が頻繁に必要な場合にのみJSON Patchを検討するとよい。

### Q2: バルク操作のAPIでトランザクション保証は必要か?
基本的には部分失敗を許容する設計（207 Multi-Status）を推奨する。全件成功または全件失敗のトランザクション保証は、大量データ処理時にパフォーマンスのボトルネックとなり、分散システムでは実装が複雑になる。ただし、金融取引のように一貫性が必須のドメインでは、トランザクション保証が必要な場合もある。その場合は、バッチサイズに上限（例: 100件）を設け、処理時間の予測可能性を確保することが重要である。

### Q3: APIレスポンスでnullと未定義（フィールド省略）をどう使い分けるべきか?
明確なルールを決めてドキュメントに記載することが最も重要である。推奨アプローチとしては、nullは「値が明示的に空である」ことを意味し、フィールド省略は「そのリソースにはこの属性が存在しない」または「リクエストで指定されなかった」ことを意味すると定義する。PATCH操作ではこの区別が特に重要で、nullの送信は「値をクリアする」、フィールド省略は「変更しない」と解釈するのが一般的なパターンである。

## まとめ

このガイドでは以下を学びました:

- RESTの6原則に基づくリソース指向設計と、名詞・複数形・ケバブケースによるURI命名規則
- HTTPメソッド・ステータスコードの正しい使い方と、RFC 9457準拠のエラーレスポンス設計
- HATEOASによる状態遷移の表現と、Idempotency-Keyを用いた冪等性の保証
- 部分更新（PATCH）、バルク操作、楽観的ロック（ETag/If-Match）の実装パターン
- ページネーション設計、レート制限、バージョニングなどの運用に不可欠なAPI設計要素

---

## 次に読むべきガイド

- [GraphQL基礎](01-graphql-fundamentals.md) -- クエリ言語、スキーマ、リゾルバ
- [APIバージョニング戦略](02-api-versioning.md) -- バージョニング戦略の詳細
- [API認証・認可](03-authentication.md) -- 認証・認可の実装パターン

---

## 参考文献

1. Fielding, R. "Architectural Styles and the Design of Network-based Software Architectures." Ph.D. Dissertation, University of California, Irvine, 2000. https://www.ics.uci.edu/~fielding/pubs/dissertation/top.htm
2. RFC 9457. "Problem Details for HTTP APIs." IETF, 2023. https://datatracker.ietf.org/doc/html/rfc9457
3. RFC 7396. "JSON Merge Patch." IETF, 2014. https://datatracker.ietf.org/doc/html/rfc7396
4. RFC 6902. "JavaScript Object Notation (JSON) Patch." IETF, 2013. https://datatracker.ietf.org/doc/html/rfc6902
5. RFC 9110. "HTTP Semantics." IETF, 2022. https://datatracker.ietf.org/doc/html/rfc9110
6. Google. "Google JSON Style Guide." https://google.github.io/styleguide/jsoncstyleguide.xml
7. Stripe. "Stripe API Reference - Idempotent Requests." https://docs.stripe.com/api/idempotent_requests
