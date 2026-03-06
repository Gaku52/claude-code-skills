# REST vs GraphQL

> RESTとGraphQLの本質的な違い、それぞれの強みと弱み、選定基準を体系的に比較する。プロジェクトの要件に応じた適切な選択と、ハイブリッドアプローチまで、実践的な判断基準を提供する。アーキテクチャの根本的な設計思想から、パフォーマンス特性、運用上の考慮事項、ユースケース別の選択指針に至るまでを網羅的に解説する。

## この章で学ぶこと

- [ ] RESTとGraphQLのアーキテクチャ上の根本的な違いを理解する
- [ ] データ取得パターンの差異とそのトレードオフを把握する
- [ ] パフォーマンス特性の違いとベンチマーク観点を学ぶ
- [ ] プロジェクト要件に基づく選定基準を体系化する
- [ ] ハイブリッドアプローチの設計パターンを習得する
- [ ] 移行戦略と段階的導入の手法を理解する
- [ ] アンチパターンとエッジケースへの対処法を身につける

---

## 1. アーキテクチャ思想の根本的な違い

REST（Representational State Transfer）とGraphQLは、APIの設計に対して根本的に異なるアプローチを取る。両者の違いを正確に理解するためには、それぞれが生まれた背景と設計哲学を把握する必要がある。

### 1.1 RESTの設計哲学

RESTは2000年にRoy Fieldingの博士論文で提唱されたアーキテクチャスタイルである。Webの成功を支える原則を体系化したものであり、以下の制約に基づいている。

```
REST の6つの制約:

  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Client-Server（クライアント・サーバー分離）                │
  │     → 関心の分離により独立して進化可能                         │
  │                                                                 │
  │  2. Stateless（ステートレス）                                   │
  │     → 各リクエストは自己完結的、サーバーはセッション不保持     │
  │                                                                 │
  │  3. Cacheable（キャッシュ可能）                                 │
  │     → レスポンスにキャッシュ可否を明示                         │
  │                                                                 │
  │  4. Uniform Interface（統一インターフェース）                   │
  │     → リソース識別、表現による操作、自己記述的メッセージ、     │
  │       HATEOAS（ハイパーメディアによる状態遷移）               │
  │                                                                 │
  │  5. Layered System（階層化システム）                            │
  │     → 中間層（プロキシ、ゲートウェイ）の透過的挿入             │
  │                                                                 │
  │  6. Code on Demand（オプション）                                │
  │     → サーバーからクライアントへのコード転送                   │
  └─────────────────────────────────────────────────────────────────┘
```

RESTの核心は「リソース指向」にある。世界をリソース（名詞）として捉え、HTTPメソッド（動詞）で操作する。URIはリソースのアイデンティティであり、表現はリソースの状態を伝える。

### 1.2 GraphQLの設計哲学

GraphQLは2012年にFacebook（現Meta）で開発され、2015年にオープンソース化された。モバイルアプリケーション開発における以下の課題を解決するために生まれた。

```
GraphQL が解決した課題:

  ┌─────────────────────────────────────────────────────────────────┐
  │  課題1: Over-fetching（過剰取得）                              │
  │  → モバイルで不要なデータが帯域を圧迫                         │
  │                                                                 │
  │  課題2: Under-fetching（取得不足）                             │
  │  → 1画面に必要なデータに複数リクエストが必要                  │
  │                                                                 │
  │  課題3: エンドポイント爆発                                     │
  │  → クライアントごとに専用エンドポイントが増殖                 │
  │                                                                 │
  │  課題4: バージョニング地獄                                     │
  │  → v1, v2, v3... の管理コスト                                  │
  │                                                                 │
  │  課題5: フロントエンド・バックエンドの密結合                   │
  │  → 画面変更のたびにAPI変更が必要                               │
  └─────────────────────────────────────────────────────────────────┘
```

GraphQLの核心は「クライアント駆動のデータ取得」にある。サーバーはデータグラフを公開し、クライアントが必要なデータの形状を宣言的に指定する。

### 1.3 アーキテクチャモデルの比較図

```
REST アーキテクチャモデル:

  Client                     Server
  ┌──────┐                   ┌──────────────────────────┐
  │      │  GET /users/1     │  /users/:id      → UserController.show     │
  │      │ ──────────────→   │  /users/:id/posts → PostController.index   │
  │      │  GET /users/1/    │  /posts/:id       → PostController.show    │
  │      │      posts        │  /comments/:id    → CommentController.show │
  │      │ ──────────────→   │                                            │
  │      │  GET /posts/1/    │  各エンドポイントが独立したリソースを      │
  │      │      comments     │  表現し、固定のレスポンス構造を返す        │
  │      │ ──────────────→   │                                            │
  └──────┘                   └──────────────────────────────────────────┘
  3回のリクエスト             3つのエンドポイント


GraphQL アーキテクチャモデル:

  Client                     Server
  ┌──────┐                   ┌──────────────────────────┐
  │      │  POST /graphql    │  Schema（型定義）        │
  │      │  query {          │    ├── User              │
  │      │    user(id:1) {   │    │   ├── name          │
  │      │      name         │    │   └── posts         │
  │      │      posts {      │    ├── Post              │
  │      │        title      │    │   ├── title         │
  │      │        comments { │    │   └── comments      │
  │      │          body     │    └── Comment           │
  │      │        }          │        └── body          │
  │      │      }            │                          │
  │      │    }              │  Resolver が必要な        │
  │      │  }                │  データのみ取得・結合     │
  │      │ ──────────────→   │                          │
  └──────┘                   └──────────────────────────┘
  1回のリクエスト             1つのエンドポイント
```

---

## 2. 基本比較表

以下の表では、両者の主要な特性を網羅的に比較する。

### 比較表1: 技術的特性の比較

| 観点 | REST | GraphQL |
|------|------|---------|
| エンドポイント | 複数（リソースごとに1つ） | 単一（`/graphql`） |
| データ取得 | サーバーが決定（固定構造） | クライアントが宣言的に指定 |
| 型システム | なし（OpenAPI/JSON Schemaで補完） | 組み込み（SDL: Schema Definition Language） |
| HTTPキャッシュ | 標準のHTTPキャッシュ機構を完全活用 | 困難（POSTリクエストのため） |
| 学習コスト | 低い（HTTP知識で十分） | 中程度（SDL、リゾルバー、クライアントライブラリ） |
| エコシステム | 非常に成熟（20年以上の歴史） | 成長中（2015年〜） |
| ファイルアップロード | 容易（multipart/form-data） | 複雑（別途対応が必要） |
| リアルタイム通信 | WebSocket/SSEを別途実装 | Subscription が組み込み |
| エラーハンドリング | HTTPステータスコード（4xx, 5xx） | 常にHTTP 200 + errors配列 |
| テスト容易性 | curl等で即テスト可能 | 専用クライアント（GraphiQL等）が推奨 |
| バージョニング | URLベース（/v1/, /v2/）またはヘッダー | 不要（スキーマの進化的追加） |
| オーバーヘッド | 低い（HTTPメソッドのみ） | クエリの解析・検証コスト |
| ドキュメント | OpenAPI/Swagger UIで生成 | スキーマ自体がドキュメント |
| コード生成 | OpenAPI Generatorで可能 | codegen で型安全なクライアント自動生成 |
| 認証・認可 | HTTPヘッダー（標準的） | HTTPヘッダー + フィールドレベル認可 |

### 比較表2: 運用・組織面の比較

| 観点 | REST | GraphQL |
|------|------|---------|
| チーム構成 | バックエンド中心 | フロントエンド・バックエンド協調 |
| API設計プロセス | エンドポイント設計（URL設計） | スキーマ設計（型とリレーション） |
| モニタリング | エンドポイント単位で明確 | クエリ単位で複雑 |
| レート制限 | リクエスト単位で容易 | クエリの複雑度ベースが必要 |
| セキュリティ | エンドポイント単位のアクセス制御 | フィールド単位の認可が必要 |
| CDN連携 | 標準対応 | Persisted Queries + APQ で対応 |
| ログ分析 | URLパスで分類容易 | クエリ解析が必要 |
| SLA定義 | エンドポイント単位で明確 | クエリパターン別に定義が必要 |
| 段階的ロールアウト | エンドポイント単位 | フィールド単位の@deprecatedディレクティブ |
| サードパーティ連携 | 広く受け入れられている | 対応サービスが限定的 |

---

## 3. データ取得パターンの詳細比較

### 3.1 Over-fetching と Under-fetching

RESTとGraphQLの最も顕著な違いは、データ取得のパターンにある。

```typescript
// ====================================================================
// コード例1: ユーザーダッシュボード画面のデータ取得
// ====================================================================

// --- REST での実装 ---
// 画面に表示する情報: ユーザー名、直近5件の注文、各注文の商品名

// リクエスト1: ユーザー情報の取得
// GET /api/v1/users/123
const userResponse = await fetch('/api/v1/users/123');
const user = await userResponse.json();
// レスポンス（不要なフィールドも含む = Over-fetching）:
// {
//   "id": "123",
//   "name": "Tanaka Taro",
//   "email": "tanaka@example.com",
//   "avatar": "https://cdn.example.com/avatars/123.jpg",  ← 不要
//   "address": "Tokyo, Japan",                             ← 不要
//   "phone": "+81-90-1234-5678",                           ← 不要
//   "preferences": { ... },                                ← 不要
//   "createdAt": "2024-01-15T10:30:00Z"                    ← 不要
// }

// リクエスト2: 注文一覧の取得（Under-fetching: 別リクエストが必要）
// GET /api/v1/users/123/orders?limit=5&sort=-createdAt
const ordersResponse = await fetch(
  '/api/v1/users/123/orders?limit=5&sort=-createdAt'
);
const orders = await ordersResponse.json();
// レスポンス:
// [
//   { "id": "ord-001", "total": 15000, "status": "delivered", ... },
//   { "id": "ord-002", "total": 8500, "status": "shipped", ... },
//   ...
// ]

// リクエスト3〜7: 各注文の商品詳細（N+1問題）
const orderDetails = await Promise.all(
  orders.map(order =>
    fetch(`/api/v1/orders/${order.id}/items`).then(r => r.json())
  )
);
// 合計リクエスト数: 2 + N（注文数）= 最大7リクエスト


// --- GraphQL での実装 ---
// 1リクエストで必要なデータのみ取得
const DASHBOARD_QUERY = `
  query UserDashboard($userId: ID!) {
    user(id: $userId) {
      name
      email
      orders(first: 5, orderBy: CREATED_AT_DESC) {
        edges {
          node {
            id
            total
            status
            items {
              productName
              price
              quantity
            }
          }
        }
      }
    }
  }
`;

const result = await graphqlClient.query({
  query: DASHBOARD_QUERY,
  variables: { userId: '123' }
});
// 合計リクエスト数: 1
// レスポンスには指定したフィールドのみが含まれる
```

### 3.2 RESTにおけるOver-fetching対策

RESTでもOver-fetchingやUnder-fetchingを軽減する手法は存在する。ただし、これらはGraphQLの機能を部分的に再発明している側面がある。

```typescript
// ====================================================================
// コード例2: REST でのデータ取得最適化パターン
// ====================================================================

// パターンA: フィールド選択（Sparse Fieldsets）
// GET /api/v1/users/123?fields=name,email
// → JSON:API 仕様で標準化されたアプローチ
const user = await fetch('/api/v1/users/123?fields=name,email');
// 利点: Over-fetching を軽減
// 欠点: サーバー側でフィールドフィルタリングの実装が必要
//        ネストされたリソースのフィールド選択が複雑化

// パターンB: リソース展開（Embedding / Include）
// GET /api/v1/users/123?include=orders,orders.items
// → JSON:API の include パラメータ
const userWithOrders = await fetch(
  '/api/v1/users/123?include=orders,orders.items'
);
// 利点: Under-fetching を軽減（1リクエストで関連リソース取得）
// 欠点: サーバー側の実装が複雑
//        include の深さ制限の管理が必要

// パターンC: 専用エンドポイント（View / Projection）
// GET /api/v1/users/123/dashboard-summary
const summary = await fetch('/api/v1/users/123/dashboard-summary');
// 利点: クライアントに最適化されたレスポンス
// 欠点: 画面ごとにエンドポイントが増殖
//        → BFF（Backend for Frontend）パターンの台頭

// パターンD: OData クエリオプション
// GET /api/v1/users?$select=name,email&$expand=orders($top=5)
const odataResult = await fetch(
  '/api/v1/users?$select=name,email&$expand=orders($top=5)'
);
// 利点: 標準化されたクエリ言語
// 欠点: OData 仕様の複雑さ、学習コスト

// パターンE: カスタムクエリパラメータ
// GET /api/v1/users/123?view=detailed&depth=2
const detailedUser = await fetch(
  '/api/v1/users/123?view=detailed&depth=2'
);
// 利点: シンプルに実装可能
// 欠点: 非標準、APIごとに異なるルール
```

### 3.3 Mutation（データ更新）の比較

```typescript
// ====================================================================
// コード例3: データ更新操作の比較
// ====================================================================

// --- REST での更新 ---

// 完全な更新（PUT）
// PUT /api/v1/users/123
await fetch('/api/v1/users/123', {
  method: 'PUT',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'Tanaka Jiro',
    email: 'jiro@example.com',
    address: 'Osaka, Japan',
    phone: '+81-90-9876-5432'
    // 全フィールドを送信する必要がある
  })
});

// 部分更新（PATCH）
// PATCH /api/v1/users/123
await fetch('/api/v1/users/123', {
  method: 'PATCH',
  headers: { 'Content-Type': 'application/merge-patch+json' },
  body: JSON.stringify({
    name: 'Tanaka Jiro'
    // 変更するフィールドのみ
  })
});

// 複数リソースの同時更新は標準的でない
// → バッチエンドポイントの独自実装が必要
// POST /api/v1/batch
await fetch('/api/v1/batch', {
  method: 'POST',
  body: JSON.stringify({
    operations: [
      { method: 'PATCH', path: '/users/123', body: { name: 'New Name' } },
      { method: 'POST', path: '/notifications', body: { ... } }
    ]
  })
});


// --- GraphQL での更新 ---

// 単一の Mutation
const UPDATE_USER = `
  mutation UpdateUser($input: UpdateUserInput!) {
    updateUser(input: $input) {
      user {
        id
        name
        email
        updatedAt
      }
      errors {
        field
        message
      }
    }
  }
`;

await graphqlClient.mutate({
  mutation: UPDATE_USER,
  variables: {
    input: {
      id: '123',
      name: 'Tanaka Jiro'
    }
  }
});
// 利点: 更新後のデータを同じリクエストで取得可能
// 利点: errors フィールドでバリデーションエラーを構造化

// 複数操作の同時実行（標準的にサポート）
const BATCH_MUTATION = `
  mutation BatchUpdate($userInput: UpdateUserInput!, $notif: CreateNotificationInput!) {
    updateUser(input: $userInput) {
      user { id name }
    }
    createNotification(input: $notif) {
      notification { id }
    }
  }
`;
// → 単一リクエストで複数の Mutation を実行
// → ただし、各 Mutation は順次実行される（並列ではない）
```

---

## 4. パフォーマンス比較

パフォーマンスの観点では、RESTとGraphQLはそれぞれ異なる特性を持つ。単純に「どちらが速い」とは言えず、ユースケースによって有利不利が変わる。

### 4.1 レイテンシ特性

```
レイテンシ比較（典型的なシナリオ）:

シナリオ1: 単一リソースの取得（ユーザー情報のみ）
──────────────────────────────────────────────────
  REST:   Client ──GET /users/1──→ Server ──→ DB
          合計: ネットワーク往復1回 + DB 1回
          CDNキャッシュヒット時: 数ms（最速）

  GraphQL: Client ──POST /graphql──→ Parse → Validate → Execute → DB
           合計: ネットワーク往復1回 + 解析コスト + DB 1回
           CDNキャッシュ: 困難（POSTのため）

  結論: 単一リソース → REST が有利（特にCDNキャッシュ活用時）


シナリオ2: 関連リソースの取得（ユーザー + 注文 + 商品）
──────────────────────────────────────────────────
  REST:   Client ──GET /users/1──→ Server
          Client ──GET /users/1/orders──→ Server
          Client ──GET /orders/1/items──→ Server
          Client ──GET /orders/2/items──→ Server
          合計: ネットワーク往復4回（シーケンシャル）

  GraphQL: Client ──POST /graphql──→ Parse → Validate → Execute
           → DB(user) + DB(orders) + DB(items) ← DataLoader でバッチ化
           合計: ネットワーク往復1回 + DB 数回（並列可）

  結論: 関連データ → GraphQL が有利（ラウンドトリップ削減）


シナリオ3: 高トラフィック環境（1000 req/sec）
──────────────────────────────────────────────────
  REST:   CDNキャッシュ活用で大半をオフロード
          Cache-Control ヘッダーによる粒度制御
          キャッシュヒット率80%以上を達成可能

  GraphQL: クエリの多様性によりキャッシュ効率が低下
           Persisted Queries で改善可能
           APQ（Automatic Persisted Queries）で運用負荷軽減

  結論: 高トラフィック + 単純データ → REST が有利
```

### 4.2 ペイロードサイズの比較

```
ペイロードサイズ比較:

ユーザーダッシュボード画面で必要なデータ:
  → ユーザー名、メールアドレス、注文5件のタイトルと金額

REST レスポンス（Over-fetching あり）:
  ┌──────────────────────────────────┐
  │ User レスポンス:         ~800B   │  ← name, email 以外に
  │   id, name, email, avatar,      │     avatar, address, phone,
  │   address, phone, preferences,  │     preferences 等が含まれる
  │   createdAt, updatedAt, ...     │
  ├──────────────────────────────────┤
  │ Orders レスポンス:      ~2000B   │  ← 各注文の全フィールド
  │   [{id, total, status,          │
  │     shippingAddress, items,     │
  │     createdAt, ...}, ...]       │
  ├──────────────────────────────────┤
  │ 合計: ~2800B + HTTPヘッダ×2     │
  └──────────────────────────────────┘

GraphQL レスポンス（必要なデータのみ）:
  ┌──────────────────────────────────┐
  │ { "data": {                      │
  │     "user": {                    │
  │       "name": "Tanaka",          │
  │       "email": "t@example.com",  │
  │       "orders": [                │
  │         {"title":"..","total":..}│
  │       ]                          │
  │     }                            │
  │   }                              │
  │ }                                │
  ├──────────────────────────────────┤
  │ 合計: ~600B + HTTPヘッダ×1      │
  └──────────────────────────────────┘

  差分: REST は約4.7倍のデータ転送量
  → 低帯域環境（モバイルネットワーク）で特に影響大
```

### 4.3 サーバーサイドのパフォーマンス考慮事項

```typescript
// ====================================================================
// コード例4: GraphQL の N+1 問題と DataLoader による解決
// ====================================================================

// N+1 問題が発生するリゾルバー（アンチパターン）
const resolvers = {
  Query: {
    users: () => db.query('SELECT * FROM users LIMIT 10')
    // → 1回のクエリ
  },
  User: {
    orders: (user) => db.query('SELECT * FROM orders WHERE user_id = ?', [user.id])
    // → ユーザーごとに1回 = 10回のクエリ
    // 合計: 1 + 10 = 11クエリ（N+1問題）
  }
};

// DataLoader による解決
import DataLoader from 'dataloader';

const ordersByUserLoader = new DataLoader(async (userIds: string[]) => {
  // バッチクエリ: 1回のSQLで全ユーザーの注文を取得
  const orders = await db.query(
    'SELECT * FROM orders WHERE user_id IN (?)',
    [userIds]
  );
  // userIds の順序に合わせてグループ化して返す
  const ordersByUserId = new Map<string, Order[]>();
  for (const order of orders) {
    const existing = ordersByUserId.get(order.userId) || [];
    existing.push(order);
    ordersByUserId.set(order.userId, existing);
  }
  return userIds.map(id => ordersByUserId.get(id) || []);
});

const optimizedResolvers = {
  Query: {
    users: () => db.query('SELECT * FROM users LIMIT 10')
    // → 1回のクエリ
  },
  User: {
    orders: (user) => ordersByUserLoader.load(user.id)
    // → DataLoader がバッチ化: 1回のクエリ
    // 合計: 1 + 1 = 2クエリ（N+1問題解決）
  }
};

// DataLoader の動作原理:
// 1. 同一イベントループ内の .load() 呼び出しを収集
// 2. process.nextTick() でバッチ関数を実行
// 3. 結果を各呼び出し元に分配
// 4. リクエストスコープでキャッシュ（重複排除）
```

---

## 5. 選定基準の体系化

### 5.1 ユースケース別選定マトリクス

```
RESTを選ぶべき場合:
  [1] シンプルなCRUD操作が中心のアプリケーション
  [2] HTTPキャッシュ（CDN含む）を最大限活用したい
  [3] ファイルアップロード/ダウンロードが主要な機能
  [4] サードパーティ向けの公開API
  [5] チームにGraphQL経験者がいない（学習コスト考慮）
  [6] マイクロサービス間の同期通信（gRPC も検討対象）
  [7] 低レイテンシが最優先事項（CDNキャッシュ前提）
  [8] 規制対応でAPI仕様の厳密な管理が必要
  [9] WebHook との連携が多い

GraphQLを選ぶべき場合:
  [1] 複雑なデータ関係がある（ソーシャルグラフ、EC商品カタログ等）
  [2] 多様なクライアント（Web、iOS、Android、スマートTV等）
  [3] フロントエンドの柔軟性・開発速度が重要
  [4] 1画面で多くの関連データを表示（ダッシュボード等）
  [5] リアルタイム更新が必要（Subscription）
  [6] BFF（Backend for Frontend）層を構築する
  [7] スキーマ駆動開発でフロントエンド・バックエンドを並行開発
  [8] API のバージョニングを避けたい
  [9] 型安全なクライアントコードの自動生成を活用したい

gRPCを選ぶべき場合:
  [1] マイクロサービス間の高速な内部通信
  [2] ストリーミング通信（双方向含む）
  [3] 多言語環境でのサービス間通信
  [4] バイナリデータの効率的な転送
  [5] パフォーマンスが最優先（Protocol Buffersの効率）
```

### 5.2 判断フローチャート

```
API 技術選定フローチャート:

  START
    │
    ├── Q1: 公開API（サードパーティ向け）か？
    │     ├── YES → REST（標準的、ドキュメント豊富、採用障壁低）
    │     └── NO ↓
    │
    ├── Q2: マイクロサービス間の内部通信か？
    │     ├── YES → Q2a: レイテンシが最重要か？
    │     │           ├── YES → gRPC（バイナリ、HTTP/2）
    │     │           └── NO → REST（シンプルさ優先）
    │     └── NO ↓
    │
    ├── Q3: クライアントが3種類以上あるか？
    │     ├── YES → GraphQL（各クライアント最適化）
    │     └── NO ↓
    │
    ├── Q4: 画面に複雑なデータ関係を表示するか？
    │     ├── YES → GraphQL（データグラフの柔軟な探索）
    │     └── NO ↓
    │
    ├── Q5: HTTPキャッシュ/CDNを重視するか？
    │     ├── YES → REST（標準HTTPキャッシュ活用）
    │     └── NO ↓
    │
    ├── Q6: チームにGraphQL経験者がいるか？
    │     ├── YES → GraphQL
    │     └── NO → REST（学習コスト考慮）
    │
    └── 迷ったら → REST（シンプルさは正義）
                    + 後から部分的にGraphQLを追加可能
```

---

## 6. 開発体験（Developer Experience）の比較

### 6.1 API設計プロセス

```
REST の API 設計プロセス:

  1. リソースの特定
     → 名詞を洗い出す（User, Order, Product, ...）

  2. URI 設計
     → /api/v1/users
     → /api/v1/users/:id
     → /api/v1/users/:id/orders
     → ネストの深さ、命名規則の統一

  3. HTTPメソッドのマッピング
     → GET（取得）、POST（作成）、PUT/PATCH（更新）、DELETE（削除）

  4. レスポンス構造の設計
     → 各エンドポイントの JSON 構造を定義
     → ページネーション方式の決定（offset, cursor）

  5. OpenAPI 仕様書の作成
     → YAML/JSON で API 仕様を記述
     → Swagger UI でドキュメント生成

  6. バージョニング戦略
     → URL(/v1/, /v2/)、ヘッダー、メディアタイプ


GraphQL の API 設計プロセス:

  1. ドメインモデルの定義
     → 型（Type）としてエンティティを定義

  2. スキーマ定義（SDL）
     → type User { id: ID!, name: String!, orders: [Order!]! }
     → スキーマ = 仕様書 = ドキュメント

  3. Query / Mutation / Subscription の設計
     → どのデータをどう取得・更新・監視するか

  4. リゾルバーの実装
     → 各フィールドのデータ取得ロジック

  5. スキーマの進化
     → 新しいフィールド追加は非破壊的
     → @deprecated ディレクティブで段階的廃止
```

### 6.2 型安全性とコード生成

```typescript
// ====================================================================
// コード例5: GraphQL Code Generator によるフロントエンド型生成
// ====================================================================

// --- スキーマ定義（サーバー側: schema.graphql）---
// type User {
//   id: ID!
//   name: String!
//   email: String!
//   role: UserRole!
//   orders(first: Int, after: String): OrderConnection!
// }
//
// enum UserRole {
//   ADMIN
//   MEMBER
//   GUEST
// }
//
// type OrderConnection {
//   edges: [OrderEdge!]!
//   pageInfo: PageInfo!
// }
//
// type OrderEdge {
//   node: Order!
//   cursor: String!
// }
//
// type Order {
//   id: ID!
//   total: Int!
//   status: OrderStatus!
//   items: [OrderItem!]!
//   createdAt: DateTime!
// }

// --- クエリ定義（フロントエンド側: queries/user.graphql）---
// query GetUserDashboard($userId: ID!) {
//   user(id: $userId) {
//     name
//     email
//     role
//     orders(first: 5) {
//       edges {
//         node {
//           id
//           total
//           status
//         }
//       }
//     }
//   }
// }

// --- codegen が自動生成する型（generated/graphql.ts）---
// ※ 以下は codegen の出力イメージ

export type UserRole = 'ADMIN' | 'MEMBER' | 'GUEST';
export type OrderStatus = 'PENDING' | 'SHIPPED' | 'DELIVERED' | 'CANCELLED';

export interface GetUserDashboardQuery {
  user: {
    __typename: 'User';
    name: string;
    email: string;
    role: UserRole;
    orders: {
      edges: Array<{
        node: {
          __typename: 'Order';
          id: string;
          total: number;
          status: OrderStatus;
        };
      }>;
    };
  } | null;
}

export interface GetUserDashboardQueryVariables {
  userId: string;
}

// --- 型安全なコンポーネント（React + Apollo Client）---
import { useQuery } from '@apollo/client';
import { GetUserDashboardQuery, GetUserDashboardQueryVariables } from './generated/graphql';
import { GET_USER_DASHBOARD } from './queries/user';

function UserDashboard({ userId }: { userId: string }) {
  const { data, loading, error } = useQuery<
    GetUserDashboardQuery,
    GetUserDashboardQueryVariables
  >(GET_USER_DASHBOARD, {
    variables: { userId }
  });

  if (loading) return <Loading />;
  if (error) return <Error message={error.message} />;
  if (!data?.user) return <NotFound />;

  // data.user.name → string（型安全）
  // data.user.role → 'ADMIN' | 'MEMBER' | 'GUEST'（型安全）
  // data.user.orders.edges[0].node.total → number（型安全）
  // data.user.nonExistent → コンパイルエラー（存在しないフィールド）

  return (
    <div>
      <h1>{data.user.name}</h1>
      <p>{data.user.email}</p>
      <OrderList orders={data.user.orders.edges.map(e => e.node)} />
    </div>
  );
}
```

### 6.3 テスト・デバッグ体験の比較

```
REST のテスト・デバッグ:

  ┌─────────────────────────────────────────────────────────┐
  │ ツールチェーン:                                         │
  │   - curl / HTTPie: コマンドラインから即テスト           │
  │   - Postman / Insomnia: GUI ベースのテスト             │
  │   - Swagger UI: OpenAPI から自動生成のテスト画面       │
  │   - REST Client (VS Code): エディタ内でテスト         │
  │                                                         │
  │ 利点:                                                   │
  │   - ブラウザのアドレスバーで GET リクエストテスト       │
  │   - curl 一行でテスト完結                               │
  │   - HTTPステータスコードで即座にエラー原因を特定       │
  │   - ネットワークタブで直感的にデバッグ                 │
  │                                                         │
  │ 例:                                                     │
  │   $ curl -s http://api.example.com/users/1 | jq .      │
  │   $ curl -X POST http://api.example.com/users \         │
  │       -H "Content-Type: application/json" \             │
  │       -d '{"name":"test"}'                              │
  └─────────────────────────────────────────────────────────┘

GraphQL のテスト・デバッグ:

  ┌─────────────────────────────────────────────────────────┐
  │ ツールチェーン:                                         │
  │   - GraphiQL: インタラクティブ IDE（自動補完付き）     │
  │   - Apollo Studio / Explorer: 高機能なテスト環境       │
  │   - Altair GraphQL Client: デスクトップクライアント    │
  │   - Apollo DevTools: ブラウザ拡張（キャッシュ可視化）  │
  │                                                         │
  │ 利点:                                                   │
  │   - スキーマの自動補完によるクエリ作成支援             │
  │   - ドキュメントエクスプローラーで API 探索            │
  │   - クエリのパフォーマンス分析                         │
  │   - キャッシュの状態を可視化（Apollo DevTools）        │
  │                                                         │
  │ 注意:                                                   │
  │   - curl でのテストが冗長（POST + JSON ボディ）        │
  │   - エラーが常に HTTP 200 → ステータスコードで判断不可 │
  │   - ネットワークタブでは全て POST /graphql に見える    │
  └─────────────────────────────────────────────────────────┘
```

---

## 7. セキュリティの比較

### 7.1 REST のセキュリティモデル

RESTでは、エンドポイント単位でのアクセス制御が基本となる。これはWebアプリケーションフレームワークの標準的なミドルウェア/フィルターと相性が良い。

```typescript
// REST のセキュリティ実装パターン
// Express.js + ミドルウェアの例

// エンドポイント単位の認可
app.get('/api/v1/users', authenticate, authorize('admin'), usersController.list);
app.get('/api/v1/users/:id', authenticate, usersController.show);
app.post('/api/v1/users', authenticate, authorize('admin'), usersController.create);
app.delete('/api/v1/users/:id', authenticate, authorize('admin'), usersController.delete);

// レート制限（エンドポイント単位で容易）
app.use('/api/v1/', rateLimit({
  windowMs: 15 * 60 * 1000,  // 15分
  max: 100,                   // 100リクエスト
  standardHeaders: true
}));

// 入力バリデーション（ルートごとに定義）
app.post('/api/v1/users',
  body('name').isString().isLength({ min: 1, max: 100 }),
  body('email').isEmail(),
  validateRequest,
  usersController.create
);
```

### 7.2 GraphQL のセキュリティモデル

GraphQLでは、クライアントがクエリを自由に構成できるため、より細かなセキュリティ対策が必要となる。

```typescript
// ====================================================================
// コード例6: GraphQL のセキュリティ対策
// ====================================================================

// --- 1. クエリの深度制限 ---
import depthLimit from 'graphql-depth-limit';

const server = new ApolloServer({
  schema,
  validationRules: [
    depthLimit(7)  // ネストの深さを7レベルに制限
  ]
});

// 攻撃例（深いネストによるDoS）:
// query {
//   user(id: "1") {
//     friends {
//       friends {
//         friends {
//           friends {
//             friends { ... }  ← 深いネストでサーバーリソース消費
//           }
//         }
//       }
//     }
//   }
// }

// --- 2. クエリの複雑度制限 ---
import { createComplexityLimitRule } from 'graphql-validation-complexity';

const complexityRule = createComplexityLimitRule(1000, {
  scalarCost: 1,
  objectCost: 2,
  listFactor: 10
});
// → クエリ全体の「コスト」を計算し、閾値を超えたら拒否

// --- 3. フィールドレベルの認可 ---
const resolvers = {
  User: {
    email: (user, args, context) => {
      // 本人またはAdmin のみメールアドレスを閲覧可能
      if (context.currentUser.id === user.id ||
          context.currentUser.role === 'ADMIN') {
        return user.email;
      }
      return null;  // 権限がない場合は null を返す
    },
    salary: (user, args, context) => {
      // HR部門のみ給与情報を閲覧可能
      if (!context.currentUser.departments.includes('HR')) {
        throw new ForbiddenError('Insufficient permissions');
      }
      return user.salary;
    }
  }
};

// --- 4. Persisted Queries（クエリの事前登録）---
// 本番環境では任意のクエリを受け付けず、事前登録されたクエリのみ実行
const server = new ApolloServer({
  schema,
  persistedQueries: {
    cache: new InMemoryLRUCache()
  }
});
// クライアントはクエリのハッシュ値を送信
// POST /graphql
// { "extensions": { "persistedQuery": { "sha256Hash": "abc123..." } } }

// --- 5. インジェクション対策 ---
// GraphQL の変数は型システムで検証されるため、
// SQLインジェクション等はリゾルバー実装に依存
const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      // NG: 文字列結合（SQLインジェクション脆弱性）
      // return db.query(`SELECT * FROM users WHERE id = '${id}'`);

      // OK: パラメータ化クエリ
      return db.query('SELECT * FROM users WHERE id = $1', [id]);
    }
  }
};
```

### 7.3 セキュリティ比較サマリー

```
セキュリティ観点の比較:

  ┌──────────────────────┬───────────────────┬────────────────────┐
  │ 観点                 │ REST              │ GraphQL            │
  ├──────────────────────┼───────────────────┼────────────────────┤
  │ アクセス制御の粒度   │ エンドポイント    │ フィールド         │
  │ レート制限           │ 容易              │ 複雑度ベースが必要 │
  │ DoS対策              │ 標準的            │ 深度/複雑度制限    │
  │ 入力バリデーション   │ 手動定義          │ 型システムで一部   │
  │ イントロスペクション │ 該当なし          │ 本番で無効化推奨   │
  │ クエリ制御           │ サーバーが決定    │ Persisted Queries  │
  │ 認証                 │ 標準HTTPヘッダー  │ 同左               │
  │ CORS                 │ 標準対応          │ 同左               │
  └──────────────────────┴───────────────────┴────────────────────┘
```

---

## 8. ハイブリッドアプローチの設計パターン

実務では REST と GraphQL を共存させるパターンが多く採用されている。プロジェクトの特性に応じて、最適な組み合わせを選択することが重要である。

### 8.1 パターン一覧

```
パターン1: REST + GraphQL BFF（Backend for Frontend）

  ┌─────────┐     GraphQL      ┌──────────┐     REST      ┌──────────────┐
  │  Web    │ ──────────────→  │  GraphQL │ ──────────→  │ User Service │
  │  App    │                   │  BFF     │              │              │
  └─────────┘                   │          │              └──────────────┘
                                │          │     REST      ┌──────────────┐
  ┌─────────┐     GraphQL      │          │ ──────────→  │ Order Service│
  │  Mobile │ ──────────────→  │          │              │              │
  │  App    │                   │          │              └──────────────┘
  └─────────┘                   └──────────┘     REST      ┌──────────────┐
                                              ──────────→  │ Product Svc  │
                                                           └──────────────┘

  利点:
  → フロントエンドは GraphQL の柔軟性を享受
  → バックエンドは REST の安定性を維持
  → BFF がデータの集約・変換を担当
  → 既存の REST マイクロサービスを変更不要


パターン2: 機能別の使い分け

  ┌─────────────────────────────────────────────┐
  │ アプリケーション                            │
  │                                              │
  │  CRUD操作         → REST API                │
  │  ダッシュボード   → GraphQL（複雑なデータ） │
  │  ファイル操作     → REST API                │
  │  リアルタイム     → GraphQL Subscription    │
  │  Webhook受信      → REST API                │
  │  検索             → REST（Elasticsearch）   │
  └─────────────────────────────────────────────┘


パターン3: 公開 API / 内部 API の分離

  ┌──────────────┐    REST       ┌──────────────┐
  │ Third Party  │ ──────────→  │              │
  │ Developers   │              │              │
  └──────────────┘              │   API        │
                                │   Gateway    │
  ┌──────────────┐   GraphQL    │              │
  │ 自社         │ ──────────→  │              │
  │ フロントエンド│              │              │
  └──────────────┘              └──────────────┘

  → サードパーティ: REST（標準的、キャッシュ可能、ドキュメント容易）
  → 自社フロントエンド: GraphQL（柔軟、型安全、開発効率）


パターン4: GraphQL Federation（スーパーグラフ）

  ┌─────────┐      ┌──────────────────────┐
  │ Client  │ ──→  │  GraphQL Gateway     │
  └─────────┘      │  (Apollo Router /    │
                    │   GraphQL Mesh)      │
                    └──────┬───────────────┘
                           │
               ┌───────────┼───────────────┐
               │           │               │
        ┌──────▼──┐  ┌─────▼───┐  ┌───────▼────┐
        │  User   │  │  Order  │  │  Product   │
        │  SubG   │  │  SubG   │  │  SubG      │
        │ (REST)  │  │ (gRPC)  │  │ (GraphQL)  │
        └─────────┘  └─────────┘  └────────────┘

  → Apollo Federation / GraphQL Mesh で統一インターフェース
  → 各サービスは最適な技術を選択
  → Gateway が自動的にクエリを分解・統合
```

### 8.2 Apollo Federation の実装例

```typescript
// ====================================================================
// コード例7: Apollo Federation によるマイクロサービス統合
// ====================================================================

// --- ユーザーサービス（サブグラフ） ---
import { buildSubgraphSchema } from '@apollo/subgraph';
import { gql } from 'graphql-tag';

const userTypeDefs = gql`
  extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
                      import: ["@key"])

  type User @key(fields: "id") {
    id: ID!
    name: String!
    email: String!
    role: UserRole!
  }

  enum UserRole {
    ADMIN
    MEMBER
    GUEST
  }

  type Query {
    user(id: ID!): User
    users(first: Int, after: String): UserConnection!
  }
`;

const userResolvers = {
  Query: {
    user: (_, { id }) => userRepository.findById(id),
    users: (_, { first, after }) => userRepository.paginate({ first, after })
  },
  User: {
    __resolveReference: (ref) => userRepository.findById(ref.id)
  }
};

const userSchema = buildSubgraphSchema([
  { typeDefs: userTypeDefs, resolvers: userResolvers }
]);


// --- 注文サービス（サブグラフ） ---
const orderTypeDefs = gql`
  extend schema @link(url: "https://specs.apollo.dev/federation/v2.0",
                      import: ["@key", "@external"])

  type Order @key(fields: "id") {
    id: ID!
    total: Int!
    status: OrderStatus!
    items: [OrderItem!]!
    createdAt: DateTime!
  }

  enum OrderStatus {
    PENDING
    PROCESSING
    SHIPPED
    DELIVERED
    CANCELLED
  }

  type OrderItem {
    productId: ID!
    productName: String!
    price: Int!
    quantity: Int!
  }

  # User 型を拡張して orders フィールドを追加
  extend type User @key(fields: "id") {
    id: ID! @external
    orders(first: Int, after: String): OrderConnection!
  }

  type OrderConnection {
    edges: [OrderEdge!]!
    pageInfo: PageInfo!
    totalCount: Int!
  }

  type OrderEdge {
    node: Order!
    cursor: String!
  }

  type PageInfo {
    hasNextPage: Boolean!
    hasPreviousPage: Boolean!
    startCursor: String
    endCursor: String
  }
`;

const orderResolvers = {
  User: {
    orders: (user, { first, after }) =>
      orderRepository.findByUserId(user.id, { first, after })
  },
  Order: {
    __resolveReference: (ref) => orderRepository.findById(ref.id)
  }
};


// --- Gateway（ルーター）の設定 ---
// supergraph.yaml
// subgraphs:
//   users:
//     routing_url: http://users-service:4001/graphql
//     schema:
//       subgraph_url: http://users-service:4001/graphql
//   orders:
//     routing_url: http://orders-service:4002/graphql
//     schema:
//       subgraph_url: http://orders-service:4002/graphql

// Gateway がスキーマを統合した結果、
// クライアントは以下のようなクエリを実行できる:
//
// query {
//   user(id: "1") {
//     name          ← ユーザーサービスから取得
//     email         ← ユーザーサービスから取得
//     orders(first: 5) {  ← 注文サービスから取得
//       edges {
//         node {
//           id
//           total
//           status
//           items {
//             productName
//             price
//           }
//         }
//       }
//     }
//   }
// }
```

---

## 9. 移行戦略

### 9.1 REST から GraphQL への段階的移行

```
Phase 1: GraphQL Layer の追加（2-4週間）
──────────────────────────────────────────

  ┌─────────┐    GraphQL     ┌──────────────┐    REST     ┌───────────┐
  │ Client  │ ──────────→   │  GraphQL     │ ─────────→ │ 既存 REST │
  │ (新規)  │                │  Layer       │            │ API       │
  └─────────┘                │              │            │           │
                              │ リゾルバーが │            └───────────┘
  ┌─────────┐    REST        │ 内部で REST  │
  │ Client  │ ──────────→   │ を呼び出す   │
  │ (既存)  │                └──────────────┘
  └─────────┘
  ※ 既存クライアントは REST を継続

  実施内容:
  → 既存 REST API の上に GraphQL レイヤーを構築
  → GraphQL のリゾルバーが内部で REST API を呼ぶ
  → 新規クライアントから GraphQL を使い始める
  → 既存クライアントは影響を受けない


Phase 2: 新機能は GraphQL で開発（1-3ヶ月）
──────────────────────────────────────────

  実施内容:
  → 新しい画面/機能から GraphQL を使用
  → 既存画面は引き続き REST を使用
  → チームが GraphQL に習熟する期間
  → GraphQL リゾルバーの一部を直接 DB アクセスに変更


Phase 3: 既存機能の段階的移行（3-6ヶ月）
──────────────────────────────────────────

  実施内容:
  → 使用量の少ない REST エンドポイントから移行
  → クライアント側のデータ取得層を GraphQL に切り替え
  → REST エンドポイントの利用状況をモニタリング
  → 未使用の REST エンドポイントを段階的に廃止


Phase 4: 最適化と安定化（1-2ヶ月）
──────────────────────────────────────────

  実施内容:
  → GraphQL リゾルバーの全てを直接 DB アクセスに変更
  → DataLoader の最適化
  → Persisted Queries の導入
  → パフォーマンスモニタリングの整備
```

### 9.2 移行時の注意事項

```
移行のリスクと対策:

  リスク1: パフォーマンス劣化
  → 対策: GraphQL Layer のリゾルバーが REST を呼ぶ構成では
           ネットワークホップが増加する
           → 内部通信のレイテンシをモニタリング
           → 早期に直接 DB アクセスに切り替え

  リスク2: チームの学習コスト
  → 対策: 小さな機能から始める
           → ペアプログラミングで知見を共有
           → GraphQL の社内勉強会を開催

  リスク3: キャッシュ戦略の変更
  → 対策: REST 時代の HTTP キャッシュから
           GraphQL のクライアントキャッシュへ
           → Apollo Client の正規化キャッシュを活用
           → CDN キャッシュは Persisted Queries で対応

  リスク4: モニタリングの変更
  → 対策: エンドポイント単位 → クエリ単位への切り替え
           → Apollo Studio / GraphQL トレーシングを導入
           → クエリのパフォーマンスを可視化

  リスク5: ロールバックの困難さ
  → 対策: REST API を一定期間維持
           → Feature Flag で GraphQL/REST を切り替え可能に
           → 段階的にトラフィックを移行
```

---

## 10. アンチパターン

### 10.1 アンチパターン1: GraphQL で REST を再発明する

```typescript
// ====================================================================
// アンチパターン: GraphQL で REST のようなクエリ設計
// ====================================================================

// --- NG: リソースごとに別々のクエリを定義 ---
const BAD_SCHEMA = `
  type Query {
    # REST のエンドポイントをそのまま GraphQL に移植
    getUser(id: ID!): User
    getUserOrders(userId: ID!, page: Int): [Order]
    getUserOrderItems(orderId: ID!): [OrderItem]
    getUserAddress(userId: ID!): Address
    getUserPaymentMethods(userId: ID!): [PaymentMethod]
  }
`;

// クライアント側のコード（REST と変わらない）
const user = await query({ query: GET_USER, variables: { id: '1' } });
const orders = await query({ query: GET_USER_ORDERS, variables: { userId: '1' } });
const items = await query({ query: GET_USER_ORDER_ITEMS, variables: { orderId: orders[0].id } });
// → GraphQL を使う意味がない。REST の方が適切。

// --- OK: グラフ構造を活かした設計 ---
const GOOD_SCHEMA = `
  type Query {
    user(id: ID!): User
  }

  type User {
    id: ID!
    name: String!
    email: String!
    address: Address
    orders(first: Int, after: String): OrderConnection!
    paymentMethods: [PaymentMethod!]!
  }

  type Order {
    id: ID!
    total: Int!
    items: [OrderItem!]!
    shippingAddress: Address!
  }
`;

// クライアントが必要なデータ構造を宣言的に指定
const GOOD_QUERY = `
  query UserDashboard($id: ID!) {
    user(id: $id) {
      name
      orders(first: 5) {
        edges {
          node {
            total
            items { productName, price }
          }
        }
      }
    }
  }
`;
// → 1リクエストで必要なデータを全て取得
```

**解説**: GraphQL の価値は「データグラフの柔軟な探索」にある。REST のエンドポイント構造をそのまま GraphQL に移植すると、クエリ解析のオーバーヘッドが加わるだけで利点が得られない。スキーマ設計では、リソース間の関係（グラフ構造）を型のフィールドとして表現し、クライアントが必要な深さまでトラバースできるようにすることが重要である。

### 10.2 アンチパターン2: 認可を GraphQL リゾルバーに分散配置する

```typescript
// ====================================================================
// アンチパターン: リゾルバーごとに認可ロジックを記述
// ====================================================================

// --- NG: 各リゾルバーに認可コードが分散 ---
const BAD_RESOLVERS = {
  Query: {
    users: (_, args, ctx) => {
      if (!ctx.user) throw new AuthenticationError('Not authenticated');
      if (ctx.user.role !== 'ADMIN') throw new ForbiddenError('Admin only');
      return db.users.findAll();
    },
    orders: (_, args, ctx) => {
      if (!ctx.user) throw new AuthenticationError('Not authenticated');
      // role チェックを忘れがち → セキュリティホール
      return db.orders.findAll();
    },
    // 新しいリゾルバーを追加するたびに認可コードをコピペ
    // → メンテナンスコスト増大、漏れのリスク
  },
  User: {
    email: (user, _, ctx) => {
      if (!ctx.user) throw new AuthenticationError('Not authenticated');
      if (ctx.user.id !== user.id && ctx.user.role !== 'ADMIN') {
        return null;
      }
      return user.email;
    }
  }
};

// --- OK: 認可レイヤーを分離（graphql-shield 等）---
import { shield, rule, and, or } from 'graphql-shield';

const isAuthenticated = rule()((_, args, ctx) => {
  return ctx.user !== null;
});

const isAdmin = rule()((_, args, ctx) => {
  return ctx.user?.role === 'ADMIN';
});

const isOwner = rule()((parent, args, ctx) => {
  return parent.id === ctx.user?.id;
});

const permissions = shield({
  Query: {
    users: and(isAuthenticated, isAdmin),
    orders: isAuthenticated,
    user: isAuthenticated
  },
  User: {
    email: and(isAuthenticated, or(isAdmin, isOwner)),
    salary: and(isAuthenticated, isAdmin)
  },
  Mutation: {
    updateUser: and(isAuthenticated, or(isAdmin, isOwner)),
    deleteUser: and(isAuthenticated, isAdmin)
  }
});

// リゾルバーはビジネスロジックのみに集中
const GOOD_RESOLVERS = {
  Query: {
    users: () => db.users.findAll(),
    orders: () => db.orders.findAll(),
    user: (_, { id }) => db.users.findById(id)
  },
  User: {
    email: (user) => user.email,
    salary: (user) => user.salary
  }
};
// → 認可ルールが一箇所に集約、漏れが起きにくい
// → リゾルバーのテストが容易（認可を気にしなくてよい）
```

**解説**: 認可ロジックをリゾルバーに直接埋め込むと、新しいフィールドやクエリの追加時に認可チェックを忘れるリスクが高まる。graphql-shield のような認可レイヤーを使うことで、認可ルールを宣言的に一箇所で管理できる。これは REST における認可ミドルウェアに相当するアプローチである。

### 10.3 アンチパターン3: GraphQL でファイルアップロードを無理に実装する

```
NG パターン:
  → GraphQL の Mutation で Base64 エンコードしたファイルを送信
  → multipart/form-data の GraphQL 拡張（graphql-upload）を使う
  → 大きなファイルでメモリ問題が発生

OK パターン:
  → ファイルアップロードは REST エンドポイント（または S3 直接）で処理
  → アップロード完了後、GraphQL Mutation でメタデータを登録

  手順:
  1. REST: POST /api/upload → { "fileUrl": "https://cdn.example.com/file.jpg" }
  2. GraphQL: mutation { attachFile(url: "https://cdn.example.com/file.jpg") { ... } }

  → REST と GraphQL の適材適所
```

