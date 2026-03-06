# GraphQL基礎

> GraphQLはFacebookが開発したクエリ言語。スキーマ駆動開発、型システム、Query/Mutation/Subscription、リゾルバーの仕組みを理解し、REST APIとは異なるアプローチでのAPI設計を習得する。

## この章で学ぶこと

- [ ] GraphQLの型システムとスキーマ定義を理解する
- [ ] Query・Mutation・Subscriptionの使い分けを把握する
- [ ] リゾルバーの実装パターンを学ぶ
- [ ] Apollo Serverの構築と運用手法を身につける
- [ ] N+1問題とDataLoaderによる最適化を理解する
- [ ] エラーハンドリングと認証・認可パターンを習得する

---

## 1. GraphQLとは

### 1.1 概要と歴史

GraphQLは2012年にFacebook社内でモバイルアプリ向けのデータ取得基盤として開発された。2015年にオープンソースとして公開され、2019年にはLinux Foundation傘下のGraphQL Foundationに移管された。

```
GraphQL = Graph Query Language
  → API のためのクエリ言語 + 型システム + ランタイム
  → Facebook が 2012年に内部開発、2015年に公開、2019年にGraphQL Foundation設立

歴史年表:
  2012  Facebook内部で開発開始（モバイルニュースフィード向け）
  2015  React.jsカンファレンスで公開、仕様のオープンソース化
  2016  GitHub API v4がGraphQLを採用（大規模事例の先駆け）
  2017  Apollo, Relay Modern, Prismaなどエコシステムが拡大
  2018  GraphQL仕様にSubscriptionが正式追加
  2019  GraphQL Foundation設立（Linux Foundation傘下）
  2021  @defer, @streamディレクティブの仕様策定開始
  2023  GraphQL仕様 October 2021 Editionの安定リリース
  2024  Composite Schema（Federation統一仕様）のRFC進行中
```

### 1.2 GraphQLの3つの柱

```
+-------------------------------------------------------------------+
|                    GraphQLの3つの柱                                 |
+-------------------------------------------------------------------+
|                                                                   |
|  [1] クエリ言語           [2] 型システム         [3] ランタイム      |
|  (Query Language)        (Type System)         (Execution Engine) |
|                                                                   |
|  クライアントが            スキーマで              リクエストを       |
|  欲しいデータを            APIの形を               解析・検証し       |
|  宣言的に記述              厳密に定義               結果を返す        |
|                                                                   |
|  ・Query                 ・Scalar型              ・パース           |
|  ・Mutation              ・Object型              ・バリデーション     |
|  ・Subscription          ・Enum型                ・実行（リゾルバー） |
|  ・Fragment              ・Interface/Union       ・シリアライズ       |
|  ・Variable              ・Input型               ・エラーハンドリング  |
|                                                                   |
+-------------------------------------------------------------------+
```

### 1.3 RESTとの比較

```
REST vs GraphQL（イメージ）:

  REST:
    GET /users/123            → { id, name, email, address, ... }
    GET /users/123/orders     → [{ id, total, items, ... }]
    GET /orders/456/items     → [{ id, product, price, ... }]
    → 3リクエスト、不要なデータも含む

  GraphQL:
    POST /graphql
    query {
      user(id: "123") {
        name
        orders(first: 5) {
          total
          items { productName, price }
        }
      }
    }
    → 1リクエスト、必要なデータのみ
```

**比較表1: REST vs GraphQL 機能比較**

| 観点 | REST | GraphQL |
|------|------|---------|
| エンドポイント | リソースごとに複数 (`/users`, `/orders`) | 単一 (`/graphql`) |
| データ取得量 | サーバーが決定（Over-fetching発生） | クライアントが必要なフィールドを指定 |
| 型システム | OpenAPI/Swaggerで別途定義 | スキーマに内蔵（SDL） |
| バージョニング | URLパス (`/v1/`, `/v2/`) が一般的 | スキーマ進化（deprecated + 新フィールド追加） |
| キャッシュ | HTTPキャッシュヘッダーで容易 | 専用キャッシュ戦略が必要（Apollo Cacheなど） |
| リアルタイム | WebSocket / SSE を別途実装 | Subscriptionで標準サポート |
| 学習コスト | 広く知られており低い | SDL・リゾルバーなど独自概念の学習が必要 |
| ファイルアップロード | multipart/form-data で標準対応 | 仕様外（Apollo Upload等で拡張） |
| エラーハンドリング | HTTPステータスコード | 常に200、errorsフィールドで表現 |
| ドキュメント | Swagger UI等で生成 | GraphiQL/Playground等で自動生成+対話的実行 |
| ネストデータ | 複数リクエストまたはInclude指定 | 1リクエストで任意の深さまで取得可能 |

**比較表2: ユースケース別適性**

| ユースケース | REST推奨度 | GraphQL推奨度 | 理由 |
|-------------|-----------|-------------|------|
| CRUDが中心のシンプルAPI | ★★★★★ | ★★★☆☆ | RESTの方がシンプルで十分 |
| モバイルアプリ向けBFF | ★★☆☆☆ | ★★★★★ | 帯域節約・1リクエストが大きい利点 |
| マイクロサービス集約 | ★★★☆☆ | ★★★★★ | Federation/Stitchingで統合容易 |
| 外部公開API | ★★★★★ | ★★★☆☆ | REST+OpenAPIの方が汎用的 |
| ダッシュボード/管理画面 | ★★★☆☆ | ★★★★★ | 複雑なデータ要件に柔軟対応 |
| IoT/組み込み | ★★★★★ | ★★☆☆☆ | HTTP GETの方が軽量 |
| リアルタイム通知 | ★★★☆☆ | ★★★★☆ | Subscriptionで標準サポート |
| ファイル配信/ストリーム | ★★★★★ | ★☆☆☆☆ | GraphQLはJSONデータ向け |

### 1.4 GraphQLのリクエスト/レスポンスフロー

```
┌──────────────┐     POST /graphql      ┌──────────────────────────┐
│              │ ─────────────────────→ │  GraphQL Server          │
│   Client     │     {                  │                          │
│  (Browser/   │       query: "...",    │  1. Parse (構文解析)      │
│   Mobile)    │       variables: {}    │         ↓                │
│              │     }                  │  2. Validate (検証)       │
│              │                        │     - 型チェック           │
│              │                        │     - フィールド存在確認   │
│              │                        │         ↓                │
│              │     {                  │  3. Execute (実行)        │
│              │       data: {...},     │     - リゾルバー呼び出し   │
│              │ ←───────────────────── │     - データソースアクセス  │
│              │       errors: [...]    │         ↓                │
│              │     }                  │  4. Serialize (直列化)    │
└──────────────┘                        └──────────────────────────┘
                                               ↕
                                        ┌──────────────┐
                                        │  DataSources │
                                        │  - Database  │
                                        │  - REST API  │
                                        │  - gRPC      │
                                        │  - Cache     │
                                        └──────────────┘
```

---

## 2. スキーマ定義（SDL）

### 2.1 スカラー型

GraphQLには5つの組み込みスカラー型がある。

```graphql
# 組み込みスカラー型
# Int     : 符号付き32ビット整数
# Float   : 倍精度浮動小数点数
# String  : UTF-8文字列
# Boolean : true / false
# ID      : 一意識別子（内部的にはString）

# カスタムスカラー型の定義
scalar DateTime    # ISO 8601形式の日時文字列
scalar Email       # メールアドレス形式の文字列
scalar URL         # URL形式の文字列
scalar JSON        # 任意のJSONオブジェクト
scalar BigInt      # 64ビット整数（Int範囲を超える場合）
scalar Void        # 戻り値なし（副作用のみのMutationに使用）
```

### 2.2 オブジェクト型と型修飾子

```graphql
# 型修飾子の組み合わせと意味
#
# String    → null許容文字列（値はnullまたはString）
# String!   → 非null文字列（値は必ずString）
# [String]  → null許容の配列（配列自体がnull、要素もnull可）
# [String]! → 非nullの配列（配列自体は非null、要素はnull可）
# [String!] → null許容の配列（配列自体はnull可、要素は非null）
# [String!]!→ 非nullの配列（配列自体も要素も非null）

# ┌──────────────────────────────────────────────────────┐
# │  型修飾子の許容パターン一覧                            │
# ├───────────────┬──────────────────────────────────────┤
# │  宣言         │  許容される値                          │
# ├───────────────┼──────────────────────────────────────┤
# │  String       │  null, "hello"                       │
# │  String!      │  "hello"                             │
# │  [String]     │  null, [], [null], ["a", null]       │
# │  [String]!    │  [], [null], ["a", null]             │
# │  [String!]    │  null, [], ["a", "b"]                │
# │  [String!]!   │  [], ["a", "b"]                      │
# └───────────────┴──────────────────────────────────────┘
```

### 2.3 列挙型

```graphql
# 列挙型
enum UserRole {
  USER
  ADMIN
  EDITOR
  MODERATOR
}

enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}

# 列挙型はフィルタリング、バリデーション、ドキュメント化に有用
# 実行時にスキーマに存在しない値が渡されるとバリデーションエラーになる
```

### 2.4 オブジェクト型の定義

```graphql
# オブジェクト型
type User {
  id: ID!                     # !は非null
  name: String!
  email: Email!
  role: UserRole!
  avatar: String              # nullable
  bio: String
  createdAt: DateTime!
  updatedAt: DateTime!
  orders: [Order!]!           # 非nullの配列（配列自体も非null）
  orderCount: Int!
  posts: [Post!]!
  followers: [User!]!
  following: [User!]!
}

type Order {
  id: ID!
  user: User!
  status: OrderStatus!
  total: Int!                 # 金額（円）
  items: [OrderItem!]!
  shippingAddress: Address
  note: String
  createdAt: DateTime!
  updatedAt: DateTime!
}

type OrderItem {
  id: ID!
  order: Order!
  product: Product!
  quantity: Int!
  unitPrice: Int!
  subtotal: Int!              # quantity * unitPrice（計算フィールド）
}

type Product {
  id: ID!
  name: String!
  price: Int!
  description: String
  category: Category!
  tags: [String!]!
  imageUrl: URL
  stock: Int!
  isAvailable: Boolean!
}

type Category {
  id: ID!
  name: String!
  slug: String!
  parent: Category            # 再帰的な型参照（親カテゴリ）
  children: [Category!]!
  products: [Product!]!
}

type Address {
  postalCode: String!
  prefecture: String!
  city: String!
  street: String!
  building: String
}

type Post {
  id: ID!
  author: User!
  title: String!
  body: String!
  tags: [String!]!
  publishedAt: DateTime
  createdAt: DateTime!
}
```

### 2.5 入力型（Input Types）

```graphql
# 入力型（Mutation の引数に使用）
# input型とtype型の違い:
#   - input型はMutation/Queryの引数にのみ使用可能
#   - input型のフィールドにはtype型を含められない（inputのみ）
#   - input型にはリゾルバーを定義できない

input CreateUserInput {
  name: String!
  email: Email!
  role: UserRole = USER       # デフォルト値
  bio: String
  avatar: String
}

input UpdateUserInput {
  name: String
  email: Email
  role: UserRole
  bio: String
  avatar: String
}

input CreateOrderInput {
  items: [OrderItemInput!]!
  shippingAddress: AddressInput!
  note: String
}

input OrderItemInput {
  productId: ID!
  quantity: Int!
}

input AddressInput {
  postalCode: String!
  prefecture: String!
  city: String!
  street: String!
  building: String
}
```

### 2.6 Interface と Union

```graphql
# Interface: 共通フィールドを持つ型の抽象化
interface Node {
  id: ID!
}

interface Timestamped {
  createdAt: DateTime!
  updatedAt: DateTime!
}

type User implements Node & Timestamped {
  id: ID!
  name: String!
  email: Email!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Post implements Node & Timestamped {
  id: ID!
  title: String!
  body: String!
  createdAt: DateTime!
  updatedAt: DateTime!
}

# Union: 共通フィールドを持たない型の集合
union SearchResult = User | Post | Product

type Query {
  search(query: String!): [SearchResult!]!
  node(id: ID!): Node         # Relay Global Object Identificationパターン
}

# Unionのクエリ例
# query {
#   search(query: "GraphQL") {
#     ... on User { name, email }
#     ... on Post { title, body }
#     ... on Product { name, price }
#   }
# }
```

### 2.7 ディレクティブ

```graphql
# 組み込みディレクティブ
# @skip(if: Boolean!)    - trueの場合そのフィールドを除外
# @include(if: Boolean!) - trueの場合そのフィールドを含める
# @deprecated(reason: String) - フィールドの非推奨化

type User {
  id: ID!
  name: String!
  email: Email!
  username: String @deprecated(reason: "Use 'name' instead.")
}

# クエリでのディレクティブ使用
# query GetUser($id: ID!, $includeOrders: Boolean!) {
#   user(id: $id) {
#     name
#     email
#     orders @include(if: $includeOrders) {
#       id
#       total
#     }
#   }
# }

# カスタムディレクティブの定義（サーバー側で実装が必要）
directive @auth(requires: UserRole!) on FIELD_DEFINITION
directive @cacheControl(maxAge: Int!) on FIELD_DEFINITION
directive @rateLimit(max: Int!, window: String!) on FIELD_DEFINITION

type Query {
  users: [User!]! @auth(requires: ADMIN) @rateLimit(max: 100, window: "1m")
  publicPosts: [Post!]! @cacheControl(maxAge: 300)
}
```

### 2.8 ページネーション用型（Relay Connection仕様）

```graphql
# Relay Connection仕様に基づくページネーション

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type UserEdge {
  node: User!
  cursor: String!
}

type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type OrderEdge {
  node: Order!
  cursor: String!
}

type OrderConnection {
  edges: [OrderEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

# Cursor vs Offset ページネーション比較:
#
# Offset方式:  users(offset: 20, limit: 10)
#   利点: 実装が簡単、任意のページにジャンプ可能
#   欠点: データ挿入/削除時にずれが発生、大きなオフセットはDB負荷大
#
# Cursor方式:  users(first: 10, after: "abc123")
#   利点: データ変更に強い、一貫した結果、インデックス利用で高速
#   欠点: 任意ページジャンプ不可、実装がやや複雑
```

---

## 3. Query（データ取得）

### 3.1 Query型の定義

```graphql
# スキーマ定義
type Query {
  # 単一リソース
  user(id: ID!): User
  order(id: ID!): Order
  product(id: ID!): Product

  # コレクション（Cursor ページネーション）
  users(
    first: Int
    after: String
    last: Int
    before: String
    filter: UserFilter
    sort: UserSort
  ): UserConnection!

  orders(
    first: Int
    after: String
    filter: OrderFilter
  ): OrderConnection!

  # 検索
  searchUsers(query: String!, limit: Int = 10): [User!]!
  search(query: String!, types: [SearchType!]): [SearchResult!]!

  # 集計
  userStats: UserStats!
  orderStats(period: StatPeriod!): OrderStats!

  # ヘルスチェック
  health: HealthStatus!

  # 現在のログインユーザー
  me: User
}

input UserFilter {
  role: UserRole
  createdAfter: DateTime
  createdBefore: DateTime
  nameContains: String
}

input OrderFilter {
  status: OrderStatus
  minTotal: Int
  maxTotal: Int
  userId: ID
}

enum UserSort {
  CREATED_AT_ASC
  CREATED_AT_DESC
  NAME_ASC
  NAME_DESC
}

enum SearchType {
  USER
  POST
  PRODUCT
}

enum StatPeriod {
  TODAY
  THIS_WEEK
  THIS_MONTH
  THIS_YEAR
}

type UserStats {
  totalUsers: Int!
  activeUsers: Int!
  newUsersToday: Int!
  roleDistribution: [RoleCount!]!
}

type RoleCount {
  role: UserRole!
  count: Int!
}

type OrderStats {
  totalOrders: Int!
  totalRevenue: Int!
  averageOrderValue: Float!
  statusDistribution: [StatusCount!]!
}

type StatusCount {
  status: OrderStatus!
  count: Int!
}

type HealthStatus {
  status: String!
  uptime: Float!
  version: String!
}
```

### 3.2 クエリの書き方

```graphql
# クライアントからのクエリ例

# 基本的なクエリ
query GetUser {
  user(id: "123") {
    id
    name
    email
    role
  }
}

# ネストされたクエリ
query GetUserWithOrders {
  user(id: "123") {
    name
    orders(first: 5) {
      edges {
        node {
          id
          status
          total
          items {
            product {
              name
              price
            }
            quantity
            subtotal
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
      totalCount
    }
  }
}

# 変数を使ったクエリ
query GetUsers($first: Int!, $after: String, $role: UserRole) {
  users(first: $first, after: $after, filter: { role: $role }) {
    edges {
      node {
        id
        name
        email
        role
        createdAt
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
    totalCount
  }
}
# 変数: { "first": 20, "after": null, "role": "ADMIN" }

# エイリアス（同じフィールドを異なる引数で取得）
query CompareUsers {
  admin: user(id: "1") { name, role, orderCount }
  editor: user(id: "2") { name, role, orderCount }
}

# フラグメント（共通フィールドの再利用）
fragment UserBasic on User {
  id
  name
  email
  role
}

fragment OrderSummary on Order {
  id
  status
  total
  createdAt
}

query GetMultipleUsers {
  user1: user(id: "1") {
    ...UserBasic
    orderCount
    orders(first: 3) {
      edges {
        node { ...OrderSummary }
      }
    }
  }
  user2: user(id: "2") {
    ...UserBasic
    orderCount
    orders(first: 3) {
      edges {
        node { ...OrderSummary }
      }
    }
  }
}
```

### 3.3 インラインフラグメントとUnion型のクエリ

```graphql
# Union型に対するインラインフラグメント
query SearchAll($q: String!) {
  search(query: $q) {
    ... on User {
      __typename
      id
      name
      email
    }
    ... on Post {
      __typename
      id
      title
      body
    }
    ... on Product {
      __typename
      id
      name
      price
    }
  }
}

# __typename はオブジェクトの型名を返す特殊フィールド
# レスポンス例:
# {
#   "data": {
#     "search": [
#       { "__typename": "User", "id": "1", "name": "Taro", "email": "..." },
#       { "__typename": "Post", "id": "10", "title": "GraphQL入門", "body": "..." },
#       { "__typename": "Product", "id": "100", "name": "GraphQL本", "price": 3000 }
#     ]
#   }
# }
```

---

## 4. Mutation（データ変更）

### 4.1 Mutation型の定義

```graphql
# スキーマ定義
type Mutation {
  # ユーザー
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!

  # ユーザー認証
  signUp(input: SignUpInput!): AuthPayload!
  signIn(email: Email!, password: String!): AuthPayload!
  refreshToken(token: String!): AuthPayload!

  # 注文
  createOrder(input: CreateOrderInput!): CreateOrderPayload!
  updateOrderStatus(id: ID!, status: OrderStatus!): UpdateOrderPayload!
  cancelOrder(id: ID!): CancelOrderPayload!

  # 商品
  createProduct(input: CreateProductInput!): CreateProductPayload!
  updateProduct(id: ID!, input: UpdateProductInput!): UpdateProductPayload!
  deleteProduct(id: ID!): DeleteProductPayload!
}

# Payload パターン（成功/エラーを表現）
type CreateUserPayload {
  user: User
  errors: [UserError!]!
}

type UpdateUserPayload {
  user: User
  errors: [UserError!]!
}

type DeleteUserPayload {
  deletedId: ID
  errors: [UserError!]!
}

type AuthPayload {
  token: String
  user: User
  errors: [UserError!]!
}

type CreateOrderPayload {
  order: Order
  errors: [UserError!]!
}

type UpdateOrderPayload {
  order: Order
  errors: [UserError!]!
}

type CancelOrderPayload {
  order: Order
  errors: [UserError!]!
}

# エラー型
type UserError {
  field: String               # エラーが発生したフィールド名
  message: String!            # 人間が読めるメッセージ
  code: ErrorCode!            # 機械処理用のエラーコード
}

enum ErrorCode {
  NOT_FOUND
  VALIDATION_ERROR
  ALREADY_EXISTS
  UNAUTHORIZED
  FORBIDDEN
  INTERNAL_ERROR
  RATE_LIMITED
  INVALID_INPUT
}

input SignUpInput {
  name: String!
  email: Email!
  password: String!
}

input CreateProductInput {
  name: String!
  price: Int!
  description: String
  categoryId: ID!
  tags: [String!]
  imageUrl: URL
  stock: Int!
}

input UpdateProductInput {
  name: String
  price: Int
  description: String
  categoryId: ID
  tags: [String!]
  imageUrl: URL
  stock: Int
}
```

### 4.2 Mutationの実行例

```graphql
# ユーザー作成
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    user {
      id
      name
      email
      role
      createdAt
    }
    errors {
      field
      message
      code
    }
  }
}
# 変数: { "input": { "name": "Taro", "email": "taro@example.com" } }

# 成功レスポンス:
# {
#   "data": {
#     "createUser": {
#       "user": {
#         "id": "456",
#         "name": "Taro",
#         "email": "taro@example.com",
#         "role": "USER",
#         "createdAt": "2024-01-15T10:30:00Z"
#       },
#       "errors": []
#     }
#   }
# }

# エラーレスポンス:
# {
#   "data": {
#     "createUser": {
#       "user": null,
#       "errors": [
#         {
#           "field": "email",
#           "message": "Email already exists",
#           "code": "ALREADY_EXISTS"
#         }
#       ]
#     }
#   }
# }
```

```graphql
# 認証（サインイン）
mutation SignIn($email: Email!, $password: String!) {
  signIn(email: $email, password: $password) {
    token
    user {
      id
      name
      role
    }
    errors {
      message
      code
    }
  }
}

# 注文作成
mutation CreateOrder($input: CreateOrderInput!) {
  createOrder(input: $input) {
    order {
      id
      status
      total
      items {
        product { name }
        quantity
        unitPrice
        subtotal
      }
    }
    errors {
      field
      message
      code
    }
  }
}
# 変数:
# {
#   "input": {
#     "items": [
#       { "productId": "prod-1", "quantity": 2 },
#       { "productId": "prod-2", "quantity": 1 }
#     ],
#     "shippingAddress": {
#       "postalCode": "100-0001",
#       "prefecture": "東京都",
#       "city": "千代田区",
#       "street": "丸の内1-1-1"
#     }
#   }
# }
```

### 4.3 Mutationの設計原則

```
Mutation設計の5原則:

  1. 入力はInput型にまとめる
     ✗ createUser(name: String!, email: String!, role: UserRole!)
     ○ createUser(input: CreateUserInput!)
     → 引数追加時にクエリを変更せずInput型の拡張だけで済む

  2. 戻り値はPayload型で統一する
     ✗ createUser(input: ...): User!      ← エラー情報なし
     ○ createUser(input: ...): CreateUserPayload!
     → 成功時のデータとエラー情報を同一レスポンスで返す

  3. 冪等性を意識する
     → 同じMutationを複数回実行しても結果が同じ
     → クライアントIDやリクエストIDで重複排除

  4. 命名は動詞 + 名詞
     ○ createUser, updateOrder, cancelSubscription
     ✗ userCreate, orderUpdate

  5. 1つのMutationで1つの操作
     ✗ updateUserAndCreateOrder(...)
     ○ updateUser(...) + createOrder(...) を別々に
```

---

## 5. Subscription（リアルタイム通知）

### 5.1 Subscriptionの仕組み

```
┌──────────────────────────────────────────────────────────────┐
│                 Subscription のフロー                         │
│                                                              │
│  Client                    Server                PubSub      │
│    │                         │                      │        │
│    │  subscription {         │                      │        │
│    │    orderUpdated(userId)  │                      │        │
│    │  }                      │                      │        │
│    │ ───WebSocket接続───→    │                      │        │
│    │                         │  subscribe(topic) →  │        │
│    │                         │                      │        │
│    │         ... 待機中 ...   │                      │        │
│    │                         │                      │        │
│    │                         │  ← publish(topic,    │        │
│    │                         │      payload)        │        │
│    │  ← { data: {           │                      │        │
│    │       orderUpdated: {   │                      │        │
│    │         id, status      │                      │        │
│    │       }                 │                      │        │
│    │     }                   │                      │        │
│    │    }                    │                      │        │
│    │                         │  ← publish(...)      │        │
│    │  ← { data: {...} }     │                      │        │
│    │                         │                      │        │
│    │  unsubscribe           │                      │        │
│    │ ───WebSocket切断───→    │                      │        │
│    │                         │                      │        │
└──────────────────────────────────────────────────────────────┘
```

### 5.2 Subscription型の定義

```graphql
type Subscription {
  # 注文ステータスの変更を購読
  orderStatusChanged(userId: ID!): OrderStatusEvent!

  # 新しいメッセージの購読（チャット機能）
  messageSent(channelId: ID!): Message!

  # 商品在庫の変更
  stockUpdated(productId: ID!): StockEvent!

  # 全体通知
  notificationReceived(userId: ID!): Notification!
}

type OrderStatusEvent {
  order: Order!
  previousStatus: OrderStatus!
  newStatus: OrderStatus!
  changedAt: DateTime!
}

type Message {
  id: ID!
  sender: User!
  content: String!
  sentAt: DateTime!
}

type StockEvent {
  product: Product!
  previousStock: Int!
  newStock: Int!
  changedAt: DateTime!
}

type Notification {
  id: ID!
  type: NotificationType!
  title: String!
  message: String!
  createdAt: DateTime!
}

enum NotificationType {
  ORDER_UPDATE
  PROMOTION
  SYSTEM
  MENTION
}
```

### 5.3 Subscriptionリゾルバーの実装

```javascript
// PubSubを使ったSubscriptionリゾルバー
import { PubSub } from 'graphql-subscriptions';

const pubsub = new PubSub();

// イベント名の定数
const EVENTS = {
  ORDER_STATUS_CHANGED: 'ORDER_STATUS_CHANGED',
  MESSAGE_SENT: 'MESSAGE_SENT',
  STOCK_UPDATED: 'STOCK_UPDATED',
  NOTIFICATION: 'NOTIFICATION',
};

const resolvers = {
  Subscription: {
    orderStatusChanged: {
      // subscribe関数がAsyncIteratorを返す
      subscribe: (_, { userId }) => {
        return pubsub.asyncIterator(
          `${EVENTS.ORDER_STATUS_CHANGED}.${userId}`
        );
      },
      // resolve関数でペイロードを変換（オプション）
      resolve: (payload) => payload,
    },

    messageSent: {
      subscribe: (_, { channelId }, context) => {
        // 認証チェック
        if (!context.user) {
          throw new Error('Authentication required');
        }
        return pubsub.asyncIterator(
          `${EVENTS.MESSAGE_SENT}.${channelId}`
        );
      },
    },

    notificationReceived: {
      subscribe: (_, { userId }, context) => {
        if (context.user?.id !== userId) {
          throw new Error('Cannot subscribe to other user notifications');
        }
        return pubsub.asyncIterator(
          `${EVENTS.NOTIFICATION}.${userId}`
        );
      },
    },
  },

  Mutation: {
    updateOrderStatus: async (_, { id, status }, context) => {
      const order = await context.dataSources.orderAPI.updateStatus(id, status);

      // Subscriptionにイベントを発行
      pubsub.publish(`${EVENTS.ORDER_STATUS_CHANGED}.${order.userId}`, {
        orderStatusChanged: {
          order,
          previousStatus: order.previousStatus,
          newStatus: status,
          changedAt: new Date().toISOString(),
        },
      });

      return { order, errors: [] };
    },
  },
};
```

### 5.4 クライアントでのSubscription利用

```javascript
// Apollo Client でのSubscription利用
import {
  ApolloClient,
  InMemoryCache,
  split,
  HttpLink,
} from '@apollo/client';
import { GraphQLWsLink } from '@apollo/client/link/subscriptions';
import { createClient } from 'graphql-ws';
import { getMainDefinition } from '@apollo/client/utilities';

// HTTP接続（Query/Mutation用）
const httpLink = new HttpLink({
  uri: 'http://localhost:4000/graphql',
});

// WebSocket接続（Subscription用）
const wsLink = new GraphQLWsLink(
  createClient({
    url: 'ws://localhost:4000/graphql',
    connectionParams: {
      authToken: localStorage.getItem('token'),
    },
  })
);

// オペレーションタイプに応じてリンクを切り替え
const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    );
  },
  wsLink,   // Subscriptionの場合
  httpLink  // Query/Mutationの場合
);

const client = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
});

// Reactコンポーネントでの使用
import { useSubscription, gql } from '@apollo/client';

const ORDER_STATUS_SUBSCRIPTION = gql`
  subscription OnOrderStatusChanged($userId: ID!) {
    orderStatusChanged(userId: $userId) {
      order {
        id
        status
        total
      }
      previousStatus
      newStatus
      changedAt
    }
  }
`;

function OrderTracker({ userId }) {
  const { data, loading, error } = useSubscription(
    ORDER_STATUS_SUBSCRIPTION,
    { variables: { userId } }
  );

  if (loading) return <p>注文状態を監視中...</p>;
  if (error) return <p>接続エラー: {error.message}</p>;

  if (data) {
    const { order, previousStatus, newStatus } = data.orderStatusChanged;
    return (
      <div>
        <p>注文 #{order.id} のステータスが変更されました</p>
        <p>{previousStatus} → {newStatus}</p>
      </div>
    );
  }

  return <p>更新待ち...</p>;
}
```

---

## 6. リゾルバー実装

### 6.1 リゾルバーの基本構造

```
リゾルバーの4つの引数:

  resolver(parent, args, context, info)

  parent  : 親フィールドのリゾルバーが返した値
            （ルートリゾルバーではundefined）
  args    : クエリで渡された引数
  context : リクエスト全体で共有されるオブジェクト
            （認証情報、DataSource、DataLoaderなど）
  info    : クエリのAST情報
            （フィールド名、パス、選択セットなど）

  ┌──────────────────────────────────────────────────────┐
  │  リゾルバーチェーン（実行順序）                         │
  │                                                      │
  │  query {                                             │
  │    user(id: "1") {        ← Query.user リゾルバー     │
  │      name                 ← デフォルトリゾルバー        │
  │      orders {             ← User.orders リゾルバー     │
  │        items {            ← Order.items リゾルバー     │
  │          product {        ← OrderItem.product リゾルバー│
  │            name           ← デフォルトリゾルバー        │
  │          }                                            │
  │        }                                              │
  │      }                                                │
  │    }                                                  │
  │  }                                                    │
  │                                                      │
  │  デフォルトリゾルバー:                                  │
  │    parent[fieldName] を返す                            │
  │    → parentオブジェクトに同名プロパティがあれば自動解決  │
  └──────────────────────────────────────────────────────┘
```

### 6.2 リゾルバーの実装

```javascript
// Apollo Server でのリゾルバー実装
import { GraphQLScalarType, Kind } from 'graphql';

const resolvers = {
  // === ルートリゾルバー ===
  Query: {
    // 単一ユーザー取得
    user: async (parent, { id }, context) => {
      // 認証チェック
      if (!context.user) {
        throw new AuthenticationError('ログインが必要です');
      }
      const user = await context.dataSources.userAPI.getUser(id);
      if (!user) return null;
      return user;
    },

    // ユーザー一覧（Cursorページネーション）
    users: async (parent, { first = 20, after, filter, sort }, context) => {
      const { nodes, totalCount, hasNextPage, hasPreviousPage } =
        await context.dataSources.userAPI.listUsers({
          first,
          after,
          filter,
          sort,
        });

      const edges = nodes.map((node) => ({
        node,
        cursor: Buffer.from(`cursor:${node.id}`).toString('base64'),
      }));

      return {
        edges,
        pageInfo: {
          hasNextPage,
          hasPreviousPage,
          startCursor: edges[0]?.cursor ?? null,
          endCursor: edges[edges.length - 1]?.cursor ?? null,
        },
        totalCount,
      };
    },

    // 検索
    search: async (parent, { query, types }, context) => {
      const results = [];

      if (!types || types.includes('USER')) {
        const users = await context.dataSources.userAPI.search(query);
        results.push(...users);
      }
      if (!types || types.includes('POST')) {
        const posts = await context.dataSources.postAPI.search(query);
        results.push(...posts);
      }
      if (!types || types.includes('PRODUCT')) {
        const products = await context.dataSources.productAPI.search(query);
        results.push(...products);
      }

      return results;
    },

    // 現在のユーザー
    me: (parent, args, context) => {
      return context.user || null;
    },
  },

  // === Mutationリゾルバー ===
  Mutation: {
    createUser: async (parent, { input }, context) => {
      try {
        // バリデーション
        if (!input.name || input.name.trim().length === 0) {
          return {
            user: null,
            errors: [{
              field: 'name',
              message: '名前は必須です',
              code: 'VALIDATION_ERROR',
            }],
          };
        }

        const user = await context.dataSources.userAPI.createUser(input);
        return { user, errors: [] };
      } catch (error) {
        if (error.code === 'DUPLICATE_EMAIL') {
          return {
            user: null,
            errors: [{
              field: 'email',
              message: '既に登録済みのメールアドレスです',
              code: 'ALREADY_EXISTS',
            }],
          };
        }
        return {
          user: null,
          errors: [{
            field: null,
            message: '予期しないエラーが発生しました',
            code: 'INTERNAL_ERROR',
          }],
        };
      }
    },

    updateUser: async (parent, { id, input }, context) => {
      // 認可チェック（自分自身またはADMINのみ）
      if (context.user.id !== id && context.user.role !== 'ADMIN') {
        return {
          user: null,
          errors: [{
            field: null,
            message: '他のユーザーの情報を変更する権限がありません',
            code: 'FORBIDDEN',
          }],
        };
      }

      try {
        const user = await context.dataSources.userAPI.updateUser(id, input);
        return { user, errors: [] };
      } catch (error) {
        return {
          user: null,
          errors: [{
            field: error.field || null,
            message: error.message,
            code: error.code || 'INTERNAL_ERROR',
          }],
        };
      }
    },

    deleteUser: async (parent, { id }, context) => {
      if (context.user.role !== 'ADMIN') {
        return {
          deletedId: null,
          errors: [{
            field: null,
            message: '管理者権限が必要です',
            code: 'FORBIDDEN',
          }],
        };
      }
      await context.dataSources.userAPI.deleteUser(id);
      return { deletedId: id, errors: [] };
    },
  },

  // === フィールドレベルリゾルバー ===
  User: {
    // user.orders は別テーブルから取得
    orders: async (user, { first = 10, after }, context) => {
      return context.dataSources.orderAPI.getOrdersByUserId(
        user.id, first, after
      );
    },

    // 計算フィールド
    orderCount: async (user, args, context) => {
      return context.dataSources.orderAPI.countByUserId(user.id);
    },

    // フォロワー
    followers: async (user, args, context) => {
      return context.dataSources.userAPI.getFollowers(user.id);
    },
  },

  Order: {
    // 注文に紐づくユーザー
    user: async (order, args, context) => {
      return context.dataSources.userAPI.getUser(order.userId);
    },

    items: async (order, args, context) => {
      return context.dataSources.orderAPI.getOrderItems(order.id);
    },
  },

  OrderItem: {
    product: async (item, args, context) => {
      return context.dataSources.productAPI.getProduct(item.productId);
    },

    // 計算フィールド
    subtotal: (item) => item.quantity * item.unitPrice,
  },

  // === Union型のリゾルバー ===
  SearchResult: {
    __resolveType(obj) {
      // オブジェクトの型を判定
      if (obj.email) return 'User';
      if (obj.body) return 'Post';
      if (obj.price !== undefined) return 'Product';
      return null;
    },
  },

  // === カスタムスカラー ===
  DateTime: new GraphQLScalarType({
    name: 'DateTime',
    description: 'ISO 8601形式の日時文字列',
    serialize(value) {
      return value instanceof Date ? value.toISOString() : value;
    },
    parseValue(value) {
      const date = new Date(value);
      if (isNaN(date.getTime())) {
        throw new Error('Invalid DateTime format');
      }
      return date;
    },
    parseLiteral(ast) {
      if (ast.kind === Kind.STRING) {
        const date = new Date(ast.value);
        if (isNaN(date.getTime())) {
          throw new Error('Invalid DateTime format');
        }
        return date;
      }
      return null;
    },
  }),

  Email: new GraphQLScalarType({
    name: 'Email',
    description: 'メールアドレス形式の文字列',
    serialize(value) {
      return value;
    },
    parseValue(value) {
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
      if (!emailRegex.test(value)) {
        throw new Error('Invalid email format');
      }
      return value.toLowerCase();
    },
    parseLiteral(ast) {
      if (ast.kind === Kind.STRING) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(ast.value)) {
          throw new Error('Invalid email format');
        }
        return ast.value.toLowerCase();
      }
      return null;
    },
  }),
};
```

---

## 7. Apollo Server セットアップ

### 7.1 基本セットアップ

```javascript
// server.js - Apollo Server v4
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { expressMiddleware } from '@apollo/server/express4';
import { readFileSync } from 'fs';
import express from 'express';
import cors from 'cors';
import http from 'http';

// スキーマファイルの読み込み
const typeDefs = readFileSync('./schema.graphql', 'utf-8');

// サーバー作成
const server = new ApolloServer({
  typeDefs,
  resolvers,
  // イントロスペクション（本番では無効推奨）
  introspection: process.env.NODE_ENV !== 'production',
  // プラグイン
  plugins: [
    // ランディングページ（開発環境でGraphQL Playgroundを表示）
    process.env.NODE_ENV === 'production'
      ? ApolloServerPluginLandingPageDisabled()
      : ApolloServerPluginLandingPageLocalDefault(),
    // レスポンスキャッシュ
    responseCachePlugin(),
    // ログプラグイン
    {
      async requestDidStart(requestContext) {
        const start = Date.now();
        return {
          async willSendResponse(ctx) {
            const duration = Date.now() - start;
            console.log(
              `[GraphQL] ${ctx.operation?.operation} ` +
              `${ctx.operation?.name?.value || 'anonymous'} ` +
              `${duration}ms`
            );
          },
          async didEncounterErrors(ctx) {
            for (const err of ctx.errors) {
              console.error('[GraphQL Error]', err.message, err.extensions);
            }
          },
        };
      },
    },
  ],
  // フォーマットエラー（本番ではスタックトレースを隠す）
  formatError: (formattedError, error) => {
    if (process.env.NODE_ENV === 'production') {
      // 内部エラーの詳細を隠蔽
      if (formattedError.extensions?.code === 'INTERNAL_SERVER_ERROR') {
        return {
          message: 'Internal server error',
          extensions: { code: 'INTERNAL_SERVER_ERROR' },
        };
      }
    }
    return formattedError;
  },
});

// スタンドアロンモード（最もシンプル）
const { url } = await startStandaloneServer(server, {
  listen: { port: 4000 },
  context: async ({ req }) => ({
    // 認証
    user: await authenticateUser(req.headers.authorization),
    // データソース
    dataSources: {
      userAPI: new UserAPI(),
      orderAPI: new OrderAPI(),
      productAPI: new ProductAPI(),
    },
  }),
});

console.log(`GraphQL server ready at ${url}`);
```

### 7.2 Express統合セットアップ

```javascript
// express-server.js - Express + Apollo Server + WebSocket (Subscription対応)
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { ApolloServerPluginDrainHttpServer } from '@apollo/server/plugin/drainHttpServer';
import { makeExecutableSchema } from '@graphql-tools/schema';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import express from 'express';
import http from 'http';
import cors from 'cors';
import bodyParser from 'body-parser';

// スキーマ作成
const schema = makeExecutableSchema({ typeDefs, resolvers });

// Express + HTTPサーバー
const app = express();
const httpServer = http.createServer(app);

// WebSocketサーバー（Subscription用）
const wsServer = new WebSocketServer({
  server: httpServer,
  path: '/graphql',
});

// WebSocketの終了処理を設定
const serverCleanup = useServer(
  {
    schema,
    context: async (ctx) => {
      // WebSocket接続時の認証
      const token = ctx.connectionParams?.authToken;
      const user = await authenticateToken(token);
      return {
        user,
        dataSources: {
          userAPI: new UserAPI(),
          orderAPI: new OrderAPI(),
        },
      };
    },
    onConnect: async (ctx) => {
      console.log('WebSocket client connected');
    },
    onDisconnect: (ctx) => {
      console.log('WebSocket client disconnected');
    },
  },
  wsServer
);

// Apollo Server
const server = new ApolloServer({
  schema,
  plugins: [
    // HTTPサーバーの正常終了
    ApolloServerPluginDrainHttpServer({ httpServer }),
    // WebSocketサーバーの正常終了
    {
      async serverWillStart() {
        return {
          async drainServer() {
            await serverCleanup.dispose();
          },
        };
      },
    },
  ],
});

await server.start();

// Expressミドルウェアとして設定
app.use(
  '/graphql',
  cors(),
  bodyParser.json(),
  expressMiddleware(server, {
    context: async ({ req }) => ({
      user: await authenticateUser(req.headers.authorization),
      dataSources: {
        userAPI: new UserAPI(),
        orderAPI: new OrderAPI(),
      },
    }),
  })
);

// ヘルスチェックエンドポイント（REST）
app.get('/health', (req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() });
});

const PORT = process.env.PORT || 4000;
httpServer.listen(PORT, () => {
  console.log(`Server ready at http://localhost:${PORT}/graphql`);
  console.log(`Subscriptions ready at ws://localhost:${PORT}/graphql`);
});
```

### 7.3 認証・認可パターン

```javascript
// auth.js - 認証・認可ユーティリティ

import jwt from 'jsonwebtoken';

// JWTトークンからユーザーを取得
async function authenticateUser(authHeader) {
  if (!authHeader) return null;

  const token = authHeader.replace('Bearer ', '');
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await UserModel.findById(decoded.userId);
    return user;
  } catch (error) {
    return null; // トークン無効でもnullを返す（エラーにしない）
  }
}

// ディレクティブベースの認可
import { mapSchema, getDirective, MapperKind } from '@graphql-tools/utils';
import { defaultFieldResolver } from 'graphql';

function authDirectiveTransformer(schema) {
  return mapSchema(schema, {
    [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
      const authDirective = getDirective(schema, fieldConfig, 'auth')?.[0];
      if (authDirective) {
        const { requires } = authDirective;
        const originalResolver = fieldConfig.resolve || defaultFieldResolver;

        fieldConfig.resolve = async (parent, args, context, info) => {
          // 認証チェック
          if (!context.user) {
            throw new Error('認証が必要です');
          }

          // 認可チェック
          if (requires && context.user.role !== requires) {
            throw new Error(
              `この操作には${requires}権限が必要です`
            );
          }

          return originalResolver(parent, args, context, info);
        };
      }
      return fieldConfig;
    },
  });
}

// 使用例（スキーマ変換）
let schema = makeExecutableSchema({ typeDefs, resolvers });
schema = authDirectiveTransformer(schema);
```

### 7.4 DataSourceパターン

```javascript
// data-sources/user-api.js
// RESTDataSourceを使ったデータソースの実装

import { RESTDataSource } from '@apollo/datasource-rest';

class UserAPI extends RESTDataSource {
  constructor() {
    super();
    this.baseURL = 'http://internal-api:3000/';
  }

  // キャッシュTTLの設定
  override cacheOptionsFor() {
    return { ttl: 60 }; // 60秒キャッシュ
  }

  async getUser(id) {
    return this.get(`users/${id}`);
  }

  async listUsers({ first, after, filter, sort }) {
    const params = { limit: first };
    if (after) params.cursor = after;
    if (filter?.role) params.role = filter.role;
    if (sort) params.sort = sort;

    return this.get('users', { params });
  }

  async createUser(input) {
    return this.post('users', { body: input });
  }

  async updateUser(id, input) {
    return this.patch(`users/${id}`, { body: input });
  }

  async deleteUser(id) {
    return this.delete(`users/${id}`);
  }

  async search(query) {
    const results = await this.get('users/search', {
      params: { q: query },
    });
    return results.map((user) => ({ ...user, __typename: 'User' }));
  }
}

// data-sources/database-source.js
// SQLデータベースを直接利用するデータソース

class DatabaseSource {
  constructor(pool) {
    this.pool = pool; // データベース接続プール
  }

  async getUser(id) {
    const { rows } = await this.pool.query(
      'SELECT * FROM users WHERE id = $1',
      [id]
    );
    return rows[0] || null;
  }

  async listUsers({ first, after, filter, sort }) {
    let query = 'SELECT * FROM users WHERE 1=1';
    const params = [];
    let paramIndex = 1;

    if (after) {
      const decodedCursor = Buffer.from(after, 'base64')
        .toString('utf-8')
        .replace('cursor:', '');
      query += ` AND id > $${paramIndex++}`;
      params.push(decodedCursor);
    }

    if (filter?.role) {
      query += ` AND role = $${paramIndex++}`;
      params.push(filter.role);
    }

    if (filter?.nameContains) {
      query += ` AND name ILIKE $${paramIndex++}`;
      params.push(`%${filter.nameContains}%`);
    }

    // ソート
    const sortMap = {
      CREATED_AT_ASC: 'created_at ASC',
      CREATED_AT_DESC: 'created_at DESC',
      NAME_ASC: 'name ASC',
      NAME_DESC: 'name DESC',
    };
    query += ` ORDER BY ${sortMap[sort] || 'created_at DESC'}`;

    // ページネーション（+1で次ページの有無を判定）
    query += ` LIMIT $${paramIndex++}`;
    params.push(first + 1);

    const { rows } = await this.pool.query(query, params);
    const hasNextPage = rows.length > first;
    const nodes = hasNextPage ? rows.slice(0, first) : rows;

    return {
      nodes,
      totalCount: await this.countUsers(filter),
      hasNextPage,
      hasPreviousPage: !!after,
    };
  }

  async countUsers(filter) {
    let query = 'SELECT COUNT(*) FROM users WHERE 1=1';
    const params = [];
    let paramIndex = 1;

    if (filter?.role) {
      query += ` AND role = $${paramIndex++}`;
      params.push(filter.role);
    }

    const { rows } = await this.pool.query(query, params);
    return parseInt(rows[0].count, 10);
  }
}
```

---

## 8. N+1問題とDataLoader

### 8.1 N+1問題とは

```
N+1問題のイメージ:

  query {
    users(first: 10) {        ← 1回のSQLクエリ（ユーザー10件取得）
      edges {
        node {
          name
          orders {             ← ユーザーごとに1回のSQLクエリ（×10回）
            id
            total
          }
        }
      }
    }
  }

  実行されるSQL:
    1. SELECT * FROM users LIMIT 10              -- 1回
    2. SELECT * FROM orders WHERE user_id = 1    -- User 1の注文
    3. SELECT * FROM orders WHERE user_id = 2    -- User 2の注文
    4. SELECT * FROM orders WHERE user_id = 3    -- User 3の注文
    ...
   11. SELECT * FROM orders WHERE user_id = 10   -- User 10の注文

  → 合計 11回のDBクエリ（1 + N = 1 + 10 = 11）

  DataLoader で解決:
    1. SELECT * FROM users LIMIT 10              -- 1回
    2. SELECT * FROM orders WHERE user_id IN (1,2,3,...,10) -- 1回

  → 合計 2回のDBクエリ
```

### 8.2 DataLoaderの実装

```javascript
// data-loaders.js
import DataLoader from 'dataloader';

// DataLoaderファクトリ（リクエストごとに新しいインスタンスを作成）
function createLoaders(db) {
  return {
    // ユーザーローダー
    userLoader: new DataLoader(async (userIds) => {
      // バッチ関数: IDの配列を受け取り、同じ順序で結果を返す
      const users = await db.query(
        'SELECT * FROM users WHERE id = ANY($1)',
        [userIds]
      );

      // IDの順序を保持してマッピング
      const userMap = new Map(users.rows.map((u) => [u.id, u]));
      return userIds.map((id) => userMap.get(id) || null);
    }),

    // ユーザーの注文ローダー（1:Nの関係）
    ordersByUserIdLoader: new DataLoader(async (userIds) => {
      const orders = await db.query(
        'SELECT * FROM orders WHERE user_id = ANY($1) ORDER BY created_at DESC',
        [userIds]
      );

      // ユーザーIDごとにグループ化
      const orderMap = new Map();
      for (const order of orders.rows) {
        if (!orderMap.has(order.user_id)) {
          orderMap.set(order.user_id, []);
        }
        orderMap.get(order.user_id).push(order);
      }

      return userIds.map((id) => orderMap.get(id) || []);
    }),

    // 商品ローダー
    productLoader: new DataLoader(async (productIds) => {
      const products = await db.query(
        'SELECT * FROM products WHERE id = ANY($1)',
        [productIds]
      );

      const productMap = new Map(products.rows.map((p) => [p.id, p]));
      return productIds.map((id) => productMap.get(id) || null);
    }),

    // 注文アイテムローダー（1:Nの関係）
    orderItemsByOrderIdLoader: new DataLoader(async (orderIds) => {
      const items = await db.query(
        'SELECT * FROM order_items WHERE order_id = ANY($1)',
        [orderIds]
      );

      const itemMap = new Map();
      for (const item of items.rows) {
        if (!itemMap.has(item.order_id)) {
          itemMap.set(item.order_id, []);
        }
        itemMap.get(item.order_id).push(item);
      }

      return orderIds.map((id) => itemMap.get(id) || []);
    }),
  };
}

// コンテキストでDataLoaderを設定
const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => ({
    user: await authenticateUser(req.headers.authorization),
    // リクエストごとに新しいLoaderを作成（キャッシュはリクエストスコープ）
    loaders: createLoaders(db),
    db,
  }),
});

// リゾルバーでDataLoaderを使用
const resolversWithLoader = {
  Query: {
    user: (_, { id }, { loaders }) => loaders.userLoader.load(id),
  },
  User: {
    orders: (user, _, { loaders }) =>
      loaders.ordersByUserIdLoader.load(user.id),
  },
  Order: {
    user: (order, _, { loaders }) => loaders.userLoader.load(order.userId),
    items: (order, _, { loaders }) =>
      loaders.orderItemsByOrderIdLoader.load(order.id),
  },
  OrderItem: {
    product: (item, _, { loaders }) =>
      loaders.productLoader.load(item.productId),
  },
};
```

### 8.3 DataLoaderの注意点

```
DataLoader使用時の注意:

  1. リクエストスコープで作成する
     ✗ グローバルにDataLoaderを1つだけ作成
       → キャッシュが他のリクエストに漏れる（セキュリティリスク）
     ○ コンテキスト生成時にリクエストごとに新規作成

  2. バッチ関数は入力と同じ順序で結果を返す
     ✗ [id=3, id=1, id=2] → [user1, user2, user3]  （ID順）
     ○ [id=3, id=1, id=2] → [user3, user1, user2]  （入力順）

  3. 見つからないキーにはnullを返す（エラーではなく）
     ✗ throw new Error('User not found')
     ○ return null

  4. キャッシュの無効化
     → Mutation後に loader.clear(id) または loader.clearAll()
     → 更新されたデータを再読み込みするために必要

  5. バッチサイズの制限
     → maxBatchSize オプションで設定可能
     → DBの IN句制限に合わせる（PostgreSQLは約65000パラメータ）
```

---

## 9. クライアント実装

### 9.1 Apollo Client セットアップ

```javascript
// apollo-client.js
import {
  ApolloClient,
  InMemoryCache,
  ApolloLink,
  from,
} from '@apollo/client';
import { onError } from '@apollo/client/link/error';
import { RetryLink } from '@apollo/client/link/retry';

// エラーハンドリングリンク
const errorLink = onError(({ graphQLErrors, networkError, operation }) => {
  if (graphQLErrors) {
    graphQLErrors.forEach(({ message, locations, path, extensions }) => {
      console.error(
        `[GraphQL Error] Message: ${message}, ` +
        `Location: ${JSON.stringify(locations)}, ` +
        `Path: ${path}, Code: ${extensions?.code}`
      );

      // 認証エラー時はログアウト処理
      if (extensions?.code === 'UNAUTHENTICATED') {
        localStorage.removeItem('token');
        window.location.href = '/login';
      }
    });
  }

  if (networkError) {
    console.error(`[Network Error] ${networkError}`);
  }
});

// リトライリンク
const retryLink = new RetryLink({
  delay: {
    initial: 300,
    max: 3000,
    jitter: true,
  },
  attempts: {
    max: 3,
    retryIf: (error) => !!error,
  },
});

// 認証リンク（リクエストヘッダーにトークンを付与）
const authLink = new ApolloLink((operation, forward) => {
  const token = localStorage.getItem('token');
  operation.setContext({
    headers: {
      Authorization: token ? `Bearer ${token}` : '',
    },
  });
  return forward(operation);
});

// キャッシュ設定
const cache = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        // ページネーションのマージポリシー
        users: {
          keyArgs: ['filter', 'sort'],
          merge(existing, incoming, { args }) {
            if (!args?.after) return incoming;
            return {
              ...incoming,
              edges: [...(existing?.edges || []), ...incoming.edges],
            };
          },
        },
      },
    },
    User: {
      // ユーザーのキャッシュキー
      keyFields: ['id'],
    },
    Order: {
      keyFields: ['id'],
    },
  },
});

// クライアント作成
const client = new ApolloClient({
  link: from([errorLink, retryLink, authLink, httpLink]),
  cache,
  defaultOptions: {
    watchQuery: {
      fetchPolicy: 'cache-and-network', // キャッシュ優先で最新も取得
      errorPolicy: 'all',
    },
    query: {
      fetchPolicy: 'network-only',
      errorPolicy: 'all',
    },
    mutate: {
      errorPolicy: 'all',
    },
  },
});
```

### 9.2 Reactコンポーネントでの使用

```javascript
// components/UserList.jsx
import { useQuery, useMutation, gql } from '@apollo/client';

// クエリ定義
const GET_USERS = gql`
  query GetUsers($first: Int!, $after: String, $filter: UserFilter) {
    users(first: $first, after: $after, filter: $filter) {
      edges {
        node {
          id
          name
          email
          role
          createdAt
          orderCount
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
      totalCount
    }
  }
`;

const DELETE_USER = gql`
  mutation DeleteUser($id: ID!) {
    deleteUser(id: $id) {
      deletedId
      errors {
        message
        code
      }
    }
  }
`;

function UserList() {
  const { loading, error, data, fetchMore } = useQuery(GET_USERS, {
    variables: { first: 20 },
  });

  const [deleteUser] = useMutation(DELETE_USER, {
    // Mutation後のキャッシュ更新
    update(cache, { data: { deleteUser: result } }) {
      if (result.deletedId) {
        cache.modify({
          fields: {
            users(existingConnection, { readField }) {
              return {
                ...existingConnection,
                edges: existingConnection.edges.filter(
                  (edge) => readField('id', edge.node) !== result.deletedId
                ),
                totalCount: existingConnection.totalCount - 1,
              };
            },
          },
        });
      }
    },
  });

  if (loading && !data) return <p>読み込み中...</p>;
  if (error) return <p>エラー: {error.message}</p>;

  const { edges, pageInfo, totalCount } = data.users;

  return (
    <div>
      <h1>ユーザー一覧 ({totalCount}件)</h1>
      <ul>
        {edges.map(({ node }) => (
          <li key={node.id}>
            {node.name} ({node.email}) - {node.role}
            <span>注文数: {node.orderCount}</span>
            <button onClick={() => deleteUser({ variables: { id: node.id } })}>
              削除
            </button>
          </li>
        ))}
      </ul>
      {pageInfo.hasNextPage && (
        <button
          onClick={() =>
            fetchMore({
              variables: { after: pageInfo.endCursor },
            })
          }
        >
          もっと読み込む
        </button>
      )}
    </div>
  );
}

// components/UserProfile.jsx
const GET_USER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      email
      role
      bio
      createdAt
      orders(first: 5) {
        edges {
          node {
            id
            status
            total
            createdAt
            items {
              product { name, price }
              quantity
              subtotal
            }
          }
        }
        pageInfo {
          hasNextPage
          endCursor
        }
      }
    }
  }
`;

function UserProfile({ userId }) {
  const { loading, error, data } = useQuery(GET_USER, {
    variables: { id: userId },
    // ポーリング（10秒ごとに自動更新）
    // pollInterval: 10000,
  });

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;
  if (!data.user) return <p>ユーザーが見つかりません</p>;

  const { user } = data;
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      <p>役割: {user.role}</p>
      {user.bio && <p>{user.bio}</p>}
      <h2>注文履歴</h2>
      {user.orders.edges.map(({ node }) => (
        <div key={node.id}>
          <h3>注文 #{node.id}</h3>
          <p>ステータス: {node.status}</p>
          <p>合計: {node.total.toLocaleString()}円</p>
          <ul>
            {node.items.map((item, i) => (
              <li key={i}>
                {item.product.name} x {item.quantity} = {item.subtotal.toLocaleString()}円
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
```
