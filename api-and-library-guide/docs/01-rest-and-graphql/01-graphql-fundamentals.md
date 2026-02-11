# GraphQL基礎

> GraphQLはFacebookが開発したクエリ言語。スキーマ駆動開発、型システム、Query/Mutation/Subscription、リゾルバーの仕組みを理解し、REST APIとは異なるアプローチでのAPI設計を習得する。

## この章で学ぶこと

- [ ] GraphQLの型システムとスキーマ定義を理解する
- [ ] Query・Mutation・Subscriptionの使い分けを把握する
- [ ] リゾルバーの実装パターンを学ぶ

---

## 1. GraphQLとは

```
GraphQL = Graph Query Language
  → API のためのクエリ言語 + 型システム + ランタイム
  → Facebook が 2012年に内部開発、2015年に公開

特徴:
  ① クライアントが必要なデータを正確に指定:
     → Over-fetching（不要データの取得）を排除
     → Under-fetching（追加リクエスト）を排除

  ② 単一エンドポイント:
     POST /graphql
     → RESTの複数エンドポイントが1つに

  ③ 強い型システム:
     → スキーマ定義 = API仕様書
     → バリデーション、補完、ドキュメント自動生成

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

---

## 2. スキーマ定義（SDL）

```graphql
# Schema Definition Language (SDL)

# スカラー型（組み込み）
# Int, Float, String, Boolean, ID

# カスタムスカラー
scalar DateTime
scalar Email

# 列挙型
enum UserRole {
  USER
  ADMIN
  EDITOR
}

enum OrderStatus {
  PENDING
  PROCESSING
  SHIPPED
  DELIVERED
  CANCELLED
}

# オブジェクト型
type User {
  id: ID!                     # !は非null
  name: String!
  email: Email!
  role: UserRole!
  avatar: String              # nullable
  createdAt: DateTime!
  orders: [Order!]!           # 非nullの配列（配列自体も非null）
  orderCount: Int!
}

type Order {
  id: ID!
  user: User!
  status: OrderStatus!
  total: Int!                 # 金額（円）
  items: [OrderItem!]!
  createdAt: DateTime!
}

type OrderItem {
  id: ID!
  product: Product!
  quantity: Int!
  unitPrice: Int!
}

type Product {
  id: ID!
  name: String!
  price: Int!
  description: String
  category: String!
}

# 入力型（Mutation の引数に使用）
input CreateUserInput {
  name: String!
  email: Email!
  role: UserRole = USER       # デフォルト値
}

input UpdateUserInput {
  name: String
  email: Email
  role: UserRole
}

# ページネーション用の型
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
```

---

## 3. Query（データ取得）

```graphql
# スキーマ定義
type Query {
  # 単一リソース
  user(id: ID!): User
  order(id: ID!): Order

  # コレクション（Cursor ページネーション）
  users(
    first: Int
    after: String
    filter: UserFilter
    sort: UserSort
  ): UserConnection!

  # 検索
  searchUsers(query: String!, limit: Int = 10): [User!]!

  # 集計
  userStats: UserStats!
}

input UserFilter {
  role: UserRole
  createdAfter: DateTime
  createdBefore: DateTime
}

enum UserSort {
  CREATED_AT_ASC
  CREATED_AT_DESC
  NAME_ASC
  NAME_DESC
}

type UserStats {
  totalUsers: Int!
  activeUsers: Int!
  newUsersToday: Int!
}
```

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
            }
            quantity
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

# 変数を使ったクエリ
query GetUsers($first: Int!, $after: String, $role: UserRole) {
  users(first: $first, after: $after, filter: { role: $role }) {
    edges {
      node {
        id
        name
        email
        role
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
  admin: user(id: "1") { name, role }
  editor: user(id: "2") { name, role }
}

# フラグメント（共通フィールドの再利用）
fragment UserBasic on User {
  id
  name
  email
  role
}

query GetMultipleUsers {
  user1: user(id: "1") { ...UserBasic, orderCount }
  user2: user(id: "2") { ...UserBasic, orderCount }
}
```

---

## 4. Mutation（データ変更）

```graphql
# スキーマ定義
type Mutation {
  # ユーザー
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!

  # 注文
  createOrder(input: CreateOrderInput!): CreateOrderPayload!
  cancelOrder(id: ID!): CancelOrderPayload!
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

type UserError {
  field: String
  message: String!
  code: ErrorCode!
}

enum ErrorCode {
  NOT_FOUND
  VALIDATION_ERROR
  ALREADY_EXISTS
  UNAUTHORIZED
  INTERNAL_ERROR
}
```

```graphql
# Mutation の実行例

mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    user {
      id
      name
      email
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
#       "user": { "id": "456", "name": "Taro", "email": "taro@example.com" },
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
#         { "field": "email", "message": "Email already exists", "code": "ALREADY_EXISTS" }
#       ]
#     }
#   }
# }
```

---

## 5. リゾルバー実装

```javascript
// Apollo Server でのリゾルバー実装

const resolvers = {
  Query: {
    // 単一ユーザー取得
    user: async (parent, { id }, context) => {
      const user = await context.dataSources.userAPI.getUser(id);
      if (!user) return null;
      return user;
    },

    // ユーザー一覧（Cursorページネーション）
    users: async (parent, { first = 20, after, filter, sort }, context) => {
      const { nodes, totalCount, hasNextPage, hasPreviousPage } =
        await context.dataSources.userAPI.listUsers({
          first, after, filter, sort,
        });

      const edges = nodes.map(node => ({
        node,
        cursor: Buffer.from(node.id).toString('base64'),
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
  },

  Mutation: {
    createUser: async (parent, { input }, context) => {
      try {
        const user = await context.dataSources.userAPI.createUser(input);
        return { user, errors: [] };
      } catch (error) {
        return {
          user: null,
          errors: [{ field: error.field, message: error.message, code: error.code }],
        };
      }
    },
  },

  // フィールドレベルリゾルバー
  User: {
    // user.orders は別テーブルから取得
    orders: async (user, { first = 10 }, context) => {
      return context.dataSources.orderAPI.getOrdersByUserId(user.id, first);
    },

    // 計算フィールド
    orderCount: async (user, args, context) => {
      return context.dataSources.orderAPI.countByUserId(user.id);
    },
  },

  // カスタムスカラー
  DateTime: new GraphQLScalarType({
    name: 'DateTime',
    description: 'ISO 8601 date-time string',
    serialize(value) {
      return value instanceof Date ? value.toISOString() : value;
    },
    parseValue(value) {
      return new Date(value);
    },
    parseLiteral(ast) {
      if (ast.kind === Kind.STRING) {
        return new Date(ast.value);
      }
      return null;
    },
  }),
};
```

---

## 6. Apollo Server セットアップ

```javascript
// server.js
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { readFileSync } from 'fs';

// スキーマファイルの読み込み
const typeDefs = readFileSync('./schema.graphql', 'utf-8');

const server = new ApolloServer({
  typeDefs,
  resolvers,
  // イントロスペクション（本番では無効推奨）
  introspection: process.env.NODE_ENV !== 'production',
});

const { url } = await startStandaloneServer(server, {
  listen: { port: 4000 },
  context: async ({ req }) => ({
    // 認証
    user: await authenticateUser(req.headers.authorization),
    // データソース
    dataSources: {
      userAPI: new UserAPI(),
      orderAPI: new OrderAPI(),
    },
  }),
});

console.log(`GraphQL server ready at ${url}`);
```

---

## 7. クライアント実装

```javascript
// Apollo Client（React）
import { ApolloClient, InMemoryCache, gql, useQuery } from '@apollo/client';

const client = new ApolloClient({
  uri: 'http://localhost:4000/graphql',
  cache: new InMemoryCache(),
  headers: {
    Authorization: `Bearer ${token}`,
  },
});

// クエリ定義
const GET_USER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      email
      orders {
        edges {
          node {
            id
            total
          }
        }
      }
    }
  }
`;

// React コンポーネント
function UserProfile({ userId }) {
  const { loading, error, data } = useQuery(GET_USER, {
    variables: { id: userId },
  });

  if (loading) return <p>Loading...</p>;
  if (error) return <p>Error: {error.message}</p>;

  const { user } = data;
  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      <h2>Orders</h2>
      {user.orders.edges.map(({ node }) => (
        <div key={node.id}>Order #{node.id}: ¥{node.total}</div>
      ))}
    </div>
  );
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| SDL | 型システムでスキーマを定義 |
| Query | クライアントが必要なデータを正確に指定 |
| Mutation | Payload パターンでエラーハンドリング |
| リゾルバー | 親→子の階層的なデータ取得 |
| フラグメント | 共通フィールドの再利用 |

---

## 次に読むべきガイド
→ [[02-graphql-advanced.md]] — GraphQL応用

---

## 参考文献
1. GraphQL Foundation. "GraphQL Specification." graphql.org, 2024.
2. Apollo. "Apollo Server Documentation." apollographql.com, 2024.
3. Lee, B. "GraphQL in Action." Manning, 2021.
