# GraphQL応用

> GraphQLの応用トピック。Subscription（リアルタイム通信）、DataLoader（N+1問題の解決）、キャッシュ戦略、エラーハンドリング、セキュリティまで、プロダクション運用に必要な知識を習得する。

## この章で学ぶこと

- [ ] Subscriptionによるリアルタイム通信を理解する
- [ ] DataLoaderでN+1問題を解決する方法を把握する
- [ ] GraphQLのセキュリティ対策を学ぶ

---

## 1. Subscription（リアルタイム）

```graphql
# スキーマ定義
type Subscription {
  # 新しいメッセージの購読
  messageAdded(channelId: ID!): Message!

  # 注文ステータスの変更
  orderStatusChanged(orderId: ID!): Order!

  # ユーザーのオンライン状態
  userPresenceChanged: UserPresence!
}

type Message {
  id: ID!
  content: String!
  author: User!
  channel: Channel!
  createdAt: DateTime!
}

type UserPresence {
  user: User!
  isOnline: Boolean!
  lastSeen: DateTime!
}
```

```javascript
// サーバー側（Apollo Server + WebSocket）
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import { ApolloServer } from '@apollo/server';
import { PubSub } from 'graphql-subscriptions';

const pubsub = new PubSub();

const resolvers = {
  Subscription: {
    messageAdded: {
      subscribe: (_, { channelId }) => {
        return pubsub.asyncIterator(`MESSAGE_ADDED_${channelId}`);
      },
    },

    orderStatusChanged: {
      subscribe: (_, { orderId }) => {
        return pubsub.asyncIterator(`ORDER_STATUS_${orderId}`);
      },
    },
  },

  Mutation: {
    sendMessage: async (_, { input }, context) => {
      const message = await context.dataSources.messageAPI.create(input);

      // Subscriptionに通知
      pubsub.publish(`MESSAGE_ADDED_${input.channelId}`, {
        messageAdded: message,
      });

      return { message, errors: [] };
    },
  },
};

// WebSocketサーバーのセットアップ
const httpServer = createServer(app);
const wsServer = new WebSocketServer({
  server: httpServer,
  path: '/graphql',
});

useServer(
  {
    schema,
    context: async (ctx) => {
      // WebSocket接続時の認証
      const token = ctx.connectionParams?.authorization;
      const user = await authenticateUser(token);
      return { user };
    },
  },
  wsServer
);
```

```javascript
// クライアント側（React + Apollo Client）
import { useSubscription, gql } from '@apollo/client';

const MESSAGE_SUBSCRIPTION = gql`
  subscription OnMessageAdded($channelId: ID!) {
    messageAdded(channelId: $channelId) {
      id
      content
      author { name }
      createdAt
    }
  }
`;

function ChatMessages({ channelId }) {
  const { data, loading } = useSubscription(MESSAGE_SUBSCRIPTION, {
    variables: { channelId },
  });

  // 新しいメッセージが来たら自動更新
  if (data) {
    console.log('New message:', data.messageAdded);
  }

  return <div>...</div>;
}
```

---

## 2. N+1問題とDataLoader

```
N+1問題:
  query {
    users(first: 10) {        ← 1回のクエリ（usersテーブル）
      edges {
        node {
          name
          orders {             ← 10回のクエリ（ordersテーブル × ユーザー数）
            total
          }
        }
      }
    }
  }

  実行されるSQL:
    SELECT * FROM users LIMIT 10;              -- 1回
    SELECT * FROM orders WHERE user_id = 1;    -- +1回
    SELECT * FROM orders WHERE user_id = 2;    -- +1回
    SELECT * FROM orders WHERE user_id = 3;    -- +1回
    ...                                         -- = N+1回

  → 10ユーザー = 11クエリ（1 + 10）
  → 100ユーザー = 101クエリ
```

```javascript
// DataLoader による解決
import DataLoader from 'dataloader';

// バッチ関数: 複数のIDをまとめて1クエリで取得
const ordersByUserLoader = new DataLoader(async (userIds) => {
  // 1回のクエリで全ユーザーの注文を取得
  const orders = await db.query(
    'SELECT * FROM orders WHERE user_id = ANY($1)',
    [userIds]
  );

  // userIdでグループ化して返す（入力の順序を保持）
  const orderMap = new Map();
  orders.forEach(order => {
    if (!orderMap.has(order.userId)) {
      orderMap.set(order.userId, []);
    }
    orderMap.get(order.userId).push(order);
  });

  return userIds.map(id => orderMap.get(id) || []);
});

// リゾルバーでDataLoaderを使用
const resolvers = {
  User: {
    orders: (user) => ordersByUserLoader.load(user.id),
  },
};

// 実行されるSQL:
//   SELECT * FROM users LIMIT 10;                          -- 1回
//   SELECT * FROM orders WHERE user_id = ANY([1,2,...10]); -- 1回
// → 合計2クエリ（N+1が解消）
```

```javascript
// DataLoaderのコンテキスト設定
// 重要: DataLoaderはリクエストごとに新しいインスタンスを作る

function createLoaders(db) {
  return {
    userLoader: new DataLoader(async (ids) => {
      const users = await db.query(
        'SELECT * FROM users WHERE id = ANY($1)', [ids]
      );
      const userMap = new Map(users.map(u => [u.id, u]));
      return ids.map(id => userMap.get(id) || null);
    }),

    ordersByUserLoader: new DataLoader(async (userIds) => {
      const orders = await db.query(
        'SELECT * FROM orders WHERE user_id = ANY($1)', [userIds]
      );
      const map = new Map();
      orders.forEach(o => {
        if (!map.has(o.userId)) map.set(o.userId, []);
        map.get(o.userId).push(o);
      });
      return userIds.map(id => map.get(id) || []);
    }),
  };
}

// Apollo Server のコンテキスト
const server = new ApolloServer({
  typeDefs,
  resolvers,
});

startStandaloneServer(server, {
  context: async ({ req }) => ({
    user: await authenticateUser(req),
    loaders: createLoaders(db), // リクエストごとに新規作成
  }),
});
```

---

## 3. キャッシュ戦略

```
GraphQLのキャッシュの課題:
  → RESTはURLベースでキャッシュ可能
  → GraphQLは全てPOST /graphql（URLが同じ）
  → HTTPキャッシュが使えない

解決策:

① クライアントサイドキャッシュ（Apollo Client）:
  → 正規化キャッシュ
  → __typename + id で一意に管理
  → 同じオブジェクトを参照する全てのクエリが自動更新

② Persisted Queries:
  → クエリ文字列をハッシュに変換
  → GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc..."}}
  → CDNキャッシュが可能に

③ サーバーサイドキャッシュ:
  → Redis等でリゾルバーレベルのキャッシュ
  → @cacheControl ディレクティブ
```

```javascript
// Apollo Client の正規化キャッシュ
const cache = new InMemoryCache({
  typePolicies: {
    User: {
      // idフィールドでキャッシュのキーを生成
      keyFields: ['id'],
    },
    Query: {
      fields: {
        // usersクエリのキャッシュマージ
        users: {
          keyArgs: ['filter', 'sort'], // これらの引数でキャッシュを分ける
          merge(existing, incoming, { args }) {
            // ページネーションのマージ
            if (!existing) return incoming;
            return {
              ...incoming,
              edges: [...existing.edges, ...incoming.edges],
            };
          },
        },
      },
    },
  },
});

// キャッシュの手動更新
client.cache.modify({
  id: client.cache.identify({ __typename: 'User', id: '123' }),
  fields: {
    name: () => 'Updated Name',
  },
});
```

---

## 4. エラーハンドリング

```graphql
# GraphQLの2つのエラーパターン

# ① トップレベルエラー（GraphQL仕様のerrors配列）
# → 認証エラー、構文エラー、サーバーエラー
{
  "data": null,
  "errors": [
    {
      "message": "Not authenticated",
      "locations": [{ "line": 2, "column": 3 }],
      "path": ["user"],
      "extensions": {
        "code": "UNAUTHENTICATED",
        "http": { "status": 401 }
      }
    }
  ]
}

# ② ビジネスロジックエラー（Payload内のerrors）
# → バリデーション、ビジネスルール違反
{
  "data": {
    "createUser": {
      "user": null,
      "errors": [
        { "field": "email", "message": "Already exists", "code": "ALREADY_EXISTS" }
      ]
    }
  }
}
```

```javascript
// サーバー側のエラーハンドリング
import { GraphQLError } from 'graphql';

const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      // 認証チェック
      if (!context.user) {
        throw new GraphQLError('Not authenticated', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      // 権限チェック
      if (!context.user.canViewUser(id)) {
        throw new GraphQLError('Not authorized', {
          extensions: { code: 'FORBIDDEN' },
        });
      }

      const user = await context.dataSources.userAPI.getUser(id);
      if (!user) {
        throw new GraphQLError('User not found', {
          extensions: { code: 'NOT_FOUND' },
        });
      }

      return user;
    },
  },

  Mutation: {
    createUser: async (_, { input }, context) => {
      // ビジネスロジックエラーはPayloadで返す
      const errors = validateCreateUser(input);
      if (errors.length > 0) {
        return { user: null, errors };
      }

      try {
        const user = await context.dataSources.userAPI.create(input);
        return { user, errors: [] };
      } catch (error) {
        if (error.code === 'UNIQUE_VIOLATION') {
          return {
            user: null,
            errors: [{ field: 'email', message: 'Already exists', code: 'ALREADY_EXISTS' }],
          };
        }
        throw error; // 予期しないエラーはトップレベルに
      }
    },
  },
};
```

---

## 5. セキュリティ

```
GraphQL特有のセキュリティリスク:

① クエリの深さ攻撃:
  query {
    user(id: "1") {
      orders {
        items {
          product {
            reviews {
              author {
                orders {        ← 再帰的にネスト
                  items { ... }
                }
              }
            }
          }
        }
      }
    }
  }
  → 対策: クエリ深さの制限

② クエリの複雑度攻撃:
  query {
    users(first: 1000) {
      orders(first: 1000) {
        items(first: 1000) { ... }
      }
    }
  }
  → 対策: クエリコストの制限

③ イントロスペクション:
  query { __schema { types { name fields { name } } } }
  → 対策: 本番では無効化

④ Batch攻撃:
  [
    { "query": "query { user(id: \"1\") { ... } }" },
    { "query": "query { user(id: \"2\") { ... } }" },
    ... × 1000
  ]
  → 対策: バッチサイズの制限
```

```javascript
// セキュリティ対策の実装

// ① クエリ深さ制限
import depthLimit from 'graphql-depth-limit';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [depthLimit(7)], // 最大7階層
});

// ② クエリコスト分析
import { createComplexityRule, simpleEstimator, fieldExtensionsEstimator } from 'graphql-query-complexity';

const complexityRule = createComplexityRule({
  maximumComplexity: 1000,
  estimators: [
    fieldExtensionsEstimator(),
    simpleEstimator({ defaultComplexity: 1 }),
  ],
  onComplete: (complexity) => {
    console.log('Query Complexity:', complexity);
  },
});

// ③ レート制限
const rateLimitDirective = {
  rateLimitDirectiveTypeDefs: `
    directive @rateLimit(max: Int!, window: String!) on FIELD_DEFINITION
  `,
};

// ④ Persisted Queries（許可されたクエリのみ実行）
const server = new ApolloServer({
  typeDefs,
  resolvers,
  persistedQueries: {
    cache: new KeyValueCache(),
  },
});
```

---

## 6. スキーマ設計パターン

```graphql
# ① インターフェース
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
  createdAt: DateTime!
  updatedAt: DateTime!
}

# ② ユニオン型
union SearchResult = User | Product | Order

type Query {
  search(query: String!): [SearchResult!]!
}

# クエリ側
query Search($q: String!) {
  search(query: $q) {
    ... on User { id, name, email }
    ... on Product { id, name, price }
    ... on Order { id, total, status }
  }
}

# ③ Relay Connection仕様
type Query {
  node(id: ID!): Node           # 任意のNodeをIDで取得
  users(first: Int, after: String): UserConnection!
}

# ④ ディレクティブ
directive @auth(requires: Role!) on FIELD_DEFINITION
directive @deprecated(reason: String) on FIELD_DEFINITION

type Query {
  publicData: String!
  sensitiveData: String! @auth(requires: ADMIN)
  oldField: String @deprecated(reason: "Use newField instead")
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Subscription | WebSocket + PubSub でリアルタイム |
| DataLoader | バッチ処理でN+1問題を解消 |
| キャッシュ | 正規化キャッシュ + Persisted Queries |
| セキュリティ | 深さ制限、コスト制限、イントロスペクション無効化 |
| スキーマ設計 | Interface、Union、Relay Connection |

---

## 次に読むべきガイド
→ [[03-rest-vs-graphql.md]] — REST vs GraphQL

---

## 参考文献
1. Apollo. "Production Readiness Checklist." apollographql.com, 2024.
2. Facebook. "DataLoader." github.com/graphql/dataloader, 2024.
3. Relay. "Relay Specification." relay.dev, 2024.
