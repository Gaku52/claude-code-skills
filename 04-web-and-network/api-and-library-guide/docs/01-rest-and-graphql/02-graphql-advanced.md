# GraphQL応用

> GraphQLの応用トピック。Subscription（リアルタイム通信）、DataLoader（N+1問題の解決）、キャッシュ戦略、エラーハンドリング、セキュリティ、Federation、パフォーマンスチューニングまで、プロダクション運用に必要な知識を網羅的に習得する。

## この章で学ぶこと

- [ ] Subscriptionによるリアルタイム通信を理解する
- [ ] DataLoaderでN+1問題を解決する方法を把握する
- [ ] GraphQLのキャッシュ戦略を複数レイヤーで学ぶ
- [ ] エラーハンドリングの設計パターンを習得する
- [ ] GraphQL特有のセキュリティ対策を実装できる
- [ ] スキーマ設計の高度なパターンを使いこなす
- [ ] Apollo Federationによるマイクロサービス統合を理解する
- [ ] パフォーマンスの計測と最適化手法を把握する
- [ ] テスト戦略とモニタリングを実践できる

---

## 1. Subscription（リアルタイム通信）

### 1.1 Subscriptionの基本概念

GraphQL Subscriptionは、サーバーからクライアントへのリアルタイムデータ配信を実現する仕組みである。RESTでのポーリングやWebSocketの直接利用と比較して、型安全なリアルタイム通信を提供する。

```
Subscriptionの動作フロー:

  Client                    Server
    |                         |
    |-- subscription req ---->|  ① WebSocket接続確立
    |                         |
    |                         |  ② サーバー側でイベント発生
    |<-- data push -----------|  ③ データをクライアントへ配信
    |                         |
    |                         |  ④ 再度イベント発生
    |<-- data push -----------|  ⑤ 再度データ配信
    |                         |
    |-- unsubscribe --------->|  ⑥ 購読解除
    |                         |

  vs ポーリング:
    Client → Server: GET /api/messages?since=xxx  (毎秒)
    → 無駄なリクエストが大量に発生
    → リアルタイム性が低い（ポーリング間隔に依存）

  vs WebSocket直接利用:
    → 型安全性がない
    → メッセージフォーマットの統一が困難
    → GraphQLのSubscriptionは型付きリアルタイム通信を提供
```

### 1.2 スキーマ定義

```graphql
# Subscription スキーマ定義
type Subscription {
  # 新しいメッセージの購読
  messageAdded(channelId: ID!): Message!

  # 注文ステータスの変更
  orderStatusChanged(orderId: ID!): Order!

  # ユーザーのオンライン状態
  userPresenceChanged: UserPresence!

  # 通知のリアルタイム配信
  notificationReceived(userId: ID!): Notification!

  # ダッシュボードメトリクスの更新
  metricsUpdated(dashboardId: ID!): DashboardMetrics!

  # タイピングインジケーター
  userTyping(channelId: ID!): TypingIndicator!
}

type Message {
  id: ID!
  content: String!
  author: User!
  channel: Channel!
  attachments: [Attachment!]!
  reactions: [Reaction!]!
  createdAt: DateTime!
  editedAt: DateTime
}

type UserPresence {
  user: User!
  isOnline: Boolean!
  lastSeen: DateTime!
  status: PresenceStatus!
}

enum PresenceStatus {
  ONLINE
  AWAY
  DO_NOT_DISTURB
  OFFLINE
}

type Notification {
  id: ID!
  type: NotificationType!
  title: String!
  body: String!
  actionUrl: String
  isRead: Boolean!
  createdAt: DateTime!
}

enum NotificationType {
  MESSAGE
  MENTION
  ORDER_UPDATE
  SYSTEM
  PROMOTION
}

type TypingIndicator {
  user: User!
  channelId: ID!
  isTyping: Boolean!
}

type DashboardMetrics {
  activeUsers: Int!
  requestsPerSecond: Float!
  errorRate: Float!
  averageResponseTime: Float!
  timestamp: DateTime!
}
```

### 1.3 サーバー側実装（Apollo Server + WebSocket）

```javascript
// サーバー側（Apollo Server + WebSocket）
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { ApolloServerPluginDrainHttpServer } from '@apollo/server/plugin/drainHttpServer';
import { makeExecutableSchema } from '@graphql-tools/schema';
import { PubSub, withFilter } from 'graphql-subscriptions';
import express from 'express';

const pubsub = new PubSub();

// イベント名の定数定義
const EVENTS = {
  MESSAGE_ADDED: 'MESSAGE_ADDED',
  ORDER_STATUS_CHANGED: 'ORDER_STATUS_CHANGED',
  USER_PRESENCE_CHANGED: 'USER_PRESENCE_CHANGED',
  NOTIFICATION_RECEIVED: 'NOTIFICATION_RECEIVED',
  METRICS_UPDATED: 'METRICS_UPDATED',
  USER_TYPING: 'USER_TYPING',
};

const resolvers = {
  Subscription: {
    // withFilterでチャンネルIDによるフィルタリング
    messageAdded: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(EVENTS.MESSAGE_ADDED),
        (payload, variables) => {
          // 指定されたチャンネルのメッセージのみ配信
          return payload.messageAdded.channel.id === variables.channelId;
        }
      ),
    },

    orderStatusChanged: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(EVENTS.ORDER_STATUS_CHANGED),
        (payload, variables) => {
          return payload.orderStatusChanged.id === variables.orderId;
        }
      ),
    },

    userPresenceChanged: {
      subscribe: () => pubsub.asyncIterator(EVENTS.USER_PRESENCE_CHANGED),
    },

    notificationReceived: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(EVENTS.NOTIFICATION_RECEIVED),
        (payload, variables, context) => {
          // 認証済みユーザー自身の通知のみ配信
          return payload.notificationReceived.userId === variables.userId
            && context.user.id === variables.userId;
        }
      ),
    },

    userTyping: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(EVENTS.USER_TYPING),
        (payload, variables) => {
          return payload.userTyping.channelId === variables.channelId;
        }
      ),
    },

    metricsUpdated: {
      subscribe: withFilter(
        () => pubsub.asyncIterator(EVENTS.METRICS_UPDATED),
        (payload, variables, context) => {
          // 管理者のみメトリクスを購読可能
          if (!context.user?.roles?.includes('ADMIN')) {
            throw new Error('Not authorized to subscribe to metrics');
          }
          return payload.metricsUpdated.dashboardId === variables.dashboardId;
        }
      ),
    },
  },

  Mutation: {
    sendMessage: async (_, { input }, context) => {
      // 認証チェック
      if (!context.user) {
        throw new GraphQLError('Not authenticated', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      const message = await context.dataSources.messageAPI.create({
        ...input,
        authorId: context.user.id,
      });

      // Subscriptionに通知
      pubsub.publish(EVENTS.MESSAGE_ADDED, {
        messageAdded: message,
      });

      return { message, errors: [] };
    },

    updateOrderStatus: async (_, { orderId, status }, context) => {
      const order = await context.dataSources.orderAPI.updateStatus(
        orderId,
        status
      );

      // 注文ステータス変更を通知
      pubsub.publish(EVENTS.ORDER_STATUS_CHANGED, {
        orderStatusChanged: order,
      });

      // 顧客への通知も同時に発行
      pubsub.publish(EVENTS.NOTIFICATION_RECEIVED, {
        notificationReceived: {
          userId: order.customerId,
          type: 'ORDER_UPDATE',
          title: '注文ステータスが更新されました',
          body: `注文 #${orderId} のステータスが「${status}」に変更されました`,
          createdAt: new Date().toISOString(),
        },
      });

      return order;
    },

    setTypingStatus: async (_, { channelId, isTyping }, context) => {
      pubsub.publish(EVENTS.USER_TYPING, {
        userTyping: {
          user: context.user,
          channelId,
          isTyping,
        },
      });
      return true;
    },
  },
};

// Express + Apollo Server + WebSocket のセットアップ
const app = express();
const httpServer = createServer(app);

const schema = makeExecutableSchema({ typeDefs, resolvers });

// WebSocketサーバーのセットアップ
const wsServer = new WebSocketServer({
  server: httpServer,
  path: '/graphql',
});

const serverCleanup = useServer(
  {
    schema,
    context: async (ctx, msg, args) => {
      // WebSocket接続時の認証
      const token = ctx.connectionParams?.authorization;
      if (!token) {
        throw new Error('Missing authentication token');
      }

      const user = await authenticateUser(token);
      if (!user) {
        throw new Error('Invalid authentication token');
      }

      return { user };
    },
    onConnect: async (ctx) => {
      console.log('Client connected:', ctx.connectionParams);
      // 接続時のバリデーション
      const token = ctx.connectionParams?.authorization;
      if (!token) {
        return false; // 接続を拒否
      }
      return true;
    },
    onDisconnect: async (ctx, code, reason) => {
      console.log('Client disconnected:', code, reason);
      // ユーザーのオフライン状態を通知
      if (ctx.extra?.user) {
        pubsub.publish(EVENTS.USER_PRESENCE_CHANGED, {
          userPresenceChanged: {
            user: ctx.extra.user,
            isOnline: false,
            lastSeen: new Date().toISOString(),
            status: 'OFFLINE',
          },
        });
      }
    },
    onSubscribe: (ctx, msg) => {
      console.log('Subscription started:', msg.payload.query);
    },
    onNext: (ctx, msg, args, result) => {
      // 各メッセージ送信時のフック
      console.log('Sending subscription data');
    },
    onError: (ctx, msg, errors) => {
      console.error('Subscription error:', errors);
    },
    onComplete: (ctx, msg) => {
      console.log('Subscription completed');
    },
  },
  wsServer
);

// Apollo Server のセットアップ
const server = new ApolloServer({
  schema,
  plugins: [
    // HTTPサーバーの適切なシャットダウン
    ApolloServerPluginDrainHttpServer({ httpServer }),
    // WebSocketサーバーの適切なシャットダウン
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

app.use(
  '/graphql',
  express.json(),
  expressMiddleware(server, {
    context: async ({ req }) => ({
      user: await authenticateUser(req.headers.authorization),
      dataSources: createDataSources(),
    }),
  })
);

httpServer.listen(4000, () => {
  console.log('Server running on http://localhost:4000/graphql');
  console.log('WebSocket running on ws://localhost:4000/graphql');
});
```

### 1.4 スケーラブルなPubSub実装（Redis）

```javascript
// プロダクション環境ではインメモリPubSubの代わりにRedis PubSubを使用
import { RedisPubSub } from 'graphql-redis-subscriptions';
import Redis from 'ioredis';

// Redis接続の設定
const redisOptions = {
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
  retryStrategy: (times) => {
    return Math.min(times * 50, 2000);
  },
  maxRetriesPerRequest: 3,
};

// PubSub用に別々のRedis接続を作成（推奨）
const pubsub = new RedisPubSub({
  publisher: new Redis(redisOptions),
  subscriber: new Redis(redisOptions),
  // メッセージのシリアライゼーション
  serializer: (data) => JSON.stringify(data),
  deserializer: (message) => JSON.parse(message),
  // 接続エラーハンドリング
  connectionListener: (err) => {
    if (err) {
      console.error('Redis connection error:', err);
    }
  },
});

// 複数サーバーインスタンスでの利用
// Server A で publish → Redis → Server B の subscriber に配信
// → 水平スケーリングが可能

// Kafka を使ったPubSub（大規模システム向け）
import { KafkaPubSub } from 'graphql-kafka-subscriptions';

const kafkaPubSub = await KafkaPubSub.create({
  topic: 'graphql-subscriptions',
  host: process.env.KAFKA_HOST || 'localhost',
  port: process.env.KAFKA_PORT || '9092',
  groupIdPrefix: 'graphql-server',
  globalConfig: {
    'client.id': 'graphql-subscriptions-client',
  },
});
```

### 1.5 クライアント側実装（React + Apollo Client）

```javascript
// Apollo Client のWebSocket設定
import { ApolloClient, InMemoryCache, split, HttpLink } from '@apollo/client';
import { GraphQLWsLink } from '@apollo/client/link/subscriptions';
import { createClient } from 'graphql-ws';
import { getMainDefinition } from '@apollo/client/utilities';

// HTTP リンク（Query, Mutation用）
const httpLink = new HttpLink({
  uri: 'https://api.example.com/graphql',
  headers: {
    authorization: `Bearer ${getToken()}`,
  },
});

// WebSocket リンク（Subscription用）
const wsLink = new GraphQLWsLink(
  createClient({
    url: 'wss://api.example.com/graphql',
    connectionParams: () => ({
      authorization: `Bearer ${getToken()}`,
    }),
    // 再接続設定
    retryAttempts: 5,
    shouldRetry: () => true,
    retryWait: async (retryCount) => {
      // 指数バックオフ
      const delay = Math.min(1000 * Math.pow(2, retryCount), 30000);
      await new Promise((resolve) => setTimeout(resolve, delay));
    },
    on: {
      connected: () => console.log('WebSocket connected'),
      closed: (event) => console.log('WebSocket closed:', event),
      error: (error) => console.error('WebSocket error:', error),
    },
    // KeepAlive設定
    keepAlive: 10000, // 10秒ごとにping
  })
);

// 操作タイプに応じてリンクを振り分け
const splitLink = split(
  ({ query }) => {
    const definition = getMainDefinition(query);
    return (
      definition.kind === 'OperationDefinition' &&
      definition.operation === 'subscription'
    );
  },
  wsLink,   // Subscription → WebSocket
  httpLink  // Query, Mutation → HTTP
);

const client = new ApolloClient({
  link: splitLink,
  cache: new InMemoryCache(),
});
```

```tsx
// React コンポーネントでの利用
import { useSubscription, useQuery, gql } from '@apollo/client';
import { useCallback, useEffect, useState } from 'react';

const MESSAGE_SUBSCRIPTION = gql`
  subscription OnMessageAdded($channelId: ID!) {
    messageAdded(channelId: $channelId) {
      id
      content
      author {
        id
        name
        avatar
      }
      createdAt
    }
  }
`;

const GET_MESSAGES = gql`
  query GetMessages($channelId: ID!, $first: Int!, $after: String) {
    messages(channelId: $channelId, first: $first, after: $after) {
      edges {
        node {
          id
          content
          author {
            id
            name
            avatar
          }
          createdAt
        }
        cursor
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
`;

function ChatMessages({ channelId }) {
  // 既存メッセージの取得
  const { data, loading, fetchMore, subscribeToMore } = useQuery(
    GET_MESSAGES,
    {
      variables: { channelId, first: 50 },
    }
  );

  // subscribeToMore で既存クエリに新しいメッセージを追加
  useEffect(() => {
    const unsubscribe = subscribeToMore({
      document: MESSAGE_SUBSCRIPTION,
      variables: { channelId },
      updateQuery: (prev, { subscriptionData }) => {
        if (!subscriptionData.data) return prev;

        const newMessage = subscriptionData.data.messageAdded;

        // 重複チェック
        const exists = prev.messages.edges.some(
          (edge) => edge.node.id === newMessage.id
        );
        if (exists) return prev;

        return {
          ...prev,
          messages: {
            ...prev.messages,
            edges: [
              {
                __typename: 'MessageEdge',
                node: newMessage,
                cursor: newMessage.id,
              },
              ...prev.messages.edges,
            ],
          },
        };
      },
    });

    return () => unsubscribe();
  }, [channelId, subscribeToMore]);

  if (loading) return <div>Loading...</div>;

  return (
    <div className="chat-messages">
      {data?.messages.edges.map(({ node: message }) => (
        <div key={message.id} className="message">
          <img src={message.author.avatar} alt={message.author.name} />
          <div>
            <strong>{message.author.name}</strong>
            <p>{message.content}</p>
            <small>{new Date(message.createdAt).toLocaleString()}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

// タイピングインジケーターの実装
const TYPING_SUBSCRIPTION = gql`
  subscription OnUserTyping($channelId: ID!) {
    userTyping(channelId: $channelId) {
      user {
        id
        name
      }
      isTyping
    }
  }
`;

function TypingIndicator({ channelId }) {
  const [typingUsers, setTypingUsers] = useState(new Map());

  const { data } = useSubscription(TYPING_SUBSCRIPTION, {
    variables: { channelId },
    onData: ({ data: { data } }) => {
      if (!data) return;

      const { user, isTyping } = data.userTyping;

      setTypingUsers((prev) => {
        const next = new Map(prev);
        if (isTyping) {
          next.set(user.id, { name: user.name, timestamp: Date.now() });
        } else {
          next.delete(user.id);
        }
        return next;
      });
    },
  });

  // 5秒後にタイピング状態を自動クリア
  useEffect(() => {
    const interval = setInterval(() => {
      setTypingUsers((prev) => {
        const now = Date.now();
        const next = new Map();
        prev.forEach((value, key) => {
          if (now - value.timestamp < 5000) {
            next.set(key, value);
          }
        });
        return next;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  if (typingUsers.size === 0) return null;

  const names = Array.from(typingUsers.values()).map((u) => u.name);

  return (
    <div className="typing-indicator">
      {names.length === 1
        ? `${names[0]} is typing...`
        : names.length === 2
          ? `${names[0]} and ${names[1]} are typing...`
          : `${names.length} people are typing...`}
    </div>
  );
}
```

---

## 2. N+1問題とDataLoader

### 2.1 N+1問題の詳細

```
N+1問題:
  query {
    users(first: 10) {        <- 1回のクエリ（usersテーブル）
      edges {
        node {
          name
          orders {             <- 10回のクエリ（ordersテーブル x ユーザー数）
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

  -> 10ユーザー = 11クエリ（1 + 10）
  -> 100ユーザー = 101クエリ
  -> ネストが深いと指数的に増加:

  query {
    users(first: 10) {              -- 1
      orders(first: 5) {            -- 10
        items(first: 10) {          -- 50
          product {                 -- 500
            reviews(first: 5) {     -- 500
              author { name }       -- 2500
            }
          }
        }
      }
    }
  }
  -> 合計: 3,061 クエリ！
```

### 2.2 DataLoaderによる解決

```javascript
// DataLoader による解決
import DataLoader from 'dataloader';

// === バッチ関数の基本パターン ===

// パターン1: 1対1（ユーザーIDからユーザーを取得）
const userLoader = new DataLoader(async (userIds) => {
  console.log(`Batch loading users: [${userIds.join(', ')}]`);

  // 1回のクエリで全ユーザーを取得
  const users = await db.query(
    'SELECT * FROM users WHERE id = ANY($1)',
    [userIds]
  );

  // 入力IDの順序を保持してマッピング
  const userMap = new Map(users.map(u => [u.id, u]));
  return userIds.map(id => userMap.get(id) || null);
});

// パターン2: 1対多（ユーザーIDから注文一覧を取得）
const ordersByUserLoader = new DataLoader(async (userIds) => {
  console.log(`Batch loading orders for users: [${userIds.join(', ')}]`);

  const orders = await db.query(
    'SELECT * FROM orders WHERE user_id = ANY($1) ORDER BY created_at DESC',
    [userIds]
  );

  // userIdでグループ化して返す
  const orderMap = new Map();
  orders.forEach(order => {
    if (!orderMap.has(order.userId)) {
      orderMap.set(order.userId, []);
    }
    orderMap.get(order.userId).push(order);
  });

  return userIds.map(id => orderMap.get(id) || []);
});

// パターン3: 条件付きローダー（ステータスでフィルタ）
const activeOrdersByUserLoader = new DataLoader(async (keys) => {
  // keysは { userId, status } のオブジェクト配列
  const userIds = [...new Set(keys.map(k => k.userId))];
  const statuses = [...new Set(keys.map(k => k.status))];

  const orders = await db.query(
    'SELECT * FROM orders WHERE user_id = ANY($1) AND status = ANY($2)',
    [userIds, statuses]
  );

  return keys.map(key =>
    orders.filter(o => o.userId === key.userId && o.status === key.status)
  );
}, {
  // カスタムキャッシュキー（オブジェクトをキーにする場合に必要）
  cacheKeyFn: (key) => `${key.userId}:${key.status}`,
});

// リゾルバーでDataLoaderを使用
const resolvers = {
  User: {
    orders: (user, _, context) => context.loaders.ordersByUserLoader.load(user.id),
    activeOrders: (user, _, context) =>
      context.loaders.activeOrdersByUserLoader.load({
        userId: user.id,
        status: 'ACTIVE',
      }),
    // 集計もDataLoaderで効率化
    orderCount: async (user, _, context) => {
      const orders = await context.loaders.ordersByUserLoader.load(user.id);
      return orders.length;
    },
  },
  Order: {
    customer: (order, _, context) => context.loaders.userLoader.load(order.userId),
    items: (order, _, context) => context.loaders.orderItemsLoader.load(order.id),
  },
  OrderItem: {
    product: (item, _, context) => context.loaders.productLoader.load(item.productId),
  },
};

// 実行されるSQL（DataLoader使用後）:
//   SELECT * FROM users LIMIT 10;                          -- 1回
//   SELECT * FROM orders WHERE user_id = ANY([1,2,...10]); -- 1回
//   SELECT * FROM products WHERE id = ANY([...]);          -- 1回
// -> 合計3クエリ（N+1が解消）
```

### 2.3 DataLoaderのコンテキスト管理

```javascript
// DataLoaderのコンテキスト設定
// 重要: DataLoaderはリクエストごとに新しいインスタンスを作る
// → キャッシュの不整合を防ぐため

function createLoaders(db) {
  return {
    // ユーザーローダー
    userLoader: new DataLoader(async (ids) => {
      const users = await db.query(
        'SELECT * FROM users WHERE id = ANY($1)', [ids]
      );
      const userMap = new Map(users.map(u => [u.id, u]));
      return ids.map(id => userMap.get(id) || null);
    }, {
      // オプション設定
      batch: true,          // バッチ処理を有効化（デフォルト: true）
      maxBatchSize: 100,    // 1バッチの最大サイズ
      cache: true,          // キャッシュを有効化（デフォルト: true）
      batchScheduleFn: (callback) => setTimeout(callback, 10),
      // 10ms待ってからバッチ実行（より多くのリクエストをまとめる）
    }),

    // 注文ローダー
    ordersByUserLoader: new DataLoader(async (userIds) => {
      const orders = await db.query(
        'SELECT * FROM orders WHERE user_id = ANY($1) ORDER BY created_at DESC',
        [userIds]
      );
      const map = new Map();
      orders.forEach(o => {
        if (!map.has(o.userId)) map.set(o.userId, []);
        map.get(o.userId).push(o);
      });
      return userIds.map(id => map.get(id) || []);
    }),

    // 商品ローダー
    productLoader: new DataLoader(async (ids) => {
      const products = await db.query(
        'SELECT * FROM products WHERE id = ANY($1)', [ids]
      );
      const map = new Map(products.map(p => [p.id, p]));
      return ids.map(id => map.get(id) || null);
    }),

    // 注文アイテムローダー
    orderItemsLoader: new DataLoader(async (orderIds) => {
      const items = await db.query(
        `SELECT oi.*, p.name as product_name, p.price
         FROM order_items oi
         JOIN products p ON oi.product_id = p.id
         WHERE oi.order_id = ANY($1)`,
        [orderIds]
      );
      const map = new Map();
      items.forEach(item => {
        if (!map.has(item.orderId)) map.set(item.orderId, []);
        map.get(item.orderId).push(item);
      });
      return orderIds.map(id => map.get(id) || []);
    }),

    // カテゴリ別商品ローダー
    productsByCategoryLoader: new DataLoader(async (categoryIds) => {
      const products = await db.query(
        'SELECT * FROM products WHERE category_id = ANY($1) ORDER BY name',
        [categoryIds]
      );
      const map = new Map();
      products.forEach(p => {
        if (!map.has(p.categoryId)) map.set(p.categoryId, []);
        map.get(p.categoryId).push(p);
      });
      return categoryIds.map(id => map.get(id) || []);
    }),
  };
}

// Apollo Server のコンテキスト
const server = new ApolloServer({
  typeDefs,
  resolvers,
});

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => ({
    user: await authenticateUser(req),
    loaders: createLoaders(db), // リクエストごとに新規作成
    dataSources: createDataSources(),
  }),
});
```

### 2.4 DataLoaderのプライミングとキャッシュ制御

```javascript
// DataLoaderの高度な利用パターン

// 1. プライミング（事前にキャッシュをセット）
const resolvers = {
  Mutation: {
    createUser: async (_, { input }, { loaders }) => {
      const user = await db.query(
        'INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *',
        [input.name, input.email]
      );

      // 作成したユーザーをDataLoaderのキャッシュに事前登録
      // → 後続のリゾルバーでDBクエリを回避
      loaders.userLoader.prime(user.id, user);

      return { user, errors: [] };
    },
  },
};

// 2. キャッシュのクリア
async function updateUser(id, input, loaders) {
  const user = await db.query(
    'UPDATE users SET name = $2 WHERE id = $1 RETURNING *',
    [id, input.name]
  );

  // 古いキャッシュをクリアして新しいデータをセット
  loaders.userLoader.clear(id);
  loaders.userLoader.prime(id, user);

  return user;
}

// 3. 全キャッシュのクリア
function clearAllLoaderCaches(loaders) {
  Object.values(loaders).forEach(loader => {
    if (loader instanceof DataLoader) {
      loader.clearAll();
    }
  });
}

// 4. エラーハンドリング付きバッチ関数
const robustUserLoader = new DataLoader(async (ids) => {
  try {
    const users = await db.query(
      'SELECT * FROM users WHERE id = ANY($1)', [ids]
    );
    const userMap = new Map(users.map(u => [u.id, u]));

    return ids.map(id => {
      const user = userMap.get(id);
      if (!user) {
        // 個別のエラーを返す（バッチ全体を失敗させない）
        return new Error(`User not found: ${id}`);
      }
      return user;
    });
  } catch (error) {
    // DBエラーの場合は全IDに対してエラーを返す
    return ids.map(() => new Error(`Database error: ${error.message}`));
  }
});
```

---

## 3. キャッシュ戦略

### 3.1 GraphQLキャッシュの課題と解決策

```
GraphQLのキャッシュの課題:
  -> RESTはURLベースでキャッシュ可能
     GET /api/users/123 → Cache-Control: max-age=3600
  -> GraphQLは全てPOST /graphql（URLが同じ）
  -> HTTPキャッシュが使えない

解決策（4つのレイヤー）:

  ┌──────────────────────────────────────────┐
  │  Layer 1: クライアントサイドキャッシュ       │
  │  → Apollo Client InMemoryCache            │
  │  → 正規化キャッシュ（__typename + id）      │
  ├──────────────────────────────────────────┤
  │  Layer 2: CDNキャッシュ                     │
  │  → Persisted Queries（GET変換）            │
  │  → Automatic Persisted Queries (APQ)      │
  ├──────────────────────────────────────────┤
  │  Layer 3: サーバーサイドキャッシュ            │
  │  → Redis/Memcachedによるレスポンスキャッシュ  │
  │  → @cacheControl ディレクティブ             │
  ├──────────────────────────────────────────┤
  │  Layer 4: データソースキャッシュ              │
  │  → DataLoaderのリクエスト内キャッシュ         │
  │  → RESTDataSourceのHTTPキャッシュ           │
  └──────────────────────────────────────────┘
```

### 3.2 Apollo Client 正規化キャッシュ

```javascript
// Apollo Client の正規化キャッシュ
import { InMemoryCache, makeVar } from '@apollo/client';

const cache = new InMemoryCache({
  typePolicies: {
    // ユーザー型のキャッシュ設定
    User: {
      keyFields: ['id'], // idフィールドでキャッシュのキーを生成
      fields: {
        // フルネームの計算フィールド
        fullName: {
          read(_, { readField }) {
            const firstName = readField('firstName');
            const lastName = readField('lastName');
            return `${firstName} ${lastName}`;
          },
        },
      },
    },

    // 商品型のキャッシュ設定
    Product: {
      keyFields: ['sku'], // SKUをキーとして使用（idの代わり）
      fields: {
        // 価格の表示フォーマット
        formattedPrice: {
          read(_, { readField }) {
            const price = readField('price');
            return `¥${price.toLocaleString()}`;
          },
        },
      },
    },

    // クエリフィールドのキャッシュ設定
    Query: {
      fields: {
        // usersクエリのキャッシュとページネーション
        users: {
          keyArgs: ['filter', 'sort'], // これらの引数でキャッシュを分ける
          merge(existing, incoming, { args }) {
            // ページネーションのマージ
            if (!existing) return incoming;
            if (args?.after) {
              // 追加読み込み（infinite scroll）
              return {
                ...incoming,
                edges: [...existing.edges, ...incoming.edges],
              };
            }
            // 新規取得（フィルタ変更等）
            return incoming;
          },
          read(existing, { args }) {
            // キャッシュからの読み取り
            return existing;
          },
        },

        // 単一ユーザーのキャッシュ参照
        user: {
          read(_, { args, toReference }) {
            // キャッシュに既にあるユーザーを参照
            return toReference({
              __typename: 'User',
              id: args.id,
            });
          },
        },

        // 検索結果のキャッシュ
        search: {
          keyArgs: ['query', 'type'],
          merge(existing = { results: [] }, incoming) {
            return {
              ...incoming,
              results: [...existing.results, ...incoming.results],
            };
          },
        },
      },
    },

    // ページネーション接続のキャッシュ設定
    UserConnection: {
      fields: {
        edges: {
          merge(existing = [], incoming) {
            return [...existing, ...incoming];
          },
        },
      },
    },
  },

  // 可能なタイプの定義（Union/Interfaceの解決用）
  possibleTypes: {
    SearchResult: ['User', 'Product', 'Order'],
    Node: ['User', 'Product', 'Order', 'Category'],
  },
});

// === キャッシュの手動操作 ===

// 1. キャッシュの直接更新
client.cache.modify({
  id: client.cache.identify({ __typename: 'User', id: '123' }),
  fields: {
    name: () => 'Updated Name',
    email: (prevEmail) => prevEmail, // 変更しない
    orderCount: (prevCount) => prevCount + 1,
  },
});

// 2. キャッシュへの書き込み
client.cache.writeQuery({
  query: GET_USER,
  variables: { id: '123' },
  data: {
    user: {
      __typename: 'User',
      id: '123',
      name: 'New User',
      email: 'new@example.com',
    },
  },
});

// 3. キャッシュからの読み取り
const cachedUser = client.cache.readQuery({
  query: GET_USER,
  variables: { id: '123' },
});

// 4. キャッシュからの削除（evict）
client.cache.evict({
  id: client.cache.identify({ __typename: 'User', id: '123' }),
});
// ガベージコレクション（参照されなくなったオブジェクトを削除）
client.cache.gc();

// 5. Reactive Variables（ローカルステート管理）
const isLoggedInVar = makeVar(false);
const cartItemsVar = makeVar([]);

const cache2 = new InMemoryCache({
  typePolicies: {
    Query: {
      fields: {
        isLoggedIn: {
          read() {
            return isLoggedInVar();
          },
        },
        cartItems: {
          read() {
            return cartItemsVar();
          },
        },
      },
    },
  },
});

// Reactive Variableの更新（自動的にUIが再レンダリングされる）
isLoggedInVar(true);
cartItemsVar([...cartItemsVar(), { productId: '123', quantity: 1 }]);
```

### 3.3 Persisted Queries

```javascript
// Automatic Persisted Queries (APQ)
// → クエリ文字列をSHA256ハッシュに変換してGETリクエストに

import { createPersistedQueryLink } from '@apollo/client/link/persisted-queries';
import { sha256 } from 'crypto-hash';

const persistedQueryLink = createPersistedQueryLink({
  sha256,
  useGETForHashedQueries: true, // GETリクエストを使用 → CDNキャッシュ可能
});

const client = new ApolloClient({
  link: persistedQueryLink.concat(httpLink),
  cache: new InMemoryCache(),
});

// APQの動作フロー:
// 1. 初回: GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc..."}}
// 2. サーバー: "PersistedQueryNotFound" を返す
// 3. クライアント: POST /graphql でフルクエリを送信
// 4. サーバー: クエリをハッシュと紐付けて保存
// 5. 次回以降: GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc..."}}
//    → CDNでキャッシュ可能！

// サーバー側の設定（Apollo Server）
const server = new ApolloServer({
  typeDefs,
  resolvers,
  persistedQueries: {
    // Redisキャッシュで永続化
    cache: new KeyValueCache({
      url: process.env.REDIS_URL,
      ttl: 86400, // 24時間
    }),
  },
});
```

### 3.4 サーバーサイドキャッシュ

```javascript
// @cacheControl ディレクティブによるキャッシュ制御
const typeDefs = gql`
  # ディレクティブ定義
  enum CacheControlScope {
    PUBLIC
    PRIVATE
  }

  directive @cacheControl(
    maxAge: Int
    scope: CacheControlScope
    inheritMaxAge: Boolean
  ) on FIELD_DEFINITION | OBJECT | INTERFACE | UNION

  # 型レベルのキャッシュ設定
  type Product @cacheControl(maxAge: 3600) {
    id: ID!
    name: String!
    description: String!
    price: Float! @cacheControl(maxAge: 300)  # 価格は5分
    inventory: Int! @cacheControl(maxAge: 30) # 在庫は30秒
  }

  type User @cacheControl(maxAge: 0, scope: PRIVATE) {
    id: ID!
    name: String!
    email: String!
    orders: [Order!]! @cacheControl(maxAge: 60, scope: PRIVATE)
  }

  type Query {
    products(category: String): [Product!]! @cacheControl(maxAge: 600)
    product(id: ID!): Product @cacheControl(maxAge: 3600)
    me: User @cacheControl(maxAge: 0, scope: PRIVATE)
  }
`;

// Redisを使ったリゾルバーレベルのキャッシュ
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

function withCache(resolver, options = {}) {
  const { ttl = 300, keyPrefix = 'gql' } = options;

  return async (parent, args, context, info) => {
    const cacheKey = `${keyPrefix}:${info.fieldName}:${JSON.stringify(args)}`;

    // キャッシュチェック
    const cached = await redis.get(cacheKey);
    if (cached) {
      return JSON.parse(cached);
    }

    // リゾルバー実行
    const result = await resolver(parent, args, context, info);

    // キャッシュに保存
    await redis.setex(cacheKey, ttl, JSON.stringify(result));

    return result;
  };
}

// 利用例
const resolvers = {
  Query: {
    products: withCache(
      async (_, { category }, { dataSources }) => {
        return dataSources.productAPI.getProducts({ category });
      },
      { ttl: 600, keyPrefix: 'products' }
    ),

    product: withCache(
      async (_, { id }, { dataSources }) => {
        return dataSources.productAPI.getProduct(id);
      },
      { ttl: 3600, keyPrefix: 'product' }
    ),
  },
};

// キャッシュの無効化
async function invalidateProductCache(productId) {
  // 個別商品のキャッシュを削除
  await redis.del(`product:product:{"id":"${productId}"}`);

  // 商品一覧のキャッシュを全て削除
  const keys = await redis.keys('products:products:*');
  if (keys.length > 0) {
    await redis.del(...keys);
  }
}
```

---

## 4. エラーハンドリング

### 4.1 エラーパターンの分類

```graphql
# GraphQLの2つのエラーパターン

# (1) トップレベルエラー（GraphQL仕様のerrors配列）
# -> 認証エラー、構文エラー、サーバーエラー
# -> クライアントが予期できないエラー
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

# (2) ビジネスロジックエラー（Payload内のerrors）
# -> バリデーション、ビジネスルール違反
# -> クライアントが処理すべきエラー
{
  "data": {
    "createUser": {
      "user": null,
      "errors": [
        {
          "field": "email",
          "message": "Already exists",
          "code": "ALREADY_EXISTS"
        }
      ]
    }
  }
}

# (3) 部分成功パターン
# -> 一部のフィールドはnull、他は正常に返る
{
  "data": {
    "user": {
      "name": "Alice",
      "orders": null,
      "profile": {
        "bio": "Developer"
      }
    }
  },
  "errors": [
    {
      "message": "Failed to fetch orders",
      "path": ["user", "orders"],
      "extensions": { "code": "INTERNAL_SERVER_ERROR" }
    }
  ]
}
```

### 4.2 Result型パターン（Union型によるエラー表現）

```graphql
# Result型パターン: Union型でエラーを型安全に表現
# → GraphQLの型システムを活用した最も堅牢な方法

# エラー型の定義
interface UserError {
  message: String!
  path: [String!]
}

type ValidationError implements UserError {
  message: String!
  path: [String!]
  field: String!
  constraint: String!
}

type NotFoundError implements UserError {
  message: String!
  path: [String!]
  resourceType: String!
  resourceId: ID!
}

type AuthorizationError implements UserError {
  message: String!
  path: [String!]
  requiredPermission: String!
}

type BusinessRuleError implements UserError {
  message: String!
  path: [String!]
  code: String!
  details: JSON
}

# Mutation結果のUnion型
union CreateUserResult = CreateUserSuccess | ValidationError | AuthorizationError

type CreateUserSuccess {
  user: User!
}

union UpdateOrderResult =
  | UpdateOrderSuccess
  | NotFoundError
  | AuthorizationError
  | BusinessRuleError

type UpdateOrderSuccess {
  order: Order!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserResult!
  updateOrder(id: ID!, input: UpdateOrderInput!): UpdateOrderResult!
}

# クライアント側のクエリ
mutation CreateUser($input: CreateUserInput!) {
  createUser(input: $input) {
    ... on CreateUserSuccess {
      user {
        id
        name
        email
      }
    }
    ... on ValidationError {
      message
      field
      constraint
    }
    ... on AuthorizationError {
      message
      requiredPermission
    }
  }
}
```

### 4.3 サーバー側のエラーハンドリング実装

```javascript
// サーバー側のエラーハンドリング
import { GraphQLError } from 'graphql';

// カスタムエラークラスの定義
class AppError extends GraphQLError {
  constructor(message, code, extensions = {}) {
    super(message, {
      extensions: {
        code,
        ...extensions,
      },
    });
  }
}

class AuthenticationError extends AppError {
  constructor(message = 'Not authenticated') {
    super(message, 'UNAUTHENTICATED', { http: { status: 401 } });
  }
}

class ForbiddenError extends AppError {
  constructor(message = 'Not authorized') {
    super(message, 'FORBIDDEN', { http: { status: 403 } });
  }
}

class NotFoundError extends AppError {
  constructor(resource, id) {
    super(`${resource} not found: ${id}`, 'NOT_FOUND', {
      http: { status: 404 },
      resource,
      resourceId: id,
    });
  }
}

class ValidationError extends AppError {
  constructor(errors) {
    super('Validation failed', 'VALIDATION_ERROR', {
      http: { status: 400 },
      validationErrors: errors,
    });
  }
}

class RateLimitError extends AppError {
  constructor(retryAfter) {
    super('Rate limit exceeded', 'RATE_LIMITED', {
      http: { status: 429 },
      retryAfter,
    });
  }
}

// リゾルバーでの使用
const resolvers = {
  Query: {
    user: async (_, { id }, context) => {
      // 認証チェック
      if (!context.user) {
        throw new AuthenticationError();
      }

      // 権限チェック
      if (!context.user.canViewUser(id)) {
        throw new ForbiddenError('You do not have permission to view this user');
      }

      const user = await context.dataSources.userAPI.getUser(id);
      if (!user) {
        throw new NotFoundError('User', id);
      }

      return user;
    },
  },

  Mutation: {
    createUser: async (_, { input }, context) => {
      // バリデーション
      const validationErrors = validateCreateUserInput(input);
      if (validationErrors.length > 0) {
        // Result型パターンの場合
        return {
          __typename: 'ValidationError',
          message: 'Validation failed',
          field: validationErrors[0].field,
          constraint: validationErrors[0].constraint,
          path: ['createUser'],
        };
      }

      try {
        const user = await context.dataSources.userAPI.create(input);
        return {
          __typename: 'CreateUserSuccess',
          user,
        };
      } catch (error) {
        if (error.code === 'UNIQUE_VIOLATION') {
          return {
            __typename: 'ValidationError',
            message: 'Email already exists',
            field: 'email',
            constraint: 'unique',
            path: ['createUser', 'input', 'email'],
          };
        }
        throw error; // 予期しないエラーはトップレベルに
      }
    },

    updateOrder: async (_, { id, input }, context) => {
      if (!context.user) {
        return {
          __typename: 'AuthorizationError',
          message: 'Authentication required',
          requiredPermission: 'orders:write',
          path: ['updateOrder'],
        };
      }

      const order = await context.dataSources.orderAPI.getOrder(id);
      if (!order) {
        return {
          __typename: 'NotFoundError',
          message: `Order not found: ${id}`,
          resourceType: 'Order',
          resourceId: id,
          path: ['updateOrder'],
        };
      }

      // ビジネスルールチェック
      if (order.status === 'SHIPPED' && input.status === 'CANCELLED') {
        return {
          __typename: 'BusinessRuleError',
          message: 'Cannot cancel a shipped order',
          code: 'ORDER_ALREADY_SHIPPED',
          details: { currentStatus: order.status, requestedStatus: input.status },
          path: ['updateOrder'],
        };
      }

      const updated = await context.dataSources.orderAPI.update(id, input);
      return {
        __typename: 'UpdateOrderSuccess',
        order: updated,
      };
    },
  },
};

// グローバルエラーフォーマッター
const server = new ApolloServer({
  typeDefs,
  resolvers,
  formatError: (formattedError, error) => {
    // 内部エラーの詳細をログに記録
    console.error('GraphQL Error:', {
      message: formattedError.message,
      code: formattedError.extensions?.code,
      path: formattedError.path,
      originalError: error,
    });

    // プロダクション環境では内部エラーの詳細を隠す
    if (process.env.NODE_ENV === 'production') {
      if (formattedError.extensions?.code === 'INTERNAL_SERVER_ERROR') {
        return {
          ...formattedError,
          message: 'An internal error occurred',
          extensions: {
            code: 'INTERNAL_SERVER_ERROR',
          },
        };
      }
    }

    // スタックトレースを削除
    delete formattedError.extensions?.stacktrace;

    return formattedError;
  },
});
```

### 4.4 クライアント側のエラーハンドリング

```typescript
// クライアント側のエラーハンドリング（React + Apollo Client）
import { ApolloError, useQuery, useMutation } from '@apollo/client';

// エラーリンクの設定
import { onError } from '@apollo/client/link/error';

const errorLink = onError(({ graphQLErrors, networkError, operation, forward }) => {
  if (graphQLErrors) {
    for (const error of graphQLErrors) {
      switch (error.extensions?.code) {
        case 'UNAUTHENTICATED':
          // トークンリフレッシュを試みる
          const oldHeaders = operation.getContext().headers;
          return fromPromise(
            refreshToken().then((newToken) => {
              operation.setContext({
                headers: {
                  ...oldHeaders,
                  authorization: `Bearer ${newToken}`,
                },
              });
              return forward(operation);
            })
          ).flatMap((result) => result);

        case 'FORBIDDEN':
          // 権限エラーページへリダイレクト
          window.location.href = '/forbidden';
          break;

        case 'RATE_LIMITED':
          // リトライ
          const retryAfter = error.extensions?.retryAfter || 60;
          console.warn(`Rate limited. Retrying after ${retryAfter}s`);
          break;

        default:
          console.error('GraphQL Error:', error.message);
      }
    }
  }

  if (networkError) {
    console.error('Network Error:', networkError);

    if ('statusCode' in networkError) {
      switch (networkError.statusCode) {
        case 503:
          // サービス一時停止
          showMaintenanceNotification();
          break;
        case 502:
        case 504:
          // ゲートウェイエラー → リトライ
          return forward(operation);
      }
    }
  }
});

// React コンポーネントでのエラーハンドリング
function UserProfile({ userId }: { userId: string }) {
  const { data, loading, error } = useQuery(GET_USER, {
    variables: { id: userId },
    errorPolicy: 'all', // エラーがあっても部分データを受け取る
  });

  if (loading) return <LoadingSpinner />;

  if (error) {
    // ネットワークエラー
    if (error.networkError) {
      return <NetworkErrorMessage onRetry={() => window.location.reload()} />;
    }

    // GraphQLエラー
    const authError = error.graphQLErrors?.find(
      (e) => e.extensions?.code === 'UNAUTHENTICATED'
    );
    if (authError) {
      return <LoginPrompt />;
    }

    const notFoundError = error.graphQLErrors?.find(
      (e) => e.extensions?.code === 'NOT_FOUND'
    );
    if (notFoundError) {
      return <NotFoundPage resource="User" />;
    }

    return <GenericErrorMessage error={error} />;
  }

  // 部分データの表示（errorsがあってもdataは利用可能）
  return (
    <div>
      <h1>{data.user.name}</h1>
      {data.user.orders ? (
        <OrderList orders={data.user.orders} />
      ) : (
        <p>注文データの取得に失敗しました</p>
      )}
    </div>
  );
}

// Mutation のエラーハンドリング（Result型パターン）
function CreateUserForm() {
  const [createUser, { loading }] = useMutation(CREATE_USER);

  const handleSubmit = async (input: CreateUserInput) => {
    try {
      const { data } = await createUser({ variables: { input } });

      const result = data.createUser;

      switch (result.__typename) {
        case 'CreateUserSuccess':
          toast.success('ユーザーが作成されました');
          navigate(`/users/${result.user.id}`);
          break;

        case 'ValidationError':
          toast.error(`${result.field}: ${result.message}`);
          break;

        case 'AuthorizationError':
          toast.error('権限がありません');
          break;
      }
    } catch (error) {
      // ネットワークエラー等の予期しないエラー
      if (error instanceof ApolloError) {
        toast.error('通信エラーが発生しました。再度お試しください。');
      }
    }
  };

  return <UserForm onSubmit={handleSubmit} loading={loading} />;
}
```

---

## 5. セキュリティ

### 5.1 GraphQL特有のセキュリティリスク

```
GraphQL特有のセキュリティリスク:

(1) クエリの深さ攻撃（Query Depth Attack）:
  query {
    user(id: "1") {
      orders {
        items {
          product {
            reviews {
              author {
                orders {        <- 再帰的にネスト
                  items { ... }
                }
              }
            }
          }
        }
      }
    }
  }
  -> 対策: クエリ深さの制限

(2) クエリの複雑度攻撃（Query Complexity Attack）:
  query {
    users(first: 1000) {
      orders(first: 1000) {
        items(first: 1000) { ... }
      }
    }
  }
  -> 対策: クエリコストの制限

(3) イントロスペクション悪用:
  query { __schema { types { name fields { name } } } }
  -> 対策: 本番では無効化

(4) Batch攻撃（Query Batching Attack）:
  [
    { "query": "query { user(id: \"1\") { ... } }" },
    { "query": "query { user(id: \"2\") { ... } }" },
    ... x 1000
  ]
  -> 対策: バッチサイズの制限

(5) フィールドサジェスション攻撃:
  query { user { passwrd } }
  -> "Did you mean 'password'?" がスキーマ情報を漏洩
  -> 対策: サジェスションの無効化

(6) Aliasベースの攻撃:
  query {
    a1: user(id: "1") { email }
    a2: user(id: "2") { email }
    ... x 1000
  }
  -> 同一フィールドをエイリアスで大量リクエスト
  -> 対策: エイリアス数の制限
```

### 5.2 セキュリティ対策の実装

```javascript
// セキュリティ対策の実装

// (1) クエリ深さ制限
import depthLimit from 'graphql-depth-limit';

// (2) クエリコスト分析
import {
  createComplexityRule,
  simpleEstimator,
  fieldExtensionsEstimator,
} from 'graphql-query-complexity';

// (3) クエリ数制限（エイリアス攻撃対策）
import { createComplexityLimitRule } from 'graphql-validation-complexity';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [
    // 最大7階層のネスト
    depthLimit(7, { ignore: ['__schema'] }),

    // クエリコスト制限
    createComplexityRule({
      maximumComplexity: 1000,
      estimators: [
        fieldExtensionsEstimator(),
        simpleEstimator({ defaultComplexity: 1 }),
      ],
      onComplete: (complexity) => {
        console.log('Query Complexity:', complexity);
        // メトリクス記録
        metrics.recordComplexity(complexity);
      },
    }),
  ],

  // イントロスペクションの無効化（本番環境）
  introspection: process.env.NODE_ENV !== 'production',

  // フィールドサジェスションの無効化
  includeStacktraceInErrorResponses: false,

  plugins: [
    // CSRF対策
    {
      async requestDidStart() {
        return {
          async didResolveOperation(requestContext) {
            // Content-Type チェック
            const contentType = requestContext.request.http?.headers.get('content-type');
            if (!contentType?.includes('application/json')) {
              throw new GraphQLError('Content-Type must be application/json');
            }
          },
        };
      },
    },

    // ロギングプラグイン
    {
      async requestDidStart(requestContext) {
        const start = Date.now();
        return {
          async willSendResponse(requestContext) {
            const duration = Date.now() - start;
            console.log({
              operation: requestContext.request.operationName,
              duration,
              errors: requestContext.errors?.length || 0,
            });
          },
        };
      },
    },
  ],
});

// (4) レート制限（フィールドレベル）
import { mapSchema, getDirective, MapperKind } from '@graphql-tools/utils';

const rateLimitDirectiveTypeDefs = `
  directive @rateLimit(
    max: Int!
    window: String!
    message: String
  ) on FIELD_DEFINITION
`;

function rateLimitDirective(directiveName = 'rateLimit') {
  return {
    rateLimitDirectiveTypeDefs,
    rateLimitDirectiveTransformer: (schema) =>
      mapSchema(schema, {
        [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
          const directive = getDirective(schema, fieldConfig, directiveName)?.[0];
          if (!directive) return fieldConfig;

          const { max, window: windowStr, message } = directive;
          const originalResolve = fieldConfig.resolve;

          fieldConfig.resolve = async (source, args, context, info) => {
            const key = `rateLimit:${context.user?.id || context.ip}:${info.fieldName}`;
            const current = await redis.incr(key);

            if (current === 1) {
              await redis.expire(key, parseWindow(windowStr));
            }

            if (current > max) {
              throw new GraphQLError(
                message || `Rate limit exceeded for ${info.fieldName}`,
                { extensions: { code: 'RATE_LIMITED' } }
              );
            }

            return originalResolve(source, args, context, info);
          };

          return fieldConfig;
        },
      }),
  };
}

// スキーマでの使用
const typeDefs = gql`
  ${rateLimitDirectiveTypeDefs}

  type Query {
    login(email: String!, password: String!): AuthPayload!
      @rateLimit(max: 5, window: "15m", message: "Too many login attempts")

    search(query: String!): [SearchResult!]!
      @rateLimit(max: 30, window: "1m")

    sendPasswordReset(email: String!): Boolean!
      @rateLimit(max: 3, window: "1h")
  }
`;
```

### 5.3 認可（Authorization）の実装

```javascript
// フィールドレベル認可
import { mapSchema, getDirective, MapperKind } from '@graphql-tools/utils';

const authDirectiveTypeDefs = `
  directive @auth(
    requires: [Role!]!
  ) on FIELD_DEFINITION | OBJECT

  enum Role {
    USER
    ADMIN
    SUPER_ADMIN
    MODERATOR
  }
`;

function authDirective(directiveName = 'auth') {
  return {
    authDirectiveTypeDefs,
    authDirectiveTransformer: (schema) =>
      mapSchema(schema, {
        [MapperKind.OBJECT_FIELD]: (fieldConfig) => {
          const directive = getDirective(schema, fieldConfig, directiveName)?.[0];
          if (!directive) return fieldConfig;

          const { requires } = directive;
          const originalResolve = fieldConfig.resolve;

          fieldConfig.resolve = async (source, args, context, info) => {
            if (!context.user) {
              throw new AuthenticationError();
            }

            const hasRole = requires.some((role) =>
              context.user.roles.includes(role)
            );

            if (!hasRole) {
              throw new ForbiddenError(
                `Requires one of: ${requires.join(', ')}`
              );
            }

            return originalResolve
              ? originalResolve(source, args, context, info)
              : source[info.fieldName];
          };

          return fieldConfig;
        },
      }),
  };
}

// スキーマでの使用
const typeDefs = gql`
  ${authDirectiveTypeDefs}

  type Query {
    me: User!
    users: [User!]! @auth(requires: [ADMIN])
    analytics: Analytics! @auth(requires: [ADMIN, SUPER_ADMIN])
    moderationQueue: [Report!]! @auth(requires: [MODERATOR, ADMIN])
  }

  type User {
    id: ID!
    name: String!
    email: String! @auth(requires: [ADMIN])
    phone: String @auth(requires: [ADMIN])
    orders: [Order!]!
    internalNotes: String @auth(requires: [ADMIN, SUPER_ADMIN])
  }

  type Mutation {
    deleteUser(id: ID!): Boolean! @auth(requires: [SUPER_ADMIN])
    banUser(id: ID!): User! @auth(requires: [MODERATOR, ADMIN])
  }
`;

// Persisted Queries（許可されたクエリのみ実行）
// → 最も強力なセキュリティ対策
import { readFileSync } from 'fs';
import { join } from 'path';

// ビルド時にクエリを抽出してホワイトリストを作成
const allowedQueries = new Map();
const queryFiles = fs.readdirSync('./queries');
queryFiles.forEach((file) => {
  const query = readFileSync(join('./queries', file), 'utf-8');
  const hash = crypto.createHash('sha256').update(query).digest('hex');
  allowedQueries.set(hash, query);
});

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    {
      async requestDidStart() {
        return {
          async didResolveOperation(requestContext) {
            if (process.env.NODE_ENV === 'production') {
              const queryHash = crypto
                .createHash('sha256')
                .update(requestContext.request.query)
                .digest('hex');

              if (!allowedQueries.has(queryHash)) {
                throw new GraphQLError('Query not allowed', {
                  extensions: { code: 'PERSISTED_QUERY_NOT_FOUND' },
                });
              }
            }
          },
        };
      },
    },
  ],
});

// バッチサイズ制限
// Express ミドルウェアで実装
app.use('/graphql', (req, res, next) => {
  if (Array.isArray(req.body)) {
    if (req.body.length > 10) {
      return res.status(400).json({
        errors: [{ message: 'Batch size exceeds maximum of 10' }],
      });
    }
  }
  next();
});
```

---

## 6. スキーマ設計パターン

### 6.1 インターフェースとユニオン型

```graphql
# (1) インターフェース（共通フィールドの定義）
interface Node {
  id: ID!
}

interface Timestamped {
  createdAt: DateTime!
  updatedAt: DateTime!
}

interface Auditable {
  createdBy: User!
  updatedBy: User
  version: Int!
}

type User implements Node & Timestamped {
  id: ID!
  name: String!
  email: String!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Product implements Node & Timestamped & Auditable {
  id: ID!
  name: String!
  price: Float!
  createdAt: DateTime!
  updatedAt: DateTime!
  createdBy: User!
  updatedBy: User
  version: Int!
}

# (2) ユニオン型（異なる型の集合）
union SearchResult = User | Product | Order | Category

type Query {
  search(query: String!, type: SearchResultType): [SearchResult!]!
}

enum SearchResultType {
  ALL
  USERS
  PRODUCTS
  ORDERS
}

# クエリ側のフラグメント活用
query Search($q: String!) {
  search(query: $q) {
    ... on User {
      id
      name
      email
    }
    ... on Product {
      id
      name
      price
      category { name }
    }
    ... on Order {
      id
      total
      status
      customer { name }
    }
  }
}

# (3) Relay Node仕様（Global Object Identification）
type Query {
  node(id: ID!): Node           # 任意のNodeをIDで取得
  nodes(ids: [ID!]!): [Node]!   # 複数のNodeを一括取得
  users(first: Int, after: String): UserConnection!
}

# Node IDはBase64エンコードされた "Type:id" 形式
# User:123 → "VXNlcjoxMjM="
# Product:456 → "UHJvZHVjdDo0NTY="

# (4) カスタムスカラー
scalar DateTime    # ISO 8601 日時
scalar JSON        # 任意のJSON
scalar URL         # URL文字列
scalar Email       # メールアドレス
scalar Currency    # 通貨コード（ISO 4217）
scalar PhoneNumber # E.164形式の電話番号

type User {
  id: ID!
  email: Email!
  phone: PhoneNumber
  website: URL
  metadata: JSON
  registeredAt: DateTime!
}
```

### 6.2 Relay Connection仕様（カーソルベースページネーション）

```graphql
# Relay Connection仕様
# → カーソルベースのページネーション標準

# Connection型の定義
type UserConnection {
  edges: [UserEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type UserEdge {
  node: User!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

type Query {
  # 前方ページネーション
  users(first: Int!, after: String, filter: UserFilter): UserConnection!

  # 後方ページネーション
  # users(last: Int!, before: String): UserConnection!
}

input UserFilter {
  name: StringFilter
  email: StringFilter
  status: UserStatus
  createdAt: DateRangeFilter
  roles: [Role!]
}

input StringFilter {
  eq: String
  contains: String
  startsWith: String
  in: [String!]
}

input DateRangeFilter {
  gte: DateTime
  lte: DateTime
}
```

```javascript
// Connection リゾルバーの実装
const resolvers = {
  Query: {
    users: async (_, { first, after, filter }, context) => {
      // カーソルのデコード
      const cursor = after ? decodeCursor(after) : null;

      // フィルタ条件の構築
      const where = buildWhereClause(filter);

      // N+1 を取得（hasNextPage判定のため）
      const limit = first + 1;

      let query = db('users').where(where).orderBy('created_at', 'desc').limit(limit);

      if (cursor) {
        query = query.where('created_at', '<', cursor.createdAt)
          .orWhere(function () {
            this.where('created_at', '=', cursor.createdAt)
              .where('id', '<', cursor.id);
          });
      }

      const rows = await query;

      const hasNextPage = rows.length > first;
      const nodes = hasNextPage ? rows.slice(0, first) : rows;

      // 総件数（オプション、パフォーマンスに注意）
      const [{ count: totalCount }] = await db('users').where(where).count('* as count');

      return {
        edges: nodes.map((node) => ({
          node,
          cursor: encodeCursor({
            id: node.id,
            createdAt: node.createdAt,
          }),
        })),
        pageInfo: {
          hasNextPage,
          hasPreviousPage: !!after,
          startCursor: nodes.length > 0
            ? encodeCursor({ id: nodes[0].id, createdAt: nodes[0].createdAt })
            : null,
          endCursor: nodes.length > 0
            ? encodeCursor({
                id: nodes[nodes.length - 1].id,
                createdAt: nodes[nodes.length - 1].createdAt,
              })
            : null,
        },
        totalCount: parseInt(totalCount, 10),
      };
    },
  },
};

// カーソルのエンコード/デコード
function encodeCursor(data) {
  return Buffer.from(JSON.stringify(data)).toString('base64');
}

function decodeCursor(cursor) {
  return JSON.parse(Buffer.from(cursor, 'base64').toString('utf-8'));
}

// フィルタ条件の構築
function buildWhereClause(filter) {
  if (!filter) return {};

  const conditions = {};

  if (filter.name?.contains) {
    conditions.name = ['ILIKE', `%${filter.name.contains}%`];
  }
  if (filter.status) {
    conditions.status = filter.status;
  }
  if (filter.createdAt?.gte) {
    conditions['created_at >='] = filter.createdAt.gte;
  }
  if (filter.createdAt?.lte) {
    conditions['created_at <='] = filter.createdAt.lte;
  }

  return conditions;
}
```

### 6.3 ディレクティブの活用

```graphql
# カスタムディレクティブ
directive @auth(requires: [Role!]!) on FIELD_DEFINITION | OBJECT
directive @deprecated(reason: String!) on FIELD_DEFINITION | ENUM_VALUE
directive @cacheControl(maxAge: Int!, scope: CacheControlScope) on FIELD_DEFINITION | OBJECT
directive @rateLimit(max: Int!, window: String!) on FIELD_DEFINITION
directive @log(level: LogLevel = INFO) on FIELD_DEFINITION
directive @computed on FIELD_DEFINITION
directive @validate(
  min: Int
  max: Int
  minLength: Int
  maxLength: Int
  pattern: String
  email: Boolean
) on INPUT_FIELD_DEFINITION | ARGUMENT_DEFINITION

enum LogLevel {
  DEBUG
  INFO
  WARN
  ERROR
}

# ディレクティブの利用例
type Query {
  publicData: String!
  sensitiveData: String! @auth(requires: [ADMIN])
  oldField: String @deprecated(reason: "Use newField instead")
  cachedProducts: [Product!]! @cacheControl(maxAge: 3600)
  criticalOperation: Result! @rateLimit(max: 10, window: "1m") @log(level: WARN)
}

type User @auth(requires: [USER]) {
  id: ID!
  name: String!
  email: String! @auth(requires: [ADMIN])
  fullName: String! @computed
}

input CreateUserInput {
  name: String! @validate(minLength: 2, maxLength: 50)
  email: String! @validate(email: true)
  age: Int @validate(min: 0, max: 150)
  password: String! @validate(minLength: 8, pattern: "^(?=.*[A-Z])(?=.*[0-9])")
}
```

### 6.4 Input型とMutation設計パターン

```graphql
# Mutation設計のベストプラクティス

# (1) 単一のInput型を使用
input CreateUserInput {
  name: String!
  email: String!
  password: String!
  profile: CreateProfileInput
}

input CreateProfileInput {
  bio: String
  avatar: URL
  location: String
}

input UpdateUserInput {
  name: String
  email: String
  profile: UpdateProfileInput
}

input UpdateProfileInput {
  bio: String
  avatar: URL
  location: String
}

# (2) Payload型で結果を返す
type CreateUserPayload {
  user: User
  errors: [UserError!]!
}

type DeleteUserPayload {
  deletedUserId: ID
  errors: [UserError!]!
}

# (3) 一貫性のあるMutation命名
type Mutation {
  # CRUD操作: create/update/delete + リソース名
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!

  # アクション: 動詞 + リソース名
  activateUser(id: ID!): ActivateUserPayload!
  deactivateUser(id: ID!): DeactivateUserPayload!
  resetPassword(email: String!): ResetPasswordPayload!
  verifyEmail(token: String!): VerifyEmailPayload!

  # 関連リソースの操作
  addUserToTeam(userId: ID!, teamId: ID!): AddUserToTeamPayload!
  removeUserFromTeam(userId: ID!, teamId: ID!): RemoveUserFromTeamPayload!

  # バッチ操作
  bulkCreateUsers(inputs: [CreateUserInput!]!): BulkCreateUsersPayload!
  bulkDeleteUsers(ids: [ID!]!): BulkDeleteUsersPayload!
}

# (4) ファイルアップロード
scalar Upload

type Mutation {
  uploadAvatar(file: Upload!): UploadAvatarPayload!
  uploadDocument(file: Upload!, metadata: DocumentMetadataInput!): UploadDocumentPayload!
}

input DocumentMetadataInput {
  title: String!
  description: String
  category: DocumentCategory!
  tags: [String!]
}
```

---

## 7. Apollo Federation（マイクロサービス統合）

### 7.1 Federationの概要

```
Apollo Federation アーキテクチャ:

  ┌─────────────────────────────────────┐
  │           Apollo Gateway             │
  │  (統合GraphQLエンドポイント)           │
  │  クエリプランニング & 実行             │
  └───┬──────────┬──────────┬───────────┘
      │          │          │
  ┌───┴───┐  ┌──┴───┐  ┌──┴───┐
  │ Users │  │Orders│  │Prods │
  │Service│  │Srvce │  │Srvce │
  │ :4001 │  │:4002 │  │:4003 │
  └───────┘  └──────┘  └──────┘
      │          │          │
  ┌───┴───┐  ┌──┴───┐  ┌──┴───┐
  │UserDB │  │OrderDB│  │ProdDB│
  └───────┘  └──────┘  └──────┘

  利点:
  → 各サービスが独立してデプロイ可能
  → チームごとにスキーマを管理
  → 単一のGraphQLエンドポイントをクライアントに提供
  → スキーマの型をサービス間で共有（Entity）
```

### 7.2 Subgraph（サブグラフ）の定義

```graphql
# === Users Service (Subgraph) ===
# users-service/schema.graphql

extend schema @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@shareable"])

type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
  role: Role!
  createdAt: DateTime!
}

enum Role {
  USER
  ADMIN
}

type Query {
  me: User
  user(id: ID!): User
  users(first: Int!, after: String): UserConnection!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(id: ID!, input: UpdateUserInput!): UpdateUserPayload!
}
```

```graphql
# === Orders Service (Subgraph) ===
# orders-service/schema.graphql

extend schema @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key", "@external", "@requires"])

# Users ServiceのUser型を拡張
type User @key(fields: "id") {
  id: ID!
  orders(first: Int!, after: String): OrderConnection!
  totalSpent: Float! @requires(fields: "id")
}

type Order @key(fields: "id") {
  id: ID!
  customer: User!
  items: [OrderItem!]!
  total: Float!
  status: OrderStatus!
  createdAt: DateTime!
}

type OrderItem {
  product: Product!
  quantity: Int!
  unitPrice: Float!
  subtotal: Float!
}

# Products Serviceの型を参照
type Product @key(fields: "id") {
  id: ID!
}

enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
}

type Query {
  order(id: ID!): Order
  orders(filter: OrderFilter): OrderConnection!
}

type Mutation {
  createOrder(input: CreateOrderInput!): CreateOrderPayload!
  cancelOrder(id: ID!): CancelOrderPayload!
}
```

```graphql
# === Products Service (Subgraph) ===
# products-service/schema.graphql

extend schema @link(url: "https://specs.apollo.dev/federation/v2.0", import: ["@key"])

type Product @key(fields: "id") {
  id: ID!
  name: String!
  description: String!
  price: Float!
  category: Category!
  inventory: Int!
  reviews: [Review!]!
}

type Category @key(fields: "id") {
  id: ID!
  name: String!
  products(first: Int!, after: String): ProductConnection!
}

type Review {
  id: ID!
  author: User!
  rating: Int!
  comment: String!
  createdAt: DateTime!
}

type User @key(fields: "id") {
  id: ID!
}

type Query {
  product(id: ID!): Product
  products(filter: ProductFilter, first: Int!, after: String): ProductConnection!
  categories: [Category!]!
}
```

### 7.3 Gatewayの設定

```javascript
// Apollo Gateway の設定
import { ApolloServer } from '@apollo/server';
import { ApolloGateway, IntrospectAndCompose, RemoteGraphQLDataSource } from '@apollo/gateway';
import { startStandaloneServer } from '@apollo/server/standalone';

const gateway = new ApolloGateway({
  supergraphSdl: new IntrospectAndCompose({
    subgraphs: [
      { name: 'users', url: 'http://users-service:4001/graphql' },
      { name: 'orders', url: 'http://orders-service:4002/graphql' },
      { name: 'products', url: 'http://products-service:4003/graphql' },
    ],
    pollIntervalInMs: 10000, // 10秒ごとにスキーマ更新をチェック
  }),

  // カスタムDataSource（認証ヘッダーの転送）
  buildService({ url }) {
    return new RemoteGraphQLDataSource({
      url,
      willSendRequest({ request, context }) {
        // クライアントの認証情報をサブグラフに転送
        if (context.token) {
          request.http.headers.set('authorization', context.token);
        }
        // リクエストIDの伝播（分散トレーシング）
        if (context.requestId) {
          request.http.headers.set('x-request-id', context.requestId);
        }
      },
      didReceiveResponse({ response, context }) {
        // レスポンスヘッダーの処理
        const cacheControl = response.http.headers.get('cache-control');
        if (cacheControl) {
          context.cacheControl = cacheControl;
        }
        return response;
      },
    });
  },
});

const server = new ApolloServer({
  gateway,
  // Gateway固有のプラグイン
  plugins: [
    {
      async requestDidStart() {
        const start = Date.now();
        return {
          async willSendResponse(requestContext) {
            const duration = Date.now() - start;
            // Gateway レベルのメトリクス記録
            metrics.recordGatewayLatency(
              requestContext.request.operationName,
              duration
            );
          },
        };
      },
    },
  ],
});

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => ({
    token: req.headers.authorization,
    requestId: req.headers['x-request-id'] || crypto.randomUUID(),
  }),
  listen: { port: 4000 },
});

console.log(`Gateway running at ${url}`);
```

### 7.4 サブグラフのリゾルバー実装

```javascript
// Users Service のリゾルバー
import { buildSubgraphSchema } from '@apollo/subgraph';

const resolvers = {
  Query: {
    me: (_, __, context) => context.dataSources.userAPI.getUser(context.userId),
    user: (_, { id }, context) => context.dataSources.userAPI.getUser(id),
    users: (_, args, context) => context.dataSources.userAPI.getUsers(args),
  },

  User: {
    // __resolveReference: Federation がEntity解決時に呼ぶ
    __resolveReference: (ref, context) => {
      return context.dataSources.userAPI.getUser(ref.id);
    },
  },
};

const server = new ApolloServer({
  schema: buildSubgraphSchema({ typeDefs, resolvers }),
});

// Orders Service のリゾルバー
const orderResolvers = {
  Query: {
    order: (_, { id }, context) => context.dataSources.orderAPI.getOrder(id),
    orders: (_, { filter }, context) => context.dataSources.orderAPI.getOrders(filter),
  },

  User: {
    // User型の拡張フィールド
    orders: (user, args, context) => {
      return context.dataSources.orderAPI.getOrdersByUser(user.id, args);
    },
    totalSpent: async (user, _, context) => {
      const orders = await context.dataSources.orderAPI.getOrdersByUser(user.id);
      return orders.reduce((sum, order) => sum + order.total, 0);
    },
  },

  Order: {
    customer: (order) => ({ __typename: 'User', id: order.customerId }),
    items: (order, _, context) => {
      return context.dataSources.orderAPI.getOrderItems(order.id);
    },
  },

  OrderItem: {
    product: (item) => ({ __typename: 'Product', id: item.productId }),
  },
};
```

---

## 8. パフォーマンスチューニング

### 8.1 クエリパフォーマンスの計測

```javascript
// パフォーマンス計測プラグイン
const performancePlugin = {
  async requestDidStart(requestContext) {
    const start = process.hrtime.bigint();
    const resolverTimings = [];

    return {
      async executionDidStart() {
        return {
          willResolveField({ info }) {
            const fieldStart = process.hrtime.bigint();

            return (error, result) => {
              const duration = Number(process.hrtime.bigint() - fieldStart) / 1e6;
              resolverTimings.push({
                path: info.path,
                parentType: info.parentType.name,
                fieldName: info.fieldName,
                returnType: info.returnType.toString(),
                duration,
                error: error?.message,
              });
            };
          },
        };
      },

      async willSendResponse(requestContext) {
        const totalDuration = Number(process.hrtime.bigint() - start) / 1e6;

        // 遅いリゾルバーの検出（100ms以上）
        const slowResolvers = resolverTimings.filter((t) => t.duration > 100);

        if (slowResolvers.length > 0) {
          console.warn('Slow resolvers detected:', {
            operation: requestContext.request.operationName,
            totalDuration: `${totalDuration.toFixed(2)}ms`,
            slowResolvers: slowResolvers.map((r) => ({
              path: printPath(r.path),
              duration: `${r.duration.toFixed(2)}ms`,
            })),
          });
        }

        // メトリクス送信
        await metrics.send({
          operation: requestContext.request.operationName,
          totalDuration,
          resolverCount: resolverTimings.length,
          slowResolverCount: slowResolvers.length,
          errors: requestContext.errors?.length || 0,
        });

        // 開発環境ではレスポンスにタイミング情報を追加
        if (process.env.NODE_ENV === 'development') {
          requestContext.response.extensions = {
            ...requestContext.response.extensions,
            tracing: {
              totalDuration: `${totalDuration.toFixed(2)}ms`,
              resolvers: resolverTimings.map((t) => ({
                path: printPath(t.path),
                duration: `${t.duration.toFixed(2)}ms`,
              })),
            },
          };
        }
      },
    };
  },
};

function printPath(path) {
  const parts = [];
  let current = path;
  while (current) {
    parts.unshift(current.key);
    current = current.prev;
  }
  return parts.join('.');
}
```

### 8.2 クエリの最適化テクニック

```javascript
// 1. フィールドレベルの遅延解決（必要なフィールドのみ解決）
const resolvers = {
  User: {
    // ordersフィールドがクエリに含まれている場合のみ実行される
    orders: async (user, args, context, info) => {
      // info.fieldNodesからサブフィールドを確認
      const requestedFields = getRequestedFields(info);

      // 必要なフィールドのみSELECT
      const selectFields = mapFieldsToColumns(requestedFields);

      return context.dataSources.orderAPI.getOrdersByUser(user.id, {
        select: selectFields,
        ...args,
      });
    },

    // 重い計算フィールドの最適化
    statistics: async (user, _, context, info) => {
      // statisticsフィールドが実際にリクエストされたサブフィールドを確認
      const requestedStats = getRequestedFields(info);

      const result = {};

      // リクエストされた統計のみ計算
      if (requestedStats.includes('orderCount')) {
        result.orderCount = await context.loaders.orderCountLoader.load(user.id);
      }
      if (requestedStats.includes('totalSpent')) {
        result.totalSpent = await context.loaders.totalSpentLoader.load(user.id);
      }
      if (requestedStats.includes('averageOrderValue')) {
        const [count, total] = await Promise.all([
          context.loaders.orderCountLoader.load(user.id),
          context.loaders.totalSpentLoader.load(user.id),
        ]);
        result.averageOrderValue = count > 0 ? total / count : 0;
      }

      return result;
    },
  },
};

// 2. @defer / @stream ディレクティブ（段階的レスポンス）
// GraphQL Incremental Delivery
const GET_USER_WITH_DEFER = gql`
  query GetUser($id: ID!) {
    user(id: $id) {
      id
      name
      email
      # 重いフィールドを遅延ロード
      ... @defer(label: "orders") {
        orders(first: 10) {
          edges {
            node {
              id
              total
              status
            }
          }
        }
      }
      ... @defer(label: "recommendations") {
        recommendations {
          products {
            id
            name
            price
          }
        }
      }
    }
  }
`;

// クライアント側での @defer 利用
function UserProfile({ userId }) {
  const { data, loading } = useQuery(GET_USER_WITH_DEFER, {
    variables: { id: userId },
  });

  return (
    <div>
      {/* 基本情報は即座に表示 */}
      <h1>{data?.user?.name}</h1>
      <p>{data?.user?.email}</p>

      {/* 注文は遅延ロード */}
      <Suspense fallback={<OrdersSkeleton />}>
        <OrderList orders={data?.user?.orders} />
      </Suspense>

      {/* レコメンデーションも遅延ロード */}
      <Suspense fallback={<RecommendationsSkeleton />}>
        <Recommendations items={data?.user?.recommendations} />
      </Suspense>
    </div>
  );
}

// 3. クエリプランの最適化
// lookahead パターン: 親リゾルバーで子の必要データを先読み
const resolvers = {
  Query: {
    users: async (_, args, context, info) => {
      // 子フィールドで何が要求されているか確認
      const selections = info.fieldNodes[0].selectionSet;
      const needsOrders = hasField(selections, 'orders');
      const needsProfile = hasField(selections, 'profile');

      // JOINまたはサブクエリで一括取得
      let query = db('users').select('users.*');

      if (needsProfile) {
        query = query.leftJoin('profiles', 'users.id', 'profiles.user_id')
          .select('profiles.bio', 'profiles.avatar');
      }

      const users = await query.where(buildFilter(args.filter)).limit(args.first);

      // DataLoaderにプライミング
      if (needsOrders) {
        const userIds = users.map(u => u.id);
        const allOrders = await db('orders').whereIn('user_id', userIds);
        const ordersByUser = groupBy(allOrders, 'userId');
        userIds.forEach(id => {
          context.loaders.ordersByUserLoader.prime(id, ordersByUser[id] || []);
        });
      }

      return users;
    },
  },
};
```

### 8.3 プロダクション運用のベストプラクティス

```javascript
// Apollo Server のプロダクション設定
import { ApolloServer } from '@apollo/server';
import { ApolloServerPluginLandingPageDisabled } from '@apollo/server/plugin/disabled';
import { ApolloServerPluginUsageReporting } from '@apollo/server/plugin/usageReporting';

const server = new ApolloServer({
  typeDefs,
  resolvers,

  // プロダクション設定
  introspection: false,
  includeStacktraceInErrorResponses: false,

  plugins: [
    // ランディングページの無効化
    ApolloServerPluginLandingPageDisabled(),

    // Apollo Studio へのメトリクス送信
    ApolloServerPluginUsageReporting({
      sendVariableValues: { none: true }, // 変数値を送信しない
      sendHeaders: { none: true },        // ヘッダーを送信しない
    }),

    // パフォーマンス計測
    performancePlugin,

    // リクエストログ
    {
      async requestDidStart({ request }) {
        return {
          async didEncounterErrors({ errors }) {
            errors.forEach((error) => {
              // エラーログ（Sentry等に送信）
              Sentry.captureException(error.originalError || error, {
                extra: {
                  query: request.query,
                  variables: request.variables,
                  operationName: request.operationName,
                },
              });
            });
          },
        };
      },
    },
  ],

  // リクエストボディサイズの制限
  // expressMiddleware側で設定
});

// Express設定
app.use(
  '/graphql',
  express.json({ limit: '1mb' }), // リクエストサイズ制限
  expressMiddleware(server, {
    context: async ({ req }) => ({
      user: await authenticateUser(req),
      loaders: createLoaders(db),
      dataSources: createDataSources(),
      requestId: req.headers['x-request-id'] || crypto.randomUUID(),
    }),
  })
);

// ヘルスチェックエンドポイント
app.get('/health', async (req, res) => {
  try {
    // DB接続チェック
    await db.raw('SELECT 1');
    // Redisチェック
    await redis.ping();
    res.json({ status: 'healthy', timestamp: new Date().toISOString() });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
});
```

---

## 9. テスト戦略

### 9.1 リゾルバーの単体テスト

```javascript
// リゾルバーの単体テスト（Jest）
import { resolvers } from '../resolvers';

describe('User resolvers', () => {
  const mockContext = {
    user: { id: '1', roles: ['USER'] },
    dataSources: {
      userAPI: {
        getUser: jest.fn(),
        create: jest.fn(),
      },
    },
    loaders: {
      userLoader: {
        load: jest.fn(),
      },
      ordersByUserLoader: {
        load: jest.fn(),
      },
    },
  };

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Query.user', () => {
    it('should return user by id', async () => {
      const mockUser = { id: '1', name: 'Alice', email: 'alice@example.com' };
      mockContext.dataSources.userAPI.getUser.mockResolvedValue(mockUser);

      const result = await resolvers.Query.user(
        null,
        { id: '1' },
        mockContext
      );

      expect(result).toEqual(mockUser);
      expect(mockContext.dataSources.userAPI.getUser).toHaveBeenCalledWith('1');
    });

    it('should throw AuthenticationError when not authenticated', async () => {
      const unauthContext = { ...mockContext, user: null };

      await expect(
        resolvers.Query.user(null, { id: '1' }, unauthContext)
      ).rejects.toThrow('Not authenticated');
    });

    it('should throw NotFoundError when user does not exist', async () => {
      mockContext.dataSources.userAPI.getUser.mockResolvedValue(null);

      await expect(
        resolvers.Query.user(null, { id: '999' }, mockContext)
      ).rejects.toThrow('User not found');
    });
  });

  describe('Mutation.createUser', () => {
    it('should create user successfully', async () => {
      const input = { name: 'Bob', email: 'bob@example.com', password: 'Pass123!' };
      const mockUser = { id: '2', ...input };
      mockContext.dataSources.userAPI.create.mockResolvedValue(mockUser);

      const result = await resolvers.Mutation.createUser(
        null,
        { input },
        mockContext
      );

      expect(result.__typename).toBe('CreateUserSuccess');
      expect(result.user).toEqual(mockUser);
    });

    it('should return ValidationError for invalid email', async () => {
      const input = { name: 'Bob', email: 'invalid', password: 'Pass123!' };

      const result = await resolvers.Mutation.createUser(
        null,
        { input },
        mockContext
      );

      expect(result.__typename).toBe('ValidationError');
      expect(result.field).toBe('email');
    });
  });

  describe('User.orders', () => {
    it('should load orders via DataLoader', async () => {
      const mockOrders = [
        { id: 'o1', total: 100 },
        { id: 'o2', total: 200 },
      ];
      mockContext.loaders.ordersByUserLoader.load.mockResolvedValue(mockOrders);

      const result = await resolvers.User.orders(
        { id: '1' },
        {},
        mockContext
      );

      expect(result).toEqual(mockOrders);
      expect(mockContext.loaders.ordersByUserLoader.load).toHaveBeenCalledWith('1');
    });
  });
});
```

### 9.2 統合テスト

```javascript
// GraphQL統合テスト（Apollo Server + Supertest）
import { ApolloServer } from '@apollo/server';
import request from 'supertest';
import { createTestServer, createTestDatabase } from '../test/helpers';

describe('GraphQL API Integration Tests', () => {
  let server;
  let testDb;

  beforeAll(async () => {
    testDb = await createTestDatabase();
    server = await createTestServer(testDb);
  });

  afterAll(async () => {
    await testDb.destroy();
    await server.stop();
  });

  beforeEach(async () => {
    // テストデータの投入
    await testDb.seed.run();
  });

  afterEach(async () => {
    // テストデータのクリア
    await testDb.raw('TRUNCATE users, orders, products CASCADE');
  });

  describe('Users Query', () => {
    it('should fetch paginated users', async () => {
      const query = `
        query GetUsers($first: Int!, $after: String) {
          users(first: $first, after: $after) {
            edges {
              node {
                id
                name
                email
              }
              cursor
            }
            pageInfo {
              hasNextPage
              endCursor
            }
            totalCount
          }
        }
      `;

      const response = await request(server.app)
        .post('/graphql')
        .set('Authorization', 'Bearer test-admin-token')
        .send({
          query,
          variables: { first: 5 },
        });

      expect(response.status).toBe(200);
      expect(response.body.errors).toBeUndefined();

      const { users } = response.body.data;
      expect(users.edges).toHaveLength(5);
      expect(users.pageInfo.hasNextPage).toBe(true);
      expect(users.totalCount).toBeGreaterThan(5);
    });

    it('should handle cursor-based pagination', async () => {
      // 1ページ目
      const page1 = await graphqlRequest(server, {
        query: GET_USERS,
        variables: { first: 3 },
      });

      const endCursor = page1.data.users.pageInfo.endCursor;

      // 2ページ目
      const page2 = await graphqlRequest(server, {
        query: GET_USERS,
        variables: { first: 3, after: endCursor },
      });

      // 重複がないことを確認
      const page1Ids = page1.data.users.edges.map(e => e.node.id);
      const page2Ids = page2.data.users.edges.map(e => e.node.id);
      const intersection = page1Ids.filter(id => page2Ids.includes(id));
      expect(intersection).toHaveLength(0);
    });
  });

  describe('Create User Mutation', () => {
    it('should create user and return via subscription', async () => {
      const mutation = `
        mutation CreateUser($input: CreateUserInput!) {
          createUser(input: $input) {
            ... on CreateUserSuccess {
              user {
                id
                name
                email
              }
            }
            ... on ValidationError {
              message
              field
            }
          }
        }
      `;

      const response = await graphqlRequest(server, {
        query: mutation,
        variables: {
          input: {
            name: 'Test User',
            email: 'test@example.com',
            password: 'SecurePass123!',
          },
        },
      });

      expect(response.data.createUser.__typename).toBe('CreateUserSuccess');
      expect(response.data.createUser.user.name).toBe('Test User');

      // DBに保存されていることを確認
      const dbUser = await testDb('users')
        .where({ email: 'test@example.com' })
        .first();
      expect(dbUser).toBeTruthy();
      expect(dbUser.name).toBe('Test User');
    });
  });
});
```

### 9.3 スキーマテスト

```javascript
// スキーマの構造テスト
import { buildSchema, validateSchema, introspectionFromSchema } from 'graphql';

describe('GraphQL Schema', () => {
  const schema = buildSchema(typeDefs);

  it('should have no validation errors', () => {
    const errors = validateSchema(schema);
    expect(errors).toHaveLength(0);
  });

  it('should have required query fields', () => {
    const queryType = schema.getQueryType();
    const fields = queryType.getFields();

    expect(fields).toHaveProperty('user');
    expect(fields).toHaveProperty('users');
    expect(fields).toHaveProperty('me');
    expect(fields).toHaveProperty('products');
  });

  it('should have required mutation fields', () => {
    const mutationType = schema.getMutationType();
    const fields = mutationType.getFields();

    expect(fields).toHaveProperty('createUser');
    expect(fields).toHaveProperty('updateUser');
    expect(fields).toHaveProperty('deleteUser');
  });

  it('should have Node interface implemented correctly', () => {
    const userType = schema.getType('User');
    const interfaces = userType.getInterfaces();
    const nodeInterface = interfaces.find(i => i.name === 'Node');

    expect(nodeInterface).toBeDefined();
    expect(userType.getFields()).toHaveProperty('id');
  });

  // スキーマの破壊的変更チェック
  it('should not have breaking changes from previous version', async () => {
    const { findBreakingChanges } = await import('graphql');

    const oldSchema = buildSchema(readFileSync('./schema-v1.graphql', 'utf-8'));
    const newSchema = schema;

    const breakingChanges = findBreakingChanges(oldSchema, newSchema);

    // 許容される破壊的変更がある場合はフィルタ
    const unexpectedChanges = breakingChanges.filter(
      change => !allowedBreakingChanges.includes(change.description)
    );

    expect(unexpectedChanges).toHaveLength(0);
  });
});
```

---

## 10. モニタリングとオブザーバビリティ

### 10.1 メトリクス収集

```javascript
// Prometheus メトリクスの収集
import { register, Counter, Histogram, Gauge } from 'prom-client';

// メトリクス定義
const graphqlRequestDuration = new Histogram({
  name: 'graphql_request_duration_seconds',
  help: 'Duration of GraphQL requests',
  labelNames: ['operation', 'operationType', 'status'],
  buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
});

const graphqlResolverDuration = new Histogram({
  name: 'graphql_resolver_duration_seconds',
  help: 'Duration of individual GraphQL resolvers',
  labelNames: ['parentType', 'fieldName'],
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
});

const graphqlErrors = new Counter({
  name: 'graphql_errors_total',
  help: 'Total number of GraphQL errors',
  labelNames: ['code', 'operation'],
});

const graphqlComplexity = new Histogram({
  name: 'graphql_query_complexity',
  help: 'Complexity of GraphQL queries',
  labelNames: ['operation'],
  buckets: [10, 50, 100, 200, 500, 1000],
});

const activeSubscriptions = new Gauge({
  name: 'graphql_active_subscriptions',
  help: 'Number of active GraphQL subscriptions',
  labelNames: ['subscription'],
});

// メトリクス収集プラグイン
const metricsPlugin = {
  async requestDidStart({ request }) {
    const timer = graphqlRequestDuration.startTimer();

    return {
      async executionDidStart() {
        return {
          willResolveField({ info }) {
            const resolverTimer = graphqlResolverDuration.startTimer({
              parentType: info.parentType.name,
              fieldName: info.fieldName,
            });

            return () => resolverTimer();
          },
        };
      },

      async willSendResponse({ response }) {
        const operationType = request.query?.includes('mutation')
          ? 'mutation'
          : request.query?.includes('subscription')
            ? 'subscription'
            : 'query';

        const status = response.body?.singleResult?.errors ? 'error' : 'success';

        timer({
          operation: request.operationName || 'anonymous',
          operationType,
          status,
        });
      },

      async didEncounterErrors({ errors }) {
        errors.forEach((error) => {
          graphqlErrors.inc({
            code: error.extensions?.code || 'UNKNOWN',
            operation: request.operationName || 'anonymous',
          });
        });
      },
    };
  },
};

// メトリクスエンドポイント
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

### 10.2 分散トレーシング

```javascript
// OpenTelemetry による分散トレーシング
import { trace, SpanStatusCode } from '@opentelemetry/api';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { GraphQLInstrumentation } from '@opentelemetry/instrumentation-graphql';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';

// トレーサープロバイダーの設定
const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new BatchSpanProcessor(
    new JaegerExporter({
      endpoint: 'http://jaeger:14268/api/traces',
    })
  )
);
provider.register();

// GraphQL 自動計装
const graphqlInstrumentation = new GraphQLInstrumentation({
  mergeItems: true,
  allowValues: process.env.NODE_ENV !== 'production',
  depth: 5,
});

graphqlInstrumentation.setTracerProvider(provider);
graphqlInstrumentation.enable();

// カスタムスパンの追加
const tracer = trace.getTracer('graphql-api');

const resolvers = {
  Query: {
    users: async (_, args, context) => {
      return tracer.startActiveSpan('fetchUsers', async (span) => {
        try {
          span.setAttribute('user.filter', JSON.stringify(args.filter));
          span.setAttribute('user.first', args.first);

          const result = await context.dataSources.userAPI.getUsers(args);

          span.setAttribute('user.count', result.edges.length);
          span.setStatus({ code: SpanStatusCode.OK });

          return result;
        } catch (error) {
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.recordException(error);
          throw error;
        } finally {
          span.end();
        }
      });
    },
  },
};
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Subscription | WebSocket + PubSub でリアルタイム、Redis PubSubでスケーリング |
| DataLoader | バッチ処理でN+1問題を解消、リクエストごとにインスタンス作成 |
| キャッシュ | 正規化キャッシュ + Persisted Queries + サーバーサイドRedis |
| エラーハンドリング | トップレベルエラー vs Payload内エラー、Result型パターン |
| セキュリティ | 深さ制限、コスト制限、レート制限、イントロスペクション無効化 |
| スキーマ設計 | Interface、Union、Relay Connection、カスタムディレクティブ |
| Federation | マイクロサービス統合、Entity解決、Gateway |
| パフォーマンス | リゾルバー計測、@defer、クエリプラン最適化 |
| テスト | 単体テスト、統合テスト、スキーマテスト |
| モニタリング | Prometheus メトリクス、OpenTelemetry トレーシング |

---

## 次に読むべきガイド
-> [[03-rest-vs-graphql.md]] -- REST vs GraphQL

---

## 参考文献
1. Apollo. "Production Readiness Checklist." apollographql.com, 2024.
2. Facebook. "DataLoader." github.com/graphql/dataloader, 2024.
3. Relay. "Relay Specification." relay.dev, 2024.
4. Apollo. "Apollo Federation." apollographql.com/docs/federation, 2024.
5. GraphQL Foundation. "GraphQL Specification." spec.graphql.org, 2024.
6. Marc-Andre Giroux. "Production Ready GraphQL." book.productionreadygraphql.com, 2024.
7. OpenTelemetry. "GraphQL Instrumentation." opentelemetry.io, 2024.
8. Lee Byron et al. "GraphQL Subscriptions." github.com/graphql/graphql-spec, 2024.
