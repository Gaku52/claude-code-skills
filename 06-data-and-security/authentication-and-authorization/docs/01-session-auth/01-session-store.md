# セッションストア

> セッションデータの保存先はアプリケーションのスケーラビリティとパフォーマンスに直結する。メモリ、Redis、データベースの各セッションストアの特徴と選定基準、Redis を使ったスケーラブルなセッション管理を解説する。

## この章で学ぶこと

- [ ] セッションストアの種類と選定基準を理解する
- [ ] Redis セッションストアの設計と実装を把握する
- [ ] スケーリング戦略とセッションの永続化を学ぶ

---

## 1. セッションストアの比較

```
セッションストアの種類:

  ストア     │ 速度  │ スケール │ 永続化 │ 用途
  ──────────┼──────┼────────┼──────┼──────────────
  メモリ     │ ◎最速 │ ✗ 不可  │ ✗     │ 開発/単一サーバー
  Redis     │ ○ 高速│ ✓ 可能  │ △     │ 本番推奨
  PostgreSQL│ △ 中  │ ✓ 可能  │ ✓     │ 追加インフラ不要
  MongoDB   │ △ 中  │ ✓ 可能  │ ✓     │ MongoDB使用中
  DynamoDB  │ ○ 高速│ ✓ 自動  │ ✓     │ AWS環境

メモリストア:
  利点: 最速、設定不要
  欠点: サーバー再起動で消失、複数サーバーで共有不可
  用途: 開発環境、プロトタイプ

Redis:
  利点: 高速、TTL自動削除、Pub/Subでリアルタイム通知
  欠点: 追加インフラ、メモリ消費
  用途: 本番環境の第一選択

データベース:
  利点: 既存インフラ活用、永続化、クエリ可能
  欠点: I/Oが遅い、定期クリーンアップ必要
  用途: Redis導入が困難な場合
```

---

## 2. Redis セッションストア

```typescript
// Redis セッションストアの実装
import Redis from 'ioredis';

interface SessionData {
  userId: string;
  role: string;
  createdAt: number;
  lastAccessedAt: number;
  metadata: Record<string, any>;
}

class RedisSessionStore {
  private redis: Redis;
  private prefix = 'sess:';
  private userPrefix = 'user_sess:';

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl, {
      retryStrategy: (times) => Math.min(times * 50, 2000),
      maxRetriesPerRequest: 3,
    });
  }

  // セッション保存
  async set(sessionId: string, data: SessionData, ttl: number): Promise<void> {
    const key = this.prefix + sessionId;
    const pipeline = this.redis.pipeline();

    // セッションデータを保存
    pipeline.setex(key, ttl, JSON.stringify(data));

    // ユーザー → セッション のマッピング（全セッション取得用）
    const userKey = this.userPrefix + data.userId;
    pipeline.sadd(userKey, sessionId);
    pipeline.expire(userKey, ttl);

    await pipeline.exec();
  }

  // セッション取得
  async get(sessionId: string): Promise<SessionData | null> {
    const key = this.prefix + sessionId;
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  // セッション削除
  async delete(sessionId: string): Promise<void> {
    const key = this.prefix + sessionId;
    const data = await this.get(sessionId);

    const pipeline = this.redis.pipeline();
    pipeline.del(key);

    if (data) {
      pipeline.srem(this.userPrefix + data.userId, sessionId);
    }

    await pipeline.exec();
  }

  // ユーザーの全セッション取得
  async findByUserId(userId: string): Promise<Array<{ id: string; data: SessionData }>> {
    const userKey = this.userPrefix + userId;
    const sessionIds = await this.redis.smembers(userKey);

    const sessions: Array<{ id: string; data: SessionData }> = [];
    for (const id of sessionIds) {
      const data = await this.get(id);
      if (data) {
        sessions.push({ id, data });
      } else {
        // 期限切れのセッション ID を Set から削除
        await this.redis.srem(userKey, id);
      }
    }

    return sessions;
  }

  // ユーザーの全セッション削除
  async deleteAllForUser(userId: string): Promise<void> {
    const userKey = this.userPrefix + userId;
    const sessionIds = await this.redis.smembers(userKey);

    const pipeline = this.redis.pipeline();
    for (const id of sessionIds) {
      pipeline.del(this.prefix + id);
    }
    pipeline.del(userKey);
    await pipeline.exec();
  }

  // セッション数の取得（監視用）
  async count(): Promise<number> {
    const keys = await this.redis.keys(this.prefix + '*');
    return keys.length;
  }
}
```

---

## 3. データベースセッションストア

```typescript
// Prisma を使ったセッションストア
// schema.prisma
// model Session {
//   id            String   @id @default(cuid())
//   sessionToken  String   @unique
//   userId        String
//   data          Json
//   expiresAt     DateTime
//   createdAt     DateTime @default(now())
//   updatedAt     DateTime @updatedAt
//   user          User     @relation(fields: [userId], references: [id])
//
//   @@index([userId])
//   @@index([expiresAt])
// }

class DatabaseSessionStore {
  // セッション保存
  async set(sessionId: string, data: SessionData, ttl: number): Promise<void> {
    const expiresAt = new Date(Date.now() + ttl * 1000);

    await prisma.session.upsert({
      where: { sessionToken: sessionId },
      create: {
        sessionToken: sessionId,
        userId: data.userId,
        data: data as any,
        expiresAt,
      },
      update: {
        data: data as any,
        expiresAt,
      },
    });
  }

  // セッション取得
  async get(sessionId: string): Promise<SessionData | null> {
    const session = await prisma.session.findUnique({
      where: {
        sessionToken: sessionId,
        expiresAt: { gt: new Date() }, // 有効期限チェック
      },
    });

    return session ? (session.data as SessionData) : null;
  }

  // セッション削除
  async delete(sessionId: string): Promise<void> {
    await prisma.session.delete({
      where: { sessionToken: sessionId },
    }).catch(() => {}); // 存在しない場合は無視
  }

  // 期限切れセッションのクリーンアップ（定期実行）
  async cleanup(): Promise<number> {
    const result = await prisma.session.deleteMany({
      where: { expiresAt: { lt: new Date() } },
    });
    return result.count;
  }

  // ユーザーの全セッション削除
  async deleteAllForUser(userId: string): Promise<void> {
    await prisma.session.deleteMany({ where: { userId } });
  }
}

// 定期クリーンアップ（cron ジョブ）
// 1時間ごとに期限切れセッションを削除
// cron: 0 * * * *
async function cleanupExpiredSessions() {
  const store = new DatabaseSessionStore();
  const count = await store.cleanup();
  console.log(`Cleaned up ${count} expired sessions`);
}
```

---

## 4. スケーリング戦略

```
セッションとスケーリング:

  問題: 複数サーバーでセッションを共有する必要

  ┌─────────────┐      ┌──────────────┐
  │   Server A  │──┐   │              │
  │  (session?) │  ├──→│  Redis       │
  │             │  │   │  (Session    │
  ├─────────────┤  │   │   Store)     │
  │   Server B  │──┤   │              │
  │  (session?) │  │   │              │
  │             │  │   └──────────────┘
  ├─────────────┤  │
  │   Server C  │──┘
  │  (session?) │
  └─────────────┘

  戦略:

  ① 共有セッションストア（推奨）:
     → Redis / Memcached を全サーバーで共有
     → 最もシンプルで信頼性が高い

  ② スティッキーセッション:
     → ロードバランサーが同じユーザーを同じサーバーに振り分け
     → 問題: サーバーダウン時にセッション消失
     → 推奨しない

  ③ セッションレプリケーション:
     → サーバー間でセッションを同期
     → 複雑、ネットワークオーバーヘッド
     → 大規模システム向け

  推奨:
  → Redis Sentinel（高可用性）
  → Redis Cluster（水平スケーリング）
  → Elasticache（AWS の場合）
  → Upstash Redis（サーバーレスの場合）
```

```typescript
// Redis Sentinel 構成（高可用性）
import Redis from 'ioredis';

const redis = new Redis({
  sentinels: [
    { host: 'sentinel-1', port: 26379 },
    { host: 'sentinel-2', port: 26379 },
    { host: 'sentinel-3', port: 26379 },
  ],
  name: 'mymaster',  // マスター名
  sentinelPassword: process.env.REDIS_SENTINEL_PASSWORD,
  password: process.env.REDIS_PASSWORD,
});

// Upstash Redis（サーバーレス）
import { Redis } from '@upstash/redis';

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL,
  token: process.env.UPSTASH_REDIS_REST_TOKEN,
});
```

---

## 5. セッションのセキュリティ強化

```
セッション ID の要件:

  ① 十分な長さ: 最低128ビット（推奨256ビット）
  ② 暗号的ランダム: crypto.randomBytes() を使用
  ③ 予測不可能: シーケンシャルな ID は不可
  ④ ユニーク: 衝突がないこと

追加のセキュリティ対策:

  → IP アドレス検証:
    セッション作成時の IP と比較
    ※ モバイルネットワーク等で IP 変更は頻繁に起きるため、
      警告のみで遮断しないことも多い

  → User-Agent 検証:
    ブラウザ変更の検知
    ※ 偽装が容易なため補助的な対策

  → アクティビティ監視:
    異常なアクティビティパターンの検知
    短時間での大量リクエスト等
```

---

## まとめ

| 項目 | 推奨 |
|------|------|
| ストア | Redis（本番）、メモリ（開発） |
| スケーリング | 共有 Redis（Sentinel/Cluster） |
| TTL | Redis の TTL 機能で自動削除 |
| クリーンアップ | DB の場合は定期バッチ |
| セッション ID | 256ビットの暗号ランダム値 |

---

## 次に読むべきガイド
→ [[02-csrf-protection.md]] — CSRF 防御

---

## 参考文献
1. Redis. "Redis as a Session Store." redis.io, 2024.
2. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. Express. "express-session." github.com/expressjs/session.
