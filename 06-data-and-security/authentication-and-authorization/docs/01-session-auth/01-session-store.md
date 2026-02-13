# セッションストア

> セッションデータの保存先はアプリケーションのスケーラビリティとパフォーマンスに直結する。メモリ、Redis、データベースの各セッションストアの特徴と選定基準、Redis を使ったスケーラブルなセッション管理、セッション ID の生成・検証・ローテーション、分散環境でのセッション共有戦略を網羅的に解説する。

## この章で学ぶこと

- [ ] セッションストアの種類と選定基準を理解する
- [ ] Redis セッションストアの設計と実装を完全に把握する
- [ ] データベースセッションストアのスキーマ設計と最適化を学ぶ
- [ ] スケーリング戦略とセッションの永続化・レプリケーションを理解する
- [ ] セッション ID の生成要件とセキュリティ強化手法を実践する

### 前提知識

- HTTP Cookie の基本（→ [[00-session-mechanism.md]]）
- Node.js / TypeScript の非同期処理
- Redis の基本操作（GET / SET / DEL / EXPIRE）

---

## 1. セッションストアの全体像

### 1.1 なぜセッションストアの選択が重要なのか

```
セッションストアの役割:

  ┌────────────┐      ┌────────────────┐      ┌────────────────┐
  │  ブラウザ   │      │  Web サーバー   │      │ セッションストア │
  │            │      │               │      │               │
  │  Cookie:   │─────→│  Session ID   │─────→│  Session Data  │
  │  sess_id   │      │  の検証        │      │  の取得/保存   │
  │            │      │               │      │               │
  └────────────┘      └────────────────┘      └────────────────┘

  セッションストアに求められる要件:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① 高速なRead/Write（全リクエストで実行される）     │
  │  ② スケーラビリティ（複数サーバーでの共有）          │
  │  ③ 永続性（サーバー再起動に耐える）                 │
  │  ④ TTL（有効期限の自動管理）                       │
  │  ⑤ 同時アクセスの整合性（Race Condition 防止）      │
  │  ⑥ セキュリティ（データの暗号化・アクセス制御）      │
  │                                                    │
  └────────────────────────────────────────────────────┘

  選択を間違えるとどうなるか:
    → メモリストアで本番運用 → サーバー再起動で全ユーザーログアウト
    → DB ストアでハイトラフィック → レイテンシ増大・DB 過負荷
    → ストア障害 → 全ユーザーがアクセス不能
```

### 1.2 セッションストアの比較

```
セッションストアの種類と詳細比較:

  ┌───────────┬───────┬─────────┬───────┬──────────┬───────────────┐
  │ ストア     │ 速度   │ スケール │ 永続化 │ TTL 管理  │ 用途          │
  ├───────────┼───────┼─────────┼───────┼──────────┼───────────────┤
  │ メモリ     │ ◎最速  │ ✗ 不可  │ ✗     │ 手動     │ 開発/単一サーバー│
  │ Redis     │ ○ 高速 │ ✓ 可能  │ △     │ ✓ 自動   │ 本番推奨       │
  │ PostgreSQL│ △ 中   │ ✓ 可能  │ ✓     │ 手動     │ 追加インフラ不要│
  │ MySQL     │ △ 中   │ ✓ 可能  │ ✓     │ 手動     │ MySQL環境      │
  │ MongoDB   │ △ 中   │ ✓ 可能  │ ✓     │ ✓ TTL idx│ MongoDB使用中  │
  │ DynamoDB  │ ○ 高速 │ ✓ 自動  │ ✓     │ ✓ TTL   │ AWS環境        │
  │ Memcached │ ◎ 高速 │ ✓ 可能  │ ✗     │ ✓ 自動   │ 純粋キャッシュ │
  │ Cookie    │ ◎ 最速 │ ✓ 自動  │ ✓     │ ✓ Cookie │ JWT/小データ   │
  └───────────┴───────┴─────────┴───────┴──────────┴───────────────┘

  レイテンシ比較（参考値）:
  ┌───────────┬──────────────────┬───────────────────┐
  │ ストア     │ Read（p50）       │ Read（p99）        │
  ├───────────┼──────────────────┼───────────────────┤
  │ メモリ     │ < 0.01 ms        │ < 0.1 ms          │
  │ Redis     │ 0.1 - 0.5 ms     │ 1 - 5 ms          │
  │ Memcached │ 0.1 - 0.5 ms     │ 1 - 3 ms          │
  │ DynamoDB  │ 1 - 5 ms         │ 10 - 25 ms        │
  │ PostgreSQL│ 1 - 10 ms        │ 20 - 50 ms        │
  │ MongoDB   │ 1 - 10 ms        │ 20 - 50 ms        │
  │ Cookie    │ 0 ms (ネットワーク不要)│ 0 ms          │
  └───────────┴──────────────────┴───────────────────┘

  ※ ネットワーク遅延、データサイズ、同時接続数により変動
```

### 1.3 選定フローチャート

```
セッションストア選定の意思決定:

  開始
  │
  ├─ 開発環境？ ──Yes──→ メモリストア
  │
  ├─ サーバーレス（Vercel/Lambda）？
  │   ├─ Yes → Cookie-based（JWT）or DynamoDB / Upstash Redis
  │   └─ No ↓
  │
  ├─ 既に Redis を使用中？
  │   ├─ Yes → Redis セッションストア（第一選択）
  │   └─ No ↓
  │
  ├─ 追加インフラを避けたい？
  │   ├─ Yes → データベースセッションストア
  │   └─ No ↓
  │
  ├─ 高トラフィック（1000+ req/sec）？
  │   ├─ Yes → Redis（Sentinel/Cluster）
  │   └─ No → データベースセッションストアでも可
  │
  └─ AWS 環境？ → DynamoDB + DAX（キャッシュ）

  推奨:
  → ほとんどのケースで Redis が最適解
  → Redis を追加できない場合は DB セッション
  → サーバーレスの場合は Upstash Redis or Cookie-based
```

---

## 2. メモリセッションストア

### 2.1 実装と注意点

```typescript
// メモリセッションストアの実装
// 開発環境・プロトタイプ用。本番では使用しないこと。

interface SessionData {
  userId: string;
  role: string;
  createdAt: number;
  lastAccessedAt: number;
  metadata: Record<string, unknown>;
}

class InMemorySessionStore {
  private sessions: Map<string, { data: SessionData; expiresAt: number }> = new Map();
  private userSessions: Map<string, Set<string>> = new Map();
  private cleanupInterval: ReturnType<typeof setInterval>;

  constructor(private cleanupIntervalMs: number = 60_000) {
    // 定期クリーンアップ（メモリリーク防止）
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, cleanupIntervalMs);
  }

  async set(sessionId: string, data: SessionData, ttlSeconds: number): Promise<void> {
    const expiresAt = Date.now() + ttlSeconds * 1000;
    this.sessions.set(sessionId, { data, expiresAt });

    // ユーザー → セッションのマッピング
    if (!this.userSessions.has(data.userId)) {
      this.userSessions.set(data.userId, new Set());
    }
    this.userSessions.get(data.userId)!.add(sessionId);
  }

  async get(sessionId: string): Promise<SessionData | null> {
    const entry = this.sessions.get(sessionId);
    if (!entry) return null;

    // 有効期限チェック
    if (entry.expiresAt < Date.now()) {
      this.sessions.delete(sessionId);
      return null;
    }

    return entry.data;
  }

  async delete(sessionId: string): Promise<void> {
    const entry = this.sessions.get(sessionId);
    if (entry) {
      const userSet = this.userSessions.get(entry.data.userId);
      userSet?.delete(sessionId);
      if (userSet?.size === 0) {
        this.userSessions.delete(entry.data.userId);
      }
    }
    this.sessions.delete(sessionId);
  }

  async deleteAllForUser(userId: string): Promise<void> {
    const sessionIds = this.userSessions.get(userId);
    if (sessionIds) {
      for (const id of sessionIds) {
        this.sessions.delete(id);
      }
      this.userSessions.delete(userId);
    }
  }

  async count(): Promise<number> {
    return this.sessions.size;
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [id, entry] of this.sessions) {
      if (entry.expiresAt < now) {
        this.delete(id);
      }
    }
  }

  destroy(): void {
    clearInterval(this.cleanupInterval);
    this.sessions.clear();
    this.userSessions.clear();
  }
}
```

```
メモリストアの制約:

  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ✗ サーバー再起動で全セッション消失                  │
  │  ✗ 複数サーバー間で共有不可                         │
  │  ✗ メモリ使用量が増加し続ける可能性                  │
  │  ✗ Node.js プロセスのヒープサイズに制限される        │
  │                                                    │
  │  1 セッション ≈ 1KB とした場合:                     │
  │  → 10,000 セッション ≈ 10 MB                       │
  │  → 100,000 セッション ≈ 100 MB                     │
  │  → 1,000,000 セッション ≈ 1 GB                     │
  │                                                    │
  │  Node.js デフォルトヒープ: ~1.5 GB                  │
  │  → 100万セッションでメモリ不足の危険                │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

---

## 3. Redis セッションストア

### 3.1 なぜ Redis が最適なのか

```
Redis がセッションストアに最適な理由:

  ① インメモリ + 永続化:
     → データはメモリ上で処理（高速）
     → RDB / AOF で永続化可能（再起動に耐える）

  ② TTL の自動管理:
     → SETEX / EXPIRE コマンドで自動削除
     → クリーンアップバッチが不要

  ③ 豊富なデータ構造:
     → String: セッションデータ（JSON）
     → Set: ユーザーの全セッション ID
     → Hash: セッション内のフィールド操作
     → Sorted Set: セッション一覧（最終アクセス順）

  ④ アトミック操作:
     → MULTI/EXEC: トランザクション
     → Lua スクリプト: 複合操作のアトミック実行
     → Race Condition を防止

  ⑤ スケーリング:
     → Sentinel: 高可用性（自動フェイルオーバー）
     → Cluster: 水平スケーリング（シャーディング）
     → Pub/Sub: セッション無効化の通知

  Redis のセッション関連コマンド:
  ┌──────────────┬─────────────────────────────────────┐
  │ コマンド      │ 用途                                │
  ├──────────────┼─────────────────────────────────────┤
  │ SETEX        │ セッション保存 + TTL 設定             │
  │ GET          │ セッション取得                        │
  │ DEL          │ セッション削除                        │
  │ EXPIRE       │ TTL 更新（Sliding Expiration）       │
  │ TTL          │ 残り有効期限の確認                    │
  │ SADD/SMEMBERS│ ユーザーのセッション Set 操作          │
  │ SREM         │ Set からセッション ID 削除            │
  │ PIPELINE     │ 複数コマンドの一括実行                │
  │ SCAN         │ キーの安全な列挙（KEYS の代替）       │
  └──────────────┴─────────────────────────────────────┘
```

### 3.2 Redis セッションストアの完全実装

```typescript
// Redis セッションストアの完全実装
import Redis from 'ioredis';
import { randomBytes, createHash } from 'crypto';

interface SessionData {
  userId: string;
  role: string;
  createdAt: number;
  lastAccessedAt: number;
  ip: string;
  userAgent: string;
  metadata: Record<string, unknown>;
}

interface SessionEntry {
  id: string;
  data: SessionData;
  ttl: number;  // 残り秒数
}

class RedisSessionStore {
  private redis: Redis;
  private readonly prefix: string;
  private readonly userPrefix: string;
  private readonly defaultTtl: number;

  constructor(options: {
    redisUrl: string;
    prefix?: string;
    defaultTtl?: number;
  }) {
    this.redis = new Redis(options.redisUrl, {
      // 接続の堅牢性
      retryStrategy: (times) => {
        if (times > 10) return null; // 10回超で諦める
        return Math.min(times * 100, 3000); // 最大3秒待機
      },
      maxRetriesPerRequest: 3,
      enableReadyCheck: true,
      // パフォーマンス
      lazyConnect: true,
      keepAlive: 10000,
      connectTimeout: 5000,
    });

    this.prefix = options.prefix ?? 'sess:';
    this.userPrefix = `${this.prefix}user:`;
    this.defaultTtl = options.defaultTtl ?? 86400; // 24時間

    // エラーハンドリング
    this.redis.on('error', (err) => {
      console.error('[SessionStore] Redis error:', err);
    });

    this.redis.on('connect', () => {
      console.log('[SessionStore] Redis connected');
    });
  }

  // セッション保存
  async set(
    sessionId: string,
    data: SessionData,
    ttl: number = this.defaultTtl
  ): Promise<void> {
    const key = this.prefix + sessionId;
    const userKey = this.userPrefix + data.userId;
    const serialized = JSON.stringify(data);

    // Pipeline で複数操作をアトミックに実行
    const pipeline = this.redis.pipeline();

    // セッションデータを保存（TTL 付き）
    pipeline.setex(key, ttl, serialized);

    // ユーザー → セッション ID のマッピング（Set）
    pipeline.sadd(userKey, sessionId);
    pipeline.expire(userKey, ttl);

    const results = await pipeline.exec();

    // エラーチェック
    if (results) {
      for (const [err] of results) {
        if (err) throw err;
      }
    }
  }

  // セッション取得
  async get(sessionId: string): Promise<SessionData | null> {
    const key = this.prefix + sessionId;
    const data = await this.redis.get(key);

    if (!data) return null;

    try {
      return JSON.parse(data) as SessionData;
    } catch {
      // 不正なデータの場合は削除
      await this.redis.del(key);
      return null;
    }
  }

  // セッション取得 + TTL 延長（Sliding Expiration）
  async getAndRefresh(
    sessionId: string,
    ttl: number = this.defaultTtl
  ): Promise<SessionData | null> {
    const key = this.prefix + sessionId;

    // Lua スクリプトでアトミックに取得 + TTL 更新
    const luaScript = `
      local data = redis.call('GET', KEYS[1])
      if data then
        redis.call('EXPIRE', KEYS[1], ARGV[1])
        return data
      end
      return nil
    `;

    const data = await this.redis.eval(
      luaScript,
      1,
      key,
      ttl
    ) as string | null;

    if (!data) return null;

    try {
      const session = JSON.parse(data) as SessionData;
      // lastAccessedAt を更新
      session.lastAccessedAt = Date.now();
      await this.redis.setex(key, ttl, JSON.stringify(session));
      return session;
    } catch {
      return null;
    }
  }

  // セッション削除（ログアウト）
  async delete(sessionId: string): Promise<void> {
    const key = this.prefix + sessionId;
    const data = await this.get(sessionId);

    const pipeline = this.redis.pipeline();
    pipeline.del(key);

    if (data) {
      // ユーザーの Session Set からも削除
      pipeline.srem(this.userPrefix + data.userId, sessionId);
    }

    await pipeline.exec();
  }

  // ユーザーの全セッション取得（アクティブセッション一覧）
  async findByUserId(userId: string): Promise<SessionEntry[]> {
    const userKey = this.userPrefix + userId;
    const sessionIds = await this.redis.smembers(userKey);

    if (sessionIds.length === 0) return [];

    // Pipeline で一括取得（N+1 問題を防止）
    const pipeline = this.redis.pipeline();
    for (const id of sessionIds) {
      pipeline.get(this.prefix + id);
      pipeline.ttl(this.prefix + id);
    }

    const results = await pipeline.exec();
    if (!results) return [];

    const sessions: SessionEntry[] = [];
    const expiredIds: string[] = [];

    for (let i = 0; i < sessionIds.length; i++) {
      const [getErr, data] = results[i * 2];
      const [ttlErr, ttl] = results[i * 2 + 1];

      if (getErr || ttlErr || !data || (ttl as number) <= 0) {
        // 期限切れまたはエラー
        expiredIds.push(sessionIds[i]);
        continue;
      }

      try {
        sessions.push({
          id: sessionIds[i],
          data: JSON.parse(data as string),
          ttl: ttl as number,
        });
      } catch {
        expiredIds.push(sessionIds[i]);
      }
    }

    // 期限切れセッション ID を Set から削除
    if (expiredIds.length > 0) {
      const cleanPipeline = this.redis.pipeline();
      for (const id of expiredIds) {
        cleanPipeline.srem(userKey, id);
      }
      await cleanPipeline.exec();
    }

    return sessions;
  }

  // ユーザーの全セッション削除（パスワード変更時、全デバイスログアウト）
  async deleteAllForUser(userId: string): Promise<number> {
    const userKey = this.userPrefix + userId;
    const sessionIds = await this.redis.smembers(userKey);

    if (sessionIds.length === 0) return 0;

    const pipeline = this.redis.pipeline();
    for (const id of sessionIds) {
      pipeline.del(this.prefix + id);
    }
    pipeline.del(userKey);

    await pipeline.exec();
    return sessionIds.length;
  }

  // 特定セッション以外を全削除（他のデバイスをログアウト）
  async deleteOthersForUser(
    userId: string,
    currentSessionId: string
  ): Promise<number> {
    const userKey = this.userPrefix + userId;
    const sessionIds = await this.redis.smembers(userKey);

    const toDelete = sessionIds.filter((id) => id !== currentSessionId);
    if (toDelete.length === 0) return 0;

    const pipeline = this.redis.pipeline();
    for (const id of toDelete) {
      pipeline.del(this.prefix + id);
      pipeline.srem(userKey, id);
    }

    await pipeline.exec();
    return toDelete.length;
  }

  // セッション数の取得（SCAN ベース、本番安全）
  async count(): Promise<number> {
    let count = 0;
    let cursor = '0';

    do {
      const [nextCursor, keys] = await this.redis.scan(
        cursor,
        'MATCH',
        `${this.prefix}*`,
        'COUNT',
        100
      );
      cursor = nextCursor;
      // user: プレフィックスを除外
      count += keys.filter((k) => !k.startsWith(this.userPrefix)).length;
    } while (cursor !== '0');

    return count;
  }

  // 接続の正常性確認
  async healthCheck(): Promise<boolean> {
    try {
      const result = await this.redis.ping();
      return result === 'PONG';
    } catch {
      return false;
    }
  }

  // クリーンシャットダウン
  async close(): Promise<void> {
    await this.redis.quit();
  }
}
```

### 3.3 Redis 接続オプションの詳細

```typescript
// ioredis 接続設定の詳細解説
import Redis, { RedisOptions } from 'ioredis';

// 開発環境
const devOptions: RedisOptions = {
  host: 'localhost',
  port: 6379,
  db: 0, // セッション用 DB
};

// 本番環境（シングルノード）
const prodOptions: RedisOptions = {
  host: process.env.REDIS_HOST!,
  port: Number(process.env.REDIS_PORT) || 6379,
  password: process.env.REDIS_PASSWORD,

  // TLS 設定（Redis 6+）
  tls: process.env.REDIS_TLS === 'true' ? {} : undefined,

  // 接続プール
  lazyConnect: true,     // 明示的に connect() するまで待機
  keepAlive: 10000,      // TCP KeepAlive（ミリ秒）
  connectTimeout: 5000,  // 接続タイムアウト

  // コマンドリトライ
  maxRetriesPerRequest: 3,
  retryStrategy: (times) => {
    if (times > 20) return null; // 諦める
    return Math.min(times * 200, 5000);
  },

  // 再接続
  reconnectOnError: (err) => {
    const targetErrors = ['READONLY', 'ECONNREFUSED'];
    return targetErrors.some((e) => err.message.includes(e));
  },
};

// Upstash Redis（サーバーレス向け）
import { Redis as UpstashRedis } from '@upstash/redis';

const upstashRedis = new UpstashRedis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});
// 注意: Upstash は HTTP ベースなので ioredis の Pipeline/Lua は使えない
// → @upstash/redis 固有の API を使用

// Elasticache（AWS）
const elasticacheOptions: RedisOptions = {
  host: process.env.ELASTICACHE_ENDPOINT!,
  port: 6379,
  tls: {},
  // Elasticache はパスワード不要の場合あり（VPC 内アクセス）
};
```

### 3.4 Sliding Expiration と Absolute Expiration

```
セッション有効期限の 2 つの方式:

  ① Sliding Expiration（スライディング有効期限）:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  アクセスするたびに有効期限がリセットされる           │
  │                                                    │
  │  時間 ──────────────────────────────────────→       │
  │  ├──────┤ アクセス                                  │
  │         ├──────┤ アクセス                            │
  │                ├──────┤ アクセス                     │
  │                       ├──────┤ 期限切れ              │
  │                                                    │
  │  利点: アクティブユーザーはログアウトされない         │
  │  欠点: 永遠にセッションが延長される可能性            │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ② Absolute Expiration（絶対有効期限）:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  作成時刻から固定時間で期限切れ                      │
  │                                                    │
  │  時間 ──────────────────────────────────────→       │
  │  ├─────────────────────────────┤ 期限切れ（固定）   │
  │  ├──┤ アクセス                                     │
  │      ├──┤ アクセス（延長されない）                   │
  │          ├──┤ アクセス                              │
  │                                                    │
  │  利点: セッション乗っ取りの影響を時間的に制限        │
  │  欠点: アクティブでも強制ログアウト                  │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ③ 推奨: ハイブリッド方式
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  Sliding: 30 分（非アクティブでログアウト）          │
  │  Absolute: 24 時間（最大セッション寿命）             │
  │  → 両方の制限のうち早い方で期限切れ                 │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

```typescript
// ハイブリッド有効期限の実装
class HybridExpirationSessionStore extends RedisSessionStore {
  private readonly slidingTtl: number;   // 非アクティブ期限（秒）
  private readonly absoluteTtl: number;  // 最大セッション寿命（秒）

  constructor(options: {
    redisUrl: string;
    slidingTtl?: number;
    absoluteTtl?: number;
  }) {
    super({ redisUrl: options.redisUrl });
    this.slidingTtl = options.slidingTtl ?? 1800;    // 30分
    this.absoluteTtl = options.absoluteTtl ?? 86400;  // 24時間
  }

  async getWithHybridExpiration(sessionId: string): Promise<SessionData | null> {
    const data = await this.get(sessionId);
    if (!data) return null;

    // Absolute Expiration チェック
    const sessionAge = Date.now() - data.createdAt;
    if (sessionAge > this.absoluteTtl * 1000) {
      await this.delete(sessionId);
      return null;
    }

    // Sliding Expiration: 残り時間を計算
    const remainingAbsolute = Math.ceil(
      (this.absoluteTtl * 1000 - sessionAge) / 1000
    );
    const ttl = Math.min(this.slidingTtl, remainingAbsolute);

    // TTL を更新
    await this.set(sessionId, {
      ...data,
      lastAccessedAt: Date.now(),
    }, ttl);

    return data;
  }
}
```

---

## 4. データベースセッションストア

### 4.1 Prisma スキーマ設計

```prisma
// schema.prisma - セッションテーブルの最適設計

model Session {
  id            String   @id @default(cuid())
  sessionToken  String   @unique @map("session_token")
  userId        String   @map("user_id")
  data          Json?    // セッション追加データ
  expiresAt     DateTime @map("expires_at")
  createdAt     DateTime @default(now()) @map("created_at")
  updatedAt     DateTime @updatedAt @map("updated_at")

  // セッションのメタデータ
  ipAddress     String?  @map("ip_address")
  userAgent     String?  @map("user_agent")
  deviceName    String?  @map("device_name")
  lastAccessedAt DateTime? @map("last_accessed_at")

  // リレーション
  user          User     @relation(fields: [userId], references: [id], onDelete: Cascade)

  // インデックス
  @@index([userId])                    // ユーザーのセッション検索
  @@index([expiresAt])                 // 期限切れクリーンアップ
  @@index([userId, expiresAt])         // ユーザーのアクティブセッション
  @@map("sessions")
}

// インデックスの重要性:
// ① userId: ユーザーの全セッション取得（O(1) → O(n)を防止）
// ② expiresAt: クリーンアップバッチの WHERE 句を高速化
// ③ 複合インデックス: 特定ユーザーの有効セッション検索
```

### 4.2 データベースセッションストアの完全実装

```typescript
// Prisma を使ったデータベースセッションストアの完全実装

import { PrismaClient, Prisma } from '@prisma/client';

const prisma = new PrismaClient({
  log: process.env.NODE_ENV === 'development'
    ? ['query', 'warn', 'error']
    : ['error'],
});

class DatabaseSessionStore {
  // セッション保存（Upsert: 存在すれば更新、なければ作成）
  async set(
    sessionId: string,
    data: SessionData,
    ttl: number
  ): Promise<void> {
    const expiresAt = new Date(Date.now() + ttl * 1000);

    await prisma.session.upsert({
      where: { sessionToken: sessionId },
      create: {
        sessionToken: sessionId,
        userId: data.userId,
        data: data as unknown as Prisma.JsonObject,
        expiresAt,
        ipAddress: data.ip,
        userAgent: data.userAgent,
        lastAccessedAt: new Date(),
      },
      update: {
        data: data as unknown as Prisma.JsonObject,
        expiresAt,
        lastAccessedAt: new Date(),
      },
    });
  }

  // セッション取得（有効期限チェック付き）
  async get(sessionId: string): Promise<SessionData | null> {
    const session = await prisma.session.findFirst({
      where: {
        sessionToken: sessionId,
        expiresAt: { gt: new Date() },  // 有効期限内のみ
      },
    });

    if (!session) return null;

    return session.data as unknown as SessionData;
  }

  // セッション取得 + Sliding Expiration
  async getAndRefresh(
    sessionId: string,
    ttl: number
  ): Promise<SessionData | null> {
    // トランザクションで取得と更新をアトミックに
    const session = await prisma.$transaction(async (tx) => {
      const found = await tx.session.findFirst({
        where: {
          sessionToken: sessionId,
          expiresAt: { gt: new Date() },
        },
      });

      if (!found) return null;

      // 有効期限を延長
      await tx.session.update({
        where: { id: found.id },
        data: {
          expiresAt: new Date(Date.now() + ttl * 1000),
          lastAccessedAt: new Date(),
        },
      });

      return found;
    });

    if (!session) return null;
    return session.data as unknown as SessionData;
  }

  // セッション削除
  async delete(sessionId: string): Promise<void> {
    await prisma.session.deleteMany({
      where: { sessionToken: sessionId },
    });
  }

  // ユーザーの全セッション取得
  async findByUserId(userId: string): Promise<Array<{
    id: string;
    sessionToken: string;
    data: SessionData;
    ipAddress: string | null;
    userAgent: string | null;
    createdAt: Date;
    lastAccessedAt: Date | null;
  }>> {
    const sessions = await prisma.session.findMany({
      where: {
        userId,
        expiresAt: { gt: new Date() },
      },
      orderBy: { lastAccessedAt: 'desc' },
    });

    return sessions.map((s) => ({
      id: s.id,
      sessionToken: s.sessionToken,
      data: s.data as unknown as SessionData,
      ipAddress: s.ipAddress,
      userAgent: s.userAgent,
      createdAt: s.createdAt,
      lastAccessedAt: s.lastAccessedAt,
    }));
  }

  // ユーザーの全セッション削除
  async deleteAllForUser(userId: string): Promise<number> {
    const result = await prisma.session.deleteMany({
      where: { userId },
    });
    return result.count;
  }

  // 期限切れセッションのクリーンアップ（バッチ削除）
  async cleanup(batchSize: number = 1000): Promise<number> {
    let totalDeleted = 0;

    // 大量削除はバッチで行う（テーブルロック防止）
    while (true) {
      const result = await prisma.$executeRaw`
        DELETE FROM sessions
        WHERE expires_at < NOW()
        LIMIT ${batchSize}
      `;

      totalDeleted += result;

      // 削除対象がバッチサイズ未満なら完了
      if (result < batchSize) break;

      // 短い待機を入れて他のクエリに CPU を譲る
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    return totalDeleted;
  }

  // アクティブセッション数
  async count(): Promise<number> {
    return prisma.session.count({
      where: { expiresAt: { gt: new Date() } },
    });
  }

  // ユーザーのアクティブセッション数
  async countByUser(userId: string): Promise<number> {
    return prisma.session.count({
      where: {
        userId,
        expiresAt: { gt: new Date() },
      },
    });
  }
}
```

### 4.3 定期クリーンアップの実装

```typescript
// cron ジョブによる期限切れセッションのクリーンアップ

// ① Node.js の cron ライブラリを使用する場合
import { CronJob } from 'cron';

const store = new DatabaseSessionStore();

// 毎時 0 分に実行
const cleanupJob = new CronJob('0 * * * *', async () => {
  const startTime = Date.now();

  try {
    const deletedCount = await store.cleanup();
    const elapsed = Date.now() - startTime;

    console.log(
      `[Session Cleanup] Deleted ${deletedCount} expired sessions in ${elapsed}ms`
    );

    // メトリクスの記録（Prometheus 等）
    sessionCleanupCounter.inc(deletedCount);
    sessionCleanupDuration.observe(elapsed / 1000);
  } catch (error) {
    console.error('[Session Cleanup] Failed:', error);
  }
});

cleanupJob.start();

// ② PostgreSQL のネイティブ機能を使用する場合
// pg_cron 拡張を使用（Supabase、RDS 等で利用可能）
// SELECT cron.schedule(
//   'cleanup_expired_sessions',
//   '0 * * * *',
//   $$DELETE FROM sessions WHERE expires_at < NOW()$$
// );

// ③ Vercel Cron Jobs を使用する場合
// vercel.json:
// {
//   "crons": [{
//     "path": "/api/cron/cleanup-sessions",
//     "schedule": "0 * * * *"
//   }]
// }

// app/api/cron/cleanup-sessions/route.ts
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  // Vercel Cron の認証ヘッダー確認
  const authHeader = request.headers.get('Authorization');
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return new NextResponse('Unauthorized', { status: 401 });
  }

  const store = new DatabaseSessionStore();
  const count = await store.cleanup();

  return NextResponse.json({ deletedCount: count });
}
```

---

## 5. スケーリング戦略

### 5.1 Redis を使ったスケーリング

```
セッションとスケーリングの全体図:

  ┌─────────────────────────────────────────────────────┐
  │                Load Balancer                        │
  │          (ラウンドロビン / Least Connections)         │
  └───────────┬───────────┬───────────┬────────────────┘
              │           │           │
  ┌───────────┴┐ ┌────────┴──┐ ┌─────┴──────┐
  │ Server A   │ │ Server B  │ │ Server C   │
  │ (stateless)│ │ (stateless)│ │ (stateless)│
  └──────┬─────┘ └─────┬─────┘ └─────┬──────┘
         │             │             │
         └──────────┬──┴─────────────┘
                    │
         ┌──────────┴──────────┐
         │                     │
    ┌────┴────┐          ┌────┴────┐
    │ Redis   │          │ Redis   │
    │ Primary │──────────│ Replica │
    │ (Write) │  複製    │ (Read)  │
    └────┬────┘          └─────────┘
         │
    ┌────┴────┐
    │ Redis   │
    │ Sentinel│  監視 + 自動フェイルオーバー
    └─────────┘

  スケーリング戦略の比較:

  ┌──────────────────┬──────────────┬──────────────┬──────────────┐
  │ 方式              │ 可用性       │ 複雑度       │ 推奨規模     │
  ├──────────────────┼──────────────┼──────────────┼──────────────┤
  │ 単一 Redis        │ 低           │ 低           │ 小規模       │
  │ Redis Sentinel   │ 高           │ 中           │ 中〜大規模   │
  │ Redis Cluster    │ 高           │ 高           │ 大規模       │
  │ Upstash Redis    │ 高           │ 低           │ サーバーレス │
  │ Elasticache      │ 高           │ 低（AWS）    │ AWS 環境     │
  └──────────────────┴──────────────┴──────────────┴──────────────┘
```

### 5.2 Redis Sentinel 構成（高可用性）

```typescript
// Redis Sentinel 構成
import Redis from 'ioredis';

const redis = new Redis({
  sentinels: [
    { host: 'sentinel-1.internal', port: 26379 },
    { host: 'sentinel-2.internal', port: 26379 },
    { host: 'sentinel-3.internal', port: 26379 },
  ],
  name: 'mymaster',  // マスター名
  sentinelPassword: process.env.REDIS_SENTINEL_PASSWORD,
  password: process.env.REDIS_PASSWORD,

  // フェイルオーバー時の挙動
  sentinelRetryStrategy: (times) => {
    return Math.min(times * 100, 3000);
  },

  // Read Replica からの読み取り
  role: 'master',  // 書き込みはマスターのみ
  // preferredSlaves を設定すると読み取りをレプリカに分散可能
});

// フェイルオーバーイベントの監視
redis.on('reconnecting', () => {
  console.log('[Redis] Reconnecting...');
});

redis.on('+failover-end', () => {
  console.log('[Redis] Failover completed');
});
```

### 5.3 Redis Cluster 構成（水平スケーリング）

```typescript
// Redis Cluster 構成
import Redis from 'ioredis';

const cluster = new Redis.Cluster(
  [
    { host: 'cluster-1.internal', port: 6379 },
    { host: 'cluster-2.internal', port: 6379 },
    { host: 'cluster-3.internal', port: 6379 },
  ],
  {
    // クラスター固有オプション
    clusterRetryStrategy: (times) => {
      return Math.min(times * 100, 3000);
    },
    redisOptions: {
      password: process.env.REDIS_PASSWORD,
      tls: {},
    },
    // 読み取りの分散
    scaleReads: 'slave', // レプリカから読み取り
    // scaleReads: 'all'   // 全ノードから読み取り
  }
);

// Cluster 使用時の注意点:
// ① KEYS コマンドは使用不可 → SCAN を使用
// ② Pipeline のキーは同じスロットに属する必要がある
//    → {user:123}:session のようにハッシュタグを使用
// ③ Lua スクリプトのキーも同一スロット制約あり
// ④ MULTI/EXEC も同一スロット制約あり

// ハッシュタグを使ったキー設計（同一スロットに配置）
class ClusterAwareSessionStore {
  private prefix(userId: string): string {
    // {userId} でハッシュタグを使い、同一ユーザーのキーを同じスロットに
    return `{sess:${userId}}:`;
  }

  async set(sessionId: string, data: SessionData, ttl: number): Promise<void> {
    const key = `${this.prefix(data.userId)}${sessionId}`;
    const userKey = `${this.prefix(data.userId)}sessions`;

    // 同一ハッシュタグなので Pipeline 使用可能
    const pipeline = cluster.pipeline();
    pipeline.setex(key, ttl, JSON.stringify(data));
    pipeline.sadd(userKey, sessionId);
    pipeline.expire(userKey, ttl);
    await pipeline.exec();
  }
}
```

### 5.4 Upstash Redis（サーバーレス向け）

```typescript
// Upstash Redis を使ったセッションストア
// サーバーレス環境（Vercel、Cloudflare Workers）に最適

import { Redis } from '@upstash/redis';

const redis = new Redis({
  url: process.env.UPSTASH_REDIS_REST_URL!,
  token: process.env.UPSTASH_REDIS_REST_TOKEN!,
});

class UpstashSessionStore {
  private prefix = 'sess:';

  async set(sessionId: string, data: SessionData, ttl: number): Promise<void> {
    const key = this.prefix + sessionId;

    // Upstash は HTTP ベースなので Pipeline の書き方が異なる
    const pipeline = redis.pipeline();
    pipeline.setex(key, ttl, JSON.stringify(data));
    pipeline.sadd(`user:${data.userId}:sessions`, sessionId);
    pipeline.expire(`user:${data.userId}:sessions`, ttl);

    await pipeline.exec();
  }

  async get(sessionId: string): Promise<SessionData | null> {
    const key = this.prefix + sessionId;
    const data = await redis.get<string>(key);
    if (!data) return null;

    return typeof data === 'string' ? JSON.parse(data) : data;
  }

  async delete(sessionId: string): Promise<void> {
    const data = await this.get(sessionId);
    const key = this.prefix + sessionId;

    const pipeline = redis.pipeline();
    pipeline.del(key);
    if (data) {
      pipeline.srem(`user:${data.userId}:sessions`, sessionId);
    }
    await pipeline.exec();
  }
}

// 注意: Upstash の制約
// → 各コマンドが HTTP リクエスト
// → Pipeline で複数コマンドをバッチ化可能
// → Lua スクリプトも利用可能（制限あり）
// → Free tier: 10,000 コマンド/日
// → Pro: 従量課金（$0.2 / 100,000 コマンド）
```

---

## 6. セッション ID の生成とセキュリティ

### 6.1 セッション ID の要件

```
セッション ID のセキュリティ要件（OWASP 準拠）:

  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① 十分な長さ:                                     │
  │     → 最低 128 ビット（推奨 256 ビット）             │
  │     → Base64 エンコードで 32〜43 文字                │
  │                                                    │
  │  ② 暗号学的にランダム:                              │
  │     → crypto.randomBytes() を使用                  │
  │     → Math.random() は不可（予測可能）              │
  │     → UUID v4 は十分（122 ビットのランダム性）       │
  │                                                    │
  │  ③ 予測不可能:                                      │
  │     → シーケンシャルな ID は不可                    │
  │     → タイムスタンプベースは不可                    │
  │     → 前のセッション ID から次を推測不可            │
  │                                                    │
  │  ④ 衝突耐性:                                       │
  │     → 256 ビットで衝突確率は天文学的に低い          │
  │     → 2^128 個のセッションで 50% の衝突確率         │
  │                                                    │
  └────────────────────────────────────────────────────┘

  ブルートフォース試行に必要な時間（参考）:
  ┌──────────┬──────────────────┬────────────────────┐
  │ ビット数  │ パターン数        │ 100万回/秒での試行  │
  ├──────────┼──────────────────┼────────────────────┤
  │ 64 ビット │ 1.8 × 10^19     │ ≈ 585,000 年       │
  │ 128 ビット│ 3.4 × 10^38     │ ≈ 10^25 年         │
  │ 256 ビット│ 1.2 × 10^77     │ ≈ 10^64 年         │
  └──────────┴──────────────────┴────────────────────┘
```

### 6.2 セッション ID 生成の実装

```typescript
// セッション ID の安全な生成
import { randomBytes, createHash, createHmac } from 'crypto';

// 方法 1: ランダムバイト（最もシンプル）
function generateSessionId(): string {
  return randomBytes(32).toString('base64url');
  // 結果例: "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2"
  // 256 ビットのランダム性
}

// 方法 2: HMAC 付き（改ざん検知）
function generateSignedSessionId(secret: string): string {
  const id = randomBytes(32).toString('hex');
  const signature = createHmac('sha256', secret)
    .update(id)
    .digest('base64url');

  return `${id}.${signature}`;
}

function verifySignedSessionId(
  signedId: string,
  secret: string
): string | null {
  const [id, signature] = signedId.split('.');
  if (!id || !signature) return null;

  const expectedSignature = createHmac('sha256', secret)
    .update(id)
    .digest('base64url');

  // タイミング攻撃防止のため、定数時間比較を使用
  const expected = Buffer.from(expectedSignature);
  const actual = Buffer.from(signature);

  if (expected.length !== actual.length) return null;

  // crypto.timingSafeEqual は同じ長さの Buffer を要求
  if (!require('crypto').timingSafeEqual(expected, actual)) {
    return null;
  }

  return id;
}

// 方法 3: cuid2（衝突耐性 + ソート可能）
import { createId } from '@paralleldrive/cuid2';

function generateCuid2SessionId(): string {
  // cuid2 は暗号学的にランダムだが、ソート可能
  return createId();
  // 結果例: "clh3am8w000003b5y0h8xk8q9"
}
```

### 6.3 セッション固定化攻撃の防止

```
セッション固定化攻撃（Session Fixation）:

  攻撃手順:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① 攻撃者がサーバーから有効なセッション ID を取得    │
  │  ② 攻撃者がこの ID を被害者のブラウザにセット       │
  │     （URL パラメータ、Cookie 操作等）               │
  │  ③ 被害者がこの ID でログイン                      │
  │  ④ 攻撃者が同じ ID でアクセス → 認証済みセッション  │
  │                                                    │
  └────────────────────────────────────────────────────┘

  防止策: ログイン成功時にセッション ID を再生成

  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ログイン前: sessionId = "abc123"                   │
  │  ↓ 認証成功                                        │
  │  ログイン後: sessionId = "xyz789"（新しい ID）      │
  │                                                    │
  │  → 攻撃者が知っている "abc123" は無効になる         │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

```typescript
// セッション ID ローテーションの実装
async function rotateSession(
  store: RedisSessionStore,
  oldSessionId: string,
  response: Response
): Promise<string> {
  // 古いセッションデータを取得
  const data = await store.get(oldSessionId);
  if (!data) throw new Error('Session not found');

  // 新しいセッション ID を生成
  const newSessionId = generateSessionId();

  // 新しい ID でセッションを保存
  await store.set(newSessionId, data, 86400);

  // 古いセッションを削除
  await store.delete(oldSessionId);

  // Cookie を更新
  response.headers.set('Set-Cookie', [
    `session_id=${newSessionId}`,
    'Path=/',
    'HttpOnly',
    'Secure',
    'SameSite=Lax',
    `Max-Age=86400`,
  ].join('; '));

  return newSessionId;
}

// ログイン処理内で使用
async function handleLogin(
  credentials: { email: string; password: string },
  request: Request,
  response: Response
) {
  const user = await authenticateUser(credentials);
  if (!user) throw new Error('Invalid credentials');

  // 既存セッションがある場合はローテーション
  const existingSessionId = getSessionIdFromCookie(request);
  if (existingSessionId) {
    await store.delete(existingSessionId);
  }

  // 新しいセッションを作成
  const newSessionId = generateSessionId();
  await store.set(newSessionId, {
    userId: user.id,
    role: user.role,
    createdAt: Date.now(),
    lastAccessedAt: Date.now(),
    ip: getClientIp(request),
    userAgent: request.headers.get('user-agent') || '',
    metadata: {},
  }, 86400);

  // Cookie にセット
  response.headers.set('Set-Cookie', [
    `session_id=${newSessionId}`,
    'Path=/',
    'HttpOnly',
    'Secure',
    'SameSite=Lax',
    'Max-Age=86400',
  ].join('; '));
}
```

---

## 7. セッションの監視と運用

### 7.1 アクティブセッション管理 UI

```typescript
// app/settings/sessions/page.tsx - セッション管理画面
import { auth } from '@/auth';
import { redirect } from 'next/navigation';

export default async function SessionsPage() {
  const session = await auth();
  if (!session) redirect('/login');

  const store = new RedisSessionStore({
    redisUrl: process.env.REDIS_URL!,
  });

  const sessions = await store.findByUserId(session.user.id);
  const currentSessionId = getCurrentSessionId();

  return (
    <div>
      <h1>アクティブセッション</h1>
      <p className="text-gray-500">
        お使いのアカウントでログインしているデバイスの一覧です。
      </p>

      <div className="space-y-4 mt-6">
        {sessions.map((s) => {
          const isCurrent = s.id === currentSessionId;
          const ua = parseUserAgent(s.data.userAgent);

          return (
            <div key={s.id} className="border rounded-lg p-4">
              <div className="flex justify-between items-center">
                <div>
                  <p className="font-medium">
                    {ua.browser} on {ua.os}
                    {isCurrent && (
                      <span className="ml-2 text-green-600 text-sm">
                        (このデバイス)
                      </span>
                    )}
                  </p>
                  <p className="text-sm text-gray-500">
                    IP: {s.data.ip} ・
                    最終アクセス: {formatRelativeTime(s.data.lastAccessedAt)}
                  </p>
                  <p className="text-xs text-gray-400">
                    セッション開始: {formatDate(s.data.createdAt)}
                  </p>
                </div>
                {!isCurrent && (
                  <RevokeSessionButton sessionId={s.id} />
                )}
              </div>
            </div>
          );
        })}
      </div>

      {sessions.length > 1 && (
        <div className="mt-6">
          <RevokeOtherSessionsButton />
        </div>
      )}
    </div>
  );
}
```

### 7.2 セッションメトリクスの収集

```typescript
// Prometheus 互換のメトリクス収集
import { Registry, Counter, Gauge, Histogram } from 'prom-client';

const registry = new Registry();

// セッション数（ゲージ）
const activeSessionsGauge = new Gauge({
  name: 'session_active_total',
  help: 'Total number of active sessions',
  registers: [registry],
});

// セッション操作のカウンター
const sessionOperationCounter = new Counter({
  name: 'session_operations_total',
  help: 'Total number of session operations',
  labelNames: ['operation', 'status'],
  registers: [registry],
});

// セッション操作のレイテンシ
const sessionOperationDuration = new Histogram({
  name: 'session_operation_duration_seconds',
  help: 'Session operation duration in seconds',
  labelNames: ['operation'],
  buckets: [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
  registers: [registry],
});

// インストルメンテーション付きストア
class InstrumentedSessionStore {
  constructor(private inner: RedisSessionStore) {}

  async get(sessionId: string): Promise<SessionData | null> {
    const timer = sessionOperationDuration.startTimer({ operation: 'get' });

    try {
      const result = await this.inner.get(sessionId);
      sessionOperationCounter.inc({
        operation: 'get',
        status: result ? 'hit' : 'miss',
      });
      return result;
    } catch (error) {
      sessionOperationCounter.inc({ operation: 'get', status: 'error' });
      throw error;
    } finally {
      timer();
    }
  }

  async set(sessionId: string, data: SessionData, ttl: number): Promise<void> {
    const timer = sessionOperationDuration.startTimer({ operation: 'set' });

    try {
      await this.inner.set(sessionId, data, ttl);
      sessionOperationCounter.inc({ operation: 'set', status: 'success' });
    } catch (error) {
      sessionOperationCounter.inc({ operation: 'set', status: 'error' });
      throw error;
    } finally {
      timer();
    }
  }

  // 定期的にアクティブセッション数を更新
  async updateMetrics(): Promise<void> {
    const count = await this.inner.count();
    activeSessionsGauge.set(count);
  }
}
```

---

## 8. セッションのセキュリティ強化

### 8.1 追加の検証レイヤー

```typescript
// セッション検証ミドルウェア
async function validateSession(
  sessionId: string,
  request: Request,
  store: RedisSessionStore
): Promise<{
  valid: boolean;
  data?: SessionData;
  warning?: string;
}> {
  const data = await store.get(sessionId);
  if (!data) return { valid: false };

  const warnings: string[] = [];

  // ① IP アドレス変更の検知
  const currentIp = getClientIp(request);
  if (data.ip !== currentIp) {
    warnings.push(`IP changed: ${data.ip} → ${currentIp}`);
    // モバイルネットワークでは頻繁に変わるため、
    // 警告のみで遮断しない
  }

  // ② User-Agent 変更の検知
  const currentUa = request.headers.get('user-agent') || '';
  if (data.userAgent !== currentUa) {
    warnings.push('User-Agent changed');
    // User-Agent の変更はセッション乗っ取りの可能性
    // ただし UA は偽装が容易なので補助的な指標
  }

  // ③ Absolute Expiration チェック
  const maxSessionAge = 24 * 60 * 60 * 1000; // 24 時間
  if (Date.now() - data.createdAt > maxSessionAge) {
    await store.delete(sessionId);
    return { valid: false };
  }

  // ④ 同時セッション数の制限
  const maxSessionsPerUser = 5;
  const userSessions = await store.findByUserId(data.userId);
  if (userSessions.length > maxSessionsPerUser) {
    // 最も古いセッションを削除
    const sorted = userSessions.sort(
      (a, b) => a.data.lastAccessedAt - b.data.lastAccessedAt
    );
    const toDelete = sorted.slice(0, sorted.length - maxSessionsPerUser);
    for (const s of toDelete) {
      await store.delete(s.id);
    }
  }

  return {
    valid: true,
    data,
    warning: warnings.length > 0 ? warnings.join('; ') : undefined,
  };
}
```

### 8.2 セッションデータの暗号化

```typescript
// セッションデータの暗号化（Redis に保存するデータを暗号化）
import { createCipheriv, createDecipheriv, randomBytes, scryptSync } from 'crypto';

const ENCRYPTION_KEY = scryptSync(
  process.env.SESSION_ENCRYPTION_SECRET!,
  'session-store-salt',
  32
);

function encryptSessionData(data: SessionData): string {
  const iv = randomBytes(16);
  const cipher = createCipheriv('aes-256-gcm', ENCRYPTION_KEY, iv);

  const json = JSON.stringify(data);
  let encrypted = cipher.update(json, 'utf8', 'hex');
  encrypted += cipher.final('hex');

  const authTag = cipher.getAuthTag().toString('hex');

  return `${iv.toString('hex')}:${authTag}:${encrypted}`;
}

function decryptSessionData(encrypted: string): SessionData {
  const [ivHex, authTagHex, data] = encrypted.split(':');

  const iv = Buffer.from(ivHex, 'hex');
  const authTag = Buffer.from(authTagHex, 'hex');
  const decipher = createDecipheriv('aes-256-gcm', ENCRYPTION_KEY, iv);
  decipher.setAuthTag(authTag);

  let decrypted = decipher.update(data, 'hex', 'utf8');
  decrypted += decipher.final('utf8');

  return JSON.parse(decrypted);
}

// 暗号化対応のセッションストア
class EncryptedSessionStore {
  constructor(private inner: RedisSessionStore) {}

  async set(sessionId: string, data: SessionData, ttl: number): Promise<void> {
    const encrypted = encryptSessionData(data);
    // Redis には暗号化された文字列として保存
    await this.inner.setRaw(sessionId, encrypted, ttl);
  }

  async get(sessionId: string): Promise<SessionData | null> {
    const encrypted = await this.inner.getRaw(sessionId);
    if (!encrypted) return null;

    try {
      return decryptSessionData(encrypted);
    } catch {
      // 復号失敗（鍵の変更等）→ セッション無効
      await this.inner.delete(sessionId);
      return null;
    }
  }
}
```

---

## 9. アンチパターン

### 9.1 メモリストアを本番環境で使用

```typescript
// ✗ 危険: メモリストアで本番運用
// サーバー再起動で全ユーザーがログアウトされる
const sessions = new Map<string, SessionData>();

// ✓ 正しい: Redis または DB ストアを使用
const store = new RedisSessionStore({
  redisUrl: process.env.REDIS_URL!,
});
```

### 9.2 KEYS コマンドを使用

```typescript
// ✗ 危険: KEYS * は本番で使用してはいけない
// → O(N) でブロッキング、全キーをスキャンする
const allKeys = await redis.keys('sess:*');

// ✓ 正しい: SCAN を使用（非ブロッキング、イテレーティブ）
let cursor = '0';
const keys: string[] = [];
do {
  const [nextCursor, batch] = await redis.scan(cursor, 'MATCH', 'sess:*', 'COUNT', 100);
  cursor = nextCursor;
  keys.push(...batch);
} while (cursor !== '0');
```

### 9.3 セッションに大量のデータを保存

```typescript
// ✗ 問題: セッションに大きなオブジェクトを保存
await store.set(sessionId, {
  userId: user.id,
  role: user.role,
  cart: hugeCartData,           // NG: 買い物かごのデータ全体
  searchHistory: allHistory,     // NG: 検索履歴全件
  preferences: allPreferences,   // NG: 設定データ全体
}, ttl);

// ✓ 正しい: セッションは最小限に、詳細データは DB に保存
await store.set(sessionId, {
  userId: user.id,
  role: user.role,
  // カートや履歴は DB に保存し、userId で参照
}, ttl);
```

---

## 10. 演習問題

### 演習 1: 基本 — Redis セッションストアの構築（難易度: 基本）

```
課題:
  Redis を使ったセッションストアを実装し、Express/Next.js の
  ミドルウェアとして組み込んでください。

要件:
  ① セッションの CRUD 操作（set, get, delete）
  ② TTL による自動期限切れ
  ③ Sliding Expiration（アクセスで延長）
  ④ ユーザーの全セッション取得

ヒント:
  → ioredis の Pipeline を活用
  → セッション ID は crypto.randomBytes(32) で生成

確認ポイント:
  □ Redis に接続できるか
  □ セッションが保存・取得できるか
  □ TTL 経過後に自動削除されるか
  □ アクセスで有効期限が延長されるか
```

### 演習 2: 応用 — セッション管理 UI の実装（難易度: 応用）

```
課題:
  演習 1 の上に、ユーザーがアクティブセッションを
  管理できる設定画面を実装してください。

要件:
  ① アクティブセッション一覧の表示
  ② 各セッションの情報（IP, UA, 最終アクセス日時）
  ③ 個別セッションの無効化（ログアウト）
  ④ 「他のすべてのデバイスからログアウト」機能
  ⑤ 現在のセッションのハイライト表示

ヒント:
  → User-Agent パーサーで OS / ブラウザ名を取得
  → 現在のセッション ID は Cookie から取得
  → Server Actions で無効化を実装

確認ポイント:
  □ 複数ブラウザでログインした場合に一覧表示されるか
  □ 個別ログアウトが正しく動作するか
  □ 一括ログアウト後に現在のセッションは維持されるか
```

### 演習 3: 発展 — 高可用性セッションストアの設計（難易度: 発展）

```
課題:
  Redis Sentinel を使った高可用性セッションストアを設計し、
  フェイルオーバーのテストを行ってください。

要件:
  ① Redis Sentinel 構成（Master 1 + Replica 2 + Sentinel 3）
  ② フェイルオーバー時のセッション維持
  ③ セッションメトリクスの収集（Prometheus 互換）
  ④ セッションデータの暗号化
  ⑤ Health Check エンドポイント
  ⑥ グレースフルシャットダウン

ヒント:
  → Docker Compose で Sentinel 環境を構築
  → docker stop でマスターを停止し、フェイルオーバーを確認
  → prom-client でメトリクスを公開

確認ポイント:
  □ マスター停止後に自動フェイルオーバーするか
  □ フェイルオーバー中のリクエストはどう処理されるか
  □ メトリクスが正しく収集されるか
  □ 暗号化/復号が正しく動作するか
```

---

## 11. FAQ

### Q1: Redis と Memcached のどちらがセッションストアに適していますか？

```
A: ほとんどのケースで Redis が推奨です。

比較:
  ┌──────────┬──────────────────┬──────────────────┐
  │ 項目      │ Redis            │ Memcached        │
  ├──────────┼──────────────────┼──────────────────┤
  │ データ構造│ 豊富（Set等）    │ Key-Value のみ   │
  │ 永続化   │ RDB / AOF        │ なし             │
  │ TTL管理  │ キーごとに設定可 │ キーごとに設定可  │
  │ クラスタ │ Redis Cluster    │ 一貫性ハッシュ    │
  │ Pub/Sub  │ ✓                │ ✗               │
  │ Lua      │ ✓                │ ✗               │
  │ メモリ効率│ やや劣る         │ 優れている        │
  └──────────┴──────────────────┴──────────────────┘

Memcached が適するケース:
  → 純粋なキャッシュとしてのみ使用
  → メモリ効率が最優先
  → 高頻度の読み取り
```

### Q2: Cookie-based セッションと Server-side セッションの違いは？

```
A: データの保存場所が異なります。

Cookie-based（JWT 等）:
  → セッションデータを Cookie / JWT に含める
  → サーバー側にストア不要
  → ステートレス（スケーリング容易）
  → データサイズに制限（4KB）
  → 即座の無効化が困難

Server-side:
  → Cookie にはセッション ID のみ
  → データはサーバー側ストアに保存
  → サイズ制限なし
  → 即座の無効化が可能
  → ストアが SPOF になりうる

推奨: 機密データが多い場合は Server-side、
      ステートレス性を重視する場合は Cookie-based
```

### Q3: Auth.js (NextAuth) のセッションストアはどう設定しますか？

```
A: Auth.js は adapter を設定することでセッションストアを変更できます。

// JWT セッション（デフォルト、ストア不要）
export const { handlers, auth } = NextAuth({
  session: { strategy: 'jwt' },
});

// DB セッション（Prisma）
export const { handlers, auth } = NextAuth({
  adapter: PrismaAdapter(prisma),
  session: { strategy: 'database' },
});

// Upstash Redis（@auth/upstash-redis-adapter）
import { UpstashRedisAdapter } from '@auth/upstash-redis-adapter';
export const { handlers, auth } = NextAuth({
  adapter: UpstashRedisAdapter(redis),
  session: { strategy: 'database' },
});

注意: JWT 戦略の場合、セッションストアは不要（Cookie に含まれる）
     Database 戦略の場合、adapter 必須
```

---

## まとめ

| 項目 | 推奨 |
|------|------|
| ストア | Redis（本番）、メモリ（開発）、DB（Redis 導入不可時） |
| スケーリング | Redis Sentinel（高可用性）/ Cluster（水平スケール） |
| TTL | Sliding + Absolute のハイブリッド |
| セッション ID | crypto.randomBytes(32)、256 ビット以上 |
| セキュリティ | 固定化防止（ローテーション）、データ暗号化、同時セッション制限 |
| 運用 | メトリクス収集、アクティブセッション管理 UI、定期クリーンアップ |

---

## 次に読むべきガイド

- [[02-csrf-protection.md]] — CSRF 防御
- [[00-session-mechanism.md]] — セッションの仕組み
- [[../04-implementation/00-nextauth-setup.md]] — NextAuth.js セットアップ

---

## 参考文献

1. Redis. "Redis as a Session Store." redis.io, 2024.
2. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. ioredis. "ioredis Documentation." github.com/redis/ioredis, 2024.
4. Upstash. "Serverless Redis." upstash.com/docs, 2024.
5. IETF. "RFC 6265 — HTTP State Management Mechanism." tools.ietf.org, 2011.
6. Express. "express-session." github.com/expressjs/session, 2024.
7. Auth.js. "Database Adapters." authjs.dev/getting-started/adapters, 2024.
