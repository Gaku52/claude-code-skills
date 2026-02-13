# Cookie とセッション管理

> Cookie はブラウザとサーバー間の状態管理の基盤。Cookie の属性設計、セッション管理の仕組み、セキュアなセッション ID 生成、セッションライフサイクルまで、安全なセッション管理の全てを解説する。HTTP はステートレスプロトコルであり、Cookie とセッションの組み合わせがステートフルな Web アプリケーションを実現する唯一の標準的手段である。

## この章で学ぶこと

- [ ] Cookie の属性とセキュリティ設定を理解する
- [ ] セッションの作成・管理・破棄のライフサイクルを把握する
- [ ] セキュアなセッション管理を実装できるようになる
- [ ] セッション固定攻撃・セッションハイジャックの原理と対策を理解する
- [ ] Cookie のスコープ設計とサブドメイン戦略を習得する
- [ ] セッションストアの選択基準と実装パターンを学ぶ

## 前提知識

- HTTP プロトコルの基礎（リクエスト/レスポンスモデル）
- TypeScript / Node.js の基本
- Web セキュリティの基礎概念（XSS, CSRF）

## 関連ガイド

- [[01-session-store.md]] — セッションストア（Redis, DB）の実装
- [[../02-token-auth/00-jwt-basics.md]] — JWT ベースの認証との比較
- [[../04-implementation/02-email-password-auth.md]] — メール・パスワード認証
- [[../03-authorization/01-abac-and-policies.md]] — 認可とポリシー

---

## 1. HTTP のステートレス性と Cookie の役割

```
なぜ Cookie が必要か:

  HTTP はステートレスプロトコル:
    → 各リクエストは独立しており、前回のリクエストの情報を保持しない
    → サーバーは「同じユーザーからの連続リクエスト」を識別できない

  ┌──────────┐                    ┌──────────┐
  │ ブラウザ  │  GET /dashboard    │ サーバー  │
  │          │ ─────────────────> │          │
  │          │                    │ 「誰？」  │
  │          │  GET /profile      │          │
  │          │ ─────────────────> │          │
  │          │                    │ 「誰？」  │
  └──────────┘                    └──────────┘

  Cookie がこの問題を解決:
    → サーバーがレスポンスに Set-Cookie ヘッダーを付与
    → ブラウザが自動的に Cookie をリクエストに付与
    → サーバーは Cookie からユーザーを識別

  ┌──────────┐                           ┌──────────┐
  │ ブラウザ  │  POST /login              │ サーバー  │
  │          │ ─────────────────────────> │          │
  │          │  Set-Cookie: sid=abc123    │ セッション │
  │          │ <───────────────────────── │ 作成      │
  │          │                           │          │
  │          │  GET /dashboard            │          │
  │          │  Cookie: sid=abc123        │          │
  │          │ ─────────────────────────> │          │
  │          │                           │ 「ユーザー │
  │          │  200 OK (認証済み)          │  Aだ」    │
  │          │ <───────────────────────── │          │
  └──────────┘                           └──────────┘

  歴史的経緯:
    1994年: Netscape が Cookie を発明（Lou Montulli）
    1997年: RFC 2109（Cookie の初期標準化）
    2000年: RFC 2965（Cookie2、普及せず）
    2011年: RFC 6265（現行標準、Set-Cookie/Cookie の仕様統一）
    2025年: RFC 6265bis（SameSite のデフォルト等、最新仕様）
```

---

## 2. Cookie の基礎

```
Cookie の仕組み:

  サーバー → レスポンスヘッダーで設定:
    Set-Cookie: session_id=abc123; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=3600

  ブラウザ → リクエストヘッダーで自動送信:
    Cookie: session_id=abc123

  複数の Cookie を設定する場合:
    Set-Cookie: session_id=abc123; HttpOnly; Secure
    Set-Cookie: theme=dark; Path=/
    Set-Cookie: lang=ja; Path=/

  ブラウザからの送信（すべての Cookie が1行で送信される）:
    Cookie: session_id=abc123; theme=dark; lang=ja
```

### 2.1 Cookie の属性一覧

```
Cookie の属性:

  属性          │ 値              │ 説明
  ────────────┼────────────────┼────────────────────────
  HttpOnly     │ true            │ JavaScript からアクセス不可
               │                │ → XSS でのトークン窃取を防止
  Secure       │ true            │ HTTPS のみで送信
               │                │ → 中間者攻撃を防止
  SameSite     │ Strict/Lax/None│ クロスサイトでの送信制御
               │                │ → CSRF 防御
  Path         │ /               │ Cookie が送信されるパス
  Domain       │ .example.com    │ Cookie が有効なドメイン
  Max-Age      │ 3600            │ 有効期間（秒）
  Expires      │ Date            │ 有効期限（日時）
  Partitioned  │ true            │ CHIPS（Cookie Having
               │                │ Independent Partitioned State）
               │                │ → サードパーティ Cookie 制限対応
  __Prefix     │ (名前の接頭辞)   │ Cookie 名による追加セキュリティ
```

### 2.2 SameSite 属性の詳細

```
SameSite の値と動作の違い:

  ┌─────────────────────────────────────────────────────────────┐
  │                    SameSite=Strict                          │
  │                                                            │
  │  同一サイトからのリクエストのみ Cookie 送信                     │
  │                                                            │
  │  ✓ same-origin POST → Cookie 送信                          │
  │  ✓ same-origin GET → Cookie 送信                           │
  │  ✗ cross-site リンク → Cookie 送信されない                   │
  │  ✗ cross-site POST → Cookie 送信されない                    │
  │  ✗ cross-site iframe → Cookie 送信されない                  │
  │                                                            │
  │  問題: 外部サイトからリンクでアクセスするとログイン状態が失われる  │
  │  例: Google 検索結果 → 自社サイト → 未ログイン表示              │
  │      メール内リンク → 自社サイト → 未ログイン表示               │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                    SameSite=Lax（推奨）                      │
  │                                                            │
  │  トップレベルナビゲーション（リンク遷移）の GET は許可           │
  │                                                            │
  │  ✓ same-origin POST → Cookie 送信                          │
  │  ✓ same-origin GET → Cookie 送信                           │
  │  ✓ cross-site リンク（GET）→ Cookie 送信                    │
  │  ✗ cross-site POST → Cookie 送信されない → CSRF 防御        │
  │  ✗ cross-site iframe → Cookie 送信されない                  │
  │  ✗ cross-site img/script → Cookie 送信されない              │
  │                                                            │
  │  バランスが良い: UX を損なわずに CSRF を防御                   │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                    SameSite=None                            │
  │                                                            │
  │  全てのクロスサイトリクエストで送信                             │
  │  Secure 属性が必須（HTTPS のみ）                              │
  │                                                            │
  │  ✓ 全てのリクエストで Cookie 送信                             │
  │                                                            │
  │  用途:                                                      │
  │    → サードパーティ Cookie（広告、埋め込みウィジェット）         │
  │    → クロスドメイン認証（iframe 内ログイン）                    │
  │    → 決済ゲートウェイの 3D Secure                             │
  │                                                            │
  │  注意: ブラウザのサードパーティ Cookie 廃止の影響を受ける        │
  │  → Chrome: Privacy Sandbox / CHIPS への移行                  │
  └─────────────────────────────────────────────────────────────┘
```

### 2.3 Cookie Prefix（名前による追加セキュリティ）

```
Cookie Prefix:

  __Secure- 接頭辞:
    → Secure 属性が必須
    → HTTPS でのみ設定可能
    → 例: __Secure-session_id=abc123

  __Host- 接頭辞（最も厳格）:
    → Secure 属性が必須
    → Domain 属性を指定不可（現在のホストのみ）
    → Path=/ が必須
    → 例: __Host-session_id=abc123

  なぜ重要か:
    → 攻撃者が http:// 経由で Cookie を上書きすることを防止
    → サブドメインからの Cookie 上書き攻撃を防止
    → 「Cookie Tossing」攻撃の防御

  ブラウザの挙動:
    Set-Cookie: __Host-sid=abc123; Secure; Path=/
    → ✓ ブラウザが受け入れる

    Set-Cookie: __Host-sid=abc123; Secure; Path=/; Domain=example.com
    → ✗ ブラウザが拒否する（Domain 属性があるため）

    Set-Cookie: __Host-sid=abc123; Path=/
    → ✗ ブラウザが拒否する（Secure がないため）
```

```typescript
// Cookie 設定のベストプラクティス
import { cookies } from 'next/headers';

async function setSessionCookie(sessionId: string) {
  const cookieStore = await cookies();

  // 本番環境では __Host- prefix を使用
  const cookieName = process.env.NODE_ENV === 'production'
    ? '__Host-session_id'
    : 'session_id';

  cookieStore.set(cookieName, sessionId, {
    httpOnly: true,        // JavaScript からアクセス不可（XSS 対策）
    secure: process.env.NODE_ENV === 'production',  // 本番は HTTPS のみ
    sameSite: 'lax',       // CSRF 防御（クロスサイト POST を拒否）
    path: '/',             // 全パスで有効
    maxAge: 24 * 60 * 60,  // 24時間
    // domain は省略（現在のドメインのみ = __Host- と同等）
  });
}

// Cookie の削除
async function clearSessionCookie() {
  const cookieStore = await cookies();
  const cookieName = process.env.NODE_ENV === 'production'
    ? '__Host-session_id'
    : 'session_id';

  cookieStore.set(cookieName, '', {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: 0,  // 即座に失効
  });
}
```

```typescript
// Express での Cookie 設定
import express from 'express';
import session from 'express-session';

const app = express();

app.use(session({
  name: '__Host-sid',     // Cookie 名（デフォルトの 'connect.sid' から変更）
  secret: process.env.SESSION_SECRET!,
  resave: false,          // セッションが変更されていなければ再保存しない
  saveUninitialized: false, // 未初期化セッションは保存しない（GDPR 対応）
  cookie: {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 24 * 60 * 60 * 1000, // 24時間（ミリ秒）
    // domain を省略して __Host- prefix の要件を満たす
  },
}));

// セッション変数の設定（ログイン時）
app.post('/login', async (req, res) => {
  const user = await authenticateUser(req.body.email, req.body.password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // セッション再生成（セッション固定攻撃対策）
  req.session.regenerate((err) => {
    if (err) {
      return res.status(500).json({ error: 'Session error' });
    }
    req.session.userId = user.id;
    req.session.role = user.role;
    req.session.loginAt = Date.now();

    req.session.save((err) => {
      if (err) {
        return res.status(500).json({ error: 'Session save error' });
      }
      res.json({ user: { id: user.id, email: user.email } });
    });
  });
});
```

---

## 3. Cookie のスコープ設計

```
Domain 属性とスコープ:

  Domain を省略した場合:
    → 設定元のホストのみで有効（サブドメインには送信されない）
    → 例: app.example.com で設定 → app.example.com のみ
    → 最も安全

  Domain=.example.com を指定:
    → example.com とすべてのサブドメインで有効
    → 例: app.example.com, api.example.com, admin.example.com
    → 便利だがリスクがある

  ┌──────────────────────────────────────────────────────────┐
  │             Domain スコープの設計パターン                   │
  │                                                          │
  │  パターン1: 単一ドメイン（推奨）                            │
  │    app.example.com → Cookie: Domain 省略                  │
  │    api.example.com → Cookie なし（トークン認証）             │
  │    → 最小権限の原則に沿う                                   │
  │                                                          │
  │  パターン2: サブドメイン共有                                 │
  │    *.example.com → Cookie: Domain=.example.com            │
  │    → SSO 的に使いたい場合                                   │
  │    → サブドメインの XSS が全体に影響（リスク）                │
  │                                                          │
  │  パターン3: BFF パターン（推奨）                             │
  │    app.example.com → Cookie: Domain 省略                   │
  │    app.example.com/api/* → リバースプロキシで API へ          │
  │    → Cookie は同一オリジンで管理                             │
  │    → API サーバーと Cookie を分離                            │
  └──────────────────────────────────────────────────────────┘

  Path 属性の注意点:
    Path=/ → 全パスで Cookie 送信
    Path=/admin → /admin 以下のみ
    ⚠️ Path はセキュリティ機能ではない
      → 同一オリジンの JavaScript は任意パスの Cookie を読み取れる
      → iframe で /admin を開けば Cookie にアクセス可能
```

```typescript
// マルチサービス構成での Cookie スコープ設計
// BFF (Backend for Frontend) パターン

// next.config.js - リバースプロキシ設定
const nextConfig = {
  async rewrites() {
    return [
      {
        // /api/* へのリクエストを API サーバーに転送
        // Cookie は同一ドメインなので自動送信される
        source: '/api/:path*',
        destination: 'http://internal-api:3001/api/:path*',
      },
    ];
  },
};

// API サーバー側 - Cookie の検証
// Cookie は BFF 経由で同一ドメインから送信される
app.use('/api', (req, res, next) => {
  const sessionId = req.cookies['__Host-session_id'];
  if (!sessionId) {
    return res.status(401).json({ error: 'No session' });
  }
  next();
});
```

---

## 4. セッション管理

### 4.1 セッションのライフサイクル

```
セッションのライフサイクル:

  ① 作成（ログイン時）:
     → セッション ID 生成（暗号的に安全なランダム値）
     → セッションデータをストアに保存
     → Cookie にセッション ID を設定

  ② 使用（リクエストごと）:
     → Cookie からセッション ID を取得
     → ストアからセッションデータを取得
     → ユーザー情報を元にリクエスト処理

  ③ 更新（セキュリティイベント時）:
     → セッション ID のローテーション
     → 権限変更時、パスワード変更時
     → セッション固定攻撃の防御

  ④ 破棄（ログアウト時）:
     → ストアからセッションデータを削除
     → Cookie を無効化
     → 関連するすべてのセッションを無効化（オプション）

  タイムライン:

  ログイン     リクエスト   権限変更    ログアウト
    │            │          │           │
    ▼            ▼          ▼           ▼
  ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐
  │ 作成 │───>│ 使用 │───>│ 更新 │───>│ 破棄 │
  │     │    │     │    │(ID  │    │     │
  │ ID  │    │ ID  │    │ 再生│    │ ID  │
  │ =A  │    │ =A  │    │ =B) │    │ =B  │
  └─────┘    └─────┘    └─────┘    └─────┘
                │                     │
              ストア               ストア
              から                 から
              取得                 削除
```

### 4.2 セキュアなセッション ID 生成

```
セッション ID の要件（OWASP 準拠）:

  ① エントロピー: 最低128ビット（推奨: 256ビット）
     → 2^128 の組み合わせ → ブルートフォース不可能
     → crypto.randomBytes(32) = 256ビット

  ② 予測不可能性: CSPRNG（暗号論的擬似乱数生成器）を使用
     → Math.random() は使用禁止（予測可能）
     → Date.now() は使用禁止（推測可能）
     → UUID v4 は使用可能だが、エントロピーが122ビットと少なめ

  ③ 一意性: 衝突が事実上発生しない
     → 256ビットランダム値の衝突確率 ≈ 0
     → 誕生日攻撃を考慮しても 2^128 の試行が必要

  各言語でのセッション ID 生成:

  Node.js:  crypto.randomBytes(32).toString('hex')     // 64文字の16進数
  Python:   secrets.token_hex(32)                      // 64文字の16進数
  Go:       rand.Read(make([]byte, 32))                // crypto/rand
  Ruby:     SecureRandom.hex(32)                       // 64文字の16進数
  Java:     new SecureRandom().generateSeed(32)        // 32バイト

  ⚠️ 危険なセッション ID 生成の例:
    × Math.random().toString(36).substr(2)  // 予測可能
    × Date.now().toString(36)               // 推測可能
    × `user_${userId}_${timestamp}`         // 情報漏洩＋推測可能
    × md5(username + password)              // 固定値＋レインボーテーブル
```

```typescript
// セッション管理の完全実装
import crypto from 'crypto';

interface SessionData {
  userId: string;
  role: string;
  createdAt: number;          // セッション作成時刻
  lastAccessedAt: number;     // 最終アクセス時刻
  ipAddress: string;          // クライアント IP
  userAgent: string;          // ブラウザ情報
  absoluteExpiry: number;     // 絶対有効期限
  csrfToken: string;          // CSRF トークン
  metadata?: Record<string, unknown>; // 追加メタデータ
}

interface SessionStore {
  get(id: string): Promise<SessionData | null>;
  set(id: string, data: SessionData, options?: { ttl?: number }): Promise<void>;
  delete(id: string): Promise<void>;
  findByUserId(userId: string): Promise<Array<{ id: string; data: SessionData }>>;
  deleteByUserId(userId: string): Promise<number>;
}

interface SessionInfo {
  id: string;              // 短縮表示用
  createdAt: Date;
  lastAccessedAt: Date;
  ipAddress: string;
  userAgent: string;
  isCurrent: boolean;
}

class SessionManager {
  private readonly SESSION_ID_BYTES = 32;      // 256ビット
  private readonly IDLE_TIMEOUT = 30 * 60;     // 30分（スライディング）
  private readonly ABSOLUTE_TIMEOUT = 24 * 60 * 60; // 24時間（絶対）
  private readonly MAX_SESSIONS_PER_USER = 5;  // ユーザーあたりの最大セッション数

  constructor(private store: SessionStore) {}

  // セッション ID 生成
  private generateSessionId(): string {
    return crypto.randomBytes(this.SESSION_ID_BYTES).toString('hex');
  }

  // CSRF トークン生成
  private generateCsrfToken(): string {
    return crypto.randomBytes(32).toString('hex');
  }

  // セッション作成
  async create(
    userData: { userId: string; role: string },
    req: Request
  ): Promise<{ sessionId: string; csrfToken: string }> {
    // セッション数の制限チェック
    await this.enforceSessionLimit(userData.userId);

    const sessionId = this.generateSessionId();
    const csrfToken = this.generateCsrfToken();
    const now = Date.now();

    const sessionData: SessionData = {
      userId: userData.userId,
      role: userData.role,
      createdAt: now,
      lastAccessedAt: now,
      ipAddress: this.getClientIP(req),
      userAgent: req.headers.get('user-agent') || '',
      absoluteExpiry: now + this.ABSOLUTE_TIMEOUT * 1000,
      csrfToken,
    };

    await this.store.set(sessionId, sessionData, { ttl: this.IDLE_TIMEOUT });

    return { sessionId, csrfToken };
  }

  // セッション取得と検証
  async get(sessionId: string): Promise<SessionData | null> {
    if (!sessionId || typeof sessionId !== 'string') {
      return null;
    }

    // セッション ID のフォーマット検証（64文字の16進数）
    if (!/^[0-9a-f]{64}$/.test(sessionId)) {
      return null;
    }

    const data = await this.store.get(sessionId);
    if (!data) return null;

    // 絶対有効期限チェック
    if (Date.now() > data.absoluteExpiry) {
      await this.store.delete(sessionId);
      return null;
    }

    // アクセス時間を更新（スライディング有効期限）
    data.lastAccessedAt = Date.now();
    await this.store.set(sessionId, data, { ttl: this.IDLE_TIMEOUT });

    return data;
  }

  // セッション ID ローテーション（セッション固定攻撃対策）
  async rotate(oldSessionId: string): Promise<string> {
    const data = await this.store.get(oldSessionId);
    if (!data) throw new Error('Session not found');

    const newSessionId = this.generateSessionId();

    // 旧セッションを削除、新セッションを作成
    await this.store.delete(oldSessionId);

    // CSRF トークンも再生成
    data.csrfToken = this.generateCsrfToken();
    await this.store.set(newSessionId, data, { ttl: this.IDLE_TIMEOUT });

    return newSessionId;
  }

  // ログアウト（単一セッション）
  async destroy(sessionId: string): Promise<void> {
    await this.store.delete(sessionId);
  }

  // 全セッション無効化（パスワード変更時等）
  async destroyAllForUser(userId: string): Promise<number> {
    return this.store.deleteByUserId(userId);
  }

  // 特定セッション以外を全て無効化（「他の全デバイスからログアウト」）
  async destroyOtherSessions(userId: string, currentSessionId: string): Promise<number> {
    const sessions = await this.store.findByUserId(userId);
    let count = 0;
    for (const session of sessions) {
      if (session.id !== currentSessionId) {
        await this.store.delete(session.id);
        count++;
      }
    }
    return count;
  }

  // セッション数制限の強制
  private async enforceSessionLimit(userId: string): Promise<void> {
    const sessions = await this.store.findByUserId(userId);
    if (sessions.length >= this.MAX_SESSIONS_PER_USER) {
      // 最も古いセッションを削除
      const sorted = sessions.sort(
        (a, b) => a.data.lastAccessedAt - b.data.lastAccessedAt
      );
      const toRemove = sorted.slice(0, sessions.length - this.MAX_SESSIONS_PER_USER + 1);
      for (const session of toRemove) {
        await this.store.delete(session.id);
      }
    }
  }

  // アクティブセッション一覧（ユーザーに表示）
  async getActiveSessions(
    userId: string,
    currentSessionId: string
  ): Promise<SessionInfo[]> {
    const sessions = await this.store.findByUserId(userId);
    return sessions.map((s) => ({
      id: s.id.substring(0, 8) + '...',  // 完全な ID は非公開
      createdAt: new Date(s.data.createdAt),
      lastAccessedAt: new Date(s.data.lastAccessedAt),
      ipAddress: this.maskIP(s.data.ipAddress),
      userAgent: s.data.userAgent,
      isCurrent: s.id === currentSessionId,
    }));
  }

  // IP アドレスのマスキング
  private maskIP(ip: string): string {
    if (ip.includes(':')) {
      // IPv6: 最初の4セグメントのみ
      return ip.split(':').slice(0, 4).join(':') + ':...';
    }
    // IPv4: 最後のオクテットをマスク
    return ip.replace(/\.\d+$/, '.***');
  }

  // クライアント IP の取得
  private getClientIP(req: Request): string {
    // プロキシ環境ではヘッダーから取得
    const forwarded = req.headers.get('x-forwarded-for');
    if (forwarded) {
      // 最初の IP（クライアント IP）を取得
      return forwarded.split(',')[0].trim();
    }
    const realIP = req.headers.get('x-real-ip');
    if (realIP) return realIP;

    // 直接接続の場合
    return '0.0.0.0'; // フレームワーク依存
  }
}
```

### 4.3 Redis ベースのセッションストア実装

```typescript
// Redis を使ったセッションストアの実装
import Redis from 'ioredis';

class RedisSessionStore implements SessionStore {
  private redis: Redis;
  private readonly PREFIX = 'sess:';
  private readonly USER_INDEX_PREFIX = 'user_sess:';

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl, {
      // 接続プール設定
      maxRetriesPerRequest: 3,
      retryStrategy: (times) => {
        if (times > 3) return null; // 3回リトライ後に停止
        return Math.min(times * 200, 2000); // 200ms, 400ms, 600ms
      },
      // TLS 設定（本番環境）
      tls: process.env.NODE_ENV === 'production' ? {} : undefined,
    });
  }

  async get(id: string): Promise<SessionData | null> {
    const data = await this.redis.get(this.PREFIX + id);
    if (!data) return null;

    try {
      return JSON.parse(data);
    } catch {
      // 破損したデータは削除
      await this.delete(id);
      return null;
    }
  }

  async set(id: string, data: SessionData, options?: { ttl?: number }): Promise<void> {
    const key = this.PREFIX + id;
    const serialized = JSON.stringify(data);

    if (options?.ttl) {
      // TTL 付きで保存（スライディング有効期限）
      await this.redis.setex(key, options.ttl, serialized);
    } else {
      await this.redis.set(key, serialized);
    }

    // ユーザー → セッション ID のインデックスを更新
    await this.redis.sadd(this.USER_INDEX_PREFIX + data.userId, id);
    // インデックス自体にも TTL を設定（孤立防止）
    await this.redis.expire(
      this.USER_INDEX_PREFIX + data.userId,
      (options?.ttl || 86400) + 3600 // セッション TTL + 余裕1時間
    );
  }

  async delete(id: string): Promise<void> {
    // セッションデータを取得してからユーザーインデックスを更新
    const data = await this.get(id);
    if (data) {
      await this.redis.srem(this.USER_INDEX_PREFIX + data.userId, id);
    }
    await this.redis.del(this.PREFIX + id);
  }

  async findByUserId(userId: string): Promise<Array<{ id: string; data: SessionData }>> {
    const sessionIds = await this.redis.smembers(this.USER_INDEX_PREFIX + userId);
    const results: Array<{ id: string; data: SessionData }> = [];

    for (const id of sessionIds) {
      const data = await this.get(id);
      if (data) {
        results.push({ id, data });
      } else {
        // 期限切れのセッション ID をインデックスから削除
        await this.redis.srem(this.USER_INDEX_PREFIX + userId, id);
      }
    }

    return results;
  }

  async deleteByUserId(userId: string): Promise<number> {
    const sessionIds = await this.redis.smembers(this.USER_INDEX_PREFIX + userId);
    if (sessionIds.length === 0) return 0;

    // パイプラインで一括削除（パフォーマンス最適化）
    const pipeline = this.redis.pipeline();
    for (const id of sessionIds) {
      pipeline.del(this.PREFIX + id);
    }
    pipeline.del(this.USER_INDEX_PREFIX + userId);
    await pipeline.exec();

    return sessionIds.length;
  }

  // ヘルスチェック
  async ping(): Promise<boolean> {
    try {
      const result = await this.redis.ping();
      return result === 'PONG';
    } catch {
      return false;
    }
  }

  // クリーンアップ（定期実行）
  async cleanup(): Promise<number> {
    // Redis の TTL により自動的にクリーンアップされるが、
    // ユーザーインデックスの孤立エントリを掃除
    let cleaned = 0;
    let cursor = '0';

    do {
      const [nextCursor, keys] = await this.redis.scan(
        cursor,
        'MATCH',
        this.USER_INDEX_PREFIX + '*',
        'COUNT',
        100
      );
      cursor = nextCursor;

      for (const key of keys) {
        const sessionIds = await this.redis.smembers(key);
        for (const id of sessionIds) {
          const exists = await this.redis.exists(this.PREFIX + id);
          if (!exists) {
            await this.redis.srem(key, id);
            cleaned++;
          }
        }
      }
    } while (cursor !== '0');

    return cleaned;
  }
}
```

---

## 5. セッション固定攻撃と対策

```
セッション固定攻撃（Session Fixation）:

  攻撃フロー:

  攻撃者                         被害者                       サーバー
    │                             │                           │
    │ ① サイトにアクセス            │                           │
    │ ─────────────────────────────────────────────────────>  │
    │                             │                           │
    │ ② セッション ID 取得          │                           │
    │ (session_id = "known_id")   │                           │
    │ <─────────────────────────────────────────────────────  │
    │                             │                           │
    │ ③ リンクを送信               │                           │
    │ (https://site.com/?sid=     │                           │
    │  known_id)                  │                           │
    │ ──────────────────────────> │                           │
    │                             │                           │
    │                             │ ④ リンクをクリック          │
    │                             │ ──────────────────────>   │
    │                             │                           │
    │                             │ ⑤ ログイン                │
    │                             │ (session_id=known_id)     │
    │                             │ ──────────────────────>   │
    │                             │                           │
    │ ⑥ 同じ session_id でアクセス │                           │
    │ ─────────────────────────────────────────────────────>  │
    │                             │                           │
    │ ⑦ ログイン済み状態でアクセス  │                           │
    │ <─────────────────────────────────────────────────────  │

  対策（必須）:
    1. ログイン成功時にセッション ID を再生成（ローテーション）
    2. URL パラメータからのセッション ID 受け入れを拒否
    3. Cookie のみでセッション ID を管理
    4. ログイン前のセッションデータを新セッションにコピーしない
```

```typescript
// ログイン処理でのセッション ID ローテーション
async function handleLogin(
  email: string,
  password: string,
  req: Request,
  res: Response
) {
  // 1. ユーザー認証
  const user = await authenticateUser(email, password);
  if (!user) {
    // ユーザー列挙攻撃を防ぐため、具体的なエラーメッセージは返さない
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // 2. 既存セッションがあれば破棄（セッション固定攻撃対策の核心）
  const oldSessionId = req.cookies['__Host-session_id'];
  if (oldSessionId) {
    await sessionManager.destroy(oldSessionId);
  }

  // 3. 新しいセッション ID で作成
  const { sessionId, csrfToken } = await sessionManager.create(
    { userId: user.id, role: user.role },
    req
  );

  // 4. Cookie に新しいセッション ID を設定
  res.cookie('__Host-session_id', sessionId, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 24 * 60 * 60 * 1000,
  });

  // 5. CSRF トークンを返す（JavaScript で取得させる）
  return res.json({
    user: { id: user.id, email: user.email },
    csrfToken,
  });
}

// 権限変更時のセッション ID ローテーション
async function handleRoleChange(
  userId: string,
  newRole: string,
  req: Request,
  res: Response
) {
  const sessionId = req.cookies['__Host-session_id'];
  if (!sessionId) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  // セッション ID を再生成（権限昇格攻撃の防止）
  const newSessionId = await sessionManager.rotate(sessionId);

  // 新しい Cookie を設定
  res.cookie('__Host-session_id', newSessionId, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 24 * 60 * 60 * 1000,
  });

  res.json({ success: true });
}
```

---

## 6. セッションハイジャック攻撃と対策

```
セッションハイジャックの手法:

  ① ネットワーク盗聴（Sniffing）:
     → 暗号化されていない通信でセッション ID を傍受
     → 対策: HTTPS 必須 + Secure Cookie 属性

  ② XSS（Cross-Site Scripting）:
     → JavaScript でdocument.cookie を読み取り
     → 対策: HttpOnly Cookie 属性 + CSP ヘッダー

  ③ CSRF（Cross-Site Request Forgery）:
     → ユーザーのブラウザから意図しないリクエストを送信
     → 対策: SameSite=Lax + CSRF トークン

  ④ セッション ID 推測:
     → 弱い乱数で生成されたセッション ID を推測
     → 対策: CSPRNG で 256ビット以上のランダム値

  ⑤ サイドチャネル攻撃:
     → Referer ヘッダー経由でセッション ID 漏洩
     → 対策: セッション ID は URL に含めない + Referrer-Policy ヘッダー

  多層防御の全体像:

  ┌────────────────────────────────────────────┐
  │              防御レイヤー                    │
  │                                            │
  │  Layer 1: 通信路          HTTPS 必須        │
  │  Layer 2: Cookie 属性     HttpOnly, Secure  │
  │  Layer 3: CSRF 防御       SameSite, Token   │
  │  Layer 4: セッション ID   CSPRNG 256bit     │
  │  Layer 5: ローテーション  ログイン/権限変更時  │
  │  Layer 6: 有効期限        スライディング+絶対 │
  │  Layer 7: 異常検知        IP/UA変更検知      │
  │  Layer 8: 並行制御        最大セッション数    │
  └────────────────────────────────────────────┘
```

```typescript
// セッションの異常検知ミドルウェア
async function sessionAnomalyDetection(
  req: Request,
  session: SessionData
): Promise<{ valid: boolean; reason?: string }> {
  const currentIP = getClientIP(req);
  const currentUA = req.headers.get('user-agent') || '';

  // 1. IP アドレスの急激な変更を検知
  if (session.ipAddress !== currentIP) {
    // 同一 ISP/国からの変更は許容（モバイル回線等）
    const isSameRegion = await checkSameGeoRegion(session.ipAddress, currentIP);

    if (!isSameRegion) {
      // 異なる地域からのアクセス → 要注意
      await logSecurityEvent({
        type: 'session_ip_change',
        userId: session.userId,
        oldIP: session.ipAddress,
        newIP: currentIP,
        severity: 'warning',
      });

      // 高セキュリティモードでは再認証を要求
      if (await isHighSecurityMode(session.userId)) {
        return { valid: false, reason: 'ip_change_detected' };
      }
    }
  }

  // 2. User-Agent の変更を検知
  if (session.userAgent && session.userAgent !== currentUA) {
    // User-Agent が完全に変わった場合（ブラウザの変更は通常ありえない）
    await logSecurityEvent({
      type: 'session_ua_change',
      userId: session.userId,
      oldUA: session.userAgent,
      newUA: currentUA,
      severity: 'high',
    });

    return { valid: false, reason: 'useragent_change_detected' };
  }

  // 3. 不可能な移動速度の検知（Impossible Travel）
  if (session.ipAddress !== currentIP) {
    const timeDiff = Date.now() - session.lastAccessedAt;
    const distance = await calculateGeoDistance(session.ipAddress, currentIP);
    const speedKmH = distance / (timeDiff / 3600000);

    if (speedKmH > 1000) { // 時速1000km以上は物理的に不可能
      await logSecurityEvent({
        type: 'impossible_travel',
        userId: session.userId,
        speed: speedKmH,
        severity: 'critical',
      });

      return { valid: false, reason: 'impossible_travel' };
    }
  }

  return { valid: true };
}
```

---

## 7. セッションの有効期限戦略

```
有効期限の種類:

  ① 絶対有効期限（Absolute Timeout）:
     → セッション作成からN時間後に失効
     → 例: 24時間後に自動ログアウト
     → セキュリティが厳しいシステム向け
     → アクティブであっても強制ログアウト

  ② スライディング有効期限（Sliding/Idle Timeout）:
     → 最後のアクティビティからN分後に失効
     → アクティブなユーザーはセッション維持
     → 例: 30分操作なしで失効
     → Redis の TTL 更新で実現

  ③ ハイブリッド（推奨）:
     → スライディング + 絶対有効期限の組合せ
     → 例: 操作があれば延長するが、最大72時間で失効
     → 両方の利点を組み合わせ

  タイムライン:

  ログイン        アクセス   アクセス   30分放置     失効
    │              │         │         │           │
    ▼              ▼         ▼         ▼           ▼
    ├──────────────┼─────────┼─────────┼───────────┤
    │← 30min TTL →│← reset →│← reset →│← 30min → │
    │                                              │
    │← ─── 絶対有効期限（24時間）─── ─── ─── ──→    │
    │                                              │
    │  アクティブなら TTL がリセットされ続ける         │
    │  しかし絶対有効期限に達したら強制失効            │

  推奨の組合せ:

  ┌──────────────────┬───────────┬──────────┬──────────────────┐
  │ アプリケーション   │ スライディング│ 絶対     │ Remember Me     │
  ├──────────────────┼───────────┼──────────┼──────────────────┤
  │ 一般的なWebアプリ  │ 30分       │ 24時間   │ 30日             │
  │ ECサイト          │ 60分       │ 72時間   │ 90日             │
  │ 金融・医療        │ 15分       │ 8時間    │ なし             │
  │ ソーシャルメディア │ 24時間     │ 30日     │ 365日            │
  │ 管理画面          │ 15分       │ 4時間    │ なし             │
  │ B2B SaaS         │ 60分       │ 12時間   │ 30日             │
  └──────────────────┴───────────┴──────────┴──────────────────┘
```

```typescript
// Remember Me 機能の安全な実装
import crypto from 'crypto';

interface RememberMeToken {
  id: string;
  userId: string;
  tokenHash: string;    // ハッシュ化して保存（DB 漏洩対策）
  series: string;       // トークンシリーズ（盗難検知用）
  expiresAt: Date;
  createdAt: Date;
  lastUsedAt: Date;
  userAgent: string;
  ipAddress: string;
}

class RememberMeManager {
  // Remember Me トークンの作成
  async create(
    userId: string,
    req: Request
  ): Promise<{ token: string; series: string }> {
    const token = crypto.randomBytes(32).toString('hex');
    const series = crypto.randomBytes(16).toString('hex');
    const tokenHash = crypto.createHash('sha256').update(token).digest('hex');

    await db.rememberToken.create({
      data: {
        userId,
        tokenHash,
        series,
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30日
        createdAt: new Date(),
        lastUsedAt: new Date(),
        userAgent: req.headers.get('user-agent') || '',
        ipAddress: getClientIP(req),
      },
    });

    return { token: `${series}:${token}`, series };
  }

  // Remember Me トークンの検証と更新
  async verify(
    cookieValue: string,
    req: Request
  ): Promise<{ userId: string; newToken: string } | null> {
    const [series, token] = cookieValue.split(':');
    if (!series || !token) return null;

    const tokenHash = crypto.createHash('sha256').update(token).digest('hex');

    // シリーズでトークンを検索
    const record = await db.rememberToken.findFirst({
      where: {
        series,
        expiresAt: { gt: new Date() },
      },
    });

    if (!record) return null;

    // トークンハッシュの検証
    if (record.tokenHash !== tokenHash) {
      // シリーズは一致するがトークンが不一致 → トークン盗難の可能性
      // このシリーズの全トークンを無効化
      await db.rememberToken.deleteMany({ where: { series } });

      // ユーザーに警告通知
      await sendSecurityAlert(record.userId, {
        type: 'remember_me_theft_detected',
        ip: getClientIP(req),
        userAgent: req.headers.get('user-agent') || '',
      });

      // 全セッションを無効化（被害を最小化）
      await sessionManager.destroyAllForUser(record.userId);

      return null;
    }

    // トークンローテーション（使用したら新しいトークンを発行）
    const newToken = crypto.randomBytes(32).toString('hex');
    const newTokenHash = crypto.createHash('sha256').update(newToken).digest('hex');

    await db.rememberToken.update({
      where: { id: record.id },
      data: {
        tokenHash: newTokenHash,
        lastUsedAt: new Date(),
        userAgent: req.headers.get('user-agent') || '',
        ipAddress: getClientIP(req),
      },
    });

    return {
      userId: record.userId,
      newToken: `${series}:${newToken}`,
    };
  }

  // Remember Me トークンの無効化
  async revoke(series: string): Promise<void> {
    await db.rememberToken.deleteMany({ where: { series } });
  }

  // ユーザーの全 Remember Me トークンを無効化
  async revokeAllForUser(userId: string): Promise<void> {
    await db.rememberToken.deleteMany({ where: { userId } });
  }
}

// セッション切れ時の Remember Me による自動復元
async function restoreSession(req: Request, res: Response): Promise<SessionData | null> {
  // まずセッションを確認
  const sessionId = getCookie(req, '__Host-session_id');
  if (sessionId) {
    const session = await sessionManager.get(sessionId);
    if (session) return session;
  }

  // セッション切れなら Remember Me トークンを確認
  const rememberCookie = getCookie(req, '__Secure-remember_me');
  if (!rememberCookie) return null;

  const rememberMeManager = new RememberMeManager();
  const result = await rememberMeManager.verify(rememberCookie, req);
  if (!result) {
    // 無効なトークン → Cookie を削除
    clearCookie(res, '__Secure-remember_me');
    return null;
  }

  // 新しいセッションを作成
  const user = await db.user.findUnique({ where: { id: result.userId } });
  if (!user || !user.active) return null;

  const { sessionId: newSessionId } = await sessionManager.create(
    { userId: user.id, role: user.role },
    req
  );

  // 新しいセッション Cookie を設定
  setCookie(res, '__Host-session_id', newSessionId, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 24 * 60 * 60,
  });

  // 新しい Remember Me Cookie を設定（ローテーション）
  setCookie(res, '__Secure-remember_me', result.newToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 30 * 24 * 60 * 60,
  });

  // 監査ログ
  await logAuthEvent({
    type: 'session_restored_via_remember_me',
    userId: user.id,
    ip: getClientIP(req),
  });

  return sessionManager.get(newSessionId);
}
```

---

## 8. CSRF 対策とセッション

```
CSRF（Cross-Site Request Forgery）攻撃:

  攻撃フロー:
    ① ユーザーが bank.com にログイン済み（Cookie あり）
    ② 攻撃者のサイト evil.com にアクセス
    ③ evil.com に隠しフォーム: <form action="https://bank.com/transfer" method="POST">
    ④ JavaScript で自動送信
    ⑤ ブラウザが bank.com の Cookie を自動付与
    ⑥ bank.com は正規リクエストと区別できない

  対策の比較:

  ┌──────────────────┬──────────────┬────────────────┬──────────────┐
  │ 対策             │ 実装難易度     │ 保護レベル      │ 副作用       │
  ├──────────────────┼──────────────┼────────────────┼──────────────┤
  │ SameSite=Lax     │ 低           │ 高             │ 少ない       │
  │ SameSite=Strict  │ 低           │ 最高           │ UX 影響あり  │
  │ CSRF トークン     │ 中           │ 高             │ なし         │
  │ Double Submit    │ 中           │ 中             │ なし         │
  │ Origin ヘッダー   │ 低           │ 中             │ 古いブラウザ  │
  │ Referer 検証     │ 低           │ 低             │ プライバシー  │
  └──────────────────┴──────────────┴────────────────┴──────────────┘

  推奨: SameSite=Lax + CSRF トークン（多層防御）
```

```typescript
// CSRF 対策の実装（Synchronizer Token Pattern）

// セッション作成時に CSRF トークンを生成
function generateCsrfToken(): string {
  return crypto.randomBytes(32).toString('hex');
}

// CSRF トークン検証ミドルウェア
function csrfProtection() {
  return async (req: Request, res: Response, next: Function) => {
    // GET, HEAD, OPTIONS はスキップ（安全なメソッド）
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      return next();
    }

    const sessionId = req.cookies['__Host-session_id'];
    if (!sessionId) {
      return res.status(403).json({ error: 'No session' });
    }

    const session = await sessionManager.get(sessionId);
    if (!session) {
      return res.status(403).json({ error: 'Invalid session' });
    }

    // CSRF トークンの検証
    const csrfToken = req.headers['x-csrf-token']
      || req.body?._csrf
      || req.query?._csrf;

    if (!csrfToken || !timingSafeEqual(csrfToken, session.csrfToken)) {
      await logSecurityEvent({
        type: 'csrf_token_mismatch',
        userId: session.userId,
        ip: getClientIP(req),
        severity: 'warning',
      });
      return res.status(403).json({ error: 'Invalid CSRF token' });
    }

    next();
  };
}

// タイミングセーフな文字列比較（タイミング攻撃防止）
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  const bufA = Buffer.from(a);
  const bufB = Buffer.from(b);
  return crypto.timingSafeEqual(bufA, bufB);
}

// Double Submit Cookie パターン（代替手法）
// Cookie と リクエストヘッダーの両方に同じトークンを設定
function doubleSubmitCsrf() {
  return async (req: Request, res: Response, next: Function) => {
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      // GET リクエスト時に CSRF トークン Cookie を設定
      if (!req.cookies['csrf_token']) {
        const token = crypto.randomBytes(32).toString('hex');
        res.cookie('csrf_token', token, {
          httpOnly: false,  // JavaScript からの読み取りを許可
          secure: true,
          sameSite: 'lax',
          path: '/',
        });
      }
      return next();
    }

    // POST リクエスト時に Cookie とヘッダーの一致を検証
    const cookieToken = req.cookies['csrf_token'];
    const headerToken = req.headers['x-csrf-token'];

    if (!cookieToken || !headerToken || cookieToken !== headerToken) {
      return res.status(403).json({ error: 'CSRF validation failed' });
    }

    next();
  };
}

// フロントエンド（React）での CSRF トークン送信
// fetch のラッパー
async function secureFetch(url: string, options: RequestInit = {}) {
  const csrfToken = document.querySelector<HTMLMetaElement>(
    'meta[name="csrf-token"]'
  )?.content;

  return fetch(url, {
    ...options,
    credentials: 'same-origin', // Cookie を送信
    headers: {
      ...options.headers,
      'Content-Type': 'application/json',
      'X-CSRF-Token': csrfToken || '',
    },
  });
}
```

---

## 9. 並行セッション管理

```
並行セッションの制御:

  ① 無制限（デフォルト）:
     → 複数デバイスから同時ログイン可能
     → 一般的な Web アプリ
     → 管理が簡単だが、不正利用の検知が難しい

  ② 最大数制限:
     → 最大5デバイスまで等
     → 古いセッションから自動ログアウト
     → バランスが良い（推奨）

  ③ 単一セッション:
     → 1デバイスのみ
     → 金融系で多い
     → 新しいログインで旧セッション無効化
     → UX への影響が大きい（別デバイスで強制ログアウト）

  ④ デバイスタイプ別:
     → PC: 1セッション、モバイル: 1セッション
     → 各デバイスタイプ1つずつ許可
     → デバイスフィンガープリントの信頼性が課題

  セッション管理 UI のパターン:

  ┌─────────────────────────────────────────────────────┐
  │  アクティブなセッション                                │
  │                                                     │
  │  🖥️ Chrome on macOS          ★ 現在のセッション      │
  │     最終アクセス: 2分前                                │
  │     IP: 203.0.113.***                                │
  │                                                     │
  │  📱 Safari on iOS                                    │
  │     最終アクセス: 3時間前        [ログアウト]           │
  │     IP: 198.51.100.***                               │
  │                                                     │
  │  🖥️ Firefox on Windows                              │
  │     最終アクセス: 2日前          [ログアウト]           │
  │     IP: 192.0.2.***                                  │
  │                                                     │
  │  [他の全デバイスからログアウト]                          │
  └─────────────────────────────────────────────────────┘
```

```typescript
// アクティブセッション管理 API
import { parseUserAgent } from './utils/ua-parser';

// セッション一覧の取得
app.get('/api/sessions', async (req, res) => {
  const sessionId = req.cookies['__Host-session_id'];
  const session = await sessionManager.get(sessionId);
  if (!session) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  const activeSessions = await sessionManager.getActiveSessions(
    session.userId,
    sessionId
  );

  // User-Agent を人間が読める形式に変換
  const formatted = activeSessions.map((s) => ({
    ...s,
    device: parseUserAgent(s.userAgent),
  }));

  res.json({ sessions: formatted });
});

// 特定セッションの無効化
app.delete('/api/sessions/:sessionDisplayId', async (req, res) => {
  const currentSessionId = req.cookies['__Host-session_id'];
  const session = await sessionManager.get(currentSessionId);
  if (!session) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  // 表示用 ID から実際のセッションを特定
  const allSessions = await sessionManager.getActiveSessions(
    session.userId,
    currentSessionId
  );

  const target = allSessions.find(
    (s) => s.id === req.params.sessionDisplayId
  );

  if (!target) {
    return res.status(404).json({ error: 'Session not found' });
  }

  if (target.isCurrent) {
    return res.status(400).json({ error: 'Cannot revoke current session' });
  }

  // 実際のセッション ID で削除（表示用 ID ではない）
  // セキュリティ上、内部のマッピングが必要
  await revokeSessionByDisplayId(session.userId, req.params.sessionDisplayId);

  res.json({ success: true });
});

// 他の全デバイスからログアウト
app.post('/api/sessions/revoke-others', async (req, res) => {
  const sessionId = req.cookies['__Host-session_id'];
  const session = await sessionManager.get(sessionId);
  if (!session) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  const count = await sessionManager.destroyOtherSessions(
    session.userId,
    sessionId
  );

  // 監査ログ
  await logAuthEvent({
    type: 'revoke_other_sessions',
    userId: session.userId,
    revokedCount: count,
    ip: getClientIP(req),
  });

  res.json({ success: true, revokedCount: count });
});
```

---

## 10. ログアウトの実装

```
ログアウトの種類:

  ① 単一ログアウト:
     → 現在のセッションのみ破棄
     → 他のデバイスのセッションは維持

  ② グローバルログアウト:
     → ユーザーの全セッションを破棄
     → パスワード変更時、セキュリティインシデント時

  ③ フェデレーテッドログアウト:
     → SSO 環境で IdP にもログアウト通知
     → SAML SLO（Single Logout）
     → OIDC RP-Initiated Logout

  ログアウトのチェックリスト:
    ✓ サーバー側のセッションデータを削除
    ✓ セッション Cookie を無効化（Max-Age=0）
    ✓ Remember Me トークンを無効化
    ✓ CSRF トークンを無効化
    ✓ クライアント側のキャッシュをクリア（Cache-Control ヘッダー）
    ✓ WebSocket 接続を切断
    ✓ 監査ログを記録
```

```typescript
// 完全なログアウト実装
app.post('/api/auth/logout', async (req, res) => {
  const sessionId = req.cookies['__Host-session_id'];

  if (sessionId) {
    const session = await sessionManager.get(sessionId);

    if (session) {
      // 監査ログ
      await logAuthEvent({
        type: 'logout',
        userId: session.userId,
        ip: getClientIP(req),
        userAgent: req.headers['user-agent'] || '',
      });

      // セッション破棄
      await sessionManager.destroy(sessionId);
    }
  }

  // Remember Me トークンの無効化
  const rememberCookie = req.cookies['__Secure-remember_me'];
  if (rememberCookie) {
    const [series] = rememberCookie.split(':');
    if (series) {
      await rememberMeManager.revoke(series);
    }
  }

  // 全 Cookie の無効化
  const cookieOptions = {
    httpOnly: true,
    secure: true,
    sameSite: 'lax' as const,
    path: '/',
    maxAge: 0,  // 即座に失効
  };

  res.cookie('__Host-session_id', '', cookieOptions);
  res.cookie('__Secure-remember_me', '', cookieOptions);
  res.cookie('csrf_token', '', { ...cookieOptions, httpOnly: false });

  // キャッシュ制御（ログアウト後にブラウザバックで認証済みページが表示されることを防止）
  res.set({
    'Cache-Control': 'no-store, no-cache, must-revalidate',
    'Pragma': 'no-cache',
    'Clear-Site-Data': '"cache", "cookies", "storage"', // モダンブラウザ向け
  });

  res.json({ success: true, redirectTo: '/login' });
});

// グローバルログアウト（全デバイス）
app.post('/api/auth/logout-all', async (req, res) => {
  const sessionId = req.cookies['__Host-session_id'];
  const session = await sessionManager.get(sessionId);

  if (!session) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  // パスワード確認（セキュリティ強化）
  const { password } = req.body;
  const user = await db.user.findUnique({ where: { id: session.userId } });
  if (!user || !await verifyPassword(password, user.passwordHash)) {
    return res.status(403).json({ error: 'Password verification failed' });
  }

  // 全セッション破棄
  const count = await sessionManager.destroyAllForUser(session.userId);

  // 全 Remember Me トークン無効化
  await rememberMeManager.revokeAllForUser(session.userId);

  // 監査ログ
  await logAuthEvent({
    type: 'global_logout',
    userId: session.userId,
    revokedCount: count,
    ip: getClientIP(req),
  });

  // Cookie 無効化
  res.cookie('__Host-session_id', '', { maxAge: 0, path: '/' });
  res.cookie('__Secure-remember_me', '', { maxAge: 0, path: '/' });
  res.set('Clear-Site-Data', '"cache", "cookies", "storage"');

  res.json({ success: true, revokedCount: count, redirectTo: '/login' });
});
```

---

## 11. セッション vs JWT の比較

```
セッションベース認証 vs JWT ベース認証:

  ┌──────────────────┬──────────────────────┬──────────────────────┐
  │ 項目             │ セッション             │ JWT                  │
  ├──────────────────┼──────────────────────┼──────────────────────┤
  │ 状態管理         │ サーバー側（Stateful） │ クライアント（Stateless）│
  │ ストレージ       │ Redis/DB が必要       │ 不要                  │
  │ 即時無効化       │ ✓ ストアから削除       │ ✗ 期限まで有効         │
  │ スケーラビリティ │ ストアが必要           │ 高い（ストア不要）     │
  │ セキュリティ     │ Cookie（HttpOnly）    │ localStorage or Cookie│
  │ ペイロード       │ サーバー側で管理       │ トークンに含まれる     │
  │ CSRF 脆弱性     │ あり（Cookie 自動送信）│ なし（手動付与の場合） │
  │ XSS 脆弱性      │ 低い（HttpOnly Cookie）│ 高い（localStorage）  │
  │ マイクロサービス │ 共有ストアが必要       │ 各サービスで検証可能   │
  │ モバイル対応     │ Cookie サポートに依存  │ Bearer トークン        │
  │ 実装の複雑さ     │ シンプル              │ リフレッシュ等が複雑   │
  └──────────────────┴──────────────────────┴──────────────────────┘

  推奨:
    Web アプリ（SPA + サーバー）→ セッション + Cookie
    API（モバイル、マイクロサービス）→ JWT
    ハイブリッド → セッション（Web）+ JWT（API）

  なぜ Web アプリにはセッション + Cookie が推奨されるのか:
    1. HttpOnly Cookie で XSS からトークンを保護できる
    2. 即時無効化（ログアウト、アカウント停止）が可能
    3. サーバー側でセッションデータを完全に制御
    4. JWT のようなリフレッシュトークン管理が不要
    5. OWASP もこのパターンを推奨
```

---

## 12. パフォーマンス最適化

```
セッション管理のパフォーマンス考慮点:

  ① Redis レイテンシ:
     → 通常 <1ms（同一リージョン）
     → リクエストごとに GET + SET（TTL 更新）= 2回
     → パイプライニングで最適化可能

  ② セッションデータのサイズ:
     → 小さく保つ（<1KB 推奨）
     → ユーザー情報は DB から取得、セッションには ID のみ
     → 大きなデータはセッション外に保存

  ③ Cookie のサイズ:
     → Cookie ヘッダーの上限: 約4KB（ブラウザ依存）
     → セッション ID のみ保存（64文字 = 約64バイト）
     → Cookie が大きすぎると全リクエストに影響

  ④ セッションストアの選択:
     → Redis: 最速、TTL 自動管理、クラスタリング対応
     → PostgreSQL: 追加インフラ不要、ACID 保証
     → DynamoDB: サーバーレス、自動スケーリング
     → メモリ: 開発環境のみ（再起動で全消失）

  Redis パフォーマンスチューニング:
    → Connection Pooling: 接続の再利用
    → Pipeline: 複数コマンドのバッチ実行
    → Cluster: 大規模環境での水平分散
    → Sentinel: 高可用性（自動フェイルオーバー）
```

```typescript
// パフォーマンス最適化されたセッションミドルウェア
class OptimizedSessionMiddleware {
  private cache: Map<string, { data: SessionData; cachedAt: number }> = new Map();
  private readonly CACHE_TTL = 5000; // 5秒のインメモリキャッシュ

  async getSession(req: Request): Promise<SessionData | null> {
    const sessionId = req.cookies['__Host-session_id'];
    if (!sessionId) return null;

    // インメモリキャッシュをチェック（同一リクエスト内の重複取得を防止）
    const cached = this.cache.get(sessionId);
    if (cached && Date.now() - cached.cachedAt < this.CACHE_TTL) {
      return cached.data;
    }

    // Redis から取得
    const data = await sessionManager.get(sessionId);
    if (!data) return null;

    // キャッシュに保存
    this.cache.set(sessionId, { data, cachedAt: Date.now() });

    // キャッシュのクリーンアップ（メモリリーク防止）
    if (this.cache.size > 10000) {
      const now = Date.now();
      for (const [key, value] of this.cache) {
        if (now - value.cachedAt > this.CACHE_TTL) {
          this.cache.delete(key);
        }
      }
    }

    return data;
  }

  // TTL 更新の最適化（毎リクエストではなく間隔を空ける）
  async touchSession(sessionId: string, data: SessionData): Promise<void> {
    const lastAccess = data.lastAccessedAt;
    const now = Date.now();

    // 最後の更新から60秒以内は TTL 更新をスキップ
    if (now - lastAccess < 60 * 1000) {
      return;
    }

    data.lastAccessedAt = now;
    await sessionManager.store.set(sessionId, data, { ttl: 30 * 60 });
  }
}
```

---

## 13. セキュリティヘッダーとの組み合わせ

```typescript
// セッション管理に関連するセキュリティヘッダー
import helmet from 'helmet';

app.use(helmet({
  // Content Security Policy（XSS 防御）
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", 'data:', 'https:'],
      connectSrc: ["'self'"],
      frameSrc: ["'none'"],  // iframe 埋め込み禁止
      objectSrc: ["'none'"],
      upgradeInsecureRequests: [],
    },
  },

  // HSTS（HTTPS 強制）
  strictTransportSecurity: {
    maxAge: 31536000,        // 1年
    includeSubDomains: true,
    preload: true,
  },

  // X-Frame-Options（クリックジャッキング防御）
  frameguard: { action: 'deny' },

  // Referrer Policy（セッション ID の漏洩防止）
  referrerPolicy: { policy: 'strict-origin-when-cross-origin' },

  // X-Content-Type-Options
  noSniff: true,
}));

// Cache-Control（認証済みページのキャッシュ防止）
app.use((req, res, next) => {
  if (req.cookies['__Host-session_id']) {
    // 認証済みリクエストのレスポンスをキャッシュしない
    res.set({
      'Cache-Control': 'private, no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0',
    });
  }
  next();
});
```

---

## 14. アンチパターン

```
セッション管理のアンチパターン:

  ❌ アンチパターン1: セッション ID を URL に含める
     × https://example.com/dashboard?sid=abc123
     → Referer ヘッダーで漏洩
     → ブラウザ履歴に残る
     → ブックマークで共有される
     → アクセスログに記録される

  ❌ アンチパターン2: Math.random() でセッション ID 生成
     × const sid = Math.random().toString(36)
     → 予測可能（V8 の xorshift128+ は逆算可能）
     → エントロピーが不足（約52ビット）
     → ブルートフォースで推測可能

  ❌ アンチパターン3: セッションデータを Cookie に直接保存
     × Set-Cookie: user={"id":"123","role":"admin","email":"..."}
     → クライアントが改ざん可能（role を admin に変更）
     → Cookie サイズ制限（4KB）
     → 個人情報の漏洩リスク
     ○ 正しい方法: Cookie にはセッション ID のみ、データはサーバー側

  ❌ アンチパターン4: ログアウト時に Cookie のみ削除
     × res.clearCookie('session_id')  // サーバー側のセッションは残存
     → 攻撃者がセッション ID を知っていれば引き続きアクセス可能
     ○ 正しい方法: サーバー側のセッションデータも必ず削除

  ❌ アンチパターン5: セッション ID のローテーションを行わない
     × ログイン前後で同じセッション ID を使用
     → セッション固定攻撃に脆弱
     ○ 正しい方法: ログイン成功時に必ず新しいセッション ID を生成
```

---

## 15. エッジケース

```
セッション管理のエッジケース:

  ① Redis ダウン時の対応:
     → セッション取得が失敗 → 全ユーザーがログアウト状態
     → 対策: Redis Sentinel/Cluster で冗長化
     → フォールバック: DB から直接セッション取得（遅い）
     → グレースフルデグラデーション: 一時的に JWT を発行

  ② 時刻同期のずれ:
     → サーバー間で時刻がずれるとセッション有効期限が不正確
     → 対策: NTP で時刻同期、Redis の TTL（サーバー時刻に依存しない）

  ③ Cookie の同時更新競合:
     → 複数タブから同時にリクエスト → セッション ID ローテーションの競合
     → 旧セッション ID で後続リクエストが失敗
     → 対策: ローテーション後も旧 ID を短時間（30秒）有効に保持

  ④ ブラウザの Cookie 削除:
     → ユーザーが手動で Cookie を削除
     → Remember Me があれば自動復元
     → なければ再ログインが必要

  ⑤ Cookie のサイズ上限超過:
     → 多数の Cookie で合計4KBを超えるとブラウザが切り捨て
     → セッション Cookie が失われる可能性
     → 対策: 不要な Cookie を削減、セッション Cookie の優先度を確保

  ⑥ サードパーティ Cookie ブロック:
     → Safari ITP、Chrome の Cookie 制限
     → 自社ドメインのファーストパーティ Cookie は影響なし
     → iframe 内での認証は SameSite=None + CHIPS が必要
```

```typescript
// Redis ダウン時のフォールバック
class ResilientSessionManager {
  constructor(
    private primaryStore: RedisSessionStore,
    private fallbackStore: DatabaseSessionStore
  ) {}

  async get(sessionId: string): Promise<SessionData | null> {
    try {
      // まず Redis を試行
      const data = await this.primaryStore.get(sessionId);
      if (data) return data;
    } catch (error) {
      // Redis 接続エラー → フォールバック
      console.error('Redis unavailable, falling back to DB:', error);

      try {
        return await this.fallbackStore.get(sessionId);
      } catch (dbError) {
        console.error('Fallback DB also failed:', dbError);
        return null;
      }
    }

    return null;
  }

  // セッション ID ローテーション時の競合対策
  async rotateWithGracePeriod(
    oldSessionId: string,
    gracePeriodMs: number = 30000 // 30秒
  ): Promise<string> {
    const data = await this.primaryStore.get(oldSessionId);
    if (!data) throw new Error('Session not found');

    const newSessionId = crypto.randomBytes(32).toString('hex');

    // 新セッションを作成
    await this.primaryStore.set(newSessionId, data, { ttl: 30 * 60 });

    // 旧セッションを短い TTL で維持（猶予期間）
    // 新セッション ID へのポインターを設定
    await this.primaryStore.set(
      oldSessionId,
      { ...data, redirectTo: newSessionId } as any,
      { ttl: Math.ceil(gracePeriodMs / 1000) }
    );

    return newSessionId;
  }
}
```

---

## 16. 演習問題

### 演習1（基礎）: セッション管理の基本実装

```
課題:
  Express.js アプリケーションで以下を実装せよ:
  1. ログイン時のセッション作成（セッション ID ローテーション付き）
  2. 認証ミドルウェア（セッション検証）
  3. ログアウト（サーバー側セッション削除 + Cookie 無効化）
  4. Cookie 属性の適切な設定

検証ポイント:
  - crypto.randomBytes() でセッション ID を生成しているか
  - HttpOnly, Secure, SameSite が設定されているか
  - ログイン時にセッション ID が再生成されているか
  - ログアウト時にサーバー側のセッションも削除されているか
```

### 演習2（応用）: Remember Me とセッション管理 UI

```
課題:
  以下の機能を追加実装せよ:
  1. Remember Me 機能（30日間のトークン + ローテーション）
  2. アクティブセッション一覧 API
  3. 特定セッションの無効化 API
  4. 他の全デバイスからログアウト API

検証ポイント:
  - Remember Me トークンがハッシュ化されて DB に保存されているか
  - トークン使用時にローテーションが行われているか
  - トークン盗難検知（シリーズ一致・トークン不一致）が実装されているか
  - セッション一覧で完全なセッション ID が公開されていないか
```

### 演習3（発展）: 異常検知とセキュリティ強化

```
課題:
  セッション管理にセキュリティ強化機能を追加せよ:
  1. IP アドレス変更検知（Impossible Travel Detection）
  2. User-Agent 変更検知
  3. 並行セッション数制限（最大5セッション、古いものから無効化）
  4. CSRF 対策（Synchronizer Token Pattern）
  5. セキュリティイベントの監査ログ

検証ポイント:
  - Impossible Travel の速度計算が正しいか
  - User-Agent 変更時にセッションが無効化されるか
  - セッション数制限が正しく機能するか
  - CSRF トークンがタイミングセーフに比較されているか
  - 全セキュリティイベントがログに記録されているか
```

---

## 17. FAQ / トラブルシューティング

```
Q: Cookie が設定されない / 送信されない
A: 以下をチェック:
   1. Secure=true だが HTTP でアクセスしている → HTTPS を使用
   2. SameSite=None だが Secure がない → Secure を追加
   3. Domain 属性がリクエスト先のドメインと一致しない
   4. Path 属性がリクエストパスと一致しない
   5. Max-Age=0 で即座に失効している
   6. ブラウザのサードパーティ Cookie ブロックが有効
   7. 開発ツール > Application > Cookies で確認

Q: セッションが頻繁に切れる
A: 以下をチェック:
   1. Redis の maxmemory-policy が allkeys-lru → セッションが追い出される
      → volatile-lru に変更（TTL 付きキーのみ追い出し）
   2. Redis のメモリ不足 → メモリ増設 or セッションデータの削減
   3. スライディング TTL の更新が動作していない
   4. 絶対有効期限が短すぎる
   5. ロードバランサが異なるサーバーにルーティング
      → Redis を共有ストアとして使用（Sticky Session は非推奨）

Q: SameSite=Lax でもまだ CSRF が心配
A: SameSite=Lax は GET リクエストの CSRF を防げない:
   → GET でデータ変更を行わない（REST の原則に従う）
   → 重要な操作には CSRF トークンを追加
   → 状態変更は必ず POST/PUT/DELETE で行う

Q: セッション ID のローテーション後に Ajax リクエストが失敗する
A: 複数タブ問題:
   → タブ A でローテーション → 新しい Cookie が設定される
   → タブ B はまだ古い Cookie を保持 → リクエスト失敗
   → 対策: ローテーション後に旧 ID を30秒間有効に保持
   → または: 401 レスポンス時にフロントでリロードを促す

Q: Cookie の HttpOnly を設定すると CSRF トークンが取得できない
A: CSRF トークンの取得方法:
   → <meta name="csrf-token"> でHTMLに埋め込む
   → 専用の GET エンドポイント（/api/csrf-token）を用意
   → Double Submit Cookie パターン（HttpOnly=false の別 Cookie）
   → セッション Cookie 自体は HttpOnly のまま維持

Q: JWT とセッションのどちらを使うべきか
A: 判断基準:
   → Web アプリのみ → セッション + Cookie（推奨）
   → モバイルアプリ → JWT
   → マイクロサービス間 → JWT
   → 即時無効化が必要 → セッション
   → サーバーレス → JWT（ストア不要）
   → ハイブリッド → Web はセッション、API は JWT
```

---

## まとめ

| 項目 | ベストプラクティス |
|------|--------------------|
| Cookie 属性 | HttpOnly + Secure + SameSite=Lax + __Host- prefix |
| セッション ID | crypto.randomBytes(32) = 256ビット |
| ローテーション | ログイン時・権限変更時に ID 再生成 |
| 有効期限 | スライディング(30分) + 絶対(24時間) |
| Remember Me | 別トークン、ハッシュ保存、ローテーション付き |
| ログアウト | ストア削除 + Cookie 無効化 + Clear-Site-Data |
| CSRF 対策 | SameSite=Lax + CSRF トークン（多層防御） |
| 異常検知 | IP/UA 変更検知、Impossible Travel |
| セッション管理 | 一覧表示、個別/全体無効化 |
| パフォーマンス | Redis + Connection Pool + TTL 更新最適化 |

---

## 次に読むべきガイド

- [[01-session-store.md]] — セッションストア（Redis, DB）の詳細実装
- [[../02-token-auth/00-jwt-basics.md]] — JWT ベース認証との比較と使い分け
- [[../04-implementation/02-email-password-auth.md]] — メール・パスワード認証の実装
- [[../03-authorization/01-abac-and-policies.md]] — セッションベースの認可ポリシー

---

## 参考文献

1. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. RFC 6265. "HTTP State Management Mechanism." IETF, 2011.
3. RFC 6265bis. "Cookies: HTTP State Management Mechanism." IETF, 2024. (Draft)
4. MDN. "Set-Cookie." developer.mozilla.org, 2024.
5. MDN. "Using HTTP cookies." developer.mozilla.org, 2024.
6. OWASP. "Cross-Site Request Forgery Prevention Cheat Sheet." cheatsheetseries.owasp.org, 2024.
7. OWASP. "Session Management Testing." owasp.org/www-project-web-security-testing-guide, 2024.
8. NIST SP 800-63B. "Digital Identity Guidelines: Authentication and Lifecycle Management." NIST, 2017.
9. Chrome Developers. "SameSite cookies explained." web.dev, 2024.
10. Chrome Developers. "Cookies Having Independent Partitioned State (CHIPS)." developer.chrome.com, 2024.
