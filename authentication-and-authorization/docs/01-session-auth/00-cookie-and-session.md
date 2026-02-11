# Cookie とセッション管理

> Cookie はブラウザとサーバー間の状態管理の基盤。Cookie の属性設計、セッション管理の仕組み、セキュアなセッション ID 生成、セッションライフサイクルまで、安全なセッション管理の全てを解説する。

## この章で学ぶこと

- [ ] Cookie の属性とセキュリティ設定を理解する
- [ ] セッションの作成・管理・破棄のライフサイクルを把握する
- [ ] セキュアなセッション管理を実装できるようになる

---

## 1. Cookie の基礎

```
Cookie の仕組み:

  サーバー → レスポンスヘッダーで設定:
    Set-Cookie: session_id=abc123; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=3600

  ブラウザ → リクエストヘッダーで自動送信:
    Cookie: session_id=abc123

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

SameSite の値:
  Strict: 同一サイトからのリクエストのみ Cookie 送信
          → 外部サイトからのリンクでもCookie送信されない
          → ユーザーが外部リンクからアクセスすると未ログイン状態
  Lax:    トップレベルナビゲーション（リンク遷移）は許可
          → GET リクエストのみ Cookie 送信
          → POST は送信されない → CSRF 防御
  None:   全てのクロスサイトリクエストで送信
          → Secure 属性が必須
          → サードパーティ Cookie（広告、埋め込み等）

  推奨: SameSite=Lax（デフォルト）
```

```typescript
// Cookie 設定のベストプラクティス
import { cookies } from 'next/headers';

async function setSessionCookie(sessionId: string) {
  const cookieStore = await cookies();

  cookieStore.set('session_id', sessionId, {
    httpOnly: true,        // JavaScript からアクセス不可
    secure: process.env.NODE_ENV === 'production',  // 本番は HTTPS のみ
    sameSite: 'lax',       // CSRF 防御
    path: '/',             // 全パスで有効
    maxAge: 24 * 60 * 60,  // 24時間
    // domain は省略（現在のドメインのみ）
  });
}

// Express での設定
app.use(session({
  name: 'sid',            // Cookie 名（デフォルトの 'connect.sid' から変更）
  secret: process.env.SESSION_SECRET,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 24 * 60 * 60 * 1000, // 24時間（ms）
  },
}));
```

---

## 2. セッション管理

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
```

```typescript
// セッション管理の実装
import crypto from 'crypto';

interface SessionData {
  userId: string;
  role: string;
  createdAt: number;
  lastAccessedAt: number;
  ipAddress: string;
  userAgent: string;
}

class SessionManager {
  constructor(private store: SessionStore) {}

  // セッション作成
  async create(userData: { userId: string; role: string }, req: Request): Promise<string> {
    // 暗号的に安全なセッション ID（32バイト = 256ビット）
    const sessionId = crypto.randomBytes(32).toString('hex');

    const sessionData: SessionData = {
      userId: userData.userId,
      role: userData.role,
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
      ipAddress: getClientIP(req),
      userAgent: req.headers.get('user-agent') || '',
    };

    await this.store.set(sessionId, sessionData, { ttl: 24 * 60 * 60 });

    return sessionId;
  }

  // セッション取得
  async get(sessionId: string): Promise<SessionData | null> {
    const data = await this.store.get(sessionId);
    if (!data) return null;

    // アクセス時間を更新（スライディング有効期限）
    data.lastAccessedAt = Date.now();
    await this.store.set(sessionId, data, { ttl: 24 * 60 * 60 });

    return data;
  }

  // セッション ID ローテーション（セッション固定攻撃対策）
  async rotate(oldSessionId: string): Promise<string> {
    const data = await this.store.get(oldSessionId);
    if (!data) throw new Error('Session not found');

    // 新しいセッション ID を生成
    const newSessionId = crypto.randomBytes(32).toString('hex');

    // 旧セッションを削除、新セッションを作成
    await this.store.delete(oldSessionId);
    await this.store.set(newSessionId, data, { ttl: 24 * 60 * 60 });

    return newSessionId;
  }

  // ログアウト
  async destroy(sessionId: string): Promise<void> {
    await this.store.delete(sessionId);
  }

  // 全セッション無効化（パスワード変更時等）
  async destroyAllForUser(userId: string): Promise<void> {
    const sessions = await this.store.findByUserId(userId);
    await Promise.all(sessions.map((s) => this.store.delete(s.id)));
  }

  // アクティブセッション一覧（ユーザーに表示）
  async getActiveSessions(userId: string): Promise<SessionInfo[]> {
    const sessions = await this.store.findByUserId(userId);
    return sessions.map((s) => ({
      id: s.id.substring(0, 8) + '...', // 完全な ID は非公開
      createdAt: new Date(s.data.createdAt),
      lastAccessedAt: new Date(s.data.lastAccessedAt),
      ipAddress: s.data.ipAddress,
      userAgent: parseUserAgent(s.data.userAgent),
      isCurrent: false, // 呼び出し時に判定
    }));
  }
}
```

---

## 3. セッション固定攻撃と対策

```
セッション固定攻撃（Session Fixation）:

  攻撃フロー:
    ① 攻撃者がサイトにアクセスしてセッション ID を取得
       session_id = "attacker_known_id"
    ② 攻撃者が被害者にリンクを送信
       https://example.com/?session_id=attacker_known_id
    ③ 被害者がリンクをクリックしてログイン
    ④ 被害者のセッション ID = attacker_known_id のまま
    ⑤ 攻撃者が同じセッション ID でアクセス → ログイン済み状態

  対策:
    → ログイン成功時にセッション ID を再生成（ローテーション）
    → URL パラメータからのセッション ID 受け入れを拒否
    → Cookie のみでセッション ID を管理
```

```typescript
// ログイン処理でのセッション ID ローテーション
async function handleLogin(email: string, password: string, req: Request, res: Response) {
  // 1. ユーザー認証
  const user = await authenticateUser(email, password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // 2. 既存セッションがあれば破棄
  const oldSessionId = req.cookies.session_id;
  if (oldSessionId) {
    await sessionManager.destroy(oldSessionId);
  }

  // 3. 新しいセッション ID で作成（セッション固定攻撃対策）
  const newSessionId = await sessionManager.create(
    { userId: user.id, role: user.role },
    req
  );

  // 4. Cookie に新しいセッション ID を設定
  res.cookie('session_id', newSessionId, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 24 * 60 * 60 * 1000,
  });

  return res.json({ user: { id: user.id, email: user.email } });
}
```

---

## 4. セッションの有効期限戦略

```
有効期限の種類:

  ① 絶対有効期限（Absolute Timeout）:
     → セッション作成からN時間後に失効
     → 例: 24時間後に自動ログアウト
     → セキュリティが厳しいシステム向け

  ② スライディング有効期限（Sliding/Idle Timeout）:
     → 最後のアクティビティからN分後に失効
     → アクティブなユーザーはセッション維持
     → 例: 30分操作なしで失効

  ③ ハイブリッド:
     → スライディング + 絶対有効期限の組合せ
     → 例: 操作があれば延長するが、最大72時間で失効

  推奨の組合せ:
    一般的な Web アプリ:
      → スライディング: 30分
      → 絶対: 24時間
      → 「ログイン状態を維持」: 30日（Remember Me）

    金融・医療:
      → スライディング: 15分
      → 絶対: 8時間（業務時間内）
      → 重要操作時に再認証要求

    ソーシャルメディア:
      → スライディング: 24時間
      → 絶対: 30日
      → 長期セッション（UX重視）
```

```typescript
// Remember Me 機能
async function loginWithRememberMe(
  userId: string,
  role: string,
  rememberMe: boolean,
  req: Request
) {
  const sessionId = await sessionManager.create({ userId, role }, req);

  if (rememberMe) {
    // Remember Me: 30日間の長期セッション
    const rememberToken = crypto.randomBytes(32).toString('hex');
    const hashedToken = crypto.createHash('sha256').update(rememberToken).digest('hex');

    await db.rememberToken.create({
      data: {
        userId,
        token: hashedToken,
        expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
      },
    });

    // 長期 Cookie
    setCookie('remember_me', rememberToken, {
      httpOnly: true,
      secure: true,
      sameSite: 'lax',
      maxAge: 30 * 24 * 60 * 60, // 30日
      path: '/',
    });
  }

  // 通常のセッション Cookie（ブラウザ閉じたら消える or 24時間）
  setCookie('session_id', sessionId, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: rememberMe ? 30 * 24 * 60 * 60 : undefined, // Remember Me 時のみ永続化
    path: '/',
  });
}

// セッション切れ時の Remember Me による自動復元
async function restoreSession(req: Request): Promise<SessionData | null> {
  // まずセッションを確認
  const sessionId = getCookie(req, 'session_id');
  if (sessionId) {
    const session = await sessionManager.get(sessionId);
    if (session) return session;
  }

  // セッション切れなら Remember Me トークンを確認
  const rememberToken = getCookie(req, 'remember_me');
  if (!rememberToken) return null;

  const hashedToken = crypto.createHash('sha256').update(rememberToken).digest('hex');
  const record = await db.rememberToken.findFirst({
    where: { token: hashedToken, expiresAt: { gt: new Date() } },
  });

  if (!record) return null;

  // 新しいセッションを作成
  const user = await db.user.findUnique({ where: { id: record.userId } });
  if (!user) return null;

  const newSessionId = await sessionManager.create(
    { userId: user.id, role: user.role },
    req
  );

  // トークンローテーション（使用したら新しいトークンを発行）
  await rotateRememberToken(record.id, record.userId);

  return sessionManager.get(newSessionId);
}
```

---

## 5. 並行セッション管理

```
並行セッションの制御:

  ① 無制限（デフォルト）:
     → 複数デバイスから同時ログイン可能
     → 一般的な Web アプリ

  ② 最大数制限:
     → 最大5デバイスまで等
     → 古いセッションから自動ログアウト

  ③ 単一セッション:
     → 1デバイスのみ
     → 金融系で多い
     → 新しいログインで旧セッション無効化

  ④ デバイスタイプ別:
     → PC: 1セッション、モバイル: 1セッション
     → 各デバイスタイプ1つずつ許可
```

---

## まとめ

| 項目 | ベストプラクティス |
|------|-----------------|
| Cookie 属性 | HttpOnly + Secure + SameSite=Lax |
| セッション ID | crypto.randomBytes(32) |
| ローテーション | ログイン時・権限変更時に ID 再生成 |
| 有効期限 | スライディング(30分) + 絶対(24時間) |
| Remember Me | 別トークン、ローテーション付き |
| ログアウト | ストア削除 + Cookie 無効化 |

---

## 次に読むべきガイド
→ [[01-session-store.md]] — セッションストア

---

## 参考文献
1. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. RFC 6265. "HTTP State Management Mechanism." IETF, 2011.
3. MDN. "Set-Cookie." developer.mozilla.org, 2024.
