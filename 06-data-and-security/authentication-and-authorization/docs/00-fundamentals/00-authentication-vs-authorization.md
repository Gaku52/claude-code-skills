# 認証と認可の基礎

> 認証（Authentication）は「あなたは誰か？」、認可（Authorization）は「あなたに何が許されているか？」。この2つの根本的な違いを理解し、脅威モデル、セキュリティ原則、認証フローの全体像を把握することが、安全なシステム構築の第一歩となる。

## この章で学ぶこと

- [ ] 認証と認可の違いを正確に理解する
- [ ] 主要な脅威モデルとセキュリティ原則を把握する
- [ ] 認証・認可の全体アーキテクチャを設計できるようになる
- [ ] RBAC / ABAC / ReBAC の認可モデルを比較・選択できるようになる
- [ ] セッション方式とトークン方式の内部動作を理解する
- [ ] 実運用で必要なセキュリティヘッダーとミドルウェアを実装できる

---

## 1. 認証と認可の違い

```
認証（Authentication / AuthN）:
  質問: 「あなたは誰ですか？」
  目的: ユーザーの身元を確認する
  例:   パスワード入力、指紋認証、Google ログイン

  ┌─────────────────────────────────────────┐
  │  ユーザー: 「私は alice@example.com です」  │
  │  システム: 「パスワードで証明してください」     │
  │  ユーザー: 「********」                    │
  │  システム: 「確認しました。alice さんですね」   │
  └─────────────────────────────────────────┘

認可（Authorization / AuthZ）:
  質問: 「あなたに何が許されていますか？」
  目的: ユーザーのアクセス権限を判定する
  例:   管理画面へのアクセス、ファイルの編集権限

  ┌─────────────────────────────────────────┐
  │  alice: 「管理画面にアクセスしたい」          │
  │  システム: 「alice の権限を確認中...」         │
  │  システム: 「admin ロール確認。アクセス許可」    │
  └─────────────────────────────────────────┘

重要な関係:
  認証 → 認可 の順序（必ず認証が先）
  認証なしに認可はできない
  認証されても認可されるとは限らない
```

```
比較表:

  項目       │ 認証（AuthN）      │ 認可（AuthZ）
  ───────────┼───────────────────┼──────────────────
  質問       │ Who are you?      │ What can you do?
  タイミング  │ 最初に実行          │ 認証後に実行
  入力       │ クレデンシャル       │ ユーザー属性・ロール
  出力       │ ユーザー ID         │ 許可 / 拒否
  失敗時     │ 401 Unauthorized   │ 403 Forbidden
  技術例     │ パスワード、JWT      │ RBAC、ABAC
  変更頻度   │ ユーザー主導        │ 管理者主導
  保存場所   │ DB / IdP           │ ポリシーエンジン
  キャッシュ  │ セッション / トークン │ 権限テーブル / ポリシー
  スケール   │ IdP に集約可能      │ サービスごとに分散可能
```

### 1.1 認証と認可の境界が曖昧になるケース

```
実務で混乱しやすいケース:

  ケース1: API キー
    → 認証? 認可? → 実は「両方」を兼ねることが多い
    → API キー自体がクライアントの身元証明（認証）
    → API キーに紐づくスコープで権限制御（認可）
    → 問題: キー漏洩時に認証・認可の両方が突破される

  ケース2: OAuth 2.0 のスコープ
    → OAuth は「認可の委譲」プロトコル
    → しかし OpenID Connect を加えると「認証」も行う
    → scope=openid が認証、scope=read:user が認可

  ケース3: JWT のクレーム
    → sub クレーム: 認証情報（誰か）
    → role / permissions クレーム: 認可情報（何ができるか）
    → 1つのトークンに認証・認可の両方が含まれる

  ケース4: IP アドレス制限
    → 「社内ネットワークからのみアクセス可能」
    → これは認証? 認可?
    → 厳密には認可（ネットワーク位置に基づくアクセス制御）
    → ただし暗黙的に「社内にいる = 社員である」という認証を含意
```

### 1.2 認証・認可パイプラインの内部動作

```
リクエスト処理パイプライン（詳細版）:

  HTTP リクエスト
    │
    ▼
  ┌──────────────────────┐
  │ ① TLS 終端            │  通信の暗号化を確認
  │    証明書検証           │  MITM 防止
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ ② レート制限           │  IP / ユーザー単位
  │    DDoS 防御           │  ブルートフォース防止
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ ③ CORS チェック        │  Origin ヘッダー検証
  │    プリフライト処理     │  ブラウザセキュリティ
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ ④ 認証（AuthN）        │  Cookie / Bearer Token
  │    身元確認            │  → ユーザー ID 特定
  │    セッション検証       │  → 401 or 続行
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ ⑤ 認可（AuthZ）        │  RBAC / ABAC チェック
  │    権限確認            │  → 403 or 続行
  │    リソースアクセス判定  │  → ポリシー評価
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ ⑥ 入力検証             │  バリデーション
  │    サニタイズ           │  XSS / SQLi 防止
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ ⑦ ビジネスロジック      │  実際の処理
  │    データアクセス       │  DB クエリ
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ ⑧ 監査ログ            │  誰が何をしたか記録
  │    レスポンス生成       │  セキュリティヘッダー付与
  └──────────────────────┘
```

```typescript
// 完全な認証・認可パイプライン実装（Express）
import express, { Request, Response, NextFunction } from 'express';

// ユーザー型定義
interface AuthenticatedUser {
  id: string;
  email: string;
  roles: string[];
  permissions: string[];
  sessionId: string;
  authenticatedAt: Date;
  mfaVerified: boolean;
}

// Request 拡張
declare global {
  namespace Express {
    interface Request {
      user?: AuthenticatedUser;
      requestId?: string;
    }
  }
}

// ① リクエストID付与（トレーサビリティ）
function requestIdMiddleware(req: Request, _res: Response, next: NextFunction) {
  req.requestId = req.headers['x-request-id'] as string
    || crypto.randomUUID();
  next();
}

// ② レート制限
const loginAttempts = new Map<string, { count: number; resetAt: number }>();

function rateLimitMiddleware(maxAttempts: number, windowMs: number) {
  return (req: Request, res: Response, next: NextFunction) => {
    const key = req.ip || 'unknown';
    const now = Date.now();
    const record = loginAttempts.get(key);

    if (record && record.resetAt > now) {
      if (record.count >= maxAttempts) {
        res.status(429).json({
          error: 'Too many requests',
          retryAfter: Math.ceil((record.resetAt - now) / 1000),
        });
        return;
      }
      record.count++;
    } else {
      loginAttempts.set(key, { count: 1, resetAt: now + windowMs });
    }

    next();
  };
}

// ④ 認証ミドルウェア（複数方式対応）
async function authenticationMiddleware(
  req: Request,
  res: Response,
  next: NextFunction,
) {
  // 方式1: Cookie セッション
  const sessionId = req.cookies?.session_id;
  if (sessionId) {
    const session = await sessionStore.get(sessionId);
    if (session && session.expiresAt > new Date()) {
      req.user = {
        id: session.userId,
        email: session.email,
        roles: session.roles,
        permissions: await resolvePermissions(session.roles),
        sessionId,
        authenticatedAt: session.authenticatedAt,
        mfaVerified: session.mfaVerified,
      };
      return next();
    }
  }

  // 方式2: Bearer トークン
  const authHeader = req.headers.authorization;
  if (authHeader?.startsWith('Bearer ')) {
    const token = authHeader.slice(7);
    try {
      const payload = await verifyJWT(token);
      req.user = {
        id: payload.sub,
        email: payload.email,
        roles: payload.roles || [],
        permissions: payload.permissions || [],
        sessionId: payload.jti,
        authenticatedAt: new Date(payload.iat * 1000),
        mfaVerified: payload.mfa_verified ?? false,
      };
      return next();
    } catch (err) {
      // トークン検証失敗 → 401
    }
  }

  // 方式3: API キー
  const apiKey = req.headers['x-api-key'] as string;
  if (apiKey) {
    const client = await apiKeyStore.verify(apiKey);
    if (client) {
      req.user = {
        id: client.id,
        email: client.contactEmail,
        roles: ['api_client'],
        permissions: client.scopes,
        sessionId: 'api-key',
        authenticatedAt: new Date(),
        mfaVerified: false,
      };
      return next();
    }
  }

  // 認証失敗
  res.status(401).json({
    error: 'Authentication required',
    message: 'Valid session, token, or API key is required',
  });
  res.setHeader('WWW-Authenticate', 'Bearer realm="api"');
}

// ⑤ 認可ミドルウェア（権限チェック）
function requirePermission(...requiredPermissions: string[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    const hasAll = requiredPermissions.every(
      perm => req.user!.permissions.includes(perm),
    );

    if (!hasAll) {
      // 監査ログに記録（認可失敗は重要なセキュリティイベント）
      auditLog.warn('authorization_denied', {
        userId: req.user.id,
        required: requiredPermissions,
        actual: req.user.permissions,
        resource: req.originalUrl,
        method: req.method,
        ip: req.ip,
        requestId: req.requestId,
      });

      return res.status(403).json({
        error: 'Forbidden',
        message: 'Insufficient permissions',
      });
    }

    next();
  };
}

// 使用例
const app = express();

app.use(requestIdMiddleware);

// 公開エンドポイント（認証不要）
app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));

// ログイン（レート制限あり、認証不要）
app.post('/api/auth/login',
  rateLimitMiddleware(5, 15 * 60 * 1000), // 15分に5回まで
  loginHandler,
);

// 保護エンドポイント（認証 + 認可）
app.get('/api/users',
  authenticationMiddleware,
  requirePermission('users:read'),
  listUsersHandler,
);

app.delete('/api/users/:id',
  authenticationMiddleware,
  requirePermission('users:delete'),
  deleteUserHandler,
);

// 管理者エンドポイント（認証 + 管理者認可 + MFA 必須）
app.post('/api/admin/settings',
  authenticationMiddleware,
  requireMFA,                              // MFA 検証済みか確認
  requirePermission('admin:settings:write'),
  updateSettingsHandler,
);
```

---

## 2. 認証の要素（Authentication Factors）

```
3つの認証要素:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  ① Something You Know（知識要素）              │
  │     → パスワード、PIN、秘密の質問               │
  │     → 最も一般的だが最も脆弱                    │
  │                                              │
  │  ② Something You Have（所有要素）              │
  │     → スマートフォン、ハードウェアキー、ICカード   │
  │     → TOTP、SMS コード、FIDO2 セキュリティキー   │
  │                                              │
  │  ③ Something You Are（生体要素）               │
  │     → 指紋、顔認証、虹彩、声紋                  │
  │     → 変更不可（漏洩時のリスクが高い）            │
  │                                              │
  └──────────────────────────────────────────────┘

多要素認証（MFA）:
  2つ以上の異なる要素を組み合わせる
  例: パスワード（知識）+ TOTP（所有）= 2FA

  ✗ パスワード + 秘密の質問 = 単一要素（両方「知識」）
  ✓ パスワード + TOTP = 多要素（「知識」+「所有」）
  ✓ パスワード + 指紋 = 多要素（「知識」+「生体」）

強度の順:
  パスワードのみ < パスワード + SMS < パスワード + TOTP < パスワード + FIDO2
```

### 2.1 認証要素の詳細比較

```
認証要素の強度・利便性比較:

  方式            │ 強度 │ 利便性 │ フィッシング │ リモート攻撃 │ コスト
  ───────────────┼─────┼──────┼───────────┼───────────┼──────
  パスワードのみ    │ ★☆☆ │ ★★★  │ 脆弱       │ 脆弱       │ 低
  パスワード+SMS   │ ★★☆ │ ★★☆  │ やや脆弱    │ SIM swap   │ 中
  パスワード+TOTP  │ ★★★ │ ★★☆  │ やや脆弱    │ 耐性あり    │ 低
  パスワード+Push  │ ★★★ │ ★★★  │ やや脆弱    │ MFA疲労    │ 中
  Passkeys        │ ★★★ │ ★★★  │ 耐性あり    │ 耐性あり    │ 低
  FIDO2 HW Key   │ ★★★ │ ★★☆  │ 耐性あり    │ 耐性あり    │ 高

  各攻撃への耐性:
  ───────────────────────────────────────────────────────
  攻撃手法         │ パスワード │ SMS  │ TOTP │ Push │ FIDO2
  ───────────────┼─────────┼─────┼─────┼─────┼──────
  ブルートフォース   │ ✗        │ △    │ △    │ ○    │ ○
  クレデンシャル     │ ✗        │ △    │ △    │ ○    │ ○
  　スタッフィング   │          │      │      │      │
  フィッシング      │ ✗        │ ✗    │ ✗    │ △    │ ○
  SIM スワップ     │ -        │ ✗    │ ○    │ ○    │ ○
  MITM プロキシ    │ ✗        │ ✗    │ ✗    │ △    │ ○
  MFA 疲労攻撃    │ -        │ -    │ -    │ ✗    │ ○
  デバイス窃取     │ -        │ ✗    │ ✗    │ ✗    │ △

  ○ = 耐性あり  △ = 条件次第  ✗ = 脆弱  - = 該当なし
```

### 2.2 NIST SP 800-63B の認証保証レベル（AAL）

```
NIST が定義する3つの認証保証レベル:

  AAL1（低保証）:
    → 単一要素認証で十分
    → パスワードのみ、または生体認証のみ
    → 用途: 一般的な Web サービス、SNS

  AAL2（中保証）:
    → 多要素認証（MFA）が必須
    → 2つの異なる要素の組み合わせ
    → 用途: オンラインバンキング、企業システム
    → 要件: フィッシング耐性は不要だが推奨

  AAL3（高保証）:
    → ハードウェアベースの暗号認証が必須
    → FIDO2 セキュリティキー等
    → フィッシング耐性が必須
    → 用途: 政府システム、医療、金融の高リスク取引
    → 要件: Verifier impersonation resistance

  レベル選択の指針:

    リスク              │ 推奨 AAL
    ──────────────────┼──────────
    個人的な不便        │ AAL1
    金銭的損害（小）     │ AAL2
    金銭的損害（大）     │ AAL2 or AAL3
    個人情報漏洩        │ AAL2
    機密情報漏洩        │ AAL3
    人命に関わる        │ AAL3
```

---

## 3. 脅威モデル

```
主要な認証への脅威:

  ┌─────────────────────┬─────────────────────────────────┐
  │ 脅威                 │ 説明                            │
  ├─────────────────────┼─────────────────────────────────┤
  │ ブルートフォース      │ パスワードの総当たり攻撃           │
  │ クレデンシャル        │ 漏洩したパスワードリストで         │
  │   スタッフィング      │ 他サービスにログイン試行           │
  │ フィッシング         │ 偽サイトでクレデンシャル詐取        │
  │ セッションハイジャック │ セッションIDの窃取                │
  │ CSRF               │ 認証済みユーザーの操作を偽造        │
  │ XSS → トークン窃取  │ スクリプト注入でトークン読取        │
  │ 中間者攻撃（MITM）   │ 通信の傍受・改ざん                │
  │ リプレイ攻撃         │ 正規のリクエストを再送             │
  │ 権限昇格            │ 一般ユーザーが管理者権限を取得      │
  └─────────────────────┴─────────────────────────────────┘
```

### 3.1 攻撃フローの詳細分析

```
攻撃1: クレデンシャルスタッフィング

  ① 攻撃者がサービスAの漏洩データを入手
  ② alice@example.com / password123 を取得
  ③ サービスB, C, D... に同じ認証情報でログイン試行
  ④ パスワード使い回しのユーザーがアカウント侵害される

  対策:
  → パスワード漏洩チェック（Have I Been Pwned API）
  → レート制限（ログイン試行回数制限）
  → MFA の強制
  → アカウントロックアウト

攻撃2: セッション固定攻撃（Session Fixation）

  攻撃者           被害者          サーバー
    │                │               │
    │ セッションID取得  │               │
    │───────────────────────────────>│
    │ SID=abc123      │               │
    │<───────────────────────────────│
    │                │               │
    │ SID=abc123 を    │               │
    │ 被害者に注入     │               │
    │───────────────>│               │
    │                │ ログイン        │
    │                │ (SID=abc123)   │
    │                │──────────────>│
    │                │ 認証成功        │
    │                │<──────────────│
    │                │               │
    │ SID=abc123 で    │               │
    │ アクセス（認証済み）│              │
    │───────────────────────────────>│
    │ 被害者のデータ    │               │
    │<───────────────────────────────│

  対策:
  → ログイン成功時にセッションIDを再生成
  → セッションIDをURLパラメータに含めない
  → Cookie の Secure, HttpOnly, SameSite 属性
```

```
攻撃3: CSRF（Cross-Site Request Forgery）

  被害者（認証済み）   悪意あるサイト     正規サーバー
    │                  │                │
    │ 悪意サイト訪問     │                │
    │────────────────>│                │
    │                  │                │
    │ 不正リクエスト     │                │
    │ （被害者のCookie   │                │
    │  が自動送信）      │                │
    │─────────────────────────────────>│
    │                  │                │
    │                  │  送金完了       │
    │<─────────────────────────────────│

  悪意あるサイトの HTML:
  <form action="https://bank.com/transfer" method="POST">
    <input type="hidden" name="to" value="attacker">
    <input type="hidden" name="amount" value="1000000">
  </form>
  <script>document.forms[0].submit();</script>

  対策:
  → SameSite Cookie 属性（Lax or Strict）
  → CSRF トークン（Synchronizer Token Pattern）
  → カスタムヘッダー検証（X-Requested-With）
  → Origin / Referer ヘッダー検証

攻撃4: 権限昇格（Privilege Escalation）

  水平権限昇格:
    → 同じ権限レベルの他ユーザーのリソースにアクセス
    → 例: /api/users/123/profile → /api/users/456/profile
    → 対策: オブジェクトレベル認可（リソースの所有者チェック）

  垂直権限昇格:
    → より高い権限レベルの機能にアクセス
    → 例: 一般ユーザーが /api/admin/users にアクセス
    → 対策: ロールベース認可チェック

  コンテキスト依存の権限昇格:
    → ビジネスフローの順序を無視
    → 例: 支払い前に商品ダウンロードリンクにアクセス
    → 対策: ステート管理 + フロー検証
```

```typescript
// CSRF 防御の実装
import crypto from 'crypto';

class CSRFProtection {
  private secret: string;

  constructor(secret: string) {
    this.secret = secret;
  }

  // トークン生成
  generateToken(sessionId: string): string {
    const timestamp = Date.now().toString(36);
    const random = crypto.randomBytes(16).toString('hex');
    const data = `${sessionId}:${timestamp}:${random}`;
    const hmac = crypto
      .createHmac('sha256', this.secret)
      .update(data)
      .digest('hex');
    return `${data}:${hmac}`;
  }

  // トークン検証
  verifyToken(token: string, sessionId: string): boolean {
    const parts = token.split(':');
    if (parts.length !== 4) return false;

    const [storedSessionId, timestamp, random, providedHmac] = parts;

    // セッションID一致確認
    if (storedSessionId !== sessionId) return false;

    // 有効期限チェック（1時間）
    const tokenTime = parseInt(timestamp, 36);
    if (Date.now() - tokenTime > 3600 * 1000) return false;

    // HMAC 検証（タイミングセーフ比較）
    const data = `${storedSessionId}:${timestamp}:${random}`;
    const expectedHmac = crypto
      .createHmac('sha256', this.secret)
      .update(data)
      .digest('hex');

    return crypto.timingSafeEqual(
      Buffer.from(providedHmac, 'hex'),
      Buffer.from(expectedHmac, 'hex'),
    );
  }
}

// ミドルウェアとして使用
function csrfMiddleware(csrf: CSRFProtection) {
  return (req: Request, res: Response, next: NextFunction) => {
    // GET, HEAD, OPTIONS は CSRF チェック不要
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      // レスポンスにトークンをセット
      const token = csrf.generateToken(req.cookies.session_id);
      res.cookie('csrf_token', token, {
        httpOnly: false,  // JavaScript から読み取り可能にする
        secure: true,
        sameSite: 'strict',
      });
      return next();
    }

    // 変更系リクエストはトークン検証
    const token = req.headers['x-csrf-token'] as string
      || req.body?._csrf;

    if (!token || !csrf.verifyToken(token, req.cookies.session_id)) {
      return res.status(403).json({
        error: 'Invalid CSRF token',
        message: 'Request rejected due to missing or invalid CSRF token',
      });
    }

    next();
  };
}
```

### 3.2 STRIDE 脅威モデリング

```
STRIDE フレームワーク（Microsoft）:

  脅威カテゴリ                     │ 対策
  ────────────────────────────────┼──────────────────────
  S: Spoofing（なりすまし）          │ 認証
    → 他のユーザーになりすます       │ MFA、Passkeys
                                   │
  T: Tampering（改ざん）            │ 完全性検証
    → データや通信を改ざん          │ HMAC、デジタル署名
                                   │
  R: Repudiation（否認）            │ 監査ログ
    → 「自分はやっていない」と主張   │ タイムスタンプ、署名
                                   │
  I: Information Disclosure        │ 暗号化
    （情報漏洩）                    │ アクセス制御
    → 機密情報の漏洩               │ TLS、暗号化
                                   │
  D: Denial of Service             │ レート制限
    （サービス拒否）                │ キャパシティ管理
    → サービスを利用不能に          │ CDN、WAF
                                   │
  E: Elevation of Privilege        │ 認可
    （権限昇格）                    │ 最小権限の原則
    → より高い権限を取得           │ RBAC/ABAC

  認証・認可システムで特に重要:
  → S（なりすまし）: 認証の強度が直接防御になる
  → E（権限昇格）: 認可の設計が直接防御になる
  → T（改ざん）: トークンの署名検証が防御になる
```

---

## 4. セキュリティ原則

```
認証・認可のセキュリティ原則:

  ① 最小権限の原則（Principle of Least Privilege）:
     → 必要最小限の権限のみ付与
     → デフォルトは「拒否」
     → 管理者権限は必要な人にのみ
     → 時間制限付き権限昇格（Just-In-Time Access）

  ② 多層防御（Defense in Depth）:
     → 単一の防御に頼らない
     → ネットワーク + アプリ + DB の各層で防御
     → 1つ突破されても次の層で止める
     → 例: WAF → API Gateway → Middleware → DB RLS

  ③ フェイルセキュア（Fail Secure）:
     → エラー時は安全側に倒す
     → 認証エラー → アクセス拒否（許可ではなく）
     → 例外発生 → ログアウト状態に
     → DB接続断 → デフォルト拒否（キャッシュ許可しない）

  ④ セキュリティバイデフォルト:
     → 安全な設定をデフォルトに
     → Cookie: HttpOnly=true, Secure=true, SameSite=Lax
     → HTTPS 強制
     → 新規ユーザーは最小権限ロール

  ⑤ ゼロトラスト:
     → 「信頼しない、常に検証する」
     → 内部ネットワークも信頼しない
     → リクエストごとに認証・認可
     → マイクロセグメンテーション
```

### 4.1 多層防御の実装例

```
多層防御アーキテクチャ:

  インターネット
    │
    ▼
  ┌──────────────────────────────────┐
  │ Layer 1: ネットワーク層           │
  │ → WAF（Web Application Firewall） │
  │ → DDoS 防御（Cloudflare 等）      │
  │ → IP 制限、Geo-blocking          │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │ Layer 2: API Gateway             │
  │ → レート制限                      │
  │ → API キー検証                    │
  │ → リクエストサイズ制限             │
  │ → TLS 終端                       │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │ Layer 3: アプリケーション層        │
  │ → 認証ミドルウェア                │
  │ → 認可ミドルウェア                │
  │ → CSRF 防御                      │
  │ → 入力検証・サニタイズ             │
  └──────────────┬───────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │ Layer 4: データ層                 │
  │ → Row Level Security（RLS）      │
  │ → 暗号化（at rest / in transit） │
  │ → DB ユーザー権限分離             │
  │ → クエリパラメータ化              │
  └──────────────────────────────────┘
```

```typescript
// フェイルセキュアの実装パターン
class FailSecureAuthService {
  private tokenVerifier: TokenVerifier;
  private permissionCache: PermissionCache;
  private circuitBreaker: CircuitBreaker;

  async authenticate(token: string): Promise<AuthResult> {
    try {
      // トークン検証
      const payload = await this.tokenVerifier.verify(token);
      return { authenticated: true, user: payload };
    } catch (error) {
      // あらゆるエラーで「拒否」に倒す
      if (error instanceof TokenExpiredError) {
        return { authenticated: false, reason: 'token_expired' };
      }
      if (error instanceof InvalidSignatureError) {
        return { authenticated: false, reason: 'invalid_signature' };
      }
      // 未知のエラーも安全側に倒す
      console.error('Unexpected auth error:', error);
      return { authenticated: false, reason: 'internal_error' };
    }
  }

  async authorize(userId: string, resource: string, action: string): Promise<boolean> {
    try {
      // 権限サービスへの問い合わせ
      if (this.circuitBreaker.isOpen()) {
        // サーキットブレーカー発動時 → デフォルト拒否
        console.warn('Permission service circuit breaker open, denying access');
        return false;  // フェイルセキュア: 許可しない
      }

      const permissions = await this.permissionCache.getOrFetch(userId);
      return permissions.includes(`${resource}:${action}`);
    } catch (error) {
      // 権限確認でエラー → 拒否
      console.error('Authorization check failed:', error);
      return false;  // フェイルセキュア: 許可しない
    }
  }
}

// アンチパターン: フェイルオープン（絶対にやってはいけない）
// async authorize(userId, resource, action) {
//   try {
//     const permissions = await getPermissions(userId);
//     return permissions.includes(`${resource}:${action}`);
//   } catch (error) {
//     return true;  // ✗ エラー時に許可 = フェイルオープン
//   }
// }
```

---

## 5. 認証・認可の全体アーキテクチャ

```
一般的な認証フロー:

  ユーザー         フロントエンド       バックエンド        DB / IdP
    │                 │                  │                │
    │ ログイン画面     │                  │                │
    │────────────────>│                  │                │
    │                 │ POST /auth/login │                │
    │                 │────────────────>│                │
    │                 │                  │ パスワード検証   │
    │                 │                  │───────────────>│
    │                 │                  │ ユーザー情報     │
    │                 │                  │<───────────────│
    │                 │                  │                │
    │                 │  セッション or    │                │
    │                 │  トークン発行     │                │
    │                 │<────────────────│                │
    │ ログイン成功     │                  │                │
    │<────────────────│                  │                │
    │                 │                  │                │
    │ 保護リソース要求  │                  │                │
    │────────────────>│                  │                │
    │                 │ + Cookie/Bearer  │                │
    │                 │────────────────>│                │
    │                 │                  │ 認証チェック     │
    │                 │                  │ 認可チェック     │
    │                 │                  │───────────────>│
    │                 │  200 OK + データ  │                │
    │                 │<────────────────│                │
    │ データ表示       │                  │                │
    │<────────────────│                  │                │
```

### 5.1 セッション方式 vs トークン方式の内部動作

```
セッション方式の内部動作:

  クライアント               サーバー                 セッションストア
    │                        │                       │
    │ POST /login            │                       │
    │ {email, password}      │                       │
    │───────────────────────>│                       │
    │                        │                       │
    │                        │ パスワード検証          │
    │                        │ セッション作成          │
    │                        │ sid = crypto.randomUUID()
    │                        │                       │
    │                        │ SET sid → {           │
    │                        │   userId, roles,      │
    │                        │   expiresAt, ip,      │
    │                        │   userAgent           │
    │                        │ }                     │
    │                        │──────────────────────>│
    │                        │                       │
    │ Set-Cookie:            │                       │
    │ session_id=<sid>;      │                       │
    │ HttpOnly; Secure;      │                       │
    │ SameSite=Lax;          │                       │
    │ Max-Age=86400          │                       │
    │<───────────────────────│                       │
    │                        │                       │
    │ GET /api/data          │                       │
    │ Cookie: session_id=sid │                       │
    │───────────────────────>│                       │
    │                        │ GET sid               │
    │                        │──────────────────────>│
    │                        │ {userId, roles, ...}  │
    │                        │<──────────────────────│
    │                        │                       │
    │                        │ 有効期限チェック        │
    │                        │ IP / UA 検証          │
    │                        │ ロールで認可チェック    │
    │                        │                       │
    │ 200 OK + data          │                       │
    │<───────────────────────│                       │

  セッション方式の特徴:
    利点:
      → サーバー側で即座にセッション無効化可能
      → セッションデータの更新が容易
      → Cookie の HttpOnly で XSS からの保護
      → セッションデータのサイズに制限なし

    欠点:
      → サーバーにステート（セッションストア）が必要
      → スケーリング時にセッション共有が必要
        → Redis / Memcached 等の分散ストア
      → マイクロサービス間での共有が困難
      → モバイルアプリとの統合がやや面倒
```

```
トークン方式（JWT）の内部動作:

  クライアント               サーバー                 DB
    │                        │                       │
    │ POST /login            │                       │
    │ {email, password}      │                       │
    │───────────────────────>│                       │
    │                        │                       │
    │                        │ パスワード検証          │
    │                        │──────────────────────>│
    │                        │<──────────────────────│
    │                        │                       │
    │                        │ JWT 生成:             │
    │                        │ Header = {            │
    │                        │   alg: "RS256",       │
    │                        │   typ: "JWT"          │
    │                        │ }                     │
    │                        │ Payload = {           │
    │                        │   sub: "user123",     │
    │                        │   roles: ["admin"],   │
    │                        │   exp: 1700000000,    │
    │                        │   iat: 1699996400     │
    │                        │ }                     │
    │                        │ Signature = RS256(    │
    │                        │   header + payload,   │
    │                        │   privateKey          │
    │                        │ )                     │
    │                        │                       │
    │ {                      │                       │
    │   access_token: "ey..",│                       │
    │   refresh_token: "ey.."│                       │
    │   expires_in: 3600     │                       │
    │ }                      │                       │
    │<───────────────────────│                       │
    │                        │                       │
    │ GET /api/data          │                       │
    │ Authorization:         │                       │
    │ Bearer ey..            │                       │
    │───────────────────────>│                       │
    │                        │                       │
    │                        │ JWT 検証:             │
    │                        │ ① 署名検証（公開鍵）   │
    │                        │ ② exp チェック        │
    │                        │ ③ iss, aud チェック   │
    │                        │ ④ roles で認可        │
    │                        │ （DBアクセス不要!）     │
    │                        │                       │
    │ 200 OK + data          │                       │
    │<───────────────────────│                       │

  トークン方式の特徴:
    利点:
      → ステートレス（サーバーにセッション不要）
      → スケーリングが容易（どのサーバーでも検証可能）
      → マイクロサービスに最適
      → モバイル / SPA との統合が容易
      → 検証時に DB アクセス不要

    欠点:
      → トークン無効化が困難（ブラックリスト必要）
      → ペイロードサイズに実質制限（HTTP ヘッダー）
      → トークン漏洩時のリスクが高い
      → 有効期限まで権限変更が反映されない
```

```typescript
// セッション方式の実装
import { Redis } from 'ioredis';
import crypto from 'crypto';

class SessionManager {
  private redis: Redis;
  private defaultTTL: number = 24 * 60 * 60; // 24時間

  constructor(redis: Redis) {
    this.redis = redis;
  }

  // セッション作成
  async createSession(
    userId: string,
    metadata: { ip: string; userAgent: string; roles: string[] },
  ): Promise<string> {
    const sessionId = crypto.randomBytes(32).toString('hex');

    const sessionData = {
      userId,
      roles: metadata.roles,
      createdAt: Date.now(),
      lastAccessedAt: Date.now(),
      ip: metadata.ip,
      userAgent: metadata.userAgent,
    };

    await this.redis.setex(
      `session:${sessionId}`,
      this.defaultTTL,
      JSON.stringify(sessionData),
    );

    // ユーザーのアクティブセッション一覧にも記録
    await this.redis.sadd(`user_sessions:${userId}`, sessionId);

    return sessionId;
  }

  // セッション検証
  async getSession(sessionId: string): Promise<SessionData | null> {
    const raw = await this.redis.get(`session:${sessionId}`);
    if (!raw) return null;

    const session = JSON.parse(raw) as SessionData;

    // スライディングウィンドウ: アクセスのたびに有効期限を延長
    session.lastAccessedAt = Date.now();
    await this.redis.setex(
      `session:${sessionId}`,
      this.defaultTTL,
      JSON.stringify(session),
    );

    return session;
  }

  // セッション破棄
  async destroySession(sessionId: string): Promise<void> {
    const raw = await this.redis.get(`session:${sessionId}`);
    if (raw) {
      const session = JSON.parse(raw);
      await this.redis.srem(`user_sessions:${session.userId}`, sessionId);
    }
    await this.redis.del(`session:${sessionId}`);
  }

  // 特定ユーザーの全セッションを破棄（パスワード変更時等）
  async destroyAllSessions(userId: string): Promise<void> {
    const sessionIds = await this.redis.smembers(`user_sessions:${userId}`);
    if (sessionIds.length > 0) {
      const keys = sessionIds.map(id => `session:${id}`);
      await this.redis.del(...keys);
    }
    await this.redis.del(`user_sessions:${userId}`);
  }
}

// セッション固定攻撃対策
async function loginHandler(req: Request, res: Response) {
  const { email, password } = req.body;

  const user = await authenticateUser(email, password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // 重要: ログイン成功時に既存セッションを破棄して新規作成
  // → セッション固定攻撃の防止
  if (req.cookies.session_id) {
    await sessionManager.destroySession(req.cookies.session_id);
  }

  const newSessionId = await sessionManager.createSession(user.id, {
    ip: req.ip,
    userAgent: req.headers['user-agent'] || '',
    roles: user.roles,
  });

  res.cookie('session_id', newSessionId, {
    httpOnly: true,    // JavaScript からアクセス不可
    secure: true,      // HTTPS のみ
    sameSite: 'lax',   // CSRF 防御
    maxAge: 24 * 60 * 60 * 1000,  // 24時間
    path: '/',
  });

  res.json({ success: true, user: { id: user.id, email: user.email } });
}
```

### 5.2 セッション方式 vs トークン方式の選択基準

```
方式選択の判断基準:

  項目              │ セッション        │ JWT トークン
  ────────────────┼────────────────┼─────────────────
  ステート管理      │ サーバー側        │ クライアント側
  ストレージ        │ Redis 等必要      │ 不要
  スケーラビリティ  │ △ ストア共有必要  │ ○ ステートレス
  即時無効化        │ ○ 即座に可能      │ △ ブラックリスト要
  ペイロードサイズ  │ 制限なし          │ ~8KB（ヘッダー制限）
  DB アクセス       │ 毎リクエスト      │ 不要（検証のみ）
  マイクロサービス  │ △ 共有ストア      │ ○ 公開鍵配布のみ
  モバイル対応      │ △ Cookie 管理    │ ○ ヘッダー送信
  XSS 耐性        │ ○ HttpOnly       │ △ 保存場所に依存
  CSRF 耐性       │ △ 対策必要        │ ○ ヘッダー送信

  推奨パターン:

    Web アプリ（SSR）       → セッション + Cookie
    SPA + BFF             → セッション（BFF が Cookie 管理）
    SPA + API 直接        → JWT（短寿命 access + refresh）
    モバイルアプリ          → JWT
    マイクロサービス間      → JWT（Client Credentials）
    ハイブリッド            → セッション（Web）+ JWT（API）
```

---

## 6. 認可モデルの詳細

### 6.1 RBAC（Role-Based Access Control）

```
RBAC の仕組み:

  ユーザー ──has──> ロール ──has──> パーミッション

  例:
  alice ──has──> admin ──has──> users:read
                              users:write
                              users:delete
                              settings:read
                              settings:write

  bob   ──has──> editor ──has──> users:read
                                posts:read
                                posts:write

  carol ──has──> viewer ──has──> users:read
                                posts:read

  RBAC の階層（Hierarchical RBAC）:
  ┌────────┐
  │ admin  │ → すべての権限を包含
  ├────────┤
  │ editor │ → viewer の権限 + 編集権限
  ├────────┤
  │ viewer │ → 閲覧権限のみ
  └────────┘
```

```typescript
// RBAC 実装
interface Role {
  name: string;
  permissions: string[];
  inherits?: string[];  // 継承するロール
}

class RBACEngine {
  private roles: Map<string, Role> = new Map();

  registerRole(role: Role): void {
    this.roles.set(role.name, role);
  }

  // ロールの全権限を取得（継承を含む）
  getPermissions(roleName: string, visited = new Set<string>()): string[] {
    if (visited.has(roleName)) return []; // 循環防止
    visited.add(roleName);

    const role = this.roles.get(roleName);
    if (!role) return [];

    const permissions = new Set(role.permissions);

    // 親ロールの権限を継承
    for (const parent of role.inherits || []) {
      for (const perm of this.getPermissions(parent, visited)) {
        permissions.add(perm);
      }
    }

    return Array.from(permissions);
  }

  // 権限チェック
  hasPermission(userRoles: string[], permission: string): boolean {
    for (const roleName of userRoles) {
      const permissions = this.getPermissions(roleName);
      if (permissions.includes(permission)) return true;

      // ワイルドカード対応: "admin:*" は "admin:read" にマッチ
      for (const perm of permissions) {
        if (perm.endsWith(':*')) {
          const prefix = perm.slice(0, -1);
          if (permission.startsWith(prefix)) return true;
        }
      }
    }
    return false;
  }
}

// 使用例
const rbac = new RBACEngine();

rbac.registerRole({
  name: 'viewer',
  permissions: ['posts:read', 'users:read'],
});

rbac.registerRole({
  name: 'editor',
  permissions: ['posts:write', 'posts:delete'],
  inherits: ['viewer'],  // viewer の権限を継承
});

rbac.registerRole({
  name: 'admin',
  permissions: ['users:write', 'users:delete', 'settings:*'],
  inherits: ['editor'],  // editor（+ viewer）の権限を継承
});

// チェック
rbac.hasPermission(['editor'], 'posts:read');    // true（viewer から継承）
rbac.hasPermission(['editor'], 'users:delete');  // false
rbac.hasPermission(['admin'], 'settings:write'); // true（ワイルドカード）
```

### 6.2 ABAC（Attribute-Based Access Control）

```
ABAC の仕組み:

  ポリシー = 主体の属性 + リソースの属性 + 環境属性 + アクション

  例: 「平日の営業時間内に、東京オフィスの正社員が、
       自部署の機密文書を閲覧できる」

  主体（Subject）属性:
    → role: employee
    → department: engineering
    → office: tokyo
    → employment_type: full_time

  リソース（Resource）属性:
    → type: document
    → classification: confidential
    → department: engineering

  環境（Environment）属性:
    → time: 10:30 JST（営業時間内）
    → day: Monday（平日）
    → ip: 10.0.1.xxx（社内ネットワーク）

  アクション（Action）:
    → read

  ポリシー評価:
    主体.role == "employee"
    AND 主体.department == リソース.department
    AND 環境.time BETWEEN "09:00" AND "18:00"
    AND 環境.day IN ["Monday".."Friday"]
    → PERMIT
```

```typescript
// ABAC 実装
interface ABACContext {
  subject: Record<string, any>;     // ユーザー属性
  resource: Record<string, any>;    // リソース属性
  action: string;                   // アクション
  environment: Record<string, any>; // 環境属性
}

interface ABACPolicy {
  name: string;
  description: string;
  effect: 'permit' | 'deny';
  condition: (ctx: ABACContext) => boolean;
  priority: number;  // 数値が大きいほど優先
}

class ABACEngine {
  private policies: ABACPolicy[] = [];

  addPolicy(policy: ABACPolicy): void {
    this.policies.push(policy);
    // 優先度順にソート
    this.policies.sort((a, b) => b.priority - a.priority);
  }

  evaluate(context: ABACContext): { allowed: boolean; matchedPolicy?: string } {
    for (const policy of this.policies) {
      try {
        if (policy.condition(context)) {
          return {
            allowed: policy.effect === 'permit',
            matchedPolicy: policy.name,
          };
        }
      } catch {
        // ポリシー評価エラー → スキップ（フェイルセキュア）
        continue;
      }
    }

    // どのポリシーにもマッチしない → デフォルト拒否
    return { allowed: false, matchedPolicy: 'default_deny' };
  }
}

// 使用例
const abac = new ABACEngine();

// ポリシー: 自部署の機密文書は正社員のみ閲覧可能（営業時間内）
abac.addPolicy({
  name: 'confidential_doc_access',
  description: '正社員は営業時間内に自部署の機密文書を閲覧可能',
  effect: 'permit',
  priority: 10,
  condition: (ctx) => {
    const { subject, resource, action, environment } = ctx;
    return (
      action === 'read' &&
      resource.classification === 'confidential' &&
      subject.employment_type === 'full_time' &&
      subject.department === resource.department &&
      environment.hour >= 9 && environment.hour < 18 &&
      environment.dayOfWeek >= 1 && environment.dayOfWeek <= 5
    );
  },
});

// ポリシー: 管理者はすべてのリソースにアクセス可能
abac.addPolicy({
  name: 'admin_full_access',
  description: '管理者は全リソースにアクセス可能',
  effect: 'permit',
  priority: 100,  // 最高優先度
  condition: (ctx) => ctx.subject.role === 'admin',
});

// 評価
const result = abac.evaluate({
  subject: { role: 'employee', department: 'engineering', employment_type: 'full_time' },
  resource: { type: 'document', classification: 'confidential', department: 'engineering' },
  action: 'read',
  environment: { hour: 14, dayOfWeek: 3 },
});
// → { allowed: true, matchedPolicy: 'confidential_doc_access' }
```

### 6.3 認可モデルの比較

```
認可モデル比較表:

  項目          │ RBAC           │ ABAC            │ ReBAC
  ─────────────┼───────────────┼────────────────┼──────────────
  基本単位      │ ロール          │ 属性             │ 関係性
  柔軟性        │ 中             │ 高              │ 高
  複雑性        │ 低             │ 高              │ 中〜高
  パフォーマンス │ 高             │ 中              │ 中
  管理の容易さ  │ ○ 直感的       │ △ ポリシー複雑   │ ○ グラフで表現
  監査の容易さ  │ ○ ロール追跡   │ △ 条件分岐多い   │ ○ 関係追跡
  適用場面      │ 企業内アプリ    │ ヘルスケア、金融  │ SNS、ファイル共有
  代表ツール    │ Casbin         │ OPA/Rego        │ SpiceDB/Zanzibar

  具体例:
    RBAC: 「管理者はユーザー一覧を閲覧できる」
    ABAC: 「東京オフィスの正社員が営業時間内に機密文書を閲覧できる」
    ReBAC: 「このフォルダの owner はファイルを削除できる」
           「このファイルの viewer は閲覧できる」
           「親フォルダの editor は子ファイルの editor でもある」

  選択指針:
    → シンプルな権限管理 → RBAC
    → 条件付きの細かい制御 → ABAC
    → リソースの所有関係ベース → ReBAC
    → 複合 → RBAC + ABAC のハイブリッド（実務で最も多い）
```

---

## 7. 認証方式の全体像

```
認証方式の分類:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  サーバーサイド（ステートフル）                     │
  │  ├── セッション + Cookie                          │
  │  │   → サーバーにセッション状態を保持               │
  │  │   → Cookie でセッション ID を送信               │
  │  │   → 伝統的な Web アプリに最適                   │
  │  │                                               │
  │  クライアントサイド（ステートレス）                   │
  │  ├── JWT（JSON Web Token）                        │
  │  │   → トークンに情報を含む（自己完結型）            │
  │  │   → API 認証に最適                             │
  │  │                                               │
  │  委譲型（第三者認証）                               │
  │  ├── OAuth 2.0                                   │
  │  │   → 認可の委譲（第三者アプリへのアクセス許可）     │
  │  ├── OpenID Connect                              │
  │  │   → OAuth 2.0 上の認証レイヤー                  │
  │  ├── SAML                                        │
  │  │   → エンタープライズ SSO                        │
  │  │                                               │
  │  パスワードレス                                    │
  │  ├── Magic Link                                  │
  │  │   → メールでワンタイムリンクを送信                │
  │  ├── WebAuthn / Passkeys                         │
  │  │   → 公開鍵暗号による認証                         │
  │  │   → フィッシング耐性が最強                       │
  │  └── OTP                                         │
  │      → ワンタイムパスワード（SMS / メール）           │
  │                                                 │
  └─────────────────────────────────────────────────┘
```

### 7.1 各認証方式の詳細比較

```
認証方式の詳細比較:

  方式             │ セキュリティ │ UX    │ 実装コスト │ 運用コスト │ 適用
  ────────────────┼───────────┼──────┼─────────┼──────────┼──────
  パスワード        │ ★★☆       │ ★★★  │ ★☆☆      │ ★★★      │ 汎用
  セッション+Cookie │ ★★★       │ ★★★  │ ★★☆      │ ★★☆      │ Web
  JWT              │ ★★☆       │ ★★☆  │ ★★☆      │ ★☆☆      │ API
  OAuth 2.0/OIDC  │ ★★★       │ ★★★  │ ★★★      │ ★★☆      │ SSO
  SAML             │ ★★★       │ ★★☆  │ ★★★      │ ★★★      │ 企業
  Magic Link       │ ★★☆       │ ★★★  │ ★☆☆      │ ★★☆      │ B2C
  Passkeys         │ ★★★       │ ★★★  │ ★★★      │ ★☆☆      │ 次世代
  API Key          │ ★★☆       │ ★★★  │ ★☆☆      │ ★☆☆      │ M2M
  mTLS             │ ★★★       │ ★☆☆  │ ★★★      │ ★★★      │ 内部
```

```typescript
// マルチ認証プロバイダーの統一インターフェース
interface AuthProvider {
  name: string;
  authenticate(credentials: unknown): Promise<AuthResult>;
  supports(req: Request): boolean;
}

class AuthProviderChain {
  private providers: AuthProvider[] = [];

  register(provider: AuthProvider): void {
    this.providers.push(provider);
  }

  async authenticate(req: Request): Promise<AuthResult> {
    for (const provider of this.providers) {
      if (provider.supports(req)) {
        try {
          const result = await provider.authenticate(req);
          if (result.authenticated) {
            return result;
          }
        } catch (error) {
          // プロバイダーエラー → 次のプロバイダーを試行
          console.warn(`Auth provider ${provider.name} failed:`, error);
        }
      }
    }
    return { authenticated: false, reason: 'no_valid_credentials' };
  }
}

// セッションプロバイダー
class SessionAuthProvider implements AuthProvider {
  name = 'session';

  supports(req: Request): boolean {
    return !!req.cookies?.session_id;
  }

  async authenticate(req: Request): Promise<AuthResult> {
    const session = await sessionStore.get(req.cookies.session_id);
    if (!session || session.expiresAt < new Date()) {
      return { authenticated: false, reason: 'session_expired' };
    }
    return { authenticated: true, user: session.user };
  }
}

// JWT プロバイダー
class JWTAuthProvider implements AuthProvider {
  name = 'jwt';

  supports(req: Request): boolean {
    return !!req.headers.authorization?.startsWith('Bearer ');
  }

  async authenticate(req: Request): Promise<AuthResult> {
    const token = req.headers.authorization!.slice(7);
    const payload = await verifyJWT(token);
    return { authenticated: true, user: payload };
  }
}

// API キープロバイダー
class APIKeyAuthProvider implements AuthProvider {
  name = 'api_key';

  supports(req: Request): boolean {
    return !!req.headers['x-api-key'];
  }

  async authenticate(req: Request): Promise<AuthResult> {
    const key = req.headers['x-api-key'] as string;
    const client = await apiKeyStore.verify(key);
    if (!client) {
      return { authenticated: false, reason: 'invalid_api_key' };
    }
    return { authenticated: true, user: { id: client.id, roles: client.roles } };
  }
}

// 組み合わせ
const authChain = new AuthProviderChain();
authChain.register(new SessionAuthProvider());
authChain.register(new JWTAuthProvider());
authChain.register(new APIKeyAuthProvider());
```

---

## 8. HTTPステータスコードと認証・認可

```
認証・認可関連のHTTPステータスコード:

  401 Unauthorized（認証エラー）:
    → 認証が必要、または認証に失敗
    → WWW-Authenticate ヘッダーを返すべき
    → 例: トークン未送信、トークン期限切れ

  403 Forbidden（認可エラー）:
    → 認証済みだが権限がない
    → 再認証しても結果は変わらない
    → 例: 一般ユーザーが管理画面にアクセス

  407 Proxy Authentication Required:
    → プロキシ認証が必要
    → Proxy-Authenticate ヘッダー

よくある間違い:
  ✗ 未ログインで 403 を返す → 401 が正しい
  ✗ 権限不足で 401 を返す → 403 が正しい
  ✗ リソースの存在を隠す場合 → 404 を返す（情報漏洩防止）
```

### 8.1 ステータスコードの正確な使い分け

```
ステータスコード判定フローチャート:

  リクエスト受信
    │
    ├── 認証情報なし?
    │     └── YES → 401 Unauthorized
    │              + WWW-Authenticate ヘッダー
    │
    ├── 認証情報が無効?（期限切れ、署名不正等）
    │     └── YES → 401 Unauthorized
    │
    ├── 認証成功、だが権限不足?
    │     └── YES → 403 Forbidden
    │
    ├── リソースが存在しない?
    │     ├── ユーザーがリソースの存在を知るべき
    │     │     └── YES → 404 Not Found
    │     └── リソースの存在自体を隠すべき
    │           └── YES → 404 Not Found（403 ではなく）
    │                     ※ 情報漏洩防止
    │
    ├── メソッドが許可されていない?
    │     └── YES → 405 Method Not Allowed
    │
    └── すべて OK
          └── 200 OK / 201 Created / 204 No Content

  実務での複雑なケース:

    ケース: 管理者 API に一般ユーザーがアクセス
      → 403 が正解（認証済み、権限なし）
      → ただし API の存在を隠したい場合は 404

    ケース: 他ユーザーのリソースにアクセス
      → 403 が基本（アクセス権限なし）
      → ただし 404 の方がセキュア（リソースの存在を隠す）

    ケース: ログイン試行でユーザーが存在しない
      → 「ユーザーが存在しません」は NG（ユーザー列挙攻撃）
      → 「メールアドレスまたはパスワードが正しくありません」が正解
      → ステータスコードは 401（存在有無に関わらず同じ）
```

```typescript
// 正しいレスポンスの使い分け（完全版）
import { Request, Response } from 'express';

class AuthResponseHandler {
  // 認証エラー（401）
  static unauthorized(res: Response, scheme: string = 'Bearer'): Response {
    return res
      .status(401)
      .set('WWW-Authenticate', `${scheme} realm="api"`)
      .json({
        error: 'unauthorized',
        message: 'Authentication is required to access this resource',
        // エラーの詳細はセキュリティ上あいまいにする
      });
  }

  // 認可エラー（403）
  static forbidden(res: Response): Response {
    return res
      .status(403)
      .json({
        error: 'forbidden',
        message: 'You do not have permission to access this resource',
      });
  }

  // リソース不存在（404）- 存在を隠す目的でも使う
  static notFound(res: Response): Response {
    return res
      .status(404)
      .json({
        error: 'not_found',
        message: 'The requested resource was not found',
      });
  }

  // レート制限（429）
  static tooManyRequests(res: Response, retryAfter: number): Response {
    return res
      .status(429)
      .set('Retry-After', String(retryAfter))
      .json({
        error: 'too_many_requests',
        message: 'Rate limit exceeded',
        retryAfter,
      });
  }

  // 統合判定
  static handleAccessCheck(
    res: Response,
    user: User | null,
    resource: Resource | null,
    requiredPermission: string,
  ): Response | null {
    // 未認証
    if (!user) {
      return this.unauthorized(res);
    }

    // リソースが存在しない
    if (!resource) {
      return this.notFound(res);
    }

    // 権限チェック
    if (!hasPermission(user, resource, requiredPermission)) {
      // リソースの存在を隠すべきか?
      if (shouldHideResourceExistence(resource)) {
        return this.notFound(res); // 403 ではなく 404
      }
      return this.forbidden(res);
    }

    // アクセス許可 → null を返す（呼び出し元で続行）
    return null;
  }
}

// ユーザー列挙攻撃の防止
async function loginHandler(req: Request, res: Response) {
  const { email, password } = req.body;

  // ユーザーが存在しない場合もパスワード検証と同等の時間をかける
  const user = await findUserByEmail(email);

  if (!user) {
    // ダミーのハッシュ比較（タイミング攻撃防止）
    await bcrypt.compare(password, '$2b$12$dummy.hash.to.prevent.timing.attacks');
    return res.status(401).json({
      error: 'invalid_credentials',
      message: 'Invalid email or password',  // 曖昧なメッセージ
    });
  }

  const valid = await bcrypt.compare(password, user.passwordHash);
  if (!valid) {
    return res.status(401).json({
      error: 'invalid_credentials',
      message: 'Invalid email or password',  // 同じメッセージ
    });
  }

  // ログイン成功処理...
}
```

---

## 9. セキュリティヘッダーの実装

```
認証・認可に関連する重要なセキュリティヘッダー:

  ┌────────────────────────────────────────────────────────┐
  │ ヘッダー                          │ 目的              │
  ├────────────────────────────────────────────────────────┤
  │ Strict-Transport-Security          │ HTTPS 強制        │
  │ X-Content-Type-Options             │ MIME スニッフィング │
  │ X-Frame-Options                    │ クリックジャッキング │
  │ Content-Security-Policy            │ XSS / インジェクション │
  │ X-XSS-Protection                   │ XSS フィルター     │
  │ Referrer-Policy                    │ リファラー制御      │
  │ Permissions-Policy                 │ ブラウザ機能制限    │
  │ Cache-Control                      │ 認証データの       │
  │                                    │ キャッシュ防止      │
  └────────────────────────────────────────────────────────┘
```

```typescript
// セキュリティヘッダーミドルウェア
function securityHeaders(req: Request, res: Response, next: NextFunction) {
  // HTTPS 強制（1年間、サブドメイン含む）
  res.setHeader(
    'Strict-Transport-Security',
    'max-age=31536000; includeSubDomains; preload',
  );

  // MIME タイプスニッフィング防止
  res.setHeader('X-Content-Type-Options', 'nosniff');

  // クリックジャッキング防止
  res.setHeader('X-Frame-Options', 'DENY');

  // XSS 防御（CSP が主力だが追加防御として）
  res.setHeader('X-XSS-Protection', '1; mode=block');

  // Content Security Policy
  res.setHeader('Content-Security-Policy', [
    "default-src 'self'",
    "script-src 'self' 'nonce-${nonce}'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: https:",
    "connect-src 'self' https://api.example.com",
    "frame-ancestors 'none'",
    "form-action 'self'",
    "base-uri 'self'",
  ].join('; '));

  // リファラー制御
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

  // 認証済みレスポンスのキャッシュ防止
  if (req.user) {
    res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, private');
    res.setHeader('Pragma', 'no-cache');
  }

  next();
}
```

---

## 10. 監査ログ

```
監査ログの重要性:

  「誰が、いつ、何をしたか」を記録する
  → セキュリティインシデントの調査
  → コンプライアンス要件の充足
  → 不正アクセスの早期検知

  記録すべきイベント:
  ┌──────────────────────────┬────────────┐
  │ イベント                  │ 重要度      │
  ├──────────────────────────┼────────────┤
  │ ログイン成功              │ INFO       │
  │ ログイン失敗              │ WARN       │
  │ 連続ログイン失敗          │ ALERT      │
  │ パスワード変更            │ INFO       │
  │ MFA 有効化 / 無効化      │ INFO       │
  │ 権限変更                 │ WARN       │
  │ 管理者操作               │ INFO       │
  │ 認可拒否                 │ WARN       │
  │ トークン無効化            │ INFO       │
  │ 異常なアクセスパターン     │ ALERT      │
  │ 新しいデバイスからのログイン│ WARN       │
  └──────────────────────────┴────────────┘
```

```typescript
// 監査ログサービス
interface AuditLogEntry {
  timestamp: Date;
  eventType: string;
  severity: 'info' | 'warn' | 'alert' | 'critical';
  userId?: string;
  ip: string;
  userAgent: string;
  resource?: string;
  action?: string;
  result: 'success' | 'failure' | 'denied';
  metadata?: Record<string, unknown>;
  requestId: string;
}

class AuditLogger {
  private transport: AuditTransport;

  constructor(transport: AuditTransport) {
    this.transport = transport;
  }

  // 認証イベント記録
  async logAuthEvent(
    eventType: 'login' | 'logout' | 'login_failed' | 'mfa_verify' | 'password_change',
    req: Request,
    details: { userId?: string; reason?: string },
  ): Promise<void> {
    const entry: AuditLogEntry = {
      timestamp: new Date(),
      eventType: `auth.${eventType}`,
      severity: eventType.includes('failed') ? 'warn' : 'info',
      userId: details.userId || req.user?.id,
      ip: req.ip || 'unknown',
      userAgent: req.headers['user-agent'] || 'unknown',
      result: eventType.includes('failed') ? 'failure' : 'success',
      metadata: {
        reason: details.reason,
        sessionId: req.cookies?.session_id,
      },
      requestId: req.requestId || 'unknown',
    };

    // 連続失敗の検知
    if (eventType === 'login_failed') {
      await this.checkBruteForce(req.ip!, details.userId);
    }

    await this.transport.write(entry);
  }

  // 認可イベント記録
  async logAuthzEvent(
    req: Request,
    resource: string,
    action: string,
    result: 'success' | 'denied',
  ): Promise<void> {
    const entry: AuditLogEntry = {
      timestamp: new Date(),
      eventType: 'authz.check',
      severity: result === 'denied' ? 'warn' : 'info',
      userId: req.user?.id,
      ip: req.ip || 'unknown',
      userAgent: req.headers['user-agent'] || 'unknown',
      resource,
      action,
      result,
      requestId: req.requestId || 'unknown',
    };

    await this.transport.write(entry);
  }

  // ブルートフォース検知
  private async checkBruteForce(ip: string, userId?: string): Promise<void> {
    const key = userId ? `bf:user:${userId}` : `bf:ip:${ip}`;
    const count = await redis.incr(key);
    await redis.expire(key, 900); // 15分ウィンドウ

    if (count >= 10) {
      // アラート発火
      await this.transport.write({
        timestamp: new Date(),
        eventType: 'security.brute_force_detected',
        severity: 'alert',
        userId,
        ip,
        userAgent: '',
        result: 'failure',
        metadata: { failureCount: count, windowMinutes: 15 },
        requestId: 'system',
      });

      // 自動対策: IP / アカウントの一時ロック
      if (userId) {
        await redis.setex(`locked:user:${userId}`, 900, '1');
      }
      await redis.setex(`locked:ip:${ip}`, 900, '1');
    }
  }
}
```

---

## 11. アンチパターン

```
認証・認可のアンチパターン:

  ① フェイルオープン:
     ✗ エラー時にアクセスを許可する
     → 認可サービスダウン時に「全許可」
     → DB 接続エラー時に「認証スキップ」
     対策: エラー時は必ずアクセス拒否

  ② セキュリティ through Obscurity:
     ✗ 「この URL は誰も知らないから安全」
     → /admin-secret-panel-xyz は推測可能
     → API の隠しエンドポイントも発見される
     対策: すべてのエンドポイントに認証・認可を適用

  ③ クライアント側認可:
     ✗ フロントエンドだけで権限チェック
     → ボタンの非表示は認可ではない
     → JavaScript の条件分岐はバイパス可能
     対策: サーバー側で必ず認可チェック（UI は UX 向上用）

  ④ 過度な権限の付与:
     ✗ 「面倒だから全員 admin にしよう」
     → 1人の侵害 = システム全体の侵害
     対策: 最小権限の原則を徹底

  ⑤ ハードコードされたクレデンシャル:
     ✗ ソースコードに API キーやパスワードを直接記述
     → Git 履歴に残る
     → デプロイ環境ごとに変更できない
     対策: 環境変数 / Secrets Manager

  ⑥ 曖昧なエラーメッセージの欠如:
     ✗ 「このメールアドレスは登録されていません」
     → ユーザー列挙攻撃に利用される
     対策: 「メールアドレスまたはパスワードが正しくありません」

  ⑦ セッションの不適切な管理:
     ✗ ログアウト時にサーバー側セッションを破棄しない
     ✗ パスワード変更後に既存セッションを無効化しない
     ✗ セッション ID を URL パラメータに含める
     対策: 明示的なセッション破棄 + Cookie 属性の適切な設定

  ⑧ 認証と認可の混同:
     ✗ 認証されたら何でもできる前提の設計
     → 全ユーザーがすべての API にアクセス可能
     対策: 認証と認可を明確に分離
```

---

## 12. エッジケース

```
認証・認可のエッジケース:

  ① 権限の即時反映 vs パフォーマンス:
     → 管理者がユーザーの権限を変更した
     → JWT の場合、トークン有効期限まで旧権限が有効
     → セッションの場合、キャッシュしていると反映が遅延
     対策: 重要な権限変更は即時反映の仕組みを用意
           （ブラックリスト、セッション無効化、短寿命トークン）

  ② 並行セッション:
     → ユーザーが複数デバイスで同時ログイン
     → 1つのデバイスでパスワード変更 → 他のセッションは?
     → 金融系: 同時セッション禁止（最新のみ有効）
     → 一般: 全セッション無効化 + 再ログイン要求

  ③ タイムゾーンと有効期限:
     → JWT の exp はUTCで記録
     → サーバーとクライアントの時刻のずれ
     → 対策: サーバー側でのみ検証 + NTP 同期
     → 許容範囲: ±30秒のクロックスキュー

  ④ ロールの削除:
     → 「editor」ロールを廃止したい
     → 既存の「editor」ユーザーの権限はどうなる?
     → マイグレーション戦略が必要
     → ソフトデリート + 新ロールへの段階的移行

  ⑤ 管理者自身の権限削除:
     → 最後の管理者が自分の admin ロールを削除
     → システムに管理者が不在になる
     → 対策: 「最低1人の admin 必須」制約
```

---

## 13. 演習問題

```
演習1（基礎）: 認証・認可の判定

  以下の各シナリオに対して、適切な HTTP ステータスコードと
  レスポンスメッセージを回答せよ。

  シナリオ:
  a) API リクエストに Authorization ヘッダーがない
  b) JWT トークンの署名が不正
  c) 有効なトークンだが、一般ユーザーが /admin にアクセス
  d) ログインフォームで存在しないメールアドレスを入力
  e) 認証済みユーザーが他人のプロフィールにアクセス
     （リソースの存在を隠したい場合）

  期待する回答:
  → ステータスコード
  → レスポンスボディのメッセージ
  → 必要なヘッダー
```

```
演習2（応用）: RBAC エンジンの設計

  以下の要件を満たす RBAC エンジンを実装せよ。

  要件:
  1. ロールの階層構造（admin > editor > viewer）
  2. ワイルドカード権限（"posts:*" が "posts:read" にマッチ）
  3. 否定権限（"!users:delete" で特定操作を明示的に拒否）
  4. ロールの動的追加・削除
  5. 権限キャッシュ（Redis）

  テストケース:
  → admin が "users:delete" を持つことを確認
  → editor が "posts:read" を持つことを確認（継承）
  → editor に "!users:delete" があれば拒否を確認
```

```
演習3（発展）: ゼロトラスト API Gateway の実装

  以下の機能を持つ API Gateway を実装せよ。

  要件:
  1. 複数の認証方式をサポート（Session, JWT, API Key）
  2. リクエストごとに認証・認可を実行
  3. レート制限（IP + ユーザー単位）
  4. 監査ログの記録
  5. Circuit Breaker パターン（認可サービスダウン時の制御）
  6. リクエストの正規化（path traversal 防止等）

  テストシナリオ:
  → 正常な認証フロー（各方式）
  → 認証失敗時の正しいステータスコード
  → レート制限到達時の 429 レスポンス
  → 認可サービスダウン時のフォールバック（deny）
```

---

## 14. FAQ・トラブルシューティング

```
Q1: 401 と 403 の使い分けがわかりません。
A1: 以下のルールで判断してください:
    → 認証情報がない / 無効 → 401
    → 認証済みだが権限不足 → 403
    → 迷ったら: 「再認証したら結果が変わるか?」
      YES → 401（認証の問題）
      NO  → 403（認可の問題）

Q2: JWT とセッションはどちらを使うべき?
A2: アーキテクチャ次第です:
    → モノリシック Web アプリ → セッション（シンプル、即時無効化可能）
    → SPA + API → JWT（ステートレス、CORS 対応しやすい）
    → マイクロサービス → JWT（サービス間で公開鍵のみ共有）
    → ハイブリッド → BFF パターン（セッション + JWT 内部利用）

Q3: RBAC と ABAC はどう使い分ける?
A3: 複雑さで判断してください:
    → 権限が「誰が」で決まる → RBAC（シンプル）
    → 権限が「条件」で決まる → ABAC（柔軟）
    → 実務では RBAC + 条件付きチェック（ハイブリッド）が最も多い

Q4: API キーはセキュアですか?
A4: 条件付きで安全です:
    → サーバー間通信 → OK（安全な環境で管理）
    → フロントエンド → NG（漏洩リスク大）
    → API キーは認証（身元確認）のみ、認可は別途必要
    → ローテーション（定期的な更新）の仕組みが必須

Q5: パスワードリセット機能はどう実装すべき?
A5: 安全な実装のポイント:
    → ユーザー存在の有無に関わらず同じレスポンスを返す
    → リセットトークンは暗号的に安全なランダム値
    → 有効期限は短く（15-30分）
    → 使用後は即座に無効化
    → リセット成功時に全セッションを無効化

Q6: CORS の設定を間違えると何が起きますか?
A6: 設定ミスの影響:
    → Access-Control-Allow-Origin: * + credentials: true
      → ブラウザがブロック（仕様で禁止）
    → 特定 Origin のみ許可しないと
      → 任意のサイトから API コール可能（Cookie 送信含む）
    → preflight キャッシュが長すぎると
      → CORS 設定変更が反映されない

Q7: 認証情報をどこに保存すべき?
A7: 保存場所ごとのリスク:
    → Cookie（HttpOnly, Secure）: XSS 耐性あり、CSRF 対策必要
    → localStorage: XSS で窃取可能、CSRF 耐性あり
    → sessionStorage: タブを閉じると消える、XSS で窃取可能
    → メモリ（変数）: 最もセキュアだがリロードで消失
    推奨: セッション → HttpOnly Cookie
         JWT → メモリ + refresh を HttpOnly Cookie（BFF パターン）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 認証 | 「誰か」を確認する。失敗時は 401 |
| 認可 | 「何ができるか」を判定する。失敗時は 403 |
| 認証要素 | 知識・所有・生体の3要素。MFAは異なる要素の組合せ |
| 脅威 | ブルートフォース、フィッシング、セッションハイジャック、CSRF、権限昇格等 |
| 原則 | 最小権限、多層防御、フェイルセキュア、ゼロトラスト |
| 方式 | セッション、JWT、OAuth 2.0、Passkeys 等 |
| RBAC | ロールベース。シンプルで管理しやすい |
| ABAC | 属性ベース。柔軟だが複雑 |
| 監査 | 誰が何をしたかの記録。インシデント対応の基盤 |
| ヘッダー | HSTS、CSP、X-Frame-Options 等で多層防御 |

---

## 次に読むべきガイド
--> [[01-password-security.md]] -- パスワードセキュリティ

---

## 参考文献
1. OWASP. "Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. NIST. "SP 800-63B: Digital Identity Guidelines." nist.gov, 2020.
3. RFC 7235. "Hypertext Transfer Protocol (HTTP/1.1): Authentication." IETF, 2014.
4. OWASP. "Authorization Cheat Sheet." cheatsheetseries.owasp.org, 2024.
5. NIST. "SP 800-162: Guide to Attribute Based Access Control." nist.gov, 2014.
6. Zanzibar. "Google's Consistent, Global Authorization System." USENIX ATC, 2019.
7. STRIDE. "The STRIDE Threat Model." Microsoft, 2024.
