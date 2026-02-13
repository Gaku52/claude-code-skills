# CSRF 防御

> CSRF（Cross-Site Request Forgery）は認証済みユーザーの操作を偽造する攻撃。Synchronizer Token パターン、Double Submit Cookie、SameSite Cookie、Origin ヘッダー検証まで、CSRF 攻撃の仕組みと多層防御を解説する。

## 前提知識

- [[01-session-store.md]] — セッションストアの基本
- HTTP Cookie の仕組み（Set-Cookie ヘッダー、Cookie 属性）
- 同一オリジンポリシー（Same-Origin Policy）の基本

## この章で学ぶこと

- [ ] CSRF 攻撃の仕組みとリスクを理解する
- [ ] 主要な CSRF 防御パターンを実装できるようになる
- [ ] SameSite Cookie による防御の効果と限界を把握する
- [ ] Next.js / Express での実践的な CSRF 対策を実装できる
- [ ] 多層防御の設計原則を把握する

---

## 1. CSRF 攻撃の仕組み

### 1.1 基本的な攻撃フロー

```
CSRF 攻撃フロー:

  ① ユーザーが bank.com にログイン中（Cookie 有効）
  ② 攻撃者が evil.com に以下のHTMLを仕込む:
     <form action="https://bank.com/transfer" method="POST">
       <input type="hidden" name="to" value="attacker" />
       <input type="hidden" name="amount" value="1000000" />
     </form>
     <script>document.forms[0].submit();</script>
  ③ ユーザーが evil.com を訪問
  ④ フォームが自動送信
  ⑤ ブラウザが bank.com の Cookie を自動付与
  ⑥ bank.com はユーザーからの正規リクエストと判断
  ⑦ 送金が実行される

  なぜ成功するか:
  → ブラウザはクロスサイトリクエストでも Cookie を送信する
  → サーバーは Cookie のみでユーザーを認証している
  → リクエストが正規のユーザーからか判別できない
```

### 1.2 攻撃の詳細な分類

```
CSRF 攻撃の種類:

  ┌─────────────────────────────────────────────────────────┐
  │                    CSRF 攻撃分類                         │
  ├──────────────┬──────────────────────────────────────────┤
  │ 種類         │ 説明                                     │
  ├──────────────┼──────────────────────────────────────────┤
  │ POST CSRF    │ 隠しフォームの自動送信                     │
  │              │ → 最も一般的な攻撃ベクター                 │
  │              │ → 状態変更操作を狙う                      │
  ├──────────────┼──────────────────────────────────────────┤
  │ GET CSRF     │ img/script タグで GET リクエスト発行       │
  │              │ → GET で状態変更する API が対象             │
  │              │ → <img src="bank.com/transfer?to=evil">   │
  ├──────────────┼──────────────────────────────────────────┤
  │ Login CSRF   │ 攻撃者のアカウントでログインさせる          │
  │              │ → ユーザーが攻撃者のアカウントで操作        │
  │              │ → 入力した情報が攻撃者に漏洩              │
  ├──────────────┼──────────────────────────────────────────┤
  │ JSON CSRF    │ Content-Type を偽装した JSON 送信          │
  │              │ → enctype="text/plain" の悪用              │
  │              │ → CORS 制限のバイパス                     │
  ├──────────────┼──────────────────────────────────────────┤
  │ XHR CSRF     │ JavaScript による非同期リクエスト           │
  │              │ → withCredentials: true で Cookie 送信     │
  │              │ → CORS が許可されている場合に成功           │
  └──────────────┴──────────────────────────────────────────┘
```

### 1.3 攻撃が成立する条件

```
CSRF 攻撃成立の3条件:

  ┌──────────────────────────────────────────────────┐
  │ 条件①: Cookie ベースの認証を使用している           │
  │   → セッション Cookie が自動送信される              │
  │   → Authorization ヘッダーは自動送信されない        │
  └──────────────────┬───────────────────────────────┘
                     ↓
  ┌──────────────────────────────────────────────────┐
  │ 条件②: 予測可能なリクエスト構造                    │
  │   → パラメータ名と値が推測可能                     │
  │   → ランダムトークン等の秘密値が不要               │
  └──────────────────┬───────────────────────────────┘
                     ↓
  ┌──────────────────────────────────────────────────┐
  │ 条件③: 状態変更を行う操作がある                    │
  │   → 送金、購入、パスワード変更                     │
  │   → メールアドレス変更 → アカウント乗っ取り         │
  │   → 管理操作（ユーザー削除、権限変更）              │
  └──────────────────────────────────────────────────┘

  3つすべてが揃った場合のみ CSRF 攻撃が成立する
```

### 1.4 実際の攻撃シナリオ

```typescript
// 攻撃シナリオ1: 送金（POST CSRF）
// evil.com に設置されたHTML
`
<html>
<body onload="document.forms[0].submit()">
  <form action="https://bank.com/api/transfer" method="POST">
    <input type="hidden" name="recipient" value="attacker-account" />
    <input type="hidden" name="amount" value="50000" />
    <input type="hidden" name="currency" value="JPY" />
  </form>
</body>
</html>
`;

// 攻撃シナリオ2: パスワード変更（POST CSRF）
`
<iframe style="display:none" name="csrf-frame"></iframe>
<form action="https://target.com/api/change-password" method="POST" target="csrf-frame">
  <input type="hidden" name="new_password" value="hacked123" />
  <input type="hidden" name="confirm_password" value="hacked123" />
</form>
<script>document.forms[0].submit();</script>
`;

// 攻撃シナリオ3: メールアドレス変更 → アカウント乗っ取り
`
<img src="https://target.com/api/update-email?email=attacker@evil.com"
     style="display:none" />
`;
// ↑ GET で状態変更する設計が危険

// 攻撃シナリオ4: JSON リクエストの偽装
`
<form action="https://target.com/api/update-profile" method="POST"
      enctype="text/plain">
  <input name='{"name":"attacker","ignore":"' value='"}' />
</form>
`;
// Content-Type: text/plain でもサーバーが JSON としてパースする場合
```

### 1.5 Login CSRF の詳細

```
Login CSRF 攻撃:

  通常の CSRF とは異なるパターン

  ① 攻撃者が自分のアカウントの認証情報でログインリクエストを偽造
  ② ユーザーが攻撃者のアカウントでログインした状態になる
  ③ ユーザーがサービスを利用（個人情報入力、ファイルアップロード等）
  ④ 攻撃者が自分のアカウントでログインし、情報を閲覧

  攻撃フロー:
  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │  User   │────→│ evil.com │────→│ target  │
  │ Browser │     │  (CSRF)  │     │  .com   │
  └─────────┘     └──────────┘     └─────────┘
       │          攻撃者の認証情報で         │
       │          ログインリクエスト          │
       │                                    │
       │←── 攻撃者のセッション Cookie ──────│
       │                                    │
       │ ユーザーは攻撃者のアカウントで       │
       │ 個人情報を入力...                   │

  対策:
  → ログインフォームにも CSRF トークンを設定
  → ログイン後にセッション ID を再生成
  → ログイン通知をユーザーに送信
```

---

## 2. 防御パターン

### 2.1 4つの防御パターンの概要

```
CSRF 防御の4つのパターン:

  ① Synchronizer Token Pattern（同期トークン）:
     → サーバーがランダムトークンを生成
     → フォームに hidden field として埋め込み
     → サーバーでトークンを検証
     → 最も確実な防御

  ② Double Submit Cookie:
     → トークンを Cookie と リクエスト両方に設定
     → 両者が一致するか検証
     → サーバーに状態不要

  ③ SameSite Cookie:
     → Cookie の SameSite 属性で制御
     → ブラウザレベルの防御
     → 追加実装不要

  ④ Origin / Referer ヘッダー検証:
     → リクエスト元のオリジンを検証
     → 補助的な防御
     → ヘッダーが省略される場合がある

  推奨: ③ SameSite=Lax + ① or ② の組合せ
```

### 2.2 防御パターン比較表

```
┌────────────────────┬──────────────┬──────────────┬──────────┬──────────────┐
│ パターン            │ セキュリティ │ 実装コスト    │ 状態管理 │ SPA 適合性    │
├────────────────────┼──────────────┼──────────────┼──────────┼──────────────┤
│ Synchronizer Token │ ★★★★★      │ ★★★         │ 必要     │ △ 要工夫     │
│ Double Submit      │ ★★★★       │ ★★           │ 不要     │ ○ 適合      │
│ SameSite Cookie    │ ★★★★       │ ★（最少）     │ 不要     │ ○ 自動      │
│ Origin 検証        │ ★★★         │ ★★           │ 不要     │ ○ 適合      │
│ Custom Header      │ ★★★★       │ ★★           │ 不要     │ ◎ 最適     │
│ Encrypted Token    │ ★★★★★      │ ★★★★        │ 不要     │ △ 要工夫     │
└────────────────────┴──────────────┴──────────────┴──────────┴──────────────┘
```

### 2.3 Synchronizer Token Pattern（同期トークン）

```
Synchronizer Token Pattern の内部動作:

  ┌──────────┐          ┌──────────────┐          ┌──────────┐
  │ Browser  │          │    Server    │          │ Session  │
  │          │          │              │          │  Store   │
  └────┬─────┘          └──────┬───────┘          └────┬─────┘
       │                       │                       │
       │  GET /form            │                       │
       │──────────────────────→│                       │
       │                       │ generateToken()       │
       │                       │──────────────────────→│
       │                       │  store(sid, token)    │
       │                       │                       │
       │  HTML with hidden     │                       │
       │  <input name="_csrf"  │                       │
       │   value="abc123">     │                       │
       │←──────────────────────│                       │
       │                       │                       │
       │  POST /action         │                       │
       │  _csrf=abc123         │                       │
       │──────────────────────→│                       │
       │                       │  getToken(sid)        │
       │                       │──────────────────────→│
       │                       │  "abc123"             │
       │                       │←──────────────────────│
       │                       │                       │
       │                       │ compare(req, stored)  │
       │                       │ → 一致 → 処理実行     │
       │  200 OK               │                       │
       │←──────────────────────│                       │
       │                       │                       │
```

```typescript
// ① Synchronizer Token Pattern - 完全な実装
import crypto from 'crypto';
import { Request, Response, NextFunction } from 'express';

interface CSRFStore {
  setToken(sessionId: string, token: string): Promise<void>;
  getToken(sessionId: string): Promise<string | null>;
  deleteToken(sessionId: string): Promise<void>;
}

// Redis ベースのトークンストア
class RedisCSRFStore implements CSRFStore {
  private redis: Redis;
  private prefix = 'csrf:';
  private ttl = 3600; // 1時間

  constructor(redis: Redis) {
    this.redis = redis;
  }

  async setToken(sessionId: string, token: string): Promise<void> {
    await this.redis.setex(`${this.prefix}${sessionId}`, this.ttl, token);
  }

  async getToken(sessionId: string): Promise<string | null> {
    return this.redis.get(`${this.prefix}${sessionId}`);
  }

  async deleteToken(sessionId: string): Promise<void> {
    await this.redis.del(`${this.prefix}${sessionId}`);
  }
}

// トークン生成（暗号論的に安全なランダム値）
function generateCSRFToken(): string {
  return crypto.randomBytes(32).toString('hex');
  // 32バイト = 256ビット → 64文字の16進数文字列
  // 2^256 の可能性 → 推測は事実上不可能
}

// トークン比較（タイミング攻撃対策）
function safeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b));
  // ※ 通常の === 比較は最初の不一致文字で返却するため
  //    レスポンス時間からトークンを推測される可能性がある
  //    timingSafeEqual は常に同じ時間で比較する
}

// CSRF 保護ミドルウェア
function csrfProtection(store: CSRFStore) {
  return async (req: Request, res: Response, next: NextFunction) => {
    // GET, HEAD, OPTIONS はスキップ（安全なメソッド = RFC 7231 §4.2.1）
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      // トークンを生成してレスポンスに含める
      const sessionId = req.session?.id;
      if (sessionId) {
        let token = await store.getToken(sessionId);
        if (!token) {
          token = generateCSRFToken();
          await store.setToken(sessionId, token);
        }
        // テンプレートで使えるようにローカル変数に設定
        res.locals.csrfToken = token;
      }
      return next();
    }

    // POST/PUT/DELETE 等の状態変更メソッド
    const sessionId = req.session?.id;
    if (!sessionId) {
      return res.status(403).json({ error: 'No session' });
    }

    // トークンの取得元（優先順位）
    const token =
      req.headers['x-csrf-token'] as string ||   // カスタムヘッダー
      req.body?._csrf ||                           // フォームの hidden field
      req.query?._csrf as string;                  // クエリパラメータ（非推奨）

    const storedToken = await store.getToken(sessionId);

    if (!token || !storedToken || !safeCompare(token, storedToken)) {
      return res.status(403).json({
        error: 'Invalid CSRF token',
        message: 'CSRF token validation failed. Please reload the page and try again.',
      });
    }

    // トークンを再生成（ワンタイム使用 - Per-Request Token）
    // ※ Per-Session Token の場合はここを削除
    const newToken = generateCSRFToken();
    await store.setToken(sessionId, newToken);
    res.setHeader('X-CSRF-Token', newToken);

    next();
  };
}

// Express アプリケーションへの適用
import express from 'express';

const app = express();
const redis = new Redis(process.env.REDIS_URL!);
const csrfStore = new RedisCSRFStore(redis);

app.use(csrfProtection(csrfStore));

// フォームに埋め込み（サーバーサイドレンダリング）
// <input type="hidden" name="_csrf" value="${res.locals.csrfToken}" />

// SPA の場合: メタタグ or API で取得
// <meta name="csrf-token" content="${res.locals.csrfToken}" />
```

### 2.4 Per-Session Token vs Per-Request Token

```
トークンの更新戦略:

  Per-Session Token:
  ┌────────────────────────────────────────────┐
  │ セッション開始時に1つのトークンを生成       │
  │ セッション中はそのトークンを使い続ける       │
  │                                            │
  │ 利点:                                      │
  │ → 実装がシンプル                            │
  │ → ブラウザの「戻る」ボタンで問題が起きない   │
  │ → タブ間で共有可能                          │
  │                                            │
  │ 欠点:                                      │
  │ → トークン漏洩時の影響が大きい              │
  │ → XSS と組み合わさると危険                  │
  └────────────────────────────────────────────┘

  Per-Request Token:
  ┌────────────────────────────────────────────┐
  │ リクエストごとに新しいトークンを生成         │
  │ 使用後は無効化                              │
  │                                            │
  │ 利点:                                      │
  │ → より高いセキュリティ                      │
  │ → トークン漏洩時の影響が限定的              │
  │                                            │
  │ 欠点:                                      │
  │ → 「戻る」ボタンでフォーム再送信が失敗       │
  │ → 複数タブでの同時操作で問題                │
  │ → AJAX 多用のアプリで複雑                   │
  └────────────────────────────────────────────┘

  推奨:
  → 一般的な Web アプリ: Per-Session Token
  → 金融系・高セキュリティ: Per-Request Token
  → SPA: Per-Session Token + Custom Header
```

### 2.5 Double Submit Cookie

```
Double Submit Cookie の内部動作:

  ポイント: 攻撃者は Cookie を「設定」できないが
           ブラウザは Cookie を「自動送信」する

  ┌──────────┐          ┌──────────────┐
  │ Browser  │          │    Server    │
  └────┬─────┘          └──────┬───────┘
       │                       │
       │  GET /page            │
       │──────────────────────→│
       │                       │ token = random()
       │  Set-Cookie:          │
       │  csrf=token123        │
       │  (httpOnly=false)     │
       │←──────────────────────│
       │                       │
       │  JavaScript が        │
       │  Cookie を読み取り     │
       │  ↓                    │
       │  POST /action         │
       │  Cookie: csrf=token123│ ← ブラウザが自動送信
       │  X-CSRF-Token: token123│ ← JS が明示的に設定
       │──────────────────────→│
       │                       │
       │                       │ Cookie の値 === ヘッダーの値？
       │                       │ → 一致 → 正規リクエスト
       │                       │
       │  攻撃者の場合:         │
       │  Cookie: csrf=token123│ ← ブラウザが自動送信
       │  X-CSRF-Token: ???    │ ← 攻撃者は Cookie を読めない
       │                       │   → 不一致 → 拒否
```

```typescript
// ② Double Submit Cookie - 完全な実装
import crypto from 'crypto';
import { Request, Response, NextFunction } from 'express';

interface DoubleSubmitOptions {
  cookieName?: string;
  headerName?: string;
  cookieOptions?: {
    secure?: boolean;
    sameSite?: 'strict' | 'lax' | 'none';
    path?: string;
    domain?: string;
  };
  // HMAC 署名を使用するか（推奨）
  signedCookie?: boolean;
  secret?: string;
}

function doubleSubmitCSRF(options: DoubleSubmitOptions = {}) {
  const {
    cookieName = 'csrf-token',
    headerName = 'x-csrf-token',
    cookieOptions = {},
    signedCookie = true,
    secret = process.env.CSRF_SECRET || 'default-change-me',
  } = options;

  // HMAC 署名付きトークン生成
  function createSignedToken(): string {
    const value = crypto.randomBytes(32).toString('hex');
    if (!signedCookie) return value;

    const signature = crypto
      .createHmac('sha256', secret)
      .update(value)
      .digest('hex');
    return `${value}.${signature}`;
  }

  // 署名の検証
  function verifySignature(token: string): boolean {
    if (!signedCookie) return true;

    const [value, signature] = token.split('.');
    if (!value || !signature) return false;

    const expected = crypto
      .createHmac('sha256', secret)
      .update(value)
      .digest('hex');
    return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expected));
  }

  return (req: Request, res: Response, next: NextFunction) => {
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      // GET 時にトークンを Cookie に設定
      if (!req.cookies[cookieName]) {
        const token = createSignedToken();
        res.cookie(cookieName, token, {
          httpOnly: false,   // JavaScript で読める必要あり
          secure: cookieOptions.secure ?? process.env.NODE_ENV === 'production',
          sameSite: cookieOptions.sameSite ?? 'lax',
          path: cookieOptions.path ?? '/',
          domain: cookieOptions.domain,
          maxAge: 24 * 60 * 60 * 1000, // 24時間
        });
      }
      return next();
    }

    // POST/PUT/DELETE 時: Cookie とヘッダーのトークンを比較
    const cookieToken = req.cookies[cookieName];
    const headerToken = req.headers[headerName] as string
      || req.body?._csrf;

    if (!cookieToken || !headerToken) {
      return res.status(403).json({
        error: 'CSRF validation failed',
        message: 'Missing CSRF token',
      });
    }

    // 署名の検証
    if (!verifySignature(cookieToken)) {
      return res.status(403).json({
        error: 'CSRF validation failed',
        message: 'Invalid CSRF token signature',
      });
    }

    // Cookie とヘッダーの値を比較
    if (!crypto.timingSafeEqual(Buffer.from(cookieToken), Buffer.from(headerToken))) {
      return res.status(403).json({
        error: 'CSRF validation failed',
        message: 'CSRF token mismatch',
      });
    }

    next();
  };
}

// クライアント側の実装
// const csrfToken = document.cookie.match(/csrf-token=([^;]+)/)?.[1];
// fetch('/api/data', {
//   method: 'POST',
//   headers: {
//     'Content-Type': 'application/json',
//     'X-CSRF-Token': csrfToken,
//   },
//   credentials: 'same-origin',
//   body: JSON.stringify(data),
// });
```

### 2.6 Signed Double Submit Cookie（署名付き）

```
なぜ署名が必要か:

  通常の Double Submit Cookie の弱点:
  ┌──────────────────────────────────────────────────┐
  │ 攻撃者がサブドメインを制御している場合:            │
  │                                                  │
  │ evil.sub.example.com から                         │
  │ Set-Cookie: csrf=attacker-value; domain=.example.com │
  │ を設定可能                                        │
  │                                                  │
  │ → Cookie を上書きして、ヘッダーにも同じ値を設定    │
  │ → 検証を通過してしまう（Cookie Injection 攻撃）   │
  └──────────────────────────────────────────────────┘

  HMAC 署名付きの場合:
  ┌──────────────────────────────────────────────────┐
  │ トークン = value.HMAC(secret, value)              │
  │                                                  │
  │ 攻撃者は secret を知らないため、                   │
  │ 正しい署名を生成できない                          │
  │ → Cookie を上書きしても署名検証で失敗              │
  └──────────────────────────────────────────────────┘
```

### 2.7 Custom Request Header パターン

```typescript
// ③ Custom Request Header パターン
// fetch API は Same-Origin でなければカスタムヘッダーを送信できない
// → CORS プリフライトが必要 → 攻撃者のサイトからは送信不可

// サーバー側
function customHeaderCSRF(req: Request, res: Response, next: NextFunction) {
  if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
    return next();
  }

  // カスタムヘッダーの存在を確認（値は問わない）
  const csrfHeader = req.headers['x-requested-with'];

  if (csrfHeader !== 'XMLHttpRequest') {
    return res.status(403).json({
      error: 'CSRF validation failed',
      message: 'Missing X-Requested-With header',
    });
  }

  next();
}

// クライアント側
// fetch('/api/data', {
//   method: 'POST',
//   headers: {
//     'Content-Type': 'application/json',
//     'X-Requested-With': 'XMLHttpRequest', // カスタムヘッダー
//   },
//   credentials: 'same-origin',
//   body: JSON.stringify(data),
// });

// ※ 注意: Content-Type が application/json の場合、
//    CORS プリフライト（OPTIONS）が発行され、
//    クロスオリジンからのリクエストはブロックされる
//    ただし Content-Type: text/plain の場合はプリフライトが不要
//    → Content-Type の検証も併用すべき
```

### 2.8 Origin / Referer ヘッダー検証

```typescript
// ④ Origin / Referer ヘッダー検証 - 完全な実装
function originVerification(allowedOrigins: string[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      return next();
    }

    const origin = req.headers.origin;
    const referer = req.headers.referer;

    // Origin ヘッダーがある場合（推奨）
    if (origin) {
      if (!allowedOrigins.includes(origin)) {
        console.warn(`CSRF: Rejected request from origin: ${origin}`);
        return res.status(403).json({
          error: 'Origin validation failed',
          message: `Origin ${origin} is not allowed`,
        });
      }
      return next();
    }

    // Origin がない場合は Referer をフォールバック
    // ※ Origin ヘッダーは一部のブラウザ/条件で省略される
    if (referer) {
      try {
        const refererUrl = new URL(referer);
        const refererOrigin = refererUrl.origin;
        if (!allowedOrigins.includes(refererOrigin)) {
          console.warn(`CSRF: Rejected request from referer: ${referer}`);
          return res.status(403).json({ error: 'Referer validation failed' });
        }
        return next();
      } catch {
        return res.status(403).json({ error: 'Invalid Referer header' });
      }
    }

    // Origin も Referer もない場合の判断
    // → プライバシー設定やプロキシで省略される場合がある
    // → 厳格: 拒否する（セキュリティ優先）
    // → 緩和: 許可する（互換性優先）
    console.warn('CSRF: Request without Origin or Referer header');
    return res.status(403).json({
      error: 'Missing Origin or Referer header',
    });
  };
}

// 使用例
app.use(originVerification([
  'https://myapp.com',
  'https://www.myapp.com',
  ...(process.env.NODE_ENV === 'development' ? ['http://localhost:3000'] : []),
]));
```

```
Origin / Referer ヘッダーが省略されるケース:

  ┌──────────────────────────────────────────────────┐
  │ Origin ヘッダーが省略される場合:                   │
  │                                                  │
  │ → Same-Origin の GET/HEAD リクエスト               │
  │ → Referrer-Policy: no-referrer 設定時             │
  │ → 一部の古いブラウザ                              │
  │ → ブックマークからの直接アクセス                    │
  │ → アドレスバーからの直接入力                        │
  │                                                  │
  │ Referer ヘッダーが省略される場合:                   │
  │                                                  │
  │ → HTTPS → HTTP のダウングレード                    │
  │ → Referrer-Policy: no-referrer 設定時             │
  │ → プライバシー拡張機能の使用                       │
  │ → メールクライアントからのリンク                    │
  │                                                  │
  │ → Origin 検証は補助的な防御として位置づける          │
  └──────────────────────────────────────────────────┘
```

---

## 3. SameSite Cookie による防御

### 3.1 SameSite 属性の詳細

```
SameSite 属性の効果:

  SameSite=Strict:
    → クロスサイトリクエストで Cookie を一切送信しない
    → CSRF を完全に防御
    → ただし: 外部リンクからのアクセスで未ログイン状態になる
    → 例: Google検索から bank.com をクリック → ログイン画面

  SameSite=Lax（推奨デフォルト）:
    → トップレベルの GET ナビゲーションのみ Cookie 送信
    → POST, iframe, img, fetch 等のクロスサイトリクエストはブロック
    → CSRF の主要な攻撃ベクターを防御
    → UX への影響が少ない

  SameSite=None:
    → すべてのクロスサイトリクエストで Cookie 送信
    → Secure 属性が必須
    → サードパーティ Cookie が必要な場合のみ
```

### 3.2 SameSite の判定ロジック

```
「サイト」の定義（SameSite の判定基準）:

  SameSite は eTLD+1（有効トップレベルドメイン + 1）で判定

  eTLD+1 の例:
    example.com          → eTLD+1 = example.com
    app.example.com      → eTLD+1 = example.com
    sub.app.example.com  → eTLD+1 = example.com
    example.co.jp        → eTLD+1 = example.co.jp (co.jp が eTLD)
    myapp.github.io      → eTLD+1 = myapp.github.io (github.io が eTLD)

  Same-Site の判定:
  ┌──────────────────────┬──────────────────────┬──────────┐
  │ リクエスト元          │ リクエスト先          │ 判定     │
  ├──────────────────────┼──────────────────────┼──────────┤
  │ app.example.com      │ api.example.com      │ Same-Site│
  │ example.com          │ sub.example.com      │ Same-Site│
  │ example.com          │ other.com            │ Cross-Site│
  │ myapp.github.io      │ other.github.io      │ Cross-Site│
  │ http://example.com   │ https://example.com  │ Cross-Site│
  └──────────────────────┴──────────────────────┴──────────┘

  ※ スキーム（http/https）も考慮される（Schemeful Same-Site）
  ※ ポート番号は同一でなくても Same-Site
```

### 3.3 リクエスト種別ごとの Cookie 送信

```
SameSite によるリクエスト種別ごとの Cookie 送信:

  ┌──────────────────────────────┬────────┬──────┬──────┐
  │ リクエスト種別                │ Strict │ Lax  │ None │
  ├──────────────────────────────┼────────┼──────┼──────┤
  │ <a href="...">リンク         │ ✗      │ ✓    │ ✓    │
  │ <form method="GET">          │ ✗      │ ✓    │ ✓    │
  │ <form method="POST">         │ ✗      │ ✗    │ ✓    │
  │ <img src="...">              │ ✗      │ ✗    │ ✓    │
  │ <iframe src="...">           │ ✗      │ ✗    │ ✓    │
  │ <script src="...">           │ ✗      │ ✗    │ ✓    │
  │ fetch(url, {credentials})    │ ✗      │ ✗    │ ✓    │
  │ XMLHttpRequest               │ ✗      │ ✗    │ ✓    │
  │ window.location = url        │ ✗      │ ✓    │ ✓    │
  │ <link rel="prerender">       │ ✗      │ ✓    │ ✓    │
  └──────────────────────────────┴────────┴──────┴──────┘

  Lax で Cookie が送信される条件:
  ① トップレベルナビゲーション（URLバーが変わる）
  ② HTTP GET メソッド
  ③ ①②の両方を満たす場合のみ
```

### 3.4 SameSite の限界

```
SameSite の限界:

  ① GET リクエストでの状態変更:
     → GET /delete-account のようなAPIは SameSite=Lax でも攻撃可能
     → 対策: 状態変更は必ず POST/PUT/DELETE を使用

  ② サブドメイン間:
     → SameSite は eTLD+1 で判定
     → app.example.com と evil.example.com は同一サイト
     → サブドメインの信頼性に依存
     → 対策: サブドメインの管理を厳格化

  ③ 古いブラウザ:
     → SameSite をサポートしないブラウザが存在
     → iOS 12 以前の Safari は SameSite=None を Unknown として扱う
     → 追加の防御策との併用が推奨

  ④ Lax+POST の2分間ウィンドウ（Chrome）:
     → Chrome は新しい Cookie に対して最初の2分間
        POST リクエストでも Lax Cookie を送信する
     → トップレベルのクロスサイト POST ナビゲーションが対象
     → 2020年に導入、互換性のための措置

  ⑤ Same-Site ≠ Same-Origin:
     → Same-Site 判定は Same-Origin より緩い
     → サブドメインからの攻撃を防げない
     → XSS がサブドメインにあると危険
```

```typescript
// SameSite Cookie の正しい設定
import { CookieOptions } from 'express';

// セッション Cookie の推奨設定
const sessionCookieOptions: CookieOptions = {
  httpOnly: true,        // JavaScript からアクセス不可
  secure: true,          // HTTPS のみ
  sameSite: 'lax',       // クロスサイト POST をブロック
  path: '/',             // すべてのパスで有効
  maxAge: 30 * 24 * 60 * 60 * 1000, // 30日
  // domain は設定しない（発行元ドメインのみに限定）
};

// 高セキュリティ Cookie（管理画面等）
const strictCookieOptions: CookieOptions = {
  httpOnly: true,
  secure: true,
  sameSite: 'strict',    // クロスサイトリクエストで一切送信しない
  path: '/admin',        // 管理画面パスのみ
  maxAge: 4 * 60 * 60 * 1000, // 4時間
};

// サードパーティ Cookie が必要な場合（埋め込みウィジェット等）
const thirdPartyCookieOptions: CookieOptions = {
  httpOnly: true,
  secure: true,          // SameSite=None は Secure 必須
  sameSite: 'none',      // クロスサイトリクエストを許可
  path: '/',
  maxAge: 24 * 60 * 60 * 1000, // 24時間
};
```

---

## 4. Next.js での CSRF 対策

### 4.1 Server Actions の自動保護

```typescript
// Next.js App Router での CSRF 対策

// Server Actions は自動的に CSRF 保護される
// Next.js が内部的に Origin ヘッダーを検証する
// → 追加の CSRF 対策は不要

// Server Actions の内部動作:
// 1. クライアントが POST リクエストを送信
// 2. Next.js が Origin ヘッダーを検証
// 3. Origin が一致しない場合は 403 を返す
// 4. Content-Type: multipart/form-data で送信
// 5. Next-Action ヘッダーでアクション ID を指定
// 6. これらの検証により CSRF 攻撃を防御

// app/actions/article.ts
'use server';

import { auth } from '@/auth';
import { revalidatePath } from 'next/cache';

export async function createArticle(formData: FormData) {
  // Server Actions は自動的に CSRF 保護される
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  const title = formData.get('title') as string;
  const content = formData.get('content') as string;

  // 入力バリデーション
  if (!title || title.length > 200) {
    throw new Error('Invalid title');
  }

  await prisma.article.create({
    data: {
      title,
      content,
      authorId: session.user.id,
    },
  });

  revalidatePath('/articles');
}

// app/articles/new/page.tsx
export default function NewArticlePage() {
  return (
    <form action={createArticle}>
      <input name="title" type="text" required />
      <textarea name="content" required />
      <button type="submit">Create</button>
    </form>
  );
}
```

### 4.2 API Routes の CSRF 対策

```typescript
// API Routes の場合は手動で対策が必要

// middleware.ts
import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  // API ルートの POST/PUT/DELETE を保護
  if (
    request.nextUrl.pathname.startsWith('/api/') &&
    !['GET', 'HEAD', 'OPTIONS'].includes(request.method)
  ) {
    const origin = request.headers.get('origin');
    const host = request.headers.get('host');

    // Origin ヘッダーの検証
    if (origin) {
      try {
        const originUrl = new URL(origin);
        const expectedHost = host?.split(':')[0]; // ポート番号を除去
        const originHost = originUrl.hostname;

        if (originHost !== expectedHost) {
          console.warn(
            `CSRF blocked: Origin ${origin} does not match host ${host}`
          );
          return NextResponse.json(
            { error: 'CSRF validation failed' },
            { status: 403 }
          );
        }
      } catch {
        return NextResponse.json(
          { error: 'Invalid Origin header' },
          { status: 403 }
        );
      }
    } else {
      // Origin ヘッダーがない場合
      // → API Routes は通常 fetch で呼ばれるため Origin が存在するはず
      // → ない場合は不正なリクエストの可能性
      const referer = request.headers.get('referer');
      if (!referer) {
        return NextResponse.json(
          { error: 'Missing Origin header' },
          { status: 403 }
        );
      }
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/api/:path*'],
};
```

### 4.3 Next.js で Double Submit Cookie を使う

```typescript
// lib/csrf.ts
import { cookies } from 'next/headers';
import crypto from 'crypto';

const CSRF_COOKIE = '__csrf';
const CSRF_HEADER = 'x-csrf-token';

// トークン生成（Server Component / Server Action から呼出し）
export async function getCSRFToken(): Promise<string> {
  const cookieStore = await cookies();
  let token = cookieStore.get(CSRF_COOKIE)?.value;

  if (!token) {
    token = crypto.randomBytes(32).toString('hex');
    cookieStore.set(CSRF_COOKIE, token, {
      httpOnly: false, // JS から読み取り可能
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      path: '/',
      maxAge: 60 * 60, // 1時間
    });
  }

  return token;
}

// トークン検証（API Route / Server Action から呼出し）
export async function validateCSRFToken(request: Request): Promise<boolean> {
  const cookieStore = await cookies();
  const cookieToken = cookieStore.get(CSRF_COOKIE)?.value;
  const headerToken = request.headers.get(CSRF_HEADER);

  if (!cookieToken || !headerToken) return false;

  return crypto.timingSafeEqual(
    Buffer.from(cookieToken),
    Buffer.from(headerToken)
  );
}

// Client Component 用のフック
// hooks/useCSRF.ts
'use client';

export function useCSRFToken(): string | null {
  const match = document.cookie.match(/__csrf=([^;]+)/);
  return match ? match[1] : null;
}

// API 呼び出しラッパー
export function csrfFetch(url: string, options: RequestInit = {}): Promise<Response> {
  const csrfToken = document.cookie.match(/__csrf=([^;]+)/)?.[1];

  return fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'X-CSRF-Token': csrfToken || '',
    },
    credentials: 'same-origin',
  });
}
```

### 4.4 Next.js Middleware による包括的な CSRF 対策

```typescript
// middleware.ts - 包括的な CSRF 保護
import { NextRequest, NextResponse } from 'next/server';

// 保護対象のパス
const PROTECTED_API_PATHS = ['/api/'];
// CSRF 検証をスキップするパス（Webhook 等）
const SKIP_CSRF_PATHS = ['/api/webhooks/stripe', '/api/webhooks/github'];

function isProtectedRoute(pathname: string): boolean {
  return PROTECTED_API_PATHS.some(p => pathname.startsWith(p))
    && !SKIP_CSRF_PATHS.some(p => pathname.startsWith(p));
}

export function middleware(request: NextRequest) {
  // Safe methods はスキップ
  if (['GET', 'HEAD', 'OPTIONS'].includes(request.method)) {
    return NextResponse.next();
  }

  // 保護対象でないパスはスキップ
  if (!isProtectedRoute(request.nextUrl.pathname)) {
    return NextResponse.next();
  }

  // 1. Origin ヘッダー検証
  const origin = request.headers.get('origin');
  if (origin) {
    const allowedOrigins = [
      process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
    ];

    if (!allowedOrigins.some(allowed => {
      try {
        return new URL(allowed).origin === new URL(origin).origin;
      } catch {
        return false;
      }
    })) {
      return NextResponse.json({ error: 'CSRF: Invalid origin' }, { status: 403 });
    }
  }

  // 2. Double Submit Cookie 検証
  const cookieToken = request.cookies.get('__csrf')?.value;
  const headerToken = request.headers.get('x-csrf-token');

  if (cookieToken && headerToken) {
    // Double Submit が設定されている場合は検証
    if (cookieToken !== headerToken) {
      return NextResponse.json({ error: 'CSRF: Token mismatch' }, { status: 403 });
    }
  }

  return NextResponse.next();
}
```

---

## 5. SPA (React / Vue) での CSRF 対策

### 5.1 SPA 特有の考慮事項

```
SPA と CSRF:

  SPA が CSRF に比較的強い理由:
  ┌──────────────────────────────────────────────────┐
  │ ① API リクエストは通常 JSON                       │
  │   → Content-Type: application/json                │
  │   → CORS プリフライト (OPTIONS) が発行される       │
  │   → クロスオリジンからは送信不可（CORS 設定次第）   │
  │                                                   │
  │ ② カスタムヘッダーを使用                           │
  │   → Authorization: Bearer token                   │
  │   → X-Requested-With: XMLHttpRequest              │
  │   → CORS プリフライトが必要                        │
  │                                                   │
  │ ③ トークン認証の場合                               │
  │   → Cookie ではなく localStorage/memory に保存     │
  │   → 自動送信されない                               │
  └──────────────────────────────────────────────────┘

  SPA でも CSRF 対策が必要な場合:
  ┌──────────────────────────────────────────────────┐
  │ ① Cookie ベースの認証を使用                       │
  │   → httpOnly Cookie でセッション管理               │
  │   → ブラウザが自動送信                             │
  │                                                   │
  │ ② API が application/x-www-form-urlencoded を受容 │
  │   → CORS プリフライトなしで送信可能                 │
  │                                                   │
  │ ③ CORS 設定が緩い                                 │
  │   → Access-Control-Allow-Origin: *                │
  │   → credentials: true との組合せは危険              │
  └──────────────────────────────────────────────────┘
```

### 5.2 React での CSRF 対策実装

```typescript
// React アプリでの CSRF 対策

// CSRFProvider.tsx
import { createContext, useContext, useEffect, useState } from 'react';

interface CSRFContextType {
  token: string | null;
  refresh: () => Promise<void>;
}

const CSRFContext = createContext<CSRFContextType>({
  token: null,
  refresh: async () => {},
});

export function CSRFProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);

  const fetchToken = async () => {
    try {
      const res = await fetch('/api/csrf-token', {
        credentials: 'same-origin',
      });
      const data = await res.json();
      setToken(data.token);
    } catch (err) {
      console.error('Failed to fetch CSRF token:', err);
    }
  };

  useEffect(() => {
    fetchToken();
  }, []);

  return (
    <CSRFContext.Provider value={{ token, refresh: fetchToken }}>
      {children}
    </CSRFContext.Provider>
  );
}

export function useCSRF() {
  return useContext(CSRFContext);
}

// CSRF 対応の fetch ラッパー
export function useCSRFFetch() {
  const { token, refresh } = useCSRF();

  return async (url: string, options: RequestInit = {}): Promise<Response> => {
    const res = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        'X-CSRF-Token': token || '',
      },
      credentials: 'same-origin',
    });

    // 403 の場合はトークンを更新してリトライ
    if (res.status === 403) {
      await refresh();
      return fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          'X-CSRF-Token': token || '',
        },
        credentials: 'same-origin',
      });
    }

    return res;
  };
}
```

### 5.3 Axios インターセプターによる自動付与

```typescript
// Axios での CSRF トークン自動付与
import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  withCredentials: true, // Cookie を送信
});

// リクエストインターセプター: CSRF トークンを自動付与
api.interceptors.request.use((config) => {
  // Cookie から CSRF トークンを取得
  const token = document.cookie
    .split('; ')
    .find(row => row.startsWith('csrf-token='))
    ?.split('=')[1];

  if (token && config.method !== 'get') {
    config.headers['X-CSRF-Token'] = token;
  }

  return config;
});

// レスポンスインターセプター: 新しいトークンを Cookie に保存
api.interceptors.response.use(
  (response) => {
    // サーバーが新しいトークンをヘッダーで返す場合
    const newToken = response.headers['x-csrf-token'];
    if (newToken) {
      document.cookie = `csrf-token=${newToken}; path=/; SameSite=Lax`;
    }
    return response;
  },
  (error) => {
    if (error.response?.status === 403) {
      // CSRF エラーの場合はページをリロード
      console.warn('CSRF token expired. Refreshing...');
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

export default api;
```

---

## 6. CSRF 対策が不要なケース

```
CSRF 対策が不要な場合:

  ① Bearer トークン認証:
     → Authorization ヘッダーは自動送信されない
     → JavaScript で明示的に設定する必要がある
     → 攻撃者はトークンを設定できない

  ② SameSite=Strict の Cookie:
     → クロスサイトリクエストで Cookie が送信されない
     → ただし UX に影響

  ③ API Key 認証:
     → カスタムヘッダーで送信
     → 自動送信されない

  ④ CORS を正しく設定した API:
     → Access-Control-Allow-Origin を限定
     → Access-Control-Allow-Credentials: true
     → プリフライトチェックで不正リクエストをブロック

  CSRF 対策が必要な場合:
     → Cookie ベースのセッション認証
     → SameSite=None の Cookie
     → SameSite=Lax で GET に状態変更がある場合
     → CORS 設定が緩い場合
```

---

## 7. 多層防御の設計

### 7.1 防御レイヤーの構成

```
CSRF 多層防御の推奨構成:

  ┌─────────────────────────────────────────────────┐
  │ Layer 1: SameSite=Lax Cookie（ブラウザレベル）    │
  │ → 追加実装不要、ほとんどの攻撃をブロック           │
  ├─────────────────────────────────────────────────┤
  │ Layer 2: Origin ヘッダー検証（ネットワークレベル）  │
  │ → リクエスト元の検証                              │
  ├─────────────────────────────────────────────────┤
  │ Layer 3: CSRF トークン（アプリケーションレベル）    │
  │ → Synchronizer Token または Double Submit Cookie  │
  ├─────────────────────────────────────────────────┤
  │ Layer 4: Content-Type 検証                       │
  │ → application/json のみ受け入れ                   │
  ├─────────────────────────────────────────────────┤
  │ Layer 5: カスタムヘッダー要求                     │
  │ → X-Requested-With 等の存在確認                   │
  └─────────────────────────────────────────────────┘

  一般的な Web アプリ: Layer 1 + Layer 2 で十分
  金融/医療系: Layer 1 + Layer 2 + Layer 3 推奨
  パブリック API: Layer 1 + Layer 4 + Layer 5
```

### 7.2 包括的な CSRF 対策ミドルウェア

```typescript
// 多層防御の統合ミドルウェア
import { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';

interface CSRFProtectionOptions {
  allowedOrigins: string[];
  cookieName?: string;
  headerName?: string;
  ignorePaths?: string[];
  requireContentType?: boolean;
  enableDoubleSubmit?: boolean;
}

function comprehensiveCSRF(options: CSRFProtectionOptions) {
  const {
    allowedOrigins,
    cookieName = 'csrf-token',
    headerName = 'x-csrf-token',
    ignorePaths = [],
    requireContentType = true,
    enableDoubleSubmit = true,
  } = options;

  return (req: Request, res: Response, next: NextFunction) => {
    // Safe methods はスキップ
    if (['GET', 'HEAD', 'OPTIONS'].includes(req.method)) {
      // Double Submit Cookie の設定
      if (enableDoubleSubmit && !req.cookies[cookieName]) {
        const token = crypto.randomBytes(32).toString('hex');
        res.cookie(cookieName, token, {
          httpOnly: false,
          secure: process.env.NODE_ENV === 'production',
          sameSite: 'lax',
          path: '/',
        });
      }
      return next();
    }

    // 除外パスのチェック（Webhook 等）
    if (ignorePaths.some(p => req.path.startsWith(p))) {
      return next();
    }

    // Layer 2: Origin ヘッダー検証
    const origin = req.headers.origin;
    if (origin && !allowedOrigins.includes(origin)) {
      return res.status(403).json({
        error: 'csrf_origin_mismatch',
        message: 'Request origin is not allowed',
      });
    }

    // Layer 4: Content-Type 検証
    if (requireContentType) {
      const contentType = req.headers['content-type'] || '';
      const allowedTypes = [
        'application/json',
        'multipart/form-data',
        'application/x-www-form-urlencoded',
      ];
      const isAllowed = allowedTypes.some(t => contentType.includes(t));
      if (!isAllowed) {
        return res.status(415).json({
          error: 'unsupported_content_type',
          message: `Content-Type ${contentType} is not supported`,
        });
      }
    }

    // Layer 3: Double Submit Cookie 検証
    if (enableDoubleSubmit) {
      const cookieToken = req.cookies[cookieName];
      const headerToken = req.headers[headerName] as string;

      if (!cookieToken || !headerToken) {
        return res.status(403).json({
          error: 'csrf_token_missing',
          message: 'CSRF token is required',
        });
      }

      if (!crypto.timingSafeEqual(
        Buffer.from(cookieToken),
        Buffer.from(headerToken)
      )) {
        return res.status(403).json({
          error: 'csrf_token_mismatch',
          message: 'CSRF token validation failed',
        });
      }
    }

    next();
  };
}

// 使用例
app.use(comprehensiveCSRF({
  allowedOrigins: [
    'https://myapp.com',
    ...(process.env.NODE_ENV === 'development' ? ['http://localhost:3000'] : []),
  ],
  ignorePaths: ['/api/webhooks/'],
  requireContentType: true,
  enableDoubleSubmit: true,
}));
```

---

## 8. エッジケースと注意点

### 8.1 CORS と CSRF の関係

```
CORS と CSRF の関係:

  よくある誤解: 「CORS を設定すれば CSRF は防げる」
  → 部分的に正しいが不十分

  ┌──────────────────────────────────────────────────┐
  │ CORS が防ぐもの:                                  │
  │ → fetch() / XMLHttpRequest のクロスオリジンリクエスト│
  │ → カスタムヘッダーの送信                           │
  │ → レスポンスの読み取り                             │
  │                                                   │
  │ CORS が防がないもの:                               │
  │ → <form> の送信（CORS の管轄外）                   │
  │ → <img> タグによる GET リクエスト                   │
  │ → 単純リクエスト（Simple Request）のレスポンス読取   │
  │   以外の副作用                                     │
  └──────────────────────────────────────────────────┘

  ★ 重要: CORS はリクエストの送信を防ぐのではなく
          レスポンスの読み取りを制御する
          → リクエスト自体はサーバーに到達する場合がある
```

### 8.2 サブドメイン攻撃

```
サブドメインからの CSRF 攻撃:

  シナリオ:
  → evil.example.com が攻撃者に制御されている
  → app.example.com を攻撃したい

  ① SameSite Cookie:
     → evil.example.com と app.example.com は Same-Site
     → SameSite=Lax でも Cookie が送信される場合がある

  ② Cookie Injection:
     → evil.example.com から
        Set-Cookie: session=...; domain=.example.com
     → app.example.com の Cookie を上書き可能

  対策:
  → __Host- プレフィックスを使用
     Set-Cookie: __Host-session=abc123; Secure; Path=/
     → Domain 属性を設定できない
     → Secure 属性が必須
     → Path=/ が必須
     → サブドメインからの Cookie 上書きを防止
```

```typescript
// __Host- プレフィックスの使用
res.cookie('__Host-session', sessionId, {
  httpOnly: true,
  secure: true,         // 必須
  sameSite: 'lax',
  path: '/',            // 必須（Path=/ のみ）
  // domain: 設定不可   // __Host- では domain を設定できない
});

// __Secure- プレフィックスの使用（制約が少ない）
res.cookie('__Secure-session', sessionId, {
  httpOnly: true,
  secure: true,         // 必須
  sameSite: 'lax',
  path: '/',
  domain: '.example.com', // 設定可能
});
```

### 8.3 JSON CSRF の詳細

```
JSON CSRF 攻撃の詳細:

  通常、JSON リクエストは CSRF に強い:
  → Content-Type: application/json は「非単純リクエスト」
  → CORS プリフライト（OPTIONS）が発行される
  → サーバーが許可しなければリクエストがブロックされる

  しかし以下の場合に攻撃可能:
  ┌──────────────────────────────────────────────────┐
  │ ① サーバーが Content-Type を無視して JSON パース   │
  │                                                   │
  │ <form enctype="text/plain" method="POST"           │
  │       action="https://target.com/api/transfer">    │
  │   <input name='{"amount":1000,"to":"evil","x":"'   │
  │          value='"}' />                             │
  │ </form>                                            │
  │                                                   │
  │ 送信される body:                                    │
  │ {"amount":1000,"to":"evil","x":"="}                │
  │ → text/plain だが JSON としてパースされる            │
  │                                                   │
  │ 対策:                                              │
  │ → Content-Type: application/json を厳格にチェック   │
  │ → text/plain を拒否                                │
  └──────────────────────────────────────────────────┘
```

### 8.4 ファイルアップロード CSRF

```
ファイルアップロードと CSRF:

  <form enctype="multipart/form-data"> は
  CORS プリフライトなしで送信可能（単純リクエスト）

  攻撃者のサイトから:
  <form action="https://target.com/api/upload" method="POST"
        enctype="multipart/form-data">
    <input type="hidden" name="filename" value="malware.exe" />
    <input type="hidden" name="content" value="..." />
  </form>

  対策:
  → CSRF トークンを multipart/form-data に含める
  → Origin ヘッダー検証
  → ファイルアップロード API にも CSRF 保護を適用
```

---

## 9. アンチパターン

### 9.1 よくある間違い

```
CSRF 対策のアンチパターン:

  ✗ アンチパターン①: Referer ヘッダーのみに依存
  ┌──────────────────────────────────────────────────┐
  │ // 危険: Referer は省略・偽装される場合がある       │
  │ if (req.headers.referer?.includes('mysite.com')) { │
  │   // OK                                           │
  │ }                                                  │
  │ → Referrer-Policy: no-referrer で省略される        │
  │ → 一部のプロキシで削除される                        │
  │ → 補助的な防御としてのみ使用すべき                  │
  └──────────────────────────────────────────────────┘

  ✗ アンチパターン②: GET で状態変更
  ┌──────────────────────────────────────────────────┐
  │ // 危険: GET は SameSite=Lax でも Cookie 送信       │
  │ app.get('/api/delete-user/:id', deleteUser);       │
  │ app.get('/api/transfer', handleTransfer);          │
  │                                                    │
  │ → <img src="/api/delete-user/123"> で攻撃可能      │
  │ → 状態変更は必ず POST/PUT/DELETE を使用             │
  │ → RFC 7231: GET は安全なメソッド（副作用なし）      │
  └──────────────────────────────────────────────────┘

  ✗ アンチパターン③: CSRF トークンの予測可能な生成
  ┌──────────────────────────────────────────────────┐
  │ // 危険: Math.random() は暗号的に安全ではない       │
  │ const token = Math.random().toString(36);          │
  │                                                    │
  │ // 危険: タイムスタンプベースは推測可能              │
  │ const token = Date.now().toString(16);             │
  │                                                    │
  │ // 正しい: crypto.randomBytes() を使用              │
  │ const token = crypto.randomBytes(32).toString('hex');│
  └──────────────────────────────────────────────────┘

  ✗ アンチパターン④: CSRF トークンを URL に含める
  ┌──────────────────────────────────────────────────┐
  │ // 危険: URL はログ・Referer ヘッダーで漏洩         │
  │ <a href="/api/action?csrf=token123">              │
  │                                                    │
  │ → アクセスログに記録される                          │
  │ → 外部サイトへのリンクの Referer で漏洩             │
  │ → ブラウザ履歴に残る                               │
  │ → トークンはヘッダーまたは POST body で送信すべき    │
  └──────────────────────────────────────────────────┘
```

---

## 10. 演習

### 演習1: 基礎 - CSRF 攻撃のシミュレーション

```
【演習1】CSRF 攻撃のシミュレーション

目的: CSRF 攻撃の仕組みを実際に体験し、危険性を理解する

手順:
1. Express で簡単な銀行アプリを作成
   - POST /transfer (送金 API、Cookie 認証)
   - GET /balance (残高確認)
   - SameSite 属性を設定しない Cookie を使用

2. 別のポートで攻撃者サイトを作成
   - 自動送信フォームを設置
   - 銀行アプリに POST リクエストを送信

3. 以下を確認:
   - 攻撃者サイトにアクセスすると送金が実行されること
   - Cookie が自動送信されることを DevTools で確認
   - SameSite=Lax を設定すると攻撃がブロックされること

4. 発展:
   - img タグを使った GET CSRF を試す
   - JSON CSRF を試す（enctype="text/plain"）

評価基準:
  □ CSRF 攻撃が成功することを確認
  □ SameSite Cookie で防御できることを確認
  □ なぜ攻撃が成立するか説明できる
```

### 演習2: 応用 - CSRF 防御の実装

```
【演習2】CSRF 防御の実装

目的: 主要な CSRF 防御パターンを実装し、比較する

手順:
1. Synchronizer Token Pattern を実装
   - Redis にトークンを保存
   - フォームに hidden field として埋め込み
   - POST 時にトークンを検証
   - タイミング攻撃対策（timingSafeEqual）

2. Double Submit Cookie を実装
   - HMAC 署名付きトークンを使用
   - Cookie とヘッダーの両方にトークンを設定
   - サーバー側で比較検証

3. 両方の実装で以下をテスト:
   - 正常なリクエストが通ること
   - トークンなしのリクエストが拒否されること
   - 不正なトークンのリクエストが拒否されること
   - ブラウザの「戻る」ボタンでの挙動

4. 比較レポートを作成:
   - 実装の複雑さ
   - スケーラビリティ
   - UX への影響

評価基準:
  □ 両方のパターンが正しく動作する
  □ エッジケースをテストしている
  □ 比較レポートが適切
```

### 演習3: 発展 - Next.js での包括的 CSRF 対策

```
【演習3】Next.js での包括的 CSRF 対策

目的: Next.js App Router で多層防御を実装する

手順:
1. Next.js プロジェクトをセットアップ
   - App Router を使用
   - Prisma + SQLite でデータベース設定
   - Auth.js でセッション認証

2. 以下の CSRF 対策を実装:
   a. Middleware での Origin ヘッダー検証
   b. Double Submit Cookie（API Routes 用）
   c. Server Actions の CSRF 保護を確認
   d. Cookie に __Host- プレフィックスを使用

3. テストスイートを作成:
   - クロスオリジンリクエストが拒否されること
   - 同一オリジンリクエストが成功すること
   - トークン検証が機能すること
   - Server Actions が安全であること

4. セキュリティ監査:
   - OWASP CSRF テストガイドに沿って検証
   - 各防御レイヤーの効果を確認
   - 改善点を文書化

評価基準:
  □ 多層防御が正しく機能する
  □ テストが網羅的
  □ パフォーマンスへの影響を測定している
  □ セキュリティ監査レポートを作成
```

---

## 11. FAQ・トラブルシューティング

### Q1: SPA で CSRF 対策は本当に必要ですか？

```
A: Cookie ベースの認証を使用しているなら必要です。

  JWT を localStorage に保存し Authorization ヘッダーで送信する場合:
  → 不要（自動送信されないため）

  httpOnly Cookie でセッション管理する場合:
  → 必要（ブラウザが自動送信するため）

  ただし SPA でも SameSite=Lax を設定していれば、
  追加のトークン検証なしでもほとんどの攻撃を防げます。
  高セキュリティ要件の場合は Double Submit Cookie を追加してください。
```

### Q2: CSRF トークンが 403 エラーを返す場合のデバッグ方法

```
A: 以下の順序で確認してください:

  1. トークンの送信確認:
     → DevTools の Network タブでリクエストヘッダーを確認
     → X-CSRF-Token ヘッダーまたは _csrf フィールドがあるか

  2. Cookie の確認:
     → DevTools の Application > Cookies で CSRF Cookie を確認
     → SameSite 属性が正しいか
     → Secure 属性と HTTPS の整合性

  3. サーバーログの確認:
     → 受信したトークンと保存されたトークンの比較
     → セッション ID が正しいか

  4. よくある原因:
     → ページキャッシュにより古いトークンを使用
     → 複数タブでトークンが上書きされた
     → セッション切れでトークンも無効化
     → httpOnly Cookie でクライアントから読めない
```

### Q3: Webhook エンドポイントで CSRF 対策をスキップしても安全ですか？

```
A: はい、ただし別の認証メカニズムが必要です。

  Webhook の場合:
  → 外部サービスからのリクエストなので CSRF トークンは使えない
  → 代替手段:
     ① HMAC 署名検証（Stripe, GitHub 等）
     ② IP ホワイトリスト
     ③ Webhook Secret による検証
     ④ mTLS（相互TLS認証）

  実装例（Stripe Webhook）:
  const sig = req.headers['stripe-signature'];
  const event = stripe.webhooks.constructEvent(
    req.body, sig, process.env.STRIPE_WEBHOOK_SECRET
  );
```

### Q4: モバイルアプリのバックエンドで CSRF 対策は必要ですか？

```
A: 通常は不要です。

  理由:
  → モバイルアプリはブラウザではない
  → Cookie の自動送信は発生しない
  → API トークンで認証するのが一般的
  → CSRF はブラウザの Cookie 自動送信を悪用する攻撃

  ただし:
  → WebView を使用する場合は Cookie が使われる可能性
  → ブラウザと API を共有する場合は CSRF 対策が必要
  → モバイル専用 API なら Authorization ヘッダー認証で OK
```

---

## 12. パフォーマンスに関する考察

```
CSRF 対策のパフォーマンス影響:

  ┌──────────────────────┬───────────────┬──────────────┐
  │ 防御パターン          │ レイテンシ影響 │ メモリ影響    │
  ├──────────────────────┼───────────────┼──────────────┤
  │ SameSite Cookie      │ なし（0ms）    │ なし         │
  │ Origin 検証           │ 極小（<1ms）   │ なし         │
  │ Double Submit Cookie │ 極小（<1ms）   │ なし         │
  │ Synchronizer Token   │ 小（1-5ms）    │ Redis 依存   │
  │ Encrypted Token      │ 小（1-3ms）    │ なし         │
  └──────────────────────┴───────────────┴──────────────┘

  Synchronizer Token の Redis アクセス:
  → セッションストアと同じ Redis を使えばコネクション共有可能
  → Redis のレイテンシは通常 0.5-1ms
  → パイプライン処理でセッション読取と同時実行可能

  最適化:
  → Per-Session Token にすればリクエストごとの書込みが不要
  → Double Submit Cookie はサーバー状態不要で最速
  → SameSite Cookie は CPU/メモリ影響ゼロ
```

---

## まとめ

| 防御方法 | 効果 | 状態管理 | SPA 適合性 | 推奨用途 |
|---------|------|---------|-----------|---------|
| SameSite=Lax | 高 | 不要 | 自動 | 全アプリ必須 |
| Synchronizer Token | 最高 | 必要（Redis等） | 要工夫 | 高セキュリティ |
| Double Submit Cookie | 高 | 不要 | 適合 | SPA + Cookie 認証 |
| Signed Double Submit | 最高 | 不要 | 適合 | サブドメインリスク時 |
| Origin 検証 | 中 | 不要 | 適合 | 補助防御 |
| Custom Header | 高 | 不要 | 最適 | API 限定 |
| Content-Type 検証 | 中 | 不要 | 適合 | JSON API |

---

## 次に読むべきガイド
→ [[../02-token-auth/00-jwt-deep-dive.md]] — JWT 詳解

---

## 参考文献
1. OWASP. "Cross-Site Request Forgery Prevention Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. MDN. "SameSite cookies." developer.mozilla.org, 2024.
3. Next.js. "Server Actions and Mutations." nextjs.org/docs, 2024.
4. RFC 7231 §4.2.1. "Safe Methods." IETF, 2014.
5. Barth, A. "Robust Defenses for Cross-Site Request Forgery." ACM CCS, 2008.
6. RFC 6265bis. "Cookies: HTTP State Management Mechanism." IETF, 2024.
7. Chromium. "SameSite Cookies Explained." web.dev, 2024.
8. OWASP. "Testing for CSRF (WSTG-SESS-05)." owasp.org, 2024.
