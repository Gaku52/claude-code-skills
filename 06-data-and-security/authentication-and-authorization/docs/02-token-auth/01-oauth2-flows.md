# OAuth 2.0 フロー

> OAuth 2.0 は「認可の委譲」のための標準プロトコル。Authorization Code + PKCE、Client Credentials、Device Code、Implicit（非推奨）の各フローの仕組み、適用場面、セキュリティ上の注意点を網羅的に解説する。

## この章で学ぶこと

- [ ] OAuth 2.0 の役割とアクターを理解する
- [ ] 各フローの仕組みと適用場面を把握する
- [ ] PKCE による SPA/モバイル向けセキュリティ強化を実装できる
- [ ] トークンのライフサイクル管理（発行・リフレッシュ・失効）を実装できる
- [ ] OAuth 2.0 のセキュリティ脅威と防御策を理解する
- [ ] 認可サーバーの内部実装を理解する

---

## 1. OAuth 2.0 の基本概念

```
OAuth 2.0 の役割:
  「第三者アプリケーションにユーザーのリソースへの
   限定的なアクセスを許可する仕組み」

  例: 「GitHub のリポジトリ一覧を MyApp に許可する」
  → ユーザーは GitHub のパスワードを MyApp に渡さない
  → MyApp は必要な権限（スコープ）のみ取得

4つのアクター:

  ┌─────────────────────────────────────────┐
  │                                         │
  │  Resource Owner（リソースオーナー）        │
  │  → ユーザー本人                           │
  │                                         │
  │  Client（クライアント）                    │
  │  → アクセスを要求するアプリケーション        │
  │  → MyApp                                │
  │                                         │
  │  Authorization Server（認可サーバー）      │
  │  → アクセス許可を発行するサーバー            │
  │  → GitHub の OAuth サーバー               │
  │                                         │
  │  Resource Server（リソースサーバー）        │
  │  → 保護されたリソースを持つサーバー          │
  │  → GitHub API                           │
  │                                         │
  └─────────────────────────────────────────┘

フローの選択:

  アプリ種別              │ 推奨フロー
  ──────────────────────┼──────────────────────
  Web アプリ（サーバーあり）│ Authorization Code
  SPA（サーバーなし）      │ Authorization Code + PKCE
  モバイルアプリ           │ Authorization Code + PKCE
  サーバー間通信           │ Client Credentials
  IoT / テレビ            │ Device Code
  SPA（レガシー）         │ Implicit（非推奨）
```

### 1.1 OAuth 2.0 の歴史と設計思想

```
OAuth の進化:

  OAuth 1.0（2007年）:
    → 署名ベース（HMAC-SHA1）
    → 各リクエストに署名が必要
    → 実装が複雑（正規化、署名生成）
    → TLS が必須ではなかった

  OAuth 2.0（2012年、RFC 6749）:
    → Bearer トークンベース
    → TLS 必須（署名を簡素化）
    → 複数のグラントタイプ（フロー）
    → スコープによる権限制御
    → リフレッシュトークンの導入

  OAuth 2.1（策定中）:
    → PKCE の必須化（全フロー）
    → Implicit フローの廃止
    → Resource Owner Password の廃止
    → リフレッシュトークンのローテーション推奨
    → Bearer トークンの sender-constraint 推奨

  設計上の重要な判断:
    ① OAuth 2.0 は「認可」プロトコル（認証ではない）
       → 「このアプリに GitHub リポジトリへのアクセスを許可」
       → ユーザーの身元確認は OpenID Connect が担う

    ② アクセストークンはクライアントにとって「不透明」
       → クライアントはトークンの中身を解析してはいけない
       → リソースサーバーのみがトークンを検証する
       → JWT を使うかどうかは認可サーバーの実装次第

    ③ フロントチャネル vs バックチャネル
       → フロントチャネル: ブラウザのリダイレクト（傍受リスク）
       → バックチャネル: サーバー間の直接通信（安全）
       → 認可コードはフロントチャネルで受け取り
       → トークン交換はバックチャネルで実行
```

### 1.2 クライアントの分類

```
OAuth 2.0 におけるクライアントの分類:

  Confidential Client（秘密クライアント）:
    → client_secret を安全に保持できる
    → サーバーサイドアプリケーション
    → バックエンド Web アプリ
    → client_secret をトークン交換時に使用

  Public Client（公開クライアント）:
    → client_secret を安全に保持できない
    → SPA（ブラウザ上の JavaScript）
    → モバイルアプリ（デコンパイル可能）
    → デスクトップアプリ
    → PKCE を使用して保護

  ┌───────────────────────────────────────────────────┐
  │                                                   │
  │  Confidential Client:                              │
  │  ┌─────────────┐    client_secret    ┌──────────┐ │
  │  │ Web Server  │───────────────────>│ Auth     │ │
  │  │ (backend)   │    安全に送信可能    │ Server   │ │
  │  └─────────────┘                    └──────────┘ │
  │                                                   │
  │  Public Client:                                    │
  │  ┌─────────────┐    client_secret    ┌──────────┐ │
  │  │ SPA / Mobile│───────────× ───────│ Auth     │ │
  │  │ (frontend)  │  ソースから漏洩する   │ Server   │ │
  │  └─────────────┘                    └──────────┘ │
  │                    代わりに PKCE を使用            │
  │                                                   │
  └───────────────────────────────────────────────────┘

  クライアント登録時の設定:
    → client_id: 公開識別子
    → client_secret: 秘密鍵（Confidential のみ）
    → redirect_uris: 許可されたリダイレクト先
    → grant_types: 許可されたフロー
    → scopes: 要求可能なスコープ
    → token_endpoint_auth_method: 認証方式
      → client_secret_basic: Basic 認証
      → client_secret_post: POST パラメータ
      → private_key_jwt: JWT ベース認証
      → none: Public Client
```

---

## 2. Authorization Code フロー

```
Authorization Code フロー（最も安全、最も一般的）:

  ユーザー    クライアント   認可サーバー    リソースサーバー
    │           │            │               │
    │ ログイン   │            │               │
    │──────────>│            │               │
    │           │ ① 認可リクエスト             │
    │           │───────────>│               │
    │           │            │               │
    │ ② ログイン画面          │               │
    │<──────────────────────│               │
    │           │            │               │
    │ ③ 認証 + 権限承認      │               │
    │──────────────────────>│               │
    │           │            │               │
    │ ④ redirect_uri +      │               │
    │   authorization_code  │               │
    │──────────>│            │               │
    │           │            │               │
    │           │ ⑤ code → token 交換        │
    │           │───────────>│               │
    │           │            │               │
    │           │ ⑥ access_token +           │
    │           │   refresh_token             │
    │           │<───────────│               │
    │           │            │               │
    │           │ ⑦ API リクエスト             │
    │           │ + Bearer access_token       │
    │           │───────────────────────────>│
    │           │            │               │
    │           │ ⑧ リソース                  │
    │           │<───────────────────────────│
    │ レスポンス  │            │               │
    │<──────────│            │               │
```

### 2.1 各ステップの詳細

```
① 認可リクエストの内部:

  GET /authorize?
    response_type=code          ← 認可コードを要求
    &client_id=my-app-id        ← クライアント識別子
    &redirect_uri=https://myapp.com/callback  ← コールバック先
    &scope=read:user repo       ← 要求する権限
    &state=xyzabc123            ← CSRF 防御用ランダム値
    &code_challenge=E9Melhoa... ← PKCE チャレンジ（推奨）
    &code_challenge_method=S256 ← PKCE メソッド

  各パラメータの意味と重要性:

    response_type:
      → "code" = Authorization Code フロー
      → "token" = Implicit フロー（非推奨）

    state:
      → CSRF 防御の要
      → 暗号的にランダムな値（32バイト以上）
      → セッションに紐づけて保存
      → コールバック時に一致確認
      → 一致しなければ → 攻撃の可能性

    redirect_uri:
      → 事前登録された URI と完全一致が必須
      → オープンリダイレクタの防止
      → ワイルドカード禁止（セキュリティリスク）
      → localhost は開発時のみ許可

④ 認可レスポンス:

  HTTP/1.1 302 Found
  Location: https://myapp.com/callback
    ?code=SplxlOBeZQQYbYS6WxSbIA    ← 認可コード（短寿命）
    &state=xyzabc123                 ← 送信した state がそのまま返る

  認可コードの特性:
    → 有効期限: 10分以内（推奨: 30秒〜1分）
    → 一度使い切り（使用後は無効化）
    → client_id に紐づく
    → redirect_uri に紐づく
    → 漏洩しても client_secret なしでは使えない
      （Confidential Client の場合）

⑤ トークン交換リクエスト:

  POST /token
  Content-Type: application/x-www-form-urlencoded
  Authorization: Basic <base64(client_id:client_secret)>

  grant_type=authorization_code
  &code=SplxlOBeZQQYbYS6WxSbIA
  &redirect_uri=https://myapp.com/callback
  &code_verifier=dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk

  認可サーバーの検証:
    ✓ client_id と client_secret の一致
    ✓ code が有効（未使用、未期限切れ）
    ✓ code が client_id に紐づくか
    ✓ redirect_uri が認可リクエスト時と同一か
    ✓ code_verifier のハッシュが code_challenge と一致（PKCE）
    ✓ すべて OK → トークン発行

⑥ トークンレスポンス:

  HTTP/1.1 200 OK
  Content-Type: application/json
  Cache-Control: no-store
  Pragma: no-cache

  {
    "access_token": "eyJhbGciOiJSUzI1NiIs...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "8xLOxBtZp8",
    "scope": "read:user repo"
  }

  重要なレスポンスヘッダー:
    → Cache-Control: no-store（トークンをキャッシュしない）
    → Pragma: no-cache（HTTP/1.0 互換）
```

```typescript
// Authorization Code フロー 完全実装

// ① 認可リクエスト（クライアント → 認可サーバー）
function getAuthorizationUrl(state: string): string {
  const params = new URLSearchParams({
    response_type: 'code',
    client_id: process.env.GITHUB_CLIENT_ID!,
    redirect_uri: 'https://myapp.com/callback',
    scope: 'read:user repo',
    state, // CSRF 防御用ランダム値
  });

  return `https://github.com/login/oauth/authorize?${params}`;
}

// ④⑤ コールバック処理（認可コード → トークン交換）
async function handleCallback(code: string, state: string) {
  // state の検証（CSRF 防御）
  if (state !== storedState) {
    throw new Error('Invalid state parameter');
  }

  // ⑤ トークン交換（サーバー側で実行 → client_secret を安全に使用）
  const tokenResponse = await fetch('https://github.com/login/oauth/access_token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({
      client_id: process.env.GITHUB_CLIENT_ID,
      client_secret: process.env.GITHUB_CLIENT_SECRET, // サーバーのみ
      code,
      redirect_uri: 'https://myapp.com/callback',
    }),
  });

  const { access_token, refresh_token, scope, token_type } = await tokenResponse.json();

  // ⑦ リソース取得
  const userResponse = await fetch('https://api.github.com/user', {
    headers: { Authorization: `Bearer ${access_token}` },
  });

  return userResponse.json();
}
```

### 2.2 認可サーバーの内部実装

```typescript
// 認可サーバーの内部実装（Express）
import express from 'express';
import crypto from 'crypto';

interface AuthorizationCode {
  code: string;
  clientId: string;
  userId: string;
  redirectUri: string;
  scope: string;
  codeChallenge?: string;
  codeChallengeMethod?: string;
  createdAt: Date;
  used: boolean;
}

interface TokenRecord {
  accessToken: string;
  refreshToken: string;
  clientId: string;
  userId: string;
  scope: string;
  accessTokenExpiresAt: Date;
  refreshTokenExpiresAt: Date;
  revoked: boolean;
}

class AuthorizationServer {
  private codes: Map<string, AuthorizationCode> = new Map();
  private tokens: Map<string, TokenRecord> = new Map();
  private refreshTokenIndex: Map<string, string> = new Map(); // refreshToken → accessToken

  // ① 認可エンドポイント
  async handleAuthorize(req: express.Request, res: express.Response) {
    const {
      response_type,
      client_id,
      redirect_uri,
      scope,
      state,
      code_challenge,
      code_challenge_method,
    } = req.query as Record<string, string>;

    // バリデーション
    if (response_type !== 'code') {
      return res.status(400).json({ error: 'unsupported_response_type' });
    }

    // クライアント検証
    const client = await this.getClient(client_id);
    if (!client) {
      return res.status(400).json({ error: 'invalid_client' });
    }

    // redirect_uri の完全一致確認
    if (!client.redirectUris.includes(redirect_uri)) {
      // redirect_uri が不正な場合はリダイレクトしてはいけない
      // （オープンリダイレクタ防止）
      return res.status(400).json({ error: 'invalid_redirect_uri' });
    }

    // スコープ検証
    const requestedScopes = scope.split(' ');
    const allowedScopes = requestedScopes.filter(s => client.allowedScopes.includes(s));

    // ユーザー認証済みか確認（未認証ならログイン画面へ）
    const user = req.session?.user;
    if (!user) {
      // ログイン画面にリダイレクト（認可パラメータを保持）
      return res.redirect(`/login?return_to=${encodeURIComponent(req.originalUrl)}`);
    }

    // 同意画面を表示（または以前の同意を確認）
    const existingConsent = await this.getConsent(user.id, client_id, allowedScopes);
    if (!existingConsent) {
      return res.render('consent', {
        client,
        scopes: allowedScopes,
        state,
        // 同意フォーム送信先
      });
    }

    // 認可コード生成
    const code = crypto.randomBytes(32).toString('hex');
    this.codes.set(code, {
      code,
      clientId: client_id,
      userId: user.id,
      redirectUri: redirect_uri,
      scope: allowedScopes.join(' '),
      codeChallenge: code_challenge,
      codeChallengeMethod: code_challenge_method,
      createdAt: new Date(),
      used: false,
    });

    // 認可コードは短寿命（30秒後に自動削除）
    setTimeout(() => this.codes.delete(code), 30 * 1000);

    // リダイレクト
    const callbackUrl = new URL(redirect_uri);
    callbackUrl.searchParams.set('code', code);
    if (state) callbackUrl.searchParams.set('state', state);

    res.redirect(302, callbackUrl.toString());
  }

  // ⑤ トークンエンドポイント
  async handleToken(req: express.Request, res: express.Response) {
    const { grant_type } = req.body;

    // Cache-Control ヘッダー（RFC 6749 Section 5.1）
    res.set('Cache-Control', 'no-store');
    res.set('Pragma', 'no-cache');

    switch (grant_type) {
      case 'authorization_code':
        return this.handleAuthorizationCodeGrant(req, res);
      case 'refresh_token':
        return this.handleRefreshTokenGrant(req, res);
      case 'client_credentials':
        return this.handleClientCredentialsGrant(req, res);
      default:
        return res.status(400).json({ error: 'unsupported_grant_type' });
    }
  }

  private async handleAuthorizationCodeGrant(
    req: express.Request,
    res: express.Response,
  ) {
    const { code, redirect_uri, code_verifier } = req.body;
    const { clientId, clientSecret } = this.extractClientCredentials(req);

    // クライアント認証
    const client = await this.authenticateClient(clientId, clientSecret);
    if (!client) {
      return res.status(401).json({ error: 'invalid_client' });
    }

    // 認可コード検証
    const authCode = this.codes.get(code);
    if (!authCode) {
      return res.status(400).json({ error: 'invalid_grant', error_description: 'Code not found or expired' });
    }

    // 使用済みチェック
    if (authCode.used) {
      // 認可コード再利用 = 攻撃の可能性
      // → 該当コードで発行された全トークンを無効化
      await this.revokeTokensByCode(code);
      return res.status(400).json({ error: 'invalid_grant', error_description: 'Code already used' });
    }

    // client_id 一致確認
    if (authCode.clientId !== clientId) {
      return res.status(400).json({ error: 'invalid_grant' });
    }

    // redirect_uri 一致確認
    if (authCode.redirectUri !== redirect_uri) {
      return res.status(400).json({ error: 'invalid_grant' });
    }

    // PKCE 検証
    if (authCode.codeChallenge) {
      if (!code_verifier) {
        return res.status(400).json({ error: 'invalid_grant', error_description: 'code_verifier required' });
      }

      const valid = this.verifyPKCE(
        code_verifier,
        authCode.codeChallenge,
        authCode.codeChallengeMethod || 'plain',
      );

      if (!valid) {
        return res.status(400).json({ error: 'invalid_grant', error_description: 'PKCE verification failed' });
      }
    }

    // コードを使用済みにマーク
    authCode.used = true;

    // トークン発行
    const tokens = await this.issueTokens(clientId, authCode.userId, authCode.scope);

    res.json({
      access_token: tokens.accessToken,
      token_type: 'Bearer',
      expires_in: 3600,
      refresh_token: tokens.refreshToken,
      scope: authCode.scope,
    });
  }

  // PKCE 検証
  private verifyPKCE(
    verifier: string,
    challenge: string,
    method: string,
  ): boolean {
    if (method === 'S256') {
      const hash = crypto.createHash('sha256').update(verifier).digest();
      const computed = hash
        .toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=+$/, '');
      return computed === challenge;
    }
    if (method === 'plain') {
      return verifier === challenge;
    }
    return false;
  }

  // クライアント認証情報の抽出
  private extractClientCredentials(req: express.Request): {
    clientId: string;
    clientSecret: string | null;
  } {
    // Basic 認証ヘッダーから取得
    const authHeader = req.headers.authorization;
    if (authHeader?.startsWith('Basic ')) {
      const decoded = Buffer.from(authHeader.slice(6), 'base64').toString();
      const [clientId, clientSecret] = decoded.split(':');
      return { clientId, clientSecret };
    }

    // POST ボディから取得
    return {
      clientId: req.body.client_id,
      clientSecret: req.body.client_secret || null,
    };
  }

  // トークン発行
  private async issueTokens(
    clientId: string,
    userId: string,
    scope: string,
  ): Promise<{ accessToken: string; refreshToken: string }> {
    const accessToken = crypto.randomBytes(32).toString('hex');
    const refreshToken = crypto.randomBytes(48).toString('hex');

    const record: TokenRecord = {
      accessToken,
      refreshToken,
      clientId,
      userId,
      scope,
      accessTokenExpiresAt: new Date(Date.now() + 3600 * 1000),    // 1時間
      refreshTokenExpiresAt: new Date(Date.now() + 30 * 24 * 3600 * 1000), // 30日
      revoked: false,
    };

    this.tokens.set(accessToken, record);
    this.refreshTokenIndex.set(refreshToken, accessToken);

    return { accessToken, refreshToken };
  }
}
```

---

## 3. PKCE（Proof Key for Code Exchange）

```
PKCE の必要性:

  SPA / モバイルアプリの問題:
  → client_secret を安全に保持できない
  → 認可コードの横取り攻撃（Authorization Code Interception）
  → 悪意あるアプリが redirect_uri を傍受して code を取得

  PKCE の仕組み:
  → code_verifier: クライアントが生成するランダム文字列
  → code_challenge: code_verifier のハッシュ（SHA-256）
  → 認可リクエスト時に challenge を送信
  → トークン交換時に verifier を送信
  → サーバーが verifier をハッシュして challenge と比較

  PKCE フロー:
  ① code_verifier = ランダム文字列（43〜128文字）
  ② code_challenge = BASE64URL(SHA256(code_verifier))
  ③ 認可リクエスト: code_challenge + code_challenge_method=S256
  ④ コールバック: authorization_code を受信
  ⑤ トークン交換: code + code_verifier を送信
  ⑥ サーバー: SHA256(code_verifier) == code_challenge を検証
```

### 3.1 PKCE が防ぐ攻撃

```
認可コード横取り攻撃（PKCE なし）:

  正規アプリ        悪意あるアプリ     認可サーバー
    │                  │                │
    │ 認可リクエスト     │                │
    │──────────────────────────────────>│
    │                  │                │
    │ 302 Redirect     │                │
    │ ?code=abc123     │                │
    │<──────────────────────────────────│
    │                  │                │
    │ ★ 悪意あるアプリが │                │
    │   code を傍受!    │                │
    │─────────────────>│                │
    │                  │                │
    │                  │ code → token   │
    │                  │──────────────>│
    │                  │                │
    │                  │ access_token   │
    │                  │<──────────────│
    │                  │                │
    │                  │ ★ 不正にリソースアクセス!

  傍受の方法:
  → カスタム URL スキームの乗っ取り（モバイル）
  → ブラウザ拡張の悪用
  → OS レベルのリダイレクト傍受

認可コード横取り攻撃（PKCE あり）:

  正規アプリ        悪意あるアプリ     認可サーバー
    │                  │                │
    │ verifier を生成    │                │
    │ challenge = SHA256(verifier)       │
    │                  │                │
    │ 認可リクエスト     │                │
    │ + challenge      │                │
    │──────────────────────────────────>│
    │                  │                │
    │ 302 Redirect     │                │
    │ ?code=abc123     │                │
    │<──────────────────────────────────│
    │                  │                │
    │ ★ 悪意あるアプリが │                │
    │   code を傍受     │                │
    │─────────────────>│                │
    │                  │                │
    │                  │ code → token   │
    │                  │ verifier = ??? │
    │                  │──────────────>│
    │                  │                │
    │                  │ ✗ PKCE 検証失敗!│
    │                  │ verifier 不明   │
    │                  │<──────────────│
    │                  │                │
    │ 正規アプリ:        │                │
    │ code + verifier   │                │
    │──────────────────────────────────>│
    │                  │ ✓ PKCE 検証成功 │
    │ access_token     │                │
    │<──────────────────────────────────│
```

```typescript
// PKCE 実装

// code_verifier と code_challenge の生成
async function generatePKCE(): Promise<{ verifier: string; challenge: string }> {
  // code_verifier: 43〜128文字の暗号ランダム文字列
  const verifier = base64URLEncode(crypto.getRandomValues(new Uint8Array(32)));

  // code_challenge: SHA-256 ハッシュの Base64URL エンコード
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const challenge = base64URLEncode(new Uint8Array(hashBuffer));

  return { verifier, challenge };
}

function base64URLEncode(buffer: Uint8Array): string {
  return btoa(String.fromCharCode(...buffer))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

// ① 認可リクエスト（PKCE 付き）
async function startAuthFlow() {
  const { verifier, challenge } = await generatePKCE();
  const state = crypto.randomUUID();

  // verifier と state をセッションに保存
  sessionStorage.setItem('pkce_verifier', verifier);
  sessionStorage.setItem('oauth_state', state);

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: 'my-spa-client-id',
    redirect_uri: 'https://myapp.com/callback',
    scope: 'openid profile email',
    state,
    code_challenge: challenge,
    code_challenge_method: 'S256',
  });

  window.location.href = `https://auth.example.com/authorize?${params}`;
}

// ⑤ コールバック（PKCE + トークン交換）
async function handlePKCECallback(code: string, state: string) {
  // state 検証
  const storedState = sessionStorage.getItem('oauth_state');
  if (state !== storedState) throw new Error('Invalid state');

  const verifier = sessionStorage.getItem('pkce_verifier');

  // トークン交換（client_secret 不要）
  const response = await fetch('https://auth.example.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: 'https://myapp.com/callback',
      client_id: 'my-spa-client-id',
      code_verifier: verifier!,  // client_secret の代わり
    }),
  });

  const tokens = await response.json();
  // { access_token, refresh_token, id_token, token_type, expires_in }

  // クリーンアップ
  sessionStorage.removeItem('pkce_verifier');
  sessionStorage.removeItem('oauth_state');

  return tokens;
}
```

### 3.2 PKCE の内部：SHA-256 による検証

```
PKCE の暗号学的な安全性:

  code_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

  code_challenge = BASE64URL(SHA256(code_verifier))
                 = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

  なぜ安全か:
  ① SHA-256 は一方向関数
     → challenge から verifier を逆算することは計算上不可能
     → 2^256 の探索空間 ≈ 1.16 × 10^77

  ② verifier は 43〜128 文字（RFC 7636）
     → 十分なエントロピー
     → ブルートフォース不可能

  ③ verifier はネットワークに流れない（バックチャネルのみ）
     → フロントチャネルに challenge のみ
     → challenge を傍受しても verifier は不明

  plain メソッド（非推奨）:
    → code_challenge = code_verifier そのもの
    → 傍受されると意味がない
    → SHA-256 をサポートしないクライアント向け
    → OAuth 2.1 では S256 のみ必須
```

---

## 4. Client Credentials フロー

```
Client Credentials フロー:

  用途: サーバー間通信（ユーザーが介在しない）
  例: バッチ処理、マイクロサービス間、バックエンド API

  クライアント          認可サーバー         リソースサーバー
    │                   │                   │
    │ client_id +       │                   │
    │ client_secret     │                   │
    │──────────────────>│                   │
    │                   │                   │
    │ access_token      │                   │
    │<──────────────────│                   │
    │                   │                   │
    │ Bearer token      │                   │
    │──────────────────────────────────────>│
    │                   │                   │
    │ リソース           │                   │
    │<──────────────────────────────────────│

  特徴:
  → ユーザーの同意画面なし
  → client_id + client_secret で認証
  → refresh_token は発行されない
  → トークンの有効期限で管理
```

### 4.1 Client Credentials の実装パターン

```typescript
// Client Credentials フロー（基本）
async function getServiceToken(): Promise<string> {
  const response = await fetch('https://auth.example.com/token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
      // Basic 認証で client_id:client_secret を送信
      'Authorization': `Basic ${btoa(`${CLIENT_ID}:${CLIENT_SECRET}`)}`,
    },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      scope: 'read:data write:data',
    }),
  });

  const { access_token } = await response.json();
  return access_token;
}

// トークンキャッシュ付きの実装（本番向け）
class ServiceTokenManager {
  private token: string | null = null;
  private expiresAt: number = 0;
  private refreshPromise: Promise<string> | null = null;
  private readonly bufferSeconds = 60; // 期限の60秒前に更新

  constructor(
    private clientId: string,
    private clientSecret: string,
    private tokenUrl: string,
    private scope: string,
  ) {}

  async getToken(): Promise<string> {
    // キャッシュが有効ならそのまま返す
    if (this.token && Date.now() < this.expiresAt - this.bufferSeconds * 1000) {
      return this.token;
    }

    // 同時リクエスト時に重複リフレッシュを防止
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = this.fetchNewToken();
    try {
      const token = await this.refreshPromise;
      return token;
    } finally {
      this.refreshPromise = null;
    }
  }

  private async fetchNewToken(): Promise<string> {
    const response = await fetch(this.tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${Buffer.from(
          `${this.clientId}:${this.clientSecret}`,
        ).toString('base64')}`,
      },
      body: new URLSearchParams({
        grant_type: 'client_credentials',
        scope: this.scope,
      }),
    });

    if (!response.ok) {
      throw new Error(`Token request failed: ${response.status}`);
    }

    const data = await response.json();
    this.token = data.access_token;
    this.expiresAt = Date.now() + data.expires_in * 1000;

    return this.token;
  }
}

// マイクロサービス間通信での使用例
class OrderService {
  private tokenManager: ServiceTokenManager;

  constructor() {
    this.tokenManager = new ServiceTokenManager(
      process.env.ORDER_SERVICE_CLIENT_ID!,
      process.env.ORDER_SERVICE_CLIENT_SECRET!,
      'https://auth.example.com/token',
      'inventory:read inventory:reserve payments:create',
    );
  }

  async createOrder(orderData: OrderRequest): Promise<Order> {
    const token = await this.tokenManager.getToken();

    // 在庫サービスに問い合わせ
    const inventory = await fetch('https://inventory.internal/api/check', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ items: orderData.items }),
    });

    // 決済サービスに送信
    const payment = await fetch('https://payments.internal/api/charge', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        amount: orderData.total,
        currency: 'JPY',
      }),
    });

    // ...
  }
}
```

### 4.2 private_key_jwt 認証

```
Client Credentials の高度な認証方式:

  client_secret_basic（基本）:
    → Authorization: Basic base64(client_id:client_secret)
    → シンプルだが、secret の安全な管理が必要

  client_secret_post:
    → POST ボディに client_id と client_secret を含める
    → TLS 必須

  private_key_jwt（推奨・高セキュリティ）:
    → client_secret の代わりに RSA/EC 秘密鍵で署名した JWT を送信
    → 秘密鍵が外部に送信されない
    → 鍵のローテーションが容易

  private_key_jwt のフロー:
    ┌──────────────┐                    ┌──────────────┐
    │ クライアント   │                    │ 認可サーバー   │
    │              │                    │              │
    │ JWT 生成:    │                    │              │
    │ {            │                    │              │
    │  iss: client │                    │              │
    │  sub: client │                    │              │
    │  aud: auth   │                    │              │
    │  exp: +5min  │                    │              │
    │  jti: random │                    │              │
    │ }            │                    │              │
    │ → 秘密鍵で署名 │                   │              │
    │              │                    │              │
    │ POST /token  │                    │              │
    │ grant_type=  │                    │              │
    │ client_creds │                    │              │
    │ client_      │                    │              │
    │ assertion=JWT│                    │              │
    │─────────────────────────────────>│              │
    │              │                    │ 公開鍵で検証   │
    │              │                    │ (JWKS)       │
    │              │                    │              │
    │ access_token │                    │              │
    │<─────────────────────────────────│              │
    └──────────────┘                    └──────────────┘
```

```typescript
// private_key_jwt 認証の実装
import * as jose from 'jose';

async function getTokenWithPrivateKeyJWT(
  privateKey: jose.KeyLike,
  clientId: string,
  tokenUrl: string,
  scope: string,
): Promise<string> {
  // クライアントアサーション JWT の生成
  const assertion = await new jose.SignJWT({})
    .setProtectedHeader({ alg: 'RS256', typ: 'JWT' })
    .setIssuer(clientId)
    .setSubject(clientId)
    .setAudience(tokenUrl)
    .setIssuedAt()
    .setExpirationTime('5m')
    .setJti(crypto.randomUUID())
    .sign(privateKey);

  const response = await fetch(tokenUrl, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      scope,
      client_id: clientId,
      client_assertion_type: 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer',
      client_assertion: assertion,
    }),
  });

  const data = await response.json();
  return data.access_token;
}
```

---

## 5. Device Code フロー

```
Device Code フロー:

  用途: キーボード入力が困難なデバイス（テレビ、IoT、CLI）
  例: 「テレビで https://example.com/activate にアクセスして
       コード ABC-123 を入力してください」

  デバイス        認可サーバー      ユーザーのスマホ/PC
    │               │                 │
    │ device auth    │                 │
    │──────────────>│                 │
    │               │                 │
    │ device_code + │                 │
    │ user_code +   │                 │
    │ verification  │                 │
    │ _uri          │                 │
    │<──────────────│                 │
    │               │                 │
    │ 画面に表示:    │                 │
    │ "コード:       │                 │
    │  ABC-123"     │                 │
    │               │ ユーザーが        │
    │               │ verification_uri│
    │               │ にアクセス        │
    │               │<────────────────│
    │               │ コード入力+承認   │
    │               │<────────────────│
    │               │                 │
    │ ポーリング      │                 │
    │ (device_code)  │                 │
    │──────────────>│                 │
    │               │                 │
    │ access_token  │                 │
    │<──────────────│                 │
```

### 5.1 Device Code フローのセキュリティ

```
Device Code フローの攻撃と対策:

  攻撃1: リモートフィッシング
    → 攻撃者が自分のデバイスコードをフィッシングメールで送信
    → 「ログインして ABC-123 を入力してください」
    → ユーザーが攻撃者のデバイスにアクセスを許可してしまう
    対策:
      → ユーザーにデバイス情報を表示
      → 「このデバイスからのアクセスを許可しますか?」と明示
      → verification_uri_complete を使用しない（自動承認防止）

  攻撃2: ポーリング過多（DoS）
    → 悪意あるクライアントが高頻度でポーリング
    対策:
      → interval パラメータで最小間隔を指定
      → slow_down エラーで間隔延長を要求
      → レート制限

  攻撃3: デバイスコードのブルートフォース
    → user_code の総当たり試行
    対策:
      → user_code の試行回数制限
      → 十分なエントロピー（8文字英数字 = ~40ビット）
      → 短い有効期限（15分程度）
```

```typescript
// Device Code フロー（CLI ツール等）
async function deviceCodeFlow() {
  // デバイス認可リクエスト
  const deviceRes = await fetch('https://auth.example.com/device/code', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      client_id: 'my-cli-app',
      scope: 'openid profile',
    }),
  });

  const {
    device_code,
    user_code,
    verification_uri,
    verification_uri_complete,
    expires_in,
    interval,
  } = await deviceRes.json();

  // ユーザーに表示
  console.log(`ブラウザで ${verification_uri} にアクセスし、`);
  console.log(`コード ${user_code} を入力してください。`);

  // ポーリングでトークン取得を試行
  let pollInterval = interval;
  const deadline = Date.now() + expires_in * 1000;

  while (Date.now() < deadline) {
    await sleep(pollInterval * 1000);

    const tokenRes = await fetch('https://auth.example.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'urn:ietf:params:oauth:grant-type:device_code',
        device_code,
        client_id: 'my-cli-app',
      }),
    });

    const data = await tokenRes.json();

    if (data.error === 'authorization_pending') continue;
    if (data.error === 'slow_down') {
      pollInterval += 5; // ポーリング間隔を延長
      continue;
    }
    if (data.error) throw new Error(data.error_description);

    return data; // { access_token, refresh_token, ... }
  }

  throw new Error('Device code expired');
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
```

---

## 6. Implicit フロー（非推奨）

```
Implicit フロー（歴史的背景と非推奨の理由）:

  OAuth 2.0 初期の SPA 向けフロー:
    → ブラウザから直接 access_token を取得
    → トークン交換ステップが不要（シンプル）
    → CORS が普及する前の妥協策

  フロー:
    クライアント          認可サーバー
      │                   │
      │ response_type=    │
      │ token             │
      │──────────────────>│
      │                   │
      │ #access_token=... │  ← URL フラグメントに含まれる
      │<──────────────────│

  非推奨の理由:

    ① アクセストークンがフロントチャネルに露出:
       → URL フラグメントに含まれる
       → ブラウザ履歴に残る
       → Referer ヘッダーで漏洩する可能性
       → ログファイルに記録される可能性

    ② リフレッシュトークンが使えない:
       → トークン期限切れ = 再認可が必要
       → UX が悪い

    ③ トークン置換攻撃:
       → 攻撃者のトークンを注入可能
       → aud 検証が困難

    ④ PKCE が使えない:
       → トークン交換ステップがないため

  代替策:
    → Authorization Code + PKCE を使用
    → BFF（Backend for Frontend）パターン
    → OAuth 2.1 では Implicit は廃止予定

  移行ガイド:
    Before (Implicit):
      response_type=token
      → #access_token=xxx が直接返される

    After (Auth Code + PKCE):
      response_type=code
      code_challenge=xxx
      code_challenge_method=S256
      → ?code=yyy が返される
      → バックチャネルで code → token 交換
```

---

## 7. トークンのライフサイクル管理

### 7.1 リフレッシュトークンフロー

```
リフレッシュトークンの仕組み:

  access_token の有効期限が切れた場合:

  クライアント          認可サーバー
    │                   │
    │ API リクエスト      │
    │ + 期限切れ token   │
    │──────────────────>│
    │                   │
    │ 401 Unauthorized  │
    │<──────────────────│
    │                   │
    │ POST /token       │
    │ grant_type=       │
    │ refresh_token     │
    │ + refresh_token   │
    │──────────────────>│
    │                   │
    │ 新 access_token + │
    │ 新 refresh_token  │
    │ (ローテーション)    │
    │<──────────────────│
    │                   │
    │ API リクエスト      │
    │ + 新 token        │
    │──────────────────>│
    │                   │
    │ 200 OK + データ    │
    │<──────────────────│

  トークンの有効期限設計:

    access_token:
      → 短寿命: 5分〜1時間
      → 漏洩時の影響を最小化
      → 権限変更の反映を早める
      → 推奨: 15分（金融系: 5分）

    refresh_token:
      → 長寿命: 7日〜90日
      → ユーザー体験（頻繁な再ログイン防止）
      → 使用時にローテーション（新しいものに交換）
      → 推奨: 30日（activity に応じて延長も可）

    id_token（OIDC）:
      → 認証時刻の証明のみ
      → リフレッシュ不可
      → 推奨: 1時間
```

```typescript
// リフレッシュトークン ローテーション実装
class TokenRefreshManager {
  private refreshing: Promise<TokenPair> | null = null;

  constructor(
    private tokenUrl: string,
    private clientId: string,
    private onTokenRefresh: (tokens: TokenPair) => void,
  ) {}

  // トークンリフレッシュ（重複防止付き）
  async refreshTokens(currentRefreshToken: string): Promise<TokenPair> {
    // 既にリフレッシュ中なら同じ Promise を返す
    if (this.refreshing) {
      return this.refreshing;
    }

    this.refreshing = this.doRefresh(currentRefreshToken);

    try {
      const tokens = await this.refreshing;
      this.onTokenRefresh(tokens);
      return tokens;
    } finally {
      this.refreshing = null;
    }
  }

  private async doRefresh(refreshToken: string): Promise<TokenPair> {
    const response = await fetch(this.tokenUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'refresh_token',
        refresh_token: refreshToken,
        client_id: this.clientId,
      }),
    });

    if (!response.ok) {
      const error = await response.json();

      if (error.error === 'invalid_grant') {
        // リフレッシュトークンが無効 → 再ログインが必要
        throw new TokenRefreshError('refresh_token_invalid', 'Re-authentication required');
      }

      throw new TokenRefreshError(error.error, error.error_description);
    }

    const data = await response.json();
    return {
      accessToken: data.access_token,
      refreshToken: data.refresh_token,  // ローテーション: 新しい refresh_token
      expiresIn: data.expires_in,
    };
  }
}

// HTTP クライアントとの統合（自動リフレッシュ）
class AuthenticatedHttpClient {
  private accessToken: string;
  private refreshToken: string;
  private expiresAt: number;
  private refreshManager: TokenRefreshManager;

  constructor(
    initialTokens: TokenPair,
    refreshManager: TokenRefreshManager,
  ) {
    this.accessToken = initialTokens.accessToken;
    this.refreshToken = initialTokens.refreshToken;
    this.expiresAt = Date.now() + initialTokens.expiresIn * 1000;
    this.refreshManager = refreshManager;
  }

  async fetch(url: string, options: RequestInit = {}): Promise<Response> {
    // アクセストークンの期限チェック
    await this.ensureValidToken();

    const response = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        'Authorization': `Bearer ${this.accessToken}`,
      },
    });

    // 401 が返ってきた場合はリフレッシュを試行
    if (response.status === 401) {
      try {
        await this.doTokenRefresh();

        // リトライ
        return fetch(url, {
          ...options,
          headers: {
            ...options.headers,
            'Authorization': `Bearer ${this.accessToken}`,
          },
        });
      } catch (error) {
        if (error instanceof TokenRefreshError) {
          // リフレッシュ失敗 → 再ログインが必要
          this.redirectToLogin();
        }
        throw error;
      }
    }

    return response;
  }

  private async ensureValidToken(): Promise<void> {
    // 有効期限の30秒前にプロアクティブにリフレッシュ
    if (Date.now() > this.expiresAt - 30 * 1000) {
      await this.doTokenRefresh();
    }
  }

  private async doTokenRefresh(): Promise<void> {
    const tokens = await this.refreshManager.refreshTokens(this.refreshToken);
    this.accessToken = tokens.accessToken;
    this.refreshToken = tokens.refreshToken;
    this.expiresAt = Date.now() + tokens.expiresIn * 1000;
  }

  private redirectToLogin(): void {
    window.location.href = '/login?reason=session_expired';
  }
}
```

### 7.2 リフレッシュトークンのセキュリティ

```
リフレッシュトークン ローテーションの重要性:

  ローテーションなし（危険）:
    → リフレッシュトークンが漏洩 → 無期限にアクセス可能
    → 攻撃者と正規ユーザーが同じトークンを使用
    → 検知が困難

  ローテーションあり（推奨）:
    → リフレッシュするたびに新しいトークンを発行
    → 古いトークンは無効化

  リプレイ検知:

    正常フロー:
      RT1 → 新 AT + RT2 → 新 AT + RT3 → ...

    攻撃フロー:
      攻撃者が RT1 を窃取
      正規ユーザー: RT1 → RT2（正常）
      攻撃者:       RT1 → ✗ 使用済み!
        → 全リフレッシュトークンを無効化
        → ユーザーに再ログインを要求

  リプレイ検知の仕組み:
  ┌──────────────────────────────────────────┐
  │ Token Family                              │
  │                                           │
  │ RT1 → RT2 → RT3 → RT4（現在有効）          │
  │                                           │
  │ RT1 が再使用された場合:                      │
  │ → RT1, RT2, RT3, RT4 すべて無効化          │
  │ → 「トークンファミリー」全体を無効化          │
  │ → ユーザーに再認証を要求                     │
  └──────────────────────────────────────────┘
```

```typescript
// リフレッシュトークン ローテーション + リプレイ検知
class RefreshTokenStore {
  private redis: Redis;

  async storeToken(
    tokenFamily: string,
    refreshToken: string,
    userId: string,
    clientId: string,
    expiresIn: number,
  ): Promise<void> {
    const data = JSON.stringify({
      userId,
      clientId,
      tokenFamily,
      createdAt: Date.now(),
    });

    // リフレッシュトークン → データのマッピング
    await this.redis.setex(`rt:${refreshToken}`, expiresIn, data);

    // トークンファミリーに追加（リプレイ検知用）
    await this.redis.sadd(`tf:${tokenFamily}`, refreshToken);
    await this.redis.expire(`tf:${tokenFamily}`, expiresIn);
  }

  async useToken(refreshToken: string): Promise<{
    userId: string;
    clientId: string;
    tokenFamily: string;
  } | null> {
    const raw = await this.redis.get(`rt:${refreshToken}`);

    if (!raw) {
      // トークンが存在しない
      // → 有効期限切れ or 既に使用済み（リプレイ攻撃の可能性）
      return null;
    }

    const data = JSON.parse(raw);

    // トークンを使用済みにする（一度きりの使用）
    await this.redis.del(`rt:${refreshToken}`);

    return data;
  }

  // トークンファミリー全体を無効化（リプレイ検知時）
  async revokeFamily(tokenFamily: string): Promise<void> {
    const tokens = await this.redis.smembers(`tf:${tokenFamily}`);

    if (tokens.length > 0) {
      const keys = tokens.map(t => `rt:${t}`);
      await this.redis.del(...keys);
    }

    await this.redis.del(`tf:${tokenFamily}`);
  }

  // 特定ユーザーの全トークンを無効化
  async revokeAllForUser(userId: string): Promise<void> {
    // ユーザーのトークンファミリー一覧を取得して全無効化
    const families = await this.redis.smembers(`user_families:${userId}`);
    for (const family of families) {
      await this.revokeFamily(family);
    }
    await this.redis.del(`user_families:${userId}`);
  }
}
```

---

## 8. スコープ設計

```
スコープの設計パターン:

  resource:action 形式:
    → read:user, write:user, delete:user
    → read:repo, write:repo, admin:repo

  GitHub のスコープ例:
    → repo: リポジトリ全般
    → read:user: ユーザープロフィール読取
    → user:email: メールアドレス読取
    → admin:org: 組織管理

  設計原則:
    → 最小権限: 必要最小限のスコープのみ要求
    → 粒度: 細かすぎず粗すぎず
    → 命名: resource:action の一貫したパターン
    → ドキュメント: 各スコープの意味を明確に
```

### 8.1 スコープ設計の詳細パターン

```
スコープの階層設計:

  粗粒度（GitHub 方式）:
    repo                → リポジトリ全般（読取・書込・削除）
    repo:status         → コミットステータスのみ
    read:user           → ユーザー情報読取
    user:email          → メールアドレスのみ

    利点: シンプルでユーザーが理解しやすい
    欠点: 細かい制御が困難

  中粒度（Google 方式）:
    https://www.googleapis.com/auth/calendar
    https://www.googleapis.com/auth/calendar.readonly
    https://www.googleapis.com/auth/gmail.send
    https://www.googleapis.com/auth/gmail.readonly

    利点: URI で名前空間を明確に
    欠点: URL が長い

  細粒度（resource:action 方式）:
    users:read           → ユーザー一覧の閲覧
    users:write          → ユーザー情報の更新
    users:delete         → ユーザーの削除
    posts:read           → 記事の閲覧
    posts:write          → 記事の作成・編集
    posts:publish        → 記事の公開
    admin:settings:read  → 管理設定の閲覧
    admin:settings:write → 管理設定の変更

    利点: 最小権限の原則を徹底
    欠点: スコープ数が増大

  同意画面での表示:
  ┌─────────────────────────────────────────────┐
  │  MyApp がアクセスを要求しています              │
  │                                              │
  │  □ プロフィール情報の閲覧（users:read）        │
  │  □ メールアドレスの読取（users:email）          │
  │  □ リポジトリの読取（repos:read）              │
  │                                              │
  │  [許可する]  [拒否する]                         │
  └─────────────────────────────────────────────┘
```

```typescript
// スコープベースの認可チェック実装
class ScopeValidator {
  // スコープ文字列を配列に変換
  static parse(scopeString: string): string[] {
    return scopeString.split(' ').filter(Boolean);
  }

  // 要求されたスコープが付与されたスコープに含まれるか確認
  static hasScope(grantedScopes: string[], requiredScope: string): boolean {
    // 完全一致チェック
    if (grantedScopes.includes(requiredScope)) return true;

    // 階層チェック（"repo" は "repo:status" を包含）
    for (const granted of grantedScopes) {
      if (requiredScope.startsWith(granted + ':')) return true;
    }

    // ワイルドカードチェック
    for (const granted of grantedScopes) {
      if (granted.endsWith(':*')) {
        const prefix = granted.slice(0, -1);
        if (requiredScope.startsWith(prefix)) return true;
      }
    }

    return false;
  }

  // 複数スコープの確認（全て必要）
  static hasAllScopes(
    grantedScopes: string[],
    requiredScopes: string[],
  ): boolean {
    return requiredScopes.every(scope => this.hasScope(grantedScopes, scope));
  }

  // 複数スコープの確認（いずれか1つ）
  static hasAnyScope(
    grantedScopes: string[],
    requiredScopes: string[],
  ): boolean {
    return requiredScopes.some(scope => this.hasScope(grantedScopes, scope));
  }
}

// Express ミドルウェアとして使用
function requireScope(...scopes: string[]) {
  return (req: Request, res: Response, next: NextFunction) => {
    const tokenScopes = req.tokenPayload?.scope?.split(' ') || [];

    if (!ScopeValidator.hasAllScopes(tokenScopes, scopes)) {
      return res.status(403).json({
        error: 'insufficient_scope',
        required_scope: scopes.join(' '),
        granted_scope: tokenScopes.join(' '),
      });
    }

    next();
  };
}

// 使用例
app.get('/api/repos',
  authenticateBearer,
  requireScope('repos:read'),
  listReposHandler,
);

app.post('/api/repos/:id/deploy',
  authenticateBearer,
  requireScope('repos:write', 'deploy:execute'),
  deployHandler,
);
```

---

## 9. トークン失効（Revocation）

```
トークン失効の仕組み（RFC 7009）:

  POST /revoke
  Content-Type: application/x-www-form-urlencoded
  Authorization: Basic <client_credentials>

  token=<access_token or refresh_token>
  &token_type_hint=refresh_token

  失効が必要なケース:
    → ユーザーがログアウト
    → ユーザーがアプリ連携を解除
    → 管理者がユーザーのアクセスを取り消し
    → パスワード変更時に全トークンを無効化
    → セキュリティインシデント発生時

  失効の範囲:
    → refresh_token を失効 → 関連する access_token も失効（推奨）
    → access_token のみ失効 → refresh_token で再取得可能（不十分）
```

```typescript
// トークン失効エンドポイントの実装
async function handleRevocation(req: express.Request, res: express.Response) {
  const { token, token_type_hint } = req.body;
  const { clientId } = extractClientCredentials(req);

  // RFC 7009: 成功時は常に 200 OK を返す
  // （トークンが存在しない場合も 200 を返す）
  res.status(200);

  try {
    if (token_type_hint === 'refresh_token' || !token_type_hint) {
      // リフレッシュトークンの失効を試行
      const revoked = await tokenStore.revokeRefreshToken(token, clientId);
      if (revoked) {
        // 関連するアクセストークンも失効
        await tokenStore.revokeAccessTokensByRefreshToken(token);
        return res.json({ revoked: true });
      }
    }

    if (token_type_hint === 'access_token' || !token_type_hint) {
      // アクセストークンの失効を試行
      await tokenStore.revokeAccessToken(token, clientId);
      return res.json({ revoked: true });
    }

    res.json({ revoked: false });
  } catch (error) {
    // エラーでも 200 OK を返す（RFC 7009 準拠）
    res.json({ revoked: false });
  }
}

// トークンイントロスペクション（RFC 7662）
async function handleIntrospection(req: express.Request, res: express.Response) {
  const { token } = req.body;

  const tokenData = await tokenStore.getToken(token);

  if (!tokenData || tokenData.revoked || tokenData.expiresAt < new Date()) {
    // 無効なトークン
    return res.json({ active: false });
  }

  // 有効なトークン
  res.json({
    active: true,
    scope: tokenData.scope,
    client_id: tokenData.clientId,
    username: tokenData.username,
    token_type: 'Bearer',
    exp: Math.floor(tokenData.expiresAt.getTime() / 1000),
    iat: Math.floor(tokenData.createdAt.getTime() / 1000),
    sub: tokenData.userId,
    aud: tokenData.audience,
    iss: 'https://auth.example.com',
  });
}
```

---

## 10. セキュリティ脅威と対策

```
OAuth 2.0 に対する主要な攻撃:

  ┌─────────────────────────┬──────────────────────────────┐
  │ 攻撃                     │ 対策                          │
  ├─────────────────────────┼──────────────────────────────┤
  │ CSRF（認可リクエスト偽造） │ state パラメータ               │
  │ 認可コード横取り          │ PKCE                          │
  │ オープンリダイレクタ      │ redirect_uri の完全一致        │
  │ トークン漏洩             │ 短寿命 + TLS + HttpOnly       │
  │ コード再利用攻撃          │ 一度きりの使用 + ファミリー無効化│
  │ クリックジャッキング      │ X-Frame-Options: DENY         │
  │ 混同攻撃（Mix-Up）       │ iss パラメータ検証             │
  │ トークン置換             │ aud / azp クレーム検証          │
  └─────────────────────────┴──────────────────────────────┘
```

### 10.1 オープンリダイレクタ攻撃

```
オープンリダイレクタ攻撃の仕組み:

  ① 攻撃者が redirect_uri を操作:
     /authorize?
       client_id=legit-app
       &redirect_uri=https://evil.com/steal  ← 不正な URI

  ② 認可サーバーが redirect_uri を検証しない場合:
     → 認可コードが evil.com に送信される
     → 攻撃者がコードを取得

  対策:
    → redirect_uri の完全一致検証（部分一致禁止）
    → ワイルドカード禁止
    → 事前登録された URI のみ許可
    → パスの追加やクエリパラメータの変更も拒否

  安全な実装:
    ✗ redirect_uri.startsWith('https://myapp.com')
      → https://myapp.com.evil.com にマッチしてしまう

    ✗ redirect_uri が登録済みのいずれかで「始まる」
      → https://myapp.com/callback/../../../evil にマッチしうる

    ✓ redirect_uri === 登録済み URI（完全一致のみ）
```

### 10.2 CSRF 攻撃と state パラメータ

```
OAuth 2.0 における CSRF 攻撃:

  攻撃シナリオ（state パラメータなし）:

    攻撃者                 被害者              認可サーバー
      │                    │                   │
      │ ① 攻撃者のアカウント │                   │
      │    で認可コード取得  │                   │
      │───────────────────────────────────────>│
      │                    │                   │
      │ code=ATTACKER_CODE │                   │
      │<───────────────────────────────────────│
      │                    │                   │
      │ ② 被害者に偽の       │                   │
      │    callback URL を   │                   │
      │    踏ませる         │                   │
      │ myapp.com/callback  │                   │
      │ ?code=ATTACKER_CODE │                   │
      │───────────────────>│                   │
      │                    │                   │
      │                    │ ③ ATTACKER_CODE    │
      │                    │   で token 取得    │
      │                    │──────────────────>│
      │                    │                   │
      │                    │ ④ 攻撃者のアカウントの│
      │                    │   トークンを使用!    │
      │                    │<──────────────────│

    結果:
      → 被害者のアプリが攻撃者のアカウントに紐づく
      → 被害者が攻撃者のアカウントにデータをアップロード
      → 攻撃者がそのデータを取得

  対策: state パラメータ
    → 暗号的にランダムな値を生成
    → セッションに紐づけて保存
    → コールバック時に一致確認
    → 一致しなければリクエストを拒否
```

---

## 11. BFF（Backend for Frontend）パターン

```
BFF パターンの概要:

  SPA の OAuth セキュリティ課題:
    → access_token をブラウザに保存 → XSS リスク
    → refresh_token をブラウザに保存 → 漏洩リスク
    → PKCE でも、トークン自体の保護は不十分

  BFF パターンの解決策:
    → フロントエンドとバックエンドの間に薄い BFF サーバーを配置
    → BFF が OAuth フローを代行
    → トークンはサーバー側（BFF）のみに保存
    → ブラウザとの通信は HttpOnly Cookie で行う

  アーキテクチャ:
  ┌─────────┐  Cookie    ┌──────┐  Bearer   ┌──────────┐
  │ SPA     │──────────>│ BFF  │─────────>│ API      │
  │ (React) │<──────────│      │<─────────│ Server   │
  └─────────┘           └──┬───┘          └──────────┘
                           │
                           │ OAuth 2.0
                           │
                        ┌──┴───┐
                        │ Auth │
                        │Server│
                        └──────┘

  BFF の役割:
    → /bff/login → 認可リクエスト開始
    → /bff/callback → コールバック処理 + セッション作成
    → /bff/api/* → API プロキシ（Bearer トークン付与）
    → /bff/logout → セッション破棄 + トークン失効
```

```typescript
// BFF パターンの実装
import express from 'express';
import session from 'express-session';

const bff = express();

bff.use(session({
  secret: process.env.SESSION_SECRET!,
  resave: false,
  saveUninitialized: false,
  cookie: {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 24 * 60 * 60 * 1000,
  },
}));

// ログイン開始
bff.get('/bff/login', async (req, res) => {
  const { verifier, challenge } = await generatePKCE();
  const state = crypto.randomUUID();

  // セッションに保存
  req.session.pkceVerifier = verifier;
  req.session.oauthState = state;

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: process.env.OAUTH_CLIENT_ID!,
    redirect_uri: `${process.env.BFF_URL}/bff/callback`,
    scope: 'openid profile email',
    state,
    code_challenge: challenge,
    code_challenge_method: 'S256',
  });

  res.redirect(`${process.env.AUTH_URL}/authorize?${params}`);
});

// コールバック
bff.get('/bff/callback', async (req, res) => {
  const { code, state } = req.query as Record<string, string>;

  // state 検証
  if (state !== req.session.oauthState) {
    return res.status(403).json({ error: 'Invalid state' });
  }

  // トークン交換
  const tokenResponse = await fetch(`${process.env.AUTH_URL}/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: `${process.env.BFF_URL}/bff/callback`,
      client_id: process.env.OAUTH_CLIENT_ID!,
      client_secret: process.env.OAUTH_CLIENT_SECRET!, // BFF は Confidential Client
      code_verifier: req.session.pkceVerifier,
    }),
  });

  const tokens = await tokenResponse.json();

  // トークンをセッションに保存（ブラウザには渡さない!）
  req.session.accessToken = tokens.access_token;
  req.session.refreshToken = tokens.refresh_token;
  req.session.tokenExpiresAt = Date.now() + tokens.expires_in * 1000;

  // クリーンアップ
  delete req.session.pkceVerifier;
  delete req.session.oauthState;

  // SPA にリダイレクト
  res.redirect(process.env.SPA_URL!);
});

// API プロキシ
bff.all('/bff/api/*', async (req, res) => {
  if (!req.session.accessToken) {
    return res.status(401).json({ error: 'Not authenticated' });
  }

  // トークンの期限チェック + リフレッシュ
  if (Date.now() > req.session.tokenExpiresAt - 30000) {
    try {
      const refreshed = await refreshTokens(req.session.refreshToken);
      req.session.accessToken = refreshed.access_token;
      req.session.refreshToken = refreshed.refresh_token;
      req.session.tokenExpiresAt = Date.now() + refreshed.expires_in * 1000;
    } catch {
      return res.status(401).json({ error: 'Session expired' });
    }
  }

  // API にプロキシ
  const apiPath = req.path.replace('/bff/api', '');
  const apiResponse = await fetch(`${process.env.API_URL}${apiPath}`, {
    method: req.method,
    headers: {
      'Authorization': `Bearer ${req.session.accessToken}`,
      'Content-Type': req.headers['content-type'] || 'application/json',
    },
    body: ['POST', 'PUT', 'PATCH'].includes(req.method)
      ? JSON.stringify(req.body)
      : undefined,
  });

  const data = await apiResponse.json();
  res.status(apiResponse.status).json(data);
});

// ログアウト
bff.post('/bff/logout', async (req, res) => {
  // トークン失効
  if (req.session.refreshToken) {
    await fetch(`${process.env.AUTH_URL}/revoke`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Authorization': `Basic ${Buffer.from(
          `${process.env.OAUTH_CLIENT_ID}:${process.env.OAUTH_CLIENT_SECRET}`,
        ).toString('base64')}`,
      },
      body: new URLSearchParams({
        token: req.session.refreshToken,
        token_type_hint: 'refresh_token',
      }),
    });
  }

  // セッション破棄
  req.session.destroy(() => {
    res.clearCookie('connect.sid');
    res.json({ success: true });
  });
});
```

---

## 12. アンチパターン

```
OAuth 2.0 のアンチパターン:

  ① フロントエンドに client_secret を含める:
     ✗ SPA の JavaScript に client_secret をハードコード
     → ブラウザの開発者ツールで閲覧可能
     → ソースマップから発見可能
     対策: Public Client + PKCE を使用

  ② redirect_uri の部分一致検証:
     ✗ startsWith や contains で検証
     → オープンリダイレクタ攻撃が可能
     対策: 完全一致のみ許可

  ③ state パラメータの省略:
     ✗ CSRF 防御なしで OAuth フロー実行
     → 攻撃者のアカウントに紐づけ攻撃が可能
     対策: 暗号的にランダムな state を必ず使用

  ④ アクセストークンの長寿命化:
     ✗ access_token の有効期限を24時間以上に設定
     → 漏洩時の影響が長期化
     対策: 短寿命（15分）+ リフレッシュトークン

  ⑤ リフレッシュトークンのローテーションなし:
     ✗ 同じリフレッシュトークンを永続的に使用
     → 漏洩時に永続的なアクセスを許す
     対策: ローテーション + リプレイ検知

  ⑥ Implicit フローの使用:
     ✗ SPA で response_type=token を使用
     → トークンが URL フラグメントに露出
     対策: Authorization Code + PKCE に移行

  ⑦ スコープの過剰要求:
     ✗ 必要以上のスコープを要求
     → ユーザーの信頼を損なう
     → 漏洩時の影響範囲が拡大
     対策: 最小権限の原則（必要なスコープのみ要求）

  ⑧ トークンの不適切な保存:
     ✗ localStorage にアクセストークンを保存
     → XSS で窃取可能
     対策: BFF パターン or メモリ内保存 + HttpOnly Cookie
```

---

## 13. エッジケース

```
OAuth 2.0 のエッジケース:

  ① スコープのダウングレード:
     → ユーザーが要求された一部のスコープのみ承認
     → クライアントは実際に付与されたスコープを確認すべき
     → レスポンスの scope パラメータで確認

  ② 認可サーバーのダウンタイム:
     → トークンリフレッシュが失敗
     → 対策: 短期的にキャッシュされたトークンで動作継続
     → access_token の有効期限まではリソースアクセス可能

  ③ クロックスキュー:
     → JWT の exp/iat 検証時にサーバー間の時刻のずれ
     → 対策: ±30秒の許容範囲（clock skew tolerance）

  ④ 複数の redirect_uri:
     → 開発環境と本番環境で異なるコールバック URL
     → 認可リクエストの redirect_uri は登録済みのいずれかと完全一致
     → 環境ごとにクライアント ID を分離する方が安全

  ⑤ トークンの最大サイズ:
     → JWT をアクセストークンとして使用する場合
     → HTTP ヘッダーの制限（通常 8KB）
     → スコープやクレームが多いと超過する可能性
     → 対策: トークンイントロスペクション、参照トークン
```

---

## 14. 演習問題

```
演習1（基礎）: Authorization Code + PKCE フローの実装

  以下の要件で OAuth 2.0 クライアントを実装せよ。

  要件:
  1. PKCE 対応の認可リクエスト生成
  2. state パラメータによる CSRF 防御
  3. コールバック処理（検証 + トークン交換）
  4. トークンの安全な保存

  テストケース:
  → code_verifier の長さが 43〜128 文字であること
  → code_challenge が SHA-256 で正しく計算されること
  → state の不一致で例外が発生すること
  → トークン交換後に verifier がクリーンアップされること

演習2（応用）: リフレッシュトークン ローテーション

  以下の機能を持つリフレッシュトークン管理システムを実装せよ。

  要件:
  1. リフレッシュトークンのローテーション（毎回新しいトークン）
  2. リプレイ検知（使用済みトークンの再利用を検知）
  3. トークンファミリーの無効化
  4. ユーザー単位の全トークン無効化

  テストケース:
  → 正常なリフレッシュフロー
  → 使用済みトークンの再利用 → ファミリー全体が無効化
  → パスワード変更 → 全トークン無効化

演習3（発展）: 認可サーバーの構築

  OAuth 2.0 認可サーバーを実装せよ。

  要件:
  1. Authorization Code + PKCE フロー対応
  2. Client Credentials フロー対応
  3. リフレッシュトークンフロー
  4. トークン失効エンドポイント（RFC 7009）
  5. トークンイントロスペクション（RFC 7662）
  6. クライアント登録管理

  セキュリティ要件:
  → 認可コードの一度切り使用
  → redirect_uri の完全一致検証
  → PKCE の S256 検証
  → リフレッシュトークンのローテーション
```

---

## 15. FAQ・トラブルシューティング

```
Q1: OAuth 2.0 と OpenID Connect の違いは何ですか?
A1: OAuth 2.0 は「認可」、OIDC は「認証」です:
    → OAuth 2.0: 「MyApp に GitHub のリポジトリへのアクセスを許可」
    → OIDC: 「このユーザーは alice@example.com である」
    → OIDC は OAuth 2.0 の上に構築された認証レイヤー
    → OIDC 追加要素: id_token、userinfo エンドポイント、標準クレーム

Q2: PKCE は Confidential Client にも必要ですか?
A2: OAuth 2.1 では全クライアントに PKCE が必須になります:
    → Confidential Client でも追加のセキュリティ層として有効
    → 認可コードの漏洩リスクをさらに低減
    → 実装コストが低い（数十行のコード）ため、常に使用を推奨

Q3: アクセストークンの適切な有効期限は?
A3: ユースケースによります:
    → 一般的な Web アプリ: 15〜60分
    → 金融系: 5〜15分
    → マイクロサービス間: 30〜60分
    → 原則: 短いほど安全、長いほど UX が良い
    → リフレッシュトークンと組み合わせて調整

Q4: SPA でトークンをどこに保存すべき?
A4: 推奨される方法（安全な順）:
    ① BFF パターン（サーバー側セッション）← 最も安全
    ② メモリ（JavaScript 変数）+ HttpOnly Cookie の refresh
    ③ sessionStorage（タブ単位で隔離）
    ✗ localStorage は XSS リスクが高く非推奨

Q5: state パラメータと PKCE の違いは何ですか?
A5: 防御する攻撃が異なります:
    → state: CSRF 攻撃の防御（攻撃者のコードを注入）
    → PKCE: 認可コード横取り攻撃の防御（正規のコードを窃取）
    → 両方を使うことが推奨（防御する脅威が異なるため）

Q6: Device Code フローで user_code の形式に制約はありますか?
A6: RFC 8628 の推奨:
    → 8文字以上の英数字（大文字推奨）
    → ハイフン区切り推奨（例: ABCD-EFGH）
    → 混同しやすい文字を除外（0/O, 1/I/l）
    → 十分なエントロピー（ブルートフォース防止）

Q7: リフレッシュトークンのローテーションで古いトークンが使われたら?
A7: リプレイ攻撃の可能性が高いです:
    → そのトークンファミリー全体を無効化
    → ユーザーに再認証を要求
    → セキュリティアラートを発行
    → 監査ログに記録
```

---

## まとめ

| フロー | 用途 | セキュリティ |
|--------|------|------------|
| Authorization Code | Web アプリ（サーバーあり） | 最高 |
| Auth Code + PKCE | SPA、モバイル | 高い |
| Client Credentials | サーバー間 | 高い |
| Device Code | IoT、CLI | 中程度 |
| Implicit | 非推奨 | 低い |

| トピック | ポイント |
|---------|---------|
| PKCE | SHA-256 による認可コード横取り防止。OAuth 2.1 で必須 |
| state | CSRF 防御。暗号的ランダム値をセッションに紐づけ |
| リフレッシュトークン | ローテーション + リプレイ検知が推奨 |
| スコープ | 最小権限の原則。resource:action 形式 |
| BFF パターン | SPA の最もセキュアなトークン管理方法 |
| トークン失効 | RFC 7009。ログアウト時に必ず実行 |

---

## 次に読むべきガイド
--> [[02-openid-connect.md]] -- OpenID Connect

---

## 参考文献
1. RFC 6749. "The OAuth 2.0 Authorization Framework." IETF, 2012.
2. RFC 7636. "Proof Key for Code Exchange by OAuth Public Clients." IETF, 2015.
3. RFC 8628. "OAuth 2.0 Device Authorization Grant." IETF, 2019.
4. RFC 7009. "OAuth 2.0 Token Revocation." IETF, 2013.
5. RFC 7662. "OAuth 2.0 Token Introspection." IETF, 2015.
6. OAuth 2.0 Security Best Current Practice. IETF draft, 2024.
7. OAuth 2.1 Authorization Framework. IETF draft, 2024.
8. OAuth.net. "OAuth 2.0." oauth.net, 2024.
