# OpenID Connect

> OpenID Connect（OIDC）は OAuth 2.0 の上に構築された認証レイヤー。OAuth 2.0 が「認可」のプロトコルであるのに対し、OIDC は「認証」を標準化する。ID Token、UserInfo エンドポイント、Discovery、ソーシャルログインの基盤を解説する。

## この章で学ぶこと

- [ ] OAuth 2.0 と OIDC の関係と違いを理解する
- [ ] ID Token の構造と検証フローを把握する
- [ ] OIDC Discovery とプロバイダー連携を実装できるようになる

---

## 1. OIDC と OAuth 2.0 の関係

```
OAuth 2.0 vs OpenID Connect:

  OAuth 2.0:
  → 目的: 認可（Authorization）
  → 質問: 「このアプリに何を許可しますか？」
  → 結果: アクセストークン（リソースアクセス用）
  → ユーザーが誰かは保証しない

  OpenID Connect:
  → 目的: 認証（Authentication）
  → 質問: 「あなたは誰ですか？」
  → 結果: ID トークン（ユーザー情報）+ アクセストークン
  → ユーザーの身元を保証する

  関係:
  ┌──────────────────────────────────┐
  │         OpenID Connect           │
  │  ┌──────────────────────────┐    │
  │  │       OAuth 2.0          │    │
  │  │  （認可フレームワーク）     │    │
  │  └──────────────────────────┘    │
  │  + ID Token（認証）              │
  │  + UserInfo エンドポイント        │
  │  + Discovery                    │
  │  + Dynamic Registration         │
  └──────────────────────────────────┘

  OIDC = OAuth 2.0 + 認証の標準化
```

---

## 2. ID Token

```
ID Token の構造（JWT）:

  {
    // 標準クレーム
    "iss": "https://accounts.google.com",  // 発行者
    "sub": "110169484474386276334",         // ユーザー識別子（一意）
    "aud": "my-client-id",                 // クライアント ID
    "exp": 1700000000,                     // 有効期限
    "iat": 1699999100,                     // 発行時刻
    "auth_time": 1699999000,               // 認証時刻
    "nonce": "random-nonce-value",         // リプレイ攻撃防止

    // ユーザー情報クレーム
    "email": "alice@example.com",
    "email_verified": true,
    "name": "Alice Example",
    "picture": "https://example.com/alice.jpg",
    "locale": "ja"
  }

  Access Token vs ID Token:
    Access Token: API アクセスに使用（リソースサーバーに送信）
    ID Token:    ユーザー情報の確認に使用（クライアント内で消費）

    ✗ ID Token を API アクセスに使用してはいけない
    ✗ Access Token からユーザー情報を取得してはいけない
```

```typescript
// ID Token の検証
import { jwtVerify, createRemoteJWKSet } from 'jose';

// Google の JWKS URL
const JWKS = createRemoteJWKSet(
  new URL('https://www.googleapis.com/oauth2/v3/certs')
);

async function verifyIdToken(idToken: string, nonce: string) {
  const { payload } = await jwtVerify(idToken, JWKS, {
    issuer: 'https://accounts.google.com',
    audience: process.env.GOOGLE_CLIENT_ID,
    algorithms: ['RS256'],
  });

  // nonce の検証（リプレイ攻撃防止）
  if (payload.nonce !== nonce) {
    throw new Error('Invalid nonce');
  }

  // auth_time の検証（必要に応じて）
  const maxAge = 3600; // 1時間以内の認証を要求
  if (payload.auth_time && Date.now() / 1000 - (payload.auth_time as number) > maxAge) {
    throw new Error('Authentication too old');
  }

  return {
    sub: payload.sub,        // ユーザー固有 ID
    email: payload.email as string,
    name: payload.name as string,
    picture: payload.picture as string,
    emailVerified: payload.email_verified as boolean,
  };
}
```

---

## 3. OIDC Discovery

```
OpenID Connect Discovery:

  プロバイダーの設定を自動取得:
  GET https://accounts.google.com/.well-known/openid-configuration

  レスポンス:
  {
    "issuer": "https://accounts.google.com",
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
    "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    "scopes_supported": ["openid", "email", "profile"],
    "response_types_supported": ["code", "token", "id_token"],
    "subject_types_supported": ["public"],
    "id_token_signing_alg_values_supported": ["RS256"],
    "claims_supported": ["sub", "email", "name", "picture", ...]
  }

  利点:
  → エンドポイント URL をハードコードしない
  → プロバイダーの変更に自動追従
  → 複数プロバイダーの一元管理
```

```typescript
// OIDC Discovery を活用した汎用プロバイダー
interface OIDCConfig {
  issuer: string;
  authorization_endpoint: string;
  token_endpoint: string;
  userinfo_endpoint: string;
  jwks_uri: string;
  scopes_supported: string[];
}

class OIDCProvider {
  private config: OIDCConfig | null = null;

  constructor(private issuerUrl: string, private clientId: string, private clientSecret: string) {}

  // Discovery で設定を取得
  async discover(): Promise<OIDCConfig> {
    if (this.config) return this.config;

    const res = await fetch(`${this.issuerUrl}/.well-known/openid-configuration`);
    this.config = await res.json();
    return this.config!;
  }

  // 認可 URL を生成
  async getAuthorizationUrl(redirectUri: string, state: string, nonce: string): Promise<string> {
    const config = await this.discover();
    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.clientId,
      redirect_uri: redirectUri,
      scope: 'openid email profile',
      state,
      nonce,
    });
    return `${config.authorization_endpoint}?${params}`;
  }

  // トークン交換
  async exchangeCode(code: string, redirectUri: string): Promise<{
    accessToken: string;
    idToken: string;
    refreshToken?: string;
  }> {
    const config = await this.discover();
    const res = await fetch(config.token_endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code,
        redirect_uri: redirectUri,
        client_id: this.clientId,
        client_secret: this.clientSecret,
      }),
    });

    const data = await res.json();
    return {
      accessToken: data.access_token,
      idToken: data.id_token,
      refreshToken: data.refresh_token,
    };
  }

  // UserInfo エンドポイント
  async getUserInfo(accessToken: string): Promise<Record<string, any>> {
    const config = await this.discover();
    const res = await fetch(config.userinfo_endpoint, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    return res.json();
  }
}

// 使用例
const google = new OIDCProvider(
  'https://accounts.google.com',
  process.env.GOOGLE_CLIENT_ID!,
  process.env.GOOGLE_CLIENT_SECRET!
);

const github = new OIDCProvider(
  'https://token.actions.githubusercontent.com', // GitHub OIDC
  process.env.GITHUB_CLIENT_ID!,
  process.env.GITHUB_CLIENT_SECRET!
);
```

---

## 4. OIDC スコープとクレーム

```
標準スコープ:

  スコープ   │ 返されるクレーム
  ─────────┼──────────────────────────────
  openid   │ sub（必須スコープ）
  profile  │ name, family_name, given_name,
           │ middle_name, nickname, picture,
           │ preferred_username, website,
           │ gender, birthdate, zoneinfo,
           │ locale, updated_at
  email    │ email, email_verified
  address  │ address（構造化住所）
  phone    │ phone_number, phone_number_verified

  最小限の推奨:
  scope: "openid email profile"
  → ユーザー ID + メール + 名前・アイコン

UserInfo エンドポイント vs ID Token:
  ID Token:     認証時に最小限の情報を含む
  UserInfo:     詳細なプロフィール情報を取得
  使い分け:     ID Token で認証、UserInfo で追加情報取得
```

---

## 5. 認証フローの完全実装

```typescript
// OIDC 認証フロー（Next.js API Route）

// GET /api/auth/login — 認証開始
export async function GET(request: Request) {
  const state = crypto.randomUUID();
  const nonce = crypto.randomUUID();

  // セッションに state と nonce を保存
  const cookieStore = await cookies();
  cookieStore.set('oauth_state', state, { httpOnly: true, maxAge: 600 });
  cookieStore.set('oauth_nonce', nonce, { httpOnly: true, maxAge: 600 });

  const authUrl = await google.getAuthorizationUrl(
    'https://myapp.com/api/auth/callback',
    state,
    nonce
  );

  return Response.redirect(authUrl);
}

// GET /api/auth/callback — コールバック処理
export async function GET(request: Request) {
  const url = new URL(request.url);
  const code = url.searchParams.get('code');
  const state = url.searchParams.get('state');

  const cookieStore = await cookies();
  const storedState = cookieStore.get('oauth_state')?.value;
  const storedNonce = cookieStore.get('oauth_nonce')?.value;

  // state 検証
  if (!code || !state || state !== storedState) {
    return Response.redirect('/login?error=invalid_state');
  }

  // トークン交換
  const tokens = await google.exchangeCode(code, 'https://myapp.com/api/auth/callback');

  // ID Token 検証
  const user = await verifyIdToken(tokens.idToken, storedNonce!);

  // ユーザーの作成 or 更新
  const dbUser = await db.user.upsert({
    where: { providerId_provider: { providerId: user.sub, provider: 'google' } },
    create: {
      email: user.email,
      name: user.name,
      avatar: user.picture,
      provider: 'google',
      providerId: user.sub,
      emailVerified: user.emailVerified,
    },
    update: {
      name: user.name,
      avatar: user.picture,
      emailVerified: user.emailVerified,
    },
  });

  // セッション作成
  const sessionToken = await createSession(dbUser.id);

  // Cookie クリーンアップ
  cookieStore.delete('oauth_state');
  cookieStore.delete('oauth_nonce');

  // セッション Cookie 設定
  cookieStore.set('session', sessionToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 30 * 24 * 60 * 60,
  });

  return Response.redirect('/dashboard');
}
```

---

## 6. 主要プロバイダーの特徴

```
OIDC プロバイダー比較:

  プロバイダー │ Discovery │ PKCE │ 特記事項
  ────────────┼──────────┼─────┼──────────────
  Google      │ ✓        │ ✓   │ 最も標準準拠
  Microsoft   │ ✓        │ ✓   │ テナント別 issuer
  Apple       │ ✓        │ ✓   │ name は初回のみ返却
  GitHub      │ △        │ ✓   │ 標準 OIDC に完全準拠せず
  Auth0       │ ✓        │ ✓   │ カスタマイズ性高い
  Keycloak    │ ✓        │ ✓   │ セルフホスト可能

  Apple Sign In の注意点:
  → name / email は初回認可時のみ返却
  → 2回目以降は sub のみ
  → 初回レスポンスを確実に保存する必要がある
  → Web では redirect 方式のみ（ポップアップ不可の場合あり）

  GitHub の注意点:
  → 標準 OIDC ではなく独自 OAuth 実装
  → ID Token を返さない（/user API で取得）
  → email が null の場合がある（別 API で取得）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| OIDC | OAuth 2.0 上の認証レイヤー |
| ID Token | ユーザー情報を含む JWT（クライアント用） |
| Access Token | API アクセス用（リソースサーバー用） |
| Discovery | プロバイダー設定の自動取得 |
| nonce | リプレイ攻撃防止（ID Token に含める） |
| UserInfo | 詳細プロフィールの取得エンドポイント |

---

## 次に読むべきガイド
→ [[03-token-management.md]] — トークン管理

---

## 参考文献
1. OpenID Foundation. "OpenID Connect Core 1.0." openid.net/specs, 2014.
2. OpenID Foundation. "OpenID Connect Discovery 1.0." openid.net/specs, 2014.
3. Google. "OpenID Connect." developers.google.com, 2024.
