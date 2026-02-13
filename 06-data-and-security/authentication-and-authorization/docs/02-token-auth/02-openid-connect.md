# OpenID Connect

> OpenID Connect（OIDC）は OAuth 2.0 の上に構築された認証レイヤー。OAuth 2.0 が「認可」のプロトコルであるのに対し、OIDC は「認証」を標準化する。ID Token、UserInfo エンドポイント、Discovery、ソーシャルログインの基盤を解説する。

## 前提知識

- [[01-jwt-basics.md]] — JWT の基礎（署名検証、クレーム）
- [[../01-session-auth/00-cookie-and-session.md]] — Cookie とセッション管理
- OAuth 2.0 の基本概念（認可コードフロー、アクセストークン）
- HTTP の基礎（リダイレクト、ヘッダー）

## この章で学ぶこと

- [ ] OAuth 2.0 と OIDC の関係と根本的な違いを理解する
- [ ] ID Token の構造、クレーム、検証フローを完全に把握する
- [ ] OIDC Discovery の仕組みと活用方法を学ぶ
- [ ] 標準スコープとクレームの対応関係を理解する
- [ ] OIDC 認証フローの完全な実装を習得する
- [ ] 主要プロバイダーの差異と注意点を把握する
- [ ] OIDC のセキュリティ上の落とし穴を回避できるようになる

---

## 1. OIDC と OAuth 2.0 の関係

### 1.1 なぜ OIDC が必要か

OAuth 2.0 は本来「認可（Authorization）」のためのプロトコルであり、「認証（Authentication）」を目的としたものではない。OAuth 2.0 で得られるアクセストークンは「このトークンの持ち主にリソースへのアクセスを許可する」ことを意味するが、「このトークンの持ち主が誰であるか」は保証しない。

多くの開発者が OAuth 2.0 を認証に流用した結果（いわゆる「OAuth Dance」）、セキュリティ上の脆弱性が生じた。OIDC はこの問題を解決するために、OAuth 2.0 の上に認証のための標準化レイヤーを追加したものである。

```
OAuth 2.0 vs OpenID Connect:

  OAuth 2.0:
  → 目的: 認可（Authorization）
  → 質問: 「このアプリに何を許可しますか？」
  → 結果: アクセストークン（リソースアクセス用）
  → ユーザーが誰かは保証しない
  → RFC 6749, 6750

  OpenID Connect:
  → 目的: 認証（Authentication）
  → 質問: 「あなたは誰ですか？」
  → 結果: ID トークン（ユーザー情報）+ アクセストークン
  → ユーザーの身元を保証する
  → OpenID Connect Core 1.0

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
  │  + Session Management           │
  │  + Front-Channel Logout         │
  │  + Back-Channel Logout          │
  └──────────────────────────────────┘

  OIDC = OAuth 2.0 + 認証の標準化
```

### 1.2 OAuth 2.0 を認証に使う場合の問題

```
OAuth 2.0 で認証を試みる場合の脆弱性:

  問題 1: トークン置換攻撃（Token Substitution）
  ┌────────────────────────────────────────────┐
  │                                            │
  │  正規フロー:                                │
  │    ユーザー → App A → IdP → Access Token    │
  │    App A が Access Token で /userinfo       │
  │    → ユーザー情報を取得（OK）                 │
  │                                            │
  │  攻撃フロー:                                │
  │    攻撃者 → 悪意 App B → IdP → Access Token │
  │    攻撃者が App B の Access Token を盗む     │
  │    その Token を App A に送信                │
  │    App A が /userinfo → 攻撃者の情報取得     │
  │    → 攻撃者として App A にログイン（NG!）     │
  │                                            │
  │  OIDC の解決策:                              │
  │    ID Token の aud クレームで                 │
  │    「どのクライアント向けか」を検証             │
  │    → 他のクライアントの Token は拒否           │
  │                                            │
  └────────────────────────────────────────────┘

  問題 2: Access Token の不透明性
  → Access Token の形式は規定されていない
  → JWT かもしれないし opaque かもしれない
  → ユーザー情報が含まれる保証がない

  問題 3: 認証時刻の保証がない
  → Access Token は認証時刻を含まない
  → 古い Access Token が再利用される可能性
  → OIDC の auth_time クレームで解決

  問題 4: リプレイ攻撃
  → Access Token にはリプレイ防止の仕組みがない
  → OIDC の nonce クレームで解決
```

---

## 2. ID Token

### 2.1 ID Token の構造

ID Token は JWT（JSON Web Token）形式で、ユーザーの認証情報を含む。アクセストークンとは異なり、クライアントアプリケーション内で消費されることを目的としている。

```
ID Token の構造（JWT）:

  ┌─────────────────────────────────────────────────┐
  │  Header（ヘッダー）                               │
  │  {                                               │
  │    "alg": "RS256",       ← 署名アルゴリズム       │
  │    "typ": "JWT",                                 │
  │    "kid": "key-id-123"   ← 署名鍵の識別子         │
  │  }                                               │
  ├─────────────────────────────────────────────────┤
  │  Payload（ペイロード）                             │
  │  {                                               │
  │    // 必須クレーム                                │
  │    "iss": "https://accounts.google.com",         │
  │    "sub": "110169484474386276334",               │
  │    "aud": "my-client-id",                        │
  │    "exp": 1700000000,                            │
  │    "iat": 1699999100,                            │
  │                                                  │
  │    // 認証情報クレーム                              │
  │    "auth_time": 1699999000,                      │
  │    "nonce": "random-nonce-value",                │
  │    "acr": "urn:mace:incommon:iap:silver",        │
  │    "amr": ["pwd", "mfa"],                        │
  │    "azp": "my-client-id",                        │
  │                                                  │
  │    // ユーザー情報クレーム                          │
  │    "email": "alice@example.com",                 │
  │    "email_verified": true,                       │
  │    "name": "Alice Example",                      │
  │    "picture": "https://example.com/alice.jpg",   │
  │    "locale": "ja"                                │
  │  }                                               │
  ├─────────────────────────────────────────────────┤
  │  Signature（署名）                                │
  │  RS256(base64(header).base64(payload), secret)   │
  └─────────────────────────────────────────────────┘

  必須クレームの説明:
    iss (Issuer):     トークンの発行者 URL
    sub (Subject):    ユーザーの一意識別子（IdP内で一意）
    aud (Audience):   トークンの対象クライアント ID
    exp (Expiration): 有効期限（UNIX タイムスタンプ）
    iat (Issued At):  発行時刻（UNIX タイムスタンプ）

  重要な任意クレーム:
    auth_time:  実際に認証が行われた時刻
    nonce:      リプレイ攻撃防止用のランダム値
    acr:        認証コンテキストクラス（認証レベル）
    amr:        使用された認証方式のリスト
    azp:        認可されたパーティ（client_id）

  Access Token vs ID Token:
    Access Token: API アクセスに使用（リソースサーバーに送信）
    ID Token:    ユーザー情報の確認に使用（クライアント内で消費）

    ✗ ID Token を API アクセスに使用してはいけない
    ✗ Access Token からユーザー情報を取得してはいけない
    ✗ ID Token をリソースサーバーに送信してはいけない
```

### 2.2 ID Token の検証

ID Token の検証は OIDC のセキュリティの要であり、以下の全ステップを漏れなく実行する必要がある。

```
ID Token 検証の完全な手順:

  Step 1: JWT の形式検証
    → 3つのBase64URLエンコードされたパートに分割できるか
    → ヘッダーの alg が期待するアルゴリズムか（RS256等）
    → "none" アルゴリズムを絶対に受け入れない

  Step 2: 署名の検証
    → IdP の公開鍵（JWKS エンドポイント）で検証
    → kid ヘッダーで正しい鍵を選択
    → 鍵のローテーションに対応（キャッシュ + フォールバック）

  Step 3: iss（発行者）の検証
    → 期待する IdP の URL と完全一致するか
    → 例: "https://accounts.google.com"

  Step 4: aud（対象）の検証
    → 自分の client_id が含まれているか
    → 複数の aud がある場合は azp も検証

  Step 5: exp（有効期限）の検証
    → 現在時刻が exp より前か
    → クロックスキューを考慮（通常5分の猶予）

  Step 6: iat（発行時刻）の検証（推奨）
    → 未来の時刻でないか
    → 極端に古い時刻でないか

  Step 7: nonce の検証（認証リクエスト時に送信した場合）
    → セッションに保存した nonce と一致するか
    → リプレイ攻撃の防止

  Step 8: auth_time の検証（max_age を指定した場合）
    → 認証時刻が max_age 以内か
    → 例: 1時間以内の認証を要求
```

```typescript
// ID Token の完全な検証実装
import { jwtVerify, createRemoteJWKSet, JWTVerifyResult } from 'jose';

// Google の JWKS URL
const GOOGLE_JWKS = createRemoteJWKSet(
  new URL('https://www.googleapis.com/oauth2/v3/certs')
);

// JWKS のキャッシュ設定（jose ライブラリ内蔵）
// デフォルトで鍵はキャッシュされ、kid が見つからない場合に再取得される

interface VerifiedIdToken {
  sub: string;
  email: string;
  name: string;
  picture: string;
  emailVerified: boolean;
  authTime?: number;
  amr?: string[];
}

async function verifyIdToken(
  idToken: string,
  expectedNonce: string,
  options?: {
    maxAge?: number; // 秒
    requiredAmr?: string[]; // 必要な認証方式
  }
): Promise<VerifiedIdToken> {
  // Step 1-5: jose ライブラリが自動で検証
  // - JWT 形式チェック
  // - 署名検証（JWKS から公開鍵を取得）
  // - issuer の検証
  // - audience の検証
  // - exp の検証
  let result: JWTVerifyResult;
  try {
    result = await jwtVerify(idToken, GOOGLE_JWKS, {
      issuer: 'https://accounts.google.com',
      audience: process.env.GOOGLE_CLIENT_ID!,
      algorithms: ['RS256'],
      clockTolerance: 5, // 5秒のクロックスキュー許容
    });
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`ID Token verification failed: ${error.message}`);
    }
    throw error;
  }

  const { payload } = result;

  // Step 6: iat の追加検証
  if (payload.iat && payload.iat > Date.now() / 1000 + 5) {
    throw new Error('ID Token was issued in the future');
  }

  // Step 7: nonce の検証（リプレイ攻撃防止）
  if (payload.nonce !== expectedNonce) {
    throw new Error('Invalid nonce - possible replay attack');
  }

  // Step 8: auth_time の検証（max_age が指定された場合）
  if (options?.maxAge) {
    const authTime = payload.auth_time as number | undefined;
    if (!authTime) {
      throw new Error('auth_time claim is required when maxAge is specified');
    }
    if (Date.now() / 1000 - authTime > options.maxAge) {
      throw new Error(
        `Authentication is too old (${Math.floor(Date.now() / 1000 - authTime)}s > ${options.maxAge}s)`
      );
    }
  }

  // 追加: amr（認証方式）の検証
  if (options?.requiredAmr) {
    const amr = payload.amr as string[] | undefined;
    if (!amr) {
      throw new Error('amr claim is required');
    }
    for (const required of options.requiredAmr) {
      if (!amr.includes(required)) {
        throw new Error(`Required authentication method not used: ${required}`);
      }
    }
  }

  return {
    sub: payload.sub!,
    email: payload.email as string,
    name: payload.name as string,
    picture: payload.picture as string,
    emailVerified: payload.email_verified as boolean,
    authTime: payload.auth_time as number | undefined,
    amr: payload.amr as string[] | undefined,
  };
}
```

### 2.3 JWKS（JSON Web Key Set）の仕組み

```
JWKS の仕組み:

  IdP は公開鍵を JWKS エンドポイントで公開:
  GET https://www.googleapis.com/oauth2/v3/certs

  レスポンス:
  {
    "keys": [
      {
        "kty": "RSA",           ← 鍵タイプ
        "alg": "RS256",         ← アルゴリズム
        "kid": "key-id-1",      ← 鍵 ID（JWT の kid ヘッダーと対応）
        "use": "sig",           ← 用途（署名）
        "n": "0vx7a...",        ← RSA 公開鍵の modulus
        "e": "AQAB"             ← RSA 公開鍵の exponent
      },
      {
        "kty": "RSA",
        "alg": "RS256",
        "kid": "key-id-2",      ← 次の鍵（ローテーション用）
        "use": "sig",
        "n": "1wy8b...",
        "e": "AQAB"
      }
    ]
  }

  鍵のローテーション:
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  Time 0: key-1 がアクティブ                    │
  │    → 新しい ID Token は key-1 で署名           │
  │                                              │
  │  Time 1: key-2 を追加（JWKS に2つの鍵）       │
  │    → 新しい ID Token は key-2 で署名           │
  │    → key-1 で署名された古いトークンもまだ有効    │
  │                                              │
  │  Time 2: key-1 を削除                         │
  │    → key-1 で署名されたトークンは検証不可        │
  │    → 期限切れトークンなので問題なし              │
  │                                              │
  └──────────────────────────────────────────────┘

  クライアント側の対応:
  → JWKS をキャッシュ（通常24時間）
  → kid が見つからない場合は JWKS を再取得
  → Cache-Control ヘッダーを尊重
```

```typescript
// JWKS キャッシュの手動実装（jose ライブラリ内蔵のキャッシュを使わない場合）
class JWKSCache {
  private keys: Map<string, CryptoKey> = new Map();
  private lastFetch: number = 0;
  private readonly cacheDuration = 24 * 60 * 60 * 1000; // 24時間

  constructor(private jwksUrl: string) {}

  async getKey(kid: string): Promise<CryptoKey> {
    // キャッシュにある場合はそれを返す
    if (this.keys.has(kid) && Date.now() - this.lastFetch < this.cacheDuration) {
      return this.keys.get(kid)!;
    }

    // JWKS を再取得
    await this.refresh();

    const key = this.keys.get(kid);
    if (!key) {
      throw new Error(`Key with kid "${kid}" not found in JWKS`);
    }

    return key;
  }

  private async refresh(): Promise<void> {
    const res = await fetch(this.jwksUrl);
    const jwks = await res.json();

    this.keys.clear();
    for (const jwk of jwks.keys) {
      const key = await crypto.subtle.importKey(
        'jwk',
        jwk,
        { name: 'RSASSA-PKCS1-v1_5', hash: 'SHA-256' },
        false,
        ['verify']
      );
      this.keys.set(jwk.kid, key);
    }

    this.lastFetch = Date.now();
  }
}
```

---

## 3. OIDC Discovery

### 3.1 Discovery の仕組み

OpenID Connect Discovery は、IdP の設定情報を標準化された形式で自動取得する仕組みである。これにより、エンドポイント URL をハードコードする必要がなくなり、IdP の変更に自動的に追従できる。

```
OpenID Connect Discovery:

  プロバイダーの設定を自動取得:
  GET https://accounts.google.com/.well-known/openid-configuration

  レスポンス:
  {
    // 基本情報
    "issuer": "https://accounts.google.com",

    // エンドポイント
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
    "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    "revocation_endpoint": "https://oauth2.googleapis.com/revoke",

    // サポートする機能
    "scopes_supported": ["openid", "email", "profile"],
    "response_types_supported": ["code", "token", "id_token", "code token",
                                  "code id_token", "token id_token",
                                  "code token id_token"],
    "response_modes_supported": ["query", "fragment", "form_post"],
    "grant_types_supported": ["authorization_code", "implicit",
                               "refresh_token"],
    "subject_types_supported": ["public"],

    // 署名・暗号化
    "id_token_signing_alg_values_supported": ["RS256"],
    "token_endpoint_auth_methods_supported": ["client_secret_post",
                                              "client_secret_basic"],

    // クレーム
    "claims_supported": ["sub", "email", "email_verified",
                          "name", "given_name", "family_name",
                          "picture", "locale"]
  }

  Discovery の利点:
  → エンドポイント URL をハードコードしない
  → プロバイダーの変更に自動追従
  → 複数プロバイダーの一元管理
  → サポートする機能の動的な確認
  → IdP のバージョンアップに自動対応
```

### 3.2 汎用 OIDC プロバイダーの実装

```typescript
// OIDC Discovery を活用した汎用プロバイダー
interface OIDCConfig {
  issuer: string;
  authorization_endpoint: string;
  token_endpoint: string;
  userinfo_endpoint: string;
  jwks_uri: string;
  scopes_supported: string[];
  response_types_supported: string[];
  id_token_signing_alg_values_supported: string[];
  token_endpoint_auth_methods_supported: string[];
  claims_supported: string[];
  revocation_endpoint?: string;
  end_session_endpoint?: string;
}

class OIDCProvider {
  private config: OIDCConfig | null = null;
  private configFetchedAt: number = 0;
  private readonly configCacheDuration = 24 * 60 * 60 * 1000; // 24時間

  constructor(
    private issuerUrl: string,
    private clientId: string,
    private clientSecret: string
  ) {}

  // Discovery で設定を取得（キャッシュ付き）
  async discover(): Promise<OIDCConfig> {
    if (
      this.config &&
      Date.now() - this.configFetchedAt < this.configCacheDuration
    ) {
      return this.config;
    }

    const url = `${this.issuerUrl}/.well-known/openid-configuration`;
    const res = await fetch(url);

    if (!res.ok) {
      throw new Error(`OIDC Discovery failed: ${res.status} ${res.statusText}`);
    }

    this.config = await res.json();
    this.configFetchedAt = Date.now();

    // issuer の一致を検証
    if (this.config!.issuer !== this.issuerUrl) {
      throw new Error(
        `Issuer mismatch: expected ${this.issuerUrl}, got ${this.config!.issuer}`
      );
    }

    return this.config!;
  }

  // 認可 URL を生成
  async getAuthorizationUrl(
    redirectUri: string,
    state: string,
    nonce: string,
    options?: {
      scope?: string;
      prompt?: 'none' | 'login' | 'consent' | 'select_account';
      loginHint?: string;
      maxAge?: number;
      acrValues?: string;
    }
  ): Promise<string> {
    const config = await this.discover();
    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.clientId,
      redirect_uri: redirectUri,
      scope: options?.scope || 'openid email profile',
      state,
      nonce,
    });

    // オプションパラメータ
    if (options?.prompt) params.set('prompt', options.prompt);
    if (options?.loginHint) params.set('login_hint', options.loginHint);
    if (options?.maxAge !== undefined) params.set('max_age', String(options.maxAge));
    if (options?.acrValues) params.set('acr_values', options.acrValues);

    return `${config.authorization_endpoint}?${params}`;
  }

  // トークン交換
  async exchangeCode(
    code: string,
    redirectUri: string
  ): Promise<{
    accessToken: string;
    idToken: string;
    refreshToken?: string;
    expiresIn: number;
    tokenType: string;
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

    if (!res.ok) {
      const error = await res.json();
      throw new Error(
        `Token exchange failed: ${error.error} - ${error.error_description}`
      );
    }

    const data = await res.json();
    return {
      accessToken: data.access_token,
      idToken: data.id_token,
      refreshToken: data.refresh_token,
      expiresIn: data.expires_in,
      tokenType: data.token_type,
    };
  }

  // UserInfo エンドポイント
  async getUserInfo(accessToken: string): Promise<Record<string, any>> {
    const config = await this.discover();
    const res = await fetch(config.userinfo_endpoint, {
      headers: { Authorization: `Bearer ${accessToken}` },
    });

    if (!res.ok) {
      throw new Error(`UserInfo request failed: ${res.status}`);
    }

    return res.json();
  }

  // トークンの無効化
  async revokeToken(token: string, tokenTypeHint?: 'access_token' | 'refresh_token') {
    const config = await this.discover();
    if (!config.revocation_endpoint) {
      throw new Error('Revocation endpoint not supported');
    }

    const body = new URLSearchParams({
      token,
      client_id: this.clientId,
      client_secret: this.clientSecret,
    });
    if (tokenTypeHint) body.set('token_type_hint', tokenTypeHint);

    await fetch(config.revocation_endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    });
  }

  // OIDC ログアウト
  async getLogoutUrl(idTokenHint: string, postLogoutRedirectUri: string): Promise<string | null> {
    const config = await this.discover();
    if (!config.end_session_endpoint) return null;

    const params = new URLSearchParams({
      id_token_hint: idTokenHint,
      post_logout_redirect_uri: postLogoutRedirectUri,
      client_id: this.clientId,
    });

    return `${config.end_session_endpoint}?${params}`;
  }
}

// 使用例: 複数プロバイダーの一元管理
const providers = {
  google: new OIDCProvider(
    'https://accounts.google.com',
    process.env.GOOGLE_CLIENT_ID!,
    process.env.GOOGLE_CLIENT_SECRET!
  ),
  microsoft: new OIDCProvider(
    'https://login.microsoftonline.com/common/v2.0',
    process.env.MICROSOFT_CLIENT_ID!,
    process.env.MICROSOFT_CLIENT_SECRET!
  ),
  auth0: new OIDCProvider(
    `https://${process.env.AUTH0_DOMAIN}`,
    process.env.AUTH0_CLIENT_ID!,
    process.env.AUTH0_CLIENT_SECRET!
  ),
};
```

---

## 4. OIDC スコープとクレーム

### 4.1 標準スコープとクレームの対応

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

  address クレームの構造:
  {
    "formatted": "東京都千代田区...",
    "street_address": "千代田区...",
    "locality": "東京都",
    "region": "関東",
    "postal_code": "100-0001",
    "country": "JP"
  }

UserInfo エンドポイント vs ID Token:
  ID Token:     認証時に最小限の情報を含む
  UserInfo:     詳細なプロフィール情報を取得
  使い分け:     ID Token で認証、UserInfo で追加情報取得

  ID Token に含まれるクレーム:
  → 認証に必要な最小限の情報
  → iss, sub, aud, exp, iat, nonce
  → 追加情報はプロバイダーの裁量

  UserInfo で取得するクレーム:
  → scope で要求した詳細情報
  → プロフィール写真、住所、電話番号等
  → Access Token で認証して取得

  注意:
  → ID Token の情報は認証時点のスナップショット
  → UserInfo は最新の情報を返す
  → ユーザー情報の更新には UserInfo を使う
```

### 4.2 カスタムクレーム

```typescript
// Auth0 でカスタムクレームを追加する例
// Auth0 の Action / Rule で設定

// Action: Login / Post Login
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://myapp.com/claims';

  // カスタムクレームを ID Token に追加
  api.idToken.setCustomClaim(`${namespace}/roles`, event.authorization?.roles || []);
  api.idToken.setCustomClaim(`${namespace}/org_id`, event.user.app_metadata?.org_id);
  api.idToken.setCustomClaim(`${namespace}/permissions`, event.authorization?.permissions || []);

  // Access Token にもカスタムクレームを追加
  api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization?.roles || []);
};

// クライアント側でカスタムクレームを使用
interface CustomIdTokenClaims {
  sub: string;
  email: string;
  name: string;
  'https://myapp.com/claims/roles': string[];
  'https://myapp.com/claims/org_id': string;
  'https://myapp.com/claims/permissions': string[];
}

function extractCustomClaims(payload: any): {
  roles: string[];
  orgId: string;
  permissions: string[];
} {
  const namespace = 'https://myapp.com/claims';
  return {
    roles: payload[`${namespace}/roles`] || [],
    orgId: payload[`${namespace}/org_id`] || '',
    permissions: payload[`${namespace}/permissions`] || [],
  };
}
```

---

## 5. 認証フローの完全実装

### 5.1 Authorization Code Flow + PKCE

OIDC で推奨される認証フローは Authorization Code Flow with PKCE（Proof Key for Code Exchange）である。

```
OIDC Authorization Code Flow + PKCE:

  ブラウザ          サーバー          IdP
    │                │                │
    │ ログインクリック  │                │
    │───────────────>│                │
    │                │                │
    │                │ code_verifier 生成│
    │                │ code_challenge   │
    │                │ = SHA256(        │
    │                │   code_verifier) │
    │                │                │
    │                │ state, nonce 生成│
    │                │ セッションに保存  │
    │                │                │
    │ 302 Redirect   │                │
    │<───────────────│                │
    │                                 │
    │ GET /authorize?                 │
    │   response_type=code            │
    │   client_id=xxx                 │
    │   redirect_uri=xxx              │
    │   scope=openid email profile    │
    │   state=xxx                     │
    │   nonce=xxx                     │
    │   code_challenge=xxx            │
    │   code_challenge_method=S256    │
    │────────────────────────────────>│
    │                                 │
    │         ログイン画面              │
    │<────────────────────────────────│
    │ 認証情報入力                      │
    │────────────────────────────────>│
    │                                 │
    │ 302 Redirect                    │
    │ ?code=AUTH_CODE&state=xxx       │
    │<────────────────────────────────│
    │                                 │
    │ GET /callback                   │
    │   ?code=AUTH_CODE               │
    │   &state=xxx                    │
    │───────────────>│                │
    │                │                │
    │                │ state 検証      │
    │                │                │
    │                │ POST /token     │
    │                │   code=AUTH_CODE│
    │                │   code_verifier │
    │                │────────────────>│
    │                │                │
    │                │ ID Token +      │
    │                │ Access Token    │
    │                │<────────────────│
    │                │                │
    │                │ ID Token 検証   │
    │                │ nonce 検証      │
    │                │ セッション作成   │
    │                │                │
    │ Set-Cookie     │                │
    │<───────────────│                │
```

### 5.2 Next.js での完全実装

```typescript
// OIDC 認証フロー（Next.js App Router）
import crypto from 'crypto';
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import { jwtVerify, createRemoteJWKSet } from 'jose';

// PKCE: code_verifier と code_challenge の生成
function generatePKCE(): { codeVerifier: string; codeChallenge: string } {
  // code_verifier: 43-128文字のランダム文字列
  const codeVerifier = crypto.randomBytes(32).toString('base64url');

  // code_challenge: SHA256(code_verifier) の Base64URL エンコード
  const codeChallenge = crypto
    .createHash('sha256')
    .update(codeVerifier)
    .digest('base64url');

  return { codeVerifier, codeChallenge };
}

// GET /api/auth/login — 認証開始
export async function GET(request: Request) {
  const state = crypto.randomUUID();
  const nonce = crypto.randomUUID();
  const { codeVerifier, codeChallenge } = generatePKCE();

  // セッションに state, nonce, code_verifier を保存
  const cookieStore = await cookies();
  cookieStore.set('oauth_state', state, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 600, // 10分
    path: '/',
  });
  cookieStore.set('oauth_nonce', nonce, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 600,
    path: '/',
  });
  cookieStore.set('oauth_code_verifier', codeVerifier, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    maxAge: 600,
    path: '/',
  });

  // 認可リクエスト URL の構築
  const provider = providers.google;
  const authUrl = await provider.getAuthorizationUrl(
    `${process.env.APP_URL}/api/auth/callback`,
    state,
    nonce,
    {
      prompt: 'select_account', // アカウント選択画面を表示
    }
  );

  // PKCE パラメータを追加
  const url = new URL(authUrl);
  url.searchParams.set('code_challenge', codeChallenge);
  url.searchParams.set('code_challenge_method', 'S256');

  return Response.redirect(url.toString());
}

// GET /api/auth/callback — コールバック処理
export async function GET(request: Request) {
  const url = new URL(request.url);
  const code = url.searchParams.get('code');
  const state = url.searchParams.get('state');
  const error = url.searchParams.get('error');

  // エラーチェック（ユーザーがキャンセルした場合等）
  if (error) {
    const errorDescription = url.searchParams.get('error_description') || 'Unknown error';
    console.error(`OIDC error: ${error} - ${errorDescription}`);
    return Response.redirect(`${process.env.APP_URL}/login?error=${error}`);
  }

  const cookieStore = await cookies();
  const storedState = cookieStore.get('oauth_state')?.value;
  const storedNonce = cookieStore.get('oauth_nonce')?.value;
  const codeVerifier = cookieStore.get('oauth_code_verifier')?.value;

  // state 検証（CSRF 防止）
  if (!code || !state || state !== storedState) {
    return Response.redirect(`${process.env.APP_URL}/login?error=invalid_state`);
  }

  if (!storedNonce || !codeVerifier) {
    return Response.redirect(`${process.env.APP_URL}/login?error=missing_session`);
  }

  try {
    // トークン交換（code_verifier を含む）
    const tokens = await providers.google.exchangeCode(
      code,
      `${process.env.APP_URL}/api/auth/callback`
    );

    // ID Token 検証
    const userInfo = await verifyIdToken(tokens.idToken, storedNonce, {
      maxAge: 3600, // 1時間以内の認証を要求
    });

    // ユーザーの作成 or 更新（Upsert）
    const dbUser = await prisma.user.upsert({
      where: {
        provider_providerId: {
          provider: 'google',
          providerId: userInfo.sub,
        },
      },
      create: {
        email: userInfo.email,
        name: userInfo.name,
        avatar: userInfo.picture,
        provider: 'google',
        providerId: userInfo.sub,
        emailVerified: userInfo.emailVerified ? new Date() : null,
      },
      update: {
        name: userInfo.name,
        avatar: userInfo.picture,
        emailVerified: userInfo.emailVerified ? new Date() : null,
        lastLoginAt: new Date(),
      },
    });

    // セッション作成
    const sessionToken = await createSession(dbUser.id);

    // Cookie クリーンアップ
    cookieStore.delete('oauth_state');
    cookieStore.delete('oauth_nonce');
    cookieStore.delete('oauth_code_verifier');

    // セッション Cookie 設定
    cookieStore.set('session', sessionToken, {
      httpOnly: true,
      secure: true,
      sameSite: 'lax',
      maxAge: 30 * 24 * 60 * 60, // 30日
      path: '/',
    });

    return Response.redirect(`${process.env.APP_URL}/dashboard`);
  } catch (error) {
    console.error('OIDC callback error:', error);
    return Response.redirect(`${process.env.APP_URL}/login?error=auth_failed`);
  }
}

// GET /api/auth/logout — ログアウト
export async function GET(request: Request) {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get('session')?.value;

  if (sessionToken) {
    // セッション削除
    await deleteSession(sessionToken);

    // IdP のログアウト URL を取得（オプション）
    const logoutUrl = await providers.google.getLogoutUrl(
      sessionToken, // ID Token Hint（保存している場合）
      `${process.env.APP_URL}/`
    );

    cookieStore.delete('session');

    if (logoutUrl) {
      return Response.redirect(logoutUrl);
    }
  }

  return Response.redirect(`${process.env.APP_URL}/`);
}
```

### 5.3 アカウントリンク（複数プロバイダー対応）

```typescript
// 同一ユーザーが複数のプロバイダーでログインする場合
// schema.prisma のモデル設計

// model User {
//   id            String    @id @default(cuid())
//   email         String    @unique
//   name          String?
//   avatar        String?
//   accounts      Account[]
//   sessions      Session[]
// }
//
// model Account {
//   id                String  @id @default(cuid())
//   userId            String
//   provider          String  // "google", "github", "microsoft"
//   providerAccountId String  // IdP 側のユーザー ID
//   accessToken       String?
//   refreshToken      String?
//   expiresAt         Int?
//   tokenType         String?
//   scope             String?
//   idToken           String?
//
//   user User @relation(fields: [userId], references: [id])
//
//   @@unique([provider, providerAccountId])
// }

// アカウントリンクの実装
async function handleOIDCCallback(
  provider: string,
  userInfo: VerifiedIdToken,
  tokens: TokenResponse
) {
  // 1. 既存のアカウントリンクを確認
  const existingAccount = await prisma.account.findUnique({
    where: {
      provider_providerAccountId: {
        provider,
        providerAccountId: userInfo.sub,
      },
    },
    include: { user: true },
  });

  if (existingAccount) {
    // 既存アカウントでログイン
    await prisma.account.update({
      where: { id: existingAccount.id },
      data: {
        accessToken: tokens.accessToken,
        refreshToken: tokens.refreshToken,
        expiresAt: tokens.expiresIn ? Math.floor(Date.now() / 1000) + tokens.expiresIn : null,
        idToken: tokens.idToken,
      },
    });
    return existingAccount.user;
  }

  // 2. メールアドレスで既存ユーザーを検索
  if (userInfo.email && userInfo.emailVerified) {
    const existingUser = await prisma.user.findUnique({
      where: { email: userInfo.email },
    });

    if (existingUser) {
      // 既存ユーザーに新しいプロバイダーをリンク
      await prisma.account.create({
        data: {
          userId: existingUser.id,
          provider,
          providerAccountId: userInfo.sub,
          accessToken: tokens.accessToken,
          refreshToken: tokens.refreshToken,
          idToken: tokens.idToken,
        },
      });
      return existingUser;
    }
  }

  // 3. 新規ユーザー作成
  const newUser = await prisma.user.create({
    data: {
      email: userInfo.email,
      name: userInfo.name,
      avatar: userInfo.picture,
      accounts: {
        create: {
          provider,
          providerAccountId: userInfo.sub,
          accessToken: tokens.accessToken,
          refreshToken: tokens.refreshToken,
          idToken: tokens.idToken,
        },
      },
    },
  });

  return newUser;
}
```

---

## 6. 主要プロバイダーの特徴と注意点

### 6.1 プロバイダー比較

```
OIDC プロバイダー比較:

  プロバイダー │ Discovery │ PKCE │ Refresh │ 特記事項
  ────────────┼──────────┼─────┼────────┼──────────────
  Google      │ ✓        │ ✓   │ ✓      │ 最も標準準拠
  Microsoft   │ ✓        │ ✓   │ ✓      │ テナント別 issuer
  Apple       │ ✓        │ ✓   │ ✓      │ name は初回のみ返却
  GitHub      │ △        │ ✓   │ ✓      │ 標準 OIDC に非準拠
  LINE        │ ✓        │ ✗   │ ✓      │ 日本市場で重要
  Auth0       │ ✓        │ ✓   │ ✓      │ カスタマイズ性高い
  Keycloak    │ ✓        │ ✓   │ ✓      │ セルフホスト可能
  Okta        │ ✓        │ ✓   │ ✓      │ エンタープライズ向け
```

### 6.2 各プロバイダーの注意点

```
Apple Sign In の注意点:
  → name / email は初回認可時のみ返却
  → 2回目以降は sub のみ（アプリ再インストール含む）
  → 初回レスポンスを確実に保存する必要がある
  → Web では redirect 方式のみ（ポップアップ不可の場合あり）
  → App Store に出すアプリは Apple Sign In 対応必須
  → ユーザーはメールを非公開にできる（リレーメール）
  → client_secret は JWT 形式で自分で生成する必要がある

  Apple のメール非公開の仕組み:
  ┌──────────────────────────────────────────┐
  │                                          │
  │  ユーザーが「メールを非公開」を選択した場合   │
  │                                          │
  │  提供されるメール:                          │
  │  abc123@privaterelay.appleid.com         │
  │                                          │
  │  このアドレスに送信したメールは               │
  │  ユーザーの実際のメールに転送される            │
  │                                          │
  │  注意:                                    │
  │  → 自社ドメインを Apple に登録する必要あり   │
  │  → SPF/DKIM の設定が必要                   │
  │  → リレーメールは永続的ではない場合がある      │
  │                                          │
  └──────────────────────────────────────────┘

GitHub の注意点:
  → 標準 OIDC ではなく独自 OAuth 実装
  → ID Token を返さない（/user API で取得）
  → email が null の場合がある（非公開設定）
  → 別途 GET /user/emails API で取得が必要
  → Discovery エンドポイントがない（または限定的）
  → scope は "user:email" のように GitHub 独自形式

Microsoft / Azure AD の注意点:
  → テナント別 issuer URL
  → common: https://login.microsoftonline.com/common/v2.0
  → 特定テナント: https://login.microsoftonline.com/{tenant-id}/v2.0
  → ID Token の iss がテナント ID を含む
  → 個人アカウント (MSA) と組織アカウント (AAD) で動作が異なる
  → v1.0 と v2.0 エンドポイントがある（v2.0 を使用）
```

### 6.3 GitHub の OIDC 非準拠への対応

```typescript
// GitHub は標準 OIDC ではないため、独自実装が必要
class GitHubOAuthProvider {
  private readonly authUrl = 'https://github.com/login/oauth/authorize';
  private readonly tokenUrl = 'https://github.com/login/oauth/access_token';
  private readonly apiUrl = 'https://api.github.com';

  constructor(
    private clientId: string,
    private clientSecret: string
  ) {}

  getAuthorizationUrl(redirectUri: string, state: string): string {
    const params = new URLSearchParams({
      client_id: this.clientId,
      redirect_uri: redirectUri,
      scope: 'user:email read:user',
      state,
    });
    return `${this.authUrl}?${params}`;
  }

  async exchangeCode(code: string): Promise<string> {
    const res = await fetch(this.tokenUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({
        client_id: this.clientId,
        client_secret: this.clientSecret,
        code,
      }),
    });

    const data = await res.json();
    if (data.error) {
      throw new Error(`GitHub token exchange failed: ${data.error_description}`);
    }
    return data.access_token;
  }

  // GitHub の /user API でユーザー情報取得（ID Token の代替）
  async getUser(accessToken: string) {
    const [userRes, emailsRes] = await Promise.all([
      fetch(`${this.apiUrl}/user`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      }),
      fetch(`${this.apiUrl}/user/emails`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      }),
    ]);

    const user = await userRes.json();
    const emails: Array<{ email: string; primary: boolean; verified: boolean }> =
      await emailsRes.json();

    // プライマリかつ認証済みメールを取得
    const primaryEmail = emails.find((e) => e.primary && e.verified);

    return {
      sub: String(user.id),
      email: primaryEmail?.email || user.email,
      emailVerified: primaryEmail?.verified ?? false,
      name: user.name || user.login,
      picture: user.avatar_url,
      login: user.login, // GitHub 固有: ユーザー名
    };
  }
}
```

---

## 7. エッジケースとセキュリティ

### 7.1 エッジケース

```
OIDC のエッジケース:

  (1) メールアドレスの変更
     → ユーザーが IdP 側でメールを変更した場合
     → sub は変わらないがメールが変わる
     → sub をプライマリキーとして使用する
     → メールを一意制約にしない or 更新を追従

  (2) アカウント削除と再作成
     → ユーザーが IdP アカウントを削除して再作成
     → Google: 同じメールでも sub が変わる場合がある
     → Apple: sub は永続（デバイスリセットでも）
     → sub + provider の組み合わせで識別

  (3) トークンの失効とリフレッシュ
     → Access Token の有効期限切れ
     → Refresh Token でサイレントリフレッシュ
     → Refresh Token も失効した場合は再認証

  (4) 複数タブでの同時認証
     → 複数タブで同時にログインフローを開始
     → state/nonce が異なるため片方は失敗
     → state を Cookie に保存（最後の値が有効）
     → 対策: ログイン中は他のタブをブロック or 共有

  (5) IdP のダウンタイム
     → Discovery エンドポイントが応答しない
     → JWKS エンドポイントが応答しない
     → 設定と鍵のキャッシュで一時的に対応
     → フォールバック認証方式の提供
```

### 7.2 セキュリティのベストプラクティス

```
OIDC セキュリティのベストプラクティス:

  ✓ 必ず実行すべきこと:
    → state パラメータで CSRF を防止
    → nonce パラメータでリプレイ攻撃を防止
    → PKCE を使用（特にパブリッククライアント）
    → ID Token の全クレームを検証
    → HTTPS を必須にする
    → redirect_uri を厳密に検証（完全一致）
    → client_secret を安全に管理

  ✗ 避けるべきこと:
    → ID Token を API アクセスに使用
    → Access Token でユーザー認証
    → Implicit Flow の使用（非推奨）
    → redirect_uri にワイルドカード
    → client_secret をフロントエンドに含める
    → ID Token のペイロードを検証なしで使用
    → "none" アルゴリズムの受け入れ

  redirect_uri の攻撃:
  ┌────────────────────────────────────────┐
  │                                        │
  │  Open Redirect 攻撃:                    │
  │    攻撃者が redirect_uri を                │
  │    自分のサーバーに書き換え                  │
  │    → 認可コードが攻撃者に送信される          │
  │                                        │
  │  対策:                                  │
  │    → 登録済み redirect_uri のみ許可       │
  │    → 完全一致で検証（前方一致NG）           │
  │    → ワイルドカード禁止                    │
  │    → localhost はデバッグ時のみ許可         │
  │                                        │
  └────────────────────────────────────────┘
```

### 7.3 Token Leakage の防止

```typescript
// トークン漏洩の防止策

// 1. Authorization Code は1回のみ使用可能
// → IdP 側で実装されているが、クライアント側でも確認

// 2. Token をログに出力しない
function safeLog(message: string, data: any) {
  const sanitized = { ...data };
  const sensitiveFields = ['access_token', 'id_token', 'refresh_token', 'code', 'client_secret'];
  for (const field of sensitiveFields) {
    if (sanitized[field]) {
      sanitized[field] = `[REDACTED:${sanitized[field].length}chars]`;
    }
  }
  console.log(message, sanitized);
}

// 3. Referrer-Policy で Token の漏洩を防止
// HTTP ヘッダー:
// Referrer-Policy: no-referrer
// → リダイレクト時に URL パラメータ（code, state）が漏洩しない

// 4. Token の安全な保存
// → Access Token: メモリ内（変数）
// → Refresh Token: HttpOnly Cookie or サーバーサイド
// → ID Token: 検証後に破棄（必要な情報はセッションに保存）
```

---

## 8. アンチパターン

```
OIDC のアンチパターン:

  (1) ID Token を API の認証に使用
     ✗ 悪い例:
       fetch('/api/data', {
         headers: { Authorization: `Bearer ${idToken}` }
       });
     → ID Token はクライアント向け
     → API には Access Token を使用する

  (2) JWT の検証を省略
     ✗ 悪い例:
       const payload = JSON.parse(atob(token.split('.')[1]));
       // 署名検証なしでペイロードを使用
     → 改ざんされたトークンを受け入れてしまう
     → 必ず署名を検証する

  (3) sub 以外でユーザーを識別
     ✗ 悪い例:
       // メールアドレスでユーザーを識別
       const user = await db.user.findUnique({ where: { email: payload.email } });
     → メールは変更される可能性がある
     → sub + provider の組み合わせで識別する

  (4) Implicit Flow の使用
     ✗ 悪い例:
       response_type: 'id_token token'
     → Access Token がフラグメントで公開される
     → Authorization Code Flow + PKCE を使用する
```

---

## 9. パフォーマンスに関する考察

```
OIDC のパフォーマンス最適化:

  (1) Discovery のキャッシュ
     → .well-known/openid-configuration は頻繁に変わらない
     → 24時間のキャッシュが一般的
     → アプリ起動時に事前取得（warm-up）

  (2) JWKS のキャッシュ
     → 公開鍵は鍵ローテーション時のみ変更
     → kid が見つからない場合のみ再取得
     → Cache-Control ヘッダーを尊重

  (3) Token 交換のレイテンシ
     → IdP のトークンエンドポイントへの HTTP リクエスト
     → 通常 100-500ms のレイテンシ
     → ログインフロー全体で 1-3 秒

  (4) UserInfo リクエストの最適化
     → 必要な情報が ID Token にあれば UserInfo は不要
     → UserInfo の結果はキャッシュ可能
     → Access Token の有効期間中のみキャッシュ

  (5) セッションの活用
     → 認証後はセッションベースに切り替え
     → 毎回 IdP と通信しない
     → セッション有効期限の設計が重要
```

---

## 10. 演習問題

### 演習 1: OIDC 基本実装（基礎）

Google OIDC を使ったログイン機能を実装せよ。

```
要件:
- Discovery で設定を自動取得
- Authorization Code Flow + PKCE
- ID Token の完全な検証（issuer, audience, nonce, exp）
- ユーザー情報の DB 保存
- セッション Cookie の設定

環境:
- Node.js + Express or Next.js
- jose ライブラリ（JWT検証）
- Google Cloud Console でクライアント ID 取得済み

テスト:
- 正常系: ログイン成功 → セッション作成
- 異常系: state 不一致、nonce 不一致、期限切れ ID Token
```

### 演習 2: マルチプロバイダー対応（応用）

Google + GitHub + Apple のマルチプロバイダーログインを実装せよ。

```
要件:
- 汎用 OIDCProvider クラスを拡張
- GitHub の非標準 OAuth に対応
- Apple Sign In の初回データ保存に対応
- メールアドレスベースのアカウントリンク
- 同一メールで異なるプロバイダーからのログインをマージ

テスト:
- 各プロバイダーでログイン成功
- 同一メールの自動リンク
- プロバイダー追加（既存ユーザーに新プロバイダーをリンク）
```

### 演習 3: セキュリティ強化（発展）

OIDC 認証にセキュリティ強化機能を追加せよ。

```
要件:
- ステップアップ認証（sensitive な操作時に再認証要求）
  → max_age=0 で認証を強制
  → acr_values で認証レベルを指定
- RP-Initiated Logout の実装
- Back-Channel Logout の受信
- Token Binding（DPoP）の概念理解
- 不正アクセス検知（異常な認証パターンの検出）

実装:
- ステップアップ認証ミドルウェア
- ログアウト API + IdP 連携
- セキュリティイベントの監査ログ
```

---

## 11. FAQ・トラブルシューティング

### Q1: "invalid_grant" エラーが発生する

**原因**: 認可コードの二重使用、コードの期限切れ、redirect_uri の不一致が一般的。

```
対処法:
1. 認可コードは1回のみ使用可能 → ブラウザの「戻る」ボタンで再送信されていないか
2. 認可コードの有効期限は通常10分 → ユーザーが長時間放置していないか
3. redirect_uri がトークン交換時と認可リクエスト時で完全一致するか
4. client_id / client_secret が正しいか
```

### Q2: ID Token の署名検証に失敗する

**原因**: JWKS の鍵ローテーション、kid の不一致、アルゴリズムの不一致が考えられる。

```
対処法:
1. JWKS キャッシュを強制的にクリアして再取得
2. ID Token ヘッダーの kid が JWKS に含まれているか確認
3. alg が期待するアルゴリズム（RS256）か確認
4. IdP 側で鍵ローテーションが行われた可能性
```

### Q3: UserInfo で取得できる情報が少ない

**原因**: scope の不足、プロバイダーの制限、ユーザーの設定。

```
対処法:
1. 認可リクエストの scope に必要なスコープが含まれているか
2. プロバイダーによってはスコープの承認が必要
3. Apple: name は初回のみ、GitHub: email は非公開の場合あり
4. Google: 追加スコープは Google Cloud Console で有効化が必要
```

### Q4: Silent Refresh が動作しない

**原因**: サードパーティ Cookie の制限、Refresh Token の失効。

```
対処法:
1. ITP/ETP によりサードパーティ Cookie がブロック
2. prompt=none による Silent Auth は Cookie 制限の影響を受ける
3. サーバーサイドでの Refresh Token 使用が推奨
4. Refresh Token の有効期限を確認（Google: 7日〜無期限）
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| OIDC | OAuth 2.0 上の認証レイヤー。認証と認可の明確な分離 |
| ID Token | ユーザー情報を含む JWT。クライアント内で消費する |
| Access Token | API アクセス用。リソースサーバーに送信する |
| Discovery | プロバイダー設定の自動取得。キャッシュ推奨 |
| PKCE | 認可コード横取り攻撃の防止。全クライアントで推奨 |
| nonce | リプレイ攻撃防止。ID Token に含め検証する |
| UserInfo | 詳細プロフィールの取得。Access Token で認証 |
| JWKS | 公開鍵の配布。鍵ローテーション対応が必要 |
| アカウントリンク | sub + provider で識別。メールベースのマージ |

---

## 次に読むべきガイド
→ [[03-token-management.md]] — トークン管理

---

## 参考文献
1. OpenID Foundation. "OpenID Connect Core 1.0." openid.net/specs/openid-connect-core-1_0.html, 2014.
2. OpenID Foundation. "OpenID Connect Discovery 1.0." openid.net/specs/openid-connect-discovery-1_0.html, 2014.
3. RFC 7636. "Proof Key for Code Exchange by OAuth Public Clients." IETF, 2015.
4. RFC 7517. "JSON Web Key (JWK)." IETF, 2015.
5. Google. "OpenID Connect." developers.google.com/identity/openid-connect, 2024.
6. Apple. "Sign in with Apple." developer.apple.com/sign-in-with-apple, 2024.
7. Microsoft. "Microsoft identity platform and OpenID Connect protocol." learn.microsoft.com, 2024.
8. OWASP. "Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
