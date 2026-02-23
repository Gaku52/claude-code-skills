# 認証方式

> Web APIの認証方式を体系的に理解する。Basic認証、Bearer Token、OAuth 2.0、JWTの仕組みと使い分けを学び、安全な認証システムを設計する。

## この章で学ぶこと

- [ ] 主要な認証方式の仕組みと違いを理解する
- [ ] OAuth 2.0のフローを把握する
- [ ] JWTの構造とセキュリティ上の注意点を学ぶ
- [ ] OpenID Connectによる認証の実装パターンを習得する
- [ ] パスキー・多要素認証など最新の認証技術を理解する

---

## 1. 認証と認可

```
認証（Authentication）:
  → 「あなたは誰？」
  → ユーザーの本人確認
  → ログイン処理

認可（Authorization）:
  → 「あなたは何ができる？」
  → アクセス権限の確認
  → ロールベースアクセス制御（RBAC）

  例:
  認証: ログインして「Taroさん」であることを確認
  認可: 「Taroさん」は管理者なので全データにアクセス可能

認証と認可の関係:
  ┌─────────────────────────────────────────┐
  │  リクエスト                              │
  │    │                                     │
  │    ▼                                     │
  │  認証（Authentication）                  │
  │    → ユーザーを特定                      │
  │    → 失敗 → 401 Unauthorized            │
  │    │                                     │
  │    ▼                                     │
  │  認可（Authorization）                   │
  │    → 権限を確認                          │
  │    → 失敗 → 403 Forbidden              │
  │    │                                     │
  │    ▼                                     │
  │  リソースアクセス                        │
  └─────────────────────────────────────────┘

認証の3要素（Authentication Factors）:
  ① 知識要素（Something you know）: パスワード、PIN
  ② 所持要素（Something you have）: スマホ、セキュリティキー
  ③ 生体要素（Something you are）: 指紋、顔認証

  多要素認証（MFA）= 2つ以上の要素の組み合わせ
  二要素認証（2FA）= 2つの要素の組み合わせ
```

### 1.1 認可モデル

```
主要な認可モデル:

① RBAC（Role-Based Access Control）:
  → ロール（役割）に権限を割り当て
  → ユーザーにロールを付与

  ロール定義:
  admin:  read, write, delete, manage_users
  editor: read, write
  viewer: read

  ユーザー → ロール → 権限:
  Taro  → admin  → 全操作可能
  Hanako → editor → 読み書き可能
  Jiro  → viewer → 読み取りのみ

② ABAC（Attribute-Based Access Control）:
  → 属性（ユーザー属性、リソース属性、環境属性）で判定
  → より柔軟だが複雑

  例: 「所属部署が営業部で、勤務時間内で、社内ネットワークから
       アクセスした場合のみ、顧客データの読み取りを許可」

③ ReBAC（Relationship-Based Access Control）:
  → エンティティ間の関係で権限を判定
  → Google Zanzibar / OpenFGA

  例: 「ドキュメントのオーナーまたは共有されたユーザーのみ閲覧可能」
  → user:taro → owner → document:123
  → user:hanako → viewer → document:123

④ ACL（Access Control List）:
  → リソースごとにアクセス許可リスト
  → シンプルだがスケールしにくい

実務での選択:
  小規模アプリ → RBAC（シンプルで十分）
  複雑な権限要件 → ABAC
  ソーシャル/コラボレーション → ReBAC
  ファイルシステム → ACL
```

---

## 2. Basic認証

```
Basic認証:
  → ユーザー名:パスワードをBase64エンコード
  → リクエストごとに送信

  Authorization: Basic dGFybzpwYXNzd29yZA==
                       ↑ "taro:password" のBase64

  リクエスト/レスポンスの流れ:
  1. クライアント → サーバー: 認証なしでリクエスト
  2. サーバー → クライアント: 401 + WWW-Authenticate: Basic realm="API"
  3. クライアント → サーバー: Authorization: Basic <credentials>
  4. サーバー → クライアント: 200 OK

  利点:
  ✓ 実装が極めてシンプル
  ✓ サーバー側の状態管理不要
  ✓ HTTP標準仕様（RFC 7617）

  欠点:
  ✗ パスワードが毎回送信される（Base64は暗号化ではない）
  ✗ HTTPS必須（平文で流れる）
  ✗ ログアウト機能がない
  ✗ ブルートフォース攻撃に弱い
  ✗ パスワード変更時の対応が困難

  用途:
  → 内部ツール、CI/CDのAPI認証
  → 本番APIには非推奨
  → Docker Registry の認証（内部用）
  → 開発環境のアクセス制限
```

```python
# Basic認証の実装例（Python/FastAPI）
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

app = FastAPI()
security = HTTPBasic()

# 安全な比較（タイミング攻撃対策）
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(
        credentials.username.encode("utf8"),
        b"admin"
    )
    correct_password = secrets.compare_digest(
        credentials.password.encode("utf8"),
        b"secret"
    )
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@app.get("/api/data")
def read_data(username: str = Depends(verify_credentials)):
    return {"message": f"Hello, {username}"}
```

---

## 3. Bearer Token / API Key

```
Bearer Token:
  Authorization: Bearer eyJhbGciOiJIUzI1NiIs...

  → サーバーが発行したトークンをヘッダーに含める
  → トークンの形式は自由（JWT、ランダム文字列等）
  → RFC 6750で規定

API Key:
  X-API-Key: sk_live_abcdef123456
  または
  ?api_key=sk_live_abcdef123456

  → サービス間連携でよく使用
  → ユーザーではなくアプリケーションの認証
  → プレフィックスで種類を区別:
     sk_live_  → 本番用シークレットキー
     sk_test_  → テスト用シークレットキー
     pk_live_  → 本番用公開キー

  API Keyのベストプラクティス:
  ✓ ヘッダーで送信（URLに含めない）
  ✓ 環境変数で管理（コードにハードコードしない）
  ✓ ローテーション可能な設計
  ✓ スコープ/権限の制限
  ✓ レート制限の適用
  ✓ 監査ログの記録
```

### 3.1 セッションベース vs トークンベース

```
セッションベース vs トークンベース:
  ┌──────────────┬──────────────────┬──────────────────┐
  │              │ セッション        │ トークン          │
  ├──────────────┼──────────────────┼──────────────────┤
  │ 状態管理     │ サーバー側        │ クライアント側    │
  │ スケーラビリティ│ 低い（共有必要）│ 高い（ステートレス）│
  │ 無効化       │ 容易             │ 困難              │
  │ ストレージ   │ サーバーメモリ/DB│ クライアント       │
  │ CSRF        │ 脆弱             │ 安全              │
  │ XSS         │ HttpOnly Cookie  │ 注意が必要        │
  │ モバイル対応 │ Cookie管理が複雑 │ 容易              │
  │ マイクロサービス│ セッション共有問題│ 各サービスで検証可│
  └──────────────┴──────────────────┴──────────────────┘

セッションベース認証のフロー:
  1. POST /login { username, password }
  2. サーバー: セッション生成 → Redis/DB等に保存
  3. Set-Cookie: session_id=abc123; HttpOnly; Secure; SameSite=Lax
  4. クライアント: 以降のリクエストにCookieが自動送信
  5. サーバー: Cookieからセッションを復元、ユーザーを特定

トークンベース認証のフロー:
  1. POST /login { username, password }
  2. サーバー: JWT生成 → レスポンスボディで返却
  3. クライアント: トークンをメモリ/ローカルストレージに保存
  4. クライアント: Authorization: Bearer <token> を付与
  5. サーバー: トークンの署名を検証、ユーザーを特定

ハイブリッドアプローチ（推奨）:
  → JWTをHttpOnly Cookieに格納
  → CSRF対策: SameSite=Strict + CSRFトークン
  → XSS対策: HttpOnly（JSからアクセス不可）
  → セッション管理: Refresh TokenをDBで管理（無効化可能）
```

---

## 4. JWT（JSON Web Token）

```
JWTの構造:
  eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3.SflKxwRJSMeKKF2QT4
  ↑ ヘッダー               ↑ ペイロード          ↑ 署名

  ヘッダー（Base64URL）:
  {
    "alg": "RS256",     // 署名アルゴリズム
    "typ": "JWT",
    "kid": "key-2024-01" // Key ID（鍵のローテーション用）
  }

  ペイロード（Base64URL）:
  {
    "sub": "user_123",        // Subject（ユーザーID）
    "name": "Taro",
    "role": "admin",
    "iat": 1704067200,        // Issued At（発行時刻）
    "exp": 1704070800,        // Expiration（有効期限）
    "nbf": 1704067200,        // Not Before（有効開始時刻）
    "iss": "api.example.com", // Issuer（発行者）
    "aud": "web.example.com", // Audience（対象者）
    "jti": "unique-id-123"   // JWT ID（一意識別子）
  }

  署名:
  HMACSHA256(
    base64UrlEncode(header) + "." + base64UrlEncode(payload),
    secret
  )

  重要: ペイロードは暗号化されていない（Base64デコードで読める）
  → 機密情報（パスワード等）を含めてはいけない
  → 署名は改ざん検知のため（暗号化ではない）
```

### 4.1 JWTの署名アルゴリズム

```
署名アルゴリズムの選択:

① HS256（HMAC-SHA256）:
  → 共通鍵（対称鍵）で署名・検証
  → 署名者と検証者が同じシークレットを共有
  → シンプルだが鍵の配布が課題
  → 単一サービス向け

② RS256（RSA-SHA256）:
  → 秘密鍵で署名、公開鍵で検証
  → 署名者と検証者で鍵が異なる
  → マイクロサービス向け（認証サーバーが署名、各サービスが検証）
  → 鍵サイズが大きい（2048bit以上）

③ ES256（ECDSA-SHA256）:
  → 楕円曲線暗号（P-256）
  → RSAより小さい鍵サイズで同等のセキュリティ
  → 署名サイズも小さい
  → 推奨

④ EdDSA（Ed25519）:
  → 最新の楕円曲線署名
  → 高速、安全
  → まだ一部ライブラリでサポート限定

比較:
  ┌──────────┬────────────┬────────────┬──────────┐
  │ アルゴリズム│ 鍵の種類   │ 署名サイズ  │ 推奨度   │
  ├──────────┼────────────┼────────────┼──────────┤
  │ HS256    │ 共通鍵     │ 32バイト    │ 単一サービス│
  │ RS256    │ 公開鍵/秘密鍵│ 256バイト  │ 広くサポート│
  │ ES256    │ 公開鍵/秘密鍵│ 64バイト   │ 推奨     │
  │ EdDSA    │ 公開鍵/秘密鍵│ 64バイト   │ 最新推奨 │
  └──────────┴────────────┴────────────┴──────────┘
```

### 4.2 Access Token と Refresh Token

```
トークンの種類:
  Access Token:
    → 短い有効期限（15分〜1時間）
    → APIアクセスに使用
    → ステートレス検証（署名検証のみ）
    → 漏洩時の影響を最小限に

  Refresh Token:
    → 長い有効期限（7日〜30日）
    → Access Token の再発行に使用
    → サーバー側で管理（無効化可能）
    → ローテーション推奨

フロー:
  1. ログイン → Access Token + Refresh Token を取得
  2. API呼び出し → Access Token をヘッダーに付与
  3. Access Token 期限切れ → 401 Unauthorized
  4. Refresh Token で新しい Access Token を取得
  5. Refresh Token 期限切れ → 再ログイン

Refresh Token Rotation:
  → Refresh Token使用時に新しいRefresh Tokenも発行
  → 古いRefresh Tokenは無効化
  → 漏洩検知: 無効化済みTokenが使用された → 全Token無効化

  1. POST /token/refresh { refresh_token: "rt_old" }
  2. レスポンス: { access_token: "at_new", refresh_token: "rt_new" }
  3. rt_old は無効化（DB/Redisから削除）
  4. もしrt_oldが再度使用 → 不正検知 → 全Token無効化
```

```typescript
// JWT実装例（TypeScript / jose ライブラリ）
import { SignJWT, jwtVerify, generateKeyPair } from 'jose';

// 鍵ペアの生成（ES256）
const { publicKey, privateKey } = await generateKeyPair('ES256');

// Access Token の生成
async function createAccessToken(userId: string, role: string): Promise<string> {
  return new SignJWT({
    sub: userId,
    role: role,
    type: 'access',
  })
    .setProtectedHeader({ alg: 'ES256', typ: 'JWT' })
    .setIssuedAt()
    .setExpirationTime('15m')
    .setIssuer('api.example.com')
    .setAudience('web.example.com')
    .setJti(crypto.randomUUID())
    .sign(privateKey);
}

// Refresh Token の生成
async function createRefreshToken(userId: string): Promise<string> {
  const token = new SignJWT({
    sub: userId,
    type: 'refresh',
  })
    .setProtectedHeader({ alg: 'ES256', typ: 'JWT' })
    .setIssuedAt()
    .setExpirationTime('7d')
    .setIssuer('api.example.com')
    .setJti(crypto.randomUUID())
    .sign(privateKey);

  // Refresh Token をDBに保存（無効化のため）
  // await db.refreshTokens.create({ token, userId, expiresAt: ... });
  return token;
}

// トークンの検証
async function verifyToken(token: string): Promise<any> {
  try {
    const { payload } = await jwtVerify(token, publicKey, {
      issuer: 'api.example.com',
      audience: 'web.example.com',
    });
    return payload;
  } catch (error) {
    if (error.code === 'ERR_JWT_EXPIRED') {
      throw new Error('Token expired');
    }
    throw new Error('Invalid token');
  }
}

// トークンリフレッシュ
async function refreshTokens(refreshToken: string) {
  const payload = await verifyToken(refreshToken);

  if (payload.type !== 'refresh') {
    throw new Error('Invalid token type');
  }

  // DBでRefresh Tokenの有効性を確認
  // const stored = await db.refreshTokens.findByToken(refreshToken);
  // if (!stored) throw new Error('Token revoked');

  // 古いRefresh Tokenを無効化
  // await db.refreshTokens.delete(refreshToken);

  // 新しいトークンペアを発行
  const accessToken = await createAccessToken(payload.sub, payload.role);
  const newRefreshToken = await createRefreshToken(payload.sub);

  return { accessToken, refreshToken: newRefreshToken };
}
```

### 4.3 JWTのセキュリティ注意事項

```
JWTのセキュリティ注意:

① alg: "none" 攻撃:
  → アルゴリズムを"none"に変更して署名なしのJWTを送信
  → 対策: サーバー側で許可するアルゴリズムを明示的に指定

② アルゴリズム混同攻撃:
  → RS256の公開鍵をHS256のシークレットとして使用
  → 対策: トークン内のalgヘッダーを信用しない

③ ペイロードへの機密情報格納:
  → Base64デコードで誰でも読める
  → 対策: パスワード、個人情報を含めない

④ 有効期限の設定:
  → 長すぎる有効期限はリスク
  → Access Token: 15分〜1時間
  → Refresh Token: 7日〜30日

⑤ トークンの保存場所:
  ┌─────────────────┬────────────┬────────────┬────────────┐
  │ 保存場所         │ XSS耐性   │ CSRF耐性   │ 推奨度     │
  ├─────────────────┼────────────┼────────────┼────────────┤
  │ LocalStorage    │ ✗ 脆弱    │ ✓ 安全    │ △ 非推奨  │
  │ SessionStorage  │ ✗ 脆弱    │ ✓ 安全    │ △ 非推奨  │
  │ HttpOnly Cookie │ ✓ 安全    │ ✗ 要対策  │ ○ 推奨   │
  │ メモリ          │ ✓ 安全    │ ✓ 安全    │ ○ 推奨   │
  └─────────────────┴────────────┴────────────┴────────────┘

  推奨パターン:
  → Access Token: メモリ（変数）に保持
  → Refresh Token: HttpOnly Cookie
  → CSRF対策: SameSite=Strict + CSRFトークン

⑥ トークン失効（ブラックリスト）:
  → JWTはステートレスなので本来失効できない
  → 対策: 短い有効期限 + Refresh Token での管理
  → または: Redis等にブラックリストを保持

⑦ JWE（JSON Web Encryption）:
  → ペイロードを暗号化したい場合
  → JWS（署名）+ JWE（暗号化）= 完全な保護
  → ただし複雑になるため、本当に必要か検討
```

---

## 5. OAuth 2.0

```
OAuth 2.0 = 認可のフレームワーク（認証ではない）
  → 第三者アプリにリソースへのアクセス権を委譲

登場人物:
  Resource Owner:  ユーザー
  Client:          アプリ（アクセスを要求する側）
  Authorization Server: 認可サーバー（Google, GitHub等）
  Resource Server: リソースサーバー（API）
```

### 5.1 Authorization Code Flow（推奨）

```
Authorization Code Flow:

  ユーザー     アプリ       認可サーバー    リソースサーバー
    │           │              │                │
    │──ログイン→│              │                │
    │           │──認可リクエスト→│              │
    │←───── ログイン画面 ────│                │
    │── 同意 ──→│              │                │
    │           │←── 認可コード──│              │
    │           │── コード + シークレット →│    │
    │           │←── Access Token ────│        │
    │           │── API呼び出し + Token ──────→│
    │           │←── リソースデータ ───────── │

詳細なリクエスト/レスポンス:

Step 1: 認可リクエスト（ブラウザリダイレクト）
  GET https://auth.example.com/authorize
    ?response_type=code
    &client_id=my-app-id
    &redirect_uri=https://myapp.com/callback
    &scope=openid profile email
    &state=random-csrf-token   ← CSRF対策
    &nonce=random-nonce         ← リプレイ攻撃対策

Step 2: 認可コードの受信（コールバック）
  GET https://myapp.com/callback
    ?code=AUTH_CODE_HERE
    &state=random-csrf-token

  → stateパラメータの一致を確認（CSRF対策）

Step 3: トークンの取得（サーバー間通信）
  POST https://auth.example.com/token
  Content-Type: application/x-www-form-urlencoded

  grant_type=authorization_code
  &code=AUTH_CODE_HERE
  &redirect_uri=https://myapp.com/callback
  &client_id=my-app-id
  &client_secret=my-app-secret

  レスポンス:
  {
    "access_token": "eyJ...",
    "token_type": "Bearer",
    "expires_in": 3600,
    "refresh_token": "dGhpcyBpcyBhIHJlZnJlc2g...",
    "scope": "openid profile email",
    "id_token": "eyJ..."  ← OIDC使用時
  }
```

### 5.2 Authorization Code + PKCE

```
PKCE（Proof Key for Code Exchange）:
  → SPAやモバイルアプリ向けの拡張
  → クライアントシークレットが不要
  → 認可コード横取り攻撃を防止

  フロー:
  1. クライアントがcode_verifier（ランダム文字列）を生成
  2. code_challenge = SHA256(code_verifier) をBase64URL
  3. 認可リクエストにcode_challengeを含める
  4. トークンリクエストにcode_verifierを含める
  5. サーバーがcode_verifier → SHA256 → code_challengeと照合

  Step 1: 認可リクエスト
  GET https://auth.example.com/authorize
    ?response_type=code
    &client_id=my-spa-id
    &redirect_uri=https://myapp.com/callback
    &scope=openid profile
    &state=random-state
    &code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM
    &code_challenge_method=S256

  Step 3: トークンリクエスト（client_secret不要）
  POST https://auth.example.com/token

  grant_type=authorization_code
  &code=AUTH_CODE
  &redirect_uri=https://myapp.com/callback
  &client_id=my-spa-id
  &code_verifier=dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk
```

```typescript
// PKCE実装例（TypeScript）
function generateCodeVerifier(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return base64UrlEncode(array);
}

async function generateCodeChallenge(verifier: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(verifier);
  const digest = await crypto.subtle.digest('SHA-256', data);
  return base64UrlEncode(new Uint8Array(digest));
}

function base64UrlEncode(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=/g, '');
}

// 使用例
async function startOAuthFlow() {
  const codeVerifier = generateCodeVerifier();
  const codeChallenge = await generateCodeChallenge(codeVerifier);

  // code_verifierをセッションに保存
  sessionStorage.setItem('code_verifier', codeVerifier);

  // 認可リクエスト
  const authUrl = new URL('https://auth.example.com/authorize');
  authUrl.searchParams.set('response_type', 'code');
  authUrl.searchParams.set('client_id', 'my-spa-id');
  authUrl.searchParams.set('redirect_uri', 'https://myapp.com/callback');
  authUrl.searchParams.set('scope', 'openid profile email');
  authUrl.searchParams.set('state', crypto.randomUUID());
  authUrl.searchParams.set('code_challenge', codeChallenge);
  authUrl.searchParams.set('code_challenge_method', 'S256');

  window.location.href = authUrl.toString();
}

// コールバック処理
async function handleCallback(code: string) {
  const codeVerifier = sessionStorage.getItem('code_verifier');

  const response = await fetch('https://auth.example.com/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code,
      redirect_uri: 'https://myapp.com/callback',
      client_id: 'my-spa-id',
      code_verifier: codeVerifier!,
    }),
  });

  const tokens = await response.json();
  return tokens;
}
```

### 5.3 その他のグラントタイプ

```
主要なグラントタイプ:
  ┌─────────────────────┬──────────────────────────────┐
  │ グラントタイプ       │ 用途                          │
  ├─────────────────────┼──────────────────────────────┤
  │ Authorization Code  │ サーバーサイドアプリ（推奨）  │
  │ Auth Code + PKCE    │ SPA/モバイル（推奨）          │
  │ Client Credentials  │ マシン間通信（サーバー間）    │
  │ Device Code         │ TV/IoT等の入力制限デバイス    │
  └─────────────────────┴──────────────────────────────┘

  廃止されたグラント:
  ✗ Implicit: セキュリティ上の問題で非推奨
    → Access Tokenがフラグメントに露出
    → Refresh Tokenが使えない
  ✗ Resource Owner Password: 非推奨
    → ユーザーのパスワードをアプリに直接渡す

Client Credentials Flow:
  → マシン間通信（バッチ処理、マイクロサービス間）
  → ユーザーの介在なし

  POST https://auth.example.com/token
  Content-Type: application/x-www-form-urlencoded
  Authorization: Basic <client_id:client_secret のBase64>

  grant_type=client_credentials
  &scope=api:read api:write

  レスポンス:
  {
    "access_token": "eyJ...",
    "token_type": "Bearer",
    "expires_in": 3600
  }
  → Refresh Tokenは発行されない

Device Authorization Flow:
  → TV、ゲーム機、CLIツール等
  → ユーザーが別デバイスで認証

  1. デバイス → 認可サーバー: POST /device/code
  2. レスポンス:
     {
       "device_code": "...",
       "user_code": "ABCD-EFGH",
       "verification_uri": "https://auth.example.com/device",
       "expires_in": 900,
       "interval": 5
     }
  3. デバイス画面: 「https://auth.example.com/device でコード ABCD-EFGH を入力」
  4. ユーザー: スマホでURLにアクセス → コードを入力 → ログイン → 承認
  5. デバイス: 5秒間隔でトークンをポーリング
     POST /token { grant_type: "urn:ietf:params:oauth:grant-type:device_code", device_code: "..." }
  6. 承認完了後: Access Token を受信
```

---

## 6. OpenID Connect（OIDC）

```
OIDC = OAuth 2.0 + 認証レイヤー
  → OAuth 2.0は認可のみ、OIDCは認証も提供

  OAuth 2.0: 「このアプリにGoogleドライブへのアクセスを許可」
  OIDC:      「このアプリにGoogleアカウントでログイン」

  Access Token: リソースへのアクセス権
  ID Token:     ユーザーの認証情報（JWT形式）
```

### 6.1 ID Token

```
ID Token の中身:
  {
    "iss": "https://accounts.google.com",       // 発行者
    "sub": "110169484474386276334",              // ユーザー固有ID
    "aud": "my-app-client-id",                   // 対象アプリ
    "email": "user@gmail.com",
    "email_verified": true,
    "name": "Taro Yamada",
    "picture": "https://lh3.googleusercontent.com/...",
    "given_name": "Taro",
    "family_name": "Yamada",
    "locale": "ja",
    "iat": 1704067200,                           // 発行時刻
    "exp": 1704070800,                           // 有効期限
    "nonce": "random-nonce",                     // リプレイ攻撃対策
    "at_hash": "HK6E_P6Dh8Y93mRNtsDB1Q"         // Access Tokenのハッシュ
  }

ID Tokenの検証:
  1. 署名の検証（公開鍵/JWKSで検証）
  2. iss（発行者）の確認
  3. aud（対象アプリ）の確認
  4. exp（有効期限）の確認
  5. nonce の一致確認
  6. at_hash の検証（optional）

UserInfoエンドポイント:
  → ID Tokenの情報が不足する場合に追加取得
  GET https://accounts.google.com/userinfo
  Authorization: Bearer <access_token>

  レスポンス:
  {
    "sub": "110169484474386276334",
    "name": "Taro Yamada",
    "email": "user@gmail.com",
    "picture": "https://..."
  }
```

### 6.2 OIDCディスカバリー

```
OpenID Connect Discovery:
  → 認可サーバーの設定情報を自動取得
  → /.well-known/openid-configuration

  GET https://accounts.google.com/.well-known/openid-configuration

  レスポンス:
  {
    "issuer": "https://accounts.google.com",
    "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
    "token_endpoint": "https://oauth2.googleapis.com/token",
    "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
    "revocation_endpoint": "https://oauth2.googleapis.com/revoke",
    "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    "supported_scopes": ["openid", "email", "profile"],
    "response_types_supported": ["code", "token", "id_token"],
    "subject_types_supported": ["public"],
    "id_token_signing_alg_values_supported": ["RS256"],
    "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"]
  }

JWKS（JSON Web Key Set）:
  → 公開鍵を配布するエンドポイント
  GET https://www.googleapis.com/oauth2/v3/certs

  {
    "keys": [
      {
        "kty": "RSA",
        "kid": "key-id-1",
        "use": "sig",
        "alg": "RS256",
        "n": "...",  // modulus
        "e": "AQAB"  // exponent
      }
    ]
  }

  → ID Token の kid ヘッダーと照合して正しい公開鍵を選択
  → 公開鍵のキャッシュと定期更新

主要なOIDCプロバイダー:
  Google, Microsoft, Apple, Auth0, Okta, Keycloak, AWS Cognito
```

---

## 7. パスキー（Passkeys）/ WebAuthn / FIDO2

```
パスキー = パスワード不要の認証
  → 公開鍵暗号ベース
  → フィッシング耐性
  → FIDO2 / WebAuthn 標準

仕組み:
  登録（Registration）:
  1. サーバー → チャレンジを送信
  2. デバイス → 鍵ペアを生成（公開鍵 + 秘密鍵）
  3. 秘密鍵はデバイスに安全に保存（Secure Enclave / TPM）
  4. 公開鍵をサーバーに送信
  5. ユーザーは生体認証（指紋/顔）またはPINで承認

  認証（Authentication）:
  1. サーバー → チャレンジを送信
  2. デバイス → 秘密鍵でチャレンジに署名
  3. ユーザーは生体認証で承認
  4. 署名をサーバーに送信
  5. サーバー → 公開鍵で署名を検証

  セキュリティ上の利点:
  ✓ パスワード不要（漏洩リスクなし）
  ✓ フィッシング耐性（オリジンにバインド）
  ✓ リプレイ攻撃耐性（チャレンジベース）
  ✓ サーバーに秘密鍵が保存されない

パスキーの同期:
  → iCloud Keychain / Google Password Manager で同期
  → デバイス間でパスキーを共有
  → バックアップと復元が可能

  従来のFIDO2:
  → デバイスバウンド（デバイス紛失 = アクセス不能）
  パスキー:
  → クラウド同期（利便性向上、セキュリティはやや低下）
```

```javascript
// WebAuthn登録（Registration）のフロントエンド実装
async function registerPasskey() {
  // サーバーからチャレンジを取得
  const options = await fetch('/api/webauthn/register/options', {
    method: 'POST',
  }).then(r => r.json());

  // ブラウザのWebAuthn APIを呼び出し
  const credential = await navigator.credentials.create({
    publicKey: {
      challenge: base64ToBuffer(options.challenge),
      rp: {
        name: 'My App',
        id: 'myapp.com',
      },
      user: {
        id: base64ToBuffer(options.userId),
        name: options.userName,
        displayName: options.userDisplayName,
      },
      pubKeyCredParams: [
        { alg: -7, type: 'public-key' },   // ES256
        { alg: -257, type: 'public-key' },  // RS256
      ],
      authenticatorSelection: {
        authenticatorAttachment: 'platform', // プラットフォーム認証器
        residentKey: 'required',             // パスキー必須
        userVerification: 'required',        // 生体認証必須
      },
      timeout: 60000,
    },
  });

  // サーバーに公開鍵を送信
  await fetch('/api/webauthn/register/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: credential.id,
      rawId: bufferToBase64(credential.rawId),
      response: {
        attestationObject: bufferToBase64(
          credential.response.attestationObject
        ),
        clientDataJSON: bufferToBase64(
          credential.response.clientDataJSON
        ),
      },
      type: credential.type,
    }),
  });
}

// WebAuthn認証（Authentication）のフロントエンド実装
async function authenticatePasskey() {
  const options = await fetch('/api/webauthn/authenticate/options', {
    method: 'POST',
  }).then(r => r.json());

  const assertion = await navigator.credentials.get({
    publicKey: {
      challenge: base64ToBuffer(options.challenge),
      rpId: 'myapp.com',
      allowCredentials: [], // 空配列 = パスキー一覧から選択
      userVerification: 'required',
      timeout: 60000,
    },
  });

  const result = await fetch('/api/webauthn/authenticate/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      id: assertion.id,
      rawId: bufferToBase64(assertion.rawId),
      response: {
        authenticatorData: bufferToBase64(
          assertion.response.authenticatorData
        ),
        clientDataJSON: bufferToBase64(
          assertion.response.clientDataJSON
        ),
        signature: bufferToBase64(assertion.response.signature),
      },
      type: assertion.type,
    }),
  }).then(r => r.json());

  return result; // { token: "...", user: { ... } }
}
```

---

## 8. 多要素認証（MFA）

```
多要素認証の実装パターン:

① TOTP（Time-based One-Time Password）:
  → Google Authenticator, Authy等
  → RFC 6238
  → 30秒ごとに変わる6桁コード
  → シークレットキーとタイムスタンプからHMACで生成

  セットアップ:
  1. サーバーがシークレットキーを生成
  2. QRコードでユーザーのアプリに登録
  3. ユーザーがアプリの6桁コードを入力して確認

  検証:
  → 現在の30秒ウィンドウ ± 1ウィンドウを許容
  → 同じコードの再利用を防止（リプレイ攻撃対策）

② SMS OTP:
  → SMSで6桁コードを送信
  → 実装は簡単だがセキュリティが低い
  → SIMスワッピング攻撃のリスク
  → SS7プロトコルの脆弱性
  → NIST SP 800-63B で「制限付き」に分類

③ セキュリティキー（FIDO U2F / FIDO2）:
  → YubiKey等のハードウェアトークン
  → 最も安全な2FA
  → フィッシング耐性
  → 企業での採用が増加

④ プッシュ通知:
  → スマホアプリに承認リクエストを送信
  → 「ログインを承認しますか？」
  → MFA疲労攻撃に注意
    → 対策: 数字マッチング（画面の数字を選択）

リカバリーコード:
  → MFAデバイス紛失時の救済手段
  → 8〜10個のワンタイムコード
  → 安全な場所に保管（パスワードマネージャー等）
  → ハッシュ化してDBに保存
```

```python
# TOTP実装例（Python / pyotp）
import pyotp
import qrcode
import io

class TOTPManager:
    def generate_secret(self) -> str:
        """新しいTOTPシークレットを生成"""
        return pyotp.random_base32()

    def get_provisioning_uri(
        self, secret: str, email: str, issuer: str = "MyApp"
    ) -> str:
        """QRコード用のURIを生成"""
        totp = pyotp.TOTP(secret)
        return totp.provisioning_uri(name=email, issuer_name=issuer)

    def generate_qr_code(self, uri: str) -> bytes:
        """QRコード画像を生成"""
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def verify_code(self, secret: str, code: str, window: int = 1) -> bool:
        """TOTPコードを検証（前後1ウィンドウを許容）"""
        totp = pyotp.TOTP(secret)
        return totp.verify(code, valid_window=window)

    def generate_recovery_codes(self, count: int = 10) -> list[str]:
        """リカバリーコードを生成"""
        import secrets
        codes = []
        for _ in range(count):
            code = secrets.token_hex(4)  # 8文字の16進数
            codes.append(f"{code[:4]}-{code[4:]}")
        return codes

# 使用例
totp = TOTPManager()

# セットアップ
secret = totp.generate_secret()  # "JBSWY3DPEHPK3PXP"
uri = totp.get_provisioning_uri(secret, "user@example.com")
qr_image = totp.generate_qr_code(uri)
recovery_codes = totp.generate_recovery_codes()

# 検証
is_valid = totp.verify_code(secret, "123456")
```

---

## 9. セッション管理のベストプラクティス

```
セッション管理のセキュリティ:

① セッションIDの生成:
  → 暗号学的に安全な乱数生成器を使用
  → 十分な長さ（128ビット以上）
  → 推測不可能

② Cookie属性:
  Set-Cookie: session_id=abc123;
    HttpOnly;         ← JavaScriptからアクセス不可
    Secure;           ← HTTPS接続時のみ送信
    SameSite=Lax;     ← クロスサイトリクエストを制限
    Path=/;           ← 適切なパス制限
    Max-Age=86400;    ← 有効期限（1日）
    Domain=.example.com;

③ セッション固定攻撃の対策:
  → ログイン成功時にセッションIDを再生成
  → 古いセッションIDを無効化

④ セッションの無効化:
  → ログアウト時にサーバー側でセッション削除
  → パスワード変更時に全セッション無効化
  → 管理者による強制ログアウト

⑤ セッションタイムアウト:
  → アイドルタイムアウト: 30分〜1時間
  → 絶対タイムアウト: 8時間〜24時間
  → 重要操作時の再認証

⑥ 並行セッション制御:
  → 同時ログイン数の制限
  → 新規ログイン時に古いセッションを無効化
  → アクティブセッション一覧の表示

⑦ セッションストレージ:
  → Redis: 高速、TTL対応、クラスタリング
  → PostgreSQL: 永続性、既存インフラ活用
  → メモリ: 開発用のみ（スケール不可）
```

---

## 10. 認証アーキテクチャパターン

```
① BFF（Backend For Frontend）パターン:
  → フロントエンド専用のバックエンドを配置
  → トークン管理をBFFで行う

  ブラウザ → BFF → APIサーバー
            ↕
        認可サーバー

  ブラウザ-BFF間: HttpOnly Cookie（セッション）
  BFF-API間: Access Token（Bearer）

  利点:
  → ブラウザにトークンが露出しない
  → XSS攻撃でトークン窃取不可
  → Refresh Token のセキュアな管理

② API Gatewayパターン:
  → 認証をGatewayに集約

  クライアント → API Gateway → マイクロサービス
                     ↕
                認可サーバー

  Gateway: トークン検証、レート制限
  マイクロサービス: 認証済みリクエストのみ受信

③ Token Exchange（RFC 8693）:
  → マイクロサービス間でトークンを変換
  → サービスAのトークン → サービスB用のトークンに交換
  → 最小権限の原則の適用

④ Sidecar / Service Meshパターン:
  → Istio/Envoy等でmTLS + JWT検証
  → アプリケーションコードから認証ロジックを分離

  Pod内:
  ┌──────────────────────┐
  │ Envoy Sidecar        │ ← mTLS終端、JWT検証
  │   ↕                  │
  │ アプリケーション       │ ← 認証ロジック不要
  └──────────────────────┘
```

---

## 11. パスワードのセキュリティ

```
パスワードのハッシュ化:

推奨アルゴリズム（2024年時点）:
  1. Argon2id（推奨）: メモリハードで最もセキュア
  2. bcrypt: 広く使われており安全
  3. scrypt: メモリハード
  ✗ MD5, SHA-1, SHA-256 単体は不可
  ✗ ソルトなしは不可

Argon2id パラメータ:
  memory: 64MB（最低19MiB推奨）
  iterations: 3
  parallelism: 4
  hash_length: 32バイト

bcrypt:
  cost factor: 12以上（2024年時点）
  → cost 12 ≒ 250ms（攻撃者のブルートフォースを遅延）

ソルト:
  → ユーザーごとにランダムなソルトを付加
  → レインボーテーブル攻撃を防止
  → bcrypt/Argon2は自動的にソルトを含む

ペッパー:
  → ソルトに加えてサーバー側の秘密値
  → DBが漏洩してもペッパーがなければ解読困難
  → 環境変数やHSMで管理

パスワードポリシー:
  推奨:
  ✓ 最低8文字（NIST推奨: 8〜64文字）
  ✓ 漏洩パスワードリスト（Have I Been Pwned API）でチェック
  ✓ パスフレーズの推奨

  非推奨（NIST SP 800-63B）:
  ✗ 複雑さの強制（大文字/数字/記号の必須化）
  ✗ 定期的なパスワード変更の強制
  ✗ パスワードヒントの使用
```

```python
# パスワードハッシュの実装例（Python / argon2-cffi）
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ph = PasswordHasher(
    time_cost=3,          # iterations
    memory_cost=65536,    # 64MB
    parallelism=4,
    hash_len=32,
    salt_len=16,
)

# パスワードハッシュの生成
def hash_password(password: str) -> str:
    return ph.hash(password)

# パスワードの検証
def verify_password(hash: str, password: str) -> bool:
    try:
        return ph.verify(hash, password)
    except VerifyMismatchError:
        return False

# パラメータ更新の確認（リハッシュ）
def needs_rehash(hash: str) -> bool:
    return ph.check_needs_rehash(hash)

# 使用例
hashed = hash_password("my-secure-password")
# $argon2id$v=19$m=65536,t=3,p=4$...

is_valid = verify_password(hashed, "my-secure-password")  # True
is_valid = verify_password(hashed, "wrong-password")       # False

# パラメータが古い場合はリハッシュ
if needs_rehash(hashed):
    new_hash = hash_password("my-secure-password")
    # DBを更新
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| Basic認証 | シンプルだが本番非推奨 |
| Bearer Token | ステートレスなAPI認証 |
| JWT | ヘッダー.ペイロード.署名、暗号化ではない |
| OAuth 2.0 | 認可フレームワーク、Auth Code + PKCE推奨 |
| OIDC | OAuth 2.0 + 認証（ID Token） |
| パスキー | パスワード不要、フィッシング耐性、WebAuthn |
| MFA | TOTP/セキュリティキー推奨、SMS非推奨 |
| パスワード | Argon2id/bcrypt、NIST SP 800-63B準拠 |
| BFF | フロントエンド向けトークン管理 |
| セッション | HttpOnly + Secure + SameSite Cookie |

---

## 次に読むべきガイド
→ [[02-common-attacks.md]] — ネットワーク攻撃

---

## 参考文献
1. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
2. RFC 6749. "The OAuth 2.0 Authorization Framework." IETF, 2012.
3. RFC 7636. "Proof Key for Code Exchange (PKCE)." IETF, 2015.
4. RFC 8693. "OAuth 2.0 Token Exchange." IETF, 2020.
5. OpenID Connect Core 1.0. OpenID Foundation, 2014.
6. W3C. "Web Authentication: An API for accessing Public Key Credentials Level 2." 2021.
7. FIDO Alliance. "FIDO2: Web Authentication (WebAuthn)." 2019.
8. NIST. "SP 800-63B: Digital Identity Guidelines - Authentication and Lifecycle Management." 2017.
9. OWASP. "Authentication Cheat Sheet." 2024.
10. OWASP. "Session Management Cheat Sheet." 2024.
