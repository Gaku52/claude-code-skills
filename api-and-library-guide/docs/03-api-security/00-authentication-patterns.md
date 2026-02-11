# 認証パターン

> API認証はセキュリティの要。API Key、OAuth 2.0、JWT、mTLS、各認証方式の仕組み・セキュリティ特性・選定基準を体系的に理解し、要件に応じた適切な認証アーキテクチャを設計する。

## この章で学ぶこと

- [ ] 主要な認証方式の仕組みと比較を理解する
- [ ] OAuth 2.0のフローとセキュリティを把握する
- [ ] JWTの構造と安全な運用方法を学ぶ

---

## 1. 認証方式の比較

```
               API Key     Bearer Token   OAuth 2.0     mTLS
─────────────────────────────────────────────────────────
用途          サーバー間   モバイル/SPA   サードパーティ  サーバー間
セキュリティ   低〜中       中             高             最高
実装コスト     低           中             高             高
ユーザー認証   不可         可能           可能           不可
スコープ制御   限定的       可能           詳細           なし
有効期限       長期/無期限  短期           短期+更新      証明書有効期限
適用例         内部API      自社アプリ     外部連携        金融/医療
```

---

## 2. API Key

```
API Key の仕組み:
  → サーバーが発行する文字列トークン
  → リクエストヘッダーに含めて認証

  Authorization: Bearer sk_live_abc123def456
  または
  X-API-Key: sk_live_abc123def456

Key の命名規則（Stripe方式）:
  sk_live_xxx  → 本番シークレットキー（サーバーサイドのみ）
  sk_test_xxx  → テストシークレットキー
  pk_live_xxx  → 本番公開キー（クライアントサイドOK）
  pk_test_xxx  → テスト公開キー

セキュリティ:
  ✓ HTTPSでのみ送信
  ✓ サーバーサイドでのみ使用（クライアントに露出させない）
  ✓ 環境変数で管理（コードにハードコードしない）
  ✓ キーのローテーション機能を提供
  ✓ キーごとにスコープ/権限を設定
  ✗ ブラウザ/モバイルアプリに埋め込まない
```

```javascript
// サーバー側: API Key の検証
async function authenticateApiKey(req, res, next) {
  const apiKey = req.headers['authorization']?.replace('Bearer ', '')
                 || req.headers['x-api-key'];

  if (!apiKey) {
    return res.status(401).json({
      type: 'https://api.example.com/errors/unauthorized',
      title: 'Authentication Required',
      status: 401,
      detail: 'API key is missing. Include it in the Authorization header.',
    });
  }

  // ハッシュで検索（平文保存しない）
  const hashedKey = crypto.createHash('sha256').update(apiKey).digest('hex');
  const keyRecord = await db.apiKeys.findOne({ hash: hashedKey, revokedAt: null });

  if (!keyRecord) {
    return res.status(401).json({
      type: 'https://api.example.com/errors/invalid-api-key',
      title: 'Invalid API Key',
      status: 401,
      detail: 'The provided API key is invalid or has been revoked.',
    });
  }

  req.apiKey = keyRecord;
  req.account = await db.accounts.findOne({ id: keyRecord.accountId });
  next();
}
```

---

## 3. OAuth 2.0

```
OAuth 2.0 のフロー:

① Authorization Code Flow（推奨: Webアプリ）:

  ユーザー → アプリ → 認可サーバー → ユーザーに同意画面
  ユーザー → 同意 → 認可サーバー → アプリにcode返却
  アプリ → 認可サーバーにcode + client_secret送信
  認可サーバー → アプリにaccess_token返却

  1. 認可リクエスト:
     GET https://auth.example.com/authorize?
       response_type=code&
       client_id=client_123&
       redirect_uri=https://app.example.com/callback&
       scope=users:read+orders:read&
       state=random_state_value

  2. コールバック:
     GET https://app.example.com/callback?
       code=auth_code_xxx&
       state=random_state_value

  3. トークン交換:
     POST https://auth.example.com/oauth/token
     {
       "grant_type": "authorization_code",
       "code": "auth_code_xxx",
       "redirect_uri": "https://app.example.com/callback",
       "client_id": "client_123",
       "client_secret": "secret_456"
     }

  4. レスポンス:
     {
       "access_token": "eyJhbG...",
       "token_type": "Bearer",
       "expires_in": 3600,
       "refresh_token": "rt_abc...",
       "scope": "users:read orders:read"
     }

② Authorization Code + PKCE（推奨: SPA/モバイル）:
  → client_secret を使わない（公開クライアント向け）
  → code_verifier / code_challenge で保護

  手順:
  1. code_verifier をランダム生成（43-128文字）
  2. code_challenge = BASE64URL(SHA256(code_verifier))
  3. 認可リクエストに code_challenge を含める
  4. トークン交換に code_verifier を含める
  → 認可コードを傍受されても code_verifier がないとトークン取得不可

③ Client Credentials Flow（サーバー間通信）:
  POST https://auth.example.com/oauth/token
  {
    "grant_type": "client_credentials",
    "client_id": "client_123",
    "client_secret": "secret_456",
    "scope": "admin:read"
  }
```

---

## 4. JWT（JSON Web Token）

```
JWT の構造:
  header.payload.signature

  Header:
  {
    "alg": "RS256",
    "typ": "JWT",
    "kid": "key_123"     ← 署名鍵のID
  }

  Payload（Claims）:
  {
    "iss": "https://auth.example.com",    ← 発行者
    "sub": "user_123",                     ← 主体（ユーザーID）
    "aud": "https://api.example.com",      ← 対象者
    "exp": 1700000000,                     ← 有効期限
    "iat": 1699996400,                     ← 発行時刻
    "nbf": 1699996400,                     ← 有効開始時刻
    "jti": "unique_token_id",              ← トークンID
    "scope": "users:read orders:read",     ← スコープ
    "role": "admin"                        ← カスタムクレーム
  }

  Signature:
  RS256(base64url(header) + "." + base64url(payload), privateKey)

アルゴリズムの選択:
  HS256: HMAC + SHA-256（共有鍵）
  → シンプルだがサーバー間で鍵を共有する必要あり

  RS256: RSA + SHA-256（公開鍵/秘密鍵）
  → 秘密鍵で署名、公開鍵で検証
  → 推奨: マイクロサービス環境

  ES256: ECDSA + SHA-256（楕円曲線暗号）
  → 短い鍵でRSAと同等のセキュリティ
  → モバイル/IoT向け
```

```javascript
// JWT の検証（jose ライブラリ）
import { jwtVerify, createRemoteJWKSet } from 'jose';

// JWKS（JSON Web Key Set）から公開鍵を取得
const JWKS = createRemoteJWKSet(
  new URL('https://auth.example.com/.well-known/jwks.json')
);

async function verifyToken(token) {
  const { payload } = await jwtVerify(token, JWKS, {
    issuer: 'https://auth.example.com',
    audience: 'https://api.example.com',
  });
  return payload;
}

// ミドルウェア
async function authMiddleware(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'Token required' });

  try {
    req.user = await verifyToken(token);
    next();
  } catch (error) {
    res.status(401).json({ error: 'Invalid token' });
  }
}
```

---

## 5. Access Token + Refresh Token

```
トークン運用:

  Access Token:
  → 短い有効期限（15分〜1時間）
  → APIアクセスに使用
  → ステートレス（JWT）

  Refresh Token:
  → 長い有効期限（7日〜90日）
  → Access Token の更新に使用
  → サーバーに保存（ステートフル）
  → 1回使用で無効化（Rotation）

  フロー:
  1. ログイン → Access Token + Refresh Token 発行
  2. APIリクエスト → Access Token を使用
  3. Access Token 期限切れ → Refresh Token で更新
  4. 新しい Access Token + 新しい Refresh Token を発行
  5. 古い Refresh Token を無効化

  Refresh Token Rotation:
  → Refresh Token 使用時に新しいペアを発行
  → 古い Refresh Token は即座に無効化
  → 盗まれた Refresh Token が使われると検知可能
    （既に無効化されたトークンの使用 → 全トークンを無効化）

保存場所:
  Web（SPA）:
    Access Token  → メモリ（変数）
    Refresh Token → HttpOnly Cookie（Secure, SameSite=Strict）
    ✗ localStorage に保存しない（XSSで盗まれる）

  モバイル:
    Access Token  → メモリ
    Refresh Token → Keychain（iOS）/ Keystore（Android）
```

---

## 6. スコープ設計

```
スコープの設計原則:
  → リソース:操作 の形式
  → 最小権限の原則

  例:
  users:read        ユーザー情報の読み取り
  users:write       ユーザー情報の作成・更新
  users:delete      ユーザーの削除
  orders:read       注文情報の読み取り
  orders:write      注文の作成・更新
  admin:all         管理者権限（全操作）

スコープチェックの実装:
  function requireScope(...scopes) {
    return (req, res, next) => {
      const tokenScopes = req.user.scope.split(' ');
      const hasScope = scopes.every(s => tokenScopes.includes(s));

      if (!hasScope) {
        return res.status(403).json({
          error: 'Insufficient scope',
          required: scopes,
          granted: tokenScopes,
        });
      }
      next();
    };
  }

  app.get('/api/v1/users', requireScope('users:read'), listUsers);
  app.post('/api/v1/users', requireScope('users:write'), createUser);
  app.delete('/api/v1/users/:id', requireScope('users:delete'), deleteUser);
```

---

## まとめ

| 方式 | 用途 | セキュリティ |
|------|------|------------|
| API Key | サーバー間、内部API | 中（HTTPS必須） |
| OAuth 2.0 + PKCE | SPA、モバイル | 高 |
| OAuth 2.0 Client Credentials | サーバー間 | 高 |
| JWT | ステートレス認証 | 高（RS256推奨） |
| mTLS | 金融、医療 | 最高 |

---

## 次に読むべきガイド
→ [[01-rate-limiting.md]] — レート制限

---

## 参考文献
1. RFC 6749. "The OAuth 2.0 Authorization Framework." IETF, 2012.
2. RFC 7636. "Proof Key for Code Exchange (PKCE)." IETF, 2015.
3. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
4. Auth0. "Authentication Best Practices." auth0.com, 2024.
