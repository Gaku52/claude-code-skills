# 認証パターン

> API認証はセキュリティの要。Basic認証、API Key、Bearer Token、OAuth 2.0、JWT、PKCEなど、各認証方式の仕組み・セキュリティ特性・選定基準を体系的に理解し、要件に応じた適切な認証アーキテクチャを設計する。

## この章で学ぶこと

- [ ] 主要な認証方式（Basic認証、API Key、Bearer Token、OAuth 2.0、JWT、PKCE）の仕組みと比較を理解する
- [ ] OAuth 2.0の各フローとセキュリティ上の考慮点を把握する
- [ ] JWTの内部構造と安全な運用方法を学ぶ
- [ ] PKCEがSPA/モバイルアプリで必須とされる理由を理解する
- [ ] 認証パターンごとのアンチパターンとエッジケースを把握する
- [ ] 要件に応じた認証方式の選定ができるようになる

---

## 1. 認証と認可の基礎概念

認証（Authentication）と認可（Authorization）は混同されやすいが、明確に異なる概念である。

```
認証と認可の違い:

  認証（Authentication / AuthN）:
  ┌─────────────────────────────────────────────┐
  │  「あなたは誰ですか？」                      │
  │  → ユーザーやシステムの身元を確認するプロセス │
  │  → 結果: Identity（アイデンティティ）        │
  │  例: パスワード検証、証明書検証、生体認証     │
  └─────────────────────────────────────────────┘

  認可（Authorization / AuthZ）:
  ┌─────────────────────────────────────────────┐
  │  「あなたは何ができますか？」                 │
  │  → 認証済みユーザーの権限を判定するプロセス   │
  │  → 結果: Permission（許可・不許可）           │
  │  例: ロールベースアクセス制御、スコープ検証   │
  └─────────────────────────────────────────────┘

  処理の順序:

  クライアント ──リクエスト──→ [認証] ──→ [認可] ──→ リソース
                                │           │
                                │           └─ 403 Forbidden
                                └─ 401 Unauthorized
```

API設計において、認証と認可を分離して設計することは保守性と拡張性の面で重要である。認証はリクエスト元の身元確認に特化し、認可はリソースへのアクセス可否の判定に特化する。この分離により、認証方式の変更が認可ロジックに影響を与えず、その逆もまた然りとなる。

### 1.1 認証方式の全体分類

API認証方式は大きく以下のカテゴリに分類できる。

```
API認証方式の分類体系:

  ┌─────────────────────────────────────────────────────────────┐
  │                     API認証方式                              │
  ├─────────────┬──────────────┬──────────────┬────────────────┤
  │ 知識ベース   │ トークンベース │ 証明書ベース │ 委譲型          │
  │             │              │              │                │
  │ ・Basic認証  │ ・API Key    │ ・mTLS       │ ・OAuth 2.0    │
  │ ・Digest認証 │ ・Bearer     │ ・クライアント│ ・OpenID       │
  │             │   Token      │   証明書      │   Connect     │
  │             │ ・JWT        │              │ ・SAML         │
  │             │ ・HMAC署名   │              │                │
  ├─────────────┴──────────────┴──────────────┴────────────────┤
  │ セキュリティ強度:  低 ──────────────────────────────→ 高     │
  │ 実装複雑度:        低 ──────────────────────────────→ 高     │
  └─────────────────────────────────────────────────────────────┘
```

---

## 2. 認証方式の詳細比較

### 2.1 総合比較表

以下の表は、主要な認証方式を複数の評価軸で比較したものである。

```
                Basic認証  API Key    Bearer Token  OAuth 2.0   JWT       mTLS
────────────────────────────────────────────────────────────────────────────────
用途            内部/開発   サーバー間  モバイル/SPA  サードパーティ ステートレス サーバー間
セキュリティ     低         低〜中      中           高           高         最高
実装コスト       最低       低          中           高           中〜高     高
ユーザー認証     可能       不可        可能         可能         可能       不可
スコープ制御     不可       限定的      可能         詳細         可能       なし
有効期限管理     なし       長期/無期限  短期         短期+更新    短期       証明書期限
ステートレス     いいえ     はい        場合による    場合による    はい       はい
リプレイ攻撃耐性  低        低          中           高           中〜高     高
適用例          開発環境    内部API     自社アプリ    外部連携      マイクロSVC 金融/医療
────────────────────────────────────────────────────────────────────────────────
```

### 2.2 セキュリティ特性の詳細比較

```
セキュリティ特性比較:

                        Basic認証  API Key  OAuth 2.0  JWT     mTLS
─────────────────────────────────────────────────────────────────────
認証情報の漏洩リスク      高        中       低         低      最低
中間者攻撃への耐性        低*       低*      高         中      最高
リプレイ攻撃への耐性      低        低       高**       中      高
CSRF攻撃への耐性         低        高       高         高      高
XSS経由の漏洩リスク      中        高       低***      中      なし
認証情報の取り消し容易性   困難      容易     容易       困難****  N/A
多要素認証との統合        困難      不可     容易       不可     可能
─────────────────────────────────────────────────────────────────────

* HTTPS使用時は中〜高に向上
** state/nonceパラメータ使用時
*** Authorization Code Flowの場合
**** JWTは有効期限まで無効化できない（ブラックリスト方式を除く）
```

---

## 3. Basic認証

### 3.1 仕組み

Basic認証はHTTP標準（RFC 7617）で定義された最もシンプルな認証方式である。ユーザー名とパスワードをBase64エンコードしてリクエストヘッダーに含める。

```
Basic認証のフロー:

  クライアント                          サーバー
       │                                  │
       │  GET /api/resource                │
       │ ────────────────────────────────→ │
       │                                  │
       │  401 Unauthorized                 │
       │  WWW-Authenticate: Basic realm="API" │
       │ ←──────────────────────────────── │
       │                                  │
       │  GET /api/resource                │
       │  Authorization: Basic dXNlcjpwYXNz │
       │ ────────────────────────────────→ │
       │                                  │
       │     Base64デコード                │
       │     "user:pass" を取得            │
       │     認証情報の検証                │
       │                                  │
       │  200 OK                           │
       │  { "data": "..." }               │
       │ ←──────────────────────────────── │

  エンコード方式:
    Authorization: Basic BASE64(username:password)
    例: user:pass → dXNlcjpwYXNz
```

### 3.2 実装例

```javascript
// サーバー側: Basic認証ミドルウェア（Express.js）
function basicAuthMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Basic ')) {
    res.setHeader('WWW-Authenticate', 'Basic realm="API"');
    return res.status(401).json({
      type: 'https://api.example.com/errors/unauthorized',
      title: 'Authentication Required',
      status: 401,
      detail: 'Basic authentication credentials are required.',
    });
  }

  // Base64デコード
  const base64Credentials = authHeader.substring(6);
  const credentials = Buffer.from(base64Credentials, 'base64').toString('utf-8');
  const [username, password] = credentials.split(':');

  // タイミング攻撃を防ぐための定数時間比較
  const expectedUsername = process.env.API_USERNAME;
  const expectedPassword = process.env.API_PASSWORD;

  const usernameMatch = crypto.timingSafeEqual(
    Buffer.from(username.padEnd(256)),
    Buffer.from(expectedUsername.padEnd(256))
  );
  const passwordMatch = crypto.timingSafeEqual(
    Buffer.from(password.padEnd(256)),
    Buffer.from(expectedPassword.padEnd(256))
  );

  if (!usernameMatch || !passwordMatch) {
    return res.status(401).json({
      type: 'https://api.example.com/errors/invalid-credentials',
      title: 'Invalid Credentials',
      status: 401,
      detail: 'The provided username or password is incorrect.',
    });
  }

  req.authenticatedUser = username;
  next();
}

// 使用例
app.get('/api/v1/health', basicAuthMiddleware, (req, res) => {
  res.json({ status: 'ok', authenticatedAs: req.authenticatedUser });
});
```

```python
# Python（Flask）でのBasic認証実装
import hmac
import base64
from functools import wraps
from flask import Flask, request, jsonify

app = Flask(__name__)

def require_basic_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization

        if not auth:
            return jsonify({
                'error': 'Authentication required',
                'detail': 'Basic authentication credentials are required.'
            }), 401, {'WWW-Authenticate': 'Basic realm="API"'}

        # タイミング攻撃を防ぐための定数時間比較
        expected_user = app.config['API_USERNAME']
        expected_pass = app.config['API_PASSWORD']

        user_valid = hmac.compare_digest(auth.username, expected_user)
        pass_valid = hmac.compare_digest(auth.password, expected_pass)

        if not (user_valid and pass_valid):
            return jsonify({
                'error': 'Invalid credentials',
                'detail': 'The provided username or password is incorrect.'
            }), 401

        request.authenticated_user = auth.username
        return f(*args, **kwargs)
    return decorated

@app.route('/api/v1/health')
@require_basic_auth
def health_check():
    return jsonify({'status': 'ok', 'user': request.authenticated_user})
```

### 3.3 Basic認証の注意点

Basic認証にはいくつかの重大な制約がある。

1. **Base64はエンコードであり暗号化ではない**: 誰でもデコードできるため、HTTPS無しでは認証情報が平文で流れるのと同等
2. **リクエスト毎に認証情報を送信**: 漏洩リスクが高い
3. **ログアウト機構がない**: ブラウザがキャッシュするため、セッション終了が困難
4. **レート制限との組み合わせが必須**: ブルートフォース攻撃への対策が別途必要

Basic認証は開発環境やCI/CDパイプラインでの一時的な認証、あるいは内部APIの簡易認証に限定して使用すべきである。

---

## 4. API Key

### 4.1 仕組みと設計

API Keyはサーバーが発行する文字列トークンで、クライアントを識別するために使用する。ユーザー認証ではなくアプリケーション認証に適している。

```
API Key の仕組み:

  発行フロー:
  ┌──────────┐    キー発行依頼     ┌──────────────┐
  │          │ ──────────────────→ │              │
  │ 開発者   │                     │  管理コンソール │
  │          │ ←────────────────── │              │
  └──────────┘    sk_live_abc123   └──────────────┘
                                         │
                                    ハッシュ化して保存
                                         │
                                    ┌──────────────┐
                                    │  データベース  │
                                    │  hash: a1b2c3 │
                                    │  scope: read  │
                                    │  rate: 1000/h │
                                    └──────────────┘

  認証フロー:
  ┌──────────┐    Authorization: Bearer sk_live_abc123    ┌──────────┐
  │          │ ────────────────────────────────────────→  │          │
  │ クライアント │                                          │  API     │
  │          │ ←────────────────────────────────────────  │ サーバー  │
  └──────────┘           200 OK / 401 Unauthorized       └──────────┘
                                                               │
                                                          SHA-256(key)
                                                          DB照合
                                                          スコープ検証
                                                          レート制限検証

  ヘッダーの送信方法（主要パターン）:
    パターン1: Authorization: Bearer sk_live_abc123
    パターン2: X-API-Key: sk_live_abc123
    パターン3: ?api_key=sk_live_abc123（非推奨: URLに残る）

  Key の命名規則（Stripe方式）:
    sk_live_xxx  → 本番シークレットキー（サーバーサイドのみ）
    sk_test_xxx  → テストシークレットキー
    pk_live_xxx  → 本番公開キー（クライアントサイドOK）
    pk_test_xxx  → テスト公開キー

  セキュリティ要件:
    [必須] HTTPSでのみ送信
    [必須] サーバーサイドでのみ使用（クライアントに露出させない）
    [必須] 環境変数で管理（コードにハードコードしない）
    [推奨] キーのローテーション機能を提供
    [推奨] キーごとにスコープ/権限を設定
    [禁止] ブラウザ/モバイルアプリに埋め込まない
    [禁止] URLクエリパラメータでの送信（アクセスログに残る）
```

### 4.2 実装例

```javascript
// サーバー側: API Key の検証（Express.js）
import crypto from 'crypto';

async function authenticateApiKey(req, res, next) {
  // 複数のヘッダー形式に対応
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

  // キーのフォーマット検証（プレフィックスチェック）
  if (!/^(sk|pk)_(live|test)_[a-zA-Z0-9]{24,}$/.test(apiKey)) {
    return res.status(401).json({
      type: 'https://api.example.com/errors/invalid-api-key-format',
      title: 'Invalid API Key Format',
      status: 401,
      detail: 'The API key format is invalid.',
    });
  }

  // ハッシュで検索（平文保存しない）
  const hashedKey = crypto.createHash('sha256').update(apiKey).digest('hex');
  const keyRecord = await db.apiKeys.findOne({
    hash: hashedKey,
    revokedAt: null,
  });

  if (!keyRecord) {
    return res.status(401).json({
      type: 'https://api.example.com/errors/invalid-api-key',
      title: 'Invalid API Key',
      status: 401,
      detail: 'The provided API key is invalid or has been revoked.',
    });
  }

  // テストキーでの本番アクセスを防止
  if (apiKey.includes('_test_') && process.env.NODE_ENV === 'production') {
    return res.status(403).json({
      type: 'https://api.example.com/errors/test-key-in-production',
      title: 'Test Key Not Allowed',
      status: 403,
      detail: 'Test API keys cannot be used in the production environment.',
    });
  }

  // 最終使用日時の更新
  await db.apiKeys.updateOne(
    { hash: hashedKey },
    { $set: { lastUsedAt: new Date() } }
  );

  req.apiKey = keyRecord;
  req.account = await db.accounts.findOne({ id: keyRecord.accountId });
  next();
}

// API Keyの発行
async function issueApiKey(accountId, options = {}) {
  const prefix = options.isPublic ? 'pk' : 'sk';
  const env = options.isTest ? 'test' : 'live';

  // 暗号学的に安全なランダム文字列を生成
  const randomPart = crypto.randomBytes(32).toString('base64url');
  const apiKey = `${prefix}_${env}_${randomPart}`;

  // ハッシュ化して保存（平文は保存しない）
  const hashedKey = crypto.createHash('sha256').update(apiKey).digest('hex');

  await db.apiKeys.insertOne({
    hash: hashedKey,
    prefix: `${prefix}_${env}_${randomPart.substring(0, 4)}`,
    accountId,
    scopes: options.scopes || ['read'],
    rateLimit: options.rateLimit || 1000,
    createdAt: new Date(),
    expiresAt: options.expiresAt || null,
    revokedAt: null,
    lastUsedAt: null,
  });

  // 平文のキーは発行時のみ返却（以後は取得不可）
  return {
    key: apiKey,
    prefix: `${prefix}_${env}_${randomPart.substring(0, 4)}...`,
    scopes: options.scopes || ['read'],
    expiresAt: options.expiresAt || null,
  };
}
```

### 4.3 API Keyのローテーション

安全なAPI Key運用にはローテーション（定期的な更新）が不可欠である。

```javascript
// API Keyのローテーション実装
async function rotateApiKey(accountId, oldKeyPrefix) {
  // 旧キーを検索
  const oldKeyRecord = await db.apiKeys.findOne({
    accountId,
    prefix: { $regex: `^${oldKeyPrefix}` },
    revokedAt: null,
  });

  if (!oldKeyRecord) {
    throw new Error('Active API key not found');
  }

  // 新しいキーを発行
  const newKey = await issueApiKey(accountId, {
    scopes: oldKeyRecord.scopes,
    rateLimit: oldKeyRecord.rateLimit,
  });

  // 旧キーにグレースピリオドを設定（24時間後に無効化）
  const gracePeriod = new Date(Date.now() + 24 * 60 * 60 * 1000);
  await db.apiKeys.updateOne(
    { _id: oldKeyRecord._id },
    {
      $set: {
        deprecatedAt: new Date(),
        revokedAt: gracePeriod,
      },
    }
  );

  return {
    newKey: newKey.key,
    oldKeyRevokedAt: gracePeriod,
    message: 'Old key will remain valid for 24 hours.',
  };
}
```

---

## 5. Bearer Token

### 5.1 仕組み

Bearer Token（RFC 6750）は「トークンの持参人（bearer）に対してアクセスを許可する」方式である。トークン自体が認証情報として機能するため、トークンの保護が極めて重要となる。

```
Bearer Tokenのフロー:

  ┌──────────┐                    ┌──────────────┐                ┌──────────┐
  │          │  1. 認証リクエスト  │              │                │          │
  │          │ ─────────────────→ │              │                │          │
  │          │                    │  認証サーバー  │                │          │
  │ クライアント │  2. Bearer Token  │              │                │ リソース  │
  │          │ ←───────────────── │              │                │ サーバー  │
  │          │                    └──────────────┘                │          │
  │          │                                                    │          │
  │          │  3. Authorization: Bearer <token>                  │          │
  │          │ ─────────────────────────────────────────────────→ │          │
  │          │                                                    │          │
  │          │  4. リソースレスポンス                                │          │
  │          │ ←───────────────────────────────────────────────── │          │
  └──────────┘                                                    └──────────┘

  Bearer Tokenの特徴:
    ・トークンの種類を問わない（JWT、ランダム文字列、など）
    ・トークンを持っていれば誰でもアクセス可能（= 漏洩に注意）
    ・HTTPSが必須（平文通信では傍受される）
    ・Authorization ヘッダーでの送信が標準
```

### 5.2 Opaque Token vs JWT

Bearer Tokenの実体は大きく2種類に分かれる。

```
Opaque Token vs JWT:

  Opaque Token（不透明トークン）:
  ┌─────────────────────────────────────────────┐
  │  例: "at_x7k2m9p3q8r1"                      │
  │                                              │
  │  ・ランダム文字列（意味を持たない）            │
  │  ・検証にはDBやキャッシュへの問い合わせが必要   │
  │  ・即座に無効化可能                           │
  │  ・トークンからは情報を読み取れない            │
  │  ・サーバー側にステート（状態）が必要           │
  └─────────────────────────────────────────────┘

  JWT（自己完結型トークン）:
  ┌─────────────────────────────────────────────┐
  │  例: "eyJhbGciOiJSUzI1NiIs..."               │
  │                                              │
  │  ・署名付きJSONペイロード                     │
  │  ・ローカルで検証可能（公開鍵があれば）        │
  │  ・有効期限まで無効化が困難                    │
  │  ・ペイロードにクレーム（情報）を含められる    │
  │  ・ステートレス（DB問い合わせ不要）            │
  └─────────────────────────────────────────────┘

  使い分け:
    Opaque Token → 即座にトークン無効化が必要な場合
    JWT         → マイクロサービス間でのステートレス認証
```

---

## 6. OAuth 2.0

### 6.1 概要と設計思想

OAuth 2.0（RFC 6749）は認可の委譲を目的としたフレームワークである。「ユーザーの代わりに」サードパーティアプリケーションがリソースにアクセスすることを可能にする。

重要な点として、OAuth 2.0は「認可」のプロトコルであり、「認証」のプロトコルではない。認証を行うためにはOpenID Connectを上層に追加する必要がある。

### 6.2 登場人物（ロール）

```
OAuth 2.0 の4つのロール:

  ┌────────────────────────────────────────────────────────┐
  │                                                        │
  │  Resource Owner（リソースオーナー）                      │
  │  → リソースの所有者。通常はエンドユーザー               │
  │  例: Googleアカウントのユーザー                         │
  │                                                        │
  │  Client（クライアント）                                 │
  │  → リソースにアクセスしたいアプリケーション              │
  │  例: Googleカレンダーと連携するタスク管理アプリ          │
  │                                                        │
  │  Authorization Server（認可サーバー）                   │
  │  → トークンを発行するサーバー                           │
  │  例: Google OAuth Server                               │
  │                                                        │
  │  Resource Server（リソースサーバー）                     │
  │  → 保護されたリソースを提供するサーバー                 │
  │  例: Google Calendar API                               │
  │                                                        │
  └────────────────────────────────────────────────────────┘
```

### 6.3 Authorization Code Flow（推奨: Webアプリ）

最もセキュアで推奨されるフローである。

```
Authorization Code Flow:

  Resource    Client         Authorization     Resource
  Owner       (Webアプリ)     Server            Server
    │           │               │                 │
    │  1. 「Googleでログイン」をクリック           │
    │ ────────→ │               │                 │
    │           │               │                 │
    │           │  2. 認可リクエスト（リダイレクト） │
    │ ←──────── │               │                 │
    │           │               │                 │
    │  3. 認可サーバーにリダイレクト                 │
    │ ──────────────────────── → │                 │
    │           │               │                 │
    │  4. ログイン画面/同意画面   │                 │
    │ ← ─────────────────────── │                 │
    │           │               │                 │
    │  5. ユーザーが同意          │                 │
    │ ──────────────────────── → │                 │
    │           │               │                 │
    │  6. 認可コード付きリダイレクト                 │
    │ ←──────────────────────── │                 │
    │ ────────→ │               │                 │
    │           │               │                 │
    │           │  7. 認可コード + client_secret    │
    │           │ ────────────→ │                 │
    │           │               │                 │
    │           │  8. access_token + refresh_token │
    │           │ ←──────────── │                 │
    │           │               │                 │
    │           │  9. APIリクエスト（Bearer token）  │
    │           │ ──────────────────────────────→  │
    │           │               │                 │
    │           │  10. リソースレスポンス            │
    │           │ ←──────────────────────────────  │
    │           │               │                 │

  ポイント:
  ・認可コードはフロントチャネル（ブラウザ）経由で渡される
  ・トークン交換はバックチャネル（サーバー間）で行われる
  ・client_secretはサーバー側に安全に保管される
```

```javascript
// Authorization Code Flow の実装例（Express.js）
import express from 'express';
import crypto from 'crypto';

const app = express();

const OAUTH_CONFIG = {
  clientId: process.env.OAUTH_CLIENT_ID,
  clientSecret: process.env.OAUTH_CLIENT_SECRET,
  authorizationEndpoint: 'https://auth.example.com/authorize',
  tokenEndpoint: 'https://auth.example.com/oauth/token',
  redirectUri: 'https://app.example.com/callback',
  scopes: ['users:read', 'orders:read'],
};

// ステップ1: 認可リクエストの開始
app.get('/auth/login', (req, res) => {
  // CSRF防止用のstateパラメータを生成
  const state = crypto.randomBytes(32).toString('hex');
  req.session.oauthState = state;

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: OAUTH_CONFIG.clientId,
    redirect_uri: OAUTH_CONFIG.redirectUri,
    scope: OAUTH_CONFIG.scopes.join(' '),
    state: state,
  });

  res.redirect(`${OAUTH_CONFIG.authorizationEndpoint}?${params}`);
});

// ステップ2: コールバック処理
app.get('/callback', async (req, res) => {
  const { code, state, error } = req.query;

  // エラーチェック
  if (error) {
    return res.status(400).json({
      error: 'OAuth error',
      detail: req.query.error_description || error,
    });
  }

  // stateパラメータの検証（CSRF防止）
  if (state !== req.session.oauthState) {
    return res.status(403).json({
      error: 'Invalid state',
      detail: 'State parameter mismatch. Possible CSRF attack.',
    });
  }
  delete req.session.oauthState;

  // ステップ3: 認可コードをトークンに交換
  const tokenResponse = await fetch(OAUTH_CONFIG.tokenEndpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code: code,
      redirect_uri: OAUTH_CONFIG.redirectUri,
      client_id: OAUTH_CONFIG.clientId,
      client_secret: OAUTH_CONFIG.clientSecret,
    }),
  });

  if (!tokenResponse.ok) {
    const errorData = await tokenResponse.json();
    return res.status(400).json({
      error: 'Token exchange failed',
      detail: errorData.error_description || 'Failed to exchange code for token.',
    });
  }

  const tokens = await tokenResponse.json();
  // {
  //   access_token: "eyJhbG...",
  //   token_type: "Bearer",
  //   expires_in: 3600,
  //   refresh_token: "rt_abc...",
  //   scope: "users:read orders:read"
  // }

  // セッションにトークンを保存
  req.session.accessToken = tokens.access_token;
  req.session.refreshToken = tokens.refresh_token;
  req.session.tokenExpiresAt = Date.now() + tokens.expires_in * 1000;

  res.redirect('/dashboard');
});
```

### 6.4 Authorization Code + PKCE（推奨: SPA/モバイル）

PKCE（Proof Key for Code Exchange、RFC 7636）は、パブリッククライアント（SPA・モバイルアプリ）において認可コード横取り攻撃を防ぐための拡張である。

```
PKCE のメカニズム:

  なぜPKCEが必要か:
  ┌──────────────────────────────────────────────────────────┐
  │ SPA/モバイルアプリでは client_secret を安全に保持できない  │
  │                                                          │
  │ 問題: 認可コードが傍受された場合                          │
  │                                                          │
  │   正規アプリ → 認可サーバー → 認可コード → [傍受] → 攻撃者 │
  │                                                          │
  │   攻撃者が認可コードを使ってトークンを取得できてしまう      │
  │                                                          │
  │ PKCE の解決策:                                            │
  │   認可コードだけでは不十分にする                           │
  │   → code_verifier（秘密の値）を持っているアプリのみ       │
  │     トークン交換が可能                                    │
  └──────────────────────────────────────────────────────────┘

  PKCEのフロー:

  1. クライアントが code_verifier をランダム生成（43-128文字）
     code_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"

  2. code_challenge を計算
     code_challenge = BASE64URL(SHA256(code_verifier))
     code_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"

  3. 認可リクエストに code_challenge を含める
     GET /authorize?
       response_type=code&
       client_id=client_123&
       redirect_uri=https://app.example.com/callback&
       scope=openid profile&
       state=xyz&
       code_challenge=E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM&
       code_challenge_method=S256

  4. トークン交換時に code_verifier を含める
     POST /oauth/token
     {
       "grant_type": "authorization_code",
       "code": "auth_code_xxx",
       "redirect_uri": "https://app.example.com/callback",
       "client_id": "client_123",
       "code_verifier": "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
     }

  5. 認可サーバーが検証
     SHA256(code_verifier) == code_challenge ?
     → 一致すればトークンを発行
     → 不一致なら拒否
```

```javascript
// SPA向け PKCE実装例
class OAuthPKCEClient {
  constructor(config) {
    this.config = config;
  }

  // code_verifierの生成（43-128文字のランダム文字列）
  generateCodeVerifier() {
    const array = new Uint8Array(32);
    crypto.getRandomValues(array);
    return this.base64UrlEncode(array);
  }

  // code_challengeの計算
  async generateCodeChallenge(verifier) {
    const encoder = new TextEncoder();
    const data = encoder.encode(verifier);
    const digest = await crypto.subtle.digest('SHA-256', data);
    return this.base64UrlEncode(new Uint8Array(digest));
  }

  // Base64URLエンコード
  base64UrlEncode(buffer) {
    return btoa(String.fromCharCode(...buffer))
      .replace(/\+/g, '-')
      .replace(/\//g, '_')
      .replace(/=+$/, '');
  }

  // 認可リクエストの開始
  async startAuthorization() {
    const codeVerifier = this.generateCodeVerifier();
    const codeChallenge = await this.generateCodeChallenge(codeVerifier);
    const state = crypto.randomUUID();

    // code_verifierとstateをセッションストレージに保存
    sessionStorage.setItem('pkce_code_verifier', codeVerifier);
    sessionStorage.setItem('oauth_state', state);

    const params = new URLSearchParams({
      response_type: 'code',
      client_id: this.config.clientId,
      redirect_uri: this.config.redirectUri,
      scope: this.config.scopes.join(' '),
      state: state,
      code_challenge: codeChallenge,
      code_challenge_method: 'S256',
    });

    window.location.href =
      `${this.config.authorizationEndpoint}?${params}`;
  }

  // コールバック処理（トークン交換）
  async handleCallback() {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const state = params.get('state');
    const error = params.get('error');

    if (error) {
      throw new Error(`OAuth error: ${params.get('error_description') || error}`);
    }

    // stateの検証
    const savedState = sessionStorage.getItem('oauth_state');
    if (state !== savedState) {
      throw new Error('State mismatch: possible CSRF attack');
    }

    // code_verifierの取得
    const codeVerifier = sessionStorage.getItem('pkce_code_verifier');
    if (!codeVerifier) {
      throw new Error('Code verifier not found');
    }

    // セッションストレージのクリーンアップ
    sessionStorage.removeItem('oauth_state');
    sessionStorage.removeItem('pkce_code_verifier');

    // トークン交換
    const response = await fetch(this.config.tokenEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code: code,
        redirect_uri: this.config.redirectUri,
        client_id: this.config.clientId,
        code_verifier: codeVerifier,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(
        `Token exchange failed: ${errorData.error_description || errorData.error}`
      );
    }

    return response.json();
  }
}

// 使用例
const oauth = new OAuthPKCEClient({
  clientId: 'spa_client_123',
  authorizationEndpoint: 'https://auth.example.com/authorize',
  tokenEndpoint: 'https://auth.example.com/oauth/token',
  redirectUri: 'https://spa.example.com/callback',
  scopes: ['openid', 'profile', 'email'],
});

// ログインボタンクリック時
document.getElementById('loginBtn').addEventListener('click', () => {
  oauth.startAuthorization();
});

// コールバックページ
if (window.location.pathname === '/callback') {
  oauth.handleCallback()
    .then(tokens => {
      console.log('Login successful:', tokens);
      // access_tokenをメモリに保持（localStorageには保存しない）
    })
    .catch(error => {
      console.error('Login failed:', error);
    });
}
```

### 6.5 Client Credentials Flow（サーバー間通信）

ユーザーが介在しないサーバー間通信に使用するフローである。

```javascript
// Client Credentials Flow の実装例
async function getServiceToken() {
  const response = await fetch('https://auth.example.com/oauth/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'client_credentials',
      client_id: process.env.SERVICE_CLIENT_ID,
      client_secret: process.env.SERVICE_CLIENT_SECRET,
      scope: 'internal:admin',
    }),
  });

  if (!response.ok) {
    throw new Error(`Token request failed: ${response.status}`);
  }

  return response.json();
}

// トークンキャッシュ付きクライアント
class ServiceAuthClient {
  constructor() {
    this.token = null;
    this.expiresAt = 0;
  }

  async getToken() {
    // トークンの有効期限を5分前にチェック（バッファ）
    if (this.token && Date.now() < this.expiresAt - 5 * 60 * 1000) {
      return this.token;
    }

    const tokenData = await getServiceToken();
    this.token = tokenData.access_token;
    this.expiresAt = Date.now() + tokenData.expires_in * 1000;

    return this.token;
  }

  async authenticatedFetch(url, options = {}) {
    const token = await this.getToken();
    return fetch(url, {
      ...options,
      headers: {
        ...options.headers,
        Authorization: `Bearer ${token}`,
      },
    });
  }
}
```

### 6.6 Implicit Flow（非推奨）

Implicit Flowはかつてブラウザベースのアプリケーション向けに設計されたが、現在ではセキュリティ上の理由から非推奨とされている。

```
Implicit Flow が非推奨とされる理由:

  Implicit Flow:
    認可リクエスト → access_token がURLフラグメントで直接返却
    例: https://app.example.com/callback#access_token=xxx&token_type=Bearer

  問題点:
  1. access_tokenがブラウザ履歴に残る
  2. Refererヘッダー経由で漏洩する可能性がある
  3. トークンの検証がクライアント側で行われるため、
     トークン置換攻撃（Token Substitution Attack）に脆弱
  4. refresh_tokenが発行されないため、
     トークン期限切れ時にユーザー再認証が必要

  推奨される代替:
    Authorization Code Flow + PKCE
    → すべてのパブリッククライアントでPKCEを使用すべき
    → OAuth 2.1 ドラフトではImplicit Flowは削除予定
```

---

## 7. JWT（JSON Web Token）

### 7.1 構造の詳細

JWT（RFC 7519）は、当事者間で情報を安全にJSON形式で転送するためのコンパクトなトークン形式である。

```
JWT の構造（3つのパート）:

  eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyXzEyMyJ9.SflKxwRJSMeKKF2QT4fw...
  ├─── Header ───┤├──── Payload ────┤├──── Signature ────┤

  各パートはBase64URLエンコードされ、ドット（.）で連結される

  ┌─────────────────────────────────────────────────────────┐
  │ Header（ヘッダー）                                       │
  │ アルゴリズムとトークンタイプの宣言                       │
  │                                                         │
  │ {                                                       │
  │   "alg": "RS256",        ← 署名アルゴリズム             │
  │   "typ": "JWT",          ← トークンタイプ               │
  │   "kid": "key_2024_01"   ← 署名鍵のID（Key ID）        │
  │ }                                                       │
  ├─────────────────────────────────────────────────────────┤
  │ Payload（ペイロード / Claims）                           │
  │ トークンに含まれるデータ                                │
  │                                                         │
  │ 登録済みクレーム（Registered Claims）:                   │
  │ {                                                       │
  │   "iss": "https://auth.example.com",  ← Issuer 発行者   │
  │   "sub": "user_123",                  ← Subject 主体    │
  │   "aud": "https://api.example.com",   ← Audience 対象者 │
  │   "exp": 1700000000,                  ← Expiration 期限 │
  │   "iat": 1699996400,                  ← Issued At 発行  │
  │   "nbf": 1699996400,                  ← Not Before 開始 │
  │   "jti": "unique_token_id"            ← JWT ID 識別子   │
  │ }                                                       │
  │                                                         │
  │ パブリッククレーム（Public Claims）:                     │
  │ {                                                       │
  │   "email": "user@example.com",                          │
  │   "name": "John Doe"                                    │
  │ }                                                       │
  │                                                         │
  │ プライベートクレーム（Private Claims）:                  │
  │ {                                                       │
  │   "scope": "users:read orders:read",                    │
  │   "role": "admin",                                      │
  │   "tenant_id": "org_456"                                │
  │ }                                                       │
  ├─────────────────────────────────────────────────────────┤
  │ Signature（署名）                                        │
  │ ヘッダーとペイロードの改ざん検知                        │
  │                                                         │
  │ RS256の場合:                                            │
  │ RSASHA256(                                              │
  │   base64UrlEncode(header) + "." +                       │
  │   base64UrlEncode(payload),                             │
  │   privateKey                                            │
  │ )                                                       │
  └─────────────────────────────────────────────────────────┘
```

### 7.2 署名アルゴリズムの選択

```
署名アルゴリズムの比較:

  アルゴリズム  種類         鍵の長さ    用途              パフォーマンス
  ──────────────────────────────────────────────────────────────────
  HS256        対称鍵       256bit      単一サービス       最速
  HS384        対称鍵       384bit      単一サービス       速い
  HS512        対称鍵       512bit      単一サービス       速い
  RS256        非対称鍵     2048bit     マイクロサービス   中程度
  RS384        非対称鍵     3072bit     マイクロサービス   遅い
  RS512        非対称鍵     4096bit     マイクロサービス   遅い
  ES256        楕円曲線     256bit      モバイル/IoT      速い
  ES384        楕円曲線     384bit      高セキュリティ     中程度
  ES512        楕円曲線     521bit      高セキュリティ     中程度
  EdDSA        Edwards曲線  256bit      最新のシステム     最速（非対称）
  ──────────────────────────────────────────────────────────────────

  選択指針:
  ┌──────────────────────────────────────────────────────────────┐
  │ 単一サーバー → HS256（対称鍵、シンプル）                     │
  │ マイクロサービス → RS256 or ES256（公開鍵で検証可能）        │
  │ モバイル/IoT → ES256（短い鍵でRSA同等のセキュリティ）       │
  │ 新規設計 → EdDSA（最新かつ高性能、ライブラリ対応要確認）    │
  │                                                              │
  │ [重要] "alg": "none" は絶対に許可しない                     │
  │ → 署名なしのJWTを受け入れる脆弱性（CVE-2015-9235）          │
  └──────────────────────────────────────────────────────────────┘
```

### 7.3 JWTの検証実装

```javascript
// JWT の検証（jose ライブラリ）- 本番品質の実装
import { jwtVerify, createRemoteJWKSet, errors } from 'jose';

// JWKS（JSON Web Key Set）から公開鍵を取得
// JWKSエンドポイントは認可サーバーが公開する
const JWKS = createRemoteJWKSet(
  new URL('https://auth.example.com/.well-known/jwks.json'),
  {
    cooldownDuration: 30000,  // 30秒のクールダウン（連続リクエスト防止）
    cacheMaxAge: 600000,      // 10分のキャッシュ
  }
);

async function verifyToken(token) {
  try {
    const { payload, protectedHeader } = await jwtVerify(token, JWKS, {
      issuer: 'https://auth.example.com',
      audience: 'https://api.example.com',
      algorithms: ['RS256', 'ES256'],  // 許可するアルゴリズムを明示
      maxTokenAge: '1h',               // 発行から1時間以内
      clockTolerance: 30,              // 30秒のクロックスキュー許容
    });

    return {
      userId: payload.sub,
      scopes: payload.scope?.split(' ') || [],
      roles: payload.role ? [payload.role] : [],
      expiresAt: new Date(payload.exp * 1000),
      issuedAt: new Date(payload.iat * 1000),
    };
  } catch (error) {
    if (error instanceof errors.JWTExpired) {
      throw new AuthError('TOKEN_EXPIRED', 'The access token has expired.');
    }
    if (error instanceof errors.JWTClaimValidationFailed) {
      throw new AuthError('INVALID_CLAIMS', `Token claim validation failed: ${error.message}`);
    }
    if (error instanceof errors.JWSSignatureVerificationFailed) {
      throw new AuthError('INVALID_SIGNATURE', 'Token signature verification failed.');
    }
    throw new AuthError('INVALID_TOKEN', 'The access token is invalid.');
  }
}

// カスタムエラークラス
class AuthError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
    this.name = 'AuthError';
  }
}

// Express.js ミドルウェア
async function jwtAuthMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({
      type: 'https://api.example.com/errors/missing-token',
      title: 'Authentication Required',
      status: 401,
      detail: 'A Bearer token is required in the Authorization header.',
    });
  }

  const token = authHeader.substring(7);

  try {
    req.user = await verifyToken(token);
    next();
  } catch (error) {
    if (error instanceof AuthError) {
      const status = error.code === 'TOKEN_EXPIRED' ? 401 : 403;
      return res.status(status).json({
        type: `https://api.example.com/errors/${error.code.toLowerCase().replace(/_/g, '-')}`,
        title: error.code,
        status: status,
        detail: error.message,
      });
    }
    return res.status(401).json({
      type: 'https://api.example.com/errors/authentication-failed',
      title: 'Authentication Failed',
      status: 401,
      detail: 'Failed to authenticate the request.',
    });
  }
}
```

### 7.4 JWT発行の実装

```javascript
// JWT の発行（jose ライブラリ）
import { SignJWT, importPKCS8 } from 'jose';
import fs from 'fs';

// 秘密鍵の読み込み（RS256）
const privateKeyPem = fs.readFileSync('./keys/private.pem', 'utf-8');
const privateKey = await importPKCS8(privateKeyPem, 'RS256');

async function issueAccessToken(user, scopes) {
  const token = await new SignJWT({
    scope: scopes.join(' '),
    role: user.role,
    email: user.email,
  })
    .setProtectedHeader({
      alg: 'RS256',
      typ: 'JWT',
      kid: 'key_2024_01',
    })
    .setSubject(user.id)
    .setIssuer('https://auth.example.com')
    .setAudience('https://api.example.com')
    .setIssuedAt()
    .setExpirationTime('15m')  // 15分の有効期限
    .setJti(crypto.randomUUID())
    .sign(privateKey);

  return token;
}

async function issueRefreshToken(user) {
  const refreshTokenId = crypto.randomUUID();

  // Refresh Tokenはデータベースに保存（ステートフル）
  await db.refreshTokens.insertOne({
    id: refreshTokenId,
    userId: user.id,
    createdAt: new Date(),
    expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30日
    revokedAt: null,
    family: crypto.randomUUID(), // Token Rotation用のファミリーID
  });

  // Refresh TokenもJWTとして発行（ただしペイロードは最小限）
  const token = await new SignJWT({ type: 'refresh' })
    .setProtectedHeader({ alg: 'RS256', typ: 'JWT' })
    .setSubject(user.id)
    .setIssuer('https://auth.example.com')
    .setExpirationTime('30d')
    .setJti(refreshTokenId)
    .sign(privateKey);

  return token;
}
```

---

## 8. Access Token + Refresh Token

### 8.1 トークンライフサイクル

```
トークンライフサイクルの全体像:

  ┌──────────────────────────────────────────────────────────────┐
  │                       ユーザーログイン                       │
  │                           │                                  │
  │                    ┌──────┴──────┐                           │
  │                    │ トークン発行 │                           │
  │                    └──────┬──────┘                           │
  │                           │                                  │
  │              ┌────────────┼────────────┐                    │
  │              │                         │                    │
  │         Access Token            Refresh Token               │
  │         (短命: 15分)            (長命: 30日)                │
  │              │                         │                    │
  │         ┌────┴────┐                    │                    │
  │         │ API呼出 │                    │                    │
  │         └────┬────┘                    │                    │
  │              │                         │                    │
  │         [期限切れ]                     │                    │
  │              │                         │                    │
  │         ┌────┴─────────┐               │                    │
  │         │ リフレッシュ  │←─────────────┘                    │
  │         └────┬─────────┘                                    │
  │              │                                              │
  │    ┌─────────┼─────────┐                                    │
  │    │                   │                                    │
  │ 新Access Token    新Refresh Token                           │
  │ (15分)            (30日)                                    │
  │                   旧Refresh Token → 即座に無効化            │
  │                                                             │
  │         [異常検知時]                                         │
  │              │                                              │
  │    ┌─────────┴─────────┐                                    │
  │    │  全トークン無効化  │                                    │
  │    │  (ファミリー単位)  │                                    │
  │    └───────────────────┘                                    │
  └──────────────────────────────────────────────────────────────┘
```

### 8.2 Refresh Token Rotation

Refresh Token Rotationは、Refresh Token使用時に新しいペアを発行し、古いトークンを即座に無効化する手法である。盗難検知に有効なセキュリティパターンである。

```javascript
// Refresh Token Rotation の実装
async function refreshTokens(refreshToken) {
  // 1. Refresh Tokenの検証
  let payload;
  try {
    const result = await jwtVerify(refreshToken, JWKS, {
      issuer: 'https://auth.example.com',
    });
    payload = result.payload;
  } catch {
    throw new AuthError('INVALID_REFRESH_TOKEN', 'The refresh token is invalid.');
  }

  // 2. データベースでトークンの状態を確認
  const tokenRecord = await db.refreshTokens.findOne({ id: payload.jti });

  if (!tokenRecord) {
    throw new AuthError('TOKEN_NOT_FOUND', 'Refresh token not found.');
  }

  // 3. 既に無効化されたトークンが使われた場合 → 盗難の可能性
  if (tokenRecord.revokedAt) {
    // 同じファミリーの全トークンを無効化（セキュリティ対策）
    await db.refreshTokens.updateMany(
      { family: tokenRecord.family },
      { $set: { revokedAt: new Date(), revokeReason: 'reuse_detected' } }
    );

    // セキュリティアラートの送信
    await notifySecurityTeam({
      type: 'REFRESH_TOKEN_REUSE',
      userId: tokenRecord.userId,
      tokenId: tokenRecord.id,
      family: tokenRecord.family,
    });

    throw new AuthError(
      'TOKEN_REUSE_DETECTED',
      'Refresh token reuse detected. All sessions have been revoked.'
    );
  }

  // 4. 有効期限の確認
  if (tokenRecord.expiresAt < new Date()) {
    throw new AuthError('REFRESH_TOKEN_EXPIRED', 'The refresh token has expired.');
  }

  // 5. 古いRefresh Tokenを無効化
  await db.refreshTokens.updateOne(
    { id: tokenRecord.id },
    { $set: { revokedAt: new Date(), revokeReason: 'rotated' } }
  );

  // 6. ユーザー情報を取得
  const user = await db.users.findOne({ id: tokenRecord.userId });

  // 7. 新しいトークンペアを発行
  const newAccessToken = await issueAccessToken(user, user.scopes);
  const newRefreshToken = await issueRefreshToken(user);

  // 新しいRefresh Tokenは同じファミリーに属させる
  await db.refreshTokens.updateOne(
    { id: (await jwtVerify(newRefreshToken, JWKS)).payload.jti },
    { $set: { family: tokenRecord.family } }
  );

  return {
    access_token: newAccessToken,
    token_type: 'Bearer',
    expires_in: 900, // 15分
    refresh_token: newRefreshToken,
  };
}
```

### 8.3 トークンの保存場所

```
トークンの安全な保存場所:

  ┌────────────────────────────────────────────────────────────┐
  │ プラットフォーム別の推奨保存場所                           │
  ├────────────────────────────────────────────────────────────┤
  │                                                            │
  │ [Web SPA]                                                  │
  │   Access Token  → JavaScript変数（メモリ内）              │
  │   Refresh Token → HttpOnly Cookie                         │
  │     属性: Secure; HttpOnly; SameSite=Strict; Path=/auth   │
  │                                                            │
  │   NG: localStorage（XSSで盗まれる）                       │
  │   NG: sessionStorage（XSSで盗まれる）                     │
  │   NG: 通常のCookie（JavaScriptからアクセス可能）          │
  │                                                            │
  │ [モバイル（iOS）]                                          │
  │   Access Token  → メモリ                                  │
  │   Refresh Token → Keychain Services                       │
  │     kSecAttrAccessible: kSecAttrAccessibleAfterFirstUnlock│
  │                                                            │
  │ [モバイル（Android）]                                      │
  │   Access Token  → メモリ                                  │
  │   Refresh Token → EncryptedSharedPreferences              │
  │     または Android Keystore                               │
  │                                                            │
  │ [サーバーサイド]                                           │
  │   Access Token  → メモリ / Redis                          │
  │   Refresh Token → 暗号化されたデータベース                │
  └────────────────────────────────────────────────────────────┘
```

---

## 9. スコープ設計

### 9.1 設計原則

スコープはOAuth 2.0における権限制御の単位であり、クライアントがアクセスできるリソースと操作の範囲を定義する。

```
スコープの設計原則:

  形式: リソース:操作
  原則: 最小権限の原則（Principle of Least Privilege）

  基本的なスコープ例:
  ┌──────────────────────────────────────────────────────┐
  │ スコープ名          説明                              │
  ├──────────────────────────────────────────────────────┤
  │ users:read          ユーザー情報の読み取り            │
  │ users:write         ユーザー情報の作成・更新          │
  │ users:delete        ユーザーの削除                    │
  │ orders:read         注文情報の読み取り                │
  │ orders:write        注文の作成・更新                  │
  │ orders:delete       注文の削除                        │
  │ billing:read        請求情報の読み取り                │
  │ billing:manage      請求の管理（作成・更新・削除）    │
  │ admin:all           管理者権限（全操作）              │
  │ openid              OpenID Connect必須スコープ        │
  │ profile             ユーザープロフィール情報          │
  │ email               メールアドレス                    │
  └──────────────────────────────────────────────────────┘

  階層的スコープの設計:
    read  < write < admin
    users:read ⊂ users:write ⊂ users:admin ⊂ admin:all

  スコープの粒度指針:
    粗すぎる: api:access（全APIアクセス）→ 権限が広すぎる
    細かすぎる: users:name:read（名前の読取）→ 管理が煩雑
    適切: users:read（ユーザー情報の読取）→ バランスが良い
```

### 9.2 スコープ検証の実装

```javascript
// スコープチェックミドルウェア
function requireScope(...requiredScopes) {
  return (req, res, next) => {
    if (!req.user || !req.user.scopes) {
      return res.status(401).json({
        type: 'https://api.example.com/errors/unauthenticated',
        title: 'Authentication Required',
        status: 401,
      });
    }

    const tokenScopes = req.user.scopes;

    // 階層的スコープの解決
    const effectiveScopes = resolveHierarchicalScopes(tokenScopes);

    const hasAllScopes = requiredScopes.every(
      scope => effectiveScopes.includes(scope)
    );

    if (!hasAllScopes) {
      return res.status(403).json({
        type: 'https://api.example.com/errors/insufficient-scope',
        title: 'Insufficient Scope',
        status: 403,
        detail: `Required scopes: ${requiredScopes.join(', ')}`,
        required_scopes: requiredScopes,
        granted_scopes: tokenScopes,
      });
    }

    next();
  };
}

// 階層的スコープの解決
function resolveHierarchicalScopes(scopes) {
  const hierarchy = {
    'admin:all': ['users:read', 'users:write', 'users:delete',
                  'orders:read', 'orders:write', 'orders:delete',
                  'billing:read', 'billing:manage'],
    'users:write': ['users:read'],
    'users:delete': ['users:read', 'users:write'],
    'orders:write': ['orders:read'],
    'orders:delete': ['orders:read', 'orders:write'],
    'billing:manage': ['billing:read'],
  };

  const resolved = new Set(scopes);
  for (const scope of scopes) {
    if (hierarchy[scope]) {
      hierarchy[scope].forEach(s => resolved.add(s));
    }
  }

  return Array.from(resolved);
}

// ルーティングへの適用
app.get('/api/v1/users', requireScope('users:read'), listUsers);
app.post('/api/v1/users', requireScope('users:write'), createUser);
app.delete('/api/v1/users/:id', requireScope('users:delete'), deleteUser);
app.get('/api/v1/orders', requireScope('orders:read'), listOrders);
app.post('/api/v1/orders', requireScope('orders:write'), createOrder);
app.get('/api/v1/billing', requireScope('billing:read'), getBilling);
app.post('/api/v1/billing', requireScope('billing:manage'), updateBilling);
```

---

## 10. mTLS（相互TLS認証）

### 10.1 仕組み

mTLS（Mutual TLS）は、通常のTLS（サーバー証明書のみ）に加えて、クライアント証明書による認証を行う方式である。金融、医療、政府系APIなど、最高レベルのセキュリティが求められる場面で採用される。

```
mTLS のハンドシェイク:

  通常のTLS（一方向）:
    Client → Server: ClientHello
    Client ← Server: ServerHello + Server Certificate
    Client:          サーバー証明書を検証
    Client → Server: 暗号化通信開始

  mTLS（双方向）:
    Client → Server: ClientHello
    Client ← Server: ServerHello + Server Certificate
                      + CertificateRequest ← ★クライアント証明書を要求
    Client:          サーバー証明書を検証
    Client → Server: Client Certificate    ← ★クライアント証明書を送信
                      + CertificateVerify  ← ★署名で所有証明
    Server:          クライアント証明書を検証
    双方:            暗号化通信開始

  信頼チェーン:
    ┌──────────┐         ┌──────────┐
    │ Root CA   │ ──────→ │ 中間CA    │
    └──────────┘         └────┬─────┘
                              │
                    ┌─────────┼─────────┐
                    │                   │
              ┌─────┴─────┐       ┌─────┴─────┐
              │ サーバー   │       │ クライアント│
              │ 証明書     │       │ 証明書     │
              └───────────┘       └───────────┘
```

```javascript
// Node.js でのmTLSサーバー設定
import https from 'https';
import fs from 'fs';
import express from 'express';

const app = express();

// mTLSミドルウェア: クライアント証明書の情報を抽出
app.use((req, res, next) => {
  const cert = req.socket.getPeerCertificate();

  if (!req.client.authorized) {
    return res.status(403).json({
      error: 'Client certificate required',
      detail: 'A valid client certificate is required for this endpoint.',
    });
  }

  req.clientCert = {
    subject: cert.subject,
    issuer: cert.issuer,
    serialNumber: cert.serialNumber,
    fingerprint: cert.fingerprint256,
    validFrom: cert.valid_from,
    validTo: cert.valid_to,
  };

  next();
});

const server = https.createServer(
  {
    key: fs.readFileSync('./certs/server-key.pem'),
    cert: fs.readFileSync('./certs/server-cert.pem'),
    ca: fs.readFileSync('./certs/ca-cert.pem'),
    requestCert: true,       // クライアント証明書を要求
    rejectUnauthorized: true, // 無効な証明書は拒否
  },
  app
);

server.listen(443, () => {
  console.log('mTLS server running on port 443');
});
```
