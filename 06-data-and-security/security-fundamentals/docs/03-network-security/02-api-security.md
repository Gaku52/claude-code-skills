# API セキュリティ

> OAuth 2.0/JWT による認証認可、レートリミットによる過負荷防止、入力検証による攻撃防御、GraphQL セキュリティまで、API を安全に公開するための包括的ガイド

## この章で学ぶこと

1. **API の脅威モデル** -- OWASP API Security Top 10 の各脆弱性カテゴリとその実装レベルでの防御
2. **OAuth 2.0 / OpenID Connect** -- 認可フロー選択、PKCE、トークン管理、失効管理
3. **JWT の安全な実装** -- 署名検証、クレーム検証、トークンストレージ、リフレッシュ戦略
4. **レートリミットと API 保護** -- アルゴリズム比較、分散レートリミット、DDoS 防御
5. **入力検証とスキーマバリデーション** -- OpenAPI、express-validator、Zod によるサーバサイド検証
6. **GraphQL セキュリティ** -- クエリ深度制限、コスト解析、イントロスペクション制御

### 前提知識

- HTTP/HTTPS プロトコルの基本 (メソッド、ステータスコード、ヘッダ)
- REST API の設計原則
- JSON の構造と操作
- Node.js または Python の基本的なサーバサイド開発経験

---

## 1. API の脅威モデル

### API アーキテクチャと攻撃面

```
                    攻撃面の全体像

クライアント          API Gateway         バックエンド
  |                      |                    |
  |  [認証攻撃]           |                    |
  |  - Credential        |                    |
  |    Stuffing          |                    |
  |  - Brute Force       |                    |
  |  - Token Theft       |                    |
  |                      |                    |
  +------- HTTPS ------->|                    |
  |                      |  [認可攻撃]         |
  |                      |  - BOLA            |
  |                      |  - BFLA            |
  |                      |  - Mass Assignment |
  |                      |                    |
  |                      +------ gRPC ------->|
  |                      |                    |
  |                      |                    |  [バックエンド攻撃]
  |                      |                    |  - Injection
  |                      |                    |  - SSRF
  |                      |                    |  - Business Logic
  |                      |                    |
  |<----- Response ------|<---- Response -----|
  |                      |                    |
  |  [レスポンス攻撃]     |                    |
  |  - データ過剰露出     |                    |
  |  - エラー情報漏洩     |                    |
```

### OWASP API Security Top 10 (2023) 詳細

```
+------+------------------------------------------------+-----------+
| 順位 | 脆弱性                                          | 深刻度    |
+------+------------------------------------------------+-----------+
| API1 | Broken Object Level Authorization (BOLA)       | Critical  |
|      | → 他ユーザのリソースに不正アクセス                |           |
+------+------------------------------------------------+-----------+
| API2 | Broken Authentication                          | Critical  |
|      | → 認証メカニズムの欠陥                           |           |
+------+------------------------------------------------+-----------+
| API3 | Broken Object Property Level Authorization     | High      |
|      | → オブジェクトのプロパティレベルでの認可不備       |           |
+------+------------------------------------------------+-----------+
| API4 | Unrestricted Resource Consumption              | High      |
|      | → リソース消費の制限なし (DoS)                   |           |
+------+------------------------------------------------+-----------+
| API5 | Broken Function Level Authorization            | High      |
|      | → 管理者機能への不正アクセス                      |           |
+------+------------------------------------------------+-----------+
| API6 | Unrestricted Access to Sensitive Business Flow | Medium    |
|      | → ビジネスロジックの悪用                          |           |
+------+------------------------------------------------+-----------+
| API7 | Server Side Request Forgery (SSRF)             | High      |
|      | → サーバ側からの不正なリクエスト                  |           |
+------+------------------------------------------------+-----------+
| API8 | Security Misconfiguration                      | Medium    |
|      | → セキュリティ設定の不備                          |           |
+------+------------------------------------------------+-----------+
| API9 | Improper Inventory Management                  | Medium    |
|      | → API のバージョン・エンドポイント管理不備         |           |
+------+------------------------------------------------+-----------+
| API10| Unsafe Consumption of APIs                     | Medium    |
|      | → 外部 API の安全でない利用                       |           |
+------+------------------------------------------------+-----------+
```

### API1: BOLA (Broken Object Level Authorization) の詳細

```
攻撃シナリオ:

正規ユーザ (user_id=123):
  GET /api/orders/1001 → 200 OK (自分の注文)

攻撃者 (user_id=456):
  GET /api/orders/1001 → 200 OK (他人の注文が見えてしまう!)
  GET /api/orders/1002 → 200 OK (連番で全注文を列挙)

BOLA の種類:
  1. IDOR (Insecure Direct Object Reference)
     → /api/users/123/profile → /api/users/124/profile

  2. パラメータ操作
     → POST /api/transfer {"from": "my-account", "to": "attacker"}
     → POST /api/transfer {"from": "victim-account", "to": "attacker"}

  3. UUID でも安全ではない
     → UUID を推測できなくても、レスポンスの中に他のリソースの
        UUID が含まれていれば攻撃可能
```

```javascript
// BOLA 防御の実装パターン (Express.js)

// NG: オブジェクトIDのみで認可チェックなし
app.get('/api/orders/:orderId', async (req, res) => {
  const order = await Order.findById(req.params.orderId);
  res.json(order);  // 他人の注文も取得できてしまう
});

// OK: ミドルウェアパターンでオブジェクト所有者の検証
function authorizeResource(model, ownerField = 'userId') {
  return async (req, res, next) => {
    const resource = await model.findById(req.params.id);
    if (!resource) {
      return res.status(404).json({ error: 'Resource not found' });
    }
    if (resource[ownerField].toString() !== req.user.id) {
      // 注意: 403 ではなく 404 を返すことで、リソースの存在を隠す
      return res.status(404).json({ error: 'Resource not found' });
    }
    req.resource = resource;
    next();
  };
}

app.get('/api/orders/:id', authenticate, authorizeResource(Order), (req, res) => {
  res.json(req.resource);
});

// OK: クエリレベルでの所有者フィルタリング (より安全)
app.get('/api/orders/:orderId', authenticate, async (req, res) => {
  const order = await Order.findOne({
    _id: req.params.orderId,
    userId: req.user.id,  // 認証ユーザのIDで絞り込み
  });
  if (!order) return res.status(404).json({ error: 'Not found' });
  res.json(order);
});

// OK: RBAC + ABAC の組み合わせ
app.get('/api/orders/:orderId', authenticate, async (req, res) => {
  const order = await Order.findById(req.params.orderId);
  if (!order) return res.status(404).json({ error: 'Not found' });

  // 管理者は全注文にアクセス可能
  if (req.user.role === 'admin') return res.json(order);

  // サポート担当は自部門の注文のみ
  if (req.user.role === 'support') {
    if (order.department !== req.user.department) {
      return res.status(404).json({ error: 'Not found' });
    }
    return res.json(order);
  }

  // 一般ユーザは自分の注文のみ
  if (order.userId.toString() !== req.user.id) {
    return res.status(404).json({ error: 'Not found' });
  }
  res.json(order);
});
```

### API3: Mass Assignment 防御

```javascript
// NG: リクエストボディをそのまま使う
app.put('/api/users/:id', authenticate, async (req, res) => {
  const user = await User.findByIdAndUpdate(req.params.id, req.body);
  // 攻撃者が {"role": "admin", "verified": true} を送信可能!
  res.json(user);
});

// OK: 許可フィールドのホワイトリスト
const ALLOWED_USER_FIELDS = ['name', 'email', 'bio', 'avatar'];

app.put('/api/users/:id', authenticate, async (req, res) => {
  // ホワイトリストでフィルタリング
  const updates = {};
  for (const field of ALLOWED_USER_FIELDS) {
    if (req.body[field] !== undefined) {
      updates[field] = req.body[field];
    }
  }

  const user = await User.findOneAndUpdate(
    { _id: req.params.id, _id: req.user.id },  // 所有者チェック
    { $set: updates },
    { new: true, select: '-password -__v' }     // パスワードを除外
  );

  if (!user) return res.status(404).json({ error: 'Not found' });
  res.json(user);
});
```

---

## 2. OAuth 2.0 / OpenID Connect

### 認可フロー選択ガイド

```
アプリケーションの種類は?
  |
  +-- サーバサイド Web アプリ
  |   → Authorization Code Flow (+ PKCE 推奨)
  |   理由: client_secret をサーバ側で安全に保管可能
  |
  +-- SPA (Single Page Application)
  |   → Authorization Code Flow + PKCE (必須)
  |   理由: client_secret をブラウザに保管できない
  |   注意: Implicit Flow は非推奨 (RFC 9700)
  |
  +-- モバイルアプリ / デスクトップ
  |   → Authorization Code Flow + PKCE
  |   理由: client_secret をクライアントに保管できない
  |   注意: カスタム URL スキームでリダイレクト
  |
  +-- マシン間通信 (M2M)
  |   → Client Credentials Flow
  |   理由: ユーザの関与なし、サーバ間の直接認証
  |
  +-- IoT / 入力制限デバイス
      → Device Authorization Flow (RFC 8628)
      理由: ブラウザやキーボードがないデバイス
```

### Authorization Code Flow + PKCE の詳細

```
Browser/App              Auth Server              Resource Server
    |                         |                         |
    | (1) code_verifier を生成 (暗号学的にランダムな43-128文字)
    | code_challenge = BASE64URL(SHA256(code_verifier))
    |                         |                         |
    |-- (2) /authorize -----> |                         |
    |   + response_type=code  |                         |
    |   + client_id           |                         |
    |   + redirect_uri        |                         |
    |   + scope=openid email  |                         |
    |   + state=random123     | (CSRF 防止)              |
    |   + code_challenge      |                         |
    |   + code_challenge_method=S256                    |
    |                         |                         |
    |  <-- (3) ログイン画面 -- |                         |
    |  -- ユーザ認証 -------> |                         |
    |                         |                         |
    |  <-- (4) redirect ------|                         |
    |   + code=authcode123    |                         |
    |   + state=random123     | (state を検証)           |
    |                         |                         |
    |-- (5) POST /token ----->|                         |
    |   + grant_type=         |                         |
    |     authorization_code  |                         |
    |   + code=authcode123    |                         |
    |   + code_verifier       | (PKCE 検証)             |
    |   + redirect_uri        |                         |
    |   + client_id           |                         |
    |                         |                         |
    |  <-- (6) tokens --------|                         |
    |   + access_token (JWT)  |                         |
    |   + refresh_token       |                         |
    |   + id_token (OIDC)     |                         |
    |   + expires_in=900      |                         |
    |                         |                         |
    |-- (7) API call ---------+-----------------------> |
    |   + Authorization:      |                         |
    |     Bearer <token>      |                         |
    |                         |                         |
    |  <-- (8) Response ------+-----------------------  |
```

### PKCE の実装 (Node.js)

```javascript
const crypto = require('crypto');

// PKCE code_verifier と code_challenge の生成
function generatePKCE() {
  // code_verifier: 暗号学的にランダムな 43-128 文字
  const verifier = crypto.randomBytes(32).toString('base64url');

  // code_challenge: SHA-256 ハッシュの Base64URL エンコード
  const challenge = crypto
    .createHash('sha256')
    .update(verifier)
    .digest('base64url');

  return { verifier, challenge };
}

// 認可リクエストの構築
function buildAuthorizationUrl(config) {
  const { verifier, challenge } = generatePKCE();
  const state = crypto.randomBytes(16).toString('hex');

  // state と verifier をセッションに保存
  // (SPA の場合は sessionStorage に保存)

  const params = new URLSearchParams({
    response_type: 'code',
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    scope: 'openid email profile',
    state: state,
    code_challenge: challenge,
    code_challenge_method: 'S256',
    // nonce: OIDC の場合、リプレイ攻撃防止
    nonce: crypto.randomBytes(16).toString('hex'),
  });

  return {
    url: `${config.authorizationEndpoint}?${params}`,
    state,
    verifier,
  };
}

// トークン交換
async function exchangeCodeForTokens(code, verifier, config) {
  const response = await fetch(config.tokenEndpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'authorization_code',
      code: code,
      redirect_uri: config.redirectUri,
      client_id: config.clientId,
      code_verifier: verifier,  // PKCE: code_verifier を送信
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Token exchange failed: ${error.error_description}`);
  }

  return response.json();
}
```

### トークンのリフレッシュとローテーション

```javascript
// リフレッシュトークンのローテーション実装
async function refreshAccessToken(refreshToken, config) {
  const response = await fetch(config.tokenEndpoint, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: new URLSearchParams({
      grant_type: 'refresh_token',
      refresh_token: refreshToken,
      client_id: config.clientId,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    // リフレッシュトークンが無効 → 再ログインが必要
    if (error.error === 'invalid_grant') {
      throw new AuthenticationError('Session expired, please login again');
    }
    throw new Error(`Token refresh failed: ${error.error_description}`);
  }

  const tokens = await response.json();
  // ローテーション: 新しい refresh_token が返される
  // 古い refresh_token は無効化される (一度きりの使用)
  return {
    accessToken: tokens.access_token,
    refreshToken: tokens.refresh_token,  // 新しいリフレッシュトークン
    expiresIn: tokens.expires_in,
  };
}

// Axios インターセプタでの自動リフレッシュ
const api = axios.create({ baseURL: 'https://api.example.com' });
let isRefreshing = false;
let failedQueue = [];

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // 既にリフレッシュ中なら、完了を待つ
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then((token) => {
          originalRequest.headers['Authorization'] = `Bearer ${token}`;
          return api(originalRequest);
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const { accessToken, refreshToken } = await refreshAccessToken(
          getStoredRefreshToken(),
          config
        );
        storeTokens(accessToken, refreshToken);

        // キューに溜まったリクエストを再実行
        failedQueue.forEach(({ resolve }) => resolve(accessToken));
        failedQueue = [];

        originalRequest.headers['Authorization'] = `Bearer ${accessToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        failedQueue.forEach(({ reject }) => reject(refreshError));
        failedQueue = [];
        // セッション無効化 → ログイン画面へ
        logout();
        throw refreshError;
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);
```

---

## 3. JWT の安全な実装

### JWT の構造と検証

```
JWT の構造:

Header.Payload.Signature
  |       |        |
  v       v        v

eyJhbGci.eyJzdWIi.SflKxwRJ
  |       |        |
  +-------+--------+-- Base64URL エンコード (暗号化ではない!)
          |
          v
  {
    "sub": "user123",      ← ユーザ識別子
    "iss": "auth.example.com", ← 発行者
    "aud": "api.example.com",  ← 対象 API
    "exp": 1700000000,     ← 有効期限 (UNIX timestamp)
    "iat": 1699999100,     ← 発行時刻
    "jti": "unique-id-123", ← トークン一意識別子
    "scope": "read write",  ← 認可スコープ
    "email": "user@example.com"
  }

署名アルゴリズム:
  HS256: HMAC-SHA256 (対称鍵) → M2M、単一サーバ向け
  RS256: RSA-SHA256 (非対称鍵) → 一般的な推奨
  ES256: ECDSA-SHA256 (楕円曲線) → 高速 + 短い鍵長
  EdDSA: Ed25519 (最新) → 最高速 + 最短鍵長
```

### JWT 検証ミドルウェア (Node.js)

```javascript
const jwt = require('jsonwebtoken');
const jwksClient = require('jwks-rsa');

// JWKS クライアント (公開鍵を自動取得 + キャッシュ)
const client = jwksClient({
  jwksUri: 'https://auth.example.com/.well-known/jwks.json',
  cache: true,            // 鍵をキャッシュ
  cacheMaxAge: 600000,    // 10分間キャッシュ
  rateLimit: true,        // レートリミット
  jwksRequestsPerMinute: 10,
});

// kid から署名検証鍵を取得
function getSigningKey(kid) {
  return new Promise((resolve, reject) => {
    client.getSigningKey(kid, (err, key) => {
      if (err) return reject(err);
      resolve(key.getPublicKey());
    });
  });
}

// JWT 検証ミドルウェア
async function verifyToken(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    return res.status(401).json({
      error: 'unauthorized',
      message: 'Bearer token required',
    });
  }

  const token = authHeader.slice(7);

  try {
    // Step 1: ヘッダをデコード (検証前) して kid を取得
    const decoded = jwt.decode(token, { complete: true });
    if (!decoded || !decoded.header.kid) {
      return res.status(401).json({ error: 'Invalid token format' });
    }

    // Step 2: アルゴリズムがホワイトリストにあるか確認
    if (!['RS256', 'ES256'].includes(decoded.header.alg)) {
      return res.status(401).json({ error: 'Unsupported algorithm' });
    }

    // Step 3: JWKS から公開鍵を取得
    const publicKey = await getSigningKey(decoded.header.kid);

    // Step 4: 署名検証 + クレーム検証
    const payload = jwt.verify(token, publicKey, {
      algorithms: ['RS256', 'ES256'],  // 許可アルゴリズムを明示
      issuer: 'https://auth.example.com',
      audience: 'https://api.example.com',
      clockTolerance: 30,  // 時刻ずれ許容 (秒)
      maxAge: '1h',        // 発行からの最大経過時間
    });

    // Step 5: 追加のカスタム検証
    if (!payload.scope) {
      return res.status(403).json({ error: 'No scope in token' });
    }

    req.user = payload;
    req.token = token;
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({
        error: 'token_expired',
        message: 'Access token has expired',
      });
    }
    if (err.name === 'JsonWebTokenError') {
      return res.status(401).json({
        error: 'invalid_token',
        message: 'Token signature verification failed',
      });
    }
    // 予期しないエラーの詳細は返さない
    console.error('Token verification error:', err);
    return res.status(401).json({ error: 'Authentication failed' });
  }
}

// スコープベースの認可ミドルウェア
function requireScope(...requiredScopes) {
  return (req, res, next) => {
    const tokenScopes = (req.user.scope || '').split(' ');
    const hasAll = requiredScopes.every(s => tokenScopes.includes(s));

    if (!hasAll) {
      return res.status(403).json({
        error: 'insufficient_scope',
        message: `Required scopes: ${requiredScopes.join(', ')}`,
      });
    }
    next();
  };
}

// 使用例
app.get('/api/users', verifyToken, requireScope('users:read'), getUsers);
app.post('/api/users', verifyToken, requireScope('users:write'), createUser);
app.delete('/api/users/:id', verifyToken, requireScope('users:delete'), deleteUser);
```

### JWT クレームのベストプラクティス

| クレーム | 必須 | 説明 | 検証方法 |
|---------|------|------|---------|
| `iss` | はい | トークン発行者 | 期待する発行者と完全一致 |
| `sub` | はい | ユーザ/クライアント識別子 | DB のユーザ ID と照合 |
| `aud` | はい | 対象 API (受信者) | 自 API の識別子と一致 |
| `exp` | はい | 有効期限 (短め: 15分-1時間) | 現在時刻 < exp |
| `iat` | はい | 発行時刻 | 未来の iat を拒否 |
| `nbf` | 推奨 | 有効開始時刻 | 現在時刻 >= nbf |
| `jti` | 推奨 | トークン一意識別子 | リプレイ攻撃防止用に記録 |
| `scope` | 推奨 | 認可スコープ | 要求操作に必要なスコープを確認 |
| `azp` | 条件付 | 認可されたクライアント | クライアント ID と照合 |

### JWT のストレージ戦略

```
ストレージ別リスク比較:

+------------------+-----------+-------+--------+----------+
| ストレージ        | XSS 耐性  | CSRF  | 有効範囲 | 推奨度   |
+------------------+-----------+-------+--------+----------+
| localStorage     | 脆弱      | 安全  | タブ間  | 非推奨   |
| sessionStorage   | 脆弱      | 安全  | タブ内  | 条件付   |
| Cookie (HttpOnly)| 安全      | 脆弱  | 自動送信| 推奨     |
| Cookie + SameSite| 安全      | 安全  | 自動送信| 最推奨   |
| メモリ (変数)     | 安全      | 安全  | タブ内  | 推奨     |
+------------------+-----------+-------+--------+----------+

推奨パターン (BFF: Backend For Frontend):
  1. Access Token → メモリ変数に保持 (XSS/CSRF 両方に安全)
  2. Refresh Token → HttpOnly + Secure + SameSite=Strict Cookie
  3. BFF がトークン管理を代行 → SPA はセッション Cookie のみ使用
```

---

## 4. レートリミット

### レートリミットのアルゴリズム

```
1. Token Bucket (トークンバケット):
   +-------------------+
   |  ○ ○ ○ ○ ○ ○ ○   |  バケット容量 = 10
   |  (トークン)        |  補充レート = 1/秒
   +-------------------+
   リクエスト → トークンを1個消費
   トークンなし → 429 Too Many Requests
   特徴: バーストを許容 (溜まったトークン分)

2. Leaky Bucket (漏れバケット):
   +---+
   | ● | ← リクエストが入る
   | ● |
   | ● |    一定レートで
   +---+    処理される
     |        ↓
     ● → → → API

   特徴: リクエストを平滑化、バースト不可

3. Fixed Window Counter:
   |--- Window 1 ---|--- Window 2 ---|
   |  count = 8     |  count = 3     |
   |  limit = 10    |  limit = 10    |
   +----------------+----------------+
   問題: ウィンドウ境界でバーストが発生
   (Window 1 の末尾 8 + Window 2 の先頭 10 = 18 リクエスト)

4. Sliding Window Log:
   |------ 60秒ウィンドウ ------| → スライド →
   |  *  *   *  * *  *  *  *   |  リクエスト数 = 8
   |                           |  上限 = 10 → OK
   +---------------------------+
   特徴: 正確だがメモリ消費が大きい

5. Sliding Window Counter (推奨):
   |--- 前の窓 ---|--- 現在の窓 ---|
   |  count=6    |  count=3      |
   |  重み=30%   |  重み=100%    |
   推定: 6*0.3 + 3*1.0 = 4.8 → 上限 10 以内
   特徴: 精度と効率のバランスが良い
```

### Redis ベースの分散レートリミッター (Python)

```python
import redis
import time
from functools import wraps
from flask import request, jsonify, g
import hashlib

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

def rate_limit(
    max_requests: int,
    window_seconds: int,
    key_func=None,
    scope: str = 'default'
):
    """スライディングウィンドウ方式の分散レートリミッター"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # クライアント識別
            if key_func:
                client_id = key_func()
            else:
                # API キー > 認証ユーザ > IP アドレス の優先順位
                client_id = (
                    request.headers.get('X-API-Key') or
                    getattr(g, 'user_id', None) or
                    request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or
                    request.remote_addr
                )

            # レートリミットキー (スコープ + エンドポイント + クライアント)
            key = f"ratelimit:{scope}:{f.__name__}:{hashlib.sha256(client_id.encode()).hexdigest()[:16]}"
            now = time.time()

            # Lua スクリプトでアトミックに処理 (Redis の競合状態を防止)
            lua_script = """
            local key = KEYS[1]
            local now = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local max_requests = tonumber(ARGV[3])

            -- 古いエントリを削除
            redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

            -- 現在のリクエスト数を取得
            local current = redis.call('ZCARD', key)

            if current < max_requests then
                -- 制限内: リクエストを記録
                redis.call('ZADD', key, now, now .. '-' .. math.random(1000000))
                redis.call('EXPIRE', key, window)
                return {current + 1, 0}
            else
                -- 制限超過: 最も古いエントリからリセット時間を計算
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local reset_at = oldest[2] + window
                return {current, reset_at}
            end
            """

            result = r.eval(lua_script, 1, key, now, window_seconds, max_requests)
            current_count = int(result[0])
            reset_at = float(result[1])

            # レスポンスヘッダ (RFC 7231 + draft-ietf-httpapi-ratelimit-headers)
            headers = {
                'X-RateLimit-Limit': str(max_requests),
                'X-RateLimit-Remaining': str(max(0, max_requests - current_count)),
                'X-RateLimit-Reset': str(int(now + window_seconds)),
                'RateLimit-Policy': f'{max_requests};w={window_seconds}',
            }

            if reset_at > 0:
                headers['Retry-After'] = str(int(reset_at - now))
                return jsonify({
                    'error': 'rate_limit_exceeded',
                    'message': f'Rate limit of {max_requests} requests per {window_seconds}s exceeded',
                    'retry_after': int(reset_at - now),
                }), 429, headers

            response = f(*args, **kwargs)
            # Flask のレスポンスにヘッダを追加
            if isinstance(response, tuple):
                body, status = response[0], response[1]
                return body, status, headers
            return response, 200, headers

        return wrapper
    return decorator

# 使用例: エンドポイントごとに異なるレートリミット
@app.route('/api/data')
@rate_limit(max_requests=100, window_seconds=60, scope='general')
def get_data():
    return jsonify({'data': 'ok'})

@app.route('/api/auth/login', methods=['POST'])
@rate_limit(max_requests=5, window_seconds=300, scope='auth')
def login():
    return jsonify({'token': '...'})

@app.route('/api/export', methods=['POST'])
@rate_limit(max_requests=3, window_seconds=3600, scope='expensive')
def export_data():
    return jsonify({'job_id': '...'})
```

### レートリミット戦略の比較

| 戦略 | メモリ | 精度 | 実装複雑度 | バースト対応 | 分散環境 |
|------|--------|------|-----------|------------|---------|
| Fixed Window | 低 | 低 (境界問題) | 低 | 不可 | 容易 |
| Sliding Window Log | 高 | 高 | 中 | 可 | 中 |
| Sliding Window Counter | 中 | 中 | 中 | 可 | 容易 |
| Token Bucket | 低 | 高 | 低 | 可 (バースト許容) | 中 |
| Leaky Bucket | 低 | 高 | 低 | 不可 (平滑化) | 中 |

### 多層防御のレートリミット設計

```
クライアント → CDN/WAF → API Gateway → アプリケーション

Layer 1: CDN/WAF (Cloudflare, AWS WAF)
  - IP ベースのレートリミット: 1000 req/min/IP
  - Geographic ブロック
  - Bot 検出

Layer 2: API Gateway (Kong, AWS API Gateway)
  - API キーベースのレートリミット: 100 req/min/key
  - プランベースの制限 (Free: 100, Pro: 1000, Enterprise: 10000)
  - バースト制限

Layer 3: アプリケーション
  - ユーザベースの細かいレートリミット
  - エンドポイントごとの制限
  - ビジネスロジック固有の制限 (パスワード試行回数等)
```

---

## 5. 入力検証とスキーマバリデーション

### OpenAPI スキーマ定義

```yaml
# OpenAPI 3.0 でのセキュアなスキーマ定義
openapi: '3.0.3'
info:
  title: Secure API
  version: '1.0.0'

paths:
  /api/users:
    post:
      operationId: createUser
      security:
        - bearerAuth: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: User created
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/UserResponse'
        '400':
          $ref: '#/components/responses/ValidationError'
        '401':
          $ref: '#/components/responses/Unauthorized'

components:
  schemas:
    CreateUserRequest:
      type: object
      required: [name, email]
      additionalProperties: false  # 未定義フィールドを拒否
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
          pattern: '^[a-zA-Z\s\-]+$'
          description: 名前 (英字、スペース、ハイフンのみ)
        email:
          type: string
          format: email
          maxLength: 254
          description: メールアドレス (RFC 5321 準拠)
        age:
          type: integer
          minimum: 0
          maximum: 150
          description: 年齢 (0-150)
        bio:
          type: string
          maxLength: 500
          description: 自己紹介 (HTML タグは除去)

    UserResponse:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        email:
          type: string
        createdAt:
          type: string
          format: date-time

  responses:
    ValidationError:
      description: Validation Error
      content:
        application/json:
          schema:
            type: object
            properties:
              error:
                type: string
                example: 'validation_failed'
              details:
                type: array
                items:
                  type: object
                  properties:
                    field:
                      type: string
                    message:
                      type: string

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

### Zod によるバリデーション (TypeScript)

```typescript
import { z } from 'zod';
import { Request, Response, NextFunction } from 'express';
import DOMPurify from 'isomorphic-dompurify';

// スキーマ定義
const CreateUserSchema = z.object({
  name: z
    .string()
    .min(1, 'Name is required')
    .max(100, 'Name must be 100 characters or less')
    .regex(/^[a-zA-Z\s\-]+$/, 'Name must contain only letters, spaces, and hyphens')
    .transform(s => s.trim()),

  email: z
    .string()
    .email('Invalid email address')
    .max(254, 'Email must be 254 characters or less')
    .transform(s => s.toLowerCase().trim()),

  age: z
    .number()
    .int('Age must be an integer')
    .min(0, 'Age must be non-negative')
    .max(150, 'Age must be 150 or less')
    .optional(),

  bio: z
    .string()
    .max(500, 'Bio must be 500 characters or less')
    .transform(s => DOMPurify.sanitize(s, { ALLOWED_TAGS: [] }))  // HTML 除去
    .optional(),
}).strict();  // 未定義フィールドを拒否

// パスパラメータ
const UserIdSchema = z.object({
  id: z.string().uuid('Invalid user ID format'),
});

// クエリパラメータ
const PaginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(['createdAt', 'name', 'email']).default('createdAt'),
  order: z.enum(['asc', 'desc']).default('desc'),
});

// バリデーションミドルウェア
function validate<T extends z.ZodType>(schema: T, source: 'body' | 'params' | 'query' = 'body') {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req[source]);

    if (!result.success) {
      return res.status(400).json({
        error: 'validation_failed',
        details: result.error.issues.map(issue => ({
          field: issue.path.join('.'),
          message: issue.message,
          code: issue.code,
        })),
      });
    }

    // バリデーション済みデータで上書き (transform 済み)
    req[source] = result.data;
    next();
  };
}

// 使用例
app.post('/api/users',
  verifyToken,
  validate(CreateUserSchema, 'body'),
  createUser
);

app.get('/api/users/:id',
  verifyToken,
  validate(UserIdSchema, 'params'),
  getUser
);

app.get('/api/users',
  verifyToken,
  validate(PaginationSchema, 'query'),
  listUsers
);
```

### Express.js での入力検証 (express-validator)

```javascript
const { body, param, query, validationResult } = require('express-validator');
const createDOMPurify = require('dompurify');
const { JSDOM } = require('jsdom');
const DOMPurify = createDOMPurify(new JSDOM('').window);

// バリデーションルール
const createUserValidation = [
  body('name')
    .trim()
    .isLength({ min: 1, max: 100 })
    .matches(/^[a-zA-Z\s\-]+$/)
    .withMessage('Name must contain only letters, spaces, and hyphens'),

  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Valid email is required'),

  body('age')
    .optional()
    .isInt({ min: 0, max: 150 })
    .withMessage('Age must be between 0 and 150'),

  body('bio')
    .optional()
    .isLength({ max: 500 })
    .customSanitizer(value => DOMPurify.sanitize(value, { ALLOWED_TAGS: [] })),
];

// バリデーション結果の処理
function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: 'validation_failed',
      details: errors.array().map(e => ({
        field: e.path,
        message: e.msg,
        value: undefined,  // 入力値を返さない (シークレット漏洩防止)
      })),
    });
  }
  next();
}

app.post('/api/users', createUserValidation, validate, createUser);
```

---

## 6. API セキュリティヘッダと CORS

### セキュリティヘッダの設定

```javascript
const helmet = require('helmet');

// Helmet で基本的なセキュリティヘッダを設定
app.use(helmet());

// CORS の適切な設定
const cors = require('cors');
const allowedOrigins = [
  'https://app.example.com',
  'https://admin.example.com',
];

app.use(cors({
  origin: (origin, callback) => {
    // サーバ間通信 (origin なし) は許可
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) {
      return callback(null, true);
    }
    callback(new Error('Not allowed by CORS'));
  },
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
  exposedHeaders: ['X-RateLimit-Limit', 'X-RateLimit-Remaining', 'X-RateLimit-Reset'],
  credentials: true,
  maxAge: 86400,       // preflight キャッシュ (24時間)
  optionsSuccessStatus: 204,
}));

// 追加のセキュリティヘッダ
app.use((req, res, next) => {
  // レスポンスの MIME タイプを強制
  res.setHeader('X-Content-Type-Options', 'nosniff');
  // クリックジャッキング防止
  res.setHeader('X-Frame-Options', 'DENY');
  // API レスポンスをキャッシュしない
  res.setHeader('Cache-Control', 'no-store, no-cache, must-revalidate, private');
  res.setHeader('Pragma', 'no-cache');
  // API バージョンと廃止情報
  res.setHeader('API-Version', 'v1');
  // リクエストトレーシング
  res.setHeader('X-Request-ID', req.headers['x-request-id'] || crypto.randomUUID());
  next();
});
```

### CORS の動作フロー

```
Simple Request (GET/POST with simple headers):
  Browser → Origin: https://app.example.com → Server
  Server → Access-Control-Allow-Origin: https://app.example.com → Browser

Preflight Request (PUT/DELETE/Custom headers):
  Step 1: OPTIONS (preflight)
  Browser → OPTIONS /api/users
           Origin: https://app.example.com
           Access-Control-Request-Method: PUT
           Access-Control-Request-Headers: Authorization, Content-Type

  Server → 204 No Content
           Access-Control-Allow-Origin: https://app.example.com
           Access-Control-Allow-Methods: GET, POST, PUT, DELETE
           Access-Control-Allow-Headers: Authorization, Content-Type
           Access-Control-Max-Age: 86400

  Step 2: Actual Request
  Browser → PUT /api/users/123
           Origin: https://app.example.com
           Authorization: Bearer ...

  Server → 200 OK
           Access-Control-Allow-Origin: https://app.example.com

NG パターン:
  Access-Control-Allow-Origin: *
  → credentials: true と併用不可
  → 任意のオリジンからアクセス可能 (セキュリティリスク)
```

---

## 7. GraphQL セキュリティ

### GraphQL 固有のリスクと対策

```
GraphQL の脅威モデル:

1. クエリ深度攻撃 (Depth Attack):
   query {
     user {
       posts {
         comments {
           author {
             posts {
               comments { ... }  # 無限にネスト
             }
           }
         }
       }
     }
   }
   → 対策: depth-limit プラグイン (最大深度 = 7-10)

2. クエリ幅攻撃 (Breadth Attack):
   query {
     user1: user(id: "1") { name }
     user2: user(id: "2") { name }
     user3: user(id: "3") { name }
     ... # 数千のエイリアス
   }
   → 対策: クエリコスト解析 + 制限

3. イントロスペクション情報漏洩:
   query {
     __schema {
       types { name fields { name type { name } } }
     }
   }
   → 対策: 本番でイントロスペクションを無効化

4. バッチ攻撃:
   [
     {"query": "mutation { login(email: \"a\", pass: \"1\") { token } }"},
     {"query": "mutation { login(email: \"a\", pass: \"2\") { token } }"},
     ... # 大量のログイン試行
   ]
   → 対策: バッチリクエスト数の制限
```

```javascript
// Apollo Server セキュリティ設定
const { ApolloServer } = require('@apollo/server');
const depthLimit = require('graphql-depth-limit');
const { createComplexityLimitRule } = require('graphql-validation-complexity');

const server = new ApolloServer({
  typeDefs,
  resolvers,
  // イントロスペクションを本番で無効化
  introspection: process.env.NODE_ENV !== 'production',
  // バリデーションルール
  validationRules: [
    // クエリ深度制限 (最大 10)
    depthLimit(10),
    // クエリ複雑度制限
    createComplexityLimitRule(1000, {
      scalarCost: 1,
      objectCost: 2,
      listFactor: 10,
      onCost: (cost) => {
        console.log(`Query cost: ${cost}`);
      },
    }),
  ],
  // エラーフォーマット (内部エラーを隠す)
  formatError: (error) => {
    // スタックトレースを除去
    if (process.env.NODE_ENV === 'production') {
      return {
        message: error.message,
        extensions: {
          code: error.extensions?.code || 'INTERNAL_ERROR',
        },
      };
    }
    return error;
  },
  plugins: [
    // クエリサイズ制限
    {
      async requestDidStart() {
        return {
          async didResolveOperation(ctx) {
            const queryLength = ctx.request.query?.length || 0;
            if (queryLength > 10000) {
              throw new Error('Query too large');
            }
          },
        };
      },
    },
  ],
});
```

---

## 8. API バージョニングとライフサイクル

### バージョニング戦略の比較

| 方式 | 例 | メリット | デメリット |
|------|-----|---------|----------|
| URL パス | `/v1/users` | 明確、キャッシュ容易 | URL が変わる |
| ヘッダ | `Accept: application/vnd.api.v1+json` | URL 不変 | 発見しにくい |
| クエリパラメータ | `/users?version=1` | 簡単 | キャッシュに影響 |
| Content Negotiation | `Accept: application/json; version=1` | RESTful | 実装複雑 |

### API の廃止プロセス

```javascript
// 非推奨 API のヘッダ通知
function deprecateEndpoint(sunsetDate, link) {
  return (req, res, next) => {
    res.setHeader('Deprecation', 'true');
    res.setHeader('Sunset', sunsetDate);  // RFC 8594
    res.setHeader('Link', `<${link}>; rel="successor-version"`);

    // 廃止直前は Warning ヘッダも追加
    const sunset = new Date(sunsetDate);
    const daysUntilSunset = Math.ceil((sunset - new Date()) / (1000 * 60 * 60 * 24));
    if (daysUntilSunset <= 30) {
      res.setHeader('Warning',
        `299 - "This API version will be removed on ${sunsetDate}. Migrate to ${link}"`
      );
    }

    next();
  };
}

// v1 は 2025-06-01 に廃止
app.use('/api/v1',
  deprecateEndpoint('2025-06-01', 'https://api.example.com/v2'),
  v1Router
);
app.use('/api/v2', v2Router);
```

---

## 9. アンチパターン

### アンチパターン 1: JWT の `alg: none` 許可

```javascript
// NG: アルゴリズムを検証しない
const payload = jwt.verify(token, secret);
// 攻撃者が alg: "none" でトークンを偽造 → 署名検証がスキップされる

// NG: HS256 と RS256 の混同
const payload = jwt.verify(token, publicKey);
// 攻撃者が alg を "HS256" に変更し、公開鍵を対称鍵として使用

// OK: 許可するアルゴリズムを明示指定
const payload = jwt.verify(token, publicKey, {
  algorithms: ['RS256'],  // none, HS256 などを拒否
  issuer: 'https://auth.example.com',
  audience: 'https://api.example.com',
});
```

**影響**: 攻撃者が `alg: none` でトークンを偽造し、任意のユーザになりすませる。`alg` 混同攻撃では公開鍵を HMAC の秘密鍵として使い、有効な署名を生成できる。

### アンチパターン 2: エラーレスポンスでの情報漏洩

```javascript
// NG: 内部情報を露出するエラーレスポンス
app.use((err, req, res, next) => {
  res.status(500).json({
    error: err.message,        // "ECONNREFUSED 10.0.1.5:5432" → DB の IP が漏洩
    stack: err.stack,          // ファイルパスが漏洩
    query: err.sql,            // SQL クエリが漏洩
    config: app.get('config'), // 設定情報が漏洩
  });
});

// OK: 安全なエラーレスポンス
app.use((err, req, res, next) => {
  // 内部ログには詳細を記録
  const requestId = req.headers['x-request-id'] || crypto.randomUUID();
  console.error({
    requestId,
    error: err.message,
    stack: err.stack,
    path: req.path,
    method: req.method,
    userId: req.user?.id,
  });

  // クライアントには最小限の情報のみ
  const statusCode = err.statusCode || 500;
  res.status(statusCode).json({
    error: statusCode >= 500 ? 'Internal Server Error' : err.message,
    requestId,  // サポートへの問い合わせ用
  });
});
```

**影響**: 内部 IP、DB スキーマ、ファイルパス、使用ライブラリのバージョンなどが攻撃者に露出し、より標的を絞った攻撃が可能になる。

### アンチパターン 3: レスポンスでのデータ過剰露出

```javascript
// NG: DB のレコードをそのまま返す
app.get('/api/users/:id', async (req, res) => {
  const user = await User.findById(req.params.id);
  res.json(user);  // password_hash, internal_notes, ssn 等も含まれる
});

// OK: レスポンス DTO でフィールドを明示的に選択
function toUserDTO(user) {
  return {
    id: user.id,
    name: user.name,
    email: user.email,
    avatar: user.avatar,
    createdAt: user.createdAt,
    // password_hash, ssn, internal_notes は含めない
  };
}

app.get('/api/users/:id', authenticate, async (req, res) => {
  const user = await User.findById(req.params.id)
    .select('id name email avatar createdAt');  // DB クエリでも制限
  if (!user) return res.status(404).json({ error: 'Not found' });
  res.json(toUserDTO(user));
});
```

---

## 10. エッジケース

### エッジケース 1: JWT の時刻同期問題

サーバ間の時刻がずれている場合、JWT の `exp` や `nbf` の検証が不正確になる。NTP 同期が遅延した場合、有効なトークンが拒否されたり、期限切れトークンが受理される。

```javascript
// 対策: clockTolerance で許容範囲を設定
jwt.verify(token, key, {
  algorithms: ['RS256'],
  clockTolerance: 30,  // 30秒の時刻ずれを許容
});

// 追加対策: iat が未来でないことを確認
if (payload.iat > Math.floor(Date.now() / 1000) + 60) {
  throw new Error('Token issued in the future');
}
```

### エッジケース 2: レートリミットの分散環境での不整合

複数の API サーバが存在する場合、インメモリのレートリミッタではサーバごとにカウントが分散し、実際の制限が緩くなる。

```
サーバ A: count = 50 (limit = 100)
サーバ B: count = 50 (limit = 100)
→ 実際には 100 リクエストが通過 (想定の 100 と一致)

問題: ロードバランサのラウンドロビンが不均一な場合
サーバ A: count = 90 (limit = 100)
サーバ B: count = 10 (limit = 100)
→ サーバ B 経由ならさらに 90 リクエスト可能

対策: Redis などの中央ストアで一元管理
```

### エッジケース 3: Unicode 正規化による入力検証バイパス

Unicode の正規化形式 (NFC, NFD, NFKC, NFKD) の違いを利用して、入力検証をバイパスする攻撃がある。

```javascript
// 例: "admin" の視覚的に同一な Unicode 文字
const normalAdmin = 'admin';        // U+0061, U+0064, U+006D, U+0069, U+006E
const trickAdmin = '\u0430dmin';    // U+0430 (Cyrillic 'a'), rest is Latin

normalAdmin === trickAdmin;  // false
normalAdmin.normalize('NFKC') === trickAdmin.normalize('NFKC');  // false

// 対策: 入力を NFKC 正規化してから検証
function sanitizeUsername(input) {
  const normalized = input.normalize('NFKC');
  // ASCII 以外の文字を含む場合は拒否
  if (!/^[a-zA-Z0-9_\-]+$/.test(normalized)) {
    throw new Error('Username contains invalid characters');
  }
  return normalized;
}
```

---

## 11. パフォーマンス考慮事項

### 認証・認可のパフォーマンス

| 方式 | レイテンシ | スケーラビリティ | ステートレス |
|------|----------|----------------|------------|
| セッション (DB) | ~5-10ms | 低 (DB ボトルネック) | ステートフル |
| セッション (Redis) | ~1-3ms | 中 | ステートフル |
| JWT (HS256) | ~0.1ms | 高 | ステートレス |
| JWT (RS256) | ~0.5-1ms | 高 | ステートレス |
| JWT + JWKS | ~1-3ms (初回) | 高 | ステートレス |
| API Key (DB lookup) | ~3-5ms | 中 | ステートフル |
| API Key (cache) | ~0.5ms | 高 | 準ステートレス |
| mTLS | ~2-5ms (handshake) | 高 | ステートレス |

### API レスポンス最適化

```
+----------------------------------+-------------------+
| 手法                              | 効果              |
+----------------------------------+-------------------+
| ページネーション (cursor-based)    | レスポンスサイズ   |
| フィールド選択 (?fields=id,name)  | レスポンスサイズ   |
| 圧縮 (gzip/br)                   | 転送サイズ 60-80% |
| ETag + 304 Not Modified          | 不要な転送を削減   |
| Connection: keep-alive           | TCP ハンドシェイク |
| HTTP/2 multiplexing              | 並列リクエスト     |
| CDN キャッシュ (public API)       | オリジン負荷      |
+----------------------------------+-------------------+
```

---

## 12. 演習問題

### 演習 1: BOLA 防御の実装 (初級)

以下のエンドポイントに BOLA (Broken Object Level Authorization) 防御を実装しなさい。

```javascript
// 修正対象
app.get('/api/documents/:docId', async (req, res) => {
  const doc = await Document.findById(req.params.docId);
  res.json(doc);
});

app.put('/api/documents/:docId', async (req, res) => {
  const doc = await Document.findByIdAndUpdate(req.params.docId, req.body);
  res.json(doc);
});
```

**要件**:
1. 認証ミドルウェアを追加する
2. ドキュメントの所有者のみが GET/PUT できるようにする
3. 管理者は全ドキュメントにアクセスできる
4. 存在しないドキュメントと権限がないドキュメントを区別しない (情報漏洩防止)
5. Mass Assignment 防御も実装する

### 演習 2: JWT + レートリミット統合 (中級)

以下の要件を満たす API サーバを実装しなさい。

**要件**:
1. JWKS による JWT 検証ミドルウェア
2. スコープベースの認可 (users:read, users:write, admin)
3. Redis ベースのレートリミット (認証ユーザ: 100req/min, 匿名: 10req/min)
4. OpenAPI 3.0 スキーマに準拠した入力検証
5. セキュリティヘッダの設定 (CORS, CSP, etc.)

### 演習 3: GraphQL セキュリティ監査 (上級)

既存の GraphQL API に以下のセキュリティ対策を追加しなさい。

**要件**:
1. クエリ深度制限 (最大 7)
2. クエリコスト解析 (最大コスト 500)
3. イントロスペクション無効化 (本番)
4. バッチリクエスト数の制限 (最大 5)
5. Persisted Queries (ホワイトリスト方式)
6. フィールドレベルの認可 (sensitive フィールドは admin のみ)

---

## 13. トラブルシューティング

| 問題 | 原因 | 解決策 |
|------|------|--------|
| CORS preflight が 403 | OPTIONS メソッドが処理されていない | CORS ミドルウェアを最初に配置 |
| JWT expired エラーが頻発 | トークン有効期限が短すぎる | リフレッシュ戦略を実装 + clockTolerance 設定 |
| レートリミットが効かない | インメモリカウンタ + 複数サーバ | Redis で一元管理 |
| API キーが漏洩した | Git にコミット / ログに出力 | キーローテーション + シークレット管理ツール |
| 429 が返らず 502 が返る | バックエンドが過負荷でクラッシュ | API Gateway でレートリミット |
| GraphQL N+1 問題 | DataLoader 未使用 | DataLoader でバッチ化 |
| SSRF via redirect | URL 検証後にリダイレクト | リダイレクトを無効化 + IP 検証 |
| Cookie が送信されない | SameSite 設定不一致 | SameSite=None + Secure (cross-origin) |

---

## 14. FAQ

### Q1. API キーと OAuth トークンはどう使い分けるか?

API キーはクライアントの識別とレートリミットに使用し、認可の判断には使わない。ユーザに紐づく操作には OAuth 2.0 のアクセストークンを使用する。M2M 通信で細かい認可が不要な場合は Client Credentials フローで取得したトークンを使う。API キーはリクエストヘッダ (`X-API-Key`) で送信し、URL のクエリパラメータには含めない (ログに残るため)。

### Q2. アクセストークンの有効期限はどのくらいが適切か?

アクセストークンは 15 分から 1 時間が一般的である。短いほどセキュリティは向上するが、ユーザ体験とリフレッシュトークンの負荷が増す。リフレッシュトークンは 7-30 日とし、ローテーション (使い捨て) を必須にする。高セキュリティ環境 (金融、医療) ではアクセストークン 5-15 分、リフレッシュトークン 1-7 日が推奨される。

### Q3. API のバージョニングとセキュリティの関係は?

古い API バージョンにはセキュリティパッチが適用されにくいため、サポートするバージョン数を最小限に保つ。非推奨 API にはサンセット期限を設け、`Sunset` ヘッダと `Deprecation` ヘッダで通知する。廃止した API は 410 Gone を返し、移行先 URL を含める。

### Q4. REST と GraphQL でセキュリティの違いは?

REST は URL ベースで認可しやすいが、over-fetching/under-fetching の問題がある。GraphQL は柔軟だが、クエリ深度攻撃、バッチ攻撃、コスト攻撃の固有リスクがある。GraphQL では Persisted Queries (クエリのホワイトリスト) を本番で有効にすることで、任意のクエリ実行を防止できる。

### Q5. API Gateway はどのセキュリティ機能を担当すべきか?

API Gateway は認証 (JWT 検証)、レートリミット、IP ブロック、TLS 終端、リクエストサイズ制限を担当する。認可 (BOLA、BFLA) はビジネスロジックに依存するため、アプリケーション層で実装する。API Gateway に認可を集約すると、ポリシーが複雑化してメンテナンスが困難になる。

---

## まとめ

| 項目 | 要点 |
|------|------|
| BOLA 防御 | オブジェクトレベルの認可チェックを全エンドポイントに実装 |
| OAuth 2.0 | Authorization Code + PKCE を標準採用。Implicit は非推奨 |
| JWT | alg 固定、iss/aud/exp を必ず検証、短い有効期限、JWKS で鍵管理 |
| リフレッシュ | トークンローテーション必須、リプレイ検知、自動リフレッシュ |
| レートリミット | Redis で分散管理、多層防御 (CDN/Gateway/App)、エンドポイント別制限 |
| 入力検証 | Zod/express-validator + OpenAPI スキーマ、ホワイトリスト方式 |
| CORS | 許可オリジンを明示指定、ワイルドカード禁止、credentials に注意 |
| GraphQL | 深度制限 + コスト解析 + イントロスペクション無効化 |
| エラー処理 | 内部情報を露出しない、requestId でトレーシング |
| バージョニング | Sunset/Deprecation ヘッダで段階的廃止 |

---

## 次に読むべきガイド

- [セキュアコーディング](../04-application-security/00-secure-coding.md) -- コードレベルでの攻撃防御
- [ネットワークセキュリティ基礎](./00-network-security-basics.md) -- ネットワーク層の防御
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) -- 通信暗号化の基盤
- [OWASP Top 10](../01-web-security/00-owasp-top10.md) -- Web アプリケーションの脆弱性

---

## 参考文献

1. **OWASP API Security Top 10 (2023)** -- https://owasp.org/API-Security/
2. **RFC 6749 -- The OAuth 2.0 Authorization Framework** -- https://datatracker.ietf.org/doc/html/rfc6749
3. **RFC 7519 -- JSON Web Token (JWT)** -- https://datatracker.ietf.org/doc/html/rfc7519
4. **RFC 7636 -- PKCE (Proof Key for Code Exchange)** -- https://datatracker.ietf.org/doc/html/rfc7636
5. **RFC 8594 -- The Sunset HTTP Header Field** -- https://datatracker.ietf.org/doc/html/rfc8594
6. **RFC 8628 -- OAuth 2.0 Device Authorization Grant** -- https://datatracker.ietf.org/doc/html/rfc8628
7. **draft-ietf-httpapi-ratelimit-headers** -- https://datatracker.ietf.org/doc/draft-ietf-httpapi-ratelimit-headers/
8. **GraphQL Security Cheat Sheet (OWASP)** -- https://cheatsheetseries.owasp.org/cheatsheets/GraphQL_Cheat_Sheet.html
