# API セキュリティ

> OAuth 2.0/JWT による認証認可、レートリミットによる過負荷防止、入力検証による攻撃防御まで、API を安全に公開するための包括的ガイド

## この章で学ぶこと

1. **OAuth 2.0 / JWT の正しい実装** — 認可フロー選択、トークンの生成・検証・失効管理
2. **レートリミットと保護** — DDoS・ブルートフォース攻撃に対する API の自己防衛
3. **入力検証とセキュアな設計** — インジェクション防止、スキーマバリデーション、API 設計原則

---

## 1. API の脅威モデル

### OWASP API Security Top 10 (2023)

```
+------------------------------------------------------+
|           OWASP API Security Top 10                   |
|------------------------------------------------------|
|  API1  Broken Object Level Authorization (BOLA)      |
|  API2  Broken Authentication                         |
|  API3  Broken Object Property Level Authorization    |
|  API4  Unrestricted Resource Consumption             |
|  API5  Broken Function Level Authorization           |
|  API6  Unrestricted Access to Sensitive Business Flow|
|  API7  Server Side Request Forgery (SSRF)            |
|  API8  Security Misconfiguration                     |
|  API9  Improper Inventory Management                 |
|  API10 Unsafe Consumption of APIs                    |
+------------------------------------------------------+
```

---

## 2. OAuth 2.0 / OpenID Connect

### 認可フロー選択ガイド

```
アプリケーションの種類は？
  |
  +-- サーバサイド Web アプリ
  |   → Authorization Code Flow (+ PKCE 推奨)
  |
  +-- SPA (Single Page Application)
  |   → Authorization Code Flow + PKCE
  |   (Implicit Flow は非推奨)
  |
  +-- モバイルアプリ / デスクトップ
  |   → Authorization Code Flow + PKCE
  |
  +-- マシン間通信 (M2M)
  |   → Client Credentials Flow
  |
  +-- IoT / 入力制限デバイス
      → Device Authorization Flow
```

### Authorization Code Flow + PKCE

```
Browser/App              Auth Server              Resource Server
    |                         |                         |
    |-- (1) /authorize -----> |                         |
    |   + response_type=code  |                         |
    |   + code_challenge      |                         |
    |   + code_challenge_method=S256                    |
    |                         |                         |
    |  <-- (2) auth code ---- |                         |
    |                         |                         |
    |-- (3) /token ---------->|                         |
    |   + code                |                         |
    |   + code_verifier       |                         |
    |                         |                         |
    |  <-- (4) access_token --|                         |
    |       + refresh_token   |                         |
    |                         |                         |
    |-- (5) API call ---------+-----------------------> |
    |   + Authorization:      |                         |
    |     Bearer <token>      |                         |
    |                         |                         |
    |  <-- (6) Response ------+-----------------------  |
```

### JWT の構造と検証 (Node.js)

```javascript
const jwt = require('jsonwebtoken');
const jwksClient = require('jwks-rsa');

// JWKS クライアント (公開鍵を自動取得)
const client = jwksClient({
  jwksUri: 'https://auth.example.com/.well-known/jwks.json',
  cache: true,
  rateLimit: true,
});

// JWT 検証ミドルウェア
async function verifyToken(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'Token required' });

  try {
    // ヘッダから kid を取得
    const decoded = jwt.decode(token, { complete: true });
    const key = await client.getSigningKey(decoded.header.kid);

    // 署名検証 + クレーム検証
    const payload = jwt.verify(token, key.getPublicKey(), {
      algorithms: ['RS256'],         // アルゴリズムを明示
      issuer: 'https://auth.example.com',
      audience: 'https://api.example.com',
      clockTolerance: 30,            // 時刻ずれ許容 (秒)
    });

    req.user = payload;
    next();
  } catch (err) {
    if (err.name === 'TokenExpiredError') {
      return res.status(401).json({ error: 'Token expired' });
    }
    return res.status(403).json({ error: 'Invalid token' });
  }
}
```

### JWT クレームのベストプラクティス

| クレーム | 必須 | 説明 |
|---------|------|------|
| `iss` | はい | トークン発行者 |
| `sub` | はい | ユーザ/クライアント識別子 |
| `aud` | はい | 対象 API (受信者) |
| `exp` | はい | 有効期限 (短め: 15分-1時間) |
| `iat` | はい | 発行時刻 |
| `jti` | 推奨 | トークン一意識別子 (リプレイ防止) |
| `scope` | 推奨 | 認可スコープ |

---

## 3. レートリミット

### レートリミットのアルゴリズム

```
Token Bucket (トークンバケット):
  +-------------------+
  |  ○ ○ ○ ○ ○ ○ ○   |  バケット容量 = 10
  |  (トークン)        |  補充レート = 1/秒
  +-------------------+
  リクエスト → トークンを1個消費
  トークンなし → 429 Too Many Requests

Sliding Window Log:
  |------ 60秒ウィンドウ ------|
  |  *  *   *  * *  *  *  *   |  リクエスト数 = 8
  |                           |  上限 = 10 → OK
  +---------------------------+

Fixed Window Counter:
  |--- Window 1 ---|--- Window 2 ---|
  |  count = 8     |  count = 3     |
  |  limit = 10    |  limit = 10    |
  +----------------+----------------+
```

### Redis ベースのレートリミッター

```python
import redis
import time
from functools import wraps
from flask import request, jsonify

r = redis.Redis(host='localhost', port=6379)

def rate_limit(max_requests: int, window_seconds: int):
    """スライディングウィンドウ方式のレートリミッター"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # クライアント識別 (API キー or IP)
            client_id = request.headers.get('X-API-Key', request.remote_addr)
            key = f"ratelimit:{f.__name__}:{client_id}"
            now = time.time()

            pipe = r.pipeline()
            # 古いエントリを削除
            pipe.zremrangebyscore(key, 0, now - window_seconds)
            # 現在のリクエストを追加
            pipe.zadd(key, {str(now): now})
            # ウィンドウ内のリクエスト数を取得
            pipe.zcard(key)
            # TTL を設定
            pipe.expire(key, window_seconds)
            results = pipe.execute()

            current_count = results[2]

            # レスポンスヘッダにリミット情報を付与
            headers = {
                'X-RateLimit-Limit': str(max_requests),
                'X-RateLimit-Remaining': str(max(0, max_requests - current_count)),
                'X-RateLimit-Reset': str(int(now + window_seconds)),
            }

            if current_count > max_requests:
                return jsonify({'error': 'Rate limit exceeded'}), 429, headers

            response = f(*args, **kwargs)
            return response, 200, headers

        return wrapper
    return decorator

@app.route('/api/data')
@rate_limit(max_requests=100, window_seconds=60)
def get_data():
    return jsonify({'data': 'ok'})
```

### レートリミット戦略の比較

| 戦略 | メモリ | 精度 | 実装複雑度 | バースト対応 |
|------|--------|------|-----------|------------|
| Fixed Window | 低 | 低 (境界問題) | 低 | 不可 |
| Sliding Window Log | 高 | 高 | 中 | 可 |
| Sliding Window Counter | 中 | 中 | 中 | 可 |
| Token Bucket | 低 | 高 | 低 | 可 (バースト許容) |
| Leaky Bucket | 低 | 高 | 低 | 不可 (平滑化) |

---

## 4. 入力検証

### スキーマバリデーション (OpenAPI)

```yaml
# OpenAPI 3.0 でのスキーマ定義
paths:
  /api/users:
    post:
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [name, email]
              properties:
                name:
                  type: string
                  minLength: 1
                  maxLength: 100
                  pattern: '^[a-zA-Z\s\-]+$'
                email:
                  type: string
                  format: email
                  maxLength: 254
                age:
                  type: integer
                  minimum: 0
                  maximum: 150
              additionalProperties: false  # 未定義フィールドを拒否
```

### Express.js での入力検証

```javascript
const { body, param, query, validationResult } = require('express-validator');

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
];

// バリデーション結果の処理
function validate(req, res, next) {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      error: 'Validation failed',
      details: errors.array().map(e => ({
        field: e.path,
        message: e.msg,
      })),
    });
  }
  next();
}

app.post('/api/users', createUserValidation, validate, createUser);
```

---

## 5. API セキュリティヘッダとCORS

### セキュリティヘッダの設定

```javascript
const helmet = require('helmet');

app.use(helmet());

// CORS の設定
const cors = require('cors');
app.use(cors({
  origin: ['https://app.example.com'],  // 許可するオリジン
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
  maxAge: 86400,  // preflight キャッシュ (24時間)
}));

// 追加のセキュリティヘッダ
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Pragma', 'no-cache');
  next();
});
```

---

## 6. アンチパターン

### アンチパターン 1: BOLA (Broken Object Level Authorization)

```javascript
// NG: オブジェクトIDのみで認可チェックなし
app.get('/api/orders/:orderId', async (req, res) => {
  const order = await Order.findById(req.params.orderId);
  res.json(order);  // 他人の注文も取得できてしまう
});

// OK: オブジェクト所有者の検証
app.get('/api/orders/:orderId', authenticate, async (req, res) => {
  const order = await Order.findOne({
    _id: req.params.orderId,
    userId: req.user.id,  // 認証ユーザのIDで絞り込み
  });
  if (!order) return res.status(404).json({ error: 'Not found' });
  res.json(order);
});
```

### アンチパターン 2: JWT の `alg: none` 許可

```javascript
// NG: アルゴリズムを検証しない
const payload = jwt.verify(token, secret);  // alg:none で署名バイパス

// OK: 許可するアルゴリズムを明示指定
const payload = jwt.verify(token, publicKey, {
  algorithms: ['RS256'],  // none, HS256 などを拒否
});
```

**影響**: 攻撃者が `alg: none` でトークンを偽造し、任意のユーザになりすませる。

---

## 7. FAQ

### Q1. API キーと OAuth トークンはどう使い分けるか?

API キーはクライアントの識別とレートリミットに使用し、認可の判断には使わない。ユーザに紐づく操作には OAuth 2.0 のアクセストークンを使用する。M2M 通信で細かい認可が不要な場合は Client Credentials フローで取得したトークンを使う。

### Q2. アクセストークンの有効期限はどのくらいが適切か?

アクセストークンは 15 分から 1 時間が一般的である。短いほどセキュリティは向上するが、ユーザ体験とリフレッシュトークンの負荷が増す。リフレッシュトークンは 7-30 日とし、ローテーション (使い捨て) を必須にする。

### Q3. API のバージョニングとセキュリティの関係は?

古い API バージョンにはセキュリティパッチが適用されにくいため、サポートするバージョン数を最小限に保つ。非推奨 API にはサンセット期限を設け、`Sunset` ヘッダと `Deprecation` ヘッダで通知する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| OAuth 2.0 | Authorization Code + PKCE を標準採用 |
| JWT | alg 固定、iss/aud/exp を必ず検証、短い有効期限 |
| レートリミット | Token Bucket or Sliding Window で API を保護 |
| 入力検証 | スキーマバリデーション + ホワイトリスト方式 |
| BOLA 対策 | オブジェクトレベルの認可チェックを必ず実装 |
| CORS | 許可オリジンを明示指定、ワイルドカード禁止 |

---

## 次に読むべきガイド

- [セキュアコーディング](../04-application-security/00-secure-coding.md) — コードレベルでの攻撃防御
- [ネットワークセキュリティ基礎](./00-network-security-basics.md) — ネットワーク層の防御
- [TLS/証明書](../02-cryptography/01-tls-certificates.md) — 通信暗号化の基盤

---

## 参考文献

1. **OWASP API Security Top 10 (2023)** — https://owasp.org/API-Security/
2. **RFC 6749 — The OAuth 2.0 Authorization Framework** — https://datatracker.ietf.org/doc/html/rfc6749
3. **RFC 7519 — JSON Web Token (JWT)** — https://datatracker.ietf.org/doc/html/rfc7519
4. **RFC 9110 — HTTP Semantics (Rate Limiting Headers)** — https://datatracker.ietf.org/doc/html/rfc9110
