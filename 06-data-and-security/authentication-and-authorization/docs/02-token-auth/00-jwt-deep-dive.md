# JWT 詳解

> JWT（JSON Web Token）はモダン認証の中核技術。ヘッダー・ペイロード・署名の3部構造、署名アルゴリズムの選択、クレーム設計、検証フロー、セキュリティ上の落とし穴まで、JWTの全てを深掘りする。

## この章で学ぶこと

- [ ] JWT の構造と署名の仕組みを理解する
- [ ] 署名アルゴリズム（HS256/RS256/ES256）の選択基準を把握する
- [ ] JWT のセキュリティリスクと正しい実装を学ぶ

---

## 1. JWT の構造

```
JWT の3部構成:

  eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.     ← ヘッダー
  eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6Ikp...  ← ペイロード
  SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQ...   ← 署名

  [Header].[Payload].[Signature]
  各部はBase64URLエンコード、ドットで連結

① ヘッダー（Header）:
  {
    "alg": "RS256",    ← 署名アルゴリズム
    "typ": "JWT",      ← トークンタイプ
    "kid": "key-id-1"  ← 鍵ID（複数鍵の管理）
  }

② ペイロード（Payload / Claims）:
  {
    "sub": "user_123",           ← Subject（ユーザーID）
    "iss": "https://auth.example.com", ← Issuer（発行者）
    "aud": "https://api.example.com",  ← Audience（対象者）
    "exp": 1700000000,           ← Expiration（有効期限）
    "iat": 1699999100,           ← Issued At（発行時刻）
    "nbf": 1699999100,           ← Not Before（有効開始）
    "jti": "unique-token-id",    ← JWT ID（一意識別子）
    "role": "admin",             ← カスタムクレーム
    "permissions": ["read", "write"]
  }

③ 署名（Signature）:
  RSASHA256(
    base64UrlEncode(header) + "." + base64UrlEncode(payload),
    privateKey
  )

重要:
  ✗ JWT は暗号化ではない（Base64URLデコードで中身が読める）
  ✓ JWT は署名（改ざん検知のみ）
  → 機密情報をペイロードに含めてはいけない
```

---

## 2. 署名アルゴリズム

```
主要アルゴリズムの比較:

  アルゴリズム │ 種類    │ 鍵       │ 推奨度 │ 用途
  ──────────┼────────┼────────┼──────┼──────────
  HS256     │ 対称鍵  │ 共有秘密鍵│ △     │ 単一サービス
  RS256     │ 非対称鍵│ RSA鍵ペア │ ○     │ マイクロサービス
  ES256     │ 非対称鍵│ ECDSA鍵  │ ◎     │ 最も推奨
  PS256     │ 非対称鍵│ RSA-PSS  │ ○     │ RSA の改良版
  EdDSA     │ 非対称鍵│ Ed25519  │ ◎     │ 最新、高速

HS256（HMAC-SHA256）:
  → 対称鍵: 署名と検証に同じ秘密鍵を使用
  → シンプルだが鍵の共有が問題
  → 単一サービスでの使用のみ推奨
  → 秘密鍵が1箇所でも漏洩すると全てが危険

RS256（RSA-SHA256）:
  → 非対称鍵: 秘密鍵で署名、公開鍵で検証
  → 検証側に秘密鍵を渡す必要がない
  → マイクロサービスに最適
  → 鍵サイズが大きい（2048ビット以上）

ES256（ECDSA-P256-SHA256）:
  → 楕円曲線暗号: RSA より小さい鍵で同等の強度
  → 鍵サイズ: 256ビット（RSA の 2048ビットと同等）
  → 署名サイズが小さい → JWT サイズ削減
  → OWASP 推奨
```

```typescript
// ES256 での JWT 実装（jose ライブラリ）
import { SignJWT, jwtVerify, generateKeyPair, exportJWK } from 'jose';

// 鍵ペア生成
const { publicKey, privateKey } = await generateKeyPair('ES256');

// JWT 署名（発行）
async function issueToken(userId: string, role: string): Promise<string> {
  return new SignJWT({
    sub: userId,
    role,
    permissions: getRolePermissions(role),
  })
    .setProtectedHeader({ alg: 'ES256', kid: 'key-2024-01' })
    .setIssuer('https://auth.example.com')
    .setAudience('https://api.example.com')
    .setIssuedAt()
    .setExpirationTime('15m')  // 15分
    .setJti(crypto.randomUUID())
    .sign(privateKey);
}

// JWT 検証
async function verifyToken(token: string) {
  const { payload } = await jwtVerify(token, publicKey, {
    issuer: 'https://auth.example.com',
    audience: 'https://api.example.com',
    algorithms: ['ES256'],  // 許可するアルゴリズムを明示
  });

  return payload;
}
```

---

## 3. クレーム設計

```
登録済みクレーム（RFC 7519）:

  クレーム │ 正式名        │ 説明                │ 必須
  ───────┼──────────────┼───────────────────┼────
  iss    │ Issuer        │ トークン発行者        │ 推奨
  sub    │ Subject       │ ユーザー識別子        │ 推奨
  aud    │ Audience      │ トークン対象者        │ 推奨
  exp    │ Expiration    │ 有効期限（Unix time） │ 必須
  nbf    │ Not Before    │ 有効開始時刻          │ 任意
  iat    │ Issued At     │ 発行時刻             │ 推奨
  jti    │ JWT ID        │ 一意識別子           │ 任意

カスタムクレーム設計:

  ✓ 推奨:
    → role: ユーザーのロール
    → permissions: 権限の配列
    → org_id: 組織ID（マルチテナント）
    → email: メールアドレス（公開情報のみ）

  ✗ 含めてはいけない:
    → パスワード・シークレット
    → クレジットカード情報
    → 個人の機密情報（住所、電話番号等）
    → 大量のデータ（トークンサイズ肥大化）

  サイズの目安:
    → JWT 全体で 4KB 以下を推奨
    → Cookie に保存する場合は特に重要（Cookie上限: ~4KB）
    → ペイロードは必要最小限に
```

```typescript
// クレーム設計例

// アクセストークンのペイロード（最小限）
interface AccessTokenPayload {
  sub: string;         // ユーザーID
  role: 'user' | 'admin' | 'super_admin';
  org_id?: string;     // マルチテナント用
  // exp, iat, iss, aud は jose が自動設定
}

// ID トークンのペイロード（ユーザー情報含む）
interface IDTokenPayload {
  sub: string;
  email: string;
  name: string;
  picture?: string;
  email_verified: boolean;
}

// アクセストークンには最小限の情報のみ含め、
// 詳細なユーザー情報が必要な場合は /userinfo エンドポイントを使用
```

---

## 4. 鍵のローテーション

```
鍵ローテーションの仕組み:

  なぜローテーションが必要か:
  → 鍵の長期使用はリスク（漏洩の可能性が蓄積）
  → 定期的な更新がセキュリティベストプラクティス
  → 漏洩時の影響範囲を限定

  JWKS（JSON Web Key Set）による管理:

  ┌────────────────────────────────────────┐
  │  /.well-known/jwks.json                │
  │                                        │
  │  {                                     │
  │    "keys": [                           │
  │      {                                 │
  │        "kid": "key-2024-02",           │
  │        "kty": "EC",                    │
  │        "crv": "P-256",                 │
  │        "x": "...",                     │
  │        "y": "...",     ← 現在の鍵      │
  │        "use": "sig"                    │
  │      },                                │
  │      {                                 │
  │        "kid": "key-2024-01",           │
  │        "kty": "EC",                    │
  │        "crv": "P-256",                 │
  │        "x": "...",                     │
  │        "y": "...",     ← 旧鍵（検証用） │
  │        "use": "sig"                    │
  │      }                                 │
  │    ]                                   │
  │  }                                     │
  └────────────────────────────────────────┘

  ローテーション手順:
  ① 新しい鍵ペアを生成
  ② JWKS に新しい公開鍵を追加（旧鍵も残す）
  ③ 新しい秘密鍵で署名を開始（kid で区別）
  ④ 旧鍵で署名されたトークンが全て期限切れになるまで待機
  ⑤ JWKS から旧鍵を削除
```

```typescript
// JWKS エンドポイントの実装
import { exportJWK, generateKeyPair } from 'jose';

// 鍵の管理
class KeyManager {
  private keys: Map<string, { publicKey: any; privateKey: any; createdAt: Date }> = new Map();
  private currentKeyId: string = '';

  async initialize() {
    await this.rotateKey();
  }

  async rotateKey() {
    const keyId = `key-${Date.now()}`;
    const { publicKey, privateKey } = await generateKeyPair('ES256');

    this.keys.set(keyId, { publicKey, privateKey, createdAt: new Date() });
    this.currentKeyId = keyId;

    // 古い鍵を削除（2世代前まで保持）
    const keyIds = Array.from(this.keys.keys());
    if (keyIds.length > 2) {
      this.keys.delete(keyIds[0]);
    }
  }

  // 現在の秘密鍵（署名用）
  getCurrentSigningKey() {
    return {
      kid: this.currentKeyId,
      privateKey: this.keys.get(this.currentKeyId)!.privateKey,
    };
  }

  // JWKS（公開鍵セット）
  async getJWKS() {
    const keys = [];
    for (const [kid, { publicKey }] of this.keys) {
      const jwk = await exportJWK(publicKey);
      keys.push({ ...jwk, kid, use: 'sig', alg: 'ES256' });
    }
    return { keys };
  }

  // kid で公開鍵を取得（検証用）
  getPublicKey(kid: string) {
    return this.keys.get(kid)?.publicKey;
  }
}

// JWKS エンドポイント
// GET /.well-known/jwks.json
app.get('/.well-known/jwks.json', async (req, res) => {
  const jwks = await keyManager.getJWKS();
  res.json(jwks);
});
```

---

## 5. JWT のセキュリティリスクと対策

```
JWT の主要なセキュリティリスク:

  ① alg: "none" 攻撃:
     → ヘッダーの alg を "none" に書き換え
     → 署名検証をバイパス
     → 対策: 許可するアルゴリズムを明示的に指定

  ② アルゴリズム混乱攻撃:
     → RS256 の公開鍵を HS256 の秘密鍵として使用
     → 公開鍵は公開情報 → 攻撃者が署名可能
     → 対策: algorithms パラメータで使用アルゴリズムを限定

  ③ 即時失効の困難さ:
     → JWT は有効期限まで有効（サーバーに状態なし）
     → ユーザーがログアウトしてもトークンは有効
     → 対策: 短い有効期限 + ブラックリスト or Token Version

  ④ トークンサイズ:
     → クレームが増えるとサイズ肥大
     → ヘッダーに毎回含まれる
     → 対策: 必要最小限のクレーム

  ⑤ ペイロードの平文:
     → Base64URL は暗号化ではない
     → 誰でもデコード可能
     → 対策: 機密情報を含めない、必要なら JWE を使用
```

```typescript
// 安全な JWT 検証の実装
import { jwtVerify, errors } from 'jose';

async function verifyAccessToken(token: string) {
  try {
    const { payload, protectedHeader } = await jwtVerify(token, publicKey, {
      // 必ず指定すべきオプション
      algorithms: ['ES256'],                    // 許可アルゴリズムを限定
      issuer: 'https://auth.example.com',       // 発行者を検証
      audience: 'https://api.example.com',      // 対象者を検証
      clockTolerance: 30,                       // 時刻の許容誤差（秒）
    });

    // 追加の検証
    if (!payload.sub) throw new Error('Missing subject');
    if (!payload.role) throw new Error('Missing role');

    return {
      userId: payload.sub,
      role: payload.role as string,
      permissions: payload.permissions as string[],
    };
  } catch (error) {
    if (error instanceof errors.JWTExpired) {
      throw new AuthError('Token expired', 'TOKEN_EXPIRED');
    }
    if (error instanceof errors.JWTClaimValidationFailed) {
      throw new AuthError('Invalid token claims', 'INVALID_CLAIMS');
    }
    throw new AuthError('Invalid token', 'INVALID_TOKEN');
  }
}
```

---

## 6. JWT ブラックリスト（失効対策）

```typescript
// Redis を使った JWT ブラックリスト
class TokenBlacklist {
  constructor(private redis: Redis) {}

  // トークンをブラックリストに追加
  async revoke(jti: string, exp: number): Promise<void> {
    const ttl = exp - Math.floor(Date.now() / 1000);
    if (ttl > 0) {
      // トークンの残り有効期限分だけ保持
      await this.redis.setex(`blacklist:${jti}`, ttl, '1');
    }
  }

  // トークンがブラックリストに含まれるか
  async isRevoked(jti: string): Promise<boolean> {
    const result = await this.redis.get(`blacklist:${jti}`);
    return result !== null;
  }
}

// 検証フローに組み込み
async function verifyTokenWithBlacklist(token: string) {
  const payload = await verifyAccessToken(token);

  // ブラックリストチェック
  if (payload.jti && await blacklist.isRevoked(payload.jti)) {
    throw new AuthError('Token has been revoked', 'TOKEN_REVOKED');
  }

  return payload;
}

// ログアウト時
async function logout(token: string) {
  const payload = await verifyAccessToken(token);
  if (payload.jti && payload.exp) {
    await blacklist.revoke(payload.jti, payload.exp);
  }
}
```

---

## まとめ

| 項目 | 推奨 |
|------|------|
| アルゴリズム | ES256（ECDSA） |
| 有効期限 | Access: 15分、Refresh: 7日 |
| クレーム | 必要最小限（sub, role, exp） |
| 検証 | algorithms, issuer, audience を必ず指定 |
| 失効 | 短命トークン + ブラックリスト |
| 鍵管理 | JWKS + 定期ローテーション |

---

## 次に読むべきガイド
→ [[01-oauth2-flows.md]] — OAuth 2.0 フロー

---

## 参考文献
1. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
2. RFC 7517. "JSON Web Key (JWK)." IETF, 2015.
3. Auth0. "JWT Handbook." auth0.com, 2024.
4. OWASP. "JSON Web Token Cheat Sheet." cheatsheetseries.owasp.org, 2024.
