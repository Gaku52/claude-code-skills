# JWT 詳解

> JWT（JSON Web Token）はモダン認証の中核技術。ヘッダー・ペイロード・署名の3部構造、署名アルゴリズムの選択、クレーム設計、検証フロー、セキュリティ上の落とし穴まで、JWTの全てを深掘りする。RFC 7519 の仕様に基づき、内部実装レベルの理解から実運用のベストプラクティスまでを網羅する。

## 前提知識

- [[00-authentication-vs-authorization.md]] — 認証と認可の基礎
- [[01-password-security.md]] — パスワードセキュリティ
- Base64URL エンコーディングの基本概念
- 公開鍵暗号方式の基礎（RSA、楕円曲線暗号）
- HTTP ヘッダーと Cookie の基本

## この章で学ぶこと

- [ ] JWT の構造と署名の仕組みを理解する
- [ ] 署名アルゴリズム（HS256/RS256/ES256/EdDSA）の選択基準を把握する
- [ ] JWT のセキュリティリスクと正しい実装を学ぶ
- [ ] 鍵ローテーションと JWKS の運用方法を理解する
- [ ] アクセストークンとリフレッシュトークンの設計パターンを習得する
- [ ] JWE（暗号化）と JWS（署名）の違いと使い分けを理解する

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

### 1.1 Base64URL エンコーディングの詳細

```
Base64 と Base64URL の違い:

  標準 Base64:
    → 文字セット: A-Z, a-z, 0-9, +, /
    → パディング: =
    → URL中で問題: + → %2B, / → %2F, = → %3D

  Base64URL（RFC 4648 §5）:
    → 文字セット: A-Z, a-z, 0-9, -, _
    → パディング: 省略
    → URL セーフ: そのまま使用可能

  変換手順:
    ① JSON オブジェクトを UTF-8 バイト列に変換
    ② 標準 Base64 でエンコード
    ③ + を - に、/ を _ に置換
    ④ 末尾の = を除去
```

```typescript
// Base64URL エンコード/デコードの実装

function base64UrlEncode(input: string | Uint8Array): string {
  let bytes: Uint8Array;
  if (typeof input === 'string') {
    bytes = new TextEncoder().encode(input);
  } else {
    bytes = input;
  }

  // 標準 Base64 にエンコード
  const base64 = btoa(String.fromCharCode(...bytes));

  // Base64URL に変換: + → -, / → _, = 除去
  return base64
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '');
}

function base64UrlDecode(input: string): Uint8Array {
  // Base64URL → 標準 Base64 に復元
  let base64 = input
    .replace(/-/g, '+')
    .replace(/_/g, '/');

  // パディングを追加
  while (base64.length % 4 !== 0) {
    base64 += '=';
  }

  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

// JWT を手動でデコード（署名検証なし、デバッグ用）
function decodeJwtPayload(token: string): Record<string, unknown> {
  const parts = token.split('.');
  if (parts.length !== 3) {
    throw new Error('Invalid JWT format: expected 3 parts separated by dots');
  }

  const payloadBytes = base64UrlDecode(parts[1]);
  const payloadJson = new TextDecoder().decode(payloadBytes);
  return JSON.parse(payloadJson);
}
```

### 1.2 JWT の内部生成プロセス

```
JWT 生成の内部フロー:

  Step 1: ヘッダー構築
  ┌─────────────────────────┐
  │ { "alg": "ES256",       │
  │   "typ": "JWT",         │
  │   "kid": "key-2024-01"} │
  └──────────┬──────────────┘
             ↓
  Step 2: Base64URL エンコード
  ┌─────────────────────────┐
  │ eyJhbGciOiJFUzI1NiIs...│
  └──────────┬──────────────┘
             ↓
  Step 3: ペイロード構築
  ┌─────────────────────────┐
  │ { "sub": "user_123",    │
  │   "exp": 1700000000,    │
  │   "iat": 1699999100 }   │
  └──────────┬──────────────┘
             ↓
  Step 4: Base64URL エンコード
  ┌─────────────────────────┐
  │ eyJzdWIiOiJ1c2VyXzEy... │
  └──────────┬──────────────┘
             ↓
  Step 5: 署名入力の構築
  ┌───────────────────────────────────┐
  │ base64Header + "." + base64Payload│
  │ "eyJhbGci...IkpXVCJ9.eyJzdWIi..." │
  └──────────┬────────────────────────┘
             ↓
  Step 6: 秘密鍵で署名
  ┌───────────────────────────────────┐
  │ ECDSA-P256-SHA256(               │
  │   signingInput,                  │
  │   privateKey                     │
  │ )                                │
  └──────────┬────────────────────────┘
             ↓
  Step 7: 署名を Base64URL エンコード
  ┌───────────────────────────────────┐
  │ SflKxwRJSMeKKF2QT4fwpMeJf36PO... │
  └──────────┬────────────────────────┘
             ↓
  Step 8: 連結して JWT 完成
  ┌───────────────────────────────────┐
  │ header.payload.signature          │
  └───────────────────────────────────┘
```

---

## 2. 署名アルゴリズム

```
主要アルゴリズムの比較:

  アルゴリズム │ 種類    │ 鍵       │ 推奨度 │ 用途
  ──────────┼────────┼────────┼──────┼──────────
  HS256     │ 対称鍵  │ 共有秘密鍵│ △     │ 単一サービス
  HS384     │ 対称鍵  │ 共有秘密鍵│ △     │ 単一サービス
  HS512     │ 対称鍵  │ 共有秘密鍵│ △     │ 単一サービス
  RS256     │ 非対称鍵│ RSA鍵ペア │ ○     │ マイクロサービス
  RS384     │ 非対称鍵│ RSA鍵ペア │ ○     │ マイクロサービス
  RS512     │ 非対称鍵│ RSA鍵ペア │ ○     │ マイクロサービス
  ES256     │ 非対称鍵│ ECDSA鍵  │ ◎     │ 最も推奨
  ES384     │ 非対称鍵│ ECDSA鍵  │ ◎     │ 高セキュリティ
  ES512     │ 非対称鍵│ ECDSA鍵  │ ◎     │ 最高セキュリティ
  PS256     │ 非対称鍵│ RSA-PSS  │ ○     │ RSA の改良版
  PS384     │ 非対称鍵│ RSA-PSS  │ ○     │ RSA の改良版
  PS512     │ 非対称鍵│ RSA-PSS  │ ○     │ RSA の改良版
  EdDSA     │ 非対称鍵│ Ed25519  │ ◎     │ 最新、高速

HS256（HMAC-SHA256）:
  → 対称鍵: 署名と検証に同じ秘密鍵を使用
  → シンプルだが鍵の共有が問題
  → 単一サービスでの使用のみ推奨
  → 秘密鍵が1箇所でも漏洩すると全てが危険
  → 鍵長: 最低256ビット（32バイト）推奨

RS256（RSA-SHA256）:
  → 非対称鍵: 秘密鍵で署名、公開鍵で検証
  → 検証側に秘密鍵を渡す必要がない
  → マイクロサービスに最適
  → 鍵サイズが大きい（2048ビット以上）
  → 署名サイズ: 256バイト（2048ビット鍵の場合）

ES256（ECDSA-P256-SHA256）:
  → 楕円曲線暗号: RSA より小さい鍵で同等の強度
  → 鍵サイズ: 256ビット（RSA の 2048ビットと同等）
  → 署名サイズが小さい → JWT サイズ削減
  → OWASP 推奨
  → 署名サイズ: 64バイト（RSA の 1/4）

EdDSA（Ed25519）:
  → Twisted Edwards 曲線ベース
  → ECDSA より高速（署名・検証とも）
  → 決定論的署名（同じ入力 → 同じ署名）
  → サイドチャネル攻撃への耐性が高い
  → 鍵サイズ: 256ビット、署名サイズ: 64バイト
```

### 2.1 アルゴリズムの内部動作

```
HMAC-SHA256（HS256）の内部:

  ┌──────────────────────────────────────────┐
  │                                          │
  │  secret_key = "super-secret-key-256bit"  │
  │                                          │
  │  署名手順:                                │
  │  ① ipad = secret_key XOR 0x36（64バイト） │
  │  ② opad = secret_key XOR 0x5C（64バイト） │
  │  ③ inner = SHA256(ipad || message)        │
  │  ④ signature = SHA256(opad || inner)      │
  │                                          │
  │  結果: 256ビット（32バイト）のMAC値          │
  │                                          │
  └──────────────────────────────────────────┘

RSA-SHA256（RS256）の内部:

  ┌──────────────────────────────────────────┐
  │                                          │
  │  署名（秘密鍵で）:                         │
  │  ① hash = SHA256(header.payload)         │
  │  ② padded = PKCS#1 v1.5 padding(hash)   │
  │  ③ signature = padded^d mod n            │
  │    （d: 秘密指数, n: モジュラス）           │
  │                                          │
  │  検証（公開鍵で）:                         │
  │  ① decrypted = signature^e mod n         │
  │    （e: 公開指数, n: モジュラス）           │
  │  ② unpadded = remove PKCS#1 padding      │
  │  ③ hash = SHA256(header.payload)         │
  │  ④ compare(unpadded, hash)               │
  │                                          │
  └──────────────────────────────────────────┘

ECDSA-P256-SHA256（ES256）の内部:

  ┌──────────────────────────────────────────┐
  │                                          │
  │  楕円曲線パラメータ（P-256 / secp256r1）:  │
  │  → 素数位数: p ≈ 2^256                    │
  │  → 生成点: G（曲線上の固定点）              │
  │  → 秘密鍵: d（ランダムな整数）             │
  │  → 公開鍵: Q = d × G（スカラー倍算）       │
  │                                          │
  │  署名手順:                                │
  │  ① hash = SHA256(header.payload)         │
  │  ② k = ランダムなノンス（署名ごとに一意）   │
  │  ③ (x, y) = k × G                       │
  │  ④ r = x mod n                           │
  │  ⑤ s = k^(-1) × (hash + r × d) mod n    │
  │  ⑥ signature = (r, s)                    │
  │                                          │
  │  検証手順:                                │
  │  ① hash = SHA256(header.payload)         │
  │  ② w = s^(-1) mod n                      │
  │  ③ u1 = hash × w mod n                   │
  │  ④ u2 = r × w mod n                      │
  │  ⑤ (x', y') = u1 × G + u2 × Q           │
  │  ⑥ valid = (x' mod n == r)               │
  │                                          │
  └──────────────────────────────────────────┘
```

### 2.2 署名サイズとパフォーマンスの比較

```
署名サイズの比較:

  アルゴリズム │ 鍵サイズ    │ 署名サイズ │ セキュリティ強度
  ──────────┼───────────┼──────────┼──────────────
  HS256     │ 256 bit   │ 32 bytes │ 128 bit
  RS256     │ 2048 bit  │ 256 bytes│ 112 bit
  RS256     │ 4096 bit  │ 512 bytes│ 140 bit
  ES256     │ 256 bit   │ 64 bytes │ 128 bit
  ES384     │ 384 bit   │ 96 bytes │ 192 bit
  EdDSA     │ 256 bit   │ 64 bytes │ 128 bit

パフォーマンス比較（相対値、Node.js 20 / x86_64）:

  アルゴリズム │ 署名速度    │ 検証速度    │ JWT サイズ
  ──────────┼───────────┼───────────┼──────────
  HS256     │ ◎ 最速    │ ◎ 最速    │ 小
  RS256     │ △ 遅い    │ ○ 速い    │ 大
  ES256     │ ○ 中程度  │ ○ 中程度  │ 小
  EdDSA     │ ◎ 高速    │ ◎ 高速    │ 小

  結論:
  → パフォーマンスのみなら HS256 だが、鍵共有の問題がある
  → バランスが最良: ES256（小さい鍵+署名、十分高速）
  → 最新の選択: EdDSA（最高速+最小、ただし互換性の確認が必要）
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

```typescript
// EdDSA（Ed25519）での JWT 実装
import { SignJWT, jwtVerify, generateKeyPair } from 'jose';

// Ed25519 鍵ペア生成
const { publicKey, privateKey } = await generateKeyPair('EdDSA', {
  crv: 'Ed25519',
});

async function issueEdDSAToken(userId: string): Promise<string> {
  return new SignJWT({
    sub: userId,
    scope: 'read write',
  })
    .setProtectedHeader({ alg: 'EdDSA', crv: 'Ed25519', kid: 'ed-key-01' })
    .setIssuer('https://auth.example.com')
    .setAudience('https://api.example.com')
    .setIssuedAt()
    .setExpirationTime('15m')
    .setJti(crypto.randomUUID())
    .sign(privateKey);
}

async function verifyEdDSAToken(token: string) {
  const { payload, protectedHeader } = await jwtVerify(token, publicKey, {
    algorithms: ['EdDSA'],
    issuer: 'https://auth.example.com',
    audience: 'https://api.example.com',
  });

  console.log('Algorithm:', protectedHeader.alg);  // EdDSA
  console.log('User:', payload.sub);

  return payload;
}
```

```typescript
// 各アルゴリズムの鍵生成コマンド（OpenSSL）
// HS256: openssl rand -base64 32
// RS256: openssl genrsa -out private.pem 2048
//        openssl rsa -in private.pem -pubout -out public.pem
// ES256: openssl ecparam -genkey -name prime256v1 -noout -out private-ec.pem
//        openssl ec -in private-ec.pem -pubout -out public-ec.pem
// EdDSA: openssl genpkey -algorithm ED25519 -out private-ed.pem
//        openssl pkey -in private-ed.pem -pubout -out public-ed.pem

// Node.js での鍵生成
import { generateKeyPair, exportJWK, exportPKCS8, exportSPKI } from 'jose';

async function generateKeys(algorithm: string) {
  const { publicKey, privateKey } = await generateKeyPair(algorithm);

  // JWK 形式でエクスポート
  const publicJWK = await exportJWK(publicKey);
  const privateJWK = await exportJWK(privateKey);

  // PEM 形式でエクスポート
  const publicPEM = await exportSPKI(publicKey);
  const privatePEM = await exportPKCS8(privateKey);

  return { publicJWK, privateJWK, publicPEM, privatePEM };
}

// 使用例
const es256Keys = await generateKeys('ES256');
const eddsaKeys = await generateKeys('EdDSA');
const rs256Keys = await generateKeys('RS256');
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
    → tenant: テナント識別子
    → token_version: トークンバージョン（強制失効用）

  ✗ 含めてはいけない:
    → パスワード・シークレット
    → クレジットカード情報
    → 個人の機密情報（住所、電話番号等）
    → 大量のデータ（トークンサイズ肥大化）
    → セッション状態（JWT はステートレスであるべき）

  サイズの目安:
    → JWT 全体で 4KB 以下を推奨
    → Cookie に保存する場合は特に重要（Cookie上限: ~4KB）
    → ペイロードは必要最小限に
    → Authorization ヘッダーの一般的な上限: 8KB
    → nginx デフォルト: large_client_header_buffers 8KB
```

### 3.1 クレーム設計のパターン

```typescript
// クレーム設計例

// アクセストークンのペイロード（最小限）
interface AccessTokenPayload {
  sub: string;         // ユーザーID
  role: 'user' | 'admin' | 'super_admin';
  org_id?: string;     // マルチテナント用
  scope?: string;      // OAuth 2.0 スコープ
  token_version?: number; // 強制失効用
  // exp, iat, iss, aud は jose が自動設定
}

// ID トークンのペイロード（ユーザー情報含む）
interface IDTokenPayload {
  sub: string;
  email: string;
  name: string;
  picture?: string;
  email_verified: boolean;
  locale?: string;
  updated_at?: number;
}

// リフレッシュトークンのペイロード
interface RefreshTokenPayload {
  sub: string;
  jti: string;         // 一意ID（失効管理用）
  family: string;      // トークンファミリー（リプレイ検知用）
  token_version: number;
}

// マルチテナント向けアクセストークン
interface MultiTenantAccessToken {
  sub: string;
  org_id: string;
  org_role: 'owner' | 'admin' | 'member' | 'viewer';
  permissions: string[];  // 細粒度の権限
  features: string[];     // テナントの有効機能
}

// アクセストークンには最小限の情報のみ含め、
// 詳細なユーザー情報が必要な場合は /userinfo エンドポイントを使用
```

### 3.2 aud クレームの詳細設計

```
aud（Audience）クレームの重要性:

  目的:
  → トークンが意図した受信者のみに受け入れられるようにする
  → 異なるサービス間でのトークン流用を防止

  よくある問題:
  ┌────────────────────────────────────────────────────┐
  │  aud を検証しない場合:                                │
  │                                                    │
  │  User → AuthServer → access_token(aud: "api-a")   │
  │  User → API-A: access_token → OK                   │
  │  User → API-B: access_token → aud 未検証 → OK ✗    │
  │                                                    │
  │  → API-A 向けトークンが API-B でも使えてしまう         │
  │  → 権限昇格の脆弱性                                  │
  └────────────────────────────────────────────────────┘

  正しい設計:
  ┌────────────────────────────────────────────────────┐
  │  API-A: aud: "https://api-a.example.com" のみ受理   │
  │  API-B: aud: "https://api-b.example.com" のみ受理   │
  │                                                    │
  │  → サービスごとに異なる aud を設定                     │
  │  → 各サービスは自分の aud のトークンのみ検証            │
  └────────────────────────────────────────────────────┘
```

```typescript
// aud クレームの厳密な検証
import { jwtVerify } from 'jose';

// マイクロサービスごとの検証設定
const serviceConfig = {
  'user-service': {
    audience: 'https://user-api.example.com',
    requiredScopes: ['read:user', 'write:user'],
  },
  'order-service': {
    audience: 'https://order-api.example.com',
    requiredScopes: ['read:order', 'write:order'],
  },
  'payment-service': {
    audience: 'https://payment-api.example.com',
    requiredScopes: ['process:payment'],
  },
};

async function verifyServiceToken(
  token: string,
  serviceName: keyof typeof serviceConfig
) {
  const config = serviceConfig[serviceName];

  const { payload } = await jwtVerify(token, publicKey, {
    algorithms: ['ES256'],
    issuer: 'https://auth.example.com',
    audience: config.audience,  // このサービス専用の aud
  });

  // スコープ検証
  const tokenScopes = (payload.scope as string || '').split(' ');
  const hasRequiredScopes = config.requiredScopes.every(
    (scope) => tokenScopes.includes(scope)
  );

  if (!hasRequiredScopes) {
    throw new Error(`Insufficient scopes. Required: ${config.requiredScopes.join(', ')}`);
  }

  return payload;
}
```

---

## 4. 鍵のローテーション

```
鍵ローテーションの仕組み:

  なぜローテーションが必要か:
  → 鍵の長期使用はリスク（漏洩の可能性が蓄積）
  → 定期的な更新がセキュリティベストプラクティス
  → 漏洩時の影響範囲を限定
  → コンプライアンス要件（PCI DSS等）で義務化される場合がある

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

  タイムライン例:

  ──────────────────────────────────────────────>
  │           │              │              │
  key-1 生成  key-2 生成      key-1 期限切れ   key-1 削除
  署名開始    署名を key-2 へ  検証のみ       JWKS から除去
              │              │              │
  ←─key-1 で署名─→←─key-2 で署名───────────────→
  ←─key-1 で検証──────────────→
              ←─key-2 で検証────────────────────→
```

### 4.1 鍵管理の実装

```typescript
// JWKS エンドポイントの実装
import { exportJWK, generateKeyPair, importJWK } from 'jose';

// 鍵の管理
class KeyManager {
  private keys: Map<string, {
    publicKey: CryptoKey;
    privateKey: CryptoKey;
    createdAt: Date;
    expiresAt: Date;
  }> = new Map();
  private currentKeyId: string = '';
  private rotationIntervalMs: number = 30 * 24 * 60 * 60 * 1000; // 30日
  private gracePeriodMs: number = 24 * 60 * 60 * 1000; // 24時間の猶予

  async initialize() {
    await this.rotateKey();
    // 定期的なローテーションをスケジュール
    setInterval(() => this.rotateKey(), this.rotationIntervalMs);
  }

  async rotateKey() {
    const keyId = `key-${Date.now()}`;
    const { publicKey, privateKey } = await generateKeyPair('ES256');
    const now = new Date();

    this.keys.set(keyId, {
      publicKey,
      privateKey,
      createdAt: now,
      expiresAt: new Date(now.getTime() + this.rotationIntervalMs + this.gracePeriodMs),
    });
    this.currentKeyId = keyId;

    // 期限切れの鍵を削除
    this.cleanupExpiredKeys();

    console.log(`Key rotated. Current key: ${keyId}. Total keys: ${this.keys.size}`);
  }

  private cleanupExpiredKeys() {
    const now = new Date();
    for (const [kid, keyInfo] of this.keys) {
      if (kid !== this.currentKeyId && keyInfo.expiresAt < now) {
        this.keys.delete(kid);
        console.log(`Expired key removed: ${kid}`);
      }
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

  // JWKS はキャッシュ可能（ただし短めの TTL）
  res.set('Cache-Control', 'public, max-age=900'); // 15分
  res.json(jwks);
});
```

### 4.2 リモート JWKS の検証

```typescript
// リモート JWKS を使った JWT 検証（マイクロサービス側）
import { createRemoteJWKSet, jwtVerify } from 'jose';

// JWKS URI からの公開鍵取得（キャッシュ付き）
const JWKS = createRemoteJWKSet(
  new URL('https://auth.example.com/.well-known/jwks.json'),
  {
    cooldownDuration: 30_000,  // 最小再取得間隔: 30秒
    cacheMaxAge: 600_000,      // キャッシュ TTL: 10分
  }
);

async function verifyTokenWithRemoteJWKS(token: string) {
  try {
    const { payload, protectedHeader } = await jwtVerify(token, JWKS, {
      algorithms: ['ES256'],
      issuer: 'https://auth.example.com',
      audience: 'https://api.example.com',
      clockTolerance: 30,  // 30秒のクロックスキュー許容
    });

    console.log(`Token verified with key: ${protectedHeader.kid}`);
    return payload;
  } catch (error) {
    // JWKS 取得失敗時のフォールバック
    if (error instanceof TypeError && error.message.includes('fetch')) {
      console.error('JWKS endpoint unreachable. Using cached keys.');
      // キャッシュされた鍵で再試行（jose が自動で行う）
      throw new Error('Authentication service unavailable');
    }
    throw error;
  }
}
```

---

## 5. JWT のセキュリティリスクと対策

```
JWT の主要なセキュリティリスク:

  ① alg: "none" 攻撃:
     → ヘッダーの alg を "none" に書き換え
     → 署名検証をバイパス
     → 対策: 許可するアルゴリズムを明示的に指定

     攻撃の流れ:
     ┌──────────────────────────────────────┐
     │  攻撃者の行動:                         │
     │  1. 正規の JWT を取得                  │
     │  2. ヘッダーの alg を "none" に変更    │
     │  3. ペイロードの role を "admin" に変更 │
     │  4. 署名部分を空にする                 │
     │  5. 改ざんした JWT を送信              │
     │                                      │
     │  脆弱なサーバー:                       │
     │  → alg: "none" を許可してしまう        │
     │  → 署名検証をスキップ                  │
     │  → 攻撃者が管理者としてアクセス         │
     └──────────────────────────────────────┘

  ② アルゴリズム混乱攻撃（Algorithm Confusion）:
     → RS256 の公開鍵を HS256 の秘密鍵として使用
     → 公開鍵は公開情報 → 攻撃者が署名可能
     → 対策: algorithms パラメータで使用アルゴリズムを限定

     攻撃の流れ:
     ┌──────────────────────────────────────┐
     │  前提: サーバーは RS256 を使用          │
     │  公開鍵は JWKS で公開されている         │
     │                                      │
     │  攻撃者の行動:                         │
     │  1. 公開鍵を JWKS から取得             │
     │  2. alg を HS256 に変更               │
     │  3. 公開鍵を HS256 の秘密鍵として使用   │
     │  4. HMAC(公開鍵, header.payload)      │
     │                                      │
     │  脆弱なサーバー:                       │
     │  → alg: HS256 に従って検証            │
     │  → 公開鍵を HMAC の秘密鍵として使用    │
     │  → 攻撃者の署名が一致 → 認証成功       │
     └──────────────────────────────────────┘

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

  ⑥ kid インジェクション:
     → kid フィールドにパストラバーサルやSQLインジェクション
     → kid: "../../dev/null" → 空の鍵で検証
     → kid: "' OR '1'='1" → SQLインジェクション
     → 対策: kid をバリデーション、許可された鍵IDのみ受理

  ⑦ jwk ヘッダーインジェクション:
     → ヘッダーに jwk（公開鍵）を埋め込み
     → サーバーが埋め込まれた鍵で検証してしまう
     → 対策: ヘッダーの jwk を無視、サーバー側の鍵のみ使用
```

### 5.1 安全な JWT 検証の実装

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
      maxTokenAge: '15m',                       // 最大有効期間
    });

    // 追加の検証
    if (!payload.sub) throw new Error('Missing subject');
    if (!payload.role) throw new Error('Missing role');

    // kid の検証（許可された鍵IDのみ）
    if (protectedHeader.kid && !isValidKeyId(protectedHeader.kid)) {
      throw new Error('Invalid key ID');
    }

    // jti の一意性検証（リプレイ攻撃防止）
    if (payload.jti) {
      const isUsed = await checkJtiUsed(payload.jti as string);
      if (isUsed) throw new Error('Token already used');
      await markJtiUsed(payload.jti as string, payload.exp as number);
    }

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
    if (error instanceof errors.JWSSignatureVerificationFailed) {
      throw new AuthError('Invalid signature', 'INVALID_SIGNATURE');
    }
    throw new AuthError('Invalid token', 'INVALID_TOKEN');
  }
}

// kid バリデーション
function isValidKeyId(kid: string): boolean {
  // 英数字とハイフンのみ許可（インジェクション防止）
  return /^[a-zA-Z0-9_-]{1,64}$/.test(kid);
}

// jti 使用済みチェック（Redis）
async function checkJtiUsed(jti: string): Promise<boolean> {
  const exists = await redis.exists(`jti:${jti}`);
  return exists === 1;
}

async function markJtiUsed(jti: string, exp: number): Promise<void> {
  const ttl = exp - Math.floor(Date.now() / 1000);
  if (ttl > 0) {
    await redis.setex(`jti:${jti}`, ttl, '1');
  }
}
```

### 5.2 Express / Koa ミドルウェアの実装

```typescript
// Express 用 JWT 認証ミドルウェア
import { Request, Response, NextFunction } from 'express';
import { jwtVerify, createRemoteJWKSet, errors } from 'jose';

interface AuthenticatedRequest extends Request {
  user?: {
    userId: string;
    role: string;
    permissions: string[];
    orgId?: string;
  };
}

const JWKS = createRemoteJWKSet(
  new URL('https://auth.example.com/.well-known/jwks.json')
);

// 認証ミドルウェア
function requireAuth() {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const authHeader = req.headers.authorization;

    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({
        error: 'authentication_required',
        message: 'Bearer token is required',
      });
    }

    const token = authHeader.substring(7);

    try {
      const { payload } = await jwtVerify(token, JWKS, {
        algorithms: ['ES256'],
        issuer: 'https://auth.example.com',
        audience: 'https://api.example.com',
        clockTolerance: 30,
      });

      req.user = {
        userId: payload.sub as string,
        role: payload.role as string,
        permissions: (payload.permissions as string[]) || [],
        orgId: payload.org_id as string | undefined,
      };

      next();
    } catch (error) {
      if (error instanceof errors.JWTExpired) {
        return res.status(401).json({
          error: 'token_expired',
          message: 'Access token has expired',
        });
      }
      return res.status(401).json({
        error: 'invalid_token',
        message: 'Invalid access token',
      });
    }
  };
}

// 認可ミドルウェア
function requirePermission(...requiredPermissions: string[]) {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    const hasPermission = requiredPermissions.every(
      (perm) => req.user!.permissions.includes(perm)
    );

    if (!hasPermission) {
      return res.status(403).json({
        error: 'insufficient_permissions',
        message: `Required permissions: ${requiredPermissions.join(', ')}`,
      });
    }

    next();
  };
}

// 使用例
app.get('/api/users',
  requireAuth(),
  requirePermission('read:user'),
  async (req: AuthenticatedRequest, res) => {
    const users = await userService.listUsers(req.user!.orgId);
    res.json(users);
  }
);

app.delete('/api/users/:id',
  requireAuth(),
  requirePermission('delete:user'),
  async (req: AuthenticatedRequest, res) => {
    await userService.deleteUser(req.params.id, req.user!.userId);
    res.status(204).send();
  }
);
```

---

## 6. アクセストークンとリフレッシュトークン

```
トークンペア設計:

  アクセストークン:
    → 短命（15分〜1時間）
    → API リクエストの認証に使用
    → JWT（自己完結型、サーバー状態不要）
    → 毎リクエストで送信 → サイズを小さく

  リフレッシュトークン:
    → 長命（7日〜30日）
    → 新しいアクセストークンの取得に使用
    → 不透明トークン or JWT
    → サーバー側で状態管理（失効可能）

  フロー:

  クライアント          認可サーバー           リソースサーバー
    │                    │                     │
    │ ログイン            │                     │
    │───────────────────>│                     │
    │                    │                     │
    │ access_token (15m) │                     │
    │ refresh_token (7d) │                     │
    │<───────────────────│                     │
    │                    │                     │
    │ API リクエスト + access_token             │
    │──────────────────────────────────────────>│
    │                    │                     │
    │ 200 OK + データ                           │
    │<──────────────────────────────────────────│
    │                    │                     │
    │ （15分後）access_token 期限切れ             │
    │──────────────────────────────────────────>│
    │ 401 Token Expired                        │
    │<──────────────────────────────────────────│
    │                    │                     │
    │ refresh_token で   │                     │
    │ 新 access_token 要求                      │
    │───────────────────>│                     │
    │                    │                     │
    │ 新 access_token    │                     │
    │ 新 refresh_token   │ ← リフレッシュトークン │
    │<───────────────────│   ローテーション       │
    │                    │                     │
```

### 6.1 リフレッシュトークンローテーション

```
リフレッシュトークンローテーション:

  なぜ必要か:
  → リフレッシュトークンが漏洩した場合の被害を限定
  → トークンリプレイ攻撃の検知が可能

  仕組み:
  → リフレッシュ時に新しいリフレッシュトークンを発行
  → 旧リフレッシュトークンは無効化
  → 使用済みトークンが再使用されたら全トークンを無効化

  トークンファミリーによるリプレイ検知:

  正常フロー:
    RT-1 → (使用) → AT-2 + RT-2
    RT-2 → (使用) → AT-3 + RT-3

  攻撃シナリオ:
    攻撃者が RT-1 を窃取し、正規ユーザーが RT-2 使用後に
    攻撃者が RT-1 を使用
    → RT-1 は既に使用済み
    → 同じファミリーの全トークンを無効化
    → ユーザーは再ログインが必要（安全側に倒す）
```

```typescript
// リフレッシュトークンローテーションの実装
import crypto from 'crypto';

class RefreshTokenService {
  constructor(private db: Database, private redis: Redis) {}

  // リフレッシュトークン発行
  async issueRefreshToken(userId: string, family?: string): Promise<string> {
    const token = crypto.randomBytes(64).toString('base64url');
    const hashedToken = this.hashToken(token);
    const tokenFamily = family || crypto.randomUUID();

    await this.db.refreshToken.create({
      data: {
        hashedToken,
        userId,
        family: tokenFamily,
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7日
        used: false,
      },
    });

    return token;
  }

  // リフレッシュトークンの使用（ローテーション）
  async rotateToken(token: string): Promise<{
    accessToken: string;
    refreshToken: string;
  }> {
    const hashedToken = this.hashToken(token);

    const stored = await this.db.refreshToken.findFirst({
      where: { hashedToken },
    });

    if (!stored) {
      throw new AuthError('Invalid refresh token', 'INVALID_TOKEN');
    }

    // 期限切れチェック
    if (stored.expiresAt < new Date()) {
      throw new AuthError('Refresh token expired', 'TOKEN_EXPIRED');
    }

    // リプレイ検知: 使用済みトークンが再使用された場合
    if (stored.used) {
      // トークンファミリー全体を無効化（セキュリティ侵害の可能性）
      await this.revokeTokenFamily(stored.family);
      console.warn(
        `Refresh token reuse detected! Family: ${stored.family}, User: ${stored.userId}`
      );
      throw new AuthError('Token reuse detected. All sessions revoked.', 'TOKEN_REUSE');
    }

    // 旧トークンを使用済みに
    await this.db.refreshToken.update({
      where: { id: stored.id },
      data: { used: true, usedAt: new Date() },
    });

    // 新しいリフレッシュトークンを同じファミリーで発行
    const newRefreshToken = await this.issueRefreshToken(stored.userId, stored.family);

    // 新しいアクセストークン発行
    const accessToken = await issueAccessToken(stored.userId);

    return { accessToken, refreshToken: newRefreshToken };
  }

  // トークンファミリー全体の無効化
  async revokeTokenFamily(family: string): Promise<void> {
    await this.db.refreshToken.updateMany({
      where: { family },
      data: { revokedAt: new Date() },
    });
  }

  // ユーザーの全トークン無効化（パスワード変更、アカウント侵害時）
  async revokeAllUserTokens(userId: string): Promise<void> {
    await this.db.refreshToken.updateMany({
      where: { userId },
      data: { revokedAt: new Date() },
    });
  }

  private hashToken(token: string): string {
    return crypto.createHash('sha256').update(token).digest('hex');
  }
}
```

---

## 7. JWT ブラックリスト（失効対策）

```
JWT 失効戦略の比較:

  戦略              │ メリット            │ デメリット
  ────────────────┼───────────────────┼──────────────────
  短命トークン      │ シンプル             │ UX（頻繁な再取得）
  ブラックリスト     │ 即時失効             │ 状態管理が必要
  Token Version   │ ユーザー単位で失効    │ DB 参照が必要
  イベント駆動      │ リアルタイム          │ インフラ複雑

  推奨: 短命アクセストークン + リフレッシュトークン + ブラックリスト（jti）
```

```typescript
// Redis を使った JWT ブラックリスト
class TokenBlacklist {
  constructor(private redis: Redis) {}

  // トークンをブラックリストに追加
  async revoke(jti: string, exp: number): Promise<void> {
    const ttl = exp - Math.floor(Date.now() / 1000);
    if (ttl > 0) {
      // トークンの残り有効期限分だけ保持（メモリ節約）
      await this.redis.setex(`blacklist:${jti}`, ttl, '1');
    }
  }

  // トークンがブラックリストに含まれるか
  async isRevoked(jti: string): Promise<boolean> {
    const result = await this.redis.get(`blacklist:${jti}`);
    return result !== null;
  }

  // 複数トークンの一括失効
  async revokeMany(tokens: Array<{ jti: string; exp: number }>): Promise<void> {
    const pipeline = this.redis.pipeline();
    const now = Math.floor(Date.now() / 1000);

    for (const { jti, exp } of tokens) {
      const ttl = exp - now;
      if (ttl > 0) {
        pipeline.setex(`blacklist:${jti}`, ttl, '1');
      }
    }

    await pipeline.exec();
  }
}

// Token Version による失効（ユーザー単位）
class TokenVersionService {
  constructor(private redis: Redis, private db: Database) {}

  // ユーザーのトークンバージョンを取得
  async getVersion(userId: string): Promise<number> {
    const cached = await this.redis.get(`token_version:${userId}`);
    if (cached) return parseInt(cached, 10);

    const user = await this.db.user.findUnique({
      where: { id: userId },
      select: { tokenVersion: true },
    });

    const version = user?.tokenVersion || 0;
    await this.redis.setex(`token_version:${userId}`, 3600, version.toString());
    return version;
  }

  // トークンバージョンをインクリメント（全トークン無効化）
  async incrementVersion(userId: string): Promise<number> {
    const newVersion = await this.db.user.update({
      where: { id: userId },
      data: { tokenVersion: { increment: 1 } },
      select: { tokenVersion: true },
    });

    await this.redis.setex(
      `token_version:${userId}`,
      3600,
      newVersion.tokenVersion.toString()
    );

    return newVersion.tokenVersion;
  }

  // トークンのバージョン検証
  async isValidVersion(userId: string, tokenVersion: number): Promise<boolean> {
    const currentVersion = await this.getVersion(userId);
    return tokenVersion >= currentVersion;
  }
}

// 検証フローに組み込み
async function verifyTokenWithBlacklist(token: string) {
  const payload = await verifyAccessToken(token);

  // ブラックリストチェック
  if (payload.jti && await blacklist.isRevoked(payload.jti)) {
    throw new AuthError('Token has been revoked', 'TOKEN_REVOKED');
  }

  // Token Version チェック
  if (payload.token_version !== undefined) {
    const isValid = await tokenVersionService.isValidVersion(
      payload.userId,
      payload.token_version as number
    );
    if (!isValid) {
      throw new AuthError('Token version outdated', 'TOKEN_VERSION_OUTDATED');
    }
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

// パスワード変更時（全セッション無効化）
async function onPasswordChange(userId: string) {
  await tokenVersionService.incrementVersion(userId);
  await refreshTokenService.revokeAllUserTokens(userId);
}
```

---

## 8. JWE（JSON Web Encryption）

```
JWS vs JWE:

  JWS（JSON Web Signature）:
    → 署名のみ（改ざん検知）
    → ペイロードは Base64URL（誰でも読める）
    → 大多数の JWT は JWS
    → 形式: header.payload.signature

  JWE（JSON Web Encryption）:
    → 暗号化（内容を隠す）+ 完全性保証
    → ペイロードは暗号文（鍵がないと読めない）
    → 機密情報を含む必要がある場合に使用
    → 形式: header.encryptedKey.iv.ciphertext.authTag

  JWE の 5 部構造:

  ┌─────────────────────────────────────────────────┐
  │                                                 │
  │  Part 1: Protected Header（Base64URL）           │
  │    { "alg": "RSA-OAEP-256",                    │
  │      "enc": "A256GCM" }                         │
  │                                                 │
  │  Part 2: Encrypted Key（暗号化されたCEK）         │
  │    → CEK（Content Encryption Key）を             │
  │      受信者の公開鍵で暗号化                        │
  │                                                 │
  │  Part 3: Initialization Vector（IV）             │
  │    → AES-GCM の初期化ベクトル                     │
  │                                                 │
  │  Part 4: Ciphertext（暗号文）                     │
  │    → ペイロードを CEK + IV で暗号化               │
  │                                                 │
  │  Part 5: Authentication Tag                      │
  │    → 完全性検証タグ                               │
  │                                                 │
  └─────────────────────────────────────────────────┘

  使い分け:
    JWS: 通常の認証トークン（ほとんどのケース）
    JWE: 機密データを含む場合、トークン内容を隠す必要がある場合
    Nested JWT: JWS を JWE で包む（署名 + 暗号化）
```

```typescript
// JWE の実装（jose ライブラリ）
import { CompactEncrypt, compactDecrypt, generateKeyPair } from 'jose';

// 受信者の RSA 鍵ペア
const { publicKey, privateKey } = await generateKeyPair('RSA-OAEP-256');

// JWE 暗号化
async function encryptPayload(payload: Record<string, unknown>): Promise<string> {
  const encoder = new TextEncoder();
  const plaintext = encoder.encode(JSON.stringify(payload));

  return new CompactEncrypt(plaintext)
    .setProtectedHeader({
      alg: 'RSA-OAEP-256',  // 鍵暗号化アルゴリズム
      enc: 'A256GCM',       // コンテンツ暗号化アルゴリズム
      typ: 'JWT',
    })
    .encrypt(publicKey);
}

// JWE 復号
async function decryptPayload(jwe: string): Promise<Record<string, unknown>> {
  const { plaintext } = await compactDecrypt(jwe, privateKey);
  const decoder = new TextDecoder();
  return JSON.parse(decoder.decode(plaintext));
}

// Nested JWT（JWS を JWE で包む）
async function createNestedJwt(
  userId: string,
  sensitiveData: Record<string, unknown>
): Promise<string> {
  // Step 1: JWS（署名付きJWT）を作成
  const jws = await new SignJWT({
    sub: userId,
    ...sensitiveData,
  })
    .setProtectedHeader({ alg: 'ES256' })
    .setIssuedAt()
    .setExpirationTime('15m')
    .sign(signingPrivateKey);

  // Step 2: JWS を JWE で暗号化
  const encoder = new TextEncoder();
  return new CompactEncrypt(encoder.encode(jws))
    .setProtectedHeader({
      alg: 'RSA-OAEP-256',
      enc: 'A256GCM',
      cty: 'JWT',  // Content Type: 中身が JWT であることを示す
    })
    .encrypt(encryptionPublicKey);
}
```

---

## 9. JWT の保存場所

```
JWT の保存場所の比較:

  保存場所         │ XSS耐性 │ CSRF耐性 │ 推奨度 │ 備考
  ──────────────┼────────┼────────┼──────┼──────────
  HttpOnly Cookie│ ✓      │ △      │ ◎    │ SameSite=Lax で改善
  localStorage   │ ✗      │ ✓      │ △    │ XSS で窃取可能
  sessionStorage │ ✗      │ ✓      │ △    │ XSS で窃取可能
  メモリ（変数）   │ ✓      │ ✓      │ ○    │ リロードで消失
  IndexedDB      │ ✗      │ ✓      │ △    │ XSS で窃取可能

  推奨パターン:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  アクセストークン: メモリ内変数に保持                   │
  │  → XSS でも窃取が困難                               │
  │  → リロード時はリフレッシュトークンで再取得             │
  │                                                    │
  │  リフレッシュトークン: HttpOnly + Secure + SameSite   │
  │  → Cookie で安全に保持                               │
  │  → JavaScript からアクセス不可                       │
  │  → /api/refresh エンドポイントのみで使用              │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

```typescript
// Cookie でのリフレッシュトークン設定
function setRefreshTokenCookie(res: Response, refreshToken: string) {
  res.cookie('refresh_token', refreshToken, {
    httpOnly: true,          // JavaScript からアクセス不可
    secure: true,            // HTTPS のみ
    sameSite: 'strict',      // 同一サイトのみ送信
    path: '/api/auth',       // 認証エンドポイントのみに送信
    maxAge: 7 * 24 * 60 * 60 * 1000, // 7日
    domain: '.example.com',  // サブドメイン共有
  });
}

// リフレッシュエンドポイント
app.post('/api/auth/refresh', async (req, res) => {
  const refreshToken = req.cookies.refresh_token;

  if (!refreshToken) {
    return res.status(401).json({ error: 'No refresh token' });
  }

  try {
    const { accessToken, refreshToken: newRefreshToken } =
      await refreshTokenService.rotateToken(refreshToken);

    // 新しいリフレッシュトークンを Cookie に設定
    setRefreshTokenCookie(res, newRefreshToken);

    // アクセストークンはレスポンスボディで返す（メモリに保持）
    res.json({ access_token: accessToken, token_type: 'Bearer', expires_in: 900 });
  } catch (error) {
    // リフレッシュ失敗時は Cookie を削除
    res.clearCookie('refresh_token', { path: '/api/auth' });
    return res.status(401).json({ error: 'Invalid refresh token' });
  }
});

// フロントエンド側: アクセストークンのメモリ管理
class TokenManager {
  private accessToken: string | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;

  setAccessToken(token: string, expiresIn: number) {
    this.accessToken = token;

    // 有効期限の 80% 時点で自動リフレッシュ
    if (this.refreshTimer) clearTimeout(this.refreshTimer);
    this.refreshTimer = setTimeout(
      () => this.refresh(),
      expiresIn * 0.8 * 1000
    );
  }

  getAccessToken(): string | null {
    return this.accessToken;
  }

  async refresh(): Promise<void> {
    try {
      const res = await fetch('/api/auth/refresh', {
        method: 'POST',
        credentials: 'include', // Cookie を送信
      });

      if (!res.ok) {
        this.accessToken = null;
        window.location.href = '/login';
        return;
      }

      const { access_token, expires_in } = await res.json();
      this.setAccessToken(access_token, expires_in);
    } catch (error) {
      console.error('Token refresh failed:', error);
    }
  }

  // Axios インターセプターでの使用
  setupAxiosInterceptor(axios: AxiosInstance) {
    axios.interceptors.request.use((config) => {
      const token = this.getAccessToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    axios.interceptors.response.use(
      (response) => response,
      async (error) => {
        if (error.response?.status === 401 && !error.config._retry) {
          error.config._retry = true;
          await this.refresh();
          error.config.headers.Authorization = `Bearer ${this.getAccessToken()}`;
          return axios(error.config);
        }
        return Promise.reject(error);
      }
    );
  }
}
```

---

## 10. アンチパターン

```
JWT 実装のアンチパターン:

  ✗ アンチパターン 1: セッション代替として JWT を使う
    → JWT はステートレス。サーバー側の状態管理が必要なら
      セッションを使うべき
    → JWT ブラックリスト = 結局サーバー側に状態を持つ
    → 適切な判断: API 認証 → JWT、Web セッション → Cookie セッション

  ✗ アンチパターン 2: ペイロードに機密情報を含める
    → JWT は署名のみ。Base64URL デコードで誰でも読める
    → パスワード、SSN、クレジットカード番号を含めない
    → 必要なら JWE で暗号化

  ✗ アンチパターン 3: 長い有効期限のアクセストークン
    → 有効期限 24 時間は長すぎる
    → 漏洩時のリスクが増大
    → 推奨: 15分〜1時間

  ✗ アンチパターン 4: algorithms を指定しない検証
    → alg: "none" 攻撃やアルゴリズム混乱攻撃に脆弱
    → 必ず algorithms: ['ES256'] のように明示

  ✗ アンチパターン 5: JWT のペイロードでアクセス制御
    → JWT のクレームだけで重要な操作を許可
    → トークン窃取時にすべての権限が使える
    → 重要な操作は追加の認証（ステップアップ認証）を要求

  ✗ アンチパターン 6: トークンサイズの無制限な肥大化
    → 多数のクレームを追加してサイズが 8KB 超過
    → HTTP ヘッダーの上限を超える
    → パフォーマンス劣化（毎リクエストで送信）
```

---

## 11. パフォーマンス最適化

```
JWT パフォーマンスの考慮点:

  署名/検証の CPU コスト:
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  HS256: ~0.01ms（対称鍵、最速）               │
  │  ES256: ~0.1ms（楕円曲線、高速）              │
  │  RS256: ~1ms 署名 / ~0.05ms 検証             │
  │  EdDSA: ~0.05ms（署名・検証とも高速）          │
  │                                              │
  │  高トラフィック環境（10万 req/s）での考慮:      │
  │  → ES256 で 10,000 CPU-ms/s = 10 CPU コア分  │
  │  → RS256 署名は CPU バウンドのボトルネック      │
  │  → 検証のみなら RS256 でも十分高速             │
  │                                              │
  └──────────────────────────────────────────────┘

  ネットワークコスト:
  ┌──────────────────────────────────────────────┐
  │                                              │
  │  JWT サイズの影響:                             │
  │  → 典型的な JWT: 500-800 バイト               │
  │  → クレーム追加で 2-4 KB に                    │
  │  → 全 HTTP リクエストに付与 → 帯域幅への影響    │
  │                                              │
  │  対策:                                        │
  │  → クレームは最小限に                          │
  │  → ES256/EdDSA（署名サイズが小さい）           │
  │  → 不要な情報は /userinfo で別途取得           │
  │                                              │
  └──────────────────────────────────────────────┘
```

```typescript
// パフォーマンス計測ユーティリティ
async function benchmarkAlgorithms() {
  const algorithms = ['HS256', 'RS256', 'ES256', 'EdDSA'] as const;
  const iterations = 10000;

  for (const alg of algorithms) {
    // 鍵生成
    let signingKey: any;
    let verifyKey: any;

    if (alg === 'HS256') {
      const secret = new TextEncoder().encode('super-secret-key-at-least-256-bits!!!');
      signingKey = verifyKey = secret;
    } else {
      const options = alg === 'EdDSA' ? { crv: 'Ed25519' as const } : undefined;
      const keys = await generateKeyPair(alg, options);
      signingKey = keys.privateKey;
      verifyKey = keys.publicKey;
    }

    // 署名ベンチマーク
    const signStart = performance.now();
    let token = '';
    for (let i = 0; i < iterations; i++) {
      token = await new SignJWT({ sub: 'user_123', role: 'admin' })
        .setProtectedHeader({ alg })
        .setExpirationTime('15m')
        .sign(signingKey);
    }
    const signTime = (performance.now() - signStart) / iterations;

    // 検証ベンチマーク
    const verifyStart = performance.now();
    for (let i = 0; i < iterations; i++) {
      await jwtVerify(token, verifyKey, { algorithms: [alg] });
    }
    const verifyTime = (performance.now() - verifyStart) / iterations;

    console.log(`${alg}: sign=${signTime.toFixed(3)}ms, verify=${verifyTime.toFixed(3)}ms, size=${token.length}bytes`);
  }
}
```

---

## 12. 演習

### 演習 1: JWT の手動デコード（基礎）

```
課題:
  以下の JWT をデコードし、ヘッダーとペイロードの内容を確認せよ。
  署名の検証はせず、構造の理解に集中する。

  eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.
  eyJzdWIiOiJ1c2VyXzQ1NiIsInJvbGUiOiJhZG1pbiIsImlhdCI6MTcwMDAwMDAwMCwiZXhwIjoxNzAwMDAwOTAwfQ.
  MEUCIBvFRjy0GhtOm3cqBrRNbMGxmLNXDG3sFrSHVBdZ0sIHAiEA6XnzM0TSXwPNqSf1fXQz0rN3wGpMC2q0aHjB_7nYTqI

  手順:
  1. ドットで3つに分割
  2. 各パートを Base64URL デコード
  3. ヘッダーの alg と typ を確認
  4. ペイロードの各クレームの意味を説明
  5. exp から有効期限を人間可読な日時に変換
```

```typescript
// 演習 1 の解答テンプレート
function exercise1(jwt: string) {
  const parts = jwt.split('.');
  console.log(`パート数: ${parts.length}`);

  // ヘッダーのデコード
  const header = JSON.parse(
    Buffer.from(parts[0], 'base64url').toString()
  );
  console.log('ヘッダー:', header);

  // ペイロードのデコード
  const payload = JSON.parse(
    Buffer.from(parts[1], 'base64url').toString()
  );
  console.log('ペイロード:', payload);

  // 有効期限の変換
  const expDate = new Date(payload.exp * 1000);
  console.log('有効期限:', expDate.toISOString());

  // 署名部分（バイナリデータ、デコードしても意味のある文字列にはならない）
  console.log('署名（Base64URL）:', parts[2]);
  console.log('署名サイズ:', Buffer.from(parts[2], 'base64url').length, 'bytes');
}
```

### 演習 2: 安全な JWT 認証 API の構築（応用）

```
課題:
  Express + jose を使って以下の要件を満たす認証 API を構築せよ。

  要件:
  1. POST /auth/login: メール + パスワード → access_token + refresh_token
  2. POST /auth/refresh: refresh_token → 新 access_token + 新 refresh_token
  3. POST /auth/logout: トークンの失効
  4. GET /api/profile: 認証必須、ユーザープロフィール返却

  セキュリティ要件:
  → ES256 アルゴリズム
  → アクセストークン有効期限: 15分
  → リフレッシュトークン: HttpOnly Cookie、7日有効
  → ブラックリスト（Redis）
  → リフレッシュトークンローテーション
  → 適切なエラーハンドリング（401/403の使い分け）

  ヒント:
  → jose ライブラリの SignJWT, jwtVerify を使用
  → Cookie の設定: httpOnly, secure, sameSite
  → エラー時は具体的な情報を漏らさない
```

### 演習 3: マイクロサービス間 JWT 認証（発展）

```
課題:
  3つのマイクロサービス（Auth、User、Order）間で
  JWKS ベースの JWT 認証を実装せよ。

  アーキテクチャ:
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │  Auth    │   │  User    │   │  Order   │
  │  Service │   │  Service │   │  Service │
  │          │   │          │   │          │
  │ 鍵生成   │   │ JWKS検証 │   │ JWKS検証 │
  │ JWT発行  │   │ aud検証  │   │ aud検証  │
  │ JWKS公開 │   │          │   │          │
  └──────────┘   └──────────┘   └──────────┘
       ↑              │              │
       │ JWKS取得      │              │
       └──────────────┴──────────────┘

  要件:
  1. Auth Service: JWKS エンドポイント + JWT 発行
  2. User Service: JWKS から公開鍵取得、aud: "user-api" 検証
  3. Order Service: JWKS から公開鍵取得、aud: "order-api" 検証
  4. 鍵ローテーション: 30日ごとに新しい鍵を生成
  5. グレースピリオド: 旧鍵は24時間検証可能

  発展:
  → サービスメッシュ（Istio）との連携
  → mutual TLS + JWT の組み合わせ
  → スコープベースのアクセス制御
```

### 演習 4: JWT セキュリティ監査（発展）

```
課題:
  以下のコードのセキュリティ上の問題点を全て特定し、修正せよ。

  // 脆弱なコード（問題点を特定せよ）
  const jwt = require('jsonwebtoken');

  app.post('/login', (req, res) => {
    const user = db.findUser(req.body.email);
    if (user && req.body.password === user.password) {
      const token = jwt.sign(
        {
          userId: user.id,
          email: user.email,
          password: user.password,
          role: user.role,
          creditCard: user.creditCard,
        },
        'secret123',
        { expiresIn: '30d' }
      );
      res.json({ token });
    }
  });

  app.get('/api/data', (req, res) => {
    const token = req.headers.authorization;
    const decoded = jwt.decode(token);
    if (decoded) {
      res.json(getData(decoded.userId));
    }
  });

  ヒント: 少なくとも 10 個の問題点がある
```

---

## 13. FAQ・トラブルシューティング

```
Q1: JWT のサイズが大きすぎて Cookie に入らない
A1: → アクセストークンを Cookie ではなく Authorization ヘッダーで送信
    → クレームを最小限にする（詳細は /userinfo で取得）
    → RS256 → ES256 に変更（署名サイズが 1/4）
    → 圧縮は推奨しない（BREACH 攻撃のリスク）

Q2: JWT の有効期限切れでユーザー体験が悪い
A2: → サイレントリフレッシュ（有効期限の 80% で自動更新）
    → リフレッシュトークン + Cookie
    → Service Worker でバックグラウンドリフレッシュ

Q3: 複数タブでのトークン同期
A3: → BroadcastChannel API でタブ間通信
    → localStorage の storage イベントを監視
    → ログアウト時に全タブで同期

Q4: JWT の時刻ズレで検証が失敗する
A4: → clockTolerance パラメータを設定（30-60秒推奨）
    → サーバー間のNTP同期を確認
    → 過度な許容は避ける（5分以上は危険）

Q5: 本番環境で alg: "none" 攻撃を受けた
A5: → algorithms パラメータを必ず指定
    → jwt.decode() を認証に使わない（検証なし）
    → ライブラリのバージョンを最新に保つ
    → セキュリティヘッダーのレビューを定期的に実施

Q6: マイクロサービスで JWKS の取得に失敗する
A6: → JWKS レスポンスをキャッシュ（10-15分）
    → フォールバック用にローカルに公開鍵を保持
    → サーキットブレーカーパターンを適用
    → JWKS エンドポイントの高可用性を確保

Q7: リフレッシュトークンが頻繁に無効になる
A7: → リフレッシュトークンローテーションの実装を確認
    → 同一トークンの並行使用（レースコンディション）を疑う
    → リフレッシュリクエストの排他制御（ロック）を実装
    → トークンファミリーの管理を見直す
```

---

## まとめ

| 項目 | 推奨 |
|------|------|
| アルゴリズム | ES256（ECDSA）または EdDSA（Ed25519） |
| 有効期限 | Access: 15分、Refresh: 7日 |
| クレーム | 必要最小限（sub, role, exp） |
| 検証 | algorithms, issuer, audience を必ず指定 |
| 失効 | 短命トークン + ブラックリスト + Token Version |
| 鍵管理 | JWKS + 定期ローテーション（30日） |
| 保存場所 | Access: メモリ、Refresh: HttpOnly Cookie |
| 暗号化 | 機密データは JWE、通常は JWS で十分 |
| サイズ | 4KB 以下を推奨 |
| ライブラリ | jose（推奨）、jsonwebtoken（レガシー） |

---

## 次に読むべきガイド
→ [[01-oauth2-flows.md]] — OAuth 2.0 フロー
→ [[02-openid-connect.md]] — OpenID Connect
→ [[../01-access-control/00-rbac.md]] — ロールベースアクセス制御

---

## 参考文献
1. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
2. RFC 7517. "JSON Web Key (JWK)." IETF, 2015.
3. RFC 7516. "JSON Web Encryption (JWE)." IETF, 2015.
4. RFC 7515. "JSON Web Signature (JWS)." IETF, 2015.
5. RFC 7518. "JSON Web Algorithms (JWA)." IETF, 2015.
6. Auth0. "JWT Handbook." auth0.com, 2024.
7. OWASP. "JSON Web Token Cheat Sheet." cheatsheetseries.owasp.org, 2024.
8. IETF. "JSON Web Token Best Current Practices." RFC 8725, 2020.
9. Neil Madden. "API Security in Action." Manning, 2020.
10. Daniel Vassallo. "JWT Security Best Practices." blog, 2024.
