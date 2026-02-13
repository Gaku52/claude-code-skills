# 多要素認証（MFA）

> パスワード単体では不十分な時代。TOTP、WebAuthn/Passkeys、SMS認証、リカバリーコードまで、多要素認証の仕組みと安全な実装方法を解説する。パスワードレス認証の未来も見据える。RFC 6238（TOTP）、W3C WebAuthn、FIDO2 の仕様に基づき、内部アルゴリズムから実装・運用のベストプラクティスまでを網羅する。

## 前提知識

- [[00-authentication-vs-authorization.md]] — 認証と認可の基礎
- [[01-password-security.md]] — パスワードセキュリティ
- 対称鍵暗号・HMAC の基本概念
- 公開鍵暗号の基礎
- HTTP Cookie とセッション管理の基本

## この章で学ぶこと

- [ ] MFAの種類と各方式のセキュリティ強度を理解する
- [ ] TOTPの内部アルゴリズム（RFC 6238 / RFC 4226）を把握する
- [ ] WebAuthn/Passkeysの仕組みと実装方法を習得する
- [ ] MFAの実装とリカバリー戦略を設計できるようになる
- [ ] パスワードレス認証の設計パターンを理解する
- [ ] ステップアップ認証の実装方法を学ぶ

---

## 1. MFA の基礎理論

```
認証の3要素:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ① Something You Know（知識要素）                 │
  │     → パスワード、PIN、秘密の質問                   │
  │     → 最も一般的だが最も脆弱                       │
  │     → フィッシング、ブルートフォースに弱い            │
  │                                                  │
  │  ② Something You Have（所有要素）                 │
  │     → スマートフォン、ハードウェアキー、ICカード       │
  │     → TOTP、SMS コード、FIDO2 セキュリティキー       │
  │     → 物理的な盗難が必要 → リモート攻撃が困難        │
  │                                                  │
  │  ③ Something You Are（生体要素）                  │
  │     → 指紋、顔認証、虹彩、声紋                      │
  │     → 変更不可（漏洩時のリスクが高い）               │
  │     → デバイスローカルで処理（サーバーに送信しない）   │
  │                                                  │
  └──────────────────────────────────────────────────┘

MFA の原則:
  → 異なるカテゴリの要素を組み合わせる
  → 同一カテゴリの複数要素は MFA にならない

  ✗ パスワード + 秘密の質問 = 単一要素（両方「知識」）
  ✓ パスワード + TOTP = 多要素（「知識」+「所有」）
  ✓ パスワード + 指紋 = 多要素（「知識」+「生体」）
  ✓ Passkey = 多要素（「所有」+「生体」を1つのデバイスで実現）
```

---

## 2. MFA方式の比較

```
MFA方式の詳細比較:

  方式          │ セキュリティ │ UX    │ フィッシング │ コスト  │ オフライン
  ─────────────┼────────────┼──────┼───────────┼───────┼────────
  SMS OTP      │ △ 低い     │ ○ 良  │ ✗ 弱い     │ 通信費  │ ✗ 不可
  Email OTP    │ △ 低い     │ ○ 良  │ ✗ 弱い     │ 無料   │ ✗ 不可
  TOTP         │ ○ 中程度   │ ○ 良  │ △ やや弱い  │ 無料   │ ✓ 可能
  Push通知     │ ○ 中程度   │ ◎ 最良│ △ やや弱い  │ 有料   │ ✗ 不可
  FIDO2/       │ ◎ 最強    │ ◎ 最良│ ✓ 耐性あり  │ キー代  │ ✓ 可能
  WebAuthn     │            │       │            │        │
  Passkeys     │ ◎ 最強    │ ◎ 最良│ ✓ 耐性あり  │ 無料   │ ✓ 可能

推奨の優先順位:
  1. Passkeys / WebAuthn（最も安全 + UX最良）
  2. TOTP（広く普及、無料）
  3. Push 通知（UX が良い）
  4. SMS OTP（最後の手段）

攻撃耐性の詳細比較:

  攻撃手法            │ SMS  │ TOTP │ Push │ WebAuthn
  ───────────────────┼─────┼─────┼─────┼────────
  フィッシング         │ ✗    │ ✗    │ ✗    │ ✓ 耐性
  SIM スワップ        │ ✗    │ ✓    │ ✓    │ ✓
  SS7 傍受           │ ✗    │ ✓    │ ✓    │ ✓
  リアルタイムフィッシング│ ✗   │ ✗    │ △    │ ✓
  MitM プロキシ       │ ✗    │ ✗    │ △    │ ✓
  ソーシャルエンジニアリング│ ✗  │ △    │ ✗    │ ✓
  マルウェア           │ △    │ △    │ △    │ ○
```

---

## 3. TOTP（Time-based One-Time Password）

### 3.1 TOTP の内部アルゴリズム

```
TOTP の仕組み（RFC 6238 + RFC 4226）:

  セットアップ:
    ① サーバーが秘密鍵（secret）を生成（160ビット以上推奨）
    ② QRコードとして表示（otpauth:// URI）
    ③ ユーザーが認証アプリ（Google Authenticator等）でスキャン
    ④ ユーザーが表示された6桁コードを入力して検証

  認証時:
    ① ユーザーが認証アプリに表示された6桁コードを入力
    ② サーバーが同じアルゴリズムでコードを計算
    ③ 一致すれば認証成功

  アルゴリズムの内部動作:

  Step 1: タイムステップの計算
  ┌──────────────────────────────────────────┐
  │  T = floor(unix_time / period)           │
  │                                          │
  │  unix_time = 1700000000（秒）             │
  │  period = 30（秒）                        │
  │  T = floor(1700000000 / 30) = 56666666   │
  │                                          │
  │  → 30秒ごとに T が変わる                   │
  │  → サーバーとクライアントで T が同期        │
  └──────────────────────────────────────────┘

  Step 2: HOTP の計算（RFC 4226）
  ┌──────────────────────────────────────────┐
  │  HOTP(K, C) = Truncate(HMAC-SHA1(K, C))  │
  │                                          │
  │  K = 秘密鍵（Base32エンコード前の生バイト）  │
  │  C = T（8バイトのビッグエンディアン表現）    │
  │                                          │
  │  ① hmac = HMAC-SHA1(K, C)               │
  │     → 20バイト（160ビット）のハッシュ値     │
  │                                          │
  │  ② offset = hmac[19] & 0x0F              │
  │     → 最後のバイトの下位4ビット（0-15）     │
  │                                          │
  │  ③ binary = (hmac[offset] & 0x7F) << 24  │
  │           | hmac[offset+1] << 16          │
  │           | hmac[offset+2] << 8           │
  │           | hmac[offset+3]                │
  │     → 4バイトを31ビット整数に変換           │
  │                                          │
  │  ④ otp = binary % 10^digits              │
  │     → 6桁: binary % 1000000              │
  │     → 結果: "481592"                      │
  └──────────────────────────────────────────┘

  QRコード URI:
    otpauth://totp/MyApp:alice@example.com
      ?secret=JBSWY3DPEHPK3PXP
      &issuer=MyApp
      &algorithm=SHA1
      &digits=6
      &period=30
```

```typescript
// TOTP の内部実装（教育目的、本番では otplib を使用）
import crypto from 'crypto';

function generateTOTP(
  secret: Buffer,      // 秘密鍵（生バイト）
  period: number = 30, // タイムステップ（秒）
  digits: number = 6,  // OTP の桁数
  algorithm: string = 'sha1'
): string {
  // Step 1: タイムステップの計算
  const time = Math.floor(Date.now() / 1000 / period);

  // Step 2: タイムステップを8バイトのビッグエンディアンに変換
  const timeBuffer = Buffer.alloc(8);
  timeBuffer.writeBigUInt64BE(BigInt(time));

  // Step 3: HMAC-SHA1 を計算
  const hmac = crypto.createHmac(algorithm, secret).update(timeBuffer).digest();

  // Step 4: Dynamic Truncation
  const offset = hmac[hmac.length - 1] & 0x0f;
  const binary = (
    ((hmac[offset] & 0x7f) << 24) |
    ((hmac[offset + 1] & 0xff) << 16) |
    ((hmac[offset + 2] & 0xff) << 8) |
    (hmac[offset + 3] & 0xff)
  );

  // Step 5: 指定桁数に切り詰め
  const otp = binary % Math.pow(10, digits);

  // Step 6: 先頭のゼロを保持してゼロパディング
  return otp.toString().padStart(digits, '0');
}

// 検証（前後のウィンドウも許容）
function verifyTOTP(
  token: string,
  secret: Buffer,
  window: number = 1  // 前後何ステップを許容するか
): boolean {
  const period = 30;
  const currentTime = Math.floor(Date.now() / 1000);

  for (let i = -window; i <= window; i++) {
    const time = Math.floor(currentTime / period) + i;
    const timeBuffer = Buffer.alloc(8);
    timeBuffer.writeBigUInt64BE(BigInt(time));

    const hmac = crypto.createHmac('sha1', secret).update(timeBuffer).digest();
    const offset = hmac[hmac.length - 1] & 0x0f;
    const binary = (
      ((hmac[offset] & 0x7f) << 24) |
      ((hmac[offset + 1] & 0xff) << 16) |
      ((hmac[offset + 2] & 0xff) << 8) |
      (hmac[offset + 3] & 0xff)
    );
    const otp = (binary % 1000000).toString().padStart(6, '0');

    // タイミングセーフな比較
    if (crypto.timingSafeEqual(Buffer.from(otp), Buffer.from(token))) {
      return true;
    }
  }

  return false;
}
```

### 3.2 TOTP の実装（otplib ライブラリ）

```typescript
// TOTP 実装（otplib）
import { authenticator } from 'otplib';
import qrcode from 'qrcode';

// セットアップ: 秘密鍵の生成
async function setupTOTP(userId: string, email: string) {
  const secret = authenticator.generateSecret(); // Base32 エンコード

  // 秘密鍵を暗号化してDBに保存（MFA有効化前は仮保存）
  await db.mfaSetup.create({
    data: {
      userId,
      secret: encrypt(secret),  // AES で暗号化して保存
      verified: false,
    },
  });

  // QRコード生成
  const otpauthUrl = authenticator.keyuri(email, 'MyApp', secret);
  const qrCodeUrl = await qrcode.toDataURL(otpauthUrl);

  return { qrCodeUrl, secret }; // secret はバックアップ用に表示
}

// セットアップ検証: ユーザーが入力したコードを確認
async function verifyTOTPSetup(userId: string, token: string) {
  const setup = await db.mfaSetup.findUnique({ where: { userId } });
  if (!setup) throw new Error('MFA setup not found');

  const secret = decrypt(setup.secret);
  const isValid = authenticator.verify({ token, secret });

  if (!isValid) {
    throw new Error('Invalid TOTP code');
  }

  // MFA を有効化
  await db.$transaction([
    db.user.update({
      where: { id: userId },
      data: { mfaEnabled: true },
    }),
    db.mfaSetup.update({
      where: { userId },
      data: { verified: true },
    }),
  ]);

  // リカバリーコード生成
  const recoveryCodes = generateRecoveryCodes();
  await saveRecoveryCodes(userId, recoveryCodes);

  return { recoveryCodes }; // ユーザーに1度だけ表示
}

// 認証時の検証
async function verifyTOTP(userId: string, token: string): Promise<boolean> {
  const setup = await db.mfaSetup.findUnique({
    where: { userId, verified: true },
  });
  if (!setup) return false;

  const secret = decrypt(setup.secret);

  // リプレイ攻撃防止: 使用済みコードをチェック
  const codeKey = `totp:used:${userId}:${token}`;
  const isUsed = await redis.exists(codeKey);
  if (isUsed) return false;

  // ウィンドウ=1: 前後30秒のコードも許容
  const isValid = authenticator.verify({ token, secret });

  if (isValid) {
    // 使用済みとして記録（90秒間保持 = period * 3）
    await redis.setex(codeKey, 90, '1');
  }

  return isValid;
}
```

### 3.3 TOTP のセキュリティ考慮事項

```
TOTP のセキュリティリスクと対策:

  ① 秘密鍵の保護:
     → DB に平文保存しない（AES-256-GCM 等で暗号化）
     → 暗号化鍵は HSM や KMS で管理
     → バックアップも暗号化して保存

  ② リプレイ攻撃:
     → 同じ OTP コードの再使用を防止
     → 使用済みコードを Redis に短期間保存
     → 30秒 × 3 = 90秒間のウィンドウで管理

  ③ タイムドリフト:
     → サーバーとクライアントの時刻ズレ
     → NTP で時刻同期が前提
     → ウィンドウ（前後1-2ステップ）で許容
     → 大きなドリフト → ユーザーに時刻同期を案内

  ④ ブルートフォース:
     → 6桁 = 100万通り（30秒以内に試行が必要）
     → レート制限: 5回失敗でロックアウト
     → ロックアウト時間: 15-30分

  ⑤ フィッシング（リアルタイムプロキシ）:
     → 攻撃者がユーザーと本物のサーバーの間に立つ
     → ユーザーが入力した OTP をリアルタイムで転送
     → TOTP はこの攻撃に脆弱（WebAuthn は耐性あり）
     → 対策: WebAuthn の併用推奨

  ⑥ TOTP から WebAuthn への移行パス:
     → まず TOTP を導入
     → WebAuthn/Passkey を追加オプションとして提供
     → ユーザーの移行を促す
     → 最終的に TOTP を補助方式に
```

---

## 4. WebAuthn / Passkeys

### 4.1 WebAuthn の内部動作

```
WebAuthn の仕組み:

  公開鍵暗号ベースの認証:
    → パスワードを使わない
    → 秘密鍵はデバイスに保存（サーバーには送信されない）
    → フィッシング完全耐性（オリジン検証あり）

  登録フロー:

  ユーザー    ブラウザ        認証器          サーバー
    │          │              │               │
    │ 登録開始  │              │               │
    │─────────>│              │               │
    │          │ オプション要求  │               │
    │          │──────────────────────────────>│
    │          │              │               │
    │          │ challenge +   │               │
    │          │ rpId + userId │               │
    │          │<──────────────────────────────│
    │          │              │               │
    │          │ credentials  │               │
    │          │ .create()    │               │
    │          │─────────────>│               │
    │          │              │               │
    │ 指紋/顔   │              │ ユーザー検証   │
    │ 認証     │              │ 鍵ペア生成     │
    │<─────────│              │               │
    │ OK       │              │               │
    │─────────>│              │               │
    │          │              │ 公開鍵 +      │
    │          │              │ attestation   │
    │          │<─────────────│               │
    │          │              │               │
    │          │ 公開鍵 +      │               │
    │          │ attestation  │               │
    │          │──────────────────────────────>│
    │          │              │               │
    │          │              │   署名検証     │
    │          │              │   公開鍵保存   │
    │          │ 登録完了       │               │
    │          │<──────────────────────────────│
    │ 完了     │              │               │
    │<─────────│              │               │

  認証フロー:

  ユーザー    ブラウザ        認証器          サーバー
    │          │              │               │
    │ ログイン  │              │               │
    │─────────>│              │               │
    │          │ challenge 要求│               │
    │          │──────────────────────────────>│
    │          │              │               │
    │          │ challenge +  │               │
    │          │ allowCredentials              │
    │          │<──────────────────────────────│
    │          │              │               │
    │          │ credentials  │               │
    │          │ .get()       │               │
    │          │─────────────>│               │
    │          │              │               │
    │ 指紋/顔   │              │ ユーザー検証   │
    │ 認証     │              │ challenge署名  │
    │<─────────│              │               │
    │ OK       │              │               │
    │─────────>│              │               │
    │          │              │ 署名データ     │
    │          │<─────────────│               │
    │          │              │               │
    │          │ 署名データ送信  │               │
    │          │──────────────────────────────>│
    │          │              │               │
    │          │              │   公開鍵で     │
    │          │              │   署名検証     │
    │          │ 認証成功       │               │
    │          │<──────────────────────────────│
    │ ログイン  │              │               │
    │ 成功     │              │               │
    │<─────────│              │               │

  Passkeys（WebAuthn の進化）:
    → iCloud Keychain / Google Password Manager で同期
    → デバイス間で使える（iPhone で登録 → Mac で使用）
    → パスワードレス認証のデファクトスタンダード
    → Conditional UI: 入力フィールドに Passkey 候補を表示

  なぜフィッシング耐性があるか:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  WebAuthn の認証データに含まれるもの:               │
  │  → rpIdHash: RP（Relying Party）のドメインハッシュ  │
  │  → origin: リクエスト元のオリジン                   │
  │                                                  │
  │  example.com で登録した鍵:                          │
  │  → rpIdHash = SHA256("example.com")              │
  │  → evil.com からの認証要求 → rpId が不一致          │
  │  → 認証器が自動的に拒否                            │
  │                                                  │
  │  ユーザーが偽サイト evil.com にアクセスしても:        │
  │  → 認証器は example.com の鍵を使用しない            │
  │  → フィッシング攻撃が原理的に不可能                  │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 4.2 WebAuthn サーバー実装

```typescript
// WebAuthn 実装（@simplewebauthn/server + @simplewebauthn/browser）

// サーバー側: 登録オプション生成
import {
  generateRegistrationOptions,
  verifyRegistrationResponse,
  generateAuthenticationOptions,
  verifyAuthenticationResponse,
} from '@simplewebauthn/server';

const rpName = 'My App';
const rpID = 'example.com';
const origin = 'https://example.com';

// 登録: オプション生成
async function getRegistrationOptions(userId: string) {
  const user = await db.user.findUnique({ where: { id: userId } });
  const existingDevices = await db.credential.findMany({
    where: { userId },
  });

  const options = await generateRegistrationOptions({
    rpName,
    rpID,
    userID: new TextEncoder().encode(userId),
    userName: user!.email,
    attestationType: 'none',
    excludeCredentials: existingDevices.map((d) => ({
      id: d.credentialId,
      transports: d.transports,
    })),
    authenticatorSelection: {
      residentKey: 'preferred',          // Passkey をサポート
      userVerification: 'preferred',     // 生体認証を推奨
    },
  });

  // チャレンジを一時保存（5分間有効）
  await redis.setex(
    `webauthn:challenge:${userId}`,
    300,
    options.challenge
  );

  return options;
}

// 登録: レスポンス検証
async function verifyRegistration(userId: string, response: any) {
  const expectedChallenge = await redis.get(`webauthn:challenge:${userId}`);
  if (!expectedChallenge) {
    throw new Error('Challenge expired or not found');
  }

  const verification = await verifyRegistrationResponse({
    response,
    expectedChallenge,
    expectedOrigin: origin,
    expectedRPID: rpID,
  });

  if (verification.verified && verification.registrationInfo) {
    const { credential } = verification.registrationInfo;

    // クレデンシャルを保存
    await db.credential.create({
      data: {
        userId,
        credentialId: Buffer.from(credential.id),
        publicKey: Buffer.from(credential.publicKey),
        counter: credential.counter,
        transports: response.response.transports,
        deviceType: verification.registrationInfo.credentialDeviceType,
        backedUp: verification.registrationInfo.credentialBackedUp,
        createdAt: new Date(),
      },
    });

    // チャレンジを削除
    await redis.del(`webauthn:challenge:${userId}`);
  }

  return verification;
}

// 認証: オプション生成
async function getAuthenticationOptions(email?: string) {
  let allowCredentials: any[] = [];

  if (email) {
    // メールアドレスが指定された場合、そのユーザーのクレデンシャルを取得
    const user = await db.user.findUnique({ where: { email } });
    if (user) {
      const credentials = await db.credential.findMany({
        where: { userId: user.id },
      });
      allowCredentials = credentials.map((c) => ({
        id: c.credentialId,
        transports: c.transports,
      }));
    }
  }
  // email が指定されない場合、allowCredentials を空にする
  // → Discoverable Credential（Passkey）が使われる

  const options = await generateAuthenticationOptions({
    rpID,
    allowCredentials,
    userVerification: 'preferred',
  });

  // チャレンジを一時保存
  const challengeKey = email || 'anonymous';
  await redis.setex(
    `webauthn:auth:challenge:${challengeKey}`,
    300,
    options.challenge
  );

  return options;
}

// 認証: レスポンス検証
async function verifyAuthentication(response: any, email?: string) {
  // クレデンシャルIDからユーザーを特定
  const credential = await db.credential.findFirst({
    where: {
      credentialId: Buffer.from(response.id, 'base64url'),
    },
    include: { user: true },
  });

  if (!credential) {
    throw new Error('Credential not found');
  }

  const challengeKey = email || 'anonymous';
  const expectedChallenge = await redis.get(
    `webauthn:auth:challenge:${challengeKey}`
  );

  if (!expectedChallenge) {
    throw new Error('Challenge expired');
  }

  const verification = await verifyAuthenticationResponse({
    response,
    expectedChallenge,
    expectedOrigin: origin,
    expectedRPID: rpID,
    credential: {
      id: credential.credentialId,
      publicKey: credential.publicKey,
      counter: credential.counter,
    },
  });

  if (verification.verified) {
    // カウンターを更新（クローン検知）
    await db.credential.update({
      where: { id: credential.id },
      data: {
        counter: verification.authenticationInfo.newCounter,
        lastUsedAt: new Date(),
      },
    });

    // チャレンジを削除
    await redis.del(`webauthn:auth:challenge:${challengeKey}`);

    return { verified: true, user: credential.user };
  }

  return { verified: false, user: null };
}
```

### 4.3 WebAuthn クライアント実装

```typescript
// クライアント側: WebAuthn 登録・認証
import {
  startRegistration,
  startAuthentication,
  browserSupportsWebAuthn,
  platformAuthenticatorIsAvailable,
} from '@simplewebauthn/browser';

// WebAuthn サポートチェック
async function checkWebAuthnSupport() {
  const supported = browserSupportsWebAuthn();
  const platformAvailable = await platformAuthenticatorIsAvailable();

  return {
    supported,              // ブラウザが WebAuthn をサポート
    platformAvailable,       // 指紋/顔認証が使用可能
    canUsePasskeys: supported && platformAvailable,
  };
}

// Passkey 登録
async function registerPasskey() {
  const support = await checkWebAuthnSupport();
  if (!support.supported) {
    throw new Error('このブラウザは Passkey をサポートしていません');
  }

  // サーバーからオプション取得
  const optionsRes = await fetch('/api/webauthn/register/options', {
    method: 'POST',
    credentials: 'include',
  });
  const options = await optionsRes.json();

  try {
    // ブラウザの認証ダイアログを表示
    const registration = await startRegistration({ optionsJSON: options });

    // サーバーに検証を送信
    const verifyRes = await fetch('/api/webauthn/register/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(registration),
      credentials: 'include',
    });

    if (!verifyRes.ok) {
      throw new Error('登録の検証に失敗しました');
    }

    return verifyRes.json();
  } catch (error: any) {
    if (error.name === 'NotAllowedError') {
      throw new Error('認証がキャンセルされました');
    }
    if (error.name === 'InvalidStateError') {
      throw new Error('この認証器は既に登録されています');
    }
    throw error;
  }
}

// Passkey 認証
async function authenticateWithPasskey(email?: string) {
  const optionsRes = await fetch('/api/webauthn/authenticate/options', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  });
  const options = await optionsRes.json();

  try {
    const authentication = await startAuthentication({ optionsJSON: options });

    const verifyRes = await fetch('/api/webauthn/authenticate/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(authentication),
      credentials: 'include',
    });

    if (!verifyRes.ok) {
      throw new Error('認証に失敗しました');
    }

    return verifyRes.json();
  } catch (error: any) {
    if (error.name === 'NotAllowedError') {
      throw new Error('認証がキャンセルされました');
    }
    throw error;
  }
}

// Conditional UI（入力フィールドに Passkey 候補を表示）
async function setupConditionalUI() {
  if (!browserSupportsWebAuthn()) return;

  try {
    const optionsRes = await fetch('/api/webauthn/authenticate/options', {
      method: 'POST',
    });
    const options = await optionsRes.json();

    // mediation: 'conditional' で Conditional UI を有効化
    const authentication = await startAuthentication({
      optionsJSON: options,
      useBrowserAutofill: true, // Conditional UI を使用
    });

    // 自動的に認証が完了した場合の処理
    const verifyRes = await fetch('/api/webauthn/authenticate/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(authentication),
      credentials: 'include',
    });

    if (verifyRes.ok) {
      window.location.href = '/dashboard';
    }
  } catch (error) {
    console.log('Conditional UI not available or cancelled');
  }
}
```

### 4.4 Passkey と従来の WebAuthn の違い

```
Passkey vs 従来の WebAuthn:

  ┌───────────────────┬─────────────────┬─────────────────┐
  │ 項目              │ 従来の WebAuthn  │ Passkey          │
  ├───────────────────┼─────────────────┼─────────────────┤
  │ 鍵の同期          │ デバイスに紐付き  │ クラウド同期      │
  │ デバイス間共有     │ 不可            │ 可能             │
  │ バックアップ       │ なし            │ 自動             │
  │ デバイス紛失       │ アクセス不可     │ 他のデバイスで可   │
  │ Discoverable      │ 任意            │ 必須             │
  │ ユーザー名不要     │ 条件付き         │ 可能             │
  │ クロスデバイス認証  │ 不可            │ QR + Bluetooth   │
  └───────────────────┴─────────────────┴─────────────────┘

  Passkey のクロスデバイス認証:
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  PC（Passkey未登録） → QR コード表示              │
  │         ↕                                        │
  │  スマホ（Passkey登録済み）                         │
  │  → QR をスキャン                                  │
  │  → Bluetooth Proximity で近くにいることを確認      │
  │  → 指紋/顔認証で本人確認                          │
  │  → PC 側で認証完了                               │
  │                                                  │
  │  このフローは CTAP 2.2 Hybrid Transport で定義     │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

---

## 5. SMS / Email OTP

```
SMS OTP のリスク:

  ✗ SIM スワップ攻撃:
    → 攻撃者が携帯ショップで SIM 再発行
    → ソーシャルエンジニアリングで店員を騙す
    → 被害者の電話番号が攻撃者のデバイスに移行
    → SMS OTP が攻撃者に届く

  ✗ SS7 プロトコルの脆弱性:
    → 電話網の制御プロトコル（1975年設計）
    → 認証メカニズムが貧弱
    → SMS の傍受・リダイレクトが技術的に可能
    → 国家レベルの攻撃者が悪用可能

  ✗ フィッシング:
    → コードを偽サイトに入力させる
    → リアルタイムプロキシ攻撃で即座に使用
    → SMS のリンクフィッシングも一般的

  ✗ ソーシャルエンジニアリング:
    → 「認証コードが届いたので教えてください」
    → カスタマーサポートを装った詐取

  ✗ マルウェア:
    → Android の SMS 読み取りマルウェア
    → iOS ではリスクが低い（サンドボックス）

  それでも SMS OTP を使う場合:
  → MFA なしよりは大幅に安全（99.9% の攻撃を防ぐ）
  → ユーザーにとって最も馴染みがある
  → TOTP / WebAuthn を主、SMS をフォールバックに
  → NIST SP 800-63B では「制限付きの認証器」として許容
```

```typescript
// OTP の実装（レート制限、ブルートフォース対策付き）
import crypto from 'crypto';

class OTPService {
  constructor(
    private db: Database,
    private redis: Redis,
    private smsService: SMSService,
    private emailService: EmailService
  ) {}

  // OTP 生成・送信
  async sendOTP(userId: string, channel: 'sms' | 'email'): Promise<void> {
    const user = await this.db.user.findUnique({ where: { id: userId } });
    if (!user) throw new Error('User not found');

    // レート制限: 1分に1回まで
    const rateLimitKey = `otp:ratelimit:${userId}`;
    const isLimited = await this.redis.exists(rateLimitKey);
    if (isLimited) {
      throw new Error('Please wait before requesting a new code');
    }

    // 1日の送信上限: 10回まで
    const dailyKey = `otp:daily:${userId}:${new Date().toISOString().slice(0, 10)}`;
    const dailyCount = await this.redis.incr(dailyKey);
    if (dailyCount === 1) {
      await this.redis.expire(dailyKey, 86400); // 24時間
    }
    if (dailyCount > 10) {
      throw new Error('Daily OTP limit exceeded');
    }

    // 6桁のランダムコード
    const code = crypto.randomInt(100000, 999999).toString();

    // ハッシュ化して保存
    const hashedCode = crypto.createHash('sha256').update(code).digest('hex');

    // 既存の未使用コードを無効化
    await this.db.otpCode.updateMany({
      where: { userId, usedAt: null },
      data: { usedAt: new Date() },
    });

    await this.db.otpCode.create({
      data: {
        userId,
        code: hashedCode,
        channel,
        expiresAt: new Date(Date.now() + 10 * 60 * 1000), // 10分有効
        attempts: 0,
      },
    });

    // レート制限を設定（60秒）
    await this.redis.setex(rateLimitKey, 60, '1');

    // 送信
    if (channel === 'sms') {
      await this.smsService.send(user.phone!, `認証コード: ${code}`);
    } else {
      await this.emailService.send(user.email, {
        subject: '認証コード',
        html: `
          <p>あなたの認証コードは <strong>${code}</strong> です。</p>
          <p>このコードは10分間有効です。</p>
          <p>心当たりがない場合は、このメールを無視してください。</p>
        `,
      });
    }
  }

  // OTP 検証
  async verifyOTP(userId: string, code: string): Promise<boolean> {
    const otpRecord = await this.db.otpCode.findFirst({
      where: {
        userId,
        expiresAt: { gt: new Date() },
        usedAt: null,
      },
      orderBy: { createdAt: 'desc' },
    });

    if (!otpRecord) return false;

    // 試行回数制限（ブルートフォース対策）
    if (otpRecord.attempts >= 5) {
      await this.db.otpCode.update({
        where: { id: otpRecord.id },
        data: { usedAt: new Date() }, // 無効化
      });

      // アカウントロックアウト
      await this.lockAccount(userId, 15 * 60); // 15分

      return false;
    }

    // 試行回数をインクリメント
    await this.db.otpCode.update({
      where: { id: otpRecord.id },
      data: { attempts: { increment: 1 } },
    });

    // タイミングセーフな比較
    const hashedInput = crypto.createHash('sha256').update(code).digest('hex');
    const isValid = crypto.timingSafeEqual(
      Buffer.from(hashedInput),
      Buffer.from(otpRecord.code)
    );

    if (!isValid) return false;

    // 使用済みに
    await this.db.otpCode.update({
      where: { id: otpRecord.id },
      data: { usedAt: new Date() },
    });

    return true;
  }

  private async lockAccount(userId: string, durationSeconds: number): Promise<void> {
    await this.redis.setex(`account:locked:${userId}`, durationSeconds, '1');
  }
}
```

---

## 6. リカバリーコード

```
リカバリーコードの設計:

  目的:
  → MFAデバイスを紛失した場合のアカウント復旧
  → バックアップとしての最後の手段

  要件:
  → 8〜10個のコード（各コード1回使い切り）
  → 十分なエントロピー（推測不可能）
  → 安全な場所に保管するようユーザーに案内
  → 使用済みコードは無効化
  → ハッシュ化して保存（平文保存しない）

  エントロピーの計算:
  ┌──────────────────────────────────────────┐
  │  8文字の英数字コード（例: "a3f8-e2b1"）    │
  │  → 16進数8桁 = 4バイト = 32ビット         │
  │  → 2^32 = 約43億通り                     │
  │  → ブルートフォースには十分だが、           │
  │    試行回数制限が必須                       │
  │                                          │
  │  推奨: 10文字の英数字（a-z, 0-9）         │
  │  → 36^10 = 3.6 × 10^15                  │
  │  → 十分なエントロピー                     │
  └──────────────────────────────────────────┘
```

```typescript
// リカバリーコード生成・管理
import crypto from 'crypto';

class RecoveryCodeService {
  constructor(private db: Database, private redis: Redis) {}

  // リカバリーコード生成
  generateRecoveryCodes(count: number = 10): string[] {
    return Array.from({ length: count }, () => {
      // 10文字の安全なランダムコード（例: "a3f8-e2b1-c7d9"）
      const bytes = crypto.randomBytes(6);
      const hex = bytes.toString('hex');
      return `${hex.slice(0, 4)}-${hex.slice(4, 8)}-${hex.slice(8, 12)}`;
    });
  }

  // リカバリーコードの保存
  async saveCodes(userId: string, codes: string[]): Promise<void> {
    // 既存のコードを削除
    await this.db.recoveryCode.deleteMany({ where: { userId } });

    // ハッシュ化して保存
    const hashedCodes = codes.map((code) => ({
      userId,
      code: crypto.createHash('sha256').update(code).digest('hex'),
      used: false,
    }));

    await this.db.recoveryCode.createMany({ data: hashedCodes });
  }

  // リカバリーコードの検証
  async verifyCode(userId: string, code: string): Promise<boolean> {
    // レート制限
    const rateLimitKey = `recovery:ratelimit:${userId}`;
    const attempts = await this.redis.incr(rateLimitKey);
    if (attempts === 1) {
      await this.redis.expire(rateLimitKey, 3600); // 1時間
    }
    if (attempts > 10) {
      throw new Error('Too many recovery attempts. Try again later.');
    }

    const hashedCode = crypto.createHash('sha256').update(code).digest('hex');

    const record = await this.db.recoveryCode.findFirst({
      where: { userId, code: hashedCode, used: false },
    });

    if (!record) return false;

    // 使用済みにマーク
    await this.db.recoveryCode.update({
      where: { id: record.id },
      data: { used: true, usedAt: new Date() },
    });

    // 残りのコード数を確認
    const remaining = await this.db.recoveryCode.count({
      where: { userId, used: false },
    });

    // 残り少ない場合は通知
    if (remaining <= 2) {
      await this.notifyLowCodes(userId, remaining);
    }

    // 全て使用済みの場合は MFA を一時無効化して再設定を促す
    if (remaining === 0) {
      await this.promptMfaReset(userId);
    }

    return true;
  }

  // リカバリーコードの再生成
  async regenerateCodes(userId: string): Promise<string[]> {
    const codes = this.generateRecoveryCodes();
    await this.saveCodes(userId, codes);
    return codes;
  }

  private async notifyLowCodes(userId: string, remaining: number): Promise<void> {
    const user = await this.db.user.findUnique({ where: { id: userId } });
    if (user) {
      await emailService.send(user.email, {
        subject: 'リカバリーコードが残りわずかです',
        html: `<p>リカバリーコードの残りが ${remaining} 個です。安全のため、新しいコードを生成してください。</p>`,
      });
    }
  }

  private async promptMfaReset(userId: string): Promise<void> {
    // 一時的なMFAリセットトークンを発行
    const resetToken = crypto.randomBytes(32).toString('hex');
    await this.redis.setex(`mfa:reset:${userId}`, 3600, resetToken);
  }
}
```

---

## 7. ステップアップ認証

```
ステップアップ認証（Step-up Authentication）:

  概念:
  → 通常の操作: ベースライン認証（パスワード + TOTP）
  → 高リスク操作: 追加の認証ステップを要求

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  リスクレベルと要求する認証:                        │
  │                                                  │
  │  Level 0: パスワードのみ                           │
  │    → プロフィール閲覧、設定変更                     │
  │                                                  │
  │  Level 1: パスワード + TOTP                       │
  │    → ログイン、一般的な操作                        │
  │                                                  │
  │  Level 2: パスワード + TOTP + 再認証               │
  │    → パスワード変更、MFA設定変更、支払い情報変更     │
  │                                                  │
  │  Level 3: パスワード + TOTP + 生体認証 + 承認      │
  │    → 大額の送金、アカウント削除、管理者操作          │
  │                                                  │
  └──────────────────────────────────────────────────┘

  JWT での実装:
  → acr (Authentication Context Class Reference) クレーム
  → amr (Authentication Methods References) クレーム
  → auth_time: 最後の認証時刻
```

```typescript
// ステップアップ認証の実装
interface AuthLevel {
  level: number;
  methods: string[];
  authenticatedAt: Date;
}

class StepUpAuthService {
  // 認証レベルの定義
  private readonly AUTH_LEVELS = {
    basic: { level: 1, maxAge: 24 * 60 * 60 },      // 24時間
    elevated: { level: 2, maxAge: 15 * 60 },         // 15分
    critical: { level: 3, maxAge: 5 * 60 },          // 5分
  };

  // 操作に必要な認証レベル
  private readonly OPERATION_LEVELS: Record<string, keyof typeof this.AUTH_LEVELS> = {
    'view:profile': 'basic',
    'update:profile': 'basic',
    'change:password': 'elevated',
    'change:mfa': 'elevated',
    'delete:account': 'critical',
    'transfer:funds': 'critical',
    'admin:users': 'critical',
  };

  // 現在の認証レベルを確認
  async checkAuthLevel(
    userId: string,
    operation: string
  ): Promise<{ allowed: boolean; requiredLevel: string; currentLevel: number }> {
    const requiredLevelName = this.OPERATION_LEVELS[operation] || 'basic';
    const requiredLevel = this.AUTH_LEVELS[requiredLevelName];

    // セッションから認証情報を取得
    const authInfo = await this.getAuthInfo(userId);

    if (!authInfo) {
      return { allowed: false, requiredLevel: requiredLevelName, currentLevel: 0 };
    }

    // 認証レベルが足りるか
    const levelSufficient = authInfo.level >= requiredLevel.level;

    // 認証の鮮度が十分か
    const ageSeconds = (Date.now() - authInfo.authenticatedAt.getTime()) / 1000;
    const freshEnough = ageSeconds <= requiredLevel.maxAge;

    return {
      allowed: levelSufficient && freshEnough,
      requiredLevel: requiredLevelName,
      currentLevel: authInfo.level,
    };
  }

  // ステップアップ認証を実行
  async performStepUp(
    userId: string,
    method: 'totp' | 'webauthn' | 'password',
    credential: string
  ): Promise<boolean> {
    let verified = false;

    switch (method) {
      case 'totp':
        verified = await verifyTOTP(userId, credential);
        break;
      case 'webauthn':
        const result = await verifyAuthentication(JSON.parse(credential));
        verified = result.verified;
        break;
      case 'password':
        const user = await db.user.findUnique({ where: { id: userId } });
        verified = user ? await argon2.verify(user.password, credential) : false;
        break;
    }

    if (verified) {
      await this.upgradeAuthLevel(userId, method);
    }

    return verified;
  }

  private async upgradeAuthLevel(userId: string, method: string): Promise<void> {
    const current = await this.getAuthInfo(userId) || {
      level: 0,
      methods: [],
      authenticatedAt: new Date(),
    };

    current.level = Math.min(current.level + 1, 3);
    current.methods.push(method);
    current.authenticatedAt = new Date();

    await redis.setex(
      `auth:level:${userId}`,
      24 * 60 * 60,
      JSON.stringify(current)
    );
  }

  private async getAuthInfo(userId: string): Promise<AuthLevel | null> {
    const data = await redis.get(`auth:level:${userId}`);
    return data ? JSON.parse(data) : null;
  }
}

// Express ミドルウェアとして使用
function requireAuthLevel(operation: string) {
  return async (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    const result = await stepUpAuth.checkAuthLevel(req.user!.userId, operation);

    if (result.allowed) {
      return next();
    }

    return res.status(403).json({
      error: 'step_up_required',
      requiredLevel: result.requiredLevel,
      currentLevel: result.currentLevel,
      message: 'Additional authentication required for this operation',
    });
  };
}

// 使用例
app.post('/api/transfer',
  requireAuth(),
  requireAuthLevel('transfer:funds'),
  async (req, res) => {
    // 高リスク操作の実行
    await transferFunds(req.body);
    res.json({ success: true });
  }
);
```

---

## 8. MFA の UX 設計

```
MFA セットアップ UX:

  ① オンボーディング時に推奨（強制しない）
  ② ステップバイステップのガイド表示
  ③ QRコード + 手動入力の両方を提供
  ④ セットアップ直後にコードを検証
  ⑤ リカバリーコードを確実に保存させる

  ✓ リカバリーコード保存の確認:
    → ダウンロードボタンを提供
    → 「コードを安全な場所に保存しましたか？」チェックボックス
    → コードの一部を再入力させて確認
    → PDF/テキストファイルとしてエクスポート

MFA セットアップフロー:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  Step 1: MFA 方式の選択                       │
  │  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
  │  │ Passkey  │  │  TOTP   │  │   SMS   │      │
  │  │ (推奨)   │  │         │  │         │      │
  │  └────┬────┘  └────┬────┘  └────┬────┘      │
  │       ↓            ↓            ↓            │
  │                                              │
  │  Step 2: セットアップ                          │
  │  → Passkey: 指紋/顔認証で鍵を生成              │
  │  → TOTP: QRコードスキャン → コード入力          │
  │  → SMS: 電話番号入力 → コード入力              │
  │                                              │
  │  Step 3: 検証                                 │
  │  → 実際にコードを入力して動作確認               │
  │                                              │
  │  Step 4: リカバリーコード                      │
  │  → 10個のリカバリーコードを表示                 │
  │  → ダウンロード/コピーボタン                    │
  │  → 「保存しました」のチェック                   │
  │  → コードの一部を再入力して確認                 │
  │                                              │
  │  Step 5: 完了                                 │
  │  → MFA が有効になりました                      │
  │  → 信頼できるデバイスの登録を提案               │
  │                                              │
  └──────────────────────────────────────────────┘

MFA 認証時の UX:
  ✓ 「このデバイスを信頼する」オプション（30日間）
  ✓ 複数のMFA方式からの選択
  ✓ リカバリーコードへのフォールバック
  ✗ MFA強制でアカウントロック → 必ず復旧手段を用意
```

```typescript
// 信頼済みデバイスの管理
class TrustedDeviceService {
  constructor(private db: Database, private redis: Redis) {}

  // デバイスを信頼済みとして登録
  async trustDevice(userId: string, req: Request): Promise<string> {
    const deviceId = crypto.randomUUID();
    const hashedDeviceId = crypto.createHash('sha256').update(deviceId).digest('hex');

    const deviceInfo = {
      userId,
      hashedDeviceId,
      userAgent: req.headers['user-agent'] || 'unknown',
      ip: req.ip,
      trustedAt: new Date(),
      expiresAt: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30日
    };

    await this.db.trustedDevice.create({ data: deviceInfo });

    return deviceId;
  }

  // デバイスが信頼済みか確認
  async isDeviceTrusted(userId: string, deviceId: string): Promise<boolean> {
    if (!deviceId) return false;

    const hashedDeviceId = crypto.createHash('sha256').update(deviceId).digest('hex');

    const device = await this.db.trustedDevice.findFirst({
      where: {
        userId,
        hashedDeviceId,
        expiresAt: { gt: new Date() },
      },
    });

    return device !== null;
  }

  // Cookie でデバイスIDを管理
  setDeviceCookie(res: Response, deviceId: string): void {
    res.cookie('trusted_device', deviceId, {
      httpOnly: true,
      secure: true,
      sameSite: 'lax',
      maxAge: 30 * 24 * 60 * 60 * 1000, // 30日
      path: '/',
    });
  }
}

// ログインフローへの組み込み
async function loginWithMFA(email: string, password: string, req: Request, res: Response) {
  // Step 1: パスワード認証
  const user = await authenticatePassword(email, password);
  if (!user) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }

  // Step 2: MFA が有効な場合
  if (user.mfaEnabled) {
    // 信頼済みデバイスか確認
    const deviceId = req.cookies.trusted_device;
    const isTrusted = await trustedDeviceService.isDeviceTrusted(user.id, deviceId);

    if (isTrusted) {
      // MFA をスキップ
      const tokens = await issueTokens(user);
      return res.json(tokens);
    }

    // MFA が必要
    const mfaToken = await issueMfaToken(user.id); // 一時的なトークン（5分有効）
    return res.json({
      requireMFA: true,
      mfaToken,
      availableMethods: await getAvailableMFAMethods(user.id),
    });
  }

  // MFA なしでログイン
  const tokens = await issueTokens(user);
  return res.json(tokens);
}
```

---

## 9. アンチパターン

```
MFA 実装のアンチパターン:

  ✗ アンチパターン 1: TOTP シークレットの平文保存
    → DB が漏洩すると全ユーザーの MFA が無効化される
    → 必ず暗号化して保存（AES-256-GCM + KMS）

  ✗ アンチパターン 2: SMS OTP のみに依存
    → SIM スワップ攻撃で完全にバイパス可能
    → 少なくとも TOTP を主方式として提供

  ✗ アンチパターン 3: リカバリー手段なしの MFA 強制
    → デバイス紛失でアカウントに永久にアクセス不可
    → 必ずリカバリーコードを提供
    → サポートによるアイデンティティ検証フローも用意

  ✗ アンチパターン 4: MFA の存在を認証前に漏らす
    → 「MFA コードを入力してください」とエラー表示
    → 攻撃者にそのアカウントに MFA が有効だと伝わる
    → ユーザー列挙攻撃の手がかり

  ✗ アンチパターン 5: OTP のリプレイを許可
    → 同じ OTP コードの複数回使用を許可してしまう
    → 使用済みコードを追跡して拒否する

  ✗ アンチパターン 6: WebAuthn のカウンター検証を省略
    → クローン検知ができない
    → 認証器の複製攻撃に脆弱
```

---

## 10. パスワードレス認証の設計

```
パスワードレス認証の方式:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  1. Passkey（推奨）                               │
  │     → 公開鍵暗号ベース                            │
  │     → フィッシング耐性                            │
  │     → UX 最良（指紋/顔認証のみ）                   │
  │     → クラウド同期でデバイス間共有                  │
  │                                                  │
  │  2. Magic Link                                   │
  │     → メールにワンタイムリンクを送信                │
  │     → リンクをクリックしてログイン                  │
  │     → メールセキュリティに依存                     │
  │                                                  │
  │  3. OTP（Email / SMS）                            │
  │     → ワンタイムパスワードを送信                    │
  │     → 入力して認証                                │
  │     → フィッシングに弱い                           │
  │                                                  │
  └──────────────────────────────────────────────────┘

パスワードレスのメリット:
  → パスワード漏洩リスクの完全排除
  → クレデンシャルスタッフィング攻撃の防止
  → ユーザー体験の向上（パスワードを覚えなくてよい）
  → サポートコストの削減（パスワードリセットが不要）
```

```typescript
// Magic Link 認証の実装
class MagicLinkService {
  constructor(
    private db: Database,
    private redis: Redis,
    private emailService: EmailService
  ) {}

  // Magic Link の送信
  async sendMagicLink(email: string): Promise<void> {
    // ユーザーの存在有無に関わらず同じレスポンス
    const user = await this.db.user.findUnique({ where: { email } });

    if (!user) {
      // タイミング攻撃防止のため、同じ時間をかける
      await new Promise((resolve) => setTimeout(resolve, 200));
      return;
    }

    // レート制限
    const rateLimitKey = `magiclink:ratelimit:${email}`;
    const isLimited = await this.redis.exists(rateLimitKey);
    if (isLimited) return;

    // トークン生成
    const token = crypto.randomBytes(32).toString('base64url');
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    await this.db.magicLink.create({
      data: {
        userId: user.id,
        token: hashedToken,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000), // 15分
      },
    });

    await this.redis.setex(rateLimitKey, 60, '1');

    // メール送信
    const loginUrl = `${process.env.APP_URL}/auth/magic-link?token=${token}`;
    await this.emailService.send(email, {
      subject: 'ログインリンク',
      html: `
        <p>以下のリンクをクリックしてログインしてください（15分有効）:</p>
        <a href="${loginUrl}">ログインする</a>
        <p>心当たりがない場合は、このメールを無視してください。</p>
      `,
    });
  }

  // Magic Link の検証
  async verifyMagicLink(token: string): Promise<{ userId: string } | null> {
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    const magicLink = await this.db.magicLink.findFirst({
      where: {
        token: hashedToken,
        expiresAt: { gt: new Date() },
        usedAt: null,
      },
    });

    if (!magicLink) return null;

    // 使用済みにする
    await this.db.magicLink.update({
      where: { id: magicLink.id },
      data: { usedAt: new Date() },
    });

    return { userId: magicLink.userId };
  }
}
```

---

## 11. 演習

### 演習 1: TOTP アルゴリズムの実装（基礎）

```
課題:
  RFC 6238 に基づき、TOTP アルゴリズムをゼロから実装せよ。

  要件:
  1. HMAC-SHA1 ベースの TOTP 生成
  2. 6桁のワンタイムパスワード
  3. 30秒のタイムステップ
  4. 前後1ステップの検証ウィンドウ
  5. タイミングセーフな比較

  テストベクトル（RFC 6238 Appendix B）:
  Secret: "12345678901234567890"（ASCII）
  Time 0: OTP = 755224（RFC 4226 テストベクトル）
  Time 1: OTP = 287082

  検証: otplib ライブラリの結果と一致させる
```

### 演習 2: WebAuthn を使ったパスワードレスログイン（応用）

```
課題:
  @simplewebauthn を使って、パスワードレスログインページを実装せよ。

  要件:
  1. Passkey の登録フロー
  2. Passkey による認証フロー
  3. Conditional UI（入力フィールドに Passkey 候補表示）
  4. 複数の認証器の管理（一覧表示、削除）
  5. フォールバック（Passkey 非対応ブラウザ向け）

  技術スタック:
  → サーバー: Express + @simplewebauthn/server
  → クライアント: React + @simplewebauthn/browser
  → DB: PostgreSQL（Prisma）
```

### 演習 3: MFA ポリシーエンジンの設計（発展）

```
課題:
  組織レベルの MFA ポリシーを管理するエンジンを設計・実装せよ。

  要件:
  1. 組織ごとに MFA ポリシーを設定可能
     → 全員に MFA 強制
     → 管理者のみ MFA 強制
     → 推奨のみ（強制しない）
  2. 許可する MFA 方式の制限
     → 例: SMS を禁止、WebAuthn のみ許可
  3. ステップアップ認証ルール
     → 操作ごとに要求する認証レベルを設定
  4. リスクベース認証
     → 新しいデバイス → MFA 必須
     → 異常な地理的位置 → MFA 必須
     → 通常のアクセスパターン → 信頼デバイスなら MFA スキップ
  5. 監査ログ
     → 全ての MFA 操作をログに記録
     → 管理画面で確認可能

  データモデル:
  ┌──────────────┐  ┌──────────────┐
  │ Organization │  │ MFAPolicy    │
  ├──────────────┤  ├──────────────┤
  │ id           │  │ orgId        │
  │ name         │  │ enforcement  │
  │              │  │ allowedMethods│
  │              │  │ stepUpRules  │
  │              │  │ riskRules    │
  └──────────────┘  └──────────────┘
```

---

## 12. FAQ・トラブルシューティング

```
Q1: TOTP コードが常に不一致になる
A1: → サーバーの時刻を確認（NTP 同期）
    → ウィンドウを広げて検証（window=2）
    → クライアントの時刻も確認（スマホの自動時刻設定）
    → 秘密鍵の保存/復号が正しいか確認

Q2: WebAuthn 登録がブラウザでエラーになる
A2: → HTTPS 環境か確認（localhost は例外的に可能）
    → rpID がドメインと一致しているか確認
    → ブラウザの WebAuthn サポート状況を確認
    → Content-Security-Policy がブロックしていないか確認

Q3: Passkey がデバイス間で同期されない
A3: → iCloud Keychain / Google Password Manager が有効か確認
    → residentKey: 'required' に設定しているか確認
    → Passkey 対応のプラットフォーム認証器が使われているか確認

Q4: SMS OTP が届かない
A4: → 電話番号の形式を確認（国際形式: +81...）
    → SMS 送信サービスのクォータを確認
    → ブロックされた番号でないか確認
    → フォールバックとして音声通話オプションを提供

Q5: MFA 有効後にアカウントがロックされた
A5: → リカバリーコードで復旧
    → サポートによるアイデンティティ検証フロー
    → 管理者による MFA リセット（監査ログ必須）
    → 予防: 複数の MFA 方式の登録を推奨

Q6: リカバリーコードを紛失した
A6: → 別の MFA 方式でログイン可能か確認
    → 身分証明によるアイデンティティ検証
    → 管理者によるリセット（承認フロー付き）
    → 予防: リカバリーコード保存確認のUXを改善

Q7: MFA の導入でユーザーの離脱が増えた
A7: → MFA を強制ではなく推奨にする（段階的導入）
    → Passkey を優先的に提案（UX が最良）
    → 信頼済みデバイス機能で頻度を下げる
    → セットアップの UI/UX を改善
    → MFA のメリットをユーザーに説明する
```

---

## まとめ

| MFA方式 | セキュリティ | フィッシング耐性 | オフライン | 推奨度 |
|---------|------------|----------------|-----------|--------|
| Passkeys/WebAuthn | 最強 | あり | 可能 | 最推奨 |
| TOTP | 中程度 | なし | 可能 | 推奨 |
| Push通知 | 中程度 | 限定的 | 不可 | 良い |
| SMS OTP | 低い | なし | 不可 | 最後の手段 |

| 設計要素 | 推奨 |
|---------|------|
| 主要方式 | Passkey + TOTP のデュアル |
| リカバリー | 10個のワンタイムコード |
| 信頼デバイス | 30日間、Cookie で管理 |
| ステップアップ | 高リスク操作に追加認証 |
| OTP 保存 | ハッシュ化 + 暗号化 |
| リプレイ防止 | 使用済みコードの追跡 |

---

## 次に読むべきガイド
→ [[../02-token-auth/00-jwt-deep-dive.md]] — JWT 詳解
→ [[../02-token-auth/01-oauth2-flows.md]] — OAuth 2.0 フロー
→ [[03-session-vs-token.md]] — セッション vs トークン

---

## 参考文献
1. RFC 6238. "TOTP: Time-Based One-Time Password Algorithm." IETF, 2011.
2. RFC 4226. "HOTP: An HMAC-Based One-Time Password Algorithm." IETF, 2005.
3. W3C. "Web Authentication: An API for accessing Public Key Credentials Level 2." w3.org, 2021.
4. FIDO Alliance. "Passkeys." fidoalliance.org, 2024.
5. NIST. "SP 800-63B §5.1.3: Out-of-Band Devices." nist.gov, 2020.
6. OWASP. "Multifactor Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
7. Apple. "Supporting Passkeys." developer.apple.com, 2024.
8. Google. "Passkeys on Android." developers.google.com, 2024.
9. Yubico. "WebAuthn Developer Guide." developers.yubico.com, 2024.
10. FIDO Alliance. "CTAP 2.2 Specification." fidoalliance.org, 2023.
