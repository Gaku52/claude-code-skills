# 多要素認証（MFA）

> パスワード単体では不十分な時代。TOTP、WebAuthn/Passkeys、SMS認証、リカバリーコードまで、多要素認証の仕組みと安全な実装方法を解説する。パスワードレス認証の未来も見据える。

## この章で学ぶこと

- [ ] MFAの種類と各方式のセキュリティ強度を理解する
- [ ] TOTPとWebAuthn/Passkeysの仕組みを把握する
- [ ] MFAの実装とリカバリー戦略を設計できるようになる

---

## 1. MFAの方式比較

```
MFA方式の比較:

  方式          │ セキュリティ │ UX    │ フィッシング │ コスト
  ─────────────┼────────────┼──────┼───────────┼───────
  SMS OTP      │ △ 低い     │ ○ 良  │ ✗ 弱い     │ 通信費
  Email OTP    │ △ 低い     │ ○ 良  │ ✗ 弱い     │ 無料
  TOTP         │ ○ 中程度   │ ○ 良  │ △ やや弱い  │ 無料
  Push通知     │ ○ 中程度   │ ◎ 最良│ △ やや弱い  │ 有料
  FIDO2/       │ ◎ 最強    │ ◎ 最良│ ✓ 耐性あり  │ キー代
  WebAuthn     │            │       │            │
  Passkeys     │ ◎ 最強    │ ◎ 最良│ ✓ 耐性あり  │ 無料

推奨の優先順位:
  1. Passkeys / WebAuthn（最も安全 + UX最良）
  2. TOTP（広く普及、無料）
  3. Push 通知（UX が良い）
  4. SMS OTP（最後の手段）
```

---

## 2. TOTP（Time-based One-Time Password）

```
TOTPの仕組み:

  セットアップ:
    ① サーバーが秘密鍵（secret）を生成
    ② QRコードとして表示（otpauth:// URI）
    ③ ユーザーが認証アプリ（Google Authenticator等）でスキャン
    ④ ユーザーが表示された6桁コードを入力して検証

  認証時:
    ① ユーザーが認証アプリに表示された6桁コードを入力
    ② サーバーが同じアルゴリズムでコードを計算
    ③ 一致すれば認証成功

  アルゴリズム（RFC 6238）:
    TOTP = HOTP(secret, floor(unixTime / 30))
    → 30秒ごとに新しいコードが生成される
    → サーバーは前後1ステップ（±30秒）も許容

  QRコード URI:
    otpauth://totp/MyApp:alice@example.com
      ?secret=JBSWY3DPEHPK3PXP
      &issuer=MyApp
      &algorithm=SHA1
      &digits=6
      &period=30
```

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

  // ウィンドウ=1: 前後30秒のコードも許容
  return authenticator.verify({ token, secret });
}
```

---

## 3. WebAuthn / Passkeys

```
WebAuthn の仕組み:

  公開鍵暗号ベースの認証:
    → パスワードを使わない
    → 秘密鍵はデバイスに保存（サーバーには送信されない）
    → フィッシング完全耐性（オリジン検証あり）

  登録フロー:
    ① サーバー: チャレンジ（ランダム値）を生成
    ② ブラウザ: navigator.credentials.create() を呼出
    ③ 認証器: ユーザー検証（指紋/顔/PIN）→ 鍵ペア生成
    ④ ブラウザ: 公開鍵 + 署名付きデータをサーバーに送信
    ⑤ サーバー: 公開鍵を保存

  認証フロー:
    ① サーバー: チャレンジを生成
    ② ブラウザ: navigator.credentials.get() を呼出
    ③ 認証器: ユーザー検証 → チャレンジに秘密鍵で署名
    ④ ブラウザ: 署名をサーバーに送信
    ⑤ サーバー: 保存済み公開鍵で署名を検証

  Passkeys（WebAuthn の進化）:
    → iCloud Keychain / Google Password Manager で同期
    → デバイス間で使える（iPhone で登録 → Mac で使用）
    → パスワードレス認証のデファクトスタンダード

  なぜフィッシング耐性があるか:
    → 認証器がオリジン（ドメイン）を検証
    → example.com で登録した鍵は evil.com では使えない
    → ユーザーが偽サイトにアクセスしても鍵が動作しない
```

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

  // チャレンジを一時保存
  await db.challenge.upsert({
    where: { userId },
    create: { userId, challenge: options.challenge },
    update: { challenge: options.challenge },
  });

  return options;
}

// 登録: レスポンス検証
async function verifyRegistration(userId: string, response: any) {
  const challenge = await db.challenge.findUnique({ where: { userId } });

  const verification = await verifyRegistrationResponse({
    response,
    expectedChallenge: challenge!.challenge,
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
      },
    });
  }

  return verification;
}
```

```typescript
// クライアント側: WebAuthn 登録
import {
  startRegistration,
  startAuthentication,
} from '@simplewebauthn/browser';

// 登録
async function registerPasskey() {
  // サーバーからオプション取得
  const optionsRes = await fetch('/api/webauthn/register/options');
  const options = await optionsRes.json();

  // ブラウザの認証ダイアログを表示
  const registration = await startRegistration({ optionsJSON: options });

  // サーバーに検証を送信
  const verifyRes = await fetch('/api/webauthn/register/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(registration),
  });

  return verifyRes.json();
}

// 認証
async function authenticateWithPasskey() {
  const optionsRes = await fetch('/api/webauthn/authenticate/options');
  const options = await optionsRes.json();

  const authentication = await startAuthentication({ optionsJSON: options });

  const verifyRes = await fetch('/api/webauthn/authenticate/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(authentication),
  });

  return verifyRes.json();
}
```

---

## 4. SMS / Email OTP

```
SMS OTP のリスク:

  ✗ SIM スワップ攻撃: 攻撃者が電話番号を乗っ取り
  ✗ SS7 プロトコルの脆弱性: 通信の傍受が可能
  ✗ フィッシング: コードを偽サイトに入力させる
  ✗ ソーシャルエンジニアリング: 携帯ショップで SIM 再発行

  それでも SMS OTP を使う場合:
  → MFA なしよりは大幅に安全
  → ユーザーにとって最も馴染みがある
  → TOTP / WebAuthn を主、SMS をフォールバックに
```

```typescript
// OTP の実装
import crypto from 'crypto';

// OTP 生成・送信
async function sendOTP(userId: string, channel: 'sms' | 'email') {
  // 6桁のランダムコード
  const code = crypto.randomInt(100000, 999999).toString();

  // ハッシュ化して保存
  const hashedCode = crypto.createHash('sha256').update(code).digest('hex');

  await db.otpCode.create({
    data: {
      userId,
      code: hashedCode,
      channel,
      expiresAt: new Date(Date.now() + 10 * 60 * 1000), // 10分有効
      attempts: 0,
    },
  });

  // 送信
  if (channel === 'sms') {
    await smsService.send(user.phone, `認証コード: ${code}`);
  } else {
    await emailService.send(user.email, `認証コード: ${code}`);
  }
}

// OTP 検証
async function verifyOTP(userId: string, code: string): Promise<boolean> {
  const otpRecord = await db.otpCode.findFirst({
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
    await db.otpCode.update({
      where: { id: otpRecord.id },
      data: { usedAt: new Date() }, // 無効化
    });
    return false;
  }

  // 試行回数をインクリメント
  await db.otpCode.update({
    where: { id: otpRecord.id },
    data: { attempts: { increment: 1 } },
  });

  const hashedInput = crypto.createHash('sha256').update(code).digest('hex');

  if (hashedInput !== otpRecord.code) return false;

  // 使用済みに
  await db.otpCode.update({
    where: { id: otpRecord.id },
    data: { usedAt: new Date() },
  });

  return true;
}
```

---

## 5. リカバリーコード

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
```

```typescript
// リカバリーコード生成
import crypto from 'crypto';

function generateRecoveryCodes(count = 10): string[] {
  return Array.from({ length: count }, () => {
    // 8文字の英数字コード（例: "a3f8-e2b1"）
    const bytes = crypto.randomBytes(4);
    const code = bytes.toString('hex');
    return `${code.slice(0, 4)}-${code.slice(4, 8)}`;
  });
}

async function saveRecoveryCodes(userId: string, codes: string[]) {
  // 既存のコードを削除
  await db.recoveryCode.deleteMany({ where: { userId } });

  // ハッシュ化して保存
  const hashedCodes = codes.map((code) => ({
    userId,
    code: crypto.createHash('sha256').update(code).digest('hex'),
    used: false,
  }));

  await db.recoveryCode.createMany({ data: hashedCodes });
}

async function verifyRecoveryCode(userId: string, code: string): Promise<boolean> {
  const hashedCode = crypto.createHash('sha256').update(code).digest('hex');

  const record = await db.recoveryCode.findFirst({
    where: { userId, code: hashedCode, used: false },
  });

  if (!record) return false;

  // 使用済みにマーク
  await db.recoveryCode.update({
    where: { id: record.id },
    data: { used: true },
  });

  // 残りのコード数を確認
  const remaining = await db.recoveryCode.count({
    where: { userId, used: false },
  });

  // 残り少ない場合は通知
  if (remaining <= 2) {
    await notifyLowRecoveryCodes(userId, remaining);
  }

  return true;
}
```

---

## 6. MFA の UX 設計

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

MFA 認証時の UX:
  ✓ 「このデバイスを信頼する」オプション（30日間）
  ✓ 複数のMFA方式からの選択
  ✓ リカバリーコードへのフォールバック
  ✗ MFA強制でアカウントロック → 必ず復旧手段を用意
```

---

## まとめ

| MFA方式 | セキュリティ | フィッシング耐性 | 推奨度 |
|---------|------------|----------------|--------|
| Passkeys/WebAuthn | 最強 | あり | 最推奨 |
| TOTP | 中程度 | なし | 推奨 |
| Push通知 | 中程度 | なし | 良い |
| SMS OTP | 低い | なし | 最後の手段 |

---

## 次に読むべきガイド
→ [[03-session-vs-token.md]] — セッション vs トークン

---

## 参考文献
1. RFC 6238. "TOTP: Time-Based One-Time Password Algorithm." IETF, 2011.
2. W3C. "Web Authentication: An API for accessing Public Key Credentials." w3.org, 2021.
3. FIDO Alliance. "Passkeys." fidoalliance.org, 2024.
4. NIST. "SP 800-63B §5.1.3: Out-of-Band Devices." nist.gov, 2020.
