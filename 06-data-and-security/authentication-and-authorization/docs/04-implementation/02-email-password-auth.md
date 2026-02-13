# メール・パスワード認証

> ソーシャルログインだけでは不十分な場面で必要となるメール・パスワード認証。ユーザー登録、メール確認、ログイン、パスワードリセット、アカウントロックまで、安全なメール認証の完全フローを解説する。

## 前提知識

- [[../../01-session-auth/00-cookie-and-session.md]] — Cookie とセッション管理
- [[../../01-session-auth/01-session-store.md]] — セッションストア
- HTTP の基礎（POST リクエスト、ステータスコード）
- TypeScript / JavaScript の基礎
- データベースの基本操作（Prisma）

## この章で学ぶこと

- [ ] ユーザー登録とメール確認の安全なフローを実装する
- [ ] パスワードハッシュ化の内部実装と bcrypt/Argon2 の使い分けを理解する
- [ ] 安全なログインとレート制限の設計・実装を把握する
- [ ] パスワードリセットとアカウント保護の完全なフローを設計できるようになる
- [ ] ユーザー列挙攻撃やタイミング攻撃への対策を講じられる
- [ ] NIST SP 800-63B に準拠したパスワードポリシーを設計できる

---

## 1. パスワードハッシュ化の基礎

### 1.1 なぜハッシュ化が必要か

パスワードを平文で保存してはならない。データベースが漏洩した場合、全ユーザーのパスワードが攻撃者に露出する。ハッシュ化により、漏洩しても元のパスワードを復元できないようにする。

```
パスワード保存の進化:

  ✗ Level 0: 平文保存
    password: "MySecret123"
    → DB 漏洩で即座に全パスワード露出

  ✗ Level 1: 単純ハッシュ（MD5/SHA-256）
    hash: SHA256("MySecret123")
    → レインボーテーブル攻撃で突破可能

  ✗ Level 2: ソルト付きハッシュ
    hash: SHA256("random_salt" + "MySecret123")
    → GPUで高速に総当たり可能（SHA-256は高速すぎる）

  ✓ Level 3: 専用ハッシュ関数（bcrypt/Argon2）
    hash: bcrypt("MySecret123", cost=12)
    → 意図的に低速化されたハッシュ関数
    → 総当たり攻撃のコストが非常に高い

  パスワードハッシュ関数の内部動作:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  bcrypt の構造:                                │
  │  $2b$12$LJ3m4ysKlcWBzBH8PsYBte.JZj2gLSf...   │
  │   │  │  │                    │                │
  │   │  │  │                    └─ ハッシュ値     │
  │   │  │  └─ ソルト（22文字 Base64）              │
  │   │  └─ コストファクター（2^12 = 4096回）       │
  │   └─ アルゴリズム識別子（2b = bcrypt）           │
  │                                              │
  │  Argon2id の構造:                              │
  │  $argon2id$v=19$m=65536,t=3,p=4$salt$hash    │
  │   │        │    │       │  │                  │
  │   │        │    │       │  └─ 並列度           │
  │   │        │    │       └─ 反復回数             │
  │   │        │    └─ メモリ使用量（KB）            │
  │   │        └─ バージョン                        │
  │   └─ アルゴリズム識別子                          │
  │                                              │
  └──────────────────────────────────────────────┘
```

### 1.2 bcrypt vs Argon2 の比較

```
パスワードハッシュ関数の比較:

  項目           │ bcrypt          │ Argon2id        │ scrypt
  ──────────────┼────────────────┼────────────────┼────────────────
  設計年         │ 1999            │ 2015            │ 2009
  メモリハード   │ ✗               │ ✓（主要な利点）  │ ✓
  GPU 耐性      │ 中              │ 高              │ 高
  設定の容易さ   │ コスト1つ        │ 3つのパラメータ  │ 3つのパラメータ
  ライブラリ     │ 豊富            │ 増加中          │ 中程度
  推奨ユース     │ 既存システム     │ 新規システム     │ 暗号通貨で多い
  OWASP 推奨    │ ✓（代替）       │ ✓（第一推奨）   │ ✓（代替）
  標準化        │ ─               │ PHC Winner      │ RFC 7914

  推奨設定:
    bcrypt:    cost = 12 （ログインに 250ms 程度）
    Argon2id:  m=65536 (64MB), t=3, p=4
    → サーバーのスペックに合わせて調整
    → ログイン処理が 250ms-1s になるよう設定

  重要: MD5, SHA-1, SHA-256 はパスワードハッシュに使用してはならない
  → これらは高速ハッシュであり、パスワード用ではない
```

### 1.3 パスワードハッシュの実装

```typescript
// bcrypt でのパスワードハッシュ化
import bcrypt from 'bcrypt';

// ハッシュ化（登録時）
const BCRYPT_ROUNDS = 12; // コストファクター

async function hashPassword(password: string): Promise<string> {
  // bcrypt は自動でソルトを生成
  // $2b$12$[22文字のソルト][31文字のハッシュ]
  return bcrypt.hash(password, BCRYPT_ROUNDS);
}

// 検証（ログイン時）
async function verifyPassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

// Argon2id でのパスワードハッシュ化
import argon2 from 'argon2';

async function hashPasswordArgon2(password: string): Promise<string> {
  return argon2.hash(password, {
    type: argon2.argon2id,    // Argon2id（推奨バリアント）
    memoryCost: 65536,        // 64MB のメモリ使用
    timeCost: 3,              // 3回の反復
    parallelism: 4,           // 4並列
  });
}

async function verifyPasswordArgon2(password: string, hash: string): Promise<boolean> {
  return argon2.verify(hash, password);
}

// ハッシュ関数の自動判別（マイグレーション対応）
async function verifyPasswordAuto(password: string, hash: string): Promise<{
  valid: boolean;
  needsRehash: boolean;
}> {
  let valid: boolean;
  let needsRehash = false;

  if (hash.startsWith('$argon2')) {
    valid = await argon2.verify(hash, password);
    needsRehash = argon2.needsRehash(hash, {
      type: argon2.argon2id,
      memoryCost: 65536,
      timeCost: 3,
      parallelism: 4,
    });
  } else if (hash.startsWith('$2')) {
    valid = await bcrypt.compare(password, hash);
    // bcrypt から Argon2 への移行を示す
    needsRehash = true;
  } else {
    throw new Error('Unknown hash format');
  }

  return { valid, needsRehash };
}

// ログイン時のハッシュ自動アップグレード
async function loginWithHashUpgrade(email: string, password: string) {
  const user = await prisma.user.findUnique({ where: { email } });
  if (!user?.password) return null;

  const { valid, needsRehash } = await verifyPasswordAuto(password, user.password);
  if (!valid) return null;

  // ハッシュのアップグレード（バックグラウンドで実行）
  if (needsRehash) {
    const newHash = await hashPasswordArgon2(password);
    await prisma.user.update({
      where: { id: user.id },
      data: { password: newHash },
    });
  }

  return user;
}
```

---

## 2. NIST SP 800-63B に基づくパスワードポリシー

### 2.1 現代のパスワードポリシー

```
NIST SP 800-63B の推奨事項（2020年改訂）:

  ✓ すべきこと:
    → 最低8文字を要求（推奨は最低15文字）
    → 最大64文字以上を許容
    → Unicode文字を許容（日本語パスワード等）
    → 漏洩パスワードリストとの照合（haveibeenpwned API）
    → パスワード強度メーターの提供
    → ペーストの許可（パスワードマネージャー対応）

  ✗ すべきでないこと:
    → 定期的なパスワード変更の強制
    → 複雑さの要件（大文字/小文字/数字/記号の組合せ）
    → セキュリティの質問
    → パスワードヒント

  理由:
  → 複雑さの要件は弱いパスワードのパターン化を招く
    （例: Password1! → 覚えやすいが弱い）
  → 定期変更は微小な変更を招く
    （例: MyPass1 → MyPass2 → MyPass3）
  → 長いパスフレーズの方が安全
    （例: "correct horse battery staple" = 高いエントロピー）
```

### 2.2 パスワードバリデーションの実装

```typescript
// NIST準拠のパスワードバリデーション
import { z } from 'zod';

// 漏洩パスワードチェック（Have I Been Pwned API）
async function isPasswordBreached(password: string): Promise<boolean> {
  const hash = crypto.createHash('sha1').update(password).digest('hex').toUpperCase();
  const prefix = hash.substring(0, 5);
  const suffix = hash.substring(5);

  // k-Anonymity: プレフィックスのみ送信（パスワード自体は送信しない）
  const res = await fetch(`https://api.pwnedpasswords.com/range/${prefix}`);
  const text = await res.text();

  // レスポンスからサフィックスを検索
  return text.split('\n').some((line) => {
    const [hashSuffix, count] = line.split(':');
    return hashSuffix.trim() === suffix;
  });
}

// パスワード強度の計算
function calculatePasswordStrength(password: string): {
  score: number; // 0-4
  feedback: string[];
} {
  const feedback: string[] = [];
  let score = 0;

  // 長さ
  if (password.length >= 8) score += 1;
  if (password.length >= 12) score += 1;
  if (password.length >= 16) score += 1;
  if (password.length < 8) feedback.push('8文字以上にしてください');

  // 文字種の多様性
  const hasLower = /[a-z]/.test(password);
  const hasUpper = /[A-Z]/.test(password);
  const hasDigit = /[0-9]/.test(password);
  const hasSymbol = /[^a-zA-Z0-9]/.test(password);
  const charTypes = [hasLower, hasUpper, hasDigit, hasSymbol].filter(Boolean).length;
  if (charTypes >= 3) score += 1;

  // 繰り返し文字
  if (/(.)\1{2,}/.test(password)) {
    feedback.push('同じ文字の繰り返しを避けてください');
  }

  // 一般的なパターン
  const commonPatterns = [
    /^123456/,
    /^password/i,
    /^qwerty/i,
    /^abcdef/i,
  ];
  if (commonPatterns.some((p) => p.test(password))) {
    score = Math.max(0, score - 2);
    feedback.push('一般的なパスワードパターンを避けてください');
  }

  return { score: Math.min(4, Math.max(0, score)), feedback };
}

// 登録フォームのバリデーション
const registerSchema = z.object({
  name: z.string().min(1, '名前を入力してください').max(100),
  email: z.string().email('有効なメールアドレスを入力してください'),
  password: z.string()
    .min(8, 'パスワードは8文字以上必要です')
    .max(128, 'パスワードは128文字以下にしてください')
    .refine(
      (val) => !/(.)\1{2,}/.test(val),
      '同じ文字を3回以上連続して使用できません'
    ),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: 'パスワードが一致しません',
  path: ['confirmPassword'],
});
```

---

## 3. ユーザー登録

### 3.1 登録フローの全体像

```
ユーザー登録フロー:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ユーザー                   サーバー               │
  │    │                         │                    │
  │    │ 登録フォーム送信          │                    │
  │    │ (name, email, password) │                    │
  │    │────────────────────────>│                    │
  │    │                         │                    │
  │    │                    バリデーション               │
  │    │                    ├─ フォーマット検証          │
  │    │                    ├─ パスワード強度チェック     │
  │    │                    ├─ 漏洩パスワードチェック     │
  │    │                    └─ メール重複チェック         │
  │    │                         │                    │
  │    │                    パスワードハッシュ化          │
  │    │                    (bcrypt/Argon2)            │
  │    │                         │                    │
  │    │                    ユーザー作成(未確認)         │
  │    │                         │                    │
  │    │                    確認トークン生成             │
  │    │                    (crypto.randomBytes)       │
  │    │                         │                    │
  │    │                    確認メール送信               │
  │    │                         │                    │
  │    │ 「確認メールを送信しました」│                    │
  │    │<────────────────────────│                    │
  │    │                         │                    │
  │    │ メール内リンクをクリック   │                    │
  │    │────────────────────────>│                    │
  │    │                         │                    │
  │    │                    トークン検証                 │
  │    │                    emailVerified = true        │
  │    │                    トークン削除                 │
  │    │                         │                    │
  │    │ 「確認完了」             │                    │
  │    │<────────────────────────│                    │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 3.2 登録の実装

```typescript
// 登録 Server Action
'use server';
import bcrypt from 'bcrypt';
import crypto from 'crypto';

async function register(formData: FormData) {
  // 1. バリデーション
  const parsed = registerSchema.safeParse({
    name: formData.get('name'),
    email: formData.get('email'),
    password: formData.get('password'),
    confirmPassword: formData.get('confirmPassword'),
  });

  if (!parsed.success) {
    return { error: parsed.error.flatten().fieldErrors };
  }

  const { name, email, password } = parsed.data;

  // 2. 漏洩パスワードチェック
  const breached = await isPasswordBreached(password);
  if (breached) {
    return {
      error: {
        password: ['このパスワードは漏洩データベースに含まれています。別のパスワードを使用してください'],
      },
    };
  }

  // 3. メールの重複チェック
  const existingUser = await prisma.user.findUnique({ where: { email } });
  if (existingUser) {
    // ユーザー列挙攻撃を防止するため、同じメッセージを返す
    // 既存ユーザーには「このメールは既に登録されています」メールを送信
    if (existingUser.emailVerified) {
      await sendEmail({
        to: email,
        subject: 'アカウント登録の試行',
        html: `
          <p>${email} で既にアカウントが登録されています。</p>
          <p>ログインは <a href="${process.env.APP_URL}/login">こちら</a> から。</p>
          <p>パスワードをお忘れの場合は <a href="${process.env.APP_URL}/forgot-password">リセット</a> してください。</p>
        `,
      });
    }
    return { success: true, message: '確認メールを送信しました' };
  }

  // 4. パスワードハッシュ化
  const hashedPassword = await bcrypt.hash(password, 12);

  // 5. ユーザー作成
  const user = await prisma.user.create({
    data: {
      name,
      email,
      password: hashedPassword,
      role: 'viewer',
      emailVerified: null, // 未確認
    },
  });

  // 6. メール確認トークン生成
  const verificationToken = crypto.randomBytes(32).toString('hex');
  const hashedToken = crypto.createHash('sha256').update(verificationToken).digest('hex');

  await prisma.verificationToken.create({
    data: {
      identifier: email,
      token: hashedToken,
      expires: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24時間
    },
  });

  // 7. 確認メール送信
  await sendEmail({
    to: email,
    subject: 'メールアドレスの確認',
    html: `
      <h1>メールアドレスの確認</h1>
      <p>${name} さん、ご登録ありがとうございます。</p>
      <p>以下のリンクをクリックしてメールアドレスを確認してください：</p>
      <a href="${process.env.APP_URL}/verify-email?token=${verificationToken}">
        メールアドレスを確認
      </a>
      <p>このリンクは24時間有効です。</p>
      <p>このメールに心当たりがない場合は無視してください。</p>
    `,
  });

  return { success: true, message: '確認メールを送信しました' };
}
```

### 3.3 ユーザー列挙攻撃への対策

```
ユーザー列挙攻撃（User Enumeration）:

  攻撃手法:
  ┌────────────────────────────────────────────┐
  │                                            │
  │  攻撃者がメールアドレスの存在を確認する手法:    │
  │                                            │
  │  (1) 登録時のエラーメッセージ                  │
  │     ✗ 「このメールは既に登録されています」       │
  │     → メールの存在を確認できてしまう             │
  │                                            │
  │  (2) ログイン時のエラーメッセージ               │
  │     ✗ 「メールアドレスが見つかりません」          │
  │     ✗ 「パスワードが間違っています」              │
  │     → どちらが間違いかで存在を判定               │
  │                                            │
  │  (3) パスワードリセット                        │
  │     ✗ 「このメールは登録されていません」          │
  │     → メールの存在を確認できてしまう             │
  │                                            │
  │  (4) レスポンス時間の差                        │
  │     ✗ 存在するメール: ハッシュ比較で遅い         │
  │     ✗ 存在しないメール: DB検索のみで速い         │
  │     → タイミング攻撃で存在を判定                 │
  │                                            │
  └────────────────────────────────────────────┘

  対策:
  → 全てのケースで同一のレスポンスメッセージ
  → 全てのケースで同一のレスポンス時間（ダミー処理）
  → メール送信の有無は外部から観察不可能
```

```typescript
// タイミング攻撃対策
async function loginSafe(email: string, password: string) {
  const user = await prisma.user.findUnique({ where: { email } });

  if (!user?.password) {
    // ユーザーが存在しなくても bcrypt.compare を実行
    // → レスポンス時間を均一にしてタイミング攻撃を防止
    await bcrypt.compare(password, '$2b$12$dummy.hash.for.timing.protection');
    return { error: 'メールアドレスまたはパスワードが正しくありません' };
  }

  const isValid = await bcrypt.compare(password, user.password);
  if (!isValid) {
    return { error: 'メールアドレスまたはパスワードが正しくありません' };
  }

  return { user };
}
```

---

## 4. メール確認

### 4.1 メール確認の重要性

メール確認はなぜ必要か。(1) メールアドレスの所有権を検証する。(2) 他人のメールでアカウントが作成されるのを防ぐ。(3) パスワードリセット機能の安全性を担保する。(4) コミュニケーション経路を確保する。

```typescript
// メール確認処理
async function verifyEmail(token: string) {
  const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

  const verificationToken = await prisma.verificationToken.findFirst({
    where: {
      token: hashedToken,
      expires: { gt: new Date() },
    },
  });

  if (!verificationToken) {
    return { error: '無効または期限切れのリンクです' };
  }

  // トランザクションでメールを確認済みに更新 + トークン削除
  await prisma.$transaction([
    prisma.user.update({
      where: { email: verificationToken.identifier },
      data: { emailVerified: new Date() },
    }),
    prisma.verificationToken.delete({
      where: { id: verificationToken.id },
    }),
  ]);

  return { success: true };
}

// メール確認の再送信
async function resendVerificationEmail(email: string) {
  const user = await prisma.user.findUnique({ where: { email } });

  // ユーザーが存在しない or 既に確認済みでも同じレスポンス
  if (!user || user.emailVerified) {
    return { message: '確認メールを送信しました（メールが登録されている場合）' };
  }

  // レート制限: 同じメールへの再送は1時間に3回まで
  const recentTokens = await prisma.verificationToken.count({
    where: {
      identifier: email,
      expires: { gt: new Date(Date.now() - 60 * 60 * 1000) },
    },
  });

  if (recentTokens >= 3) {
    return { message: '確認メールを送信しました（メールが登録されている場合）' };
  }

  // 既存トークンを削除
  await prisma.verificationToken.deleteMany({
    where: { identifier: email },
  });

  // 新しいトークン生成
  const verificationToken = crypto.randomBytes(32).toString('hex');
  const hashedToken = crypto.createHash('sha256').update(verificationToken).digest('hex');

  await prisma.verificationToken.create({
    data: {
      identifier: email,
      token: hashedToken,
      expires: new Date(Date.now() + 24 * 60 * 60 * 1000),
    },
  });

  await sendEmail({
    to: email,
    subject: 'メールアドレスの確認',
    html: `
      <h1>メールアドレスの確認</h1>
      <p>以下のリンクをクリックしてメールアドレスを確認してください：</p>
      <a href="${process.env.APP_URL}/verify-email?token=${verificationToken}">
        メールアドレスを確認
      </a>
      <p>このリンクは24時間有効です。</p>
    `,
  });

  return { message: '確認メールを送信しました（メールが登録されている場合）' };
}

// メール確認ページ
async function VerifyEmailPage({ searchParams }: { searchParams: { token?: string } }) {
  if (!searchParams.token) {
    return <p>無効なリンクです。</p>;
  }

  const result = await verifyEmail(searchParams.token);

  if (result.error) {
    return (
      <div>
        <h1>確認に失敗しました</h1>
        <p>{result.error}</p>
        <Link href="/resend-verification">確認メールを再送信</Link>
      </div>
    );
  }

  return (
    <div>
      <h1>メールアドレスが確認されました</h1>
      <Link href="/login">ログインする</Link>
    </div>
  );
}
```

---

## 5. ログインとレート制限

### 5.1 ログインフローの全体像

```
ログインフロー:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ユーザー                   サーバー               │
  │    │                         │                    │
  │    │ ログイン送信              │                    │
  │    │ (email, password)       │                    │
  │    │────────────────────────>│                    │
  │    │                         │                    │
  │    │                    ① レート制限チェック         │
  │    │                    ├─ IP ベース（15分/5回）    │
  │    │                    └─ メールベース（15分/5回）  │
  │    │                         │                    │
  │    │                    ② ユーザー取得              │
  │    │                         │                    │
  │    │                    ③ アカウントロックチェック    │
  │    │                         │                    │
  │    │                    ④ パスワード検証             │
  │    │                    (bcrypt.compare)           │
  │    │                         │                    │
  │    │                    ⑤ メール確認チェック          │
  │    │                         │                    │
  │    │                    ⑥ 失敗カウントリセット       │
  │    │                    ⑦ セッション作成             │
  │    │                    ⑧ セキュリティ通知           │
  │    │                         │                    │
  │    │ Set-Cookie: session    │                    │
  │    │<────────────────────────│                    │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 5.2 多層レート制限の実装

```typescript
// 多層レート制限の設計
// Layer 1: グローバルレート制限（IP ベース）
// Layer 2: アカウントレート制限（メールベース）
// Layer 3: アカウントロック（DB ベース）

interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: Date;
  retryAfter?: number; // 秒
}

class LoginRateLimiter {
  constructor(private redis: Redis) {}

  // Layer 1: IP ベースのレート制限
  async checkIPLimit(ip: string): Promise<RateLimitResult> {
    const key = `login:ip:${ip}`;
    const limit = 20;     // 15分間に20回まで
    const window = 900;   // 15分

    const current = await this.redis.incr(key);
    if (current === 1) {
      await this.redis.expire(key, window);
    }

    const ttl = await this.redis.ttl(key);

    return {
      allowed: current <= limit,
      remaining: Math.max(0, limit - current),
      resetAt: new Date(Date.now() + ttl * 1000),
      retryAfter: current > limit ? ttl : undefined,
    };
  }

  // Layer 2: メールベースのレート制限
  async checkEmailLimit(email: string): Promise<RateLimitResult> {
    const key = `login:email:${email}`;
    const limit = 5;      // 15分間に5回まで
    const window = 900;   // 15分

    const current = await this.redis.incr(key);
    if (current === 1) {
      await this.redis.expire(key, window);
    }

    const ttl = await this.redis.ttl(key);

    return {
      allowed: current <= limit,
      remaining: Math.max(0, limit - current),
      resetAt: new Date(Date.now() + ttl * 1000),
      retryAfter: current > limit ? ttl : undefined,
    };
  }

  // 成功時にカウントをリセット
  async resetOnSuccess(email: string): Promise<void> {
    await this.redis.del(`login:email:${email}`);
  }
}

// ログイン処理（Auth.js Credentials プロバイダー）
async function authorize(credentials: { email: string; password: string }, req: Request) {
  const { email, password } = credentials;
  const ip = getClientIP(req);

  // Layer 1: IP レート制限
  const ipLimit = await rateLimiter.checkIPLimit(ip);
  if (!ipLimit.allowed) {
    throw new Error(`Too many requests. Try again in ${ipLimit.retryAfter} seconds.`);
  }

  // Layer 2: メールレート制限
  const emailLimit = await rateLimiter.checkEmailLimit(email);
  if (!emailLimit.allowed) {
    throw new Error(`Too many login attempts. Try again in ${emailLimit.retryAfter} seconds.`);
  }

  // ユーザー取得
  const user = await prisma.user.findUnique({ where: { email } });

  if (!user?.password) {
    // タイミング攻撃対策: ダミーの bcrypt 比較
    await bcrypt.compare(password, '$2b$12$dummy.hash.for.timing.attack.prevention.only');
    return null;
  }

  // Layer 3: アカウントロックチェック
  if (user.lockedUntil && user.lockedUntil > new Date()) {
    const remainingMinutes = Math.ceil((user.lockedUntil.getTime() - Date.now()) / 60000);
    throw new Error(`Account is locked. Try again in ${remainingMinutes} minutes.`);
  }

  // パスワード検証
  const isValid = await bcrypt.compare(password, user.password);

  if (!isValid) {
    // 失敗回数を記録
    const failedAttempts = (user.failedLoginAttempts || 0) + 1;
    const updateData: any = { failedLoginAttempts: failedAttempts };

    // 10回失敗でアカウントロック
    if (failedAttempts >= 10) {
      updateData.lockedUntil = new Date(Date.now() + 30 * 60 * 1000); // 30分

      // アカウントロック通知
      await sendSecurityNotification(user.id, 'account_locked');
    }

    await prisma.user.update({
      where: { id: user.id },
      data: updateData,
    });

    return null;
  }

  // メール未確認チェック
  if (!user.emailVerified) {
    throw new Error('Please verify your email before logging in.');
  }

  // ログイン成功: 失敗カウントリセット
  await prisma.user.update({
    where: { id: user.id },
    data: {
      failedLoginAttempts: 0,
      lockedUntil: null,
      lastLoginAt: new Date(),
      lastLoginIp: ip,
    },
  });

  // レート制限カウントリセット
  await rateLimiter.resetOnSuccess(email);

  // 新しいデバイスからのログイン検知
  const knownDevice = await isKnownDevice(user.id, req);
  if (!knownDevice) {
    await sendSecurityNotification(user.id, 'new_device');
    await recordDevice(user.id, req);
  }

  return {
    id: user.id,
    email: user.email,
    name: user.name,
    image: user.image,
    role: user.role,
  };
}
```

### 5.3 デバイスフィンガープリントによる不正検知

```typescript
// デバイスフィンガープリント（簡易版）
interface DeviceFingerprint {
  userAgent: string;
  ipPrefix: string; // /24 サブネット
  acceptLanguage: string;
}

function generateDeviceFingerprint(req: Request): string {
  const fp: DeviceFingerprint = {
    userAgent: req.headers.get('user-agent') || '',
    ipPrefix: getClientIP(req).split('.').slice(0, 3).join('.'), // /24
    acceptLanguage: req.headers.get('accept-language') || '',
  };

  return crypto
    .createHash('sha256')
    .update(JSON.stringify(fp))
    .digest('hex')
    .substring(0, 16);
}

async function isKnownDevice(userId: string, req: Request): Promise<boolean> {
  const fingerprint = generateDeviceFingerprint(req);

  const device = await prisma.knownDevice.findFirst({
    where: {
      userId,
      fingerprint,
      lastSeenAt: { gt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000) }, // 90日以内
    },
  });

  return !!device;
}

async function recordDevice(userId: string, req: Request): Promise<void> {
  const fingerprint = generateDeviceFingerprint(req);

  await prisma.knownDevice.upsert({
    where: { userId_fingerprint: { userId, fingerprint } },
    create: {
      userId,
      fingerprint,
      userAgent: req.headers.get('user-agent') || '',
      ipAddress: getClientIP(req),
      lastSeenAt: new Date(),
    },
    update: {
      lastSeenAt: new Date(),
      ipAddress: getClientIP(req),
    },
  });
}
```

---

## 6. パスワードリセット

### 6.1 リセットフローの設計

```
パスワードリセットフロー:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ユーザー                   サーバー               │
  │    │                         │                    │
  │    │ メールアドレス送信        │                    │
  │    │────────────────────────>│                    │
  │    │                         │                    │
  │    │                    レート制限チェック           │
  │    │                    (1時間に3回まで)            │
  │    │                         │                    │
  │    │                    ユーザー検索                 │
  │    │                    ├─ 存在: トークン生成        │
  │    │                    │       メール送信          │
  │    │                    └─ 不在: 何もしない          │
  │    │                         │                    │
  │    │ 「リセットメールを送信     │                    │
  │    │  しました（登録済の場合）」│                    │
  │    │<────────────────────────│                    │
  │    │                         │                    │
  │    │                     ...メール受信...           │
  │    │                         │                    │
  │    │ リセットリンクをクリック   │                    │
  │    │────────────────────────>│                    │
  │    │                         │                    │
  │    │                    トークン検証                 │
  │    │                    (SHA-256ハッシュ比較)       │
  │    │                    有効期限チェック (1時間)      │
  │    │                         │                    │
  │    │ 新パスワード入力画面      │                    │
  │    │<────────────────────────│                    │
  │    │                         │                    │
  │    │ 新パスワード送信          │                    │
  │    │────────────────────────>│                    │
  │    │                         │                    │
  │    │                    ① 旧パスワードと同一でないか │
  │    │                    ② 新パスワードハッシュ化     │
  │    │                    ③ パスワード更新             │
  │    │                    ④ 全セッション無効化         │
  │    │                    ⑤ リセットトークン削除       │
  │    │                    ⑥ 変更通知メール送信         │
  │    │                         │                    │
  │    │ 「パスワードが変更         │                    │
  │    │  されました」             │                    │
  │    │<────────────────────────│                    │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

### 6.2 リセットの実装

```typescript
// パスワードリセット要求
'use server';

async function requestPasswordReset(email: string) {
  // レート制限
  const key = `password_reset:${email}`;
  const attempts = await redis.get(key);
  if (attempts && parseInt(attempts) >= 3) {
    // 常に同じメッセージ（ユーザー列挙防止）
    return { message: 'メールアドレスが登録されていればリセットメールを送信しました' };
  }
  await redis.incr(key);
  await redis.expire(key, 3600); // 1時間

  // ユーザーの存在に関わらず同じレスポンス
  const user = await prisma.user.findUnique({ where: { email } });

  if (user) {
    // 既存のリセットトークンを削除
    await prisma.passwordResetToken.deleteMany({
      where: { userId: user.id },
    });

    const token = crypto.randomBytes(32).toString('hex');
    const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

    await prisma.passwordResetToken.create({
      data: {
        userId: user.id,
        token: hashedToken,
        expiresAt: new Date(Date.now() + 60 * 60 * 1000), // 1時間
      },
    });

    await sendEmail({
      to: email,
      subject: 'パスワードリセット',
      html: `
        <h1>パスワードリセット</h1>
        <p>以下のリンクからパスワードをリセットしてください（1時間有効）:</p>
        <a href="${process.env.APP_URL}/reset-password?token=${token}">
          パスワードをリセット
        </a>
        <p>このリクエストに心当たりがない場合は無視してください。</p>
        <p>パスワードは変更されません。</p>
      `,
    });
  }

  // 常に同じメッセージ（ユーザー列挙防止）
  return { message: 'メールアドレスが登録されていればリセットメールを送信しました' };
}

// パスワードリセット実行
async function resetPassword(token: string, newPassword: string) {
  // バリデーション
  if (newPassword.length < 8 || newPassword.length > 128) {
    return { error: 'パスワードは8文字以上128文字以下にしてください' };
  }

  // 漏洩チェック
  const breached = await isPasswordBreached(newPassword);
  if (breached) {
    return { error: 'このパスワードは漏洩データベースに含まれています' };
  }

  const hashedToken = crypto.createHash('sha256').update(token).digest('hex');

  const resetToken = await prisma.passwordResetToken.findFirst({
    where: {
      token: hashedToken,
      expiresAt: { gt: new Date() },
    },
    include: { user: true },
  });

  if (!resetToken) {
    return { error: '無効または期限切れのリンクです' };
  }

  // 新しいパスワードが前のパスワードと同じでないかチェック
  const isSame = await bcrypt.compare(newPassword, resetToken.user.password!);
  if (isSame) {
    return { error: '前のパスワードとは異なるパスワードを設定してください' };
  }

  const hashedPassword = await bcrypt.hash(newPassword, 12);

  await prisma.$transaction([
    // パスワード更新
    prisma.user.update({
      where: { id: resetToken.userId },
      data: {
        password: hashedPassword,
        failedLoginAttempts: 0,
        lockedUntil: null,
        passwordChangedAt: new Date(),
      },
    }),
    // トークン削除
    prisma.passwordResetToken.deleteMany({
      where: { userId: resetToken.userId },
    }),
    // 全セッション無効化（パスワード変更後は全デバイスからログアウト）
    prisma.session.deleteMany({
      where: { userId: resetToken.userId },
    }),
  ]);

  // パスワード変更通知メール
  await sendEmail({
    to: resetToken.user.email!,
    subject: 'パスワードが変更されました',
    html: `
      <p>パスワードが正常に変更されました。</p>
      <p>日時: ${new Date().toLocaleString('ja-JP', { timeZone: 'Asia/Tokyo' })}</p>
      <p>この変更に心当たりがない場合は、直ちに
        <a href="${process.env.APP_URL}/forgot-password">パスワードをリセット</a>
        するか、サポートにご連絡ください。
      </p>
    `,
  });

  return { success: true };
}
```

---

## 7. パスワード変更（ログイン中）

```typescript
// パスワード変更（要現在のパスワード）
'use server';

const changePasswordSchema = z.object({
  currentPassword: z.string().min(1),
  newPassword: z.string().min(8).max(128),
  confirmPassword: z.string(),
}).refine((data) => data.newPassword === data.confirmPassword, {
  message: 'パスワードが一致しません',
  path: ['confirmPassword'],
}).refine((data) => data.currentPassword !== data.newPassword, {
  message: '現在のパスワードと異なるパスワードを設定してください',
  path: ['newPassword'],
});

async function changePassword(formData: FormData) {
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  const parsed = changePasswordSchema.safeParse({
    currentPassword: formData.get('currentPassword'),
    newPassword: formData.get('newPassword'),
    confirmPassword: formData.get('confirmPassword'),
  });

  if (!parsed.success) {
    return { error: parsed.error.flatten().fieldErrors };
  }

  // 漏洩チェック
  const breached = await isPasswordBreached(parsed.data.newPassword);
  if (breached) {
    return { error: { newPassword: ['このパスワードは漏洩データベースに含まれています'] } };
  }

  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
  });

  // 現在のパスワードを検証
  const isValid = await bcrypt.compare(parsed.data.currentPassword, user!.password!);
  if (!isValid) {
    return { error: { currentPassword: ['現在のパスワードが正しくありません'] } };
  }

  // 新しいパスワードで更新
  const hashedPassword = await bcrypt.hash(parsed.data.newPassword, 12);

  await prisma.$transaction([
    prisma.user.update({
      where: { id: session.user.id },
      data: {
        password: hashedPassword,
        passwordChangedAt: new Date(),
      },
    }),
    // 現在のセッション以外を無効化
    prisma.session.deleteMany({
      where: {
        userId: session.user.id,
        id: { not: session.sessionId },
      },
    }),
  ]);

  // 通知
  await sendSecurityNotification(session.user.id, 'password_change');

  return { success: true };
}
```

---

## 8. セキュリティ通知

```typescript
// 重要なアカウントイベントの通知
async function sendSecurityNotification(
  userId: string,
  event: 'login' | 'password_change' | 'email_change' | 'new_device' | 'account_locked'
) {
  const user = await prisma.user.findUnique({ where: { id: userId } });
  if (!user?.email) return;

  const messages = {
    login: {
      subject: '新しいログインがありました',
      body: '新しいデバイスからログインがありました。',
    },
    password_change: {
      subject: 'パスワードが変更されました',
      body: 'パスワードが正常に変更されました。',
    },
    email_change: {
      subject: 'メールアドレスが変更されました',
      body: 'アカウントのメールアドレスが変更されました。',
    },
    new_device: {
      subject: '新しいデバイスからのアクセス',
      body: '認識されていないデバイスからアクセスがありました。',
    },
    account_locked: {
      subject: 'アカウントがロックされました',
      body: 'ログイン試行の失敗が多数あり、アカウントが一時的にロックされました。30分後に自動解除されます。',
    },
  };

  const { subject, body } = messages[event];

  // 監査ログ
  await prisma.securityEvent.create({
    data: {
      userId,
      event,
      timestamp: new Date(),
      metadata: { notificationSent: true },
    },
  });

  await sendEmail({
    to: user.email,
    subject,
    html: `
      <p>${body}</p>
      <p>日時: ${new Date().toLocaleString('ja-JP', { timeZone: 'Asia/Tokyo' })}</p>
      <p>心当たりがない場合は、直ちにパスワードを変更してください。</p>
      <a href="${process.env.APP_URL}/settings/security">セキュリティ設定</a>
    `,
  });
}
```

---

## 9. データベーススキーマ

```typescript
// Prisma スキーマ（認証関連の完全版）
// schema.prisma

/*
model User {
  id                   String    @id @default(cuid())
  email                String    @unique
  name                 String?
  password             String?   // ソーシャルログインユーザーは null
  image                String?
  role                 String    @default("viewer")
  emailVerified        DateTime?
  failedLoginAttempts  Int       @default(0)
  lockedUntil          DateTime?
  lastLoginAt          DateTime?
  lastLoginIp          String?
  passwordChangedAt    DateTime?
  createdAt            DateTime  @default(now())
  updatedAt            DateTime  @updatedAt

  sessions             Session[]
  accounts             Account[]
  verificationTokens   VerificationToken[]
  passwordResetTokens  PasswordResetToken[]
  knownDevices         KnownDevice[]
  securityEvents       SecurityEvent[]
}

model Session {
  id        String   @id @default(cuid())
  userId    String
  token     String   @unique
  expiresAt DateTime
  createdAt DateTime @default(now())
  ipAddress String?
  userAgent String?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
}

model VerificationToken {
  id         String   @id @default(cuid())
  identifier String   // email
  token      String   // SHA-256 ハッシュ
  expires    DateTime

  @@unique([identifier, token])
}

model PasswordResetToken {
  id        String   @id @default(cuid())
  userId    String
  token     String   // SHA-256 ハッシュ
  expiresAt DateTime
  createdAt DateTime @default(now())

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)
}

model KnownDevice {
  id          String   @id @default(cuid())
  userId      String
  fingerprint String
  userAgent   String
  ipAddress   String
  lastSeenAt  DateTime

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([userId, fingerprint])
}

model SecurityEvent {
  id        String   @id @default(cuid())
  userId    String
  event     String
  timestamp DateTime @default(now())
  metadata  Json?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@index([userId, timestamp])
}
*/
```

---

## 10. エッジケースとアンチパターン

### 10.1 エッジケース

```
メール・パスワード認証のエッジケース:

  (1) メール配信の遅延・不達
     → 確認メールの再送機能を提供
     → 迷惑メールフォルダの確認を促す
     → 代替のメール確認方法（コード入力）を検討

  (2) メールアドレスの大文字・小文字
     → RFC 5321: ローカルパートは大文字小文字を区別する
     → 実務上: ほぼ全てのメールプロバイダーで区別しない
     → 推奨: 保存時に小文字に正規化する
     → email.toLowerCase() で統一

  (3) パスワードの Unicode 正規化
     → "cafe\u0301" と "caf\u00e9" は見た目が同じだが異なるバイト列
     → NIST SP 800-63B: SASLprep (RFC 7613) で正規化を推奨
     → 最低限: NFC 正規化を適用
     → password.normalize('NFC')

  (4) 大量の同時登録（ボット）
     → CAPTCHA の導入（reCAPTCHA, hCaptcha, Turnstile）
     → ハニーポットフィールド
     → 登録速度制限

  (5) 既存ユーザーがパスワード未設定（ソーシャルログインのみ）
     → パスワード設定フローを別途提供
     → 「パスワードを設定」はリセットとは別フロー
     → メール確認済みであることを前提にする
```

### 10.2 アンチパターン

```
メール・パスワード認証のアンチパターン:

  (1) パスワードの平文ログ出力
     ✗ console.log(`Login: ${email}, ${password}`);
     → パスワードは一切ログに出力してはならない
     → 本番環境のログにパスワードが残ると重大インシデント

  (2) リセットトークンの平文保存
     ✗ await db.resetToken.create({ token: rawToken });
     → DB 漏洩時にトークンが露出
     → SHA-256 でハッシュ化して保存

  (3) エラーメッセージの差異
     ✗ 「メールが見つかりません」「パスワードが間違っています」
     → ユーザー列挙攻撃を許す
     → 「メールアドレスまたはパスワードが正しくありません」に統一

  (4) セッション無効化の欠如
     ✗ パスワード変更後も旧セッションが有効
     → 攻撃者がパスワードを知っている場合にセッションが残る
     → パスワード変更時は全セッションを無効化
```

---

## 11. 演習問題

### 演習 1: 基本的なメール・パスワード認証（基礎）

以下の要件でメール・パスワード認証を実装せよ。

```
要件:
- ユーザー登録（名前、メール、パスワード）
- bcrypt でのパスワードハッシュ化
- メール確認（24時間有効のトークン）
- ログイン（セッション作成）
- ログアウト（セッション破棄）

テスト:
- 登録成功 → メール確認 → ログイン成功
- 未確認メールでのログイン拒否
- 不正パスワードでのログイン失敗
```

### 演習 2: セキュリティ強化（応用）

演習 1 に以下のセキュリティ機能を追加せよ。

```
要件:
- レート制限（IP + メールの2層）
- アカウントロック（10回失敗で30分ロック）
- パスワードリセット（1時間有効のトークン）
- パスワード変更（現在のパスワード確認必須）
- ユーザー列挙攻撃対策
- タイミング攻撃対策

テスト:
- レート制限の動作確認
- アカウントロック → 自動解除
- パスワードリセットの完全フロー
```

### 演習 3: エンタープライズ機能（発展）

本番環境を想定した機能を追加せよ。

```
要件:
- Have I Been Pwned API との連携
- デバイスフィンガープリントによる不正検知
- セキュリティイベントの監査ログ
- Argon2id への移行（bcrypt からの自動アップグレード）
- パスワード履歴（過去5個のパスワードを禁止）
- CAPTCHA 統合（reCAPTCHA or Turnstile）

テスト:
- 漏洩パスワードの拒否
- 未知デバイスからのログイン通知
- ハッシュ関数の自動アップグレード
```

---

## 12. FAQ・トラブルシューティング

### Q1: bcrypt の比較が常に遅い

**原因**: bcrypt は意図的に低速に設計されている。cost=12 の場合、約250msかかる。

```
対処法:
- cost を下げるのは非推奨（セキュリティが低下）
- ログイン処理全体のパフォーマンスが問題なら:
  1. ワーカースレッドで bcrypt を実行
  2. Node.js の場合は bcrypt ネイティブモジュールを使用
  3. bcryptjs（pure JS）より bcrypt（C++バインディング）を推奨
```

### Q2: メール配信が遅い / 届かない

```
対処法:
1. メール送信は非同期（バックグラウンドジョブ）で実行
2. SendGrid, Resend, Amazon SES 等の専用サービスを使用
3. SPF, DKIM, DMARC を設定
4. 送信元ドメインの評判を維持
5. 迷惑メールフォルダの確認を促すUI
```

### Q3: アカウントロックが頻繁に発生する

```
対処法:
1. CAPTCHA でボット攻撃を防止
2. IP ベースのレート制限を先に適用
3. ロック閾値を調整（5回 → 10回）
4. ロック期間を段階的に増加（5分 → 15分 → 30分）
5. 管理者によるロック解除機能を提供
```

### Q4: パスワードリセットメールが悪用される

```
対処法:
1. リセットメールの送信にもレート制限を適用
2. トークンの有効期限を短くする（1時間以下）
3. トークンは1回使用で無効化
4. リセット完了時にメール通知
5. 不審なリセット要求のモニタリング
```

---

## まとめ

| フロー | セキュリティ要件 |
|--------|----------------|
| 登録 | bcrypt/Argon2 ハッシュ、メール確認必須、漏洩チェック |
| ログイン | 多層レート制限、アカウントロック、タイミング攻撃対策 |
| メール確認 | SHA-256 ハッシュトークン、24時間有効、再送レート制限 |
| リセット | ハッシュトークン、1時間有効、全セッション無効化 |
| 変更 | 現在パスワード確認、他セッション無効化 |
| 通知 | 重要イベントのメール通知、監査ログ |
| 列挙対策 | 統一エラーメッセージ、タイミング均一化 |

---

## 次に読むべきガイド
→ [[03-sso-and-enterprise.md]] — SSO とエンタープライズ認証

---

## 参考文献
1. NIST. "Digital Identity Guidelines: Authentication and Lifecycle Management." SP 800-63B, 2020.
2. OWASP. "Password Storage Cheat Sheet." cheatsheetseries.owasp.org, 2024.
3. OWASP. "Forgot Password Cheat Sheet." cheatsheetseries.owasp.org, 2024.
4. OWASP. "Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
5. Auth.js. "Credentials Provider." authjs.dev, 2024.
6. Troy Hunt. "Have I Been Pwned." haveibeenpwned.com, 2024.
7. RFC 7613. "Preparation, Enforcement, and Comparison of Internationalized Strings (PRECIS)." IETF, 2015.
8. Password Hashing Competition. "Argon2." password-hashing.net, 2015.
