# メール・パスワード認証

> ソーシャルログインだけでは不十分な場面で必要となるメール・パスワード認証。ユーザー登録、メール確認、ログイン、パスワードリセット、アカウントロックまで、安全なメール認証の完全フローを解説する。

## この章で学ぶこと

- [ ] ユーザー登録とメール確認のフローを実装する
- [ ] 安全なログインとレート制限を把握する
- [ ] パスワードリセットとアカウント保護を設計できるようになる

---

## 1. ユーザー登録

```typescript
// 登録フォームのバリデーション
import { z } from 'zod';

const registerSchema = z.object({
  name: z.string().min(1, '名前を入力してください').max(100),
  email: z.string().email('有効なメールアドレスを入力してください'),
  password: z.string()
    .min(8, 'パスワードは8文字以上必要です')
    .max(128),
  confirmPassword: z.string(),
}).refine((data) => data.password === data.confirmPassword, {
  message: 'パスワードが一致しません',
  path: ['confirmPassword'],
});

// 登録 Server Action
'use server';
import bcrypt from 'bcrypt';
import crypto from 'crypto';

async function register(formData: FormData) {
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

  // メールの重複チェック
  const existingUser = await prisma.user.findUnique({ where: { email } });
  if (existingUser) {
    // ユーザー列挙攻撃を防止するため、同じメッセージを返す
    return { success: true, message: '確認メールを送信しました' };
  }

  // パスワードハッシュ化
  const hashedPassword = await bcrypt.hash(password, 12);

  // ユーザー作成
  const user = await prisma.user.create({
    data: {
      name,
      email,
      password: hashedPassword,
      role: 'viewer',
      emailVerified: null, // 未確認
    },
  });

  // メール確認トークン生成
  const verificationToken = crypto.randomBytes(32).toString('hex');
  const hashedToken = crypto.createHash('sha256').update(verificationToken).digest('hex');

  await prisma.verificationToken.create({
    data: {
      identifier: email,
      token: hashedToken,
      expires: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24時間
    },
  });

  // 確認メール送信
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
    `,
  });

  return { success: true, message: '確認メールを送信しました' };
}
```

---

## 2. メール確認

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

  // メールを確認済みに更新
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

## 3. ログインとレート制限

```typescript
// ログイン処理（Auth.js Credentials プロバイダー）
async authorize(credentials) {
  const { email, password } = credentials;

  // レート制限チェック
  const rateLimitKey = `login_attempts:${email}`;
  const attempts = await redis.get(rateLimitKey);

  if (attempts && parseInt(attempts) >= 5) {
    throw new Error('Too many login attempts. Please try again in 15 minutes.');
  }

  // ユーザー取得
  const user = await prisma.user.findUnique({ where: { email } });

  if (!user?.password) {
    // 失敗カウント増加
    await redis.incr(rateLimitKey);
    await redis.expire(rateLimitKey, 900); // 15分
    return null;
  }

  // アカウントロックチェック
  if (user.lockedUntil && user.lockedUntil > new Date()) {
    throw new Error('Account is locked. Please try again later.');
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
    }

    await prisma.user.update({
      where: { id: user.id },
      data: updateData,
    });

    await redis.incr(rateLimitKey);
    await redis.expire(rateLimitKey, 900);
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
    },
  });

  // レート制限カウントリセット
  await redis.del(rateLimitKey);

  return {
    id: user.id,
    email: user.email,
    name: user.name,
    image: user.image,
    role: user.role,
  };
}
```

---

## 4. パスワードリセット

```typescript
// パスワードリセット要求
'use server';

async function requestPasswordReset(email: string) {
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
      `,
    });
  }

  // 常に同じメッセージ（ユーザー列挙防止）
  return { message: 'メールアドレスが登録されていればリセットメールを送信しました' };
}

// パスワードリセット実行
async function resetPassword(token: string, newPassword: string) {
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
      },
    }),
    // トークン削除
    prisma.passwordResetToken.deleteMany({
      where: { userId: resetToken.userId },
    }),
    // 全セッション無効化
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
      <p>この変更に心当たりがない場合は、直ちにサポートにご連絡ください。</p>
    `,
  });

  return { success: true };
}
```

---

## 5. パスワード変更（ログイン中）

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
  await prisma.user.update({
    where: { id: session.user.id },
    data: { password: hashedPassword },
  });

  return { success: true };
}
```

---

## 6. セキュリティ通知

```typescript
// 重要なアカウントイベントの通知
async function sendSecurityNotification(
  userId: string,
  event: 'login' | 'password_change' | 'email_change' | 'new_device'
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
  };

  const { subject, body } = messages[event];

  await sendEmail({
    to: user.email,
    subject,
    html: `
      <p>${body}</p>
      <p>心当たりがない場合は、直ちにパスワードを変更してください。</p>
      <a href="${process.env.APP_URL}/settings/security">セキュリティ設定</a>
    `,
  });
}
```

---

## まとめ

| フロー | セキュリティ要件 |
|--------|----------------|
| 登録 | ハッシュ化、メール確認必須 |
| ログイン | レート制限、アカウントロック |
| リセット | ハッシュトークン、1時間有効 |
| 変更 | 現在パスワード確認、全セッション無効化 |
| 通知 | 重要イベントのメール通知 |

---

## 次に読むべきガイド
→ [[03-sso-and-enterprise.md]] — SSO とエンタープライズ認証

---

## 参考文献
1. OWASP. "Forgot Password Cheat Sheet." cheatsheetseries.owasp.org, 2024.
2. Auth.js. "Credentials Provider." authjs.dev, 2024.
3. NIST. "SP 800-63B." nist.gov, 2020.
