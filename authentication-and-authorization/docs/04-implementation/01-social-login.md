# ソーシャルログイン

> Google、GitHub、Apple のソーシャルログインは現代の Web アプリに不可欠。各プロバイダーの設定、アカウントリンク、プロフィール同期、プロバイダー固有の注意点を解説する。

## この章で学ぶこと

- [ ] 主要プロバイダー（Google、GitHub、Apple）の設定を理解する
- [ ] アカウントリンクとプロバイダー切り替えを実装する
- [ ] プロバイダー固有の制約と対処法を把握する

---

## 1. Google ログイン

```
Google Cloud Console での設定:

  ① APIs & Services > Credentials に移動
  ② OAuth 2.0 クライアント ID を作成
  ③ アプリケーションの種類: Web アプリケーション
  ④ 承認済みリダイレクト URI:
     開発: http://localhost:3000/api/auth/callback/google
     本番: https://myapp.com/api/auth/callback/google
  ⑤ OAuth 同意画面の設定:
     → アプリ名、サポートメール
     → スコープ: email, profile, openid
     → テストユーザーの追加（公開前）
```

```typescript
// Google プロバイダー設定
import Google from 'next-auth/providers/google';

Google({
  clientId: process.env.GOOGLE_CLIENT_ID!,
  clientSecret: process.env.GOOGLE_CLIENT_SECRET!,

  // 追加スコープの要求
  authorization: {
    params: {
      scope: 'openid email profile',
      prompt: 'consent',  // 毎回同意画面を表示
      access_type: 'offline', // Refresh Token を取得
    },
  },

  // プロフィールのカスタマイズ
  profile(profile) {
    return {
      id: profile.sub,
      name: profile.name,
      email: profile.email,
      image: profile.picture,
      role: 'viewer', // デフォルトロール
    };
  },
})
```

---

## 2. GitHub ログイン

```
GitHub OAuth App の設定:

  ① Settings > Developer settings > OAuth Apps
  ② New OAuth App を作成
  ③ Authorization callback URL:
     開発: http://localhost:3000/api/auth/callback/github
     本番: https://myapp.com/api/auth/callback/github
```

```typescript
// GitHub プロバイダー設定
import GitHub from 'next-auth/providers/github';

GitHub({
  clientId: process.env.GITHUB_CLIENT_ID!,
  clientSecret: process.env.GITHUB_CLIENT_SECRET!,

  // 追加スコープ
  authorization: {
    params: {
      scope: 'read:user user:email',
    },
  },

  // GitHub はメールが null の場合がある
  profile(profile) {
    return {
      id: String(profile.id),
      name: profile.name || profile.login,
      email: profile.email, // null の場合がある
      image: profile.avatar_url,
      role: 'viewer',
    };
  },
})

// GitHub のメール取得（email が null の場合の対処）
// Auth.js のコールバックで処理
callbacks: {
  async signIn({ user, account, profile }) {
    if (account?.provider === 'github' && !user.email) {
      // GitHub API でメール取得
      const emailRes = await fetch('https://api.github.com/user/emails', {
        headers: { Authorization: `Bearer ${account.access_token}` },
      });
      const emails = await emailRes.json();
      const primary = emails.find((e: any) => e.primary && e.verified);
      if (primary) {
        user.email = primary.email;
      }
    }
    return true;
  },
}
```

---

## 3. Apple ログイン

```
Apple Developer での設定:

  ① Certificates, Identifiers & Profiles
  ② Identifiers > App IDs で App ID を登録
  ③ Sign in with Apple を有効化
  ④ Services IDs を作成（Web用）
  ⑤ Return URLs を設定:
     https://myapp.com/api/auth/callback/apple
  ⑥ Keys を作成（AuthKey_XXXXXXXXXX.p8）

Apple の特殊な要件:
  → Apple Developer Program（年額 $99）が必要
  → name と email は初回認可時のみ返却される
  → 2回目以降は sub（ユーザーID）のみ
  → id_token の検証に RS256 を使用
```

```typescript
// Apple プロバイダー設定
import Apple from 'next-auth/providers/apple';

Apple({
  clientId: process.env.APPLE_CLIENT_ID!,  // Services ID
  clientSecret: process.env.APPLE_CLIENT_SECRET!, // 生成した JWT

  // Apple の clientSecret は JWT（動的生成が必要）
  // https://authjs.dev/guides/providers/apple
})

// Apple clientSecret の生成（p8 ファイルから）
import jwt from 'jsonwebtoken';
import fs from 'fs';

function generateAppleClientSecret(): string {
  const privateKey = fs.readFileSync('AuthKey_XXXXXXXXXX.p8', 'utf8');

  return jwt.sign({}, privateKey, {
    algorithm: 'ES256',
    expiresIn: '180d',
    audience: 'https://appleid.apple.com',
    issuer: process.env.APPLE_TEAM_ID,
    subject: process.env.APPLE_CLIENT_ID,
    keyid: process.env.APPLE_KEY_ID,
  });
}
```

---

## 4. アカウントリンク

```
アカウントリンクの問題:

  同じユーザーが異なるプロバイダーでログイン:
    ① Google (alice@gmail.com) でサインアップ
    ② 後日 GitHub (alice@gmail.com) でログイン
    → 同じメールだが別アカウントとして作成される？

  Auth.js のデフォルト動作:
    → 同じメールの既存アカウントにはリンクしない（セキュリティ）
    → 理由: メール所有権が未検証の場合がある

  安全なリンク方法:
    → メールが検証済み（email_verified: true）の場合のみリンク
    → ユーザーに明示的にリンクを確認させる
```

```typescript
// アカウントリンクの実装
callbacks: {
  async signIn({ user, account, profile }) {
    if (!user.email) return true;

    // 同じメールの既存ユーザーをチェック
    const existingUser = await prisma.user.findUnique({
      where: { email: user.email },
      include: { accounts: true },
    });

    if (existingUser) {
      // 既にこのプロバイダーでリンク済み
      const existingAccount = existingUser.accounts.find(
        (a) => a.provider === account?.provider
      );
      if (existingAccount) return true;

      // メールが検証済みの場合のみ自動リンク
      const emailVerified =
        (account?.provider === 'google' && profile?.email_verified) ||
        (account?.provider === 'github'); // GitHub はメール検証済みのみ返す

      if (emailVerified) {
        // 既存ユーザーにアカウントをリンク
        await prisma.account.create({
          data: {
            userId: existingUser.id,
            type: account!.type,
            provider: account!.provider,
            providerAccountId: account!.providerAccountId,
            access_token: account!.access_token,
            refresh_token: account!.refresh_token,
            expires_at: account!.expires_at,
          },
        });

        // Auth.js のユーザー ID を既存ユーザーに合わせる
        user.id = existingUser.id;
        return true;
      }

      // 検証されていない場合はエラー
      return '/login?error=AccountExists';
    }

    return true;
  },
}
```

```typescript
// 手動アカウントリンク（設定画面から）
'use server';
import { auth } from '@/auth';

async function linkAccount(provider: string) {
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  // OAuth フローを開始（state にユーザー ID を含める）
  const { url } = await signIn(provider, {
    redirect: false,
    callbackUrl: '/settings/accounts',
  });

  return url;
}

// アカウントのリンク解除
async function unlinkAccount(provider: string) {
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  // 最後のログイン方法を削除しないようにチェック
  const accounts = await prisma.account.findMany({
    where: { userId: session.user.id },
  });

  const hasPassword = await prisma.user.findUnique({
    where: { id: session.user.id },
    select: { password: true },
  });

  if (accounts.length <= 1 && !hasPassword?.password) {
    throw new Error('Cannot unlink the only login method');
  }

  await prisma.account.deleteMany({
    where: {
      userId: session.user.id,
      provider,
    },
  });
}
```

---

## 5. プロフィール同期

```typescript
// ログイン時にプロフィールを同期
callbacks: {
  async signIn({ user, account, profile }) {
    if (!account) return true;

    // プロフィール情報の更新
    await prisma.user.update({
      where: { id: user.id },
      data: {
        name: user.name || undefined,
        image: user.image || undefined,
        // メールは変更しない（セキュリティ上の理由）
        lastLoginAt: new Date(),
        lastLoginProvider: account.provider,
      },
    });

    return true;
  },
}
```

---

## 6. ソーシャルログインの UX

```
ソーシャルログインの UX ベストプラクティス:

  ✓ 推奨:
    → 最も利用されるプロバイダーを上に配置
    → ブランドガイドラインに従ったボタンデザイン
    → 「Continue with Google」（「Sign in with」ではなく）
    → ログイン・サインアップで同じ画面（区別しない）
    → エラーメッセージは具体的に

  ✗ 避けるべき:
    → 5個以上のプロバイダーを並べる（選択疲れ）
    → 自前のアイコン使用（ブランドガイドライン違反）
    → ログイン後のリダイレクト先が不明
    → アカウントリンクの説明なし

  ボタンの順序（推奨）:
    ① Google（最も普及）
    ② Apple（iOS ユーザー）
    ③ GitHub（開発者向け）
    ④ メール・パスワード（フォールバック）
```

---

## まとめ

| プロバイダー | 注意点 |
|------------|--------|
| Google | 最も標準的。prompt=consent で同意画面制御 |
| GitHub | メールが null の場合あり。別 API で取得 |
| Apple | name は初回のみ。clientSecret は JWT 生成 |
| リンク | email_verified の場合のみ自動リンク |

---

## 次に読むべきガイド
→ [[02-email-password-auth.md]] — メール・パスワード認証

---

## 参考文献
1. Auth.js. "Providers." authjs.dev, 2024.
2. Google. "Sign in with Google." developers.google.com, 2024.
3. Apple. "Sign in with Apple." developer.apple.com, 2024.
