# NextAuth.js (Auth.js) セットアップ

> NextAuth.js（現 Auth.js）は Next.js のデファクト認証ライブラリ。プロバイダー設定、セッション管理、データベースアダプター、コールバックのカスタマイズまで、Auth.js の基本セットアップから本番運用まで解説する。

## この章で学ぶこと

- [ ] Auth.js の基本セットアップを理解する
- [ ] プロバイダー・アダプター・コールバックの設定を把握する
- [ ] セッション管理とカスタマイズを実装できるようになる

---

## 1. 基本セットアップ

```typescript
// インストール
// npm install next-auth@beta @auth/prisma-adapter

// auth.ts（認証設定のメインファイル）
import NextAuth from 'next-auth';
import Google from 'next-auth/providers/google';
import GitHub from 'next-auth/providers/github';
import Credentials from 'next-auth/providers/credentials';
import { PrismaAdapter } from '@auth/prisma-adapter';
import { prisma } from '@/lib/prisma';
import bcrypt from 'bcrypt';
import { z } from 'zod';

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: PrismaAdapter(prisma),

  providers: [
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    GitHub({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    }),
    Credentials({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        const parsed = z.object({
          email: z.string().email(),
          password: z.string().min(8),
        }).safeParse(credentials);

        if (!parsed.success) return null;

        const user = await prisma.user.findUnique({
          where: { email: parsed.data.email },
        });

        if (!user?.password) return null;

        const isValid = await bcrypt.compare(parsed.data.password, user.password);
        if (!isValid) return null;

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image,
        };
      },
    }),
  ],

  session: {
    strategy: 'jwt',  // JWT セッション（Credentials 使用時は必須）
    maxAge: 30 * 24 * 60 * 60, // 30日
  },

  pages: {
    signIn: '/login',
    error: '/login',
    newUser: '/onboarding',
  },

  callbacks: {
    // JWT にカスタムデータを追加
    async jwt({ token, user, trigger, session }) {
      if (user) {
        token.role = user.role;
        token.orgId = user.orgId;
      }

      // セッション更新時（update() 呼び出し時）
      if (trigger === 'update' && session) {
        token.name = session.name;
      }

      return token;
    },

    // セッションにカスタムデータを公開
    async session({ session, token }) {
      session.user.id = token.sub!;
      session.user.role = token.role as string;
      session.user.orgId = token.orgId as string;
      return session;
    },

    // アクセス制御
    async authorized({ auth, request }) {
      const isLoggedIn = !!auth?.user;
      const isProtected = request.nextUrl.pathname.startsWith('/dashboard');

      if (isProtected && !isLoggedIn) {
        return false; // ログインページにリダイレクト
      }

      return true;
    },
  },
});
```

```typescript
// app/api/auth/[...nextauth]/route.ts
import { handlers } from '@/auth';

export const { GET, POST } = handlers;
```

```typescript
// middleware.ts
export { auth as middleware } from '@/auth';

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};
```

---

## 2. 型定義の拡張

```typescript
// types/next-auth.d.ts
import { DefaultSession, DefaultUser } from 'next-auth';

declare module 'next-auth' {
  interface User extends DefaultUser {
    role: string;
    orgId?: string;
  }

  interface Session extends DefaultSession {
    user: {
      id: string;
      role: string;
      orgId?: string;
    } & DefaultSession['user'];
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    role?: string;
    orgId?: string;
  }
}
```

---

## 3. セッションの使用

```typescript
// Server Component でセッション取得
import { auth } from '@/auth';

async function DashboardPage() {
  const session = await auth();

  if (!session) {
    redirect('/login');
  }

  return (
    <div>
      <h1>Welcome, {session.user.name}</h1>
      <p>Role: {session.user.role}</p>
    </div>
  );
}

// Client Component でセッション取得
'use client';
import { useSession } from 'next-auth/react';

function UserMenu() {
  const { data: session, status } = useSession();

  if (status === 'loading') return <Skeleton />;
  if (!session) return <Link href="/login">Login</Link>;

  return (
    <div>
      <img src={session.user.image!} alt={session.user.name!} />
      <span>{session.user.name}</span>
      <span className="text-xs text-gray-500">{session.user.role}</span>
    </div>
  );
}

// Server Action でのセッション
'use server';
import { auth } from '@/auth';

async function createArticle(formData: FormData) {
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  await prisma.article.create({
    data: {
      title: formData.get('title') as string,
      content: formData.get('content') as string,
      authorId: session.user.id,
    },
  });

  revalidatePath('/articles');
}
```

---

## 4. サインイン・サインアウト

```typescript
// サインインフォーム
'use client';
import { signIn } from 'next-auth/react';

function LoginPage() {
  const [error, setError] = useState('');

  // ソーシャルログイン
  const handleSocialLogin = (provider: string) => {
    signIn(provider, { callbackUrl: '/dashboard' });
  };

  // メール・パスワードログイン
  const handleCredentialsLogin = async (e: FormEvent) => {
    e.preventDefault();
    const formData = new FormData(e.target as HTMLFormElement);

    const result = await signIn('credentials', {
      email: formData.get('email'),
      password: formData.get('password'),
      redirect: false,
    });

    if (result?.error) {
      setError('メールアドレスまたはパスワードが正しくありません');
    } else {
      window.location.href = '/dashboard';
    }
  };

  return (
    <div className="max-w-md mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Login</h1>

      {/* ソーシャルログイン */}
      <div className="space-y-2 mb-6">
        <button
          onClick={() => handleSocialLogin('google')}
          className="w-full p-3 border rounded flex items-center justify-center gap-2"
        >
          Continue with Google
        </button>
        <button
          onClick={() => handleSocialLogin('github')}
          className="w-full p-3 border rounded flex items-center justify-center gap-2"
        >
          Continue with GitHub
        </button>
      </div>

      <div className="relative my-6">
        <div className="absolute inset-0 flex items-center">
          <div className="w-full border-t" />
        </div>
        <div className="relative flex justify-center text-sm">
          <span className="bg-white px-2 text-gray-500">or</span>
        </div>
      </div>

      {/* メール・パスワード */}
      <form onSubmit={handleCredentialsLogin} className="space-y-4">
        {error && <p className="text-red-500 text-sm">{error}</p>}
        <input
          name="email"
          type="email"
          placeholder="Email"
          className="w-full p-3 border rounded"
          required
        />
        <input
          name="password"
          type="password"
          placeholder="Password"
          className="w-full p-3 border rounded"
          required
        />
        <button type="submit" className="w-full p-3 bg-blue-500 text-white rounded">
          Sign In
        </button>
      </form>
    </div>
  );
}
```

```typescript
// サインアウト

// Server Action
import { signOut } from '@/auth';

async function handleSignOut() {
  'use server';
  await signOut({ redirectTo: '/' });
}

// Client Component
import { signOut } from 'next-auth/react';

function LogoutButton() {
  return (
    <button onClick={() => signOut({ callbackUrl: '/' })}>
      Logout
    </button>
  );
}
```

---

## 5. SessionProvider の設定

```typescript
// app/layout.tsx
import { SessionProvider } from 'next-auth/react';
import { auth } from '@/auth';

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth();

  return (
    <html>
      <body>
        <SessionProvider session={session}>
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}
```

---

## 6. 環境変数

```
# .env.local

# NextAuth
AUTH_SECRET=your-random-secret-at-least-32-characters
AUTH_URL=http://localhost:3000

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

AUTH_SECRET の生成:
  npx auth secret
  または: openssl rand -base64 32
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| セットアップ | auth.ts で一元管理 |
| プロバイダー | Google, GitHub, Credentials 等 |
| セッション | JWT 戦略（Credentials使用時） |
| コールバック | jwt + session で型拡張 |
| 型定義 | next-auth.d.ts で拡張 |
| Server | auth() でセッション取得 |
| Client | useSession() + SessionProvider |

---

## 次に読むべきガイド
→ [[01-social-login.md]] — ソーシャルログイン

---

## 参考文献
1. Auth.js. "Getting Started." authjs.dev, 2024.
2. Auth.js. "Providers." authjs.dev/reference, 2024.
3. Next.js. "Authentication." nextjs.org/docs, 2024.
