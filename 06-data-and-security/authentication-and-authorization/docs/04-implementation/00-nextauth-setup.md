# NextAuth.js (Auth.js) セットアップ

> NextAuth.js（現 Auth.js）は Next.js のデファクト認証ライブラリ。プロバイダー設定、セッション管理、データベースアダプター、コールバックのカスタマイズまで、Auth.js の基本セットアップから本番運用まで解説する。

## 前提知識

- [[../../01-session-auth/01-session-store.md]] — セッションストアの基本
- [[../../02-token-auth/00-jwt-deep-dive.md]] — JWT の基礎
- Next.js App Router の基本（Server Components, Server Actions）
- OAuth 2.0 / OpenID Connect の概念

## この章で学ぶこと

- [ ] Auth.js の基本セットアップを理解する
- [ ] プロバイダー・アダプター・コールバックの設定を把握する
- [ ] セッション管理とカスタマイズを実装できるようになる
- [ ] JWT と Database セッション戦略の使い分けを理解する
- [ ] 本番環境でのセキュリティ設定を把握する
- [ ] エラーハンドリングとトラブルシューティングを習得する

---

## 1. Auth.js のアーキテクチャ

### 1.1 全体構成

```
Auth.js のアーキテクチャ:

  ┌──────────────────────────────────────────────────────────┐
  │                    Next.js Application                    │
  │                                                          │
  │  ┌─────────────┐  ┌───────────────┐  ┌──────────────┐  │
  │  │ Server      │  │ Client        │  │ Middleware    │  │
  │  │ Components  │  │ Components    │  │              │  │
  │  │             │  │               │  │ ルートガード  │  │
  │  │ auth()      │  │ useSession()  │  │ auth()       │  │
  │  └──────┬──────┘  └───────┬───────┘  └──────┬───────┘  │
  │         │                 │                  │          │
  │         └─────────────────┼──────────────────┘          │
  │                           │                             │
  │                   ┌───────┴───────┐                     │
  │                   │   auth.ts     │ ← 認証設定の中心    │
  │                   │               │                     │
  │                   │ - providers   │                     │
  │                   │ - adapter     │                     │
  │                   │ - callbacks   │                     │
  │                   │ - session     │                     │
  │                   │ - pages       │                     │
  │                   └───────┬───────┘                     │
  │                           │                             │
  │              ┌────────────┼────────────┐                │
  │              │            │            │                │
  │         ┌────┴────┐  ┌───┴───┐  ┌────┴─────┐          │
  │         │Providers│  │Adapter│  │Callbacks │          │
  │         │         │  │       │  │          │          │
  │         │Google   │  │Prisma │  │jwt()     │          │
  │         │GitHub   │  │Drizzle│  │session() │          │
  │         │Creds    │  │       │  │signIn()  │          │
  │         └─────────┘  └───┬───┘  └──────────┘          │
  │                          │                             │
  │                    ┌─────┴─────┐                       │
  │                    │ Database  │                       │
  │                    │ (Users,   │                       │
  │                    │  Accounts,│                       │
  │                    │  Sessions)│                       │
  │                    └───────────┘                       │
  └──────────────────────────────────────────────────────────┘
```

### 1.2 認証フローの詳細

```
OAuth プロバイダー認証フロー（Auth.js 内部）:

  ┌────────┐     ┌──────────┐     ┌─────────┐     ┌──────────┐
  │Browser │     │ Next.js  │     │Auth.js  │     │Provider  │
  │        │     │          │     │         │     │(Google)  │
  └───┬────┘     └────┬─────┘     └────┬────┘     └────┬─────┘
      │               │               │               │
      │ Click         │               │               │
      │ "Sign in"     │               │               │
      │──────────────→│               │               │
      │               │ signIn()      │               │
      │               │──────────────→│               │
      │               │               │ Build Auth URL│
      │               │               │──────────────→│
      │               │ Redirect 302  │               │
      │←──────────────│               │               │
      │                               │               │
      │ User consents on Google        │               │
      │───────────────────────────────────────────────→│
      │                               │               │
      │ Redirect to callback URL      │               │
      │ /api/auth/callback/google     │               │
      │  ?code=AUTH_CODE&state=...    │               │
      │──────────────→│               │               │
      │               │ handleCallback│               │
      │               │──────────────→│               │
      │               │               │ Exchange code │
      │               │               │──────────────→│
      │               │               │ access_token  │
      │               │               │←──────────────│
      │               │               │               │
      │               │               │ Fetch profile │
      │               │               │──────────────→│
      │               │               │ user info     │
      │               │               │←──────────────│
      │               │               │               │
      │               │ signIn callback│              │
      │               │ jwt callback   │              │
      │               │ session callback│             │
      │               │               │               │
      │               │ Create/Update │               │
      │               │ User in DB    │               │
      │               │               │               │
      │ Set Session   │               │               │
      │ Cookie        │               │               │
      │←──────────────│               │               │
      │               │               │               │
      │ Redirect to   │               │               │
      │ callbackUrl   │               │               │
      │←──────────────│               │               │
```

### 1.3 JWT vs Database セッション

```
セッション戦略の比較:

  ┌─────────────────┬────────────────────┬────────────────────┐
  │                 │ JWT 戦略           │ Database 戦略      │
  ├─────────────────┼────────────────────┼────────────────────┤
  │ データ保存場所   │ Cookie（暗号化JWT）│ DB + Cookie(ID)    │
  │ サーバー状態     │ ステートレス       │ ステートフル        │
  │ スケーラビリティ │ ◎ 高             │ ○ DB依存          │
  │ セッション失効   │ △ 困難           │ ◎ 即時可能        │
  │ データ容量       │ △ Cookie制限(4KB) │ ◎ 制限なし        │
  │ DB負荷          │ ◎ なし           │ △ 毎リクエスト     │
  │ Credentials対応 │ ◎ 可能           │ ✗ 非対応          │
  │ セッション一覧   │ ✗ 不可           │ ◎ 可能           │
  │ 強制ログアウト   │ △ 要ブラックリスト │ ◎ DB削除で即時    │
  │ デフォルト       │ ○               │ Adapter設定時      │
  └─────────────────┴────────────────────┴────────────────────┘

  選定基準:
  → Credentials Provider を使う → JWT 必須
  → セッション即時失効が必要 → Database
  → スケーラビリティ重視 → JWT
  → セッション管理機能が必要 → Database
  → 一般的な Web アプリ → JWT（シンプル）
```

---

## 2. 基本セットアップ

### 2.1 インストールと初期設定

```bash
# インストール
npm install next-auth@beta @auth/prisma-adapter
npm install @prisma/client prisma bcrypt
npm install -D @types/bcrypt

# AUTH_SECRET の生成
npx auth secret
# または
openssl rand -base64 32
```

### 2.2 メインの認証設定ファイル

```typescript
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
  // データベースアダプター（ユーザー・アカウント・セッションの永続化）
  adapter: PrismaAdapter(prisma),

  // 認証プロバイダー
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
        // 入力バリデーション
        const parsed = z.object({
          email: z.string().email(),
          password: z.string().min(8),
        }).safeParse(credentials);

        if (!parsed.success) return null;

        // ユーザー検索
        const user = await prisma.user.findUnique({
          where: { email: parsed.data.email },
        });

        // パスワードが設定されていない場合（ソーシャルログインのみ）
        if (!user?.password) return null;

        // パスワード検証（bcrypt）
        const isValid = await bcrypt.compare(parsed.data.password, user.password);
        if (!isValid) return null;

        // 認証成功: ユーザーオブジェクトを返す
        return {
          id: user.id,
          email: user.email,
          name: user.name,
          image: user.image,
          role: user.role,
          orgId: user.orgId,
        };
      },
    }),
  ],

  // セッション設定
  session: {
    strategy: 'jwt',  // JWT セッション（Credentials 使用時は必須）
    maxAge: 30 * 24 * 60 * 60, // 30日（秒単位）
    updateAge: 24 * 60 * 60,   // 24時間ごとにセッション更新
  },

  // JWT 設定
  jwt: {
    maxAge: 30 * 24 * 60 * 60, // 30日
  },

  // カスタムページ
  pages: {
    signIn: '/login',       // ログインページ
    error: '/login',        // エラー時のリダイレクト先
    newUser: '/onboarding', // 新規ユーザーのリダイレクト先
    // signOut: '/logout',  // サインアウトページ（任意）
    // verifyRequest: '/verify', // メール検証ページ（任意）
  },

  // コールバック関数
  callbacks: {
    // JWT にカスタムデータを追加
    async jwt({ token, user, trigger, session, account }) {
      // 初回サインイン時（user が存在する）
      if (user) {
        token.role = user.role;
        token.orgId = user.orgId;
      }

      // プロバイダーの access_token を保存する場合
      if (account) {
        token.accessToken = account.access_token;
        token.refreshToken = account.refresh_token;
        token.accessTokenExpires = account.expires_at
          ? account.expires_at * 1000
          : undefined;
      }

      // セッション更新時（update() 呼び出し時）
      if (trigger === 'update' && session) {
        token.name = session.name;
        token.image = session.image;
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

    // アクセス制御（middleware で使用）
    async authorized({ auth, request }) {
      const isLoggedIn = !!auth?.user;
      const { pathname } = request.nextUrl;

      // 保護ルートのチェック
      const protectedPaths = ['/dashboard', '/admin', '/settings'];
      const isProtected = protectedPaths.some(p => pathname.startsWith(p));

      if (isProtected && !isLoggedIn) {
        return false; // ログインページにリダイレクト
      }

      // 管理者ルートのチェック
      if (pathname.startsWith('/admin')) {
        return auth?.user?.role === 'admin';
      }

      return true;
    },

    // サインイン時の制御
    async signIn({ user, account, profile }) {
      // メール検証チェック（Google の場合）
      if (account?.provider === 'google') {
        return profile?.email_verified === true;
      }

      // ブロックされたユーザーのチェック
      if (user.id) {
        const dbUser = await prisma.user.findUnique({
          where: { id: user.id },
          select: { blockedAt: true },
        });
        if (dbUser?.blockedAt) {
          return false; // サインイン拒否
        }
      }

      return true;
    },
  },

  // イベントハンドラー
  events: {
    async signIn({ user, account, isNewUser }) {
      // ログイン監査ログ
      console.log(`User ${user.email} signed in via ${account?.provider}`);

      if (isNewUser) {
        // 新規ユーザーへのウェルカムメール送信
        // await sendWelcomeEmail(user.email!);
      }

      // 最終ログイン日時の更新
      if (user.id) {
        await prisma.user.update({
          where: { id: user.id },
          data: { lastLoginAt: new Date() },
        });
      }
    },

    async signOut(message) {
      // サインアウト監査ログ
      if ('token' in message) {
        console.log(`User ${message.token?.email} signed out`);
      }
    },

    async createUser({ user }) {
      // 新規ユーザー作成時の処理
      console.log(`New user created: ${user.email}`);
    },
  },

  // デバッグモード（開発時のみ）
  debug: process.env.NODE_ENV === 'development',

  // Cookie 設定のカスタマイズ
  cookies: {
    sessionToken: {
      name: process.env.NODE_ENV === 'production'
        ? '__Secure-authjs.session-token'
        : 'authjs.session-token',
      options: {
        httpOnly: true,
        sameSite: 'lax',
        path: '/',
        secure: process.env.NODE_ENV === 'production',
      },
    },
  },
});
```

### 2.3 API ルートハンドラー

```typescript
// app/api/auth/[...nextauth]/route.ts
import { handlers } from '@/auth';

export const { GET, POST } = handlers;

// ※ Auth.js v5 では handlers を export するだけでよい
// GET: OAuth コールバック、CSRF トークン取得等
// POST: サインイン、サインアウト等
```

### 2.4 Middleware 設定

```typescript
// middleware.ts
export { auth as middleware } from '@/auth';

export const config = {
  // マッチするパスを指定
  // 静的ファイルと API ルートは除外
  matcher: [
    '/((?!api/auth|_next/static|_next/image|favicon.ico|public).*)',
  ],
};
```

```
Middleware のマッチングパターン:

  matcher の正規表現:
  /((?!api/auth|_next/static|_next/image|favicon.ico|public).*)

  ┌────────────────────────┬──────────┬──────────────────────┐
  │ パス                    │ マッチ   │ 理由                  │
  ├────────────────────────┼──────────┼──────────────────────┤
  │ /dashboard             │ ✓        │ 保護対象              │
  │ /admin/users           │ ✓        │ 保護対象              │
  │ /api/auth/callback     │ ✗        │ Auth.js 内部ルート    │
  │ /api/users             │ ✓        │ カスタム API          │
  │ /_next/static/...      │ ✗        │ 静的ファイル          │
  │ /favicon.ico           │ ✗        │ ファビコン            │
  │ /login                 │ ✓        │ ページ                │
  └────────────────────────┴──────────┴──────────────────────┘
```

---

## 3. 型定義の拡張

### 3.1 TypeScript 型宣言

```typescript
// types/next-auth.d.ts
import { DefaultSession, DefaultUser } from 'next-auth';
import { DefaultJWT } from 'next-auth/jwt';

// User 型の拡張
declare module 'next-auth' {
  interface User extends DefaultUser {
    role: string;
    orgId?: string;
    blockedAt?: Date | null;
  }

  interface Session extends DefaultSession {
    user: {
      id: string;
      role: string;
      orgId?: string;
    } & DefaultSession['user'];
  }
}

// JWT 型の拡張
declare module 'next-auth/jwt' {
  interface JWT extends DefaultJWT {
    role?: string;
    orgId?: string;
    accessToken?: string;
    refreshToken?: string;
    accessTokenExpires?: number;
  }
}
```

### 3.2 Prisma スキーマ

```prisma
// prisma/schema.prisma

// Auth.js が必要とするモデル
model User {
  id            String    @id @default(cuid())
  name          String?
  email         String?   @unique
  emailVerified DateTime?
  image         String?
  password      String?   // Credentials 認証用
  role          String    @default("viewer")
  orgId         String?
  blockedAt     DateTime?
  lastLoginAt   DateTime?
  createdAt     DateTime  @default(now())
  updatedAt     DateTime  @updatedAt

  accounts      Account[]
  sessions      Session[]

  org           Organization? @relation(fields: [orgId], references: [id])

  @@index([email])
  @@index([orgId])
}

model Account {
  id                String  @id @default(cuid())
  userId            String
  type              String
  provider          String
  providerAccountId String
  refresh_token     String? @db.Text
  access_token      String? @db.Text
  expires_at        Int?
  token_type        String?
  scope             String?
  id_token          String? @db.Text
  session_state     String?

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@unique([provider, providerAccountId])
  @@index([userId])
}

model Session {
  id           String   @id @default(cuid())
  sessionToken String   @unique
  userId       String
  expires      DateTime

  user User @relation(fields: [userId], references: [id], onDelete: Cascade)

  @@index([userId])
}

model VerificationToken {
  identifier String
  token      String   @unique
  expires    DateTime

  @@unique([identifier, token])
}

model Organization {
  id        String   @id @default(cuid())
  name      String
  slug      String   @unique
  createdAt DateTime @default(now())

  users User[]
}
```

```
Auth.js のデータベーススキーマ解説:

  ┌──────────────────────────────────────────────────┐
  │                    User                           │
  │ ─ アプリのユーザー情報                             │
  │ ─ email はソーシャルログインで取得                  │
  │ ─ password は Credentials 認証用（任意）           │
  │ ─ role, orgId 等はカスタムフィールド                │
  └──────────────┬───────────────┬───────────────────┘
                 │               │
                 │ 1:N           │ 1:N
                 │               │
  ┌──────────────┴──────┐  ┌────┴──────────────────┐
  │      Account        │  │      Session          │
  │ ─ OAuth アカウント   │  │ ─ DB セッション戦略用  │
  │ ─ provider ごとに1件 │  │ ─ JWT 戦略では未使用   │
  │ ─ access_token 保存  │  │ ─ 有効期限付き         │
  └─────────────────────┘  └───────────────────────┘

  ┌───────────────────────┐
  │  VerificationToken    │
  │ ─ メール検証用         │
  │ ─ Magic Link 用       │
  │ ─ 使い捨てトークン     │
  └───────────────────────┘
```

---

## 4. セッションの使用

### 4.1 Server Component でセッション取得

```typescript
// Server Component でセッション取得
import { auth } from '@/auth';
import { redirect } from 'next/navigation';

async function DashboardPage() {
  const session = await auth();

  if (!session) {
    redirect('/login');
  }

  return (
    <div>
      <h1>Welcome, {session.user.name}</h1>
      <p>Role: {session.user.role}</p>
      <p>Organization: {session.user.orgId}</p>
    </div>
  );
}

export default DashboardPage;
```

### 4.2 Client Component でセッション取得

```typescript
// Client Component でセッション取得
'use client';
import { useSession } from 'next-auth/react';
import Link from 'next/link';

function UserMenu() {
  const { data: session, status, update } = useSession();

  if (status === 'loading') return <Skeleton />;
  if (!session) return <Link href="/login">Login</Link>;

  // セッション情報の更新
  const handleNameChange = async (newName: string) => {
    await update({ name: newName });
    // → jwt callback の trigger === 'update' が呼ばれる
  };

  return (
    <div className="flex items-center gap-3">
      <img
        src={session.user.image!}
        alt={session.user.name!}
        className="w-8 h-8 rounded-full"
      />
      <div>
        <span className="font-medium">{session.user.name}</span>
        <span className="text-xs text-gray-500 block">{session.user.role}</span>
      </div>
    </div>
  );
}

export default UserMenu;
```

### 4.3 Server Action でのセッション

```typescript
// Server Action でのセッション
'use server';
import { auth } from '@/auth';
import { revalidatePath } from 'next/cache';

export async function createArticle(formData: FormData) {
  const session = await auth();
  if (!session) throw new Error('Unauthorized');

  // ロールチェック
  if (!['editor', 'admin'].includes(session.user.role)) {
    throw new Error('Forbidden: editors only');
  }

  const title = formData.get('title') as string;
  const content = formData.get('content') as string;

  // バリデーション
  if (!title || title.length < 1 || title.length > 200) {
    throw new Error('Invalid title');
  }

  await prisma.article.create({
    data: {
      title,
      content,
      authorId: session.user.id,
      orgId: session.user.orgId,
    },
  });

  revalidatePath('/articles');
}
```

### 4.4 API Route でのセッション

```typescript
// app/api/articles/route.ts
import { auth } from '@/auth';
import { NextResponse } from 'next/server';

export async function GET() {
  const session = await auth();

  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const articles = await prisma.article.findMany({
    where: { orgId: session.user.orgId },
    orderBy: { createdAt: 'desc' },
  });

  return NextResponse.json(articles);
}

export async function POST(request: Request) {
  const session = await auth();

  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  if (!['editor', 'admin'].includes(session.user.role)) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }

  const body = await request.json();

  const article = await prisma.article.create({
    data: {
      ...body,
      authorId: session.user.id,
      orgId: session.user.orgId,
    },
  });

  return NextResponse.json(article, { status: 201 });
}
```

---

## 5. サインイン・サインアウト

### 5.1 ログインページの実装

```typescript
// app/login/page.tsx
'use client';
import { signIn } from 'next-auth/react';
import { useState, FormEvent } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';

function LoginPage() {
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const searchParams = useSearchParams();
  const router = useRouter();
  const callbackUrl = searchParams.get('callbackUrl') || '/dashboard';

  // エラーメッセージのマッピング
  const errorMessages: Record<string, string> = {
    OAuthSignin: 'ソーシャルログインの開始に失敗しました',
    OAuthCallback: 'ソーシャルログインのコールバックでエラーが発生しました',
    OAuthAccountNotLinked: 'このメールアドレスは別のログイン方法で登録されています',
    CredentialsSignin: 'メールアドレスまたはパスワードが正しくありません',
    SessionRequired: 'ログインが必要です',
    Default: 'ログインに失敗しました。もう一度お試しください',
  };

  const urlError = searchParams.get('error');
  const displayError = error || (urlError ? errorMessages[urlError] || errorMessages.Default : '');

  // ソーシャルログイン
  const handleSocialLogin = async (provider: string) => {
    setLoading(true);
    try {
      await signIn(provider, { callbackUrl });
    } catch {
      setError('ログインに失敗しました');
      setLoading(false);
    }
  };

  // メール・パスワードログイン
  const handleCredentialsLogin = async (e: FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const formData = new FormData(e.target as HTMLFormElement);

    try {
      const result = await signIn('credentials', {
        email: formData.get('email'),
        password: formData.get('password'),
        redirect: false,
      });

      if (result?.error) {
        setError(errorMessages.CredentialsSignin);
      } else {
        router.push(callbackUrl);
        router.refresh(); // セッション状態を更新
      }
    } catch {
      setError(errorMessages.Default);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Login</h1>

      {displayError && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          {displayError}
        </div>
      )}

      {/* ソーシャルログイン */}
      <div className="space-y-2 mb-6">
        <button
          onClick={() => handleSocialLogin('google')}
          disabled={loading}
          className="w-full p-3 border rounded flex items-center justify-center gap-2
                     hover:bg-gray-50 disabled:opacity-50"
        >
          <GoogleIcon className="w-5 h-5" />
          Continue with Google
        </button>
        <button
          onClick={() => handleSocialLogin('github')}
          disabled={loading}
          className="w-full p-3 border rounded flex items-center justify-center gap-2
                     hover:bg-gray-50 disabled:opacity-50"
        >
          <GitHubIcon className="w-5 h-5" />
          Continue with GitHub
        </button>
      </div>

      {/* セパレーター */}
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
        <input
          name="email"
          type="email"
          placeholder="Email"
          className="w-full p-3 border rounded focus:ring-2 focus:ring-blue-500 outline-none"
          required
          disabled={loading}
        />
        <input
          name="password"
          type="password"
          placeholder="Password"
          minLength={8}
          className="w-full p-3 border rounded focus:ring-2 focus:ring-blue-500 outline-none"
          required
          disabled={loading}
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full p-3 bg-blue-500 text-white rounded hover:bg-blue-600
                     disabled:opacity-50 transition-colors"
        >
          {loading ? 'Signing in...' : 'Sign In'}
        </button>
      </form>

      <p className="mt-4 text-center text-sm text-gray-500">
        Don&apos;t have an account?{' '}
        <a href="/register" className="text-blue-500 hover:underline">Sign up</a>
      </p>
    </div>
  );
}

export default LoginPage;
```

### 5.2 サインアウト

```typescript
// Server Action でのサインアウト（推奨）
// components/SignOutButton.tsx
import { signOut } from '@/auth';

function SignOutButton() {
  return (
    <form
      action={async () => {
        'use server';
        await signOut({ redirectTo: '/' });
      }}
    >
      <button type="submit" className="text-gray-600 hover:text-gray-900">
        Logout
      </button>
    </form>
  );
}

// Client Component でのサインアウト
'use client';
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

## 6. SessionProvider の設定

```typescript
// app/layout.tsx
import { SessionProvider } from 'next-auth/react';
import { auth } from '@/auth';

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await auth();

  return (
    <html lang="ja">
      <body>
        <SessionProvider
          session={session}
          // セッションの自動更新間隔（秒）
          refetchInterval={5 * 60}
          // ウィンドウフォーカス時に再取得
          refetchOnWindowFocus={true}
          // refetchWhenOffline={false}
        >
          {children}
        </SessionProvider>
      </body>
    </html>
  );
}
```

```
SessionProvider の動作:

  ┌──────────────────────────────────────────────────┐
  │ SessionProvider                                   │
  │                                                   │
  │ ① 初期セッションを props で受け取る               │
  │    → Server Component で auth() を呼んで渡す      │
  │    → 初回レンダリングでセッションが即座に利用可能   │
  │                                                   │
  │ ② refetchInterval で定期的に更新                   │
  │    → /api/auth/session にリクエスト               │
  │    → セッション期限切れの検知                      │
  │                                                   │
  │ ③ refetchOnWindowFocus でフォーカス時に更新        │
  │    → タブを切り替えて戻った時                      │
  │    → セッション状態の最新化                        │
  │                                                   │
  │ ④ useSession() でどこからでもアクセス可能          │
  │    → status: 'loading' | 'authenticated'          │
  │              | 'unauthenticated'                   │
  └──────────────────────────────────────────────────┘
```

---

## 7. 環境変数

```bash
# .env.local

# NextAuth 必須
AUTH_SECRET=your-random-secret-at-least-32-characters
AUTH_URL=http://localhost:3000  # 本番では https://myapp.com

# Google OAuth
GOOGLE_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-client-secret

# GitHub OAuth
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb

# AUTH_SECRET の生成方法:
#   npx auth secret
#   または: openssl rand -base64 32

# AUTH_TRUST_HOST=true  # プロキシ背後の場合
```

```
環境変数の注意点:

  ┌────────────────────┬─────────────────────────────────────┐
  │ 変数               │ 注意                                 │
  ├────────────────────┼─────────────────────────────────────┤
  │ AUTH_SECRET        │ 本番では必ずランダム生成の値を使用     │
  │                    │ 最低32文字                            │
  │                    │ 環境ごとに異なる値を設定               │
  ├────────────────────┼─────────────────────────────────────┤
  │ AUTH_URL           │ v5 では自動検出されるため通常は不要    │
  │                    │ プロキシ背後の場合は明示的に設定       │
  ├────────────────────┼─────────────────────────────────────┤
  │ GOOGLE_CLIENT_*    │ Google Cloud Console で発行           │
  │                    │ リダイレクト URI の設定を忘れずに       │
  ├────────────────────┼─────────────────────────────────────┤
  │ GITHUB_CLIENT_*    │ GitHub Settings > Developer settings  │
  │                    │ OAuth App で発行                      │
  ├────────────────────┼─────────────────────────────────────┤
  │ DATABASE_URL       │ 接続文字列                            │
  │                    │ 本番では接続プール設定を追加           │
  └────────────────────┴─────────────────────────────────────┘
```

---

## 8. 高度なカスタマイズ

### 8.1 アクセストークンの更新（Refresh Token Rotation）

```typescript
// auth.ts のコールバックに追加
callbacks: {
  async jwt({ token, account }) {
    // 初回サインイン: トークン情報を保存
    if (account) {
      return {
        ...token,
        accessToken: account.access_token,
        refreshToken: account.refresh_token,
        accessTokenExpires: account.expires_at
          ? account.expires_at * 1000
          : Date.now() + 3600 * 1000,
      };
    }

    // アクセストークンが有効期限内
    if (token.accessTokenExpires && Date.now() < token.accessTokenExpires) {
      return token;
    }

    // アクセストークンの更新
    try {
      const response = await fetch('https://oauth2.googleapis.com/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({
          client_id: process.env.GOOGLE_CLIENT_ID!,
          client_secret: process.env.GOOGLE_CLIENT_SECRET!,
          grant_type: 'refresh_token',
          refresh_token: token.refreshToken as string,
        }),
      });

      const tokens = await response.json();

      if (!response.ok) throw tokens;

      return {
        ...token,
        accessToken: tokens.access_token,
        accessTokenExpires: Date.now() + tokens.expires_in * 1000,
        // refresh_token が返された場合は更新（Rotation）
        refreshToken: tokens.refresh_token ?? token.refreshToken,
      };
    } catch (error) {
      console.error('Error refreshing access token:', error);
      return {
        ...token,
        error: 'RefreshTokenError',
      };
    }
  },
}
```

### 8.2 カスタムサインインページのサーバーサイド実装

```typescript
// Server Action ベースのサインイン（推奨パターン）
// app/login/actions.ts
'use server';
import { signIn } from '@/auth';
import { AuthError } from 'next-auth';

export async function authenticate(
  prevState: { error: string } | undefined,
  formData: FormData
) {
  try {
    await signIn('credentials', {
      email: formData.get('email'),
      password: formData.get('password'),
      redirectTo: '/dashboard',
    });
  } catch (error) {
    if (error instanceof AuthError) {
      switch (error.type) {
        case 'CredentialsSignin':
          return { error: 'Invalid credentials' };
        case 'AccessDenied':
          return { error: 'Account is blocked' };
        default:
          return { error: 'Something went wrong' };
      }
    }
    throw error; // 予期しないエラーは再スロー
  }
}

// app/login/page.tsx
'use client';
import { useActionState } from 'react';
import { authenticate } from './actions';

export default function LoginPage() {
  const [state, action, isPending] = useActionState(authenticate, undefined);

  return (
    <form action={action}>
      {state?.error && <p className="text-red-500">{state.error}</p>}

      <input name="email" type="email" placeholder="Email" required />
      <input name="password" type="password" placeholder="Password" required />

      <button type="submit" disabled={isPending}>
        {isPending ? 'Signing in...' : 'Sign In'}
      </button>
    </form>
  );
}
```

### 8.3 ロールベースのレイアウト

```typescript
// app/dashboard/layout.tsx
import { auth } from '@/auth';
import { redirect } from 'next/navigation';

export default async function DashboardLayout({
  children,
  admin,
  viewer,
}: {
  children: React.ReactNode;
  admin: React.ReactNode;
  viewer: React.ReactNode;
}) {
  const session = await auth();
  if (!session) redirect('/login');

  return (
    <div className="flex">
      <aside className="w-64 bg-gray-100 min-h-screen p-4">
        <nav>
          <NavLink href="/dashboard">Dashboard</NavLink>
          {session.user.role === 'admin' && (
            <>
              <NavLink href="/dashboard/users">Users</NavLink>
              <NavLink href="/dashboard/settings">Settings</NavLink>
            </>
          )}
        </nav>
      </aside>
      <main className="flex-1 p-6">
        {children}
      </main>
    </div>
  );
}
```

---

## 9. エッジケースとトラブルシューティング

### 9.1 よくある問題と解決策

```
Auth.js よくある問題:

  ┌─────────────────────────────┬──────────────────────────────────┐
  │ 問題                        │ 解決策                            │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ NEXTAUTH_URL が見つからない │ AUTH_URL を .env.local に設定      │
  │                             │ v5 では自動検出、不要な場合もある  │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ Credentials + Adapter で    │ session.strategy = 'jwt' を設定   │
  │ エラー                      │ Credentials は JWT 戦略が必須     │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ OAuth コールバックエラー     │ リダイレクト URI を正確に設定      │
  │                             │ プロバイダーの設定画面で確認       │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ セッションに id がない       │ jwt callback で token.sub を使う  │
  │                             │ session callback で設定           │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ Cookie が設定されない        │ Secure 属性と HTTP/HTTPS を確認   │
  │                             │ SameSite 設定を確認               │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ TypeScript 型エラー          │ next-auth.d.ts を作成             │
  │                             │ tsconfig の include に追加        │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ ログイン後にリダイレクトしない │ router.refresh() を呼ぶ          │
  │                             │ callbackUrl を適切に設定          │
  ├─────────────────────────────┼──────────────────────────────────┤
  │ セッションが取れない          │ SessionProvider を確認            │
  │                             │ Server/Client の使い分けを確認    │
  └─────────────────────────────┴──────────────────────────────────┘
```

### 9.2 アンチパターン

```
Auth.js のアンチパターン:

  ✗ アンチパターン①: authorize 内で直接エラーメッセージを返す
  ┌──────────────────────────────────────────────────┐
  │ // 危険: ユーザー列挙攻撃の情報源になる            │
  │ if (!user) throw new Error('User not found');     │
  │ if (!isValid) throw new Error('Wrong password');  │
  │                                                   │
  │ // 正しい: 一般的なエラーメッセージを使う           │
  │ if (!user || !isValid) return null;               │
  └──────────────────────────────────────────────────┘

  ✗ アンチパターン②: JWT にセンシティブ情報を含める
  ┌──────────────────────────────────────────────────┐
  │ // 危険: JWT はクライアントで復号可能               │
  │ token.creditCardNumber = user.creditCard;         │
  │ token.ssn = user.socialSecurityNumber;            │
  │                                                   │
  │ // 正しい: 最小限の情報のみ含める                   │
  │ token.role = user.role;                           │
  │ token.orgId = user.orgId;                         │
  └──────────────────────────────────────────────────┘

  ✗ アンチパターン③: Client Component でのみ認可チェック
  ┌──────────────────────────────────────────────────┐
  │ // 不十分: DevTools でバイパス可能                  │
  │ if (session?.user.role !== 'admin') return null;  │
  │                                                   │
  │ // 正しい: Server + Client の両方でチェック         │
  │ // Server: auth() + redirect()                    │
  │ // Client: 表示の最適化のみ                        │
  └──────────────────────────────────────────────────┘
```

---

## 10. 本番環境の設定

### 10.1 セキュリティチェックリスト

```
本番デプロイ前のチェックリスト:

  □ AUTH_SECRET は環境ごとにランダム生成
  □ OAuth プロバイダーのリダイレクト URI を本番 URL に更新
  □ Cookie の secure 属性が true（HTTPS）
  □ Cookie の sameSite が 'lax' 以上
  □ debug: false（本番環境）
  □ Credentials Provider のレート制限を設定
  □ 入力バリデーション（Zod 等）を全エンドポイントで実施
  □ CSRF 保護が有効
  □ セッションの maxAge が適切
  □ 機密情報が JWT に含まれていない
  □ エラーメッセージがユーザー列挙に使えない
  □ データベース接続のプーリング設定
  □ ログの出力レベルが適切
```

### 10.2 パフォーマンス最適化

```
Auth.js のパフォーマンス最適化:

  ┌──────────────────────┬──────────────────────────────────┐
  │ 最適化ポイント        │ 手法                              │
  ├──────────────────────┼──────────────────────────────────┤
  │ DB 接続              │ PgBouncer / PrismaAccelerate     │
  │                      │ コネクションプーリング             │
  ├──────────────────────┼──────────────────────────────────┤
  │ セッション取得        │ JWT 戦略で DB アクセス回避         │
  │                      │ DB 戦略の場合はキャッシュ考慮      │
  ├──────────────────────┼──────────────────────────────────┤
  │ Middleware           │ matcher で不要なパスを除外         │
  │                      │ 静的ファイルをスキップ              │
  ├──────────────────────┼──────────────────────────────────┤
  │ SessionProvider      │ refetchInterval を適切に設定       │
  │                      │ 短すぎると API 負荷増              │
  ├──────────────────────┼──────────────────────────────────┤
  │ bcrypt               │ saltRounds = 10-12（デフォルト10） │
  │                      │ 高すぎるとログインが遅い            │
  └──────────────────────┴──────────────────────────────────┘
```

---

## 11. 演習

### 演習1: 基礎 - Auth.js 基本セットアップ

```
【演習1】Auth.js 基本セットアップ

目的: Next.js プロジェクトに Auth.js を導入し、基本的な認証フローを実装する

手順:
1. Next.js プロジェクトを作成（App Router）
2. Auth.js をインストール・設定
3. Google OAuth プロバイダーを設定
4. Prisma アダプターを設定（SQLite）
5. ログインページを作成
6. 保護されたダッシュボードページを作成
7. SessionProvider を設定

評価基準:
  □ Google ログインが動作する
  □ セッション情報が表示される
  □ 未認証ユーザーがリダイレクトされる
  □ サインアウトが動作する
```

### 演習2: 応用 - Credentials + ロールベースアクセス制御

```
【演習2】Credentials + ロールベースアクセス制御

目的: メールパスワード認証とロールベースの認可を実装する

手順:
1. Credentials Provider を追加
2. ユーザー登録フォームを実装（bcrypt でハッシュ化）
3. JWT に role を含める型拡張
4. Middleware でロールベースのルートガード
5. Server Component でロールに基づく表示制御
6. Server Action でロールチェック

評価基準:
  □ 登録・ログインが動作する
  □ admin / editor / viewer の3ロール
  □ 各ロールで異なるページアクセス
  □ 型安全にセッション情報が取得できる
```

### 演習3: 発展 - マルチプロバイダー + アカウントリンク

```
【演習3】マルチプロバイダー + アカウントリンク

目的: 複数プロバイダーとアカウントリンクを実装する

手順:
1. Google + GitHub + Credentials の3プロバイダー設定
2. 同一メールアドレスの自動リンク（email_verified チェック）
3. 設定画面でのアカウントリンク / アンリンク
4. Refresh Token Rotation の実装
5. セキュリティ監査ログの実装

評価基準:
  □ 3つのプロバイダーでログイン可能
  □ 同一メールのアカウントが自動リンクされる
  □ 設定画面でリンク管理ができる
  □ 最後のログイン方法は削除できない
```

---

## 12. FAQ

### Q1: v4 と v5 の主な違いは？

```
Auth.js v4 → v5 の主な変更点:

  ┌──────────────────────┬──────────────────┬──────────────────┐
  │ 機能                  │ v4               │ v5               │
  ├──────────────────────┼──────────────────┼──────────────────┤
  │ パッケージ名          │ next-auth        │ next-auth@beta   │
  │ 設定ファイル          │ [...nextauth].ts │ auth.ts          │
  │ API ルート           │ pages/api/auth/  │ app/api/auth/    │
  │ Server で取得        │ getServerSession │ auth()           │
  │ Middleware           │ withAuth         │ auth as middleware│
  │ 型拡張               │ next-auth.d.ts   │ 同じ             │
  │ Edge 対応            │ 実験的           │ 正式対応          │
  │ Server Actions       │ 非対応           │ 対応             │
  │ signIn/signOut export│ なし             │ auth.ts から      │
  └──────────────────────┴──────────────────┴──────────────────┘
```

### Q2: Credentials Provider を使う場合の注意点は？

```
A: Credentials Provider は以下の制約があります:

  1. セッション戦略: JWT 必須（Database 戦略は使用不可）
  2. Adapter: Account / Session テーブルは使われない
  3. セキュリティ:
     → パスワードのハッシュ化は自前で実装
     → レート制限は自前で実装
     → ブルートフォース対策は自前で実装
  4. 推奨:
     → 可能であれば OAuth プロバイダーを優先
     → Credentials は補助的なログイン手段として
     → Magic Link も検討
```

### Q3: Drizzle ORM でも使えますか？

```
A: はい。@auth/drizzle-adapter が公式に提供されています。

  npm install @auth/drizzle-adapter

  import { DrizzleAdapter } from '@auth/drizzle-adapter';
  import { db } from '@/lib/db';

  export const { handlers, auth } = NextAuth({
    adapter: DrizzleAdapter(db),
    ...
  });
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| セットアップ | auth.ts で一元管理、handlers/auth/signIn/signOut を export |
| プロバイダー | Google, GitHub, Credentials 等を組合せ |
| セッション | JWT 戦略（Credentials使用時）/ Database 戦略 |
| コールバック | jwt → session の順でデータを流す |
| 型定義 | next-auth.d.ts で User, Session, JWT を拡張 |
| Server | auth() でセッション取得（Server Components, Actions, API Routes） |
| Client | useSession() + SessionProvider |
| Middleware | authorized callback でルートガード |
| 本番運用 | AUTH_SECRET, Cookie設定, エラーハンドリング |

---

## 次に読むべきガイド
→ [[01-social-login.md]] — ソーシャルログイン

---

## 参考文献
1. Auth.js. "Getting Started." authjs.dev, 2024.
2. Auth.js. "Providers." authjs.dev/reference, 2024.
3. Auth.js. "Adapters." authjs.dev/reference/adapter, 2024.
4. Auth.js. "Callbacks." authjs.dev/reference/callbacks, 2024.
5. Next.js. "Authentication." nextjs.org/docs, 2024.
6. Auth.js. "Upgrade Guide (v4 to v5)." authjs.dev/getting-started/migrating-to-v5, 2024.
7. Prisma. "Auth.js Adapter." prisma.io/docs, 2024.
