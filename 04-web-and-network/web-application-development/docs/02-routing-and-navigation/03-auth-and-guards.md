# 認証ガード

> 認証ガードはアプリケーションの入口を守る門番。ルート保護、ロールベースアクセス制御、リダイレクト、セッション管理まで、安全で使いやすい認証フローの設計と実装を習得する。

## この章で学ぶこと

- [ ] ルート保護のパターンと実装を理解する
- [ ] ロールベースアクセス制御（RBAC）の設計を把握する
- [ ] Next.js Middleware での認証チェックを学ぶ
- [ ] React Router / Vue Router / Angular Router でのガード実装を理解する
- [ ] 属性ベースアクセス制御（ABAC）の概念を学ぶ
- [ ] セッション管理とトークンリフレッシュの戦略を理解する
- [ ] 多要素認証（MFA）フロー統合を学ぶ
- [ ] OAuth / OpenID Connect 連携の認証ガードを実装する
- [ ] 認証ガードのテスト手法を習得する
- [ ] セキュリティのベストプラクティスとアンチパターンを把握する

---

## 1. ルート保護のパターン

### 1.1 認証ガードの全体アーキテクチャ

認証ガードは、ユーザーがアプリケーション内のリソースにアクセスする際に、適切な認証・認可状態を確認する仕組みである。モダンな Web アプリケーションでは、複数のレイヤーで段階的にアクセス制御を行うことが推奨される。

```
認証ガードの3つのレイヤー:

  ① Middleware（最前線）:
     → リクエスト到達前にチェック
     → 最も早い段階でリダイレクト
     → Next.js middleware.ts
     → サーバーサイドで実行される
     → ネットワークレベルでの保護

  ② Layout（レイアウト層）:
     → Server Component でセッション確認
     → 認証が必要なエリア全体を保護
     → 共通UIの制御（サイドバー、ナビゲーション）
     → ページ群単位での保護

  ③ Page / Component（ページ層）:
     → 個別ページでの権限チェック
     → きめ細かいアクセス制御
     → ボタンやメニューの表示/非表示
     → フィーチャーフラグとの連携

推奨設計:
  → Middleware: 大まかなルート保護（/app/* は認証必要）
  → Layout: 認証エリアのレイアウト + ロール確認
  → Component: ボタン表示/非表示等の細かい制御
```

### 1.2 保護パターンの分類

認証ガードのパターンは、いくつかの軸で分類できる。

```
【認証パターンの分類】

1. リダイレクト型（最も一般的）
   ├── 未認証 → ログインページへリダイレクト
   ├── 認証済み → callbackUrl へリダイレクト
   └── 権限不足 → 403ページ or ダッシュボードへ

2. ブロック型（API向け）
   ├── 未認証 → 401 Unauthorized
   ├── 権限不足 → 403 Forbidden
   └── トークン期限切れ → 401 + WWW-Authenticate ヘッダー

3. 条件付き表示型（UI向け）
   ├── 認証状態に応じてUIを切り替え
   ├── ロールに応じて機能を表示/非表示
   └── グレースフルデグラデーション

4. プログレッシブ型（段階的認証）
   ├── 基本閲覧 → 認証不要
   ├── インタラクション → 認証必要
   └── 高セキュリティ操作 → 再認証 + MFA
```

### 1.3 クライアントサイドガード vs サーバーサイドガード

```typescript
// ============================================
// クライアントサイドガード（React の例）
// ============================================

// ⚠️ クライアントサイドのみの保護は不十分
// UIの表示/非表示の制御には使えるが、
// セキュリティの最終防御線にしてはならない

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !user) {
      router.push('/login');
    }
  }, [user, isLoading, router]);

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (!user) {
    return null; // リダイレクト中
  }

  return <>{children}</>;
}

// ============================================
// サーバーサイドガード（Next.js の例）
// ============================================

// ✅ サーバーサイドでの保護が推奨
// APIルートやサーバーコンポーネントで認証チェック

// app/(protected)/layout.tsx
import { redirect } from 'next/navigation';
import { validateSession } from '@/lib/auth';

export default async function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await validateSession();

  if (!session) {
    redirect('/login');
  }

  return <>{children}</>;
}
```

### 1.4 多層防御（Defense in Depth）の原則

認証ガードにおいて最も重要な原則は「多層防御」である。1つのレイヤーが突破されても、他のレイヤーが保護を提供する設計が必要だ。

```typescript
// ============================================
// 多層防御の実装例
// ============================================

// Layer 1: Middleware（ネットワークレベル）
// middleware.ts
export function middleware(request: NextRequest) {
  const token = request.cookies.get('session-token')?.value;
  if (!token && isProtectedRoute(request.nextUrl.pathname)) {
    return redirectToLogin(request);
  }
  // トークンの基本的な検証（署名チェック等）
  if (token && !isValidTokenFormat(token)) {
    return redirectToLogin(request);
  }
  return NextResponse.next();
}

// Layer 2: Server Component（アプリケーションレベル）
// app/(app)/layout.tsx
export default async function AppLayout({ children }) {
  const session = await getSession(); // DB問い合わせ含む完全な検証
  if (!session || session.isExpired) {
    redirect('/login');
  }
  return <SessionProvider session={session}>{children}</SessionProvider>;
}

// Layer 3: API Route（データアクセスレベル）
// app/api/users/route.ts
export async function GET(request: NextRequest) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
  if (!hasPermission(session.user.role, 'users:read')) {
    return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
  }
  const users = await db.users.findMany();
  return NextResponse.json(users);
}

// Layer 4: データベースレベル（RLS: Row-Level Security）
// Supabase の例
// CREATE POLICY "Users can only read their own data"
// ON users FOR SELECT
// USING (auth.uid() = user_id);
```

---

## 2. Next.js Middleware による認証ガード

### 2.1 基本的な Middleware 実装

```typescript
// middleware.ts（プロジェクトルート）
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// 認証不要なパス
const publicPaths = ['/', '/login', '/register', '/about', '/pricing'];

// API の公開エンドポイント
const publicApiPaths = ['/api/public', '/api/health', '/api/auth'];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 公開パスはスキップ
  if (publicPaths.some(path => pathname === path)) {
    return NextResponse.next();
  }

  // 公開APIパスはスキップ
  if (publicApiPaths.some(path => pathname.startsWith(path))) {
    return NextResponse.next();
  }

  // 静的ファイルはスキップ
  if (pathname.startsWith('/_next') || pathname.includes('.')) {
    return NextResponse.next();
  }

  // セッショントークンの確認
  const token = request.cookies.get('session-token')?.value;

  if (!token) {
    // APIリクエストの場合は401を返す
    if (pathname.startsWith('/api/')) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    // ページリクエストの場合はログインへリダイレクト
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('callbackUrl', pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
```

### 2.2 高度な Middleware パターン

```typescript
// middleware.ts — 高度な認証ガード
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { jwtVerify } from 'jose';

// ルート設定の型定義
interface RouteConfig {
  pattern: RegExp;
  requireAuth: boolean;
  requiredRoles?: string[];
  rateLimit?: { max: number; windowMs: number };
}

// ルート設定
const routeConfigs: RouteConfig[] = [
  // 公開ルート
  { pattern: /^\/$/, requireAuth: false },
  { pattern: /^\/(login|register|forgot-password)$/, requireAuth: false },
  { pattern: /^\/api\/public\//, requireAuth: false },
  { pattern: /^\/api\/auth\//, requireAuth: false },
  { pattern: /^\/api\/webhooks\//, requireAuth: false },

  // 認証が必要なルート
  { pattern: /^\/dashboard/, requireAuth: true },
  { pattern: /^\/settings/, requireAuth: true },
  { pattern: /^\/api\//, requireAuth: true },

  // 管理者専用ルート
  { pattern: /^\/admin/, requireAuth: true, requiredRoles: ['admin'] },
  { pattern: /^\/api\/admin\//, requireAuth: true, requiredRoles: ['admin'] },

  // エディター以上のルート
  {
    pattern: /^\/content\/(create|edit)/,
    requireAuth: true,
    requiredRoles: ['editor', 'admin'],
  },
];

// JWT の検証
async function verifyToken(token: string) {
  try {
    const secret = new TextEncoder().encode(process.env.JWT_SECRET!);
    const { payload } = await jwtVerify(token, secret);
    return payload as { sub: string; role: string; exp: number };
  } catch {
    return null;
  }
}

// ルート設定の検索
function findRouteConfig(pathname: string): RouteConfig | undefined {
  return routeConfigs.find(config => config.pattern.test(pathname));
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 静的アセットはスキップ
  if (
    pathname.startsWith('/_next') ||
    pathname.startsWith('/static') ||
    pathname.match(/\.(ico|png|jpg|jpeg|svg|gif|webp|css|js|woff|woff2)$/)
  ) {
    return NextResponse.next();
  }

  // ルート設定を取得
  const routeConfig = findRouteConfig(pathname);

  // 設定が見つからない場合はデフォルトで認証を要求
  if (!routeConfig) {
    const token = request.cookies.get('session-token')?.value;
    if (!token) {
      return redirectToLogin(request, pathname);
    }
    return NextResponse.next();
  }

  // 公開ルート
  if (!routeConfig.requireAuth) {
    // ログイン済みユーザーがログインページにアクセスした場合
    const token = request.cookies.get('session-token')?.value;
    if (token && (pathname === '/login' || pathname === '/register')) {
      const payload = await verifyToken(token);
      if (payload) {
        return NextResponse.redirect(new URL('/dashboard', request.url));
      }
    }
    return NextResponse.next();
  }

  // 認証チェック
  const token = request.cookies.get('session-token')?.value;
  if (!token) {
    return redirectToLogin(request, pathname);
  }

  // トークン検証
  const payload = await verifyToken(token);
  if (!payload) {
    // 無効なトークン → Cookie を削除してログインへ
    const response = redirectToLogin(request, pathname);
    response.cookies.delete('session-token');
    return response;
  }

  // トークンの有効期限チェック（残り5分以内なら更新）
  const now = Math.floor(Date.now() / 1000);
  if (payload.exp && payload.exp - now < 300) {
    // トークンリフレッシュのヘッダーを追加
    const response = NextResponse.next();
    response.headers.set('X-Token-Refresh', 'true');
    return response;
  }

  // ロールベースのアクセスチェック
  if (routeConfig.requiredRoles && routeConfig.requiredRoles.length > 0) {
    if (!routeConfig.requiredRoles.includes(payload.role)) {
      // APIの場合は403
      if (pathname.startsWith('/api/')) {
        return NextResponse.json(
          { error: 'Insufficient permissions' },
          { status: 403 }
        );
      }
      // ページの場合は403ページへ
      return NextResponse.redirect(new URL('/403', request.url));
    }
  }

  // リクエストヘッダーにユーザー情報を追加
  const requestHeaders = new Headers(request.headers);
  requestHeaders.set('X-User-Id', payload.sub);
  requestHeaders.set('X-User-Role', payload.role);

  return NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });
}

// ログインページへのリダイレクトヘルパー
function redirectToLogin(request: NextRequest, callbackUrl: string) {
  if (request.nextUrl.pathname.startsWith('/api/')) {
    return NextResponse.json(
      { error: 'Authentication required' },
      { status: 401 }
    );
  }

  const loginUrl = new URL('/login', request.url);
  loginUrl.searchParams.set('callbackUrl', callbackUrl);
  return NextResponse.redirect(loginUrl);
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
```

### 2.3 Middleware のチェーン化

複数の Middleware を連結して実行するパターンは、関心の分離に有効である。

```typescript
// lib/middleware/chain.ts
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

type MiddlewareFunction = (
  request: NextRequest,
  response: NextResponse
) => Promise<NextResponse | null>;

// Middleware チェーンの構築
export function createMiddlewareChain(...middlewares: MiddlewareFunction[]) {
  return async function chainedMiddleware(request: NextRequest) {
    let response = NextResponse.next();

    for (const middleware of middlewares) {
      const result = await middleware(request, response);
      if (result) {
        // リダイレクトや早期レスポンスの場合はチェーンを中断
        if (result.status === 301 || result.status === 302 || result.status === 401 || result.status === 403) {
          return result;
        }
        response = result;
      }
    }

    return response;
  };
}

// lib/middleware/auth.ts
export async function authMiddleware(
  request: NextRequest,
  response: NextResponse
): Promise<NextResponse | null> {
  const token = request.cookies.get('session-token')?.value;

  if (!token && isProtectedRoute(request.nextUrl.pathname)) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('callbackUrl', request.nextUrl.pathname);
    return NextResponse.redirect(loginUrl);
  }

  return null; // チェーンを継続
}

// lib/middleware/rateLimit.ts
const rateLimitMap = new Map<string, { count: number; timestamp: number }>();

export async function rateLimitMiddleware(
  request: NextRequest,
  response: NextResponse
): Promise<NextResponse | null> {
  if (!request.nextUrl.pathname.startsWith('/api/')) {
    return null;
  }

  const ip = request.headers.get('x-forwarded-for') || 'unknown';
  const now = Date.now();
  const windowMs = 60 * 1000; // 1分
  const maxRequests = 100;

  const current = rateLimitMap.get(ip);
  if (current && now - current.timestamp < windowMs) {
    if (current.count >= maxRequests) {
      return NextResponse.json(
        { error: 'Rate limit exceeded' },
        { status: 429, headers: { 'Retry-After': '60' } }
      );
    }
    current.count++;
  } else {
    rateLimitMap.set(ip, { count: 1, timestamp: now });
  }

  return null;
}

// lib/middleware/logging.ts
export async function loggingMiddleware(
  request: NextRequest,
  response: NextResponse
): Promise<NextResponse | null> {
  const start = Date.now();
  console.log(`[${new Date().toISOString()}] ${request.method} ${request.nextUrl.pathname}`);

  // レスポンスヘッダーにリクエストIDを追加
  const requestId = crypto.randomUUID();
  response.headers.set('X-Request-Id', requestId);

  return null;
}

// middleware.ts — チェーンの使用
import { createMiddlewareChain } from './lib/middleware/chain';
import { authMiddleware } from './lib/middleware/auth';
import { rateLimitMiddleware } from './lib/middleware/rateLimit';
import { loggingMiddleware } from './lib/middleware/logging';

export default createMiddlewareChain(
  loggingMiddleware,
  rateLimitMiddleware,
  authMiddleware
);
```

### 2.4 NextAuth.js（Auth.js）との統合

```typescript
// middleware.ts — NextAuth.js v5 との統合
import { auth } from './auth';

export default auth((req) => {
  const { pathname } = req.nextUrl;
  const isLoggedIn = !!req.auth;

  // 管理者ルートの保護
  if (pathname.startsWith('/admin')) {
    if (!isLoggedIn) {
      return Response.redirect(new URL('/login', req.url));
    }
    if (req.auth?.user?.role !== 'admin') {
      return Response.redirect(new URL('/403', req.url));
    }
  }

  // 認証エリアの保護
  if (pathname.startsWith('/dashboard') || pathname.startsWith('/settings')) {
    if (!isLoggedIn) {
      const loginUrl = new URL('/login', req.url);
      loginUrl.searchParams.set('callbackUrl', pathname);
      return Response.redirect(loginUrl);
    }
  }

  // ログイン済みユーザーの認証ページアクセスを防止
  if (isLoggedIn && (pathname === '/login' || pathname === '/register')) {
    return Response.redirect(new URL('/dashboard', req.url));
  }
});

export const config = {
  matcher: ['/((?!api|_next/static|_next/image|favicon.ico).*)'],
};

// auth.ts — NextAuth.js v5 の設定
import NextAuth from 'next-auth';
import Credentials from 'next-auth/providers/credentials';
import GitHub from 'next-auth/providers/github';
import Google from 'next-auth/providers/google';
import { PrismaAdapter } from '@auth/prisma-adapter';
import { prisma } from '@/lib/prisma';
import bcrypt from 'bcryptjs';

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: PrismaAdapter(prisma),
  providers: [
    GitHub({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    }),
    Google({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    Credentials({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          throw new Error('Email and password are required');
        }

        const user = await prisma.user.findUnique({
          where: { email: credentials.email as string },
        });

        if (!user || !user.hashedPassword) {
          throw new Error('Invalid credentials');
        }

        const isValid = await bcrypt.compare(
          credentials.password as string,
          user.hashedPassword
        );

        if (!isValid) {
          throw new Error('Invalid credentials');
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name,
          role: user.role,
          image: user.image,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.role = user.role;
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.role = token.role as string;
        session.user.id = token.id as string;
      }
      return session;
    },
    async authorized({ auth, request }) {
      const isLoggedIn = !!auth?.user;
      const isOnDashboard = request.nextUrl.pathname.startsWith('/dashboard');
      if (isOnDashboard) return isLoggedIn;
      return true;
    },
  },
  pages: {
    signIn: '/login',
    error: '/auth/error',
    newUser: '/onboarding',
  },
  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30日
  },
});
```

---

## 3. Layout での認証ガード

### 3.1 Server Component による認証チェック

```typescript
// app/(app)/layout.tsx — 認証必要エリアのレイアウト
import { redirect } from 'next/navigation';
import { getSession } from '@/shared/lib/auth';
import { SessionProvider } from '@/shared/providers/SessionProvider';
import { Sidebar } from '@/shared/components/Sidebar';
import { Header } from '@/shared/components/Header';

export default async function AuthenticatedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();

  if (!session) {
    redirect('/login');
  }

  // セッションの有効期限チェック
  if (session.expiresAt && new Date(session.expiresAt) < new Date()) {
    redirect('/login?reason=session_expired');
  }

  return (
    <SessionProvider session={session}>
      <div className="flex h-screen">
        <Sidebar user={session.user} />
        <div className="flex-1 flex flex-col overflow-hidden">
          <Header user={session.user} />
          <main className="flex-1 overflow-y-auto p-6">
            {children}
          </main>
        </div>
      </div>
    </SessionProvider>
  );
}

// app/(app)/admin/layout.tsx — 管理者専用エリア
import { redirect } from 'next/navigation';
import { getSession } from '@/shared/lib/auth';

export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();

  if (!session) {
    redirect('/login');
  }

  if (session.user.role !== 'admin') {
    redirect('/dashboard'); // 権限不足はダッシュボードへ
  }

  return (
    <div className="admin-layout">
      <div className="bg-yellow-50 border-b border-yellow-200 px-4 py-2 text-sm text-yellow-800">
        管理者モード — 操作には十分注意してください
      </div>
      {children}
    </div>
  );
}
```

### 3.2 ネストされたレイアウトによる段階的保護

```typescript
// ============================================
// ネストレイアウトによる段階的認証の例
// ============================================

// app/(marketing)/layout.tsx — 公開エリア（認証不要）
export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="marketing-layout">
      <PublicNavbar />
      {children}
      <Footer />
    </div>
  );
}

// app/(app)/layout.tsx — 認証エリア（ログイン必須）
export default async function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();
  if (!session) redirect('/login');

  return (
    <SessionProvider session={session}>
      <AppShell user={session.user}>
        {children}
      </AppShell>
    </SessionProvider>
  );
}

// app/(app)/settings/layout.tsx — 設定エリア（メール認証済み必須）
export default async function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();

  if (!session?.user.emailVerified) {
    redirect('/verify-email?reason=settings_access');
  }

  return (
    <div className="max-w-4xl mx-auto">
      <SettingsNav />
      {children}
    </div>
  );
}

// app/(app)/settings/billing/layout.tsx — 課金設定（追加認証必須）
export default async function BillingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();

  // 課金関連は最終認証から10分以内であることを要求
  const lastAuthAt = session?.user.lastAuthenticatedAt;
  const tenMinutesAgo = new Date(Date.now() - 10 * 60 * 1000);

  if (!lastAuthAt || new Date(lastAuthAt) < tenMinutesAgo) {
    redirect('/reauth?reason=billing_access&callbackUrl=/settings/billing');
  }

  return <>{children}</>;
}
```

### 3.3 SessionProvider の実装

```typescript
// shared/providers/SessionProvider.tsx
'use client';

import {
  createContext,
  useContext,
  useCallback,
  useEffect,
  useState,
  useTransition,
} from 'react';
import { useRouter } from 'next/navigation';

// セッション型定義
interface Session {
  user: {
    id: string;
    email: string;
    name: string;
    role: string;
    image?: string;
    emailVerified?: boolean;
    lastAuthenticatedAt?: string;
  };
  expiresAt: string;
}

interface SessionContextType {
  session: Session | null;
  isLoading: boolean;
  update: () => Promise<void>;
  signOut: () => Promise<void>;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({
  session: initialSession,
  children,
}: {
  session: Session;
  children: React.ReactNode;
}) {
  const [session, setSession] = useState<Session | null>(initialSession);
  const [isLoading, setIsLoading] = useState(false);
  const [isPending, startTransition] = useTransition();
  const router = useRouter();

  // セッションの自動更新（15分ごと）
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/auth/session');
        if (res.ok) {
          const data = await res.json();
          setSession(data.session);
        } else if (res.status === 401) {
          setSession(null);
          router.push('/login?reason=session_expired');
        }
      } catch (error) {
        console.error('Session refresh failed:', error);
      }
    }, 15 * 60 * 1000);

    return () => clearInterval(interval);
  }, [router]);

  // ウィンドウフォーカス時にセッションを確認
  useEffect(() => {
    const handleFocus = async () => {
      try {
        const res = await fetch('/api/auth/session');
        if (res.ok) {
          const data = await res.json();
          setSession(data.session);
        } else if (res.status === 401) {
          setSession(null);
          router.push('/login?reason=session_expired');
        }
      } catch (error) {
        console.error('Session check on focus failed:', error);
      }
    };

    window.addEventListener('focus', handleFocus);
    return () => window.removeEventListener('focus', handleFocus);
  }, [router]);

  // セッション更新
  const update = useCallback(async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/auth/session', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setSession(data.session);
        startTransition(() => {
          router.refresh();
        });
      }
    } finally {
      setIsLoading(false);
    }
  }, [router]);

  // サインアウト
  const signOut = useCallback(async () => {
    setIsLoading(true);
    try {
      await fetch('/api/auth/signout', { method: 'POST' });
      setSession(null);
      router.push('/login');
    } finally {
      setIsLoading(false);
    }
  }, [router]);

  return (
    <SessionContext.Provider value={{ session, isLoading: isLoading || isPending, update, signOut }}>
      {children}
    </SessionContext.Provider>
  );
}

// カスタムフック
export function useSession() {
  const context = useContext(SessionContext);
  if (!context) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}

// 認証必須のカスタムフック
export function useRequireAuth() {
  const { session, isLoading } = useSession();
  const router = useRouter();

  useEffect(() => {
    if (!isLoading && !session) {
      router.push('/login');
    }
  }, [session, isLoading, router]);

  return { session, isLoading };
}
```

---

## 4. ロールベースアクセス制御（RBAC）

### 4.1 権限モデルの設計

```typescript
// ============================================
// 包括的な RBAC 設計
// ============================================

// 基本型定義
type Role = 'viewer' | 'user' | 'editor' | 'moderator' | 'admin' | 'superadmin';

type Resource = 'users' | 'posts' | 'comments' | 'orders' | 'products' | 'settings' | 'analytics' | 'billing';

type Action = 'create' | 'read' | 'update' | 'delete' | 'publish' | 'moderate' | 'export';

// Permission を文字列リテラル型で定義
type Permission = `${Resource}:${Action}`;

// ロールの階層定義
const roleHierarchy: Record<Role, Role[]> = {
  viewer: [],
  user: ['viewer'],
  editor: ['user', 'viewer'],
  moderator: ['user', 'viewer'],
  admin: ['editor', 'moderator', 'user', 'viewer'],
  superadmin: ['admin', 'editor', 'moderator', 'user', 'viewer'],
};

// ロールごとの直接権限（継承分は含まない）
const directPermissions: Record<Role, Permission[]> = {
  viewer: [
    'posts:read',
    'comments:read',
    'products:read',
  ],
  user: [
    'posts:create',
    'comments:create',
    'orders:create',
    'orders:read',
  ],
  editor: [
    'posts:update',
    'posts:publish',
    'posts:delete',
    'comments:update',
    'products:create',
    'products:update',
  ],
  moderator: [
    'comments:moderate',
    'comments:delete',
    'users:read',
  ],
  admin: [
    'users:create',
    'users:update',
    'users:delete',
    'products:delete',
    'orders:update',
    'orders:delete',
    'settings:read',
    'settings:update',
    'analytics:read',
    'analytics:export',
  ],
  superadmin: [
    'billing:read',
    'billing:update',
    'settings:delete',
  ],
};

// 全権限の取得（継承を含む）
function getAllPermissions(role: Role): Permission[] {
  const direct = directPermissions[role] || [];
  const inherited = roleHierarchy[role]
    .flatMap(parentRole => getAllPermissions(parentRole));

  return [...new Set([...direct, ...inherited])];
}

// 権限チェック
function hasPermission(role: Role, permission: Permission): boolean {
  return getAllPermissions(role).includes(permission);
}

// 複数権限のチェック（AND条件）
function hasAllPermissions(role: Role, permissions: Permission[]): boolean {
  const userPermissions = getAllPermissions(role);
  return permissions.every(p => userPermissions.includes(p));
}

// 複数権限のチェック（OR条件）
function hasAnyPermission(role: Role, permissions: Permission[]): boolean {
  const userPermissions = getAllPermissions(role);
  return permissions.some(p => userPermissions.includes(p));
}

// 使用例
console.log(hasPermission('editor', 'posts:publish'));  // true
console.log(hasPermission('editor', 'users:delete'));   // false
console.log(hasPermission('admin', 'posts:publish'));   // true（editor から継承）
console.log(hasAllPermissions('admin', ['users:create', 'users:delete'])); // true
```

### 4.2 React コンポーネントでの RBAC 実装

```typescript
// ============================================
// 権限チェックコンポーネント
// ============================================

// components/auth/RequirePermission.tsx
'use client';

import { useSession } from '@/shared/providers/SessionProvider';

interface RequirePermissionProps {
  permission: Permission | Permission[];
  mode?: 'all' | 'any'; // all: AND条件, any: OR条件
  children: React.ReactNode;
  fallback?: React.ReactNode;
  showDisabled?: boolean; // true の場合、非活性で表示
}

export function RequirePermission({
  permission,
  mode = 'all',
  children,
  fallback = null,
  showDisabled = false,
}: RequirePermissionProps) {
  const { session } = useSession();

  if (!session) return fallback;

  const permissions = Array.isArray(permission) ? permission : [permission];
  const hasAccess = mode === 'all'
    ? hasAllPermissions(session.user.role as Role, permissions)
    : hasAnyPermission(session.user.role as Role, permissions);

  if (!hasAccess) {
    if (showDisabled) {
      return (
        <div className="opacity-50 cursor-not-allowed pointer-events-none">
          {children}
        </div>
      );
    }
    return <>{fallback}</>;
  }

  return <>{children}</>;
}

// components/auth/RequireRole.tsx
export function RequireRole({
  role,
  children,
  fallback = null,
}: {
  role: Role | Role[];
  children: React.ReactNode;
  fallback?: React.ReactNode;
}) {
  const { session } = useSession();

  if (!session) return fallback;

  const roles = Array.isArray(role) ? role : [role];
  const userRole = session.user.role as Role;

  // ロール階層を考慮したチェック
  const hasRole = roles.some(r => {
    if (r === userRole) return true;
    // ユーザーのロールが、要求されたロールを継承しているか
    return roleHierarchy[userRole]?.includes(r) ?? false;
  });

  if (!hasRole) return <>{fallback}</>;

  return <>{children}</>;
}

// ============================================
// 使用例
// ============================================

function UserManagementPage() {
  const { session } = useSession();

  return (
    <div>
      <h1>ユーザー管理</h1>

      {/* 管理者のみ新規ユーザー作成ボタンを表示 */}
      <RequirePermission permission="users:create">
        <Button onClick={handleCreateUser}>
          新規ユーザー作成
        </Button>
      </RequirePermission>

      {/* ユーザー一覧（閲覧権限が必要） */}
      <RequirePermission
        permission="users:read"
        fallback={<Alert>ユーザー一覧を閲覧する権限がありません</Alert>}
      >
        <UserTable>
          {users.map(user => (
            <UserRow key={user.id} user={user}>
              {/* 編集ボタン：権限なしの場合は非活性で表示 */}
              <RequirePermission
                permission="users:update"
                showDisabled
              >
                <EditButton userId={user.id} />
              </RequirePermission>

              {/* 削除ボタン：権限なしの場合はツールチップ付きで表示 */}
              <RequirePermission
                permission="users:delete"
                fallback={
                  <Tooltip content="削除権限がありません">
                    <DeleteButton disabled userId={user.id} />
                  </Tooltip>
                }
              >
                <DeleteButton userId={user.id} />
              </RequirePermission>
            </UserRow>
          ))}
        </UserTable>
      </RequirePermission>

      {/* 複数権限の組み合わせ */}
      <RequirePermission
        permission={['analytics:read', 'analytics:export']}
        mode="all"
      >
        <AnalyticsDashboard />
      </RequirePermission>
    </div>
  );
}
```

### 4.3 サーバーサイドでの RBAC 実装

```typescript
// ============================================
// API ルートでの権限チェック
// ============================================

// lib/auth/withPermission.ts
import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';

type ApiHandler = (
  request: NextRequest,
  context: { params: Record<string, string>; session: Session }
) => Promise<NextResponse>;

// 権限チェックの Higher-Order Function
export function withPermission(
  permission: Permission | Permission[],
  handler: ApiHandler,
  mode: 'all' | 'any' = 'all'
) {
  return async (
    request: NextRequest,
    context: { params: Record<string, string> }
  ) => {
    const session = await getSession();

    if (!session) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }

    const permissions = Array.isArray(permission) ? permission : [permission];
    const userRole = session.user.role as Role;

    const hasAccess = mode === 'all'
      ? hasAllPermissions(userRole, permissions)
      : hasAnyPermission(userRole, permissions);

    if (!hasAccess) {
      return NextResponse.json(
        {
          error: 'Insufficient permissions',
          required: permissions,
          userRole: userRole,
        },
        { status: 403 }
      );
    }

    return handler(request, { ...context, session });
  };
}

// 使用例: API ルート
// app/api/users/route.ts
import { withPermission } from '@/lib/auth/withPermission';

export const GET = withPermission('users:read', async (request, { session }) => {
  const users = await prisma.user.findMany({
    select: {
      id: true,
      name: true,
      email: true,
      role: true,
      createdAt: true,
    },
  });

  return NextResponse.json({ users });
});

export const POST = withPermission('users:create', async (request, { session }) => {
  const body = await request.json();

  // 監査ログ
  await auditLog({
    action: 'users:create',
    actorId: session.user.id,
    details: { email: body.email },
  });

  const user = await prisma.user.create({
    data: body,
  });

  return NextResponse.json({ user }, { status: 201 });
});

export const DELETE = withPermission(
  ['users:delete', 'admin:access'],
  async (request, { session }) => {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('id');

    if (userId === session.user.id) {
      return NextResponse.json(
        { error: 'Cannot delete your own account' },
        { status: 400 }
      );
    }

    await prisma.user.delete({ where: { id: userId! } });

    return NextResponse.json({ success: true });
  },
  'all'
);
```

---

## 5. 属性ベースアクセス制御（ABAC）

### 5.1 ABAC の概念と RBAC との違い

RBAC はロール（役割）に基づいてアクセスを制御するのに対し、ABAC は属性（Attribute）に基づいてより柔軟なアクセス制御を実現する。ユーザーの属性、リソースの属性、環境条件を組み合わせてポリシーを定義できる。

```
【RBAC vs ABAC 比較】

RBAC:
  判定基準 = ユーザーのロール
  例: 「admin ロールはユーザーを削除できる」
  長所: シンプル、理解しやすい
  短所: 複雑な条件に対応しづらい

ABAC:
  判定基準 = ユーザー属性 + リソース属性 + 環境条件
  例: 「部門マネージャーは自部門のユーザーのみ編集できる、
       ただし営業時間内に限る」
  長所: 非常に柔軟、きめ細かい制御
  短所: 複雑になりがち、デバッグが難しい

ハイブリッド（推奨）:
  RBAC で大まかな制御 + ABAC で細かい条件を追加
  例: 「editor ロール」 AND 「リソースの所有者」 AND 「公開前」
```

### 5.2 ABAC の実装

```typescript
// ============================================
// ABAC（属性ベースアクセス制御）の実装
// ============================================

// 属性の型定義
interface UserAttributes {
  id: string;
  role: Role;
  department: string;
  location: string;
  clearanceLevel: number;
  isActive: boolean;
}

interface ResourceAttributes {
  ownerId: string;
  department: string;
  sensitivity: 'public' | 'internal' | 'confidential' | 'secret';
  status: 'draft' | 'review' | 'published' | 'archived';
  createdAt: Date;
}

interface EnvironmentAttributes {
  currentTime: Date;
  ipAddress: string;
  isBusinessHours: boolean;
  isVpnConnected: boolean;
  deviceType: 'desktop' | 'mobile' | 'tablet';
}

// ポリシーの型定義
interface Policy {
  name: string;
  description: string;
  effect: 'allow' | 'deny';
  condition: (
    user: UserAttributes,
    resource: ResourceAttributes,
    environment: EnvironmentAttributes,
    action: string
  ) => boolean;
}

// ポリシーエンジン
class PolicyEngine {
  private policies: Policy[] = [];

  addPolicy(policy: Policy): void {
    this.policies.push(policy);
  }

  evaluate(
    user: UserAttributes,
    resource: ResourceAttributes,
    environment: EnvironmentAttributes,
    action: string
  ): { allowed: boolean; matchedPolicies: string[] } {
    const matchedPolicies: string[] = [];
    let hasExplicitDeny = false;
    let hasExplicitAllow = false;

    for (const policy of this.policies) {
      if (policy.condition(user, resource, environment, action)) {
        matchedPolicies.push(policy.name);
        if (policy.effect === 'deny') {
          hasExplicitDeny = true;
        } else {
          hasExplicitAllow = true;
        }
      }
    }

    // Deny は常に Allow より優先（Deny-Override）
    return {
      allowed: hasExplicitAllow && !hasExplicitDeny,
      matchedPolicies,
    };
  }
}

// ポリシーの定義例
const policyEngine = new PolicyEngine();

// ポリシー1: 所有者は自分のリソースを編集できる
policyEngine.addPolicy({
  name: 'owner-can-edit',
  description: '所有者は自分のリソースを編集できる',
  effect: 'allow',
  condition: (user, resource, env, action) => {
    return action === 'update' && resource.ownerId === user.id;
  },
});

// ポリシー2: 同じ部門のマネージャーは閲覧できる
policyEngine.addPolicy({
  name: 'department-manager-read',
  description: '同じ部門のマネージャーは閲覧できる',
  effect: 'allow',
  condition: (user, resource, env, action) => {
    return (
      action === 'read' &&
      user.role === 'admin' &&
      user.department === resource.department
    );
  },
});

// ポリシー3: 機密データは営業時間内かつVPN接続時のみアクセス可能
policyEngine.addPolicy({
  name: 'confidential-business-hours-vpn',
  description: '機密データは営業時間内かつVPN接続時のみ',
  effect: 'deny',
  condition: (user, resource, env, action) => {
    return (
      resource.sensitivity === 'confidential' &&
      (!env.isBusinessHours || !env.isVpnConnected)
    );
  },
});

// ポリシー4: アーカイブされたリソースは削除不可
policyEngine.addPolicy({
  name: 'no-delete-archived',
  description: 'アーカイブ済みリソースは削除不可',
  effect: 'deny',
  condition: (user, resource, env, action) => {
    return action === 'delete' && resource.status === 'archived';
  },
});

// ポリシー5: クリアランスレベルに基づくアクセス制御
policyEngine.addPolicy({
  name: 'clearance-level-access',
  description: 'クリアランスレベルに基づくアクセス',
  effect: 'allow',
  condition: (user, resource, env, action) => {
    const sensitivityLevels = {
      public: 0,
      internal: 1,
      confidential: 2,
      secret: 3,
    };
    return user.clearanceLevel >= sensitivityLevels[resource.sensitivity];
  },
});

// 使用例
const result = policyEngine.evaluate(
  {
    id: 'user-1',
    role: 'editor',
    department: 'engineering',
    location: 'tokyo',
    clearanceLevel: 2,
    isActive: true,
  },
  {
    ownerId: 'user-1',
    department: 'engineering',
    sensitivity: 'confidential',
    status: 'draft',
    createdAt: new Date(),
  },
  {
    currentTime: new Date(),
    ipAddress: '192.168.1.1',
    isBusinessHours: true,
    isVpnConnected: true,
    deviceType: 'desktop',
  },
  'update'
);

console.log(result);
// { allowed: true, matchedPolicies: ['owner-can-edit', 'clearance-level-access'] }
```

### 5.3 RBAC と ABAC のハイブリッド実装

```typescript
// ============================================
// ハイブリッドアクセス制御
// ============================================

interface AccessDecision {
  allowed: boolean;
  reason: string;
  requiredActions?: string[];
}

class HybridAccessControl {
  // Step 1: RBAC でベースライン権限をチェック
  private checkRBAC(role: Role, permission: Permission): boolean {
    return hasPermission(role, permission);
  }

  // Step 2: ABAC で追加条件をチェック
  private checkABAC(
    user: UserAttributes,
    resource: ResourceAttributes,
    environment: EnvironmentAttributes,
    action: string
  ): { allowed: boolean; matchedPolicies: string[] } {
    return policyEngine.evaluate(user, resource, environment, action);
  }

  // 統合された権限チェック
  authorize(
    user: UserAttributes,
    resource: ResourceAttributes,
    environment: EnvironmentAttributes,
    permission: Permission,
    action: string
  ): AccessDecision {
    // Step 1: RBAC チェック
    if (!this.checkRBAC(user.role, permission)) {
      return {
        allowed: false,
        reason: `Role '${user.role}' does not have permission '${permission}'`,
      };
    }

    // Step 2: ABAC チェック
    const abacResult = this.checkABAC(user, resource, environment, action);
    if (!abacResult.allowed) {
      return {
        allowed: false,
        reason: `ABAC policy denied: ${abacResult.matchedPolicies.join(', ')}`,
      };
    }

    // Step 3: アクティブユーザーチェック
    if (!user.isActive) {
      return {
        allowed: false,
        reason: 'User account is deactivated',
      };
    }

    return {
      allowed: true,
      reason: 'Access granted',
    };
  }
}

// API ルートでの使用
const accessControl = new HybridAccessControl();

export async function PUT(request: NextRequest) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const body = await request.json();
  const resource = await getResource(body.resourceId);

  const decision = accessControl.authorize(
    session.user as UserAttributes,
    resource as ResourceAttributes,
    {
      currentTime: new Date(),
      ipAddress: request.headers.get('x-forwarded-for') || '',
      isBusinessHours: checkBusinessHours(),
      isVpnConnected: checkVpnConnection(request),
      deviceType: detectDeviceType(request),
    },
    'posts:update',
    'update'
  );

  if (!decision.allowed) {
    return NextResponse.json(
      { error: decision.reason },
      { status: 403 }
    );
  }

  // 更新処理...
  const updated = await updateResource(body);
  return NextResponse.json({ data: updated });
}
```

---

## 6. 認証フロー

### 6.1 ログイン / ログアウトフロー

```typescript
// ============================================
// ログインフロー — Server Actions
// ============================================

// app/login/page.tsx
import { LoginForm } from '@/components/auth/LoginForm';

export default function LoginPage({
  searchParams,
}: {
  searchParams: { callbackUrl?: string; error?: string; reason?: string };
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8">
        <div className="text-center">
          <h2 className="text-3xl font-bold">ログイン</h2>
          {searchParams.reason === 'session_expired' && (
            <p className="mt-2 text-sm text-amber-600">
              セッションが期限切れです。再度ログインしてください。
            </p>
          )}
          {searchParams.error && (
            <p className="mt-2 text-sm text-red-600">
              {searchParams.error}
            </p>
          )}
        </div>
        <LoginForm callbackUrl={searchParams.callbackUrl} />
      </div>
    </div>
  );
}

// components/auth/LoginForm.tsx
'use client';

import { useFormState, useFormStatus } from 'react-dom';
import { login } from '@/actions/auth';
import { useState } from 'react';

function SubmitButton() {
  const { pending } = useFormStatus();
  return (
    <button
      type="submit"
      disabled={pending}
      className="w-full flex justify-center py-2 px-4 border border-transparent
                 rounded-md shadow-sm text-sm font-medium text-white bg-blue-600
                 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {pending ? 'ログイン中...' : 'ログイン'}
    </button>
  );
}

export function LoginForm({ callbackUrl }: { callbackUrl?: string }) {
  const [state, formAction] = useFormState(login, { error: null });
  const [showPassword, setShowPassword] = useState(false);

  return (
    <form action={formAction} className="mt-8 space-y-6">
      <input type="hidden" name="callbackUrl" value={callbackUrl || '/dashboard'} />

      <div className="space-y-4">
        <div>
          <label htmlFor="email" className="block text-sm font-medium text-gray-700">
            メールアドレス
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            required
            className="mt-1 block w-full px-3 py-2 border border-gray-300
                       rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700">
            パスワード
          </label>
          <div className="relative">
            <input
              id="password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="current-password"
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300
                         rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
            />
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="absolute inset-y-0 right-0 pr-3 flex items-center"
            >
              {showPassword ? '非表示' : '表示'}
            </button>
          </div>
        </div>
      </div>

      {state?.error && (
        <div className="text-sm text-red-600 bg-red-50 p-3 rounded-md">
          {state.error}
        </div>
      )}

      <SubmitButton />

      <div className="flex items-center justify-between text-sm">
        <a href="/forgot-password" className="text-blue-600 hover:text-blue-500">
          パスワードを忘れた場合
        </a>
        <a href="/register" className="text-blue-600 hover:text-blue-500">
          新規登録
        </a>
      </div>
    </form>
  );
}

// actions/auth.ts — Server Actions
'use server';

import { redirect } from 'next/navigation';
import { cookies } from 'next/headers';
import { z } from 'zod';
import bcrypt from 'bcryptjs';
import { SignJWT } from 'jose';
import { prisma } from '@/lib/prisma';

const loginSchema = z.object({
  email: z.string().email('有効なメールアドレスを入力してください'),
  password: z.string().min(8, 'パスワードは8文字以上必要です'),
  callbackUrl: z.string().optional(),
});

export async function login(
  prevState: { error: string | null },
  formData: FormData
) {
  // バリデーション
  const parsed = loginSchema.safeParse({
    email: formData.get('email'),
    password: formData.get('password'),
    callbackUrl: formData.get('callbackUrl'),
  });

  if (!parsed.success) {
    return { error: parsed.error.errors[0].message };
  }

  const { email, password, callbackUrl } = parsed.data;

  // ユーザー検索
  const user = await prisma.user.findUnique({
    where: { email },
  });

  if (!user || !user.hashedPassword) {
    // タイミング攻撃対策: 存在しないユーザーでも同程度の処理時間にする
    await bcrypt.hash('dummy-password', 12);
    return { error: 'メールアドレスまたはパスワードが正しくありません' };
  }

  // パスワード検証
  const isValid = await bcrypt.compare(password, user.hashedPassword);
  if (!isValid) {
    // ログイン失敗回数を記録
    await prisma.user.update({
      where: { id: user.id },
      data: {
        failedLoginAttempts: { increment: 1 },
        lastFailedLoginAt: new Date(),
      },
    });

    // アカウントロック判定
    if (user.failedLoginAttempts >= 4) {
      await prisma.user.update({
        where: { id: user.id },
        data: { lockedUntil: new Date(Date.now() + 15 * 60 * 1000) },
      });
      return { error: 'アカウントがロックされました。15分後に再試行してください。' };
    }

    return { error: 'メールアドレスまたはパスワードが正しくありません' };
  }

  // アカウントロックチェック
  if (user.lockedUntil && new Date(user.lockedUntil) > new Date()) {
    return { error: 'アカウントがロックされています。しばらく待ってから再試行してください。' };
  }

  // ログイン成功: 失敗回数リセット
  await prisma.user.update({
    where: { id: user.id },
    data: {
      failedLoginAttempts: 0,
      lastFailedLoginAt: null,
      lockedUntil: null,
      lastLoginAt: new Date(),
    },
  });

  // JWT トークン生成
  const secret = new TextEncoder().encode(process.env.JWT_SECRET!);
  const token = await new SignJWT({
    sub: user.id,
    role: user.role,
    email: user.email,
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('7d')
    .sign(secret);

  // セッション Cookie 設定
  const cookieStore = await cookies();
  cookieStore.set('session-token', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 7 * 24 * 60 * 60, // 7日
    path: '/',
  });

  // 監査ログ
  await prisma.auditLog.create({
    data: {
      userId: user.id,
      action: 'login',
      ipAddress: '', // Request から取得
      userAgent: '',
    },
  });

  redirect(callbackUrl || '/dashboard');
}

// ログアウト
export async function logout() {
  const cookieStore = await cookies();
  const token = cookieStore.get('session-token')?.value;

  if (token) {
    // トークンをブラックリストに追加（オプション）
    await prisma.revokedToken.create({
      data: {
        token,
        revokedAt: new Date(),
      },
    });
  }

  cookieStore.delete('session-token');
  redirect('/login');
}
```

### 6.2 セッション管理

```typescript
// ============================================
// セッション管理ユーティリティ
// ============================================

// lib/auth/session.ts
import { cookies } from 'next/headers';
import { jwtVerify, SignJWT } from 'jose';
import { prisma } from '@/lib/prisma';

const JWT_SECRET = new TextEncoder().encode(process.env.JWT_SECRET!);

export interface SessionUser {
  id: string;
  email: string;
  name: string;
  role: string;
  image?: string;
  emailVerified?: boolean;
  lastAuthenticatedAt?: string;
}

export interface Session {
  user: SessionUser;
  expiresAt: string;
  issuedAt: string;
}

// セッション取得
export async function getSession(): Promise<Session | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get('session-token')?.value;

  if (!token) return null;

  try {
    // JWT 検証
    const { payload } = await jwtVerify(token, JWT_SECRET);

    // ブラックリストチェック（オプション）
    const isRevoked = await prisma.revokedToken.findFirst({
      where: { token },
    });
    if (isRevoked) return null;

    // ユーザー情報を DB から取得（最新の状態を反映）
    const user = await prisma.user.findUnique({
      where: { id: payload.sub as string },
      select: {
        id: true,
        email: true,
        name: true,
        role: true,
        image: true,
        emailVerified: true,
        isActive: true,
      },
    });

    if (!user || !user.isActive) return null;

    return {
      user: {
        id: user.id,
        email: user.email,
        name: user.name || '',
        role: user.role,
        image: user.image || undefined,
        emailVerified: !!user.emailVerified,
      },
      expiresAt: new Date((payload.exp || 0) * 1000).toISOString(),
      issuedAt: new Date((payload.iat || 0) * 1000).toISOString(),
    };
  } catch {
    return null;
  }
}

// セッション更新（スライディングウィンドウ）
export async function refreshSession(): Promise<Session | null> {
  const session = await getSession();
  if (!session) return null;

  // 新しいトークンを発行
  const token = await new SignJWT({
    sub: session.user.id,
    role: session.user.role,
    email: session.user.email,
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('7d')
    .sign(JWT_SECRET);

  const cookieStore = await cookies();
  cookieStore.set('session-token', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: 7 * 24 * 60 * 60,
    path: '/',
  });

  return getSession();
}

// セッション削除
export async function deleteSession(): Promise<void> {
  const cookieStore = await cookies();
  cookieStore.delete('session-token');
}

// API Route: セッション確認
// app/api/auth/session/route.ts
export async function GET() {
  const session = await getSession();

  if (!session) {
    return NextResponse.json({ error: 'No session' }, { status: 401 });
  }

  return NextResponse.json({ session });
}

export async function POST() {
  const session = await refreshSession();

  if (!session) {
    return NextResponse.json({ error: 'Session refresh failed' }, { status: 401 });
  }

  return NextResponse.json({ session });
}
```

### 6.3 トークンリフレッシュ戦略

```typescript
// ============================================
// トークンリフレッシュの実装パターン
// ============================================

// パターン1: Axios インターセプターによる自動リフレッシュ
import axios from 'axios';

const api = axios.create({
  baseURL: '/api',
  withCredentials: true,
});

// リフレッシュ中の複数リクエストを管理
let isRefreshing = false;
let failedQueue: Array<{
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
}> = [];

const processQueue = (error: unknown = null) => {
  failedQueue.forEach(({ resolve, reject }) => {
    if (error) {
      reject(error);
    } else {
      resolve(undefined);
    }
  });
  failedQueue = [];
};

api.interceptors.response.use(
  (response) => {
    // X-Token-Refresh ヘッダーがある場合、バックグラウンドでリフレッシュ
    if (response.headers['x-token-refresh'] === 'true') {
      refreshToken().catch(console.error);
    }
    return response;
  },
  async (error) => {
    const originalRequest = error.config;

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // リフレッシュ中の場合はキューに追加
        return new Promise((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        }).then(() => api(originalRequest));
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        await refreshToken();
        processQueue();
        return api(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError);
        // ログインページへリダイレクト
        window.location.href = '/login?reason=session_expired';
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    return Promise.reject(error);
  }
);

async function refreshToken(): Promise<void> {
  const response = await fetch('/api/auth/session', { method: 'POST' });
  if (!response.ok) {
    throw new Error('Token refresh failed');
  }
}

// パターン2: fetch のラッパー
export async function fetchWithAuth(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  let response = await fetch(url, {
    ...options,
    credentials: 'include',
  });

  if (response.status === 401) {
    // トークンリフレッシュを試行
    const refreshResponse = await fetch('/api/auth/session', {
      method: 'POST',
      credentials: 'include',
    });

    if (refreshResponse.ok) {
      // リフレッシュ成功: 元のリクエストを再実行
      response = await fetch(url, {
        ...options,
        credentials: 'include',
      });
    } else {
      // リフレッシュ失敗: ログインへリダイレクト
      window.location.href = '/login?reason=session_expired';
    }
  }

  return response;
}
```

---

## 7. 他フレームワークでの認証ガード

### 7.1 React Router v6 での認証ガード

```typescript
// ============================================
// React Router v6 認証ガード
// ============================================

// auth/AuthContext.tsx
import { createContext, useContext, useState, useCallback } from 'react';

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (credentials: Credentials) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // 初回ロード時にセッション確認
  useEffect(() => {
    checkSession().finally(() => setIsLoading(false));
  }, []);

  const login = useCallback(async (credentials: Credentials) => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || 'Login failed');
    }

    const { user: loggedInUser } = await response.json();
    setUser(loggedInUser);
  }, []);

  const logout = useCallback(async () => {
    await fetch('/api/auth/logout', { method: 'POST' });
    setUser(null);
  }, []);

  const checkSession = useCallback(async () => {
    try {
      const response = await fetch('/api/auth/session');
      if (response.ok) {
        const { user: sessionUser } = await response.json();
        setUser(sessionUser);
      }
    } catch {
      setUser(null);
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}

// auth/ProtectedRoute.tsx
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from './AuthContext';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: Role | Role[];
  requiredPermission?: Permission | Permission[];
  fallbackPath?: string;
}

export function ProtectedRoute({
  children,
  requiredRole,
  requiredPermission,
  fallbackPath = '/login',
}: ProtectedRouteProps) {
  const { user, isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return <FullPageSpinner />;
  }

  if (!isAuthenticated) {
    return (
      <Navigate
        to={fallbackPath}
        state={{ from: location.pathname }}
        replace
      />
    );
  }

  // ロールチェック
  if (requiredRole) {
    const roles = Array.isArray(requiredRole) ? requiredRole : [requiredRole];
    if (!roles.includes(user!.role as Role)) {
      return <Navigate to="/403" replace />;
    }
  }

  // 権限チェック
  if (requiredPermission) {
    const permissions = Array.isArray(requiredPermission)
      ? requiredPermission
      : [requiredPermission];
    const hasAccess = permissions.every(p =>
      hasPermission(user!.role as Role, p)
    );
    if (!hasAccess) {
      return <Navigate to="/403" replace />;
    }
  }

  return <>{children}</>;
}

// ルート定義
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: [
      // 公開ルート
      { index: true, element: <HomePage /> },
      { path: 'login', element: <LoginPage /> },
      { path: 'register', element: <RegisterPage /> },

      // 認証が必要なルート
      {
        path: 'dashboard',
        element: (
          <ProtectedRoute>
            <DashboardPage />
          </ProtectedRoute>
        ),
      },
      {
        path: 'settings/*',
        element: (
          <ProtectedRoute>
            <SettingsLayout />
          </ProtectedRoute>
        ),
        children: [
          { path: 'profile', element: <ProfileSettings /> },
          { path: 'security', element: <SecuritySettings /> },
          {
            path: 'billing',
            element: (
              <ProtectedRoute requiredPermission="billing:read">
                <BillingSettings />
              </ProtectedRoute>
            ),
          },
        ],
      },

      // 管理者ルート
      {
        path: 'admin/*',
        element: (
          <ProtectedRoute requiredRole="admin">
            <AdminLayout />
          </ProtectedRoute>
        ),
        children: [
          { path: 'users', element: <UserManagement /> },
          { path: 'analytics', element: <AdminAnalytics /> },
        ],
      },

      // エラーページ
      { path: '403', element: <ForbiddenPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]);

function App() {
  return (
    <AuthProvider>
      <RouterProvider router={router} />
    </AuthProvider>
  );
}
```

### 7.2 Vue Router でのナビゲーションガード

```typescript
// ============================================
// Vue Router ナビゲーションガード
// ============================================

// router/index.ts
import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router';
import { useAuthStore } from '@/stores/auth';

// ルートメタ型の拡張
declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean;
    requiredRoles?: string[];
    requiredPermissions?: string[];
    title?: string;
  }
}

const routes: RouteRecordRaw[] = [
  // 公開ルート
  {
    path: '/',
    component: () => import('@/layouts/PublicLayout.vue'),
    children: [
      { path: '', name: 'home', component: () => import('@/pages/Home.vue') },
      {
        path: 'login',
        name: 'login',
        component: () => import('@/pages/Login.vue'),
        meta: { title: 'ログイン' },
      },
    ],
  },

  // 認証が必要なルート
  {
    path: '/app',
    component: () => import('@/layouts/AppLayout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: 'dashboard',
        name: 'dashboard',
        component: () => import('@/pages/Dashboard.vue'),
        meta: { title: 'ダッシュボード' },
      },
      {
        path: 'settings',
        name: 'settings',
        component: () => import('@/pages/Settings.vue'),
        meta: { title: '設定' },
      },
    ],
  },

  // 管理者ルート
  {
    path: '/admin',
    component: () => import('@/layouts/AdminLayout.vue'),
    meta: { requiresAuth: true, requiredRoles: ['admin'] },
    children: [
      {
        path: 'users',
        name: 'admin-users',
        component: () => import('@/pages/admin/Users.vue'),
        meta: { title: 'ユーザー管理', requiredPermissions: ['users:read'] },
      },
    ],
  },

  // 403 ページ
  {
    path: '/403',
    name: 'forbidden',
    component: () => import('@/pages/Forbidden.vue'),
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

// グローバルナビゲーションガード
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore();

  // ページタイトルの更新
  document.title = to.meta.title
    ? `${to.meta.title} | MyApp`
    : 'MyApp';

  // 認証が不要なルート
  if (!to.meta.requiresAuth) {
    // ログイン済みユーザーがログインページにアクセス
    if (to.name === 'login' && authStore.isAuthenticated) {
      return next({ name: 'dashboard' });
    }
    return next();
  }

  // 未認証の場合
  if (!authStore.isAuthenticated) {
    // セッション確認を試行
    await authStore.checkSession();

    if (!authStore.isAuthenticated) {
      return next({
        name: 'login',
        query: { redirect: to.fullPath },
      });
    }
  }

  // ロールチェック
  if (to.meta.requiredRoles?.length) {
    const hasRole = to.meta.requiredRoles.includes(authStore.user!.role);
    if (!hasRole) {
      return next({ name: 'forbidden' });
    }
  }

  // 権限チェック
  if (to.meta.requiredPermissions?.length) {
    const hasAllPermissions = to.meta.requiredPermissions.every(p =>
      authStore.hasPermission(p)
    );
    if (!hasAllPermissions) {
      return next({ name: 'forbidden' });
    }
  }

  next();
});

// ナビゲーション後のフック
router.afterEach((to, from) => {
  // アナリティクスの送信
  if (typeof window !== 'undefined' && window.gtag) {
    window.gtag('event', 'page_view', {
      page_path: to.fullPath,
    });
  }
});

export default router;
```

### 7.3 Angular Router のガード

```typescript
// ============================================
// Angular Router ガード
// ============================================

// auth.guard.ts — Functional Guard（Angular 15+）
import { inject } from '@angular/core';
import { Router, CanActivateFn, CanMatchFn } from '@angular/router';
import { AuthService } from './auth.service';
import { map, tap } from 'rxjs/operators';

// canActivate ガード
export const authGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  return authService.isAuthenticated$.pipe(
    tap(isAuthenticated => {
      if (!isAuthenticated) {
        router.navigate(['/login'], {
          queryParams: { returnUrl: state.url },
        });
      }
    })
  );
};

// ロールベースのガード
export const roleGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  const requiredRoles = route.data?.['roles'] as string[] | undefined;

  if (!requiredRoles?.length) return true;

  return authService.user$.pipe(
    map(user => {
      if (!user) return false;
      return requiredRoles.includes(user.role);
    }),
    tap(hasRole => {
      if (!hasRole) {
        router.navigate(['/403']);
      }
    })
  );
};

// 権限ベースのガード
export const permissionGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);
  const requiredPermissions = route.data?.['permissions'] as string[] | undefined;

  if (!requiredPermissions?.length) return true;

  return authService.user$.pipe(
    map(user => {
      if (!user) return false;
      return requiredPermissions.every(p => authService.hasPermission(user.role, p));
    }),
    tap(hasPermission => {
      if (!hasPermission) {
        router.navigate(['/403']);
      }
    })
  );
};

// ルート定義
// app.routes.ts
import { Routes } from '@angular/router';
import { authGuard, roleGuard, permissionGuard } from './auth/auth.guard';

export const routes: Routes = [
  { path: '', component: HomeComponent },
  { path: 'login', component: LoginComponent },
  {
    path: 'dashboard',
    component: DashboardComponent,
    canActivate: [authGuard],
  },
  {
    path: 'admin',
    canActivate: [authGuard, roleGuard],
    data: { roles: ['admin'] },
    children: [
      {
        path: 'users',
        component: UserManagementComponent,
        canActivate: [permissionGuard],
        data: { permissions: ['users:read'] },
      },
    ],
  },
];
```

---

## 8. 認証ガードの比較表

### 8.1 フレームワーク別比較

| 機能 | Next.js | React Router | Vue Router | Angular |
|------|---------|-------------|------------|---------|
| ガードの実装場所 | middleware.ts + Layout | ProtectedRoute コンポーネント | beforeEach フック | canActivate ガード |
| サーバーサイド対応 | Middleware + Server Component | SSR 時は別途対応 | SSR 時は別途対応 | SSR 時は別途対応 |
| ロールベース | カスタム実装 | カスタム実装 | ルートメタ + ガード | ルートデータ + ガード |
| リダイレクト | redirect() / NextResponse.redirect | Navigate コンポーネント | next({ name: 'login' }) | router.navigate() |
| 遅延ロード保護 | Dynamic import + 認証チェック | React.lazy + ProtectedRoute | 動的インポート + ガード | loadChildren + canMatch |
| TypeScript 対応 | ネイティブ | ネイティブ | RouteMeta 拡張 | ネイティブ |

### 8.2 認証方式の比較

| 方式 | セキュリティ | UX | 実装の複雑さ | 適用場面 |
|------|------------|-----|------------|---------|
| Session Cookie | 高 | 良好 | 低 | 一般的な Web アプリ |
| JWT (Cookie) | 高 | 良好 | 中 | API との統合 |
| JWT (localStorage) | 低（XSS 脆弱） | 良好 | 低 | 非推奨 |
| OAuth 2.0 | 高 | 良好 | 高 | ソーシャルログイン |
| Session + JWT ハイブリッド | 最高 | 良好 | 高 | エンタープライズ |
| パスキー / WebAuthn | 最高 | 優秀 | 高 | 次世代認証 |

### 8.3 トークン保存場所の比較

| 保存場所 | XSS 耐性 | CSRF 耐性 | サーバーサイドアクセス | 備考 |
|---------|----------|-----------|-------------------|------|
| HttpOnly Cookie | 高（JS からアクセス不可） | 要対策（SameSite） | 可 | 推奨 |
| localStorage | 低（XSS で盗取可能） | 高（自動送信なし） | 不可 | 非推奨 |
| sessionStorage | 低（XSS で盗取可能） | 高（自動送信なし） | 不可 | タブ限定 |
| メモリ（変数） | 中（リロードで消失） | 高 | 不可 | SPA 向け |
| IndexedDB | 低（XSS で盗取可能） | 高 | 不可 | 大容量データ用 |

---

## 9. 多要素認証（MFA）フローの統合

### 9.1 MFA の概要と認証ガードへの組み込み

多要素認証（Multi-Factor Authentication）は、「知識」「所持」「生体」の複数の認証要素を組み合わせることで、セキュリティを大幅に強化する仕組みである。認証ガードに MFA を統合するには、通常のログインフローに加えて、追加の認証ステップを設ける必要がある。

```
【MFA の認証要素】

1. 知識要素（Something you know）
   ├── パスワード
   ├── PIN コード
   └── セキュリティの質問

2. 所持要素（Something you have）
   ├── TOTP（Google Authenticator 等）
   ├── SMS / メール OTP
   ├── ハードウェアキー（YubiKey 等）
   └── プッシュ通知

3. 生体要素（Something you are）
   ├── 指紋認証
   ├── 顔認証
   └── 虹彩認証

【MFA ログインフロー】

  1. ユーザーが email + password でログイン
  2. パスワード検証成功
  3. MFA が有効かチェック
     ├── MFA 無効 → セッション作成 → ダッシュボードへ
     └── MFA 有効 → 一時トークン発行 → MFA 検証ページへ
  4. MFA コード入力
  5. MFA コード検証
     ├── 成功 → セッション作成（MFA 認証済みフラグ付き）
     └── 失敗 → エラー表示（リトライ回数制限あり）
```

### 9.2 TOTP ベースの MFA 実装

```typescript
// ============================================
// TOTP（Time-based One-Time Password）MFA 実装
// ============================================

// lib/auth/totp.ts
import { createHmac, randomBytes } from 'crypto';
import { encode as base32Encode, decode as base32Decode } from 'hi-base32';

// TOTP シークレット生成
export function generateTOTPSecret(): string {
  const buffer = randomBytes(20);
  return base32Encode(buffer).replace(/=/g, '');
}

// TOTP コード生成
export function generateTOTPCode(secret: string, timeStep = 30): string {
  const time = Math.floor(Date.now() / 1000 / timeStep);
  const timeBuffer = Buffer.alloc(8);
  timeBuffer.writeBigInt64BE(BigInt(time));

  const decodedSecret = Buffer.from(base32Decode(secret));
  const hmac = createHmac('sha1', decodedSecret);
  hmac.update(timeBuffer);
  const hash = hmac.digest();

  const offset = hash[hash.length - 1] & 0x0f;
  const code = (
    ((hash[offset] & 0x7f) << 24) |
    ((hash[offset + 1] & 0xff) << 16) |
    ((hash[offset + 2] & 0xff) << 8) |
    (hash[offset + 3] & 0xff)
  ) % 1000000;

  return code.toString().padStart(6, '0');
}

// TOTP コード検証（前後のタイムウィンドウも許容）
export function verifyTOTPCode(
  secret: string,
  code: string,
  window = 1
): boolean {
  for (let i = -window; i <= window; i++) {
    const timeStep = 30;
    const time = Math.floor(Date.now() / 1000 / timeStep) + i;
    const timeBuffer = Buffer.alloc(8);
    timeBuffer.writeBigInt64BE(BigInt(time));

    const decodedSecret = Buffer.from(base32Decode(secret));
    const hmac = createHmac('sha1', decodedSecret);
    hmac.update(timeBuffer);
    const hash = hmac.digest();

    const offset = hash[hash.length - 1] & 0x0f;
    const expectedCode = (
      ((hash[offset] & 0x7f) << 24) |
      ((hash[offset + 1] & 0xff) << 16) |
      ((hash[offset + 2] & 0xff) << 8) |
      (hash[offset + 3] & 0xff)
    ) % 1000000;

    if (expectedCode.toString().padStart(6, '0') === code) {
      return true;
    }
  }
  return false;
}

// QR コード用 URI 生成
export function generateTOTPUri(
  secret: string,
  email: string,
  issuer: string
): string {
  return `otpauth://totp/${encodeURIComponent(issuer)}:${encodeURIComponent(email)}?secret=${secret}&issuer=${encodeURIComponent(issuer)}&algorithm=SHA1&digits=6&period=30`;
}

// ============================================
// MFA セットアップ API
// ============================================

// app/api/auth/mfa/setup/route.ts
export async function POST(request: NextRequest) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const secret = generateTOTPSecret();
  const uri = generateTOTPUri(secret, session.user.email, 'MyApp');

  // 一時的にシークレットを保存（まだ有効化しない）
  await prisma.user.update({
    where: { id: session.user.id },
    data: { pendingMFASecret: secret },
  });

  return NextResponse.json({ secret, uri });
}

// app/api/auth/mfa/verify/route.ts
export async function POST(request: NextRequest) {
  const session = await getSession();
  if (!session) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const { code } = await request.json();
  const user = await prisma.user.findUnique({
    where: { id: session.user.id },
  });

  if (!user?.pendingMFASecret) {
    return NextResponse.json(
      { error: 'MFA setup not initiated' },
      { status: 400 }
    );
  }

  // コード検証
  const isValid = verifyTOTPCode(user.pendingMFASecret, code);
  if (!isValid) {
    return NextResponse.json(
      { error: 'Invalid verification code' },
      { status: 400 }
    );
  }

  // バックアップコード生成
  const backupCodes = Array.from({ length: 10 }, () =>
    randomBytes(4).toString('hex')
  );

  // MFA を有効化
  await prisma.user.update({
    where: { id: session.user.id },
    data: {
      mfaSecret: user.pendingMFASecret,
      pendingMFASecret: null,
      mfaEnabled: true,
      mfaBackupCodes: backupCodes.map(code =>
        bcrypt.hashSync(code, 10)
      ),
    },
  });

  return NextResponse.json({
    success: true,
    backupCodes, // これは一度だけ表示する
  });
}
```

### 9.3 MFA 認証ガードの実装

```typescript
// ============================================
// MFA 対応の認証ガード
// ============================================

// middleware.ts での MFA チェック
export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get('session-token')?.value;
  const mfaToken = request.cookies.get('mfa-verified')?.value;

  // 認証チェック（通常のフロー）
  if (!token && isProtectedRoute(pathname)) {
    return redirectToLogin(request, pathname);
  }

  if (token) {
    const payload = await verifyToken(token);
    if (!payload) {
      return redirectToLogin(request, pathname);
    }

    // MFA チェック
    // MFA が有効なユーザーで、MFA 検証が完了していない場合
    if (payload.mfaEnabled && !mfaToken && pathname !== '/auth/mfa') {
      // MFA 検証ページ以外へのアクセスは MFA ページへリダイレクト
      return NextResponse.redirect(new URL('/auth/mfa', request.url));
    }
  }

  return NextResponse.next();
}

// app/auth/mfa/page.tsx — MFA 検証ページ
'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function MFAVerificationPage() {
  const [code, setCode] = useState(['', '', '', '', '', '']);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);
  const router = useRouter();

  // 最初の入力にフォーカス
  useEffect(() => {
    inputRefs.current[0]?.focus();
  }, []);

  const handleChange = (index: number, value: string) => {
    if (!/^\d*$/.test(value)) return; // 数字のみ

    const newCode = [...code];
    newCode[index] = value;
    setCode(newCode);

    // 次の入力に自動フォーカス
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }

    // 6桁入力完了で自動送信
    if (newCode.every(c => c !== '') && index === 5) {
      handleSubmit(newCode.join(''));
    }
  };

  const handleKeyDown = (index: number, e: React.KeyboardEvent) => {
    // Backspace で前の入力に戻る
    if (e.key === 'Backspace' && !code[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pastedData = e.clipboardData.getData('text').slice(0, 6);
    if (!/^\d+$/.test(pastedData)) return;

    const newCode = [...code];
    for (let i = 0; i < pastedData.length; i++) {
      newCode[i] = pastedData[i];
    }
    setCode(newCode);

    if (pastedData.length === 6) {
      handleSubmit(pastedData);
    }
  };

  const handleSubmit = async (verificationCode: string) => {
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('/api/auth/mfa/challenge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: verificationCode }),
      });

      if (response.ok) {
        router.push('/dashboard');
      } else {
        const data = await response.json();
        setError(data.error || '認証コードが正しくありません');
        setCode(['', '', '', '', '', '']);
        inputRefs.current[0]?.focus();
      }
    } catch {
      setError('認証に失敗しました。もう一度お試しください。');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-6 p-8">
        <div className="text-center">
          <h2 className="text-2xl font-bold">二段階認証</h2>
          <p className="mt-2 text-gray-600">
            認証アプリに表示されている6桁のコードを入力してください
          </p>
        </div>

        <div className="flex justify-center gap-2">
          {code.map((digit, index) => (
            <input
              key={index}
              ref={el => { inputRefs.current[index] = el; }}
              type="text"
              inputMode="numeric"
              maxLength={1}
              value={digit}
              onChange={e => handleChange(index, e.target.value)}
              onKeyDown={e => handleKeyDown(index, e)}
              onPaste={index === 0 ? handlePaste : undefined}
              disabled={isLoading}
              className="w-12 h-14 text-center text-2xl font-mono
                         border-2 border-gray-300 rounded-lg
                         focus:border-blue-500 focus:ring-2 focus:ring-blue-200
                         disabled:opacity-50"
            />
          ))}
        </div>

        {error && (
          <p className="text-center text-sm text-red-600">{error}</p>
        )}

        <div className="text-center">
          <button
            type="button"
            className="text-sm text-blue-600 hover:text-blue-500"
            onClick={() => {/* バックアップコード入力モードに切り替え */}}
          >
            バックアップコードを使用
          </button>
        </div>
      </div>
    </div>
  );
}
```

### 9.4 段階的認証（Step-up Authentication）

高セキュリティ操作時に追加認証を要求するパターンである。

```typescript
// ============================================
// 段階的認証の実装
// ============================================

// lib/auth/stepUpAuth.ts

// 認証レベルの定義
enum AuthLevel {
  BASIC = 1,       // パスワードのみ
  MFA = 2,         // パスワード + MFA
  RECENT_MFA = 3,  // 直近の MFA（5分以内）
  BIOMETRIC = 4,   // 生体認証
}

// 操作ごとの必要認証レベル
const operationAuthLevels: Record<string, AuthLevel> = {
  'view:dashboard': AuthLevel.BASIC,
  'edit:profile': AuthLevel.BASIC,
  'change:password': AuthLevel.MFA,
  'change:email': AuthLevel.MFA,
  'enable:mfa': AuthLevel.MFA,
  'view:billing': AuthLevel.MFA,
  'update:billing': AuthLevel.RECENT_MFA,
  'delete:account': AuthLevel.RECENT_MFA,
  'transfer:funds': AuthLevel.RECENT_MFA,
  'export:all-data': AuthLevel.RECENT_MFA,
};

// 現在の認証レベルを取得
function getCurrentAuthLevel(session: Session): AuthLevel {
  if (!session) return 0;

  const now = Date.now();
  const lastMfaAt = session.user.lastMfaVerifiedAt
    ? new Date(session.user.lastMfaVerifiedAt).getTime()
    : 0;
  const fiveMinutesAgo = now - 5 * 60 * 1000;

  if (lastMfaAt > fiveMinutesAgo) {
    return AuthLevel.RECENT_MFA;
  }

  if (session.mfaVerified) {
    return AuthLevel.MFA;
  }

  return AuthLevel.BASIC;
}

// 認証レベルチェック
export function requireAuthLevel(
  session: Session,
  operation: string
): { allowed: boolean; requiredLevel: AuthLevel; currentLevel: AuthLevel } {
  const requiredLevel = operationAuthLevels[operation] || AuthLevel.BASIC;
  const currentLevel = getCurrentAuthLevel(session);

  return {
    allowed: currentLevel >= requiredLevel,
    requiredLevel,
    currentLevel,
  };
}

// React Hook
export function useStepUpAuth() {
  const { session } = useSession();
  const [isReauthModalOpen, setIsReauthModalOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState<(() => void) | null>(null);

  const requireAuth = useCallback(
    (operation: string, action: () => void) => {
      const result = requireAuthLevel(session!, operation);

      if (result.allowed) {
        action();
      } else {
        setPendingAction(() => action);
        setIsReauthModalOpen(true);
      }
    },
    [session]
  );

  const onReauthSuccess = useCallback(() => {
    setIsReauthModalOpen(false);
    pendingAction?.();
    setPendingAction(null);
  }, [pendingAction]);

  return { requireAuth, isReauthModalOpen, setIsReauthModalOpen, onReauthSuccess };
}

// 使用例
function BillingPage() {
  const { requireAuth, isReauthModalOpen, setIsReauthModalOpen, onReauthSuccess } = useStepUpAuth();

  const handleUpdateBilling = () => {
    requireAuth('update:billing', () => {
      // 課金情報更新の処理
      updateBillingInfo();
    });
  };

  return (
    <div>
      <h1>課金設定</h1>
      <Button onClick={handleUpdateBilling}>
        課金情報を更新
      </Button>

      <ReauthModal
        isOpen={isReauthModalOpen}
        onClose={() => setIsReauthModalOpen(false)}
        onSuccess={onReauthSuccess}
      />
    </div>
  );
}
```

---

## 10. OAuth / OpenID Connect 連携の認証ガード

### 10.1 OAuth 2.0 フロー概要

```
【OAuth 2.0 Authorization Code Flow】

  1. ユーザーが「Google でログイン」をクリック
  2. アプリが Google の認可エンドポイントにリダイレクト
     URL: https://accounts.google.com/o/oauth2/v2/auth
     パラメータ:
       - client_id: アプリのクライアント ID
       - redirect_uri: コールバック URL
       - response_type: code
       - scope: openid email profile
       - state: CSRF 対策のランダム文字列
       - code_challenge: PKCE のチャレンジ
  3. ユーザーが Google で認証・承認
  4. Google がコールバック URL にリダイレクト（認可コード付き）
  5. アプリがバックエンドで認可コードをトークンに交換
     POST https://oauth2.googleapis.com/token
  6. アクセストークン + ID トークンを取得
  7. ID トークンからユーザー情報を取得
  8. セッション作成、ダッシュボードへリダイレクト

【セキュリティ上の重要ポイント】
  - state パラメータで CSRF 攻撃を防止
  - PKCE（Proof Key for Code Exchange）で認可コード傍受を防止
  - トークン交換はバックエンドで実施（クライアントシークレットを保護）
  - nonce パラメータでリプレイ攻撃を防止
```

### 10.2 OAuth コールバックハンドラー

```typescript
// ============================================
// OAuth コールバック処理
// ============================================

// app/api/auth/callback/[provider]/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';

const providers: Record<string, OAuthProvider> = {
  google: {
    tokenUrl: 'https://oauth2.googleapis.com/token',
    userInfoUrl: 'https://www.googleapis.com/oauth2/v3/userinfo',
    clientId: process.env.GOOGLE_CLIENT_ID!,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
  },
  github: {
    tokenUrl: 'https://github.com/login/oauth/access_token',
    userInfoUrl: 'https://api.github.com/user',
    clientId: process.env.GITHUB_CLIENT_ID!,
    clientSecret: process.env.GITHUB_CLIENT_SECRET!,
  },
};

export async function GET(
  request: NextRequest,
  { params }: { params: { provider: string } }
) {
  const provider = providers[params.provider];
  if (!provider) {
    return NextResponse.json({ error: 'Unknown provider' }, { status: 400 });
  }

  const { searchParams } = new URL(request.url);
  const code = searchParams.get('code');
  const state = searchParams.get('state');
  const error = searchParams.get('error');

  // エラーチェック
  if (error) {
    return NextResponse.redirect(
      new URL(`/login?error=${encodeURIComponent(error)}`, request.url)
    );
  }

  if (!code || !state) {
    return NextResponse.redirect(
      new URL('/login?error=missing_params', request.url)
    );
  }

  // State 検証（CSRF 対策）
  const cookieStore = await cookies();
  const storedState = cookieStore.get('oauth-state')?.value;
  if (state !== storedState) {
    return NextResponse.redirect(
      new URL('/login?error=invalid_state', request.url)
    );
  }
  cookieStore.delete('oauth-state');

  // PKCE verifier 取得
  const codeVerifier = cookieStore.get('oauth-code-verifier')?.value;
  cookieStore.delete('oauth-code-verifier');

  try {
    // トークン交換
    const tokenResponse = await fetch(provider.tokenUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'authorization_code',
        code,
        redirect_uri: `${process.env.NEXT_PUBLIC_APP_URL}/api/auth/callback/${params.provider}`,
        client_id: provider.clientId,
        client_secret: provider.clientSecret,
        ...(codeVerifier ? { code_verifier: codeVerifier } : {}),
      }),
    });

    const tokens = await tokenResponse.json();

    if (!tokens.access_token) {
      throw new Error('Token exchange failed');
    }

    // ユーザー情報取得
    const userInfoResponse = await fetch(provider.userInfoUrl, {
      headers: { Authorization: `Bearer ${tokens.access_token}` },
    });
    const userInfo = await userInfoResponse.json();

    // ユーザーの作成 or 更新
    const user = await prisma.user.upsert({
      where: {
        email: userInfo.email,
      },
      update: {
        name: userInfo.name,
        image: userInfo.picture || userInfo.avatar_url,
        lastLoginAt: new Date(),
      },
      create: {
        email: userInfo.email,
        name: userInfo.name,
        image: userInfo.picture || userInfo.avatar_url,
        role: 'user',
        emailVerified: new Date(),
        accounts: {
          create: {
            provider: params.provider,
            providerAccountId: String(userInfo.sub || userInfo.id),
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token,
            expiresAt: tokens.expires_in
              ? Math.floor(Date.now() / 1000) + tokens.expires_in
              : undefined,
          },
        },
      },
    });

    // セッション作成
    const sessionToken = await createSessionToken(user);
    cookieStore.set('session-token', sessionToken, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 7 * 24 * 60 * 60,
      path: '/',
    });

    // callbackUrl にリダイレクト
    const callbackUrl = cookieStore.get('oauth-callback-url')?.value || '/dashboard';
    cookieStore.delete('oauth-callback-url');

    return NextResponse.redirect(new URL(callbackUrl, request.url));
  } catch (error) {
    console.error('OAuth callback error:', error);
    return NextResponse.redirect(
      new URL('/login?error=oauth_failed', request.url)
    );
  }
}
```

---

## 11. 認証ガードのテスト

### 11.1 ユニットテスト

```typescript
// ============================================
// 認証ガードのユニットテスト
// ============================================

// __tests__/auth/permissions.test.ts
import { describe, it, expect } from 'vitest';
import {
  hasPermission,
  hasAllPermissions,
  hasAnyPermission,
  getAllPermissions,
} from '@/lib/auth/permissions';

describe('RBAC Permission System', () => {
  describe('hasPermission', () => {
    it('viewer は投稿を閲覧できる', () => {
      expect(hasPermission('viewer', 'posts:read')).toBe(true);
    });

    it('viewer はユーザーを削除できない', () => {
      expect(hasPermission('viewer', 'users:delete')).toBe(false);
    });

    it('admin は editor の権限を継承する', () => {
      expect(hasPermission('admin', 'posts:publish')).toBe(true);
    });

    it('editor は admin の権限を持たない', () => {
      expect(hasPermission('editor', 'users:delete')).toBe(false);
    });

    it('superadmin は全権限を持つ', () => {
      expect(hasPermission('superadmin', 'billing:update')).toBe(true);
      expect(hasPermission('superadmin', 'users:delete')).toBe(true);
      expect(hasPermission('superadmin', 'posts:publish')).toBe(true);
    });
  });

  describe('hasAllPermissions', () => {
    it('admin は users:create と users:delete の両方を持つ', () => {
      expect(
        hasAllPermissions('admin', ['users:create', 'users:delete'])
      ).toBe(true);
    });

    it('editor は users:create を持たない', () => {
      expect(
        hasAllPermissions('editor', ['posts:publish', 'users:create'])
      ).toBe(false);
    });
  });

  describe('hasAnyPermission', () => {
    it('editor は posts:publish または users:create のいずれかを持つ', () => {
      expect(
        hasAnyPermission('editor', ['posts:publish', 'users:create'])
      ).toBe(true);
    });

    it('viewer は posts:create も users:create も持たない', () => {
      expect(
        hasAnyPermission('viewer', ['posts:create', 'users:create'])
      ).toBe(false);
    });
  });

  describe('getAllPermissions', () => {
    it('ロール階層に基づいて全権限を取得する', () => {
      const adminPerms = getAllPermissions('admin');
      // admin 直接の権限
      expect(adminPerms).toContain('users:create');
      expect(adminPerms).toContain('users:delete');
      // editor から継承
      expect(adminPerms).toContain('posts:publish');
      // viewer から継承
      expect(adminPerms).toContain('posts:read');
      // superadmin の権限は含まない
      expect(adminPerms).not.toContain('billing:update');
    });

    it('重複なく権限を返す', () => {
      const perms = getAllPermissions('admin');
      const unique = [...new Set(perms)];
      expect(perms.length).toBe(unique.length);
    });
  });
});

// __tests__/auth/session.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { getSession, refreshSession } from '@/lib/auth/session';

// モック設定
vi.mock('next/headers', () => ({
  cookies: vi.fn(() => ({
    get: vi.fn(),
    set: vi.fn(),
    delete: vi.fn(),
  })),
}));

vi.mock('jose', () => ({
  jwtVerify: vi.fn(),
  SignJWT: vi.fn().mockImplementation(() => ({
    setProtectedHeader: vi.fn().mockReturnThis(),
    setIssuedAt: vi.fn().mockReturnThis(),
    setExpirationTime: vi.fn().mockReturnThis(),
    sign: vi.fn().mockResolvedValue('new-token'),
  })),
}));

describe('Session Management', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('有効なトークンがない場合は null を返す', async () => {
    const { cookies } = await import('next/headers');
    (cookies as any).mockReturnValue({
      get: () => null,
    });

    const session = await getSession();
    expect(session).toBeNull();
  });

  it('無効なトークンの場合は null を返す', async () => {
    const { cookies } = await import('next/headers');
    (cookies as any).mockReturnValue({
      get: () => ({ value: 'invalid-token' }),
    });

    const { jwtVerify } = await import('jose');
    (jwtVerify as any).mockRejectedValue(new Error('Invalid token'));

    const session = await getSession();
    expect(session).toBeNull();
  });
});
```

### 11.2 統合テスト（E2E）

```typescript
// ============================================
// Playwright E2E テスト
// ============================================

// e2e/auth/login.spec.ts
import { test, expect } from '@playwright/test';

test.describe('認証フロー', () => {
  test.beforeEach(async ({ page }) => {
    // テストユーザーのシードデータを投入
    await page.request.post('/api/test/seed', {
      data: {
        users: [
          { email: 'user@test.com', password: 'password123', role: 'user' },
          { email: 'admin@test.com', password: 'admin123', role: 'admin' },
        ],
      },
    });
  });

  test('未認証ユーザーはダッシュボードにアクセスできない', async ({ page }) => {
    await page.goto('/dashboard');
    // ログインページにリダイレクトされる
    await expect(page).toHaveURL(/\/login/);
    // callbackUrl が設定されている
    expect(page.url()).toContain('callbackUrl=%2Fdashboard');
  });

  test('正しい認証情報でログインできる', async ({ page }) => {
    await page.goto('/login');

    await page.fill('input[name="email"]', 'user@test.com');
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');

    // ダッシュボードにリダイレクトされる
    await expect(page).toHaveURL('/dashboard');
    // ユーザー名が表示される
    await expect(page.locator('[data-testid="user-name"]')).toBeVisible();
  });

  test('間違った認証情報でエラーが表示される', async ({ page }) => {
    await page.goto('/login');

    await page.fill('input[name="email"]', 'user@test.com');
    await page.fill('input[name="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');

    // エラーメッセージが表示される
    await expect(
      page.locator('text=メールアドレスまたはパスワードが正しくありません')
    ).toBeVisible();
    // ログインページに留まる
    await expect(page).toHaveURL(/\/login/);
  });

  test('一般ユーザーは管理者ページにアクセスできない', async ({ page }) => {
    // ログイン
    await page.goto('/login');
    await page.fill('input[name="email"]', 'user@test.com');
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/dashboard');

    // 管理者ページにアクセス
    await page.goto('/admin');
    // 403 ページまたはダッシュボードにリダイレクト
    await expect(page).toHaveURL(/\/(403|dashboard)/);
  });

  test('管理者は管理者ページにアクセスできる', async ({ page }) => {
    // 管理者でログイン
    await page.goto('/login');
    await page.fill('input[name="email"]', 'admin@test.com');
    await page.fill('input[name="password"]', 'admin123');
    await page.click('button[type="submit"]');

    // 管理者ページにアクセス
    await page.goto('/admin');
    await expect(page).toHaveURL('/admin');
    await expect(page.locator('text=管理者モード')).toBeVisible();
  });

  test('ログアウト後はダッシュボードにアクセスできない', async ({ page }) => {
    // ログイン
    await page.goto('/login');
    await page.fill('input[name="email"]', 'user@test.com');
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/dashboard');

    // ログアウト
    await page.click('[data-testid="logout-button"]');
    await expect(page).toHaveURL('/login');

    // ダッシュボードにアクセス
    await page.goto('/dashboard');
    await expect(page).toHaveURL(/\/login/);
  });

  test('callbackUrl が正しく機能する', async ({ page }) => {
    // 設定ページにアクセス（未認証）
    await page.goto('/settings/profile');
    await expect(page).toHaveURL(/\/login.*callbackUrl/);

    // ログイン
    await page.fill('input[name="email"]', 'user@test.com');
    await page.fill('input[name="password"]', 'password123');
    await page.click('button[type="submit"]');

    // 設定ページにリダイレクトされる
    await expect(page).toHaveURL('/settings/profile');
  });
});
```

### 11.3 Middleware のテスト

```typescript
// ============================================
// Next.js Middleware のテスト
// ============================================

// __tests__/middleware.test.ts
import { describe, it, expect, vi } from 'vitest';
import { NextRequest } from 'next/server';
import { middleware } from '@/middleware';

// NextRequest のモック作成ヘルパー
function createMockRequest(
  url: string,
  options: {
    cookies?: Record<string, string>;
    headers?: Record<string, string>;
  } = {}
): NextRequest {
  const request = new NextRequest(new URL(url, 'http://localhost:3000'));

  if (options.cookies) {
    Object.entries(options.cookies).forEach(([name, value]) => {
      request.cookies.set(name, value);
    });
  }

  if (options.headers) {
    Object.entries(options.headers).forEach(([name, value]) => {
      request.headers.set(name, value);
    });
  }

  return request;
}

describe('Middleware', () => {
  it('公開パスはそのまま通す', async () => {
    const request = createMockRequest('/');
    const response = await middleware(request);
    expect(response.status).not.toBe(302);
    expect(response.status).not.toBe(401);
  });

  it('認証なしで保護されたパスにアクセスするとリダイレクト', async () => {
    const request = createMockRequest('/dashboard');
    const response = await middleware(request);
    expect(response.status).toBe(307); // Temporary Redirect
    expect(response.headers.get('location')).toContain('/login');
  });

  it('認証なしで API にアクセスすると 401 を返す', async () => {
    const request = createMockRequest('/api/users');
    const response = await middleware(request);
    expect(response.status).toBe(401);
  });

  it('有効なトークンがあればそのまま通す', async () => {
    const request = createMockRequest('/dashboard', {
      cookies: { 'session-token': 'valid-jwt-token' },
    });

    // JWT検証のモック
    vi.mocked(jwtVerify).mockResolvedValueOnce({
      payload: { sub: 'user-1', role: 'user', exp: Date.now() / 1000 + 3600 },
      protectedHeader: { alg: 'HS256' },
    } as any);

    const response = await middleware(request);
    expect(response.status).toBe(200);
  });

  it('管理者でないユーザーが /admin にアクセスすると 403', async () => {
    const request = createMockRequest('/admin', {
      cookies: { 'session-token': 'valid-jwt-token' },
    });

    vi.mocked(jwtVerify).mockResolvedValueOnce({
      payload: { sub: 'user-1', role: 'user', exp: Date.now() / 1000 + 3600 },
      protectedHeader: { alg: 'HS256' },
    } as any);

    const response = await middleware(request);
    expect(response.status).toBe(307);
    expect(response.headers.get('location')).toContain('/403');
  });
});
```

---

## 12. セキュリティのベストプラクティス

### 12.1 推奨事項

```
【認証ガードのベストプラクティス】

1. 多層防御を実装する
   - Middleware + Layout + API Route の3層でチェック
   - クライアントとサーバーの両方で検証
   - データベースレベルでの RLS も検討

2. トークン管理
   - HttpOnly + Secure + SameSite=Lax な Cookie を使用
   - トークンの有効期限を適切に設定（長すぎない）
   - リフレッシュトークンの回転（Rotation）を実装
   - ログアウト時にサーバーサイドでトークンを無効化

3. パスワードセキュリティ
   - bcrypt / argon2 でハッシュ化（ストレッチング 12 ラウンド以上）
   - タイミング攻撃対策（存在しないユーザーでも同じ処理時間）
   - アカウントロック機構（5回失敗で15分ロック）
   - パスワード強度チェック（zxcvbn 等）

4. セッション管理
   - セッションの有効期限を設定
   - スライディングウィンドウでセッション更新
   - 複数デバイスのセッション管理
   - 不審なログイン検知（異なる IP / デバイス）

5. CSRF 対策
   - SameSite Cookie 属性を設定
   - CSRF トークンを使用（フォーム送信時）
   - Origin / Referer ヘッダーの検証

6. 入力検証
   - サーバーサイドでのバリデーション（Zod 等）
   - SQL インジェクション対策（ORM / パラメータバインディング）
   - XSS 対策（入力のサニタイズ）

7. 監査ログ
   - ログイン / ログアウトの記録
   - 権限変更の記録
   - 重要な操作の記録
   - 不正アクセス試行の記録
```

### 12.2 Cookie セキュリティ設定

```typescript
// ============================================
// セキュアな Cookie 設定のガイドライン
// ============================================

// lib/auth/cookies.ts

interface CookieOptions {
  httpOnly: boolean;
  secure: boolean;
  sameSite: 'strict' | 'lax' | 'none';
  maxAge: number;
  path: string;
  domain?: string;
}

// セッション Cookie の推奨設定
export const SESSION_COOKIE_OPTIONS: CookieOptions = {
  httpOnly: true,     // JavaScript からアクセス不可（XSS 対策）
  secure: process.env.NODE_ENV === 'production', // HTTPS のみ
  sameSite: 'lax',    // CSRF 対策（GET リクエストのクロスサイトは許可）
  maxAge: 7 * 24 * 60 * 60, // 7日
  path: '/',
  // domain: '.example.com', // サブドメイン共有が必要な場合
};

// CSRF トークン Cookie の設定
export const CSRF_COOKIE_OPTIONS: CookieOptions = {
  httpOnly: false,    // JavaScript から読み取り可能（フォームに含めるため）
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'strict', // 厳格な CSRF 対策
  maxAge: 60 * 60,    // 1時間
  path: '/',
};

// OAuth State Cookie の設定
export const OAUTH_STATE_COOKIE_OPTIONS: CookieOptions = {
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax',    // OAuth リダイレクトで必要
  maxAge: 10 * 60,    // 10分（OAuth フロー中のみ有効）
  path: '/',
};

// ============================================
// SameSite 属性の選択ガイド
// ============================================
//
// 'strict': 最も厳格。外部サイトからのリクエストに Cookie を送信しない。
//           OAuth リダイレクトでも Cookie が送信されないため注意。
//
// 'lax':    GET リクエストのクロスサイトナビゲーションでは Cookie を送信。
//           POST リクエストのクロスサイトでは送信しない。
//           一般的な Web アプリに推奨。
//
// 'none':   クロスサイトで常に Cookie を送信。Secure 属性が必須。
//           サードパーティ Cookie が必要な場合のみ使用。
```

---

## 13. アンチパターンと対策

### 13.1 よくある間違い

```typescript
// ============================================
// アンチパターン集
// ============================================

// NG アンチパターン1: クライアントサイドのみの保護
// クライアントサイドの条件分岐は簡単にバイパスできる
function BadProtectedPage() {
  const { user } = useAuth();
  if (!user) return <LoginPage />;  // NG: DevTools で回避可能
  return <SecretContent />;
}

// OK 正しいパターン: サーバーサイドで保護
export default async function ProtectedPage() {
  const session = await getSession();
  if (!session) redirect('/login'); // OK: サーバーで制御
  return <SecretContent />;
}

// ─────────────────────────────────────────

// NG アンチパターン2: トークンを localStorage に保存
localStorage.setItem('token', jwt); // NG: XSS で盗取可能
fetch('/api/data', {
  headers: { Authorization: `Bearer ${localStorage.getItem('token')}` }
});

// OK 正しいパターン: HttpOnly Cookie を使用
// サーバーで Cookie を設定（JavaScript からアクセス不可）
cookieStore.set('session-token', jwt, {
  httpOnly: true,  // OK: XSS から保護
  secure: true,
  sameSite: 'lax',
});

// ─────────────────────────────────────────

// NG アンチパターン3: フロントエンドで権限チェックのみ
function DeleteButton({ userId }: { userId: string }) {
  const { user } = useAuth();
  // NG: フロントだけのチェックは不十分
  if (user.role !== 'admin') return null;
  return <button onClick={() => deleteUser(userId)}>削除</button>;
}

// OK 正しいパターン: API でも権限チェック
// フロントエンド: UI の制御
function DeleteButton({ userId }: { userId: string }) {
  const { user } = useAuth();
  if (user.role !== 'admin') return null; // UI 制御
  return <button onClick={() => deleteUser(userId)}>削除</button>;
}

// バックエンド: 実際の権限チェック OK
export async function DELETE(request: NextRequest) {
  const session = await getSession();
  if (!session) return unauthorized();
  if (!hasPermission(session.user.role, 'users:delete')) {
    return forbidden(); // OK: サーバーでも必ずチェック
  }
  await prisma.user.delete({ where: { id: userId } });
  return NextResponse.json({ success: true });
}

// ─────────────────────────────────────────

// NG アンチパターン4: ユーザー入力をそのまま callbackUrl に使用
const callbackUrl = searchParams.get('callbackUrl');
redirect(callbackUrl); // NG: オープンリダイレクト脆弱性

// OK 正しいパターン: callbackUrl を検証
function sanitizeCallbackUrl(url: string | null): string {
  if (!url) return '/dashboard';

  // 相対パスのみ許可
  if (!url.startsWith('/')) return '/dashboard';

  // プロトコルを含むURLは拒否
  if (url.includes('://')) return '/dashboard';

  // ダブルスラッシュを拒否（//evil.com 対策）
  if (url.startsWith('//')) return '/dashboard';

  // 許可されたパスのプレフィックスチェック
  const allowedPrefixes = ['/dashboard', '/settings', '/admin', '/app'];
  if (!allowedPrefixes.some(prefix => url.startsWith(prefix))) {
    return '/dashboard';
  }

  return url;
}

// ─────────────────────────────────────────

// NG アンチパターン5: エラーメッセージでの情報漏洩
if (!user) return { error: 'ユーザーが見つかりません' };
// NG: メールアドレスの存在を確認できてしまう

if (!isValid) return { error: 'パスワードが間違っています' };
// NG: アカウントの存在を確認できてしまう

// OK 正しいパターン: 曖昧なエラーメッセージ
return { error: 'メールアドレスまたはパスワードが正しくありません' };
// OK: どちらが間違いか特定できない

// ─────────────────────────────────────────

// NG アンチパターン6: JWT シークレットのハードコーディング
const secret = 'my-super-secret-key-12345'; // NG: ソースコードに秘密情報

// OK 正しいパターン: 環境変数を使用
const secret = process.env.JWT_SECRET!;
// さらに、シークレットが設定されていない場合は起動時にエラー
if (!process.env.JWT_SECRET) {
  throw new Error('JWT_SECRET environment variable is required');
}

// ─────────────────────────────────────────

// NG アンチパターン7: トークンの有効期限が長すぎる
.setExpirationTime('365d') // NG: 1年は長すぎる

// OK 正しいパターン: 適切な有効期限
// アクセストークン: 15分〜1時間
.setExpirationTime('1h')
// リフレッシュトークン: 7日〜30日
// セッション Cookie: 7日（スライディングウィンドウで延長）
```

### 13.2 セキュリティチェックリスト

```
【認証ガード セキュリティチェックリスト】

□ 認証
  □ パスワードは bcrypt / argon2 でハッシュ化しているか
  □ タイミング攻撃対策を実装しているか
  □ アカウントロック機構があるか
  □ パスワードリセットフローは安全か
  □ MFA オプションを提供しているか

□ セッション管理
  □ セッショントークンは HttpOnly Cookie に保存しているか
  □ Secure フラグを設定しているか
  □ SameSite 属性を設定しているか
  □ セッションの有効期限は適切か
  □ ログアウト時にサーバーサイドでセッションを無効化しているか

□ 認可
  □ サーバーサイドで権限チェックを行っているか
  □ API エンドポイントすべてに認可チェックがあるか
  □ IDOR（Insecure Direct Object Reference）対策をしているか
  □ 水平権限昇格の対策をしているか
  □ 垂直権限昇格の対策をしているか

□ 入力検証
  □ callbackUrl / redirectUrl の検証を行っているか
  □ ユーザー入力のサニタイズを行っているか
  □ SQL インジェクション対策をしているか

□ 通信
  □ HTTPS を強制しているか
  □ HSTS ヘッダーを設定しているか
  □ CSP（Content Security Policy）を設定しているか

□ 監査
  □ ログイン / ログアウトの監査ログがあるか
  □ 権限変更の監査ログがあるか
  □ 不正アクセス試行の検知と通知があるか
```

---

## 14. トラブルシューティング

### 14.1 よくある問題と解決策

```
【問題1: 無限リダイレクトループ】

症状: /login → /dashboard → /login → ... のループが発生

原因:
  - Middleware がログインページ自体も保護している
  - セッション Cookie の設定が不正（path や domain の問題）
  - リダイレクト先のルートも保護されている

解決策:
  - publicPaths にログインページを含める
  - Cookie の path を '/' に設定
  - matcher 設定でログインページを除外

  // middleware.ts
  export const config = {
    matcher: ['/((?!_next/static|_next/image|favicon.ico|login|register).*)'],
  };

---

【問題2: Server Component でのリダイレクトが動作しない】

症状: redirect() を呼んでもリダイレクトされない

原因:
  - redirect() を try-catch で捕捉している
  - redirect() の後にレンダリングが続いている
  - Client Component で Server 用の redirect を使っている

解決策:
  - redirect() は例外をスローするため catch しない

  // NG
  try {
    const session = await getSession();
    if (!session) redirect('/login');
  } catch (error) {
    // redirect の例外もここで捕捉されてしまう
  }

  // OK
  const session = await getSession();
  if (!session) redirect('/login');
  // redirect 後のコードは実行されない

---

【問題3: Middleware で DB アクセスができない】

症状: Middleware から Prisma を呼ぶとエラーが発生

原因:
  Middleware は Edge Runtime で動作し、Node.js API が制限される
  Prisma クライアントは Node.js ランタイムを前提としている

解決策:
  - Middleware では JWT の署名検証のみ行い、DB アクセスは避ける
  - DB が必要な完全な検証は Layout / Server Component で行う
  - Edge 対応の DB クライアントを使う（例: @prisma/client/edge）

  // Middleware: JWT 署名検証のみ（軽量）
  export function middleware(request: NextRequest) {
    const token = request.cookies.get('session-token')?.value;
    if (!token) return redirectToLogin(request);

    // JWT の署名のみ検証（DB アクセスなし）
    try {
      jwtVerify(token, secret); // jose は Edge Runtime 対応
    } catch {
      return redirectToLogin(request);
    }
    return NextResponse.next();
  }

  // Layout: 完全な検証（DB アクセスあり）
  export default async function Layout({ children }) {
    const session = await getSession(); // DB から最新情報を取得
    if (!session) redirect('/login');
    return <>{children}</>;
  }

---

【問題4: CORS エラーでログインできない】

症状: フロントエンドとバックエンドが別ドメインの場合、
      Cookie が送信されない / 設定されない

原因:
  - credentials: 'include' を設定していない
  - サーバーの CORS 設定が不適切
  - Cookie の SameSite 設定が厳しすぎる

解決策:
  // フロントエンド
  fetch('https://api.example.com/auth/login', {
    method: 'POST',
    credentials: 'include', // Cookie を送信
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });

  // バックエンド（Next.js API Route）
  const response = NextResponse.json({ success: true });
  response.headers.set('Access-Control-Allow-Origin', 'https://app.example.com');
  response.headers.set('Access-Control-Allow-Credentials', 'true');
  response.headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  response.headers.set('Access-Control-Allow-Headers', 'Content-Type');

  // Cookie 設定
  response.cookies.set('session-token', token, {
    httpOnly: true,
    secure: true,
    sameSite: 'none',  // クロスドメイン必要時（Secure 必須）
    domain: '.example.com', // サブドメイン共有
  });

---

【問題5: セッションが予期せず切れる】

症状: ユーザーがアクティブなのにログアウトされる

原因:
  - トークンの有効期限が短すぎる
  - スライディングウィンドウが実装されていない
  - 複数タブ間でのセッション競合
  - サーバー時刻のずれ

解決策:
  - トークン有効期限を適切に設定（推奨: 7日）
  - Middleware でトークンの有効期限延長を実装
  - BroadcastChannel でタブ間同期
  - NTP でサーバー時刻を同期

  // タブ間セッション同期
  const channel = new BroadcastChannel('auth');
  channel.addEventListener('message', (event) => {
    if (event.data.type === 'LOGOUT') {
      // 他のタブでログアウトされた
      window.location.href = '/login';
    }
    if (event.data.type === 'SESSION_UPDATED') {
      // 他のタブでセッションが更新された
      refreshSession();
    }
  });

  // ログアウト時に通知
  function logout() {
    channel.postMessage({ type: 'LOGOUT' });
    // ... ログアウト処理
  }

---

【問題6: 権限変更がリアルタイムに反映されない】

症状: 管理者がユーザーのロールを変更しても、
      そのユーザーのセッションに反映されない

原因:
  JWT にロール情報を含めている場合、
  トークンが更新されるまで古い情報が使われる

解決策:
  - セッション取得時に必ず DB からロールを確認する
  - トークンにはユーザーIDのみ含め、権限は毎回 DB から取得
  - 権限変更時にユーザーのセッションを強制無効化

  // セッション取得時に DB から最新の権限を取得
  export async function getSession(): Promise<Session | null> {
    const token = getTokenFromCookie();
    const payload = await verifyToken(token);

    // DB から最新の情報を取得（ここがポイント）
    const user = await prisma.user.findUnique({
      where: { id: payload.sub },
      select: { id: true, role: true, isActive: true },
    });

    if (!user || !user.isActive) return null;

    return { user }; // DB の最新情報を返す
  }
```

### 14.2 デバッグテクニック

```typescript
// ============================================
// 認証ガードのデバッグ
// ============================================

// 1. Middleware のデバッグログ
export async function middleware(request: NextRequest) {
  if (process.env.NODE_ENV === 'development') {
    console.log('[Middleware]', {
      pathname: request.nextUrl.pathname,
      method: request.method,
      hasCookie: !!request.cookies.get('session-token'),
      headers: Object.fromEntries(request.headers),
    });
  }

  // ... 認証ロジック
}

// 2. セッションデバッグエンドポイント（開発環境のみ）
// app/api/debug/session/route.ts
export async function GET(request: NextRequest) {
  if (process.env.NODE_ENV !== 'development') {
    return NextResponse.json({ error: 'Not found' }, { status: 404 });
  }

  const session = await getSession();
  const token = request.cookies.get('session-token')?.value;

  let tokenInfo = null;
  if (token) {
    try {
      const { payload } = await jwtVerify(
        token,
        new TextEncoder().encode(process.env.JWT_SECRET!)
      );
      tokenInfo = {
        sub: payload.sub,
        role: payload.role,
        iat: new Date((payload.iat || 0) * 1000).toISOString(),
        exp: new Date((payload.exp || 0) * 1000).toISOString(),
        expiresIn: `${Math.floor(((payload.exp || 0) - Date.now() / 1000) / 60)} minutes`,
      };
    } catch (e) {
      tokenInfo = { error: (e as Error).message };
    }
  }

  return NextResponse.json({
    session,
    token: tokenInfo,
    cookies: Object.fromEntries(
      request.cookies.getAll().map(c => [c.name, '***'])
    ),
  });
}

// 3. 権限デバッグコンポーネント（開発環境のみ）
function PermissionDebugger() {
  if (process.env.NODE_ENV !== 'development') return null;

  const { session } = useSession();
  const [isOpen, setIsOpen] = useState(false);

  if (!session) return null;

  const allPerms = getAllPermissions(session.user.role as Role);

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="bg-gray-800 text-white px-3 py-1 rounded text-xs"
      >
        Debug: {session.user.role}
      </button>
      {isOpen && (
        <div className="absolute bottom-10 right-0 bg-white border shadow-lg
                        rounded-lg p-4 w-80 max-h-96 overflow-y-auto">
          <h3 className="font-bold mb-2">権限一覧</h3>
          <ul className="text-xs space-y-1">
            {allPerms.map(p => (
              <li key={p} className="font-mono">{p}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
```

---

## まとめ

### 認証ガードの設計原則

| 原則 | 説明 | 実装 |
|------|------|------|
| 多層防御 | 複数レイヤーで保護 | Middleware + Layout + API |
| 最小権限 | 必要最小限の権限を付与 | RBAC / ABAC |
| フェイルセーフ | デフォルトで拒否 | 明示的な許可が必要 |
| 完全な仲介 | すべてのアクセスをチェック | API ルートごとに認証チェック |
| セキュアなデフォルト | 安全な初期設定 | HttpOnly + Secure + SameSite |

### レイヤー別の役割

| レイヤー | 役割 | ツール | 適切な用途 |
|---------|------|--------|-----------|
| Middleware | ルート保護 | middleware.ts | 大まかなアクセス制御、トークン検証 |
| Layout | エリア保護 | Server Component | セッション検証、共通UI制御 |
| Component | 要素制御 | RBAC + RequirePermission | ボタン表示/非表示 |
| API Route | データ保護 | withPermission HOF | データアクセスの最終防御 |
| Database | 行レベル保護 | RLS / Policies | データの最終保護 |

### フレームワーク選択ガイド

| 要件 | 推奨フレームワーク | 理由 |
|------|-----------------|------|
| SSR + 認証 | Next.js | Middleware + Server Component の統合 |
| SPA | React Router + バックエンド | クライアントガード + API 保護 |
| エンタープライズ | Angular | 組み込みのガードシステム |
| 軽量 SPA | Vue Router | シンプルなナビゲーションガード |

---

## 次に読むべきガイド
- [[00-form-design.md]] -- フォーム設計
- [[01-route-design.md]] -- ルート設計の基本
- [[02-dynamic-routing.md]] -- 動的ルーティング

---

## 参考文献
1. Next.js. "Authentication." nextjs.org/docs, 2024.
2. Auth.js. "NextAuth.js Documentation." authjs.dev, 2024.
3. Clerk. "Authentication for Next.js." clerk.com, 2024.
4. OWASP. "Authentication Cheat Sheet." cheatsheetseries.owasp.org, 2024.
5. OWASP. "Authorization Cheat Sheet." cheatsheetseries.owasp.org, 2024.
6. OWASP. "Session Management Cheat Sheet." cheatsheetseries.owasp.org, 2024.
7. RFC 6749. "The OAuth 2.0 Authorization Framework." datatracker.ietf.org, 2012.
8. RFC 7519. "JSON Web Token (JWT)." datatracker.ietf.org, 2015.
9. React Router. "Authentication." reactrouter.com/docs, 2024.
10. Vue Router. "Navigation Guards." router.vuejs.org, 2024.
11. Angular. "Route Guards." angular.dev/guide/routing, 2024.
12. NIST SP 800-63B. "Digital Identity Guidelines: Authentication and Lifecycle Management." nist.gov, 2020.
