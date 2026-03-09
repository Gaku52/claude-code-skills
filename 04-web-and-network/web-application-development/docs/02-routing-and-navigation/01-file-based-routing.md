# ファイルベースルーティング

> ファイルベースルーティングは「ファイル構造 = URL構造」の直感的なアプローチ。Next.js App Router、Remix のルーティング規約、レイアウト、ローディング、エラーハンドリングまで、ファイルベースルーティングの全パターンを習得する。

## この章で学ぶこと

- [ ] ファイルベースルーティングの概念と歴史的背景を理解する
- [ ] Next.js App Routerのファイル規約を完全に理解する
- [ ] レイアウト、ローディング、エラーの設計パターンを把握する
- [ ] 動的ルート、ルートグループ、パラレルルートを学ぶ
- [ ] Remix / React Router v7 のファイルルーティングとの比較を理解する
- [ ] Nuxt.js / SvelteKit など他フレームワークとの比較ができる
- [ ] 実践的なプロジェクトでのディレクトリ設計ができる
- [ ] トラブルシューティングとアンチパターンの回避ができる

## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- クライアントサイドルーティングの基本概念 — [クライアントルーティング](./00-client-side-routing.md)
- Next.js の基本構造（App Router と Pages Router の違い、Server Components と Client Components の概念）
- ファイルシステムの概念（ディレクトリ構造、相対パス・絶対パス、ファイル命名規則）

---

## 0. ファイルベースルーティングとは何か

### 0.1 概要と歴史的背景

ファイルベースルーティングとは、ファイルシステム上のディレクトリ構造がそのままURLパスに対応するルーティングの仕組みである。従来の React アプリケーションでは `react-router` を使い、コード内でルートを宣言的に定義する必要があったが、ファイルベースルーティングではディレクトリにファイルを配置するだけで自動的にルートが生成される。

```
従来のアプローチ（コードベースルーティング）:
  // routes.tsx
  <Routes>
    <Route path="/" element={<Home />} />
    <Route path="/about" element={<About />} />
    <Route path="/users" element={<Users />} />
    <Route path="/users/:id" element={<UserDetail />} />
    <Route path="/users/:id/edit" element={<UserEdit />} />
    <Route path="/settings" element={<Settings />} />
  </Routes>

ファイルベースルーティング:
  app/
  ├── page.tsx            → /
  ├── about/page.tsx      → /about
  ├── users/
  │   ├── page.tsx        → /users
  │   ├── [id]/
  │   │   ├── page.tsx    → /users/:id
  │   │   └── edit/
  │   │       └── page.tsx → /users/:id/edit
  └── settings/
      └── page.tsx        → /settings
```

この仕組みは PHP 時代から存在し、`index.php` を配置するだけでそのディレクトリのURLにアクセスできた。現代のフレームワークはこの直感的なアプローチを進化させ、レイアウト・エラー処理・ローディング状態など、より高度な機能をファイル規約として取り入れている。

### 0.2 ファイルベースルーティングのメリット

| メリット | 詳細 |
|---------|------|
| 直感的な構造 | URLとファイルパスが1対1対応するため、コードの場所が即座にわかる |
| 設定不要 | ルーティング設定ファイルが不要で、ファイルを置くだけで動作する |
| コロケーション | ページに関連するコンポーネント・テスト・スタイルを同じディレクトリに配置できる |
| 自動コード分割 | フレームワークがページ単位で自動的にコード分割を行える |
| 型安全性 | フレームワークがルートパラメータの型を自動生成できる |
| チーム開発の効率化 | ファイル構造がルートの一覧表として機能し、新メンバーのオンボーディングが容易 |

### 0.3 ファイルベースルーティングのデメリット

| デメリット | 詳細 |
|----------|------|
| フレームワーク依存 | 各フレームワーク固有の規約を覚える必要がある |
| 複雑なルートの表現 | 条件付きルートや高度なルーティングロジックが煩雑になることがある |
| ファイル数の増加 | 小さなファイルが大量に生まれ、ディレクトリが深くなりがち |
| リファクタリングの困難さ | URL変更にファイル移動が伴い、import パスも変わる |
| テスト設計 | ファイル規約に縛られるため、ルーティングロジックの単体テストが難しい |

### 0.4 主要フレームワークのファイルベースルーティング比較

| フレームワーク | ディレクトリ | 動的ルート | キャッチオール | レイアウト | ルートグループ |
|---------------|------------|-----------|-------------|-----------|-------------|
| Next.js App Router | `app/` | `[param]` | `[...slug]` | `layout.tsx` | `(group)` |
| Next.js Pages Router | `pages/` | `[param]` | `[...slug]` | `_app.tsx` | N/A |
| Remix v2 | `app/routes/` | `$param` → `[param]` | `$.tsx` | `_layout.tsx` | `_index` |
| Nuxt.js 3 | `pages/` | `[param]` | `[...slug]` | `layouts/` | N/A |
| SvelteKit | `src/routes/` | `[param]` | `[...rest]` | `+layout.svelte` | `(group)` |
| Astro | `src/pages/` | `[param]` | `[...slug]` | `layouts/` | N/A |

---

## 1. Next.js App Router のファイル規約

### 1.1 ディレクトリ構造の全体像

Next.js 13 で導入された App Router は、React Server Components を前提とした新しいルーティングシステムである。`app/` ディレクトリ内のファイル配置がそのままルーティングとなる。

```
ファイル規約:
  app/
  ├── layout.tsx          ← ルートレイアウト（必須）
  ├── page.tsx            ← / のページ
  ├── loading.tsx         ← ローディングUI
  ├── error.tsx           ← エラーUI
  ├── not-found.tsx       ← 404 UI
  ├── global-error.tsx    ← グローバルエラーUI（layout.tsxのエラーをキャッチ）
  ├── template.tsx        ← テンプレート（遷移ごとに再マウント）
  ├── default.tsx         ← パラレルルートのデフォルト表示
  ├── favicon.ico         ← ファビコン（自動設定）
  ├── opengraph-image.png ← OGP画像（自動設定）
  ├── sitemap.ts          ← サイトマップ生成
  ├── robots.ts           ← robots.txt 生成
  ├── manifest.ts         ← Web App Manifest
  ├── users/
  │   ├── page.tsx        ← /users
  │   ├── loading.tsx     ← /users のローディング
  │   ├── error.tsx       ← /users のエラー
  │   ├── [id]/
  │   │   ├── page.tsx    ← /users/:id
  │   │   ├── layout.tsx  ← /users/:id 共有レイアウト
  │   │   └── edit/
  │   │       └── page.tsx ← /users/:id/edit
  │   └── new/
  │       └── page.tsx    ← /users/new
  ├── blog/
  │   ├── page.tsx        ← /blog
  │   └── [...slug]/
  │       └── page.tsx    ← /blog/any/path/here
  └── api/
      └── webhooks/
          └── route.ts    ← API Route: POST /api/webhooks

特殊ファイル一覧:
  page.tsx       → ルートのUIコンポーネント（これがないとルートとして認識されない）
  layout.tsx     → 共有レイアウト（再レンダリングされない、状態が保持される）
  template.tsx   → layout同様だが遷移ごとに再マウント（状態がリセットされる）
  loading.tsx    → Suspense の fallback（自動ラップ）
  error.tsx      → ErrorBoundary（自動ラップ、'use client' 必須）
  global-error.tsx → ルートレイアウトのエラーをキャッチ（'use client' 必須）
  not-found.tsx  → notFound() 呼び出し時のUI
  route.ts       → API Route（HTTPハンドラー、page.tsx と共存不可）
  default.tsx    → パラレルルートでマッチしない場合のフォールバック
  middleware.ts  → ルートレベルのミドルウェア（app/直下ではなくプロジェクトルートに配置）
```

### 1.2 特殊ファイルの実行順序と階層

Next.js App Router における特殊ファイルは、コンポーネントツリーとして以下の階層でレンダリングされる。この階層を理解することが、正しいエラー処理とローディング設計の鍵となる。

```
コンポーネント階層（上から下へネスト）:

  layout.tsx
  └── template.tsx
      └── error.tsx (ErrorBoundary)
          └── loading.tsx (Suspense)
              └── not-found.tsx
                  └── page.tsx

実際に生成されるReactツリー:

  <Layout>
    <Template>
      <ErrorBoundary fallback={<Error />}>
        <Suspense fallback={<Loading />}>
          <NotFoundBoundary fallback={<NotFound />}>
            <Page />
          </NotFoundBoundary>
        </Suspense>
      </ErrorBoundary>
    </Template>
  </Layout>
```

この階層から導かれる重要な特性:

1. **layout.tsx のエラーは error.tsx でキャッチできない** — error.tsx は layout の子として配置されるため、layout 自身のエラーをキャッチするには親セグメントの error.tsx か global-error.tsx が必要
2. **loading.tsx は error.tsx の内側** — エラーが発生した場合、ローディング状態よりエラー表示が優先される
3. **template.tsx は layout.tsx の子** — template が再マウントされても layout の状態は保持される

```typescript
// この階層を理解するための実験的コード
// app/test/layout.tsx
export default function TestLayout({ children }: { children: React.ReactNode }) {
  console.log('Layout rendered');  // ページ遷移時に再実行されない
  return <div className="test-layout">{children}</div>;
}

// app/test/template.tsx
export default function TestTemplate({ children }: { children: React.ReactNode }) {
  console.log('Template rendered');  // ページ遷移ごとに再実行される
  return <div className="test-template">{children}</div>;
}

// app/test/error.tsx
'use client';
export default function TestError({ error, reset }: { error: Error; reset: () => void }) {
  console.log('Error boundary caught:', error.message);
  return <button onClick={reset}>Retry</button>;
}

// app/test/loading.tsx
export default function TestLoading() {
  console.log('Loading rendered');
  return <div>Loading...</div>;
}

// app/test/page.tsx
export default async function TestPage() {
  console.log('Page rendered');
  return <div>Test Page Content</div>;
}
```

### 1.3 page.tsx の詳細

`page.tsx` はルートをレンダリング可能にする最も重要なファイルである。`page.tsx` が存在しないディレクトリはルートとして認識されず、URLにアクセスしても404になる。

```typescript
// app/page.tsx — トップページ（Server Component がデフォルト）
import { Suspense } from 'react';

// Server Component ではデータフェッチを直接 await できる
export default async function HomePage() {
  const featuredPosts = await getFeaturedPosts();
  const categories = await getCategories();

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-4xl font-bold mb-8">Welcome to Our Blog</h1>

      {/* 重いデータフェッチは Suspense で分離 */}
      <Suspense fallback={<FeaturedPostsSkeleton />}>
        <FeaturedPosts posts={featuredPosts} />
      </Suspense>

      <Suspense fallback={<CategoriesSkeleton />}>
        <Categories categories={categories} />
      </Suspense>
    </div>
  );
}

// メタデータの静的定義
export const metadata = {
  title: 'Home | My Blog',
  description: 'Welcome to our blog featuring the latest articles.',
  openGraph: {
    title: 'Home | My Blog',
    description: 'Welcome to our blog featuring the latest articles.',
    type: 'website',
  },
};
```

```typescript
// app/dashboard/page.tsx — クライアントコンポーネントが必要な場合
'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function DashboardPage() {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    fetch('/api/dashboard')
      .then(res => res.json())
      .then(data => {
        setData(data);
        setIsLoading(false);
      });
  }, []);

  if (isLoading) return <DashboardSkeleton />;

  return (
    <div>
      <h1>Dashboard</h1>
      <DashboardContent data={data} />
    </div>
  );
}

// 注意: 'use client' を page.tsx に付けると、
// そのページ全体が Client Component になり、
// Server Component のメリット（データフェッチ、バンドルサイズ削減）を失う。
// 可能な限り page.tsx は Server Component として保ち、
// インタラクティブな部分だけを Client Component として分離する。
```

### 1.4 route.ts（API Route）の詳細

`route.ts` は RESTful な API エンドポイントを定義するための特殊ファイルである。同じディレクトリに `page.tsx` と `route.ts` を共存させることはできない。

```typescript
// app/api/users/route.ts — RESTful API エンドポイント
import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';

// GET /api/users — ユーザー一覧の取得
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const page = parseInt(searchParams.get('page') ?? '1', 10);
  const limit = parseInt(searchParams.get('limit') ?? '20', 10);
  const search = searchParams.get('search') ?? '';

  try {
    const users = await db.user.findMany({
      where: search
        ? { name: { contains: search, mode: 'insensitive' } }
        : undefined,
      skip: (page - 1) * limit,
      take: limit,
      orderBy: { createdAt: 'desc' },
    });

    const total = await db.user.count({
      where: search
        ? { name: { contains: search, mode: 'insensitive' } }
        : undefined,
    });

    return NextResponse.json({
      users,
      pagination: {
        page,
        limit,
        total,
        totalPages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error('Failed to fetch users:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}

// POST /api/users — ユーザーの作成
const createUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  role: z.enum(['admin', 'user', 'editor']).default('user'),
});

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const validated = createUserSchema.parse(body);

    const existingUser = await db.user.findUnique({
      where: { email: validated.email },
    });

    if (existingUser) {
      return NextResponse.json(
        { error: 'Email already exists' },
        { status: 409 }
      );
    }

    const user = await db.user.create({ data: validated });

    return NextResponse.json(user, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json(
        { error: 'Validation failed', details: error.errors },
        { status: 400 }
      );
    }
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}
```

```typescript
// app/api/users/[id]/route.ts — 個別リソースの操作
import { NextRequest, NextResponse } from 'next/server';

// GET /api/users/:id
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const user = await db.user.findUnique({ where: { id } });

  if (!user) {
    return NextResponse.json(
      { error: 'User not found' },
      { status: 404 }
    );
  }

  return NextResponse.json(user);
}

// PUT /api/users/:id
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const body = await request.json();

  try {
    const user = await db.user.update({
      where: { id },
      data: body,
    });
    return NextResponse.json(user);
  } catch (error) {
    return NextResponse.json(
      { error: 'User not found' },
      { status: 404 }
    );
  }
}

// DELETE /api/users/:id
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;

  try {
    await db.user.delete({ where: { id } });
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    return NextResponse.json(
      { error: 'User not found' },
      { status: 404 }
    );
  }
}

// PATCH /api/users/:id
export async function PATCH(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const body = await request.json();

  try {
    const user = await db.user.update({
      where: { id },
      data: body,
    });
    return NextResponse.json(user);
  } catch (error) {
    return NextResponse.json(
      { error: 'User not found' },
      { status: 404 }
    );
  }
}
```

```typescript
// app/api/webhooks/stripe/route.ts — Webhook エンドポイントの例
import { NextRequest, NextResponse } from 'next/server';
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!);
const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET!;

export async function POST(request: NextRequest) {
  const body = await request.text(); // raw body が必要
  const sig = request.headers.get('stripe-signature')!;

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(body, sig, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return NextResponse.json(
      { error: 'Invalid signature' },
      { status: 400 }
    );
  }

  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object as Stripe.Checkout.Session;
      await handleCheckoutComplete(session);
      break;
    }
    case 'customer.subscription.deleted': {
      const subscription = event.data.object as Stripe.Subscription;
      await handleSubscriptionCancelled(subscription);
      break;
    }
    default:
      console.log(`Unhandled event type: ${event.type}`);
  }

  return NextResponse.json({ received: true });
}

// route segment config: Webhook は動的である必要がある
export const dynamic = 'force-dynamic';
```

### 1.5 Route Segment Config（ルートセグメント設定）

各ルートセグメントで export できる設定値があり、キャッシュ・再検証・ランタイムなどの挙動を制御できる。

```typescript
// app/blog/page.tsx
// Route Segment Config の全オプション

// 動的レンダリングの制御
export const dynamic = 'auto';
// 'auto'          — デフォルト、フレームワークが判断
// 'force-dynamic' — 常に動的レンダリング（SSR）
// 'error'         — 静的レンダリングを強制（動的関数があるとビルドエラー）
// 'force-static'  — 動的関数の戻り値を空にして静的レンダリングを強制

// 動的パラメータの制御
export const dynamicParams = true;
// true  — generateStaticParams にないパラメータも動的に生成
// false — generateStaticParams にないパラメータは404

// 再検証の間隔（秒）
export const revalidate = 3600; // 1時間ごとに再検証
// false — 再検証しない（無期限キャッシュ）
// 0     — 常に動的レンダリング

// ランタイムの選択
export const runtime = 'nodejs';
// 'nodejs'  — Node.js ランタイム（デフォルト）
// 'edge'    — Edge Runtime（軽量、制限あり）

// 使用する Node.js API の明示
export const preferredRegion = 'auto';
// 'auto'    — フレームワークが判断
// 'global'  — グローバル
// 'home'    — ホームリージョン
// ['iad1', 'sfo1'] — 特定リージョン

// 最大実行時間（秒）
export const maxDuration = 30;

export default async function BlogPage() {
  const posts = await fetch('https://api.example.com/posts', {
    next: { revalidate: 3600 }, // fetch レベルでも再検証設定可能
  }).then(res => res.json());

  return <PostList posts={posts} />;
}
```

---

## 2. レイアウトの設計

### 2.1 ルートレイアウト（Root Layout）

ルートレイアウトは `app/layout.tsx` に配置される必須のファイルで、`<html>` と `<body>` タグを含む必要がある。全ページで共有され、ページ遷移時にも再レンダリングされない。

```typescript
// app/layout.tsx — ルートレイアウト
import type { Metadata, Viewport } from 'next';
import { Inter, Noto_Sans_JP } from 'next/font/google';
import { Analytics } from '@vercel/analytics/react';
import { SpeedInsights } from '@vercel/speed-insights/next';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
});

const notoSansJP = Noto_Sans_JP({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-noto-sans-jp',
});

export const metadata: Metadata = {
  title: {
    template: '%s | My App',       // 子ページのtitleが %s に入る
    default: 'My App',              // titleが設定されていない場合
  },
  description: 'A modern web application built with Next.js',
  metadataBase: new URL('https://example.com'),
  openGraph: {
    type: 'website',
    locale: 'ja_JP',
    url: 'https://example.com',
    siteName: 'My App',
  },
  twitter: {
    card: 'summary_large_image',
    creator: '@example',
  },
  robots: {
    index: true,
    follow: true,
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#000000' },
  ],
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="ja"
      className={`${inter.variable} ${notoSansJP.variable}`}
      suppressHydrationWarning  // next-themes 等のダークモード対応時に必要
    >
      <body className="min-h-screen bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <AuthProvider>
            <QueryProvider>
              <Header />
              <main className="flex-1">{children}</main>
              <Footer />
              <Toaster />
            </QueryProvider>
          </AuthProvider>
        </ThemeProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
```

### 2.2 ネストされたレイアウト

レイアウトはディレクトリごとにネストでき、親レイアウトの中に子レイアウトが配置される。これにより、セクションごとに異なるレイアウトを適用できる。

```typescript
// app/dashboard/layout.tsx — ダッシュボード用レイアウト
import { Sidebar } from '@/components/dashboard/sidebar';
import { DashboardHeader } from '@/components/dashboard/header';
import { getSession } from '@/lib/auth';
import { redirect } from 'next/navigation';

export const metadata = {
  title: 'Dashboard',
};

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Server Component なので直接認証チェックが可能
  const session = await getSession();
  if (!session) {
    redirect('/login');
  }

  return (
    <div className="flex h-screen">
      {/* サイドバー */}
      <Sidebar user={session.user} />

      {/* メインコンテンツエリア */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader user={session.user} />
        <main className="flex-1 overflow-y-auto p-6 bg-gray-50 dark:bg-gray-900">
          {children}
        </main>
      </div>
    </div>
  );
}
```

```typescript
// app/dashboard/settings/layout.tsx — 設定画面のサブレイアウト
import { SettingsNav } from '@/components/settings/nav';

const settingsNavItems = [
  { href: '/dashboard/settings', label: 'General', icon: 'settings' },
  { href: '/dashboard/settings/profile', label: 'Profile', icon: 'user' },
  { href: '/dashboard/settings/billing', label: 'Billing', icon: 'credit-card' },
  { href: '/dashboard/settings/notifications', label: 'Notifications', icon: 'bell' },
  { href: '/dashboard/settings/security', label: 'Security', icon: 'shield' },
  { href: '/dashboard/settings/api-keys', label: 'API Keys', icon: 'key' },
];

export default function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>
      <div className="flex gap-8">
        <aside className="w-64 shrink-0">
          <SettingsNav items={settingsNavItems} />
        </aside>
        <div className="flex-1 min-w-0">
          {children}
        </div>
      </div>
    </div>
  );
}

// 結果として以下のレイアウト階層が形成される:
// RootLayout → DashboardLayout → SettingsLayout → Page
//
// /dashboard/settings/profile にアクセスすると:
//   <RootLayout>
//     <DashboardLayout>
//       <SettingsLayout>
//         <ProfilePage />
//       </SettingsLayout>
//     </DashboardLayout>
//   </RootLayout>
```

### 2.3 ルートグループ（Route Groups）

ルートグループは `(name)` の形式でディレクトリ名を括弧で囲むことで、URL構造に影響を与えずにファイルを論理的にグループ化する機能である。

```
ルートグループの活用例:

app/
├── (marketing)/              ← URLに含まれない
│   ├── layout.tsx            ← マーケティングページ用レイアウト
│   ├── page.tsx              ← / (トップページ)
│   ├── about/
│   │   └── page.tsx          ← /about
│   ├── pricing/
│   │   └── page.tsx          ← /pricing
│   ├── blog/
│   │   ├── page.tsx          ← /blog
│   │   └── [slug]/
│   │       └── page.tsx      ← /blog/:slug
│   └── contact/
│       └── page.tsx          ← /contact
│
├── (app)/                    ← URLに含まれない
│   ├── layout.tsx            ← アプリケーション用レイアウト（認証必須）
│   ├── dashboard/
│   │   └── page.tsx          ← /dashboard
│   ├── projects/
│   │   ├── page.tsx          ← /projects
│   │   └── [id]/
│   │       └── page.tsx      ← /projects/:id
│   └── settings/
│       └── page.tsx          ← /settings
│
├── (auth)/                   ← URLに含まれない
│   ├── layout.tsx            ← 認証ページ用レイアウト（センタリング等）
│   ├── login/
│   │   └── page.tsx          ← /login
│   ├── register/
│   │   └── page.tsx          ← /register
│   └── forgot-password/
│       └── page.tsx          ← /forgot-password
│
└── layout.tsx                ← ルートレイアウト（全グループ共通）
```

```typescript
// app/(marketing)/layout.tsx — マーケティング用レイアウト
export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="max-w-7xl mx-auto">
      <MarketingNav />
      {children}
      <MarketingFooter />
    </div>
  );
}

// app/(app)/layout.tsx — アプリ用レイアウト（認証付き）
export default async function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();
  if (!session) redirect('/login');

  return (
    <div className="flex min-h-screen">
      <AppSidebar />
      <div className="flex-1">
        <AppHeader user={session.user} />
        <main className="p-6">{children}</main>
      </div>
    </div>
  );
}

// app/(auth)/layout.tsx — 認証ページ用レイアウト
export default async function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // すでにログイン済みならダッシュボードへリダイレクト
  const session = await getSession();
  if (session) redirect('/dashboard');

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full">
        <Logo className="mx-auto mb-8" />
        {children}
      </div>
    </div>
  );
}
```

### 2.4 layout.tsx と template.tsx の違い

`layout.tsx` と `template.tsx` は似た役割を持つが、重要な違いがある。

```typescript
// layout.tsx の特性:
// - ページ遷移時に再レンダリングされない（状態が保持される）
// - useEffect が再実行されない
// - DOM が再利用される

// template.tsx の特性:
// - ページ遷移ごとに新しいインスタンスが作成される
// - useEffect が毎回実行される
// - DOM が再作成される

// template.tsx が適している場面:
// 1. ページ遷移アニメーション
// 2. ページビューのログ記録
// 3. ページごとのフィードバックフォーム

// app/dashboard/template.tsx — ページ遷移ログの例
'use client';

import { useEffect } from 'react';
import { usePathname } from 'next/navigation';
import { analytics } from '@/lib/analytics';

export default function DashboardTemplate({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  useEffect(() => {
    // template.tsx なので遷移ごとに実行される
    analytics.pageView(pathname);
  }, [pathname]);

  return (
    <div className="animate-fadeIn">
      {children}
    </div>
  );
}

// もし layout.tsx に同じコードを書いた場合、
// 初回レンダリング時にしか useEffect が実行されず、
// 子ページ間の遷移ではログが記録されない。
```

| 特性 | layout.tsx | template.tsx |
|------|-----------|-------------|
| 再マウント | されない | ページ遷移ごとに再マウント |
| 状態保持 | される | リセットされる |
| useEffect | 初回のみ | 遷移ごとに実行 |
| パフォーマンス | 高い（再利用） | 低い（再作成） |
| 用途 | ナビゲーション、サイドバー | アニメーション、ログ記録 |

---

## 3. 動的ルートとキャッチオール

### 3.1 動的ルートの基本

動的ルートはURLの一部をパラメータとして受け取るためのルーティングパターンである。ディレクトリ名を角括弧で囲むことで定義する。

```
動的ルートの種類:

  [id]           → 単一の動的セグメント
                   /users/123        → params.id = "123"
                   /users/abc        → params.id = "abc"

  [slug]         → 命名は自由（慣習的に slug, id, name 等を使用）
                   /posts/hello-world → params.slug = "hello-world"

  [...slug]      → キャッチオール（1つ以上のセグメント）
                   /docs/a           → params.slug = ["a"]
                   /docs/a/b         → params.slug = ["a", "b"]
                   /docs/a/b/c       → params.slug = ["a", "b", "c"]
                   /docs             → 404（マッチしない）

  [[...slug]]    → オプショナルキャッチオール（0個以上のセグメント）
                   /docs             → params.slug = undefined
                   /docs/a           → params.slug = ["a"]
                   /docs/a/b         → params.slug = ["a", "b"]
```

### 3.2 動的ルートの実装例

```typescript
// app/users/[id]/page.tsx — 基本的な動的ルート
import { notFound } from 'next/navigation';
import { Suspense } from 'react';
import { UserProfile } from '@/components/user/profile';
import { UserPosts } from '@/components/user/posts';
import { UserPostsSkeleton } from '@/components/user/posts-skeleton';

interface UserPageProps {
  params: Promise<{ id: string }>;
}

export default async function UserPage({ params }: UserPageProps) {
  const { id } = await params;

  // パラメータのバリデーション
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
    notFound();
  }

  const user = await getUser(id);
  if (!user) {
    notFound();
  }

  return (
    <div className="max-w-4xl mx-auto py-8">
      {/* ユーザー基本情報（すぐに表示） */}
      <UserProfile user={user} />

      {/* ユーザーの投稿一覧（遅延ロード） */}
      <section className="mt-8">
        <h2 className="text-xl font-bold mb-4">Posts</h2>
        <Suspense fallback={<UserPostsSkeleton />}>
          <UserPosts userId={user.id} />
        </Suspense>
      </section>
    </div>
  );
}

// 静的パラメータの生成（ビルド時に生成するページを指定）
export async function generateStaticParams() {
  const users = await db.user.findMany({
    select: { id: true },
    take: 100, // 主要なユーザーページのみ事前生成
  });

  return users.map((user) => ({
    id: user.id,
  }));
}

// メタデータの動的生成
export async function generateMetadata({ params }: UserPageProps) {
  const { id } = await params;
  const user = await getUser(id);

  if (!user) {
    return {
      title: 'User Not Found',
    };
  }

  return {
    title: user.name,
    description: user.bio ?? `${user.name}'s profile`,
    openGraph: {
      title: user.name,
      description: user.bio ?? `${user.name}'s profile`,
      images: user.avatar ? [{ url: user.avatar }] : [],
    },
  };
}
```

### 3.3 複数の動的セグメント

```typescript
// app/[locale]/blog/[category]/[slug]/page.tsx
// URL: /ja/blog/tech/nextjs-routing
interface BlogPostPageProps {
  params: Promise<{
    locale: string;
    category: string;
    slug: string;
  }>;
}

export default async function BlogPostPage({ params }: BlogPostPageProps) {
  const { locale, category, slug } = await params;

  // ロケールバリデーション
  const supportedLocales = ['ja', 'en', 'zh', 'ko'];
  if (!supportedLocales.includes(locale)) {
    notFound();
  }

  const post = await getPost({ locale, category, slug });
  if (!post) {
    notFound();
  }

  return (
    <article className="prose prose-lg mx-auto">
      <header>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <span>{category}</span>
          <span>/</span>
          <time dateTime={post.publishedAt}>
            {new Date(post.publishedAt).toLocaleDateString(locale)}
          </time>
        </div>
        <h1>{post.title}</h1>
      </header>
      <div dangerouslySetInnerHTML={{ __html: post.contentHtml }} />
    </article>
  );
}

export async function generateStaticParams() {
  const posts = await getAllPosts();

  return posts.flatMap((post) =>
    post.locales.map((locale) => ({
      locale,
      category: post.category,
      slug: post.slug,
    }))
  );
}
```

### 3.4 キャッチオールルートの実践例

```typescript
// app/docs/[[...slug]]/page.tsx — ドキュメントサイトのキャッチオール
import { notFound } from 'next/navigation';
import { getDocBySlug, getAllDocs } from '@/lib/docs';
import { TableOfContents } from '@/components/docs/toc';
import { DocBreadcrumb } from '@/components/docs/breadcrumb';
import { DocPagination } from '@/components/docs/pagination';

interface DocsPageProps {
  params: Promise<{ slug?: string[] }>;
}

export default async function DocsPage({ params }: DocsPageProps) {
  const { slug } = await params;

  // /docs にアクセスした場合は introduction を表示
  const docPath = slug?.join('/') ?? 'introduction';
  const doc = await getDocBySlug(docPath);

  if (!doc) {
    notFound();
  }

  return (
    <div className="flex gap-8">
      {/* メインコンテンツ */}
      <article className="flex-1 min-w-0 prose prose-lg dark:prose-invert">
        <DocBreadcrumb segments={slug ?? []} />
        <h1>{doc.title}</h1>
        <div dangerouslySetInnerHTML={{ __html: doc.contentHtml }} />
        <DocPagination current={docPath} />
      </article>

      {/* 目次サイドバー */}
      <aside className="hidden xl:block w-64 shrink-0">
        <TableOfContents headings={doc.headings} />
      </aside>
    </div>
  );
}

export async function generateStaticParams() {
  const docs = await getAllDocs();

  return [
    { slug: undefined },  // /docs (introduction)
    ...docs.map((doc) => ({
      slug: doc.path.split('/'),
    })),
  ];
}

export async function generateMetadata({ params }: DocsPageProps) {
  const { slug } = await params;
  const docPath = slug?.join('/') ?? 'introduction';
  const doc = await getDocBySlug(docPath);

  return {
    title: doc?.title ?? 'Documentation',
    description: doc?.description ?? 'Project documentation',
  };
}
```

### 3.5 動的ルートの優先順位

Next.js App Router では、静的なルートが動的ルートより優先される。この優先順位を理解することは、予期しない挙動を防ぐために重要である。

```
ルートの優先順位（高い順）:

  1. 静的ルート          /users/new        → app/users/new/page.tsx
  2. 動的ルート          /users/123        → app/users/[id]/page.tsx
  3. キャッチオール       /users/123/posts  → app/users/[...slug]/page.tsx
  4. オプショナルキャッチ  /users            → app/users/[[...slug]]/page.tsx

例: 以下のファイル構造で /users/new にアクセスした場合
  app/users/
  ├── [id]/page.tsx        ← /users/new はここにマッチしない
  ├── new/page.tsx         ← こちらが優先される ✓
  └── [...slug]/page.tsx   ← マッチしない

注意事項:
  - /users/new は静的ルートなので [id] より優先
  - 明示的に new/page.tsx を作らないと [id] にマッチしてしまう
  - API Route でも同じ優先順位が適用される
```

### 3.6 searchParams の活用

動的ルートのパラメータに加え、クエリパラメータ（searchParams）も Server Component で直接アクセスできる。

```typescript
// app/products/page.tsx — フィルタリング・ソート・ページネーション
interface ProductsPageProps {
  searchParams: Promise<{
    category?: string;
    sort?: string;
    order?: 'asc' | 'desc';
    page?: string;
    q?: string;
  }>;
}

export default async function ProductsPage({
  searchParams,
}: ProductsPageProps) {
  const {
    category,
    sort = 'createdAt',
    order = 'desc',
    page = '1',
    q,
  } = await searchParams;

  const currentPage = Math.max(1, parseInt(page, 10) || 1);
  const limit = 20;

  const { products, total } = await getProducts({
    category,
    sort,
    order,
    page: currentPage,
    limit,
    search: q,
  });

  return (
    <div>
      <h1>Products</h1>

      {/* フィルタバー */}
      <ProductFilters
        currentCategory={category}
        currentSort={sort}
        currentOrder={order}
        searchQuery={q}
      />

      {/* 商品一覧 */}
      <ProductGrid products={products} />

      {/* ページネーション */}
      <Pagination
        currentPage={currentPage}
        totalPages={Math.ceil(total / limit)}
        baseUrl="/products"
        searchParams={{ category, sort, order, q }}
      />
    </div>
  );
}

// メタデータにも searchParams を使用可能
export async function generateMetadata({
  searchParams,
}: ProductsPageProps) {
  const { category, q } = await searchParams;

  let title = 'Products';
  if (category) title = `${category} Products`;
  if (q) title = `Search: ${q}`;

  return { title };
}
```

---

## 4. ローディングとエラーハンドリング

### 4.1 loading.tsx の詳細設計

`loading.tsx` は React の `<Suspense>` をファイル規約で表現したものである。配置されたディレクトリ以下のすべてのページコンポーネントに対して自動的にローディング UI を提供する。

```typescript
// app/users/loading.tsx — スケルトンUIの実装
export default function UsersLoading() {
  return (
    <div className="space-y-4">
      {/* ヘッダースケルトン */}
      <div className="flex items-center justify-between">
        <div className="h-8 w-48 bg-gray-200 animate-pulse rounded" />
        <div className="h-10 w-32 bg-gray-200 animate-pulse rounded" />
      </div>

      {/* 検索バースケルトン */}
      <div className="h-10 w-full bg-gray-200 animate-pulse rounded" />

      {/* テーブルヘッダー */}
      <div className="h-12 w-full bg-gray-100 animate-pulse rounded-t" />

      {/* テーブル行スケルトン */}
      {Array.from({ length: 10 }).map((_, i) => (
        <div
          key={i}
          className="flex items-center gap-4 p-4 border-b"
        >
          {/* アバター */}
          <div className="h-10 w-10 bg-gray-200 animate-pulse rounded-full" />
          {/* 名前 */}
          <div className="h-4 w-32 bg-gray-200 animate-pulse rounded" />
          {/* メール */}
          <div className="h-4 w-48 bg-gray-200 animate-pulse rounded" />
          {/* ロール */}
          <div className="h-4 w-20 bg-gray-200 animate-pulse rounded" />
          {/* 日付 */}
          <div className="h-4 w-24 bg-gray-200 animate-pulse rounded ml-auto" />
        </div>
      ))}

      {/* ページネーションスケルトン */}
      <div className="flex justify-center gap-2 mt-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-8 w-8 bg-gray-200 animate-pulse rounded" />
        ))}
      </div>
    </div>
  );
}
```

```typescript
// 再利用可能なスケルトンコンポーネント
// components/ui/skeleton.tsx
import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

export function Skeleton({
  className,
  variant = 'text',
  width,
  height,
  animation = 'pulse',
}: SkeletonProps) {
  return (
    <div
      className={cn(
        'bg-gray-200 dark:bg-gray-700',
        animation === 'pulse' && 'animate-pulse',
        animation === 'wave' && 'animate-shimmer',
        variant === 'circular' && 'rounded-full',
        variant === 'rectangular' && 'rounded',
        variant === 'text' && 'rounded h-4',
        className
      )}
      style={{ width, height }}
    />
  );
}

// app/dashboard/loading.tsx — Skeleton コンポーネントを使った例
import { Skeleton } from '@/components/ui/skeleton';

export default function DashboardLoading() {
  return (
    <div className="space-y-6">
      {/* KPIカード */}
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="p-6 border rounded-lg">
            <Skeleton className="h-4 w-24 mb-2" />
            <Skeleton className="h-8 w-16" />
            <Skeleton className="h-3 w-32 mt-2" />
          </div>
        ))}
      </div>

      {/* チャートエリア */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-6 border rounded-lg">
          <Skeleton className="h-6 w-32 mb-4" />
          <Skeleton variant="rectangular" className="h-64 w-full" />
        </div>
        <div className="p-6 border rounded-lg">
          <Skeleton className="h-6 w-32 mb-4" />
          <Skeleton variant="rectangular" className="h-64 w-full" />
        </div>
      </div>
    </div>
  );
}
```

### 4.2 Suspense との組み合わせ

`loading.tsx` はルートセグメント全体に対するローディング UI だが、より細かい粒度で制御したい場合は `<Suspense>` を直接使用する。

```typescript
// app/dashboard/page.tsx — Suspense で部分的なストリーミング
import { Suspense } from 'react';
import { KPICards } from '@/components/dashboard/kpi-cards';
import { RecentOrders } from '@/components/dashboard/recent-orders';
import { SalesChart } from '@/components/dashboard/sales-chart';
import { TopProducts } from '@/components/dashboard/top-products';
import { Skeleton } from '@/components/ui/skeleton';

export default function DashboardPage() {
  return (
    <div className="space-y-6">
      {/* KPI は最優先で表示 */}
      <Suspense fallback={<KPICardsSkeleton />}>
        <KPICards />
      </Suspense>

      <div className="grid grid-cols-2 gap-6">
        {/* チャートは独立してロード */}
        <Suspense fallback={<ChartSkeleton />}>
          <SalesChart />
        </Suspense>

        {/* 人気商品も独立してロード */}
        <Suspense fallback={<ListSkeleton />}>
          <TopProducts />
        </Suspense>
      </div>

      {/* 最近の注文は最後でよい */}
      <Suspense fallback={<TableSkeleton rows={5} />}>
        <RecentOrders />
      </Suspense>
    </div>
  );
}

// 各セクションが独立して fetch → レンダリングされるため、
// 最も速いものから順に表示される（ストリーミング）
```

### 4.3 error.tsx の詳細設計

`error.tsx` は React の `ErrorBoundary` をファイル規約で表現したものである。`'use client'` ディレクティブが必須で、Client Component として動作する。

```typescript
// app/dashboard/error.tsx — 詳細なエラーハンドリング
'use client';

import { useEffect } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';
import { Button } from '@/components/ui/button';
import Link from 'next/link';

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // エラーをモニタリングサービスに送信
    if (process.env.NODE_ENV === 'production') {
      // Sentry, Datadog, etc.
      reportError(error);
    }
    console.error('Dashboard error:', error);
  }, [error]);

  // エラーの種類に応じた表示分岐
  const isNetworkError = error.message.includes('fetch') ||
    error.message.includes('network');
  const isAuthError = error.message.includes('unauthorized') ||
    error.message.includes('401');

  if (isAuthError) {
    return (
      <div className="flex flex-col items-center justify-center p-16">
        <AlertTriangle className="h-12 w-12 text-yellow-500 mb-4" />
        <h2 className="text-xl font-bold mb-2">セッションが切れました</h2>
        <p className="text-gray-500 mb-6">もう一度ログインしてください。</p>
        <Link href="/login">
          <Button>ログインページへ</Button>
        </Link>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center p-16">
      <AlertTriangle className="h-12 w-12 text-red-500 mb-4" />
      <h2 className="text-xl font-bold mb-2">
        {isNetworkError
          ? 'ネットワークエラーが発生しました'
          : '予期しないエラーが発生しました'}
      </h2>
      <p className="text-gray-500 mb-2">
        {isNetworkError
          ? 'インターネット接続を確認してください。'
          : 'しばらく時間をおいてもう一度お試しください。'}
      </p>

      {/* 開発環境ではエラー詳細を表示 */}
      {process.env.NODE_ENV === 'development' && (
        <details className="mt-4 p-4 bg-red-50 border border-red-200 rounded max-w-lg w-full">
          <summary className="cursor-pointer text-red-700 font-mono text-sm flex items-center gap-2">
            <Bug className="h-4 w-4" />
            エラー詳細
          </summary>
          <pre className="mt-2 text-xs text-red-600 overflow-x-auto whitespace-pre-wrap">
            {error.message}
            {error.stack && `\n\n${error.stack}`}
          </pre>
        </details>
      )}

      {/* Error Digest（本番環境でのエラー追跡用） */}
      {error.digest && (
        <p className="text-xs text-gray-400 mt-2">
          Error ID: {error.digest}
        </p>
      )}

      <div className="flex gap-4 mt-6">
        <Button onClick={reset} variant="default">
          <RefreshCw className="h-4 w-4 mr-2" />
          再試行
        </Button>
        <Link href="/">
          <Button variant="outline">
            <Home className="h-4 w-4 mr-2" />
            ホームへ戻る
          </Button>
        </Link>
      </div>
    </div>
  );
}
```

### 4.4 global-error.tsx

`global-error.tsx` はルートレイアウト（`app/layout.tsx`）のエラーをキャッチするための特殊ファイルである。通常の `error.tsx` はレイアウトの子として配置されるため、レイアウト自身のエラーをキャッチできない。

```typescript
// app/global-error.tsx
'use client';

// global-error.tsx は独自の <html> と <body> を含む必要がある
// （RootLayout が壊れている可能性があるため）
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="ja">
      <body className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center p-8">
          <h1 className="text-4xl font-bold text-red-600 mb-4">
            重大なエラーが発生しました
          </h1>
          <p className="text-gray-600 mb-6">
            アプリケーションの起動中にエラーが発生しました。
          </p>
          {error.digest && (
            <p className="text-sm text-gray-400 mb-4">
              Error ID: {error.digest}
            </p>
          )}
          <button
            onClick={reset}
            className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
          >
            アプリケーションを再起動
          </button>
        </div>
      </body>
    </html>
  );
}
```

### 4.5 not-found.tsx の設計パターン

`not-found.tsx` は `notFound()` 関数が呼ばれたとき、またはマッチしないURLにアクセスしたときに表示されるUIである。

```typescript
// app/not-found.tsx — グローバル 404 ページ
import Link from 'next/link';
import { Search } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center">
      <h1 className="text-8xl font-bold text-gray-200 dark:text-gray-800">
        404
      </h1>
      <h2 className="text-2xl font-bold mt-4 mb-2">
        ページが見つかりません
      </h2>
      <p className="text-gray-500 mb-8 text-center max-w-md">
        お探しのページは移動または削除された可能性があります。
        URLが正しいかご確認ください。
      </p>

      {/* 検索ボックス */}
      <div className="relative w-full max-w-md mb-8">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          type="text"
          placeholder="サイト内を検索..."
          className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div className="flex gap-4">
        <Link
          href="/"
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition"
        >
          ホームへ戻る
        </Link>
        <Link
          href="/contact"
          className="px-6 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition"
        >
          お問い合わせ
        </Link>
      </div>
    </div>
  );
}

// app/users/[id]/not-found.tsx — ルートセグメント固有の 404
export default function UserNotFound() {
  return (
    <div className="text-center p-16">
      <h2 className="text-2xl font-bold mb-4">ユーザーが見つかりません</h2>
      <p className="text-gray-500 mb-6">
        指定されたユーザーは存在しないか、削除された可能性があります。
      </p>
      <Link
        href="/users"
        className="text-blue-500 hover:underline"
      >
        ユーザー一覧に戻る
      </Link>
    </div>
  );
}
```

---

## 5. パラレルルートとインターセプトルート

### 5.1 パラレルルートの概要

パラレルルートは、同じレイアウト内で複数のページを並列にレンダリングする機能である。`@slot` というディレクトリ命名規約を使い、レイアウトコンポーネントの props としてスロットを受け取る。

```
パラレルルートのディレクトリ構造:

  app/dashboard/
  ├── layout.tsx             ← children + analytics + activity を受け取る
  ├── page.tsx               ← children スロット（デフォルト）
  ├── @analytics/
  │   ├── page.tsx           ← analytics スロット
  │   ├── loading.tsx        ← analytics 専用のローディング
  │   └── error.tsx          ← analytics 専用のエラー
  ├── @activity/
  │   ├── page.tsx           ← activity スロット
  │   └── loading.tsx        ← activity 専用のローディング
  └── @notifications/
      ├── page.tsx           ← notifications スロット
      └── default.tsx        ← サブナビゲーション時のデフォルト表示
```

```typescript
// app/dashboard/layout.tsx — パラレルルートのレイアウト
export default function DashboardLayout({
  children,
  analytics,
  activity,
  notifications,
}: {
  children: React.ReactNode;
  analytics: React.ReactNode;
  activity: React.ReactNode;
  notifications: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-12 gap-6">
      {/* メインコンテンツ（8列） */}
      <div className="col-span-8 space-y-6">
        {children}

        {/* 分析チャート（独立ローディング） */}
        <section className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-bold mb-4">Analytics</h2>
          {analytics}
        </section>
      </div>

      {/* サイドバー（4列） */}
      <div className="col-span-4 space-y-6">
        {/* 通知パネル */}
        <section className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-bold mb-4">Notifications</h2>
          {notifications}
        </section>

        {/* アクティビティフィード */}
        <section className="bg-white rounded-lg shadow p-6">
          <h2 className="text-lg font-bold mb-4">Recent Activity</h2>
          {activity}
        </section>
      </div>
    </div>
  );
}
```

### 5.2 パラレルルートの利点

パラレルルートには以下の利点がある。

```
1. 独立したローディング/エラー状態
   → 各スロットが独自の loading.tsx と error.tsx を持てる
   → 一つのセクションのエラーが他のセクションに影響しない

2. 独立したデータフェッチ
   → 各スロットが独立してデータを取得・表示
   → ストリーミングにより、取得完了順に表示

3. 条件付きレンダリング
   → ユーザーの権限に応じて異なるスロットを表示可能

4. URL駆動の表示制御
   → URLパスに応じて各スロットの表示内容を切り替え可能
```

```typescript
// app/dashboard/@analytics/page.tsx — 独立したデータフェッチ
export default async function AnalyticsSlot() {
  // このデータフェッチは他のスロットとは独立して実行される
  const analyticsData = await getAnalytics();

  return (
    <div>
      <BarChart data={analyticsData.dailyVisits} />
      <div className="grid grid-cols-3 gap-4 mt-4">
        <StatCard label="Page Views" value={analyticsData.pageViews} />
        <StatCard label="Unique Visitors" value={analyticsData.uniqueVisitors} />
        <StatCard label="Bounce Rate" value={`${analyticsData.bounceRate}%`} />
      </div>
    </div>
  );
}

// app/dashboard/@analytics/loading.tsx — スロット専用のローディング
export default function AnalyticsLoading() {
  return (
    <div className="space-y-4">
      <div className="h-48 bg-gray-100 animate-pulse rounded" />
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map(i => (
          <div key={i} className="h-20 bg-gray-100 animate-pulse rounded" />
        ))}
      </div>
    </div>
  );
}

// app/dashboard/@analytics/error.tsx — スロット専用のエラー
'use client';
export default function AnalyticsError({ reset }: { error: Error; reset: () => void }) {
  return (
    <div className="text-center p-4 bg-red-50 rounded">
      <p className="text-red-600 mb-2">分析データの読み込みに失敗しました</p>
      <button onClick={reset} className="text-blue-500 underline">
        再試行
      </button>
    </div>
  );
}
```

### 5.3 default.tsx の役割

パラレルルートでサブナビゲーション時にスロットのURLがマッチしない場合、`default.tsx` がフォールバックとして表示される。`default.tsx` がない場合は404になる。

```typescript
// 問題のあるケース:
// app/dashboard/@notifications/page.tsx は /dashboard で表示される
// しかし /dashboard/settings に遷移すると
// @notifications スロットに対応する settings/page.tsx がない
// → default.tsx がないと404になる

// app/dashboard/@notifications/default.tsx
export default function NotificationsDefault() {
  // page.tsx と同じ内容を返すか、簡略版を返す
  return <NotificationsList />;
}

// ソフトナビゲーション vs ハードナビゲーション:
// - ソフトナビゲーション（Link クリック）: 前の状態が保持される
// - ハードナビゲーション（ページリロード、URL直接入力）: default.tsx が使われる
```

### 5.4 インターセプトルートの概要

インターセプトルートは、現在のレイアウトを維持しながら別のルートのコンテンツをモーダルやオーバーレイとして表示する機能である。Instagram のフィード上での画像表示のようなUXを実現できる。

```
インターセプトルートの記法:

  (.)   → 同じレベルのルートをインターセプト
  (..)  → 1つ上のレベルのルートをインターセプト
  (..)(..) → 2つ上のレベル
  (...) → ルート（app/）からのルートをインターセプト
```

```
実践例: 写真ギャラリー + モーダル

  app/
  ├── layout.tsx
  ├── feed/
  │   ├── page.tsx                     ← 写真フィード一覧
  │   └── @modal/
  │       ├── default.tsx              ← モーダルなし（空）
  │       └── (.)photo/[id]/
  │           └── page.tsx             ← モーダルで写真表示
  └── photo/
      └── [id]/
          └── page.tsx                 ← 写真の全画面表示（直接アクセス用）

動作:
  1. /feed にアクセス → フィード表示、モーダルなし
  2. フィード内の写真をクリック → URL が /photo/123 に変わるが
     実際は (.)photo/[id]/page.tsx がインターセプトし、
     フィードを背景に保ちつつモーダルで写真を表示
  3. /photo/123 に直接アクセス → photo/[id]/page.tsx の全画面表示
  4. モーダル表示中にリロード → 全画面表示に切り替わる
```

```typescript
// app/feed/page.tsx — 写真フィード
import Link from 'next/link';
import Image from 'next/image';

export default async function FeedPage() {
  const photos = await getPhotos();

  return (
    <div className="grid grid-cols-3 gap-1">
      {photos.map((photo) => (
        <Link key={photo.id} href={`/photo/${photo.id}`}>
          <Image
            src={photo.thumbnailUrl}
            alt={photo.title}
            width={300}
            height={300}
            className="w-full aspect-square object-cover hover:opacity-80 transition"
          />
        </Link>
      ))}
    </div>
  );
}

// app/feed/layout.tsx — モーダルスロット付きレイアウト
export default function FeedLayout({
  children,
  modal,
}: {
  children: React.ReactNode;
  modal: React.ReactNode;
}) {
  return (
    <>
      {children}
      {modal}
    </>
  );
}

// app/feed/@modal/default.tsx — モーダルなし
export default function Default() {
  return null;
}

// app/feed/@modal/(.)photo/[id]/page.tsx — モーダル表示
import { Modal } from '@/components/modal';

export default async function PhotoModal({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const photo = await getPhoto(id);

  return (
    <Modal>
      <Image
        src={photo.url}
        alt={photo.title}
        width={800}
        height={600}
        className="w-full"
      />
      <div className="p-4">
        <h2 className="text-xl font-bold">{photo.title}</h2>
        <p className="text-gray-500">{photo.description}</p>
      </div>
    </Modal>
  );
}

// app/photo/[id]/page.tsx — 全画面表示（直接アクセス）
export default async function PhotoPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const photo = await getPhoto(id);

  return (
    <div className="max-w-4xl mx-auto py-8">
      <Image
        src={photo.url}
        alt={photo.title}
        width={1200}
        height={800}
        className="w-full rounded-lg"
      />
      <h1 className="text-3xl font-bold mt-4">{photo.title}</h1>
      <p className="text-gray-500 mt-2">{photo.description}</p>
      <PhotoComments photoId={photo.id} />
    </div>
  );
}
```

```typescript
// components/modal.tsx — 汎用モーダルコンポーネント
'use client';

import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useRef } from 'react';
import { X } from 'lucide-react';

export function Modal({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const overlayRef = useRef<HTMLDivElement>(null);

  const onDismiss = useCallback(() => {
    router.back();
  }, [router]);

  // ESC キーでモーダルを閉じる
  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onDismiss();
    };
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [onDismiss]);

  // オーバーレイクリックで閉じる
  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === overlayRef.current) onDismiss();
  };

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center p-4"
      onClick={handleOverlayClick}
    >
      <div className="bg-white dark:bg-gray-900 rounded-lg max-w-3xl w-full max-h-[90vh] overflow-auto relative">
        <button
          onClick={onDismiss}
          className="absolute top-4 right-4 p-2 rounded-full bg-gray-100 hover:bg-gray-200 transition z-10"
        >
          <X className="h-5 w-5" />
        </button>
        {children}
      </div>
    </div>
  );
}
```

---

## 6. Middleware とルーティング制御

### 6.1 Middleware の基本

Next.js の Middleware は、リクエストが完了する前にコードを実行できる仕組みである。`middleware.ts` はプロジェクトルート（`app/` と同じ階層）に配置する。

```typescript
// middleware.ts（プロジェクトルートに配置）
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 1. 認証チェック
  const token = request.cookies.get('session-token')?.value;
  const protectedPaths = ['/dashboard', '/settings', '/admin'];

  if (protectedPaths.some(path => pathname.startsWith(path))) {
    if (!token) {
      const loginUrl = new URL('/login', request.url);
      loginUrl.searchParams.set('callbackUrl', pathname);
      return NextResponse.redirect(loginUrl);
    }
  }

  // 2. 国際化（i18n）リダイレクト
  const locale = request.headers.get('accept-language')?.split(',')[0]?.split('-')[0] ?? 'ja';
  const supportedLocales = ['ja', 'en'];
  const defaultLocale = 'ja';

  if (!pathname.startsWith('/_next') && !pathname.startsWith('/api')) {
    const pathnameLocale = supportedLocales.find(
      loc => pathname.startsWith(`/${loc}/`) || pathname === `/${loc}`
    );

    if (!pathnameLocale) {
      const detectedLocale = supportedLocales.includes(locale) ? locale : defaultLocale;
      return NextResponse.redirect(
        new URL(`/${detectedLocale}${pathname}`, request.url)
      );
    }
  }

  // 3. レスポンスヘッダーの追加
  const response = NextResponse.next();
  response.headers.set('x-request-id', crypto.randomUUID());
  response.headers.set('x-pathname', pathname);

  return response;
}

// Middleware を適用するパスの設定
export const config = {
  matcher: [
    // 静的ファイルと内部パスを除外
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};
```

### 6.2 Middleware の実践パターン

```typescript
// middleware.ts — 高度な Middleware パターン

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { jwtVerify } from 'jose';

// Rate Limiting（簡易版）
const rateLimit = new Map<string, { count: number; resetTime: number }>();

function checkRateLimit(ip: string, limit: number, windowMs: number): boolean {
  const now = Date.now();
  const record = rateLimit.get(ip);

  if (!record || now > record.resetTime) {
    rateLimit.set(ip, { count: 1, resetTime: now + windowMs });
    return true;
  }

  if (record.count >= limit) {
    return false;
  }

  record.count++;
  return true;
}

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // ---- API Rate Limiting ----
  if (pathname.startsWith('/api/')) {
    const ip = request.headers.get('x-forwarded-for') ?? 'unknown';
    const isAllowed = checkRateLimit(ip, 100, 60 * 1000); // 100 req/min

    if (!isAllowed) {
      return NextResponse.json(
        { error: 'Too many requests' },
        { status: 429, headers: { 'Retry-After': '60' } }
      );
    }
  }

  // ---- JWT 認証 ----
  if (pathname.startsWith('/dashboard') || pathname.startsWith('/api/protected')) {
    const token = request.cookies.get('auth-token')?.value;

    if (!token) {
      if (pathname.startsWith('/api/')) {
        return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
      }
      return NextResponse.redirect(new URL('/login', request.url));
    }

    try {
      const secret = new TextEncoder().encode(process.env.JWT_SECRET!);
      const { payload } = await jwtVerify(token, secret);

      // ---- RBAC（ロールベースアクセス制御） ----
      if (pathname.startsWith('/admin') && payload.role !== 'admin') {
        return NextResponse.redirect(new URL('/dashboard', request.url));
      }

      // リクエストヘッダーにユーザー情報を追加
      const response = NextResponse.next();
      response.headers.set('x-user-id', payload.sub as string);
      response.headers.set('x-user-role', payload.role as string);
      return response;
    } catch (error) {
      // トークン無効 → ログインページへ
      const response = NextResponse.redirect(new URL('/login', request.url));
      response.cookies.delete('auth-token');
      return response;
    }
  }

  // ---- A/Bテスト ----
  if (pathname === '/pricing') {
    const bucket = request.cookies.get('ab-test-pricing')?.value;
    if (!bucket) {
      const newBucket = Math.random() > 0.5 ? 'A' : 'B';
      const response = NextResponse.rewrite(
        new URL(`/pricing/${newBucket.toLowerCase()}`, request.url)
      );
      response.cookies.set('ab-test-pricing', newBucket, {
        maxAge: 60 * 60 * 24 * 30, // 30日
      });
      return response;
    }
    return NextResponse.rewrite(
      new URL(`/pricing/${bucket.toLowerCase()}`, request.url)
    );
  }

  // ---- リダイレクト（旧URL対応） ----
  const redirects: Record<string, string> = {
    '/blog': '/articles',
    '/docs/getting-started': '/docs/introduction',
    '/help': '/support',
  };

  if (redirects[pathname]) {
    return NextResponse.redirect(
      new URL(redirects[pathname], request.url),
      { status: 301 }
    );
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    '/((?!_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};
```

---

## 7. Remix / React Router v7 のファイルベースルーティング

### 7.1 Remix v2 のルーティング規約

Remix は Next.js とは異なるファイル規約を採用している。フラットファイル構造（flat routes）を基本とし、ドット（`.`）区切りでネストを表現する。

```
Remix v2 のファイル構造:

  app/routes/
  ├── _index.tsx                    → /
  ├── about.tsx                     → /about
  ├── blog._index.tsx               → /blog
  ├── blog.$slug.tsx                → /blog/:slug
  ├── users._index.tsx              → /users
  ├── users.$id.tsx                 → /users/:id
  ├── users.$id_.edit.tsx           → /users/:id/edit
  ├── dashboard.tsx                 → /dashboard のレイアウト
  ├── dashboard._index.tsx          → /dashboard
  ├── dashboard.settings.tsx        → /dashboard/settings
  ├── dashboard.analytics.tsx       → /dashboard/analytics
  ├── $.tsx                         → キャッチオール（404）
  ├── _auth.tsx                     → 認証レイアウト（URLに含まれない）
  ├── _auth.login.tsx               → /login
  ├── _auth.register.tsx            → /register
  └── files.$.tsx                   → /files/*（キャッチオール）

命名規約:
  .        → ネストの区切り（URLの / に対応）
  $param   → 動的セグメント
  _index   → インデックスルート
  _prefix  → パスレスレイアウト（URLに含まれない）
  $        → キャッチオール
  name_    → トレイリングアンダースコア（レイアウトのネストから離脱）
```

### 7.2 Remix のルートコンポーネント

Remix では、各ルートファイルが `loader`（データ取得）、`action`（データ変更）、`default export`（UI）を一つのファイルに含む。

```typescript
// app/routes/users.$id.tsx — Remix のルートモジュール
import type { LoaderFunctionArgs, ActionFunctionArgs, MetaFunction } from '@remix-run/node';
import { json, redirect } from '@remix-run/node';
import { useLoaderData, useActionData, Form } from '@remix-run/react';
import { getUser, updateUser } from '~/models/user.server';

// ---- loader: サーバーサイドのデータ取得 ----
export async function loader({ params, request }: LoaderFunctionArgs) {
  const user = await getUser(params.id!);
  if (!user) {
    throw new Response('Not Found', { status: 404 });
  }
  return json({ user });
}

// ---- action: フォーム送信の処理 ----
export async function action({ params, request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const intent = formData.get('intent');

  if (intent === 'update') {
    const name = formData.get('name') as string;
    const email = formData.get('email') as string;

    const errors: Record<string, string> = {};
    if (!name) errors.name = '名前は必須です';
    if (!email) errors.email = 'メールは必須です';

    if (Object.keys(errors).length > 0) {
      return json({ errors }, { status: 400 });
    }

    await updateUser(params.id!, { name, email });
    return redirect(`/users/${params.id}`);
  }

  if (intent === 'delete') {
    await deleteUser(params.id!);
    return redirect('/users');
  }

  return json({ errors: { form: '不明なアクション' } }, { status: 400 });
}

// ---- meta: メタデータの定義 ----
export const meta: MetaFunction<typeof loader> = ({ data }) => {
  return [
    { title: data?.user.name ?? 'User Not Found' },
    { name: 'description', content: `${data?.user.name}のプロフィール` },
  ];
};

// ---- ErrorBoundary ----
export function ErrorBoundary() {
  return (
    <div className="text-center p-8">
      <h2 className="text-xl font-bold text-red-600">エラーが発生しました</h2>
    </div>
  );
}

// ---- UI コンポーネント ----
export default function UserPage() {
  const { user } = useLoaderData<typeof loader>();
  const actionData = useActionData<typeof action>();

  return (
    <div className="max-w-2xl mx-auto py-8">
      <h1 className="text-2xl font-bold mb-6">{user.name}</h1>

      <Form method="post" className="space-y-4">
        <input type="hidden" name="intent" value="update" />

        <div>
          <label htmlFor="name" className="block text-sm font-medium">名前</label>
          <input
            id="name"
            name="name"
            defaultValue={user.name}
            className="mt-1 block w-full border rounded px-3 py-2"
          />
          {actionData?.errors?.name && (
            <p className="text-red-500 text-sm mt-1">{actionData.errors.name}</p>
          )}
        </div>

        <div>
          <label htmlFor="email" className="block text-sm font-medium">メール</label>
          <input
            id="email"
            name="email"
            type="email"
            defaultValue={user.email}
            className="mt-1 block w-full border rounded px-3 py-2"
          />
          {actionData?.errors?.email && (
            <p className="text-red-500 text-sm mt-1">{actionData.errors.email}</p>
          )}
        </div>

        <button
          type="submit"
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          更新
        </button>
      </Form>
    </div>
  );
}
```

### 7.3 Next.js App Router と Remix の比較

| 機能 | Next.js App Router | Remix v2 |
|------|-------------------|----------|
| ルーティング方式 | ネストされたディレクトリ | フラットファイル（ドット区切り） |
| データフェッチ | Server Component での async/await | loader 関数 |
| データ変更 | Server Actions | action 関数 + Form |
| レイアウト | layout.tsx（ディレクトリ） | 親ルートの default export + Outlet |
| ローディング | loading.tsx（自動Suspense） | useNavigation().state |
| エラー処理 | error.tsx（自動ErrorBoundary） | ErrorBoundary export |
| メタデータ | generateMetadata / metadata | meta 関数 |
| ストリーミング | React Suspense + Server Components | defer + Await |
| レンダリング | SSR / SSG / ISR | SSR（+ クライアントキャッシュ） |
| ファイル配置 | コロケーション（page.tsx以外無視） | routes/ 内のみ |

---

## 8. 他フレームワークのファイルベースルーティング

### 8.1 SvelteKit

SvelteKit は Next.js App Router に近い規約を持つが、ファイル名にプレフィックスとして `+` を使う。

```
SvelteKit のディレクトリ構造:

  src/routes/
  ├── +page.svelte              → / のページ
  ├── +layout.svelte            → ルートレイアウト
  ├── +error.svelte             → エラーUI
  ├── +page.server.ts           → サーバーサイドの load 関数
  ├── +layout.server.ts         → レイアウトのサーバーサイド load
  ├── about/
  │   └── +page.svelte          → /about
  ├── blog/
  │   ├── +page.svelte          → /blog
  │   ├── +page.server.ts       → /blog のデータフェッチ
  │   └── [slug]/
  │       ├── +page.svelte      → /blog/:slug
  │       └── +page.server.ts   → /blog/:slug のデータフェッチ
  ├── (auth)/                   → ルートグループ（URLに含まれない）
  │   ├── +layout.svelte        → 認証ページ共通レイアウト
  │   ├── login/
  │   │   └── +page.svelte      → /login
  │   └── register/
  │       └── +page.svelte      → /register
  └── api/
      └── users/
          └── +server.ts        → API エンドポイント
```

```svelte
<!-- src/routes/blog/[slug]/+page.svelte -->
<script>
  export let data;  // +page.server.ts の load から
</script>

<article>
  <h1>{data.post.title}</h1>
  <div>{@html data.post.contentHtml}</div>
</article>
```

```typescript
// src/routes/blog/[slug]/+page.server.ts
import { error } from '@sveltejs/kit';
import type { PageServerLoad } from './$types';

export const load: PageServerLoad = async ({ params }) => {
  const post = await getPost(params.slug);

  if (!post) {
    throw error(404, 'Post not found');
  }

  return { post };
};
```

### 8.2 Nuxt.js 3

Nuxt.js 3 は Vue.js ベースのフレームワークで、`pages/` ディレクトリにファイルを配置する。

```
Nuxt.js 3 のディレクトリ構造:

  pages/
  ├── index.vue                 → /
  ├── about.vue                 → /about
  ├── users/
  │   ├── index.vue             → /users
  │   └── [id].vue              → /users/:id
  ├── blog/
  │   ├── index.vue             → /blog
  │   └── [...slug].vue         → /blog/* (キャッチオール)
  └── [[slug]].vue              → オプショナル動的ルート

  layouts/
  ├── default.vue               → デフォルトレイアウト
  ├── auth.vue                  → 認証ページ用レイアウト
  └── admin.vue                 → 管理画面用レイアウト
```

```vue
<!-- pages/users/[id].vue -->
<script setup lang="ts">
const route = useRoute();
const { data: user } = await useFetch(`/api/users/${route.params.id}`);

if (!user.value) {
  throw createError({
    statusCode: 404,
    statusMessage: 'User Not Found',
  });
}

useHead({
  title: user.value.name,
});
</script>

<template>
  <div class="max-w-4xl mx-auto py-8">
    <h1 class="text-2xl font-bold">{{ user.name }}</h1>
    <p class="text-gray-500">{{ user.email }}</p>
  </div>
</template>
```

### 8.3 Astro

Astro は コンテンツ重視の静的サイトジェネレーターで、`src/pages/` ディレクトリにファイルを配置する。

```
Astro のディレクトリ構造:

  src/pages/
  ├── index.astro               → /
  ├── about.astro               → /about
  ├── blog/
  │   ├── index.astro           → /blog
  │   └── [slug].astro          → /blog/:slug
  ├── [...slug].astro           → キャッチオール
  └── api/
      └── users.ts              → API エンドポイント（SSR モード時）
```

```astro
---
// src/pages/blog/[slug].astro
import Layout from '../../layouts/Layout.astro';
import { getEntry } from 'astro:content';

const { slug } = Astro.params;
const post = await getEntry('blog', slug);

if (!post) {
  return Astro.redirect('/404');
}

const { Content } = await post.render();
---

<Layout title={post.data.title}>
  <article class="prose">
    <h1>{post.data.title}</h1>
    <Content />
  </article>
</Layout>
```

### 8.4 フレームワーク間の特殊ファイル比較

| 機能 | Next.js App Router | SvelteKit | Nuxt.js 3 | Remix v2 |
|------|-------------------|-----------|-----------|----------|
| ページ | `page.tsx` | `+page.svelte` | `index.vue` / `name.vue` | `route.tsx` |
| レイアウト | `layout.tsx` | `+layout.svelte` | `layouts/name.vue` | 親ルート + `<Outlet />` |
| エラー | `error.tsx` | `+error.svelte` | `error.vue` | `ErrorBoundary` export |
| ローディング | `loading.tsx` | N/A（手動 Suspense） | N/A（`<NuxtLoadingIndicator>`） | `useNavigation()` |
| サーバーデータ | Server Component | `+page.server.ts` | `useFetch()` | `loader` |
| フォーム処理 | Server Actions | `+page.server.ts` (actions) | `useFetch()` + API | `action` + `<Form>` |
| 404 | `not-found.tsx` | `+error.svelte` (404) | `error.vue` (404) | `throw Response(404)` |
| API Route | `route.ts` | `+server.ts` | `server/api/` | `resource route` |
| ミドルウェア | `middleware.ts` | `hooks.server.ts` | `server/middleware/` | N/A |

---

## 9. 実践的なプロジェクトのディレクトリ設計

### 9.1 SaaS アプリケーションの設計例

実際の SaaS アプリケーションを想定したディレクトリ設計の完全な例を示す。

```
app/
├── layout.tsx                           ← ルートレイアウト
├── page.tsx                             ← ランディングページ (/)
├── globals.css
├── favicon.ico
├── opengraph-image.png
├── sitemap.ts
├── robots.ts
│
├── (marketing)/                         ← マーケティングサイト
│   ├── layout.tsx                       ← ヘッダー + フッター
│   ├── about/page.tsx                   ← /about
│   ├── pricing/page.tsx                 ← /pricing
│   ├── blog/
│   │   ├── page.tsx                     ← /blog（記事一覧）
│   │   └── [slug]/page.tsx              ← /blog/:slug
│   ├── changelog/page.tsx               ← /changelog
│   ├── contact/page.tsx                 ← /contact
│   ├── legal/
│   │   ├── privacy/page.tsx             ← /legal/privacy
│   │   └── terms/page.tsx               ← /legal/terms
│   └── docs/
│       ├── layout.tsx                   ← ドキュメント用サイドバー
│       └── [[...slug]]/page.tsx         ← /docs/*
│
├── (auth)/                              ← 認証フロー
│   ├── layout.tsx                       ← センタリングレイアウト
│   ├── login/page.tsx                   ← /login
│   ├── register/page.tsx                ← /register
│   ├── forgot-password/page.tsx         ← /forgot-password
│   ├── reset-password/page.tsx          ← /reset-password
│   ├── verify-email/page.tsx            ← /verify-email
│   └── sso/
│       └── [provider]/page.tsx          ← /sso/:provider (google, github等)
│
├── (app)/                               ← アプリケーション本体
│   ├── layout.tsx                       ← 認証チェック + サイドバー + ヘッダー
│   ├── onboarding/
│   │   ├── page.tsx                     ← /onboarding（初回セットアップ）
│   │   └── [step]/page.tsx              ← /onboarding/:step
│   ├── dashboard/
│   │   ├── page.tsx                     ← /dashboard
│   │   ├── loading.tsx                  ← ダッシュボードのローディング
│   │   ├── error.tsx                    ← ダッシュボードのエラー
│   │   ├── @analytics/
│   │   │   ├── page.tsx
│   │   │   ├── loading.tsx
│   │   │   └── default.tsx
│   │   └── @activity/
│   │       ├── page.tsx
│   │       ├── loading.tsx
│   │       └── default.tsx
│   ├── projects/
│   │   ├── page.tsx                     ← /projects（一覧）
│   │   ├── loading.tsx
│   │   ├── new/page.tsx                 ← /projects/new（新規作成）
│   │   └── [projectId]/
│   │       ├── layout.tsx               ← プロジェクトコンテキスト
│   │       ├── page.tsx                 ← /projects/:id（概要）
│   │       ├── settings/page.tsx        ← /projects/:id/settings
│   │       ├── members/page.tsx         ← /projects/:id/members
│   │       ├── tasks/
│   │       │   ├── page.tsx             ← /projects/:id/tasks
│   │       │   └── [taskId]/page.tsx    ← /projects/:id/tasks/:taskId
│   │       └── analytics/page.tsx       ← /projects/:id/analytics
│   ├── settings/
│   │   ├── layout.tsx                   ← 設定画面のサブナビ
│   │   ├── page.tsx                     ← /settings（一般設定）
│   │   ├── profile/page.tsx             ← /settings/profile
│   │   ├── billing/page.tsx             ← /settings/billing
│   │   ├── team/page.tsx                ← /settings/team
│   │   ├── integrations/page.tsx        ← /settings/integrations
│   │   ├── notifications/page.tsx       ← /settings/notifications
│   │   ├── security/page.tsx            ← /settings/security
│   │   └── api-keys/page.tsx            ← /settings/api-keys
│   └── admin/                           ← 管理者専用
│       ├── layout.tsx                   ← 管理者権限チェック
│       ├── page.tsx                     ← /admin
│       ├── users/
│       │   ├── page.tsx                 ← /admin/users
│       │   └── [id]/page.tsx            ← /admin/users/:id
│       └── system/page.tsx              ← /admin/system
│
├── api/                                 ← API Routes
│   ├── auth/
│   │   ├── [...nextauth]/route.ts       ← NextAuth.js
│   │   └── session/route.ts             ← セッション確認
│   ├── users/
│   │   ├── route.ts                     ← GET/POST /api/users
│   │   └── [id]/route.ts               ← GET/PUT/DELETE /api/users/:id
│   ├── projects/
│   │   ├── route.ts                     ← GET/POST /api/projects
│   │   └── [id]/
│   │       ├── route.ts                 ← GET/PUT/DELETE /api/projects/:id
│   │       └── tasks/route.ts           ← GET/POST /api/projects/:id/tasks
│   ├── webhooks/
│   │   ├── stripe/route.ts              ← Stripe Webhook
│   │   └── github/route.ts              ← GitHub Webhook
│   └── upload/route.ts                  ← ファイルアップロード
│
└── _components/                         ← ルートに含まれないコンポーネント
    ├── providers.tsx                     ← グローバルプロバイダー
    └── analytics.tsx                    ← アナリティクス
```

### 9.2 コロケーションパターン

Next.js App Router では、`page.tsx` がないディレクトリはルートとして認識されないため、ページに関連するコンポーネントを同じディレクトリに配置できる（コロケーション）。

```
推奨: コロケーションパターン

  app/projects/[projectId]/
  ├── page.tsx                    ← ページコンポーネント
  ├── loading.tsx                 ← ローディング
  ├── error.tsx                   ← エラー
  ├── _components/                ← ページ専用コンポーネント
  │   ├── project-header.tsx
  │   ├── project-stats.tsx
  │   ├── project-timeline.tsx
  │   └── project-members.tsx
  ├── _hooks/                     ← ページ専用フック
  │   ├── use-project.ts
  │   └── use-project-tasks.ts
  ├── _lib/                       ← ページ専用ユーティリティ
  │   ├── queries.ts
  │   └── actions.ts
  └── _types/                     ← ページ専用型定義
      └── index.ts

注意:
  - _（アンダースコア）プレフィックスは慣習であり、
    Next.js のルーティングには影響しない
  - page.tsx がないディレクトリはそもそもルートにならない
  - ただし、ディレクトリ名が page, layout, loading, error,
    not-found, route, template, default のいずれかの場合は
    特殊ファイルとして認識される
```

```typescript
// app/projects/[projectId]/page.tsx
// コロケーションされたコンポーネントをインポート
import { ProjectHeader } from './_components/project-header';
import { ProjectStats } from './_components/project-stats';
import { ProjectTimeline } from './_components/project-timeline';
import { getProject } from './_lib/queries';

export default async function ProjectPage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const { projectId } = await params;
  const project = await getProject(projectId);

  if (!project) notFound();

  return (
    <div>
      <ProjectHeader project={project} />
      <ProjectStats project={project} />
      <ProjectTimeline projectId={project.id} />
    </div>
  );
}
```

### 9.3 Private Folders（プライベートフォルダ）

Next.js ではアンダースコア `_` プレフィックスを付けたフォルダは、ルーティングの対象外となるプライベートフォルダとして扱える。

```
app/
├── _components/            ← ルーティング対象外
│   ├── header.tsx
│   └── footer.tsx
├── _lib/                   ← ルーティング対象外
│   ├── db.ts
│   └── auth.ts
├── _utils/                 ← ルーティング対象外
│   └── format.ts
├── page.tsx
└── dashboard/
    ├── page.tsx
    └── _components/        ← ルーティング対象外
        └── chart.tsx
```

---

## 10. Pages Router から App Router への移行

### 10.1 移行戦略

Next.js Pages Router（`pages/` ディレクトリ）から App Router（`app/` ディレクトリ）への移行は、段階的に行うことが推奨される。両方のルーターは共存できるため、ページ単位で移行を進められる。

```
段階的移行の手順:

  1. app/ ディレクトリを作成し、layout.tsx を配置
  2. ページを一つずつ pages/ から app/ に移動
  3. 各ページで以下を変換:
     - getServerSideProps → async Server Component
     - getStaticProps → async Server Component + generateStaticParams
     - getStaticPaths → generateStaticParams
     - useRouter (next/router) → useRouter (next/navigation)
     - Head → metadata export
  4. _app.tsx のプロバイダーを app/layout.tsx に移行
  5. _document.tsx の設定を app/layout.tsx に移行
  6. API Routes はそのまま pages/api/ に残すか、app/api/ に移行
```

```typescript
// ---- 移行前: pages/users/[id].tsx ----
import { GetServerSideProps } from 'next';
import Head from 'next/head';
import { useRouter } from 'next/router';

interface Props {
  user: User;
}

export const getServerSideProps: GetServerSideProps<Props> = async (context) => {
  const { id } = context.params!;
  const user = await getUser(id as string);

  if (!user) {
    return { notFound: true };
  }

  return { props: { user } };
};

export default function UserPage({ user }: Props) {
  const router = useRouter();

  return (
    <>
      <Head>
        <title>{user.name} | My App</title>
        <meta name="description" content={`${user.name}'s profile`} />
      </Head>
      <div>
        <h1>{user.name}</h1>
        <button onClick={() => router.push('/users')}>
          Back to Users
        </button>
      </div>
    </>
  );
}

// ---- 移行後: app/users/[id]/page.tsx ----
import { notFound } from 'next/navigation';
import type { Metadata } from 'next';

interface UserPageProps {
  params: Promise<{ id: string }>;
}

export async function generateMetadata({ params }: UserPageProps): Promise<Metadata> {
  const { id } = await params;
  const user = await getUser(id);
  return {
    title: user?.name ?? 'User Not Found',
    description: user ? `${user.name}'s profile` : undefined,
  };
}

export default async function UserPage({ params }: UserPageProps) {
  const { id } = await params;
  const user = await getUser(id);

  if (!user) {
    notFound();
  }

  return (
    <div>
      <h1>{user.name}</h1>
      <BackButton />  {/* Client Component に分離 */}
    </div>
  );
}

// app/users/[id]/_components/back-button.tsx
'use client';
import { useRouter } from 'next/navigation';

export function BackButton() {
  const router = useRouter();
  return (
    <button onClick={() => router.push('/users')}>
      Back to Users
    </button>
  );
}
```

### 10.2 移行時の主な変更点

| 項目 | Pages Router | App Router |
|------|-------------|-----------|
| データ取得 | `getServerSideProps` / `getStaticProps` | `async` Server Component |
| 静的パス生成 | `getStaticPaths` | `generateStaticParams` |
| メタデータ | `<Head>` コンポーネント | `metadata` export / `generateMetadata` |
| ルーター | `useRouter` (next/router) | `useRouter` (next/navigation) |
| リダイレクト | `getServerSideProps` で redirect | `redirect()` 関数 |
| 404 | `{ notFound: true }` | `notFound()` 関数 |
| レイアウト | `_app.tsx` + `_document.tsx` | `layout.tsx` |
| API Route | `pages/api/route.ts` | `app/api/route/route.ts` |
| クライアント状態 | デフォルト（Client Component） | `'use client'` 明示必要 |
| ストリーミング | 不可 | `<Suspense>` / `loading.tsx` |

---

## 11. トラブルシューティング

### 11.1 よくある問題と解決策

```
問題1: ルートが認識されない（404になる）
  原因: page.tsx が配置されていない、またはファイル名が間違っている
  解決策:
    - ディレクトリ内に page.tsx（小文字）が存在するか確認
    - Page.tsx や page.jsx ではないか確認
    - TypeScript の場合は page.tsx、JavaScript の場合は page.jsx
    - page.tsx が default export を持っているか確認

問題2: layout.tsx のエラーがキャッチされない
  原因: error.tsx は layout.tsx の子なので、layout のエラーをキャッチできない
  解決策:
    - 親セグメントに error.tsx を配置
    - ルートレイアウトの場合は global-error.tsx を配置

問題3: error.tsx が動作しない
  原因: 'use client' ディレクティブがない
  解決策:
    - error.tsx の先頭に 'use client' を必ず追加
    - global-error.tsx も同様

問題4: loading.tsx が表示されない
  原因: ページが Server Component でない、または async でない
  解決策:
    - page.tsx が async function であることを確認
    - Client Component ('use client') の場合、loading.tsx は初回のみ動作
    - Suspense を明示的に使用する

問題5: パラレルルートで 404 が表示される
  原因: サブナビゲーション時にスロットのURLがマッチしない
  解決策:
    - 各スロットに default.tsx を配置
    - ソフトナビゲーション時は前の状態が保持されるが、
      ハードナビゲーションでは default.tsx が必要

問題6: route.ts と page.tsx が同じディレクトリにある
  原因: 同じルートセグメントに page.tsx と route.ts は共存不可
  解決策:
    - API Route は api/ ディレクトリに移動
    - または page.tsx を別のディレクトリに配置

問題7: searchParams が undefined になる
  原因: Next.js 15+ で searchParams が Promise になった
  解決策:
    - const { q } = await searchParams; のように await する
    - TypeScript の型定義も Promise<...> に更新
```

### 11.2 デバッグ手法

```typescript
// ルーティングのデバッグ方法

// 1. 現在のルート情報の確認（Client Component）
'use client';
import { usePathname, useSearchParams, useParams } from 'next/navigation';

function DebugRouting() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const params = useParams();

  if (process.env.NODE_ENV !== 'development') return null;

  return (
    <div className="fixed bottom-4 right-4 p-4 bg-black text-green-400 font-mono text-xs rounded-lg max-w-md z-50">
      <div>pathname: {pathname}</div>
      <div>searchParams: {searchParams.toString()}</div>
      <div>params: {JSON.stringify(params)}</div>
    </div>
  );
}

// 2. Server Component でのログ
export default async function Page({
  params,
  searchParams,
}: {
  params: Promise<{ id: string }>;
  searchParams: Promise<{ q?: string }>;
}) {
  const resolvedParams = await params;
  const resolvedSearch = await searchParams;

  // サーバーログに出力される
  console.log('[Page] params:', resolvedParams);
  console.log('[Page] searchParams:', resolvedSearch);

  // ...
}

// 3. Middleware でのログ
export function middleware(request: NextRequest) {
  console.log('[Middleware]', request.method, request.nextUrl.pathname);
  return NextResponse.next();
}
```

### 11.3 パフォーマンスの問題

```
問題: ページの初回ロードが遅い
  確認事項:
    1. データフェッチがウォーターフォールになっていないか
       → Promise.all() で並列化、または Suspense で分割
    2. 'use client' の範囲が広すぎないか
       → Server Component を最大限活用し、Client Component を最小化
    3. generateStaticParams を活用しているか
       → 頻繁にアクセスされるページは事前生成
    4. revalidate が適切に設定されているか
       → 不要な再フェッチを避ける

問題: ページ遷移が遅い
  確認事項:
    1. Link コンポーネントの prefetch が無効になっていないか
       → prefetch={false} を不要に設定していないか確認
    2. レイアウトで重い処理をしていないか
       → layout.tsx は再レンダリングされないが、
         template.tsx は毎回実行される
    3. Suspense バウンダリが適切か
       → 大きなコンポーネントを Suspense で分割
```

```typescript
// パフォーマンス最適化: ウォーターフォールの回避

// NG: ウォーターフォール（直列実行）
export default async function DashboardPage() {
  const user = await getUser();           // 1. まずユーザーを取得
  const projects = await getProjects();   // 2. 次にプロジェクトを取得（待機）
  const notifications = await getNotifs(); // 3. 最後に通知を取得（待機）

  return (/* ... */);
}

// OK: 並列実行
export default async function DashboardPage() {
  const [user, projects, notifications] = await Promise.all([
    getUser(),
    getProjects(),
    getNotifications(),
  ]);

  return (/* ... */);
}

// BEST: Suspense で段階的表示
export default async function DashboardPage() {
  const user = await getUser(); // 軽い処理はすぐに表示

  return (
    <div>
      <UserHeader user={user} />

      <Suspense fallback={<ProjectsSkeleton />}>
        <ProjectsList />  {/* 独立してfetch */}
      </Suspense>

      <Suspense fallback={<NotificationsSkeleton />}>
        <NotificationsFeed />  {/* 独立してfetch */}
      </Suspense>
    </div>
  );
}
```

---

## 12. アンチパターンと回避策

### 12.1 よくあるアンチパターン

```typescript
// ---- アンチパターン 1: page.tsx を不必要に Client Component にする ----

// NG: ページ全体を Client Component に
'use client';
export default function UsersPage() {
  const [users, setUsers] = useState([]);
  useEffect(() => {
    fetch('/api/users').then(r => r.json()).then(setUsers);
  }, []);
  return <UserList users={users} />;
}

// OK: Server Component + Client Component の分離
// page.tsx（Server Component）
export default async function UsersPage() {
  const users = await getUsers(); // サーバーで直接取得
  return <UserList users={users} />;
}

// _components/user-list.tsx（Client Component、インタラクティブ部分のみ）
'use client';
export function UserList({ users }: { users: User[] }) {
  const [filter, setFilter] = useState('');
  const filtered = users.filter(u => u.name.includes(filter));
  return (
    <div>
      <input value={filter} onChange={e => setFilter(e.target.value)} />
      {filtered.map(user => <UserCard key={user.id} user={user} />)}
    </div>
  );
}
```

```typescript
// ---- アンチパターン 2: layout.tsx でデータを props で渡そうとする ----

// NG: layout.tsx から children にデータを渡すことはできない
export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await getUser();
  // children に user を渡す方法がない!
  return (
    <div>
      <Sidebar user={user} />
      {children}  {/* user を渡せない */}
    </div>
  );
}

// OK: 共有コンテキストまたは個別のデータフェッチ
// 方法1: React Context + Client Component Provider
// layout.tsx
export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const user = await getUser();
  return (
    <UserProvider initialUser={user}>
      <Sidebar />
      {children}
    </UserProvider>
  );
}

// 方法2: 各 page.tsx で個別にデータフェッチ（推奨）
// Next.js はデフォルトで fetch を deduplicate するため、
// 同じリクエストは1回しか実行されない
```

```typescript
// ---- アンチパターン 3: 深すぎるディレクトリ構造 ----

// NG: 深すぎるネスト
// app/dashboard/settings/account/profile/edit/confirm/page.tsx
// → /dashboard/settings/account/profile/edit/confirm

// OK: ルートグループとフラットな構造を活用
// app/(app)/settings/page.tsx         → /settings
// app/(app)/settings/profile/page.tsx → /settings/profile
// 深さは3-4レベルまでに抑える
```

```typescript
// ---- アンチパターン 4: API Route の濫用 ----

// NG: Server Component で直接取得できるのに API Route 経由
// app/api/users/route.ts
export async function GET() {
  const users = await db.user.findMany();
  return NextResponse.json(users);
}

// app/users/page.tsx
export default async function UsersPage() {
  // わざわざ API Route を呼ぶ必要はない
  const res = await fetch('http://localhost:3000/api/users');
  const users = await res.json();
  return <UserList users={users} />;
}

// OK: Server Component でデータベースに直接アクセス
export default async function UsersPage() {
  const users = await db.user.findMany();
  return <UserList users={users} />;
}

// API Route は以下の場合に使用:
// - 外部サービスからの Webhook
// - クライアントからの fetch（Client Component）
// - 外部APIとしての公開
// - Cron Job のエンドポイント
```

```typescript
// ---- アンチパターン 5: generateStaticParams の不適切な使用 ----

// NG: 全レコードを事前生成しようとする
export async function generateStaticParams() {
  // 100万件のユーザーを全て事前生成 → ビルド時間が膨大に
  const users = await db.user.findMany();
  return users.map(u => ({ id: u.id }));
}

// OK: アクセス頻度の高いページのみ事前生成
export async function generateStaticParams() {
  // 上位100件のみ事前生成、残りはオンデマンド
  const topUsers = await db.user.findMany({
    orderBy: { viewCount: 'desc' },
    take: 100,
    select: { id: true },
  });
  return topUsers.map(u => ({ id: u.id }));
}

// dynamicParams = true（デフォルト）により、
// 事前生成されていないパラメータはオンデマンドで生成される
```

### 12.2 セキュリティ上の注意点

```typescript
// 1. 動的ルートパラメータのバリデーション
// パラメータは常にユーザー入力として扱い、バリデーションする

// NG: パラメータを信頼してそのまま使用
export default async function UserPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // SQL インジェクションのリスク（ORMを使わない場合）
  const user = await sql`SELECT * FROM users WHERE id = ${id}`;
  return <div>{user.name}</div>;
}

// OK: バリデーション + Parameterized Query
export default async function UserPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;

  // UUID バリデーション
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  if (!uuidRegex.test(id)) {
    notFound();
  }

  // ORM の使用（パラメータ化されたクエリ）
  const user = await db.user.findUnique({ where: { id } });
  if (!user) notFound();

  return <div>{user.name}</div>;
}

// 2. Server Component からの機密情報漏洩防止
// Server Component のレンダリング結果はクライアントに送信されるため、
// 機密情報をそのまま含めてはいけない

// NG: 機密情報をクライアントに送信
export default async function AdminPage() {
  const config = await getSystemConfig();
  return (
    <div>
      {/* DB接続文字列がクライアントに送信される！ */}
      <pre>{JSON.stringify(config, null, 2)}</pre>
    </div>
  );
}

// OK: 必要な情報のみを選別
export default async function AdminPage() {
  const config = await getSystemConfig();
  return (
    <div>
      <p>App Version: {config.version}</p>
      <p>Environment: {config.environment}</p>
      {/* DB接続文字列などの機密情報は含めない */}
    </div>
  );
}

// 3. middleware.ts での認証
// Server Component での認証チェックだけでなく、
// middleware.ts でも事前チェックを行う（二重チェック）
```

---

## 13. ベストプラクティスチェックリスト

### 13.1 ディレクトリ設計

- [ ] ルートグループ `(name)` を使い、マーケティング・アプリ・認証でレイアウトを分離している
- [ ] ディレクトリの深さは4レベル以内に抑えている
- [ ] コロケーション（`_components/` 等）を活用し、関連ファイルを近くに配置している
- [ ] `page.tsx` のない中間ディレクトリはレイアウト用途のみに使用している
- [ ] Private Folders（`_` プレフィックス）で非ルーティングファイルを明示している

### 13.2 レイアウト設計

- [ ] ルートレイアウトに `<html>` と `<body>` タグを配置している
- [ ] 共有プロバイダー（Theme、Auth、Query）はルートレイアウトに配置している
- [ ] 認証チェックは対応するルートグループの layout.tsx で行っている
- [ ] `template.tsx` は本当に必要な場合にのみ使用している
- [ ] レイアウトでの重い処理を避け、パフォーマンスを維持している

### 13.3 データフェッチ

- [ ] Server Component でデータを直接取得し、API Route 経由を避けている
- [ ] `Promise.all()` や `Suspense` でウォーターフォールを回避している
- [ ] `generateStaticParams` で頻繁にアクセスされるページを事前生成している
- [ ] `revalidate` を適切に設定し、不要な再フェッチを避けている
- [ ] `dynamicParams` の設定を意図的に行っている

### 13.4 エラーハンドリング

- [ ] 各主要セクションに `error.tsx` を配置している
- [ ] `error.tsx` に `'use client'` ディレクティブを付けている
- [ ] `global-error.tsx` をルートに配置している
- [ ] エラーの種類に応じた表示分岐を実装している
- [ ] 本番環境ではエラーをモニタリングサービスに送信している
- [ ] `not-found.tsx` をカスタマイズし、ユーザーフレンドリーな404を表示している

### 13.5 パフォーマンス

- [ ] `'use client'` の使用を最小限に抑え、Client Component のバウンダリを意識している
- [ ] 重いコンポーネントは `<Suspense>` で分割し、ストリーミングを活用している
- [ ] `loading.tsx` でスケルトンUIを実装し、CLS（Cumulative Layout Shift）を防いでいる
- [ ] 静的メタデータは `metadata` オブジェクトで定義し、動的な場合のみ `generateMetadata` を使用している

---

## FAQ

### Q1: App Router と Pages Router の移行戦略は？
段階的移行が推奨される。`app/` と `pages/` は共存可能なため、新規ページから App Router で実装し、既存ページは必要に応じて移行する。Phase 1で `app/layout.tsx` とルートレイアウトを作成して共存を開始し、Phase 2で既存ページを徐々に移行し、Phase 3で完全移行する。注意点として、同じパスで `pages/` と `app/` が競合する場合は `app/` が優先される。`getServerSideProps` は Server Component に、`getStaticProps` は `generateStaticParams` に置き換える。

### Q2: 動的ルートと catch-all ルートの使い分けは？
URL構造が事前に定義されている場合（`/users/123`, `/posts/abc`）は動的ルート `[id]` を使い、可変長のパス（`/docs/getting-started/installation`）が必要な場合は catch-all `[...slug]` を使用する。Optional catch-all `[[...slug]]` は、パスが省略可能な場合（`/blog` でも `/blog/2024/01/hello` でもマッチ）に使う。具体的には、ECサイトの商品詳細は `[id]`、ドキュメントサイトの階層構造は `[...slug]`、ブログの一覧/詳細の両方を1ファイルで処理するなら `[[...slug]]` が適切である。

### Q3: ルートグループの活用法は？
ルートグループ `(name)` はURLに影響を与えずにレイアウトやミドルウェアのスコープを分けるために使用する。主な活用パターンは4つ: (1) レイアウト分離（マーケティングサイトとアプリケーションで異なるレイアウト）、(2) 認証エリアの分離（`(auth)/` と `(protected)/` で認証チェックの有無を制御）、(3) 国際化対応（`[locale]/(shop)/` と `[locale]/(blog)/` でセクションごとにレイアウトを変更）、(4) A/Bテスト（`(variant-a)/` と `(variant-b)/` をMiddlewareで振り分け）。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ファイル規約 | page, layout, loading, error, not-found, template, default, route |
| ルートグループ | `(name)` でURLに含めずレイアウト分割 |
| 動的ルート | `[id]`, `[...slug]`, `[[...slug]]` |
| パラレルルート | `@slot` で並列表示、独立したローディング/エラー |
| インターセプト | `(.)path` でモーダル表示、直接アクセスは全画面 |
| コンポーネント階層 | Layout > Template > ErrorBoundary > Suspense > NotFound > Page |
| Middleware | 認証・i18n・Rate Limiting・A/Bテスト |
| Route Segment Config | `dynamic`, `revalidate`, `runtime`, `dynamicParams` |
| コロケーション | `_components/` 等でページ専用ファイルを同居 |
| 移行 | Pages Router から段階的に移行可能 |

### フレームワーク選択の判断基準

| 要件 | 推奨フレームワーク | 理由 |
|------|------------------|------|
| React + SSR/SSG | Next.js App Router | エコシステムが最も豊富 |
| React + Web Standards | Remix / React Router v7 | progressive enhancement |
| Vue.js | Nuxt.js 3 | Vue エコシステムとの統合 |
| Svelte | SvelteKit | 軽量で高速 |
| コンテンツサイト | Astro | Islands Architecture で最小 JS |
| 型安全性重視 | SvelteKit / Next.js | 自動型生成が充実 |

---

## 次に読むべきガイド
- [[02-navigation-patterns.md]] -- ナビゲーション設計（Link、useRouter、リダイレクト）
- [[03-dynamic-routes-and-params.md]] -- 動的ルーティングの高度なパターン
- [[04-middleware-and-guards.md]] -- ミドルウェアとルートガード

---

## 参考文献
1. Next.js. "Routing Fundamentals." nextjs.org/docs/app/building-your-application/routing, 2025.
2. Next.js. "File Conventions." nextjs.org/docs/app/api-reference/file-conventions, 2025.
3. Next.js. "Parallel Routes." nextjs.org/docs/app/building-your-application/routing/parallel-routes, 2025.
4. Next.js. "Intercepting Routes." nextjs.org/docs/app/building-your-application/routing/intercepting-routes, 2025.
5. Next.js. "Route Handlers." nextjs.org/docs/app/building-your-application/routing/route-handlers, 2025.
6. Next.js. "Middleware." nextjs.org/docs/app/building-your-application/routing/middleware, 2025.
7. Remix. "Route File Naming v2." remix.run/docs/en/main/file-conventions/routes, 2025.
8. Remix. "Route Module." remix.run/docs/en/main/route/component, 2025.
9. SvelteKit. "Routing." kit.svelte.dev/docs/routing, 2025.
10. Nuxt.js. "Pages Directory." nuxt.com/docs/guide/directory-structure/pages, 2025.
11. Astro. "Routing." docs.astro.build/en/guides/routing, 2025.
12. Vercel. "Understanding Next.js App Router." vercel.com/blog, 2025.
13. Kent C. Dodds. "Full Stack Components." kentcdodds.com, 2024.
14. Lee Robinson. "Next.js App Router: Routing Patterns." leerob.io, 2024.
