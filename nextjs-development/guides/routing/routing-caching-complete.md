# Next.js Routing & Caching 完全ガイド

## 目次
1. [App Routerの基礎](#app-routerの基礎)
2. [Dynamic Routes](#dynamic-routes)
3. [Route Groups](#route-groups)
4. [Parallel Routes](#parallel-routes)
5. [Intercepting Routes](#intercepting-routes)
6. [キャッシング戦略](#キャッシング戦略)
7. [Revalidation](#revalidation)
8. [ISR & On-Demand Revalidation](#isr--on-demand-revalidation)

---

## App Routerの基礎

### ファイルシステムルーティング

```
app/
├── page.tsx                    # / ルート
├── about/
│   └── page.tsx                # /about
├── blog/
│   ├── page.tsx                # /blog
│   ├── [slug]/
│   │   └── page.tsx            # /blog/[slug]
│   └── [...slug]/
│       └── page.tsx            # /blog/[...slug] (Catch-all)
├── dashboard/
│   ├── layout.tsx              # ダッシュボードレイアウト
│   ├── page.tsx                # /dashboard
│   ├── settings/
│   │   └── page.tsx            # /dashboard/settings
│   └── (overview)/             # Route Group（URLに含まれない）
│       ├── analytics/
│       │   └── page.tsx        # /dashboard/analytics
│       └── reports/
│           └── page.tsx        # /dashboard/reports
└── api/
    └── users/
        └── route.ts            # /api/users API Route
```

### Page Component

```typescript
// app/page.tsx

export default function HomePage() {
  return (
    <main>
      <h1>Welcome to Next.js</h1>
    </main>
  );
}

// メタデータの設定
export const metadata = {
  title: 'Home',
  description: 'Welcome to our application',
};
```

### Layout Component

```typescript
// app/layout.tsx

import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata = {
  title: {
    default: 'My App',
    template: '%s | My App', // サブページで使用される
  },
  description: 'My awesome application',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <header>Header</header>
        <main>{children}</main>
        <footer>Footer</footer>
      </body>
    </html>
  );
}
```

### ネストされたLayout

```typescript
// app/dashboard/layout.tsx

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <nav>
          <a href="/dashboard">Overview</a>
          <a href="/dashboard/settings">Settings</a>
        </nav>
      </aside>
      <div className="content">{children}</div>
    </div>
  );
}
```

---

## Dynamic Routes

### 基本的なDynamic Route

```typescript
// app/blog/[slug]/page.tsx

interface PageProps {
  params: {
    slug: string;
  };
  searchParams: {
    [key: string]: string | string[] | undefined;
  };
}

export default async function BlogPost({ params, searchParams }: PageProps) {
  const { slug } = params;

  // データフェッチ
  const post = await getPost(slug);

  return (
    <article>
      <h1>{post.title}</h1>
      <p>{post.content}</p>
    </article>
  );
}

// 静的パスの生成
export async function generateStaticParams() {
  const posts = await getPosts();

  return posts.map((post) => ({
    slug: post.slug,
  }));
}

// メタデータの動的生成
export async function generateMetadata({ params }: PageProps) {
  const post = await getPost(params.slug);

  return {
    title: post.title,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      images: [{ url: post.image }],
    },
  };
}
```

### Catch-all Routes

```typescript
// app/docs/[...slug]/page.tsx

interface PageProps {
  params: {
    slug: string[];
  };
}

export default async function DocsPage({ params }: PageProps) {
  // /docs/a/b/c → slug = ['a', 'b', 'c']
  const path = params.slug.join('/');

  const doc = await getDoc(path);

  return (
    <div>
      <h1>{doc.title}</h1>
      <div dangerouslySetInnerHTML={{ __html: doc.content }} />
    </div>
  );
}

export async function generateStaticParams() {
  const docs = await getAllDocs();

  return docs.map((doc) => ({
    slug: doc.path.split('/'),
  }));
}
```

### Optional Catch-all Routes

```typescript
// app/shop/[[...slug]]/page.tsx

interface PageProps {
  params: {
    slug?: string[];
  };
}

export default async function ShopPage({ params }: PageProps) {
  // /shop → slug = undefined
  // /shop/category → slug = ['category']
  // /shop/category/product → slug = ['category', 'product']

  if (!params.slug) {
    return <ShopHome />;
  }

  if (params.slug.length === 1) {
    return <Category slug={params.slug[0]} />;
  }

  return <Product category={params.slug[0]} slug={params.slug[1]} />;
}
```

---

## Route Groups

### Route Groupsの使い方

```
app/
├── (marketing)/           # Route Group（URLに影響しない）
│   ├── layout.tsx        # マーケティングページ用レイアウト
│   ├── page.tsx          # /
│   ├── about/
│   │   └── page.tsx      # /about
│   └── contact/
│       └── page.tsx      # /contact
├── (shop)/               # Route Group
│   ├── layout.tsx        # ショップページ用レイアウト
│   ├── products/
│   │   └── page.tsx      # /products
│   └── cart/
│       └── page.tsx      # /cart
└── (dashboard)/          # Route Group
    ├── layout.tsx        # ダッシュボード用レイアウト
    ├── analytics/
    │   └── page.tsx      # /analytics
    └── settings/
        └── page.tsx      # /settings
```

```typescript
// app/(marketing)/layout.tsx

export default function MarketingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <header className="marketing-header">
        <nav>{/* マーケティング用ナビゲーション */}</nav>
      </header>
      {children}
    </>
  );
}

// app/(dashboard)/layout.tsx

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="dashboard">
      <Sidebar />
      <main>{children}</main>
    </div>
  );
}
```

---

## Parallel Routes

### 並列ルートの基本

```
app/
└── dashboard/
    ├── @user/               # Slot: user
    │   └── page.tsx
    ├── @team/               # Slot: team
    │   └── page.tsx
    ├── @analytics/          # Slot: analytics
    │   └── page.tsx
    └── layout.tsx
```

```typescript
// app/dashboard/layout.tsx

export default function DashboardLayout({
  user,
  team,
  analytics,
}: {
  user: React.ReactNode;
  team: React.ReactNode;
  analytics: React.ReactNode;
}) {
  return (
    <div className="dashboard-grid">
      <div className="user-section">{user}</div>
      <div className="team-section">{team}</div>
      <div className="analytics-section">{analytics}</div>
    </div>
  );
}

// app/dashboard/@user/page.tsx
export default function UserSlot() {
  return <div>User Information</div>;
}

// app/dashboard/@team/page.tsx
export default function TeamSlot() {
  return <div>Team Information</div>;
}

// app/dashboard/@analytics/page.tsx
export default function AnalyticsSlot() {
  return <div>Analytics</div>;
}
```

### 条件付きレンダリング

```typescript
// app/dashboard/layout.tsx

export default async function DashboardLayout({
  user,
  team,
  analytics,
}: {
  user: React.ReactNode;
  team: React.ReactNode;
  analytics: React.ReactNode;
}) {
  const session = await getServerSession();
  const hasTeamAccess = await checkTeamAccess(session.user.id);

  return (
    <div className="dashboard-grid">
      {user}
      {hasTeamAccess && team}
      {analytics}
    </div>
  );
}
```

---

## Intercepting Routes

### Intercepting Routesの使い方

```
app/
├── feed/
│   ├── page.tsx
│   └── (..)photo/          # Intercepting Route
│       └── [id]/
│           └── page.tsx    # モーダル表示
└── photo/
    └── [id]/
        └── page.tsx        # 通常ページ
```

```typescript
// app/feed/(..)photo/[id]/page.tsx (モーダル)

import Modal from '@/components/Modal';

export default async function PhotoModal({
  params,
}: {
  params: { id: string };
}) {
  const photo = await getPhoto(params.id);

  return (
    <Modal>
      <img src={photo.url} alt={photo.title} />
      <h2>{photo.title}</h2>
    </Modal>
  );
}

// app/photo/[id]/page.tsx (通常ページ)

export default async function PhotoPage({
  params,
}: {
  params: { id: string };
}) {
  const photo = await getPhoto(params.id);

  return (
    <div>
      <img src={photo.url} alt={photo.title} />
      <h1>{photo.title}</h1>
      <p>{photo.description}</p>
    </div>
  );
}
```

### Modal Component

```typescript
// components/Modal.tsx

'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useRef } from 'react';

export default function Modal({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const dialogRef = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    dialogRef.current?.showModal();
  }, []);

  const closeModal = () => {
    dialogRef.current?.close();
    router.back();
  };

  return (
    <dialog
      ref={dialogRef}
      className="modal"
      onClose={closeModal}
      onClick={(e) => {
        if (e.target === dialogRef.current) {
          closeModal();
        }
      }}
    >
      <button onClick={closeModal}>Close</button>
      {children}
    </dialog>
  );
}
```

---

## キャッシング戦略

### Request Memoization

```typescript
// lib/data.ts

export async function getUser(id: string) {
  // 同じリクエスト内で複数回呼ばれても、実際のfetchは1回だけ
  const res = await fetch(`https://api.example.com/users/${id}`);
  return res.json();
}

// app/user/[id]/page.tsx
export default async function UserPage({ params }: { params: { id: string } }) {
  const user = await getUser(params.id); // 1回目
  const sameUser = await getUser(params.id); // キャッシュから取得
  const againUser = await getUser(params.id); // キャッシュから取得

  return <div>{user.name}</div>;
}
```

### Data Cache

```typescript
// デフォルト: cache: 'force-cache' (永続キャッシュ)
async function getStaticData() {
  const res = await fetch('https://api.example.com/data', {
    cache: 'force-cache', // デフォルト
  });
  return res.json();
}

// cache: 'no-store' (キャッシュなし、常に最新)
async function getDynamicData() {
  const res = await fetch('https://api.example.com/data', {
    cache: 'no-store',
  });
  return res.json();
}

// next.revalidate (時間ベースのrevalidation)
async function getRevalidatedData() {
  const res = await fetch('https://api.example.com/data', {
    next: { revalidate: 3600 }, // 1時間ごとにrevalidate
  });
  return res.json();
}

// タグベースのrevalidation
async function getTaggedData() {
  const res = await fetch('https://api.example.com/data', {
    next: { tags: ['posts'] },
  });
  return res.json();
}
```

### Full Route Cache

```typescript
// app/blog/[slug]/page.tsx

// Static Generation (デフォルト)
export const dynamic = 'auto'; // デフォルト: 可能な限り静的生成

// または明示的に指定
export const dynamic = 'force-static'; // 強制的に静的生成
export const dynamic = 'force-dynamic'; // 強制的に動的レンダリング
export const dynamic = 'error'; // 動的関数使用時にエラー

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const post = await fetch(`https://api.example.com/posts/${params.slug}`, {
    next: { revalidate: 60 }, // 60秒ごとにrevalidate
  });

  return <article>{/* ... */}</article>;
}
```

### Router Cache (Client-side)

```typescript
'use client';

import { useRouter } from 'next/navigation';

export default function Navigation() {
  const router = useRouter();

  const handleNavigation = () => {
    // プリフェッチ（自動的にキャッシュされる）
    router.prefetch('/dashboard');

    // ナビゲーション（キャッシュから即座に表示）
    router.push('/dashboard');

    // キャッシュをリフレッシュ
    router.refresh();
  };

  return <button onClick={handleNavigation}>Go to Dashboard</button>;
}
```

---

## Revalidation

### Time-based Revalidation

```typescript
// app/posts/page.tsx

export const revalidate = 3600; // 1時間ごとにrevalidate

export default async function PostsPage() {
  const posts = await fetch('https://api.example.com/posts').then((res) =>
    res.json()
  );

  return (
    <div>
      {posts.map((post) => (
        <div key={post.id}>{post.title}</div>
      ))}
    </div>
  );
}

// または個別のfetchで設定
async function getPosts() {
  const res = await fetch('https://api.example.com/posts', {
    next: { revalidate: 3600 },
  });
  return res.json();
}
```

### On-Demand Revalidation (Tag-based)

```typescript
// app/posts/[id]/page.tsx

export default async function PostPage({ params }: { params: { id: string } }) {
  const post = await fetch(`https://api.example.com/posts/${params.id}`, {
    next: { tags: ['posts', `post-${params.id}`] },
  }).then((res) => res.json());

  return <article>{post.content}</article>;
}

// app/api/revalidate/route.ts

import { revalidateTag } from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const secret = request.nextUrl.searchParams.get('secret');

  // セキュリティチェック
  if (secret !== process.env.REVALIDATE_SECRET) {
    return NextResponse.json({ message: 'Invalid secret' }, { status: 401 });
  }

  const tag = request.nextUrl.searchParams.get('tag');

  if (tag) {
    revalidateTag(tag);
    return NextResponse.json({ revalidated: true, tag });
  }

  return NextResponse.json({ message: 'Missing tag' }, { status: 400 });
}

// 使用例:
// POST /api/revalidate?secret=xxx&tag=posts
// POST /api/revalidate?secret=xxx&tag=post-123
```

### On-Demand Revalidation (Path-based)

```typescript
// app/api/revalidate-path/route.ts

import { revalidatePath } from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const path = request.nextUrl.searchParams.get('path');

  if (path) {
    revalidatePath(path);
    return NextResponse.json({ revalidated: true, path });
  }

  return NextResponse.json({ message: 'Missing path' }, { status: 400 });
}

// 使用例:
// POST /api/revalidate-path?path=/posts
// POST /api/revalidate-path?path=/posts/123
```

---

## ISR & On-Demand Revalidation

### Incremental Static Regeneration (ISR)

```typescript
// app/products/[id]/page.tsx

export const revalidate = 60; // 60秒ごとにrevalidate

export async function generateStaticParams() {
  // ビルド時に人気商品のみ生成
  const popularProducts = await getPopularProducts();

  return popularProducts.map((product) => ({
    id: product.id,
  }));
}

export default async function ProductPage({
  params,
}: {
  params: { id: string };
}) {
  // ビルド時に生成されていないページは、
  // 初回アクセス時に生成され、以降はキャッシュされる
  const product = await getProduct(params.id);

  return (
    <div>
      <h1>{product.name}</h1>
      <p>{product.description}</p>
      <p>${product.price}</p>
    </div>
  );
}
```

### WebhookによるOn-Demand Revalidation

```typescript
// app/api/webhook/cms/route.ts

import { revalidateTag, revalidatePath } from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // CMSからのWebhookペイロード
    const { event, data } = body;

    if (event === 'post.published') {
      // 特定の記事をrevalidate
      revalidateTag(`post-${data.id}`);
      revalidatePath(`/blog/${data.slug}`);

      // 記事一覧もrevalidate
      revalidateTag('posts');
      revalidatePath('/blog');

      return NextResponse.json({ revalidated: true });
    }

    if (event === 'post.deleted') {
      revalidateTag('posts');
      revalidatePath('/blog');

      return NextResponse.json({ revalidated: true });
    }

    return NextResponse.json({ message: 'No action taken' });
  } catch (error) {
    return NextResponse.json(
      { message: 'Error revalidating' },
      { status: 500 }
    );
  }
}

// CMS側の設定例（Contentful, Sanity, Strapi等）:
// Webhook URL: https://yoursite.com/api/webhook/cms
// Events: Entry published, Entry unpublished
```

### 複数パスの一括Revalidation

```typescript
// app/api/revalidate-all/route.ts

import { revalidateTag, revalidatePath } from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  const { tags, paths } = await request.json();

  // 複数のタグをrevalidate
  if (tags && Array.isArray(tags)) {
    tags.forEach((tag) => revalidateTag(tag));
  }

  // 複数のパスをrevalidate
  if (paths && Array.isArray(paths)) {
    paths.forEach((path) => revalidatePath(path));
  }

  return NextResponse.json({ revalidated: true, tags, paths });
}

// 使用例:
// POST /api/revalidate-all
// {
//   "tags": ["posts", "products"],
//   "paths": ["/blog", "/shop"]
// }
```

---

このガイドでは、Next.js App Routerの高度なルーティング機能（Dynamic Routes、Route Groups、Parallel Routes、Intercepting Routes）と、包括的なキャッシング戦略（Request Memoization、Data Cache、Full Route Cache、Router Cache）、そしてRevalidationの各手法（Time-based、On-Demand、ISR）について詳細に解説しました。これらを適切に組み合わせることで、高速でスケーラブルなアプリケーションを構築できます。
