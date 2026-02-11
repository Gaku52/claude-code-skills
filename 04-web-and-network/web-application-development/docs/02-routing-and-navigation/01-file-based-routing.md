# ファイルベースルーティング

> ファイルベースルーティングは「ファイル構造 = URL構造」の直感的なアプローチ。Next.js App Router、Remix のルーティング規約、レイアウト、ローディング、エラーハンドリングまで、ファイルベースルーティングの全パターンを習得する。

## この章で学ぶこと

- [ ] Next.js App Routerのファイル規約を理解する
- [ ] レイアウト、ローディング、エラーの設計を把握する
- [ ] 動的ルート、ルートグループ、パラレルルートを学ぶ

---

## 1. Next.js App Router

```
ファイル規約:
  app/
  ├── layout.tsx          ← ルートレイアウト（必須）
  ├── page.tsx            ← / のページ
  ├── loading.tsx         ← ローディングUI
  ├── error.tsx           ← エラーUI
  ├── not-found.tsx       ← 404 UI
  ├── users/
  │   ├── page.tsx        ← /users
  │   ├── loading.tsx     ← /users のローディング
  │   ├── [id]/
  │   │   ├── page.tsx    ← /users/:id
  │   │   └── edit/
  │   │       └── page.tsx ← /users/:id/edit
  │   └── new/
  │       └── page.tsx    ← /users/new
  └── api/
      └── webhooks/
          └── route.ts    ← API Route: POST /api/webhooks

特殊ファイル:
  page.tsx      → ルートのUIコンポーネント
  layout.tsx    → 共有レイアウト（再レンダリングされない）
  template.tsx  → layout同様だが遷移ごとに再マウント
  loading.tsx   → Suspense の fallback（自動ラップ）
  error.tsx     → ErrorBoundary（自動ラップ）
  not-found.tsx → notFound() 呼び出し時のUI
  route.ts      → API Route（HTTPハンドラー）
```

---

## 2. レイアウトの設計

```typescript
// app/layout.tsx — ルートレイアウト
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ja">
      <body>
        <Providers>
          <Header />
          <main>{children}</main>
          <Footer />
        </Providers>
      </body>
    </html>
  );
}

// app/(marketing)/layout.tsx — マーケティング用レイアウト
export default function MarketingLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="max-w-4xl mx-auto">
      <MarketingNav />
      {children}
    </div>
  );
}

// app/(app)/layout.tsx — アプリ用レイアウト（認証必要）
export default async function AppLayout({ children }: { children: React.ReactNode }) {
  const session = await getSession();
  if (!session) redirect('/login');

  return (
    <div className="flex">
      <Sidebar />
      <div className="flex-1 p-6">{children}</div>
    </div>
  );
}

// ルートグループ:
// (marketing) → URL に含まれない、レイアウト用のグループ
// (app)       → URL に含まれない、認証エリアのグループ
//
// app/(marketing)/page.tsx       → /
// app/(marketing)/about/page.tsx → /about
// app/(app)/dashboard/page.tsx   → /dashboard
// app/(app)/settings/page.tsx    → /settings
```

---

## 3. 動的ルートとキャッチオール

```
動的ルート:
  [id]         → /users/123      params.id = "123"
  [slug]       → /posts/hello    params.slug = "hello"
  [...slug]    → /docs/a/b/c     params.slug = ["a", "b", "c"]
  [[...slug]]  → /docs (or) /docs/a/b  params.slug = [] or ["a", "b"]

// app/users/[id]/page.tsx
export default async function UserPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const user = await getUser(id);

  if (!user) notFound();

  return <UserProfile user={user} />;
}

// 静的パラメータの生成（SSG用）
export async function generateStaticParams() {
  const users = await getUsers();
  return users.map(user => ({ id: user.id }));
}

// メタデータの動的生成
export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const user = await getUser(id);
  return {
    title: user?.name ?? 'User Not Found',
    description: `${user?.name}のプロフィール`,
  };
}
```

---

## 4. ローディングとエラー

```typescript
// app/users/loading.tsx — 自動Suspense
export default function UsersLoading() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="h-16 bg-gray-200 animate-pulse rounded" />
      ))}
    </div>
  );
}

// app/users/error.tsx — 自動ErrorBoundary
'use client'; // error.tsx は必ず Client Component

export default function UsersError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="text-center p-8">
      <h2 className="text-xl font-bold text-red-600">Something went wrong</h2>
      <p className="text-gray-600 mt-2">{error.message}</p>
      <button
        onClick={reset}
        className="mt-4 px-4 py-2 bg-blue-500 text-white rounded"
      >
        Try again
      </button>
    </div>
  );
}

// app/not-found.tsx — グローバル404
export default function NotFound() {
  return (
    <div className="text-center p-16">
      <h1 className="text-6xl font-bold">404</h1>
      <p className="text-xl mt-4">Page not found</p>
      <Link href="/" className="mt-8 inline-block text-blue-500">
        Go home
      </Link>
    </div>
  );
}
```

---

## 5. パラレルルートとインターセプトルート

```
パラレルルート:
  → 同じレイアウト内で複数のページを並列に表示
  → ダッシュボード: メイン + サイドバー

  app/dashboard/
  ├── layout.tsx
  ├── page.tsx
  ├── @analytics/
  │   └── page.tsx          ← analytics スロット
  └── @activity/
      └── page.tsx          ← activity スロット

  // layout.tsx
  export default function DashboardLayout({
    children,     // page.tsx
    analytics,    // @analytics/page.tsx
    activity,     // @activity/page.tsx
  }) {
    return (
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">{children}</div>
        <div>
          {analytics}
          {activity}
        </div>
      </div>
    );
  }

インターセプトルート:
  → モーダルでの表示（URLは変わるが背景ページは維持）
  → Instagram風: フィード上でクリック → モーダル表示

  app/
  ├── feed/
  │   └── page.tsx
  ├── photo/
  │   └── [id]/
  │       └── page.tsx       ← 直接アクセス時の全画面表示
  └── @modal/
      └── (.)photo/
          └── [id]/
              └── page.tsx   ← フィードからのモーダル表示

  インターセプト記法:
  (.)   → 同じレベル
  (..)  → 1つ上のレベル
  (..)(..) → 2つ上
  (...) → ルートから
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ファイル規約 | page, layout, loading, error, not-found |
| ルートグループ | (name) でURLに含めずレイアウト分割 |
| 動的ルート | [id], [...slug], [[...slug]] |
| パラレルルート | @slot で並列表示 |
| インターセプト | (.)path でモーダル表示 |

---

## 次に読むべきガイド
→ [[02-navigation-patterns.md]] — ナビゲーション設計

---

## 参考文献
1. Next.js. "Routing Fundamentals." nextjs.org/docs, 2024.
2. Remix. "Route File Naming." remix.run/docs, 2024.
