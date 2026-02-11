# クライアントサイドルーティング

> クライアントルーティングはSPAの基盤技術。React Router v6、TanStack Routerの型安全なルーティング、ローダー/アクション、ルート分割まで、モダンなクライアントルーティングの全パターンを習得する。

## この章で学ぶこと

- [ ] React Router v6のデータルーティングを理解する
- [ ] TanStack Routerの型安全なルーティングを把握する
- [ ] コード分割とプリロードの実装を学ぶ

---

## 1. ルーティングの仕組み

```
クライアントサイドルーティング:
  → ブラウザの History API を使用
  → ページ全体のリロードなしでURLを変更
  → 該当するコンポーネントだけを再レンダリング

  history.pushState(state, '', '/users')  ← URLを変更
  window.addEventListener('popstate', handler)  ← ブラウザバック検知

  Hash Router:
  → /#/users — HashRouter（古い方式）
  → サーバー設定不要
  → SEO不利

  Browser Router:
  → /users — BrowserRouter（推奨）
  → サーバー側でSPA fallback設定が必要
  → SEO可能
```

---

## 2. React Router v6

```typescript
// ルート定義（データルーティング）
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    errorElement: <ErrorPage />,
    children: [
      { index: true, element: <HomePage /> },
      {
        path: 'users',
        element: <UsersLayout />,
        children: [
          {
            index: true,
            element: <UserList />,
            loader: usersLoader,    // データ取得
          },
          {
            path: ':userId',
            element: <UserDetail />,
            loader: userLoader,
          },
          {
            path: 'new',
            element: <CreateUser />,
            action: createUserAction, // データ変更
          },
        ],
      },
      {
        path: 'settings',
        lazy: () => import('./pages/Settings'), // 遅延読み込み
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}

// --- Loader（データ取得）---
async function usersLoader({ request }) {
  const url = new URL(request.url);
  const page = url.searchParams.get('page') ?? '1';
  return api.users.list({ page: Number(page) });
}

// --- Action（データ変更）---
async function createUserAction({ request }) {
  const formData = await request.formData();
  const user = await api.users.create({
    name: formData.get('name'),
    email: formData.get('email'),
  });
  return redirect(`/users/${user.id}`);
}

// --- コンポーネント ---
function UserList() {
  const users = useLoaderData();
  const navigation = useNavigation();
  const isLoading = navigation.state === 'loading';

  return (
    <div>
      {isLoading && <LoadingBar />}
      {users.map(user => (
        <Link to={user.id} key={user.id}>{user.name}</Link>
      ))}
    </div>
  );
}
```

---

## 3. TanStack Router

```typescript
// TanStack Router: 100% 型安全なルーティング
import { createRouter, createRoute, createRootRoute } from '@tanstack/react-router';

const rootRoute = createRootRoute({
  component: RootLayout,
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: HomePage,
});

const usersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/users',
  component: UsersLayout,
  validateSearch: (search) => ({
    page: Number(search.page ?? 1),
    filter: search.filter as string | undefined,
  }),
  loaderDeps: ({ search }) => ({ page: search.page, filter: search.filter }),
  loader: ({ deps }) => api.users.list(deps),
});

const userDetailRoute = createRoute({
  getParentRoute: () => usersRoute,
  path: '$userId',           // 型安全なパスパラメータ
  component: UserDetail,
  loader: ({ params }) => api.users.get(params.userId), // params.userId は string 型
});

const routeTree = rootRoute.addChildren([
  indexRoute,
  usersRoute.addChildren([userDetailRoute]),
]);

const router = createRouter({ routeTree });

// 型安全なリンク
<Link
  to="/users/$userId"
  params={{ userId: '123' }}    // 型チェック
  search={{ page: 2 }}          // 型チェック
>
  User Detail
</Link>

// 型安全な useParams
function UserDetail() {
  const { userId } = useParams({ from: '/users/$userId' }); // string型
  const search = useSearch({ from: '/users' }); // { page: number, filter?: string }
  const data = useLoaderData({ from: '/users/$userId' }); // User型
  // ...
}
```

---

## 4. コード分割

```typescript
// React.lazy + Suspense
const Settings = lazy(() => import('./pages/Settings'));
const Analytics = lazy(() => import('./pages/Analytics'));

function App() {
  return (
    <Suspense fallback={<PageSkeleton />}>
      <Routes>
        <Route path="/settings" element={<Settings />} />
        <Route path="/analytics" element={<Analytics />} />
      </Routes>
    </Suspense>
  );
}

// React Router の lazy（推奨）
const router = createBrowserRouter([
  {
    path: '/settings',
    lazy: async () => {
      const { Settings } = await import('./pages/Settings');
      return { Component: Settings };
    },
  },
]);

// プリロード（hover時にコード分割チャンクを先読み）
function NavLink({ to, children }) {
  const preload = () => {
    // ルートに対応するチャンクをプリロード
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = `/assets/pages/${to.replace('/', '')}.js`;
    document.head.appendChild(link);
  };

  return (
    <Link to={to} onMouseEnter={preload}>
      {children}
    </Link>
  );
}
```

---

## 5. 選定基準

```
React Router v6:
  ✓ 最も広く使われている
  ✓ ドキュメントが豊富
  ✓ Remix と同じ API
  → Vite + React プロジェクトの標準

TanStack Router:
  ✓ 100% 型安全（パラメータ、search params）
  ✓ 型安全な search params バリデーション
  ✓ ファーストクラスの search params サポート
  → 型安全性を最重視する場合

Next.js App Router:
  ✓ ファイルベースルーティング
  ✓ RSC / Streaming SSR
  ✓ Server Actions
  → フルスタック Next.js プロジェクト
```

---

## まとめ

| ライブラリ | 型安全性 | Search Params | エコシステム |
|-----------|---------|---------------|------------|
| React Router v6 | △ | 手動 | 最大 |
| TanStack Router | ◎ | 組み込み | 成長中 |
| Next.js App Router | ○ | nuqs推奨 | Next.js |

---

## 次に読むべきガイド
→ [[01-file-based-routing.md]] — ファイルベースルーティング

---

## 参考文献
1. React Router. "React Router v6 Documentation." reactrouter.com, 2024.
2. TanStack. "TanStack Router Documentation." tanstack.com, 2024.
