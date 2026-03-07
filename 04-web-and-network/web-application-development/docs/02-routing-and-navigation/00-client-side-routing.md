# クライアントサイドルーティング

> クライアントルーティングはSPAの基盤技術。React Router v6、TanStack Routerの型安全なルーティング、ローダー/アクション、ルート分割まで、モダンなクライアントルーティングの全パターンを習得する。

## この章で学ぶこと

- [ ] クライアントサイドルーティングの仕組みと History API を深く理解する
- [ ] React Router v6 のデータルーティング（loader/action）を実践的に使いこなす
- [ ] TanStack Router の型安全なルーティングを把握する
- [ ] コード分割とプリロードの実装パターンを学ぶ
- [ ] ルーティングに関するセキュリティとアクセス制御を理解する
- [ ] パフォーマンス最適化とトラブルシューティング手法を習得する

---

## 1. ルーティングの基礎概念

### 1.1 サーバーサイドルーティング vs クライアントサイドルーティング

Web アプリケーションにおけるルーティングには、大きく分けて2つのアプローチが存在する。それぞれの特徴を理解することが、適切なアーキテクチャ選定の第一歩となる。

```
サーバーサイドルーティング（従来方式）:
  ブラウザ → サーバーへHTTPリクエスト → サーバーがHTML生成 → ブラウザが全ページ描画

  [ユーザーがリンクをクリック]
       ↓
  [ブラウザがサーバーへGETリクエスト]
       ↓
  [サーバーがルーティングテーブルを参照]
       ↓
  [対応するコントローラー/ハンドラーが実行]
       ↓
  [完全なHTMLページを生成]
       ↓
  [ブラウザが全ページをリロード・描画]

  特徴:
  - ページ遷移ごとに白画面（FOUC）が発生
  - サーバー負荷が高い
  - SEOに自然に対応
  - JavaScript が無効でも動作

クライアントサイドルーティング（SPA方式）:
  ブラウザ → History API でURL変更 → JavaScriptがコンポーネント切替 → 部分更新のみ

  [ユーザーがリンクをクリック]
       ↓
  [JavaScriptがイベントをインターセプト（preventDefault）]
       ↓
  [History API でURLを書き換え]
       ↓
  [ルーターが新URLにマッチするコンポーネントを特定]
       ↓
  [該当コンポーネントのみをレンダリング]
       ↓
  [DOMの部分更新]

  特徴:
  - 瞬時のページ遷移（ネイティブアプリに近い体験）
  - サーバー負荷が軽い（APIリクエストのみ）
  - 初回ロードが重い（JS バンドル全体のダウンロード）
  - SEO対策に追加の工夫が必要
```

### 1.2 History API の詳細

クライアントサイドルーティングの根幹を成すのが、ブラウザの History API である。この API を正確に理解することで、ルーティングライブラリの内部動作を把握できる。

```typescript
// === History API 基本操作 ===

// 1. pushState: 履歴スタックに新しいエントリを追加
// ページリロードなしでURLを変更する
history.pushState(
  { userId: 123, page: 'profile' },  // state オブジェクト（任意のデータ）
  '',                                  // title（ほとんどのブラウザで無視される）
  '/users/123/profile'                // 新しいURL
);

// 2. replaceState: 現在の履歴エントリを置き換え
// リダイレクト時やフォーム送信後に使用
history.replaceState(
  { redirectedFrom: '/old-path' },
  '',
  '/new-path'
);

// 3. popstate イベント: ブラウザの戻る/進むボタンの検知
window.addEventListener('popstate', (event) => {
  console.log('Navigation detected');
  console.log('State:', event.state);  // pushState で渡した state
  console.log('Current URL:', window.location.href);

  // ここでルーティングロジックを実行
  handleRouteChange(window.location.pathname);
});

// 4. 履歴の操作
history.back();      // ブラウザの「戻る」と同じ
history.forward();   // ブラウザの「進む」と同じ
history.go(-2);      // 2つ前のページに戻る
history.go(0);       // 現在のページをリロード

// 5. 現在の状態を取得
console.log(history.state);   // 現在のエントリの state
console.log(history.length);  // 履歴スタックのエントリ数
```

```typescript
// === 最小限のクライアントサイドルーター実装 ===
// ルーティングライブラリの内部動作を理解するための教育的実装

interface Route {
  path: string;
  pattern: RegExp;
  paramNames: string[];
  handler: (params: Record<string, string>) => void;
}

class SimpleRouter {
  private routes: Route[] = [];
  private currentPath: string = '';

  constructor() {
    // popstate イベントでブラウザバック/フォワードに対応
    window.addEventListener('popstate', () => {
      this.handleRouteChange(window.location.pathname);
    });

    // 全てのリンククリックをインターセプト
    document.addEventListener('click', (e) => {
      const target = e.target as HTMLElement;
      const anchor = target.closest('a');

      if (anchor && anchor.href && anchor.origin === window.location.origin) {
        e.preventDefault();
        this.navigate(anchor.pathname);
      }
    });
  }

  // ルートを登録する
  addRoute(path: string, handler: (params: Record<string, string>) => void): void {
    const paramNames: string[] = [];

    // パスパターンを正規表現に変換
    // 例: '/users/:userId/posts/:postId' → /^\/users\/([^\/]+)\/posts\/([^\/]+)$/
    const patternStr = path.replace(/:(\w+)/g, (_, paramName) => {
      paramNames.push(paramName);
      return '([^/]+)';
    });

    this.routes.push({
      path,
      pattern: new RegExp(`^${patternStr}$`),
      paramNames,
      handler,
    });
  }

  // プログラマティックナビゲーション
  navigate(path: string, options?: { replace?: boolean }): void {
    if (options?.replace) {
      history.replaceState({ path }, '', path);
    } else {
      history.pushState({ path }, '', path);
    }
    this.handleRouteChange(path);
  }

  // ルートマッチングとハンドラー実行
  private handleRouteChange(path: string): void {
    this.currentPath = path;

    for (const route of this.routes) {
      const match = path.match(route.pattern);
      if (match) {
        // パラメータを抽出
        const params: Record<string, string> = {};
        route.paramNames.forEach((name, index) => {
          params[name] = match[index + 1];
        });

        route.handler(params);
        return;
      }
    }

    // 404: マッチするルートが見つからない
    console.warn(`No route matched for path: ${path}`);
    this.handle404();
  }

  private handle404(): void {
    document.getElementById('app')!.innerHTML = '<h1>404 - Page Not Found</h1>';
  }
}

// 使用例
const router = new SimpleRouter();

router.addRoute('/', () => {
  document.getElementById('app')!.innerHTML = '<h1>Home</h1>';
});

router.addRoute('/users', () => {
  document.getElementById('app')!.innerHTML = '<h1>Users List</h1>';
});

router.addRoute('/users/:userId', (params) => {
  document.getElementById('app')!.innerHTML = `<h1>User: ${params.userId}</h1>`;
});

router.addRoute('/users/:userId/posts/:postId', (params) => {
  document.getElementById('app')!.innerHTML =
    `<h1>User ${params.userId} - Post ${params.postId}</h1>`;
});

// 初期ルートを処理
router.navigate(window.location.pathname, { replace: true });
```

### 1.3 Hash Router vs Browser Router

```
Hash Router（ハッシュルーター）:
  URL形式: https://example.com/#/users/123

  仕組み:
  - URL のハッシュ部分（#以降）を利用
  - hashchange イベントで変更を検知
  - ハッシュ部分はサーバーに送信されない

  メリット:
  ✓ サーバー設定が一切不要
  ✓ 静的ファイルホスティングでそのまま動作
  ✓ GitHub Pages, S3 静的ホスティングに最適
  ✓ 古いブラウザでも動作

  デメリット:
  ✗ URL が不格好（#が入る）
  ✗ SEO に不利（クローラーがハッシュを無視する場合がある）
  ✗ SSR との組み合わせが困難
  ✗ アンカーリンク（ページ内リンク）と競合する

  使用場面:
  - 管理画面・ダッシュボード（SEO不要）
  - Electron アプリ内のルーティング
  - 静的ホスティング環境（サーバー設定不可）

Browser Router（ブラウザルーター）:
  URL形式: https://example.com/users/123

  仕組み:
  - History API の pushState/replaceState を使用
  - popstate イベントで変更を検知
  - 通常のURLパスを使用

  メリット:
  ✓ クリーンなURL
  ✓ SEO 対応が容易
  ✓ SSR / SSG と組み合わせ可能
  ✓ アンカーリンクが正常に動作

  デメリット:
  ✗ サーバー側で SPA fallback 設定が必要
  ✗ 設定を誤ると直接アクセスで 404 エラー

  サーバー設定例:

  Nginx:
    location / {
      try_files $uri $uri/ /index.html;
    }

  Apache (.htaccess):
    RewriteEngine On
    RewriteBase /
    RewriteRule ^index\.html$ - [L]
    RewriteCond %{REQUEST_FILENAME} !-f
    RewriteCond %{REQUEST_FILENAME} !-d
    RewriteRule . /index.html [L]

  Vercel (vercel.json):
    { "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }] }

  Netlify (_redirects):
    /*    /index.html   200

Memory Router（メモリルーター）:
  URL形式: URLは変更されない（メモリ内のみ）

  仕組み:
  - 履歴をメモリ内の配列で管理
  - ブラウザのURLバーは変更されない

  使用場面:
  - テスト環境（Jest, Vitest）
  - React Native
  - Storybook での UI コンポーネント開発
  - iframe 内のアプリケーション
```

```typescript
// === Hash Router の実装 ===
import { createHashRouter, RouterProvider } from 'react-router-dom';

const hashRouter = createHashRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'users', element: <Users /> },
    ],
  },
]);

// URL: https://example.com/#/users

// === Browser Router の実装 ===
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

const browserRouter = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'users', element: <Users /> },
    ],
  },
]);

// URL: https://example.com/users

// === Memory Router の実装（テスト用）===
import { createMemoryRouter, RouterProvider } from 'react-router-dom';

// テストでの使用例
function renderWithRouter(element: React.ReactElement, initialEntries = ['/']) {
  const routes = [
    {
      path: '/',
      element: <RootLayout />,
      children: [
        { index: true, element: <Home /> },
        { path: 'users', element: <Users /> },
        { path: 'users/:userId', element: <UserDetail /> },
      ],
    },
  ];

  const router = createMemoryRouter(routes, {
    initialEntries,
    initialIndex: 0,
  });

  return render(<RouterProvider router={router} />);
}

// テストコード
describe('UserDetail', () => {
  it('should display user information', async () => {
    renderWithRouter(<UserDetail />, ['/users/42']);
    expect(await screen.findByText('User #42')).toBeInTheDocument();
  });
});
```

### 1.4 ルーティングのライフサイクル

クライアントサイドルーティングにおけるナビゲーションは、以下のライフサイクルで処理される。

```
ナビゲーションのライフサイクル:

1. トリガー
   ├── ユーザーアクション（リンククリック、フォーム送信）
   ├── プログラマティック（navigate(), router.push()）
   └── ブラウザアクション（戻る、進む、URL直接入力）

2. ガード/ミドルウェア チェック
   ├── 認証チェック（ログイン済みか？）
   ├── 権限チェック（アクセス権があるか？）
   ├── 離脱確認（未保存データがあるか？）
   └── リダイレクト判定

3. URL 更新
   ├── pushState / replaceState の実行
   └── ブラウザのURLバーが更新される

4. ルートマッチング
   ├── 新しいURLに対応するルートを検索
   ├── パスパラメータの抽出
   ├── クエリパラメータのパース
   └── ワイルドカード/キャッチオール マッチング

5. データ取得（loader の実行）
   ├── 並列データフェッチ
   ├── キャッシュの確認
   └── ローディング状態の管理

6. レンダリング
   ├── 新しいコンポーネントのマウント
   ├── 共有レイアウトの保持
   ├── アニメーション/トランジション
   └── スクロール位置の復元

7. 後処理
   ├── ページタイトルの更新
   ├── アナリティクスイベントの送信
   ├── フォーカス管理（アクセシビリティ）
   └── メタタグの更新
```

---

## 2. React Router v6 徹底解説

### 2.1 基本的なルート定義

React Router v6 では、従来の `<Routes>` コンポーネントベースの定義と、新しいデータルーティング（`createBrowserRouter`）の2つのアプローチが用意されている。現在はデータルーティングが推奨されている。

```typescript
// === 方式1: コンポーネントベース（旧方式・後方互換性のため残存）===
import { BrowserRouter, Routes, Route, Outlet } from 'react-router-dom';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RootLayout />}>
          <Route index element={<HomePage />} />
          <Route path="about" element={<AboutPage />} />
          <Route path="users" element={<UsersLayout />}>
            <Route index element={<UserList />} />
            <Route path=":userId" element={<UserDetail />} />
          </Route>
          <Route path="*" element={<NotFoundPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

function RootLayout() {
  return (
    <div>
      <nav>
        <NavLink to="/" end>Home</NavLink>
        <NavLink to="/users">Users</NavLink>
        <NavLink to="/about">About</NavLink>
      </nav>
      <main>
        <Outlet /> {/* 子ルートがここにレンダリングされる */}
      </main>
    </div>
  );
}

// === 方式2: データルーティング（推奨）===
import {
  createBrowserRouter,
  RouterProvider,
  Outlet,
  useLoaderData,
  useActionData,
  useNavigation,
  useRouteError,
  isRouteErrorResponse,
  redirect,
  json,
} from 'react-router-dom';

const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    errorElement: <RootErrorBoundary />,
    children: [
      {
        index: true,
        element: <HomePage />,
        loader: homeLoader,
      },
      {
        path: 'users',
        element: <UsersLayout />,
        errorElement: <UsersErrorBoundary />,
        children: [
          {
            index: true,
            element: <UserList />,
            loader: usersLoader,
          },
          {
            path: ':userId',
            element: <UserDetail />,
            loader: userDetailLoader,
            children: [
              {
                path: 'edit',
                element: <UserEdit />,
                loader: userEditLoader,
                action: userEditAction,
              },
            ],
          },
          {
            path: 'new',
            element: <CreateUser />,
            action: createUserAction,
          },
        ],
      },
      {
        path: 'settings',
        lazy: () => import('./pages/Settings'),
      },
      {
        path: 'admin',
        element: <AdminLayout />,
        loader: adminLoader,  // 認証チェック
        children: [
          { index: true, element: <AdminDashboard /> },
          { path: 'users', element: <AdminUserManagement /> },
          { path: 'analytics', element: <AdminAnalytics /> },
        ],
      },
      {
        path: '*',
        element: <NotFoundPage />,
      },
    ],
  },
]);

function App() {
  return <RouterProvider router={router} />;
}
```

### 2.2 Loader パターン（データ取得）

Loader は React Router v6.4 で導入された、ルートレベルでのデータ取得メカニズムである。コンポーネントがレンダリングされる前にデータを取得し、ウォーターフォール問題を解消する。

```typescript
// === 基本的な Loader ===
import type { LoaderFunctionArgs } from 'react-router-dom';

// シンプルなデータ取得
async function usersLoader({ request }: LoaderFunctionArgs) {
  const url = new URL(request.url);
  const page = Number(url.searchParams.get('page') ?? '1');
  const limit = Number(url.searchParams.get('limit') ?? '20');
  const search = url.searchParams.get('q') ?? '';

  const response = await fetch(
    `/api/users?page=${page}&limit=${limit}&q=${encodeURIComponent(search)}`,
    {
      signal: request.signal, // ナビゲーションキャンセル時にリクエストを中断
    }
  );

  if (!response.ok) {
    // エラーレスポンスを throw すると errorElement で捕捉される
    throw new Response('Failed to fetch users', {
      status: response.status,
      statusText: response.statusText,
    });
  }

  return response.json();
}

// パラメータ付きの Loader
async function userDetailLoader({ params, request }: LoaderFunctionArgs) {
  const { userId } = params;

  if (!userId) {
    throw new Response('User ID is required', { status: 400 });
  }

  const response = await fetch(`/api/users/${userId}`, {
    signal: request.signal,
  });

  if (response.status === 404) {
    throw new Response('User not found', { status: 404 });
  }

  if (!response.ok) {
    throw new Response('Server error', { status: 500 });
  }

  return response.json();
}

// 認証ガード付き Loader
async function adminLoader({ request }: LoaderFunctionArgs) {
  const user = await getCurrentUser();

  if (!user) {
    // ログインページにリダイレクト（元のURLを保持）
    const url = new URL(request.url);
    return redirect(`/login?redirectTo=${encodeURIComponent(url.pathname)}`);
  }

  if (user.role !== 'admin') {
    throw new Response('Forbidden: Admin access required', { status: 403 });
  }

  return { user };
}

// 並列データ取得（Promise.all を活用）
async function dashboardLoader({ request }: LoaderFunctionArgs) {
  const [stats, recentUsers, notifications] = await Promise.all([
    fetch('/api/stats', { signal: request.signal }).then(r => r.json()),
    fetch('/api/users/recent', { signal: request.signal }).then(r => r.json()),
    fetch('/api/notifications', { signal: request.signal }).then(r => r.json()),
  ]);

  return { stats, recentUsers, notifications };
}

// defer を使った段階的データ取得
import { defer, Await } from 'react-router-dom';

async function dashboardLoaderWithDefer({ request }: LoaderFunctionArgs) {
  // 重要なデータは即座に取得（await する）
  const stats = await fetch('/api/stats', { signal: request.signal })
    .then(r => r.json());

  // 重要度の低いデータは遅延取得（Promise のまま返す）
  const recentUsersPromise = fetch('/api/users/recent', { signal: request.signal })
    .then(r => r.json());
  const notificationsPromise = fetch('/api/notifications', { signal: request.signal })
    .then(r => r.json());

  return defer({
    stats,                                    // 即座に利用可能
    recentUsers: recentUsersPromise,          // Suspense で遅延表示
    notifications: notificationsPromise,      // Suspense で遅延表示
  });
}

// defer を使ったコンポーネント
function Dashboard() {
  const { stats, recentUsers, notifications } = useLoaderData() as {
    stats: StatsData;
    recentUsers: Promise<User[]>;
    notifications: Promise<Notification[]>;
  };

  return (
    <div>
      {/* 即座に表示される */}
      <StatsPanel data={stats} />

      {/* データ到着まで Skeleton を表示 */}
      <Suspense fallback={<UserListSkeleton />}>
        <Await resolve={recentUsers} errorElement={<UserListError />}>
          {(users: User[]) => <RecentUsersList users={users} />}
        </Await>
      </Suspense>

      <Suspense fallback={<NotificationsSkeleton />}>
        <Await resolve={notifications} errorElement={<NotificationsError />}>
          {(items: Notification[]) => <NotificationsList items={items} />}
        </Await>
      </Suspense>
    </div>
  );
}
```

### 2.3 Action パターン（データ変更）

Action はフォーム送信やデータ変更を処理するためのメカニズムである。HTML の `<form>` の動作をエミュレートしつつ、JavaScript でインターセプトすることで、プログレッシブエンハンスメントを実現する。

```typescript
// === 基本的な Action ===
import type { ActionFunctionArgs } from 'react-router-dom';

// ユーザー作成 Action
async function createUserAction({ request }: ActionFunctionArgs) {
  const formData = await request.formData();

  // バリデーション
  const name = formData.get('name') as string;
  const email = formData.get('email') as string;
  const role = formData.get('role') as string;

  const errors: Record<string, string> = {};

  if (!name || name.length < 2) {
    errors.name = '名前は2文字以上で入力してください';
  }
  if (!email || !email.includes('@')) {
    errors.email = '有効なメールアドレスを入力してください';
  }
  if (!role || !['admin', 'user', 'viewer'].includes(role)) {
    errors.role = '有効なロールを選択してください';
  }

  if (Object.keys(errors).length > 0) {
    return json({ errors, values: { name, email, role } }, { status: 400 });
  }

  // API 呼び出し
  try {
    const response = await fetch('/api/users', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, role }),
    });

    if (!response.ok) {
      const error = await response.json();
      return json(
        { errors: { server: error.message }, values: { name, email, role } },
        { status: response.status }
      );
    }

    const user = await response.json();
    return redirect(`/users/${user.id}`);
  } catch (error) {
    return json(
      { errors: { server: 'ネットワークエラーが発生しました' }, values: { name, email, role } },
      { status: 500 }
    );
  }
}

// ユーザー編集 Action
async function userEditAction({ request, params }: ActionFunctionArgs) {
  const { userId } = params;
  const formData = await request.formData();
  const intent = formData.get('intent');

  // intent パターン: 1つの Action で複数の操作を処理
  switch (intent) {
    case 'update': {
      const name = formData.get('name') as string;
      const email = formData.get('email') as string;

      await fetch(`/api/users/${userId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email }),
      });

      return redirect(`/users/${userId}`);
    }

    case 'delete': {
      await fetch(`/api/users/${userId}`, { method: 'DELETE' });
      return redirect('/users');
    }

    case 'deactivate': {
      await fetch(`/api/users/${userId}/deactivate`, { method: 'POST' });
      return json({ success: true, message: 'ユーザーを無効化しました' });
    }

    default:
      return json({ error: 'Unknown intent' }, { status: 400 });
  }
}

// === Action を使うフォームコンポーネント ===
import { Form, useActionData, useNavigation } from 'react-router-dom';

function CreateUserForm() {
  const actionData = useActionData() as {
    errors?: Record<string, string>;
    values?: Record<string, string>;
  } | undefined;
  const navigation = useNavigation();
  const isSubmitting = navigation.state === 'submitting';

  return (
    <Form method="post" className="user-form">
      <div className="form-field">
        <label htmlFor="name">名前</label>
        <input
          id="name"
          name="name"
          type="text"
          defaultValue={actionData?.values?.name}
          aria-invalid={!!actionData?.errors?.name}
          aria-describedby={actionData?.errors?.name ? 'name-error' : undefined}
        />
        {actionData?.errors?.name && (
          <p id="name-error" className="error">{actionData.errors.name}</p>
        )}
      </div>

      <div className="form-field">
        <label htmlFor="email">メールアドレス</label>
        <input
          id="email"
          name="email"
          type="email"
          defaultValue={actionData?.values?.email}
          aria-invalid={!!actionData?.errors?.email}
          aria-describedby={actionData?.errors?.email ? 'email-error' : undefined}
        />
        {actionData?.errors?.email && (
          <p id="email-error" className="error">{actionData.errors.email}</p>
        )}
      </div>

      <div className="form-field">
        <label htmlFor="role">ロール</label>
        <select
          id="role"
          name="role"
          defaultValue={actionData?.values?.role ?? 'user'}
        >
          <option value="user">一般ユーザー</option>
          <option value="admin">管理者</option>
          <option value="viewer">閲覧者</option>
        </select>
        {actionData?.errors?.role && (
          <p className="error">{actionData.errors.role}</p>
        )}
      </div>

      {actionData?.errors?.server && (
        <div className="alert alert-error">{actionData.errors.server}</div>
      )}

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? '送信中...' : 'ユーザーを作成'}
      </button>
    </Form>
  );
}

// === useFetcher: ナビゲーションなしのデータ操作 ===
import { useFetcher } from 'react-router-dom';

function UserRow({ user }: { user: User }) {
  const fetcher = useFetcher();
  const isDeleting = fetcher.state === 'submitting'
    && fetcher.formData?.get('intent') === 'delete';

  if (isDeleting) {
    // Optimistic UI: 削除中は行を非表示
    return null;
  }

  return (
    <tr>
      <td>{user.name}</td>
      <td>{user.email}</td>
      <td>
        {/* fetcher.Form はナビゲーションを発生させない */}
        <fetcher.Form method="post" action={`/users/${user.id}/edit`}>
          <input type="hidden" name="intent" value="delete" />
          <button
            type="submit"
            onClick={(e) => {
              if (!confirm('本当に削除しますか？')) {
                e.preventDefault();
              }
            }}
          >
            削除
          </button>
        </fetcher.Form>
      </td>
    </tr>
  );
}

// === 複数の useFetcher を使った並行操作 ===
function TodoList({ todos }: { todos: Todo[] }) {
  return (
    <ul>
      {todos.map((todo) => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}

function TodoItem({ todo }: { todo: Todo }) {
  const toggleFetcher = useFetcher();
  const deleteFetcher = useFetcher();

  // Optimistic UI
  const isToggling = toggleFetcher.state !== 'idle';
  const isDeleting = deleteFetcher.state !== 'idle';
  const optimisticCompleted = isToggling ? !todo.completed : todo.completed;

  if (isDeleting) return null;

  return (
    <li style={{ opacity: isToggling ? 0.5 : 1 }}>
      <toggleFetcher.Form method="post" action={`/todos/${todo.id}`}>
        <input type="hidden" name="intent" value="toggle" />
        <input type="hidden" name="completed" value={String(!todo.completed)} />
        <button type="submit">
          {optimisticCompleted ? '✓' : '○'}
        </button>
      </toggleFetcher.Form>

      <span className={optimisticCompleted ? 'completed' : ''}>
        {todo.title}
      </span>

      <deleteFetcher.Form method="post" action={`/todos/${todo.id}`}>
        <input type="hidden" name="intent" value="delete" />
        <button type="submit">削除</button>
      </deleteFetcher.Form>
    </li>
  );
}
```

### 2.4 エラーハンドリング

```typescript
// === エラーバウンダリ ===
import { useRouteError, isRouteErrorResponse, Link } from 'react-router-dom';

function RootErrorBoundary() {
  const error = useRouteError();

  if (isRouteErrorResponse(error)) {
    // Response オブジェクトが throw された場合
    switch (error.status) {
      case 401:
        return (
          <div className="error-page">
            <h1>認証が必要です</h1>
            <p>このページにアクセスするにはログインが必要です。</p>
            <Link to="/login">ログインページへ</Link>
          </div>
        );
      case 403:
        return (
          <div className="error-page">
            <h1>アクセス権限がありません</h1>
            <p>このページを表示する権限がありません。</p>
            <Link to="/">ホームに戻る</Link>
          </div>
        );
      case 404:
        return (
          <div className="error-page">
            <h1>ページが見つかりません</h1>
            <p>お探しのページは存在しないか、移動した可能性があります。</p>
            <Link to="/">ホームに戻る</Link>
          </div>
        );
      default:
        return (
          <div className="error-page">
            <h1>エラーが発生しました</h1>
            <p>ステータスコード: {error.status}</p>
            <p>{error.statusText}</p>
            <Link to="/">ホームに戻る</Link>
          </div>
        );
    }
  }

  // 予期しないエラー（JavaScript エラーなど）
  console.error('Unexpected error:', error);

  return (
    <div className="error-page">
      <h1>予期しないエラーが発生しました</h1>
      <p>申し訳ございません。問題が発生しました。</p>
      <pre>{error instanceof Error ? error.message : '不明なエラー'}</pre>
      <button onClick={() => window.location.reload()}>
        ページを再読み込み
      </button>
    </div>
  );
}

// 子ルート専用のエラーバウンダリ
function UsersErrorBoundary() {
  const error = useRouteError();

  return (
    <div className="error-container">
      <h2>ユーザー情報の取得に失敗しました</h2>
      {isRouteErrorResponse(error) && error.status === 404 ? (
        <p>指定されたユーザーは存在しません。</p>
      ) : (
        <p>サーバーとの通信中にエラーが発生しました。しばらく経ってから再度お試しください。</p>
      )}
      <Link to="/users">ユーザー一覧に戻る</Link>
    </div>
  );
}
```

### 2.5 ナビゲーション状態の管理

```typescript
// === useNavigation: グローバルなナビゲーション状態 ===
import { useNavigation } from 'react-router-dom';

function GlobalLoadingIndicator() {
  const navigation = useNavigation();

  // navigation.state の値:
  // 'idle'       - 何もしていない
  // 'loading'    - loader が実行中（GET ナビゲーション）
  // 'submitting' - action が実行中（POST/PUT/DELETE）

  if (navigation.state === 'idle') return null;

  return (
    <div className="global-loading-bar">
      <div
        className="progress"
        style={{
          width: navigation.state === 'submitting' ? '30%' : '70%'
        }}
      />
    </div>
  );
}

// NProgress スタイルのローディングバー
function NProgressBar() {
  const navigation = useNavigation();
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (navigation.state !== 'idle') {
      setProgress(30);
      const timer1 = setTimeout(() => setProgress(60), 200);
      const timer2 = setTimeout(() => setProgress(80), 500);
      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
      };
    } else {
      setProgress(100);
      const timer = setTimeout(() => setProgress(0), 300);
      return () => clearTimeout(timer);
    }
  }, [navigation.state]);

  if (progress === 0) return null;

  return (
    <div
      className="nprogress-bar"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        height: '3px',
        width: `${progress}%`,
        backgroundColor: '#0070f3',
        transition: progress === 100 ? 'width 0.2s' : 'width 0.5s ease',
        zIndex: 9999,
      }}
    />
  );
}

// === useNavigation の詳細情報 ===
function DebugNavigation() {
  const navigation = useNavigation();

  return (
    <div className="debug-panel">
      <h4>Navigation State</h4>
      <dl>
        <dt>State</dt>
        <dd>{navigation.state}</dd>

        <dt>Location</dt>
        <dd>{navigation.location?.pathname ?? 'N/A'}</dd>

        <dt>Form Method</dt>
        <dd>{navigation.formMethod ?? 'N/A'}</dd>

        <dt>Form Action</dt>
        <dd>{navigation.formAction ?? 'N/A'}</dd>

        <dt>Form Data</dt>
        <dd>
          {navigation.formData
            ? JSON.stringify(Object.fromEntries(navigation.formData))
            : 'N/A'}
        </dd>
      </dl>
    </div>
  );
}
```

### 2.6 プログラマティックナビゲーション

```typescript
// === useNavigate フック ===
import { useNavigate, useSearchParams } from 'react-router-dom';

function UserProfile() {
  const navigate = useNavigate();

  const handleLogout = async () => {
    await api.auth.logout();
    // ログインページにリダイレクト
    navigate('/login', { replace: true }); // 履歴を置き換え（戻るで戻れなくする）
  };

  const handleEditClick = () => {
    // 相対パスでナビゲーション
    navigate('edit');  // 現在が /users/123 なら /users/123/edit へ
  };

  const handleBackClick = () => {
    navigate(-1); // ブラウザバックと同じ
  };

  const handleNavigateWithState = () => {
    // state を渡してナビゲーション
    navigate('/checkout', {
      state: {
        fromPage: 'cart',
        selectedItems: [1, 2, 3],
      },
    });
  };

  return (
    <div>
      <button onClick={handleEditClick}>編集</button>
      <button onClick={handleBackClick}>戻る</button>
      <button onClick={handleLogout}>ログアウト</button>
      <button onClick={handleNavigateWithState}>チェックアウト</button>
    </div>
  );
}

// === useSearchParams: クエリパラメータの管理 ===
function UserSearchPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const query = searchParams.get('q') ?? '';
  const page = Number(searchParams.get('page') ?? '1');
  const sort = searchParams.get('sort') ?? 'name';

  const handleSearch = (newQuery: string) => {
    setSearchParams((prev) => {
      prev.set('q', newQuery);
      prev.set('page', '1'); // 検索時はページをリセット
      return prev;
    });
  };

  const handlePageChange = (newPage: number) => {
    setSearchParams((prev) => {
      prev.set('page', String(newPage));
      return prev;
    });
  };

  const handleSortChange = (newSort: string) => {
    setSearchParams((prev) => {
      prev.set('sort', newSort);
      prev.set('page', '1');
      return prev;
    });
  };

  return (
    <div>
      <SearchInput value={query} onChange={handleSearch} />
      <SortSelector value={sort} onChange={handleSortChange} />
      <UserList query={query} page={page} sort={sort} />
      <Pagination page={page} onChange={handlePageChange} />
    </div>
  );
}

// === useLocation: 現在のロケーション情報 ===
import { useLocation } from 'react-router-dom';

function CheckoutPage() {
  const location = useLocation();
  const state = location.state as { fromPage?: string; selectedItems?: number[] } | null;

  // location オブジェクトの内容:
  // {
  //   pathname: '/checkout',
  //   search: '?coupon=SAVE20',
  //   hash: '#summary',
  //   state: { fromPage: 'cart', selectedItems: [1, 2, 3] },
  //   key: 'default'
  // }

  return (
    <div>
      {state?.fromPage === 'cart' && (
        <p>カートから遷移しました。選択アイテム: {state.selectedItems?.join(', ')}</p>
      )}
    </div>
  );
}

// === アナリティクス用のナビゲーション追跡 ===
function AnalyticsTracker() {
  const location = useLocation();

  useEffect(() => {
    // Google Analytics にページビューを送信
    if (typeof window.gtag !== 'undefined') {
      window.gtag('config', 'GA_MEASUREMENT_ID', {
        page_path: location.pathname + location.search,
      });
    }

    // ページタイトルを更新
    const routeTitles: Record<string, string> = {
      '/': 'ホーム',
      '/users': 'ユーザー一覧',
      '/settings': '設定',
      '/about': 'サービスについて',
    };
    document.title = routeTitles[location.pathname] ?? 'MyApp';
  }, [location]);

  return null; // UI はレンダリングしない
}
```

---

## 3. TanStack Router 徹底解説

### 3.1 型安全なルーティングの基本

TanStack Router は 100% 型安全なルーティングを提供するライブラリである。パスパラメータ、search params、loader データの全てが TypeScript の型システムによって厳密にチェックされる。

```typescript
// === TanStack Router: 完全な型安全ルーティング ===
import {
  createRouter,
  createRoute,
  createRootRoute,
  createRootRouteWithContext,
  Outlet,
  Link,
  useParams,
  useSearch,
  useLoaderData,
  useNavigate,
} from '@tanstack/react-router';

// ルートの型安全な定義
const rootRoute = createRootRoute({
  component: () => (
    <div>
      <nav>
        {/* Link の to は定義されたルートのみ許可 */}
        <Link to="/">Home</Link>
        <Link to="/users" search={{ page: 1 }}>Users</Link>
        <Link to="/about">About</Link>
      </nav>
      <Outlet />
    </div>
  ),
  notFoundComponent: () => <div>ページが見つかりません</div>,
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: () => <h1>ホームページ</h1>,
});

const aboutRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/about',
  component: () => <h1>このサービスについて</h1>,
});

// search params の型安全なバリデーション
import { z } from 'zod';

const usersSearchSchema = z.object({
  page: z.number().int().positive().default(1),
  limit: z.number().int().positive().max(100).default(20),
  q: z.string().optional(),
  sort: z.enum(['name', 'email', 'createdAt']).default('name'),
  order: z.enum(['asc', 'desc']).default('asc'),
});

type UsersSearch = z.infer<typeof usersSearchSchema>;

const usersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/users',
  validateSearch: usersSearchSchema,
  loaderDeps: ({ search }) => ({
    page: search.page,
    limit: search.limit,
    q: search.q,
    sort: search.sort,
    order: search.order,
  }),
  loader: async ({ deps }) => {
    const params = new URLSearchParams({
      page: String(deps.page),
      limit: String(deps.limit),
      sort: deps.sort,
      order: deps.order,
      ...(deps.q ? { q: deps.q } : {}),
    });
    const response = await fetch(`/api/users?${params}`);
    return response.json() as Promise<{
      users: User[];
      total: number;
      hasMore: boolean;
    }>;
  },
  component: UsersPage,
});

function UsersPage() {
  // 全てが型安全：search は UsersSearch 型
  const search = useSearch({ from: '/users' });
  const { users, total, hasMore } = useLoaderData({ from: '/users' });
  const navigate = useNavigate();

  const handlePageChange = (newPage: number) => {
    navigate({
      to: '/users',
      search: (prev) => ({
        ...prev,
        page: newPage,
      }),
    });
  };

  const handleSearchChange = (query: string) => {
    navigate({
      to: '/users',
      search: (prev) => ({
        ...prev,
        q: query || undefined,
        page: 1, // 検索時はページリセット
      }),
    });
  };

  return (
    <div>
      <h1>ユーザー一覧（全 {total} 件）</h1>
      <SearchInput value={search.q ?? ''} onChange={handleSearchChange} />
      <ul>
        {users.map((user) => (
          <li key={user.id}>
            {/* params の型チェック: userId は string であることが保証される */}
            <Link
              to="/users/$userId"
              params={{ userId: String(user.id) }}
            >
              {user.name}
            </Link>
          </li>
        ))}
      </ul>
      <Pagination
        page={search.page}
        hasMore={hasMore}
        onChange={handlePageChange}
      />
    </div>
  );
}

// 動的パラメータの型安全なルート
const userDetailRoute = createRoute({
  getParentRoute: () => usersRoute,
  path: '$userId',
  loader: async ({ params }) => {
    // params.userId は自動的に string 型
    const response = await fetch(`/api/users/${params.userId}`);
    if (!response.ok) throw new Error('User not found');
    return response.json() as Promise<User>;
  },
  component: UserDetailPage,
});

function UserDetailPage() {
  const { userId } = useParams({ from: '/users/$userId' });
  const user = useLoaderData({ from: '/users/$userId' });

  return (
    <div>
      <h1>{user.name}</h1>
      <p>ID: {userId}</p>
      <p>Email: {user.email}</p>
    </div>
  );
}

// ルートツリーの構築
const routeTree = rootRoute.addChildren([
  indexRoute,
  aboutRoute,
  usersRoute.addChildren([
    userDetailRoute,
  ]),
]);

// ルーターの作成
const router = createRouter({
  routeTree,
  defaultPreload: 'intent',    // hover 時にプリロード
  defaultPreloadDelay: 100,    // プリロード開始までの遅延
});

// 型安全のための型宣言
declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}

// アプリケーションのエントリーポイント
function App() {
  return <RouterProvider router={router} />;
}
```

### 3.2 コンテキスト付きルーティング

```typescript
// === 認証コンテキストの注入 ===
import { createRootRouteWithContext } from '@tanstack/react-router';

interface RouterContext {
  auth: {
    user: User | null;
    isAuthenticated: boolean;
    login: (credentials: Credentials) => Promise<void>;
    logout: () => Promise<void>;
  };
  queryClient: QueryClient;
}

const rootRoute = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
});

// 認証ガード付きルート
const protectedRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/dashboard',
  beforeLoad: async ({ context }) => {
    // コンテキストから認証情報を取得
    if (!context.auth.isAuthenticated) {
      throw redirect({
        to: '/login',
        search: { redirect: '/dashboard' },
      });
    }
  },
  loader: async ({ context }) => {
    // TanStack Query と連携
    return context.queryClient.ensureQueryData({
      queryKey: ['dashboard'],
      queryFn: fetchDashboardData,
    });
  },
  component: DashboardPage,
});

// ルーターにコンテキストを注入
const router = createRouter({
  routeTree,
  context: {
    auth: undefined!, // アプリのレンダリング時に提供
    queryClient,
  },
});

function App() {
  const auth = useAuth();
  return <RouterProvider router={router} context={{ auth, queryClient }} />;
}
```

### 3.3 TanStack Router と TanStack Query の統合

```typescript
// === TanStack Query との完全統合 ===
import { queryOptions } from '@tanstack/react-query';

// Query オプションを定義
const usersQueryOptions = (search: UsersSearch) =>
  queryOptions({
    queryKey: ['users', search],
    queryFn: () => fetchUsers(search),
    staleTime: 5 * 60 * 1000, // 5分間キャッシュ
  });

const userQueryOptions = (userId: string) =>
  queryOptions({
    queryKey: ['users', userId],
    queryFn: () => fetchUser(userId),
    staleTime: 5 * 60 * 1000,
  });

// ルート定義で Query を活用
const usersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/users',
  validateSearch: usersSearchSchema,
  loaderDeps: ({ search }) => search,
  loader: async ({ context, deps }) => {
    // ensureQueryData でキャッシュがあればそれを返し、なければフェッチ
    await context.queryClient.ensureQueryData(usersQueryOptions(deps));
  },
  component: UsersPage,
});

function UsersPage() {
  const search = useSearch({ from: '/users' });
  // コンポーネントでは useSuspenseQuery で型安全にデータを取得
  const { data } = useSuspenseQuery(usersQueryOptions(search));

  return (
    <div>
      {data.users.map((user) => (
        <UserCard key={user.id} user={user} />
      ))}
    </div>
  );
}

// Prefetch の活用
function UserCard({ user }: { user: User }) {
  const queryClient = useQueryClient();

  return (
    <Link
      to="/users/$userId"
      params={{ userId: String(user.id) }}
      // hover 時にデータをプリフェッチ
      preload="intent"
      onMouseEnter={() => {
        queryClient.prefetchQuery(userQueryOptions(String(user.id)));
      }}
    >
      {user.name}
    </Link>
  );
}
```

---

## 4. コード分割とパフォーマンス最適化

### 4.1 ルートベースのコード分割

```typescript
// === React.lazy + Suspense（基本パターン）===
import { lazy, Suspense } from 'react';

const Settings = lazy(() => import('./pages/Settings'));
const Analytics = lazy(() => import('./pages/Analytics'));
const AdminDashboard = lazy(() => import('./pages/AdminDashboard'));

function App() {
  return (
    <Suspense fallback={<PageSkeleton />}>
      <Routes>
        <Route path="/settings" element={<Settings />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/admin/*" element={<AdminDashboard />} />
      </Routes>
    </Suspense>
  );
}

// === React Router の lazy（推奨：loader/action も分割可能）===
const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: [
      {
        path: 'settings',
        lazy: async () => {
          // コンポーネントだけでなく loader/action も分割
          const { Settings, settingsLoader, settingsAction } =
            await import('./pages/Settings');
          return {
            Component: Settings,
            loader: settingsLoader,
            action: settingsAction,
          };
        },
      },
      {
        path: 'analytics',
        lazy: async () => {
          const { Analytics, analyticsLoader } =
            await import('./pages/Analytics');
          return {
            Component: Analytics,
            loader: analyticsLoader,
          };
        },
      },
      {
        path: 'admin',
        lazy: async () => {
          const { AdminLayout, adminLoader } =
            await import('./pages/admin/AdminLayout');
          return {
            Component: AdminLayout,
            loader: adminLoader,
          };
        },
        children: [
          {
            index: true,
            lazy: async () => {
              const { AdminDashboard, dashboardLoader } =
                await import('./pages/admin/Dashboard');
              return {
                Component: AdminDashboard,
                loader: dashboardLoader,
              };
            },
          },
          {
            path: 'users',
            lazy: async () => {
              const { AdminUsers, adminUsersLoader } =
                await import('./pages/admin/Users');
              return {
                Component: AdminUsers,
                loader: adminUsersLoader,
              };
            },
          },
        ],
      },
    ],
  },
]);

// === TanStack Router のコード分割 ===
// TanStack Router ではルートファイルの自動分割が可能
const usersRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/users',
}).lazy(() => import('./routes/users.lazy').then((m) => m.Route));

// routes/users.lazy.ts
import { createLazyRoute } from '@tanstack/react-router';

export const Route = createLazyRoute('/users')({
  component: UsersPage,
  pendingComponent: UsersSkeleton,
  errorComponent: UsersError,
});
```

### 4.2 プリロード戦略

```typescript
// === hover 時のプリロード（最も効果的）===
import { usePrefetchableLink } from './hooks/usePrefetchableLink';

function NavLink({ to, children }: { to: string; children: React.ReactNode }) {
  const prefetch = () => {
    // 方法1: link[rel=prefetch] でチャンクを先読み
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.as = 'script';
    link.href = `/assets/pages/${to.replace(/^\//, '')}.js`;
    document.head.appendChild(link);
  };

  return (
    <Link to={to} onMouseEnter={prefetch} onFocus={prefetch}>
      {children}
    </Link>
  );
}

// === React Router でのプリロード ===
// データルーティングでは router.preloadRoute を使用
function PrefetchableLink({ to, children }: { to: string; children: React.ReactNode }) {
  const router = useRouter();

  const handleMouseEnter = () => {
    // ルートの lazy コンポーネントと loader を事前に実行
    router.preloadRoute(to);
  };

  return (
    <Link to={to} onMouseEnter={handleMouseEnter}>
      {children}
    </Link>
  );
}

// === TanStack Router の preload 設定 ===
const router = createRouter({
  routeTree,
  defaultPreload: 'intent',   // 'intent' = hover/focus 時にプリロード
  defaultPreloadDelay: 50,     // 50ms の遅延（素早いマウス通過を無視）
  defaultPreloadStaleTime: 30_000, // プリロードデータの有効期限（30秒）
});

// 個別のリンクでプリロードを制御
<Link to="/users" preload="viewport">  {/* ビューポートに入ったらプリロード */}
  Users
</Link>

<Link to="/settings" preload="intent">  {/* hover/focus 時にプリロード */}
  Settings
</Link>

<Link to="/about" preload={false}>  {/* プリロードしない */}
  About
</Link>

// === Intersection Observer を使った Viewport プリロード ===
function ViewportPrefetchLink({ to, children }: { to: string; children: React.ReactNode }) {
  const ref = useRef<HTMLAnchorElement>(null);
  const [prefetched, setPrefetched] = useState(false);

  useEffect(() => {
    if (!ref.current || prefetched) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          // ビューポートに入ったらプリフェッチ
          const link = document.createElement('link');
          link.rel = 'prefetch';
          link.href = `/assets/pages/${to.replace(/^\//, '')}.js`;
          document.head.appendChild(link);
          setPrefetched(true);
          observer.disconnect();
        }
      },
      { rootMargin: '100px' } // 100px 手前からプリフェッチ
    );

    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [to, prefetched]);

  return (
    <Link ref={ref} to={to}>
      {children}
    </Link>
  );
}
```

### 4.3 スクロール位置の管理

```typescript
// === React Router のスクロール復元 ===
import { ScrollRestoration } from 'react-router-dom';

function RootLayout() {
  return (
    <div>
      <Header />
      <main>
        <Outlet />
      </main>
      <Footer />
      {/* ナビゲーション時のスクロール位置を自動管理 */}
      <ScrollRestoration
        getKey={(location) => {
          // URL パスをキーにしてスクロール位置を復元
          // デフォルトは location.key（ブラウザ履歴エントリごと）
          return location.pathname;
        }}
      />
    </div>
  );
}

// === カスタムスクロール管理 ===
function useScrollToTop() {
  const location = useLocation();

  useEffect(() => {
    // ハッシュがない場合はページトップにスクロール
    if (!location.hash) {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
      // ハッシュがある場合は該当要素にスクロール
      const element = document.getElementById(location.hash.slice(1));
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
  }, [location.pathname, location.hash]);
}

// 無限スクロールリストでのスクロール位置保持
function InfiniteUserList() {
  const [scrollPosition, setScrollPosition] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const navigate = useNavigate();

  // スクロール位置を state に保存
  useEffect(() => {
    const handleScroll = () => {
      if (listRef.current) {
        setScrollPosition(listRef.current.scrollTop);
      }
    };

    const list = listRef.current;
    list?.addEventListener('scroll', handleScroll);
    return () => list?.removeEventListener('scroll', handleScroll);
  }, []);

  // 戻ってきた時にスクロール位置を復元
  useEffect(() => {
    const savedPosition = location.state?.scrollPosition;
    if (savedPosition && listRef.current) {
      listRef.current.scrollTop = savedPosition;
    }
  }, [location.state]);

  const handleItemClick = (userId: string) => {
    navigate(`/users/${userId}`, {
      state: { scrollPosition },
    });
  };

  return (
    <div ref={listRef} style={{ height: '100vh', overflow: 'auto' }}>
      {users.map((user) => (
        <div key={user.id} onClick={() => handleItemClick(user.id)}>
          {user.name}
        </div>
      ))}
    </div>
  );
}
```
