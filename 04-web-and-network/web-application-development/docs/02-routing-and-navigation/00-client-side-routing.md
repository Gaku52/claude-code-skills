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

---

## 5. 認証・認可とルートガード

### 5.1 認証パターンの実装

ルーティングにおける認証は、アプリケーションのセキュリティを確保する上で最も重要な要素の一つである。クライアントサイドの認証ガードはあくまでUX向上のためのものであり、真のセキュリティはサーバーサイドで担保する必要がある点に注意が必要である。

```typescript
// === React Router での認証パターン ===

// 方法1: Loader ベースの認証ガード（推奨）
async function protectedLoader({ request }: LoaderFunctionArgs) {
  const token = getAuthToken();

  if (!token) {
    const url = new URL(request.url);
    return redirect(`/login?redirectTo=${encodeURIComponent(url.pathname + url.search)}`);
  }

  // トークンの有効性をサーバーに確認
  try {
    const response = await fetch('/api/auth/verify', {
      headers: { Authorization: `Bearer ${token}` },
      signal: request.signal,
    });

    if (!response.ok) {
      // トークン無効: ログインページにリダイレクト
      clearAuthToken();
      return redirect('/login?reason=session_expired');
    }

    const user = await response.json();
    return { user };
  } catch (error) {
    throw new Response('Authentication service unavailable', { status: 503 });
  }
}

// 方法2: レイアウトルートによる認証ガード
const router = createBrowserRouter([
  {
    path: '/',
    element: <PublicLayout />,
    children: [
      { index: true, element: <LandingPage /> },
      { path: 'login', element: <LoginPage />, action: loginAction },
      { path: 'register', element: <RegisterPage />, action: registerAction },
    ],
  },
  {
    path: '/app',
    element: <AuthenticatedLayout />,
    loader: protectedLoader,  // 全子ルートに認証を適用
    errorElement: <AuthErrorBoundary />,
    children: [
      { index: true, element: <Dashboard /> },
      { path: 'profile', element: <Profile /> },
      {
        path: 'admin',
        loader: adminRoleLoader,  // 追加の権限チェック
        element: <AdminPanel />,
        children: [
          { index: true, element: <AdminOverview /> },
          { path: 'users', element: <AdminUsers /> },
        ],
      },
    ],
  },
]);

// ロールベースアクセス制御（RBAC）
async function adminRoleLoader({ request }: LoaderFunctionArgs) {
  const parentData = await protectedLoader({ request } as LoaderFunctionArgs);

  if ('user' in parentData && parentData.user.role !== 'admin') {
    throw new Response('Forbidden: Admin access required', { status: 403 });
  }

  return parentData;
}

// === ログインアクション ===
async function loginAction({ request }: ActionFunctionArgs) {
  const formData = await request.formData();
  const email = formData.get('email') as string;
  const password = formData.get('password') as string;

  const errors: Record<string, string> = {};
  if (!email) errors.email = 'メールアドレスを入力してください';
  if (!password) errors.password = 'パスワードを入力してください';

  if (Object.keys(errors).length > 0) {
    return json({ errors }, { status: 400 });
  }

  try {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (response.status === 401) {
      return json({
        errors: { form: 'メールアドレスまたはパスワードが正しくありません' },
      }, { status: 401 });
    }

    if (!response.ok) {
      return json({
        errors: { form: 'ログインに失敗しました。しばらく経ってから再度お試しください' },
      }, { status: 500 });
    }

    const { token } = await response.json();
    setAuthToken(token);

    // ログイン前のページにリダイレクト
    const url = new URL(request.url);
    const redirectTo = url.searchParams.get('redirectTo') ?? '/app';
    return redirect(redirectTo);
  } catch (error) {
    return json({
      errors: { form: 'ネットワークエラーが発生しました' },
    }, { status: 500 });
  }
}
```

### 5.2 離脱防止（Unsaved Changes Guard）

```typescript
// === useBlocker を使った離脱防止 ===
import { useBlocker } from 'react-router-dom';

function EditForm() {
  const [isDirty, setIsDirty] = useState(false);
  const [formData, setFormData] = useState({ name: '', email: '' });

  // フォームに変更がある場合にナビゲーションをブロック
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      isDirty && currentLocation.pathname !== nextLocation.pathname
  );

  return (
    <div>
      <Form
        method="post"
        onChange={() => setIsDirty(true)}
        onSubmit={() => setIsDirty(false)}
      >
        <input
          name="name"
          value={formData.name}
          onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
        />
        <input
          name="email"
          value={formData.email}
          onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
        />
        <button type="submit">保存</button>
      </Form>

      {/* ブロック時の確認ダイアログ */}
      {blocker.state === 'blocked' && (
        <div className="modal-overlay">
          <div className="modal">
            <h2>変更が保存されていません</h2>
            <p>このページを離れると、変更内容が失われます。</p>
            <div className="modal-actions">
              <button onClick={() => blocker.proceed()}>
                変更を破棄して移動
              </button>
              <button onClick={() => blocker.reset()}>
                このページに留まる
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// === beforeunload との組み合わせ（ブラウザタブを閉じる場合）===
function useUnsavedChangesWarning(isDirty: boolean) {
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = ''; // ブラウザの標準ダイアログが表示される
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);
}

function EditPage() {
  const [isDirty, setIsDirty] = useState(false);

  // ブラウザのタブを閉じる・リロード時の警告
  useUnsavedChangesWarning(isDirty);

  // React Router のナビゲーション時の警告
  const blocker = useBlocker(isDirty);

  return (
    <div>
      {/* フォーム内容 */}
    </div>
  );
}
```

---

## 6. ページ遷移アニメーション

### 6.1 View Transitions API の活用

```typescript
// === View Transitions API（モダンブラウザ対応）===
import { useNavigate } from 'react-router-dom';

function AnimatedLink({ to, children }: { to: string; children: React.ReactNode }) {
  const navigate = useNavigate();

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();

    if (document.startViewTransition) {
      // View Transitions API がサポートされている場合
      document.startViewTransition(() => {
        navigate(to);
      });
    } else {
      // フォールバック：通常のナビゲーション
      navigate(to);
    }
  };

  return (
    <a href={to} onClick={handleClick}>
      {children}
    </a>
  );
}

// CSS で遷移アニメーションを定義
// styles.css:
// ::view-transition-old(root) {
//   animation: fade-out 0.15s ease-in;
// }
// ::view-transition-new(root) {
//   animation: fade-in 0.15s ease-out;
// }
// @keyframes fade-out {
//   from { opacity: 1; }
//   to { opacity: 0; }
// }
// @keyframes fade-in {
//   from { opacity: 0; }
//   to { opacity: 1; }
// }

// === framer-motion を使ったページ遷移 ===
import { AnimatePresence, motion } from 'framer-motion';
import { useLocation, Outlet } from 'react-router-dom';

function AnimatedLayout() {
  const location = useLocation();

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={location.pathname}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        transition={{ duration: 0.2, ease: 'easeInOut' }}
      >
        <Outlet />
      </motion.div>
    </AnimatePresence>
  );
}

// ルート定義でAnimatedLayoutを使用
const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    children: [
      {
        element: <AnimatedLayout />,
        children: [
          { index: true, element: <Home /> },
          { path: 'about', element: <About /> },
          { path: 'users', element: <Users /> },
        ],
      },
    ],
  },
]);

// === スライド方向を制御するアニメーション ===
function DirectionalLayout() {
  const location = useLocation();
  const [direction, setDirection] = useState(1);
  const previousPath = useRef(location.pathname);

  // パスの深さに基づいてスライド方向を決定
  useEffect(() => {
    const prevDepth = previousPath.current.split('/').length;
    const currentDepth = location.pathname.split('/').length;
    setDirection(currentDepth >= prevDepth ? 1 : -1);
    previousPath.current = location.pathname;
  }, [location.pathname]);

  const variants = {
    enter: (dir: number) => ({
      x: dir > 0 ? '100%' : '-100%',
      opacity: 0,
    }),
    center: {
      x: 0,
      opacity: 1,
    },
    exit: (dir: number) => ({
      x: dir > 0 ? '-100%' : '100%',
      opacity: 0,
    }),
  };

  return (
    <AnimatePresence mode="wait" custom={direction}>
      <motion.div
        key={location.pathname}
        custom={direction}
        variants={variants}
        initial="enter"
        animate="center"
        exit="exit"
        transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
      >
        <Outlet />
      </motion.div>
    </AnimatePresence>
  );
}
```

---

## 7. アンチパターンと注意点

### 7.1 よくあるアンチパターン

```typescript
// === アンチパターン 1: useEffect 内でのナビゲーション（条件付きリダイレクト）===

// NG: コンポーネントレベルでのリダイレクト
function ProtectedPage() {
  const navigate = useNavigate();
  const { user } = useAuth();

  // 問題: コンポーネントがマウントされてからリダイレクトされるため、
  // 一瞬だけ保護されたコンテンツが表示される（フラッシュ）
  useEffect(() => {
    if (!user) {
      navigate('/login');
    }
  }, [user, navigate]);

  if (!user) return null; // フラッシュ防止のため null を返すが不十分

  return <div>Protected Content</div>;
}

// OK: Loader レベルでのリダイレクト
async function protectedLoader({ request }: LoaderFunctionArgs) {
  const user = await getUser();
  if (!user) {
    return redirect('/login');
  }
  return { user };
}
// コンポーネントがレンダリングされる前にリダイレクトされる


// === アンチパターン 2: 状態管理とURLの不整合 ===

// NG: フィルター状態を useState で管理
function UserList() {
  const [filter, setFilter] = useState('active');
  const [page, setPage] = useState(1);
  // 問題: URLを共有してもフィルター状態が復元されない
  // ブラウザバックでも状態が復元されない
  return <div>...</div>;
}

// OK: フィルター状態をURLパラメータで管理
function UserList() {
  const [searchParams, setSearchParams] = useSearchParams();
  const filter = searchParams.get('filter') ?? 'active';
  const page = Number(searchParams.get('page') ?? '1');
  // URL: /users?filter=active&page=1
  // URLを共有すれば同じ状態が復元される
  // ブラウザバックでも正しく動作する
  return <div>...</div>;
}


// === アンチパターン 3: ハードコードされたパス ===

// NG: パスを文字列リテラルで散在させる
function Navigation() {
  return (
    <nav>
      <Link to="/users">Users</Link>
      <Link to="/users/new">New User</Link>
      <Link to="/settings/profile">Profile Settings</Link>
    </nav>
  );
}

// OK: ルートパスを定数化して一元管理
const ROUTES = {
  HOME: '/',
  USERS: {
    LIST: '/users',
    NEW: '/users/new',
    DETAIL: (id: string) => `/users/${id}`,
    EDIT: (id: string) => `/users/${id}/edit`,
  },
  SETTINGS: {
    ROOT: '/settings',
    PROFILE: '/settings/profile',
    NOTIFICATIONS: '/settings/notifications',
  },
  ADMIN: {
    ROOT: '/admin',
    USERS: '/admin/users',
    ANALYTICS: '/admin/analytics',
  },
} as const;

function Navigation() {
  return (
    <nav>
      <Link to={ROUTES.USERS.LIST}>Users</Link>
      <Link to={ROUTES.USERS.NEW}>New User</Link>
      <Link to={ROUTES.SETTINGS.PROFILE}>Profile Settings</Link>
    </nav>
  );
}

// TanStack Router なら型安全にパスを指定できるため、この問題は発生しない


// === アンチパターン 4: window.location での直接ナビゲーション ===

// NG: window.location を使うとSPAの状態が全て失われる
function LogoutButton() {
  const handleLogout = () => {
    clearToken();
    window.location.href = '/login'; // フルページリロードが発生
  };
  return <button onClick={handleLogout}>ログアウト</button>;
}

// OK: ルーターのナビゲーション機能を使用
function LogoutButton() {
  const navigate = useNavigate();
  const handleLogout = async () => {
    await api.auth.logout();
    clearToken();
    navigate('/login', { replace: true }); // SPA内でナビゲーション
  };
  return <button onClick={handleLogout}>ログアウト</button>;
}
// 例外: 外部サイトへのリダイレクト（OAuth callback等）は window.location を使う


// === アンチパターン 5: 過度にネストされたルート ===

// NG: 不必要に深いネスト
const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout1 />,
    children: [{
      element: <Layout2 />,
      children: [{
        element: <Layout3 />,
        children: [{
          path: 'users',
          element: <Users />,
          // 3段階の Outlet を経由する必要がある
        }],
      }],
    }],
  },
]);

// OK: フラットな構造で必要なレイアウトだけをネスト
const router = createBrowserRouter([
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { index: true, element: <Home /> },
      { path: 'users', element: <Users /> },
      {
        path: 'admin',
        element: <AdminLayout />,
        children: [
          { index: true, element: <AdminDashboard /> },
        ],
      },
    ],
  },
]);
```

### 7.2 セキュリティ上の注意点

```
クライアントサイドルーティングのセキュリティ:

1. クライアントサイドの認証は信頼しない
   - ルートガードはUX向上のためのもの
   - 真のアクセス制御はAPIサーバー側で実装
   - JWT の有効期限チェックはサーバーでも行う
   - フロントエンドのロール判定は参考情報に過ぎない

2. パスパラメータのインジェクション対策
   - :userId に悪意のある値が入る可能性を考慮
   - loader/action 内でパラメータをバリデーション
   - API に渡す前にサニタイズ

3. Open Redirect 防止
   - redirectTo パラメータの検証
   - 外部URLへのリダイレクトを禁止
   - ホワイトリストによるリダイレクト先の制限

4. 機密情報のURL露出防止
   - パスパラメータにトークンや秘密情報を含めない
   - search params に個人情報を含めない
   - state を使って機密データを渡す（ただしブラウザ履歴に残る）

5. CSRF 対策
   - Action でのフォーム送信時にCSRFトークンを検証
   - SameSite Cookie の設定
```

```typescript
// === Open Redirect 防止の実装例 ===
function safeRedirect(to: string, defaultRedirect: string = '/'): string {
  // 安全なリダイレクト先かどうかを検証
  if (
    !to ||
    !to.startsWith('/') ||    // 相対パスのみ許可
    to.startsWith('//') ||    // プロトコル相対URLを拒否
    to.includes('\\')         // バックスラッシュを拒否
  ) {
    return defaultRedirect;
  }

  // 許可されたパスのプレフィックスをチェック
  const allowedPrefixes = ['/app', '/dashboard', '/settings', '/users'];
  const isAllowed = allowedPrefixes.some((prefix) => to.startsWith(prefix));

  return isAllowed ? to : defaultRedirect;
}

// 使用例
async function loginAction({ request }: ActionFunctionArgs) {
  // ... ログイン処理 ...

  const url = new URL(request.url);
  const redirectTo = safeRedirect(
    url.searchParams.get('redirectTo') ?? '',
    '/app'
  );

  return redirect(redirectTo);
}

// === パラメータバリデーション ===
import { z } from 'zod';

const userIdSchema = z.string().regex(/^\d+$/, 'User ID must be numeric');

async function userLoader({ params }: LoaderFunctionArgs) {
  const result = userIdSchema.safeParse(params.userId);

  if (!result.success) {
    throw new Response('Invalid user ID format', { status: 400 });
  }

  const response = await fetch(`/api/users/${result.data}`);
  if (!response.ok) {
    throw new Response('User not found', { status: 404 });
  }

  return response.json();
}
```

---

## 8. テスト戦略

### 8.1 ルーティングのテスト

```typescript
// === React Router のテスト ===
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { createMemoryRouter, RouterProvider } from 'react-router-dom';

// テスト用のルーターファクトリ
function createTestRouter(initialEntries: string[] = ['/']) {
  return createMemoryRouter(
    [
      {
        path: '/',
        element: <RootLayout />,
        children: [
          { index: true, element: <Home /> },
          {
            path: 'users',
            element: <UserList />,
            loader: () => [
              { id: 1, name: 'Alice' },
              { id: 2, name: 'Bob' },
            ],
          },
          {
            path: 'users/:userId',
            element: <UserDetail />,
            loader: ({ params }) => ({
              id: params.userId,
              name: `User ${params.userId}`,
            }),
          },
          {
            path: 'login',
            element: <LoginPage />,
            action: loginAction,
          },
        ],
      },
    ],
    { initialEntries }
  );
}

// テストケース
describe('Routing', () => {
  it('should render the home page at /', () => {
    const router = createTestRouter(['/']);
    render(<RouterProvider router={router} />);

    expect(screen.getByText('ホームページ')).toBeInTheDocument();
  });

  it('should navigate to users page', async () => {
    const router = createTestRouter(['/']);
    render(<RouterProvider router={router} />);

    const user = userEvent.setup();
    await user.click(screen.getByRole('link', { name: 'Users' }));

    await waitFor(() => {
      expect(screen.getByText('Alice')).toBeInTheDocument();
      expect(screen.getByText('Bob')).toBeInTheDocument();
    });
  });

  it('should display user detail with correct params', async () => {
    const router = createTestRouter(['/users/42']);
    render(<RouterProvider router={router} />);

    await waitFor(() => {
      expect(screen.getByText('User 42')).toBeInTheDocument();
    });
  });

  it('should handle loader errors', async () => {
    const failingRouter = createMemoryRouter(
      [
        {
          path: '/',
          element: <div />,
          errorElement: <div>エラーが発生しました</div>,
          loader: () => {
            throw new Response('Not Found', { status: 404 });
          },
        },
      ],
      { initialEntries: ['/'] }
    );

    render(<RouterProvider router={failingRouter} />);

    await waitFor(() => {
      expect(screen.getByText('エラーが発生しました')).toBeInTheDocument();
    });
  });

  it('should handle form submission', async () => {
    const router = createTestRouter(['/login']);
    render(<RouterProvider router={router} />);

    const user = userEvent.setup();
    await user.type(screen.getByLabelText('メール'), 'test@example.com');
    await user.type(screen.getByLabelText('パスワード'), 'password123');
    await user.click(screen.getByRole('button', { name: 'ログイン' }));

    await waitFor(() => {
      // ログイン成功後のリダイレクトを確認
      expect(screen.getByText('ホームページ')).toBeInTheDocument();
    });
  });
});

// === Loader/Action の単体テスト ===
describe('usersLoader', () => {
  it('should fetch users with pagination', async () => {
    // API モック
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve([
        { id: 1, name: 'Alice' },
        { id: 2, name: 'Bob' },
      ]),
    });

    const request = new Request('http://localhost/users?page=2&limit=10');
    const result = await usersLoader({ request, params: {} } as LoaderFunctionArgs);

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('page=2'),
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
    expect(result).toHaveLength(2);
  });

  it('should throw on API error', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
    });

    const request = new Request('http://localhost/users');

    await expect(
      usersLoader({ request, params: {} } as LoaderFunctionArgs)
    ).rejects.toBeInstanceOf(Response);
  });
});
```

---

## 9. ルーティングライブラリ比較と選定ガイド

### 9.1 詳細比較表

| 項目 | React Router v6 | TanStack Router | Next.js App Router |
|------|-----------------|-----------------|---------------------|
| **型安全性** | 部分的（手動型付け） | 完全（自動推論） | 部分的 |
| **Search Params** | 手動管理（useSearchParams） | 組み込みバリデーション | nuqs 推奨 |
| **データ取得** | loader/action | loader/beforeLoad | Server Components |
| **コード分割** | lazy() | lazy routes + 自動分割 | 自動（ファイルベース） |
| **SSR対応** | Remix 経由 | SSR サポートあり | ファーストクラス |
| **ファイルベース** | 非対応（手動定義） | プラグインで対応 | デフォルト |
| **バンドルサイズ** | ~14KB (gzip) | ~12KB (gzip) | Next.js に含まれる |
| **学習コスト** | 低〜中 | 中 | 中〜高 |
| **エコシステム** | 最大（最も広く使用） | 成長中 | Next.js エコシステム |
| **プリロード** | 手動実装 | 組み込み（intent/viewport） | 自動 |
| **Devtools** | なし | 公式 Devtools あり | Next.js Devtools |
| **ミドルウェア** | loader で実現 | beforeLoad | middleware.ts |
| **エラー処理** | errorElement | errorComponent | error.tsx |
| **Pending UI** | useNavigation | pendingComponent | loading.tsx |
| **Optimistic UI** | useFetcher | 手動実装 | useOptimistic |
| **初回リリース** | 2014年（v1） | 2022年 | 2023年（App Router） |

### 9.2 選定フローチャート

```
ルーティングライブラリの選定:

Q1: フルスタックフレームワークを使用する？
  Yes → Q2: Next.js を選択？
    Yes → Next.js App Router を使用
    No  → Q3: Remix を選択？
      Yes → React Router（Remix 内蔵）を使用
      No  → 他のフレームワークのルーターを使用

  No → Q4: 型安全性をどの程度重視する？
    最重要 → TanStack Router
      - search params の型安全バリデーションが必要
      - パスパラメータの自動型推論が必要
      - TypeScript を最大限活用したい

    重要だが必須ではない → Q5: プロジェクトの規模は？
      大規模・長期運用 → TanStack Router
        - 型安全性の恩恵が大きい
        - リファクタリング時の安全性
        - 成長するコードベースへの対応

      中小規模 → React Router v6
        - 豊富なドキュメントとコミュニティ
        - 学習コストが低い
        - 多くのチュートリアルやサンプルが存在

    重視しない → React Router v6
      - 最も広く使われている標準
      - チームの採用が容易
```

### 9.3 マイグレーション戦略

```typescript
// === React Router v5 → v6 マイグレーション ===

// v5 の書き方
import { Switch, Route, useHistory, useParams } from 'react-router-dom';

function AppV5() {
  return (
    <Switch>
      <Route exact path="/" component={Home} />
      <Route path="/users/:userId" component={UserDetail} />
      <Route component={NotFound} />  {/* 404 */}
    </Switch>
  );
}

function UserDetailV5() {
  const { userId } = useParams<{ userId: string }>();
  const history = useHistory();

  const handleBack = () => history.push('/users');

  return <div>User: {userId}</div>;
}

// v6 の書き方
import { Routes, Route, useNavigate, useParams } from 'react-router-dom';

function AppV6() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/users/:userId" element={<UserDetail />} />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

function UserDetailV6() {
  const { userId } = useParams();
  const navigate = useNavigate();

  const handleBack = () => navigate('/users');

  return <div>User: {userId}</div>;
}

// マイグレーションチェックリスト:
// 1. Switch → Routes に変更
// 2. component/render → element に変更（JSXで渡す）
// 3. exact を削除（v6 はデフォルトで exact マッチ）
// 4. useHistory → useNavigate に変更
// 5. history.push() → navigate() に変更
// 6. history.replace() → navigate(path, { replace: true }) に変更
// 7. Redirect → Navigate コンポーネントに変更
// 8. 入れ子ルートの構造を Outlet ベースに変更
// 9. withRouter HOC を削除し、フックに置き換え
// 10. activeClassName → className コールバックに変更（NavLink）
```

---

## 10. トラブルシューティング

### 10.1 よくある問題と解決策

```
問題1: ブラウザで直接URLにアクセスすると404が返る
  原因: サーバーがSPA fallbackに対応していない
  解決策:
  - Nginx: try_files $uri $uri/ /index.html;
  - Vercel: vercel.json に rewrites を追加
  - Netlify: _redirects ファイルを public/ に配置
  - Express: app.get('*', (req, res) => res.sendFile('index.html'));
  - 開発環境: Vite は自動対応（historyApiFallback）

問題2: ページ遷移後にスクロール位置がリセットされない
  原因: ScrollRestoration が設定されていない
  解決策:
  - React Router: <ScrollRestoration /> コンポーネントを追加
  - TanStack Router: scrollRestoration オプションを設定
  - 手動: useEffect で window.scrollTo(0, 0) を実行

問題3: ネストされたルートで Outlet が表示されない
  原因: 親ルートに <Outlet /> コンポーネントが配置されていない
  解決策:
  - 親ルートのコンポーネントに <Outlet /> を追加
  - レイアウトコンポーネントの構造を確認

問題4: useLoaderData が undefined を返す
  原因: ルート定義に loader が設定されていない、
        または createBrowserRouter を使っていない
  解決策:
  - ルート定義に loader 関数を追加
  - BrowserRouter → createBrowserRouter に移行
  - データルーティングAPIを使用していることを確認

問題5: NavLink の active スタイルが正しく適用されない
  原因: パスのマッチング設定が不正
  解決策:
  - ルートパスの NavLink に end プロパティを追加
    <NavLink to="/" end>Home</NavLink>
  - className を関数形式で指定
    <NavLink to="/users" className={({ isActive }) =>
      isActive ? 'active' : ''
    }>

問題6: loader が何度も実行される
  原因: React の Strict Mode による二重実行、
        または search params の変更トリガー
  解決策:
  - Strict Mode は開発環境のみ（本番では1回）
  - loader 内でキャッシュを活用（TanStack Query など）
  - loaderDeps で依存値を明示（TanStack Router）

問題7: lazy ルートのチャンクロードが失敗する
  原因: デプロイ後に古いチャンクファイルが削除された
  解決策:
  - エラーバウンダリでリロードを提案
  - Service Worker でキャッシュを管理
  - チャンクファイルにハッシュを付与（Vite/Webpackのデフォルト）
  - window.location.reload() でフルリロードを実行

問題8: ブラウザバック時にフォームデータが消える
  原因: ブラウザの bfcache からの復元時の状態不整合
  解決策:
  - フォームデータを sessionStorage に保存
  - location.state を活用してデータを保持
  - useBlocker で離脱確認を実装
```

```typescript
// === チャンクロードエラーのリカバリ ===
const router = createBrowserRouter([
  {
    path: '/settings',
    lazy: async () => {
      try {
        const { Settings } = await import('./pages/Settings');
        return { Component: Settings };
      } catch (error) {
        // チャンクロード失敗時のフォールバック
        if (
          error instanceof TypeError &&
          error.message.includes('Failed to fetch dynamically imported module')
        ) {
          // 新しいバージョンがデプロイされた可能性
          // ページをリロードして最新のマニフェストを取得
          window.location.reload();
          return { Component: () => <div>リロード中...</div> };
        }
        throw error;
      }
    },
  },
]);

// === デバッグ用ルーティングログ ===
function RouteLogger() {
  const location = useLocation();
  const navigation = useNavigation();

  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.group(`[Router] Navigation to ${location.pathname}`);
      console.log('Search:', location.search);
      console.log('Hash:', location.hash);
      console.log('State:', location.state);
      console.log('Key:', location.key);
      console.groupEnd();
    }
  }, [location]);

  useEffect(() => {
    if (process.env.NODE_ENV === 'development' && navigation.state !== 'idle') {
      console.log(
        `[Router] ${navigation.state}: ${navigation.location?.pathname ?? 'unknown'}`
      );
    }
  }, [navigation]);

  return null;
}
```

---

## 11. アクセシビリティとルーティング

### 11.1 フォーカス管理

クライアントサイドルーティングでは、ページ遷移時にブラウザのデフォルトのフォーカス管理が機能しないため、開発者が明示的にフォーカスを制御する必要がある。

```typescript
// === ページ遷移時のフォーカス管理 ===
function useFocusOnNavigate() {
  const location = useLocation();
  const mainRef = useRef<HTMLElement>(null);

  useEffect(() => {
    // ページ遷移後にメインコンテンツにフォーカスを移動
    if (mainRef.current) {
      mainRef.current.focus();
    }

    // スクリーンリーダーにページ変更を通知
    const pageTitle = document.title;
    const announcement = document.getElementById('route-announcement');
    if (announcement) {
      announcement.textContent = `${pageTitle} ページに移動しました`;
    }
  }, [location.pathname]);

  return mainRef;
}

function RootLayout() {
  const mainRef = useFocusOnNavigate();

  return (
    <div>
      {/* スキップリンク */}
      <a href="#main-content" className="skip-link">
        メインコンテンツにスキップ
      </a>

      {/* スクリーンリーダー用のライブリージョン */}
      <div
        id="route-announcement"
        role="status"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      />

      <nav aria-label="メインナビゲーション">
        <NavLink to="/">ホーム</NavLink>
        <NavLink to="/users">ユーザー</NavLink>
      </nav>

      <main
        id="main-content"
        ref={mainRef}
        tabIndex={-1}  // プログラマティックにフォーカス可能にする
        style={{ outline: 'none' }}  // フォーカスリングを非表示
      >
        <Outlet />
      </main>
    </div>
  );
}

// === CSS: スキップリンク ===
// .skip-link {
//   position: absolute;
//   top: -40px;
//   left: 0;
//   background: #000;
//   color: #fff;
//   padding: 8px;
//   z-index: 100;
//   transition: top 0.2s;
// }
// .skip-link:focus {
//   top: 0;
// }
// .sr-only {
//   position: absolute;
//   width: 1px;
//   height: 1px;
//   padding: 0;
//   margin: -1px;
//   overflow: hidden;
//   clip: rect(0, 0, 0, 0);
//   white-space: nowrap;
//   border: 0;
// }
```

### 11.2 アクセシブルなリンクとナビゲーション

```typescript
// === aria 属性を活用したナビゲーション ===
function AccessibleNavigation() {
  const location = useLocation();

  return (
    <nav aria-label="メインナビゲーション">
      <ul role="menubar">
        <li role="none">
          <NavLink
            to="/"
            end
            role="menuitem"
            aria-current={location.pathname === '/' ? 'page' : undefined}
          >
            ホーム
          </NavLink>
        </li>
        <li role="none">
          <NavLink
            to="/users"
            role="menuitem"
            aria-current={location.pathname.startsWith('/users') ? 'page' : undefined}
          >
            ユーザー
          </NavLink>
        </li>
        <li role="none">
          <NavLink
            to="/settings"
            role="menuitem"
            aria-current={location.pathname.startsWith('/settings') ? 'page' : undefined}
          >
            設定
          </NavLink>
        </li>
      </ul>
    </nav>
  );
}

// === パンくずリスト（Breadcrumbs）===
import { useMatches, Link } from 'react-router-dom';

function Breadcrumbs() {
  const matches = useMatches();

  // handle に breadcrumb 情報を持つルートのみフィルタリング
  const crumbs = matches.filter((match) =>
    (match.handle as any)?.breadcrumb
  );

  return (
    <nav aria-label="パンくずリスト">
      <ol className="breadcrumbs">
        {crumbs.map((match, index) => {
          const isLast = index === crumbs.length - 1;
          const breadcrumb = (match.handle as any).breadcrumb(match.data);

          return (
            <li key={match.id}>
              {isLast ? (
                <span aria-current="page">{breadcrumb}</span>
              ) : (
                <Link to={match.pathname}>{breadcrumb}</Link>
              )}
              {!isLast && <span aria-hidden="true"> / </span>}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

// ルート定義でパンくず情報を設定
const router = createBrowserRouter([
  {
    path: '/',
    element: <RootLayout />,
    handle: { breadcrumb: () => 'ホーム' },
    children: [
      {
        path: 'users',
        element: <UsersLayout />,
        handle: { breadcrumb: () => 'ユーザー' },
        loader: usersLoader,
        children: [
          {
            path: ':userId',
            element: <UserDetail />,
            handle: {
              breadcrumb: (data: User) => data.name,
            },
            loader: userDetailLoader,
          },
        ],
      },
    ],
  },
]);
```

---

## まとめ

### ルーティングライブラリの選定早見表

| ユースケース | 推奨ライブラリ | 理由 |
|------------|--------------|------|
| Vite + React（新規プロジェクト） | React Router v6 | 最も標準的、ドキュメント豊富 |
| 型安全性を最重視 | TanStack Router | search params まで完全な型推論 |
| フルスタック React | Next.js App Router | SSR/RSC のファーストクラスサポート |
| Remix プロジェクト | React Router（内蔵） | Remix の標準ルーター |
| 管理画面・ダッシュボード | React Router v6 (Hash) | サーバー設定不要 |
| 大規模 TypeScript プロジェクト | TanStack Router | リファクタリング安全性、開発体験 |
| SEO が重要なサイト | Next.js App Router | SSR/SSG の自動対応 |

### 実装チェックリスト

```
ルーティング実装の確認事項:

基本設定:
  [ ] ルータータイプの選定（Browser / Hash / Memory）
  [ ] SPA fallback のサーバー設定
  [ ] 404 ページの実装
  [ ] エラーバウンダリの設置

データ取得:
  [ ] loader でのデータフェッチ実装
  [ ] request.signal によるキャンセル対応
  [ ] defer による段階的データ取得
  [ ] エラーハンドリング（ネットワークエラー、404等）

認証・認可:
  [ ] 認証ガード（loader ベース）
  [ ] ロールベースアクセス制御
  [ ] リダイレクトURLの安全性検証
  [ ] ログインフローの実装

パフォーマンス:
  [ ] ルートベースのコード分割
  [ ] プリロード戦略の設定
  [ ] スクロール位置の復元
  [ ] ローディングUI の実装

アクセシビリティ:
  [ ] フォーカス管理（ページ遷移時）
  [ ] スキップリンクの実装
  [ ] aria-current の設定（ナビゲーション）
  [ ] スクリーンリーダーへのページ変更通知

テスト:
  [ ] ルーティングの統合テスト
  [ ] loader/action の単体テスト
  [ ] エラーパスのテスト
  [ ] 認証フローのテスト
```

---

## 前提知識

この章を最大限に活用するために、以下の知識を事前に習得しておくことを推奨する。

- **URL状態管理**: URL をアプリケーション状態の一部として扱う設計パターン → `../01-state-management/03-url-state.md`
- **ブラウザの History API**: `pushState`, `replaceState`, `popstate` イベントの仕組みと使い方
- **SPA/MPA/SSR の概念**: アーキテクチャの違いと、それぞれのルーティング要件 → `../00-architecture/00-spa-mpa-ssr.md`

これらの概念を理解することで、クライアントサイドルーティングの設計判断をより適切に行うことができる。

---

## FAQ

### Q1: Hash routing と History routing の違いは何ですか?

**A:** Hash routing は URL のフラグメント（`#` 以降）を利用するため、サーバー設定が不要で古いブラウザでも動作する。一方、History routing は History API を用いてクリーンな URL を実現するが、サーバー側で SPA フォールバック設定が必要となる。

```
Hash Routing:
  URL: https://example.com/#/users/123
  メリット:
    - サーバー設定不要（# 以降はサーバーに送信されない）
    - 古いブラウザ対応
    - GitHub Pages 等の静的ホスティングでそのまま動作
  デメリット:
    - SEO に不利（クローラーが # 以降を無視する場合がある）
    - URLが美しくない

History Routing:
  URL: https://example.com/users/123
  メリット:
    - クリーンで SEO フレンドリーな URL
    - ユーザー体験が向上
  デメリット:
    - サーバーで SPA フォールバック設定が必要
      (例: すべてのパスで index.html を返す)
```

**推奨**: 新規プロジェクトでは History routing を採用し、サーバー設定で対応する。Hash routing は静的ホスティング環境でのみ使用する。

### Q2: ルーティングライブラリの選択基準は何ですか?

**A:** プロジェクトの規模、型安全性の要求、チームの習熟度、フレームワークとの統合度で判断する。

```
選択基準マトリクス:

1. プロジェクト規模
   小〜中規模（〜50ルート） → React Router v6（シンプル、学習コスト低）
   大規模（50ルート以上） → TanStack Router（型安全、リファクタ容易）

2. 型安全性
   TypeScript + 型安全重視 → TanStack Router（search params まで型推論）
   JavaScript or 型は緩く → React Router v6

3. フレームワーク統合
   Next.js → App Router（ファイルベース）
   Remix → React Router（内蔵）
   Vite + React → React Router v6 or TanStack Router

4. チーム習熟度
   React Router 経験者多 → React Router v6（移行容易）
   新規チーム → TanStack Router（最新パターン）
```

### Q3: 動的ルーティングの実装方法と注意点は?

**A:** パスパラメータを URL セグメントとして定義し、コンポーネント内で取得する。バリデーションとエラーハンドリングを必ず実装すること。

```typescript
// React Router v6 の例
// ルート定義
<Route path="/users/:userId/posts/:postId" element={<PostDetail />} />

// コンポーネント内で取得
import { useParams } from 'react-router-dom';

function PostDetail() {
  const { userId, postId } = useParams<{ userId: string; postId: string }>();

  // 🚨 注意: useParams は常に string | undefined を返す
  // 数値として扱う場合は変換とバリデーションが必要

  const userIdNum = Number(userId);
  const postIdNum = Number(postId);

  if (isNaN(userIdNum) || isNaN(postIdNum)) {
    // エラーハンドリング
    return <NotFound />;
  }

  // データ取得
  const { data } = useQuery({
    queryKey: ['post', userIdNum, postIdNum],
    queryFn: () => fetchPost(userIdNum, postIdNum),
  });

  return <div>{/* ... */}</div>;
}

// TanStack Router の例（型安全）
const postRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/users/$userId/posts/$postId',
  // パラメータのバリデーション + パース
  parseParams: (params) => ({
    userId: z.number().parse(Number(params.userId)),
    postId: z.number().parse(Number(params.postId)),
  }),
  // loader で型推論が効く
  loader: async ({ params }) => {
    // params.userId, params.postId は number 型
    return fetchPost(params.userId, params.postId);
  },
});
```

**注意点**:
- パラメータは常に文字列で取得されるため、数値・日付等への変換処理を忘れない
- 不正な値（`userId: "abc"`）に対するエラーハンドリングを実装
- SEO が重要な場合、動的ルートのメタデータも動的に生成する

---

## 次に読むべきガイド
-> [[01-file-based-routing.md]] -- ファイルベースルーティング

---

## 参考文献
1. React Router. "React Router v6 Documentation." reactrouter.com, 2024.
2. TanStack. "TanStack Router Documentation." tanstack.com, 2024.
3. MDN Web Docs. "History API." developer.mozilla.org, 2024.
4. Web.dev. "View Transitions API." web.dev, 2024.
5. Kent C. Dodds. "Client-side Routing in React Applications." kentcdodds.com, 2023.
6. TkDodo. "Type-safe Search Params." tkdodo.eu, 2024.
7. Ryan Florence. "Data Loading in React Router." remix.run/blog, 2023.
8. W3C. "WAI-ARIA Authoring Practices - Navigation." w3.org/WAI, 2024.
