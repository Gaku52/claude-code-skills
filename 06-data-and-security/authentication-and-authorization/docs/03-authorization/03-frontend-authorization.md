# フロントエンド認可

> フロントエンドの認可はUX向上のための「表示制御」であり、セキュリティの最終防衛線ではない。ルートガード、コンポーネントの条件表示、権限に基づくUI制御、サーバーとの責務分担を解説する。

## この章で学ぶこと

- [ ] フロントエンド認可の役割と限界を理解する
- [ ] ルートガードと権限ベースの UI 制御を実装できるようになる
- [ ] サーバー認可とフロントエンド表示制御の責務分担を把握する

---

## 1. フロントエンド認可の原則

```
重要な原則:

  フロントエンドの認可 = UX の最適化
  サーバーの認可 = セキュリティの保証

  ┌──────────────────────────────────────────┐
  │                                          │
  │  フロントエンド（表示制御）:                 │
  │  → 権限のないメニューを非表示              │
  │  → 権限のないボタンを無効化                │
  │  → 権限のないページへのリダイレクト          │
  │  → ユーザーに不要な選択肢を見せない         │
  │                                          │
  │  バックエンド（セキュリティ）:               │
  │  → 全 API リクエストで権限チェック           │
  │  → フロントエンドの表示状態に依存しない      │
  │  → フロントエンドがバイパスされても安全       │
  │                                          │
  └──────────────────────────────────────────┘

  ✗ フロントエンドのみの認可は危険:
    → DevTools で表示を操作可能
    → API を直接叩ける
    → JavaScript のロジックを改変可能

  ✓ 正しいアプローチ:
    → バックエンドで必ず認可チェック
    → フロントエンドは UX のために補助的に表示制御
```

---

## 2. ルートガード

```typescript
// Next.js Middleware によるルートガード
// middleware.ts
import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';

// 保護ルートの定義
const protectedRoutes: Record<string, string[]> = {
  '/dashboard': ['viewer', 'editor', 'admin'],
  '/articles/new': ['editor', 'admin'],
  '/admin': ['admin'],
  '/settings': ['editor', 'admin'],
};

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 保護ルートかチェック
  const matchedRoute = Object.keys(protectedRoutes).find(
    (route) => pathname.startsWith(route)
  );

  if (!matchedRoute) return NextResponse.next();

  // セッション確認
  const session = await getSession(request);

  if (!session) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('callbackUrl', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // ロールチェック
  const allowedRoles = protectedRoutes[matchedRoute];
  if (!allowedRoles.includes(session.user.role)) {
    return NextResponse.redirect(new URL('/unauthorized', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/dashboard/:path*', '/articles/:path*', '/admin/:path*', '/settings/:path*'],
};
```

```typescript
// React Router でのルートガード
function ProtectedRoute({
  children,
  requiredRole,
  fallback = <Navigate to="/login" />,
}: {
  children: React.ReactNode;
  requiredRole?: string[];
  fallback?: React.ReactNode;
}) {
  const { user, isLoading } = useAuth();

  if (isLoading) return <LoadingSkeleton />;

  if (!user) return <Navigate to="/login" />;

  if (requiredRole && !requiredRole.includes(user.role)) {
    return <Navigate to="/unauthorized" />;
  }

  return <>{children}</>;
}

// ルート定義
<Routes>
  <Route path="/login" element={<LoginPage />} />
  <Route
    path="/dashboard"
    element={
      <ProtectedRoute>
        <DashboardPage />
      </ProtectedRoute>
    }
  />
  <Route
    path="/admin"
    element={
      <ProtectedRoute requiredRole={['admin']}>
        <AdminPage />
      </ProtectedRoute>
    }
  />
</Routes>
```

---

## 3. 権限ベースの UI 制御

```typescript
// 権限コンテキスト
interface AuthContext {
  user: User | null;
  permissions: Set<string>;
  can: (action: string, resource?: string) => boolean;
}

const AuthContext = createContext<AuthContext>({
  user: null,
  permissions: new Set(),
  can: () => false,
});

function AuthProvider({ children }: { children: React.ReactNode }) {
  const { data: session } = useSession();
  const { data: permissions } = useQuery({
    queryKey: ['permissions', session?.user?.id],
    queryFn: () => fetchUserPermissions(),
    enabled: !!session?.user,
  });

  const can = useCallback(
    (action: string, resource?: string) => {
      if (!permissions) return false;
      const permission = resource ? `${resource}:${action}` : action;
      return permissions.has(permission) || permissions.has('admin');
    },
    [permissions]
  );

  return (
    <AuthContext.Provider value={{
      user: session?.user ?? null,
      permissions: permissions ?? new Set(),
      can,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

function useAuth() {
  return useContext(AuthContext);
}

// 条件表示コンポーネント
function Authorized({
  permission,
  fallback = null,
  children,
}: {
  permission: string;
  fallback?: React.ReactNode;
  children: React.ReactNode;
}) {
  const { can } = useAuth();
  const [resource, action] = permission.split(':');

  if (!can(action, resource)) return <>{fallback}</>;
  return <>{children}</>;
}

// 使用例
function ArticleCard({ article }: { article: Article }) {
  return (
    <div>
      <h2>{article.title}</h2>
      <p>{article.excerpt}</p>

      <div className="flex gap-2">
        <Authorized permission="articles:update">
          <Link href={`/articles/${article.id}/edit`}>
            Edit
          </Link>
        </Authorized>

        <Authorized permission="articles:delete">
          <button className="text-red-500" onClick={() => deleteArticle(article.id)}>
            Delete
          </button>
        </Authorized>

        <Authorized permission="articles:publish">
          {article.status === 'draft' && (
            <button className="text-green-500" onClick={() => publishArticle(article.id)}>
              Publish
            </button>
          )}
        </Authorized>
      </div>
    </div>
  );
}
```

---

## 4. ナビゲーションの権限制御

```typescript
// 権限ベースのナビゲーション
interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType;
  permission?: string;
  children?: NavItem[];
}

const navItems: NavItem[] = [
  { label: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { label: 'Articles', href: '/articles', icon: DocumentIcon, permission: 'articles:read' },
  { label: 'Users', href: '/users', icon: UsersIcon, permission: 'users:read' },
  {
    label: 'Settings',
    href: '/settings',
    icon: CogIcon,
    permission: 'org:settings',
    children: [
      { label: 'General', href: '/settings/general', icon: CogIcon },
      { label: 'Billing', href: '/settings/billing', icon: CreditCardIcon, permission: 'org:billing' },
      { label: 'Members', href: '/settings/members', icon: UsersIcon, permission: 'users:read' },
    ],
  },
];

function Sidebar() {
  const { can } = useAuth();

  const filterNavItems = (items: NavItem[]): NavItem[] => {
    return items.filter((item) => {
      if (item.permission) {
        const [resource, action] = item.permission.split(':');
        if (!can(action, resource)) return false;
      }
      return true;
    }).map((item) => ({
      ...item,
      children: item.children ? filterNavItems(item.children) : undefined,
    }));
  };

  const visibleItems = filterNavItems(navItems);

  return (
    <nav>
      {visibleItems.map((item) => (
        <NavLink key={item.href} item={item} />
      ))}
    </nav>
  );
}
```

---

## 5. サーバーコンポーネントでの認可

```typescript
// Next.js Server Components での認可（推奨）

// サーバーサイドで権限チェック → クライアントに不要な情報を送らない
async function ArticlePage({ params }: { params: { id: string } }) {
  const session = await getServerSession();
  if (!session) redirect('/login');

  const article = await db.article.findUnique({ where: { id: params.id } });
  if (!article) notFound();

  // サーバーサイドで権限を判定
  const canEdit = session.user.role === 'admin' || article.authorId === session.user.id;
  const canPublish = ['admin', 'publisher'].includes(session.user.role);
  const canDelete = session.user.role === 'admin';

  return (
    <div>
      <h1>{article.title}</h1>
      <ArticleContent content={article.content} />

      {/* 権限フラグに基づく表示制御 */}
      <ArticleActions
        articleId={article.id}
        canEdit={canEdit}
        canPublish={canPublish}
        canDelete={canDelete}
      />
    </div>
  );
}

// クライアントコンポーネント（権限はpropsで受取り）
'use client';
function ArticleActions({
  articleId,
  canEdit,
  canPublish,
  canDelete,
}: {
  articleId: string;
  canEdit: boolean;
  canPublish: boolean;
  canDelete: boolean;
}) {
  return (
    <div className="flex gap-2">
      {canEdit && <Link href={`/articles/${articleId}/edit`}>Edit</Link>}
      {canPublish && <button onClick={() => publish(articleId)}>Publish</button>}
      {canDelete && <button onClick={() => remove(articleId)}>Delete</button>}
    </div>
  );
}
```

---

## 6. 権限のプリフェッチとキャッシュ

```typescript
// ログイン直後に権限をプリフェッチ
function usePermissions() {
  const { data: session } = useSession();

  return useQuery({
    queryKey: ['permissions'],
    queryFn: async () => {
      const res = await fetch('/api/auth/permissions');
      const data = await res.json();
      return new Set<string>(data.permissions);
    },
    enabled: !!session,
    staleTime: 5 * 60 * 1000,  // 5分キャッシュ
    gcTime: 30 * 60 * 1000,    // 30分保持
  });
}

// 権限変更時の更新
function useInvalidatePermissions() {
  const queryClient = useQueryClient();

  return useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['permissions'] });
  }, [queryClient]);
}
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 原則 | フロント=UX最適化、バックエンド=セキュリティ |
| ルートガード | Middleware + ProtectedRoute |
| UI制御 | Authorized コンポーネント |
| ナビゲーション | 権限ベースのフィルタリング |
| Server Components | サーバーで権限判定→propsで渡す |
| キャッシュ | 権限をプリフェッチ、5分キャッシュ |

---

## 次に読むべきガイド
→ [[../04-implementation/00-nextauth-setup.md]] — NextAuth.js セットアップ

---

## 参考文献
1. OWASP. "Authorization Testing." owasp.org, 2024.
2. Next.js. "Middleware." nextjs.org/docs, 2024.
3. CASL. "React Integration." casl.js.org/v6/en/package/casl-react, 2024.
