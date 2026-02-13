# フロントエンド認可

> フロントエンドの認可はUX向上のための「表示制御」であり、セキュリティの最終防衛線ではない。ルートガード、コンポーネントの条件表示、権限に基づくUI制御、CASL/Zod を使った宣言的認可、Server Components との責務分担、権限のプリフェッチ戦略を網羅的に解説する。

## この章で学ぶこと

- [ ] フロントエンド認可の役割と限界を正確に理解する
- [ ] Next.js Middleware と React Router のルートガードを実装できるようになる
- [ ] 権限ベースの UI 制御を宣言的に実装する方法を学ぶ
- [ ] Server Components と Client Components の認可責務分担を把握する
- [ ] 権限のプリフェッチ、キャッシュ、リアルタイム更新を実践する

### 前提知識

- React（Context API, Hooks）の基礎
- Next.js App Router（Server Components, Middleware）の基礎
- 認可の基本概念（ロール、パーミッション）→ [[01-rbac-abac.md]]

---

## 1. フロントエンド認可の原則

### 1.1 フロントエンドとバックエンドの責務分担

```
重要な原則:

  フロントエンドの認可 = UX の最適化
  サーバーの認可 = セキュリティの保証

  ┌──────────────────────────────────────────────────────┐
  │                 認可の階層構造                        │
  │                                                      │
  │  ┌──────────────────────────────────────────────┐    │
  │  │ Layer 1: フロントエンド（表示制御）            │    │
  │  │                                              │    │
  │  │  → 権限のないメニュー項目を非表示             │    │
  │  │  → 権限のないボタンを無効化（disabled）       │    │
  │  │  → 権限のないページへアクセス時にリダイレクト  │    │
  │  │  → ユーザーに不要な選択肢を見せない           │    │
  │  │  → 操作不可の理由をユーザーに説明             │    │
  │  │                                              │    │
  │  │  目的: ユーザー体験の向上                     │    │
  │  │  信頼度: 低（バイパス可能）                   │    │
  │  └──────────────────────────────────────────────┘    │
  │                                                      │
  │  ┌──────────────────────────────────────────────┐    │
  │  │ Layer 2: BFF / API Gateway                    │    │
  │  │                                              │    │
  │  │  → ルートレベルの認可チェック                 │    │
  │  │  → トークン検証                              │    │
  │  │  → レート制限                                │    │
  │  │                                              │    │
  │  │  目的: 粗いアクセス制御                       │    │
  │  │  信頼度: 中                                   │    │
  │  └──────────────────────────────────────────────┘    │
  │                                                      │
  │  ┌──────────────────────────────────────────────┐    │
  │  │ Layer 3: バックエンド（セキュリティ）          │    │
  │  │                                              │    │
  │  │  → 全 API リクエストで権限チェック             │    │
  │  │  → リソースレベルの認可（所有権チェック）      │    │
  │  │  → フロントエンドの表示状態に依存しない        │    │
  │  │  → フロントエンドがバイパスされても安全         │    │
  │  │                                              │    │
  │  │  目的: セキュリティの保証                     │    │
  │  │  信頼度: 高（唯一の信頼境界）                 │    │
  │  └──────────────────────────────────────────────┘    │
  │                                                      │
  └──────────────────────────────────────────────────────┘
```

### 1.2 なぜフロントエンドのみの認可は危険なのか

```
フロントエンドのみの認可が危険な理由:

  攻撃手法①: DevTools による操作
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  // DevTools Console で実行                        │
  │  document.querySelector('[disabled]')               │
  │    .removeAttribute('disabled');                    │
  │  // → 無効化されたボタンがクリック可能に             │
  │                                                    │
  │  // hidden 要素の表示                              │
  │  document.querySelector('.hidden')                  │
  │    .style.display = 'block';                       │
  │  // → 非表示だった管理者メニューが表示される        │
  │                                                    │
  └────────────────────────────────────────────────────┘

  攻撃手法②: API の直接呼び出し
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  // フロントエンドを完全にスキップ                  │
  │  curl -X DELETE https://api.example.com/users/123  │
  │    -H "Cookie: session=stolen_session"             │
  │  // → フロントエンドの表示制御は意味なし            │
  │                                                    │
  └────────────────────────────────────────────────────┘

  攻撃手法③: JavaScript の改変
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  // React DevTools でステートを書き換え             │
  │  // user.role = 'viewer' → user.role = 'admin'     │
  │  // → クライアント側の権限チェックがバイパスされる   │
  │                                                    │
  └────────────────────────────────────────────────────┘

  正しいアプローチ:
  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ✓ バックエンドで必ず認可チェック                   │
  │  ✓ フロントエンドは UX のために補助的に表示制御     │
  │  ✓ フロントエンドの権限データはバックエンドから取得  │
  │  ✓ 操作の最終判断はバックエンドが行う              │
  │                                                    │
  └────────────────────────────────────────────────────┘
```

### 1.3 フロントエンド認可で実現すべきこと

```
フロントエンド認可の具体的な目的:

  ① プリベンティブ UI:
     → 権限のない操作を事前にブロック
     → ユーザーがエラーに遭遇する前に防止
     → 例: 編集権限がなければ Edit ボタンを非表示

  ② ガイダンス:
     → なぜ操作できないかを説明
     → 権限取得の方法を案内
     → 例: 「編集するにはエディターロールが必要です」

  ③ パフォーマンス:
     → 不要な API コールを防止
     → 権限のないデータの取得を避ける
     → 例: 管理者でなければ管理者 API を呼ばない

  ④ 情報の最小化:
     → 権限のないデータをそもそも取得しない
     → Server Components で権限に応じたデータのみ返す
     → 例: 一般ユーザーには管理者メニューの存在すら見せない
```

---

## 2. ルートガード

### 2.1 Next.js Middleware によるルートガード

```typescript
// middleware.ts - Next.js Middleware によるルートガード
import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@/auth';

// 保護ルートの定義（パス → 許可ロール）
const protectedRoutes: Record<string, string[]> = {
  '/dashboard': ['viewer', 'editor', 'admin'],
  '/articles/new': ['editor', 'admin'],
  '/articles/edit': ['editor', 'admin'],
  '/admin': ['admin'],
  '/settings': ['editor', 'admin'],
  '/billing': ['admin'],
};

// 公開ルート（認証不要）
const publicRoutes = new Set([
  '/',
  '/login',
  '/register',
  '/forgot-password',
  '/reset-password',
  '/verify-email',
  '/about',
  '/pricing',
]);

export default auth((request) => {
  const { pathname } = request.nextUrl;
  const session = request.auth;

  // 静的ファイルと API ルートはスキップ
  if (
    pathname.startsWith('/_next') ||
    pathname.startsWith('/api/') ||
    pathname.includes('.')
  ) {
    return NextResponse.next();
  }

  // 公開ルートはそのまま
  if (publicRoutes.has(pathname)) {
    // ログイン済みユーザーが /login にアクセスした場合はリダイレクト
    if (session && pathname === '/login') {
      return NextResponse.redirect(new URL('/dashboard', request.url));
    }
    return NextResponse.next();
  }

  // 認証チェック
  if (!session) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('callbackUrl', pathname);
    return NextResponse.redirect(loginUrl);
  }

  // 保護ルートのロールチェック
  const matchedRoute = Object.keys(protectedRoutes).find(
    (route) => pathname.startsWith(route)
  );

  if (matchedRoute) {
    const allowedRoles = protectedRoutes[matchedRoute];
    const userRole = session.user?.role;

    if (!userRole || !allowedRoles.includes(userRole)) {
      // 権限不足: 403 ページにリダイレクト
      return NextResponse.redirect(new URL('/unauthorized', request.url));
    }
  }

  return NextResponse.next();
});

export const config = {
  matcher: [
    // 静的ファイルと内部パスを除外
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
};
```

```
Middleware の動作フロー:

  リクエスト
  │
  ├─ 静的ファイル？ ──Yes──→ スキップ（NextResponse.next()）
  │
  ├─ 公開ルート？ ──Yes──→ そのまま通過
  │   └─ ログイン済み + /login？ ──→ /dashboard にリダイレクト
  │
  ├─ セッションあり？ ──No──→ /login にリダイレクト（callbackUrl 付き）
  │
  ├─ 保護ルート？ ──Yes──→ ロールチェック
  │   ├─ 許可ロール？ ──Yes──→ NextResponse.next()
  │   └─ 不許可？ ──→ /unauthorized にリダイレクト
  │
  └─ その他 ──→ NextResponse.next()

  注意事項:
  ┌────────────────────────────────────────────────────┐
  │ ・Middleware は Edge Runtime で動作                  │
  │ ・DB アクセスは制限がある（Prisma は使用不可）       │
  │ ・JWT の検証のみで判断するのが一般的                │
  │ ・matcher で対象パスを適切に制限する                │
  │ ・パフォーマンスに影響するため処理は軽量に          │
  └────────────────────────────────────────────────────┘
```

### 2.2 動的ルートの権限チェック

```typescript
// 動的ルートでのリソースレベル認可
// middleware だけでは不十分な場合 → ページコンポーネントで追加チェック

// app/articles/[id]/edit/page.tsx
import { auth } from '@/auth';
import { redirect, notFound } from 'next/navigation';

export default async function EditArticlePage({
  params,
}: {
  params: { id: string };
}) {
  const session = await auth();
  if (!session) redirect('/login');

  // 記事の取得
  const article = await prisma.article.findUnique({
    where: { id: params.id },
  });

  if (!article) notFound();

  // リソースレベルの認可チェック
  const canEdit =
    session.user.role === 'admin' ||
    article.authorId === session.user.id;

  if (!canEdit) {
    redirect('/unauthorized');
  }

  return <ArticleEditor article={article} />;
}
```

### 2.3 React Router でのルートガード

```typescript
// React Router v6 でのルートガード（SPA 向け）

import { Navigate, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';

// 基本のルートガード
function ProtectedRoute({
  children,
  requiredRole,
  requiredPermission,
  fallback,
}: {
  children?: React.ReactNode;
  requiredRole?: string[];
  requiredPermission?: string;
  fallback?: React.ReactNode;
}) {
  const { user, isLoading, can } = useAuth();
  const location = useLocation();

  // ローディング中はスケルトンを表示
  if (isLoading) {
    return fallback ?? <FullPageSkeleton />;
  }

  // 未認証 → ログインページにリダイレクト
  if (!user) {
    return (
      <Navigate
        to="/login"
        state={{ from: location.pathname }}
        replace
      />
    );
  }

  // ロールチェック
  if (requiredRole && !requiredRole.includes(user.role)) {
    return <Navigate to="/unauthorized" replace />;
  }

  // パーミッションチェック
  if (requiredPermission && !can(requiredPermission)) {
    return <Navigate to="/unauthorized" replace />;
  }

  return children ?? <Outlet />;
}

// 使用例: ルート定義
import { createBrowserRouter, RouterProvider } from 'react-router-dom';

const router = createBrowserRouter([
  // 公開ルート
  { path: '/login', element: <LoginPage /> },
  { path: '/register', element: <RegisterPage /> },

  // 認証必須ルート
  {
    element: <ProtectedRoute />,
    children: [
      { path: '/dashboard', element: <DashboardPage /> },
      { path: '/profile', element: <ProfilePage /> },
    ],
  },

  // エディター以上
  {
    element: <ProtectedRoute requiredRole={['editor', 'admin']} />,
    children: [
      { path: '/articles/new', element: <NewArticlePage /> },
      { path: '/articles/:id/edit', element: <EditArticlePage /> },
    ],
  },

  // 管理者のみ
  {
    element: <ProtectedRoute requiredRole={['admin']} />,
    children: [
      { path: '/admin', element: <AdminDashboard /> },
      { path: '/admin/users', element: <UserManagementPage /> },
      { path: '/admin/settings', element: <AdminSettingsPage /> },
    ],
  },

  // パーミッションベース
  {
    element: <ProtectedRoute requiredPermission="billing:manage" />,
    children: [
      { path: '/billing', element: <BillingPage /> },
    ],
  },

  // 404 / 403
  { path: '/unauthorized', element: <UnauthorizedPage /> },
  { path: '*', element: <NotFoundPage /> },
]);
```

---

## 3. 権限ベースの UI 制御

### 3.1 AuthContext の設計

```typescript
// lib/auth-context.tsx - 権限コンテキストの完全実装
'use client';

import {
  createContext,
  useContext,
  useCallback,
  useMemo,
  type ReactNode,
} from 'react';
import { useSession } from 'next-auth/react';
import { useQuery } from '@tanstack/react-query';

// 権限の型定義
type Permission = string; // "resource:action" 形式

interface AuthContextValue {
  // ユーザー情報
  user: {
    id: string;
    name: string;
    email: string;
    role: string;
    image?: string;
  } | null;

  // 認証状態
  isAuthenticated: boolean;
  isLoading: boolean;

  // 権限
  permissions: Set<Permission>;
  permissionsLoading: boolean;

  // 権限チェック関数
  can: (action: string, resource?: string) => boolean;
  canAny: (permissions: string[]) => boolean;
  canAll: (permissions: string[]) => boolean;

  // ロールチェック
  hasRole: (role: string) => boolean;
  hasAnyRole: (roles: string[]) => boolean;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  isAuthenticated: false,
  isLoading: true,
  permissions: new Set(),
  permissionsLoading: true,
  can: () => false,
  canAny: () => false,
  canAll: () => false,
  hasRole: () => false,
  hasAnyRole: () => false,
});

// 権限のフェッチ
async function fetchPermissions(): Promise<Set<Permission>> {
  const res = await fetch('/api/auth/permissions');
  if (!res.ok) throw new Error('Failed to fetch permissions');
  const data = await res.json();
  return new Set<Permission>(data.permissions);
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const { data: session, status } = useSession();
  const isAuthenticated = status === 'authenticated';

  // 権限のフェッチ（React Query でキャッシュ管理）
  const {
    data: permissions = new Set<Permission>(),
    isLoading: permissionsLoading,
  } = useQuery({
    queryKey: ['permissions', session?.user?.id],
    queryFn: fetchPermissions,
    enabled: isAuthenticated,
    staleTime: 5 * 60 * 1000,   // 5 分間はキャッシュを使用
    gcTime: 30 * 60 * 1000,     // 30 分間メモリに保持
    refetchOnWindowFocus: false, // タブ切替時の再取得を防止
    retry: 2,
  });

  // 権限チェック: resource:action 形式
  const can = useCallback(
    (action: string, resource?: string): boolean => {
      if (!isAuthenticated) return false;

      // admin ロールは全権限を持つ
      if (session?.user?.role === 'admin') return true;

      const permission = resource ? `${resource}:${action}` : action;
      return permissions.has(permission);
    },
    [isAuthenticated, session?.user?.role, permissions]
  );

  // いずれかの権限を持っているか
  const canAny = useCallback(
    (perms: string[]): boolean => {
      return perms.some((p) => {
        const [resource, action] = p.split(':');
        return can(action, resource);
      });
    },
    [can]
  );

  // 全ての権限を持っているか
  const canAll = useCallback(
    (perms: string[]): boolean => {
      return perms.every((p) => {
        const [resource, action] = p.split(':');
        return can(action, resource);
      });
    },
    [can]
  );

  // ロールチェック
  const hasRole = useCallback(
    (role: string): boolean => {
      return session?.user?.role === role;
    },
    [session?.user?.role]
  );

  const hasAnyRole = useCallback(
    (roles: string[]): boolean => {
      return roles.includes(session?.user?.role ?? '');
    },
    [session?.user?.role]
  );

  const value = useMemo<AuthContextValue>(
    () => ({
      user: session?.user
        ? {
            id: session.user.id!,
            name: session.user.name!,
            email: session.user.email!,
            role: session.user.role as string,
            image: session.user.image ?? undefined,
          }
        : null,
      isAuthenticated,
      isLoading: status === 'loading',
      permissions,
      permissionsLoading,
      can,
      canAny,
      canAll,
      hasRole,
      hasAnyRole,
    }),
    [session, isAuthenticated, status, permissions, permissionsLoading,
     can, canAny, canAll, hasRole, hasAnyRole]
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
```

### 3.2 権限 API エンドポイント

```typescript
// app/api/auth/permissions/route.ts
import { auth } from '@/auth';
import { NextResponse } from 'next/server';

// ロールごとの権限マッピング
const rolePermissions: Record<string, string[]> = {
  viewer: [
    'articles:read',
    'comments:read',
    'comments:create',
    'profile:read',
    'profile:update',
  ],
  editor: [
    'articles:read',
    'articles:create',
    'articles:update',
    'articles:publish',
    'comments:read',
    'comments:create',
    'comments:update',
    'comments:delete',
    'media:upload',
    'profile:read',
    'profile:update',
  ],
  admin: [
    'admin', // 特殊権限: 全権限を包含
  ],
};

export async function GET() {
  const session = await auth();

  if (!session?.user) {
    return NextResponse.json({ permissions: [] }, { status: 401 });
  }

  const role = session.user.role as string;
  const permissions = rolePermissions[role] ?? [];

  // ユーザー固有の追加権限（DB から取得する場合）
  // const userPermissions = await prisma.userPermission.findMany({
  //   where: { userId: session.user.id },
  //   select: { permission: true },
  // });
  // const extraPermissions = userPermissions.map(p => p.permission);

  return NextResponse.json({
    permissions: [...permissions],
    role,
  });
}
```

### 3.3 Authorized コンポーネント（宣言的認可）

```typescript
// components/authorized.tsx - 宣言的な権限チェックコンポーネント
'use client';

import { useAuth } from '@/lib/auth-context';
import type { ReactNode } from 'react';

interface AuthorizedProps {
  // 単一権限チェック
  permission?: string;
  // 複数権限（いずれか）
  anyPermission?: string[];
  // 複数権限（全て）
  allPermissions?: string[];
  // ロールチェック
  role?: string;
  anyRole?: string[];
  // 権限がない場合の表示
  fallback?: ReactNode;
  // 権限がない場合にdisabledで表示するか
  showDisabled?: boolean;
  // 子要素
  children: ReactNode;
}

export function Authorized({
  permission,
  anyPermission,
  allPermissions,
  role,
  anyRole,
  fallback = null,
  showDisabled = false,
  children,
}: AuthorizedProps) {
  const { can, canAny, canAll, hasRole, hasAnyRole, permissionsLoading } = useAuth();

  // 権限ロード中は何も表示しない（ちらつき防止）
  if (permissionsLoading) return null;

  let authorized = true;

  // 権限チェック
  if (permission) {
    const [resource, action] = permission.split(':');
    authorized = can(action, resource);
  }

  if (anyPermission) {
    authorized = canAny(anyPermission);
  }

  if (allPermissions) {
    authorized = canAll(allPermissions);
  }

  // ロールチェック
  if (role) {
    authorized = hasRole(role);
  }

  if (anyRole) {
    authorized = hasAnyRole(anyRole);
  }

  if (!authorized) {
    if (showDisabled) {
      // disabled 状態で表示
      return (
        <div className="opacity-50 pointer-events-none" aria-disabled="true">
          {children}
        </div>
      );
    }
    return <>{fallback}</>;
  }

  return <>{children}</>;
}

// 使用例
function ArticleCard({ article }: { article: Article }) {
  return (
    <div className="border rounded-lg p-4">
      <h2 className="text-xl font-bold">{article.title}</h2>
      <p className="text-gray-600 mt-2">{article.excerpt}</p>

      <div className="flex gap-2 mt-4">
        {/* 閲覧権限のある人だけに表示 */}
        <Authorized permission="articles:update">
          <Link href={`/articles/${article.id}/edit`} className="btn-secondary">
            編集
          </Link>
        </Authorized>

        {/* 削除権限: 無効化状態で表示 */}
        <Authorized
          permission="articles:delete"
          showDisabled
          fallback={
            <button disabled className="btn-danger opacity-50" title="削除権限がありません">
              削除
            </button>
          }
        >
          <button
            className="btn-danger"
            onClick={() => deleteArticle(article.id)}
          >
            削除
          </button>
        </Authorized>

        {/* 公開権限 */}
        <Authorized permission="articles:publish">
          {article.status === 'draft' && (
            <button
              className="btn-primary"
              onClick={() => publishArticle(article.id)}
            >
              公開
            </button>
          )}
        </Authorized>

        {/* 管理者のみ */}
        <Authorized role="admin">
          <button
            className="btn-outline"
            onClick={() => viewAuditLog(article.id)}
          >
            監査ログ
          </button>
        </Authorized>
      </div>
    </div>
  );
}
```

### 3.4 useAuthorized フック

```typescript
// hooks/useAuthorized.ts - フックとしての権限チェック
'use client';

import { useAuth } from '@/lib/auth-context';
import { useMemo } from 'react';

interface UseAuthorizedOptions {
  permission?: string;
  anyPermission?: string[];
  allPermissions?: string[];
  role?: string;
  anyRole?: string[];
}

interface UseAuthorizedResult {
  authorized: boolean;
  loading: boolean;
}

export function useAuthorized(options: UseAuthorizedOptions): UseAuthorizedResult {
  const { can, canAny, canAll, hasRole, hasAnyRole, permissionsLoading } = useAuth();

  const authorized = useMemo(() => {
    if (permissionsLoading) return false;

    if (options.permission) {
      const [resource, action] = options.permission.split(':');
      if (!can(action, resource)) return false;
    }

    if (options.anyPermission) {
      if (!canAny(options.anyPermission)) return false;
    }

    if (options.allPermissions) {
      if (!canAll(options.allPermissions)) return false;
    }

    if (options.role) {
      if (!hasRole(options.role)) return false;
    }

    if (options.anyRole) {
      if (!hasAnyRole(options.anyRole)) return false;
    }

    return true;
  }, [options, can, canAny, canAll, hasRole, hasAnyRole, permissionsLoading]);

  return {
    authorized,
    loading: permissionsLoading,
  };
}

// 使用例
function ArticleActions({ article }: { article: Article }) {
  const { authorized: canEdit } = useAuthorized({ permission: 'articles:update' });
  const { authorized: canDelete } = useAuthorized({ permission: 'articles:delete' });
  const { authorized: isAdmin } = useAuthorized({ role: 'admin' });

  return (
    <div className="flex gap-2">
      {canEdit && <EditButton articleId={article.id} />}
      {canDelete && <DeleteButton articleId={article.id} />}
      {isAdmin && <AdminActions articleId={article.id} />}
    </div>
  );
}
```

---

## 4. ナビゲーションの権限制御

### 4.1 権限ベースのナビゲーション定義

```typescript
// lib/navigation.ts - ナビゲーション項目の定義
import {
  HomeIcon,
  DocumentIcon,
  UsersIcon,
  CogIcon,
  CreditCardIcon,
  ChartBarIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

export interface NavItem {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  permission?: string;   // 必要な権限
  role?: string;         // 必要なロール
  badge?: string;        // バッジ表示
  children?: NavItem[];
}

export const navItems: NavItem[] = [
  {
    label: 'ダッシュボード',
    href: '/dashboard',
    icon: HomeIcon,
    // 全ユーザーがアクセス可能
  },
  {
    label: '記事',
    href: '/articles',
    icon: DocumentIcon,
    permission: 'articles:read',
    children: [
      { label: '一覧', href: '/articles', icon: DocumentIcon },
      { label: '新規作成', href: '/articles/new', icon: DocumentIcon, permission: 'articles:create' },
      { label: '下書き', href: '/articles/drafts', icon: DocumentIcon, permission: 'articles:update' },
    ],
  },
  {
    label: 'ユーザー管理',
    href: '/admin/users',
    icon: UsersIcon,
    permission: 'users:read',
  },
  {
    label: '分析',
    href: '/analytics',
    icon: ChartBarIcon,
    permission: 'analytics:read',
  },
  {
    label: '設定',
    href: '/settings',
    icon: CogIcon,
    children: [
      { label: '一般', href: '/settings/general', icon: CogIcon },
      { label: '請求', href: '/settings/billing', icon: CreditCardIcon, permission: 'billing:manage' },
      { label: 'メンバー', href: '/settings/members', icon: UsersIcon, permission: 'users:read' },
      { label: 'セキュリティ', href: '/settings/security', icon: ShieldCheckIcon, role: 'admin' },
    ],
  },
];
```

### 4.2 サイドバーの権限フィルタリング

```typescript
// components/sidebar.tsx - 権限ベースのサイドバー
'use client';

import { useAuth } from '@/lib/auth-context';
import { navItems, type NavItem } from '@/lib/navigation';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

export function Sidebar() {
  const { can, hasRole, permissionsLoading } = useAuth();
  const pathname = usePathname();

  // 権限に基づいてナビゲーション項目をフィルタリング
  const filterNavItems = (items: NavItem[]): NavItem[] => {
    return items
      .filter((item) => {
        // 権限チェック
        if (item.permission) {
          const [resource, action] = item.permission.split(':');
          if (!can(action, resource)) return false;
        }
        // ロールチェック
        if (item.role && !hasRole(item.role)) return false;
        return true;
      })
      .map((item) => ({
        ...item,
        // 子項目も再帰的にフィルタリング
        children: item.children ? filterNavItems(item.children) : undefined,
      }))
      // 子項目がすべてフィルタリングされた親は非表示
      .filter((item) => {
        if (item.children && item.children.length === 0) return false;
        return true;
      });
  };

  // 権限ロード中はスケルトン表示
  if (permissionsLoading) {
    return <SidebarSkeleton />;
  }

  const visibleItems = filterNavItems(navItems);

  return (
    <nav className="w-64 bg-gray-900 text-white min-h-screen p-4">
      {visibleItems.map((item) => (
        <NavLink key={item.href} item={item} pathname={pathname} />
      ))}
    </nav>
  );
}

function NavLink({ item, pathname }: { item: NavItem; pathname: string }) {
  const [isOpen, setIsOpen] = useState(
    item.children?.some((child) => pathname.startsWith(child.href)) ?? false
  );
  const isActive = pathname === item.href;
  const Icon = item.icon;

  if (item.children && item.children.length > 0) {
    return (
      <div>
        <button
          onClick={() => setIsOpen(!isOpen)}
          className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg
            hover:bg-gray-800 transition-colors ${
              isActive ? 'bg-gray-800' : ''
            }`}
        >
          <Icon className="w-5 h-5" />
          <span className="flex-1 text-left">{item.label}</span>
          <ChevronIcon className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-90' : ''}`} />
        </button>
        {isOpen && (
          <div className="ml-4 mt-1 space-y-1">
            {item.children.map((child) => (
              <NavLink key={child.href} item={child} pathname={pathname} />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <Link
      href={item.href}
      className={`flex items-center gap-3 px-3 py-2 rounded-lg
        hover:bg-gray-800 transition-colors ${
          isActive ? 'bg-gray-800 text-white' : 'text-gray-300'
        }`}
    >
      <Icon className="w-5 h-5" />
      <span>{item.label}</span>
      {item.badge && (
        <span className="ml-auto bg-blue-500 text-xs px-2 py-0.5 rounded-full">
          {item.badge}
        </span>
      )}
    </Link>
  );
}
```

---

## 5. Server Components での認可

### 5.1 サーバーサイドでの権限判定（推奨パターン）

```typescript
// Next.js Server Components での認可（推奨）
// サーバーサイドで権限チェック → クライアントに不要な情報を送らない

// app/articles/[id]/page.tsx
import { auth } from '@/auth';
import { redirect, notFound } from 'next/navigation';

export default async function ArticlePage({
  params,
}: {
  params: { id: string };
}) {
  const session = await auth();
  if (!session) redirect('/login');

  // 記事データの取得
  const article = await prisma.article.findUnique({
    where: { id: params.id },
    include: {
      author: { select: { name: true, image: true } },
      comments: {
        orderBy: { createdAt: 'desc' },
        take: 20,
      },
    },
  });

  if (!article) notFound();

  // サーバーサイドで権限を判定
  const permissions = {
    canEdit:
      session.user.role === 'admin' ||
      article.authorId === session.user.id,
    canPublish:
      ['admin', 'editor'].includes(session.user.role) &&
      article.status === 'draft',
    canDelete:
      session.user.role === 'admin',
    canModerateComments:
      ['admin', 'moderator'].includes(session.user.role),
    canViewAnalytics:
      session.user.role === 'admin' ||
      article.authorId === session.user.id,
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* 記事コンテンツ（Server Component） */}
      <article>
        <h1 className="text-3xl font-bold">{article.title}</h1>
        <AuthorInfo author={article.author} />
        <ArticleContent content={article.content} />
      </article>

      {/* アクションバー（Client Component、権限は props で渡す） */}
      <ArticleActions
        articleId={article.id}
        status={article.status}
        {...permissions}
      />

      {/* コメントセクション */}
      <CommentSection
        comments={article.comments}
        canModerate={permissions.canModerateComments}
      />

      {/* 分析データ（権限がある場合のみサーバーで取得） */}
      {permissions.canViewAnalytics && (
        <ArticleAnalytics articleId={article.id} />
      )}
    </div>
  );
}
```

```typescript
// クライアントコンポーネント（権限は props で受け取る）
// サーバーで判定済みの権限フラグを使用
'use client';

interface ArticleActionsProps {
  articleId: string;
  status: string;
  canEdit: boolean;
  canPublish: boolean;
  canDelete: boolean;
}

function ArticleActions({
  articleId,
  status,
  canEdit,
  canPublish,
  canDelete,
}: ArticleActionsProps) {
  return (
    <div className="flex gap-2 border-t border-b py-4 my-8">
      {canEdit && (
        <Link href={`/articles/${articleId}/edit`} className="btn-secondary">
          編集
        </Link>
      )}
      {canPublish && (
        <button
          className="btn-primary"
          onClick={() => publishArticle(articleId)}
        >
          公開
        </button>
      )}
      {canDelete && (
        <button
          className="btn-danger"
          onClick={() => {
            if (confirm('本当に削除しますか？')) {
              deleteArticle(articleId);
            }
          }}
        >
          削除
        </button>
      )}
    </div>
  );
}
```

### 5.2 Server Components vs Client Components の使い分け

```
Server Components と Client Components の認可パターン比較:

  ┌──────────────────┬───────────────────┬──────────────────────┐
  │ 項目              │ Server Component  │ Client Component     │
  ├──────────────────┼───────────────────┼──────────────────────┤
  │ 権限チェック場所   │ サーバー          │ クライアント         │
  │ データ取得        │ DB に直接アクセス  │ API 経由             │
  │ セキュリティ      │ 高（改ざん不可）  │ 低（バイパス可能）    │
  │ 不要データの送信   │ なし             │ 権限データの送信が必要 │
  │ インタラクティブ性 │ なし             │ あり（onClick 等）    │
  │ パフォーマンス     │ 高（JS バンドルなし）│ 権限フェッチのコスト │
  │ 推奨用途          │ 初期表示制御      │ インタラクティブ UI   │
  └──────────────────┴───────────────────┴──────────────────────┘

  推奨パターン:

  ① 初期表示: Server Component で権限判定
     → 不要な UI 要素をそもそもレンダリングしない
     → クライアントに権限のないデータを送信しない

  ② インタラクティブ操作: Client Component + props
     → Server Component で判定した権限を props で渡す
     → Client Component はフラグに基づいて表示制御

  ③ 動的な権限変更: Client Component + Context
     → リアルタイムで権限が変わる場合（ロール変更等）
     → AuthContext から権限を取得して表示制御
```

---

## 6. 権限のプリフェッチとキャッシュ

### 6.1 React Query を使ったプリフェッチ

```typescript
// lib/permissions-provider.tsx
'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { useState, type ReactNode } from 'react';

export function PermissionsProvider({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60 * 1000, // 5分
            gcTime: 30 * 60 * 1000,   // 30分
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  );
}
```

```typescript
// hooks/usePermissions.ts
'use client';

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useSession } from 'next-auth/react';
import { useCallback } from 'react';

export function usePermissions() {
  const { data: session } = useSession();

  return useQuery({
    queryKey: ['permissions'],
    queryFn: async (): Promise<Set<string>> => {
      const res = await fetch('/api/auth/permissions');
      if (!res.ok) throw new Error('Failed to fetch permissions');
      const data = await res.json();
      return new Set<string>(data.permissions);
    },
    enabled: !!session,
    staleTime: 5 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    refetchOnWindowFocus: false,
    // ログインイベントで自動更新される
  });
}

// 権限の手動更新（ロール変更時、権限変更時）
export function useInvalidatePermissions() {
  const queryClient = useQueryClient();

  return useCallback(() => {
    queryClient.invalidateQueries({ queryKey: ['permissions'] });
  }, [queryClient]);
}

// 権限のプリフェッチ（ログイン直後に呼ぶ）
export function usePrefetchPermissions() {
  const queryClient = useQueryClient();

  return useCallback(() => {
    queryClient.prefetchQuery({
      queryKey: ['permissions'],
      queryFn: async () => {
        const res = await fetch('/api/auth/permissions');
        const data = await res.json();
        return new Set<string>(data.permissions);
      },
    });
  }, [queryClient]);
}
```

### 6.2 権限変更時のリアルタイム更新

```typescript
// hooks/usePermissionsSync.ts
// WebSocket または Server-Sent Events で権限変更を検知

'use client';

import { useEffect } from 'react';
import { useInvalidatePermissions } from './usePermissions';
import { useSession } from 'next-auth/react';

export function usePermissionsSync() {
  const invalidatePermissions = useInvalidatePermissions();
  const { data: session } = useSession();

  useEffect(() => {
    if (!session?.user?.id) return;

    // Server-Sent Events で権限変更を監視
    const eventSource = new EventSource(
      `/api/auth/permissions/stream?userId=${session.user.id}`
    );

    eventSource.addEventListener('permissions_changed', () => {
      // 権限キャッシュを無効化して再取得
      invalidatePermissions();
    });

    eventSource.addEventListener('role_changed', () => {
      invalidatePermissions();
    });

    eventSource.onerror = () => {
      // 接続エラー時は 5 秒後に再接続
      eventSource.close();
      setTimeout(() => {
        // 再接続ロジック（実装は省略）
      }, 5000);
    };

    return () => {
      eventSource.close();
    };
  }, [session?.user?.id, invalidatePermissions]);
}
```

---

## 7. CASL を使った宣言的認可

### 7.1 CASL の概要

```
CASL (Isomorphic Authorization) の特徴:

  ┌────────────────────────────────────────────────────┐
  │                                                    │
  │  ① Isomorphic: サーバーとクライアントで同じルール    │
  │  ② 宣言的: ルールを定義するだけで権限チェック        │
  │  ③ React 統合: Can コンポーネント                   │
  │  ④ TypeScript: 型安全な権限定義                     │
  │  ⑤ パフォーマンス: ルール評価のキャッシュ            │
  │                                                    │
  └────────────────────────────────────────────────────┘

  CASL vs 自前実装:

  ┌──────────────────┬──────────────────┬──────────────┐
  │ 項目              │ CASL             │ 自前実装      │
  ├──────────────────┼──────────────────┼──────────────┤
  │ ルール定義        │ DSL で宣言的     │ コードで手続的 │
  │ 条件付き権限      │ 組み込みサポート  │ 自前で実装    │
  │ React 統合       │ @casl/react      │ 自前コンポーネント│
  │ フィールドレベル  │ サポートあり      │ 自前で実装    │
  │ 学習コスト        │ 中               │ 低           │
  │ バンドルサイズ     │ ~8KB (gzip)      │ 依存なし     │
  │ 推奨              │ 複雑な権限体系   │ シンプルな RBAC│
  └──────────────────┴──────────────────┴──────────────┘
```

### 7.2 CASL の実装

```typescript
// lib/ability.ts - CASL による権限定義
import { AbilityBuilder, createMongoAbility, MongoAbility } from '@casl/ability';

// アクションの型定義
type Actions = 'create' | 'read' | 'update' | 'delete' | 'publish' | 'manage';

// サブジェクト（リソース）の型定義
type Subjects =
  | 'Article'
  | 'Comment'
  | 'User'
  | 'Organization'
  | 'all';

export type AppAbility = MongoAbility<[Actions, Subjects]>;

// ロールに基づく Ability の構築
export function defineAbilityFor(user: {
  id: string;
  role: string;
  organizationId?: string;
}): AppAbility {
  const { can, cannot, build } = new AbilityBuilder<AppAbility>(createMongoAbility);

  switch (user.role) {
    case 'admin':
      // 管理者は全権限
      can('manage', 'all');
      break;

    case 'editor':
      can('read', 'Article');
      can('create', 'Article');
      // 自分の記事のみ編集・削除可能
      can('update', 'Article', { authorId: user.id });
      can('delete', 'Article', { authorId: user.id });
      can('publish', 'Article', { authorId: user.id });
      // コメント
      can('read', 'Comment');
      can('create', 'Comment');
      can('update', 'Comment', { authorId: user.id });
      can('delete', 'Comment', { authorId: user.id });
      break;

    case 'viewer':
      can('read', 'Article');
      can('read', 'Comment');
      can('create', 'Comment');
      can('update', 'Comment', { authorId: user.id });
      // 公開記事のみ
      cannot('read', 'Article', { status: 'draft' });
      break;

    default:
      // ゲスト: 公開記事の閲覧のみ
      can('read', 'Article', { status: 'published' });
  }

  return build();
}
```

```typescript
// components/can.tsx - CASL の React コンポーネント
'use client';

import { createContext, useContext, type ReactNode } from 'react';
import { createContextualCan } from '@casl/react';
import { type AppAbility } from '@/lib/ability';

// Ability コンテキスト
const AbilityContext = createContext<AppAbility>(undefined!);

export function AbilityProvider({
  ability,
  children,
}: {
  ability: AppAbility;
  children: ReactNode;
}) {
  return (
    <AbilityContext.Provider value={ability}>
      {children}
    </AbilityContext.Provider>
  );
}

export function useAbility(): AppAbility {
  return useContext(AbilityContext);
}

// CASL の Can コンポーネント
export const Can = createContextualCan(AbilityContext.Consumer);

// 使用例
function ArticleActions({ article }: { article: Article }) {
  return (
    <div className="flex gap-2">
      <Can I="update" this={article}>
        <Link href={`/articles/${article.id}/edit`}>編集</Link>
      </Can>

      <Can I="delete" this={article}>
        <button onClick={() => deleteArticle(article.id)}>削除</button>
      </Can>

      <Can I="publish" this={article}>
        {article.status === 'draft' && (
          <button onClick={() => publishArticle(article.id)}>公開</button>
        )}
      </Can>

      {/* not で否定 */}
      <Can not I="update" this={article}>
        <p className="text-gray-500">この記事を編集する権限がありません</p>
      </Can>
    </div>
  );
}
```

---

## 8. エラーページとフォールバック

### 8.1 認可エラーページの実装

```typescript
// app/unauthorized/page.tsx - 403 Unauthorized ページ
import { auth } from '@/auth';
import Link from 'next/link';

export default async function UnauthorizedPage() {
  const session = await auth();

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="text-center max-w-md">
        <div className="text-6xl font-bold text-gray-300">403</div>
        <h1 className="text-2xl font-bold mt-4">アクセスが拒否されました</h1>
        <p className="text-gray-600 mt-2">
          このページにアクセスする権限がありません。
        </p>

        {session ? (
          <div className="mt-6 space-y-3">
            <p className="text-sm text-gray-500">
              ログイン中: {session.user?.email}
              （ロール: {session.user?.role}）
            </p>
            <div className="flex gap-3 justify-center">
              <Link href="/dashboard" className="btn-primary">
                ダッシュボードに戻る
              </Link>
              <Link href="/settings" className="btn-secondary">
                権限を確認
              </Link>
            </div>
          </div>
        ) : (
          <div className="mt-6">
            <Link href="/login" className="btn-primary">
              ログインする
            </Link>
          </div>
        )}

        <p className="text-xs text-gray-400 mt-8">
          この問題が続く場合は、管理者にお問い合わせください。
        </p>
      </div>
    </div>
  );
}
```

---

## 9. アンチパターン

### 9.1 フロントエンドのみでの認可

```typescript
// ✗ 危険: フロントエンドのみで認可（バックエンドチェックなし）
function DeleteButton({ articleId }: { articleId: string }) {
  const { user } = useAuth();

  if (user?.role !== 'admin') return null; // これだけでは不十分！

  const handleDelete = async () => {
    // API 側で権限チェックがないと、直接 API を叩かれる
    await fetch(`/api/articles/${articleId}`, { method: 'DELETE' });
  };

  return <button onClick={handleDelete}>削除</button>;
}

// ✓ 正しい: フロントエンド + バックエンドの両方でチェック
// API 側:
// if (session.user.role !== 'admin') {
//   return NextResponse.json({ error: 'Forbidden' }, { status: 403 });
// }
```

### 9.2 権限データのハードコーディング

```typescript
// ✗ 問題: 権限をフロントエンドにハードコード
const ADMIN_EMAILS = ['admin@example.com', 'boss@example.com'];

function AdminPanel() {
  const { user } = useAuth();
  if (!ADMIN_EMAILS.includes(user?.email ?? '')) return null;
  return <AdminDashboard />;
}

// ✓ 正しい: サーバーから権限を取得
function AdminPanel() {
  const { hasRole } = useAuth();
  if (!hasRole('admin')) return null;
  return <AdminDashboard />;
}
```

### 9.3 権限チェックの不整合

```typescript
// ✗ 問題: フロントとバックで異なるロジック
// フロントエンド: editor でも公開可能
const canPublish = hasRole('editor') || hasRole('admin');

// バックエンド: admin のみ公開可能
// if (user.role !== 'admin') return 403;

// → エディターが公開ボタンを押すと 403 エラー
// → UX が悪い（ボタンは見えるのにエラー）

// ✓ 正しい: 権限の判定ロジックを共有する
// 共通の権限ファイルを定義し、フロントとバックで共有
// または、権限 API からフラグを取得
```

---

## 10. 演習問題

### 演習 1: 基本 — ルートガードとProtectedRouteの実装（難易度: 基本）

```
課題:
  Next.js App Router で、Middleware を使ったルートガードと
  ProtectedRoute コンポーネントを実装してください。

要件:
  ① /dashboard は全認証ユーザーがアクセス可能
  ② /admin は admin ロールのみ
  ③ /articles/new は editor と admin のみ
  ④ 未認証ユーザーは /login にリダイレクト（callbackUrl 付き）
  ⑤ 権限不足は /unauthorized にリダイレクト

ヒント:
  → auth() 関数で Middleware 内のセッション取得
  → matcher で対象パスを制限

確認ポイント:
  □ 未認証で /dashboard → /login にリダイレクト
  □ viewer で /admin → /unauthorized にリダイレクト
  □ editor で /articles/new → アクセス可能
  □ ログイン後に元のページに戻る
```

### 演習 2: 応用 — 権限ベースの UI 制御（難易度: 応用）

```
課題:
  AuthContext と Authorized コンポーネントを実装し、
  記事管理画面で権限に基づく UI 制御を行ってください。

要件:
  ① 権限を API からフェッチしてキャッシュ
  ② Authorized コンポーネントで条件表示
  ③ 権限のないボタンは非表示 or disabled
  ④ ナビゲーションの権限フィルタリング
  ⑤ Server Component で初期権限判定

ヒント:
  → React Query で権限をキャッシュ
  → AuthProvider → Authorized → useAuth のレイヤー構成
  → Server Component から props で権限フラグを渡す

確認ポイント:
  □ viewer は記事の閲覧のみ
  □ editor は記事の作成・編集・削除が可能
  □ admin はすべての操作が可能
  □ ナビゲーションが権限に応じて変化
```

### 演習 3: 発展 — CASL + リアルタイム権限更新（難易度: 発展）

```
課題:
  CASL ライブラリを使った宣言的認可と、
  WebSocket を使った権限のリアルタイム更新を実装してください。

要件:
  ① CASL で条件付き認可を定義（自分の記事のみ編集可能等）
  ② Can コンポーネントで UI 制御
  ③ WebSocket で権限変更をリアルタイム通知
  ④ 管理者が他ユーザーのロールを変更 → 即座に UI が更新
  ⑤ フィールドレベルの権限（特定フィールドの編集制限）

ヒント:
  → @casl/ability + @casl/react を使用
  → subject() ヘルパーでリソースの型を指定
  → WebSocket: socket.io-client を使用

確認ポイント:
  □ editor が自分の記事のみ編集できる
  □ admin がロール変更 → 即座に対象ユーザーの UI が変化
  □ フィールドレベルの制限が動作する
```

---

## 11. FAQ

### Q1: Server Components と Client Components のどちらで権限チェックすべき？

```
A: 可能な限り Server Components で行うのが推奨です。

理由:
  → サーバーで判定するため改ざん不可能
  → 権限のないデータをクライアントに送信しない
  → JS バンドルサイズを削減（権限ロジックがサーバー側に）

Client Components を使う場合:
  → インタラクティブな UI（onClick、状態変更）
  → リアルタイムの権限更新が必要な場合
  → ユーザー操作に応じた動的な表示切替

推奨パターン:
  Server Component で権限を判定 → boolean フラグとして
  Client Component に props で渡す
```

### Q2: 権限キャッシュの有効期間はどのくらいが適切？

```
A: 一般的に 5 分が推奨です。

考慮事項:
  → 短すぎる: API 呼び出しが増え、パフォーマンス低下
  → 長すぎる: 権限変更が反映されるまでの遅延

  ┌──────────────┬──────────────────────────────┐
  │ staleTime     │ 用途                         │
  ├──────────────┼──────────────────────────────┤
  │ 1 分          │ 厳密な権限制御が必要な場合     │
  │ 5 分（推奨）  │ 一般的な Web アプリ           │
  │ 15 分         │ 権限変更が稀な場合            │
  │ リアルタイム   │ WebSocket + invalidate       │
  └──────────────┴──────────────────────────────┘

  リアルタイム性が必要な場合:
  → WebSocket / SSE で権限変更を通知
  → invalidateQueries で即座にキャッシュ更新
```

### Q3: ナビゲーションで権限のないページを完全に非表示にすべき？

```
A: 状況によります。

完全非表示にすべき場合:
  → セキュリティ上、ページの存在を知られたくない
  → 管理者機能（一般ユーザーは知る必要なし）

disabled / グレーアウトにすべき場合:
  → 機能の存在は知ってほしい（アップグレード促進）
  → 「Pro プランにアップグレードすると利用可能」

推奨:
  → セキュリティ系: 完全非表示
  → ビジネス系: disabled + ツールチップで説明
```

### Q4: 大規模アプリでの権限管理のベストプラクティスは？

```
A: 以下の構成を推奨します。

  ① 権限定義: 一元管理ファイル（lib/permissions.ts）
  ② 権限取得: 専用 API + React Query キャッシュ
  ③ 権限チェック: AuthContext + Authorized コンポーネント
  ④ ルートガード: Middleware（粗いチェック）
  ⑤ ページレベル: Server Components（詳細チェック）
  ⑥ UIレベル: Authorized コンポーネント / Can（CASL）
  ⑦ APIレベル: バックエンドで最終チェック（必須）

権限定義の一元管理例:
  // lib/permissions.ts
  export const PERMISSIONS = {
    ARTICLES_CREATE: 'articles:create',
    ARTICLES_READ: 'articles:read',
    // ...
  } as const;

  // フロントとバックで同じ定数を使用
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 原則 | フロント=UX最適化、バックエンド=セキュリティ保証 |
| ルートガード | Next.js Middleware + Auth.js で実装 |
| UI制御 | AuthContext + Authorized コンポーネント |
| Server Components | サーバーで権限判定 → props でクライアントに渡す（推奨） |
| ナビゲーション | 権限ベースの再帰的フィルタリング |
| キャッシュ | React Query で 5 分キャッシュ、WebSocket で即時更新 |
| CASL | 複雑な条件付き認可に最適。シンプルな RBAC なら自前でも可 |
| 必須 | フロントエンドの認可は補助的。API 側での認可が必須 |

---

## 次に読むべきガイド

- [[../04-implementation/00-nextauth-setup.md]] — NextAuth.js セットアップ
- [[01-rbac-abac.md]] — RBAC と ABAC
- [[02-api-authorization.md]] — API 認可
- [[../01-session-auth/02-csrf-protection.md]] — CSRF 防御

---

## 参考文献

1. OWASP. "Authorization Testing." owasp.org, 2024.
2. Next.js. "Middleware." nextjs.org/docs/app/building-your-application/routing/middleware, 2024.
3. CASL. "React Integration." casl.js.org/v6/en/package/casl-react, 2024.
4. TanStack. "React Query." tanstack.com/query/latest, 2024.
5. Auth.js. "Session Management." authjs.dev/getting-started/session-management, 2024.
6. React Router. "Authentication." reactrouter.com/en/main/start/concepts, 2024.
