# 認証ガード

> 認証ガードはアプリケーションの入口を守る門番。ルート保護、ロールベースアクセス制御、リダイレクト、セッション管理まで、安全で使いやすい認証フローの設計と実装を習得する。

## この章で学ぶこと

- [ ] ルート保護のパターンと実装を理解する
- [ ] ロールベースアクセス制御の設計を把握する
- [ ] Next.js Middleware での認証チェックを学ぶ

---

## 1. ルート保護のパターン

```
認証ガードの3つのレイヤー:

  ① Middleware（最前線）:
     → リクエスト到達前にチェック
     → 最も早い段階でリダイレクト
     → Next.js middleware.ts

  ② Layout（レイアウト層）:
     → Server Component でセッション確認
     → 認証が必要なエリア全体を保護

  ③ Page / Component（ページ層）:
     → 個別ページでの権限チェック
     → きめ細かいアクセス制御

推奨:
  → Middleware: 大まかなルート保護（/app/* は認証必要）
  → Layout: 認証エリアのレイアウト
  → Component: ボタン表示/非表示等の細かい制御
```

---

## 2. Next.js Middleware

```typescript
// middleware.ts（プロジェクトルート）
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// 認証不要なパス
const publicPaths = ['/', '/login', '/register', '/about', '/pricing'];

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 公開パスはスキップ
  if (publicPaths.some(path => pathname === path || pathname.startsWith('/api/public'))) {
    return NextResponse.next();
  }

  // 静的ファイルはスキップ
  if (pathname.startsWith('/_next') || pathname.includes('.')) {
    return NextResponse.next();
  }

  // セッショントークンの確認
  const token = request.cookies.get('session-token')?.value;

  if (!token) {
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

---

## 3. Layout での認証

```typescript
// app/(app)/layout.tsx — 認証必要エリアのレイアウト
import { redirect } from 'next/navigation';
import { getSession } from '@/shared/lib/auth';

export default async function AuthenticatedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();

  if (!session) {
    redirect('/login');
  }

  return (
    <SessionProvider session={session}>
      <div className="flex">
        <Sidebar user={session.user} />
        <main className="flex-1">{children}</main>
      </div>
    </SessionProvider>
  );
}

// app/(app)/admin/layout.tsx — 管理者専用エリア
export default async function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();

  if (session?.user.role !== 'admin') {
    redirect('/dashboard'); // 権限不足はダッシュボードへ
  }

  return <>{children}</>;
}
```

---

## 4. ロールベースアクセス制御（RBAC）

```typescript
// 権限定義
type Role = 'user' | 'editor' | 'admin';
type Permission = 'users:read' | 'users:write' | 'users:delete'
  | 'orders:read' | 'orders:write' | 'admin:access';

const rolePermissions: Record<Role, Permission[]> = {
  user: ['users:read', 'orders:read'],
  editor: ['users:read', 'users:write', 'orders:read', 'orders:write'],
  admin: ['users:read', 'users:write', 'users:delete',
          'orders:read', 'orders:write', 'admin:access'],
};

function hasPermission(role: Role, permission: Permission): boolean {
  return rolePermissions[role]?.includes(permission) ?? false;
}

// コンポーネントレベルの制御
function UserActions({ user }: { user: User }) {
  const session = useSession();

  return (
    <div>
      <ViewButton />
      {hasPermission(session.user.role, 'users:write') && (
        <EditButton userId={user.id} />
      )}
      {hasPermission(session.user.role, 'users:delete') && (
        <DeleteButton userId={user.id} />
      )}
    </div>
  );
}

// 権限チェックコンポーネント
function RequirePermission({
  permission,
  children,
  fallback = null,
}: {
  permission: Permission;
  children: React.ReactNode;
  fallback?: React.ReactNode;
}) {
  const session = useSession();
  if (!hasPermission(session.user.role, permission)) {
    return fallback;
  }
  return <>{children}</>;
}

// 使用例
<RequirePermission permission="users:delete" fallback={<Tooltip>権限がありません</Tooltip>}>
  <DeleteButton userId={user.id} />
</RequirePermission>
```

---

## 5. 認証フロー

```
ログインフロー:
  1. /login にアクセス
  2. 認証情報を入力
  3. Server Action でセッション作成
  4. callbackUrl にリダイレクト（なければ /dashboard）

  // app/login/page.tsx
  export default function LoginPage({ searchParams }) {
    return <LoginForm callbackUrl={searchParams.callbackUrl} />;
  }

  // Server Action
  'use server';
  async function login(formData: FormData) {
    const { email, password } = Object.fromEntries(formData);
    const user = await authenticateUser(email, password);

    if (!user) {
      return { error: 'Invalid credentials' };
    }

    // セッション作成（HTTPOnly Cookie）
    await createSession(user.id);

    const callbackUrl = formData.get('callbackUrl') as string;
    redirect(callbackUrl || '/dashboard');
  }

ログアウトフロー:
  1. ログアウトボタンクリック
  2. Server Action でセッション削除
  3. /login にリダイレクト

  'use server';
  async function logout() {
    await deleteSession();
    redirect('/login');
  }

セッション更新:
  → Middleware でセッションの有効期限を延長
  → スライディングウィンドウ方式
```

---

## まとめ

| レイヤー | 役割 | ツール |
|---------|------|--------|
| Middleware | ルート保護 | middleware.ts |
| Layout | エリア保護 | Server Component |
| Component | 要素制御 | RBAC + RequirePermission |

---

## 次に読むべきガイド
→ [[00-form-design.md]] — フォーム設計

---

## 参考文献
1. Next.js. "Authentication." nextjs.org/docs, 2024.
2. Auth.js. "NextAuth.js Documentation." authjs.dev, 2024.
3. Clerk. "Authentication for Next.js." clerk.com, 2024.
