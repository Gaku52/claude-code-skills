# データフェッチングパターン

> データフェッチングはWebアプリの核心。TanStack Query、SWR、React Server Components、Server Actions、それぞれのキャッシュ戦略と使い分けを理解し、高速で安定したデータ取得を実現する。

## この章で学ぶこと

- [ ] TanStack QueryとSWRのキャッシュ戦略を理解する
- [ ] Server ComponentsとServer Actionsの使い分けを把握する
- [ ] オプティミスティックアップデートとリアルタイム更新を学ぶ

---

## 1. データフェッチングの選択肢

```
方式の比較:

  TanStack Query / SWR（クライアント）:
  → CSR/SPA向け
  → キャッシュ、リトライ、ポーリング
  → 楽観的更新
  → リアルタイム対応

  React Server Components（サーバー）:
  → SSR/RSC向け
  → async/await で直接データ取得
  → JSバンドルに影響なし
  → Next.js の fetch() キャッシュ

  Server Actions（サーバーミューテーション）:
  → フォーム送信、データ変更
  → プログレッシブエンハンスメント
  → JS無効でも動作

選定基準:
  データ読み取り（SEO不要）: TanStack Query / SWR
  データ読み取り（SEO必要）: Server Components
  データ変更: Server Actions + revalidate
  リアルタイム: TanStack Query + WebSocket
```

---

## 2. TanStack Query

```typescript
// セットアップ
import { QueryClient, QueryClientProvider, useQuery, useMutation } from '@tanstack/react-query';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,           // 60秒間はキャッシュを使用
      gcTime: 5 * 60 * 1000,          // 5分間キャッシュを保持
      retry: 3,                        // 3回リトライ
      refetchOnWindowFocus: true,      // ウィンドウフォーカスで再取得
    },
  },
});

// Provider
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <MyApp />
    </QueryClientProvider>
  );
}

// --- データ取得 ---
function useUsers(params?: { role?: string; page?: number }) {
  return useQuery({
    queryKey: ['users', params],       // キャッシュキー
    queryFn: () => api.users.list(params),
    staleTime: 30 * 1000,
  });
}

function UserList() {
  const { data, isLoading, error } = useUsers({ role: 'admin' });

  if (isLoading) return <Skeleton />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <ul>
      {data.users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}

// --- データ変更（Mutation）---
function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: CreateUserInput) => api.users.create(data),
    onSuccess: () => {
      // usersキャッシュを無効化 → 再取得
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}

function CreateUserForm() {
  const { mutate, isPending } = useCreateUser();

  return (
    <form onSubmit={(e) => {
      e.preventDefault();
      const formData = new FormData(e.currentTarget);
      mutate({
        name: formData.get('name') as string,
        email: formData.get('email') as string,
      });
    }}>
      <input name="name" required />
      <input name="email" type="email" required />
      <button disabled={isPending}>
        {isPending ? 'Creating...' : 'Create User'}
      </button>
    </form>
  );
}
```

---

## 3. オプティミスティックアップデート

```typescript
// 楽観的更新: APIレスポンスを待たずにUIを更新
function useToggleFavorite() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (productId: string) => api.favorites.toggle(productId),

    // 楽観的更新
    onMutate: async (productId) => {
      // 進行中のクエリをキャンセル
      await queryClient.cancelQueries({ queryKey: ['products'] });

      // 現在のデータをスナップショット
      const previous = queryClient.getQueryData(['products']);

      // キャッシュを楽観的に更新
      queryClient.setQueryData(['products'], (old: Product[]) =>
        old.map(p =>
          p.id === productId
            ? { ...p, isFavorite: !p.isFavorite }
            : p
        )
      );

      return { previous }; // ロールバック用
    },

    // エラー時: ロールバック
    onError: (error, productId, context) => {
      queryClient.setQueryData(['products'], context?.previous);
      toast.error('Failed to update favorite');
    },

    // 成功/失敗に関わらず: 最新データを再取得
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['products'] });
    },
  });
}
```

---

## 4. Server Components でのデータ取得

```typescript
// Next.js App Router — Server Component

// app/users/page.tsx
async function UsersPage() {
  // Server Component で直接データ取得
  const users = await prisma.user.findMany({
    orderBy: { createdAt: 'desc' },
    take: 20,
  });

  return (
    <div>
      <h1>Users</h1>
      <UserList users={users} />
      <Suspense fallback={<Skeleton />}>
        <UserStats />
      </Suspense>
    </div>
  );
}

// 並列データ取得
async function DashboardPage() {
  // Promise.all で並列取得（ウォーターフォール防止）
  const [users, orders, stats] = await Promise.all([
    getRecentUsers(),
    getRecentOrders(),
    getDashboardStats(),
  ]);

  return (
    <Dashboard>
      <UserSection users={users} />
      <OrderSection orders={orders} />
      <StatsSection stats={stats} />
    </Dashboard>
  );
}

// Streaming SSR（Suspense境界で段階的に表示）
async function ProductPage({ params }: { params: { id: string } }) {
  const product = await getProduct(params.id);

  return (
    <div>
      <ProductHeader product={product} />    {/* 即表示 */}
      <Suspense fallback={<ReviewsSkeleton />}>
        <ProductReviews productId={params.id} /> {/* 遅延表示 */}
      </Suspense>
      <Suspense fallback={<RecommendationsSkeleton />}>
        <Recommendations productId={params.id} /> {/* 遅延表示 */}
      </Suspense>
    </div>
  );
}
```

---

## 5. Server Actions

```typescript
// Server Actions — サーバーサイドのミューテーション
'use server';

import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import { z } from 'zod';

const CreateUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
});

export async function createUser(formData: FormData) {
  // バリデーション
  const parsed = CreateUserSchema.safeParse({
    name: formData.get('name'),
    email: formData.get('email'),
  });

  if (!parsed.success) {
    return { errors: parsed.error.flatten().fieldErrors };
  }

  // データ作成
  await prisma.user.create({ data: parsed.data });

  // キャッシュ無効化
  revalidatePath('/users');

  // リダイレクト
  redirect('/users');
}

// クライアント側の使用
'use client';
import { useActionState } from 'react';
import { createUser } from './actions';

function CreateUserForm() {
  const [state, action, isPending] = useActionState(createUser, null);

  return (
    <form action={action}>
      <input name="name" required />
      {state?.errors?.name && <p className="text-red-500">{state.errors.name}</p>}

      <input name="email" type="email" required />
      {state?.errors?.email && <p className="text-red-500">{state.errors.email}</p>}

      <button disabled={isPending}>
        {isPending ? 'Creating...' : 'Create'}
      </button>
    </form>
  );
}
```

---

## 6. パターンの使い分け

```
推奨パターン（Next.js App Router）:

  読み取り（SEO必要）:
  → Server Component + async/await

  読み取り（インタラクティブ）:
  → Client Component + TanStack Query

  書き込み（フォーム）:
  → Server Actions + useActionState

  書き込み（インタラクティブ）:
  → TanStack Query useMutation + optimistic update

  リアルタイム:
  → TanStack Query + WebSocket / SSE

  無限スクロール:
  → TanStack Query useInfiniteQuery

実装フロー:
  1. まず Server Component で試す
  2. インタラクティブ性が必要 → Client Component
  3. キャッシュ・リトライが必要 → TanStack Query
  4. フォーム送信 → Server Actions
  5. 楽観的更新 → useMutation + onMutate
```

---

## まとめ

| パターン | 用途 | 方式 |
|---------|------|------|
| Server Components | SEO必要な読み取り | サーバー |
| TanStack Query | インタラクティブな読み取り | クライアント |
| Server Actions | フォーム送信 | サーバー |
| useMutation | 楽観的更新 | クライアント |
| useInfiniteQuery | 無限スクロール | クライアント |

---

## 次に読むべきガイド
→ [[00-state-management-overview.md]] — 状態管理概論

---

## 参考文献
1. TanStack. "TanStack Query Documentation." tanstack.com, 2024.
2. Next.js. "Data Fetching." nextjs.org/docs, 2024.
3. SWR. "React Hooks for Data Fetching." swr.vercel.app, 2024.
