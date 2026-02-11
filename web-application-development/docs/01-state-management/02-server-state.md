# サーバー状態

> サーバー状態はクライアント状態とは根本的に異なる。TanStack QueryとSWRのキャッシュ戦略、stale-while-revalidateパターン、無限スクロール、ポーリング、プリフェッチまで、サーバーデータ管理の全技法を習得する。

## この章で学ぶこと

- [ ] サーバー状態の特性とキャッシュ戦略を理解する
- [ ] TanStack Queryの高度なパターンを把握する
- [ ] 無限スクロールとリアルタイム更新を学ぶ

---

## 1. サーバー状態の特性

```
サーバー状態 vs クライアント状態:

  クライアント状態:
  → アプリが所有（唯一の情報源）
  → 同期的にアクセス可能
  → 常に最新

  サーバー状態:
  → サーバーが所有（クライアントはコピー）
  → 非同期的にアクセス
  → 古くなる可能性がある
  → 他のクライアントが変更する可能性

  サーバー状態の課題:
  → キャッシュ: いつデータを再取得するか
  → 同期: クライアントのコピーをどう最新に保つか
  → 重複排除: 同じデータへの複数リクエストをまとめる
  → 楽観的更新: APIレスポンス前にUIを更新
  → エラーハンドリング: リトライ、フォールバック
  → ページネーション: 無限スクロール、ページ送り
```

---

## 2. TanStack Query のキャッシュ

```
キャッシュのライフサイクル:

  staleTime（鮮度期限）:
  → データが「新鮮」とみなされる期間
  → 新鮮なデータは再取得しない
  → デフォルト: 0（常にstale）

  gcTime（ガベージコレクション時間）:
  → 使用されなくなったキャッシュの保持期間
  → デフォルト: 5分

  フロー:
  1. クエリ実行 → データ取得 → キャッシュに保存
  2. staleTime 以内 → キャッシュを返す（リクエストなし）
  3. staleTime 経過後 → キャッシュを返す + バックグラウンドで再取得
  4. 再取得成功 → キャッシュ更新 → UIが自動更新
  5. コンポーネントアンマウント → gcTime 後にキャッシュ削除

  再取得のトリガー:
  → ウィンドウフォーカス時（refetchOnWindowFocus）
  → ネットワーク再接続時（refetchOnReconnect）
  → ポーリング間隔（refetchInterval）
  → 手動（queryClient.invalidateQueries）
```

```typescript
// キャッシュ戦略の設計

// ① 頻繁に変わるデータ（チャット、通知）
const { data } = useQuery({
  queryKey: ['notifications'],
  queryFn: getNotifications,
  staleTime: 0,              // 常にstale
  refetchInterval: 10000,    // 10秒ごとにポーリング
});

// ② ほとんど変わらないデータ（設定、マスタ）
const { data } = useQuery({
  queryKey: ['categories'],
  queryFn: getCategories,
  staleTime: 24 * 60 * 60 * 1000, // 24時間
  gcTime: Infinity,                  // 永久にキャッシュ
});

// ③ ユーザー操作で変わるデータ（CRUD）
const { data } = useQuery({
  queryKey: ['users', { page, filter }],
  queryFn: () => getUsers({ page, filter }),
  staleTime: 30 * 1000,    // 30秒
  placeholderData: keepPreviousData, // ページ遷移時に前のデータを表示
});
```

---

## 3. 無限スクロール

```typescript
import { useInfiniteQuery } from '@tanstack/react-query';
import { useInView } from 'react-intersection-observer';

function useInfiniteUsers(filter?: string) {
  return useInfiniteQuery({
    queryKey: ['users', 'infinite', filter],
    queryFn: ({ pageParam }) => api.users.list({
      cursor: pageParam,
      limit: 20,
      filter,
    }),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) =>
      lastPage.meta.hasNextPage ? lastPage.meta.nextCursor : undefined,
  });
}

function UserInfiniteList() {
  const {
    data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading,
  } = useInfiniteUsers();

  const { ref } = useInView({
    onChange: (inView) => {
      if (inView && hasNextPage && !isFetchingNextPage) {
        fetchNextPage();
      }
    },
  });

  if (isLoading) return <Skeleton />;

  const allUsers = data?.pages.flatMap(page => page.data) ?? [];

  return (
    <div>
      {allUsers.map(user => (
        <UserCard key={user.id} user={user} />
      ))}
      <div ref={ref}>
        {isFetchingNextPage ? <Spinner /> : hasNextPage ? 'Load more' : 'No more users'}
      </div>
    </div>
  );
}
```

---

## 4. プリフェッチ

```typescript
// マウス hover でプリフェッチ
function UserListItem({ user }: { user: User }) {
  const queryClient = useQueryClient();

  const prefetchUser = () => {
    queryClient.prefetchQuery({
      queryKey: ['users', user.id],
      queryFn: () => api.users.get(user.id),
      staleTime: 60 * 1000, // 1分間はキャッシュ
    });
  };

  return (
    <Link
      to={`/users/${user.id}`}
      onMouseEnter={prefetchUser}  // hover でプリフェッチ
      onFocus={prefetchUser}       // フォーカスでもプリフェッチ
    >
      {user.name}
    </Link>
  );
}

// Next.js Server Component でのプリフェッチ
// app/users/page.tsx
import { HydrationBoundary, QueryClient, dehydrate } from '@tanstack/react-query';

export default async function UsersPage() {
  const queryClient = new QueryClient();

  await queryClient.prefetchQuery({
    queryKey: ['users'],
    queryFn: getUsers,
  });

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <UserList />
    </HydrationBoundary>
  );
}
```

---

## 5. TanStack Query vs SWR

```
                 TanStack Query    SWR
──────────────────────────────────────────
開発元            TanStack          Vercel
サイズ            ~39KB             ~12KB
Devtools          ✓（優秀）         △
Mutation          useMutation       手動
楽観的更新        組み込み           手動
無限スクロール    useInfiniteQuery   useSWRInfinite
オフライン        ✓                 △
プリフェッチ      ✓                 ✓
依存クエリ        ✓（enabled）       ✓
Suspense          ✓                 ✓
SSR統合           ✓                 ✓

推奨:
  → 機能重視: TanStack Query（推奨）
  → 軽量重視: SWR
  → Next.js プロジェクト: どちらでもOK（SWRはVercel製で親和性高い）
```

---

## まとめ

| パターン | 用途 |
|---------|------|
| staleTime | データの鮮度期限 |
| invalidateQueries | Mutation後のキャッシュ無効化 |
| useInfiniteQuery | 無限スクロール |
| prefetchQuery | ホバー/ルート遷移前のプリフェッチ |
| placeholderData | ページ遷移時の前データ表示 |

---

## 次に読むべきガイド
→ [[03-url-state.md]] — URL状態

---

## 参考文献
1. TkDodo. "Practical React Query." tkdodo.eu, 2024.
2. TanStack. "Query Documentation." tanstack.com, 2024.
3. SWR. "React Hooks for Data Fetching." swr.vercel.app, 2024.
