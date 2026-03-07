# サーバー状態管理

> サーバー状態はクライアント状態とは根本的に異なる。TanStack QueryとSWRのキャッシュ戦略、stale-while-revalidateパターン、無限スクロール、ポーリング、プリフェッチまで、サーバーデータ管理の全技法を習得する。

## この章で学ぶこと

- [ ] サーバー状態の特性とクライアント状態との本質的な違いを理解する
- [ ] キャッシュ戦略の設計原則（staleTime, gcTime, invalidation）を把握する
- [ ] TanStack Query の基本 API から高度なパターンまでを習得する
- [ ] 楽観的更新（Optimistic Updates）の実装パターンを学ぶ
- [ ] 無限スクロールとページネーションの実装技法を身につける
- [ ] プリフェッチによるUX最適化の手法を理解する
- [ ] SWR との比較を通じてライブラリ選定の判断基準を確立する
- [ ] サーバー状態管理のアンチパターンとトラブルシューティングを学ぶ

---

## 1. サーバー状態の本質

### 1.1 サーバー状態とクライアント状態の根本的な違い

Webアプリケーションにおける「状態」は、その所有者と特性によって大きく2つに分類される。この区別を正しく理解することが、適切な状態管理設計の第一歩である。

```
サーバー状態 vs クライアント状態:

  クライアント状態（Client State）:
  ┌─────────────────────────────────────────────────┐
  │  所有者: アプリケーション自身                        │
  │  アクセス: 同期的（メモリ上に即座にアクセス可能）       │
  │  鮮度:   常に最新（唯一の情報源 = Single Source of Truth）│
  │  例:     UIの開閉状態、フォーム入力、テーマ設定        │
  │  更新:   ユーザー操作に即座に反映                     │
  │  永続性: セッション中のみ（リロードで消える）           │
  └─────────────────────────────────────────────────┘

  サーバー状態（Server State）:
  ┌─────────────────────────────────────────────────┐
  │  所有者: リモートサーバー（クライアントはコピーを保持）  │
  │  アクセス: 非同期的（ネットワーク経由で取得）            │
  │  鮮度:   時間とともに古くなる（stale になる）           │
  │  例:     ユーザー一覧、商品データ、通知一覧            │
  │  更新:   他のクライアントが同時に変更する可能性がある    │
  │  永続性: サーバー側で永続化されている                  │
  └─────────────────────────────────────────────────┘
```

### 1.2 サーバー状態管理の5つの課題

サーバー状態を適切に管理するためには、以下の5つの課題を解決する必要がある。

```
サーバー状態管理の5大課題:

  1. キャッシュ管理（Caching）
     → いつデータを再取得するか？
     → キャッシュの有効期限をどう設定するか？
     → メモリ使用量をどう制御するか？

  2. データ同期（Synchronization）
     → クライアント側のコピーをどう最新に保つか？
     → 複数タブ間でデータを同期するか？
     → バックグラウンドでの自動再取得をどう実装するか？

  3. 重複排除（Deduplication）
     → 同じデータへの複数リクエストをどうまとめるか？
     → 複数コンポーネントが同じデータを必要とする場合は？
     → ネットワーク帯域をどう節約するか？

  4. 楽観的更新（Optimistic Updates）
     → APIレスポンス前にUIを更新して体感速度を向上させるには？
     → 更新失敗時のロールバックをどう実装するか？
     → 競合状態（Race Condition）をどう防ぐか？

  5. ライフサイクル管理（Lifecycle Management）
     → コンポーネントのマウント/アンマウント時の挙動は？
     → 不要になったキャッシュをいつ破棄するか？
     → メモリリークをどう防ぐか？
```

### 1.3 なぜ専用ライブラリが必要か

サーバー状態の管理を素のReact（useEffect + useState）で実装しようとすると、多くの問題に直面する。

```typescript
// アンチパターン: useEffect + useState によるデータ取得
// 問題だらけのコード例

function UserList() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    let cancelled = false; // クリーンアップ用のフラグ

    const fetchUsers = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch('/api/users');
        if (!response.ok) throw new Error('Failed to fetch');
        const data = await response.json();

        if (!cancelled) {
          setUsers(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchUsers();

    return () => {
      cancelled = true; // アンマウント時にキャンセル
    };
  }, []);

  // 問題点:
  // 1. キャッシュがない → 毎回ネットワークリクエストが発生
  // 2. 他のコンポーネントとデータ共有できない
  // 3. ウィンドウフォーカス時の再取得がない
  // 4. エラー時のリトライがない
  // 5. 楽観的更新が実装困難
  // 6. ローディング/エラー/データの3状態を毎回管理
  // 7. Race Conditionの処理が手動
  // 8. ページネーション/無限スクロールの実装が複雑

  if (loading) return <Loading />;
  if (error) return <Error message={error.message} />;
  return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>;
}
```

```typescript
// 推奨: TanStack Query を使った場合
// 上記の全問題が解決される

function UserList() {
  const { data: users, isLoading, error } = useQuery({
    queryKey: ['users'],
    queryFn: () => fetch('/api/users').then(res => res.json()),
    staleTime: 30 * 1000,          // 30秒間キャッシュを新鮮扱い
    retry: 3,                       // エラー時に3回リトライ
    refetchOnWindowFocus: true,     // ウィンドウフォーカスで再取得
  });

  // メリット:
  // 1. 自動キャッシュ管理
  // 2. 複数コンポーネント間でキャッシュ共有
  // 3. バックグラウンド再取得
  // 4. 自動リトライ
  // 5. DevToolsによるデバッグ
  // 6. TypeScript完全対応

  if (isLoading) return <Loading />;
  if (error) return <Error message={error.message} />;
  return <ul>{users.map(u => <li key={u.id}>{u.name}</li>)}</ul>;
}
```

### 1.4 stale-while-revalidate パターンの詳細

サーバー状態管理ライブラリの核となる概念が「stale-while-revalidate」パターンである。これはHTTPのCache-Controlヘッダーに由来する概念で、キャッシュされた古い（stale）データを即座に返しつつ、バックグラウンドで最新データを再取得（revalidate）する戦略である。

```
stale-while-revalidate フロー:

  1回目のリクエスト:
  ┌─────────┐     GET /api/users      ┌──────────┐
  │ Client  │ ──────────────────────→  │  Server  │
  │         │ ←────────────────────── │          │
  └─────────┘     Response + Data     └──────────┘
       │
       ▼
  ┌─────────────────────┐
  │ キャッシュに保存       │
  │ status: "fresh"      │
  │ staleTime: 30s       │
  └─────────────────────┘

  30秒以内の2回目のリクエスト（fresh期間）:
  ┌─────────┐                          ┌──────────┐
  │ Client  │ ─→ キャッシュから即座に返す │  Server  │
  │         │    （リクエストなし）        │          │
  └─────────┘                          └──────────┘

  30秒経過後のリクエスト（stale期間）:
  ┌─────────┐                          ┌──────────┐
  │ Client  │ ─→ 古いキャッシュを即座に返す               │
  │         │ ──── バックグラウンドで再取得 ──→ │ Server │
  │         │ ←──── 新しいデータ受信 ────────  │        │
  │         │ → UIを自動更新                  └────────┘
  └─────────┘

  利点:
  → ユーザーは常に即座にデータを見られる（UXの向上）
  → バックグラウンドで最新データに更新される
  → ネットワーク遅延を体感させない
```

```typescript
// stale-while-revalidate を体感する設定例

// ケース1: staleTime = 0（デフォルト）
// → キャッシュは常にstale → 毎回バックグラウンドで再取得
const { data } = useQuery({
  queryKey: ['users'],
  queryFn: fetchUsers,
  staleTime: 0, // デフォルト値
});

// ケース2: staleTime = Infinity
// → キャッシュは永遠にfresh → 明示的にinvalidateしない限り再取得しない
const { data } = useQuery({
  queryKey: ['config'],
  queryFn: fetchConfig,
  staleTime: Infinity,
});

// ケース3: 現実的な設定
// → 5分間はfresh → 5分経過後にバックグラウンド再取得
const { data } = useQuery({
  queryKey: ['products'],
  queryFn: fetchProducts,
  staleTime: 5 * 60 * 1000, // 5分
});
```

---

## 2. TanStack Query の基礎から応用まで

### 2.1 セットアップとプロバイダー設定

```typescript
// src/lib/query-client.ts
import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // グローバルなデフォルト設定
      staleTime: 60 * 1000,           // 1分間はfresh
      gcTime: 5 * 60 * 1000,          // 5分間キャッシュを保持
      retry: 3,                        // 3回リトライ
      retryDelay: (attemptIndex) =>    // 指数バックオフ
        Math.min(1000 * 2 ** attemptIndex, 30000),
      refetchOnWindowFocus: true,      // ウィンドウフォーカスで再取得
      refetchOnReconnect: true,        // ネットワーク再接続で再取得
      refetchOnMount: true,            // マウント時に再取得
    },
    mutations: {
      retry: 1,                        // Mutationは1回リトライ
    },
  },
});
```

```typescript
// src/app/providers.tsx
'use client'; // Next.js App Router の場合

import { QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import { queryClient } from '@/lib/query-client';

export function Providers({ children }: { children: React.ReactNode }) {
  // 注意: QueryClientをコンポーネント内で生成しない（SSR対策）
  // useState で1回だけ生成するか、モジュールレベルで定義する
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools
          initialIsOpen={false}
          position="bottom"
          buttonPosition="bottom-right"
        />
      )}
    </QueryClientProvider>
  );
}
```

### 2.2 Query Key の設計パターン

Query Key はキャッシュの識別子であり、その設計はアプリケーション全体のキャッシュ戦略に直結する。

```typescript
// Query Key の設計原則

// 原則1: 階層構造を使う（配列で表現）
// → invalidateQueries で部分的に無効化できる

// 基本パターン
['users']                          // ユーザー一覧
['users', userId]                  // 特定ユーザー
['users', userId, 'posts']         // 特定ユーザーの投稿一覧
['users', userId, 'posts', postId] // 特定ユーザーの特定投稿

// フィルター・ソート・ページネーション含む
['users', { page: 1, sort: 'name', filter: 'active' }]
['users', { search: 'John', role: 'admin' }]

// 無効化の粒度
queryClient.invalidateQueries({ queryKey: ['users'] });
// → ['users'], ['users', 1], ['users', 1, 'posts'] 全て無効化

queryClient.invalidateQueries({ queryKey: ['users', 1] });
// → ['users', 1], ['users', 1, 'posts'] のみ無効化
```

```typescript
// 推奨: Query Key Factory パターン
// src/lib/query-keys.ts

export const userKeys = {
  all: ['users'] as const,
  lists: () => [...userKeys.all, 'list'] as const,
  list: (filters: UserFilters) => [...userKeys.lists(), filters] as const,
  details: () => [...userKeys.all, 'detail'] as const,
  detail: (id: string) => [...userKeys.details(), id] as const,
  posts: (id: string) => [...userKeys.detail(id), 'posts'] as const,
} as const;

export const productKeys = {
  all: ['products'] as const,
  lists: () => [...productKeys.all, 'list'] as const,
  list: (filters: ProductFilters) => [...productKeys.lists(), filters] as const,
  details: () => [...productKeys.all, 'detail'] as const,
  detail: (id: string) => [...productKeys.details(), id] as const,
  reviews: (id: string) => [...productKeys.detail(id), 'reviews'] as const,
  related: (id: string) => [...productKeys.detail(id), 'related'] as const,
} as const;

// 型定義
type UserFilters = {
  page?: number;
  search?: string;
  role?: 'admin' | 'user';
  sort?: 'name' | 'createdAt';
  order?: 'asc' | 'desc';
};

type ProductFilters = {
  category?: string;
  minPrice?: number;
  maxPrice?: number;
  inStock?: boolean;
};
```

```typescript
// Query Key Factory の使用例

// 一覧取得
const { data } = useQuery({
  queryKey: userKeys.list({ page: 1, role: 'admin' }),
  queryFn: () => api.users.list({ page: 1, role: 'admin' }),
});

// 詳細取得
const { data } = useQuery({
  queryKey: userKeys.detail(userId),
  queryFn: () => api.users.get(userId),
});

// 無効化（ユーザー関連のキャッシュを全て無効化）
queryClient.invalidateQueries({ queryKey: userKeys.all });

// 無効化（一覧のみ）
queryClient.invalidateQueries({ queryKey: userKeys.lists() });

// 無効化（特定ユーザーのみ）
queryClient.invalidateQueries({ queryKey: userKeys.detail(userId) });
```

### 2.3 キャッシュのライフサイクル詳細

```
キャッシュのライフサイクル:

  ┌──────────────────────────────────────────────────────────────┐
  │                    キャッシュエントリの状態遷移                  │
  ├──────────────────────────────────────────────────────────────┤
  │                                                              │
  │  [初回フェッチ]                                               │
  │       │                                                      │
  │       ▼                                                      │
  │  ┌─────────┐  staleTime経過  ┌─────────┐                    │
  │  │  Fresh   │ ──────────→   │  Stale   │                    │
  │  │ (新鮮)   │               │ (古い)    │                    │
  │  └─────────┘               └─────────┘                    │
  │       │                         │                            │
  │       │                         │  トリガー発火                │
  │       │                         │  (windowFocus, mount等)     │
  │       │                         ▼                            │
  │       │                    ┌──────────┐                      │
  │       │                    │ Fetching │                      │
  │       │                    │ (再取得中) │                      │
  │       │                    └──────────┘                      │
  │       │                         │                            │
  │       │                         ▼                            │
  │       │                    ┌─────────┐                      │
  │       │                    │  Fresh   │ ← 再びfreshに        │
  │       │                    └─────────┘                      │
  │       │                                                      │
  │  [オブザーバーなし（全コンポーネントがアンマウント）]              │
  │       │                                                      │
  │       ▼                                                      │
  │  ┌──────────┐  gcTime経過   ┌──────────┐                    │
  │  │ Inactive  │ ──────────→  │ Garbage  │                    │
  │  │ (非活性)   │              │ Collected │                    │
  │  └──────────┘              └──────────┘                    │
  │                                                              │
  └──────────────────────────────────────────────────────────────┘
```

```typescript
// staleTime と gcTime の関係を理解する設定例

// パターン1: 高頻度更新データ（チャット、通知）
const notificationsQuery = {
  queryKey: ['notifications'],
  queryFn: fetchNotifications,
  staleTime: 0,                    // 常にstale → 毎回バックグラウンド再取得
  gcTime: 5 * 60 * 1000,           // 5分間キャッシュ保持
  refetchInterval: 10 * 1000,      // 10秒ごとにポーリング
  refetchIntervalInBackground: false, // バックグラウンドタブではポーリングしない
};

// パターン2: 中頻度更新データ（ユーザー一覧、投稿一覧）
const usersQuery = {
  queryKey: ['users'],
  queryFn: fetchUsers,
  staleTime: 30 * 1000,            // 30秒間fresh
  gcTime: 10 * 60 * 1000,          // 10分間キャッシュ保持
};

// パターン3: 低頻度更新データ（マスタデータ、カテゴリ）
const categoriesQuery = {
  queryKey: ['categories'],
  queryFn: fetchCategories,
  staleTime: 24 * 60 * 60 * 1000,  // 24時間fresh
  gcTime: Infinity,                 // 永久にキャッシュ
};

// パターン4: ユーザー操作依存データ（CRUD対象）
const userDetailQuery = (id: string) => ({
  queryKey: userKeys.detail(id),
  queryFn: () => fetchUser(id),
  staleTime: 60 * 1000,            // 1分間fresh
  gcTime: 5 * 60 * 1000,           // 5分間キャッシュ保持
  placeholderData: keepPreviousData, // 前のデータをプレースホルダーに
});
```

### 2.4 useQuery の全オプション詳解

```typescript
import { useQuery, keepPreviousData } from '@tanstack/react-query';

const {
  // === 返却されるデータ ===
  data,                    // 取得したデータ（型安全）
  dataUpdatedAt,           // データ最終更新のタイムスタンプ
  error,                   // エラーオブジェクト
  errorUpdatedAt,          // エラー最終更新のタイムスタンプ
  failureCount,            // 連続失敗回数
  failureReason,           // 直近の失敗理由

  // === ステータスフラグ ===
  status,                  // 'pending' | 'error' | 'success'
  fetchStatus,             // 'fetching' | 'paused' | 'idle'
  isLoading,               // status === 'pending' && fetchStatus === 'fetching'
  isFetching,              // fetchStatus === 'fetching'（バックグラウンド再取得含む）
  isPending,               // status === 'pending'
  isError,                 // status === 'error'
  isSuccess,               // status === 'success'
  isRefetching,            // isFetching && !isLoading
  isStale,                 // データがstaleかどうか
  isPaused,                // fetchStatus === 'paused'
  isPlaceholderData,       // placeholderDataが使われているか
  isFetched,               // 最低1回はフェッチ完了したか
  isFetchedAfterMount,     // マウント後にフェッチ完了したか

  // === メソッド ===
  refetch,                 // 手動で再取得
} = useQuery({
  // === 必須オプション ===
  queryKey: ['users', userId],  // キャッシュキー（配列）
  queryFn: ({ signal }) =>      // データ取得関数（AbortSignal対応推奨）
    fetch(`/api/users/${userId}`, { signal }).then(r => r.json()),

  // === キャッシュ制御 ===
  staleTime: 60 * 1000,         // データがfreshとみなされる期間（ms）
  gcTime: 5 * 60 * 1000,        // 非活性キャッシュの保持期間（ms）

  // === 再取得制御 ===
  refetchOnWindowFocus: true,    // ウィンドウフォーカス時
  refetchOnReconnect: true,      // ネットワーク再接続時
  refetchOnMount: true,          // コンポーネントマウント時
  refetchInterval: false,        // ポーリング間隔（ms, falseで無効）
  refetchIntervalInBackground: false, // バックグラウンドタブでもポーリングするか

  // === リトライ制御 ===
  retry: 3,                      // リトライ回数（true=無限, false=0回）
  retryDelay: (attempt) =>       // リトライ間隔（指数バックオフ推奨）
    Math.min(1000 * 2 ** attempt, 30000),
  retryOnMount: true,            // マウント時にリトライするか

  // === 条件付きクエリ ===
  enabled: !!userId,             // falseでクエリ実行を停止

  // === データ変換 ===
  select: (data) => data.users,  // 返却データの変換・フィルタリング

  // === プレースホルダー ===
  placeholderData: keepPreviousData,  // 前のデータをプレースホルダーに
  // または固定値: placeholderData: { users: [] },
  // または関数: placeholderData: (previousData) => previousData,

  // === initialData ===
  initialData: undefined,        // 初期データ（キャッシュに保存される）
  initialDataUpdatedAt: undefined, // initialDataのタイムスタンプ

  // === 構造共有 ===
  structuralSharing: true,       // 参照同一性の最適化

  // === ネットワークモード ===
  networkMode: 'online',         // 'online' | 'always' | 'offlineFirst'
});
```

### 2.5 enabled オプションによる依存クエリ

```typescript
// 依存クエリ: あるクエリの結果に基づいて次のクエリを実行

// ステップ1: ユーザー情報を取得
function useUserWithPosts(userId: string) {
  // まずユーザー情報を取得
  const userQuery = useQuery({
    queryKey: userKeys.detail(userId),
    queryFn: () => api.users.get(userId),
  });

  // ユーザーの所属組織IDが取得できたら、組織情報を取得
  const organizationQuery = useQuery({
    queryKey: ['organizations', userQuery.data?.organizationId],
    queryFn: () => api.organizations.get(userQuery.data!.organizationId),
    enabled: !!userQuery.data?.organizationId, // ユーザーデータがあるときだけ実行
  });

  // ユーザーの投稿一覧を取得
  const postsQuery = useQuery({
    queryKey: userKeys.posts(userId),
    queryFn: () => api.users.getPosts(userId),
    enabled: userQuery.isSuccess, // ユーザー取得成功後に実行
  });

  return {
    user: userQuery.data,
    organization: organizationQuery.data,
    posts: postsQuery.data,
    isLoading: userQuery.isLoading,
    isError: userQuery.isError,
  };
}
```

```typescript
// 依存クエリの別パターン: 検索フォーム

function useSearchResults(searchTerm: string) {
  // 検索語が2文字以上の場合のみクエリ実行
  return useQuery({
    queryKey: ['search', searchTerm],
    queryFn: () => api.search(searchTerm),
    enabled: searchTerm.length >= 2, // 2文字未満では実行しない
    staleTime: 5 * 60 * 1000,       // 検索結果は5分間キャッシュ
    placeholderData: keepPreviousData, // 検索語が変わっても前の結果を表示
  });
}

function SearchPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const debouncedSearch = useDebounce(searchTerm, 300); // 300msデバウンス
  const { data, isLoading, isPlaceholderData } = useSearchResults(debouncedSearch);

  return (
    <div>
      <input
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder="Search..."
      />
      <div style={{ opacity: isPlaceholderData ? 0.5 : 1 }}>
        {data?.results.map(result => (
          <SearchResultCard key={result.id} result={result} />
        ))}
      </div>
    </div>
  );
}
```

### 2.6 select オプションによるデータ変換

```typescript
// select を使ってサーバーレスポンスをクライアント用に変換

// 例1: 必要なフィールドだけ抽出
const { data: userNames } = useQuery({
  queryKey: userKeys.lists(),
  queryFn: () => api.users.list(),
  select: (data) => data.map(user => ({
    id: user.id,
    name: `${user.firstName} ${user.lastName}`,
  })),
});

// 例2: フィルタリング
const { data: activeUsers } = useQuery({
  queryKey: userKeys.lists(),
  queryFn: () => api.users.list(),
  select: (data) => data.filter(user => user.status === 'active'),
});

// 例3: 集計
const { data: userCount } = useQuery({
  queryKey: userKeys.lists(),
  queryFn: () => api.users.list(),
  select: (data) => data.length,
});

// 重要: select はキャッシュされたデータに対して実行される
// → 同じqueryKeyで異なるselectを使う複数のコンポーネントは
//   ネットワークリクエストを1回しか発行しない

// パフォーマンス注意: selectの安定した参照
// useCallback で参照を安定させることで不要な再計算を防ぐ
const selectActiveUsers = useCallback(
  (data: User[]) => data.filter(u => u.status === 'active'),
  []
);

const { data } = useQuery({
  queryKey: userKeys.lists(),
  queryFn: () => api.users.list(),
  select: selectActiveUsers,
});
```

---

## 3. Mutation（データ更新）の完全ガイド

### 3.1 useMutation の基本

```typescript
import { useMutation, useQueryClient } from '@tanstack/react-query';

// 基本的なMutationの定義
function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    // === Mutation関数 ===
    mutationFn: (newUser: CreateUserInput) =>
      api.users.create(newUser),

    // === コールバック ===
    onMutate: async (variables) => {
      // Mutationが開始される前に呼ばれる
      // 楽観的更新のためのキャッシュ操作に使う
      console.log('Creating user:', variables);
    },

    onSuccess: (data, variables, context) => {
      // Mutation成功時に呼ばれる
      // data: サーバーからの応答
      // variables: mutationFnに渡した引数
      // context: onMutateの返り値

      // キャッシュを無効化して再取得をトリガー
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });

      // または直接キャッシュを更新
      queryClient.setQueryData(
        userKeys.detail(data.id),
        data
      );
    },

    onError: (error, variables, context) => {
      // Mutation失敗時に呼ばれる
      console.error('Failed to create user:', error);
    },

    onSettled: (data, error, variables, context) => {
      // 成功・失敗に関わらず最後に呼ばれる
      // キャッシュの無効化はここで行うのも良い
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
    },

    // === リトライ ===
    retry: 1,

    // === ネットワークモード ===
    networkMode: 'online',
  });
}
```

```typescript
// Mutationの使用例
function CreateUserForm() {
  const createUser = useCreateUser();

  const handleSubmit = async (formData: CreateUserInput) => {
    try {
      const newUser = await createUser.mutateAsync(formData);
      // mutateAsync は Promise を返す → try/catch で使える
      toast.success(`ユーザー「${newUser.name}」を作成しました`);
      router.push(`/users/${newUser.id}`);
    } catch (error) {
      toast.error('ユーザーの作成に失敗しました');
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* フォームフィールド */}
      <button
        type="submit"
        disabled={createUser.isPending}
      >
        {createUser.isPending ? '作成中...' : 'ユーザーを作成'}
      </button>
      {createUser.isError && (
        <p className="error">{createUser.error.message}</p>
      )}
    </form>
  );
}
```

### 3.2 CRUD操作の完全な実装パターン

```typescript
// src/hooks/useUserMutations.ts
// CRUD操作を1つのカスタムフックにまとめるパターン

import { useMutation, useQueryClient } from '@tanstack/react-query';
import { userKeys } from '@/lib/query-keys';
import { api } from '@/lib/api';

// === Create ===
export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: api.users.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
    },
  });
}

// === Update ===
export function useUpdateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: UpdateUserInput }) =>
      api.users.update(id, data),
    onSuccess: (updatedUser) => {
      // 詳細キャッシュを直接更新
      queryClient.setQueryData(
        userKeys.detail(updatedUser.id),
        updatedUser
      );
      // 一覧キャッシュを無効化
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
    },
  });
}

// === Delete ===
export function useDeleteUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => api.users.delete(id),
    onSuccess: (_, deletedId) => {
      // 詳細キャッシュを削除
      queryClient.removeQueries({ queryKey: userKeys.detail(deletedId) });
      // 一覧キャッシュを無効化
      queryClient.invalidateQueries({ queryKey: userKeys.lists() });
    },
  });
}

// === Bulk Operations ===
export function useBulkDeleteUsers() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (ids: string[]) => api.users.bulkDelete(ids),
    onSuccess: () => {
      // ユーザー関連のキャッシュを全て無効化
      queryClient.invalidateQueries({ queryKey: userKeys.all });
    },
  });
}
```

### 3.3 楽観的更新（Optimistic Updates）の完全実装

楽観的更新は、サーバーの応答を待たずにUIを即座に更新するパターンである。ユーザー体験を大幅に向上させるが、実装には注意が必要である。

```typescript
// 楽観的更新の完全な実装パターン

export function useUpdateTodo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<Todo> }) =>
      api.todos.update(id, data),

    // ステップ1: onMutate で楽観的にキャッシュを更新
    onMutate: async ({ id, data }) => {
      // 進行中のリフェッチをキャンセル（楽観的更新を上書きしないように）
      await queryClient.cancelQueries({ queryKey: ['todos'] });
      await queryClient.cancelQueries({ queryKey: ['todos', id] });

      // 現在のキャッシュを保存（ロールバック用）
      const previousTodos = queryClient.getQueryData<Todo[]>(['todos']);
      const previousTodo = queryClient.getQueryData<Todo>(['todos', id]);

      // 一覧キャッシュを楽観的に更新
      if (previousTodos) {
        queryClient.setQueryData<Todo[]>(['todos'], (old) =>
          old?.map(todo =>
            todo.id === id ? { ...todo, ...data } : todo
          )
        );
      }

      // 詳細キャッシュを楽観的に更新
      if (previousTodo) {
        queryClient.setQueryData<Todo>(['todos', id], (old) =>
          old ? { ...old, ...data } : old
        );
      }

      // ロールバック用のコンテキストを返す
      return { previousTodos, previousTodo };
    },

    // ステップ2: onError でロールバック
    onError: (error, { id }, context) => {
      // エラー時にキャッシュを元に戻す
      if (context?.previousTodos) {
        queryClient.setQueryData(['todos'], context.previousTodos);
      }
      if (context?.previousTodo) {
        queryClient.setQueryData(['todos', id], context.previousTodo);
      }

      // エラー通知
      toast.error('更新に失敗しました。変更を元に戻しました。');
    },

    // ステップ3: onSettled でキャッシュを再検証
    onSettled: (_, __, { id }) => {
      // 成功/失敗に関わらず、サーバーの最新データで同期
      queryClient.invalidateQueries({ queryKey: ['todos'] });
      queryClient.invalidateQueries({ queryKey: ['todos', id] });
    },
  });
}
```

```typescript
// 楽観的更新: リスト要素の追加

export function useAddTodo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (newTodo: CreateTodoInput) => api.todos.create(newTodo),

    onMutate: async (newTodo) => {
      await queryClient.cancelQueries({ queryKey: ['todos'] });

      const previousTodos = queryClient.getQueryData<Todo[]>(['todos']);

      // 仮のIDでリストに追加（UIがすぐに反映される）
      const optimisticTodo: Todo = {
        id: `temp-${Date.now()}`, // 仮のID
        ...newTodo,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      queryClient.setQueryData<Todo[]>(['todos'], (old) =>
        old ? [...old, optimisticTodo] : [optimisticTodo]
      );

      return { previousTodos };
    },

    onSuccess: (serverTodo) => {
      // サーバーから返された正式なデータでキャッシュを更新
      // （仮IDが本物のIDに置き換わる）
      queryClient.setQueryData<Todo[]>(['todos'], (old) =>
        old?.map(todo =>
          todo.id.startsWith('temp-') ? serverTodo : todo
        )
      );
    },

    onError: (error, newTodo, context) => {
      if (context?.previousTodos) {
        queryClient.setQueryData(['todos'], context.previousTodos);
      }
      toast.error('追加に失敗しました');
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['todos'] });
    },
  });
}
```

```typescript
// 楽観的更新: リスト要素の削除

export function useDeleteTodo() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: string) => api.todos.delete(id),

    onMutate: async (id) => {
      await queryClient.cancelQueries({ queryKey: ['todos'] });

      const previousTodos = queryClient.getQueryData<Todo[]>(['todos']);

      // リストから即座に削除
      queryClient.setQueryData<Todo[]>(['todos'], (old) =>
        old?.filter(todo => todo.id !== id)
      );

      return { previousTodos };
    },

    onError: (error, id, context) => {
      // 失敗時にリストを復元
      if (context?.previousTodos) {
        queryClient.setQueryData(['todos'], context.previousTodos);
      }
      toast.error('削除に失敗しました');
    },

    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['todos'] });
    },
  });
}
```

### 3.4 Mutation の状態管理とUIパターン

```typescript
// Mutationの全状態を活用したUI表示

function TodoItem({ todo }: { todo: Todo }) {
  const updateTodo = useUpdateTodo();
  const deleteTodo = useDeleteTodo();

  return (
    <div
      className={cn(
        'flex items-center gap-2 p-3 rounded-lg',
        deleteTodo.isPending && 'opacity-50 pointer-events-none',
      )}
    >
      <Checkbox
        checked={todo.completed}
        disabled={updateTodo.isPending}
        onCheckedChange={(checked) => {
          updateTodo.mutate({
            id: todo.id,
            data: { completed: checked as boolean },
          });
        }}
      />

      <span className={cn(
        todo.completed && 'line-through text-muted-foreground',
        updateTodo.isPending && 'animate-pulse',
      )}>
        {todo.title}
      </span>

      <button
        onClick={() => {
          if (confirm('本当に削除しますか？')) {
            deleteTodo.mutate(todo.id);
          }
        }}
        disabled={deleteTodo.isPending}
      >
        {deleteTodo.isPending ? <Spinner /> : <TrashIcon />}
      </button>

      {/* エラー表示 */}
      {updateTodo.isError && (
        <span className="text-red-500 text-sm">
          更新失敗
          <button onClick={() => updateTodo.reset()}>閉じる</button>
        </span>
      )}
    </div>
  );
}
```

---

## 4. 無限スクロールとページネーション

### 4.1 useInfiniteQuery の詳細

```typescript
import { useInfiniteQuery, keepPreviousData } from '@tanstack/react-query';

// カーソルベースの無限スクロール
function useInfiniteUsers(filters?: UserFilters) {
  return useInfiniteQuery({
    queryKey: ['users', 'infinite', filters],

    queryFn: async ({ pageParam, signal }) => {
      const response = await api.users.list({
        cursor: pageParam,
        limit: 20,
        ...filters,
        signal, // AbortSignal を渡してキャンセル対応
      });
      return response;
    },

    // 初期ページパラメータ
    initialPageParam: undefined as string | undefined,

    // 次のページパラメータを決定
    getNextPageParam: (lastPage) =>
      lastPage.meta.hasNextPage ? lastPage.meta.nextCursor : undefined,

    // 前のページパラメータ（双方向スクロールの場合）
    getPreviousPageParam: (firstPage) =>
      firstPage.meta.hasPreviousPage ? firstPage.meta.previousCursor : undefined,

    // キャッシュ設定
    staleTime: 30 * 1000,
    gcTime: 10 * 60 * 1000,

    // 最大ページ数を制限（メモリ対策）
    maxPages: 10,
  });
}
```

```typescript
// Intersection Observer による自動読み込み

import { useInView } from 'react-intersection-observer';

function UserInfiniteList() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
    error,
    isFetching,
  } = useInfiniteUsers();

  // Intersection Observer: 要素が画面に表示されたら次のページを読み込む
  const { ref: loadMoreRef } = useInView({
    threshold: 0,
    rootMargin: '200px', // 200px手前で発火（先読み）
    onChange: (inView) => {
      if (inView && hasNextPage && !isFetchingNextPage) {
        fetchNextPage();
      }
    },
  });

  // 全ページのデータをフラットに結合
  const allUsers = data?.pages.flatMap(page => page.data) ?? [];

  if (isLoading) {
    return (
      <div className="grid grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <UserCardSkeleton key={i} />
        ))}
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorMessage
        error={error}
        onRetry={() => fetchNextPage()}
      />
    );
  }

  return (
    <div>
      {/* バックグラウンド再取得インジケーター */}
      {isFetching && !isFetchingNextPage && (
        <div className="fixed top-0 left-0 right-0 h-1 bg-blue-500 animate-pulse" />
      )}

      {/* ユーザーリスト */}
      <div className="grid grid-cols-3 gap-4">
        {allUsers.map(user => (
          <UserCard key={user.id} user={user} />
        ))}
      </div>

      {/* 読み込みトリガー */}
      <div ref={loadMoreRef} className="py-8 text-center">
        {isFetchingNextPage ? (
          <Spinner />
        ) : hasNextPage ? (
          <p className="text-muted-foreground">
            スクロールして続きを読み込む
          </p>
        ) : (
          <p className="text-muted-foreground">
            すべてのユーザーを表示しました（{allUsers.length}件）
          </p>
        )}
      </div>
    </div>
  );
}
```

### 4.2 オフセットベースのページネーション

```typescript
// オフセットベースのページネーション（従来型）

function usePagedUsers(page: number, pageSize: number = 20) {
  return useQuery({
    queryKey: ['users', 'paged', { page, pageSize }],
    queryFn: () => api.users.list({
      offset: (page - 1) * pageSize,
      limit: pageSize,
    }),
    placeholderData: keepPreviousData, // ページ遷移時にちらつかない
    staleTime: 30 * 1000,
  });
}

function UserPagedList() {
  const [page, setPage] = useState(1);
  const pageSize = 20;

  const {
    data,
    isLoading,
    isPlaceholderData,
    isFetching,
  } = usePagedUsers(page, pageSize);

  // 次のページをプリフェッチ
  const queryClient = useQueryClient();
  useEffect(() => {
    if (data?.meta.hasNextPage) {
      queryClient.prefetchQuery({
        queryKey: ['users', 'paged', { page: page + 1, pageSize }],
        queryFn: () => api.users.list({
          offset: page * pageSize,
          limit: pageSize,
        }),
      });
    }
  }, [data, page, pageSize, queryClient]);

  if (isLoading) return <TableSkeleton rows={pageSize} />;

  return (
    <div style={{ opacity: isPlaceholderData ? 0.7 : 1 }}>
      {isFetching && <ProgressBar />}

      <table>
        <thead>
          <tr>
            <th>名前</th>
            <th>メール</th>
            <th>ロール</th>
          </tr>
        </thead>
        <tbody>
          {data?.data.map(user => (
            <tr key={user.id}>
              <td>{user.name}</td>
              <td>{user.email}</td>
              <td>{user.role}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <Pagination
        currentPage={page}
        totalPages={data?.meta.totalPages ?? 1}
        onPageChange={setPage}
        disabled={isPlaceholderData}
      />
    </div>
  );
}
```

### 4.3 仮想化（Virtualization）との組み合わせ

大量のデータを表示する場合、仮想化ライブラリと組み合わせることでパフォーマンスを大幅に向上できる。

```typescript
// @tanstack/react-virtual との組み合わせ

import { useVirtualizer } from '@tanstack/react-virtual';

function VirtualizedInfiniteList() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
  } = useInfiniteUsers();

  const allItems = data?.pages.flatMap(page => page.data) ?? [];

  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: hasNextPage ? allItems.length + 1 : allItems.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 80, // 各行の推定高さ（px）
    overscan: 5,            // 画面外に余分にレンダリングする行数
  });

  // 最後の要素が表示されたら次のページを読み込む
  useEffect(() => {
    const lastItem = virtualizer.getVirtualItems().at(-1);
    if (!lastItem) return;

    if (
      lastItem.index >= allItems.length - 1 &&
      hasNextPage &&
      !isFetchingNextPage
    ) {
      fetchNextPage();
    }
  }, [
    virtualizer.getVirtualItems(),
    hasNextPage,
    isFetchingNextPage,
    allItems.length,
    fetchNextPage,
  ]);

  return (
    <div
      ref={parentRef}
      className="h-[600px] overflow-auto"
    >
      <div
        style={{
          height: `${virtualizer.getTotalSize()}px`,
          width: '100%',
          position: 'relative',
        }}
      >
        {virtualizer.getVirtualItems().map((virtualRow) => {
          const isLoaderRow = virtualRow.index > allItems.length - 1;
          const item = allItems[virtualRow.index];

          return (
            <div
              key={virtualRow.index}
              data-index={virtualRow.index}
              ref={virtualizer.measureElement}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                transform: `translateY(${virtualRow.start}px)`,
              }}
            >
              {isLoaderRow ? (
                isFetchingNextPage ? <Spinner /> : null
              ) : (
                <UserCard user={item} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

---

## 5. プリフェッチとSSR統合

### 5.1 ユーザー操作に基づくプリフェッチ

プリフェッチは、ユーザーが実際にデータを必要とする前にバックグラウンドでデータを取得しておく技法である。適切に実装すればページ遷移が瞬時に感じられるようになる。

```typescript
// パターン1: マウスホバーでプリフェッチ
function UserListItem({ user }: { user: User }) {
  const queryClient = useQueryClient();

  const prefetchUserDetail = () => {
    queryClient.prefetchQuery({
      queryKey: userKeys.detail(user.id),
      queryFn: () => api.users.get(user.id),
      staleTime: 60 * 1000, // 1分間はキャッシュを新鮮扱い
    });
  };

  return (
    <Link
      to={`/users/${user.id}`}
      onMouseEnter={prefetchUserDetail}  // hover でプリフェッチ開始
      onFocus={prefetchUserDetail}       // キーボードフォーカスでもプリフェッチ
    >
      <div className="flex items-center gap-3 p-3 hover:bg-gray-50 rounded-lg">
        <Avatar src={user.avatarUrl} alt={user.name} />
        <div>
          <p className="font-medium">{user.name}</p>
          <p className="text-sm text-muted-foreground">{user.email}</p>
        </div>
      </div>
    </Link>
  );
}
```

```typescript
// パターン2: ルート遷移時のプリフェッチ（React Router v6）

import { useQueryClient } from '@tanstack/react-query';

// loader 関数でプリフェッチ
export const userDetailLoader =
  (queryClient: QueryClient) =>
  async ({ params }: LoaderFunctionArgs) => {
    const userId = params.userId!;

    // キャッシュが新鮮ならリクエストしない
    await queryClient.ensureQueryData({
      queryKey: userKeys.detail(userId),
      queryFn: () => api.users.get(userId),
      staleTime: 60 * 1000,
    });

    // 関連データも並列でプリフェッチ
    await Promise.all([
      queryClient.prefetchQuery({
        queryKey: userKeys.posts(userId),
        queryFn: () => api.users.getPosts(userId),
      }),
      queryClient.prefetchQuery({
        queryKey: ['organizations', userId],
        queryFn: () => api.users.getOrganization(userId),
      }),
    ]);

    return null; // loader の返り値は使わない（TanStack Query のキャッシュを使う）
  };

// ルーター設定
const router = createBrowserRouter([
  {
    path: '/users/:userId',
    element: <UserDetailPage />,
    loader: userDetailLoader(queryClient),
  },
]);
```

```typescript
// パターン3: スクロール位置に基づくプリフェッチ

function ProductGrid({ products }: { products: Product[] }) {
  const queryClient = useQueryClient();

  // 画面に近づいた商品のデータをプリフェッチ
  const prefetchProduct = useCallback((productId: string) => {
    queryClient.prefetchQuery({
      queryKey: productKeys.detail(productId),
      queryFn: () => api.products.get(productId),
      staleTime: 5 * 60 * 1000,
    });
  }, [queryClient]);

  return (
    <div className="grid grid-cols-4 gap-4">
      {products.map((product) => (
        <ProductCard
          key={product.id}
          product={product}
          onVisible={() => prefetchProduct(product.id)} // 画面に表示されたらプリフェッチ
        />
      ))}
    </div>
  );
}

// Intersection Observer で可視性を検知するProductCard
function ProductCard({
  product,
  onVisible,
}: {
  product: Product;
  onVisible: () => void;
}) {
  const { ref } = useInView({
    triggerOnce: true, // 1回だけ発火
    rootMargin: '100px',
    onChange: (inView) => {
      if (inView) onVisible();
    },
  });

  return (
    <div ref={ref}>
      <Link to={`/products/${product.id}`}>
        <img src={product.imageUrl} alt={product.name} />
        <h3>{product.name}</h3>
        <p>{product.price}円</p>
      </Link>
    </div>
  );
}
```

### 5.2 Next.js App Router でのSSR統合

```typescript
// Next.js Server Component でのプリフェッチ（App Router）
// app/users/page.tsx

import {
  HydrationBoundary,
  QueryClient,
  dehydrate,
} from '@tanstack/react-query';
import { UserList } from '@/components/UserList';

export default async function UsersPage() {
  // Server Component ではリクエストごとに新しい QueryClient を生成
  const queryClient = new QueryClient();

  // サーバー側でデータを取得
  await queryClient.prefetchQuery({
    queryKey: userKeys.lists(),
    queryFn: () => fetchUsersFromDB(), // サーバー側で直接DB/APIにアクセス
  });

  return (
    // dehydrate でサーバー側のキャッシュをクライアントに引き渡す
    <HydrationBoundary state={dehydrate(queryClient)}>
      <UserList />
    </HydrationBoundary>
  );
}
```

```typescript
// app/users/[userId]/page.tsx
// 動的ルートのプリフェッチ

import {
  HydrationBoundary,
  QueryClient,
  dehydrate,
} from '@tanstack/react-query';

interface PageProps {
  params: { userId: string };
}

export default async function UserDetailPage({ params }: PageProps) {
  const queryClient = new QueryClient();

  // 並列でデータを取得
  await Promise.all([
    queryClient.prefetchQuery({
      queryKey: userKeys.detail(params.userId),
      queryFn: () => fetchUserFromDB(params.userId),
    }),
    queryClient.prefetchQuery({
      queryKey: userKeys.posts(params.userId),
      queryFn: () => fetchUserPostsFromDB(params.userId),
    }),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <UserDetail userId={params.userId} />
    </HydrationBoundary>
  );
}

// クライアントコンポーネント
// components/UserDetail.tsx
'use client';

export function UserDetail({ userId }: { userId: string }) {
  // サーバーでプリフェッチされたデータがキャッシュから即座に返される
  // → 初期レンダリングでローディング表示がない
  const { data: user } = useQuery({
    queryKey: userKeys.detail(userId),
    queryFn: () => api.users.get(userId), // クライアント側のフォールバック
  });

  const { data: posts } = useQuery({
    queryKey: userKeys.posts(userId),
    queryFn: () => api.users.getPosts(userId),
  });

  if (!user) return null;

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      <h2>投稿一覧</h2>
      {posts?.map(post => (
        <PostCard key={post.id} post={post} />
      ))}
    </div>
  );
}
```

### 5.3 SSR時のQueryClient設定の注意点

```typescript
// 重要: SSR環境での QueryClient 生成パターン

// アンチパターン: モジュールレベルで1つだけ生成
// → リクエスト間でキャッシュが共有されてしまう！
// const queryClient = new QueryClient(); // 危険！

// 推奨パターン1: Server Component で毎回生成
export default async function Page() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        staleTime: 60 * 1000, // SSRデータは1分間fresh
      },
    },
  });
  // ...
}

// 推奨パターン2: Provider で useState を使って1回だけ生成
'use client';

export function QueryProvider({ children }: { children: React.ReactNode }) {
  // useState で初期化することで、コンポーネントのライフタイム中は同じインスタンスを使用
  // かつ、SSR時にリクエスト間で共有されない
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}
```

---

## 6. ポーリングとリアルタイム更新

### 6.1 ポーリング（定期的な再取得）

```typescript
// 基本的なポーリング
function useNotifications() {
  return useQuery({
    queryKey: ['notifications'],
    queryFn: fetchNotifications,
    refetchInterval: 30 * 1000,            // 30秒ごとに再取得
    refetchIntervalInBackground: false,     // バックグラウンドタブでは停止
  });
}
```

```typescript
// 条件付きポーリング: 処理状況に応じて間隔を変更

function useJobStatus(jobId: string) {
  return useQuery({
    queryKey: ['jobs', jobId],
    queryFn: () => api.jobs.getStatus(jobId),

    // ジョブが完了したらポーリングを停止
    refetchInterval: (query) => {
      const status = query.state.data?.status;

      if (status === 'completed' || status === 'failed') {
        return false; // ポーリング停止
      }
      if (status === 'processing') {
        return 2 * 1000; // 処理中は2秒間隔
      }
      return 10 * 1000; // それ以外は10秒間隔
    },

    // 初期データなしの場合のみfetch
    enabled: !!jobId,
  });
}

// 使用例: ファイルアップロード進捗の監視
function UploadProgress({ jobId }: { jobId: string }) {
  const { data: job } = useJobStatus(jobId);

  if (!job) return <Spinner />;

  return (
    <div>
      <ProgressBar value={job.progress} max={100} />
      <p>ステータス: {job.status}</p>
      {job.status === 'completed' && <p>完了しました！</p>}
      {job.status === 'failed' && <p>エラー: {job.error}</p>}
    </div>
  );
}
```

### 6.2 WebSocket との統合

```typescript
// WebSocket でリアルタイム更新を受信してキャッシュを更新

// src/hooks/useRealtimeUpdates.ts
import { useQueryClient } from '@tanstack/react-query';
import { useEffect } from 'react';

function useRealtimeUpdates() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const ws = new WebSocket(process.env.NEXT_PUBLIC_WS_URL!);

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      switch (message.type) {
        case 'USER_UPDATED':
          // 特定ユーザーのキャッシュを直接更新
          queryClient.setQueryData(
            userKeys.detail(message.payload.id),
            message.payload
          );
          // ユーザー一覧も無効化
          queryClient.invalidateQueries({ queryKey: userKeys.lists() });
          break;

        case 'USER_CREATED':
          // 一覧キャッシュを無効化して再取得をトリガー
          queryClient.invalidateQueries({ queryKey: userKeys.lists() });
          break;

        case 'USER_DELETED':
          // キャッシュから削除
          queryClient.removeQueries({
            queryKey: userKeys.detail(message.payload.id),
          });
          queryClient.invalidateQueries({ queryKey: userKeys.lists() });
          break;

        case 'NOTIFICATION':
          // 通知キャッシュを無効化
          queryClient.invalidateQueries({ queryKey: ['notifications'] });
          break;
      }
    };

    ws.onclose = () => {
      // 再接続ロジック
      console.log('WebSocket closed, attempting reconnect...');
    };

    return () => {
      ws.close();
    };
  }, [queryClient]);
}

// App.tsx で使用
function App() {
  useRealtimeUpdates();
  return <RouterProvider router={router} />;
}
```

```typescript
// Server-Sent Events (SSE) との統合

function useSSEUpdates() {
  const queryClient = useQueryClient();

  useEffect(() => {
    const eventSource = new EventSource('/api/events');

    eventSource.addEventListener('data-update', (event) => {
      const data = JSON.parse(event.data);

      // 関連するクエリを無効化
      queryClient.invalidateQueries({
        queryKey: [data.entity],
      });
    });

    eventSource.addEventListener('cache-invalidate', (event) => {
      const { queryKey } = JSON.parse(event.data);
      queryClient.invalidateQueries({ queryKey });
    });

    eventSource.onerror = () => {
      console.error('SSE connection error');
      eventSource.close();
      // 再接続ロジック
      setTimeout(() => {
        // 再接続
      }, 5000);
    };

    return () => {
      eventSource.close();
    };
  }, [queryClient]);
}
```

### 6.3 ポーリングとWebSocketの使い分け

```
リアルタイム更新の方式比較:

┌─────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 特性             │ ポーリング        │ SSE              │ WebSocket        │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 通信方向         │ クライアント→     │ サーバー→         │ 双方向           │
│                 │ サーバー          │ クライアント       │                  │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ リアルタイム性    │ 低い             │ 高い              │ 最も高い          │
│                 │ （間隔に依存）     │ （サーバープッシュ）│ （サーバープッシュ）│
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 実装の複雑さ     │ 簡単             │ 中程度            │ 複雑              │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ サーバー負荷     │ 高い             │ 中程度            │ 低い〜中程度      │
│                 │ （定期リクエスト） │ （接続維持）       │ （接続維持）       │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ スケーラビリティ  │ 良い             │ 良い              │ 要注意            │
│                 │ （ステートレス）   │ （HTTP準拠）       │ （ステートフル）   │
├─────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 適用シーン       │ ダッシュボード     │ 通知、ニュース     │ チャット、          │
│                 │ ジョブ監視         │ フィード           │ コラボレーション    │
└─────────────────┴──────────────────┴──────────────────┴──────────────────┘

推奨:
  → 更新頻度が低い（30秒以上）: ポーリング
  → サーバーからの一方向プッシュ: SSE
  → 双方向のリアルタイム通信: WebSocket
  → 簡単に始めたい: ポーリング → 必要に応じてSSE/WebSocketに移行
```

---

## 7. エラーハンドリングとリトライ

### 7.1 グローバルエラーハンドリング

```typescript
// src/lib/query-client.ts
// グローバルなエラーハンドリング設定

import { QueryClient, QueryCache, MutationCache } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  queryCache: new QueryCache({
    onError: (error, query) => {
      // 全てのクエリエラーをここでハンドリング

      // 認証エラー: ログインページにリダイレクト
      if (error instanceof ApiError && error.status === 401) {
        // トークン失効 → ログインページにリダイレクト
        window.location.href = '/login';
        return;
      }

      // 403 Forbidden: 権限不足の通知
      if (error instanceof ApiError && error.status === 403) {
        toast.error('この操作を行う権限がありません');
        return;
      }

      // 既にキャッシュにデータがある場合のみエラー通知
      // （初回ロードのエラーはコンポーネントレベルでハンドリング）
      if (query.state.data !== undefined) {
        toast.error(`データの更新に失敗しました: ${error.message}`);
      }
    },
  }),

  mutationCache: new MutationCache({
    onError: (error, variables, context, mutation) => {
      // 全てのMutationエラーをここでハンドリング

      // 認証エラー
      if (error instanceof ApiError && error.status === 401) {
        window.location.href = '/login';
        return;
      }

      // バリデーションエラーはコンポーネントレベルでハンドリング
      if (error instanceof ApiError && error.status === 422) {
        return; // グローバルではスキップ
      }

      // その他のエラー
      toast.error(`操作に失敗しました: ${error.message}`);
    },
  }),

  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        // 特定のHTTPステータスではリトライしない
        if (error instanceof ApiError) {
          if ([400, 401, 403, 404, 422].includes(error.status)) {
            return false; // リトライしない
          }
        }
        // それ以外は3回までリトライ
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) =>
        Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
});
```

### 7.2 コンポーネントレベルのエラーハンドリング

```typescript
// Error Boundary と Suspense を使ったエラーハンドリング

import { QueryErrorResetBoundary } from '@tanstack/react-query';
import { ErrorBoundary } from 'react-error-boundary';
import { Suspense } from 'react';

function UserSection() {
  return (
    <QueryErrorResetBoundary>
      {({ reset }) => (
        <ErrorBoundary
          onReset={reset}
          fallbackRender={({ error, resetErrorBoundary }) => (
            <div className="p-4 border border-red-300 rounded-lg bg-red-50">
              <h3 className="font-bold text-red-800">
                データの取得に失敗しました
              </h3>
              <p className="text-red-600 mt-1">{error.message}</p>
              <button
                onClick={resetErrorBoundary}
                className="mt-3 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
              >
                再試行
              </button>
            </div>
          )}
        >
          <Suspense fallback={<UserListSkeleton />}>
            <UserList />
          </Suspense>
        </ErrorBoundary>
      )}
    </QueryErrorResetBoundary>
  );
}

// useSuspenseQuery でSuspense対応
function UserList() {
  // useSuspenseQuery は data が必ず存在する（undefined にならない）
  const { data: users } = useSuspenseQuery({
    queryKey: userKeys.lists(),
    queryFn: () => api.users.list(),
  });

  return (
    <ul>
      {users.map(user => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

### 7.3 カスタムエラークラスとAPIクライアント

```typescript
// src/lib/api-error.ts
// カスタムエラークラス

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public details?: Record<string, string[]>,
  ) {
    super(message);
    this.name = 'ApiError';
  }

  // バリデーションエラーかどうか
  get isValidationError(): boolean {
    return this.status === 422;
  }

  // 認証エラーかどうか
  get isAuthError(): boolean {
    return this.status === 401;
  }

  // 権限エラーかどうか
  get isForbidden(): boolean {
    return this.status === 403;
  }

  // Not Found かどうか
  get isNotFound(): boolean {
    return this.status === 404;
  }

  // サーバーエラーかどうか
  get isServerError(): boolean {
    return this.status >= 500;
  }
}
```

```typescript
// src/lib/api-client.ts
// エラーハンドリング付きAPIクライアント

const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

async function apiClient<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;

  const config: RequestInit = {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders(),
      ...options.headers,
    },
  };

  const response = await fetch(url, config);

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));

    throw new ApiError(
      errorBody.message || `HTTP error ${response.status}`,
      response.status,
      errorBody.code,
      errorBody.details,
    );
  }

  // 204 No Content の場合
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

// API定義
export const api = {
  users: {
    list: (params?: UserFilters) =>
      apiClient<PaginatedResponse<User>>(`/users?${new URLSearchParams(params as any)}`),
    get: (id: string) =>
      apiClient<User>(`/users/${id}`),
    create: (data: CreateUserInput) =>
      apiClient<User>('/users', { method: 'POST', body: JSON.stringify(data) }),
    update: (id: string, data: UpdateUserInput) =>
      apiClient<User>(`/users/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
    delete: (id: string) =>
      apiClient<void>(`/users/${id}`, { method: 'DELETE' }),
  },
};
```

### 7.4 リトライ戦略の詳細

```typescript
// リトライ戦略のパターン

// パターン1: 指数バックオフ（推奨）
const retryWithExponentialBackoff = {
  retry: 3,
  retryDelay: (attemptIndex: number) => {
    // 1秒 → 2秒 → 4秒（最大30秒）
    return Math.min(1000 * 2 ** attemptIndex, 30000);
  },
};

// パターン2: ジッタ付き指数バックオフ（大量のクライアントがある場合）
const retryWithJitter = {
  retry: 3,
  retryDelay: (attemptIndex: number) => {
    const baseDelay = Math.min(1000 * 2 ** attemptIndex, 30000);
    // ±25%のジッタを追加（サーバーへの同時リクエストを分散）
    const jitter = baseDelay * 0.25 * (Math.random() * 2 - 1);
    return baseDelay + jitter;
  },
};

// パターン3: エラー種別に応じたリトライ判定
const smartRetry = {
  retry: (failureCount: number, error: unknown) => {
    if (error instanceof ApiError) {
      // クライアントエラーはリトライしない
      if (error.status >= 400 && error.status < 500) {
        return false;
      }
      // レート制限エラーはリトライする（間隔を長めに）
      if (error.status === 429) {
        return failureCount < 5;
      }
    }
    // ネットワークエラーやサーバーエラーは3回まで
    return failureCount < 3;
  },
  retryDelay: (attemptIndex: number, error: unknown) => {
    // レート制限の場合は長めの間隔
    if (error instanceof ApiError && error.status === 429) {
      const retryAfter = error.details?.retryAfter;
      if (retryAfter) {
        return parseInt(retryAfter) * 1000;
      }
      return 60 * 1000; // デフォルト1分待機
    }
    return Math.min(1000 * 2 ** attemptIndex, 30000);
  },
};
```

---

## 8. TanStack Query vs SWR 徹底比較

### 8.1 機能比較表

```
TanStack Query vs SWR 詳細比較:

┌──────────────────────────┬──────────────────┬──────────────────┐
│ 機能                      │ TanStack Query   │ SWR              │
├──────────────────────────┼──────────────────┼──────────────────┤
│ 開発元                    │ TanStack         │ Vercel           │
│ バンドルサイズ（gzip）     │ ~13KB            │ ~4KB             │
│ TypeScript                │ 完全対応          │ 完全対応          │
│ DevTools                  │ 優秀（専用GUI）   │ 限定的（SWR用）   │
│ フレームワーク対応         │ React, Vue,      │ React のみ        │
│                          │ Solid, Svelte,   │                  │
│                          │ Angular          │                  │
├──────────────────────────┼──────────────────┼──────────────────┤
│ [データ取得]              │                  │                  │
│ 基本クエリ                │ useQuery         │ useSWR           │
│ 並列クエリ                │ useQueries       │ 個別hook併用      │
│ 依存クエリ                │ enabled          │ 条件付きfetcher   │
│ Suspense対応             │ useSuspenseQuery │ suspense: true   │
│ プリフェッチ              │ prefetchQuery    │ preload          │
│ 初期データ                │ initialData      │ fallbackData     │
├──────────────────────────┼──────────────────┼──────────────────┤
│ [データ更新]              │                  │                  │
│ Mutation                 │ useMutation      │ useSWRMutation   │
│ 楽観的更新                │ 組み込み          │ optimisticData   │
│ キャッシュ無効化           │ invalidateQueries│ mutate           │
│ キャッシュ直接更新         │ setQueryData     │ mutate(data)     │
├──────────────────────────┼──────────────────┼──────────────────┤
│ [ページネーション]         │                  │                  │
│ 無限スクロール            │ useInfiniteQuery │ useSWRInfinite   │
│ ページネーション           │ keepPreviousData │ keepPreviousData │
├──────────────────────────┼──────────────────┼──────────────────┤
│ [キャッシュ制御]           │                  │                  │
│ staleTime                │ 柔軟に設定可能    │ dedupingInterval │
│                          │                  │ で代替            │
│ gcTime                   │ 設定可能          │ 限定的            │
│ 構造共有                  │ あり             │ なし              │
│ オフライン対応             │ 3モード          │ 限定的            │
│ キャッシュ永続化           │ persistQueryClient│ カスタム実装      │
├──────────────────────────┼──────────────────┼──────────────────┤
│ [その他]                  │                  │                  │
│ リトライ                  │ 詳細設定可能      │ 基本的な設定      │
│ ポーリング                │ refetchInterval  │ refreshInterval  │
│ ウィンドウフォーカス        │ あり             │ あり              │
│ ネットワーク再接続         │ あり             │ あり              │
│ SSR統合                  │ HydrationBoundary│ SWRConfig        │
│ ミドルウェア              │ なし             │ あり              │
│ 学習コスト                │ やや高い          │ 低い              │
└──────────────────────────┴──────────────────┴──────────────────┘
```

### 8.2 SWR での実装例

```typescript
// SWR を使った基本的なデータ取得

import useSWR from 'swr';

// フェッチャー関数
const fetcher = (url: string) =>
  fetch(url).then(res => {
    if (!res.ok) throw new Error('Failed to fetch');
    return res.json();
  });

// 基本的な使用法
function UserList() {
  const { data, error, isLoading, isValidating, mutate } = useSWR(
    '/api/users',
    fetcher,
    {
      revalidateOnFocus: true,      // ウィンドウフォーカスで再取得
      revalidateOnReconnect: true,  // ネットワーク再接続で再取得
      refreshInterval: 0,           // ポーリング無効（0 = 無効）
      dedupingInterval: 2000,       // 2秒間の重複リクエストを排除
      errorRetryCount: 3,           // 3回リトライ
    }
  );

  if (isLoading) return <Loading />;
  if (error) return <Error message={error.message} />;

  return (
    <div>
      {isValidating && <RefetchIndicator />}
      <ul>
        {data.map((user: User) => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

```typescript
// SWR: 条件付きフェッチ（依存クエリ）
function useUserDetail(userId: string | null) {
  return useSWR(
    userId ? `/api/users/${userId}` : null, // nullでフェッチしない
    fetcher,
  );
}

// SWR: Mutationと楽観的更新
function TodoList() {
  const { data: todos, mutate } = useSWR('/api/todos', fetcher);

  const toggleTodo = async (id: string, completed: boolean) => {
    // 楽観的更新
    const optimisticData = todos.map((todo: Todo) =>
      todo.id === id ? { ...todo, completed } : todo
    );

    await mutate(
      async () => {
        await api.todos.update(id, { completed });
        return await fetcher('/api/todos');
      },
      {
        optimisticData,
        rollbackOnError: true,
        populateCache: true,
        revalidate: false,
      }
    );
  };

  return (
    <ul>
      {todos?.map((todo: Todo) => (
        <li key={todo.id}>
          <input
            type="checkbox"
            checked={todo.completed}
            onChange={(e) => toggleTodo(todo.id, e.target.checked)}
          />
          {todo.title}
        </li>
      ))}
    </ul>
  );
}
```

```typescript
// SWR: 無限スクロール
import useSWRInfinite from 'swr/infinite';

function useInfiniteUsers() {
  const getKey = (pageIndex: number, previousPageData: any) => {
    if (previousPageData && !previousPageData.data.length) return null; // 最後のページ
    return `/api/users?page=${pageIndex + 1}&limit=20`;
  };

  const { data, error, size, setSize, isLoading, isValidating } =
    useSWRInfinite(getKey, fetcher);

  const users = data ? data.flatMap(page => page.data) : [];
  const isLoadingMore = isLoading || (size > 0 && data && typeof data[size - 1] === 'undefined');
  const isEmpty = data?.[0]?.data?.length === 0;
  const isReachingEnd = isEmpty || (data && data[data.length - 1]?.data?.length < 20);

  return {
    users,
    isLoading,
    isLoadingMore,
    isReachingEnd,
    loadMore: () => setSize(size + 1),
    isValidating,
    error,
  };
}
```

### 8.3 ライブラリ選定ガイドライン

```
ライブラリ選定の判断基準:

  TanStack Query を選ぶべき場合:
  ┌─────────────────────────────────────────────────────────┐
  │ ・大規模なCRUDアプリケーション                              │
  │ ・複雑なキャッシュ管理が必要（階層的な無効化が頻繁）          │
  │ ・楽観的更新を多用する                                     │
  │ ・DevToolsによるデバッグが重要                              │
  │ ・オフライン対応が必要                                     │
  │ ・React以外のフレームワークでも使いたい                      │
  │ ・チームにサーバー状態管理の経験者がいる                      │
  └─────────────────────────────────────────────────────────┘

  SWR を選ぶべき場合:
  ┌─────────────────────────────────────────────────────────┐
  │ ・シンプルなデータ取得が主                                  │
  │ ・バンドルサイズを最小限にしたい                             │
  │ ・Next.js プロジェクト（Vercel製で親和性が高い）              │
  │ ・学習コストを抑えたい                                     │
  │ ・ミドルウェアパターンを使いたい                             │
  │ ・すぐに使い始めたい（API がシンプル）                       │
  └─────────────────────────────────────────────────────────┘

  結論:
  → 迷ったら TanStack Query を選択（機能が豊富で後から困りにくい）
  → 小規模プロジェクトやプロトタイプなら SWR で十分
  → どちらを選んでも stale-while-revalidate の恩恵は受けられる
```

---

## 9. アンチパターンと落とし穴

### 9.1 よくあるアンチパターン集

```typescript
// アンチパターン1: useEffect 内で queryClient を操作する
// Bad
function UserDetail({ userId }: { userId: string }) {
  const queryClient = useQueryClient();
  const { data } = useQuery({
    queryKey: ['users', userId],
    queryFn: () => api.users.get(userId),
  });

  // useEffect でキャッシュを操作すると無限ループの危険
  useEffect(() => {
    if (data) {
      queryClient.setQueryData(['currentUser'], data);
    }
  }, [data, queryClient]);
}

// Good: select を使う
function UserDetail({ userId }: { userId: string }) {
  const { data } = useQuery({
    queryKey: ['users', userId],
    queryFn: () => api.users.get(userId),
  });
  // 必要なら別のクエリで同じデータを参照する
}
```

```typescript
// アンチパターン2: queryFn 内でステートを参照する
// Bad
function SearchResults() {
  const [filter, setFilter] = useState('');

  const { data } = useQuery({
    queryKey: ['search'],        // queryKey にフィルターが含まれていない
    queryFn: () => api.search(filter), // filter が変わってもクエリが再実行されない
  });
}

// Good: queryKey にパラメータを含める
function SearchResults() {
  const [filter, setFilter] = useState('');

  const { data } = useQuery({
    queryKey: ['search', filter],      // filter を queryKey に含める
    queryFn: () => api.search(filter), // filter が変わるとクエリが再実行される
    enabled: filter.length >= 2,
  });
}
```

```typescript
// アンチパターン3: コンポーネント内で QueryClient を生成
// Bad
function App() {
  // 毎回レンダリングで新しい QueryClient が生成される
  const queryClient = new QueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      <MyApp />
    </QueryClientProvider>
  );
}

// Good: useState で1回だけ生成する
function App() {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <QueryClientProvider client={queryClient}>
      <MyApp />
    </QueryClientProvider>
  );
}
```

```typescript
// アンチパターン4: mutate と mutateAsync の使い分けミス
// Bad: mutate の返り値で何かしようとしている
function CreateButton() {
  const createUser = useCreateUser();

  const handleClick = () => {
    // mutate は void を返す → thenもcatchも動かない
    const result = createUser.mutate(userData);
    console.log(result); // undefined
  };
}

// Good: 返り値が必要なら mutateAsync を使う
function CreateButton() {
  const createUser = useCreateUser();

  const handleClick = async () => {
    try {
      const newUser = await createUser.mutateAsync(userData);
      router.push(`/users/${newUser.id}`);
    } catch (error) {
      // エラーハンドリング
    }
  };
}

// Good: コールバックで処理するなら mutate でOK
function CreateButton() {
  const createUser = useCreateUser();

  const handleClick = () => {
    createUser.mutate(userData, {
      onSuccess: (newUser) => {
        router.push(`/users/${newUser.id}`);
      },
      onError: (error) => {
        toast.error(error.message);
      },
    });
  };
}
```

```typescript
// アンチパターン5: staleTime と gcTime の設定ミス
// Bad: gcTime < staleTime はナンセンス
const { data } = useQuery({
  queryKey: ['users'],
  queryFn: fetchUsers,
  staleTime: 10 * 60 * 1000,    // 10分間fresh
  gcTime: 1 * 60 * 1000,        // 1分でGC → freshのまま消える可能性
});

// Good: gcTime >= staleTime にする
const { data } = useQuery({
  queryKey: ['users'],
  queryFn: fetchUsers,
  staleTime: 10 * 60 * 1000,    // 10分間fresh
  gcTime: 15 * 60 * 1000,       // 15分でGC（staleTimeより長い）
});
```

```typescript
// アンチパターン6: 楽観的更新でonSettledのinvalidateを忘れる
// Bad
const mutation = useMutation({
  mutationFn: updateTodo,
  onMutate: async (newData) => {
    await queryClient.cancelQueries({ queryKey: ['todos'] });
    const previous = queryClient.getQueryData(['todos']);
    queryClient.setQueryData(['todos'], newData);
    return { previous };
  },
  onError: (err, newData, context) => {
    queryClient.setQueryData(['todos'], context?.previous);
  },
  // onSettled がない → サーバーの最新データと同期されない
});

// Good: onSettled で必ず再検証
const mutation = useMutation({
  mutationFn: updateTodo,
  onMutate: async (newData) => {
    await queryClient.cancelQueries({ queryKey: ['todos'] });
    const previous = queryClient.getQueryData(['todos']);
    queryClient.setQueryData(['todos'], newData);
    return { previous };
  },
  onError: (err, newData, context) => {
    queryClient.setQueryData(['todos'], context?.previous);
  },
  onSettled: () => {
    queryClient.invalidateQueries({ queryKey: ['todos'] }); // 必ず再検証
  },
});
```

### 9.2 パフォーマンスの落とし穴

```typescript
// 落とし穴1: select で新しいオブジェクト参照を毎回作成
// Bad: インラインselectが毎回新しい関数参照
function ActiveUserCount() {
  const { data: count } = useQuery({
    queryKey: ['users'],
    queryFn: () => api.users.list(),
    // 毎回新しい関数参照 → structuralSharingが効かない場合がある
    select: (data) => data.filter(u => u.active).length,
  });
  return <span>{count}</span>;
}

// Good: useCallbackで安定した参照を使う
function ActiveUserCount() {
  const selectCount = useCallback(
    (data: User[]) => data.filter(u => u.active).length,
    []
  );

  const { data: count } = useQuery({
    queryKey: ['users'],
    queryFn: () => api.users.list(),
    select: selectCount,
  });
  return <span>{count}</span>;
}
```

```typescript
// 落とし穴2: 無限スクロールのメモリリーク
// Bad: ページ数が際限なく増える
function InfiniteList() {
  const { data } = useInfiniteQuery({
    queryKey: ['items'],
    queryFn: fetchItems,
    initialPageParam: 0,
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    // maxPages がない → 長時間使用するとメモリが増大
  });
}

// Good: maxPages を設定してメモリ使用量を制限
function InfiniteList() {
  const { data } = useInfiniteQuery({
    queryKey: ['items'],
    queryFn: fetchItems,
    initialPageParam: 0,
    getNextPageParam: (lastPage) => lastPage.nextCursor,
    maxPages: 10, // 最大10ページまでキャッシュ
  });
}
```

---

## 10. テスト戦略

### 10.1 カスタムフックのテスト

```typescript
// src/hooks/__tests__/useUsers.test.tsx
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';

// MSW サーバー設定
const server = setupServer(
  http.get('/api/users', () => {
    return HttpResponse.json([
      { id: '1', name: 'Alice', email: 'alice@example.com' },
      { id: '2', name: 'Bob', email: 'bob@example.com' },
    ]);
  }),
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// テスト用のラッパー
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,     // テストではリトライしない
        gcTime: Infinity, // テスト中にキャッシュが消えないように
      },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

// テスト
describe('useUsers', () => {
  it('ユーザー一覧を取得できる', async () => {
    const { result } = renderHook(() => useUsers(), {
      wrapper: createWrapper(),
    });

    // 初期状態はローディング
    expect(result.current.isLoading).toBe(true);

    // データが取得されるまで待機
    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // データの検証
    expect(result.current.data).toHaveLength(2);
    expect(result.current.data?.[0].name).toBe('Alice');
  });

  it('エラー時にエラー状態になる', async () => {
    // このテストだけエラーを返すように上書き
    server.use(
      http.get('/api/users', () => {
        return HttpResponse.json(
          { message: 'Internal Server Error' },
          { status: 500 },
        );
      }),
    );

    const { result } = renderHook(() => useUsers(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error).toBeDefined();
  });
});
```

### 10.2 Mutation のテスト

```typescript
describe('useCreateUser', () => {
  it('ユーザーを作成してキャッシュを更新する', async () => {
    const newUser = { name: 'Charlie', email: 'charlie@example.com' };

    server.use(
      http.post('/api/users', async ({ request }) => {
        const body = await request.json();
        return HttpResponse.json({ id: '3', ...body as object });
      }),
    );

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    // 先にユーザー一覧をキャッシュに設定
    queryClient.setQueryData(['users', 'list'], [
      { id: '1', name: 'Alice' },
    ]);

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );

    const { result } = renderHook(() => useCreateUser(), { wrapper });

    // Mutation を実行
    result.current.mutate(newUser);

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    // Mutation の結果
    expect(result.current.data).toEqual({ id: '3', ...newUser });

    // キャッシュが無効化されたか確認
    const queryState = queryClient.getQueryState(['users', 'list']);
    expect(queryState?.isInvalidated).toBe(true);
  });
});
```

### 10.3 楽観的更新のテスト

```typescript
describe('useUpdateTodo (楽観的更新)', () => {
  it('即座にUIが更新され、エラー時にロールバックする', async () => {
    // APIがエラーを返すように設定
    server.use(
      http.patch('/api/todos/:id', () => {
        return HttpResponse.json(
          { message: 'Server Error' },
          { status: 500 },
        );
      }),
    );

    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });

    const initialTodos = [
      { id: '1', title: 'Buy milk', completed: false },
      { id: '2', title: 'Walk dog', completed: false },
    ];
    queryClient.setQueryData(['todos'], initialTodos);

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );

    const { result } = renderHook(() => useUpdateTodo(), { wrapper });

    // 楽観的更新を実行
    result.current.mutate({ id: '1', data: { completed: true } });

    // 即座にキャッシュが更新されていることを確認（楽観的更新）
    await waitFor(() => {
      const todos = queryClient.getQueryData<Todo[]>(['todos']);
      expect(todos?.[0].completed).toBe(true);
    });

    // エラー後にロールバックされることを確認
    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    // キャッシュが元に戻っていることを確認
    const todos = queryClient.getQueryData<Todo[]>(['todos']);
    expect(todos?.[0].completed).toBe(false);
  });
});
```

---

## 11. 高度なパターン

### 11.1 useQueries による並列クエリ

```typescript
// 複数のクエリを並列実行

import { useQueries } from '@tanstack/react-query';

function DashboardStats({ userIds }: { userIds: string[] }) {
  // 動的な数のクエリを並列実行
  const userQueries = useQueries({
    queries: userIds.map(id => ({
      queryKey: userKeys.detail(id),
      queryFn: () => api.users.get(id),
      staleTime: 5 * 60 * 1000,
    })),
  });

  const isLoading = userQueries.some(q => q.isLoading);
  const isError = userQueries.some(q => q.isError);
  const users = userQueries
    .filter(q => q.isSuccess)
    .map(q => q.data!);

  if (isLoading) return <Spinner />;
  if (isError) return <ErrorMessage />;

  return (
    <div className="grid grid-cols-3 gap-4">
      {users.map(user => (
        <UserStatCard key={user.id} user={user} />
      ))}
    </div>
  );
}
```

### 11.2 キャッシュの永続化

```typescript
// TanStack Query のキャッシュを永続化する
// アプリ再起動時にキャッシュを復元できる

import { persistQueryClient } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      gcTime: 1000 * 60 * 60 * 24, // 24時間（永続化のために長めに設定）
    },
  },
});

// localStorage を使った永続化
const persister = createSyncStoragePersister({
  storage: window.localStorage,
  key: 'REACT_QUERY_OFFLINE_CACHE',
  throttleTime: 1000, // 1秒に1回しか保存しない（パフォーマンス対策）
  serialize: (data) => JSON.stringify(data),
  deserialize: (data) => JSON.parse(data),
});

// 永続化の設定
persistQueryClient({
  queryClient,
  persister,
  maxAge: 1000 * 60 * 60 * 24, // 24時間で期限切れ
  dehydrateOptions: {
    shouldDehydrateQuery: (query) => {
      // 特定のクエリのみ永続化
      const queryKey = query.queryKey as string[];
      return ['categories', 'config', 'user-preferences'].some(key =>
        queryKey.includes(key)
      );
    },
  },
});
```

### 11.3 オフライン対応

```typescript
// オフラインファーストのアプリケーション

import { onlineManager } from '@tanstack/react-query';

// カスタムオンライン検知（navigator.onLine の代替）
onlineManager.setEventListener((setOnline) => {
  const onlineHandler = () => setOnline(true);
  const offlineHandler = () => setOnline(false);

  window.addEventListener('online', onlineHandler);
  window.addEventListener('offline', offlineHandler);

  return () => {
    window.removeEventListener('online', onlineHandler);
    window.removeEventListener('offline', offlineHandler);
  };
});

// オフライン時のMutation キュー
// networkMode: 'offlineFirst' を使用
const mutation = useMutation({
  mutationFn: api.todos.create,
  networkMode: 'offlineFirst', // オフラインでもキューに入れて後で実行
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['todos'] });
  },
});

// オフライン状態のUI表示
function OfflineIndicator() {
  const isOnline = onlineManager.isOnline();

  if (isOnline) return null;

  return (
    <div className="fixed bottom-4 right-4 bg-yellow-500 text-white px-4 py-2 rounded-lg shadow-lg">
      オフラインです。変更はオンライン復帰後に同期されます。
    </div>
  );
}
```

### 11.4 カスタムフックの設計原則

```typescript
// カスタムフックの設計原則

// 原則1: クエリオプションを外部から注入可能にする
function useUsers(options?: Partial<UseQueryOptions<User[]>>) {
  return useQuery({
    queryKey: userKeys.lists(),
    queryFn: () => api.users.list(),
    staleTime: 30 * 1000,
    ...options, // 呼び出し側でオーバーライド可能
  });
}

// 使用例
const { data } = useUsers({
  staleTime: Infinity,      // この画面では再取得不要
  enabled: isAuthenticated, // 認証済みの場合のみ
});

// 原則2: 関連するQuery + Mutation をまとめる
function useTodos() {
  const queryClient = useQueryClient();

  const todosQuery = useQuery({
    queryKey: ['todos'],
    queryFn: () => api.todos.list(),
  });

  const addMutation = useMutation({
    mutationFn: api.todos.create,
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['todos'] }),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }: { id: string; data: Partial<Todo> }) =>
      api.todos.update(id, data),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['todos'] }),
  });

  const deleteMutation = useMutation({
    mutationFn: api.todos.delete,
    onSettled: () => queryClient.invalidateQueries({ queryKey: ['todos'] }),
  });

  return {
    todos: todosQuery.data ?? [],
    isLoading: todosQuery.isLoading,
    error: todosQuery.error,
    addTodo: addMutation.mutate,
    updateTodo: updateMutation.mutate,
    deleteTodo: deleteMutation.mutate,
    isAdding: addMutation.isPending,
    isUpdating: updateMutation.isPending,
    isDeleting: deleteMutation.isPending,
  };
}

// 原則3: queryOptions ヘルパー関数で設定を共有する
import { queryOptions } from '@tanstack/react-query';

export function userListQueryOptions(filters?: UserFilters) {
  return queryOptions({
    queryKey: userKeys.list(filters ?? {}),
    queryFn: () => api.users.list(filters),
    staleTime: 30 * 1000,
  });
}

// コンポーネントで使用
function UserList() {
  const { data } = useQuery(userListQueryOptions({ role: 'admin' }));
}

// ローダーで使用
export async function loader({ params }: LoaderFunctionArgs) {
  return queryClient.ensureQueryData(userListQueryOptions());
}

// プリフェッチで使用
queryClient.prefetchQuery(userListQueryOptions());
```

---

## まとめ

### パターン早見表

| パターン | 用途 | API |
|---------|------|-----|
| staleTime | データの鮮度期限の設定 | `useQuery({ staleTime })` |
| gcTime | 非活性キャッシュの保持期間 | `useQuery({ gcTime })` |
| invalidateQueries | Mutation後のキャッシュ無効化 | `queryClient.invalidateQueries()` |
| setQueryData | キャッシュの直接更新 | `queryClient.setQueryData()` |
| useInfiniteQuery | 無限スクロール | `useInfiniteQuery({ getNextPageParam })` |
| prefetchQuery | ホバー/ルート遷移前のプリフェッチ | `queryClient.prefetchQuery()` |
| placeholderData | ページ遷移時の前データ表示 | `useQuery({ placeholderData })` |
| enabled | 条件付きクエリ実行 | `useQuery({ enabled })` |
| select | キャッシュデータの変換・フィルタ | `useQuery({ select })` |
| useSuspenseQuery | Suspense対応クエリ | `useSuspenseQuery()` |
| refetchInterval | ポーリング（定期再取得） | `useQuery({ refetchInterval })` |
| networkMode | オフライン対応 | `useQuery({ networkMode })` |

### キャッシュ戦略チートシート

```
データの種類別キャッシュ戦略:

┌──────────────────┬──────────┬──────────┬──────────────────────┐
│ データ種別        │ staleTime│ gcTime   │ その他                │
├──────────────────┼──────────┼──────────┼──────────────────────┤
│ リアルタイム      │ 0        │ 5分      │ refetchInterval      │
│ （通知、チャット）  │          │          │ WebSocket連携         │
├──────────────────┼──────────┼──────────┼──────────────────────┤
│ ユーザーデータ    │ 30秒〜2分│ 10分     │ invalidateQueries    │
│ （一覧、詳細）     │          │          │ 楽観的更新            │
├──────────────────┼──────────┼──────────┼──────────────────────┤
│ マスタデータ      │ 1〜24時間│ Infinity │ 初回のみフェッチ       │
│ （カテゴリ、設定） │          │          │ staleTime: Infinity  │
├──────────────────┼──────────┼──────────┼──────────────────────┤
│ 検索結果         │ 5分      │ 10分     │ keepPreviousData     │
│                  │          │          │ デバウンス            │
├──────────────────┼──────────┼──────────┼──────────────────────┤
│ ページネーション  │ 30秒     │ 5分      │ keepPreviousData     │
│                  │          │          │ 次ページプリフェッチ    │
└──────────────────┴──────────┴──────────┴──────────────────────┘
```

### 実装チェックリスト

```
サーバー状態管理の実装チェックリスト:

  セットアップ:
  □ QueryClient のデフォルト設定を定義した
  □ QueryClientProvider を配置した
  □ DevTools を開発環境で有効にした
  □ グローバルエラーハンドリング（QueryCache, MutationCache）を設定した

  Query Key:
  □ Query Key Factory パターンを導入した
  □ queryFn で使用する全パラメータが queryKey に含まれている
  □ 階層構造で設計し、部分的な無効化が可能になっている

  キャッシュ戦略:
  □ データ種別ごとに staleTime を設定した
  □ gcTime が staleTime 以上であることを確認した
  □ 適切な再取得トリガー（windowFocus, reconnect）を設定した

  Mutation:
  □ onSettled で関連キャッシュの invalidateQueries を呼んでいる
  □ 楽観的更新が必要な箇所は onMutate + onError + onSettled を実装した
  □ mutate と mutateAsync を適切に使い分けている

  エラーハンドリング:
  □ 認証エラー（401）のグローバルハンドリングを設定した
  □ リトライ戦略を設定した（クライアントエラーはリトライしない）
  □ Error Boundary で UI レベルのエラー表示を実装した

  パフォーマンス:
  □ 重要なページのプリフェッチを設定した
  □ 無限スクロールで maxPages を設定した
  □ select の参照安定性を確認した
  □ 大量データ表示に仮想化を検討した

  テスト:
  □ MSW でAPIモックを設定した
  □ カスタムフックのテストを書いた
  □ 楽観的更新のロールバックをテストした
```

---

## 次に読むべきガイド
→ [[03-url-state.md]] -- URL状態管理

---

## 参考文献
1. TkDodo. "Practical React Query." tkdodo.eu, 2024.
2. TanStack. "TanStack Query Documentation v5." tanstack.com, 2024.
3. SWR. "React Hooks for Data Fetching." swr.vercel.app, 2024.
4. Kent C. Dodds. "Application State Management with React." kentcdodds.com, 2020.
5. Jason Watmore. "React Query vs SWR - Feature Comparison." jasonwatmore.com, 2024.
6. Web.dev. "stale-while-revalidate." web.dev, 2023.
7. TkDodo. "React Query and React Router." tkdodo.eu, 2023.
8. TanStack. "React Query DevTools." tanstack.com, 2024.
9. Vercel. "SWR v2 Documentation." swr.vercel.app, 2024.
10. MDN Web Docs. "Cache-Control: stale-while-revalidate." developer.mozilla.org, 2024.
