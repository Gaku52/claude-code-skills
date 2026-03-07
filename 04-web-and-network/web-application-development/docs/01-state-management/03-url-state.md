# URL状態（URL State Management）

> URL状態はWebアプリの「共有可能な状態」。検索クエリ、フィルタ、ページ番号、ソート順をURLに反映することで、ブックマーク・共有・ブラウザバックが自然に動作するUXを実現する。URL は単なるリソースの識別子ではなく、アプリケーションの状態を永続化するための強力なメカニズムである。

## この章で学ぶこと

- [ ] URL状態の重要性と設計原則を理解する
- [ ] URLSearchParams API の基本操作を習得する
- [ ] useSearchParams（React Router / Next.js）の使い方を把握する
- [ ] nuqs（型安全なURL状態管理）の活用方法を学ぶ
- [ ] Server Component との統合パターンを理解する
- [ ] URL状態のデバウンス・バリデーション・セキュリティを理解する
- [ ] URLとアプリ状態の同期パターンを学ぶ
- [ ] 実践的なURL状態管理のアーキテクチャを設計できるようになる

---

## 1. URL状態の重要性と基礎概念

### 1.1 なぜURL状態が重要なのか

Webアプリケーションにおいて、URLは単なるページの住所ではない。URLはアプリケーションの状態を表現する最も基本的で強力なメカニズムである。REST（Representational State Transfer）の原則においても、URLはリソースの状態を表す識別子として位置づけられている。

URL状態管理が適切に実装されたアプリケーションでは、以下のユーザー体験が自然に実現される:

```
URL状態がもたらす優れたUX:

1. ブックマーク可能性（Bookmarkability）
   - ユーザーが特定の検索結果やフィルタ状態をブックマークできる
   - 後日アクセスしても同じ状態が復元される
   - 例: /products?q=laptop&sort=price-asc&page=2

2. 共有可能性（Shareability）
   - URLをコピーして他のユーザーに送れば同じ画面が表示される
   - SlackやTeamsでリンクを共有する際に完全な状態が伝わる
   - 例: 「この検索結果見て」→ URLを送るだけでOK

3. ブラウザナビゲーション（Browser Navigation）
   - ブラウザの戻る/進むボタンが期待通りに動作する
   - 各状態変更がブラウザ履歴に記録される
   - ユーザーの操作フローが自然になる

4. SEOフレンドリー（SEO Friendly）
   - 検索エンジンのクローラーがパラメータ付きURLを解析できる
   - カテゴリページやフィルタページがインデックスされる
   - canonical URL の適切な設定が可能

5. サーバーサイドレンダリング対応（SSR Compatible）
   - Server Component で searchParams を受け取れる
   - 初期描画時にサーバーで適切なデータを取得できる
   - SEOとパフォーマンスの両立が可能

6. デバッグ容易性（Debuggability）
   - URLを見れば現在の状態がわかる
   - バグ報告時に「このURLで再現します」と伝えられる
   - QAテストで状態の再現が容易
```

### 1.2 URL状態に反映すべきもの・すべきでないもの

URLに反映すべき状態とそうでない状態を明確に区別することが、URL状態設計の第一歩である。

```
URL に状態を反映すべき場面:
  ✓ 検索クエリ: /products?q=laptop
  ✓ フィルタ: /products?category=electronics&brand=apple
  ✓ ソート: /products?sort=price&order=asc
  ✓ ページネーション: /products?page=3&per_page=20
  ✓ タブ/ビュー: /dashboard?view=chart
  ✓ モーダルの状態: /products?modal=create
  ✓ 日付範囲: /analytics?from=2024-01-01&to=2024-03-31
  ✓ 表示密度: /products?density=compact
  ✓ 言語/ロケール: /products?lang=ja（pathname推奨: /ja/products）
  ✓ 比較対象: /compare?ids=1,2,3
  ✓ アコーディオン/セクション展開: /faq?section=billing
  ✓ 地図の表示範囲: /map?lat=35.68&lng=139.76&zoom=12

URL に反映すべきでない状態:
  ✗ 一時的なUI状態（ツールチップ、ホバー、アニメーション）
  ✗ フォームの入力途中値（送信前のドラフト）
  ✗ 認証トークンやセッション情報
  ✗ 大量のデータ（URLは2048文字が実質上限）
  ✗ 機密情報（パスワード、個人情報）
  ✗ ドラッグ中の位置情報
  ✗ ローディング状態やエラー状態
  ✗ アプリ内通知の未読数
  ✗ ユーザー固有の設定（ダークモードなど → localStorageが適切）
```

### 1.3 URL状態の判断基準フローチャート

状態をURLに含めるべきかどうかを判断するためのフローチャートを示す。

```
その状態は他のユーザーと共有できるべきか？
├── Yes → ブックマークして後で同じ状態に戻りたいか？
│   ├── Yes → URL状態に含める ✓
│   └── No  → ブラウザバックで戻りたいか？
│       ├── Yes → URL状態に含める ✓
│       └── No  → React state / Context で管理
└── No  → その状態はセキュリティ上問題ないか？
    ├── Yes → 永続化が必要か？
    │   ├── Yes → localStorage / Cookie で管理
    │   └── No  → React state / Context で管理
    └── No  → Cookie（HttpOnly）/ サーバーセッションで管理
```

### 1.4 URLの構造と各部分の役割

URL状態管理を正しく行うためには、URLの構造を理解することが前提条件である。

```
完全なURL構造:
https://example.com:443/products/electronics?q=laptop&page=2#reviews
└─┬─┘   └───┬──────┘└┬┘└───────┬──────────┘└──────┬───────┘└──┬───┘
scheme     host    port    pathname           search       hash
                                           (query string) (fragment)

各部分の状態管理での役割:

pathname（パス）:
  - リソースの種類・階層を表す
  - 例: /products, /products/123, /users/profile
  - ルーティングで使用（React Router, Next.js App Router）
  - RESTful設計で重要

search / query string（クエリパラメータ）:
  - リソースの表示方法・フィルタ条件を表す
  - 例: ?q=laptop&sort=price&page=2
  - URL状態管理のメインターゲット
  - key=value のペアで表現

hash / fragment（フラグメント）:
  - ページ内の位置を表す
  - 例: #reviews, #section-3
  - サーバーに送信されない（クライアントのみ）
  - SPA時代にはルーティングに使われた（Hash Router）
```

### 1.5 URLSearchParams API の基本

URL状態を操作する最も基本的なWeb標準APIが `URLSearchParams` である。ブラウザとNode.js の両方で利用可能である。

```typescript
// === URLSearchParams の基本操作 ===

// 1. 生成方法
const params1 = new URLSearchParams('q=laptop&page=2');
const params2 = new URLSearchParams({ q: 'laptop', page: '2' });
const params3 = new URLSearchParams([['q', 'laptop'], ['page', '2']]);
const params4 = new URLSearchParams(window.location.search);

// 2. 値の取得
params1.get('q');        // 'laptop'
params1.get('page');     // '2'（常に string）
params1.get('missing');  // null

// 3. 値の設定
params1.set('sort', 'price');     // 追加（既存キーがあれば上書き）
params1.append('tag', 'sale');    // 追加（同じキーが複数存在可能）
params1.append('tag', 'new');

// 4. 値の削除
params1.delete('page');           // キーと値を削除

// 5. 存在確認
params1.has('q');                 // true
params1.has('page');              // false（削除済み）

// 6. イテレーション
for (const [key, value] of params1) {
  console.log(`${key}: ${value}`);
}
// q: laptop
// sort: price
// tag: sale
// tag: new

// 7. 配列値の取得
params1.getAll('tag');            // ['sale', 'new']

// 8. 文字列化
params1.toString();               // 'q=laptop&sort=price&tag=sale&tag=new'

// 9. ソート（キー順）
params1.sort();
params1.toString();               // 'q=laptop&sort=price&tag=new&tag=sale'

// 10. サイズ取得
params1.size;                     // 4（エントリ数）
```

```typescript
// === URLSearchParams の実用的なヘルパー関数 ===

/**
 * 現在のURLのクエリパラメータを更新するヘルパー
 */
function updateSearchParams(
  updates: Record<string, string | string[] | null | undefined>
): string {
  const params = new URLSearchParams(window.location.search);

  for (const [key, value] of Object.entries(updates)) {
    if (value === null || value === undefined) {
      params.delete(key);
    } else if (Array.isArray(value)) {
      params.delete(key);
      value.forEach(v => params.append(key, v));
    } else {
      params.set(key, value);
    }
  }

  // デフォルト値のキーを除去してURLをクリーンに保つ
  return params.toString();
}

// 使用例
const newSearch = updateSearchParams({
  q: 'laptop',
  page: null,          // 削除
  tags: ['sale', 'new'] // 配列
});
// → 'q=laptop&tags=sale&tags=new'

/**
 * URLSearchParamsからオブジェクトに変換
 */
function searchParamsToObject(
  params: URLSearchParams
): Record<string, string | string[]> {
  const result: Record<string, string | string[]> = {};

  for (const key of new Set(params.keys())) {
    const values = params.getAll(key);
    result[key] = values.length === 1 ? values[0] : values;
  }

  return result;
}

// 使用例
const params = new URLSearchParams('q=laptop&tag=sale&tag=new&page=2');
const obj = searchParamsToObject(params);
// { q: 'laptop', tag: ['sale', 'new'], page: '2' }

/**
 * オブジェクトの差分をURLSearchParamsに適用
 */
function mergeSearchParams(
  base: URLSearchParams,
  updates: Record<string, string | null>
): URLSearchParams {
  const merged = new URLSearchParams(base.toString());

  for (const [key, value] of Object.entries(updates)) {
    if (value === null) {
      merged.delete(key);
    } else {
      merged.set(key, value);
    }
  }

  return merged;
}
```

```typescript
// === エンコーディングの注意点 ===

// URLSearchParams は自動的にエンコード/デコードを行う
const params = new URLSearchParams();
params.set('q', '日本語 検索');
params.toString();  // 'q=%E6%97%A5%E6%9C%AC%E8%AA%9E+%E6%A4%9C%E7%B4%A2'

params.get('q');    // '日本語 検索'（自動デコード）

// 注意: + はスペースとして扱われる
const params2 = new URLSearchParams('q=hello+world');
params2.get('q');   // 'hello world'

// 注意: encodeURIComponent との違い
encodeURIComponent('hello world');  // 'hello%20world'（%20）
new URLSearchParams({ q: 'hello world' }).toString();  // 'q=hello+world'（+）

// 特殊文字のエンコーディング
const specialParams = new URLSearchParams();
specialParams.set('filter', 'price>100&stock>0');
specialParams.toString();  // 'filter=price%3E100%26stock%3E0'
// & や > が正しくエンコードされる
```

---

## 2. useSearchParams（React Router / Next.js）

### 2.1 React Router v6 での useSearchParams

React Router v6 は `useSearchParams` フックを提供し、URLSearchParams のReactラッパーとして機能する。

```typescript
// === React Router v6 の useSearchParams ===
import { useSearchParams } from 'react-router-dom';

function ProductListPage() {
  // URLSearchParams のReactラッパー
  const [searchParams, setSearchParams] = useSearchParams();

  // 値の取得（常に string | null）
  const query = searchParams.get('q') ?? '';
  const page = Number(searchParams.get('page') ?? '1');
  const sort = searchParams.get('sort') ?? 'newest';
  const categories = searchParams.getAll('category');

  // 方法1: 関数型アップデート（推奨）
  function handleSortChange(newSort: string) {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev);
      next.set('sort', newSort);
      next.delete('page'); // ソート変更時にページリセット
      return next;
    });
  }

  // 方法2: オブジェクト指定（全パラメータを置換）
  function handleReset() {
    setSearchParams({}); // 全パラメータをクリア
  }

  // 方法3: URLSearchParams を直接渡す
  function handleSearch(q: string) {
    const params = new URLSearchParams(searchParams);
    if (q) {
      params.set('q', q);
    } else {
      params.delete('q');
    }
    params.delete('page'); // 検索変更時にページリセット
    setSearchParams(params);
  }

  // 方法4: replace オプション（ブラウザ履歴に残さない）
  function handlePageChange(newPage: number) {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev);
      next.set('page', String(newPage));
      return next;
    }, { replace: true }); // 履歴スタックに追加しない
  }

  return (
    <div>
      <SearchInput value={query} onSearch={handleSearch} />
      <SortSelect value={sort} onChange={handleSortChange} />
      <CategoryFilter selected={categories} onChange={handleCategoryChange} />
      <ProductGrid query={query} page={page} sort={sort} />
      <Pagination
        currentPage={page}
        onPageChange={handlePageChange}
      />
      <button onClick={handleReset}>Clear Filters</button>
    </div>
  );
}
```

```typescript
// === React Router: useSearchParams のデフォルト値パターン ===

// パターン1: インラインデフォルト値
function useProductFilters() {
  const [searchParams, setSearchParams] = useSearchParams();

  return {
    query: searchParams.get('q') ?? '',
    page: Number(searchParams.get('page')) || 1,
    sort: searchParams.get('sort') ?? 'newest',
    category: searchParams.get('category') ?? 'all',
    perPage: Number(searchParams.get('per_page')) || 20,
  };
}

// パターン2: デフォルト値をuseSearchParamsに渡す
function ProductPage() {
  const [searchParams, setSearchParams] = useSearchParams({
    sort: 'newest',
    page: '1',
    per_page: '20',
  });

  // デフォルト値が設定された状態でスタート
  const sort = searchParams.get('sort')!; // non-null
  const page = Number(searchParams.get('page')!);
}

// パターン3: カスタムフックでラップ
interface ProductFilterState {
  query: string;
  page: number;
  sort: 'newest' | 'price-asc' | 'price-desc' | 'popular';
  category: string | null;
  tags: string[];
  priceMin: number | null;
  priceMax: number | null;
}

const DEFAULT_FILTERS: ProductFilterState = {
  query: '',
  page: 1,
  sort: 'newest',
  category: null,
  tags: [],
  priceMin: null,
  priceMax: null,
};

function useProductFiltersAdvanced() {
  const [searchParams, setSearchParams] = useSearchParams();

  const filters: ProductFilterState = {
    query: searchParams.get('q') ?? DEFAULT_FILTERS.query,
    page: Number(searchParams.get('page')) || DEFAULT_FILTERS.page,
    sort: (searchParams.get('sort') as ProductFilterState['sort'])
      ?? DEFAULT_FILTERS.sort,
    category: searchParams.get('category') ?? DEFAULT_FILTERS.category,
    tags: searchParams.getAll('tag'),
    priceMin: searchParams.has('price_min')
      ? Number(searchParams.get('price_min'))
      : null,
    priceMax: searchParams.has('price_max')
      ? Number(searchParams.get('price_max'))
      : null,
  };

  function setFilters(updates: Partial<ProductFilterState>) {
    setSearchParams(prev => {
      const params = new URLSearchParams(prev);

      // query
      if ('query' in updates) {
        if (updates.query) {
          params.set('q', updates.query);
        } else {
          params.delete('q');
        }
      }

      // page
      if ('page' in updates) {
        if (updates.page && updates.page > 1) {
          params.set('page', String(updates.page));
        } else {
          params.delete('page');
        }
      }

      // sort
      if ('sort' in updates) {
        if (updates.sort && updates.sort !== DEFAULT_FILTERS.sort) {
          params.set('sort', updates.sort);
        } else {
          params.delete('sort');
        }
      }

      // category
      if ('category' in updates) {
        if (updates.category) {
          params.set('category', updates.category);
        } else {
          params.delete('category');
        }
      }

      // tags（配列）
      if ('tags' in updates) {
        params.delete('tag');
        updates.tags?.forEach(tag => params.append('tag', tag));
      }

      // price range
      if ('priceMin' in updates) {
        if (updates.priceMin !== null && updates.priceMin !== undefined) {
          params.set('price_min', String(updates.priceMin));
        } else {
          params.delete('price_min');
        }
      }

      if ('priceMax' in updates) {
        if (updates.priceMax !== null && updates.priceMax !== undefined) {
          params.set('price_max', String(updates.priceMax));
        } else {
          params.delete('price_max');
        }
      }

      // フィルタ変更時にページをリセット
      if (!('page' in updates)) {
        params.delete('page');
      }

      return params;
    });
  }

  function resetFilters() {
    setSearchParams({});
  }

  const isFiltered = searchParams.toString() !== '';

  return { filters, setFilters, resetFilters, isFiltered };
}
```

### 2.2 Next.js App Router での useSearchParams

Next.js App Router では、`useSearchParams` は読み取り専用のフックであり、URL更新には `useRouter` を併用する。

```typescript
// === Next.js App Router の useSearchParams ===
'use client';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { useCallback, useTransition } from 'react';

function ProductFilters() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [isPending, startTransition] = useTransition();

  // 値の取得
  const category = searchParams.get('category') ?? 'all';
  const sort = searchParams.get('sort') ?? 'newest';
  const page = Number(searchParams.get('page') ?? '1');
  const query = searchParams.get('q') ?? '';

  // URL更新ヘルパー（useCallback でメモ化）
  const updateParams = useCallback(
    (updates: Record<string, string | null>) => {
      const params = new URLSearchParams(searchParams.toString());

      for (const [key, value] of Object.entries(updates)) {
        if (value === null) {
          params.delete(key);
        } else {
          params.set(key, value);
        }
      }

      // フィルタ変更時はページをリセット
      if (!('page' in updates)) {
        params.delete('page');
      }

      // useTransition でUIのブロッキングを防ぐ
      startTransition(() => {
        router.push(`${pathname}?${params.toString()}`, {
          scroll: false, // スクロール位置を維持
        });
      });
    },
    [searchParams, router, pathname, startTransition]
  );

  return (
    <div className={isPending ? 'opacity-50' : ''}>
      {/* 検索入力 */}
      <SearchInput
        defaultValue={query}
        onSearch={(q) => updateParams({ q: q || null })}
      />

      {/* カテゴリ選択 */}
      <select
        value={category}
        onChange={(e) => updateParams({
          category: e.target.value === 'all' ? null : e.target.value,
        })}
      >
        <option value="all">All Categories</option>
        <option value="electronics">Electronics</option>
        <option value="books">Books</option>
        <option value="clothing">Clothing</option>
      </select>

      {/* ソート選択 */}
      <select
        value={sort}
        onChange={(e) => updateParams({ sort: e.target.value })}
      >
        <option value="newest">Newest</option>
        <option value="price-asc">Price: Low to High</option>
        <option value="price-desc">Price: High to Low</option>
        <option value="popular">Most Popular</option>
      </select>

      {/* ページネーション */}
      <Pagination
        currentPage={page}
        onPageChange={(p) => updateParams({ page: String(p) })}
      />

      {/* ペンディングインジケータ */}
      {isPending && <LoadingSpinner />}
    </div>
  );
}
```

```typescript
// === Next.js: createQueryString ユーティリティ ===
'use client';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';

/**
 * Next.js App Router 向け URL パラメータ管理フック
 */
function useURLParams() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  /**
   * 新しいクエリ文字列を生成する（イミュータブル）
   */
  const createQueryString = useCallback(
    (params: Record<string, string | string[] | null>) => {
      const newParams = new URLSearchParams(searchParams.toString());

      for (const [key, value] of Object.entries(params)) {
        if (value === null) {
          newParams.delete(key);
        } else if (Array.isArray(value)) {
          newParams.delete(key);
          value.forEach(v => newParams.append(key, v));
        } else {
          newParams.set(key, value);
        }
      }

      return newParams.toString();
    },
    [searchParams]
  );

  /**
   * URLを更新（push）
   */
  const pushParams = useCallback(
    (params: Record<string, string | string[] | null>, options?: {
      scroll?: boolean;
      resetPage?: boolean;
    }) => {
      const updates = { ...params };
      if (options?.resetPage && !('page' in updates)) {
        updates.page = null;
      }

      const queryString = createQueryString(updates);
      const url = queryString ? `${pathname}?${queryString}` : pathname;
      router.push(url, { scroll: options?.scroll ?? false });
    },
    [createQueryString, pathname, router]
  );

  /**
   * URLを更新（replace - 履歴に残さない）
   */
  const replaceParams = useCallback(
    (params: Record<string, string | string[] | null>) => {
      const queryString = createQueryString(params);
      const url = queryString ? `${pathname}?${queryString}` : pathname;
      router.replace(url, { scroll: false });
    },
    [createQueryString, pathname, router]
  );

  /**
   * 全パラメータをクリア
   */
  const clearParams = useCallback(() => {
    router.push(pathname, { scroll: false });
  }, [pathname, router]);

  return {
    searchParams,
    createQueryString,
    pushParams,
    replaceParams,
    clearParams,
  };
}

// 使用例
function FilterComponent() {
  const { searchParams, pushParams, clearParams } = useURLParams();

  return (
    <div>
      <button onClick={() => pushParams(
        { category: 'electronics', sort: 'price-asc' },
        { resetPage: true }
      )}>
        Electronics (Cheapest)
      </button>
      <button onClick={() => clearParams()}>
        Clear All Filters
      </button>
    </div>
  );
}
```

### 2.3 React Router v6 と Next.js の比較

```
┌─────────────────────────┬────────────────────────┬───────────────────────────┐
│ 機能                     │ React Router v6        │ Next.js App Router        │
├─────────────────────────┼────────────────────────┼───────────────────────────┤
│ フック名                 │ useSearchParams        │ useSearchParams           │
│ 戻り値                   │ [params, setParams]    │ ReadonlyURLSearchParams   │
│ 書き込み方法             │ setSearchParams()      │ router.push / replace     │
│ replace オプション       │ { replace: true }      │ router.replace()          │
│ scroll 制御              │ なし（手動）           │ { scroll: false }         │
│ Suspense 必要            │ 不要                   │ 必要（Suspense boundary） │
│ Server Component 対応    │ なし                   │ searchParams prop         │
│ Transition 対応          │ 手動                   │ useTransition 統合        │
│ 配列パラメータ           │ getAll()               │ getAll()                  │
│ 型安全性                 │ 低い（手動パース）     │ 低い（手動パース）        │
└─────────────────────────┴────────────────────────┴───────────────────────────┘
```

### 2.4 Next.js の Suspense boundary と useSearchParams

Next.js App Router で `useSearchParams` を使う場合、Suspense boundary が必要である。これは、静的レンダリング時にクライアントサイドの値が確定しないためである。

```typescript
// === Suspense boundary が必要な理由と対処法 ===

// NG: Suspense なしで useSearchParams を使うとビルドエラー
function Page() {
  return <ProductFilters />; // エラー: useSearchParams requires Suspense
}

// OK: Suspense で囲む
import { Suspense } from 'react';

function Page() {
  return (
    <Suspense fallback={<FiltersSkeleton />}>
      <ProductFilters />
    </Suspense>
  );
}

// OK: コンポーネントを分離するパターン
// app/products/page.tsx（Server Component）
export default function ProductsPage() {
  return (
    <div>
      <h1>Products</h1>
      <Suspense fallback={<div>Loading filters...</div>}>
        <ProductFilters />  {/* Client Component with useSearchParams */}
      </Suspense>
      <Suspense fallback={<ProductGridSkeleton />}>
        <ProductGrid />     {/* Server Component or Client Component */}
      </Suspense>
    </div>
  );
}

// ProductFilters.tsx（Client Component）
'use client';
import { useSearchParams } from 'next/navigation';

function ProductFilters() {
  const searchParams = useSearchParams();
  // ... フィルタのUI
}
```

---

## 3. nuqs（型安全なURL状態管理）

### 3.1 nuqs の概要と利点

nuqs（旧名: next-usequerystate）は、Next.js 向けに設計された型安全な search params 管理ライブラリである。2024年以降、React Router や Remix でも利用可能になり、フレームワーク非依存のURL状態管理ライブラリへと進化している。

```
nuqs の主な利点:

1. 型安全性
   - パーサーを使って string → 適切な型に自動変換
   - TypeScript の型推論が完全に効く
   - コンパイル時にエラーを検出

2. デフォルト値
   - withDefault() で型安全なデフォルト値を設定
   - URL にデフォルト値は含まれない（URLがクリーン）

3. シリアライズ/デシリアライズ
   - 数値、boolean、日付、列挙型、JSON を自動処理
   - カスタムパーサーも定義可能

4. バッチ更新
   - 複数のパラメータを一度に更新
   - 不要な再レンダリングを防止

5. shallow routing 対応
   - デフォルトでページ再読み込みなし
   - Next.js のデータ再取得と統合可能

6. Server Component 統合
   - createSearchParamsCache で Server Component からも利用可能
   - SSR 時のパフォーマンス最適化
```

### 3.2 インストールとセットアップ

```bash
# インストール
npm install nuqs

# Next.js App Router の場合: layout.tsx にプロバイダーを設定
```

```typescript
// app/layout.tsx
import { NuqsAdapter } from 'nuqs/adapters/next/app';

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ja">
      <body>
        <NuqsAdapter>{children}</NuqsAdapter>
      </body>
    </html>
  );
}

// Next.js Pages Router の場合
// pages/_app.tsx
import { NuqsAdapter } from 'nuqs/adapters/next/pages';

export default function App({ Component, pageProps }) {
  return (
    <NuqsAdapter>
      <Component {...pageProps} />
    </NuqsAdapter>
  );
}

// React Router の場合
import { NuqsAdapter } from 'nuqs/adapters/react-router/v7';

// Remix の場合
import { NuqsAdapter } from 'nuqs/adapters/remix';
```

### 3.3 基本的な使い方

```typescript
// === nuqs の基本: useQueryState ===
'use client';
import {
  useQueryState,
  parseAsInteger,
  parseAsBoolean,
  parseAsStringEnum,
  parseAsFloat,
  parseAsIsoDateTime,
  parseAsJson,
  parseAsArrayOf,
  parseAsString,
} from 'nuqs';

function ProductPage() {
  // 文字列パラメータ（デフォルトのパーサー）
  const [query, setQuery] = useQueryState('q', { defaultValue: '' });
  // URL: ?q=laptop → query = 'laptop'
  // URL: (なし)    → query = ''

  // 数値パラメータ
  const [page, setPage] = useQueryState(
    'page',
    parseAsInteger.withDefault(1)
  );
  // URL: ?page=3 → page = 3（number型）
  // URL: (なし)  → page = 1

  // 列挙型パラメータ
  const [sort, setSort] = useQueryState(
    'sort',
    parseAsStringEnum(['newest', 'price-asc', 'price-desc', 'popular'])
      .withDefault('newest')
  );
  // URL: ?sort=price-asc → sort = 'price-asc'
  // 不正な値は無視される

  // boolean パラメータ
  const [inStock, setInStock] = useQueryState(
    'in_stock',
    parseAsBoolean.withDefault(false)
  );
  // URL: ?in_stock=true → inStock = true

  // null 可能なパラメータ（withDefault なし）
  const [category, setCategory] = useQueryState('category');
  // URL: ?category=books → category = 'books'
  // URL: (なし)           → category = null

  return (
    <div>
      {/* 文字列入力 */}
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value || null)}
        placeholder="Search..."
      />

      {/* ソート選択 */}
      <select
        value={sort}
        onChange={(e) => setSort(e.target.value as typeof sort)}
      >
        <option value="newest">Newest</option>
        <option value="price-asc">Price: Low to High</option>
        <option value="price-desc">Price: High to Low</option>
        <option value="popular">Most Popular</option>
      </select>

      {/* ページネーション */}
      <button
        disabled={page <= 1}
        onClick={() => setPage(page - 1)}
      >
        Previous
      </button>
      <span>Page {page}</span>
      <button onClick={() => setPage(page + 1)}>
        Next
      </button>

      {/* 在庫フィルタ */}
      <label>
        <input
          type="checkbox"
          checked={inStock}
          onChange={(e) => setInStock(e.target.checked || null)}
        />
        In Stock Only
      </label>

      {/* カテゴリ（null 可能） */}
      <select
        value={category ?? ''}
        onChange={(e) => setCategory(e.target.value || null)}
      >
        <option value="">All Categories</option>
        <option value="electronics">Electronics</option>
        <option value="books">Books</option>
      </select>
    </div>
  );
}
```

### 3.4 高度なパーサー

```typescript
// === nuqs の高度なパーサー ===

// 1. 浮動小数点数
const [price, setPrice] = useQueryState(
  'price',
  parseAsFloat.withDefault(0)
);
// URL: ?price=29.99 → price = 29.99

// 2. ISO日時
const [date, setDate] = useQueryState(
  'date',
  parseAsIsoDateTime.withDefault(new Date())
);
// URL: ?date=2024-03-15T10:30:00.000Z → Date オブジェクト

// 3. 配列（カンマ区切り）
const [tags, setTags] = useQueryState(
  'tags',
  parseAsArrayOf(parseAsString, ',').withDefault([])
);
// URL: ?tags=sale,new,popular → tags = ['sale', 'new', 'popular']

// 4. 数値配列
const [ids, setIds] = useQueryState(
  'ids',
  parseAsArrayOf(parseAsInteger, ',').withDefault([])
);
// URL: ?ids=1,2,3 → ids = [1, 2, 3]

// 5. JSON パラメータ（複雑なオブジェクト）
interface PriceRange {
  min: number;
  max: number;
}

const [priceRange, setPriceRange] = useQueryState<PriceRange>(
  'price_range',
  parseAsJson<PriceRange>().withDefault({ min: 0, max: 10000 })
);
// URL: ?price_range={"min":100,"max":500}

// 6. カスタムパーサー
import { createParser } from 'nuqs';

// カスタム: カンマ区切りの数値範囲
const parseAsRange = createParser({
  parse: (value: string) => {
    const [min, max] = value.split('-').map(Number);
    if (isNaN(min) || isNaN(max)) return null;
    return { min, max };
  },
  serialize: (value: { min: number; max: number }) =>
    `${value.min}-${value.max}`,
});

const [range, setRange] = useQueryState(
  'range',
  parseAsRange.withDefault({ min: 0, max: 100 })
);
// URL: ?range=10-50 → { min: 10, max: 50 }

// 7. カスタム: スラッグ化された文字列
const parseAsSlug = createParser({
  parse: (value: string) => value.replace(/-/g, ' '),
  serialize: (value: string) =>
    value.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, ''),
});

const [searchTerm, setSearchTerm] = useQueryState(
  'q',
  parseAsSlug.withDefault('')
);
// URL: ?q=red-shoes → searchTerm = 'red shoes'
// setSearchTerm('Blue Jacket') → URL: ?q=blue-jacket
```

### 3.5 useQueryStates（複数パラメータの一括管理）

```typescript
// === useQueryStates: 複数パラメータを一度に管理 ===
import { useQueryStates, parseAsInteger, parseAsStringEnum } from 'nuqs';

// パーサー定義を一箇所にまとめる
const productFiltersParsers = {
  q: parseAsString.withDefault(''),
  page: parseAsInteger.withDefault(1),
  per_page: parseAsInteger.withDefault(20),
  sort: parseAsStringEnum(['newest', 'price-asc', 'price-desc', 'popular'])
    .withDefault('newest'),
  category: parseAsString,
  tags: parseAsArrayOf(parseAsString, ',').withDefault([]),
  in_stock: parseAsBoolean.withDefault(false),
  price_min: parseAsInteger,
  price_max: parseAsInteger,
};

function ProductPage() {
  // 全パラメータを一括で管理
  const [filters, setFilters] = useQueryStates(productFiltersParsers);

  // filters の型は自動推論される:
  // {
  //   q: string;
  //   page: number;
  //   per_page: number;
  //   sort: 'newest' | 'price-asc' | 'price-desc' | 'popular';
  //   category: string | null;
  //   tags: string[];
  //   in_stock: boolean;
  //   price_min: number | null;
  //   price_max: number | null;
  // }

  // 一括更新（バッチ処理 → 1回のURL更新）
  function handleCategoryChange(category: string) {
    setFilters({
      category: category || null,
      page: 1,  // ページリセット
    });
  }

  // 全リセット
  function handleReset() {
    setFilters({
      q: '',
      page: 1,
      per_page: 20,
      sort: 'newest',
      category: null,
      tags: [],
      in_stock: false,
      price_min: null,
      price_max: null,
    });
  }

  // 条件付き更新
  function handlePriceRangeChange(min: number | null, max: number | null) {
    setFilters({
      price_min: min,
      price_max: max,
      page: 1,  // ページリセット
    });
  }

  return (
    <div>
      <p>
        {filters.q && `Search: "${filters.q}"`}
        {filters.category && ` | Category: ${filters.category}`}
        {filters.tags.length > 0 && ` | Tags: ${filters.tags.join(', ')}`}
      </p>
      {/* UI components */}
    </div>
  );
}
```

### 3.6 nuqs + TanStack Query の統合

```typescript
// === nuqs + TanStack Query: URL状態をデータフェッチに連動 ===
import { useQueryStates, parseAsInteger, parseAsString, parseAsStringEnum } from 'nuqs';
import { useQuery, keepPreviousData } from '@tanstack/react-query';

const searchParsers = {
  q: parseAsString.withDefault(''),
  page: parseAsInteger.withDefault(1),
  sort: parseAsStringEnum(['newest', 'price-asc', 'price-desc'])
    .withDefault('newest'),
  category: parseAsString,
};

function useProducts() {
  const [filters] = useQueryStates(searchParsers);

  return useQuery({
    // URL パラメータをそのまま queryKey に使用
    queryKey: ['products', filters],
    queryFn: () => api.products.list({
      q: filters.q || undefined,
      page: filters.page,
      sort: filters.sort,
      category: filters.category ?? undefined,
    }),
    // ページ切り替え時にチラつきを防止
    placeholderData: keepPreviousData,
    // URL パラメータが変わるたびに自動再フェッチ
    staleTime: 30_000,
  });
}

function ProductListPage() {
  const [filters, setFilters] = useQueryStates(searchParsers);
  const { data, isLoading, isFetching } = useProducts();

  return (
    <div>
      {/* フィルタUI */}
      <SearchInput
        value={filters.q}
        onChange={(q) => setFilters({ q: q || null, page: 1 })}
      />

      {/* ローディング状態 */}
      {isLoading && <ProductGridSkeleton />}

      {/* データ表示 */}
      {data && (
        <>
          <p>{data.meta.total} products found</p>
          {isFetching && <ProgressBar />}
          <ProductGrid products={data.items} />
          <Pagination
            currentPage={filters.page}
            totalPages={data.meta.totalPages}
            onPageChange={(page) => setFilters({ page })}
          />
        </>
      )}
    </div>
  );
}
```

### 3.7 nuqs の Server Component 統合

```typescript
// === nuqs: Server Component での searchParams キャッシュ ===
import { createSearchParamsCache } from 'nuqs/server';
import { parseAsInteger, parseAsString, parseAsStringEnum } from 'nuqs';

// Server 用のパーサーキャッシュを定義
const searchParamsCache = createSearchParamsCache({
  q: parseAsString.withDefault(''),
  page: parseAsInteger.withDefault(1),
  sort: parseAsStringEnum(['newest', 'price-asc', 'price-desc'])
    .withDefault('newest'),
  category: parseAsString,
});

// app/products/page.tsx（Server Component）
export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[]>>;
}) {
  // 型安全に searchParams をパース
  const { q, page, sort, category } = await searchParamsCache.parse(
    await searchParams
  );

  // Server Component でデータフェッチ
  const products = await getProducts({
    query: q || undefined,
    page,
    sort,
    category: category ?? undefined,
  });

  return (
    <div>
      <h1>Products</h1>
      <Suspense fallback={<FiltersSkeleton />}>
        <ProductFilters />  {/* Client Component で nuqs を使用 */}
      </Suspense>
      <ProductGrid products={products.data} />
      <Pagination
        currentPage={page}
        totalPages={products.meta.totalPages}
      />
    </div>
  );
}
```

---

## 4. Server Component との統合

### 4.1 Next.js App Router での searchParams

Next.js App Router では、Server Component のページコンポーネントが `searchParams` をpropsとして受け取ることができる。これにより、サーバーサイドでURL状態に基づいたデータ取得が可能になる。

```typescript
// === Next.js App Router: Server Component での searchParams ===
// app/products/page.tsx

interface SearchParams {
  q?: string;
  page?: string;
  sort?: string;
  category?: string;
  tags?: string | string[];
  price_min?: string;
  price_max?: string;
}

export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;

  // パラメータのパースとバリデーション
  const page = Math.max(1, Number(params.page ?? '1'));
  const sort = validateSort(params.sort ?? 'newest');
  const category = params.category ?? null;
  const tags = Array.isArray(params.tags)
    ? params.tags
    : params.tags
      ? [params.tags]
      : [];
  const priceMin = params.price_min ? Number(params.price_min) : undefined;
  const priceMax = params.price_max ? Number(params.price_max) : undefined;

  // サーバーサイドでデータフェッチ
  const products = await getProducts({
    query: params.q,
    page,
    sort,
    category,
    tags,
    priceMin,
    priceMax,
  });

  return (
    <div>
      {/* SEO: 動的メタデータ */}
      <h1>
        {category ? `${category} Products` : 'All Products'}
        {params.q && ` - "${params.q}"`}
      </h1>

      {/* Client Component: フィルタUI */}
      <Suspense fallback={<FiltersSkeleton />}>
        <ProductFilters />
      </Suspense>

      {/* Server Component: 結果表示 */}
      <p>{products.meta.total} products found</p>
      <ProductGrid products={products.data} />

      {/* Server Component: ページネーション（リンクベース） */}
      <ServerPagination
        currentPage={page}
        totalPages={products.meta.totalPages}
        searchParams={params}
      />
    </div>
  );
}

// バリデーション関数
function validateSort(sort: string): string {
  const validSorts = ['newest', 'oldest', 'price-asc', 'price-desc', 'popular'];
  return validSorts.includes(sort) ? sort : 'newest';
}
```

### 4.2 Server Component のページネーション（リンクベース）

Server Component では `useSearchParams` が使えないため、`<Link>` コンポーネントを使ったページネーションが推奨される。

```typescript
// === Server Component: リンクベースのページネーション ===
import Link from 'next/link';

interface ServerPaginationProps {
  currentPage: number;
  totalPages: number;
  searchParams: Record<string, string | string[] | undefined>;
}

function ServerPagination({
  currentPage,
  totalPages,
  searchParams,
}: ServerPaginationProps) {
  function createPageURL(page: number): string {
    const params = new URLSearchParams();

    for (const [key, value] of Object.entries(searchParams)) {
      if (key === 'page') continue; // page は別途設定
      if (value === undefined) continue;

      if (Array.isArray(value)) {
        value.forEach(v => params.append(key, v));
      } else {
        params.set(key, value);
      }
    }

    if (page > 1) {
      params.set('page', String(page));
    }

    const queryString = params.toString();
    return queryString ? `?${queryString}` : '';
  }

  // ページ番号のリストを生成
  const pages = generatePageNumbers(currentPage, totalPages);

  return (
    <nav aria-label="Pagination">
      {/* 前のページ */}
      {currentPage > 1 ? (
        <Link href={createPageURL(currentPage - 1)}>
          Previous
        </Link>
      ) : (
        <span aria-disabled="true">Previous</span>
      )}

      {/* ページ番号 */}
      {pages.map((page, index) => (
        page === '...' ? (
          <span key={`ellipsis-${index}`}>...</span>
        ) : (
          <Link
            key={page}
            href={createPageURL(Number(page))}
            aria-current={Number(page) === currentPage ? 'page' : undefined}
            className={Number(page) === currentPage ? 'active' : ''}
          >
            {page}
          </Link>
        )
      ))}

      {/* 次のページ */}
      {currentPage < totalPages ? (
        <Link href={createPageURL(currentPage + 1)}>
          Next
        </Link>
      ) : (
        <span aria-disabled="true">Next</span>
      )}
    </nav>
  );
}

// ページ番号生成ロジック
function generatePageNumbers(
  current: number,
  total: number
): (string | number)[] {
  if (total <= 7) {
    return Array.from({ length: total }, (_, i) => i + 1);
  }

  if (current <= 3) {
    return [1, 2, 3, 4, '...', total];
  }

  if (current >= total - 2) {
    return [1, '...', total - 3, total - 2, total - 1, total];
  }

  return [1, '...', current - 1, current, current + 1, '...', total];
}
```

### 4.3 動的メタデータとURL状態

URL状態に基づいて動的にメタデータを生成することで、SEO最適化を実現できる。

```typescript
// === 動的メタデータの生成 ===
// app/products/page.tsx

import { Metadata } from 'next';

interface Props {
  searchParams: Promise<{
    q?: string;
    category?: string;
    page?: string;
  }>;
}

export async function generateMetadata({
  searchParams,
}: Props): Promise<Metadata> {
  const params = await searchParams;

  let title = 'Products';
  let description = 'Browse our product catalog';

  if (params.q) {
    title = `Search: "${params.q}" - Products`;
    description = `Search results for "${params.q}"`;
  }

  if (params.category) {
    title = `${params.category} - Products`;
    description = `Browse ${params.category} products`;
  }

  const page = Number(params.page ?? '1');
  if (page > 1) {
    title += ` (Page ${page})`;
  }

  return {
    title,
    description,
    // canonical URL でページネーションの重複を防ぐ
    alternates: {
      canonical: params.q
        ? `/products?q=${encodeURIComponent(params.q)}`
        : params.category
          ? `/products?category=${params.category}`
          : '/products',
    },
    // 検索結果ページはインデックスしない
    robots: params.q ? { index: false } : undefined,
  };
}
```

### 4.4 Streaming SSR とURL状態

Next.js の Streaming SSR を活用して、URL状態に基づくデータフェッチを効率化できる。

```typescript
// === Streaming SSR パターン ===
// app/products/page.tsx

import { Suspense } from 'react';

export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string>>;
}) {
  const params = await searchParams;

  return (
    <div>
      <h1>Products</h1>

      {/* フィルタUI: すぐに表示（Client Component） */}
      <Suspense fallback={<FiltersSkeleton />}>
        <ProductFilters />
      </Suspense>

      {/* 検索結果: データフェッチを待ってストリーミング */}
      <Suspense fallback={<ProductGridSkeleton count={20} />}>
        <ProductResults searchParams={params} />
      </Suspense>

      {/* サイドバー: 独立してストリーミング */}
      <Suspense fallback={<SidebarSkeleton />}>
        <CategorySidebar />
      </Suspense>
    </div>
  );
}

// データフェッチを含む Server Component
async function ProductResults({
  searchParams,
}: {
  searchParams: Record<string, string>;
}) {
  // このデータフェッチが完了するまでフォールバックが表示される
  const products = await getProducts({
    query: searchParams.q,
    page: Number(searchParams.page ?? '1'),
    sort: searchParams.sort ?? 'newest',
    category: searchParams.category,
  });

  return (
    <>
      <p>{products.meta.total} products found</p>
      <ProductGrid products={products.data} />
      <ServerPagination
        currentPage={Number(searchParams.page ?? '1')}
        totalPages={products.meta.totalPages}
        searchParams={searchParams}
      />
    </>
  );
}
```

---

## 5. 設計パターンとベストプラクティス

### 5.1 URL状態の設計原則

```
URL状態の設計原則:

  ① デフォルト値はURL に含めない:
     /products              ← デフォルト（page=1, sort=newest）
     /products?sort=price   ← ソートだけ変更
     → URLがシンプルになる
     → デフォルト値が変更されても既存のブックマークが壊れない

  ② フィルタ変更時にページをリセット:
     /products?category=books&page=3
     → category変更 → /products?category=electronics（page=1にリセット）
     → ユーザーが存在しないページを見ることを防ぐ

  ③ 配列パラメータの方式を統一:
     方式A: /products?tag=sale&tag=new     ← 同じキーを複数（Web標準）
     方式B: /products?tags=sale,new         ← カンマ区切り（シンプル）
     → プロジェクト全体で一つの方式に統一する

  ④ デバウンス（検索入力）:
     → 入力ごとにURLを更新しない
     → 300-500ms のデバウンスを入れる
     → ブラウザ履歴が汚れるのを防ぐ

  ⑤ shallow routing:
     → URLを更新してもページ全体を再レンダリングしない
     → Next.js: router.push(url, { scroll: false })
     → データフェッチはクライアントサイドで行う

  ⑥ URL の正規化:
     → パラメータの順序を統一する（キー名の辞書順）
     → 大文字/小文字を統一する
     → 不要な空白やエンコーディングを除去する

  ⑦ 後方互換性:
     → パラメータ名を変更する際は旧パラメータもサポートする
     → リダイレクトで旧URLを新URLに転送する
```

### 5.2 デバウンスパターン

検索入力のように頻繁に変化する値は、デバウンスを使ってURL更新の頻度を制限する必要がある。

```typescript
// === デバウンス: 検索入力のURL同期 ===
'use client';
import { useState, useEffect, useCallback } from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';

/**
 * デバウンスフック
 */
function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

/**
 * デバウンス付き検索入力コンポーネント
 */
function DebouncedSearchInput() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  // ローカル状態（即座に更新）
  const [inputValue, setInputValue] = useState(
    searchParams.get('q') ?? ''
  );

  // デバウンスされた値（300ms後に確定）
  const debouncedQuery = useDebounce(inputValue, 300);

  // デバウンスされた値が変更されたらURLを更新
  useEffect(() => {
    const params = new URLSearchParams(searchParams.toString());

    if (debouncedQuery) {
      params.set('q', debouncedQuery);
    } else {
      params.delete('q');
    }
    params.delete('page'); // 検索変更時にページリセット

    const queryString = params.toString();
    const url = queryString ? `${pathname}?${queryString}` : pathname;

    // replace: 入力中の履歴を残さない
    router.replace(url, { scroll: false });
  }, [debouncedQuery, pathname, router, searchParams]);

  // URLが外部から変更された場合（ブラウザバック等）に同期
  useEffect(() => {
    const urlQuery = searchParams.get('q') ?? '';
    if (urlQuery !== inputValue) {
      setInputValue(urlQuery);
    }
  }, [searchParams]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <input
      type="search"
      value={inputValue}
      onChange={(e) => setInputValue(e.target.value)}
      placeholder="Search products..."
    />
  );
}
```

```typescript
// === nuqs でのデバウンス ===
import { useQueryState } from 'nuqs';

function SearchWithNuqs() {
  const [query, setQuery] = useQueryState('q', {
    defaultValue: '',
    // nuqs v2+ では shallow オプションで制御
    shallow: true, // サーバーへのリクエストを防ぐ
    throttleMs: 300, // URL更新をスロットル
    history: 'replace', // 入力中は履歴を汚さない
  });

  return (
    <input
      type="search"
      value={query}
      onChange={(e) => setQuery(e.target.value || null)}
      placeholder="Search..."
    />
  );
}
```

### 5.3 URL状態のバリデーション

ユーザーがURLを直接編集する可能性があるため、URL状態のバリデーションは必須である。

```typescript
// === URL状態のバリデーション ===
import { z } from 'zod';

// Zod スキーマでURL状態を定義
const searchParamsSchema = z.object({
  q: z.string().max(200).optional().default(''),
  page: z.coerce.number().int().positive().optional().default(1),
  per_page: z.coerce.number().int().min(10).max(100).optional().default(20),
  sort: z.enum(['newest', 'oldest', 'price-asc', 'price-desc', 'popular'])
    .optional().default('newest'),
  category: z.string().max(50).optional(),
  tags: z.string().transform(s => s ? s.split(',') : []).optional().default(''),
  price_min: z.coerce.number().nonnegative().optional(),
  price_max: z.coerce.number().nonnegative().optional(),
}).refine(
  (data) => {
    if (data.price_min !== undefined && data.price_max !== undefined) {
      return data.price_min <= data.price_max;
    }
    return true;
  },
  { message: 'price_min must be less than or equal to price_max' }
);

type ValidatedSearchParams = z.infer<typeof searchParamsSchema>;

// Server Component でのバリデーション
export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string>>;
}) {
  const rawParams = await searchParams;

  // バリデーションとデフォルト値の適用
  const result = searchParamsSchema.safeParse(rawParams);

  if (!result.success) {
    // 不正なパラメータの場合はデフォルト値にリダイレクト
    redirect('/products');
  }

  const validParams = result.data;

  // ... データフェッチ
}

// カスタムフックでのバリデーション（Client Component）
function useValidatedSearchParams() {
  const searchParams = useSearchParams();

  const rawParams = Object.fromEntries(searchParams.entries());
  const result = searchParamsSchema.safeParse(rawParams);

  if (!result.success) {
    // デフォルト値を返す
    return searchParamsSchema.parse({});
  }

  return result.data;
}
```

```typescript
// === XSS 対策: URL状態のサニタイズ ===

/**
 * URL パラメータのサニタイズ
 * XSSやインジェクション攻撃を防ぐ
 */
function sanitizeSearchParam(value: string): string {
  // HTMLタグを除去
  const sanitized = value.replace(/<[^>]*>/g, '');

  // 制御文字を除去
  const cleaned = sanitized.replace(/[\x00-\x1F\x7F]/g, '');

  // 最大長を制限
  return cleaned.slice(0, 500);
}

/**
 * URL パラメータを安全に表示する
 */
function SafeSearchDisplay({ query }: { query: string }) {
  // React はデフォルトでXSSを防ぐが、追加の保護として
  const safeQuery = sanitizeSearchParam(query);

  return (
    <h2>
      Search results for: <span className="highlight">{safeQuery}</span>
    </h2>
  );
}

// 危険: URLパラメータを直接 dangerouslySetInnerHTML に使わない
// NG:
function DangerousComponent() {
  const searchParams = useSearchParams();
  const q = searchParams.get('q') ?? '';

  // 絶対にやってはいけない!
  // return <div dangerouslySetInnerHTML={{ __html: q }} />;

  // OK: React のテキストノードとして表示
  return <div>{q}</div>;
}
```

### 5.4 History API との連携

URL状態管理の裏側では History API が動作している。push と replace の使い分けを理解することが重要である。

```typescript
// === History API: push vs replace の使い分け ===

/**
 * push（履歴に追加）を使うべき場面:
 * - フィルタの変更
 * - カテゴリの変更
 * - ソートの変更
 * - 検索の確定（デバウンス後）
 * → ブラウザバックで前のフィルタ状態に戻れる
 */
function handleFilterChange(newCategory: string) {
  // React Router
  setSearchParams(prev => {
    const next = new URLSearchParams(prev);
    next.set('category', newCategory);
    return next;
  }); // デフォルトは push

  // Next.js
  router.push(`${pathname}?category=${newCategory}`);
}

/**
 * replace（履歴を置換）を使うべき場面:
 * - 検索入力中のリアルタイム更新（デバウンス中）
 * - ページネーションの連続クリック
 * - 並び替えの頻繁な切り替え
 * - 初期パラメータの正規化
 * → ブラウザバックで大量の中間状態に戻らない
 */
function handleSearchInput(query: string) {
  // React Router
  setSearchParams(prev => {
    const next = new URLSearchParams(prev);
    if (query) next.set('q', query);
    else next.delete('q');
    return next;
  }, { replace: true }); // 履歴を置換

  // Next.js
  router.replace(`${pathname}?q=${encodeURIComponent(query)}`);
}

/**
 * popstate イベント: ブラウザバック/フォワードの検知
 */
useEffect(() => {
  function handlePopState(event: PopStateEvent) {
    // ブラウザの戻る/進むが押されたときの処理
    // React Router や Next.js は自動的にこれを処理するが、
    // カスタムロジックが必要な場合に使用
    console.log('Navigation via browser back/forward');
    console.log('New URL:', window.location.href);
  }

  window.addEventListener('popstate', handlePopState);
  return () => window.removeEventListener('popstate', handlePopState);
}, []);
```

### 5.5 配列パラメータの設計パターン

複数選択のフィルタなど、配列値をURLに反映する方法は複数ある。それぞれの特徴を理解し、プロジェクトに適した方式を選択する。

```typescript
// === 配列パラメータの方式比較 ===

// 方式1: 同一キー反復（Web標準 / URLSearchParams 準拠）
// URL: ?tag=sale&tag=new&tag=popular
// 取得: searchParams.getAll('tag') → ['sale', 'new', 'popular']
// 利点: Web標準に準拠、URLSearchParams で自然に扱える
// 欠点: URLが長くなりやすい

// 方式2: カンマ区切り
// URL: ?tags=sale,new,popular
// 取得: searchParams.get('tags')?.split(',') ?? []
// 利点: URLがコンパクト
// 欠点: 値にカンマが含まれる場合にエスケープが必要

// 方式3: ブラケット表記（PHP/Ruby on Rails スタイル）
// URL: ?tags[]=sale&tags[]=new&tags[]=popular
// 取得: 手動パースが必要
// 利点: バックエンドフレームワークとの互換性
// 欠点: URLSearchParams で直接扱えない

// 方式4: JSON エンコード
// URL: ?tags=["sale","new","popular"]
// 取得: JSON.parse(searchParams.get('tags') ?? '[]')
// 利点: 複雑な構造も表現可能
// 欠点: URLが読みにくい、エンコードで長くなる

// 推奨: 方式1 または 方式2
// 簡単な配列 → 方式2（カンマ区切り）
// URLSearchParams との互換性重視 → 方式1（同一キー反復）

// 実装例: カンマ区切り方式のヘルパー
function useArrayParam(key: string): [string[], (values: string[]) => void] {
  const [searchParams, setSearchParams] = useSearchParams();

  const values = searchParams.get(key)?.split(',').filter(Boolean) ?? [];

  const setValues = useCallback((newValues: string[]) => {
    setSearchParams(prev => {
      const params = new URLSearchParams(prev);
      if (newValues.length > 0) {
        params.set(key, newValues.join(','));
      } else {
        params.delete(key);
      }
      params.delete('page'); // ページリセット
      return params;
    });
  }, [key, setSearchParams]);

  return [values, setValues];
}

// 使用例
function TagFilter() {
  const [selectedTags, setSelectedTags] = useArrayParam('tags');

  const allTags = ['sale', 'new', 'popular', 'limited', 'exclusive'];

  function toggleTag(tag: string) {
    if (selectedTags.includes(tag)) {
      setSelectedTags(selectedTags.filter(t => t !== tag));
    } else {
      setSelectedTags([...selectedTags, tag]);
    }
  }

  return (
    <div>
      {allTags.map(tag => (
        <button
          key={tag}
          onClick={() => toggleTag(tag)}
          className={selectedTags.includes(tag) ? 'active' : ''}
        >
          {tag}
        </button>
      ))}
    </div>
  );
}
```

---

## 6. デバウンス・スロットル・最適化

### 6.1 パフォーマンス最適化の基本原則

URL状態の変更は、以下の一連の処理を引き起こす:
1. URLの更新（History API）
2. React の再レンダリング（状態変更の検知）
3. データフェッチ（API呼び出し）
4. UIの更新（レンダリング結果の反映）

これらの処理が頻繁に発生するとパフォーマンス問題になるため、適切な最適化が必要である。

```typescript
// === パフォーマンス最適化: useDeferredValue ===
'use client';
import { useDeferredValue, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';

function ProductListPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get('q') ?? '';

  // URLの値を遅延させる（UIのブロッキングを防ぐ）
  const deferredQuery = useDeferredValue(query);

  // 遅延された値でフィルタリング
  const filteredProducts = useMemo(
    () => products.filter(p =>
      p.name.toLowerCase().includes(deferredQuery.toLowerCase())
    ),
    [products, deferredQuery]
  );

  // 遅延中はスタイルを変更して表示
  const isStale = query !== deferredQuery;

  return (
    <div className={isStale ? 'opacity-70 transition-opacity' : ''}>
      <ProductGrid products={filteredProducts} />
    </div>
  );
}
```

```typescript
// === パフォーマンス最適化: useTransition ===
'use client';
import { useTransition } from 'react';
import { useRouter, usePathname, useSearchParams } from 'next/navigation';

function OptimizedFilters() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [isPending, startTransition] = useTransition();

  function handleFilterChange(key: string, value: string | null) {
    const params = new URLSearchParams(searchParams.toString());

    if (value) {
      params.set(key, value);
    } else {
      params.delete(key);
    }
    params.delete('page');

    // startTransition: UIの応答性を維持しながらURL更新
    startTransition(() => {
      router.push(`${pathname}?${params.toString()}`, {
        scroll: false,
      });
    });
  }

  return (
    <div>
      {/* isPending を使ってローディング状態を表示 */}
      {isPending && (
        <div className="absolute inset-0 bg-white/50 flex items-center justify-center">
          <Spinner />
        </div>
      )}

      <select
        onChange={(e) => handleFilterChange('category', e.target.value || null)}
        disabled={isPending}
      >
        <option value="">All</option>
        <option value="electronics">Electronics</option>
        <option value="books">Books</option>
      </select>
    </div>
  );
}
```

### 6.2 URL状態の変更回数を制限する

```typescript
// === スロットル: URL更新の頻度を制限 ===

/**
 * スロットルフック: 一定間隔でのみ値を更新
 */
function useThrottle<T>(value: T, intervalMs: number): T {
  const [throttledValue, setThrottledValue] = useState(value);
  const lastUpdated = useRef(Date.now());

  useEffect(() => {
    const now = Date.now();
    const timeSinceLastUpdate = now - lastUpdated.current;

    if (timeSinceLastUpdate >= intervalMs) {
      setThrottledValue(value);
      lastUpdated.current = now;
    } else {
      const timer = setTimeout(() => {
        setThrottledValue(value);
        lastUpdated.current = Date.now();
      }, intervalMs - timeSinceLastUpdate);

      return () => clearTimeout(timer);
    }
  }, [value, intervalMs]);

  return throttledValue;
}

// 使用例: スライダーの値をURLに同期
function PriceRangeSlider() {
  const [localMin, setLocalMin] = useState(0);
  const [localMax, setLocalMax] = useState(10000);

  // 100msごとにしかURLを更新しない
  const throttledMin = useThrottle(localMin, 100);
  const throttledMax = useThrottle(localMax, 100);

  const { pushParams } = useURLParams();

  useEffect(() => {
    pushParams({
      price_min: throttledMin > 0 ? String(throttledMin) : null,
      price_max: throttledMax < 10000 ? String(throttledMax) : null,
    });
  }, [throttledMin, throttledMax]);

  return (
    <div>
      <input
        type="range"
        min={0}
        max={10000}
        value={localMin}
        onChange={(e) => setLocalMin(Number(e.target.value))}
      />
      <input
        type="range"
        min={0}
        max={10000}
        value={localMax}
        onChange={(e) => setLocalMax(Number(e.target.value))}
      />
      <p>{localMin} - {localMax}</p>
    </div>
  );
}
```

### 6.3 URL状態の差分検知と不要な更新の防止

```typescript
// === 不要なURL更新を防止する ===

/**
 * URL状態が実質的に変化したかを判定する
 */
function hasSearchParamsChanged(
  prev: URLSearchParams,
  next: URLSearchParams
): boolean {
  // 同じキーの数を比較
  const prevKeys = new Set(prev.keys());
  const nextKeys = new Set(next.keys());

  if (prevKeys.size !== nextKeys.size) return true;

  for (const key of prevKeys) {
    if (!nextKeys.has(key)) return true;

    const prevValues = prev.getAll(key).sort();
    const nextValues = next.getAll(key).sort();

    if (prevValues.length !== nextValues.length) return true;
    if (prevValues.some((v, i) => v !== nextValues[i])) return true;
  }

  return false;
}

/**
 * 不要な更新を防止するフック
 */
function useSafeSearchParams() {
  const [searchParams, setSearchParams] = useSearchParams();

  const safeSetSearchParams = useCallback(
    (
      updater: URLSearchParams | ((prev: URLSearchParams) => URLSearchParams),
      options?: { replace?: boolean }
    ) => {
      const nextParams = typeof updater === 'function'
        ? updater(new URLSearchParams(searchParams))
        : updater;

      // 実質的な変化がない場合は更新しない
      if (!hasSearchParamsChanged(searchParams, nextParams)) {
        return;
      }

      setSearchParams(nextParams, options);
    },
    [searchParams, setSearchParams]
  );

  return [searchParams, safeSetSearchParams] as const;
}
```

### 6.4 URLの正規化（Normalization）

同じ意味を持つ異なるURLを統一するために、URLの正規化を行う。

```typescript
// === URL正規化 ===

/**
 * URL パラメータを正規化する
 * - キーを辞書順にソート
 * - デフォルト値を除去
 * - 空の値を除去
 */
function normalizeSearchParams(
  params: URLSearchParams,
  defaults: Record<string, string> = {}
): URLSearchParams {
  const normalized = new URLSearchParams();

  // エントリを取得してソート
  const entries = Array.from(params.entries())
    .filter(([key, value]) => {
      // 空の値を除去
      if (!value) return false;
      // デフォルト値を除去
      if (defaults[key] === value) return false;
      return true;
    })
    .sort(([a], [b]) => a.localeCompare(b));

  for (const [key, value] of entries) {
    normalized.append(key, value);
  }

  return normalized;
}

// 使用例
const params = new URLSearchParams('page=1&sort=newest&q=laptop&category=');
const normalized = normalizeSearchParams(params, {
  page: '1',
  sort: 'newest',
});
normalized.toString(); // 'q=laptop'（デフォルト値と空値が除去された）

/**
 * URL正規化ミドルウェア（Next.js）
 */
// middleware.ts
import { NextRequest, NextResponse } from 'next/server';

export function middleware(request: NextRequest) {
  const url = request.nextUrl;

  // 正規化が必要かチェック
  const normalized = normalizeSearchParams(
    url.searchParams,
    { page: '1', sort: 'newest', per_page: '20' }
  );

  const currentSearch = url.searchParams.toString();
  const normalizedSearch = normalized.toString();

  // 正規化後のURLが異なる場合はリダイレクト
  if (currentSearch !== normalizedSearch) {
    url.search = normalizedSearch ? `?${normalizedSearch}` : '';
    return NextResponse.redirect(url, { status: 301 });
  }

  return NextResponse.next();
}

export const config = {
  matcher: '/products/:path*',
};
```

---

## 7. アンチパターンと注意点

### 7.1 よくあるアンチパターン

URL状態管理でよく見られるアンチパターンを理解し、回避することが重要である。

```typescript
// === アンチパターン集 ===

// ❌ アンチパターン1: URLに機密情報を含める
// URL: /dashboard?token=eyJhbGciOiJIUzI1NiJ9...
// → URLはブラウザ履歴、サーバーログ、リファラーヘッダーに残る
// → 認証情報は Cookie（HttpOnly）に保存すべき

// ❌ アンチパターン2: URLに大量のデータを詰め込む
// URL: /products?ids=1,2,3,4,5,...,1000
// → URLは2048文字が実質上限（ブラウザ・サーバーによる）
// → 大量のデータはサーバーサイドのセッションや localStorage に保存

// ❌ アンチパターン3: デフォルト値をURLに含める
// URL: /products?page=1&sort=newest&per_page=20&view=grid
// → URLが冗長になる
// → デフォルト値はコードで管理し、URLからは省略する

// ❌ アンチパターン4: フィルタ変更時にページをリセットしない
function BadFilterChange() {
  const [searchParams, setSearchParams] = useSearchParams();

  function handleCategoryChange(category: string) {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev);
      next.set('category', category);
      // ❌ page をリセットしていない！
      // page=5 のまま category を変更すると、
      // 新カテゴリで5ページ目が存在しない可能性がある
      return next;
    });
  }
}

// ✅ 正しいパターン
function GoodFilterChange() {
  const [searchParams, setSearchParams] = useSearchParams();

  function handleCategoryChange(category: string) {
    setSearchParams(prev => {
      const next = new URLSearchParams(prev);
      next.set('category', category);
      next.delete('page'); // ✅ ページをリセット
      return next;
    });
  }
}

// ❌ アンチパターン5: 型変換なしで使用
function BadTypeHandling() {
  const [searchParams] = useSearchParams();

  // ❌ string のまま比較・演算
  const page = searchParams.get('page'); // string | null
  if (page > 1) { /* 文字列比較になる！ '9' > '10' は true */ }
}

// ✅ 正しいパターン
function GoodTypeHandling() {
  const [searchParams] = useSearchParams();

  // ✅ 明示的な型変換
  const page = Number(searchParams.get('page') ?? '1');
  if (page > 1) { /* 数値比較 */ }
}

// ❌ アンチパターン6: 毎レンダリングで新しい URLSearchParams を生成
function BadPerformance() {
  const [searchParams, setSearchParams] = useSearchParams();

  // ❌ レンダリングのたびに new URLSearchParams が実行される
  // → 参照が毎回変わるため useEffect の無限ループを引き起こす可能性
  const params = new URLSearchParams(searchParams);
  const query = params.get('q');

  useEffect(() => {
    // params は毎回新しいオブジェクトなので無限ループ！
    fetchProducts(params);
  }, [params]); // ❌
}

// ✅ 正しいパターン
function GoodPerformance() {
  const [searchParams] = useSearchParams();

  // ✅ プリミティブ値を依存配列に使用
  const query = searchParams.get('q') ?? '';
  const page = Number(searchParams.get('page') ?? '1');

  useEffect(() => {
    fetchProducts({ q: query, page });
  }, [query, page]); // ✅ プリミティブ値なので安定
}

// ❌ アンチパターン7: URLパラメータの検証なし
function BadValidation() {
  const [searchParams] = useSearchParams();

  // ❌ ユーザーが ?page=-5 や ?page=abc を入力する可能性がある
  const page = Number(searchParams.get('page'));
  // NaN や負の数になる可能性

  // ❌ ソート値の検証なし
  const sort = searchParams.get('sort');
  // 任意の文字列が入る可能性（SQLインジェクションのリスク）
}

// ✅ 正しいパターン
function GoodValidation() {
  const [searchParams] = useSearchParams();

  // ✅ バリデーション付き
  const rawPage = Number(searchParams.get('page'));
  const page = Number.isInteger(rawPage) && rawPage > 0 ? rawPage : 1;

  // ✅ ホワイトリストで検証
  const validSorts = ['newest', 'oldest', 'price-asc', 'price-desc'] as const;
  const rawSort = searchParams.get('sort');
  const sort = validSorts.includes(rawSort as any)
    ? (rawSort as typeof validSorts[number])
    : 'newest';
}

// ❌ アンチパターン8: 検索入力でデバウンスなし
function BadSearchInput() {
  const router = useRouter();
  const pathname = usePathname();

  return (
    <input
      onChange={(e) => {
        // ❌ キー入力のたびにURLを更新
        // → ブラウザ履歴が大量に作られる
        // → 毎回ネットワークリクエストが発生
        router.push(`${pathname}?q=${e.target.value}`);
      }}
    />
  );
}

// ❌ アンチパターン9: searchParams を直接オブジェクトの比較に使う
function BadComparison() {
  const [searchParams] = useSearchParams();

  // ❌ URLSearchParams はオブジェクトなので参照比較
  useEffect(() => {
    // searchParams はレンダリングごとに新しいインスタンス
    // → 毎回実行される
  }, [searchParams]);

  // ✅ 文字列に変換して比較
  const searchString = searchParams.toString();
  useEffect(() => {
    // 文字列比較なので内容が同じなら実行されない
  }, [searchString]);
}
```

### 7.2 URL状態管理のセキュリティ考慮事項

```
URL状態のセキュリティチェックリスト:

□ URLパラメータに認証トークンを含めていないか
  → Cookie (HttpOnly, Secure, SameSite) を使用する

□ URLパラメータをSQLクエリに直接使用していないか
  → パラメータ化クエリを使用する
  → ホワイトリストでバリデーションする

□ URLパラメータをHTMLに直接出力していないか
  → React のテキストノードとして表示する（自動エスケープ）
  → dangerouslySetInnerHTML は使用しない

□ URLパラメータの長さを制限しているか
  → DoS攻撃対策として最大長を設定する

□ Open Redirect 脆弱性がないか
  → redirect パラメータは内部URLのみ許可する
  → 外部URLへのリダイレクトは禁止する

□ URLパラメータでサーバーリソースを操作していないか
  → 読み取り専用のパラメータのみURLに含める
  → 副作用のある操作はPOSTリクエストで行う

□ リファラーヘッダーに機密情報が漏れないか
  → Referrer-Policy ヘッダーを適切に設定する
  → meta tag: <meta name="referrer" content="origin">
```

```typescript
// === Open Redirect 防止 ===

// ❌ 危険: 外部URLにリダイレクトされる可能性
function DangerousRedirect() {
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get('redirect');

  // /login?redirect=https://evil.com にアクセスすると
  // ログイン後に evil.com にリダイレクトされる
  router.push(redirectTo!); // ❌
}

// ✅ 安全: 内部URLのみ許可
function SafeRedirect() {
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get('redirect');

  function isInternalUrl(url: string): boolean {
    // スラッシュで始まるパスのみ許可
    if (!url.startsWith('/')) return false;
    // プロトコル相対URLを拒否
    if (url.startsWith('//')) return false;
    // バックスラッシュを拒否（IE対策）
    if (url.includes('\\')) return false;
    return true;
  }

  const safeRedirect = redirectTo && isInternalUrl(redirectTo)
    ? redirectTo
    : '/'; // デフォルトのリダイレクト先

  router.push(safeRedirect); // ✅
}
```

---

## 8. テスト戦略

### 8.1 URL状態のユニットテスト

```typescript
// === URL状態のテスト ===
import { renderHook, act } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { useProductFilters } from './useProductFilters';

// React Router のテスト
describe('useProductFilters', () => {
  function wrapper({ children }: { children: React.ReactNode }) {
    return (
      <MemoryRouter initialEntries={['/products?q=laptop&page=2']}>
        {children}
      </MemoryRouter>
    );
  }

  it('URLからフィルタ値を正しく取得する', () => {
    const { result } = renderHook(() => useProductFilters(), { wrapper });

    expect(result.current.filters.query).toBe('laptop');
    expect(result.current.filters.page).toBe(2);
    expect(result.current.filters.sort).toBe('newest'); // デフォルト値
  });

  it('フィルタ変更時にページがリセットされる', () => {
    const { result } = renderHook(() => useProductFilters(), { wrapper });

    act(() => {
      result.current.setFilters({ category: 'electronics' });
    });

    expect(result.current.filters.category).toBe('electronics');
    expect(result.current.filters.page).toBe(1); // リセットされた
  });

  it('不正な値の場合はデフォルト値が使われる', () => {
    function invalidWrapper({ children }: { children: React.ReactNode }) {
      return (
        <MemoryRouter initialEntries={['/products?page=abc&sort=invalid']}>
          {children}
        </MemoryRouter>
      );
    }

    const { result } = renderHook(
      () => useProductFilters(),
      { wrapper: invalidWrapper }
    );

    expect(result.current.filters.page).toBe(1); // NaN → デフォルト
    expect(result.current.filters.sort).toBe('newest'); // 不正値 → デフォルト
  });

  it('リセットで全パラメータがクリアされる', () => {
    const { result } = renderHook(() => useProductFilters(), { wrapper });

    act(() => {
      result.current.resetFilters();
    });

    expect(result.current.filters.query).toBe('');
    expect(result.current.filters.page).toBe(1);
    expect(result.current.isFiltered).toBe(false);
  });
});
```

### 8.2 E2Eテスト（Playwright）

```typescript
// === Playwright: URL状態のE2Eテスト ===
import { test, expect } from '@playwright/test';

test.describe('Product Filters - URL State', () => {
  test('URLパラメータからフィルタが復元される', async ({ page }) => {
    // 特定のURL状態で直接アクセス
    await page.goto('/products?q=laptop&category=electronics&sort=price-asc');

    // フィルタUIに値が反映されていることを確認
    await expect(page.locator('input[type="search"]')).toHaveValue('laptop');
    await expect(page.locator('select[name="category"]')).toHaveValue('electronics');
    await expect(page.locator('select[name="sort"]')).toHaveValue('price-asc');
  });

  test('フィルタ変更でURLが更新される', async ({ page }) => {
    await page.goto('/products');

    // カテゴリを変更
    await page.selectOption('select[name="category"]', 'electronics');

    // URLが更新されたことを確認
    await expect(page).toHaveURL(/category=electronics/);

    // ページがリセットされたことを確認
    await expect(page).not.toHaveURL(/page=/);
  });

  test('ブラウザバックでフィルタ状態が復元される', async ({ page }) => {
    await page.goto('/products');

    // フィルタを変更
    await page.selectOption('select[name="category"]', 'electronics');
    await expect(page).toHaveURL(/category=electronics/);

    // さらにフィルタを変更
    await page.selectOption('select[name="sort"]', 'price-asc');
    await expect(page).toHaveURL(/sort=price-asc/);

    // ブラウザバック
    await page.goBack();

    // 前のフィルタ状態に戻ることを確認
    await expect(page).toHaveURL(/category=electronics/);
    await expect(page).not.toHaveURL(/sort=price-asc/);
  });

  test('検索入力がデバウンスされる', async ({ page }) => {
    await page.goto('/products');

    // 素早く入力
    await page.locator('input[type="search"]').fill('laptop');

    // デバウンス期間を待つ
    await page.waitForTimeout(500);

    // URLが最終値のみ反映されていることを確認
    await expect(page).toHaveURL(/q=laptop/);
  });

  test('共有URLで同じ状態が復元される', async ({ page, context }) => {
    // ユーザーAがフィルタを設定
    await page.goto('/products');
    await page.selectOption('select[name="category"]', 'electronics');
    await page.selectOption('select[name="sort"]', 'price-asc');

    // URLを取得
    const sharedUrl = page.url();

    // ユーザーBが同じURLにアクセス（新しいタブ）
    const newPage = await context.newPage();
    await newPage.goto(sharedUrl);

    // 同じフィルタ状態が復元されることを確認
    await expect(newPage.locator('select[name="category"]')).toHaveValue('electronics');
    await expect(newPage.locator('select[name="sort"]')).toHaveValue('price-asc');
  });

  test('不正なURLパラメータが安全に処理される', async ({ page }) => {
    // 不正な値を含むURLにアクセス
    await page.goto('/products?page=-1&sort=invalid&q=<script>alert(1)</script>');

    // ページがクラッシュしないことを確認
    await expect(page.locator('h1')).toBeVisible();

    // 不正値がデフォルトに修正されていることを確認
    // （実装による: リダイレクトまたはデフォルト値適用）
  });
});
```

### 8.3 URL状態のヘルパー関数テスト

```typescript
// === ヘルパー関数のテスト ===
import { describe, it, expect } from 'vitest';
import { normalizeSearchParams, searchParamsToObject } from './url-utils';

describe('normalizeSearchParams', () => {
  it('デフォルト値を除去する', () => {
    const params = new URLSearchParams('page=1&sort=newest&q=laptop');
    const normalized = normalizeSearchParams(params, {
      page: '1',
      sort: 'newest',
    });

    expect(normalized.toString()).toBe('q=laptop');
  });

  it('空の値を除去する', () => {
    const params = new URLSearchParams('q=&category=&sort=price');
    const normalized = normalizeSearchParams(params);

    expect(normalized.toString()).toBe('sort=price');
  });

  it('キーをソートする', () => {
    const params = new URLSearchParams('z=1&a=2&m=3');
    const normalized = normalizeSearchParams(params);

    expect(normalized.toString()).toBe('a=2&m=3&z=1');
  });

  it('空のパラメータを返す', () => {
    const params = new URLSearchParams('page=1&sort=newest');
    const normalized = normalizeSearchParams(params, {
      page: '1',
      sort: 'newest',
    });

    expect(normalized.toString()).toBe('');
  });
});

describe('searchParamsToObject', () => {
  it('単一値をstring として返す', () => {
    const params = new URLSearchParams('q=laptop&page=2');
    const obj = searchParamsToObject(params);

    expect(obj).toEqual({ q: 'laptop', page: '2' });
  });

  it('複数値を配列として返す', () => {
    const params = new URLSearchParams('tag=a&tag=b&tag=c');
    const obj = searchParamsToObject(params);

    expect(obj).toEqual({ tag: ['a', 'b', 'c'] });
  });

  it('混在する値を正しく処理する', () => {
    const params = new URLSearchParams('q=laptop&tag=a&tag=b&page=1');
    const obj = searchParamsToObject(params);

    expect(obj).toEqual({
      q: 'laptop',
      tag: ['a', 'b'],
      page: '1',
    });
  });
});
```

---

## 9. 実践的なケーススタディ

### 9.1 ECサイトの商品検索ページ

実際のECサイトを想定した、URL状態管理の完全な実装例を示す。

```typescript
// === 完全な実装例: ECサイト商品検索 ===

// 1. 共有パーサー定義（searchParams.ts）
import {
  parseAsString,
  parseAsInteger,
  parseAsFloat,
  parseAsBoolean,
  parseAsStringEnum,
  parseAsArrayOf,
  createSearchParamsCache,
} from 'nuqs';

export const productSearchParsers = {
  // 検索
  q: parseAsString.withDefault(''),

  // ページネーション
  page: parseAsInteger.withDefault(1),
  per_page: parseAsInteger.withDefault(24),

  // ソート
  sort: parseAsStringEnum([
    'relevance',
    'newest',
    'price-asc',
    'price-desc',
    'rating',
    'popular',
  ]).withDefault('relevance'),

  // フィルタ
  category: parseAsString,
  brand: parseAsString,
  tags: parseAsArrayOf(parseAsString, ',').withDefault([]),
  in_stock: parseAsBoolean.withDefault(false),
  price_min: parseAsFloat,
  price_max: parseAsFloat,
  rating_min: parseAsInteger,

  // 表示設定
  view: parseAsStringEnum(['grid', 'list']).withDefault('grid'),
};

// Server Component 用キャッシュ
export const productSearchParamsCache = createSearchParamsCache(
  productSearchParsers
);

// 型定義
export type ProductSearchParams = {
  [K in keyof typeof productSearchParsers]: ReturnType<
    (typeof productSearchParsers)[K]['parse']
  >;
};
```

```typescript
// 2. Server Component（page.tsx）
import { Suspense } from 'react';
import { productSearchParamsCache } from './searchParams';

export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[]>>;
}) {
  const filters = await productSearchParamsCache.parse(await searchParams);

  // サーバーサイドでデータフェッチ
  const [products, categories, brands] = await Promise.all([
    getProducts(filters),
    getCategories(),
    getBrands(filters.category),
  ]);

  return (
    <div className="flex gap-6">
      {/* サイドバー: フィルタ */}
      <aside className="w-64 shrink-0">
        <Suspense fallback={<FiltersSkeleton />}>
          <ProductFiltersSidebar
            categories={categories}
            brands={brands}
          />
        </Suspense>
      </aside>

      {/* メインコンテンツ */}
      <main className="flex-1">
        {/* 検索バー & ソート */}
        <Suspense fallback={<SearchBarSkeleton />}>
          <SearchAndSort totalCount={products.meta.total} />
        </Suspense>

        {/* アクティブフィルタ表示 */}
        <Suspense fallback={null}>
          <ActiveFilters />
        </Suspense>

        {/* 商品グリッド */}
        <ProductGrid
          products={products.data}
          view={filters.view ?? 'grid'}
        />

        {/* ページネーション */}
        <ServerPagination
          currentPage={filters.page}
          totalPages={products.meta.totalPages}
          searchParams={Object.fromEntries(
            Object.entries(filters).filter(([_, v]) => v != null)
          )}
        />
      </main>
    </div>
  );
}
```

```typescript
// 3. Client Component: アクティブフィルタ表示
'use client';
import { useQueryStates } from 'nuqs';
import { productSearchParsers } from './searchParams';

function ActiveFilters() {
  const [filters, setFilters] = useQueryStates(productSearchParsers);

  const activeFilters: { key: string; label: string; onRemove: () => void }[] = [];

  if (filters.q) {
    activeFilters.push({
      key: 'q',
      label: `Search: "${filters.q}"`,
      onRemove: () => setFilters({ q: '', page: 1 }),
    });
  }

  if (filters.category) {
    activeFilters.push({
      key: 'category',
      label: `Category: ${filters.category}`,
      onRemove: () => setFilters({ category: null, page: 1 }),
    });
  }

  if (filters.brand) {
    activeFilters.push({
      key: 'brand',
      label: `Brand: ${filters.brand}`,
      onRemove: () => setFilters({ brand: null, page: 1 }),
    });
  }

  filters.tags.forEach((tag, index) => {
    activeFilters.push({
      key: `tag-${index}`,
      label: `Tag: ${tag}`,
      onRemove: () => setFilters({
        tags: filters.tags.filter((_, i) => i !== index),
        page: 1,
      }),
    });
  });

  if (filters.in_stock) {
    activeFilters.push({
      key: 'in_stock',
      label: 'In Stock Only',
      onRemove: () => setFilters({ in_stock: false, page: 1 }),
    });
  }

  if (filters.price_min !== null || filters.price_max !== null) {
    const min = filters.price_min ?? 0;
    const max = filters.price_max ?? 'unlimited';
    activeFilters.push({
      key: 'price',
      label: `Price: $${min} - $${max}`,
      onRemove: () => setFilters({
        price_min: null,
        price_max: null,
        page: 1,
      }),
    });
  }

  if (activeFilters.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2 mb-4">
      {activeFilters.map((filter) => (
        <span
          key={filter.key}
          className="inline-flex items-center gap-1 px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm"
        >
          {filter.label}
          <button
            onClick={filter.onRemove}
            className="ml-1 hover:text-blue-600"
            aria-label={`Remove filter: ${filter.label}`}
          >
            x
          </button>
        </span>
      ))}

      <button
        onClick={() => setFilters({
          q: '',
          category: null,
          brand: null,
          tags: [],
          in_stock: false,
          price_min: null,
          price_max: null,
          rating_min: null,
          page: 1,
          sort: 'relevance',
        })}
        className="text-sm text-gray-500 hover:text-gray-700 underline"
      >
        Clear All
      </button>
    </div>
  );
}
```

### 9.2 ダッシュボードのフィルタパネル

```typescript
// === ダッシュボード: 日付範囲フィルタ ===
'use client';
import { useQueryStates, parseAsString, parseAsStringEnum } from 'nuqs';
import { startOfDay, endOfDay, subDays, format } from 'date-fns';

const dashboardParsers = {
  from: parseAsString,
  to: parseAsString,
  preset: parseAsStringEnum([
    'today',
    '7d',
    '30d',
    '90d',
    'year',
    'custom',
  ]).withDefault('30d'),
  metric: parseAsStringEnum([
    'revenue',
    'orders',
    'visitors',
    'conversion',
  ]).withDefault('revenue'),
  granularity: parseAsStringEnum([
    'hour',
    'day',
    'week',
    'month',
  ]).withDefault('day'),
};

function DashboardFilters() {
  const [filters, setFilters] = useQueryStates(dashboardParsers);

  // プリセットから日付範囲を計算
  const dateRange = useMemo(() => {
    const now = new Date();

    switch (filters.preset) {
      case 'today':
        return { from: startOfDay(now), to: endOfDay(now) };
      case '7d':
        return { from: subDays(now, 7), to: now };
      case '30d':
        return { from: subDays(now, 30), to: now };
      case '90d':
        return { from: subDays(now, 90), to: now };
      case 'year':
        return { from: subDays(now, 365), to: now };
      case 'custom':
        return {
          from: filters.from ? new Date(filters.from) : subDays(now, 30),
          to: filters.to ? new Date(filters.to) : now,
        };
      default:
        return { from: subDays(now, 30), to: now };
    }
  }, [filters.preset, filters.from, filters.to]);

  function handlePresetChange(preset: string) {
    if (preset === 'custom') {
      setFilters({
        preset: 'custom',
        from: format(dateRange.from, 'yyyy-MM-dd'),
        to: format(dateRange.to, 'yyyy-MM-dd'),
      });
    } else {
      setFilters({
        preset: preset as any,
        from: null,
        to: null,
      });
    }
  }

  return (
    <div className="flex items-center gap-4">
      {/* プリセット選択 */}
      <select
        value={filters.preset}
        onChange={(e) => handlePresetChange(e.target.value)}
      >
        <option value="today">Today</option>
        <option value="7d">Last 7 Days</option>
        <option value="30d">Last 30 Days</option>
        <option value="90d">Last 90 Days</option>
        <option value="year">Last Year</option>
        <option value="custom">Custom Range</option>
      </select>

      {/* カスタム日付範囲（preset=custom の場合のみ表示） */}
      {filters.preset === 'custom' && (
        <>
          <input
            type="date"
            value={filters.from ?? ''}
            onChange={(e) => setFilters({ from: e.target.value || null })}
          />
          <span>to</span>
          <input
            type="date"
            value={filters.to ?? ''}
            onChange={(e) => setFilters({ to: e.target.value || null })}
          />
        </>
      )}

      {/* メトリクス選択 */}
      <select
        value={filters.metric}
        onChange={(e) => setFilters({ metric: e.target.value as any })}
      >
        <option value="revenue">Revenue</option>
        <option value="orders">Orders</option>
        <option value="visitors">Visitors</option>
        <option value="conversion">Conversion Rate</option>
      </select>

      {/* 粒度選択 */}
      <select
        value={filters.granularity}
        onChange={(e) => setFilters({ granularity: e.target.value as any })}
      >
        <option value="hour">Hourly</option>
        <option value="day">Daily</option>
        <option value="week">Weekly</option>
        <option value="month">Monthly</option>
      </select>
    </div>
  );
}
```

---

## 10. URL状態管理ツール比較

### 10.1 ツール選定ガイド

| ツール | 型安全 | フレームワーク | バッチ更新 | SSR対応 | 学習コスト | 推奨場面 |
|--------|--------|----------------|-----------|---------|----------|---------|
| URLSearchParams | 低 | なし（Web標準） | 手動 | 可 | 低 | シンプルなケース |
| useSearchParams (RR) | 低 | React Router | 手動 | なし | 低 | React Router プロジェクト |
| useSearchParams (Next) | 低 | Next.js | 手動 | 可 | 低 | Next.js プロジェクト |
| nuqs | 高 | Next/RR/Remix | 自動 | 可 | 中 | 型安全が必要な場合 |
| qs ライブラリ | 低 | なし | 手動 | 可 | 低 | ネストオブジェクトのシリアライズ |
| カスタムフック | 中 | 任意 | 手動 | 依存 | 高 | 特殊要件がある場合 |

### 10.2 プロジェクト規模別の推奨

```
小規模プロジェクト（1-3ページのフィルタ）:
  → useSearchParams + カスタムヘルパー関数
  → 追加ライブラリ不要
  → シンプルで理解しやすい

中規模プロジェクト（5-10ページのフィルタ）:
  → nuqs を導入
  → 型安全性とバッチ更新の恩恵が大きい
  → パーサー定義を共有して一貫性を保つ

大規模プロジェクト（10+ページ、複雑なフィルタ）:
  → nuqs + Zod バリデーション
  → Server Component 統合（searchParamsCache）
  → URL正規化ミドルウェア
  → E2Eテストで URL状態の回帰テスト
```

---

## 11. トラブルシューティング

### 11.1 よくある問題と解決策

```
問題1: useSearchParams が Suspense boundary を要求する（Next.js）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: Next.js App Router では useSearchParams がクライアントのみの値
解決: <Suspense> で囲む、または fallback を提供する
  <Suspense fallback={<Loading />}>
    <ComponentWithSearchParams />
  </Suspense>

問題2: URLパラメータが消える / リセットされる
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: setSearchParams でオブジェクトを渡すと全パラメータが置換される
解決: 関数型アップデートを使用する
  setSearchParams(prev => {
    const next = new URLSearchParams(prev);
    next.set('key', 'value');
    return next;
  });

問題3: useEffect の無限ループ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: searchParams オブジェクトが毎レンダリングで新規生成される
解決: 依存配列にはプリミティブ値を使用する
  const query = searchParams.get('q') ?? '';
  useEffect(() => { ... }, [query]); // string

問題4: ブラウザバックが効かない
━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: router.replace を使っている（履歴に追加されない）
解決: フィルタ変更には router.push を使用する
  replace → 検索入力中のリアルタイム更新のみ
  push → フィルタ確定時

問題5: 日本語が文字化けする
━━━━━━━━━━━━━━━━━━━━━━━━━
原因: 手動で encodeURIComponent / decodeURIComponent を使っている
解決: URLSearchParams を使用する（自動エンコード/デコード）
  params.set('q', '日本語'); // 自動エンコード
  params.get('q'); // '日本語'（自動デコード）

問題6: パラメータの順序が毎回変わる
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: URLSearchParams は挿入順
解決: params.sort() で辞書順にソートする
  → キャッシュヒット率の向上にも効果的

問題7: nuqs でURLが更新されない
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: NuqsAdapter の設定漏れ
解決: layout.tsx で NuqsAdapter を設定する
  import { NuqsAdapter } from 'nuqs/adapters/next/app';

問題8: Server Component で searchParams が undefined
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: Next.js 15+ では searchParams が Promise に変更された
解決: await で解決する
  const params = await searchParams; // Next.js 15+

問題9: 同じフィルタなのに毎回APIが呼ばれる
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: queryKey にオブジェクト参照を使っている
解決: プリミティブ値 or JSON.stringify を queryKey に使用
  queryKey: ['products', query, page, sort] // OK
  queryKey: ['products', JSON.stringify(filters)] // OK
  queryKey: ['products', filtersObject] // NG（参照比較）

問題10: iOS Safari でURLが更新されない
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
原因: iOS Safari の History API 制限（100回/30秒）
解決: スロットルを入れてURL更新頻度を制限する
  → 特にスライダーやリアルタイム入力で注意
```

### 11.2 デバッグテクニック

```typescript
// === URL状態のデバッグ ===

// 1. 現在のURL状態をログ出力
function useDebugSearchParams() {
  const searchParams = useSearchParams();

  useEffect(() => {
    if (process.env.NODE_ENV === 'development') {
      console.group('URL State');
      console.log('URL:', window.location.href);
      console.log('Search:', searchParams.toString());
      console.log('Params:', Object.fromEntries(searchParams.entries()));
      console.groupEnd();
    }
  }, [searchParams]);
}

// 2. URL変更の追跡
function useTrackURLChanges() {
  const searchParams = useSearchParams();
  const prevRef = useRef(searchParams.toString());

  useEffect(() => {
    const current = searchParams.toString();
    if (prevRef.current !== current) {
      console.log('URL changed:');
      console.log('  Before:', prevRef.current);
      console.log('  After:', current);
      prevRef.current = current;
    }
  }, [searchParams]);
}

// 3. React DevTools で URL状態を確認
// nuqs を使用している場合、React DevTools の
// コンポーネントツリーで各パラメータの値が確認できる

// 4. ブラウザの開発者ツールで History を確認
// Performance タブ → Navigation タイミングを確認
// Application タブ → History API の状態を確認
```

---

## まとめ

### URL状態管理の選択マトリクス

| 判断基準 | URLSearchParams | useSearchParams | nuqs |
|---------|----------------|----------------|------|
| 型安全性が必要 | - | - | Best |
| Next.js App Router | OK | OK | Best |
| React Router | OK | Best | OK |
| Server Component 統合 | 手動 | 手動 | Best |
| 複雑なフィルタ（10+パラメータ） | 手動 | 手動 | Best |
| 追加ライブラリを避けたい | Best | OK | - |
| バッチ更新が必要 | 手動 | 手動 | Best |
| カスタムパーサーが必要 | 手動 | 手動 | Best |

### URL状態設計のチェックリスト

```
設計時:
□ URLに含めるべき状態を特定した
□ デフォルト値をURLから除外する設計にした
□ 配列パラメータの形式を統一した
□ パラメータ名の命名規則を決めた（snake_case / camelCase）
□ URL長の上限を考慮した

実装時:
□ 型安全なパーサーを使用している（nuqs or Zod）
□ フィルタ変更時にページをリセットしている
□ 検索入力にデバウンスを適用している
□ push / replace を適切に使い分けている
□ XSSやOpen Redirectの対策をしている
□ バリデーションを実装している

テスト時:
□ URLからの状態復元テスト
□ ブラウザバック/フォワードテスト
□ 不正なURLパラメータの処理テスト
□ 共有URLの動作テスト
□ E2Eテストでフィルタ操作を検証
```

---

## 次に読むべきガイド
→ [[00-client-side-routing.md]] -- クライアントルーティング
→ [[01-component-state.md]] -- コンポーネント状態管理
→ [[02-global-state.md]] -- グローバル状態管理

---

## 参考文献
1. nuqs. "Type-safe search params state manager for Next.js." github.com/47ng/nuqs, 2024.
2. Next.js. "useSearchParams." nextjs.org/docs/app/api-reference/functions/use-search-params, 2024.
3. React Router. "useSearchParams." reactrouter.com/en/main/hooks/use-search-params, 2024.
4. MDN Web Docs. "URLSearchParams." developer.mozilla.org/en-US/docs/Web/API/URLSearchParams, 2024.
5. Lee Robinson. "Search Params in Next.js." leerob.io, 2024.
6. Web.dev. "URL pattern API." web.dev/articles/urlpattern, 2024.
7. OWASP. "Unvalidated Redirects and Forwards." owasp.org, 2024.
8. Kent C. Dodds. "URL State Management in React." epicreact.dev, 2024.
