# URL状態

> URL状態はWebアプリの「共有可能な状態」。検索クエリ、フィルタ、ページ番号、ソート順をURLに反映することで、ブックマーク・共有・ブラウザバックが自然に動作するUXを実現する。

## この章で学ぶこと

- [ ] URL状態の重要性と設計原則を理解する
- [ ] useSearchParamsとnuqsの使い方を把握する
- [ ] URLとアプリ状態の同期パターンを学ぶ

---

## 1. URL状態の重要性

```
URL に状態を反映すべき場面:
  ✓ 検索クエリ: /products?q=laptop
  ✓ フィルタ: /products?category=electronics&brand=apple
  ✓ ソート: /products?sort=price&order=asc
  ✓ ページネーション: /products?page=3
  ✓ タブ/ビュー: /dashboard?view=chart
  ✓ モーダルの状態: /products?modal=create

URL状態の利点:
  ✓ ブックマーク可能（同じURLで同じ状態を復元）
  ✓ 共有可能（URLを送れば同じ画面）
  ✓ ブラウザバック/フォワードが自然に動作
  ✓ SEO（クローラーがパラメータ付きURLを解析）
  ✓ サーバーサイドレンダリングと相性が良い

URL に反映すべきでない状態:
  ✗ 一時的なUI状態（ツールチップ、ホバー）
  ✗ フォームの入力途中値
  ✗ 認証トークン
  ✗ 大量のデータ（URLは2048文字制限）
```

---

## 2. useSearchParams（React Router / Next.js）

```typescript
// Next.js App Router
'use client';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';

function ProductFilters() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const category = searchParams.get('category') ?? 'all';
  const sort = searchParams.get('sort') ?? 'newest';
  const page = Number(searchParams.get('page') ?? '1');

  function updateParams(updates: Record<string, string | null>) {
    const params = new URLSearchParams(searchParams.toString());

    for (const [key, value] of Object.entries(updates)) {
      if (value === null) {
        params.delete(key);
      } else {
        params.set(key, value);
      }
    }

    // ページリセット（フィルタ変更時）
    if (!('page' in updates)) {
      params.delete('page');
    }

    router.push(`${pathname}?${params.toString()}`);
  }

  return (
    <div>
      <select
        value={category}
        onChange={(e) => updateParams({ category: e.target.value })}
      >
        <option value="all">All</option>
        <option value="electronics">Electronics</option>
        <option value="books">Books</option>
      </select>

      <select
        value={sort}
        onChange={(e) => updateParams({ sort: e.target.value })}
      >
        <option value="newest">Newest</option>
        <option value="price-asc">Price: Low to High</option>
        <option value="price-desc">Price: High to Low</option>
      </select>
    </div>
  );
}
```

---

## 3. nuqs（型安全なURL状態管理）

```typescript
// nuqs: Next.js 向けの型安全な search params 管理
import { useQueryState, parseAsInteger, parseAsStringEnum } from 'nuqs';

function ProductPage() {
  // 型安全なURL状態
  const [query, setQuery] = useQueryState('q', { defaultValue: '' });
  const [page, setPage] = useQueryState('page', parseAsInteger.withDefault(1));
  const [sort, setSort] = useQueryState('sort',
    parseAsStringEnum(['newest', 'price-asc', 'price-desc']).withDefault('newest')
  );
  const [category, setCategory] = useQueryState('category');

  // URLが自動的に更新される
  // /products?q=laptop&page=2&sort=price-asc&category=electronics

  return (
    <div>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value || null)} // 空文字はURL から削除
        placeholder="Search..."
      />

      <select value={sort} onChange={(e) => setSort(e.target.value as any)}>
        <option value="newest">Newest</option>
        <option value="price-asc">Price ↑</option>
        <option value="price-desc">Price ↓</option>
      </select>

      <button onClick={() => setPage(page + 1)}>Next Page</button>
    </div>
  );
}

// nuqs + TanStack Query の組み合わせ
function useProducts() {
  const [query] = useQueryState('q');
  const [page] = useQueryState('page', parseAsInteger.withDefault(1));
  const [sort] = useQueryState('sort', parseAsStringEnum(['newest', 'price-asc', 'price-desc']).withDefault('newest'));

  return useQuery({
    queryKey: ['products', { query, page, sort }],
    queryFn: () => api.products.list({ q: query, page, sort }),
  });
}
```

---

## 4. Server Component との統合

```typescript
// Next.js: Server Component で searchParams を受け取る
// app/products/page.tsx

interface SearchParams {
  q?: string;
  page?: string;
  sort?: string;
  category?: string;
}

export default async function ProductsPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const params = await searchParams;
  const page = Number(params.page ?? '1');
  const sort = params.sort ?? 'newest';

  const products = await getProducts({
    query: params.q,
    page,
    sort,
    category: params.category,
  });

  return (
    <div>
      <ProductFilters />       {/* Client Component */}
      <ProductGrid products={products.data} />  {/* Server Component */}
      <Pagination
        currentPage={page}
        totalPages={products.meta.totalPages}
      />
    </div>
  );
}
```

---

## 5. 設計パターン

```
URL状態の設計原則:

  ① デフォルト値はURL に含めない:
     /products              ← デフォルト（page=1, sort=newest）
     /products?sort=price   ← ソートだけ変更
     → URLがシンプルになる

  ② フィルタ変更時にページをリセット:
     /products?category=books&page=3
     → category変更 → /products?category=electronics（page=1にリセット）

  ③ 配列パラメータ:
     /products?tag=sale&tag=new     ← 同じキーを複数
     /products?tags=sale,new         ← カンマ区切り

  ④ デバウンス（検索入力）:
     → 入力ごとにURLを更新しない
     → 300-500ms のデバウンスを入れる

  ⑤ shallow routing:
     → URLを更新してもページ全体を再レンダリングしない
     → Next.js: router.push(url, { scroll: false })
```

---

## まとめ

| ツール | 特徴 |
|--------|------|
| useSearchParams | React標準、手動パース |
| nuqs | 型安全、パーサー組み込み |
| URLSearchParams | Web標準API |

---

## 次に読むべきガイド
→ [[00-client-side-routing.md]] — クライアントルーティング

---

## 参考文献
1. nuqs. "Type-safe search params." github.com/47ng/nuqs, 2024.
2. Next.js. "useSearchParams." nextjs.org/docs, 2024.
3. Lee Robinson. "Search Params in Next.js." leerob.io, 2024.
