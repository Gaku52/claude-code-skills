# SPA / MPA / SSR

> Webアプリのレンダリング方式は性能とUXを決定づける。SPA、MPA、SSR、SSG、ISR、Streaming SSR、React Server Componentsの特徴と選定基準を理解し、プロジェクト要件に最適なアーキテクチャを選択する。

## 前提知識

この章を学ぶ前に、以下の知識を習得しておくことを推奨する。

- HTTPの基礎（リクエスト/レスポンス、ステータスコード、キャッシュヘッダー）
  - 参照: `../../network-fundamentals/docs/02-http/00-http-basics.md`
- ブラウザのレンダリングパイプライン（DOM構築、CSSOM、レイアウト、ペイント）
  - 参照: `../../browser-and-web-platform/docs/01-rendering/00-rendering-pipeline.md`
- HTML/CSS/JavaScriptの基礎（DOM操作、イベント処理、非同期処理）

## この章で学ぶこと

- [ ] 各レンダリング方式の仕組みと特徴を理解する
- [ ] パフォーマンスとSEOの観点から選定基準を把握する
- [ ] ハイブリッドレンダリングの設計を学ぶ
- [ ] Hydration の仕組みと最適化手法を理解する
- [ ] React Server Components とStreaming SSR の実践を学ぶ
- [ ] Islands Architecture とPartial Hydrationを把握する

## 前提知識

この章を学習する前に、以下の知識を習得しておくことを推奨します。

- **HTTPの基礎**: リクエスト/レスポンスモデル、ステータスコード、ヘッダーの理解
  → 参照: `../../network-fundamentals/docs/02-http/00-http-basics.md`
- **ブラウザのレンダリング**: Critical Rendering Path、Paint、Layout の仕組み
  → 参照: `../../browser-and-web-platform/docs/01-rendering/00-rendering-pipeline.md`
- **HTML/CSS/JavaScriptの基礎**: DOM操作、イベント処理、非同期処理（Promise/async-await）の理解

---

## 1. レンダリング方式の全体像

### 1.1 方式の比較

```
方式の比較:

         初期表示  操作性  SEO   サーバー負荷  複雑度  JSバンドル
─────────────────────────────────────────────────────────────
CSR/SPA   遅い     最高    悪い   低い         低い    大きい
MPA       速い     低い    良い   中程度       低い    最小
SSR       速い     高い    良い   高い         中程度  大きい
SSG       最速     高い    最良   最低         低い    中程度
ISR       速い     高い    良い   低い         中程度  中程度
Streaming 速い     高い    良い   中程度       高い    中程度
RSC       速い     高い    良い   中程度       高い    小さい
Islands   速い     中程度  良い   低い         中程度  最小

レンダリングのタイミング:
  CSR:       クライアント（ブラウザ）でレンダリング
  MPA:       リクエスト時にサーバーでHTML全体を返す（従来型）
  SSR:       リクエスト時にサーバーでレンダリング + Hydration
  SSG:       ビルド時にサーバーでレンダリング
  ISR:       初回リクエスト時 + 定期的に再生成
  Streaming: サーバーで段階的にレンダリング
  RSC:       コンポーネント単位でサーバー/クライアント分離
  Islands:   ページの一部だけをインタラクティブ化
```

### 1.2 レンダリング方式の歴史的変遷

```
Webレンダリングの進化:

  2000年代初頭: 伝統的MPA
  → PHP, JSP, Ruby on Rails
  → サーバーが全HTMLを生成
  → ページ遷移のたびに全画面リロード

  2010年代前半: SPA の台頭
  → Backbone.js, AngularJS, React
  → クライアントサイドルーティング
  → リッチなインタラクション

  2010年代後半: SSR + SPA（Universal/Isomorphic）
  → Next.js, Nuxt.js
  → サーバーで初期HTML生成 + クライアントでHydration
  → SEO + インタラクション両立

  2020年代前半: SSG + ISR
  → Gatsby, Next.js SSG/ISR
  → ビルド時に静的HTML生成
  → CDN配信で最高速度

  2020年代中盤: RSC + Streaming + Islands
  → React Server Components
  → Streaming SSR with Suspense
  → Astro (Islands Architecture)
  → JSバンドルの最小化

  現在のトレンド:
  → ハイブリッドアプローチ（ページ単位で最適方式を選択）
  → Server-first（デフォルトはサーバー、必要時のみクライアント）
  → Progressive Enhancement（JSなしでも基本機能動作）
```

---

## 2. CSR / SPA（Client Side Rendering / Single Page Application）

### 2.1 SPA の仕組み

```
SPA（Single Page Application）:
  → ブラウザがJSを実行してHTMLを生成
  → ページ遷移はクライアントサイドルーティング
  → サーバーは空のHTMLとJSバンドルのみ配信

  フロー:
  1. ブラウザ: GET /
  2. サーバー: 空の HTML + JS バンドルを返す
  3. ブラウザ: JS を実行 → DOM を構築 → 画面表示
  4. ブラウザ: API コール → データ取得 → 画面更新

  初回リクエスト時のHTML:
  <html>
    <head>
      <title>App</title>
      <link rel="stylesheet" href="/assets/styles.a1b2c3.css">
    </head>
    <body>
      <div id="root"></div>     ← 空のHTML
      <script src="/assets/app.d4e5f6.js"></script>  ← JSが全てを描画
    </body>
  </html>

  ページ遷移（/products → /products/123）:
  → URLの変更（History API）
  → 新しいJSコンポーネントの読み込み
  → API呼び出し
  → DOMの部分更新
  → サーバーへのHTMLリクエストなし
```

### 2.2 SPA の利点と欠点

```
利点:
  ✓ ページ遷移が高速（サーバーリクエストなし）
  ✓ リッチなインタラクション（アニメーション、トランジション）
  ✓ サーバー負荷が低い（静的ファイル配信のみ）
  ✓ オフライン対応が容易（PWA、Service Worker）
  ✓ バックエンドとフロントエンドの完全分離
  ✓ モバイルアプリとAPIを共有可能
  ✓ デプロイが簡単（S3 + CloudFront等）

欠点:
  ✗ 初期表示が遅い（JSバンドルのダウンロード + パース + 実行）
  ✗ SEO が困難（クローラーがJS実行しない場合がある）
  ✗ FCP / LCPが遅い（JSが実行されるまで白画面）
  ✗ JSが無効だと何も表示されない
  ✗ メモリリークのリスク（ページ遷移でもメモリが解放されない）
  ✗ バンドルサイズの管理が必要
  ✗ ソーシャルメディアのOGP取得に工夫が必要

適用:
  → 管理画面、ダッシュボード
  → ログイン後のアプリケーション
  → SEO不要なツール系アプリ
  → メールクライアント、チャットアプリ
  → デザインツール（Figma等）
  → コードエディタ（VS Code Web）

フレームワーク:
  → React（Vite）
  → Vue（Vite）
  → Angular
  → Svelte（SvelteKit CSR mode）
```

### 2.3 SPA の実装例

```typescript
// Vite + React での SPA 構成

// main.tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import App from './App';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      retry: 3,
    },
  },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);

// App.tsx - ルーティング
import { Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { AuthProvider, RequireAuth } from './features/auth';
import { AppLayout } from './shared/layouts/AppLayout';
import { PageSkeleton } from './shared/components/PageSkeleton';

// コード分割: 各ページを遅延ロード
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Users = lazy(() => import('./pages/Users'));
const UserDetail = lazy(() => import('./pages/UserDetail'));
const Settings = lazy(() => import('./pages/Settings'));
const Login = lazy(() => import('./pages/Login'));

function App() {
  return (
    <AuthProvider>
      <Suspense fallback={<PageSkeleton />}>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<RequireAuth><AppLayout /></RequireAuth>}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/users" element={<Users />} />
            <Route path="/users/:id" element={<UserDetail />} />
            <Route path="/settings" element={<Settings />} />
          </Route>
        </Routes>
      </Suspense>
    </AuthProvider>
  );
}
```

```typescript
// SPA でのデータフェッチング（TanStack Query）
// pages/Users.tsx
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';

function Users() {
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['users', { page, search }],
    queryFn: () =>
      fetch(`/api/users?page=${page}&search=${search}`).then(r => r.json()),
    staleTime: 30 * 1000,
  });

  if (isLoading) return <UserListSkeleton />;
  if (error) return <ErrorDisplay error={error} />;

  return (
    <div>
      <h1>ユーザー一覧</h1>
      <SearchInput value={search} onChange={setSearch} />
      <UserTable users={data.users} />
      <Pagination
        currentPage={page}
        totalPages={data.totalPages}
        onPageChange={setPage}
      />
    </div>
  );
}
```

### 2.4 SPA のSEO対策

```
SPA のSEO問題と対策:

  問題:
  → Googlebot は JS を実行できるが、遅延がある
  → 他のクローラー（Bing, Twitter等）は JS を実行しない場合がある
  → 動的メタタグが反映されない
  → OGP画像が取得できない

  対策1: SSR / SSG への移行（推奨）
  → Next.js, Nuxt.js でSSRする
  → SEO必要なページのみSSR

  対策2: プリレンダリング
  → Prerender.io, Rendertron
  → クローラーのUser-Agentを検出
  → 事前レンダリングしたHTMLを返す

  対策3: react-helmet / @tanstack/react-head
  → 動的な <title>, <meta> タグの管理
  → ただしCSR単体ではクローラーに反映されない場合がある
```

```typescript
// react-helmet-async でのメタタグ管理
import { Helmet } from 'react-helmet-async';

function ProductPage({ product }: { product: Product }) {
  return (
    <>
      <Helmet>
        <title>{product.name} | MyStore</title>
        <meta name="description" content={product.description} />
        <meta property="og:title" content={product.name} />
        <meta property="og:description" content={product.description} />
        <meta property="og:image" content={product.imageUrl} />
        <meta property="og:type" content="product" />
        <link rel="canonical" href={`https://mystore.com/products/${product.slug}`} />
      </Helmet>

      <div>
        <h1>{product.name}</h1>
        {/* ... */}
      </div>
    </>
  );
}
```

---

## 3. MPA（Multi Page Application）

### 3.1 伝統的MPAの仕組み

```
MPA（Multi Page Application）:
  → 各URLに対してサーバーが完全なHTMLを生成
  → ページ遷移のたびに全画面リロード
  → サーバーサイドテンプレートエンジンで描画

  フロー:
  1. ブラウザ: GET /products
  2. サーバー: テンプレート + データ → HTML生成
  3. ブラウザ: HTML を受信 → 即座に表示
  4. ユーザー: リンクをクリック
  5. ブラウザ: GET /products/123
  6. サーバー: テンプレート + データ → 新しいHTML生成
  7. ブラウザ: 全画面リロード → HTML表示

  利点:
  ✓ 初期表示が速い（サーバーでHTML生成済み）
  ✓ SEO に最適（完全なHTMLが返る）
  ✓ シンプル（JSフレームワーク不要）
  ✓ JSが無効でも動作
  ✓ メモリリークの心配なし（ページ遷移で全て破棄）

  欠点:
  ✗ ページ遷移が遅い（全画面リロード）
  ✗ リッチなインタラクションが困難
  ✗ 状態の維持が困難（ページ遷移で失われる）
  ✗ サーバーとフロントエンドが密結合

  適用:
  → ブログ、ニュースサイト
  → ドキュメントサイト
  → EC（カタログページ）
  → コーポレートサイト

  フレームワーク:
  → Rails + ERB/Slim
  → Django + Jinja2
  → Laravel + Blade
  → Spring Boot + Thymeleaf
  → Express + EJS/Pug
```

### 3.2 モダンMPA（htmx + View Transitions）

```html
<!-- htmx: MPAにSPAライクな動作を追加 -->
<!-- ページ全体のリロードなしで部分更新 -->

<!-- 基本的なhtmx使用例 -->
<div id="user-list">
  <!-- ユーザー検索: 入力のたびにサーバーにリクエスト -->
  <input
    type="search"
    name="search"
    hx-get="/api/users/search"
    hx-trigger="input changed delay:300ms"
    hx-target="#user-results"
    hx-indicator="#search-spinner"
    placeholder="ユーザーを検索..."
  >
  <span id="search-spinner" class="htmx-indicator">🔍</span>

  <div id="user-results">
    <!-- サーバーから返されるHTMLフラグメントで置換 -->
  </div>
</div>

<!-- 無限スクロール -->
<div id="posts">
  <article>Post 1</article>
  <article>Post 2</article>
  <!-- 最後の要素が表示されたら次のページを取得 -->
  <div hx-get="/api/posts?page=2"
       hx-trigger="revealed"
       hx-swap="afterend"
       hx-select="article">
    Loading...
  </div>
</div>

<!-- View Transitions API（MPA でもスムーズな遷移） -->
<style>
  /* ページ遷移のアニメーション */
  @view-transition {
    navigation: auto;
  }

  ::view-transition-old(root) {
    animation: 0.3s ease-out fade-out;
  }

  ::view-transition-new(root) {
    animation: 0.3s ease-in fade-in;
  }

  @keyframes fade-out {
    from { opacity: 1; }
    to { opacity: 0; }
  }

  @keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
  }
</style>
```

```typescript
// Express + htmx のサーバー実装
import express from 'express';

const app = express();

// ユーザー検索 API（HTMLフラグメントを返す）
app.get('/api/users/search', async (req, res) => {
  const { search } = req.query;

  const users = await prisma.user.findMany({
    where: {
      OR: [
        { name: { contains: search as string, mode: 'insensitive' } },
        { email: { contains: search as string, mode: 'insensitive' } },
      ],
    },
    take: 20,
  });

  // HTMLフラグメントを返す（JSON ではなく）
  const html = users.map(user => `
    <div class="user-card" id="user-${user.id}">
      <h3>${user.name}</h3>
      <p>${user.email}</p>
      <button
        hx-delete="/api/users/${user.id}"
        hx-target="#user-${user.id}"
        hx-swap="outerHTML"
        hx-confirm="本当に削除しますか？"
      >削除</button>
    </div>
  `).join('');

  res.send(html);
});
```

---

## 4. SSR（Server Side Rendering）

### 4.1 SSR の仕組み

```
SSR（サーバーサイドレンダリング）:
  → リクエストごとにサーバーでHTMLを生成
  → クライアントでHydrationしてインタラクティブに

  フロー:
  1. ブラウザ: GET /users
  2. サーバー: React コンポーネントを実行 → HTML文字列生成
  3. サーバー: データ取得 → HTML にデータを埋め込み
  4. サーバー: 完全なHTMLをレスポンス
  5. ブラウザ: 即座にHTML表示（FCP高速）
  6. ブラウザ: JSバンドルをダウンロード + パース
  7. ブラウザ: Hydration（DOMにイベントリスナーをアタッチ）
  8. ブラウザ: インタラクティブに（TTI）

  サーバーで生成されるHTML:
  <html>
    <head>
      <title>Users | MyApp</title>
      <meta name="description" content="ユーザー一覧">
      <link rel="stylesheet" href="/styles.css">
      <!-- サーバーで取得したデータのハイドレーション用 -->
      <script>
        window.__INITIAL_DATA__ = {"users":[{"id":1,"name":"Taro"},...]};
      </script>
    </head>
    <body>
      <div id="root">
        <h1>Users</h1>           ← サーバーで生成済み
        <ul>
          <li>Taro</li>          ← 即座に表示される
          <li>Hanako</li>
        </ul>
      </div>
      <script src="app.js"></script>  ← Hydration用
    </body>
  </html>
```

### 4.2 SSR の利点と欠点

```
利点:
  ✓ 初期表示が速い（HTMLが即座に描画可能）
  ✓ SEO に最適（完全なHTMLがクローラーに返る）
  ✓ ソーシャルメディアのOGP対応
  ✓ ユーザー固有のコンテンツを初期表示可能
  ✓ 動的データのリアルタイム反映

欠点:
  ✗ サーバー負荷が高い（リクエストごとにレンダリング）
  ✗ TTFB（Time to First Byte）がSSGより遅い
  ✗ Hydration中はインタラクティブでない（Uncanny Valley）
  ✗ サーバーのスケーリングが必要
  ✗ サーバー/クライアント両方で動作するコードが必要
  ✗ Hydration不一致エラーのリスク

適用:
  → ECサイト（SEO + 動的データ + パーソナライズ）
  → SNS（個人プロフィールページ）
  → ニュースサイト（リアルタイム更新）
  → 検索結果ページ

フレームワーク:
  → Next.js（React）
  → Nuxt（Vue）
  → Remix（React）
  → SvelteKit（Svelte）
  → Qwik City（Qwik）
  → Solid Start（SolidJS）
```

### 4.3 Next.js での SSR 実装

```typescript
// Next.js App Router での SSR

// app/users/page.tsx
// デフォルトで Server Component = SSR

import { prisma } from '@/shared/lib/prisma';
import { UserList } from '@/features/users';
import { Metadata } from 'next';

// 動的メタデータ生成
export async function generateMetadata(): Promise<Metadata> {
  return {
    title: 'ユーザー一覧 | MyApp',
    description: '登録ユーザーの一覧を表示します',
    openGraph: {
      title: 'ユーザー一覧',
      description: '登録ユーザーの一覧',
      type: 'website',
    },
  };
}

// SSR: リクエストのたびにデータ取得 + HTML生成
export default async function UsersPage() {
  const users = await prisma.user.findMany({
    orderBy: { createdAt: 'desc' },
    take: 50,
    select: {
      id: true,
      name: true,
      email: true,
      avatar: true,
      createdAt: true,
    },
  });

  return (
    <main>
      <h1>ユーザー一覧</h1>
      <UserList users={users} />
    </main>
  );
}

// force-dynamic: キャッシュなし、常にSSR
export const dynamic = 'force-dynamic';
```

```typescript
// Next.js Pages Router での SSR（getServerSideProps）

import { GetServerSideProps } from 'next';

interface Props {
  users: User[];
  totalCount: number;
}

export const getServerSideProps: GetServerSideProps<Props> = async (context) => {
  const { page = '1', search = '' } = context.query;

  const [users, totalCount] = await Promise.all([
    prisma.user.findMany({
      where: search
        ? { name: { contains: String(search), mode: 'insensitive' } }
        : {},
      orderBy: { createdAt: 'desc' },
      take: 20,
      skip: (Number(page) - 1) * 20,
    }),
    prisma.user.count({
      where: search
        ? { name: { contains: String(search), mode: 'insensitive' } }
        : {},
    }),
  ]);

  return {
    props: {
      users: JSON.parse(JSON.stringify(users)), // Date型のシリアライズ
      totalCount,
    },
  };
};

export default function UsersPage({ users, totalCount }: Props) {
  return (
    <div>
      <h1>ユーザー一覧 ({totalCount}件)</h1>
      <ul>
        {users.map(user => (
          <li key={user.id}>{user.name}</li>
        ))}
      </ul>
    </div>
  );
}
```

### 4.4 Hydration の詳細

```
Hydration（ハイドレーション）の仕組み:

  概要:
  → サーバーで生成した静的HTMLに、クライアントで
    イベントリスナーや状態管理を追加してインタラクティブにする

  フロー:
  1. サーバーHTML:  <button>いいね (0)</button>  ← 見た目だけ
  2. JS ダウンロード + パース
  3. React が仮想DOMを構築
  4. サーバーHTMLと仮想DOMを照合（Reconciliation）
  5. イベントリスナーをアタッチ
  6. <button onClick={handleLike}>いいね (0)</button>  ← インタラクティブ

  Hydration の問題点:
  ① 処理コストが高い:
     → 全コンポーネントツリーを走査
     → 大きなアプリでは数秒かかる場合がある

  ② Uncanny Valley:
     → HTMLは表示されているがクリックが効かない期間
     → ユーザーは操作可能に見えるが反応しない

  ③ Hydration Mismatch:
     → サーバーとクライアントの出力が異なるとエラー
     → 原因: Date.now(), Math.random(), localStorage等
```

```typescript
// Hydration Mismatch の回避

// 悪い例: サーバーとクライアントで異なる出力
function Greeting() {
  // ✗ サーバーとクライアントで異なる時刻
  const now = new Date();
  return <p>現在時刻: {now.toLocaleTimeString()}</p>;
}

// 良い例: クライアントでのみ実行
'use client';
import { useState, useEffect } from 'react';

function Greeting() {
  const [time, setTime] = useState<string>('');

  useEffect(() => {
    // クライアントでのみ時刻を設定
    setTime(new Date().toLocaleTimeString());
    const timer = setInterval(() => {
      setTime(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  return <p>現在時刻: {time || '読み込み中...'}</p>;
}

// suppressHydrationWarning の使用（最終手段）
function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    // サーバーとクライアントで異なるクラスが付く場合
    <html suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
```

---

## 5. SSG（Static Site Generation）

### 5.1 SSG の仕組み

```
SSG（静的サイト生成）:
  → ビルド時に全ページのHTMLを事前生成
  → CDNから静的ファイルとして配信

  フロー:
  1. ビルド時: データ取得 → 全ページのHTML生成
  2. デプロイ: 生成されたHTMLをCDNにアップロード
  3. ブラウザ: GET /about
  4. CDN: 事前生成済みHTMLを返す（最速）
  5. ブラウザ: 即座に表示 + Hydration

  利点:
  ✓ 最速の表示速度（CDNから静的ファイル配信）
  ✓ サーバー負荷ゼロ
  ✓ SEO最適
  ✓ セキュリティが高い（サーバーサイドロジックなし）
  ✓ ホスティングコストが最低
  ✓ 安定性が高い（データベース障害の影響を受けない）

  欠点:
  ✗ ビルド時間が長い（大量ページの場合）
  ✗ データの更新にはリビルドが必要
  ✗ ユーザー固有のコンテンツに不向き
  ✗ ページ数が多いと実用的でない場合がある
  ✗ ビルド時にデータソースへのアクセスが必要

  適用:
  → ブログ、ドキュメント
  → ランディングページ
  → コーポレートサイト
  → マーケティングサイト
  → ヘルプセンター

  フレームワーク:
  → Next.js（React）
  → Astro（マルチフレームワーク、推奨）
  → Gatsby（React）
  → Hugo（Go）
  → 11ty / Eleventy（JS）
  → VitePress（Vue）
```

### 5.2 Next.js での SSG 実装

```typescript
// Next.js App Router での SSG

// app/blog/[slug]/page.tsx

import { notFound } from 'next/navigation';
import { getAllPosts, getPostBySlug } from '@/features/blog/api';
import { Metadata } from 'next';

// ビルド時に生成するパスを定義
export async function generateStaticParams() {
  const posts = await getAllPosts();
  return posts.map((post) => ({
    slug: post.slug,
  }));
}

// 動的メタデータ
export async function generateMetadata({
  params,
}: {
  params: { slug: string };
}): Promise<Metadata> {
  const post = await getPostBySlug(params.slug);
  if (!post) return {};

  return {
    title: `${post.title} | Blog`,
    description: post.excerpt,
    openGraph: {
      title: post.title,
      description: post.excerpt,
      type: 'article',
      publishedTime: post.publishedAt,
      authors: [post.author.name],
      images: [{ url: post.ogImage }],
    },
    twitter: {
      card: 'summary_large_image',
      title: post.title,
      description: post.excerpt,
      images: [post.ogImage],
    },
  };
}

// ページコンポーネント
export default async function BlogPost({
  params,
}: {
  params: { slug: string };
}) {
  const post = await getPostBySlug(params.slug);

  if (!post) {
    notFound();
  }

  return (
    <article>
      <header>
        <time dateTime={post.publishedAt}>
          {new Date(post.publishedAt).toLocaleDateString('ja-JP')}
        </time>
        <h1>{post.title}</h1>
        <p>{post.excerpt}</p>
      </header>
      <div
        className="prose prose-lg"
        dangerouslySetInnerHTML={{ __html: post.contentHtml }}
      />
    </article>
  );
}
```

### 5.3 Astro での SSG

```astro
---
// src/pages/blog/[slug].astro
import { getCollection, getEntry } from 'astro:content';
import BlogLayout from '../../layouts/BlogLayout.astro';
import TableOfContents from '../../components/TableOfContents.astro';
// React コンポーネントも使える（Islands Architecture）
import ShareButton from '../../components/ShareButton.tsx';

export async function getStaticPaths() {
  const posts = await getCollection('blog');
  return posts.map((post) => ({
    params: { slug: post.slug },
    props: { post },
  }));
}

const { post } = Astro.props;
const { Content, headings } = await post.render();
---

<BlogLayout title={post.data.title} description={post.data.description}>
  <article>
    <h1>{post.data.title}</h1>
    <time datetime={post.data.publishedAt.toISOString()}>
      {post.data.publishedAt.toLocaleDateString('ja-JP')}
    </time>

    <TableOfContents headings={headings} />

    <div class="prose">
      <Content />
    </div>

    <!-- Islands Architecture: このコンポーネントだけがインタラクティブ -->
    <ShareButton
      client:visible
      title={post.data.title}
      url={Astro.url.href}
    />
  </article>
</BlogLayout>
```

---

## 6. ISR（Incremental Static Regeneration）

### 6.1 ISR の仕組み

```
ISR = SSG + 定期的な再生成:
  → 初回アクセス時にSSGと同様に静的ページを返す
  → バックグラウンドで定期的にページを再生成
  → stale-while-revalidate パターン

  フロー:
  1. 初回: SSR → HTMLをキャッシュ
  2. revalidate秒以内: キャッシュされたHTMLを返す（即座）
  3. revalidate秒後のリクエスト:
     → キャッシュ(stale)を即座に返す
     → バックグラウンドで再生成
  4. 次のリクエスト: 新しいHTMLを返す

  利点:
  ✓ SSGの速度 + データの鮮度
  ✓ ビルド時間が短い（全ページ事前生成不要）
  ✓ CDNキャッシュが有効
  ✓ データ更新時のリビルド不要
  ✓ 大量ページ（100万+）でもスケール

  欠点:
  ✗ revalidate間隔だけデータが古い可能性
  ✗ 初回アクセスはSSRと同じ速度（キャッシュミス）
  ✗ Next.js のVercelデプロイ以外では制限がある場合も

  適用:
  → ECサイトの商品ページ
  → ブログの記事ページ
  → 更新頻度が中程度のコンテンツ
  → ドキュメント（CMS連携）
```

### 6.2 ISR の実装

```typescript
// Next.js App Router での ISR

// app/products/[id]/page.tsx
import { notFound } from 'next/navigation';

// ISR: 60秒ごとに再検証
export const revalidate = 60;

// ビルド時に生成するページ（人気商品のみ）
export async function generateStaticParams() {
  // 上位100商品のみビルド時に生成
  const topProducts = await prisma.product.findMany({
    orderBy: { salesCount: 'desc' },
    take: 100,
    select: { id: true },
  });

  return topProducts.map((p) => ({
    id: p.id,
  }));
  // ビルド時に生成されなかったページは、初回アクセス時に生成
}

export default async function ProductPage({
  params,
}: {
  params: { id: string };
}) {
  const product = await prisma.product.findUnique({
    where: { id: params.id },
    include: {
      category: true,
      reviews: {
        orderBy: { createdAt: 'desc' },
        take: 10,
      },
    },
  });

  if (!product) notFound();

  return (
    <div>
      <ProductHeader product={product} />
      <ProductGallery images={product.images} />
      <ProductInfo product={product} />
      <ReviewList reviews={product.reviews} />
    </div>
  );
}
```

```typescript
// オンデマンド ISR（On-demand Revalidation）

// app/api/revalidate/route.ts
import { revalidatePath, revalidateTag } from 'next/cache';
import { NextRequest, NextResponse } from 'next/server';

// Webhook で特定ページを即座に再生成
export async function POST(request: NextRequest) {
  const secret = request.headers.get('x-revalidation-secret');

  if (secret !== process.env.REVALIDATION_SECRET) {
    return NextResponse.json({ error: 'Invalid secret' }, { status: 401 });
  }

  const body = await request.json();

  // パスベースの再検証
  if (body.path) {
    revalidatePath(body.path);
    return NextResponse.json({ revalidated: true, path: body.path });
  }

  // タグベースの再検証
  if (body.tag) {
    revalidateTag(body.tag);
    return NextResponse.json({ revalidated: true, tag: body.tag });
  }

  return NextResponse.json({ error: 'Missing path or tag' }, { status: 400 });
}

// CMS の Webhook でオンデマンド再生成
// POST /api/revalidate
// Body: { "path": "/products/123" }
// or:   { "tag": "products" }

// タグベースのキャッシュ管理
// app/products/[id]/page.tsx
async function getProduct(id: string) {
  const res = await fetch(`${API_URL}/products/${id}`, {
    next: { tags: [`product-${id}`, 'products'] },
  });
  return res.json();
}

// product-123 タグを再検証 → /products/123 ページが再生成
// products タグを再検証 → 全商品ページが再生成
```

---

## 7. Streaming SSR

### 7.1 Streaming の仕組み

```
Streaming SSR:
  → サーバーからHTMLを段階的に送信
  → 重要な部分を先に表示、遅いデータは後から表示
  → React 18 + Suspense + Server Components

  従来のSSR:
  データ取得 ──────────→ HTML生成 ──→ 送信 ──→ 表示
  (全データが揃うまで待機)

  Streaming SSR:
  データ取得A ──→ HTML(A) ──→ 送信 ──→ 即座に表示
  データ取得B ────────→ HTML(B) ──→ 送信 ──→ フォールバック → 実データ表示
  データ取得C ──────────────→ HTML(C) → 送信 → フォールバック → 実データ表示

  利点:
  ✓ TTFB が大幅に改善（最初のバイトがすぐに返る）
  ✓ FCP が高速（重要なコンテンツが先に表示）
  ✓ 遅いデータ取得がページ全体をブロックしない
  ✓ ユーザー体験が向上（段階的なコンテンツ表示）

  技術的な仕組み:
  → HTTP Transfer-Encoding: chunked
  → React renderToPipeableStream / renderToReadableStream
  → Suspense 境界ごとに独立したストリーム
```

### 7.2 Streaming SSR の実装

```typescript
// Next.js App Router での Streaming SSR

// app/products/[id]/page.tsx
import { Suspense } from 'react';
import { ProductHeader } from '@/features/products';
import { ReviewsSkeleton, RecommendationsSkeleton } from '@/shared/components/skeletons';

export default async function ProductPage({
  params,
}: {
  params: { id: string };
}) {
  // 即座にレスポンスを開始（商品の基本情報は高速に取得可能）
  const product = await getProduct(params.id);

  return (
    <div>
      {/* 即座に表示される（First Chunk） */}
      <ProductHeader product={product} />
      <ProductGallery images={product.images} />
      <ProductPrice price={product.price} />

      {/* レビュー: 別のDBクエリが必要 → Suspense で遅延 */}
      <Suspense fallback={<ReviewsSkeleton />}>
        <ProductReviews productId={params.id} />
      </Suspense>

      {/* おすすめ: ML推論が必要 → Suspense で遅延 */}
      <Suspense fallback={<RecommendationsSkeleton />}>
        <Recommendations productId={params.id} />
      </Suspense>

      {/* 在庫情報: 外部API → Suspense で遅延 */}
      <Suspense fallback={<StockSkeleton />}>
        <StockInfo productId={params.id} />
      </Suspense>
    </div>
  );
}

// ProductReviews は async Server Component
async function ProductReviews({ productId }: { productId: string }) {
  // この取得に2秒かかっても、ページ全体はブロックされない
  const reviews = await prisma.review.findMany({
    where: { productId },
    orderBy: { createdAt: 'desc' },
    take: 20,
    include: { user: { select: { name: true, avatar: true } } },
  });

  return (
    <section>
      <h2>レビュー ({reviews.length}件)</h2>
      {reviews.map(review => (
        <ReviewCard key={review.id} review={review} />
      ))}
    </section>
  );
}
```

```typescript
// loading.tsx による自動Streaming

// app/dashboard/loading.tsx
// → /dashboard へのナビゲーション時に自動表示
export default function DashboardLoading() {
  return (
    <div className="space-y-4">
      <div className="h-8 w-48 bg-gray-200 animate-pulse rounded" />
      <div className="grid grid-cols-3 gap-4">
        {[1, 2, 3].map(i => (
          <div key={i} className="h-32 bg-gray-200 animate-pulse rounded" />
        ))}
      </div>
      <div className="h-64 bg-gray-200 animate-pulse rounded" />
    </div>
  );
}

// app/dashboard/page.tsx
// loading.tsx があると自動的に Suspense 境界が設定される
export default async function DashboardPage() {
  const [stats, recentOrders, topProducts] = await Promise.all([
    getStats(),
    getRecentOrders(),
    getTopProducts(),
  ]);

  return (
    <div>
      <StatsCards stats={stats} />
      <RecentOrdersTable orders={recentOrders} />
      <TopProductsChart products={topProducts} />
    </div>
  );
}
```

---

## 8. React Server Components（RSC）

### 8.1 RSC の概念

```
RSC（React Server Components）:
  → コンポーネントレベルでサーバー/クライアントを使い分け
  → Next.js App Router のデフォルト
  → サーバーコンポーネントのJSはクライアントに送信されない

  Server Component（デフォルト）:
  → サーバーでレンダリング
  → JSバンドルに含まれない（バンドルサイズ削減）
  → async/awaitでデータ取得可能
  → 状態管理・イベントハンドラ不可
  → Node.js API使用可（fs, crypto等）
  → 直接DB/ファイルシステムにアクセス可能

  Client Component（'use client'）:
  → ブラウザでレンダリング
  → JSバンドルに含まれる
  → useState, useEffect 使用可
  → イベントハンドラ使用可
  → ブラウザAPI使用可（localStorage, window等）

  RSC Payload:
  → Server Componentの出力はReact要素のシリアライズ形式
  → HTMLではなく、仮想DOMの記述
  → Client Componentへの参照を含む
  → 差分更新が可能（ページ遷移時に状態を維持）
```

### 8.2 RSC の実装パターン

```typescript
// Server Component（デフォルト）
// features/users/components/UserList.tsx

import { prisma } from '@/shared/lib/prisma';
import { UserCard } from './UserCard';
import { UserSearchInput } from './UserSearchInput'; // Client Component

// async Server Component: 直接DBアクセス
export async function UserList({
  searchParams,
}: {
  searchParams: { q?: string; page?: string };
}) {
  const query = searchParams.q || '';
  const page = Number(searchParams.page || '1');

  // サーバーで直接データ取得（API不要）
  const [users, total] = await Promise.all([
    prisma.user.findMany({
      where: query
        ? { name: { contains: query, mode: 'insensitive' } }
        : {},
      orderBy: { createdAt: 'desc' },
      take: 20,
      skip: (page - 1) * 20,
      select: {
        id: true,
        name: true,
        email: true,
        avatar: true,
        role: true,
        createdAt: true,
      },
    }),
    prisma.user.count({
      where: query
        ? { name: { contains: query, mode: 'insensitive' } }
        : {},
    }),
  ]);

  return (
    <div>
      {/* Client Component: インタラクション必要 */}
      <UserSearchInput defaultValue={query} />

      {/* Server Component: 静的な表示 */}
      <p>{total}件のユーザー</p>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {users.map((user) => (
          <UserCard key={user.id} user={user} />
        ))}
      </div>

      <ServerPagination total={total} page={page} perPage={20} />
    </div>
  );
}

// Client Component
// features/users/components/UserSearchInput.tsx
'use client';

import { useRouter, useSearchParams } from 'next/navigation';
import { useTransition, useState } from 'react';
import { useDebouncedCallback } from 'use-debounce';

export function UserSearchInput({ defaultValue }: { defaultValue: string }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isPending, startTransition] = useTransition();
  const [value, setValue] = useState(defaultValue);

  const handleSearch = useDebouncedCallback((term: string) => {
    const params = new URLSearchParams(searchParams);
    if (term) {
      params.set('q', term);
    } else {
      params.delete('q');
    }
    params.set('page', '1');

    startTransition(() => {
      router.push(`/users?${params.toString()}`);
    });
  }, 300);

  return (
    <div className="relative">
      <input
        type="search"
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          handleSearch(e.target.value);
        }}
        placeholder="ユーザーを検索..."
        className="w-full px-4 py-2 border rounded-lg"
      />
      {isPending && (
        <div className="absolute right-3 top-3">
          <Spinner size="sm" />
        </div>
      )}
    </div>
  );
}
```

### 8.3 Server/Client の境界設計

```
Server / Client の使い分け:

  Server Component を使う:
  ✓ データベースアクセス
  ✓ サーバーのみのAPI呼び出し（内部マイクロサービス）
  ✓ 大きな依存ライブラリ（マークダウンパーサー、構文ハイライト）
  ✓ 機密情報の処理（APIキー、トークン）
  ✓ 静的なUI表示

  Client Component を使う:
  ✓ useState, useEffect が必要
  ✓ onClick, onChange 等のイベントハンドラ
  ✓ ブラウザAPI（localStorage, window, navigator）
  ✓ サードパーティのクライアントライブラリ（地図、チャート）
  ✓ カスタムフック（状態を含む）
  ✓ React Context（Provider）

  境界設計のベストプラクティス:
  → Client の境界をなるべく葉（リーフ）に近づける
  → ページ全体を 'use client' にしない
  → インタラクティブな部分だけを Client Component に分離

  ┌─────────────────────────────────────────────┐
  │ ProductPage (Server)                         │
  │ ┌───────────────────────────────────────┐   │
  │ │ ProductInfo (Server)                   │   │
  │ │ → 商品名、説明文、スペック（静的表示）  │   │
  │ └───────────────────────────────────────┘   │
  │ ┌──────────────┐ ┌────────────────────┐    │
  │ │ AddToCart     │ │ ImageGallery       │    │
  │ │ (Client)     │ │ (Client)           │    │
  │ │ → onClick    │ │ → スワイプ操作     │    │
  │ │ → useState   │ │ → useState        │    │
  │ └──────────────┘ └────────────────────┘    │
  │ ┌───────────────────────────────────────┐   │
  │ │ Reviews (Server)                       │   │
  │ │ → async データ取得、静的表示            │   │
  │ │ ┌─────────────────────────────────┐   │   │
  │ │ │ ReviewForm (Client)              │   │   │
  │ │ │ → フォーム入力、送信             │   │   │
  │ │ └─────────────────────────────────┘   │   │
  │ └───────────────────────────────────────┘   │
  └─────────────────────────────────────────────┘
```

```typescript
// Server Component から Client Component へのデータの渡し方

// 1. Props として渡す（シリアライズ可能なデータのみ）
// Server Component
async function ProductPage({ params }: { params: { id: string } }) {
  const product = await getProduct(params.id);

  return (
    <div>
      <h1>{product.name}</h1>
      {/* シリアライズ可能な値のみ渡す */}
      <AddToCartButton
        productId={product.id}
        price={product.price}
        inStock={product.stock > 0}
      />
    </div>
  );
}

// 2. children パターン（Server Component を Client Component に渡す）
// Client Component
'use client';
function TabPanel({ children, tabs }: { children: React.ReactNode; tabs: string[] }) {
  const [activeTab, setActiveTab] = useState(0);
  return (
    <div>
      <div className="flex gap-2">
        {tabs.map((tab, i) => (
          <button key={tab} onClick={() => setActiveTab(i)}>{tab}</button>
        ))}
      </div>
      {children} {/* Server Component の子要素をそのまま表示 */}
    </div>
  );
}

// Server Component
async function ProductDetailPage() {
  return (
    <TabPanel tabs={['詳細', 'レビュー', '仕様']}>
      {/* これらは Server Component として実行される */}
      <ProductDetails />
      <ProductReviews />
      <ProductSpecs />
    </TabPanel>
  );
}

// 3. 渡せないもの
// ✗ 関数（onClick等）: シリアライズ不可
// ✗ Date オブジェクト: string/number に変換が必要
// ✗ Map, Set: 配列/オブジェクトに変換が必要
// ✗ クラスインスタンス: プレーンオブジェクトに変換が必要
```

---

## 9. Islands Architecture

### 9.1 Islands の概念

```
Islands Architecture:
  → ページの大部分は静的HTML
  → インタラクティブな部分だけをJavaScriptで「島」として実装
  → 各「島」は独立してHydration

  従来のSSR:
  ┌────────────────────────────────────────┐
  │ ████████████████████████████████████████│ ← 全体がHydration対象
  │ ██ Header ██ Nav █████████████████████ │ ← 全JSがロードされるまで
  │ ██████████████████████████████████████ │    インタラクティブにならない
  │ ██ Content ████████████████████████████│
  │ ██████████████████████████████████████ │
  │ ██ Sidebar ███ Footer █████████████████│
  └────────────────────────────────────────┘

  Islands Architecture:
  ┌────────────────────────────────────────┐
  │                      ┌──────────┐     │
  │ Header(HTML)         │ SearchBar│     │ ← インタラクティブ島
  │                      │ (Island) │     │
  │                      └──────────┘     │
  │                                        │
  │ Content (HTML) ─── 静的HTML ──────────│
  │                                        │
  │           ┌──────────────┐             │
  │           │ ImageCarousel│             │ ← インタラクティブ島
  │           │ (Island)     │             │
  │           └──────────────┘             │
  │                                        │
  │ Footer (HTML) ────────────────────────│
  └────────────────────────────────────────┘

  利点:
  ✓ JSバンドルが最小限
  ✓ TTI（Time to Interactive）が大幅改善
  ✓ 静的部分のHydration不要
  ✓ 各島が独立してロード・実行

  フレームワーク:
  → Astro（最も人気）
  → Fresh（Deno）
  → Eleventy + is-land
```

### 9.2 Astro での Islands 実装

```astro
---
// src/pages/index.astro
import Layout from '../layouts/Layout.astro';
import Hero from '../components/Hero.astro';
import Features from '../components/Features.astro';
// インタラクティブ島（React コンポーネント）
import ContactForm from '../components/ContactForm.tsx';
import TestimonialCarousel from '../components/TestimonialCarousel.tsx';
import PricingCalculator from '../components/PricingCalculator.tsx';
---

<Layout title="MyService">
  <!-- 静的HTML: JSなし -->
  <Hero />

  <!-- 静的HTML: JSなし -->
  <Features />

  <!-- Island: ビューポートに入った時にHydration -->
  <TestimonialCarousel client:visible />

  <!-- Island: ページロード時にHydration（重要なインタラクション） -->
  <PricingCalculator client:load />

  <!-- Island: アイドル時にHydration（優先度低） -->
  <ContactForm client:idle />

  <!-- Island: メディアクエリでHydration -->
  <MobileMenu client:media="(max-width: 768px)" />
</Layout>
```

```
Astro の client ディレクティブ:

  client:load      → ページロード時に即座にHydration
  client:idle      → ブラウザがアイドル時にHydration
  client:visible   → ビューポートに入った時にHydration
  client:media     → メディアクエリが一致した時にHydration
  client:only      → SSRせずクライアントのみでレンダリング

  パフォーマンスへの影響:
  ┌──────────────┬──────────┬──────────────────────┐
  │ ディレクティブ│ JS送信    │ 使用場面              │
  ├──────────────┼──────────┼──────────────────────┤
  │ (なし)       │ 0KB      │ 静的表示のみ          │
  │ client:visible│ 遅延     │ ファーストビュー外    │
  │ client:idle  │ 遅延     │ 優先度低い機能        │
  │ client:load  │ 即座     │ 重要なインタラクション│
  │ client:only  │ 即座     │ SSR不要な機能         │
  └──────────────┴──────────┴──────────────────────┘
```

---

## 10. Partial Hydration と Selective Hydration

### 10.1 React 18 の Selective Hydration

```typescript
// React 18 の Selective Hydration
// Suspense 境界ごとに独立してHydration

// ユーザーがクリックした領域を優先的にHydration
import { Suspense } from 'react';

function App() {
  return (
    <div>
      {/* この部分は先にHydration */}
      <Header />
      <Navigation />

      <main>
        {/* Hydration中にクリックされたら優先される */}
        <Suspense fallback={<ProductListSkeleton />}>
          <ProductList />
        </Suspense>

        <Suspense fallback={<SidebarSkeleton />}>
          <Sidebar />
        </Suspense>
      </main>

      {/* 最後にHydration */}
      <Suspense fallback={<FooterSkeleton />}>
        <Footer />
      </Suspense>
    </div>
  );
}

// 仕組み:
// 1. サーバーがストリームでHTMLを送信
// 2. 各 Suspense 境界は独立してHydration
// 3. ユーザーが ProductList をクリック
// 4. React は ProductList を優先的にHydration
// 5. クリックイベントはHydration完了後にリプレイ
```

### 10.2 Qwik の Resumability

```
Qwik のアプローチ（Hydration の代替）:

  従来のHydration:
  → サーバーでレンダリング
  → クライアントでコンポーネントツリー全体を再構築
  → イベントリスナーをアタッチ
  → 問題: O(n) の処理コスト（コンポーネント数に比例）

  Qwik の Resumability:
  → サーバーでレンダリング
  → HTMLにイベントハンドラの参照を埋め込み
  → クライアントで必要な時だけコードをロード（Lazy loading）
  → 問題: O(1) の初期コスト

  <!-- Qwik のHTML出力例 -->
  <button on:click="./chunk-abc.js#handleClick_1">
    いいね (0)
  </button>
  <!-- イベント発生時に初めてJSをロード・実行 -->

  比較:
  ┌──────────┬──────────┬───────────────────────┐
  │ 方式      │ 初期JS   │ TTI                    │
  ├──────────┼──────────┼───────────────────────┤
  │ SPA      │ 全バンドル│ JS ロード + 実行 後    │
  │ SSR+Hydr │ 全バンドル│ Hydration 完了後       │
  │ Islands  │ 島のみ   │ 島の Hydration 後      │
  │ Qwik     │ ~1KB     │ 即座（イベント時にロード）│
  └──────────┴──────────┴───────────────────────┘
```

---

## 11. 選定フローチャートと実務ガイド

### 11.1 選定フローチャート

```
SEO が必要？
├── NO → 管理画面/ダッシュボード？
│   ├── YES → SPA（Vite + React）
│   └── NO → リアルタイム性が重要？
│       ├── YES → SPA（WebSocket + React）
│       └── NO → 要件次第（SPA or SSR）
└── YES → コンテンツは動的？
    ├── NO → 更新頻度は？
    │   ├── ほぼなし → SSG（Astro / Next.js）
    │   ├── 低い → SSG + On-demand Revalidation
    │   └── 中程度 → ISR（Next.js, revalidate: 60）
    └── YES → ユーザー固有コンテンツ？
        ├── YES → SSR + Streaming（Next.js App Router）
        └── NO → ページ数は？
            ├── 少ない → SSR
            └── 多い → ISR + On-demand Revalidation

コンテンツサイト（ブログ, ドキュメント）？
├── YES → JSインタラクション多い？
│   ├── YES → Next.js SSG/ISR
│   └── NO → Astro（Islands Architecture）
└── NO → 上記フローに従う
```

### 11.2 ハイブリッドアプローチの実践

```
実務のベストプラクティス:
  → 1つのアプリ内でハイブリッドに使い分け
  → ページ単位で最適な方式を選択
  → Next.js App Router: RSC + ISR + Streaming を組み合わせ

例（ECサイト）:
  / (トップ)          → SSG（更新少ない）
  /products           → ISR（60秒ごと再生成）
  /products/[id]      → ISR + Streaming（商品情報 + レビュー）
  /search             → SSR（検索クエリに依存）
  /cart               → CSR（ユーザー固有、SEO不要）
  /checkout           → SSR（決済フロー、セキュリティ重要）
  /account            → CSR（ログイン後、SEO不要）
  /blog               → SSG（Astro, 最小限のJS）
  /blog/[slug]        → SSG + On-demand Revalidation

例（SaaS アプリ）:
  / (ランディングページ) → SSG
  /pricing              → SSG + ISR
  /docs                 → SSG（Astro / VitePress）
  /login                → CSR
  /dashboard            → CSR（SPA）
  /settings             → CSR（SPA）
  /admin                → CSR（SPA）
  /api/*                → サーバーレスAPI

例（メディアサイト）:
  /                     → ISR（5分ごと再生成）
  /category/[slug]      → ISR（5分ごと）
  /article/[slug]       → ISR + On-demand（CMS Webhook）
  /author/[slug]        → ISR（1時間ごと）
  /search               → SSR
```

### 11.3 パフォーマンス比較の実測値

```
実測パフォーマンス比較（同一アプリ、モバイル3G回線）:

  ECサイト商品一覧ページ（20商品表示）:
  ┌──────────────┬────────┬────────┬────────┬────────┐
  │ 方式          │ TTFB   │ FCP    │ LCP    │ TTI    │
  ├──────────────┼────────┼────────┼────────┼────────┤
  │ CSR          │ 200ms  │ 4.2s   │ 5.8s   │ 5.8s   │
  │ SSR          │ 800ms  │ 1.2s   │ 2.1s   │ 4.5s   │
  │ SSG          │ 100ms  │ 0.8s   │ 1.5s   │ 3.8s   │
  │ ISR          │ 150ms  │ 0.9s   │ 1.6s   │ 3.9s   │
  │ SSR+Streaming│ 300ms  │ 0.9s   │ 1.8s   │ 3.5s   │
  │ RSC          │ 350ms  │ 1.0s   │ 1.9s   │ 2.8s   │
  │ Astro(Islands)│ 100ms │ 0.7s   │ 1.3s   │ 1.5s   │
  └──────────────┴────────┴────────┴────────┴────────┘

  JSバンドルサイズ比較（gzip後）:
  ┌──────────────┬──────────────────┐
  │ 方式          │ 初期JSバンドル    │
  ├──────────────┼──────────────────┤
  │ CSR          │ 185KB            │
  │ SSR          │ 185KB            │
  │ SSG          │ 165KB            │
  │ RSC          │ 95KB             │
  │ Astro        │ 15KB（島のみ）    │
  │ Qwik         │ 1KB              │
  └──────────────┴──────────────────┘
```

---

## まとめ

| 方式 | 初期表示 | SEO | JSバンドル | 適用例 |
|------|---------|-----|-----------|--------|
| CSR/SPA | 遅 | 悪 | 大 | 管理画面、ダッシュボード |
| MPA | 速 | 良 | 最小 | ブログ（htmx） |
| SSR | 速 | 良 | 大 | ECサイト、SNS |
| SSG | 最速 | 最良 | 中 | ブログ、ドキュメント |
| ISR | 速 | 良 | 中 | 商品ページ、記事 |
| Streaming | 速 | 良 | 中 | 複雑なページ |
| RSC | 速 | 良 | 小 | ハイブリッド（Next.js） |
| Islands | 速 | 良 | 最小 | コンテンツサイト（Astro） |
| Qwik | 最速 | 良 | 極小 | パフォーマンス最優先 |

---

## FAQ

### Q1. SPA、MPA、SSRのどれを選ぶべきか？プロジェクトの選択基準は？

**A.** プロジェクトの要件に応じて以下の基準で判断する。

**SPA（CSR）を選ぶべき場合:**
- 管理画面やダッシュボード（SEO不要）
- ログイン後のアプリケーション（認証が前提）
- 高度なインタラクティブ性が必要（リアルタイム編集、複雑なUI）
- 例: Notion、Figma、Gmail

**MPA（従来型）を選ぶべき場合:**
- SEOが最優先でJavaScript依存を最小化したい
- 静的コンテンツが中心（ブログ、ドキュメント）
- htmxなど軽量なインタラクティブ性で十分
- 例: 企業サイト、ブログ（htmx + Go/Rails）

**SSR（Next.js App Router等）を選ぶべき場合:**
- SEOとインタラクティブ性の両立が必要
- ユーザーごとに異なるコンテンツを表示（パーソナライズ）
- ECサイト、SNS、ニュースサイト
- 例: Vercel公式サイト、ECサイト

**SSG（Next.js / Astro）を選ぶべき場合:**
- 更新頻度が低い静的コンテンツ
- 最速の初期表示が必要
- ドキュメント、ブログ、ランディングページ
- 例: 技術ブログ（Astro）、ドキュメントサイト（VitePress）

**ハイブリッド（SSR + SSG + CSR混在）を選ぶべき場合:**
- 大規模アプリケーションで複数の要件が混在
- Next.js App Router で1つのアプリ内でページごとに最適化
- 例: ECサイト（トップページ=SSG、商品ページ=ISR、カート=CSR、検索=SSR）

### Q2. Next.js の App Router と Pages Router の違いは？どちらを使うべきか？

**A.** **2024年以降の新規プロジェクトでは App Router を推奨する。** ただし、既存プロジェクトの移行は段階的に行う。

**App Router の利点:**
- React Server Components（RSC）によるバンドルサイズ削減
- Streaming SSR によるTTFB改善
- レイアウト共有機能（layout.tsx）の標準化
- Server Actions によるフォーム処理の簡略化
- Parallel Routes / Intercepting Routes などの高度なルーティング

**Pages Router の利点:**
- 安定性（枯れた技術、豊富な事例）
- 学習コストが低い（従来のReact開発者に馴染みやすい）
- 一部のライブラリがまだ App Router 未対応

**移行判断基準:**
- 新規プロジェクト → App Router
- 既存プロジェクト（小〜中規模） → 段階的に App Router へ移行
- 既存プロジェクト（大規模、安定運用中） → Pages Router のまま維持も選択肢

### Q3. SSG と ISR の使い分けは？どちらを選ぶべきか？

**A.** データの更新頻度とビルド時間のトレードオフで判断する。

**SSG（Static Site Generation）を選ぶべき場合:**
- データの更新頻度が非常に低い（月1回以下）
- ページ数が少ない（100ページ未満）
- ビルド時間が許容範囲内（数分以内）
- 例: 企業サイト、ドキュメント、小規模ブログ

**ISR（Incremental Static Regeneration）を選ぶべき場合:**
- データが定期的に更新される（数分〜数時間ごと）
- ページ数が多い（数千〜数万ページ）
- ビルド時間を短縮したい
- 例: 大規模ECサイトの商品ページ、ニュースサイト

**Next.js での実装例:**

```typescript
// SSG: ビルド時に全ページを生成
export async function generateStaticParams() {
  const posts = await getAllPosts();
  return posts.map(post => ({ slug: post.slug }));
}

// ISR: 60秒ごとに再生成 + On-demand Revalidation
export const revalidate = 60;

export default async function Page({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug);
  return <Article post={post} />;
}

// On-demand Revalidation（CMS Webhookから呼び出し）
// app/api/revalidate/route.ts
import { revalidatePath } from 'next/cache';

export async function POST(request: Request) {
  const { path } = await request.json();
  revalidatePath(path);
  return Response.json({ revalidated: true });
}
```

**ISR の注意点:**
- 初回リクエストはビルド済みページを返す（Stale）
- バックグラウンドで再生成
- 再生成失敗時は古いページを返し続ける（安全性）

---

## FAQ（よくある質問）

### Q1: SPA、MPA、SSRの選択基準は？

**A:** プロジェクト要件に応じて以下の基準で選択します。

| 要件 | 推奨方式 | 理由 |
|------|---------|------|
| SEOが最重要（ブログ、ECサイト） | SSG / ISR | 静的HTMLでクローラー最適化、LCP最速 |
| SEO必要 + 動的コンテンツ（SNS、ニュース） | SSR / RSC | サーバーレンダリングで初期HTML生成 |
| SEO不要 + 高インタラクティブ性（管理画面） | SPA (CSR) | クライアント側で高速なページ遷移 |
| コンテンツサイト（ドキュメント、マーケティング） | Islands (Astro) | 最小JSでパフォーマンス最大化 |
| ハイブリッド要件（ECサイト全体） | Next.js App Router | ページ単位でSSG/ISR/SSR/CSRを使い分け |

実際には、Next.js App Router のようなフレームワークで、ルート単位で最適な方式を組み合わせる **ハイブリッドアプローチ** が現代的です。

### Q2: Next.js の App Router と Pages Router の違いは？

**A:** App Router（Next.js 13+）は React Server Components をベースにした新しいアーキテクチャです。

| 観点 | App Router | Pages Router |
|------|-----------|--------------|
| レンダリング | デフォルトでServer Components | デフォルトでClient Components |
| レイアウト | layout.tsx で階層的に定義 | _app.tsx で全ページ共通 |
| データフェッチ | async/await 直接記述 | getServerSideProps / getStaticProps |
| ルーティング | ディレクトリベース（app/） | ファイルベース（pages/） |
| Streaming | ネイティブサポート（Suspense） | 手動実装 |
| バンドルサイズ | Server Componentsで大幅削減可能 | すべてクライアントバンドルに含まれる |

**推奨:** 新規プロジェクトはApp Routerを採用し、Server Componentsの恩恵を最大限活用すべきです。Pages Routerは既存プロジェクトのメンテナンスモードです。

### Q3: SSG と ISR の使い分けは？

**A:** データの更新頻度とページ数で判断します。

**SSG（Static Site Generation）が適している場合:**
- ビルド時に全ページを事前生成できる（ページ数が限定的）
- コンテンツがほぼ静的（ブログ記事、ドキュメント、ランディングページ）
- 更新は再デプロイで対応可能
- 例: 個人ブログ（50記事）、企業サイト（20ページ）

```typescript
// app/blog/[slug]/page.tsx (Next.js App Router)
export async function generateStaticParams() {
  const posts = await getAllPosts();
  return posts.map(post => ({ slug: post.slug }));
}

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const post = await getPost(params.slug);
  return <Article post={post} />;
}
```

**ISR（Incremental Static Regeneration）が適している場合:**
- ページ数が膨大（数千〜数万ページ）
- 定期的にコンテンツが更新される（商品情報、記事）
- On-demand Revalidation でCMS更新と連携したい
- 例: ECサイト商品ページ（10万点）、メディアサイト（1万記事）

```typescript
// app/products/[id]/page.tsx
export const revalidate = 3600; // 1時間ごとに再生成

export default async function ProductPage({ params }: { params: { id: string } }) {
  const product = await getProduct(params.id);
  return <ProductDetail product={product} />;
}
```

**ハイブリッド戦略:**
- 人気商品上位100点 → SSG（ビルド時生成）
- その他の商品 → ISR（初回アクセス時に生成、1時間ごと再検証）
- CMS更新時 → On-demand Revalidation（Webhookで即時反映）

---

## 次に読むべきガイド


---

## 参考文献

1. Vercel. "Rendering Fundamentals." nextjs.org/docs, 2024.
2. patterns.dev. "Rendering Patterns." patterns.dev, 2024.
3. web.dev. "Rendering on the Web." web.dev, 2024.
4. Astro. "Why Astro?" docs.astro.build, 2024.
5. React. "Server Components." react.dev, 2024.
6. Builder.io. "Qwik: Resumable Framework." qwik.dev, 2024.
7. htmx. "htmx - high power tools for HTML." htmx.org, 2024.
8. Jason Miller. "Islands Architecture." jasonformat.com, 2020.
9. Dan Abramov. "The Two Reacts." overreacted.io, 2023.
10. Ryan Carniato. "The Future of Rendering." dev.to, 2023.
