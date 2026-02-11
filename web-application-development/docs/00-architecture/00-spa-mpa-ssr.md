# SPA / MPA / SSR

> Webアプリのレンダリング方式は性能とUXを決定づける。SPA、MPA、SSR、SSG、ISRの特徴と選定基準を理解し、プロジェクト要件に最適なアーキテクチャを選択する。

## この章で学ぶこと

- [ ] 各レンダリング方式の仕組みと特徴を理解する
- [ ] パフォーマンスとSEOの観点から選定基準を把握する
- [ ] ハイブリッドレンダリングの設計を学ぶ

---

## 1. レンダリング方式の全体像

```
方式の比較:

         初期表示  操作性  SEO   サーバー負荷  複雑度
─────────────────────────────────────────────────
CSR/SPA   遅い     最高    悪い   低い         低い
SSR       速い     高い    良い   高い         中程度
SSG       最速     高い    最良   最低         低い
ISR       速い     高い    良い   低い         中程度
Streaming 速い     高い    良い   中程度       高い

レンダリングのタイミング:
  CSR:       クライアント（ブラウザ）でレンダリング
  SSR:       リクエスト時にサーバーでレンダリング
  SSG:       ビルド時にサーバーでレンダリング
  ISR:       初回リクエスト時 + 定期的に再生成
  Streaming: サーバーで段階的にレンダリング
```

---

## 2. CSR / SPA（Client Side Rendering）

```
SPA（Single Page Application）:
  → ブラウザがJSを実行してHTMLを生成
  → ページ遷移はクライアントサイドルーティング

  フロー:
  1. ブラウザ: GET /
  2. サーバー: 空の HTML + JS バンドルを返す
  3. ブラウザ: JS を実行 → DOM を構築 → 画面表示
  4. ブラウザ: API コール → データ取得 → 画面更新

  <html>
    <body>
      <div id="root"></div>     ← 空のHTML
      <script src="app.js"></script>  ← JSが全てを描画
    </body>
  </html>

  利点:
  ✓ ページ遷移が高速（サーバーリクエストなし）
  ✓ リッチなインタラクション
  ✓ サーバー負荷が低い（静的ファイル配信のみ）
  ✓ オフライン対応が容易（PWA）

  欠点:
  ✗ 初期表示が遅い（JSバンドルのダウンロード + 実行）
  ✗ SEO が困難（クローラーがJS実行しない場合）
  ✗ FCP / LCPが遅い
  ✗ JSが無効だと何も表示されない

  適用:
  → 管理画面、ダッシュボード
  → ログイン後のアプリケーション
  → SEO不要なツール系アプリ

  フレームワーク:
  → React（Vite）
  → Vue（Vite）
  → Angular
```

---

## 3. SSR（Server Side Rendering）

```
SSR（サーバーサイドレンダリング）:
  → リクエストごとにサーバーでHTMLを生成

  フロー:
  1. ブラウザ: GET /users
  2. サーバー: データ取得 → HTML生成 → レスポンス
  3. ブラウザ: 即座にHTML表示（FCP高速）
  4. ブラウザ: JSを実行 → Hydration → インタラクティブに

  <html>
    <body>
      <div id="root">
        <h1>Users</h1>           ← サーバーで生成済み
        <ul>
          <li>Taro</li>
          <li>Hanako</li>
        </ul>
      </div>
      <script src="app.js"></script>  ← Hydration用
    </body>
  </html>

  利点:
  ✓ 初期表示が速い（HTMLが即座に描画可能）
  ✓ SEO に最適
  ✓ ソーシャルメディアのOGP対応

  欠点:
  ✗ サーバー負荷が高い（リクエストごとにレンダリング）
  ✗ TTFB（Time to First Byte）がSSGより遅い
  ✗ Hydration中はインタラクティブでない
  ✗ サーバーのスケーリングが必要

  適用:
  → ECサイト（SEO + 動的データ）
  → SNS（個人プロフィールページ）
  → ニュースサイト

  フレームワーク:
  → Next.js（React）
  → Nuxt（Vue）
  → Remix（React）
  → SvelteKit（Svelte）
```

---

## 4. SSG（Static Site Generation）

```
SSG（静的サイト生成）:
  → ビルド時に全ページのHTMLを事前生成

  フロー:
  1. ビルド時: データ取得 → 全ページのHTML生成
  2. ブラウザ: GET /about
  3. CDN: 事前生成済みHTMLを返す（最速）
  4. ブラウザ: 即座に表示 + Hydration

  利点:
  ✓ 最速の表示速度（CDNから静的ファイル配信）
  ✓ サーバー負荷ゼロ
  ✓ SEO最適
  ✓ セキュリティが高い（サーバーサイドロジックなし）

  欠点:
  ✗ ビルド時間が長い（大量ページの場合）
  ✗ データの更新にはリビルドが必要
  ✗ ユーザー固有のコンテンツに不向き

  適用:
  → ブログ、ドキュメント
  → ランディングページ
  → コーポレートサイト

  フレームワーク:
  → Next.js（React）
  → Astro（マルチフレームワーク）
  → Gatsby（React）
  → Hugo, 11ty
```

---

## 5. ISR（Incremental Static Regeneration）

```
ISR = SSG + 定期的な再生成:
  → 初回アクセス時にSSGと同様に静的ページを返す
  → バックグラウンドで定期的にページを再生成
  → stale-while-revalidate パターン

  Next.js での実装:
  // app/products/[id]/page.tsx
  export const revalidate = 60; // 60秒ごとに再検証

  export default async function ProductPage({ params }) {
    const product = await getProduct(params.id);
    return <ProductDetail product={product} />;
  }

  フロー:
  1. 初回: SSR → HTMLをキャッシュ
  2. 60秒以内: キャッシュされたHTMLを返す
  3. 60秒後のリクエスト:
     → キャッシュ(stale)を即座に返す
     → バックグラウンドで再生成
  4. 次のリクエスト: 新しいHTMLを返す

  利点:
  ✓ SSGの速度 + データの鮮度
  ✓ ビルド時間が短い（全ページ事前生成不要）
  ✓ CDNキャッシュが有効

  適用:
  → ECサイトの商品ページ
  → ブログの記事ページ
  → 更新頻度が中程度のコンテンツ
```

---

## 6. React Server Components

```
RSC（React Server Components）:
  → コンポーネントレベルでサーバー/クライアントを使い分け
  → Next.js App Router のデフォルト

  Server Component（デフォルト）:
  → サーバーでレンダリング
  → JSバンドルに含まれない
  → async/awaitでデータ取得可能
  → 状態管理・イベントハンドラ不可

  Client Component（'use client'）:
  → ブラウザでレンダリング
  → useState, useEffect 使用可
  → イベントハンドラ使用可

  // Server Component（デフォルト）
  async function UserList() {
    const users = await db.users.findMany();  // 直接DB アクセス
    return (
      <ul>
        {users.map(u => <li key={u.id}>{u.name}</li>)}
      </ul>
    );
  }

  // Client Component
  'use client';
  function SearchInput() {
    const [query, setQuery] = useState('');
    return <input value={query} onChange={e => setQuery(e.target.value)} />;
  }

  使い分け:
  Server Component: データ取得、重い依存のレンダリング
  Client Component: インタラクション、状態管理、ブラウザAPI
```

---

## 7. 選定フローチャート

```
SEO が必要？
├── NO → 管理画面/ダッシュボード？
│   ├── YES → SPA（Vite + React）
│   └── NO → 要件次第（SPA or SSR）
└── YES → コンテンツは動的？
    ├── NO → 更新頻度は？
    │   ├── 低い → SSG（Astro, Next.js）
    │   └── 中程度 → ISR（Next.js）
    └── YES → ユーザー固有コンテンツ？
        ├── YES → SSR + Streaming（Next.js App Router）
        └── NO → ISR or SSR

実務のベストプラクティス:
  → 1つのアプリ内でハイブリッドに使い分け
  → ページ単位で最適な方式を選択
  → Next.js App Router: RSC + ISR + Streaming を組み合わせ

  例（ECサイト）:
  / (トップ)          → SSG（更新少ない）
  /products           → ISR（60秒ごと再生成）
  /products/[id]      → ISR（商品情報）
  /cart               → CSR（ユーザー固有）
  /checkout           → SSR（決済フロー）
  /account            → CSR（ログイン後）
```

---

## まとめ

| 方式 | 初期表示 | SEO | 適用例 |
|------|---------|-----|--------|
| CSR/SPA | 遅 | 悪 | 管理画面、ダッシュボード |
| SSR | 速 | 良 | ECサイト、SNS |
| SSG | 最速 | 最良 | ブログ、ドキュメント |
| ISR | 速 | 良 | 商品ページ、記事 |
| RSC | 速 | 良 | ハイブリッド（Next.js） |

---

## 次に読むべきガイド
→ [[01-project-structure.md]] — プロジェクト構成

---

## 参考文献
1. Vercel. "Rendering Fundamentals." nextjs.org/docs, 2024.
2. patterns.dev. "Rendering Patterns." patterns.dev, 2024.
3. web.dev. "Rendering on the Web." web.dev, 2024.
