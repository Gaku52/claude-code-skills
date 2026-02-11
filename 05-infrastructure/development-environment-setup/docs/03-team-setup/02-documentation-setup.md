# ドキュメント環境 (Documentation Setup)

> VitePress / Docusaurus によるドキュメントサイトの構築と、ADR (Architecture Decision Records) による意思決定の記録を通じて、チームの知識を体系的に管理する手法を学ぶ。

## この章で学ぶこと

1. **VitePress / Docusaurus の導入と設定** -- Markdown ベースのドキュメントサイトを構築し、自動デプロイまでのパイプラインを整備する
2. **ADR (Architecture Decision Records) の運用** -- アーキテクチャの意思決定を記録し、「なぜこの設計にしたのか」を追跡可能にする
3. **ドキュメント運用のベストプラクティス** -- ドキュメントの鮮度を保ち、コードと一緒にメンテナンスする文化を構築する

---

## 1. ドキュメントツールの選択

### 1.1 ツール比較

| 項目 | VitePress | Docusaurus | Nextra | GitBook |
|------|-----------|------------|--------|---------|
| フレームワーク | Vue 3 / Vite | React / Webpack | Next.js | SaaS |
| ビルド速度 | 非常に高速 | 中 | 高速 | N/A |
| カスタマイズ | Vue コンポーネント | React コンポーネント | React | 限定的 |
| 多言語 (i18n) | 対応 | 強力な対応 | 対応 | 対応 |
| バージョニング | 手動 | 標準対応 | 手動 | 対応 |
| 検索 | 内蔵(miniSearch) | Algolia 統合 | Flexsearch | 内蔵 |
| デプロイ | 静的ホスティング | 静的ホスティング | Vercel推奨 | SaaS |
| 学習コスト | 低 | 中 | 低 | 最低 |
| 適用場面 | OSS / 技術文書 | 大規模プロジェクト | Next.js利用者 | 非エンジニア含む |

### 1.2 選択ガイド

```
+------------------------------------------------------------------+
|              ドキュメントツール選択フロー                            |
+------------------------------------------------------------------+
|                                                                  |
|  チームは React を使っている?                                      |
|    |                                                             |
|   YES                          NO                                |
|    |                            |                                |
|    v                            v                                |
|  バージョニングが必要?       Vue を使っている?                      |
|    |        |                  |        |                        |
|   YES      NO                 YES      NO                        |
|    |        |                  |        |                        |
|    v        v                  v        v                        |
| Docusaurus  Nextra          VitePress  VitePress                 |
|                                        (学習コスト最低)           |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 2. VitePress の導入

### 2.1 初期セットアップ

```bash
# プロジェクト内にドキュメントを追加
# docs/ ディレクトリで管理

# pnpm (推奨)
pnpm add -D vitepress

# ディレクトリ構造
# docs/
#   .vitepress/
#     config.ts     -- サイト設定
#     theme/        -- カスタムテーマ
#   index.md        -- トップページ
#   guide/
#     getting-started.md
#     architecture.md
#   api/
#     overview.md
#   adr/
#     0001-use-typescript.md
```

### 2.2 VitePress 設定ファイル

```typescript
// docs/.vitepress/config.ts
import { defineConfig } from 'vitepress';

export default defineConfig({
  title: 'MyApp Documentation',
  description: 'MyApp の開発者向けドキュメント',
  lang: 'ja-JP',

  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
  ],

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: 'ガイド', link: '/guide/getting-started' },
      { text: 'API', link: '/api/overview' },
      { text: 'ADR', link: '/adr/' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'はじめに',
          items: [
            { text: 'クイックスタート', link: '/guide/getting-started' },
            { text: 'アーキテクチャ', link: '/guide/architecture' },
            { text: '開発環境セットアップ', link: '/guide/dev-setup' },
          ],
        },
        {
          text: '開発ガイド',
          items: [
            { text: 'コーディング規約', link: '/guide/coding-standards' },
            { text: 'テスト戦略', link: '/guide/testing' },
            { text: 'デプロイ', link: '/guide/deployment' },
          ],
        },
      ],
      '/api/': [
        {
          text: 'API リファレンス',
          items: [
            { text: '概要', link: '/api/overview' },
            { text: '認証', link: '/api/authentication' },
            { text: 'エンドポイント', link: '/api/endpoints' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/your-org/myapp' },
    ],

    search: {
      provider: 'local', // miniSearch 内蔵検索
    },

    editLink: {
      pattern: 'https://github.com/your-org/myapp/edit/main/docs/:path',
      text: 'このページを編集する',
    },

    lastUpdated: {
      text: '最終更新',
    },

    footer: {
      message: 'MIT License',
      copyright: 'Copyright (c) 2025 Your Org',
    },
  },

  markdown: {
    lineNumbers: true, // コードブロックに行番号
  },
});
```

### 2.3 package.json スクリプト

```jsonc
// package.json (docs 関連)
{
  "scripts": {
    "docs:dev": "vitepress dev docs",
    "docs:build": "vitepress build docs",
    "docs:preview": "vitepress preview docs"
  }
}
```

---

## 3. Docusaurus の導入

### 3.1 初期セットアップ

```bash
npx create-docusaurus@latest docs classic --typescript
```

### 3.2 Docusaurus 設定

```typescript
// docs/docusaurus.config.ts
import { themes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';

const config: Config = {
  title: 'MyApp Documentation',
  tagline: 'MyApp の開発者向けドキュメント',
  url: 'https://docs.example.com',
  baseUrl: '/',
  organizationName: 'your-org',
  projectName: 'myapp',
  i18n: {
    defaultLocale: 'ja',
    locales: ['ja', 'en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/your-org/myapp/edit/main/docs/',
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
          // バージョニング
          versions: {
            current: { label: 'Next', path: 'next' },
          },
        },
        blog: {
          showReadingTime: true,
          blogTitle: '開発ブログ',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'MyApp',
      items: [
        { type: 'doc', docId: 'intro', position: 'left', label: 'ドキュメント' },
        { to: '/blog', label: 'ブログ', position: 'left' },
        { type: 'docsVersionDropdown', position: 'right' },
        { type: 'localeDropdown', position: 'right' },
        { href: 'https://github.com/your-org/myapp', label: 'GitHub', position: 'right' },
      ],
    },
    algolia: {
      appId: 'YOUR_APP_ID',
      apiKey: 'YOUR_SEARCH_API_KEY',
      indexName: 'myapp',
    },
    prism: {
      theme: themes.github,
      darkTheme: themes.dracula,
      additionalLanguages: ['bash', 'json', 'yaml', 'sql'],
    },
  },
};

export default config;
```

---

## 4. ADR (Architecture Decision Records)

### 4.1 ADR テンプレート

```markdown
<!-- docs/adr/NNNN-title.md -->
# ADR-NNNN: タイトル

## ステータス

提案中 | 承認済 | 非推奨 | 廃止

## 日付

2025-01-15

## コンテキスト

<!-- どのような状況・課題が意思決定を必要としたか -->

## 決定

<!-- 何を決定したか。具体的に記述 -->

## 検討した選択肢

### 選択肢 A: xxx
- メリット: ...
- デメリット: ...

### 選択肢 B: xxx
- メリット: ...
- デメリット: ...

## 結果

<!-- この決定によってどのような影響が予想されるか -->

## 参考資料

- [リンク](URL)
```

### 4.2 ADR の例

```markdown
# ADR-0001: TypeScript の採用

## ステータス

承認済

## 日付

2025-01-10

## コンテキスト

プロジェクトの規模が拡大し、JavaScript のみでは型安全性の欠如による
ランタイムエラーが増加している。新メンバーのオンボーディング時にも
コードの理解に時間がかかっている。

## 決定

フロントエンド・バックエンド共に TypeScript を採用する。
strict モードを有効にし、any の使用を原則禁止する。

## 検討した選択肢

### 選択肢 A: TypeScript (strict mode)
- メリット: 型安全、IDE 補完、リファクタリング容易
- デメリット: 学習コスト、ビルド時間増加

### 選択肢 B: JavaScript + JSDoc
- メリット: ビルド不要、学習コスト低
- デメリット: 型チェックが不完全、大規模では限界

### 選択肢 C: JavaScript のまま
- メリット: 変更不要
- デメリット: 現状の課題が解決しない

## 結果

- 型エラーの早期検出により、本番障害が減少する見込み
- 初期の移行コスト (約2週間) が発生するが、長期的には開発速度向上
- tsconfig.json を strict: true で統一
```

### 4.3 ADR のディレクトリ構造

```
+------------------------------------------------------------------+
|              ADR ディレクトリ構造                                   |
+------------------------------------------------------------------+
|                                                                  |
|  docs/adr/                                                       |
|    +-- index.md               ← ADR 一覧 (自動生成可)            |
|    +-- 0001-use-typescript.md                                    |
|    +-- 0002-choose-postgresql.md                                 |
|    +-- 0003-adopt-monorepo.md                                    |
|    +-- 0004-api-versioning-strategy.md                           |
|    +-- 0005-authentication-with-jwt.md                           |
|    +-- template.md            ← テンプレート                      |
|                                                                  |
|  命名規則: NNNN-kebab-case-title.md                              |
|  番号は連番。非推奨になっても削除しない (履歴として残す)            |
|                                                                  |
+------------------------------------------------------------------+
```

---

## 5. ドキュメント自動デプロイ

### 5.1 GitHub Pages へのデプロイ (VitePress)

```yaml
# .github/workflows/docs.yml
name: Deploy Docs

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - '.github/workflows/docs.yml'

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # lastUpdated のために全履歴が必要

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm

      - run: pnpm install --frozen-lockfile
      - run: pnpm docs:build

      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/.vitepress/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

### 5.2 Vercel / Netlify へのデプロイ

```toml
# netlify.toml (VitePress)
[build]
  command = "pnpm docs:build"
  publish = "docs/.vitepress/dist"

[build.environment]
  NODE_VERSION = "20"
```

---

## 6. ドキュメント運用のプラクティス

### 6.1 ドキュメントの鮮度を保つ仕組み

```
+------------------------------------------------------------------+
|           ドキュメント鮮度維持の仕組み                               |
+------------------------------------------------------------------+
|                                                                  |
|  [自動化]                                                        |
|  1. PR テンプレートにドキュメント更新チェックリスト                  |
|  2. 変更されたコードに関連する docs/ があれば CI で警告             |
|  3. lastUpdated 表示で古いページを可視化                           |
|  4. API ドキュメントは OpenAPI spec から自動生成                    |
|                                                                  |
|  [文化]                                                          |
|  1. 「コードを書いたらドキュメントも書く」をルール化                 |
|  2. ドキュメントのレビューを PR レビューに含める                    |
|  3. 月次で古いドキュメントの棚卸し                                 |
|  4. ADR は意思決定のタイミングで必ず作成                           |
|                                                                  |
+------------------------------------------------------------------+
```

### 6.2 PR テンプレートへの組み込み

```markdown
<!-- .github/pull_request_template.md (抜粋) -->
## チェックリスト

- [ ] テストを追加/更新した
- [ ] ドキュメントを更新した (該当する場合)
  - [ ] API 変更: docs/api/ を更新
  - [ ] 設定変更: docs/guide/ を更新
  - [ ] アーキテクチャ変更: ADR を作成
```

---

## アンチパターン

### アンチパターン 1: コードと別リポジトリでドキュメントを管理

```
# NG: ドキュメントを別リポジトリに分離
myapp/           ← アプリコード
myapp-docs/      ← ドキュメント (別リポ)
→ コードを変更してもドキュメントの更新を忘れやすい

# OK: 同一リポジトリ内の docs/ ディレクトリ
myapp/
  src/           ← アプリコード
  docs/          ← ドキュメント (同一リポ)
  → 同じ PR でコードとドキュメントを同時に更新
```

**問題点**: 別リポジトリに分離すると、コード変更とドキュメント更新の同期が取れず、ドキュメントが急速に陳腐化する。同一リポジトリにすることで、PR レビューでドキュメント更新も確認でき、CI/CD でデプロイも自動化しやすい。

### アンチパターン 2: ADR を書かない or 後から書く

```
# NG: 「後で書こう」→ 永遠に書かれない
#     3ヶ月後: 「なんでこの技術を選んだんだっけ...」

# OK: 意思決定のタイミングで即座に ADR を書く
#     レビュー中の PR に ADR ドキュメントを含める
#     決定の背景を「今」記録する (記憶が新鮮なうちに)
```

**問題点**: ADR は意思決定の「なぜ」を記録するものであり、実装後に書くと動機や検討した代替案が曖昧になる。意思決定の議論中に ADR のドラフトを作成し、決定と同時に確定させるのが理想。

---

## FAQ

### Q1: VitePress と Docusaurus のどちらを選ぶべきですか？

**A**: プロジェクトの規模と要件で判断する。小〜中規模で高速なビルドが必要なら VitePress。大規模で多言語対応・バージョニング・プラグインエコシステムが必要なら Docusaurus。チームが Vue ベースなら VitePress、React ベースなら Docusaurus/Nextra が自然。迷ったら VitePress から始めて、不足を感じたら移行するのが低リスク。

### Q2: ADR はどのくらいの粒度で書くべきですか？

**A**: 「チームの複数人に影響する技術的意思決定」を基準にする。具体的には、フレームワークの選定、データベースの選択、API 設計方針、認証方式、テスト戦略、デプロイ方式などが対象。変数名やコーディングスタイルのような細かい決定は EditorConfig や ESLint ルールとして記録すれば十分。迷ったら書いた方が良い -- 不要になった ADR は「非推奨」ステータスにすればよい。

### Q3: ドキュメントの自動生成はどこまで有効ですか？

**A**: API リファレンス（OpenAPI/Swagger → ドキュメント生成）や型定義からのインターフェース一覧などは自動生成が非常に有効。一方、アーキテクチャの説明、チュートリアル、ADR などの「なぜ」を説明するドキュメントは手動で書くしかない。理想は「what/how は自動生成、why は手動記述」の組み合わせ。TypeDoc (TypeScript)、Storybook (コンポーネント)、Swagger UI (API) などのツールを活用する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| VitePress | Vue/Vite ベース。高速ビルド。小〜中規模に最適 |
| Docusaurus | React ベース。バージョニング・i18n が強力。大規模向け |
| ADR | アーキテクチャ意思決定の記録。意思決定時に即座に書く |
| 同一リポ管理 | コードと docs/ を同じリポジトリで管理 |
| 自動デプロイ | GitHub Pages / Vercel / Netlify で docs 変更時に自動公開 |
| 鮮度維持 | PR テンプレート + CI 警告 + 月次棚卸しで陳腐化を防止 |
| API ドキュメント | OpenAPI / TypeDoc で自動生成。手動は「なぜ」の部分のみ |

## 次に読むべきガイド

- [プロジェクト標準](./00-project-standards.md) -- EditorConfig / .npmrc の共通設定
- [オンボーディング自動化](./01-onboarding-automation.md) -- セットアップスクリプトと Makefile
- [Dev Container](../02-docker-dev/01-devcontainer.md) -- 開発環境のコンテナ化

## 参考文献

1. **VitePress 公式ドキュメント** -- https://vitepress.dev/ -- VitePress の設定と機能の包括的リファレンス
2. **Docusaurus 公式ドキュメント** -- https://docusaurus.io/ -- Docusaurus の設定・プラグイン・テーマカスタマイズ
3. **ADR GitHub Organization** -- https://adr.github.io/ -- Architecture Decision Records の標準テンプレートとツール
4. **Diátaxis フレームワーク** -- https://diataxis.fr/ -- ドキュメントの4象限分類 (Tutorial / How-to / Reference / Explanation)
