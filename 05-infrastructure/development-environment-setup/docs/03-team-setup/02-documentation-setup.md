# ドキュメント環境 (Documentation Setup)

> VitePress / Docusaurus によるドキュメントサイトの構築と、ADR (Architecture Decision Records) による意思決定の記録を通じて、チームの知識を体系的に管理する手法を学ぶ。

## この章で学ぶこと

1. **VitePress / Docusaurus の導入と設定** -- Markdown ベースのドキュメントサイトを構築し、自動デプロイまでのパイプラインを整備する
2. **ADR (Architecture Decision Records) の運用** -- アーキテクチャの意思決定を記録し、「なぜこの設計にしたのか」を追跡可能にする
3. **ドキュメント運用のベストプラクティス** -- ドキュメントの鮮度を保ち、コードと一緒にメンテナンスする文化を構築する
4. **API ドキュメントの自動生成** -- OpenAPI / TypeDoc / Storybook を活用して、常に最新のリファレンスを自動生成する
5. **Diataxis フレームワークによるドキュメント設計** -- Tutorial / How-to / Reference / Explanation の4象限でドキュメントを体系化する

---

## 1. ドキュメントツールの選択

### 1.1 ツール比較

| 項目 | VitePress | Docusaurus | Nextra | GitBook | Starlight |
|------|-----------|------------|--------|---------|-----------|
| フレームワーク | Vue 3 / Vite | React / Webpack | Next.js | SaaS | Astro |
| ビルド速度 | 非常に高速 | 中 | 高速 | N/A | 高速 |
| カスタマイズ | Vue コンポーネント | React コンポーネント | React | 限定的 | Astro コンポーネント |
| 多言語 (i18n) | 対応 | 強力な対応 | 対応 | 対応 | 対応 |
| バージョニング | 手動 | 標準対応 | 手動 | 対応 | 手動 |
| 検索 | 内蔵(miniSearch) | Algolia 統合 | Flexsearch | 内蔵 | Pagefind |
| デプロイ | 静的ホスティング | 静的ホスティング | Vercel推奨 | SaaS | 静的ホスティング |
| 学習コスト | 低 | 中 | 低 | 最低 | 低 |
| 適用場面 | OSS / 技術文書 | 大規模プロジェクト | Next.js利用者 | 非エンジニア含む | 高速サイト |

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
|  パフォーマンス最優先?                                              |
|    YES → Starlight (Astro ベース)                                |
|                                                                  |
|  非エンジニアも編集する?                                            |
|    YES → GitBook / Notion                                        |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.3 Diataxis フレームワーク

ドキュメントを効果的に構造化するためのフレームワーク。ドキュメントを4つの象限に分類する。

```
+------------------------------------------------------------------+
|              Diataxis フレームワーク                                |
+------------------------------------------------------------------+
|                                                                  |
|       学習 (Learning)          |     実践 (Doing)                 |
|  ─────────────────────────────|──────────────────────────────    |
|                               |                                  |
|   TUTORIALS                   |   HOW-TO GUIDES                  |
|   チュートリアル               |   ハウツーガイド                   |
|   ・学習体験を提供             |   ・特定タスクの手順               |
|   ・初心者向け                 |   ・問題解決型                    |
|   ・ステップバイステップ       |   ・結果指向                      |
|   例: 初めてのデプロイ         |   例: メール送信機能の追加         |
|                               |                                  |
|  ─────────────────────────────|──────────────────────────────    |
|                               |                                  |
|   EXPLANATION                 |   REFERENCE                      |
|   説明                         |   リファレンス                    |
|   ・背景・コンテキストの提供   |   ・正確な技術情報                |
|   ・概念の理解                 |   ・自動生成可能                  |
|   ・「なぜ」を説明             |   ・API仕様、型定義               |
|   例: アーキテクチャ解説       |   例: API エンドポイント一覧       |
|                               |                                  |
|       理解 (Understanding)    |     情報 (Information)            |
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
#       index.ts
#       style.css
#   index.md        -- トップページ
#   guide/
#     getting-started.md
#     architecture.md
#     dev-setup.md
#     coding-standards.md
#     testing.md
#     deployment.md
#   api/
#     overview.md
#     authentication.md
#     endpoints.md
#   adr/
#     index.md
#     0001-use-typescript.md
#     0002-choose-postgresql.md
#     template.md
#   tutorials/
#     first-feature.md
#     first-deploy.md
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
    ['meta', { name: 'theme-color', content: '#3eaf7c' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'ja_JP' }],
  ],

  // クリーン URL (/guide/getting-started.html → /guide/getting-started)
  cleanUrls: true,

  // 最終更新日時の表示 (git log ベース)
  lastUpdated: true,

  // sitemap 自動生成
  sitemap: {
    hostname: 'https://docs.example.com',
  },

  themeConfig: {
    logo: '/logo.svg',

    nav: [
      { text: 'ガイド', link: '/guide/getting-started' },
      { text: 'API', link: '/api/overview' },
      { text: 'ADR', link: '/adr/' },
      {
        text: 'リソース',
        items: [
          { text: 'チュートリアル', link: '/tutorials/first-feature' },
          { text: 'FAQ', link: '/faq' },
          { text: 'Changelog', link: '/changelog' },
        ],
      },
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
        {
          text: '運用',
          items: [
            { text: 'モニタリング', link: '/guide/monitoring' },
            { text: 'トラブルシューティング', link: '/guide/troubleshooting' },
            { text: 'セキュリティ', link: '/guide/security' },
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
            { text: 'エラーコード', link: '/api/error-codes' },
            { text: 'レートリミット', link: '/api/rate-limiting' },
          ],
        },
      ],
      '/tutorials/': [
        {
          text: 'チュートリアル',
          items: [
            { text: '初めての機能追加', link: '/tutorials/first-feature' },
            { text: '初めてのデプロイ', link: '/tutorials/first-deploy' },
            { text: 'テストの書き方', link: '/tutorials/writing-tests' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/your-org/myapp' },
      { icon: 'slack', link: 'https://your-org.slack.com/' },
    ],

    search: {
      provider: 'local', // miniSearch 内蔵検索
      options: {
        translations: {
          button: { buttonText: '検索', buttonAriaLabel: 'サイト内検索' },
          modal: {
            noResultsText: '結果が見つかりません',
            resetButtonTitle: 'リセット',
            footer: { selectText: '選択', navigateText: '移動', closeText: '閉じる' },
          },
        },
      },
    },

    editLink: {
      pattern: 'https://github.com/your-org/myapp/edit/main/docs/:path',
      text: 'このページを編集する',
    },

    lastUpdated: {
      text: '最終更新',
      formatOptions: {
        dateStyle: 'medium',
        timeStyle: 'short',
      },
    },

    footer: {
      message: 'MIT License',
      copyright: 'Copyright (c) 2025 Your Org',
    },

    // 目次の深さ設定
    outline: {
      level: [2, 3],
      label: '目次',
    },

    // 前後ページナビゲーション
    docFooter: {
      prev: '前のページ',
      next: '次のページ',
    },
  },

  markdown: {
    lineNumbers: true, // コードブロックに行番号
    math: true, // 数式サポート (KaTeX)
    image: {
      lazyLoading: true,
    },
    // カスタムコンテナ
    container: {
      tipLabel: 'ヒント',
      warningLabel: '注意',
      dangerLabel: '危険',
      infoLabel: '情報',
      detailsLabel: '詳細',
    },
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

### 2.4 VitePress のカスタムテーマ

```typescript
// docs/.vitepress/theme/index.ts
import { h } from 'vue';
import type { Theme } from 'vitepress';
import DefaultTheme from 'vitepress/theme';
import './style.css';

export default {
  extends: DefaultTheme,
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      // カスタムスロット
      // 'doc-before': () => h(Banner),
      // 'doc-after': () => h(Feedback),
    });
  },
  enhanceApp({ app, router, siteData }) {
    // カスタムコンポーネントの登録
    // app.component('CustomComponent', CustomComponent);
  },
} satisfies Theme;
```

```css
/* docs/.vitepress/theme/style.css */

/* カスタムカラーテーマ */
:root {
  --vp-c-brand-1: #3eaf7c;
  --vp-c-brand-2: #359968;
  --vp-c-brand-3: #2c8155;
  --vp-c-brand-soft: rgba(62, 175, 124, 0.14);
}

/* ダークモード */
.dark {
  --vp-c-brand-1: #5dd3a0;
  --vp-c-brand-2: #49c78d;
  --vp-c-brand-3: #3eaf7c;
}

/* カスタムコンテナのスタイル */
.custom-block.tip {
  border-color: var(--vp-c-brand-1);
}

/* コードブロックのフォント */
:root {
  --vp-code-font-size: 0.875em;
}
```

### 2.5 VitePress のトップページ

```markdown
---
# docs/index.md
layout: home

hero:
  name: "MyApp"
  text: "開発者ドキュメント"
  tagline: "MyApp の開発に必要な全ての情報"
  image:
    src: /logo.svg
    alt: MyApp
  actions:
    - theme: brand
      text: クイックスタート
      link: /guide/getting-started
    - theme: alt
      text: API リファレンス
      link: /api/overview

features:
  - icon: 🚀
    title: クイックスタート
    details: 5分で開発環境をセットアップし、最初のコードを書く
    link: /guide/getting-started
  - icon: 📖
    title: 開発ガイド
    details: コーディング規約、テスト戦略、デプロイ手順
    link: /guide/coding-standards
  - icon: 🔌
    title: API リファレンス
    details: REST API の完全なリファレンスドキュメント
    link: /api/overview
  - icon: 🏗️
    title: アーキテクチャ
    details: システム設計と意思決定の記録 (ADR)
    link: /guide/architecture
---
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
      additionalLanguages: ['bash', 'json', 'yaml', 'sql', 'docker', 'nginx'],
    },
    // アナウンスメントバー
    announcementBar: {
      id: 'v2_announcement',
      content: 'v2.0 がリリースされました! <a href="/blog/v2-release">詳細はこちら</a>',
      backgroundColor: '#fafbfc',
      textColor: '#091E42',
      isCloseable: true,
    },
  },

  plugins: [
    // OpenAPI ドキュメント自動生成
    [
      'docusaurus-plugin-openapi-docs',
      {
        id: 'api',
        docsPluginId: 'classic',
        config: {
          api: {
            specPath: 'api/openapi.yaml',
            outputDir: 'docs/api-reference',
          },
        },
      },
    ],
  ],
};

export default config;
```

### 3.3 Docusaurus のバージョニング

```bash
# 現在のドキュメントを v1.0.0 としてスナップショット
npx docusaurus docs:version 1.0.0

# ディレクトリ構造:
# docs/
#   intro.md                   ← 最新 (next)
# versioned_docs/
#   version-1.0.0/
#     intro.md                 ← v1.0.0 時点のスナップショット
# versioned_sidebars/
#   version-1.0.0-sidebars.json
# versions.json                ← ["1.0.0"]
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
|    +-- 0006-adopt-graphql.md                                     |
|    +-- 0007-use-redis-for-caching.md                             |
|    +-- 0008-container-orchestration.md                           |
|    +-- template.md            ← テンプレート                      |
|                                                                  |
|  命名規則: NNNN-kebab-case-title.md                              |
|  番号は連番。非推奨になっても削除しない (履歴として残す)            |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.4 ADR 自動生成スクリプト

```bash
#!/bin/bash
# scripts/new-adr.sh
# 新しい ADR を作成するスクリプト

set -euo pipefail

ADR_DIR="docs/adr"
TEMPLATE="$ADR_DIR/template.md"

# 次の番号を取得
LAST_NUM=$(ls "$ADR_DIR"/*.md 2>/dev/null | grep -oP '\d{4}' | sort -rn | head -1 || echo "0000")
NEXT_NUM=$(printf "%04d" $((10#$LAST_NUM + 1)))

# タイトルの入力
if [ -z "${1:-}" ]; then
  echo -n "ADR タイトルを入力してください: "
  read -r TITLE
else
  TITLE="$*"
fi

# kebab-case に変換
KEBAB=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g' | sed 's/[^a-z0-9-]//g')
FILENAME="$ADR_DIR/${NEXT_NUM}-${KEBAB}.md"

# テンプレートからコピー
if [ -f "$TEMPLATE" ]; then
  sed "s/NNNN/$NEXT_NUM/g; s/タイトル/$TITLE/g" "$TEMPLATE" > "$FILENAME"
else
  cat > "$FILENAME" << EOF
# ADR-${NEXT_NUM}: ${TITLE}

## ステータス

提案中

## 日付

$(date +%Y-%m-%d)

## コンテキスト

<!-- どのような状況・課題が意思決定を必要としたか -->

## 決定

<!-- 何を決定したか。具体的に記述 -->

## 検討した選択肢

### 選択肢 A:
- メリット:
- デメリット:

### 選択肢 B:
- メリット:
- デメリット:

## 結果

<!-- この決定によってどのような影響が予想されるか -->

## 参考資料

-
EOF
fi

echo "作成: $FILENAME"
echo "エディタで開きます..."
${EDITOR:-code} "$FILENAME"
```

### 4.5 ADR 一覧の自動生成

```bash
#!/bin/bash
# scripts/update-adr-index.sh
# ADR の一覧ページを自動生成する

set -euo pipefail

ADR_DIR="docs/adr"
INDEX_FILE="$ADR_DIR/index.md"

cat > "$INDEX_FILE" << 'HEADER'
# Architecture Decision Records

アーキテクチャに関する意思決定の記録一覧。

| 番号 | タイトル | ステータス | 日付 |
|------|---------|----------|------|
HEADER

for file in "$ADR_DIR"/[0-9][0-9][0-9][0-9]-*.md; do
  [ -f "$file" ] || continue
  BASENAME=$(basename "$file" .md)
  NUM=$(echo "$BASENAME" | grep -oP '^\d{4}')
  TITLE=$(head -1 "$file" | sed 's/^# ADR-[0-9]*: //')
  STATUS=$(grep -A1 "^## ステータス" "$file" | tail -1 | tr -d '[:space:]')
  DATE=$(grep -A1 "^## 日付" "$file" | tail -1 | tr -d '[:space:]')

  echo "| $NUM | [$TITLE](./$BASENAME) | $STATUS | $DATE |" >> "$INDEX_FILE"
done

echo ""
echo "ADR 一覧を更新しました: $INDEX_FILE"
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

      - uses: pnpm/action-setup@v4

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

# リダイレクト設定
[[redirects]]
  from = "/docs/*"
  to = "/:splat"
  status = 301
```

```jsonc
// vercel.json (VitePress)
{
  "buildCommand": "pnpm docs:build",
  "outputDirectory": "docs/.vitepress/dist",
  "framework": null,
  "rewrites": [
    { "source": "/(.*)", "destination": "/$1" }
  ]
}
```

### 5.3 Cloudflare Pages へのデプロイ

```yaml
# .github/workflows/docs-cloudflare.yml
name: Deploy Docs to Cloudflare Pages

on:
  push:
    branches: [main]
    paths:
      - 'docs/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm

      - run: pnpm install --frozen-lockfile
      - run: pnpm docs:build

      - uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CLOUDFLARE_API_TOKEN }}
          accountId: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
          projectName: myapp-docs
          directory: docs/.vitepress/dist
```

---

## 6. API ドキュメントの自動生成

### 6.1 OpenAPI (Swagger) からの生成

```yaml
# api/openapi.yaml
openapi: 3.1.0
info:
  title: MyApp API
  version: 1.0.0
  description: MyApp の REST API ドキュメント

servers:
  - url: https://api.example.com/v1
    description: Production
  - url: http://localhost:3000/api/v1
    description: Development

paths:
  /users:
    get:
      summary: ユーザー一覧取得
      operationId: listUsers
      tags:
        - Users
      parameters:
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: '#/components/schemas/User'
                  pagination:
                    $ref: '#/components/schemas/Pagination'

    post:
      summary: ユーザー作成
      operationId: createUser
      tags:
        - Users
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateUserRequest'
      responses:
        '201':
          description: 作成成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '422':
          description: バリデーションエラー
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValidationError'

components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          format: uuid
        name:
          type: string
        email:
          type: string
          format: email
        role:
          type: string
          enum: [admin, member, viewer]
        createdAt:
          type: string
          format: date-time

    CreateUserRequest:
      type: object
      required:
        - name
        - email
      properties:
        name:
          type: string
          minLength: 1
          maxLength: 100
        email:
          type: string
          format: email
        role:
          type: string
          enum: [admin, member, viewer]
          default: member

    Pagination:
      type: object
      properties:
        page:
          type: integer
        limit:
          type: integer
        total:
          type: integer
        totalPages:
          type: integer

    ValidationError:
      type: object
      properties:
        message:
          type: string
        errors:
          type: array
          items:
            type: object
            properties:
              field:
                type: string
              message:
                type: string

  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - bearerAuth: []
```

### 6.2 TypeDoc による TypeScript ドキュメント生成

```jsonc
// typedoc.json
{
  "entryPoints": ["src/index.ts"],
  "entryPointStrategy": "expand",
  "out": "docs/api-reference",
  "plugin": ["typedoc-plugin-markdown"],
  "theme": "markdown",
  "readme": "none",
  "excludePrivate": true,
  "excludeProtected": true,
  "excludeInternal": true,
  "includeVersion": true,
  "categorizeByGroup": true
}
```

```bash
# TypeDoc の実行
npx typedoc

# VitePress と統合する場合
# docs/api-reference/ に Markdown が生成される
```

### 6.3 Storybook によるコンポーネントドキュメント

```typescript
// src/components/Button/Button.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';

const meta: Meta<typeof Button> = {
  title: 'Components/Button',
  component: Button,
  tags: ['autodocs'], // 自動ドキュメント生成
  argTypes: {
    variant: {
      control: 'select',
      options: ['primary', 'secondary', 'danger'],
      description: 'ボタンのスタイルバリアント',
    },
    size: {
      control: 'select',
      options: ['sm', 'md', 'lg'],
      description: 'ボタンのサイズ',
    },
    disabled: {
      control: 'boolean',
      description: '無効状態',
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Primary: Story = {
  args: {
    variant: 'primary',
    children: 'ボタン',
  },
};

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    children: 'ボタン',
  },
};

export const Disabled: Story = {
  args: {
    variant: 'primary',
    children: 'ボタン',
    disabled: true,
  },
};
```

---

## 7. ドキュメント運用のプラクティス

### 7.1 ドキュメントの鮮度を保つ仕組み

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
|  5. TypeDoc でコードから型ドキュメントを自動生成                    |
|  6. Storybook で UI コンポーネントを自動ドキュメント化              |
|                                                                  |
|  [文化]                                                          |
|  1. 「コードを書いたらドキュメントも書く」をルール化                 |
|  2. ドキュメントのレビューを PR レビューに含める                    |
|  3. 月次で古いドキュメントの棚卸し                                 |
|  4. ADR は意思決定のタイミングで必ず作成                           |
|  5. README は常に最新の状態を維持                                  |
|                                                                  |
+------------------------------------------------------------------+
```

### 7.2 PR テンプレートへの組み込み

```markdown
<!-- .github/pull_request_template.md (抜粋) -->
## チェックリスト

- [ ] テストを追加/更新した
- [ ] ドキュメントを更新した (該当する場合)
  - [ ] API 変更: docs/api/ を更新
  - [ ] 設定変更: docs/guide/ を更新
  - [ ] アーキテクチャ変更: ADR を作成
  - [ ] コンポーネント変更: Storybook を更新
- [ ] CHANGELOG.md を更新した (ユーザー向け変更の場合)
```

### 7.3 ドキュメント品質チェックの自動化

```yaml
# .github/workflows/docs-check.yml
name: Docs Check

on:
  pull_request:
    paths:
      - 'docs/**'
      - 'src/**'

jobs:
  check-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # ドキュメントのビルドチェック
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
      - run: pnpm install --frozen-lockfile
      - run: pnpm docs:build

      # リンク切れチェック
      - name: Check broken links
        run: npx linkinator docs/.vitepress/dist --recurse --skip "^https?"

      # src/ の変更に対して docs/ の変更がないか警告
      - name: Check docs update
        run: |
          SRC_CHANGED=$(git diff --name-only origin/main...HEAD -- 'src/' | wc -l)
          DOCS_CHANGED=$(git diff --name-only origin/main...HEAD -- 'docs/' | wc -l)

          if [ "$SRC_CHANGED" -gt 0 ] && [ "$DOCS_CHANGED" -eq 0 ]; then
            echo "::warning::src/ に変更がありますが、docs/ は更新されていません。ドキュメントの更新が必要か確認してください。"
          fi
```

### 7.4 古いドキュメントの検知

```bash
#!/bin/bash
# scripts/stale-docs.sh
# 90日以上更新されていないドキュメントを一覧表示する

set -euo pipefail

DAYS=${1:-90}
STALE_DATE=$(date -d "-${DAYS} days" +%s 2>/dev/null || date -v-${DAYS}d +%s)
COUNT=0

echo "=== ${DAYS}日以上更新されていないドキュメント ==="
echo ""

while IFS= read -r file; do
  LAST_COMMIT=$(git log -1 --format="%ct" -- "$file" 2>/dev/null || echo "0")

  if [ "$LAST_COMMIT" -lt "$STALE_DATE" ]; then
    LAST_DATE=$(git log -1 --format="%ci" -- "$file" 2>/dev/null | cut -d' ' -f1)
    LAST_AUTHOR=$(git log -1 --format="%an" -- "$file" 2>/dev/null)
    echo "  $file"
    echo "    最終更新: $LAST_DATE ($LAST_AUTHOR)"
    ((COUNT++))
  fi
done < <(find docs -name "*.md" -type f)

echo ""
echo "合計: ${COUNT} ファイル"
```

---

## 8. ドキュメントのディレクトリ構造テンプレート

### 8.1 小規模プロジェクト

```
docs/
  README.md               ← プロジェクト概要
  CONTRIBUTING.md          ← 貢献ガイド
  CHANGELOG.md             ← 変更履歴
  guide/
    getting-started.md     ← クイックスタート
    dev-setup.md           ← 開発環境セットアップ
  api/
    overview.md            ← API 概要
  adr/
    0001-xxx.md            ← ADR
```

### 8.2 中規模プロジェクト

```
docs/
  .vitepress/
    config.ts
    theme/
  index.md                 ← トップページ
  guide/
    getting-started.md
    architecture.md
    dev-setup.md
    coding-standards.md
    testing.md
    deployment.md
  api/
    overview.md
    authentication.md
    endpoints.md
    error-codes.md
  tutorials/
    first-feature.md
    first-deploy.md
  adr/
    index.md
    0001-xxx.md
    template.md
  reference/
    environment-variables.md
    configuration.md
```

### 8.3 大規模プロジェクト

```
docs/
  .vitepress/
    config.ts
    theme/
  index.md
  guide/
    getting-started.md
    architecture.md
    dev-setup.md
    coding-standards.md
    testing/
      unit-testing.md
      integration-testing.md
      e2e-testing.md
    deployment/
      staging.md
      production.md
      rollback.md
  api/
    overview.md
    authentication.md
    v1/
      users.md
      orders.md
      products.md
    v2/
      users.md
    error-codes.md
    rate-limiting.md
  tutorials/
    beginner/
      first-feature.md
      first-deploy.md
    advanced/
      custom-plugin.md
      performance-tuning.md
  how-to/
    add-new-endpoint.md
    run-migrations.md
    debug-production.md
    setup-monitoring.md
  explanation/
    data-model.md
    auth-flow.md
    caching-strategy.md
  adr/
    index.md
    0001-xxx.md
    template.md
  reference/
    environment-variables.md
    configuration.md
    cli-commands.md
  operations/
    runbooks/
      incident-response.md
      database-failover.md
    monitoring.md
    alerting.md
  contributing/
    development-workflow.md
    code-review.md
    release-process.md
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

### アンチパターン 3: ドキュメントの更新を任意にする

```
# NG: ドキュメント更新はオプション
# → 誰も更新しなくなる

# OK: CI でドキュメントの鮮度を検証
# → src/ 変更時に docs/ の変更がなければ警告
# → PR テンプレートにチェックリスト
# → レビューでドキュメント更新を確認
```

**問題点**: ドキュメント更新を任意にすると、「今は急いでいるから後で」が積み重なり、ドキュメントとコードの乖離が拡大する。CI での警告やPR テンプレートのチェックリストで半強制的に更新を促す仕組みが必要。

### アンチパターン 4: 全てを手動で書く

```
# NG: API ドキュメントを手動で書く
# → コード変更のたびに手動更新 → 乖離

# OK: 自動生成 + 手動の組み合わせ
# 自動生成: API リファレンス (OpenAPI → ドキュメント)
# 自動生成: 型定義 (TypeDoc)
# 自動生成: UI コンポーネント (Storybook)
# 手動記述: アーキテクチャ説明、チュートリアル、ADR
```

**問題点**: 自動生成可能な情報（API エンドポイント、型定義、コンポーネントの Props）を手動で書くと、コードとの乖離が不可避。「what/how は自動生成、why は手動」の原則を守ることで、メンテナンスコストを最小化できる。

---

## FAQ

### Q1: VitePress と Docusaurus のどちらを選ぶべきですか？

**A**: プロジェクトの規模と要件で判断する。小〜中規模で高速なビルドが必要なら VitePress。大規模で多言語対応・バージョニング・プラグインエコシステムが必要なら Docusaurus。チームが Vue ベースなら VitePress、React ベースなら Docusaurus/Nextra が自然。迷ったら VitePress から始めて、不足を感じたら移行するのが低リスク。

### Q2: ADR はどのくらいの粒度で書くべきですか？

**A**: 「チームの複数人に影響する技術的意思決定」を基準にする。具体的には、フレームワークの選定、データベースの選択、API 設計方針、認証方式、テスト戦略、デプロイ方式などが対象。変数名やコーディングスタイルのような細かい決定は EditorConfig や ESLint ルールとして記録すれば十分。迷ったら書いた方が良い -- 不要になった ADR は「非推奨」ステータスにすればよい。

### Q3: ドキュメントの自動生成はどこまで有効ですか？

**A**: API リファレンス（OpenAPI/Swagger → ドキュメント生成）や型定義からのインターフェース一覧などは自動生成が非常に有効。一方、アーキテクチャの説明、チュートリアル、ADR などの「なぜ」を説明するドキュメントは手動で書くしかない。理想は「what/how は自動生成、why は手動記述」の組み合わせ。TypeDoc (TypeScript)、Storybook (コンポーネント)、Swagger UI (API) などのツールを活用する。

### Q4: ドキュメントの検索はどう実装しますか？

**A**: ドキュメントツールにより選択肢が異なる。

- **VitePress**: 内蔵の miniSearch (設定不要)。小〜中規模で十分な精度。
- **Docusaurus**: Algolia DocSearch (無料枠あり、OSS は無料)。大規模サイトに最適。
- **Starlight**: Pagefind (ビルド時に検索インデックスを生成)。サーバー不要。
- **自前実装**: FlexSearch や Lunr.js をクライアントサイドで使用。

### Q5: ドキュメントの多言語対応はどう進めますか？

**A**: Docusaurus は i18n サポートが最も充実しており、`docusaurus write-translations` コマンドで翻訳ファイルの雛形を自動生成できる。VitePress では手動でディレクトリを分ける。翻訳作業自体は Crowdin や Weblate などの翻訳管理サービスと統合するのが効率的。まずは英語で書き、需要に応じて日本語化する（またはその逆）のが現実的。

---

## まとめ

| 項目 | 要点 |
|------|------|
| VitePress | Vue/Vite ベース。高速ビルド。小〜中規模に最適 |
| Docusaurus | React ベース。バージョニング・i18n が強力。大規模向け |
| Starlight | Astro ベース。高速。コンテンツ重視のサイトに最適 |
| ADR | アーキテクチャ意思決定の記録。意思決定時に即座に書く |
| Diataxis | ドキュメントを4象限 (Tutorial/How-to/Reference/Explanation) に分類 |
| 同一リポ管理 | コードと docs/ を同じリポジトリで管理 |
| 自動デプロイ | GitHub Pages / Vercel / Netlify / Cloudflare Pages で自動公開 |
| API ドキュメント | OpenAPI / TypeDoc で自動生成。手動は「なぜ」の部分のみ |
| Storybook | UI コンポーネントの視覚的ドキュメント |
| 鮮度維持 | PR テンプレート + CI 警告 + 月次棚卸しで陳腐化を防止 |
| 品質チェック | リンク切れ検知、ビルドチェック、更新漏れ警告を CI で自動化 |

## 次に読むべきガイド

- [プロジェクト標準](./00-project-standards.md) -- EditorConfig / .npmrc の共通設定
- [オンボーディング自動化](./01-onboarding-automation.md) -- セットアップスクリプトと Makefile
- [Dev Container](../02-docker-dev/01-devcontainer.md) -- 開発環境のコンテナ化

## 参考文献

1. **VitePress 公式ドキュメント** -- https://vitepress.dev/ -- VitePress の設定と機能の包括的リファレンス
2. **Docusaurus 公式ドキュメント** -- https://docusaurus.io/ -- Docusaurus の設定・プラグイン・テーマカスタマイズ
3. **ADR GitHub Organization** -- https://adr.github.io/ -- Architecture Decision Records の標準テンプレートとツール
4. **Diataxis フレームワーク** -- https://diataxis.fr/ -- ドキュメントの4象限分類 (Tutorial / How-to / Reference / Explanation)
5. **Starlight 公式ドキュメント** -- https://starlight.astro.build/ -- Astro ベースのドキュメントフレームワーク
6. **Storybook 公式** -- https://storybook.js.org/ -- UI コンポーネントの開発・テスト・ドキュメント化
7. **TypeDoc** -- https://typedoc.org/ -- TypeScript コードからのドキュメント自動生成
8. **Algolia DocSearch** -- https://docsearch.algolia.com/ -- ドキュメントサイト向け検索サービス
