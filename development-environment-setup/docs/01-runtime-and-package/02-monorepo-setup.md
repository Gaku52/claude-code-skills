# モノレポ設定

> Turborepo、Nx、pnpm workspaces を活用したモノレポ構築の実践ガイド。ビルドキャッシュ・タスクオーケストレーションで開発効率を最大化する。

## この章で学ぶこと

1. モノレポの利点と設計パターン、適切なツール選定
2. pnpm workspaces + Turborepo によるモノレポ環境の構築
3. ビルドキャッシュ・リモートキャッシュによる CI 高速化

---

## 1. モノレポとは

### 1.1 モノレポ vs ポリレポ

| 観点 | モノレポ | ポリレポ |
|------|---------|---------|
| リポジトリ数 | 1つ | パッケージごと |
| コード共有 | 容易 | npm publish 必要 |
| 依存管理 | 一元化 | 個別管理 |
| CI/CD | 全体最適化可 | 個別設定 |
| コードレビュー | 横断的変更が1PR | 複数PR必要 |
| スケーラビリティ | ツール支援必要 | 自然にスケール |
| 初期構築コスト | 高い | 低い |

### 1.2 モノレポの構造

```
典型的なモノレポ構造:

my-monorepo/
├── package.json              # ルート (ワークスペース定義)
├── pnpm-workspace.yaml       # pnpm ワークスペース設定
├── turbo.json                # Turborepo 設定
├── .npmrc                    # pnpm 設定
│
├── apps/                     # アプリケーション層
│   ├── web/                  # Next.js フロントエンド
│   │   ├── package.json
│   │   └── src/
│   ├── api/                  # Express バックエンド
│   │   ├── package.json
│   │   └── src/
│   └── mobile/               # React Native
│       ├── package.json
│       └── src/
│
├── packages/                 # 共有パッケージ層
│   ├── ui/                   # 共有UIコンポーネント
│   │   ├── package.json
│   │   └── src/
│   ├── config/               # 共有設定 (ESLint, TS)
│   │   └── package.json
│   ├── utils/                # 共有ユーティリティ
│   │   ├── package.json
│   │   └── src/
│   └── types/                # 共有型定義
│       ├── package.json
│       └── src/
│
└── tooling/                  # ビルドツール設定
    ├── eslint/
    ├── typescript/
    └── tailwind/
```

### 1.3 依存関係グラフ

```
パッケージ間の依存関係:

  apps/web ──→ packages/ui ──→ packages/types
     │              │
     │              └──→ packages/utils
     │
     └──→ packages/utils ──→ packages/types
     │
     └──→ packages/types

  apps/api ──→ packages/utils ──→ packages/types
     │
     └──→ packages/types

  ビルド順序 (Turborepo が自動解決):
  1. packages/types     (依存なし)
  2. packages/utils     (types に依存)
  3. packages/ui        (types, utils に依存)
  4. apps/web, apps/api (並列実行可能)
```

---

## 2. pnpm Workspaces

### 2.1 基本設定

```yaml
# pnpm-workspace.yaml
packages:
  - "apps/*"
  - "packages/*"
  - "tooling/*"
```

```jsonc
// ルート package.json
{
  "name": "my-monorepo",
  "private": true,
  "packageManager": "pnpm@9.1.0",
  "scripts": {
    "dev": "turbo dev",
    "build": "turbo build",
    "lint": "turbo lint",
    "test": "turbo test",
    "clean": "turbo clean",
    "format": "prettier --write \"**/*.{ts,tsx,md}\""
  },
  "devDependencies": {
    "turbo": "^2.0.0",
    "prettier": "^3.2.0"
  }
}
```

### 2.2 内部パッケージの参照

```jsonc
// packages/ui/package.json
{
  "name": "@repo/ui",
  "version": "0.0.0",
  "private": true,
  "exports": {
    ".": "./src/index.ts",
    "./button": "./src/components/button.tsx"
  },
  "dependencies": {
    "@repo/types": "workspace:*",
    "@repo/utils": "workspace:*"
  }
}

// apps/web/package.json
{
  "name": "@repo/web",
  "private": true,
  "dependencies": {
    "@repo/ui": "workspace:*",
    "@repo/utils": "workspace:*",
    "next": "^14.2.0",
    "react": "^18.3.0"
  }
}
```

### 2.3 pnpm 操作コマンド

```bash
# ─── 全パッケージに依存追加 (ルート) ───
pnpm add -Dw turbo prettier

# ─── 特定パッケージに依存追加 ───
pnpm --filter @repo/web add next
pnpm --filter @repo/api add express

# ─── フィルタ実行 ───
pnpm --filter @repo/web dev           # web のみ dev
pnpm --filter "./apps/*" build        # apps 配下全て build
pnpm --filter @repo/ui... build       # ui とその依存先を build

# ─── 全パッケージ操作 ───
pnpm -r exec -- rm -rf node_modules   # 全 node_modules 削除
pnpm install                          # 再インストール
```

---

## 3. Turborepo

### 3.1 設定

```jsonc
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": [
    "**/.env.*local",
    ".env"
  ],
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tsconfig.json", "package.json"],
      "outputs": ["dist/**", ".next/**"],
      "env": ["NODE_ENV"]
    },
    "dev": {
      "dependsOn": ["^build"],
      "persistent": true,
      "cache": false
    },
    "lint": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "eslint.config.*"],
      "outputs": []
    },
    "test": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tests/**", "vitest.config.*"],
      "outputs": ["coverage/**"]
    },
    "clean": {
      "cache": false
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tsconfig.json"],
      "outputs": []
    }
  }
}
```

### 3.2 Turborepo のキャッシュ機構

```
Turborepo のキャッシュフロー:

  turbo build 実行時:

  ┌──────────────────────────────────────┐
  │  1. タスクのハッシュを計算            │
  │     入力: ソースファイル              │
  │           環境変数                    │
  │           依存パッケージのハッシュ    │
  │           turbo.json の設定          │
  │                                      │
  │  2. キャッシュを確認                  │
  │     .turbo/cache/{hash}/             │
  │                                      │
  │  ┌─── キャッシュHIT ────┐            │
  │  │ outputs を復元        │            │
  │  │ (dist/, .next/ 等)    │            │
  │  │ → ビルドスキップ      │            │
  │  │ → 数ミリ秒で完了      │            │
  │  └──────────────────────┘            │
  │                                      │
  │  ┌─── キャッシュMISS ───┐            │
  │  │ ビルド実行            │            │
  │  │ outputs をキャッシュ   │            │
  │  │ → 次回から HIT       │            │
  │  └──────────────────────┘            │
  └──────────────────────────────────────┘
```

### 3.3 リモートキャッシュ

```bash
# Vercel リモートキャッシュ (無料)
npx turbo login
npx turbo link

# または自前のキャッシュサーバー
# turbo.json に追加:
# "remoteCache": {
#   "signature": true
# }
```

```
リモートキャッシュの効果:

  開発者A: turbo build
  ├── packages/types  → ビルド (5s) → キャッシュ保存 ↑
  ├── packages/utils  → ビルド (8s) → キャッシュ保存 ↑
  └── apps/web        → ビルド (30s) → キャッシュ保存 ↑
  合計: 43秒

  開発者B (同じコミット): turbo build
  ├── packages/types  → リモートキャッシュ HIT (0.1s) ↓
  ├── packages/utils  → リモートキャッシュ HIT (0.1s) ↓
  └── apps/web        → リモートキャッシュ HIT (0.5s) ↓
  合計: 0.7秒 (98% 短縮)

  CI: turbo build
  ├── 開発者がすでにビルド済み → 全て HIT
  合計: 1秒以下
```

---

## 4. Nx

### 4.1 基本設定

```bash
# ─── 新規モノレポ作成 ───
npx create-nx-workspace@latest my-monorepo
# → パッケージマネージャー選択: pnpm
# → タイプ選択: integrated / package-based

# ─── 既存リポジトリに追加 ───
npx nx@latest init
```

```jsonc
// nx.json
{
  "$schema": "https://nx.dev/reference/nx-json",
  "targetDefaults": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["production", "^production"],
      "cache": true
    },
    "test": {
      "inputs": ["default", "^production"],
      "cache": true
    },
    "lint": {
      "inputs": ["default", "{workspaceRoot}/.eslintrc.json"],
      "cache": true
    }
  },
  "namedInputs": {
    "default": ["{projectRoot}/**/*", "sharedGlobals"],
    "production": ["default", "!{projectRoot}/**/*.spec.*"],
    "sharedGlobals": []
  },
  "defaultBase": "main"
}
```

### 4.2 Turborepo vs Nx 比較

| 特徴 | Turborepo | Nx |
|------|-----------|-----|
| 設計思想 | シンプル・軽量 | フル機能・統合型 |
| 設定ファイル | turbo.json のみ | nx.json + project.json |
| キャッシュ | ローカル + リモート | ローカル + Nx Cloud |
| 依存グラフ可視化 | なし | `nx graph` (ブラウザ) |
| コード生成 | なし | `nx generate` |
| 影響範囲分析 | `turbo --filter` | `nx affected` |
| プラグイン | なし | 豊富 (React, Next等) |
| 学習コスト | 低い | 中〜高い |
| 推奨規模 | 小〜中 | 中〜大 |

---

## 5. 共有設定パッケージ

### 5.1 TypeScript 設定共有

```jsonc
// packages/config/tsconfig/base.json
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "compilerOptions": {
    "strict": true,
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  }
}

// packages/config/tsconfig/react.json
{
  "extends": "./base.json",
  "compilerOptions": {
    "jsx": "react-jsx",
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  }
}

// apps/web/tsconfig.json (利用側)
{
  "extends": "@repo/config/tsconfig/react.json",
  "compilerOptions": {
    "outDir": "./dist"
  },
  "include": ["src/**/*"]
}
```

### 5.2 ESLint 設定共有

```javascript
// packages/config/eslint/base.js
import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      "@typescript-eslint/no-unused-vars": ["error", {
        argsIgnorePattern: "^_"
      }],
      "@typescript-eslint/no-explicit-any": "error",
    }
  }
];

// apps/web/eslint.config.js (利用側)
import baseConfig from "@repo/config/eslint/base.js";
import nextPlugin from "@next/eslint-plugin-next";

export default [
  ...baseConfig,
  // Next.js 固有ルール追加
];
```

---

## 6. CI/CD 設定

### 6.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2  # turbo の差分検出に必要

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      # Turborepo リモートキャッシュ
      - run: pnpm turbo build lint test typecheck
        env:
          TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
          TURBO_TEAM: ${{ vars.TURBO_TEAM }}
```

---

## 7. アンチパターン

### 7.1 全パッケージの依存をルートに入れる

```
❌ アンチパターン: ルートの package.json に全依存を集約

  package.json (root):
    "dependencies": {
      "react": "^18.3.0",
      "express": "^4.18.0",
      "next": "^14.2.0",
      ...全ての依存
    }

問題:
  - どのパッケージが何に依存しているか不明
  - 不要な依存がインストールされる
  - バージョン更新の影響範囲が不明

✅ 正しいアプローチ:
  - 各パッケージの package.json に必要な依存のみ記述
  - ルートには turbo, prettier 等のツールのみ
  - pnpm の厳密な依存解決に任せる
```

### 7.2 パッケージ間で循環依存を作る

```
❌ アンチパターン: パッケージ間の循環参照

  @repo/ui → @repo/utils → @repo/ui  (循環!)

問題:
  - ビルド順序が決定できない
  - TypeScript の型解決が無限ループ
  - Turborepo/Nx のキャッシュが効かない

✅ 正しいアプローチ:
  - 共通部分を別パッケージに抽出
  - @repo/ui → @repo/shared ← @repo/utils
  - nx graph / turbo で依存グラフを定期確認
```

---

## 8. FAQ

### Q1: Turborepo と Nx、どちらを選ぶべき？

**A:** 以下の基準で判断する。
- **Turborepo**: パッケージ数が 10 以下、シンプルなキャッシュとタスク実行が目的、設定を最小限にしたい場合。Vercel エコシステム（Next.js）との相性が良い。
- **Nx**: パッケージ数が多い大規模プロジェクト、コード生成や影響範囲分析が必要、プラグインエコシステムを活用したい場合。

迷ったら Turborepo から始めて、必要に応じて Nx に移行するのが低リスク。

### Q2: モノレポの git clone が遅い場合の対処法は？

**A:** 以下の方法がある。
1. `git clone --depth 1` でシャロークローン
2. `git clone --filter=blob:none` でパーシャルクローン
3. GitHub Actions では `fetch-depth: 0` ではなく必要最小限の深さを指定
4. Git LFS を使う大きなファイルがある場合は `git lfs install --skip-smudge`

### Q3: 内部パッケージはビルドすべき？ソースのまま参照すべき？

**A:** 2つのアプローチがある。
- **ビルドする方式**: `tsc` でコンパイルして `dist/` を参照。型安全性が高く、消費側のビルドが速い。
- **ソース参照方式**: `exports` で `src/index.ts` を直接指定。ビルドステップ不要で開発が速い。Next.js の `transpilePackages` や Vite の設定で消費側がトランスパイル。

小〜中規模ならソース参照方式が手軽。大規模ではビルド方式の方が CI が安定する。

---

## 9. まとめ

| 構成要素 | 推奨 | 備考 |
|---------|------|------|
| パッケージマネージャー | pnpm | ワークスペースに最適 |
| タスクランナー | Turborepo | シンプルで高速 |
| 大規模向け | Nx | 影響範囲分析・生成器 |
| 設定共有 | `packages/config/` | ESLint, TS, Tailwind |
| CI キャッシュ | Turborepo Remote Cache | Vercel 無料枠あり |
| パッケージ参照 | `workspace:*` | 内部パッケージ管理 |
| バージョン統一 | Corepack | packageManager フィールド |

---

## 次に読むべきガイド

- [01-package-managers.md](./01-package-managers.md) — パッケージマネージャーの詳細
- [03-linter-formatter.md](./03-linter-formatter.md) — モノレポでのLinter/Formatter設定共有
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) — チーム標準の設定

---

## 参考文献

1. **Turborepo Handbook** — https://turbo.build/repo/docs/handbook — Turborepo 公式ハンドブック。設計パターン解説。
2. **Nx Documentation** — https://nx.dev/getting-started/intro — Nx 公式入門ガイド。
3. **pnpm Workspaces** — https://pnpm.io/workspaces — pnpm ワークスペースの公式ドキュメント。
4. **Monorepo Tools** — https://monorepo.tools/ — モノレポツールの比較サイト。客観的なベンチマークあり。
