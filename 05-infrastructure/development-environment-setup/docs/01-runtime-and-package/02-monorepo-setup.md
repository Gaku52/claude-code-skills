# モノレポ設定

> Turborepo、Nx、pnpm workspaces を活用したモノレポ構築の実践ガイド。ビルドキャッシュ・タスクオーケストレーションで開発効率を最大化する。

## この章で学ぶこと

1. モノレポの利点と設計パターン、適切なツール選定
2. pnpm workspaces + Turborepo によるモノレポ環境の構築
3. ビルドキャッシュ・リモートキャッシュによる CI 高速化
4. Nx の高度な機能（影響範囲分析・コード生成・プラグインエコシステム）
5. 共有パッケージの設計パターンとバージョニング戦略
6. 大規模モノレポの運用ノウハウとトラブルシューティング

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
| リファクタリング | 全パッケージ一括可能 | 各リポジトリで個別対応 |
| バージョン管理 | 統一的に管理可能 | 各リポジトリで独立 |
| テスト | 統合テストが容易 | サービス間テストが困難 |

### 1.2 モノレポが適するケース

```
モノレポの採用判断フローチャート:

  プロジェクトの特性を確認:

  Q1: パッケージ間でコードを頻繁に共有する？
      │
      ├── Yes → モノレポ向き
      │
      └── No ─→ Q2: チームが同じリリースサイクルで動く？
                    │
                    ├── Yes → モノレポ向き
                    │
                    └── No ─→ Q3: 横断的なリファクタリングが頻繁？
                                  │
                                  ├── Yes → モノレポ向き
                                  │
                                  └── No ─→ ポリレポの方が適切

  モノレポが特に有効なケース:
  ┌─────────────────────────────────────────────┐
  │ - フロント + バックエンド + 共有ライブラリ      │
  │ - マイクロフロントエンド構成                    │
  │ - デザインシステム + 消費アプリケーション         │
  │ - 社内ツール群の統合管理                        │
  │ - 型定義・バリデーションの共有が必要              │
  └─────────────────────────────────────────────┘

  ポリレポが適切なケース:
  ┌─────────────────────────────────────────────┐
  │ - 完全に独立したサービス群                      │
  │ - 異なるチーム・組織がオーナー                   │
  │ - 異なる言語・ランタイムが混在                   │
  │ - 公開 npm パッケージの開発                     │
  │ - 独立したデプロイサイクルが必要                  │
  └─────────────────────────────────────────────┘
```

### 1.3 モノレポの構造

```
典型的なモノレポ構造:

my-monorepo/
├── package.json              # ルート (ワークスペース定義)
├── pnpm-workspace.yaml       # pnpm ワークスペース設定
├── turbo.json                # Turborepo 設定
├── .npmrc                    # pnpm 設定
├── .node-version             # Node.js バージョン固定
├── .gitignore                # Git 除外設定
├── tsconfig.json             # ルート TypeScript 設定
│
├── apps/                     # アプリケーション層
│   ├── web/                  # Next.js フロントエンド
│   │   ├── package.json
│   │   ├── next.config.js
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── app/          # App Router
│   │       ├── components/   # ページ固有コンポーネント
│   │       └── lib/          # ユーティリティ
│   ├── api/                  # Express / Hono バックエンド
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── routes/
│   │       ├── middleware/
│   │       └── index.ts
│   ├── admin/                # 管理画面 (別 Next.js)
│   │   ├── package.json
│   │   └── src/
│   └── mobile/               # React Native
│       ├── package.json
│       └── src/
│
├── packages/                 # 共有パッケージ層
│   ├── ui/                   # 共有UIコンポーネント
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── src/
│   │       ├── components/
│   │       │   ├── Button.tsx
│   │       │   ├── Input.tsx
│   │       │   └── Modal.tsx
│   │       └── index.ts
│   ├── config/               # 共有設定 (ESLint, TS)
│   │   ├── package.json
│   │   ├── eslint/
│   │   ├── tsconfig/
│   │   └── tailwind/
│   ├── utils/                # 共有ユーティリティ
│   │   ├── package.json
│   │   └── src/
│   │       ├── date.ts
│   │       ├── format.ts
│   │       └── validate.ts
│   ├── types/                # 共有型定義
│   │   ├── package.json
│   │   └── src/
│   │       ├── api.ts
│   │       ├── database.ts
│   │       └── user.ts
│   ├── database/             # DB クライアント (Prisma/Drizzle)
│   │   ├── package.json
│   │   ├── prisma/
│   │   │   └── schema.prisma
│   │   └── src/
│   │       ├── client.ts
│   │       └── index.ts
│   └── auth/                 # 認証ロジック共有
│       ├── package.json
│       └── src/
│
└── tooling/                  # ビルドツール設定
    ├── eslint/
    │   └── package.json
    ├── typescript/
    │   └── package.json
    └── tailwind/
        └── package.json
```

### 1.4 依存関係グラフ

```
パッケージ間の依存関係:

  apps/web ──→ packages/ui ──→ packages/types
     │              │
     │              └──→ packages/utils
     │
     └──→ packages/utils ──→ packages/types
     │
     └──→ packages/types
     │
     └──→ packages/database ──→ packages/types
     │
     └──→ packages/auth ──→ packages/database
                           ──→ packages/types

  apps/api ──→ packages/utils ──→ packages/types
     │
     └──→ packages/types
     │
     └──→ packages/database ──→ packages/types
     │
     └──→ packages/auth

  apps/admin ──→ packages/ui ──→ packages/types
     │                          ──→ packages/utils
     └──→ packages/auth

  ビルド順序 (Turborepo が自動解決):
  1. packages/types     (依存なし)
  2. packages/utils     (types に依存)
  3. packages/database  (types に依存)
  4. packages/ui        (types, utils に依存)
  5. packages/auth      (database, types に依存)
  6. apps/web, apps/api, apps/admin (並列実行可能)

  重要な原則:
  - パッケージ間の依存は単方向のみ (循環禁止)
  - apps → packages の参照は自由
  - packages → apps の参照は禁止
  - packages 間は DAG (有向非巡回グラフ) を維持
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

```ini
# .npmrc (pnpm の動作設定)
# 厳密な依存解決 (phantom dependencies 防止)
strict-peer-dependencies=false
# ホイスティング制御
shamefully-hoist=false
# ロックファイルの自動更新を防止
frozen-lockfile=false
# リンクプロトコル
link-workspace-packages=true
# パッケージインポート方式 (hardlink が最速)
package-import-method=hardlink
# 並列インストール数
network-concurrency=16
```

```jsonc
// ルート package.json
{
  "name": "my-monorepo",
  "private": true,
  "packageManager": "pnpm@9.1.0",
  "engines": {
    "node": ">=20.0.0",
    "pnpm": ">=9.0.0"
  },
  "scripts": {
    "dev": "turbo dev",
    "build": "turbo build",
    "lint": "turbo lint",
    "test": "turbo test",
    "clean": "turbo clean",
    "typecheck": "turbo typecheck",
    "format": "prettier --write \"**/*.{ts,tsx,md,json}\"",
    "format:check": "prettier --check \"**/*.{ts,tsx,md,json}\"",
    "db:migrate": "pnpm --filter @repo/database migrate",
    "db:seed": "pnpm --filter @repo/database seed",
    "changeset": "changeset",
    "version-packages": "changeset version",
    "release": "turbo build --filter='./packages/*' && changeset publish"
  },
  "devDependencies": {
    "turbo": "^2.0.0",
    "prettier": "^3.2.0",
    "@changesets/cli": "^2.27.0"
  }
}
```

### 2.2 内部パッケージの参照

```jsonc
// packages/types/package.json
{
  "name": "@repo/types",
  "version": "0.0.0",
  "private": true,
  "exports": {
    ".": {
      "types": "./src/index.ts",
      "default": "./src/index.ts"
    },
    "./api": {
      "types": "./src/api.ts",
      "default": "./src/api.ts"
    },
    "./database": {
      "types": "./src/database.ts",
      "default": "./src/database.ts"
    }
  },
  "scripts": {
    "typecheck": "tsc --noEmit",
    "lint": "eslint ."
  },
  "devDependencies": {
    "typescript": "^5.4.0"
  }
}

// packages/ui/package.json
{
  "name": "@repo/ui",
  "version": "0.0.0",
  "private": true,
  "exports": {
    ".": "./src/index.ts",
    "./button": "./src/components/Button.tsx",
    "./input": "./src/components/Input.tsx",
    "./modal": "./src/components/Modal.tsx"
  },
  "scripts": {
    "build": "tsup src/index.ts --format esm,cjs --dts",
    "dev": "tsup src/index.ts --format esm,cjs --dts --watch",
    "lint": "eslint .",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@repo/types": "workspace:*",
    "@repo/utils": "workspace:*"
  },
  "devDependencies": {
    "tsup": "^8.0.0",
    "typescript": "^5.4.0"
  },
  "peerDependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  }
}

// packages/utils/package.json
{
  "name": "@repo/utils",
  "version": "0.0.0",
  "private": true,
  "exports": {
    ".": "./src/index.ts",
    "./date": "./src/date.ts",
    "./format": "./src/format.ts",
    "./validate": "./src/validate.ts"
  },
  "dependencies": {
    "@repo/types": "workspace:*"
  },
  "devDependencies": {
    "typescript": "^5.4.0"
  }
}

// apps/web/package.json
{
  "name": "@repo/web",
  "private": true,
  "scripts": {
    "dev": "next dev --port 3000",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "typecheck": "tsc --noEmit"
  },
  "dependencies": {
    "@repo/ui": "workspace:*",
    "@repo/utils": "workspace:*",
    "@repo/types": "workspace:*",
    "@repo/auth": "workspace:*",
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0"
  },
  "devDependencies": {
    "@repo/config": "workspace:*",
    "typescript": "^5.4.0"
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
pnpm --filter @repo/ui add -D tsup

# ─── フィルタ実行 ───
pnpm --filter @repo/web dev           # web のみ dev
pnpm --filter "./apps/*" build        # apps 配下全て build
pnpm --filter @repo/ui... build       # ui とその依存先を build
pnpm --filter ...@repo/web build      # web の依存元を全て build
pnpm --filter "@repo/*" lint          # 全 @repo スコープ lint

# ─── 変更されたパッケージのみ ───
pnpm --filter "...[origin/main]" build  # main から変更されたパッケージ

# ─── 全パッケージ操作 ───
pnpm -r exec -- rm -rf node_modules dist .next  # 全クリーン
pnpm install                                     # 再インストール
pnpm -r list --depth 0                           # 全パッケージの依存一覧

# ─── ワークスペース情報 ───
pnpm ls -r --json                     # 全パッケージ情報 (JSON)
pnpm why react                        # react がどこで使われているか
```

### 2.4 pnpm の依存解決メカニズム

```
pnpm のストア構造 (コンテンツアドレッサブルストレージ):

  ~/.pnpm-store/                         # グローバルストア
  └── v3/
      └── files/
          └── {hash}/                    # ハッシュベースで一意
              ├── node_modules/
              │   └── react/
              │       ├── index.js
              │       └── package.json
              └── ...

  プロジェクト内:
  node_modules/
  ├── .pnpm/                            # フラットな実体格納
  │   ├── react@18.3.0/
  │   │   └── node_modules/
  │   │       └── react/  → ストアへのハードリンク
  │   ├── next@14.2.0/
  │   │   └── node_modules/
  │   │       ├── next/   → ストアへのハードリンク
  │   │       └── react/  → .pnpm/react@18.3.0 へのシンボリックリンク
  │   └── ...
  ├── react  → .pnpm/react@18.3.0/node_modules/react (シンボリックリンク)
  └── next   → .pnpm/next@14.2.0/node_modules/next (シンボリックリンク)

  メリット:
  1. ディスク使用量の大幅削減 (ハードリンクにより実体は1つ)
  2. Phantom dependency の防止 (宣言していないパッケージは見えない)
  3. ワークスペース間で同一バージョンのパッケージを共有
  4. インストール速度の向上 (ダウンロード不要なら即リンク)
```

### 2.5 Corepack によるパッケージマネージャーバージョン固定

```bash
# Corepack の有効化
corepack enable

# パッケージマネージャーバージョンを固定
corepack use pnpm@9.1.0

# これにより package.json に以下が追加される:
# "packageManager": "pnpm@9.1.0"

# 異なるバージョンの pnpm で install しようとするとエラー:
# This project is configured to use pnpm@9.1.0.
# Please install the correct version.
```

```jsonc
// .node-version (Node.js バージョン固定)
// fnm / nvm / volta が参照する
// 20.12.0
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
  "globalEnv": [
    "CI",
    "NODE_ENV"
  ],
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tsconfig.json", "package.json"],
      "outputs": ["dist/**", ".next/**", "build/**"],
      "env": ["NODE_ENV", "NEXT_PUBLIC_*"]
    },
    "dev": {
      "dependsOn": ["^build"],
      "persistent": true,
      "cache": false
    },
    "lint": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "eslint.config.*", "biome.json"],
      "outputs": []
    },
    "test": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tests/**", "vitest.config.*", "__tests__/**"],
      "outputs": ["coverage/**"],
      "env": ["DATABASE_URL", "TEST_DATABASE_URL"]
    },
    "test:watch": {
      "dependsOn": ["^build"],
      "persistent": true,
      "cache": false
    },
    "clean": {
      "cache": false
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tsconfig.json", "tsconfig.*.json"],
      "outputs": []
    },
    "db:migrate": {
      "cache": false,
      "env": ["DATABASE_URL"]
    },
    "db:seed": {
      "cache": false,
      "dependsOn": ["db:migrate"]
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
  │           lockfile の内容            │
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

  ハッシュ計算の詳細:
  ┌─────────────────────────────────────────────┐
  │ hash = SHA256(                               │
  │   ソースファイルの内容,                       │
  │   turbo.json の inputs 設定,                 │
  │   環境変数の値 (env に列挙したもの),           │
  │   依存パッケージのビルドハッシュ (^build),     │
  │   lockfile の該当部分,                        │
  │   .env ファイル (globalDependencies),         │
  │ )                                            │
  │                                               │
  │ 変更検知: 上記のいずれかが変わるとMISS          │
  │ 不変検知: 全て同一ならHIT                      │
  └─────────────────────────────────────────────┘
```

### 3.3 リモートキャッシュ

```bash
# Vercel リモートキャッシュ (無料)
npx turbo login
npx turbo link

# 自前のキャッシュサーバー (ducktape, turborepo-remote-cache)
# turbo.json に追加:
# {
#   "remoteCache": {
#     "signature": true,
#     "enabled": true
#   }
# }

# 環境変数で設定
export TURBO_TOKEN=your-token
export TURBO_TEAM=your-team
export TURBO_API=https://your-cache-server.example.com
```

```
リモートキャッシュの効果:

  開発者A: turbo build
  ├── packages/types  → ビルド (5s) → キャッシュ保存 ↑
  ├── packages/utils  → ビルド (8s) → キャッシュ保存 ↑
  ├── packages/ui     → ビルド (12s) → キャッシュ保存 ↑
  └── apps/web        → ビルド (30s) → キャッシュ保存 ↑
  合計: 55秒

  開発者B (同じコミット): turbo build
  ├── packages/types  → リモートキャッシュ HIT (0.1s) ↓
  ├── packages/utils  → リモートキャッシュ HIT (0.1s) ↓
  ├── packages/ui     → リモートキャッシュ HIT (0.2s) ↓
  └── apps/web        → リモートキャッシュ HIT (0.5s) ↓
  合計: 0.9秒 (98% 短縮)

  CI: turbo build
  ├── 開発者がすでにビルド済み → 全て HIT
  合計: 1秒以下

  月間での効果 (チーム5人・1日10ビルドの場合):
  ┌────────────────────────────────────┐
  │ キャッシュなし: 55s × 10回 × 5人 × 22日 = 1,683分/月  │
  │ キャッシュあり: 大半がHIT → 約30分/月                    │
  │ 削減率: 約98%                                          │
  └────────────────────────────────────┘
```

### 3.4 Turborepo の高度な機能

```bash
# ─── タスクグラフの可視化 ───
turbo build --graph              # タスク依存グラフを生成
turbo build --graph=graph.svg    # SVG で出力
turbo build --graph=graph.html   # HTML で出力

# ─── ドライラン (何が実行されるか確認) ───
turbo build --dry-run
turbo build --dry-run=json       # JSON で出力

# ─── フィルタリング ───
turbo build --filter=@repo/web                # 特定パッケージ
turbo build --filter=@repo/web...             # web とその依存
turbo build --filter=...@repo/web             # web を依存するパッケージ
turbo build --filter="[HEAD~1]"               # 直前のコミットから変更されたもの
turbo build --filter="[origin/main...HEAD]"   # main から変更されたもの

# ─── 並列度の制御 ───
turbo build --concurrency=50%    # CPU の半分を使用
turbo build --concurrency=4      # 最大4並列
turbo build --concurrency=1      # 逐次実行 (デバッグ用)

# ─── キャッシュ操作 ───
turbo build --force              # キャッシュを無視して再ビルド
turbo build --no-cache           # キャッシュに保存しない
turbo prune --scope=@repo/web    # web のみのスリムモノレポを生成
```

### 3.5 turbo prune によるデプロイ最適化

```bash
# Docker ビルド時に不要なパッケージを除外
turbo prune --scope=@repo/web --docker

# 以下の構造が out/ に生成される:
# out/
# ├── json/                    # package.json のみ (依存解決用)
# │   ├── package.json
# │   ├── apps/web/package.json
# │   ├── packages/ui/package.json
# │   └── packages/types/package.json
# ├── full/                    # ソースコード含む完全版
# │   ├── apps/web/
# │   ├── packages/ui/
# │   └── packages/types/
# ├── pnpm-lock.yaml          # 必要な依存のみのロックファイル
# └── pnpm-workspace.yaml
```

```dockerfile
# Dockerfile (turbo prune と組み合わせ)
FROM node:20-slim AS base
RUN corepack enable

# Step 1: prune で必要なパッケージのみ抽出
FROM base AS pruner
WORKDIR /app
COPY . .
RUN turbo prune --scope=@repo/web --docker

# Step 2: 依存インストール (package.json のみコピーでキャッシュ活用)
FROM base AS installer
WORKDIR /app
COPY --from=pruner /app/out/json/ .
COPY --from=pruner /app/out/pnpm-lock.yaml ./pnpm-lock.yaml
COPY --from=pruner /app/out/pnpm-workspace.yaml ./pnpm-workspace.yaml
RUN pnpm install --frozen-lockfile

# Step 3: ビルド
FROM base AS builder
WORKDIR /app
COPY --from=installer /app/ .
COPY --from=pruner /app/out/full/ .
RUN turbo build --filter=@repo/web

# Step 4: 本番イメージ
FROM node:20-slim AS runner
WORKDIR /app
COPY --from=builder /app/apps/web/.next/standalone ./
COPY --from=builder /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder /app/apps/web/public ./apps/web/public
CMD ["node", "apps/web/server.js"]
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
    },
    "e2e": {
      "inputs": ["default", "^production"],
      "cache": true
    }
  },
  "namedInputs": {
    "default": ["{projectRoot}/**/*", "sharedGlobals"],
    "production": [
      "default",
      "!{projectRoot}/**/*.spec.*",
      "!{projectRoot}/**/*.test.*",
      "!{projectRoot}/test/**/*",
      "!{projectRoot}/.eslintrc.json"
    ],
    "sharedGlobals": [
      "{workspaceRoot}/tsconfig.base.json",
      "{workspaceRoot}/.eslintrc.json"
    ]
  },
  "defaultBase": "main",
  "plugins": [
    "@nx/next/plugin",
    "@nx/eslint/plugin",
    "@nx/vite/plugin"
  ]
}
```

### 4.2 Nx の高度な機能

```bash
# ─── 影響範囲分析 (affected) ───
# main ブランチから変更されたパッケージのみビルド・テスト
nx affected -t build
nx affected -t test
nx affected -t lint

# ─── 依存グラフの可視化 ───
nx graph                         # ブラウザでインタラクティブ表示
nx graph --file=graph.json       # JSON で出力
nx affected:graph                # 変更の影響範囲をハイライト

# ─── コード生成 (Generator) ───
nx generate @nx/react:component Button --project=ui
nx generate @nx/next:page about --project=web
nx generate @nx/node:application api

# ─── マイグレーション ───
nx migrate latest                # 依存の自動アップデート
nx migrate --run-migrations      # マイグレーションスクリプト実行

# ─── タスク実行 ───
nx run web:build                 # 単一タスク
nx run-many -t build test lint   # 複数タスクを全プロジェクトで
nx run-many -t build --projects=web,api  # 指定プロジェクトのみ
```

### 4.3 Nx Cloud (リモートキャッシュ)

```bash
# Nx Cloud の接続
npx nx connect-to-nx-cloud

# CI での利用
# nx.json に自動追加:
# "nxCloud": "access-token-here"

# 分散タスク実行 (DTE)
# CI で複数マシンにタスクを分散
# - Agent マシンがタスクを受け取って実行
# - 結果をキャッシュに保存
# - メインマシンが結果を集約
```

### 4.4 Turborepo vs Nx 比較

| 特徴 | Turborepo | Nx |
|------|-----------|-----|
| 設計思想 | シンプル・軽量 | フル機能・統合型 |
| 設定ファイル | turbo.json のみ | nx.json + project.json |
| キャッシュ | ローカル + リモート | ローカル + Nx Cloud |
| 依存グラフ可視化 | `turbo --graph` (静的) | `nx graph` (インタラクティブ) |
| コード生成 | なし | `nx generate` (豊富) |
| 影響範囲分析 | `turbo --filter=[...]` | `nx affected` (高精度) |
| プラグイン | なし | 豊富 (React, Next, Node, etc.) |
| 分散実行 | なし | Nx Cloud DTE |
| マイグレーション | なし | `nx migrate` (自動更新) |
| 学習コスト | 低い | 中〜高い |
| 推奨規模 | 小〜中 (2-20パッケージ) | 中〜大 (10-500+ パッケージ) |
| Vercel 連携 | ネイティブ | プラグイン |
| パフォーマンス | Rust 製 (高速) | Node.js + Rust (タスクハッシュ) |

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
    "sourceMap": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "exactOptionalPropertyTypes": false,
    "verbatimModuleSyntax": true
  }
}

// packages/config/tsconfig/react.json
{
  "extends": "./base.json",
  "compilerOptions": {
    "jsx": "react-jsx",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler"
  }
}

// packages/config/tsconfig/nextjs.json
{
  "extends": "./react.json",
  "compilerOptions": {
    "plugins": [{ "name": "next" }],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowJs": true,
    "incremental": true
  }
}

// packages/config/tsconfig/node.json
{
  "extends": "./base.json",
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler",
    "outDir": "./dist",
    "rootDir": "./src",
    "lib": ["ES2022"]
  }
}

// apps/web/tsconfig.json (利用側)
{
  "extends": "@repo/config/tsconfig/nextjs.json",
  "compilerOptions": {
    "outDir": "./dist",
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*", "next-env.d.ts", ".next/types/**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
```

### 5.2 ESLint 設定共有

```javascript
// packages/config/eslint/base.js
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import importPlugin from "eslint-plugin-import";

export default [
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    plugins: {
      import: importPlugin,
    },
    rules: {
      // ─── 型安全性 ───
      "@typescript-eslint/no-unused-vars": ["error", {
        argsIgnorePattern: "^_",
        varsIgnorePattern: "^_",
        destructuredArrayIgnorePattern: "^_",
      }],
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/prefer-as-const": "error",
      "@typescript-eslint/no-non-null-assertion": "warn",

      // ─── インポート順序 ───
      "import/order": ["error", {
        "groups": [
          "builtin",
          "external",
          "internal",
          ["parent", "sibling", "index"],
        ],
        "newlines-between": "always",
        "alphabetize": { "order": "asc" },
      }],
      "import/no-duplicates": "error",

      // ─── コード品質 ───
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "prefer-const": "error",
      "no-var": "error",
      "eqeqeq": ["error", "always"],
    },
  },
];

// packages/config/eslint/react.js
import reactPlugin from "eslint-plugin-react";
import reactHooksPlugin from "eslint-plugin-react-hooks";
import jsxA11y from "eslint-plugin-jsx-a11y";
import baseConfig from "./base.js";

export default [
  ...baseConfig,
  {
    plugins: {
      react: reactPlugin,
      "react-hooks": reactHooksPlugin,
      "jsx-a11y": jsxA11y,
    },
    rules: {
      "react/prop-types": "off",
      "react/react-in-jsx-scope": "off",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "jsx-a11y/alt-text": "error",
      "jsx-a11y/anchor-is-valid": "error",
    },
    settings: {
      react: { version: "detect" },
    },
  },
];

// packages/config/eslint/next.js
import nextPlugin from "@next/eslint-plugin-next";
import reactConfig from "./react.js";

export default [
  ...reactConfig,
  {
    plugins: { "@next/next": nextPlugin },
    rules: {
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs["core-web-vitals"].rules,
    },
  },
];

// apps/web/eslint.config.js (利用側)
import nextConfig from "@repo/config/eslint/next.js";

export default [
  ...nextConfig,
  {
    ignores: [".next/", "dist/"],
  },
];
```

### 5.3 Tailwind CSS 設定共有

```javascript
// packages/config/tailwind/base.js
/** @type {import('tailwindcss').Config} */
export default {
  content: [],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eff6ff",
          100: "#dbeafe",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          900: "#1e3a5f",
        },
      },
      fontFamily: {
        sans: ["Inter", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      borderRadius: {
        DEFAULT: "0.5rem",
      },
    },
  },
  plugins: [],
};

// apps/web/tailwind.config.js
import baseConfig from "@repo/config/tailwind/base.js";

/** @type {import('tailwindcss').Config} */
export default {
  ...baseConfig,
  content: [
    "./src/**/*.{ts,tsx}",
    "../../packages/ui/src/**/*.{ts,tsx}",
  ],
};
```

### 5.4 Prettier 設定共有

```jsonc
// packages/config/prettier/index.json
{
  "semi": true,
  "singleQuote": true,
  "trailingComma": "all",
  "printWidth": 80,
  "tabWidth": 2,
  "useTabs": false,
  "bracketSpacing": true,
  "arrowParens": "always",
  "endOfLine": "lf",
  "plugins": ["prettier-plugin-tailwindcss"]
}

// apps/web/.prettierrc (利用側)
// "@repo/config/prettier" を直接参照
// package.json に以下を追加:
// "prettier": "@repo/config/prettier"
```

---

## 6. バージョニングとリリース

### 6.1 Changesets によるバージョン管理

```bash
# セットアップ
pnpm add -Dw @changesets/cli
pnpm changeset init

# 変更の記録
pnpm changeset
# → パッケージ選択
# → バージョンタイプ選択 (major / minor / patch)
# → 変更内容の記述
```

```yaml
# .changeset/config.json
{
  "$schema": "https://unpkg.com/@changesets/config@3.0.0/schema.json",
  "changelog": "@changesets/cli/changelog",
  "commit": false,
  "fixed": [],
  "linked": [["@repo/ui", "@repo/utils", "@repo/types"]],
  "access": "restricted",
  "baseBranch": "main",
  "updateInternalDependencies": "patch",
  "ignore": ["@repo/web", "@repo/api"]
}
```

```
Changesets のワークフロー:

  1. 開発者が変更を実装
  2. pnpm changeset で変更セットを作成
     → .changeset/random-name.md が生成される

  3. PR をマージ

  4. CI が pnpm changeset version を実行
     → package.json のバージョン更新
     → CHANGELOG.md の更新
     → 変更セットファイルの削除

  5. CI が pnpm changeset publish を実行
     → npm に公開 (private でないパッケージ)

  自動化例 (GitHub Actions + Changesets Bot):
  - PR 作成時: Bot が "Version Packages" PR を自動作成
  - マージ時: 自動的にバージョン更新 + npm publish
```

### 6.2 GitHub Actions での自動リリース

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    branches: [main]

concurrency: ${{ github.workflow }}-${{ github.ref }}

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      packages: write
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: Create Release Pull Request or Publish
        uses: changesets/action@v1
        with:
          publish: pnpm release
          version: pnpm version-packages
          commit: "chore: release packages"
          title: "chore: release packages"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## 7. CI/CD 設定

### 7.1 GitHub Actions (Turborepo)

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
    timeout-minutes: 15
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
      - name: Build, Lint, Test, Typecheck
        run: pnpm turbo build lint test typecheck
        env:
          TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
          TURBO_TEAM: ${{ vars.TURBO_TEAM }}

  # PR 時のみ: 変更されたパッケージだけチェック
  affected:
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: Check affected packages
        run: |
          pnpm turbo build --filter="[origin/main...HEAD]" --dry-run=json \
            | jq '.tasks | map(.package) | unique'
```

### 7.2 GitHub Actions (Nx)

```yaml
# .github/workflows/ci-nx.yml
name: CI (Nx)
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  main:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      # SHAs の設定 (affected の基準)
      - uses: nrwl/nx-set-shas@v4

      # 影響範囲のみビルド・テスト
      - run: npx nx affected -t build test lint typecheck
```

---

## 8. アンチパターン

### 8.1 全パッケージの依存をルートに入れる

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
  - Docker ビルド時に全依存をインストールしてしまう
  - CI キャッシュの効率が低下

✅ 正しいアプローチ:
  - 各パッケージの package.json に必要な依存のみ記述
  - ルートには turbo, prettier 等のツールのみ
  - pnpm の厳密な依存解決に任せる
  - 共通の devDependencies は packages/config に集約
```

### 8.2 パッケージ間で循環依存を作る

```
❌ アンチパターン: パッケージ間の循環参照

  @repo/ui → @repo/utils → @repo/ui  (循環!)

問題:
  - ビルド順序が決定できない
  - TypeScript の型解決が無限ループ
  - Turborepo/Nx のキャッシュが効かない
  - ホットリロードが壊れる

✅ 正しいアプローチ:
  - 共通部分を別パッケージに抽出
  - @repo/ui → @repo/shared ← @repo/utils
  - nx graph / turbo --graph で依存グラフを定期確認
  - CI で循環依存チェックを自動化:
    pnpm ls -r --json | node scripts/check-circular.js
```

### 8.3 内部パッケージにバージョン範囲を使う

```
❌ アンチパターン: workspace パッケージに ^, ~ を使う

  "dependencies": {
    "@repo/ui": "^1.0.0"    // ← npm レジストリから探しに行く
  }

問題:
  - ローカルのパッケージではなく npm のパッケージを参照してしまう
  - バージョンの不一致でビルドエラー
  - pnpm install 時にネットワークアクセスが発生

✅ 正しいアプローチ:
  - workspace: プロトコルを使う
  "dependencies": {
    "@repo/ui": "workspace:*"   // 常にローカルを参照
    "@repo/utils": "workspace:^" // publish 時に ^ に変換
  }
```

### 8.4 モノレポ全体で単一の tsconfig.json

```
❌ アンチパターン: 1つの tsconfig.json で全パッケージをカバー

  tsconfig.json (root):
    "include": ["apps/**/*", "packages/**/*"]

問題:
  - 型チェックが全パッケージに及び、極端に遅い
  - パッケージごとの設定カスタマイズができない
  - IDE の IntelliSense が遅くなる

✅ 正しいアプローチ:
  - 各パッケージに tsconfig.json を配置
  - 共通設定は packages/config/tsconfig/ で管理
  - references (Project References) で明示的に参照
```

### 8.5 テストデータベースの共有

```
❌ アンチパターン: 全アプリが同じテストDBを使う

問題:
  - 並列テストでデータ競合
  - テストの順序依存性
  - CI で不安定なテスト (Flaky Test)

✅ 正しいアプローチ:
  - パッケージごとに独立したテストDB
  - テストごとにトランザクションでロールバック
  - Docker Compose でテスト用DBコンテナを別途起動
```

---

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

```
問題: pnpm install 後に型エラーが大量に出る

原因: TypeScript の Project References が正しく設定されていない
解決:
  1. 各パッケージの tsconfig.json に "composite": true を追加
  2. 依存元パッケージの tsconfig.json に references を追加
  3. turbo build で依存パッケージを先にビルド

---

問題: turbo dev でホットリロードが効かない

原因: 内部パッケージの変更が検知されていない
解決:
  1. 内部パッケージの exports がソースを直接指しているか確認
     "exports": { ".": "./src/index.ts" }  // ✅ ソース直接参照
     "exports": { ".": "./dist/index.js" } // ❌ ビルド済みを参照
  2. Next.js の場合は next.config.js に transpilePackages を追加
     transpilePackages: ["@repo/ui", "@repo/utils"]
  3. Vite の場合は optimizeDeps.include に追加

---

問題: CI でのビルドが極端に遅い

原因: キャッシュが効いていない
解決:
  1. TURBO_TOKEN / TURBO_TEAM が正しく設定されているか確認
  2. turbo.json の inputs / outputs が適切か確認
  3. env に不要な環境変数が含まれていないか確認
     (環境変数が変わるとキャッシュが無効化される)
  4. pnpm install のキャッシュ (actions/cache) を設定

---

問題: node_modules のファントム依存性でランタイムエラー

原因: package.json に宣言していないパッケージを使っている
解決:
  1. .npmrc に strict-peer-dependencies=true を設定
  2. shamefully-hoist=false を確認 (true だと hoisting で見えてしまう)
  3. pnpm ls --depth 0 で各パッケージの依存を確認
  4. 必要な依存を明示的に package.json に追加
```

### 9.2 パフォーマンス最適化チェックリスト

```
モノレポのパフォーマンス最適化:

□ pnpm の package-import-method が hardlink になっているか
□ turbo.json の inputs が適切に絞られているか
□ outputs に不要なディレクトリが含まれていないか
□ リモートキャッシュが設定されているか
□ CI の fetch-depth が最小限か
□ TypeScript の composite と incremental が有効か
□ Next.js の standalone 出力を使っているか
□ node_modules が Volume マウントになっているか (Docker)
□ 不要な devDependencies が production ビルドに含まれていないか
□ ESLint の型チェック連携 (recommendedTypeChecked) が CI でのみ有効か
```

---

## 10. FAQ

### Q1: Turborepo と Nx、どちらを選ぶべき？

**A:** 以下の基準で判断する。
- **Turborepo**: パッケージ数が 10 以下、シンプルなキャッシュとタスク実行が目的、設定を最小限にしたい場合。Vercel エコシステム（Next.js）との相性が良い。Rust 製のため単体の実行速度も速い。
- **Nx**: パッケージ数が多い大規模プロジェクト、コード生成や影響範囲分析が必要、プラグインエコシステムを活用したい場合。分散タスク実行 (DTE) が必要な場合は Nx Cloud 一択。

迷ったら Turborepo から始めて、必要に応じて Nx に移行するのが低リスク。Nx → Turborepo の移行は設定が減る方向なので比較的容易だが、Generator 等の Nx 固有機能を使っている場合は移行コストが高くなる。

### Q2: モノレポの git clone が遅い場合の対処法は？

**A:** 以下の方法がある。
1. `git clone --depth 1` でシャロークローン
2. `git clone --filter=blob:none` でパーシャルクローン (ファイル内容は必要時にダウンロード)
3. `git clone --filter=tree:0` でツリーレスクローン (ディレクトリ構造も遅延)
4. GitHub Actions では `fetch-depth: 0` ではなく必要最小限の深さを指定
5. Git LFS を使う大きなファイルがある場合は `git lfs install --skip-smudge`
6. リポジトリが巨大な場合は `git sparse-checkout` で必要なディレクトリのみチェックアウト

```bash
# sparse-checkout の例
git clone --filter=blob:none --sparse https://github.com/org/monorepo.git
cd monorepo
git sparse-checkout set apps/web packages/ui packages/types
```

### Q3: 内部パッケージはビルドすべき？ソースのまま参照すべき？

**A:** 2つのアプローチがある。

| アプローチ | メリット | デメリット |
|-----------|---------|-----------|
| **ビルド方式** (`tsc` → `dist/`) | 型安全性が高い、消費側のビルドが速い | ビルドステップが必要、watch が複雑 |
| **ソース参照方式** (`src/index.ts` 直接) | ビルド不要、HMR が速い | 消費側でトランスパイル必要 |

小〜中規模ならソース参照方式が手軽。大規模ではビルド方式の方が CI が安定する。ハイブリッドアプローチとして、開発時はソース参照、CI/本番ビルドでは `tsup` でバンドルする方法もある。

### Q4: pnpm workspaces と yarn workspaces の違いは？

**A:** 主な違いは以下の通り。
- **pnpm**: コンテンツアドレッサブルストレージでディスク効率が高い、厳密な依存解決でファントム依存を防止、ハードリンクによる高速インストール。
- **yarn (v4 Berry)**: PnP (Plug'n'Play) でゼロインストール可能、`.yarnrc.yml` でプラグイン拡張、`yarn dlx` でパッケージ実行。

2025年時点では pnpm がモノレポのデファクトスタンダード。特に Turborepo との組み合わせでは pnpm が推奨されている。

### Q5: モノレポでの Docker ビルドを最適化するには？

**A:** `turbo prune` を使うのが最善。これにより、特定のアプリケーションとその依存パッケージのみを含むスリムなモノレポを生成できる。Docker の `COPY` レイヤーキャッシュを最大限に活用するために、`--docker` フラグを使って `json/` (package.json のみ) と `full/` (ソース含む) を分離し、依存インストールのレイヤーとソースコピーのレイヤーを分ける。

---

## 11. まとめ

| 構成要素 | 推奨 | 備考 |
|---------|------|------|
| パッケージマネージャー | pnpm | ワークスペースに最適、ディスク効率 |
| タスクランナー (小〜中規模) | Turborepo | シンプルで高速、Rust 製 |
| タスクランナー (大規模) | Nx | 影響範囲分析・生成器・DTE |
| 設定共有 | `packages/config/` | ESLint, TS, Tailwind, Prettier |
| CI キャッシュ | Turborepo Remote Cache | Vercel 無料枠あり |
| パッケージ参照 | `workspace:*` | 内部パッケージ管理 |
| バージョン統一 | Corepack | packageManager フィールド |
| バージョニング | Changesets | 公開パッケージのバージョン管理 |
| デプロイ最適化 | `turbo prune --docker` | Docker レイヤーキャッシュ活用 |
| 依存グラフ監視 | `turbo --graph` / `nx graph` | 循環依存の早期発見 |

---

## 次に読むべきガイド

- [01-package-managers.md](./01-package-managers.md) -- パッケージマネージャーの詳細
- [03-linter-formatter.md](./03-linter-formatter.md) -- モノレポでのLinter/Formatter設定共有
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) -- チーム標準の設定

---

## 参考文献

1. **Turborepo Handbook** -- https://turbo.build/repo/docs/handbook -- Turborepo 公式ハンドブック。設計パターン解説。
2. **Nx Documentation** -- https://nx.dev/getting-started/intro -- Nx 公式入門ガイド。
3. **pnpm Workspaces** -- https://pnpm.io/workspaces -- pnpm ワークスペースの公式ドキュメント。
4. **Monorepo Tools** -- https://monorepo.tools/ -- モノレポツールの比較サイト。客観的なベンチマークあり。
5. **Changesets** -- https://github.com/changesets/changesets -- モノレポ向けバージョニング・チェンジログ管理ツール。
6. **turbo prune** -- https://turbo.build/repo/docs/reference/prune -- Docker ビルド最適化のための prune 機能。
7. **Nx Cloud** -- https://nx.app/ -- Nx のリモートキャッシュ・分散タスク実行サービス。
