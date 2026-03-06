# npmパッケージ開発

> npmパッケージの設計から公開までの全工程。package.jsonの設計、ESM/CJSデュアルパッケージ、TypeScript設定、ビルドパイプライン、モノレポ管理、セマンティックバージョニング、公開ワークフローまで、プロフェッショナルなパッケージ開発の全知識を体系的に習得する。

## この章で学ぶこと

- [ ] package.jsonの設計原則とexportsフィールドの詳細を理解する
- [ ] ESM/CJSデュアルパッケージのビルド設定を構築できる
- [ ] TypeScriptでの型定義生成と公開パターンを把握する
- [ ] セマンティックバージョニングの判断基準を正しく適用できる
- [ ] モノレポでの複数パッケージ管理戦略を実践できる
- [ ] CI/CDと連携した自動公開ワークフローを構築できる
- [ ] パッケージ品質を測定・改善する指標を活用できる

---

## 1. npmパッケージ開発の全体像

npmパッケージの開発は、単にコードを書いてnpm publishするだけの作業ではない。パッケージの設計、ビルド、テスト、バージョニング、公開、保守という一連のライフサイクル全体を適切に管理する必要がある。

```
npmパッケージ開発ライフサイクル:

  +----------+     +----------+     +----------+     +----------+
  |  設計    | --> |  実装    | --> |  ビルド  | --> |  テスト  |
  | package  |     | src/     |     | tsup/    |     | vitest/  |
  | .json    |     | TypeScript|    | rollup   |     | jest     |
  +----------+     +----------+     +----------+     +----------+
       ^                                                   |
       |                                                   v
  +----------+     +----------+     +----------+     +----------+
  |  保守    | <-- |  監視    | <-- |  公開    | <-- | バージョ |
  | issue/   |     | download |     | npm      |     | ニング   |
  | PR対応   |     | stats    |     | publish  |     | semver   |
  +----------+     +----------+     +----------+     +----------+

  各フェーズの所要時間（中規模パッケージの場合）:
    設計:        数時間〜数日
    実装:        数日〜数週間
    ビルド設定:  数時間
    テスト:      継続的
    バージョニング: PR単位で記録
    公開:        自動化により数分
    保守:        継続的
```

### 1.1 npmレジストリの基本概念

npmレジストリは世界最大のソフトウェアレジストリであり、200万以上のパッケージが登録されている。パッケージ公開者として理解すべき基本概念を整理する。

```
npmレジストリの構造:

  +---------------------------+
  |     npm Registry          |
  |  (registry.npmjs.org)     |
  +---------------------------+
  |                           |
  |  Scoped Packages          |
  |  @scope/package-name      |
  |  例: @example/sdk         |
  |                           |
  |  Unscoped Packages        |
  |  package-name             |
  |  例: express              |
  |                           |
  |  Tags (dist-tags):        |
  |    latest  → 安定版       |
  |    next    → 次期版       |
  |    beta    → ベータ版     |
  |    canary  → カナリア版   |
  |                           |
  +---------------------------+

  パッケージの命名規則:
    - 214文字以下
    - 小文字のみ（大文字不可）
    - ハイフン・ドット・アンダースコア使用可
    - スコープ: @org/name 形式
    - 既存パッケージ名との類似に注意
      （typosquatting 防止）
```

### 1.2 パッケージの種類と設計判断

パッケージを開発する前に、その種類と想定される利用形態を明確にする。

| パッケージ種類 | 特徴 | 例 | 依存方針 |
|---------------|------|-----|---------|
| ライブラリ | 汎用的な関数群 | lodash, date-fns | ゼロ依存が理想 |
| SDK | API クライアント | @aws-sdk/client-s3 | 最小限の依存 |
| CLIツール | コマンドラインツール | eslint, prettier | 必要な依存を許容 |
| フレームワーク | アプリ構築基盤 | express, fastify | プラグイン設計 |
| プラグイン | 既存ツールの拡張 | eslint-plugin-xxx | peerDependencies |
| 型定義 | TypeScript型のみ | @types/xxx | ゼロ依存 |
| ユーティリティ | 小さなヘルパー | is-odd, left-pad | ゼロ依存 |
| モノレポパッケージ | 複数パッケージの集合 | @babel/xxx | 内部依存のみ |

---

## 2. package.json 完全設計ガイド

package.jsonはnpmパッケージの心臓部であり、パッケージのメタデータ、依存関係、エントリポイント、スクリプト、公開設定など、あらゆる情報を定義する。

### 2.1 フルスペック package.json

```json
{
  "name": "@example/sdk",
  "version": "1.0.0",
  "description": "Official SDK for Example API - Type-safe, zero-dependency client",
  "license": "MIT",
  "author": {
    "name": "Example Team",
    "email": "sdk@example.com",
    "url": "https://example.com"
  },
  "contributors": [
    { "name": "Alice", "email": "alice@example.com" }
  ],

  "type": "module",

  "exports": {
    ".": {
      "import": {
        "types": "./dist/index.d.ts",
        "default": "./dist/index.js"
      },
      "require": {
        "types": "./dist/index.d.cts",
        "default": "./dist/index.cjs"
      }
    },
    "./users": {
      "import": {
        "types": "./dist/users/index.d.ts",
        "default": "./dist/users/index.js"
      },
      "require": {
        "types": "./dist/users/index.d.cts",
        "default": "./dist/users/index.cjs"
      }
    },
    "./billing": {
      "import": {
        "types": "./dist/billing/index.d.ts",
        "default": "./dist/billing/index.js"
      },
      "require": {
        "types": "./dist/billing/index.d.cts",
        "default": "./dist/billing/index.cjs"
      }
    },
    "./package.json": "./package.json"
  },

  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts",

  "typesVersions": {
    "*": {
      "users": ["./dist/users/index.d.ts"],
      "billing": ["./dist/billing/index.d.ts"]
    }
  },

  "files": ["dist", "README.md", "LICENSE", "CHANGELOG.md"],

  "engines": { "node": ">=18.0.0" },
  "os": ["!win32"],
  "cpu": ["x64", "arm64"],

  "sideEffects": false,

  "keywords": ["api", "sdk", "example", "typescript", "rest-client"],
  "repository": {
    "type": "git",
    "url": "https://github.com/example/sdk",
    "directory": "packages/sdk"
  },
  "homepage": "https://example.com/docs/sdk",
  "bugs": {
    "url": "https://github.com/example/sdk/issues",
    "email": "bugs@example.com"
  },
  "funding": {
    "type": "github",
    "url": "https://github.com/sponsors/example"
  },

  "scripts": {
    "build": "tsup",
    "build:watch": "tsup --watch",
    "test": "vitest run",
    "test:watch": "vitest",
    "test:coverage": "vitest run --coverage",
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",
    "typecheck": "tsc --noEmit",
    "format": "prettier --write 'src/**/*.ts'",
    "format:check": "prettier --check 'src/**/*.ts'",
    "prepublishOnly": "npm run build && npm run test && npm run typecheck",
    "release": "changeset publish",
    "size": "size-limit",
    "clean": "rm -rf dist",
    "prepack": "clean-pkg-json"
  },

  "devDependencies": {
    "tsup": "^8.0.0",
    "typescript": "^5.4.0",
    "vitest": "^2.0.0",
    "@changesets/cli": "^2.27.0",
    "eslint": "^9.0.0",
    "prettier": "^3.2.0",
    "@size-limit/preset-small-lib": "^11.0.0",
    "size-limit": "^11.0.0",
    "clean-pkg-json": "^1.2.0"
  },

  "peerDependencies": {},
  "peerDependenciesMeta": {},
  "dependencies": {},
  "overrides": {},
  "publishConfig": {
    "access": "public",
    "registry": "https://registry.npmjs.org"
  },
  "size-limit": [
    {
      "path": "dist/index.js",
      "limit": "10 KB"
    }
  ]
}
```

### 2.2 exportsフィールドの詳細解説

`exports`フィールドはNode.js 12.7.0で導入され、パッケージのエントリポイントを厳密に制御する最も重要な設定である。

```
exports の条件解決フロー:

  import { Client } from '@example/sdk'
       |
       v
  exports["."] を参照
       |
       +-- ESM (import文) で読み込み?
       |     |
       |     +-- "import" 条件にマッチ
       |           |
       |           +-- TypeScript? → "types" を参照
       |           |     → ./dist/index.d.ts
       |           |
       |           +-- ランタイム → "default" を参照
       |                 → ./dist/index.js
       |
       +-- CJS (require) で読み込み?
             |
             +-- "require" 条件にマッチ
                   |
                   +-- TypeScript? → "types" を参照
                   |     → ./dist/index.d.cts
                   |
                   +-- ランタイム → "default" を参照
                         → ./dist/index.cjs

  条件の優先順位（上から順に評価）:
    1. "types"     → TypeScript型解決
    2. "import"    → ESM環境
    3. "require"   → CJS環境
    4. "node"      → Node.js環境
    5. "browser"   → ブラウザ環境
    6. "default"   → フォールバック

  重要: "types" は必ず各条件ブロックの最初に置く
```

#### サブパスexportsのパターン

```json
{
  "exports": {
    ".": "./dist/index.js",

    "./utils": "./dist/utils/index.js",
    "./utils/*": "./dist/utils/*.js",

    "./internal/*": null,

    "./package.json": "./package.json"
  }
}
```

サブパスexportsの設計で重要なのは、内部モジュールへの直接アクセスを防ぐことである。`"./internal/*": null`のように明示的にnullを指定することで、`@example/sdk/internal/secret`のようなインポートをエラーにできる。

### 2.3 依存関係フィールドの使い分け

```
依存関係フィールドの判断フローチャート:

  このモジュールは...
       |
       +-- ランタイムで必要?
       |     |
       |     +-- バンドルに含める? → dependencies
       |     |
       |     +-- 利用者が用意? → peerDependencies
       |           |
       |           +-- 無くても動く? → peerDependenciesMeta
       |                               { "optional": true }
       |
       +-- ビルド・テストのみ? → devDependencies
       |
       +-- バンドル済みで配布? → bundleDependencies
       |
       +-- 代替パッケージ? → optionalDependencies
```

| フィールド | 用途 | npm install時 | 具体例 |
|-----------|------|---------------|-------|
| dependencies | ランタイム必須 | インストールされる | zod, jose |
| devDependencies | 開発時のみ | 利用者にはインストールされない | vitest, tsup, eslint |
| peerDependencies | 利用者が提供 | npm 7+で自動インストール | react, vue |
| peerDependenciesMeta | peerの詳細設定 | optional指定等 | - |
| optionalDependencies | あれば使う | 失敗しても続行 | fsevents |
| bundleDependencies | バンドル同梱 | tarballに含まれる | - |
| overrides | バージョン強制 | 推移的依存を上書き | セキュリティ修正 |

### 2.4 scriptsフィールドのベストプラクティス

npmスクリプトはパッケージ開発における自動化の中核を担う。ライフサイクルスクリプトとカスタムスクリプトを適切に活用する。

```
npm ライフサイクルスクリプトの実行順序:

  npm publish 実行時:
    prepublishOnly → prepare → prepack → postpack → publish → postpublish

  npm install 実行時（依存として）:
    preinstall → install → postinstall → prepare

  npm test 実行時:
    pretest → test → posttest

  推奨するライフサイクルスクリプト:
    "prepare":        ビルド（git clone後に自動実行）
    "prepublishOnly": テスト + 型チェック + ビルド
    "prepack":        package.json のクリーンアップ

  避けるべきスクリプト:
    "postinstall":    セキュリティリスク（任意コード実行）
    "preinstall":     同上
```

---

## 3. ESM/CJSデュアルパッケージ構築

### 3.1 なぜデュアルパッケージが必要か

Node.jsのモジュールシステムはESM（ECMAScript Modules）とCJS（CommonJS）の2つが共存している。2025年現在でもCJS環境を使うプロジェクトは多数存在し、パッケージ作者は両方のフォーマットを提供する必要がある。

```
Node.js モジュールシステムの歴史と現状:

  2009  Node.js誕生 → CJS (require/module.exports)
  2015  ES2015仕様 → ESM 仕様策定 (import/export)
  2017  Node.js 8   → ESM 実験的サポート（--experimental-modules）
  2019  Node.js 12  → exports フィールド導入
  2020  Node.js 14  → ESM 安定版サポート
  2021  Node.js 16  → package.json "type": "module"
  2023  Node.js 20  → require(esm) 実験的サポート
  2024  Node.js 22  → require(esm) 安定化開始

  現在のエコシステム状況:
    ESMのみ:     新しいフレームワーク（Nuxt 3, SvelteKit等）
    CJSのみ:     レガシープロジェクト、一部のツール
    デュアル:     ほとんどの広く使われるパッケージ
    → パッケージ作者はデュアル対応が推奨される
```

### 3.2 tsupによるビルド設定

tsupはesbuildベースの高速バンドラーで、ESM/CJSデュアルパッケージの構築に最適なツールである。

```typescript
// tsup.config.ts - 基本設定
import { defineConfig } from 'tsup';

export default defineConfig({
  // エントリポイント
  entry: [
    'src/index.ts',
    'src/users/index.ts',
    'src/billing/index.ts',
  ],

  // 出力フォーマット
  format: ['esm', 'cjs'],

  // 型定義ファイルの生成
  dts: true,

  // コード分割（ESMのみ有効）
  splitting: true,

  // ソースマップ
  sourcemap: true,

  // ビルド前にdistをクリーンアップ
  clean: true,

  // minify設定（SDKはminifyしない）
  minify: false,

  // ターゲット環境
  target: 'es2022',

  // 出力ディレクトリ
  outDir: 'dist',

  // 外部依存（バンドルしない）
  external: [],

  // バナー（ライセンスヘッダー等）
  banner: {
    js: '/* @example/sdk - MIT License */',
  },

  // shims（import.meta.url等のCJS互換）
  shims: true,

  // 環境変数の定義
  define: {
    'process.env.SDK_VERSION': JSON.stringify('1.0.0'),
  },

  // ビルド後のフック
  onSuccess: 'echo "Build completed successfully"',
});
```

```typescript
// tsup.config.ts - 高度な設定（環境別ビルド）
import { defineConfig } from 'tsup';

export default defineConfig([
  // Node.js向けビルド
  {
    entry: ['src/index.ts'],
    format: ['esm', 'cjs'],
    dts: true,
    platform: 'node',
    target: 'node18',
    outDir: 'dist',
    clean: true,
    splitting: true,
    sourcemap: true,
    external: ['ws'],
  },
  // ブラウザ向けビルド
  {
    entry: ['src/index.browser.ts'],
    format: ['esm'],
    dts: true,
    platform: 'browser',
    target: 'es2022',
    outDir: 'dist/browser',
    globalName: 'ExampleSDK',
    minify: true,
    sourcemap: true,
    noExternal: [/.*/],  // 全依存をバンドル
  },
]);
```

### 3.3 ビルドツール比較

| 観点 | tsup | rollup | esbuild | tsc | unbuild |
|------|------|--------|---------|-----|---------|
| ビルド速度 | 非常に速い | 普通 | 最速 | 遅い | 速い |
| ESM+CJS出力 | 1コマンド | プラグイン必要 | 設定必要 | 別々にビルド | 1コマンド |
| 型定義生成 | 内蔵 | プラグイン必要 | 非対応 | 内蔵 | 内蔵 |
| Tree-shaking | 良好 | 最良 | 良好 | なし | 良好 |
| 設定の簡潔さ | 非常に簡潔 | 複雑 | 簡潔 | 中程度 | 簡潔 |
| プラグイン | esbuild互換 | 豊富 | 限定的 | なし | rollup互換 |
| ユースケース | SDK/ライブラリ | 大規模ライブラリ | 速度重視 | 型チェック | モノレポ |
| 推奨度 | 高（汎用） | 高（大規模） | 中 | 低（ビルド用途） | 高（モノレポ） |

### 3.4 Dual Package Hazard（二重パッケージ問題）

ESMとCJSの両方を提供する場合、同じパッケージがESMとCJSの両方として読み込まれ、シングルトンが2つ作られる「Dual Package Hazard」に注意が必要である。

```
Dual Package Hazard の発生パターン:

  App (ESM)
    |
    +-- import { Client } from '@example/sdk'
    |     → dist/index.js (ESM版) をロード
    |     → Client のインスタンスを作成
    |
    +-- require('@example/sdk') （CJS依存経由）
          → dist/index.cjs (CJS版) をロード
          → 別の Client クラスがロードされる
          → instanceof チェックが失敗する!

  対策:
    1. ステートレスな設計にする（推奨）
       → グローバル状態を持たない
       → instanceof ではなくダックタイピング

    2. CJS版をESM版のラッパーにする
       // dist/index.cjs
       module.exports = require('./index.js');
       → ただし動的 import が必要

    3. package.json で "type": "module" を設定し、
       CJS利用者には明示的にラッパーを提供
```

---

## 4. TypeScript設定の詳細

### 4.1 パッケージ開発用 tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],

    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,

    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,

    "outDir": "dist",
    "rootDir": "src",

    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    "resolveJsonModule": true,

    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "exactOptionalPropertyTypes": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "**/*.test.ts", "**/*.spec.ts"]
}
```

### 4.2 重要なTypeScriptコンパイラオプション解説

```
moduleResolution の選択ガイド:

  "bundler"（推奨: tsup/rollup使用時）
    → import './foo' で .ts ファイルを解決
    → exports フィールドを正しく解釈
    → 拡張子なしインポートが可能

  "node16" / "nodenext"（推奨: tsc直接使用時）
    → Node.js のモジュール解決に完全準拠
    → .js 拡張子が必須
    → exports フィールドを正しく解釈

  "node"（非推奨）
    → レガシーな解決方式
    → exports フィールドを無視する場合がある

  verbatimModuleSyntax: true（推奨）
    → import type { Foo } の明示が必須
    → 型のみのインポートを正しく区別
    → ビルド時の不要なインポート除去が確実
```

### 4.3 型定義ファイルの品質管理

型定義の品質はパッケージのユーザー体験に直結する。TypeScriptユーザーにとって、型定義はドキュメントそのものである。

```typescript
// src/types.ts - 丁寧な型定義の例

/**
 * SDK クライアントの設定オプション。
 *
 * @example
 * ```typescript
 * const client = new ExampleClient({
 *   apiKey: 'sk-xxx',
 *   baseURL: 'https://api.example.com',
 *   timeout: 30_000,
 * });
 * ```
 */
export interface ClientOptions {
  /**
   * APIキー。環境変数 `EXAMPLE_API_KEY` からも読み取り可能。
   * @see https://example.com/docs/authentication
   */
  apiKey: string;

  /**
   * APIのベースURL。デフォルトは `https://api.example.com/v1`。
   * @default "https://api.example.com/v1"
   */
  baseURL?: string;

  /**
   * リクエストのタイムアウト（ミリ秒）。
   * @default 30000
   */
  timeout?: number;

  /**
   * リトライの最大回数。0 でリトライ無効。
   * @default 3
   */
  maxRetries?: number;

  /**
   * カスタムfetch関数。テスト時のモック注入等に使用。
   */
  fetch?: typeof globalThis.fetch;

  /**
   * カスタムヘッダー。全リクエストに付与される。
   */
  defaultHeaders?: Record<string, string>;
}

/**
 * ページネーションされたレスポンスの共通型。
 * @typeParam T - リスト内の要素の型
 */
export interface PaginatedResponse<T> {
  /** データの配列 */
  data: T[];
  /** 次のページが存在するか */
  hasMore: boolean;
  /** 次ページ取得用のカーソル */
  cursor?: string;
  /** 結果の総数（取得可能な場合） */
  totalCount?: number;
}

/**
 * APIエラーレスポンスの型。
 */
export interface APIError {
  /** エラーコード（例: "NOT_FOUND", "RATE_LIMITED"） */
  code: string;
  /** 人間が読めるエラーメッセージ */
  message: string;
  /** エラーの詳細情報 */
  details?: Record<string, unknown>;
  /** リクエストID（サポート問い合わせ時に使用） */
  requestId?: string;
}
```

---

## 5. ゼロ依存設計の原則と実践

### 5.1 なぜゼロ依存を目指すのか

```
依存の数とリスクの関係:

  依存 0個:  リスク最小 ████
  依存 1-3個: リスク低   ████████
  依存 4-10個: リスク中  ████████████████
  依存 10+個: リスク高   ████████████████████████████

  主なリスク:
    1. サプライチェーン攻撃
       → 依存パッケージが乗っ取られる
       → event-stream事件（2018年）が有名
       → 推移的依存まで含めると影響範囲が巨大

    2. バージョン競合
       → 利用者の他の依存とバージョンが衝突
       → node_modules の肥大化
       → デバッグが困難

    3. メンテナンス負荷
       → 依存のアップデート対応
       → 非推奨化への追従
       → ライセンス互換性の確認

    4. バンドルサイズ増加
       → Tree-shakingが効かない依存
       → 利用者のアプリサイズに影響
```

### 5.2 Node.js 組み込みAPIによる代替

```typescript
// ゼロ依存で実現するユーティリティ集

// --- UUID生成（uuid パッケージ不要） ---
function generateId(): string {
  return crypto.randomUUID();
}

// --- ディープクローン（lodash.cloneDeep 不要） ---
function deepClone<T>(obj: T): T {
  return structuredClone(obj);
}

// --- クエリ文字列（qs パッケージ不要） ---
function buildQueryString(params: Record<string, string>): string {
  return new URLSearchParams(params).toString();
}

// --- Base64エンコード（buffer パッケージ不要） ---
function toBase64(str: string): string {
  return btoa(str);
}

// --- SHA-256ハッシュ（crypto-js パッケージ不要） ---
async function sha256(message: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(message);
  const hash = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(hash))
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

// --- リトライ（p-retry パッケージ不要） ---
async function withRetry<T>(
  fn: () => Promise<T>,
  options: { maxRetries: number; baseDelay: number } = {
    maxRetries: 3,
    baseDelay: 1000,
  },
): Promise<T> {
  let lastError: Error | undefined;
  for (let attempt = 0; attempt <= options.maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (attempt < options.maxRetries) {
        const delay = options.baseDelay * Math.pow(2, attempt);
        const jitter = delay * 0.1 * Math.random();
        await new Promise(r => setTimeout(r, delay + jitter));
      }
    }
  }
  throw lastError;
}

// --- タイムアウト付きfetch（node-fetch 不要、Node.js 18+） ---
async function fetchWithTimeout(
  url: string,
  options: RequestInit & { timeout?: number } = {},
): Promise<Response> {
  const { timeout = 30_000, ...fetchOptions } = options;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);

  try {
    return await fetch(url, {
      ...fetchOptions,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}
```

### 5.3 依存が許容されるケース

| カテゴリ | パッケージ例 | 理由 |
|---------|-------------|------|
| 暗号・認証 | jose, @noble/hashes | セキュリティ実装は専門家のコードを使うべき |
| Protocol Buffers | protobuf.js | プロトコル仕様の実装が複雑 |
| WebSocket (Node.js) | ws | Node.js組み込みが不十分 |
| 圧縮 | fflate | WASM実装でパフォーマンスが重要 |
| バリデーション | zod | 型推論との統合が複雑 |

---

## 6. セマンティックバージョニング詳細

### 6.1 SemVerの3つの数字

```
SemVer: MAJOR.MINOR.PATCH

  MAJOR（破壊的変更）: 1.0.0 → 2.0.0
    具体例:
    - 公開メソッドの削除またはリネーム
    - 引数の型変更（string → number）
    - 必須パラメータの追加
    - デフォルト動作の変更
    - 最小Node.jsバージョンの引き上げ
    - 例外の型変更

  MINOR（後方互換の機能追加）: 1.0.0 → 1.1.0
    具体例:
    - 新しいメソッド・クラスの追加
    - オプショナルパラメータの追加
    - 新しいイベントの追加
    - 新しいエクスポートの追加
    - 非推奨マーキング（@deprecated）

  PATCH（バグ修正）: 1.0.0 → 1.0.1
    具体例:
    - バグ修正
    - パフォーマンス改善
    - ドキュメントの修正
    - devDependenciesの更新
    - 内部リファクタリング（外部挙動は変わらない）

  プレリリース:
    1.0.0-alpha.1   → 初期テスト版（APIが不安定）
    1.0.0-beta.1    → 機能完成版（バグ修正中）
    1.0.0-rc.1      → リリース候補（重大バグのみ修正）
```

### 6.2 バージョン判断のグレーゾーン

```
判断が難しいケース:

  Q: TypeScriptの型を厳密化した（anyをstringに変更）
  A: → MINOR（型の厳密化は利用者のコードを壊す可能性）
     → ただし、明らかなバグ修正ならPATCH

  Q: エラーメッセージを変更した
  A: → PATCH（エラーメッセージは公開APIではない）
     → ただし、正規表現でパースしている利用者がいる可能性

  Q: パフォーマンスを大幅に改善した
  A: → PATCH（外部動作は変わらない）
     → ただし、メモリ使用量の変化で影響がある場合はMINOR

  Q: Node.js 16のサポートを終了した
  A: → MAJOR（利用者の環境を制限する変更）

  Q: 新しいオプションを追加し、デフォルト値を設定した
  A: → MINOR（既存コードは変更なしで動く）
     → ただし、デフォルト値が既存の動作を変える場合はMAJOR
```

### 6.3 Changesetsによるバージョン管理

```bash
# Changesets の初期設定
npx changeset init
# → .changeset/ ディレクトリが作成される

# 変更を記録（PRごとに実行）
npx changeset
# インタラクティブに:
#   1. 変更があるパッケージを選択
#   2. 変更の種類を選択（major / minor / patch）
#   3. 変更の説明を記入

# バージョンアップ + CHANGELOG 更新
npx changeset version
# → package.json の version が更新される
# → CHANGELOG.md が自動生成される

# npm に公開
npx changeset publish
# → npm publish が実行される
# → git tag が作成される
```

```
.changeset/ ディレクトリ構造:

  .changeset/
  ├── config.json           ← Changesets の設定
  ├── README.md             ← 説明
  ├── brave-fans-dance.md   ← 変更記録1（ランダム名）
  └── shy-maps-grin.md      ← 変更記録2

  変更記録ファイルの例（brave-fans-dance.md）:
  ---
  "@example/sdk": minor
  ---

  ユーザー管理APIにバッチ取得メソッドを追加。
  `client.users.list()` で最大100件の一括取得が可能。
```

---

## 7. テスト戦略

### 7.1 テスト設定

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.test.ts', 'src/**/*.spec.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov', 'html'],
      include: ['src/**/*.ts'],
      exclude: [
        'src/**/*.test.ts',
        'src/**/*.spec.ts',
        'src/**/types.ts',
        'src/**/index.ts',  // re-exports のみのファイル
      ],
      thresholds: {
        branches: 80,
        functions: 80,
        lines: 80,
        statements: 80,
      },
    },
    testTimeout: 10_000,
    hookTimeout: 10_000,
  },
});
```

### 7.2 MSWを使ったHTTPモックテスト

```typescript
// src/__tests__/client.test.ts
import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { ExampleClient } from '../index';

// MSWサーバーのセットアップ
const server = setupServer();

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('ExampleClient', () => {
  const client = new ExampleClient({
    apiKey: 'test-key-xxx',
    baseURL: 'https://api.example.com/v1',
  });

  describe('users.get()', () => {
    it('IDを指定してユーザーを取得できる', async () => {
      server.use(
        http.get('https://api.example.com/v1/users/123', ({ request }) => {
          // 認証ヘッダーの検証
          expect(request.headers.get('Authorization')).toBe(
            'Bearer test-key-xxx',
          );
          return HttpResponse.json({
            id: '123',
            name: 'Tanaka Taro',
            email: 'taro@example.com',
            role: 'admin',
            createdAt: '2024-01-01T00:00:00Z',
          });
        }),
      );

      const user = await client.users.get('123');

      expect(user.id).toBe('123');
      expect(user.name).toBe('Tanaka Taro');
      expect(user.role).toBe('admin');
    });

    it('存在しないユーザーで404エラーが返る', async () => {
      server.use(
        http.get('https://api.example.com/v1/users/999', () => {
          return HttpResponse.json(
            {
              code: 'NOT_FOUND',
              message: 'User not found',
              requestId: 'req-abc-123',
            },
            { status: 404 },
          );
        }),
      );

      await expect(client.users.get('999')).rejects.toMatchObject({
        code: 'NOT_FOUND',
        status: 404,
      });
    });

    it('500エラー時にリトライが行われる', async () => {
      let attempts = 0;
      server.use(
        http.get('https://api.example.com/v1/users/123', () => {
          attempts++;
          if (attempts < 3) {
            return HttpResponse.json(
              { code: 'INTERNAL_ERROR', message: 'Server error' },
              { status: 500 },
            );
          }
          return HttpResponse.json({
            id: '123',
            name: 'Tanaka Taro',
          });
        }),
      );

      const user = await client.users.get('123');
      expect(user.name).toBe('Tanaka Taro');
      expect(attempts).toBe(3);
    });

    it('タイムアウト時に適切なエラーが返る', async () => {
      server.use(
        http.get('https://api.example.com/v1/users/123', async () => {
          // 意図的に遅延を入れる
          await new Promise(resolve => setTimeout(resolve, 15_000));
          return HttpResponse.json({ id: '123' });
        }),
      );

      const timeoutClient = new ExampleClient({
        apiKey: 'test-key',
        timeout: 1000,
        maxRetries: 0,
      });

      await expect(timeoutClient.users.get('123')).rejects.toThrow(
        'Request timed out',
      );
    });
  });

  describe('users.list()', () => {
    it('ページネーション付きでユーザー一覧を取得できる', async () => {
      server.use(
        http.get('https://api.example.com/v1/users', ({ request }) => {
          const url = new URL(request.url);
          const limit = url.searchParams.get('limit') ?? '20';
          const cursor = url.searchParams.get('cursor');

          return HttpResponse.json({
            data: [
              { id: '1', name: 'User 1' },
              { id: '2', name: 'User 2' },
            ],
            hasMore: cursor === null,
            cursor: cursor === null ? 'cursor-abc' : undefined,
          });
        }),
      );

      const page1 = await client.users.list({ limit: 2 });
      expect(page1.data).toHaveLength(2);
      expect(page1.hasMore).toBe(true);

      const page2 = await client.users.list({
        limit: 2,
        cursor: page1.cursor,
      });
      expect(page2.hasMore).toBe(false);
    });
  });
});
```

---

## 8. 公開ワークフロー

### 8.1 公開前チェックリスト

```
公開前チェックリスト（必須）:

  コード品質:
    [x] テストが全て通る（npm test）
    [x] 型チェックが通る（npm run typecheck）
    [x] lint エラーがない（npm run lint）
    [x] フォーマットが統一されている（npm run format:check）

  パッケージ設定:
    [x] package.json の version が正しい
    [x] exports フィールドが正しく設定されている
    [x] files フィールドで不要ファイルが除外されている
    [x] engines フィールドが設定されている
    [x] license ファイルが含まれている

  ドキュメント:
    [x] README.md が最新
    [x] CHANGELOG.md が更新されている
    [x] 型定義にJSDocコメントがある

  セキュリティ:
    [x] .env ファイルが含まれていない
    [x] APIキーやシークレットが含まれていない
    [x] npm audit で脆弱性がない

  確認:
    [x] npm pack --dry-run で内容を確認
    [x] npm pack → tarball を展開して検証
```

### 8.2 npm pack による事前確認

```bash
# 含まれるファイルの確認
npm pack --dry-run

# 出力例:
# npm notice Tarball Contents
# npm notice 1.2kB  package.json
# npm notice 4.5kB  README.md
# npm notice 1.1kB  LICENSE
# npm notice 12.3kB dist/index.js
# npm notice 11.8kB dist/index.cjs
# npm notice 8.4kB  dist/index.d.ts
# npm notice 8.2kB  dist/index.d.cts
# npm notice === Tarball Details ===
# npm notice name:          @example/sdk
# npm notice version:       1.0.0
# npm notice filename:      example-sdk-1.0.0.tgz
# npm notice package size:  15.2 kB
# npm notice unpacked size: 47.5 kB
# npm notice total files:   8

# 実際にtarballを作成して検証
npm pack
tar -xzf example-sdk-1.0.0.tgz
ls package/
# → dist/  LICENSE  package.json  README.md
```

### 8.3 GitHub Actions での自動公開

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    branches: [main]

concurrency: ${{ github.workflow }}-${{ github.ref }}

jobs:
  release:
    name: Release
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      id-token: write  # npm provenance

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          registry-url: 'https://registry.npmjs.org'

      - name: Install dependencies
        run: npm ci

      - name: Build
        run: npm run build

      - name: Test
        run: npm test

      - name: Type check
        run: npm run typecheck

      - name: Create Release PR or Publish
        id: changesets
        uses: changesets/action@v1
        with:
          publish: npx changeset publish
          version: npx changeset version
          commit: 'chore: release packages'
          title: 'chore: release packages'
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## 9. モノレポ管理戦略

### 9.1 モノレポとは

モノレポ（Monorepo）は、複数のパッケージを1つのリポジトリで管理する手法である。大規模なSDKやフレームワークでは、モノレポが事実上の標準となっている。

```
モノレポの構造例:

  my-sdk/
  ├── package.json              ← ルート（private: true）
  ├── pnpm-workspace.yaml       ← ワークスペース定義
  ├── turbo.json                ← Turborepo設定
  ├── .changeset/
  │   └── config.json           ← Changesets設定
  ├── packages/
  │   ├── core/                 ← @my-sdk/core
  │   │   ├── package.json
  │   │   ├── src/
  │   │   ├── tsup.config.ts
  │   │   └── tsconfig.json
  │   ├── react/                ← @my-sdk/react
  │   │   ├── package.json      ← peerDep: react
  │   │   ├── src/
  │   │   └── tsconfig.json
  │   ├── vue/                  ← @my-sdk/vue
  │   │   ├── package.json      ← peerDep: vue
  │   │   ├── src/
  │   │   └── tsconfig.json
  │   └── cli/                  ← @my-sdk/cli
  │       ├── package.json
  │       ├── src/
  │       └── tsconfig.json
  ├── apps/
  │   ├── docs/                 ← ドキュメントサイト
  │   └── playground/           ← デモアプリ
  └── tooling/
      ├── eslint-config/        ← 共有ESLint設定
      ├── tsconfig/             ← 共有TypeScript設定
      └── prettier-config/      ← 共有Prettier設定
```

### 9.2 モノレポツール比較

| 観点 | pnpm workspaces | npm workspaces | Turborepo | Nx | Lerna |
|------|----------------|----------------|-----------|-----|-------|
| パッケージ管理 | pnpm | npm | npm/pnpm/yarn | npm/pnpm/yarn | npm/yarn |
| タスク実行 | なし | なし | 並列・キャッシュ | 並列・キャッシュ | 並列 |
| ビルドキャッシュ | なし | なし | ローカル+リモート | ローカル+リモート | なし |
| 依存関係グラフ | 基本的 | 基本的 | 自動検出 | 高度 | 基本的 |
| 設定の簡潔さ | 非常に簡潔 | 非常に簡潔 | 簡潔 | やや複雑 | 中程度 |
| 学習コスト | 低 | 低 | 低〜中 | 中〜高 | 低 |
| 推奨用途 | 小〜中規模 | 小規模 | 中〜大規模 | 大規模 | レガシー |

### 9.3 pnpm + Turborepo のセットアップ

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'apps/*'
  - 'tooling/*'
```

```json
// ルート package.json
{
  "name": "my-sdk-monorepo",
  "private": true,
  "scripts": {
    "build": "turbo build",
    "test": "turbo test",
    "lint": "turbo lint",
    "typecheck": "turbo typecheck",
    "dev": "turbo dev",
    "clean": "turbo clean",
    "format": "prettier --write '**/*.{ts,tsx,json,md}'"
  },
  "devDependencies": {
    "turbo": "^2.0.0",
    "prettier": "^3.2.0",
    "@changesets/cli": "^2.27.0"
  },
  "packageManager": "pnpm@9.0.0"
}
```

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tsup.config.ts", "tsconfig.json"],
      "outputs": ["dist/**"],
      "cache": true
    },
    "test": {
      "dependsOn": ["build"],
      "inputs": ["src/**", "vitest.config.ts"],
      "outputs": [],
      "cache": true
    },
    "lint": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "eslint.config.js"],
      "outputs": [],
      "cache": true
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "inputs": ["src/**", "tsconfig.json"],
      "outputs": [],
      "cache": true
    },
    "dev": {
      "dependsOn": ["^build"],
      "cache": false,
      "persistent": true
    },
    "clean": {
      "cache": false
    }
  }
}
```

### 9.4 モノレポ内パッケージの相互参照

```json
// packages/react/package.json
{
  "name": "@my-sdk/react",
  "version": "1.0.0",
  "dependencies": {
    "@my-sdk/core": "workspace:*"
  },
  "peerDependencies": {
    "react": "^18.0.0 || ^19.0.0",
    "react-dom": "^18.0.0 || ^19.0.0"
  },
  "peerDependenciesMeta": {
    "react-dom": {
      "optional": true
    }
  }
}
```

```
モノレポの依存関係グラフ:

  @my-sdk/react ──depends──> @my-sdk/core
       |                          |
       +──peer──> react           +──(ゼロ依存)
       +──peer──> react-dom

  @my-sdk/vue ───depends──> @my-sdk/core
       |
       +──peer──> vue

  @my-sdk/cli ───depends──> @my-sdk/core
       |
       +──dep───> commander
       +──dep───> chalk

  ビルド順序（Turborepoが自動解決）:
    1. @my-sdk/core        （依存なし）
    2. @my-sdk/react       （coreに依存）
       @my-sdk/vue         （coreに依存、並列実行可）
       @my-sdk/cli         （coreに依存、並列実行可）

  "workspace:*" は公開時に実際のバージョンに置換される:
    開発時: "@my-sdk/core": "workspace:*"
    公開時: "@my-sdk/core": "^1.0.0"
```

### 9.5 共有設定の管理

```json
// tooling/tsconfig/base.json - 共有TypeScript設定
{
  "$schema": "https://json.schemastore.org/tsconfig",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "lib": ["ES2022"],
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    "resolveJsonModule": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "exactOptionalPropertyTypes": true,
    "noUncheckedIndexedAccess": true
  },
  "exclude": ["node_modules", "dist"]
}
```

```json
// packages/core/tsconfig.json - 各パッケージのTypeScript設定
{
  "extends": "../../tooling/tsconfig/base.json",
  "compilerOptions": {
    "outDir": "dist",
    "rootDir": "src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## 10. パッケージサイズの最適化

### 10.1 サイズが重要な理由

パッケージサイズは利用者のインストール時間、CI/CDの実行時間、そしてバンドルサイズに直接影響する。特にフロントエンドで使用されるパッケージでは、バンドルサイズの削減が極めて重要である。

```
パッケージサイズの測定ポイント:

  +-------------------+
  |   npm パッケージ   |
  +-------------------+
         |
         v
  1. Install Size（インストールサイズ）
     → npm install 時にダウンロードされる合計サイズ
     → 推移的依存を含む
     → 目安: SDK なら 1MB 以下

  2. Publish Size（公開サイズ）
     → npm pack で生成される tarball のサイズ
     → files フィールドで制御
     → 目安: 100KB 以下

  3. Bundle Size（バンドルサイズ）
     → webpack/vite 等でバンドルした際のサイズ
     → Tree-shaking の効果に依存
     → 目安: gzip 後 10KB 以下（ライブラリ）

  測定ツール:
    npm pack --dry-run          → 公開サイズ
    npx size-limit              → バンドルサイズ
    https://bundlephobia.com    → オンラインで確認
    https://pkg-size.dev        → より詳細な分析
```

### 10.2 size-limit の設定

```json
// package.json に追加
{
  "size-limit": [
    {
      "path": "dist/index.js",
      "import": "{ ExampleClient }",
      "limit": "10 KB"
    },
    {
      "path": "dist/users/index.js",
      "import": "{ UsersResource }",
      "limit": "3 KB"
    },
    {
      "path": "dist/index.js",
      "import": "*",
      "limit": "15 KB"
    }
  ]
}
```

```yaml
# .github/workflows/size.yml - PRごとにサイズを測定
name: Size Check

on: [pull_request]

jobs:
  size:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: andresz1/size-limit-action@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          # PRコメントにサイズ変更を表示
```

### 10.3 Tree-shakingの最適化

```typescript
// 悪い例: バレルファイルで全てを再エクスポート
// src/index.ts
export * from './users';
export * from './billing';
export * from './analytics';
export * from './utils';
// → import { Users } from '@example/sdk' で
//   billing, analytics, utils もバンドルされる可能性

// 良い例: サブパスエクスポートで分割
// package.json の exports で個別にエントリポイントを定義
// → import { Users } from '@example/sdk/users'
//   users モジュールのみがバンドルされる

// Tree-shaking を妨げるパターン:
// 1. クラスの static プロパティへの副作用のある代入
class Client {
  // 悪い: 副作用がある
  static instances = new Map();  // モジュール読み込み時に実行される
}

// 2. トップレベルの副作用
console.log('SDK loaded');  // 副作用あり → Tree-shake不可
const config = loadConfig(); // 関数呼び出し → 副作用の可能性

// 3. enumの使用
enum Status { Active, Inactive }
// → コンパイル後に即時実行関数(IIFE)になる → Tree-shake不可

// 良い代替: const object + as const
const Status = {
  Active: 'active',
  Inactive: 'inactive',
} as const;
type Status = typeof Status[keyof typeof Status];
// → 純粋なオブジェクトリテラル → Tree-shake可能
```

---

## 11. セキュリティと品質管理

### 11.1 npm provenance（出所証明）

npm provenanceは、パッケージがどのソースコードからどのCI環境で構築されたかを暗号的に証明する仕組みである。

```bash
# provenance付きで公開
npm publish --provenance

# GitHub Actionsで自動化する場合:
# permissions に id-token: write が必要
# registry-url の設定が必要
```

```
npm provenance の仕組み:

  開発者のコード
       |
       v
  GitHub リポジトリ
       |
       v
  GitHub Actions（CI）
       |
       +-- OIDC トークンを発行
       |
       v
  npm publish --provenance
       |
       +-- Sigstore で署名
       |
       v
  npm レジストリ
       |
       +-- パッケージ + 署名 + ビルド情報
       |
       v
  利用者が検証可能:
    - どのリポジトリのコードか
    - どのコミットからビルドされたか
    - どのCI環境で実行されたか
    - ビルドログへのリンク
```

### 11.2 セキュリティチェックリスト

```
パッケージのセキュリティ対策:

  開発時:
    [x] npm audit を定期実行
    [x] dependabot / renovate で依存を自動更新
    [x] Socket.dev でサプライチェーンリスクを監視
    [x] .npmrc に ignore-scripts=true を設定

  公開時:
    [x] 2FA を有効化（npm login）
    [x] provenance を有効化
    [x] npm token の権限を最小限に
    [x] CODEOWNERS で公開権限を制限

  パッケージの設計:
    [x] eval() / Function() を使わない
    [x] 動的 require() を避ける
    [x] ユーザー入力をサニタイズ
    [x] prototype pollution 対策
    [x] ReDoS（正規表現DoS）対策

  .npmrc の推奨設定:
    //registry.npmjs.org/:_authToken=${NPM_TOKEN}
    ignore-scripts=true
    audit=true
    fund=false
```

### 11.3 品質指標の自動測定

```yaml
# .github/workflows/quality.yml
name: Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build

      # テストカバレッジ
      - run: npm run test:coverage
      - uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

      # バンドルサイズ
      - run: npx size-limit

      # 型チェック
      - run: npm run typecheck

      # lint
      - run: npm run lint

      # セキュリティ監査
      - run: npm audit --production

      # ライセンスチェック
      - run: npx license-checker --onlyAllow 'MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC'
```

---

## 12. アンチパターンと対策

### 12.1 アンチパターン: 肥大化するバレルファイル

```typescript
// [アンチパターン] 巨大なバレルファイル
// src/index.ts
export * from './users';
export * from './billing';
export * from './analytics';
export * from './notifications';
export * from './webhooks';
export * from './admin';
export * from './utils';
export * from './errors';
export * from './types';
export * from './constants';
// → 全モジュールが1つのエントリポイントに集約
// → Tree-shakingが効きにくくなる
// → 利用者のバンドルサイズが肥大化
// → 循環参照のリスクが増大
// → IDE の自動補完が遅くなる

// [対策] サブパスエクスポートで分割
// package.json
// {
//   "exports": {
//     ".":           "./dist/index.js",      ← コアのみ
//     "./users":     "./dist/users/index.js",
//     "./billing":   "./dist/billing/index.js",
//     "./analytics": "./dist/analytics/index.js"
//   }
// }

// src/index.ts - コアのみエクスポート
export { ExampleClient } from './client';
export type { ClientOptions, APIError } from './types';

// 利用者側:
// import { ExampleClient } from '@example/sdk';
// import { UsersResource } from '@example/sdk/users';
// → 必要なモジュールのみがバンドルされる
```

### 12.2 アンチパターン: peerDependenciesの誤用

```
[アンチパターン] peerDependencies を dependencies に入れる

  // 悪い例: React をdependencies に入れたReactコンポーネントライブラリ
  {
    "name": "my-react-library",
    "dependencies": {
      "react": "^18.0.0"     // ← これが問題
    }
  }

  発生する問題:
    1. 利用者のプロジェクトに別バージョンのReactが存在
    2. node_modules に2つの React がインストールされる
    3. React の内部状態が共有されず、フックが壊れる
    4. "Invalid hook call" エラーが発生

  node_modules/
  ├── react@18.3.0/          ← 利用者のReact
  ├── my-react-library/
  │   └── node_modules/
  │       └── react@18.2.0/  ← ライブラリ同梱のReact（別インスタンス）

  正しい対策:
  {
    "name": "my-react-library",
    "peerDependencies": {
      "react": "^18.0.0 || ^19.0.0"
    },
    "devDependencies": {
      "react": "^18.3.0"     ← 開発・テスト用
    }
  }

  peerDependencies にすべきもの:
    - フレームワーク本体（react, vue, angular）
    - プラグインのホスト（eslint, webpack, vite）
    - 共有される必要があるライブラリ（同一インスタンスが必須）
```

### 12.3 アンチパターン: .npmignore の使用

```
[アンチパターン] .npmignore で除外ファイルを管理

  問題点:
    - .gitignore と .npmignore の優先順位が複雑
    - 新しいファイルを追加した際に .npmignore の更新を忘れる
    - ソースコードやテストが意図せず公開される
    - 明示的な「含めるもの」ではなく「除外するもの」を管理

  正しい対策: package.json の files フィールドを使う

  {
    "files": [
      "dist",
      "README.md",
      "LICENSE",
      "CHANGELOG.md"
    ]
  }

  files の利点:
    - ホワイトリスト方式（明示的に含めるものを指定）
    - 新しいファイルが意図せず公開されない
    - package.json, README.md, LICENSE は自動的に含まれる
    - npm pack --dry-run で簡単に確認できる
```

---

## 13. エッジケース分析

### 13.1 エッジケース: ESMとCJSの相互運用

```typescript
// エッジケース: CJSからESMモジュールをrequireする

// ESMのみを提供するパッケージをCJSプロジェクトから使う場合
// → require() は使えない（ERR_REQUIRE_ESM エラー）

// Node.js 22 以降:
// require(esm) がサポートされつつあるが、
// トップレベル await を含むモジュールは依然として require 不可

// 対処法1: 動的 import() を使う（CJSでも利用可能）
// cjs-consumer.cjs
async function main() {
  // 動的 import() は CJS でも使える
  const { ExampleClient } = await import('@example/sdk');
  const client = new ExampleClient({ apiKey: 'xxx' });
}
main();

// 対処法2: パッケージ側でCJSビルドを提供する（推奨）
// → tsup で format: ['esm', 'cjs'] を設定

// 対処法3: ラッパーファイルを提供
// dist/index.cjs
// const mod = await import('./index.js');
// module.exports = mod;
// → ただしトップレベル await が必要なので Node.js 14+ のみ

// 注意: default export の扱いが異なる
// ESM: export default class Client {}
// CJS: const { default: Client } = require('@example/sdk');
//      ← ".default" が必要になる場合がある
// → named export を推奨（default export を避ける）
```

### 13.2 エッジケース: TypeScript の moduleResolution による型解決の違い

```
TypeScript moduleResolution とパッケージ型解決:

  利用者のtsconfig.json の moduleResolution 設定によって
  パッケージの型がどのように解決されるかが変わる:

  "node" (レガシー):
    → package.json の "types" フィールドのみ参照
    → "exports" フィールドの "types" は無視される
    → typesVersions によるサブパス解決が必要

  "node16" / "nodenext":
    → "exports" フィールドの "types" を参照
    → 条件分岐（import/require）に基づいて型を解決
    → .d.ts と .d.cts を区別する

  "bundler":
    → "exports" フィールドの "types" を参照
    → 拡張子なしインポートを許容
    → 最も柔軟な設定

  全ての moduleResolution に対応する方法:

  {
    "types": "./dist/index.d.ts",
    "exports": {
      ".": {
        "import": {
          "types": "./dist/index.d.ts",
          "default": "./dist/index.js"
        },
        "require": {
          "types": "./dist/index.d.cts",
          "default": "./dist/index.cjs"
        }
      },
      "./users": {
        "import": {
          "types": "./dist/users/index.d.ts",
          "default": "./dist/users/index.js"
        },
        "require": {
          "types": "./dist/users/index.d.cts",
          "default": "./dist/users/index.cjs"
        }
      }
    },
    "typesVersions": {
      "*": {
        "users": ["./dist/users/index.d.ts"]
      }
    }
  }

  → "types" トップレベル: レガシー moduleResolution 用
  → "exports" 内の "types": node16/nodenext/bundler 用
  → "typesVersions": TypeScript 4.x 以前の互換性用
```

### 13.3 エッジケース: npm publish の取り消しとバージョンの再利用

```
npm unpublish のルールと制約:

  72時間ルール:
    - 公開から72時間以内: unpublish 可能
    - 72時間を超過: unpublish 不可（サポートに連絡が必要）

  バージョンの再利用禁止:
    - 一度公開したバージョン番号は unpublish 後も再利用不可
    - 例: 1.0.0 を公開 → unpublish → 1.0.0 で再公開は不可
    - → 1.0.1 として公開する必要がある

  deprecate（非推奨化）の活用:
    $ npm deprecate @example/sdk@1.0.0 "セキュリティ脆弱性あり。2.0.0に更新してください"
    → パッケージは引き続き利用可能
    → npm install 時に警告メッセージが表示される
    → 全バージョンを deprecate: npm deprecate @example/sdk "このパッケージは非推奨です"

  dist-tag による安全なリリース:
    # ベータ版を "beta" タグで公開
    npm publish --tag beta
    # → npm install @example/sdk@beta でインストール
    # → npm install @example/sdk では latest のまま

    # カナリア版を "canary" タグで公開
    npm publish --tag canary
    # → 自動テスト用、毎日の自動ビルドに使用
```

