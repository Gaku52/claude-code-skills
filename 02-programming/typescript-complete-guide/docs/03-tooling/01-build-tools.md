# TypeScript ビルドツール完全ガイド

> tsc, esbuild, SWC, Vite を比較し、プロジェクトに最適なビルドパイプラインを構築する

## この章で学ぶこと

1. **各ビルドツールの特性** -- tsc, esbuild, SWC, Vite それぞれの設計思想、速度、機能の違い
2. **ビルドパイプラインの設計** -- 型チェックとトランスパイルの分離、開発/本番環境の構成パターン
3. **移行とチューニング** -- 既存プロジェクトのビルド高速化と、ツール間の移行手順
4. **モノレポでのビルド戦略** -- Turborepo, Nx との連携、キャッシュ活用
5. **ライブラリのビルドとパッケージング** -- tsup, unbuild, ESM/CJS デュアル出力


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [tsconfig.json 完全ガイド](./00-tsconfig.md) の内容を理解していること

---

## 1. ビルドツール全体像

### 1-1. TypeScript ビルドの 2 つの役割

```
TypeScript ビルドの分離:

  .ts ファイル
       |
  +----+----+
  |         |
  v         v
型チェック   トランスパイル
(tsc)       (esbuild / SWC / Vite)
  |         |
  v         v
エラー報告   .js ファイル
  |         |
  +----+----+
       |
       v
  デプロイ可能なコード

  ポイント: 型チェックとトランスパイルは分離できる
  tsc は型チェックのみ (--noEmit) に使い、
  高速トランスパイラで JS を生成するのが現代のベストプラクティス
```

### 1-2. 速度比較（10万行プロジェクト目安）

```
ビルド速度比較 (相対値):

  tsc        ████████████████████████████████████  30s
  webpack+ts ████████████████████████████          25s
  Rollup+ts  ████████████████████                  18s
  Vite (dev) ██                                     2s (HMR)
  esbuild    █                                      0.5s
  SWC        █                                      0.4s

  ※ tsc は型チェック込み、他は型チェックなし（トランスパイルのみ）
  ※ esbuild と SWC は型チェックなしの純粋なトランスパイル速度

型チェック + トランスパイルの合計時間:
  tsc only        ████████████████████████████████████  30s
  tsc + esbuild   ████████████████  + █                 15.5s
  tsc + SWC       ████████████████  + █                 15.4s
  tsc + Vite      ████████████████  + ██                17s

  → tsc の型チェック時間は変わらないが、
    トランスパイルは高速ツールに任せることで
    開発サーバーの起動やHMRが劇的に速くなる
```

### 1-3. ツールの選定フローチャート

```
ビルドツール選定ガイド:

  Q1: フロントエンドアプリ？
  ├── Yes → Vite を使う
  │         ├── React → @vitejs/plugin-react-swc
  │         ├── Vue → @vitejs/plugin-vue
  │         └── Svelte → @sveltejs/vite-plugin-svelte
  └── No
      Q2: npm ライブラリ？
      ├── Yes → tsup (esbuild ベース) or unbuild
      └── No
          Q3: Node.js バックエンド？
          ├── Yes
          │   ├── 開発 → tsx (esbuild ベース)
          │   └── 本番 → esbuild でバンドル
          └── No
              Q4: モノレポ？
              ├── Yes → Turborepo + 各パッケージのツール
              └── → プロジェクトに合ったツールを選択
```

---

## 2. tsc（TypeScript Compiler）

### 2-1. 基本コマンド

```typescript
// package.json scripts
{
  "scripts": {
    "build": "tsc",
    "build:watch": "tsc --watch",
    "typecheck": "tsc --noEmit",
    "typecheck:watch": "tsc --noEmit --watch",
    "build:project": "tsc --build",
    "build:clean": "tsc --build --clean"
  }
}

// tsc の主要フラグ
// tsc                    → tsconfig.json に従ってビルド
// tsc --noEmit           → 型チェックのみ（ファイル出力なし）
// tsc --watch            → ファイル変更を監視して再ビルド
// tsc --build            → プロジェクト参照を含むインクリメンタルビルド
// tsc --declaration      → .d.ts ファイルを生成
// tsc --project tsconfig.test.json → 指定した設定ファイルを使用
// tsc --extendedDiagnostics → パフォーマンス診断情報を出力
// tsc --generateTrace ./trace → パフォーマンストレースを生成
```

### 2-2. インクリメンタルビルド

```json
// tsconfig.json
{
  "compilerOptions": {
    "incremental": true,
    "tsBuildInfoFile": "./.tsbuildinfo"
  }
}
// 前回のビルド情報を .tsbuildinfo に保存し、
// 変更のあったファイルのみ再コンパイル
// → 2回目以降のビルドが大幅に高速化
```

```
インクリメンタルビルドの効果（実測例）:

  初回:     30.2s (全ファイル)
  2回目:     4.1s (変更なし、キャッシュ検証のみ)
  3回目:     6.8s (10ファイル変更)
  クリーン後: 30.5s (キャッシュなし)

  → 2回目以降は 70-85% の高速化

  .tsbuildinfo ファイルの内容:
  - 各ファイルのハッシュ
  - 依存関係グラフ
  - コンパイラオプションのスナップショット
  - 出力ファイルのシグネチャ
```

### 2-3. tsc と他ツールの併用パターン

```json
// パターン1: tsc (型チェック) + esbuild (トランスパイル)
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "build": "esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js",
    "prebuild": "npm run typecheck"
  }
}

// パターン2: tsc (型定義生成) + esbuild (JS生成)
{
  "scripts": {
    "build:types": "tsc --emitDeclarationOnly --declaration --declarationMap",
    "build:js": "esbuild src/index.ts --bundle --outfile=dist/index.js",
    "build": "npm run build:types && npm run build:js"
  }
}

// パターン3: 並列実行（concurrently を使用）
{
  "scripts": {
    "dev": "concurrently \"tsc --noEmit --watch\" \"tsx watch src/index.ts\"",
    "build": "concurrently \"tsc --noEmit\" \"esbuild src/index.ts --bundle --outfile=dist/index.js\""
  }
}
```

### 2-4. tsc --build（プロジェクト参照ビルド）

```bash
# モノレポでのビルド
tsc --build                    # 全パッケージをビルド
tsc --build packages/shared    # shared とその依存のみ
tsc --build --watch            # ウォッチモード
tsc --build --verbose          # 詳細出力
tsc --build --dry              # ドライラン
tsc --build --clean            # ビルド成果物を削除
tsc --build --force            # キャッシュを無視して全ビルド
```

```
tsc --build の動作:

  1. 依存グラフを解析
     shared → frontend (shared に依存)
           → backend (shared に依存)
           → e2e (frontend, backend に依存)

  2. トポロジカルソートで順序決定
     shared → [frontend, backend] → e2e

  3. 各パッケージを順番にビルド
     - .tsbuildinfo でキャッシュチェック
     - 変更がなければスキップ
     - 変更があれば再ビルド

  4. 下流パッケージの再ビルド判定
     - shared が変更 → frontend, backend, e2e 全て再ビルド
     - frontend のみ変更 → frontend, e2e のみ再ビルド
```

---

## 3. esbuild

### 3-1. 基本セットアップ

```typescript
// esbuild.config.ts
import * as esbuild from "esbuild";

// シンプルなビルド
await esbuild.build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  outfile: "dist/index.js",
  platform: "node",
  target: "node20",
  format: "esm",
  sourcemap: true,
  minify: process.env.NODE_ENV === "production",
});

// 複数エントリーポイント
await esbuild.build({
  entryPoints: ["src/index.ts", "src/worker.ts"],
  bundle: true,
  outdir: "dist",
  splitting: true,  // コード分割（ESM のみ）
  format: "esm",
  platform: "node",
  target: "node20",
  external: ["pg", "redis"], // バンドルしないパッケージ
});

// ブラウザ向けビルド
await esbuild.build({
  entryPoints: ["src/app.ts"],
  bundle: true,
  outfile: "dist/app.js",
  platform: "browser",
  target: ["chrome100", "firefox100", "safari16"],
  format: "esm",
  sourcemap: true,
  minify: true,
  // CSS もバンドル
  loader: {
    ".png": "file",
    ".svg": "dataurl",
    ".css": "css",
  },
  // 環境変数の埋め込み
  define: {
    "process.env.NODE_ENV": '"production"',
    "import.meta.env.VITE_API_URL": '"https://api.example.com"',
  },
});
```

### 3-2. esbuild プラグイン

```typescript
import * as esbuild from "esbuild";
import { readFile } from "fs/promises";

// カスタムプラグインの作成
const envPlugin: esbuild.Plugin = {
  name: "env-plugin",
  setup(build) {
    // .env ファイルを読み込んで定義に変換
    build.onResolve({ filter: /^env$/ }, (args) => ({
      path: args.path,
      namespace: "env-ns",
    }));

    build.onLoad({ filter: /.*/, namespace: "env-ns" }, async () => {
      const envFile = await readFile(".env", "utf-8");
      const env: Record<string, string> = {};
      for (const line of envFile.split("\n")) {
        const [key, ...valueParts] = line.split("=");
        if (key && valueParts.length > 0) {
          env[key.trim()] = valueParts.join("=").trim();
        }
      }
      return {
        contents: JSON.stringify(env),
        loader: "json",
      };
    });
  },
};

// リビルド通知プラグイン
const notifyPlugin: esbuild.Plugin = {
  name: "notify-plugin",
  setup(build) {
    let start: number;

    build.onStart(() => {
      start = Date.now();
      console.log("Build started...");
    });

    build.onEnd((result) => {
      const elapsed = Date.now() - start;
      if (result.errors.length > 0) {
        console.error(`Build failed in ${elapsed}ms with ${result.errors.length} errors`);
      } else {
        console.log(`Build completed in ${elapsed}ms`);
        if (result.warnings.length > 0) {
          console.warn(`  ${result.warnings.length} warnings`);
        }
      }
    });
  },
};

// Node.js の外部パッケージを自動検出するプラグイン
const nodeExternalsPlugin: esbuild.Plugin = {
  name: "node-externals",
  setup(build) {
    // node_modules のパッケージを全て external にする
    build.onResolve({ filter: /^[^./]/ }, (args) => ({
      path: args.path,
      external: true,
    }));
  },
};

// プラグインの使用
await esbuild.build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  outfile: "dist/index.js",
  platform: "node",
  format: "esm",
  plugins: [envPlugin, notifyPlugin, nodeExternalsPlugin],
});
```

### 3-3. 開発サーバー

```typescript
// esbuild の watch + serve
const ctx = await esbuild.context({
  entryPoints: ["src/index.ts"],
  bundle: true,
  outdir: "dist",
  platform: "node",
  format: "esm",
  sourcemap: true,
  plugins: [
    {
      name: "rebuild-notify",
      setup(build) {
        build.onEnd((result) => {
          console.log(
            `Build finished: ${result.errors.length} errors`
          );
        });
      },
    },
  ],
});

await ctx.watch(); // ファイル変更を監視して自動リビルド

// フロントエンド用の開発サーバー
const serveCtx = await esbuild.context({
  entryPoints: ["src/app.ts"],
  bundle: true,
  outdir: "public/dist",
  platform: "browser",
  format: "esm",
  sourcemap: true,
});

// 開発サーバーを起動
const { host, port } = await serveCtx.serve({
  servedir: "public",
  port: 3000,
});
console.log(`Dev server running at http://${host}:${port}`);
```

### 3-4. esbuild の制限事項

```
esbuild がサポートしない TypeScript 機能:

  1. 型チェック
     → tsc --noEmit を別途実行

  2. const enum（cross-file inlining）
     → isolatedModules: true で回避
     → 通常の enum として扱われる

  3. デコレータ（experimentalDecorators）
     → ECMAScript 標準デコレータ (Stage 3) はサポート
     → 旧式の TypeScript デコレータは --loader=ts で部分サポート

  4. 宣言ファイル（.d.ts）生成
     → tsc --emitDeclarationOnly を併用

  5. 一部のtsconfig オプション
     → emitDecoratorMetadata: 非サポート
     → paths: 限定的サポート（プラグインで対応可能）

  対策: esbuild は JS 生成のみに使い、
        型関連は全て tsc に任せる
```

### 3-5. package.json 設定

```json
{
  "scripts": {
    "build": "esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js --format=esm",
    "build:prod": "esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js --format=esm --minify --sourcemap",
    "dev": "esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js --format=esm --watch",
    "typecheck": "tsc --noEmit"
  }
}
```

---

## 4. SWC

### 4-1. 基本セットアップ

```json
// .swcrc
{
  "$schema": "https://swc.rs/schema.json",
  "jsc": {
    "parser": {
      "syntax": "typescript",
      "tsx": false,
      "decorators": true,
      "dynamicImport": true
    },
    "target": "es2022",
    "transform": {
      "decoratorVersion": "2022-03"
    },
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"]
    }
  },
  "module": {
    "type": "es6",
    "strict": true,
    "noInterop": false
  },
  "sourceMaps": true,
  "minify": false
}
```

```json
// package.json
{
  "scripts": {
    "build": "swc src -d dist --strip-leading-paths",
    "build:watch": "swc src -d dist --strip-leading-paths --watch",
    "dev": "swc src -d dist --strip-leading-paths --watch",
    "typecheck": "tsc --noEmit"
  }
}
```

### 4-2. SWC + Node.js 実行

```json
// @swc-node/register で直接 .ts を実行
{
  "scripts": {
    "start": "node --import @swc-node/register/esm src/index.ts",
    "dev": "node --import @swc-node/register/esm --watch src/index.ts"
  }
}
```

### 4-3. SWC の minification

```json
// .swcrc に minify 設定を追加
{
  "jsc": {
    "parser": { "syntax": "typescript" },
    "target": "es2022",
    "minify": {
      "compress": {
        "dead_code": true,
        "drop_console": true,
        "drop_debugger": true,
        "passes": 2,
        "unused": true
      },
      "mangle": {
        "toplevel": true,
        "keep_classnames": false,
        "keep_fnames": false
      }
    }
  },
  "minify": true
}
```

### 4-4. SWC vs esbuild の詳細比較

```
SWC と esbuild の違い:

  SWC (Rust 製):
  ├── トランスパイルのみ（バンドルは swcpack で実験的）
  ├── decorators の完全サポート（emitDecoratorMetadata 含む）
  ├── Next.js / Parcel に組み込まれている
  ├── プラグインシステム（Wasm / Rust）
  └── minification サポート

  esbuild (Go 製):
  ├── トランスパイル + バンドル
  ├── HTTP サーバー内蔵
  ├── Tree-shaking サポート
  ├── コード分割サポート
  └── プラグインシステム（JavaScript）

  選択基準:
  - バンドルが必要 → esbuild
  - NestJS / Angular（旧式デコレータ） → SWC
  - Next.js → SWC（組込み）
  - 汎用トランスパイル → どちらでも
```

---

## 5. Vite

### 5-1. 基本設定

```
Vite の開発/本番フロー:

  開発 (dev):
  +----------+     +---------+     +----------+
  | .ts ファイル| --> | esbuild | --> | ブラウザ  |
  | (ソース)   |     | (変換)  |     | (ESM)    |
  +----------+     +---------+     +----------+
       ↑ HMR（ミリ秒単位で更新）

  本番 (build):
  +----------+     +---------+     +----------+
  | .ts ファイル| --> | Rollup  | --> | バンドル  |
  | (ソース)   |     | + SWC   |     | (最適化) |
  +----------+     +---------+     +----------+

  Vite 6.x 以降:
  +----------+     +---------+     +----------+
  | .ts ファイル| --> | Rolldown| --> | バンドル  |
  | (ソース)   |     | (Rust)  |     | (高速)   |
  +----------+     +---------+     +----------+
```

```typescript
// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc"; // SWC 版 React プラグイン

export default defineConfig({
  plugins: [react()],
  build: {
    target: "es2022",
    outDir: "dist",
    sourcemap: true,
    // Rollup のオプション
    rollupOptions: {
      output: {
        // 手動チャンク分割
        manualChunks: {
          vendor: ["react", "react-dom"],
          ui: ["@radix-ui/react-dialog", "@radix-ui/react-dropdown-menu"],
        },
        // チャンクファイル名のフォーマット
        chunkFileNames: "assets/[name]-[hash].js",
        entryFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash].[ext]",
      },
    },
    // チャンクサイズ警告の閾値
    chunkSizeWarningLimit: 500,
    // CSS のコード分割
    cssCodeSplit: true,
  },
  resolve: {
    alias: {
      "@": "/src",
      "@components": "/src/components",
      "@hooks": "/src/hooks",
      "@utils": "/src/utils",
    },
  },
  server: {
    port: 3000,
    strictPort: true,
    // プロキシ設定
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  // 環境変数のプレフィックス
  envPrefix: "VITE_",
});
```

### 5-2. Vite プラグイン

```typescript
// よく使われる Vite プラグイン
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import tsconfigPaths from "vite-tsconfig-paths";
import checker from "vite-plugin-checker";
import { compression } from "vite-plugin-compression2";
import { visualizer } from "rollup-plugin-visualizer";

export default defineConfig({
  plugins: [
    // React (SWC ベース、高速)
    react(),

    // tsconfig.json の paths を自動解決
    tsconfigPaths(),

    // 開発中にリアルタイム型チェック + ESLint
    checker({
      typescript: true,
      eslint: {
        lintCommand: 'eslint "./src/**/*.{ts,tsx}"',
      },
    }),

    // Gzip / Brotli 圧縮
    compression({
      algorithm: "gzip",
      threshold: 1024,
    }),
    compression({
      algorithm: "brotliCompress",
      threshold: 1024,
    }),

    // バンドルサイズの可視化
    visualizer({
      filename: "dist/stats.html",
      open: false,
      gzipSize: true,
      brotliSize: true,
    }),
  ],
});
```

### 5-3. ライブラリモード

```typescript
// vite.config.ts -- ライブラリとしてビルド
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  plugins: [
    dts({
      rollupTypes: true,           // 型定義を1ファイルにバンドル
      insertTypesEntry: true,      // package.json の types を自動設定
      tsconfigPath: "./tsconfig.build.json",
    }),
  ],
  build: {
    lib: {
      entry: "src/index.ts",
      name: "MyLib",
      formats: ["es", "cjs"],
      fileName: (format) => `index.${format === "es" ? "mjs" : "cjs"}`,
    },
    rollupOptions: {
      // ピア依存はバンドルしない
      external: ["react", "react-dom", "react/jsx-runtime"],
      output: {
        globals: {
          react: "React",
          "react-dom": "ReactDOM",
        },
      },
    },
    // ソースマップ生成
    sourcemap: true,
    // minify しない（ライブラリの場合）
    minify: false,
  },
});
```

### 5-4. SSR / バックエンド

```typescript
// vite.config.ts -- Node.js バックエンド
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    ssr: true,
    target: "node20",
    outDir: "dist",
    rollupOptions: {
      input: "src/server.ts",
      output: {
        format: "esm",
        entryFileNames: "server.js",
      },
    },
  },
  ssr: {
    noExternal: true, // 全ての依存をバンドル
    // noExternal: ["specific-pkg"], // 特定のみバンドル
  },
});

// Vite + Express の SSR 設定例
// vite.config.ts
export default defineConfig({
  build: {
    ssr: true,
    rollupOptions: {
      input: {
        server: "src/server.ts",
        entry: "src/entry-server.tsx",
      },
    },
  },
});
```

### 5-5. 環境変数の管理

```typescript
// .env ファイルの読み込み順序:
// .env                # 常に読み込み
// .env.local          # 常に読み込み（.gitignore 推奨）
// .env.[mode]         # 指定モードで読み込み
// .env.[mode].local   # 指定モードで読み込み（.gitignore 推奨）

// .env
VITE_API_URL=https://api.example.com
VITE_APP_TITLE=My App

// .env.development
VITE_API_URL=http://localhost:8080
VITE_DEBUG=true

// .env.production
VITE_API_URL=https://api.production.com
VITE_DEBUG=false

// src/vite-env.d.ts -- 型定義
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string;
  readonly VITE_APP_TITLE: string;
  readonly VITE_DEBUG: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// 使用例
const apiUrl = import.meta.env.VITE_API_URL; // 型: string
const isDev = import.meta.env.DEV;            // 型: boolean
const isProd = import.meta.env.PROD;          // 型: boolean
const mode = import.meta.env.MODE;            // 型: string
```

---

## 6. tsx -- TypeScript Execute

### 6-1. 基本的な使い方

```bash
# インストール
npm install -D tsx

# TypeScript ファイルを直接実行
npx tsx src/index.ts

# ウォッチモード
npx tsx watch src/index.ts

# ESM として実行
npx tsx --esm src/index.ts

# REPL
npx tsx
```

```json
// package.json
{
  "scripts": {
    "start": "tsx src/index.ts",
    "dev": "tsx watch src/index.ts",
    "script": "tsx scripts/seed.ts"
  }
}
```

### 6-2. tsx vs ts-node vs node --loader

```
TypeScript 実行ランナー比較:

  tsx (esbuild ベース):
  ├── 起動時間: 非常に速い
  ├── 型チェック: なし
  ├── ESM サポート: あり
  ├── tsconfig paths: 自動解決
  └── 設定: ほぼ不要

  ts-node (tsc ベース):
  ├── 起動時間: 遅い
  ├── 型チェック: あり（swc モード時はなし）
  ├── ESM サポート: 要設定
  ├── tsconfig paths: tsconfig-paths 必要
  └── 設定: 多い

  node --experimental-strip-types (Node.js 23+):
  ├── 起動時間: 最速
  ├── 型チェック: なし
  ├── ESM サポート: あり
  ├── tsconfig paths: 非サポート
  └── 設定: 不要
  └── 注意: enum, namespace 等は非サポート

  node --import @swc-node/register/esm:
  ├── 起動時間: 速い
  ├── 型チェック: なし
  ├── ESM サポート: あり
  ├── tsconfig paths: 要設定
  └── 設定: 少ない
```

---

## 7. tsup / unbuild -- ライブラリビルダー

### 7-1. tsup の設定

```typescript
// tsup.config.ts
import { defineConfig } from "tsup";

export default defineConfig({
  // エントリーポイント
  entry: ["src/index.ts", "src/utils/index.ts"],

  // 出力形式（ESM + CJS）
  format: ["esm", "cjs"],

  // 型定義ファイル生成
  dts: true,

  // ソースマップ
  sourcemap: true,

  // クリーンビルド
  clean: true,

  // 外部パッケージ（バンドルしない）
  external: ["react", "react-dom"],

  // Tree-shaking
  treeshake: true,

  // TypeScript のターゲット
  target: "es2020",

  // 出力ディレクトリ
  outDir: "dist",

  // コード分割
  splitting: true,

  // minify
  minify: false,
});
```

```json
// package.json（デュアル ESM/CJS パッケージ）
{
  "name": "my-lib",
  "version": "1.0.0",
  "type": "module",
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
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
    "./utils": {
      "import": {
        "types": "./dist/utils/index.d.ts",
        "default": "./dist/utils/index.js"
      },
      "require": {
        "types": "./dist/utils/index.d.cts",
        "default": "./dist/utils/index.cjs"
      }
    }
  },
  "files": ["dist"],
  "scripts": {
    "build": "tsup",
    "dev": "tsup --watch",
    "typecheck": "tsc --noEmit",
    "prepublishOnly": "npm run build"
  }
}
```

### 7-2. unbuild の設定

```typescript
// build.config.ts
import { defineBuildConfig } from "unbuild";

export default defineBuildConfig({
  entries: ["src/index"],
  declaration: true,
  clean: true,
  rollup: {
    emitCJS: true,
    inlineDependencies: true,
  },
  // 自動 externals 検出
  externals: ["react"],
});
```

### 7-3. tsup vs unbuild の比較

```
ライブラリビルダー比較:

  tsup:
  ├── エンジン: esbuild
  ├── 速度: 非常に速い
  ├── 設定: シンプル
  ├── DTS: esbuild + tsc (ハイブリッド)
  ├── Tree-shaking: esbuild
  └── 推奨: 小〜中規模ライブラリ

  unbuild:
  ├── エンジン: Rollup
  ├── 速度: 速い
  ├── 設定: シンプル
  ├── DTS: rollup-plugin-dts
  ├── Tree-shaking: Rollup（高品質）
  └── 推奨: Nuxt エコシステム

  Vite lib mode:
  ├── エンジン: Rollup
  ├── 速度: 速い
  ├── 設定: Vite の知識が必要
  ├── DTS: vite-plugin-dts
  ├── Tree-shaking: Rollup（高品質）
  └── 推奨: Vite プロジェクトのライブラリ

  tsc:
  ├── エンジン: TypeScript Compiler
  ├── 速度: 遅い
  ├── 設定: tsconfig.json
  ├── DTS: ネイティブ（最も正確）
  ├── Tree-shaking: なし
  └── 推奨: 型定義の正確さが最重要の場合
```

---

## 8. モノレポでのビルド戦略

### 8-1. Turborepo

```json
// turbo.json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**"],
      "inputs": ["src/**", "tsconfig.json"]
    },
    "typecheck": {
      "dependsOn": ["^build"]
    },
    "lint": {},
    "test": {
      "dependsOn": ["build"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    }
  }
}
```

```bash
# Turborepo でのビルド
turbo build           # 全パッケージをビルド（キャッシュ活用）
turbo build --filter=web  # web パッケージのみ
turbo build --force   # キャッシュを無視
turbo dev             # 全パッケージの開発サーバー

# リモートキャッシュ（チーム間でキャッシュ共有）
turbo login
turbo link
turbo build  # リモートキャッシュを使用
```

### 8-2. モノレポのパッケージ構成例

```
monorepo/
├── apps/
│   ├── web/               (Next.js)
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── api/               (Node.js + esbuild)
│       ├── package.json
│       └── tsconfig.json
├── packages/
│   ├── shared/            (tsup でビルド)
│   │   ├── package.json
│   │   ├── tsup.config.ts
│   │   └── tsconfig.json
│   ├── ui/                (Vite lib mode)
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── tsconfig.json
│   └── config-ts/         (共有 tsconfig)
│       ├── base.json
│       ├── nextjs.json
│       ├── node.json
│       └── library.json
├── turbo.json
├── package.json
└── tsconfig.json
```

---

## 9. 最適なパイプライン設計

### 9-1. フロントエンド（React / Vue）

```
推奨パイプライン:

  開発:   Vite dev server (esbuild でトランスパイル)
  型チェック: tsc --noEmit (バックグラウンド or CI)
  本番ビルド: Vite build (Rollup + minify)
  テスト:  Vitest (Vite と設定共有)
  lint:   ESLint + Prettier

  package.json:
  {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "lint": "eslint src/",
    "typecheck": "tsc --noEmit"
  }
```

### 9-2. Node.js バックエンド

```
推奨パイプライン:

  開発:   tsx (esbuild ベースの ts-node 代替)
  型チェック: tsc --noEmit
  本番ビルド: esbuild (バンドル + minify)
  テスト:  Vitest
  lint:   ESLint

  package.json:
  {
    "dev": "tsx watch src/index.ts",
    "build": "tsc --noEmit && esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js --format=esm --minify",
    "start": "node dist/index.js",
    "test": "vitest",
    "typecheck": "tsc --noEmit"
  }
```

### 9-3. npm ライブラリ

```
推奨パイプライン:

  開発:   tsx で実行テスト
  型チェック: tsc --noEmit
  ビルド: tsup (esbuild ベース、ESM + CJS 両出力)
  型定義: tsup --dts (tsc を内部使用)
  テスト:  Vitest
  公開:   npm publish (prepublishOnly で build)

  package.json:
  {
    "build": "tsup",
    "dev": "tsup --watch",
    "typecheck": "tsc --noEmit",
    "test": "vitest",
    "prepublishOnly": "npm run build && npm run typecheck"
  }
```

### 9-4. フルスタック（Next.js + tRPC）

```
推奨パイプライン:

  フレームワーク: Next.js (SWC 組込み)
  API:    tRPC (型共有)
  DB:     Prisma (型生成)
  バリデーション: zod
  テスト:  Vitest + Playwright

  package.json:
  {
    "dev": "next dev --turbo",
    "build": "next build",
    "start": "next start",
    "test": "vitest",
    "test:e2e": "playwright test",
    "db:generate": "prisma generate",
    "db:migrate": "prisma migrate dev",
    "typecheck": "tsc --noEmit",
    "postinstall": "prisma generate"
  }
```

---

## 比較表

### ビルドツール総合比較

| 特性 | tsc | esbuild | SWC | Vite | tsup |
|------|-----|---------|-----|------|------|
| 言語 | TypeScript | Go | Rust | JS (esbuild/Rollup) | JS (esbuild) |
| 型チェック | あり | なし | なし | なし | なし |
| トランスパイル速度 | 遅い | 最速級 | 最速級 | 速い (esbuild) | 速い (esbuild) |
| バンドル | なし | あり | 実験的 | あり (Rollup) | あり |
| Tree-shaking | なし | あり | なし | あり | あり |
| HMR | なし | 簡易 | なし | 優秀 | なし |
| プラグイン | なし | あり | あり | 豊富 | esbuild互換 |
| 設定の簡単さ | 中 | 高 | 中 | 高 | 最高 |
| .d.ts 生成 | ネイティブ | なし | なし | プラグイン | あり |
| CSS バンドル | なし | あり | なし | あり | なし |
| コード分割 | なし | あり(ESM) | なし | あり | あり |

### 用途別推奨ツール

| 用途 | 推奨 | 理由 |
|------|------|------|
| React / Vue SPA | Vite | HMR, プラグイン充実 |
| Next.js | (組込みSWC) | フレームワーク統合 |
| Node.js API | esbuild / tsx | 高速、シンプル |
| npm ライブラリ | tsup (esbuild) | ESM/CJS 両出力、DTS生成 |
| モノレポ | Turborepo + Vite/esbuild | キャッシュ, 並列ビルド |
| Deno | (組込み) | 設定不要 |
| Bun | (組込み) | 設定不要、最速 |
| Cloudflare Workers | wrangler (esbuild) | Edge 最適化 |
| Electron | Vite + electron-builder | HMR + パッケージング |

### パフォーマンス指標（実測参考値）

| ツール | 1000ファイル | 5000ファイル | 10000ファイル |
|--------|------------|------------|-------------|
| tsc (初回) | 3s | 12s | 30s |
| tsc (インクリメンタル) | 0.5s | 2s | 5s |
| esbuild (バンドル) | 0.1s | 0.3s | 0.5s |
| SWC (トランスパイル) | 0.08s | 0.25s | 0.4s |
| Vite (dev起動) | 0.5s | 1.5s | 3s |
| tsup | 0.3s | 0.8s | 1.5s |

---

## アンチパターン

### AP-1: tsc でトランスパイルとバンドルを兼ねる

```json
// NG: tsc だけで全てをやろうとする
{
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js"
  }
}
// 問題:
// - バンドルされない（ファイルが分散）
// - Tree-shaking なし
// - パス解決が壊れることがある（paths）
// - ビルドが遅い
// - node_modules から直接 import が必要

// OK: 型チェックとビルドを分離
{
  "scripts": {
    "typecheck": "tsc --noEmit",
    "build": "esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js",
    "prebuild": "npm run typecheck"
  }
}
```

### AP-2: 開発サーバーなしで手動リロード

```json
// NG: 毎回手動でビルド→実行
{
  "scripts": {
    "dev": "tsc && node dist/index.js"
  }
}

// OK: ファイル監視で自動再起動
{
  "scripts": {
    "dev": "tsx watch src/index.ts"
  }
}
// もしくは
{
  "scripts": {
    "dev": "node --import @swc-node/register/esm --watch src/index.ts"
  }
}
```

### AP-3: 型チェックをビルドに含めて CI を遅くする

```json
// NG: ビルドと型チェックを直列実行
{
  "scripts": {
    "build": "tsc --noEmit && esbuild src/index.ts --bundle --outfile=dist/index.js"
  }
}

// OK: CI で並列実行
// .github/workflows/ci.yml
// jobs:
//   typecheck:
//     run: npm run typecheck
//   build:
//     run: npm run build:js
//   lint:
//     run: npm run lint
//   test:
//     run: npm test
```

### AP-4: バンドルサイズを確認せずにデプロイ

```typescript
// NG: バンドル分析なしでデプロイ
// → 不要な依存が含まれ、パフォーマンス悪化

// OK: バンドル分析を定期的に実施
// vite.config.ts
import { visualizer } from "rollup-plugin-visualizer";

export default defineConfig({
  plugins: [
    visualizer({
      filename: "dist/stats.html",
      gzipSize: true,
      brotliSize: true,
    }),
  ],
});
// npm run build 後に dist/stats.html を確認
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## FAQ

### Q1: tsx と ts-node の違いは何ですか？

tsx は esbuild ベースで起動が非常に高速です。ts-node は tsc ベースで型チェックも行えますが遅いです。開発時の実行は tsx を推奨します。型チェックは別途 `tsc --noEmit` で行ってください。ts-node にも `--swc` フラグがありますが、tsx の方がセットアップが簡単で高速です。

### Q2: esbuild は enum をサポートしていますか？

esbuild は `const enum` を通常の `enum` として扱います。`isolatedModules: true` を設定していれば問題ありません（`const enum` の cross-file inlining が無効化されるため）。`verbatimModuleSyntax` を使う場合も同様です。通常の `enum` は問題なくサポートされています。

### Q3: Vite の本番ビルドが開発時と挙動が異なることはありますか？

はい。開発時は esbuild でトランスパイルし ESM をそのまま配信しますが、本番は Rollup でバンドルします。稀に挙動の差が出る場合があります。`vite preview` で本番ビルドをローカルで確認することを推奨します。具体的な差異としては、CSS の読み込み順序、動的インポートの分割粒度、環境変数の解決タイミングなどがあります。

### Q4: Node.js 23+ の --experimental-strip-types は tsx の代替になりますか？

部分的には代替になります。Node.js のネイティブ TypeScript サポートは型注釈を単純に除去するだけで、enum、namespace、decorators などの TypeScript 固有の構文はサポートしません。また、tsconfig.json の paths も解決しません。シンプルな TypeScript ファイルの実行には使えますが、複雑なプロジェクトでは tsx が引き続き推奨されます。

### Q5: ビルドツールの移行は困難ですか？

多くの場合、トランスパイラの移行は比較的容易です。esbuild → SWC、webpack → Vite などの移行は設定ファイルの書き換えが主な作業です。ただし、カスタムプラグインや特殊な設定に依存している場合は移行コストが高くなります。まず新ツールで小規模なプロジェクトを試してから、段階的に移行することを推奨します。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| 分離原則 | 型チェック（tsc）とトランスパイルは別ツールに |
| esbuild | Go 製、最速、バンドル可能、型チェックなし |
| SWC | Rust 製、最速、Next.js 組込み、デコレータ完全サポート |
| Vite | 開発 = esbuild、本番 = Rollup、HMR 優秀 |
| tsup | esbuild ベースのライブラリビルダー、DTS 生成 |
| tsx | esbuild ベースの ts-node 代替、高速、設定不要 |
| Turborepo | モノレポのビルドキャッシュ、並列実行 |
| unbuild | Rollup ベースのライブラリビルダー |

---

## 10. Docker でのビルド最適化

### 10-1. マルチステージビルド

```dockerfile
# ---- Builder ステージ ----
FROM node:20-slim AS builder

WORKDIR /app

# 依存のインストール（キャッシュ効率化）
COPY package.json package-lock.json ./
RUN npm ci

# ソースコピー & ビルド
COPY tsconfig.json ./
COPY src/ ./src/
RUN npm run build

# ---- Production ステージ ----
FROM node:20-slim AS production

WORKDIR /app

# 本番依存のみインストール
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# ビルド成果物のみコピー
COPY --from=builder /app/dist ./dist

# Node.js で直接実行
CMD ["node", "dist/index.js"]
```

### 10-2. esbuild でバンドルした場合

```dockerfile
# esbuild でバンドルすると node_modules が不要になる
FROM node:20-slim AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig.json ./
COPY src/ ./src/
RUN npx esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js --format=esm --minify

# ---- 軽量な Production ステージ ----
FROM node:20-slim AS production
WORKDIR /app

# バンドル済みファイルのみコピー（node_modules 不要）
COPY --from=builder /app/dist/index.js ./index.js

CMD ["node", "index.js"]
# → イメージサイズが大幅に削減される
```

### 10-3. .dockerignore の設定

```
# .dockerignore
node_modules
dist
.git
.github
*.md
.env*
.tsbuildinfo
coverage
.vscode
.idea
```

---

## 11. ビルドパフォーマンスのデバッグ

### 11-1. tsc のパフォーマンス分析

```bash
# 詳細な診断情報
tsc --extendedDiagnostics --noEmit

# 出力例:
# Files:               1,234
# Lines of Library:    35,678
# Lines of Definitions: 89,012
# Lines of TypeScript:  67,890
# Nodes:               345,678
# Identifiers:         123,456
# Symbols:              67,890
# Types:                34,567
# Instantiations:      234,567  ← これが大きいと遅い
# Memory used:         456,789K
# Assignability cache size: 12,345
# Identity cache size:  1,234
# Subtype cache size:   2,345
# Strict subtype cache: 3,456
# I/O Read time:        0.12s
# Parse time:           1.23s
# ResolveModule time:   0.34s
# ResolveTypeRef time:  0.05s
# Bind time:            0.45s
# Check time:           5.67s   ← 通常最大
# printTime time:       0.89s
# Emit time:            0.89s
# Total time:           8.36s

# Instantiations が大きい場合の対処:
# 1. 複雑なジェネリクス型を簡素化
# 2. 条件型のネストを減らす
# 3. 型の再計算を避ける（type alias でキャッシュ）
```

### 11-2. 型のパフォーマンスを改善するテクニック

```typescript
// NG: 深くネストされた条件型（Instantiations が爆発）
type DeepPick<T, K extends string> =
  K extends `${infer First}.${infer Rest}`
    ? First extends keyof T
      ? { [P in First]: DeepPick<T[First], Rest> }
      : never
    : K extends keyof T
      ? { [P in K]: T[P] }
      : never;

// OK: interface で中間型を定義してキャッシュ
interface CachedDeepPick<T, First extends keyof T, Rest extends string> {
  [P in First]: DeepPick<T[First], Rest>;
}

// NG: 大量のユニオン型（チェックが O(n^2)）
type AllEvents = Event1 | Event2 | ... | Event100;

// OK: 判別共用体でマップ型を使用
interface EventMap {
  event1: Event1;
  event2: Event2;
  // ...
}
type AllEvents = EventMap[keyof EventMap];
```

---


## まとめ

このガイドでは以下の重要なポイントを学びました:

- 基本概念と原則の理解
- 実践的な実装パターン
- ベストプラクティスと注意点
- 実務での活用方法

---

## 次に読むべきガイド

- [tsconfig.json](./00-tsconfig.md) -- ビルドツールと連携する TypeScript 設定
- [テスト](./02-testing-typescript.md) -- Vitest の設定とビルドツールの連携
- [ESLint + TypeScript](./04-eslint-typescript.md) -- ビルドパイプラインへの lint 統合

---

## 参考文献

1. **esbuild** -- An extremely fast bundler for the web
   https://esbuild.github.io/

2. **SWC** -- Rust-based platform for the Web
   https://swc.rs/

3. **Vite** -- Next Generation Frontend Tooling
   https://vitejs.dev/

4. **tsx** -- TypeScript Execute
   https://tsx.is/

5. **tsup** -- Bundle your TypeScript library with no config
   https://tsup.egoist.dev/

6. **Turborepo** -- High-performance build system for JavaScript and TypeScript codebases
   https://turbo.build/repo

7. **unbuild** -- A unified JavaScript build system
   https://github.com/unjs/unbuild
