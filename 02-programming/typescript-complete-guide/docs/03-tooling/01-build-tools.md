# TypeScript ビルドツール完全ガイド

> tsc, esbuild, SWC, Vite を比較し、プロジェクトに最適なビルドパイプラインを構築する

## この章で学ぶこと

1. **各ビルドツールの特性** -- tsc, esbuild, SWC, Vite それぞれの設計思想、速度、機能の違い
2. **ビルドパイプラインの設計** -- 型チェックとトランスパイルの分離、開発/本番環境の構成パターン
3. **移行とチューニング** -- 既存プロジェクトのビルド高速化と、ツール間の移行手順

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
    "typecheck:watch": "tsc --noEmit --watch"
  }
}

// tsc の主要フラグ
// tsc                    → tsconfig.json に従ってビルド
// tsc --noEmit           → 型チェックのみ（ファイル出力なし）
// tsc --watch            → ファイル変更を監視して再ビルド
// tsc --build            → プロジェクト参照を含むインクリメンタルビルド
// tsc --declaration      → .d.ts ファイルを生成
// tsc --project tsconfig.test.json → 指定した設定ファイルを使用
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
```

### 3-2. 開発サーバー

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
```

### 3-3. package.json 設定

```json
{
  "scripts": {
    "build": "esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js --format=esm",
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
    }
  },
  "module": {
    "type": "es6"
  },
  "sourceMaps": true
}
```

```json
// package.json
{
  "scripts": {
    "build": "swc src -d dist --strip-leading-paths",
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
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ["react", "react-dom"],
        },
      },
    },
  },
  resolve: {
    alias: {
      "@": "/src",
    },
  },
  server: {
    port: 3000,
    strictPort: true,
  },
});
```

### 5-2. ライブラリモード

```typescript
// vite.config.ts -- ライブラリとしてビルド
import { defineConfig } from "vite";
import dts from "vite-plugin-dts";

export default defineConfig({
  plugins: [dts({ rollupTypes: true })],
  build: {
    lib: {
      entry: "src/index.ts",
      name: "MyLib",
      formats: ["es", "cjs"],
      fileName: (format) => `index.${format === "es" ? "mjs" : "cjs"}`,
    },
    rollupOptions: {
      external: ["react", "react-dom"],
    },
  },
});
```

### 5-3. SSR / バックエンド

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
    },
  },
  ssr: {
    noExternal: true, // 全ての依存をバンドル
    // noExternal: ["specific-pkg"], // 特定のみバンドル
  },
});
```

---

## 6. 最適なパイプライン設計

### 6-1. フロントエンド（React / Vue）

```
推奨パイプライン:

  開発:   Vite dev server (esbuild でトランスパイル)
  型チェック: tsc --noEmit (バックグラウンド or CI)
  本番ビルド: Vite build (Rollup + minify)
  テスト:  Vitest (Vite と設定共有)

  package.json:
  {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "test": "vitest"
  }
```

### 6-2. Node.js バックエンド

```
推奨パイプライン:

  開発:   tsx (esbuild ベースの ts-node 代替)
  型チェック: tsc --noEmit
  本番ビルド: esbuild (バンドル + minify)
  テスト:  Vitest

  package.json:
  {
    "dev": "tsx watch src/index.ts",
    "build": "tsc --noEmit && esbuild src/index.ts --bundle --platform=node --outfile=dist/index.js",
    "start": "node dist/index.js",
    "test": "vitest"
  }
```

### 6-3. npm ライブラリ

```
推奨パイプライン:

  開発:   tsx で実行テスト
  型チェック: tsc --noEmit
  ビルド: tsup (esbuild ベース、ESM + CJS 両出力)
  型定義: tsc --declaration --emitDeclarationOnly

  package.json:
  {
    "build": "tsup src/index.ts --format esm,cjs --dts",
    "typecheck": "tsc --noEmit",
    "test": "vitest"
  }
```

---

## 比較表

### ビルドツール総合比較

| 特性 | tsc | esbuild | SWC | Vite |
|------|-----|---------|-----|------|
| 言語 | TypeScript | Go | Rust | JS (esbuild/Rollup) |
| 型チェック | あり | なし | なし | なし |
| トランスパイル速度 | 遅い | 最速級 | 最速級 | 速い (esbuild) |
| バンドル | なし | あり | なし | あり (Rollup) |
| Tree-shaking | なし | あり | なし | あり |
| HMR | なし | 簡易 | なし | 優秀 |
| プラグイン | なし | あり | あり | 豊富 |
| 設定の簡単さ | 中 | 高 | 中 | 高 |

### 用途別推奨ツール

| 用途 | 推奨 | 理由 |
|------|------|------|
| React / Vue SPA | Vite | HMR, プラグイン充実 |
| Next.js | (組込みSWC) | フレームワーク統合 |
| Node.js API | esbuild / tsx | 高速、シンプル |
| npm ライブラリ | tsup (esbuild) | ESM/CJS 両出力 |
| モノレポ | Turborepo + Vite/esbuild | キャッシュ, 並列ビルド |
| Deno | (組込み) | 設定不要 |

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

---

## FAQ

### Q1: tsx と ts-node の違いは何ですか？

tsx は esbuild ベースで起動が非常に高速です。ts-node は tsc ベースで型チェックも行えますが遅いです。開発時の実行は tsx を推奨します。型チェックは別途 `tsc --noEmit` で行ってください。

### Q2: esbuild は enum をサポートしていますか？

esbuild は `const enum` を通常の `enum` として扱います。`isolatedModules: true` を設定していれば問題ありません（`const enum` の cross-file inlining が無効化されるため）。`verbatimModuleSyntax` を使う場合も同様です。

### Q3: Vite の本番ビルドが開発時と挙動が異なることはありますか？

はい。開発時は esbuild でトランスパイルし ESM をそのまま配信しますが、本番は Rollup でバンドルします。稀に挙動の差が出る場合があります。`vite preview` で本番ビルドをローカルで確認することを推奨します。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| 分離原則 | 型チェック（tsc）とトランスパイルは別ツールに |
| esbuild | Go 製、最速、バンドル可能、型チェックなし |
| SWC | Rust 製、最速、Next.js 組込み |
| Vite | 開発 = esbuild、本番 = Rollup、HMR 優秀 |
| tsup | esbuild ベースのライブラリビルダー |
| tsx | esbuild ベースの ts-node 代替、高速 |

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
