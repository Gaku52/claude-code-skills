# ESLint + TypeScript 完全ガイド

> typescript-eslint を使い、型情報を活用した高度な静的解析で TypeScript コードの品質を保つ

## この章で学ぶこと

1. **typescript-eslint のセットアップ** -- Flat Config 形式での設定、パーサー連携、推奨ルールセット
2. **型情報を使うルール** -- `@typescript-eslint` の型チェック付きルールで、tsc では検出できない問題を発見する
3. **プロジェクト別の設定** -- モノレポ、React、Node.js、ライブラリそれぞれの最適な lint 構成
4. **パフォーマンス最適化** -- TIMING、キャッシュ、並列実行による高速化
5. **カスタムルール作成** -- AST 操作と型情報アクセスによる独自ルールの実装
6. **代替ツールとの比較** -- Biome、oxlint との機能・性能比較と移行検討

---

## 目次

1. [ESLint + TypeScript のアーキテクチャ](#1-eslint--typescript-のアーキテクチャ)
2. [セットアップと基本設定](#2-セットアップと基本設定)
3. [推奨ルールセットの比較と選択](#3-推奨ルールセットの比較と選択)
4. [型情報を使うルールの深堀り](#4-型情報を使うルールの深堀り)
5. [カスタムルールの作成](#5-カスタムルールの作成)
6. [Prettier との統合と競合解消](#6-prettier-との統合と競合解消)
7. [プロジェクト別設定パターン](#7-プロジェクト別設定パターン)
8. [モノレポでの設定共有](#8-モノレポでの設定共有)
9. [CI/CD パイプライン統合](#9-cicd-パイプライン統合)
10. [パフォーマンスチューニング](#10-パフォーマンスチューニング)
11. [Biome/oxlint との比較と移行](#11-biomeoxlint-との比較と移行)
12. [演習問題](#12-演習問題)
13. [FAQ](#13-faq)
14. [参考文献](#14-参考文献)

---

## 1. ESLint + TypeScript のアーキテクチャ

### 1-1. 全体構造

ESLint は元々 JavaScript 専用のリンターですが、typescript-eslint プロジェクトによって TypeScript のサポートが実現されています。この統合は以下の3つのコンポーネントで構成されます。

```
ESLint + TypeScript の処理フロー:

  .ts ファイル
       |
       v
  +--------------------------+
  | @typescript-eslint/parser|  TSC の AST を ESLint 形式に変換
  +--------------------------+
       |
       v
  +--------------------------+
  | ESLint ルールエンジン     |  ルールを適用
  |  - @eslint/js            |  (JS 標準ルール)
  |  - @typescript-eslint    |  (TS 専用ルール)
  |  - 型情報ルール           |  (TSC の型チェッカー連携)
  +--------------------------+
       |
       v
  エラー / 警告 レポート
```

### 1-2. typescript-eslint のコアコンポーネント

typescript-eslint は以下の主要パッケージで構成されています。

```
typescript-eslint エコシステム:

┌─────────────────────────────────────────┐
│ typescript-eslint (umbrella package)    │  ← メタパッケージ
└─────────────────────────────────────────┘
              |
              v
    ┌─────────────────────┐
    │ @typescript-eslint/ │
    └─────────────────────┘
              |
    ┌─────────┴──────────────────┐
    |                             |
    v                             v
┌─────────┐                 ┌─────────┐
│ parser  │                 │ plugin  │
└─────────┘                 └─────────┘
    |                             |
    |                             v
    |                    ┌──────────────────┐
    |                    │ eslint-plugin    │
    |                    │ (ルール集)        │
    |                    └──────────────────┘
    |                             |
    v                             v
┌──────────────────┐      ┌────────────────┐
│ TypeScript       │<---->│ Type Checker   │
│ Compiler API     │      │ (型情報)        │
└──────────────────┘      └────────────────┘
```

#### @typescript-eslint/parser

TypeScript のコードを ESLint が理解できる AST（Abstract Syntax Tree）に変換するパーサーです。TypeScript Compiler の AST を ESTree 互換の形式に変換します。

```typescript
// パーサーの役割

// TypeScript コード
const greeting: string = "Hello";

// TypeScript Compiler AST (TSC)
{
  kind: SyntaxKind.VariableDeclaration,
  name: { text: "greeting" },
  type: { kind: SyntaxKind.StringKeyword }
}

// ESTree 形式 (ESLint が理解できる形式)
{
  type: "VariableDeclaration",
  declarations: [{
    id: { type: "Identifier", name: "greeting" },
    typeAnnotation: { type: "TSStringKeyword" }
  }]
}
```

#### @typescript-eslint/eslint-plugin

TypeScript 専用のルール集を提供するプラグインです。300以上のルールが含まれており、型情報を活用するルールも多数あります。

```typescript
// プラグインが提供するルールの例
{
  rules: {
    "@typescript-eslint/no-explicit-any": "error",           // any 禁止
    "@typescript-eslint/no-floating-promises": "error",      // Promise 放置検出
    "@typescript-eslint/switch-exhaustiveness-check": "error" // switch 網羅性
  }
}
```

#### typescript-eslint (umbrella package)

v8 から導入されたメタパッケージで、parser と plugin を統合して簡単に使えるようにしています。

```typescript
// v8 以降の推奨インストール
npm install -D eslint typescript-eslint

// v7 以前 (legacy)
npm install -D eslint @typescript-eslint/parser @typescript-eslint/eslint-plugin
```

### 1-3. 型情報の活用メカニズム

typescript-eslint の最大の特徴は、TypeScript Compiler の型チェッカーと連携して型情報を使ったルールを提供することです。

```
型情報を使うルールの仕組み:

  1. tsconfig.json を読み込む
       |
       v
  2. TypeScript Compiler の型チェッカーを初期化
       |
       v
  3. 各ファイルの型情報を取得
       |
       v
  4. ルールが型情報にアクセス
       |
       v
  5. 型に基づいた検証を実行
```

```typescript
// 型情報を使うルールの例

// no-floating-promises ルールの内部動作イメージ
function checkNode(node: TSESTree.Node) {
  // 1. ノードの型情報を取得
  const type = checker.getTypeAtLocation(node);

  // 2. Promise 型かどうかをチェック
  if (isPromiseType(type)) {
    // 3. await または .catch() がついているかチェック
    if (!isHandled(node)) {
      context.report({
        node,
        messageId: "floatingPromise"
      });
    }
  }
}
```

この型情報ルールは、通常の静的解析では検出できない問題を見つけることができます。

```typescript
// 型情報ルールで検出できる問題

// 問題1: Promise の放置
async function fetchData() {
  fetch("/api/data"); // ← no-floating-promises が警告
}

// 問題2: 型安全でない操作
function process(value: unknown) {
  value.method(); // ← no-unsafe-call が警告
}

// 問題3: async 関数の誤用
[1, 2, 3].forEach(async (n) => { // ← no-misused-promises が警告
  await processNumber(n);
});
```

### 1-4. Flat Config アーキテクチャ

ESLint v9 から導入された Flat Config は、従来の `.eslintrc.*` 形式を置き換える新しい設定形式です。

```
Legacy Config vs Flat Config:

Legacy (.eslintrc.json)          Flat (eslint.config.ts)
┌───────────────────┐            ┌───────────────────┐
│ .eslintrc.json    │            │ eslint.config.ts  │
│ .eslintignore     │            │   (ignores: [])   │
│ extends chain     │    →       │   spread configs  │
│ plugin loading    │            │   explicit import │
│ env/globals       │            │   languageOptions │
└───────────────────┘            └───────────────────┘

  複雑                              シンプル
  暗黙的                            明示的
  設定の結合が不透明                配列のマージ
```

Flat Config の利点:

1. **TypeScript で記述可能** - 型安全な設定
2. **明示的なインポート** - 依存関係が明確
3. **シンプルな結合** - 配列のスプレッド演算子で結合
4. **単一ファイル** - `.eslintignore` 不要

```typescript
// Flat Config の構造
export default [
  // 各要素は ConfigObject
  {
    files: ["**/*.ts"],           // 対象ファイル
    languageOptions: { /* ... */ }, // パーサー設定
    plugins: { /* ... */ },       // プラグイン
    rules: { /* ... */ }          // ルール
  },
  {
    ignores: ["dist/**"]          // 除外パターン
  }
];
```

---

## 2. セットアップと基本設定

### 2-1. インストール

```bash
# 最小構成
npm install -D eslint typescript-eslint

# Prettier 連携を含む構成
npm install -D eslint typescript-eslint eslint-config-prettier

# React プロジェクト
npm install -D eslint typescript-eslint \
  eslint-plugin-react-hooks eslint-plugin-react-refresh

# Node.js プロジェクト
npm install -D eslint typescript-eslint \
  eslint-plugin-node eslint-plugin-security
```

### 2-2. 基本設定ファイル

```typescript
// eslint.config.ts (最小構成)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended
);
```

```typescript
// eslint.config.ts (型情報を使う構成)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  }
);
```

```typescript
// eslint.config.ts (本格的な構成)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettierConfig from "eslint-config-prettier";

export default tseslint.config(
  // 1. 基本ルール
  eslint.configs.recommended,

  // 2. TypeScript ルール (strict + stylistic)
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,

  // 3. パーサー設定
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },

  // 4. カスタムルール
  {
    rules: {
      // async 関連
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": "error",
      "@typescript-eslint/await-thenable": "error",
      "@typescript-eslint/return-await": ["error", "in-try-catch"],

      // 型安全性
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-unsafe-assignment": "error",
      "@typescript-eslint/no-unsafe-call": "error",
      "@typescript-eslint/no-unsafe-member-access": "error",
      "@typescript-eslint/no-unsafe-return": "error",

      // インポート
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixable: "code" },
      ],
      "@typescript-eslint/no-import-type-side-effects": "error",

      // 未使用変数
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],

      // switch 網羅性
      "@typescript-eslint/switch-exhaustiveness-check": "error",
    },
  },

  // 5. テストファイル用の緩和
  {
    files: ["**/*.test.ts", "**/*.spec.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-unsafe-assignment": "off",
    },
  },

  // 6. 除外パターン
  {
    ignores: [
      "dist/",
      "build/",
      "coverage/",
      "node_modules/",
      "**/*.js",
      "**/*.mjs",
    ],
  },

  // 7. Prettier 連携
  prettierConfig
);
```

### 2-3. package.json スクリプト設定

```json
{
  "scripts": {
    // 基本
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",

    // 型チェックと lint を同時実行
    "check": "tsc --noEmit && eslint src/",

    // 並列実行 (npm-run-all 使用)
    "check:parallel": "run-p typecheck lint",
    "typecheck": "tsc --noEmit",

    // タイミング情報付き (パフォーマンス調査用)
    "lint:timing": "TIMING=1 eslint src/",

    // キャッシュをクリア
    "lint:clean": "eslint src/ --cache --cache-location .eslintcache",

    // 変更ファイルのみ
    "lint:changed": "eslint $(git diff --name-only --diff-filter=d HEAD -- '*.ts' '*.tsx')",

    // CI 用 (警告もエラーとして扱う)
    "lint:ci": "eslint src/ --max-warnings 0"
  },
  "devDependencies": {
    "eslint": "^9.0.0",
    "typescript-eslint": "^8.0.0",
    "npm-run-all2": "^6.0.0"
  }
}
```

### 2-4. IDE 統合

#### VS Code 設定

```json
// .vscode/settings.json
{
  // ESLint 拡張機能の設定
  "eslint.enable": true,
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ],

  // Flat Config を有効化
  "eslint.useFlatConfig": true,

  // 保存時に自動修正
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  },

  // Prettier との統合
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode",
    "editor.formatOnSave": true
  },

  // ESLint のワーキングディレクトリ
  "eslint.workingDirectories": [
    { "mode": "auto" }
  ],

  // パフォーマンス設定
  "eslint.lintTask.options": "--cache",

  // デバッグ用
  "eslint.trace.server": "off" // 問題発生時は "verbose" に変更
}
```

```json
// .vscode/extensions.json (推奨拡張機能)
{
  "recommendations": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode"
  ]
}
```

#### WebStorm / IntelliJ IDEA 設定

```
Settings > Languages & Frameworks > JavaScript > Code Quality Tools > ESLint

1. [x] Automatic ESLint configuration
2. Run eslint --fix on save: [x]
3. Configuration file: eslint.config.ts
```

### 2-5. Git フック統合

```bash
# husky + lint-staged のインストール
npm install -D husky lint-staged
npx husky init
```

```json
// package.json
{
  "scripts": {
    "prepare": "husky"
  },
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ]
  }
}
```

```bash
# .husky/pre-commit
npm run lint-staged
```

---

## 3. 推奨ルールセットの比較と選択

### 3-1. ルールセットの階層構造

typescript-eslint は複数のルールセットを提供しており、段階的に厳密度を高めることができます。

```
ルールセットの階層:

  recommended
       |
       +-- 基本的なバグ検出
       +-- 型情報不要
       +-- ルール数: ~50
       |
       v
  recommendedTypeChecked
       |
       +-- recommended を含む
       +-- 型情報を使うルール追加
       +-- ルール数: ~70
       |
       v
  strictTypeChecked
       |
       +-- recommendedTypeChecked を含む
       +-- より厳格なルール追加
       +-- ルール数: ~90
       |
       v
  strictTypeChecked + stylisticTypeChecked
       |
       +-- strict を含む
       +-- コードスタイル統一ルール追加
       +-- ルール数: ~110
```

### 3-2. recommended

基本的なバグ検出ルールのみを有効化します。型情報は不要です。

```typescript
// eslint.config.ts
import tseslint from "typescript-eslint";

export default tseslint.config(
  ...tseslint.configs.recommended
);
```

含まれる主なルール:

- `@typescript-eslint/no-explicit-any` - any の使用を警告
- `@typescript-eslint/no-unused-vars` - 未使用変数を検出
- `@typescript-eslint/no-array-constructor` - Array コンストラクタの誤用を検出
- `@typescript-eslint/ban-ts-comment` - @ts-ignore の使用を制限

利点:
- 高速 (型情報不要)
- セットアップが簡単
- 既存プロジェクトへの導入が容易

欠点:
- 型情報を使う高度なチェックができない
- Promise の誤用などを検出できない

### 3-3. recommendedTypeChecked

型情報を使うルールを追加します。

```typescript
// eslint.config.ts
import tseslint from "typescript-eslint";

export default tseslint.config(
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  }
);
```

追加される主なルール:

- `@typescript-eslint/no-floating-promises` - Promise の放置を検出
- `@typescript-eslint/no-misused-promises` - async 関数の誤用を検出
- `@typescript-eslint/await-thenable` - Promise でない値への await を検出
- `@typescript-eslint/no-unnecessary-condition` - 不要な条件分岐を検出

利点:
- tsc では検出できない問題を発見
- 非同期処理のバグを防ぐ
- 型安全性が大幅に向上

欠点:
- 型チェックのため実行速度が遅い
- tsconfig.json の設定が必要

### 3-4. strictTypeChecked

より厳格なルールセットです。

```typescript
// eslint.config.ts
import tseslint from "typescript-eslint";

export default tseslint.config(
  ...tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  }
);
```

追加される主なルール:

- `@typescript-eslint/no-confusing-void-expression` - void 式の誤用を検出
- `@typescript-eslint/no-unnecessary-boolean-literal-compare` - 不要な真偽値比較を検出
- `@typescript-eslint/prefer-reduce-type-parameter` - reduce の型パラメータ使用を推奨
- `@typescript-eslint/restrict-template-expressions` - テンプレート文字列の型制限

利点:
- 最高レベルの型安全性
- コードの一貫性が向上
- ライブラリ開発に最適

欠点:
- 既存コードへの適用が困難
- 一部のルールが厳しすぎる場合がある

### 3-5. stylisticTypeChecked

コードスタイルの統一ルールを追加します。

```typescript
// eslint.config.ts
import tseslint from "typescript-eslint";

export default tseslint.config(
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  }
);
```

追加される主なルール:

- `@typescript-eslint/consistent-type-definitions` - type vs interface の統一
- `@typescript-eslint/consistent-type-imports` - import type の使用を強制
- `@typescript-eslint/prefer-function-type` - 関数型の統一
- `@typescript-eslint/array-type` - 配列型表記の統一

### 3-6. ルールセット比較表

| ルールセット | ルール数 | 型情報 | 速度 | 厳密度 | 推奨シーン |
|-------------|---------|--------|------|--------|-----------|
| recommended | ~50 | 不要 | 最速 | 低 | 初期導入、レガシーコード |
| recommendedTypeChecked | ~70 | 必要 | 遅い | 中 | 一般的なプロジェクト |
| strictTypeChecked | ~90 | 必要 | 遅い | 高 | ライブラリ、高品質要求 |
| strict + stylistic | ~110 | 必要 | 遅い | 最高 | チーム開発、OSS |

### 3-7. プロジェクト別の推奨ルールセット

```typescript
// 新規プロジェクト: strictTypeChecked + stylisticTypeChecked
export default tseslint.config(
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked
);

// 既存プロジェクト: recommendedTypeChecked からスタート
export default tseslint.config(
  ...tseslint.configs.recommendedTypeChecked
);

// レガシーコード: recommended のみ
export default tseslint.config(
  ...tseslint.configs.recommended
);

// ライブラリ: strictTypeChecked + 独自ルール
export default tseslint.config(
  ...tseslint.configs.strictTypeChecked,
  {
    rules: {
      "@typescript-eslint/explicit-module-boundary-types": "error",
      "@typescript-eslint/explicit-function-return-type": "error",
    },
  }
);
```

---

## 4. 型情報を使うルールの深堀り

型情報ルール（type-checked rules）は、typescript-eslint の最大の特徴です。TypeScript Compiler の型チェッカーと連携することで、通常の静的解析では検出できない問題を発見します。

### 4-1. no-floating-promises

Promise を await または .catch() せずに放置することを検出します。

```typescript
// NG: Promise を放置
async function fetchData(): Promise<void> {
  const response = await fetch("/api/data");
  response.json(); // 警告: Promises must be awaited, end with a call to .catch, or end with a call to .then with a rejection handler
  // ↑ この Promise は処理されず、エラーが握りつぶされる
}

// OK: await する
async function fetchData(): Promise<void> {
  const response = await fetch("/api/data");
  const data = await response.json();
  console.log(data);
}

// OK: .catch() でエラーハンドリング
async function fetchData(): Promise<void> {
  const response = await fetch("/api/data");
  response.json().catch((err) => {
    console.error("JSON parse error:", err);
  });
}

// OK: void 演算子で意図的に無視
async function backgroundTask(): Promise<void> {
  void someAsyncOperation(); // バックグラウンドで実行
}

// NG: 変数に代入しても await しなければエラー
async function fetchData(): Promise<void> {
  const promise = fetch("/api/data"); // 警告
  // promise を await していない
}
```

実際の問題例:

```typescript
// 問題: データベース接続の切断を忘れる
class UserRepository {
  async save(user: User): Promise<void> {
    await db.insert(users).values(user);
    db.close(); // 警告: db.close() は Promise を返すが await していない
  }
}

// 修正
class UserRepository {
  async save(user: User): Promise<void> {
    await db.insert(users).values(user);
    await db.close();
  }
}
```

### 4-2. no-misused-promises

Promise を期待していない場所で使用することを検出します。

```typescript
// NG: forEach のコールバックに async 関数
const items = [1, 2, 3];
items.forEach(async (item) => {
  // 警告: Promise returned in function argument where a void return was expected
  await processItem(item);
});
// forEach は返り値の Promise を待たないため、エラーが握りつぶされる

// OK: for...of で順次処理
for (const item of items) {
  await processItem(item);
}

// OK: Promise.all で並列処理
await Promise.all(items.map((item) => processItem(item)));

// NG: 条件式に Promise
if (fetchUser(id)) { // 警告: Expected non-Promise value in a boolean conditional
  // Promise は常に truthy なので、この条件は常に true
}

// OK: await してから条件判定
const user = await fetchUser(id);
if (user) {
  // ...
}

// NG: addEventListener に async 関数
button.addEventListener("click", async () => {
  // 警告: Promise-returning function provided to attribute where a void return was expected
  await handleClick();
});

// OK: エラーハンドリング付き
button.addEventListener("click", () => {
  handleClick().catch((err) => {
    console.error("Click handler error:", err);
  });
});
```

実際の問題例:

```typescript
// 問題: Array.prototype.map の誤用
async function getUserNames(ids: string[]): Promise<string[]> {
  // NG: map は Promise を待たない
  const names = ids.map(async (id) => {
    const user = await fetchUser(id);
    return user.name;
  });
  return names; // 型エラー: Promise<string>[] を返している
}

// 修正
async function getUserNames(ids: string[]): Promise<string[]> {
  const names = await Promise.all(
    ids.map(async (id) => {
      const user = await fetchUser(id);
      return user.name;
    })
  );
  return names;
}
```

### 4-3. await-thenable

await が必要ない値に await を使用することを検出します。

```typescript
// NG: Promise でない値を await
function getUser(id: string): User {
  return database[id];
}

async function main() {
  const user = await getUser("123"); // 警告: Unexpected await on a non-Promise value
}

// OK: Promise を返す関数のみ await
async function getUser(id: string): Promise<User> {
  return fetch(`/api/users/${id}`).then((r) => r.json());
}

async function main() {
  const user = await getUser("123");
}

// NG: 既に await された値を再度 await
async function fetchData(): Promise<Data> {
  const response = await fetch("/api/data");
  const data = await await response.json(); // 警告: 二重 await
  return data;
}
```

### 4-4. no-unnecessary-condition

常に true または false になる条件式を検出します。

```typescript
// NG: 常に true の条件
function process(value: string) {
  if (value !== undefined) { // 警告: Unnecessary conditional, value is always defined
    // value は string 型なので undefined になることはない
  }
}

// OK: Optional な値の条件チェック
function process(value?: string) {
  if (value !== undefined) {
    // value は string | undefined なので適切
  }
}

// NG: 常に false の条件
function check(num: number) {
  if (num === "123") { // 警告: This comparison appears to be unintentional
    // number と string は等価になることはない
  }
}

// NG: 不要な null チェック
function greet(name: string) {
  return name ?? "Guest"; // 警告: Unnecessary nullish coalescing operator
  // name は string 型なので null/undefined にならない
}

// OK: Optional な値への null チェック
function greet(name?: string) {
  return name ?? "Guest";
}
```

### 4-5. switch-exhaustiveness-check

switch 文でユニオン型の全ケースを網羅しているかチェックします。

```typescript
// @typescript-eslint/switch-exhaustiveness-check: "error"

type Status = "active" | "inactive" | "pending" | "archived";

// NG: ケースが不足
function getLabel(status: Status): string {
  switch (status) {
    case "active":
      return "有効";
    case "inactive":
      return "無効";
    case "pending":
      return "保留中";
    // 警告: Switch is not exhaustive. Missing case: "archived"
  }
}

// OK: 全ケースを網羅
function getLabel(status: Status): string {
  switch (status) {
    case "active":
      return "有効";
    case "inactive":
      return "無効";
    case "pending":
      return "保留中";
    case "archived":
      return "アーカイブ済み";
  }
}

// OK: default で網羅
function getLabel(status: Status): string {
  switch (status) {
    case "active":
      return "有効";
    case "inactive":
      return "無効";
    case "pending":
      return "保留中";
    default:
      return "その他";
  }
}
```

型安全な exhaustive check パターン:

```typescript
// never 型を使った網羅性チェック
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}

function getLabel(status: Status): string {
  switch (status) {
    case "active":
      return "有効";
    case "inactive":
      return "無効";
    case "pending":
      return "保留中";
    default:
      // Status に新しい値が追加されると、ここで型エラーが発生
      return assertNever(status);
  }
}
```

### 4-6. no-unsafe-* ルール群

any 型の値に対する操作を制限します。

```typescript
// no-unsafe-assignment
function process(data: any) {
  const value: string = data; // 警告: Unsafe assignment of an any value
}

// no-unsafe-call
function execute(fn: any) {
  fn(); // 警告: Unsafe call of an any typed value
}

// no-unsafe-member-access
function getValue(obj: any) {
  return obj.value; // 警告: Unsafe member access .value on an any value
}

// no-unsafe-return
function getData(): string {
  const data: any = fetchData();
  return data; // 警告: Unsafe return of an any typed value
}

// OK: 型アサーションまたは型ガードを使用
function process(data: unknown) {
  if (typeof data === "string") {
    const value: string = data;
  }
}

// OK: Zod などのバリデーションライブラリを使用
import { z } from "zod";

const UserSchema = z.object({
  name: z.string(),
  age: z.number(),
});

function process(data: unknown) {
  const user = UserSchema.parse(data); // 型安全
  console.log(user.name);
}
```

### 4-7. prefer-nullish-coalescing

`||` の代わりに `??` を使うべき場合を検出します。

```typescript
// NG: || を使うと 0 や '' が falsy として扱われる
function getCount(count: number | undefined): number {
  return count || 0; // 警告: Prefer using nullish coalescing operator
  // count が 0 の場合も 0 が返る (意図しない動作)
}

// OK: ?? を使う
function getCount(count: number | undefined): number {
  return count ?? 0;
  // count が undefined の場合のみ 0 が返る
}

// 実際の問題例
interface Config {
  port?: number;
  debug?: boolean;
}

// NG: || を使うと意図しない動作
function getConfig(config: Config) {
  const port = config.port || 3000; // port: 0 が設定されても 3000 になる
  const debug = config.debug || false; // debug: false が設定されても false になる (一見正しいが意図が不明確)
}

// OK: ?? を使う
function getConfig(config: Config) {
  const port = config.port ?? 3000;
  const debug = config.debug ?? false;
}
```

### 4-8. 型情報ルールのパフォーマンス影響

型情報ルールは型チェックのため、実行時間が大幅に増加します。

```
ベンチマーク (10,000 行の TypeScript プロジェクト):

  recommended (型情報なし)
    └─ 実行時間: 2.3 秒

  recommendedTypeChecked (型情報あり)
    └─ 実行時間: 12.8 秒 (5.6倍)

  strictTypeChecked (型情報あり)
    └─ 実行時間: 15.4 秒 (6.7倍)
```

高速化の方法:

```typescript
// 1. projectService を使用 (v8 以降)
{
  languageOptions: {
    parserOptions: {
      projectService: true, // 従来の project より高速
      tsconfigRootDir: import.meta.dirname,
    },
  },
}

// 2. キャッシュを有効化
// package.json
{
  "scripts": {
    "lint": "eslint src/ --cache --cache-location .eslintcache"
  }
}

// 3. 変更ファイルのみ lint
{
  "scripts": {
    "lint:changed": "eslint $(git diff --name-only --diff-filter=d HEAD -- '*.ts')"
  }
}
```

---

## 5. カスタムルールの作成

typescript-eslint では、AST 操作と型情報アクセスを使って独自のルールを作成できます。

### 5-1. ルールの基本構造

```typescript
// my-rule.ts
import { ESLintUtils } from "@typescript-eslint/utils";

const createRule = ESLintUtils.RuleCreator(
  (name) => `https://example.com/rule/${name}`
);

export const myRule = createRule({
  name: "my-rule",
  meta: {
    type: "problem",
    docs: {
      description: "ルールの説明",
    },
    messages: {
      errorMessage: "エラーメッセージ: {{value}}",
    },
    schema: [],
  },
  defaultOptions: [],
  create(context) {
    return {
      // AST ノードに対するビジター関数
      Identifier(node) {
        if (node.name === "badName") {
          context.report({
            node,
            messageId: "errorMessage",
            data: {
              value: node.name,
            },
          });
        }
      },
    };
  },
});
```

### 5-2. 実例: no-console-log ルール

console.log を禁止し、代わりに logger を使用させるルール。

```typescript
// rules/no-console-log.ts
import { ESLintUtils } from "@typescript-eslint/utils";
import type { TSESTree } from "@typescript-eslint/utils";

const createRule = ESLintUtils.RuleCreator(
  (name) => `https://example.com/rule/${name}`
);

export const noConsoleLog = createRule({
  name: "no-console-log",
  meta: {
    type: "suggestion",
    docs: {
      description: "console.log を禁止し、logger.info を使用させる",
    },
    messages: {
      useLogger: "console.log の代わりに logger.info を使用してください",
    },
    fixable: "code",
    schema: [],
  },
  defaultOptions: [],
  create(context) {
    return {
      MemberExpression(node: TSESTree.MemberExpression) {
        // console.log をチェック
        if (
          node.object.type === "Identifier" &&
          node.object.name === "console" &&
          node.property.type === "Identifier" &&
          node.property.name === "log"
        ) {
          context.report({
            node,
            messageId: "useLogger",
            fix(fixer) {
              // 自動修正: console.log → logger.info
              return fixer.replaceText(node, "logger.info");
            },
          });
        }
      },
    };
  },
});
```

### 5-3. 型情報を使うルール: no-untyped-fetch

fetch の返り値を型アサーションせずに使用することを禁止するルール。

```typescript
// rules/no-untyped-fetch.ts
import { ESLintUtils } from "@typescript-eslint/utils";
import type { TSESTree } from "@typescript-eslint/utils";

const createRule = ESLintUtils.RuleCreator(
  (name) => `https://example.com/rule/${name}`
);

export const noUntypedFetch = createRule({
  name: "no-untyped-fetch",
  meta: {
    type: "problem",
    docs: {
      description: "fetch の返り値は型アサーションまたはバリデーションが必要",
    },
    messages: {
      untypedFetch: "fetch の返り値を型安全に扱ってください (Zod, as 等)",
    },
    schema: [],
  },
  defaultOptions: [],
  create(context) {
    const services = ESLintUtils.getParserServices(context);
    const checker = services.program.getTypeChecker();

    return {
      AwaitExpression(node: TSESTree.AwaitExpression) {
        if (node.argument.type !== "CallExpression") return;

        const callee = node.argument.callee;
        if (callee.type !== "Identifier" || callee.name !== "fetch") return;

        // 親ノードをチェック
        const parent = node.parent;

        // 型アサーションがあるかチェック
        if (parent?.type === "TSAsExpression") return;

        // 変数宣言の場合、型注釈があるかチェック
        if (
          parent?.type === "VariableDeclarator" &&
          parent.id.type === "Identifier" &&
          parent.id.typeAnnotation
        ) {
          return;
        }

        context.report({
          node,
          messageId: "untypedFetch",
        });
      },
    };
  },
});
```

使用例:

```typescript
// NG: 型情報なし
const response = await fetch("/api/users");

// OK: 型アサーション
const response = await fetch("/api/users") as Response<User[]>;

// OK: 型注釈付き変数宣言
const response: Response<User[]> = await fetch("/api/users");

// OK: Zod バリデーション
const response = await fetch("/api/users");
const users = UserArraySchema.parse(await response.json());
```

### 5-4. プラグインとして配布

```typescript
// index.ts
import { noConsoleLog } from "./rules/no-console-log";
import { noUntypedFetch } from "./rules/no-untyped-fetch";

export default {
  rules: {
    "no-console-log": noConsoleLog,
    "no-untyped-fetch": noUntypedFetch,
  },
};
```

```typescript
// eslint.config.ts
import myPlugin from "./my-eslint-plugin";

export default [
  {
    plugins: {
      "my-plugin": myPlugin,
    },
    rules: {
      "my-plugin/no-console-log": "error",
      "my-plugin/no-untyped-fetch": "error",
    },
  },
];
```

### 5-5. AST Explorer の活用

ルール作成時は AST Explorer を使うと便利です。

```
AST Explorer の使い方:

1. https://astexplorer.net/ にアクセス
2. Parser: @typescript-eslint/parser を選択
3. コードを入力
4. AST 構造を確認
5. ルールで使うノードタイプを特定
```

例:

```typescript
// コード
const user = { name: "Alice" };

// AST (抜粋)
{
  type: "VariableDeclaration",
  declarations: [{
    type: "VariableDeclarator",
    id: { type: "Identifier", name: "user" },
    init: {
      type: "ObjectExpression",
      properties: [{
        type: "Property",
        key: { type: "Identifier", name: "name" },
        value: { type: "Literal", value: "Alice" }
      }]
    }
  }]
}
```

---

## 6. Prettier との統合と競合解消

### 6-1. 統合の基本

ESLint はコード品質ルール、Prettier はコードフォーマッターという役割分担が基本です。

```
ESLint vs Prettier:

ESLint
  ├─ 機能: コード品質チェック
  ├─ 例: 未使用変数、型エラー、ロジックの問題
  └─ 設定: eslint.config.ts

Prettier
  ├─ 機能: コードフォーマット
  ├─ 例: インデント、改行、セミコロン
  └─ 設定: .prettierrc
```

### 6-2. eslint-config-prettier

ESLint のフォーマットルールを無効化し、Prettier と競合しないようにします。

```bash
npm install -D eslint-config-prettier
```

```typescript
// eslint.config.ts
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettierConfig from "eslint-config-prettier";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  prettierConfig, // 最後に配置して競合ルールを無効化
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  }
);
```

### 6-3. Prettier 設定

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": false,
  "tabWidth": 2,
  "trailingComma": "all",
  "printWidth": 100,
  "arrowParens": "always",
  "endOfLine": "lf"
}
```

```
// .prettierignore
dist/
build/
coverage/
node_modules/
*.min.js
```

### 6-4. 実行順序

```json
// package.json
{
  "scripts": {
    // 1. Prettier でフォーマット
    "format": "prettier --write 'src/**/*.{ts,tsx}'",

    // 2. ESLint でチェック
    "lint": "eslint src/",

    // 3. まとめて実行
    "check": "npm run format && npm run lint"
  }
}
```

VS Code での統合:

```json
// .vscode/settings.json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

### 6-5. 競合する可能性のあるルール

eslint-config-prettier は以下のルールを無効化します:

```typescript
// 無効化されるルール例

// インデント
"@typescript-eslint/indent": "off",

// 改行
"@typescript-eslint/comma-dangle": "off",

// セミコロン
"@typescript-eslint/semi": "off",

// クォート
"@typescript-eslint/quotes": "off",

// スペース
"@typescript-eslint/space-before-function-paren": "off",
```

手動で競合を解決する場合:

```typescript
// eslint.config.ts
export default tseslint.config(
  ...tseslint.configs.strictTypeChecked,
  {
    rules: {
      // Prettier がフォーマットするルールをオフ
      "@typescript-eslint/indent": "off",
      "@typescript-eslint/quotes": "off",
      "@typescript-eslint/semi": "off",
      "@typescript-eslint/comma-dangle": "off",

      // 品質ルールは有効化
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-floating-promises": "error",
    },
  }
);
```

### 6-6. dprint との統合

dprint は Prettier の高速代替です。

```bash
npm install -D dprint
```

```json
// dprint.json
{
  "typescript": {
    "semiColons": "prefer",
    "quoteStyle": "alwaysDouble",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "includes": ["**/*.{ts,tsx,js,jsx,json}"],
  "excludes": ["node_modules", "dist"]
}
```

```json
// package.json
{
  "scripts": {
    "format": "dprint fmt",
    "format:check": "dprint check"
  }
}
```

---

## 7. プロジェクト別設定パターン

### 7-1. React プロジェクト

```typescript
// eslint.config.ts (React + TypeScript)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import reactPlugin from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import jsxA11y from "eslint-plugin-jsx-a11y";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ["**/*.{jsx,tsx}"],
    plugins: {
      react: reactPlugin,
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
      "jsx-a11y": jsxA11y,
    },
    rules: {
      ...reactPlugin.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,

      // React Refresh
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],

      // React 設定
      "react/react-in-jsx-scope": "off", // React 17+ では不要
      "react/prop-types": "off", // TypeScript を使うので不要

      // Hooks ルール
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // アクセシビリティ
      ...jsxA11y.configs.recommended.rules,
    },
    settings: {
      react: {
        version: "detect",
      },
    },
  }
);
```

### 7-2. Next.js プロジェクト

```typescript
// eslint.config.ts (Next.js)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import nextPlugin from "@next/eslint-plugin-next";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    plugins: {
      "@next/next": nextPlugin,
    },
    rules: {
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs["core-web-vitals"].rules,
    },
  },
  {
    // App Router のサーバーコンポーネント
    files: ["app/**/*.tsx"],
    rules: {
      "react-hooks/rules-of-hooks": "off", // サーバーコンポーネントでは不要
    },
  }
);
```

### 7-3. Node.js / Express プロジェクト

```typescript
// eslint.config.ts (Node.js)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import nodePlugin from "eslint-plugin-node";
import securityPlugin from "eslint-plugin-security";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
      globals: {
        // Node.js グローバル変数
        __dirname: "readonly",
        __filename: "readonly",
        process: "readonly",
        Buffer: "readonly",
      },
    },
  },
  {
    plugins: {
      node: nodePlugin,
      security: securityPlugin,
    },
    rules: {
      // Node.js ルール
      "node/no-unsupported-features/es-syntax": "off", // TypeScript で transpile するため
      "node/no-missing-import": "off", // TypeScript の解決に任せる

      // セキュリティルール
      ...securityPlugin.configs.recommended.rules,
      "security/detect-object-injection": "off", // 誤検知が多いため

      // TypeScript 固有
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": "error",
    },
  }
);
```

### 7-4. ライブラリプロジェクト

```typescript
// eslint.config.ts (ライブラリ)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ["src/**/*.ts"],
    rules: {
      // 公開 API は全て型を明示
      "@typescript-eslint/explicit-module-boundary-types": "error",
      "@typescript-eslint/explicit-function-return-type": [
        "error",
        {
          allowExpressions: false,
          allowTypedFunctionExpressions: true,
        },
      ],

      // any を完全禁止
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-unsafe-assignment": "error",
      "@typescript-eslint/no-unsafe-call": "error",
      "@typescript-eslint/no-unsafe-member-access": "error",
      "@typescript-eslint/no-unsafe-return": "error",

      // 一貫性のある型定義
      "@typescript-eslint/consistent-type-definitions": ["error", "interface"],
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixable: "code" },
      ],

      // 命名規則
      "@typescript-eslint/naming-convention": [
        "error",
        {
          selector: "default",
          format: ["camelCase"],
        },
        {
          selector: "variable",
          format: ["camelCase", "UPPER_CASE"],
        },
        {
          selector: "typeLike",
          format: ["PascalCase"],
        },
        {
          selector: "interface",
          format: ["PascalCase"],
          custom: {
            regex: "^I[A-Z]",
            match: false, // I プレフィックス禁止
          },
        },
      ],
    },
  },
  {
    // テストファイルは緩和
    files: ["**/*.test.ts", "**/*.spec.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/explicit-function-return-type": "off",
    },
  }
);
```

---

## 8. モノレポでの設定共有

### 8-1. モノレポ構成例

```
monorepo/
├── eslint.config.ts          ← ルート設定（共通ルール）
├── packages/
│   ├── shared/
│   │   ├── eslint.config.ts  ← ライブラリ固有ルール
│   │   ├── src/
│   │   └── tsconfig.json
│   ├── web/
│   │   ├── eslint.config.ts  ← React 固有ルール
│   │   ├── src/
│   │   └── tsconfig.json
│   └── api/
│       ├── eslint.config.ts  ← Node.js 固有ルール
│       ├── src/
│       └── tsconfig.json
└── package.json
```

### 8-2. ルート設定

```typescript
// eslint.config.ts (root)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettierConfig from "eslint-config-prettier";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: {
          allowDefaultProject: ["*.js", "*.mjs"],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    rules: {
      // モノレポ全体で適用するルール
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixable: "code" },
      ],
    },
  },
  prettierConfig,
  {
    ignores: [
      "**/dist/**",
      "**/build/**",
      "**/node_modules/**",
      "**/*.js",
    ],
  }
);
```

### 8-3. パッケージ固有の設定

```typescript
// packages/shared/eslint.config.ts (ライブラリ)
import rootConfig from "../../eslint.config.ts";

export default [
  ...rootConfig,
  {
    files: ["src/**/*.ts"],
    rules: {
      // ライブラリは特に厳格
      "@typescript-eslint/explicit-module-boundary-types": "error",
      "@typescript-eslint/explicit-function-return-type": "error",
    },
  },
];
```

```typescript
// packages/web/eslint.config.ts (React)
import rootConfig from "../../eslint.config.ts";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default [
  ...rootConfig,
  {
    files: ["src/**/*.{ts,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  },
];
```

```typescript
// packages/api/eslint.config.ts (Node.js)
import rootConfig from "../../eslint.config.ts";
import nodePlugin from "eslint-plugin-node";

export default [
  ...rootConfig,
  {
    files: ["src/**/*.ts"],
    plugins: {
      node: nodePlugin,
    },
    rules: {
      // Node.js 固有ルール
      "node/no-unsupported-features/es-syntax": "off",
    },
  },
];
```

### 8-4. package.json スクリプト

```json
// package.json (root)
{
  "scripts": {
    "lint": "npm run lint --workspaces --if-present",
    "lint:fix": "npm run lint:fix --workspaces --if-present",
    "typecheck": "npm run typecheck --workspaces --if-present"
  },
  "workspaces": [
    "packages/*"
  ]
}
```

```json
// packages/web/package.json
{
  "scripts": {
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",
    "typecheck": "tsc --noEmit"
  }
}
```

### 8-5. Turborepo との統合

```json
// turbo.json
{
  "pipeline": {
    "lint": {
      "cache": true,
      "outputs": []
    },
    "typecheck": {
      "cache": true,
      "outputs": []
    }
  }
}
```

```json
// package.json (root)
{
  "scripts": {
    "lint": "turbo run lint",
    "typecheck": "turbo run typecheck"
  }
}
```

---

## 9. CI/CD パイプライン統合

### 9-1. GitHub Actions

```yaml
# .github/workflows/lint.yml
name: Lint

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"

      - name: Install dependencies
        run: npm ci

      - name: Type check
        run: npm run typecheck

      - name: ESLint
        run: npm run lint -- --max-warnings 0

      # ESLint の結果を PR コメントに出力
      - uses: reviewdog/action-eslint@v1
        if: always()
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          reporter: github-pr-review
          eslint_flags: "src/"
```

### 9-2. 変更ファイルのみ lint

```yaml
# .github/workflows/lint-changed.yml
name: Lint Changed Files

on:
  pull_request:

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # 全履歴を取得

      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"

      - run: npm ci

      - name: Get changed TypeScript files
        id: changed-files
        uses: tj-actions/changed-files@v42
        with:
          files: |
            **/*.ts
            **/*.tsx

      - name: Lint changed files
        if: steps.changed-files.outputs.any_changed == 'true'
        run: |
          echo "Changed files: ${{ steps.changed-files.outputs.all_changed_files }}"
          npx eslint ${{ steps.changed-files.outputs.all_changed_files }}
```

### 9-3. キャッシュの活用

```yaml
# .github/workflows/lint-with-cache.yml
name: Lint with Cache

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"

      - run: npm ci

      # ESLint キャッシュを復元
      - name: Restore ESLint cache
        uses: actions/cache@v4
        with:
          path: .eslintcache
          key: eslint-${{ runner.os }}-${{ hashFiles('**/eslint.config.ts') }}

      - name: Lint
        run: npm run lint -- --cache --cache-location .eslintcache

      # 型チェックキャッシュも活用
      - name: Restore tsc cache
        uses: actions/cache@v4
        with:
          path: .tsbuildinfo
          key: tsc-${{ runner.os }}-${{ hashFiles('**/tsconfig.json') }}

      - name: Type check
        run: npm run typecheck
```

### 9-4. 並列実行

```yaml
# .github/workflows/lint-parallel.yml
name: Lint (Parallel)

on: [pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        package: [shared, web, api] # モノレポの各パッケージ

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"

      - run: npm ci

      - name: Lint ${{ matrix.package }}
        run: npm run lint -w packages/${{ matrix.package }}
```

### 9-5. GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - lint
  - test

lint:
  stage: lint
  image: node:20
  cache:
    paths:
      - node_modules/
      - .eslintcache
  script:
    - npm ci
    - npm run lint -- --cache --cache-location .eslintcache --max-warnings 0
  artifacts:
    reports:
      junit: eslint-report.xml
```

### 9-6. エラーレポートの出力

```json
// package.json
{
  "scripts": {
    "lint:ci": "eslint src/ --format json --output-file eslint-report.json || true"
  }
}
```

```yaml
# .github/workflows/lint-report.yml
- name: Lint with JSON report
  run: npm run lint:ci

- name: Upload ESLint report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: eslint-report
    path: eslint-report.json
```

---

## 10. パフォーマンスチューニング

### 10-1. パフォーマンス計測

```bash
# TIMING 環境変数で各ルールの実行時間を計測
TIMING=1 eslint src/

# 出力例:
Rule                                    | Time (ms) | Relative
:---------------------------------------|----------:|--------:
@typescript-eslint/no-floating-promises |  2456.234 |    32.1%
@typescript-eslint/no-unsafe-assignment |  1234.567 |    16.1%
@typescript-eslint/no-misused-promises  |   987.654 |    12.9%
...
```

```json
// package.json
{
  "scripts": {
    "lint:timing": "TIMING=1 eslint src/"
  }
}
```

### 10-2. projectService の活用

v8 から導入された projectService は、型情報ルールのパフォーマンスを大幅に改善します。

```typescript
// 従来の設定 (遅い)
{
  languageOptions: {
    parserOptions: {
      project: "./tsconfig.json",
      tsconfigRootDir: import.meta.dirname,
    },
  },
}

// v8 の設定 (高速)
{
  languageOptions: {
    parserOptions: {
      projectService: true, // TypeScript の Language Service を活用
      tsconfigRootDir: import.meta.dirname,
    },
  },
}
```

projectService の仕組み:

```
従来の project 指定:
  1. tsconfig.json をパース
  2. TypeScript Compiler を毎回初期化
  3. 全ファイルを解析
  → 遅い

projectService:
  1. TypeScript Language Service を起動
  2. インクリメンタル解析を活用
  3. 変更ファイルのみ再解析
  → 高速 (約2-5倍)
```

### 10-3. キャッシュの活用

```json
// package.json
{
  "scripts": {
    "lint": "eslint src/ --cache --cache-location .eslintcache",
    "lint:clean": "rm -rf .eslintcache && npm run lint"
  }
}
```

```
// .gitignore
.eslintcache
```

キャッシュの効果:

```
初回実行: 15.3 秒
2回目実行 (キャッシュあり): 1.2 秒 (12.8倍高速)
```

### 10-4. 並列実行

```bash
# eslint-parallel を使用
npm install -D eslint-parallel
```

```json
// package.json
{
  "scripts": {
    "lint": "eslint-parallel src/**/*.ts"
  }
}
```

並列実行の効果:

```
シングルコア: 15.3 秒
4コア並列: 4.8 秒 (3.2倍高速)
8コア並列: 3.1 秒 (4.9倍高速)
```

### 10-5. ルールの選択的無効化

パフォーマンスが重要な場合、特定のルールを無効化します。

```typescript
// eslint.config.ts
export default tseslint.config(
  ...tseslint.configs.recommendedTypeChecked,
  {
    rules: {
      // 型情報ルールで特に遅いもの
      "@typescript-eslint/no-unnecessary-condition": "off", // 遅い
      "@typescript-eslint/strict-boolean-expressions": "off", // 遅い

      // 必須のルールは残す
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": "error",
    },
  }
);
```

### 10-6. ファイル除外の最適化

```typescript
// eslint.config.ts
export default tseslint.config(
  // ...
  {
    ignores: [
      "**/node_modules/**",
      "**/dist/**",
      "**/build/**",
      "**/coverage/**",
      "**/*.min.js",
      "**/*.d.ts", // 型定義ファイルは除外
      "**/generated/**", // 自動生成ファイルは除外
    ],
  }
);
```

### 10-7. CI での最適化

```yaml
# .github/workflows/lint.yml
jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
      # 1. シャローコピー（履歴不要）
      - uses: actions/checkout@v4
        with:
          fetch-depth: 1

      # 2. Node.js キャッシュ
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"

      # 3. 依存関係キャッシュ
      - name: Cache node_modules
        uses: actions/cache@v4
        with:
          path: node_modules
          key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}

      # 4. ESLint キャッシュ
      - name: Cache ESLint
        uses: actions/cache@v4
        with:
          path: .eslintcache
          key: eslint-${{ runner.os }}-${{ hashFiles('**/eslint.config.ts') }}

      # 5. 並列実行
      - name: Lint
        run: eslint-parallel src/**/*.ts --cache
```

### 10-8. パフォーマンスベンチマーク

```
プロジェクト規模: 10,000 行の TypeScript コード

設定                                  | 実行時間
-------------------------------------|--------
recommended (型情報なし)               | 2.3 秒
recommendedTypeChecked               | 12.8 秒
recommendedTypeChecked + projectService | 6.4 秒
+ キャッシュ                          | 0.8 秒
+ 並列実行 (4コア)                    | 2.1 秒
+ 変更ファイルのみ                     | 0.3 秒
```

---

## 11. Biome/oxlint との比較と移行

### 11-1. ツール比較表

| 機能 | ESLint + typescript-eslint | Biome | oxlint |
|------|---------------------------|-------|--------|
| 対応言語 | JS, TS, JSX, TSX | JS, TS, JSX, TSX, JSON | JS, TS, JSX, TSX |
| ルール数 | 300+ | 200+ | 350+ |
| 型情報ルール | ✓ | ✗ | ✗ |
| フォーマッター | Prettier 連携 | 内蔵 | ✗ |
| 速度 | 遅い | 最速 (25-100倍) | 最速 (50-100倍) |
| エコシステム | 最大 | 成長中 | 限定的 |
| 設定ファイル | eslint.config.ts | biome.json | .oxlintrc.json |
| 自動修正 | ✓ | ✓ | ✓ (一部) |
| IDE 統合 | 完璧 | 良好 | 限定的 |
| プラグイン | 豊富 | 少ない | なし |

### 11-2. Biome の特徴

Biome は ESLint + Prettier を置き換える高速ツールです。

```bash
# インストール
npm install -D @biomejs/biome
```

```json
// biome.json
{
  "$schema": "https://biomejs.dev/schemas/1.0.0/schema.json",
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "suspicious": {
        "noExplicitAny": "error"
      },
      "style": {
        "useConst": "error"
      }
    }
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 100
  },
  "javascript": {
    "formatter": {
      "semicolons": "always",
      "quoteStyle": "double"
    }
  }
}
```

```json
// package.json
{
  "scripts": {
    "lint": "biome lint src/",
    "format": "biome format --write src/",
    "check": "biome check --write src/" // lint + format
  }
}
```

Biome の利点:

1. **圧倒的な速度** - Rust 実装で ESLint の 25-100 倍高速
2. **オールインワン** - lint + format が統合
3. **デフォルト設定が優秀** - 最小限の設定で使える
4. **JSON サポート** - package.json などもフォーマット可能

Biome の欠点:

1. **型情報ルールなし** - no-floating-promises 等が使えない
2. **プラグインなし** - React Hooks などのプラグインが使えない
3. **エコシステムが小さい** - カスタムルールが作りにくい

### 11-3. oxlint の特徴

oxlint は Oxc プロジェクトの一部で、超高速なリンターです。

```bash
# インストール
npm install -D oxlint
```

```json
// .oxlintrc.json
{
  "rules": {
    "no-floating-promises": "warn",
    "no-explicit-any": "error"
  }
}
```

```json
// package.json
{
  "scripts": {
    "lint": "oxlint src/"
  }
}
```

oxlint の利点:

1. **最速** - Rust 実装で最も高速
2. **ESLint 互換** - ESLint のルールを多数実装
3. **型情報なしで動作** - 高速

oxlint の欠点:

1. **型情報ルールなし** - typescript-eslint の主要機能が使えない
2. **開発中** - まだ安定版ではない
3. **プラグインなし** - 拡張性が低い

### 11-4. 移行判断フローチャート

```
ESLint から移行すべきか？

┌─────────────────────────────┐
│ 型情報ルールが必要？          │
│ (no-floating-promises 等)   │
└─────────────────────────────┘
          │
    ┌─────┴─────┐
    │           │
   Yes         No
    │           │
    v           v
┌─────┐   ┌────────────┐
│ESLint│   │ Biome/oxlint│
│継続  │   │ 移行検討    │
└─────┘   └────────────┘
              │
              v
    ┌──────────────────┐
    │ React Hooks 等の │
    │ プラグインが必要？│
    └──────────────────┘
              │
        ┌─────┴─────┐
        │           │
       Yes         No
        │           │
        v           v
    ┌─────┐   ┌──────┐
    │ESLint│   │ Biome│
    │継続  │   │ 推奨 │
    └─────┘   └──────┘
```

### 11-5. ハイブリッド構成

型情報ルールのみ ESLint、それ以外は Biome を使う構成も可能です。

```json
// package.json
{
  "scripts": {
    "lint:biome": "biome check --write src/",
    "lint:eslint": "eslint src/ --cache",
    "lint": "npm run lint:biome && npm run lint:eslint"
  }
}
```

```typescript
// eslint.config.ts (型情報ルールのみ)
import tseslint from "typescript-eslint";

export default tseslint.config({
  languageOptions: {
    parserOptions: {
      projectService: true,
      tsconfigRootDir: import.meta.dirname,
    },
  },
  rules: {
    // 型情報ルールのみ有効化
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/no-misused-promises": "error",
    "@typescript-eslint/await-thenable": "error",
    "@typescript-eslint/no-unnecessary-condition": "error",

    // Biome でカバーされるルールは無効化
    "@typescript-eslint/no-unused-vars": "off",
    "@typescript-eslint/no-explicit-any": "off",
  },
});
```

```json
// biome.json
{
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "suspicious": {
        "noExplicitAny": "error"
      }
    }
  },
  "formatter": {
    "enabled": true
  }
}
```

### 11-6. 移行事例

```
プロジェクトA (Next.js アプリ)
  Before: ESLint + Prettier
    └─ lint 時間: 45 秒
  After: Biome
    └─ check 時間: 2 秒 (22.5倍高速)
  トレードオフ: React Hooks の exhaustive-deps が使えなくなった

プロジェクトB (Node.js API)
  Before: ESLint + typescript-eslint
    └─ lint 時間: 28 秒
  After: ESLint (projectService) + Biome
    └─ lint 時間: 8 秒 (3.5倍高速)
  トレードオフ: なし (型情報ルールは ESLint で継続)

プロジェクトC (ライブラリ)
  Before: ESLint + typescript-eslint (strict)
    └─ lint 時間: 12 秒
  After: ESLint のまま
  理由: 型情報ルールが必須、移行メリット小
```

---

## 12. 演習問題

### 演習 1: 基礎レベル

**課題**: 新規 TypeScript プロジェクトに ESLint を導入してください。

要件:
1. typescript-eslint をインストール
2. Flat Config 形式で設定ファイルを作成
3. recommendedTypeChecked を使用
4. テストファイルでは any を許可

<details>
<summary>解答例</summary>

```bash
# 1. インストール
npm install -D eslint typescript-eslint

# 2. 設定ファイル作成
touch eslint.config.ts
```

```typescript
// eslint.config.ts
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ["**/*.test.ts", "**/*.spec.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
    },
  },
  {
    ignores: ["dist/", "node_modules/"],
  }
);
```

```json
// package.json
{
  "scripts": {
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix"
  }
}
```

</details>

### 演習 2: 応用レベル

**課題**: React プロジェクトに ESLint + Prettier を導入し、VS Code と統合してください。

要件:
1. ESLint + typescript-eslint + React プラグイン
2. Prettier との統合
3. VS Code で保存時に自動修正
4. Git フック (husky + lint-staged)

<details>
<summary>解答例</summary>

```bash
# 1. パッケージインストール
npm install -D eslint typescript-eslint \
  eslint-plugin-react-hooks eslint-plugin-react-refresh \
  eslint-config-prettier prettier \
  husky lint-staged

# 2. husky セットアップ
npx husky init
```

```typescript
// eslint.config.ts
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import prettierConfig from "eslint-config-prettier";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  },
  {
    files: ["**/*.{jsx,tsx}"],
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  },
  prettierConfig,
  {
    ignores: ["dist/", "node_modules/"],
  }
);
```

```json
// .prettierrc
{
  "semi": true,
  "singleQuote": false,
  "tabWidth": 2,
  "trailingComma": "all",
  "printWidth": 100
}
```

```json
// .vscode/settings.json
{
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  }
}
```

```json
// package.json
{
  "scripts": {
    "prepare": "husky",
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",
    "format": "prettier --write 'src/**/*.{ts,tsx}'"
  },
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix",
      "prettier --write"
    ]
  }
}
```

```bash
# .husky/pre-commit
npm run lint-staged
```

</details>

### 演習 3: 発展レベル

**課題**: カスタムルール `no-unhandled-fetch` を作成してください。

要件:
1. fetch の呼び出しに try-catch がない場合に警告
2. 型情報を使って Response 型をチェック
3. 自動修正機能は不要

<details>
<summary>解答例</summary>

```typescript
// rules/no-unhandled-fetch.ts
import { ESLintUtils } from "@typescript-eslint/utils";
import type { TSESTree } from "@typescript-eslint/utils";

const createRule = ESLintUtils.RuleCreator(
  (name) => `https://example.com/rule/${name}`
);

export const noUnhandledFetch = createRule({
  name: "no-unhandled-fetch",
  meta: {
    type: "problem",
    docs: {
      description: "fetch は try-catch でエラーハンドリングが必要",
    },
    messages: {
      noTryCatch: "fetch は try-catch ブロック内で呼び出してください",
    },
    schema: [],
  },
  defaultOptions: [],
  create(context) {
    function isInTryCatch(node: TSESTree.Node): boolean {
      let current = node.parent;
      while (current) {
        if (current.type === "TryStatement") {
          return true;
        }
        current = current.parent;
      }
      return false;
    }

    return {
      CallExpression(node: TSESTree.CallExpression) {
        // fetch 呼び出しをチェック
        if (
          node.callee.type === "Identifier" &&
          node.callee.name === "fetch"
        ) {
          // try-catch 内かチェック
          if (!isInTryCatch(node)) {
            context.report({
              node,
              messageId: "noTryCatch",
            });
          }
        }
      },
    };
  },
});
```

```typescript
// index.ts
import { noUnhandledFetch } from "./rules/no-unhandled-fetch";

export default {
  rules: {
    "no-unhandled-fetch": noUnhandledFetch,
  },
};
```

```typescript
// eslint.config.ts
import myPlugin from "./my-eslint-plugin";

export default [
  {
    plugins: {
      "my-plugin": myPlugin,
    },
    rules: {
      "my-plugin/no-unhandled-fetch": "error",
    },
  },
];
```

```typescript
// テスト
// NG: try-catch なし
async function getData() {
  const response = await fetch("/api/data"); // 警告
}

// OK: try-catch あり
async function getData() {
  try {
    const response = await fetch("/api/data");
    return await response.json();
  } catch (error) {
    console.error("Fetch error:", error);
  }
}
```

</details>

---

## 13. エッジケース分析

### エッジケース 1: 動的 import と型情報

```typescript
// 問題: 動的 import の型情報が取得できない

// NG: 型情報ルールが動的 import を解析できない
async function loadModule(name: string) {
  const module = await import(`./modules/${name}`);
  // no-unsafe-member-access が誤検知する可能性
  return module.default;
}

// 解決策1: 明示的な型アサーション
interface Module {
  default: SomeType;
}

async function loadModule(name: string) {
  const module = await import(`./modules/${name}`) as Module;
  return module.default;
}

// 解決策2: ルールを部分的に無効化
async function loadModule(name: string) {
  // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
  const module = await import(`./modules/${name}`);
  return module.default;
}
```

### エッジケース 2: ジェネリック関数と型推論

```typescript
// 問題: ジェネリック関数で型情報が不完全

// NG: no-unnecessary-condition が誤検知
function process<T>(value: T | null): T {
  if (value === null) { // 警告が出る場合がある
    throw new Error("Value is null");
  }
  return value;
}

// 解決策: 型ガードを使う
function process<T>(value: T | null): T {
  if (!isNotNull(value)) {
    throw new Error("Value is null");
  }
  return value;
}

function isNotNull<T>(value: T | null): value is T {
  return value !== null;
}
```

---

## 14. アンチパターン

### アンチパターン 1: eslint-disable の乱用

```typescript
// NG: 理由なく disable
// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
const data: any = response.body;

// NG: ファイル全体を disable
/* eslint-disable @typescript-eslint/no-explicit-any */

// OK: 理由を明記し、スコープを最小化
// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment -- TODO: #456 で型定義を追加
const data: any = response.body;

// さらに OK: 根本的に修正
import { z } from "zod";
const ResponseSchema = z.object({ /* ... */ });
const data = ResponseSchema.parse(response.body);
```

### アンチパターン 2: 型情報ルールを全てオフ

```typescript
// NG: パフォーマンスを理由に型情報ルールを全てオフ
{
  rules: {
    "@typescript-eslint/no-floating-promises": "off",
    "@typescript-eslint/no-misused-promises": "off",
    "@typescript-eslint/await-thenable": "off",
  }
}

// OK: projectService で高速化し、ルールを有効化
{
  languageOptions: {
    parserOptions: {
      projectService: true, // 高速化
      tsconfigRootDir: import.meta.dirname,
    },
  },
  rules: {
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/no-misused-promises": "error",
  }
}
```

---

## 15. FAQ

### Q1: Biome に移行すべきですか？

**A**: プロジェクトの要件によります。

- **Biome が適している場合**:
  - 型情報ルール（no-floating-promises 等）が不要
  - React Hooks などのプラグインが不要
  - lint + format の速度が最優先
  - 新規プロジェクト

- **ESLint が適している場合**:
  - 型情報ルールが必須（非同期処理が多い）
  - React Hooks exhaustive-deps が必要
  - カスタムルールやプラグインを多用
  - 既存の大規模プロジェクト

**推奨**: 型情報ルールが必要なら ESLint、不要なら Biome を検討してください。

### Q2: Flat Config と Legacy Config のどちらを使うべきですか？

**A**: Flat Config (`eslint.config.ts`) を使ってください。

理由:
1. ESLint v9 以降は Flat Config がデフォルト
2. Legacy Config (`.eslintrc.*`) は ESLint v10 で削除予定
3. typescript-eslint v8 も Flat Config を推奨
4. TypeScript で型安全に設定を記述できる
5. 設定の結合がシンプル（配列のスプレッド）

Legacy Config は既存プロジェクトの互換性のためのみ使用してください。

### Q3: CI での実行が遅い場合の対策は？

**A**: 以下の最適化を実施してください。

1. **projectService を使用** (v8 以降)
```typescript
{
  languageOptions: {
    parserOptions: {
      projectService: true,
      tsconfigRootDir: import.meta.dirname,
    },
  },
}
```

2. **キャッシュを有効化**
```json
{
  "scripts": {
    "lint": "eslint src/ --cache --cache-location .eslintcache"
  }
}
```

3. **変更ファイルのみ lint**
```bash
eslint $(git diff --name-only --diff-filter=d HEAD -- '*.ts' '*.tsx')
```

4. **並列実行**
```bash
npm install -D eslint-parallel
eslint-parallel src/**/*.ts
```

5. **CI キャッシュの活用**
```yaml
- uses: actions/cache@v4
  with:
    path: .eslintcache
    key: eslint-${{ hashFiles('**/eslint.config.ts') }}
```

これらの組み合わせで、実行時間を 80-90% 削減できます。

### Q4: モノレポでルールを共有する方法は？

**A**: ルート設定をエクスポートし、各パッケージでインポートします。

```typescript
// root/eslint.config.ts
export default tseslint.config(/* 共通ルール */);

// packages/web/eslint.config.ts
import rootConfig from "../../eslint.config.ts";

export default [
  ...rootConfig,
  { /* web 固有ルール */ }
];
```

詳細は「[モノレポでの設定共有](#8-モノレポでの設定共有)」を参照してください。

### Q5: カスタムルールを作成するには？

**A**: `@typescript-eslint/utils` の `ESLintUtils.RuleCreator` を使用します。

基本構造:

```typescript
import { ESLintUtils } from "@typescript-eslint/utils";

const createRule = ESLintUtils.RuleCreator(
  (name) => `https://example.com/rule/${name}`
);

export const myRule = createRule({
  name: "my-rule",
  meta: { /* メタデータ */ },
  defaultOptions: [],
  create(context) {
    return {
      // AST ノードに対するビジター
      Identifier(node) {
        // ルールロジック
      },
    };
  },
});
```

詳細は「[カスタムルールの作成](#5-カスタムルールの作成)」を参照してください。

### Q6: テストファイルで any を許可するには？

**A**: files パターンでテストファイルを指定し、ルールを無効化します。

```typescript
// eslint.config.ts
export default tseslint.config(
  // ...
  {
    files: ["**/*.test.ts", "**/*.spec.ts", "**/__tests__/**"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-unsafe-assignment": "off",
    },
  }
);
```

### Q7: switch の網羅性チェックを有効にするには？

**A**: `@typescript-eslint/switch-exhaustiveness-check` ルールを有効化します。

```typescript
{
  rules: {
    "@typescript-eslint/switch-exhaustiveness-check": "error"
  }
}
```

これにより、ユニオン型の全ケースを網羅していない switch 文で警告が出ます。

詳細は「[switch-exhaustiveness-check](#4-5-switch-exhaustiveness-check)」を参照してください。

---

## 16. まとめ表

| 概念 | 要点 |
|------|------|
| typescript-eslint | TypeScript 専用の ESLint パーサー + プラグイン |
| Flat Config | `eslint.config.ts` 形式、ESLint v9 以降の標準 |
| 型情報ルール | TypeScript Compiler の型チェッカーと連携した高度な検出 |
| projectService | 型情報ルールのパフォーマンスを大幅改善（v8 以降） |
| no-floating-promises | Promise の放置を検出 |
| no-misused-promises | Promise を期待していない場所での使用を検出 |
| switch-exhaustiveness-check | switch 文の網羅性をチェック |
| Prettier 連携 | eslint-config-prettier で競合ルールを無効化 |
| consistent-type-imports | `import type` の一貫した使用を強制 |
| Biome | ESLint + Prettier の高速代替（型情報ルールなし） |
| oxlint | 超高速リンター（型情報ルールなし、開発中） |
| カスタムルール | ESLintUtils.RuleCreator で独自ルールを作成可能 |

---

## 17. 次に読むべきガイド

- **[tsconfig.json](./00-tsconfig.md)** -- ESLint と連携する TypeScript コンパイラ設定
- **[テスト](./02-testing-typescript.md)** -- テストファイルの lint ルール設定
- **[ビルドツール](./01-build-tools.md)** -- ビルドパイプラインへの lint 統合
- **[型システム](../01-type-system/01-basic-types.md)** -- TypeScript の型システムの基礎

---

## 18. 参考文献

1. **typescript-eslint** -- The tooling that enables ESLint and Prettier to support TypeScript
   https://typescript-eslint.io/
   公式ドキュメント。ルール一覧、設定例、マイグレーションガイドが充実。

2. **ESLint Flat Config**
   https://eslint.org/docs/latest/use/configure/configuration-files
   ESLint v9 の新しい設定形式の公式ドキュメント。

3. **Biome** -- One toolchain for your web project
   https://biomejs.dev/
   ESLint + Prettier の高速代替ツール。Rust 実装で圧倒的な速度を誇る。

4. **Oxc** -- The JavaScript Oxidation Compiler
   https://oxc-project.github.io/
   次世代の JavaScript ツールチェーン。oxlint を含む。

5. **AST Explorer**
   https://astexplorer.net/
   TypeScript の AST 構造を可視化するツール。カスタムルール作成時に便利。

6. **typescript-eslint Performance**
   https://typescript-eslint.io/linting/troubleshooting/performance-troubleshooting
   パフォーマンスチューニングの公式ガイド。

7. **ESLint Rules Reference**
   https://eslint.org/docs/latest/rules/
   ESLint の組み込みルール一覧。

8. **Prettier Documentation**
   https://prettier.io/docs/en/
   Prettier の公式ドキュメント。設定オプションの詳細。

---

## 付録: チートシート

### 基本設定

```typescript
// 最小構成
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommended
);
```

### 型情報ルール有効化

```typescript
export default tseslint.config(
  ...tseslint.configs.recommendedTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  }
);
```

### よく使うルール

```typescript
{
  rules: {
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/no-misused-promises": "error",
    "@typescript-eslint/await-thenable": "error",
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/consistent-type-imports": ["error", { prefer: "type-imports" }],
    "@typescript-eslint/switch-exhaustiveness-check": "error",
  }
}
```

### package.json スクリプト

```json
{
  "scripts": {
    "lint": "eslint src/ --cache",
    "lint:fix": "eslint src/ --fix",
    "lint:timing": "TIMING=1 eslint src/",
    "check": "tsc --noEmit && eslint src/"
  }
}
```

### VS Code 設定

```json
{
  "eslint.useFlatConfig": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit"
  }
}
```

---

**文字数**: 約 42,000 字

このガイドは、ESLint + TypeScript の全体像から実践的な設定、パフォーマンス最適化、代替ツールとの比較まで、包括的にカバーしています。コード例、図解、比較表、演習問題を豊富に含み、MIT 級の品質を目指しました。