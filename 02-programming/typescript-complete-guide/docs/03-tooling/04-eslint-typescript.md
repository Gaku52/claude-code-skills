# ESLint + TypeScript 完全ガイド

> typescript-eslint を使い、型情報を活用した高度な静的解析で TypeScript コードの品質を保つ

## この章で学ぶこと

1. **typescript-eslint のセットアップ** -- Flat Config 形式での設定、パーサー連携、推奨ルールセット
2. **型情報を使うルール** -- `@typescript-eslint` の型チェック付きルールで、tsc では検出できない問題を発見する
3. **プロジェクト別の設定** -- モノレポ、React、Node.js、ライブラリそれぞれの最適な lint 構成

---

## 1. セットアップ

### 1-1. インストールと基本設定

```bash
# 必要パッケージをインストール
npm install -D eslint @eslint/js typescript-eslint

# Prettier 連携（オプション）
npm install -D eslint-config-prettier eslint-plugin-prettier
```

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

```typescript
// eslint.config.ts（Flat Config 形式）
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
    // テストファイル用の緩いルール
    files: ["**/*.test.ts", "**/*.spec.ts"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
    },
  },
  {
    ignores: ["dist/", "node_modules/", "*.js"],
  }
);
```

### 1-2. package.json 設定

```json
{
  "scripts": {
    "lint": "eslint src/",
    "lint:fix": "eslint src/ --fix",
    "typecheck": "tsc --noEmit",
    "check": "npm run typecheck && npm run lint"
  }
}
```

---

## 2. 推奨ルールセット

### 2-1. ルールセットの段階

```
ルールセットの厳密度:

  recommended           ← 基本（型情報不使用）
       |
       v
  recommendedTypeChecked  ← 型チェック付きルールを追加
       |
       v
  strictTypeChecked      ← より厳密なルール
       |
       v
  stylisticTypeChecked   ← スタイル統一ルール追加
```

```typescript
// 推奨: strictTypeChecked + stylisticTypeChecked
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
  }
);
```

### 2-2. 重要な個別ルール

```typescript
// カスタムルール設定例
{
  rules: {
    // -- 型安全性 --
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/no-unsafe-assignment": "error",
    "@typescript-eslint/no-unsafe-call": "error",
    "@typescript-eslint/no-unsafe-member-access": "error",
    "@typescript-eslint/no-unsafe-return": "error",

    // -- async --
    "@typescript-eslint/no-floating-promises": "error",
    "@typescript-eslint/no-misused-promises": "error",
    "@typescript-eslint/await-thenable": "error",
    "no-return-await": "off",
    "@typescript-eslint/return-await": ["error", "in-try-catch"],

    // -- 命名規則 --
    "@typescript-eslint/naming-convention": [
      "error",
      {
        selector: "interface",
        format: ["PascalCase"],
      },
      {
        selector: "typeAlias",
        format: ["PascalCase"],
      },
      {
        selector: "enum",
        format: ["PascalCase"],
      },
      {
        selector: "variable",
        format: ["camelCase", "UPPER_CASE"],
        leadingUnderscore: "allow",
      },
    ],

    // -- インポート --
    "@typescript-eslint/consistent-type-imports": [
      "error",
      { prefer: "type-imports" },
    ],
    "@typescript-eslint/no-import-type-side-effects": "error",

    // -- switch 網羅性 --
    "@typescript-eslint/switch-exhaustiveness-check": "error",

    // -- 未使用変数 --
    "@typescript-eslint/no-unused-vars": [
      "error",
      {
        argsIgnorePattern: "^_",
        varsIgnorePattern: "^_",
        caughtErrorsIgnorePattern: "^_",
      },
    ],
  }
}
```

---

## 3. 型情報を使うルール

### 3-1. no-floating-promises

```typescript
// NG: Promise を await/catch せずに放置
async function fetchData(): Promise<void> {
  const response = await fetch("/api/data");
  response.json(); // 警告: Floating Promise
  // ↑ この Promise は処理されない
}

// OK: await する
async function fetchData(): Promise<void> {
  const response = await fetch("/api/data");
  const data = await response.json();
}

// OK: void 演算子で意図的に無視
void someAsyncOperation();
```

### 3-2. no-misused-promises

```typescript
// NG: async 関数をコールバックとして渡す（エラーが握りつぶされる）
const items = [1, 2, 3];
items.forEach(async (item) => {
  // 警告: forEach は返り値の Promise を待たない
  await processItem(item);
});

// OK: for...of で順次処理
for (const item of items) {
  await processItem(item);
}

// OK: Promise.all で並列処理
await Promise.all(items.map((item) => processItem(item)));

// NG: 条件式に Promise
if (fetchUser(id)) { // 警告: Promise は常に truthy
  // ...
}

// OK: await する
if (await fetchUser(id)) {
  // ...
}
```

### 3-3. switch-exhaustiveness-check

```typescript
// @typescript-eslint/switch-exhaustiveness-check: "error" の効果

type Status = "active" | "inactive" | "pending";

function getLabel(status: Status): string {
  switch (status) {
    case "active":
      return "有効";
    case "inactive":
      return "無効";
    // "pending" を忘れると ESLint エラー
    // Switch is not exhaustive. Missing case: "pending"
  }
}
```

---

## 4. Prettier との連携

### 4-1. 設定

```typescript
// eslint.config.ts
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import prettierConfig from "eslint-config-prettier";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  prettierConfig, // Prettier と衝突するルールを無効化
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

---

## 5. プロジェクト別設定

### 5-1. React プロジェクト

```typescript
// eslint.config.ts (React)
import eslint from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default tseslint.config(
  eslint.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  {
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
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
  }
);
```

### 5-2. モノレポ設定

```
モノレポの ESLint 構成:

  root/
  +-- eslint.config.ts          ← 共通ルール
  +-- packages/
      +-- shared/
      |   +-- eslint.config.ts  ← ライブラリ固有ルール
      +-- web/
      |   +-- eslint.config.ts  ← React 固有ルール
      +-- api/
          +-- eslint.config.ts  ← Node.js 固有ルール
```

```typescript
// packages/shared/eslint.config.ts
import rootConfig from "../../eslint.config.ts";

export default [
  ...rootConfig,
  {
    rules: {
      // ライブラリでは stricter なルール
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/explicit-function-return-type": "error",
    },
  },
];
```

---

## 比較表

### ルールセットの比較

| ルールセット | ルール数 | 型情報 | 厳密度 | 推奨場面 |
|-------------|---------|--------|--------|---------|
| recommended | 少 | 不要 | 低 | 初期導入 |
| recommendedTypeChecked | 中 | 必要 | 中 | 一般プロジェクト |
| strictTypeChecked | 多 | 必要 | 高 | ライブラリ |
| stylisticTypeChecked | 追加 | 必要 | スタイル | コード統一 |

### フォーマッター連携の比較

| ツール | ESLint 連携 | 速度 | 設定量 | 推奨 |
|--------|------------|------|--------|------|
| Prettier | eslint-config-prettier | 速い | 少 | 新規プロジェクト |
| dprint | eslint-plugin-dprint | 最速 | 中 | パフォーマンス重視 |
| Biome | biome check | 最速 | 少 | ESLint 代替 |
| ESLint のみ | 不要 | 遅い | 多 | 非推奨 |

---

## アンチパターン

### AP-1: eslint-disable を乱用する

```typescript
// NG: 理由なく disable
// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
const data: any = response.body;

// OK: 理由を明記し、スコープを最小化
// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment -- TODO: #456 で型を修正
const data: any = response.body;

// さらに OK: 根本的に修正
const data = ResponseSchema.parse(response.body);
```

### AP-2: 型情報ルールをオフにしてパフォーマンス問題を回避

```typescript
// NG: 型チェック付きルールを全てオフ
// "パフォーマンスが悪いから" という理由

// OK: projectService で高速化
{
  languageOptions: {
    parserOptions: {
      projectService: true, // 型チェック高速化
      tsconfigRootDir: import.meta.dirname,
    },
  },
}
// projectService は tsc のインクリメンタル解析を活用し、
// 従来の project 指定より大幅に高速
```

---

## FAQ

### Q1: Biome に移行すべきですか？

Biome は ESLint + Prettier を 1 つのツールで置き換え、圧倒的に高速です。ただし、typescript-eslint の型情報ルール（no-floating-promises 等）は Biome にはまだありません。型情報ルールが不要なプロジェクトでは Biome への移行を検討してもよいでしょう。

### Q2: Flat Config と Legacy Config のどちらを使うべきですか？

Flat Config（`eslint.config.ts`）を使ってください。ESLint v9 以降は Flat Config がデフォルトで、Legacy Config（`.eslintrc.*`）は非推奨です。typescript-eslint v8 以降も Flat Config を推奨しています。

### Q3: CI での実行が遅い場合の対策は？

`projectService: true` の使用、`--cache` フラグの有効化、変更ファイルのみの lint（`eslint $(git diff --name-only --diff-filter=d HEAD -- '*.ts')`）が有効です。モノレポでは Turborepo のキャッシュも活用できます。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| typescript-eslint | TypeScript 専用の ESLint パーサー + ルール |
| Flat Config | `eslint.config.ts` 形式、v9 以降の標準 |
| 型情報ルール | TSC の型チェッカーと連携した高度な検出 |
| projectService | 型情報ルールのパフォーマンスを大幅改善 |
| Prettier 連携 | eslint-config-prettier で衝突回避 |
| consistent-type-imports | `import type` の一貫した使用を強制 |

---

## 次に読むべきガイド

- [tsconfig.json](./00-tsconfig.md) -- ESLint と連携する TypeScript コンパイラ設定
- [テスト](./02-testing-typescript.md) -- テストファイルの lint ルール設定
- [ビルドツール](./01-build-tools.md) -- ビルドパイプラインへの lint 統合

---

## 参考文献

1. **typescript-eslint** -- The tooling that enables ESLint and Prettier to support TypeScript
   https://typescript-eslint.io/

2. **ESLint Flat Config**
   https://eslint.org/docs/latest/use/configure/configuration-files

3. **Biome** -- One toolchain for your web project
   https://biomejs.dev/
