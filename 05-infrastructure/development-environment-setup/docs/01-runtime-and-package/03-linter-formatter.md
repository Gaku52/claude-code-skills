# Linter / Formatter

> ESLint、Prettier、Biome、Ruff を活用したコード品質管理の実践ガイド。設定共有と CI 統合で、チーム全体のコードスタイルを統一する。

## この章で学ぶこと

1. ESLint (v9 Flat Config) と Prettier の正しい設定と連携
2. Biome (Rust 製高速ツール) と Ruff (Python) の導入方法
3. 設定共有パターンと CI / pre-commit フックの統合
4. Stylelint による CSS/SCSS のリンティング
5. エディタ連携と自動修正の最適化
6. モノレポでの設定共有パターンと大規模プロジェクトでの運用

---

## 1. ツール全体像

### 1.1 Linter vs Formatter の違い

```
Linter と Formatter の役割分担:

  ソースコード
      │
      ▼
  ┌──────────────┐     ┌──────────────┐
  │   Formatter   │     │    Linter     │
  │               │     │               │
  │ コードの       │     │ コードの       │
  │ "見た目" を    │     │ "品質" を      │
  │ 統一する       │     │ 検査する       │
  │               │     │               │
  │ 例:           │     │ 例:           │
  │ - インデント   │     │ - 未使用変数   │
  │ - 改行位置     │     │ - any 型使用   │
  │ - 引用符統一   │     │ - 安全でない   │
  │ - セミコロン   │     │   型変換       │
  │ - 括弧の位置   │     │ - 到達不能     │
  │ - 空白の調整   │     │   コード       │
  │               │     │ - セキュリティ  │
  │               │     │   脆弱性       │
  └──────────────┘     └──────────────┘
        │                      │
        ▼                      ▼
  自動修正可能            一部自動修正可能
  (100%)                (ルールによる)

  重要な原則:
  ┌─────────────────────────────────────────┐
  │ Formatter → 見た目の統一 (議論の余地なし)  │
  │ Linter   → 品質の担保 (ルール選択が重要)   │
  │                                           │
  │ 両者の責務を分離することで:                 │
  │ - 設定の競合を防止                         │
  │ - 実行速度の最適化                         │
  │ - メンテナンスの簡素化                      │
  └─────────────────────────────────────────┘
```

### 1.2 主要ツール比較

| ツール | 対象言語 | 種類 | 速度 | 設定形式 | エコシステム |
|--------|---------|------|------|---------|------------|
| ESLint | JS/TS | Linter | 普通 | Flat Config (JS) | 最大 (1000+ プラグイン) |
| Prettier | 多言語 | Formatter | 普通 | JSON/JS | 広い (プラグイン対応) |
| Biome | JS/TS/JSON/CSS | 両方 | 超高速 | JSON | 成長中 |
| Ruff | Python | 両方 | 超高速 | TOML | Python 特化 |
| Stylelint | CSS/SCSS | Linter | 普通 | JSON/JS | CSS 特化 |
| oxlint | JS/TS | Linter | 超高速 | JSON | ESLint 互換 (一部) |
| dprint | 多言語 | Formatter | 高速 | JSON | Rust 製・プラグイン対応 |

### 1.3 ツール選定フローチャート

```
プロジェクトに最適なツール選定:

  Q1: 言語は何か？
  │
  ├── JavaScript / TypeScript
  │   │
  │   └── Q2: プラグインの豊富さは重要？
  │       │
  │       ├── Yes → ESLint + Prettier (定番構成)
  │       │   - react-hooks, jsx-a11y 等が必要
  │       │   - 型チェック連携 (recommendedTypeChecked)
  │       │   - カスタムルールの作成
  │       │
  │       └── No → Biome (高速・シンプル)
  │           - 設定ファイル1つ
  │           - Linter + Formatter 統合
  │           - ESLint からの移行ツールあり
  │
  ├── Python
  │   └── Ruff (デファクト)
  │       - Flake8 + isort + Black + pyupgrade を統合
  │       - 10-100倍高速
  │
  ├── CSS / SCSS
  │   └── Stylelint + Prettier
  │
  └── Go / Rust / その他
      └── 各言語の公式ツール
          - Go: gofmt + golangci-lint
          - Rust: rustfmt + clippy
```

---

## 2. ESLint (v9 Flat Config)

### 2.1 セットアップ

```bash
# インストール
pnpm add -D eslint @eslint/js typescript-eslint globals

# 型チェック統合が必要な場合
pnpm add -D @typescript-eslint/parser

# React プロジェクトの場合
pnpm add -D eslint-plugin-react eslint-plugin-react-hooks eslint-plugin-jsx-a11y

# Next.js プロジェクトの場合
pnpm add -D @next/eslint-plugin-next

# インポート整理
pnpm add -D eslint-plugin-import eslint-plugin-unused-imports

# Prettier との競合回避
pnpm add -D eslint-config-prettier
```

### 2.2 設定ファイル (基本)

```javascript
// eslint.config.js (Flat Config 形式 -- v9 推奨)
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";

export default tseslint.config(
  // グローバル無視
  {
    ignores: [
      "dist/",
      "build/",
      "node_modules/",
      "coverage/",
      ".next/",
      "*.config.js",
      "*.config.mjs",
      "*.config.cjs",
    ],
  },

  // JavaScript 推奨ルール
  js.configs.recommended,

  // TypeScript 推奨ルール
  ...tseslint.configs.recommendedTypeChecked,

  // プロジェクト共通設定
  {
    languageOptions: {
      ecmaVersion: 2024,
      sourceType: "module",
      globals: {
        ...globals.browser,
        ...globals.node,
      },
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // ─── 型安全性 ───
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-unsafe-assignment": "error",
      "@typescript-eslint/no-unsafe-call": "error",
      "@typescript-eslint/no-unsafe-return": "error",
      "@typescript-eslint/no-unsafe-member-access": "error",
      "@typescript-eslint/no-unsafe-argument": "error",
      "@typescript-eslint/prefer-as-const": "error",
      "@typescript-eslint/no-non-null-assertion": "warn",
      "@typescript-eslint/consistent-type-imports": ["error", {
        prefer: "type-imports",
        fixStyle: "inline-type-imports",
      }],
      "@typescript-eslint/consistent-type-exports": "error",

      // ─── コード品質 ───
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "prefer-const": "error",
      "no-var": "error",
      eqeqeq: ["error", "always"],
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
      curly: ["error", "all"],
      "no-throw-literal": "error",

      // ─── Promise / Async ───
      "@typescript-eslint/no-floating-promises": "error",
      "@typescript-eslint/no-misused-promises": "error",
      "@typescript-eslint/require-await": "warn",
      "no-return-await": "off",
      "@typescript-eslint/return-await": ["error", "in-try-catch"],

      // ─── 命名規則 ───
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
          selector: "enumMember",
          format: ["UPPER_CASE"],
        },
      ],
    },
  },

  // テストファイル用の緩和ルール
  {
    files: ["**/*.test.ts", "**/*.spec.ts", "**/__tests__/**"],
    rules: {
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-non-null-assertion": "off",
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "no-console": "off",
    },
  },

  // 設定ファイル用
  {
    files: ["*.config.ts", "*.config.js"],
    rules: {
      "no-console": "off",
      "@typescript-eslint/no-require-imports": "off",
    },
  }
);
```

### 2.3 React / Next.js 用設定

```javascript
// eslint.config.js (React + Next.js)
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";
import reactPlugin from "eslint-plugin-react";
import reactHooksPlugin from "eslint-plugin-react-hooks";
import jsxA11y from "eslint-plugin-jsx-a11y";
import nextPlugin from "@next/eslint-plugin-next";
import prettierConfig from "eslint-config-prettier";

export default tseslint.config(
  {
    ignores: ["dist/", "node_modules/", ".next/", "coverage/"],
  },

  js.configs.recommended,
  ...tseslint.configs.recommended,

  // React 設定
  {
    files: ["**/*.tsx", "**/*.jsx"],
    plugins: {
      react: reactPlugin,
      "react-hooks": reactHooksPlugin,
      "jsx-a11y": jsxA11y,
    },
    languageOptions: {
      globals: {
        ...globals.browser,
      },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    settings: {
      react: { version: "detect" },
    },
    rules: {
      // React
      "react/prop-types": "off",
      "react/react-in-jsx-scope": "off",
      "react/self-closing-comp": "error",
      "react/jsx-no-target-blank": "error",
      "react/jsx-boolean-value": ["error", "never"],
      "react/jsx-curly-brace-presence": ["error", {
        props: "never",
        children: "never",
      }],
      "react/no-array-index-key": "warn",
      "react/no-unstable-nested-components": "error",

      // React Hooks
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // アクセシビリティ
      "jsx-a11y/alt-text": "error",
      "jsx-a11y/anchor-is-valid": "error",
      "jsx-a11y/click-events-have-key-events": "warn",
      "jsx-a11y/no-static-element-interactions": "warn",
      "jsx-a11y/heading-has-content": "error",
      "jsx-a11y/label-has-associated-control": "error",
    },
  },

  // Next.js 固有ルール
  {
    plugins: { "@next/next": nextPlugin },
    rules: {
      ...nextPlugin.configs.recommended.rules,
      ...nextPlugin.configs["core-web-vitals"].rules,
    },
  },

  // Prettier と競合するルールを無効化 (必ず最後)
  prettierConfig,
);
```

### 2.4 ESLint v8 から v9 への移行

```
ESLint v8 (Legacy) → v9 (Flat Config) の主要な変更点:

  設定ファイル:
  ┌────────────────────────────────────────────┐
  │ v8: .eslintrc.json / .eslintrc.js          │
  │ v9: eslint.config.js / eslint.config.mjs   │
  └────────────────────────────────────────────┘

  プラグインの指定方法:
  ┌────────────────────────────────────────────┐
  │ v8:                                        │
  │   "plugins": ["@typescript-eslint"]        │
  │   "extends": ["plugin:@typescript-eslint/  │
  │                recommended"]               │
  │                                            │
  │ v9:                                        │
  │   import tseslint from "typescript-eslint" │
  │   export default tseslint.config(          │
  │     ...tseslint.configs.recommended,       │
  │   )                                        │
  └────────────────────────────────────────────┘

  ignorePatterns → ignores:
  ┌────────────────────────────────────────────┐
  │ v8: "ignorePatterns": ["dist/"]            │
  │ v9: { ignores: ["dist/"] }                │
  │     (.eslintignore は不要)                 │
  └────────────────────────────────────────────┘

  env → globals:
  ┌────────────────────────────────────────────┐
  │ v8: "env": { "browser": true, "node": true } │
  │ v9: languageOptions: {                     │
  │       globals: {                           │
  │         ...globals.browser,                │
  │         ...globals.node,                   │
  │       }                                    │
  │     }                                      │
  └────────────────────────────────────────────┘

  移行コマンド:
  npx @eslint/migrate-config .eslintrc.json
  → eslint.config.mjs が自動生成される
```

### 2.5 実行コマンド

```bash
# リント実行
pnpm eslint .

# 自動修正
pnpm eslint --fix .

# 特定ファイル
pnpm eslint src/utils/validate.ts

# キャッシュを使って高速化
pnpm eslint --cache .
pnpm eslint --cache --cache-location .eslintcache .

# デバッグ (どのルールが適用されているか確認)
pnpm eslint --print-config src/index.ts
pnpm eslint --debug src/index.ts

# package.json に scripts を追加
# {
#   "scripts": {
#     "lint": "eslint .",
#     "lint:fix": "eslint --fix .",
#     "lint:cache": "eslint --cache ."
#   }
# }
```

### 2.6 カスタムルールの作成

```javascript
// eslint-rules/no-hardcoded-credentials.js
/** @type {import('eslint').Rule.RuleModule} */
export default {
  meta: {
    type: "problem",
    docs: {
      description: "ハードコードされた認証情報を禁止する",
    },
    messages: {
      hardcodedCredential: "認証情報をハードコードしないでください。環境変数を使用してください。",
    },
    schema: [],
  },
  create(context) {
    const suspiciousPatterns = [
      /password\s*[:=]\s*['"][^'"]+['"]/i,
      /api[_-]?key\s*[:=]\s*['"][^'"]+['"]/i,
      /secret\s*[:=]\s*['"][^'"]+['"]/i,
      /token\s*[:=]\s*['"][^'"]+['"]/i,
    ];

    return {
      Literal(node) {
        if (typeof node.value === "string") {
          for (const pattern of suspiciousPatterns) {
            if (pattern.test(`${context.getSourceCode().getText(node.parent)}`)) {
              context.report({
                node,
                messageId: "hardcodedCredential",
              });
            }
          }
        }
      },
    };
  },
};
```

---

## 3. Prettier

### 3.1 セットアップ

```bash
# インストール
pnpm add -D prettier

# ESLint との競合回避
pnpm add -D eslint-config-prettier

# プラグイン
pnpm add -D prettier-plugin-tailwindcss    # Tailwind クラスソート
pnpm add -D prettier-plugin-organize-imports  # インポートソート
pnpm add -D @ianvs/prettier-plugin-sort-imports  # インポートソート (高機能)
pnpm add -D prettier-plugin-prisma          # Prisma スキーマ
pnpm add -D prettier-plugin-packagejson     # package.json ソート
```

### 3.2 設定ファイル

```jsonc
// .prettierrc
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
  "bracketSameLine": false,
  "singleAttributePerLine": false,
  "htmlWhitespaceSensitivity": "css",
  "proseWrap": "preserve",
  "plugins": [
    "prettier-plugin-tailwindcss",
    "prettier-plugin-packagejson"
  ],
  "overrides": [
    {
      "files": "*.md",
      "options": {
        "printWidth": 100,
        "proseWrap": "always"
      }
    },
    {
      "files": "*.json",
      "options": {
        "trailingComma": "none"
      }
    },
    {
      "files": ["*.yml", "*.yaml"],
      "options": {
        "tabWidth": 2,
        "singleQuote": false
      }
    }
  ]
}
```

```bash
# .prettierignore
dist/
build/
node_modules/
coverage/
.next/
.turbo/
pnpm-lock.yaml
package-lock.json
yarn.lock
*.min.js
*.min.css
```

### 3.3 ESLint + Prettier の連携

```javascript
// eslint.config.js に追加
import prettierConfig from "eslint-config-prettier";

export default tseslint.config(
  // ...既存の設定...

  // Prettier と競合するルールを無効化 (必ず最後に配置)
  prettierConfig,
);
```

```
ESLint + Prettier の役割分担:

  ┌─────────────────────────────────────┐
  │         Prettier (Formatter)         │
  │  インデント、改行、引用符、セミコロン  │
  │  → 見た目に関する全てを担当           │
  └──────────────┬──────────────────────┘
                 │
                 │  eslint-config-prettier
                 │  (Prettier と競合する
                 │   ESLint ルールを OFF)
                 │
  ┌──────────────┴──────────────────────┐
  │          ESLint (Linter)             │
  │  型安全性、未使用変数、パターン検出    │
  │  → コード品質に関する検査を担当        │
  └─────────────────────────────────────┘

  実行順序 (推奨):
  1. ESLint --fix (自動修正可能なルールを適用)
  2. Prettier --write (フォーマットを統一)

  lint-staged での設定例:
  "*.{ts,tsx}": ["eslint --fix", "prettier --write"]
```

### 3.4 Prettier の主要オプション解説

```
よく議論になるオプションとその推奨値:

┌─────────────────────┬──────────────┬─────────────────────────┐
│ オプション            │ 推奨値       │ 理由                    │
├─────────────────────┼──────────────┼─────────────────────────┤
│ semi                │ true         │ ASI の罠を避ける        │
│ singleQuote         │ true         │ タイプ数削減             │
│ trailingComma       │ "all"        │ diff がクリーン          │
│ printWidth          │ 80           │ 分割画面で読みやすい     │
│ tabWidth            │ 2            │ JS/TS の慣習            │
│ arrowParens         │ "always"     │ 型注釈追加時に楽        │
│ endOfLine           │ "lf"         │ OS 間の差異を排除       │
│ bracketSameLine     │ false        │ 可読性重視              │
└─────────────────────┴──────────────┴─────────────────────────┘

※ これらは「決め」の問題。チームで合意した値を使い、議論を終わらせる
※ Prettier の哲学: 「オプションは少なく、議論を減らす」
```

---

## 4. Biome (高速オールインワン)

### 4.1 セットアップ

```bash
# インストール
pnpm add -D @biomejs/biome

# 初期設定
pnpm biome init

# ESLint / Prettier からの移行
pnpm biome migrate eslint --write
pnpm biome migrate prettier --write
```

### 4.2 設定ファイル

```jsonc
// biome.json
{
  "$schema": "https://biomejs.dev/schemas/1.9.0/schema.json",
  "organizeImports": {
    "enabled": true
  },
  "formatter": {
    "enabled": true,
    "indentStyle": "space",
    "indentWidth": 2,
    "lineWidth": 80,
    "lineEnding": "lf"
  },
  "linter": {
    "enabled": true,
    "rules": {
      "recommended": true,
      "complexity": {
        "noBannedTypes": "error",
        "noExcessiveCognitiveComplexity": {
          "level": "warn",
          "options": { "maxAllowedComplexity": 15 }
        },
        "noForEach": "warn",
        "useSimplifiedLogicExpression": "warn"
      },
      "correctness": {
        "noUnusedVariables": "error",
        "noUnusedImports": "error",
        "useExhaustiveDependencies": "warn",
        "noConstAssign": "error",
        "noUndeclaredVariables": "error"
      },
      "style": {
        "noNonNullAssertion": "warn",
        "useConst": "error",
        "useTemplate": "error",
        "useBlockStatements": "error",
        "noParameterAssign": "error",
        "useDefaultParameterLast": "error"
      },
      "suspicious": {
        "noExplicitAny": "error",
        "noDoubleEquals": "error",
        "noConfusingVoidType": "error",
        "noArrayIndexKey": "warn",
        "noConsoleLog": "warn"
      },
      "security": {
        "noDangerouslySetInnerHtml": "error"
      },
      "a11y": {
        "noBlankTarget": "error",
        "useAltText": "error",
        "useValidAnchor": "error",
        "useKeyWithClickEvents": "warn"
      }
    }
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "single",
      "semicolons": "always",
      "trailingCommas": "all",
      "arrowParentheses": "always"
    },
    "parser": {
      "unsafeParameterDecoratorsEnabled": true
    }
  },
  "json": {
    "formatter": {
      "trailingCommas": "none"
    }
  },
  "css": {
    "formatter": {
      "indentStyle": "space",
      "indentWidth": 2
    },
    "linter": {
      "enabled": true
    }
  },
  "files": {
    "ignore": [
      "dist/",
      "build/",
      "node_modules/",
      ".next/",
      "coverage/",
      "*.min.js",
      "*.min.css"
    ],
    "maxSize": 1048576
  },
  "overrides": [
    {
      "include": ["**/*.test.ts", "**/*.spec.ts", "**/__tests__/**"],
      "linter": {
        "rules": {
          "suspicious": {
            "noExplicitAny": "off",
            "noConsoleLog": "off"
          }
        }
      }
    }
  ]
}
```

### 4.3 Biome コマンド

```bash
# ─── リント ───
pnpm biome lint .
pnpm biome lint --write .          # 自動修正

# ─── フォーマット ───
pnpm biome format .
pnpm biome format --write .        # フォーマット適用

# ─── チェックのみ (CI 用) ───
pnpm biome check .                 # lint + format を同時チェック
pnpm biome ci .                    # CI モード (エラー時に非ゼロ終了)

# ─── 全自動修正 ───
pnpm biome check --write .        # lint fix + format を同時適用

# ─── インポートのソート ───
pnpm biome check --write --organize-imports-enabled=true .

# ─── 特定ファイル ───
pnpm biome lint src/utils/validate.ts
pnpm biome format src/components/Button.tsx
```

### 4.4 ESLint + Prettier vs Biome 比較

| 観点 | ESLint + Prettier | Biome |
|------|-------------------|-------|
| 速度 | 1x (基準) | 20-100x |
| 設定ファイル数 | 2-3個 | 1個 |
| プラグイン | 豊富 (1000+) | 限定的 |
| TypeScript対応 | 型チェック連携可 | 構文ベースのみ |
| CSS 対応 | Stylelint 別途 | 組み込み |
| JSON 対応 | 限定的 | 組み込み (フォーマット+リント) |
| インポートソート | eslint-plugin-import | 組み込み |
| エコシステム成熟度 | 非常に高い | 成長中 |
| 移行コスト | - | `biome migrate` で自動化 |
| 推奨 | 大規模・カスタム | 高速・シンプル |
| メモリ使用量 | 多い (Node.js) | 少ない (Rust native) |
| VS Code 拡張 | 各ツール別 | 1つで完結 |

### 4.5 ESLint から Biome への段階的移行

```bash
# Step 1: 移行分析
pnpm biome migrate eslint --include-inspired
# → どのルールが Biome に移行可能か表示

# Step 2: biome.json 生成
pnpm biome migrate eslint --write
pnpm biome migrate prettier --write

# Step 3: 並行運用期間
# - Biome で lint + format
# - ESLint は型チェック連携ルールのみ残す
# - CI で両方実行して結果を比較

# Step 4: ESLint の削除
pnpm remove eslint eslint-config-prettier @typescript-eslint/eslint-plugin \
  @typescript-eslint/parser eslint-plugin-import prettier
```

---

## 5. Ruff (Python)

### 5.1 セットアップ

```bash
# インストール
pip install ruff
# または
brew install ruff
# または uv
uv add --dev ruff
# または pipx
pipx install ruff
```

### 5.2 設定

```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"
line-length = 88
indent-width = 4
fix = true

# ソースディレクトリの指定
src = ["src", "tests"]

# 除外パターン
exclude = [
    ".git",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "*.egg-info",
    "migrations",
]

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort (インポートソート)
    "N",    # pep8-naming
    "UP",   # pyupgrade (古い構文の検出)
    "B",    # flake8-bugbear (バグ候補の検出)
    "SIM",  # flake8-simplify (簡略化可能なコード)
    "C4",   # flake8-comprehensions (内包表記の最適化)
    "DTZ",  # flake8-datetimez (タイムゾーン関連)
    "T20",  # flake8-print (print 文の検出)
    "RUF",  # Ruff固有ルール
    "ANN",  # flake8-annotations (型ヒント)
    "S",    # flake8-bandit (セキュリティ)
    "PT",   # flake8-pytest-style (pytest スタイル)
    "RET",  # flake8-return (return 文)
    "ARG",  # flake8-unused-arguments (未使用引数)
    "ERA",  # eradicate (コメントアウトされたコード)
    "PL",   # Pylint (一部ルール)
    "PERF", # Perflint (パフォーマンス)
    "FURB", # refurb (モダン Python)
]
ignore = [
    "E501",   # line too long (formatter に任せる)
    "ANN101", # self の型ヒント (不要)
    "ANN102", # cls の型ヒント (不要)
    "ANN401", # Any 型 (場合による)
]

# 自動修正可能なルール
fixable = ["ALL"]
unfixable = []

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["T20", "S101", "ANN"]  # テストでは print, assert, 型ヒント省略OK
"conftest.py" = ["ANN"]
"__init__.py" = ["F401"]  # 再エクスポートの未使用インポート

[tool.ruff.lint.isort]
known-first-party = ["myproject"]
force-single-line = false
lines-after-imports = 2

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
docstring-code-format = true
docstring-code-line-length = 72
```

### 5.3 実行コマンド

```bash
# ─── リント ───
ruff check .
ruff check --fix .              # 自動修正
ruff check --fix --unsafe-fixes . # 安全でない修正も含む

# ─── フォーマット ───
ruff format .
ruff format --check .           # チェックのみ (CI 用)
ruff format --diff .            # 差分表示

# ─── 特定ルールの確認 ───
ruff rule E501                  # ルールの詳細説明
ruff linter                     # 利用可能なルール一覧

# ─── 設定の確認 ───
ruff check --show-settings      # 現在の設定を表示
ruff check --statistics         # 違反統計

# Ruff は Flake8 + isort + Black + pyupgrade を1つで置き換える
# 速度は Flake8 の 10-100倍
```

### 5.4 mypy との併用

```toml
# pyproject.toml (mypy 設定)
[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_any_generics = true
check_untyped_defs = true

# ライブラリのスタブ
[[tool.mypy.overrides]]
module = ["redis.*", "celery.*"]
ignore_missing_imports = true
```

```
Ruff と mypy の役割分担:

  Ruff (高速):
  ├── スタイルチェック (PEP 8)
  ├── バグ候補検出 (Bugbear)
  ├── セキュリティチェック (Bandit)
  ├── インポートソート (isort)
  ├── コードフォーマット (Black 互換)
  └── モダン構文への変換 (pyupgrade)

  mypy (型チェック):
  ├── 型整合性の検証
  ├── 型ガードの検証
  ├── ジェネリクスの検証
  └── None チェックの検証

  実行順序:
  1. ruff check --fix .  (高速: 数ミリ秒)
  2. ruff format .       (高速: 数ミリ秒)
  3. mypy .              (低速: 数秒-数分)
```

---

## 6. Stylelint (CSS / SCSS)

### 6.1 セットアップ

```bash
# インストール
pnpm add -D stylelint stylelint-config-standard

# SCSS の場合
pnpm add -D stylelint-config-standard-scss

# CSS-in-JS の場合
pnpm add -D postcss-styled-syntax

# Prettier との連携
pnpm add -D stylelint-config-prettier-scss

# プロパティ順序
pnpm add -D stylelint-order stylelint-config-recess-order
```

### 6.2 設定ファイル

```jsonc
// .stylelintrc.json
{
  "extends": [
    "stylelint-config-standard-scss",
    "stylelint-config-recess-order",
    "stylelint-config-prettier-scss"
  ],
  "plugins": [
    "stylelint-order"
  ],
  "rules": {
    "color-named": "never",
    "color-hex-length": "short",
    "declaration-no-important": true,
    "selector-max-id": 0,
    "selector-max-specificity": "0,3,3",
    "max-nesting-depth": 3,
    "no-descending-specificity": true,
    "font-family-name-quotes": "always-where-recommended",
    "scss/dollar-variable-pattern": "^[a-z][a-z0-9-]*$",
    "scss/at-mixin-pattern": "^[a-z][a-z0-9-]*$",
    "selector-class-pattern": [
      "^[a-z][a-z0-9]*(-[a-z0-9]+)*$",
      { "message": "BEM パターンを使用してください" }
    ]
  },
  "ignoreFiles": [
    "dist/**",
    "node_modules/**",
    "coverage/**"
  ]
}
```

### 6.3 Tailwind CSS プロジェクトでの注意

```jsonc
// .stylelintrc.json (Tailwind 対応)
{
  "extends": ["stylelint-config-standard"],
  "rules": {
    "at-rule-no-unknown": [true, {
      "ignoreAtRules": [
        "tailwind",
        "apply",
        "layer",
        "config",
        "screen",
        "variants",
        "responsive"
      ]
    }],
    "function-no-unknown": [true, {
      "ignoreFunctions": ["theme", "screen"]
    }],
    "no-descending-specificity": null
  }
}
```

---

## 7. Pre-commit フック

### 7.1 lint-staged + husky

```bash
# インストール
pnpm add -D husky lint-staged

# husky 初期化
pnpm husky init
```

```jsonc
// package.json
{
  "lint-staged": {
    "*.{ts,tsx}": [
      "eslint --fix --cache",
      "prettier --write"
    ],
    "*.{js,jsx,mjs,cjs}": [
      "eslint --fix --cache",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ],
    "*.{css,scss}": [
      "stylelint --fix",
      "prettier --write"
    ],
    "*.py": [
      "ruff check --fix",
      "ruff format"
    ],
    "*.prisma": [
      "prettier --write"
    ],
    "package.json": [
      "prettier --write"
    ]
  }
}
```

```bash
# .husky/pre-commit
pnpm lint-staged
```

### 7.2 Biome を使った高速 lint-staged

```jsonc
// package.json (Biome 版)
{
  "lint-staged": {
    "*.{ts,tsx,js,jsx,json,css}": [
      "biome check --write --no-errors-on-unmatched"
    ],
    "*.{md,yml,yaml}": [
      "prettier --write"
    ],
    "*.py": [
      "ruff check --fix",
      "ruff format"
    ]
  }
}
```

### 7.3 pre-commit フローの動作

```
git commit 実行時のフロー:

  git commit -m "Add feature"
       │
       ▼
  ┌──────────────────────┐
  │  husky (pre-commit)   │
  │       │               │
  │       ▼               │
  │  lint-staged          │
  │  (ステージされた       │
  │   ファイルのみ対象)    │
  │       │               │
  │       ├── *.ts,*.tsx  │
  │       │   → eslint    │
  │       │   → prettier  │
  │       │               │
  │       ├── *.json,*.md │
  │       │   → prettier  │
  │       │               │
  │       ├── *.css,*.scss│
  │       │   → stylelint │
  │       │   → prettier  │
  │       │               │
  │       └── *.py        │
  │           → ruff      │
  │                       │
  │  全てパス？            │
  │  ├── Yes → コミット    │
  │  └── No  → コミット   │
  │           中止 + エラー │
  └──────────────────────┘

  ポイント:
  - ステージされたファイルのみが対象 (全ファイルではない)
  - --fix / --write で自動修正し、修正結果を再ステージ
  - CI では --check モードで確認するだけ (修正しない)
  - lint-staged v15+ はデフォルトで修正ファイルを再ステージ
```

### 7.4 lefthook (husky の代替)

```yaml
# lefthook.yml (husky + lint-staged の代替)
pre-commit:
  parallel: true
  commands:
    lint:
      glob: "*.{ts,tsx}"
      run: pnpm eslint --fix {staged_files} && pnpm prettier --write {staged_files}
      stage_fixed: true
    format-json:
      glob: "*.{json,md}"
      run: pnpm prettier --write {staged_files}
      stage_fixed: true
    python:
      glob: "*.py"
      run: ruff check --fix {staged_files} && ruff format {staged_files}
      stage_fixed: true
```

```bash
# lefthook のセットアップ
pnpm add -D lefthook
pnpm lefthook install
```

---

## 8. エディタ連携

### 8.1 VS Code 設定

```jsonc
// .vscode/settings.json
{
  // ─── デフォルトフォーマッター ───
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.formatOnSave": true,
  "editor.formatOnPaste": false,

  // ─── ESLint 連携 ───
  "eslint.enable": true,
  "eslint.useFlatConfig": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": "explicit",
    "source.organizeImports": "never"
  },

  // ─── 言語別フォーマッター ───
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[json]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[jsonc]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[css]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[scss]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "[prisma]": {
    "editor.defaultFormatter": "Prisma.prisma"
  },

  // ─── Stylelint 連携 ───
  "stylelint.validate": ["css", "scss"],
  "css.validate": false,
  "scss.validate": false,

  // ─── ファイル設定 ───
  "files.eol": "\n",
  "files.insertFinalNewline": true,
  "files.trimTrailingWhitespace": true,

  // ─── formatOnSave の最適化 ───
  "editor.formatOnSaveMode": "modificationsIfAvailable"
}
```

### 8.2 VS Code 推奨拡張機能

```jsonc
// .vscode/extensions.json
{
  "recommendations": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "stylelint.vscode-stylelint",
    "charliermarsh.ruff",
    "bradlc.vscode-tailwindcss",
    "EditorConfig.EditorConfig",
    "Prisma.prisma"
  ],
  "unwantedRecommendations": [
    // Biome 使用時は ESLint + Prettier を非推奨に
    // "biomejs.biome"
  ]
}
```

### 8.3 Biome の VS Code 設定

```jsonc
// .vscode/settings.json (Biome 版)
{
  "editor.defaultFormatter": "biomejs.biome",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "quickfix.biome": "explicit",
    "source.organizeImports.biome": "explicit"
  },
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  },
  "[markdown]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

### 8.4 EditorConfig

```ini
# .editorconfig (エディタ横断の基本設定)
root = true

[*]
charset = utf-8
end_of_line = lf
indent_style = space
indent_size = 2
insert_final_newline = true
trim_trailing_whitespace = true

[*.md]
trim_trailing_whitespace = false

[*.py]
indent_size = 4

[Makefile]
indent_style = tab

[*.{yml,yaml}]
indent_size = 2

[*.go]
indent_style = tab
indent_size = 4
```

---

## 9. CI 統合

### 9.1 GitHub Actions (ESLint + Prettier)

```yaml
# .github/workflows/lint.yml
name: Lint & Format
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      # ESLint (キャッシュ使用)
      - name: Lint
        run: pnpm eslint --cache .

      # Format チェック (修正なし -- 差分検出)
      - name: Format check
        run: pnpm prettier --check .

      # 型チェック
      - name: Typecheck
        run: pnpm tsc --noEmit

  lint-python:
    runs-on: ubuntu-latest
    if: ${{ hashFiles('**/*.py') != '' }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - run: pip install ruff mypy

      - name: Ruff lint
        run: ruff check .

      - name: Ruff format check
        run: ruff format --check .

      - name: Mypy
        run: mypy .
```

### 9.2 GitHub Actions (Biome)

```yaml
# .github/workflows/lint-biome.yml
name: Lint (Biome)
on: [push, pull_request]

jobs:
  biome:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: Biome CI
        run: pnpm biome ci .
```

### 9.3 PR レビューコメントの自動投稿

```yaml
# .github/workflows/lint-review.yml
name: Lint Review
on: pull_request

jobs:
  lint-review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4

      - uses: pnpm/action-setup@v4

      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      # ESLint の結果を PR コメントとして投稿
      - name: ESLint
        uses: reviewdog/action-eslint@v1
        with:
          reporter: github-pr-review
          eslint_flags: '.'
```

---

## 10. モノレポでの設定共有

### 10.1 共有設定パッケージの構成

```
packages/config/
├── package.json
├── eslint/
│   ├── base.js          # JavaScript/TypeScript 基本ルール
│   ├── react.js         # React 用ルール (base を extends)
│   ├── next.js          # Next.js 用ルール (react を extends)
│   └── node.js          # Node.js バックエンド用ルール
├── prettier/
│   └── index.json       # Prettier 共通設定
├── tsconfig/
│   ├── base.json        # TypeScript 基本設定
│   ├── react.json       # React 用 (JSX 有効化)
│   ├── nextjs.json      # Next.js 用
│   └── node.json        # Node.js バックエンド用
├── stylelint/
│   └── index.json       # Stylelint 共通設定
└── biome/
    └── biome.json       # Biome 共通設定 (代替構成)
```

```jsonc
// packages/config/package.json
{
  "name": "@repo/config",
  "version": "0.0.0",
  "private": true,
  "exports": {
    "./eslint/base": "./eslint/base.js",
    "./eslint/react": "./eslint/react.js",
    "./eslint/next": "./eslint/next.js",
    "./eslint/node": "./eslint/node.js",
    "./prettier": "./prettier/index.json",
    "./tsconfig/base": "./tsconfig/base.json",
    "./tsconfig/react": "./tsconfig/react.json",
    "./tsconfig/nextjs": "./tsconfig/nextjs.json",
    "./tsconfig/node": "./tsconfig/node.json",
    "./stylelint": "./stylelint/index.json"
  },
  "dependencies": {
    "@eslint/js": "^9.0.0",
    "typescript-eslint": "^8.0.0",
    "eslint-plugin-react": "^7.35.0",
    "eslint-plugin-react-hooks": "^4.6.0",
    "eslint-plugin-jsx-a11y": "^6.9.0",
    "@next/eslint-plugin-next": "^14.2.0",
    "eslint-config-prettier": "^9.1.0",
    "globals": "^15.0.0"
  }
}
```

---

## 11. アンチパターン

### 11.1 ESLint に Formatter の仕事をさせる

```
❌ アンチパターン: ESLint でインデントや引用符を矯正

  eslint.config.js:
    rules: {
      "indent": ["error", 2],           // ← Formatter の仕事
      "quotes": ["error", "single"],     // ← Formatter の仕事
      "semi": ["error", "always"],       // ← Formatter の仕事
      "max-len": ["error", 80],          // ← Formatter の仕事
      "comma-dangle": ["error", "always-multiline"], // ← Formatter の仕事
    }

問題:
  - Prettier との競合で無限修正ループ
  - ESLint の実行が遅くなる
  - 役割の重複でメンテナンスコスト増大

✅ 正しいアプローチ:
  - 見た目のルールは Prettier に任せる
  - eslint-config-prettier で競合ルールを OFF
  - ESLint はコード品質チェックに専念
  - ESLint の stylistic ルールは全て無効化
```

### 11.2 チームで設定を共有しない

```
❌ アンチパターン: 各開発者が独自の Linter 設定を使用

問題:
  - PR の差分がスタイル変更で埋もれる
  - "好みの違い" でコードレビューが紛糾
  - CI で別の設定が動いてエラー
  - 新メンバーのオンボーディングが困難

✅ 正しいアプローチ:
  - 設定ファイルをリポジトリにコミット
  - .vscode/settings.json で formatOnSave を強制
  - .vscode/extensions.json で推奨拡張を提示
  - pre-commit フックで強制フォーマット
  - CI で --check モードでゲート
  - EditorConfig でエディタ横断の基本設定
```

### 11.3 全ルールを有効化する

```
❌ アンチパターン: recommended + すべてのプラグインを有効

問題:
  - ルール同士が矛盾する場合がある
  - 過度に厳しい設定で開発速度が低下
  - 意味のない lint エラーへの対処に時間を浪費
  - // eslint-disable の乱用

✅ 正しいアプローチ:
  - recommended をベースに、プロジェクトに必要なルールのみ追加
  - warn と error を適切に使い分け
    - error: セキュリティ、型安全性 (絶対に許容しない)
    - warn: コード品質 (改善すべきだが緊急ではない)
  - 段階的に厳しくする (最初は recommended のみ)
```

### 11.4 CI でのみ lint を実行する

```
❌ アンチパターン: ローカルでは lint せず、CI で初めてエラーを発見

問題:
  - CI でエラー → 修正 → 再プッシュのサイクルが遅い
  - 開発者体験が悪い
  - CI のコンピュートリソースを浪費

✅ 正しいアプローチ:
  - エディタ連携: formatOnSave + codeActionsOnSave
  - pre-commit フック: lint-staged で差分のみチェック
  - CI: --check モードで最終ゲート (修正なし)
  - 3段階の防御で品質を担保
```

---

## 12. FAQ

### Q1: Biome は ESLint + Prettier を完全に置き換えられる？

**A:** 多くのプロジェクトでは可能だが、以下の場合は ESLint が必要。
- TypeScript の型情報に基づくルール（`no-unsafe-*` 系、`no-floating-promises`）を使いたい
- eslint-plugin-react-hooks 等の特定プラグインに依存
- カスタムルールを自作している
- eslint-plugin-import の高度なインポートルールが必要

新規プロジェクトで特殊な要件がなければ Biome がシンプルで高速。既存の ESLint 設定が複雑なプロジェクトは `biome migrate` で段階的に移行を検討する。ハイブリッド構成（Biome でフォーマット + ESLint で型チェック連携ルールのみ）も有効。

### Q2: formatOnSave が遅い場合の対処法は？

**A:**
1. `editor.formatOnSaveMode` を `"modificationsIfAvailable"` に設定（変更行のみフォーマット）
2. Biome に切り替える（Prettier の 20-100倍高速）
3. `.prettierignore` で不要なファイルを除外
4. ESLint の `codeActionsOnSave` と Prettier の `formatOnSave` が二重実行されていないか確認
5. ESLint のキャッシュを有効化 (`eslint --cache`)
6. `eslint.codeActionsOnSave.mode` を `"problems"` に設定
7. 大きなファイルでは `editor.formatOnSaveTimeout` を調整

### Q3: Ruff だけで Python の Linter + Formatter は十分？

**A:** はい。Ruff は Flake8、isort、Black、pyupgrade、flake8-bugbear、flake8-bandit 等の機能を1つのツールで提供する。速度は Flake8 の 10-100倍。2025年時点で Python の新規プロジェクトでは Ruff がデファクトスタンダードになっている。ただし mypy（型チェック）は別途必要。Ruff は構文ベースの解析のみで、型情報に基づくチェックは行わない。

### Q4: ESLint の Flat Config と Legacy Config は混在できる？

**A:** いいえ。ESLint v9 は Flat Config のみをサポートする。ただし `ESLINT_USE_FLAT_CONFIG=false` 環境変数で一時的にレガシーモードに戻すことは可能（v9.x の間のみ）。プラグインがまだ Flat Config に対応していない場合は `@eslint/compat` パッケージの `fixupPluginRules` を使って互換レイヤーを挟む。

### Q5: モノレポで各パッケージの ESLint 設定を変えたい場合は？

**A:** 共有設定パッケージ (`@repo/config`) に複数の設定プリセットを用意し、各パッケージの `eslint.config.js` で適切なものを import する。例えば `@repo/config/eslint/react` はフロントエンド用、`@repo/config/eslint/node` はバックエンド用。各パッケージの `eslint.config.js` はプリセットを extends した上で、パッケージ固有のルールを追加する。

---

## 13. まとめ

| エコシステム | Linter | Formatter | 推奨度 |
|------------|--------|-----------|--------|
| JS/TS (標準) | ESLint v9 | Prettier | 最も汎用的 |
| JS/TS (高速) | Biome | Biome | シンプル・新規向け |
| JS/TS (ハイブリッド) | ESLint (型ルール) + Biome (他) | Biome | バランス型 |
| Python | Ruff + mypy | Ruff | デファクト |
| CSS/SCSS | Stylelint | Prettier | CSS専用 |
| Pre-commit | husky + lint-staged / lefthook | - | 必須レベル |
| CI | `--check` モード | `--check` モード | ゲート必須 |
| エディタ | VS Code + 各拡張 | formatOnSave | 自動化必須 |
| モノレポ | packages/config/ で共有 | 同上 | 設定の一元管理 |

---

## 次に読むべきガイド

- [02-monorepo-setup.md](./02-monorepo-setup.md) -- モノレポでの設定共有
- [../00-editor-and-tools/00-vscode-setup.md](../00-editor-and-tools/00-vscode-setup.md) -- VS Code との連携設定
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) -- チーム標準ルールの策定

---

## 参考文献

1. **ESLint v9 Flat Config** -- https://eslint.org/docs/latest/use/configure/configuration-files -- ESLint v9 の新設定形式の公式ガイド。
2. **Biome Documentation** -- https://biomejs.dev/guides/getting-started/ -- Biome 公式ドキュメント。ESLint からの移行ガイドあり。
3. **Ruff Documentation** -- https://docs.astral.sh/ruff/ -- Ruff 公式。対応ルール一覧とベンチマーク。
4. **Prettier Options** -- https://prettier.io/docs/en/options -- Prettier の全オプション解説。
5. **typescript-eslint** -- https://typescript-eslint.io/ -- TypeScript ESLint の公式サイト。型チェック連携の詳細。
6. **Stylelint** -- https://stylelint.io/ -- CSS/SCSS Linter の公式ドキュメント。
7. **lint-staged** -- https://github.com/lint-staged/lint-staged -- ステージングファイル限定の lint 実行ツール。
8. **lefthook** -- https://github.com/evilmartians/lefthook -- husky + lint-staged の高速代替ツール。
