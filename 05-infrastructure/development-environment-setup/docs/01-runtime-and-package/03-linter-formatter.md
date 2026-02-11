# Linter / Formatter

> ESLint、Prettier、Biome、Ruff を活用したコード品質管理の実践ガイド。設定共有と CI 統合で、チーム全体のコードスタイルを統一する。

## この章で学ぶこと

1. ESLint (v9 Flat Config) と Prettier の正しい設定と連携
2. Biome (Rust 製高速ツール) と Ruff (Python) の導入方法
3. 設定共有パターンと CI / pre-commit フックの統合

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
  │               │     │ - 到達不能     │
  │               │     │   コード       │
  └──────────────┘     └──────────────┘
        │                      │
        ▼                      ▼
  自動修正可能            一部自動修正可能
  (100%)                (ルールによる)
```

### 1.2 主要ツール比較

| ツール | 対象言語 | 種類 | 速度 | 設定形式 |
|--------|---------|------|------|---------|
| ESLint | JS/TS | Linter | 普通 | Flat Config (JS) |
| Prettier | 多言語 | Formatter | 普通 | JSON/JS |
| Biome | JS/TS/JSON/CSS | 両方 | 超高速 | JSON |
| Ruff | Python | 両方 | 超高速 | TOML |
| Stylelint | CSS/SCSS | Linter | 普通 | JSON/JS |
| oxlint | JS/TS | Linter | 超高速 | JSON |

---

## 2. ESLint (v9 Flat Config)

### 2.1 セットアップ

```bash
# インストール
pnpm add -D eslint @eslint/js typescript-eslint globals

# 型チェック統合が必要な場合
pnpm add -D @typescript-eslint/parser
```

### 2.2 設定ファイル

```javascript
// eslint.config.js (Flat Config 形式 — v9 推奨)
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import globals from "globals";

export default tseslint.config(
  // グローバル無視
  {
    ignores: [
      "dist/",
      "node_modules/",
      "coverage/",
      ".next/",
      "*.config.js",
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

      // ─── コード品質 ───
      "@typescript-eslint/no-unused-vars": [
        "error",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "prefer-const": "error",
      "no-var": "error",
      eqeqeq: ["error", "always"],
    },
  },

  // テストファイル用の緩和ルール
  {
    files: ["**/*.test.ts", "**/*.spec.ts"],
    rules: {
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-explicit-any": "off",
    },
  }
);
```

### 2.3 実行コマンド

```bash
# リント実行
pnpm eslint .

# 自動修正
pnpm eslint --fix .

# 特定ファイル
pnpm eslint src/utils/validate.ts

# package.json に scripts を追加
{
  "scripts": {
    "lint": "eslint .",
    "lint:fix": "eslint --fix ."
  }
}
```

---

## 3. Prettier

### 3.1 セットアップ

```bash
# インストール
pnpm add -D prettier

# ESLint との競合回避
pnpm add -D eslint-config-prettier
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
  "plugins": ["prettier-plugin-tailwindcss"],
  "overrides": [
    {
      "files": "*.md",
      "options": {
        "printWidth": 100,
        "proseWrap": "always"
      }
    }
  ]
}
```

```bash
# .prettierignore
dist/
node_modules/
coverage/
.next/
pnpm-lock.yaml
*.min.js
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
```

---

## 4. Biome (高速オールインワン)

### 4.1 セットアップ

```bash
# インストール
pnpm add -D @biomejs/biome

# 初期設定
pnpm biome init
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
        }
      },
      "correctness": {
        "noUnusedVariables": "error",
        "noUnusedImports": "error",
        "useExhaustiveDependencies": "warn"
      },
      "style": {
        "noNonNullAssertion": "warn",
        "useConst": "error"
      },
      "suspicious": {
        "noExplicitAny": "error"
      }
    }
  },
  "javascript": {
    "formatter": {
      "quoteStyle": "single",
      "semicolons": "always",
      "trailingCommas": "all",
      "arrowParentheses": "always"
    }
  },
  "files": {
    "ignore": [
      "dist/",
      "node_modules/",
      ".next/",
      "coverage/"
    ]
  }
}
```

### 4.3 ESLint + Prettier vs Biome 比較

| 観点 | ESLint + Prettier | Biome |
|------|-------------------|-------|
| 速度 | 1x (基準) | 20-100x |
| 設定ファイル数 | 2-3個 | 1個 |
| プラグイン | 豊富 (1000+) | 限定的 |
| TypeScript対応 | 型チェック連携可 | 構文ベースのみ |
| CSS 対応 | Stylelint 別途 | 組み込み |
| エコシステム成熟度 | 非常に高い | 成長中 |
| 移行コスト | - | `biome migrate` |
| 推奨 | 大規模・カスタム | 高速・シンプル |

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
```

### 5.2 設定

```toml
# pyproject.toml
[tool.ruff]
target-version = "py312"
line-length = 88
indent-width = 4

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # Pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "C4",   # flake8-comprehensions
    "DTZ",  # flake8-datetimez
    "T20",  # flake8-print
    "RUF",  # Ruff固有ルール
]
ignore = [
    "E501",  # line too long (formatter に任せる)
]

[tool.ruff.lint.per-file-ignores]
"tests/**/*.py" = ["T20"]  # テストでは print 許可

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
docstring-code-format = true
```

### 5.3 実行コマンド

```bash
# リント
ruff check .
ruff check --fix .

# フォーマット
ruff format .

# Ruff は Flake8 + isort + Black を1つで置き換える
# 速度は Flake8 の 10-100倍
```

---

## 6. Pre-commit フック

### 6.1 lint-staged + husky

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
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ],
    "*.py": [
      "ruff check --fix",
      "ruff format"
    ]
  }
}
```

```bash
# .husky/pre-commit
pnpm lint-staged
```

### 6.2 pre-commit フローの動作

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
  │       └── *.py        │
  │           → ruff      │
  │                       │
  │  全てパス？            │
  │  ├── Yes → コミット    │
  │  └── No  → コミット   │
  │           中止 + エラー │
  └──────────────────────┘
```

---

## 7. CI 統合

### 7.1 GitHub Actions

```yaml
# .github/workflows/lint.yml
name: Lint & Format
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
      - uses: actions/setup-node@v4
        with:
          node-version-file: '.node-version'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      # Lint (修正なし — エラーのみ報告)
      - run: pnpm eslint .

      # Format チェック (修正なし — 差分検出)
      - run: pnpm prettier --check .

      # 型チェック
      - run: pnpm tsc --noEmit
```

---

## 8. アンチパターン

### 8.1 ESLint に Formatter の仕事をさせる

```
❌ アンチパターン: ESLint でインデントや引用符を矯正

  eslint.config.js:
    rules: {
      "indent": ["error", 2],           // ← Formatter の仕事
      "quotes": ["error", "single"],     // ← Formatter の仕事
      "semi": ["error", "always"],       // ← Formatter の仕事
    }

問題:
  - Prettier との競合で無限修正ループ
  - ESLint の実行が遅くなる
  - 役割の重複

✅ 正しいアプローチ:
  - 見た目のルールは Prettier に任せる
  - eslint-config-prettier で競合ルールを OFF
  - ESLint はコード品質チェックに専念
```

### 8.2 チームで設定を共有しない

```
❌ アンチパターン: 各開発者が独自の Linter 設定を使用

問題:
  - PR の差分がスタイル変更で埋もれる
  - "好みの違い" でコードレビューが紛糾
  - CI で別の設定が動いてエラー

✅ 正しいアプローチ:
  - 設定ファイルをリポジトリにコミット
  - .vscode/settings.json で formatOnSave を強制
  - pre-commit フックで強制フォーマット
  - CI で --check モードでゲート
```

---

## 9. FAQ

### Q1: Biome は ESLint + Prettier を完全に置き換えられる？

**A:** 多くのプロジェクトでは可能だが、以下の場合は ESLint が必要。
- TypeScript の型情報に基づくルール（`no-unsafe-*` 系）を使いたい
- eslint-plugin-react-hooks 等の特定プラグインに依存
- カスタムルールを自作している

新規プロジェクトで特殊な要件がなければ Biome がシンプルで高速。既存の ESLint 設定が複雑なプロジェクトは段階的に移行を検討する。

### Q2: formatOnSave が遅い場合の対処法は？

**A:**
1. `editor.formatOnSaveMode` を `"modificationsIfAvailable"` に設定（変更行のみフォーマット）
2. Biome に切り替える（Prettier の 20-100倍高速）
3. `.prettierignore` で不要なファイルを除外
4. ESLint の `codeActionsOnSave` と Prettier の `formatOnSave` が二重実行されていないか確認

### Q3: Ruff だけで Python の Linter + Formatter は十分？

**A:** はい。Ruff は Flake8、isort、Black、pyupgrade 等の機能を1つのツールで提供する。速度は Flake8 の 10-100倍。2025年時点で Python の新規プロジェクトでは Ruff がデファクトスタンダードになりつつある。mypy（型チェック）は別途必要。

---

## 10. まとめ

| エコシステム | Linter | Formatter | 推奨度 |
|------------|--------|-----------|--------|
| JS/TS (標準) | ESLint v9 | Prettier | 最も汎用的 |
| JS/TS (高速) | Biome | Biome | シンプル・新規向け |
| Python | Ruff | Ruff | デファクト |
| CSS | Stylelint | Prettier | CSS専用 |
| Pre-commit | husky + lint-staged | - | 必須レベル |
| CI | `--check` モード | `--check` モード | ゲート必須 |

---

## 次に読むべきガイド

- [02-monorepo-setup.md](./02-monorepo-setup.md) — モノレポでの設定共有
- [../00-editor-and-tools/00-vscode-setup.md](../00-editor-and-tools/00-vscode-setup.md) — VS Code との連携設定
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) — チーム標準ルールの策定

---

## 参考文献

1. **ESLint v9 Flat Config** — https://eslint.org/docs/latest/use/configure/configuration-files — ESLint v9 の新設定形式の公式ガイド。
2. **Biome Documentation** — https://biomejs.dev/guides/getting-started/ — Biome 公式ドキュメント。ESLint からの移行ガイドあり。
3. **Ruff Documentation** — https://docs.astral.sh/ruff/ — Ruff 公式。対応ルール一覧とベンチマーク。
4. **Prettier Options** — https://prettier.io/docs/en/options — Prettier の全オプション解説。
